"""Model for fetching from GitHub repos (tag/branch/commit)."""

import hashlib
import logging
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

import requests
from pydantic import BaseModel, SecretStr

_logger = logging.getLogger(__name__)


class GitHubFileMeta(BaseModel):
    """File metadata from GitHub contents API."""

    name: str
    size: int
    sha: str
    download_url: str


_RAW_RE = re.compile(
    r"raw\.githubusercontent\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/"
    r"(?:refs/(?:tags|heads)/)?(?P<ref>[^/]+)/(?P<path>.+)"
)


class GitHubStorage(BaseModel):
    """Fetch file from a GitHub repo.

    Handles both blob and raw urls. Private repos are supported via a
    ``?token=`` query parameter; the token is only ever sent as a Bearer
    header, never in a request URL.
    """

    chunk_size: int = 8192

    @staticmethod
    def _parse(url: str) -> tuple[str, str, str, str]:
        parts = urlsplit(url)
        stripped = urlunsplit(parts._replace(query=""))
        m = _RAW_RE.search(stripped)
        if not m:
            raise ValueError(f"Unrecognized / unsupported GitHub URL: {stripped}")
        return m["owner"], m["repo"], m["ref"], m["path"]

    @staticmethod
    def _extract_token(url: str) -> SecretStr | None:
        """Return the ``?token=`` value (private repo auth), if present."""
        value = parse_qs(urlsplit(url).query).get("token", [None])[0]
        return SecretStr(value) if value is not None else None

    @staticmethod
    def _auth_header(token: SecretStr | None) -> dict[str, str] | None:
        """Bearer ``Authorization`` header for ``token``, or ``None``."""
        if token is None:
            return None
        return {"Authorization": f"Bearer {token.get_secret_value()}"}

    def get_meta(self, url: str) -> GitHubFileMeta:
        """Get metadata from remote GitHub file (given blob URL)."""
        owner, repo, ref, path = self._parse(url)
        token = self._extract_token(url)
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        r = requests.get(
            api_url, params={"ref": ref}, headers=self._auth_header(token), timeout=90
        )
        r.raise_for_status()
        return GitHubFileMeta(**r.json())

    def download(self, url: str, dest_dir: Path) -> Path:
        """Download the file into ``dest_dir`` and return its local path.

        The file is stored under its storage-side name. An existing file is
        reused only if it verifies against the blob SHA; otherwise a warning
        is logged and the file is overwritten.
        """
        meta = self.get_meta(url)
        token = self._extract_token(url)
        target = dest_dir / meta.name
        if target.exists():
            if self._verify(target, meta):
                return target
            _logger.warning("%s already exists and is being overwritten.", target)
        self._download_to(meta.download_url, target, meta, token)
        return target

    def _download_to(
        self, url: str, target: Path, meta: GitHubFileMeta, token: SecretStr | None
    ) -> None:
        """Stream the file to a temp file and move it to ``target`` if valid.

        The download lands in a sibling ``.part`` file and is only renamed
        into place with :meth:`Path.replace` once the blob SHA verifies, so a
        failed download never leaves a partial ``target`` behind.
        """
        part = target.with_name(target.name + ".part")
        r = requests.get(url, stream=True, headers=self._auth_header(token), timeout=90)
        try:
            r.raise_for_status()

            content = bytearray()
            with part.open("wb") as f:
                for chunk in r.iter_content(self.chunk_size):
                    f.write(chunk)
                    content.extend(chunk)

            header = f"blob {len(content)}\0".encode()
            actual = hashlib.sha1(
                header + bytes(content), usedforsecurity=False
            ).hexdigest()
            if actual != meta.sha:
                raise ValueError(f"Checksum mismatch: {actual} != {meta.sha}")

            part.replace(target)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        finally:
            r.close()

    @staticmethod
    def _verify(path: Path, meta: GitHubFileMeta) -> bool:
        """Check that ``path`` matches the storage-side size and blob SHA."""
        if path.stat().st_size != meta.size:
            return False
        content = path.read_bytes()
        header = f"blob {len(content)}\0".encode()
        actual = hashlib.sha1(header + content, usedforsecurity=False).hexdigest()
        return actual == meta.sha
