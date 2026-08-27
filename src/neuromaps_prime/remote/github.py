"""Model for fetching from GitHub repos (tag/branch/commit)."""

import hashlib
import logging
import re
from pathlib import Path

import requests
from pydantic import BaseModel

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
    """Fetch file from public GitHub repo.

    Handles both blob and raw urls.
    """

    chunk_size: int = 8192

    @staticmethod
    def _parse(url: str) -> tuple[str, str, str, str]:
        m = _RAW_RE.search(url)
        if not m:
            raise ValueError(f"Unrecognized / unsupported GitHub URL: {url}")
        return m["owner"], m["repo"], m["ref"], m["path"]

    def get_meta(self, url: str) -> GitHubFileMeta:
        """Get metadata from remote GitHub file (given blob URL)."""
        owner, repo, ref, path = self._parse(url)
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        r = requests.get(api_url, params={"ref": ref}, timeout=90)
        r.raise_for_status()
        return GitHubFileMeta(**r.json())

    def download(self, url: str, dest_dir: Path) -> Path:
        """Download the file into ``dest_dir`` and return its local path.

        The file is stored under its storage-side name. An existing file is
        reused only if it verifies against the blob SHA; otherwise a warning
        is logged and the file is overwritten.
        """
        meta = self.get_meta(url)
        target = dest_dir / meta.name
        if target.exists():
            if self._verify(target, meta):
                return target
            _logger.warning("%s already exists and is being overwritten.", target)
        self._download_to(meta.download_url, target, meta)
        return target

    def _download_to(self, url: str, target: Path, meta: GitHubFileMeta) -> None:
        """Stream the file to ``target`` and verify its blob SHA."""
        r = requests.get(url, stream=True, timeout=90)
        r.raise_for_status()

        content = bytearray()
        with target.open("wb") as f:
            for chunk in r.iter_content(self.chunk_size):
                f.write(chunk)
                content.extend(chunk)

        header = f"blob {len(content)}\0".encode()
        actual = hashlib.sha1(
            header + bytes(content), usedforsecurity=False
        ).hexdigest()
        if actual != meta.sha:
            raise ValueError(f"Checksum mismatch: {actual} != {meta.sha}")

    @staticmethod
    def _verify(path: Path, meta: GitHubFileMeta) -> bool:
        """Check that ``path`` matches the storage-side size and blob SHA."""
        if path.stat().st_size != meta.size:
            return False
        content = path.read_bytes()
        header = f"blob {len(content)}\0".encode()
        actual = hashlib.sha1(header + content, usedforsecurity=False).hexdigest()
        return actual == meta.sha
