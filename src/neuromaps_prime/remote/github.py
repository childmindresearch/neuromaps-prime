"""Model for fetching from GitHub repos (tag/branch/commit)."""

import hashlib
import re
from pathlib import Path

import requests
from pydantic import BaseModel


class GitHubFileMeta(BaseModel):
    """File metadata from GitHub contents API."""

    name: str
    size: int
    sha: str
    download_url: str


_BLOB_RE = re.compile(
    r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<ref>[^/]+)/(?P<path>.+)"
)


class GitHubStorage(BaseModel):
    """Fetch file from public GitHub repo.

    Blob URL provided would be similar to:

    https://github.com/{owner}/{repo}/blob/{ref}/{path}
    """

    chunk_size: int = 8192

    @staticmethod
    def _parse(url: str) -> tuple[str, str, str, str]:
        m = _BLOB_RE.search(url)
        if not m:
            raise ValueError(f"Unrecognized GitHub blob URL: {url}")
        return m["owner"], m["repo"], m["ref"], m["path"]

    def get_meta(self, url: str) -> GitHubFileMeta:
        """Get metadata from remote GitHub file (given blob URL)."""
        owner, repo, ref, path = self._parse(url)
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        r = requests.get(api_url, params={"ref": ref}, timeout=90)
        r.raise_for_status()
        return GitHubFileMeta(**r.json())

    def download(self, url: str, dest: Path) -> None:
        """Download remote GitHub file (given blob URL)."""
        meta = self.get_meta(url)
        r = requests.get(meta.download_url, stream=True, timeout=90)
        r.raise_for_status()

        content = bytearray()
        with dest.open("wb") as f:
            for chunk in r.iter_content(self.chunk_size):
                f.write(chunk)
                content.extend(chunk)

        header = f"blob {len(content)}\0".encode()
        actual = hashlib.sha1(
            header + bytes(content), usedforsecurity=False
        ).hexdigest()
        if actual != meta.sha:
            raise ValueError(f"Checksum mismatch: {actual} != {meta.sha}")
