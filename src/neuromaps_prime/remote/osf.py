"""Model associated with fetching from OSF storages."""

import hashlib
from abc import ABC
from pathlib import Path

import requests
from pydantic import BaseModel


class OSFHashes(BaseModel):
    """Model for OSF hashes."""

    md5: str
    sha256: str | None = None


class OSFFileExtra(BaseModel):
    """Model for alternative storage of OSF hashes."""

    hashes: OSFHashes


class OSFFileMeta(BaseModel):
    """Model for file metadata."""

    name: str
    size: int
    extra: OSFFileExtra


class OSFStorage(BaseModel, ABC):
    """Model for OSF storage.

    Note: This currently only works for storages that are publicly accessible.
    """

    chunk_size: int = 8192

    def get_meta(self, url: str) -> OSFFileMeta:
        """Get metadata from remote OSF file."""
        r = requests.get(url, params={"meta": ""}, timeout=90)
        r.raise_for_status()
        return OSFFileMeta(**r.json()["data"]["attributes"])

    def download(self, url: str, dest: Path) -> None:
        """Download remote OSF file."""
        meta = self.get_meta(url)
        r = requests.get(url, stream=True, timeout=90)
        r.raise_for_status()

        h = hashlib.md5(usedforsecurity=False)
        with dest.open("wb") as f:
            for chunk in r.iter_content(self.chunk_size):
                f.write(chunk)
                h.update(chunk)

        if (actual := h.hexdigest()) != meta.extra.hashes.md5:
            raise ValueError(f"Checksum mismatch: {actual} != {meta.extra.hashes.md5}")
