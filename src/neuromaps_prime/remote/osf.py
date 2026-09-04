"""Model associated with fetching from OSF storages."""

import hashlib
import logging
from abc import ABC
from pathlib import Path

import requests
from pydantic import BaseModel

_logger = logging.getLogger(__name__)


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

    def download(self, url: str, dest_dir: Path) -> Path:
        """Download the file into ``dest_dir`` and return its local path.

        The file is stored under its storage-side name. An existing file is
        reused only if it verifies against the storage MD5; otherwise a
        warning is logged and the file is overwritten.
        """
        meta = self.get_meta(url)
        target = dest_dir / meta.name
        if target.exists():
            if self._verify(target, meta):
                return target
            _logger.warning("%s already exists and is being overwritten.", target)
        self._download_to(url, target, meta)
        return target

    def _download_to(self, url: str, target: Path, meta: OSFFileMeta) -> None:
        """Stream the file to a temp file and move it to ``target`` if valid.

        The download lands in a sibling ``.part`` file and is only renamed
        into place with :meth:`Path.replace` once the MD5 verifies, so a
        failed download never leaves a partial ``target`` behind.
        """
        part = target.with_name(target.name + ".part")
        r = requests.get(url, stream=True, timeout=90)
        try:
            r.raise_for_status()

            h = hashlib.md5(usedforsecurity=False)
            with part.open("wb") as f:
                for chunk in r.iter_content(self.chunk_size):
                    f.write(chunk)
                    h.update(chunk)

            if (actual := h.hexdigest()) != meta.extra.hashes.md5:
                raise ValueError(
                    f"Checksum mismatch: {actual} != {meta.extra.hashes.md5}"
                )

            part.replace(target)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        finally:
            r.close()

    @staticmethod
    def _verify(path: Path, meta: OSFFileMeta) -> bool:
        """Check that ``path`` matches the storage-side size and MD5."""
        if path.stat().st_size != meta.size:
            return False
        h = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest() == meta.extra.hashes.md5
