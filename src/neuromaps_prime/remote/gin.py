"""Model for fetching from GIN (gin.g-node.org) repositories."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel


class GINFileMeta(BaseModel):
    """File metadata for a GIN raw file."""

    name: str


_FILENAME_RE = re.compile(r'filename=["\']?([^"\';]+)["\']?', re.IGNORECASE)


def _filename_from_disposition(value: str | None) -> str | None:
    """Extract a filename from a ``Content-Disposition`` header value.

    Args:
        value: Raw ``Content-Disposition`` header value (may be ``None``).

    Returns:
        The parsed filename, or ``None`` if no ``filename`` is present.
    """
    match = _FILENAME_RE.search(value) if value else None
    return match.group(1) if match else None


class GINStorage(BaseModel):
    """Fetch a file from a public GIN repository via its raw URL."""

    chunk_size: int = 8192

    def get_meta(self, url: str) -> GINFileMeta:
        """Get the storage-side filename for a remote GIN raw file.

        Sends a HEAD request so that a bad ref or path fails fast (before any
        large download), and reads the authoritative filename from the
        ``Content-Disposition`` header, falling back to the URL's final path
        segment.
        """
        r = requests.head(url, timeout=90)
        r.raise_for_status()
        name = _filename_from_disposition(r.headers.get("Content-Disposition"))
        if not name:
            name = urlsplit(url).path.rsplit("/", 1)[-1] or "download"
        return GINFileMeta(name=name)

    def download(self, url: str, dest_dir: Path) -> Path:
        """Download the file into ``dest_dir`` and return its local path.

        The file is stored under its storage-side name. An existing file is
        reused as-is. Because GIN provides no checksum to re-validate against,
        the file is streamed to a temporary sibling and atomically renamed into
        place, so an interrupted download never leaves a partial target file.
        """
        meta = self.get_meta(url)
        target = dest_dir / meta.name
        if target.exists():
            return target
        tmp = target.with_name(f"{target.name}.part")
        try:
            self._download_to(url, tmp)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
        return target

    def _download_to(self, url: str, target: Path) -> None:
        """Stream the raw file from *url* to *target* in chunks."""
        r = requests.get(url, stream=True, timeout=90)
        r.raise_for_status()
        with target.open("wb") as f:
            for chunk in r.iter_content(self.chunk_size):
                f.write(chunk)
