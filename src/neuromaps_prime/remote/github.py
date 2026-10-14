"""Download GitHub raw files directly, with HTTP cache validation."""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from requests import Response

_logger = logging.getLogger(__name__)
_RAW_RE = re.compile(
    r"https://raw\.githubusercontent\.com/(?P<owner>[^/?#]+)/"
    r"(?P<repo>[^/?#]+)/(?:refs/(?:tags|heads)/)?"
    r"(?P<ref>[^/?#]+)/(?P<path>[^?#]+)"
)


class GitHubCacheMeta(BaseModel):
    """Local cache identity and HTTP validators; no API metadata is needed."""

    url: str
    size: int
    sha256: str
    etag: str | None = None
    last_modified: str | None = None


class GitHubStorage(BaseModel):
    """Fetch public raw URLs without querying the GitHub REST API.

    Cached bytes are checked locally before sending an HTTP conditional GET.
    A 304 reuses the cache; a 200 atomically replaces it. Branch URLs therefore
    follow updates without requiring tags, tokens, or Contents API requests.
    """

    chunk_size: int = 8192

    @staticmethod
    def _parse(url: str) -> tuple[str, str, str, str]:
        match = _RAW_RE.fullmatch(url)
        if match is None:
            raise ValueError(f"Unrecognized / unsupported GitHub URL: {url}")
        return match["owner"], match["repo"], match["ref"], match["path"]

    @staticmethod
    def _read_cache(
        metadata_path: Path, target: Path, url: str
    ) -> GitHubCacheMeta | None:
        if not target.is_file() or not metadata_path.is_file():
            return None
        try:
            meta = GitHubCacheMeta.model_validate_json(metadata_path.read_bytes())
            if meta.url != url or meta.size != target.stat().st_size:
                return None
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            return meta if digest == meta.sha256 else None
        except (OSError, ValidationError):
            return None

    @staticmethod
    def _write_cache(metadata_path: Path, meta: GitHubCacheMeta) -> None:
        with tempfile.NamedTemporaryFile(
            dir=metadata_path.parent, suffix=".part", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            temporary.write_text(meta.model_dump_json(), encoding="utf-8")
            temporary.replace(metadata_path)
        finally:
            temporary.unlink(missing_ok=True)

    def download(self, url: str, dest_dir: Path) -> Path:
        """Fetch a raw file, revalidating any existing cache over HTTP.

        Args:
            url: Public raw.githubusercontent.com URL.
            dest_dir: Cache directory; files retain their URL basename.

        Returns:
            Local path to the complete file.

        Raises:
            ValueError: If the URL is unsupported or the response is incomplete.
            requests.RequestException: If the raw request or stream fails.
        """
        _, _, _, file_path = self._parse(url)
        name = Path(file_path).name
        if name in ("", ".", ".."):
            raise ValueError(f"Unsupported GitHub filename: {name!r}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / name
        metadata_path = target.with_name(target.name + ".http.json")
        cached = self._read_cache(metadata_path, target, url)
        headers = {"Accept-Encoding": "identity"}
        if cached is not None:
            if cached.etag:
                headers["If-None-Match"] = cached.etag
            elif cached.last_modified:
                headers["If-Modified-Since"] = cached.last_modified

        response = requests.get(url, headers=headers, stream=True, timeout=90)
        try:
            if response.status_code == 304:
                if cached is None:
                    raise ValueError("Received HTTP 304 without a valid cached file.")
                return target
            response.raise_for_status()
            meta = self._download_to(response, target, url)
            self._write_cache(metadata_path, meta)
        finally:
            response.close()
        return target

    def _download_to(
        self, response: Response, target: Path, url: str
    ) -> GitHubCacheMeta:
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(
            dir=target.parent, suffix=".part", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                for chunk in response.iter_content(self.chunk_size):
                    if chunk:
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            except BaseException:
                stream.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            length = response.headers.get("Content-Length")
            if (
                length is not None
                and not response.headers.get("Content-Encoding")
                and size != int(length)
            ):
                raise ValueError(
                    f"Incomplete GitHub download: {size} != {length} bytes"
                )
            meta = GitHubCacheMeta(
                url=url,
                size=size,
                sha256=digest.hexdigest(),
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
            )
            if target.exists():
                _logger.info("Updating cached GitHub file %s.", target)
            temporary.replace(target)
            return meta
        finally:
            temporary.unlink(missing_ok=True)
