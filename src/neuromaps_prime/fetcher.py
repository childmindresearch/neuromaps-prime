"""Helpers for grabbing from remote repositories."""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse, urlsplit, urlunsplit

import requests

from neuromaps_prime import remote

_STORAGES = {
    "osf": remote.OSFStorage(),
    "github": remote.GitHubStorage(),
    "gin": remote.GINStorage(),
}
_HOST_MAP = {
    "osf.io": _STORAGES["osf"],
    "raw.githubusercontent.com": _STORAGES["github"],
    "gin.g-node.org": _STORAGES["gin"],
}

# Transient HTTP statuses worth retrying
_RETRYABLE_STATUS = {403, 429, 500, 502, 503, 504}
# Connection-level failures worth retrying (drops, timeouts, interrupted streams).
_RETRYABLE_EXC = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
_MAX_ATTEMPTS = 5
_MAX_DELAY_S = 15.0

# Download index stored inside the cache dir (normalized URI -> filename).
_MANIFEST_NAME = ".neuromaps_manifest.json"

T = TypeVar("T")

_logger = logging.getLogger(__name__)


def _backoff(response: requests.Response | None, attempt: int) -> float:
    """Seconds to wait before the next attempt.

    Honors a sane ``Retry-After`` header when present, falling back to capped
    exponential backoff otherwise.
    """
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw is not None:
            try:
                return min(max(float(raw), 0.0), _MAX_DELAY_S)
            except ValueError:
                pass  # non-numeric (e.g. an HTTP-date); use backoff instead
    return min(2.0 * 2**attempt, _MAX_DELAY_S)


def _is_retryable(exc: BaseException) -> bool:
    """Return whether ``exc`` is a transient failure worth retrying.

    Args:
        exc: Exception raised by a download attempt.

    Returns:
        True if the failure is transient, False otherwise.
    """
    if isinstance(exc, requests.HTTPError):
        return (
            exc.response is not None and exc.response.status_code in _RETRYABLE_STATUS
        )
    return isinstance(exc, _RETRYABLE_EXC)


def _with_retries(fn: Callable[[], T]) -> T:  # noqa: UP047
    """Call ``fn``, retrying transient network errors with backoff.

    A transient error is an HTTP status in ``_RETRYABLE_STATUS`` (rate
    limiting, brief server errors) or a connection-level failure (a dropped
    connection, a timeout, an interrupted stream). A non-retryable error
    propagates immediately.

    Args:
        fn: Zero-argument callable performing the (possibly network-bound) work.

    Returns:
        The value ``fn`` returns on the first attempt that is not retried.

    Raises:
        requests.RequestException: if the final attempt fails with a transient
            error, or any attempt fails with a non-retryable one.
    """
    for attempt in range(_MAX_ATTEMPTS):
        response = None
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == _MAX_ATTEMPTS - 1:
                raise
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                response = exc.response
                reason = f"HTTP {exc.response.status_code}"
            else:
                reason = type(exc).__name__
        delay = _backoff(response, attempt)
        _logger.warning(
            "%s; retrying in %.1fs (attempt %d/%d)",
            reason,
            delay,
            attempt + 1,
            _MAX_ATTEMPTS,
        )
        time.sleep(delay)
    raise RuntimeError("_with_retries exhausted its attempts without a result")


def id_storage(uri: str) -> str | None:
    """Identify the storage type.

    Args:
        uri: Remote URI to fetch data from

    Returns:
        String indicating type of storage (one of 'osf', 'github', 'gin')
    """
    host = urlparse(uri).hostname
    if host is None:
        return None
    host = host.lower()
    for k, v in _HOST_MAP.items():
        if host == k or host.endswith(k):
            return next(name for name, s in _STORAGES.items() if s is v)
    return None


def _normalize_uri(uri: str) -> str:
    """Return the manifest key for ``uri``.

    The query string and fragment are dropped so a rotating ``?token=``
    never changes the key.

    Args:
        uri: Remote URI to normalize.

    Returns:
        The normalized URI.
    """
    parts = urlsplit(uri)
    return urlunsplit(parts._replace(query="", fragment=""))


def _read_manifest(dest_dir: Path) -> dict[str, str]:
    """Read the download manifest from ``dest_dir``.

    A missing, corrupt, or malformed manifest is treated as empty.

    Args:
        dest_dir: Directory holding the download manifest.

    Returns:
        Mapping of normalized URI to stored filename.
    """
    path = dest_dir / _MANIFEST_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _logger.debug("Ignoring unreadable manifest at %s: %s.", path, exc)
        return {}
    if not isinstance(raw, dict):
        _logger.debug("Ignoring malformed manifest at %s.", path)
        return {}
    return {
        key: value
        for key, value in raw.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _record_download(dest_dir: Path, uri: str, filename: str) -> None:
    """Record that ``uri`` was downloaded to ``dest_dir / filename``.

    Best-effort: write failures are logged, never raised. The manifest is
    written to a sibling ``.part`` file and moved into place atomically.

    Args:
        dest_dir: Directory the file was downloaded into.
        uri: Remote URI that was fetched.
        filename: Storage-side name of the downloaded file.
    """
    try:
        entries = _read_manifest(dest_dir)
        entries[_normalize_uri(uri)] = filename
        path = dest_dir / _MANIFEST_NAME
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # best-effort: never fail a good download
        _logger.warning("Could not update download manifest: %s", exc)


def _cached_copy(dest_dir: Path, uri: str) -> Path | None:
    """Return a previously downloaded copy of ``uri`` in ``dest_dir``, if any.

    The manifest is consulted first; on a miss the URI's final path segment is
    used. Only a regular file directly inside ``dest_dir`` is returned.

    Args:
        dest_dir: Directory the file would be downloaded into.
        uri: Remote URI whose fetch failed.

    Returns:
        Path to the cached file, or None if no usable copy is present.
    """
    name = _read_manifest(dest_dir).get(_normalize_uri(uri))
    if name is None:
        name = urlsplit(uri).path.rsplit("/", 1)[-1] or None
    if name is None:
        return None
    candidate = dest_dir / name
    if not candidate.is_file() or candidate.parent != dest_dir:
        return None
    return candidate


def download_and_validate(uri: str, dest_dir: str | Path) -> Path:
    """Download and validate the file.

    The file is stored in ``dest_dir`` under its storage-side name. Transient
    network failures are retried with backoff; if retries are exhausted and a
    cached copy exists in ``dest_dir``, it is returned with a warning.
    Non-retryable failures raise.

    Args:
        uri: Remote URI to fetch data from
        dest_dir: Directory to download the file into

    Returns:
        Path to the downloaded (or cached) file.

    Raises:
        ValueError: if storage cannot be identified from provided URI
        requests.RequestException: if the download keeps failing with a
            transient error and no cached copy is available
    """
    name = id_storage(uri)
    if name is None:
        clean_uri = str(urlparse(uri)._replace(query=""))
        raise ValueError(f"Could not identify storage from uri: {clean_uri}")
    storage = _STORAGES[name]
    dest = Path(dest_dir)
    try:
        path = _with_retries(lambda: storage.download(uri, dest))  # type: ignore[attr-defined]
    except Exception as exc:
        if not _is_retryable(exc):
            raise
        cached = _cached_copy(dest, uri)
        if cached is not None:
            _logger.warning(
                "Remote fetch of %s resource failed (%s); serving cached copy"
                " at %s; it may be stale.",
                name,
                type(exc).__name__,
                cached,
            )
            return cached
        raise
    _record_download(dest, uri, path.name)
    return path
