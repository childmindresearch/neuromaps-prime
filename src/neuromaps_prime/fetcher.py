"""Helpers for grabbing from remote repositories."""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse

import requests

from neuromaps_prime import remote

_logger = logging.getLogger(__name__)

_STORAGES: dict[str, remote.OSFStorage | remote.GitHubStorage] = {
    "osf": remote.OSFStorage(),
    "github": remote.GitHubStorage(),
}
_HOST_MAP = {
    "osf.io": _STORAGES["osf"],
    "raw.githubusercontent.com": _STORAGES["github"],
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

T = TypeVar("T")


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
        except requests.HTTPError as exc:
            response = exc.response
            status = response.status_code if response is not None else None
            if status not in _RETRYABLE_STATUS or attempt == _MAX_ATTEMPTS - 1:
                raise
            reason = f"HTTP {status}"
        except _RETRYABLE_EXC as exc:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
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
        String indicating type of storage (one of 'osf', 'github')
    """
    host = urlparse(uri).hostname
    if host is None:
        return None
    host = host.lower()
    for k, v in _HOST_MAP.items():
        if host == k or host.endswith(k):
            return next(name for name, s in _STORAGES.items() if s is v)
    return None


def download_and_validate(uri: str, dest_dir: str | Path) -> Path:
    """Download and validate the file.

    The file is stored in ``dest_dir`` under its storage-side name. Transient
    network failures (rate-limiting, brief server errors, dropped connections,
    timeouts) are retried with backoff; a persistent failure raises.

    Args:
        uri: Remote URI to fetch data from
        dest_dir: Directory to download the file into

    Returns:
        Path to the downloaded (or cached) file.

    Raises:
        ValueError: if storage cannot be identified from provided URI
        requests.RequestException: if the download keeps failing with a
            transient error
    """
    name = id_storage(uri)
    if name is None:
        raise ValueError(f"Could not identify storage from uri: {uri}")
    storage = _STORAGES[name]
    return _with_retries(lambda: storage.download(uri, Path(dest_dir)))
