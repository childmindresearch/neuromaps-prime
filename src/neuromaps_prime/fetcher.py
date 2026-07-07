"""Helpers for grabbing from remote repositories."""

from pathlib import Path
from urllib.parse import urlparse

from neuromaps_prime import remote

_STORAGES = {"osf": remote.OSFStorage(), "github": remote.GitHubStorage()}
_HOST_MAP = {"osf.io": _STORAGES["osf"], "github.com": _STORAGES["github"]}


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


def download_and_validate(uri: str, dest: str | Path) -> None:
    """Download and validate the file.

    Args:
        uri: Remote URI to fetch data from
        dest: Output file path name

    Raises:
        ValueError: if storage cannot be identified from provided URI
    """
    host = urlparse(uri).hostname
    storage = None
    if host is not None:
        host = host.lower()
        storage = next(
            (v for k, v in _HOST_MAP.items() if host == k or host.endswith(k)), None
        )

    if storage is None:
        raise ValueError(f"Could not identify storage from uri: {uri}")
    storage.download(uri, Path(dest))  # type: ignore[attr-defined]
