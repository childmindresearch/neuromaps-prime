"""Helpers for grabbing from remote repositories."""

from pathlib import Path
from typing import Literal

from neuromaps_prime.remote import OSFStorage


def id_storage(uri: str) -> Literal["osf"] | None:
    """Identify the storage type.

    Args:
        uri: Remote URI to fetch data from

    Returns:
        String indicating type of storage (one of 'osf')
    """
    if "osf.io" in uri:
        return "osf"
    return None


def download_and_validate(uri: str, dest: str | Path) -> None:
    """Download and validate the file.

    Args:
        uri: Remote URI to fetch data from
        dest: Output file path name
        token: Optional token to use for remote storage
    """
    match id_storage(uri):
        case "osf":
            OSFStorage().download(uri, Path(dest))
        case _:
            raise ValueError(f"Could not identify storage from uri: {uri}")
