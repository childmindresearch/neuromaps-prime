"""Integration tests for the remote file fetchers against live storage.

Every test routes through ``download_and_validate`` — the same production
entry point ``Resource.fetch()`` uses — and writes into pytest's fresh
per-test ``tmp_path``. Because the target file can never already exist there,
the real download is always exercised instead of the "already cached"
short-circuit, and the user's persistent cache is left untouched.
``download_and_validate`` retries transient (rate-limited or brief server)
HTTP errors, so a single flaky response from OSF or GitHub does not fail a run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuromaps_prime.fetcher import download_and_validate

if TYPE_CHECKING:
    from pathlib import Path

    from neuromaps_prime.graph import NeuromapsGraph

# Real resources to fetch, kept small and coarse where possible.
OSF_ATLAS = ("Yerkes19", "10k", "left", "sphere")  # space, density, hemi, type
GITHUB_ATLAS = ("D99", "168k", "left", "sphere")  # space, density, hemi, type
OSF_VOLUME = ("Yerkes29", "800um", "T1w")  # space, resolution, type
OSF_ANNOTATION = ("Yerkes29", "AT_ChimpBNA", "32k", "left")  # space, label, dens, hemi


def _atlas_uri(graph: NeuromapsGraph, coords: tuple[str, str, str, str]) -> str:
    """Return the remote URI the graph stores for a surface atlas tuple."""
    space, density, hemisphere, resource_type = coords
    atlas = graph.fetch_surface_atlas(space, density, hemisphere, resource_type)
    assert atlas is not None, f"Graph has no surface atlas for {coords}"
    assert atlas.uri is not None, f"Graph surface atlas {coords} has no remote URI"
    return atlas.uri


def _volume_uri(graph: NeuromapsGraph, coords: tuple[str, str, str]) -> str:
    """Return the remote URI the graph stores for a volume atlas tuple."""
    space, resolution, resource_type = coords
    atlas = graph.fetch_volume_atlas(space, resolution, resource_type)
    assert atlas is not None, f"Graph has no volume atlas for {coords}"
    assert atlas.uri is not None, f"Graph volume atlas {coords} has no remote URI"
    return atlas.uri


def _annotation_uri(graph: NeuromapsGraph, coords: tuple[str, str, str, str]) -> str:
    """Return the remote URI the graph stores for a surface annotation tuple."""
    space, label, density, hemisphere = coords
    annotation = graph.fetch_surface_annotation(space, label, density, hemisphere)
    assert annotation is not None, f"Graph has no surface annotation for {coords}"
    assert annotation.uri is not None, (
        f"Graph surface annotation {coords} has no remote URI"
    )
    return annotation.uri


def _fetch(uri: str, tmp_path: Path) -> None:
    """Fetch ``uri`` through the production path into ``tmp_path``.

    A successful call already implies the backend verified the file's hash
    (OSF MD5 / GitHub blob SHA) and parsed live metadata, so the assertions
    confirm a real, non-empty file landed in the requested directory.
    """
    result = download_and_validate(uri, tmp_path)
    assert result.parent == tmp_path
    assert result.is_file()
    assert result.stat().st_size > 0


class TestOSFStorage:
    """Fetch real OSF resources through the production path, end to end."""

    def test_download_surface_atlas(
        self, graph: NeuromapsGraph, tmp_path: Path
    ) -> None:
        """Fetch a real surface atlas and verify it lands in the target dir."""
        _fetch(_atlas_uri(graph, OSF_ATLAS), tmp_path)

    def test_download_volume(self, graph: NeuromapsGraph, tmp_path: Path) -> None:
        """Fetch a real volume atlas and verify it lands in the target dir."""
        _fetch(_volume_uri(graph, OSF_VOLUME), tmp_path)

    def test_download_annotation(self, graph: NeuromapsGraph, tmp_path: Path) -> None:
        """Fetch a real surface annotation and verify it lands in the target dir."""
        _fetch(_annotation_uri(graph, OSF_ANNOTATION), tmp_path)


class TestGitHubStorage:
    """Fetch real GitHub resources through the production path, end to end."""

    def test_download_surface_atlas(
        self, graph: NeuromapsGraph, tmp_path: Path
    ) -> None:
        """Fetch a real surface atlas and verify it lands in the target dir."""
        _fetch(_atlas_uri(graph, GITHUB_ATLAS), tmp_path)
