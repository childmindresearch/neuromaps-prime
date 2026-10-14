"""Check that branch-linked resources revalidate without changing other caches."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from neuromaps_prime.graph.models import SurfaceAnnotation

if TYPE_CHECKING:
    from pathlib import Path


def test_github_resource_revalidates_resolved_path(tmp_path: Path) -> None:
    """A repeated fetch of a raw resource reaches the HTTP cache backend."""
    stored = tmp_path / "map.shape.gii"
    stored.write_bytes(b"cached")
    uri = "https://raw.githubusercontent.com/owner/repo/main/map.shape.gii"
    resource = SurfaceAnnotation(
        name="map",
        uri=uri,
        data_dir=tmp_path,
        file_path=stored,
        space="D99",
        density="41k",
        hemisphere="left",
        label="CT",
    )
    with patch(
        "neuromaps_prime.graph.models.download_and_validate", return_value=stored
    ) as download:
        assert resource.fetch() == stored
        assert resource.fetch() == stored
    assert download.call_count == 2
    download.assert_called_with(uri=uri, dest_dir=tmp_path)


def test_osf_resolved_path_stays_cached(tmp_path: Path) -> None:
    """Already resolved OSF paths keep their existing fast-cache behavior."""
    stored = tmp_path / "map.shape.gii"
    stored.touch()
    resource = SurfaceAnnotation(
        name="map",
        uri="https://files.osf.io/v1/resources/example",
        data_dir=tmp_path,
        file_path=stored,
        space="D99",
        density="41k",
        hemisphere="left",
        label="CT",
    )
    with patch("neuromaps_prime.graph.models.download_and_validate") as download:
        assert resource.fetch() == stored
    download.assert_not_called()
