"""Tests for surface transformations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from neuromaps_prime.graph import NeuromapsGraph

from neuromaps_prime.transforms import utils
from neuromaps_prime.transforms.surface import surface_sphere_project_unproject


class TestSurfaceTransformIntegration:
    """Integration tests calling Workbench and using real data."""

    def test_surface_sphere_project_unproject(
        self, tmp_path: Path, graph: NeuromapsGraph
    ) -> None:
        """Integration test of surface_sphere_project_unproject."""
        sphere_in = graph.fetch_surface_to_surface_transform(
            source="fsLR",
            target="Yerkes19",
            density="32k",
            hemisphere="left",
            resource_type="sphere",
        )
        sphere_project_to = graph.fetch_surface_atlas(
            space="Yerkes19",
            density="32k",
            hemisphere="left",
            resource_type="sphere",
        )
        sphere_unproject_from = graph.fetch_surface_to_surface_transform(
            source="Yerkes19",
            target="D99",
            density="32k",
            hemisphere="left",
            resource_type="sphere",
        )
        sphere_out = tmp_path / "out_sphere.surf.gii"
        assert sphere_in is not None
        assert sphere_project_to is not None
        assert sphere_unproject_from is not None
        result = surface_sphere_project_unproject(
            sphere_in=sphere_in.fetch(),
            sphere_project_to=sphere_project_to.fetch(),
            sphere_unproject_from=sphere_unproject_from.fetch(),
            sphere_out=str(sphere_out),
        )
        assert utils.get_vertex_count(
            Path(sphere_in.fetch())
        ) == utils.get_vertex_count(result.sphere_out)

    @pytest.mark.skip(reason="No metric data for resmapling integration test")
    def test_metric_resample(self) -> None:
        """Test metric resampling."""

    @pytest.mark.skip(reason="No label data for resampling integration test")
    def test_label_resample(self) -> None:
        """Test label resampling."""
