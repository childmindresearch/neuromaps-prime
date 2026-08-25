"""Tests for surface transformations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np
import pytest
from nibabel import GiftiImage
from nibabel.gifti.gifti import GiftiDataArray

from neuromaps_prime.analysis.images import load_data

if TYPE_CHECKING:
    from neuromaps_prime.graph import NeuromapsGraph

from neuromaps_prime.transforms import utils
from neuromaps_prime.transforms.surface import surface_sphere_project_unproject


class TestSurfaceTransformIntegration:
    """Integration tests calling Workbench and using real data."""

    ORIGIN = "D99"
    HEMISPHERE: Literal["left"] = "left"

    @pytest.fixture
    def surface_metric(self, graph: NeuromapsGraph, tmp_path: Path) -> Path:
        """Create a metric from highest-density surface available for given space.

        The metric is constructed by summing the x, y, and z coordinates of each
        given surface vertex.
        """
        density = graph.find_highest_density(self.ORIGIN)
        sphere = graph.fetch_surface_atlas(
            self.ORIGIN,
            density,
            self.HEMISPHERE,
            "sphere",
        )

        assert sphere is not None
        if not sphere.file_path.exists():
            sphere.fetch()

        data = load_data(sphere.file_path)
        coords = data.array[0]
        metric = coords.sum(axis=1, dtype=np.float32)
        output = tmp_path / f"{self.ORIGIN}_{density}_metric.func.gii"

        nib.save(GiftiImage(darrays=[GiftiDataArray(metric)]), output)

        return output

    def test_single_hop_surface_transform(
        self,
        graph: NeuromapsGraph,
        surface_metric: Path,
    ) -> None:
        """Verify Workbench executes a real single-hop surface transformation."""
        target = "Yerkes19"
        output = f"{self.ORIGIN}_to_{target}.func.gii"

        result = graph.surface_to_surface_transformer(
            transformer_type="metric",
            input_file=surface_metric,
            source_space=self.ORIGIN,
            target_space=target,
            hemisphere=self.HEMISPHERE,
            output_file_path=output,
            add_edge=False,
        )

        assert result.path is not None
        assert result.path.exists()

        transformed = load_data(result.path).array
        assert transformed.size > 0
        assert np.all(np.isfinite(transformed))

    def test_multihop_surface_transform(
        self,
        graph: NeuromapsGraph,
        surface_metric: Path,
    ) -> None:
        """Verify Workbench executes a real multi-hop surface transformation."""
        target = "fsLR"
        output = f"{self.ORIGIN}_to_{target}.func.gii"

        result = graph.surface_to_surface_transformer(
            transformer_type="metric",
            input_file=surface_metric,
            source_space=self.ORIGIN,
            target_space=target,
            hemisphere=self.HEMISPHERE,
            output_file_path=output,
            add_edge=False,
        )

        assert result.path is not None
        assert result.path.exists()

        transformed = load_data(result.path).array
        assert transformed.size > 0
        assert np.all(np.isfinite(transformed))

    @pytest.mark.skip(reason="Tested as part of single / multi-hop surface transform")
    def test_metric_resample(self) -> None:
        """Test metric resampling."""

    @pytest.mark.skip(reason="No label data for resampling integration test")
    def test_label_resample(self) -> None:
        """Test label resampling."""

    def test_surface_sphere_project_unproject(
        self, tmp_path: Path, graph: NeuromapsGraph
    ) -> None:
        """Integration test of surface_sphere_project_unproject.

        Note: hard code source / targets to ensure method called.
        """
        sphere_in = graph.fetch_surface_to_surface_transform(
            source="fsLR",
            target="Yerkes19",
            density="32k",
            hemisphere=self.HEMISPHERE,
            resource_type="sphere",
        )
        sphere_project_to = graph.fetch_surface_atlas(
            space="Yerkes19",
            density="32k",
            hemisphere=self.HEMISPHERE,
            resource_type="sphere",
        )
        sphere_unproject_from = graph.fetch_surface_to_surface_transform(
            source="Yerkes19",
            target="D99",
            density="32k",
            hemisphere=self.HEMISPHERE,
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
