"""Tests for surface transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np
import pytest
from nibabel import GiftiImage
from nibabel.gifti.gifti import GiftiDataArray

from neuromaps_prime.analysis.images import load_data

if TYPE_CHECKING:
    from pathlib import Path

    from neuromaps_prime.graph import NeuromapsGraph


class TestSurfaceTransformIntegration:
    """Integration tests calling Workbench and using real data."""

    ORIGIN = "D99"
    HEMISPHERE: Literal["left"] = "left"

    def _sphere_vertices(self, graph: NeuromapsGraph) -> np.ndarray:
        """Highest-density ORIGIN sphere vertex array (fetch if needed)."""
        density = graph.find_highest_density(self.ORIGIN)
        sphere = graph.fetch_surface_atlas(
            self.ORIGIN, density, self.HEMISPHERE, "sphere"
        )
        assert sphere is not None
        if not sphere.file_path.exists():
            sphere.fetch()
        return load_data(sphere.file_path).array[0]

    @pytest.fixture
    def surface_metric(self, graph: NeuromapsGraph, tmp_path: Path) -> Path:
        """Create a metric from highest-density surface available for given space.

        The metric is constructed by summing the x, y, and z coordinates of each
        given surface vertex.
        """
        metric = self._sphere_vertices(graph).sum(axis=1, dtype=np.float32)
        output = tmp_path / f"{self.ORIGIN}_metric.func.gii"
        nib.save(GiftiImage(darrays=[GiftiDataArray(metric)]), output)
        return output

    @pytest.fixture
    def surface_label(self, graph: NeuromapsGraph, tmp_path: Path) -> Path:
        """Create a label file from highest-density surface available for given space.

        Each vertex is assigned one of N consecutive integer labels so
        the file is a valid parcellation-like label for label resampling.
        """
        n = self._sphere_vertices(graph).shape[0]
        labels = np.arange(n, dtype=np.int32) % 7 + 1
        output = tmp_path / f"{self.ORIGIN}_label.func.gii"
        darr = GiftiDataArray(
            labels, intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32"
        )
        nib.save(GiftiImage(darrays=[darr]), output)
        return output

    @pytest.mark.parametrize(
        "target", ["Yerkes19", "fsLR"], ids=["single_hop", "multi_hop"]
    )
    def test_surface_transform(
        self, graph: NeuromapsGraph, surface_metric: Path, target: str
    ) -> None:
        """Verify Workbench executes a real metric surface transformations."""
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

    # Metric resample tested as part of surface transformation test

    def test_label_resample(self, graph: NeuromapsGraph, surface_label: Path) -> None:
        """Verify Workbench executes a real single-hop label resampling."""
        target = "Yerkes19"
        output = f"{self.ORIGIN}_to_{target}.label.gii"

        result = graph.surface_to_surface_transformer(
            transformer_type="label",
            input_file=surface_label,
            source_space=self.ORIGIN,
            target_space=target,
            hemisphere=self.HEMISPHERE,
            output_file_path=output,
            add_edge=False,
        )

        assert result.path is not None
        assert result.path.exists()

        labels = load_data(result.path).array
        assert labels.size > 0
        assert np.all(np.isfinite(labels))
        # Label resampling preserves discrete (whole-number) labels.
        assert np.all(labels == np.rint(labels))

    def test_computed_edge(self, graph: NeuromapsGraph, surface_metric: Path) -> None:
        """Test fetching transform with computed edge; also tests project_unproject."""
        target = "fsLR"
        output = f"{self.ORIGIN}_to_{target}.func.gii"

        _ = graph.surface_to_surface_transformer(
            transformer_type="metric",
            input_file=surface_metric,
            source_space=self.ORIGIN,
            target_space=target,
            hemisphere=self.HEMISPHERE,
            output_file_path=output,
        )
        assert graph.has_edge(self.ORIGIN, target, key=graph.surface_to_surface_key)

        path = graph.find_path(
            source=self.ORIGIN, target=target, edge_type=graph.surface_to_surface_key
        )
        assert len(path) == 2

        source = "MEBRAINS"
        shortest_path = graph.find_path(
            source, target, edge_type=graph.surface_to_surface_key
        )
        assert len(shortest_path) > 2
        assert self.ORIGIN not in shortest_path
