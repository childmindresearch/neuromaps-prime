"""Tests for grapb object."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from neuromaps_prime.transforms.utils import estimate_surface_density

if TYPE_CHECKING:
    from typing import Literal

    from neuromaps_prime.graph.core import NeuromapsGraph


class TestGraphIntegration:
    """Integration tests for graph module."""

    def test_graph_initialization_with_data_dir(self, graph: NeuromapsGraph) -> None:
        """Test initializing graph with data directory."""
        for node_name in graph.nodes:
            node_data = graph.get_node_data(node_name)
            assert isinstance(node_data.surfaces, list)
            assert isinstance(node_data.volumes, list)
            assert isinstance(node_data.surface_annotations, list)
            assert isinstance(node_data.volume_annotations, list)

    @pytest.mark.parametrize(
        ("transformer_type", "input_file"),
        [
            (
                "label",
                "share/Inputs/CIVETNMT/"
                "src-CIVETNMT_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii",
            ),
            (
                "metric",
                "share/Inputs/CIVETNMT/"
                "src-CIVETNMT_den-41k_hemi-R_desc-vaavg_midthickness.shape.gii",
            ),
        ],
    )
    def test_surface_to_surface(
        self,
        graph: NeuromapsGraph,
        tmp_path: Path,
        transformer_type: Literal["label", "metric"],
        input_file: str,
    ) -> None:
        """Test surface_to_surface transformer."""
        assert isinstance(graph.data_dir, Path)
        output = graph.surface_to_surface_transformer(
            transformer_type=transformer_type,
            input_file=graph.data_dir / input_file,
            source_space="CIVETNMT",
            target_space="fsLR",
            hemisphere="right",
            output_file_path=str(tmp_path / f"test_{transformer_type}.func.gii"),
        )
        assert output is not None
        target_density = graph.find_highest_density(space="fsLR")
        assert estimate_surface_density(output) == target_density

    def test_computed_surface_to_surface(
        self, graph: NeuromapsGraph, tmp_path: Path
    ) -> None:
        """Test fetching surface-to-surface transform with computed edge key."""
        source_space = "CIVETNMT"
        target_space = "fsLR"
        assert graph.data_dir is not None
        input_file = graph.data_dir / (
            "share/Inputs/CIVETNMT/src-CIVETNMT_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii"
        )
        output_file_path = str(
            tmp_path / f"space-{target_space}_output_label.label.gii"
        )

        _ = graph.surface_to_surface_transformer(
            transformer_type="label",
            input_file=input_file,
            source_space=source_space,
            target_space=target_space,
            hemisphere="right",
            output_file_path=output_file_path,
        )
        assert graph.has_edge(
            source_space, target_space, key=graph.surface_to_surface_key
        )

        path = graph.find_path(
            source=source_space,
            target=target_space,
            edge_type=graph.surface_to_surface_key,
        )
        assert len(path) == 2

        source_space = "MEBRAINS"
        shortest_path = graph.find_path(
            source=source_space,
            target=target_space,
            edge_type=graph.surface_to_surface_key,
        )
        assert len(shortest_path) == 3
        assert "CIVETNMT" not in shortest_path

    def test_volume_to_volume(self, graph: NeuromapsGraph, tmp_path: Path) -> None:
        """Test volume_to_volume transformer."""
        assert isinstance(graph.data_dir, Path)

        input_file = (
            graph.data_dir / "share/Inputs/Yerkes19/src-Yerkes19_res-0p50mm_T1w.nii"
        )

        output = graph.volume_to_volume_transformer(
            input_file=input_file,
            source_space="Yerkes19",
            target_space="NMT2Sym",
            resolution="250um",
            resource_type="T1w",
            output_file_path=str(tmp_path / "test_output.nii"),
        )
        assert output.exists()
