"""Tests for grapb object."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
