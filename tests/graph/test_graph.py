"""Tests associated for broader graph functionality."""

from pathlib import Path
from typing import Literal
from unittest.mock import patch

import networkx as nx
import pytest

from neuromaps_prime import models
from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.transforms.utils import estimate_surface_density


class TestGraphUnit:
    """Unit tests with mocked data for Graph object."""

    def test_graph_build_yaml(self, tmp_path: Path) -> None:
        """Test (independently) building graph from YAML file."""
        lh = tmp_path / "src-ALIEN_den-999k_hemi-L_midthickness.surf.gii"
        rh = tmp_path / "src-ALIEN_den-999k_hemi-R_midthickness.surf.gii"
        lh.touch()
        rh.touch()
        yaml_content = f"""
nodes:
    - ALIEN:
        species: extraterrestrial
        description: E.T. phone home
        surfaces:
            999k:
                midthickness:
                    left: {lh}
                    right: {rh}
        """
        yaml_file = tmp_path / "test_graph.yaml"
        yaml_file.write_text(yaml_content)

        graph = NeuromapsGraph(yaml_file=yaml_file)
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0

    def test_graph_build(self, graph: NeuromapsGraph) -> None:
        """Test graph initialization."""
        assert graph is not None
        info = graph.get_graph_info()
        assert info["num_nodes"] > 0
        assert info["num_edges"] >= 0
        assert info["num_surfaces"] > 0
        assert info["num_volumes"] >= 0
        assert info["num_surface_to_surface_transforms"] >= 0
        assert info["num_volume_to_volume_transforms"] >= 0

    def test_get_node_data(self, graph: NeuromapsGraph) -> None:
        """Test getting node data with proper error raised."""
        for node_name in graph.nodes:
            node = graph.get_node_data(node_name)
            assert node.name == node_name
            assert hasattr(node, "surfaces")
            assert hasattr(node, "volumes")
        with pytest.raises(ValueError, match="not found"):
            graph.get_node_data("non_existent_node")

    def test_validate(self, graph: NeuromapsGraph) -> None:
        """Test validation method with proper raises."""
        nodes = list(graph.nodes)
        graph.validate(*nodes[:2])
        with pytest.raises(ValueError, match="source space"):
            graph.validate("fake_source", nodes[1])
        with pytest.raises(ValueError, match="target space"):
            graph.validate(nodes[0], "fake_target")

    def test_find_path(self, graph: NeuromapsGraph) -> None:
        """Test finding of shortest path."""
        nodes = list(graph.nodes)
        path = graph.find_path(*nodes[:2])
        assert isinstance(path, list)

    def test_no_valid_path(self, graph: NeuromapsGraph) -> None:
        """Testing no paths return empty."""
        with patch("networkx.shortest_path", side_effect=nx.NetworkXNoPath):
            path = graph.find_path("A", "B")
        assert len(path) == 0

    def test_add_surface_transform_and_fetch(
        self, tmp_path: Path, graph: NeuromapsGraph
    ) -> None:
        """Test adding and fetching surface transform."""
        test_surf = tmp_path / "fake.surf.gii"
        test_surf.touch()

        nodes = list(graph.nodes)
        source, target = nodes[:2]
        sf = models.SurfaceTransform(
            name="surface_xfm_test",
            source_space=source,
            target_space=target,
            density="32k",
            hemisphere="left",
            resource_type="midthickness",
            file_path=test_surf,
            description="Test surface transform",
        )
        graph.add_transform(source, target, graph.surface_to_surface_key, sf)
        fetched = graph.fetch_surface_to_surface_transform(
            source, target, "32k", "left", "midthickness"
        )
        assert fetched is sf

    def test_fetch_surface_atlas(self, graph: NeuromapsGraph) -> None:
        """Test fetching surface atlas."""
        atlas = graph.fetch_surface_atlas(
            space="Yerkes19", density="32k", hemisphere="left", resource_type="sphere"
        )
        assert isinstance(atlas, models.SurfaceAtlas)

    def test_add_volume_transform_and_fetch(
        self, tmp_path: Path, graph: NeuromapsGraph
    ) -> None:
        """Test adding and fetching surface transform."""
        test_vol = tmp_path / "fake.nii.gz"
        test_vol.touch()

        nodes = list(graph.nodes)
        source, target = nodes[:2]
        vf = models.VolumeTransform(
            name="volume_xfm_test",
            source_space=source,
            target_space=target,
            resolution="1mm",
            resource_type="T1w",
            file_path=test_vol,
            description="Test volume transform",
        )
        graph.add_transform(source, target, graph.volume_to_volume_key, vf)
        fetched = graph.fetch_volume_to_volume_transform(source, target, "1mm", "T1w")
        assert fetched is vf

    def test_add_invalid_trasnform(self, graph: NeuromapsGraph) -> None:
        """Test adding invalid transform type raises error."""
        with pytest.raises(TypeError, match="Unsupported transform type"):
            graph.add_transform("source", "target", key="key", transform="invalid")  # type: ignore[arg-type]

    def test_fetch_volume_atlas(self, graph: NeuromapsGraph) -> None:
        """Test fetching volume atlas."""
        atlas = graph.fetch_volume_atlas(
            space="D99", resolution="250um", resource_type="T1w"
        )
        assert isinstance(atlas, models.VolumeAtlas)

    def test_search_surf_atlases_and_xfms(self, graph: NeuromapsGraph) -> None:
        """Test searching for surface atlases and transforms."""
        nodes = list(graph.nodes)
        source, target = nodes[:2]
        atlases = graph.search_surface_atlases(space=source)
        assert all(isinstance(atlas, models.SurfaceAtlas) for atlas in atlases)
        assert all(hasattr(a, "density") for a in atlases)

        transforms = graph.search_surface_transforms(source, target)
        for t in transforms:
            assert isinstance(t, models.SurfaceTransform)

    def test_no_surface_atlases_search(self, graph: NeuromapsGraph) -> None:
        """Test empty list returned if no criterias met for atlases."""
        atlases = graph.search_surface_atlases(space="alien")
        assert len(atlases) == 0

    def test_find_highest_and_common_densities(self, graph: NeuromapsGraph) -> None:
        """Test density searching methods."""
        nodes = list(graph.nodes)
        source, target = nodes[:2]

        highest = graph.find_highest_density(source)
        assert isinstance(highest, str)

        if graph.search_surface_transforms(source, target):
            common = graph.find_common_density(source, target)
            assert isinstance(common, str)

    def test_no_common_densities(self, graph: NeuromapsGraph) -> None:
        """Test error raised if no common densities found."""
        with pytest.raises(ValueError, match="No common density"):
            graph.find_common_density("alien", "Yerkes19")

    def test_no_densities(self, graph: NeuromapsGraph) -> None:
        """Test error raised if no densities found."""
        with pytest.raises(ValueError, match="No atlases found"):
            graph.find_highest_density("alien")

    @pytest.mark.parametrize("edges", ("surface_to_surface", "volume_to_volume"))
    def test_get_subgraph(self, graph: NeuromapsGraph, edges: str) -> None:
        """Test getting subgraph."""
        subgraph = graph.get_subgraph(edges=edges)
        assert isinstance(subgraph, nx.MultiDiGraph)
        assert set(subgraph.nodes) == set(graph.nodes)


class TestGraphIntegration:
    """Integration tests for graph module."""

    def test_graph_initialization_with_data_dir(self, graph: NeuromapsGraph) -> None:
        """Test initializing graph with data directory."""
        assert graph is not None
        info = graph.get_graph_info()
        assert info["num_nodes"] > 0
        assert info["num_edges"] >= 0
        for node_name in graph.nodes:
            node_data = graph.get_node_data(node_name)
            assert isinstance(node_data.surfaces, list)
            assert isinstance(node_data.volumes, list)

    @pytest.mark.parametrize(
        "transformer_type,input_file",
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
    @pytest.mark.usefixtures("require_workbench")
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
            target_space="S1200",
            hemisphere="right",
            output_file_path=str(tmp_path / f"test_{transformer_type}.func.gii"),
        )
        assert output is not None
        target_density = graph.find_highest_density(space="S1200")
        if transformer_type == "metric":
            assert estimate_surface_density(output) == target_density
        elif transformer_type == "label":
            assert estimate_surface_density(output) == target_density

    @pytest.mark.usefixtures("require_workbench")
    def test_computed_surface_to_surface(
        self, graph: NeuromapsGraph, tmp_path: Path
    ) -> None:
        """Test fetching surface-to-surface transform with computed edge key."""
        source_space = "CIVETNMT"
        target_space = "S1200"
        input_file = graph.data_dir / (
            "share/Inputs/CIVETNMT/src-CIVETNMT_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii"
        )
        output_file_path = str(
            tmp_path / f"space-{target_space}_output_label.label.gii"
        )

        # A dummy transform to add edge to the graph
        _ = graph.surface_to_surface_transformer(
            transformer_type="label",
            input_file=input_file,
            source_space=source_space,
            target_space=target_space,
            hemisphere="right",
            output_file_path=output_file_path,
        )

        # Assert the graph has the computed edge
        assert graph.has_edge(
            source_space, target_space, key=graph.surface_to_surface_key
        )

        # Assert if the computed edge is used.
        path = graph.find_path(
            source=source_space,
            target=target_space,
            edge_type=graph.surface_to_surface_key,
        )
        assert len(path) == 2

        # Assert if this computed edge does not corrupt the shortest path finding.
        source_space = "MEBRAINS"
        shortest_path = graph.find_path(
            source=source_space,
            target=target_space,
            edge_type=graph.surface_to_surface_key,
        )
        assert len(shortest_path) == 3
        assert "CIVETNMT" not in shortest_path

    @pytest.mark.usefixtures("require_ants")
    def test_volume_to_volume(self, graph: NeuromapsGraph, tmp_path: Path) -> None:
        """Test surface_to_surface transformer."""
        assert isinstance(graph.data_dir, Path)

        input_file = (
            graph.data_dir / "share/Inputs/Yerkes19/src-Yerkes19_res-0p50mm_T1w.nii"
        )

        output = graph.volume_to_volume_transformer(
            input_file=input_file,
            source_space="Yerkes19",
            target_space="NMT2Sym",
            resolution="250um",
            resource_type="composite",
            output_file_path=str(tmp_path / "test_output.nii"),
        )
        assert output.exists()
