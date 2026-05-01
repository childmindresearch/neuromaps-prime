"""Unit tests for graph function functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import networkx as nx
import pytest

from neuromaps_prime.graph import NeuromapsGraph, models

if TYPE_CHECKING:
    from pathlib import Path


class TestNeuromapsGraph:
    """Unit tests with mocked data for Graph object."""

    def test_graph_build_yaml(self, tmp_path: Path) -> None:
        """Test (independently) building graph from YAML file."""
        lh = tmp_path / "src-ALIEN_den-999k_hemi-L_midthickness.surf.gii"
        rh = tmp_path / "src-ALIEN_den-999k_hemi-R_midthickness.surf.gii"
        annot_lh = tmp_path / "src-ALIEN_den-999k_hemi-L_myelin.func.gii"
        annot_rh = tmp_path / "src-ALIEN_den-999k_hemi-R_myelin.func.gii"
        vol = tmp_path / "src-ALIEN_res-1mm_T1w.nii"
        vol_annot = tmp_path / "src-ALIEN_res-1mm_myelin.nii"
        for f in (lh, rh, annot_lh, annot_rh, vol, vol_annot):
            f.touch()

        yaml_content = (
            "nodes:\n"
            "  - ALIEN:\n"
            "      species: extraterrestrial\n"
            "      description: E.T. phone home\n"
            "      surfaces:\n"
            "        999k:\n"
            "          midthickness:\n"
            f"            left: {lh}\n"
            f"            right: {rh}\n"
            "          annotation:\n"
            "            myelin:\n"
            f"              left: {annot_lh}\n"
            f"              right: {annot_rh}\n"
            "      volumes:\n"
            "        1mm:\n"
            f"          T1w: {vol}\n"
            "          annotation:\n"
            "            myelin:\n"
            f"              uri: {vol_annot}\n"
        )
        yaml_file = tmp_path / "test_graph.yaml"
        yaml_file.write_text(yaml_content)

        graph = NeuromapsGraph(yaml_file=yaml_file)
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0

        node = graph.get_node_data("ALIEN")
        assert len(node.surface_annotations) == 2  # left + right
        assert len(node.volume_annotations) == 1
        assert node.surface_annotations[0].label == "myelin"
        assert node.volume_annotations[0].label == "myelin"

    def test_graph_build(self, graph: NeuromapsGraph) -> None:
        """Test graph initialization."""
        info = graph.utils.get_graph_info()
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
            assert hasattr(node, "surface_annotations")
            assert hasattr(node, "volume_annotations")

    def test_get_non_node_data(self, graph: NeuromapsGraph) -> None:
        """Test error is raised if a non-existent node is grabbed."""
        with pytest.raises(ValueError, match="not found"):
            graph.get_node_data("non_existent_node")

    def test_validate_valid_spaces(self, graph: NeuromapsGraph) -> None:
        """Test validation passes for valid spaces."""
        nodes = list(graph.nodes)
        graph.utils.validate_spaces(*nodes[:2])

    def test_validate_invalid_source(self, graph: NeuromapsGraph) -> None:
        """Test validation raises for invalid source space."""
        with pytest.raises(ValueError, match="Source space"):
            graph.utils.validate_spaces("fake_source", next(iter(graph.nodes)))

    def test_validate_invalid_target(self, graph: NeuromapsGraph) -> None:
        """Test validation raises for invalid target space."""
        with pytest.raises(ValueError, match="Target space"):
            graph.utils.validate_spaces(next(iter(graph.nodes)), "fake_target")

    def test_find_path(self, graph: NeuromapsGraph) -> None:
        """Test finding of shortest path."""
        nodes = list(graph.nodes)
        path = graph.utils.find_path(*nodes[:2])
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
        source, target = list(graph.nodes)[:2]
        sf = models.SurfaceTransform(
            name="surface_xfm_test",
            source_space=source,
            target_space=target,
            density="32k",
            hemisphere="left",
            resource_type="midthickness",
            file_path=test_surf,
            provider="test",
            description="Test surface transform",
        )
        graph.add_transform(sf, graph.surface_to_surface_key)
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

    @pytest.mark.xfail(reason="Not in graph YAML yet")
    def test_fetch_surface_annotation(self, graph: NeuromapsGraph) -> None:
        """Test fetching surface atlas."""
        atlas = graph.fetch_surface_annotation(
            space="Yerkes19", density="32k", label="myelin", hemisphere="left"
        )
        assert isinstance(atlas, models.SurfaceAnnotation)

    def test_add_volume_transform_and_fetch(
        self, tmp_path: Path, graph: NeuromapsGraph
    ) -> None:
        """Test adding and fetching volume transform."""
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
            provider="test",
            description="Test volume transform",
        )
        graph.add_transform(vf, graph.volume_to_volume_key)
        fetched = graph.fetch_volume_to_volume_transform(source, target, "1mm", "T1w")
        assert fetched is vf

    def test_add_invalid_transform(self, graph: NeuromapsGraph) -> None:
        """Test adding invalid transform type raises error."""
        with pytest.raises(TypeError, match="Unsupported transform type"):
            graph.add_transform("invalid", key="key")  # type: ignore[arg-type]

    def test_fetch_volume_atlas(self, graph: NeuromapsGraph) -> None:
        """Test fetching volume atlas."""
        atlas = graph.fetch_volume_atlas(
            space="D99", resolution="250um", resource_type="T1w"
        )
        assert isinstance(atlas, models.VolumeAtlas)

    @pytest.mark.xfail(reason="Not in graph YAML yet")
    def test_fetch_volume_annotation(self, graph: NeuromapsGraph) -> None:
        """Test fetching surface atlas."""
        atlas = graph.fetch_volume_annotation(
            space="Yerkes19", resolution="250um", label="myelin"
        )
        assert isinstance(atlas, models.VolumeAnnotation)

    def test_add_surface_atlas(self, tmp_path: Path, graph: NeuromapsGraph) -> None:
        """Test adding a SurfaceAtlas registers it to node and cache."""
        test_surf = tmp_path / "fake_atlas.surf.gii"
        test_surf.touch()
        space = next(iter(graph.nodes))

        atlas = models.SurfaceAtlas(
            name="surface_atlas_test",
            space=space,
            density="32k",
            hemisphere="left",
            resource_type="sphere",
            file_path=test_surf,
            description="Test surface atlas",
        )
        initial_count = len(graph.nodes[space]["data"].surfaces)
        graph.add_atlas(atlas)

        node_data = graph.nodes[space]["data"]
        assert len(node_data.surfaces) == initial_count + 1
        assert atlas in node_data.surfaces

        fetched = graph.fetch_surface_atlas(
            space=space, density="32k", hemisphere="left", resource_type="sphere"
        )
        assert fetched is atlas

    def test_add_volume_atlas(self, tmp_path: Path, graph: NeuromapsGraph) -> None:
        """Test adding a VolumeAtlas registers it to node and cache."""
        test_vol = tmp_path / "fake_atlas.nii.gz"
        test_vol.touch()
        space = next(iter(graph.nodes))

        atlas = models.VolumeAtlas(
            name="volume_atlas_test",
            space=space,
            resolution="1mm",
            resource_type="T1w",
            file_path=test_vol,
            description="Test volume atlas",
        )
        initial_count = len(graph.nodes[space]["data"].volumes)
        graph.add_atlas(atlas)

        node_data = graph.nodes[space]["data"]
        assert len(node_data.volumes) == initial_count + 1
        assert atlas in node_data.volumes

        fetched = graph.fetch_volume_atlas(
            space=space, resolution="1mm", resource_type="T1w"
        )
        assert fetched is atlas

    def test_add_invalid_atlas_type(self, graph: NeuromapsGraph) -> None:
        """Test that a non-atlas type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported atlas type"):
            graph.add_atlas("not_an_atlas")  # type: ignore[arg-type]

    def test_add_atlas_invalid_space(
        self, tmp_path: Path, graph: NeuromapsGraph
    ) -> None:
        """Test that an atlas with a non-existent space raises ValueError."""
        test_surf = tmp_path / "ghost.surf.gii"
        test_surf.touch()

        atlas = models.SurfaceAtlas(
            name="orphan_atlas",
            space="non_existent_space",
            density="32k",
            hemisphere="left",
            resource_type="sphere",
            file_path=test_surf,
            provider="test",
            description="Atlas with unknown space",
        )
        with pytest.raises(ValueError, match="not found"):
            graph.add_atlas(atlas)

    def test_add_multiple_surface_atlases(
        self, tmp_path: Path, graph: NeuromapsGraph
    ) -> None:
        """Test adding multiple SurfaceAtlases accumulates correctly."""
        space = next(iter(graph.nodes))
        initial_count = len(graph.nodes[space]["data"].surfaces)

        for i, hemi in enumerate(("left", "right")):
            surf = tmp_path / f"atlas_{hemi}.surf.gii"
            surf.touch()
            atlas = models.SurfaceAtlas(
                name=f"multi_atlas_{i}",
                space=space,
                density="32k",
                hemisphere=hemi,
                resource_type="midthickness",
                file_path=surf,
                provider="test",
                description=f"Multi surface atlas {hemi}",
            )
            graph.add_atlas(atlas)

        assert len(graph.nodes[space]["data"].surfaces) == initial_count + 2

    def test_search_surf_atlases(self, graph: NeuromapsGraph) -> None:
        """Test searching for surface atlases."""
        atlases = graph.search_surface_atlases(space=next(iter(graph.nodes)))
        assert all(isinstance(a, models.SurfaceAtlas) for a in atlases)
        assert all(hasattr(a, "density") for a in atlases)

    def test_search_surf_transforms(self, graph: NeuromapsGraph) -> None:
        """Test searching for surface transforms."""
        transforms = graph.search_surface_transforms(*list(graph.nodes)[:2])
        assert all(isinstance(t, models.SurfaceTransform) for t in transforms)

    def test_no_surface_atlases_search(self, graph: NeuromapsGraph) -> None:
        """Test empty list returned if no criteria met for atlases."""
        atlases = graph.search_surface_atlases(space="alien")
        assert len(atlases) == 0

    def test_search_vol_atlases(self, graph: NeuromapsGraph) -> None:
        """Test searching for volume atlases."""
        atlases = graph.search_volume_atlases(space=next(iter(graph.nodes)))
        assert all(isinstance(a, models.VolumeAtlas) for a in atlases)
        assert all(hasattr(a, "resolution") for a in atlases)

    def test_search_vol_transforms(self, graph: NeuromapsGraph) -> None:
        """Test searching for volume transforms."""
        transforms = graph.search_volume_transforms("Yerkes19", "CIVETNMT")
        assert all(isinstance(t, models.VolumeTransform) for t in transforms)

    def test_no_volume_atlases_search(self, graph: NeuromapsGraph) -> None:
        """Test empty list returned if no criteria met for atlases."""
        atlases = graph.search_volume_atlases(space="alien")
        assert len(atlases) == 0

    def test_find_highest_density(self, graph: NeuromapsGraph) -> None:
        """Test finding highest density for a space."""
        assert isinstance(graph.find_highest_density(next(iter(graph.nodes))), str)

    def test_find_common_density(self, graph: NeuromapsGraph) -> None:
        """Test finding common density between two spaces."""
        common = graph.find_common_density("Yerkes19", "fsLR")
        assert isinstance(common, str)

    def test_no_common_densities(self, graph: NeuromapsGraph) -> None:
        """Test error raised if no common densities found."""
        with pytest.raises(ValueError, match="No common density"):
            graph.find_common_density("alien", "Yerkes19")

    def test_no_densities(self, graph: NeuromapsGraph) -> None:
        """Test error raised if no densities found."""
        with pytest.raises(ValueError, match="No surface atlases found"):
            graph.find_highest_density("alien")

    @pytest.mark.parametrize("edges", ["surface_to_surface", "volume_to_volume"])
    def test_get_subgraph(self, graph: NeuromapsGraph, edges: str) -> None:
        """Test getting subgraph."""
        subgraph = graph.utils.get_subgraph(edge_type=edges)
        assert isinstance(subgraph, nx.MultiDiGraph)
        assert set(subgraph.nodes) == set(graph.nodes)

    def test_clear_evicts_all_cache_tables(self, graph: NeuromapsGraph) -> None:
        """Test all cache tables are cleared including annotation tables."""
        assert len(graph._cache.surface_atlas) > 0
        assert len(graph._cache.surface_transform) > 0
        graph._cache.clear()
        assert len(graph._cache.surface_atlas) == 0
        assert len(graph._cache.surface_transform) == 0
        assert len(graph._cache.surface_annotation) == 0
        assert len(graph._cache.volume_annotation) == 0
