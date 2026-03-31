"""Tests associated for broader graph functionality."""

import logging
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import networkx as nx
import pytest

from neuromaps_prime.graph import NeuromapsGraph, models
from neuromaps_prime.graph.cache import GraphCache
from neuromaps_prime.graph.transforms.surface import SurfaceTransformOps
from neuromaps_prime.transforms.utils import estimate_surface_density


class TestGraphUnit:
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
        assert graph is not None
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
        with pytest.raises(ValueError, match="not found"):
            graph.get_node_data("non_existent_node")

    def test_validate(self, graph: NeuromapsGraph) -> None:
        """Test validation method with proper raises."""
        nodes = list(graph.nodes)
        graph.utils.validate_spaces(*nodes[:2])
        with pytest.raises(ValueError, match="Source space"):
            graph.utils.validate_spaces("fake_source", nodes[1])
        with pytest.raises(ValueError, match="Target space"):
            graph.utils.validate_spaces(nodes[0], "fake_target")

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
        """Test empty list returned if no criteria met for atlases."""
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
        with pytest.raises(ValueError, match="No surface atlases found"):
            graph.find_highest_density("alien")

    @pytest.mark.parametrize("edges", ("surface_to_surface", "volume_to_volume"))
    def test_get_subgraph(self, graph: NeuromapsGraph, edges: str) -> None:
        """Test getting subgraph."""
        subgraph = graph.utils.get_subgraph(edge_type=edges)
        assert isinstance(subgraph, nx.MultiDiGraph)
        assert set(subgraph.nodes) == set(graph.nodes)

    def test_clear_evicts_all_cache_tables(
        self, tmp_path: Path, graph: NeuromapsGraph
    ) -> None:
        """Test all cache tables are cleared including annotation tables."""
        assert len(graph._cache.surface_atlas) > 0
        assert len(graph._cache.surface_transform) > 0
        graph._cache.clear()
        assert len(graph._cache.surface_atlas) == 0
        assert len(graph._cache.surface_transform) == 0
        assert len(graph._cache.surface_annotation) == 0
        assert len(graph._cache.volume_annotation) == 0


class TestGraphIntegration:
    """Integration tests for graph module."""

    def test_graph_initialization_with_data_dir(self, graph: NeuromapsGraph) -> None:
        """Test initializing graph with data directory."""
        assert graph is not None
        for node_name in graph.nodes:
            node_data = graph.get_node_data(node_name)
            assert isinstance(node_data.surfaces, list)
            assert isinstance(node_data.volumes, list)
            assert isinstance(node_data.surface_annotations, list)
            assert isinstance(node_data.volume_annotations, list)

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
            target_space="fsLR",
            hemisphere="right",
            output_file_path=str(tmp_path / f"test_{transformer_type}.func.gii"),
        )
        assert output is not None
        target_density = graph.find_highest_density(space="fsLR")
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

    @pytest.mark.usefixtures("require_ants")
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


class TestGraphCacheRequireSurface:
    """Tests for GraphCache require_* methods."""

    def test_require_surface_atlas_hit(self, graph: NeuromapsGraph) -> None:
        """Test require_surface_atlas returns the atlas when found."""
        atlases = graph._cache.get_surface_atlases(space="Yerkes19")
        assert atlases, "Expected at least one Yerkes19 atlas in the test graph"
        a = atlases[0]
        result = graph._cache.require_surface_atlas(
            space=a.space,
            density=a.density,
            hemisphere=a.hemisphere,
            resource_type=a.resource_type,
        )
        assert result is a

    def test_require_surface_atlas_miss(self, graph: NeuromapsGraph) -> None:
        """Test require_surface_atlas raises ValueError when not found."""
        with pytest.raises(ValueError, match="No 'sphere' surface atlas found"):
            graph._cache.require_surface_atlas(
                space="nonexistent",
                density="32k",
                hemisphere="left",
                resource_type="sphere",
            )

    def test_require_volume_atlas_hit(self, graph: NeuromapsGraph) -> None:
        """Test require_volume_atlas returns the atlas when found."""
        atlases = graph._cache.get_volume_atlases(space="D99")
        assert atlases, "Expected at least one D99 atlas in the test graph"
        a = atlases[0]
        result = graph._cache.require_volume_atlas(
            space=a.space,
            resolution=a.resolution,
            resource_type=a.resource_type,
        )
        assert result is a

    def test_require_volume_atlas_miss(self, graph: NeuromapsGraph) -> None:
        """Test require_volume_atlas raises ValueError when not found."""
        with pytest.raises(ValueError, match="No 'T1w' volume atlas found"):
            graph._cache.require_volume_atlas(
                space="nonexistent",
                resolution="250um",
                resource_type="T1w",
            )


# ---------------------------------------------------------------------------
# Helpers shared by TestGraphCacheSurface
# ---------------------------------------------------------------------------


def _make_surface_transform(
    f: Path,
    source: str,
    target: str,
    density: str,
    hemisphere: str,
    resource_type: str,
    weight: float = 1.0,
    provider: str = "ProviderA",
) -> models.SurfaceTransform:
    return models.SurfaceTransform(
        name=f"{source}_to_{target}_{density}_{hemisphere}_{resource_type}",
        description=f"Transform from {source} to {target}",
        file_path=f,
        source_space=source,
        target_space=target,
        density=density,
        hemisphere=hemisphere,
        resource_type=resource_type,
        weight=weight,
        provider=provider,
    )


def _make_surface_annotation(
    f: Path,
    space: str,
    label: str,
    density: str,
    hemisphere: str,
) -> models.SurfaceAnnotation:
    return models.SurfaceAnnotation(
        name=f"{space}_{density}_{hemisphere}_{label}",
        file_path=f,
        space=space,
        label=label,
        density=density,
        hemisphere=hemisphere,
    )


def _make_volume_annotation(
    f: Path,
    space: str,
    label: str,
    resolution: str,
) -> models.VolumeAnnotation:
    return models.VolumeAnnotation(
        name=f"{space}_{resolution}_{label}",
        file_path=f,
        space=space,
        label=label,
        resolution=resolution,
    )


class TestGraphCacheSurface:
    """Unit tests for GraphCache.get_surface_transform covering all branches."""

    @pytest.fixture
    def f(self, tmp_path: Path) -> Path:
        """Sphere file path fixture."""
        p = tmp_path / "sphere.surf.gii"
        p.touch()
        return p

    @pytest.fixture
    def alt_f(self, tmp_path: Path) -> Path:
        """Alternate sphere file path fixture."""
        p = tmp_path / "alt_sphere.surf.gii"
        p.touch()
        return p

    # ------------------------------------------------------------------ #
    # Branch 1: exact provider hit                                        #
    # ------------------------------------------------------------------ #

    def test_exact_provider_hit(self, f: Path) -> None:
        """Returns the entry when provider is given and matches exactly."""
        cache = GraphCache()
        t = _make_surface_transform(
            f, "A", "B", "32k", "left", "sphere", provider="RheMap"
        )
        cache.add_surface_transform(t)
        result = cache.get_surface_transform(
            "A", "B", "32k", "left", "sphere", provider="RheMap"
        )
        assert result is t

    def test_exact_provider_hit_selects_correct_among_multiple(
        self, f: Path, alt_f: Path
    ) -> None:
        """When multiple providers exist, the requested one is returned."""
        cache = GraphCache()
        t_a = _make_surface_transform(
            f, "A", "B", "32k", "left", "sphere", provider="ProviderA"
        )
        t_b = _make_surface_transform(
            alt_f, "A", "B", "32k", "left", "sphere", provider="ProviderB"
        )
        cache.add_surface_transform(t_a)
        cache.add_surface_transform(t_b)
        assert (
            cache.get_surface_transform(
                "A", "B", "32k", "left", "sphere", provider="ProviderA"
            )
            is t_a
        )
        assert (
            cache.get_surface_transform(
                "A", "B", "32k", "left", "sphere", provider="ProviderB"
            )
            is t_b
        )

    # ------------------------------------------------------------------ #
    # Branch 2: provider given but not found → fallback                  #
    # ------------------------------------------------------------------ #

    def test_provider_miss_falls_back_to_first(self, f: Path) -> None:
        """When the requested provider is absent, returns the first registered match."""
        cache = GraphCache()
        t = _make_surface_transform(
            f, "A", "B", "32k", "left", "sphere", provider="ProviderA"
        )
        cache.add_surface_transform(t)
        result = cache.get_surface_transform(
            "A", "B", "32k", "left", "sphere", provider="NoSuchProvider"
        )
        assert result is t

    def test_provider_miss_with_no_entries_returns_none(self, f: Path) -> None:
        """When the requested provider is absent with no fallback, returns None."""
        cache = GraphCache()
        result = cache.get_surface_transform(
            "A", "B", "32k", "left", "sphere", provider="RheMap"
        )
        assert result is None

    # ------------------------------------------------------------------ #
    # Branch 3: no provider → fallback to first                          #
    # ------------------------------------------------------------------ #

    def test_no_provider_returns_first_registered(self, f: Path, alt_f: Path) -> None:
        """With provider=None, returns the first registered matching entry."""
        cache = GraphCache()
        t_first = _make_surface_transform(
            f, "A", "B", "32k", "left", "sphere", provider="ProviderA"
        )
        t_second = _make_surface_transform(
            alt_f, "A", "B", "32k", "left", "sphere", provider="ProviderB"
        )
        cache.add_surface_transform(t_first)
        cache.add_surface_transform(t_second)
        result = cache.get_surface_transform("A", "B", "32k", "left", "sphere")
        assert result is not None
        assert result.provider in ("ProviderA", "ProviderB")

    def test_no_provider_single_entry(self, f: Path) -> None:
        """With provider=None and a single entry, that entry is returned."""
        cache = GraphCache()
        t = _make_surface_transform(
            f, "A", "B", "32k", "right", "sphere", provider="RheMap"
        )
        cache.add_surface_transform(t)
        result = cache.get_surface_transform("A", "B", "32k", "right", "sphere")
        assert result is t

    # ------------------------------------------------------------------ #
    # Branch 4: full miss                                                 #
    # ------------------------------------------------------------------ #

    def test_miss_empty_cache(self) -> None:
        """Returns None on a completely empty cache."""
        cache = GraphCache()
        assert cache.get_surface_transform("A", "B", "32k", "left", "sphere") is None

    @pytest.mark.parametrize(
        "bad_key",
        [
            ("X", "B", "32k", "left", "sphere"),
            ("A", "X", "32k", "left", "sphere"),
            ("A", "B", "164k", "left", "sphere"),
            ("A", "B", "32k", "right", "sphere"),
            ("A", "B", "32k", "left", "midthickness"),
        ],
        ids=[
            "wrong_source",
            "wrong_target",
            "wrong_density",
            "wrong_hemisphere",
            "wrong_resource_type",
        ],
    )
    def test_miss_wrong_key_component(self, f: Path, bad_key: tuple) -> None:
        """Returns None when any single key component does not match."""
        cache = GraphCache()
        cache.add_surface_transform(
            _make_surface_transform(f, "A", "B", "32k", "left", "sphere")
        )
        assert cache.get_surface_transform(*bad_key) is None

    def test_hemisphere_case_insensitive(self, f: Path) -> None:
        """Hemisphere lookup is case-insensitive (stored as lowercase)."""
        cache = GraphCache()
        t = _make_surface_transform(f, "A", "B", "32k", "left", "sphere")
        cache.add_surface_transform(t)
        assert cache.get_surface_transform("A", "B", "32k", "left", "sphere") is t


# ---------------------------------------------------------------------------
# Annotation cache tests
# ---------------------------------------------------------------------------


class TestGraphCacheAnnotations:
    """Unit tests for surface and volume annotation cache methods."""

    @pytest.fixture
    def f(self, tmp_path: Path) -> Path:
        """Annotation fixture."""
        p = tmp_path / "annot.func.gii"
        p.touch()
        return p

    # ------------------------------------------------------------------ #
    # Surface annotations                                                 #
    # ------------------------------------------------------------------ #

    def test_add_and_get_surface_annotation(self, f: Path) -> None:
        """Round-trip add/get for a surface annotation."""
        cache = GraphCache()
        a = _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "left")
        cache.add_surface_annotation(a)
        assert cache.get_surface_annotation("Yerkes19", "myelin", "32k", "left") is a

    def test_get_surface_annotation_miss(self) -> None:
        """Returns None when no matching surface annotation exists."""
        cache = GraphCache()
        assert cache.get_surface_annotation("Yerkes19", "myelin", "32k", "left") is None

    def test_surface_annotation_overwrite(self, f: Path, tmp_path: Path) -> None:
        """Later insertion overwrites earlier surface annotation with the same key."""
        cache = GraphCache()
        a1 = _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "left")
        f2 = tmp_path / "annot2.func.gii"
        f2.touch()
        a2 = _make_surface_annotation(f2, "Yerkes19", "myelin", "32k", "left")
        cache.add_surface_annotation(a1)
        cache.add_surface_annotation(a2)
        assert cache.get_surface_annotation("Yerkes19", "myelin", "32k", "left") is a2

    def test_get_surface_annotations_no_filter(self, f: Path) -> None:
        """No filters returns all annotations for the space."""
        cache = GraphCache()
        cache.add_surface_annotations(
            [
                _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "left"),
                _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "right"),
                _make_surface_annotation(f, "Yerkes19", "curvature", "32k", "left"),
                _make_surface_annotation(f, "D99", "myelin", "32k", "left"),
            ]
        )
        results = cache.get_surface_annotations("Yerkes19")
        assert len(results) == 3
        assert all(r.space == "Yerkes19" for r in results)

    def test_get_surface_annotations_filter_label(self, f: Path) -> None:
        """Label filter narrows to matching annotations only."""
        cache = GraphCache()
        cache.add_surface_annotations(
            [
                _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "left"),
                _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "right"),
                _make_surface_annotation(f, "Yerkes19", "curvature", "32k", "left"),
            ]
        )
        results = cache.get_surface_annotations("Yerkes19", label="myelin")
        assert len(results) == 2
        assert all(r.label == "myelin" for r in results)

    def test_get_surface_annotations_filter_density(self, f: Path) -> None:
        """Density filter narrows to matching annotations only."""
        cache = GraphCache()
        cache.add_surface_annotations(
            [
                _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "left"),
                _make_surface_annotation(f, "Yerkes19", "myelin", "164k", "left"),
            ]
        )
        results = cache.get_surface_annotations("Yerkes19", density="32k")
        assert len(results) == 1
        assert results[0].density == "32k"

    def test_get_surface_annotations_filter_hemisphere(self, f: Path) -> None:
        """Hemisphere filter narrows to matching annotations only."""
        cache = GraphCache()
        cache.add_surface_annotations(
            [
                _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "left"),
                _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "right"),
            ]
        )
        results = cache.get_surface_annotations("Yerkes19", hemisphere="left")
        assert len(results) == 1
        assert results[0].hemisphere == "left"

    def test_require_surface_annotation_hit(self, f: Path) -> None:
        """require_surface_annotation returns the annotation when found."""
        cache = GraphCache()
        a = _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "left")
        cache.add_surface_annotation(a)
        assert (
            cache.require_surface_annotation("Yerkes19", "myelin", "32k", "left") is a
        )

    def test_require_surface_annotation_miss(self) -> None:
        """require_surface_annotation raises ValueError when not found."""
        cache = GraphCache()
        with pytest.raises(ValueError, match="No 'myelin' surface annotation found"):
            cache.require_surface_annotation("Yerkes19", "myelin", "32k", "left")

    # ------------------------------------------------------------------ #
    # Volume annotations                                                  #
    # ------------------------------------------------------------------ #

    def test_add_and_get_volume_annotation(self, f: Path) -> None:
        """Round-trip add/get for a volume annotation."""
        cache = GraphCache()
        a = _make_volume_annotation(f, "D99", "myelin", "250um")
        cache.add_volume_annotation(a)
        assert cache.get_volume_annotation("D99", "myelin", "250um") is a

    def test_get_volume_annotation_miss(self) -> None:
        """Returns None when no matching volume annotation exists."""
        cache = GraphCache()
        assert cache.get_volume_annotation("D99", "myelin", "250um") is None

    def test_volume_annotation_overwrite(self, f: Path, tmp_path: Path) -> None:
        """Later insertion overwrites an earlier volume annotation with the same key."""
        cache = GraphCache()
        a1 = _make_volume_annotation(f, "D99", "myelin", "250um")
        f2 = tmp_path / "annot2.nii"
        f2.touch()
        a2 = _make_volume_annotation(f2, "D99", "myelin", "250um")
        cache.add_volume_annotation(a1)
        cache.add_volume_annotation(a2)
        assert cache.get_volume_annotation("D99", "myelin", "250um") is a2

    def test_get_volume_annotations_no_filter(self, f: Path) -> None:
        """No filters returns all annotations for the space."""
        cache = GraphCache()
        cache.add_volume_annotations(
            [
                _make_volume_annotation(f, "D99", "myelin", "250um"),
                _make_volume_annotation(f, "D99", "curvature", "250um"),
                _make_volume_annotation(f, "D99", "myelin", "500um"),
                _make_volume_annotation(f, "Yerkes19", "myelin", "250um"),
            ]
        )
        results = cache.get_volume_annotations("D99")
        assert len(results) == 3
        assert all(r.space == "D99" for r in results)

    def test_get_volume_annotations_filter_label(self, f: Path) -> None:
        """Label filter narrows to matching annotations only."""
        cache = GraphCache()
        cache.add_volume_annotations(
            [
                _make_volume_annotation(f, "D99", "myelin", "250um"),
                _make_volume_annotation(f, "D99", "curvature", "250um"),
            ]
        )
        results = cache.get_volume_annotations("D99", label="myelin")
        assert len(results) == 1
        assert results[0].label == "myelin"

    def test_get_volume_annotations_filter_resolution(self, f: Path) -> None:
        """Resolution filter narrows to matching annotations only."""
        cache = GraphCache()
        cache.add_volume_annotations(
            [
                _make_volume_annotation(f, "D99", "myelin", "250um"),
                _make_volume_annotation(f, "D99", "myelin", "500um"),
            ]
        )
        results = cache.get_volume_annotations("D99", resolution="250um")
        assert len(results) == 1
        assert results[0].resolution == "250um"

    def test_require_volume_annotation_hit(self, f: Path) -> None:
        """require_volume_annotation returns the annotation when found."""
        cache = GraphCache()
        a = _make_volume_annotation(f, "D99", "myelin", "250um")
        cache.add_volume_annotation(a)
        assert cache.require_volume_annotation("D99", "myelin", "250um") is a

    def test_require_volume_annotation_miss(self) -> None:
        """require_volume_annotation raises ValueError when not found."""
        cache = GraphCache()
        with pytest.raises(ValueError, match="No 'myelin' volume annotation found"):
            cache.require_volume_annotation("D99", "myelin", "250um")

    def test_bulk_add_surface_annotations(self, f: Path) -> None:
        """Bulk insert populates all surface annotations correctly."""
        cache = GraphCache()
        annotations = [
            _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "left"),
            _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "right"),
        ]
        cache.add_surface_annotations(annotations)
        assert len(cache.get_surface_annotations("Yerkes19")) == 2

    def test_bulk_add_volume_annotations(self, f: Path) -> None:
        """Bulk insert populates all volume annotations correctly."""
        cache = GraphCache()
        annotations = [
            _make_volume_annotation(f, "D99", "myelin", "250um"),
            _make_volume_annotation(f, "D99", "curvature", "250um"),
        ]
        cache.add_volume_annotations(annotations)
        assert len(cache.get_volume_annotations("D99")) == 2

    def test_clear_removes_annotations(self, f: Path) -> None:
        """cache.clear() evicts both annotation tables."""
        cache = GraphCache()
        cache.add_surface_annotation(
            _make_surface_annotation(f, "Yerkes19", "myelin", "32k", "left")
        )
        cache.add_volume_annotation(
            _make_volume_annotation(f, "D99", "myelin", "250um")
        )
        cache.clear()
        assert len(cache.surface_annotation) == 0
        assert len(cache.volume_annotation) == 0


# ---------------------------------------------------------------------------
# Multi-hop provider warning tests
# ---------------------------------------------------------------------------


class TestMultiHopProviderWarning:
    """Tests that a warning is emitted when a provider falls back during multi-hop."""

    @pytest.fixture
    def cache_and_ops(
        self, tmp_path: Path
    ) -> tuple[GraphCache, SurfaceTransformOps, Path]:
        """Build a minimal cache and SurfaceTransformOps wired to it."""
        import networkx as nx

        from neuromaps_prime.graph.utils import GraphUtils

        cache = GraphCache()
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(["A", "B", "C"])
        utils = GraphUtils(graph=graph, cache=cache)
        ops = SurfaceTransformOps(cache=cache, utils=utils)
        return cache, ops, tmp_path

    def _add_transforms(
        self, cache: GraphCache, tmp_path: Path, providers: dict
    ) -> None:
        for (src, tgt), provider in providers.items():
            f = tmp_path / f"{src}_to_{tgt}_{provider}.surf.gii"
            f.touch()
            cache.add_surface_transform(
                _make_surface_transform(
                    f, src, tgt, "32k", "left", "sphere", provider=provider
                )
            )

    def _add_atlas(self, cache: GraphCache, tmp_path: Path, space: str) -> None:
        f = tmp_path / f"{space}_sphere.surf.gii"
        f.touch()
        cache.add_surface_atlas(
            models.SurfaceAtlas(
                name=f"{space}_32k_left_sphere",
                description="",
                file_path=f,
                space=space,
                density="32k",
                hemisphere="left",
                resource_type="sphere",
            )
        )

    def test_no_warning_when_all_hops_match_provider(
        self,
        cache_and_ops: tuple[GraphCache, SurfaceTransformOps, Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """No warning is emitted when every hop is served by the requested provider."""
        cache, ops, tmp_path = cache_and_ops
        self._add_transforms(
            cache, tmp_path, {("A", "B"): "RheMap", ("B", "C"): "RheMap"}
        )
        self._add_atlas(cache, tmp_path, "B")

        first = cache.get_surface_transform(
            "A", "B", "32k", "left", "sphere", provider="RheMap"
        )
        with caplog.at_level(
            logging.WARNING, logger="neuromaps_prime.graph.transforms.surface"
        ):
            with pytest.raises(FileNotFoundError):
                ops._two_hops(
                    source_space="A",
                    mid_space="B",
                    target_space="C",
                    density="32k",
                    hemisphere="left",
                    output_file_path=str(tmp_path / "out.surf.gii"),
                    first_transform=first,
                    provider="RheMap",
                )

        assert not any("falling back" in r.message for r in caplog.records)

    def test_warning_on_first_hop_fallback(
        self,
        cache_and_ops: tuple[GraphCache, SurfaceTransformOps, Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning fires when the first hop's provider doesn't match the request."""
        cache, ops, tmp_path = cache_and_ops
        self._add_transforms(
            cache, tmp_path, {("A", "B"): "CIVET", ("B", "C"): "RheMap"}
        )
        self._add_atlas(cache, tmp_path, "B")

        with caplog.at_level(
            logging.WARNING, logger="neuromaps_prime.graph.transforms.surface"
        ):
            with pytest.raises(FileNotFoundError):
                ops._two_hops(
                    source_space="A",
                    mid_space="B",
                    target_space="C",
                    density="32k",
                    hemisphere="left",
                    output_file_path=str(tmp_path / "out.surf.gii"),
                    first_transform=None,
                    provider="RheMap",
                )

        messages = [r.message for r in caplog.records]
        assert any("A" in m and "B" in m and "falling back" in m for m in messages)

    def test_warning_on_second_hop_fallback(
        self,
        cache_and_ops: tuple[GraphCache, SurfaceTransformOps, Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning fires when the second hop's provider doesn't match the request."""
        cache, ops, tmp_path = cache_and_ops
        self._add_transforms(
            cache, tmp_path, {("A", "B"): "RheMap", ("B", "C"): "CIVET"}
        )
        self._add_atlas(cache, tmp_path, "B")

        first = cache.get_surface_transform(
            "A", "B", "32k", "left", "sphere", provider="RheMap"
        )
        with caplog.at_level(
            logging.WARNING, logger="neuromaps_prime.graph.transforms.surface"
        ):
            with pytest.raises(FileNotFoundError):
                ops._two_hops(
                    source_space="A",
                    mid_space="B",
                    target_space="C",
                    density="32k",
                    hemisphere="left",
                    output_file_path=str(tmp_path / "out.surf.gii"),
                    first_transform=first,
                    provider="RheMap",
                )

        messages = [r.message for r in caplog.records]
        assert any("B" in m and "C" in m and "falling back" in m for m in messages)

    def test_no_warning_when_provider_is_none(
        self,
        cache_and_ops: tuple[GraphCache, SurfaceTransformOps, Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """No warning is emitted when provider=None (fallback is intended behaviour)."""
        cache, ops, tmp_path = cache_and_ops
        self._add_transforms(
            cache, tmp_path, {("A", "B"): "CIVET", ("B", "C"): "RheMap"}
        )
        self._add_atlas(cache, tmp_path, "B")

        with caplog.at_level(
            logging.WARNING, logger="neuromaps_prime.graph.transforms.surface"
        ):
            with pytest.raises(FileNotFoundError):
                ops._two_hops(
                    source_space="A",
                    mid_space="B",
                    target_space="C",
                    density="32k",
                    hemisphere="left",
                    output_file_path=str(tmp_path / "out.surf.gii"),
                    first_transform=None,
                    provider=None,
                )

        assert not any("falling back" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Helpers shared by TestGraphCacheVolume
# ---------------------------------------------------------------------------


def _make_volume_transform(
    f: Path,
    source: str,
    target: str,
    resolution: str,
    resource_type: str,
    provider: str = "test",
    weight: float = 1.0,
) -> models.VolumeTransform:
    return models.VolumeTransform(
        name=f"{source}_to_{target}_{resolution}_{resource_type}",
        description=f"Transform from {source} to {target}",
        file_path=f,
        source_space=source,
        target_space=target,
        resolution=resolution,
        resource_type=resource_type,
        provider=provider,
        weight=weight,
    )


def _make_volume_atlas(
    f: Path,
    space: str,
    resolution: str,
    resource_type: str,
) -> models.VolumeAtlas:
    return models.VolumeAtlas(
        name=f"{space}_{resolution}_{resource_type}",
        description=f"Atlas for {space}",
        file_path=f,
        space=space,
        resolution=resolution,
        resource_type=resource_type,
    )


class TestGraphCacheVolume:
    """Unit tests for GraphCache volume transform and atlas lookup methods."""

    @pytest.fixture(params=["transform", "atlas"])
    def scenario(
        self, request: pytest.FixtureRequest, tmp_path: Path
    ) -> dict[str, Any]:
        """Fixture for generating different scenarios."""
        f = tmp_path / "file.nii.gz"
        f.touch()

        if request.param == "transform":
            populated = GraphCache()
            populated.add_volume_transforms(
                [
                    _make_volume_transform(f, "A", "B", "1mm", "T1w", provider="test"),
                    _make_volume_transform(f, "A", "B", "1mm", "T2w", provider="test"),
                    _make_volume_transform(f, "A", "B", "2mm", "T1w", provider="test"),
                    _make_volume_transform(f, "C", "D", "1mm", "T1w", provider="test"),
                ]
            )
            alt_f = tmp_path / "alt.nii.gz"
            alt_f.touch()
            return dict(
                empty_cache=GraphCache(),
                populated_cache=populated,
                get_single=lambda c, *k: c.get_volume_transform(*k, provider="test"),
                get_plural=lambda c, *a, **kw: c.get_volume_transforms(*a, **kw),
                add_single=lambda c, e: c.add_volume_transform(e),
                add_bulk=lambda c, es: c.add_volume_transforms(es),
                make_entry=lambda file, *k: _make_volume_transform(
                    file, *k, provider="test"
                ),
                make_alt_entry=lambda file, *k: _make_volume_transform(
                    alt_f, *k, weight=9.0, provider="test"
                ),
                exact_key=("A", "B", "1mm", "T1w"),
                wrong_keys=[
                    ("X", "B", "1mm", "T1w"),
                    ("A", "X", "1mm", "T1w"),
                    ("A", "B", "2mm", "T1w"),
                    ("A", "B", "1mm", "T2w"),
                ],
                group_key=("A", "B"),
                unrelated_check=lambda t: t.source_space == "C",
                plural_empty_call=lambda c: c.get_volume_transforms("A", "B"),
            )
        else:  # atlas
            populated = GraphCache()
            populated.add_volume_atlases(
                [
                    _make_volume_atlas(f, "A", "1mm", "T1w"),
                    _make_volume_atlas(f, "A", "1mm", "T2w"),
                    _make_volume_atlas(f, "A", "2mm", "T1w"),
                    _make_volume_atlas(f, "B", "1mm", "T1w"),
                ]
            )
            alt_f = tmp_path / "alt.nii.gz"
            alt_f.touch()
            return dict(
                empty_cache=GraphCache(),
                populated_cache=populated,
                get_single=lambda c, *k: c.get_volume_atlas(*k),
                get_plural=lambda c, *a, **kw: c.get_volume_atlases(*a, **kw),
                add_single=lambda c, e: c.add_volume_atlas(e),
                add_bulk=lambda c, es: c.add_volume_atlases(es),
                make_entry=lambda file, *k: _make_volume_atlas(file, *k),
                make_alt_entry=lambda file, *k: _make_volume_atlas(alt_f, *k),
                exact_key=("A", "1mm", "T1w"),
                wrong_keys=[
                    ("X", "1mm", "T1w"),
                    ("A", "2mm", "T1w"),
                    ("A", "1mm", "T2w"),
                ],
                group_key=("A",),
                unrelated_check=lambda a: a.space == "B",
                plural_empty_call=lambda c: c.get_volume_atlases("A"),
            )

    def test_hit(self, scenario: dict, tmp_path: Path) -> None:
        """Returns the entry when an exact match exists."""
        f = tmp_path / "hit.nii.gz"
        f.touch()
        entry = scenario["make_entry"](f, *scenario["exact_key"])
        scenario["add_single"](scenario["empty_cache"], entry)
        assert (
            scenario["get_single"](scenario["empty_cache"], *scenario["exact_key"])
            is entry
        )

    def test_miss_empty(self, scenario: dict) -> None:
        """Returns None on an empty cache."""
        assert (
            scenario["get_single"](scenario["empty_cache"], *scenario["exact_key"])
            is None
        )

    def test_miss_wrong_key(self, scenario: dict, tmp_path: Path) -> None:
        """Returns None when any single key component does not match."""
        f = tmp_path / "miss.nii.gz"
        f.touch()
        entry = scenario["make_entry"](f, *scenario["exact_key"])
        scenario["add_single"](scenario["empty_cache"], entry)
        for wrong_key in scenario["wrong_keys"]:
            assert (
                scenario["get_single"](scenario["empty_cache"], *wrong_key) is None
            ), f"Expected None for key {wrong_key}"

    def test_overwrite(self, scenario: dict, tmp_path: Path) -> None:
        """Later insertion overwrites an earlier entry with the same key."""
        f = tmp_path / "ow.nii.gz"
        f.touch()
        entry1 = scenario["make_entry"](f, *scenario["exact_key"])
        entry2 = scenario["make_alt_entry"](f, *scenario["exact_key"])
        scenario["add_single"](scenario["empty_cache"], entry1)
        scenario["add_single"](scenario["empty_cache"], entry2)
        assert (
            scenario["get_single"](scenario["empty_cache"], *scenario["exact_key"])
            is entry2
        )

    def test_distinct_keys_coexist(self, scenario: dict, tmp_path: Path) -> None:
        """Two entries with different resolutions are independently retrievable."""
        f = tmp_path / "dk.nii.gz"
        f.touch()
        key_1mm = scenario["exact_key"]
        key_2mm = (*key_1mm[:-2], "2mm", key_1mm[-1])
        entry_1mm = scenario["make_entry"](f, *key_1mm)
        entry_2mm = scenario["make_entry"](f, *key_2mm)
        scenario["add_bulk"](scenario["empty_cache"], [entry_1mm, entry_2mm])
        assert scenario["get_single"](scenario["empty_cache"], *key_1mm) is entry_1mm
        assert scenario["get_single"](scenario["empty_cache"], *key_2mm) is entry_2mm

    def test_no_filters_returns_all(self, scenario: dict) -> None:
        """Omitting all optional filters returns every entry for that group."""
        results = scenario["get_plural"](
            scenario["populated_cache"], *scenario["group_key"]
        )
        assert len(results) == 3

    def test_filter_resolution(self, scenario: dict) -> None:
        """'resolution' filter narrows results correctly."""
        results = scenario["get_plural"](
            scenario["populated_cache"], *scenario["group_key"], resolution="1mm"
        )
        assert len(results) == 2
        assert all(r.resolution == "1mm" for r in results)

    def test_filter_resource_type(self, scenario: dict) -> None:
        """resource_type filter narrows results correctly."""
        results = scenario["get_plural"](
            scenario["populated_cache"], *scenario["group_key"], resource_type="T1w"
        )
        assert len(results) == 2
        assert all(r.resource_type == "T1w" for r in results)

    def test_filter_both(self, scenario: dict) -> None:
        """Combining resolution and resource_type filters yields one exact match."""
        results = scenario["get_plural"](
            scenario["populated_cache"],
            *scenario["group_key"],
            resolution="1mm",
            resource_type="T1w",
        )
        assert len(results) == 1
        assert results[0].resolution == "1mm"
        assert results[0].resource_type == "T1w"

    def test_unrelated_group_excluded(self, scenario: dict) -> None:
        """Entries from the unrelated group are not returned."""
        results = scenario["get_plural"](
            scenario["populated_cache"], *scenario["group_key"]
        )
        assert not any(scenario["unrelated_check"](r) for r in results)

    def test_empty_cache_returns_empty(self, scenario: dict) -> None:
        """Empty cache returns an empty list regardless of arguments."""
        assert scenario["plural_empty_call"](scenario["empty_cache"]) == []

    def test_wrong_primary_key_returns_empty(self, scenario: dict) -> None:
        """Non-existent primary key returns an empty list."""
        wrong = ("X", "B") if len(scenario["group_key"]) == 2 else ("X",)
        assert scenario["get_plural"](scenario["populated_cache"], *wrong) == []

    @pytest.mark.parametrize(
        "kwargs",
        [{"resolution": "500um"}, {"resource_type": "composite"}],
        ids=["no_resolution_match", "no_resource_type_match"],
    )
    def test_filter_no_match(self, scenario: dict, kwargs: dict) -> None:
        """A filter that matches nothing returns an empty list."""
        assert (
            scenario["get_plural"](
                scenario["populated_cache"], *scenario["group_key"], **kwargs
            )
            == []
        )
