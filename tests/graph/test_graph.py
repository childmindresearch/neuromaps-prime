"""Tests associated for broader graph functionality."""

from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import networkx as nx
import pytest

from neuromaps_prime.graph import NeuromapsGraph, models
from neuromaps_prime.graph.cache import GraphCache
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
        graph.add_transform(vf, graph.volume_to_volume_key)
        fetched = graph.fetch_volume_to_volume_transform(source, target, "1mm", "T1w")
        assert fetched is vf

    def test_add_invalid_transform(self, graph: NeuromapsGraph) -> None:
        """Test adding invalid transform type raises error."""
        with pytest.raises(TypeError, match="Unsupported transform type"):
            graph.add_transform("invalid", key="key")  # type: ignore[arg-type]

    def test_add_missing_atlas_space(self, graph: NeuromapsGraph) -> None:
        """Test adding atlas with an invalid space."""
        atlas = models.SurfaceAtlas(
            name="Test",
            description="Test",
            file_path=Path("."),
            space="invalid",
            density="10k",
            hemisphere="left",
            resource_type="test",
        )
        with pytest.raises(ValueError, match="not found"):
            graph.add_atlas(atlas)

    def test_add_invalid_atlas_type(self, graph: NeuromapsGraph) -> None:
        """Test adding invalid atlas raises error."""
        with pytest.raises(TypeError, match="Unsupported atlas type"):
            graph.add_atlas("invalid")  # type: ignore [arg-type]

    def test_add_surface_atlas(self, graph: NeuromapsGraph) -> None:
        """Test adding surface atlas."""
        atlas = models.SurfaceAtlas(
            name="Test",
            description="Test",
            file_path=Path("."),
            space="Yerkes19",
            density="10k",
            hemisphere="left",
            resource_type="test",
        )
        graph.add_atlas(atlas)
        assert atlas in graph.nodes["Yerkes19"]["data"].surfaces
        assert atlas not in graph.nodes["Yerkes19"]["data"].volumes
        assert (
            graph._cache.get_surface_atlas(
                atlas.space, atlas.density, atlas.hemisphere, atlas.resource_type
            )
            == atlas
        )

    def test_add_volume_atlas(self, graph: NeuromapsGraph) -> None:
        """Test adding volume atlas."""
        atlas = models.VolumeAtlas(
            name="Test",
            description="Test",
            file_path=Path("."),
            space="Yerkes19",
            resolution="1mm",
            resource_type="test",
        )
        graph.add_atlas(atlas)
        assert atlas not in graph.nodes["Yerkes19"]["data"].surfaces
        assert atlas in graph.nodes["Yerkes19"]["data"].volumes
        assert (
            graph._cache.get_volume_atlas(
                atlas.space, atlas.resolution, atlas.resource_type
            )
            == atlas
        )

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
        """Test cache entries are cleared cache table."""
        assert len(graph._cache.surface_atlas) > 0
        assert len(graph._cache.surface_transform) > 0
        graph._cache.clear()
        assert len(graph._cache.surface_atlas) == 0
        assert len(graph._cache.surface_transform) == 0


class TestGraphIntegration:
    """Integration tests for graph module."""

    def test_graph_initialization_with_data_dir(self, graph: NeuromapsGraph) -> None:
        """Test initializing graph with data directory."""
        assert graph is not None
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
            resource_type="T1w",
            output_file_path=str(tmp_path / "test_output.nii"),
        )
        assert output.exists()


class TestGraphCacheRequireSurface:
    """Tests for GraphCache.require_surface_atlas."""

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


# ---------------------------------------------------------------------------
# Helpers shared by TestGraphCacheVolume
# ---------------------------------------------------------------------------


def _make_volume_transform(
    f: Path,
    source: str,
    target: str,
    resolution: str,
    resource_type: str,
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
    """Unit tests for GraphCache volume transform and atlas lookup methods.

    Each test runs twice via the ``scenario`` fixture — once for
    VolumeTransform (keyed by source+target+resolution+resource_type) and
    once for VolumeAtlas (keyed by space+resolution+resource_type).
    """

    @pytest.fixture(params=["transform", "atlas"])
    def scenario(
        self, request: pytest.FixtureRequest, tmp_path: Path
    ) -> dict[str, Any]:
        """Scenario-specific helpers and caches for both volume resource types.

        Returns a dict with:
            empty_cache      - fresh GraphCache
            populated_cache  - GraphCache with 4 entries across 2 logical groups
            get_single       - fn(cache, *key) -> model | None
            get_plural       - fn(cache, *group_key, **filters) -> list
            add_single       - fn(cache, entry)
            add_bulk         - fn(cache, entries)
            make_entry       - fn(file, *key) -> model
            make_alt_entry   - fn(file, *key) -> distinct model with same key
            exact_key        - tuple for the canonical A/1mm/T1w entry
            wrong_keys       - list of tuples that should all miss
            group_key        - primary key used in plural queries
            unrelated_check  - fn(result) -> bool; True if from the unrelated group
            plural_empty_call - fn(cache) -> list; plural call on an empty cache
        """
        f = tmp_path / "file.nii.gz"
        f.touch()

        if request.param == "transform":
            populated = GraphCache()
            populated.add_volume_transforms(
                [
                    _make_volume_transform(f, "A", "B", "1mm", "T1w"),
                    _make_volume_transform(f, "A", "B", "1mm", "T2w"),
                    _make_volume_transform(f, "A", "B", "2mm", "T1w"),
                    _make_volume_transform(f, "C", "D", "1mm", "T1w"),
                ]
            )
            alt_f = tmp_path / "alt.nii.gz"
            alt_f.touch()
            return dict(
                empty_cache=GraphCache(),
                populated_cache=populated,
                get_single=lambda c, *k: c.get_volume_transform(*k),
                get_plural=lambda c, *a, **kw: c.get_volume_transforms(*a, **kw),
                add_single=lambda c, e: c.add_volume_transform(e),
                add_bulk=lambda c, es: c.add_volume_transforms(es),
                make_entry=lambda file, *k: _make_volume_transform(file, *k),
                make_alt_entry=lambda file, *k: _make_volume_transform(
                    alt_f, *k, weight=9.0
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

    # ------------------------------------------------------------------ #
    # get_volume_{transform,atlas} — singular                             #
    # ------------------------------------------------------------------ #

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
        # Swap the second-to-last element (resolution) from "1mm" to "2mm"
        key_2mm = (*key_1mm[:-2], "2mm", key_1mm[-1])
        entry_1mm = scenario["make_entry"](f, *key_1mm)
        entry_2mm = scenario["make_entry"](f, *key_2mm)
        scenario["add_bulk"](scenario["empty_cache"], [entry_1mm, entry_2mm])
        assert scenario["get_single"](scenario["empty_cache"], *key_1mm) is entry_1mm
        assert scenario["get_single"](scenario["empty_cache"], *key_2mm) is entry_2mm

    # ------------------------------------------------------------------ #
    # get_volume_{transforms,atlases} — plural                            #
    # ------------------------------------------------------------------ #

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
