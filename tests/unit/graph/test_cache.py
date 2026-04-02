"""Tests for graph cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from neuromaps_prime.graph import models
from neuromaps_prime.graph.cache import GraphCache

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Literal

    from neuromaps_prime.graph.core import NeuromapsGraph


class TestGraphCacheRequireSurface:
    """Tests for GraphCache require_* methods."""

    def test_require_surface_atlas_hit(self, graph: NeuromapsGraph) -> None:
        """Test require_surface_atlas returns the atlas when found."""
        atlases = graph._cache.get_surface_atlases(space="Yerkes19")
        assert atlases
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
        assert atlases
        a = atlases[0]
        result = graph._cache.require_volume_atlas(
            space=a.space, resolution=a.resolution, resource_type=a.resource_type
        )
        assert result is a

    def test_require_volume_atlas_miss(self, graph: NeuromapsGraph) -> None:
        """Test require_volume_atlas raises ValueError when not found."""
        with pytest.raises(ValueError, match="No 'T1w' volume atlas found"):
            graph._cache.require_volume_atlas(
                space="nonexistent", resolution="250um", resource_type="T1w"
            )


# ---------------------------------------------------------------------------
# Helpers shared by TestGraphCacheSurface
# ---------------------------------------------------------------------------
def _make_surface_transform(
    f: Path,
    source: str,
    target: str,
    density: str,
    hemisphere: Literal["left", "right"],
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
    hemisphere: Literal["left", "right"],
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
    f: Path, space: str, label: str, resolution: str
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

    def test_provider_miss_with_no_entries_returns_none(self) -> None:
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
    f: Path, space: str, resolution: str, resource_type: str
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
            return {
                "empty_cache": GraphCache(),
                "populated_cache": populated,
                "get_single": lambda c, *k: c.get_volume_transform(*k, provider="test"),
                "get_plural": lambda c, *a, **kw: c.get_volume_transforms(*a, **kw),
                "add_single": lambda c, e: c.add_volume_transform(e),
                "add_bulk": lambda c, es: c.add_volume_transforms(es),
                "make_entry": lambda file, source, target, resolution, resource_type: (
                    _make_volume_transform(
                        file, source, target, resolution, resource_type, provider="test"
                    )
                ),
                "make_alt_entry": lambda _, source, target, resolution, resource_type: (
                    _make_volume_transform(
                        alt_f,
                        source,
                        target,
                        resolution,
                        resource_type,
                        weight=9.0,
                        provider="test",
                    )
                ),
                "exact_key": ("A", "B", "1mm", "T1w"),
                "wrong_keys": [
                    ("X", "B", "1mm", "T1w"),
                    ("A", "X", "1mm", "T1w"),
                    ("A", "B", "2mm", "T1w"),
                    ("A", "B", "1mm", "T2w"),
                ],
                "group_key": ("A", "B"),
                "unrelated_check": lambda t: t.source_space == "C",
                "plural_empty_call": lambda c: c.get_volume_transforms("A", "B"),
            }
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
        return {
            "empty_cache": GraphCache(),
            "populated_cache": populated,
            "get_single": lambda c, *k: c.get_volume_atlas(*k),
            "get_plural": lambda c, *a, **kw: c.get_volume_atlases(*a, **kw),
            "add_single": lambda c, e: c.add_volume_atlas(e),
            "add_bulk": lambda c, es: c.add_volume_atlases(es),
            "make_entry": lambda file, space, resolution, resource_type: (
                _make_volume_atlas(file, space, resolution, resource_type)
            ),
            "make_alt_entry": lambda _, space, resolution, resource_type: (
                _make_volume_atlas(alt_f, space, resolution, resource_type)
            ),
            "exact_key": ("A", "1mm", "T1w"),
            "wrong_keys": [
                ("X", "1mm", "T1w"),
                ("A", "2mm", "T1w"),
                ("A", "1mm", "T2w"),
            ],
            "group_key": ("A",),
            "unrelated_check": lambda a: a.space == "B",
            "plural_empty_call": lambda c: c.get_volume_atlases("A"),
        }

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
