"""Tests for graph operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from neuromaps_prime.graph import models
from neuromaps_prime.graph.cache import GraphCache
from neuromaps_prime.graph.transforms.surface import SurfaceTransformOps
from tests.unit.graph.test_cache import _make_surface_transform

if TYPE_CHECKING:
    from pathlib import Path


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
        with (
            caplog.at_level(
                logging.WARNING, logger="neuromaps_prime.graph.transforms.surface"
            ),
            pytest.raises(FileNotFoundError),
        ):
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

        with (
            caplog.at_level(
                logging.WARNING, logger="neuromaps_prime.graph.transforms.surface"
            ),
            pytest.raises(FileNotFoundError),
        ):
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
        with (
            caplog.at_level(
                logging.WARNING, logger="neuromaps_prime.graph.transforms.surface"
            ),
            pytest.raises(FileNotFoundError),
        ):
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

        with (
            caplog.at_level(
                logging.WARNING, logger="neuromaps_prime.graph.transforms.surface"
            ),
            pytest.raises(FileNotFoundError),
        ):
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
