"""Tests for surface transformation operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime.graph import models
from neuromaps_prime.graph.cache import GraphCache
from neuromaps_prime.graph.transforms.surface import SurfaceTransformOps
from neuromaps_prime.graph.utils import GraphUtils

if TYPE_CHECKING:
    from pathlib import Path

_LOGGER = "neuromaps_prime.graph.transforms.surface"


class TestTwoHops:
    """Unit tests for SurfaceTransformOps._two_hops with mocked outputs."""

    @pytest.fixture
    def ops(self) -> SurfaceTransformOps:
        """Create SurfaceTransformOps with mocked cache and utils."""
        cache = MagicMock(spec=GraphCache)
        utils = MagicMock(spec=GraphUtils)
        return SurfaceTransformOps(cache=cache, utils=utils)

    @pytest.fixture
    def mock_transforms(self, tmp_path: Path) -> dict[str, MagicMock]:
        """Create mock transforms with fetchable paths."""
        sphere_in = tmp_path / "sphere_in.surf.gii"
        sphere_project_to = tmp_path / "sphere_project_to.surf.gii"
        sphere_unproject_from = tmp_path / "sphere_unproject_from.surf.gii"
        for f in (sphere_in, sphere_project_to, sphere_unproject_from):
            f.touch()

        first = MagicMock(
            spec=models.SurfaceTransform,
            fetch=MagicMock(return_value=sphere_project_to),
            provider="RheMap",
        )
        mid_atlas = MagicMock(
            spec=models.SurfaceAtlas,
            fetch=MagicMock(return_value=sphere_project_to),
        )
        second = MagicMock(
            spec=models.SurfaceTransform,
            fetch=MagicMock(return_value=sphere_unproject_from),
            provider="RheMap",
        )
        return {"first": first, "mid_atlas": mid_atlas, "second": second}

    def test_two_hops_success(
        self,
        ops: SurfaceTransformOps,
        mock_transforms: dict[str, MagicMock],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test successful two-hop transform returns correct output with no warnings."""
        expected_out = tmp_path / "out.surf.gii"
        expected_out.touch()

        ops.cache.get_surface_transform.side_effect = [mock_transforms["second"]]
        ops.cache.get_surface_atlas.return_value = mock_transforms["mid_atlas"]
        ops.utils.find_common_density.return_value = "32k"

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER),
            patch(
                "neuromaps_prime.graph.transforms.surface.surface_sphere_project_unproject",
                return_value=MagicMock(sphere_out=expected_out),
            ),
        ):
            result = ops._two_hops(
                source_space="A",
                mid_space="B",
                target_space="C",
                density="32k",
                hemisphere="left",
                output_file_path=str(expected_out),
                first_transform=mock_transforms["first"],
                provider="RheMap",
            )

        assert result == expected_out
        assert not any("falling back" in r.message for r in caplog.records)

    def test_missing_first_transform_raises(
        self,
        ops: SurfaceTransformOps,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test ValueError raised when first transform not found."""
        ops.cache.get_surface_transform.return_value = None

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER),
            pytest.raises(
                ValueError, match="No surface transform found from 'A' to 'B'"
            ),
        ):
            ops._two_hops(
                source_space="A",
                mid_space="B",
                target_space="C",
                density="32k",
                hemisphere="left",
                output_file_path=str(tmp_path / "out.surf.gii"),
            )

        assert not any("falling back" in r.message for r in caplog.records)

    def test_missing_mid_atlas_raises(
        self,
        ops: SurfaceTransformOps,
        mock_transforms: dict[str, MagicMock],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test ValueError raised when mid-space atlas not found."""
        ops.cache.get_surface_transform.return_value = mock_transforms["second"]
        ops.cache.get_surface_atlas.return_value = None
        ops.utils.find_common_density.return_value = "32k"

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER),
            pytest.raises(ValueError, match="No sphere atlas found for 'B'"),
        ):
            ops._two_hops(
                source_space="A",
                mid_space="B",
                target_space="C",
                density="32k",
                hemisphere="left",
                output_file_path=str(tmp_path / "out.surf.gii"),
                first_transform=mock_transforms["first"],
            )

        assert not any("falling back" in r.message for r in caplog.records)

    def test_missing_second_transform_raises(
        self,
        ops: SurfaceTransformOps,
        mock_transforms: dict[str, MagicMock],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test ValueError raised when second transform not found."""
        ops.cache.get_surface_transform.return_value = None
        ops.cache.get_surface_atlas.return_value = mock_transforms["mid_atlas"]
        ops.utils.find_common_density.return_value = "32k"

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER),
            pytest.raises(
                ValueError, match="No surface transform found from 'B' to 'C'"
            ),
        ):
            ops._two_hops(
                source_space="A",
                mid_space="B",
                target_space="C",
                density="32k",
                hemisphere="left",
                output_file_path=str(tmp_path / "out.surf.gii"),
                first_transform=mock_transforms["first"],
            )

        assert not any("falling back" in r.message for r in caplog.records)

    def test_warning_on_first_hop_provider_mismatch(
        self,
        ops: SurfaceTransformOps,
        mock_transforms: dict[str, MagicMock],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning emitted when first hop uses a different provider than requested."""
        expected_out = tmp_path / "out.surf.gii"
        first = MagicMock(spec=models.SurfaceTransform, provider="CIVET")
        first.fetch.return_value = tmp_path / "sphere_in.surf.gii"

        ops.cache.get_surface_transform.return_value = mock_transforms["second"]
        ops.cache.get_surface_atlas.return_value = mock_transforms["mid_atlas"]
        ops.utils.find_common_density.return_value = "32k"

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER),
            patch(
                "neuromaps_prime.graph.transforms.surface.surface_sphere_project_unproject",
                return_value=MagicMock(sphere_out=expected_out),
            ),
        ):
            ops._two_hops(
                source_space="A",
                mid_space="B",
                target_space="C",
                density="32k",
                hemisphere="left",
                output_file_path=str(expected_out),
                first_transform=first,
                provider="RheMap",
            )

        messages = [r.message for r in caplog.records]
        assert any(
            "'RheMap'" in m
            and "'A'" in m
            and "'B'" in m
            and "falling back" in m
            and "'CIVET'" in m
            for m in messages
        )

    def test_warning_on_second_hop_provider_mismatch(
        self,
        ops: SurfaceTransformOps,
        mock_transforms: dict[str, MagicMock],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning emitted when second hop uses a different provider than requested."""
        expected_out = tmp_path / "out.surf.gii"
        second = MagicMock(spec=models.SurfaceTransform, provider="CIVET")
        second.fetch.return_value = tmp_path / "sphere_unproject_from.surf.gii"

        ops.cache.get_surface_transform.return_value = second
        ops.cache.get_surface_atlas.return_value = mock_transforms["mid_atlas"]
        ops.utils.find_common_density.return_value = "32k"

        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER),
            patch(
                "neuromaps_prime.graph.transforms.surface.surface_sphere_project_unproject",
                return_value=MagicMock(sphere_out=expected_out),
            ),
        ):
            ops._two_hops(
                source_space="A",
                mid_space="B",
                target_space="C",
                density="32k",
                hemisphere="left",
                output_file_path=str(expected_out),
                first_transform=mock_transforms["first"],
                provider="RheMap",
            )

        messages = [r.message for r in caplog.records]
        assert any(
            "'RheMap'" in m
            and "'B'" in m
            and "'C'" in m
            and "falling back" in m
            and "'CIVET'" in m
            for m in messages
        )
