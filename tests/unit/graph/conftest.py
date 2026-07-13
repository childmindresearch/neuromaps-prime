"""Shared fixtures for graph unit tests.

Provides factory fixtures that construct MagicMock instances with the
standard transform attributes (source_space, target_space, provider,
references, notes) so individual test modules don't repeat the same
boilerplate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from neuromaps_prime.graph import models

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def mock_surface_transform_factory(tmp_path: Path) -> Callable[..., MagicMock]:
    """Factory fixture that returns a MagicMock with SurfaceTransform attributes.

    All five attributes required for metadata collection are set to
    sensible defaults.  Pass keyword arguments to override.

    Examples::

        xfm = mock_surface_transform_factory(
            source_space="A", target_space="B"
        )
        xfm.fetch.return_value  -> tmp_path / "sphere.surf.gii"
    """

    def _make(
        source_space: str = "X",
        target_space: str = "Y",
        provider: str = "RheMap",
        references: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> MagicMock:
        sphere_file = tmp_path / "sphere.surf.gii"
        sphere_file.touch()
        mock = MagicMock(
            spec=models.SurfaceTransform,
            source_space=source_space,
            target_space=target_space,
            provider=provider,
            references=references,
            notes=notes,
        )
        mock.fetch.return_value = sphere_file
        return mock

    return _make


@pytest.fixture
def mock_volume_transform_factory(tmp_path: Path) -> Callable[..., MagicMock]:
    """Factory fixture that returns a MagicMock with VolumeTransform attributes.

    Examples::

        xfm = mock_volume_transform_factory(
            source_space="Yerkes19", target_space="NMT2"
        )
        xfm.fetch.return_value  -> tmp_path / "transform.nii.gz"
    """

    def _make(
        source_space: str = "X",
        target_space: str = "Y",
        provider: str = "Test",
        references: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> MagicMock:
        warp_file = tmp_path / "transform.nii.gz"
        warp_file.touch()
        mock = MagicMock(
            spec=models.VolumeTransform,
            source_space=source_space,
            target_space=target_space,
            provider=provider,
            references=references,
            notes=notes,
        )
        mock.fetch.return_value = warp_file
        return mock

    return _make
