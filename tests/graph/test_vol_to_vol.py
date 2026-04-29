"""Tests associated with volume-to-volume transformation."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime.graph import NeuromapsGraph, models


class BasicParams(NamedTuple):
    """Basic parameters typed object."""

    input_file: Path
    source_space: str
    target_space: str
    resolution: str
    resource_type: str
    output_file_path: str


class TestVolumeToVolumeTransformer:
    """Unit tests for volume to volume transformer."""

    @pytest.fixture
    def mock_transformer(self, graph: NeuromapsGraph) -> NeuromapsGraph:
        """Create mock transformer with volume_ops internals replaced.

        Replaces volume_ops.utils and volume_ops.cache with MagicMocks so
        internal calls are interceptable without hitting Pydantic's __setattr__
        validation.
        """
        graph.volume_ops.utils = MagicMock()
        graph.volume_ops.cache = MagicMock()
        return graph

    @pytest.fixture
    def mock_volume_atlas(self, tmp_path: Path) -> MagicMock:
        """Create mock volume atlas object."""
        atlas_file = tmp_path / "atlas.nii.gz"
        atlas_file.touch()
        return MagicMock(
            spec=models.VolumeAtlas, fetch=MagicMock(return_value=atlas_file)
        )

    @pytest.fixture
    def mock_volume_transform(self, tmp_path: Path) -> MagicMock:
        """Create mock volume transform object."""
        transform_file = tmp_path / "transform.nii.gz"
        transform_file.touch()
        return MagicMock(
            spec=models.VolumeTransform, fetch=MagicMock(return_value=transform_file)
        )

    @pytest.fixture
    def basic_params(self, tmp_path: Path) -> BasicParams:
        """Basic parameters for volume transformation testing."""
        input_file = tmp_path / "input.nii.gz"
        input_file.touch()
        return BasicParams(
            input_file=input_file,
            source_space="Yerkes19",
            target_space="NMT2Sym",
            resolution="250um",
            resource_type="T1w",
            output_file_path=str(tmp_path / "output.nii.gz"),
        )

    def test_volume_transformation_success(
        self,
        mock_transformer: NeuromapsGraph,
        mock_volume_atlas: MagicMock,
        mock_volume_transform: MagicMock,
        basic_params: BasicParams,
    ) -> None:
        """Test successful volume transformation."""
        expected_output = Path(basic_params.output_file_path)
        mock_transformer.volume_ops.cache.get_volume_transform.return_value = (
            mock_volume_transform
        )
        mock_transformer.volume_ops.cache.get_volume_atlas.return_value = (
            mock_volume_atlas
        )

        with (
            patch(
                "neuromaps_prime.graph.transforms.volume.vol_to_vol",
                return_value=expected_output,
            ) as mock_vol_to_vol,
        ):
            result = mock_transformer.volume_to_volume_transformer(
                **basic_params._asdict()
            )
        mock_transformer.volume_ops.utils.validate_spaces.assert_called_once_with(
            basic_params.source_space, basic_params.target_space
        )
        mock_transformer.volume_ops.cache.get_volume_transform.assert_called_once()
        mock_transformer.volume_ops.cache.get_volume_atlas.assert_called_once()
        mock_vol_to_vol.assert_called_once()
        assert result == expected_output

    def test_no_transform(
        self, mock_transformer: NeuromapsGraph, basic_params: BasicParams
    ) -> None:
        """Test error raised if no transform found."""
        mock_transformer.volume_ops.cache.get_volume_transform.return_value = None
        with pytest.raises(ValueError, match="No volume transform found"):
            mock_transformer.volume_to_volume_transformer(**basic_params._asdict())
        mock_transformer.volume_ops.cache.get_volume_transform.assert_called_once()
        mock_transformer.volume_ops.cache.get_volume_atlas.assert_not_called()

    def test_no_target_atlas(
        self,
        mock_transformer: NeuromapsGraph,
        mock_volume_transform: MagicMock,
        basic_params: BasicParams,
    ) -> None:
        """Test error raised if no target volume found."""
        mock_transformer.volume_ops.cache.get_volume_transform.return_value = (
            mock_volume_transform
        )
        mock_transformer.volume_ops.cache.get_volume_atlas.return_value = None

        with pytest.raises(ValueError, match="No volume atlas found"):
            mock_transformer.volume_to_volume_transformer(**basic_params._asdict())
        mock_transformer.volume_ops.cache.get_volume_atlas.assert_called_once()