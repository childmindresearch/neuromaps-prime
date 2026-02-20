"""Tests associated with volume-to-volume transformation."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime.graph import NeuromapsGraph, models


class TestVolumeToVolumeTransformer:
    """Unit tests for volume to volume transformer."""

    @pytest.fixture
    def mock_transformer(self, graph: NeuromapsGraph) -> NeuromapsGraph:
        """Create mock transformer with volume_ops internals replaced.

        Replaces volume_ops.utils and volume_ops.fetchers with MagicMocks so
        internal calls are interceptable without hitting Pydantic's __setattr__
        validation.
        """
        graph.volume_ops.utils = MagicMock()
        graph.volume_ops.fetchers = MagicMock()
        return graph

    @pytest.fixture
    def mock_volume_atlas(self, tmp_path: Path) -> MagicMock:
        """Create mock volume atlas object."""
        atlas_file = tmp_path / "atlas.nii.gz"
        atlas_file.touch()
        atlas = MagicMock(spec=models.VolumeAtlas)
        atlas.fetch = MagicMock(return_value=atlas_file)
        return atlas

    @pytest.fixture
    def mock_volume_transform(self, tmp_path: Path) -> MagicMock:
        """Create mock volume transform object."""
        transform_file = tmp_path / "transform.nii.gz"
        transform_file.touch()
        transform = MagicMock(spec=models.VolumeTransform)
        transform.fetch = MagicMock(return_value=transform_file)
        return transform

    @pytest.fixture
    def basic_params(self, tmp_path: Path) -> dict[str, Any]:
        """Basic parameters for volume transformation testing."""
        input_file = tmp_path / "input.nii.gz"
        input_file.touch()
        return {
            "input_file": input_file,
            "source_space": "Yerkes19",
            "target_space": "NMT2Sym",
            "resolution": "250um",
            "resource_type": "T1w",
            "output_file_path": str(tmp_path / "output.nii.gz"),
        }

    @patch("neuromaps_prime.graph.transforms.volume.vol_to_vol")
    def test_volume_transformation_success(
        self,
        mock_vol_to_vol: MagicMock,
        mock_transformer: NeuromapsGraph,
        mock_volume_atlas: MagicMock,
        mock_volume_transform: MagicMock,
        basic_params: dict[str, Any],
    ) -> None:
        """Test successful volume transformation."""
        expected_output = Path(basic_params["output_file_path"])
        mock_transformer.volume_ops.fetchers.fetch_volume_transform.return_value = (
            mock_volume_transform
        )
        mock_transformer.volume_ops.fetchers.fetch_volume_atlas.return_value = (
            mock_volume_atlas
        )
        mock_vol_to_vol.return_value = expected_output

        result = mock_transformer.volume_to_volume_transformer(**basic_params)

        mock_transformer.volume_ops.utils.validate_spaces.assert_called_once_with(
            basic_params["source_space"], basic_params["target_space"]
        )
        mock_transformer.volume_ops.fetchers.fetch_volume_transform.assert_called_once()
        mock_transformer.volume_ops.fetchers.fetch_volume_atlas.assert_called_once()
        mock_vol_to_vol.assert_called_once()
        assert result == expected_output

    def test_no_transform(
        self, mock_transformer: NeuromapsGraph, basic_params: dict[str, Any]
    ) -> None:
        """Test error raised if no transform found."""
        mock_transformer.volume_ops.fetchers.fetch_volume_transform.return_value = None

        with pytest.raises(ValueError, match="No volume transform found"):
            mock_transformer.volume_to_volume_transformer(**basic_params)
        mock_transformer.volume_ops.fetchers.fetch_volume_transform.assert_called_once()
        mock_transformer.volume_ops.fetchers.fetch_volume_atlas.assert_not_called()

    def test_no_target_atlas(
        self,
        mock_transformer: NeuromapsGraph,
        mock_volume_transform: MagicMock,
        basic_params: dict[str, Any],
    ) -> None:
        """Test error raised if no target volume found."""
        mock_transformer.volume_ops.fetchers.fetch_volume_transform.return_value = (
            mock_volume_transform
        )
        mock_transformer.volume_ops.fetchers.fetch_volume_atlas.return_value = None

        with pytest.raises(ValueError, match="No volume atlas found"):
            mock_transformer.volume_to_volume_transformer(**basic_params)
        mock_transformer.volume_ops.fetchers.fetch_volume_atlas.assert_called_once()
