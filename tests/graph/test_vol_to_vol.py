"""Tests associated with volume-to-volume transformation."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime import models
from neuromaps_prime.graph import NeuromapsGraph


class TestVolumeToVolumeTransformer:
    """Unit tests for volume to volume transformer."""

    @pytest.fixture
    def mock_transformer(self, graph: NeuromapsGraph) -> NeuromapsGraph:
        """Create mock transformer with necessary methods."""
        graph.validate = MagicMock()
        graph.fetch_volume_to_volume_transform = MagicMock()
        graph.fetch_volume_atlas = MagicMock()
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

    @patch("neuromaps_prime.transforms.volume.vol_to_vol")
    def test_volume_transformation_success(
        self,
        mock_vol_to_vol: MagicMock,
        mock_transformer: NeuromapsGraph,
        mock_volume_atlas: MagicMock,
        mock_volume_transform: MagicMock,
        basic_params: dict[str, Any],
    ) -> None:
        """Test successful volume transformation."""
        # Setup
        mock_transformer.fetch_volume_to_volume_transform.return_value = (
            mock_volume_transform
        )
        mock_transformer.fetch_volume_atlas.return_value = mock_volume_atlas
        mock_vol_to_vol.return_value = Path(basic_params["output_file_path"])

        with patch("niwrap.ants.ants_apply_transforms") as mock_ants:
            mock_output = MagicMock()
            mock_output.output = MagicMock()
            mock_output.output.output_image_outfile = Path(
                basic_params["output_file_path"]
            )
            mock_output.output.output_image_outfile.touch()
            mock_ants.return_value = mock_output
            result = mock_transformer.volume_to_volume_transformer(**basic_params)
        mock_transformer.validate.assert_called_once()
        mock_transformer.fetch_volume_to_volume_transform.assert_called_once()
        mock_transformer.fetch_volume_atlas.assert_called_once()
        mock_ants.assert_called_once()
        assert result.exists()

    def test_no_input_file(
        self, mock_transformer: NeuromapsGraph, basic_params: dict[str, Any]
    ) -> None:
        """Test FileNotFoundError raised when no input found."""
        basic_params["input_file"] = Path("invalid.nii.gz")
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            mock_transformer.volume_to_volume_transformer(**basic_params)
        mock_transformer.fetch_volume_to_volume_transform.assert_not_called()

    def test_no_transform(
        self, mock_transformer: NeuromapsGraph, basic_params: dict[str, Any]
    ) -> None:
        """Test error raised if no transform found."""
        mock_transformer.fetch_volume_to_volume_transform.return_value = None

        with pytest.raises(ValueError, match="No volume transform found"):
            mock_transformer.volume_to_volume_transformer(**basic_params)
        mock_transformer.fetch_volume_to_volume_transform.assert_called_once()
        mock_transformer.fetch_volume_atlas.assert_not_called()

    def test_no_target_atlas(
        self,
        mock_transformer: NeuromapsGraph,
        mock_volume_transform: MagicMock,
        basic_params: dict[str, Any],
    ) -> None:
        """Test error raised if no target volume found."""
        mock_transformer.fetch_volume_to_volume_transform.return_value = (
            mock_volume_transform
        )
        mock_transformer.fetch_volume_atlas.return_value = None

        with pytest.raises(ValueError, match="No target volume atlas found"):
            mock_transformer.volume_to_volume_transformer(**basic_params)
        mock_transformer.fetch_volume_atlas.assert_called_once()
