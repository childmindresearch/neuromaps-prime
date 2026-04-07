"""Tests for transforms utils module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import pytest

from neuromaps_prime.transforms import utils

VERTEX_COUNT = 32_492


@pytest.fixture
def mock_gifti() -> MagicMock:
    mock_darray = MagicMock()
    mock_darray.data.shape = (VERTEX_COUNT,)
    mock_img = MagicMock(spec=nib.GiftiImage)
    mock_img.darrays = [mock_darray]
    return mock_img


@patch("nibabel.load")
def test_get_vertex_count_valid(mock_load: MagicMock, mock_gifti: MagicMock):
    mock_load.return_value = mock_gifti
    count = utils.get_vertex_count(Path("test.surf.gii"))
    assert isinstance(count, int) and count == VERTEX_COUNT


@patch("nibabel.load")
def test_get_vertex_count_invalid(mock_load: MagicMock):
    mock_load.return_value = MagicMock()
    with pytest.raises(TypeError):
        utils.get_vertex_count("test.nii.gz")


@patch("neuromaps_prime.transforms.utils.get_vertex_count")
def test_estimate_surface_density(mock_count: MagicMock):
    mock_count.return_value = VERTEX_COUNT
    density = utils.estimate_surface_density(Path("test.surf.gii"))
    assert isinstance(density, str) and density == "32k"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("32k", 32000),
        (" 10k ", 10000),
        ("128", 128),
    ],
)
def test_get_density_key(value: str, expected: int) -> None:
    assert utils._get_density_key(value) == expected


class TestValidateVolume:
    """Testing suite for volume validation."""

    @pytest.mark.parametrize("ext", [".nii", ".nii.gz"])
    def test_valid_volume(self, tmp_path: Path, ext: str) -> None:
        """Test able to validate volume files."""
        file_path = tmp_path / f"test{ext}"
        file_path.touch()
        res = utils.validate_volume_file(file_path)
        assert res is True

    def test_file_not_exist(self) -> None:
        """Raise FileNotFoundError if file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            utils.validate_volume_file("invalid_file.nii")

    def test_incorrect_ext(self, tmp_path: Path) -> None:
        """Raise ValueError if file has wrong extension."""
        file_path = tmp_path / "file.invalid_ext"
        file_path.touch()
        with pytest.raises(ValueError, match="Expected volume nifti"):
            utils.validate_volume_file(file_path)


class TestValidateSurface:
    """Testing suite for surface validation."""

    def test_valid_volume(self, tmp_path: Path) -> None:
        """Test able to validate volume files."""
        file_path = tmp_path / "test.surf.gii"
        file_path.touch()
        res = utils.validate_surface_file(file_path)
        assert res is True

    def test_file_not_exist(self) -> None:
        """Raise FileNotFoundError if file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            utils.validate_surface_file("invalid_file.surf.gii")

    def test_incorrect_ext(self, tmp_path: Path) -> None:
        """Raise ValueError if file has wrong extension."""
        file_path = tmp_path / "file.invalid_ext"
        file_path.touch()
        with pytest.raises(ValueError, match="Expected surface"):
            utils.validate_surface_file(file_path)
