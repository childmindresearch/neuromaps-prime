"""Tests for transforms utils module."""

from pathlib import Path

import pytest

from neuromaps_prime.transforms.utils import estimate_surface_density, get_vertex_count


@pytest.mark.parametrize(
    "surface_fpath,expected_density",
    [
        ("Inputs/Yerkes19/src-Yerkes19_den-32k_hemi-L_sphere.surf.gii", "32k"),
        ("Inputs/Yerkes19/src-Yerkes19_den-10k_hemi-L_sphere.surf.gii", "10k"),
        (
            "Outputs/D99-Yerkes19/src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii",
            "32k",
        ),
    ],
)
@pytest.mark.usefixtures("require_data")
def test_estimate_surface_density(
    data_dir: Path, surface_fpath: str, expected_density: str
) -> None:
    """Test estimate_surface_density function with various mesh densities."""
    data_dir = data_dir / "share"
    result = estimate_surface_density(data_dir / surface_fpath)
    assert isinstance(result, str)
    assert result == expected_density, f"Expected {expected_density}, but got {result}"


@pytest.mark.parametrize(
    "surface_fpath,expected_count",
    [
        ("Inputs/Yerkes19/src-Yerkes19_den-32k_hemi-L_sphere.surf.gii", 32492),
        ("Inputs/Yerkes19/src-Yerkes19_den-10k_hemi-L_sphere.surf.gii", 10242),
        (
            "Outputs/D99-Yerkes19/src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii",
            32492,
        ),
    ],
)
@pytest.mark.usefixtures("require_data")
def test_get_vertex_count(
    data_dir: Path, surface_fpath: str, expected_count: int
) -> None:
    """Test get_vertex_count function returns correct vertex count."""
    data_dir = data_dir / "share"
    result = get_vertex_count(data_dir / surface_fpath)
    assert isinstance(result, int)
    assert result == expected_count, f"Expected {expected_count}, but got {result}"
