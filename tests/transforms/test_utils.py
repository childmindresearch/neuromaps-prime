"""Tests for transforms utils module."""

from pathlib import Path

import pytest

from neuromaps_prime.transforms.utils import (
    estimate_surface_density,
    get_vertex_count,
)

DATA_DIR = Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share")


@pytest.mark.parametrize(
    "surface_file,expected_density",
    [
        (
            DATA_DIR / "Inputs/Yerkes19/src-Yerkes19_den-32k_hemi-L_sphere.surf.gii",
            "32k",
        ),
        (
            DATA_DIR / "Inputs/Yerkes19/src-Yerkes19_den-10k_hemi-L_sphere.surf.gii",
            "10k",
        ),
        (
            DATA_DIR / "Outputs/D99-Yerkes19/"
            "src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii",
            "32k",
        ),
    ],
)
def test_estimate_surface_density(surface_file: Path, expected_density: str) -> None:
    """Test estimate_surface_density function with various mesh densities."""
    result = estimate_surface_density(surface_file)
    assert isinstance(result, str)
    assert result == expected_density, f"Expected {expected_density}, but got {result}"


@pytest.mark.parametrize(
    "surface_file,expected_count",
    [
        (
            DATA_DIR / "Inputs/Yerkes19/src-Yerkes19_den-32k_hemi-L_sphere.surf.gii",
            32492,
        ),
        (
            DATA_DIR / "Inputs/Yerkes19/src-Yerkes19_den-10k_hemi-L_sphere.surf.gii",
            10242,
        ),
        (
            DATA_DIR / "Outputs/D99-Yerkes19/"
            "src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii",
            32492,
        ),
    ],
)
def test_get_vertex_count(surface_file: Path, expected_count: int) -> None:
    """Test get_vertex_count function returns correct vertex count."""
    result = get_vertex_count(surface_file)
    assert isinstance(result, int)
    assert result == expected_count, f"Expected {expected_count}, but got {result}"
