"""Utility functions for working with GIFTI files and surface projections."""

from functools import lru_cache
from pathlib import Path

import nibabel as nib


def estimate_surface_density(surface_file: Path) -> str:
    """Return density string of GIFTI surface based on vertex count.

    Args:
    surface_file: Path to the input GIFTI surface file.

    Returns:
        Density string (e.g., '32k', '10k').
    """
    count = get_vertex_count(surface_file)
    return f"{round(count / 1000)}k"


@lru_cache
def get_vertex_count(surface_file: Path) -> int:
    """Get number of vertices in a GIFTI surface file.

    Args:
        surface_file: Path to the input GIFTI surface file.

    Returns:
        Number of vertices in the surface file.
    """
    surface = nib.load(surface_file)
    if not isinstance(surface, nib.GiftiImage):
        raise TypeError(f"Input file is not a GIFTI surface file: {surface_file}.")
    return surface.darrays[0].data.shape[0]


def _get_density_key(density: str) -> int:
    """Sort density strings like '32k' numerically.

    Args:
        density: String density key.

    Returns:
        Approximate integer density value.
    """
    density = density.strip().lower()
    if density.endswith("k"):
        return int(density.rstrip("k")) * 1000
    return int(density)


def validate_volume_file(file_path: str | Path) -> bool:
    """Validate that the file passed exists and is a volume.

    A crude check based on file extension - checks the file ends with .nii or .nii.gz

    Args:
        filepath: Path to volume file

    Returns:
        Boolean indicating the file exists and is a volume.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{file_path} does not exist.")

    if not str(file_path).endswith((".nii", ".nii.gz")):
        raise ValueError("Expected volume nifti.")

    return True


def validate_surface_file(file_path: str | Path) -> bool:
    """Validate that the file passed exists and is a surface.

    A crude check based on file extension - checks the file ends with .gii

    Args:
        filepath: Path to surface file

    Returns:
        Boolean indicating the file exists and is a surface.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{file_path} does not exist.")

    if not str(file_path).endswith(".gii"):
        raise ValueError("Expected surface file.")
    return True
