"""Utility functions for working with GIFTI files and surface projections."""

from pathlib import Path

import nibabel as nib


def estimate_surface_density(surface_file: Path) -> str:
    """Estimate density of a gifti surface file based on number of vertices.

    Parameters
    ----------
    surface_file : Path
        Path to the input GIFTI surface file.

    Returns:
    -------
    str
        Density string (e.g., '32k', '10k').
    """
    return str(round(get_vertex_count(surface_file) / 1000)) + "k"


def get_vertex_count(surface_file: Path) -> int:
    """Get number of vertices in a gifti surface file.

    Parameters
    ----------
    surface_file : Path
        Path to the input GIFTI surface file.

    Returns:
    -------
    int
        Number of vertices in the surface file.
    """
    surface = nib.load(surface_file)
    if not isinstance(surface, nib.GiftiImage):
        raise TypeError("Input file is not a GIFTI surface file.")
    return surface.darrays[0].data.shape[0]


def _get_density_key(d: str) -> int:
    """Key function to sort density strings like '32k' numerically."""
    return int(d.rstrip("kK")) if d.lower().endswith("k") else int(d)
