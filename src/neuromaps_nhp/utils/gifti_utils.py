"""Utility functions for working with GIFTI files and surface projections."""

from pathlib import Path

import nibabel as nib


def get_density(input_gifti: Path) -> str:
    """Get density of a gifti surface file based on number of vertices.
    
    Parameters
    ----------
    input_gifti : Path
        Path to the input GIFTI surface file.
    
    Returns:
    -------
    str
        Density string (e.g., '32k', '10k').
    """
    surface = nib.load(str(input_gifti))
    n_vertices = surface.darrays[0].data.shape[0]
    density = str(round(n_vertices / 1000)) + "k"
    return density


def get_num_vertices(input_gifti: Path) -> int:
    """Get number of vertices in a gifti surface file.
    
    Parameters
    ----------
    input_gifti : Path
        Path to the input GIFTI surface file.

    Returns:
    -------
    int
        Number of vertices in the surface file.
    """
    surface = nib.load(str(input_gifti))
    return surface.darrays[0].data.shape[0]
