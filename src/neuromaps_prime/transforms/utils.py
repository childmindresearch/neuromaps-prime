"""Utility functions for working with GIFTI files and surface projections."""

from pathlib import Path

import nibabel as nib

from neuromaps_prime.graph import NeuromapsGraph


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


def validate(
    graph: NeuromapsGraph,
    source: str,
    target: str,
) -> None:
    """Validate that source and target spaces exist in the graph."""
    if source not in graph.nodes(data=False) or target not in graph.nodes(data=False):
        raise ValueError(
            f"Either source space '{source}' or "
            f"target space '{target}' does not exist in the graph."
            f" Available spaces: {list(graph.nodes(data=False))}"
        )
