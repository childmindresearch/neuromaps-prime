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
    if source not in graph.nodes(data=False):
        raise ValueError(
            f"source space '{source}' does not exist in the graph."
            f" Available spaces: {list(graph.nodes(data=False))}"
        )
    elif target not in graph.nodes(data=False):
        raise ValueError(
            f"target space '{target}' does not exist in the graph."
            f" Available spaces: {list(graph.nodes(data=False))}"
        )


def _get_density_key(d: str) -> int:
    """Key function to sort density strings like '32k' numerically."""
    return int(d.rstrip("kK")) if d.lower().endswith("k") else int(d)


def find_common_density(
    graph: NeuromapsGraph, mid_space: str, target_space: str
) -> str:
    """Find a common density between source and target spaces."""
    atlases = graph.search_surface_atlases(space=mid_space)
    print(f"Resources found for transformation: {atlases}")

    transforms = graph.search_surface_transforms(
        source_space=mid_space, target_space=target_space
    )
    print(f"Transforms found for transformation: {transforms}")

    # Find highest common density between list of atlases and transforms
    atlas_densities = {atlas.density for atlas in atlases}
    transform_densities = {transform.density for transform in transforms}
    common_densities = atlas_densities & transform_densities

    if common_densities:
        # If densities are strings like "32k", sort numerically
        highest_common_density = max(common_densities, key=_get_density_key)
        print(f"Highest common density: {highest_common_density}")
    else:
        print("No common density found between atlases and transforms.")

    return highest_common_density


def find_highest_density(graph: NeuromapsGraph, space: str) -> str:
    """Find the highest density available for a given space in the graph."""
    atlases = graph.search_surface_atlases(space=space)
    densities = {atlas.density for atlas in atlases}

    if not densities:
        raise ValueError(f"No atlases found for space '{space}'.")

    highest_density = max(densities, key=_get_density_key)
    print(f"Highest density for space '{space}': {highest_density}")
    return highest_density
