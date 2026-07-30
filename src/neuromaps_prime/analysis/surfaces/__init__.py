"""Surface-specific operations for cortical mesh analysis.

Contains tools for vertex-level geometry (mesh adjacency graphs, geodesic
distance computation, parcel centroids) and surface-based null model
generation (spin permutations, surrogate creation). Complements the
space-agnostic statistics in :mod:`neuromaps_prime.analysis.stats` with
operations that require surface topology.
"""

from neuromaps_prime.analysis.surfaces.points import (
    get_surface_distance,
    make_surf_graph,
)

__all__ = ["get_surface_distance", "make_surf_graph"]
