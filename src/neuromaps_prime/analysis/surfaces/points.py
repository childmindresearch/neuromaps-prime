"""Surface mesh geometry operations.

Provides tools for building mesh adjacency graphs, computing geodesic
distances between vertices or parcels, and finding parcel centroids on
cortical surfaces.  Unlike the graph module, which models brain spaces as
abstract nodes and edges, this module works directly with triangular mesh
topology to support spatial operations that have no CTF or workbench
equivalent (e.g., Dijkstra-based shortest-path distance on a surface).

Adapted from the neuromaps codebase
(https://github.com/netneurolab/neuromaps/blob/ffcc2e0f657943ce00a1b6a968396f32250e495c/neuromaps/points.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from neuromaps_prime.analysis.images import PARC_IGNORE, load_data, relabel_gifti

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from numpy.typing import ArrayLike

__all__ = ["get_surface_distance", "make_surf_graph"]


def _get_edges(faces: ArrayLike) -> np.ndarray:
    """Extract all unique edge pairs from a triangular mesh face array.

    Each triangular face contributes three edges: ``(v0, v1)``,
    ``(v1, v2)``, and ``(v2, v0)``.  Vertex indices within each edge
    are sorted so that the representation is canonical (direction-
    independent), making it straightforward to deduplicate with
    ``np.unique``.

    Args:
        faces: Array of shape ``(n_faces, 3)`` where each row contains
            vertex indices that define a triangular face of the mesh.

    Returns:
        Array of shape ``(n_faces * 3, 2)`` containing all directed edge
        pairs, sorted within each row so that the smaller vertex index
        appears first.

    Note:
        This function returns *all* edges, including duplicates from
        adjacent triangles that share an edge.  Deduplication is left to
        downstream callers.
    """
    faces = np.asarray(faces)
    return np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2)), axis=1)


def _get_directed_edges(
    vertices: ArrayLike, faces: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unique mesh edges and their Euclidean lengths.

    Extracts all edges from the triangular faces, deduplicates them,
    then computes the straight-line distance between the two vertices
    of each edge.

    Args:
        vertices: Array of shape ``(n_vertices, 3)`` containing the
            3-D coordinates of every mesh vertex.
        faces: Array of shape ``(n_faces, 3)`` where each row contains
            vertex indices that define a triangular face.

    Returns:
        A tuple ``(edges, weights)`` where:

            edges:  Array of shape ``(n_edges, 2)`` with unique,
                sorted vertex-index pairs.
            weights: 1-D array of shape ``(n_edges,)`` containing the
                Euclidean length of each edge.
    """
    faces = np.asarray(faces)
    vertices = np.asarray(vertices)
    edges = np.unique(_get_edges(faces), axis=0)
    weights = np.linalg.norm(np.diff(vertices[edges], axis=1), axis=-1).squeeze()
    return edges, weights


def _get_indirect_edges(
    vertices: ArrayLike, faces: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Compute indirect edges and approximated geodesic distances.

    Indirect edges connect pairs of opposite vertices from triangles
    that share a common edge.  The weight approximates the geodesic
    distance: each opposite vertex is orthogonally projected onto the
    shared edge, the midpoint between the two projection points is
    computed, and the sum of distances from each opposite vertex to
    that midpoint forms the edge weight.

    Args:
        vertices: Array of shape ``(n_vertices, 3)`` containing the
            3-D coordinates of every mesh vertex.
        faces: Array of shape ``(n_faces, 3)`` where each row contains
            vertex indices that define a triangular face.

    Returns:
        A tuple ``(edges, weights)`` where:

            edges: Array of shape ``(n_pairs, 2)`` with vertex-index
                pairs of opposite vertices from adjacent triangles.
            weights: 1-D array of shape ``(n_pairs,)`` containing the
                approximated geodesic distance between each pair.

    Note:
        The projection can fall outside the shared edge when either
        triangle has an obtuse angle along that edge, introducing
        small errors. Adapted from trimesh
        (https://github.com/mikedh/trimesh).
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    triangles = np.stack(list(_get_shared_triangles(faces).values()), axis=0)
    indirect_edges = triangles[..., -1]

    v0, v1, opp = vertices[triangles].transpose(2, 3, 0, 1)

    edge_vec = v0 - v1
    proj = np.sum(edge_vec * (opp - v1), axis=0, keepdims=True) / np.sum(
        edge_vec**2, axis=0, keepdims=True
    )
    feet = v1 + proj * edge_vec
    midpoints = np.sum(feet.transpose(1, 2, 0), axis=1) / 2
    norms = np.linalg.norm(vertices[indirect_edges] - midpoints[:, None], axis=-1)
    weights = np.sum(norms, axis=-1)

    return indirect_edges, weights


def _point_in_triangle(
    point: ArrayLike, triangle: ArrayLike, *, return_pdist: bool = True
) -> tuple[bool, np.float64 | None]:
    """Test whether a 3-D point lies inside a triangular face.

    Uses barycentric coordinates: expresses the point relative to vertex
    ``A`` as ``P = A + u*v0 + v*v1`` and checks ``u >= 0``, ``v >= 0``,
    and ``u + v < 1``.  See
    https://blackpawn.com/texts/pointinpoly/ for a full derivation.

    Args:
        point: Array of shape ``(3,)`` with the 3-D coordinates to test.
        triangle: Array of shape ``(3, 3)`` with the 3-D coordinates of
            the three vertices defining a triangular face.
        return_pdist: Whether to also return the perpendicular distance
            from the point to the plane of the triangle. Default ``True``.

    Returns:
        A tuple ``(inside, pdist)`` where:

            inside: Whether the point lies inside the triangle.
            pdist: The volume of the parallelepiped spanned by the two
                triangle edge vectors and the vector from a vertex to
                the point. ``None`` when *return_pdist* is ``False``.
    """
    point = np.asarray(point)
    triangle = np.asarray(triangle)
    a, b, c = triangle
    v0, v1, v2 = c - a, b - a, point - a

    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d02 = v0 @ v2
    d12 = v1 @ v2
    inv_denom = 1.0 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * inv_denom
    v = (d00 * d12 - d01 * d02) * inv_denom
    inside = u >= 0 and v >= 0 and u + v < 1

    if return_pdist:
        return inside, np.abs(v2 @ np.cross(v1, v0))
    return inside, None


def which_triangle(point: ArrayLike, triangles: ArrayLike) -> int | None:
    """Find the triangle that best contains a 3-D point.

    Iterates over candidate triangles, keeping the one with the smallest
    planar distance to *point* among those that contain it.

    Args:
        point: Array of shape ``(3,)`` with the 3-D coordinates to test.
        triangles: Array of shape ``(n_triangles, 3, 3)`` where each
            inner ``(3, 3)`` block is a set of three vertex coordinates.

    Returns:
        The index of the best-matching triangle, or ``None`` if no
        triangle contains the point.
    """
    triangles = np.asarray(triangles)
    best = None
    best_dist = np.inf
    for i, tri in enumerate(triangles):
        inside, pdist = _point_in_triangle(point, tri)
        if inside and pdist is not None and pdist < best_dist:
            best = i
            best_dist = pdist
    return best


def _get_shared_triangles(faces: ArrayLike) -> dict[tuple[int, int], np.ndarray]:
    """Build a lookup from shared edges to adjacent triangle vertex triplets.

    In a watertight triangular mesh every internal edge belongs to exactly
    two triangles.  This function finds all such shared edges and maps each
    to the vertex triplets of its two adjacent triangles.

    Algorithm:
        1. Extract every edge from the face array (3 edges per face, vertex
           indices sorted so each edge is canonical).
        2. Lexicographic sort groups duplicate edges together.
        3. Keep only groups of size 2 — edges shared by exactly two
           different triangles (boundary edges are excluded).
        4. For each shared edge, identify the vertex in each adjacent
           triangle that is NOT part of the edge (the "opposite" vertex).
        5. Return a dict mapping each shared edge to an array of shape
           ``(2, 3)`` containing the two vertex triplets.

    Args:
        faces: Array of shape ``(n_faces, 3)`` where each row contains
            vertex indices that define a triangular face.

    Returns:
        Dictionary where keys are 2-tuples ``(v0, v1)`` representing a
        shared edge (with ``v0 < v1``), and values are arrays of shape
        ``(2, 3)``. Each row of a value is a vertex triplet
        ``(v0, v1, opposite_vertex)`` where ``opposite_vertex`` is the
        third vertex in that triangle.

    Note:
        Boundary edges (belonging to only one triangle) are excluded.
        This function assumes a manifold mesh.
    """
    faces = np.asarray(faces)
    edges = _get_edges(faces)
    edge_face = np.repeat(np.arange(len(faces)), 3)

    order = np.lexsort(edges.T[::-1])
    sorted_edges = edges[order]

    group_boundary = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    group_starts = np.concatenate([[0], np.where(group_boundary)[0] + 1])
    group_lengths = np.diff(np.concatenate([group_starts, [len(sorted_edges)]]))
    starts = group_starts[group_lengths == 2]

    pair_idx = order[starts[:, None] + np.arange(2)]
    face_pairs = edge_face[pair_idx]

    valid = face_pairs[:, 0] != face_pairs[:, 1]
    face_pairs = face_pairs[valid]
    pair_idx = pair_idx[valid]
    shared_edges = edges[pair_idx[:, 0]]

    n = len(face_pairs)
    opposite = np.empty((n, 2), dtype=faces.dtype)
    for col in range(2):
        tri = faces[face_pairs[:, col]]
        in_edge = (tri == shared_edges[:, 0:1]) | (tri == shared_edges[:, 1:2])
        opp_pos = (~in_edge).argmax(axis=1)
        opposite[:, col] = tri[np.arange(n), opp_pos]

    triplets = np.empty((n, 2, 3), dtype=faces.dtype)
    triplets[:, :, :2] = shared_edges[:, None, :]
    triplets[:, :, 2] = opposite

    return dict(zip(map(tuple, shared_edges), triplets, strict=True))


def make_surf_graph(
    vertices: ArrayLike, faces: ArrayLike, mask: ArrayLike | None = None
) -> csr_matrix:
    """Build a sparse adjacency graph from a triangular surface mesh.

    Combines direct edges (shared between two vertices of a face) with
    indirect edges (connecting opposite vertices of adjacent triangles)
    into a single sparse matrix.  Edge weights are the Euclidean length
    of direct edges and the approximated geodesic distance of indirect
    edges.  The resulting graph is suitable for Dijkstra-based shortest-
    path distance computation on the surface.

    Args:
        vertices: Array of shape ``(n_vertices, 3)`` containing the
            3-D coordinates of every mesh vertex.
        faces: Array of shape ``(n_faces, 3)`` where each row contains
            vertex indices that define a triangular face.
        mask: Boolean array of shape ``(n_vertices,)``. If provided,
            edges touching any ``True`` vertex are excluded from the
            graph.

    Returns:
        A sparse CSR matrix of shape ``(n_vertices, n_vertices)`` where
        non-zero entries are the edge weights between connected vertices.

    Raises:
        ValueError: If *mask* is provided and has a different length
            than *vertices*.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    if mask is not None:
        mask = np.asarray(mask)
        if len(mask) != len(vertices):
            raise ValueError(
                "Supplied mask and vertices array have different number "
                f"of vertices ({len(mask)} != {len(vertices)})"
            )

    direct_edges, direct_weights = _get_directed_edges(vertices, faces)
    indirect_edges, indirect_weights = _get_indirect_edges(vertices, faces)
    edges = np.vstack([direct_edges, indirect_edges])
    weights = np.concatenate([direct_weights, indirect_weights])

    if mask is not None:
        (excluded,) = np.where(mask)
        keep = ~np.any(np.isin(edges, excluded), axis=1)
        edges, weights = edges[keep], weights[keep]

    return csr_matrix(
        (weights, (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    )


def _geodesic_parcel_centroid(
    vertices: ArrayLike, faces: ArrayLike, inds: ArrayLike
) -> np.ndarray:
    """Find the geodesic centroid vertex within a parcel.

    Builds a surface mesh graph restricted to the parcel's vertices,
    computes all-pairs shortest-path distances via Dijkstra, then
    returns the vertex with the minimum mean distance to all other
    vertices in the parcel.  This is the graph-theoretic *median* of
    the parcel, analogous to a 1-median facility location problem.

    The graph is constructed by masking out all vertices outside the
    parcel, which removes edges touching those vertices and leaves only
    the parcel's internal connectivity.

    Args:
        vertices: Array of shape ``(n_vertices, 3)`` containing the
            3-D coordinates of every mesh vertex.
        faces: Array of shape ``(n_faces, 3)`` where each row contains
            vertex indices that define a triangular face.
        inds: 1-D array of vertex indices belonging to the parcel.

    Returns:
        Array of shape ``(3,)`` with the 3-D coordinates of the centroid
        vertex.

    Raises:
        ValueError: If *inds* is empty.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    inds = np.asarray(inds)

    if inds.size == 0:
        raise ValueError("Parcel vertex indices (inds) are empty")

    mask = np.ones(len(vertices), dtype=bool)
    mask[inds] = False

    mat = make_surf_graph(vertices, faces, mask)
    dist_matrix = dijkstra(mat, directed=False, indices=inds)[:, inds]
    centroid_idx = int(dist_matrix.mean(axis=1).argmin())

    return vertices[inds[centroid_idx]]


def _get_graph_distance(
    vertex: int,
    graph: csr_matrix,
    labels: np.ndarray | None = None,
    unique_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Compute shortest-path distances from a single vertex.

    Runs single-source Dijkstra on the mesh adjacency graph.  If parcel
    labels are provided, distances are aggregated to the parcel level by
    computing the mean distance within each parcel (excluding the source
    vertex).

    Args:
        vertex: Source vertex index.
        graph: Sparse adjacency matrix of shape ``(n_vertices, n_vertices)``.
        labels: Optional 1-D array of parcel labels (length ``n_vertices``).
        unique_labels: Sorted unique parcel labels. Precomputed to avoid
            redundant work inside the parallel worker. Must be provided
            alongside *labels*, or both must be ``None``.

    Returns:
        1-D array of distances.  If *labels* is ``None``, shape is
        ``(n_vertices,)``.  If provided, shape is ``(n_parcels,)``.

    Raises:
        ValueError: If one of *labels* or *unique_labels* is ``None``
            and the other is not.
    """
    dist = dijkstra(graph, directed=False, indices=[vertex]).squeeze()
    if not (labels is None) == (unique_labels is None):
        raise ValueError(
            "Both 'labels' and 'unique_labels' must both be provided or None"
        )
    if labels is not None:
        dist = ndimage.mean(
            input=np.delete(dist, vertex),
            labels=np.delete(labels, vertex),
            index=unique_labels,
        )
    return dist.astype("float32")


def get_surface_distance(
    surface: str | Path,
    *,
    parcellation: str | Path | None = None,
    medial: str | Path | None = None,
    medial_labels: Iterable[str] | None = None,
    drop: Iterable[str] = PARC_IGNORE,
    n_proc: int = 1,
) -> np.ndarray:
    """Compute a geodesic distance matrix on a cortical surface.

    Loads a surface, builds a mesh adjacency graph, then runs
    single-source Dijkstra from every vertex.  If a parcellation is
    provided, vertex-level distances are aggregated to the parcel level
    by averaging within each parcel.  Medial-wall vertices and
    background parcels are excluded from the graph before computation.

    Args:
        surface: Path to a GIFTI sphere surface file.
        parcellation: Path to a GIFTI parcellation file. If provided,
            vertex-level distances are averaged within each parcel to
            produce a parcel-to-parcel distance matrix.
        medial: Path to a GIFTI medial-wall mask file. If provided,
            vertices marked as medial wall are excluded from the graph.
        medial_labels: Iterable of label names to intersect with *drop*
            to narrow which labels are excluded. When provided, only
            labels present in both *drop* and *medial_labels* are
            dropped.
        drop: Iterable of label names to exclude from the graph.
            Defaults to ``PARC_IGNORE``.
        n_proc: Number of parallel workers for Dijkstra computation.

    Returns:
        A distance matrix.  If *parcellation* is provided, returns a
        parcel-to-parcel matrix of shape ``(n_parcels - 1,``
        ``n_parcels - 1)`` (background parcel stripped).  Otherwise
        returns a vertex-to-vertex matrix of shape
        ``(n_vertices, n_vertices)``.
    """
    if medial_labels is not None:
        drop = set(drop) & set(medial_labels)

    vert, faces = load_data(surface).array
    n_vert = vert.shape[0]
    labels, mask = None, np.zeros(n_vert, dtype=bool)

    if medial is not None:
        mask = load_data(medial, dtype=bool).array
    if parcellation is not None:
        parcellation_img = relabel_gifti(parcellation, background=drop)
        labels = parcellation_img.agg_data()
        mask[labels == 0] = True

    graph = make_surf_graph(vert, faces, mask=mask)
    unique_labels = np.unique(labels) if labels is not None else None
    dist = np.vstack(
        Parallel(n_jobs=n_proc, max_bytes=None)(
            delayed(_get_graph_distance)(n, graph, labels, unique_labels)
            for n in range(n_vert)
        )
    )

    if labels is not None and unique_labels is not None:
        dist = np.vstack([dist[labels == lab].mean(axis=0) for lab in unique_labels])
        dist[np.diag_indices_from(dist)] = 0
        dist = dist[1:, 1:]

    return dist
