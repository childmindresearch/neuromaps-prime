"""Spatial rotation (spin) permutation utilities.

Provides utilities for the spin permutation method: loading precomputed
rotation matrices, computing parcel centroids from mesh geometry,
generating random sphere rotations, building spin samples via various
assignment methods, rotating parcellation boundaries, and projecting
data between vertex and parcel levels.

Spins preserve the large-scale spatial autocorrelation structure of
cortical maps by rotating region boundaries across the sphere rather
than shuffling values.

Adapted from the neuromaps codebase
(https://github.com/netneurolab/neuromaps/blob/ffcc2e0f657943ce00a1b6a968396f32250e495c/neuromaps/nulls/spins.py).
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, spatial
from scipy.ndimage import labeled_comprehension

from neuromaps_prime.analysis.images import PARC_IGNORE, load_gifti
from neuromaps_prime.analysis.surfaces.points import _geodesic_parcel_centroid

__all__ = [
    "gen_spin_samples",
    "get_parcel_centroids",
    "load_spins",
    "parcels_to_vertices",
    "spin_data",
    "spin_parcels",
    "vertices_to_parcels",
]


_logger = logging.getLogger(__name__)


def load_spins(fn: str | Path | ArrayLike, n_perm: int | None = None) -> np.ndarray:
    """Load a spin permutation matrix from a file or array-like object.

    Reads precomputed spin permutations from a NumPy ``.npy`` file,
    comma-separated file, or array-like object.  Each column is a
    permutation of vertex indices.

    Args:
        fn: Path to a ``.npy`` or comma-separated file, or an array-like
            object.  For paths without the ``.npy`` suffix, the function
            checks for a corresponding ``.npy`` file first, then falls
            back to the text file.
        n_perm: If provided, truncate to the first ``n_perm``
            permutations along the last axis.

    Returns:
        Spin permutation array of shape ``(n, P)``.

    Raises:
        FileNotFoundError: If ``fn`` is a path and neither ``fn`` nor
            ``fn.with_suffix('.npy')`` exists.
    """
    if not isinstance(fn, str | Path):
        spins = np.asarray(fn, dtype=np.int32)
    else:
        fn = Path(fn)
        npy_path = fn.with_suffix(".npy")

        if npy_path.exists():
            spins = np.load(npy_path, allow_pickle=False, mmap_mode="c")
        elif fn.exists():
            spins = np.loadtxt(fn, delimiter=",", dtype=np.int32)
        else:
            raise FileNotFoundError(f"Neither {fn} nor {npy_path} was found.")

    if n_perm is not None:
        spins = spins[..., :n_perm]
    return spins


def _get_parcel_centroids(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    *,
    method: Literal["average", "surface", "geodesic"] = "surface",
    drop: Iterable[str] | None = None,
    labeltable: dict[int, str] | None = None,
) -> np.ndarray:
    """Compute parcel centroids from pre-loaded arrays.

    Core logic for :func:`get_parcel_centroids`.
    """
    if method not in (methods := frozenset(["average", "surface", "geodesic"])):
        raise ValueError(f"Expected one of {methods} to be provided, got {method}")

    drop_set = set(drop) if drop is not None else set()
    if not drop_set:
        labeltable = None

    centroids = []
    for lab in np.unique(labels):
        if labeltable is not None and labeltable.get(lab) in drop_set:
            continue

        mask = labels == lab
        if method in ("average", "surface"):
            roi = vertices[mask].mean(axis=0)
            if method == "surface":
                idx = np.linalg.norm(vertices - roi, axis=1).argmin()
                roi = vertices[idx]
        elif method == "geodesic":
            inds = np.nonzero(mask)[0]
            roi = _geodesic_parcel_centroid(vertices=vertices, faces=faces, inds=inds)

        centroids.append(roi)

    return np.vstack(centroids)


def get_parcel_centroids(
    surface: str | Path,
    *,
    parcellation: str | Path | None = None,
    method: Literal["average", "surface", "geodesic"] = "surface",
    drop: Iterable[str] = PARC_IGNORE,
) -> np.ndarray:
    """Return vertex coordinates for each parcel centroid in a surface.

    Computes centroid coordinates for every parcel defined in a
    parcellation.  If *parcellation* is ``None``, returns all vertex
    coordinates.

    Args:
        surface: Path to a GIFTI surface file.
        parcellation: Path to a GIFTI label file defining parcels.
            If ``None``, returns all vertex coordinates.
        method: How to compute parcel centroids.  ``'average'`` uses
            the arithmetic mean of vertex coordinates.  ``'surface'``
            projects the mean to the nearest surface vertex.  ``'geodesic'``
            finds the vertex with the minimum average geodesic distance
            to all other vertices in the parcel (slower).
            Default is ``'surface'``.
        drop: Iterable of label names to skip.  Defaults to
            :data:`~neuromaps_prime.analysis.surfaces.points.PARC_IGNORE`
            which excludes unknown and medial-wall regions.

    Returns:
        Array of shape ``(n, 3)``.  If *parcellation* was provided,
        ``n`` is the number of parcels (minus dropped labels);
        otherwise ``n`` is the vertex count.

    Raises:
        ValueError: If *method* is not one of the recognised values.
        FileNotFoundError: If *surface* or *parcellation* do not exist.
    """
    vertices, faces = load_gifti(surface).agg_data()

    if parcellation is None:
        return vertices

    label_data = load_gifti(parcellation)
    labels = label_data.agg_data()
    labeltable = label_data.labeltable.get_labels_as_dict()

    return _get_parcel_centroids(
        vertices, faces, labels, method=method, drop=drop, labeltable=labeltable
    )


class Rotation(NamedTuple):
    """Hemispheric rotation matrix pair.

    Attributes:
        left:  ``(3, 3)`` orthogonal rotation matrix for the left
            hemisphere.
        right: ``(3, 3)`` orthogonal rotation matrix for the right
            hemisphere, obtained by reflecting *left* across the Y-Z
            plane so that both hemispheres rotate coherently.
    """

    left: np.ndarray
    right: np.ndarray


def _gen_rotation(seed: int | None = None) -> Rotation:
    """Generate a random orthogonal rotation matrix for spherical coordinates.

    Produces a uniform random ``(3, 3)`` rotation matrix via QR decomposition,
    corrects the determinant, then reflects across the Y-Z plane for the
    right hemisphere.

    Args:
        seed: Seed for the random number generator.

    Returns:
        A :class:`Rotation` namedtuple with ``left`` and ``right``
        ``(3, 3)`` rotation matrices.
    """
    rs = np.random.default_rng(seed)
    reflect_x = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    rot_l, temp = np.linalg.qr(rs.normal(size=(3, 3)))
    rot_l = rot_l @ np.diag(np.sign(np.diag(temp)))
    if np.linalg.det(rot_l) < 0:
        rot_l[:, 0] = -rot_l[:, 0]

    rot_r = reflect_x @ rot_l @ reflect_x

    return Rotation(left=rot_l, right=rot_r)


def _validate_spin_inputs(
    coords: np.ndarray,
    hemi_id: np.ndarray,
) -> None:
    """Validate shape and hemisphere designation arrays."""
    if coords.shape[-1] != 3 or coords.squeeze().ndim != 2:
        raise ValueError(
            f"Provided `coords` must be of shape (N, 3), not {coords.shape}"
        )
    if hemi_id.ndim != 1:
        raise ValueError("Provided `hemi_id` array must be one-dimensional.")
    if len(coords) != len(hemi_id):
        raise ValueError(
            f"Provided `coords` and `hemi_id` must have the same "
            f"length. Provided lengths: coords={len(coords)}, hemi_id={len(hemi_id)}"
        )
    if np.max(hemi_id) > 1 or np.min(hemi_id) < 0:
        raise ValueError(
            f"hemi_id must have values in {{0, 1}} denoting left and "
            f"right hemisphere coordinates, respectively. "
            f"Provided array contains values: {np.unique(hemi_id)}"
        )


class SpinResult(NamedTuple):
    """Result of spin sample generation.

    Attributes:
        spin: ``(N, P)`` resampling array where each column is a
            permutation of vertex indices.
        cost: ``(N, P)`` array of Euclidean reassignment distances
            if ``return_cost`` was ``True``, otherwise ``None``.
    """

    spin: np.ndarray
    cost: np.ndarray | None


def _assign_coordinates(
    coor: np.ndarray,
    rotated: np.ndarray,
    method: Literal["original", "vasa", "hungarian"],
) -> tuple[np.ndarray, np.ndarray]:
    """Match rotated coordinates to original grid."""
    n = len(coor)
    if method == "original":
        dist, col = spatial.KDTree(rotated).query(coor, 1)
        return col, dist
    if method == "vasa":
        dist = spatial.distance.cdist(coor, rotated)
        col = np.zeros(n, dtype=np.int32)
        costs = np.empty(n)
        for _ in range(n):
            row = dist.min(axis=1).argmax()
            col[row] = dist[row].argmin()
            costs[row] = dist[row, col[row]]
            dist[row] = -np.inf
            dist[:, col[row]] = np.inf
        return col, costs
    # optimization of total cost uusing Hungarian algorithm; may result in
    # certain parcels having higher cost (relativa to 'vasa'), but total
    # cost should always be lower
    if method == "hungarian":
        dist = spatial.distance.cdist(coor, rotated)
        row, col = optimize.linear_sum_assignment(dist)
        costs = np.empty(n)
        costs[row] = dist[row, col]
        return col, costs
    raise ValueError(f"Unknown method: {method}")


def gen_spin_samples(
    coords: ArrayLike,
    hemi_id: ArrayLike,
    *,
    n_rotate: int = 1000,
    method: Literal["original", "vasa", "hungarian"] = "original",
    check_duplicates: bool = True,
    seed: int | None = None,
    return_cost: bool = False,
) -> SpinResult:
    """Generate spin permutation samples by randomly rotating coordinates.

    Applies random rotations to *coords* on the unit sphere to produce a
    resampling array that preserves the spatial embedding of the original
    coordinates.  Rotations are generated for one hemisphere and mirrored
    across the Y-Z plane for the other via :func:`_gen_rotation`.

    Args:
        coords: ``(N, 3)`` array of X, Y, Z coordinates for ``N`` nodes,
            parcels, or vertices.
        hemi_id: ``(N,)`` array of hemisphere designations as ``{0, 1}``.
        n_rotate: Number of rotations to generate. Default is ``1000``.
        method: Strategy for matching rotated coordinates back to the
            original grid.  ``'original'`` uses a KD-tree
            (memory-efficient but may assign multiple coordinates to the
            same target).  ``'vasa'`` uses a greedy min-max assignment.
            ``'hungarian'`` uses the Hungarian algorithm for optimal
            assignment. Default is ``'original'``.
        check_duplicates: If ``True``, retry rotations (up to 500 times)
            to ensure each spin is unique and not identical to the input.
            Default is ``True``.
        seed: Seed for the random number generator.
        return_cost: If ``True``, populate the ``cost`` field with the
            Euclidean reassignment distances. Default is ``False``.

    Returns:
        A :class:`SpinResult` namedtuple with ``(N, n_rotate)`` spin
        array and optional cost array.

    Raises:
        ValueError: If *coords* is not ``(N, 3)``, *hemi_id* is not
            ``(N,)``, or *method* is not recognised.
    """
    if method not in (methods := frozenset(["original", "vasa", "hungarian"])):
        raise ValueError(
            f"Provided method '{method}' invalid. Must be one of {methods}."
        )

    coords = np.asanyarray(coords)
    hemi_id = np.squeeze(np.asanyarray(hemi_id, dtype=np.int8))
    _validate_spin_inputs(coords=coords, hemi_id=hemi_id)

    n_coords = len(coords)
    spin = np.zeros((n_coords, n_rotate), dtype=np.int32)
    cost = np.zeros((n_coords, n_rotate), dtype=np.float64) if return_cost else None
    inds = np.arange(n_coords, dtype=np.int32)
    seen: set[tuple[np.int32, ...]] | None = set() if check_duplicates else None

    rs = np.random.default_rng(seed)
    warned = False
    for k in range(n_rotate):
        count, duplicated = 0, True

        _logger.info("Generating spin %5d of %d", k, n_rotate)

        while duplicated and count < 500:
            count += 1
            duplicated = False
            resampled = np.zeros(n_coords, dtype=np.int32)

            # Mirrored rotation pair for both hemispheres, rotating separately
            for h, rot in enumerate(
                _gen_rotation(seed=rs.integers(0, np.iinfo(np.int32).max, dtype=int))
            ):
                hinds = hemi_id == h
                coor = coords[hinds]
                if coor.size == 0:
                    continue

                rotated = coor @ rot
                col, costs = _assign_coordinates(coor, rotated, method)
                resampled[hinds] = inds[hinds][col]
                if cost is not None:
                    cost[hinds, k] = costs

            if seen is not None:
                resampled_tuple = tuple(resampled)
                duplicated = bool(resampled_tuple in seen or np.all(resampled == inds))

        if count == 500 and not warned:
            _logger.warning(
                "Duplicate rotations used. Check resampling array to determine "
                "real number of unique permutations.",
                stacklevel=2,
            )
            warned = True

        spin[:, k] = resampled
        if seen is not None:
            seen.add(tuple(resampled))

    return SpinResult(spin=spin, cost=cost)


def _max_overlap(vals: np.ndarray) -> int:
    """Return the most common positive label in *vals* minus 1, or -1 if none.

    The ``-1`` offset maps consecutive 1-based parcel labels to
    0-based row indices for :func:`ndimage.labeled_comprehension`.
    """
    vals, counts = np.unique(vals[vals > 0], return_counts=True)
    try:
        return int(vals[counts.argmax()]) - 1
    except (IndexError, ValueError):
        return -1


def _to_hemisphere_list(
    item: str | Path | tuple[str | Path, str | Path],
) -> list[str | Path]:
    """Normalise *item* to a list of one or two file paths."""
    if isinstance(item, str | Path):
        return [item]
    return list(item)


def spin_parcels(
    surfaces: str | Path | tuple[str | Path, str | Path],
    parcellation: str | Path | tuple[str | Path, str | Path],
    *,
    method: Literal["average", "surface", "geodesic"] = "surface",
    n_rotate: int = 1000,
    spins: ArrayLike | None = None,
    seed: int | None = None,
    return_cost: bool = False,
    **kwargs: Any,  # noqa: ANN401 kwargs passed to :func:`gen_spin_samples`
) -> SpinResult:
    """Rotate parcel boundaries and reassign by maximum overlap.

    Vertex-level labels are rotated and each *parcel* is reassigned
    based on the region that maximally overlaps with its boundaries.
    This produces a resampling matrix for permuting parcel-level data
    while preserving spatial autocorrelation.

    Args:
        surfaces: Path to a single GIFTI surface file or ``(left,
            right)`` pair. Spherical surfaces are recommended.
        parcellation: Path to a single GIFTI label file or ``(left,
            right)`` pair containing parcel labels on the
            corresponding surface(s).
        method: How to compute parcel centroids for spin generation.
            Passed to :func:`get_parcel_centroids`. Default is
            ``'surface'``.
        n_rotate: Number of rotations to generate. Default is ``1000``.
        spins: Pre-computed spin permutations to use instead of
            generating them on the fly. If provided, *surfaces*,
            *method*, *n_rotate*, and *seed* are ignored. Default is
            ``None``.
        seed: Seed for random number generation.
        return_cost: Whether to return the cost array (Euclidean
            reassignment distances). Default is ``False``.
        **kwargs: Additional keyword arguments passed to
            :func:`gen_spin_samples`.

    Returns:
        :class:`SpinResult` with ``(n_parcels, P)`` resampling array
        and optional ``(n_coords, P)`` cost array.

    Raises:
        ValueError: If the parcellation and surface have mismatched
            vertex counts.

    Note:
        Background label (0) is excluded from the output. The row
        order corresponds to the sorted unique non-zero parcel labels.
    """
    surfaces_list = _to_hemisphere_list(surfaces)
    parcellation_list = _to_hemisphere_list(parcellation)

    if len(surfaces_list) != len(parcellation_list):
        raise ValueError(
            "Number of surface and parcellation files must match. "
            f"Got {len(surfaces_list)} surface(s) and "
            f"{len(parcellation_list)} parcellation(s)."
        )

    vertex_labels = np.hstack(
        [load_gifti(parc).agg_data() for parc in parcellation_list]
    )
    label_values = np.unique(vertex_labels)
    parcel_mask = label_values != 0
    n_vertices = len(vertex_labels)
    n_parcels = int(parcel_mask.sum())

    if spins is None:
        centroid_list = []
        hemi_list = []
        for hemi, surf in enumerate(surfaces_list):
            centroids = get_parcel_centroids(surf, method=method)
            centroid_list.append(centroids)
            hemi_list.append(np.full(len(centroids), hemi, dtype=np.int8))
        centroid_coords = np.vstack(centroid_list)
        hemi_ids = np.hstack(hemi_list)

        result = gen_spin_samples(
            centroid_coords,
            hemi_ids,
            n_rotate=n_rotate,
            seed=seed,
            return_cost=return_cost,
            **kwargs,
        )
        spin_perm = result.spin
        cost = result.cost
    else:
        spin_perm = load_spins(spins)
        cost = None

    n_spins = spin_perm.shape[1]

    if n_vertices != len(spin_perm):
        raise ValueError(
            f"Parcellation vertex count ({n_vertices}) does not match "
            f"spin array length ({len(spin_perm)})"
        )

    regions = np.zeros((n_parcels, n_spins), dtype=np.int32)
    for spin_idx in range(n_spins):
        _logger.info("Calculating parcel overlap: %5d/%d", spin_idx, n_spins)
        regions[:, spin_idx] = labeled_comprehension(
            vertex_labels[spin_perm[:, spin_idx]],
            vertex_labels,
            label_values,
            _max_overlap,
            np.int32,
            -1,
        )[parcel_mask]

    return SpinResult(spin=regions, cost=cost if return_cost else None)


def _project_parcels_to_verts(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Project parcel-level data to vertices given pre-loaded labels."""
    n_parcels = int(labels.max())

    if n_parcels != data.shape[0]:
        raise ValueError(
            f"Number of parcels ({n_parcels}) does not match "
            f"data shape ({data.shape[0]})."
        )

    # Prepend NaN row for background (label 0); data[labels] fills
    # background vertices with NaN
    return np.vstack([np.full((1, data.shape[1]), np.nan), data])[labels, :].squeeze()


def parcels_to_vertices(
    data: ArrayLike, parcellation: str | Path | tuple[str | Path, str | Path]
) -> np.ndarray:
    """Project parcellated *data* to vertices.

    Each vertex receives the value of the parcel it belongs to.
    Background vertices (label 0) are filled with ``NaN``.

    Args:
        data: Parcellated data.  A 1-D array ``(n_parcels,)`` or 2-D
            ``(n_parcels, n_features)``.  For dual-hemisphere inputs,
            parcel labels must share the same numeric range across
            hemispheres.
        parcellation: Path to a single GIFTI label file or ``(left,
            right)`` pair.  Label indices must be consecutive integers
            starting at 1 (background is 0). Use
            :func:`~neuromaps_prime.analysis.images.relabel_gifti` to
            ensure consecutive labeling.

    Returns:
        Vertex-level data of shape ``(n_vertices,)`` or
        ``(n_vertices, n_features)``.  If both hemispheres are
        provided, vertex order is ``[left, right]``.  Singleton
        trailing dimensions are squeezed.

    Raises:
        ValueError: If the number of parcels in the parcellation does
            not match the size of *data*.
    """
    data = np.asarray(data, dtype=float)
    data = np.vstack(data)
    parcellation_list = _to_hemisphere_list(parcellation)
    labels = np.hstack([load_gifti(parc).agg_data() for parc in parcellation_list])
    return _project_parcels_to_verts(data, labels)


def _reduce_vertices_to_parcels(
    data: np.ndarray,
    labels: np.ndarray,
    *,
    parcel_values: np.ndarray,
    background: float | None = None,
) -> np.ndarray:
    """Reduce vertex-level data to parcels given pre-loaded labels.

    Core logic for :func:`vertices_to_parcels`.
    """
    if background is not None:
        data = data.copy()
        data[data == background] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        reduced = np.array(
            [np.nanmean(data[labels == lab], axis=0) for lab in parcel_values],
            dtype=float,
        )

    # Strip background parcel (label 0)
    return reduced[parcel_values != 0].squeeze()


def vertices_to_parcels(
    data: ArrayLike,
    parcellation: str | Path | tuple[str | Path, str | Path],
    *,
    background: float | None = None,
) -> np.ndarray:
    """Reduce vertex-level *data* to parcels.

    Computes the mean of *data* within each parcel, ignoring vertices
    that are ``NaN`` or equal to *background*.  Parcels whose vertices
    are all ``NaN`` or *background* receive ``NaN``.

    Args:
        data: Vertex-level data.  A 1-D array ``(n_vertices,)`` or 2-D
            ``(n_vertices, n_features)``.  For dual-hemisphere inputs,
            vertex order should be ``[left, right]``.
        parcellation: Path to a single GIFTI label file or ``(left,
            right)`` pair defining parcels.
        background: Optional value to treat as background and ignore
            when computing means.  Default is ``None``.

    Returns:
        Parcellated data of shape ``(n_parcels,)`` or
        ``(n_parcels, n_features)``.  Singleton trailing dimensions
        are squeezed.  Background parcels (label 0) are excluded.

    Raises:
        ValueError: If the number of vertices in the parcellation does
            not match the size of *data*.
    """
    data = np.asarray(data, dtype=float)
    data = np.expand_dims(data, axis=-1) if data.ndim == 1 else data

    parcellation_list = _to_hemisphere_list(parcellation)
    labels = np.hstack([load_gifti(p).agg_data() for p in parcellation_list])

    if data.shape[0] != len(labels):
        raise ValueError(
            f"Vertex count ({len(labels)}) does not match data shape ({data.shape[0]})."
        )

    parcel_values = np.unique(labels)
    return _reduce_vertices_to_parcels(
        data, labels, parcel_values=parcel_values, background=background
    )


def spin_data(
    data: ArrayLike,
    surfaces: str | Path | tuple[str | Path, str | Path],
    parcellation: str | Path | tuple[str | Path, str | Path],
    *,
    method: Literal["average", "surface", "geodesic"] = "surface",
    n_rotate: int = 1000,
    spins: ArrayLike | None = None,
    seed: int | None = None,
    return_cost: bool = False,
    **kwargs: Any,  # noqa: ANN401 kwargs passed to :func:`gen_spin_samples`
) -> SpinResult:
    """Spin parcellated *data* by rotating parcel boundaries on a surface.

    Projects parcel data to vertices, applies random spherical
    rotations, then re-averages back to the parcel level.  Re-averaging
    means spun values will differ slightly from the original; parcels
    whose vertices fall entirely within background regions receive
    ``NaN``.

    Args:
        data: 1-D array of shape ``(n_parcels,)`` indexed by parcel
            label value (``data[0]`` is label 1, ``data[1]`` is label 2,
            etc.).
        surfaces: Path to a single GIFTI surface file or ``(left,
            right)`` pair. Spherical surfaces are recommended.
        parcellation: Path to a single GIFTI label file or ``(left,
            right)`` pair containing parcel labels on the
            corresponding surface(s).
        method: How to compute parcel centroids for spin generation.
            Passed to :func:`get_parcel_centroids`. Default is
            ``'surface'``.
        n_rotate: Number of rotations to generate. Default is ``1000``.
        spins: Pre-computed spin permutations to use instead of
            generating them on the fly. If provided, *surfaces*,
            *method*, *n_rotate*, and *seed* are ignored. Default is
            ``None``.
        seed: Seed for random number generation.
        return_cost: Whether to return the cost array (Euclidean
            reassignment distances). Default is ``False``.
        **kwargs: Additional keyword arguments passed to
            :func:`gen_spin_samples`.

    Returns:
        :class:`SpinResult` with ``(n_parcels, n_rotate)`` spin array
        and optional ``(n_coords, n_rotate)`` cost array.

    Raises:
        ValueError: If the parcellation and surface have mismatched
            vertex counts.

    Note:
        Background parcels (label 0) are excluded from the output.
        When both hemispheres are provided, the parcel mean is computed
        across all vertices sharing that label value in both hemispheres.
    """
    surfaces_list = _to_hemisphere_list(surfaces)
    parcellation_list = _to_hemisphere_list(parcellation)
    if len(surfaces_list) != len(parcellation_list):
        raise ValueError(
            "Number of surface and parcellation files must match. "
            f"Got {len(surfaces_list)} surface(s) and "
            f"{len(parcellation_list)} parcellation(s)."
        )

    data = np.asarray(data, dtype=float)
    data_2d = np.vstack(data) if data.ndim == 1 else data[..., np.newaxis]

    # Load parcellation images once; reuse data + labeltables
    label_img_list = [load_gifti(parc) for parc in parcellation_list]
    label_list = [img.agg_data() for img in label_img_list]
    combined_labels = np.hstack(label_list)
    parcel_values = np.unique(combined_labels)
    parcel_mask = parcel_values != 0

    vertex_data = _project_parcels_to_verts(data_2d, combined_labels)

    if spins is None:
        centroid_list = []
        hemi_list = []
        for hemi, surf in enumerate(surfaces_list):
            centroids = get_parcel_centroids(surf, method=method)
            centroid_list.append(centroids)
            hemi_list.append(np.full(len(centroids), hemi, dtype=np.int8))
        centroid_coords = np.vstack(centroid_list)
        hemi_ids = np.hstack(hemi_list)

        result = gen_spin_samples(
            centroid_coords,
            hemi_ids,
            n_rotate=n_rotate,
            seed=seed,
            return_cost=return_cost,
            **kwargs,
        )
        spin_arr = result.spin
        cost = result.cost
    else:
        spin_arr = load_spins(spins)
        cost = None

    n_vert = len(vertex_data)
    if n_vert != len(spin_arr):
        raise ValueError(
            f"Vertex count ({n_vert}) does not match "
            f"spin array length ({len(spin_arr)})"
        )

    spun = np.zeros((int(parcel_mask.sum()), spin_arr.shape[1]), dtype=float)
    for spin_idx in range(n_rotate):
        _logger.info("Reducing vertices to parcels: %5d/%d", spin_idx, n_rotate)
        rotated = vertex_data[spin_arr[:, spin_idx]]
        reduced = _reduce_vertices_to_parcels(
            rotated, combined_labels, parcel_values=parcel_values
        )
        spun[:, spin_idx] = reduced

    return SpinResult(spin=spun, cost=cost)
