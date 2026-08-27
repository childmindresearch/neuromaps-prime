"""Spatial null model convenience wrappers.

Combines centroid computation, spin generation, and data permutation
into single-call functions.  Adapted from neuromaps.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from numpy.typing import ArrayLike

from neuromaps_prime.analysis.images import PARC_IGNORE, load_data, relabel_gifti
from neuromaps_prime.analysis.surfaces.nulls.burt import batch_surrogates
from neuromaps_prime.analysis.surfaces.nulls.spins import (
    _get_parcel_centroids,
    _to_hemisphere_list,
    gen_spin_samples,
    load_spins,
    spin_data,
    spin_parcels,
)
from neuromaps_prime.analysis.surfaces.points import get_surface_distance

__all__ = ["alexander_bloch", "baum", "burt2018", "cornblath", "hungarian", "vasa"]

_DataT = tuple[str | Path | nib.GiftiImage, str | Path | nib.GiftiImage]
_SpinMethod = Literal["original", "vasa", "hungarian"]
_CentroidMethod = Literal["average", "surface", "geodesic"]


def _resolve_data(
    data: ArrayLike | str | Path | nib.GiftiImage | _DataT,
    gifti_cache: dict[str | Path, nib.GiftiImage],
) -> np.ndarray:
    """Load *data* as a 1-D array, reusing cached GIFTI images."""

    def _load_one(item: str | Path | nib.GiftiImage) -> np.ndarray:
        if isinstance(item, nib.GiftiImage):
            return item.agg_data().ravel()

        if isinstance(item, str | Path) and item in gifti_cache:
            return gifti_cache[item].agg_data().ravel()

        data = load_data(item, dtype=float)
        return data.array.ravel()

    if isinstance(data, tuple):
        return np.hstack([_load_one(d) for d in data])
    return _load_one(data)  # type: ignore[arg-type] # Not a tuple


def _generate_spins(
    surface: str | Path | tuple[str | Path, str | Path],
    parcellation: str | Path | tuple[str | Path, str | Path] | None,
    n_perm: int,
    seed: int | None,
    centroid_method: _CentroidMethod,
    spin_method: _SpinMethod,
    drop: Iterable[str],
    gifti_cache: dict[str | Path, nib.GiftiImage],
) -> tuple[np.ndarray, int]:
    """Generate spin matrix from surface, returning (spin_matrix, n_elements)."""
    surfaces_list = _to_hemisphere_list(surface)
    parcellation_list = (
        _to_hemisphere_list(parcellation) if parcellation is not None else []
    )
    if parcellation is not None and len(surfaces_list) != len(parcellation_list):
        raise ValueError(
            "Number of surface and parcellation files must match. "
            f"Got {len(surfaces_list)} surface(s) and "
            f"{len(parcellation_list)} parcellation(s)."
        )

    # Pre-load all GIFTI files once
    for item in [*surfaces_list, *parcellation_list]:
        if isinstance(item, str | Path) and item not in gifti_cache:
            _, img = load_data(item, return_image=True)
            if isinstance(img, nib.GiftiImage):
                gifti_cache[item] = img

    centroid_list = []
    hemi_list = []
    for hemi, surf in enumerate(surfaces_list):
        verts, faces = gifti_cache[surf].agg_data()
        if hemi < len(parcellation_list):
            parc_img = gifti_cache[parcellation_list[hemi]]
            labels = parc_img.agg_data()
            labeltable = parc_img.labeltable.get_labels_as_dict()
            centroids = _get_parcel_centroids(
                verts,
                faces,
                labels,
                method=centroid_method,
                drop=drop,
                labeltable=labeltable,
            )
        else:
            centroids = verts
        centroid_list.append(centroids)
        hemi_list.append(np.full(len(centroids), hemi, dtype=np.int8))

    centroid_coords = np.vstack(centroid_list)
    hemi_ids = np.hstack(hemi_list)
    spin_matrix = gen_spin_samples(
        centroid_coords, hemi_ids, n_rotate=n_perm, method=spin_method, seed=seed
    ).spin
    return spin_matrix, len(centroid_coords)


def _spin_permute(
    data: ArrayLike | str | Path | nib.GiftiImage | _DataT | None,
    surface: str | Path | tuple[str | Path, str | Path],
    *,
    parcellation: str | Path | tuple[str | Path, str | Path] | None,
    n_perm: int,
    seed: int | None,
    spins: ArrayLike | None,
    spin_method: _SpinMethod,
    method: _CentroidMethod,
    drop: Iterable[str],
) -> np.ndarray:
    """Core spin-permutation pipeline, shared by :func:`alexander_bloch`, etc."""
    gifti_cache: dict[str | Path, nib.GiftiImage] = {}

    if spins is not None:
        spin_matrix = load_spins(spins)
        n_elements = spin_matrix.shape[0]
    else:
        spin_matrix, n_elements = _generate_spins(
            surface, parcellation, n_perm, seed, method, spin_method, drop, gifti_cache
        )

    if data is None:
        return spin_matrix

    data_array = _resolve_data(data, gifti_cache)
    if len(data_array) != n_elements:
        raise ValueError(
            f"Data length ({len(data_array)}) does not match "
            f"number of elements ({n_elements}) in the spin matrix."
        )

    return data_array[spin_matrix]


def alexander_bloch(
    data: ArrayLike | str | Path | nib.GiftiImage | _DataT | None,
    surface: str | Path | tuple[str | Path, str | Path],
    *,
    parcellation: str | Path | tuple[str | Path, str | Path] | None = None,
    n_perm: int = 1000,
    seed: int | None = None,
    spins: ArrayLike | None = None,
    method: _CentroidMethod = "surface",
    drop: Iterable[str] = PARC_IGNORE,
) -> np.ndarray:
    """Spin-permute data on a cortical surface.

    Method projects data to a spherical surface and uses arbitrary
    rotations to generate null distribution.  If *data* are parcellated
    then parcel centroids are projected to surface and parcels are
    reassigned based on minimum distances.

    Args:
        data: 1-D array of shape ``(n,)``, a GIFTI file path, a
            ``nibabel.GiftiImage``, or ``(left, right)`` pair of any of
            these.  If ``None``, returns the spin permutation matrix
            instead of permuted data.
        surface: Path to a single-hemisphere GIFTI surface file, or
            ``(left, right)`` pair.
        parcellation: Path to a single-hemisphere GIFTI label file, or
            ``(left, right)`` pair.  If ``None``, *data* is vertex-level
            and all vertices are used as centroids.
        n_perm: Number of spin permutations. Default ``1000``.
        seed: Random seed.
        spins: Pre-computed ``(n, P)`` spin array. If provided, the
            surface and parcellation arguments are ignored.
        method: Centroid computation strategy. Default ``'surface'``.
        drop: Label names to exclude. Default ``PARC_IGNORE``.

    Returns:
        Array of shape ``(n, n_perm)`` with spin-permuted data, or the
        raw permutation indices if *data* is ``None``.

    Raises:
        ValueError: If *data* length does not match the number of
            elements implied by the surface or parcellation.

    References:
        Alexander-Bloch et al. (2018). NeuroImage, 178, 540-51.
    """
    return _spin_permute(
        data,
        surface,
        parcellation=parcellation,
        n_perm=n_perm,
        seed=seed,
        spins=spins,
        spin_method="original",
        method=method,
        drop=drop,
    )


def vasa(
    data: ArrayLike | str | Path | nib.GiftiImage | _DataT | None,
    surface: str | Path | tuple[str | Path, str | Path],
    *,
    parcellation: str | Path | tuple[str | Path, str | Path],
    n_perm: int = 1000,
    seed: int | None = None,
    spins: ArrayLike | None = None,
    method: _CentroidMethod = "surface",
    drop: Iterable[str] = PARC_IGNORE,
) -> np.ndarray:
    """Spin-permute parcellated data using the Vasa assignment strategy.

    Method projects parcels to a spherical surface and uses arbitrary
    rotations with iterative reassignments to generate null distribution.
    All nulls are "perfect" permutations of the input data (at the slight
    expense of spatial topology).

    Args:
        data: 1-D array of shape ``(n,)``, a GIFTI file path, a
            ``nibabel.GiftiImage``, or ``(left, right)`` pair of any of
            these.  If ``None``, returns the spin permutation matrix
            instead of permuted data.
        surface: Path to a single-hemisphere GIFTI surface file, or
            ``(left, right)`` pair.
        parcellation: Path to a single-hemisphere GIFTI label file, or
            ``(left, right)`` pair.  Must be provided.
        n_perm: Number of spin permutations. Default ``1000``.
        seed: Random seed.
        spins: Pre-computed ``(n, P)`` spin array. If provided, the
            surface and parcellation arguments are ignored.
        method: Centroid computation strategy. Default ``'surface'``.
        drop: Label names to exclude. Default ``PARC_IGNORE``.

    Returns:
        Array of shape ``(n, n_perm)`` with spin-permuted data, or the
        raw permutation indices if *data* is ``None``.

    Raises:
        ValueError: If *parcellation* is not provided, or if *data*
            length does not match the number of elements implied by the
            surface or parcellation.

    References:
        Váša et al. (2018). Cerebral Cortex, 28(1), 281-294.
    """
    return _spin_permute(
        data,
        surface,
        parcellation=parcellation,
        n_perm=n_perm,
        seed=seed,
        spins=spins,
        spin_method="vasa",
        method=method,
        drop=drop,
    )


def hungarian(
    data: ArrayLike | str | Path | nib.GiftiImage | _DataT | None,
    surface: str | Path | tuple[str | Path, str | Path],
    *,
    parcellation: str | Path | tuple[str | Path, str | Path],
    n_perm: int = 1000,
    seed: int | None = None,
    spins: ArrayLike | None = None,
    method: _CentroidMethod = "surface",
    drop: Iterable[str] = PARC_IGNORE,
) -> np.ndarray:
    """Spin-permute parcellated data using the Hungarian assignment algorithm.

    Method projects parcels to a spherical surface and uses arbitrary
    rotations with iterative reassignments to generate null distribution.
    All nulls are "perfect" permutations of the input data (at the slight
    expense of spatial topology).

    Args:
        data: 1-D array of shape ``(n,)``, a GIFTI file path, a
            ``nibabel.GiftiImage``, or ``(left, right)`` pair of any of
            these.  If ``None``, returns the spin permutation matrix
            instead of permuted data.
        surface: Path to a single-hemisphere GIFTI surface file, or
            ``(left, right)`` pair.
        parcellation: Path to a single-hemisphere GIFTI label file, or
            ``(left, right)`` pair.  Must be provided.
        n_perm: Number of spin permutations. Default ``1000``.
        seed: Random seed.
        spins: Pre-computed ``(n, P)`` spin array. If provided, the
            surface and parcellation arguments are ignored.
        method: Centroid computation strategy. Default ``'surface'``.
        drop: Label names to exclude. Default ``PARC_IGNORE``.

    Returns:
        Array of shape ``(n, n_perm)`` with spin-permuted data, or the
        raw permutation indices if *data* is ``None``.

    Raises:
        ValueError: If *parcellation* is not provided, or if *data*
            length does not match the number of elements implied by the
            surface or parcellation.

    References:
        Kuhn (1955). Naval Research Logistics Quarterly, 2(1-2), 83-97.
    """
    return _spin_permute(
        data,
        surface,
        parcellation=parcellation,
        n_perm=n_perm,
        seed=seed,
        spins=spins,
        spin_method="hungarian",
        method=method,
        drop=drop,
    )


def baum(
    data: ArrayLike | str | Path | nib.GiftiImage | _DataT | None,
    surface: str | Path | tuple[str | Path, str | Path],
    *,
    parcellation: str | Path | tuple[str | Path, str | Path],
    n_perm: int = 1000,
    seed: int | None = None,
    spins: ArrayLike | None = None,
    method: _CentroidMethod = "surface",
) -> np.ndarray:
    """Spin-permute parcellated data using the Baum max-overlap strategy.

    Method projects *data* to spherical surface and uses arbitrary
    rotations to generate null distributions.  Reassigned parcels are
    based on the most common (i.e., modal) value of the vertices in
    each parcel within the rotated data.

    Args:
        data: 1-D array of shape ``(n,)``, a GIFTI file path, a
            ``nibabel.GiftiImage``, or ``(left, right)`` pair of any of
            these.  If ``None``, returns the spin permutation matrix
            instead of permuted data.
        surface: Path to a single-hemisphere GIFTI surface file, or
            ``(left, right)`` pair.
        parcellation: Path to a single-hemisphere GIFTI label file, or
            ``(left, right)`` pair.  Must be provided.
        n_perm: Number of spin permutations. Default ``1000``.
        seed: Random seed.
        spins: Pre-computed ``(n_parcels, P)`` spin array. If provided,
            the surface and parcellation arguments are ignored.
        method: Centroid computation strategy. Default ``'surface'``.

    Returns:
        Array of shape ``(n_parcels, n_perm)`` with spin-permuted data.
        Unmatched parcels are ``NaN``. Returns the raw permutation
        indices if *data* is ``None``.

    Raises:
        ValueError: If *data* length does not match the number of
            parcels implied by the parcellation.

    References:
        Baum et al. (2020). PNAS, 117(1), 771-778.
    """
    result = spin_parcels(
        surface, parcellation, method=method, n_rotate=n_perm, spins=spins, seed=seed
    )
    spin_matrix = result.spin
    n_parcels = spin_matrix.shape[0]

    if data is None:
        return spin_matrix

    data_array = _resolve_data(data, {})
    if len(data_array) != n_parcels:
        raise ValueError(
            f"Data length ({len(data_array)}) does not match "
            f"number of parcels ({n_parcels})."
        )

    nulls = data_array[spin_matrix]
    nulls[spin_matrix == -1] = np.nan
    return nulls


def cornblath(
    data: ArrayLike | str | Path | nib.GiftiImage | _DataT,
    surface: str | Path | tuple[str | Path, str | Path],
    *,
    parcellation: str | Path | tuple[str | Path, str | Path],
    n_perm: int = 1000,
    seed: int | None = None,
    spins: ArrayLike | None = None,
    method: _CentroidMethod = "surface",
) -> np.ndarray:
    """Spin-permute parcellated data using the Cornblath re-averaging strategy.

    Method projects *data* to spherical surface and uses arbitrary
    rotations to generate null distributions.  Reassigned parcels are
    based on the average value of the vertices in each parcel within
    the rotated data.

    Args:
        data: 1-D array of shape ``(n,)``, a GIFTI file path, a
            ``nibabel.GiftiImage``, or ``(left, right)`` pair of any of
            these.
        surface: Path to a single-hemisphere GIFTI surface file, or
            ``(left, right)`` pair.
        parcellation: Path to a single-hemisphere GIFTI label file, or
            ``(left, right)`` pair.  Must be provided.
        n_perm: Number of spin permutations. Default ``1000``.
        seed: Random seed.
        spins: Pre-computed ``(n_parcels, P)`` spin array. If provided,
            the surface and parcellation arguments are ignored.
        method: Centroid computation strategy. Default ``'surface'``.

    Returns:
        Array of shape ``(n_parcels, n_perm)`` with spin-permuted data.
        Values may differ slightly from the original due to re-averaging.

    Raises:
        ValueError: If *data* length does not match the number of
            parcels implied by the parcellation.

    References:
        Cornblath et al. (2020). Communications Biology, 3(1), 1-12.
    """
    data_array = _resolve_data(data, {})
    result = spin_data(
        data_array,
        surface,
        parcellation,
        method=method,
        n_rotate=n_perm,
        spins=spins,
        seed=seed,
    )
    return result.spin


def burt2018(
    data: ArrayLike | str | Path | nib.GiftiImage | _DataT,
    surface: str | Path | tuple[str | Path, str | Path],
    *,
    parcellation: str | Path | tuple[str | Path, str | Path] | None = None,
    n_perm: int = 1000,
    seed: int | None = None,
    distmat: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    n_proc: int = 1,
) -> np.ndarray:
    """Generate spatial surrogates using the Burt 2018 method (surface only).

    Method uses a spatial auto-regressive model to estimate the
    distance-dependent relationship of *data* and generates surrogate
    maps with similar spatial autocorrelation properties.

    Args:
        data: 1-D array of shape ``(n,)``, a GIFTI file path, a
            ``nibabel.GiftiImage``, or ``(left, right)`` pair of any of
            these.
        surface: Path to a single-hemisphere GIFTI surface file, or
            ``(left, right)`` pair.
        parcellation: Path to a single-hemisphere GIFTI label file, or
            ``(left, right)`` pair. If ``None``, *data* is vertex-level.
        n_perm: Number of surrogate maps to generate. Default ``1000``.
        seed: Random seed.
        distmat: Pre-computed distance matrix (single) or
            ``(left, right)`` pair. If ``None``, distances are computed
            from the surface.
        n_proc: Number of parallel workers. Default ``1``.

    Returns:
        Array of shape ``(n, n_perm)`` with surrogate data.

    Raises:
        ValueError: If *data* length does not match the surface or
            parcellation.

    Note:
        Surface-only implementation. Volumetric surrogates were not
        adopted.

    References:
        Burt et al. (2018). Nature Neuroscience, 21(9), 1251-1259.
    """
    data_array = _resolve_data(data, {})
    surfaces_list = _to_hemisphere_list(surface)
    parcellation_list = (
        _to_hemisphere_list(parcellation) if parcellation is not None else []
    )
    n_hemis = len(surfaces_list)
    shift = np.abs(np.nanmin(data_array)) + 0.1
    surrogates = np.full((len(data_array), n_perm), np.nan, dtype=float)
    parcel_offset = 0

    for hemi, surf in enumerate(surfaces_list):
        hparc_path = (
            parcellation_list[hemi]
            if parcellation is not None and hemi < len(parcellation_list)
            else None
        )
        if hparc_path is not None:
            parc_img = relabel_gifti(hparc_path)
            vertex_labels = parc_img.agg_data()
            parc_labels = np.trim_zeros(np.unique(vertex_labels)) - 1
            # relabel_gifti makes labels per-hemisphere 1-indexed, so offset
            idx = parc_labels + parcel_offset
            hdata = data_array[idx]
            parcel_offset += len(parc_labels)
        elif n_hemis == 1:
            hdata = data_array
            idx = np.arange(len(data_array))
        else:
            lo, hi = hemi * (len(data_array) // 2), (hemi + 1) * (len(data_array) // 2)
            hdata = data_array[lo:hi]
            idx = np.arange(lo, hi)

        if distmat is not None:
            hdist = distmat[hemi] if isinstance(distmat, tuple) else distmat
        else:
            hdist = get_surface_distance(
                surf, parcellation=hparc_path, drop=PARC_IGNORE, n_proc=n_proc
            )
        # Mask out medial wall (via ``drop=PARC_IGNORE``)
        med = np.isinf(hdist + np.diag([np.inf] * len(hdist))).all(axis=1)
        mask = ~np.logical_or(np.isnan(hdata), med)
        hdata_masked = hdata[mask]
        hdist_masked = hdist[np.ix_(mask, mask)]
        hsl = idx[mask]
        # Ensure data is positive for ``batch_surrogates``
        hdata_masked = hdata_masked + shift

        hsurr = batch_surrogates(
            hdist_masked, hdata_masked, n_surr=n_perm, seed=seed, n_jobs=n_proc
        )

        surrogates[hsl, :] = hsurr

    return surrogates
