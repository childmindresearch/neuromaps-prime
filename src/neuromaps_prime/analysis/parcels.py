"""Reduce continuous brain maps to region-of-interest summaries."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from scipy import ndimage

from neuromaps_prime.analysis.images import load_data

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike

__all__ = ["ParcelSummary", "parcel_reduce"]


ReduceMethod = (
    Literal["mean", "median", "sum", "std", "min", "max"] | Callable[..., float]
)


class ParcelSummary(NamedTuple):
    """Region-wise summary of a continuous brain map.

    Attributes:
        labels: Integer ID of each region, ascending, excluding background.
        values: Summarised value per region. Shape ``(n_regions,)`` for 1-D
            input, or ``(n_regions, n_features)`` for 2-D input.
        dense: Summaries broadcast back onto the input space, so every
            voxel/vertex carries its region's value. ``None`` unless
            ``propagate=True``. Same spatial shape as the input map.
    """

    labels: np.ndarray
    values: np.ndarray
    dense: np.ndarray | None = None


# scipy.ndimage reducers, called as ``func(values, labels, index)``.
_NDIMAGE_REDUCERS: dict[str, Callable] = {
    "mean": ndimage.mean,
    "median": ndimage.median,
    "sum": ndimage.sum_labels,
    "std": ndimage.standard_deviation,
    "min": ndimage.minimum,
    "max": ndimage.maximum,
}


def _collapse(arr: np.ndarray) -> np.ndarray:
    """Collapse spatial dimensions, preserving a trailing feature axis.

    Surface data are ``(n_vertices,)`` or ``(n_vertices, n_features)``.
    Volumetric data are ``(i, j, k)`` or ``(i, j, k, n_features)``.
    """
    if arr.ndim in (1, 3):
        return arr.reshape(-1, 1)
    if arr.ndim in (2, 4):
        return arr.reshape(-1, arr.shape[-1])
    raise ValueError(f"Unsupported array with {arr.ndim} dimensions.")


def parcel_reduce(
    data: ArrayLike | str | Path,
    parcellation: ArrayLike | str | Path,
    *,
    method: ReduceMethod = "mean",
    background: float | int | None = 0,
    min_valid: int = 1,
    *,
    drop_nonfinite: bool = True,
    propagate: bool = False,
) -> ParcelSummary:
    """Summarises a continuous brain map within each region of a parcellation.

    Works for surface and volumetric data alike: the parcellation simply has to
    have the same spatial shape as the map. No resampling is performed, so
    transform the map into the parcellation's space first if needed.

    Args:
        data: Continuous map. An array, a path to a GIFTI or NIfTI file, or a
            nibabel image. Surface data may be ``(n_vertices,)`` or
            ``(n_vertices, n_features)``; volumetric data ``(i, j, k)`` or
            ``(i, j, k, n_features)``.
        parcellation: Discrete parcellation with the same spatial shape as
            ``data``, where each region is a unique integer ID.
        method: How to summarise within each region. One of ``'mean'``
            (default), ``'median'``, ``'sum'``, ``'std'``, ``'min'``,
            ``'max'`` (computed with :mod:`scipy.ndimage`), or any callable
            taking an array and an ``axis`` keyword.
        background: Parcellation label treated as background and excluded from
            the output. Defaults to ``0``. Pass ``None`` to keep every label.
        min_valid: Minimum number of finite elements a region needs before a
            value is computed. Regions below this return ``NaN``.
        drop_nonfinite: When ``True`` (default), ``NaN`` and ``Inf`` values in
            ``data`` are excluded from every summary. When ``False``, they are
            passed through to the reducer unchanged.
        propagate: When ``True``, also return a ``dense`` map with each region's
            summary broadcast onto all of its voxels/vertices. Useful for
            visualisation or treating ROI values as pseudo-continuous.

    Returns:
        A :class:`ParcelSummary` with ``labels``, ``values`` and, when
        ``propagate=True``, a ``dense`` map matching the input's spatial shape.

    Raises:
        ValueError: If the spatial shapes of ``data`` and ``parcellation``
            differ, or if ``method`` is not recognised.

    Examples:
        >>> values = parcel_reduce("thickness.func.gii", "atlas.label.gii").values
        >>> med = parcel_reduce(volume, atlas, method="median")
        >>> roi_map = parcel_reduce(data, atlas, propagate=True).dense
        >>> from scipy.stats import trim_mean
        >>> trimmed = parcel_reduce(
        ...     data, atlas, method=lambda a, axis: trim_mean(a, 0.1, axis=axis)
        ... )
    """
    if isinstance(method, str):
        if method not in _NDIMAGE_REDUCERS:
            raise ValueError(
                f"Unknown method '{method}'. Expected one of "
                f"{sorted(_NDIMAGE_REDUCERS)}, or a callable."
            )
    elif not callable(method):
        raise ValueError("'method' must be a string or a callable.")

    data_arr = load_data(data, dtype=np.float64).array
    labels_arr = np.rint(load_data(parcellation, np.int64).array)

    data_flat = _collapse(data_arr)
    labels_flat = _collapse(labels_arr)

    if labels_flat.shape[1] != 1:
        raise ValueError(
            "Parcellation must be a single volume or surface, got "
            f"{labels_flat.shape[1]} features."
        )
    labels_1d = labels_flat[:, 0]

    if data_flat.shape[0] != labels_1d.shape[0]:
        raise ValueError(
            "data and parcellation must cover the same elements: got "
            f"{data_flat.shape[0]} and {labels_1d.shape[0]}. Transform the map "
            "into the parcellation's space and density first."
        )

    unique = np.unique(labels_1d)
    if background is not None:
        unique = unique[unique != int(background)]

    n_features = data_flat.shape[1]
    values = np.full((unique.size, n_features), np.nan, dtype=float)

    if unique.size:
        with (
            np.errstate(invalid="ignore", divide="ignore"),
            warnings.catch_warnings(),
        ):
            # Empty or all-NaN regions are resolved to NaN via min_valid below;
            # scipy/numpy warnings about them would just be noise.
            warnings.simplefilter("ignore", RuntimeWarning)
            if isinstance(method, str):
                _reduce_ndimage(
                    values,
                    data_flat,
                    labels_1d,
                    unique,
                    method,
                    min_valid,
                    drop_nonfinite,
                )
            else:
                _reduce_callable(
                    values,
                    data_flat,
                    labels_1d,
                    unique,
                    method,
                    min_valid,
                    drop_nonfinite,
                )

    dense = (
        _propagate(data_arr, data_flat, labels_1d, unique, values)
        if propagate
        else None
    )

    if n_features == 1:
        values = values[:, 0]

    return ParcelSummary(labels=unique, values=values, dense=dense)


def _reduce_ndimage(
    values: np.ndarray,
    data_flat: np.ndarray,
    labels_1d: np.ndarray,
    unique: np.ndarray,
    method: str,
    min_valid: int,
    *,
    drop_nonfinite: bool,
) -> None:
    """Fill ``values`` in place using a :mod:`scipy.ndimage` reducer.

    Non-finite elements are excluded by relabelling them to a sentinel that is
    absent from ``index``, so ``ndimage`` skips them without touching the data.
    """
    reducer = _NDIMAGE_REDUCERS[method]
    sentinel = int(unique.max()) + 1
    for f in range(data_flat.shape[1]):
        col = data_flat[:, f]
        finite = np.isfinite(col)
        if drop_nonfinite:
            eff_labels = np.where(finite, labels_1d, sentinel)
        else:
            eff_labels = labels_1d
        counts = ndimage.sum_labels(finite.astype(float), labels_1d, index=unique)
        summarised = np.asarray(reducer(col, eff_labels, unique), dtype=float)
        values[:, f] = np.where(counts >= min_valid, summarised, np.nan)


def _reduce_callable(
    values: np.ndarray,
    data_flat: np.ndarray,
    labels_1d: np.ndarray,
    unique: np.ndarray,
    method: Callable[..., float],
    min_valid: int,
    drop_nonfinite: bool,
) -> None:
    """Fill ``values`` in place by applying a user callable per region.

    Groups elements with a single sort rather than rescanning the whole array
    once per region, which matters for volumetric data.
    """
    if drop_nonfinite:
        data_flat = np.where(np.isfinite(data_flat), data_flat, np.nan)

    order = np.argsort(labels_1d, kind="stable")
    sorted_labels = labels_1d[order]
    sorted_data = data_flat[order]
    starts = np.searchsorted(sorted_labels, unique, side="left")
    stops = np.searchsorted(sorted_labels, unique, side="right")

    for i, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        if stop <= start:
            continue
        block = sorted_data[start:stop]
        valid = np.sum(np.isfinite(block), axis=0)
        summarised = method(block, axis=0)
        values[i] = np.where(valid >= min_valid, summarised, np.nan)


def _propagate(
    data_arr: np.ndarray,
    data_flat: np.ndarray,
    labels_1d: np.ndarray,
    unique: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Broadcast per-region ``values`` back onto the input's spatial shape.

    Elements whose label is background or otherwise absent from ``unique``
    become ``NaN``.
    """
    dense_flat = np.full(data_flat.shape, np.nan, dtype=float)
    pos = np.searchsorted(unique, labels_1d)
    in_range = pos < unique.size
    matched = in_range & (unique[np.clip(pos, 0, unique.size - 1)] == labels_1d)
    dense_flat[matched] = values[pos[matched]]
    return dense_flat.reshape(data_arr.shape)
