"""Reduce continuous brain maps to region-of-interest summaries."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

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
    """

    labels: np.ndarray
    values: np.ndarray


_REDUCERS: dict[str, Callable] = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "sum": np.nansum,
    "std": np.nanstd,
    "min": np.nanmin,
    "max": np.nanmax,
}


def _collapse(arr: np.ndarray) -> np.ndarray:
    """Collapse spatial dimensions, preserving a trailing feature axis.

    Surface data are ``(n_vertices,)`` or ``(n_vertices, n_features)``.
    Volumetric data are ``(i, j, k)`` or ``(i, j, k, n_features)``.
    """
    if arr.ndim not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported array with {arr.ndim} dimensions.")
    out = arr.reshape(-1, *arr.shape[2:])
    return out[:, None] if out.ndim == 1 else out


def parcel_reduce(
    data: ArrayLike | str | Path,
    parcellation: ArrayLike | str | Path,
    *,
    method: ReduceMethod = "mean",
    background: float | int | None = 0,
    min_valid: int = 1,
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
            ``'max'``, or any callable taking an array and an ``axis`` keyword.
        background: Parcellation label treated as background and excluded from
            the output. Defaults to ``0``. Pass ``None`` to keep every label.
        min_valid: Minimum number of non-NaN elements a region needs before a
            value is computed. Regions below this return ``NaN``.

    Returns:
        A :class:`ParcellatedData` with ``labels`` and ``values``.

    Raises:
        ValueError: If the spatial shapes of ``data`` and ``parcellation``
            differ, or if ``method`` is not recognised.

    Examples:
        >>> values = parcellate("thickness.func.gii", "atlas.label.gii").values
        >>> med = parcellate(volume, atlas, method="median")
        >>> from scipy.stats import trim_mean
        >>> trimmed = parcellate(
        ...     data, atlas, method=lambda a, axis: trim_mean(a, 0.1, axis=axis)
        ... )
    """
    if isinstance(method, str):
        if method not in _REDUCERS:
            raise ValueError(
                f"Unknown method '{method}'. Expected one of "
                f"{sorted(_REDUCERS)}, or a callable."
            )
        reducer = _REDUCERS[method]
    elif callable(method):
        reducer = method
    else:
        raise ValueError("'method' must be a string or a callable.")

    data_arr = load_data(data, dtype=np.float64)
    labels_arr = load_data(parcellation, np.int64)

    data_flat = _collapse(data_arr)
    labels_flat = _collapse(labels_arr)

    if labels_flat.shape[1] != 1:
        raise ValueError(
            "Parcellation must be a single volume or surface, got "
            f"{labels_flat.shape[1]} features."
        )
    labels_1d = np.rint(labels_flat[:, 0])

    if data_flat.shape[0] != labels_1d.shape[0]:
        raise ValueError(
            "data and parcellation must cover the same elements: got "
            f"{data_flat.shape[0]} and {labels_1d.shape[0]}. Transform the map "
            "into the parcellation's space and density first."
        )

    unique = np.unique(labels_1d)
    if background is not None:
        unique = unique[unique != int(background)]
    unique = unique[~np.isnan(unique.astype(float))]

    n_features = data_flat.shape[1]
    values = np.full((unique.size, n_features), np.nan, dtype=float)

    # Group elements by label with a single sort rather than rescanning the
    # whole array once per region, which matters for volumetric data.
    order = np.argsort(labels_1d, kind="stable")
    sorted_labels = labels_1d[order]
    sorted_data = data_flat[order]
    starts = np.searchsorted(sorted_labels, unique, side="left")
    stops = np.searchsorted(sorted_labels, unique, side="right")

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        # Regions that are entirely NaN yield NaN by design; numpy's
        # "Mean of empty slice" warning would be noise here.
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, (start, stop) in enumerate(zip(starts, stops, strict=True)):
            if stop <= start:
                continue
            block = sorted_data[start:stop]
            valid = np.sum(np.isfinite(block), axis=0)
            summarised = reducer(block, axis=0)
            values[i] = np.where(valid >= min_valid, summarised, np.nan)

    if n_features == 1:
        values = values[:, 0]

    return ParcelSummary(labels=unique, values=values)
