"""Core statistical engine.

Provides functions for computing correlations, statistical tests, and
null-distribution metrics over pairs of neuroimaging maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import special
from scipy import stats as sstats

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["efficient_pearsonr"]

_NAN_POLICY = ("propagate", "raise", "omit")


def _chk2_asarray(
    a: ArrayLike,
    b: ArrayLike,
    *,
    axis: int | None = None,
) -> tuple[NDArray[Any], NDArray[Any], int]:
    """Validate and convert two input sequences into NumPy arrays.

    When *axis* is ``None``, both inputs are flattened to 1-D and the
    output axis is set to ``0``.  Otherwise the inputs are kept in their
    original shape.  Scalar-like inputs are always promoted to at least
    1-D arrays.

    This function was part of the (now removed) private API of scipy
    (https://github.com/scipy/scipy/pull/23088).

    Args:
        a: First input array or array-like object.
        b: Second input array or array-like object.
        axis: Axis along which to operate.  If ``None`` (default), both
            inputs are flattened.

    Returns:
        A tuple of ``(a, b, out_axis)`` where both arrays are at least
        1-D and ``out_axis`` is the axis to pass to downstream
        operations.

    Note:
        Lifted from the neuromaps codebase
        (https://github.com/netneurolab/neuromaps/blob/ffcc2e0f657943ce00a1b6a968396f32250e495c/neuromaps/utils.py#L131).
    """
    out_axis: int

    if axis is None:
        a = np.ravel(a)
        b = np.ravel(b)
        out_axis = 0
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        out_axis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)
    if b.ndim == 0:
        b = np.atleast_1d(b)

    return a, b, out_axis


def efficient_pearsonr(
    a: ArrayLike,
    b: ArrayLike,
    *,
    ddof: int = 1,
    nan_policy: Literal["propagate", "raise", "omit"] = "propagate",
    return_pval: bool = True,
) -> tuple[NDArray[Any] | float, NDArray[Any] | float | None]:
    """Compute column-wise Pearson correlation and two-tailed p-values.

    Computes Pearson's *r* between each pair of matching columns in *a*
    and *b*.  Single-column inputs are returned as scalars (0-D arrays);
    multi-column inputs return 1-D arrays.

    Args:
        a: Sample observations. Must have the same number of rows as *b*.
        b: Sample observations. Must have the same number of rows as *a*.
            The number of columns must match *a*, or be broadcastable.
        ddof: Delta degrees-of-freedom applied to the standard deviation
            in the z-score normalization step. Default is ``1``
            (unbiased estimator).
        nan_policy: How to handle NaN values in the inputs.

            - ``'propagate'``: return NaN wherever a NaN appears in
              either input.
            - ``'raise'``: raise ``ValueError`` if either input contains
              NaN.
            - ``'omit'``: ignore NaN entries when computing means,
              standard deviations, and correlations.

            Default is ``'propagate'``.
        return_pval: Whether to compute and return two-tailed p-values.
            Default is ``True``. When ``False``, the second element of
            the returned tuple is ``None``.

    Returns:
        A tuple ``(corr, pval)`` where:

            corr: Pearson correlation coefficient for each column pair,
                clipped to ``[-1, 1]`` to guard against floating-point
                drift. Returned as ``np.nan`` when either input is empty.
            pval: Two-tailed p-values computed via the regularized
                incomplete beta function. Returned as ``None`` when
                *return_pval* is ``False``. Returned as ``np.nan`` when
                either input is empty.

    Raises:
        ValueError: If *a* and *b* have different numbers of rows, if
            *nan_policy* is ``'raise'`` and either input contains NaN,
            or if *nan_policy* is not one of the recognized values.

    Note:
        The correlation is computed as::

            corr = sum(zscore(a) * zscore(b), axis=0) / (n - 1)

        where *zscore* centres and normalizes each column using the
        sample mean and standard deviation (corrected by *ddof*).  The
        p-value is derived from the relationship between Pearson's *r*
        and the beta distribution.

        When *nan_policy* is ``'omit'``, both arrays are masked at
        positions where **either** contains NaN, and per-column
        observation counts are used in the denominator and p-value
        calculation.
    """
    if nan_policy not in _NAN_POLICY:
        raise ValueError(f'Value for nan_policy "{nan_policy}" not allowed')

    a, b, _ = _chk2_asarray(a, b, axis=0)
    if len(a) != len(b):
        raise ValueError(f"Arrays are not the same length ({len(a)} != {len(b)})")

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    mask = np.logical_or(np.isnan(a), np.isnan(b))
    if nan_policy == "raise" and np.any(mask):
        raise ValueError("Input contains NaN values")

    # Reshape to 2-D: (n_observations, n_columns)
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)

    # Broadcast column counts if they differ
    if a.shape[1] != b.shape[1]:
        a, b = np.broadcast_arrays(a, b)

    # NaN handling - avoid making copies of data
    if nan_policy == "omit":
        a = np.ma.masked_array(a, mask, copy=False, fill_value=np.nan)
        b = np.ma.masked_array(b, mask, copy=False, fill_value=np.nan)

    # Correlation
    with np.errstate(invalid="ignore"):
        corr = sstats.zscore(a, ddof=ddof, nan_policy=nan_policy) * sstats.zscore(
            b, ddof=ddof, nan_policy=nan_policy
        )

    if nan_policy == "omit":
        corr = corr.filled(np.nan)
        n_obs = np.sum(np.logical_not(np.isnan(corr)), axis=0)
        corr = np.nansum(corr, axis=0)
    else:
        n_obs = len(a)
        corr = np.sum(corr, axis=0)
    corr = np.squeeze(np.clip(corr / (n_obs - 1), -1.0, 1.0))

    if return_pval:
        ab = n_obs / 2.0 - 1.0
        pval = 2.0 * special.betainc(ab, ab, 0.5 * (1.0 - np.abs(corr)))
        return corr, pval
    return corr, None
