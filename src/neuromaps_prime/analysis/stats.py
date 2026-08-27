"""Core statistical engine.

Provides functions for computing correlations, statistical tests, and
null-distribution metrics over pairs of neuroimaging maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, TypeGuard, get_args

import numpy as np
from scipy import special
from scipy import stats as sstats

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

__all__ = ["PermResult", "compare_images", "efficient_pearsonr", "permtest_metric"]

_METRIC_TYPE = Literal["pearsonr", "spearmanr"]
_NAN_POLICY_TYPE = Literal["propagate", "raise", "omit"]
_NAN_POLICY = get_args(_NAN_POLICY_TYPE)


def _chk2_asarray(
    a: ArrayLike, b: ArrayLike, *, axis: int | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert two inputs into 1-D NumPy arrays.

    Args:
        a: First array-like.
        b: Second array-like.
        axis: If ``None``, flatten both inputs; otherwise preserve shape.

    Returns:
        ``(a, b, out_axis)`` where both arrays are at least 1-D.

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
    nan_policy: _NAN_POLICY_TYPE = "propagate",
    return_pval: bool = True,
) -> tuple[np.ndarray | float, np.ndarray | float | None]:
    """Column-wise Pearson correlation and two-tailed p-values.

    Single-column inputs return scalars; multi-column inputs return
    1-D arrays.

    Args:
        a: Sample observations. Same row count as *b*.
        b: Sample observations. Same row count as *a*.
        ddof: Delta degrees-of-freedom for standard deviation.
        nan_policy: ``'propagate'``, ``'raise'``, or ``'omit'``.
        return_pval: Compute two-tailed p-values.

    Returns:
        ``(corr, pval)`` — Pearson's *r* clipped to ``[-1, 1]``, and
        p-values from the regularized incomplete beta function
        (or ``None`` if *return_pval* is ``False``).

    Raises:
        ValueError: On row count mismatch, unrecognized *nan_policy*,
        or NaN input with ``'raise'`` policy.

    Note:
        Computed as ``sum(zscore(a) * zscore(b), axis=0) / (n - 1)``.
        When *nan_policy* is ``'omit'``, arrays are masked where
        **either** contains NaN and per-column observation counts are
        used in the denominator.
    """
    if nan_policy not in _NAN_POLICY:
        raise ValueError(f'Value for nan_policy "{nan_policy}" not allowed')

    a, b, _ = _chk2_asarray(a, b, axis=0)
    if len(a) != len(b):
        raise ValueError(f"Arrays are not the same length ({len(a)} != {len(b)})")

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    # Reshape to 2-D: (n_observations, n_columns)
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)

    # Broadcast column counts if they differ
    if a.shape[1] != b.shape[1]:
        a, b = np.broadcast_arrays(a, b)

    mask = np.logical_or(np.isnan(a), np.isnan(b))
    if nan_policy == "raise" and np.any(mask):
        raise ValueError("Input contains NaN values")

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
        n_obs = a.shape[0]
        corr = np.sum(corr, axis=0)
    corr = np.squeeze(np.clip(corr / (n_obs - 1), -1.0, 1.0))

    if return_pval:
        ab = n_obs / 2.0 - 1.0
        pval = 2.0 * special.betainc(ab, ab, 0.5 * (1.0 - np.abs(corr)))
        return corr, pval
    return corr, None


def _compute_metric(
    a: np.ndarray,
    b: np.ndarray,
    *,
    metric: _METRIC_TYPE | Callable,
    nan_policy: _NAN_POLICY_TYPE,
) -> tuple[np.ndarray | float, np.ndarray, np.ndarray]:
    """Computes the observed similarity metric between `a` and `b`."""
    if callable(metric):
        mask = np.logical_or(np.isnan(a), np.isnan(b))
        if nan_policy == "raise" and np.any(mask):
            raise ValueError("Input contains NaN values")
        if nan_policy == "omit":
            a = np.ma.masked_array(a, mask, copy=False, fill_value=np.nan)
            b = np.ma.masked_array(b, mask, copy=False, fill_value=np.nan)
        corr = metric(a, b)
        if not isinstance(corr, np.ndarray | int | float):
            raise ValueError("Expected int, float, or np.ndarray output from metric")
        return corr, a, b

    if metric == "spearmanr":
        a = sstats.rankdata(a)
        b = sstats.rankdata(b)
    corr, _ = efficient_pearsonr(a, b, nan_policy=nan_policy, return_pval=False)
    return corr, a, b


def _permutation_indices(
    rng: np.random.Generator, n_perm: int, n_obs: int
) -> np.ndarray:
    """Generates random permutation indices for building a null distribution."""
    return np.argsort(rng.random((n_perm, n_obs)), axis=1)


def _null_distribution_callable(
    a: np.ndarray,
    b: np.ndarray,
    metric: Callable,
    perm_idx: np.ndarray | None,
    nulls: np.ndarray | None,
    n_perm: int,
    corr_shape: tuple[int, ...],
    nan_policy: _NAN_POLICY_TYPE,
) -> np.ndarray:
    """Builds a null distribution by looping a callable metric over permutations."""
    if (perm_idx is None) == (nulls is None):
        raise ValueError("Exactly one of `perm_idx` or `nulls` must be provided.")

    null_dist = np.empty((n_perm, *corr_shape))
    for perm in range(n_perm):
        a_permuted = (
            a[perm_idx[perm]] if nulls is None else nulls[:, perm]  # type: ignore[index] # perm_idx provided
        )
        corr, _, _ = _compute_metric(
            a_permuted, b, metric=metric, nan_policy=nan_policy
        )
        null_dist[perm] = corr
    return null_dist


def _null_distribution_pearsonr(
    a: np.ndarray,
    b: np.ndarray,
    perm_idx: np.ndarray | None,
    nulls: np.ndarray | None,
    nan_policy: _NAN_POLICY_TYPE,
) -> np.ndarray:
    """Builds null distribution of Pearson/Spearman correlations in vectorized call."""
    a_perm = nulls if nulls is not None else a[perm_idx].T
    null_dist, _ = efficient_pearsonr(
        a_perm, b[:, None], nan_policy=nan_policy, return_pval=False
    )
    if not isinstance(null_dist, np.ndarray):
        raise ValueError(f"Expected array output, got {type(null_dist)}")
    return null_dist


class PermResult(NamedTuple):
    """Result of a permutation-based similarity test.

    Attributes:
        similarity: Observed similarity metric between the two input maps.
            May be a scalar (single comparison) or an array (multiple
            comparisons).
        pvalue: Two-tailed non-parametric p-value estimated from the null
            distribution. The smallest achievable value is
            ``1 / (n_perm + 1)``.
        nulls: Null distribution of similarity metrics obtained under
            permutation, shape ``(n_perm,)``. Present only when
            *return_nulls* is ``True``; otherwise ``None``.
    """

    similarity: np.ndarray | float
    pvalue: np.ndarray | float
    nulls: np.ndarray | float | None = None


def _is_callable_metric(m: _METRIC_TYPE | Callable) -> TypeGuard[Callable]:
    """Return True if *m* is a callable metric rather than a string literal."""
    return callable(m)


def permtest_metric(
    a: ArrayLike,
    b: ArrayLike,
    *,
    metric: _METRIC_TYPE | Callable = "pearsonr",
    n_perm: int = 1000,
    seed: int | None = 0,
    nulls: ArrayLike | None = None,
    nan_policy: _NAN_POLICY_TYPE = "propagate",
    return_nulls: bool = False,
) -> PermResult:
    """Non-parametric p-value for the similarity of *a* and *b*.

    Shuffles *a* repeatedly to build an empirical null distribution
    against which the observed similarity is compared.

    Args:
        a: First sample, shape ``(N,)``.
        b: Second sample, shape ``(N,)``.
        metric: ``'pearsonr'``, ``'spearmanr'``, or a callable.
        n_perm: Number of permutations. Ignored if *nulls* is provided.
        seed: RNG seed; ``None`` for non-deterministic behavior.
        nulls: Precomputed null array, shape ``(N, P)``. Overrides
            *n_perm* when provided.
        nan_policy: How to handle NaN values.
        return_nulls: Include the null distribution in the result.

    Returns:
        :class:`PermResult` with the observed similarity, p-value,
        and optionally the null distribution.

    Raises:
        ValueError: If *nan_policy* is unrecognized, *a* and *b* have
            different lengths, or a callable *metric* returns an
            unsupported type.

    Note:
        Adapted from the neuromaps codebase
        (https://github.com/netneurolab/neuromaps/blob/ffcc2e0f657943ce00a1b6a968396f32250e495c/neuromaps/stats.py#L102).
    """
    if nan_policy not in _NAN_POLICY:
        raise ValueError(f'Value for nan_policy "{nan_policy}" not allowed')

    a, b, _ = _chk2_asarray(a, b, axis=0)
    if len(a) != len(b):
        raise ValueError("Provided arrays do not have same length")

    if a.size == 0 or b.size == 0:
        return PermResult(similarity=np.nan, pvalue=np.nan, nulls=np.nan)

    corr, a, b = _compute_metric(a, b, metric=metric, nan_policy=nan_policy)

    if nulls is not None:
        nulls = np.asarray(nulls)
        n_perm = nulls.shape[-1]

    corr = np.asarray(corr, dtype=np.float64)
    corr_abs = np.abs(corr)

    rng = np.random.default_rng(seed)
    perm_idx = _permutation_indices(rng, n_perm, len(a)) if nulls is None else None

    if _is_callable_metric(metric):
        null_dist = _null_distribution_callable(
            a,
            b,
            metric=metric,
            perm_idx=perm_idx,
            nulls=nulls,
            n_perm=n_perm,
            corr_shape=corr_abs.shape,
            nan_policy=nan_policy,
        )
    else:
        null_dist = _null_distribution_pearsonr(
            a, b, perm_idx=perm_idx, nulls=nulls, nan_policy=nan_policy
        )

    permutations = 1 + np.sum(np.abs(null_dist) >= corr_abs, axis=0)
    pvals = permutations / (n_perm + 1)

    if return_nulls:
        return PermResult(similarity=corr, pvalue=pvals, nulls=null_dist)
    return PermResult(similarity=corr, pvalue=pvals)


def _make_compare_mask(
    src: np.ndarray, trg: np.ndarray, *, ignore_zero: bool, nan_policy: _NAN_POLICY_TYPE
) -> np.ndarray:
    """Build a boolean mask of valid (non-zero, non-NaN) elements for comparison.

    Returns ``True`` for elements that should be **kept**.
    """
    nan_mask = np.isnan(src) | np.isnan(trg)
    if nan_policy == "raise" and np.any(nan_mask):
        raise ValueError("Inputs contain NaN values")

    if ignore_zero:
        zero_mask = np.isclose(src, 0) | np.isclose(trg, 0)
    else:
        zero_mask = np.zeros(len(src), dtype=bool)

    if nan_policy == "omit":
        return ~zero_mask & ~nan_mask
    # "propagate" or "raise" — only exclude zeros
    return ~zero_mask


def _compute_observed(
    src: np.ndarray, trg: np.ndarray, metric: _METRIC_TYPE | Callable
) -> np.ndarray | float:
    """Compute the observed similarity metric between two masked arrays."""
    if _is_callable_metric(metric):
        return metric(src, trg)
    if metric == "spearmanr":
        src, trg = sstats.rankdata(src), sstats.rankdata(trg)
    corr, _ = efficient_pearsonr(src, trg, return_pval=False)
    return corr


def compare_images(
    src: ArrayLike,
    trg: ArrayLike,
    *,
    metric: _METRIC_TYPE | Callable = "pearsonr",
    ignore_zero: bool = True,
    nulls: ArrayLike | None = None,
    nan_policy: _NAN_POLICY_TYPE = "omit",
    return_nulls: bool = False,
) -> PermResult:
    """Compare two maps and return a similarity metric.

    Masks out zero and/or NaN elements, computes *metric*, and when
    *nulls* is provided returns a non-parametric p-value via
    :func:`permtest_metric`.

    Args:
        src: First map, shape ``(N,)``.
        trg: Second map, shape ``(N,)``.
        metric: ``'pearsonr'``, ``'spearmanr'``, or a callable.
        ignore_zero: Exclude near-zero elements. Default ``True``.
        nulls: Precomputed null data, shape ``(N, P)``. If ``None``,
            returns ``np.nan`` for the p-value.
        nan_policy: How to handle NaN values. Default ``'omit'``.
        return_nulls: Include the null distribution in the result.

    Returns:
        :class:`PermResult` with the observed similarity and p-value.

    Raises:
        ValueError: If *nan_policy* is unrecognized, *src* and *trg*
            have different lengths, or ``return_nulls`` is ``True``
            without providing *nulls*.

    Note:
        Adapted from the neuromaps codebase
        (https://github.com/netneurolab/neuromaps/blob/ffcc2e0f657943ce00a1b6a968396f32250e495c/neuromaps/stats.py#L14).
    """
    if return_nulls and nulls is None:
        raise ValueError("`return_nulls` cannot be True when `nulls` is None.")
    if nan_policy not in _NAN_POLICY:
        raise ValueError(f'Value for nan_policy "{nan_policy}" not allowed')

    if _is_callable_metric(metric):
        if not np.isscalar(metric([1, 1], [1, 1])):
            raise ValueError(
                "Provided callable `metric` must accept two inputs and return "
                "a single scalar value."
            )
    elif metric not in ("pearsonr", "spearmanr"):
        raise ValueError(
            f"Expected 'pearsonr', 'spearmanr', or a callable — got: {metric!r}"
        )

    src = np.asarray(src)
    trg = np.asarray(trg)

    if len(src) != len(trg):
        raise ValueError(f"Arrays are not the same length ({len(src)} != {len(trg)})")

    mask = _make_compare_mask(src, trg, ignore_zero=ignore_zero, nan_policy=nan_policy)
    src_m, trg_m = src[mask], trg[mask]

    if src_m.size == 0 or trg_m.size == 0:
        return PermResult(similarity=np.nan, pvalue=np.nan)

    if nulls is not None:
        null_arr = np.asarray(nulls)[mask]
        return permtest_metric(
            src_m,
            trg_m,
            metric=metric,
            nulls=null_arr,
            nan_policy=nan_policy,
            return_nulls=return_nulls,
        )
    return PermResult(similarity=_compute_observed(src_m, trg_m, metric), pvalue=np.nan)
