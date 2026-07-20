"""Core statistical engine.

Provides functions for computing correlations, statistical tests, and
null-distribution metrics over pairs of neuroimaging maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
from scipy import special
from scipy import stats as sstats

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

__all__ = ["efficient_pearsonr", "permtest_metric"]

_METRIC_TYPE = Literal["pearsonr", "spearmanr"]
_NAN_POLICY_TYPE = Literal["propagate", "raise", "omit"]
_NAN_POLICY = get_args(_NAN_POLICY_TYPE)


def _chk2_asarray(
    a: ArrayLike,
    b: ArrayLike,
    *,
    axis: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
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
    nan_policy: _NAN_POLICY_TYPE = "propagate",
    return_pval: bool = True,
) -> tuple[np.ndarray | float, np.ndarray | float | None]:
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
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float | None]:
    """Computes a non-parametric p-value for the similarity of `a` and `b`.

    Calculates a two-tailed p-value for the hypothesis that samples `a`
    and `b` are related, using a permutation test: `a` is repeatedly
    shuffled and re-correlated with `b` to build an empirical null
    distribution, against which the observed statistic is compared.

    Args:
        a: First sample of observations, shape (N,).
        b: Second sample of observations, shape (N,). Must have the
            same length as `a`.
        metric: Similarity metric used to compare `a` and `b`. One of
            'pearsonr', 'spearmanr', or a callable accepting two inputs
            and returning a single similarity value. Default 'pearsonr'.
        n_perm: Number of permutations to evaluate. Unless `a` and `b`
            are very small this approximates a randomization test via
            Monte Carlo simulation. Ignored if `nulls` is provided.
            Default 1000.
        seed: Seed for random number generation. Set to `None` for
            nondeterministic behavior. Default 0.
        nulls: Precomputed null array used in place of shuffled `a` to
            compute the null distribution of correlations, shape
            `(N, P)`. Must have the same length as `a` and `b`.
            Providing this overrides `n_perm`. When not specified, a
            standard permutation is used to shuffle `a`. Default None.
        nan_policy: How to handle NaN values in the inputs. One of
            'propagate' (return NaN), 'raise' (raise an error), or
            'omit' (ignore NaN values in the calculation). Default
            'propagate'.
        return_nulls: Whether to also return the null distribution of
            similarity metrics. Default False.

    Returns:
        A tuple `(corr, pvals, null_dist)` where:

            corr: The observed similarity metric.
            pvals: Two-tailed non-parametric p-value(s). The smallest
                value this can take is `1 / (n_perm + 1)`.
            null_dist: Null distribution of similarity metrics, shape
                `(n_perm,)`. `None` unless `return_nulls` is True.

        Returns `(np.nan, np.nan)` if either input is empty.

    Raises:
        ValueError: If `nan_policy` is not a recognized value, if `a`
            and `b` have different lengths, or if a callable `metric`
            returns an unsupported type.
    """
    # Catch invalid NaN policy early
    if nan_policy not in _NAN_POLICY:
        raise ValueError(f'Value for nan_policy "{nan_policy}" not allowed')

    a, b, _ = _chk2_asarray(a, b, axis=0)
    if len(a) != len(b):
        raise ValueError("Provided arrays do not have same length")

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan, np.nan

    _is_callable_metric = callable(metric)
    corr, a, b = _compute_metric(a, b, metric=metric, nan_policy=nan_policy)

    if nulls is not None:
        nulls = np.asarray(nulls)
        n_perm = nulls.shape[-1]

    corr = np.asarray(corr, dtype=np.float64)
    corr_abs = np.abs(corr)

    rng = np.random.default_rng(seed)
    perm_idx = _permutation_indices(rng, n_perm, len(a)) if nulls is None else None

    if _is_callable_metric:
        null_dist = _null_distribution_callable(
            a,
            b,
            metric=metric,  # type: ignore[arg-type] # non-string metric
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
        return corr, pvals, null_dist
    return corr, pvals, None
