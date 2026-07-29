"""Surrogate map generation following Burt et al., 2018.

Implements the spatial-autocorrelation-preserving surrogacy method from

    Burt, S. M. et al. (2018). *Nature Neuroscience*, 21, pp. 642.

The core idea is to generate surrogate cortical maps that preserve the
spatial autocorrelation structure of the original data while destroying
any meaningful signal.  This module provides the weight-matrix building
block used by the higher-level Burt2018 surrogate pipeline.

Adapted from the neuromaps codebase
(https://github.com/netneurolab/neuromaps/blob/main/neuromaps/nulls/burt.py).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares
from scipy.stats import boxcox

__all__ = ["batch_surrogates", "estimate_rho_d0", "make_surrogate"]


class SurrogateResult(NamedTuple):
    """Result of surrogate map generation.

    Attributes:
        surrogate: The surrogate cortical map, shape ``(n,)``, preserving
            the marginal distribution of the original data.
        order: Rank-order array if ``return_order`` was ``True``,
            otherwise ``None``.
        params: The ``(rho, d0)`` tuple used to build the weight matrix
            if ``return_params`` was ``True``, otherwise ``None``.
    """

    surrogate: np.ndarray
    order: np.ndarray | None
    params: tuple[float, float] | None


def _make_weight_matrix(x: np.ndarray, d0: float) -> np.ndarray:
    """Build a row-normalized spatial weight matrix from a distance matrix.

    Each off-diagonal entry ``(i, j)`` receives an exponential-decay
    weight ``exp(-d[i, j] / d0)`` where ``d0`` controls the scale of
    spatial autocorrelation.  The diagonal is zeroed so that a vertex
    never weighs against itself, then each row is normalized to sum to
    one.

    Args:
        x: Square distance matrix of shape ``(n, n)``.  Values should
            be non-negative distances (geodesic or Euclidean).
        d0: Autocorrelation scale parameter.  Smaller values concentrate
            weight on nearby neighbours; larger values spread weight more
            diffusely.

    Returns:
        Row-normalized weight matrix of shape ``(n, n)`` with zeros on
        the diagonal and each row summing to ``1.0``.

    Note:
        Overflow (``exp(-x/d0)`` when distances are very small relative
        to ``d0``) and invalid (``0/0`` during row normalization) warnings
        are suppressed because these edge cases resolve naturally under
        numpy's floating-point semantics.
    """
    x = np.asarray(x)
    with np.errstate(over="ignore"):
        weight = np.exp(-x / d0)
    np.fill_diagonal(weight, 0)

    with np.errstate(invalid="ignore"):
        return weight / weight.sum(axis=1, keepdims=True)


def estimate_rho_d0(
    x: np.ndarray, y: np.ndarray, *, rho: float | None = None, d0: float | None = None
) -> tuple[float, float]:
    """Estimate the spatial autoregressive parameters ``rho`` and ``d0``.

    Fits a spatial autoregressive (SAR) model in which the expected
    value of the data vector is ``rho * W(d0) @ y``, where ``W(d0)`` is
    the row-normalized exponential-decay weight matrix built from the
    distance matrix ``x`` and the scale parameter ``d0``.  The
    Levenberg-Marquardt algorithm minimises the residual
    ``y - rho * W @ y`` with respect to both ``rho`` and ``d0``.

    Before optimisation the data vector is Box-Cox transformed to
    approximate normality and centred by subtracting its mean.

    Args:
        x: Square distance matrix of shape ``(n, n)``.
        y: 1-D brain-imaging data vector of length ``n``.  All values
            must be strictly positive (requirement of the Box-Cox
            transformation).
        rho: Initial guess for the spatial autocorrelation coefficient.
            If ``None`` defaults to ``1.0``, which assumes strong spatial
            dependence.
        d0: Initial guess for the spatial decay scale.  Controls the
            distance over which neighbours remain influential.  If
            ``None`` defaults to ``1.0``.

    Returns:
        A tuple of ``(rho_hat, d0_hat)`` — the fitted autocorrelation
        coefficient and decay scale from the least-squares optimiser.
    """

    def _estimate(parameters: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute residuals between observed and SAR-predicted values."""
        rho_p, d0_p = parameters
        y_hat = rho_p * (_make_weight_matrix(x, d0_p) @ y)
        return y - y_hat

    rho = 1.0 if rho is None else rho
    d0 = 1.0 if d0 is None else d0

    y, *_ = boxcox(y)
    y -= y.mean()
    result = least_squares(_estimate, [rho, d0], args=(x, y), method="lm")
    return tuple(result.x)


def make_surrogate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rho: float | None = None,
    d0: float | None = None,
    seed: int | None = None,
    return_order: bool = False,
    return_params: bool = False,
) -> SurrogateResult:
    """Generate a single surrogate map that preserves spatial autocorrelation.

    Solves ``(I - rho * W) s = u`` for a vector of standard-normal draws
    ``u``, where ``W`` is the exponential-decay weight matrix built from
    the distance matrix ``x``.  The raw solution is rank-ordered and then
    the sorted values of ``y`` are assigned back, so the surrogate shares
    the exact marginal distribution of the original data while retaining
    its spatial autocorrelation structure.

    Args:
        x: Square distance matrix of shape ``(n, n)``.
        y: 1-D brain-imaging data vector of length ``n``.
        rho: Spatial autocorrelation coefficient.  If ``None`` will be
            estimated from the data via :func:`estimate_rho_d0`.
        d0: Spatial decay scale.  If ``None`` will be estimated from the
            data via :func:`estimate_rho_d0`.
        seed: Seed for the random number generator.  Set to ``None`` for
            non-deterministic surrogates.
        return_order: If ``True``, also return the rank-order array used
            to rank-match the surrogate.
        return_params: If ``True``, also return the ``(rho, d0)`` tuple
            that was used to build the weight matrix.

    Returns:
        A :class:`SurrogateResult` namedtuple with fields:

        - ``surrogate``: the surrogate map of shape ``(n,)``.
        - ``order``: rank-order array if ``return_order`` is ``True``,
          otherwise ``None``.
        - ``params``: ``(rho, d0)`` tuple if ``return_params`` is ``True``,
          otherwise ``None``.
    """
    rs = np.random.default_rng(seed)

    if rho is None or d0 is None:
        rho, d0 = estimate_rho_d0(x, y, rho=rho, d0=d0)

    w = _make_weight_matrix(x, d0)
    u = rs.standard_normal(len(x))
    iw = -rho * w
    np.fill_diagonal(iw, 1.0)
    surr = np.linalg.solve(iw, u)

    order = surr.argsort()
    surr[order] = np.sort(y)
    return SurrogateResult(
        surrogate=surr,
        order=order if return_order else None,
        params=(rho, d0) if return_params else None,
    )


def batch_surrogates(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rho: float | None = None,
    d0: float | None = None,
    seed: int | None = None,
    n_surr: int = 1_000,
    n_jobs: int = 1,
) -> np.ndarray:
    """Generate multiple surrogate maps in parallel.

    Estimates ``rho`` and ``d0`` once from the data, builds the system
    matrix ``(I - rho * W)``, and then spawns ``n_surr`` independent
    surrogate draws by solving the linear system.  If more than half the
    entries of the system matrix are near-zero it is converted to a
    sparse CSR matrix for faster solves.

    Args:
        x: Square distance matrix of shape ``(n, n)``.
        y: 1-D brain-imaging data vector of length ``n``.
        rho: Spatial autocorrelation coefficient.  If ``None`` will be
            estimated from the data via :func:`estimate_rho_d0`.
        d0: Spatial decay scale.  If ``None`` will be estimated from the
            data via :func:`estimate_rho_d0`.
        seed: Seed for the master random number generator used to produce
            per-surrogate seeds.  Set to ``None`` for non-deterministic
            surrogates.
        n_surr: Number of surrogate maps to generate.
        n_jobs: Number of parallel workers for surrogate generation.
            Set to ``-1`` to use all available CPU cores.

    Returns:
        Array of shape ``(n, n_surr)`` where each column is an independent
        surrogate map.
    """

    def _quick_surr(
        iw_or_lu: tuple[np.ndarray, np.ndarray] | sparse.csr_matrix,
        ysort: np.ndarray,
        *,
        seed: int,
    ) -> np.ndarray:
        """Generate a single surrogate given a system matrix or LU factorization."""
        rs = np.random.default_rng(seed)
        u = rs.standard_normal(len(ysort))
        surr = (
            sparse.linalg.spsolve(iw_or_lu, u)
            if sparse.issparse(iw_or_lu)
            else lu_solve(iw_or_lu, u)
        )
        surr[surr.argsort()] = ysort
        return surr

    if n_surr <= 0:
        raise ValueError("n_surr must be positive")

    rs = np.random.default_rng(seed)
    seeds = rs.integers(np.iinfo(np.int32).max, size=n_surr, dtype=np.int32)

    if rho is None or d0 is None:
        rho, d0 = estimate_rho_d0(x, y, rho=rho, d0=d0)
    iw = -rho * _make_weight_matrix(x, d0)
    np.fill_diagonal(iw, 1.0)
    zeros = np.isclose(iw, 0)
    if (zeros.sum() / iw.size) > 0.5:
        iw[zeros] = 0
        iw = sparse.csr_matrix(iw)
    else:
        iw = lu_factor(iw)
    ysort = np.sort(y)

    if n_jobs != 1:
        surrs = Parallel(n_jobs=n_jobs)(
            delayed(_quick_surr)(iw, ysort, seed=seed) for seed in seeds
        )
    else:
        surrs = [_quick_surr(iw, ysort, seed=seed) for seed in seeds]

    return np.column_stack(surrs)
