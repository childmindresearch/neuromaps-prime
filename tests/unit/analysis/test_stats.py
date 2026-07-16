"""Tests for the statistical utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import stats as sstats

from neuromaps_prime.analysis import stats

if TYPE_CHECKING:
    from numpy.random import Generator


# Fixtures
@pytest.fixture(scope="module")
def rng() -> Generator:
    """Deterministic random number generator."""
    return np.random.default_rng(12345)


@pytest.fixture(scope="module")
def random_vectors(rng: Generator) -> np.ndarray:
    """Random vectors for correlation testing."""
    return rng.normal(size=(2, 100))


@pytest.fixture(scope="module")
def random_matrix(rng: Generator) -> np.ndarray:
    """Random two-column matrix."""
    return rng.normal(size=(100, 2))


@pytest.fixture(scope="module")
def perfectly_correlated() -> tuple[np.ndarray, np.ndarray]:
    """Perfectly correlated vectors."""
    x = np.arange(10)
    return x, x.copy()


@pytest.fixture(scope="module")
def vectors_with_nan() -> tuple[np.ndarray, np.ndarray]:
    """Vectors containing one NaN."""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return x, y


@pytest.fixture(scope="module")
def precomputed_nulls(rng: Generator) -> np.ndarray:
    """Precomputed null distribution."""
    return rng.normal(size=(100, 25))


class TestChk2Asarray:
    """Tests for _chk2_asarray()."""

    def test_axis_none_flattens(self) -> None:
        """Verify axis=None, flattens both inputs."""
        a = np.arange(6).reshape(2, 3)
        b = np.arange(6).reshape(2, 3)
        aa, bb, axis = stats._chk2_asarray(a, b, axis=None)
        assert aa.ndim == 1
        assert bb.ndim == 1
        assert axis == 0

    def test_axis_preserves_shape(self) -> None:
        """Verify specifying axis preserves input shapes."""
        a = np.arange(6).reshape(2, 3)
        b = np.arange(6).reshape(2, 3)
        aa, bb, axis = stats._chk2_asarray(a, b, axis=1)
        assert aa.shape == (2, 3)
        assert bb.shape == (2, 3)
        assert axis == 1

    def test_promotes_scalars(self) -> None:
        """Verify scalar inputs promoted to 1-D arrays."""
        a, b, _ = stats._chk2_asarray(1, 2)
        assert a.shape == (1,)
        assert b.shape == (1,)


class TestEfficientPearsonR:
    """Tests for efficient_pearsonr()."""

    def test_matches_scipy(self, random_vectors: np.ndarray) -> None:
        """Verify agreement with scipy.stats.pearsonr()."""
        x, y = random_vectors
        corr, pval = stats.efficient_pearsonr(x, y)
        expected_corr, expected_p = sstats.pearsonr(x, y)
        assert corr == pytest.approx(expected_corr)
        assert pval == pytest.approx(expected_p)

    def test_return_pval_false(self, perfectly_correlated: np.ndarray) -> None:
        """Verify p-values omitted when requested."""
        corr, pval = stats.efficient_pearsonr(*perfectly_correlated, return_pval=False)
        assert corr == pytest.approx(1.0)
        assert pval is None

    @pytest.mark.parametrize(
        ("nan_policy", "expect_nan"), [("propagate", True), ("omit", False)]
    )
    def test_nan_policies(
        self,
        *,
        vectors_with_nan: tuple[np.ndarray, ...],
        nan_policy: stats._NAN_POLICY_TYPE,
        expect_nan: bool,
    ) -> None:
        """Verify supported NaN handling policy."""
        corr, _ = stats.efficient_pearsonr(*vectors_with_nan, nan_policy=nan_policy)

        if expect_nan:
            assert np.isnan(corr)
        else:
            assert corr == pytest.approx(1.0)

    def test_nan_policy_raise(
        self, vectors_with_nan: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify nan_policy='raise' rejects NaN-containing inputs."""
        with pytest.raises(ValueError, match="NaN"):
            stats.efficient_pearsonr(*vectors_with_nan, nan_policy="raise")

    def test_invalid_nan_policy(self) -> None:
        """Verify invalid NaN policy raises ValueError."""
        with pytest.raises(ValueError, match="nan_policy"):
            stats.efficient_pearsonr([1, 2], [1, 2], nan_policy="invalid")

    def test_length_mismatch(self) -> None:
        """Verify inputs of different lenghts raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            stats.efficient_pearsonr(np.arange(5), np.arange(4))

    def test_empty_arrays(self) -> None:
        """Verify empty inputs return NaN outputs."""
        corr, pval = stats.efficient_pearsonr([], [])
        assert np.isnan(corr)
        assert np.isnan(pval)

    def test_broadcast_columns(self) -> None:
        """Verify single-column inputs broadcast across multiple columns."""
        x = np.arange(5)
        y = np.column_stack((x, x[::-1]))
        corr, _ = stats.efficient_pearsonr(x, y)
        np.testing.assert_allclose(corr, [1.0, -1.0])


class TestComputeMetric:
    """Tests for _compute_metric()."""

    def test_callable_metric(self) -> None:
        """Verify callable metrics evaluated correctly."""

        def metric(a: int, b: int) -> int | float | np.ndarray:
            return float(np.sum(a - b))

        corr, _, _ = stats._compute_metric(
            np.arange(5), np.arange(5), metric=metric, nan_policy="propagate"
        )
        assert corr == 0

    def test_invalid_callable_output(self) -> None:
        """Verify invalid callable return types raise ValueError."""

        def metric(a: int, b: int) -> str:  # noqa: ARG001 # unused, necessary params
            return "invalid"

        with pytest.raises(ValueError, match="Expected"):
            stats._compute_metric(
                np.arange(5), np.arange(5), metric=metric, nan_policy="propagate"
            )

    def test_spearman_matches_scipy(
        self, random_vectors: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify Spearman metric agrees with SciPy."""
        x, y = random_vectors
        corr, _, _ = stats._compute_metric(
            x, y, metric="spearmanr", nan_policy="propagate"
        )
        expected, _ = sstats.spearmanr(x, y)
        assert corr == pytest.approx(expected)


class TestPermutationIndices:
    """Tests for _permutation_indices()."""

    def test_shape(self) -> None:
        """Verify permutation index array has expected shape."""
        rng = np.random.default_rng(0)
        idx = stats._permutation_indices(rng, n_perm=20, n_obs=7)
        assert idx.shape == (20, 7)

    def test_every_row_is_permutation(self) -> None:
        """Verify each row contains valid permutation of indices."""
        rng = np.random.default_rng(0)
        idx = stats._permutation_indices(rng, n_perm=20, n_obs=7)
        expected = np.arange(7)
        for row in idx:
            np.testing.assert_array_equal(np.sort(row), expected)


class TestPermtestMetric:
    """Tests for permtest_metric()."""

    def test_reproducible_seed(
        self, random_vectors: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify fixed seed produces reproducible results."""
        x, y = random_vectors
        result1 = stats.permtest_metric(x, y, seed=42)
        result2 = stats.permtest_metric(x, y, seed=42)
        assert result1 == pytest.approx(result2)

    def test_return_null_distribution(
        self, random_vectors: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify null distribution returned when requested."""
        x, y = random_vectors
        corr, pval, nulls = stats.permtest_metric(x, y, n_perm=20, return_nulls=True)
        assert np.ndim(corr) == 0
        assert np.ndim(pval) == 0
        assert isinstance(nulls, np.ndarray)
        assert nulls.shape == (20,)

    def test_precomputed_nulls(
        self,
        random_vectors: tuple[np.ndarray, np.ndarray],
        precomputed_nulls: np.ndarray,
    ) -> None:
        """Verify precomputed null distributions are used."""
        x, y = random_vectors
        _, _, nulls = stats.permtest_metric(
            x, y, nulls=precomputed_nulls, return_nulls=True
        )
        assert isinstance(nulls, np.ndarray)
        assert nulls.shape == (25,)

    def test_precomputed_nulls_path(self) -> None:
        """Verify precomputed nulls are used for Pearson null distributions."""
        a = np.arange(5)
        b = np.arange(5)
        nulls = np.column_stack([np.arange(5), np.arange(5)[::-1]])
        result = stats._null_distribution_pearsonr(
            a, b, perm_idx=None, nulls=nulls, nan_policy="propagate"
        )
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [1.0, -1.0])

    def test_callable_metric(
        self, random_vectors: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify callable metrics supported."""
        x, y = random_vectors

        def metric(a: int, b: int) -> np.ndarray:
            return np.corrcoef(a, b)[0, 1]

        corr, pval, _ = stats.permtest_metric(x, y, metric=metric, n_perm=20)
        assert -1 <= corr <= 1
        assert 0 <= pval <= 1

    def test_callable_metric_nan_policy_raise(self) -> None:
        """Verify callable metrics respect NaN rejection."""

        def metric(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.corrcoef(a, b)[0, 1])

        with pytest.raises(ValueError, match="NaN"):
            stats._compute_metric(
                np.array([1.0, np.nan]),
                np.array([1.0, 2.0]),
                metric=metric,
                nan_policy="raise",
            )

    def test_callable_metric_with_precomputed_nulls(
        self, random_vectors: np.ndarray, precomputed_nulls: np.ndarray
    ) -> None:
        """Verify callable metrics work with supplied null distributions."""
        x, y = random_vectors

        def metric(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.corrcoef(a, b)[0, 1])

        _, _, null_dist = stats.permtest_metric(
            x, y, metric=metric, nulls=precomputed_nulls, return_nulls=True
        )
        assert isinstance(null_dist, np.ndarray)
        assert null_dist.shape == (25,)

    def test_invalid_nan_policy(self) -> None:
        """Verify invalid NaN policy raises ValueError."""
        with pytest.raises(ValueError, match="nan_policy"):
            stats.permtest_metric([1, 2], [1, 2], nan_policy="invalid")

    def test_length_mismatch(self) -> None:
        """Verify inputs of different lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            stats.permtest_metric(np.arange(5), np.arange(4))

    def test_empty_arrays(self) -> None:
        """Verify empty inputs return NaN outputs."""
        corr, pval, nulls = stats.permtest_metric([], [])
        assert np.isnan(corr)
        assert np.isnan(pval)
        assert np.isnan(nulls)


class TestNullDistributionCallable:
    """Tests for _null_distribution_callable()."""

    def test_requires_exactly_one_permutation_source(self) -> None:
        """Verify exactly one permutation source is required."""

        def metric(a: np.ndarray, b: np.ndarray) -> float:  # noqa: ARG001 # unused, necessary params
            return 1.0

        with pytest.raises(ValueError, match="Exactly one"):
            stats._null_distribution_callable(
                np.arange(5),
                np.arange(5),
                metric=metric,
                perm_idx=None,
                nulls=None,
                n_perm=5,
                corr_shape=(),
                nan_policy="propagate",
            )

        with pytest.raises(ValueError, match="Exactly one"):
            stats._null_distribution_callable(
                np.arange(5),
                np.arange(5),
                metric=metric,
                perm_idx=np.arange(5)[None, :],
                nulls=np.ones((5, 5)),
                n_perm=5,
                corr_shape=(),
                nan_policy="propagate",
            )
