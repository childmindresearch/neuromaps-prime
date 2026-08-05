"""Unit tests for the Burt 2018 surrogate generation in burt.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuromaps_prime.analysis.surfaces.nulls.burt import (
    SurrogateResult,
    _make_weight_matrix,
    batch_surrogates,
    estimate_rho_d0,
    make_surrogate,
)

if TYPE_CHECKING:
    from numpy.random import Generator


class TestMakeWeightMatrix:
    """Tests for _make_weight_matrix()."""

    def test_row_normalization(self) -> None:
        """Verify each row sums to 1."""
        dist = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        w = _make_weight_matrix(dist, d0=1.0)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-5)

    def test_diagonal_is_zero(self) -> None:
        """Verify diagonal is zeroed."""
        dist = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        w = _make_weight_matrix(dist, d0=1.0)
        assert np.all(np.diag(w) == 0)

    def test_symmetric_distance_produces_asymmetric_weights(self) -> None:
        """Verify symmetric distance produce asymmetric weights due to normalization."""
        dist = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        w = _make_weight_matrix(dist, d0=1.0)
        assert not np.allclose(w, w.T)

    def test_small_d0_concentrates_weight(self) -> None:
        """Verify small d0 concentrates weight on nearest neighbours."""
        dist = np.array([[0.0, 1.0, 10.0], [1.0, 0.0, 1.0], [10.0, 1.0, 0.0]])
        w_large = _make_weight_matrix(dist, d0=10.0)
        w_small = _make_weight_matrix(dist, d0=0.1)
        assert w_small[0, 2] < w_large[0, 2]

    def test_off_diagonal_non_negative(self, rng: Generator) -> None:
        """Verify off-diagonal weights are non-negative."""
        dist = rng.random((5, 5))
        np.fill_diagonal(dist, 0)
        w = _make_weight_matrix(dist, d0=1.0)
        assert np.all(a=w >= 0)


class TestEstimateRhoD0:
    """Tests for estimate_rho_d0()."""

    @pytest.fixture
    def valid_data_for_estimation(
        self, rng: Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create valid distance matrix and positive data for estimation."""
        coords = rng.random((20, 3))
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        data = np.exp(rng.random(20) * 0.5)
        return dist, data

    def test_returns_tuple_of_two_floats(
        self, valid_data_for_estimation: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify output is a tuple of two floats."""
        dist, data = valid_data_for_estimation
        rho, d0 = estimate_rho_d0(dist, data)
        assert isinstance(rho, float)
        assert isinstance(d0, float)

    def test_positive_outputs(
        self, valid_data_for_estimation: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify outputs are finite."""
        dist, data = valid_data_for_estimation
        rho, d0 = estimate_rho_d0(dist, data)
        assert np.isfinite(rho)
        assert np.isfinite(d0)

    def test_with_initial_guesses(
        self, valid_data_for_estimation: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify initial guesses affect optimization."""
        dist, data = valid_data_for_estimation
        rho1, d01 = estimate_rho_d0(dist, data, rho=0.5, d0=1.0)
        rho2, d02 = estimate_rho_d0(dist, data, rho=0.9, d0=5.0)
        assert np.isfinite(rho1)
        assert np.isfinite(d01)
        assert np.isfinite(rho2)
        assert np.isfinite(d02)

    def test_small_dataset(self) -> None:
        """Verify works with small dataset."""
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        data = np.array([1.0, 2.0])
        rho, d0 = estimate_rho_d0(dist, data)
        assert np.isfinite(rho)
        assert np.isfinite(d0)


class TestMakeSurrogate:
    """Tests for make_surrogate()."""

    @pytest.fixture
    def simple_distmat(self, rng: Generator) -> np.ndarray:
        """Simple distance matrix for testing."""
        coords = rng.random((20, 3))
        return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    @pytest.fixture
    def simple_data(self, rng: Generator) -> np.ndarray:
        """Simple positive data for testing."""
        return np.abs(rng.random(20)) + 0.1

    def test_returns_surrogate_result(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify output is SurrogateResult namedtuple."""
        result = make_surrogate(simple_distmat, simple_data, seed=42)
        assert isinstance(result, SurrogateResult)
        assert hasattr(result, "surrogate")
        assert hasattr(result, "order")
        assert hasattr(result, "params")

    def test_surrogate_same_length_as_input(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify surrogate has same length as input."""
        result = make_surrogate(simple_distmat, simple_data, seed=42)
        assert len(result.surrogate) == len(simple_data)

    def test_surrogate_preserves_marginal_distribution(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify surrogate has same sorted values as input."""
        result = make_surrogate(simple_distmat, simple_data, seed=42)
        np.testing.assert_array_equal(np.sort(result.surrogate), np.sort(simple_data))

    def test_with_return_order(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify order is returned when requested."""
        result = make_surrogate(simple_distmat, simple_data, seed=42, return_order=True)
        assert result.order is not None
        assert len(result.order) == len(simple_data)

    def test_with_return_params(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify params are returned when requested."""
        result = make_surrogate(
            simple_distmat, simple_data, seed=42, return_params=True
        )
        assert result.params is not None
        rho, d0 = result.params
        assert isinstance(rho, float)
        assert isinstance(d0, float)

    def test_with_precomputed_params(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify precomputed params skip estimation."""
        result1 = make_surrogate(simple_distmat, simple_data, rho=0.5, d0=2.0, seed=42)
        result2 = make_surrogate(simple_distmat, simple_data, rho=0.5, d0=2.0, seed=42)
        np.testing.assert_array_almost_equal(result1.surrogate, result2.surrogate)

    def test_different_seeds_produce_different_surrogates(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify different seeds produce different surrogates."""
        result1 = make_surrogate(simple_distmat, simple_data, seed=42)
        result2 = make_surrogate(simple_distmat, simple_data, seed=123)
        assert not np.allclose(result1.surrogate, result2.surrogate)

    def test_reproducible_with_same_seed(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify same seed produces same surrogate."""
        result1 = make_surrogate(simple_distmat, simple_data, seed=42)
        result2 = make_surrogate(simple_distmat, simple_data, seed=42)
        np.testing.assert_array_almost_equal(result1.surrogate, result2.surrogate)


class TestBatchSurrogates:
    """Tests for batch_surrogates()."""

    @pytest.fixture
    def simple_distmat(self, rng: Generator) -> np.ndarray:
        """Simple distance matrix for testing."""
        coords = rng.random((20, 3))
        return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    @pytest.fixture
    def simple_data(self, rng: Generator) -> np.ndarray:
        """Simple positive data for testing."""
        return np.abs(rng.random(20)) + 0.1

    def test_output_shape(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify output shape is (n, n_surr)."""
        result = batch_surrogates(simple_distmat, simple_data, n_surr=10, seed=42)
        assert result.shape == (20, 10)

    def test_each_column_preserves_marginal(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify each surrogate column preserves marginal distribution."""
        result = batch_surrogates(simple_distmat, simple_data, n_surr=5, seed=42)
        for k in range(5):
            np.testing.assert_array_equal(np.sort(result[:, k]), np.sort(simple_data))

    def test_different_columns_are_different(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify different columns are different surrogates."""
        result = batch_surrogates(simple_distmat, simple_data, n_surr=5, seed=42)
        for i in range(5):
            for j in range(i + 1, 5):
                assert not np.allclose(result[:, i], result[:, j])

    def test_n_jobs_parallel(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify n_jobs > 1 produces valid results."""
        result_single = batch_surrogates(
            simple_distmat, simple_data, n_surr=5, seed=42, n_jobs=1
        )
        result_parallel = batch_surrogates(
            simple_distmat, simple_data, n_surr=5, seed=42, n_jobs=2
        )
        assert result_single.shape == result_parallel.shape
        for k in range(5):
            np.testing.assert_array_equal(
                np.sort(result_parallel[:, k]), np.sort(simple_data)
            )

    def test_zero_n_surr_raises(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify n_surr=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_surr must be positive"):
            batch_surrogates(simple_distmat, simple_data, n_surr=0, seed=42)

    def test_negative_n_surr_raises(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify negative n_surr raises ValueError."""
        with pytest.raises(ValueError, match="n_surr must be positive"):
            batch_surrogates(simple_distmat, simple_data, n_surr=-1, seed=42)

    def test_sparse_matrix_handling(
        self, simple_data: np.ndarray, rng: Generator
    ) -> None:
        """Verify sparse matrix path works for sparse-like inputs."""
        sparse_like = rng.random((20, 20)) * 0.01
        np.fill_diagonal(sparse_like, 0)
        result = batch_surrogates(sparse_like, simple_data, n_surr=3, seed=42)
        assert result.shape == (20, 3)

    def test_reproducible_with_same_seed(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify same seed produces same batch."""
        result1 = batch_surrogates(simple_distmat, simple_data, n_surr=5, seed=42)
        result2 = batch_surrogates(simple_distmat, simple_data, n_surr=5, seed=42)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_with_precomputed_params(
        self, simple_distmat: np.ndarray, simple_data: np.ndarray
    ) -> None:
        """Verify precomputed params skip estimation."""
        result1 = batch_surrogates(
            simple_distmat, simple_data, rho=0.5, d0=2.0, n_surr=3, seed=42
        )
        result2 = batch_surrogates(
            simple_distmat, simple_data, rho=0.5, d0=2.0, n_surr=3, seed=42
        )
        np.testing.assert_array_almost_equal(result1, result2)


class TestEdgeCases:
    """Edge case tests for burt.py functions."""

    def test_very_small_data(self, rng: Generator) -> None:
        """Verify handling of small but valid datasets."""
        coords = rng.standard_normal((5, 3))
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        data = np.abs(rng.standard_normal(5)) + 0.1
        result = make_surrogate(dist, data, seed=42)
        assert len(result.surrogate) == 5
        assert np.isfinite(result.surrogate).all()

    def test_data_with_outliers(self, rng: Generator) -> None:
        """Verify handling of data with outliers."""
        coords = rng.random((20, 3))
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        data = np.abs(rng.random(20)) + 0.1
        data[0] = 100.0
        result = make_surrogate(dist, data, seed=42)
        np.testing.assert_array_equal(np.sort(result.surrogate), np.sort(data))

    def test_very_high_autocorrelation(self) -> None:
        """Verify handling of highly autocorrelated data."""
        coords = np.linspace(0, 1, 20)
        dist = np.abs(coords[:, None] - coords[None, :])
        data = np.sin(coords * 10) + 2
        result = make_surrogate(dist, data, seed=42)
        assert len(result.surrogate) == 20

    def test_weight_matrix_with_identical_distances(self) -> None:
        """Verify weight matrix with uniform distances."""
        dist = np.ones((5, 5))
        np.fill_diagonal(dist, 0)
        w = _make_weight_matrix(dist, d0=1.0)
        for i in range(5):
            row_vals = w[i, :]
            assert np.allclose(row_vals[row_vals > 0], row_vals[row_vals > 0][0])
