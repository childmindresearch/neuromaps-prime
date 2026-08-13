"""Tests for :mod:`neuromaps_prime.analysis.parcels`."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuromaps_prime.analysis.parcels import ParcelSummary, ReduceMethod, parcel_reduce


@pytest.fixture
def surface_data() -> tuple[np.ndarray, np.ndarray]:
    """Provide a 10-vertex surface data array and corresponding label array.

    The label array contains a background label (0) and three valid parcel
    labels (1, 2, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the 1D surface values
            and 1D parcellation labels.
    """
    data = np.arange(10, dtype=float)
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    return data, labels


@pytest.fixture
def volume_data() -> tuple[np.ndarray, np.ndarray]:
    """Provide a 3x3x3 volumetric data array and corresponding label array.

    The labels divide the volume into three distinct axial slabs (1, 2, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the 3D volume values
            and 3D parcellation labels.
    """
    data = np.arange(27, dtype=float).reshape(3, 3, 3)
    labels = np.zeros((3, 3, 3), dtype=int)
    labels[0], labels[1], labels[2] = 1, 2, 3
    return data, labels


def _write_label_gii(labels: np.ndarray, tmp_path: Path) -> Path:
    """Write a label array to a temporary GIFTI label file.

    Args:
        labels: 1D array of parcel integer labels.
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        Path: Path to the generated ``.label.gii`` file.
    """
    img = nib.GiftiImage()
    img.add_gifti_data_array(
        nib.gifti.GiftiDataArray(labels.astype(np.int32), intent="NIFTI_INTENT_LABEL")
    )
    path = tmp_path / "atlas.label.gii"
    nib.save(img, path)
    return path


def _write_shape_gii(data: np.ndarray, tmp_path: Path) -> Path:
    """Write a functional data array to a temporary GIFTI metric file.

    Args:
        data: 1D array of continuous metric values.
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        Path: Path to the generated ``.func.gii`` file.
    """
    img = nib.GiftiImage()
    img.add_gifti_data_array(
        nib.gifti.GiftiDataArray(data.astype(np.float32), intent="NIFTI_INTENT_SHAPE")
    )
    path = tmp_path / "map.func.gii"
    nib.save(img, path)
    return path


class TestParcelReduce:
    """Test suite for the :func:`parcel_reduce` function."""

    def test_returns_named_tuple(
        self, surface_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify that the result is a ParcelSummary instance with matching shape."""
        result = parcel_reduce(*surface_data)
        assert isinstance(result, ParcelSummary)
        assert result.labels.shape == result.values.shape

    def test_surface_mean(self, surface_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Verify mean reduction across surface vertices for each parcel."""
        result = parcel_reduce(*surface_data)
        np.testing.assert_array_equal(result.labels, [1, 2, 3])
        np.testing.assert_allclose(result.values, [3.0, 5.5, 8.0])

    def test_background_excluded_by_default(
        self, surface_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify that label 0 is treated as background and excluded by default."""
        assert 0 not in parcel_reduce(*surface_data).labels

    def test_background_none_keeps_zero(
        self, surface_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify that setting background=None retains label 0 in the output."""
        result = parcel_reduce(*surface_data, background=None)
        np.testing.assert_array_equal(result.labels, [0, 1, 2, 3])

    def test_custom_background(
        self, surface_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify that specifying a custom background label excludes it correctly."""
        data, labels = surface_data
        result = parcel_reduce(data, labels, background=1)
        np.testing.assert_array_equal(result.labels, [0, 2, 3])

    def test_volume(self, volume_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Verify parcellation on 3D volumetric data and labels."""
        data, labels = volume_data
        result = parcel_reduce(data, labels)
        np.testing.assert_allclose(
            result.values, [data[0].mean(), data[1].mean(), data[2].mean()]
        )

    def test_gifti_paths(
        self, surface_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Verify parcellation when inputs are GIFTI file paths."""
        data, labels = surface_data
        result = parcel_reduce(
            _write_shape_gii(data, tmp_path), _write_label_gii(labels, tmp_path)
        )
        np.testing.assert_allclose(result.values, [3.0, 5.5, 8.0])

    def test_nifti_paths(
        self, volume_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Verify parcellation when inputs are NIfTI file paths."""
        data, labels = volume_data
        dpath, lpath = tmp_path / "map.nii", tmp_path / "atlas.nii"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(dpath))
        nib.save(nib.Nifti1Image(labels.astype(np.int16), np.eye(4)), str(lpath))
        np.testing.assert_allclose(
            parcel_reduce(dpath, lpath).values, [4.0, 13.0, 22.0]
        )

    def test_nans_ignored(self, surface_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Verify that NaN values in data are ignored during aggregation."""
        data, labels = surface_data
        data = data.copy()
        data[2] = np.nan
        np.testing.assert_allclose(parcel_reduce(data, labels).values, [3.5, 5.5, 8.0])

    def test_all_nan_region_returns_nan(
        self, surface_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify that a parcel where all data values are NaN yields NaN."""
        _, labels = surface_data
        result = parcel_reduce(np.full(10, np.nan), labels)
        assert np.all(np.isnan(result.values))

    def test_min_valid(self, surface_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Verify parcels with fewer valid observations than min_valid yield NaN."""
        data, labels = surface_data
        data = data.copy()
        data[[2, 3]] = np.nan  # region 1 keeps a single finite vertex
        result = parcel_reduce(data, labels, min_valid=2)
        assert np.isnan(result.values[0])
        assert np.isfinite(result.values[1:]).all()

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("mean", [3.0, 5.5, 8.0]),
            ("median", [3.0, 5.5, 8.0]),
            ("std", [np.sqrt(2 / 3), 0.5, np.sqrt(2 / 3)]),
            ("sum", [9.0, 11.0, 24.0]),
            ("min", [2.0, 5.0, 7.0]),
            ("max", [4.0, 6.0, 9.0]),
        ],
    )
    def test_methods(
        self,
        surface_data: tuple[np.ndarray, np.ndarray],
        method: ReduceMethod,
        expected: list[float],
    ) -> None:
        """Verify that named reduction methods compute expected summaries."""
        result = parcel_reduce(*surface_data, method=method)
        np.testing.assert_allclose(result.values, expected)

    def test_callable_method(self, surface_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Verify that a custom reduction callable can be supplied."""
        result = parcel_reduce(
            *surface_data, method=lambda a, axis: np.nanmean(a, axis=axis)
        )
        np.testing.assert_allclose(result.values, [3.0, 5.5, 8.0])

    def test_multi_feature(self, surface_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Verify parcellation on multi-feature 2D data arrays."""
        data, labels = surface_data
        result = parcel_reduce(np.c_[data, data * 2], labels)
        assert result.values.shape == (3, 2)
        np.testing.assert_allclose(result.values[:, 1], [6.0, 11.0, 16.0])

    def test_non_contiguous_labels_sorted(
        self, surface_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify that output labels are sorted numerically when non-contiguous."""
        data, _ = surface_data
        labels = np.array([7, 7, 3, 3, 3, 99, 99, 12, 12, 12])
        np.testing.assert_array_equal(
            parcel_reduce(data, labels).labels, [3, 7, 12, 99]
        )

    def test_matches_naive_loop(self) -> None:
        """Verify vectorized parcellation output matches a reference Python loop."""
        rng = np.random.default_rng(1)
        data = rng.normal(size=500)
        labels = rng.integers(0, 20, 500)
        expected = [np.nanmean(data[labels == k]) for k in range(1, 20)]
        np.testing.assert_allclose(parcel_reduce(data, labels).values, expected)

    def test_agrees_with_vertices_to_parcels(
        self, surface_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Verify equivalence with legacy vertices_to_parcels function."""
        from neuromaps_prime.analysis.surfaces.nulls.spins import vertices_to_parcels

        data, labels = surface_data
        path = _write_label_gii(labels, tmp_path)
        np.testing.assert_allclose(
            parcel_reduce(data, labels).values, vertices_to_parcels(data, path)
        )

    def test_shape_mismatch_raises(
        self, surface_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify that mismatched data and label sizes raise a ValueError."""
        _, labels = surface_data
        with pytest.raises(ValueError, match="same elements"):
            parcel_reduce(np.arange(5.0), labels)

    def test_unknown_method_raises(
        self, surface_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify that an invalid reduction method string raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            parcel_reduce(*surface_data, method="nope")  # ty: ignore[invalid-argument-type]

    def test_missing_file_raises(
        self, surface_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify that a non-existent file path raises a FileNotFoundError."""
        _, labels = surface_data
        with pytest.raises(FileNotFoundError):
            parcel_reduce("does_not_exist.func.gii", labels)

    def test_multi_feature_parcellation_raises(
        self, surface_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify that passing 2D label maps raises a ValueError."""
        data, labels = surface_data
        with pytest.raises(ValueError, match="single volume or surface"):
            parcel_reduce(data, np.c_[labels, labels])
