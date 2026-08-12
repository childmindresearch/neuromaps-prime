"""Tests for :mod:`neuromaps_prime.analysis.parcellate`."""

import nibabel as nib
import numpy as np
import pytest

from neuromaps_prime.analysis.parcellate import ParcellatedData, parcellate


@pytest.fixture
def surface_data():
    """Ten vertices with a four-label parcellation, one of them background."""
    data = np.arange(10, dtype=float)
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    return data, labels


@pytest.fixture
def volume_data():
    """A 3x3x3 volume split into three axial slabs."""
    data = np.arange(27, dtype=float).reshape(3, 3, 3)
    labels = np.zeros((3, 3, 3), dtype=int)
    labels[0], labels[1], labels[2] = 1, 2, 3
    return data, labels


def _write_label_gii(labels, tmp_path):
    img = nib.GiftiImage()
    img.add_gifti_data_array(
        nib.gifti.GiftiDataArray(labels.astype(np.int32), intent="NIFTI_INTENT_LABEL")
    )
    path = tmp_path / "atlas.label.gii"
    nib.save(img, str(path))
    return path


def _write_shape_gii(data, tmp_path):
    img = nib.GiftiImage()
    img.add_gifti_data_array(
        nib.gifti.GiftiDataArray(data.astype(np.float32), intent="NIFTI_INTENT_SHAPE")
    )
    path = tmp_path / "map.func.gii"
    nib.save(img, str(path))
    return path


class TestParcellate:
    def test_returns_named_tuple(self, surface_data):
        result = parcellate(*surface_data)
        assert isinstance(result, ParcellatedData)
        assert result.labels.shape == result.values.shape

    def test_surface_mean(self, surface_data):
        result = parcellate(*surface_data)
        np.testing.assert_array_equal(result.labels, [1, 2, 3])
        np.testing.assert_allclose(result.values, [3.0, 5.5, 8.0])

    def test_background_excluded_by_default(self, surface_data):
        assert 0 not in parcellate(*surface_data).labels

    def test_background_none_keeps_zero(self, surface_data):
        result = parcellate(*surface_data, background=None)
        np.testing.assert_array_equal(result.labels, [0, 1, 2, 3])

    def test_custom_background(self, surface_data):
        data, labels = surface_data
        result = parcellate(data, labels, background=1)
        np.testing.assert_array_equal(result.labels, [0, 2, 3])

    def test_volume(self, volume_data):
        data, labels = volume_data
        result = parcellate(data, labels)
        np.testing.assert_allclose(
            result.values, [data[0].mean(), data[1].mean(), data[2].mean()]
        )

    def test_gifti_paths(self, surface_data, tmp_path):
        data, labels = surface_data
        result = parcellate(
            _write_shape_gii(data, tmp_path), _write_label_gii(labels, tmp_path)
        )
        np.testing.assert_allclose(result.values, [3.0, 5.5, 8.0])

    def test_nifti_paths(self, volume_data, tmp_path):
        data, labels = volume_data
        dpath, lpath = tmp_path / "map.nii.gz", tmp_path / "atlas.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(dpath))
        nib.save(nib.Nifti1Image(labels.astype(np.int16), np.eye(4)), str(lpath))
        np.testing.assert_allclose(parcellate(dpath, lpath).values, [4.0, 13.0, 22.0])

    def test_nans_ignored(self, surface_data):
        data, labels = surface_data
        data = data.copy()
        data[2] = np.nan
        np.testing.assert_allclose(parcellate(data, labels).values, [3.5, 5.5, 8.0])

    def test_all_nan_region_returns_nan(self, surface_data):
        _, labels = surface_data
        result = parcellate(np.full(10, np.nan), labels)
        assert np.all(np.isnan(result.values))

    def test_min_valid(self, surface_data):
        data, labels = surface_data
        data = data.copy()
        data[[2, 3]] = np.nan  # region 1 keeps a single finite vertex
        result = parcellate(data, labels, min_valid=2)
        assert np.isnan(result.values[0])
        assert np.isfinite(result.values[1:]).all()

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("mean", [3.0, 5.5, 8.0]),
            ("median", [3.0, 5.5, 8.0]),
            ("sum", [9.0, 11.0, 24.0]),
            ("min", [2.0, 5.0, 7.0]),
            ("max", [4.0, 6.0, 9.0]),
        ],
    )
    def test_methods(self, surface_data, method, expected):
        result = parcellate(*surface_data, method=method)
        np.testing.assert_allclose(result.values, expected)

    def test_callable_method(self, surface_data):
        result = parcellate(
            *surface_data, method=lambda a, axis: np.nanmean(a, axis=axis)
        )
        np.testing.assert_allclose(result.values, [3.0, 5.5, 8.0])

    def test_multi_feature(self, surface_data):
        data, labels = surface_data
        result = parcellate(np.c_[data, data * 2], labels)
        assert result.values.shape == (3, 2)
        np.testing.assert_allclose(result.values[:, 1], [6.0, 11.0, 16.0])

    def test_non_contiguous_labels_sorted(self, surface_data):
        data, _ = surface_data
        labels = np.array([7, 7, 3, 3, 3, 99, 99, 12, 12, 12])
        np.testing.assert_array_equal(parcellate(data, labels).labels, [3, 7, 12, 99])

    def test_matches_naive_loop(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=500)
        labels = rng.integers(0, 20, 500)
        expected = [np.nanmean(data[labels == k]) for k in range(1, 20)]
        np.testing.assert_allclose(parcellate(data, labels).values, expected)

    def test_agrees_with_vertices_to_parcels(self, surface_data, tmp_path):
        from neuromaps_prime.analysis.surfaces.nulls.spins import vertices_to_parcels

        data, labels = surface_data
        path = _write_label_gii(labels, tmp_path)
        np.testing.assert_allclose(
            parcellate(data, labels).values, vertices_to_parcels(data, path)
        )

    def test_shape_mismatch_raises(self, surface_data):
        _, labels = surface_data
        with pytest.raises(ValueError, match="same elements"):
            parcellate(np.arange(5.0), labels)

    def test_unknown_method_raises(self, surface_data):
        with pytest.raises(ValueError, match="Unknown method"):
            parcellate(*surface_data, method="nope")

    def test_missing_file_raises(self, surface_data):
        _, labels = surface_data
        with pytest.raises(FileNotFoundError):
            parcellate("does_not_exist.func.gii", labels)

    def test_multi_feature_parcellation_raises(self, surface_data):
        data, labels = surface_data
        with pytest.raises(ValueError, match="single volume or surface"):
            parcellate(data, np.c_[labels, labels])
