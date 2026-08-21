"""Unit tests for the null model convenience wrappers in nulls.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from neuromaps_prime.analysis.images import PARC_IGNORE
from neuromaps_prime.analysis.surfaces.nulls.nulls import (
    _generate_spins,
    _resolve_data,
    _spin_permute,
    alexander_bloch,
    baum,
    burt2018,
    cornblath,
    hungarian,
    vasa,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.random import Generator


class TestResolveData:
    """Tests for _resolve_data()."""

    def test_array_input(self) -> None:
        """Verify array input is flattened."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _resolve_data(data, {})
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, [1, 2, 3, 4])

    def test_gifti_file_input(self, tmp_path: Path) -> None:
        """Verify GIFTI file is loaded and flattened."""
        data_arr = nib.gifti.GiftiDataArray(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            intent="NIFTI_INTENT_NONE",
        )
        img = nib.GiftiImage(darrays=[data_arr])
        gii_path = tmp_path / "data.func.gii"
        img.to_filename(str(gii_path))
        result = _resolve_data(gii_path, {})
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result, [1, 2, 3, 4])

    def test_tuple_input(self, tmp_path: Path) -> None:
        """Verify tuple of GIFTI files are stacked."""
        left_arr = nib.gifti.GiftiDataArray(
            np.array([1.0, 2.0], dtype=np.float32), intent="NIFTI_INTENT_NONE"
        )
        left_path = tmp_path / "lh.func.gii"
        nib.GiftiImage(darrays=[left_arr]).to_filename(str(left_path))
        right_arr = nib.gifti.GiftiDataArray(
            np.array([3.0, 4.0, 5.0], dtype=np.float32), intent="NIFTI_INTENT_NONE"
        )
        right_path = tmp_path / "rh.func.gii"
        nib.GiftiImage(darrays=[right_arr]).to_filename(str(right_path))
        result = _resolve_data((left_path, right_path), {})
        assert result.shape == (5,)
        np.testing.assert_array_almost_equal(result, [1, 2, 3, 4, 5])

    def test_gifti_image_input(self) -> None:
        """Verify GiftiImage object is loaded directly."""
        data_arr = nib.gifti.GiftiDataArray(
            np.array([10.0, 20.0, 30.0], dtype=np.float32), intent="NIFTI_INTENT_NONE"
        )
        result = _resolve_data(nib.GiftiImage(darrays=[data_arr]), {})
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [10, 20, 30])


class TestSpinPermute:
    """Tests for _spin_permute() pipeline."""

    def test_returns_spin_matrix_when_data_none(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify data=None returns spin permutation matrix."""
        result = _spin_permute(
            None,
            simple_sphere,
            parcellation=simple_parc,
            n_perm=5,
            seed=42,
            spins=None,
            spin_method="original",
            method="surface",
            drop=PARC_IGNORE,
        )
        assert result.shape == (2, 5)
        assert result.dtype == np.int32

    def test_vertex_level_permutation(self, simple_sphere: Path) -> None:
        """Verify vertex-level spin permutation works."""
        result = _spin_permute(
            np.array([1.0, 2.0, 3.0, 4.0]),
            simple_sphere,
            parcellation=None,
            n_perm=3,
            seed=123,
            spins=None,
            spin_method="original",
            method="surface",
            drop=PARC_IGNORE,
        )
        assert result.shape == (4, 3)

    def test_mismatched_data_length_raises(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify ValueError when data length doesn't match parcel count."""
        with pytest.raises(ValueError, match="does not match"):
            _spin_permute(
                np.array([1.0, 2.0, 3.0]),
                simple_sphere,
                parcellation=simple_parc,
                n_perm=3,
                seed=42,
                spins=None,
                spin_method="original",
                method="surface",
                drop=PARC_IGNORE,
            )

    def test_precomputed_spins_ignores_surface(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify precomputed spins bypass surface loading."""
        spins = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
        data = np.array([10.0, 20.0])
        result = _spin_permute(
            data,
            simple_sphere,
            parcellation=simple_parc,
            n_perm=3,
            seed=42,
            spins=spins,
            spin_method="original",
            method="surface",
            drop=PARC_IGNORE,
        )
        np.testing.assert_array_equal(result, data[spins])


class TestAlexanderBloch:
    """Tests for alexander_bloch()."""

    def test_vertex_level_output_shape(self, simple_sphere: Path) -> None:
        """Verify vertex-level output shape."""
        result = alexander_bloch(
            np.array([1.0, 2.0, 3.0, 4.0]), simple_sphere, n_perm=10, seed=42
        )
        assert result.shape == (4, 10)

    def test_parcellated_output_shape(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify parcellated output shape."""
        result = alexander_bloch(
            np.array([10.0, 20.0]),
            simple_sphere,
            parcellation=simple_parc,
            n_perm=10,
            seed=42,
        )
        assert result.shape == (2, 10)

    def test_returns_spin_matrix_when_data_none(self, simple_sphere: Path) -> None:
        """Verify data=None returns spin matrix."""
        result = alexander_bloch(None, simple_sphere, n_perm=7, seed=99)
        assert result.shape == (4, 7)

    def test_dual_hemisphere_vertex_level(
        self, two_hemi_surfaces: tuple[Path, Path]
    ) -> None:
        """Verify dual-hemisphere vertex-level works."""
        result = alexander_bloch(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            two_hemi_surfaces,
            n_perm=5,
            seed=42,
        )
        assert result.shape == (8, 5)

    def test_dual_hemisphere_parcellated(
        self, two_hemi_surfaces: tuple[Path, Path], two_hemi_parcs: tuple[Path, Path]
    ) -> None:
        """Verify dual-hemisphere parcellated works."""
        result = alexander_bloch(
            np.array([10.0, 20.0, 30.0, 40.0]),
            two_hemi_surfaces,
            parcellation=two_hemi_parcs,
            n_perm=5,
            seed=42,
        )
        assert result.shape == (4, 5)

    def test_drop_unknown_labels(
        self, simple_sphere: Path, parc_with_unknown: Path
    ) -> None:
        """Verify 'unknown' label is dropped from parcellation."""
        result = alexander_bloch(
            np.array([10.0, 20.0]),
            simple_sphere,
            parcellation=parc_with_unknown,
            n_perm=5,
            seed=42,
        )
        assert result.shape == (2, 5)


class TestVasa:
    """Tests for vasa()."""

    def test_requires_parcellation(self, simple_sphere: Path) -> None:
        """Verify vasa requires parcellation argument."""
        with pytest.raises(TypeError, match="parcellation"):
            vasa(np.array([1.0, 2.0, 3.0, 4.0]), simple_sphere, n_perm=5, seed=42)  # type: ignore

    def test_output_shape_with_parcel(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify output shape with parcellation."""
        result = vasa(
            np.array([10.0, 20.0]),
            simple_sphere,
            parcellation=simple_parc,
            n_perm=5,
            seed=42,
        )
        assert result.shape == (2, 5)


class TestHungarian:
    """Tests for hungarian()."""

    def test_requires_parcellation(self, simple_sphere: Path) -> None:
        """Verify hungarian requires parcellation argument."""
        with pytest.raises(TypeError, match="parcellation"):
            hungarian(np.array([1.0, 2.0, 3.0, 4.0]), simple_sphere, n_perm=5, seed=42)  # type: ignore

    def test_output_shape_with_parcel(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify output shape with parcellation."""
        result = hungarian(
            np.array([10.0, 20.0]),
            simple_sphere,
            parcellation=simple_parc,
            n_perm=5,
            seed=42,
        )
        assert result.shape == (2, 5)


class TestBaum:
    """Tests for baum()."""

    def test_requires_parcellation(self, simple_sphere: Path) -> None:
        """Verify baum requires parcellation argument."""
        with pytest.raises(TypeError, match="parcellation"):
            baum(np.array([1.0, 2.0, 3.0, 4.0]), simple_sphere, n_perm=5, seed=42)  # type: ignore

    def test_output_shape_with_parcel(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify output shape with parcellation."""
        result = baum(
            np.array([10.0, 20.0]),
            simple_sphere,
            parcellation=simple_parc,
            n_perm=5,
            seed=42,
        )
        assert result.shape == (2, 5)

    def test_returns_spin_matrix_when_data_none(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify data=None returns spin matrix."""
        result = baum(None, simple_sphere, parcellation=simple_parc, n_perm=5, seed=42)
        assert result.shape == (2, 5)

    def test_nan_for_unmatched_parcels(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify unmatched parcels get NaN in output."""
        result = baum(
            np.array([10.0, 20.0]),
            simple_sphere,
            parcellation=simple_parc,
            n_perm=3,
            seed=999,
        )
        assert result.shape == (2, 3)
        assert np.isfinite(result).any()

    def test_mismatched_data_length_raises(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify ValueError when data length doesn't match parcel count."""
        with pytest.raises(ValueError, match="does not match"):
            baum(
                np.array([1.0, 2.0, 3.0]),
                simple_sphere,
                parcellation=simple_parc,
                n_perm=5,
                seed=42,
            )


class TestCornblath:
    """Tests for cornblath()."""

    def test_requires_parcellation(self, simple_sphere: Path) -> None:
        """Verify cornblath requires parcellation argument."""
        with pytest.raises(TypeError, match="parcellation"):
            cornblath(np.array([1.0, 2.0, 3.0, 4.0]), simple_sphere, n_perm=5, seed=42)  # type: ignore

    def test_output_shape_with_parcel(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify output shape with parcellation."""
        result = cornblath(
            np.array([10.0, 20.0]),
            simple_sphere,
            parcellation=simple_parc,
            n_perm=5,
            seed=42,
        )
        assert result.shape == (2, 5)
        assert np.isfinite(result).all()


class TestBurt2018:
    """Tests for burt2018()."""

    def test_vertex_level_output_shape(self, simple_sphere: Path) -> None:
        """Verify vertex-level output shape."""
        result = burt2018(
            np.array([1.0, 2.0, 3.0, 4.0]), simple_sphere, n_perm=5, seed=42, n_proc=1
        )
        assert result.shape == (4, 5)

    def test_parcellated_output_shape(
        self, large_sphere: Path, large_parc: Path
    ) -> None:
        """Verify parcellated output shape."""
        result = burt2018(
            np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            large_sphere,
            parcellation=large_parc,
            n_perm=5,
            seed=42,
            n_proc=1,
        )
        assert result.shape == (6, 5)
        assert np.isfinite(result).all()

    def test_dual_hemisphere_vertex_level(
        self, two_hemi_surfaces: tuple[Path, Path]
    ) -> None:
        """Verify dual-hemisphere vertex-level works."""
        result = burt2018(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            two_hemi_surfaces,
            n_perm=5,
            seed=42,
            n_proc=1,
        )
        assert result.shape == (8, 5)

    def test_dual_hemisphere_parcellated(
        self,
        two_hemi_large_surfaces: tuple[Path, Path],
        two_hemi_large_parcs: tuple[Path, Path],
    ) -> None:
        """Verify dual-hemisphere parcellated works."""
        result = burt2018(
            np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            two_hemi_large_surfaces,
            parcellation=two_hemi_large_parcs,
            n_perm=5,
            seed=42,
            n_proc=1,
        )
        assert result.shape == (6, 5)
        assert np.isfinite(result).all()

    def test_with_precomputed_distmat(
        self, simple_sphere: Path, rng: Generator
    ) -> None:
        """Verify precomputed distance matrix is used."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        distmat = rng.random((4, 4))
        distmat = distmat + distmat.T
        result = burt2018(
            data, simple_sphere, n_perm=3, seed=42, distmat=distmat, n_proc=1
        )
        assert result.shape == (4, 3)

    def test_with_tuple_distmat(
        self, two_hemi_surfaces: tuple[Path, Path], rng: Generator
    ) -> None:
        """Verify tuple of distance matrices works for dual hemi."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        dist_left = rng.random((4, 4))
        dist_right = np.random.default_rng(12346).random((4, 4))
        dist_left = dist_left + dist_left.T
        dist_right = dist_right + dist_right.T
        result = burt2018(
            data,
            two_hemi_surfaces,
            n_perm=3,
            seed=42,
            distmat=(dist_left, dist_right),
            n_proc=1,
        )
        assert result.shape == (8, 3)

    def test_mismatched_data_length_raises(
        self, large_sphere: Path, large_parc: Path
    ) -> None:
        """Verify ValueError when data length doesn't match parcel count."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        with pytest.raises(IndexError, match="out of bounds"):
            burt2018(
                data, large_sphere, parcellation=large_parc, n_perm=3, seed=42, n_proc=1
            )

    def test_handles_negative_data(self, simple_sphere: Path) -> None:
        """Verify negative data is handled via shift."""
        result = burt2018(
            np.array([-5.0, -3.0, 0.0, 2.0]), simple_sphere, n_perm=3, seed=42, n_proc=1
        )
        assert result.shape == (4, 3)
        assert np.isfinite(result).all()

    def test_drop_unknown_labels(
        self, large_sphere: Path, large_parc_with_unknown: Path
    ) -> None:
        """Verify 'unknown' label handling in burt2018."""
        result = burt2018(
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            large_sphere,
            parcellation=large_parc_with_unknown,
            n_perm=5,
            seed=42,
            n_proc=1,
        )
        assert result.shape == (5, 5)
        assert np.isfinite(result).all()


class TestGenerateSpins:
    """Tests for _generate_spins() helper."""

    def test_single_hemisphere(self, simple_sphere: Path, simple_parc: Path) -> None:
        """Verify single hemisphere spin generation."""
        spin_matrix, n_elements = _generate_spins(
            simple_sphere,
            simple_parc,
            n_perm=5,
            seed=42,
            centroid_method="surface",
            spin_method="original",
            drop=PARC_IGNORE,
            gifti_cache={},
        )
        assert spin_matrix.shape == (2, 5)
        assert n_elements == 2

    def test_dual_hemisphere(self, two_hemi_surfaces: tuple[Path, Path]) -> None:
        """Verify dual hemisphere spin generation."""
        spin_matrix, n_elements = _generate_spins(
            two_hemi_surfaces,
            None,
            n_perm=5,
            seed=42,
            centroid_method="surface",
            spin_method="original",
            drop=PARC_IGNORE,
            gifti_cache={},
        )
        assert spin_matrix.shape == (8, 5)
        assert n_elements == 8

    def test_mismatched_surface_parcellation_count_raises(
        self, two_hemi_surfaces: tuple[Path, Path], simple_parc: Path
    ) -> None:
        """Verify ValueError when surface and parcellation counts mismatch."""
        with pytest.raises(ValueError, match="Number of surface and parcellation"):
            _generate_spins(
                two_hemi_surfaces,
                simple_parc,
                n_perm=5,
                seed=42,
                centroid_method="surface",
                spin_method="original",
                drop=PARC_IGNORE,
                gifti_cache={},
            )

    def test_centroid_method_average(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify average centroid method works."""
        spin_matrix, n_elements = _generate_spins(
            simple_sphere,
            simple_parc,
            n_perm=3,
            seed=42,
            centroid_method="average",
            spin_method="original",
            drop=PARC_IGNORE,
            gifti_cache={},
        )
        assert spin_matrix.shape == (2, 3)
        assert n_elements == 2

    def test_centroid_method_geodesic(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify geodesic centroid method works."""
        spin_matrix, n_elements = _generate_spins(
            simple_sphere,
            simple_parc,
            n_perm=3,
            seed=42,
            centroid_method="geodesic",
            spin_method="original",
            drop=PARC_IGNORE,
            gifti_cache={},
        )
        assert spin_matrix.shape == (2, 3)
        assert n_elements == 2
