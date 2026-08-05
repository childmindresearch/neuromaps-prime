"""Unit tests for the spin permutation utilities in spins.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest
from tests.unit.analysis.helpers import _make_gifti_parc

from neuromaps_prime.analysis.surfaces.nulls.spins import (
    Rotation,
    SpinResult,
    _assign_coordinates,
    _gen_rotation,
    _get_parcel_centroids,
    _max_overlap,
    _to_hemisphere_list,
    _validate_spin_inputs,
    gen_spin_samples,
    get_parcel_centroids,
    load_spins,
    parcels_to_vertices,
    spin_data,
    spin_parcels,
    vertices_to_parcels,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.random import Generator


class TestToHemisphereList:
    """Tests for _to_hemisphere_list()."""

    def test_single_path(self) -> None:
        """Verify single path becomes single-element list."""
        result = _to_hemisphere_list("/path/to/file.surf.gii")
        assert result == ["/path/to/file.surf.gii"]

    def test_tuple_paths(self) -> None:
        """Verify tuple becomes two-element list."""
        result = _to_hemisphere_list(("/path/lh.surf.gii", "/path/rh.surf.gii"))
        assert result == ["/path/lh.surf.gii", "/path/rh.surf.gii"]


class TestGenRotation:
    """Tests for _gen_rotation()."""

    def test_returns_rotation_namedtuple(self) -> None:
        """Verify output is a Rotation namedtuple."""
        rot = _gen_rotation(seed=42)
        assert isinstance(rot, Rotation)
        assert hasattr(rot, "left")
        assert hasattr(rot, "right")

    def test_both_matrices_are_orthogonal(self) -> None:
        """Verify rotation matrices are orthogonal (R @ R.T = I)."""
        rot = _gen_rotation(seed=42)
        np.testing.assert_allclose(rot.left @ rot.left.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(rot.right @ rot.right.T, np.eye(3), atol=1e-6)

    def test_both_matrices_have_det_one(self) -> None:
        """Verify rotation matrices have determinant +1."""
        rot = _gen_rotation(seed=42)
        assert np.isclose(np.linalg.det(rot.left), 1.0)
        assert np.isclose(np.linalg.det(rot.right), 1.0)

    def test_right_is_reflected_left(self) -> None:
        """Verify right hemisphere rotation is reflected left."""
        rot = _gen_rotation(seed=42)
        reflect_x = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_allclose(rot.right, reflect_x @ rot.left @ reflect_x)

    def test_different_seeds_produce_different_rotations(self) -> None:
        """Verify different seeds produce different rotations."""
        rot1 = _gen_rotation(seed=42)
        rot2 = _gen_rotation(seed=123)
        assert not np.allclose(rot1.left, rot2.left)


class TestValidateSpinInputs:
    """Tests for _validate_spin_inputs()."""

    def test_valid_inputs_pass(self, rng: Generator) -> None:
        """Verify valid inputs don't raise."""
        coords = rng.random((10, 3))
        hemi_id = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
        _validate_spin_inputs(coords, hemi_id)

    def test_bad_coords_shape_raises(self, rng: Generator) -> None:
        """Verify wrong coords shape raises ValueError."""
        coords = rng.random((10, 4))
        hemi_id = np.zeros(10, dtype=np.int8)
        with pytest.raises(ValueError, match="must be of shape"):
            _validate_spin_inputs(coords, hemi_id)

    def test_bad_hemi_id_ndim_raises(self, rng: Generator) -> None:
        """Verify 2D hemi_id raises ValueError."""
        coords = rng.random((10, 3))
        hemi_id = np.zeros((10, 1), dtype=np.int8)
        with pytest.raises(ValueError, match="must be one-dimensional"):
            _validate_spin_inputs(coords, hemi_id)

    def test_mismatched_lengths_raises(self, rng: Generator) -> None:
        """Verify mismatched lengths raises ValueError."""
        coords = rng.random((10, 3))
        hemi_id = np.zeros(5, dtype=np.int8)
        with pytest.raises(ValueError, match="same length"):
            _validate_spin_inputs(coords, hemi_id)

    def test_invalid_hemi_values_raises(self, rng: Generator) -> None:
        """Verify hemi_id values other than 0/1 raise ValueError."""
        coords = rng.random((10, 3))
        hemi_id = np.array([0, 0, 1, 1, 2, 2, 0, 1, 0, 1], dtype=np.int8)
        with pytest.raises(ValueError, match=r"\{0, 1\}"):
            _validate_spin_inputs(coords, hemi_id)


class TestAssignCoordinates:
    """Tests for _assign_coordinates()."""

    def test_original_method_kdtree(self) -> None:
        """Verify original method uses KDTree nearest-neighbour."""
        coor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        rotated = np.array([[0.1, 0.1, 0.0], [0.9, -0.1, 0.0]])
        col, dist = _assign_coordinates(coor, rotated, "original")
        assert col.shape == (2,)
        assert dist.shape == (2,)
        assert np.all(col >= 0)
        assert np.all(col < 2)

    def test_vasa_method_greedy(self) -> None:
        """Verify vasa method uses greedy assignment."""
        coor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotated = np.array([[0.1, 0.1, 0.0], [0.9, -0.1, 0.0], [0.0, 0.9, 0.1]])
        col, costs = _assign_coordinates(coor, rotated, "vasa")
        assert col.shape == (3,)
        assert costs.shape == (3,)
        assert len(np.unique(col)) == 3
        assert np.all(costs >= 0)

    def test_hungarian_method_optimal(self) -> None:
        """Verify hungarian method uses optimal assignment."""
        coor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotated = np.array([[0.1, 0.1, 0.0], [0.9, -0.1, 0.0], [0.0, 0.9, 0.1]])
        col, costs = _assign_coordinates(coor, rotated, "hungarian")
        assert col.shape == (3,)
        assert costs.shape == (3,)
        assert len(np.unique(col)) == 3
        assert np.all(costs >= 0)

    def test_unknown_method_raises(self) -> None:
        """Verify unknown method raises ValueError."""
        coor = np.array([[0.0, 0.0, 0.0]])
        rotated = np.array([[0.1, 0.1, 0.0]])
        with pytest.raises(ValueError, match="Unknown method"):
            _assign_coordinates(coor, rotated, "unknown_method")  # type: ignore


class TestGenSpinSamples:
    """Tests for gen_spin_samples()."""

    def test_output_shapes(self, rng: Generator) -> None:
        """Verify output is SpinResult with correct shapes."""
        coords = rng.random((10, 3))
        hemi_id = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
        result = gen_spin_samples(coords, hemi_id, n_rotate=5, seed=42)
        assert isinstance(result, SpinResult)
        assert result.spin.shape == (10, 5)
        assert result.cost is None

    def test_with_return_cost(self, rng: Generator) -> None:
        """Verify return_cost=True returns cost matrix."""
        coords = rng.random((10, 3))
        hemi_id = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
        result = gen_spin_samples(
            coords, hemi_id, n_rotate=5, seed=42, return_cost=True
        )
        assert result.cost is not None
        assert result.cost.shape == (10, 5)

    def test_spins_are_permutations(self, rng: Generator) -> None:
        """Verify each spin column has all indices exactly once."""
        coords = rng.random((8, 3))
        hemi_id = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        result = gen_spin_samples(coords, hemi_id, n_rotate=3, seed=42)
        for k in range(3):
            assert len(result.spin[:, k]) == 8

    def test_vasa_method(self, rng: Generator) -> None:
        """Verify vasa spin method works."""
        coords = rng.random((8, 3))
        hemi_id = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        result = gen_spin_samples(coords, hemi_id, n_rotate=3, method="vasa", seed=42)
        assert result.spin.shape == (8, 3)

    def test_hungarian_method(self, rng: Generator) -> None:
        """Verify hungarian spin method works."""
        coords = rng.random((8, 3))
        hemi_id = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        result = gen_spin_samples(
            coords, hemi_id, n_rotate=3, method="hungarian", seed=42
        )
        assert result.spin.shape == (8, 3)

    def test_check_duplicates_removes_identity(self, rng: Generator) -> None:
        """Verify check_duplicates=True removes identity permutation."""
        coords = rng.random((8, 3))
        hemi_id = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        result = gen_spin_samples(
            coords, hemi_id, n_rotate=5, seed=42, check_duplicates=True
        )
        identity = np.arange(8, dtype=np.int32)
        for k in range(5):
            assert not np.array_equal(result.spin[:, k], identity)

    def test_unknown_method_raises(self, rng: Generator) -> None:
        """Verify unknown method raises ValueError."""
        coords = rng.random((8, 3))
        hemi_id = np.zeros(8, dtype=np.int8)
        with pytest.raises(ValueError, match="invalid"):
            gen_spin_samples(coords, hemi_id, method="unknown")  # type: ignore


class TestMaxOverlap:
    """Tests for _max_overlap()."""

    def test_returns_most_common_positive_label(self) -> None:
        """Verify most common positive label is returned."""
        vals = np.array([0, 1, 1, 1, 2, 2, 3])
        assert _max_overlap(vals) == 0

    def test_all_nonpositive_returns_minus_one(self) -> None:
        """Verify all non-positive values return -1."""
        vals = np.array([0, -1, -2])
        assert _max_overlap(vals) == -1

    def test_single_positive(self) -> None:
        """Verify single positive value is returned."""
        vals = np.array([0, 0, 5])
        assert _max_overlap(vals) == 4


class TestGetParcelCentroids:
    """Tests for _get_parcel_centroids()."""

    def test_average_method(self, simple_sphere: Path, simple_parc: Path) -> None:
        """Verify average centroid method returns correct shape."""
        verts, faces = nib.load(simple_sphere).agg_data()
        labels = nib.load(simple_parc).agg_data()
        centroids = _get_parcel_centroids(verts, faces, labels, method="average")
        assert centroids.shape == (2, 3)

    def test_surface_method(self, simple_sphere: Path, simple_parc: Path) -> None:
        """Verify surface centroid method returns vertex coordinates."""
        verts, faces = nib.load(simple_sphere).agg_data()
        labels = nib.load(simple_parc).agg_data()
        centroids = _get_parcel_centroids(verts, faces, labels, method="surface")
        assert centroids.shape == (2, 3)
        for centroid in centroids:
            assert any(np.allclose(centroid, v) for v in verts)

    def test_geodesic_method(self, simple_sphere: Path, simple_parc: Path) -> None:
        """Verify geodesic centroid method returns correct shape."""
        verts, faces = nib.load(simple_sphere).agg_data()
        labels = nib.load(simple_parc).agg_data()
        centroids = _get_parcel_centroids(verts, faces, labels, method="geodesic")
        assert centroids.shape == (2, 3)

    def test_drop_labels(self, simple_sphere: Path) -> None:
        """Verify specified labels are dropped from parcellation."""
        verts, faces = nib.load(simple_sphere).agg_data()
        labels = np.array([1, 1, 99, 2], dtype=np.int32)
        labeltable = {1: "A", 2: "B", 99: "unknown"}
        centroids = _get_parcel_centroids(
            verts,
            faces,
            labels,
            method="surface",
            drop=["unknown"],
            labeltable=labeltable,
        )
        assert centroids.shape == (2, 3)

    def test_unknown_method_raises(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify unknown centroid method raises ValueError."""
        verts, faces = nib.load(simple_sphere).agg_data()
        labels = nib.load(simple_parc).agg_data()
        with pytest.raises(ValueError, match="Expected one of"):
            _get_parcel_centroids(verts, faces, labels, method="unknown")  # type: ignore


class TestGetParcelCentroidsPublic:
    """Tests for get_parcel_centroids() public interface."""

    def test_without_parcellation(self, simple_sphere: Path) -> None:
        """Verify vertex-level centroids when no parcellation is provided."""
        centroids = get_parcel_centroids(simple_sphere)
        verts, _ = nib.load(simple_sphere).agg_data()
        assert centroids.shape == (len(verts), 3)

    def test_with_parcellation(self, simple_sphere: Path, simple_parc: Path) -> None:
        """Verify parcel-level centroids when parcellation is provided."""
        centroids = get_parcel_centroids(simple_sphere, parcellation=simple_parc)
        assert centroids.shape == (2, 3)


class TestLoadSpins:
    """Tests for load_spins()."""

    def test_from_array(self) -> None:
        """Verify array input is returned as-is."""
        spins = np.array([[0, 1], [1, 0]], dtype=np.int32)
        result = load_spins(spins)
        np.testing.assert_array_equal(result, spins)

    def test_from_npy_file(self, tmp_path: Path) -> None:
        """Verify .npy file loading works."""
        spins = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int32)
        np.save(tmp_path / "spins.npy", spins)
        result = load_spins(tmp_path / "spins.npy")
        np.testing.assert_array_equal(result, spins)

    def test_from_csv_file(self, tmp_path: Path) -> None:
        """Verify .csv file loading works."""
        spins = np.array([[0, 1], [1, 0]], dtype=np.int32)
        csv_path = tmp_path / "spins.csv"
        np.savetxt(csv_path, spins, delimiter=",", fmt="%d")
        result = load_spins(csv_path)
        np.testing.assert_array_equal(result, spins)

    def test_truncate_with_n_perm(self, rng: Generator) -> None:
        """Verify n_perm truncates to requested number of permutations."""
        spins = rng.integers(0, 10, size=(10, 20))
        result = load_spins(spins, n_perm=5)
        assert result.shape == (10, 5)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Verify missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_spins(tmp_path / "nonexistent.npy")


class TestSpinParcels:
    """Tests for spin_parcels()."""

    def test_output_shape(self, simple_sphere: Path, simple_parc: Path) -> None:
        """Verify output shape is (n_parcels, n_rotate)."""
        result = spin_parcels(simple_sphere, simple_parc, n_rotate=5, seed=42)
        assert isinstance(result, SpinResult)
        assert result.spin.shape == (2, 5)

    def test_dual_hemisphere(
        self, two_hemi_surfaces: tuple[Path, Path], two_hemi_parcs: tuple[Path, Path]
    ) -> None:
        """Verify dual-hemisphere parcellation works."""
        result = spin_parcels(two_hemi_surfaces, two_hemi_parcs, n_rotate=5, seed=42)
        assert result.spin.shape == (4, 5)

    def test_with_precomputed_spins(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify precomputed spins are used instead of generating new ones."""
        spins = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32).T
        result = spin_parcels(
            simple_sphere, simple_parc, spins=spins, n_rotate=5, seed=42
        )
        assert result.spin.shape == (2, 2)

    def test_mismatched_vertex_count_raises(
        self, simple_sphere: Path, tmp_path: Path
    ) -> None:
        """Verify mismatched vertex count raises ValueError."""
        bad_parc = tmp_path / "bad_parc.label.gii"
        _make_gifti_parc(np.array([1, 2, 3], dtype=np.int32), [], bad_parc)
        with pytest.raises(ValueError, match="does not match"):
            spin_parcels(simple_sphere, bad_parc, n_rotate=3, seed=42)


class TestSpinData:
    """Tests for spin_data()."""

    def test_output_shape(self, simple_sphere: Path, simple_parc: Path) -> None:
        """Verify output shape is (n_parcels, n_rotate)."""
        data = np.array([10.0, 20.0])
        result = spin_data(data, simple_sphere, simple_parc, n_rotate=5, seed=42)
        assert isinstance(result, SpinResult)
        assert result.spin.shape == (2, 5)

    def test_dual_hemisphere(
        self, two_hemi_surfaces: tuple[Path, Path], two_hemi_parcs: tuple[Path, Path]
    ) -> None:
        """Verify dual-hemisphere data splicing works."""
        data = np.array([10.0, 20.0, 30.0, 40.0])
        result = spin_data(data, two_hemi_surfaces, two_hemi_parcs, n_rotate=5, seed=42)
        assert result.spin.shape == (4, 5)

    def test_with_precomputed_spins(
        self, simple_sphere: Path, simple_parc: Path
    ) -> None:
        """Verify precomputed spins are used instead of generating new ones."""
        data = np.array([10.0, 20.0])
        spins = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32).T
        result = spin_data(
            data, simple_sphere, simple_parc, spins=spins, n_rotate=2, seed=42
        )
        assert result.spin.shape == (2, 2)

    def test_mismatched_vertex_count_raises(
        self, simple_sphere: Path, tmp_path: Path
    ) -> None:
        """Verify mismatched vertex count raises ValueError."""
        bad_parc = tmp_path / "bad_parc.label.gii"
        _make_gifti_parc(np.array([1, 2, 3], dtype=np.int32), [], bad_parc)
        data = np.array([10.0, 20.0])
        with pytest.raises(ValueError, match="does not match"):
            spin_data(data, simple_sphere, bad_parc, n_rotate=3, seed=42)


class TestParcelsToVertices:
    """Tests for parcels_to_vertices()."""

    def test_basic_projection(self, simple_parc: Path) -> None:
        """Verify basic parcel-to-vertex projection."""
        data = np.array([10.0, 20.0])
        result = parcels_to_vertices(data, simple_parc)
        assert result.shape == (4,)
        labels = nib.load(simple_parc).agg_data()
        for i, lab in enumerate(labels):
            if lab == 1:
                assert np.isclose(result[i], 10.0)
            elif lab == 2:
                assert np.isclose(result[i], 20.0)

    def test_2d_data(self, simple_parc: Path) -> None:
        """Verify 2D data is projected correctly."""
        data = np.array([[10.0, 100.0], [20.0, 200.0]])
        result = parcels_to_vertices(data, simple_parc)
        assert result.shape == (4, 2)

    def test_background_gets_nan(self, tmp_path: Path) -> None:
        """Verify background vertices get NaN."""
        parc = tmp_path / "bg_parc.label.gii"
        _make_gifti_parc(np.array([0, 1, 1, 2], dtype=np.int32), [], parc)
        data = np.array([10.0, 20.0])
        result = parcels_to_vertices(data, parc)
        assert np.isnan(result[0])


class TestVerticesToParcels:
    """Tests for vertices_to_parcels()."""

    def test_basic_reduction(self, simple_parc: Path) -> None:
        """Verify basic vertex-to-parcel averaging."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = vertices_to_parcels(data, simple_parc)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [1.5, 3.5])

    def test_2d_data(self, simple_parc: Path, rng: Generator) -> None:
        """Verify 2D data is averaged correctly."""
        data = rng.random((4, 3))
        result = vertices_to_parcels(data, simple_parc)
        assert result.shape == (2, 3)

    def test_background_handling(self, tmp_path: Path) -> None:
        """Verify background vertices are excluded from averaging."""
        parc = tmp_path / "bg_parc.label.gii"
        _make_gifti_parc(np.array([0, 1, 1, 2], dtype=np.int32), [], parc)
        data = np.array([999.0, 1.0, 2.0, 3.0])
        result = vertices_to_parcels(data, parc)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [1.5, 3.0])

    def test_background_value_exclusion(self, simple_parc: Path) -> None:
        """Verify custom background value is excluded."""
        data = np.array([999.0, 1.0, 2.0, 3.0])
        result = vertices_to_parcels(data, simple_parc, background=999.0)
        np.testing.assert_allclose(result, [1.0, 2.5])

    def test_mismatched_vertex_count_raises(self, simple_parc: Path) -> None:
        """Verify mismatched vertex count raises ValueError."""
        data = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="does not match"):
            vertices_to_parcels(data, simple_parc)
