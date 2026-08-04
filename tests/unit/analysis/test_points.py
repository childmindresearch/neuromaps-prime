"""Tests for surface mesh geometry operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from neuromaps_prime.analysis.surfaces import points

if TYPE_CHECKING:
    from pathlib import Path

    from scipy.sparse import csr_matrix


# Simple triangle in the XY plane.
simple_vertices = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
simple_faces = np.array([[0, 1, 2]], dtype=np.int32)

# Unit square split into two triangles sharing edge (1, 2).
square_vertices = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    dtype=np.float32,
)
square_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)


def _make_gifti_surface(coords: np.ndarray, faces: np.ndarray, path: Path) -> None:
    """Write a GIFTI surface file."""
    ptarr = nib.gifti.GiftiDataArray(
        coords.astype(np.float32), intent="NIFTI_INTENT_POINTSET"
    )
    tris = nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE")
    nib.GiftiImage(darrays=[ptarr, tris]).to_filename(path)


def _make_gifti_parc(
    data: np.ndarray,
    labels: list[tuple[int, str]],
    path: Path,
) -> None:
    """Write a GIFTI parcellation file."""
    darr = nib.gifti.GiftiDataArray(
        data.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32"
    )
    lt = nib.gifti.GiftiLabelTable()
    for key, name in labels:
        lbl = nib.gifti.GiftiLabel(key=key)
        lbl.label = name
        lt.labels.append(lbl)
    nib.GiftiImage(darrays=[darr], labeltable=lt).to_filename(path)


@pytest.fixture(scope="module")
def surf_gifti_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """GIFTI surface file for the simple triangle."""
    p = tmp_path_factory.mktemp("data") / "sphere.surf.gii"
    _make_gifti_surface(coords=simple_vertices, faces=simple_faces, path=p)
    return p


@pytest.fixture(scope="module")
def square_surf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """GIFTI surface file for the square mesh."""
    p = tmp_path_factory.mktemp("data") / "square.surf.gii"
    _make_gifti_surface(coords=square_vertices, faces=square_faces, path=p)
    return p


@pytest.fixture(scope="module")
def parc_gifti_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """GIFTI parcellation file."""
    p = tmp_path_factory.mktemp("data") / "parc.label.gii"
    _make_gifti_parc(
        data=np.array([1, 1, 2], dtype=np.int32), labels=[(1, "A"), (2, "B")], path=p
    )
    return p


@pytest.fixture(scope="module")
def square_graph() -> csr_matrix:
    """Adjacency graph for the square mesh."""
    return points.make_surf_graph(vertices=square_vertices, faces=square_faces)


class TestGetEdges:
    """Tests for _get_edges()."""

    def test_shape_and_sorted(self) -> None:
        """Verify output shape and canonical edge ordering."""
        edges = points._get_edges(simple_faces)
        assert edges.shape == (3, 2)
        assert np.all(edges[:, 0] <= edges[:, 1])
        edge_set = set(map(tuple, edges))
        assert (0, 1) in edge_set
        assert (1, 2) in edge_set
        assert (0, 2) in edge_set


class TestGetDirectedEdges:
    """Tests for _get_directed_edges()."""

    def test_deduplication(self) -> None:
        """Verify square mesh produces deduplicated edges."""
        edges, _ = points._get_directed_edges(
            vertices=square_vertices, faces=square_faces
        )
        assert edges.shape[0] == 5

    def test_weights(self) -> None:
        """Verify weights are 1-D with expected Euclidean lengths."""
        _, weights = points._get_directed_edges(
            vertices=square_vertices, faces=square_faces
        )
        assert weights.ndim == 1
        assert sum(np.isclose(weights, 1.0)) == 4
        diag = weights[~np.isclose(weights, 1.0)]
        assert np.isclose(diag[0], np.sqrt(2))


class TestGetIndirectEdges:
    """Tests for _get_indirect_edges()."""

    def test_shared_edge(self) -> None:
        """Verify indirect edge exists between opposite vertices of shared edge."""
        edges, weights = points._get_indirect_edges(
            vertices=square_vertices, faces=square_faces
        )
        edge_set = set(map(frozenset, edges))
        assert frozenset({0, 3}) in edge_set
        assert np.all(weights > 0)


class TestPointInTriangle:
    """Tests for _point_in_triangle()."""

    def test_centroid_inside(self) -> None:
        """Verify centroid is inside and pdist is non-negative."""
        centroid = simple_vertices.mean(axis=0)
        inside, pdist = points._point_in_triangle(
            point=centroid, triangle=simple_vertices
        )
        assert inside
        assert pdist is not None
        assert pdist >= 0

    def test_outside(self) -> None:
        """Verify a point outside the triangle is rejected."""
        inside, _ = points._point_in_triangle(
            point=np.array([10.0, 10.0, 0.0]), triangle=simple_vertices
        )
        assert not inside

    def test_no_pdist(self) -> None:
        """Verify pdist is None when disabled."""
        _, pdist = points._point_in_triangle(
            point=simple_vertices.mean(axis=0),
            triangle=simple_vertices,
            return_pdist=False,
        )
        assert pdist is None


class TestWhichTriangle:
    """Tests for which_triangle()."""

    def test_found(self) -> None:
        """Verify the centroid is found inside triangle 0."""
        assert (
            points.which_triangle(
                point=simple_vertices.mean(axis=0),
                triangles=simple_vertices[simple_faces],
            )
            == 0
        )

    def test_not_found(self) -> None:
        """Verify a far-away point returns None."""
        assert (
            points.which_triangle(
                point=np.array([100.0, 100.0, 100.0]),
                triangles=simple_vertices[simple_faces],
            )
            is None
        )


class TestMakeSurfGraph:
    """Tests for make_surf_graph()."""

    def test_shape_and_weights(self, square_graph: csr_matrix) -> None:
        """Verify graph shape, no self-loops, and positive weights."""
        assert square_graph.shape == (4, 4)
        assert np.all(square_graph.diagonal() == 0)
        assert np.all(square_graph.data > 0)

    def test_mask_excludes(self) -> None:
        """Verify mask removes edges touching excluded vertices."""
        g = points.make_surf_graph(
            vertices=square_vertices,
            faces=square_faces,
            mask=np.array([False, False, True, False]),
        )
        assert g[2, :].nnz == 0
        assert g[:, 2].nnz == 0

    def test_bad_mask_raises(self) -> None:
        """Verify mismatched mask size raises ValueError."""
        with pytest.raises(ValueError, match="different number"):
            points.make_surf_graph(
                vertices=square_vertices,
                faces=square_faces,
                mask=np.array([True, False]),
            )


class TestGeodesicParcelCentroid:
    """Tests for _geodesic_parcel_centroid()."""

    def test_returns_vertex(self) -> None:
        """Verify the centroid is one of the parcel's vertices."""
        centroid = points._geodesic_parcel_centroid(
            vertices=square_vertices, faces=square_faces, inds=np.array([0, 1])
        )
        assert tuple(centroid) in (tuple(square_vertices[0]), tuple(square_vertices[1]))

    def test_empty_raises(self) -> None:
        """Verify empty parcel index list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            points._geodesic_parcel_centroid(
                vertices=square_vertices,
                faces=square_faces,
                inds=np.array([], dtype=int),
            )


class TestLoadGifti:
    """Tests for _load_gifti()."""

    def test_returns_gifti(self, surf_gifti_path: Path) -> None:
        """Verify loading a valid GIFTI file returns a GiftiImage."""
        assert isinstance(points._load_gifti(surf_gifti_path), nib.GiftiImage)

    def test_wrong_type_raises(self, tmp_path: Path) -> None:
        """Verify loading a non-GIFTI file raises ValueError."""
        nii_path = tmp_path / "dummy.nii.gz"
        nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)).to_filename(str(nii_path))
        with pytest.raises(ValueError, match="Gifti"):
            points._load_gifti(nii_path)


class TestRelabelGifti:
    """Tests for _relabel_gifti()."""

    def test_consecutive(self, parc_gifti_path: Path) -> None:
        """Verify output labels are remapped to consecutive indices."""
        unique = np.unique(points._relabel_gifti(parc_gifti_path).agg_data())
        np.testing.assert_array_equal(unique, [0, 1])

    def test_background_zeroed(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        """Verify background labels are zeroed out."""
        p = tmp_path_factory.mktemp("data") / "parc_bg.label.gii"
        _make_gifti_parc(
            data=np.array([1, 2, 3], dtype=np.int32),
            labels=[(1, "Cortex"), (2, "unknown"), (3, "Stem")],
            path=p,
        )
        data = points._relabel_gifti(p).agg_data()
        assert data[1] == 0
        np.testing.assert_array_equal(np.unique(data[data > 0]), [1, 2])


class TestGetGraphDistance:
    """Tests for _get_graph_distance()."""

    def test_dtype_and_self(self, square_graph: csr_matrix) -> None:
        """Verify output dtype is float32 and distance to self is zero."""
        dist = points._get_graph_distance(0, square_graph)
        assert dist.dtype == np.float32
        assert dist[0] == 0.0
        assert np.all(dist[1:] > 0)

    def test_parcel_aggregation(self, square_graph: csr_matrix) -> None:
        """Verify parcel-level aggregation returns correct shape."""
        labels = np.array([1, 1, 2, 2], dtype=int)
        dist = points._get_graph_distance(
            0, square_graph, labels=labels, unique_labels=np.array([1, 2], dtype=int)
        )
        assert dist.shape == (2,)

    def test_arg_mismatch_raises(self, square_graph: csr_matrix) -> None:
        """Verify mismatched labels/unique_labels raises ValueError."""
        labels = np.array([1, 1, 2, 2], dtype=int)
        for args in (
            {"labels": labels, "unique_labels": None},
            {"labels": None, "unique_labels": np.array([1, 2])},
        ):
            with pytest.raises(ValueError, match="both"):
                points._get_graph_distance(0, square_graph, **args)


class TestGetSurfaceDistance:
    """Tests for get_surface_distance()."""

    def test_vertex_level(self, square_surf_path: str) -> None:
        """Verify vertex-level distance matrix shape and zero diagonal."""
        dist = points.get_surface_distance(square_surf_path, n_proc=1)
        assert dist.shape == (4, 4)
        np.testing.assert_allclose(np.diag(dist), 0)

    def test_parcel_level(
        self, square_surf_path: str, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Verify parcel-to-parcel distance matrix shape."""
        parc_path = tmp_path_factory.mktemp("data") / "parc.label.gii"
        _make_gifti_parc(
            np.array([1, 1, 2, 3], dtype=np.int32),
            [(1, "A"), (2, "B"), (3, "C")],
            parc_path,
        )
        dist = points.get_surface_distance(
            square_surf_path, parcellation=parc_path, n_proc=1
        )
        # 3 parcels (incl. background) -> strip row/col 0 -> (2, 2)
        assert dist.shape == (2, 2)
        np.testing.assert_allclose(np.diag(dist), 0)
