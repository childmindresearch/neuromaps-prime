"""Shared fixtures for nulls module tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from tests.unit.analysis.helpers import _make_gifti_parc, _make_gifti_surface

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def simple_sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Simple 4-vertex sphere-like surface."""
    p = tmp_path_factory.mktemp("data") / "sphere.surf.gii"
    coords = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=np.float32,
    )
    coords = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
    _make_gifti_surface(coords, faces, p)
    return p


@pytest.fixture(scope="module")
def two_hemi_surfaces(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two hemisphere surfaces (left and right)."""
    tmp_dir = tmp_path_factory.mktemp("data")
    p_left = tmp_dir / "lh.sphere.surf.gii"
    p_right = tmp_dir / "rh.sphere.surf.gii"

    left_coords = np.array(
        [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    left_coords = left_coords / np.linalg.norm(left_coords, axis=1, keepdims=True)
    _make_gifti_surface(
        left_coords,
        np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32),
        p_left,
    )

    right_coords = np.array(
        [[-1.0, 0.5, 0.5], [-0.5, 1.0, 0.5], [-0.5, 0.5, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    right_coords = right_coords / np.linalg.norm(right_coords, axis=1, keepdims=True)
    _make_gifti_surface(
        right_coords,
        np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32),
        p_right,
    )

    return p_left, p_right


@pytest.fixture(scope="module")
def simple_parc(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Simple parcellation with 2 parcels + background."""
    p = tmp_path_factory.mktemp("data") / "parc.label.gii"
    _make_gifti_parc(
        np.array([1, 1, 2, 2], dtype=np.int32), [(1, "Parcel_A"), (2, "Parcel_B")], p
    )
    return p


@pytest.fixture(scope="module")
def two_hemi_parcs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Parcellations for both hemispheres with unique label values per hemisphere."""
    tmp_dir = tmp_path_factory.mktemp("data")
    p_left = tmp_dir / "lh.parc.label.gii"
    p_right = tmp_dir / "rh.parc.label.gii"
    _make_gifti_parc(
        np.array([1, 1, 2, 2], dtype=np.int32),
        [(1, "L_Parcel_A"), (2, "L_Parcel_B")],
        p_left,
    )
    _make_gifti_parc(
        np.array([3, 4, 4, 3], dtype=np.int32),
        [(3, "R_Parcel_A"), (4, "R_Parcel_B")],
        p_right,
    )
    return p_left, p_right


@pytest.fixture(scope="module")
def parc_with_unknown(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Parcellation with an 'unknown' label that should be dropped."""
    p = tmp_path_factory.mktemp("data") / "parc_unknown.label.gii"
    _make_gifti_parc(
        np.array([1, 1, 99, 2], dtype=np.int32),
        [(1, "Parcel_A"), (2, "Parcel_B"), (99, "unknown")],
        p,
    )
    return p


_ICOSA_COORDS = np.array(
    [
        [-1.0, 1.61803399, 0.0],
        [1.0, 1.61803399, 0.0],
        [-1.0, -1.61803399, 0.0],
        [1.0, -1.61803399, 0.0],
        [0.0, -1.0, 1.61803399],
        [0.0, 1.0, 1.61803399],
        [0.0, -1.0, -1.61803399],
        [0.0, 1.0, -1.61803399],
        [1.61803399, 0.0, -1.0],
        [1.61803399, 0.0, 1.0],
        [-1.61803399, 0.0, -1.0],
        [-1.61803399, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_ICOSA_COORDS = _ICOSA_COORDS / np.linalg.norm(_ICOSA_COORDS, axis=1, keepdims=True)
_ICOSA_FACES = np.array(
    [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ],
    dtype=np.int32,
)


def _make_icosahedron(path: Path) -> None:
    """Write a 12-vertex icosahedron surface to *path*."""
    _make_gifti_surface(_ICOSA_COORDS, _ICOSA_FACES, path)


@pytest.fixture(scope="module")
def large_sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """12-vertex icosahedron sphere surface."""
    p = tmp_path_factory.mktemp("data") / "icosa.surf.gii"
    _make_icosahedron(p)
    return p


@pytest.fixture(scope="module")
def large_parc(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Parcellation with background (0) and 6 parcels over 12 vertices."""
    p = tmp_path_factory.mktemp("data") / "icosa.parc.label.gii"
    labels = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], dtype=np.int32)
    _make_gifti_parc(
        labels,
        [
            (0, "background"),
            (1, "P1"),
            (2, "P2"),
            (3, "P3"),
            (4, "P4"),
            (5, "P5"),
            (6, "P6"),
        ],
        p,
    )
    return p


@pytest.fixture(scope="module")
def large_parc_with_unknown(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Icosahedron parcellation with background, 5 real parcels, and 'unknown'."""
    p = tmp_path_factory.mktemp("data") / "icosa_unknown.parc.label.gii"
    labels = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], dtype=np.int32)
    _make_gifti_parc(
        labels,
        [
            (0, "background"),
            (1, "P1"),
            (2, "P2"),
            (3, "P3"),
            (4, "P4"),
            (5, "P5"),
            (6, "unknown"),
        ],
        p,
    )
    return p


@pytest.fixture(scope="module")
def two_hemi_large_surfaces(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    """Two icosahedron hemisphere surfaces."""
    tmp_dir = tmp_path_factory.mktemp("data")
    p_left = tmp_dir / "lh.icosa.surf.gii"
    p_right = tmp_dir / "rh.icosa.surf.gii"
    _make_icosahedron(p_left)
    _make_icosahedron(p_right)
    return p_left, p_right


@pytest.fixture(scope="module")
def two_hemi_large_parcs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two icosahedron parcellations with background + unique labels per hemi."""
    tmp_dir = tmp_path_factory.mktemp("data")
    p_left = tmp_dir / "lh.icosa.parc.label.gii"
    p_right = tmp_dir / "rh.icosa.parc.label.gii"
    _make_gifti_parc(
        np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0], dtype=np.int32),
        [(0, "background"), (1, "L1"), (2, "L2"), (3, "L3")],
        p_left,
    )
    _make_gifti_parc(
        np.array([0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 5, 6], dtype=np.int32),
        [(0, "background"), (4, "R1"), (5, "R2"), (6, "R3")],
        p_right,
    )
    return p_left, p_right
