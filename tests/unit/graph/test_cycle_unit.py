"""Unit tests for cyclic surface transformations.

These tests validate cycle traversal and round-trip transformation behavior
using a small synthetic three-node graph.

The synthetic graph contains three spaces (A, B, and C) connected by known
rotational surface transformations. Forward edges rotate the sphere by
+120 degrees and reverse edges rotate by -120 degrees. Therefore, all closed
paths return to the original coordinate system:

    A → B → A          (+120 - 120)       = identity
    A → C → A          (-120 + 120)       = identity
    A → B → C → A      (+120 + 120 +120)  = identity
    A → C → B → A      (-120 -120 -120)   = identity

A synthetic vertex-wise metric is transformed around each cycle. Since each
cycle represents an identity transformation, the round-tripped metric should
match the original metric with near-perfect correlation.

The Workbench metric resampling dependency is replaced with a lightweight
numpy implementation. This keeps the tests fast, deterministic, and isolated
from external software dependencies.

These tests validate the cycle-testing machinery itself. End-to-end accuracy
of real surface transformations is evaluated separately in the regression
test suite using real templates and Workbench resampling.

Run with::

    pytest tests/unit/graph/unit_test_cycle.py -v -s

Run only the metric preservation test::

    pytest tests/unit/graph/unit_test_cycle.py::test_closed_cycles_preserve_metric -v -s
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from nibabel.gifti import GiftiDataArray, GiftiImage
from scipy.spatial import ConvexHull, cKDTree

from neuromaps_prime.graph import NeuromapsGraph
from tests.cycle import find_return_paths, load_metric, run_cycle_test

# -------------------------------------------------------------------------
# Test parameters
# -------------------------------------------------------------------------

N_VERTICES = 642
DENSITY = "1k"
HEMISPHERE = "left"

EXPECTED_CYCLES = {
    ("A", "B", "A"),
    ("A", "C", "A"),
    ("A", "B", "C", "A"),
    ("A", "C", "B", "A"),
}

DENOM = 0.00000000001


# -------------------------------------------------------------------------
# Synthetic geometry helpers
# -------------------------------------------------------------------------


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate approximately uniform points on a unit sphere."""
    indices = np.arange(n) + 0.5

    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices

    return np.c_[
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ].astype(np.float32)


def _rotation_x(degrees: float) -> np.ndarray:
    """Create a rotation matrix around the x-axis."""
    theta = np.deg2rad(degrees)

    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )


def _save_surface(
    path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> None:
    """Save vertices and triangles as a GIFTI surface."""
    image = GiftiImage()

    image.add_gifti_data_array(
        GiftiDataArray(
            vertices.astype(np.float32),
            intent="NIFTI_INTENT_POINTSET",
        )
    )

    image.add_gifti_data_array(
        GiftiDataArray(
            triangles.astype(np.int32),
            intent="NIFTI_INTENT_TRIANGLE",
        )
    )

    nib.save(image, path)


def _save_metric(
    path: Path,
    metric: np.ndarray,
) -> None:
    """Save a vertex-wise metric as a GIFTI functional file."""
    image = GiftiImage()

    image.add_gifti_data_array(
        GiftiDataArray(
            metric.astype(np.float32),
            intent="NIFTI_INTENT_NONE",
        )
    )

    nib.save(image, path)


def _load_vertices(path: Path | str) -> np.ndarray:
    """Load vertex coordinates from a GIFTI surface."""
    image = nib.load(str(path))

    for array in image.darrays:
        if array.data.ndim == 2 and array.data.shape[1] == 3:
            return np.asarray(array.data, dtype=float)

    raise ValueError(f"No vertices found in {path}")


# -------------------------------------------------------------------------
# Resampling mock
# -------------------------------------------------------------------------


def _resample_metric(
    metric: np.ndarray,
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
) -> np.ndarray:
    """Approximate barycentric metric resampling using nearest vertices.

    This provides a lightweight replacement for Workbench resampling while
    preserving the behavior needed for cycle testing.
    """
    _, indices = cKDTree(source_vertices).query(
        target_vertices,
        k=3,
    )

    triangle = source_vertices[indices]

    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]

    v0 = b - a
    v1 = c - a
    v2 = target_vertices - a

    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)

    # prevent division by near-zero values
    denominator = d00 * d11 - d01 * d01
    denominator = np.where(
        np.abs(denominator) < DENOM,
        DENOM,
        denominator,
    )

    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    u = 1 - v - w

    weights = np.column_stack((u, v, w))

    return np.einsum(
        "ij,ij->i",
        weights,
        metric[indices],
    )


def _fake_metric_resample(
    input_file_path: str | Path,
    current_sphere: str | Path,
    new_sphere: str | Path,
    _method: None,
    _area_surfs: None,
    output_file_path: str,
) -> SimpleNamespace:
    """Mock Workbench metric resampling for isolated unit testing."""
    result = _resample_metric(
        load_metric(input_file_path),
        _load_vertices(current_sphere),
        _load_vertices(new_sphere),
    )

    _save_metric(Path(output_file_path), result)

    return SimpleNamespace(metric_out=Path(output_file_path))


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Directory for storing cycle test outputs."""
    directory = tmp_path / "output"
    directory.mkdir()

    return directory


@pytest.fixture
def patch_metric_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace Workbench resampling with the numpy test implementation."""
    monkeypatch.setattr(
        "neuromaps_prime.graph.transforms.surface.metric_resample",
        _fake_metric_resample,
    )


@pytest.fixture
def rotation_network(
    tmp_path: Path,
) -> tuple[NeuromapsGraph, Path]:
    """Create a three-node graph with known identity cycles."""
    vertices = _fibonacci_sphere(N_VERTICES)
    triangles = ConvexHull(vertices).simplices

    sphere = tmp_path / "sphere.surf.gii"
    _save_surface(sphere, vertices, triangles)

    rotations = {
        ("A", "B"): 120,
        ("B", "C"): 120,
        ("C", "A"): 120,
        ("B", "A"): -120,
        ("C", "B"): -120,
        ("A", "C"): -120,
    }

    edges = []

    for (source, target), angle in rotations.items():
        transformed = (_rotation_x(angle) @ vertices.T).T

        edge_surface = tmp_path / f"{source}_{target}.surf.gii"

        _save_surface(
            edge_surface,
            transformed,
            triangles,
        )

        edges.append(
            {
                "from": source,
                "to": target,
                "surfaces": {
                    "synthetic": {
                        DENSITY: {
                            "sphere": {
                                "left": str(edge_surface),
                                "right": str(edge_surface),
                            }
                        }
                    }
                },
            }
        )

    def node(name: str) -> dict:
        return {
            name: {
                "surfaces": {
                    DENSITY: {
                        "sphere": {
                            "left": str(sphere),
                            "right": str(sphere),
                        },
                        "midthickness": {
                            "left": str(sphere),
                            "right": str(sphere),
                        },
                    }
                }
            }
        }

    graph = NeuromapsGraph(
        runner="local",
        data_dir=tmp_path / "cache",
        _testing=True,
    )

    graph._builder.build_from_dict(
        graph,
        {
            "nodes": [node("A"), node("B"), node("C")],
            "edges": {
                "surface_to_surface": edges,
                "volume_to_volume": [],
            },
        },
    )

    metric = tmp_path / "metric.func.gii"
    _save_metric(metric, vertices.sum(axis=1))

    return graph, metric


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


def test_find_return_paths_returns_expected_cycles(
    rotation_network: tuple[NeuromapsGraph, Path],
) -> None:
    """Return all simple closed paths from the synthetic graph."""
    graph, _ = rotation_network

    paths = find_return_paths(graph, "A")

    assert set(paths) == EXPECTED_CYCLES


def test_find_return_paths_respects_length_limit(
    rotation_network: tuple[NeuromapsGraph, Path],
) -> None:
    """Only cycles within the requested hop limit are returned."""
    graph, _ = rotation_network

    paths = find_return_paths(
        graph,
        "A",
        max_length=2,
    )

    assert set(paths) == {
        ("A", "B", "A"),
        ("A", "C", "A"),
    }


def test_find_return_paths_rejects_unknown_node(
    rotation_network: tuple[NeuromapsGraph, Path],
) -> None:
    """Invalid graph origins raise a clear error."""
    graph, _ = rotation_network

    with pytest.raises(
        ValueError,
        match="not in the 'surface_to_surface' layer",
    ):
        find_return_paths(graph, "unknown")


@pytest.mark.usefixtures("patch_metric_resample")
def test_closed_cycles_preserve_metric(
    rotation_network: tuple[NeuromapsGraph, Path],
    output_dir: Path,
) -> None:
    """A closed transformation cycle preserves the input metric."""
    graph, metric = rotation_network

    results = run_cycle_test(
        graph,
        "A",
        metric,
        HEMISPHERE,
        output_dir,
        density=DENSITY,
    )

    assert {result.path for result in results} == EXPECTED_CYCLES

    for result in results:
        assert result.pearson_r > 0.999
