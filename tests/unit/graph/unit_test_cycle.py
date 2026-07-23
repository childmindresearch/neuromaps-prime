"""Unit test for the cycle-test machinery (``tests/regression/cycle.py``).

The deployed cycle test round-trips a metric through every return path in the
graph and correlates the result against the original. To trust that number we
need a case where the *right* answer is known in advance. This test builds one.

Construction
------------
Three nodes ``A``, ``B``, ``C`` all share one template sphere. Each directed
edge is that template sphere pushed through a rigid rotation about the x-axis:
``+120`` degrees in the forward direction and ``-120`` degrees in reverse (this
is exactly what ``wb_command -surface-apply-affine`` would produce from the
corresponding affine). Because ``3 * 120 = 360`` and ``+120`` cancels ``-120``,
*every* return path composes to a full turn, i.e. the identity:

    A -> B -> A            (+120, -120)          = identity
    A -> C -> A            (-120, +120)          = identity
    A -> B -> C -> A       (+120, +120, +120)    = 360 deg = identity
    A -> C -> B -> A       (-120, -120, -120)    = -360 deg = identity

The seed metric is the sum of each vertex's coordinates (a smooth degree-1
function on the sphere). Under a perfect identity it must return unchanged, so
every cycle's correlation must be ~1. A small residual remains because
resampling interpolates between non-coincident vertices.

To keep the test hermetic and fast (no ``wb_command``, no downloads), the single
``wb_command`` call inside the transformer (``metric_resample``) is monkeypatched
with a vectorized barycentric resampler. Everything else -- graph construction,
cache lookups, path enumeration, per-hop plumbing, correlation -- is the real
production code from ``NeuromapsGraph`` and ``tests/regression/cycle.py``.
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
from tests.regression.cycle import find_return_paths, run_cycle_test

# A small sphere keeps the unit test well under a second while still exercising
# genuine interpolation (a Fibonacci layout has no rotational symmetry, so a
# 120-degree rotation never maps vertices onto vertices).
N_VERTICES = 642
DENSITY = f"{round(N_VERTICES / 1000)}k"  # -> "1k"
HEMISPHERE = "left"


# --------------------------------------------------------------------------- #
# Geometry / GIFTI helpers                                                     #
# --------------------------------------------------------------------------- #
def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return ``n`` roughly equidistant points on the unit sphere."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    golden = np.pi * (1 + 5**0.5)
    theta = golden * k
    return np.c_[
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ].astype(np.float32)


def _rot_x(degrees: float) -> np.ndarray:
    """Return the 3x3 rotation matrix for a rotation about the x-axis."""
    theta = np.deg2rad(degrees)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float64)


def _save_surface(path: Path, coords: np.ndarray, triangles: np.ndarray) -> None:
    """Write a spherical surface (points + triangles) as a ``.surf.gii``."""
    img = GiftiImage()
    img.add_gifti_data_array(
        GiftiDataArray(np.asarray(coords, np.float32), intent="NIFTI_INTENT_POINTSET")
    )
    img.add_gifti_data_array(
        GiftiDataArray(np.asarray(triangles, np.int32), intent="NIFTI_INTENT_TRIANGLE")
    )
    nib.save(img, str(path))


def _save_metric(path: Path, data: np.ndarray) -> None:
    """Write a per-vertex scalar metric as a ``.func.gii``."""
    img = GiftiImage()
    img.add_gifti_data_array(
        GiftiDataArray(np.asarray(data, np.float32), intent="NIFTI_INTENT_NONE")
    )
    nib.save(img, str(path))


def _load_coords(path: str | Path) -> np.ndarray:
    """Return the ``(n, 3)`` point set from a surface GIFTI."""
    for darray in nib.load(str(path)).darrays:
        if darray.data.ndim == 2 and darray.data.shape[1] == 3:
            return np.asarray(darray.data, np.float64)
    raise ValueError(f"No point set found in {path}")


def _load_metric(path: str | Path) -> np.ndarray:
    """Return the 1-D scalar array from a metric GIFTI."""
    return np.asarray(nib.load(str(path)).darrays[0].data, np.float64)


def _resample_barycentric(
    metric_in: np.ndarray, current_verts: np.ndarray, new_verts: np.ndarray
) -> np.ndarray:
    """Barycentric resample a metric onto ``new_verts`` from ``current_verts``.

    A vectorized stand-in for ``wb_command -metric-resample ... BARYCENTRIC``:
    for each new vertex it interpolates the metric within the triangle formed by
    its three nearest current vertices. The metric value ``metric_in[i]`` is
    carried at ``current_verts[i]``.

    Args:
        metric_in: Scalar value per current-sphere vertex.
        current_verts: Vertices the metric currently lives on.
        new_verts: Vertices to resample the metric onto.

    Returns:
        Scalar value per new-sphere vertex.
    """
    _, idx = cKDTree(current_verts).query(new_verts, k=3)
    tri = current_verts[idx]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, new_verts - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-18, 1e-18, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    weights = np.c_[u, v, w]
    return np.einsum("ij,ij->i", weights, metric_in[idx])


# --------------------------------------------------------------------------- #
# Synthetic network builder                                                    #
# --------------------------------------------------------------------------- #
# Each edge stores the SOURCE template sphere rotated into the TARGET frame,
# matching how transform_surface treats the edge sphere as the "current" sphere
# and the target's native sphere as the "new" sphere.
_EDGE_ROTATIONS = {
    ("A", "B"): 120.0,
    ("B", "C"): 120.0,
    ("C", "A"): 120.0,
    ("B", "A"): -120.0,
    ("C", "B"): -120.0,
    ("A", "C"): -120.0,
}


def _build_rotation_network(
    tmp_path: Path, corrupt_edge: tuple[str, str] | None = None
) -> tuple[NeuromapsGraph, Path]:
    """Build a synthetic three-node rotation graph and its seed metric.

    Args:
        tmp_path: Directory to write the generated GIFTI fixtures into.
        corrupt_edge: Optional edge whose forward rotation is set to 90 degrees
            instead of 120, breaking the identity for any path through it.

    Returns:
        Tuple of the populated graph and the seed metric file on node ``A``.
    """
    verts = _fibonacci_sphere(N_VERTICES)
    triangles = ConvexHull(verts.astype(np.float64)).simplices.astype(np.int32)

    template = tmp_path / "template_sphere.surf.gii"
    _save_surface(template, verts, triangles)

    edge_files: dict[tuple[str, str], Path] = {}
    for edge, degrees in _EDGE_ROTATIONS.items():
        if corrupt_edge is not None and edge == corrupt_edge:
            degrees = 90.0
        rotated = (_rot_x(degrees) @ verts.T).T
        path = tmp_path / f"{edge[0]}_to_{edge[1]}_sphere.surf.gii"
        _save_surface(path, rotated, triangles)
        edge_files[edge] = path

    metric_file = tmp_path / "metric_A.func.gii"
    _save_metric(metric_file, verts.sum(axis=1))

    def _node(name: str) -> dict:
        return {
            name: {
                "species": "synthetic",
                "description": f"synthetic node {name}",
                "surfaces": {
                    DENSITY: {
                        "sphere": {"left": str(template), "right": str(template)},
                        # A real midthickness would drive area correction; here
                        # the template sphere is a self-consistent stand-in.
                        "midthickness": {"left": str(template), "right": str(template)},
                    }
                },
            }
        }

    def _edge(src: str, dst: str) -> dict:
        surf = str(edge_files[(src, dst)])
        return {
            "from": src,
            "to": dst,
            "surfaces": {
                "synthetic": {DENSITY: {"sphere": {"left": surf, "right": surf}}}
            },
        }

    data = {
        "nodes": [_node("A"), _node("B"), _node("C")],
        "edges": {
            "surface_to_surface": [_edge(s, d) for (s, d) in _EDGE_ROTATIONS],
            "volume_to_volume": [],
        },
    }

    graph = NeuromapsGraph(runner="local", data_dir=tmp_path / ".cache", _testing=True)
    graph._builder.build_from_dict(graph, data)
    return graph, metric_file


def _fake_metric_resample(
    input_file_path: str | Path,
    current_sphere: str | Path,
    new_sphere: str | Path,
    method: str,  # noqa: ARG001 (kept for signature parity with the real fn)
    area_surfs: dict,  # noqa: ARG001
    output_file_path: str,
) -> SimpleNamespace:
    """Numpy stand-in for ``metric_resample`` that avoids ``wb_command``."""
    resampled = _resample_barycentric(
        _load_metric(input_file_path),
        _load_coords(current_sphere),
        _load_coords(new_sphere),
    )
    _save_metric(Path(output_file_path), resampled)
    return SimpleNamespace(metric_out=Path(output_file_path))


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture
def patch_resample(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the transformer's ``wb_command`` resample for the numpy stand-in.

    ``monkeypatch`` reverts the attribute automatically at the end of the test.
    """
    monkeypatch.setattr(
        "neuromaps_prime.graph.transforms.surface.metric_resample",
        _fake_metric_resample,
    )


@pytest.fixture
def rotation_network(tmp_path: Path) -> tuple[NeuromapsGraph, Path]:
    """Build the healthy (+/-120 degree) three-node rotation network."""
    return _build_rotation_network(tmp_path)


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #
EXPECTED_PATHS = {
    ("A", "B", "A"),
    ("A", "C", "A"),
    ("A", "B", "C", "A"),
    ("A", "C", "B", "A"),
}


def test_find_return_paths_enumerates_every_cycle(
    rotation_network: tuple[NeuromapsGraph, Path],
) -> None:
    """Every simple return path from A is found and closed at A."""
    graph, _ = rotation_network
    paths = find_return_paths(graph, "A")
    assert set(paths) == EXPECTED_PATHS
    assert all(p[0] == "A" and p[-1] == "A" for p in paths)


def test_find_return_paths_respects_max_length(
    rotation_network: tuple[NeuromapsGraph, Path],
) -> None:
    """A length bound drops the longer three-hop cycles."""
    graph, _ = rotation_network
    two_hop = find_return_paths(graph, "A", max_length=2)
    assert set(two_hop) == {("A", "B", "A"), ("A", "C", "A")}


def test_find_return_paths_unknown_origin(
    rotation_network: tuple[NeuromapsGraph, Path],
) -> None:
    """An origin outside the surface layer raises a clear error."""
    graph, _ = rotation_network
    with pytest.raises(ValueError, match="not in the 'surface_to_surface' layer"):
        find_return_paths(graph, "Z")


@pytest.mark.usefixtures("patch_resample")
def test_all_cycles_recover_identity(
    rotation_network: tuple[NeuromapsGraph, Path], tmp_path: Path
) -> None:
    """Because +/-120 degree cycles compose to identity, every r is ~1."""
    graph, metric_file = rotation_network
    results = run_cycle_test(
        graph, "A", metric_file, HEMISPHERE, tmp_path, density=DENSITY
    )

    assert {r.path for r in results} == EXPECTED_PATHS
    for result in results:
        assert result.pearson_r > 0.999, (
            f"{result.label}: r={result.pearson_r:.6f} should be ~1 for an "
            "identity round-trip"
        )
        # Residual is pure interpolation error, small relative to the metric
        # range (coordinate sums span roughly [-1.7, 1.7]).
        assert result.max_abs_diff < 0.1


@pytest.mark.usefixtures("patch_resample")
def test_corrupted_edge_breaks_only_paths_through_it(tmp_path: Path) -> None:
    """A wrong transform lowers r for paths that use it, and only those.

    This is the discriminating-power check: it proves the cycle test would
    actually catch a bad transform rather than passing unconditionally.
    """
    graph, metric_file = _build_rotation_network(tmp_path, corrupt_edge=("A", "B"))
    results = {
        r.path: r.pearson_r
        for r in run_cycle_test(
            graph, "A", metric_file, HEMISPHERE, tmp_path, density=DENSITY
        )
    }

    # Paths that traverse the corrupted A -> B edge degrade markedly...
    assert results[("A", "B", "A")] < 0.95
    assert results[("A", "B", "C", "A")] < 0.95
    # ...while paths that avoid it stay essentially perfect.
    assert results[("A", "C", "A")] > 0.999
    assert results[("A", "C", "B", "A")] > 0.999
