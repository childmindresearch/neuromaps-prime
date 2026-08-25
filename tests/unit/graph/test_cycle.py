"""Unit tests for cyclic surface transformation utilities.

These tests validate graph traversal and round-trip metric preservation for
surface transformation cycles using a controlled synthetic graph.

The synthetic graph contains three spaces (A, B, and C) connected by known
rotational surface transformations. Forward edges apply +120 degree rotations
around the x-axis, while reverse edges apply -120 degree rotations.

    A → B → A          (+120 - 120)       = origin
    A → C → A          (-120 + 120)       = origin
    A → B → C → A      (+120 + 120 +120)  = origin
    A → C → B → A      (-120 -120 -120)   = origin

A deterministic vertex-wise synthetic metric is propagated through each
cycle. Because each closed path represents an identity transformation, the
round-tripped metric should remain highly correlated with the original metric.

The tests replace Workbench metric resampling with a lightweight numpy-based
implementation to keep execution fast, deterministic, and independent of
external neuroimaging software. This implementation provides only the
resampling behavior required to exercise the cycle-testing code.

These tests validate the cycle traversal and evaluation framework itself.
Accuracy of real surface transformations is evaluated separately in the
regression test suite using real templates and Workbench-based resampling.

Run with::

    pytest tests/unit/graph/test_cycle.py -v -s

Run only the metric preservation test::

    pytest tests/unit/graph/test_cycle.py -v -s -k preserve_metric
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import nibabel as nib
import numpy as np
import pytest
from nibabel.gifti import GiftiDataArray, GiftiImage
from scipy.spatial import ConvexHull, KDTree

from neuromaps_prime.graph import NeuromapsGraph
from tests.cycle import (
    find_return_paths,
    load_metric,
    roundtrip_metric,
    run_cycle_test,
    score_roundtrip,
)


class TestGraphCycle:
    """Round-trip surface cycle tests on a synthetic three-space rotation graph.

    Spaces A, B, and C are connected by known x-axis rotations that compose to
    identity, so a metric propagated around any closed cycle should return to
    its original representation. Workbench resampling is replaced with a
    lightweight numpy implementation to keep the tests fast and deterministic.
    """

    # --- Test parameters ---------------------------------------------------
    N_VERTICES = 642
    DENSITY = "1k"
    HEMISPHERE: Literal["left"] = "left"
    DENOM = 1e-11

    EXPECTED_CYCLES = frozenset(
        {("A", "B", "A"), ("A", "C", "A"), ("A", "B", "C", "A"), ("A", "C", "B", "A")}
    )

    # --- Synthetic geometry helpers ----------------------------------------

    def _fibonacci_sphere(self, n: int) -> np.ndarray:
        """Generate approximately uniformly distributed points on a unit sphere.

        The generated vertices provide deterministic synthetic surface geometry
        for constructing test spheres and vertex-wise metrics.
        """
        indices = np.arange(n) + 0.5

        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5**0.5) * indices

        return np.c_[
            np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
        ].astype(np.float32)

    def _rotation_x(self, degrees: float) -> np.ndarray:
        """Create a 3D rotation matrix around the x-axis.

        Parameters
        ----------
        degrees
            Rotation angle in degrees. Positive and negative values are used to
            create forward and reverse synthetic surface transformations.
        """
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
        self, path: Path, vertices: np.ndarray, triangles: np.ndarray
    ) -> None:
        """Save vertices and triangles as a GIFTI surface."""
        image = GiftiImage()
        image.add_gifti_data_array(
            GiftiDataArray(vertices.astype(np.float32), intent="NIFTI_INTENT_POINTSET")
        )
        image.add_gifti_data_array(
            GiftiDataArray(triangles.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE")
        )
        nib.save(image, path)

    def _save_metric(self, path: Path, metric: np.ndarray) -> None:
        """Save a vertex-wise metric as a GIFTI functional file."""
        image = GiftiImage()
        image.add_gifti_data_array(
            GiftiDataArray(metric.astype(np.float32), intent="NIFTI_INTENT_NONE")
        )
        nib.save(image, path)

    def _load_vertices(self, path: Path | str) -> np.ndarray:
        """Load vertex coordinates from a GIFTI surface."""
        image = nib.load(path)
        for array in image.darrays:
            if array.data.ndim == 2 and array.data.shape[1] == 3:
                return np.asarray(array.data, dtype=float)
        raise ValueError(f"No vertices found in {path}")

    # --- Resampling mock ---------------------------------------------------

    def _resample_metric(
        self,
        metric: np.ndarray,
        source_vertices: np.ndarray,
        target_vertices: np.ndarray,
    ) -> np.ndarray:
        """Resample a vertex-wise metric using a lightweight barycentric approximation.

        The three nearest source vertices are used to estimate interpolation
        weights for each target vertex. This approximates the behavior needed for
        testing metric propagation through surface transformation cycles without
        requiring Workbench.
        """
        _, indices = KDTree(source_vertices).query(target_vertices, k=3)

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
            np.abs(denominator) < self.DENOM, self.DENOM, denominator
        )

        v = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
        u = 1 - v - w

        weights = np.column_stack((u, v, w))
        return np.einsum("ij,ij->i", weights, metric[indices])

    def _fake_metric_resample(
        self,
        input_file_path: str | Path,
        current_sphere: str | Path,
        new_sphere: str | Path,
        method: str,  # noqa: ARG002 - required to match Workbench interface
        area_surfs: dict[str, str | Path],  # noqa: ARG002 - required to match Workbench interface
        output_file_path: str,
    ) -> SimpleNamespace:
        """Mock Workbench metric resampling for isolated cycle tests.

        Loads the input metric, applies the numpy-based resampling approximation
        between synthetic sphere geometries, writes the resulting GIFTI metric,
        and returns an object matching the interface expected from Workbench.
        """
        # load input and resample from current sphere to new sphere
        result = self._resample_metric(
            load_metric(input_file_path),
            self._load_vertices(current_sphere),
            self._load_vertices(new_sphere),
        )
        # write resampled metric to the expected output path
        self._save_metric(Path(output_file_path), result)
        # expose path to the newly created metric file like workbench would
        return SimpleNamespace(metric_out=Path(output_file_path))

    # --- Fixtures ----------------------------------------------------------

    @pytest.fixture
    def rotation_graph(self, tmp_path: Path) -> NeuromapsGraph:
        """Create a synthetic graph containing identity surface cycles.

        The graph contains three spaces (A, B, and C) with forward and reverse
        rotational surface transformations. Each closed path starting at A
        composes to an identity rotation, allowing cycle traversal logic to be
        tested independently of real template transformations.
        """
        vertices = self._fibonacci_sphere(self.N_VERTICES)
        triangles = ConvexHull(vertices).simplices

        sphere = tmp_path / "sphere.surf.gii"
        self._save_surface(sphere, vertices, triangles)

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
            transformed = (self._rotation_x(angle) @ vertices.T).T

            edge_surface = tmp_path / f"{source}_{target}.surf.gii"
            self._save_surface(edge_surface, transformed, triangles)

            edges.append(
                {
                    "from": source,
                    "to": target,
                    "surfaces": {
                        "synthetic": {
                            self.DENSITY: {
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
                        self.DENSITY: {
                            "sphere": {"left": str(sphere), "right": str(sphere)},
                            "midthickness": {"left": str(sphere), "right": str(sphere)},
                        }
                    }
                }
            }

        graph = NeuromapsGraph(
            runner="local", data_dir=tmp_path / "cache", _testing=True
        )
        graph._builder.build_from_dict(
            graph,
            {
                "nodes": [node("A"), node("B"), node("C")],
                "edges": {"surface_to_surface": edges, "volume_to_volume": []},
            },
        )
        return graph

    @pytest.fixture
    def rotation_metric(self, tmp_path: Path) -> Path:
        """Create a deterministic synthetic vertex-wise metric.

        The metric is generated by summing the x, y, and z coordinates of the
        synthetic sphere vertices. This produces a spatial pattern that can be
        propagated through transformation cycles and compared against the original
        metric after round-trip resampling.

        Returns:
        -------
        Path
            Path to the saved GIFTI functional metric file.
        """
        vertices = self._fibonacci_sphere(self.N_VERTICES)
        metric = tmp_path / "metric.func.gii"
        self._save_metric(metric, vertices.sum(axis=1))
        return metric

    @pytest.fixture
    def patch_metric_resample(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace Workbench metric resampling with the numpy test implementation.

        This isolates cycle testing from external Workbench dependencies while
        preserving the expected resampling function interface.
        """
        monkeypatch.setattr(
            "neuromaps_prime.graph.transforms.surface.metric_resample",
            self._fake_metric_resample,
        )

    # --- Tests -------------------------------------------------------------

    def test_find_return_paths_returns_expected_cycles(
        self, rotation_graph: NeuromapsGraph
    ) -> None:
        """Find all simple directed cycles containing the synthetic graph node A."""
        paths = find_return_paths(rotation_graph, "A")
        assert set(paths) == self.EXPECTED_CYCLES

    def test_find_return_paths_rejects_unknown_node(
        self, rotation_graph: NeuromapsGraph
    ) -> None:
        """Raise an error when the graph origin node does not exist."""
        with pytest.raises(ValueError, match="not in the 'surface_to_surface' layer"):
            find_return_paths(rotation_graph, "unknown")

    @pytest.mark.usefixtures("patch_metric_resample")
    def test_closed_cycles_preserve_metric(
        self, rotation_graph: NeuromapsGraph, rotation_metric: Path, tmp_path: Path
    ) -> None:
        """Verify that identity transformation cycles preserve a vertex-wise metric.

        Each discovered cycle propagates the synthetic metric through the surface
        transformations and back to the starting space. Since the synthetic
        transformations compose to identity, the returned metric should closely
        match the original metric.
        """
        workdir = tmp_path / "output"
        workdir.mkdir()

        results = run_cycle_test(
            rotation_graph,
            "A",
            rotation_metric,
            self.HEMISPHERE,
            workdir,
            density=self.DENSITY,
        )

        paths = {result.path for result in results}

        assert self.EXPECTED_CYCLES.issubset(paths)

        for result in results:
            assert result.pearson_r == pytest.approx(1.0, abs=1e-3), (
                f"Cycle {result.path} had pearson_r={result.pearson_r}"
            )

    @pytest.mark.usefixtures("patch_metric_resample")
    def test_concatenated_transform_matches_direct_transform(
        self, rotation_graph: NeuromapsGraph, rotation_metric: Path, tmp_path: Path
    ) -> None:
        """A composed transform matches the equivalent direct transform.

        The synthetic graph contains both a direct A → C transformation and an
        indirect A → B → C route. Since the transforms are constructed from known
        rotations, concatenating the intermediate transforms should produce the
        same result as applying the direct transform.

        This validates transform composition independently of cycle closure.
        """
        workdir = tmp_path / "output"
        workdir.mkdir()

        # Transform metric directly from A to C.
        direct = roundtrip_metric(
            rotation_graph,
            rotation_metric,
            ("A", "C"),
            self.HEMISPHERE,
            workdir,
            density=self.DENSITY,
        )

        # Transform metric through the intermediate space B.
        concatenated = roundtrip_metric(
            rotation_graph,
            rotation_metric,
            ("A", "B", "C"),
            self.HEMISPHERE,
            workdir,
            density=self.DENSITY,
        )

        pearson_r, max_abs_diff = score_roundtrip(
            direct.final_metric,
            concatenated.final_metric,
        )

        assert pearson_r == pytest.approx(1.0, abs=1e-3)
        assert max_abs_diff == pytest.approx(0.0, abs=0.02)
