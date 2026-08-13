"""Regression benchmark for shortest surface transformation cycles.

This test benchmarks end-to-end surface transformation accuracy on the real
NeuromapsPrime graph using standard template surfaces.

A deterministic vertex-wise metric is generated from the origin template's
midthickness surface, propagated through every shortest executable return path
in the surface transformation graph, and compared with the original metric
after returning to the starting space.

The benchmark is intentionally independent of surface annotations. The metric
is generated directly from the standard midthickness surface so that private
annotation resources do not affect the test.

Only the shortest return paths are evaluated. This keeps the benchmark focused
on the most direct transformation routes while still covering every shortest
path possibility.

Transformation outputs are written to pytest's temporary directory. The
benchmark summary is saved separately under ``tests/regression/output`` so
that results can be inspected after the test run.

Metrics recorded for each transformation cycle include:

- Pearson correlation between original and round-tripped metric.
- Maximum absolute difference between original and round-tripped values.
- Number of transformations in the cycle.

This test is intended primarily as a benchmark rather than a strict regression
test. Results are recorded for comparison across changes to transformation
infrastructure.

Run with::

    pytest tests/regression/test_cycle.py -v -s
"""

from __future__ import annotations

import logging
from itertools import pairwise
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neuromaps_prime.graph import NeuromapsGraph
from tests.cycle import find_return_paths, roundtrip_metric, score_roundtrip

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Regression parameters
# -------------------------------------------------------------------------

ORIGIN = "Yerkes19"
HEMISPHERE = "left"


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def graph() -> NeuromapsGraph:
    """Create a Neuromaps graph for regression benchmarking."""
    return NeuromapsGraph()


@pytest.fixture
def output_dir() -> Path:
    """Directory for storing benchmark summaries."""
    directory = Path(__file__).resolve().parent / "output/cycle_outputs"
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _make_surface_metric(
    surface_file: Path,
    output_file: Path,
) -> Path:
    """Create a deterministic vertex-wise metric from a midthickness surface."""
    image = nib.load(surface_file)

    vertices = np.asarray(
        image.darrays[0].data,
        dtype=np.float64,
    )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Expected surface vertices with shape (n_vertices, 3), "
            f"got {vertices.shape}."
        )

    # A smooth deterministic metric based only on the standard surface
    # geometry. This avoids dependence on any external annotation.
    metric = vertices.sum(axis=1)

    metric_image = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                metric.astype(np.float32),
            )
        ]
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        metric_image,
        output_file,
    )

    return output_file


def _shortest_paths(
    paths: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Return all paths having the minimum number of transformation hops."""
    if not paths:
        return []

    shortest_length = min(len(path) - 1 for path in paths)

    return [
        path
        for path in paths
        if len(path) - 1 == shortest_length
    ]


def _valid_cycle_paths(
    graph: NeuromapsGraph,
    paths: list[tuple[str, ...]],
    density: str,
    hemisphere: str,
) -> list[tuple[str, ...]]:
    """Return cycles whose required surface resources are available."""
    valid_paths: list[tuple[str, ...]] = []

    for path in paths:
        valid = True

        for source, target in pairwise(path):
            try:
                graph._cache.require_surface_atlas(
                    source,
                    density,
                    hemisphere,
                    "sphere",
                )
                graph._cache.require_surface_atlas(
                    target,
                    density,
                    hemisphere,
                    "sphere",
                )
            except ValueError:
                valid = False
                break

        if valid:
            valid_paths.append(path)

    return valid_paths


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


def test_surface_transform_cycles(
    graph: NeuromapsGraph,
    output_dir: Path,
    tmp_path: Path,
) -> None:
    """Benchmark shortest executable surface transformation cycles."""
    logging.basicConfig(level=logging.INFO)

    density = graph.find_highest_density(ORIGIN)

    # ------------------------------------------------------------------
    # Create a deterministic metric from the standard midthickness surface.
    # ------------------------------------------------------------------

    surface = graph._cache.require_surface_atlas(
        ORIGIN,
        density,
        HEMISPHERE,
        "midthickness",
    )

    surface_file = Path(surface)

    assert surface_file.exists(), (
        f"Missing midthickness surface: {surface_file}"
    )

    metric_file = _make_surface_metric(
        surface_file,
        tmp_path / f"{ORIGIN}_{density}_{HEMISPHERE}_metric.func.gii",
    )

    assert metric_file.exists()
    assert metric_file.stat().st_size > 0

    # ------------------------------------------------------------------
    # Find all return paths and retain every shortest possibility.
    # ------------------------------------------------------------------

    all_paths = find_return_paths(
        graph,
        ORIGIN,
    )

    assert all_paths, (
        f"No return paths found from '{ORIGIN}' "
        "in the surface transformation graph."
    )

    shortest_paths = _shortest_paths(all_paths)

    valid_paths = _valid_cycle_paths(
        graph,
        shortest_paths,
        density,
        HEMISPHERE,
    )

    logger.info(
        "Surface cycle benchmark for %s",
        ORIGIN,
    )
    logger.info(
        "Density: %s",
        density,
    )
    logger.info(
        "Total return paths: %d",
        len(all_paths),
    )
    logger.info(
        "Shortest return paths: %d",
        len(shortest_paths),
    )
    logger.info(
        "Executable shortest return paths: %d",
        len(valid_paths),
    )

    assert valid_paths, (
        f"No executable shortest transformation cycles found from '{ORIGIN}'."
    )

    # ------------------------------------------------------------------
    # Run each shortest cycle.
    #
    # Intermediate transformation files stay in tmp_path, just like the
    # unit tests. Nothing from the transformation itself is written into
    # the persistent regression output directory.
    # ------------------------------------------------------------------

    workdir = tmp_path / "cycle_outputs"
    workdir.mkdir()

    results: list[dict[str, object]] = []

    for path in valid_paths:
        roundtrip = roundtrip_metric(
            graph,
            metric_file,
            path,
            HEMISPHERE,
            workdir,
            density=density,
        )

        pearson_r, max_abs_diff = score_roundtrip(
            metric_file,
            roundtrip,
        )

        result = {
            "path": " -> ".join(path),
            "n_hops": len(path) - 1,
            "pearson_r": pearson_r,
            "max_abs_diff": max_abs_diff,
        }

        results.append(result)

        logger.info(
            "cycle %s: r=%.6f max|delta|=%.3e",
            result["path"],
            pearson_r,
            max_abs_diff,
        )

    # ------------------------------------------------------------------
    # Summarize benchmark results.
    # ------------------------------------------------------------------

    frame = pd.DataFrame(results).sort_values(
        "pearson_r",
        ascending=False,
    )

    logger.info(
        "\n=== Surface Cycle Benchmark: %s (%s, %s) ===\n%s",
        ORIGIN,
        density,
        HEMISPHERE,
        frame.to_string(index=False),
    )

    output_file = (
        output_dir
        / f"cycle_{ORIGIN}_{density}_{HEMISPHERE}.csv"
    )

    frame.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved cycle benchmark results: %s",
        output_file,
    )

    # ------------------------------------------------------------------
    # Basic sanity checks.
    #
    # This is a benchmark, not a hard threshold regression test. We still
    # require finite comparison metrics so that an invalid transformation
    # cannot silently produce a meaningless benchmark.
    # ------------------------------------------------------------------

    assert np.isfinite(frame["pearson_r"]).all(), (
        "Benchmark produced non-finite Pearson correlations."
    )

    assert np.isfinite(frame["max_abs_diff"]).all(), (
        "Benchmark produced non-finite maximum absolute differences."
    )