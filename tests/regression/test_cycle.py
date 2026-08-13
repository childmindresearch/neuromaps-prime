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

Transformation outputs are temporary and are removed after the test.

Metrics recorded for each transformation cycle include:

- Pearson correlation between original and round-tripped metric.
- Maximum absolute difference between original and round-tripped values.
- Number of transformations in the cycle.

Run with::

    pytest tests/regression/test_cycle.py -v -s
"""

from __future__ import annotations

import logging
from itertools import pairwise
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from tests.cycle import (
    find_return_paths,
    save_cycle_figure,
    score_roundtrip,
)

from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Regression parameters
# -------------------------------------------------------------------------

ORIGIN = "Yerkes19"
HEMISPHERE: Literal["left", "right"] = "left"


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


def _find_executable_density(
    graph: NeuromapsGraph,
    source: str,
    target: str,
    hemisphere: Literal["left", "right"],
) -> str | None:
    """Find a density at which a surface transform can be executed."""
    try:
        density = graph.find_common_density(source, target)
    except ValueError:
        return None

    transform = graph.fetch_surface_to_surface_transform(
        source=source,
        target=target,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    )

    if transform is None:
        return None

    return density


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
            "Expected surface vertices with shape (n_vertices, 3), "
            f"got {vertices.shape}."
        )

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

    return [path for path in paths if len(path) - 1 == shortest_length]


def _valid_cycle_paths(
    graph: NeuromapsGraph,
    paths: list[tuple[str, ...]],
    hemisphere: Literal["left", "right"],
) -> list[tuple[str, ...]]:
    """Return cycles for which every hop has a common executable density."""
    return [
        path
        for path in paths
        if all(
            _find_executable_density(
                graph,
                source,
                target,
                hemisphere,
            )
            is not None
            for source, target in pairwise(path)
        )
    ]


def _transform_cycle(
    graph: NeuromapsGraph,
    metric_file: Path,
    path: tuple[str, ...],
    hemisphere: Literal["left", "right"],
) -> Path:
    """Transform a metric through every hop in a cycle."""
    current_metric = metric_file

    for hop, (source, target) in enumerate(pairwise(path)):
        density = graph.find_common_density(source, target)

        output_name = f"hop{hop:02d}_{source}-to-{target}.func.gii"

        result = graph.surface_to_surface_transformer(
            transformer_type="metric",
            input_file=current_metric,
            source_space=source,
            target_space=target,
            hemisphere=hemisphere,
            output_file_path=output_name,
            source_density=None,
            target_density=density,
        )

        if result is None:
            raise RuntimeError(
                f"No surface transform for hop '{source}' -> '{target}' "
                f"on path {' -> '.join(path)}"
            )

        current_metric = Path(result)

        if not current_metric.exists():
            raise FileNotFoundError(
                "Surface transformation did not produce an output file: "
                f"{current_metric}"
            )

    return current_metric


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

    surface = graph.fetch_surface_atlas(
        space=ORIGIN,
        density=density,
        hemisphere=HEMISPHERE,
        resource_type="midthickness",
    )

    assert surface is not None, (
        f"Missing midthickness surface for {ORIGIN} at {density} ({HEMISPHERE})."
    )

    surface_file = Path(surface.fetch())

    assert surface_file.exists(), f"Missing midthickness surface: {surface_file}"

    metric_file = _make_surface_metric(
        surface_file,
        tmp_path / f"{ORIGIN}_{density}_{HEMISPHERE}_metric.func.gii",
    )

    assert metric_file.exists(), f"Failed to create metric: {metric_file}"
    assert metric_file.stat().st_size > 0, f"Created metric is empty: {metric_file}"

    # ------------------------------------------------------------------
    # Find all return paths and retain every shortest possibility.
    # ------------------------------------------------------------------

    all_paths = find_return_paths(
        graph,
        ORIGIN,
    )

    assert all_paths, (
        f"No return paths found from '{ORIGIN}' in the surface transformation graph."
    )

    shortest_paths = _shortest_paths(all_paths)

    valid_paths = _valid_cycle_paths(
        graph,
        shortest_paths,
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
    # All transformation outputs are kept under pytest's temporary
    # directory. They are automatically removed when the test completes.
    # ------------------------------------------------------------------

    results: list[dict[str, object]] = []

    for cycle_number, path in enumerate(valid_paths):
        workdir = tmp_path / f"cycle_{cycle_number:02d}"
        workdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        roundtrip = _transform_cycle(
            graph,
            metric_file,
            shortest_paths,
            HEMISPHERE,
            workdir,
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

    output_file = output_dir / f"cycle_{ORIGIN}_{density}_{HEMISPHERE}.csv"

    frame.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved cycle benchmark results: %s",
        output_file,
    )

    figure_file = output_dir / f"cycle_{ORIGIN}_{density}_{HEMISPHERE}.png"

    save_cycle_figure(
        results,
        figure_file,
        title=(f"Surface Cycle Benchmark: {ORIGIN} ({density}, {HEMISPHERE})"),
    )

    logger.info(
        "Saved cycle benchmark figure: %s",
        figure_file,
    )

    # ------------------------------------------------------------------
    # Basic sanity checks.
    #
    # This is a benchmark, not a hard-threshold regression test. We still
    # require finite comparison metrics so that an invalid transformation
    # cannot silently produce a meaningless benchmark.
    # ------------------------------------------------------------------

    assert np.isfinite(frame["pearson_r"]).all(), (
        "Benchmark produced non-finite Pearson correlations."
    )

    assert np.isfinite(frame["max_abs_diff"]).all(), (
        "Benchmark produced non-finite maximum absolute differences."
    )
