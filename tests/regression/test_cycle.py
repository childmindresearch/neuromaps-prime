"""Regression benchmark for surface transformation cycle accuracy.

This benchmark evaluates round-trip surface transformations on the real
NeuromapsPrime graph.

A real surface annotation is initialized in an origin template space and
propagated through the shortest executable return paths in the surface
transformation graph. The round-tripped annotation is compared with the
original using Pearson correlation and maximum vertex-wise absolute
difference.

The benchmark records transformation accuracy rather than enforcing fixed
pass/fail thresholds. Results are written to CSV for comparison across
changes to the transformation machinery.

Only cycles with all required surface resources available are evaluated.

Intermediate transformation files are written to pytest's temporary
directory and are removed after the test completes. Plotting is handled
separately by ``plot_cycle_regression.py``.

To test another template or annotation, update ``ORIGIN``, ``LABEL``, and
``HEMISPHERE`` below.

Run with::

    pytest tests/regression/test_cycle_regression.py -v -s
"""

from __future__ import annotations

import csv
import logging
from itertools import pairwise
from pathlib import Path

import pytest
from tests.cycle import find_return_paths, roundtrip_metric, score_roundtrip

from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Regression parameters
# -------------------------------------------------------------------------

ORIGIN = "Yerkes19"
LABEL = "RM_auto_ampa"
HEMISPHERE = "left"


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def output_dir() -> Path:
    """Directory for storing persistent cycle benchmark outputs."""
    directory = Path(__file__).resolve().parent / "output/cycle_outputs"
    directory.mkdir(parents=True, exist_ok=True)

    return directory


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _shortest_paths(
    paths: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Return all shortest paths from a collection of return paths."""
    if not paths:
        return []

    shortest_length = min(len(path) for path in paths)

    return [
        path
        for path in paths
        if len(path) == shortest_length
    ]


def _valid_cycle_paths(
    graph: NeuromapsGraph,
    paths: list[tuple[str, ...]],
    density: str,
    hemisphere: str,
) -> list[tuple[str, ...]]:
    """Return cycles whose required surface resources are available."""
    valid_paths = []

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


def _write_benchmark_csv(
    results: list[dict[str, object]],
    output_file: Path,
) -> None:
    """Write cycle benchmark results to CSV."""
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "origin",
                "label",
                "hemisphere",
                "path",
                "n_hops",
                "pearson_r",
                "max_abs_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


def test_surface_transform_cycles(
    graph: NeuromapsGraph,
    output_dir: Path,
    tmp_path: Path,
) -> None:
    """Benchmark shortest executable transformation cycles."""
    logging.basicConfig(level=logging.INFO)

    density = graph.find_highest_density(ORIGIN)

    annotation = graph.fetch_surface_annotation(
        space=ORIGIN,
        label=LABEL,
        density=density,
        hemisphere=HEMISPHERE,
    )

    assert annotation is not None, (
        f"Missing annotation '{LABEL}' for {ORIGIN} "
        f"({HEMISPHERE}, density={density})."
    )

    metric_file = Path(annotation.fetch())

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

    # Transformation outputs are temporary, like the unit tests.
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
            "origin": ORIGIN,
            "label": LABEL,
            "hemisphere": HEMISPHERE,
            "path": " -> ".join(path),
            "n_hops": len(path) - 1,
            "pearson_r": pearson_r,
            "max_abs_diff": max_abs_diff,
        }

        results.append(result)

        logger.info(
            "%-50s  r=%10.6f  max|delta|=%12.3e  hops=%d",
            result["path"],
            pearson_r,
            max_abs_diff,
            len(path) - 1,
        )

    output_file = (
        output_dir
        / f"cycle_{ORIGIN}_{LABEL}_{HEMISPHERE}.csv"
    )

    _write_benchmark_csv(
        results,
        output_file,
    )

    logger.info(
        "Saved cycle benchmark metrics: %s",
        output_file,
    )