"""Regression test for surface transformation cycle accuracy.

This test validates end-to-end transformation accuracy on the real
NeuromapsPrime graph using available surface transformations.

A surface annotation is initialized in an origin template space, transformed
through valid return paths in the surface transformation graph, and compared
against the original annotation after returning to the starting space.

The unit cycle tests validate the cycle-testing machinery using a synthetic
identity graph. This regression test extends that validation to real surface
transformations and detects future changes that affect transformation accuracy.

Metrics recorded for each cycle include:
- Pearson correlation between the original and round-tripped annotation.
- Maximum absolute difference between original and transformed metrics.
- Number of transformations in the cycle.

Additional domain-specific metrics (e.g., eigenmode preservation) can be added
as regression criteria as additional validation targets are implemented.

To test another template or annotation, update ``ORIGIN``, ``LABEL``, and
``HEMISPHERE`` below.

Run with::

    pytest tests/regression/regression_test_cycle.py -v -s
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from neuromaps_prime.graph import NeuromapsGraph
from tests.cycle import find_return_paths, run_cycle_test


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Test parameters
# -------------------------------------------------------------------------

# Start with spaces containing available surface transformations.
ORIGIN = "Yerkes19"
LABEL = "RM_auto_ampa"
HEMISPHERE = "left"

# Limit cycle enumeration to keep runtime tractable as the graph grows.
MAX_CYCLE_LENGTH = 4


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def output_dir() -> Path:
    """Directory for storing cycle regression outputs."""
    directory = Path(__file__).resolve().parent / "cycle_outputs"
    directory.mkdir(parents=True, exist_ok=True)

    return directory


@pytest.fixture
def graph() -> NeuromapsGraph:
    """Create a Neuromaps graph for regression testing."""
    return NeuromapsGraph()


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


def test_surface_transform_cycles(
    graph: NeuromapsGraph,
    output_dir: Path,
) -> None:
    """Validate preservation through complete surface transformation cycles.

    A real surface annotation is propagated through all return paths from the
    origin space and compared with the original data after returning to that
    space. Successful transformations should preserve the annotation with high
    correlation.
    """
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

    return_paths = find_return_paths(
        graph,
        ORIGIN,
        max_length=MAX_CYCLE_LENGTH,
    )

    logger.info(
        "Testing %d transformation cycles from %s",
        len(return_paths),
        ORIGIN,
    )

    assert return_paths, (
        f"No return paths found from '{ORIGIN}' "
        "in the surface transformation graph."
    )

    cycle_results = run_cycle_test(
        graph,
        ORIGIN,
        metric_file,
        HEMISPHERE,
        output_dir,
        density=density,
        max_length=MAX_CYCLE_LENGTH,
    )

    frame = pd.DataFrame(
        {
            "path": result.label,
            "n_hops": len(result.path) - 1,
            "pearson_r": result.pearson_r,
            "max_abs_diff": result.max_abs_diff,
        }
        for result in cycle_results
    ).sort_values(
        "pearson_r",
        ascending=False,
    )

    logger.info(
        "\n=== Surface Cycle Regression: %s (%s, %s) ===\n%s",
        ORIGIN,
        LABEL,
        HEMISPHERE,
        frame.to_string(index=False),
    )

    output_file = (
        output_dir
        / f"cycle_{ORIGIN}_{LABEL}_{HEMISPHERE}.csv"
    )
    frame.to_csv(output_file, index=False)

    logger.info("Saved cycle metrics: %s", output_file)

    # Regression threshold: round-tripped annotations should remain correlated.
    # Thresholds can be adjusted for known lossy transformations.
    assert (frame["pearson_r"] > 0.5).all(), (
        "Surface transformation cycle failed: at least one path "
        f"had poor preservation. See {output_file}."
    )