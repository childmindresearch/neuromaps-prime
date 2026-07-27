"""Regression test for surface transformation cycle accuracy.

This test validates end-to-end transformation accuracy on the real
NeuromapsPrime graph using available surface transformations.

A real surface annotation is initialized in an origin template space,
propagated through every executable complete return path in the surface
transformation graph, and compared with the original annotation after
returning to the starting space.

The unit cycle tests validate the cycle-testing machinery using a synthetic
identity graph. This regression test extends that validation to real
transformations and detects future changes that affect transformation
accuracy.

Only cycles with all required surface resources available are evaluated.
This prevents failures caused by graph paths that exist logically but cannot
be executed due to missing template densities or surface resources.

Metrics recorded for each transformation cycle include:
- Pearson correlation between original and round-tripped annotation.
- Maximum absolute difference between original and round-tripped values.
- Number of transformations in the cycle.

Regression criteria:
- Minimum Pearson correlation preserves spatial pattern.
- Maximum absolute difference detects numerical transformation errors.

Future domain-specific metrics (e.g., eigenmode preservation) can be added as
additional regression criteria.

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
# Regression parameters
# -------------------------------------------------------------------------

# Start with spaces containing available RheMAP surface transformations.
ORIGIN = "Yerkes19"
LABEL = "RM_auto_ampa"
HEMISPHERE = "left"

# Regression thresholds.
#
# Pearson correlation measures preservation of the spatial pattern.
MIN_PEARSON_R = 0.95

# Maximum vertex-wise numerical deviation allowed after round-trip.
MAX_ABS_DIFF = 0.1


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def graph() -> NeuromapsGraph:
    """Create a Neuromaps graph for regression testing."""
    return NeuromapsGraph()


@pytest.fixture
def output_dir() -> Path:
    """Directory for storing cycle regression outputs."""
    directory = Path(__file__).resolve().parent / "cycle_outputs"
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _valid_cycle_paths(
    graph: NeuromapsGraph,
    paths: list[tuple[str, ...]],
    density: str,
    hemisphere: str,
) -> list[tuple[str, ...]]:
    """Return cycles that have all required surface resources.

    Graph connectivity does not guarantee that every transformation can be
    executed. A cycle is considered valid only if every source and target
    space has the required sphere surface available at the requested density.
    """
    valid_paths = []

    for path in paths:
        valid = True

        for source, target in zip(path[:-1], path[1:]):
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
) -> None:
    """Validate preservation through complete executable transformation cycles.

    A real annotation is propagated through every executable closed
    transformation path originating from the selected template space.

    Each round-trip returns to the original space, where the transformed
    annotation is compared against the input annotation.
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

    # Find all graph cycles.
    all_paths = find_return_paths(
        graph,
        ORIGIN,
    )

    assert all_paths, (
        f"No return paths found from '{ORIGIN}' "
        "in the surface transformation graph."
    )

    # Remove cycles that cannot be executed due to missing surface resources.
    valid_paths = _valid_cycle_paths(
        graph,
        all_paths,
        density,
        HEMISPHERE,
    )

    logger.info(
        "Testing %d/%d executable transformation cycles from %s",
        len(valid_paths),
        len(all_paths),
        ORIGIN,
    )

    assert valid_paths, (
        f"No executable transformation cycles found from '{ORIGIN}'."
    )

    cycle_results = run_cycle_test(
        graph,
        ORIGIN,
        metric_file,
        HEMISPHERE,
        output_dir,
        density=density,
        paths=valid_paths,
    )

    assert cycle_results, "No cycle results were generated."

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

    frame.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved cycle regression metrics: %s",
        output_file,
    )

    failed_correlation = frame[
        frame["pearson_r"] < MIN_PEARSON_R
    ]

    assert failed_correlation.empty, (
        "Surface transformation cycle regression failed: "
        f"Pearson correlation below {MIN_PEARSON_R}. "
        f"See {output_file}."
    )

    failed_error = frame[
        frame["max_abs_diff"] > MAX_ABS_DIFF
    ]

    assert failed_error.empty, (
        "Surface transformation cycle regression failed: "
        f"maximum absolute difference exceeded {MAX_ABS_DIFF}. "
        f"See {output_file}."
    )