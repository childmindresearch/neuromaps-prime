"""Cycle regression test on the real graph.

Round-trips a real surface annotation around every return path from an origin
space and reports the Pearson correlation between the original metric and each
round-trip. High correlation means the transforms on that path compose close to
the identity; lower correlation flags lower-quality paths.

This is the deployed counterpart to the hermetic unit test in
``tests/unit/graph/_unittest_cycle.py``. Both call the same machinery in
``tests/cycle.py``; the unit test proves that machinery returns
r ~ 1 on a synthetic identity network, while this test measures the *real*
transforms and therefore needs Workbench and network access (like
``test_surf_matrix.py``).

Edit ``ORIGIN``, ``LABEL``, and ``HEMISPHERE`` for the space/annotation you want
to probe, then run::

    pytest tests/regression/regression_test_cycle.py -s
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tests.cycle import find_return_paths, run_cycle_test

from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)

output_dir = Path(__file__).resolve().parent / "cycle_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# --- configure the probe ---------------------------------------------------- #
ORIGIN = "fsLR"
LABEL = "RM_scalinghcp"
HEMISPHERE = "left"
# Bound path length so cycle enumeration stays tractable on the dense real
# graph (number of simple cycles grows combinatorially).
MAX_CYCLE_LENGTH = 4


def test_cycle_roundtrip(tmp_path: Path) -> None:
    """Round-trip an annotation through every return path and log correlations."""
    logging.basicConfig(level=logging.INFO)
    graph = NeuromapsGraph()

    # Seed the metric at the origin's highest density so the round-trip returns
    # to a matching mesh, then score every return path.
    density = graph.find_highest_density(ORIGIN)
    annotation = graph.fetch_surface_annotation(
        space=ORIGIN, label=LABEL, density=density, hemisphere=HEMISPHERE
    )
    assert annotation is not None, (
        f"No annotation '{LABEL}' for {ORIGIN} at density '{density}'. "
        "Pick an origin/label that exists in the node YAMLs."
    )
    metric_file = Path(annotation.fetch())

    paths = find_return_paths(graph, ORIGIN, max_length=MAX_CYCLE_LENGTH)
    logger.info("Found %d return paths from %s", len(paths), ORIGIN)
    assert paths, f"No return paths from '{ORIGIN}' on the surface layer."

    results = run_cycle_test(
        graph,
        ORIGIN,
        metric_file,
        HEMISPHERE,
        tmp_path,
        max_length=MAX_CYCLE_LENGTH,
    )

    frame = pd.DataFrame(
        [
            {
                "path": r.label,
                "n_hops": len(r.path) - 1,
                "pearson_r": r.pearson_r,
                "max_abs_diff": r.max_abs_diff,
            }
            for r in results
        ]
    ).sort_values("pearson_r", ascending=False)

    logger.info("\n=== CYCLE TEST (%s, %s, %s) ===", ORIGIN, LABEL, HEMISPHERE)
    logger.info("\n%s", frame.to_string(index=False))

    csv_path = output_dir / f"cycle_{ORIGIN}_{LABEL}_{HEMISPHERE}.csv"
    frame.to_csv(csv_path, index=False)
    logger.info("Saved CSV -> %s", csv_path)

    # A metric round-tripped to its own space and back should stay correlated
    # with itself; a hard failure here indicates a broken or misdirected
    # transform rather than mere registration error.
    assert (frame["pearson_r"] > 0.5).all(), (
        "At least one return path lost nearly all correlation - inspect the "
        f"transforms on the low-scoring paths in {csv_path}."
    )
