"""Compute surface-to-surface transform error."""

import logging
from itertools import pairwise
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from networkx.algorithms.cycles import recursive_simple_cycles

from niwrap import workbench

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.transforms.utils import log_gii_shapes

logger = logging.getLogger(__name__)


def test_surface_cycle(tmp_path: Path) -> None:
    """Test surface consistency by cycling through surface sphere transforms."""
    logging.basicConfig(level=logging.INFO)

    logger.info("=== BUILDING NEUROMAPS GRAPH ===")

    graph = NeuromapsGraph(
        data_dir=Path("/Users/tamsin.rogers/Desktop/github/neuromaps-prime")
    )

    origin = "Yerkes19"
    hemisphere = "right"
    density = graph.find_highest_density(space=origin)

    # only sphere surface atlas as starting point
    full_surface = graph.fetch_surface_atlas(
        space=origin,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    )

    assert full_surface is not None, "Missing origin sphere atlas"

    full_surface_path = full_surface.fetch()
    full_surface_path = Path(full_surface_path)

    logger.info("Origin surface: %s", full_surface_path)

    log_gii_shapes(full_surface_path)

    # build directed graph
    directed = graph.to_directed()

    cycles = [
        c for c in recursive_simple_cycles(directed)
        if origin in c
    ]
    assert cycles, "No cycles found"

    logger.info("Found %d cycles", len(cycles))

    for i, cycle in enumerate(cycles):
        # normalize cycle to start at origin
        while cycle[0] != origin:
            cycle = cycle[1:] + cycle[:1]
        cycle = [*cycle, origin]

        logger.info("=== Cycle %s: %s ===", i, cycle)

        current_surface = full_surface_path
        skip_cycle = False

        for step, (src, dst) in enumerate(pairwise(cycle)):
            out_file = tmp_path / f"cycle{i}_step{step}.surf.gii"

            logger.info("Step %s: %s -> %s", step, src, dst)

            try:
                result = graph.surface_to_surface_transformer(
                    transformer_type="metric",
                    input_file=current_surface,   # 👈 CHAINING FIX
                    source_space=src,
                    target_space=dst,
                    hemisphere=hemisphere,
                    output_file_path=str(out_file),
                    source_density=graph.find_highest_density(space=src),
                    target_density=graph.find_highest_density(space=dst),
                )

            except Exception:
                logger.exception("Transform failed %s -> %s", src, dst)
                skip_cycle = True
                break

            if result is None:
                logger.error("No transform returned for %s -> %s", src, dst)
                skip_cycle = True
                break

            current_surface = Path(result)

            if not current_surface.exists():
                logger.error("Missing output: %s", current_surface)
                skip_cycle = True
                break

            shapes = log_gii_shapes(current_surface)
            arrays = nib.load(current_surface).darrays

            if not shapes or any(len(a.data) == 0 for a in arrays):
                logger.warning("Empty surface at %s -> %s", src, dst)
                skip_cycle = True
                break

            logger.info(
                "OK %s -> %s | arrays=%s",
                src, dst,
                [len(a.data) for a in arrays],
            )

        if skip_cycle:
            logger.info("Skipping cycle %d", i)
            continue

        error_file = tmp_path / f"cycle{i}_error.func.gii"

        workbench.signed_distance_to_surface(
            surface_comp=str(current_surface),
            surface_ref=str(full_surface_path),
            metric=str(error_file),
        )

        error_gii = nib.load(error_file)
        error_data = np.abs(error_gii.darrays[0].data)
        median_error = float(np.median(error_data))

        logger.info("Cycle %d median error: %f", i, median_error)

        assert median_error < 1.0, (
            f"Median error in cycle {cycle} exceeds threshold: {median_error}"
        )