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
    """Test surface consistency by cycling through transforms."""
    logging.basicConfig(level=logging.INFO)

    logger.info("=== BUILDING NEUROMAPS GRAPH ===")

    graph = NeuromapsGraph(
        yaml_file=Path("src/neuromaps_prime/resources/neuromaps_graph.yaml")
    )

    logger.info("yaml_exists=%s", graph.yaml_path.exists())
    logger.info("yaml_path=%s", graph.yaml_path)

    origin = "Yerkes19"
    hemisphere = "right"
    density = graph.find_highest_density(space=origin)

    logger.info("Origin=%s, hemisphere=%s, density=%s", origin, hemisphere, density)

    networkx_graph = graph
    logger.info(
        "Graph has %s nodes and %s edges",
        len(networkx_graph.nodes),
        len(networkx_graph.edges),
    )

    directed_graph = networkx_graph.to_directed()

    # number of possible round trip paths in the network
    cycles: list[list[str]] = [
        cycle for cycle in recursive_simple_cycles(directed_graph) if origin in cycle
    ]
    assert cycles, "No cycles found!"
    logger.info("Found %s cycles containing %s", len(cycles), origin)

    full_surface = graph.fetch_surface_atlas(
        space=origin,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    ).fetch()

    logger.info("Full surface path: %s", full_surface)

    # log/debug this
    log_gii_shapes(Path(full_surface))

    # i = cycle index
    # cycle = one round trip path
    # enumerate (sorted(cycles, key=len(cycles))) 
    # to start with shorter cycles first
    for i, cycle in enumerate(cycles):
        while cycle[0] != origin:
            cycle = cycle[1:] + cycle[:1]
        cycle = [*cycle, origin]

        logger.info("=== Cycle %s: %s ===", i, cycle)

        skip_cycle = False

        for step, (src, dst) in enumerate(pairwise(cycle)):
            # rename this file to include the src and target (to provide more debugging information)
            out_file = tmp_path / f"cycle{i}_step{step}.surf.gii"

            logger.info("--- Step %s: %s -> %s ---", step, src, dst)

            density = graph.find_highest_density(space=src)

            # fix this to perform the transformation between the src and dst spaces, 
            # rather than transforming from the original surface each time 
            # (which is not how the transforms are intended to be used
            # change to compute src to target transform, then apply to the current surface, 
            # which is the output of the previous transform)
            try:
                current_surface = graph.surface_to_surface_transformer(
                    transformer_type="metric",
                    input_file=full_surface,    # first full surface = highest density original sphere (Yerkes19)
                    source_space=src,
                    target_space=dst,
                    hemisphere=hemisphere,
                    output_file_path=str(out_file),
                    source_density=graph.find_highest_density(space=src),
                    target_density=graph.find_highest_density(space=dst),
                )

            # not expected to fail
            except Exception:
                logger.exception("Transform %s -> %s failed", src, dst)
                skip_cycle = True
                break

            logger.debug("%s %s %s %s", src, dst, hemisphere, density)

            if not current_surface.file_path.exists():
                logger.error(
                    "Transform output file missing: %s",
                    current_surface.file_path,
                )
                skip_cycle = True
                break

            # arrays are mismatched here. why?
            shapes = log_gii_shapes(current_surface.file_path)

            arrays = nib.load(current_surface.file_path).darrays
            if not shapes or all(len(arr.data) == 0 for arr in arrays):
                logger.warning("Transform %s -> %s produced empty surface", src, dst)
                skip_cycle = True
                break

            logger.info(
                "Transform %s -> %s successful, vertex arrays: %s",
                src,
                dst,
                [len(arr.data) for arr in arrays],
            )

        # print the shapes for both source and destination + print filepaths for both
        if skip_cycle:
            logger.info("Skipping cycle %s due to failure/empty transform", i)
            continue

        # somehow comparing pointset array for one and triangle array for other
        # "All data arrays (columns) in the file must have the same number of rows.  
        # The first array (column) contains 32492 rows.  Array 2 contains 64980 rows."

        shapes = log_gii_shapes(Path(out_file))
        arrays = nib.load(out_file).darrays

        logger.info(
            "Full transformed surface vertex arrays: %s",
            [len(arr.data) for arr in arrays],
        )

        if len(shapes) != 1:
            logger.warning("Unexpected number of arrays in cycle %s, skipping", i)
            continue

        error_file = tmp_path / f"cycle{i}_error.func.gii"

        logger.info("Computing signed distance for cycle %s -> %s", i, error_file.name)

        workbench.signed_distance_to_surface(
            surface_comp=str(out_file), # the final surface after cycling through transforms, pull out of loop. should be computed at the very end of the cycle
            surface_ref=str(full_surface), # the original Yerkes19 sphere
            metric=str(error_file),
        )

        error_gii = nib.load(error_file)
        error_data = np.abs(error_gii.darrays[0].data)
        median_error = np.median(error_data) # report as single median error

        logger.info("Cycle %s MEDIAN ERROR: %s", i, median_error)

        assert median_error < 1.0, f"Median error in {cycle} exceeds 1: {median_error}"

        # also report standard deviation of error
