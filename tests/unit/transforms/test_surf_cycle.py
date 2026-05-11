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
from neuromaps_prime.transforms.surface import metric_resample

logger = logging.getLogger(__name__)


def test_surface_cycle(tmp_path: Path) -> None:
    """Test surface consistency by cycling through surface sphere transforms."""
    logging.basicConfig(level=logging.INFO)

    logger.info("=== BUILDING NEUROMAPS GRAPH ===")

    graph = NeuromapsGraph(
        runner="local",
        data_dir=Path("/Users/tamsin.rogers/Desktop/github/neuromaps-prime"),
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
        logger.info("=== Cycle type %s: ===", type(cycle[0]))

        metric_output = full_surface_path

        for step, (src, dst) in enumerate(pairwise(cycle)):
            out_file = tmp_path / f"cycle{i}_step{step}_{src}_to_{dst}.shape.gii"

            logger.info("Step %s: %s -> %s", step, src, dst)

            # original Yerkes19 midthickness
            if step==0:
                current_metric = Path("/Users/tamsin.rogers/Desktop/github/neuromaps-prime/share/Inputs/Yerkes19/src-Yerkes19_den-32k_hemi-R_desc-vaavg_midthickness.shape.gii")
            # result of transform in previous step
            else:
                current_metric = result
            
            # original Yerkes19 surf sphere
            current_sphere = graph.fetch_surface_atlas(
                space=src,
                    density=graph.find_highest_density(space=src),
                    hemisphere=hemisphere,
                    resource_type="sphere",
                ).file_path

            # now fetch the new target sphere
            target_sphere = graph.surface_ops._resolve_sphere_transform(
                source=src,
                target=dst,
                density=graph.find_common_density(mid_space=src, target_space=dst),
                hemisphere=hemisphere,
                output_file_path="output.sphere.gii",
            ).file_path

            # midthickness files - so we know where (on the sphere) to map transformation values to
            area_surfs = {
                "current-area": graph.fetch_surface_atlas(
                    space=src,
                    density=graph.find_highest_density(space=src),
                    hemisphere=hemisphere,
                    resource_type="midthickness"
                    ).file_path,
                "new-area": graph.fetch_surface_atlas(
                    space=dst,
                    density=graph.find_highest_density(space=src),
                    hemisphere=hemisphere,
                    resource_type="midthickness"
                    ).file_path
                }
            
            result = metric_resample(
                input_file_path=str(current_metric),
                current_sphere=str(current_sphere),
                new_sphere=str(target_sphere),
                method="ADAP_BARY_AREA",
                area_surfs=area_surfs,
                output_file_path=str(out_file),
            ).metric_out

            metric_output = Path(result)

            #shapes = log_gii_shapes(metric_output)
            arrays = nib.load(metric_output).darrays

            logger.info(
                "OK %s -> %s | arrays=%s",
                src,
                dst,
                [len(a.data) for a in arrays],
            )

            logger.info("Step %s complete: %s", step, metric_output)

    error_file = tmp_path / f"cycle{i}_error.func.gii"

    workbench.signed_distance_to_surface(
        surface_comp=str(metric_output),
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