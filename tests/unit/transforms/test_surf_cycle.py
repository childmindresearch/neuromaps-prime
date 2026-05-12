"""Compute surface-to-surface transform error."""

import logging
from itertools import pairwise
from pathlib import Path

import nibabel as nib
import numpy as np
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

    cycles = [c for c in recursive_simple_cycles(directed) if origin in c]
    assert cycles, "No cycles found"

    logger.info("Found %d cycles", len(cycles))

    for i, cycle in enumerate(cycles):
        # normalize cycle to start at origin
        while cycle[0] != origin:
            cycle = cycle[1:] + cycle[:1]
        cycle = [*cycle, origin]

        logger.info("=== Cycle %s: %s ===", i, cycle)
        logger.info("=== Cycle type %s: ===", type(cycle[0]))

        output = full_surface_path

        for step, (src, dst) in enumerate(pairwise(cycle)):
            out_file = tmp_path / f"cycle{i}_step{step}_{src}_to_{dst}.surf.gii"

            logger.info("Step %s: %s -> %s", step, src, dst)

            # original Yerkes19 midthickness
            if step == 0:
                # original Yerkes19 surf sphere
                current_sphere = graph.fetch_surface_atlas(
                    space=src,
                    density=graph.find_highest_density(space=src),
                    hemisphere=hemisphere,
                    resource_type="sphere",
                ).file_path
            # result of transform in previous step
            else:
                current_sphere = full_surface

            # now fetch the new target sphere
            target_sphere = graph.surface_ops._resolve_sphere_transform(
                source=src,
                target=dst,
                density=graph.find_highest_density(space=src),
                hemisphere=hemisphere,
                output_file_path="output.sphere.gii",
            )

            dst_density = graph.find_highest_density(space=dst)

            logger.info(
                "src,dst,density %s: %s -> %s",
                src,
                dst,
                dst_density,
            )
            target_sphere = target_sphere.file_path

            # midthickness files - so we know where (on the sphere)
            # to map transformation values to
            area_surfs = {
                "current-area": graph.fetch_surface_atlas(
                    space=src,
                    density=graph.find_highest_density(space=src),
                    hemisphere=hemisphere,
                    resource_type="midthickness",
                ).file_path,
                "new-area": graph.fetch_surface_atlas(
                    space=dst,
                    density=graph.find_highest_density(space=dst),
                    hemisphere=hemisphere,
                    resource_type="midthickness",
                ).file_path,
            }

            result = workbench.surface_resample(
                surface_in=str(current_sphere),
                current_sphere=str(current_sphere),
                new_sphere=str(target_sphere),
                method="ADAP_BARY_AREA",
                area_surfs=area_surfs,
                surface_out=str(out_file),
            ).surface_out

            output = Path(result)

            arrays = nib.load(output).darrays

            logger.info(
                "OK %s -> %s | arrays=%s",
                src,
                dst,
                [len(a.data) for a in arrays],
            )

            logger.info("Step %s complete: %s", step, output)

    error_file = tmp_path / f"cycle{i}_error.func.gii"

    workbench.signed_distance_to_surface(
        surface_comp=str(output),
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
