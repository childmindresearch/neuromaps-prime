"""Compute surface-to-surface transform error."""

import logging
from itertools import pairwise
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from networkx.algorithms.cycles import recursive_simple_cycles
from niwrap import workbench

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.transforms.utils import log_gii_shapes

logger = logging.getLogger(__name__)


def test_surface_cycle(tmp_path: Path) -> None:
    """Test surface consistency by cycling through surface transforms."""
    logging.basicConfig(level=logging.INFO)

    logger.info("=== BUILDING NEUROMAPS GRAPH ===")

    graph = NeuromapsGraph(
        runner="local",
        data_dir=Path("/Users/tamsin.rogers/Desktop/github/neuromaps-prime"),
    )

    origin = "Yerkes19"
    hemisphere = "left"

    density = graph.find_highest_density(space=origin)

    # LOAD ORIGIN METRIC
    origin_metric = Path(
        "/Users/tamsin.rogers/Desktop/github/neuromaps-prime/resources/"
        "Yerkes19/annotations/receptor_maps/"
        "src-Yerkes19_den-32k_hemi-L_acq-auto_trc-ampa_desc-RM_annot.func.gii"
    )

    logger.info("Origin metric: %s", origin_metric)
    log_gii_shapes(origin_metric)

    # GET ORIGIN SPHERE
    origin_sphere = graph.fetch_surface_atlas(
        space=origin,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    ).file_path

    logger.info("Origin sphere: %s", origin_sphere)

    # PROJECT METRIC → SPHERE SPACE (START OF CYCLING STATE)
    metric_on_surface = workbench.metric_resample(
        metric_in=str(origin_metric),
        current_sphere=str(origin_sphere),
        new_sphere=str(origin_sphere),
        method="ADAP_BARY_AREA",
        area_surfs={
            "current-area": graph.fetch_surface_atlas(
                space=origin,
                density=density,
                hemisphere=hemisphere,
                resource_type="midthickness",
            ).file_path,
            "new-area": graph.fetch_surface_atlas(
                space=origin,
                density=density,
                hemisphere=hemisphere,
                resource_type="midthickness",
            ).file_path,
        },
        metric_out=str(tmp_path / "metric_on_origin_sphere.func.gii"),
    ).metric_out

    metric_on_surface = Path(metric_on_surface)

    log_gii_shapes(metric_on_surface)

    # BUILD GRAPH CYCLES
    directed = graph.to_directed()

    cycles = [c for c in recursive_simple_cycles(directed) if origin in c]

    assert cycles, "No cycles found in graph"

    logger.info("Found %d cycles", len(cycles))

    # store per-cycle summary stats
    cycle_errors = []

    # RUN CYCLIC TRANSFORMS
    for i, cycle in enumerate(cycles):
        # normalize cycle to start at origin
        while cycle[0] != origin:
            cycle = cycle[1:] + cycle[:1]

        cycle = [*cycle, origin]

        logger.info("=== Cycle %d: %s ===", i, cycle)

        current_metric = metric_on_surface
        cycle_completed = True

        # ----------------------------------------------------------
        # RUN ENTIRE CYCLE
        # ----------------------------------------------------------
        for step, (src, dst) in enumerate(pairwise(cycle)):
            logger.info("Step %d: %s -> %s", step, src, dst)

            out_file = tmp_path / f"cycle{i}_step{step}_{src}_to_{dst}.func.gii"

            try:
                common_density = graph.find_common_density(src, dst)

                logger.info(
                    "Using common density %s for %s -> %s",
                    common_density,
                    src,
                    dst,
                )

                # GET SPHERE TRANSFORM
                target_sphere_transform = graph.fetch_surface_to_surface_transform(
                    source=src,
                    target=dst,
                    density=common_density,
                    hemisphere=hemisphere,
                    resource_type="sphere",
                )

                if target_sphere_transform is None:
                    logger.warning(
                        "Missing transform %s -> %s",
                        src,
                        dst,
                    )
                    cycle_completed = False
                    break

                target_sphere = target_sphere_transform.file_path

                current_sphere_obj = graph.fetch_surface_atlas(
                    space=src,
                    density=common_density,
                    hemisphere=hemisphere,
                    resource_type="sphere",
                )

                if current_sphere_obj is None:
                    logger.warning(
                        "Missing source sphere for %s (%s)",
                        src,
                        common_density,
                    )
                    cycle_completed = False
                    break

                current_sphere = current_sphere_obj.file_path

                # AREA CORRECTION SURFACES
                current_area = graph.fetch_surface_atlas(
                    space=src,
                    density=common_density,
                    hemisphere=hemisphere,
                    resource_type="midthickness",
                )

                new_area = graph.fetch_surface_atlas(
                    space=dst,
                    density=common_density,
                    hemisphere=hemisphere,
                    resource_type="midthickness",
                )

                if current_area is None or new_area is None:
                    logger.warning(
                        "Skipping cycle %d step %d (%s -> %s): "
                        "missing midthickness surface at density %s",
                        i,
                        step,
                        src,
                        dst,
                        common_density,
                    )
                    cycle_completed = False
                    break

                area_surfs = {
                    "current-area": current_area.file_path,
                    "new-area": new_area.file_path,
                }

                # RESAMPLE METRIC THROUGH SPHERE TRANSFORM
                current_metric = workbench.metric_resample(
                    metric_in=str(current_metric),
                    current_sphere=str(current_sphere),
                    new_sphere=str(target_sphere),
                    method="ADAP_BARY_AREA",
                    area_surfs=area_surfs,
                    metric_out=str(out_file),
                ).metric_out

                current_metric = Path(current_metric)

                log_gii_shapes(current_metric)

            except Exception as e:
                logger.warning(
                    "Skipping cycle %d step %d (%s -> %s): %s",
                    i,
                    step,
                    src,
                    dst,
                    e,
                )
                cycle_completed = False
                break

        # ----------------------------------------------------------
        # SKIP INCOMPLETE CYCLES
        # ----------------------------------------------------------
        if not cycle_completed:
            logger.warning("Skipping incomplete cycle %d", i)
            continue

        # ----------------------------------------------------------
        # CYCLE CLOSURE ERROR
        # ONLY AFTER ENTIRE CYCLE COMPLETES
        # ----------------------------------------------------------
        error_file = tmp_path / f"cycle{i}_error.func.gii"

        ref_gii = nib.load(metric_on_surface)
        comp_gii = nib.load(current_metric)

        ref = ref_gii.darrays[0].data.astype(float)
        comp = comp_gii.darrays[0].data.astype(float)

        valid_mask = np.isfinite(ref) & np.isfinite(comp)

        if not np.any(valid_mask):
            logger.warning(
                "Cycle %d: no valid vertices",
                i,
            )
            continue

        error = np.abs(comp[valid_mask] - ref[valid_mask])

        median_error = float(np.median(error)) if error.size > 0 else float("nan")

        mean_error = float(np.mean(error)) if error.size > 0 else float("nan")

        sd_error = float(np.std(error)) if error.size > 0 else float("nan")

        # save vertex-wise error map
        full_error = np.full(ref.shape, np.nan, dtype=float)
        full_error[valid_mask] = error

        np.save(
            error_file.with_suffix(".npy"),
            full_error,
        )

        logger.info(
            "Cycle %d COMPLETE: median=%f mean=%f sd=%f",
            i,
            median_error,
            mean_error,
            sd_error,
        )

        if not np.isfinite(median_error):
            logger.warning(
                "Cycle %d produced non-finite median error",
                i,
            )
            continue

        # store summary stats
        cycle_errors.append(
            {
                "cycle": i,
                "path": " -> ".join(cycle),
                "median_error": median_error,
                "mean_error": mean_error,
                "sd_error": sd_error,
                "n_vertices": int(error.size),
            }
        )

    # ------------------------------------------------------------------
    # AFTER ALL CYCLES COMPLETE
    # ------------------------------------------------------------------

    assert cycle_errors, "No valid cycles completed"

    df = pd.DataFrame(cycle_errors)

    csv_file = tmp_path / "cycle_error_summary.csv"

    df.to_csv(csv_file, index=False)

    logger.info(
        "Saved cycle summary CSV: %s",
        csv_file,
    )

    overall = {
        "median_of_medians": float(df["median_error"].median()),
        "mean_of_means": float(df["mean_error"].mean()),
        "sd_of_means": float(df["mean_error"].std()),
        "mean_sd": float(df["sd_error"].mean()),
        "n_cycles": len(df),
    }

    logger.info("=== OVERALL CYCLE ERROR SUMMARY ===")

    for key, value in overall.items():
        logger.info("%s: %s", key, value)

    logger.info("=== PER-CYCLE ERRORS ===\n%s", df)

    logger.info("=== OVERALL SUMMARY ===")

    for key, value in overall.items():
        logger.info("%s: %s", key, value)
