"""Compute annotation transform error."""

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


# SETUP
def _build_graph() -> NeuromapsGraph:
    return NeuromapsGraph(
        runner="local",
        data_dir=Path("/Users/tamsin.rogers/Desktop/github/neuromaps-prime"),
    )


def _load_origin(
    origin: str,
    hemisphere: str,
    graph: NeuromapsGraph,
    tmp_path: Path,
) -> tuple[int, Path, Path, Path]:
    density = graph.find_highest_density(space=origin)

    origin_metric = Path(
        "/Users/tamsin.rogers/Desktop/github/neuromaps-prime/resources/"
        "Yerkes19/annotations/receptor_maps/"
        "src-Yerkes19_den-32k_hemi-L_acq-auto_trc-ampa_desc-RM_annot.func.gii"
    )

    logger.info("Origin metric: %s", origin_metric)
    log_gii_shapes(origin_metric)

    origin_sphere = graph.fetch_surface_atlas(
        space=origin,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    ).file_path

    logger.info("Origin sphere: %s", origin_sphere)

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

    return density, origin_metric, origin_sphere, metric_on_surface


# CYCLE
def _run_cycle(
    i: int,
    cycle: list[str],
    graph: NeuromapsGraph,
    metric_on_surface: Path,
    hemisphere: str,
    tmp_path: Path,
) -> tuple[list[str], Path] | tuple[None, None]:
    # normalize cycle to start at origin
    origin = cycle[0]
    while cycle[0] != origin:
        cycle = cycle[1:] + cycle[:1]
    cycle = [*cycle, origin]

    logger.info("=== Cycle %d: %s ===", i, cycle)

    current_metric = metric_on_surface

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

            target_sphere_transform = graph.fetch_surface_to_surface_transform(
                source=src,
                target=dst,
                density=common_density,
                hemisphere=hemisphere,
                resource_type="sphere",
            )

            if target_sphere_transform is None:
                logger.warning("Missing transform %s -> %s", src, dst)
                return None, None

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
                return None, None

            current_sphere = current_sphere_obj.file_path

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
                return None, None

            area_surfs = {
                "current-area": current_area.file_path,
                "new-area": new_area.file_path,
            }

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
            return None, None

    return cycle, current_metric


# ERROR COMPUTATION
def _compute_error(
    i: int,
    cycle: list[str],
    current_metric: Path,
    metric_on_surface: Path,
    tmp_path: Path,
) -> dict | None:
    error_file = tmp_path / f"cycle{i}_error.func.gii"

    ref_gii = nib.load(metric_on_surface)
    comp_gii = nib.load(current_metric)

    ref = ref_gii.darrays[0].data.astype(float)
    comp = comp_gii.darrays[0].data.astype(float)

    valid_mask = np.isfinite(ref) & np.isfinite(comp)

    if not np.any(valid_mask):
        logger.warning("Cycle %d: no valid vertices", i)
        return None

    error = np.abs(comp[valid_mask] - ref[valid_mask])

    median_error = float(np.median(error)) if error.size > 0 else float("nan")
    mean_error = float(np.mean(error)) if error.size > 0 else float("nan")
    sd_error = float(np.std(error)) if error.size > 0 else float("nan")

    full_error = np.full(ref.shape, np.nan, dtype=float)
    full_error[valid_mask] = error

    np.save(error_file.with_suffix(".npy"), full_error)

    logger.info(
        "Cycle %d COMPLETE: median=%f mean=%f sd=%f",
        i,
        median_error,
        mean_error,
        sd_error,
    )

    if not np.isfinite(median_error):
        logger.warning("Cycle %d produced non-finite median error", i)
        return None

    return {
        "cycle": i,
        "path": " -> ".join(cycle),
        "median_error": median_error,
        "mean_error": mean_error,
        "sd_error": sd_error,
        "n_vertices": int(error.size),
    }


# TEST
def test_annotation_cycle(tmp_path: Path) -> None:
    """Test annotation consistency by cycling through annotation transforms."""
    logging.basicConfig(level=logging.INFO)

    logger.info("=== BUILDING NEUROMAPS GRAPH ===")

    graph = _build_graph()

    origin = "Yerkes19"
    hemisphere = "left"

    _density, _, _, metric_on_surface = _load_origin(
        origin, hemisphere, graph, tmp_path
    )

    directed = graph.to_directed()

    cycles = [c for c in recursive_simple_cycles(directed) if origin in c]
    assert cycles, "No cycles found in graph"

    logger.info("Found %d cycles", len(cycles))

    cycle_errors = []

    for i, cycle in enumerate(cycles):
        result = _run_cycle(i, cycle, graph, metric_on_surface, hemisphere, tmp_path)

        if result is None or result[0] is None:
            logger.warning("Skipping incomplete cycle %d", i)
            continue

        cycle, current_metric = result

        entry = _compute_error(i, cycle, current_metric, metric_on_surface, tmp_path)

        if entry:
            cycle_errors.append(entry)

    assert cycle_errors, "No valid cycles completed"

    df = pd.DataFrame(cycle_errors)

    csv_file = tmp_path / "cycle_error_summary.csv"
    df.to_csv(csv_file, index=False)

    logger.info("Saved cycle summary CSV: %s", csv_file)

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
