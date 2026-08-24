"""Utilities for cycle regression tests."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib_surface_plotting import plot_surf
from nibabel.gifti import GiftiDataArray, GiftiImage
from tests.cycle import Hemisphere, RoundtripResult, path_token

from neuromaps_prime.analysis.images import load_data

if TYPE_CHECKING:
    from pathlib import Path

    from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Metric generation
# -------------------------------------------------------------------------


def write_metric(
    metric_file: Path,
    values: np.ndarray,
) -> Path:
    """Write one scalar value per vertex as a GIFTI metric."""
    image = GiftiImage(
        darrays=[
            GiftiDataArray(
                np.asarray(
                    values,
                    dtype=np.float32,
                ),
                intent="NIFTI_INTENT_NONE",
            )
        ]
    )

    metric_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        image,
        metric_file,
    )

    return metric_file


def make_sphere(
    graph: NeuromapsGraph,
    origin: str,
    density: str,
    hemisphere: Hemisphere,
    output_dir: Path,
) -> Path:
    """Create a deterministic synthetic metric from sphere coordinates."""
    sphere = graph.fetch_surface_atlas(
        space=origin,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    )

    if sphere is None:
        raise ValueError(
            f"No sphere atlas for {origin} at {density} ({hemisphere})."
        )

    if not sphere.file_path.exists():
        sphere.fetch()

    data = load_data(sphere.file_path)
    coords = data.array[0]

    values = np.prod(
        coords,
        axis=1,
    )

    metric_file = output_dir / f"metric_{origin}_{density}_{hemisphere}.func.gii"

    return write_metric(
        metric_file,
        values,
    )


# -------------------------------------------------------------------------
# Surface matching for visualization
# -------------------------------------------------------------------------


def find_matching_surface(
    graph: NeuromapsGraph,
    space: str,
    hemisphere: Hemisphere,
    resource_type: str,
    n_vertices: int,
) -> object | None:
    """Find a surface whose vertex count matches the metric."""
    atlases = graph.utils.cache.get_surface_atlases(
        space=space,
        hemisphere=hemisphere,
        resource_type=resource_type,
    )

    for atlas in atlases:
        try:
            if not atlas.file_path.exists():
                atlas.fetch()

            data = load_data(atlas.file_path)
            coords = data.array[0]
        except (
            ValueError,
            FileNotFoundError,
            OSError,
        ) as exc:
            logger.warning(
                "Could not load %s surface for %s (%s): %s",
                resource_type,
                space,
                hemisphere,
                exc,
            )
            continue

        if coords.shape[0] == n_vertices:
            return atlas

    return None


# -------------------------------------------------------------------------
# Surface plotting
# -------------------------------------------------------------------------


def plot_single_surface(
    graph: NeuromapsGraph,
    space: str,
    metric_values: np.ndarray,
    hemisphere: Hemisphere,
    resource_type: str,
    vmin: float,
    vmax: float,
    pearson_r: float,
    hop_index: int,
    output_file: Path,
) -> bool:
    """Plot one metric on one cortical surface."""
    surface_atlas = find_matching_surface(
        graph,
        space,
        hemisphere,
        resource_type,
        metric_values.shape[0],
    )

    if surface_atlas is None:
        logger.warning(
            "No %s surface for %s (%s) matching %d vertices; skipping plot.",
            resource_type,
            space,
            hemisphere,
            metric_values.shape[0],
        )
        return False

    try:
        surface_file = surface_atlas.fetch()

        data = load_data(
            surface_file,
            return_image=True,
        )

        if data.image is None:
            raise ValueError(f"Could not load surface image: {surface_file}")

        coords = data.image.darrays[0].data
        faces = data.image.darrays[1].data

    except (
        ValueError,
        FileNotFoundError,
        OSError,
    ) as exc:
        logger.warning(
            "Could not load %s mesh for %s (%s): %s",
            resource_type,
            space,
            hemisphere,
            exc,
        )
        return False

    try:
        plot_surf(
            coords,
            faces,
            metric_values,
            rotate=[270, 0],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            title=f"{space} | node {hop_index} | r={pearson_r:.5f}",
        )

        fig = plt.gcf()

        fig.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight",
        )

    finally:
        plt.close(fig)

    logger.info(
        "Saved surface plot: %s",
        output_file,
    )

    return True


def plot_cycle_cortical_surfaces(
    graph: NeuromapsGraph,
    path: tuple[str, ...],
    metrics_by_hop: list[tuple[str, np.ndarray]],
    hemisphere: Hemisphere,
    pearson_r: float,
    plot_dir: Path,
) -> None:
    """Plot each cycle node as an individual cortical surface figure."""
    if not metrics_by_hop:
        return

    token = path_token(path)

    for hop_index, (space, metric_values) in enumerate(metrics_by_hop):
        finite = np.isfinite(metric_values)

        if np.any(finite):
            vmin, vmax = np.percentile(
                metric_values[finite],
                [2, 98],
            )
        else:
            vmin, vmax = 0.0, 1.0

        plotted = False

        for resource_type in (
            "midthickness",
            "pial",
            "white",
        ):
            output_file = (
                plot_dir / f"{token}_hop{hop_index}_{space}_{resource_type}.png"
            )

            if plot_single_surface(
                graph=graph,
                space=space,
                metric_values=metric_values,
                hemisphere=hemisphere,
                resource_type=resource_type,
                vmin=vmin,
                vmax=vmax,
                pearson_r=pearson_r,
                hop_index=hop_index,
                output_file=output_file,
            ):
                plotted = True
                break

        if not plotted:
            logger.warning(
                "Could not create any surface plot for %s, hop %d (%s, %s).",
                " -> ".join(path),
                hop_index,
                space,
                hemisphere,
            )


# -------------------------------------------------------------------------
# Transform manifests
# -------------------------------------------------------------------------


def write_transform_manifest(
    roundtrip: RoundtripResult,
    output_file: Path,
) -> None:
    """Write a per-path manifest describing the executed transformations."""
    lines = ["source_space,target_space,area_resource,output_file"]

    lines.extend(
        ",".join(
            [
                hop.source,
                hop.target,
                hop.area_resource,
                str(hop.output_file),
            ]
        )
        for hop in roundtrip.hops
    )

    output_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def load_latest_cycle_baseline(
    baseline_dir: Path,
) -> dict[tuple[str, str], float]:
    """Load the most recent valid cycle baseline CSV."""
    baseline_files = sorted(
        baseline_dir.glob("cycle_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    required_columns = {
        "origin",
        "species",
        "hemisphere",
        "mean_pearson_r",
    }

    for baseline_file in baseline_files:
        try:
            frame = pd.read_csv(baseline_file)
        except (OSError, pd.errors.ParserError):
            continue

        if not required_columns.issubset(frame.columns):
            continue

        baseline: dict[tuple[str, str], float] = {}

        for _, row in frame.iterrows():
            key = (
                str(row["origin"]),
                str(row["hemisphere"]),
            )

            if key in baseline:
                raise ValueError(f"Duplicate baseline entry in {baseline_file}: {key}")

            baseline[key] = float(row["mean_pearson_r"])

        logger.info(
            "Using cycle baseline: %s",
            baseline_file,
        )

        return baseline

    raise FileNotFoundError(f"No valid cycle baseline CSV found in {baseline_dir}.")


def save_cycle_baseline(
    baseline_dir: Path,
    values: dict[tuple[str, str], float],
    graph: NeuromapsGraph,
) -> Path:
    """Save cycle baseline values to a timestamped CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    output_file = baseline_dir / f"cycle_{timestamp}.csv"

    rows = []

    for (origin, hemisphere), mean_pearson_r in sorted(values.items()):
        species = "all"

        if origin != "all":
            species = graph.get_node_data(origin).species

        rows.append(
            {
                "origin": origin,
                "species": species,
                "hemisphere": hemisphere,
                "mean_pearson_r": mean_pearson_r,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "origin",
            "species",
            "hemisphere",
            "mean_pearson_r",
        ],
    )

    frame.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved cycle baseline: %s",
        output_file,
    )

    return output_file
