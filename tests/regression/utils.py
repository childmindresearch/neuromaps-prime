"""Utilities for cycle regression tests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib_surface_plotting import plot_surf
from tests.cycle import Hemisphere, path_token, write_metric

from neuromaps_prime.analysis.images import load_data

if TYPE_CHECKING:
    from pathlib import Path

    from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Metric generation
# -------------------------------------------------------------------------


def make_sphere(
    graph: NeuromapsGraph,
    origin: str,
    density: str,
    hemisphere: Hemisphere,
    output_dir: Path,
) -> Path:
    """Create a deterministic synthetic metric from sphere coordinates."""
    sphere = graph.fetch_surface_atlas(
        space=origin, density=density, hemisphere=hemisphere, resource_type="sphere"
    )

    if sphere is None:
        raise ValueError(f"No sphere atlas for {origin} at {density} ({hemisphere}).")

    if not sphere.file_path.exists():
        sphere.fetch()

    data = load_data(sphere.file_path)
    coords = data.array[0]

    values = np.prod(coords, axis=1)

    metric_file = output_dir / f"metric_{origin}_{density}_{hemisphere}.func.gii"

    return write_metric(metric_file, values)


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
        space=space, hemisphere=hemisphere, resource_type=resource_type
    )

    for atlas in atlases:
        try:
            if not atlas.file_path.exists():
                atlas.fetch()

            data = load_data(atlas.file_path)
            coords = data.array[0]
        except (ValueError, FileNotFoundError, OSError) as exc:
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
    source: str,
    target: str,
    cycle_origin: str,
    cycle_target: str,
    output_file: Path,
) -> bool:
    """Plot one metric on one cortical surface."""
    surface_atlas = find_matching_surface(
        graph, space, hemisphere, resource_type, metric_values.shape[0]
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

        data = load_data(surface_file, return_image=True)

        if data.image is None:
            raise ValueError(f"Could not load surface image: {surface_file}")

        coords = data.image.darrays[0].data
        faces = data.image.darrays[1].data

    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.warning(
            "Could not load %s mesh for %s (%s): %s",
            resource_type,
            space,
            hemisphere,
            exc,
        )
        return False

    if hop_index == 0:
        title = (
            f"{space} | "
            f"Hop 0 — origin | "
            f"Cycle: {cycle_origin} → {cycle_target} | "
            f"{resource_type} | "
            f"r={pearson_r:.5f}"
        )
    else:
        title = (
            f"{source} → {target} | "
            f"Hop {hop_index} | "
            f"Cycle: {cycle_origin} → {cycle_target} | "
            f"{resource_type} | "
            f"r={pearson_r:.5f}"
        )

    try:
        plot_surf(
            coords,
            faces,
            metric_values,
            rotate=[270, 0],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            title=title,
        )

        fig = plt.gcf()

        fig.savefig(output_file, dpi=200, bbox_inches="tight")

    finally:
        plt.close(fig)

    logger.info("Saved surface plot: %s", output_file)

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

    # The first and final spaces define the complete cycle.
    cycle_origin = path[0]
    cycle_target = path[-1]

    for hop_index, (space, metric_values) in enumerate(metrics_by_hop):
        finite = np.isfinite(metric_values)

        if np.any(finite):
            vmin, vmax = np.percentile(metric_values[finite], [2, 98])
        else:
            vmin, vmax = 0.0, 1.0

        plotted = False

        if hop_index == 0:
            source = cycle_origin
            target = cycle_origin
        else:
            source = path[hop_index - 1]
            target = path[hop_index]

        for resource_type in ("midthickness", "pial", "white"):
            output_file = plot_dir / f"{token}_hop-{hop_index}_{resource_type}.png"

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
                source=source,
                target=target,
                cycle_origin=cycle_origin,
                cycle_target=cycle_target,
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
