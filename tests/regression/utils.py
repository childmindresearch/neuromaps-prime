"""Utilities shared by the regression suites.

Provides the synthetic seed metric used by the cycle regression tests: a
deterministic vertex-wise value derived from an origin space's sphere atlas,
so every run starts from a comparable, fully reproducible value set without
hand-authored data — and the space-selection helper shared by the pairwise
regression tests (surface-matrix, distance-map).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tests.cycle import Hemisphere, write_metric

from neuromaps_prime.analysis.images import load_data

if TYPE_CHECKING:
    from pathlib import Path

    from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)


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

    data = load_data(sphere.fetch())
    coords = data.array[0]

    values = np.prod(coords, axis=1)

    metric_file = output_dir / f"metric_{origin}_{density}_{hemisphere}.func.gii"

    return write_metric(metric_file, values)


def get_valid_spaces(
    graph: NeuromapsGraph, hemisphere: str, *, surface_type: str = "midthickness"
) -> list[str]:
    """Return the graph nodes that expose both a sphere and a surface.

    A space is usable only if it provides both a sphere and ``surface_type``
    at its highest available density for the hemisphere. Nodes that fail to
    probe are skipped, not fatal.
    """
    valid = []

    for node in graph.nodes:
        try:
            density = graph.find_highest_density(node)

            sphere = graph.fetch_surface_atlas(
                space=node,
                density=density,
                hemisphere=hemisphere,
                resource_type="sphere",
            )

            surface = graph.fetch_surface_atlas(
                space=node,
                density=density,
                hemisphere=hemisphere,
                resource_type=surface_type,
            )

            if sphere is not None and surface is not None:
                valid.append(node)

        except Exception as exc:  # A node that fails to probe is skipped, not fatal.
            logger.debug("Skipping node %s due to error: %s", node, exc)

    return valid
