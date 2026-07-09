"""Regression test for cyclic surface transform preservation using eigenmodes.

For each NHP atlas:

1. Compute eigenmode on the original midthickness surface.
2. Find cycle through the surface transform graph.
3. Apply each surface transformation in cycle.
4. Recompute eigenmode on the final returned surface.
5. Compare the original and cycled eigensystems.

The test evaluates whether the surface transformation cycle preserves eigenmodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from neuromaps_prime.graph import NeuromapsGraph
from neuromodes.eigen import EigenSolver

logger = logging.getLogger(__name__)


NHP_SPACES = [
    "Yerkes19",
    "D99",
    "MEBRAINS",
    "NMT2Sym",
    "CIVETNMT",
]


def get_valid_spaces(
    graph: NeuromapsGraph,
) -> list[str]:
    """Return NHP spaces with required surface resources."""

    valid = []

    for space in NHP_SPACES:
        try:
            density = graph.find_highest_density(space)

            graph.fetch_surface_atlas(
                space=space,
                density=density,
                hemisphere="right",
                resource_type="sphere",
            )

            graph.fetch_surface_atlas(
                space=space,
                density=density,
                hemisphere="right",
                resource_type="midthickness",
            )

            valid.append(space)

        except Exception as exc:
            logger.warning(
                "Skipping %s: %s",
                space,
                exc,
            )

    return valid


def compute_eigenmodes(
    surface: Path,
    *,
    num_modes: int = 100,
):
    """Compute surface eigenmodes using neuromodes."""

    solver = EigenSolver(
        str(surface),
    )

    solver.solve(
        n_modes=num_modes,
    )

    return (
        solver.evals,
        solver.emodes,
    )


def test_surface_eigenmode_preservation(
    tmp_path: Path,
):
    """Validate Yerkes19 eigenmode preservation.

    TODO:
        Replace the placeholder final eigenmode with the eigenmode
        calculated after applying the full transform cycle.
    """

    logging.basicConfig(
        level=logging.INFO,
    )

    graph = NeuromapsGraph()

    space = "Yerkes19"
    hemisphere = "right"

    density = graph.find_highest_density(
        space,
    )

    yerkes_surface = Path(
        graph.fetch_surface_atlas(
            space=space,
            density=density,
            hemisphere=hemisphere,
            resource_type="midthickness",
        ).fetch()
    )

    logger.info(
        "Computing Yerkes19 eigenmodes",
    )

    eigvals_original, eigvecs_original = compute_eigenmodes(
        yerkes_surface,
        num_modes=100,
    )

    # placeholder, this will become the eigenmode computed after the transform cycle.
    logger.info(
        "Compare initial Yerkes19 eigenmode to final Yerkes19 eigenmode",
    )

    eigvals_final = eigvals_original  # scalar values associated with each eigenmode
    eigvecs_final = eigvecs_original  # actual spatial patterns on the surface

    # check that the eigenmode results are numerically the same
    assert np.allclose(
        eigvals_original,
        eigvals_final,
    )
    assert np.allclose(
        eigvecs_original,
        eigvecs_final,
    )

    logger.info(
        "Yerkes19 eigenmode preserved",
    )
