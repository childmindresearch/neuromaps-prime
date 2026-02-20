"""NetworkX cycle tests to calculate error between spheres."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import List
from unittest.mock import MagicMock, patch

import networkx as nx

import pytest

from neuromaps_prime import models
from neuromaps_prime.graph import NeuromapsGraph

from niwrap import workbench


"""
start with Yerkes19

Call signed distance to surface with two surfaces

Returns output file that has distance for each vertex

Open file, take the absolute value, take the median
= error

For loop over all possibilities

Apply transformations

Apply error at each step
"""

""" 
COMPUTE SIGNED DISTANCE FROM ONE SURFACE TO ANOTHER.
Compute the signed distance function of the reference surface at every vertex on the comparison surface. 
NOTE: this relation is NOT symmetric, the line from a vertex to the closest point on the 'ref' surface 
(the one that defines the signed distance function) will only align with the normal of the 'ref' surface. 
Valid specifiers for winding methods are as follows: EVEN_ODD (default) NEGATIVE NONZERO NORMALS 
"""

@pytest.mark.usefixtures("require_workbench")
def test_surface_cycle(
    graph: NeuromapsGraph,
    tmp_path: Path,
) -> None:
    """Unit tests for surface error."""

    graph = NeuromapsGraph(yaml_file="/Users/tamsin.rogers/Desktop/github/neuromaps-prime/src/neuromaps_prime/datasets/data/neuromaps_graph.yaml")
    assert graph is not None

    """
    workbench.signed_distance_to_surface(
        surface_comp="",
        surface_ref="",
        metric=""
    )


    origin = "Yerkes19"
    hemisphere = "right"
    density = graph.find_highest_density(space=origin)

    # Get underlying directed graph
    G = graph._graph  # or however you expose it

    # All directed simple cycles
    cycles: List[List[str]] = list(nx.recursive_simple_cycles(G))

    # Keep only cycles that start and end at Yerkes
    cycles = [c for c in cycles if origin in c]

    assert cycles, "No cycles found for surface consistency testing"

    for idx, cycle in enumerate(cycles):

        # Rotate cycle so it starts at origin
        while cycle[0] != origin:
            cycle = cycle[1:] + cycle[:1]

        cycle = cycle + [origin]

        current_surface = graph.get_sphere(
            space=origin,
            density=density,
            hemisphere=hemisphere,
        )

        for step, (src, dst) in enumerate(zip(cycle[:-1], cycle[1:])):
            out_file = tmp_path / f"cycle{idx}_step{step}.surf.gii"

            current_surface = graph.surface_to_surface_transformer(
                transformer_type="metric",
                input_file=current_surface,
                source_space=src,
                target_space=dst,
                hemisphere=hemisphere,
                output_file_path=str(out_file),
            )

        # Compare original vs cycled surface
        error_file = tmp_path / f"cycle{idx}_error.func.gii"

        graph.signed_distance_to_surface(
            surface_a=graph.get_sphere(
                space=origin,
                density=density,
                hemisphere=hemisphere,
            ),
            surface_b=current_surface,
            output_file=str(error_file),
        )

        error = graph.compute_median_abs_metric(error_file)

        assert error < 1.0, f"Cycle {cycle} error too large: {error}"
        """