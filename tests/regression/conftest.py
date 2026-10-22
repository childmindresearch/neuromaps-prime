"""Fixtures shared by the real-data regression suites.

The regression suites all drive the same production transformation graph on
real data, so the graph is built once per test class via a shared class-scoped
fixture rather than redeclared in each test class.
"""

from __future__ import annotations

import pytest

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.graph.transforms.surface import EXPERIMENTAL_XFMS


@pytest.fixture(scope="class")
def graph(request: pytest.FixtureRequest) -> NeuromapsGraph:
    """Load the production graph, excluding experimental transforms from regression paths."""
    runner = request.config.getoption("--runner").lower()
    graph = NeuromapsGraph(runner=runner)

    for source, target, key, data in list(graph.edges(keys=True, data=True)):
        if key != graph.surface_to_surface_key:
            continue

        edge = data["data"]
        transforms = edge.surface_transforms
        retained = [
            transform
            for transform in transforms
            if not any(
                transform.source_space == pair[0]
                and transform.target_space == pair[1]
                and (provider is None or transform.provider == provider)
                for pair, provider in EXPERIMENTAL_XFMS
            )
        ]

        if len(retained) == len(transforms):
            continue

        if retained:
            edge.surface_transforms = retained
        else:
            graph.remove_edge(source, target, key=key)

    return graph
