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
    """Load the real transformation graph with automatic runner selection, without experimental transforms."""
    runner = request.config.getoption("--runner").lower()
    graph = NeuromapsGraph(runner=runner)

    experimental_pairs = {
        (source, target)
        for path, _provider in EXPERIMENTAL_XFMS
        for source, target in (path, path[::-1])
    }

    edges_to_remove = [
        (source, target, key)
        for source, target, key in graph.edges(keys=True)
        if key == graph.surface_to_surface_key
        and (source, target) in experimental_pairs
    ]

    graph.remove_edges_from(edges_to_remove)

    return graph