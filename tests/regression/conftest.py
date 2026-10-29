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
    """Load the real transformation graph without experimental transforms."""
    runner = request.config.getoption("--runner").lower()
    graph = NeuromapsGraph(runner=runner)

    experimental = {
        (source, target, provider)
        for (source, target), provider in EXPERIMENTAL_XFMS
        for source, target in ((source, target), (target, source))
    }

    graph.remove_edges_from(
        (source, target, key)
        for source, target, key, data in graph.edges(keys=True, data=True)
        if key == graph.surface_to_surface_key
        and all(
            (
                transform.source_space,
                transform.target_space,
                transform.provider,
            )
            in experimental
            for transform in data["data"].surface_transforms
        )
    )

    return graph