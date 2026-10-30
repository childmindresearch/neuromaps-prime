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

    experimental_transforms = {
        (source, target, provider)
        for path, provider in EXPERIMENTAL_XFMS
        for source, target in (path, path[::-1])
    }

    edges_to_remove = []

    for source, target, key, edge_data in graph.edges(
        keys=True,
        data=True,
    ):
        if key != graph.surface_to_surface_key:
            continue

        edge = edge_data["data"]
        retained_transforms = []

        for transform in edge.surface_transforms:
            identifier = (
                transform.source_space,
                transform.target_space,
                transform.provider,
            )

            if identifier in experimental_transforms:
                cache_key = (
                    transform.source_space,
                    transform.target_space,
                    transform.density,
                    transform.hemisphere.lower(),
                    transform.resource_type,
                    transform.provider,
                )
                graph._cache.surface_transform.pop(cache_key, None)
            else:
                retained_transforms.append(transform)

        edge.surface_transforms = retained_transforms

        if not retained_transforms:
            edges_to_remove.append((source, target, key))

    for source, target, key in edges_to_remove:
        graph.remove_edge(source, target, key)

    return graph