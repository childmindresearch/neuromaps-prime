"""Fixtures shared by the real-data regression suites.

The regression suites all drive the same production transformation graph on
real data, so the graph is built once per test class via a shared class-scoped
fixture rather than redeclared in each test class.
"""

from __future__ import annotations

import pytest

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.graph.transforms.surface import EXPERIMENTAL_XFMS


def _is_experimental(transform: object) -> bool:
    """Return whether a surface transform is marked experimental."""
    source = transform.source_space
    target = transform.target_space
    provider = transform.provider

    return any(
        source == pair[0]
        and target == pair[1]
        and (experimental_provider is None or provider == experimental_provider)
        for pair, experimental_provider in EXPERIMENTAL_XFMS
    )


@pytest.fixture(scope="class")
def graph(request: pytest.FixtureRequest) -> NeuromapsGraph:
    """Load the graph without experimental transforms for regression tests."""
    runner = request.config.getoption("--runner").lower()
    graph = NeuromapsGraph(runner=runner)

    # Remove experimental transforms from the surface-transform cache.
    for cache_key, transform in list(graph._cache.surface_transform.items()):
        if _is_experimental(transform):
            del graph._cache.surface_transform[cache_key]

    # Remove experimental transforms from graph edges, preserving alternatives.
    for source, target, key, data in list(graph.edges(keys=True, data=True)):
        if key != graph.surface_to_surface_key:
            continue

        edge = data["data"]
        retained = [
            transform
            for transform in edge.surface_transforms
            if not _is_experimental(transform)
        ]

        if len(retained) == len(edge.surface_transforms):
            continue

        if retained:
            edge.surface_transforms = retained
        else:
            graph.remove_edge(source, target, key=key)

    return graph

