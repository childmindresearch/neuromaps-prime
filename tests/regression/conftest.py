"""Fixtures shared by the real-data regression suites.

The regression suites all drive the same production transformation graph on
real data, so the graph is built once per test class via a shared class-scoped
fixture rather than redeclared in each test class.
"""

from __future__ import annotations

import pytest

from neuromaps_prime.graph import NeuromapsGraph


@pytest.fixture(scope="class")
def graph(request: pytest.FixtureRequest) -> NeuromapsGraph:
    """Load the real transformation graph with automatic runner selection."""
    runner = request.config.getoption("--runner").lower()
    return NeuromapsGraph(runner=runner)
