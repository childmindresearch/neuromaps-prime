"""Fixtures shared across integration tests."""

from __future__ import annotations

import pytest

from neuromaps_prime.graph import NeuromapsGraph


@pytest.fixture(scope="module")
def graph(request: pytest.FixtureRequest) -> NeuromapsGraph:
    """Load the real transformation graph with automatic runner selection."""
    runner = request.config.getoption("--runner").lower()
    return NeuromapsGraph(runner=runner)
