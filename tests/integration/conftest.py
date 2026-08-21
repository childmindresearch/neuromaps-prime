"""Fixtures shared across integration tests."""

from __future__ import annotations

import pytest

from neuromaps_prime.graph import NeuromapsGraph


@pytest.fixture(scope="module")
def graph() -> NeuromapsGraph:
    """Load the real transformation graph with automatic runner selection."""
    return NeuromapsGraph(runner="local")
