"""Fixtures shared across integration tests."""

from __future__ import annotations

import pytest
import requests

from neuromaps_prime.fetcher import _RETRYABLE_STATUS
from neuromaps_prime.graph import NeuromapsGraph

_TIMEOUT = 90


@pytest.fixture(scope="module")
def graph(request: pytest.FixtureRequest) -> NeuromapsGraph:
    """Load the real transformation graph with automatic runner selection."""
    runner = request.config.getoption("--runner").lower()
    return NeuromapsGraph(runner=runner)


@pytest.fixture(scope="class")
def osf_api_status() -> None:
    """Check OSF API status."""
    try:
        response = requests.get("https://api.osf.io", timeout=_TIMEOUT)
        if response.status_code in _RETRYABLE_STATUS:
            pytest.skip("OSF API currently unreachable (retryable status code)")
    except (requests.ConnectionError, requests.Timeout):
        pytest.skip("OSF API is down, unreachable, or timed out")


@pytest.fixture(scope="class")
def gh_api_status() -> None:
    """Check GitHub API status."""
    try:
        response = requests.get("https://api.github.com", timeout=_TIMEOUT)
        if response.status_code in _RETRYABLE_STATUS:
            pytest.skip("GitHub API currently unreachable (retryable status code)")
    except (requests.ConnectionError, requests.Timeout):
        pytest.skip("GitHub API is down, unreachable, or timed out")
