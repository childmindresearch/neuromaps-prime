"""Shared fixtures for analysis tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.random import Generator


@pytest.fixture(scope="module")
def rng(seed: int = 12345) -> Generator:
    """Deterministic random number generator."""
    return np.random.default_rng(seed)
