"""Global pytest fixtures, arguments, and options."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence


def pytest_collection_modifyitems(items: Sequence[pytest.Item]) -> None:
    """Apply appropriate markers based on test location."""
    markers = {"unit", "integration", "regression"}

    for item in items:
        test_path = Path(item.fspath)
        for marker in markers & set(test_path.parts):
            item.add_marker(getattr(pytest.mark, marker))
