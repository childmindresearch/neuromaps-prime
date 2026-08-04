"""Integration tests for real surface transformation workflows.

These tests validate surface transformation cycles using real atlas spaces
and Workbench transformations.

They verify:

- graph traversal resolves real transformation paths,
- multi-hop surface transformations execute,
- transformed metrics preserve expected properties.

Run with::

    pytest tests/integration/test_cycle_integration.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.gifti import GiftiDataArray, GiftiImage

from neuromaps_prime.graph import NeuromapsGraph
from tests.cycle import (
    find_return_paths,
    roundtrip_metric,
    score_roundtrip,
)

HEMISPHERE = "left"
ORIGIN = "D99"


@pytest.fixture(scope="module")
def graph() -> NeuromapsGraph:
    """Load the real transformation graph."""
    return NeuromapsGraph(runner="local")


@pytest.fixture
def real_data_metric(
    graph: NeuromapsGraph,
    tmp_path: Path,
) -> tuple[Path, str]:
    """Create a metric from a real atlas surface."""
    density = graph.find_highest_density(ORIGIN)

    sphere = graph.fetch_surface_atlas(
        ORIGIN,
        density,
        HEMISPHERE,
        "sphere",
    )

    assert sphere is not None
    assert sphere.file_path.exists()

    coords = nib.load(str(sphere.file_path)).darrays[0].data

    metric = coords.sum(axis=1).astype(np.float32)

    output = tmp_path / f"{ORIGIN}_{density}_metric.func.gii"

    nib.save(
        GiftiImage(
            darrays=[
                GiftiDataArray(metric),
            ]
        ),
        output,
    )

    return output, density


def find_executable_cycle(
    graph: NeuromapsGraph,
    metric: Path,
    tmp_path: Path,
) -> tuple[str, ...]:
    """Find a real cycle that can execute all transformations.

    Graph cycles do not necessarily correspond to executable workflows because
    individual edges may lack the resources required for a specific density.
    Try available cycles until one successfully completes all transformation
    hops.
    """
    paths = find_return_paths(
        graph,
        ORIGIN,
        max_length=4,
    )

    for path in paths:
        if len(path) <= 3:
            continue

        try:
            roundtrip_metric(
                graph,
                metric,
                path,
                HEMISPHERE,
                tmp_path,
            )
        except RuntimeError:
            continue

        return path

    raise RuntimeError("No executable surface transformation cycle found.")


def test_real_graph_contains_return_paths(
    graph: NeuromapsGraph,
) -> None:
    """Verify the real graph contains transformation cycles."""
    paths = find_return_paths(
        graph,
        ORIGIN,
        max_length=4,
    )

    assert len(paths) > 0

    for path in paths:
        assert path[0] == ORIGIN
        assert path[-1] == ORIGIN


def test_real_multihop_surface_transforms(
    graph: NeuromapsGraph,
    real_data_metric: tuple[Path, str],
    tmp_path: Path,
) -> None:
    """Verify real multi-hop surface transformations execute."""
    metric, _ = real_data_metric

    path = find_executable_cycle(
        graph,
        metric,
        tmp_path,
    )

    output = roundtrip_metric(
        graph,
        metric,
        path,
        HEMISPHERE,
        tmp_path,
    )

    assert output.exists()


def test_real_transform_preserves_metric_properties(
    graph: NeuromapsGraph,
    real_data_metric: tuple[Path, str],
    tmp_path: Path,
) -> None:
    """Verify round-trip transformations preserve metric structure."""
    metric, _ = real_data_metric

    path = find_executable_cycle(
        graph,
        metric,
        tmp_path,
    )

    roundtrip = roundtrip_metric(
        graph,
        metric,
        path,
        HEMISPHERE,
        tmp_path,
    )

    pearson_r, max_abs_diff = score_roundtrip(
        metric,
        roundtrip,
    )

    assert pearson_r > 0.95
    assert np.isfinite(max_abs_diff)