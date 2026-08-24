"""Integration tests for real surface transformations.

These tests verify that real Workbench surface transformations can execute
through the NeuromapsPrime transformation graph.

The tests cover:

- a real single-hop surface transformation,
- a real multi-hop surface transformation,
- propagation of a transformed metric from one hop into the next.

The tests intentionally do not evaluate round-trip accuracy or surface
transformation cycles. Those behaviors are tested separately.

Run with::

    pytest tests/integration/test_surface_transform.py -v -s
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest
from nibabel.gifti import GiftiDataArray, GiftiImage

from neuromaps_prime.analysis.images import load_data

if TYPE_CHECKING:
    from pathlib import Path

    from neuromaps_prime.graph import NeuromapsGraph

HEMISPHERE = "left"
ORIGIN = "D99"


@pytest.fixture
def surface_metric(graph: NeuromapsGraph, tmp_path: Path) -> Path:
    """Create a metric from the highest-density surface available for the given space.

    The metric is constructed by summing the x, y, and z coordinates of each
    given surface vertex. This avoids requiring a separate metric file in the
    neuromaps_prime cache while still providing surface data for Workbench to
    transform.
    """
    density = graph.find_highest_density(ORIGIN)

    sphere = graph.fetch_surface_atlas(
        ORIGIN,
        density,
        HEMISPHERE,
        "sphere",
    )

    assert sphere is not None

    data, _ = load_data(sphere.file_path)
    coords = np.asarray(data[0], dtype=np.float64)

    metric = coords.sum(axis=1, dtype=np.float32)

    output = tmp_path / f"{ORIGIN}_{density}_metric.func.gii"

    nib.save(GiftiImage(darrays=[GiftiDataArray(metric)]), output)

    return output


def test_single_hop_surface_transform(
    graph: NeuromapsGraph,
    surface_metric: Path,
    tmp_path: Path,
) -> None:
    """Verify Workbench executes a real single-hop surface transformation."""
    target = "Yerkes19"
    output = tmp_path / f"{ORIGIN}_to_{target}.func.gii"

    result = graph.surface_to_surface_transformer(
        transformer_type="metric",
        input_file=surface_metric,
        source_space=ORIGIN,
        target_space=target,
        hemisphere=HEMISPHERE,
        output_file_path=output,
        source_density=None,
        target_density=None,
        add_edge=False,
    )

    assert result.path.exists()

    transformed = load_data(result.path).array
    assert transformed.size > 0
    assert np.all(np.isfinite(transformed))


def test_multihop_surface_transform(
    graph: NeuromapsGraph,
    surface_metric: Path,
    tmp_path: Path,
) -> None:
    """Verify Workbench executes a real multi-hop surface transformation."""
    target = "fsLR"
    output = tmp_path / f"{ORIGIN}_to_{target}.func.gii"

    result = graph.surface_to_surface_transformer(
        transformer_type="metric",
        input_file=surface_metric,
        source_space=ORIGIN,
        target_space=target,
        hemisphere=HEMISPHERE,
        output_file_path=output,
        source_density=None,
        target_density=None,
        add_edge=False,
    )

    assert result.path.exists()

    transformed = load_data(result.path).array
    assert transformed.size > 0
    assert np.all(np.isfinite(transformed))
