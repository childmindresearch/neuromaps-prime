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

from itertools import pairwise
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest
from nibabel.gifti import GiftiDataArray, GiftiImage

if TYPE_CHECKING:
    from pathlib import Path

    from neuromaps_prime.graph import NeuromapsGraph

HEMISPHERE = "left"
ORIGIN = "D99"

SINGLE_HOP_PATH = ("D99", "Yerkes19")
MULTI_HOP_PATH = ("D99", "Yerkes19", "CIVETNMT")


@pytest.fixture
def real_d99_metric(
    graph: NeuromapsGraph,
    tmp_path: Path,
) -> Path:
    """Create a real metric from the highest-density D99 surface.

    The metric is constructed by summing the x, y, and z coordinates of each
    D99 surface vertex. This avoids requiring a separate metric file in the
    neuromaps_prime cache while still providing real surface data for
    Workbench to transform.
    """
    density = graph.find_highest_density(ORIGIN)

    sphere = graph.fetch_surface_atlas(
        ORIGIN,
        density,
        HEMISPHERE,
        "sphere",
    )

    assert sphere is not None
    assert sphere.file_path.exists()

    coords = np.asarray(
        nib.load(str(sphere.file_path)).darrays[0].data,
        dtype=np.float32,
    )

    metric = coords.sum(axis=1)

    output = tmp_path / f"{ORIGIN}_{density}_metric.func.gii"

    nib.save(
        GiftiImage(
            darrays=[
                GiftiDataArray(metric),
            ]
        ),
        output,
    )

    return output


def _run_surface_transform(
    graph: NeuromapsGraph,
    metric: Path,
    path: tuple[str, ...],
    tmp_path: Path,
) -> Path:
    """Execute each hop in a real surface transformation path.

    Each hop writes to a separate file. The output from one hop becomes the
    input to the next hop, so a multi-hop test verifies that transformed
    output can actually be consumed by a subsequent Workbench operation.
    """
    current_file = metric

    for source_space, target_space in pairwise(path):
        output_file = tmp_path / f"{source_space}_to_{target_space}.func.gii"

        result = graph.surface_to_surface_transformer(
            transformer_type="metric",
            input_file=current_file,
            source_space=source_space,
            target_space=target_space,
            hemisphere=HEMISPHERE,
            output_file_path=output_file,
            source_density=None,
            target_density=None,
            add_edge=False,
        )

        assert result.path.exists()

        transformed = np.asarray(
            nib.load(str(result.path)).darrays[0].data,
        )

        assert transformed.size > 0
        assert np.all(np.isfinite(transformed))

        current_file = result.path

    return current_file


def test_real_single_hop_surface_transform(
    graph: NeuromapsGraph,
    real_d99_metric: Path,
    tmp_path: Path,
) -> None:
    """Verify Workbench executes a real single-hop surface transformation."""
    output = _run_surface_transform(
        graph=graph,
        metric=real_d99_metric,
        path=SINGLE_HOP_PATH,
        tmp_path=tmp_path,
    )

    assert output.exists()


def test_real_multihop_surface_transform(
    graph: NeuromapsGraph,
    real_d99_metric: Path,
    tmp_path: Path,
) -> None:
    """Verify Workbench executes a real multi-hop surface transformation."""
    output = _run_surface_transform(
        graph=graph,
        metric=real_d99_metric,
        path=MULTI_HOP_PATH,
        tmp_path=tmp_path,
    )

    assert output.exists()
