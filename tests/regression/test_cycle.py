"""Regression benchmark for shortest surface transformation cycles.

This test benchmarks end-to-end surface transformation accuracy on the real
NeuromapsPrime graph using standard template surfaces.

A deterministic vertex-wise metric is generated from the origin template's
midthickness surface, propagated through every shortest executable return path
in the surface transformation graph, and compared with the original metric
after returning to the starting space.

The benchmark is intentionally independent of surface annotations. The metric
is generated directly from the standard midthickness surface so that private
annotation resources do not affect the test.

Only the shortest return paths are evaluated. Every shortest path is retained
when each hop has a compatible surface transform and target surface at the
benchmark density.

The benchmark uses a single density throughout each cycle whenever that
density is available for every hop. This prevents a lower-density transform
from being selected simply because it sorts first in the graph.

Metrics recorded for each transformation cycle include:

- Pearson correlation between original and round-tripped metric.
- Maximum absolute difference between original and round-tripped values.
- Number of transformations in the cycle.

Run with::

    pytest tests/regression/test_cycle.py -v -s
"""

from __future__ import annotations

import logging
from itertools import pairwise
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neuromaps_prime.graph import NeuromapsGraph
from tests.cycle import (
    find_return_paths,
    save_cycle_figure,
    score_roundtrip,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Regression parameters
# -------------------------------------------------------------------------

ORIGIN = "Yerkes19"
HEMISPHERE = "left"


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def graph() -> NeuromapsGraph:
    """Create a Neuromaps graph for regression benchmarking."""
    return NeuromapsGraph()


@pytest.fixture
def output_dir() -> Path:
    """Directory for storing benchmark summaries."""
    directory = Path(__file__).resolve().parent / "output/cycle_outputs"
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    return directory


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _find_executable_surface(
    graph: NeuromapsGraph,
    source: str,
    target: str,
    hemisphere: Literal["left", "right"],
    preferred_density: str | None = None,
) -> tuple[str, str] | None:
    """Find an executable surface transform and its target density.

    Args:
        graph: NeuromapsPrime graph.
        source: Source template space.
        target: Target template space.
        hemisphere: Hemisphere to use.
        preferred_density: Density to prefer when selecting a transform.
            When supplied, a transform at this density is selected if one
            exists with a corresponding target surface.

    Returns:
        ``(density, resource_type)`` for an executable hop, or ``None`` when
        no compatible transform and target surface are available.

    Notes:
        The graph may contain multiple transforms between the same spaces at
        different densities. Density selection must therefore be explicit.

        In particular, both Yerkes19 <-> fsLR have 10k and 32k transforms.
        Sorting densities lexicographically would select 10k first, even when
        the benchmark metric was created on a 32k surface. The preferred
        density avoids that unintended downsampling.
    """
    transforms = graph.search_surface_transforms(
        source_space=source,
        target_space=target,
        hemisphere=hemisphere,
    )

    if not transforms:
        return None

    # Prefer the requested benchmark density.
    if preferred_density is not None:
        preferred = [
            transform
            for transform in transforms
            if transform.density == preferred_density
        ]

        # Only use the preferred-density transform if the corresponding
        # target surface actually exists.
        preferred = sorted(
            preferred,
            key=lambda transform: (
                transform.resource_type != "sphere",
            ),
        )

        for transform in preferred:
            target_surface = graph.fetch_surface_atlas(
                space=target,
                density=preferred_density,
                hemisphere=hemisphere,
                resource_type=transform.resource_type,
            )

            if target_surface is not None:
                return preferred_density, transform.resource_type

    # Fall back to any executable transform. Sphere transforms are preferred
    # because they are the standard resource for Workbench spherical
    # resampling.
    transforms = sorted(
        transforms,
        key=lambda transform: (
            transform.resource_type != "sphere",
            transform.density,
        ),
    )

    for transform in transforms:
        density = transform.density

        target_surface = graph.fetch_surface_atlas(
            space=target,
            density=density,
            hemisphere=hemisphere,
            resource_type=transform.resource_type,
        )

        if target_surface is None:
            continue

        return density, transform.resource_type

    return None


def _make_surface_metric(
    surface_file: Path,
    output_file: Path,
) -> Path:
    """Create a deterministic vertex-wise metric from a midthickness surface."""
    image = nib.load(surface_file)

    vertices = np.asarray(
        image.darrays[0].data,
        dtype=np.float64,
    )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            "Expected surface vertices with shape (n_vertices, 3), "
            f"got {vertices.shape}."
        )

    metric = vertices.sum(axis=1)

    metric_image = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                metric.astype(np.float32),
            )
        ]
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        metric_image,
        output_file,
    )

    return output_file


def _shortest_paths(
    paths: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Return all paths having the minimum number of transformation hops."""
    if not paths:
        return []

    shortest_length = min(len(path) - 1 for path in paths)

    return [
        path
        for path in paths
        if len(path) - 1 == shortest_length
    ]


def _valid_cycle_paths(
    graph: NeuromapsGraph,
    paths: list[tuple[str, ...]],
    density: str,
) -> list[tuple[str, ...]]:
    """Return cycles executable entirely at the requested density."""
    valid_paths: list[tuple[str, ...]] = []

    for path in paths:
        executable = all(
            _find_executable_surface(
                graph,
                source,
                target,
                HEMISPHERE,
                preferred_density=density,
            )
            is not None
            for source, target in pairwise(path)
        )

        if executable:
            valid_paths.append(path)

    return valid_paths


def _resolve_transform_output(
    result: str | Path | None,
    output_name: str,
    workdir: Path,
) -> Path:
    """Resolve a transformation output returned by the graph transformer.

    Workbench runs inside a Docker container with a writable ``/styx_output``
    directory. Therefore the transformer receives a filename rather than an
    absolute host path. Depending on the runner version, the transformer may
    return either the generated path or just the output filename.
    """
    if result is not None:
        result_path = Path(result)

        if result_path.exists():
            return result_path

        workdir_result = workdir / result_path.name

        if workdir_result.exists():
            return workdir_result

    expected = workdir / output_name

    if expected.exists():
        return expected

    raise FileNotFoundError(
        "Surface transformation did not produce an output file. "
        f"Expected '{expected}', got result={result!r}."
    )


def _transform_cycle(
    graph: NeuromapsGraph,
    metric_file: Path,
    path: tuple[str, ...],
    hemisphere: Literal["left", "right"],
    workdir: Path,
    density: str,
) -> Path:
    """Transform a metric through every hop of a cycle.

    Every hop is explicitly performed at ``density``. This ensures that a
    cycle such as Yerkes19 -> fsLR -> Yerkes19 stays at 32k when 32k resources
    are available in both directions.
    """
    current_metric = metric_file
    current_density = density

    for hop, (source, target) in enumerate(pairwise(path)):
        available = _find_executable_surface(
            graph,
            source,
            target,
            hemisphere,
            preferred_density=density,
        )

        if available is None:
            raise RuntimeError(
                f"No executable surface transform for hop '{source}' -> "
                f"'{target}' at density '{density}' on path "
                f"{' -> '.join(path)}"
            )

        transform_density, geometry = available

        if transform_density != density:
            raise RuntimeError(
                f"Unexpected density for hop '{source}' -> '{target}': "
                f"expected '{density}', got '{transform_density}'."
            )

        target_density = density

        logger.info(
            "hop %d: %s -> %s | source_density=%s | target_density=%s | "
            "geometry=%s",
            hop,
            source,
            target,
            current_density,
            target_density,
            geometry,
        )

        output_name = (
            f"hop{hop:02d}_{source}-to-{target}.func.gii"
        )

        result = graph.surface_to_surface_transformer(
            transformer_type="metric",
            input_file=current_metric,
            source_space=source,
            target_space=target,
            hemisphere=hemisphere,
            output_file_path=output_name,
            source_density=current_density,
            target_density=target_density,
        )

        current_metric = _resolve_transform_output(
            result,
            output_name,
            workdir,
        )

        current_density = target_density

    return current_metric


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


def test_surface_transform_cycles(
    graph: NeuromapsGraph,
    output_dir: Path,
    tmp_path: Path,
) -> None:
    """Benchmark shortest executable surface transformation cycles."""
    logging.basicConfig(level=logging.INFO)

    density = graph.find_highest_density(ORIGIN)

    # ------------------------------------------------------------------
    # Create a deterministic metric from the standard midthickness surface.
    # ------------------------------------------------------------------

    surface = graph.fetch_surface_atlas(
        space=ORIGIN,
        density=density,
        hemisphere=HEMISPHERE,
        resource_type="midthickness",
    )

    assert surface is not None, (
        f"Missing midthickness surface for {ORIGIN} "
        f"at {density} ({HEMISPHERE})."
    )

    surface_file = Path(surface.fetch())

    assert surface_file.exists(), (
        f"Missing midthickness surface: {surface_file}"
    )

    metric_file = _make_surface_metric(
        surface_file,
        tmp_path / f"{ORIGIN}_{density}_{HEMISPHERE}_metric.func.gii",
    )

    assert metric_file.exists(), (
        f"Failed to create metric: {metric_file}"
    )

    assert metric_file.stat().st_size > 0, (
        f"Created metric is empty: {metric_file}"
    )

    # ------------------------------------------------------------------
    # Find all return paths and retain every shortest possibility.
    # ------------------------------------------------------------------

    all_paths = find_return_paths(
        graph,
        ORIGIN,
    )

    assert all_paths, (
        f"No return paths found from '{ORIGIN}' "
        "in the surface transformation graph."
    )

    shortest_paths = _shortest_paths(all_paths)

    for path in shortest_paths:
        executable = all(
            _find_executable_surface(
                graph,
                source,
                target,
                HEMISPHERE,
                preferred_density=density,
            )
            is not None
            for source, target in pairwise(path)
        )

        logger.info(
            "Shortest path: %s | executable=%s",
            " -> ".join(path),
            executable,
        )

    valid_paths = _valid_cycle_paths(
        graph,
        shortest_paths,
        density,
    )

    logger.info(
        "Surface cycle benchmark for %s",
        ORIGIN,
    )
    logger.info(
        "Density: %s",
        density,
    )
    logger.info(
        "Total return paths: %d",
        len(all_paths),
    )
    logger.info(
        "Shortest return paths: %d",
        len(shortest_paths),
    )
    logger.info(
        "Executable shortest return paths at %s: %d",
        density,
        len(valid_paths),
    )

    assert valid_paths, (
        f"No executable shortest transformation cycles found from "
        f"'{ORIGIN}' at density '{density}'."
    )

    # ------------------------------------------------------------------
    # Run each shortest cycle.
    # ------------------------------------------------------------------

    results: list[dict[str, object]] = []

    for cycle_number, path in enumerate(valid_paths):
        workdir = tmp_path / f"cycle_{cycle_number:02d}"
        workdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Running cycle %d/%d: %s",
            cycle_number + 1,
            len(valid_paths),
            " -> ".join(path),
        )

        roundtrip = _transform_cycle(
            graph,
            metric_file,
            path,
            HEMISPHERE,
            workdir,
            density,
        )

        pearson_r, max_abs_diff = score_roundtrip(
            metric_file,
            roundtrip,
        )

        result = {
            "path": " -> ".join(path),
            "n_hops": len(path) - 1,
            "pearson_r": pearson_r,
            "max_abs_diff": max_abs_diff,
        }

        results.append(result)

        logger.info(
            "cycle %s: r=%.6f max|delta|=%.3e",
            result["path"],
            pearson_r,
            max_abs_diff,
        )

    # ------------------------------------------------------------------
    # Summarize benchmark results.
    # ------------------------------------------------------------------

    frame = pd.DataFrame(results).sort_values(
        "pearson_r",
        ascending=False,
    )

    logger.info(
        "\n=== Surface Cycle Benchmark: %s (%s, %s) ===\n%s",
        ORIGIN,
        density,
        HEMISPHERE,
        frame.to_string(index=False),
    )

    output_file = (
        output_dir
        / f"cycle_{ORIGIN}_{density}_{HEMISPHERE}.csv"
    )

    frame.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved cycle benchmark results: %s",
        output_file,
    )

    figure_file = (
        output_dir
        / f"cycle_{ORIGIN}_{density}_{HEMISPHERE}.png"
    )

    save_cycle_figure(
        results,
        figure_file,
        title=(
            f"Surface Cycle Benchmark: "
            f"{ORIGIN} ({density}, {HEMISPHERE})"
        ),
    )

    logger.info(
        "Saved cycle benchmark figure: %s",
        figure_file,
    )

    # ------------------------------------------------------------------
    # Basic sanity checks.
    # ------------------------------------------------------------------

    assert np.isfinite(frame["pearson_r"]).all(), (
        "Benchmark produced non-finite Pearson correlations."
    )

    assert np.isfinite(frame["max_abs_diff"]).all(), (
        "Benchmark produced non-finite maximum absolute differences."
    )