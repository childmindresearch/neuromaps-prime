"""Regression benchmark for surface transformation round trips.

This test benchmarks end-to-end surface transformation accuracy on the real
NeuromapsPrime graph.

A deterministic vertex-wise metric is generated from the origin template's
midthickness surface, propagated through every available round-trip route, and
compared with the original metric after returning to the starting space.

Unlike a shortest-path benchmark, this test evaluates all logical return
cycles represented in the graph. A logical transformation edge may itself
require multiple executable surface transformations.

For example, a logical route such as::

    Yerkes19 -> fsaverage -> Yerkes19

may be executed as::

    Yerkes19 -> fsLR -> fsaverage -> fsLR -> Yerkes19

when the direct Yerkes19 <-> fsaverage resources are unavailable but the
intermediate fsLR resources make the transformation executable.

The benchmark is intentionally independent of surface annotations. The
metric is generated directly from the standard midthickness surface so that
private annotation resources do not affect the test.

Metrics recorded for each round trip include:

- Logical transformation path.
- Actual executable transformation path.
- Number of executable transformations.
- Number of logical transformations.
- Pearson correlation between original and round-tripped metric.
- Maximum absolute difference between original and round-tripped values.

Run with::

    pytest tests/regression/test_cycle.py -v -s
"""

from __future__ import annotations

import logging
from itertools import pairwise
from pathlib import Path
from typing import Literal

import nibabel as nib
import networkx as nx
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
DENSITY = "32k"


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
# Surface resource helpers
# -------------------------------------------------------------------------


def _has_surface_transform(
    graph: NeuromapsGraph,
    source: str,
    target: str,
    hemisphere: Literal["left", "right"],
    density: str,
) -> bool:
    """Return whether a surface transform is executable at ``density``."""
    transforms = graph.search_surface_transforms(
        source_space=source,
        target_space=target,
        hemisphere=hemisphere,
        density=density,
    )

    if not transforms:
        return False

    for transform in sorted(
        transforms,
        key=lambda transform: (
            transform.resource_type != "sphere",
        ),
    ):
        target_surface = graph.fetch_surface_atlas(
            space=target,
            density=density,
            hemisphere=hemisphere,
            resource_type=transform.resource_type,
        )

        if target_surface is not None:
            return True

    return False


def _find_executable_surface(
    graph: NeuromapsGraph,
    source: str,
    target: str,
    hemisphere: Literal["left", "right"],
    density: str,
) -> tuple[str, str] | None:
    """Find an executable surface transform at the requested density.

    Sphere transforms are preferred because they are the standard resources
    for Workbench spherical resampling.

    Returns:
        ``(density, resource_type)`` if an executable transform exists,
        otherwise ``None``.
    """
    transforms = graph.search_surface_transforms(
        source_space=source,
        target_space=target,
        hemisphere=hemisphere,
        density=density,
    )

    transforms = sorted(
        transforms,
        key=lambda transform: (
            transform.resource_type != "sphere",
        ),
    )

    for transform in transforms:
        target_surface = graph.fetch_surface_atlas(
            space=target,
            density=density,
            hemisphere=hemisphere,
            resource_type=transform.resource_type,
        )

        if target_surface is not None:
            return density, transform.resource_type

    return None


# -------------------------------------------------------------------------
# Executable route resolution
# -------------------------------------------------------------------------


def _surface_graph(
    graph: NeuromapsGraph,
    hemisphere: Literal["left", "right"],
    density: str,
) -> nx.DiGraph:
    """Build a graph containing executable surface transformations.

    Every directed edge represents a surface transformation that can be
    executed at the benchmark density.
    """
    surface_graph = nx.DiGraph()

    for source, target, key in graph.edges(keys=True):
        if key != graph.surface_to_surface_key:
            continue

        if _has_surface_transform(
            graph,
            source,
            target,
            hemisphere,
            density,
        ):
            surface_graph.add_edge(
                source,
                target,
            )

    return surface_graph


def _find_executable_route(
    graph: NeuromapsGraph,
    source: str,
    target: str,
    hemisphere: Literal["left", "right"],
    density: str,
    forbidden: set[str] | None = None,
) -> tuple[str, ...] | None:
    """Find an executable surface route between two spaces.

    The route may contain intermediate template spaces.

    Args:
        graph: NeuromapsPrime graph.
        source: Starting space.
        target: Destination space.
        hemisphere: Hemisphere to use.
        density: Surface density required for every hop.
        forbidden: Spaces that should not be traversed.

    Returns:
        Tuple containing the executable route, including source and target,
        or ``None`` when no route exists.
    """
    surface_graph = _surface_graph(
        graph,
        hemisphere,
        density,
    )

    if source not in surface_graph or target not in surface_graph:
        return None

    if forbidden:
        surface_graph = surface_graph.copy()
        surface_graph.remove_nodes_from(
            node
            for node in forbidden
            if node not in {source, target}
        )

    try:
        return tuple(
            nx.shortest_path(
                surface_graph,
                source=source,
                target=target,
            )
        )
    except nx.NetworkXNoPath:
        return None


def _expand_logical_path(
    graph: NeuromapsGraph,
    logical_path: tuple[str, ...],
    hemisphere: Literal["left", "right"],
    density: str,
) -> tuple[str, ...] | None:
    """Expand a logical cycle into executable surface transformations.

    Each logical edge is resolved independently. Intermediate spaces are
    allowed, but the resolver avoids using spaces that have already been
    traversed unnecessarily.

    Example::

        logical:
            Yerkes19 -> fsaverage -> Yerkes19

        executable:
            Yerkes19 -> fsLR -> fsaverage -> fsLR -> Yerkes19
    """
    executable_path: list[str] = [logical_path[0]]
    used_spaces: set[str] = {logical_path[0]}

    for source, target in pairwise(logical_path):
        current_source = executable_path[-1]

        route = _find_executable_route(
            graph,
            current_source,
            target,
            hemisphere,
            density,
            forbidden=used_spaces - {current_source, target},
        )

        if route is None:
            return None

        executable_path.extend(route[1:])
        used_spaces.update(route)

    return tuple(executable_path)


# -------------------------------------------------------------------------
# Metric generation
# -------------------------------------------------------------------------


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


# -------------------------------------------------------------------------
# Transformation output
# -------------------------------------------------------------------------


def _resolve_transform_output(
    result: str | Path | None,
    output_name: str,
    workdir: Path,
) -> Path:
    """Resolve a transformation output returned by the graph transformer."""
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


# -------------------------------------------------------------------------
# Cycle execution
# -------------------------------------------------------------------------


def _transform_cycle(
    graph: NeuromapsGraph,
    metric_file: Path,
    path: tuple[str, ...],
    hemisphere: Literal["left", "right"],
    workdir: Path,
    density: str,
) -> Path:
    """Transform a metric through every executable hop in a cycle."""
    current_metric = metric_file

    for hop, (source, target) in enumerate(pairwise(path)):
        available = _find_executable_surface(
            graph,
            source,
            target,
            hemisphere,
            density,
        )

        if available is None:
            raise RuntimeError(
                f"No executable surface transform for hop "
                f"'{source}' -> '{target}' at density '{density}' "
                f"on path {' -> '.join(path)}"
            )

        _, geometry = available

        logger.info(
            "hop %d: %s -> %s | source_density=%s | target_density=%s | "
            "geometry=%s",
            hop,
            source,
            target,
            density,
            density,
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
            source_density=density,
            target_density=density,
        )

        current_metric = _resolve_transform_output(
            result,
            output_name,
            workdir,
        )

    return current_metric


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


def test_surface_transform_cycles(
    graph: NeuromapsGraph,
    output_dir: Path,
    tmp_path: Path,
) -> None:
    """Benchmark all executable surface transformation round trips."""
    logging.basicConfig(level=logging.INFO)

    density = DENSITY

    # ------------------------------------------------------------------
    # Create deterministic metric.
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
    # Find every logical return path.
    # ------------------------------------------------------------------

    logical_paths = find_return_paths(
        graph,
        ORIGIN,
    )

    assert logical_paths, (
        f"No return paths found from '{ORIGIN}' "
        "in the surface transformation graph."
    )

    logger.info(
        "Found %d logical return paths from %s",
        len(logical_paths),
        ORIGIN,
    )

    # ------------------------------------------------------------------
    # Expand every logical path into an executable surface path.
    # ------------------------------------------------------------------

    cycles: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    for logical_path in logical_paths:
        executable_path = _expand_logical_path(
            graph,
            logical_path,
            HEMISPHERE,
            density,
        )

        if executable_path is None:
            logger.info(
                "Skipping non-executable logical path: %s",
                " -> ".join(logical_path),
            )
            continue

        cycles.append(
            (
                logical_path,
                executable_path,
            )
        )

    logger.info(
        "Executable return cycles at %s: %d",
        density,
        len(cycles),
    )

    for logical_path, executable_path in cycles:
        logger.info(
            "Executable cycle: %s | path: %s",
            " -> ".join(logical_path),
            " -> ".join(executable_path),
        )

    assert cycles, (
        f"No executable surface transformation cycles found from "
        f"'{ORIGIN}' at density '{density}'."
    )

    # ------------------------------------------------------------------
    # Run every executable round trip.
    # ------------------------------------------------------------------

    results: list[dict[str, object]] = []

    for cycle_number, (logical_path, path) in enumerate(cycles):
        workdir = tmp_path / f"cycle_{cycle_number:03d}"
        workdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Running cycle %d/%d: %s",
            cycle_number + 1,
            len(cycles),
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
            "logical_path": " -> ".join(logical_path),
            "path": " -> ".join(path),
            "n_hops": len(path) - 1,
            "n_logical_hops": len(logical_path) - 1,
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

    assert results, (
        f"No executable round-trip results were produced from '{ORIGIN}'."
    )

    # ------------------------------------------------------------------
    # Summarize all round trips.
    # ------------------------------------------------------------------

    frame = pd.DataFrame(results).sort_values(
        ["n_logical_hops", "logical_path"],
        ascending=[True, True],
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

    # ------------------------------------------------------------------
    # Figure.
    # ------------------------------------------------------------------

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