"""Utilities for evaluating cyclic surface transformations.

A cycle (or return path) is a path through the surface transformation graph
that starts and ends at the same space, for example ``A -> B -> A`` or
``A -> B -> C -> A``. A vertex-wise metric propagated around such a path should
return to its original representation if all traversed transformations are
perfectly invertible. In practice, each resampling step introduces numerical
error, so the agreement between the original and round-tripped metric provides
a measure of accumulated cycle error.

This module provides reusable machinery to:

1. enumerate return paths through a graph's surface transformation layer;
2. propagate a seed metric through each transformation hop;
3. compare the returned metric against the original using Pearson correlation
   and maximum vertex-wise absolute difference.

The cycle evaluation operates on complete transformation paths rather than
individual edges, allowing errors introduced across multiple transforms and
resampling operations to be assessed together.

The unit tests exercise this machinery using a synthetic three-space graph with
known rotational transforms. Because the synthetic transformations compose to
identity, all closed paths are expected to return the input metric with near
perfect agreement. Production regression tests use real surface templates and
transformations to evaluate accumulated errors in realistic transformation
networks.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import networkx as nx
import nibabel as nib
import numpy as np

from neuromaps_prime.analysis.images import load_data
from neuromaps_prime.graph import NeuromapsGraph

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

Hemisphere = Literal["left", "right"]


# -------------------------------------------------------------------------
# Path identifiers
# -------------------------------------------------------------------------


def path_token(path: tuple[str, ...]) -> str:
    """Create a deterministic identifier for a transformation path."""
    return hashlib.sha256("->".join(path).encode("utf-8")).hexdigest()[:12]


# -------------------------------------------------------------------------
# Result containers
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class HopResult:
    """Metadata and output from one transformation hop."""

    source: str
    target: str
    area_resource: str
    output_file: Path
    metric_values: np.ndarray


@dataclass(frozen=True)
class RoundtripResult:
    """Output from executing one complete transformation cycle."""

    path: tuple[str, ...]
    final_metric: Path
    hops: tuple[HopResult, ...]


@dataclass(frozen=True)
class CycleResult:
    """Outcome of round-tripping a metric through one return path."""

    path: tuple[str, ...]
    pearson_r: float
    max_abs_diff: float


# -------------------------------------------------------------------------
# Path enumeration
# -------------------------------------------------------------------------


def _iter_roundtrip_paths(
    subgraph: nx.DiGraph, origin: str, max_length: int
) -> Iterator[tuple[str, ...]]:
    """Yield bounded round-trip paths from an origin node.

    Paths consist of two simple directed legs:

    ``origin -> ... -> turn -> ... -> origin``

    A node may therefore occur once on each leg, allowing legitimate
    bidirectional round trips through bridge spaces while avoiding arbitrary
    graph walks and ping-ponging. Because each leg is a simple path and the
    turn node is not the origin, every path visits the origin exactly twice
    and uses at most ``max_length`` hops, so no further filtering is needed.
    """
    turn_nodes = sorted(node for node in subgraph.nodes if node != origin)

    for turn in turn_nodes:
        outbound_paths = nx.all_simple_paths(
            subgraph, source=origin, target=turn, cutoff=max_length - 1
        )

        for outbound in outbound_paths:
            out_hops = len(outbound) - 1

            # Inbound leg gets the remaining budget so the full path
            # never exceeds max_length hops.
            inbound_paths = nx.all_simple_paths(
                subgraph, source=turn, target=origin, cutoff=max_length - out_hops
            )

            for inbound in inbound_paths:
                yield tuple(outbound + inbound[1:])


def find_return_paths(
    graph: NeuromapsGraph,
    origin: str,
    *,
    edge_type: str = NeuromapsGraph.surface_to_surface_key,
    max_length: int | None = None,
    allow_revisits: bool = False,
) -> list[tuple[str, ...]]:
    """Enumerate directed return paths from an origin space.

    By default, paths are directed simple cycles. When ``allow_revisits`` is
    enabled, paths consist of an outbound simple leg and an inbound simple leg,
    permitting bridge spaces to be visited once in each direction.

    Args:
        graph: Populated :class:`NeuromapsGraph`.
        origin: Starting and ending space.
        edge_type: Graph edge layer to traverse.
        max_length: Maximum number of transformation hops.
        allow_revisits: Allow nodes to occur once on each leg.

    Returns:
        Paths sorted by hop count (length) and then lexicographically.
    """
    subgraph = graph.utils.get_subgraph(edge_type)

    if origin not in subgraph:
        raise ValueError(
            f"Origin space '{origin}' is not in the '{edge_type}' layer. "
            f"Available: {sorted(subgraph.nodes)}"
        )

    if allow_revisits:
        if max_length is None:
            raise ValueError("max_length is required when allow_revisits=True.")

        paths = set(_iter_roundtrip_paths(subgraph, origin, max_length))
    else:
        cycles = nx.simple_cycles(subgraph, length_bound=max_length)

        paths = set()

        for cycle in cycles:
            # Rotate the directed cycle so it starts at origin, then close it.
            if origin not in cycle:
                continue

            start = cycle.index(origin)
            rotated = cycle[start:] + cycle[:start] + [origin]

            paths.add(tuple(rotated))

    return sorted(paths, key=lambda path: (len(path) - 1, path))


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------


def write_metric(metric_file: str | Path, values: np.ndarray) -> Path:
    """Write one scalar value per vertex as a GIFTI metric."""
    image = nib.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                np.asarray(values, dtype=np.float32), intent="NIFTI_INTENT_NONE"
            )
        ]
    )

    metric_file = Path(metric_file)
    metric_file.parent.mkdir(parents=True, exist_ok=True)

    nib.save(image, metric_file)

    return metric_file


def load_metric(metric_file: str | Path) -> np.ndarray:
    """Load a scalar GIFTI metric as a one-dimensional array."""
    data, image = load_data(metric_file, dtype=np.float64, return_image=True)

    if not isinstance(image, nib.GiftiImage):
        raise ValueError(f"Expected GIFTI metric file, got {type(image)}.")

    if data.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional metric data; got shape {data.shape}."
        )

    return data


# -------------------------------------------------------------------------
# Hop execution
# -------------------------------------------------------------------------


def _execute_hop(
    graph: NeuromapsGraph,
    metric_file: Path,
    source: str,
    target: str,
    hemisphere: Hemisphere,
    output_file: Path,
    *,
    density: str | None,
    add_edge: bool,
) -> HopResult:
    """Execute one surface transformation with area-surface fallback.

    The production transformer is attempted with midthickness first, followed
    by pial and white if the requested area resource cannot produce a usable
    output.
    """
    for area_resource in ("midthickness", "pial", "white"):
        try:
            result = graph.surface_to_surface_transformer(
                transformer_type="metric",
                input_file=metric_file,
                source_space=source,
                target_space=target,
                hemisphere=hemisphere,
                output_file_path=output_file,
                source_density=density,
                target_density=density,
                area_resource=area_resource,
                add_edge=add_edge,
            )

            if result.path is None:
                raise FileNotFoundError(
                    f"Surface transformation did not produce an output for "
                    f"{source} -> {target} ({hemisphere})."
                )

            output_file = result.path

            if not output_file.exists():
                raise FileNotFoundError(
                    f"Surface transformation output does not exist: {output_file}"
                )

            metric_values = load_metric(output_file)

            if area_resource != "midthickness":
                logger.warning(
                    "Using fallback area surface '%s' for %s -> %s (%s).",
                    area_resource,
                    source,
                    target,
                    hemisphere,
                )

            return HopResult(
                source=source,
                target=target,
                area_resource=area_resource,
                output_file=output_file,
                metric_values=metric_values,
            )

        except (RuntimeError, FileNotFoundError, OSError, ValueError, TypeError) as exc:
            logger.debug(
                "Area surface '%s' failed for %s -> %s (%s): %s",
                area_resource,
                source,
                target,
                hemisphere,
                exc,
            )

    raise RuntimeError(
        f"Could not execute surface transform "
        f"'{source}' -> '{target}' ({hemisphere}). "
        f"Tried midthickness, pial, and white."
    )


# -------------------------------------------------------------------------
# Cycle execution
# -------------------------------------------------------------------------


def _resolve_workdir(workdir: str | Path | None) -> Path:
    """Resolve and create the artifact directory for cycle outputs.

    Callers running under pytest should pass the test's ``tmp_path`` so
    artifacts follow pytest's temporary-file retention policy. Otherwise a
    shared directory under the system temporary area is used.
    """
    resolved = (
        Path(workdir)
        if workdir is not None
        else Path(tempfile.gettempdir()) / "neuromaps_prime" / "cycles"
    )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def roundtrip_metric(
    graph: NeuromapsGraph,
    metric_file: str | Path,
    path: tuple[str, ...],
    hemisphere: Hemisphere,
    *,
    workdir: str | Path | None = None,
    density: str | None = None,
    add_edge: bool = False,
) -> RoundtripResult:
    """Propagate a metric through every hop in a return path.

    When ``density`` is ``None``, the production transformation engine is
    allowed to determine source and target densities independently for each
    hop. This permits cross-density cycles such as:

    ``CIVETNMT -> D99 -> CIVETNMT``

    to use the native mesh available in each space.

    Area surfaces are attempted in order:

    1. midthickness
    2. pial
    3. white

    Args:
        graph: Populated :class:`NeuromapsGraph`.
        metric_file: Seed metric.
        path: Closed transformation path.
        hemisphere: Hemisphere being tested.
        workdir: Directory for intermediate artifacts. When ``None``, a
            shared directory under the system temporary area is used; pass
            the test's pytest ``tmp_path`` in tests.
        density: Optional fixed density. ``None`` delegates density handling
            to the production transformer.
        add_edge: Whether transformations may mutate the graph.

    Returns:
        :class:`RoundtripResult` containing the final metric and every
        intermediate hop.

    Raises:
        RuntimeError: If any hop cannot be executed.
        FileNotFoundError: If a transformation output cannot be recovered.
    """
    workdir = _resolve_workdir(workdir)

    current = Path(metric_file)
    token = path_token(path)
    hops: list[HopResult] = []

    for hop_number, (source, target) in enumerate(pairwise(path)):
        output_file = (
            workdir / f"cycle_{token}_hop{hop_number:02d}_{source}-to-{target}.func.gii"
        )

        logger.info(
            "Executing hop %d: %s -> %s (%s)", hop_number, source, target, hemisphere
        )

        hop_result = _execute_hop(
            graph=graph,
            metric_file=current,
            source=source,
            target=target,
            hemisphere=hemisphere,
            output_file=output_file,
            density=density,
            add_edge=add_edge,
        )

        current = hop_result.output_file
        hops.append(hop_result)

        logger.info(
            "Completed hop %d: %s -> %s using %s",
            hop_number,
            source,
            target,
            hop_result.area_resource,
        )

    return RoundtripResult(path=path, final_metric=current, hops=tuple(hops))


# -------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------


def score_roundtrip(
    original_file: str | Path, roundtrip_file: str | Path
) -> tuple[float, float]:
    """Return ``(pearson_r, max_abs_diff)``."""
    original = load_metric(original_file)
    roundtrip = load_metric(roundtrip_file)

    if original.shape != roundtrip.shape:
        raise ValueError(
            "Round-tripped metric did not return to the origin mesh: "
            f"{roundtrip.shape} vs {original.shape}."
        )

    finite_mask = np.isfinite(original) & np.isfinite(roundtrip)
    finite_count = int(np.count_nonzero(finite_mask))

    if finite_count == 0:
        logger.warning("Round-trip metric contains no finite values.")
        return 0.0, np.nan

    max_abs_diff = float(np.max(np.abs(original[finite_mask] - roundtrip[finite_mask])))

    original_finite = original[finite_mask]
    roundtrip_finite = roundtrip[finite_mask]

    # Pearson correlation is undefined for a constant vector or for fewer
    # than two values; fall back to exact vector agreement.
    constant = bool(
        np.isclose(np.std(original_finite), 0.0)
        or np.isclose(np.std(roundtrip_finite), 0.0)
    )

    if finite_count < 2 or constant:
        if finite_count < 2:
            logger.warning(
                "Fewer than two finite values available for Pearson correlation."
            )
        return (
            1.0 if np.allclose(original, roundtrip, equal_nan=True) else 0.0,
            max_abs_diff,
        )

    pearson_r = float(np.corrcoef(original_finite, roundtrip_finite)[0, 1])

    if np.isnan(pearson_r) and np.allclose(original, roundtrip, equal_nan=True):
        pearson_r = 1.0

    return pearson_r, max_abs_diff


def run_cycle_test(
    graph: NeuromapsGraph,
    origin: str,
    metric_file: str | Path,
    hemisphere: Hemisphere,
    *,
    workdir: str | Path | None = None,
    density: str | None = None,
    max_length: int | None = None,
    allow_revisits: bool = False,
    add_edge: bool = False,
) -> list[CycleResult]:
    """Execute and score all return paths from an origin.

    This is a lightweight orchestration helper intended primarily for unit
    tests. Individual paths that cannot be executed are skipped, allowing the
    caller to inspect all successfully executed cycles.

    Args:
        graph: Populated :class:`NeuromapsGraph`.
        origin: Starting and ending space.
        metric_file: Seed metric.
        hemisphere: Hemisphere being tested.
        workdir: Directory for transformation artifacts. When ``None``, a
            shared directory under the system temporary area is used; pass
            the test's pytest ``tmp_path`` in tests.
        density: Optional fixed surface density.
        max_length: Maximum number of transformation hops.
        allow_revisits: Allow bridge nodes to occur once on each leg.
        add_edge: Whether transformations may mutate the graph.

    Returns:
        A list of :class:`CycleResult` objects for successfully executed
        return paths.
    """
    workdir = _resolve_workdir(workdir)

    paths = find_return_paths(
        graph, origin, max_length=max_length, allow_revisits=allow_revisits
    )

    results: list[CycleResult] = []

    for path in paths:
        token = path_token(path)
        path_workdir = workdir / f"path_{token}"

        try:
            roundtrip = roundtrip_metric(
                graph=graph,
                metric_file=metric_file,
                path=path,
                hemisphere=hemisphere,
                workdir=path_workdir,
                density=density,
                add_edge=add_edge,
            )

            pearson_r, max_abs_diff = score_roundtrip(
                metric_file, roundtrip.final_metric
            )

        except (RuntimeError, FileNotFoundError, OSError, ValueError) as exc:
            logger.warning(
                "Skipping non-executable cycle %s (%s): %s",
                " -> ".join(path),
                hemisphere,
                exc,
            )
            continue

        results.append(
            CycleResult(path=path, pearson_r=pearson_r, max_abs_diff=max_abs_diff)
        )

    return results
