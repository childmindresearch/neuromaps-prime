"""Cycle test: round-trip a surface metric through every return path.

A *return path* (or *cycle*) is a simple path that starts and ends at the same
space, e.g. ``A -> B -> A`` or ``A -> B -> C -> A``. If every surface transform
along the path were perfect, a metric carried around the path would map back
onto itself exactly. In practice each resampling step introduces error, so the
Pearson correlation between the original metric and its round-trip is a proxy
for the *combined* quality of the transforms traversed on that path.

The cycle test therefore:

1. enumerates every simple return path from an origin space over the
   surface-to-surface layer of the graph;
2. carries a seed metric hop-by-hop around each path back to the origin;
3. correlates the round-tripped metric against the original.

This module holds the reusable machinery (enumeration, round-trip, scoring) so
that the deployed regression test and its unit test exercise *the same code*.
The unit test (``tests/unit/test_cycle.py``) validates this machinery on a
synthetic three-node network whose transforms are pure ``+/-120`` degree
rotations, for which the ground-truth answer is known: every return path is the
identity, so every correlation must be ~1.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import networkx as nx
import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)

SURFACE_EDGE = "surface_to_surface"


def resolve_hop_transforms(
    graph: "NeuromapsGraph",
    path: tuple[str, ...],
    hemisphere: Literal["left", "right"],
    density: str,
    edge_type: str = SURFACE_EDGE,
) -> list[tuple[str, str, str, Path]]:
    """Return the actual sphere transform file for every direct graph edge used.

    Each *logical* hop ``(src, dst)`` in *path* may itself span multiple edges
    inside the graph (if there is no direct registration).  This function
    expands every logical hop into its shortest internal edge sequence and
    looks up the on-disk transform file for each direct edge, giving a
    complete audit trail of which files were used.

    For each direct edge the cache is first queried at *density*; if no match
    is found at that exact density (e.g. the reverse-direction transform is
    registered at a different density), the first available density is used as
    a fallback so the entry is never silently dropped.

    Args:
        graph: Populated :class:`~neuromaps_prime.graph.NeuromapsGraph`.
        path: Cycle path, e.g. ``('A', 'B', 'C', 'A')``.
        hemisphere: ``'left'`` or ``'right'``.
        density: Preferred surface mesh density for the lookup.
        edge_type: Graph edge layer to traverse (default ``'surface_to_surface'``).

    Returns:
        List of ``(source_space, target_space, density_used, file_path)``
        tuples, one per direct graph edge, in traversal order.
    """
    rows: list[tuple[str, str, str, Path]] = []
    for src, dst in pairwise(path):
        internal_path = graph.surface_ops.utils.find_path(src, dst, edge_type)
        for hop_src, hop_dst in pairwise(internal_path):
            # Try exact density first.
            transform = graph.surface_ops.cache.get_surface_transform(
                source=hop_src,
                target=hop_dst,
                density=density,
                hemisphere=hemisphere,
                resource_type="sphere",
            )
            density_used = density
            if transform is None:
                # Fall back to any registered density for this edge.
                candidates = graph.surface_ops.cache.get_surface_transforms(
                    source=hop_src,
                    target=hop_dst,
                    hemisphere=hemisphere,
                    resource_type="sphere",
                )
                if candidates:
                    transform = candidates[0]
                    density_used = transform.density
                    logger.warning(
                        "No sphere transform for %s -> %s at density %s (%s); "
                        "using density %s instead.",
                        hop_src,
                        hop_dst,
                        density,
                        hemisphere,
                        density_used,
                    )
                else:
                    logger.warning(
                        "No sphere transform found for %s -> %s (%s) at any density.",
                        hop_src,
                        hop_dst,
                        hemisphere,
                    )
            if transform is not None:
                rows.append((hop_src, hop_dst, density_used, transform.file_path))
    return rows


def _path_token(path: tuple[str, ...]) -> str:
    """Return a short deterministic token for a traversal path."""
    digest = hashlib.sha1("->".join(path).encode("utf-8")).hexdigest()[:12]
    return f"{path[0]}_{path[-1]}_{len(path) - 1}h_{digest}"


@dataclass(frozen=True)
class CycleResult:
    """Outcome of round-tripping a metric around one return path.

    Attributes:
        path: Ordered space names, starting and ending at the origin space.
        pearson_r: Pearson correlation between the original metric and its
            round-trip. Approaches ``1.0`` as the traversed transforms approach
            a perfect identity.
        max_abs_diff: Largest absolute vertex-wise difference between the
            original and round-tripped metric.
    """

    path: tuple[str, ...]
    pearson_r: float
    max_abs_diff: float

    @property
    def label(self) -> str:
        """Return a human-readable ``A -> B -> A`` label for the path."""
        return " -> ".join(self.path)


def find_return_paths(
    graph: NeuromapsGraph,
    origin: str,
    edge_type: str = SURFACE_EDGE,
    *,
    max_length: int | None = None,
    allow_revisits: bool = False,
    max_paths: int | None = None,
) -> list[tuple[str, ...]]:
    """Enumerate every simple return path ``origin -> ... -> origin``.

    Traversal is restricted to a single edge layer (surface-to-surface by
    default) so that the cycle test scores one transform modality at a time.
    By default this enumerates directed simple cycles (no repeated interior
    nodes), rotated so that they start and end at ``origin``.

    When ``allow_revisits=True``, this instead enumerates round-trips composed
    of two directed simple legs: an outbound simple path from ``origin`` to a
    turning node, plus a return simple path from that node back to ``origin``.
    A node can therefore appear at most once per leg (at most twice overall),
    which avoids degenerate ping-pong walks while still permitting bridge nodes
    to be crossed once out and once back.

    Args:
        graph: Populated :class:`~neuromaps_prime.graph.NeuromapsGraph`.
        origin: Space the metric starts from and must return to.
        edge_type: Edge key to traverse (``'surface_to_surface'`` or
            ``'volume_to_volume'``).
        max_length: Optional cap on the number of edges in a path. ``None``
            enumerates all simple cycles, which grows combinatorially on dense
            graphs; bound it on the real graph. Required when
            ``allow_revisits=True``.
        allow_revisits: If ``True``, enumerate two-leg round-trips (outbound +
            return simple paths) up to ``max_length``.
        max_paths: Optional cap on number of returned paths. Useful with
            ``allow_revisits=True`` to avoid combinatorial blow-up.

    Returns:
        Return paths sorted by length then lexicographically, e.g.
        ``[('A', 'B', 'A'), ('A', 'C', 'A'), ('A', 'B', 'C', 'A'), ...]``.

    Raises:
        ValueError: If ``origin`` is not a node in the graph.
    """
    subgraph = graph.utils.get_subgraph(edge_type)
    if origin not in subgraph:
        raise ValueError(
            f"Origin space '{origin}' is not in the '{edge_type}' layer. "
            f"Available: {sorted(subgraph.nodes)}"
        )

    if not allow_revisits:
        paths: list[tuple[str, ...]] = []
        for cycle in nx.simple_cycles(subgraph, length_bound=max_length):
            if origin not in cycle:
                continue
            # Rotate the directed cycle so it starts at origin, then close it.
            start = cycle.index(origin)
            rotated = cycle[start:] + cycle[:start] + [origin]
            paths.append(tuple(rotated))
        return sorted(paths, key=lambda p: (len(p), p))

    if max_length is None:
        raise ValueError("max_length is required when allow_revisits=True")

    # Enumerate bounded round-trips built from two simple directed legs.
    found: set[tuple[str, ...]] = set()
    # Turning node cannot be origin, otherwise round-trip is length 0.
    turn_nodes = [n for n in subgraph.nodes if n != origin]
    for turn in turn_nodes:
        # Outbound path: origin -> ... -> turn
        for outbound in nx.all_simple_paths(
            subgraph,
            source=origin,
            target=turn,
            cutoff=max_length - 1,
        ):
            out_hops = len(outbound) - 1
            if out_hops < 1:
                continue

            remaining = max_length - out_hops
            if remaining < 1:
                continue

            # Return path: turn -> ... -> origin
            for inbound in nx.all_simple_paths(
                subgraph,
                source=turn,
                target=origin,
                cutoff=remaining,
            ):
                in_hops = len(inbound) - 1
                total_hops = out_hops + in_hops
                if total_hops < 2 or total_hops > max_length:
                    continue

                # Stitch by dropping duplicated turning node.
                roundtrip = tuple(outbound + inbound[1:])

                # Enforce "once per leg" / "at most twice overall" semantics.
                counts: dict[str, int] = {}
                for node in roundtrip:
                    counts[node] = counts.get(node, 0) + 1
                if counts.get(origin, 0) != 2:
                    continue
                if any(node != origin and count > 2 for node, count in counts.items()):
                    continue

                found.add(roundtrip)

    ordered = sorted(found, key=lambda p: (len(p), p))
    if max_paths is not None:
        return ordered[:max_paths]
    return ordered


def _load_metric(metric_file: str | Path) -> np.ndarray:
    """Load the first data array of a metric GIFTI as a 1-D float array."""
    return np.asarray(nib.load(str(metric_file)).darrays[0].data, dtype=np.float64)


def roundtrip_metric(
    graph: NeuromapsGraph,
    metric_file: str | Path,
    path: tuple[str, ...],
    hemisphere: Literal["left", "right"],
    workdir: str | Path,
    *,
    density: str | None = None,
    add_edge: bool = False,
) -> Path:
    """Carry a metric hop-by-hop along ``path``, returning the final file.

    Every consecutive pair in ``path`` is a direct edge, so each hop is a
    single-hop resample performed by the production transformer
    (:meth:`NeuromapsGraph.surface_to_surface_transformer`). The output of one
    hop becomes the input of the next.

    Args:
        graph: Populated :class:`~neuromaps_prime.graph.NeuromapsGraph`.
        metric_file: Seed metric GIFTI defined on the origin space.
        path: Return path from :func:`find_return_paths`.
        hemisphere: ``'left'`` or ``'right'``.
        workdir: Directory for the per-hop intermediate metric files.
        density: Fixed mesh density to use for every hop. When ``None`` the
            transformer estimates the source density and resamples to each
            target's highest available density.
        add_edge: Whether the transformer may register composed transforms as
            new edges. Off by default so the cycle test does not mutate the
            graph it is measuring.

    Returns:
        Path to the metric after it has returned to the origin space.

    Raises:
        RuntimeError: If any hop cannot be resolved to a transform.
    """
    workdir = Path(workdir)
    current = Path(metric_file)
    path_token = _path_token(path)
    for hop, (src, dst) in enumerate(pairwise(path)):
        out_name = f"cycle_{path_token}_hop{hop:02d}_{src}-to-{dst}.func.gii"
        result = graph.surface_to_surface_transformer(
            transformer_type="metric",
            input_file=current,
            source_space=src,
            target_space=dst,
            hemisphere=hemisphere,
            # Keep output path relative so dockerized wb_command writes inside
            # its mounted output directory instead of an unmapped host path.
            output_file_path=out_name,
            source_density=density,
            target_density=density,
            add_edge=add_edge,
        )
        if result is None:
            raise RuntimeError(
                f"No surface transform for hop '{src}' -> '{dst}' "
                f"on path {' -> '.join(path)}"
            )
        current = Path(result)
    return current


def score_roundtrip(
    original_file: str | Path, roundtrip_file: str | Path
) -> tuple[float, float]:
    """Return ``(pearson_r, max_abs_diff)`` for an original vs round-tripped metric.

    Args:
        original_file: The seed metric on the origin space.
        roundtrip_file: The metric after it has returned to the origin space.

    Returns:
        Tuple of the Pearson correlation and the maximum absolute vertex-wise
        difference between the two metrics.

        Pearson correlation is undefined when either metric is constant
        (zero variance). For cycle-test sanity checks we return ``1.0`` when
        both constant vectors are equal (treating matching NaNs as equal) and
        ``0.0`` otherwise.

    Raises:
        ValueError: If the two metrics have different vertex counts (which means
            the metric did not return to the origin mesh).
    """
    original = _load_metric(original_file)
    roundtrip = _load_metric(roundtrip_file)
    if original.shape != roundtrip.shape:
        raise ValueError(
            "Round-tripped metric did not return to the origin mesh: "
            f"{roundtrip.shape} vs {original.shape}."
        )
    finite_mask = np.isfinite(original) & np.isfinite(roundtrip)
    if np.any(finite_mask):
        max_abs_diff = float(np.max(np.abs(original[finite_mask] - roundtrip[finite_mask])))
    else:
        max_abs_diff = float("nan")

    vectors_equal = bool(np.allclose(original, roundtrip, equal_nan=True))
    if np.count_nonzero(finite_mask) < 2:
        pearson_r = 1.0 if vectors_equal else 0.0
        return pearson_r, max_abs_diff

    original_finite = original[finite_mask]
    roundtrip_finite = roundtrip[finite_mask]
    original_std = float(np.std(original_finite))
    roundtrip_std = float(np.std(roundtrip_finite))
    if np.isclose(original_std, 0.0) or np.isclose(roundtrip_std, 0.0):
        pearson_r = 1.0 if vectors_equal else 0.0
    else:
        pearson_r = float(np.corrcoef(original_finite, roundtrip_finite)[0, 1])
        if np.isnan(pearson_r) and vectors_equal:
            pearson_r = 1.0

    return pearson_r, max_abs_diff


def run_cycle_test(
    graph: NeuromapsGraph,
    origin: str,
    metric_file: str | Path,
    hemisphere: Literal["left", "right"],
    workdir: str | Path,
    *,
    density: str | None = None,
    max_length: int | None = None,
    output_file: str | Path | None = None,
) -> list[CycleResult]:
    """Round-trip ``metric_file`` around every return path from ``origin``.

    Args:
        graph: Populated :class:`~neuromaps_prime.graph.NeuromapsGraph`.
        origin: Space the metric starts from and must return to.
        metric_file: Seed metric GIFTI defined on the origin space.
        hemisphere: ``'left'`` or ``'right'``.
        workdir: Directory for intermediate metric files.
        density: Fixed mesh density for every hop, or ``None`` to let the
            transformer choose per hop (see :func:`roundtrip_metric`).
        max_length: Optional cap on path length passed to
            :func:`find_return_paths`.
        output_file: Optional path to a text file where the results summary
            will be saved. When ``None`` no file is written.

    Returns:
        One :class:`CycleResult` per return path, in enumeration order.
    """
    results: list[CycleResult] = []
    for path in find_return_paths(graph, origin, max_length=max_length):
        roundtrip = roundtrip_metric(
            graph, metric_file, path, hemisphere, workdir, density=density
        )
        pearson_r, max_abs_diff = score_roundtrip(metric_file, roundtrip)
        logger.info(
            "cycle %s: r=%.6f max|delta|=%.3e",
            " -> ".join(path),
            pearson_r,
            max_abs_diff,
        )
        results.append(
            CycleResult(path=path, pearson_r=pearson_r, max_abs_diff=max_abs_diff)
        )

    # Print summary table to stdout.
    header = f"{'Transformation path':<50}  {'Pearson r':>10}  {'Max |delta|':>14}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for r in results:
        print(f"{r.label:<50}  {r.pearson_r:>10.6f}  {r.max_abs_diff:>14.3e}")
    print(separator)
    print(f"Total cycles: {len(results)}")

    # Optionally save the same summary to a text file.
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            fh.write(f"Cycle test results — origin: {origin}, hemisphere: {hemisphere}\n")
            fh.write(separator + "\n")
            fh.write(header + "\n")
            fh.write(separator + "\n")
            for r in results:
                fh.write(
                    f"{r.label:<50}  {r.pearson_r:>10.6f}  {r.max_abs_diff:>14.3e}\n"
                )
            fh.write(separator + "\n")
            fh.write(f"Total cycles: {len(results)}\n")
        logger.info("Cycle test results saved to %s", output_path)

    return results
