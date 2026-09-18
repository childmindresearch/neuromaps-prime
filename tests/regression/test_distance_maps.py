"""Anatomical distance-map regression test.

For each space, plant three seeds at anatomical extremes (caudal, lateral,
dorsal), build a great-circle distance map on the sphere for each, transform it
into every other space, and correlate the result against that space's native
map. Two scopes are scored over the same maps: **connected** (every reachable
pair, including composed multi-hop transforms) and **direct** (single-edge
pairs only). Transforms run with ``add_edge=False`` so the graph is left
unmutated. Correlation is signed -- a low or negative r on a transform is the
red flag this suite is meant to surface.

This is a pure producer: it records per-seed matrices, heatmaps, and a
timestamped run-summary CSV, and the pytest checks only confirm the
correlations are well-formed; version-to-version comparison is left to the
accumulated summaries and ``scripts/plot_distance_history.py``.

Run with:

    pytest tests/regression/test_distance_maps.py -v -s
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from tests.cycle import load_metric, resolve_artifact_dir, write_metric
from tests.regression.utils import get_valid_spaces

from neuromaps_prime.analysis.stats import efficient_pearsonr

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)

# --- configuration ---------------------------------------------------------- #
SEEDS = ("caudal", "lateral", "dorsal")
# Surface used to locate the anatomical seed vertices.
SURFACE_TYPE = "midthickness"
HEMISPHERE = "left"
SURFACE_EDGE = "surface_to_surface"

# Where run artifacts (per-seed CSVs, heatmaps, timestamped summaries) go.
OUTPUT_ENV_VAR = "NEUROMAPS_DISTANCE_OUTPUT_DIR"
OUTPUT_SUBDIR = "distance_map_outputs"


def _furthest(column: np.ndarray) -> int:
    """Index of the vertex furthest from the mid-sagittal plane."""
    return int(np.argmax(np.abs(column)))


# Each seed is the extreme vertex along one RAS axis, measured from the mesh
# centroid: (axis index, reducer).
_SEED_AXES: dict[str, tuple[int, Callable[[np.ndarray], int]]] = {
    "caudal": (1, np.argmin),  # most posterior (min y)
    "lateral": (0, _furthest),  # furthest from midline (|x|)
    "dorsal": (2, np.argmax),  # most superior (max z)
}


class StagedMaps(NamedTuple):
    """Native seed maps, staged metric files, and densities, keyed by space."""

    native: dict[str, dict[str, np.ndarray]]
    files: dict[str, dict[str, Path]]
    density: dict[str, str]


class DistanceMapResults(NamedTuple):
    """Per-seed connected/direct matrices and the spaces they cover.

    Attributes:
        connected: Seed -> matrix over every reachable ordered pair.
        direct: Seed -> single-edge pairs scored, others NaN.
        direct_pairs: Ordered pairs joined by a single surface edge.
        spaces: Spaces included in the matrices.
    """

    connected: dict[str, pd.DataFrame]
    direct: dict[str, pd.DataFrame]
    direct_pairs: set[tuple[str, str]]
    spaces: list[str]


def find_direct_pairs(graph: NeuromapsGraph, spaces: list[str]) -> set[tuple[str, str]]:
    """Return ordered space pairs joined by a single surface edge (no concatenation)."""
    space_set = set(spaces)
    subgraph = graph.utils.get_subgraph(SURFACE_EDGE)
    return {
        (u, v)
        for u, v, _key in subgraph.edges
        if u != v and u in space_set and v in space_set
    }


def load_surface_coords(path: str | Path) -> np.ndarray:
    """Return the ``(n, 3)`` point set from a surface GIFTI."""
    for darray in nib.load(path).darrays:
        if darray.data.ndim == 2 and darray.data.shape[1] == 3:
            return np.asarray(darray.data, np.float64)
    raise ValueError(f"No point set found in {path}")


def find_seed_vertices(coords: np.ndarray) -> dict[str, int]:
    """Locate the anatomical seed vertices as RAS-axis extrema on a surface."""
    centered = coords - coords.mean(axis=0)
    return {
        name: int(reducer(centered[:, axis]))
        for name, (axis, reducer) in _SEED_AXES.items()
    }


def distance_maps(
    sphere_coords: np.ndarray, seeds: dict[str, int]
) -> dict[str, np.ndarray]:
    """Great-circle distance from each seed to every sphere vertex, in mm."""
    radius = np.linalg.norm(sphere_coords, axis=1)
    unit = sphere_coords / radius[:, None]
    maps: dict[str, np.ndarray] = {}
    for name, idx in seeds.items():
        cos_angle = np.clip(unit @ unit[idx], -1.0, 1.0)
        maps[name] = radius.mean() * np.arccos(cos_angle)
    return maps


def compute_node_maps(
    graph: NeuromapsGraph, space: str, density: str, hemisphere: str
) -> dict[str, np.ndarray]:
    """Build the three seed distance maps for one node."""
    surface = graph.fetch_surface_atlas(
        space=space, density=density, hemisphere=hemisphere, resource_type=SURFACE_TYPE
    )
    sphere = graph.fetch_surface_atlas(
        space=space, density=density, hemisphere=hemisphere, resource_type="sphere"
    )
    seeds = find_seed_vertices(load_surface_coords(surface.fetch()))
    return distance_maps(load_surface_coords(sphere.fetch()), seeds)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Pearson correlation; NaN when the vertex counts disagree."""
    if a.shape != b.shape:
        logger.warning("Shape mismatch %s vs %s; scoring NaN", a.shape, b.shape)
        return np.nan
    corr, _ = efficient_pearsonr(a, b, return_pval=False)
    return float(corr)


def _stage_native_maps(
    graph: NeuromapsGraph, spaces: list[str], hemisphere: str, workdir: Path
) -> StagedMaps:
    """Compute native maps per space and stage each as a metric file."""
    native: dict[str, dict[str, np.ndarray]] = {}
    files: dict[str, dict[str, Path]] = {}
    density: dict[str, str] = {}
    for space in spaces:
        density[space] = graph.find_highest_density(space)
        native[space] = compute_node_maps(graph, space, density[space], hemisphere)
        files[space] = {
            seed: write_metric(workdir / f"{space}_{seed}_{hemisphere}.func.gii", data)
            for seed, data in native[space].items()
        }
    return StagedMaps(native, files, density)


def _transform_similarity(
    graph: NeuromapsGraph,
    src: str,
    dst: str,
    src_file: Path,
    dst_native: np.ndarray,
    density: dict[str, str],
    hemisphere: str,
    out: Path,
) -> float:
    """Transform one map ``src -> dst`` and correlate it against dst's native map.

    Returns NaN when the pair is unreachable or the transform fails.
    """
    try:
        result = graph.surface_to_surface_transformer(
            transformer_type="metric",
            input_file=src_file,
            source_space=src,
            target_space=dst,
            hemisphere=hemisphere,
            output_file_path=str(out),
            source_density=density[src],
            target_density=density[dst],
            add_edge=False,
        )
    except Exception as exc:
        logger.debug("No transform %s -> %s: %s", src, dst, exc)
        return np.nan
    if result.path is None:
        return np.nan
    return pearson(load_metric(result.path), dst_native)


def _build_seed_matrix(
    graph: NeuromapsGraph,
    spaces: list[str],
    seed: str,
    staged: StagedMaps,
    hemisphere: str,
    workdir: Path,
) -> pd.DataFrame:
    """Assemble the source-by-target correlation matrix for one seed map."""
    matrix = pd.DataFrame(index=spaces, columns=spaces, dtype=float)
    for src in spaces:
        for dst in spaces:
            if src == dst:
                matrix.loc[src, dst] = 1.0
                continue
            out = workdir / f"{src}_to_{dst}_{seed}_{hemisphere}.func.gii"
            matrix.loc[src, dst] = _transform_similarity(
                graph,
                src,
                dst,
                staged.files[src][seed],
                staged.native[dst][seed],
                staged.density,
                hemisphere,
                out,
            )
    return matrix


def _direct_only(
    connected: pd.DataFrame, direct_pairs: set[tuple[str, str]]
) -> pd.DataFrame:
    """Copy a connected matrix, keeping only single-edge pairs (others NaN)."""
    matrix = connected.copy()
    for src in matrix.index:
        for dst in matrix.columns:
            if src != dst and (src, dst) not in direct_pairs:
                matrix.loc[src, dst] = np.nan
    return matrix


def _save_heatmap(matrix: pd.DataFrame, seed: str, kind: str, output_dir: Path) -> Path:
    """Render and save a correlation heatmap for one seed/subtest."""
    mat = matrix.to_numpy(dtype=float)
    size = 1.1 * len(matrix) + 2
    cmap = plt.get_cmap("nipy_spectral")
    fig, ax = plt.subplots(figsize=(size, size))
    # Fixed to the full signed range: a negative r is the warning, not noise.
    im = ax.imshow(mat, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
    ax.set_xlabel("target space (native map)")
    ax.set_ylabel("source space (transformed map)")
    ax.set_title(f"'{seed}' distance-map Pearson r ({kind})")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            if np.isnan(value):
                continue
            red, green, blue, _ = cmap(float(np.clip(value, -1.0, 1.0)))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if luminance < 0.5 else "black",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = output_dir / f"distance_map_{seed}_{kind}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _offdiag_values(matrix: pd.DataFrame) -> np.ndarray:
    """Return the finite off-diagonal values of a correlation matrix."""
    array = matrix.to_numpy(dtype=float)
    off_diagonal = array[~np.eye(len(matrix), dtype=bool)]
    return off_diagonal[np.isfinite(off_diagonal)]


def _pooled_offdiag(matrices: dict[str, pd.DataFrame]) -> np.ndarray:
    """Return all finite off-diagonal correlations across a scope's matrices."""
    per_seed = [_offdiag_values(matrix) for matrix in matrices.values()]
    return np.concatenate(per_seed) if per_seed else np.array([])


def _mean_or_nan(values: np.ndarray) -> float:
    """Mean of a value array; NaN when empty."""
    return float(values.mean()) if values.size else np.nan


def summarize_run(results: DistanceMapResults, hemisphere: str) -> pd.DataFrame:
    """Build the run-summary frame read by scripts/plot_distance_history.py.

    One row per (seed, scope) plus a pooled ``seed='all'`` row per scope, each
    carrying the mean off-diagonal correlation.
    """
    rows: list[dict[str, object]] = []
    for scope, matrices in (
        ("connected", results.connected),
        ("direct", results.direct),
    ):
        for seed, matrix in matrices.items():
            rows.append(
                {
                    "seed": seed,
                    "scope": scope,
                    "hemisphere": hemisphere,
                    "mean_pearson_r": _mean_or_nan(_offdiag_values(matrix)),
                }
            )
        rows.append(
            {
                "seed": "all",
                "scope": scope,
                "hemisphere": hemisphere,
                "mean_pearson_r": _mean_or_nan(_pooled_offdiag(matrices)),
            }
        )
    return pd.DataFrame(rows, columns=["seed", "scope", "hemisphere", "mean_pearson_r"])


def write_run_summary(output_dir: Path, summary: pd.DataFrame) -> Path:
    """Write the timestamped run-summary CSV; return its path."""
    name = f"distance_map_{datetime.now():%Y%m%d_%H%M%S}.csv"
    path = output_dir / name
    summary.to_csv(path, index=False)
    logger.info("Saved run summary CSV: %s", path)
    return path


def run_distance_map_test(
    graph: NeuromapsGraph, hemisphere: str, workdir: Path, output_dir: Path
) -> DistanceMapResults:
    """Compute the connected and direct matrices per seed and record artifacts."""
    spaces = get_valid_spaces(graph, hemisphere, surface_type=SURFACE_TYPE)
    logger.info("Distance-map test over %d spaces: %s", len(spaces), spaces)
    if len(spaces) < 2:
        raise RuntimeError("Need at least two spaces with the seed surface.")

    direct_pairs = find_direct_pairs(graph, spaces)
    logger.info("%d directly connected (single-edge) pairs", len(direct_pairs))
    staged = _stage_native_maps(graph, spaces, hemisphere, workdir)

    connected: dict[str, pd.DataFrame] = {}
    direct: dict[str, pd.DataFrame] = {}
    for seed in SEEDS:
        cmat = _build_seed_matrix(graph, spaces, seed, staged, hemisphere, workdir)
        dmat = _direct_only(cmat, direct_pairs)
        connected[seed] = cmat
        direct[seed] = dmat
        for kind, matrix in (("connected", cmat), ("direct", dmat)):
            matrix.to_csv(output_dir / f"distance_map_{seed}_{kind}.csv")
            _save_heatmap(matrix, seed, kind, output_dir)
        logger.info("\n=== '%s' connected r ===\n%s", seed, cmat.round(3))

    results = DistanceMapResults(connected, direct, direct_pairs, spaces)

    summary = summarize_run(results, hemisphere)
    write_run_summary(output_dir, summary)
    logger.info(
        "NEW MEAN R (all): connected=%.6f, direct=%.6f",
        _mean_or_nan(_pooled_offdiag(results.connected)),
        _mean_or_nan(_pooled_offdiag(results.direct)),
    )
    return results


def _assert_well_formed(values: np.ndarray, scope: str) -> None:
    """Shared assertion: at least one scored, and all values within [-1, 1]."""
    assert values.size > 0, f"No {scope} transforms were evaluated."
    assert np.all((values >= -1.0) & (values <= 1.0)), (
        f"{scope.capitalize()} distance-map correlations fell outside [-1, 1]."
    )


class TestDistanceMaps:
    """Distance-map alignment regression on the real Neuromaps-PRIME graph.

    A class-scoped run scores both the connected and direct scopes across all
    spaces, then records per-seed matrices, heatmaps, and a timestamped
    run-summary CSV. The tests only confirm the correlations are well-formed;
    version-to-version comparison is left to the accumulated summaries and
    ``scripts/plot_distance_history.py``.

    Outputs land under ``<tmp>/distance_map_outputs`` unless
    ``NEUROMAPS_DISTANCE_OUTPUT_DIR`` points at a persistent folder.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def distance_map_run(
        cls, graph: NeuromapsGraph, tmp_path_factory: pytest.TempPathFactory
    ) -> DistanceMapResults:
        """Run the transforms once and share the matrices across both subtests."""
        output_dir = resolve_artifact_dir(
            tmp_path_factory.getbasetemp() / OUTPUT_SUBDIR, env_var=OUTPUT_ENV_VAR
        )
        logger.info("Distance-map artifacts -> %s", output_dir)
        workdir = tmp_path_factory.mktemp("distance_work")
        return run_distance_map_test(graph, HEMISPHERE, workdir, output_dir)

    def test_distance_maps_all_connected(
        self, distance_map_run: DistanceMapResults
    ) -> None:
        """Connected transforms (including composed) run and score within [-1, 1]."""
        _assert_well_formed(_pooled_offdiag(distance_map_run.connected), "connected")

    def test_distance_maps_direct_edges(
        self, distance_map_run: DistanceMapResults
    ) -> None:
        """Direct single-edge transforms run and score within [-1, 1]."""
        _assert_well_formed(
            _pooled_offdiag(distance_map_run.direct), "direct (single-edge)"
        )
