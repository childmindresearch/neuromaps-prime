"""Anatomical distance-map regression test.

This is a coarse alignment sanity check. For each node it plants three seed
vertices at anatomical extremes of a surface (the caudal-most, lateral-most and
dorsal-most vertices of the midthickness by default), builds a smooth
distance-from-seed map over the *sphere* for each seed, transforms those maps
into every other space, and correlates each transformed map against the target
space's own native map. The result is one space-by-space correlation matrix per
seed.

The maps are deliberately smooth, low-frequency gradients: distance from the
back of the brain, from the lateral convexity, and from the top. Cortical
folding differs across species, so these will not correlate perfectly, but the
gross layout is conserved -- the front of the brain stays the front. High
correlation therefore does *not* prove an alignment is good, but a low
correlation is a red flag that a transform may be misdirected or broken and
warrants a look.

Two subtests run over the same computed maps:

* **all connected** -- every reachable ordered pair of spaces, including pairs
  whose transform is a *concatenation* of several edges;
* **direct edges** -- only pairs joined by a single surface edge (no
  concatenation). When a composed path looks bad, the direct-edge matrix tells
  you which individual edge to suspect, which a multi-hop result cannot isolate.

Transforms run with ``add_edge=False`` so composing a multi-hop path never
registers a new direct edge and blurs the direct/composed distinction; the graph
is left unmutated.

Distances are great-circle distances on the registration sphere (an exact
geodesic for a spherical mesh), while the seeds are located on the anatomical
midthickness. Correlation is signed: these gradients have a definite
orientation, so a near-zero or negative correlation is itself the warning.

Cross-version history
---------------------
Like the cycle regression test, each run writes a timestamped run-summary CSV
(``distance_map_<YYYYmmdd_HHMMSS>.csv``; columns ``seed, scope, hemisphere,
mean_pearson_r``) into a resolved artifact directory. By default that directory
is ephemeral (pytest's temp dir); set ``NEUROMAPS_DISTANCE_OUTPUT_DIR`` to a
persistent location and the summaries accumulate, one per run. Render the
timeline across runs with ``scripts/plot_distance_history.py``. This lets you
watch whether a given transform's distance-map correlation drifts between
versions -- the real value of this style of regression test. Like the cycle
test, this suite is a pure producer: the pytest checks only confirm that
transforms executed and that correlations are well-formed, leaving
version-to-version comparison to the accumulated summaries and the history plot.

Seed vertices are chosen as extrema along the RAS anatomical axes
(x = left-right, y = posterior-anterior, z = inferior-superior); adjust
``_SEED_AXES`` if a template's meshes use a different convention.

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
    """Per-seed correlation matrices for both subtests.

    Attributes:
        connected: Seed -> full matrix over every reachable ordered pair.
        direct: Seed -> matrix with only single-edge pairs scored (others NaN).
        direct_pairs: Ordered pairs joined by a single surface edge.
        spaces: Spaces included in the matrices.
    """

    connected: dict[str, pd.DataFrame]
    direct: dict[str, pd.DataFrame]
    direct_pairs: set[tuple[str, str]]
    spaces: list[str]


def find_direct_pairs(graph: NeuromapsGraph, spaces: list[str]) -> set[tuple[str, str]]:
    """Return ordered space pairs joined by a single surface edge.

    A pair is "direct" when the surface layer has an edge between the two
    spaces, i.e. the transform needs no concatenation. Computed from the
    unmutated graph.
    """
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
    """Locate the anatomical seed vertices on a surface.

    Args:
        coords: ``(n, 3)`` vertex coordinates in RAS millimetres.

    Returns:
        Mapping of seed name to vertex index.
    """
    centered = coords - coords.mean(axis=0)
    return {
        name: int(reducer(centered[:, axis]))
        for name, (axis, reducer) in _SEED_AXES.items()
    }


def distance_maps(
    sphere_coords: np.ndarray, seeds: dict[str, int]
) -> dict[str, np.ndarray]:
    """Great-circle distance from each seed to every vertex, over the sphere.

    Args:
        sphere_coords: ``(n, 3)`` sphere vertex coordinates.
        seeds: Mapping of seed name to vertex index.

    Returns:
        Mapping of seed name to a per-vertex distance map (millimetres).
    """
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
        return float("nan")
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
    """Transform one map ``src -> dst`` and correlate against the native map.

    Returns NaN when the pair is unreachable or the transform fails. Runs with
    ``add_edge=False`` so the graph is never mutated.
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
        return float("nan")
    if result.path is None:
        return float("nan")
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
                matrix.loc[src, dst] = float("nan")
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
    return float(values.mean()) if values.size else float("nan")


def summarize_run(results: DistanceMapResults, hemisphere: str) -> pd.DataFrame:
    """Build the canonical run-summary frame (seed x scope means).

    One row per (seed, scope) plus a pooled ``seed='all'`` row per scope, giving
    the mean off-diagonal correlation. This is the schema accumulated across runs
    and read by ``scripts/plot_distance_history.py``.
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
    """Compute connected and direct-edge matrices per seed and record artifacts.

    Args:
        graph: A fully built :class:`NeuromapsGraph`.
        hemisphere: ``'left'`` or ``'right'``.
        workdir: Scratch directory for intermediate GIFTI files.
        output_dir: Directory for per-seed CSVs, heatmaps, and the timestamped
            run summary.

    Returns:
        A :class:`DistanceMapResults`.
    """
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

    One class-scoped run plants seed vertices at anatomical extremes, builds
    smooth distance-from-seed maps, transforms each map into every other space,
    and correlates each transformed map against the target's own native map,
    scoring both the all-connected and the direct-edge scopes. The suite is a
    pure producer: it records this run's artifacts and only checks that the
    correlations are well-formed, leaving version-to-version comparison to the
    accumulated summaries and ``scripts/plot_distance_history.py``.

    Outputs land under the pytest temporary directory
    (``<base>/distance_map_outputs``) unless the ``NEUROMAPS_DISTANCE_OUTPUT_DIR``
    environment variable is set; the resolved location is logged at run start.
    Point the variable at a persistent location to accumulate run summaries
    across runs.
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
        """Connected transforms (including composed) run and score in range.

        Pure producer: this records the connected matrices for cross-version
        tracking and only checks the correlations are well-formed, not that they
        clear any floor.
        """
        _assert_well_formed(_pooled_offdiag(distance_map_run.connected), "connected")

    def test_distance_maps_direct_edges(
        self, distance_map_run: DistanceMapResults
    ) -> None:
        """Direct single-edge transforms run and score in range.

        The direct matrices isolate individual edges (no concatenation), which is
        what localises drift in the history plot. Well-formedness only, no floor.
        """
        _assert_well_formed(
            _pooled_offdiag(distance_map_run.direct), "direct (single-edge)"
        )
