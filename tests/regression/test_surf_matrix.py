"""Surface-transform-matrix regression test on the real Neuromaps-PRIME graph.

Computes the directed pairwise surface-to-surface transform error matrix: for
every ordered pair of atlas spaces, the source's midthickness surface is
resampled onto the destination's sphere (sphere-based barycentric mapping) and
compared against the destination's midthickness surface. Each matrix entry is
the median vertex-wise absolute signed distance of that resampled surface from
the target.

The suite is a pure producer: one class-scoped run executes the transforms,
computes the directed matrix and its derived (asymmetric / symmetric) forms
plus the off-diagonal statistics, and writes this run's artifacts (the matrix
CSV, one raw error GIFTI per ordered pair, a human-readable summary, and the
heatmap / histogram plots). It reads no previous baseline and makes no
cross-run comparison — those are CI concerns (see
``.notes/rf_regression_test/plan.md``, Deferred). Plots are currently written
inline and move to ``scripts/plot_surf_matrix.py`` in Phase 4.

Outputs land under the pytest temporary directory
(``<tmp_path>/surface_matrix_outputs``) unless the
``NEUROMAPS_SURF_MATRIX_OUTPUT_DIR`` environment variable is set; the resolved
location is logged at run start. Point the variable at a persistent location to
accumulate run artifacts across runs.

Run with:

    pytest tests/regression/test_surf_matrix.py -v -s
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.axes._axes as mpl_axes
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from niwrap import workbench
from tests.cycle import resolve_artifact_dir

from neuromaps_prime.transforms.utils import relocate_output

if TYPE_CHECKING:
    from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)

# The matrix is evaluated for a single hemisphere (matching the original test).
HEMISPHERE = "right"


# -------------------------------------------------------------------------
# Result containers
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class OffDiagStats:
    """Off-diagonal (cross-space) transform-error statistics.

    ``min_pair`` / ``max_pair`` are ``(source, target, value)`` tuples for the
    lowest and highest off-diagonal matrix entries. ``nhp_median`` and
    ``human_nhp_median`` are medians restricted to NHP-only and human<->NHP
    pairs respectively; ``global_median`` spans every off-diagonal entry.
    """

    global_median: float
    min_pair: tuple[str, str, float]
    max_pair: tuple[str, str, float]
    nhp_median: float
    human_nhp_median: float


@dataclass(frozen=True)
class SurfaceMatrixResult:
    """Outcome of one surface-transform-matrix run.

    ``matrix`` is directed: rows are the source space, columns the target
    space, each entry the median absolute signed distance after resampling.
    ``asymmetry`` is ``matrix - matrix.T`` (directionality bias); ``symmetric``
    is ``(matrix + matrix.T) / 2`` (undirected distance). ``all_errors`` is the
    concatenation of every pair's vertex-wise absolute signed distance (the
    histogram input). ``output_dir`` is where this run's artifacts were written.
    """

    hemisphere: str
    spaces: list[str]
    matrix: pd.DataFrame
    asymmetry: pd.DataFrame
    symmetric: pd.DataFrame
    all_errors: np.ndarray
    off_diag: OffDiagStats
    output_dir: Path
    csv_path: Path
    summary_path: Path


# -------------------------------------------------------------------------
# Space selection
# -------------------------------------------------------------------------


def get_valid_spaces(graph: NeuromapsGraph, hemisphere: str) -> list[str]:
    """Return the graph nodes that expose both a sphere and a midthickness.

    A space is usable for the matrix only if it provides both a sphere (to
    define the resampling mapping) and a midthickness surface (the geometry
    being compared) at its highest available density for the hemisphere.
    """
    valid = []

    for node in graph.nodes:
        try:
            density = graph.find_highest_density(node)

            sphere = graph.fetch_surface_atlas(
                space=node,
                density=density,
                hemisphere=hemisphere,
                resource_type="sphere",
            )

            midthickness = graph.fetch_surface_atlas(
                space=node,
                density=density,
                hemisphere=hemisphere,
                resource_type="midthickness",
            )

            if sphere is not None and midthickness is not None:
                valid.append(node)

        except Exception as exc:  # A node that fails to probe is skipped, not fatal.
            logger.debug("Skipping node %s due to error: %s", node, exc)

    return valid


# -------------------------------------------------------------------------
# Pair computation
# -------------------------------------------------------------------------


def _fetch_atlas(
    graph: NeuromapsGraph, space: str, density: str, hemisphere: str, resource_type: str
) -> Path:
    """Fetch one surface-atlas resource as a host path.

    Raises:
        FileNotFoundError: If the resource is not present in the graph cache.
    """
    atlas = graph.fetch_surface_atlas(
        space=space, density=density, hemisphere=hemisphere, resource_type=resource_type
    )

    if atlas is None:
        raise FileNotFoundError(
            f"No {resource_type} atlas for {space} at {density} ({hemisphere})."
        )
    return Path(atlas.fetch())


def _compute_pair(
    graph: NeuromapsGraph,
    src: str,
    dst: str,
    hemisphere: str,
    workdir: Path,
    errors_dir: Path,
) -> tuple[float, np.ndarray]:
    """Resample ``src`` midthickness onto ``dst`` and measure the error.

    The source midthickness surface is resampled onto the destination sphere,
    and the vertex-wise absolute signed distance from that resampled surface to
    the destination midthickness surface is measured. The raw per-pair error is
    persisted under ``errors_dir`` so it can be rebuilt for post-hoc plotting.

    Returns:
        ``(median_error, vertex_errors)`` for the ordered pair ``src -> dst``.
    """
    src_density = graph.find_highest_density(src)
    dst_density = graph.find_highest_density(dst)

    # Midthickness defines the geometry being compared; the sphere defines the
    # mapping used for the resampling.
    src_surface = _fetch_atlas(graph, src, src_density, hemisphere, "midthickness")
    dst_surface = _fetch_atlas(graph, dst, dst_density, hemisphere, "midthickness")
    src_sphere = _fetch_atlas(graph, src, src_density, hemisphere, "sphere")
    dst_sphere = _fetch_atlas(graph, dst, dst_density, hemisphere, "sphere")

    out_surface = workdir / f"{src}_to_{dst}.surf.gii"
    area_surfs = {"current-area": src_surface, "new-area": dst_surface}

    # The source and destination are not the same mesh, so resample the source
    # surface onto the destination sphere.
    resampled = workbench.surface_resample(
        surface_in=src_surface,
        current_sphere=src_sphere,
        new_sphere=dst_sphere,
        method="ADAP_BARY_AREA",
        area_surfs=area_surfs,
        surface_out=out_surface.name,
    )
    relocate_output(resampled.surface_out, out_surface)

    error_file = errors_dir / f"{src}_to_{dst}.func.gii"
    distance = workbench.signed_distance_to_surface(
        surface_comp=out_surface, surface_ref=dst_surface, metric=error_file.name
    )
    relocate_output(distance.metric, error_file)

    vertex_errors = np.abs(nib.load(error_file).darrays[0].data)
    median_err = float(np.median(vertex_errors))

    logger.info(
        "Error %s -> %s: median=%.5f mean=%.5f std=%.5f",
        src,
        dst,
        median_err,
        float(np.mean(vertex_errors)),
        float(np.std(vertex_errors)),
    )

    return median_err, vertex_errors


# -------------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------------


def _off_diag_stats(spaces: list[str], matrix: pd.DataFrame) -> OffDiagStats:
    """Compute off-diagonal (cross-space) statistics from the matrix."""
    mat = matrix.to_numpy()
    n = len(matrix)

    mask = ~np.eye(n, dtype=bool)
    off_diag_vals = mat[mask]

    human_spaces = [s for s in spaces if s == "fsLR"]
    nhp_spaces = [s for s in spaces if s != "fsLR"]

    global_median = float(np.nanmedian(off_diag_vals))

    pairs = [
        (spaces[i], spaces[j], float(mat[i, j]))
        for i in range(n)
        for j in range(n)
        if i != j
    ]

    min_src, min_dst, min_val = min(pairs, key=lambda pair: pair[2])
    max_src, max_dst, max_val = max(pairs, key=lambda pair: pair[2])

    nhp_index = [spaces.index(s) for s in nhp_spaces]
    nhp_mask = np.zeros((n, n), dtype=bool)
    for i in nhp_index:
        for j in nhp_index:
            nhp_mask[i, j] = True

    nhp_off_diag = mat[np.logical_and(mask, nhp_mask)]
    nhp_median = (
        float(np.nanmedian(nhp_off_diag)) if nhp_off_diag.size else float("nan")
    )

    human_nhp_vals = [
        float(mat[i, j])
        for i, si in enumerate(spaces)
        for j, sj in enumerate(spaces)
        if i != j
        and (
            (si in human_spaces and sj in nhp_spaces)
            or (si in nhp_spaces and sj in human_spaces)
        )
    ]

    human_nhp_vals = np.array(human_nhp_vals) if human_nhp_vals else np.array([np.nan])

    return OffDiagStats(
        global_median=global_median,
        min_pair=(min_src, min_dst, min_val),
        max_pair=(max_src, max_dst, max_val),
        nhp_median=nhp_median,
        human_nhp_median=float(np.nanmedian(human_nhp_vals)),
    )


# -------------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------------


def _format_summary(
    hemisphere: str,
    matrix: pd.DataFrame,
    asymmetry: pd.DataFrame,
    symmetric: pd.DataFrame,
    off_diag: OffDiagStats,
) -> str:
    """Render a human-readable summary of the matrix and its statistics."""
    lines = [
        f"Surface transform error matrix — hemisphere: {hemisphere}",
        "=" * 60,
        "",
        "MATRIX DEFINITIONS",
        "  M[A, B] = median vertex-wise |signed-distance| error after resampling",
        "           A's midthickness onto B's sphere (sphere-based barycentric).",
        "  A[A, B] = M[A, B] - M[B, A]       (directionality bias).",
        "  S[A, B] = (M[A, B] + M[B, A]) / 2  (undirected distance).",
        "  Rows = source space, Columns = target space.",
        "",
        "=== TRANSFORM ERROR MATRIX (median) ===",
        matrix.to_string(),
        "",
        "=== ASYMMETRIC MATRIX ===",
        asymmetry.to_string(),
        "",
        "=== SYMMETRIC MATRIX ===",
        symmetric.to_string(),
        "",
        "=== OFF-DIAGONAL TRANSFORM ERROR STATS ===",
        f"global median:      {off_diag.global_median:.5f}",
        f"global min:         {off_diag.min_pair[2]:.5f}  "
        f"({off_diag.min_pair[0]} -> {off_diag.min_pair[1]})",
        f"global max:         {off_diag.max_pair[2]:.5f}  "
        f"({off_diag.max_pair[0]} -> {off_diag.max_pair[1]})",
        f"NHP-only median:    {off_diag.nhp_median:.5f}",
        f"Human<->NHP median: {off_diag.human_nhp_median:.5f}",
        "",
    ]

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Producer
# -------------------------------------------------------------------------


def run_surface_matrix(
    graph: NeuromapsGraph, hemisphere: str, *, output_dir: str | Path
) -> SurfaceMatrixResult:
    """Compute the full directed surface-transform error matrix.

    For every ordered pair of valid spaces, the source midthickness surface is
    resampled onto the destination sphere and the vertex-wise absolute signed
    distance to the destination midthickness surface is measured. The directed
    matrix, its asymmetric and symmetric derived forms, and the off-diagonal
    statistics are computed, and this run's artifacts are written to
    ``output_dir``:

    * ``surface_transform_matrix.csv`` — the directed matrix (rows=source, cols=target);
    * ``errors/{src}_to_{dst}.func.gii`` — the raw per-pair signed-distance error;
    * ``surface_matrix_summary.txt`` — a human-readable summary.

    Args:
        graph: Populated :class:`NeuromapsGraph`.
        hemisphere: Hemisphere to evaluate (e.g. ``"left"`` or ``"right"``).
        output_dir: Directory for this run's artifacts; created if missing.

    Returns:
        A :class:`SurfaceMatrixResult` carrying the matrices, statistics, and
        the artifact locations.

    Raises:
        ValueError: If fewer than two spaces are usable for the hemisphere.
    """
    output_dir = resolve_artifact_dir(output_dir)
    workdir = resolve_artifact_dir(output_dir / "work")
    errors_dir = resolve_artifact_dir(output_dir / "errors")

    spaces = get_valid_spaces(graph, hemisphere)

    if len(spaces) < 2:
        raise ValueError(
            f"Fewer than two usable spaces for {hemisphere}; "
            f"cannot form a pairwise matrix (found {len(spaces)})."
        )

    logger.info("=== BUILDING SURFACE TRANSFORM MATRIX (%s) ===", hemisphere)

    results: dict[tuple[str, str], float] = {}
    all_errors: list[np.ndarray] = []

    for src, dst in product(spaces, spaces):
        logger.info("=== %s -> %s ===", src, dst)

        median_err, vertex_errors = _compute_pair(
            graph=graph,
            src=src,
            dst=dst,
            hemisphere=hemisphere,
            workdir=workdir,
            errors_dir=errors_dir,
        )

        results[(src, dst)] = median_err
        all_errors.append(vertex_errors)

    matrix = pd.DataFrame(index=spaces, columns=spaces, dtype=float)

    for (src, dst), value in results.items():
        matrix.loc[src, dst] = value

    asymmetry = matrix - matrix.T
    symmetric = (matrix + matrix.T) / 2.0
    off_diag = _off_diag_stats(spaces, matrix)
    concatenated = np.concatenate(all_errors) if all_errors else np.array([])

    csv_path = output_dir / "surface_transform_matrix.csv"
    matrix.to_csv(csv_path)

    summary_path = output_dir / "surface_matrix_summary.txt"
    summary_path.write_text(
        _format_summary(hemisphere, matrix, asymmetry, symmetric, off_diag),
        encoding="utf-8",
    )

    logger.info("Saved matrix CSV: %s", csv_path)
    logger.info("Saved summary: %s", summary_path)

    return SurfaceMatrixResult(
        hemisphere=hemisphere,
        spaces=spaces,
        matrix=matrix,
        asymmetry=asymmetry,
        symmetric=symmetric,
        all_errors=concatenated,
        off_diag=off_diag,
        output_dir=output_dir,
        csv_path=csv_path,
        summary_path=summary_path,
    )


# -------------------------------------------------------------------------
# Plotting (moves to scripts/plot_surf_matrix.py in Phase 4)
# -------------------------------------------------------------------------


def annotate_heatmap(ax: mpl_axes.Axes, mat: np.ndarray) -> None:
    """Annotate a heatmap with the numeric value in each cell."""
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )


def _style_matrix_axes(ax: mpl_axes.Axes, spaces: list[str]) -> None:
    """Apply the shared axis configuration to a matrix heatmap."""
    n = len(spaces)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Target template space", labelpad=12)
    ax.set_ylabel("Source template space", labelpad=12)
    ax.set_xticklabels(spaces, rotation=45, ha="right")
    ax.set_yticklabels(spaces)


def _plot_matrix_heatmaps(result: SurfaceMatrixResult, output_dir: Path) -> None:
    """Write the full-scale and NHP-scaled transform-error heatmaps."""
    mat = result.matrix.to_numpy()
    spaces = result.spaces
    nhp_spaces = [s for s in spaces if s != "fsLR"]

    # Full-scale heatmap.
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im1 = ax1.imshow(mat, interpolation="nearest", cmap="turbo")
    annotate_heatmap(ax1, mat)
    _style_matrix_axes(ax1, spaces)
    ax1.set_title(
        "Surface Transform Error Matrix (Full Scale, Including fsLR)",
        fontweight="bold",
        color="#0044AA",
        pad=10,
    )
    cbar1 = fig1.colorbar(im1, ax=ax1)
    cbar1.set_label("Median absolute signed-distance error", rotation=90, labelpad=12)
    fig1.tight_layout()
    full_caption = (
        "Figure 3. Heatmap of pairwise surface-to-surface transform error "
        "between atlas spaces. Each matrix entry represents the median "
        "vertex-wise absolute signed-distance error after resampling one "
        "midthickness surface onto another using sphere-based barycentric "
        "registration."
    )
    fig1.subplots_adjust(left=0.10, bottom=0.30)
    fig1.text(
        0.5, 0.02, full_caption, ha="center", fontsize=9, fontstyle="italic", wrap=True
    )
    full_path = output_dir / "surface_transform_matrix_full.png"
    fig1.savefig(full_path, dpi=200)
    plt.close(fig1)
    logger.info("Saved full-scale heatmap: %s", full_path)

    # NHP-scaled heatmap: clip the color scale to the NHP value range so the
    # interspecies (fsLR) cells do not dominate the colormap.
    nhp_sub = (
        result.matrix.loc[nhp_spaces, nhp_spaces].to_numpy() if nhp_spaces else mat
    )
    vmin = 0.0
    vmax = float(np.nanpercentile(nhp_sub, 95))

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.imshow(mat, interpolation="nearest", cmap="turbo", vmin=vmin, vmax=vmax)
    annotate_heatmap(ax2, mat)
    _style_matrix_axes(ax2, spaces)
    ax2.set_title(
        "Surface Transform Error Matrix (NHP-Scaled View; fsLR clipped)",
        fontweight="bold",
        color="#0044AA",
        pad=10,
    )
    cbar2 = fig2.colorbar(im2, ax=ax2)
    cbar2.set_label("Median absolute signed-distance error", rotation=90, labelpad=12)
    fig2.tight_layout()
    nhp_caption = (
        "Figure 3. Heatmap of pairwise surface-to-surface transform error "
        "between atlas spaces. Each matrix entry represents the median "
        "vertex-wise absolute signed-distance error after resampling one "
        "midthickness surface onto another using sphere-based barycentric "
        "registration. Color scaling is clipped to the 95th percentile of "
        "non-human primate (NHP) values to improve visualization of "
        "interspecies differences."
    )
    fig2.subplots_adjust(left=0.10, bottom=0.30)
    fig2.text(
        0.5, 0.02, nhp_caption, ha="center", fontsize=9, fontstyle="italic", wrap=True
    )
    nhp_path = output_dir / "surface_transform_matrix_nhp_scaled.png"
    fig2.savefig(nhp_path, dpi=200)
    plt.close(fig2)
    logger.info("Saved NHP-scaled heatmap: %s", nhp_path)


def _plot_error_histogram(result: SurfaceMatrixResult, output_dir: Path) -> None:
    """Write the vertex-wise transform-error histogram."""
    all_errors = result.all_errors

    if all_errors.size == 0:
        logger.warning("No vertex errors available; skipping the histogram.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    max_val = float(np.nanmax(all_errors))
    ax.hist(all_errors, bins=200)
    ax.set_xlim(0, max_val)
    ax.set_title(
        "Vertex-wise Surface Transform Error Distribution",
        fontweight="bold",
        color="#0044AA",
        pad=10,
    )
    ax.set_xlabel("Absolute signed distance error", labelpad=12)
    ax.set_ylabel("Vertex count", labelpad=12)

    fig4_caption = (
        "Figure 4. Distribution of vertex-wise absolute signed-distance errors"
        " across all pairwise surface transformations."
    )
    fig.subplots_adjust(left=0.15, bottom=0.30)
    fig.text(
        0.5, 0.02, fig4_caption, ha="center", fontsize=9, fontstyle="italic", wrap=True
    )
    hist_path = output_dir / "surface_transform_histogram.png"
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
    logger.info("Saved global histogram: %s", hist_path)


# -------------------------------------------------------------------------
# Test suite
# -------------------------------------------------------------------------


class TestSurfaceTransformMatrix:
    """Pairwise surface-transform error matrix regression.

    One class-scoped run computes the full directed matrix for every ordered
    pair of valid atlas spaces (a single hemisphere) and records this run's
    artifacts (matrix CSV, per-pair error GIFTIs, summary, and the inline
    heatmap / histogram plots). The suite is a pure producer: it reads no
    previous baseline and makes no cross-run comparison — those are CI
    concerns. The test methods assert the run's outcomes only; writing the
    artifacts is part of the run flow, not an object of the assertions.
    """

    @pytest.fixture(scope="class")
    def matrix_run(
        self, graph: NeuromapsGraph, tmp_path_factory: pytest.TempPathFactory
    ) -> SurfaceMatrixResult:
        """Compute the matrix once; write this run's artifacts and plots."""
        output_dir = resolve_artifact_dir(
            tmp_path_factory.getbasetemp() / "surface_matrix_outputs",
            env_var="NEUROMAPS_SURF_MATRIX_OUTPUT_DIR",
        )

        logger.info("Surface matrix outputs will be written to: %s", output_dir)

        result = run_surface_matrix(graph, HEMISPHERE, output_dir=output_dir)

        _plot_matrix_heatmaps(result, output_dir)
        _plot_error_histogram(result, output_dir)

        logger.info(
            "Generate plots with: uv run --group tests "
            "scripts/plot_surf_matrix.py --run-dir %s",
            output_dir,
        )

        return result

    def test_matrix_executed(self, matrix_run: SurfaceMatrixResult) -> None:
        """At least one off-diagonal pair produced a finite error."""
        assert len(matrix_run.spaces) >= 2, (
            "Fewer than two valid spaces — cannot form a pairwise matrix."
        )

        matrix = matrix_run.matrix.to_numpy()
        off_diag = matrix[~np.eye(len(matrix_run.spaces), dtype=bool)]

        assert np.isfinite(off_diag).any(), (
            "No finite off-diagonal surface transform error was computed."
        )

    def test_matrix_well_formed(self, matrix_run: SurfaceMatrixResult) -> None:
        """Matrix is square with finite off-diagonals and consistent derived forms."""
        spaces = matrix_run.spaces
        matrix = matrix_run.matrix

        assert list(matrix.index) == spaces
        assert list(matrix.columns) == spaces

        n = len(spaces)
        off_diag = matrix.to_numpy()[~np.eye(n, dtype=bool)]

        assert np.isfinite(off_diag).all(), (
            "Non-finite off-diagonal matrix entry — a transform failed to "
            "produce a usable error."
        )

        # The derived forms must match the matrix and their defining symmetry.
        base = matrix.to_numpy()
        asymmetry = matrix_run.asymmetry.to_numpy()
        symmetric = matrix_run.symmetric.to_numpy()

        np.testing.assert_allclose(asymmetry, base - base.T)
        np.testing.assert_allclose(symmetric, (base + base.T) / 2.0)
        np.testing.assert_allclose(asymmetry, -asymmetry.T)
        np.testing.assert_allclose(symmetric, symmetric.T)
