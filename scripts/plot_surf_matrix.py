"""Render surface-transform-matrix plots from a regression run's artifacts.

Reads the matrix CSV and the per-pair error GIFTIs written by
``tests/regression/test_surf_matrix.py`` and writes three figures into the
run directory:

* ``surface_transform_matrix_full.png`` — the directed transform-error
  heatmap (rows = source space, columns = target space);
* ``surface_transform_matrix_nhp_scaled.png`` — the same heatmap with the
  color scale clipped to the NHP value range so interspecies (fsLR) cells do
  not dominate the colormap;
* ``surface_transform_histogram.png`` — the distribution of vertex-wise
  signed-distance errors across every ordered pair.

The script is a post-hoc consumer of a single finished run's artifacts: it
reads no previous baseline and makes no cross-run comparison.

Run with:

    uv run scripts/plot_surf_matrix.py --run-dir <dir> [--output-dir <dir>]
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.10.7",
#     "nibabel>=5.2.1",
#     "numpy>=2.4.6",
#     "pandas>=3.0.3",
# ]
# ///

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def annotate_heatmap(ax: plt.Axes, mat: np.ndarray) -> None:
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


def _style_matrix_axes(ax: plt.Axes, spaces: list[str]) -> None:
    """Apply the shared axis configuration to a matrix heatmap."""
    n = len(spaces)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Target template space", labelpad=12)
    ax.set_ylabel("Source template space", labelpad=12)
    ax.set_xticklabels(spaces, rotation=45, ha="right")
    ax.set_yticklabels(spaces)


def _plot_matrix_heatmaps(matrix: pd.DataFrame, output_dir: Path) -> None:
    """Write the full-scale and NHP-scaled transform-error heatmaps."""
    mat = matrix.to_numpy()
    spaces = list(matrix.index)
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
        "Heatmap of pairwise surface-to-surface transform error "
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
    nhp_sub = matrix.loc[nhp_spaces, nhp_spaces].to_numpy() if nhp_spaces else mat
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
        "Heatmap of pairwise surface-to-surface transform error "
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


def _collect_vertex_errors(errors_dir: Path) -> np.ndarray:
    """Load and concatenate every per-pair vertex-error GIFTI under ``errors_dir``."""
    if not errors_dir.is_dir():
        logger.warning("No errors directory at %s; skipping the histogram.", errors_dir)
        return np.array([])

    files = sorted(errors_dir.glob("*.func.gii"))

    if not files:
        logger.warning("No error GIFTIs in %s; skipping the histogram.", errors_dir)
        return np.array([])

    return np.concatenate([np.abs(nib.load(path).darrays[0].data) for path in files])


def _plot_error_histogram(all_errors: np.ndarray, output_dir: Path) -> None:
    """Write the vertex-wise transform-error histogram."""
    if all_errors.size == 0:
        logger.warning("No vertex errors available; skipping the histogram.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    max_val = float(np.nanmax(all_errors))
    ax.hist(all_errors, bins=50)
    ax.set_xlim(0, max_val)
    ax.set_title(
        "Vertex-wise Surface Transform Error Distribution",
        fontweight="bold",
        color="#0044AA",
        pad=10,
    )
    ax.set_xlabel("Absolute signed distance error [mm]", labelpad=12)
    ax.set_ylabel("Vertex count", labelpad=12)

    fig4_caption = (
        "Distribution of vertex-wise absolute signed-distance errors "
        "across all pairwise surface transformations."
    )
    fig.subplots_adjust(left=0.15, bottom=0.30)
    fig.text(
        0.5, 0.02, fig4_caption, ha="center", fontsize=9, fontstyle="italic", wrap=True
    )
    hist_path = output_dir / "surface_transform_histogram.png"
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
    logger.info("Saved global histogram: %s", hist_path)


def main() -> int:
    """Parse arguments, load the run artifacts, and write the plots."""
    parser = argparse.ArgumentParser(
        description="Render surface-transform-matrix plots from a regression run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Directory holding a finished regression run's artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the PNGs into (default: --run-dir).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dir = args.run_dir

    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    csv_path = run_dir / "surface_transform_matrix.csv"

    if not csv_path.is_file():
        raise SystemExit(f"Matrix CSV not found: {csv_path}")

    output_dir = args.output_dir if args.output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(csv_path, index_col=0)

    _plot_matrix_heatmaps(matrix, output_dir)
    _plot_error_histogram(_collect_vertex_errors(run_dir / "errors"), output_dir)

    logger.info("Wrote surface-transform-matrix plots to %s", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
