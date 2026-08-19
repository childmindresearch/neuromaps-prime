"""Cycle regression test on the real NeuromapsPrime graph.

This branch adds end-to-end cycle regression testing on the
real Neuromaps graph to validate transform roundtrip quality across multi-hop paths.

Resulting fles are written to a run-specific directory:

    tests/regression/cycle_outputs_<datetime>/

Each run may contain:

* CSV summaries
* TXT summaries
* per-path transformation manifests
* intermediate metric files
* surface visualizations
* optional wb_command/niwrap/styx logs

Area surfaces are attempted in this order:

1. midthickness
2. pial
3. white

Run with::

    pytest tests/regression/test_cycle.py -v -s
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from matplotlib_surface_plotting import plot_surf
from nibabel.gifti import GiftiDataArray, GiftiImage
from tests.cycle import (
    RoundtripResult,
    _path_token,
    find_return_paths,
    roundtrip_metric,
    score_roundtrip,
)

from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Run configuration
# -------------------------------------------------------------------------

_RUN_SUFFIX = datetime.now().strftime("%Y%m%d_%H%M")

OUTPUT_DIR = Path(__file__).resolve().parent / f"cycle_outputs_{_RUN_SUFFIX}"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Set to a specific space to test only that origin, or use "all" to test
# every configured origin.
ORIGIN: str = "all"

HEMISPHERES = (
    "left",
    "right",
)

MAX_CYCLE_LENGTH = 4
MAX_PATHS: int | None = None

# Minimum acceptable mean Pearson correlation for each regression run.
# Values should be updated deliberately.  Last updated 08/18 (TR).
MIN_MEAN_PEARSON: dict[tuple[str, str], float] = {
    ("Yerkes19", "left"): 0.572255,
    ("Yerkes19", "right"): 0.553471,
    ("CIVETNMT", "left"): 0.501674,
    ("CIVETNMT", "right"): 0.5172,
    ("MEBRAINS", "left"): 0.561245,
    ("MEBRAINS", "right"): 0.555567,
    ("D99", "left"): 0.610269,
    ("D99", "right"): 0.567145,
    ("NMT2Sym", "left"): 0.541054,
    ("NMT2Sym", "right"): 0.505617,
    ("fsaverage", "left"): 0.998140,
    ("fsaverage", "right"): 0.998340,
    ("fsLR", "left"): 0.706283,
    ("fsLR", "right"): 0.606387,
    ("NCBR", "left"): 0.998746,
    ("NCBR", "right"): 0.998746,
    ("all", "left"): 0.564602,
    ("all", "right"): 0.545228,
}

LOG_COMMANDS = True


def _get_min_mean_pearson(
    origin: str,
    hemisphere: str,
) -> float:
    """Return the configured regression threshold for an origin/hemisphere."""
    threshold_origin = "all" if ORIGIN == "all" else origin

    return MIN_MEAN_PEARSON[(threshold_origin, hemisphere)]


# -------------------------------------------------------------------------
# Command logging
# -------------------------------------------------------------------------


class CommandLogHandler(logging.Handler):
    """Capture niwrap/Styx records containing command information."""

    def __init__(self, log_file: Path) -> None:
        """Initialize a command log handler."""
        super().__init__()
        self.log_file = log_file
        self.commands: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Capture records that appear to contain command information."""
        message = record.getMessage()

        if "command" in message.lower() or "wb_command" in record.name.lower():
            self.commands.append(self.format(record))

    def save(self) -> None:
        """Write captured command records to disk."""
        if not self.commands:
            return

        self.log_file.write_text(
            "# wb_command / niwrap / Styx calls used in this cycle\n"
            "# These records are intended for debugging and manual inspection.\n\n"
            + "\n".join(self.commands)
            + "\n",
            encoding="utf-8",
        )


# -------------------------------------------------------------------------
# Surface helpers
# -------------------------------------------------------------------------


def _load_surface_coords(
    surface_file: Path,
) -> np.ndarray:
    """Load surface coordinates as ``(n_vertices, 3)``."""
    image = nib.load(str(surface_file))

    for darray in image.darrays:
        data = np.asarray(darray.data)

        if data.ndim == 2 and data.shape[1] == 3:
            return np.asarray(
                data,
                dtype=np.float64,
            )

    raise ValueError(f"No pointset coordinates found in {surface_file}.")


def _load_surface_topology(
    surface_file: Path,
) -> np.ndarray:
    """Load triangle topology from a surface GIFTI."""
    image = nib.load(str(surface_file))

    for darray in image.darrays:
        if str(darray.intent) in (
            "NIFTI_INTENT_TRIANGLE",
            "1009",
        ):
            return np.asarray(
                darray.data,
                dtype=np.int32,
            )

    for darray in image.darrays:
        data = np.asarray(darray.data)

        if (
            data.ndim == 2
            and data.shape[1] == 3
            and np.issubdtype(
                data.dtype,
                np.integer,
            )
        ):
            return data.astype(np.int32)

    raise ValueError(f"No triangle topology found in {surface_file}.")


def _extract_surface_mesh(
    surface_file: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract coordinates and triangles from a surface."""
    return (
        _load_surface_coords(surface_file),
        _load_surface_topology(surface_file),
    )


# -------------------------------------------------------------------------
# Metric generation
# -------------------------------------------------------------------------


def _write_metric(
    metric_file: Path,
    values: np.ndarray,
) -> Path:
    """Write one scalar value per vertex as a GIFTI metric."""
    image = GiftiImage(
        darrays=[
            GiftiDataArray(
                np.asarray(
                    values,
                    dtype=np.float32,
                ),
                intent="NIFTI_INTENT_NONE",
            )
        ]
    )

    metric_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        image,
        str(metric_file),
    )

    return metric_file


def _make_xyz_product_metric(
    graph: NeuromapsGraph,
    origin: str,
    density: str,
    hemisphere: str,
    output_dir: Path,
) -> Path:
    """Create a deterministic synthetic metric from sphere coordinates."""
    sphere = graph.fetch_surface_atlas(
        space=origin,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    )

    if sphere is None:
        raise FileNotFoundError(
            f"No sphere atlas for {origin} at {density} ({hemisphere})."
        )

    coords = _load_surface_coords(Path(sphere.fetch()))

    values = np.prod(
        coords,
        axis=1,
    )

    metric_file = output_dir / f"metric_{origin}_{density}_{hemisphere}.func.gii"

    return _write_metric(
        metric_file,
        values,
    )


def _load_metric_values(
    metric_file: Path,
) -> np.ndarray:
    """Load scalar values from a metric GIFTI."""
    return np.asarray(
        nib.load(str(metric_file)).darrays[0].data,
        dtype=np.float64,
    )


# -------------------------------------------------------------------------
# Surface matching for visualization
# -------------------------------------------------------------------------


def _find_matching_surface(
    graph: NeuromapsGraph,
    space: str,
    hemisphere: str,
    resource_type: str,
    n_vertices: int,
) -> tuple[object, str] | None:
    """Find a surface whose vertex count matches the metric."""
    atlases = graph.utils.cache.get_surface_atlases(
        space=space,
        hemisphere=hemisphere,
        resource_type=resource_type,
    )

    for atlas in atlases:
        try:
            coords = _load_surface_coords(Path(atlas.fetch()))
        except (
            ValueError,
            FileNotFoundError,
            OSError,
        ):
            continue

        if coords.shape[0] == n_vertices:
            return atlas, atlas.density

    return None


# -------------------------------------------------------------------------
# Surface plotting
# -------------------------------------------------------------------------


def _plot_single_surface(
    graph: NeuromapsGraph,
    space: str,
    metric_values: np.ndarray,
    hemisphere: str,
    resource_type: str,
    vmin: float,
    vmax: float,
    plot_dir: Path,
    path_token: str,
    path_label: str,
    hop_index: int,
    pearson_r: float,
) -> None:
    """Plot a metric on a surface matching its vertex count."""
    match = _find_matching_surface(
        graph,
        space,
        hemisphere,
        resource_type,
        metric_values.shape[0],
    )

    if match is None:
        logger.warning(
            "No %s surface for %s (%s) matching %d vertices; skipping plot.",
            resource_type,
            space,
            hemisphere,
            metric_values.shape[0],
        )
        return

    surface_atlas, density = match

    try:
        coords, faces = _extract_surface_mesh(Path(surface_atlas.fetch()))
    except (
        ValueError,
        FileNotFoundError,
        OSError,
    ) as exc:
        logger.warning(
            "Could not load %s mesh for %s (%s): %s",
            resource_type,
            space,
            density,
            exc,
        )
        return

    output_path = plot_dir / (
        f"{path_token}_node{hop_index:02d}_{space}_{resource_type}.png"
    )

    try:
        plot_surf(
            coords,
            faces,
            metric_values,
            rotate=[270, 0],
            filename=str(output_path),
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            title=(f"{path_label}\n{space} | r={pearson_r:.5f}"),
        )

        logger.info(
            "Saved surface plot: %s",
            output_path,
        )

    except Exception as exc:
        logger.warning(
            "plot_surf failed for %s %s (%s): %s",
            space,
            resource_type,
            density,
            exc,
        )

    finally:
        plt.close("all")


def _plot_cycle_cortical_surfaces(
    graph: NeuromapsGraph,
    path: tuple[str, ...],
    metrics_by_hop: list[tuple[str, np.ndarray]],
    hemisphere: str,
    pearson_r: float,
    plot_dir: Path,
) -> None:
    """Plot midthickness and sphere views for every cycle node."""
    if not metrics_by_hop:
        return

    path_token = _path_token(path)
    path_label = " -> ".join(path)

    for hop_index, (
        space,
        metric_values,
    ) in enumerate(metrics_by_hop):
        finite = np.isfinite(metric_values)

        if np.any(finite):
            vmin, vmax = np.percentile(
                metric_values[finite],
                [2, 98],
            )
        else:
            vmin, vmax = 0.0, 1.0

        _plot_single_surface(
            graph=graph,
            space=space,
            metric_values=metric_values,
            hemisphere=hemisphere,
            resource_type="midthickness",
            vmin=float(vmin),
            vmax=float(vmax),
            plot_dir=plot_dir,
            path_token=path_token,
            path_label=path_label,
            hop_index=hop_index,
            pearson_r=pearson_r,
        )

        _plot_single_surface(
            graph=graph,
            space=space,
            metric_values=metric_values,
            hemisphere=hemisphere,
            resource_type="sphere",
            vmin=float(vmin),
            vmax=float(vmax),
            plot_dir=plot_dir,
            path_token=path_token,
            path_label=path_label,
            hop_index=hop_index,
            pearson_r=pearson_r,
        )


# -------------------------------------------------------------------------
# Transform manifests
# -------------------------------------------------------------------------


def _write_transform_manifest(
    roundtrip: RoundtripResult,
    output_file: Path,
) -> None:
    """Write a per-path manifest describing the executed transformations."""
    lines = ["source_space,target_space,area_resource,output_file"]

    lines.extend(
        ",".join(
            [
                hop.source,
                hop.target,
                hop.area_resource,
                str(hop.output_file),
            ]
        )
        for hop in roundtrip.hops
    )

    output_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# -------------------------------------------------------------------------
# Main regression test
# -------------------------------------------------------------------------


def _log_progress(
    *,
    completed: int,
    total: int,
    origin: str,
    hemisphere: str,
    path: tuple[str, ...],
) -> None:
    """Log overall cycle regression progress."""
    percentage = 100.0 * completed / total if total else 100.0

    logger.info(
        "CYCLE PROGRESS: %d/%d (%.1f%%) | origin=%s | hemisphere=%s | path=%s",
        completed,
        total,
        percentage,
        origin,
        hemisphere,
        " -> ".join(path),
    )


def test_cycle_roundtrip() -> None:
    """Round-trip synthetic metrics through real transformation cycles."""
    graph = NeuromapsGraph()

    origins = sorted(graph.nodes) if ORIGIN == "all" else [ORIGIN]

    total_usable_paths = 0

    for origin in origins:
        for hemisphere in HEMISPHERES:
            logger.info(
                "\n=== CYCLE ROUND-TRIP TEST: %s (%s) ===",
                origin,
                hemisphere,
            )

            result = _run_origin_hemisphere(
                graph=graph,
                origin=origin,
                hemisphere=hemisphere,
            )

            total_usable_paths += result

    plot_run_summaries(
        run_dir=OUTPUT_DIR,
    )

    if ORIGIN == "all":
        _save_all_summary(OUTPUT_DIR)

    assert total_usable_paths > 0, (
        "No executable surface transformation cycles were found "
        "for any configured origin/hemisphere."
    )


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------


def _plot_cycle_summary(
    run_dir: Path,
    origin: str,
) -> None:
    """Create one two-panel cycle summary plot for an origin space.

    The left panel shows left-hemisphere cycle correlations and the right
    panel shows right-hemisphere cycle correlations.
    """
    frames: dict[str, pd.DataFrame] = {}

    for hemisphere in ("left", "right"):
        csv_path = run_dir / f"cycle_{origin}_{hemisphere}.csv"

        if not csv_path.exists():
            logger.warning(
                "No %s CSV found for %s: %s",
                hemisphere,
                origin,
                csv_path,
            )
            continue

        frame = pd.read_csv(csv_path)

        if frame.empty:
            continue

        frames[hemisphere] = frame

    if not frames:
        logger.warning(
            "No cycle CSVs found for origin %s.",
            origin,
        )
        return

    paths = sorted(
        {path for frame in frames.values() for path in frame["path"]},
        key=lambda path: (
            len(str(path).split(" -> ")) - 1,
            str(path),
        ),
    )

    paths = list(reversed(paths))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(
            18,
            max(8, 0.30 * len(paths)),
        ),
        sharey=True,
    )

    for ax, hemisphere in zip(
        axes,
        ("left", "right"),
        strict=True,
    ):
        frame = frames.get(hemisphere)

        if frame is None:
            ax.text(
                0.5,
                0.5,
                "No results",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(hemisphere.capitalize())
            ax.set_xlabel("Pearson r")
            continue

        values = frame.set_index("path")["pearson_r"].reindex(paths)

        y_positions = np.arange(len(paths))

        ax.barh(
            y_positions,
            values,
        )

        ax.set_yticks(y_positions)

        if ax is axes[0]:
            ax.set_yticklabels(
                paths,
                fontsize=8,
            )
            ax.set_ylabel("Transformation path")
        else:
            ax.tick_params(
                axis="y",
                labelleft=False,
            )

        ax.set_xlabel("Pearson r")

        ax.set_title(hemisphere.capitalize())

        ax.axvline(
            0.0,
            linewidth=1,
        )

        ax.grid(
            axis="x",
            alpha=0.3,
        )

        finite_values = values.dropna()

        if not finite_values.empty:
            lower = min(
                -0.05,
                float(finite_values.min()) - 0.05,
            )
        else:
            lower = -0.05

        ax.set_xlim(
            lower,
            1.0,
        )

    fig.suptitle(
        f"{origin}\nSurface transformation cycle round-trip accuracy",
        fontsize=16,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_file = run_dir / f"cycle_{origin}_summary.png"

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Saved combined cycle summary plot: %s",
        output_file,
    )


def _save_all_summary(run_dir: Path) -> None:
    """Calculate overall left/right Pearson r across all origin spaces."""
    summaries: list[str] = []

    for hemisphere in HEMISPHERES:
        csv_files = sorted(run_dir.glob(f"cycle_*_{hemisphere}.csv"))

        frames = [
            pd.read_csv(csv_file)
            for csv_file in csv_files
            if not pd.read_csv(csv_file).empty
        ]

        if not frames:
            logger.warning(
                "No cycle results found for all-space %s summary.",
                hemisphere,
            )
            continue

        combined = pd.concat(frames, ignore_index=True)

        mean_r = float(combined["pearson_r"].mean())

        summaries.append(
            f"All spaces ({hemisphere}):\n"
            f"  Total executable cycles: {len(combined)}\n"
            f"  Mean Pearson r: {mean_r:.6f}\n"
        )

        logger.info(
            "ALL SPACES (%s): %d cycles, mean Pearson r = %.6f",
            hemisphere,
            len(combined),
            mean_r,
        )

    if not summaries:
        return

    output_file = run_dir / "cycle_all_summary.txt"

    lines = [
        "Cycle test results — all origin spaces",
        "=" * 60,
        "",
        *summaries,
    ]

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info("Saved all-space summary: %s", output_file)


def plot_run_summaries(
    run_dir: str | Path,
) -> None:
    """Create combined left/right plots for every origin in a run."""
    run_dir = Path(run_dir)

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    csv_files = sorted(run_dir.glob("cycle_*_left.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No cycle CSV files found in {run_dir}.")

    origins = []

    for csv_file in csv_files:
        name = csv_file.name

        if not name.startswith("cycle_") or not name.endswith("_left.csv"):
            continue

        origin = name[len("cycle_") : -len("_left.csv")]

        origins.append(origin)

    for origin in origins:
        _plot_cycle_summary(
            run_dir=run_dir,
            origin=origin,
        )


def _save_cycle_results(
    origin: str,
    hemisphere: str,
    rows: list[dict[str, object]],
    plot_dir: Path,
    command_handler: CommandLogHandler | None,
    min_mean_pearson: float,
) -> int:
    """Save CSV/TXT cycle results and validate regression output."""
    if not rows:
        logger.warning(
            "No executable paths for %s (%s).",
            origin,
            hemisphere,
        )

        if command_handler is not None:
            command_handler.save()

        return 0

    frame = pd.DataFrame(rows).sort_values(
        ["pearson_r", "path"],
        ascending=[False, True],
    )

    csv_path = OUTPUT_DIR / f"cycle_{origin}_{hemisphere}.csv"

    frame.to_csv(
        csv_path,
        index=False,
    )

    path_width = max(
        len("Transformation path"),
        max(len(str(row["path"])) for row in rows),
    )

    header = (
        f"{'Transformation path':<{path_width}}  "
        f"{'Hops':>4}  "
        f"{'Pearson r':>10}  "
        f"{'Max |delta|':>14}"
    )

    separator = "-" * len(header)

    lines = [
        (f"Cycle test results — origin: {origin}, hemisphere: {hemisphere}"),
        separator,
        header,
        separator,
    ]

    for _, row in frame.iterrows():
        lines.append(
            f"{row['path']!s:<{path_width}}  "
            f"{int(row['n_hops']):>4}  "
            f"{float(row['pearson_r']):>10.6f}  "
            f"{float(row['max_abs_diff']):>14.3e}"
        )

    mean_r = float(frame["pearson_r"].mean())

    lines.extend(
        [
            separator,
            f"Total cycles: {len(rows)}",
            f"Mean Pearson r: {mean_r:.6f}",
            f"Minimum required mean Pearson r: {min_mean_pearson:.6f}",
        ]
    )

    txt_path = OUTPUT_DIR / f"cycle_{origin}_{hemisphere}.txt"

    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    logger.info("Saved CSV: %s", csv_path)
    logger.info("Saved TXT summary: %s", txt_path)
    logger.info("Saved path results: %s", plot_dir)

    if command_handler is not None:
        command_handler.save()
        logger.info(
            "Saved command log: %s",
            command_handler.log_file,
        )

    if min_mean_pearson is None:
        pytest.fail(
            f"No baseline threshold configured for "
            f"origin={origin}, hemisphere={hemisphere}. "
            f"Observed mean r={mean_r:.6f}."
        )

    assert mean_r >= min_mean_pearson - 1e-6, (
        "Average round-trip correlation regressed: "
        f"origin={origin}, "
        f"hemisphere={hemisphere}, "
        f"mean r={mean_r:.6f}, "
        f"threshold={min_mean_pearson:.6f}. "
        f"Inspect outputs in {OUTPUT_DIR}."
    )

    return len(rows)


def _run_cycle_path(
    graph: NeuromapsGraph,
    metric_file: Path,
    original_metric: np.ndarray,
    path: tuple[str, ...],
    hemisphere: str,
    path_workdir: Path,
    plot_dir: Path,
) -> dict[str, object] | None:
    """Execute, score, manifest, and plot one transformation cycle."""
    path_label = " -> ".join(path)
    path_token = _path_token(path)

    try:
        roundtrip = roundtrip_metric(
            graph=graph,
            metric_file=metric_file,
            path=path,
            hemisphere=hemisphere,
            workdir=path_workdir,
            density=None,
            add_edge=False,
        )

        pearson_r, max_abs_diff = score_roundtrip(
            metric_file,
            roundtrip.final_metric,
        )

    except (
        RuntimeError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        logger.warning(
            "Skipping non-executable cycle %s (%s): %s",
            path_label,
            hemisphere,
            exc,
        )
        return None

    manifest_file = plot_dir / f"{path_token}_transforms.csv"

    try:
        _write_transform_manifest(
            roundtrip=roundtrip,
            output_file=manifest_file,
        )
    except OSError as exc:
        logger.warning(
            "Could not write transform manifest for %s: %s",
            path_label,
            exc,
        )

    try:
        metrics_by_hop = [(path[0], original_metric)] + [
            (
                hop.target,
                hop.metric_values,
            )
            for hop in roundtrip.hops
        ]

        _plot_cycle_cortical_surfaces(
            graph=graph,
            path=path,
            metrics_by_hop=metrics_by_hop,
            hemisphere=hemisphere,
            pearson_r=pearson_r,
            plot_dir=plot_dir,
        )

    except Exception as exc:
        logger.warning(
            "Failed to create surface plots for %s: %s",
            path_label,
            exc,
        )

    logger.info(
        "cycle %s: r=%.6f max|delta|=%.3e",
        path_label,
        pearson_r,
        max_abs_diff,
    )

    return {
        "path": path_label,
        "n_hops": len(path) - 1,
        "pearson_r": pearson_r,
        "max_abs_diff": max_abs_diff,
    }


def _run_origin_hemisphere(
    graph: NeuromapsGraph,
    origin: str,
    hemisphere: str,
) -> int:
    """Run all cycles for one origin and hemisphere."""
    work_dir = OUTPUT_DIR / f"work_{origin}_{hemisphere}"

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    min_mean_pearson = _get_min_mean_pearson(
        origin,
        hemisphere,
    )

    logger.info(
        "Regression threshold for %s (%s): mean Pearson r >= %.6f",
        origin,
        hemisphere,
        min_mean_pearson,
    )

    command_handler = None

    if LOG_COMMANDS:
        command_handler = CommandLogHandler(work_dir / "wb_commands.log")

        for logger_name in ("niwrap", "styx"):
            command_logger = logging.getLogger(logger_name)
            command_logger.addHandler(command_handler)
            command_logger.setLevel(logging.DEBUG)

    try:
        density = graph.find_highest_density(origin)

        metric_file = _make_xyz_product_metric(
            graph=graph,
            origin=origin,
            density=density,
            hemisphere=hemisphere,
            output_dir=work_dir,
        )

    except (
        AssertionError,
        FileNotFoundError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        logger.warning(
            "Skipping %s (%s): could not seed origin metric: %s",
            origin,
            hemisphere,
            exc,
        )
        return 0

    original_metric = _load_metric_values(metric_file)

    paths = find_return_paths(
        graph,
        origin,
        max_length=MAX_CYCLE_LENGTH,
        allow_revisits=True,
        max_paths=MAX_PATHS,
    )

    logger.info(
        "Found %d return paths from %s",
        len(paths),
        origin,
    )

    if not paths:
        logger.warning(
            "No return paths from %s; skipping %s.",
            origin,
            hemisphere,
        )
        return 0

    plot_dir = OUTPUT_DIR / f"cycle_{origin}_{hemisphere}_plots"

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []

    total_paths = len(paths)

    logger.info(
        "Starting %d cycles for %s (%s)",
        total_paths,
        origin,
        hemisphere,
    )

    for completed, path in enumerate(paths, start=1):
        _log_progress(
            completed=completed,
            total=total_paths,
            origin=origin,
            hemisphere=hemisphere,
            path=path,
        )

        row = _run_cycle_path(
            graph=graph,
            metric_file=metric_file,
            original_metric=original_metric,
            path=path,
            hemisphere=hemisphere,
            path_workdir=work_dir / f"path_{_path_token(path)}",
            plot_dir=plot_dir,
        )

        if row is not None:
            rows.append(row)

    return _save_cycle_results(
        origin=origin,
        hemisphere=hemisphere,
        rows=rows,
        plot_dir=plot_dir,
        command_handler=command_handler,
        min_mean_pearson=min_mean_pearson,
    )
