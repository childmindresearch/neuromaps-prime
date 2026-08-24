"""Cycle regression test on the real Neuromaps-PRIME graph.

Adds end-to-end cycle regression testing on the real Neuromaps graph
to validate transform roundtrip quality across multi-hop paths.

Resulting files are written to a run-specific directory:

    tests/regression/cycle_outputs_<datetime>/

Each run may contain:

* CSV summaries
* TXT summaries
* intermediate metric files
* surface visualizations

Area surfaces are attempted in this order:

1. midthickness
2. pial
3. white

Run with:

    pytest tests/regression/test_cycle.py -v -s
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tests.cycle import (
    Hemisphere,
    find_return_paths,
    load_metric,
    path_token,
    roundtrip_metric,
    score_roundtrip,
)
from tests.regression.utils import (
    load_latest_cycle_baseline,
    make_sphere,
    plot_cycle_cortical_surfaces,
    save_cycle_baseline,
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

# Always test every configured origin space.

HEMISPHERES = (
    "left",
    "right",
)

MAX_CYCLE_LENGTH: Final = 4
MAX_PATHS: Final[int | None] = None
ALLOWED_REGRESSION: Final = 1e-4

# -------------------------------------------------------------------------
# Main regression test
# -------------------------------------------------------------------------


def _log_progress(
    *,
    completed: int,
    total: int,
    origin: str,
    hemisphere: Hemisphere,
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


def _collect_cycle_values(
    run_dir: Path,
    origins: list[str],
) -> dict[tuple[str, str], float]:
    """Collect mean Pearson r values for each origin and hemisphere.

    The ``all`` value is calculated from every executable cycle across all
    origin spaces, rather than from the mean of the per-origin means.
    """
    values: dict[tuple[str, str], float] = {}

    for origin in origins:
        for hemisphere in HEMISPHERES:
            csv_path = run_dir / f"cycle_{origin}_{hemisphere}.csv"

            if not csv_path.exists():
                continue

            frame = pd.read_csv(csv_path)

            if frame.empty:
                continue

            values[(origin, hemisphere)] = float(
                frame["pearson_r"].mean(),
            )

    for hemisphere in HEMISPHERES:
        frames: list[pd.DataFrame] = []

        for origin in origins:
            csv_path = run_dir / f"cycle_{origin}_{hemisphere}.csv"

            if not csv_path.exists():
                continue

            frame = pd.read_csv(csv_path)

            if not frame.empty:
                frames.append(frame)

        if not frames:
            continue

        combined = pd.concat(
            frames,
            ignore_index=True,
        )

        values[("all", hemisphere)] = float(
            combined["pearson_r"].mean(),
        )

    return values


def test_cycle_roundtrip() -> None:
    """Round-trip synthetic metrics through real transformation cycles."""
    graph = NeuromapsGraph()
    baseline_dir = Path(__file__).resolve().parent

    baseline = load_latest_cycle_baseline(
        baseline_dir=baseline_dir,
    )

    origins = sorted(graph.nodes)
    total_usable_paths = 0

    for origin in origins:
        for hemisphere in HEMISPHERES:
            logger.info(
                "\n=== CYCLE ROUND-TRIP TEST: %s (%s) ===",
                origin,
                hemisphere,
            )

            total_usable_paths += _run_origin_hemisphere(
                graph=graph,
                origin=origin,
                hemisphere=hemisphere,
            )

    assert total_usable_paths > 0, (
        "No executable surface transformation cycles were found "
        "for any configured origin/hemisphere."
    )

    current_values = _collect_cycle_values(
        run_dir=OUTPUT_DIR,
        origins=origins,
    )

    current_all_values = _save_all_summary(
        run_dir=OUTPUT_DIR,
        baseline=baseline,
    )

    current_values.update(current_all_values)

    save_cycle_baseline(
        baseline_dir=baseline_dir,
        values=current_values,
        graph=graph,
    )

    plot_run_summaries(
        run_dir=OUTPUT_DIR,
        current_values=current_values,
        baseline=baseline,
        graph=graph,
    )

    logger.info(
        "NEW BASELINE VALUES: left=%.6f, right=%.6f",
        current_values[("all", "left")],
        current_values[("all", "right")],
    )


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------


def _plot_species_summary(
    run_dir: Path,
    origins: list[str],
) -> None:
    """Create one summary plot for all origin spaces.

    Each origin space is represented by a unique marker. Filled markers
    represent the left hemisphere and hollow markers represent the right
    hemisphere. Both hemispheres are plotted slightly offset horizontally
    so that their markers remain visually distinct.

    Args:
        run_dir: Directory containing per-origin cycle CSV files.
        origins: Origin spaces to include in the plot.
    """
    rows: list[dict[str, object]] = []

    for origin in origins:
        for hemisphere in HEMISPHERES:
            csv_path = run_dir / f"cycle_{origin}_{hemisphere}.csv"

            if not csv_path.exists():
                continue

            frame = pd.read_csv(csv_path)

            if frame.empty:
                continue

            rows.append(
                {
                    "origin": origin,
                    "hemisphere": hemisphere,
                    "pearson_r": float(frame["pearson_r"].mean()),
                }
            )

    if not rows:
        logger.warning(
            "No cycle results found for summary plot.",
        )
        return

    frame = pd.DataFrame(rows)

    unique_origins = [origin for origin in origins if origin in set(frame["origin"])]

    markers = (
        "o",
        "s",
        "^",
        "D",
        "P",
        "X",
        "*",
        "v",
        "<",
        ">",
        "p",
        "h",
        "8",
    )

    marker_map = {
        origin: markers[index % len(markers)]
        for index, origin in enumerate(unique_origins)
    }

    hemisphere_offsets = {
        "left": -0.10,
        "right": 0.10,
    }

    fig_width = max(12, 1.25 * len(unique_origins))

    fig, ax = plt.subplots(
        figsize=(fig_width, 7),
    )

    try:
        for x_position, origin in enumerate(unique_origins):
            marker = marker_map[origin]

            for hemisphere in HEMISPHERES:
                subset = frame[
                    (frame["origin"] == origin) & (frame["hemisphere"] == hemisphere)
                ]

                if subset.empty:
                    continue

                mean_r = float(
                    subset["pearson_r"].iloc[0],
                )

                ax.scatter(
                    x_position + hemisphere_offsets[hemisphere],
                    mean_r,
                    s=120,
                    marker=marker,
                    facecolors=("none" if hemisphere == "right" else None),
                    linewidths=1.5,
                    label=(f"{origin} ({'L' if hemisphere == 'left' else 'R'})"),
                    zorder=3,
                )

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_xticks(
            range(len(unique_origins)),
        )

        ax.set_xticklabels(
            unique_origins,
            rotation=45,
            ha="right",
        )

        ax.set_xlabel("Origin space")
        ax.set_ylabel("Mean Pearson r")

        ax.set_ylim(
            min(
                0.0,
                frame["pearson_r"].min() - 0.05,
            ),
            1.01,
        )

        ax.set_title(
            "Surface transformation cycle round-trip accuracy",
            fontsize=16,
        )

        ax.grid(
            axis="y",
            linestyle=":",
            alpha=0.5,
        )

        ax.legend(
            title="Origin / hemisphere",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        fig.tight_layout()

        output_file = run_dir / "cycle_summary.png"

        fig.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight",
        )

    finally:
        plt.close(fig)

    logger.info(
        "Saved cycle summary plot: %s",
        output_file,
    )


def plot_run_summaries(
    run_dir: str | Path,
    current_values: dict[tuple[str, str], float],
    baseline: dict[tuple[str, str], float],
) -> None:
    """Create summary plots for a completed cycle regression run."""
    run_dir = Path(run_dir)

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}",
        )

    csv_files = sorted(
        run_dir.glob("cycle_*_left.csv"),
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No cycle CSV files found in {run_dir}.",
        )

    origins: list[str] = []

    for csv_file in csv_files:
        name = csv_file.name

        if not name.startswith("cycle_") or not name.endswith("_left.csv"):
            continue

        origin = name[len("cycle_") : -len("_left.csv")]

        if origin == "all":
            continue

        origins.append(origin)

    origins = sorted(set(origins))

    _plot_species_summary(
        run_dir=run_dir,
        origins=origins,
    )

    _plot_baseline_comparison(
        run_dir=run_dir,
        current_values=current_values,
        baseline=baseline,
    )


def _collect_summary_rows(
    run_dir: Path,
) -> list[dict[str, object]]:
    """Collect mean Pearson r for each origin and hemisphere."""
    rows: list[dict[str, object]] = []

    for csv_file in sorted(run_dir.glob("cycle_*_*.csv")):
        stem = csv_file.stem

        if stem.startswith("cycle_all_"):
            continue

        parts = stem.removeprefix("cycle_").rsplit("_", 1)

        if len(parts) != 2:
            continue

        origin, hemisphere = parts

        if hemisphere not in HEMISPHERES:
            continue

        frame = pd.read_csv(csv_file)

        if frame.empty:
            continue

        rows.append(
            {
                "origin": origin,
                "hemisphere": hemisphere,
                "mean_pearson_r": float(frame["pearson_r"].mean()),
            }
        )

    return rows


def _calculate_all_values(
    run_dir: Path,
    baseline: dict[tuple[str, str], float],
) -> tuple[
    dict[tuple[str, str], float],
    list[dict[str, object]],
    list[str],
]:
    """Calculate all-space Pearson r values and validate regression."""
    current_all_values: dict[tuple[str, str], float] = {}
    summary_rows: list[dict[str, object]] = []
    summaries: list[str] = []

    for hemisphere in HEMISPHERES:
        frames: list[pd.DataFrame] = []

        for csv_file in sorted(
            run_dir.glob(f"cycle_*_{hemisphere}.csv"),
        ):
            if csv_file.stem.startswith("cycle_all_"):
                continue

            frame = pd.read_csv(csv_file)

            if not frame.empty:
                frames.append(frame)

        if not frames:
            logger.warning(
                "No cycle results found for all-space %s summary.",
                hemisphere,
            )
            continue

        combined = pd.concat(
            frames,
            ignore_index=True,
        )

        mean_r = float(combined["pearson_r"].mean())
        current_all_values[("all", hemisphere)] = mean_r

        summary_rows.append(
            {
                "origin": "all",
                "hemisphere": hemisphere,
                "mean_pearson_r": mean_r,
            }
        )

        baseline_r = baseline[("all", hemisphere)]
        difference = mean_r - baseline_r
        minimum_allowed = baseline_r - ALLOWED_REGRESSION

        summaries.append(
            f"All spaces ({hemisphere}):\n"
            f"  Total executable cycles: {len(combined)}\n"
            f"  Mean Pearson r: {mean_r:.6f}\n"
            f"  Baseline Pearson r: {baseline_r:.6f}\n"
            f"  Difference: {difference:+.6f}\n"
            f"  Minimum allowed Pearson r: {minimum_allowed:.6f}\n"
        )

        assert mean_r >= minimum_allowed, (
            "Average round-trip correlation regressed for all spaces: "
            f"hemisphere={hemisphere}, "
            f"current r={mean_r:.6f}, "
            f"baseline={baseline_r:.6f}, "
            f"difference={difference:+.6f}, "
            f"minimum allowed={minimum_allowed:.6f}, "
            f"allowed regression={ALLOWED_REGRESSION:.6f}. "
            f"Inspect outputs in {run_dir}."
        )

    return current_all_values, summary_rows, summaries


def _save_cycle_summary_csv(
    run_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    """Save per-origin and all-space Pearson r summary."""
    if not rows:
        return

    summary_frame = pd.DataFrame(
        rows,
        columns=[
            "origin",
            "hemisphere",
            "mean_pearson_r",
        ],
    )

    summary_frame.to_csv(
        run_dir / "cycle_summary.csv",
        index=False,
    )

    logger.info(
        "Saved cycle summary CSV: %s",
        run_dir / "cycle_summary.csv",
    )


def _save_all_summary_txt(
    run_dir: Path,
    summaries: list[str],
) -> None:
    """Save the all-space cycle summary text file."""
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

    logger.info(
        "Saved all-space summary: %s",
        output_file,
    )


def _log_final_regression_status(
    current_all_values: dict[tuple[str, str], float],
    baseline: dict[tuple[str, str], float],
) -> None:
    """Log final all-space regression status."""
    for hemisphere in HEMISPHERES:
        key = ("all", hemisphere)

        if key not in current_all_values:
            continue

        current_r = current_all_values[key]
        baseline_r = baseline[key]
        difference = current_r - baseline_r
        minimum_allowed = baseline_r - ALLOWED_REGRESSION

        logger.info(
            "FINAL ALL-SPACE MEAN — %s: current=%.6f, "
            "baseline=%.6f, difference=%+.6f, minimum allowed=%.6f",
            hemisphere,
            current_r,
            baseline_r,
            difference,
            minimum_allowed,
        )


def _save_all_summary(
    run_dir: Path,
    baseline: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """Calculate and save cycle Pearson r summaries."""
    summary_rows = _collect_summary_rows(run_dir)

    (
        current_all_values,
        all_summary_rows,
        summaries,
    ) = _calculate_all_values(
        run_dir=run_dir,
        baseline=baseline,
    )

    summary_rows.extend(all_summary_rows)

    _save_cycle_summary_csv(
        run_dir=run_dir,
        rows=summary_rows,
    )

    _save_all_summary_txt(
        run_dir=run_dir,
        summaries=summaries,
    )

    _log_final_regression_status(
        current_all_values=current_all_values,
        baseline=baseline,
    )

    return current_all_values


def _plot_baseline_comparison(
    run_dir: Path,
    current_values: dict[tuple[str, str], float],
    baseline: dict[tuple[str, str], float],
) -> None:
    """Plot current versus baseline mean Pearson r for each origin."""
    keys = [
        key
        for key in sorted(
            current_values,
            key=lambda key: (
                key[0] == "all",
                key[0],
                HEMISPHERES.index(key[1]),
            ),
        )
        if key in baseline
    ]

    if not keys:
        logger.warning(
            "No current/baseline values available for comparison plot.",
        )
        return

    labels = [f"{origin} {hemisphere[0].upper()}" for origin, hemisphere in keys]

    y_positions = np.arange(len(keys))

    baseline_values = np.array(
        [baseline[key] for key in keys],
        dtype=float,
    )

    current_values_array = np.array(
        [current_values[key] for key in keys],
        dtype=float,
    )

    fig_height = max(8, 0.45 * len(keys))

    fig, ax = plt.subplots(
        figsize=(12, fig_height),
    )

    try:
        for y, baseline_r, current_r in zip(
            y_positions,
            baseline_values,
            current_values_array,
            strict=True,
        ):
            ax.plot(
                [baseline_r, current_r],
                [y, y],
                linewidth=2,
                alpha=0.7,
            )

        ax.scatter(
            baseline_values,
            y_positions,
            s=55,
            marker="o",
            label="Baseline",
            zorder=3,
        )

        ax.scatter(
            current_values_array,
            y_positions,
            s=55,
            marker="D",
            label="Current",
            zorder=3,
        )

        ax.axvline(
            1.0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)

        ax.set_xlim(
            0.45,
            1.01,
        )

        ax.set_xlabel("Mean Pearson r")
        ax.set_ylabel("Origin / hemisphere")

        ax.set_title(
            "Cycle round-trip accuracy: current vs baseline",
            fontsize=16,
        )

        ax.grid(
            axis="x",
            linestyle=":",
            alpha=0.5,
        )

        ax.legend(
            loc="lower right",
        )

        fig.tight_layout()

        output_file = run_dir / "cycle_baseline_comparison.png"

        fig.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)

    logger.info(
        "Saved baseline comparison plot: %s",
        output_file,
    )


def _save_cycle_results(
    origin: str,
    hemisphere: Hemisphere,
    rows: list[dict[str, object]],
    plot_dir: Path,
) -> int:
    """Save CSV/TXT cycle results and validate regression output."""
    if not rows:
        logger.warning(
            "No executable paths for %s (%s).",
            origin,
            hemisphere,
        )

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

    return len(rows)


def _run_cycle_path(
    graph: NeuromapsGraph,
    metric_file: Path,
    original_metric: np.ndarray,
    path: tuple[str, ...],
    hemisphere: Hemisphere,
    path_workdir: Path,
    plot_dir: Path,
) -> dict[str, object] | None:
    """Execute, score, and plot one transformation cycle."""
    path_label = " -> ".join(path)

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

    try:
        metrics_by_hop = [(path[0], original_metric)] + [
            (
                hop.target,
                hop.metric_values,
            )
            for hop in roundtrip.hops
        ]

        plot_cycle_cortical_surfaces(
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
    hemisphere: Hemisphere,
) -> int:
    """Run all cycles for one origin and hemisphere."""
    work_dir = OUTPUT_DIR / f"work_{origin}_{hemisphere}"

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        density = graph.find_highest_density(origin)

        metric_file = make_sphere(
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

    original_metric = load_metric(metric_file)

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

        token = path_token(path)

        row = _run_cycle_path(
            graph=graph,
            metric_file=metric_file,
            original_metric=original_metric,
            path=path,
            hemisphere=hemisphere,
            path_workdir=work_dir / f"path_{token}",
            plot_dir=plot_dir,
        )

        if row is not None:
            rows.append(row)

    return _save_cycle_results(
        origin=origin,
        hemisphere=hemisphere,
        rows=rows,
        plot_dir=plot_dir,
    )
