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
    make_xyz_product_metric,
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
    """Collect current mean Pearson r values for each origin and hemisphere."""
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

    return values


def _check_cycle_baseline(
    current_values: dict[tuple[str, str], float],
    baseline: dict[tuple[str, str], float],
) -> None:
    """Compare current cycle results against the previous baseline."""
    for key, current_r in current_values.items():
        if key not in baseline:
            raise AssertionError(
                f"No baseline found for origin={key[0]}, hemisphere={key[1]}."
            )

        baseline_r = baseline[key]
        difference = current_r - baseline_r

        logger.info(
            "BASELINE (%s, %s): current=%.6f, baseline=%.6f, difference=%+.6f",
            key[0],
            key[1],
            current_r,
            baseline_r,
            difference,
        )

        assert current_r >= baseline_r - ALLOWED_REGRESSION, (
            "Average round-trip correlation regressed: "
            f"origin={key[0]}, "
            f"hemisphere={key[1]}, "
            f"current r={current_r:.6f}, "
            f"baseline={baseline_r:.6f}, "
            f"difference={difference:+.6f}, "
            f"allowed regression={ALLOWED_REGRESSION:.6f}. "
            f"Inspect outputs in {OUTPUT_DIR}."
        )


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

    current_values.update(
        _save_all_summary(
            run_dir=OUTPUT_DIR,
        )
    )

    _check_cycle_baseline(
        current_values=current_values,
        baseline=baseline,
    )

    save_cycle_baseline(
        baseline_dir=baseline_dir,
        values=current_values,
    )

    plot_run_summaries(
        run_dir=OUTPUT_DIR,
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

    for hemisphere in HEMISPHERES:
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
            logger.warning(
                "Empty %s CSV found for %s: %s",
                hemisphere,
                origin,
                csv_path,
            )
            continue

        frames[hemisphere] = frame

    if not frames:
        logger.warning(
            "No cycle CSVs found for origin %s.",
            origin,
        )
        return

    paths = sorted(
        {str(path) for frame in frames.values() for path in frame["path"]},
        key=lambda path: (
            len(path.split(" -> ")) - 1,
            path,
        ),
    )

    path_positions = {path: index for index, path in enumerate(paths)}

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(
            18,
            max(8, 0.30 * len(paths)),
        ),
        sharey=True,
    )

    try:
        for ax, hemisphere in zip(
            axes,
            HEMISPHERES,
            strict=True,
        ):
            frame = frames.get(hemisphere)

            if frame is None:
                ax.set_visible(False)
                continue

            frame = frame.copy()
            frame["path"] = frame["path"].astype(str)
            frame["position"] = frame["path"].map(path_positions)

            frame = frame.dropna(subset=["position", "pearson_r"])
            frame = frame.sort_values("position")

            ax.axvline(
                1.0,
                linestyle="--",
                linewidth=1,
            )

            ax.scatter(
                frame["pearson_r"],
                frame["position"],
                s=40,
            )

            ax.set_yticks(range(len(paths)))
            ax.set_yticklabels(
                paths if hemisphere == HEMISPHERES[0] else [],
            )

            ax.set_xlim(
                min(0.0, frame["pearson_r"].min() - 0.05),
                1.01,
            )
            ax.set_xlabel("Pearson r")
            ax.set_title(f"{hemisphere.capitalize()} hemisphere")
            ax.grid(
                axis="x",
                linestyle=":",
                alpha=0.5,
            )

            for _, row in frame.iterrows():
                ax.annotate(
                    f"{row['pearson_r']:.4f}",
                    (
                        row["pearson_r"],
                        row["position"],
                    ),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )

        axes[0].set_ylabel("Transformation path")

        fig.suptitle(
            f"{origin}\nSurface transformation cycle round-trip accuracy",
            fontsize=16,
        )

        fig.tight_layout(
            rect=(0, 0, 1, 0.95),
        )

        output_file = run_dir / f"cycle_{origin}_summary.png"

        fig.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)

    logger.info(
        "Saved combined cycle summary plot: %s",
        output_file,
    )


def _save_all_summary(
    run_dir: Path,
) -> dict[tuple[str, str], float]:
    """Calculate and save overall left/right Pearson r across all origins."""
    summaries: list[str] = []
    current_all_values: dict[tuple[str, str], float] = {}

    for hemisphere in HEMISPHERES:
        csv_files = sorted(
            run_dir.glob(f"cycle_*_{hemisphere}.csv"),
        )

        frames: list[pd.DataFrame] = []

        for csv_file in csv_files:
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
        return current_all_values

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

    return current_all_values


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

        metric_file = make_xyz_product_metric(
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
