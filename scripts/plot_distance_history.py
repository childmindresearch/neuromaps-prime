"""Render the distance-map history timeline from accumulated run summaries.

Reads every ``distance_map_<YYYYmmdd_HHMMSS>[_<sha8>].csv`` run summary under a
directory and writes ``distance_map_history.svg``: one panel per scope
(``connected``, ``direct``), tracking the mean Pearson r for every seed across
runs. Each run is one x-axis tick, labelled with its date and commit hash (when
tagged). Colours and markers distinguish seeds; left hemispheres are drawn
filled, right hemispheres open. A pure consumer of finished summaries -- it
reads no test data and performs no transforms.

Run with:

    uv run scripts/plot_distance_history.py --dir <history folder> \
        [--output-dir <dir>]
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.10.7",
#     "numpy>=2.4.6",
#     "pandas>=3.0.3",
# ]
# ///

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

HEMISPHERES = ("left", "right")

SUMMARY_NAME: Final = re.compile(
    r"^distance_map_(?P<date>\d{8})_(?P<time>\d{6})(?:_(?P<sha>[0-9a-f]{8}))?\.csv$"
)

REQUIRED_COLUMNS: Final = ("seed", "scope", "hemisphere", "mean_pearson_r")

# Distinct marker shapes cycled across seeds (colour carries the primary
# distinction; shape is a secondary cue that must read well filled and open).
MARKER_SHAPES: Final = ("o", "s", "^", "D", "p", "h", "*", "v", "<", ">")

# Connecting lines are semi-transparent so overlapping series stay legible;
# markers stay opaque so individual run values remain crisp.
LINE_ALPHA: Final = 0.6


def _parse_summary_name(name: str) -> tuple[datetime, str | None] | None:
    """Parse a summary file name into ``(timestamp, sha)``, or ``None`` on no match."""
    match = SUMMARY_NAME.match(name)
    if match is None:
        return None
    try:
        timestamp = datetime.strptime(
            f"{match['date']} {match['time']}", "%Y%m%d %H%M%S"
        )
    except ValueError:
        return None
    return timestamp, match["sha"]


def _load_history(history_dir: Path) -> pd.DataFrame:
    """Read every run summary in history_dir into one long frame, sorted by time.

    Unrecognised, unreadable, and empty files are skipped; a summary missing
    required columns, or no valid summaries at all, is a hard error (exit code 2).
    """
    frames: list[pd.DataFrame] = []
    # rglob so released runs under history/releases/ are included, matching the
    # cycle script.
    for path in sorted(history_dir.rglob("distance_map_*.csv")):
        parsed = _parse_summary_name(path.name)
        if parsed is None:
            # The artifact dir also holds per-seed matrix CSVs; stay quiet.
            logger.debug("Skipping non-summary CSV: %s", path.name)
            continue
        timestamp, sha = parsed
        try:
            frame = pd.read_csv(path)
        except (
            OSError,
            UnicodeDecodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as exc:
            logger.warning("Skipping unreadable summary %s: %s", path.name, exc)
            continue
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            logger.error(
                "Summary %s is missing required columns: %s", path.name, missing
            )
            raise SystemExit(2)
        if frame.empty:
            logger.warning("Skipping empty summary file: %s", path.name)
            continue
        frame = frame.assign(timestamp=timestamp, sha=sha)
        frames.append(frame[["timestamp", "sha", *REQUIRED_COLUMNS]])

    if not frames:
        logger.error("No run summaries found in %s", history_dir)
        raise SystemExit(2)

    history = pd.concat(frames, ignore_index=True)
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history["mean_pearson_r"] = pd.to_numeric(
        history["mean_pearson_r"], errors="coerce"
    )
    return history.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _run_labels(
    timestamps: list[datetime], sha_by_run: dict[datetime, str | None]
) -> list[str]:
    """Build one x-tick label per run: the date, plus the commit hash if tagged."""
    labels: list[str] = []
    for timestamp in timestamps:
        label = timestamp.strftime("%Y-%m-%d")
        sha = sha_by_run.get(timestamp)
        if sha:
            label = f"{label}\n{sha}"
        labels.append(label)
    return labels


def _get_marker_map(seeds: list[str]) -> dict[str, str]:
    """Return a distinct marker shape for each seed, cycling if shapes run out."""
    return {
        seed: MARKER_SHAPES[index % len(MARKER_SHAPES)]
        for index, seed in enumerate(seeds)
    }


def _get_color_map(seeds: list[str]) -> dict[str, str]:
    """Return a distinct colour per seed from nipy_spectral, avoiding its extremes."""
    cmap = plt.get_cmap("nipy_spectral")
    positions = np.linspace(0.1, 0.9, len(seeds)) if len(seeds) > 1 else np.array([0.5])
    return {seed: to_hex(cmap(pos)) for seed, pos in zip(seeds, positions, strict=True)}


def _legend_handles(
    seeds: list[str], marker_map: dict[str, str], color_map: dict[str, str]
) -> list[Line2D]:
    """Build the panel legend: a fill-convention key plus one entry per seed."""

    def entry(label: str, marker: str, facecolor: str, edgecolor: str) -> Line2D:
        """One zero-length line styled as a legend marker."""
        return Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markerfacecolor=facecolor,
            markeredgecolor=edgecolor,
            markeredgewidth=1.5,
            markersize=8,
            label=label,
        )

    gray = "#888888"
    return [
        entry("Left hemisphere", "o", gray, gray),
        entry("Right hemisphere", "o", "none", gray),
        *[
            entry(seed, marker_map[seed], color_map[seed], color_map[seed])
            for seed in seeds
        ],
    ]


def _seed_series(
    history: pd.DataFrame,
    *,
    scope: str,
    seed: str,
    hemisphere: str,
    timestamps: list[datetime],
) -> np.ndarray:
    """Return one seed/scope/hemisphere's values across the runs; NaN where absent."""
    subset = history[
        (history["scope"] == scope)
        & (history["seed"] == seed)
        & (history["hemisphere"] == hemisphere)
    ]
    if subset.empty:
        return np.full(len(timestamps), np.nan, dtype=float)
    series = (
        subset.drop_duplicates(subset="timestamp")
        .set_index("timestamp")
        .loc[:, "mean_pearson_r"]
        .reindex(pd.DatetimeIndex(timestamps))
    )
    return series.to_numpy(dtype=float)


def _plot_scope_panel(
    ax: plt.Axes,
    history: pd.DataFrame,
    *,
    scope: str,
    seeds: list[str],
    timestamps: list[datetime],
    labels: list[str],
    marker_map: dict[str, str],
    color_map: dict[str, str],
) -> None:
    """Populate one scope panel: one line-pair per seed across the runs."""
    x = np.arange(len(timestamps))
    drawn_seeds: list[str] = []

    for seed in seeds:
        series = {
            hemisphere: _seed_series(
                history,
                scope=scope,
                seed=seed,
                hemisphere=hemisphere,
                timestamps=timestamps,
            )
            for hemisphere in HEMISPHERES
        }
        if not any(np.any(np.isfinite(series[h])) for h in HEMISPHERES):
            continue
        drawn_seeds.append(seed)

        color = color_map[seed]
        marker = marker_map[seed]
        for hemisphere in HEMISPHERES:
            values = series[hemisphere]
            if not np.any(np.isfinite(values)):
                continue
            ax.plot(
                x,
                values,
                color=color,
                alpha=LINE_ALPHA,
                linestyle="-" if hemisphere == "left" else "--",
                linewidth=1.5,
                marker=marker,
                markersize=6,
                markerfacecolor=color if hemisphere == "left" else "none",
                markeredgecolor=color,
                markeredgewidth=1.5,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(-0.5, len(timestamps) - 0.5)
    # Keep the full signed [-1, 1] range: a low or negative r is the warning.
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Run")
    ax.set_ylabel("Mean Pearson r")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_title(scope.capitalize())

    if drawn_seeds:
        ax.legend(
            handles=_legend_handles(
                drawn_seeds, marker_map=marker_map, color_map=color_map
            ),
            title="Hemisphere / seed",
            fontsize=7,
            loc="lower left",
        )


def _save_figure(fig: Figure, output_file: Path) -> None:
    """Save and close a figure as a vector SVG."""
    try:
        fig.savefig(output_file, bbox_inches="tight")
    finally:
        plt.close(fig)
    logger.info("Saved timeline plot: %s", output_file)


def plot_history(history_dir: Path, output_dir: Path) -> None:
    """Write distance_map_history.svg: one panel per scope, tracking every seed.

    One panel per scope (``connected``, ``direct``) across the accumulated runs;
    pooled ``seed='all'`` rows are omitted so per-seed detail localises drift.
    """
    history = _load_history(history_dir)

    timestamps = sorted(history["timestamp"].unique().tolist())
    sha_by_run = (
        history.drop_duplicates(subset="timestamp")
        .set_index("timestamp")["sha"]
        .to_dict()
    )
    labels = _run_labels(timestamps, sha_by_run)

    per_seed = history[history["seed"] != "all"]
    present = set(per_seed["scope"])
    scopes = [scope for scope in ("connected", "direct") if scope in present]
    if not scopes:
        logger.error("No per-seed rows (seed != 'all') in the history.")
        raise SystemExit(2)

    seeds = sorted(set(per_seed["seed"]))
    marker_map = _get_marker_map(seeds)
    color_map = _get_color_map(seeds)

    # At most two scopes are ever produced, so the panels always fit one row.
    fig, axes = plt.subplots(
        1, len(scopes), figsize=(7 * len(scopes), 5.5), squeeze=False
    )
    axes_flat = axes.flatten()

    for i, scope in enumerate(scopes):
        _plot_scope_panel(
            axes_flat[i],
            per_seed,
            scope=scope,
            seeds=seeds,
            timestamps=timestamps,
            labels=labels,
            marker_map=marker_map,
            color_map=color_map,
        )

    fig.suptitle("Distance-map correlation over runs", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    _save_figure(fig, output_dir / "distance_map_history.svg")
    logger.info("Wrote timeline plot to %s", output_dir)


def main() -> int:
    """Parse arguments, load the accumulated summaries, and write the plot."""
    parser = argparse.ArgumentParser(
        description="Render the distance-map history timeline from run summaries."
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Directory of distance_map_<YYYYmmdd_HHMMSS>[_<sha8>].csv summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the SVG into (default: the --dir directory).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    history_dir = args.dir
    if not history_dir.is_dir():
        raise SystemExit(f"History directory not found: {history_dir}")

    output_dir = args.output_dir if args.output_dir is not None else history_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_history(history_dir=history_dir, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
