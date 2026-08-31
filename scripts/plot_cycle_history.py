"""Render the cycle-history timeline from accumulated run summaries.

Reads every ``cycle_<YYYYmmdd_HHMMSS>[_<sha8>].csv`` run summary in a
directory (the ``cycle_history/`` folder on the ``cycle-history`` branch) and
writes a single multi-panel figure, ``cycle_history.svg``: one panel per
species present, tracking the mean round-trip Pearson r for every origin space
in that species across runs.

Each panel's x-axis has one position per run, tick-labelled with the run date
and, when the summary was tagged, the 8-character commit hash (runs without a
hash show the date alone). Colours and marker shapes are assigned per origin
space; left hemispheres are drawn filled and right hemispheres open. The script
reads no test data and performs no transformations — it is a pure consumer of
finished run summaries.

Output is SVG (vector) so the plot stays crisp at any zoom.

Run with:

    uv run scripts/plot_cycle_history.py --dir <cycle_history folder> \
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
import math
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

HEMISPHERES = ("left", "right")

SUMMARY_NAME: Final = re.compile(
    r"^cycle_(?P<date>\d{8})_(?P<time>\d{6})(?:_(?P<sha>[0-9a-f]{8}))?\.csv$"
)

REQUIRED_COLUMNS: Final = ("origin", "species", "hemisphere", "mean_pearson_r")

# Distinct, clearly visible marker shapes cycled across origins (colour already
# carries the primary distinction; the shape is a secondary cue and must read
# well both filled and open).
MARKER_SHAPES: Final = ("o", "s", "^", "D", "p", "h", "*", "v", "<", ">", "8", "+", "x")

# Connecting lines are drawn semi-transparent so overlapping series stay
# legible; the markers stay opaque so individual run values remain crisp.
LINE_ALPHA: Final = 0.6


def _parse_summary_name(name: str) -> tuple[datetime, str | None] | None:
    """Parse a summary file name into ``(timestamp, sha)``.

    Returns ``None`` when the name does not match the run-summary pattern or
    encodes an invalid date.
    """
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
    """Read every run summary in ``history_dir`` into one long frame.

    Returns a frame with columns
    ``timestamp, sha, origin, species, hemisphere, mean_pearson_r`` sorted by
    timestamp. Unrecognised, unreadable, and empty files are skipped with a
    warning; a summary missing required columns, or no valid summaries at all,
    is a hard error (exit code 2).
    """
    frames: list[pd.DataFrame] = []

    for path in sorted(history_dir.glob("cycle_*.csv")):
        parsed = _parse_summary_name(path.name)

        if parsed is None:
            logger.warning("Skipping unrecognised summary file: %s", path.name)
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

        frames.append(
            frame[
                [
                    "timestamp",
                    "sha",
                    "origin",
                    "species",
                    "hemisphere",
                    "mean_pearson_r",
                ]
            ]
        )

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
    """Build one x-tick label per run: the date, plus the commit hash if tagged.

    The label is rendered once per run (it is the axis tick, not a per-point
    annotation). Runs without a commit hash show the date alone.
    """
    labels: list[str] = []

    for timestamp in timestamps:
        label = timestamp.strftime("%Y-%m-%d")
        sha = sha_by_run.get(timestamp)

        if sha:
            label = f"{label}\n{sha}"

        labels.append(label)

    return labels


def _get_marker_map(origins: list[str]) -> dict[str, str]:
    """Return a distinct marker shape for each origin."""
    return dict(zip(origins, MARKER_SHAPES, strict=False))


def _get_color_map(origins: list[str]) -> dict[str, str]:
    """Return a consistent colour for each origin."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    return {
        origin: color_cycle[index % len(color_cycle)]
        for index, origin in enumerate(origins)
    }


def _legend_handles(
    origins: list[str], marker_map: dict[str, str], color_map: dict[str, str]
) -> list[Line2D]:
    """Build the panel legend handles.

    A two-entry key explains the fill convention (filled = left, open = right);
    each origin then gets a single coloured marker entry.
    """
    key = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#888888",
            markeredgecolor="#888888",
            markersize=8,
            label="Left hemisphere",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#888888",
            markeredgewidth=1.5,
            markersize=8,
            label="Right hemisphere",
        ),
    ]

    origin_entries = [
        Line2D(
            [0],
            [0],
            color=color_map[origin],
            linestyle="none",
            marker=marker_map[origin],
            markerfacecolor=color_map[origin],
            markeredgecolor=color_map[origin],
            markersize=8,
            label=origin,
        )
        for origin in origins
    ]

    return [*key, *origin_entries]


def _origin_series(
    history: pd.DataFrame, *, origin: str, hemisphere: str, timestamps: list[datetime]
) -> np.ndarray:
    """Return one origin/hemisphere's values across ``timestamps``.

    The result is NaN at any run where this origin was absent, so the
    connecting line breaks across a gap instead of bridging it.
    """
    subset = history[
        (history["origin"] == origin) & (history["hemisphere"] == hemisphere)
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


def _plot_species_panel(
    ax: plt.Axes,
    history: pd.DataFrame,
    *,
    species: str,
    origins: list[str],
    timestamps: list[datetime],
    labels: list[str],
    marker_map: dict[str, str],
    color_map: dict[str, str],
) -> None:
    """Populate one species panel: one line-pair per origin across the runs."""
    x = np.arange(len(timestamps))
    drawn_origins: list[str] = []

    for origin in origins:
        series = {
            hemisphere: _origin_series(
                history, origin=origin, hemisphere=hemisphere, timestamps=timestamps
            )
            for hemisphere in HEMISPHERES
        }

        if not any(
            np.any(np.isfinite(series[hemisphere])) for hemisphere in HEMISPHERES
        ):
            continue

        drawn_origins.append(origin)

        color = color_map[origin]
        marker = marker_map[origin]

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
    ax.set_ylim(0, 1)

    ax.set_xlabel("Run")
    ax.set_ylabel("Mean Pearson r")

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_title(species.capitalize())

    if drawn_origins:
        ax.legend(
            handles=_legend_handles(
                drawn_origins, marker_map=marker_map, color_map=color_map
            ),
            title="Hemisphere / origin",
            fontsize=7,
            loc="lower left",
        )


def _save_figure(fig: Figure, output_file: Path) -> None:
    """Save and close a figure as a vector SVG."""
    try:
        fig.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
    finally:
        plt.close(fig)

    logger.info("Saved timeline plot: %s", output_file)


def plot_history(history_dir: Path, output_dir: Path) -> None:
    """Write the multi-panel timeline figure ``cycle_history.svg``.

    One panel per species present, each tracking every origin space in that
    species across the accumulated runs.
    """
    history = _load_history(history_dir)

    timestamps = sorted(history["timestamp"].unique().tolist())
    sha_by_run = (
        history.drop_duplicates(subset="timestamp")
        .set_index("timestamp")["sha"]
        .to_dict()
    )
    labels = _run_labels(timestamps, sha_by_run)

    species_origins: dict[str, list[str]] = {}

    for _, row in history.iterrows():
        origin = str(row["origin"])

        if origin == "all":
            continue

        species_origins.setdefault(str(row["species"]), []).append(origin)

    species_list = sorted(species_origins)

    if not species_list:
        logger.error("No per-origin rows (origin != 'all') in the history.")
        raise SystemExit(2)

    n = len(species_list)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)

    # Colour and marker are assigned per origin name across the whole figure,
    # so an origin looks the same in every panel.
    all_origins = sorted(
        {origin for origins in species_origins.values() for origin in origins}
    )
    marker_map = _get_marker_map(all_origins)
    color_map = _get_color_map(all_origins)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for i, species in enumerate(species_list):
        _plot_species_panel(
            axes_flat[i],
            history,
            species=species,
            origins=sorted(set(species_origins[species])),
            timestamps=timestamps,
            labels=labels,
            marker_map=marker_map,
            color_map=color_map,
        )

    # Hide any panels left unused when the species count does not fill the grid.
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    # The run axis is shared by every panel; drop the redundant tick labels on
    # all but the bottom row.
    for i in range(nrows - 1):
        for j in range(ncols):
            axes[i][j].tick_params(axis="x", labelbottom=False)

    fig.suptitle("Cycle round-trip accuracy over runs", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    _save_figure(fig, output_dir / "cycle_history.svg")

    logger.info("Wrote timeline plot to %s", output_dir)


def main() -> int:
    """Parse arguments, load the accumulated summaries, and write the plot."""
    parser = argparse.ArgumentParser(
        description="Render the cycle-history timeline from accumulated run summaries."
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Directory containing cycle_<YYYYmmdd_HHMMSS>[_<sha8>].csv summaries.",
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
