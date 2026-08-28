"""Render cycle-regression summary plots from a run's summary CSV.

Reads the run-summary CSV (``cycle_<YYYYmmdd_HHMMSS>.csv``) produced by
``tests/regression/test_cycle.py`` and writes the round-trip accuracy
figures into an output directory:

* ``cycle_summary.png`` — mean Pearson r per origin space, both hemispheres;
* ``cycle_summary_<species>.png`` — the same view restricted to one species.

The script is a post-hoc consumer of a single finished run's summary: it
reads no previous baseline and makes no cross-run comparison.

Run with:

    uv run scripts/plot_cycle.py --summary <csv> [--output-dir <dir>]
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.10.7",
#     "pandas>=3.0.3",
# ]
# ///

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.markers import MarkerStyle

logger = logging.getLogger(__name__)

HEMISPHERES = ("left", "right")


def _summary_rows_to_frame(summary_rows: list[dict[str, object]]) -> pd.DataFrame:
    """Convert per-origin summary rows to a plotting dataframe."""
    if not summary_rows:
        return pd.DataFrame(columns=["origin", "hemisphere", "pearson_r"])

    return pd.DataFrame(
        [
            {
                "origin": row["origin"],
                "hemisphere": row["hemisphere"],
                "pearson_r": row["mean_pearson_r"],
            }
            for row in summary_rows
        ]
    )


def _get_plot_origins(frame: pd.DataFrame, origins: list[str]) -> list[str]:
    """Return origins that have cycle summary data."""
    available_origins = set(frame["origin"])

    return [origin for origin in origins if origin in available_origins]


def _get_marker_map(origins: list[str]) -> dict[str, MarkerStyle]:
    """Return deterministic markers with consistent visual size."""
    markers: dict[str, MarkerStyle] = {}

    for origin, marker in zip(origins, MarkerStyle.markers, strict=False):
        style = MarkerStyle(marker)

        if style.get_path().vertices.size == 0:
            continue

        path = style.get_path().transformed(style.get_transform())
        path = path.cleaned()

        markers[origin] = MarkerStyle(path)

    return markers


def _get_color_map(origins: list[str]) -> dict[str, str]:
    """Return a consistent color for each origin."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    return {
        origin: color_cycle[index % len(color_cycle)]
        for index, origin in enumerate(origins)
    }


def _plot_hemisphere_point(
    ax: plt.Axes,
    x_position: int,
    origin: str,
    hemisphere: str,
    mean_r: float,
    marker: MarkerStyle,
    color: str,
) -> None:
    """Plot one hemisphere's cycle summary point."""
    facecolor = color if hemisphere == "left" else "none"

    ax.scatter(
        x_position,
        mean_r,
        s=120,
        marker=marker,
        facecolors=facecolor,
        edgecolors=color,
        linewidths=1.5,
        label=f"{origin} {hemisphere.upper()[0]}",
        zorder=3,
    )


def _configure_species_summary_plot(
    ax: plt.Axes, frame: pd.DataFrame, origins: list[str], species: str | None
) -> None:
    """Configure axes, labels, title, grid, and legend."""
    ax.set_xticks(range(len(origins)))
    ax.set_xticklabels(origins, rotation=45, ha="right")

    ax.set_xlabel("Origin space")
    ax.set_ylabel("Mean Pearson r")

    ax.set_ylim(min(0.0, frame["pearson_r"].min() - 0.05), 1.01)

    if species is None:
        title = "Surface transformation cycle round-trip accuracy"
    else:
        title = f"Surface transformation cycle round-trip accuracy — {species}"

    ax.set_title(title, fontsize=16)

    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax.legend(title="Origin / hemisphere", bbox_to_anchor=(1.02, 1), loc="upper left")


def _get_species_summary_output(output_dir: Path, species: str | None) -> Path:
    """Return the output path for a species summary plot."""
    if species is None:
        return output_dir / "cycle_summary.png"

    return output_dir / f"cycle_summary_{species.lower()}.png"


def _save_species_summary_plot(fig: plt.Figure, output_file: Path) -> None:
    """Save and close a species summary plot."""
    try:
        fig.tight_layout()
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)

    logger.info("Saved cycle summary plot: %s", output_file)


def _plot_species_summary(
    output_dir: Path,
    frame: pd.DataFrame,
    origins: list[str],
    species: str | None = None,
) -> None:
    """Create a cycle round-trip summary plot."""
    unique_origins = _get_plot_origins(frame=frame, origins=origins)

    if not unique_origins:
        logger.warning(
            "No matching origins found for %s summary plot.", species or "all"
        )
        return

    marker_map = _get_marker_map(unique_origins)
    color_map = _get_color_map(unique_origins)

    fig_width = max(8, 1.25 * len(unique_origins))

    fig, ax = plt.subplots(figsize=(fig_width, 7))

    for x_position, origin in enumerate(unique_origins):
        for hemisphere in HEMISPHERES:
            subset = frame[
                (frame["origin"] == origin) & (frame["hemisphere"] == hemisphere)
            ]

            if subset.empty:
                continue

            _plot_hemisphere_point(
                ax=ax,
                x_position=x_position,
                origin=origin,
                hemisphere=hemisphere,
                mean_r=float(subset["pearson_r"].iloc[0]),
                marker=marker_map[origin],
                color=color_map[origin],
            )

    _configure_species_summary_plot(
        ax=ax, frame=frame, origins=unique_origins, species=species
    )

    output_file = _get_species_summary_output(output_dir=output_dir, species=species)

    _save_species_summary_plot(fig=fig, output_file=output_file)


def plot_run_summaries(output_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    """Create overall and species-specific cycle summary plots in ``output_dir``."""
    origins = sorted(
        {str(row["origin"]) for row in summary_rows if row["origin"] != "all"}
    )

    frame = _summary_rows_to_frame(summary_rows)

    # Overall summary.
    _plot_species_summary(output_dir=output_dir, frame=frame, origins=origins)

    # Species-specific summaries.
    species_origins: dict[str, list[str]] = {}

    for row in summary_rows:
        origin = str(row["origin"])

        if origin == "all":
            continue

        species = str(row.get("species", "all"))
        species_origins.setdefault(species, []).append(origin)

    for species, species_origins_list in sorted(species_origins.items()):
        species_origins_list = sorted(set(species_origins_list))

        _plot_species_summary(
            output_dir=output_dir,
            frame=frame,
            origins=species_origins_list,
            species=species,
        )


def main() -> int:
    """Parse arguments, load the run summary, and write the plots."""
    parser = argparse.ArgumentParser(
        description="Render cycle-regression summary plots from a run's summary CSV."
    )
    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
        help="The run-summary CSV (cycle_<YYYYmmdd_HHMMSS>.csv) to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the PNGs into (default: the summary CSV's directory).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    summary = args.summary

    if not summary.is_file():
        raise SystemExit(f"Summary CSV not found: {summary}")

    output_dir = args.output_dir if args.output_dir is not None else summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(summary)
    summary_rows = frame[frame["origin"] != "all"].to_dict(orient="records")

    if not summary_rows:
        raise SystemExit(f"No per-origin rows (origin != 'all') in {summary}.")

    plot_run_summaries(output_dir=output_dir, summary_rows=summary_rows)

    logger.info("Wrote cycle summary plots to %s", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
