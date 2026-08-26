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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.markers import MarkerStyle
from tests.cycle import (
    Hemisphere,
    find_return_paths,
    load_metric,
    path_token,
    roundtrip_metric,
    score_roundtrip,
)
from tests.regression.utils import (
    load_latest_cycle_values,
    make_sphere,
    plot_cycle_cortical_surfaces,
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


@dataclass(frozen=True)
class _CycleSummary:
    """Aggregated cycle means and derived report rows."""

    pearson_r: dict[tuple[str, str], float]
    origin_rows: list[dict[str, object]]
    all_rows: list[dict[str, object]]
    summaries: list[str]


def _read_cycle_frames(
    run_dir: Path,
    origins: list[str],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Read each cycle CSV once, keyed by origin and hemisphere."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}

    for origin in origins:
        for hemisphere in HEMISPHERES:
            csv_path = run_dir / f"cycle_{origin}_{hemisphere}.csv"

            if not csv_path.exists():
                continue

            frame = pd.read_csv(csv_path)

            if frame.empty:
                continue

            frames[(origin, hemisphere)] = frame

    return frames


def _summarize(
    frames: dict[tuple[str, str], pd.DataFrame],
    species_by_origin: dict[str, str],
    previous: dict[tuple[str, str], float],
) -> _CycleSummary:
    """Calculate cycle means and derived summary rows."""
    pearson_r: dict[tuple[str, str], float] = {}
    origin_rows: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    summaries: list[str] = []

    by_hemi: dict[str, list[pd.DataFrame]] = {
        hemisphere: [] for hemisphere in HEMISPHERES
    }

    for (origin, hemisphere), frame in sorted(frames.items()):
        mean_r = float(frame["pearson_r"].mean())

        pearson_r[(origin, hemisphere)] = mean_r

        origin_rows.append(
            {
                "origin": origin,
                "species": species_by_origin.get(
                    origin,
                    "all",
                ),
                "hemisphere": hemisphere,
                "mean_pearson_r": mean_r,
            }
        )

        by_hemi[hemisphere].append(frame)

    for hemisphere in HEMISPHERES:
        frames_for_hemisphere = by_hemi[hemisphere]

        if not frames_for_hemisphere:
            continue

        combined = pd.concat(
            frames_for_hemisphere,
            ignore_index=True,
        )

        mean_r = float(combined["pearson_r"].mean())

        pearson_r[("all", hemisphere)] = mean_r

        all_rows.append(
            {
                "origin": "all",
                "species": "all",
                "hemisphere": hemisphere,
                "mean_pearson_r": mean_r,
            }
        )

        previous_r = previous[("all", hemisphere)]
        difference = mean_r - previous_r

        summaries.append(
            f"All spaces ({hemisphere}):\n"
            f"  Total executable cycles: {len(combined)}\n"
            f"  Mean Pearson r: {mean_r:.6f}\n"
            f"  Previous Pearson r: {previous_r:.6f}\n"
            f"  Difference: {difference:+.6f}\n"
        )


def test_cycle_roundtrip() -> None:
    """Round-trip synthetic metrics through real transformation cycles."""
    graph = NeuromapsGraph()
    dir = Path(__file__).resolve().parent

    previous = load_latest_cycle_values(
        dir=dir,
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

    species_by_origin = {
        origin: graph.get_node_data(origin).species for origin in origins
    }

    frames = _read_cycle_frames(
        run_dir=OUTPUT_DIR,
        origins=origins,
    )

    summary = _summarize(
        frames=frames,
        species_by_origin=species_by_origin,
        previous=previous,
    )

    for hemisphere in HEMISPHERES:
        key = ("all", hemisphere)

        current_r = summary.pearson_r[key]
        previous_r = previous[key]

        assert current_r >= previous_r - 1e-4, (
            f"Average round-trip correlation regressed: "
            f"origin=all, hemisphere={hemisphere}, "
            f"mean r={current_r:.6f}, "
            f"previous={previous_r:.6f}, "
            f"threshold={previous_r - 1e-4:.6f}"
        )

    frame = _summary_frame(
        summary.origin_rows + summary.all_rows,
    )

    baseline_path = dir / f"cycle_{datetime.now():%Y%m%d_%H%M}.csv"
    summary_path = OUTPUT_DIR / "cycle_summary.csv"

    frame.to_csv(
        baseline_path,
        index=False,
    )

    frame.to_csv(
        summary_path,
        index=False,
    )

    logger.info(
        "Saved baseline CSV: %s",
        baseline_path,
    )

    logger.info(
        "Saved cycle summary CSV: %s",
        summary_path,
    )

    _save_all_summary_txt(
        run_dir=OUTPUT_DIR,
        summaries=summary.summaries,
    )

    plot_run_summaries(
        run_dir=OUTPUT_DIR,
        current_values=summary.pearson_r,
        pearson_r=previous,
        summary_rows=summary.origin_rows,
    )

    logger.info(
        "NEW PEARSON R VALUES: left=%.6f, right=%.6f",
        summary.pearson_r[("all", "left")],
        summary.pearson_r[("all", "right")],
    )


# -------------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------------


def _summary_rows_to_frame(
    summary_rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Convert per-origin summary rows to a plotting dataframe."""
    if not summary_rows:
        return pd.DataFrame(
            columns=[
                "origin",
                "hemisphere",
                "pearson_r",
            ]
        )

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


def _get_plot_origins(
    frame: pd.DataFrame,
    origins: list[str],
) -> list[str]:
    """Return origins that have cycle summary data."""
    available_origins = set(frame["origin"])

    return [origin for origin in origins if origin in available_origins]


def _get_marker_map(
    origins: list[str],
) -> dict[str, MarkerStyle]:
    """Return deterministic markers with consistent visual size."""
    markers = {}

    for origin, marker in zip(
        origins,
        MarkerStyle.markers,
        strict=False,
    ):
        style = MarkerStyle(marker)

        if style.get_path().vertices.size == 0:
            continue

        path = style.get_path().transformed(style.get_transform())

        path = path.cleaned()

        markers[origin] = MarkerStyle(path)

    return markers


def _get_color_map(
    origins: list[str],
) -> dict[str, str]:
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
    hemisphere: Hemisphere,
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
    ax: plt.Axes,
    frame: pd.DataFrame,
    origins: list[str],
    species: str | None,
) -> None:
    """Configure axes, labels, title, grid, and legend."""
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(range(len(origins)))
    ax.set_xticklabels(
        origins,
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

    if species is None:
        title = "Surface transformation cycle round-trip accuracy"
    else:
        title = f"Surface transformation cycle round-trip accuracy — {species}"

    ax.set_title(
        title,
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


def _get_species_summary_output(
    run_dir: Path,
    species: str | None,
) -> Path:
    """Return the output path for a species summary plot."""
    if species is None:
        return run_dir / "cycle_summary.png"

    return run_dir / f"cycle_summary_{species.lower()}.png"


def _save_species_summary_plot(
    fig: plt.Figure,
    output_file: Path,
) -> None:
    """Save and close a species summary plot."""
    try:
        fig.tight_layout()
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


def _plot_species_summary(
    run_dir: Path,
    frame: pd.DataFrame,
    origins: list[str],
    species: str | None = None,
) -> None:
    """Create a cycle round-trip summary plot."""
    unique_origins = _get_plot_origins(
        frame=frame,
        origins=origins,
    )

    if not unique_origins:
        logger.warning(
            "No matching origins found for %s summary plot.",
            species or "all",
        )
        return

    marker_map = _get_marker_map(unique_origins)
    color_map = _get_color_map(unique_origins)

    fig_width = max(
        8,
        1.25 * len(unique_origins),
    )

    fig, ax = plt.subplots(
        figsize=(fig_width, 7),
    )

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
        ax=ax,
        frame=frame,
        origins=unique_origins,
        species=species,
    )

    output_file = _get_species_summary_output(
        run_dir=run_dir,
        species=species,
    )

    _save_species_summary_plot(
        fig=fig,
        output_file=output_file,
    )


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------


def plot_run_summaries(
    run_dir: str | Path,
    current_values: dict[tuple[str, str], float],
    pearson_r: dict[tuple[str, str], float],
    summary_rows: list[dict[str, object]],
) -> None:
    """Create overall and species-specific cycle summary plots."""
    run_dir = Path(run_dir)

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}",
        )

    origins = sorted(
        {str(row["origin"]) for row in summary_rows if row["origin"] != "all"}
    )

    frame = _summary_rows_to_frame(summary_rows)

    # Overall summary.
    _plot_species_summary(
        run_dir=run_dir,
        frame=frame,
        origins=origins,
    )

    # Species-specific summaries.
    species_origins: dict[str, list[str]] = {}

    for row in summary_rows:
        origin = str(row["origin"])

        if origin == "all":
            continue

        species = str(row.get("species", "all"))

        species_origins.setdefault(
            species,
            [],
        ).append(origin)

    for species, species_origins_list in sorted(
        species_origins.items(),
    ):
        species_origins_list = sorted(set(species_origins_list))

        _plot_species_summary(
            run_dir=run_dir,
            frame=frame,
            origins=species_origins_list,
            species=species,
        )

    # Current versus previous comparison.
    _plot_pearson_comparison(
        run_dir=run_dir,
        current_values=current_values,
        pearson_r=pearson_r,
    )


def _summary_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build the canonical per-origin + all-space summary frame."""
    return (
        pd.DataFrame(
            rows,
            columns=["origin", "species", "hemisphere", "mean_pearson_r"],
        )
        .sort_values(["species", "origin", "hemisphere"], kind="stable")
        .reset_index(drop=True)
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


def _plot_pearson_comparison(
    run_dir: Path,
    current_values: dict[tuple[str, str], float],
    pearson_r: dict[tuple[str, str], float],
) -> None:
    """Plot current versus previous mean Pearson r for each origin."""
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
        if key in pearson_r
    ]

    if not keys:
        logger.warning(
            "No current/pearson values available for comparison plot.",
        )
        return

    labels = [f"{origin} {hemisphere[0].upper()}" for origin, hemisphere in keys]

    y_positions = np.arange(len(keys))

    pearson_values = np.array(
        [pearson_r[key] for key in keys],
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
        for y, pearson_r, current_r in zip(
            y_positions,
            pearson_values,
            current_values_array,
            strict=True,
        ):
            ax.plot(
                [pearson_r, current_r],
                [y, y],
                linewidth=2,
                alpha=0.7,
            )

        ax.scatter(
            pearson_values,
            y_positions,
            s=55,
            marker="o",
            label="Previous Pearson r",
            zorder=3,
        )

        ax.scatter(
            current_values_array,
            y_positions,
            s=55,
            marker="D",
            label="Current Pearson r",
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
            "Cycle round-trip accuracy: current vs previous Pearson r",
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

        output_file = run_dir / "cycle_comparison.png"

        fig.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)

    logger.info(
        "Saved cycle comparison plot: %s",
        output_file,
    )


def _save_cycle_results(
    origin: str,
    hemisphere: Hemisphere,
    rows: list[dict[str, object]],
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
        max_paths=None,
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

    plot_dir = OUTPUT_DIR / f"origin-{origin}_cycle-{hemisphere}"

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
    )
