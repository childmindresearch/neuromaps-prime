"""Plot cycle regression results for an entire regression run.

Given a cycle output directory, find all left/right hemisphere CSV files,
group them by origin space, and create one figure per origin.

Example:
-------
python plot_cycle_results.py \
    tests/regression/cycle_outputs_9ac4d1f0
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "path",
    "n_hops",
    "pearson_r",
    "max_abs_diff",
}

CSV_PATTERN = re.compile(
    r"^cycle_(?P<origin>.+)_(?P<label>.+)_(?P<hemisphere>left|right)\.csv$"
)


def find_cycle_csvs(
    run_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Find cycle CSVs and group them by origin and hemisphere."""
    grouped: dict[str, dict[str, Path]] = {}

    for csv_path in sorted(run_dir.glob("cycle_*.csv")):
        match = CSV_PATTERN.match(csv_path.name)

        if match is None:
            continue

        origin = match.group("origin")
        hemisphere = match.group("hemisphere")

        grouped.setdefault(origin, {})[hemisphere] = csv_path

    return grouped


def load_cycle_csv(
    csv_path: Path,
) -> pd.DataFrame:
    """Load and validate one cycle CSV."""
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df.sort_values(
        "pearson_r",
        ascending=True,
    ).reset_index(drop=True)


def plot_origin(
    origin: str,
    hemisphere_files: dict[str, Path],
    output_path: Path,
    *,
    run_dir: Path,
    dpi: int,
) -> None:
    """Create a two-panel left/right hemisphere figure."""
    data: dict[str, pd.DataFrame] = {}

    for hemisphere in ("left", "right"):
        csv_path = hemisphere_files.get(hemisphere)

        if csv_path is None:
            continue

        data[hemisphere] = load_cycle_csv(csv_path)

    if not data:
        return

    max_cycles = max(len(frame) for frame in data.values())

    figure_height = max(
        7,
        0.32 * max_cycles + 2.5,
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, figure_height),
        sharex=True,
    )

    # Make axes iterable even if matplotlib changes behavior for one panel.
    axes = list(axes)

    for index, hemisphere in enumerate(("left", "right")):
        ax = axes[index]

        frame = data.get(hemisphere)

        if frame is None:
            ax.text(
                0.5,
                0.5,
                "No results",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{hemisphere.capitalize()} hemisphere")
            ax.set_xlim(-1, 1)
            ax.set_xlabel("Pearson r")
            continue

        ax.barh(
            frame["path"],
            frame["pearson_r"],
        )

        ax.axvline(
            0,
            linewidth=1,
        )

        ax.set_title(f"{hemisphere.capitalize()} hemisphere ({len(frame)} cycles)")

        ax.set_xlabel("Pearson r")
        ax.set_xlim(-1, 1)

        ax.grid(
            axis="x",
            alpha=0.25,
        )

        ax.tick_params(
            axis="y",
            labelsize=8,
        )

    fig.suptitle(
        f"{origin} — surface transformation cycle round-trip accuracy",
        fontsize=16,
    )

    fig.text(
        0.5,
        0.01,
        f"Run: {run_dir.name}",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.025, 1, 0.96),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info("Saved: %s", output_path)


def main() -> None:
    """Generate one plot per origin."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Create left/right hemisphere cycle plots from a regression run directory."
        )
    )

    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "Cycle regression output directory, e.g. "
            "tests/regression/cycle_outputs_9ac4d1f0"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=("Directory for generated plots. Defaults to the input run directory."),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI (default: 200).",
    )

    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    if not run_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {run_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir
    )

    grouped = find_cycle_csvs(run_dir)

    if not grouped:
        raise FileNotFoundError(f"No cycle CSV files found in {run_dir}")

    logger.info(
        "Found results for %d origin space(s):",
        len(grouped),
    )

    for origin, hemisphere_files in grouped.items():
        logger.info(
            "  %s: %s",
            origin,
            ", ".join(sorted(hemisphere_files)),
        )

    for origin, hemisphere_files in sorted(grouped.items()):
        output_path = output_dir / f"cycle_{origin}_left_right_pearson.png"

        plot_origin(
            origin=origin,
            hemisphere_files=hemisphere_files,
            output_path=output_path,
            run_dir=run_dir,
            dpi=args.dpi,
        )

    logger.info(
        "Created %d origin plot(s).",
        len(grouped),
    )


if __name__ == "__main__":
    main()
