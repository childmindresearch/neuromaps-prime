"""Cycle regression test on the real Neuromaps-PRIME graph.

End-to-end cycle regression: propagate a deterministic seed metric around
each return path in the real transformation graph and score how well it
round-trips, validating transform quality across multi-hop paths.

Cycle testing covers surface transformations only; the graph has no
volume-to-volume cycles in scope, so volume transforms are not exercised here.

Resulting files are written to a run-specific directory under the pytest
temporary directory (``<tmp_path>/cycle_outputs``). Set the
``NEUROMAPS_CYCLE_OUTPUT_DIR`` environment variable to store them elsewhere
instead. The resolved location is logged at the start of the test.

Each run may contain:

* CSV summaries
* TXT summaries
* intermediate metric files

Summary plots can be rendered separately from a run's summary CSV by
``scripts/plot_cycle.py``.

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

import numpy as np
import pandas as pd
import pytest
from tests.cycle import Hemisphere, resolve_artifact_dir, run_cycle_test
from tests.regression.utils import make_sphere

from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Run configuration
# -------------------------------------------------------------------------

# Always test every configured origin space.

HEMISPHERES = ("left", "right")
MAX_CYCLE_LENGTH: Final = 4


def _resolve_output_dir(base_dir: Path) -> Path:
    """Resolve and create the directory for cycle regression outputs.

    By default, artifacts are written to ``base_dir / "cycle_outputs"`` so
    they follow pytest's temporary-directory policy. Set the
    ``NEUROMAPS_CYCLE_OUTPUT_DIR`` environment variable to store them
    elsewhere instead.
    """
    return resolve_artifact_dir(
        base_dir / "cycle_outputs", env_var="NEUROMAPS_CYCLE_OUTPUT_DIR"
    )


# -------------------------------------------------------------------------
# Result containers
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleSummary:
    """Aggregated cycle means and derived report rows."""

    pearson_r: dict[tuple[str, str], float]
    origin_rows: list[dict[str, object]]
    all_rows: list[dict[str, object]]
    summaries: list[str]


@dataclass(frozen=True)
class CycleRunResult:
    """One executed cycle run: artifact location and per-path results.

    ``frames`` maps each (origin, hemisphere) that produced results to the
    per-path metrics read back from the run's CSVs; the summary CSV, TXT,
    and plots are written to ``output_dir`` as part of the run.
    """

    output_dir: Path
    frames: dict[tuple[str, str], pd.DataFrame]


def _read_cycle_frames(
    run_dir: Path, origins: list[str]
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
    frames: dict[tuple[str, str], pd.DataFrame], species_by_origin: dict[str, str]
) -> CycleSummary:
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
                "species": species_by_origin.get(origin, "all"),
                "hemisphere": hemisphere,
                "mean_pearson_r": mean_r,
            }
        )

        by_hemi[hemisphere].append(frame)

    for hemisphere in HEMISPHERES:
        frames_for_hemisphere = by_hemi[hemisphere]

        if not frames_for_hemisphere:
            continue

        combined = pd.concat(frames_for_hemisphere, ignore_index=True)

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

        summaries.append(
            f"All spaces ({hemisphere}):\n"
            f"  Total executable cycles: {len(combined)}\n"
            f"  Mean Pearson r: {mean_r:.6f}\n"
        )

    return CycleSummary(
        pearson_r=pearson_r,
        origin_rows=origin_rows,
        all_rows=all_rows,
        summaries=summaries,
    )


# -------------------------------------------------------------------------
# Producers
# -------------------------------------------------------------------------


def _summary_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build the canonical per-origin + all-space summary frame."""
    return (
        pd.DataFrame(
            rows, columns=["origin", "species", "hemisphere", "mean_pearson_r"]
        )
        .sort_values(["species", "origin", "hemisphere"], kind="stable")
        .reset_index(drop=True)
    )


def _save_all_summary_txt(run_dir: Path, summaries: list[str]) -> None:
    """Save the all-space cycle summary text file."""
    if not summaries:
        return

    output_file = run_dir / "cycle_all_summary.txt"

    lines = ["Cycle test results — all origin spaces", "=" * 60, "", *summaries]

    output_file.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Saved all-space summary: %s", output_file)


def _save_cycle_results(
    origin: str, hemisphere: Hemisphere, rows: list[dict[str, object]], output_dir: Path
) -> int:
    """Save CSV/TXT cycle results and validate regression output."""
    if not rows:
        logger.warning("No executable paths for %s (%s).", origin, hemisphere)

        return 0

    frame = pd.DataFrame(rows).sort_values(
        ["pearson_r", "path"], ascending=[False, True]
    )

    csv_path = output_dir / f"cycle_{origin}_{hemisphere}.csv"

    frame.to_csv(csv_path, index=False)

    path_width = max(
        len("Transformation path"), max(len(str(row["path"])) for row in rows)
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
        [separator, f"Total cycles: {len(rows)}", f"Mean Pearson r: {mean_r:.6f}"]
    )

    txt_path = output_dir / f"cycle_{origin}_{hemisphere}.txt"

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Saved CSV: %s", csv_path)
    logger.info("Saved TXT summary: %s", txt_path)

    return len(rows)


def _run_origin_hemisphere(
    graph: NeuromapsGraph, origin: str, hemisphere: Hemisphere, output_dir: Path
) -> int:
    """Run all cycles for one origin and hemisphere via the shared engine."""
    work_dir = resolve_artifact_dir(output_dir / f"work_{origin}_{hemisphere}")

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

    results = run_cycle_test(
        graph,
        origin,
        metric_file,
        hemisphere,
        workdir=work_dir,
        max_length=MAX_CYCLE_LENGTH,
        allow_revisits=True,
    )

    logger.info("Executed %d cycles for %s (%s)", len(results), origin, hemisphere)

    rows = [
        {
            "path": " -> ".join(result.path),
            "n_hops": len(result.path) - 1,
            "pearson_r": result.pearson_r,
            "max_abs_diff": result.max_abs_diff,
        }
        for result in results
    ]

    return _save_cycle_results(
        origin=origin, hemisphere=hemisphere, rows=rows, output_dir=output_dir
    )


# -------------------------------------------------------------------------
# Test suite
# -------------------------------------------------------------------------


class TestCycleRoundtrip:
    """Round-trip cycle regression on the real Neuromaps-PRIME graph.

    One class-scoped run propagates deterministic seed metrics through every
    executable surface-transformation cycle (all origins, both hemispheres)
    and records this run's artifacts. The suite is a pure producer: it reads
    no previous baseline and makes no cross-run comparison — those are CI
    concerns.

    Outputs land under the pytest temporary directory
    (``<base>/cycle_outputs``) unless the ``NEUROMAPS_CYCLE_OUTPUT_DIR``
    environment variable is set; the resolved location is logged at run
    start. Point the variable at a persistent location to accumulate run
    summaries across runs.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def cycle_run(
        cls, graph: NeuromapsGraph, tmp_path_factory: pytest.TempPathFactory
    ) -> CycleRunResult:
        """Execute every cycle once and record this run's artifacts."""
        output_dir = _resolve_output_dir(tmp_path_factory.getbasetemp())

        logger.info("Cycle outputs will be written to: %s", output_dir)

        origins = sorted(graph.nodes)

        for origin in origins:
            for hemisphere in HEMISPHERES:
                logger.info(
                    "\n=== CYCLE ROUND-TRIP RUN: %s (%s) ===", origin, hemisphere
                )

                _run_origin_hemisphere(
                    graph=graph,
                    origin=origin,
                    hemisphere=hemisphere,
                    output_dir=output_dir,
                )

        species_by_origin = {
            origin: graph.get_node_data(origin).species for origin in origins
        }

        frames = _read_cycle_frames(run_dir=output_dir, origins=origins)

        summary = _summarize(frames=frames, species_by_origin=species_by_origin)

        frame = _summary_frame(summary.origin_rows + summary.all_rows)

        summary_name = f"cycle_{datetime.now():%Y%m%d_%H%M%S}.csv"
        summary_path = output_dir / summary_name

        frame.to_csv(summary_path, index=False)

        logger.info("Saved run summary CSV: %s", summary_path)

        _save_all_summary_txt(run_dir=output_dir, summaries=summary.summaries)

        logger.info(
            "NEW PEARSON R VALUES: left=%.6f, right=%.6f",
            summary.pearson_r.get(("all", "left"), float("nan")),
            summary.pearson_r.get(("all", "right"), float("nan")),
        )

        return CycleRunResult(output_dir=output_dir, frames=frames)

    def test_cycles_executed(self, cycle_run: CycleRunResult) -> None:
        """At least one transformation cycle executed end-to-end."""
        executed = sum(len(frame) for frame in cycle_run.frames.values())

        assert executed > 0, (
            "No executable surface transformation cycles were found for any "
            "origin/hemisphere."
        )

    def test_cycle_metrics_well_formed(self, cycle_run: CycleRunResult) -> None:
        """Every reported cycle has a finite, in-range correlation."""
        for (origin, hemisphere), frame in cycle_run.frames.items():
            assert not frame.empty

            assert frame["pearson_r"].between(-1.0, 1.0).all(), (
                f"Pearson r out of [-1, 1] for {origin} ({hemisphere})."
            )

            assert np.isfinite(frame["max_abs_diff"]).all(), (
                f"Non-finite max_abs_diff for {origin} ({hemisphere}) — "
                "degenerate metric (e.g. all-NaN round-trip)."
            )
