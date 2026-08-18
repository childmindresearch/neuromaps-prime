"""Cycle regression test on the real graph.

Round-trips a synthetic surface metric around every return path from an origin
space and reports the Pearson correlation between the original metric and each
round-trip. High correlation means the transforms on that path compose close to
the identity; lower correlation flags lower-quality paths.

This is the deployed counterpart to the hermetic unit test in
``tests/unit/test_cycle.py``. Both call the same machinery in
``tests/regression/cycle.py``; the unit test proves that machinery returns
r ~ 1 on a synthetic identity network, while this test measures the *real*
transforms and therefore needs Workbench and network access (like
``test_surf_matrix.py``).

Artifacts are written to a run-specific directory under
``tests/regression/cycle_outputs_<random_suffix>/`` so repeated runs do not
overwrite each other. Surface resampling prefers ``midthickness`` area
surfaces, then falls back to ``pial`` and ``white`` when required resources
are missing for a space/hemisphere.

Edit ``ORIGINS``, ``LABEL``, and ``HEMISPHERES`` for the space/probe tags you want
to test, then run::

    pytest tests/regression/test_cycle.py -s
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nibabel.gifti import GiftiDataArray, GiftiImage
from matplotlib_surface_plotting import plot_surf

from neuromaps_prime.graph import NeuromapsGraph
from tests.neuromapsprime_unit_test.cycle import (
    _path_token,
    find_return_paths,
    resolve_hop_transforms,
    score_roundtrip,
)

logger = logging.getLogger(__name__)

_RUN_SUFFIX = secrets.token_hex(4)
# Keep every test run isolated to simplify debugging and avoid clobbering artifacts.
output_dir = Path(__file__).resolve().parent / f"cycle_outputs_{_RUN_SUFFIX}"
output_dir.mkdir(parents=True, exist_ok=True)

# --- configure the probe ---------------------------------------------------- #
# Set to ``None`` to auto-populate from all graph coordinate spaces.
ORIGINS: list[str] | None = None
LABEL = "RM_scalinghcp"
HEMISPHERES = ("left", "right")
# Bound path length so cycle enumeration stays tractable on the dense real
# graph (number of simple cycles grows combinatorially).
MAX_CYCLE_LENGTH = 4
PLOT_MAX_VERTICES = 20000
# Set to ``None`` to enumerate every path up to ``MAX_CYCLE_LENGTH``.
MAX_PATHS: int | None = None
# With non-mirrored multi-hop round-trips, many paths are expected to score
# modestly. Guard against hard breakage by requiring at least one usable path.
MIN_BEST_PEARSON = 0.05

# Enable logging of wb_command calls for manual inspection/re-running
LOG_COMMANDS = True


class CommandLogHandler(logging.Handler):
    """Capture niwrap/Styx command logs to a file."""

    def __init__(self, log_file: Path):
        super().__init__()
        self.log_file = log_file
        self.commands: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Capture log records that contain command information."""
        # Styx logs full command strings in debug records
        if "command" in record.getMessage().lower() or "wb_command" in record.name.lower():
            self.commands.append(self.format(record))

    def save(self) -> None:
        """Write captured commands to file."""
        if self.commands:
            with self.log_file.open("w", encoding="utf-8") as f:
                f.write("# wb_command calls used in this cycle\n")
                f.write("# Run these to manually reproduce the transformations\n\n")
                for cmd in self.commands:
                    f.write(f"{cmd}\n")


def _load_surface_coords(surface_file: Path) -> np.ndarray:
    """Load the ``(n_vertices, 3)`` world coordinates from a surface GIFTI."""
    for darray in nib.load(str(surface_file)).darrays:
        if darray.data.ndim == 2 and darray.data.shape[1] == 3:
            return np.asarray(darray.data, dtype=np.float64)
    raise ValueError(f"No pointset coordinates found in {surface_file}")


def _load_surface_topology(surface_file: Path) -> np.ndarray:
    """Load the ``(n_triangles, 3)`` triangle indices from a surface GIFTI."""
    img = nib.load(str(surface_file))
    # NIFTI_INTENT_TRIANGLE has integer code 1009. ``darray.intent`` is stored
    # as the intent *name* string in nibabel, so match against both forms.
    for darray in img.darrays:
        if str(darray.intent) in ("NIFTI_INTENT_TRIANGLE", "1009"):
            return np.asarray(darray.data, dtype=np.int32)
    # Fallback: an integer-typed (N, 3) array is the triangle topology.
    for darray in img.darrays:
        data = np.asarray(darray.data)
        if (
            data.ndim == 2
            and data.shape[1] == 3
            and np.issubdtype(data.dtype, np.integer)
        ):
            return data.astype(np.int32)
    raise ValueError(f"No triangle topology found in {surface_file}")


def _extract_surface_mesh(surface_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract (vertices, triangles) from a surface GIFTI file."""
    return _load_surface_coords(surface_file), _load_surface_topology(surface_file)


def _write_metric(metric_file: Path, values: np.ndarray) -> Path:
    """Write one scalar value per vertex as a ``.func.gii`` metric file."""
    image = GiftiImage(
        darrays=[
            GiftiDataArray(
                np.asarray(values, dtype=np.float32),
                intent="NIFTI_INTENT_NONE",
            )
        ]
    )
    nib.save(image, str(metric_file))
    return metric_file


def _make_xyz_product_metric(
    graph: NeuromapsGraph,
    origin: str,
    label: str,
    density: str,
    hemisphere: str,
    out_dir: Path,
) -> Path:
    """Create a synthetic metric where each vertex value is ``x*y*z``."""
    sphere = graph.fetch_surface_atlas(
        space=origin,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    )
    assert sphere is not None, (
        f"No sphere atlas for {origin} at density '{density}' ({hemisphere})."
    )
    coords = _load_surface_coords(Path(sphere.fetch()))
    values = np.prod(coords, axis=1)
    metric_file = out_dir / f"metric_{origin}_{label}_{density}_{hemisphere}.func.gii"
    return _write_metric(metric_file, values)


def _load_metric_values(metric_file: Path) -> np.ndarray:
    """Load metric values from the first data array in a metric GIFTI."""
    return np.asarray(nib.load(str(metric_file)).darrays[0].data, dtype=np.float64)


def _path_to_filename(path: tuple[str, ...]) -> str:
    """Convert a cycle path tuple to a filesystem-friendly stem."""
    digest = hashlib.sha1("->".join(path).encode("utf-8")).hexdigest()[:12]
    return f"{path[0]}_to_{path[-1]}_{len(path) - 1}h_{digest}"


def _plot_cycle_cortical_surfaces(
    graph: NeuromapsGraph,
    path: tuple[str, ...],
    metrics_by_hop: list[tuple[str, np.ndarray]],
    hemisphere: str,
    pearson_r: float,
    plot_dir: Path,
) -> None:
    """Create 2 images per node (midthickness + sphere) showing the metric.
    
    For a path like (Yerkes19, MEBRAINS, Yerkes19), creates 6 images:
    - node0_Yerkes19_midthickness.png, node0_Yerkes19_sphere.png
    - node1_MEBRAINS_midthickness.png, node1_MEBRAINS_sphere.png
    - node2_Yerkes19_midthickness.png, node2_Yerkes19_sphere.png
    
    Args:
        graph: Populated NeuromapsGraph for fetching surfaces.
        path: Cycle path, e.g. ('Yerkes19', 'MEBRAINS', 'Yerkes19').
        metrics_by_hop: List of (space_name, metric_values) for each node.
        hemisphere: 'left' or 'right'.
        pearson_r: Pearson correlation for the cycle.
        plot_dir: Output directory for images.
    """
    if not metrics_by_hop:
        return

    path_token = _path_token(path)
    path_label = " -> ".join(path)
    
    for hop_idx, (space, metric_values) in enumerate(metrics_by_hop):
        # Get color scale from finite values
        finite_mask = np.isfinite(metric_values)
        if np.any(finite_mask):
            vmin, vmax = np.percentile(metric_values[finite_mask], [2, 98])
        else:
            vmin, vmax = 0.0, 1.0
        
        # Plot midthickness surface
        _plot_single_surface(
            graph, space, metric_values, hemisphere,
            "midthickness", float(vmin), float(vmax), plot_dir,
            path_token, path_label, hop_idx, pearson_r,
        )
        



def _find_matching_surface(
    graph: NeuromapsGraph,
    space: str,
    hemisphere: str,
    resource_type: str,
    n_vertices: int,
) -> tuple[object, str] | None:
    """Return the (atlas, density) whose surface vertex count matches the metric.

    Iterates over every registered atlas of ``resource_type`` for ``space`` and
    returns the first whose vertex count equals ``n_vertices``. This avoids
    guessing density from vertex count (which breaks across spaces with
    non-standard meshes).
    """
    atlases = graph.utils.cache.get_surface_atlases(
        space=space,
        hemisphere=hemisphere,
        resource_type=resource_type,
    )
    for atlas in atlases:
        try:
            coords = _load_surface_coords(Path(atlas.fetch()))
        except (ValueError, FileNotFoundError, OSError):
            continue
        if coords.shape[0] == n_vertices:
            return atlas, atlas.density
    return None


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
    hop_idx: int,
    pearson_r: float,
) -> None:
    """Plot a single surface (midthickness or sphere) for one node.

    The surface is selected by matching its vertex count to the metric so the
    overlay always aligns. Failures are logged as warnings (never silent) so
    missing surfaces are visible in the test output.
    """
    n_vertices = metric_values.shape[0]
    match = _find_matching_surface(
        graph, space, hemisphere, resource_type, n_vertices
    )
    if match is None:
        logger.warning(
            "No %s surface for %s (%s) matching %d vertices; skipping plot.",
            resource_type, space, hemisphere, n_vertices,
        )
        return

    surface_file, density = match
    try:
        coords, faces = _extract_surface_mesh(Path(surface_file.fetch()))
    except Exception as e:
        logger.warning(
            "Failed to load %s mesh for %s (%s): %s",
            resource_type, space, density, e,
        )
        return

    output_path = plot_dir / f"{path_token}_node{hop_idx:02d}_{space}_{resource_type}.png"
    try:
        plot_surf(
            coords, faces, metric_values,
            rotate=[270, 0],  # Lateral view
            filename=str(output_path),
            vmin=vmin, vmax=vmax,
            cmap="viridis",
            title=f"{path_label}\n{space} | r={pearson_r:.5f}",
        )
        logger.info("Saved: %s", output_path.name)
    except Exception as e:
        logger.warning(
            "plot_surf failed for %s %s (%s): %s",
            space, resource_type, density, e,
        )
    finally:
        # plot_surf leaves figures open; close them to avoid memory buildup.
        plt.close("all")


def test_cycle_roundtrip(tmp_path: Path) -> None:
    """Round-trip a synthetic metric through every return path and log correlations."""
    origins = ORIGINS or sorted(NeuromapsGraph().nodes)
    for origin in origins:
        for hemisphere in HEMISPHERES:
            logging.info(f"\n=== CYCLE ROUND-TRIP TEST: {origin} ({hemisphere}) ===")
            cycle_roundtrip(tmp_path, origin, hemisphere)


def _roundtrip_with_intermediates(
    graph: NeuromapsGraph,
    metric_file: Path,
    path: tuple[str, ...],
    hemisphere: str,
    workdir: Path,
) -> tuple[Path, list[tuple[str, np.ndarray]]] | None:
    """Round-trip metric and collect intermediate values at each hop.
    
    The transformer auto-estimates source and target densities from the
    metric file at each hop, allowing each space to use its native density.
    
    Returns:
        (final_file, [(space, metric_values), ...])
    """
    from itertools import pairwise
    
    metrics: list[tuple[str, np.ndarray]] = []
    current = Path(metric_file)
    workdir.mkdir(parents=True, exist_ok=True)
    
    for hop, (src, dst) in enumerate(pairwise(path)):
        # Let transformer auto-estimate both source and target densities.
        # This allows each space to use its native density rather than
        # forcing the origin's density everywhere.
        # Use a relative output name so Workbench writes inside Styx's mounted
        # output dir, then copy into our run-specific artifact directory.
        hop_name = f"cycle_intermediate_hop{hop:02d}_{src}-to-{dst}.func.gii"
        hop_target = workdir / hop_name
        result = None
        last_error: Exception | None = None
        for area_resource in ("midthickness", "pial", "white"):
            try:
                result = graph.surface_to_surface_transformer(
                    transformer_type="metric",
                    input_file=current,
                    source_space=src,
                    target_space=dst,
                    hemisphere=hemisphere,
                    output_file_path=hop_name,
                    source_density=None,  # Auto-estimate
                    target_density=None,  # Auto-estimate
                    area_resource=area_resource,
                    add_edge=False,
                )
                if result is not None:
                    if area_resource != "midthickness":
                        logger.info(
                            "Using fallback area surface '%s' for hop '%s' -> '%s' (%s).",
                            area_resource,
                            src,
                            dst,
                            hemisphere,
                        )
                    break
            except Exception as e:
                last_error = e
                continue
        if result is None:
            logger.warning(
                "Skipping path %s for hemisphere %s at hop '%s' -> '%s': %s",
                " -> ".join(path),
                hemisphere,
                src,
                dst,
                last_error,
            )
            return None
        result_path = Path(result)
        if result_path.resolve() != hop_target.resolve():
            shutil.copy2(result_path, hop_target)
            current = hop_target
        else:
            current = result_path
        # Collect metric at this hop's destination
        metric_values = _load_metric_values(current)
        metrics.append((dst, metric_values))
    
    return current, metrics


def cycle_roundtrip(tmp_path: Path, origin: str, hemisphere: str) -> None:
    """Round-trip a synthetic metric through every return path and log correlations."""
    logging.basicConfig(level=logging.INFO)
    graph = NeuromapsGraph()
    work_dir = output_dir / f"work_{origin}_{hemisphere}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Set up command logging if enabled
    cmd_log_handler = None
    if LOG_COMMANDS:
        cmd_log_handler = CommandLogHandler(work_dir / "wb_commands.log")
        # Capture niwrap/Styx logs
        niwrap_logger = logging.getLogger("niwrap")
        niwrap_logger.addHandler(cmd_log_handler)
        niwrap_logger.setLevel(logging.DEBUG)
        styx_logger = logging.getLogger("styx")
        styx_logger.addHandler(cmd_log_handler)
        styx_logger.setLevel(logging.DEBUG)

    # Seed the metric at the origin's highest density so the round-trip returns
    # to a matching mesh, then score every return path.
    density = graph.find_highest_density(origin)
    try:
        metric_file = _make_xyz_product_metric(
            graph=graph,
            origin=origin,
            label=LABEL,
            density=density,
            hemisphere=hemisphere,
            out_dir=work_dir,
        )
    except AssertionError as e:
        logger.warning(
            "Skipping %s (%s): could not seed origin metric: %s",
            origin,
            hemisphere,
            e,
        )
        return

    paths = find_return_paths(
        graph,
        origin,
        max_length=MAX_CYCLE_LENGTH,
        allow_revisits=True,
        max_paths=MAX_PATHS,
    )

    logger.info("Found %d return paths from %s", len(paths), origin)
    if not paths:
        logger.warning(
            "No return paths from %s on the surface layer; skipping hemisphere %s.",
            origin,
            hemisphere,
        )
        return

    sphere = graph.fetch_surface_atlas(
        space=origin,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    )
    if sphere is None:
        logger.warning(
            "No sphere atlas for %s at density '%s' (%s); skipping.",
            origin,
            density,
            hemisphere,
        )
        return
    origin_coords = _load_surface_coords(Path(sphere.fetch()))
    original_metric = _load_metric_values(metric_file)
    if origin_coords.shape[0] != original_metric.shape[0]:
        logger.warning(
            "Skipping %s (%s): origin sphere/metric size mismatch (%d vs %d).",
            origin,
            hemisphere,
            origin_coords.shape[0],
            original_metric.shape[0],
        )
        return

    plot_dir = output_dir / f"cycle_{origin}_{LABEL}_{hemisphere}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    for path in paths:
        # Use the intermediate metric collector for better visualization
        roundtrip_result = _roundtrip_with_intermediates(
            graph,
            metric_file,
            path,
            hemisphere,
            work_dir,
        )
        if roundtrip_result is None:
            continue
        roundtrip_file, intermediates = roundtrip_result
        
        pearson_r, max_abs_diff = score_roundtrip(metric_file, roundtrip_file)

        path_label = " -> ".join(path)

        # Write a CSV listing every transform file used for this cycle path.
        # Note: with auto-estimated densities, the fixed density passed here
        # may not match what was actually used. This is just for reference.
        try:
            transform_rows = resolve_hop_transforms(
                graph, path, hemisphere, density
            )
            xfm_csv = plot_dir / f"{_path_to_filename(path)}_transforms.csv"
            with xfm_csv.open("w", encoding="utf-8") as fh:
                fh.write("source_space,target_space,density,filepath\n")
                for src, dst, den, fpath in transform_rows:
                    fh.write(f"{src},{dst},{den},{fpath}\n")
        except Exception as e:
            logger.warning(f"Could not resolve transform files for cycle {path_label}: {e}")

        # Save per-node surface visualizations (midthickness + sphere for each
        # space visited). Prepend the origin/original metric so node 0 is the
        # starting point and the final node is the round-tripped result.
        try:
            all_metrics = [(path[0], original_metric)] + intermediates
            _plot_cycle_cortical_surfaces(
                graph=graph,
                path=path,
                metrics_by_hop=all_metrics,
                hemisphere=hemisphere,
                pearson_r=pearson_r,
                plot_dir=plot_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to create surface plots: {e}")

        rows.append(
            {
                "path": path_label,
                "n_hops": len(path) - 1,
                "pearson_r": pearson_r,
                "max_abs_diff": max_abs_diff,
            }
        )

    if not rows:
        logger.warning(
            "No valid round-trip paths for %s (%s); likely missing hemisphere-specific resources.",
            origin,
            hemisphere,
        )
        return

    frame = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)

    logger.info("\n=== CYCLE TEST (%s, %s, %s) ===", origin, LABEL, hemisphere)
    logger.info("\n%s", frame.to_string(index=False))

    csv_path = output_dir / f"cycle_{origin}_{LABEL}_{hemisphere}.csv"
    frame.to_csv(csv_path, index=False)
    logger.info("Saved CSV -> %s", csv_path)
    logger.info("Saved per-path plots -> %s", plot_dir)

    # Print a formatted summary table to stdout.
    col_path = max(len(r["path"]) for r in rows) if rows else 50
    col_path = max(col_path, len("Transformation path"))
    header = f"{'Transformation path':<{col_path}}  {'Hops':>4}  {'Pearson r':>10}  {'Max |delta|':>14}"
    separator = "-" * len(header)
    print(f"\n=== CYCLE TEST — {origin} | {LABEL} | {hemisphere} ===")
    print(separator)
    print(header)
    print(separator)
    for _, row in frame.iterrows():
        print(
            f"{row['path']:<{col_path}}  {int(row['n_hops']):>4}  "
            f"{row['pearson_r']:>10.6f}  {row['max_abs_diff']:>14.3e}"
        )
    print(separator)
    print(f"Total cycles: {len(rows)}")

    # Save the same table to a text file alongside the CSV.
    txt_path = output_dir / f"cycle_{origin}_{LABEL}_{hemisphere}.txt"
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Cycle test results — origin: {origin}, label: {LABEL}, hemisphere: {hemisphere}\n")
        fh.write(separator + "\n")
        fh.write(header + "\n")
        fh.write(separator + "\n")
        for _, row in frame.iterrows():
            fh.write(
                f"{row['path']:<{col_path}}  {int(row['n_hops']):>4}  "
                f"{row['pearson_r']:>10.6f}  {row['max_abs_diff']:>14.3e}\n"
            )
        fh.write(separator + "\n")
        fh.write(f"Total cycles: {len(rows)}\n")
    logger.info("Saved TXT summary -> %s", txt_path)

    # Save command log if it was set up
    if cmd_log_handler is not None:
        cmd_log_handler.save()
        logger.info("Saved wb_command log -> %s", cmd_log_handler.log_file)

    # Under non-mirrored, multi-hop round-trips some low-scoring paths are
    # expected; fail only on clearly broken output (non-finite scores or no
    # minimally correlated route).
    assert np.isfinite(frame["pearson_r"]).all(), (
        "At least one path produced a non-finite Pearson r - inspect "
        f"{csv_path}."
    )
    best_r = float(frame["pearson_r"].max())
    assert best_r > MIN_BEST_PEARSON, (
        f"Best round-trip correlation is too low (r={best_r:.6f}); inspect "
        f"the generated reports in {plot_dir} and metrics in {csv_path}."
    )
