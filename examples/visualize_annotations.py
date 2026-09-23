"""Visualize surface annotations on anatomical surfaces.

Usage:
    uv run python examples/visualize_annotations.py Yerkes19

    or

    uv run python examples/visualize_annotations.py Yerkes19 --output-dir /figures
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "nibabel",
#     "nilearn",
#     "numpy",
# ]
# ///

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d.axes3d import Axes3D
from nibabel.filebasedimages import ImageFileError
from nilearn import plotting

from neuromaps_prime.analysis.images import load_data
from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.graph.models import SurfaceAnnotation, SurfaceAtlas

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------

SURFACE_PRIORITY = ("midthickness", "white", "pial", "inflated", "sphere")

N_PAIRS_PER_ROW = 4
PAIR_WIDTH = 5.2
PAIR_HEIGHT = 4.8

# ---------------------------------------------------------------------------
# Graph resources
# ---------------------------------------------------------------------------


def get_anatomical_surfaces(
    graph: NeuromapsGraph, space_name: str, density: str
) -> tuple[SurfaceAtlas, SurfaceAtlas]:
    """Find the highest-priority anatomical surface available for both hemispheres."""
    for resource_type in SURFACE_PRIORITY:
        left = graph.fetch_surface_atlas(
            space=space_name,
            density=density,
            hemisphere="left",
            resource_type=resource_type,
        )

        right = graph.fetch_surface_atlas(
            space=space_name,
            density=density,
            hemisphere="right",
            resource_type=resource_type,
        )

        if left is not None and right is not None:
            return left, right

    raise ValueError(
        f"No matching anatomical surfaces found for {space_name} {density}"
    )


# ---------------------------------------------------------------------------
# GIFTI loading
# ---------------------------------------------------------------------------


def load_annotation(resource: SurfaceAnnotation, n_vertices: int) -> np.ndarray:
    """Load an annotation from a GIFTI resource."""
    fetched = resource.fetch()
    data = load_data(fetched).array

    if isinstance(data, tuple):
        if data and all(
            isinstance(values, np.ndarray) and values.shape == (n_vertices,)
            for values in data
        ):
            logger.info("    %s: %d maps; using map 0", resource.name, len(data))
            return data[0]

        raise ValueError(
            f"Could not find annotation data with {n_vertices} vertices in {fetched}"
        )

    if data.ndim == 1 and data.shape[0] == n_vertices:
        return data

    if data.ndim == 2:
        if data.shape[0] == n_vertices:
            logger.info(
                "    %s: multi-map data %s; using map 0", resource.name, data.shape
            )
            return data[:, 0]

        if data.shape[1] == n_vertices:
            logger.info(
                "    %s: multi-map data %s; using map 0", resource.name, data.shape
            )
            return data[0, :]

    raise ValueError(
        f"Could not find annotation data with {n_vertices} vertices in {fetched}"
    )


# ---------------------------------------------------------------------------
# Annotation classification
# ---------------------------------------------------------------------------


def annotation_is_categorical(name: str, values: np.ndarray) -> bool:
    """Determine whether an annotation should be plotted as ROI labels.

    PC_* resources are atlas/parcellation-style annotations and are therefore
    categorical. Other resources are treated as continuous unless their
    values are explicitly integer-valued, non-negative, and have a small
    number of unique labels.
    """
    if name.startswith("PC_"):
        return True

    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        return False

    unique = np.unique(finite)

    return (
        np.issubdtype(values.dtype, np.integer)
        and len(unique) <= 20
        and np.all(unique >= 0)
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_surface_map(
    ax: Axes3D,
    coordinates: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    hemisphere: str,
    *,
    categorical: bool,
    cmap: Colormap,
) -> None:
    """Plot either a categorical ROI map or a continuous surface map."""
    surface = (coordinates, faces)

    if categorical:
        plotting.plot_surf_roi(
            surface,
            roi_map=values,
            hemi=hemisphere,
            view="lateral",
            bg_on_data=True,
            cmap=cmap,
            colorbar=False,
            axes=ax,
        )
    else:
        plotting.plot_surf_stat_map(
            surface,
            stat_map=values,
            hemi=hemisphere,
            view="lateral",
            bg_on_data=True,
            cmap=cmap,
            colorbar=False,
            axes=ax,
        )


def load_annotation_for_hemisphere(
    graph: NeuromapsGraph,
    space_name: str,
    density: str,
    label: str,
    hemisphere: str,
    n_vertices: int,
) -> np.ndarray | None:
    """Fetch and load an annotation for one hemisphere."""
    try:
        resource = graph.fetch_surface_annotation(
            space=space_name, label=label, density=density, hemisphere=hemisphere
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.error("  ERROR fetching %s %s: %s", label, hemisphere, exc)
        return None

    if resource is None:
        logger.info("  No annotation resource for %s %s", label, hemisphere)
        return None

    try:
        return load_annotation(resource, n_vertices)
    except (FileNotFoundError, OSError, ValueError, TypeError, ImageFileError) as exc:
        logger.error("  ERROR loading %s %s: %s", label, hemisphere, exc)
        return None


def plot_annotation_hemisphere(
    ax: Axes3D,
    coordinates: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray | None,
    hemisphere: str,
    label: str,
    *,
    categorical: bool,
    cmap: Colormap,
) -> None:
    """Plot an annotation for one hemisphere."""
    if values is None:
        ax.set_axis_off()
        return

    try:
        plot_surface_map(
            ax,
            coordinates,
            faces,
            values,
            hemisphere,
            categorical=categorical,
            cmap=cmap,
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.error("  ERROR plotting %s %s: %s", label, hemisphere, exc)
        ax.set_axis_off()


def plot_resolution(
    graph: NeuromapsGraph, space_name: str, density: str, output_dir: Path
) -> None:
    """Create one annotation figure for a surface density."""
    annotations = sorted(
        {
            annotation.label
            for annotation in graph.get_node_data(space_name).surface_annotations
            if annotation.density == density
        }
    )

    if not annotations:
        logger.info("No annotations found for %s", density)
        return

    logger.info("Processing %s %s", space_name, density)
    logger.info("Found %d annotations", len(annotations))

    # ------------------------------------------------------------------
    # Load anatomical surfaces independently for each hemisphere.
    # ------------------------------------------------------------------

    surfaces = {}

    try:
        left_resource, right_resource = get_anatomical_surfaces(
            graph, space_name, density
        )

        logger.info(
            "  using %s surface for both hemispheres", left_resource.resource_type
        )

        left_coordinates, left_faces = load_data(left_resource.fetch()).array
        right_coordinates, right_faces = load_data(right_resource.fetch()).array

        surfaces = {
            "left": {
                "coordinates": left_coordinates,
                "faces": left_faces,
                "resource": left_resource,
            },
            "right": {
                "coordinates": right_coordinates,
                "faces": right_faces,
                "resource": right_resource,
            },
        }

    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.error("ERROR loading anatomical surfaces: %s", exc)
        return

    # ------------------------------------------------------------------
    # Figure layout.
    # ------------------------------------------------------------------

    n_pairs = len(annotations)
    n_rows = int(np.ceil(n_pairs / N_PAIRS_PER_ROW))

    fig_width = N_PAIRS_PER_ROW * PAIR_WIDTH
    fig_height = n_rows * PAIR_HEIGHT

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)

    fig.suptitle(
        f"{space_name} — {density} Surface Annotations",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.subplots_adjust(top=0.96)

    outer = fig.add_gridspec(n_rows, N_PAIRS_PER_ROW, wspace=0.02, hspace=0.12)

    cmap_categorical = colormaps["tab20"]
    cmap_continuous = colormaps["viridis"]

    # ------------------------------------------------------------------
    # Plot every annotation.
    # ------------------------------------------------------------------

    for index, label in enumerate(annotations):
        row = index // N_PAIRS_PER_ROW
        col = index % N_PAIRS_PER_ROW

        pair_grid = outer[row, col].subgridspec(
            2, 2, height_ratios=[0.15, 1], wspace=0.01, hspace=0.01
        )

        title_ax = fig.add_subplot(pair_grid[0, :])
        left_ax = fig.add_subplot(pair_grid[1, 0], projection="3d")
        right_ax = fig.add_subplot(pair_grid[1, 1], projection="3d")

        title_ax.axis("off")

        title_ax.text(
            0.5, 0.5, label, ha="center", va="center", fontsize=10, fontweight="bold"
        )

        # --------------------------------------------------------------
        # Fetch and load annotation values.
        # --------------------------------------------------------------

        left_values = load_annotation_for_hemisphere(
            graph,
            space_name,
            density,
            label,
            "left",
            len(surfaces["left"]["coordinates"]),
        )

        right_values = load_annotation_for_hemisphere(
            graph,
            space_name,
            density,
            label,
            "right",
            len(surfaces["right"]["coordinates"]),
        )

        # --------------------------------------------------------------
        # Determine map type.
        # --------------------------------------------------------------

        categorical = False

        if left_values is not None:
            categorical = annotation_is_categorical(label, left_values)

        if right_values is not None:
            categorical = categorical or annotation_is_categorical(label, right_values)

        cmap = cmap_categorical if categorical else cmap_continuous

        plot_annotation_hemisphere(
            left_ax,
            surfaces["left"]["coordinates"],
            surfaces["left"]["faces"],
            left_values,
            "left",
            label,
            categorical=categorical,
            cmap=cmap,
        )

        plot_annotation_hemisphere(
            right_ax,
            surfaces["right"]["coordinates"],
            surfaces["right"]["faces"],
            right_values,
            "right",
            label,
            categorical=categorical,
            cmap=cmap,
        )

    # ------------------------------------------------------------------
    # Save figure.
    # ------------------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"visualize_{space_name}_{density}_annotations.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the surface annotation visualization."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Visualize surface annotations for a neuromaps space."
    )
    parser.add_argument("space", type=str, help="Space to visualize, e.g. Yerkes19")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "examples",
        help="Directory for output figures (default: examples/)",
    )

    args = parser.parse_args()
    space_name = args.space

    logger.info("Initializing NeuromapsGraph...")

    graph = NeuromapsGraph()
    node = graph.get_node_data(space_name)

    resolutions = sorted(
        {annotation.density for annotation in node.surface_annotations}
    )

    if not resolutions:
        raise ValueError(f"No annotated surface resolutions found for {space_name}")

    logger.info("Space: %s", space_name)
    logger.info("Surface resolutions: %s", ", ".join(resolutions))

    for density in resolutions:
        plot_resolution(graph, space_name, density, args.output_dir)


if __name__ == "__main__":
    main()
