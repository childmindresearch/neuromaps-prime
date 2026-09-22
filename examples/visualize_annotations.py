"""Visualize surface annotations on anatomical surfaces."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "nibabel",
#     "nilearn",
#     "numpy",
#     "pyyaml",
# ]
# ///

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml
from matplotlib import colormaps
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d.axes3d import Axes3D
from nilearn import plotting

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.graph.models import SurfaceAtlas

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

SURFACE_PRIORITY = ["midthickness", "white", "pial", "inflated", "sphere"]

N_PAIRS_PER_ROW = 4
PAIR_WIDTH = 5.2
PAIR_HEIGHT = 4.8


# ---------------------------------------------------------------------------
# Space / YAML discovery
# ---------------------------------------------------------------------------


def get_space_yaml(space_name: str) -> dict:
    """Find and load the YAML metadata for a space.

    Spaces are discovered recursively under:
        src/neuromaps_prime/resources/nodes/

    This allows spaces to live in species-specific subdirectories such as
    macaque, human, chimpanzee, etc.
    """
    resources_root = REPO_ROOT / "src" / "neuromaps_prime" / "resources" / "nodes"

    matches = sorted(resources_root.rglob(f"{space_name}.yaml"))

    if not matches:
        available = sorted(path.stem for path in resources_root.rglob("*.yaml"))

        raise FileNotFoundError(
            f"Could not find YAML metadata for space "
            f"'{space_name}' under {resources_root}.\n"
            f"Available spaces: {', '.join(available)}"
        )

    if len(matches) > 1:
        locations = "\n".join(
            f"  - {path.relative_to(resources_root)}" for path in matches
        )

        raise ValueError(
            f"Found multiple YAML files for space "
            f"'{space_name}':\n"
            f"{locations}\n"
            "The space name must uniquely identify one YAML file."
        )

    yaml_path = matches[0]

    logger.info("Space metadata: %s", yaml_path.relative_to(resources_root))

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    return data.get(space_name, data)


def get_surface_resolutions(space: dict) -> list[str]:
    """Return surface resolutions that contain annotations."""
    surfaces = space.get("surfaces", {})
    resolutions = []

    for density, density_data in surfaces.items():
        if isinstance(density_data, dict) and "annotation" in density_data:
            resolutions.append(density)

    def sort_key(density: str) -> tuple[int, int | str]:
        if density == "10k":
            return (0, 0)

        if density == "32k":
            return (1, 0)

        try:
            return (2, int(density.rstrip("k")))
        except ValueError:
            return (3, density)

    return sorted(resolutions, key=sort_key)


def get_annotations(space: dict, density: str) -> list[str]:
    """Return annotation names for a surface density."""
    density_data = space["surfaces"][density]
    annotations = density_data.get("annotation", {})

    if isinstance(annotations, dict):
        return list(annotations)

    return []


# ---------------------------------------------------------------------------
# Graph resources
# ---------------------------------------------------------------------------


def get_anatomical_surface(
    graph: NeuromapsGraph, space_name: str, density: str, hemisphere: str
) -> SurfaceAtlas:
    """Find the highest-priority available anatomical surface."""
    for resource_type in SURFACE_PRIORITY:
        surface = graph.fetch_surface_atlas(
            space=space_name,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )

        if surface is not None:
            return surface

    raise ValueError(
        f"No anatomical surface found for {space_name} {density} {hemisphere}"
    )


# ---------------------------------------------------------------------------
# GIFTI loading
# ---------------------------------------------------------------------------


def load_surface(resource: SurfaceAtlas) -> tuple[np.ndarray, np.ndarray]:
    """Load coordinates and faces from a GIFTI surface resource."""
    path = resource.fetch()

    if path.suffix.lower() == ".png":
        raise ValueError(f"Skipping PNG resource: {path}")

    image = nib.load(path)

    coordinates = None
    faces = None

    for darray in image.darrays:
        data = np.asarray(darray.data)

        if (
            data.ndim == 2
            and data.shape[1] == 3
            and np.issubdtype(data.dtype, np.floating)
        ):
            coordinates = data

        elif (
            data.ndim == 2
            and data.shape[1] == 3
            and np.issubdtype(data.dtype, np.integer)
        ):
            faces = data

    if coordinates is None:
        raise ValueError(f"Could not find surface coordinates in {path}")

    if faces is None:
        raise ValueError(f"Could not find surface faces in {path}")

    return coordinates, faces


def load_annotation(resource: SurfaceAtlas, n_vertices: int) -> np.ndarray:
    """Load an annotation from a GIFTI resource.

    Multi-map GIFTIs are reduced to their first map.
    PNG resources are explicitly rejected.
    """
    path = resource.fetch()

    if path.suffix.lower() == ".png":
        raise ValueError(f"Skipping PNG annotation: {path}")

    image = nib.load(path)

    candidates = []

    for darray in image.darrays:
        data = np.asarray(darray.data)

        if data.ndim == 1 and data.shape[0] == n_vertices:
            candidates.append(data)

        elif data.ndim == 2 and data.shape[0] == n_vertices:
            logger.info(
                "    %s: multi-map data %s; using map 0", resource.name, data.shape
            )
            candidates.append(data[:, 0])

        elif data.ndim == 2 and data.shape[1] == n_vertices:
            logger.info(
                "    %s: multi-map data %s; using map 0", resource.name, data.shape
            )
            candidates.append(data[0, :])

    if not candidates:
        raise ValueError(
            f"Could not find annotation data with {n_vertices} vertices in {path}"
        )

    return np.asarray(candidates[0])


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


def plot_resolution(
    graph: NeuromapsGraph, space_name: str, space: dict, density: str
) -> None:
    """Create one annotation figure for a surface density."""
    annotations = get_annotations(space, density)

    if not annotations:
        logger.info("No annotations found for %s", density)
        return

    logger.info("Processing %s %s", space_name, density)
    logger.info("Found %d annotations", len(annotations))

    # ------------------------------------------------------------------
    # Load anatomical surfaces independently for each hemisphere.
    # ------------------------------------------------------------------

    surfaces = {}

    for hemisphere in ("left", "right"):
        try:
            surface_resource = get_anatomical_surface(
                graph, space_name, density, hemisphere
            )

            logger.info(
                "  %s: using %s surface", hemisphere, surface_resource.resource_type
            )

            coordinates, faces = load_surface(surface_resource)

            surfaces[hemisphere] = {
                "coordinates": coordinates,
                "faces": faces,
                "resource": surface_resource,
            }

        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.error("  ERROR loading %s anatomical surface: %s", hemisphere, exc)

    if "left" not in surfaces or "right" not in surfaces:
        logger.error("Skipping %s: both hemispheres are required.", density)
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
        # Fetch annotation resources.
        # --------------------------------------------------------------

        try:
            left_resource = graph.fetch_surface_annotation(
                space=space_name, label=label, density=density, hemisphere="left"
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.error("  ERROR fetching %s left: %s", label, exc)
            left_resource = None

        try:
            right_resource = graph.fetch_surface_annotation(
                space=space_name, label=label, density=density, hemisphere="right"
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.error("  ERROR fetching %s right: %s", label, exc)
            right_resource = None

        # --------------------------------------------------------------
        # Load annotation values.
        # --------------------------------------------------------------

        left_values = None
        right_values = None

        if left_resource is not None:
            try:
                left_values = load_annotation(
                    left_resource, len(surfaces["left"]["coordinates"])
                )
            except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
                logger.error("  ERROR loading %s left: %s", label, exc)

        if right_resource is not None:
            try:
                right_values = load_annotation(
                    right_resource, len(surfaces["right"]["coordinates"])
                )
            except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
                logger.error("  ERROR loading %s right: %s", label, exc)

        if left_values is None and right_values is None:
            left_ax.set_axis_off()
            right_ax.set_axis_off()
            continue

        # --------------------------------------------------------------
        # Determine map type.
        # --------------------------------------------------------------

        categorical = False

        if left_values is not None:
            categorical = annotation_is_categorical(label, left_values)

        if right_values is not None:
            categorical = categorical or annotation_is_categorical(label, right_values)

        cmap = cmap_categorical if categorical else cmap_continuous

        # --------------------------------------------------------------
        # Left hemisphere.
        # --------------------------------------------------------------

        if left_values is not None:
            try:
                plot_surface_map(
                    left_ax,
                    surfaces["left"]["coordinates"],
                    surfaces["left"]["faces"],
                    left_values,
                    "left",
                    categorical=categorical,
                    cmap=cmap,
                )
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.error("  ERROR plotting %s left: %s", label, exc)
                left_ax.set_axis_off()
        else:
            left_ax.set_axis_off()

        # --------------------------------------------------------------
        # Right hemisphere.
        # --------------------------------------------------------------

        if right_values is not None:
            try:
                plot_surface_map(
                    right_ax,
                    surfaces["right"]["coordinates"],
                    surfaces["right"]["faces"],
                    right_values,
                    "right",
                    categorical=categorical,
                    cmap=cmap,
                )
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.error("  ERROR plotting %s right: %s", label, exc)
                right_ax.set_axis_off()
        else:
            right_ax.set_axis_off()

    # ------------------------------------------------------------------
    # Save figure.
    # ------------------------------------------------------------------

    output_path = (
        REPO_ROOT / "examples" / f"visualize_{space_name}_{density}_annotations.png"
    )

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
    parser.add_argument("space", help="Space to visualize, e.g. Yerkes19")

    args = parser.parse_args()
    space_name = args.space

    logger.info("Initializing NeuromapsGraph...")

    graph = NeuromapsGraph()
    space = get_space_yaml(space_name)

    resolutions = get_surface_resolutions(space)

    if not resolutions:
        raise ValueError(f"No annotated surface resolutions found for {space_name}")

    logger.info("Space: %s", space_name)
    logger.info("Surface resolutions: %s", ", ".join(resolutions))

    for density in resolutions:
        plot_resolution(graph, space_name, space, density)


if __name__ == "__main__":
    main()
