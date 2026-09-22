"""Visualize surface annotations on anatomical surfaces."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml
from matplotlib import colormaps
from nilearn import plotting


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

def get_repo_root():
    """Return the repository root from the current command-line location."""
    current = Path.cwd().resolve()

    for path in (current, *current.parents):
        if (
            (path / "pyproject.toml").exists()
            and (path / "src" / "neuromaps_prime").exists()
        ):
            return path

    # Fall back to the repository containing this script. This allows the
    # script to work when invoked with an absolute path from elsewhere.
    script_root = Path(__file__).resolve().parents[1]

    if (
        (script_root / "pyproject.toml").exists()
        and (script_root / "src" / "neuromaps_prime").exists()
    ):
        return script_root

    raise FileNotFoundError(
        "Could not determine the neuromaps-prime repository root."
    )


REPO_ROOT = get_repo_root()
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neuromaps_prime.graph import NeuromapsGraph


# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------

SURFACE_PRIORITY = [
    "midthickness",
    "white",
    "pial",
    "inflated",
    "sphere",
]

N_PAIRS_PER_ROW = 4
PAIR_WIDTH = 5.2
PAIR_HEIGHT = 4.8


# ---------------------------------------------------------------------------
# Space / YAML discovery
# ---------------------------------------------------------------------------

def get_space_yaml(space_name):
    """Find and load the YAML metadata for a space.

    Spaces are discovered recursively under:
        src/neuromaps_prime/resources/nodes/

    This allows spaces to live in species-specific subdirectories such as
    macaque, human, chimpanzee, etc.
    """
    resources_root = (
        REPO_ROOT
        / "src"
        / "neuromaps_prime"
        / "resources"
        / "nodes"
    )

    matches = sorted(
        resources_root.rglob(f"{space_name}.yaml")
    )

    if not matches:
        available = sorted(
            path.stem
            for path in resources_root.rglob("*.yaml")
        )

        raise FileNotFoundError(
            f"Could not find YAML metadata for space "
            f"'{space_name}' under {resources_root}.\n"
            f"Available spaces: {', '.join(available)}"
        )

    if len(matches) > 1:
        locations = "\n".join(
            f"  - {path.relative_to(resources_root)}"
            for path in matches
        )

        raise ValueError(
            f"Found multiple YAML files for space "
            f"'{space_name}':\n"
            f"{locations}\n"
            "The space name must uniquely identify one YAML file."
        )

    yaml_path = matches[0]

    print(
        f"  Space metadata: "
        f"{yaml_path.relative_to(resources_root)}"
    )

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    return data.get(space_name, data)


def get_surface_resolutions(space):
    """Return surface resolutions that contain annotations."""
    surfaces = space.get("surfaces", {})
    resolutions = []

    for density, density_data in surfaces.items():
        if (
            isinstance(density_data, dict)
            and "annotation" in density_data
        ):
            resolutions.append(density)

    def sort_key(density):
        if density == "10k":
            return (0, 0)

        if density == "32k":
            return (1, 0)

        try:
            return (2, int(density.rstrip("k")))
        except ValueError:
            return (3, density)

    return sorted(resolutions, key=sort_key)


def get_annotations(space, density):
    """Return annotation names for a surface density."""
    density_data = space["surfaces"][density]
    annotations = density_data.get("annotation", {})

    if isinstance(annotations, dict):
        return list(annotations)

    return []


# ---------------------------------------------------------------------------
# Graph resources
# ---------------------------------------------------------------------------

def get_graph():
    """Create the neuromaps graph."""
    return NeuromapsGraph()


def get_anatomical_surface(graph, space_name, density, hemisphere):
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
        f"No anatomical surface found for "
        f"{space_name} {density} {hemisphere}"
    )


def get_annotation(graph, space_name, label, density, hemisphere):
    """Fetch an annotation resource through the project API."""
    return graph.fetch_surface_annotation(
        space=space_name,
        label=label,
        density=density,
        hemisphere=hemisphere,
    )


# ---------------------------------------------------------------------------
# GIFTI loading
# ---------------------------------------------------------------------------

def is_png_path(path):
    """Return True when a fetched resource is a PNG."""
    return Path(path).suffix.lower() == ".png"


def load_surface(resource):
    """Load coordinates and faces from a GIFTI surface resource."""
    path = resource.fetch()

    if is_png_path(path):
        raise ValueError(f"Skipping PNG resource: {path}")

    image = nib.load(str(path))

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
        raise ValueError(
            f"Could not find surface coordinates in {path}"
        )

    if faces is None:
        raise ValueError(
            f"Could not find surface faces in {path}"
        )

    return coordinates, faces


def load_annotation(resource, n_vertices):
    """
    Load an annotation from a GIFTI resource.

    Multi-map GIFTIs are reduced to their first map.
    PNG resources are explicitly rejected.
    """
    path = resource.fetch()

    if is_png_path(path):
        raise ValueError(f"Skipping PNG annotation: {path}")

    image = nib.load(str(path))

    candidates = []

    for darray in image.darrays:
        data = np.asarray(darray.data)

        if data.ndim == 1 and data.shape[0] == n_vertices:
            candidates.append(data)

        elif data.ndim == 2 and data.shape[0] == n_vertices:
            print(
                f"    {resource.name}: "
                f"multi-map data {data.shape}; using map 0"
            )
            candidates.append(data[:, 0])

        elif data.ndim == 2 and data.shape[1] == n_vertices:
            print(
                f"    {resource.name}: "
                f"multi-map data {data.shape}; using map 0"
            )
            candidates.append(data[0, :])

    if not candidates:
        raise ValueError(
            f"Could not find annotation data with "
            f"{n_vertices} vertices in {path}"
        )

    return np.asarray(candidates[0])


# ---------------------------------------------------------------------------
# Annotation classification
# ---------------------------------------------------------------------------

def annotation_is_categorical(name, values):
    """
    Determine whether an annotation should be plotted as ROI labels.

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
    ax,
    coordinates,
    faces,
    values,
    hemisphere,
    categorical,
    cmap,
):
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


def plot_resolution(graph, space_name, space, density):
    """Create one annotation figure for a surface density."""
    annotations = get_annotations(space, density)

    if not annotations:
        print(f"No annotations found for {density}")
        return

    print(f"\nProcessing {space_name} {density}")
    print(f"Found {len(annotations)} annotations")

    # ------------------------------------------------------------------
    # Load anatomical surfaces independently for each hemisphere.
    # ------------------------------------------------------------------

    surfaces = {}

    for hemisphere in ("left", "right"):
        try:
            surface_resource = get_anatomical_surface(
                graph,
                space_name,
                density,
                hemisphere,
            )

            print(
                f"  {hemisphere}: "
                f"using {surface_resource.resource_type} surface"
            )

            coordinates, faces = load_surface(surface_resource)

            surfaces[hemisphere] = {
                "coordinates": coordinates,
                "faces": faces,
                "resource": surface_resource,
            }

        except Exception as exc:
            print(
                f"  ERROR loading {hemisphere} anatomical surface: "
                f"{exc}"
            )

    if "left" not in surfaces or "right" not in surfaces:
        print(
            f"Skipping {density}: "
            "both hemispheres are required."
        )
        return

    # ------------------------------------------------------------------
    # Figure layout.
    # ------------------------------------------------------------------

    n_pairs = len(annotations)
    n_rows = int(np.ceil(n_pairs / N_PAIRS_PER_ROW))

    fig_width = N_PAIRS_PER_ROW * PAIR_WIDTH
    fig_height = n_rows * PAIR_HEIGHT

    fig = plt.figure(
        figsize=(fig_width, fig_height),
        constrained_layout=False,
    )

    fig.suptitle(
        f"{space_name} — {density} Surface Annotations",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.subplots_adjust(top=0.96)

    outer = fig.add_gridspec(
        n_rows,
        N_PAIRS_PER_ROW,
        wspace=0.02,
        hspace=0.12,
    )

    cmap_categorical = colormaps["tab20"]
    cmap_continuous = colormaps["viridis"]

    # ------------------------------------------------------------------
    # Plot every annotation.
    # ------------------------------------------------------------------

    for index, label in enumerate(annotations):
        row = index // N_PAIRS_PER_ROW
        col = index % N_PAIRS_PER_ROW

        pair_grid = outer[row, col].subgridspec(
            2,
            2,
            height_ratios=[0.15, 1],
            wspace=0.01,
            hspace=0.01,
        )

        title_ax = fig.add_subplot(pair_grid[0, :])
        left_ax = fig.add_subplot(
            pair_grid[1, 0],
            projection="3d",
        )
        right_ax = fig.add_subplot(
            pair_grid[1, 1],
            projection="3d",
        )

        title_ax.axis("off")

        title_ax.text(
            0.5,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # --------------------------------------------------------------
        # Fetch annotation resources.
        # --------------------------------------------------------------

        try:
            left_resource = get_annotation(
                graph,
                space_name,
                label,
                density,
                "left",
            )
        except Exception as exc:
            print(f"  ERROR fetching {label} left: {exc}")
            left_resource = None

        try:
            right_resource = get_annotation(
                graph,
                space_name,
                label,
                density,
                "right",
            )
        except Exception as exc:
            print(f"  ERROR fetching {label} right: {exc}")
            right_resource = None

        # --------------------------------------------------------------
        # Load annotation values.
        # --------------------------------------------------------------

        left_values = None
        right_values = None

        if left_resource is not None:
            try:
                left_values = load_annotation(
                    left_resource,
                    len(surfaces["left"]["coordinates"]),
                )
            except Exception as exc:
                print(f"  ERROR loading {label} left: {exc}")

        if right_resource is not None:
            try:
                right_values = load_annotation(
                    right_resource,
                    len(surfaces["right"]["coordinates"]),
                )
            except Exception as exc:
                print(f"  ERROR loading {label} right: {exc}")

        if left_values is None and right_values is None:
            left_ax.set_axis_off()
            right_ax.set_axis_off()
            continue

        # --------------------------------------------------------------
        # Determine map type.
        # --------------------------------------------------------------

        categorical = False

        if left_values is not None:
            categorical = annotation_is_categorical(
                label,
                left_values,
            )

        if right_values is not None:
            categorical = (
                categorical
                or annotation_is_categorical(
                    label,
                    right_values,
                )
            )

        cmap = (
            cmap_categorical
            if categorical
            else cmap_continuous
        )

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
                    categorical,
                    cmap,
                )
            except Exception as exc:
                print(f"  ERROR plotting {label} left: {exc}")
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
                    categorical,
                    cmap,
                )
            except Exception as exc:
                print(f"  ERROR plotting {label} right: {exc}")
                right_ax.set_axis_off()
        else:
            right_ax.set_axis_off()

    # ------------------------------------------------------------------
    # Save figure.
    # ------------------------------------------------------------------

    output_path = (
        REPO_ROOT
        / "examples"
        / f"visualize_{space_name}_{density}_annotations.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(
            f"Usage: python {Path(sys.argv[0]).name} SPACE"
        )
        sys.exit(1)

    space_name = sys.argv[1]

    print("Initializing NeuromapsGraph...")

    graph = get_graph()
    space = get_space_yaml(space_name)

    resolutions = get_surface_resolutions(space)

    if not resolutions:
        raise ValueError(
            f"No annotated surface resolutions found for {space_name}"
        )

    print(f"Space: {space_name}")
    print(f"Surface resolutions: {', '.join(resolutions)}")

    for density in resolutions:
        plot_resolution(
            graph,
            space_name,
            space,
            density,
        )


if __name__ == "__main__":
    main()
