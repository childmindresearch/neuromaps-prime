"""Visualize surface annotations on anatomical surfaces."""

from collections import OrderedDict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml
from matplotlib import colormaps
from nilearn import plotting

from neuromaps_prime.graph import NeuromapsGraph


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Graph/resource helpers
# ---------------------------------------------------------------------

def get_graph():
    """Initialize the NeuromapsGraph."""
    return NeuromapsGraph()


def get_node(graph, space_name):
    """Get a graph node for a space."""
    if space_name not in graph.nodes:
        available = sorted(graph.nodes)
        raise ValueError(
            f"Space '{space_name}' was not found in the graph.\n"
            f"Available spaces: {', '.join(available)}"
        )

    return graph.nodes[space_name]["data"]


def get_space_yaml(space_name):
    """Load the YAML definition for a space."""
    yaml_path = (
        REPO_ROOT
        / "src"
        / "neuromaps_prime"
        / "resources"
        / "nodes"
        / "macaque"
        / f"{space_name}.yaml"
    )

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Could not find YAML for space '{space_name}': {yaml_path}"
        )

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    return data.get(space_name, data)


def get_surface_resolutions(space):
    """Return surface densities containing annotations."""
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
    """Return annotation labels for a surface density."""
    density_data = space["surfaces"][density]
    annotations = density_data.get("annotation", {})

    if isinstance(annotations, dict):
        return list(annotations)

    return []


def get_anatomical_surface(graph, space_name, density, hemisphere):
    """Fetch the highest-priority anatomical surface for a hemisphere."""
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
        f"{space_name} {density} {hemisphere}. "
        f"Tried: {', '.join(SURFACE_PRIORITY)}"
    )


def get_annotation(graph, space_name, label, density, hemisphere):
    """Fetch a surface annotation resource."""
    return graph.fetch_surface_annotation(
        space=space_name,
        label=label,
        density=density,
        hemisphere=hemisphere,
    )


# ---------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------

def load_surface(resource):
    """Fetch and load a surface resource."""
    path = resource.fetch()

    image = nib.load(str(path))

    coordinates = None
    faces = None

    for darray in image.darrays:
        data = np.asarray(darray.data)

        # Surface coordinates.
        if (
            data.ndim == 2
            and data.shape[1] == 3
            and np.issubdtype(data.dtype, np.floating)
        ):
            coordinates = data

        # Surface triangles.
        elif (
            data.ndim == 2
            and data.shape[1] == 3
            and np.issubdtype(data.dtype, np.integer)
        ):
            faces = data

    if coordinates is None:
        raise ValueError(
            f"No surface coordinates found in {path}"
        )

    if faces is None:
        raise ValueError(
            f"No surface triangles found in {path}"
        )

    return coordinates, faces


def load_annotation(resource, n_vertices):
    """Fetch and load a surface annotation."""
    path = resource.fetch()

    image = nib.load(str(path))

    candidates = []

    for darray in image.darrays:
        data = np.asarray(darray.data)

        # Standard one-dimensional annotation.
        if data.ndim == 1 and data.shape[0] == n_vertices:
            candidates.append(data)

        # Multi-map annotation:
        # (vertices, maps)
        elif data.ndim == 2 and data.shape[0] == n_vertices:
            print(
                f"    {resource.name}: multi-map data "
                f"{data.shape}; using map 0"
            )
            candidates.append(data[:, 0])

        # Multi-map annotation:
        # (maps, vertices)
        elif data.ndim == 2 and data.shape[1] == n_vertices:
            print(
                f"    {resource.name}: multi-map data "
                f"{data.shape}; using map 0"
            )
            candidates.append(data[0, :])

    if not candidates:
        raise ValueError(
            f"No annotation array with {n_vertices} vertices "
            f"found in {path}"
        )

    return np.asarray(candidates[0])


# ---------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------

def annotation_is_categorical(name, values):
    """Determine whether an annotation should use a categorical colormap."""
    if name.startswith("PC_"):
        return True

    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        return False

    unique = np.unique(finite)

    return (
        len(unique) <= 20
        and np.allclose(unique, np.round(unique))
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_resolution(graph, space_name, space, density):
    annotations = get_annotations(space, density)

    if not annotations:
        print(f"No annotations found for {density}")
        return

    print(f"\nProcessing {space_name} {density}")
    print(f"Found {len(annotations)} annotations")

    # -------------------------------------------------------------
    # Fetch anatomical surfaces through NeuromapsGraph.
    # -------------------------------------------------------------

    surfaces = {}

    for hemisphere in ("left", "right"):
        surface_resource = get_anatomical_surface(
            graph,
            space_name,
            density,
            hemisphere,
        )

        print(
            f"  {hemisphere}: using "
            f"{surface_resource.resource_type} surface"
        )

        try:
            coordinates, faces = load_surface(surface_resource)

            surfaces[hemisphere] = {
                "coordinates": coordinates,
                "faces": faces,
                "resource": surface_resource,
            }

        except Exception as exc:
            print(
                f"  ERROR loading {hemisphere} surface: {exc}"
            )

    if "left" not in surfaces or "right" not in surfaces:
        print(
            f"Skipping {density}: both hemispheres are required."
        )
        return

    # -------------------------------------------------------------
    # Figure dimensions.
    # -------------------------------------------------------------

    n_pairs = len(annotations)
    n_rows = int(np.ceil(n_pairs / N_PAIRS_PER_ROW))

    fig_width = N_PAIRS_PER_ROW * PAIR_WIDTH
    fig_height = n_rows * PAIR_HEIGHT

    fig = plt.figure(
        figsize=(fig_width, fig_height),
        constrained_layout=False,
    )

    outer = fig.add_gridspec(
        n_rows,
        N_PAIRS_PER_ROW,
        wspace=0.02,
        hspace=0.12,
    )

    cmap_categorical = colormaps["tab20"]
    cmap_continuous = colormaps["viridis"]

    # -------------------------------------------------------------
    # Plot each annotation pair.
    # -------------------------------------------------------------

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

        # Nilearn surface plotting requires 3D axes.
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

        # ---------------------------------------------------------
        # Fetch annotations through NeuromapsGraph.
        # ---------------------------------------------------------

        left_resource = get_annotation(
            graph,
            space_name,
            label,
            density,
            "left",
        )

        right_resource = get_annotation(
            graph,
            space_name,
            label,
            density,
            "right",
        )

        left_values = None
        right_values = None

        if left_resource is not None:
            try:
                left_values = load_annotation(
                    left_resource,
                    len(surfaces["left"]["coordinates"]),
                )
            except Exception as exc:
                print(
                    f"  ERROR loading "
                    f"{label} left: {exc}"
                )

        if right_resource is not None:
            try:
                right_values = load_annotation(
                    right_resource,
                    len(surfaces["right"]["coordinates"]),
                )
            except Exception as exc:
                print(
                    f"  ERROR loading "
                    f"{label} right: {exc}"
                )

        # ---------------------------------------------------------
        # Select colormap.
        # ---------------------------------------------------------

        categorical = False

        if left_values is not None:
            categorical = annotation_is_categorical(
                label,
                left_values,
            )

        if right_values is not None:
            categorical = categorical or annotation_is_categorical(
                label,
                right_values,
            )

        cmap = (
            cmap_categorical
            if categorical
            else cmap_continuous
        )

        # ---------------------------------------------------------
        # Plot left hemisphere.
        # ---------------------------------------------------------

        try:
            if left_values is not None:
                plotting.plot_surf_roi(
                    (
                        surfaces["left"]["coordinates"],
                        surfaces["left"]["faces"],
                    ),
                    roi_map=left_values,
                    hemi="left",
                    view="lateral",
                    bg_on_data=True,
                    cmap=cmap,
                    colorbar=False,
                    axes=left_ax,
                )
            else:
                left_ax.set_axis_off()

        except Exception as exc:
            print(
                f"  ERROR plotting "
                f"{label} left: {exc}"
            )
            left_ax.set_axis_off()

        # ---------------------------------------------------------
        # Plot right hemisphere.
        # ---------------------------------------------------------

        try:
            if right_values is not None:
                plotting.plot_surf_roi(
                    (
                        surfaces["right"]["coordinates"],
                        surfaces["right"]["faces"],
                    ),
                    roi_map=right_values,
                    hemi="right",
                    view="lateral",
                    bg_on_data=True,
                    cmap=cmap,
                    colorbar=False,
                    axes=right_ax,
                )
            else:
                right_ax.set_axis_off()

        except Exception as exc:
            print(
                f"  ERROR plotting "
                f"{label} right: {exc}"
            )
            right_ax.set_axis_off()

    # -------------------------------------------------------------
    # Save figure.
    # -------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python3 examples/visualize-annotations.py SPACE\n\n"
            "Example:\n"
            "  python3 examples/visualize-annotations.py Yerkes19"
        )
        sys.exit(1)

    space_name = sys.argv[1]

    print("Initializing NeuromapsGraph...")
    graph = get_graph()

    # YAML is only used to discover the annotations/densities.
    space = get_space_yaml(space_name)

    resolutions = get_surface_resolutions(space)

    if not resolutions:
        raise ValueError(
            f"No surface annotations found for {space_name}."
        )

    print(f"Space: {space_name}")
    print(
        f"Surface resolutions: "
        f"{', '.join(resolutions)}"
    )

    for density in resolutions:
        plot_resolution(
            graph,
            space_name,
            space,
            density,
        )


if __name__ == "__main__":
    main()