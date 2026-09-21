"""Visualize annotations on anatomical surfaces for a node space."""

from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import requests
import yaml
from matplotlib import colormaps
from nilearn import plotting


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

NODE_DIR = (
    REPO_ROOT
    / "src"
    / "neuromaps_prime"
    / "resources"
    / "nodes"
    / "macaque"
)

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

CACHE_DIR = Path(tempfile.gettempdir()) / "neuromaps_prime_visualize"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_space_yaml(space_name):
    """Load the YAML definition for a space."""
    yaml_path = NODE_DIR / f"{space_name}.yaml"

    if not yaml_path.exists():
        available = sorted(path.stem for path in NODE_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Could not find YAML for space '{space_name}'.\n"
            f"Expected: {yaml_path}\n"
            f"Available spaces: {', '.join(available)}"
        )

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    if space_name in data:
        return data[space_name]

    return data


def download_file(url, name):
    """Download and cache a file, preserving GIFTI extension when needed."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    content = response.content
    content_type = response.headers.get("Content-Type", "").lower()

    is_gifti = (
        "gifti" in content_type
        or b"<GIFTI" in content[:1000]
        or (
            b"<?xml" in content[:1000]
            and b"GIFTI" in content[:5000]
        )
    )

    extension = ".gii" if is_gifti else ""

    safe_name = name.replace("/", "_").replace(":", "_")
    output_path = CACHE_DIR / f"{safe_name}{extension}"

    if not output_path.exists():
        output_path.write_bytes(content)

    return output_path


def get_url(value):
    """Extract a URL from a YAML value."""
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ("url", "source", "path", "file"):
            if key in value and isinstance(value[key], str):
                return value[key]

    return None


def load_surface(url, name):
    """Load a GIFTI surface and return coordinates and triangles."""
    path = download_file(url, name)

    image = nib.load(str(path))

    coordinates = None
    faces = None

    for darray in image.darrays:
        data = np.asarray(darray.data)

        # Surface coordinates: N x 3 floating-point array.
        if (
            data.ndim == 2
            and data.shape[1] == 3
            and np.issubdtype(data.dtype, np.floating)
        ):
            coordinates = data

        # Surface triangles: M x 3 integer array.
        elif (
            data.ndim == 2
            and data.shape[1] == 3
            and np.issubdtype(data.dtype, np.integer)
        ):
            faces = data

    if coordinates is None:
        raise ValueError(f"No surface coordinates found in {path}")

    if faces is None:
        raise ValueError(f"No surface triangles found in {path}")

    return coordinates, faces


def load_annotation(url, name, n_vertices):
    """Load a surface annotation as a one-dimensional vertex array."""
    path = download_file(url, name)

    image = nib.load(str(path))

    candidates = []

    for darray in image.darrays:
        data = np.asarray(darray.data)

        if data.ndim == 1 and data.shape[0] == n_vertices:
            candidates.append(data)

        elif data.ndim == 2:
            if data.shape[0] == n_vertices:
                print(
                    f"  {name}: multi-map data "
                    f"{data.shape}; using map 0"
                )
                candidates.append(data[:, 0])

            elif data.shape[1] == n_vertices:
                print(
                    f"  {name}: multi-map data "
                    f"{data.shape}; using map 0"
                )
                candidates.append(data[0, :])

    if not candidates:
        raise ValueError(
            f"No annotation array with {n_vertices} vertices found in {path}"
        )

    return np.asarray(candidates[0])


def get_surface_resolutions(space):
    """Return surface resolutions containing annotations."""
    surfaces = space.get("surfaces", {})

    resolutions = []

    for density, data in surfaces.items():
        if isinstance(data, dict) and "annotation" in data:
            resolutions.append(density)

    return resolutions


def get_annotations(space, density):
    """Return annotation entries for a surface density."""
    density_data = space["surfaces"][density]
    annotations = density_data.get("annotation", {})

    if isinstance(annotations, dict):
        return list(annotations.items())

    return []


def get_anatomical_surface(space, density, hemisphere):
    """Find the highest-priority anatomical surface for a hemisphere."""
    density_data = space["surfaces"][density]

    for surface_name in SURFACE_PRIORITY:
        surface_data = density_data.get(surface_name)

        if not isinstance(surface_data, dict):
            continue

        value = surface_data.get(hemisphere)

        if value is None:
            continue

        url = get_url(value)

        if url is not None:
            return surface_name, url

    raise ValueError(
        f"No anatomical surface found for {hemisphere} at {density}"
    )


def get_annotation_hemisphere(annotation_data, hemisphere):
    """Get an annotation URL for a hemisphere."""
    if not isinstance(annotation_data, dict):
        return None

    value = annotation_data.get(hemisphere)

    if value is None:
        return None

    return get_url(value)


def annotation_is_categorical(name, values):
    """Determine whether an annotation is likely categorical."""
    if name.startswith("PC_"):
        return True

    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        return False

    unique = np.unique(finite)

    return len(unique) <= 20 and np.allclose(
        unique,
        np.round(unique),
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_resolution(space_name, space, density):
    """Plot all annotation pairs for one surface density."""
    annotations = get_annotations(space, density)

    if not annotations:
        print(f"No annotations found for {density}")
        return

    print(f"\nProcessing {space_name} {density}")
    print(f"Found {len(annotations)} annotations")

    # -------------------------------------------------------------
    # Load anatomical surfaces once.
    # -------------------------------------------------------------

    surfaces = {}

    for hemisphere in ("left", "right"):
        surface_name, surface_url = get_anatomical_surface(
            space,
            density,
            hemisphere,
        )

        print(
            f"  {hemisphere}: using {surface_name} surface"
        )

        try:
            coordinates, faces = load_surface(
                surface_url,
                f"{space_name}_{density}_{surface_name}_{hemisphere}",
            )

            surfaces[hemisphere] = {
                "coordinates": coordinates,
                "faces": faces,
                "surface_name": surface_name,
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

    for index, (annotation_name, annotation_data) in enumerate(
        annotations
    ):
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
            annotation_name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        left_url = get_annotation_hemisphere(
            annotation_data,
            "left",
        )

        right_url = get_annotation_hemisphere(
            annotation_data,
            "right",
        )

        left_values = None
        right_values = None

        # ---------------------------------------------------------
        # Load annotation data.
        # ---------------------------------------------------------

        if left_url is not None:
            try:
                left_values = load_annotation(
                    left_url,
                    f"{space_name}_{density}_{annotation_name}_left",
                    len(surfaces["left"]["coordinates"]),
                )
            except Exception as exc:
                print(
                    f"  ERROR loading "
                    f"{annotation_name} left: {exc}"
                )

        if right_url is not None:
            try:
                right_values = load_annotation(
                    right_url,
                    f"{space_name}_{density}_{annotation_name}_right",
                    len(surfaces["right"]["coordinates"]),
                )
            except Exception as exc:
                print(
                    f"  ERROR loading "
                    f"{annotation_name} right: {exc}"
                )

        # ---------------------------------------------------------
        # Select colormap.
        # ---------------------------------------------------------

        categorical = False

        if left_values is not None:
            categorical = annotation_is_categorical(
                annotation_name,
                left_values,
            )

        if right_values is not None:
            categorical = categorical or annotation_is_categorical(
                annotation_name,
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
                    darkness=0.5,
                    cmap=cmap,
                    colorbar=False,
                    axes=left_ax,
                )
            else:
                left_ax.set_axis_off()

        except Exception as exc:
            print(
                f"  ERROR plotting "
                f"{annotation_name} left: {exc}"
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
                    darkness=0.5,
                    cmap=cmap,
                    colorbar=False,
                    axes=right_ax,
                )
            else:
                right_ax.set_axis_off()

        except Exception as exc:
            print(
                f"  ERROR plotting "
                f"{annotation_name} right: {exc}"
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

    space = get_space_yaml(space_name)

    resolutions = get_surface_resolutions(space)

    if not resolutions:
        raise ValueError(
            f"No surface resolutions with annotations found "
            f"for {space_name}."
        )

    print(f"Space: {space_name}")
    print(
        f"Surface resolutions: {', '.join(resolutions)}"
    )

    for density in resolutions:
        plot_resolution(
            space_name,
            space,
            density,
        )


if __name__ == "__main__":
    main()