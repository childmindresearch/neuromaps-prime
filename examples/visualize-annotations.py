"""Visualize all Yerkes19 10k annotations on midthickness surfaces."""

from pathlib import Path
import math
import tempfile
import urllib.request

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
import yaml
from nilearn import plotting


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

SPACE_NAME = "Yerkes19"
DENSITY = "10k"

SURFACE_PRIORITY = [
    "midthickness",
    "white",
    "pial",
    "inflated",
    "sphere",
]

OUTPUT_FILE = REPO_ROOT / f"visualize_{SPACE_NAME}_{DENSITY}_annotations.png"

# Four L/R annotation pairs across.
N_PAIRS_PER_ROW = 4

CACHE_DIR = Path(tempfile.gettempdir()) / "neuromaps_prime_visualize"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# YAML resource
# ---------------------------------------------------------------------

NODE_FILE = (
    REPO_ROOT
    / "src"
    / "neuromaps_prime"
    / "resources"
    / "nodes"
    / "macaque"
    / f"{SPACE_NAME}.yaml"
)


# ---------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------

def download_file(url: str, name: str) -> Path:
    """Download a remote file and cache it locally."""
    path = CACHE_DIR / name

    if path.exists() and path.stat().st_size > 0:
        return path

    print(f"Downloading {url}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request) as response:
        data = response.read()
        content_type = response.headers.get("Content-Type", "")

    # OSF frequently returns GIFTI files as application/octet-stream.
    is_gifti = (
        "gifti" in content_type.lower()
        or data.lstrip().startswith(b"<?xml")
        or b"<GIFTI" in data[:5000]
    )

    if is_gifti and path.suffix.lower() != ".gii":
        path = path.with_suffix(".gii")

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "wb") as f:
        f.write(data)

    tmp_path.replace(path)

    return path


# ---------------------------------------------------------------------
# GIFTI loading
# ---------------------------------------------------------------------

def load_surface(url: str, name: str):
    """Load GIFTI surface coordinates and triangle indices."""
    path = download_file(url, f"{name}.gii")

    image = nib.load(str(path))

    coordinates = None
    triangles = None

    for darray in image.darrays:
        data = np.asarray(darray.data)

        # Surface coordinates are N x 3.
        if data.ndim == 2 and data.shape[1] == 3:
            if coordinates is None:
                coordinates = data

        # Triangle indices are M x 3 and integer-valued.
        elif data.ndim == 2 and data.shape[1] == 3:
            if (
                np.issubdtype(data.dtype, np.integer)
                and triangles is None
            ):
                triangles = data.astype(np.int32)

    # Some GIFTI readers expose both as N x 3, so use the intent
    # as a secondary signal when available.
    if coordinates is None or triangles is None:
        for darray in image.darrays:
            data = np.asarray(darray.data)

            intent = str(darray.intent).lower()

            if (
                coordinates is None
                and data.ndim == 2
                and data.shape[1] == 3
                and (
                    "point" in intent
                    or "coord" in intent
                    or darray.intent == 1008
                )
            ):
                coordinates = data

            if (
                triangles is None
                and data.ndim == 2
                and data.shape[1] == 3
                and (
                    "triangle" in intent
                    or "face" in intent
                    or darray.intent == 1009
                )
            ):
                triangles = data.astype(np.int32)

    if coordinates is None or triangles is None:
        print(f"\nCould not identify surface arrays in: {path}")

        for i, darray in enumerate(image.darrays):
            print(
                f"  array {i}: "
                f"intent={darray.intent!r}, "
                f"shape={np.asarray(darray.data).shape}, "
                f"dtype={np.asarray(darray.data).dtype}"
            )

        raise ValueError(
            f"Could not identify pointset and triangle arrays in {path}"
        )

    return coordinates, triangles


def load_annotation(url: str, name: str):
    """Load the first data array from a GIFTI annotation."""
    path = download_file(url, f"{name}.gii")

    image = nib.load(str(path))

    if not image.darrays:
        raise ValueError(f"No data arrays found in {path}")

    return np.asarray(image.darrays[0].data).squeeze()


# ---------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------

def load_space_data():
    """Load the Yerkes19 YAML resource directly."""
    if not NODE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find YAML resource:\n{NODE_FILE}"
        )

    with open(NODE_FILE, "r") as f:
        data = yaml.safe_load(f)

    if SPACE_NAME not in data:
        raise KeyError(
            f"{SPACE_NAME} not found in {NODE_FILE}"
        )

    return data[SPACE_NAME]


# ---------------------------------------------------------------------
# Surface selection
# ---------------------------------------------------------------------

def choose_surface(density_data, hemisphere):
    """Choose the highest-priority available anatomical surface."""
    for surface_name in SURFACE_PRIORITY:
        surface = density_data.get(surface_name)

        if (
            isinstance(surface, dict)
            and hemisphere in surface
            and surface[hemisphere]
        ):
            return surface_name, surface[hemisphere]

    raise ValueError(
        f"No anatomical surface found for {hemisphere}. "
        f"Tried: {SURFACE_PRIORITY}"
    )


# ---------------------------------------------------------------------
# Load YAML
# ---------------------------------------------------------------------

space = load_space_data()

surfaces_data = space.get("surfaces", {})

if DENSITY not in surfaces_data:
    raise ValueError(
        f"{SPACE_NAME} does not contain density {DENSITY}"
    )

density_data = surfaces_data[DENSITY]

annotations = density_data.get("annotation", {})

if not annotations:
    raise ValueError(
        f"No annotations found under "
        f"{SPACE_NAME} -> surfaces -> {DENSITY} -> annotation"
    )

print(f"Found {len(annotations)} annotations:")

for annotation_name in annotations:
    print(f"  - {annotation_name}")


# ---------------------------------------------------------------------
# Load anatomical surfaces once
# ---------------------------------------------------------------------

surface_data = {}

for hemisphere in ("left", "right"):
    surface_name, surface_url = choose_surface(
        density_data,
        hemisphere,
    )

    print(
        f"Using {surface_name} surface for "
        f"{hemisphere} hemisphere"
    )

    surface_data[hemisphere] = load_surface(
        surface_url,
        f"{SPACE_NAME}_{DENSITY}_{surface_name}_{hemisphere}",
    )


# ---------------------------------------------------------------------
# Figure layout
#
# Four complete annotation pairs per row:
#
#   [ L | R | CB ] [ L | R | CB ] [ L | R | CB ] [ L | R | CB ]
#
# ---------------------------------------------------------------------

annotation_items = list(annotations.items())

n_annotations = len(annotation_items)
n_rows = math.ceil(n_annotations / N_PAIRS_PER_ROW)

fig = plt.figure(
    figsize=(22, n_rows * 5.0),
    facecolor="white",
)

outer = fig.add_gridspec(
    n_rows,
    N_PAIRS_PER_ROW,
    wspace=0.08,
    hspace=0.28,
)


# ---------------------------------------------------------------------
# Plot each annotation
# ---------------------------------------------------------------------

for index, (annotation_name, annotation_data) in enumerate(
    annotation_items
):

    row = index // N_PAIRS_PER_ROW
    col = index % N_PAIRS_PER_ROW

    print(f"\nProcessing {annotation_name}")

    left_url = annotation_data.get("left")
    right_url = annotation_data.get("right")

    left_values = None
    right_values = None

    # -------------------------------------------------------------
    # Load left annotation
    # -------------------------------------------------------------

    if left_url:
        try:
            left_values = load_annotation(
                left_url,
                f"{SPACE_NAME}_{DENSITY}_{annotation_name}_left",
            )
        except Exception as exc:
            print(
                f"WARNING: could not load left annotation "
                f"{annotation_name}: {exc}"
            )

    # -------------------------------------------------------------
    # Load right annotation
    # -------------------------------------------------------------

    if right_url:
        try:
            right_values = load_annotation(
                right_url,
                f"{SPACE_NAME}_{DENSITY}_{annotation_name}_right",
            )
        except Exception as exc:
            print(
                f"WARNING: could not load right annotation "
                f"{annotation_name}: {exc}"
            )

    if left_values is None and right_values is None:
        print(f"WARNING: no usable data for {annotation_name}")
        continue

    # -------------------------------------------------------------
    # Validate vertex counts
    # -------------------------------------------------------------

    left_coordinates, left_faces = surface_data["left"]
    right_coordinates, right_faces = surface_data["right"]

    if (
        left_values is not None
        and len(left_values) != len(left_coordinates)
    ):
        print(
            f"WARNING: skipping left hemisphere for "
            f"{annotation_name}: "
            f"{len(left_values)} values vs "
            f"{len(left_coordinates)} vertices"
        )
        left_values = None

    if (
        right_values is not None
        and len(right_values) != len(right_coordinates)
    ):
        print(
            f"WARNING: skipping right hemisphere for "
            f"{annotation_name}: "
            f"{len(right_values)} values vs "
            f"{len(right_coordinates)} vertices"
        )
        right_values = None

    if left_values is None and right_values is None:
        continue

    # -------------------------------------------------------------
    # Shared color scale for the L/R pair
    # -------------------------------------------------------------

    valid_values = []

    if left_values is not None:
        valid_values.append(
            left_values[np.isfinite(left_values)]
        )

    if right_values is not None:
        valid_values.append(
            right_values[np.isfinite(right_values)]
        )

    if not valid_values:
        continue

    all_values = np.concatenate(valid_values)

    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))

    if vmin == vmax:
        vmax = vmin + 1

    norm = Normalize(
        vmin=vmin,
        vmax=vmax,
    )

    cmap = cm.get_cmap("tab20")

    # -------------------------------------------------------------
    # Create one annotation pair
    # -------------------------------------------------------------

    pair = outer[row, col].subgridspec(
        2,
        3,
        height_ratios=[0.25, 1],
        width_ratios=[1, 1, 0.12],
        wspace=0.01,
        hspace=0.01,
    )

    # -------------------------------------------------------------
    # Annotation label
    # -------------------------------------------------------------

    title_ax = fig.add_subplot(pair[0, :])
    title_ax.axis("off")

    title_ax.text(
        0.5,
        0.45,
        annotation_name,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # -------------------------------------------------------------
    # Left hemisphere
    # -------------------------------------------------------------

    left_ax = fig.add_subplot(
        pair[1, 0],
        projection="3d",
    )

    if left_values is not None:
        plotting.plot_surf_roi(
            (left_coordinates, left_faces),
            roi_map=left_values,
            hemi="left",
            view="lateral",
            cmap="tab20",
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            bg_map=None,
            axes=left_ax,
        )
    else:
        left_ax.axis("off")

    # -------------------------------------------------------------
    # Right hemisphere
    # -------------------------------------------------------------

    right_ax = fig.add_subplot(
        pair[1, 1],
        projection="3d",
    )

    if right_values is not None:
        plotting.plot_surf_roi(
            (right_coordinates, right_faces),
            roi_map=right_values,
            hemi="right",
            view="lateral",
            cmap="tab20",
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            bg_map=None,
            axes=right_ax,
        )
    else:
        right_ax.axis("off")

    # -------------------------------------------------------------
    # One shared color bar
    # -------------------------------------------------------------

    colorbar_ax = fig.add_subplot(pair[1, 2])

    scalar_mappable = cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar_mappable.set_array(all_values)

    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_ax,
    )

    colorbar.ax.tick_params(
        labelsize=6,
        pad=1,
    )

    # Keep integer labels for categorical annotations where
    # there are a manageable number of unique values.
    if np.all(np.isclose(all_values, np.round(all_values))):
        integer_values = np.unique(
            np.round(all_values).astype(int)
        )

        if len(integer_values) <= 20:
            colorbar.set_ticks(integer_values)

    colorbar.outline.set_linewidth(0.5)


# ---------------------------------------------------------------------
# Overall title
# ---------------------------------------------------------------------

fig.suptitle(
    f"{SPACE_NAME} — {DENSITY} annotations",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("\nSaved visualization to:")
print(OUTPUT_FILE)