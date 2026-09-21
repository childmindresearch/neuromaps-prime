"""Visualize the Yerkes19 PC_Yeo17Networks annotation on 10k midthickness."""

from pathlib import Path
import hashlib
import tempfile

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import requests
from nilearn import plotting


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

NODE_DIR = (
    REPO_ROOT
    / "src"
    / "neuromaps_prime"
    / "resources"
    / "nodes"
)

SPACE_NAME = "Yerkes19"
DENSITY = "10k"
ANNOTATION_NAME = "PC_Yeo17Networks"

OUTPUT_FILE = (
    REPO_ROOT
    / f"visualize_{SPACE_NAME}_{ANNOTATION_NAME}_{DENSITY}.png"
)

SURFACE_PRIORITY = [
    "midthickness",
    "white",
    "pial",
    "inflated",
    "sphere",
]

CACHE_DIR = Path(tempfile.gettempdir()) / "neuromaps_prime_visualize"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def find_yaml(space_name):
    """Find the YAML file containing the requested space."""
    matches = list(NODE_DIR.rglob(f"{space_name}.yaml"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find YAML for {space_name} under {NODE_DIR}"
        )

    return matches[0]


def load_yaml(path):
    """Load YAML."""
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Download/cache helpers
# ---------------------------------------------------------------------------

def detect_extension(data, content_type=""):
    """Detect the file type from bytes, falling back to Content-Type."""
    # GIFTI files are XML.
    if data[:100].lstrip().startswith(b"<?xml"):
        return ".gii"

    if b"<GIFTI" in data[:1000]:
        return ".gii"

    # NIfTI gzip.
    if data[:2] == b"\x1f\x8b":
        return ".nii.gz"

    # NIfTI magic.
    if len(data) >= 348:
        if data[344:348] in (b"n+1\x00", b"ni1\x00"):
            return ".nii"

    content_type = content_type.lower()

    if "gifti" in content_type:
        return ".gii"

    if "nifti" in content_type:
        return ".nii"

    return ""


def download_file(url):
    """Download a remote resource and cache it locally."""
    key = hashlib.sha256(url.encode()).hexdigest()

    # Reuse an existing cached file.
    for extension in (".gii", ".nii.gz", ".nii"):
        cached = CACHE_DIR / f"{key}{extension}"

        if cached.exists():
            return cached

    print(f"Downloading: {url}")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    data = response.content

    extension = detect_extension(data, content_type)

    if not extension:
        raise RuntimeError(
            f"Could not determine file type for:\n"
            f"{url}\n"
            f"Content-Type: {content_type}"
        )

    print(f"  Content-Type: {content_type}")
    print(f"  Detected format: {extension}")

    cached = CACHE_DIR / f"{key}{extension}"
    cached.write_bytes(data)

    print(f"  Cached at: {cached}")

    return cached


# ---------------------------------------------------------------------------
# GIFTI helpers
# ---------------------------------------------------------------------------

def load_gifti_data(path):
    """Load the first data array from a GIFTI file."""
    image = nib.load(str(path))

    if not isinstance(image, nib.gifti.GiftiImage):
        raise TypeError(
            f"Expected GIFTI image, got {type(image)}"
        )

    if not image.darrays:
        raise ValueError(
            f"No data arrays found in {path}"
        )

    return np.asarray(image.darrays[0].data)


def load_surface(url):
    """Download and load a GIFTI surface."""
    path = download_file(url)

    image = nib.load(str(path))

    if not isinstance(image, nib.gifti.GiftiImage):
        raise TypeError(
            f"Expected GIFTI surface, got {type(image)}"
        )

    coords = None
    faces = None

    for darray in image.darrays:
        if (
            darray.intent
            == nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]
        ):
            coords = np.asarray(darray.data)

        elif (
            darray.intent
            == nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]
        ):
            faces = np.asarray(darray.data)

    if coords is None:
        raise ValueError(
            f"No pointset found in surface: {path}"
        )

    if faces is None:
        raise ValueError(
            f"No triangle faces found in surface: {path}"
        )

    return coords, faces, path


# ---------------------------------------------------------------------------
# Surface selection
# ---------------------------------------------------------------------------

def select_surface(density_data, hemisphere):
    """Select the highest-priority available surface."""
    for surface_name in SURFACE_PRIORITY:
        surface_data = density_data.get(surface_name)

        if not surface_data:
            continue

        url = surface_data.get(hemisphere)

        if not url:
            continue

        print(
            f"  {hemisphere}: {surface_name} → {url}"
        )

        coords, faces, path = load_surface(url)

        return surface_name, coords, faces, path

    raise RuntimeError(
        f"No usable surface found for {hemisphere}"
    )


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def main():
    yaml_path = find_yaml(SPACE_NAME)
    node = load_yaml(yaml_path)

    if SPACE_NAME not in node:
        raise KeyError(
            f"{SPACE_NAME} not found in {yaml_path}"
        )

    space = node[SPACE_NAME]

    print(f"Reading nodes from: {NODE_DIR}")
    print(f"Target space: {SPACE_NAME}")
    print()
    print(f"Space:      {SPACE_NAME}")
    print(f"Species:    {space.get('species')}")
    print(f"YAML:       {yaml_path}")
    print(f"Density:    {DENSITY}")
    print(f"Annotation: {ANNOTATION_NAME}")
    print()

    surfaces = space["surfaces"]

    if DENSITY not in surfaces:
        raise KeyError(
            f"Density {DENSITY!r} not found in {SPACE_NAME}"
        )

    density_data = surfaces[DENSITY]

    annotations = density_data.get("annotation", {})

    if ANNOTATION_NAME not in annotations:
        available = ", ".join(annotations.keys())

        raise KeyError(
            f"{ANNOTATION_NAME!r} not found at "
            f"{SPACE_NAME} → surfaces → {DENSITY} → annotation.\n\n"
            f"Available annotations:\n{available}"
        )

    annotation = annotations[ANNOTATION_NAME]

    hemisphere_data = {}

    # -----------------------------------------------------------------------
    # Load left and right hemisphere surfaces + annotation
    # -----------------------------------------------------------------------

    for hemisphere in ("left", "right"):
        if hemisphere not in annotation:
            raise KeyError(
                f"{ANNOTATION_NAME} has no {hemisphere} annotation."
            )

        (
            surface_name,
            coords,
            faces,
            surface_path,
        ) = select_surface(
            density_data,
            hemisphere,
        )

        annotation_url = annotation[hemisphere]

        print(
            f"  {hemisphere}: loading {ANNOTATION_NAME}"
        )

        annotation_path = download_file(annotation_url)

        annotation_data = load_gifti_data(
            annotation_path
        )

        print(
            f"    surface vertices:    {len(coords)}"
        )
        print(
            f"    annotation vertices: {len(annotation_data)}"
        )

        if len(annotation_data) != len(coords):
            raise ValueError(
                f"Vertex-count mismatch for {hemisphere}:\n"
                f"  surface:    {len(coords)} vertices\n"
                f"  annotation: {len(annotation_data)} vertices"
            )

        hemisphere_data[hemisphere] = {
            "coords": coords,
            "faces": faces,
            "annotation": annotation_data,
            "surface_name": surface_name,
            "surface_path": surface_path,
        }

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    fig = plt.figure(figsize=(14, 7))

    for index, hemisphere in enumerate(
        ("left", "right"),
        start=1,
    ):
        data = hemisphere_data[hemisphere]

        ax = fig.add_subplot(
            1,
            2,
            index,
            projection="3d",
        )

        roi = np.asarray(
            data["annotation"]
        )

        # Handle floating-point annotations containing NaN/Inf.
        if np.issubdtype(
            roi.dtype,
            np.floating,
        ):
            roi = np.nan_to_num(
                roi,
                nan=0,
                posinf=0,
                neginf=0,
            )

        roi = roi.astype(int)

        plotting.plot_surf_roi(
            surf_mesh=(
                data["coords"],
                data["faces"],
            ),
            roi_map=roi,
            hemi=(
                "left"
                if hemisphere == "left"
                else "right"
            ),
            view="lateral",
            bg_map=None,
            cmap="tab20",
            axes=ax,
            colorbar=True,
            title=(
                f"{hemisphere.capitalize()} hemisphere\n"
                f"{ANNOTATION_NAME}"
            ),
        )

    fig.suptitle(
        f"{SPACE_NAME} — {ANNOTATION_NAME}\n"
        f"{DENSITY} {hemisphere_data['left']['surface_name']} surface",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()