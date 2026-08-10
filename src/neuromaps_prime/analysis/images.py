"""Image loading and label manipulation utilities.

Provides functions to load image data (e.g. surface) and
parcellation files and to relabel parcellation data so
that label indices are consecutive, with background regions
zeroed out.
"""

from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np

# Default parcellation labels to ignore (unknown regions, medial wall, etc.)
PARC_IGNORE = frozenset(
    {
        "unknown",
        "corpuscallosum",
        "Background+FreeSurfer_Defined_Medial_Wall",
        "???",
        "Unknown",
        "Medial_wall",
        "Medial wall",
        "medial_wall",
    }
)


def load_gifti(surface: str | Path) -> nib.GiftiImage:
    """Load a GIFTI surface or label file.

    Args:
        surface: Path to a GIFTI file (``.gii``, ``.func.gii``, etc.).

    Returns:
        A ``nibabel.GiftiImage`` instance.

    Raises:
        ValueError: If the loaded image is not a GiftiImage.
    """
    img = nib.load(surface)
    if not isinstance(img, nib.GiftiImage):
        raise ValueError(f"Expected to load Gifti surface for: {surface}")
    return img


def relabel_gifti(
    parcellation: str | Path,
    background: Iterable[str] | None = None,
) -> nib.GiftiImage:
    """Relabel a GIFTI parcellation so that label indices are consecutive.

    Loads the parcellation file, zeroes out any background labels found in
    the label table, then remaps the remaining indices to consecutive
    integers starting at ``1``.  Returns a new ``GiftiImage`` with an
    updated data array and label table.

    Args:
        parcellation: Path to a single GIFTI parcellation file.
        background: Iterable of label names to treat as background and
            zero out. Defaults to ``PARC_IGNORE``.

    Returns:
        A ``GiftiImage`` instance with consecutive label indices and an
        updated label table.
    """
    img = load_gifti(parcellation)
    data = img.agg_data().copy()
    labels = img.labeltable.labels
    lut = {v: k for k, v in img.labeltable.get_labels_as_dict().items()}

    if background is None:
        background = PARC_IGNORE

    # Zero out background labels
    if len(labels) > 0:
        for val in background:
            idx = lut.get(val)
            if idx is None:
                continue
            data[data == idx] = 0
            labels = [f for f in labels if f.key != idx]

    # Remap to consecutive indices starting at 1
    data = np.unique(data, return_inverse=True)[-1]
    new_labels = []
    if len(labels) > 0:
        for n, lab in enumerate(labels, start=1):
            lab.key = n
            new_labels.append(lab)

    # Build updated GIFTI image
    darr = nib.gifti.GiftiDataArray(
        data, intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32"
    )
    labeltable = nib.gifti.GiftiLabelTable()
    labeltable.labels = new_labels
    return nib.GiftiImage(darrays=[darr], labeltable=labeltable)
