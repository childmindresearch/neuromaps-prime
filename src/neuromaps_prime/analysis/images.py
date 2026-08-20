"""Image loading and label manipulation utilities.

Provides functions to load image data (e.g. surface) and
parcellation files and to relabel parcellation data so
that label indices are consecutive, with background regions
zeroed out.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike, DTypeLike

__all__ = [
    "PARC_IGNORE",
    "load_data",
    "load_gifti",
    "relabel_gifti",
]

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

NiftiImage: TypeAlias = nib.Nifti1Image | nib.Nifti2Image  # noqa: UP040 - <py312 compatibility


class DataOutput(NamedTuple):
    array: np.ndarray
    image: nib.GiftiImage | NiftiImage | None = None


def load_gifti(surface: str | Path) -> nib.GiftiImage:
    """Load a GIFTI surface or label file.

    Args:
        surface: Path to a GIFTI file.

    Returns:
        A ``nibabel.GiftiImage`` instance.

    Raises:
        ValueError: If the loaded image is not a ``GiftiImage``.
    """
    img = nib.load(surface)

    if not isinstance(img, nib.GiftiImage):
        raise ValueError(f"Expected to load Gifti surface for: {surface}")

    return img


def load_data(
    data: ArrayLike | str | Path,
    dtype: DTypeLike | None = None,
    *,
    return_image: bool = False,
) -> DataOutput:
    """Load a brain map into a NumPy array.

    Accepts an in-memory array, a path to a GIFTI or NIfTI file, or an
    already-loaded nibabel image. GIFTI images are extracted via
    ``agg_data()``; NIfTI images via ``get_fdata()``.

    Args:
        data: A NumPy array, array-like, file path (``.gii``, ``.nii``,
            ``.nii.gz``, etc.), or a ``nibabel.GiftiImage`` /
            ``nibabel.Nifti1Image`` / ``nibabel.Nifti2Image``.
        dtype: Optional target data type for loaded arrays (e.g., ``np.float32``).
        return_image: Boolean flag to return the image for metadata, headers, etc.

    Returns:
        A NumPy array containing the image data.

    Raises:
        ValueError: If *data* is a nibabel image of an unsupported type.

    Note:
        For label images loaded from NIfTI files, floating-point
        imprecision in ``get_fdata()`` may require rounding before
        casting to integers. Callers should apply ``np.rint()`` as
        needed.
    """
    # Nibabel to handle FileNotFoundError
    if isinstance(data, str | Path):
        img = nib.load(str(data))
    # Data (assumed) to be an array
    elif not isinstance(data, nib.GiftiImage | NiftiImage):
        return DataOutput(np.asarray(data, dtype=dtype))

    if isinstance(img, nib.GiftiImage):
        arr = img.agg_data()
    elif isinstance(img, NiftiImage):
        arr = img.get_fdata().astype(dtype) if dtype is not None else img.get_fdata()
    else:
        raise ValueError(f"Unsupported nibabel image: {type(img)}")

    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return DataOutput(array=arr, image=img if return_image else None)


def relabel_gifti(
    parcellation: str | Path,
    background: Iterable[str] | None = None,
) -> nib.GiftiImage:
    """Relabel a GIFTI parcellation so that label indices are consecutive.

    Loads the parcellation file, zeroes out any background labels found in
    the label table, then remaps the remaining indices to consecutive
    integers starting at ``1``. Returns a new ``GiftiImage`` with an
    updated data array and label table.

    Args:
        parcellation: Path to a single GIFTI parcellation file.
        background: Iterable of label names to treat as background and
            zero out. Defaults to ``PARC_IGNORE``.

    Returns:
        A ``GiftiImage`` instance with consecutive label indices and an
        updated label table.

    Raises:
        ValueError: If loaded image is not a GiftiImage.
    """
    data, img = load_data(parcellation, return_image=True)
    if not isinstance(img, nib.GiftiImage):
        raise ValueError(f"Expected to load Gifti surface for: {parcellation}")
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
        data,
        intent="NIFTI_INTENT_LABEL",
        datatype="NIFTI_TYPE_INT32",
    )
    labeltable = nib.gifti.GiftiLabelTable()
    labeltable.labels = new_labels
    return nib.GiftiImage(darrays=[darr], labeltable=labeltable)