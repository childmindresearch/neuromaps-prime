"""Helper functions for analysis unit tests."""

from pathlib import Path

import nibabel as nib
import numpy as np


def _make_gifti_surface(coords: np.ndarray, faces: np.ndarray, path: Path) -> None:
    """Write a GIFTI surface file."""
    ptarr = nib.gifti.GiftiDataArray(
        coords.astype(np.float32), intent="NIFTI_INTENT_POINTSET"
    )
    tris = nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE")
    nib.GiftiImage(darrays=[ptarr, tris]).to_filename(path)


def _make_gifti_parc(
    data: np.ndarray, labels: list[tuple[int, str]], path: Path
) -> None:
    """Write a GIFTI parcellation file with label table."""
    darr = nib.gifti.GiftiDataArray(
        data.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32"
    )
    lt = nib.gifti.GiftiLabelTable()
    for key, name in labels:
        lbl = nib.gifti.GiftiLabel(key=key)
        lbl.label = name
        lt.labels.append(lbl)
    nib.GiftiImage(darrays=[darr], labeltable=lt).to_filename(path)
