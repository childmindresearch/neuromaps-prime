"""Functions for volumetric transformations using niwrap."""

from pathlib import Path
from typing import cast

import nibabel as nib
from nibabel.nifti1 import Nifti1Header
from niwrap import ants

# Future: add something to our own utils sub-module for setting up niwrap


def _extract_res(nii_file: Path) -> tuple[float, float, float]:
    """Extract voxel spacing from a NIfTI file.

    Args:
        nii_file: Path to a NIfTI file from which to extract voxel dimensions.

    Returns:
        A tuple of voxel spacing values (x, y, z) in millimeters.
    """
    # Later: handle NIfTI-2 files.

    img = nib.load(nii_file)
    # For type checking, has no effect at run-time.
    header = cast(Nifti1Header, img.header)
    return header.get_zooms()[:3]

def _vol_to_vol(
    source: Path,
    target: Path,
    interp: str = "linear",
    label: Path | None = None
    ) -> Path:
    """Transform a volumetric image from source space to target space.

    Args:
        source: Path to the source NIfTI volume to be transformed.
        target: Path to the target NIfTI volume defining the reference space.
        interp: Interpolation method to use.
        label: Optional path to a label image (optional for multiLabel).

    Returns:
        Path to the transformed NIfTI file written to disk.

    Raises:
        ValueError: unsupported interpolator.
    """
    INTERP_PARAMS = {
        "linear": ants.ants_apply_transforms_linear_params,
        "nearestNeighbor": ants.ants_apply_transforms_nearest_neighbor_params,
        "multiLabel": ants.ants_apply_transforms_multi_label_params,
        "gaussian": ants.ants_apply_transforms_gaussian_params,
        "BSpline": ants.ants_apply_transforms_bspline_params,
        "cosineWindowedSinc": ants.ants_apply_transforms_cosine_windowed_sinc_params,
        "welchWindowedSinc": ants.ants_apply_transforms_welch_windowed_sinc_params,
        "hammingWindowedSinc": ants.ants_apply_transforms_hamming_windowed_sinc_params,
        "lanczosWindowedSinc": ants.ants_apply_transforms_lanczos_windowed_sinc_params,
    }

    if interp not in INTERP_PARAMS:
        raise ValueError("Unsupported interpolator.")

    out_file = target.parent / f"{source.stem}_to_{target.stem}.nii.gz"

    # Handle interpolators not yet fully supported 
    if interp in ["BSpline", "gaussian", "multiLabel"]:
        raise NotImplementedError(
            f"The '{interp}' interpolation method is not yet implemented."
        )

    ants.ants_apply_transforms(
        input_image=source,
        reference_image=target,
        output=ants.ants_apply_transforms_warped_output_params(str(out_file)),
        interpolation=INTERP_PARAMS[interp](),
    )

    return out_file