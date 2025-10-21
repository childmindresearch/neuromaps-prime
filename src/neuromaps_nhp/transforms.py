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
    # For type checking, has no effect at run-time.
    # Later: handle NIfTI-2 files.

    img = nib.load(nii_file)
    header = cast(Nifti1Header, img.header)
    return header.get_zooms()[:3]

def _vol_to_vol(source: Path, target: Path, interpolator: str) -> Path:
    """Transform a volumetric image from source space to target space.

    Args:
        source: Path to the source NIfTI volume to be transformed.
        target: Path to the target NIfTI volume defining the reference space.
        interpolator: Interpolation method to use. Must be one of:
            {'linear', 'nearestNeighbor', 'multiLabel', 'gaussian', 'bSpline',
             'cosineWindowedSinc', 'welchWindowedSinc', 'hammingWindowedSinc',
             'lanczosWindowedSinc', 'genericLabel'}

    Returns:
        Path to the transformed NIfTI file written to disk.

    Raises:
        ValueError: If an unsupported interpolator is provided.
    """
    accepted_interpolators = {
        "linear", "nearestNeighbor", "multiLabel", "gaussian", "bSpline",
        "cosineWindowedSinc", "welchWindowedSinc",
        "hammingWindowedSinc", "lanczosWindowedSinc", "genericLabel"
    }

    if interpolator not in accepted_interpolators:
        raise ValueError(
            f"Unsupported interpolator '{interpolator}'. "
            f"Must be one of {sorted(accepted_interpolators)}."
        )

    out_file = target.parent / f"{source.stem}_to_{target.stem}.nii.gz"

    # Set interpolation parameters based on method
    if interpolator == "linear":
        interp = ants.ants_apply_transforms_linear_params()
    elif interpolator == "nearestNeighbor":
        interp = ants.ants_apply_transforms_nearest_neighbor_params()
    else:
        interp = interpolator

    output = ants.ants_apply_transforms_warped_output_params(str(out_file))

    ants.ants_apply_transforms(
        input_image=source,
        reference_image=target,
        output=output,
        interpolation=interp,
    )

    return out_file

