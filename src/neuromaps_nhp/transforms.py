"""Functions for volumetric transformations niwrap."""

from pathlib import Path
from typing import cast

import nibabel as nib
from nibabel.nifti1 import Nifti1Header
from niwrap import ants

# future- add something to our own utils sub-module for setting up niwrap


def _extract_res(nii_file: Path) -> tuple[float, float, float]:
    """Extract voxel spacing from a NIfTI file using nibabel."""
    img = nib.load(nii_file)
    header = cast(Nifti1Header, img.header)
    return header.get_zooms()[:3]


def _vol_to_vol(source: Path, target: Path) -> Path:
    """Transform a volumetric image from source space to target space."""
    # later - include in the networkx graph, fetch and assert that the path exists

    """
    Can add a warning in the future if we want to allow for volumes to be resampled 
    back to the original resolution in the target space,but generally, we would expect 
    the output to be in the resolution of the target image. Given we are transforming 
    to the target space, the warning is left out for now.
    """

    out_file = target.parent / f"{source.stem}_to_{target.stem}.nii.gz"
    interp = ants.ants_apply_transforms_linear_params()
    output = ants.ants_apply_transforms_warped_output_params(str(out_file))

    ants.ants_apply_transforms(
        input_image=source, reference_image=target, output=output, interpolation=interp
    )

    return out_file
