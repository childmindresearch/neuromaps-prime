from pathlib import Path
from niwrap import ants     #add to utils
import nibabel as nib

'''Extract voxel spacing from a NIfTI file using wb_command.'''
def _extract_res(nii_file: Path):
    """Extract voxel spacing from a NIfTI file using nibabel."""
    img = nib.load(nii_file)
    return img.header.get_zooms()[:3]  

'''Transform a volumetric image from source space to target space.'''
def _vol_to_vol(source: Path, target: Path):

    # later - include in the networkx graph, fetch and assert that the path exists

    '''
    Can add a warning in the future if we want to allow for volumes to be resampled back 
    to the original resolution in the target space,but generally, we would expect the output 
    to be in the resolution of the target image. Given we are transforming to the target space, 
    the warning is left out for now.
    '''

    out_file = target.parent / f"{source.stem}_to_{target.stem}.nii.gz"
    interp = ants.ants_apply_transforms_linear_params()
    output = ants.ants_apply_transforms_warped_output_params(str(out_file))

    ants.ants_apply_transforms(
        input_image=source,
        reference_image=target,
        output=output,
        interpolation=interp,
        dimensionality=3
    )

    return out_file
