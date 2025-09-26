import subprocess
from pathlib import Path
import nibabel as nib
from neuromaps.utils import tmpname, run
import warnings


def _extract_res(nii_file: Path):
    """
    Extract voxel spacing from a NIfTI file using Workbench (wb_command).

    Parameters
    ----------
    nii_file : Path
        Path to a NIfTI file.

    Returns
    -------
    tuple of float
        The voxel spacing (dx, dy, dz) in mm.
    """
    cmd = ["wb_command", "-file-information", str(nii_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"wb_command failed: {result.stderr}")

    for line in result.stdout.splitlines():
        if "spacing:" in line.lower():
            spacing_values = line.split(":", 1)[1].replace("mm", "").strip()
            return tuple(float(x.strip()) for x in spacing_values.split(","))
    raise ValueError(f"Cannot determine resolution for {nii_file}")


def _vol_to_vol(source, target, method='linear', allow_res_mismatch=True):
    """
    Use niwrap to implement antsApplyTransforms to brain volumes in Neuromaps NHP.

    Parameters
    ----------
    source : str or Path
        Path to the source NIfTI volume.
    target : str or Path
        Path to the target NIfTI volume.
    method : {'linear', 'nearest'}, optional
        Interpolation method. Default is 'linear'.
    allow_res_mismatch : bool, optional
        If True, will run transformation even if source and target resolutions differ.
        Default is True.

    Returns
    -------
    transformed : nib.Nifti1Image
        Source volume resampled to the target space.
    """

    # ensure absolute paths
    source = Path(source).resolve()
    target = Path(target).resolve()

    # get voxel spacing using Workbench
    src_res = _extract_res(source)
    trg_res = _extract_res(target)

    # warn if source and target voxel spacings differ
    if src_res != trg_res:
        msg = f"Source resolution {src_res} and target resolution {trg_res} differ."
        if allow_res_mismatch:
            warnings.warn(msg + " : ", UserWarning)
        else:
            raise ValueError(msg + " Set allow_res_mismatch=True to override.")

    tmp_out = tmpname('.nii')
    interp_map = {'linear': 'Linear', 'nearest': 'NearestNeighbor'}

    # run antsApplyTransforms
    cmd = f"antsApplyTransforms -d 3 -i {source} -r {target} -o {tmp_out} -n {interp_map[method]}"
    run(cmd, quiet=True)

    # load transformed output
    transformed = nib.load(tmp_out)

    # remove temp file
    tmp_out.unlink(missing_ok=True)

    return transformed
