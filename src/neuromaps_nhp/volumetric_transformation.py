import re
from pathlib import Path
import nibabel as nib
from neuromaps.utils import tmpname, run
import warnings

def _vol_to_vol(source, target, method='linear', allow_res_mismatch=False):
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
        Default is False (raises a warning and proceeds anyway).

    Returns
    -------
    transformed : nib.Nifti1Image
        Source volume resampled to the target space.
    """
    source = Path(source).resolve()
    target = Path(target).resolve()

    # helper: extract resolution from filename
    def _extract_res(fname: Path) -> str:
        fname = Path(fname).name
        m = re.search(r"_res-([0-9p]+mm)", fname)
        if not m:
            raise ValueError(f"Cannot extract resolution from filename: {fname}")
        return m.group(1)

    src_res = _extract_res(source)
    trg_res = _extract_res(target)

    if src_res != trg_res:
        msg = f"Source ({src_res}) and target ({trg_res}) resolutions differ."
        if allow_res_mismatch:
            warnings.warn(msg + " Proceeding with transformation.", UserWarning)
        else:
            warnings.warn(msg + " Transformation may resample differently than intended.", UserWarning)

    # if source and target are identical resolution, return the source
    if src_res == trg_res:
        return nib.load(source)

    if method not in ('linear', 'nearest'):
        raise ValueError(f"Invalid method: {method}. Must be 'linear' or 'nearest'.")

    tmp_out = tmpname('.nii')
    interp_map = {'linear': 'Linear', 'nearest': 'NearestNeighbor'}

    # run antsApplyTransforms
    cmd = f"antsApplyTransforms -d 3 -i {source} -r {target} -o {tmp_out} -n {interp_map[method]}"
    run(cmd, quiet=True)

    # Load transformed output
    transformed = nib.load(tmp_out)

    # Clean up temporary file
    tmp_out.unlink(missing_ok=True)

    return transformed
