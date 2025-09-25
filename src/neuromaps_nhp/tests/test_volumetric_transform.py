from pathlib import Path
from neuromaps_nhp.volumetric_transformation import _vol_to_vol
import nibabel as nib
import subprocess

def test_vol_to_vol():
    source_file = Path("src/neuromaps_nhp/tests/test-data/Inputs/D99/src-D99_res-0p25mm.T1w.nii")
    target_file = Path("src/neuromaps_nhp/tests/test-data/Outputs/src-Yerkes19_to-D99_res-0p25mm_mode-image_desc-Composite_xfm.nii")

    resampled_img = _vol_to_vol(source_file, target_file, method='linear')

    assert resampled_img is not None
    assert isinstance(resampled_img, nib.Nifti1Image)

    # check file using Workbench
    def get_voxel_spacing(nii_file: Path):
        cmd = ["wb_command", "-file-information", str(nii_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"wb_command failed: {result.stderr}")

        for line in result.stdout.splitlines():
            if "spacing:" in line.lower():
                # split 
                spacing_values = line.split(":", 1)[1].replace("mm", "").strip()
                return tuple(float(x.strip()) for x in spacing_values.split(","))
        raise ValueError(f"cannot determine resolution {nii_file}")

    # get source and transformed resolutions
    source_spacing = get_voxel_spacing(source_file)
    transformed_spacing = get_voxel_spacing(target_file)

    # check that output res = input res
    assert source_spacing == transformed_spacing, f"mismatch: {source_spacing} != {transformed_spacing}"
