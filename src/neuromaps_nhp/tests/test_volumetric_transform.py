from pathlib import Path
from neuromaps_nhp.transforms import _vol_to_vol, _extract_res
import nibabel as nib

'''Test volumetric transformation from source to target space.'''
def test_vol_to_vol():
    source_file = Path(
        "/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/D99/src-D99_res-0p25mm.T1w.nii"
    )
    target_file_same = Path(
        "/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/NMT2Sym/src-NMT2Sym_res-0p25mm.T1w.nii"
    )
    target_file_diff = Path(
        "/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/MEBRAINS/src-MEBRAINS_res-0p40mm.T1w.nii"
    )

    # ---- Test 1: same-resolution source vs. target ----
    resampled_same_path = _vol_to_vol(source_file, target_file_same)
    assert resampled_same_path.exists()

    resampled_same_img = nib.load(resampled_same_path)
    assert isinstance(resampled_same_img, nib.Nifti1Image)

    # compare resolutions
    same_source_spacing = _extract_res(source_file)
    same_resampled_spacing = _extract_res(resampled_same_path)
    assert same_source_spacing == same_resampled_spacing, (
        f"Resolution mismatch (same-res test): {same_source_spacing} != {same_resampled_spacing}"
    )

    # ---- Test 2: different-resolution source vs. target ----
    resampled_diff_path = _vol_to_vol(source_file, target_file_diff)
    assert resampled_diff_path.exists()

    resampled_diff_img = nib.load(resampled_diff_path)
    assert isinstance(resampled_diff_img, nib.Nifti1Image)

    # transformed output should match the target spacing
    diff_target_spacing = _extract_res(target_file_diff)
    diff_resampled_spacing = _extract_res(resampled_diff_path)
    assert diff_resampled_spacing == diff_target_spacing, (
        f"Resolution mismatch (diff-res test): {diff_resampled_spacing} != {diff_target_spacing}"
    )
