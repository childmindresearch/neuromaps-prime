from pathlib import Path
import nibabel as nib
import pytest
from neuromaps_nhp.transforms import _vol_to_vol, _extract_res

class TestVolumetricTransform:
    """Class-based tests for volumetric transformation using _vol_to_vol."""

    # future- set up fixtures or use fetchers and just pass those along instead of hardcoding
    def setup_class(self):
        """Set up file paths for source and target volumes."""
        self.source_file = Path(
            "/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/D99/src-D99_res-0p25mm.T1w.nii"
        )
        self.target_same = Path(
            "/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/NMT2Sym/src-NMT2Sym_res-0p25mm.T1w.nii"
        )
        self.target_diff = Path(
            "/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/MEBRAINS/src-MEBRAINS_res-0p40mm.T1w.nii"
        )

    @pytest.mark.parametrize("target_attr", ["target_same", "target_diff"])
    def test_vol_to_vol_resolution(self, target_attr):

        target_file = getattr(self, target_attr)
        result_path = _vol_to_vol(self.source_file, target_file)
        
        assert result_path.exists(), "Transformed file was not created"
        
        img = nib.load(result_path)
        assert isinstance(img, nib.Nifti1Image), "Output is not a Nifti1Image"
        
        src_res = _extract_res(result_path)
        trg_res = _extract_res(target_file)
        assert src_res == trg_res, f"Resolution mismatch: {src_res} != {trg_res}"
