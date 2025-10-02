from pathlib import Path
import nibabel as nib
from neuromaps_nhp.transforms import _vol_to_vol, _extract_res

class TestVolToVol:
    """Class-based tests for volumetric transformation using _vol_to_vol."""

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

    def test_same_resolution(self):
        """Test transformation when source and target have the same resolution."""
        result_path = _vol_to_vol(self.source_file, self.target_same)
        assert result_path.exists(), "Transformed file was not created"
        img = nib.load(result_path)
        assert isinstance(img, nib.Nifti1Image), "Output is not a Nifti1Image"
        assert _extract_res(result_path) == _extract_res(self.target_same), (
            f"Resolution mismatch: {_extract_res(result_path)} != {_extract_res(self.target_same)}"
        )

    def test_different_resolution(self):
        """Test transformation when source and target have different resolutions."""
        result_path = _vol_to_vol(self.source_file, self.target_diff)
        assert result_path.exists(), "Transformed file was not created"
        img = nib.load(result_path)
        assert isinstance(img, nib.Nifti1Image), "Output is not a Nifti1Image"
        assert _extract_res(result_path) == _extract_res(self.target_diff), (
            f"Resolution mismatch: {_extract_res(result_path)} != {_extract_res(self.target_diff)}"
        )
