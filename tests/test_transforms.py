"""Tests for volumetric transformations using Neuromaps NHP."""

from pathlib import Path
import nibabel as nib
import pytest

from neuromaps_nhp.transforms import _extract_res, _vol_to_vol


CONTINUOUS_INTERPS = [
    "linear",
    "gaussian",
    "bSpline",
    "cosineWindowedSinc",
    "welchWindowedSinc",
    "hammingWindowedSinc",
    "lanczosWindowedSinc",
]

LABEL_INTERPS = ["nearestNeighbor", "multiLabel", "genericLabel"]


class TestVolumetricTransform:
    """Unit tests for volumetric transformations using `_vol_to_vol`."""

    # Source images
    t1w_source = Path("/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/D99/src-D99_res-0p25mm_T1w.nii") 
    label_source = Path("/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/atlas/D99_atlas_v2.0.nii")

    # Target images
    target_same = Path("/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/NMT2Sym/src-NMT2Sym_res-0p25mm_T1w.nii")
    target_diff = Path("/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/MEBRAINS/src-MEBRAINS_res-0p40mm_T1w.nii")

    @pytest.mark.parametrize("target_attr", ["target_same", "target_diff"])
    @pytest.mark.parametrize("interp", CONTINUOUS_INTERPS)
    def test_vol_to_vol_continuous(self, target_attr: str, interp: str) -> None:
        """Test continuous interpolators for volumetric transforms."""
        target_file = getattr(self, target_attr)

        if interp in ["gaussian", "bSpline"]:
            pytest.skip(f"Skipping {interp}: not supported yet.")

        result = _vol_to_vol(self.t1w_source, target_file, interp=interp)

        assert result.exists()
        img = nib.load(result)
        assert isinstance(img, nib.Nifti1Image)
        assert _extract_res(result) == _extract_res(target_file)

    @pytest.mark.parametrize("target_attr", ["target_same", "target_diff"])
    @pytest.mark.parametrize("interp", LABEL_INTERPS)
    def test_vol_to_vol_label(self, target_attr: str, interp: str) -> None:
        """Test label-based interpolators using a parcellation source image."""
        target_file = getattr(self, target_attr)

        if interp == "nearestNeighbor":
            result = _vol_to_vol(self.label_source, target_file, interp=interp)
        elif interp == "multiLabel":
            result = _vol_to_vol(self.label_source, target_file, interp=interp, label=self.label_source)
        else:  # genericLabel
            result = _vol_to_vol(self.label_source, target_file, interp=interp)

        assert result.exists()
        img = nib.load(result)
        assert isinstance(img, nib.Nifti1Image)
        assert _extract_res(result) == _extract_res(target_file)
