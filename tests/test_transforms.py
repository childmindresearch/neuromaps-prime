"""Tests for volumetric transformations using Neuromaps NHP."""

from pathlib import Path

import nibabel as nib
import pytest

from neuromaps_nhp.transforms import _extract_res, _vol_to_vol

CONTINUOUS_INTERPS = [
    "linear",
    "gaussian",
    "BSpline",
    "cosineWindowedSinc",
    "welchWindowedSinc",
    "hammingWindowedSinc",
    "lanczosWindowedSinc",
    "nearestNeighbor",
]

LABEL_INTERPS = ["multiLabel"]


class TestVolumetricTransform:
    """Unit tests for volumetric transformations using `_vol_to_vol`."""

    # Source images
    t1w_source = Path(
        "/Users/tamsin.rogers/Desktop/github/neuromaps/"
        "share_with_T1w/Inputs/D99/src-D99_res-0p25mm_T1w.nii"
    )
    label_source = Path(
        "/Users/tamsin.rogers/Desktop/github/neuromaps/"
        "share_with_T1w/atlas/D99_atlas_v2.0.nii"
    )

    # Target images
    target_same = Path(
        "/Users/tamsin.rogers/Desktop/github/neuromaps/"
        "share_with_T1w/Inputs/NMT2Sym/src-NMT2Sym_res-0p25mm_T1w.nii"
    )
    target_diff = Path(
        "/Users/tamsin.rogers/Desktop/github/neuromaps/"
        "share_with_T1w/Inputs/MEBRAINS/src-MEBRAINS_res-0p40mm_T1w.nii"
    )

    @pytest.mark.parametrize("target_attr", ["target_same", "target_diff"])
    @pytest.mark.parametrize("interp", CONTINUOUS_INTERPS)
    def test_vol_to_vol_continuous(self, target_attr: str, interp: str) -> None:
        """Test continuous interpolators for volumetric transforms."""
        target_file = getattr(self, target_attr)

        # Interpolators that should raise NotImplementedError
        if interp in {"gaussian", "BSpline"}:
            with pytest.raises(NotImplementedError):
                _vol_to_vol(self.t1w_source, target_file, interp=interp)
            return

        # All other continuous interpolators should run normally
        result = _vol_to_vol(self.t1w_source, target_file, interp=interp)

        assert result.exists(), f"Output file not created for {interp}"
        img = nib.load(result)
        assert isinstance(img, nib.Nifti1Image)
        assert _extract_res(result) == _extract_res(target_file)

    @pytest.mark.parametrize("target_attr", ["target_same", "target_diff"])
    @pytest.mark.parametrize("interp", LABEL_INTERPS)
    def test_vol_to_vol_label(self, target_attr: str, interp: str) -> None:
        """Test label-based interpolators using a parcellation source image."""
        target_file = getattr(self, target_attr)

        # Label interpolators that should raise NotImplementedError
        if interp == "multiLabel":
            with pytest.raises(NotImplementedError):
                _vol_to_vol(
                    self.label_source,
                    target_file,
                    interp=interp,
                    label=self.label_source,
                )
            return

        # (Future) other label interpolators could be tested here
        result = _vol_to_vol(
            self.label_source,
            target_file,
            interp=interp,
            label=self.label_source,
        )
        assert result.exists()
        img = nib.load(result)
        assert isinstance(img, nib.Nifti1Image)
        assert _extract_res(result) == _extract_res(target_file)
