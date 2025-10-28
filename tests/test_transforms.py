"""Tests for volumetric transformations using Neuromaps."""

from pathlib import Path
import nibabel as nib
import pytest

from neuromaps_prime.transforms import _extract_res, _vol_to_vol

# Interpolators that are currently implemented and should work
DEVELOPED_INTERPS = [
    "linear",
    "cosineWindowedSinc",
    "welchWindowedSinc",
    "hammingWindowedSinc",
    "lanczosWindowedSinc",
    "nearestNeighbor",
]

# Interpolators planned for future development (expected to raise errors)
FUTURE_INTERPS = [
    "gaussian",
    "BSpline",
    "multiLabel",
]


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
    @pytest.mark.parametrize("interp", DEVELOPED_INTERPS)
    def test_vol_to_vol_developed_interps(self, target_attr: str, interp: str) -> None:
        """Interpolators that are implemented and should complete successfully."""
        target_file = getattr(self, target_attr)
        src_file = self.t1w_source

        result = _vol_to_vol(src_file, target_file, interp=interp)
        assert result.exists(), f"Output file not created for {interp}"

        img = nib.load(result)
        assert isinstance(img, nib.Nifti1Image)
        assert _extract_res(result) == _extract_res(target_file)

    @pytest.mark.parametrize("target_attr", ["target_same", "target_diff"])
    @pytest.mark.parametrize("interp", FUTURE_INTERPS)
    def test_vol_to_vol_future_interps(self, target_attr: str, interp: str) -> None:
        """Interpolators planned for future support that should raise errors."""
        target_file = getattr(self, target_attr)
        src_file = self.t1w_source

        with pytest.raises(NotImplementedError):
            _vol_to_vol(
                src_file,
                target_file,
                interp=interp,
                label=(self.label_source if interp == "multiLabel" else None),
            )
