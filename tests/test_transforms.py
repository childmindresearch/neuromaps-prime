"""Tests for volumetric transformations using Neuromaps NHP."""

from pathlib import Path

import nibabel as nib
import pytest

from neuromaps_nhp.transforms import _extract_res, _vol_to_vol


class TestVolumetricTransform:
    """Unit tests for volumetric transformations using the `_vol_to_vol` function.

    These tests ensure that transformed volumetric images:
      - Are successfully created on disk.
      - Maintain correct voxel resolution matching the target image.
      - Work for both 'linear' and 'nearestNeighbor' interpolators.
    """

    def setup_class(self) -> None:
        """Set up file paths for source and target NIfTI volumes."""
        self.source_file = Path(
            "/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/D99/src-D99_res-0p25mm_T1w.nii"
        )
        self.target_same = Path(
            "/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/NMT2Sym/src-NMT2Sym_res-0p25mm_T1w.nii"
        )
        self.target_diff = Path(
            "/Users/tamsin.rogers/Desktop/github/neuromaps/share_with_T1w/Inputs/MEBRAINS/src-MEBRAINS_res-0p40mm_T1w.nii"
        )

    @pytest.mark.parametrize("target_attr", ["target_same", "target_diff"])
    @pytest.mark.parametrize("interp", ["linear", "nearestNeighbor"])
    def test_vol_to_vol_resolution(self, target_attr: str, interp: str) -> None:
        """Test that `_vol_to_vol` produces a file matching the target resolution.
        
        For both linear and nearest-neighbor interpolation.

        Args:
            target_attr: Name of the target attribute ('target_same' or 'target_diff').
            interp: Interpolation method ('linear' or 'nearestNeighbor').

        Raises:
            AssertionError: If the output file does not exist, is not a NIfTI image,
                or if its resolution does not match the target file.
        """
        target_file = getattr(self, target_attr)
        result = _vol_to_vol(self.source_file, target_file, interp=interp)

        # Verify that the output file was successfully created.
        assert result.exists(), f"Transformed file was not created for {interp}"

        img = nib.load(result)
        assert isinstance(img, nib.Nifti1Image), f"Output not Nifti1Image for {interp}"

        src_res = _extract_res(result)
        trg_res = _extract_res(target_file)
        assert src_res == trg_res, (
            f"Resolution mismatch ({interp}): {src_res} != {trg_res}"
        )
