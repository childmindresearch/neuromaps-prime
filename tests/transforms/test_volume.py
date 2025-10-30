"""Tests for volumetric transformations using Neuromaps."""

from pathlib import Path

import nibabel as nib
import pytest

from neuromaps_prime.transforms.volume import _vol_to_vol

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
FUTURE_INTERPS = ["gaussian", "BSpline", "multiLabel"]


@pytest.mark.usefixtures("require_ants")
class TestVolumetricTransform:
    """Unit tests for volumetric transformations using `_vol_to_vol`."""

    @pytest.fixture
    def vol_paths(self, data_dir: Path) -> dict[str, Path]:
        """Provide all possible source and target paths for tests."""
        return {
            "t1w_source": data_dir
            / "share"
            / "Inputs"
            / "D99"
            / "src-D99_res-0p25mm_T1w.nii",
            "label_source": data_dir / "resources" / "D99" / "D99_atlas_v2.0.nii.gz",
            "target_same": data_dir
            / "share"
            / "Inputs"
            / "NMT2Sym"
            / "src-NMT2Sym_res-0p25mm_T1w.nii",
            "target_diff": data_dir
            / "share"
            / "Inputs"
            / "MEBRAINS"
            / "src-MEBRAINS_res-0p40mm_T1w.nii",
        }

    def _extract_res(self, nii_file: Path) -> tuple[float, float, float]:
        """Extract voxel spacing from a NIfTI file."""
        img = nib.load(nii_file)
        return img.header.get_zooms()[:3]

    @pytest.mark.parametrize("target_key", ["target_same", "target_diff"])
    @pytest.mark.parametrize("interp", DEVELOPED_INTERPS)
    def test_vol_to_vol_developed_interps(
        self, tmp_path: Path, vol_paths: dict[str, Path], target_key: str, interp: str
    ) -> None:
        """Interpolators that are implemented and should complete successfully."""
        target_file = vol_paths[target_key]
        result = _vol_to_vol(
            vol_paths["t1w_source"],
            target_file,
            out_fpath=str(tmp_path / "test.nii.gz"),
            interp=interp,
        )
        assert result.exists(), f"Output file not created for {interp}"
        assert self._extract_res(result) == self._extract_res(target_file)

    @pytest.mark.parametrize("target_key", ["target_same", "target_diff"])
    @pytest.mark.parametrize("interp", FUTURE_INTERPS)
    def test_vol_to_vol_future_interps(
        self, vol_paths: dict[str, Path], target_key: str, interp: str
    ) -> None:
        """Interpolators planned for future support that should raise errors."""
        target_file = vol_paths[target_key]
        label_file = vol_paths["label_source"] if interp == "multiLabel" else None
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            _vol_to_vol(
                vol_paths["t1w_source"],
                target_file,
                out_fpath="test.nii.gz",
                interp=interp,
                label=label_file,
            )

    @pytest.mark.parametrize("interp", ("foo", "bar"))
    def test_vol_to_vol_unsupported_interps(
        self, vol_paths: dict[str, Path], interp: str
    ):
        """Test for unsupported interpolators."""
        with pytest.raises(ValueError, match="Unsupported"):
            _vol_to_vol(
                vol_paths["t1w_source"],
                vol_paths["target_same"],
                out_fpath="test.nii.gz",
                interp=interp,
            )
