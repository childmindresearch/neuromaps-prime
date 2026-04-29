"""Tests for surface transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from neuromaps_prime.transforms.volume import vol_to_vol

if TYPE_CHECKING:
    from pathlib import Path

    from neuromaps_prime.graph import NeuromapsGraph


@pytest.mark.usefixtures("require_ants")
class TestVolumetricTransformIntegration:
    """Integration tests calling ANTs and using real data."""

    @staticmethod
    def _extract_res(nii_file: Path) -> tuple[float]:
        """Extract voxel spacing from a NIfTI file."""
        import nibabel.nifti1

        img = nibabel.nifti1.load(nii_file)
        return img.header.get_zooms()[:3]

    def test_vol_to_vol_real_data(self, tmp_path: Path, graph: NeuromapsGraph) -> None:
        """Integration test with real ANTs processing using actual file paths."""
        source_atlas = graph.fetch_volume_atlas(
            space="D99", resolution="250um", resource_type="T1w"
        )
        target_atlas = graph.fetch_volume_atlas(
            space="NMT2Sym", resolution="250um", resource_type="T1w"
        )
        assert source_atlas is not None
        assert target_atlas is not None
        source_path = source_atlas.fetch()
        target_path = target_atlas.fetch()

        result = vol_to_vol(
            source=source_path,
            target=target_path,
            out_fpath=str(tmp_path / "test.nii.gz"),
            interp="linear",
        )
        assert result.exists()
        assert self._extract_res(result) == self._extract_res(target_path)


@pytest.mark.skip(reason="No volumetric data to project.")
class TestVolumeToSurfaceProjectionIntegration:
    """Integration tests for volume to surface projection."""
