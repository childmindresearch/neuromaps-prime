"""Tests for volume transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel
import numpy as np
import pytest

from neuromaps_prime.analysis.images import load_data
from neuromaps_prime.transforms.volume import vol_to_vol

if TYPE_CHECKING:
    from pathlib import Path

    from neuromaps_prime.graph import NeuromapsGraph


class TestVolumetricTransformIntegration:
    """Integration tests calling ANTs and using real data."""

    @staticmethod
    def _extract_res(nii_file: Path) -> tuple[float, ...]:
        """Extract voxel spacing from a NIfTI file."""
        img = nibabel.nifti1.load(nii_file)
        return img.header.get_zooms()[:3]

    def test_vol_to_vol_real_data(self, graph: NeuromapsGraph) -> None:
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
            out_fpath="test.nii.gz",
            interp="linear",
        )
        assert result.exists()
        assert self._extract_res(result) == self._extract_res(target_path)

    def test_vol_to_vol_transformer(self, graph: NeuromapsGraph) -> None:
        """Test volume_to_volume transformer."""
        _source_space = "Yerkes19"

        input_file = graph.fetch_volume_atlas(
            space=_source_space, resolution="500um", resource_type="T1w"
        )
        assert input_file is not None
        input_fpath = input_file.fetch()
        output = graph.volume_to_volume_transformer(
            input_file=input_fpath,
            source_space=_source_space,
            target_space="NMT2Sym",
            resolution="250um",
            resource_type="composite",
            output_file_path="test_output.nii.gz",
        )
        assert output.exists()


class TestVolumeToSurfaceProjectionIntegration:
    """Integration tests for volume to surface projection."""

    SOURCE = "D99"
    TARGET = "Yerkes19"

    @pytest.mark.parametrize(
        ("transformer_type", "annotation", "suffix"),
        [("label", "PC_hemi_R", "label.gii"), ("metric", "MTR", "func.gii")],
        ids=["label", "metric"],
    )
    def test_volume_to_surface(
        self, graph: NeuromapsGraph, transformer_type: str, annotation: str, suffix: str
    ) -> None:
        """Project a D99 annotation onto the surface, then resample to Yerkes19."""
        resource = graph.fetch_volume_annotation(
            space=self.SOURCE, label=annotation, resolution="250um"
        )
        assert resource is not None
        input_fpath = resource.fetch()

        result = graph.volume_to_surface_transformer(
            transformer_type=transformer_type,
            input_file=input_fpath,
            source_space=self.SOURCE,
            target_space=self.TARGET,
            hemisphere="right",
            output_file_path=f"{self.SOURCE}_to_{self.TARGET}.{suffix}",
            add_edge=False,
        )
        assert result.path is not None
        assert result.path.exists()

        data = load_data(result.path).array
        assert data.size > 0
        assert np.all(np.isfinite(data))
        if transformer_type == "label":  # label resampling keeps whole-number labels
            assert np.all(data == np.rint(data))
