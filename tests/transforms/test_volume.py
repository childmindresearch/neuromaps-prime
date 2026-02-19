"""Tests for volumetric transformations using Neuromaps."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from niwrap import workbench

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.transforms.volume import (
    _NOT_IMPLEMENTED,
    INTERP_PARAMS,
    surface_project,
    vol_to_vol,
)

# Interpolators that are currently implemented and should work
DEVELOPED_INTERPS = [k for k in INTERP_PARAMS if k not in _NOT_IMPLEMENTED]


class TestVolumetricTransform:
    """Unit tests for volumetric transformations using `vol_to_vol`."""

    @pytest.fixture
    def mock_paths(self, tmp_path: Path) -> dict[str, Path]:
        """Create mock file paths for testing."""
        source = tmp_path / "source.nii.gz"
        target = tmp_path / "target.nii.gz"
        output = tmp_path / "output.nii.gz"

        source.touch()
        target.touch()

        return {"source": source, "target": target, "output": output}

    @pytest.fixture
    def mock_ants_transform(self, mock_paths: dict[str, Path]):
        """Fixture to mock ANTs transform with consistent behavior."""
        with patch(
            "neuromaps_prime.transforms.volume.ants.ants_apply_transforms"
        ) as mock_ants:
            mock_result = MagicMock()
            mock_result.output.output_image_outfile = str(mock_paths["output"])

            def create_output(*args, **kwargs) -> MagicMock:
                mock_paths["output"].touch()
                return mock_result

            mock_ants.side_effect = create_output
            yield mock_ants

    @pytest.mark.parametrize("interp", DEVELOPED_INTERPS)
    def test_vol_to_vol_implemented_interps(
        self, mock_ants_transform: MagicMock, mock_paths: dict[str, Path], interp: str
    ) -> None:
        """Test implemented interpolators."""
        result = vol_to_vol(
            source=mock_paths["source"],
            target=mock_paths["target"],
            out_fpath=str(mock_paths["output"]),
            interp=interp,
        )

        mock_ants_transform.assert_called_once()
        call_kwargs = mock_ants_transform.call_args.kwargs
        assert call_kwargs["input_image"] == mock_paths["source"]
        assert call_kwargs["reference_image"] == mock_paths["target"]
        assert "interpolation" in call_kwargs
        assert "output" in call_kwargs
        assert result == mock_paths["output"]
        assert result.exists()

    @pytest.mark.parametrize("interp", _NOT_IMPLEMENTED)
    def test_vol_to_vol_not_implemented_interps(
        self, mock_paths: dict[str, Path], interp: str
    ) -> None:
        """Test future interpolators for raising NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            vol_to_vol(
                source=mock_paths["source"],
                target=mock_paths["target"],
                out_fpath=str(mock_paths["output"]),
                interp=interp,
            )

    @pytest.mark.parametrize("interp", ["foo", "bar", "invalid"])
    def test_vol_to_vol_unsupported_interps(
        self, mock_paths: dict[str, Path], interp: str
    ) -> None:
        """Test unsupported interpolators for raising ValueError."""
        with pytest.raises(ValueError, match="Unsupported interpolator"):
            vol_to_vol(
                source=mock_paths["source"],
                target=mock_paths["target"],
                out_fpath=str(mock_paths["output"]),
                interp=interp,
            )

    @pytest.mark.parametrize(
        "interp_params",
        [
            {"sigma": 1.5, "alpha": 0.7},
            {},
            None,
        ],
        ids=["with_params", "empty_dict", "none"],
    )
    def test_interp_params_handling(
        self,
        mock_ants_transform: MagicMock,
        mock_paths: dict[str, Path],
        interp_params: dict[str, Any] | None,
    ):
        """Test various interp_params inputs are handled correctly."""
        result = vol_to_vol(
            source=mock_paths["source"],
            target=mock_paths["target"],
            out_fpath=str(mock_paths["output"]),
            interp="gaussian",
            interp_params=interp_params,
        )

        mock_ants_transform.assert_called_once()
        assert result.exists()

    @patch("neuromaps_prime.transforms.volume._get_interp_params")
    @patch("neuromaps_prime.transforms.volume.ants.ants_apply_transforms")
    def test_interp_params_called_correctly(
        self,
        mock_ants: MagicMock,
        mock_get_params: MagicMock,
        mock_paths: dict[str, Path],
    ):
        """Test interpolation param is correctly called with args."""
        mock_get_params.return_value = {"mocked": "params"}

        mock_result = MagicMock()
        mock_result.output.output_image_outfile = str(mock_paths["output"])

        def create_output(*args, **kwargs) -> MagicMock:
            mock_paths["output"].touch()
            return mock_result

        mock_ants.side_effect = create_output

        interp_params = {"sigma": 1.5, "alpha": 0.7}
        vol_to_vol(
            source=mock_paths["source"],
            target=mock_paths["target"],
            out_fpath=str(mock_paths["output"]),
            interp="gaussian",
            interp_params=interp_params,
        )
        mock_get_params.assert_called_once_with("gaussian", interp_params)
        call_kwargs = mock_ants.call_args.kwargs
        assert call_kwargs["interpolation"] == mock_get_params.return_value


@pytest.mark.usefixtures("require_ants")
class TestVolumetricTransformIntegration:
    """Integration tests calling ANTs and using real data."""

    @staticmethod
    def _extract_res(nii_file: Path) -> tuple[float, float, float]:
        """Extract voxel spacing from a NIfTI file."""
        import nibabel as nib

        img = nib.load(nii_file)
        return img.header.get_zooms()[:3]

    def test_vol_to_vol_real_data(self, tmp_path: Path, graph: NeuromapsGraph) -> None:
        """Integration test with real ANTs processing using actual file paths."""
        source_atlas = graph.fetch_volume_atlas(
            space="D99", resolution="250um", resource_type="T1w"
        )
        target_atlas = graph.fetch_volume_atlas(
            space="NMT2Sym", resolution="250um", resource_type="T1w"
        )

        source_path = source_atlas.fetch()
        target_path = target_atlas.fetch()

        out_file = tmp_path / "test.nii.gz"

        result = vol_to_vol(
            source=source_path,
            target=target_path,
            out_fpath=out_file,
            interp="linear",
        )

        assert result.exists()
        assert self._extract_res(result) == self._extract_res(target_path)


class TestVolumeToSurfaceProjection:
    """Unit tests for projecting volumes to surface (`surface_project`)."""

    @pytest.fixture
    def mock_paths(self, tmp_path: Path) -> dict[str, Any]:
        """Create mock file paths for testing."""
        volume = tmp_path / "annotation.nii.gz"
        surface = tmp_path / "midthickness.surf.gii"
        ribbon_surfs = workbench.volume_to_surface_mapping_ribbon_constrained(
            inner_surf=tmp_path / "inner.surf.gii",
            outer_surf=tmp_path / "outer.surf.gii",
        )
        ribbon_surfs["inner-surf"].touch()
        ribbon_surfs["outer-surf"].touch()
        output = str(tmp_path / "annotation.surf.gii")

        return {
            "volume": volume,
            "surface": surface,
            "ribbon_surfs": ribbon_surfs,
            "output": output,
        }

    @pytest.fixture
    def mock_wb_project(self, mock_paths: dict[str, Path]):
        """Fixture to mock workbench volume to surface mapping behavior."""
        with patch(
            "neuromaps_prime.transforms.volume.workbench.volume_to_surface_mapping"
        ) as mock_wb:
            mock_result = MagicMock()
            mock_result.metric_out = mock_paths["output"]

            def create_output(*args, **kwargs) -> MagicMock:
                Path(mock_paths["output"]).touch()
                return mock_result

            mock_wb.side_effect = create_output
            yield mock_wb

    def test_surface_project(
        self, mock_wb_project: MagicMock, mock_paths: dict[str, Path]
    ) -> None:
        """Test volume-to-surface projection."""
        result = surface_project(
            volume=mock_paths["volume"],
            surface=mock_paths["surface"],
            ribbon_surfs=mock_paths["ribbon_surfs"],
            out_fpath=mock_paths["output"],
        )

        mock_wb_project.assert_called_once()
        assert str(result) == mock_paths["output"]
        assert result.exists()


@pytest.mark.skip(reason="No volumetric data to project.")
class TestVolumeToSurfaceProjectionIntegration:
    """Integration tests for volume to surface projection."""
