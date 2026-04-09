"""Tests associated with surface-to-volume transformation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from neuromaps_prime.graph import NeuromapsGraph


class BasicParams(NamedTuple):
    """Test param object."""

    input_file: Path
    ref_volume: Path
    source_space: str
    target_space: str
    hemisphere: Literal["left", "right"]
    output_file_path: str
    source_density: str
    target_density: str | None


class TestTransformSurfaceToVolume:
    """Unit tests for SurfaceTransformOps.transform_surface_to_volume."""

    @pytest.fixture
    def mock_ops(self, graph: NeuromapsGraph) -> NeuromapsGraph:
        """Forcefully mock surface operations methods."""
        graph.surface_ops.cache = MagicMock()
        graph.surface_ops.utils = MagicMock()

        # This bypasses Pydantic's "object has no field" check
        object.__setattr__(graph.surface_ops, "transform_surface", MagicMock())
        return graph

    @pytest.fixture
    def basic_params(self, tmp_path: Path) -> BasicParams:
        """Basic parameters for surface-to-volume transformation testing."""
        input_file = tmp_path / "in.func.gii"
        input_file.touch()
        ref_volume = tmp_path / "ref.nii.gz"
        ref_volume.touch()
        return BasicParams(
            input_file=input_file,
            ref_volume=ref_volume,
            source_space="fsLR",
            target_space="MNI152",
            hemisphere="left",
            output_file_path=str(tmp_path / "out.nii.gz"),
            source_density="32k",
            target_density="2mm",
        )

    def make_atlas_side_effect(self, tmp_path: Path) -> Callable:
        """Create a side effect returning valid atlas objects."""

        def get_surface_atlas_side_effect(
            space: str, density: str, hemisphere: str, resource_type: str
        ) -> MagicMock:
            fname = f"space-{space}_den-{density}_hemi-{hemisphere}_{resource_type}"
            surf_file = tmp_path / f"{fname}.surf.gii"
            surf_file.touch()
            return MagicMock(file_path=surf_file)

        return get_surface_atlas_side_effect

    @pytest.mark.parametrize("transformer_type", ["metric", "label"])
    def test_happy_path(
        self,
        mock_ops: NeuromapsGraph,
        basic_params: BasicParams,
        transformer_type: Literal["metric", "label"],
        tmp_path: Path,
    ) -> None:
        """Metric and label paths call correct workbench fn and return volume path."""
        expected_ext = "label.gii" if transformer_type == "label" else "shape.gii"
        out_surface = tmp_path / f"out.{expected_ext}"

        mock_ops.surface_ops.transform_surface.return_value = out_surface
        mock_ops.surface_ops.cache.get_surface_atlas.side_effect = (
            self.make_atlas_side_effect(tmp_path)
        )

        expected_out = Path(basic_params.output_file_path)
        wb_fn_name = f"{transformer_type}_to_volume_mapping"
        wb_patch = f"neuromaps_prime.graph.transforms.surface.workbench.{wb_fn_name}"

        with patch(
            wb_patch, return_value=MagicMock(volume_out=expected_out)
        ) as mock_wb:
            result = mock_ops.surface_ops.transform_surface_to_volume(
                transformer_type=transformer_type, **basic_params._asdict()
            )

        assert result == expected_out
        mock_ops.surface_ops.transform_surface.assert_called_once()
        assert mock_ops.surface_ops.transform_surface.call_args.kwargs[
            "output_file_path"
        ].endswith(expected_ext)
        mock_wb.assert_called_once()

    def test_target_density_logic(
        self, mock_ops: NeuromapsGraph, basic_params: BasicParams, tmp_path: Path
    ) -> None:
        """Test target_density fallback to highest available when None."""
        params = basic_params._replace(target_density=None)
        mock_ops.surface_ops.transform_surface.return_value = tmp_path / "out.shape.gii"
        mock_ops.surface_ops.utils.find_highest_density.return_value = "164k"
        mock_ops.surface_ops.cache.get_surface_atlas.side_effect = (
            self.make_atlas_side_effect(tmp_path)
        )

        wb_patch = (
            "neuromaps_prime.graph.transforms.surface.workbench"
            ".metric_to_volume_mapping"
        )
        with patch(
            wb_patch, return_value=MagicMock(volume_out=tmp_path / "out.nii.gz")
        ):
            mock_ops.surface_ops.transform_surface_to_volume(
                transformer_type="metric", **params._asdict()
            )

        mock_ops.surface_ops.utils.find_highest_density.assert_called_once_with(
            space=params.target_space
        )
        # Verify get_surface_atlas used the 'highest' density found
        _, kwargs = mock_ops.surface_ops.cache.get_surface_atlas.call_args
        assert kwargs["density"] == "164k"

    def test_raises_on_failed_surface_transform(
        self, mock_ops: NeuromapsGraph, basic_params: BasicParams
    ) -> None:
        """FileNotFoundError raised when the intermediate surface transform fails."""
        mock_ops.surface_ops.transform_surface.return_value = None

        with pytest.raises(FileNotFoundError, match="Unable to perform transformation"):
            mock_ops.surface_ops.transform_surface_to_volume(
                transformer_type="metric", **basic_params._asdict()
            )

    def test_raises_on_missing_target_atlas(
        self, mock_ops: NeuromapsGraph, basic_params: BasicParams, tmp_path: Path
    ) -> None:
        """FileNotFoundError raised when target surface atlas cannot be found."""
        mock_ops.surface_ops.transform_surface.return_value = tmp_path / "out.shape.gii"
        mock_ops.surface_ops.cache.get_surface_atlas.return_value = None

        with pytest.raises(FileNotFoundError, match="Unable to find target surface"):
            mock_ops.surface_ops.transform_surface_to_volume(
                transformer_type="metric", **basic_params._asdict()
            )
