"""Tests associated with volume-to-surface transformation."""

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
    source_space: str
    target_space: str
    hemisphere: Literal["left", "right"]
    output_file_path: str
    source_density: str
    target_density: str
    transformer_type: Literal["label", "metric"]


class TestVolumeToSurfaceTransformer:
    """Unit tests for volume to surface transformer."""

    @pytest.fixture
    def mock_transformer(self, graph: NeuromapsGraph) -> NeuromapsGraph:
        """Create a mock transformer with volume_ops internals replaced.

        Replaces volume_ops.utils, volume_ops.cache, and volume_ops.surface_ops
        with MagicMocks so internal calls are interceptable without hitting
        Pydantic's __setattr__ validation.
        """
        graph.volume_ops.utils = MagicMock(
            find_highest_density=MagicMock(return_value="32k")
        )
        graph.volume_ops.cache = MagicMock()
        graph.volume_ops.surface_ops = MagicMock()
        return graph

    @pytest.fixture
    def basic_params(self, tmp_path: Path) -> BasicParams:
        """Basic parameters for volume-to-surface transformation testing."""
        input_file = tmp_path / "input.nii.gz"
        input_file.touch()
        return BasicParams(
            transformer_type="metric",
            input_file=input_file,
            source_space="Yerkes19",
            target_space="CIVETNMT",
            hemisphere="left",
            output_file_path=str(tmp_path / "output.func.gii"),
            source_density="32k",
            target_density="41k",
        )

    def make_atlas_side_effect(
        self,
        tmp_path: Path,
        **entities: dict[str, str],  # noqa: ARG002
    ) -> Callable[[str, Literal["left", "right"], str, str], MagicMock]:
        """Create a side effect returning distinct atlases per resource type."""

        def fetch_surface_atlas_side_effect(
            density: str,
            hemisphere: Literal["left", "right"],
            space: str,
            resource_type: str,
        ) -> MagicMock:
            fname = f"hemi-{hemisphere}_den-{density}_space-{space}_{resource_type}"
            surf_file = tmp_path / f"{fname}.surf.gii"
            surf_file.touch()
            return MagicMock(fetch=MagicMock(return_value=surf_file))

        return fetch_surface_atlas_side_effect

    @pytest.mark.parametrize("transformer_type", ["metric", "label"])
    def test_volume_to_surface_success(
        self,
        mock_transformer: NeuromapsGraph,
        basic_params: BasicParams,
        transformer_type: Literal["label", "metric"],
        tmp_path: Path,
    ) -> None:
        """Test successful volume-to-surface transformation."""
        basic_params = basic_params._replace(transformer_type=transformer_type)
        expected_output = Path(basic_params.output_file_path)
        projected_file = tmp_path / "projected.func.gii"
        projected_file.touch()

        mock_transformer.volume_ops.cache.require_surface_atlas.side_effect = (
            self.make_atlas_side_effect(tmp_path)
        )
        with (
            patch(
                "neuromaps_prime.graph.transforms.volume.workbench.volume_to_surface_mapping_ribbon_constrained",
                return_value=MagicMock(),
            ) as mock_ribbon,
            patch(
                "neuromaps_prime.graph.transforms.volume.surface_project",
                return_value=projected_file,
            ) as mock_surface_project,
        ):
            mock_transformer.volume_ops.surface_ops.transform_surface.return_value = (
                expected_output
            )
            result = mock_transformer.volume_to_surface_transformer(
                **basic_params._asdict()
            )

        mock_transformer.volume_ops.utils.validate_spaces.assert_called_once_with(
            basic_params.source_space, basic_params.target_space
        )
        assert mock_transformer.volume_ops.cache.require_surface_atlas.call_count == 3
        mock_ribbon.assert_called_once()
        mock_surface_project.assert_called_once()
        mock_transformer.volume_ops.surface_ops.transform_surface.assert_called_once()
        assert result == expected_output

    @pytest.mark.parametrize(
        ("transformer_type", "expected_ext"), [("metric", "func"), ("label", "label")]
    )
    def test_projected_filename_extension(
        self,
        mock_transformer: NeuromapsGraph,
        basic_params: BasicParams,
        tmp_path: Path,
        transformer_type: Literal["label", "metric"],
        expected_ext: str,
    ) -> None:
        """Test projected file path uses correct extension for metric vs label."""
        basic_params = basic_params._replace(transformer_type=transformer_type)
        projected_file = tmp_path / f"projected.{expected_ext}.gii"
        projected_file.touch()

        mock_transformer.volume_ops.cache.require_surface_atlas.side_effect = (
            self.make_atlas_side_effect(tmp_path)
        )
        mock_transformer.volume_ops.surface_ops.transform_surface.return_value = (
            projected_file
        )
        with (
            patch(
                "neuromaps_prime.graph.transforms.volume.workbench.volume_to_surface_mapping_ribbon_constrained",
                return_value=MagicMock(),
            ),
            patch(
                "neuromaps_prime.graph.transforms.volume.surface_project",
                return_value=projected_file,
            ) as mock_surface_project,
        ):
            mock_transformer.volume_to_surface_transformer(**basic_params._asdict())

        _, kwargs = mock_surface_project.call_args
        assert f".{expected_ext}.gii" in kwargs["out_fpath"]

    def test_surface_to_surface_called_with_correct_args(
        self,
        mock_transformer: NeuromapsGraph,
        basic_params: BasicParams,
        tmp_path: Path,
    ) -> None:
        """Test surface_ops.transform is called with correct forwarded args."""
        projected_file = tmp_path / "projected.func.gii"
        projected_file.touch()

        mock_transformer.volume_ops.cache.require_surface_atlas.side_effect = (
            self.make_atlas_side_effect(tmp_path)
        )
        mock_transformer.volume_ops.surface_ops.transform_surface.return_value = Path(
            basic_params.output_file_path
        )

        with (
            patch(
                "neuromaps_prime.graph.transforms.volume.workbench.volume_to_surface_mapping_ribbon_constrained",
                return_value=MagicMock(),
            ),
            patch(
                "neuromaps_prime.graph.transforms.volume.surface_project",
                return_value=projected_file,
            ),
        ):
            mock_transformer.volume_to_surface_transformer(**basic_params._asdict())
        mock_transformer.volume_ops.surface_ops.transform_surface.assert_called_once()

    def test_no_source_surface_atlas(
        self, mock_transformer: NeuromapsGraph, basic_params: BasicParams
    ) -> None:
        """Test ValueError raised when source midthickness surface atlas not found."""
        mock_transformer.volume_ops.cache.require_surface_atlas.side_effect = (
            ValueError(
                "No 'midthickness' surface atlas found for space "
                f"'{basic_params.source_space}' "
                f"(density='32k', hemisphere='{basic_params.hemisphere}')"
            )
        )
        with pytest.raises(
            ValueError, match="No 'midthickness' surface atlas found for space"
        ):
            mock_transformer.volume_to_surface_transformer(**basic_params._asdict())
        mock_transformer.volume_ops.cache.require_surface_atlas.assert_called_once()

    @pytest.mark.parametrize(
        ("failing_call", "missing_surface"),
        [(2, "white"), (3, "pial")],
    )
    def test_no_ribbon_surface(
        self,
        mock_transformer: NeuromapsGraph,
        basic_params: BasicParams,
        failing_call: int,
        missing_surface: str,
        tmp_path: Path,
    ) -> None:
        """Test ValueError raised when white or pial ribbon surface not found."""
        base_side_effect = self.make_atlas_side_effect(tmp_path)
        call_count = 0

        def side_effect_with_failure(
            density: str,
            hemisphere: Literal["left", "right"],
            space: str,
            resource_type: str,
        ) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == failing_call:
                raise ValueError(
                    f"No '{missing_surface}' surface atlas found for space "
                    f"'{basic_params.source_space}' "
                    f"(density='{basic_params.source_density}', "
                    f"hemisphere='{basic_params.hemisphere}')"
                )
            return base_side_effect(
                density=density,
                hemisphere=hemisphere,
                space=space,
                resource_type=resource_type,
            )

        mock_transformer.volume_ops.cache.require_surface_atlas.side_effect = (
            side_effect_with_failure
        )
        with (
            patch(
                "neuromaps_prime.graph.transforms.volume.workbench.volume_to_surface_mapping_ribbon_constrained"
            ),
            patch("neuromaps_prime.graph.transforms.volume.surface_project"),
            pytest.raises(
                ValueError, match=f"No '{missing_surface}' surface atlas found"
            ),
        ):
            mock_transformer.volume_to_surface_transformer(**basic_params._asdict())

    def test_invalid_transformer_type(
        self, graph: NeuromapsGraph, basic_params: BasicParams
    ) -> None:
        """Test error raised if invalid type."""
        basic_params = basic_params._replace(
            transformer_type="invalid"  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="Invalid transformer_type"):
            graph.volume_ops.transform_volume_to_surface(**basic_params._asdict())
