"""Tests associated with volume-to-surface transformation."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime.graph import NeuromapsGraph


class TestVolumeToSurfaceTransformer:
    """Unit tests for volume to surface transformer."""

    @pytest.fixture
    def mock_transformer(self, graph: NeuromapsGraph) -> NeuromapsGraph:
        """Create a mock transformer instance with necessary methods."""
        graph.validate = MagicMock()
        graph.find_highest_density = MagicMock(return_value="32k")
        graph.fetch_surface_atlas = MagicMock()
        graph.surface_to_surface_transformer = MagicMock()
        return graph

    @pytest.fixture
    def basic_params(self, tmp_path: Path) -> dict[str, Any]:
        """Basic parameters for volume-to-surface transformation testing."""
        input_file = tmp_path / "input.nii.gz"
        input_file.touch()
        return {
            "transformer_type": "metric",
            "input_file": input_file,
            "source_space": "Yerkes19",
            "target_space": "CIVETNMT",
            "hemisphere": "left",
            "output_file_path": str(tmp_path / "output.func.gii"),
            "source_density": "32k",
            "target_density": "41k",
        }

    def make_atlas_side_effect(self, tmp_path: Path) -> Any:
        """Create a side effect returning distinct atlases per resource type."""

        def fetch_surface_atlas_side_effect(
            space: str, density: str, hemisphere: str, resource_type: str
        ) -> MagicMock:
            atlas = MagicMock()
            surf_file = tmp_path / f"{resource_type}.surf.gii"
            surf_file.touch()
            atlas.fetch = MagicMock(return_value=surf_file)
            return atlas

        return fetch_surface_atlas_side_effect

    @patch(
        "neuromaps_prime.graph.workbench.volume_to_surface_mapping_ribbon_constrained"
    )
    @patch("neuromaps_prime.graph.surface_project")
    @pytest.mark.parametrize("transformer_type", ["metric", "label"])
    def test_volume_to_surface_success(
        self,
        mock_surface_project: MagicMock,
        mock_ribbon: MagicMock,
        mock_transformer: NeuromapsGraph,
        basic_params: dict[str, Any],
        transformer_type: str,
        tmp_path: Path,
    ) -> None:
        """Test successful volume-to-surface transformation."""
        basic_params["transformer_type"] = transformer_type
        projected_file = tmp_path / "projected.func.gii"
        projected_file.touch()
        expected_output = Path(basic_params["output_file_path"])

        mock_transformer.fetch_surface_atlas.side_effect = self.make_atlas_side_effect(
            tmp_path
        )
        mock_ribbon.return_value = MagicMock()
        mock_surface_project.return_value = projected_file
        mock_transformer.surface_to_surface_transformer.return_value = expected_output

        result = mock_transformer.volume_to_surface_transformer(**basic_params)

        mock_transformer.validate.assert_called_once_with(
            basic_params["source_space"], basic_params["target_space"]
        )
        assert mock_transformer.fetch_surface_atlas.call_count == 3
        mock_ribbon.assert_called_once_with(
            inner_surf=tmp_path / "white.surf.gii",
            outer_surf=tmp_path / "pial.surf.gii",
        )
        mock_surface_project.assert_called_once()
        mock_transformer.surface_to_surface_transformer.assert_called_once()
        assert result == expected_output

    @patch(
        "neuromaps_prime.graph.workbench.volume_to_surface_mapping_ribbon_constrained"
    )
    @patch("neuromaps_prime.graph.surface_project")
    @pytest.mark.parametrize(
        "transformer_type,expected_ext", [("metric", "func"), ("label", "label")]
    )
    def test_projected_filename_extension(
        self,
        mock_surface_project: MagicMock,
        mock_ribbon: MagicMock,
        mock_transformer: NeuromapsGraph,
        basic_params: dict[str, Any],
        tmp_path: Path,
        transformer_type: str,
        expected_ext: str,
    ) -> None:
        """Test projected file path uses correct extension for metric vs label."""
        basic_params["transformer_type"] = transformer_type
        projected_file = tmp_path / f"projected.{expected_ext}.gii"
        projected_file.touch()

        mock_transformer.fetch_surface_atlas.side_effect = self.make_atlas_side_effect(
            tmp_path
        )
        mock_ribbon.return_value = MagicMock()
        mock_surface_project.return_value = projected_file
        mock_transformer.surface_to_surface_transformer.return_value = projected_file

        mock_transformer.volume_to_surface_transformer(**basic_params)

        _, kwargs = mock_surface_project.call_args
        assert f".{expected_ext}.gii" in kwargs["out_fpath"]

    @patch(
        "neuromaps_prime.graph.workbench.volume_to_surface_mapping_ribbon_constrained"
    )
    @patch("neuromaps_prime.graph.surface_project")
    def test_surface_to_surface_called_with_correct_args(
        self,
        mock_surface_project: MagicMock,
        mock_ribbon: MagicMock,
        mock_transformer: NeuromapsGraph,
        basic_params: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Testtransformer is called with correct forwarded args."""
        projected_file = tmp_path / "projected.func.gii"
        projected_file.touch()

        mock_transformer.fetch_surface_atlas.side_effect = self.make_atlas_side_effect(
            tmp_path
        )
        mock_ribbon.return_value = MagicMock()
        mock_surface_project.return_value = projected_file
        mock_transformer.surface_to_surface_transformer.return_value = Path(
            basic_params["output_file_path"]
        )

        mock_transformer.volume_to_surface_transformer(**basic_params)

        mock_transformer.surface_to_surface_transformer.assert_called_once_with(
            transformer_type=basic_params["transformer_type"],
            input_file=projected_file,
            source_space=basic_params["source_space"],
            target_space=basic_params["target_space"],
            hemisphere=basic_params["hemisphere"],
            output_file_path=basic_params["output_file_path"],
            source_density=basic_params["source_density"],
            target_density=basic_params["target_density"],
            area_resource="midthickness",
            add_edge=True,
        )

    def test_no_source_surface_atlas(
        self, mock_transformer: NeuromapsGraph, basic_params: dict[str, Any]
    ) -> None:
        """Test ValueError raised when source midthickness surface atlas not found."""
        mock_transformer.fetch_surface_atlas.return_value = None
        with pytest.raises(ValueError, match="No midthickness surface found for"):
            mock_transformer.volume_to_surface_transformer(**basic_params)
        mock_transformer.fetch_surface_atlas.assert_called_once()

    @pytest.mark.parametrize(
        "failing_call, missing_surface",
        [
            (2, "white"),
            (3, "pial"),
        ],
    )
    def test_no_ribbon_surface(
        self,
        mock_transformer: NeuromapsGraph,
        basic_params: dict[str, Any],
        failing_call: int,
        missing_surface: str,
        tmp_path: Path,
    ) -> None:
        """Test ValueError raised when white or pial ribbon surface not found."""
        base_side_effect = self.make_atlas_side_effect(tmp_path)
        call_count = 0

        def side_effect_with_failure(
            space: str, density: str, hemisphere: str, resource_type: str
        ) -> MagicMock | None:
            nonlocal call_count
            call_count += 1
            return (
                None
                if call_count == failing_call
                else base_side_effect(
                    space=space,
                    density=density,
                    hemisphere=hemisphere,
                    resource_type=resource_type,
                )
            )

        mock_transformer.fetch_surface_atlas.side_effect = side_effect_with_failure
        with pytest.raises(ValueError, match=f"No {missing_surface} surface found for"):
            mock_transformer.volume_to_surface_transformer(**basic_params)
        assert mock_transformer.fetch_surface_atlas.call_count == failing_call
