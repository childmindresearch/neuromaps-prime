"""Tests associated with surface-to-surface transformation."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime import models
from neuromaps_prime.graph import NeuromapsGraph


class TestSurfaceToSurfaceTransformer:
    """Unit tests for surface to surface transformer."""

    @pytest.fixture
    def mock_transformer(self, graph: NeuromapsGraph) -> NeuromapsGraph:
        """Create a mock transformer instance with necessary methods."""
        graph._surface_to_surface = MagicMock()
        graph.find_highest_density = MagicMock(return_value="32k")
        graph.fetch_surface_atlas = MagicMock()
        return graph

    @pytest.fixture
    def mock_surface_atlas(self, tmp_path: Path) -> MagicMock:
        """Create mock surface atlas object."""
        current_sphere = tmp_path / "atlas.surf.gii"
        current_sphere.touch()
        atlas = MagicMock()
        atlas.fetch = MagicMock(return_value=current_sphere)
        return atlas

    @pytest.fixture
    def basic_params(self, tmp_path: Path) -> dict[str, Any]:
        """Basic parameters for testing."""
        input_file = tmp_path / "in.func.gii"
        input_file.touch()
        return {
            "input_file": input_file,
            "source_space": "Yerkes19",
            "target_space": "CIVETNMT",
            "hemisphere": "left",
            "output_file_path": f"{tmp_path}/out.func.gii",
            "source_density": "32k",
            "target_density": "41k",
        }

    @patch("neuromaps_prime.transforms.utils.estimate_surface_density")
    @patch("niwrap.workbench.metric_resample_area_surfs")
    @pytest.mark.parametrize("transformer_type", ["label", "metric"])
    def test_metric_transformation_success(
        self,
        mock_area_params: MagicMock,
        mock_estimate_density: MagicMock,
        mock_transformer: MagicMock,
        mock_surface_atlas: MagicMock,
        basic_params: dict[str, Any],
        transformer_type: str,
    ):
        """Test successful metric transformation."""
        # Setup
        mock_estimate_density.return_value = "32k"
        mock_transformer._surface_to_surface.return_value = mock_surface_atlas
        mock_transformer.fetch_surface_atlas.return_value = mock_surface_atlas
        mock_area_params.return_value = {
            "current_area": Path("/mock/current.surf.gii"),
            "new_area": Path("/mock/new.surf.gii"),
        }

        resample_fn = f"niwrap.workbench.{transformer_type}_resample"
        with patch(resample_fn) as mock_resample:
            mock_output = MagicMock()
            mock_output.label_out = Path(basic_params["output_file_path"])
            mock_output.label_out.touch()
            mock_resample.return_value = mock_output
            mock_transformer.surface_to_surface_transformer(
                transformer_type=transformer_type, **basic_params
                            )
        mock_transformer._surface_to_surface.assert_called_once()
        assert mock_transformer.fetch_surface_atlas.call_count == 3
        mock_resample.assert_called_once()

    def test_invalid_transformer_type(
        self, graph: NeuromapsGraph, basic_params: dict[str, Any],
    ) -> None:
        """Test error raised if invalid type."""
        with pytest.raises(ValueError, match="Invalid transformer_type"):
            graph.surface_to_surface_transformer(
                transformer_type="invalid", **basic_params
            )

    @patch("neuromaps_prime.transforms.utils.estimate_surface_density")
    def test_no_transform(
        self,
        graph: NeuromapsGraph,
        basic_params: dict[str, Any],
    ) -> None:
        """Test None returned if transform not found."""

    def test_fetch_surface_atlas_errors(


class TestSurfaceToSurfaceTransformPrivate:
    """Unit tests for private utility methods for surface to surface transformer."""

    @pytest.fixture
    def mock_graph(self, graph: NeuromapsGraph) -> NeuromapsGraph:
        """Mock graph collaborators accessed by SurfaceTransformOps.

        Replaces surface_ops.cache and surface_ops.utils with MagicMocks so
        that internal calls on those objects are interceptable without hitting
        Pydantic's __setattr__ validation.
        """
        """Mock graph for faking calls."""
        graph.add_transform = MagicMock()
        graph.validate = MagicMock()
        graph.fetch_surface_atlas = MagicMock()
        graph.surface_to_surface_key = "surface"
        return graph

    def test_same_source_target(self, mock_graph: NeuromapsGraph) -> None:
        """Test error raised if source is same as target."""
        with pytest.raises(ValueError, match="Source and target"):
            mock_graph._surface_to_surface(
                source="A",
                target="A",
                density="1k",
                hemisphere="left",
                output_file_path="same_target",
            )

    def test_no_valid_path(self, mock_graph: NeuromapsGraph) -> None:
        """Test error raised if no valid paths found."""
        mock_graph.find_path = MagicMock(return_value=["only_source"])
        with pytest.raises(ValueError, match="No valid path"):
            mock_graph._surface_to_surface(
                source="NMT2Sym",
                target="fsLR",
                density="32k",
                hemisphere="left",
                output_file_path="no_path",
            )

    def test_single_hop(self, mock_graph: NeuromapsGraph) -> None:
        """Test single-hop surface transformation."""
        mock_result = MagicMock(spec=models.SurfaceTransform)
        mock_graph.find_path = MagicMock(return_value=["Yerkes19", "fsLR"])
        mock_graph.fetch_surface_to_surface_transform = MagicMock(
         out = mock_graph._surface_to_surface(
            source="Yerkes19",
            target="fsLR",
            density="32k",
            hemisphere="right",
            output_file_path="single_hop",
            add_edge=False,
        )
        mock_graph.fetch_surface_to_surface_transform.assert_called_once()
        assert out is mock_result

    def test_multi_hop(self, mock_graph: NeuromapsGraph) -> None:
        """Test multi-hop path surface transformation."""
        mock_result = MagicMock(spec=models.SurfaceTransform)
        mock_graph.find_path = MagicMock(return_value=["CIVETNMT", "Yerkes19", "fsLR"])
        mock_graph._compose_multihop_surface_transform = MagicMock(
            return_value=mock_result
        out = mock_graph._surface_to_surface(
            source="CIVETNMT",
            target="fsLR",
            density="32k",
            hemisphere="right",
            output_file_path="multi_hop",
        )
        mock_graph._compose_multihop_surface_transform.assert_called_once()
        assert out is mock_result

    def test_compose_multihop_xfm(
        self, mock_graph: NeuromapsGraph, tmp_path: Path
    ) -> None:
        """Test multi-hop surface transform composition."""
        first_xfm = MagicMock(spec=models.SurfaceTransform)
        hop2_xfm = MagicMock(spec=models.SurfaceTransform)
        hop3_xfm = MagicMock(spec=models.SurfaceTransform)
        mock_graph.fetch_surface_to_surface_transform = MagicMock(
            return_value=first_xfm
        )
        mock_graph.surface_ops._compose_next_hop = MagicMock(
            side_effect=[hop2_xfm, hop3_xfm]
        )
        output_fpath = str(tmp_path / "output.surf.gii")
        result = mock_graph._compose_multihop_surface_transform(
            paths=["A", "B", "C", "D"],
            source="A",
            density="32k",
            hemisphere="left",
            output_file_path=output_fpath,
            add_edge=True,
        )

        mock_graph.fetch_surface_to_surface_transform.assert_called_once()
        assert mock_graph._compose_next_hop.call_count == 2
        first_call = mock_graph._compose_next_hop.call_args_list[0][1]
        assert first_call["hop_idx"] == 2
        assert first_call["next_space"] == "C"
        assert first_call["current_transform"] == first_xfm

        second_call = mock_graph._compose_next_hop.call_args_list[1][1]
        assert second_call["hop_idx"] == 3
        assert second_call["next_space"] == "D"
        assert second_call["current_transform"] == hop2_xfm
        assert result == hop3_xfm

    def test_compose_multihop_no_initial_xfm(self, mock_graph: NeuromapsGraph) -> None:
        """Test error raised whe no initial transformation found."""
        mock_graph.fetch_surface_to_surface_transform = MagicMock(return_value=None)
        with pytest.raises(ValueError, match="No transform found"):
            mock_graph._compose_multihop_surface_transform(
                paths=["A", "B", "C"],
                source="A",
                hemisphere="left",
                output_file_path="output.surf.gii",
                add_edge=True,
            )
        assert mock_graph.fetch_surface_to_surface_transform.called

    def test_compose_next_hop(self, mock_graph: NeuromapsGraph, tmp_path: Path) -> None:
        """Test basic composition of next hop."""
        current_transform = MagicMock(spec=models.SurfaceTransform)
        hop_output = str(tmp_path / "hop_output.surf.gii")
        composed_path = tmp_path / "composed_surf.gii"
        composed_path.touch()
        mock_graph._get_hop_output_file = MagicMock(return_value=hop_output)
        mock_graph._two_hops = MagicMock(return_value=composed_path)

        result = mock_graph._compose_next_hop(
            paths=["A", "B", "C"],
            hop_idx=2,
            next_space="C",
            current_transform=current_transform,
            source="A",
            density="32k",
            hemisphere="right",
            output_file_path=str(tmp_path / "output.surf.gii"),
            add_edge=True,
        )
        mock_graph._get_hop_output_file.assert_called_once()
        mock_graph._two_hops.assert_called_once()
        mock_graph.add_transform.assert_called_once()
        assert isinstance(result, models.SurfaceTransform)
        assert result.source_space == "A"
        assert result.target_space == "C"
        assert result.file_path == composed_path
        assert result.weight == 2.0

    def test_two_hops(self, mock_graph: NeuromapsGraph, tmp_path: Path) -> None:
        """Test two hop functionality."""
        first_xfm = MagicMock(spec=models.SurfaceTransform)
        first_xfm.fetch.return_value = tmp_path / "sphere_in.surf.gii"
        first_xfm.fetch.return_value.touch()
        mid_atlas = MagicMock(spec=models.SurfaceAtlas)
        mid_atlas.fetch.return_value = tmp_path / "sphere_project.surf.gii"
        mid_atlas.fetch.return_value.touch()
        target_xfm = MagicMock(spec=models.SurfaceTransform)
        target_xfm.fetch.return_value = tmp_path / "sphere_unproject.surf.gii"
        target_xfm.fetch.return_value.touch()
        output_path = tmp_path / "output_surf.gii"
        output_path.touch()

        mock_graph.find_common_density = MagicMock(return_value="32k")
        mock_graph.fetch_surface_atlas = MagicMock(return_value=mid_atlas)
        mock_graph.fetch_surface_to_surface_transform = MagicMock(

       with patch("niwrap.workbench.surface_sphere_project_unproject") as mock_project:
            mock_project.return_value = SimpleNamespace(sphere_out=output_path)
            result = mock_graph.surface_ops._two_hops(
                source_space="A",
                mid_space="B",
                target_space="C",
                density="32k",
                hemisphere="left",
                output_file_path=str(output_path),
                first_transform=first_xfm,
            )

        assert result == output_path
        mock_graph.surface_ops.utils.find_common_density.assert_called_once()
        mock_graph.surface_ops.cache.get_surface_atlas.assert_called_once()
        mock_project.assert_called_once()

    def test_two_hops_no_first_transform(self, mock_graph: NeuromapsGraph) -> None:
        """Test error raised when no first transform."""
        mock_graph.surface_ops.cache.get_surface_transform = MagicMock(
            return_value=None
        )
        
        with pytest.raises(ValueError, match="No surface transform found from"):
            result = mock_graph._two_hops(
                source_space="A",
                mid_space="B",
                target_space="C",
                density="32k",
                hemisphere="left",
                output_file_path="output_surf.gii",
                first_transform=None,
            )

    def test_two_hops_no_mid_atlas(
        self, mock_graph: NeuromapsGraph, tmp_path: Path
    ) -> None:
        """Test error raised when no mid atlas."""
        first_transform = MagicMock(spec=models.SurfaceTransform)
        first_transform.fetch.return_value = tmp_path / "sphere_in.surf.gii"
        mock_graph.surface_ops.utils.find_common_density = MagicMock(return_value="41k")
        mock_graph.surface_ops.cache.get_surface_atlas = MagicMock(return_value=None)
        with pytest.raises(ValueError, match="No sphere atlas found for"):
            mock_graph.surface_ops._two_hops(
                source_space="A",
                mid_space="B",
                target_space="C",
                density="32k",
                hemisphere="left",
                output_file_path="output.surf.gii",
                first_transform=first_transform,
            )

    def test_two_hops_no_target_xfm(
        self, mock_graph: NeuromapsGraph, tmp_path: Path
    ) -> None:
        """Test error raised when no target transformation."""
        first_transform = MagicMock(spec=models.SurfaceTransform)
        first_transform.fetch.return_value = tmp_path / "sphere_in.surf.gii"
        mid_atlas = MagicMock(spec=models.SurfaceAtlas)
        mid_atlas.fetch.return_value = tmp_path / "sphere_project.surf.gii"

        mock_graph.surface_ops.utils.find_common_density = MagicMock(return_value="41k")
        mock_graph.surface_ops.cache.get_surface_atlas = MagicMock(
            return_value=mid_atlas
        )
        mock_graph.surface_ops.cache.get_surface_transform = MagicMock(
            return_value=None
        )
        with pytest.raises(ValueError, match="No surface transform found from"):
            mock_graph.surface_ops._two_hops(
                source_space="A",
                mid_space="B",
                target_space="C",
                density="32k",
                hemisphere="left",
                output_file_path=str(tmp_path / "output.surf.gii"),
                first_transform=first_transform,
            )

    def test_hop_output_path(self, graph: NeuromapsGraph, tmp_path: Path) -> None:
        """Test generation of hop output file path."""
        output = graph.surface_ops._hop_output_path(
            output_file_path=str(tmp_path / "output_file.surf.gii"),
            source="A",
            next_target="B",
            density="32k",
            hemisphere="left",
        )
        assert output == str(tmp_path / "src-A_to-B_den-32k_hemi-L_sphere.surf.gii")


class TestTransformSurface:
    """Unit tests for SurfaceTransformOps.transform_surface."""

    @pytest.fixture
    def mock_ops(self, graph: NeuromapsGraph) -> NeuromapsGraph:
        """Mock surface operations."""
        graph.surface_ops.cache = MagicMock()
        graph.surface_ops.utils = MagicMock()
        graph.surface_ops._resolve_sphere_transform = MagicMock()
        return graph

    @pytest.fixture
    def basic_params(self, tmp_path: Path) -> dict:
        """Basic parameters."""
        input_file = tmp_path / "in.func.gii"
        input_file.touch()
        return {
            "input_file": input_file,
            "source_space": "fsLR",
            "target_space": "MNI152",
            "hemisphere": "left",
            "output_file_path": str(tmp_path / "out.func.gii"),
            "source_density": "32k",
            "target_density": "2mm",
        }

    @pytest.mark.parametrize("transformer_type", ["metric", "label"])
    def test_happy_path(
        self,
        mock_ops: NeuromapsGraph,
        basic_params: dict,
        transformer_type: Literal["metric", "label"],
        tmp_path: Path,
    ) -> None:
        """Metric and label paths call correct resample fn and return output path."""
        mock_ops.surface_ops._resolve_sphere_transform.return_value = MagicMock(
            fetch=MagicMock(return_value=tmp_path / "sphere.surf.gii")
        )
        mock_ops.surface_ops.cache.require_surface_atlas.return_value = MagicMock(
            fetch=MagicMock(return_value=tmp_path / "area.surf.gii")
        )
        expected_out = tmp_path / "out.func.gii"
        resample_result = MagicMock(metric_out=expected_out, label_out=expected_out)
        resample_fn = (
            "neuromaps_prime.graph.transforms.surface.metric_resample"
            if transformer_type == "metric"
            else "neuromaps_prime.graph.transforms.surface.label_resample"
        )
        with patch(resample_fn, return_value=resample_result):
            result = mock_ops.surface_ops.transform_surface(
                transformer_type=transformer_type, **basic_params
            )
        assert result == expected_out

    def test_source_density_estimated_when_none(
        self, mock_ops: NeuromapsGraph, basic_params: dict, tmp_path: Path
    ) -> None:
        """source_density=None triggers estimate_surface_density."""
        basic_params["source_density"] = None
        mock_ops.surface_ops._resolve_sphere_transform.return_value = MagicMock()
        mock_ops.surface_ops.cache.require_surface_atlas.return_value = MagicMock(
            fetch=MagicMock(return_value=tmp_path / "area.surf.gii")
        )
        with (
            patch(
                "neuromaps_prime.graph.transforms.surface.estimate_surface_density",
                return_value="32k",
            ) as mock_est,
            patch("neuromaps_prime.graph.transforms.surface.metric_resample"),
        ):
            mock_ops.surface_ops.transform_surface(
                transformer_type="metric", **basic_params
            )
        mock_est.assert_called_once_with(basic_params["input_file"])

    def test_target_density_uses_highest_when_none(
        self, mock_ops: NeuromapsGraph, basic_params: dict, tmp_path: Path
    ) -> None:
        """target_density=None calls find_highest_density."""
        basic_params["target_density"] = None
        mock_ops.surface_ops._resolve_sphere_transform.return_value = MagicMock()
        mock_ops.surface_ops.utils.find_highest_density.return_value = "164k"
        mock_ops.surface_ops.cache.require_surface_atlas.return_value = MagicMock(
            fetch=MagicMock(return_value=tmp_path / "area.surf.gii")
        )
        with patch("neuromaps_prime.graph.transforms.surface.metric_resample"):
            mock_ops.surface_ops.transform_surface(
                transformer_type="metric", **basic_params
            )
        mock_ops.surface_ops.utils.find_highest_density.assert_called_once_with(
            space="MNI152"
        )

    def test_returns_none_when_no_sphere_transform(
        self, mock_ops: NeuromapsGraph, basic_params: dict
    ) -> None:
        """Returns None when _resolve_sphere_transform returns None."""
        mock_ops.surface_ops._resolve_sphere_transform.return_value = None
        result = mock_ops.surface_ops.transform_surface(
            transformer_type="metric", **basic_params
        )
        assert result is None
        mock_ops.surface_ops.cache.require_surface_atlas.assert_not_called()