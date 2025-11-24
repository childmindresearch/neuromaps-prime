"""Tests for surface transformations."""

from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest
from niwrap import workbench

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.transforms import utils
from neuromaps_prime.transforms.surface import (
    label_resample,
    metric_resample,
    surface_sphere_project_unproject,
)


def touch_inputs(paths: dict[str, Any], skip: set[str]) -> None:
    """Touch every Path except the skipped keys."""
    for key, p in paths.items():
        if key in skip:
            continue
        if isinstance(p, Path):
            p.touch()
        elif isinstance(p, dict):  # area_surfs
            p["current-area"].touch()
            p["new-area"].touch()


def run_patched(
    func_path: str,
    side_effect: Callable[..., Any] | None = None,
    return_value: Any = None,
):
    """Context manager to patch a workbench function."""
    return patch(func_path, side_effect=side_effect, return_value=return_value)


class TestSurfaceSphereProjectUnproject:
    """Unit test of surface sphere project unproject function."""

    @pytest.fixture
    def mock_paths(self, tmp_path: Path) -> dict[str, Any]:
        """Mock file paths."""
        return {
            "sphere_in": tmp_path / "sphere_in.surf.gii",
            "sphere_project_to": tmp_path / "sphere_project_to.surf.gii",
            "sphere_unproject_from": tmp_path / "sphere_unproject_from.surf.gii",
            "sphere_out": str(tmp_path / "out.surf.gii"),
        }

    @pytest.mark.parametrize(
        "missing, msg",
        [
            ("sphere_in", "Input sphere"),
            ("sphere_project_to", "project to"),
            ("sphere_unproject_from", "unproject from"),
        ],
    )
    def test_missing_inputs(self, mock_paths: dict[str, Any], missing: str, msg: str):
        """Each missing input should raise FileNotFoundError with correct message."""
        touch_inputs(mock_paths, skip={missing, "sphere_out"})
        with pytest.raises(FileNotFoundError, match=msg):
            surface_sphere_project_unproject(**mock_paths)

    def test_success(self, mock_paths: dict[str, Any]):
        """Test successful call."""
        touch_inputs(mock_paths, skip={"sphere_out"})
        mock_result = MagicMock()
        mock_result.sphere_out = Path(mock_paths["sphere_out"])

        def produce(*args, **kwargs) -> MagicMock:
            Path(mock_paths["sphere_out"]).touch()
            return mock_result

        func_path = (
            "neuromaps_prime.transforms.surface."
            "workbench.surface_sphere_project_unproject"
        )
        with run_patched(func_path, side_effect=lambda **_: produce()) as mock_wb:
            result = surface_sphere_project_unproject(**mock_paths)
            mock_wb.assert_called_once_with(**mock_paths)
            assert result.sphere_out.exists()

    def test_missing_output(self, mock_paths: dict[str, Any]):
        """Test FileNotFoundError raised if output file is missing."""
        touch_inputs(mock_paths, skip={"sphere_out"})

        mock_result = MagicMock()
        mock_result.sphere_out = Path(mock_paths["sphere_out"])

        func_path = (
            "neuromaps_prime.transforms.surface."
            "workbench.surface_sphere_project_unproject"
        )
        with run_patched(func_path, return_value=mock_result):
            with pytest.raises(FileNotFoundError, match="Sphere out not found"):
                surface_sphere_project_unproject(**mock_paths)


class TestMetricResample:
    """Unit test of metric resample."""

    @pytest.fixture
    def mock_paths(self, tmp_path: Path) -> dict[str, Any]:
        """Mock file paths."""
        return {
            "input_file_path": tmp_path / "metric.shape.gii",
            "current_sphere": tmp_path / "current_sphere.surf.gii",
            "new_sphere": tmp_path / "new_sphere.surf.gii",
            "output_file_path": str(tmp_path / "out.shape.gii"),
            "area_surfs": workbench.metric_resample_area_surfs(
                current_area=tmp_path / "current_midthickness.surf.gii",
                new_area=tmp_path / "new_midthickness.surf.gii",
            ),
        }

    @pytest.mark.parametrize(
        "missing, msg",
        [
            ("input_file_path", "Input file"),
            ("current_sphere", "Current sphere"),
            ("new_sphere", "New sphere"),
        ],
    )
    def test_missing_inputs(self, mock_paths: dict[str, Any], missing: str, msg: str):
        """Each missing input should raise FileNotFoundError with correct message."""
        touch_inputs(mock_paths, skip={missing, "output_file_path", "area_surfs"})
        with pytest.raises(FileNotFoundError, match=msg):
            metric_resample(**mock_paths, method="ADAP_BARY_AREA")

    def test_invalid_resample_method(self, mock_paths: dict[str, Any]):
        """Test NotImplementedError raised if invalid resample method passed."""
        touch_inputs(mock_paths, skip={"output_file_path", "area_surfs"})
        with pytest.raises(NotImplementedError, match="not implemented"):
            metric_resample(**mock_paths, method="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize("method", ["ADAP_BARY_AREA", "BARYCENTRIC"])
    def test_success(self, mock_paths: dict[str, Any], method: str):
        """Test successful call."""
        touch_inputs(mock_paths, skip={"output_file_path"})

        mock_result = MagicMock()
        mock_result.metric_out = Path(mock_paths["output_file_path"])

        def produce(*args, **kwargs) -> MagicMock:
            Path(mock_paths["output_file_path"]).touch()
            return mock_result

        func_path = "neuromaps_prime.transforms.surface.workbench.metric_resample"
        with run_patched(func_path, side_effect=lambda **_: produce()) as mock_wb:
            metric_resample(**mock_paths, method=method)  # type: ignore[arg-type]
            mock_wb.assert_called_once()
            assert Path(mock_paths["output_file_path"]).exists()

    def test_missing_output(self, mock_paths: dict[str, Any]):
        """Test FileNotFoundError raised if output file is missing."""
        touch_inputs(mock_paths, skip={"output_file_path"})

        mock_result = MagicMock()
        mock_result.metric_out = Path(mock_paths["output_file_path"])

        func_path = "neuromaps_prime.transforms.surface.workbench.metric_resample"
        with run_patched(func_path, return_value=mock_result):
            with pytest.raises(FileNotFoundError, match="Metric out not found"):
                metric_resample(**mock_paths, method="ADAP_BARY_AREA")


class TestLabelResample:
    """Unit test of label resample."""

    @pytest.fixture
    def mock_paths(self, tmp_path: Path) -> dict[str, Any]:
        """Mock file paths."""
        return {
            "input_file_path": tmp_path / "metric.label.gii",
            "current_sphere": tmp_path / "current_sphere.surf.gii",
            "new_sphere": tmp_path / "new_sphere.surf.gii",
            "output_file_path": str(tmp_path / "out.label.gii"),
            "area_surfs": workbench.label_resample_area_surfs(
                current_area=tmp_path / "current_midthickness.surf.gii",
                new_area=tmp_path / "new_midthickness.surf.gii",
            ),
        }

    @pytest.mark.parametrize(
        "missing, msg",
        [
            ("input_file_path", "Input file"),
            ("current_sphere", "Current sphere"),
            ("new_sphere", "New sphere"),
        ],
    )
    def test_missing_inputs(self, mock_paths: dict[str, Any], missing: str, msg: str):
        """Each missing input should raise FileNotFoundError with correct message."""
        touch_inputs(mock_paths, skip={missing, "output_file_path", "area_surfs"})
        with pytest.raises(FileNotFoundError, match=msg):
            label_resample(**mock_paths, method="ADAP_BARY_AREA")

    def test_invalid_resample_method(self, mock_paths: dict[str, Any]):
        """Test NotImplementedError raised if invalid resample method passed."""
        touch_inputs(mock_paths, skip={"output_file_path", "area_surfs"})
        with pytest.raises(NotImplementedError, match="not implemented"):
            label_resample(**mock_paths, method="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize("method", ["ADAP_BARY_AREA", "BARYCENTRIC"])
    def test_success(self, mock_paths: dict[str, Any], method: str):
        """Test successful call."""
        touch_inputs(mock_paths, skip={"output_file_path"})

        mock_result = MagicMock()
        mock_result.label_out = Path(mock_paths["output_file_path"])

        def produce(*args, **kwargs) -> MagicMock:
            Path(mock_paths["output_file_path"]).touch()
            return mock_result

        func_path = "neuromaps_prime.transforms.surface.workbench.label_resample"
        with run_patched(func_path, side_effect=lambda **_: produce()) as mock_wb:
            label_resample(**mock_paths, method=method)  # type: ignore[arg-type]
            mock_wb.assert_called_once()
            assert Path(mock_paths["output_file_path"]).exists()

    def test_missing_output(self, mock_paths: dict[str, Any]):
        """Test FileNotFoundError raised if output file is missing."""
        touch_inputs(mock_paths, skip={"output_file_path"})

        mock_result = MagicMock()
        mock_result.label_out = Path(mock_paths["output_file_path"])

        func_path = "neuromaps_prime.transforms.surface.workbench.label_resample"
        with run_patched(func_path, return_value=mock_result):
            with pytest.raises(FileNotFoundError, match="Label out not found"):
                label_resample(**mock_paths, method="ADAP_BARY_AREA")


@pytest.mark.usefixtures("require_workbench")
class TestSurfaceTransformIntegration:
    """Integration tests calling Workbench and using real data."""

    def test_surface_sphere_project_unproject(
        self, tmp_path: Path, graph: NeuromapsGraph
    ) -> None:
        """Integration test of surface_sphere_project_unproject."""
        sphere_in = graph.fetch_surface_to_surface_transform(
            source="S1200",
            target="Yerkes19",
            density="32k",
            hemisphere="left",
            resource_type="sphere",
        )
        sphere_in = sphere_in
        sphere_project_to = graph.fetch_surface_atlas(
            space="Yerkes19",
            density="32k",
            hemisphere="left",
            resource_type="sphere",
        )
        sphere_unproject_from = graph.fetch_surface_to_surface_transform(
            source="Yerkes19",
            target="D99",
            density="32k",
            hemisphere="left",
            resource_type="sphere",
        )
        sphere_out = tmp_path / "out_sphere.surf.gii"
        assert sphere_in is not None
        assert sphere_project_to is not None
        assert sphere_unproject_from is not None
        result = surface_sphere_project_unproject(
            sphere_in=sphere_in.fetch(),
            sphere_project_to=sphere_project_to.fetch(),
            sphere_unproject_from=sphere_unproject_from.fetch(),
            sphere_out=str(sphere_out),
        )
        assert utils.get_vertex_count(sphere_in) == utils.get_vertex_count(  # type: ignore[arg-type]
            result.sphere_out
        )

    @pytest.mark.skip(reason="No metric data to test resampling")
    def test_metric_resample(self, tmp_path: Path, graph: NeuromapsGraph) -> None:
        """Integration test of metric_resample."""
        pass

    @pytest.mark.skip(reason="No label data to test resampling")
    def test_label_resample(self, tmp_path: Path, graph: NeuromapsGraph) -> None:
        """Integration test of label_resample."""
        pass
