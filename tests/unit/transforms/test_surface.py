"""Unit tests for surface transformations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, _patch, patch

import pytest
from niwrap import workbench

from neuromaps_prime.transforms.surface import (
    label_resample,
    metric_resample,
    surface_sphere_project_unproject,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any


def _touch_inputs(paths: dict[str, Any], skip: Sequence[str]) -> None:
    """Touch every Path except the skipped keys."""
    for key, p in paths.items():
        if key in skip:
            continue
        if isinstance(p, Path):
            p.touch()
        elif isinstance(p, dict):  # area_surfs
            p["current-area"].touch()
            p["new-area"].touch()


def _run_patched(
    func_path: str,
    side_effect: Callable[..., Any] | None = None,
    return_value: Any = None,  # noqa: ANN401 - patch accepts arbitrary return values
) -> _patch[MagicMock]:
    """Context manager to patch a workbench function."""
    return patch(func_path, side_effect=side_effect, return_value=return_value)


class TestSurfaceSphereProjectUnproject:
    """Test for project-unproject function."""

    @pytest.fixture
    def mock_paths(self, tmp_path: Path) -> dict[str, str | Path]:
        """Mock file paths."""
        return {
            "sphere_in": tmp_path / "sphere_in.surf.gii",
            "sphere_project_to": tmp_path / "sphere_project_to.surf.gii",
            "sphere_unproject_from": tmp_path / "sphere_unproject_from.surf.gii",
            "sphere_out": str(tmp_path / "out.surf.gii"),
        }

    @pytest.mark.parametrize(
        ("missing", "msg"),
        [
            ("sphere_in", "Input sphere"),
            ("sphere_project_to", "project to"),
            ("sphere_unproject_from", "unproject from"),
        ],
    )
    def test_missing_inputs(
        self, mock_paths: dict[str, Any], missing: str, msg: str
    ) -> None:
        """Each missing input should raise FileNotFoundError with correct message."""
        _touch_inputs(mock_paths, skip=(missing, "sphere_out"))
        with pytest.raises(FileNotFoundError, match=msg):
            surface_sphere_project_unproject(**mock_paths)

    def test_success(self, mock_paths: dict[str, Any]) -> None:
        """Test successful call."""
        _touch_inputs(mock_paths, skip=("sphere_out",))
        mock_result = MagicMock(sphere_out=Path(mock_paths["sphere_out"]))

        def _produce() -> MagicMock:
            Path(mock_paths["sphere_out"]).touch()
            return mock_result

        func_path = (
            "neuromaps_prime.transforms.surface."
            "workbench.surface_sphere_project_unproject"
        )
        with _run_patched(func_path, side_effect=lambda **_: _produce()) as mock_wb:
            result = surface_sphere_project_unproject(**mock_paths)
            mock_wb.assert_called_once_with(**mock_paths)
            assert result.sphere_out.exists()

    def test_missing_output(self, mock_paths: dict[str, Any]) -> None:
        """Test FileNotFoundError raised if output file is missing."""
        _touch_inputs(mock_paths, skip=("sphere_out",))

        mock_result = MagicMock(sphere_out=Path(mock_paths["sphere_out"]))

        func_path = (
            "neuromaps_prime.transforms.surface."
            "workbench.surface_sphere_project_unproject"
        )
        with (
            _run_patched(func_path, return_value=mock_result),
            pytest.raises(FileNotFoundError, match="Sphere out not found"),
        ):
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
        ("missing", "msg"),
        [
            ("input_file_path", "Input file"),
            ("current_sphere", "Current sphere"),
            ("new_sphere", "New sphere"),
        ],
    )
    def test_missing_inputs(
        self, mock_paths: dict[str, Any], missing: str, msg: str
    ) -> None:
        """Each missing input should raise FileNotFoundError with correct message."""
        _touch_inputs(mock_paths, skip=(missing, "output_file_path", "area_surfs"))
        with pytest.raises(FileNotFoundError, match=msg):
            metric_resample(**mock_paths, method="ADAP_BARY_AREA")

    def test_invalid_resample_method(self, mock_paths: dict[str, Any]) -> None:
        """Test NotImplementedError raised if invalid resample method passed."""
        _touch_inputs(mock_paths, skip=("output_file_path", "area_surfs"))
        with pytest.raises(NotImplementedError, match="not implemented"):
            metric_resample(**mock_paths, method="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize("method", ["ADAP_BARY_AREA", "BARYCENTRIC"])
    def test_success(self, mock_paths: dict[str, Any], method: str) -> None:
        """Test successful call."""
        _touch_inputs(mock_paths, skip=("output_file_path",))

        mock_result = MagicMock(metric_out=Path(mock_paths["output_file_path"]))

        def _produce() -> MagicMock:
            Path(mock_paths["output_file_path"]).touch()
            return mock_result

        func_path = "neuromaps_prime.transforms.surface.workbench.metric_resample"
        with _run_patched(func_path, side_effect=lambda **_: _produce()) as mock_wb:
            metric_resample(**mock_paths, method=method)  # type: ignore[arg-type]
            mock_wb.assert_called_once()
            assert Path(mock_paths["output_file_path"]).exists()

    def test_missing_output(self, mock_paths: dict[str, Any]) -> None:
        """Test FileNotFoundError raised if output file is missing."""
        _touch_inputs(mock_paths, skip=("output_file_path",))

        mock_result = MagicMock(metric_out=Path(mock_paths["output_file_path"]))

        func_path = "neuromaps_prime.transforms.surface.workbench.metric_resample"
        with (
            _run_patched(func_path, return_value=mock_result),
            pytest.raises(FileNotFoundError, match="Metric out not found"),
        ):
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
        ("missing", "msg"),
        [
            ("input_file_path", "Input file"),
            ("current_sphere", "Current sphere"),
            ("new_sphere", "New sphere"),
        ],
    )
    def test_missing_inputs(
        self, mock_paths: dict[str, Any], missing: str, msg: str
    ) -> None:
        """Each missing input should raise FileNotFoundError with correct message."""
        _touch_inputs(mock_paths, skip=(missing, "output_file_path", "area_surfs"))
        with pytest.raises(FileNotFoundError, match=msg):
            label_resample(**mock_paths, method="ADAP_BARY_AREA")

    def test_invalid_resample_method(self, mock_paths: dict[str, Any]) -> None:
        """Test NotImplementedError raised if invalid resample method passed."""
        _touch_inputs(mock_paths, skip=("output_file_path", "area_surfs"))
        with pytest.raises(NotImplementedError, match="not implemented"):
            label_resample(**mock_paths, method="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize("method", ["ADAP_BARY_AREA", "BARYCENTRIC"])
    def test_success(self, mock_paths: dict[str, Any], method: str) -> None:
        """Test successful call."""
        _touch_inputs(mock_paths, skip=("output_file_path"))

        mock_result = MagicMock(label_out=Path(mock_paths["output_file_path"]))

        def _produce() -> MagicMock:
            Path(mock_paths["output_file_path"]).touch()
            return mock_result

        func_path = "neuromaps_prime.transforms.surface.workbench.label_resample"
        with _run_patched(func_path, side_effect=lambda **_: _produce()) as mock_wb:
            label_resample(**mock_paths, method=method)  # type: ignore[arg-type]
            mock_wb.assert_called_once()
            assert Path(mock_paths["output_file_path"]).exists()

    def test_missing_output(self, mock_paths: dict[str, Any]) -> None:
        """Test FileNotFoundError raised if output file is missing."""
        _touch_inputs(mock_paths, skip=("output_file_path",))

        mock_result = MagicMock(label_out=Path(mock_paths["output_file_path"]))

        func_path = "neuromaps_prime.transforms.surface.workbench.label_resample"
        with (
            _run_patched(func_path, return_value=mock_result),
            pytest.raises(FileNotFoundError, match="Label out not found"),
        ):
            label_resample(**mock_paths, method="ADAP_BARY_AREA")
