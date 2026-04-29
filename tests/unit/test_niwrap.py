"""Unit tests for Niwrap helpers."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from niwrap import (
    DockerRunner,
    LocalRunner,
    Runner,
    SingularityRunner,
    get_global_runner,
)
from styxpodman import PodmanRunner

from neuromaps_prime.niwrap import generate_exec_folder, resolve_runner, setup_runner

if TYPE_CHECKING:
    from pathlib import Path


class TestSetupRunner:
    """Test suite for niwrap.setup_runner."""

    def test_default(self, tmp_path: Path) -> None:
        """Test default initialization uses auto-detection."""
        ctx = setup_runner(tmp_dir=tmp_path)
        assert isinstance(ctx.logger, logging.Logger)
        assert ctx.runner is not None

    @pytest.mark.parametrize(
        ("runner", "runner_type"),
        [
            ("local", LocalRunner),
            ("docker", DockerRunner),
            ("podman", PodmanRunner),
            pytest.param(
                "singularity",
                SingularityRunner,
                marks=pytest.mark.skipif(
                    os.name == "nt",
                    reason="SingularityRunner not supported on Windows",
                ),
            ),
        ],
    )
    def test_set_runner(
        self, runner: str, runner_type: type[Runner], tmp_path: Path
    ) -> None:
        """Test explicit setting of runner."""
        ctx = setup_runner(runner=runner, tmp_dir=tmp_path)  # type: ignore [arg-type]
        assert isinstance(ctx.runner, runner_type)

    def test_invalid_runner(self, tmp_path: Path) -> None:
        """Test error raised if invalid runner selected."""
        with pytest.raises(NotImplementedError, match="Unknown runner"):
            setup_runner(runner="invalid", tmp_dir=tmp_path)  # type: ignore [arg-type]

    def test_set_tmp_dir(self, tmp_path: Path) -> None:
        """Test setting of data directory works."""
        ctx = setup_runner(tmp_dir=tmp_path)
        assert ctx.runner.data_dir.is_relative_to(tmp_path)
        assert ctx.runner.data_dir.exists()

    @pytest.mark.parametrize(
        ("verbose", "log_level"),
        [
            (0, logging.WARNING),
            (1, logging.INFO),
            (2, logging.DEBUG),
            (5, logging.DEBUG),
        ],
    )
    def test_set_log_level(self, verbose: int, log_level: int, tmp_path: Path) -> None:
        """Test setting of log levels."""
        ctx = setup_runner(verbose=verbose, tmp_dir=tmp_path)
        assert ctx.logger.level == log_level


class TestGenExecFolder:
    """Testing suite for niwrap.generate_exec_folder."""

    def test_create_folder_default(self) -> None:
        """Test folder successfully generated with default arguments."""
        runner = get_global_runner()
        result = generate_exec_folder()
        assert result.exists()
        assert result.is_dir()
        assert runner.execution_counter == 1
        assert result.name == f"{runner.uid}_{runner.execution_counter - 1}_python"

    def test_create_folder_with_suffix(self) -> None:
        """Test folder successfully generates with suffix."""
        runner = get_global_runner()
        result = generate_exec_folder(suffix="pytest")
        assert result.name == f"{runner.uid}_{runner.execution_counter - 1}_pytest"

    def test_error_if_folder_exists(self) -> None:
        """Test folder generation fails if duplicate."""
        runner = get_global_runner()
        generate_exec_folder()
        runner.execution_counter -= 1
        with pytest.raises(FileExistsError):
            generate_exec_folder()


class TestResolveRunner:
    """Test suite for resolve_runner."""

    def test_auto_detects_available_runner(self) -> None:
        """Test auto-detection picks first available runner."""
        with patch(
            "neuromaps_prime.niwrap.shutil.which",
            side_effect=lambda exe: "/usr/bin/docker" if exe == "docker" else None,
        ):
            runner, exe = resolve_runner("auto")

        assert runner == "docker"
        assert exe == "docker"

    def test_auto_falls_back_to_local(self) -> None:
        """Test auto-detection falls back to local when nothing found."""
        with patch("neuromaps_prime.niwrap.shutil.which", return_value=None):
            runner, exe = resolve_runner("auto")

        assert runner == "local"
        assert exe == "local"

    def test_auto_respects_preference_order(self) -> None:
        """Test auto-detection prefers docker over podman over singularity."""
        available = {"podman", "apptainer"}
        with patch(
            "neuromaps_prime.niwrap.shutil.which",
            side_effect=lambda exe: f"/usr/bin/{exe}" if exe in available else None,
        ):
            runner, _ = resolve_runner("auto")

        assert runner == "podman"
