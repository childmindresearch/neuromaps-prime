"""NiWrap helpers used throughout the library.

Provides utilities for setting up and tearing down runners. Allows us to use runner
of choice depending on what is available on the system.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Literal, NamedTuple

import niwrap
from styxpodman import PodmanRunner

_LOG_LEVELS = [logging.WARNING, logging.INFO, logging.DEBUG]

RunnerType = Literal["local", "docker", "podman", "singularity"]

_RUNNER_EXECUTABLES: list[tuple[RunnerType, list[str]]] = [
    ("docker", ["docker"]),
    ("podman", ["podman"]),
    ("singularity", ["apptainer", "singularity"]),
]


class StyxContext(NamedTuple):
    """Styx execution context with logger and runner."""

    logger: logging.Logger
    runner: niwrap.Runner
    verbose: bool


def resolve_runner(
    runner: RunnerType | Literal["auto"] = "auto",
) -> tuple[RunnerType, str]:
    """Resolve runner selection, auto-detecting if needed.

    When runner is "auto", checks for available container runtimes on PATH
    in order of preference: docker > podman > apptainer/singularity > local.

    Args:
        runner: Runner type or "auto" for auto-detection.

    Returns:
        Tuple of (runner_type, executable_name).
    """
    if runner != "auto":
        return runner, runner

    for runner_type, executables in _RUNNER_EXECUTABLES:
        for exe in executables:
            if shutil.which(exe):
                return runner_type, exe
    return "local", "local"


def setup_runner(
    runner: RunnerType | Literal["auto"] = "auto",
    tmp_dir: str | Path | None = None,
    image_overrides: dict[str, str] | None = None,
    verbose: int = 0,
    **kwargs,  # noqa: ANN003 (ignore annotation for kwargs)
) -> StyxContext:
    """Setup Styx with appropriate runner for NiWrap.

    Args:
        runner: Type of runner to use. "auto" detects the first available
            container runtime, falling back to "local".
        tmp_dir: Working directory to output to
        image_overrides: Dictionary containing overrides for container tags.
        verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        **kwargs: Additional keyword arguments passed for runner setup.

    Returns:
        Configured logger instance and initialized runner
    """
    runner_type, runner_exec = resolve_runner(runner)

    match runner_type:
        case "local":
            niwrap.use_local()
        case "docker":
            niwrap.use_docker(
                docker_executable=runner_exec,
                image_overrides=image_overrides,
                docker_user_id=0,
                **kwargs,
            )
        case "podman":
            niwrap.set_global_runner(
                runner=PodmanRunner(
                    podman_executable=runner_exec,
                    image_overrides=image_overrides,
                    podman_user_id=0,
                    **kwargs,
                )
            )
        case "singularity":
            niwrap.use_singularity(
                singularity_executable=runner_exec,
                image_overrides=image_overrides,
                **kwargs,
            )
        case _:
            raise NotImplementedError(
                f"Unknown runner selection '{runner}' - please select one of "
                "'auto', 'local', 'docker', 'podman', or 'singularity'"
            )

    styx_runner = niwrap.get_global_runner()
    if tmp_dir is not None:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    styx_runner.data_dir = Path(tempfile.mkdtemp(dir=tmp_dir))
    log_level = min(verbose, len(_LOG_LEVELS) - 1)
    # Expose styx execution logs at max verbosity (e.g. debug), warning otherwise
    styx_logger = logging.getLogger(styx_runner.logger_name)
    styx_logger.setLevel(
        logging.DEBUG if verbose >= len(_LOG_LEVELS) - 1 else logging.WARNING
    )

    neuromaps_prime_logger = logging.getLogger("neuromaps-PRIME")
    neuromaps_prime_logger.setLevel(_LOG_LEVELS[log_level])
    if not neuromaps_prime_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        neuromaps_prime_logger.addHandler(handler)

    return StyxContext(
        logger=neuromaps_prime_logger, runner=styx_runner, verbose=verbose > 0
    )


def generate_exec_folder(suffix: str = "python") -> Path:
    """Generate an execution folder following Styx hash pattern.

    Args:
        suffix: Task to append to suffix of folder (default: 'python')

    Returns:
        Path to created execution folder
    """
    runner = niwrap.get_global_runner()
    dir_path = (
        Path(runner.data_dir) / f"{runner.uid}_{runner.execution_counter}_{suffix}"
    )
    dir_path.mkdir(parents=True)
    runner.execution_counter += 1
    return dir_path
