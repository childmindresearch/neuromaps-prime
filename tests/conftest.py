"""Global pytest fixtures, arguments, and options."""

from pathlib import Path
from typing import Generator

import pytest
from niwrap import Runner, ants, get_global_runner, workbench

from neuromaps_prime.utils import set_runner


def pytest_addoption(parser: pytest.Parser):
    """Add option(s) to pytest parser."""
    parser.addoption(
        "--runner",
        action="store",
        default="local",
        help="Styx runner type to use: ['local', 'docker', 'singularity']",
    )
    parser.addoption(
        "--runner-images",
        action="store",
        default=None,
        help="Optional dict string of image overrides to use for StyxRunner.",
    )
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help="Directory where test data is located.",
    )


@pytest.fixture(scope="session", autouse=True)
def runner(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Generator[Runner, None, None]:
    """Globally set runner for the testing suite."""
    tmp_dir = tmp_path_factory.mktemp("styx_tmp")
    set_runner(
        runner=request.config.getoption("--runner").lower(),
        image_overrides=request.config.getoption("--runner-images"),
        data_dir=tmp_dir,
    )
    yield get_global_runner()


@pytest.fixture
def require_data(request: pytest.FixtureRequest):
    data_dir_str = request.config.getoption("--data-dir")
    if data_dir_str is None:
        pytest.skip("Data directory not provided")
    data_dir = Path(data_dir_str).resolve()
    if not data_dir.exists():
        pytest.skip("Data directory does not exist")


@pytest.fixture(scope="session", autouse=True)
def data_dir(request: pytest.FixtureRequest) -> Generator[Path | None, None, None]:
    """Yield data directory from pytest command-line."""
    data_dir = request.config.getoption("--data-dir")
    if data_dir is not None:
        yield Path(data_dir).resolve()
    else:
        yield None


@pytest.fixture
def require_ants(runner: Runner) -> None:
    try:
        ants.ants_apply_transforms(
            reference_image=".",
            output=ants.ants_apply_transforms_warped_output_params("."),
            runner=runner,
        )
    except FileNotFoundError:
        pytest.skip("ANTs not available in environment")
    except Exception:
        # Failures for other reasons are ignored
        pass


@pytest.fixture
def require_workbench(runner: Runner) -> None:
    try:
        workbench.nifti_information(nifti_file=".", runner=runner)
    except FileNotFoundError:
        pytest.skip("wb_command not available in environment")
    except Exception:
        pass
