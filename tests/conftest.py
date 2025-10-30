"""Global pytest fixtures, arguments, and options."""

from typing import Generator

import pytest
from niwrap import Runner, ants, get_global_runner

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
        help="Optional JSON/dict string of Image overrides for Styx runner.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom marker for tests."""
    config.addinivalue_line(
        "markers", "requires_ants: skip test if ANTs is not available"
    )


@pytest.fixture(scope="session", autouse=True)
def runner(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Generator[Runner, None, None]:
    """Globally set runner for the testing suite."""
    tmp_dir = tmp_path_factory.mktemp("styx_tmp")
    set_runner(
        runner=request.config.getoption("--runner").lower(),
        image_map=request.config.getoption("--runner-images"),
        data_dir=tmp_dir,
    )
    yield get_global_runner()


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
