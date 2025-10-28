"""Global pytest fixtures, arguments, and options."""

from pathlib import Path

import pytest

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
        help="Image overrides for Styx runner.",
    )


@pytest.fixture(scope="session")
def runner(request: pytest.FixtureRequest, tmp_path: Path):
    """Set runner used for the session based on parser."""
    runner = request.config.getoption("--runner")
    print(runner)
    images = request.config.getoption("--runner-images")
    set_runner(
        runner=runner.lower(),
        image_map=images,
        data_dir=tmp_path,
    )
    yield runner
