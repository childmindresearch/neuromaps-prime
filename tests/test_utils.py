"""Tests for utility functions in neuromaps-prime."""

from typing import Any

import pytest
from niwrap import (
    DockerRunner,
    LocalRunner,
    Runner,
    SingularityRunner,
    get_global_runner,
    set_global_runner,
)

from neuromaps_prime import utils


@pytest.fixture
def reset_runner():
    """Snapshot and restore runner specified."""
    original = get_global_runner()
    yield
    set_global_runner(original)


class TestSetRunner:
    """Testing of set runner utility method."""

    def test_set_runner_type(self, runner: Runner):
        """Test setting runner via Runner type."""
        test_runner = utils.set_runner(runner=runner)
        assert isinstance(test_runner, type(runner))

    @pytest.mark.parametrize(
        "test_runner, expected_runner",
        [
            ("local", LocalRunner),
            ("docker", DockerRunner),
            ("singularity", SingularityRunner),
            ("LOCAL", LocalRunner),
            ("LoCaL", LocalRunner),
            ("DOCKER", DockerRunner),
            ("doCKeR", DockerRunner),
            ("SINGULARITY", SingularityRunner),
            ("siNgUlaRiTY", SingularityRunner),
        ],
    )
    def test_set_runner_string(self, test_runner: str, expected_runner: Runner):
        """Test setting runner via literal strings."""
        test_runner = utils.set_runner(runner=test_runner)
        assert isinstance(test_runner, expected_runner)

    @pytest.mark.parametrize(
        "input_runner, expected_runner",
        [
            ("local", LocalRunner),
            ("docker", DockerRunner),
            ("singularity", SingularityRunner),
        ],
    )
    def test_runner_valid_kwargs(
        self,
        input_runner: str,
        expected_runner: Runner,
    ):
        """Test passing valid kwargs to set local runner."""
        data_dir = "/path/to/data/dir"
        test_runner = utils.set_runner(runner=input_runner, data_dir=data_dir)
        assert isinstance(test_runner, expected_runner)
        assert str(test_runner.data_dir) == data_dir

    @pytest.mark.parametrize("test_runner", ("local", "docker", "singularity"))
    def test_runner_invalid_kwargs(self, test_runner: str):
        """Test passing invalid kwargs to set local runner."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            utils.set_runner(runner=test_runner, arg="val123")

    @pytest.mark.parametrize("image_overrides", (set(), [], ()))
    def test_invalid_image_overrides_type(self, image_overrides: Any):
        """Test exception raised when invalid map type provided."""
        with pytest.raises(TypeError, match="Expected image_overrides dictionary"):
            utils.set_runner(runner="local", image_overrides=image_overrides)

    def test_unimplemented_runner(self):
        """Test exception raised for unimplemented runners."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            utils.set_runner(runner="invalid")
