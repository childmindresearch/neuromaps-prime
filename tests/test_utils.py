"""Tests for utility functions in neuromaps-prime."""

from typing import Any

import pytest
from niwrap import DockerRunner, LocalRunner, SingularityRunner, get_global_runner

from neuromaps_prime import utils


class TestSetRunner:
    """Testing of set runner utility method."""

    @pytest.mark.parametrize(
        "runner, expected_runner",
        [
            ("local", LocalRunner),
            ("docker", DockerRunner),
            ("singularity", SingularityRunner),
        ],
    )
    def test_set_runner(
        self,
        runner: str,
        expected_runner: LocalRunner | DockerRunner | SingularityRunner,
    ):
        """Test setting runner."""
        images = {"container1": "/path/to/container"} if runner == "singularity" else {}
        utils.set_runner(runner=runner, image_map=images)
        test_runner = get_global_runner()
        assert isinstance(test_runner, expected_runner)

    @pytest.mark.parametrize(
        "runner, expected_runner",
        [
            ("LOCAL", LocalRunner),
            ("LoCaL", LocalRunner),
            ("DOCKER", DockerRunner),
            ("doCKeR", DockerRunner),
            ("SINGULARITY", SingularityRunner),
            ("siNgUlaRiTY", SingularityRunner),
        ],
    )
    def test_runner_case_insensitive(
        self,
        runner: str,
        expected_runner: LocalRunner | DockerRunner | SingularityRunner,
    ):
        """Test case-insensitivty of setting runner."""
        images = (
            {"container1": "/path/to/container"}
            if runner.lower() == "singularity"
            else {}
        )
        utils.set_runner(runner=runner, image_map=images)
        test_runner = get_global_runner()
        assert isinstance(test_runner, expected_runner)

    @pytest.mark.parametrize(
        "runner, expected_runner",
        [
            ("local", LocalRunner),
            ("docker", DockerRunner),
            ("singularity", SingularityRunner),
        ],
    )
    def test_runner_valid_kwargs(
        self,
        runner: str,
        expected_runner: LocalRunner | DockerRunner | SingularityRunner,
    ):
        """Test passing valid kwargs to set local runner."""
        data_dir = "/path/to/data/dir"
        images = {"container1": "/path/to/container"} if runner == "singularity" else {}
        utils.set_runner(runner=runner, image_map=images, data_dir=data_dir)
        test_runner = get_global_runner()
        assert isinstance(test_runner, expected_runner)
        assert str(test_runner.data_dir) == data_dir

    @pytest.mark.parametrize("runner", ("local", "docker", "singularity"))
    def test_runner_invalid_kwargs(self, runner: str):
        """Test passing invalid kwargs to set local runner."""
        images = {"container1": "/path/to/container"} if runner == "singularity" else {}
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            utils.set_runner(runner=runner, image_map=images, arg="val123")

    # empty map
    def test_empty_image_map(self):
        """Test exception raised when no container mappins for singularity runner."""
        with pytest.raises(ValueError, match="No container mappings"):
            utils.set_runner("singularity")

    # invalid map
    @pytest.mark.parametrize("image_map", (set(), [], ()))
    def test_invalid_image_map_type(self, image_map: Any):
        """Test exception raised when map is not a dict."""
        with pytest.raises(TypeError, match="Expected image_map dictionary"):
            utils.set_runner(runner="local", image_map=image_map)

    # not implemented type
    def test_unimplemented_runner(self):
        """Test exception raised for unimplemented runners."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            utils.set_runner(runner="invalid")
