"""Utility functions."""

from niwrap import use_docker, use_local, use_singularity


def set_runner(
    runner: str,
    image_overrides: dict[str, str] | None = None,
    **kwargs,
) -> None:
    """Set StyxRunner to use for NiWrap.

    Args:
        runner: Styx runner type to use (one of 'local', 'docker', 'singularity).
        image_overrides: Optional dictionary of container tag overrides.

    Raises:
        TypeError: if image_overrides is not dictionary.
        NotImplementedError: if provided runner not a valid StyxRunner.
    """
    if image_overrides is not None and not isinstance(image_overrides, dict):
        raise TypeError(
            f"Expected image_overrides dictionary, got {type(image_overrides)}"
        )
    match runner_exec := runner.lower():
        case "local":
            use_local(**kwargs)
        case "docker":
            use_docker(
                docker_executable=runner_exec,
                image_overrides=image_overrides,
                **kwargs,
            )
        case "singularity":
            use_singularity(
                singularity_executable=runner_exec,
                image_overrides=image_overrides,
                **kwargs,
            )
        case _:
            raise NotImplementedError(f"'{runner_exec}' runner not implemented.")
