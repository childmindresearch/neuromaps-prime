"""Utility functions."""

from niwrap import use_docker, use_local, use_singularity


def set_runner(
    runner: str,
    image_map: dict[str, str] = {},
    **kwargs,
) -> None:
    """Set StyxRunner to use for NiWrap.

    Args:
        runner: Styx runner type to use (one of 'local', 'docker', 'singularity).
        image_map: Dictionary of container paths to override.

    Raises:
        TypeError: if image_map is not dictionary.
        ValueError: if singularity runner is used and image_map not provided.
        NotImplementedError: if provided runner not a valid StyxRunner.
    """
    if not isinstance(image_map, dict):
        raise TypeError(f"Expected image_map dictionary, got {type(image_map)}")
    match runner_exec := runner.lower():
        case "local":
            use_local(**kwargs)
        case "docker":
            use_docker(
                docker_executable=runner_exec,
                image_overrides=image_map,
                **kwargs,
            )
        case "singularity":
            if not image_map:
                raise ValueError("No container mappings provided.")
            use_singularity(
                singularity_executable=runner_exec, images=image_map, **kwargs
            )
        case _:
            raise NotImplementedError(f"'{runner_exec}' runner not implemented.")
