"""Utility functions."""

from niwrap import Runner, get_global_runner, use_docker, use_local, use_singularity
from styxdefs import set_global_runner


def set_runner(
    runner: Runner | str,
    image_overrides: dict[str, str] | None = None,
    **kwargs,
) -> Runner:
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

    if isinstance(runner, str):
        match runner_exec := runner.lower():
            case "local":
                use_local(**kwargs)
            case "docker" | "singularity":
                runner_fn = use_docker if runner_exec == "docker" else use_singularity
                runner_fn(
                    **{f"{runner_exec}_executable": runner_exec},
                    image_overrides=image_overrides,
                    **kwargs,
                )
            case _:
                raise NotImplementedError(f"'{runner_exec}' runner not implemented.")
    else:
        runner.image_overrides = image_overrides
        set_global_runner(runner)

    return get_global_runner()
