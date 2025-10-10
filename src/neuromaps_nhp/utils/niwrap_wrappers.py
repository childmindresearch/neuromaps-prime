"""Wrappers for niwrap functions."""

from pathlib import Path

from niwrap import workbench


def surface_sphere_project_unproject(
    sphere_in: Path,
    sphere_project_to: Path,
    sphere_unproject_from: Path,
    sphere_out: str,
) -> Path:
    """Project and unproject a surface from one sphere to another.

    Parameters
    ----------
    sphere_in : Path
        Path to input spherical surface.
    sphere_project_to : Path
        Path to spherical surface to project to.
    sphere_unproject_from : Path
        Path to spherical surface to unproject from.
    sphere_out : str
        Path to output spherical surface.

    Returns:
    -------
    result : Object
        Object containing the path to the output spherical surface as result.sphere_out.

    Raises:
    ------
    FileNotFoundError
        If any input file does not exist.
    """
    if not sphere_in.exists():
        raise FileNotFoundError(f"Input sphere file not found: {sphere_in}")
    if not sphere_project_to.exists():
        raise FileNotFoundError(f"Sphere to project to not found: {sphere_project_to}")
    if not sphere_unproject_from.exists():
        raise FileNotFoundError(
            f"Sphere to unproject from not found: {sphere_unproject_from}"
        )

    result = workbench.surface_sphere_project_unproject(
        sphere_in=sphere_in,
        sphere_project_to=sphere_project_to,
        sphere_unproject_from=sphere_unproject_from,
        sphere_out=sphere_out,
    )
    if not result.sphere_out.exists():
        raise FileNotFoundError(f"Sphere out not found: {sphere_out}")

    return result.sphere_out
