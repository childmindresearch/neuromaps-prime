"""Wrappers for niwrap functions."""

from pathlib import Path

from niwrap import workbench as wb


def surface_sphere_project_unproject(
    sphere_in: Path,
    sphere_project_to: Path,
    sphere_unproject_from: Path,
    sphere_out: Path,
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
    sphere_out : Path
        Path to output spherical surface.

    Returns:
    -------
    Path
        Path to the created output file.

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

    sphere_out.parent.mkdir(parents=True, exist_ok=True)

    wb.surface_sphere_project_unproject(
        str(sphere_in),
        str(sphere_project_to),
        str(sphere_unproject_from),
        str(sphere_out),
    )

    if not sphere_out.exists():
        raise RuntimeError(f"Failed to create output file: {sphere_out}")

    return Path(sphere_out)
