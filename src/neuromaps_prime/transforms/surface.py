"""Functions for surface transformations using niwrap."""

from pathlib import Path
from typing import Literal

from niwrap import workbench

_RESAMPLE_METHODS = frozenset({"ADAP_BARY_AREA", "BARYCENTRIC"})


def surface_sphere_project_unproject(
    sphere_in: str | Path,
    sphere_project_to: str | Path,
    sphere_unproject_from: str | Path,
    sphere_out: str | Path,
) -> workbench.SurfaceSphereProjectUnprojectOutputs:
    """Project and unproject a surface from one sphere to another.

    Args:
        sphere_in: Input spherical surface file path.
        sphere_project_to: File path of spherical surface to project to.
        sphere_unproject_from: File path of spherical surface to unproject from.
        sphere_out: Path to output spherical surface.

    Returns:
        Object containing the path to the output spherical surface as result.sphere_out.

    Raises:
        FileNotFoundError: If any input file does not exist.
    """
    sphere_in, sphere_project_to, sphere_unproject_from = map(
        Path, [sphere_in, sphere_project_to, sphere_unproject_from]
    )
    # Validate input files
    for path, desc in zip(
        [sphere_in, sphere_project_to, sphere_unproject_from],
        ["Input sphere", "Sphere to project to", "Sphere to unproject from"],
    ):
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    result = workbench.surface_sphere_project_unproject(
        sphere_in=sphere_in,
        sphere_project_to=sphere_project_to,
        sphere_unproject_from=sphere_unproject_from,
        sphere_out=str(sphere_out),
    )
    if not result.sphere_out.exists():
        raise FileNotFoundError(f"Sphere out not found: {sphere_out}")
    return result


def metric_resample(
    input_file_path: str | Path,
    current_sphere: str | Path,
    new_sphere: str | Path,
    method: Literal["ADAP_BARY_AREA", "BARYCENTRIC"],
    area_surfs: workbench.MetricResampleAreaSurfsParamsDict,
    output_file_path: str,
    **kwargs,
) -> workbench.MetricResampleOutputs:
    """Resample a surface metric from one sphere to another.

    Args:
        input_file_path: Input metric file path.
        current_sphere: File path to current spherical surface.
        new_sphere: File path to new spherical surface.
        method: Resampling method.
        area_surfs: Area surfaces to perform vertex area correction on.
        output_file_path: Path to output metric file.

    Returns:
        Object containing the path to the output metric as result.metric_out.

    Raises:
        FileNotFoundError: If any input file does not exist or output file not created.
        NotImplementedError: If selected resampling method is not available.
    """
    input_file_path, current_sphere, new_sphere = map(
        Path, [input_file_path, current_sphere, new_sphere]
    )
    # Validate input files
    for path, desc in zip(
        [input_file_path, current_sphere, new_sphere],
        ["Input file", "Current sphere", "New sphere"],
    ):
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    if method not in _RESAMPLE_METHODS:
        raise NotImplementedError(
            f"Resampling method '{method}' is not implemented in this function."
        )

    result = workbench.metric_resample(
        metric_in=input_file_path,
        current_sphere=current_sphere,
        new_sphere=new_sphere,
        method=method,
        area_surfs=area_surfs,
        metric_out=str(output_file_path),
    )
    if not result.metric_out.exists():
        raise FileNotFoundError(f"Metric out not found: {result.metric_out}")
    return result


def label_resample(
    input_file_path: str | Path,
    current_sphere: str | Path,
    new_sphere: str | Path,
    method: Literal["ADAP_BARY_AREA", "BARYCENTRIC"],
    area_surfs: workbench.LabelResampleAreaSurfsParamsDict,
    output_file_path: str,
    **kwargs,
) -> workbench.LabelResampleOutputs:
    """Resample a surface label from one sphere to another.

    Args:
        input_file_path: Input label file path.
        current_sphere: File path to current spherical surface.
        new_sphere: File path to new spherical surface.
        method: Resampling method.
        area_surfs: Area surfaces to perform vertex area correction on.
        output_file_path: Path to output label file.

    Returns:
        Object containing the path to the output label as result.label_out.

    Raises:
        FileNotFoundError: If any input file does not exist or output file not created.
        NotImplementedError: If selected resampling method is not available.
    """
    input_file_path, current_sphere, new_sphere = map(
        Path, [input_file_path, current_sphere, new_sphere]
    )
    # Validate input files
    for path, desc in zip(
        [input_file_path, current_sphere, new_sphere],
        ["Input file", "Current sphere", "New sphere"],
    ):
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    if method not in _RESAMPLE_METHODS:
        raise NotImplementedError(
            f"Resampling method '{method}' is not implemented in this function."
        )

    result = workbench.label_resample(
        label_in=input_file_path,
        current_sphere=current_sphere,
        new_sphere=new_sphere,
        method=method,
        area_surfs=area_surfs,
        label_out=str(output_file_path),
    )
    if not result.label_out.exists():
        raise FileNotFoundError(f"Label out not found: {result.label_out}")
    return result
