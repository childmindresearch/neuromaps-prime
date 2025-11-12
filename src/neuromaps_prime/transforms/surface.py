"""Functions for surface transformations using niwrap."""

from pathlib import Path

from niwrap import workbench


def surface_sphere_project_unproject(
    sphere_in: Path,
    sphere_project_to: Path,
    sphere_unproject_from: Path,
    sphere_out: str,
) -> workbench.SurfaceSphereProjectUnprojectOutputs:
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
    result : workbench.SurfaceSphereProjectUnprojectOutputs
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

    return result


def metric_resample(
    input_file_path: Path,
    current_sphere: Path,
    new_sphere: Path,
    method: str,
    area_surfs: workbench.metric_resample_area_surfs_params,
    output_file_path: str,
    **kwargs,
) -> workbench.MetricResampleOutputs:
    """Resample a surface metric from one sphere to another.

    Parameters
    ----------
    input_file_path : Path
        Path to input metric file.
    current_sphere : Path
        Path to current spherical surface.
    new_sphere : Path
        Path to new spherical surface.
    method : str
        Resampling method.
    area_surfs : workbench.metric_resample_area_surfs_params
        Area surfaces for adaptive barycentric resampling.
    output_file_path : str
        Path to output metric file.
    **kwargs
        Additional keyword arguments passed to workbench.metric_resample.


    Returns:
    -------
    result : workbench.MetricResampleOutputs
        Object containing the path to the output metric as result.metric_out.

    Raises:
    ------
    FileNotFoundError
        If any input file does not exist.
    """
    if not input_file_path.exists():
        raise FileNotFoundError(f"Input metric file not found: {input_file_path}")
    if not current_sphere.exists():
        raise FileNotFoundError(f"Current sphere file not found: {current_sphere}")
    if not new_sphere.exists():
        raise FileNotFoundError(f"New sphere file not found: {new_sphere}")

    if method != "ADAP_BARY_AREA":
        raise NotImplementedError(
            f"Resampling method '{method}' is not implemented in this function."
        )

    result = workbench.metric_resample(
        metric_in=input_file_path,
        current_sphere=current_sphere,
        new_sphere=new_sphere,
        method=method,
        area_surfs=area_surfs,
        metric_out=output_file_path,
    )
    if not result.metric_out.exists():
        raise FileNotFoundError(f"Metric out not found: {result.metric_out}")

    return result


def label_resample(
    input_file_path: Path,
    current_sphere: Path,
    new_sphere: Path,
    method: str,
    area_surfs: workbench.label_resample_area_surfs_params,
    output_file_path: str,
    **kwargs,
) -> workbench.LabelResampleOutputs:
    """Resample a surface label from one sphere to another.

    Parameters
    ----------
    input_file_path : Path
        Path to input label file.
    current_sphere : Path
        Path to current spherical surface.
    new_sphere : Path
        Path to new spherical surface.
    method : str
        Resampling method.
    area_surfs : workbench.label_resample_area_surfs_params
        Area surfaces for adaptive barycentric resampling.
    output_file_path : str
        Path to output label file.
    **kwargs
        Additional keyword arguments passed to workbench.label_resample.

    Returns:
    -------
    result : workbench.LabelResampleOutputs
        Object containing the path to the output label as result.label_out.

    Raises:
    ------
    FileNotFoundError
        If any input file does not exist.
    """
    if not input_file_path.exists():
        raise FileNotFoundError(f"Input label file not found: {input_file_path}")
    if not current_sphere.exists():
        raise FileNotFoundError(f"Current sphere file not found: {current_sphere}")
    if not new_sphere.exists():
        raise FileNotFoundError(f"New sphere file not found: {new_sphere}")

    if method != "ADAP_BARY_AREA":
        raise NotImplementedError(
            f"Resampling method '{method}' is not implemented in this function."
        )

    result = workbench.label_resample(
        label_in=input_file_path,
        current_sphere=current_sphere,
        new_sphere=new_sphere,
        method=method,
        area_surfs=area_surfs,
        label_out=output_file_path,
    )
    if not result.label_out.exists():
        raise FileNotFoundError(f"Label out not found: {result.label_out}")

    return result
