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
    output_file_path : str
        Path to output metric file.
    **kwargs
        Additional keyword arguments passed to workbench.metric_resample.
        Common options include:
        - method : str
            Resampling method. Default is 'ADAP_BARY_AREA'.
        - area_surfs : workbench.metric_resample_area_surfs_params(Path, Path)
            Area surfaces for adaptive barycentric resampling.

    Returns:
    -------
    result : workbench.MetricResampleOutputs
        Object containing the path to the output metric as result.metric_out.

    Raises:
    ------
    FileNotFoundError
        If any input file does not exist.
    """
    current_sphere = kwargs.get("current_sphere")
    new_sphere = kwargs.get("new_sphere")
    method = kwargs.get("method")
    area_surfs = kwargs.get("area_surfs")

    if current_sphere is None:
        raise ValueError("current_sphere must be provided in kwargs.")
    if new_sphere is None:
        raise ValueError("new_sphere must be provided in kwargs.")
    if method is None:
        raise ValueError("method must be provided in kwargs.")
    if area_surfs is None and method == "ADAP_BARY_AREA":
        raise ValueError(
            "area_surfs must be provided in kwargs for ADAP_BARY_AREA method."
        )

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
        metric_out=output_file_path,
        **kwargs,
    )
    if not result.metric_out.exists():
        raise FileNotFoundError(f"Metric out not found: {result.metric_out}")

    return result


def label_resample(
    input_file_path: Path,
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
    output_file_path : str
        Path to output label file.
    **kwargs
        Additional keyword arguments passed to workbench.label_resample.
        Common options include:
        - method : str
            Resampling method. Default is 'ADAP_BARY_AREA'.
        - area_surfs : workbench.label_resample_area_surfs_params(Path, Path)
            Area surfaces for adaptive barycentric resampling.

    Returns:
    -------
    result : workbench.LabelResampleOutputs
        Object containing the path to the output label as result.label_out.

    Raises:
    ------
    FileNotFoundError
        If any input file does not exist.
    """
    current_sphere = kwargs.get("current_sphere")
    new_sphere = kwargs.get("new_sphere")
    method = kwargs.get("method")
    area_surfs = kwargs.get("area_surfs")

    if current_sphere is None:
        raise ValueError("current_sphere must be provided in kwargs.")
    if new_sphere is None:
        raise ValueError("new_sphere must be provided in kwargs.")
    if method is None:
        raise ValueError("method must be provided in kwargs.")
    if area_surfs is None and method == "ADAP_BARY_AREA":
        raise ValueError(
            "area_surfs must be provided in kwargs for ADAP_BARY_AREA method."
        )

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
        label_out=output_file_path,
        **kwargs,
    )
    if not result.label_out.exists():
        raise FileNotFoundError(f"Label out not found: {result.label_out}")

    return result
