"""Functions for surface transformations using niwrap."""

from pathlib import Path

from niwrap import workbench

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.models import SurfaceTransform
from neuromaps_prime.transforms.utils import (
    estimate_surface_density,
    find_common_density,
    find_highest_density,
    validate,
)


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
        - area_surfs : tuple[Path, Path] | None
            Tuple of area surfaces for adaptive barycentric resampling.

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

    result = workbench.metric_resample(
        metric_in=input_file_path,
        current_sphere=current_sphere,
        new_sphere=new_sphere,
        metric_out=output_file_path,
        **kwargs,
    )
    if not result.metric_out.exists():
        raise FileNotFoundError(f"Metric out not found: {result.metric_out}")

    return result


def label_resample(
    input_file_path: Path,
    current_sphere: Path,
    new_sphere: Path,
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
        - area_surfs : tuple[Path, Path] | None
            Tuple of area surfaces for adaptive barycentric resampling.

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

    result = workbench.label_resample(
        label_in=input_file_path,
        current_sphere=current_sphere,
        new_sphere=new_sphere,
        label_out=output_file_path,
        **kwargs,
    )
    if not result.label_out.exists():
        raise FileNotFoundError(f"Label out not found: {result.label_out}")

    return result


def _get_hop_output_file(
    output_file_path: str, source: str, next_target: str, density: str, hemisphere: str
) -> str:
    """Generate hop output file path based on parameters."""
    p = Path(output_file_path)
    parent = p.parent
    # Remove all suffixes for the base name
    base = p.stem  # removes only the last suffix
    # If the file has double suffix (e.g., .func.gii), remove both
    if p.suffix == ".gii" and p.with_suffix("").suffix:
        base = p.with_suffix("").stem
    # Always use .surf.gii for intermediate sphere files
    suffix = ".surf.gii"
    return str(
        parent / f"{base}_{source}_"
        f"to_{next_target}_den-{density}_hemi-{hemisphere}{suffix}"
    )


def _surface_to_surface(
    graph: NeuromapsGraph,
    source: str,
    target: str,
    density: str,
    hemisphere: str,
    output_file_path: str,
    add_edge: bool = True,
) -> SurfaceTransform | None:
    """Perform a surface-to-surface transformation from source to target space.

    Parameters
    ----------
    graph : NeuromapsGraph
        The neuromaps graph containing the resources.
    source : str
        The source space name.
    target : str
        The target space name.
    density : str
        The density of the surfaces.
    hemisphere : str
        The hemisphere ('left' or 'right').
    add_edge : bool, optional
        Whether to add the resulting transform as an edge in the graph. Default is True.

    Returns:
    -------
    transform : SurfaceTransform
        The resulting surface-to-surface transform resource.

    Raises:
    ------
    ValueError
        If no valid path is found or if source and target are the same.
        If hemisphere value is invalid.
    FileNotFoundError
        If any input file does not exist.

    """
    validate(graph, source, target)
    shortest_path = graph.find_path(
        source=source, target=target, edge_type="surface_to_surface"
    )
    resource_type = "sphere"

    if not shortest_path or len(shortest_path) < 1:
        raise ValueError(f"No valid path found from {source} to {target}.")

    elif len(shortest_path) == 1:
        raise ValueError(f"Source and target spaces are the same: {source}.")

    elif len(shortest_path) == 2:
        return graph.fetch_surface_to_surface_transform(
            source=source,
            target=target,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )

    _transform = graph.fetch_surface_to_surface_transform(
        source=shortest_path[0],
        target=shortest_path[1],
        density=density,
        hemisphere=hemisphere,
        resource_type=resource_type,
    )

    for i in range(2, len(shortest_path)):
        next_target = shortest_path[i]

        hop_output_file = _get_hop_output_file(
            output_file_path, source, next_target, density, hemisphere
        )

        _transform = _two_hops(
            graph=graph,
            source_space=source,
            mid_space=shortest_path[i - 1],
            target_space=next_target,
            density=density,
            hemisphere=hemisphere,
            output_file_path=hop_output_file,
            first_transform=_transform,
        )

        transform = SurfaceTransform(
            name=f"{source}_to_{next_target}_{density}_{hemisphere}_{resource_type}",
            description=f"Surface Transform from {source} to {next_target}",
            source_space=source,
            target_space=next_target,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
            file_path=_transform.sphere_out,
        )

        if add_edge:
            graph.add_transform(
                source_space=source,
                target_space=next_target,
                key="surface_to_surface",
                surface_transform=transform,
            )

    return transform


def _two_hops(
    graph: NeuromapsGraph,
    source_space: str,
    mid_space: str,
    target_space: str,
    density: str,
    hemisphere: str,
    output_file_path: str,
    first_transform: SurfaceTransform | None = None,
) -> workbench.SurfaceSphereProjectUnprojectOutputs:
    """Perform a two-hop surface-to-surface transformation via an intermediate space.

    This is a wrapper around the surface_sphere_project_unproject function with
    default fetching of the intermediate resources.
    If you are going from 1 -> 2 -> 3,
    1 is the source_space, 2 is the mid_space, and 3 is the target_space.

    This function then fetches
    1 -> 2 transform (first_transform/sphere_in),
    2 sphere atlas (sphere_project_to),
    and 2 -> 3 transform (sphere_unproject_from),
    and performs the projection and unprojection to get from 1 -> 3.

    You can provide the first_transform (1 -> 2) to avoid fetching it again.
    But it is optional.

    Parameters
    ----------
    graph : NeuromapsGraph
        The neuromaps graph containing the resources.
    source_space : str
        The source space name.
    mid_space : str
        The intermediate space name.
    target_space : str
        The target space name.
    density : str
        The density of the surfaces.
    hemisphere : str
        The hemisphere ('left' or 'right').
    output_file_path : str
        Path to the output GIFTI surface file.
    first_transform : SurfaceTransform | None, optional
        Pre-fetched transform from source to mid space. If None, it will be fetched.

    Returns:
    -------
    result : workbench.SurfaceSphereProjectUnprojectOutputs
        Object containing the path to the output spherical surface as result.sphere_out.

    Raises:
    ------
    ValueError
        If no surface transform is found
        for the source to mid space or mid to target space.
    FileNotFoundError
        If any input file does not exist.
    """
    if first_transform is None:
        first_transform = graph.fetch_surface_to_surface_transform(
            source=source_space,
            target=mid_space,
            density=density,
            hemisphere=hemisphere,
            resource_type="sphere",
        )
        if first_transform is None:
            raise ValueError(
                f"No surface transform found from {source_space} to {mid_space}"
            )

    sphere_in = first_transform.fetch()

    highest_common_density = find_common_density(graph, mid_space, target_space)

    surface_atlas = graph.fetch_surface_atlas(
        space=mid_space,
        hemisphere=hemisphere,
        density=highest_common_density,
        resource_type="sphere",
    )
    if surface_atlas is None:
        raise ValueError(f"No surface atlas found for {mid_space}")

    sphere_project_to = surface_atlas.fetch()

    unproject_transform = graph.fetch_surface_to_surface_transform(
        source=mid_space,
        target=target_space,
        density=highest_common_density,
        hemisphere=hemisphere,
        resource_type="sphere",
    )
    if unproject_transform is None:
        raise ValueError(
            f"No surface transform found from {mid_space} to {target_space}"
        )

    sphere_unproject_from = unproject_transform.fetch()

    resulting_transform = surface_sphere_project_unproject(
        sphere_in=sphere_in,
        sphere_project_to=sphere_project_to,
        sphere_unproject_from=sphere_unproject_from,
        sphere_out=output_file_path,
    )

    return resulting_transform


def surface_to_surface_transformer(
    graph: NeuromapsGraph,
    transformer_type: str,
    input_file: Path,
    source_space: str,
    target_space: str,
    hemisphere: str,
    output_file_path: str,
    source_density: str | None = None,
    target_density: str | None = None,
    area_resource: str = "midthickness",
    add_edge: bool = True,
) -> workbench.MetricResampleOutputs | workbench.LabelResampleOutputs | None:
    """Public interface for performing surface-to-surface transformations.

    Parameters
    ----------
    graph : NeuromapsGraph
        The neuromaps graph containing the resources.
    transformer_type : str
        Type of transformation: 'metric' or 'label'.
    input_file : Path
        Path to the input GIFTI file (metric or label).
    source_space : str
        The source space name.
    target_space : str
        The target space name.
    hemisphere : str
        The hemisphere ('left' or 'right').
    output_file_path : str
        Path to the output GIFTI file.
    source_density : str, optional
        Density of the source surface.
        If None, it will be estimated from the input file.
    target_density : str, optional
        Density of the target surface.
        If None, the highest available density will be used.
    add_edge : bool, optional
        Whether to add the resulting transform as an edge in the graph.
        Default is True.

    Returns:
    -------
    result : workbench.MetricResampleOutputs | workbench.LabelResampleOutputs | None
        The resulting resampled metric or label output.

    Raises:
    ------
    ValueError
        If transformer_type is invalid.
        If the input file is not found.
        If the source surface is not found.
    FileNotFoundError
        If any input file does not exist.
        If the output file cannot be created.

    """
    if transformer_type not in ["metric", "label"]:
        raise ValueError(
            f"Invalid transformer_type: {transformer_type}. "
            "Must be 'metric' or 'label'."
        )
    if source_density is None:
        source_density = estimate_surface_density(input_file)

    transform = _surface_to_surface(
        graph=graph,
        source=source_space,
        target=target_space,
        density=source_density,
        hemisphere=hemisphere,
        output_file_path=output_file_path,
        add_edge=add_edge,
    )

    if transform is None:
        return None

    if target_density is None:
        target_density = find_highest_density(graph=graph, space=target_space)

    new_sphere_atlas = graph.fetch_surface_atlas(
        space=target_space,
        hemisphere=hemisphere,
        density=target_density,
        resource_type="sphere",
    )
    if new_sphere_atlas is None:
        raise ValueError(f"No surface atlas found for {target_space}")
    new_sphere = new_sphere_atlas.fetch()

    current_area_atlas = graph.fetch_surface_atlas(
        space=source_space,
        hemisphere=hemisphere,
        density=source_density,
        resource_type=area_resource,
    )
    if current_area_atlas is None:
        raise ValueError(f"No {area_resource} surface found for {source_space}")
    current_area: Path = current_area_atlas.fetch()

    new_area_atlas = graph.fetch_surface_atlas(
        space=target_space,
        hemisphere=hemisphere,
        density=target_density,
        resource_type=area_resource,
    )
    if new_area_atlas is None:
        raise ValueError(f"No {area_resource} surface found for {target_space}")
    new_area: Path = new_area_atlas.fetch()

    area_surfs = workbench.metric_resample_area_surfs_params(
        current_area=current_area, new_area=new_area
    )

    kwargs = {
        "current_sphere": transform.fetch(),
        "new_sphere": new_sphere,
        "area_surfs": area_surfs,
        "method": "ADAP_BARY_AREA",
    }

    if transformer_type == "label":
        resampled_output = label_resample(
            input_file_path=input_file, output_file_path=output_file_path, **kwargs
        )

    elif transformer_type == "metric":
        resampled_output = metric_resample(
            input_file_path=input_file, output_file_path=output_file_path, **kwargs
        )

    return resampled_output
