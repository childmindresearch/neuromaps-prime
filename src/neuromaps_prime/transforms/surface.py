"""Functions for surface transformations using niwrap."""

from pathlib import Path

from niwrap import workbench

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.models import SurfaceTransform
from neuromaps_prime.transforms.utils import validate


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


def _get_hop_output_file(
    output_file_path: str, source: str, next_target: str, density: str, hemisphere: str
) -> str:
    """Generate hop output file path based on parameters."""
    p = Path(output_file_path)
    parent = p.parent
    # Remove all suffixes for the base name
    base = p.name.split(".")[0] if p.name else "output"
    # Use the last suffix, or default
    suffix = p.suffix if p.suffix else ".surf.gii"
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
    path = graph.find_path(source=source, target=target, edge_type="surface_to_surface")
    resource_type = "sphere"

    if not path or len(path) < 1:
        raise ValueError(f"No valid path found from {source} to {target}.")

    elif len(path) == 1:
        raise ValueError(f"Source and target spaces are the same: {source}.")

    elif len(path) == 2:
        return graph.fetch_surface_to_surface_transform(
            source=source,
            target=target,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )

    _transform = graph.fetch_surface_to_surface_transform(
        source=path[0],
        target=path[1],
        density=density,
        hemisphere=hemisphere,
        resource_type=resource_type,
    )

    for i in range(2, len(path)):
        next_target = path[i]

        hop_output_file = _get_hop_output_file(
            output_file_path, source, next_target, density, hemisphere
        )

        _transform = _two_hops(
            graph=graph,
            source_space=source,
            mid_space=path[i - 1],
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
    """Perform a two-hop surface-to-surface transformation via an intermediate space."""
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

    surface_atlas = graph.fetch_surface_atlas(
        space=mid_space,
        hemisphere=hemisphere,
        density=density,
        resource_type="sphere",
    )
    if surface_atlas is None:
        raise ValueError(f"No surface atlas found for {mid_space}")

    sphere_project_to = surface_atlas.fetch()

    unproject_transform = graph.fetch_surface_to_surface_transform(
        source=mid_space,
        target=target_space,
        density=density,
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
