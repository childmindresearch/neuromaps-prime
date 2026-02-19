"""Module for managing brain template surface spaces and their transformations.

Graph structure:
  - Nodes: Brain template spaces
  - Edges: Transformations between spaces

See examples/example_graph_init.py for usage.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import yaml
from niwrap import Runner, workbench

from neuromaps_prime.models import (
    Edge,
    Node,
    SurfaceAtlas,
    SurfaceTransform,
    VolumeAtlas,
    VolumeTransform,
)
from neuromaps_prime.transforms.surface import (
    label_resample,
    metric_resample,
    surface_sphere_project_unproject,
)
from neuromaps_prime.transforms.utils import _get_density_key, estimate_surface_density
from neuromaps_prime.transforms.volume import surface_project, vol_to_vol
from neuromaps_prime.utils import set_runner


class NeuromapsGraph(nx.MultiDiGraph):
    """Multi-directed graph of brain template spaces and their transformations."""

    surface_to_surface_key = "surface_to_surface"
    volume_to_volume_key = "volume_to_volume"

    def __init__(
        self,
        runner: Runner | Literal["local", "docker", "singularity"] = "local",
        runner_kwargs: dict[str, Any] = {},
        yaml_file: Path | None = None,
        data_dir: Path | None = None,
        _testing: bool = False,
    ) -> None:
        """Initialize and populate NeuromapsGraph."""
        super().__init__()

        # Init graph
        self.data_dir = data_dir or os.getenv("NEUROMAPS_DATA", None)
        self.yaml_path = (
            yaml_file
            or Path(__file__).parent / "datasets" / "data" / "neuromaps_graph.yaml"
        )
        self.runner = set_runner(runner=runner, **runner_kwargs)

        # Build graph with internal cache
        self._surface_atlas_cache: dict[tuple[str, ...], SurfaceAtlas] = {}
        self._surface_transform_cache: dict[tuple[str, ...], SurfaceTransform] = {}
        self._volume_atlas_cache: dict[tuple[str, ...], VolumeAtlas] = {}
        self._volume_transform_cache: dict[tuple[str, ...], VolumeTransform] = {}

        # Flag to indicate testing
        if not _testing:
            self._build_from_yaml(self.yaml_path)

    # GRAPH BUILDING
    def _build_from_yaml(self, yaml_file: Path) -> None:
        """Populate graph from YAML file."""
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
        self._build_from_dict(data)

    def _build_from_dict(self, data: dict[str, Any]) -> None:
        """Populate graph from dictionary."""
        # Nodes
        for node in data.get("nodes", {}):
            ((node_name, node_data),) = node.items()
            node_obj = Node(
                name=node_name,
                species=node_data.get("species", ""),
                description=(desc := node_data.get("description", "")),
                surfaces=(
                    surfaces := self._parse_surfaces(
                        node_name, desc, node_data.get("surfaces", {})
                    )
                ),
                volumes=(
                    volumes := self._parse_volumes(
                        node_name, desc, node_data.get("volumes", {})
                    )
                ),
            )
            self.add_node(node_name, data=node_obj)
            # Update atlas caches
            for surf_atlas in surfaces:
                self._surface_atlas_cache[
                    (
                        node_name,
                        surf_atlas.density,
                        surf_atlas.hemisphere.lower(),
                        surf_atlas.resource_type,
                    )
                ] = surf_atlas
            for vol_atlas in volumes:
                self._volume_atlas_cache[
                    (node_name, vol_atlas.resolution, vol_atlas.resource_type)
                ] = vol_atlas

        # Edges
        for key, edge_list in (
            (
                self.surface_to_surface_key,
                data.get("edges", {}).get(self.surface_to_surface_key, []),
            ),
            (
                self.volume_to_volume_key,
                data.get("edges", {}).get(self.volume_to_volume_key, []),
            ),
        ):
            for edge_data in edge_list:
                source, target = edge_data.get("from"), edge_data.get("to")
                if key == self.surface_to_surface_key:
                    surf_transforms = self._parse_surface_to_surface_transforms(
                        source, target, edge_data.get("surfaces", {})
                    )
                    edge_obj = Edge(surface_transforms=surf_transforms)
                    for surf_xfm in surf_transforms:
                        self._surface_transform_cache[
                            (
                                source,
                                target,
                                surf_xfm.density,
                                surf_xfm.hemisphere.lower(),
                                surf_xfm.resource_type,
                            )
                        ] = surf_xfm
                else:
                    vol_transforms = self._parse_volume_to_volume_transforms(
                        source, target, edge_data.get("volumes", {})
                    )
                    edge_obj = Edge(volume_transforms=vol_transforms)
                    for vol_xfm in vol_transforms:
                        self._volume_transform_cache[
                            (source, target, vol_xfm.resolution, vol_xfm.resource_type)
                        ] = vol_xfm
                self.add_edge(source, target, key=key, data=edge_obj, weight=1.0)

    # PARSER HELPERS
    def _parse_surfaces(
        self, node_name: str, description: str, surfaces_dict: dict[str, Any]
    ) -> list[SurfaceAtlas]:
        """Parse surface atlas dictionary into SurfaceAtlas objects."""
        return [
            SurfaceAtlas(
                name=f"{node_name}_{density}_{hemi}_{surf_type}",
                description=description,
                file_path=(self.data_dir / path) if self.data_dir else Path(path),
                space=node_name,
                density=density,
                hemisphere=hemi,
                resource_type=surf_type,
            )
            for density, types in surfaces_dict.items()
            for surf_type, hemispheres in types.items()
            for hemi, path in hemispheres.items()
        ]

    def _parse_volumes(
        self, node_name: str, description: str, volumes_dict: dict[str, Any]
    ) -> list[VolumeAtlas]:
        """Parse volume atlas dictionary into VolumeAtlas objects."""
        return [
            VolumeAtlas(
                name=f"{node_name}_{res}_{vol_type}",
                description=description,
                file_path=(self.data_dir / path) if self.data_dir else Path(path),
                space=node_name,
                resolution=res,
                resource_type=vol_type,
            )
            for res, types in volumes_dict.items()
            for vol_type, path in types.items()
        ]

    def _parse_volume_to_volume_transforms(
        self, source_name: str, target_name: str, volumes_dict: dict[str, Any]
    ) -> list[VolumeTransform]:
        """Parse volume-to-volume dictionary into VolumeTransform objects."""
        return [
            VolumeTransform(
                name=f"{source_name}_to_{target_name}_{res}_{vol_type}",
                description=f"vol2vol transform from {source_name} to {target_name}",
                file_path=(self.data_dir / path) if self.data_dir else Path(path),
                source_space=source_name,
                target_space=target_name,
                resolution=res,
                resource_type=vol_type,
            )
            for res, types in volumes_dict.items()
            for vol_type, path in types.items()
        ]

    def _parse_surface_to_surface_transforms(
        self, source_name: str, target_name: str, surfaces_dict: dict
    ) -> list[SurfaceTransform]:
        """Parse surface-to-surface dictionary into SurfaceTransform objects."""
        return [
            SurfaceTransform(
                name=f"{source_name}_to_{target_name}_{density}_{hemi}_{surf_type}",
                description=f"surf2surf transform from {source_name} to {target_name}",
                file_path=(self.data_dir / path) if self.data_dir else Path(path),
                source_space=source_name,
                target_space=target_name,
                density=density,
                hemisphere=hemi,
                resource_type=surf_type,
            )
            for density, types in surfaces_dict.items()
            for surf_type, hemispheres in types.items()
            for hemi, path in hemispheres.items()
        ]

    # TRANSFORM UTILITIES
    def _get_hop_output_file(
        self,
        output_file_path: str,
        source: str,
        next_target: str,
        density: str,
        hemisphere: Literal["left", "right"],
    ) -> str:
        """Generate hop output file path based on parameters."""
        parent = Path(output_file_path).parent
        fname = (
            f"src-{source}_"
            f"to-{next_target}_den-{density}_hemi-{hemisphere}_sphere.surf.gii"
        )
        return f"{parent}/{fname}"

    def _surface_to_surface(
        self,
        source: str,
        target: str,
        density: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        add_edge: bool = True,
    ) -> SurfaceTransform | None:
        """Perform a surface-to-surface transformation from source to target space.

        Args:
            source: The source space name.
            target: The target space name.
            density: The density of the surfaces.
            hemisphere: The hemisphere ('left' or 'right').
            output_file_path: Path to save the resulting transform.
            add_edge: Flag to add resulting transform to graph (default: True).

        Returns:
            The resulting surface-to-surface transform resource.

        Raises:
        ------
        ValueError: If no valid path found, hemisphere is invalid, or source and target
            are the same.
        """
        if source == target:
            raise ValueError(f"Source and target spaces are the same: {source}")
        self.validate(source, target)

        s_path = self.find_path(
            source=source, target=target, edge_type=self.surface_to_surface_key
        )
        if len(s_path) < 2:
            raise ValueError(f"No valid path from {source} to {target}")
        # Single hop
        if len(s_path) == 2:
            return self.fetch_surface_to_surface_transform(
                source=source,
                target=target,
                density=density,
                hemisphere=hemisphere,
                resource_type="sphere",
            )
        # Multi-hop
        return self._compose_multihop_surface_transform(
            paths=s_path,
            source=source,
            density=density,
            hemisphere=hemisphere,
            output_file_path=output_file_path,
            add_edge=add_edge,
        )

    def _compose_multihop_surface_transform(
        self,
        paths: list[str],
        source: str,
        density: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        add_edge: bool,
    ) -> SurfaceTransform:
        """Compose multiple surface transforms along a path.

        Args:
            paths: List of space names from source to target.
            source: Original source space (for naming).
            target: Final target space (for naming).
            density: Surface density.
            hemisphere: Hemisphere identifier.
            output_file_path: Base path for output files.
            add_edge: Whether to add composed transforms to graph.

        Returns:
            Final composed transform from source to target.

        Raises:
            ValueError: If any intermediate transform cannot be fetched.
        """
        current_transform = self.fetch_surface_to_surface_transform(
            source=paths[0],
            target=paths[1],
            density=density,
            hemisphere=hemisphere,
            resource_type="sphere",
        )
        if current_transform is None:
            raise ValueError(f"No transform found from {paths[0]} -> {paths[1]}")

        for hop_idx, next_space in enumerate(paths[2:], start=2):  # type: ignore[call-overload]
            current_transform = self._compose_next_hop(
                paths=paths,
                hop_idx=hop_idx,
                next_space=next_space,
                current_transform=current_transform,
                source=source,
                density=density,
                hemisphere=hemisphere,
                output_file_path=output_file_path,
                add_edge=add_edge,
            )

        return current_transform

    def _compose_next_hop(
        self,
        paths: list[str],
        hop_idx: int,
        next_space: str,
        current_transform: SurfaceTransform,
        source: str,
        density: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        add_edge: bool,
    ) -> SurfaceTransform:
        """Compose current transform with next hop in path.

        Args:
            paths: Full transformation path.
            hop_idx: Current hop index (2-based).
            next_space: Next target space.
            current_transform: Transform accumulated so far.
            source: Original source space.
            density: Surface density.
            hemisphere: Hemisphere identifier.
            output_file_path: Base output path.
            add_edge: Whether to add transform to graph.

        Returns:
            New composed transform.
        """
        hop_output = self._get_hop_output_file(
            output_file_path=str(output_file_path),
            source=source,
            next_target=next_space,
            density=density,
            hemisphere=hemisphere,
        )
        composed_path = self._two_hops(
            source_space=paths[hop_idx - 2],
            mid_space=paths[hop_idx - 1],
            target_space=next_space,
            density=density,
            hemisphere=hemisphere,
            output_file_path=hop_output,
            first_transform=current_transform,
        )
        new_transform = SurfaceTransform(
            name=f"{source}_to_{next_space}_{density}_{hemisphere}_sphere",
            description=f"Surface Transform from {source} to {next_space}",
            source_space=source,
            target_space=next_space,
            density=density,
            hemisphere=hemisphere,
            resource_type="sphere",
            file_path=composed_path,
            weight=float(hop_idx),
        )
        if add_edge:
            self.add_transform(transform=new_transform, key=self.surface_to_surface_key)

        return new_transform

    def _two_hops(
        self,
        source_space: str,
        mid_space: str,
        target_space: str,
        density: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        first_transform: SurfaceTransform | None = None,
    ) -> Path:
        """Perform a two-hop surface-to-surface transformation via intermediate space.

        Args:
            source_space: The source space name.
            mid_space: The intermediate space name.
            target_space: The target space name.
            density: The density of the surfaces.
            hemisphere: The hemisphere ('left' or 'right').
            output_file_path: Path to the output GIFTI surface file.
            first_transform: Pre-fetched transform from source to mid space. If None,
                it will be fetched

        Returns:
            Object containing the path to the output spherical surface as
            result.sphere_out.

        Raises:
            ValueError: If no surface transform is found for the source to mid space
                or mid to target space.
            FileNotFoundError: If any input file does not exist.
        """
        first_transform = first_transform or self.fetch_surface_to_surface_transform(
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

        highest_common_density = self.find_common_density(mid_space, target_space)
        surface_atlas = self.fetch_surface_atlas(
            space=mid_space,
            hemisphere=hemisphere,
            density=highest_common_density,
            resource_type="sphere",
        )
        if surface_atlas is None:
            raise ValueError(f"No surface atlas found for {mid_space}")
        sphere_project_to = surface_atlas.fetch()

        unproject_transform = self.fetch_surface_to_surface_transform(
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

        return surface_sphere_project_unproject(
            sphere_in=sphere_in,
            sphere_project_to=sphere_project_to,
            sphere_unproject_from=sphere_unproject_from,
            sphere_out=output_file_path,
        ).sphere_out

    # RESOURCE FETCHING
    def fetch_surface_atlas(
        self,
        space: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
    ) -> SurfaceAtlas | None:
        """Fetch a surface atlas resource from the graph.

        Args:
            space: The brain template space name.
            density: The surface mesh density (e.g., '32k', '41k').
            hemisphere: The hemisphere ('left', 'right').
            resource_type: The type of surface resource (e.g., 'midthickness', 'white').

        Returns:
            The matching SurfaceAtlas resource, or None if not found.
        """
        return self._surface_atlas_cache.get(
            (space, density, hemisphere.lower(), resource_type)
        )

    def fetch_volume_atlas(
        self, space: str, resolution: str, resource_type: str
    ) -> VolumeAtlas | None:
        """Fetch a volume atlas resource from the graph.

        Args:
            space: The brain template space name.
            resolution: The volume resolution (e.g., '2mm', '1mm').
            resource_type: The type of volume resource (e.g., 'T1w', 'T2w').

        Returns:
            The matching VolumeAtlas resource, or None if not found.
        """
        return self._volume_atlas_cache.get((space, resolution, resource_type))

    def fetch_surface_to_surface_transform(
        self,
        source: str,
        target: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
    ) -> SurfaceTransform | None:
        """Fetch a surface-to-surface transform resource from the graph.

        Args:
            source: The source brain template space name.
            target: The target brain template space name.
            density: The surface mesh density (e.g., '32k', '41k').
            hemisphere: The hemisphere ('left', 'right').
            resource_type: The type of surface resource (e.g., 'midthickness', 'white').
            key: The key identifying the edge type.

        Returns:
            SurfaceTransform | None:
                The matching SurfaceTransform resource, or None if not found.
        """
        return self._surface_transform_cache.get(
            (source, target, density, hemisphere.lower(), resource_type)
        )

    def fetch_volume_to_volume_transform(
        self, source: str, target: str, resolution: str, resource_type: str
    ) -> VolumeTransform | None:
        """Fetch a volume-to-volume transform resource from the graph.

        Args:
            source: The source brain template space name.
            target: The target brain template space name.
            resolution: The volume resolution (e.g., '2mm', '1mm').
            resource_type: The type of volume resource (e.g., 'T1w', 'T2w').

        Returns:
            The matching VolumeTransform resource, or None if not found.
        """
        return self._volume_transform_cache.get(
            (source, target, resolution, resource_type)
        )

    # NODE UTILITIES
    def validate(self, source: str, target: str) -> None:
        """Validate that source and target spaces exist in the graph."""
        if source not in self.nodes(data=False):
            raise ValueError(
                f"source space '{source}' does not exist in the graph."
                f" Available spaces: {list(self.nodes(data=False))}"
            )
        if target not in self.nodes(data=False):
            raise ValueError(
                f"target space '{target}' does not exist in the graph."
                f" Available spaces: {list(self.nodes(data=False))}"
            )

    def find_path(
        self, source: str, target: str, edge_type: str | None = None
    ) -> list[str]:
        """Find a path between two brain template spaces in the graph.

        Args:
            source: The source brain template space name.
            target: The target brain template space name.
            edge_type: Type of edges to use ('surface_to_surface', 'volume_to_volume').

        Returns:
            A list of space names representing the path from source to target.
        """
        try:
            graph = self.get_subgraph(edges=edge_type) if edge_type else self
            return nx.shortest_path(
                graph, source=source, target=target, weight="weight"
            )
        except nx.NetworkXNoPath:
            return []

    @lru_cache
    def get_subgraph(self, edges: str) -> nx.MultiDiGraph:
        """Get a subgraph containing all nodes but only the specified edges keys.

        Args:
            edges: Type of edges to include ('surface_to_surface', 'volume_to_volume').

        Returns:
            The resulting subgraph.
        """
        subgraph = nx.MultiDiGraph()
        subgraph.add_nodes_from(self.nodes(data=True))
        for u, v, key, edge_data in self.edges(data=True, keys=True):
            if key == edges:
                subgraph.add_edge(u, v, key=key, **edge_data)
        return subgraph

    def get_graph_info(self) -> dict[str, Any]:
        """Get a summary of the graph structure.

        Returns:
            A dictionary summary of the graph structure.
        """
        nodes_data = [self.get_node_data(n) for n in self.nodes]
        return {
            "num_nodes": self.number_of_nodes(),
            "num_edges": self.number_of_edges(),
            "num_surfaces": sum(len(n.surfaces) for n in nodes_data),
            "num_volumes": sum(len(n.volumes) for n in nodes_data),
            "num_surface_to_surface_transforms": sum(
                1 for _, _, k in self.edges(keys=True) if k == "surface_to_surface"
            ),
            "num_volume_to_volume_transforms": sum(
                1 for _, _, k in self.edges(keys=True) if k == "volume_to_volume"
            ),
        }

    def get_node_data(self, node_name: str) -> Node:
        """Get the Node data for a given node name.

        Args:
            node_name: The name of the node.

        Returns:
            The Node data object.

        Raises:
            ValueError: if node not found in graph.
        """
        try:
            return self.nodes[node_name]["data"]
        except KeyError:
            raise ValueError(
                f"Node '{node_name}' not found.\nAvailable nodes: {list(self.nodes)}"
            )

    def add_transform(
        self,
        transform: SurfaceTransform | VolumeTransform,
        key: str,
    ) -> None:
        """Add a transform as an edge in the graph."""
        match transform:
            case SurfaceTransform():
                edge = Edge(
                    surface_transforms=[transform],
                    volume_transforms=[],
                )
                self._surface_transform_cache[
                    (
                        transform.source_space,
                        transform.target_space,
                        transform.density,
                        transform.hemisphere.lower(),
                        transform.resource_type,
                    )
                ] = transform
            case VolumeTransform():
                edge = Edge(
                    surface_transforms=[],
                    volume_transforms=[transform],
                )
                self._volume_transform_cache[
                    (
                        transform.source_space,
                        transform.target_space,
                        transform.resolution,
                        transform.resource_type,
                    )
                ] = transform
            case _:
                raise TypeError(f"Unsupported transform type: {type(transform)}")
        self.add_edge(
            transform.source_space,
            transform.target_space,
            key=key,
            data=edge,
            weight=transform.weight,
        )

    def search_surface_atlases(
        self,
        space: str,
        density: str | None = None,
        hemisphere: Literal["left", "right"] | None = None,
        resource_type: str | None = None,
    ) -> list[SurfaceAtlas]:
        """Search for surface atlases matching the given criteria.

        Args:
            space: The brain template space name.
            density: The surface mesh density to match.
            hemisphere: The hemisphere to match.
            resource_type: The resource type to match.

        Returns:
            A list of matching surface atlases.
        """
        node = self.nodes.get(space)
        if not node:
            return []
        return [
            atlas
            for atlas in node["data"].surfaces
            if (density is None or atlas.density == density)
            and (hemisphere is None or atlas.hemisphere.lower() == hemisphere.lower())
            and (resource_type is None or atlas.resource_type == resource_type)
        ]

    def search_surface_transforms(
        self,
        source_space: str,
        target_space: str,
        density: str | None = None,
        hemisphere: Literal["left", "right"] | None = None,
        resource_type: str | None = None,
    ) -> list[SurfaceTransform]:
        """Search for surface transforms matching the given criteria.

        Args:
            source_space: The source brain template space name.
            target_space: The target brain template space name.
            density: The surface mesh density to match.
            hemisphere: The hemisphere to match.
            resource_type: The resource type to match.

        Returns:
            A list of matching surface transforms.
        """
        matching_transforms = []
        for u, v, key, edge_data in self.edges(data=True, keys=True):
            if u != source_space or v != target_space or key != "surface_to_surface":
                continue
            edge: Edge = edge_data["data"]
            for transform in edge.surface_transforms:
                if (
                    (density is None or transform.density == density)
                    and (
                        hemisphere is None
                        or transform.hemisphere.lower() == hemisphere.lower()
                    )
                    and (
                        resource_type is None
                        or transform.resource_type == resource_type
                    )
                ):
                    matching_transforms.append(transform)
        return matching_transforms

    def find_common_density(self, mid_space: str, target_space: str) -> str:
        """Find a common density between source and source-to-target transformations.

        Args:
            mid_space: Intermediate space to transform between
            target_space: Final space to transform to

        Returns:
            The maximum density found between the mid and target spaces.

        Raises:
            ValueError: If no common density found.
        """
        atlases = self.search_surface_atlases(space=mid_space)
        transforms = self.search_surface_transforms(
            source_space=mid_space, target_space=target_space
        )
        common_densities = {atlas.density for atlas in atlases} & {
            transform.density for transform in transforms
        }
        if not common_densities:
            raise ValueError(
                f"No common density found between {mid_space} and {target_space}"
            )
        return max(common_densities, key=_get_density_key)

    def find_highest_density(self, space: str) -> str:
        """Find the highest density available for a given space in the graph."""
        atlases = self.search_surface_atlases(space=space)
        densities = {atlas.density for atlas in atlases}

        if not densities:
            raise ValueError(f"No atlases found for space '{space}'.")

        highest_density = max(densities, key=_get_density_key)
        return highest_density

    def surface_to_surface_transformer(
        self,
        transformer_type: Literal["metric", "label"],
        input_file: Path,
        source_space: str,
        target_space: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        source_density: str | None = None,
        target_density: str | None = None,
        area_resource: str = "midthickness",
        add_edge: bool = True,
    ) -> Path | None:
        """Perform a surface-to-surface transformation (metric or label) with caching.

        Args:
            transformer_type: 'metric' or 'label'.
            input_file: Input GIFTI file (metric or label).
            source_space: Source brain template space.
            target_space: Target brain template space.
            hemisphere: Hemisphere ('left' or 'right').
            output_file_path: Output GIFTI file path.
            source_density: Source surface density. If None, estimated from input.
            target_density: Target surface density. If None, highest available used.
            area_resource: Surface type for area-based resampling.
            add_edge: Whether to add the resulting transform as an edge in the graph.

        Returns:
            Resampled metric or label output, or None if transformation fails.

        Raises:
            ValueError: Invalid transformer type or missing surface resources.
            FileNotFoundError: If input or output files cannot be accessed.
        """
        if transformer_type not in ("metric", "label"):
            raise ValueError(
                f"Invalid transformer_type: {transformer_type}. "
                "Must be 'metric' or 'label'."
            )
        source_density = source_density or estimate_surface_density(input_file)

        transform = self._surface_to_surface(
            source=source_space,
            target=target_space,
            density=source_density,
            hemisphere=hemisphere,
            output_file_path=output_file_path,
            add_edge=add_edge,
        )
        if transform is None:
            return None

        target_density = target_density or self.find_highest_density(space=target_space)
        new_sphere_atlas = self.fetch_surface_atlas(
            space=target_space,
            hemisphere=hemisphere,
            density=target_density,
            resource_type="sphere",
        )
        if new_sphere_atlas is None:
            raise ValueError(f"No surface atlas found for {target_space}")
        new_sphere = new_sphere_atlas.fetch()

        current_area_atlas = self.fetch_surface_atlas(
            space=source_space,
            hemisphere=hemisphere,
            density=source_density,
            resource_type=area_resource,
        )
        if current_area_atlas is None:
            raise ValueError(f"No {area_resource} surface found for {source_space}")
        current_area: Path = current_area_atlas.fetch()

        new_area_atlas = self.fetch_surface_atlas(
            space=target_space,
            hemisphere=hemisphere,
            density=target_density,
            resource_type=area_resource,
        )
        if new_area_atlas is None:
            raise ValueError(f"No {area_resource} surface found for {target_space}")
        new_area: Path = new_area_atlas.fetch()

        resample_fn = {"label": label_resample, "metric": metric_resample}
        area_surfs = {"current-area": current_area, "new-area": new_area}
        resampled_output = resample_fn[transformer_type](
            input_file_path=input_file,
            current_sphere=transform.fetch(),
            new_sphere=new_sphere,
            method="ADAP_BARY_AREA",
            area_surfs=area_surfs,
            output_file_path=output_file_path,
        )
        return (
            resampled_output.label_out
            if isinstance(resampled_output, workbench.LabelResampleOutputs)
            else resampled_output.metric_out
        )

    def volume_to_volume_transformer(
        self,
        input_file: Path,
        source_space: str,
        target_space: str,
        resolution: str,
        resource_type: str,
        output_file_path: str,
        interp: str = "linear",
        interp_params: dict[str, Any] | None = None,
    ) -> Path:
        """Perform a volume-to-volume transformation.

        Args:
            input_file: File in source space to transform.
            source_space: Source template space.
            target_space: Target template space.
            resolution: Volume resolution of target (reference) image (e.g., '500um').
            resource_type: Volume type (e.g., 'T1w', 'composite').
            output_file_path: Output file path.
            interp: Interpolation method.
            interp_params: Optional interpolation parameters.

        Returns:
            Path to the transformed volume.

        Raises:
            ValueError: If required resources are missing.
            FileNotFoundError: If input file does not exist.
        """
        self.validate(source_space, target_space)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        transform = self.fetch_volume_to_volume_transform(
            source=source_space,
            target=target_space,
            resolution=resolution,
            resource_type=resource_type,
        )
        if transform is None:
            raise ValueError(
                f"No volume transform found from {source_space} to {target_space} "
                f"(res={resolution}, type={resource_type})"
            )

        target_atlas = self.fetch_volume_atlas(
            space=target_space,
            resolution=resolution,
            resource_type=resource_type,
        )
        if target_atlas is None:
            raise ValueError(
                f"No target volume atlas found for {target_space} "
                f"(res={resolution}, type={resource_type})"
            )

        return vol_to_vol(
            source=input_file,
            target=target_atlas.fetch(),
            out_fpath=output_file_path,
            interp=interp,
            interp_params=interp_params,
        )

    def volume_to_surface_transformer(
        self,
        transformer_type: Literal["metric", "label"],
        input_file: Path,
        source_space: str,
        target_space: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        source_density: str | None = None,
        target_density: str | None = None,
        area_resource: str = "midthickness",
        add_edge: bool = True,
    ) -> Path | None:
        """Perform a surface-to-surface transformation (metric or label) with caching.

        Two stage transformation:
          1. Project volume-to-surface
          2. Surface-to-surface transformation

        Args:
            transformer_type: 'metric' or 'label'.
            input_file: Input GIFTI file (metric or label).
            source_space: Source brain template space.
            target_space: Target brain template space.
            hemisphere: Hemisphere ('left' or 'right').
            output_file_path: Output GIFTI file path.
            source_density: Source surface density. If None, highest available used
            target_density: Target surface density. If None, highest available used.
            area_resource: Surface type for area-based resampling.
            add_edge: Whether to add the resulting transform as an edge in the graph.

        Returns:
            Resampled metric or label output, or None if transformation fails.

        Raises:
            ValueError: Invalid transformer type or missing surface resources.
            FileNotFoundError: If input or output files cannot be accessed.
        """
        self.validate(source_space, target_space)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # fetch source atlas to project (remember if none used, use highest density)
        source_density = source_density or self.find_highest_density(space=source_space)
        source_surface_atlas = self.fetch_surface_atlas(
            space=source_space,
            density=source_density,
            hemisphere=hemisphere,
            resource_type=area_resource,
        )
        if source_surface_atlas is None:
            raise ValueError(f"No {area_resource} surface found for {source_space}")
        source_surface = source_surface_atlas.fetch()

        ribbon = {}
        for surf_type in ("white", "pial"):
            ribbon[surf_type] = self.fetch_surface_atlas(
                space=source_space,
                density=source_density,
                hemisphere=hemisphere,
                resource_type=surf_type,
            )
            if ribbon[surf_type] is None:
                raise ValueError(f"No {surf_type} surface found for {source_space}")
        ribbon_surfs = workbench.volume_to_surface_mapping_ribbon_constrained(
            inner_surf=ribbon["white"].fetch(),  # type: ignore[union-attr]
            outer_surf=ribbon["pial"].fetch(),  # type: ignore[union-attr]
        )
        ext = "func" if transformer_type == "metric" else "label"
        projected_vol_surface = surface_project(
            volume=input_file,
            surface=source_surface,
            ribbon_surfs=ribbon_surfs,
            out_fpath=f"src-{source_space}_den-{source_density}_hemi-{hemisphere}_desc-volume_annot.{ext}.gii",
        )

        return self.surface_to_surface_transformer(
            transformer_type=transformer_type,
            input_file=projected_vol_surface,
            source_space=source_space,
            target_space=target_space,
            hemisphere=hemisphere,
            output_file_path=output_file_path,
            source_density=source_density,
            target_density=target_density,
            area_resource=area_resource,
            add_edge=add_edge,
        )
