"""Module for managing brain template surface spaces and their transformations.

The graph structure includes nodes representing brain template spaces
and edges representing transformations between these spaces.

nodes:
    - name: Name of the brain template space (e.g., 'fsaverage', 'MEBRAINS')
    - species: Species associated with the brain template space (e.g., 'human')
    - description: Description of the brain template space
    - surfaces: List of surface atlas resources available in this space
    - volumes: List of volume atlas resources available in this space

edges:
    - surface_to_surface: Transformations between surface spaces
    - volume_to_volume: Transformations between volume spaces

see examples/example_graph_init.py for usage.

## Make this more detailed later. ##

"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import yaml
from niwrap import workbench

from neuromaps_prime.models import (
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
from neuromaps_prime.transforms.utils import (
    _get_density_key,
    estimate_surface_density,
)


@dataclass
class Node:
    """Class representing a node in the surface graph."""

    name: str
    species: str
    description: str
    surfaces: list[SurfaceAtlas]
    volumes: list[VolumeAtlas]

    def __repr__(self) -> str:
        """Custom string representation for debugging."""
        surface_str = "\n".join(f"{s.name}" for s in self.surfaces)
        volume_str = "\n".join(f"{v.name}" for v in self.volumes)
        return (
            f"\nNode :\n"
            f"name={self.name},\n"
            f"species={self.species},\n"
            f"surfaces=[{surface_str}],\n"
            f"volumes=[{volume_str}]"
        )


@dataclass
class Edge:
    """Class representing an edge in the surface graph."""

    surface_transforms: list[SurfaceTransform]
    volume_transforms: list[VolumeTransform]

    def __repr__(self) -> str:
        """Custom string representation for debugging."""
        surface_str = "\n".join(f"{s.name}" for s in self.surface_transforms)
        volume_str = "\n".join(f"{v.name}" for v in self.volume_transforms)
        return f"\nEdge : \nsurfaces=[{surface_str}], \nvolumes=[{volume_str}])"


class NeuromapsGraph(nx.MultiDiGraph):
    """Multi-directed graph of brain template spaces and their transformations."""

    def __init__(
        self, yaml_file: Path | None = None, data_dir: Path | None = None
    ) -> None:
        """Initialize an empty NeuromapsGraph and populate it from a YAML file."""
        super().__init__()

        self.data_dir = data_dir or os.getenv("NEUROMAPS_DATA", None)
        self.yaml_path = (
            yaml_file
            or Path(__file__).parent / "datasets" / "data" / "neuromaps_graph.yaml"
        )
        self._build_from_yaml(self.yaml_path)

    def _build_from_yaml(self, yaml_file: Path) -> None:
        """Read in the YAML file and call _build_from_dict to populate the graph."""
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)

        self._build_from_dict(data)

    def _build_from_dict(self, data: dict[str, Any]) -> None:
        """Build the graph from a dictionary."""
        nodes = data.get("nodes", {})
        edges = data.get("edges", {})

        for node in nodes:
            ((node_name, _node_data),) = node.items()
            species = _node_data.get("species", "")
            description = _node_data.get("description", "")

            surfaces = self._parse_surfaces(
                node_name, description, _node_data.get("surfaces", {})
            )
            volumes = self._parse_volumes(
                node_name, description, _node_data.get("volumes", {})
            )

            neuromaps_node = Node(
                name=node_name,
                species=species,
                description=description,
                surfaces=surfaces,
                volumes=volumes,
            )
            self.add_node(node_name, data=neuromaps_node)

        surface_to_surface_transforms = edges.get("surface_to_surface", [])
        for transforms in surface_to_surface_transforms:
            source = transforms.get("from")
            target = transforms.get("to")
            surfaces_dict = transforms.get("surfaces", {})

            surface_transforms = self._parse_surface_to_surface_transforms(
                source, target, surfaces_dict
            )

            neuromaps_edge = Edge(
                surface_transforms=surface_transforms,
                volume_transforms=[],
            )
            self.add_edge(source, target, key="surface_to_surface", data=neuromaps_edge)

        volume_to_volume_transforms = edges.get("volume_to_volume", [])
        for transforms in volume_to_volume_transforms:
            source = transforms.get("from")
            target = transforms.get("to")
            volumes_dict = transforms.get("volumes", {})

            volume_transforms = self._parse_volume_to_volume_transforms(
                source, target, volumes_dict
            )

            neuromaps_edge = Edge(
                surface_transforms=[],
                volume_transforms=volume_transforms,
            )
            self.add_edge(source, target, key="volume_to_volume", data=neuromaps_edge)

    def _parse_surfaces(
        self, node_name: str, description: str, surfaces_dict: dict
    ) -> list[SurfaceAtlas]:
        SurfaceAtlasList = []
        for density, surface_types in surfaces_dict.items():
            for surface_type, hemispheres in surface_types.items():
                for hemisphere, path in hemispheres.items():
                    SurfaceAtlasList.append(
                        SurfaceAtlas(
                            name=f"{node_name}_{density}_{hemisphere}_{surface_type}",
                            description=description,
                            file_path=Path(path)
                            if self.data_dir is None
                            else self.data_dir / path,
                            space=node_name,
                            density=density,
                            hemisphere=hemisphere,
                            resource_type=surface_type,
                        )
                    )
        return SurfaceAtlasList

    def _parse_volumes(
        self, node_name: str, description: str, volumes_dict: dict
    ) -> list[VolumeAtlas]:
        VolumeAtlasList = []
        for resolution, volume_types in volumes_dict.items():
            for volume_type, path in volume_types.items():
                VolumeAtlasList.append(
                    VolumeAtlas(
                        name=f"{node_name}_{resolution}_{volume_type}",
                        description=description,
                        file_path=Path(path)
                        if self.data_dir is None
                        else self.data_dir / path,
                        space=node_name,
                        resolution=resolution,
                        resource_type=volume_type,
                    )
                )
        return VolumeAtlasList

    def _parse_volume_to_volume_transforms(
        self, source_name: str, target_name: str, volumes_dict: dict
    ) -> list[VolumeTransform]:
        VolumeTransformList = []
        for resolution, volume_types in volumes_dict.items():
            for volume_type, path in volume_types.items():
                VolumeTransformList.append(
                    VolumeTransform(
                        name=f"{source_name}_to_{target_name}_{resolution}_{volume_type}",
                        description=f"Transform from {source_name} to {target_name}",
                        file_path=Path(path)
                        if self.data_dir is None
                        else self.data_dir / path,
                        source_space=source_name,
                        target_space=target_name,
                        resolution=resolution,
                        resource_type=volume_type,
                    )
                )
        return VolumeTransformList

    def _parse_surface_to_surface_transforms(
        self, source_name: str, target_name: str, surfaces_dict: dict
    ) -> list[SurfaceTransform]:
        SurfaceTransformList = []
        for density, surface_types in surfaces_dict.items():
            for surface_type, hemispheres in surface_types.items():
                for hemisphere, path in hemispheres.items():
                    SurfaceTransformList.append(
                        SurfaceTransform(
                            name=(
                                f"{source_name}_to_{target_name}_{density}"
                                f"_{hemisphere}_{surface_type}"
                            ),
                            description=(
                                f"Transform from {source_name} to {target_name}"
                            ),
                            file_path=Path(path)
                            if self.data_dir is None
                            else self.data_dir / path,
                            source_space=source_name,
                            target_space=target_name,
                            density=density,
                            hemisphere=hemisphere,
                            resource_type=surface_type,
                        )
                    )
        return SurfaceTransformList

    def _get_hop_output_file(
        self,
        output_file_path: str,
        source: str,
        next_target: str,
        density: str,
        hemisphere: str,
    ) -> str:
        """Generate hop output file path based on parameters."""
        p = Path(output_file_path)
        parent = p.parent
        # Always use sphere.surf.gii for intermediate sphere files
        suffix = "sphere.surf.gii"
        return str(
            parent / f"src-{source}_"
            f"to-{next_target}_den-{density}_hemi-{hemisphere[0].upper()}_{suffix}"
        )

    def _surface_to_surface(
        self,
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
        source : str
            The source space name.
        target : str
            The target space name.
        density : str
            The density of the surfaces.
        hemisphere : str
            The hemisphere ('left' or 'right').
        add_edge : bool, optional
            Whether to add the resulting transform as an edge in the graph.
            Default is True.

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
        self.validate(source, target)
        shortest_path = self.find_path(
            source=source, target=target, edge_type="surface_to_surface"
        )
        resource_type = "sphere"

        if not shortest_path or len(shortest_path) < 1:
            raise ValueError(f"No valid path found from {source} to {target}.")

        elif len(shortest_path) == 1:
            raise ValueError(f"Source and target spaces are the same: {source}.")

        elif len(shortest_path) == 2:
            return self.fetch_surface_to_surface_transform(
                source=source,
                target=target,
                density=density,
                hemisphere=hemisphere,
                resource_type=resource_type,
            )

        _transform = self.fetch_surface_to_surface_transform(
            source=shortest_path[0],
            target=shortest_path[1],
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )

        for i in range(2, len(shortest_path)):
            next_target = shortest_path[i]

            hop_output_file = self._get_hop_output_file(
                output_file_path, source, next_target, density, hemisphere
            )

            _transform = self._two_hops(
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
                self.add_transform(
                    source_space=source,
                    target_space=next_target,
                    key="surface_to_surface",
                    surface_transform=transform,
                )

        return transform

    def validate(
        self,
        source: str,
        target: str,
    ) -> None:
        """Validate that source and target spaces exist in the graph."""
        if source not in self.nodes(data=False):
            raise ValueError(
                f"source space '{source}' does not exist in the graph."
                f" Available spaces: {list(self.nodes(data=False))}"
            )
        elif target not in self.nodes(data=False):
            raise ValueError(
                f"target space '{target}' does not exist in the graph."
                f" Available spaces: {list(self.nodes(data=False))}"
            )

    def _two_hops(
        self,
        source_space: str,
        mid_space: str,
        target_space: str,
        density: str,
        hemisphere: str,
        output_file_path: str,
        first_transform: SurfaceTransform | None = None,
    ) -> workbench.SurfaceSphereProjectUnprojectOutputs:
        """Perform a two-hop surface-to-surface transformation.

        via an intermediate space
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
            Object containing the path to the output spherical surface as
            result.sphere_out.

        Raises:
        ------
        ValueError
            If no surface transform is found
            for the source to mid space or mid to target space.
        FileNotFoundError
            If any input file does not exist.
        """
        if first_transform is None:
            first_transform = self.fetch_surface_to_surface_transform(
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

        resulting_transform = surface_sphere_project_unproject(
            sphere_in=sphere_in,
            sphere_project_to=sphere_project_to,
            sphere_unproject_from=sphere_unproject_from,
            sphere_out=output_file_path,
        )

        return resulting_transform

    ## Public Methods ##
    def fetch_surface_atlas(
        self,
        space: str,
        density: str,
        hemisphere: str,
        resource_type: str,
    ) -> SurfaceAtlas | None:
        """Fetch a surface atlas resource from the graph.

        Args:
            space (str): The brain template space name.
            density (str): The surface mesh density (e.g., '32k', '41k').
            hemisphere (str): The hemisphere ('L', 'R', 'left', 'right').
            resource_type (str): The type of surface resource
                (e.g., 'midthickness', 'white', 'pial').

        Returns:
            SurfaceAtlas | None:
                The matching SurfaceAtlas resource, or None if not found.
        """
        # Validate input parameters using the model
        try:
            # Create a temporary model to validate the input parameters
            validated_params = SurfaceAtlas(
                name="",
                description="",
                file_path=Path(""),
                space=space,
                density=density,
                hemisphere=hemisphere,
                resource_type=resource_type,
            )
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}")

        if space not in self.nodes:
            raise ValueError(
                f"Space '{space}' not found in the graph.\n"
                f"Available spaces: {list(self.nodes)}"
            )

        node: Node = self.nodes[space]["data"]

        for surface in node.surfaces:
            if (
                surface.density == validated_params.density
                and surface.hemisphere.lower() == validated_params.hemisphere.lower()
                and surface.resource_type == validated_params.resource_type
            ):
                return surface

        return None

    def fetch_volume_atlas(
        self,
        space: str,
        resolution: str,
        resource_type: str,
    ) -> VolumeAtlas | None:
        """Fetch a volume atlas resource from the graph.

        Args:
            space (str): The brain template space name.
            resolution (str): The volume resolution (e.g., '2mm', '1mm').
            resource_type (str): The type of volume resource
                (e.g., 'T1w', 'T2w', 'brain_mask').

        Returns:
            VolumeAtlas | None:
                The matching VolumeAtlas resource, or None if not found.
        """
        if space not in self.nodes:
            raise ValueError(
                f"Space '{space}' not found in the graph.\n"
                f"Available spaces: {list(self.nodes)}"
            )

        node: Node = self.nodes[space]["data"]

        for volume in node.volumes:
            if (
                volume.resolution == resolution
                and volume.resource_type == resource_type
            ):
                return volume

        return None

    def fetch_surface_to_surface_transform(
        self,
        source: str,
        target: str,
        density: str,
        hemisphere: str,
        resource_type: str,
    ) -> SurfaceTransform | None:
        """Fetch a surface-to-surface transform resource from the graph.

        Args:
            source (str): The source brain template space name.
            target (str): The target brain template space name.
            density (str): The surface mesh density (e.g., '32k', '41k').
            hemisphere (str): The hemisphere ('L', 'R', 'left', 'right').
            resource_type (str): The type of surface resource
                (e.g., 'midthickness', 'white', 'pial').

        Returns:
            SurfaceTransform | None:
                The matching SurfaceTransform resource, or None if not found.
        """
        if not self.has_edge(source, target, key="surface_to_surface"):
            raise ValueError(
                f"Surface-to-surface transform from "
                f"'{source}' to '{target}' not found in the graph.\n"
                f"Available edges: {list(self.edges)}"
            )

        edge_data = self.get_edge_data(source, target, key="surface_to_surface")
        edge: Edge = edge_data["data"]

        for transform in edge.surface_transforms:
            if (
                transform.density == density
                and transform.hemisphere.lower() == hemisphere.lower()
                and transform.resource_type == resource_type
            ):
                return transform
        return None

    def fetch_volume_to_volume_transform(
        self,
        source: str,
        target: str,
        resolution: str,
        resource_type: str,
    ) -> VolumeTransform | None:
        """Fetch a volume-to-volume transform resource from the graph.

        Args:
            source (str): The source brain template space name.
            target (str): The target brain template space name.
            resolution (str): The volume resolution (e.g., '2mm', '1mm').
            resource_type (str): The type of volume resource
                (e.g., 'T1w', 'T2w', 'brain_mask').

        Returns:
            VolumeTransform:
                The matching VolumeTransform resource, or None if not found.
        """
        if not self.has_edge(source, target, key="volume_to_volume"):
            raise ValueError(
                f"Volume-to-volume transform from "
                f"'{source}' to '{target}' not found in the graph.\n"
                f"Available edges: {list(self.edges)}"
            )

        edge_data = self.get_edge_data(source, target, key="volume_to_volume")
        edge: Edge = edge_data["data"]

        for transform in edge.volume_transforms:
            if (
                transform.resolution == resolution
                and transform.resource_type == resource_type
            ):
                return transform
        return None

    def find_path(
        self, source: str, target: str, edge_type: str | None = None
    ) -> list[str]:
        """Find a path between two brain template spaces in the graph.

        Args:
            source (str): The source brain template space name.
            target (str): The target brain template space name.
            edge_type (str): Type of edges to use
                ('surface_to_surface', 'volume_to_volume').

        Returns:
            List[str]:
                A list of space names representing the path from source to target.

                Returns empty list if no path exists.
        """
        try:
            if edge_type:
                temp_graph = self.get_subgraph(edges=edge_type)

                for u, v, key in self.edges(keys=True):
                    if key == edge_type:
                        temp_graph.add_edge(u, v)

                path = nx.shortest_path(temp_graph, source=source, target=target)
            else:
                path = nx.shortest_path(self, source=source, target=target)
            return path
        except nx.NetworkXNoPath:
            return []

    # Get a subgraph containing all nodes but only the specified edges keys
    def get_subgraph(self, edges: str) -> nx.MultiDiGraph:
        """Get a subgraph containing all nodes but only the specified edges keys.

        Args:
            edges (str): Type of edges to include
                ('surface_to_surface', 'volume_to_volume')

        Returns:
            nx.MultiDiGraph: The resulting subgraph.
        """
        subgraph = nx.MultiDiGraph()
        subgraph.add_nodes_from(self.nodes(data=True))

        for u, v, key in self.edges(keys=True):
            if key == edges:
                edge_data = self.get_edge_data(u, v, key=key)
                subgraph.add_edge(u, v, key=key, **edge_data)

        return subgraph

    def get_graph_info(self) -> dict[str, Any]:
        """Get a summary of the graph structure."""
        info = {
            "num_nodes": self.number_of_nodes(),
            "num_edges": self.number_of_edges(),
            "num_surfaces": sum(
                len(self.get_node_data(node).surfaces) for node in self.nodes
            ),
            "num_volumes": sum(
                len(self.get_node_data(node).volumes) for node in self.nodes
            ),
            "num_surface_to_surface_transforms": sum(
                1 for _, _, k in self.edges(keys=True) if k == "surface_to_surface"
            ),
            "num_volume_to_volume_transforms": sum(
                1 for _, _, k in self.edges(keys=True) if k == "volume_to_volume"
            ),
        }
        return info

    def get_node_data(self, node_name: str) -> Node:
        """Get the Node data for a given node name.

        Args:
            node_name (str): The name of the node.

        Returns:
            Node: The Node data object.
        """
        if node_name not in self.nodes:
            raise ValueError(
                f"Node '{node_name}' not found in the graph.\n"
                f"Available nodes: {list(self.nodes)}"
            )
        return self.nodes[node_name]["data"]

    def add_transform(
        self,
        source_space: str,
        target_space: str,
        key: str,
        surface_transform: SurfaceTransform | None = None,
        volume_transform: VolumeTransform | None = None,
    ) -> None:
        """Add a new surface transform edge to the graph."""
        edge = Edge(
            surface_transforms=[surface_transform] if surface_transform else [],
            volume_transforms=[volume_transform] if volume_transform else [],
        )
        self.add_edge(source_space, target_space, key=key, data=edge)

    def search_surface_atlases(
        self,
        space: str,
        density: str | None = None,
        hemisphere: str | None = None,
        resource_type: str | None = None,
    ) -> list[SurfaceAtlas]:
        """Search for surface atlases matching the given criteria.

        Args:
            space (str): The brain template space name.
            density (str | None): The surface mesh density to match.
            hemisphere (str | None): The hemisphere to match.
            resource_type (str | None): The resource type to match.

        Returns:
            list[SurfaceAtlas]: A list of matching surface atlases.
        """
        matching_atlases = []
        for node in self.nodes(data=True):
            node_name, node_data = node
            if node_name != space:
                continue
            node_obj: Node = node_data["data"]
            for atlas in node_obj.surfaces:
                if (
                    (density is None or atlas.density == density)
                    and (hemisphere is None or atlas.hemisphere == hemisphere)
                    and (resource_type is None or atlas.resource_type == resource_type)
                ):
                    matching_atlases.append(atlas)
        return matching_atlases

    def search_surface_transforms(
        self,
        source_space: str,
        target_space: str,
        density: str | None = None,
        hemisphere: str | None = None,
        resource_type: str | None = None,
    ) -> list[SurfaceTransform]:
        """Search for surface transforms matching the given criteria.

        Args:
            source_space (str): The source brain template space name.
            target_space (str): The target brain template space name.
            density (str | None): The surface mesh density to match.
            hemisphere (str | None): The hemisphere to match.
            resource_type (str | None): The resource type to match.

        Returns:
            list[SurfaceTransform]: A list of matching surface transforms.
        """
        matching_transforms = []
        for u, v, key, edge_data in self.edges(data=True, keys=True):
            if u != source_space or v != target_space or key != "surface_to_surface":
                continue
            edge: Edge = edge_data["data"]
            for transform in edge.surface_transforms:
                if (
                    (density is None or transform.density == density)
                    and (hemisphere is None or transform.hemisphere == hemisphere)
                    and (
                        resource_type is None
                        or transform.resource_type == resource_type
                    )
                ):
                    matching_transforms.append(transform)
        return matching_transforms

    def find_common_density(self, mid_space: str, target_space: str) -> str:
        """Find a common density between source and source-to-target transformations.

        This function is really needed to check if the surface_sphere_project_unproject
        function can be performed between two transformations via an intermediate space.
        If you are going from space A to space C via space B,
        you need to find the highest common density between the
        space B atlases and B->C transforms. If not the transformation is not possible.
        """
        atlases = self.search_surface_atlases(space=mid_space)
        print(f"Resources found for transformation: {atlases}")

        transforms = self.search_surface_transforms(
            source_space=mid_space, target_space=target_space
        )
        print(f"Transforms found for transformation: {transforms}")

        # Find highest common density between list of atlases and transforms
        atlas_densities = {atlas.density for atlas in atlases}
        transform_densities = {transform.density for transform in transforms}
        common_densities = atlas_densities & transform_densities

        if common_densities:
            # If densities are strings like "32k", sort numerically
            highest_common_density = max(common_densities, key=_get_density_key)
            print(f"Highest common density: {highest_common_density}")
        else:
            print("No common density found between atlases and transforms.")

        return highest_common_density

    def find_highest_density(self, space: str) -> str:
        """Find the highest density available for a given space in the graph."""
        atlases = self.search_surface_atlases(space=space)
        densities = {atlas.density for atlas in atlases}

        if not densities:
            raise ValueError(f"No atlases found for space '{space}'.")

        highest_density = max(densities, key=_get_density_key)
        print(f"Highest density for space '{space}': {highest_density}")
        return highest_density

    def surface_to_surface_transformer(
        self,
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

        if target_density is None:
            target_density = self.find_highest_density(space=target_space)

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

        kwargs = {
            "current_sphere": transform.fetch(),
            "new_sphere": new_sphere,
            "method": "ADAP_BARY_AREA",
        }

        if transformer_type == "label":
            area_surfs = workbench.label_resample_area_surfs_params(
                current_area=current_area, new_area=new_area
            )
            kwargs.update({"area_surfs": area_surfs})
            resampled_output = label_resample(
                input_file_path=input_file, output_file_path=output_file_path, **kwargs
            )

        elif transformer_type == "metric":
            area_surfs = workbench.metric_resample_area_surfs_params(
                current_area=current_area, new_area=new_area
            )
            kwargs.update({"area_surfs": area_surfs})
            resampled_output = metric_resample(
                input_file_path=input_file, output_file_path=output_file_path, **kwargs
            )

        return resampled_output
