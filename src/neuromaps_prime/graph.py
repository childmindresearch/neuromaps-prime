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

from neuromaps_prime.models import (
    SurfaceAtlas,
    SurfaceTransform,
    VolumeAtlas,
    VolumeTransform,
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
