"""Graph builder for NeuromapsGraph.

Responsible for parsing YAML/dict data into typed model objects and
populating both the NetworkX graph structure and the GraphCache.

Intentionally stateless beyond the dependencies injected at construction:
  - data_dir:  optional root path prepended to all relative file paths
  - cache:     GraphCache instance to populate during build
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import yaml
from pydantic import BaseModel, Field

from neuromaps_prime.graph.methods.cache import GraphCache
from neuromaps_prime.models import (
    Edge,
    Node,
    SurfaceAtlas,
    SurfaceTransform,
    VolumeAtlas,
    VolumeTransform,
)


# Builder
class GraphBuilder(BaseModel):
    """Parses YAML/dict definitions and populates a graph and its cache.

    Attributes:
    ----------
    cache:
        The :class:`GraphCache` instance that will be populated during build.
    data_dir:
        Optional root directory prepended to all relative file paths found in
        the YAML. When ``None``, paths are used as-is.
    """

    model_config = {"arbitrary_types_allowed": True}

    cache: GraphCache
    data_dir: Path | None = Field(default=None)

    # Public entry points

    def build_from_yaml(self, graph: nx.MultiDiGraph, yaml_file: Path) -> None:
        """Populate graph and cache from a YAML file.

        Args:
            graph: The NetworkX graph to populate with nodes and edges.
            yaml_file: Path to the YAML definition file.
        """
        with open(yaml_file, "r") as fh:
            data = yaml.safe_load(fh)
        self.build_from_dict(graph, data)

    def build_from_dict(self, graph: nx.MultiDiGraph, data: dict[str, Any]) -> None:
        """Populate graph and cache from a dictionary.

        Args:
            graph: The NetworkX graph to populate with nodes and edges.
            data: Parsed graph definition (mirrors the YAML schema).
        """
        self._build_nodes(graph, data.get("nodes", []))
        self._build_edges(graph, data.get("edges", {}))

    # Node building
    def _build_nodes(
        self, graph: nx.MultiDiGraph, nodes_list: list[dict[str, Any]]
    ) -> None:
        """Parse all node entries and add them to graph and cache."""
        for node_entry in nodes_list:
            ((node_name, node_data),) = node_entry.items()
            surfaces_dict = node_data.get("surfaces", {})
            volumes_dict = node_data.get("volumes", {})
            description = node_data.get("description", "")

            surfaces = self._parse_surfaces(node_name, description, surfaces_dict)
            volumes = self._parse_volumes(node_name, description, volumes_dict)

            node_obj = Node(
                name=node_name,
                species=node_data.get("species", ""),
                description=description,
                surfaces=surfaces,
                volumes=volumes,
            )
            graph.add_node(node_name, data=node_obj)

            # Populate cache
            self.cache.add_surface_atlases(surfaces)
            self.cache.add_volume_atlases(volumes)

    # Edge building
    def _build_edges(self, graph: nx.MultiDiGraph, edges_dict: dict[str, Any]) -> None:
        """Parse all edge entries and add them to graph and cache."""
        surface_edges = edges_dict.get("surface_to_surface", [])
        volume_edges = edges_dict.get("volume_to_volume", [])

        for edge_data in surface_edges:
            self._build_surface_edge(graph, edge_data)

        for edge_data in volume_edges:
            self._build_volume_edge(graph, edge_data)

    def _build_surface_edge(
        self, graph: nx.MultiDiGraph, edge_data: dict[str, Any]
    ) -> None:
        """Parse a single surface-to-surface edge definition."""
        source = edge_data["from"]
        target = edge_data["to"]
        transforms = self._parse_surface_to_surface_transforms(
            source, target, edge_data.get("surfaces", {})
        )
        edge_obj = Edge(surface_transforms=transforms)
        graph.add_edge(
            source, target, key="surface_to_surface", data=edge_obj, weight=1.0
        )
        self.cache.add_surface_transforms(transforms)

    def _build_volume_edge(
        self, graph: nx.MultiDiGraph, edge_data: dict[str, Any]
    ) -> None:
        """Parse a single volume-to-volume edge definition."""
        source = edge_data["from"]
        target = edge_data["to"]
        transforms = self._parse_volume_to_volume_transforms(
            source, target, edge_data.get("volumes", {})
        )
        edge_obj = Edge(volume_transforms=transforms)
        graph.add_edge(
            source, target, key="volume_to_volume", data=edge_obj, weight=1.0
        )
        self.cache.add_volume_transforms(transforms)

    # Path resolution helper
    def _resolve_path(self, path: str) -> Path:
        """Prepend data_dir to path when set, otherwise return as-is."""
        return (self.data_dir / path) if self.data_dir else Path(path)

    # Surface parsers
    def _parse_surfaces(
        self, node_name: str, description: str, surfaces_dict: dict[str, Any]
    ) -> list[SurfaceAtlas]:
        """Parse surface atlas entries from a node's surfaces dictionary."""
        return [
            SurfaceAtlas(
                name=f"{node_name}_{density}_{hemi}_{surf_type}",
                description=description,
                file_path=self._resolve_path(path),
                space=node_name,
                density=density,
                hemisphere=hemi,
                resource_type=surf_type,
            )
            for density, types in surfaces_dict.items()
            for surf_type, hemispheres in types.items()
            for hemi, path in hemispheres.items()
        ]

    # Volume parsers
    def _parse_volumes(
        self, node_name: str, description: str, volumes_dict: dict[str, Any]
    ) -> list[VolumeAtlas]:
        """Parse volume atlas entries from a node's volumes dictionary."""
        return [
            VolumeAtlas(
                name=f"{node_name}_{res}_{vol_type}",
                description=description,
                file_path=self._resolve_path(path),
                space=node_name,
                resolution=res,
                resource_type=vol_type,
            )
            for res, types in volumes_dict.items()
            for vol_type, path in types.items()
        ]

    # Transform parsers
    def _parse_surface_to_surface_transforms(
        self, source_name: str, target_name: str, surfaces_dict: dict[str, Any]
    ) -> list[SurfaceTransform]:
        """Parse surface-to-surface transform entries from an edge's surfaces dict."""
        return [
            SurfaceTransform(
                name=f"{source_name}_to_{target_name}_{density}_{hemi}_{surf_type}",
                description=f"surf2surf transform from {source_name} to {target_name}",
                file_path=self._resolve_path(path),
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

    def _parse_volume_to_volume_transforms(
        self, source_name: str, target_name: str, volumes_dict: dict[str, Any]
    ) -> list[VolumeTransform]:
        """Parse volume-to-volume transform entries from an edge's volumes dict."""
        return [
            VolumeTransform(
                name=f"{source_name}_to_{target_name}_{res}_{vol_type}",
                description=f"vol2vol transform from {source_name} to {target_name}",
                file_path=self._resolve_path(path),
                source_space=source_name,
                target_space=target_name,
                resolution=res,
                resource_type=vol_type,
            )
            for res, types in volumes_dict.items()
            for vol_type, path in types.items()
        ]
