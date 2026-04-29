"""Graph builder for NeuromapsGraph.

Responsible for parsing YAML/dict data into typed model objects and
populating both the NetworkX graph structure and the GraphCache.

Intentionally stateless beyond the dependencies injected at construction:
  - data_dir:  optional root path prepended to all relative file paths
  - cache:     GraphCache instance to populate during build
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml
from pydantic import BaseModel, Field

from neuromaps_prime.graph.cache import GraphCache  # noqa: TC001 (pydantic req'd)
from neuromaps_prime.graph.models import (
    Edge,
    Node,
    SurfaceAnnotation,
    SurfaceAtlas,
    SurfaceTransform,
    VolumeAnnotation,
    VolumeAtlas,
    VolumeTransform,
)
from neuromaps_prime.resources import NEUROMAPSPRIME_GRAPH

if TYPE_CHECKING:
    import networkx as nx


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

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def build_default(self, graph: nx.MultiDiGraph) -> None:
        """Populate graph and cache using default resources.

        Args:
            graph: The NetworkX graph to populate with nodes and edges.
        """

        def _load_dict(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
            return [yaml.safe_load(path.read_bytes()) for path in paths]

        def _load_list(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
            merged = []
            for path in paths:
                merged.extend(yaml.safe_load(path.read_bytes()))
            return merged

        data = {
            "nodes": _load_dict(NEUROMAPSPRIME_GRAPH.nodes),
            "edges": {
                "surface_to_surface": _load_list(NEUROMAPSPRIME_GRAPH.surface_edges),
                "volume_to_volume": _load_list(NEUROMAPSPRIME_GRAPH.volume_edges),
            },
        }
        self.build_from_dict(graph, data)

    def build_from_yaml(self, graph: nx.MultiDiGraph, yaml_file: Path) -> None:
        """Populate graph and cache from a YAML file.

        Args:
            graph: The NetworkX graph to populate with nodes and edges.
            yaml_file: Path to the YAML definition file.
        """
        with yaml_file.open() as fh:
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

    # ------------------------------------------------------------------ #
    # Node building                                                        #
    # ------------------------------------------------------------------ #

    def _build_nodes(
        self, graph: nx.MultiDiGraph, nodes_list: list[dict[str, Any]]
    ) -> None:
        """Parse all node entries and add them to graph and cache."""
        for node_entry in nodes_list:
            ((node_name, node_data),) = node_entry.items()
            description = node_data.get("description", "")

            surfaces, surface_annotations = self._parse_surface_resources(
                SurfaceAtlas,
                {"space": node_name, "description": description},
                node_data.get("surfaces", {}),
            )
            volumes, volume_annotations = self._parse_volume_resources(
                VolumeAtlas,
                {"space": node_name, "description": description},
                node_data.get("volumes", {}),
            )
            node_obj = Node(
                name=node_name,
                species=node_data.get("species", ""),
                description=description,
                references=node_data.get("references"),
                surfaces=cast("list[SurfaceAtlas]", surfaces),
                volumes=cast("list[VolumeAtlas]", volumes),
                surface_annotations=surface_annotations,
                volume_annotations=volume_annotations,
            )
            graph.add_node(node_name, data=node_obj)
            self.cache.add_surface_atlases(cast("list[SurfaceAtlas]", surfaces))
            self.cache.add_surface_annotations(surface_annotations)
            self.cache.add_volume_atlases(cast("list[VolumeAtlas]", volumes))
            self.cache.add_volume_annotations(volume_annotations)

    # ------------------------------------------------------------------ #
    # Edge building                                                        #
    # ------------------------------------------------------------------ #

    def _build_edges(self, graph: nx.MultiDiGraph, edges_dict: dict[str, Any]) -> None:
        """Parse all edge entries and add them to graph and cache."""
        for edge_data in edges_dict.get("surface_to_surface", []):
            self._build_surface_edge(graph, edge_data)
        for edge_data in edges_dict.get("volume_to_volume", []):
            self._build_volume_edge(graph, edge_data)

    def _build_surface_edge(
        self, graph: nx.MultiDiGraph, edge_data: dict[str, Any]
    ) -> None:
        """Parse a single surface-to-surface edge definition."""
        source, target = edge_data["from"], edge_data["to"]
        transforms, _ = self._parse_surface_resources(
            SurfaceTransform,
            {
                "source_space": source,
                "target_space": target,
                "description": f"surf2surf transform from {source} to {target}",
            },
            edge_data.get("surfaces", {}),
        )
        graph.add_edge(
            source,
            target,
            key="surface_to_surface",
            data=Edge(surface_transforms=cast("list[SurfaceTransform]", transforms)),
            weight=1.0,
        )
        self.cache.add_surface_transforms(cast("list[SurfaceTransform]", transforms))

    def _build_volume_edge(
        self, graph: nx.MultiDiGraph, edge_data: dict[str, Any]
    ) -> None:
        """Parse a single volume-to-volume edge definition."""
        source, target = edge_data["from"], edge_data["to"]
        transforms, _ = self._parse_volume_resources(
            VolumeTransform,
            {
                "source_space": source,
                "target_space": target,
                "description": f"vol2vol transform from {source} to {target}",
            },
            edge_data.get("volumes", {}),
        )
        graph.add_edge(
            source,
            target,
            key="volume_to_volume",
            data=Edge(volume_transforms=cast("list[VolumeTransform]", transforms)),
            weight=1.0,
        )
        self.cache.add_volume_transforms(cast("list[VolumeTransform]", transforms))

    # ------------------------------------------------------------------ #
    # Generic resource parsers                                             #
    # ------------------------------------------------------------------ #

    def _resolve_path(self, path: str) -> Path:
        """Prepend data_dir to path when set, otherwise return as-is."""
        return (self.data_dir / path) if self.data_dir else Path(path)

    def _parse_surface_annotations(
        self, prefix: str, space: str, density: str, hemispheres: dict[str, Any]
    ) -> list[SurfaceAnnotation]:
        """Parse annotation entries from a surface type block.

        Args:
            prefix: Name prefix derived from space or source/target space pair.
            space: Space identifier applied to every annotation.
            density: Density key (e.g. ``"32k"``) shared by all entries in this block.
            hemispheres: Dict keyed by label → {hemi: path, ...} with optional
                ``"references"`` and ``"notes"`` keys.

        Returns:
            List of :class:`SurfaceAnnotation` instances parsed from the block.
        """
        annotations = []
        for label, value in hemispheres.items():
            # Grab references and notes once
            references = value.get("references")
            notes = value.get("notes")
            for hemi, path in value.items():
                if hemi in ("notes", "references"):
                    continue
                annotations.append(
                    SurfaceAnnotation(
                        name=f"{prefix}_{density}_{hemi}_{label}",
                        space=space,
                        label=label,
                        density=density,
                        hemisphere=hemi,
                        file_path=self._resolve_path(path),
                        references=references,
                        notes=notes,
                    )
                )
        return annotations

    def _parse_surface_entries(
        self,
        cls: type[SurfaceAtlas] | type[SurfaceTransform],
        prefix: str,
        density: str,
        surf_type: str,
        hemispheres: dict[str, Any],
        fixed_fields: dict[str, Any],
        provider: str,
        transform_refs: list[Any] | None,
    ) -> list[Any]:
        """Instantiate surface resource objects for a single type/density block.

        Args:
            cls: Model class to instantiate — either :class:`SurfaceAtlas` or
                :class:`SurfaceTransform`.
            prefix: Name prefix derived from space or source/target space pair.
            density: Density key (e.g. ``"32k"``) shared by all entries in this block.
            surf_type: Resource type string (e.g. ``"midthickness"``).
            hemispheres: Dict mapping hemisphere key to file path.
            fixed_fields: Fields forwarded verbatim to every instance (e.g. space,
                description).
            provider: Provider string injected for :class:`SurfaceTransform` entries;
                empty string for atlases.
            transform_refs: References list attached to the enclosing transform block,
                or ``None`` for atlases.

        Returns:
            List of ``cls`` instances, one per hemisphere entry.
        """
        extra = {"provider": provider} if cls is SurfaceTransform else {}
        return [
            cls(
                name=f"{prefix}_{density}_{hemi}_{surf_type}",
                file_path=self._resolve_path(path),
                density=density,
                hemisphere=hemi,  # type: ignore[arg-type]
                resource_type=surf_type,
                references=transform_refs,
                **fixed_fields,  # type: ignore[arg-type]
                **extra,  # type: ignore[arg-type]
            )
            for hemi, path in hemispheres.items()
        ]

    def _parse_surface_resources(
        self,
        cls: type[SurfaceAtlas] | type[SurfaceTransform],
        fixed_fields: dict[str, Any],
        surfaces_dict: dict[str, Any],
    ) -> tuple[list[SurfaceAtlas] | list[SurfaceTransform], list[SurfaceAnnotation]]:
        """Parse surface resource entries from a nested density/type/hemi dict.

        Args:
            cls: The model class to instantiate (SurfaceAtlas or SurfaceTransform).
            fixed_fields: Fields shared by every entry (e.g. space, description).
            surfaces_dict: Nested dict keyed by density → resource_type → hemisphere
                for atlases, or provider → density → resource_type → hemisphere
                for transforms.

        Returns:
            Tuple of ``(resources, annotations)`` where ``resources`` are cast to
            the appropriate list type for ``cls`` and ``annotations`` are any
            :class:`SurfaceAnnotation` entries found inline.
        """
        is_transform = cls is SurfaceTransform
        prefix = fixed_fields.get("space") or (
            f"{fixed_fields['source_space']}_to_{fixed_fields['target_space']}"
        )
        space = fixed_fields.get("space", prefix)

        result: list[Any] = []
        annotations: list[SurfaceAnnotation] = []

        for outer_key, outer_val in surfaces_dict.items():
            if is_transform:
                provider, density_dict = outer_key, outer_val
                transform_refs = density_dict.get("references")
            else:
                provider, density_dict = "", {outer_key: outer_val}
                transform_refs = None

            for density, types in density_dict.items():
                if density == "references":
                    continue
                for surf_type, hemispheres in types.items():
                    if surf_type == "annotation":
                        annotations.extend(
                            self._parse_surface_annotations(
                                prefix, space, density, hemispheres
                            )
                        )
                    else:
                        result.extend(
                            self._parse_surface_entries(
                                cls,
                                prefix,
                                density,
                                surf_type,
                                hemispheres,
                                fixed_fields,
                                provider,
                                transform_refs,
                            )
                        )

        if cls is SurfaceAtlas:
            return cast("list[SurfaceAtlas]", result), annotations
        return cast("list[SurfaceTransform]", result), annotations

    def _parse_volume_resources(
        self,
        cls: type[VolumeAtlas] | type[VolumeTransform],
        fixed_fields: dict[str, Any],
        volumes_dict: dict[str, Any],
    ) -> tuple[list[VolumeAtlas] | list[VolumeTransform], list[VolumeAnnotation]]:
        """Parse volume resource entries from a nested dict.

        Supports both the atlas format (resolution → resource_type → path) and
        the transform format (provider → resolution → resource_type → path).

        Args:
            cls: The model class to instantiate (VolumeAtlas or VolumeTransform).
            fixed_fields: Fields shared by every entry (e.g. space, description).
            volumes_dict: Nested dict, either ``{resolution: {type: path}}``
                or ``{provider: {resolution: {type: path}}}``.

        Returns:
            Tuple of (resources, annotations) where resources are typed to cls
            and annotations are any volume annotation entries found inline.
        """
        is_transform = cls is VolumeTransform
        prefix = fixed_fields.get("space") or (
            f"{fixed_fields['source_space']}_to_{fixed_fields['target_space']}"
        )
        space = fixed_fields.get("space", prefix)

        result: list[Any] = []
        annotations: list[VolumeAnnotation] = []

        transform_refs = None
        for outer_key, outer_val in volumes_dict.items():
            if is_transform:
                provider = outer_key
                resolution_dict = outer_val
                transform_refs = resolution_dict.get("references")
            else:
                provider = ""
                resolution_dict = {outer_key: outer_val}

            for res, types in resolution_dict.items():
                if res == "references":
                    continue
                for vol_type, vol_value in types.items():
                    if vol_type == "annotation":
                        for annot_key, annot_dict in vol_value.items():
                            annotations.append(
                                VolumeAnnotation(
                                    name=f"{prefix}_{res}_{annot_key}",
                                    space=space,
                                    label=annot_key,
                                    resolution=res,
                                    file_path=self._resolve_path(annot_dict.get("uri")),
                                    references=annot_dict.get("references"),
                                    notes=annot_dict.get("notes"),
                                )
                            )
                        continue

                    extra = {"provider": provider} if is_transform else {}
                    result.append(
                        cls(
                            name=f"{prefix}_{res}_{vol_type}",
                            file_path=self._resolve_path(vol_value),
                            resolution=res,
                            resource_type=vol_type,
                            references=transform_refs,
                            **fixed_fields,  # type: ignore[arg-type]
                            **extra,  # type: ignore[arg-type]
                        )
                    )

        if cls is VolumeAtlas:
            return cast("list[VolumeAtlas]", result), annotations
        return cast("list[VolumeTransform]", result), annotations
