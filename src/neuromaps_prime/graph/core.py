"""Core NeuromapsGraph class.

Thin orchestrator that wires together GraphCache, GraphBuilder, GraphFetchers,
GraphUtils, SurfaceTransformOps, and VolumeTransformOps into the public API.

Graph structure:
  - Nodes: Brain template spaces
  - Edges: Transformations between spaces
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import networkx as nx
from niwrap import Runner

from neuromaps_prime.graph.builder import GraphBuilder
from neuromaps_prime.graph.methods.cache import GraphCache
from neuromaps_prime.graph.methods.fetchers import GraphFetchers
from neuromaps_prime.graph.models import (
    Edge,
    Node,
    SurfaceAtlas,
    SurfaceTransform,
    VolumeAtlas,
    VolumeTransform,
)
from neuromaps_prime.graph.transforms.surface import SurfaceTransformOps
from neuromaps_prime.graph.transforms.volume import VolumeTransformOps
from neuromaps_prime.graph.utils import GraphUtils
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
        """Initialize and populate NeuromapsGraph.

        Args:
            runner: Execution backend for neuroimaging tools.
            runner_kwargs: Keyword arguments forwarded to the runner.
            yaml_file: Path to the graph definition YAML. Defaults to the
                bundled ``neuromaps_graph.yaml``.
            data_dir: Optional root directory prepended to all relative file
                paths in the YAML.
            _testing: When ``True``, skip YAML loading (for unit tests).
        """
        super().__init__()

        self.runner = set_runner(runner=runner, **runner_kwargs)
        self.data_dir = data_dir or os.getenv("NEUROMAPS_DATA", None)
        self.yaml_path = (
            yaml_file
            or Path(__file__).parents[1] / "datasets" / "data" / "neuromaps_graph.yaml"
        )

        # Shared cache — single source of truth for all resource lookups
        self._cache = GraphCache()

        # Stateless helper layer — all share the same cache
        self.fetchers = GraphFetchers(cache=self._cache)
        self.utils = GraphUtils(graph=self, fetchers=self.fetchers)
        self.surface_ops = SurfaceTransformOps(
            cache=self._cache,
            fetchers=self.fetchers,
            utils=self.utils,
        )
        self.volume_ops = VolumeTransformOps(
            cache=self._cache,
            fetchers=self.fetchers,
            utils=self.utils,
            surface_ops=self.surface_ops,
        )
        self._builder = GraphBuilder(
            cache=self._cache,
            data_dir=Path(self.data_dir) if self.data_dir else None,
        )

        if not _testing:
            self._builder.build_from_yaml(self, self.yaml_path)

    # Graph mutation
    def add_transform(
        self, transform: SurfaceTransform | VolumeTransform, key: str
    ) -> None:
        """Register a transform as both a graph edge and a cache entry.

        Args:
            transform: The :class:`SurfaceTransform` or :class:`VolumeTransform`
                to register.
            key: Edge key (``'surface_to_surface'`` or ``'volume_to_volume'``).

        Raises:
            TypeError: If transform is not a supported transform type.
        """
        match transform:
            case SurfaceTransform():
                self._cache.add_surface_transform(transform)
                edge = Edge(surface_transforms=[transform], volume_transforms=[])
            case VolumeTransform():
                self._cache.add_volume_transform(transform)
                edge = Edge(surface_transforms=[], volume_transforms=[transform])
            case _:
                raise TypeError(f"Unsupported transform type: {type(transform)}")

        self.add_edge(
            transform.source_space,
            transform.target_space,
            key=key,
            data=edge,
            weight=transform.weight,
        )

    # Validation
    def validate_spaces(self, source: str, target: str) -> None:
        """Assert that both source and target exist as nodes in the graph.

        Args:
            source: Source space name.
            target: Target space name.

        Raises:
            ValueError: If either space is absent from the graph.
        """
        self.utils.validate_spaces(source, target)  # pragma: no cover (validate tested)

    # Path finding

    def find_path(
        self,
        source: str,
        target: str,
        edge_type: str | None = None,
    ) -> list[str]:
        """Find the shortest weighted path between two spaces.

        Args:
            source: Source space name.
            target: Target space name.
            edge_type: Restrict traversal to ``'surface_to_surface'`` or
                ``'volume_to_volume'`` edges. Uses all edges when ``None``.

        Returns:
            Ordered list of space names, or an empty list when no path exists.
        """
        return self.utils.find_path(source, target, edge_type)

    # Resource fetching
    def fetch_surface_atlas(
        self,
        space: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
    ) -> SurfaceAtlas | None:
        """Fetch a surface atlas resource.

        Args:
            space: Brain template space name.
            density: Surface mesh density (e.g. ``'32k'``).
            hemisphere: ``'left'`` or ``'right'``.
            resource_type: Surface type (e.g. ``'sphere'``, ``'midthickness'``).

        Returns:
            The matching :class:`~neuromaps_prime.graph.models.SurfaceAtlas`, or
            ``None`` if not found.
        """
        return self.fetchers.fetch_surface_atlas(
            space=space,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )

    def fetch_volume_atlas(
        self,
        space: str,
        resolution: str,
        resource_type: str,
    ) -> VolumeAtlas | None:
        """Fetch a volume atlas resource.

        Args:
            space: Brain template space name.
            resolution: Volume resolution (e.g. ``'1mm'``).
            resource_type: Volume type (e.g. ``'T1w'``).

        Returns:
            The matching :class:`~neuromaps_prime.graph.models.VolumeAtlas`, or
            ``None`` if not found.
        """
        return self.fetchers.fetch_volume_atlas(
            space=space,
            resolution=resolution,
            resource_type=resource_type,
        )

    def fetch_surface_to_surface_transform(
        self,
        source: str,
        target: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
    ) -> SurfaceTransform | None:
        """Fetch a surface-to-surface transform resource.

        Args:
            source: Source space name.
            target: Target space name.
            density: Surface mesh density.
            hemisphere: ``'left'`` or ``'right'``.
            resource_type: Surface type (e.g. ``'sphere'``).

        Returns:
            The matching :class:`~neuromaps_prime.graph.models.SurfaceTransform`, or
            ``None`` if not found.
        """
        return self.fetchers.fetch_surface_transform(
            source=source,
            target=target,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )

    def fetch_volume_to_volume_transform(
        self,
        source: str,
        target: str,
        resolution: str,
        resource_type: str,
    ) -> VolumeTransform | None:
        """Fetch a volume-to-volume transform resource.

        Args:
            source: Source space name.
            target: Target space name.
            resolution: Volume resolution.
            resource_type: Volume type.

        Returns:
            The matching :class:`~neuromaps_prime.graph.models.VolumeTransform`, or
            ``None`` if not found.
        """
        return self.fetchers.fetch_volume_transform(
            source=source,
            target=target,
            resolution=resolution,
            resource_type=resource_type,
        )

    # Search
    def search_surface_atlases(
        self,
        space: str,
        density: str | None = None,
        hemisphere: Literal["left", "right"] | None = None,
        resource_type: str | None = None,
    ) -> list[SurfaceAtlas]:
        """Search surface atlases for a space with optional filters.

        Args:
            space: Brain template space name.
            density: Optional density filter.
            hemisphere: Optional hemisphere filter.
            resource_type: Optional resource type filter.

        Returns:
            List of matching :class:`~neuromaps_prime.graph.models.SurfaceAtlas`
            entries.
        """
        return self.fetchers.fetch_surface_atlases(
            space=space,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )

    def search_surface_transforms(
        self,
        source_space: str,
        target_space: str,
        density: str | None = None,
        hemisphere: Literal["left", "right"] | None = None,
        resource_type: str | None = None,
    ) -> list[SurfaceTransform]:
        """Search surface transforms between two spaces with optional filters.

        Args:
            source_space: Source space name.
            target_space: Target space name.
            density: Optional density filter.
            hemisphere: Optional hemisphere filter.
            resource_type: Optional resource type filter.

        Returns:
            List of matching :class:`~neuromaps_prime.graph.models.SurfaceTransform`
            entries.
        """
        return self.fetchers.fetch_surface_transforms(
            source=source_space,
            target=target_space,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )

    # Density helpers

    def find_common_density(self, mid_space: str, target_space: str) -> str:
        """Find the highest density shared between mid_space and target_space.

        Args:
            mid_space: Intermediate space name.
            target_space: Target space name.

        Returns:
            Highest common density string.

        Raises:
            ValueError: If no common density exists.
        """
        return self.utils.find_common_density(mid_space, target_space)

    def find_highest_density(self, space: str) -> str:
        """Return the highest surface density available for space.

        Args:
            space: Brain template space name.

        Returns:
            Highest density string.

        Raises:
            ValueError: If no atlases are registered for space.
        """
        return self.utils.find_highest_density(space)

    # Node introspection
    def get_node_data(self, node_name: str) -> Node:
        """Return the :class:`~neuromaps_prime.graph.models.Node` for node_name.

        Args:
            node_name: Name of the node to retrieve.

        Returns:
            The ``Node`` data object.

        Raises:
            ValueError: If node_name is not present in the graph.
        """
        return self.utils.get_node_data(node_name)

    # Transformers
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
        """Resample a metric or label GIFTI from source_space to target_space.

        Args:
            transformer_type: ``'metric'`` or ``'label'``.
            input_file: Input GIFTI file to resample.
            source_space: Source brain template space.
            target_space: Target brain template space.
            hemisphere: ``'left'`` or ``'right'``.
            output_file_path: Path for the resampled output file.
            source_density: Source mesh density. Estimated from input_file
                when ``None``.
            target_density: Target mesh density. Highest available used when
                ``None``.
            area_resource: Surface type for area correction
                (default ``'midthickness'``).
            add_edge: Whether to register composed transforms.

        Returns:
            Path to the resampled output, or ``None`` if the transform could
            not be resolved.
        """
        return self.surface_ops.transform(
            transformer_type=transformer_type,
            input_file=input_file,
            source_space=source_space,
            target_space=target_space,
            hemisphere=hemisphere,
            output_file_path=output_file_path,
            source_density=source_density,
            target_density=target_density,
            area_resource=area_resource,
            add_edge=add_edge,
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
        """Warp a volume image from source_space to target_space.

        Args:
            input_file: NIfTI volume to transform.
            source_space: Source brain template space.
            target_space: Target brain template space.
            resolution: Target volume resolution (e.g. ``'2mm'``).
            resource_type: Volume resource type (e.g. ``'T1w'``).
            output_file_path: Path for the warped output volume.
            interp: Interpolation method.
            interp_params: Optional interpolation parameters.

        Returns:
            Path to the warped output volume.
        """
        return self.volume_ops.transform_volume(
            input_file=input_file,
            source_space=source_space,
            target_space=target_space,
            resolution=resolution,
            resource_type=resource_type,
            output_file_path=output_file_path,
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
        """Project a volume to surface then resample to target_space.

        Args:
            transformer_type: ``'metric'`` or ``'label'``.
            input_file: NIfTI volume in source_space.
            source_space: Source brain template space.
            target_space: Target brain template space.
            hemisphere: ``'left'`` or ``'right'``.
            output_file_path: Path for the final resampled GIFTI output.
            source_density: Source surface density. Highest available used
                when ``None``.
            target_density: Target surface density. Highest available used
                when ``None``.
            area_resource: Surface type for area correction
                (default ``'midthickness'``).
            add_edge: Whether to register composed transforms.

        Returns:
            Path to the resampled output, or ``None`` if the transform could
            not be resolved.
        """
        return self.volume_ops.transform_volume_to_surface(
            transformer_type=transformer_type,
            input_file=input_file,
            source_space=source_space,
            target_space=target_space,
            hemisphere=hemisphere,
            output_file_path=output_file_path,
            source_density=source_density,
            target_density=target_density,
            area_resource=area_resource,
            add_edge=add_edge,
        )
