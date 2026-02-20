"""Graph utility operations for NeuromapsGraph.

Provides graph traversal, validation, density resolution, and introspection
utilities that operate on the NetworkX graph structure. All methods that need
to query resources do so via GraphFetchers rather than touching the cache
directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import networkx as nx
from pydantic import BaseModel

from neuromaps_prime.graph.methods.fetchers import GraphFetchers
from neuromaps_prime.transforms.utils import _get_density_key


class GraphUtils(BaseModel):
    """Graph traversal, validation, and introspection utilities.

    Attributes:
    ----------
    graph:
        The underlying NetworkX :class:`~networkx.MultiDiGraph`.
    fetchers:
        The :class:`GraphFetchers` instance used for resource lookups.
    """

    model_config = {"arbitrary_types_allowed": True}

    graph: nx.MultiDiGraph
    fetchers: GraphFetchers

    # Validation

    def validate_spaces(self, source: str, target: str) -> None:
        """Assert that both *source* and *target* exist as nodes in the graph.

        Args:
            source: Source space name.
            target: Target space name.

        Raises:
            ValueError: If either space is absent from the graph.
        """
        nodes = set(self.graph.nodes)
        if source not in nodes:
            raise ValueError(
                f"Source space '{source}' does not exist in the graph."
                f" Available spaces: {sorted(nodes)}"
            )
        if target not in nodes:
            raise ValueError(
                f"Target space '{target}' does not exist in the graph."
                f" Available spaces: {sorted(nodes)}"
            )

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
            edge_type: If provided, restrict traversal to edges of this key
                (``'surface_to_surface'`` or ``'volume_to_volume'``).

        Returns:
            Ordered list of space names from *source* to *target*, or an
            empty list when no path exists.
        """
        try:
            g = self.get_subgraph(edge_type) if edge_type else self.graph
            return nx.shortest_path(g, source=source, target=target, weight="weight")
        except nx.NetworkXNoPath:
            return []

    def get_subgraph(self, edge_type: str) -> nx.MultiDiGraph:
        """Return a view containing all nodes but only edges of *edge_type*.

        Results are cached per *edge_type* value — the cache is intentionally
        unbounded because the set of edge types is small and fixed.

        Args:
            edge_type: Edge key to retain (e.g. ``'surface_to_surface'``).

        Returns:
            A new :class:`~networkx.MultiDiGraph` containing only the
            requested edges.
        """
        # lru_cache cannot be applied directly to methods on a Pydantic model,
        # so we delegate to a module-level cached helper.
        return _cached_subgraph(self.graph, edge_type)

    # Density helpers
    def find_common_density(self, mid_space: str, target_space: str) -> str:
        """Find the highest density shared by *mid_space* atlases and transforms.

        Used during multi-hop surface composition to select the best
        intermediate resolution.

        Args:
            mid_space: Intermediate space name.
            target_space: Final target space name.

        Returns:
            The highest common density string (e.g. ``'32k'``).

        Raises:
            ValueError: If no common density exists.
        """
        atlas_densities = {
            a.density for a in self.fetchers.fetch_surface_atlases(space=mid_space)
        }
        transform_densities = {
            t.density
            for t in self.fetchers.fetch_surface_transforms(
                source=mid_space, target=target_space
            )
        }
        common = atlas_densities & transform_densities
        if not common:
            raise ValueError(
                f"No common density found between '{mid_space}' and '{target_space}'."
            )
        return max(common, key=_get_density_key)

    def find_highest_density(self, space: str) -> str:
        """Return the highest surface density available for *space*.

        Args:
            space: Brain template space name.

        Returns:
            The highest density string (e.g. ``'164k'``).

        Raises:
            ValueError: If no surface atlases are registered for *space*.
        """
        densities = {
            a.density for a in self.fetchers.fetch_surface_atlases(space=space)
        }
        if not densities:
            raise ValueError(f"No surface atlases found for space '{space}'.")
        return max(densities, key=_get_density_key)

    # Introspection

    def get_node_data(self, node_name: str) -> Any:  # noqa: ANN401
        """Return the :class:`~neuromaps_prime.graph.models.Node` stored on *node_name*.

        Args:
            node_name: Name of the node to retrieve.

        Returns:
            The ``Node`` data object attached to the node.

        Raises:
            ValueError: If node_name is not present in the graph.
        """
        try:
            return self.graph.nodes[node_name]["data"]
        except KeyError:
            raise ValueError(
                f"Node '{node_name}' not found."
                f" Available nodes: {sorted(self.graph.nodes)}"
            )

    def get_graph_info(self) -> dict[str, int]:
        """Return a summary of the graph structure.

        Returns:
            Dictionary with counts of nodes, edges, surfaces, volumes, and
            each transform type.
        """
        nodes_data = [self.get_node_data(n) for n in self.graph.nodes]
        edge_keys = [k for _, _, k in self.graph.edges(keys=True)]
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_surfaces": sum(len(n.surfaces) for n in nodes_data),
            "num_volumes": sum(len(n.volumes) for n in nodes_data),
            "num_surface_to_surface_transforms": edge_keys.count("surface_to_surface"),
            "num_volume_to_volume_transforms": edge_keys.count("volume_to_volume"),
        }


# ---------------------------------------------------------------------------
# Module-level cached subgraph builder
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _cached_subgraph(graph: nx.MultiDiGraph, edge_type: str) -> nx.MultiDiGraph:
    """Build and cache a subgraph filtered to *edge_type* edges.

    Separated from the class so that :func:`functools.lru_cache` can be
    applied — Pydantic model methods are not directly cacheable because
    ``self`` is not hashable.

    Args:
        graph: The full graph to filter.
        edge_type: Edge key to retain.

    Returns:
        A new :class:`~networkx.MultiDiGraph` with all nodes and only the
        matching edges.
    """
    subgraph = nx.MultiDiGraph()
    subgraph.add_nodes_from(graph.nodes(data=True))
    for u, v, key, data in graph.edges(data=True, keys=True):
        if key == edge_type:
            subgraph.add_edge(u, v, key=key, **data)
    return subgraph
