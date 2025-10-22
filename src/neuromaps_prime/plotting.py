"""Functions for plotting neuromaps graphs and subgraphs."""

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def plot_graph(
    graph: nx.MultiDiGraph,
    graph_type: str = "combined",
    figsize: tuple[int, int] = (20, 16),
    font_size: int = 12,
    save_path: Path | None = None,
    layout: str = "kamada_kawai",
    legend_rect: tuple[float, float, float, float] = (0, 0, 1, 0.95),
    legend_loc: str = "upper center",
    k: float = 3.0,
    iterations: int = 100,
    seed: int = 42,
) -> None:
    """Plot a neuromaps graph or subgraph.

    Args:
        graph: The neuromaps graph to plot.
        graph_type: Type of graph to plot ('surface', 'volume', or 'combined').
        figsize: Size of the figure.
        font_size: Font size for labels and legend.
        save_path: Path to save the plot. If None, the plot is shown.
        layout: Layout algorithm to use ('spring', 'circular', 'shell', 'planar',
            'kamada_kawai', 'hierarchical').
        legend_rect: Rectangle (left, bottom, right, top) for legend placement.
        legend_loc: Location of the legend.
        k: Optimal distance between nodes for spring layout.
        iterations: Number of iterations for layout algorithms.
        seed: Random seed for layout algorithms.
    """
    if graph_type == "combined":
        figsize = (figsize[0] * 2, figsize[1] * 2)
        _plot_combined_graph(
            graph,
            figsize,
            font_size,
            save_path,
            layout,
            legend_rect,
            legend_loc,
            k,
            iterations,
            seed,
        )
    else:
        _plot_single_graph(
            graph,
            graph_type,
            figsize,
            font_size,
            save_path,
            layout,
            legend_rect,
            legend_loc,
            k,
            iterations,
            seed,
        )


def _get_optimized_layout(
    graph: nx.MultiDiGraph,
    layout: str = "kamada_kawai",
    k: float = 3.0,
    iterations: int = 100,
    seed: int = 42,
) -> dict:
    """Get optimized node positions to minimize edge crossings."""
    if layout == "hierarchical":
        # Group nodes by species for hierarchical layout
        species_groups: dict[str, list[str]] = {}
        for node, data in graph.nodes(data=True):
            species = data["data"].species
            species_groups.setdefault(species, []).append(node)

        # Use graphviz dot layout if available, otherwise use multipartite
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
        except (ImportError, FileNotFoundError):
            # Fallback to multipartite layout
            pos = _hierarchical_multipartite_layout(graph, species_groups)

    elif layout == "circular":
        # Circular layout with species grouping
        pos = _species_circular_layout(graph)

    elif layout == "shell":
        # Shell layout with species as shells
        species_groups = {}
        for node, data in graph.nodes(data=True):
            species = data["data"].species
            species_groups.setdefault(species, []).append(node)
        shells = list(species_groups.values())
        pos = nx.shell_layout(graph, nlist=shells)

    elif layout == "kamada_kawai":
        # Kamada-Kawai layout (good for small graphs)
        pos = nx.kamada_kawai_layout(graph)

    elif layout == "planar":
        # Planar layout (if graph is planar)
        try:
            pos = nx.planar_layout(graph)
        except nx.NetworkXException:
            print("Graph is not planar, falling back to spring layout")
            pos = nx.spring_layout(graph, k=k, iterations=iterations, seed=seed)

    else:  # spring (default)
        # Enhanced spring layout with more iterations
        pos = nx.spring_layout(graph, k=k, iterations=iterations, seed=seed)

    return pos


def _hierarchical_multipartite_layout(
    graph: nx.MultiDiGraph, species_groups: dict
) -> dict:
    """Create a hierarchical layout based on species groups."""
    import numpy as np

    pos = {}
    y_offset = 0
    species_list = sorted(species_groups.keys())

    for i, species in enumerate(species_list):
        nodes = species_groups[species]
        n_nodes = len(nodes)

        # Arrange nodes in a horizontal line for each species
        if n_nodes == 1:
            x_positions = [0]
        else:
            x_positions = np.linspace(-n_nodes / 2, n_nodes / 2, n_nodes)

        for j, node in enumerate(nodes):
            pos[node] = (x_positions[j], y_offset)

        y_offset += 2  # Space between species levels

    return pos


def _species_circular_layout(graph: nx.MultiDiGraph) -> dict:
    """Create a circular layout with species grouped together."""
    import numpy as np

    # Group nodes by species
    species_groups: dict[str, list[str]] = {}
    for node, data in graph.nodes(data=True):
        species = data["data"].species
        species_groups.setdefault(species, []).append(node)

    pos = {}
    species_list = sorted(species_groups.keys())
    n_species = len(species_list)

    # Calculate angular positions for each species group
    for i, species in enumerate(species_list):
        nodes = species_groups[species]
        n_nodes = len(nodes)

        # Base angle for this species group
        base_angle = 2 * np.pi * i / n_species

        # Spread nodes within a sector
        if n_nodes == 1:
            angles = [base_angle]
        else:
            sector_width = 2 * np.pi / n_species * 0.8  # 80% of available space
            angles = np.linspace(
                base_angle - sector_width / 2, base_angle + sector_width / 2, n_nodes
            )

        # Different radii for variety
        radii = [1.0 + 0.3 * (j % 2) for j in range(n_nodes)]

        for j, node in enumerate(nodes):
            angle = angles[j]
            radius = radii[j]
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

    return pos


def _plot_combined_graph(
    graph: nx.MultiDiGraph,
    figsize: tuple[int, int],
    font_size: int,
    save_path: Path | None,
    layout: str,
    legend_rect: tuple[float, float, float, float],
    legend_loc: str,
    k: float,
    iterations: int,
    seed: int,
) -> None:
    """Plot combined surface and volume transforms in separate subplots."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Use optimized layout
    pos = _get_optimized_layout(graph, layout, k, iterations, seed)

    # Get node colors by species
    node_colors, species_colors_map = _get_node_colors(graph)

    # Separate edges by type
    surface_edges, volume_edges = _separate_edges(graph)

    # Get color maps for edges
    surface_colors = _get_edge_colors(surface_edges, plt.cm.Set1)
    volume_colors = _get_edge_colors(volume_edges, plt.cm.Set2)

    # Plot surface transforms
    _draw_subplot(
        graph,
        pos,
        axes[0],
        node_colors,
        surface_edges,
        surface_colors,
        "Surface Transforms",
        "-",
        "->",
        font_size,
    )

    # Plot volume transforms
    _draw_subplot(
        graph,
        pos,
        axes[1],
        node_colors,
        volume_edges,
        volume_colors,
        "Volume Transforms",
        "--",
        "-|>",
        font_size,
    )

    # Create combined legend
    legend_elements = _create_legend_elements(
        species_colors_map, surface_colors, volume_colors
    )
    fig.legend(handles=legend_elements, fontsize=font_size, loc=legend_loc, ncol=5)
    plt.tight_layout(rect=legend_rect)

    _save_or_show(save_path)


def _plot_single_graph(
    graph: nx.MultiDiGraph,
    graph_type: str,
    figsize: tuple[int, int],
    font_size: int,
    save_path: Path | None,
    layout: str,
    legend_rect: tuple[float, float, float, float],
    legend_loc: str,
    k: float,
    iterations: int,
    seed: int,
) -> None:
    """Plot either surface or volume transforms in a single plot."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Use optimized layout
    pos = _get_optimized_layout(graph, layout)

    # Get node colors by species
    node_colors, species_colors_map = _get_node_colors(graph)

    if graph_type == "surface":
        edges = _extract_surface_edges(graph)
        edge_colors = _get_edge_colors(edges, plt.cm.Set1)
        title = "Surface Transforms"
        linestyle, arrowstyle = "-", "->"
        legend_prefix = "Surface"
    else:  # volume
        edges = _extract_volume_edges(graph)
        edge_colors = _get_edge_colors(edges, plt.cm.Set2)
        title = "Volume Transforms"
        linestyle, arrowstyle = "--", "-|>"
        legend_prefix = "Volume"

    # Draw the graph
    _draw_subplot(
        graph,
        pos,
        ax,
        node_colors,
        edges,
        edge_colors,
        title,
        linestyle,
        arrowstyle,
        font_size,
    )

    # Create legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=12,
            label=f"Species: {s}",
        )
        for s, color in species_colors_map.items()
    ] + [
        Line2D(
            [0],
            [0],
            color=color,
            lw=3,
            linestyle=linestyle,
            label=f"{legend_prefix} {attr}",
        )
        for attr, color in edge_colors.items()
    ]

    ax.legend(handles=legend_elements, fontsize=font_size, loc="upper right")
    plt.tight_layout()

    _save_or_show(save_path)


def _get_node_colors(graph: nx.MultiDiGraph) -> tuple[list, dict]:
    """Get node colors based on species from Node dataclass."""
    # Extract unique species from Node objects
    species_set = {node_data["data"].species for _, node_data in graph.nodes(data=True)}
    species_list = sorted(list(species_set))

    # Create color mapping for each species
    species_colors_map = {
        species: plt.cm.tab20(i / max(1, len(species_list) - 1))
        for i, species in enumerate(species_list)
    }

    # Assign colors to each node based on species
    node_colors = [
        species_colors_map[node_data["data"].species]
        for _, node_data in graph.nodes(data=True)
    ]

    return node_colors, species_colors_map


def _separate_edges(graph: nx.MultiDiGraph) -> tuple[dict, dict]:
    """Separate surface and volume edges."""
    surface_edges: dict = {}
    volume_edges: dict = {}

    for u, v, data in graph.edges(data=True):
        edge_data = data.get("data")
        if not edge_data:
            continue
        for st in getattr(edge_data, "surface_transforms", []):
            key = (u, v, st.density)
            surface_edges.setdefault(key, []).append(st)
        for vt in getattr(edge_data, "volume_transforms", []):
            key = (u, v, vt.resolution)
            volume_edges.setdefault(key, []).append(vt)

    return surface_edges, volume_edges


def _extract_surface_edges(graph: nx.MultiDiGraph) -> dict:
    """Extract only surface edges from graph."""
    surface_edges: dict = {}
    for u, v, data in graph.edges(data=True):
        edge_data = data.get("data")
        if not edge_data:
            continue
        for st in getattr(edge_data, "surface_transforms", []):
            key = (u, v, st.density)
            surface_edges.setdefault(key, []).append(st)
    return surface_edges


def _extract_volume_edges(graph: nx.MultiDiGraph) -> dict:
    """Extract only volume edges from graph."""
    volume_edges: dict = {}
    for u, v, data in graph.edges(data=True):
        edge_data = data.get("data")
        if not edge_data:
            continue
        for vt in getattr(edge_data, "volume_transforms", []):
            key = (u, v, vt.resolution)
            volume_edges.setdefault(key, []).append(vt)
    return volume_edges


def _get_edge_colors(edges: dict, colormap: Callable[[float], Any]) -> dict:
    """Get color mapping for edge attributes."""
    attrs = {attr for (_, _, attr) in edges.keys()}
    return {attr: colormap(i / max(1, len(attrs))) for i, attr in enumerate(attrs)}


def _draw_subplot(
    graph: nx.MultiDiGraph,
    pos: dict,
    ax: plt.Axes,
    node_colors: list,
    edges: dict,
    edge_colors: dict,
    title: str,
    linestyle: str,
    arrowstyle: str,
    font_size: int,
) -> None:
    """Draw nodes, edges, and labels on a subplot."""
    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=5500,
        alpha=0.8,
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=font_size, font_weight="bold", ax=ax)

    # Draw edges
    _draw_edges(graph, pos, edges, edge_colors, ax, linestyle, arrowstyle)

    ax.set_title(f"{title} ({len(edges)} edges)", fontsize=font_size + 2)
    ax.axis("off")


def _draw_edges(
    graph: nx.MultiDiGraph,
    pos: dict,
    edges: dict,
    edge_colors: dict,
    ax: plt.Axes,
    linestyle: str,
    arrowstyle: str,
) -> None:
    """Draw edges with proper curves for multiple connections."""
    edge_groups: dict[tuple[str, str], list[tuple[str, list]]] = {}
    for (u, v, attr), edges_list in edges.items():
        edge_groups.setdefault((u, v), []).append((attr, edges_list))

    for (u, v), group in edge_groups.items():
        n = len(group)
        base_rad = 0.1 if linestyle == "-" else -0.1
        for i, (attr, edges_list) in enumerate(group):
            rad = base_rad + (i - (n - 1) / 2) * 0.15 if n > 1 else base_rad
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(u, v)],
                ax=ax,
                edge_color=edge_colors[attr],
                style=linestyle,
                arrows=True,
                arrowsize=15,
                arrowstyle=arrowstyle,
                alpha=0.7,
                connectionstyle=f"arc3,rad={rad}",
                width=2,
                min_source_margin=40,
                min_target_margin=40,
            )


def _create_legend_elements(
    species_colors_map: dict, surface_colors: dict, volume_colors: dict
) -> list:
    """Create legend elements for combined plot."""
    return (
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=12,
                label=f"Species: {s}",
            )
            for s, color in species_colors_map.items()
        ]
        + [
            Line2D([0], [0], color=color, lw=3, label=f"Surface {dens}")
            for dens, color in surface_colors.items()
        ]
        + [
            Line2D([0], [0], color=color, lw=3, linestyle="--", label=f"Volume {res}")
            for res, color in volume_colors.items()
        ]
    )


def _save_or_show(save_path: Path | None) -> None:
    """Save plot to file or show it."""
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Graph saved to: {save_path}")
    else:
        plt.show()
