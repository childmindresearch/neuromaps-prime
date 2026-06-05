"""Example script to test and demonstrate the neuromaps_nhp graph functionality.

Covers:
- Loads the graph and prints a summary
- Inspects nodes and species
- Finding paths between spaces
- Plotting full, surface-only, and volume-only graphs

Usage:

Set DATA_DIR to the root of your neuromaps data directory, then run:

    python examples/example_graph_init.py
"""

from pathlib import Path

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.plotting import plot_graph


# Configuration (EDIT this path before running)
DATA_DIR = Path("/Users/janhavi.pillai/Desktop/projects/neuromaps-nhp-prep")

graph = NeuromapsGraph(data_dir=DATA_DIR)

# Brief summary of graph
print(graph)
print(graph.utils.get_graph_info())

# All registered spaces
print("Nodes:", list(graph.nodes(data=False)))

# Print species of each node
for node in graph.nodes(data=False):
    print(f"Node: {node}")
    print(f"  Species: {graph.get_node_data(node).species}")

# Find and print paths between two nodes for different edge types
source, target = "CIVETNMT", "Yerkes19"

vol_path = graph.find_path(source, target, edge_type="volume_to_volume")
print(f"Volume path {source} -> {target}: {vol_path}")

surf_path = graph.find_path(source, target, edge_type="surface_to_surface")
print(f"Surface path {source} -> {target}: {surf_path}")

# Plot full graph with both surface and volume transforms combined
plot_graph(
    graph,
    graph_type="combined",
    layout="kamada_kawai",
    figsize=(14, 10),
    save_path=Path("examples/neuromaps_graph.png"),
)

# Plot only surface transforms
surface_subgraph = graph.utils.get_subgraph("surface_to_surface")
plot_graph(
    surface_subgraph,
    graph_type="surface",
    layout="kamada_kawai",
    save_path=Path("examples/neuromaps_surface.png"),
)

# Plot only volume transforms
volume_subgraph = graph.utils.get_subgraph("volume_to_volume")
plot_graph(
    volume_subgraph,
    graph_type="volume",
    layout="kamada_kawai",
    save_path=Path("examples/neuromaps_volume.png"),
)
