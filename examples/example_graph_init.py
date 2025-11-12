"""Example script to test and demonstrate the neuromaps_nhp graph functionality."""

from pathlib import Path

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.plotting import plot_graph

if __name__ == "__main__":
    # Load the default Neuromaps graph
    data_dir = Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep")
    graph = NeuromapsGraph(data_dir=data_dir)

    # Print all nodes in the graph
    print(graph.nodes(data=False))

    # Brief summary of the graph
    print(graph)

    # Some extra info about the graph
    print(graph.get_graph_info())

    # Print species of each node
    for node in graph.nodes(data=False):
        print(f"Node: {node}")
        print(f"  Species: {graph.get_node_data(node).species}")

    # Find and print paths between two nodes for different edge types
    print(graph.find_path("fsaverage", "MEBRAINS", edge_type="volume_to_volume"))

    print(graph.find_path("fsaverage", "MEBRAINS", edge_type="surface_to_surface"))

    # Plot full graph with both surface and volume transforms combined
    plot_graph(
        graph,
        graph_type="combined",
        layout="kamada_kawai",
        figsize=(14, 10),
        save_path=Path("examples/neuromaps_graph.png"),
    )

    # Plot only surface transforms
    surface_subgraph = graph.get_subgraph("surface_to_surface")
    plot_graph(
        surface_subgraph,
        graph_type="surface",
        layout="kamada_kawai",
        save_path=Path("examples/neuromaps_surface.png"),
    )

    # Plot only volume transforms
    volume_subgraph = graph.get_subgraph("volume_to_volume")
    plot_graph(
        volume_subgraph,
        graph_type="volume",
        layout="kamada_kawai",
        save_path=Path("examples/neuromaps_volume.png"),
    )
