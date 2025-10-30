"""Test suite for graph functionalities in neuromaps_nhp."""

import pytest

from neuromaps_prime.graph import NeuromapsGraph


@pytest.fixture
def graph() -> NeuromapsGraph:
    """Fixture to create a default NeuromapsGraph instance for testing."""
    return NeuromapsGraph()


def test_default_graph_initialization(graph: NeuromapsGraph) -> None:
    """Test if the graph initializes correctly."""
    assert graph is not None
    assert len(graph.nodes) > 0, "Graph should have nodes after initialization."
    assert len(graph.edges) > 0, "Graph should have edges after initialization."


def test_fetch_surface_atlas(graph: NeuromapsGraph) -> None:
    """Test fetching a surface atlas from the graph."""
    atlas = graph.fetch_surface_atlas(
        space="Yerkes19", density="32k", hemisphere="left", resource_type="sphere"
    )
    assert atlas is not None, "Failed to fetch the surface atlas."


def test_fetch_surface_atlas_invalid_space(graph: NeuromapsGraph) -> None:
    """Test fetching a surface atlas with an invalid space."""
    with pytest.raises(ValueError):
        graph.fetch_surface_atlas(
            space="InvalidSpace",
            density="32k",
            hemisphere="left",
            resource_type="sphere",
        )


def test_fetch_surface_to_surface_transform(graph: NeuromapsGraph) -> None:
    """Test fetching a surface-to-surface transformation."""
    transform = graph.fetch_surface_to_surface_transform(
        source="Yerkes19",
        target="S1200",
        density="10k",
        hemisphere="left",
        resource_type="sphere",
    )
    assert transform is not None, (
        "Failed to fetch the surface-to-surface transformation."
    )


def test_find_path(graph: NeuromapsGraph) -> None:
    """Test finding a path between two spaces."""
    path = graph.find_path(source="Yerkes19", target="S1200")
    assert path is not None, "Failed to find a path between the specified spaces."


def test_get_subgraph(graph: NeuromapsGraph) -> None:
    """Test getting a subgraph for specified spaces."""
    subgraph = graph.get_subgraph(edges="surface_to_surface")
    assert subgraph is not None, "Failed to get the subgraph."
    assert len(subgraph.nodes(data=False)) == 8, "Subgraph should have 8 nodes."


def test_get_graph_info(graph: NeuromapsGraph) -> None:
    """Test retrieving graph information."""
    info = graph.get_graph_info()
    assert isinstance(info, dict), "Graph info should be a dictionary."
    assert info["num_nodes"] == 8, "Graph should have 8 surface nodes."


def test_get_node_data(graph: NeuromapsGraph) -> None:
    """Test retrieving node data."""
    yerkes_data = graph.get_node_data("Yerkes19")
    assert yerkes_data is not None, "Failed to retrieve node data."
    assert yerkes_data.name == "Yerkes19", "Node name should be 'Yerkes19'."
