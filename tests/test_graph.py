"""Test suite for graph functionalities in neuromaps_nhp."""

from pathlib import Path

import pytest
from niwrap import Runner

from neuromaps_prime.graph import NeuromapsGraph

NUM_GRAPH_NODES = 6  # Expected number of nodes in the default graph


@pytest.mark.usefixtures("require_data")
def test_graph_initialization_with_data_dir(data_dir: Path, tmp_path: Path) -> None:
    """Test initializing graph with data directory."""
    graph = NeuromapsGraph(data_dir=data_dir)
    assert graph is not None
    assert graph.data_dir == data_dir
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    surf = graph.fetch_surface_atlas(
        space="CIVETNMT", density="41k", hemisphere="left", resource_type="midthickness"
    )
    assert surf is not None
    assert data_dir.resolve() in surf.file_path.resolve().parents


@pytest.fixture
def graph(data_dir: Path, runner: Runner) -> NeuromapsGraph:
    """Fixture to create a default NeuromapsGraph instance for testing."""
    return NeuromapsGraph(data_dir=data_dir, runner=runner)


@pytest.mark.usefixtures("require_data")
def test_default_graph_initialization(graph: NeuromapsGraph) -> None:
    """Test if the graph initializes correctly."""
    assert graph is not None
    assert len(graph.nodes) > 0, "Graph should have nodes after initialization."
    assert len(graph.edges) > 0, "Graph should have edges after initialization."


@pytest.mark.usefixtures("require_data")
def test_fetch_surface_atlas(graph: NeuromapsGraph) -> None:
    """Test fetching a surface atlas from the graph."""
    atlas = graph.fetch_surface_atlas(
        space="Yerkes19", density="32k", hemisphere="left", resource_type="sphere"
    )
    assert atlas is not None, "Failed to fetch the surface atlas."


@pytest.mark.usefixtures("require_data")
def test_fetch_surface_atlas_invalid_space(graph: NeuromapsGraph) -> None:
    """Test fetching a surface atlas with an invalid space."""
    with pytest.raises(ValueError):
        graph.fetch_surface_atlas(
            space="InvalidSpace",
            density="32k",
            hemisphere="left",
            resource_type="sphere",
        )


@pytest.mark.usefixtures("require_data")
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


@pytest.mark.usefixtures("require_data")
def test_find_path(graph: NeuromapsGraph) -> None:
    """Test finding a path between two spaces."""
    path = graph.find_path(source="Yerkes19", target="S1200")
    assert path is not None, "Failed to find a path between the specified spaces."


@pytest.mark.usefixtures("require_data")
def test_get_subgraph(graph: NeuromapsGraph) -> None:
    """Test getting a subgraph for specified spaces."""
    subgraph = graph.get_subgraph(edges="surface_to_surface")
    assert subgraph is not None, "Failed to get the subgraph."
    assert len(subgraph.nodes(data=False)) == NUM_GRAPH_NODES, (
        f"Subgraph should have {NUM_GRAPH_NODES} nodes."
    )


@pytest.mark.usefixtures("require_data")
def test_get_graph_info(graph: NeuromapsGraph) -> None:
    """Test retrieving graph information."""
    info = graph.get_graph_info()
    assert isinstance(info, dict), "Graph info should be a dictionary."
    assert info["num_nodes"] == NUM_GRAPH_NODES, (
        f"Graph should have {NUM_GRAPH_NODES} surface nodes."
    )


@pytest.mark.usefixtures("require_data")
def test_get_node_data(graph: NeuromapsGraph) -> None:
    """Test retrieving node data."""
    yerkes_data = graph.get_node_data("Yerkes19")
    assert yerkes_data is not None, "Failed to retrieve node data."
    assert yerkes_data.name == "Yerkes19", "Node name should be 'Yerkes19'."


@pytest.mark.usefixtures("require_data")
def test_computed_surface_to_surface(
    graph: NeuromapsGraph, data_dir: Path, tmp_path: Path
) -> None:
    """Test fetching surface-to-surface transform with computed edge key."""
    source_space = "CIVETNMT"
    target_space = "S1200"
    hemisphere = "right"
    input_file = data_dir / (
        "share/Inputs/CIVETNMT/src-CIVETNMT_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii"
    )
    output_file_path = str(tmp_path / f"space-{target_space}_output_label.label.gii")

    # A dummy transform to add edge to the graph
    _ = graph.surface_to_surface_transformer(
        transformer_type="label",
        input_file=input_file,
        source_space=source_space,
        target_space=target_space,
        hemisphere=hemisphere,
        output_file_path=output_file_path,
    )

    # Assert the graph has the computed edge
    assert graph.has_edge(
        source_space, target_space, key=graph.surface_to_surface_key
    ), "Graph should have the computed surface_to_surface edge."

    # Assert if the computed edge is used.
    path = graph.find_path(
        source=source_space,
        target=target_space,
        edge_type=graph.surface_to_surface_key,
    )
    assert len(path) == 2, "Shortest path should be direct and use the computed edge."

    # Assert if this computed edge does not corrupt the shortest path finding.
    source_space = "MEBRAINS"
    shortest_path = graph.find_path(
        source=source_space, target=target_space, edge_type=graph.surface_to_surface_key
    )
    assert len(shortest_path) == 3, "Shortest path length should be 3."
    assert "CIVETNMT" not in shortest_path, "Path should not include CIVETNMT node."


@pytest.mark.usefixtures("require_data")
def test_add_transform(graph: NeuromapsGraph, data_dir: Path) -> None:
    """Test adding a new transform to the graph."""
    source = "Yerkes19"
    target = "S1200"
    density = "10k"
    hemisphere = "left"
    resource_type = "sphere"

    surface_transform = graph.fetch_surface_to_surface_transform(
        source=source,
        target=target,
        density=density,
        hemisphere=hemisphere,
        resource_type=resource_type,
    )

    assert surface_transform is not None, "Could not fetch existing surface transform."

    surface_transform.resource_type = "test_transform_resource"

    graph.add_transform(
        source_space=source,
        target_space=target,
        key=graph.surface_to_surface_key,
        transform=surface_transform,
    )

    added_surface_transform = graph.fetch_surface_to_surface_transform(
        source=source,
        target=target,
        density=density,
        hemisphere=hemisphere,
        resource_type="test_transform_resource",
    )
    assert added_surface_transform is not None, (
        "Failed to fetch the newly added surface transform."
    )
    assert added_surface_transform.resource_type == "test_transform_resource", (
        "Resource type of the added transform does not match."
    )


@pytest.mark.usefixtures("require_data")
def test_add_transform_invalid_type(graph: NeuromapsGraph, data_dir: Path) -> None:
    """Test adding a new transform with an invalid type."""
    source = "Yerkes19"
    target = "S1200"

    surface_transform = "InvalidTransformType"

    with pytest.raises(TypeError, match="Unsupported transform type: <class 'str'>"):
        graph.add_transform(
            source_space=source,
            target_space=target,
            key=graph.surface_to_surface_key,
            transform=surface_transform,  # type: ignore[arg-type]
        )

@pytest.mark.usefixtures("require_data")
def test_fetch_volume_atlas_paths(graph: NeuromapsGraph) -> None:
    """Test that volume atlas paths exist for each node with volumes."""
    for node_name in ["D99", "MEBRAINS", "NMT2Sym", "Yerkes19"]:
        volumes = graph.get_node_data(node_name).volumes
        assert volumes, f"Node {node_name} should have volume entries."
        for res, vol_types in volumes.items():
            for vol_type, vol_path in vol_types.items():
                vol_path = Path(vol_path)
                assert vol_path.exists(), f"Volume file {vol_path} for {node_name} does not exist."


@pytest.mark.usefixtures("require_data")
def test_fetch_volume_to_volume_transform_paths(graph: NeuromapsGraph) -> None:
    """Test that volume-to-volume transform paths exist."""
    edges = graph.edges.get("volume_to_volume", [])
    assert edges, "Graph should have volume_to_volume edges."

    for edge in edges:
        source = edge["from"]
        target = edge["to"]
        volumes = edge.get("volumes", {})
        assert volumes, f"Edge {source} -> {target} should have volume transforms."
        for res, vol_dict in volumes.items():
            for key, vol_path in vol_dict.items():
                vol_path = Path(vol_path)
                assert vol_path.exists(), (
                    f"Volume transform file {vol_path} for {source} -> {target} does not exist."
                )


@pytest.mark.usefixtures("require_data")
def test_volume_to_volume_transformer_execution(graph: NeuromapsGraph, tmp_path: Path) -> None:
    """Test executing a volume-to-volume transformation from the graph."""
    # Pick a known volume transform edge
    edge = next(e for e in graph.edges["volume_to_volume"] if e["from"] == "MEBRAINS" and e["to"] == "Yerkes19")
    vol_path = Path(edge["volumes"]["500um"]["composite"])
    output_path = tmp_path / "transformed_vol.nii.gz"

    transformed_vol = graph.volume_to_volume_transformer(
        input_file=vol_path,
        source_space="MEBRAINS",
        target_space="Yerkes19",
        output_file_path=str(output_path),
        resolution="500um",
        interp="linear",
    )

    assert transformed_vol.exists(), "Transformed volume file was not created."
    assert transformed_vol == output_path, "Returned path does not match expected output."