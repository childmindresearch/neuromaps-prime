"""Test suite for graph functionalities in neuromaps_nhp."""

from pathlib import Path
import pytest
from niwrap import Runner
from neuromaps_prime.graph import NeuromapsGraph, SurfaceAtlas, VolumeAtlas

NUM_GRAPH_NODES = 6  # Expected number of nodes in the default graph


@pytest.mark.usefixtures("require_data")
def test_graph_initialization_with_data_dir(data_dir: Path, tmp_path: Path) -> None:
    """Test initializing graph with data directory."""
    graph = NeuromapsGraph(data_dir=data_dir)
    assert graph is not None
    assert graph.data_dir == data_dir
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0

    # Test SurfaceAtlas fetch
    surf = graph.fetch_surface_atlas(
        space="CIVETNMT", density="41k", hemisphere="left", resource_type="midthickness"
    )
    assert isinstance(surf, SurfaceAtlas)
    assert surf.file_path.exists()
    assert data_dir.resolve() in surf.file_path.resolve().parents

    # Test VolumeAtlas fetch
    vol = graph.fetch_volume_atlas(
        space="MEBRAINS", resolution="400um", resource_type="T1w"
    )
    assert isinstance(vol, VolumeAtlas)
    assert vol.file_path.exists()
    assert hasattr(vol, "resolution")
    assert hasattr(vol, "resource_type")


@pytest.fixture
def graph(data_dir: Path, runner: Runner) -> NeuromapsGraph:
    """Fixture to create a default NeuromapsGraph instance for testing."""
    return NeuromapsGraph(data_dir=data_dir, runner=runner)


@pytest.mark.usefixtures("require_data")
def test_default_graph_initialization(graph: NeuromapsGraph) -> None:
    """Test if the graph initializes correctly."""
    assert graph is not None
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0


@pytest.mark.usefixtures("require_data")
def test_fetch_surface_atlas(graph: NeuromapsGraph) -> None:
    """Test fetching a surface atlas."""
    atlas = graph.fetch_surface_atlas(
        space="Yerkes19", density="32k", hemisphere="left", resource_type="sphere"
    )
    assert isinstance(atlas, SurfaceAtlas)
    assert atlas.file_path.exists()


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
    assert transform is not None


@pytest.mark.usefixtures("require_data")
def test_find_path(graph: NeuromapsGraph) -> None:
    """Test finding a path between two spaces."""
    path = graph.find_path(source="Yerkes19", target="S1200")
    assert path is not None
    assert len(path) >= 2


@pytest.mark.usefixtures("require_data")
def test_get_subgraph(graph: NeuromapsGraph) -> None:
    """Test getting a subgraph for specified edges."""
    subgraph = graph.get_subgraph(edges="surface_to_surface")
    assert subgraph is not None
    assert len(subgraph.nodes(data=False)) == NUM_GRAPH_NODES


@pytest.mark.usefixtures("require_data")
def test_get_graph_info(graph: NeuromapsGraph) -> None:
    """Test retrieving graph information."""
    info = graph.get_graph_info()
    assert isinstance(info, dict)
    assert info["num_nodes"] == NUM_GRAPH_NODES


@pytest.mark.usefixtures("require_data")
def test_get_node_data(graph: NeuromapsGraph) -> None:
    """Test retrieving node data."""
    node = graph.get_node_data("Yerkes19")
    assert node is not None
    assert node.name == "Yerkes19"


@pytest.mark.usefixtures("require_data")
def test_computed_surface_to_surface(graph: NeuromapsGraph, data_dir: Path, tmp_path: Path) -> None:
    """Test fetching surface-to-surface transform with computed edge key."""
    source_space = "CIVETNMT"
    target_space = "S1200"
    hemisphere = "right"
    input_file = data_dir / (
        "share/Inputs/CIVETNMT/src-CIVETNMT_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii"
    )
    output_file_path = str(tmp_path / f"space-{target_space}_output_label.label.gii")

    _ = graph.surface_to_surface_transformer(
        transformer_type="label",
        input_file=input_file,
        source_space=source_space,
        target_space=target_space,
        hemisphere=hemisphere,
        output_file_path=output_file_path,
    )

    assert graph.has_edge(source_space, target_space, key=graph.surface_to_surface_key)

    path = graph.find_path(
        source=source_space, target=target_space, edge_type=graph.surface_to_surface_key
    )
    assert len(path) == 2

    source_space = "MEBRAINS"
    shortest_path = graph.find_path(
        source=source_space, target=target_space, edge_type=graph.surface_to_surface_key
    )
    assert len(shortest_path) == 3
    assert "CIVETNMT" not in shortest_path


@pytest.mark.usefixtures("require_data")
def test_add_transform(graph: NeuromapsGraph) -> None:
    """Test adding a new surface transform to the graph."""
    source = "Yerkes19"
    target = "S1200"
    density = "10k"
    hemisphere = "left"
    resource_type = "sphere"

    surface_transform = graph.fetch_surface_to_surface_transform(
        source=source, target=target, density=density, hemisphere=hemisphere, resource_type=resource_type
    )
    surface_transform.resource_type = "test_transform_resource"

    graph.add_transform(
        source_space=source,
        target_space=target,
        key=graph.surface_to_surface_key,
        transform=surface_transform,
    )

    added_transform = graph.fetch_surface_to_surface_transform(
        source=source, target=target, density=density, hemisphere=hemisphere, resource_type="test_transform_resource"
    )
    assert added_transform is not None
    assert added_transform.resource_type == "test_transform_resource"


@pytest.mark.usefixtures("require_data")
def test_add_transform_invalid_type(graph: NeuromapsGraph) -> None:
    """Test adding a new transform with an invalid type."""
    surface_transform = "InvalidTransformType"
    with pytest.raises(TypeError):
        graph.add_transform(
            source_space="Yerkes19",
            target_space="S1200",
            key=graph.surface_to_surface_key,
            transform=surface_transform,  # type: ignore[arg-type]
        )


@pytest.mark.usefixtures("require_data")
def test_volume_atlases_exist(graph: NeuromapsGraph) -> None:
    """Test that volume atlases exist and are valid VolumeAtlas objects."""
    for node_name in ["D99", "MEBRAINS", "NMT2Sym", "Yerkes19"]:
        node = graph.get_node_data(node_name)
        assert node.volumes
        for atlas in node.volumes:
            assert isinstance(atlas, VolumeAtlas)
            assert atlas.file_path.exists()
            assert hasattr(atlas, "resolution")
            assert hasattr(atlas, "resource_type")


@pytest.mark.usefixtures("require_data")
def test_volume_to_volume_transform_objects_exist(graph: NeuromapsGraph) -> None:
    """Test that volume-to-volume transforms exist and are valid."""
    found = False
    for _, _, key, edge_data in graph.edges(data=True, keys=True):
        if key != graph.volume_to_volume_key:
            continue
        found = True
        edge = edge_data["data"]
        assert edge.volume_transforms
        for transform in edge.volume_transforms:
            assert transform.file_path.exists()
            assert hasattr(transform, "resolution")
            assert hasattr(transform, "resource_type")
    assert found


@pytest.mark.usefixtures("require_data")
def test_fetch_volume_atlas(graph: NeuromapsGraph) -> None:
    """Test fetching a volume atlas returns a valid VolumeAtlas."""
    atlas = graph.fetch_volume_atlas(space="Yerkes19", resolution="500um", resource_type="T1w")
    assert isinstance(atlas, VolumeAtlas)
    assert atlas.file_path.exists()
    assert atlas.resolution == "500um"
    assert atlas.resource_type == "T1w"


@pytest.mark.usefixtures("require_data")
def test_fetch_volume_to_volume_transform(graph: NeuromapsGraph) -> None:
    """Test fetching a volume-to-volume transform returns valid objects."""
    transform = graph.fetch_volume_to_volume_transform(
        source="Yerkes19", target="NMT2Sym", resolution="250um", resource_type="composite"
    )
    assert transform is not None
    assert transform.file_path.exists()
    assert hasattr(transform, "resolution")
    assert hasattr(transform, "resource_type")
