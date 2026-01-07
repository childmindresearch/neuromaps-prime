"""Tests for pydantic Graph models."""

from pathlib import Path
from typing import Any

import pytest

from neuromaps_prime import graph


@pytest.fixture
def tmp_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.file"
    p.touch()
    return p


class TestResources:
    """Test different resource models."""

    @pytest.mark.parametrize(
        "model_cls, extra_kwargs",
        [
            (
                graph.SurfaceAtlas,
                dict(
                    space="Yerkes19",
                    density="32k",
                    hemisphere="left",
                    resource_type="surf_atlas",
                ),
            ),
            (
                graph.VolumeAtlas,
                dict(
                    space="Yerkes19",
                    resolution="2mm",
                    resource_type="vol_atlas",
                ),
            ),
            (
                graph.SurfaceTransform,
                dict(
                    source_space="Yerkes19",
                    target_space="CIVETNMT",
                    density="32k",
                    hemisphere="right",
                    resource_type="surface_transform",
                ),
            ),
            (
                graph.VolumeTransform,
                dict(
                    source_space="Yerkes19",
                    target_space="CIVETNMT",
                    resolution="2mm",
                    resource_type="volume_transform",
                    weight=3.0,
                ),
            ),
        ],
    )
    def test_validate_and_fetch(
        self, tmp_file: Path, model_cls: Any, extra_kwargs: dict[str, Any]
    ):
        """Test resource instantation and fetching."""
        obj = model_cls(
            name="test",
            description="testing description",
            file_path=tmp_file,
            **extra_kwargs,
        )
        assert obj.file_path == tmp_file
        assert obj.fetch() == tmp_file

    def test_missing_file(self):
        """Test missing file raises error."""
        with pytest.raises(FileNotFoundError, match="File path does not exist"):
            graph.SurfaceAtlas(
                name="invalid",
                description="missing",
                file_path="some_random_path",
                space="Yerkes19",
                density="32k",
                hemisphere="left",
                resource_type="SurfaceAtlas",
            )


class TestNode:
    """Tests associated with Node model."""

    def test_init(self, tmp_file: Path):
        """Test initialization of a node."""
        surf = graph.SurfaceAtlas(
            name="TestSurface",
            description="Test surface",
            file_path=tmp_file,
            space="Yerkes19",
            density="32k",
            hemisphere="left",
            resource_type="SurfaceAtlas",
        )
        node = graph.Node(
            name="TestNode",
            species="Test",
            description="A test node",
            surfaces=[surf],
        )

        assert node.name == "TestNode"
        assert len(node.surfaces) == 1
        assert len(node.volumes) == 0
        assert "A test node" in repr(node)
        assert "TestSurface" in repr(node)


class TestEdge:
    """Tests associated with Edge model."""

    def test_init(self, tmp_file: Path):
        """Test initialization of an edge."""
        surf_edge = graph.SurfaceTransform(
            name="SurfaceTransform",
            description="surface edge",
            file_path=tmp_file,
            source_space="Yerkes19",
            target_space="S1200",
            density="32k",
            hemisphere="left",
            resource_type="surface_transform",
        )
        edge = graph.Edge(surface_transforms=[surf_edge])

        assert len(edge.surface_transforms) == 1
        assert len(edge.volume_transforms) == 0
        assert "SurfaceTransform" in repr(edge)
