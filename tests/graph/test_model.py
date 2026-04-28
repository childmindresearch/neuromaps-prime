"""Tests for pydantic Graph models."""

from pathlib import Path
from typing import Any

import pytest

from neuromaps_prime.graph import models


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
                models.SurfaceAtlas,
                dict(
                    space="Yerkes19",
                    density="32k",
                    hemisphere="left",
                    resource_type="surf_atlas",
                ),
            ),
            (
                models.VolumeAtlas,
                dict(
                    space="Yerkes19",
                    resolution="2mm",
                    resource_type="vol_atlas",
                ),
            ),
            (
                models.SurfaceTransform,
                dict(
                    source_space="Yerkes19",
                    target_space="CIVETNMT",
                    density="32k",
                    hemisphere="right",
                    resource_type="surface_transform",
                    provider="test",
                ),
            ),
            (
                models.VolumeTransform,
                dict(
                    source_space="Yerkes19",
                    target_space="CIVETNMT",
                    resolution="2mm",
                    resource_type="volume_transform",
                    provider="test",
                    weight=3.0,
                ),
            ),
            (
                models.SurfaceAnnotation,
                dict(
                    space="Yerkes19",
                    density="32k",
                    hemisphere="left",
                    label="myelin",
                ),
            ),
            (
                models.VolumeAnnotation,
                dict(
                    space="Yerkes19",
                    resolution="2mm",
                    label="myelin",
                ),
            ),
        ],
    )
    def test_validate_and_fetch(
        self, tmp_file: Path, model_cls: Any, extra_kwargs: dict[str, Any]
    ):
        """Test resource instantiation and fetching."""
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
            models.SurfaceAtlas(
                name="invalid",
                description="missing",
                file_path="some_random_path",  # type: ignore
                space="Yerkes19",
                density="32k",
                hemisphere="left",
                resource_type="SurfaceAtlas",
            )

    def test_description_optional(self, tmp_file: Path):
        """Test that description defaults to None for annotation models."""
        surf_annot = models.SurfaceAnnotation(
            name="test",
            file_path=tmp_file,
            space="Yerkes19",
            density="32k",
            hemisphere="left",
            label="myelin",
        )
        vol_annot = models.VolumeAnnotation(
            name="test",
            file_path=tmp_file,
            space="Yerkes19",
            resolution="2mm",
            label="myelin",
        )
        assert surf_annot.description is None
        assert vol_annot.description is None


class TestNode:
    """Tests associated with Node model."""

    @pytest.fixture
    def surface(self, tmp_file: Path) -> models.SurfaceAtlas:
        """Surface atlas fixture."""
        return models.SurfaceAtlas(
            name="TestSurface",
            description="Test surface",
            file_path=tmp_file,
            space="Yerkes19",
            density="32k",
            hemisphere="left",
            resource_type="SurfaceAtlas",
        )

    @pytest.fixture
    def volume(self, tmp_file: Path) -> models.VolumeAtlas:
        """Volume atlas fixture."""
        return models.VolumeAtlas(
            name="TestVolume",
            description="Test volume",
            file_path=tmp_file,
            space="Yerkes19",
            resolution="2mm",
            resource_type="VolumeAtlas",
        )

    @pytest.fixture
    def surface_annotation(self, tmp_file: Path) -> models.SurfaceAnnotation:
        """Surface annotation fixture."""
        return models.SurfaceAnnotation(
            name="TestSurfaceAnnotation",
            file_path=tmp_file,
            space="Yerkes19",
            density="32k",
            hemisphere="left",
            label="myelin",
        )

    @pytest.fixture
    def volume_annotation(self, tmp_file: Path) -> models.VolumeAnnotation:
        """Volume annotation fixture."""
        return models.VolumeAnnotation(
            name="TestVolumeAnnotation",
            file_path=tmp_file,
            space="Yerkes19",
            resolution="2mm",
            label="myelin",
        )

    def test_init(self, surface: models.SurfaceAtlas):
        """Test initialization of a node with surfaces only."""
        node = models.Node(
            name="TestNode",
            species="Test",
            description="A test node",
            surfaces=[surface],
        )
        assert node.name == "TestNode"
        assert len(node.surfaces) == 1
        assert len(node.volumes) == 0
        assert len(node.surface_annotations) == 0
        assert len(node.volume_annotations) == 0

    def test_init_with_annotations(
        self,
        surface: models.SurfaceAtlas,
        volume: models.VolumeAtlas,
        surface_annotation: models.SurfaceAnnotation,
        volume_annotation: models.VolumeAnnotation,
    ):
        """Test initialization of a node with all resource types."""
        node = models.Node(
            name="TestNode",
            species="Test",
            description="A test node",
            surfaces=[surface],
            volumes=[volume],
            surface_annotations=[surface_annotation],
            volume_annotations=[volume_annotation],
        )
        assert len(node.surfaces) == 1
        assert len(node.volumes) == 1
        assert len(node.surface_annotations) == 1
        assert len(node.volume_annotations) == 1

    def test_repr(
        self,
        surface: models.SurfaceAtlas,
        volume: models.VolumeAtlas,
        surface_annotation: models.SurfaceAnnotation,
        volume_annotation: models.VolumeAnnotation,
    ):
        """Test repr contains all resource names."""
        node = models.Node(
            name="TestNode",
            species="Test",
            description="A test node",
            surfaces=[surface],
            volumes=[volume],
            surface_annotations=[surface_annotation],
            volume_annotations=[volume_annotation],
        )
        r = repr(node)
        assert "A test node" in r
        assert "TestSurface" in r
        assert "TestVolume" in r
        assert "TestSurfaceAnnotation" in r
        assert "TestVolumeAnnotation" in r


class TestEdge:
    """Tests associated with Edge model."""

    def test_init(self, tmp_file: Path):
        """Test initialization of an edge."""
        surf_edge = models.SurfaceTransform(
            name="SurfaceTransform",
            description="surface edge",
            file_path=tmp_file,
            source_space="Yerkes19",
            target_space="S1200",
            density="32k",
            hemisphere="left",
            resource_type="surface_transform",
            provider="test",
        )
        edge = models.Edge(surface_transforms=[surf_edge])

        assert len(edge.surface_transforms) == 1
        assert len(edge.volume_transforms) == 0
        assert "SurfaceTransform" in repr(edge)

    def test_init_defaults_to_empty(self):
        """Test that edge initializes with empty transform lists."""
        edge = models.Edge()
        assert len(edge.surface_transforms) == 0
        assert len(edge.volume_transforms) == 0
