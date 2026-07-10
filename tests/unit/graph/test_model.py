"""Tests for Graph models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime.graph import models

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def tmp_file(tmp_path: Path) -> Path:
    """Temporary file fixture."""
    p = tmp_path / "test.file"
    p.touch()
    return p


class ModelProtocol(Protocol):
    """Protocol mocking Model type."""

    file_path: Path

    def __init__(  # noqa: D107
        self,
        name: str,
        description: str,
        file_path: Path,
        **kwargs: Any,  # noqa: ANN401
    ) -> None: ...

    def fetch(self) -> Path: ...  # noqa: D102


class TestResources:
    """Test different resource models."""

    @pytest.mark.parametrize(
        ("model_cls", "extra_kwargs"),
        [
            (
                models.SurfaceAtlas,
                {
                    "space": "Yerkes19",
                    "density": "32k",
                    "hemisphere": "left",
                    "resource_type": "surf_atlas",
                },
            ),
            (
                models.VolumeAtlas,
                {
                    "space": "Yerkes19",
                    "resolution": "2mm",
                    "resource_type": "vol_atlas",
                },
            ),
            (
                models.SurfaceTransform,
                {
                    "source_space": "Yerkes19",
                    "target_space": "CIVETNMT",
                    "density": "32k",
                    "hemisphere": "right",
                    "resource_type": "surface_transform",
                    "provider": "test",
                },
            ),
            (
                models.VolumeTransform,
                {
                    "source_space": "Yerkes19",
                    "target_space": "CIVETNMT",
                    "resolution": "2mm",
                    "resource_type": "volume_transform",
                    "provider": "test",
                    "weight": 3.0,
                },
            ),
            (
                models.SurfaceAnnotation,
                {
                    "space": "Yerkes19",
                    "density": "32k",
                    "hemisphere": "left",
                    "label": "myelin",
                },
            ),
            (
                models.VolumeAnnotation,
                {"space": "Yerkes19", "resolution": "2mm", "label": "myelin"},
            ),
        ],
    )
    def test_validate_and_fetch(
        self,
        tmp_file: Path,
        model_cls: Callable[..., ModelProtocol],
        extra_kwargs: dict[str, Any],
    ) -> None:
        """Test resource instantiation and fetching."""
        obj = model_cls(
            name="test",
            description="testing description",
            uri=str(tmp_file),
            file_path=Path("abc"),
            **extra_kwargs,
        )
        assert obj.file_path.name == "abc"
        assert obj.fetch() == tmp_file
        assert obj.file_path == tmp_file

    def test_fetch_existing_file_path(self, tmp_file: Path) -> None:
        """Test resource instantiation and fetching."""
        obj = models.SurfaceAtlas(
            name="test",
            description="testing description",
            space="Yerkes19",
            density="32k",
            hemisphere="left",
            resource_type="surf_atlas",
            file_path=tmp_file,
        )
        assert obj.file_path == tmp_file
        assert obj.fetch() == tmp_file

    def test_missing_file_no_uri(self) -> None:
        """Test missing file raises error."""
        with pytest.raises(FileNotFoundError, match="cannot be fetched"):
            models.SurfaceAtlas(
                name="invalid",
                description="missing",
                file_path="some_random_path",  # type: ignore
                space="Yerkes19",
                density="32k",
                hemisphere="left",
                resource_type="SurfaceAtlas",
            ).fetch()

    @patch("neuromaps_prime.graph.models.download_and_validate")
    def test_fetch_raises_if_download_fails(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        """Test fetch raises if download doesn't produce file."""
        dest = tmp_path / "file.txt"
        obj = models.SurfaceAnnotation(
            name="test",
            file_path=dest,
            uri="https://files.osf.io/v1/resources/abcde",
            space="Yerkes19",
            density="32k",
            hemisphere="left",
            label="myelin",
        )
        mock_download.return_value = None
        with pytest.raises(FileNotFoundError, match="does not exist"):
            obj.fetch()

    def test_description_optional(self, tmp_file: Path) -> None:
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

    def test_init(self, surface: models.SurfaceAtlas) -> None:
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
    ) -> None:
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
    ) -> None:
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

    def test_init(self, tmp_file: Path) -> None:
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

    def test_init_defaults_to_empty(self) -> None:
        """Test that edge initializes with empty transform lists."""
        edge = models.Edge()
        assert len(edge.surface_transforms) == 0
        assert len(edge.volume_transforms) == 0


class TestTransformMetadata:
    """Tests for TransformMetadata class."""

    def test_init_defaults_to_none(self) -> None:
        """TransformMetadata defaults to None for both attributes."""
        meta = models.TransformMetadata()
        assert meta.transforms is None
        assert meta.spaces is None

    def test_init_with_transforms(self) -> None:
        """TransformMetadata stores transforms list."""
        transforms = [
            {
                "source_space": "A",
                "target_space": "B",
                "provider": "RheMap",
                "references": ["Smith et al. 2020"],
                "notes": ["Note 1"],
            }
        ]
        meta = models.TransformMetadata(transforms=transforms)
        assert meta.transforms == transforms
        assert meta.spaces is None

    def test_init_with_spaces(self) -> None:
        """TransformMetadata stores spaces list."""
        spaces = [{"space": "A", "references": ["Space A citation"]}]
        meta = models.TransformMetadata(spaces=spaces)
        assert meta.spaces == spaces
        assert meta.transforms is None

    def test_init_with_both(self) -> None:
        """TransformMetadata stores both transforms and spaces."""
        transforms = [
            {
                "source_space": "A",
                "target_space": "B",
                "provider": "RheMap",
                "references": ["Smith et al. 2020"],
                "notes": ["Note 1"],
            }
        ]
        spaces = [{"space": "A", "references": ["Space A citation"]}]
        meta = models.TransformMetadata(transforms=transforms, spaces=spaces)
        assert meta.transforms == transforms
        assert meta.spaces == spaces


class TestTransformResult:
    """Tests for TransformResult class."""

    def test_init_defaults(self) -> None:
        """TransformResult defaults to None for path and metadata."""
        result = models.TransformResult()
        assert result.path is None
        assert result.metadata is None

    def test_backward_compat_references_empty(self) -> None:
        """References property returns None when no metadata."""
        result = models.TransformResult()
        assert result.references is None

    def test_backward_compat_notes_empty(self) -> None:
        """Notes property returns None when no metadata."""
        result = models.TransformResult()
        assert result.notes is None

    def test_backward_compat_references_flattens_spaces_and_hops(
        self, tmp_path: Path
    ) -> None:
        """References property flattens space and hop references."""
        meta = models.TransformMetadata(
            transforms=[
                {"references": ["Hop ref 1"]},
                {"references": ["Hop ref 2"]},
            ],
            spaces=[
                {"references": ["Space ref 1"]},
            ],
        )
        result = models.TransformResult(output_path=tmp_path / "out.nii", metadata=meta)
        assert result.references == ["Space ref 1", "Hop ref 1", "Hop ref 2"]

    def test_backward_compat_notes_flattens_hops(self, tmp_path: Path) -> None:
        """Notes property flattens hop notes."""
        meta = models.TransformMetadata(
            transforms=[
                {"notes": ["Note 1"]},
                {"notes": ["Note 2"]},
            ],
        )
        result = models.TransformResult(output_path=tmp_path / "out.nii", metadata=meta)
        assert result.notes == ["Note 1", "Note 2"]

    def test_backward_compat_references_returns_none_when_empty(
        self, tmp_path: Path
    ) -> None:
        """References returns None when all lists are empty."""
        meta = models.TransformMetadata(transforms=[{"references": []}], spaces=[])
        result = models.TransformResult(output_path=tmp_path / "out.nii", metadata=meta)
        assert result.references is None

    def test_backward_compat_notes_returns_none_when_empty(
        self, tmp_path: Path
    ) -> None:
        """Notes returns None when all lists are empty."""
        meta = models.TransformMetadata(transforms=[{"notes": []}])
        result = models.TransformResult(output_path=tmp_path / "out.nii", metadata=meta)
        assert result.notes is None

    def test_repr_shows_hop_and_space_counts(self, tmp_path: Path) -> None:
        """__repr__ shows hop count and space count."""
        meta = models.TransformMetadata(
            transforms=[{"references": []}, {"references": []}],
            spaces=[{"references": []}, {"references": []}, {"references": []}],
        )
        result = models.TransformResult(output_path=tmp_path / "out.nii", metadata=meta)
        r = repr(result)
        assert "hops=2" in r
        assert "spaces=3" in r

    def test_repr_shows_zero_when_no_metadata(self, tmp_path: Path) -> None:
        """__repr__ shows zero counts when no metadata."""
        result = models.TransformResult(output_path=tmp_path / "out.nii")
        r = repr(result)
        assert "hops=0" in r
        assert "spaces=0" in r

    def test_eq_with_transform_result(self, tmp_path: Path) -> None:
        """Equality compares path and metadata."""
        meta = models.TransformMetadata(transforms=[{"references": ["Ref"]}], spaces=[])
        result_a = models.TransformResult(
            output_path=tmp_path / "out.nii", metadata=meta
        )
        result_b = models.TransformResult(
            output_path=tmp_path / "out.nii", metadata=meta
        )
        result_c = models.TransformResult(output_path=tmp_path / "out.nii")
        assert result_a == result_b
        assert result_a != result_c

    def test_eq_with_path(self, tmp_path: Path) -> None:
        """Equality with Path only compares output path."""
        p = tmp_path / "out.nii"
        result = models.TransformResult(output_path=p)
        assert result == p


class TestTransformResultSaveMetadata:
    """Tests for TransformResult.save_metadata."""

    def _make_result(self, tmp_path: Path) -> models.TransformResult:
        """Create a TransformResult with sample metadata."""
        meta = models.TransformMetadata(
            transforms=[
                {
                    "source_space": "A",
                    "target_space": "B",
                    "provider": "RheMap",
                    "references": ["Smith et al. 2020"],
                    "notes": ["Test note"],
                }
            ],
            spaces=[{"space": "A", "references": ["Space A citation"]}],
        )
        return models.TransformResult(output_path=tmp_path / "out.gii", metadata=meta)

    def test_raises_value_error_when_path_is_none(self) -> None:
        """save_metadata raises ValueError when output path is None."""
        result = models.TransformResult()
        with pytest.raises(ValueError, match="output path is None"):
            result.save_metadata()

    def test_raises_not_implemented_for_unsupported_extension(
        self, tmp_path: Path
    ) -> None:
        """save_metadata raises NotImplementedError for unsupported extensions."""
        result = self._make_result(tmp_path)
        with pytest.raises(NotImplementedError, match="Unsupported file extension"):
            result.save_metadata(metadata_path=tmp_path / "out.txt")

    def test_default_creates_md_file(self, tmp_path: Path) -> None:
        """By default, save_metadata creates a .md file."""
        result = self._make_result(tmp_path)
        meta_path = result.save_metadata()
        assert meta_path == tmp_path / "out.gii.md"
        assert meta_path.exists()

    def test_md_contains_header_and_sections(self, tmp_path: Path) -> None:
        """Markdown output contains transformation header, spaces, and transforms."""
        result = self._make_result(tmp_path)
        result.save_metadata()
        content = (tmp_path / "out.gii.md").read_text()
        assert "# Transformation:" in content
        assert "A -> B" in content
        assert "## Spaces" in content
        assert "## Transforms" in content
        assert "**References:**" in content
        assert "**Caveats:**" in content

    def test_md_contains_space_references(self, tmp_path: Path) -> None:
        """Markdown output includes space-level references."""
        result = self._make_result(tmp_path)
        result.save_metadata()
        content = (tmp_path / "out.gii.md").read_text()
        assert "Space A citation" in content

    def test_md_contains_hop_references_and_notes(self, tmp_path: Path) -> None:
        """Markdown output includes hop-level references and notes."""
        result = self._make_result(tmp_path)
        result.save_metadata()
        content = (tmp_path / "out.gii.md").read_text()
        assert "Smith et al. 2020" in content
        assert "Test note" in content

    def test_md_contains_timestamp(self, tmp_path: Path) -> None:
        """Markdown output includes a timestamp footer."""
        result = self._make_result(tmp_path)
        result.save_metadata()
        content = (tmp_path / "out.gii.md").read_text()
        assert "Generated at" in content

    def test_json_file_creation(self, tmp_path: Path) -> None:
        """save_metadata creates a .json file when extension is .json."""
        result = self._make_result(tmp_path)
        meta_path = result.save_metadata(metadata_path=tmp_path / "out.json")
        assert meta_path == tmp_path / "out.json"
        assert meta_path.exists()

    def test_json_contains_expected_keys(self, tmp_path: Path) -> None:
        """JSON output contains expected top-level keys."""
        import json

        result = self._make_result(tmp_path)
        result.save_metadata(metadata_path=tmp_path / "out.json")
        data = json.loads((tmp_path / "out.json").read_text())
        assert "output_file" in data
        assert "transform_path" in data
        assert "spaces" in data
        assert "transforms" in data
        assert "timestamp" in data

    def test_json_transform_path(self, tmp_path: Path) -> None:
        """JSON transform_path is derived from hop metadata."""
        import json

        result = self._make_result(tmp_path)
        result.save_metadata(metadata_path=tmp_path / "out.json")
        data = json.loads((tmp_path / "out.json").read_text())
        assert data["transform_path"] == ["A", "B"]

    def test_json_spaces_content(self, tmp_path: Path) -> None:
        """JSON spaces contain space-level references."""
        import json

        result = self._make_result(tmp_path)
        result.save_metadata(metadata_path=tmp_path / "out.json")
        data = json.loads((tmp_path / "out.json").read_text())
        assert data["spaces"][0]["space"] == "A"
        assert "Space A citation" in data["spaces"][0]["references"]

    def test_json_transforms_content(self, tmp_path: Path) -> None:
        """JSON transforms contain hop-level references and notes."""
        import json

        result = self._make_result(tmp_path)
        result.save_metadata(metadata_path=tmp_path / "out.json")
        data = json.loads((tmp_path / "out.json").read_text())
        hop = data["transforms"][0]
        assert hop["source_space"] == "A"
        assert hop["target_space"] == "B"
        assert hop["provider"] == "RheMap"
        assert "Smith et al. 2020" in hop["references"]
        assert "Test note" in hop["notes"]

    def test_json_timestamp_is_iso_format(self, tmp_path: Path) -> None:
        """JSON timestamp is ISO 8601 format."""
        import json
        from datetime import datetime

        result = self._make_result(tmp_path)
        result.save_metadata(metadata_path=tmp_path / "out.json")
        data = json.loads((tmp_path / "out.json").read_text())
        ts = data["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_returns_path_to_written_file(self, tmp_path: Path) -> None:
        """save_metadata returns the path to the written file."""
        result = self._make_result(tmp_path)
        meta_path = result.save_metadata()
        assert meta_path.exists()
        assert meta_path.suffix == ".md"

    def test_no_metadata_produces_minimal_output(self, tmp_path: Path) -> None:
        """save_metadata with no metadata produces minimal output."""
        result = models.TransformResult(output_path=tmp_path / "out.gii")
        result.save_metadata()
        content = (tmp_path / "out.gii.md").read_text()
        assert "# Transformation:" in content
        assert "## Spaces" not in content
        assert "## Transforms" not in content
