"""Models for resources in the neuromaps_prime graph."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from neuromaps_prime.fetcher import download_and_validate

_logger = logging.getLogger(__name__)


class Resource(BaseModel):
    """Base model for resources in the neuromaps_prime graph."""

    name: str
    description: str | None
    file_path: Path
    uri: str | None = None
    references: Sequence[str | dict[str, str]] | None = None
    notes: Sequence[str] | None = None

    def fetch(self) -> Path:
        """Return the path to this resource's file, downloading if necessary.

        Returns:
            Path to the resource file.

        Raises:
            FileNotFoundError: if file cannot be fetched
        """
        if not self.file_path.exists():
            if self.uri is None:
                raise FileNotFoundError("File does not exist and cannot be fetched.")
            _logger.info(f"Fetching {self.file_path.name} from remote server.")
            download_and_validate(uri=self.uri, dest=self.file_path)
            if not self.file_path.exists():
                raise FileNotFoundError("File does not exist.")
        return self.file_path

    def __repr__(self) -> str:
        """Custom string representation for debugging."""
        return self.name  # pragma: nocover


class SurfaceAtlas(Resource):
    """Model for surface atlas resources."""

    space: str
    density: str
    hemisphere: Literal["left", "right"]
    resource_type: str


class SurfaceTransform(Resource):
    """Model for surface transform resources."""

    source_space: str
    target_space: str
    density: str
    hemisphere: Literal["left", "right"]
    resource_type: str
    provider: str
    weight: float = 1.0


class SurfaceAnnotation(Resource):
    """Model for surface annotation."""

    description: str | None = None
    space: str
    label: str
    density: str
    hemisphere: Literal["left", "right"]


class VolumeAtlas(Resource):
    """Model for volume atlas resources."""

    space: str
    resolution: str
    resource_type: str


class VolumeTransform(Resource):
    """Model for volume transform resources."""

    source_space: str
    target_space: str
    resolution: str
    resource_type: str
    provider: str
    weight: float = 1.0


class VolumeAnnotation(Resource):
    """Model for volume annotation resources."""

    description: str | None = None
    space: str
    label: str
    resolution: str


class Node(BaseModel):
    """Node representation in transformation graph."""

    name: str
    species: str
    description: str
    references: Sequence[str | dict[str, str]] | None = None
    surfaces: Sequence[SurfaceAtlas] = Field(default_factory=list)
    volumes: Sequence[VolumeAtlas] = Field(default_factory=list)
    surface_annotations: Sequence[SurfaceAnnotation] = Field(default_factory=list)
    volume_annotations: Sequence[VolumeAnnotation] = Field(default_factory=list)

    def __repr__(self) -> str:
        """String representation."""
        surface_str = "\n".join(s.name for s in self.surfaces)
        volume_str = "\n".join(v.name for v in self.volumes)
        surface_annot_str = "\n".join(s.name for s in self.surface_annotations)
        volume_annot_str = "\n".join(v.name for v in self.volume_annotations)
        return (
            "\nNode:"
            f"\n\tname={self.name},\n"
            f"\tspecies={self.species}\n"
            f"\tdescription={self.description}\n"
            f"\tsurfaces=[{surface_str}]\n"
            f"\tvolumes=[{volume_str}]"
            f"\tsurface annotations=[{surface_annot_str}]\n"
            f"\tvolume annotations=[{volume_annot_str}]\n"
        )


class Edge(BaseModel):
    """Edge representation in transformation graph."""

    surface_transforms: Sequence[SurfaceTransform] = Field(default_factory=list)
    volume_transforms: Sequence[VolumeTransform] = Field(default_factory=list)

    def __repr__(self) -> str:
        """String representation."""
        surface_str = "\n".join(s.name for s in self.surface_transforms)
        volume_str = "\n".join(v.name for v in self.volume_transforms)
        return f"\nEdge:\n\tsurfaces=[{surface_str}],\n\tvolumes=[{volume_str}]"
