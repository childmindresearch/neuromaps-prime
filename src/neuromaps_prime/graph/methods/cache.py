"""Cache layer for NeuromapsGraph atlas and transform resources.

Provides O(1) keyed lookups for all resource types.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from neuromaps_prime.graph.models import (
    SurfaceAtlas,
    SurfaceTransform,
    VolumeAtlas,
    VolumeTransform,
)

# Key type aliases
SurfaceAtlasKey = tuple[str, str, str, str]
SurfaceTransformKey = tuple[str, str, str, str, str]
VolumeAtlasKey = tuple[str, str, str]
VolumeTransformKey = tuple[str, str, str, str]


class GraphCache(BaseModel):
    """Container for all atlas and transform lookup tables.

    All dictionaries are keyed by stable tuples so that lookups are O(1).
    The cache is intentionally mutable: the graph builder populates it during
    construction and transform operations may extend it with composed
    multi-hop transforms at runtime.

    Attributes:
    ----------
    surface_atlas:
        Maps ``(space, density, hemisphere, resource_type)`` to a
        :class:`SurfaceAtlas`.
    surface_transform:
        Maps ``(source, target, density, hemisphere, resource_type)`` to a
        :class:`SurfaceTransform`.
    volume_atlas:
        Maps ``(space, resolution, resource_type)`` to a
        :class:`VolumeAtlas`.
    volume_transform:
        Maps ``(source, target, resolution, resource_type)`` to a
        :class:`VolumeTransform`.
    """

    model_config = {"arbitrary_types_allowed": True}

    surface_atlas: dict[SurfaceAtlasKey, SurfaceAtlas] = Field(default_factory=dict)
    surface_transform: dict[SurfaceTransformKey, SurfaceTransform] = Field(
        default_factory=dict
    )
    volume_atlas: dict[VolumeAtlasKey, VolumeAtlas] = Field(default_factory=dict)
    volume_transform: dict[VolumeTransformKey, VolumeTransform] = Field(
        default_factory=dict
    )

    # Surface atlas
    def add_surface_atlas(self, atlas: SurfaceAtlas) -> None:
        """Insert or overwrite a surface atlas entry."""
        self.surface_atlas[
            (atlas.space, atlas.density, atlas.hemisphere.lower(), atlas.resource_type)
        ] = atlas

    def get_surface_atlas(
        self,
        space: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
    ) -> SurfaceAtlas | None:
        """Return the matching :class:`SurfaceAtlas`, or ``None``."""
        return self.surface_atlas.get(
            (space, density, hemisphere.lower(), resource_type)
        )

    # Surface transform
    def add_surface_transform(self, transform: SurfaceTransform) -> None:
        """Insert or overwrite a surface transform entry."""
        self.surface_transform[
            (
                transform.source_space,
                transform.target_space,
                transform.density,
                transform.hemisphere.lower(),
                transform.resource_type,
            )
        ] = transform

    def get_surface_transform(
        self,
        source: str,
        target: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
    ) -> SurfaceTransform | None:
        """Return the matching :class:`SurfaceTransform`, or ``None``."""
        return self.surface_transform.get(
            (source, target, density, hemisphere.lower(), resource_type)
        )

    # Volume atlas
    def add_volume_atlas(self, atlas: VolumeAtlas) -> None:
        """Insert or overwrite a volume atlas entry."""
        self.volume_atlas[(atlas.space, atlas.resolution, atlas.resource_type)] = atlas

    def get_volume_atlas(
        self, space: str, resolution: str, resource_type: str
    ) -> VolumeAtlas | None:
        """Return the matching :class:`VolumeAtlas`, or ``None``."""
        return self.volume_atlas.get((space, resolution, resource_type))

    # Volume transforms
    def add_volume_transform(self, transform: VolumeTransform) -> None:
        """Insert or overwrite a volume transform entry."""
        self.volume_transform[
            (
                transform.source_space,
                transform.target_space,
                transform.resolution,
                transform.resource_type,
            )
        ] = transform

    def get_volume_transform(
        self, source: str, target: str, resolution: str, resource_type: str
    ) -> VolumeTransform | None:
        """Return the matching :class:`VolumeTransform`, or ``None``."""
        return self.volume_transform.get((source, target, resolution, resource_type))

    # Helpers (used by GraphBuilder)
    def add_surface_atlases(self, atlases: list[SurfaceAtlas]) -> None:
        """Bulk-insert surface atlases."""
        for atlas in atlases:
            self.add_surface_atlas(atlas)

    def add_surface_transforms(self, transforms: list[SurfaceTransform]) -> None:
        """Bulk-insert surface transforms."""
        for transform in transforms:
            self.add_surface_transform(transform)

    def add_volume_atlases(self, atlases: list[VolumeAtlas]) -> None:
        """Bulk-insert volume atlases."""
        for atlas in atlases:
            self.add_volume_atlas(atlas)

    def add_volume_transforms(self, transforms: list[VolumeTransform]) -> None:
        """Bulk-insert volume transforms."""
        for transform in transforms:
            self.add_volume_transform(transform)

    def clear(self) -> None:
        """Evict all entries from every cache table."""
        self.surface_atlas.clear()
        self.surface_transform.clear()
        self.volume_atlas.clear()
        self.volume_transform.clear()
