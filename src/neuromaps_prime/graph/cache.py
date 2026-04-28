"""Cache layer for NeuromapsGraph atlas and transform resources.

Provides O(1) keyed lookups for all resource types, plus filtered list
queries and require_* helpers that raise on a miss.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from neuromaps_prime.graph.models import (
    SurfaceAnnotation,
    SurfaceAtlas,
    SurfaceTransform,
    VolumeAnnotation,
    VolumeAtlas,
    VolumeTransform,
)

# Key type aliases
SurfaceAtlasKey = tuple[str, str, str, str]  # (space, density, hemi, resource_type)
SurfaceTransformKey = tuple[
    str, str, str, str, str, str
]  # (src, tgt, density, hemi, resource_type, provider)
SurfaceAnnotationKey = tuple[str, str, str, str]  # (space, label, density, hemi)
VolumeAtlasKey = tuple[str, str, str]  # (space, resolution, resource_type)
VolumeTransformKey = tuple[
    str, str, str, str, str
]  # (src, tgt, resolution, resource_type, provider)
VolumeAnnotationKey = tuple[str, str, str]  # (space, label, resolution)


class GraphCache(BaseModel):
    """Container for all atlas, transform, and annotation lookup tables.

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
        Maps ``(source, target, density, hemisphere, resource_type, provider)`` to a
        :class:`SurfaceTransform`.
    surface_annotation:
        Maps ``(space, label, density, hemisphere)`` to a
        :class:`SurfaceAnnotation`.
    volume_atlas:
        Maps ``(space, resolution, resource_type)`` to a
        :class:`VolumeAtlas`.
    volume_transform:
        Maps ``(source, target, resolution, resource_type, provider)`` to a
        :class:`VolumeTransform`.
    volume_annotation:
        Maps ``(space, label, resolution)`` to a
        :class:`VolumeAnnotation`.
    """

    model_config = {"arbitrary_types_allowed": True}

    surface_atlas: dict[SurfaceAtlasKey, SurfaceAtlas] = Field(default_factory=dict)
    surface_transform: dict[SurfaceTransformKey, SurfaceTransform] = Field(
        default_factory=dict
    )
    surface_annotation: dict[SurfaceAnnotationKey, SurfaceAnnotation] = Field(
        default_factory=dict
    )
    volume_atlas: dict[VolumeAtlasKey, VolumeAtlas] = Field(default_factory=dict)
    volume_transform: dict[VolumeTransformKey, VolumeTransform] = Field(
        default_factory=dict
    )
    volume_annotation: dict[VolumeAnnotationKey, VolumeAnnotation] = Field(
        default_factory=dict
    )

    # ------------------------------------------------------------------ #
    # Surface atlas                                                        #
    # ------------------------------------------------------------------ #

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

    def get_surface_atlases(
        self,
        space: str,
        density: str | None = None,
        hemisphere: Literal["left", "right"] | None = None,
        resource_type: str | None = None,
    ) -> list[SurfaceAtlas]:
        """Return all surface atlases for *space* with optional filters.

        Args:
            space: Brain template space name.
            density: Optional density filter.
            hemisphere: Optional hemisphere filter.
            resource_type: Optional resource type filter.

        Returns:
            All matching :class:`SurfaceAtlas` entries (may be empty).
        """
        return [
            atlas
            for (sp, d, h, rt), atlas in self.surface_atlas.items()
            if sp == space
            and (density is None or d == density)
            and (hemisphere is None or h == hemisphere.lower())
            and (resource_type is None or rt == resource_type)
        ]

    def require_surface_atlas(
        self,
        space: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
    ) -> SurfaceAtlas:
        """Return the matching :class:`SurfaceAtlas` or raise ``ValueError``.

        Args:
            space: Brain template space name.
            density: Surface mesh density.
            hemisphere: ``'left'`` or ``'right'``.
            resource_type: Surface resource type.

        Returns:
            The matching :class:`SurfaceAtlas`.

        Raises:
            ValueError: If no matching atlas is found.
        """
        atlas = self.get_surface_atlas(space, density, hemisphere, resource_type)
        if atlas is None:
            raise ValueError(
                f"No '{resource_type}' surface atlas found for space '{space}' "
                f"(density='{density}', hemisphere='{hemisphere}')"
            )
        return atlas

    # ------------------------------------------------------------------ #
    # Surface annotation                                                   #
    # ------------------------------------------------------------------ #

    def add_surface_annotation(self, annotation: SurfaceAnnotation) -> None:
        """Insert or overwrite a surface annotation entry."""
        self.surface_annotation[
            (
                annotation.space,
                annotation.label,
                annotation.density,
                annotation.hemisphere.lower(),
            )
        ] = annotation

    def get_surface_annotation(
        self,
        space: str,
        label: str,
        density: str,
        hemisphere: Literal["left", "right"],
    ) -> SurfaceAnnotation | None:
        """Return the matching :class:`SurfaceAnnotation`, or ``None``."""
        return self.surface_annotation.get((space, label, density, hemisphere.lower()))

    def get_surface_annotations(
        self,
        space: str,
        label: str | None = None,
        density: str | None = None,
        hemisphere: Literal["left", "right"] | None = None,
    ) -> list[SurfaceAnnotation]:
        """Return all surface annotations for *space* with optional filters.

        Args:
            space: Brain template space name.
            label: Optional annotation label filter (e.g. ``'myelin'``).
            density: Optional density filter.
            hemisphere: Optional hemisphere filter.

        Returns:
            All matching :class:`SurfaceAnnotation` entries (may be empty).
        """
        return [
            annotation
            for (sp, lb, d, h), annotation in self.surface_annotation.items()
            if sp == space
            and (label is None or lb == label)
            and (density is None or d == density)
            and (hemisphere is None or h == hemisphere.lower())
        ]

    def require_surface_annotation(
        self,
        space: str,
        label: str,
        density: str,
        hemisphere: Literal["left", "right"],
    ) -> SurfaceAnnotation:
        """Return the matching :class:`SurfaceAnnotation` or raise ``ValueError``.

        Args:
            space: Brain template space name.
            label: Annotation label (e.g. ``'myelin'``).
            density: Surface mesh density.
            hemisphere: ``'left'`` or ``'right'``.

        Returns:
            The matching :class:`SurfaceAnnotation`.

        Raises:
            ValueError: If no matching annotation is found.
        """
        annotation = self.get_surface_annotation(space, label, density, hemisphere)
        if annotation is None:
            raise ValueError(
                f"No '{label}' surface annotation found for space '{space}' "
                f"(density='{density}', hemisphere='{hemisphere}')"
            )
        return annotation

    # ------------------------------------------------------------------ #
    # Surface transform                                                    #
    # ------------------------------------------------------------------ #

    def add_surface_transform(self, transform: SurfaceTransform) -> None:
        """Insert or overwrite a surface transform entry."""
        self.surface_transform[
            (
                transform.source_space,
                transform.target_space,
                transform.density,
                transform.hemisphere.lower(),
                transform.resource_type,
                transform.provider,
            )
        ] = transform

    def get_surface_transform(
        self,
        source: str,
        target: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
        provider: str | None = None,
    ) -> SurfaceTransform | None:
        """Return the matching :class:`SurfaceTransform`, or ``None``.

        If *provider* is ``None`` or not found, falls back to the first
        registered transform matching the other fields.
        """
        if provider is not None:
            result = self.surface_transform.get(
                (source, target, density, hemisphere.lower(), resource_type, provider)
            )
            if result is not None:
                return result
        for (src, tgt, d, h, rt, _), transform in self.surface_transform.items():
            if (
                src == source
                and tgt == target
                and d == density
                and h == hemisphere.lower()
                and rt == resource_type
            ):
                return transform
        return None

    def get_surface_transforms(
        self,
        source: str,
        target: str,
        density: str | None = None,
        hemisphere: Literal["left", "right"] | None = None,
        resource_type: str | None = None,
        provider: str | None = None,
    ) -> list[SurfaceTransform]:
        """Return all surface transforms between two spaces with optional filters.

        Args:
            source: Source brain template space name.
            target: Target brain template space name.
            density: Optional density filter.
            hemisphere: Optional hemisphere filter.
            resource_type: Optional resource type filter.
            provider: Optional provider filter.

        Returns:
            All matching :class:`SurfaceTransform` entries (may be empty).
        """
        return [
            transform
            for (src, tgt, d, h, rt, prov), transform in self.surface_transform.items()
            if src == source
            and tgt == target
            and (density is None or d == density)
            and (hemisphere is None or h == hemisphere.lower())
            and (resource_type is None or rt == resource_type)
            and (provider is None or prov == provider)
        ]

    # ------------------------------------------------------------------ #
    # Volume atlas                                                         #
    # ------------------------------------------------------------------ #

    def add_volume_atlas(self, atlas: VolumeAtlas) -> None:
        """Insert or overwrite a volume atlas entry."""
        self.volume_atlas[(atlas.space, atlas.resolution, atlas.resource_type)] = atlas

    def get_volume_atlas(
        self, space: str, resolution: str, resource_type: str
    ) -> VolumeAtlas | None:
        """Return the matching :class:`VolumeAtlas`, or ``None``."""
        return self.volume_atlas.get((space, resolution, resource_type))

    def get_volume_atlases(
        self,
        space: str,
        resolution: str | None = None,
        resource_type: str | None = None,
    ) -> list[VolumeAtlas]:
        """Return all volume atlases for *space* with optional filters.

        Args:
            space: Brain template space name.
            resolution: Optional resolution filter.
            resource_type: Optional resource type filter.

        Returns:
            All matching :class:`VolumeAtlas` entries (may be empty).
        """
        return [
            atlas
            for (sp, res, rt), atlas in self.volume_atlas.items()
            if sp == space
            and (resolution is None or res == resolution)
            and (resource_type is None or rt == resource_type)
        ]

    def require_volume_atlas(
        self, space: str, resolution: str, resource_type: str
    ) -> VolumeAtlas:
        """Return the matching :class:`VolumeAtlas` or raise ``ValueError``.

        Args:
            space: Brain template space name.
            resolution: Voxel resolution string (e.g. ``'250um'``).
            resource_type: Volume resource type.

        Returns:
            The matching :class:`VolumeAtlas`.

        Raises:
            ValueError: If no matching atlas is found.
        """
        atlas = self.get_volume_atlas(space, resolution, resource_type)
        if atlas is None:
            raise ValueError(
                f"No '{resource_type}' volume atlas found for space '{space}' "
                f"(resolution='{resolution}')"
            )
        return atlas

    # ------------------------------------------------------------------ #
    # Volume annotation                                                    #
    # ------------------------------------------------------------------ #

    def add_volume_annotation(self, annotation: VolumeAnnotation) -> None:
        """Insert or overwrite a volume annotation entry."""
        self.volume_annotation[
            (annotation.space, annotation.label, annotation.resolution)
        ] = annotation

    def get_volume_annotation(
        self, space: str, label: str, resolution: str
    ) -> VolumeAnnotation | None:
        """Return the matching :class:`VolumeAnnotation`, or ``None``."""
        return self.volume_annotation.get((space, label, resolution))

    def get_volume_annotations(
        self,
        space: str,
        label: str | None = None,
        resolution: str | None = None,
    ) -> list[VolumeAnnotation]:
        """Return all volume annotations for *space* with optional filters.

        Args:
            space: Brain template space name.
            label: Optional annotation label filter (e.g. ``'myelin'``).
            resolution: Optional resolution filter.

        Returns:
            All matching :class:`VolumeAnnotation` entries (may be empty).
        """
        return [
            annotation
            for (sp, lb, res), annotation in self.volume_annotation.items()
            if sp == space
            and (label is None or lb == label)
            and (resolution is None or res == resolution)
        ]

    def require_volume_annotation(
        self, space: str, label: str, resolution: str
    ) -> VolumeAnnotation:
        """Return the matching :class:`VolumeAnnotation` or raise ``ValueError``.

        Args:
            space: Brain template space name.
            label: Annotation label (e.g. ``'myelin'``).
            resolution: Voxel resolution string (e.g. ``'250um'``).

        Returns:
            The matching :class:`VolumeAnnotation`.

        Raises:
            ValueError: If no matching annotation is found.
        """
        annotation = self.get_volume_annotation(space, label, resolution)
        if annotation is None:
            raise ValueError(
                f"No '{label}' volume annotation found for space '{space}' "
                f"(resolution='{resolution}')"
            )
        return annotation

    # ------------------------------------------------------------------ #
    # Volume transform                                                     #
    # ------------------------------------------------------------------ #

    def add_volume_transform(self, transform: VolumeTransform) -> None:
        """Insert or overwrite a volume transform entry."""
        self.volume_transform[
            (
                transform.source_space,
                transform.target_space,
                transform.resolution,
                transform.resource_type,
                transform.provider,
            )
        ] = transform

    def get_volume_transform(
        self,
        source: str,
        target: str,
        resolution: str,
        resource_type: str,
        provider: str | None = None,
    ) -> VolumeTransform | None:
        """Return the matching :class:`VolumeTransform`, or ``None``.

        If *provider* is ``None`` or not found, falls back to the first
        registered transform matching the other fields.
        """
        if provider is not None:
            result = self.volume_transform.get(
                (source, target, resolution, resource_type, provider)
            )
            if result is not None:
                return result
        for (src, tgt, res, rt, _), transform in self.volume_transform.items():
            if (
                src == source
                and tgt == target
                and res == resolution
                and rt == resource_type
            ):
                return transform
        return None

    def get_volume_transforms(
        self,
        source: str,
        target: str,
        resolution: str | None = None,
        resource_type: str | None = None,
        provider: str | None = None,
    ) -> list[VolumeTransform]:
        """Return all volume transforms between two spaces with optional filters.

        Args:
            source: Source brain template space name.
            target: Target brain template space name.
            resolution: Optional resolution filter.
            resource_type: Optional resource type filter.
            provider: Optional provider filter.

        Returns:
            All matching :class:`VolumeTransform` entries (may be empty).
        """
        return [
            transform
            for (src, tgt, res, rt, prov), transform in self.volume_transform.items()
            if src == source
            and tgt == target
            and (resolution is None or res == resolution)
            and (resource_type is None or rt == resource_type)
            and (provider is None or prov == provider)
        ]

    # ------------------------------------------------------------------ #
    # Bulk helpers (used by GraphBuilder)                                  #
    # ------------------------------------------------------------------ #

    def add_surface_atlases(self, atlases: list[SurfaceAtlas]) -> None:
        """Bulk-insert surface atlases."""
        for atlas in atlases:
            self.add_surface_atlas(atlas)

    def add_surface_annotations(self, annotations: list[SurfaceAnnotation]) -> None:
        """Bulk-insert surface annotations."""
        for annotation in annotations:
            self.add_surface_annotation(annotation)

    def add_surface_transforms(self, transforms: list[SurfaceTransform]) -> None:
        """Bulk-insert surface transforms."""
        for transform in transforms:
            self.add_surface_transform(transform)

    def add_volume_atlases(self, atlases: list[VolumeAtlas]) -> None:
        """Bulk-insert volume atlases."""
        for atlas in atlases:
            self.add_volume_atlas(atlas)

    def add_volume_annotations(self, annotations: list[VolumeAnnotation]) -> None:
        """Bulk-insert volume annotations."""
        for annotation in annotations:
            self.add_volume_annotation(annotation)

    def add_volume_transforms(self, transforms: list[VolumeTransform]) -> None:
        """Bulk-insert volume transforms."""
        for transform in transforms:
            self.add_volume_transform(transform)

    def clear(self) -> None:
        """Evict all entries from every cache table."""
        self.surface_atlas.clear()
        self.surface_transform.clear()
        self.surface_annotation.clear()
        self.volume_atlas.clear()
        self.volume_transform.clear()
        self.volume_annotation.clear()