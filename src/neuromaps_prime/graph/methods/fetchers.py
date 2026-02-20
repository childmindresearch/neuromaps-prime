"""Fetcher layer for NeuromapsGraph atlas and transform resources.

Thin read-only accessors over GraphCache that mirror the public fetch_*
API from the original NeuromapsGraph class. Keeping fetchers separate from
the cache means the cache stays a pure data container while this layer owns
the public-facing argument signatures and any fetch-time logic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from neuromaps_prime.graph.methods.cache import GraphCache
from neuromaps_prime.graph.models import (
    SurfaceAtlas,
    SurfaceTransform,
    VolumeAtlas,
    VolumeTransform,
)


class GraphFetchers(BaseModel):
    """Read-only accessors over a :class:`GraphCache` instance.

    All methods return the requested resource or ``None`` if no matching
    entry exists — callers are responsible for raising domain errors.

    Attributes:
    ----------
    cache:
        The :class:`GraphCache` instance to query.
    """

    model_config = {"arbitrary_types_allowed": True}

    cache: GraphCache

    # Surface atlas
    def fetch_surface_atlas(
        self,
        space: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
    ) -> SurfaceAtlas | None:
        """Fetch a surface atlas resource.

        Args:
            space: Brain template space name (e.g. ``'MNI152'``).
            density: Surface mesh density (e.g. ``'32k'``, ``'41k'``).
            hemisphere: ``'left'`` or ``'right'``.
            resource_type: Surface type (e.g. ``'sphere'``, ``'midthickness'``).

        Returns:
            The matching :class:`SurfaceAtlas`, or ``None`` if not found.
        """
        return self.cache.get_surface_atlas(
            space=space,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )

    def fetch_surface_atlases(
        self,
        space: str,
        density: str | None = None,
        hemisphere: Literal["left", "right"] | None = None,
        resource_type: str | None = None,
    ) -> list[SurfaceAtlas]:
        """Fetch all surface atlases for a space, with optional filters.

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
            for key, atlas in self.cache.surface_atlas.items()
            if key[0] == space
            and (density is None or key[1] == density)
            and (hemisphere is None or key[2] == hemisphere.lower())
            and (resource_type is None or key[3] == resource_type)
        ]

    # Surface transform
    def fetch_surface_transform(
        self,
        source: str,
        target: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
    ) -> SurfaceTransform | None:
        """Fetch a surface-to-surface transform resource.

        Args:
            source: Source brain template space name.
            target: Target brain template space name.
            density: Surface mesh density (e.g. ``'32k'``).
            hemisphere: ``'left'`` or ``'right'``.
            resource_type: Surface type (e.g. ``'sphere'``).

        Returns:
            The matching :class:`SurfaceTransform`, or ``None`` if not found.
        """
        return self.cache.get_surface_transform(
            source=source,
            target=target,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )

    def fetch_surface_transforms(
        self,
        source: str,
        target: str,
        density: str | None = None,
        hemisphere: Literal["left", "right"] | None = None,
        resource_type: str | None = None,
    ) -> list[SurfaceTransform]:
        """Fetch all surface transforms between two spaces, with optional filters.

        Args:
            source: Source brain template space name.
            target: Target brain template space name.
            density: Optional density filter.
            hemisphere: Optional hemisphere filter.
            resource_type: Optional resource type filter.

        Returns:
            All matching :class:`SurfaceTransform` entries (may be empty).
        """
        return [
            transform
            for key, transform in self.cache.surface_transform.items()
            if key[0] == source
            and key[1] == target
            and (density is None or key[2] == density)
            and (hemisphere is None or key[3] == hemisphere.lower())
            and (resource_type is None or key[4] == resource_type)
        ]

    # Volume atlas
    def fetch_volume_atlas(
        self,
        space: str,
        resolution: str,
        resource_type: str,
    ) -> VolumeAtlas | None:
        """Fetch a volume atlas resource.

        Args:
            space: Brain template space name.
            resolution: Volume resolution (e.g. ``'1mm'``, ``'2mm'``).
            resource_type: Volume type (e.g. ``'T1w'``, ``'T2w'``).

        Returns:
            The matching :class:`VolumeAtlas`, or ``None`` if not found.
        """
        return self.cache.get_volume_atlas(
            space=space,
            resolution=resolution,
            resource_type=resource_type,
        )

    def fetch_volume_atlases(  # pragma: no cover (already tested)
        self,
        space: str,
        resolution: str | None = None,
        resource_type: str | None = None,
    ) -> list[VolumeAtlas]:
        """Fetch all volume atlases for a space, with optional filters.

        Args:
            space: Brain template space name.
            resolution: Optional resolution filter.
            resource_type: Optional resource type filter.

        Returns:
            All matching :class:`VolumeAtlas` entries (may be empty).
        """
        return [
            atlas
            for key, atlas in self.cache.volume_atlas.items()
            if key[0] == space
            and (resolution is None or key[1] == resolution)
            and (resource_type is None or key[2] == resource_type)
        ]

    # Volume transform
    def fetch_volume_transform(
        self,
        source: str,
        target: str,
        resolution: str,
        resource_type: str,
    ) -> VolumeTransform | None:
        """Fetch a volume-to-volume transform resource.

        Args:
            source: Source brain template space name.
            target: Target brain template space name.
            resolution: Volume resolution (e.g. ``'2mm'``).
            resource_type: Volume type (e.g. ``'T1w'``, ``'composite'``).

        Returns:
            The matching :class:`VolumeTransform`, or ``None`` if not found.
        """
        return self.cache.get_volume_transform(
            source=source,
            target=target,
            resolution=resolution,
            resource_type=resource_type,
        )

    def fetch_volume_transforms(  # pragma: no cover (already tested)
        self,
        source: str,
        target: str,
        resolution: str | None = None,
        resource_type: str | None = None,
    ) -> list[VolumeTransform]:
        """Fetch all volume transforms between two spaces, with optional filters.

        Args:
            source: Source brain template space name.
            target: Target brain template space name.
            resolution: Optional resolution filter.
            resource_type: Optional resource type filter.

        Returns:
            All matching :class:`VolumeTransform` entries (may be empty).
        """
        return [
            transform
            for key, transform in self.cache.volume_transform.items()
            if key[0] == source
            and key[1] == target
            and (resolution is None or key[2] == resolution)
            and (resource_type is None or key[3] == resource_type)
        ]
