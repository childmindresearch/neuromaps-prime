"""Surface-to-surface transformation operations for NeuromapsGraph.

Handles single-hop, multi-hop, and resampling (metric/label) surface
transformations. All resource lookups go through GraphCache directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, PrivateAttr

from neuromaps_prime.graph.cache import GraphCache  # noqa: TC001 (pydantic req'd)
from neuromaps_prime.graph.models import SurfaceTransform
from neuromaps_prime.graph.utils import GraphUtils  # noqa: TC001 (pydantic req'd)
from neuromaps_prime.transforms.surface import (
    label_resample,
    metric_resample,
    surface_sphere_project_unproject,
)
from neuromaps_prime.transforms.utils import (
    estimate_surface_density,
    validate_surface_file,
)


class SurfaceTransformOps(BaseModel):
    """Surface-to-surface transformation operations.

    Orchestrates single-hop fetches, multi-hop composition, and
    metric/label resampling. Writes composed transforms back into the
    cache and graph via the injected helpers.

    Attributes:
    ----------
    cache:
        The :class:`GraphCache` instance — used for all resource lookups and
        receives newly composed transforms.
    utils:
        The :class:`GraphUtils` instance for path-finding and density helpers.
    surface_to_surface_key:
        Edge key used for surface-to-surface edges in the graph.
    """

    model_config = {"arbitrary_types_allowed": True}

    cache: GraphCache
    utils: GraphUtils
    surface_to_surface_key: str = "surface_to_surface"
    _logger: logging.Logger = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Post initialization steps.

        Lazily grab logger from Graph initialization.
        """
        self._logger = logging.getLogger("neuromaps-PRIME")

    # ------------------------------------------------------------------ #
    # Surface-to-surface                                                 #
    # ------------------------------------------------------------------ #

    def transform_surface(
        self,
        transformer_type: Literal["metric", "label"],
        input_file: Path,
        source_space: str,
        target_space: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        source_density: str | None = None,
        target_density: str | None = None,
        area_resource: str = "midthickness",
        *,
        add_edge: bool = True,
        provider: str | None = None,
    ) -> Path | None:
        """Resample a metric or label GIFTI from source_space to target_space.

        Two-stage process:
          1. Resolve/compose the sphere-to-sphere transform.
          2. Resample the data file using area-corrected interpolation.

        Args:
            transformer_type: ``'metric'`` or ``'label'``.
            input_file: Input GIFTI file to resample.
            source_space: Source brain template space.
            target_space: Target brain template space.
            hemisphere: ``'left'`` or ``'right'``.
            output_file_path: Path for the resampled output file.
            source_density: Source mesh density. Estimated from input_file
                when ``None``.
            target_density: Target mesh density. Highest available used when
                ``None``.
            area_resource: Surface type used for area correction
                (default ``'midthickness'``).
            add_edge: Whether to cache and register composed multi-hop
                transforms as new graph edges.
            provider: Optional provider name. Falls back to the first
                registered provider when ``None``.

        Returns:
            Path to the resampled output file, or ``None`` if the sphere
            transform could not be resolved.

        Raises:
            ValueError: If transformer_type is invalid, or required surface
                resources are missing.
            FileNotFoundError: If input_file does not exist.
        """
        if transformer_type not in ("metric", "label"):
            raise ValueError(
                f"Invalid transformer_type: '{transformer_type}'. "
                "Must be 'metric' or 'label'."
            )
        validate_surface_file(input_file)
        self.utils.validate_spaces(source_space, target_space)

        source_density = source_density or estimate_surface_density(input_file)

        sphere_transform = self._resolve_sphere_transform(
            source=source_space,
            target=target_space,
            density=source_density,
            hemisphere=hemisphere,
            output_file_path=output_file_path,
            add_edge=add_edge,
            provider=provider,
        )
        if sphere_transform is None:
            return None

        target_density = target_density or self.utils.find_highest_density(
            space=target_space
        )

        new_sphere = self.cache.require_surface_atlas(
            space=target_space,
            density=target_density,
            hemisphere=hemisphere,
            resource_type="sphere",
        ).fetch()

        current_area = self.cache.require_surface_atlas(
            space=source_space,
            density=source_density,
            hemisphere=hemisphere,
            resource_type=area_resource,
        ).fetch()

        new_area = self.cache.require_surface_atlas(
            space=target_space,
            density=target_density,
            hemisphere=hemisphere,
            resource_type=area_resource,
        ).fetch()

        match transformer_type:
            case "label":
                return label_resample(
                    input_file_path=input_file,
                    current_sphere=sphere_transform.fetch(),
                    new_sphere=new_sphere,
                    method="ADAP_BARY_AREA",
                    area_surfs={"current-area": current_area, "new-area": new_area},
                    output_file_path=output_file_path,
                ).label_out
            case "metric":
                return metric_resample(
                    input_file_path=input_file,
                    current_sphere=sphere_transform.fetch(),
                    new_sphere=new_sphere,
                    method="ADAP_BARY_AREA",
                    area_surfs={"current-area": current_area, "new-area": new_area},
                    output_file_path=output_file_path,
                ).metric_out

    # ------------------------------------------------------------------ #
    # Sphere transform resolution                                          #
    # ------------------------------------------------------------------ #

    def _resolve_sphere_transform(
        self,
        source: str,
        target: str,
        density: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        *,
        add_edge: bool = True,
        provider: str | None = None,
    ) -> SurfaceTransform | None:
        """Return the sphere transform from source to target, composing if needed.

        Args:
            source: Source space name.
            target: Target space name.
            density: Surface mesh density.
            hemisphere: ``'left'`` or ``'right'``.
            output_file_path: Base path used for intermediate output files.
            add_edge: Whether to register composed transforms.
            provider: Optional provider name. Falls back to the first
                registered provider when ``None``.

        Returns:
            Resolved :class:`SurfaceTransform`, or ``None`` if no path exists.

        Raises:
            ValueError: If source and target are the same, or no path exists.
        """
        if source == target:
            raise ValueError(f"Source and target spaces are the same: '{source}'")

        path = self.utils.find_path(
            source=source, target=target, edge_type=self.surface_to_surface_key
        )
        if len(path) < 2:
            raise ValueError(f"No valid surface path from '{source}' to '{target}'")

        if len(path) == 2:
            return self.cache.get_surface_transform(
                source=source,
                target=target,
                density=density,
                hemisphere=hemisphere,
                resource_type="sphere",
                provider=provider,
            )

        return self._compose_multihop(
            path=path,
            density=density,
            hemisphere=hemisphere,
            output_file_path=output_file_path,
            add_edge=add_edge,
            provider=provider,
        )

    # ------------------------------------------------------------------ #
    # Multi-hop composition                                                #
    # ------------------------------------------------------------------ #

    def _compose_multihop(
        self,
        path: list[str],
        density: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        *,
        add_edge: bool,
        provider: str | None = None,
    ) -> SurfaceTransform:
        """Compose sphere transforms along a multi-hop path.

        Args:
            path: Ordered list of space names from source to target.
            density: Surface mesh density.
            hemisphere: ``'left'`` or ``'right'``.
            output_file_path: Base path for intermediate output files.
            add_edge: Whether to register composed transforms.
            provider: Optional provider name for the first hop lookup.

        Returns:
            The final composed :class:`SurfaceTransform` from path[0] to
            path[-1].

        Raises:
            ValueError: If any intermediate transform is missing.
        """
        current_transform = self.cache.get_surface_transform(
            source=path[0],
            target=path[1],
            density=density,
            hemisphere=hemisphere,
            resource_type="sphere",
            provider=provider,
        )
        if current_transform is None:
            raise ValueError(
                f"No surface transform found from '{path[0]}' to '{path[1]}'"
            )

        for hop_idx, next_space in enumerate(path[2:], start=2):
            current_transform = self._compose_next_hop(
                path=path,
                hop_idx=hop_idx,
                next_space=next_space,
                current_transform=current_transform,
                source=path[0],
                density=density,
                hemisphere=hemisphere,
                output_file_path=output_file_path,
                add_edge=add_edge,
                provider=provider,
            )

        return current_transform

    def _compose_next_hop(
        self,
        path: list[str],
        hop_idx: int,
        next_space: str,
        current_transform: SurfaceTransform,
        source: str,
        density: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        *,
        add_edge: bool,
        provider: str | None = None,
    ) -> SurfaceTransform:
        """Extend current_transform by one hop towards next_space.

        Args:
            path: Full transformation path.
            hop_idx: Current position in path (2-based).
            next_space: The space being added in this hop.
            current_transform: Accumulated transform so far.
            source: Original source space (used for naming).
            density: Surface mesh density.
            hemisphere: ``'left'`` or ``'right'``.
            output_file_path: Base path for output files.
            add_edge: Whether to register the new transform.
            provider: Optional provider name for intermediate lookups.

        Returns:
            New :class:`SurfaceTransform` from source to next_space.
        """
        hop_output = self._hop_output_path(
            output_file_path=output_file_path,
            source=source,
            next_target=next_space,
            density=density,
            hemisphere=hemisphere,
        )
        composed_path = self._two_hops(
            source_space=path[hop_idx - 2],
            mid_space=path[hop_idx - 1],
            target_space=next_space,
            density=density,
            hemisphere=hemisphere,
            output_file_path=hop_output,
            first_transform=current_transform,
            provider=provider,
        )
        new_transform = SurfaceTransform(
            name=f"{source}_to_{next_space}_{density}_{hemisphere}_sphere",
            description=f"Surface transform from '{source}' to '{next_space}'",
            source_space=source,
            target_space=next_space,
            density=density,
            hemisphere=hemisphere,
            resource_type="sphere",
            file_path=composed_path,
            weight=float(hop_idx),
            provider=provider or "",
        )
        if add_edge:
            self.cache.add_surface_transform(new_transform)

        return new_transform

    def _two_hops(
        self,
        source_space: str,
        mid_space: str,
        target_space: str,
        density: str,
        hemisphere: Literal["left", "right"],
        output_file_path: str,
        first_transform: SurfaceTransform | None = None,
        provider: str | None = None,
    ) -> Path:
        """Compose two sphere transforms via project-unproject through mid_space.

        Args:
            source_space: Source space name.
            mid_space: Intermediate space name.
            target_space: Target space name.
            density: Surface mesh density.
            hemisphere: ``'left'`` or ``'right'``.
            output_file_path: Path for the composed output sphere.
            first_transform: Pre-fetched source→mid transform. Fetched from
                cache when ``None``.
            provider: Optional provider name. Falls back to the first
                registered provider when ``None``.

        Returns:
            Path to the composed output sphere file.

        Raises:
            ValueError: If any required transform or atlas is missing.
        """
        first_transform = first_transform or self.cache.get_surface_transform(
            source=source_space,
            target=mid_space,
            density=density,
            hemisphere=hemisphere,
            resource_type="sphere",
            provider=provider,
        )
        if first_transform is None:
            raise ValueError(
                f"No surface transform found from {source_space!r} to {mid_space!r}"
            )
        if provider is not None and first_transform.provider != provider:
            self._logger.warning(
                f"Provider {provider!r} not found for hop {source_space!r} to "
                f"{mid_space!r}; falling back to {first_transform.provider!r}. The "
                "composed transform will use mixed providers.",
            )
        sphere_in = first_transform.fetch()

        common_density = self.utils.find_common_density(mid_space, target_space)
        mid_atlas = self.cache.get_surface_atlas(
            space=mid_space,
            density=common_density,
            hemisphere=hemisphere,
            resource_type="sphere",
        )
        if mid_atlas is None:
            raise ValueError(
                f"No sphere atlas found for '{mid_space}' at density '{common_density}'"
            )
        sphere_project_to = mid_atlas.fetch()

        unproject_transform = self.cache.get_surface_transform(
            source=mid_space,
            target=target_space,
            density=common_density,
            hemisphere=hemisphere,
            resource_type="sphere",
            provider=provider,
        )
        if unproject_transform is None:
            raise ValueError(
                f"No surface transform found from {mid_space!r} to {target_space!r}"
            )
        if provider is not None and unproject_transform.provider != provider:
            self._logger.warning(
                f"Provider {provider!r} not found for hop {mid_space!r} to "
                f"{target_space!r}; falling back to {unproject_transform.provider!r}. "
                "The composed transform will use mixed providers.",
            )
        sphere_unproject_from = unproject_transform.fetch()

        return surface_sphere_project_unproject(
            sphere_in=sphere_in,
            sphere_project_to=sphere_project_to,
            sphere_unproject_from=sphere_unproject_from,
            sphere_out=output_file_path,
        ).sphere_out

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _hop_output_path(
        self,
        output_file_path: str,
        source: str,
        next_target: str,
        density: str,
        hemisphere: Literal["left", "right"],
    ) -> str:
        """Build a deterministic intermediate file path for a single hop.

        Args:
            output_file_path: Final output path (parent directory is reused).
            source: Original source space name.
            next_target: Space being reached in this hop.
            density: Surface mesh density.
            hemisphere: ``'left'`` or ``'right'``.

        Returns:
            Absolute path string for the intermediate sphere file.
        """
        parent = Path(output_file_path).parent
        fname = (
            f"src-{source}_"
            f"to-{next_target}_"
            f"den-{density}_"
            f"hemi-{hemisphere[0].upper()}_"
            f"sphere.surf.gii"
        )
        return str(parent / fname)
