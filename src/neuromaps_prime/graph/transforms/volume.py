"""Volume-to-volume and volume-to-surface transformation operations.

Handles direct volume warping and the two-stage volume→surface→surface
pipeline. All graph queries go through GraphUtils and GraphFetchers rather
than touching the cache or NetworkX graph directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from niwrap import workbench
from pydantic import BaseModel

from neuromaps_prime.graph.methods.cache import GraphCache
from neuromaps_prime.graph.methods.fetchers import GraphFetchers
from neuromaps_prime.graph.models import SurfaceAtlas
from neuromaps_prime.graph.transforms.surface import SurfaceTransformOps
from neuromaps_prime.graph.utils import GraphUtils
from neuromaps_prime.transforms.utils import validate_volume_file
from neuromaps_prime.transforms.volume import surface_project, vol_to_vol


class VolumeTransformOps(BaseModel):
    """Volume-to-volume and volume-to-surface transformation operations.

    Attributes:
    ----------
    cache:
        The :class:`GraphCache` instance — used indirectly via fetchers.
    fetchers:
        The :class:`GraphFetchers` instance for resource lookups.
    utils:
        The :class:`GraphUtils` instance for validation and density helpers.
    surface_ops:
        The :class:`SurfaceTransformOps` instance for the surface stage of
        volume-to-surface transformations.
    volume_to_volume_key:
        Edge key used for volume-to-volume edges in the graph.
    """

    model_config = {"arbitrary_types_allowed": True}

    cache: GraphCache
    fetchers: GraphFetchers
    utils: GraphUtils
    surface_ops: SurfaceTransformOps
    volume_to_volume_key: str = "volume_to_volume"

    # Volume-to-volume
    def transform_volume(
        self,
        input_file: Path,
        source_space: str,
        target_space: str,
        resolution: str,
        resource_type: str,
        output_file_path: str,
        interp: str = "linear",
        interp_params: dict[str, Any] | None = None,
    ) -> Path:
        """Warp a volume image from source_space to target_space.

        Args:
            input_file: NIfTI volume in source_space to transform.
            source_space: Source brain template space name.
            target_space: Target brain template space name.
            resolution: Target volume resolution (e.g. ``'1mm'``, ``'500um'``).
            resource_type: Volume resource type (e.g. ``'T1w'``, ``'composite'``).
            output_file_path: Path for the warped output volume.
            interp: Interpolation method passed to the warp tool.
            interp_params: Optional additional interpolation parameters.

        Returns:
            Path to the warped output volume.

        Raises:
            FileNotFoundError: If input_file does not exist.
            ValueError: If required transform or reference atlas is missing.
        """
        self.utils.validate_spaces(source_space, target_space)
        validate_volume_file(input_file)

        transform = self.fetchers.fetch_volume_transform(
            source=source_space,
            target=target_space,
            resolution=resolution,
            resource_type=resource_type,
        )
        if transform is None:
            raise ValueError(
                f"No volume transform found from '{source_space}' to '{target_space}' "
                f"(resolution='{resolution}', resource_type='{resource_type}')"
            )

        target_atlas = self.fetchers.fetch_volume_atlas(
            space=target_space,
            resolution=resolution,
            resource_type=resource_type,
        )
        if target_atlas is None:
            raise ValueError(
                f"No volume atlas found for '{target_space}' "
                f"(resolution='{resolution}', resource_type='{resource_type}')"
            )

        return vol_to_vol(
            source=input_file,
            target=target_atlas.fetch(),
            out_fpath=output_file_path,
            interp=interp,
            interp_params=interp_params,
        )

    # Volume-to-surface
    def transform_volume_to_surface(
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
        add_edge: bool = True,
    ) -> Path | None:
        """Project a volume into surface space then resample to target_space.

        Two-stage pipeline:
          1. Project volume → surface (ribbon-constrained mapping).
          2. Surface → surface resampling via :class:`SurfaceTransformOps`.

        Args:
            transformer_type: ``'metric'`` or ``'label'``.
            input_file: NIfTI volume in source_space.
            source_space: Source brain template space name.
            target_space: Target brain template space name.
            hemisphere: ``'left'`` or ``'right'``.
            output_file_path: Path for the final resampled GIFTI output.
            source_density: Source surface density. Highest available used
                when ``None``.
            target_density: Target surface density. Highest available used
                when ``None``.
            area_resource: Surface type used for area correction
                (default ``'midthickness'``).
            add_edge: Whether to register composed transforms as graph edges.

        Returns:
            Path to the resampled output GIFTI, or ``None`` if the surface
            transform could not be resolved.

        Raises:
            FileNotFoundError: If input_file does not exist.
            ValueError: If transformer_type is invalid or required surface
                resources are missing.
        """
        if transformer_type not in ("metric", "label"):
            raise ValueError(
                f"Invalid transformer_type: '{transformer_type}'. "
                "Must be 'metric' or 'label'."
            )

        self.utils.validate_spaces(source_space, target_space)
        validate_volume_file(input_file)

        source_density = source_density or self.utils.find_highest_density(
            space=source_space
        )

        projected = self._project_volume_to_surface(
            transformer_type=transformer_type,
            input_file=input_file,
            source_space=source_space,
            source_density=source_density,
            hemisphere=hemisphere,
            area_resource=area_resource,
        )

        return self.surface_ops.transform(
            transformer_type=transformer_type,
            input_file=projected,
            source_space=source_space,
            target_space=target_space,
            hemisphere=hemisphere,
            output_file_path=output_file_path,
            source_density=source_density,
            target_density=target_density,
            area_resource=area_resource,
            add_edge=add_edge,
        )

    # Private helpers
    def _project_volume_to_surface(
        self,
        transformer_type: Literal["metric", "label"],
        input_file: Path,
        source_space: str,
        source_density: str,
        hemisphere: Literal["left", "right"],
        area_resource: str,
    ) -> Path:
        """Ribbon-constrained projection of input_file onto the surface.

        Fetches the midthickness (or area_resource), white, and pial
        surfaces for the source space, builds ribbon constraints, then
        calls :func:`surface_project`.

        Args:
            transformer_type: ``'metric'`` or ``'label'`` — determines the
                output file extension.
            input_file: NIfTI volume to project.
            source_space: Brain template space of the volume.
            source_density: Surface mesh density to use for projection.
            hemisphere: ``'left'`` or ``'right'``.
            area_resource: Primary surface type for projection mapping.

        Returns:
            Path to the projected GIFTI surface file.

        Raises:
            ValueError: If any required surface atlas is missing.
        """
        source_surface = self._require_surface_atlas(
            space=source_space,
            density=source_density,
            hemisphere=hemisphere,
            resource_type=area_resource,
        ).fetch()

        ribbon = {}
        for surf_type in ("white", "pial"):
            ribbon[surf_type] = self._require_surface_atlas(
                space=source_space,
                density=source_density,
                hemisphere=hemisphere,
                resource_type=surf_type,
            ).fetch()

        ribbon_surfs = workbench.volume_to_surface_mapping_ribbon_constrained(
            inner_surf=ribbon["white"],
            outer_surf=ribbon["pial"],
        )
        ext = "func" if transformer_type == "metric" else "label"
        out_fpath = (
            f"src-{source_space}_"
            f"den-{source_density}_"
            f"hemi-{hemisphere}_"
            f"desc-volume_annot.{ext}.gii"
        )
        return surface_project(
            volume=input_file,
            surface=source_surface,
            ribbon_surfs=ribbon_surfs,
            out_fpath=out_fpath,
        )

    def _require_surface_atlas(
        self,
        space: str,
        density: str,
        hemisphere: Literal["left", "right"],
        resource_type: str,
    ) -> SurfaceAtlas:
        """Fetch a surface atlas or raise a descriptive ``ValueError``.

        Args:
            space: Brain template space name.
            density: Surface mesh density.
            hemisphere: ``'left'`` or ``'right'``.
            resource_type: Surface resource type.

        Returns:
            The matching :class:`~neuromaps_prime.graph.models.SurfaceAtlas`.

        Raises:
            ValueError: If no matching atlas is found.
        """
        atlas = self.fetchers.fetch_surface_atlas(
            space=space,
            density=density,
            hemisphere=hemisphere,
            resource_type=resource_type,
        )
        if atlas is None:
            raise ValueError(
                f"No '{resource_type}' surface atlas found for space '{space}' "
                f"(density='{density}', hemisphere='{hemisphere}')"
            )
        return atlas
