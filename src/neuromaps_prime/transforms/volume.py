"""Functions for volumetric transformations using niwrap."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from niwrap import ants, workbench

from neuromaps_prime.transforms.utils import relocate_output

INTERP_PARAMS: dict[str, Callable[..., Any]] = {
    "linear": ants.ants_apply_transforms_linear,
    "nearestNeighbor": ants.ants_apply_transforms_nearest_neighbor,
    "multiLabel": ants.ants_apply_transforms_multi_label,
    "gaussian": ants.ants_apply_transforms_gaussian,
    "BSpline": ants.ants_apply_transforms_bspline,
    "cosineWindowedSinc": ants.ants_apply_transforms_cosine_windowed_sinc,
    "welchWindowedSinc": ants.ants_apply_transforms_welch_windowed_sinc,
    "hammingWindowedSinc": ants.ants_apply_transforms_hamming_windowed_sinc,
    "lanczosWindowedSinc": ants.ants_apply_transforms_lanczos_windowed_sinc,
}
INTERP_NOPARAMS: dict[str, Callable[..., Any]] = {
    "multiLabel": ants.ants_apply_transforms_multi_labelnoparams
}


def _get_interp_params(
    interp: str, interp_params: dict[str, Any] | None = None
) -> dict:
    """Get the appropriate interpolation parameters object.

    Args:
        interp: Interpolation method name / key.
        interp_params: Optional parameters to pass to the interpolation method.

    Returns:
        Configured interpolation parameters dictionary.
    """
    if not interp_params and interp in INTERP_NOPARAMS:
        return INTERP_NOPARAMS[interp]()
    return INTERP_PARAMS[interp](**(interp_params or {}))


def vol_to_vol(
    source: Path,
    target: Path,
    out_fpath: str,
    interp: str = "linear",
    interp_params: dict[str, Any] | None = None,
) -> Path:
    """Transform a volumetric image from source space to target space.

    Args:
        source: Path to the source NIfTI volume to be transformed.
        target: Path to the target NIfTI volume defining the reference space.
        out_fpath: Full output file path to save transformed file
        interp: Interpolation method to use.
        interp_params: Optional parameters to pass to the interpolation method.

    Returns:
        Path to the transformed NIfTI file written to disk.

    Raises:
        ValueError: unsupported interpolator.
        FileNotFoundError: If the output file is not created.
    """
    if interp not in INTERP_PARAMS:
        raise ValueError(f"Unsupported interpolator '{interp}'.")

    interpolation = _get_interp_params(interp, interp_params)
    final_path = Path(out_fpath)
    xfm = ants.ants_apply_transforms(
        input_image=source,
        reference_image=target,
        output=ants.ants_apply_transforms_warped_output(final_path.name),
        interpolation=interpolation,  # type: ignore[arg-type]
    )
    written = Path(xfm.output.output_image_outfile)

    if not written.exists():
        raise FileNotFoundError(f"Warped volume not computed: {written}")

    if final_path.is_absolute():
        relocate_output(written, final_path)
        if not final_path.exists():
            raise FileNotFoundError(f"Warped volume not found: {final_path}")
        return final_path
    return written


def surface_project(
    volume: Path,
    surface: Path,
    ribbon_surfs: workbench.VolumeToSurfaceMappingRibbonConstrainedParamsDict,  # type: ignore[valid-type]
    out_fpath: str,
) -> Path:
    """Project a volumetric image to a surface from source space to target space.

    Args:
        volume: Path to the source NIfTI annotation to be projected.
        surface: Path to the target surface to project to.
        ribbon_surfs: Ribbon surfaces to constrain projections to.
        out_fpath: Full output file path to projected annotation.

    Returns:
        Path to the projected surface annotation file written to disk.

    Raises:
        FileNotFoundError: If the output file is not created.
    """
    final_path = Path(out_fpath)

    projected_vol = workbench.volume_to_surface_mapping(
        volume=volume,
        surface=surface,
        ribbon_constrained=ribbon_surfs,
        metric_out=final_path.name,
    )
    written = Path(projected_vol.metric_out)

    if not written.exists():
        raise FileNotFoundError(f"Projected volume not computed: {written}")

    if final_path.is_absolute():
        relocate_output(written, final_path)
        if not final_path.exists():
            raise FileNotFoundError(f"Projected volume out not found: {final_path}")
        return final_path
    return written
