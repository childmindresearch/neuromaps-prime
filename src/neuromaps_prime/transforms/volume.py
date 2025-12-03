"""Functions for volumetric transformations using niwrap."""

from pathlib import Path
from typing import Any, Callable

from niwrap import ants

INTERP_PARAMS: dict[str, Callable[..., dict]] = {
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
INTERP_NOPARAMS: dict[str, Callable[..., dict]] = {
    "multiLabel": ants.ants_apply_transforms_multi_labelnoparams,
}

_NOT_IMPLEMENTED = frozenset({"BSpline"})


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
        NotImplementedError: not yet implemented interpolator.
    """
    if interp in _NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"The '{interp}' interpolation method is not yet implemented"
        )
    if interp not in INTERP_PARAMS:
        raise ValueError(f"Unsupported interpolator '{interp}'.")

    interpolation = _get_interp_params(interp, interp_params)
    xfm = ants.ants_apply_transforms(
        input_image=source,
        reference_image=target,
        output=ants.ants_apply_transforms_warped_output(out_fpath),
        interpolation=interpolation,
    )
    return Path(xfm.output.output_image_outfile)
