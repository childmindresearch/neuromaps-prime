"""Functions for volumetric transformations using niwrap."""

from pathlib import Path
from typing import Callable

from niwrap import ants

INTERP_PARAMS: dict[str, Callable[[], dict]] = {
    "linear": ants.ants_apply_transforms_linear_params,
    "nearestNeighbor": ants.ants_apply_transforms_nearest_neighbor_params,
    "multiLabel": ants.ants_apply_transforms_multi_label_params,
    "gaussian": ants.ants_apply_transforms_gaussian_params,
    "BSpline": ants.ants_apply_transforms_bspline_params,
    "cosineWindowedSinc": ants.ants_apply_transforms_cosine_windowed_sinc_params,
    "welchWindowedSinc": ants.ants_apply_transforms_welch_windowed_sinc_params,
    "hammingWindowedSinc": ants.ants_apply_transforms_hamming_windowed_sinc_params,
    "lanczosWindowedSinc": ants.ants_apply_transforms_lanczos_windowed_sinc_params,
}

_NOT_IMPLEMENTED = {"BSpline", "gaussian", "multiLabel"}


def _vol_to_vol(
    source: Path,
    target: Path,
    out_fpath: str,
    interp: str = "linear",
    label: Path | None = None,
) -> Path:
    """Transform a volumetric image from source space to target space.

    Args:
        source: Path to the source NIfTI volume to be transformed.
        target: Path to the target NIfTI volume defining the reference space.
        out_fpath: Full output file path to save transformed file
        interp: Interpolation method to use.
        label: Optional path to a label image (optional for multiLabel).

    Returns:
        Path to the transformed NIfTI file written to disk.

    Raises:
        ValueError: unsupported interpolator.
        NotImplementedError: not yet implemented interpolator.
    """
    if interp not in INTERP_PARAMS:
        raise ValueError(f"Unsupported interpolator '{interp}'.")
    if interp in _NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"The '{interp}' interpolation method is not yet implemented."
        )

    ants.ants_apply_transforms(
        input_image=source,
        reference_image=target,
        output=ants.ants_apply_transforms_warped_output_params(out_fpath),
        interpolation=INTERP_PARAMS[interp](),
    )
    return Path(out_fpath)
