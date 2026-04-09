"""Utility functions."""

from pathlib import Path
import nibabel as nib
from niwrap import Runner, get_global_runner, set_global_runner, use_docker, use_local, use_singularity


def set_runner(
    runner: Runner | str,
    image_overrides: dict[str, str] | None = None,
    **kwargs,
) -> Runner:
    """Set StyxRunner to use for NiWrap.

    Args:
        runner: Styx runner type to use (one of 'local', 'docker', 'singularity').
        image_overrides: Optional dictionary of container tag overrides.
    """
    if image_overrides is not None and not isinstance(image_overrides, dict):
        raise TypeError(
            f"Expected image_overrides dictionary, got {type(image_overrides)}"
        )

    if isinstance(runner, str):
        match runner_exec := runner.lower():
            case "local":
                use_local(**kwargs)
            case "docker" | "singularity":
                runner_fn = use_docker if runner_exec == "docker" else use_singularity
                runner_fn(
                    **{f"{runner_exec}_executable": runner_exec},
                    image_overrides=image_overrides,
                    **kwargs,
                )
            case _:
                raise NotImplementedError(f"'{runner_exec}' runner not implemented.")
    else:
        runner.image_overrides = image_overrides
        set_global_runner(runner)

    return get_global_runner()


def extract_vertex_only(in_file: Path, out_file: Path) -> Path:
    """Extract only the vertex (pointset) arrays from a GIFTI surface."""
    gii = nib.load(str(in_file))
    vertex_arrays = [
        arr for arr in gii.darrays
        if arr.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
    ]
    if not vertex_arrays:
        raise ValueError(f"No vertex arrays found in {in_file}")
    new_gii = nib.gifti.GiftiImage(darrays=vertex_arrays)
    nib.save(new_gii, str(out_file))
    return out_file


def merge_vertices_with_faces(vertex_file: Path, template_file: Path, out_file: Path) -> Path:
    """Merge a vertex-only GIFTI surface with triangle arrays from a template surface."""
    vertex_gii = nib.load(str(vertex_file))
    template_gii = nib.load(str(template_file))
    triangles = [
        arr for arr in template_gii.darrays
        if arr.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'] # this is wrong
    ]
    if not triangles:
        raise ValueError(f"No triangle arrays found in {template_file}")
    new_gii = nib.gifti.GiftiImage(darrays=vertex_gii.darrays + triangles)
    nib.save(new_gii, str(out_file))
    return out_file


def log_gii_shapes(path: Path) -> list[int]:
    """Log the number of vertices in each pointset array of a GIFTI surface file."""
    gii = nib.load(path)
    vertex_arrays = [
        arr for arr in gii.darrays
        if arr.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
    ]
    shapes = [arr.data.shape[0] for arr in vertex_arrays]
    print(f"{path.name} has {shapes} vertices")
    return shapes