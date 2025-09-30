from pathlib import Path
import subprocess
from niwrap import ants

'''Extract voxel spacing from a NIfTI file using wb_command.'''
def _extract_res(nii_file: Path):
    
    """Extract voxel spacing from a NIfTI file using wb_command."""
    cmd = ["wb_command", "-file-information", str(nii_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"wb_command failed: {result.stderr}")

    for line in result.stdout.splitlines():
        if "spacing:" in line.lower():
            spacing_values = line.split(":", 1)[1].replace("mm", "").strip()
            return tuple(float(x.strip()) for x in spacing_values.split(","))
    raise ValueError(f"cannot determine resolution {nii_file}")

'''Transform a volumetric image from source space to target space.'''
def _vol_to_vol(source: Path, target: Path) -> Path:

    source = Path(source)
    target = Path(target)
    out_file = target.parent / f"{source.stem}_to_{target.stem}.nii"
    interp = ants.ants_apply_transforms_linear_params()
    output = ants.ants_apply_transforms_warped_output_params(str(out_file))

    ants.ants_apply_transforms(
        input_image=str(source),
        reference_image=str(target),
        output=output,
        interpolation=interp,
        dimensionality=3
    )

    return out_file
