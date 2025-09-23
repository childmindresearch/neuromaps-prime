import pytest
import os
from pathlib import Path

from niwrap import workbench as wb
import nibabel as nib

from neuromaps_nhp.utils.surface_sphere_project_unproject import surface_sphere_project_unproject
from neuromaps_nhp.config import Config

@pytest.mark.parametrize(
    "sphere_in,sphere_project_to,sphere_unproject_from,sphere_out",
    [
        (
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Outputs/Yerkes19-S1200/src-S1200_to-Yerkes19_den-32k_hemi-L_sphere.surf.gii")),
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Inputs/Yerkes19/src-Yerkes19_den-32k_hemi-L_sphere.surf.gii")),
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Outputs/D99-Yerkes19/src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii")),
            str(Config.data_dir / "out_sphere.surf.gii"),
        ),
        # Add more tuples here for additional test cases
    ]
)
def test_surface_sphere_project_unproject(sphere_in, sphere_project_to, sphere_unproject_from, sphere_out):
    """
    Test surface_sphere_project_unproject wrapper function.

    == Example ==
    ------------------------
    S1200 to Yerkes19 to D99
    ------------------------
    sphere_in               = S1200_aligned_t-_Yerkes19 (Input)
    project_to_sphere       = Yerkes19 (Intermediate)
    unproject_from_sphere   = Yerkes19_to_D99 (Target)
    out_sphere              = str(Path(f"{data_dir}/out_sphere.surf.gii").resolve())
    ------------------------
    """

    result = surface_sphere_project_unproject(
        sphere_in, sphere_project_to, sphere_unproject_from, sphere_out
    )
    assert os.path.exists(result)
    
    # check if the output file has the same number of vertices as the input file
    assert nib.load(result).darrays[0].data.shape == nib.load(sphere_in).darrays[0].data.shape
