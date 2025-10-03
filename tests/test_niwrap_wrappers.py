import pytest
import os
from pathlib import Path
import nibabel as nib

from neuromaps_nhp.config import config

@pytest.mark.parametrize(
    "sphere_in,sphere_project_to,sphere_unproject_from,sphere_out",
    [
        (
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Outputs/Yerkes19-S1200/src-S1200_to-Yerkes19_den-32k_hemi-L_sphere.surf.gii")),
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Inputs/Yerkes19/src-Yerkes19_den-32k_hemi-L_sphere.surf.gii")),
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Outputs/D99-Yerkes19/src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii")),
            str(config.data_dir / "out_sphere.surf.gii"),
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
    sphere_in               = S1200_aligned_to_Yerkes19 (Input)
    project_to_sphere       = Yerkes19 (Intermediate)
    unproject_from_sphere   = Yerkes19_to_D99 (Target)
    out_sphere              = str(Path(f"{data_dir}/out_sphere.surf.gii").resolve())
    ------------------------
    """
    from neuromaps_nhp.utils.niwrap_wrappers import surface_sphere_project_unproject

    result = surface_sphere_project_unproject(
        sphere_in, sphere_project_to, sphere_unproject_from, sphere_out
    )
    assert os.path.exists(result)
    
    # check if the output file has the same number of vertices as the input file
    assert nib.load(result).darrays[0].data.shape == nib.load(sphere_in).darrays[0].data.shape


@pytest.mark.parametrize(
    "surface_in,current_sphere,new_sphere,method,surface_out",
    [
        (
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Inputs/Yerkes19/src-Yerkes19_den-10k_hemi-L_midthickness.surf.gii")),
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Inputs/Yerkes19/src-Yerkes19_den-10k_hemi-L_sphere.surf.gii")),
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Outputs/D99-Yerkes19/src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii")),
            'BARYCENTRIC',
            str(config.data_dir / "test_surface_resampled.surf.gii"),
        )
    ]
)
def test_surface_resample(surface_in, current_sphere, new_sphere, method, surface_out):
    """Test surface_resample wrapper function with BARYCENTRIC method."""
    # Skip test if input files don't exist
    if not os.path.exists(current_sphere) or not os.path.exists(new_sphere):
        pytest.skip("Required sphere files not found")
    
    if not os.path.exists(surface_in):
        pytest.skip("Test surface file not found")
    
    from neuromaps_nhp.utils.niwrap_wrappers import surface_resample
    from neuromaps_nhp.utils.gifti_utils import get_num_vertices, get_density
    
    result = surface_resample(
        surface_in=surface_in,
        current_sphere=current_sphere,
        new_sphere=new_sphere,
        method=method,
        surface_out=surface_out
    )
    
    assert get_num_vertices(result) == get_num_vertices(new_sphere)

