
import os
from niwrap import workbench as wb


def surface_sphere_project_unproject(sphere_in, sphere_project_to, sphere_unproject_from, sphere_out):
    """Project and unproject a surface from one sphere to another.

    Parameters
    ----------
    sphere_in : str
        Path to input spherical surface.
    sphere_project_to : str
        Path to spherical surface to project to.
    sphere_unproject_from : str
        Path to spherical surface to unproject from.
    sphere_out : str
        Path to output spherical surface.

    Returns
    -------
    out_sphere : str
        Path to output spherical surface.
    
    
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
    wb.surface_sphere_project_unproject(sphere_in, sphere_project_to, sphere_unproject_from, sphere_out)
    return sphere_out
    