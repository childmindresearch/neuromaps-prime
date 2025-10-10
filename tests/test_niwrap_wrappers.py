"""Tests for niwrap wrapper functions."""

from pathlib import Path

import pytest

from neuromaps_nhp import config


@pytest.mark.parametrize(
    "sphere_in,sphere_project_to,sphere_unproject_from,sphere_out",
    [
        (
            Path(
                "/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Outputs/Yerkes19-S1200/src-S1200_to-Yerkes19_den-32k_hemi-L_sphere.surf.gii"
            ),
            Path(
                "/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Inputs/Yerkes19/src-Yerkes19_den-32k_hemi-L_sphere.surf.gii"
            ),
            Path(
                "/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Outputs/D99-Yerkes19/src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii"
            ),
            config.data_dir / "out_sphere.surf.gii",
        ),
        # Add more tuples here for additional test cases
    ],
)
def test_surface_sphere_project_unproject(
sphere_in: Path,
sphere_project_to: Path,
sphere_unproject_from; Path,
sphere_out_name: str,
tmp_path: Path
):
    sphere_out = tmp_path / sphere_out_name
) -> None:
    """Test surface_sphere_project_unproject wrapper function.

    == Example ==
    ------------------------
    S1200 to Yerkes19 to D99
    ------------------------
    sphere_in               = S1200_aligned_to_Yerkes19 (Input)
    project_to_sphere       = Yerkes19 (Intermediate)
    unproject_from_sphere   = Yerkes19_to_D99 (Target)
    out_sphere              = Path(f"{data_dir}/out_sphere.surf.gii").resolve()
    ------------------------
    """
    from neuromaps_nhp.utils.gifti_utils import get_num_vertices
    from neuromaps_nhp.utils import gifti_utils, niwrap_wrappers

    result = surface_sphere_project_unproject(
        sphere_in, sphere_project_to, sphere_unproject_from, sphere_out
    )

    assert get_num_vertices(sphere_in) == get_num_vertices(result)
