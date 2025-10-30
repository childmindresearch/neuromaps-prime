"""Tests for surface transformations using Neuromaps NHP."""

from pathlib import Path

import pytest

from neuromaps_prime.transforms.surface import surface_sphere_project_unproject
from neuromaps_prime.transforms.utils import get_vertex_count


@pytest.mark.usefixtures("require_workbench")
def test_surface_sphere_project_unproject(data_dir: Path, tmp_path: Path) -> None:
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
    data_dir = data_dir / "share"
    sphere_in = (
        data_dir
        / "Outputs"
        / "Yerkes19-S1200"
        / "src-S1200_to-Yerkes19_den-32k_hemi-L_sphere.surf.gii"
    )
    sphere_project_to = (
        data_dir / "Inputs" / "Yerkes19" / "src-Yerkes19_den-32k_hemi-L_sphere.surf.gii"
    )
    sphere_unproject_from = (
        data_dir
        / "Outputs"
        / "D99-Yerkes19"
        / "src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii"
    )
    sphere_out = tmp_path / "out_sphere.surf.gii"

    result = surface_sphere_project_unproject(
        sphere_in=sphere_in,
        sphere_project_to=sphere_project_to,
        sphere_unproject_from=sphere_unproject_from,
        sphere_out=str(sphere_out),
    )
    assert get_vertex_count(sphere_in) == get_vertex_count(result.sphere_out)
