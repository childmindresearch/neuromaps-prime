"""Tests for surface transformations using Neuromaps NHP."""

from pathlib import Path

import pytest

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.transforms.surface import (
    _surface_to_surface,
    surface_sphere_project_unproject,
    surface_to_surface_transformer,
)
from neuromaps_prime.transforms.utils import (
    estimate_surface_density,
    find_highest_density,
    get_vertex_count,
)

DATA_DIR = Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share")


@pytest.mark.usefixtures("require_workbench")
@pytest.mark.usefixtures("require_data")
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

    vertices_sphere_in = get_vertex_count(sphere_in)
    vertices_sphere_out = get_vertex_count(result.sphere_out)
    assert vertices_sphere_in == vertices_sphere_out


def test_surface_to_surface(tmp_path: Path) -> None:
    """Test _surface_to_surface function."""
    graph = NeuromapsGraph()

    source_space = "S1200"
    target_space = "D99"
    density = "32k"
    hemisphere = "left"

    transform = _surface_to_surface(
        graph=graph,
        source=source_space,
        target=target_space,
        density=density,
        hemisphere=hemisphere,
        output_file_path=str(tmp_path / "output.surf.gii"),
    )

    assert transform is not None
    assert estimate_surface_density(transform.fetch()) == density


@pytest.mark.parametrize(
    "transformer_type,input_file",
    [
        (
            "label",
            DATA_DIR / "Inputs/CIVETNMT/"
            "src-CIVETNMT_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii",
        ),
        (
            "metric",
            DATA_DIR / "Inputs/CIVETNMT/"
            "src-CIVETNMT_den-41k_hemi-R_desc-vaavg_midthickness.shape.gii",
        ),
    ],
)
def test_surface_to_surface_transformer(
    tmp_path: Path, transformer_type: str, input_file: Path
) -> None:
    """Test surface_to_surface_transformer function."""
    graph = NeuromapsGraph()

    source_space = "CIVETNMT"
    target_space = "S1200"
    hemisphere = "right"
    output_file_path = str(
        tmp_path / f"space-{target_space}_output_{transformer_type}.func.gii"
    )

    output = surface_to_surface_transformer(
        transformer_type=transformer_type,
        graph=graph,
        input_file=input_file,
        source_space=source_space,
        target_space=target_space,
        hemisphere=hemisphere,
        output_file_path=output_file_path,
    )

    assert output is not None
    target_density = find_highest_density(graph=graph, space=target_space)
    if transformer_type == "metric":
        assert estimate_surface_density(output.metric_out) == target_density
    elif transformer_type == "label":
        assert estimate_surface_density(output.label_out) == target_density
