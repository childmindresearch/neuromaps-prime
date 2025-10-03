import pytest
import os
from pathlib import Path

from neuromaps_nhp.config import config

@pytest.mark.parametrize(
    "gifti_file,expected_density",
    [
        (
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Inputs/Yerkes19/src-Yerkes19_den-32k_hemi-L_sphere.surf.gii")),
            "32k"
        ),
        (
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Inputs/Yerkes19/src-Yerkes19_den-10k_hemi-L_sphere.surf.gii")),
            "10k"
        ),
        (
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Outputs/D99-Yerkes19/src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii")),
            "32k"
        ),
    ]
)
def test_get_density(gifti_file, expected_density):
    """Test get_density function with various mesh densities."""
    if not os.path.exists(gifti_file):
        pytest.skip(f"Test file not found: {gifti_file}")
    
    from neuromaps_nhp.utils.gifti_utils import get_density
    
    result = get_density(Path(gifti_file))
    assert result == expected_density
    assert result.endswith("k")


@pytest.mark.parametrize(
    "gifti_file,expected_range",
    [
        (
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Inputs/Yerkes19/src-Yerkes19_den-32k_hemi-L_sphere.surf.gii")),
            (30000, 35000) 
        ),
        (
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Inputs/Yerkes19/src-Yerkes19_den-10k_hemi-L_sphere.surf.gii")),
            (9000, 11000) 
        ),
        (
            str(Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share/Outputs/D99-Yerkes19/src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii")),
            (30000, 35000)
        ),
    ]
)
def test_get_num_vertices(gifti_file, expected_range):
    """Test get_num_vertices function returns correct vertex count."""
    if not os.path.exists(gifti_file):
        pytest.skip(f"Test file not found: {gifti_file}")
    
    from neuromaps_nhp.utils.gifti_utils import get_num_vertices
    
    result = get_num_vertices(Path(gifti_file))
    assert isinstance(result, int)
    assert expected_range[0] <= result <= expected_range[1]