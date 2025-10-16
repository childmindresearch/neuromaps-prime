from pathlib import Path
from typing import Iterable
import nibabel as nib

def get_density(input_gifti: Path) -> str:
    """Get density of a gifti surface file based on number of vertices."""
    surface = nib.load(str(input_gifti))
    n_vertices = surface.darrays[0].data.shape[0]
    density = str(round(n_vertices / 1000)) + "k"
    return density


def get_num_vertices(input_gifti: Path) -> int:
    """Get number of vertices in a gifti surface file."""
    surface = nib.load(str(input_gifti))
    return surface.darrays[0].data.shape[0]
