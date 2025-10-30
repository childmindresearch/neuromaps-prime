"""Example script demonstrating how to use surface-to-surface transforms."""

from pathlib import Path

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.transforms.surface import _surface_to_surface

if __name__ == "__main__":
    # Load the Neuromaps graph
    graph = NeuromapsGraph()

    # Define source and target spaces
    source_space = "S1200"
    target_space = "D99"
    density = "32k"
    hemisphere = "left"

    # Fetch the surface-to-surface transform
    transform = _surface_to_surface(
        graph=graph,
        source=source_space,
        target=target_space,
        density=density,
        hemisphere=hemisphere,
        output_file_path=str(Path(__file__).parent / "output.surf.gii"),
    )

    if transform is not None:
        print(f"Transform found: {transform.fetch()}")
    else:
        print("No transform found for the specified parameters.")
