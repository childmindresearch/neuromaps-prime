"""Example script demonstrating volume transformations.

Covers:
- Volume-to-volume warping
- Volume-to-surface projection

Usage:

Set DATA_DIR to the root of your neuromaps data directory, then run:

    python examples/example_volume_transform.py
"""

from pathlib import Path

from neuromaps_prime.graph import NeuromapsGraph

# Configuration (EDIT this path before running)
DATA_DIR = Path("/path/to/data")

SOURCE_SPACE = "Yerkes19"
TARGET_SPACE = "MEBRAINS"
HEMISPHERE = "right"

# Load graph
graph = NeuromapsGraph()
print("Graph summary:")
print(graph)
print(graph.utils.get_graph_info())

# ------------------------------------------------------------------ #
# Volume-to-Volume Warping                                           #
# ------------------------------------------------------------------ #

# Warps a NIfTI volume from MEBRAINS -> Yerkes19 volume space.
# Resolution and resource type are specified explicitly.
vol_input = DATA_DIR / "share/Inputs/Yerkes19/src-Yerkes19_res-0p50mm_T1w.nii"

vol_result = graph.volume_to_volume_transformer(
    input_file=vol_input,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    resolution="400um",
    resource_type="composite",
    output_file_path=f"space-{TARGET_SPACE}_output_T1w.nii.gz",
)

if vol_result is not None:
    print(f"Warped volume saved at: {vol_result}")
else:
    print("Volume-to-volume warp failed.")


# ------------------------------------------------------------------ #
# Volume-to-Surface Projection                                       #
# ------------------------------------------------------------------ #
# Two-stage pipeline:
#   1. Projects the volume into MEBRAINS surface space using ribbon-constrained mapping with white/pial surfaces.
#   2. Resamples the surface from MEBRAINS -> Yerkes19 surface space.

surf_result = graph.volume_to_surface_transformer(
    transformer_type="metric",
    input_file=vol_input,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    hemisphere=HEMISPHERE,
    source_density="32k",
    output_file_path=f"space-{TARGET_SPACE}_output_metric.shape.gii",
)

if surf_result is not None:
    print(f"Volume-to-surface output: {surf_result}")
else:
    print("Volume-to-surface transformation failed.")


# After running transforms, new edges are added to the graph automatically.
# You can inspect what's now available between source and target spaces.
available_transforms = graph.search_volume_transforms(
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
)
print(f"Available transforms after run: {available_transforms}")