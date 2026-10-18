"""Example script demonstrating volume transformations.

Covers:
- Volume-to-volume warping
- Volume-to-surface projection

Usage (from repository root):

    uv run examples/example_volume_transform.py
"""

# ///
# requires-python = ">=3.11"
# dependencies = [
#     # Uses this repo's checkout (editable) via ${PWD}; swap in a pinned
#     # release (e.g. "neuromaps-prime==0.2.0") if you prefer.
#     "neuromaps-prime @ file:///${PWD}",
# ]
# ///

from pathlib import Path

from neuromaps_prime.graph import NeuromapsGraph


SOURCE_SPACE = "D99"
SOURCE_RES = "250um"
TARGET_SPACE = "MEBRAINS"
TARGET_RES = "400um"
HEMISPHERE = "right"

# Load graph
graph = NeuromapsGraph(runner="podman")
print("Graph summary:")
print(graph)
print(graph.utils.get_graph_info())

# ------------------------------------------------------------------ #
# Volume-to-Volume Warping                                           #
# ------------------------------------------------------------------ #

# Warps a NIfTI volume from MEBRAINS -> Yerkes19 volume space.
# Resolution and resource type are specified explicitly.
vol_input = graph.fetch_volume_atlas(
    space=SOURCE_SPACE, resolution=SOURCE_RES, resource_type="T1w"
).fetch()

vol_result = graph.volume_to_volume_transformer(
    input_file=vol_input,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    resolution=TARGET_RES,
    resource_type="composite",
    output_file_path=f"src-{TARGET_SPACE}_res-{TARGET_RES}_T1w.nii.gz",
)

if vol_result.path is None:
    print("Volume-to-volume warp failed.")
print(f"Warped volume saved at: {vol_result}")



# ------------------------------------------------------------------ #
# Volume-to-Surface Projection                                       #
# ------------------------------------------------------------------ #
# Two-stage pipeline:
#   1. Projects the volume into MEBRAINS surface space using ribbon-constrained mapping with white/pial surfaces.
#   2. Resamples the surface from MEBRAINS -> Yerkes19 surface space.

metric_input = graph.fetch_volume_annotation(
    space=SOURCE_SPACE, label="MTR", resolution=SOURCE_RES
).fetch()

surf_result = graph.volume_to_surface_transformer(
    transformer_type="metric",
    input_file=metric_input,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    hemisphere=HEMISPHERE,
    source_density="41k",
    output_file_path=f"src-{TARGET_SPACE}_den-41k_desc-MTR_annot.func.gii",
)

if surf_result.path is None:
    print("Volume-to-surface transformation failed.")
print(f"Volume-to-surface output: {surf_result}")


# After running transforms, new edges are added to the graph automatically.
# You can inspect what's now available between source and target spaces.
available_transforms = graph.search_volume_transforms(
    source_space=SOURCE_SPACE, target_space=TARGET_SPACE
)
print(f"Available transforms after run: {available_transforms}")
