"""Example script demonstrating surface transformations.

Covers:
- Surface-to-surface resampling (label and metric)
- Surface-to-volume projection

Usage (from repository root):

    uv run examples/example_surface_transform.py
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

SOURCE_DENSITY = "32k"
SOURCE_SPACE = "Yerkes19"
TARGET_SPACE = "CIVETNMT"
LABEL = "PC_Yeo7Networks"
HEMISPHERE = "left"

# Load graph
graph = NeuromapsGraph()
print("Graph summary:")
print(graph)
print(graph.utils.get_graph_info())

# ------------------------------------------------------------------ #
# Surface-to-Surface Label Resampling                                      #
# ------------------------------------------------------------------ #

# Label resampling (explicity source density; automatic target density)
# Use annotation shipped with package (can replace with file path)
label_input = graph.fetch_surface_annotation(
    space=SOURCE_SPACE, label=LABEL, density=SOURCE_DENSITY, hemisphere=HEMISPHERE,
).fetch()

label_result = graph.surface_to_surface_transformer(
    transformer_type="label",
    input_file=label_input,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    source_density=SOURCE_DENSITY,
    hemisphere=HEMISPHERE,
    output_file_path=f"src-{TARGET_SPACE}_hemi-L_atlas-{LABEL}_desc-PC_annot.label.gii",
)
if label_result.path is None:
    print("Label surface-to-surface transformation failed.")
print(f"Transformed label saved at: {label_result.path}")


# Metric resampling (both source and target densities explicit)
# Use annotation shipped with package (can replace with file path)
metric_input = graph.fetch_surface_annotation(
    space=SOURCE_SPACE, label="CT", density=SOURCE_DENSITY, hemisphere=HEMISPHERE
).fetch()

metric_result = graph.surface_to_surface_transformer(
    transformer_type="metric",
    input_file=metric_input,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    source_density=SOURCE_DENSITY,
    target_density="41k",
    hemisphere=HEMISPHERE,
    output_file_path=f"src-{TARGET_SPACE}_hemi-L_desc-CT_annot.func.gii",
)

if metric_result.path is None:
    print("Metric surface-to-surface transformation failed.")
print(f"Transformed metric saved at: {metric_result.path}")


# ------------------------------------------------------------------ #
# Surface-to-Volume Projection                                       #
# ------------------------------------------------------------------ #
# Two-stage pipeline:
#   1. Resamples the metric between two surface spaces (same as above)
#   2. Projects the resampled surface into volume space using ribbon-constrained
#   mapping with white/pial surfaces.

vol_tgt = "MEBRAINS"
vol_ref = graph.fetch_volume_atlas(
    space=vol_tgt, resolution="400um", resource_type="T1w"
).fetch()

vol_result = graph.surface_to_volume_transformer(
    transformer_type="metric",
    input_file=metric_input,
    ref_volume=vol_ref,
    source_space=SOURCE_SPACE,
    target_space=vol_tgt,
    hemisphere=HEMISPHERE,
    source_density=SOURCE_DENSITY,
    target_density="101k", # MEBRAINS target density
    output_file_path=f"src-{vol_tgt}_desc-CT_T1w.nii.gz",
)

if vol_result.path is None:
    print("Surface-to-volume transformation failed.")
print(f"Surface-to-volume output: {vol_result}")



# After running transforms, new edges are added to the graph automatically (if
# necessary). You can inspect what's now available between source and target spaces.
available_transforms = graph.search_surface_transforms(
    source_space=SOURCE_SPACE, target_space=TARGET_SPACE, hemisphere=HEMISPHERE,
)
print(f"Available transforms after run: {available_transforms}")
