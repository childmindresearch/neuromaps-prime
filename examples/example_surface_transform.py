"""Example script demonstrating surface transformations.

Covers:
- Surface-to-surface resampling (label and metric)
- Surface-to-volume projection

Usage:

Set DATA_DIR to the root of your neuromaps data directory, then run:

    python examples/example_surface_transform.py
"""

from pathlib import Path

from neuromaps_prime.graph import NeuromapsGraph

# Configuration (EDIT this path before running)
DATA_DIR = Path("/path/to/data")

SOURCE_SPACE = "CIVETNMT"
TARGET_SPACE = "Yerkes19"
HEMISPHERE = "right"

# Load graph
graph = NeuromapsGraph()
print("Graph summary:")
print(graph)
print(graph.utils.get_graph_info())

# ------------------------------------------------------------------ #
# Surface-to-Surface Resampling                                      #
# ------------------------------------------------------------------ #

# Label Resampling: resamples a parcellation label file from CIVETNMT -> Yerkes19 surface space.
# Source density is specified explicitly; target density is auto-selected.
label_input = DATA_DIR / "share/Inputs/CIVETNMT/src-CIVETNMT_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii"

label_result = graph.surface_to_surface_transformer(
    transformer_type="label",
    input_file=label_input,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    hemisphere=HEMISPHERE,
    output_file_path=f"space-{TARGET_SPACE}_output_label.label.gii",
)

if label_result is not None:
    print(f"Transformed label saved at: {label_result}")
else:
    print("Label surface-to-surface transformation failed.")


# Metric Resampling: resamples a metric (shape) file from CIVETNMT -> Yerkes19 surface space.
# Both source and target densities are specified explicitly.
metric_input = DATA_DIR / "share/Inputs/CIVETNMT/src-CIVETNMT_den-41k_hemi-R_desc-vaavg_midthickness.shape.gii"

metric_result = graph.surface_to_surface_transformer(
    transformer_type="metric",
    input_file=metric_input,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    source_density="41k",
    target_density="10k",
    hemisphere=HEMISPHERE,
    output_file_path=f"space-{TARGET_SPACE}_output_metric.shape.gii",
)

if metric_result is not None:
    print(f"Transformed metric saved at: {metric_result}")
else:
    print("Metric surface-to-surface transformation failed.")


# ------------------------------------------------------------------ #
# Surface-to-Volume Projection                                       #
# ------------------------------------------------------------------ #
# Two-stage pipeline:
#   1. Resamples the metric from CIVETNMT -> Yerkes19 surface space (same as above)
#   2. Projects the resampled surface into Yerkes19 volume space using ribbon-constrained mapping with white/pial surfaces.

vol_ref = DATA_DIR / "share/Inputs/Yerkes19/src-Yerkes19_res-0p50mm_T1w.nii"

vol_result = graph.surface_to_volume_transformer(
    transformer_type="metric",
    input_file=metric_input,
    ref_volume=vol_ref,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    hemisphere=HEMISPHERE,
    source_density="41k",
    target_density="32k",
    output_file_path=f"space-{TARGET_SPACE}_output_metric.nii.gz",
)

if vol_result is not None:
    print(f"Surface-to-volume output: {vol_result}")
else:
    print("Surface-to-volume transformation failed.")


# After running transforms, new edges are added to the graph automatically.
# You can inspect what's now available between source and target spaces.
available_transforms = graph.search_surface_transforms(
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    hemisphere=HEMISPHERE,
)
print(f"Available transforms after run: {available_transforms}")