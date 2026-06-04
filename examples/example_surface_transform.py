"""Example script demonstrating surface transformations.

Covers:
- Surface-to-surface resampling
- Surface-to-volume projection
- Inspecting newly composed transforms in the graph

Usage:

Set DATA_DIR to the root of your neuromaps data directory, then run:

    python examples/example_surface_transform.py
"""

from pathlib import Path

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.plotting import plot_graph


# Configuration (EDIT this path before running)
DATA_DIR = Path("/Users/janhavi.pillai/Desktop/projects/neuromaps-nhp-prep")

SOURCE_SPACE = "D99"
TARGET_SPACE = "MEBRAINS"
HEMISPHERE = "right"

graph = NeuromapsGraph(data_dir=DATA_DIR)
print("Graph summary:")
print(graph)
print(graph.utils.get_graph_info())

# Label resample (densities inferred automatically)
label_input = DATA_DIR / Path(
    "share/Inputs/D99/src-D99_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii"
)
label_output = str(
    Path(__file__).parent / f"space-{TARGET_SPACE}_output_label.label.gii"
)

label_result = graph.surface_to_surface_transformer(
    transformer_type="label",
    input_file=label_input,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    hemisphere=HEMISPHERE,
    output_file_path=str(label_output),
)

if label_result is not None:
    print(f"Transformed label saved at: {label_result}")
else:
    print("Label surface-to-surface transformation failed.")

# Metric resample (explicit source and target densities)
metric_input = DATA_DIR / Path(
    "share/Inputs/D99/src-D99_den-41k_hemi-R_desc-vaavg_midthickness.shape.gii"
)
metric_output = str(
    Path(__file__).parent / f"space-{TARGET_SPACE}_output_metric.shape.gii"
)
metric_result = graph.surface_to_surface_transformer(
    transformer_type="metric",
    input_file=metric_input,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    source_density="41k",
    target_density="10k",
    hemisphere=HEMISPHERE,
    output_file_path=str(metric_output),
)

if metric_result is not None:
    print(f"Transformed metric saved at: {metric_result}")
else:
    print("Metric surface-to-surface transformation failed.")

# Surface-to-volume projection
vol_ref = DATA_DIR / "share/Inputs/D99/src-D99_res-0p25mm_T1w.nii"
surf_to_vol_output = (
    Path(__file__).parent / f"space-{TARGET_SPACE}_output_metric.nii.gz"
)

vol_result = graph.surface_to_volume_transformer(
    transformer_type="metric",
    input_file=metric_input,
    ref_volume=vol_ref,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    hemisphere=HEMISPHERE,
    output_file_path=str(surf_to_vol_output),
    source_density="41k",
)

if vol_result is not None:
    print(f"Surface-to-volume output: {vol_result}")
else:
    print("Surface-to-volume transform failed.")


# Inspect the composed transforms added to the graph
available_transforms = graph.search_surface_transforms(
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    hemisphere=HEMISPHERE,
)
print(
    f"Available surface-to-surface transforms after addition: "
    f"{available_transforms}"
)
transform = graph.fetch_surface_to_surface_transform(
    source=SOURCE_SPACE,
    target=TARGET_SPACE,
    density="41k",
    hemisphere=HEMISPHERE,
    resource_type="sphere",
)

# Plot updated surface subgraph
surface_subgraph = graph.utils.get_subgraph("surface_to_surface")
plot_graph(
    surface_subgraph,
    graph_type="surface",
    layout="kamada_kawai",
    save_path=Path("examples/updated_neuromaps_surface.png")
)
