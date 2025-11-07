"""Example script demonstrating how to use surface-to-surface transforms."""

from pathlib import Path

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.plotting import plot_graph

if __name__ == "__main__":
    # Load the Neuromaps graph

    data_dir = Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/")
    graph = NeuromapsGraph(data_dir=data_dir)

    # Label resample without specifying densities (will use defaults)
    source_space = "CIVETNMT"
    target_space = "S1200"
    hemisphere = "right"
    input_file = data_dir / Path(
        "share/Inputs/CIVETNMT/src-CIVETNMT_den-41k_hemi-R_desc-nomedialwall_dparc.label.gii"
    )
    output_file_path = str(
        Path(__file__).parent / f"space-{target_space}_output_label.label.gii"
    )

    label_output = graph.surface_to_surface_transformer(
        transformer_type="label",
        input_file=input_file,
        source_space=source_space,
        target_space=target_space,
        hemisphere=hemisphere,
        output_file_path=output_file_path,
    )

    if label_output is not None:
        print(f"Transformed label saved at: {label_output.label_out}")
    else:
        print("Label surface-to-surface transformation failed.")

    # see updated graph
    surface_subgraph = graph.get_subgraph("surface_to_surface")
    plot_graph(
        surface_subgraph,
        graph_type="surface",
        layout="kamada_kawai",
        save_path=Path("examples/updated_neuromaps_surface.png"),
    )

    # see new edge/transform added to graph
    available_transforms = graph.search_surface_transforms(
        source_space=source_space,
        target_space=target_space,
        hemisphere=hemisphere,
    )
    print(
        f"Available surface-to-surface transforms after addition: "
        f"{available_transforms}"
    )
    transform = graph.fetch_surface_to_surface_transform(
        source=source_space,
        target=target_space,
        density="41k",
        hemisphere=hemisphere,
        resource_type="sphere",
    )

    # Metric resample with source and target densities specified
    source_space = "CIVETNMT"
    target_space = "S1200"
    source_density = "41k"
    target_density = "10k"
    hemisphere = "right"

    input_file = data_dir / Path(
        "share/Inputs/CIVETNMT/src-CIVETNMT_den-41k_hemi-R_desc-vaavg_midthickness.shape.gii"
    )
    output_file_path = str(
        Path(__file__).parent / f"space-{target_space}_output_metric.shape.gii"
    )
    metric_output = graph.surface_to_surface_transformer(
        transformer_type="metric",
        input_file=input_file,
        source_space=source_space,
        target_space=target_space,
        source_density=source_density,
        target_density=target_density,
        hemisphere=hemisphere,
        output_file_path=output_file_path,
    )

    if metric_output is not None:
        print(f"Transformed metric saved at: {metric_output.metric_out}")
    else:
        print("Metric surface-to-surface transformation failed.")
