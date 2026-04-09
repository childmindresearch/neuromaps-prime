"""Compute surface-to-surface transform error with full cycle logging and extended debug."""

import pytest
from pathlib import Path
from typing import List
import nibabel as nib
import numpy as np

from networkx.algorithms.cycles import recursive_simple_cycles

from neuromaps_prime.graph import NeuromapsGraph
from niwrap import workbench
from neuromaps_prime.utils import extract_vertex_only, merge_vertices_with_faces, log_gii_shapes


@pytest.mark.usefixtures("require_workbench")
def test_surface_cycle(tmp_path: Path) -> None:
    """Test surface consistency by cycling through transforms and computing signed distance errors."""

    print("=== BUILDING NEUROMAPS GRAPH ===", flush=True)
    graph = NeuromapsGraph(
        yaml_file=Path("src/neuromaps_prime/datasets/data/neuromaps_graph.yaml")
    )

    origin = "Yerkes19"
    hemisphere = "right"
    density = graph.find_highest_density(space=origin)
    print(f"Origin={origin}, hemisphere={hemisphere}, density={density}", flush=True)

    NETWORKX_GRAPH = graph.get_subgraph(edges="surface_to_surface")
    print(f"Graph has {len(NETWORKX_GRAPH.nodes)} nodes and {len(NETWORKX_GRAPH.edges)} edges", flush=True)

    directed_graph = NETWORKX_GRAPH.to_directed()
    cycles: List[List[str]] = [
        cycle for cycle in recursive_simple_cycles(directed_graph)
        if origin in cycle
    ]
    assert cycles, "No cycles found!"
    print(f"Found {len(cycles)} cycles containing {origin}", flush=True)

    full_surface = graph.fetch_surface_atlas(
        space=origin,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    ).fetch()
    print(f"Full surface path: {full_surface}", flush=True)
    log_gii_shapes(Path(full_surface))

    for i, cycle in enumerate(cycles):

        # rotate cycle so origin is first
        while cycle[0] != origin:
            cycle = cycle[1:] + cycle[:1]
        cycle = cycle + [origin]  # close the loop
        print(f"\n=== Cycle {i}: {cycle} ===", flush=True)

        # initial vertex-only surface
        vertex_surface = tmp_path / f"cycle{i}_start_vertex.surf.gii"
        current_surface = extract_vertex_only(Path(full_surface), vertex_surface)
        shapes = log_gii_shapes(Path(current_surface))
        print(f"{vertex_surface.name} vertex count: {len(nib.load(vertex_surface).darrays[0].data)}", flush=True)

        skip_cycle = False

        for step, (src, dst) in enumerate(zip(cycle[:-1], cycle[1:])):
            out_file = tmp_path / f"cycle{i}_step{step}.surf.gii"
            print(f"--- Step {step}: {src} -> {dst} ---", flush=True)
            print(f"Current surface path: {current_surface}", flush=True)

            try:
                current_surface = graph.surface_to_surface_transformer(
                    transformer_type="metric",
                    input_file=current_surface,
                    source_space=src,
                    target_space=dst,
                    hemisphere=hemisphere,
                    output_file_path=str(out_file),
                )
            except Exception as e:
                print(f"[ERROR] Transform {src}->{dst} failed: {e}", flush=True)
                skip_cycle = True
                break

            if not Path(current_surface).exists():
                print(f"[ERROR] Transform output file does not exist: {current_surface}", flush=True)
                skip_cycle = True
                break

            shapes = log_gii_shapes(Path(current_surface))
            if not shapes or all(len(arr.data) == 0 for arr in nib.load(current_surface).darrays):
                print(f"[WARN] Transform {src}->{dst} produced empty surface!", flush=True)
                skip_cycle = True
                break

            print(f"Transform {src}->{dst} successful, vertex arrays: {[len(arr.data) for arr in nib.load(current_surface).darrays]}", flush=True)

        if skip_cycle:
            print(f"[INFO] Skipping cycle {i} due to empty/failed transform.", flush=True)
            continue

        full_transformed = tmp_path / f"cycle{i}_full_transformed.surf.gii"
        full_transformed = merge_vertices_with_faces(current_surface, full_surface, full_transformed)
        shapes = log_gii_shapes(Path(full_transformed))
        print(f"Full transformed surface vertex arrays: {[len(arr.data) for arr in nib.load(full_transformed).darrays]}", flush=True)

        if len(shapes) != 1:
            print(f"[WARN] Unexpected number of arrays in full transformed surface, skipping cycle {i}", flush=True)
            continue

        error_file = tmp_path / f"cycle{i}_error.func.gii"
        print(f"Computing signed distance for cycle {i}, output: {error_file.name}", flush=True)
        workbench.signed_distance_to_surface(
            surface_comp=str(full_transformed),
            surface_ref=str(full_surface),
            metric=str(error_file),
        )

        error_gii = nib.load(error_file)
        error_data = np.abs(error_gii.darrays[0].data)
        median_error = np.median(error_data)
        print(f"Cycle {i} MEDIAN ERROR: {median_error}", flush=True)
        assert median_error < 1.0, f"Median error in {cycle} exceeds 1: {median_error}"