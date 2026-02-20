"""Compute surface-to-surface transform error."""

import pytest
from pathlib import Path
from typing import List
import nibabel as nib

from networkx.algorithms.cycles import recursive_simple_cycles

from neuromaps_prime.graph import NeuromapsGraph
from niwrap import workbench
import numpy as np
from neuromaps_prime.utils import (
    extract_vertex_only,
    merge_vertices_with_faces,
    log_gii_shapes,
)

@pytest.mark.usefixtures("require_workbench")
def test_surface_cycle(tmp_path: Path) -> None:
    """Test surface consistency by cycling through transforms and computing signed distance errors."""

    # build a NeuromapsGraph object
    print("Building NeuromapsGraph from YAML")
    graph = NeuromapsGraph(
        yaml_file=Path("src/neuromaps_prime/datasets/data/neuromaps_graph.yaml")
    )

    # set origin node (can change this)
    origin = "Yerkes19"
    hemisphere = "right"
    density = graph.find_highest_density(space=origin)
    print(f"Origin={origin}, hemisphere={hemisphere}, density={density}")

    # build subgraph of just surface-to-surface transforms
    NETWORKX_GRAPH = graph.get_subgraph(edges="surface_to_surface")
    print(f"Graph has {len(NETWORKX_GRAPH.nodes)} nodes and {len(NETWORKX_GRAPH.edges)} edges")

    # build a list of cycles containing the origin node
    directed_graph = NETWORKX_GRAPH.to_directed()
    cycles: List[List[str]] = [
        cycle for cycle in recursive_simple_cycles(directed_graph)
        if origin in cycle
    ]

    assert cycles, "no cycles found"
    print(f"Found {len(cycles)} cycles containing {origin}")

    # fetch the actual src-Yerkes19_den-32k_hemi-R_sphere.surf.gii
    full_surface = graph.fetch_surface_atlas(
        space=origin,
        density=density,
        hemisphere=hemisphere,
        resource_type="sphere",
    ).fetch()
    log_gii_shapes(Path(full_surface))

    # loop over available cycles
    for i, cycle in enumerate(cycles):

        # rotate cycle to start at origin
        while cycle[0] != origin:
            cycle = cycle[1:] + cycle[:1]
        cycle = cycle + [origin]
        print(f"\nCycle {i}: {cycle}")

        # extract vertex coordinates
        vertex_surface = tmp_path / f"cycle{i}_start_vertex.surf.gii"
        current_surface = extract_vertex_only(Path(full_surface), vertex_surface)
        shapes = log_gii_shapes(Path(current_surface))
        if not shapes:
            print(f"[WARN] Starting surface for cycle {i} has no vertices. Skipping cycle.")
            continue

        skip_cycle = False

        # the "steps"

        """
        running 
        list(zip(cycle[:-1], cycle[1:])) 
        
        on 
        cycle = ['A', 'B', 'C', 'D']

        gives 
        [('A','B'), ('B','C'), ('C','D')]
        """

        for step, (src, dst) in enumerate(zip(cycle[:-1], cycle[1:])):
            out_file = tmp_path / f"cycle{i}_step{step}.surf.gii"
            print(f"Transform {src} -> {dst}, output: {out_file.name}")

            # surface-to-surface transforms
            current_surface = graph.surface_to_surface_transformer(
                transformer_type="metric",
                input_file=current_surface,
                source_space=src,
                target_space=dst,
                hemisphere=hemisphere,
                output_file_path=str(out_file),
            )

            # only count 3D surfaces
            shapes = log_gii_shapes(Path(current_surface))
            if not shapes:
                print(f"[WARNING] The {src} -> {dst} transform is an empty surface. Skipping cycle {i}.")
                skip_cycle = True
                break

        if skip_cycle:
            continue

        # workbench expects a full mesh, not just vertices
        full_transformed = tmp_path / f"cycle{i}_full_transformed.surf.gii"
        full_transformed = merge_vertices_with_faces(
            current_surface,
            full_surface,
            full_transformed,
        )

        # should be something like 32492 vertices
        shapes = log_gii_shapes(Path(full_transformed))
        if len(shapes) != 1:
            print(f"[WARN] Full transformed surface has unexpected vertex arrays. Skipping cycle {i}.")
            continue

        # metric file for error output
        error_file = tmp_path / f"cycle{i}_error.func.gii"

        # compute signed distance
        workbench.signed_distance_to_surface(
            surface_comp=str(full_transformed),
            surface_ref=str(full_surface),
            metric=str(error_file),
        )

        # load metric file
        error_gii = nib.load(error_file)

        # absolute value
        error_data = np.abs(error_gii.darrays[0].data)

        # compute median absolute value
        median_error = np.median(error_data)

        print(f"MEDIAN ERROR: {median_error}")
        assert median_error < 1.0, f"Median error in {cycle} exceeds 1: {median_error}"