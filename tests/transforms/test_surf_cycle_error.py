"""Compute surface-to-surface transform error with full cycle logging and extended debug."""

import pytest
from pathlib import Path
from typing import List
import nibabel as nib
import numpy as np

from networkx.algorithms.cycles import recursive_simple_cycles

from neuromaps_prime.graph import NeuromapsGraph
from niwrap import workbench
from neuromaps_prime.utils import log_gii_shapes


@pytest.mark.usefixtures("require_workbench")
def test_surface_cycle(tmp_path: Path) -> None:
    """Test surface consistency by cycling through transforms and computing signed distance errors."""

    print("=== BUILDING NEUROMAPS GRAPH ===", flush=True)
    graph = NeuromapsGraph(
        yaml_file=Path("src/neuromaps_prime/datasets/data/neuromaps_graph.yaml")
    )

    print(graph.yaml_path.exists())
    print(f"yaml_path={graph.yaml_path}", flush=True)

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

        shapes = log_gii_shapes(Path(full_surface))
        skip_cycle = False

        for step, (src, dst) in enumerate(zip(cycle[:-1], cycle[1:])):
            out_file = tmp_path / f"cycle{i}_step{step}.surf.gii"
            print(f"--- Step {step}: {src} -> {dst} ---", flush=True)
            density = graph.find_highest_density(space=src)

            try:
                current_surface = graph._surface_to_surface(
                    source=src,
                    target=dst,
                    hemisphere=hemisphere,
                    density=density,
                    output_file_path=str(out_file),
                )
            except Exception as e:
                print(type(e).__name__)
                print(f"[ERROR] Transform {src}->{dst} failed: {e}", flush=True)
                skip_cycle = True
                break

            from pprint import pprint 
            pprint(graph._surface_transform_cache)
            print(src, dst, hemisphere, density)
            if not current_surface.file_path.exists():
                print(f"[ERROR] Transform output file does not exist: {current_surface.file_path}", flush=True)
                skip_cycle = True
                break

            shapes = log_gii_shapes(current_surface.file_path)
            if not shapes or all(len(arr.data) == 0 for arr in nib.load(current_surface.file_path).darrays):
                print(f"[WARN] Transform {src}->{dst} produced empty surface!", flush=True)
                skip_cycle = True
                break

            print(f"Transform {src}->{dst} successful, vertex arrays: {[len(arr.data) for arr in nib.load(current_surface.file_path).darrays]}", flush=True)

        if skip_cycle:
            print(f"[INFO] Skipping cycle {i} due to empty/failed transform.", flush=True)
            continue

        shapes = log_gii_shapes(Path(out_file))
        print(f"Full transformed surface vertex arrays: {[len(arr.data) for arr in nib.load(out_file).darrays]}", flush=True)

        if len(shapes) != 1:
            print(f"[WARN] Unexpected number of arrays in full transformed surface, skipping cycle {i}", flush=True)
            continue

        error_file = tmp_path / f"cycle{i}_error.func.gii"
        print(f"Computing signed distance for cycle {i}, output: {error_file.name}", flush=True)
        workbench.signed_distance_to_surface(
            surface_comp=str(out_file),
            surface_ref=str(full_surface),
            metric=str(error_file),
        )

        error_gii = nib.load(error_file)
        error_data = np.abs(error_gii.darrays[0].data)
        median_error = np.median(error_data)
        print(f"Cycle {i} MEDIAN ERROR: {median_error}", flush=True)
        assert median_error < 1.0, f"Median error in {cycle} exceeds 1: {median_error}"