"""Compute direct surface-to-surface transform error matrix."""

import logging
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from niwrap import workbench

from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)


def median_abs_signed_distance(metric_file: Path) -> float:
    """Compute median absolute signed-distance error from func.gii."""
    gii = nib.load(metric_file)
    data = np.abs(gii.darrays[0].data)
    return float(np.median(data))


def get_valid_spaces(graph: NeuromapsGraph, hemisphere: str) -> list[str]:
    """Return graph nodes that have required surface resources."""
    valid = []

    for node in graph.nodes:
        try:
            sphere = graph.fetch_surface_atlas(
                space=node,
                density=graph.find_highest_density(node),
                hemisphere=hemisphere,
                resource_type="sphere",
            )

            midthickness = graph.fetch_surface_atlas(
                space=node,
                density=graph.find_highest_density(node),
                hemisphere=hemisphere,
                resource_type="midthickness",
            )

            if sphere is not None and midthickness is not None:
                valid.append(node)

        except Exception as e:
            logger.debug("Skipping node %s due to error: %s", node, e)
            continue

    return valid


def test_surface_transform_matrix(tmp_path: Path) -> None:
    """Pairwise surface transform error matrix.

    Each entry measures:
        A_midthickness → B_midthickness
    using sphere-defined barycentric mapping (triangle coordiantes).
    """
    logging.basicConfig(level=logging.INFO)

    graph = NeuromapsGraph(
        runner="local",
        data_dir=Path("/Users/tamsin.rogers/Desktop/github/neuromaps-prime"),
    )

    hemisphere = "right"
    spaces = get_valid_spaces(graph, hemisphere)

    results = {}

    logger.info("=== BUILDING SURFACE TRANSFORM MATRIX ===")

    for src, dst in product(spaces, spaces):
        logger.info("=== %s → %s ===", src, dst)

        src_density = graph.find_highest_density(space=src)
        dst_density = graph.find_highest_density(space=dst)

        # midthickness defines the geometry of the surface,
        # so we use it as the reference for error computation
        src_surface = Path(
            graph.fetch_surface_atlas(
                space=src,
                density=src_density,
                hemisphere=hemisphere,
                resource_type="midthickness",
            ).fetch()
        )

        dst_surface = Path(
            graph.fetch_surface_atlas(
                space=dst,
                density=dst_density,
                hemisphere=hemisphere,
                resource_type="midthickness",
            ).fetch()
        )

        # sphere defines the mapping between surfaces,
        # so we use it for resampling
        src_sphere = Path(
            graph.fetch_surface_atlas(
                space=src,
                density=src_density,
                hemisphere=hemisphere,
                resource_type="sphere",
            ).fetch()
        )

        dst_sphere = Path(
            graph.fetch_surface_atlas(
                space=dst,
                density=dst_density,
                hemisphere=hemisphere,
                resource_type="sphere",
            ).fetch()
        )

        # apply transform
        out_surface = tmp_path / f"{src}_to_{dst}.surf.gii"

        area_surfs = {
            "current-area": src_surface,
            "new-area": dst_surface,
        }

        # resample because the surfaces are not the same mesh
        workbench.surface_resample(
            surface_in=src_surface,
            current_sphere=src_sphere,
            new_sphere=dst_sphere,
            method="ADAP_BARY_AREA",
            area_surfs=area_surfs,
            surface_out=str(out_surface),
        )

        # compute error between resampled surface and target surface
        error_file = tmp_path / f"{src}_to_{dst}_error.func.gii"

        # now compute vertex-wise signed distance from the
        # resampled surface to the target surface
        workbench.signed_distance_to_surface(
            surface_comp=str(out_surface),
            surface_ref=str(dst_surface),
            metric=str(error_file),
        )

        # the absolute signed distance gives us a measure of how far the resampled
        # surface is from the target surface at each vertex
        # the sign indicates the direction of error (inside vs outside)
        error = median_abs_signed_distance(error_file)
        results[(src, dst)] = error

        logger.info("Error %s → %s = %.5f", src, dst, error)

    # build matrix
    matrix = pd.DataFrame(index=spaces, columns=spaces, dtype=float)

    # loop through results and populate matrix
    for (src, dst), val in results.items():
        matrix.loc[src, dst] = val

    # directed transform error matrix
    logger.info("\n\n=== TRANSFORM ERROR MATRIX ===")
    logger.info("median surface-to-surface registration error (A → B)")
    logger.info("\n%s", matrix)
    """
    MATRIX DEFINITIONS

    1. TRANSFORM ERROR MATRIX (directed)
    M[A, B] =
        median vertex-wise absolute signed-distance error after:
        A_midthickness → B_midthickness
        using sphere-based barycentric resampling.

    Interpretation:
    - Rows = source space
    - Columns = target space
    - Asymmetric by construction (A→B ≠ B→A)

    2. ASYMMETRY MATRIX
    A[A, B] = M[A, B] - M[B, A]

    Interpretation:
    - Measures directional distortion bias
    - Large magnitude = non-invertible or unstable mapping

    3. SYMMETRIC MATRIX
    S[A, B] = (M[A, B] + M[B, A]) / 2

    Interpretation:
    - Undirected “geometric distance” between atlas spaces
    - Best representation of intrinsic atlas similarity
    """

    # assess symmetry
    asymmetry = matrix - matrix.T
    logger.info("\n\n=== ASYMMETRIC MATRIX ===")
    logger.info("directionality bias in mapping; A → B vs B → A")
    logger.info("\n%s", asymmetry)

    symmetric = (matrix + matrix.T) / 2.0
    logger.info("\n\n=== SYMMETRIC MATRIX ===")
    logger.info("average bidirectional transform error between spaces (A ↔ B)")
    logger.info("\n%s", symmetric)

    # assess global error, excluding diagonal
    mat = matrix.to_numpy()
    n = len(matrix)

    mask = ~np.eye(n, dtype=bool)
    off_diag_vals = mat[mask]

    # identify human vs NHP
    human_spaces = [s for s in spaces if s == "fsLR"]
    nhp_spaces = [s for s in spaces if s != "fsLR"]

    global_median = np.nanmedian(off_diag_vals)
    global_min = np.nanmin(off_diag_vals)
    global_max = np.nanmax(off_diag_vals)

    pairs = [
        (spaces[i], spaces[j], matrix.iloc[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    ]

    min_src, min_dst, min_val = min(pairs, key=lambda x: x[2])
    max_src, max_dst, max_val = max(pairs, key=lambda x: x[2])

    # nhp
    nhp_mask = np.array(
        [[(i in [spaces.index(s) for s in nhp_spaces]) and
        (j in [spaces.index(s) for s in nhp_spaces])]
        for i in range(n) for j in range(n)]
    ).reshape(n, n)

    nhp_off_diag = mat[np.logical_and(mask, nhp_mask)]
    nhp_median = np.nanmedian(nhp_off_diag)

    human_nhp_vals = []
    for i, si in enumerate(spaces):
        for j, sj in enumerate(spaces):
            if i != j and ((si in human_spaces and sj in nhp_spaces) or
                        (si in nhp_spaces and sj in human_spaces)):
                human_nhp_vals.append(mat[i, j])

    human_nhp_vals = np.array(human_nhp_vals) if human_nhp_vals else np.array([np.nan])
    human_nhp_median = np.nanmedian(human_nhp_vals)

    logger.info("\n\n=== OFF-DIAGONAL TRANSFORM ERROR STATS ===")
    logger.info("global median: %.5f", global_median)

    logger.info("global min: %.5f (%s → %s)", min_val, min_src, min_dst)
    logger.info("global max: %.5f (%s → %s)", max_val, max_src, max_dst)

    logger.info("NHP-only median: %.5f", nhp_median)
    logger.info("Human↔NHP median: %.5f", human_nhp_median)

    # CSV EXPORT
    csv_path = output_dir / "surface_transform_matrix.csv"
    matrix.to_csv(csv_path)
    logger.info("\n\nSaved CSV → %s", csv_path)

    # HEATMAP EXPORT
    mat = matrix.to_numpy()
    n = len(spaces)

    human_spaces = [s for s in spaces if s == "fsLR"]
    nhp_spaces = [s for s in spaces if s not in human_spaces]

    # HEATMAP
    _fig1, ax1 = plt.subplots(figsize=(8, 6))
    im1 = ax1.imshow(mat, interpolation="nearest")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(spaces, rotation=45, ha="right")
    ax1.set_yticklabels(spaces)
    ax1.set_title(
        "Surface Transform Error Matrix (Full Scale, Including fsLR)"
    )
    plt.colorbar(im1, ax=ax1)

    full_path = output_dir / "surface_transform_matrix_full.png"
    plt.tight_layout()
    plt.savefig(full_path, dpi=200)
    logger.info("Saved full-scale heatmap → %s", full_path)

    # NHP-SCALED HEATMAP
    nhp_vals = matrix.loc[nhp_spaces, nhp_spaces].to_numpy()
    vmin = 0.0
    vmax = np.nanpercentile(nhp_vals, 95)

    _fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.imshow(
        mat,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(spaces, rotation=45, ha="right")
    ax2.set_yticklabels(spaces)
    ax2.set_title(
        "Surface Transform Error Matrix (NHP-Scaled View; fsLR excluded from scaling only)"
    )
    plt.colorbar(im2, ax=ax2)

    nhp_path = output_dir / "surface_transform_matrix_nhp_scaled.png"
    plt.tight_layout()
    plt.savefig(nhp_path, dpi=200)

    logger.info("Saved NHP-scaled heatmap → %s", nhp_path)