"""Compute direct surface-to-surface transform error matrix."""

import logging
from itertools import product
from pathlib import Path

import matplotlib.axes._axes as mpl_axes
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from niwrap import workbench

from neuromaps_prime.graph import NeuromapsGraph

logger = logging.getLogger(__name__)

output_dir = Path(__file__).resolve().parent / "surface_matrix_outputs"
output_dir.mkdir(parents=True, exist_ok=True)


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


def annotate_heatmap(ax: mpl_axes.Axes, mat: np.ndarray) -> None:
    """Annotate a heatmap with numeric values in each cell."""
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )


def test_surface_transform_matrix(tmp_path: Path) -> None:
    """Pairwise surface transform error matrix.

    Each entry measures:
        A_midthickness → B_midthickness
    using sphere-defined barycentric mapping (triangle coordiantes).
    """
    logging.basicConfig(level=logging.INFO)

    graph = NeuromapsGraph()

    hemisphere = "right"
    spaces = get_valid_spaces(graph, hemisphere)

    results = {}
    all_errors = []

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

        # load resampled surface coordinates
        resampled_gii = nib.load(out_surface)
        target_gii = nib.load(dst_surface)

        resampled_coords = resampled_gii.darrays[0].data
        target_coords = target_gii.darrays[0].data

        # vertex-wise Euclidean displacement
        vertex_errors = np.linalg.norm(
            np.asarray(resampled_coords, dtype=float)
            - np.asarray(target_coords, dtype=float),
            axis=1,
        )
        vertex_errors = np.abs(vertex_errors)

        all_errors.append(vertex_errors)

        median_err = float(np.median(vertex_errors))
        mean_err = float(np.mean(vertex_errors))
        std_err = float(np.std(vertex_errors))

        results[(src, dst)] = median_err
        logger.info(
            "Geometric distance %s → %s\n"
            "  src_surface: %s\n"
            "  dst_surface: %s\n"
            "  src_sphere:  %s\n"
            "  dst_sphere:  %s\n"
            "  median=%.5f mean=%.5f std=%.5f",
            src,
            dst,
            src_surface,
            dst_surface,
            src_sphere,
            dst_sphere,
            median_err,
            mean_err,
            std_err,
        )

    # build matrix
    matrix = pd.DataFrame(index=spaces, columns=spaces, dtype=float)

    # loop through results and populate matrix
    for (src, dst), val in results.items():
        matrix.loc[src, dst] = val
    """
    MATRIX DEFINITION

    Interpretation:
    - Rows = source space
    - Columns = target space
    - Asymmetric: M[A,B] ≠ M[B,A]

    ASYMMETRIC MATRIX
    A[A, B] = M[A, B] - M[B, A]
    """

    asymmetry = (matrix - matrix.T).abs()
    logger.info("\n\n=== ASYMMETRIC MATRIX ===")
    logger.info("A → B vs B → A")
    logger.info("\n%s", asymmetry)

    # assess global error, excluding diagonal
    mat = matrix.to_numpy()
    n = len(matrix)

    mask = ~np.eye(n, dtype=bool)
    off_diag_vals = mat[mask]

    nhp_spaces = [s for s in spaces if s not in {"fsLR", "NCBR"}]

    asymmetry_nhp = asymmetry.loc[nhp_spaces, nhp_spaces]
    global_median = np.nanmedian(off_diag_vals)

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
        [
            [
                (i in [spaces.index(s) for s in nhp_spaces])
                and (j in [spaces.index(s) for s in nhp_spaces])
            ]
            for i in range(n)
            for j in range(n)
        ]
    ).reshape(n, n)

    nhp_off_diag = mat[np.logical_and(mask, nhp_mask)]
    nhp_median = np.nanmedian(nhp_off_diag)

    logger.info("\n\n=== OFF-DIAGONAL TRANSFORM ERROR STATS ===")
    logger.info("global median: %.5f", global_median)
    logger.info("global min: %.5f (%s → %s)", min_val, min_src, min_dst)
    logger.info("global max: %.5f (%s → %s)", max_val, max_src, max_dst)
    logger.info("NHP-only median: %.5f", nhp_median)

    # CSV EXPORT
    csv_path = output_dir / "surface_transform_matrix.csv"
    matrix.to_csv(csv_path)
    logger.info("\n\nSaved CSV → %s", csv_path)

    # ============================================================
    # ASYMMETRIC MATRIX HEATMAP
    # ============================================================

    asym_mat = asymmetry_nhp.to_numpy(dtype=float)
    asym_mat = np.abs(asym_mat)

    _fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.nanmax(asym_mat)

    im = ax.imshow(
        asym_mat,
        cmap="turbo",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )

    annotate_heatmap(ax, asym_mat)

    ax.set_xticks(range(len(nhp_spaces)))
    ax.set_yticks(range(len(nhp_spaces)))
    ax.set_xticklabels(nhp_spaces, rotation=45, ha="right")
    ax.set_yticklabels(nhp_spaces)
    ax.set_xlabel("Target template space")
    ax.set_ylabel("Source template space")
    ax.set_title(
        "Transform Error Matrix (Geometric Distance, NHP Only)",
        fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Geometric discrepancy (mm)")

    plt.tight_layout()
    plt.savefig(
        output_dir / "surface_transform_matrix_asymmetric.png",
        dpi=200,
    )
    plt.close()

    # HISTOGRAM
    all_errors = np.concatenate(all_errors)
    plt.figure()
    max_val = np.nanmax(all_errors)
    plt.hist(all_errors, bins=200)
    plt.xlim(0, max_val)
    plt.title(
        "Vertex-wise Surface Transform Error Distribution",
        fontweight="bold",
        color="#0044AA",
        pad=10,
    )
    plt.xlabel("Vertex displacement (mm)", labelpad=12)
    plt.ylabel("Vertex count", labelpad=12)
    fig4_caption = (
        "Figure 4. Distribution of vertex-wise geometric displacements"
        " across all pairwise surface transformations."
    )
    hist_path = output_dir / "surface_transform_histogram.png"
    plt.subplots_adjust(left=0.15, bottom=0.30)
    plt.figtext(
        0.5,
        0.02,
        fig4_caption,
        ha="center",
        fontsize=9,
        fontstyle="italic",
        wrap=True,
    )
    plt.savefig(hist_path, dpi=200)
    plt.close()
    logger.info("Saved global histogram → %s", hist_path)
