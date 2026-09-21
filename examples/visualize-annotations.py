"""Create an outline figure showing resources and annotations for each space."""

from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

NODES_DIR = (
    REPO_ROOT
    / "src"
    / "neuromaps_prime"
    / "resources"
    / "nodes"
)

OUTPUT_FILE = REPO_ROOT / "annotation_outline.png"

CARD_WIDTH = 1.8
CARD_HEIGHT = 0.7
CARD_GAP = 0.2
SPACE_WIDTH = 4.4
ROW_GAP = 0.2

IGNORE_SPECIES = {"marmoset"}


def parse_nodes(nodes_dir: Path):
    """Parse all node YAML files, organized by species folder."""
    spaces = {}

    for species_dir in sorted(nodes_dir.iterdir()):
        if not species_dir.is_dir():
            continue

        species = species_dir.name

        if species.lower() in IGNORE_SPECIES:
            continue

        yaml_files = sorted(
            list(species_dir.glob("*.yml"))
            + list(species_dir.glob("*.yaml"))
        )

        for yaml_file in yaml_files:
            data = yaml.safe_load(
                yaml_file.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(data, dict):
                continue

            for space_name, space_data in data.items():
                if not isinstance(space_data, dict):
                    continue

                annotations = []

                surfaces = space_data.get(
                    "surfaces",
                    {},
                )

                volumes = space_data.get(
                    "volumes",
                    {},
                )

                # ------------------------------------------------------
                # Annotations
                # ------------------------------------------------------

                if isinstance(surfaces, dict):
                    for density, density_data in surfaces.items():
                        if not isinstance(
                            density_data,
                            dict,
                        ):
                            continue

                        annotation_data = (
                            density_data.get(
                                "annotation",
                                {},
                            )
                        )

                        if not isinstance(
                            annotation_data,
                            dict,
                        ):
                            continue

                        for (
                            annotation_name,
                            annotation,
                        ) in annotation_data.items():

                            if not isinstance(
                                annotation,
                                dict,
                            ):
                                continue

                            hemispheres = []

                            if "left" in annotation:
                                hemispheres.append("L")

                            if "right" in annotation:
                                hemispheres.append("R")

                            annotations.append(
                                {
                                    "name": str(
                                        annotation_name
                                    ),
                                    "density": str(
                                        density
                                    ),
                                    "hemispheres": (
                                        hemispheres
                                    ),
                                }
                            )

                # ------------------------------------------------------
                # Store space
                # ------------------------------------------------------

                spaces[str(space_name)] = {
                    "species": species,
                    "surfaces": surfaces,
                    "volumes": volumes,
                    "annotations": annotations,
                }

    return spaces


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def draw_card(
    ax,
    x,
    y,
    width,
    height,
    text,
):
    """Draw a fixed-size card."""
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=(
            "round,pad=0.03,"
            "rounding_size=0.05"
        ),
        fill=False,
        linestyle="--",
        linewidth=1,
    )

    ax.add_patch(card)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
    )


def draw_lr_cards(
    ax,
    x,
    y,
    name,
    hemispheres,
):
    """Draw fixed L/R cards for a resource."""
    left_x = x
    right_x = (
        x
        + CARD_WIDTH
        + CARD_GAP
    )

    # Always reserve both positions.
    draw_card(
        ax,
        left_x,
        y,
        CARD_WIDTH,
        CARD_HEIGHT,
        "L" if "L" in hemispheres else "",
    )

    draw_card(
        ax,
        right_x,
        y,
        CARD_WIDTH,
        CARD_HEIGHT,
        "R" if "R" in hemispheres else "",
    )

    ax.text(
        x - 0.15,
        y + CARD_HEIGHT / 2,
        name,
        ha="right",
        va="center",
        fontsize=9,
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_plot(
    spaces,
    output_file,
):
    """Create the outline figure."""
    species_groups = {}

    for space_name, space in spaces.items():
        species = space["species"]

        species_groups.setdefault(
            species,
            [],
        ).append(space_name)

    for species in species_groups:
        species_groups[species].sort()

    # Estimate figure dimensions.
    total_rows = 0

    for species_spaces in species_groups.values():
        for space_name in species_spaces:
            space = spaces[space_name]

            rows = 2  # Surface + Annotation + Volume headings

            surfaces = space["surfaces"]

            for density_data in surfaces.values():
                if isinstance(
                    density_data,
                    dict,
                ):
                    rows += sum(
                        1
                        for key in density_data
                        if key != "annotation"
                    )

            rows += len(
                space["annotations"]
            )

            volumes = space["volumes"]

            for density_data in volumes.values():
                if isinstance(
                    density_data,
                    dict,
                ):
                    rows += len(
                        density_data
                    )

            total_rows = max(
                total_rows,
                rows,
            )

    max_spaces = max(
        (
            len(species_spaces)
            for species_spaces
            in species_groups.values()
        ),
        default=1,
    )

    figure_width = max(
        12,
        3
        + max_spaces
        * SPACE_WIDTH,
    )

    figure_height = max(
        8,
        3
        + total_rows
        * (
            CARD_HEIGHT
            + ROW_GAP
        ),
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    ax.set_axis_off()

    y = (
        figure_height
        - 0.8
    )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    ax.text(
        0,
        y,
        "Neuromaps-PRIME",
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
    )

    y -= 0.65

    # ------------------------------------------------------------------
    # Species
    # ------------------------------------------------------------------

    for species_index, (
        species,
        species_spaces,
    ) in enumerate(
        species_groups.items()
    ):
        if species_index:
            y -= 0.7

        ax.text(
            0,
            y,
            species,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )

        y -= 0.55

        # --------------------------------------------------------------
        # Space columns
        # --------------------------------------------------------------

        for space_index, space_name in enumerate(
            species_spaces
        ):
            space = spaces[space_name]

            x = (
                3.0
                + space_index
                * SPACE_WIDTH
            )

            current_y = y

            # Space title
            ax.text(
                x
                + SPACE_WIDTH / 2,
                current_y,
                space_name,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

            current_y -= 0.5

            # Hemisphere headings
            left_x = x + 0.2

            right_x = (
                left_x
                + CARD_WIDTH
                + CARD_GAP
            )

            ax.text(
                left_x
                + CARD_WIDTH / 2,
                current_y,
                "L",
                fontsize=8,
                fontweight="bold",
                ha="center",
            )

            ax.text(
                right_x
                + CARD_WIDTH / 2,
                current_y,
                "R",
                fontsize=8,
                fontweight="bold",
                ha="center",
            )

            current_y -= 0.35

            # ----------------------------------------------------------
            # Surfaces
            # ----------------------------------------------------------

            ax.text(
                x,
                current_y,
                "Surface",
                fontsize=8,
                fontweight="bold",
                alpha=0.6,
                ha="left",
            )

            current_y -= 0.35

            for density, density_data in (
                space["surfaces"].items()
            ):
                if not isinstance(
                    density_data,
                    dict,
                ):
                    continue

                for surface_name, surface in (
                    density_data.items()
                ):
                    if surface_name == "annotation":
                        continue

                    if not isinstance(
                        surface,
                        dict,
                    ):
                        continue

                    hemispheres = []

                    if "left" in surface:
                        hemispheres.append("L")

                    if "right" in surface:
                        hemispheres.append("R")

                    draw_lr_cards(
                        ax,
                        x + 0.2,
                        current_y,
                        f"{surface_name} ({density})",
                        hemispheres,
                    )

                    current_y -= (
                        CARD_HEIGHT
                        + ROW_GAP
                    )

            # ----------------------------------------------------------
            # Annotations
            # ----------------------------------------------------------

            ax.text(
                x,
                current_y,
                "Annotation",
                fontsize=8,
                fontweight="bold",
                alpha=0.6,
                ha="left",
            )

            current_y -= 0.35

            for annotation in space[
                "annotations"
            ]:
                draw_lr_cards(
                    ax,
                    x + 0.2,
                    current_y,
                    annotation["name"],
                    annotation["hemispheres"],
                )

                current_y -= (
                    CARD_HEIGHT
                    + ROW_GAP
                )

            # ----------------------------------------------------------
            # Volumes
            # ----------------------------------------------------------

            ax.text(
                x,
                current_y,
                "Volume",
                fontsize=8,
                fontweight="bold",
                alpha=0.6,
                ha="left",
            )

            current_y -= 0.35

            for density, density_data in (
                space["volumes"].items()
            ):
                if not isinstance(
                    density_data,
                    dict,
                ):
                    continue

                for volume_name in density_data:
                    draw_card(
                        ax,
                        x + 0.2,
                        current_y,
                        CARD_WIDTH * 2
                        + CARD_GAP,
                        CARD_HEIGHT,
                        f"{volume_name} ({density})",
                    )

                    current_y -= (
                        CARD_HEIGHT
                        + ROW_GAP
                    )

        y -= (
            total_rows
            * (
                CARD_HEIGHT
                + ROW_GAP
            )
            + 0.4
        )

    ax.set_xlim(
        -0.5,
        3.0
        + max_spaces
        * SPACE_WIDTH,
    )

    ax.set_ylim(
        0,
        figure_height,
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure to {output_file}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Parse node YAML files and create the figure."""
    print(
        f"Reading nodes from: {NODES_DIR}"
    )

    spaces = parse_nodes(
        NODES_DIR
    )

    print(
        f"Found {len(spaces)} spaces"
    )

    for space_name, space in spaces.items():
        print(
            f"\n{space['species']} / "
            f"{space_name}"
        )

        print("  annotations:")

        for annotation in space[
            "annotations"
        ]:
            print(
                f"    - {annotation['name']}: "
                f"{', '.join(annotation['hemispheres'])}"
            )

    make_plot(
        spaces,
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
