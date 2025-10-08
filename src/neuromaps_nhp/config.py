"""Configuration Module."""

from pathlib import Path

from niwrap import get_global_runner, use_singularity


class Config:
    """Configuration settings for neuromaps_nhp."""

    def __init__(self) -> None:
        """Initialize configuration settings."""
        self.container = Path("/home/bshrestha/projects/Tfunck/neuromaps.sif")
        self.data_dir = Path(
            "/home/bshrestha/projects/neuromaps_nhp/styx_tmp"
        ).resolve()

        images_dict = {
            "brainlife/connectome_workbench:1.5.0-freesurfer-update": self.container,
        }

        use_singularity(images=images_dict, data_dir=self.data_dir)
        self.runner = get_global_runner()
