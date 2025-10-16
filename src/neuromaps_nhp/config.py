"""Configuration module for neuromaps-nhp library."""
import pandas as pd
from pathlib import Path

from styxdefs import set_global_runner
from styxsingularity import SingularityRunner

from neuromaps_nhp.resources.build_paths import Paths

class Config():

    def __init__(self):
        self.my_runner = SingularityRunner(
            images={"brainlife/connectome_workbench:1.5.0-freesurfer-update": "/home/bshrestha/projects/Tfunck/neuromaps.sif"}
        )
        self.data_dir = Path("/home/bshrestha/projects/neuromaps_nhp/styx_tmp").resolve()
        self.my_runner.data_dir = self.data_dir

        self.paths = Paths()
        self.transforms_df = pd.read_csv('/home/bshrestha/projects/neuromaps_nhp/src/neuromaps_nhp/resources/transforms_paths.csv')

config = Config()
set_global_runner(config.my_runner)