
from pathlib import Path

from styxdefs import set_global_runner
from styxsingularity import SingularityRunner


class Config:
    container_path = Path("/home/bshrestha/projects/Tfunck/neuromaps.sif")
    my_runner = SingularityRunner(
        images={"brainlife/connectome_workbench:1.5.0-freesurfer-update": container_path}
    )
    data_dir = Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp/.temp-niwrap-data").resolve()
    my_runner.data_dir = data_dir

    def __init__(self):
        set_global_runner(self.my_runner)