"""Top-level package for neuromaps_nhp.

This package provides tools for transforming and analyzing neuroimaging data
in non-human primates.
"""

from .config import Config
from .utils import gifti_utils as gifti_utils
from .utils import niwrap_wrappers as niwrap_wrappers

config = Config()
