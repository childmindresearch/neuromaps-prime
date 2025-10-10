"""Top-level package for neuromaps_nhp.

This package provides tools for transforming and analyzing neuroimaging data
in non-human primates.
"""

from .config import Config
from .utils import gifti_utils, niwrap_wrappers

config = Config()
