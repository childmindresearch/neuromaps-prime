"""Statistical analysis and null-distribution tools.

Provides functions for computing correlations, statistical tests, and
null-distribution metrics over pairs of neuroimaging maps. Designed to be
graph-agnostic: functions accept NumPy arrays and file paths, and any
integration with the brain-map graph happens externally.
"""

from neuromaps_prime.analysis.parcels import ParcelSummary, parcel_reduce
from neuromaps_prime.analysis.stats import (
    compare_images,
    efficient_pearsonr,
    permtest_metric,
)

__all__ = [
    "ParcelSummary",
    "compare_images",
    "efficient_pearsonr",
    "parcel_reduce",
    "permtest_metric",
]
