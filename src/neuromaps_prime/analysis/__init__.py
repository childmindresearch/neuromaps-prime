"""Statistical analysis and null-distribution tools.

Provides functions for computing correlations, statistical tests, and
null-distribution metrics over pairs of neuroimaging maps. Designed to be
graph-agnostic: functions accept NumPy arrays and file paths, and any
integration with the brain-map graph happens externally.
"""

from neuromaps_prime.analysis.stats import efficient_pearsonr, permtest_metric

__all__ = ["efficient_pearsonr", "permtest_metric"]
