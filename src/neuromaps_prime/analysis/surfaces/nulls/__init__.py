"""Surrogate generation for null-distribution estimation.

Provides methods for producing spatially structured surrogate maps that
preserve the autocorrelation properties of cortical data while destroying
meaningful signal.  These surrogates form the basis of permutation-based
null distributions for statistical inference on brain surfaces.
"""

from neuromaps_prime.analysis.surfaces.nulls import burt, spins

__all__ = ["burt", "spins"]
