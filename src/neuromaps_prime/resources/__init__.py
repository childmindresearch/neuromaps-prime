"""Bundled resources for Neuromaps-PRIME.

Provides resolved paths to:

- Default nodes and edges
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

_ROOT = Path(__file__).parent.resolve()


class NeuromapsPrimeYAML(NamedTuple):
    """Path to default Neuromaps-PRIME graph config.

    Attributes:
        yaml: Default graph framework for neuromaps-prime.
    """

    yaml: Path


NEUROMAPSPRIME_GRAPH = NeuromapsPrimeYAML(yaml=_ROOT / "neuromaps_graph.yaml")


__all__ = ["NEUROMAPSPRIME_GRAPH"]
