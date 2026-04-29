"""Bundled resources for Neuromaps-PRIME.

Provides resolved paths to:

- Default nodes and edges
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

_ROOT = Path(__file__).parent.resolve()


class NeuromapsPrimeYAML(NamedTuple):
    """Path to default Neuromaps-PRIME graph resources.

    Attributes:
        yaml: Default graph framework for neuromaps-prime.
    """

    nodes: tuple[Path, ...]
    surface_edges: tuple[Path, ...]
    volume_edges: tuple[Path, ...]


def _rglob(subdir: str) -> tuple[Path, ...]:
    return tuple(sorted((_ROOT / subdir).rglob("*.yaml")))


NEUROMAPSPRIME_GRAPH = NeuromapsPrimeYAML(
    nodes=_rglob("nodes"),
    surface_edges=_rglob("edges/surface"),
    volume_edges=_rglob("edges/volume"),
)


__all__ = ["NEUROMAPSPRIME_GRAPH"]
