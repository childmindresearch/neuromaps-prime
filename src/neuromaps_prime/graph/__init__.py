"""Brain template space graph and transformation pipeline.

Public API
----------
The primary entry point is :class:`NeuromapsGraph`. The submodule classes
are exported for typing, testing, and advanced use cases (e.g. injecting a
custom cache or ops instance).

Typical usage::

    from neuromaps_prime.graph import NeuromapsGraph

    graph = NeuromapsGraph(data_dir="/path/to/data")
    graph.surface_to_surface_transformer(
        transformer_type="metric",
        input_file=Path("my_map.func.gii"),
        source_space="fsLR",
        target_space="MNI152",
        hemisphere="left",
        output_file_path="my_map_MNI152.func.gii",
    )
"""

from neuromaps_prime.graph.builder import GraphBuilder
from neuromaps_prime.graph.core import NeuromapsGraph
from neuromaps_prime.graph.methods.cache import GraphCache
from neuromaps_prime.graph.methods.fetchers import GraphFetchers
from neuromaps_prime.graph.transforms.surface import SurfaceTransformOps
from neuromaps_prime.graph.transforms.volume import VolumeTransformOps
from neuromaps_prime.graph.utils import GraphUtils

__all__ = [
    "NeuromapsGraph",
    "GraphCache",
    "GraphBuilder",
    "GraphFetchers",
    "GraphUtils",
    "SurfaceTransformOps",
    "VolumeTransformOps",
]
