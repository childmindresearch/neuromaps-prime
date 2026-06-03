# Neuromaps-PRIME <img src=".github/sticker.png" align="right" width="25%"/>

[![Build](https://github.com/childmindresearch/neuromaps-prime/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/neuromaps-prime/actions/workflows/test.yaml?query=branch%3Amain)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/childmindresearch/neuromaps-prime/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/neuromaps-prime)

> [!Important]
> This project is currently in active development. The API is subject to breaking
> changes without notice.

The `neuromaps-prime` toolbox integrates multiscale, multimodal annotations across NHP
brains, enabling comprehensive comparative analyses of brain organization. This package
extends the neuromaps ecosystem to provide unified access to diverse NHP brain datasets
and specialized tools for NHP-specific analyses.

## Features

- Robust transformation between NHP spaces (Yerkes19, NMT2, CIVETNMT, D99, MEBRAINS)
- Cross-species transformation between NHP and human (Yerkes19, fsLR)

## Installation

Get the newest development version via:

```sh
pip install git+https://github.com/childmindresearch/neuromaps-prime
```

## Usage

```python
from pathlib import Path
from neuromaps_prime.graph import NeuromapsGraph

graph = NeuromapsGraph(data_dir=Path("/path/to/neuromaps-data"))

# Resample a metric GIFTI from one surface space to another
output = graph.surface_to_surface_transformer(
    transformer_type="metric",
    input_file=Path("my_map.shape.gii"),
    source_space="CIVETNMT",
    target_space="S1200",
    hemisphere="right",
    output_file_path="my_map_S1200.shape.gii",
)

# Warp a volume from one template space to another
output = graph.volume_to_volume_transformer(
    input_file=Path("my_volume.nii.gz"),
    source_space="MEBRAINS",
    target_space="S1200",
    resolution="500um",
    resource_type="T1w",
    output_file_path="my_volume_S1200.nii.gz",
)
```

See the [`examples/`](examples/) directory for sample scripts:

- [`example_graph.py`](examples/example_graph.py) — graph inspection and plotting
- [`example_surface_transform.py`](examples/example_surface_transform.py) — surface-to-surface resampling and surface-to-volume projection
- [`example_volume_transform.py`](examples/example_volume_transform.py) — volume-to-volume warping and volume-to-surface projection

