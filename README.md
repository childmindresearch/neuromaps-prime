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
    input_file=Path("src-CIVETNMT_den-41k_hemi-R_desc-vaavg_midthickness.shape.gii"),
    source_space="CIVETNMT",
    target_space="Yerkes19",
    hemisphere="right",
    source_density="41k",
    target_density="32k",
    output_file_path="space-Yerkes19_output_metric.shape.gii",
)
```

See the [`examples/`](examples/) directory for sample scripts:

- [`example_graph.py`](examples/example_graph_init.py) — graph inspection and plotting
- [`example_surface_transform.py`](examples/example_surface_transform.py) — surface-to-surface resampling and surface-to-volume projection
- [`example_volume_transform.py`](examples/example_volume_transform.py) — volume-to-volume warping and volume-to-surface projection

