"""Global pytest fixtures, arguments, and options."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.resources import NEUROMAPSPRIME_GRAPH

if TYPE_CHECKING:
    from collections.abc import Sequence


def pytest_collection_modifyitems(items: Sequence[pytest.Item]) -> None:
    """Apply appropriate markers based on test location."""
    markers = {"unit", "integration", "regression"}

    for item in items:
        test_path = Path(item.fspath)
        for marker in markers & set(test_path.parts):
            item.add_marker(getattr(pytest.mark, marker))


def _mk(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.touch()
    return str(path)


def _rewrite_surface_node(
    tmp_path: Path, node_name: str, density: str, surf_type: str, hemis: dict[str, Any]
) -> None:
    if surf_type == "annotation":
        for label, hemi_paths in hemis.items():
            for hemi in hemi_paths:
                if hemi in ("notes", "references"):
                    continue
                hemi_paths[hemi] = _mk(
                    tmp_path, f"{node_name}_{density}_{hemi}_{label}.func.gii"
                )
    else:
        for hemi in hemis:
            hemis[hemi] = _mk(
                tmp_path, f"{node_name}_{density}_{hemi}_{surf_type}.surf.gii"
            )


def _rewrite_volume_node(
    tmp_path: Path, node_name: str, res: str, vol_type: str, types: dict[str, Any]
) -> None:
    if vol_type == "annotation":
        for label in types[vol_type]:
            types[vol_type][label]["uri"] = _mk(
                tmp_path, f"{node_name}_{res}_{label}.nii.gz"
            )
    else:
        types[vol_type] = _mk(tmp_path, f"{node_name}_{res}_{vol_type}.nii.gz")


def _rewrite_node_files(tmp_path: Path, node_name: str, node: dict[str, Any]) -> None:
    for density, types in node.get("surfaces", {}).items():
        for surf_type, hemis in types.items():
            _rewrite_surface_node(tmp_path, node_name, density, surf_type, hemis)

    for res, types in node.get("volumes", {}).items():
        for vol_type in list(types):
            _rewrite_volume_node(tmp_path, node_name, res, vol_type, types)


def _rewrite_edge_volumes(
    tmp_path: Path, src: str, dst: str, volumes: dict[str, Any]
) -> None:
    for provider, res_dict in volumes.items():
        for res, types in res_dict.items():
            if res == "references":
                continue
            for vol_type in list(types):
                types[vol_type] = _mk(
                    tmp_path, f"{src}_to_{dst}_{provider}_{res}_{vol_type}.nii.gz"
                )


def _rewrite_edge_surfaces(
    tmp_path: Path, src: str, dst: str, surfaces: dict[str, Any]
) -> None:
    for provider, density_dict in surfaces.items():
        for density, types in density_dict.items():
            if density == "references":
                continue
            for surf_type, hemis in types.items():
                for hemi in list(hemis):
                    hemis[hemi] = _mk(
                        tmp_path,
                        f"{src}_to_{dst}_{provider}_{density}_{hemi}_{surf_type}.surf.gii",
                    )


def _rewrite_edge_files(tmp_path: Path, edge: dict[str, Any]) -> None:
    src, dst = edge["from"], edge["to"]
    _rewrite_edge_surfaces(tmp_path, src, dst, edge.get("surfaces", {}))
    _rewrite_edge_volumes(tmp_path, src, dst, edge.get("volumes", {}))


@pytest.fixture
def graph(tmp_path: Path) -> NeuromapsGraph:
    """Create a graph fixture to use for tests."""
    graph = NeuromapsGraph(runner="auto", data_dir=tmp_path / ".cache", _testing=True)

    def _load_dict(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
        return [yaml.safe_load(path.read_bytes()) for path in paths]

    def _load_list(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
        merged = []
        for path in paths:
            merged.extend(yaml.safe_load(path.read_bytes()))
        return merged

    data = {
        "nodes": _load_dict(NEUROMAPSPRIME_GRAPH.nodes),
        "edges": {
            "surface_to_surface": _load_list(NEUROMAPSPRIME_GRAPH.surface_edges),
            "volume_to_volume": _load_list(NEUROMAPSPRIME_GRAPH.volume_edges),
        },
    }

    for node_block in data.get("nodes", {}):
        for node_name, node in node_block.items():
            _rewrite_node_files(tmp_path, node_name, node)

    for edge_list in data.get("edges", {}).values():
        for edge in edge_list:
            _rewrite_edge_files(tmp_path, edge)

    graph._builder.build_from_dict(graph=graph, data=data)
    return graph
