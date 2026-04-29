"""Global pytest fixtures, arguments, and options."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import niwrap
import pytest
import yaml
from styxpodman import PodmanRunner

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.niwrap import resolve_runner

if TYPE_CHECKING:
    from collections.abc import Sequence


def pytest_collection_modifyitems(items: Sequence[pytest.Item]) -> None:
    """Apply appropriate markers based on test location."""
    markers = {"unit", "integration", "regression"}

    for item in items:
        test_path = Path(item.fspath)
        for marker in markers & set(test_path.parts):
            item.add_marker(getattr(pytest.mark, marker))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add option(s) to pytest parser."""
    parser.addoption(
        "--runner",
        action="store",
        default="local",
        help="Styx runner type to use: ['local', 'docker', 'singularity']",
    )
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help="Directory where test data is located.",
    )


@pytest.fixture(scope="session", autouse=True)
def niwrap_runner(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> niwrap.Runner:
    """Globally set niwrap runner for the testing suite."""
    # Set up niwrap runner
    runner_type, runner_exec = resolve_runner(
        request.config.getoption("--runner").lower()
    )
    match runner_type:
        case "docker":
            niwrap.use_docker(docker_executable=runner_exec)
        case "podman":
            niwrap.set_global_runner(
                # UserID = 0 currently necessary for some containers used
                runner=PodmanRunner(podman_executable=runner_exec, podman_user_id=0)
            )
        case "singularity":
            niwrap.use_singularity(singularity_executable=runner_exec)
        case _:
            niwrap.use_local()
    runner = niwrap.get_global_runner()
    runner.data_dir = tmp_path_factory.mktemp(f"{os.urandom(8).hex()}")
    # Set up logging for debugging
    logger = logging.getLogger(runner.logger_name)
    logger.setLevel(logging.DEBUG)
    return runner


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
def graph(tmp_path: Path, request: pytest.FixtureRequest) -> NeuromapsGraph:
    """Create a graph fixture to use for tests."""
    data_dir = request.config.getoption("--data-dir")
    if data_dir is not None:
        return NeuromapsGraph(runner="auto", data_dir=Path(data_dir).resolve())

    graph = NeuromapsGraph(_testing=True)
    data = yaml.safe_load(graph.yaml_path.read_text())

    for node_block in data.get("nodes", {}):
        for node_name, node in node_block.items():
            _rewrite_node_files(tmp_path, node_name, node)

    for edge_list in data.get("edges", {}).values():
        for edge in edge_list:
            _rewrite_edge_files(tmp_path, edge)

    graph._builder.build_from_dict(graph=graph, data=data)
    return graph
