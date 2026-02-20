"""Global pytest fixtures, arguments, and options."""

from pathlib import Path
from typing import Any, Generator

import pytest
import yaml
from niwrap import Runner, ants, workbench

from neuromaps_prime.graph import NeuromapsGraph
from neuromaps_prime.utils import set_runner


def pytest_addoption(parser: pytest.Parser):
    """Add option(s) to pytest parser."""
    parser.addoption(
        "--runner",
        action="store",
        default="local",
        help="Styx runner type to use: ['local', 'docker', 'singularity']",
    )
    parser.addoption(
        "--runner-images",
        action="store",
        default=None,
        help="Optional dict string of image overrides to use for StyxRunner.",
    )
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help="Directory where test data is located.",
    )


@pytest.fixture(scope="session", autouse=True)
def runner(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Generator[Runner, None, None]:
    """Globally set runner for the testing suite."""
    tmp_dir = tmp_path_factory.mktemp("styx_tmp")
    pytest_runner = set_runner(
        runner=request.config.getoption("--runner").lower(),
        image_overrides=request.config.getoption("--runner-images"),
        data_dir=tmp_dir,
    )
    yield pytest_runner


@pytest.fixture
def graph(
    runner: Runner, tmp_path: Path, request: pytest.FixtureRequest
) -> NeuromapsGraph:
    """Create a graph fixture to use for tests."""
    data_dir = request.config.getoption("--data-dir")
    if data_dir is not None:
        return NeuromapsGraph(runner=runner, data_dir=Path(data_dir).resolve())

    graph = NeuromapsGraph(_testing=True)
    data = yaml.safe_load(graph.yaml_path.read_text())

    def mk(path: Path) -> str:
        path.touch()
        return str(path)

    def rewrite_node_files(name: str, node: dict[str, Any]) -> None:
        # Surfaces
        surfaces = node.get("surfaces", {})
        for density, types in surfaces.items():
            for surf_type, hemis in types.items():
                for hemi in list(hemis):
                    hemis[hemi] = mk(
                        tmp_path / f"{name}_{density}_{hemi}_{surf_type}.surf.gii"
                    )
        # Volumes
        volumes = node.get("volumes", {})
        for res, types in volumes.items():
            for vol_type in list(types):
                types[vol_type] = mk(tmp_path / f"{name}_{res}_{vol_type}.nii.gz")

    def rewrite_edge_files(edge: dict[str, Any]) -> None:
        src = edge["from"]
        dst = edge["to"]

        # Surfaces
        surfaces = edge.get("surfaces", {})
        for density, types in surfaces.items():
            for surf_type, hemis in types.items():
                for hemi in list(hemis):
                    hemis[hemi] = mk(
                        tmp_path
                        / f"{src}_to_{dst}_{density}_{hemi}_{surf_type}.surf.gii"
                    )
        # Volumes
        volumes = edge.get("volumes", {})
        for res, types in volumes.items():
            for vol_type in list(types):
                types[vol_type] = mk(
                    tmp_path / f"{src}_to_{dst}_{res}_{vol_type}.nii.gz"
                )

    # Rewrite node file paths
    for node_block in data.get("nodes", {}):
        for node_name, node in node_block.items():
            rewrite_node_files(node_name, node)

    # Rewrite transform file paths
    for edge_list in data.get("edges", {}).values():
        for edge in edge_list:
            rewrite_edge_files(edge)

    graph._build_from_dict(data)
    return graph


@pytest.fixture
def require_ants(runner: Runner) -> None:
    try:
        ants.ants_apply_transforms(
            reference_image=".",
            output=ants.ants_apply_transforms_warped_output("."),
            runner=runner,
        )
    except FileNotFoundError:
        pytest.skip("ANTs not available in environment")
    except Exception:
        # Failures for other reasons are ignored
        pass


@pytest.fixture
def require_workbench(runner: Runner) -> None:
    try:
        workbench.nifti_information(nifti_file=".", runner=runner)
    except FileNotFoundError:
        pytest.skip("wb_command not available in environment")
    except Exception:
        pass