"""Test shared resource URLs with a mocked standalone HTTP transport."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).parents[2]
BASE = "https://raw.githubusercontent.com/zhouzx-UCAS/RheMAP-Surf-resources/main/"
SHARED = tuple(
    f"{BASE}surfaces/{space}/src-{space}_den-{density}_hemi-{hemi}_sphere.surf.gii"
    for space, density in (("F99", "74k"), ("Yerkes19", "32k"))
    for hemi in ("L", "R")
)


@pytest.fixture
def validator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the script without installing its optional HTTP dependencies."""
    # The script installs aiohttp through its inline uv dependencies. Unit tests
    # exercise indexing and dispatch with a fake transport, without network IO.
    transport = ModuleType("aiohttp")
    monkeypatch.setitem(sys.modules, "aiohttp", transport)
    spec = importlib.util.spec_from_file_location(
        "_test_neuromaps_url_validator", ROOT / "scripts/validate_urls.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _yaml_file(tmp_path: Path, urls: tuple[str, ...]) -> Path:
    path = tmp_path / "resources.yaml"
    path.write_text("resources:\n" + "".join(f"  - {url}\n" for url in urls))
    return path


def test_duplicate_fails_before_network(
    validator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unapproved duplicate fails even when URL filtering would skip it."""
    path = _yaml_file(tmp_path, (SHARED[0], SHARED[0]))
    run = AsyncMock(return_value=([], []))
    monkeypatch.setattr(validator, "run", run)
    monkeypatch.setattr(sys, "argv", ["validator", str(path), "--only", "other"])
    with pytest.raises(SystemExit) as exc:
        validator.main()
    assert exc.value.code == 1
    run.assert_not_called()


def test_all_four_exact_shared_urls_are_allowed(
    validator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Repeated exact allowances retain every occurrence and dispatch each URL once."""
    path = _yaml_file(tmp_path, SHARED + SHARED)
    run = AsyncMock(return_value=([], []))
    monkeypatch.setattr(validator, "run", run)
    args = ["validator", str(path)]
    for url in SHARED:
        args.extend(("--allow-shared-url", url))
    monkeypatch.setattr(sys, "argv", args)
    validator.main()
    run.assert_awaited_once()
    index = run.call_args.args[0]
    assert len(index) == 4
    assert all(len(index.sources(url)) == 2 for url in SHARED)


@pytest.mark.parametrize("allowance", [BASE, SHARED[0] + "?download=1"])
def test_allowance_is_not_a_prefix_or_near_match(
    validator: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    allowance: str,
) -> None:
    """An allowance must match the complete URL exactly."""
    path = _yaml_file(tmp_path, (SHARED[0], SHARED[0]))
    monkeypatch.setattr(
        sys, "argv", ["validator", str(path), "--allow-shared-url", allowance]
    )
    with pytest.raises(SystemExit) as exc:
        validator.main()
    assert exc.value.code == 1


def test_other_duplicates_remain_errors(
    validator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Allowing a legacy sphere does not hide an unrelated repeated resource."""
    other = BASE + "unexpected.shape.gii"
    path = _yaml_file(tmp_path, (SHARED[0], SHARED[0], other, other))
    monkeypatch.setattr(
        sys, "argv", ["validator", str(path), "--allow-shared-url", SHARED[0]]
    )
    with pytest.raises(SystemExit) as exc:
        validator.main()
    assert exc.value.code == 1


def test_shared_url_still_fails_when_unreachable(
    validator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The duplicate allowance never exempts an unreachable URL from failure."""
    path = _yaml_file(tmp_path, (SHARED[0], SHARED[0]))
    failed = validator.CheckResult(SHARED[0], ok=False, detail="HTTP 404", sources=[])
    run = AsyncMock(return_value=([failed], []))
    monkeypatch.setattr(validator, "run", run)
    monkeypatch.setattr(
        sys, "argv", ["validator", str(path), "--allow-shared-url", SHARED[0]]
    )
    with pytest.raises(SystemExit) as exc:
        validator.main()
    assert exc.value.code == 1
    run.assert_awaited_once()


def test_probe_once_and_honor_proxy_configuration(
    validator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dispatch one probe per URL with all occurrences and environment proxies."""
    path = _yaml_file(tmp_path, (SHARED[0], SHARED[0]))
    index = validator.build_index([path])
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(validator.aiohttp, "ClientSession", factory, raising=False)
    monkeypatch.setattr(validator.aiohttp, "TCPConnector", MagicMock(), raising=False)
    check = AsyncMock(
        return_value=validator.CheckResult(
            SHARED[0], ok=True, detail="200", sources=index.sources(SHARED[0])
        )
    )
    monkeypatch.setattr(validator, "check_url", check)
    failed, skipped = asyncio.run(validator.run(index, workers=2, fail_fast=False))
    assert failed == []
    assert skipped == []
    check.assert_awaited_once_with(session, SHARED[0], index.sources(SHARED[0]))
    assert factory.call_args.kwargs["trust_env"] is True


def test_released_registry_has_only_four_intentional_duplicates(
    validator: ModuleType,
) -> None:
    """The real macaque registry contains exactly the four shared legacy spheres."""
    resources = ROOT / "src/neuromaps_prime/resources"
    paths = [
        *sorted((resources / "nodes/macaque").glob("*.yaml")),
        resources / "edges/surface/macaque_macaque.yaml",
    ]
    index = validator.build_index(paths)
    assert set(index.duplicate_urls()) == set(SHARED)
    assert all(len(index.sources(url)) == 2 for url in SHARED)
