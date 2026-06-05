"""Validate URLs from YAML files using HEAD requests (no download).

Usage::
    uv run scripts/validate_urls.py \\
        input_files ... \\
        [--workers 4] [--fail-fast] \\
        [--verbose]

Or standalone:
    uv run --script validate_urls.py input_files ...
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyyaml>=6.0.2"
# ]
# ///

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

_HEADERS: dict[str, str] = {
    "User-Agent": "url-validator/1.0",
    "Connection": "close",
}
_TIMEOUT: int = 10
_GET_RANGE_TRIGGER: frozenset[int] = frozenset({403, 405})

log = logging.getLogger("url-validator")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a single URL probe.

    Attributes:
        url: The URL that was checked.
        ok: True if the server responded with a non-error status.
        detail: Human-readable status string (e.g. ``"200"`` or ``"HTTP 404"``).
    """

    url: str
    ok: bool
    detail: str


@dataclass(slots=True)
class UrlIndex:
    """Maps each unique URL to the YAML files it appears in.

    Attributes:
        _data: Internal mapping of URL → list of source paths.
    """

    _data: dict[str, list[Path]] = field(default_factory=dict)

    def add(self, url: str, source: Path) -> None:
        """Register *url* as originating from *source*.

        Args:
            url: The URL string to index.
            source: Path of the YAML file containing the URL.
        """
        self._data.setdefault(url, []).append(source)

    def urls(self) -> list[str]:
        """Return all unique URLs in insertion order."""
        return list(self._data)

    def sources(self, url: str) -> list[Path]:
        """Return every YAML file that contains *url*.

        Args:
            url: A URL previously registered via :meth:`add`.
        """
        return self._data[url]

    def __len__(self) -> int:
        return len(self._data)


def _make_request(url: str, method: str, range_header: bool = False) -> Request:
    """Build an :class:`urllib.request.Request` with shared headers.

    Args:
        url: Target URL.
        method: HTTP method (``"HEAD"`` or ``"GET"``).
        range_header: If True, add ``Range: bytes=0-0`` to avoid body download.

    Returns:
        A configured :class:`~urllib.request.Request` instance.
    """
    req = Request(url, method=method, headers=_HEADERS)
    if range_header:
        req.add_header("Range", "bytes=0-0")
    return req


def _get_range(url: str) -> CheckResult:
    """Fallback probe: GET with ``Range: bytes=0-0`` to avoid body download.

    Used when the server rejects HEAD (403/405). Fetches at most one byte.

    Args:
        url: Target URL.

    Returns:
        :class:`CheckResult` with ``ok=True`` on any non-error response.
    """
    try:
        with urlopen(
            _make_request(url, "GET", range_header=True), timeout=_TIMEOUT
        ) as r:
            return CheckResult(url, True, f"{r.status} (GET fallback)")
    except Exception as exc:
        return CheckResult(url, False, str(exc))


def check_url(url: str) -> CheckResult:
    """Probe *url* without downloading its body.

    Strategy:
        1. Send a HEAD request.
        2. On 403/405 (server rejects HEAD), fall back to :func:`_get_range`.
        3. Map any other HTTP/network error to a failed :class:`CheckResult`.

    Args:
        url: The URL to validate.

    Returns:
        :class:`CheckResult` describing whether the URL is reachable.
    """
    try:
        with urlopen(_make_request(url, "HEAD"), timeout=_TIMEOUT) as r:
            if r.status < 400:
                return CheckResult(url, True, str(r.status))
            return _get_range(url)
    except HTTPError as exc:
        if exc.code in _GET_RANGE_TRIGGER:
            return _get_range(url)
        return CheckResult(url, False, f"HTTP {exc.code}")
    except URLError as exc:
        return CheckResult(url, False, str(exc.reason))
    except Exception as exc:
        return CheckResult(url, False, str(exc))


def _extract_urls(node: Any, out: list[str]) -> None:
    """Recursively extract HTTP/HTTPS strings from an arbitrary YAML node.

    Uses DFS with a caller-supplied *out* buffer to avoid per-call allocations.

    Args:
        node: A YAML value — may be a str, dict, list, or scalar.
        out: Mutable list to append discovered URLs into.
    """
    if isinstance(node, str):
        if node.startswith(("http://", "https://")):
            out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            _extract_urls(v, out)
    elif isinstance(node, list):
        for item in node:
            _extract_urls(item, out)


def build_index(yaml_files: list[Path]) -> UrlIndex:
    """Parse *yaml_files* and return a :class:`UrlIndex` of all URLs found.

    Files that cannot be read or parsed are logged as warnings and skipped.

    Args:
        yaml_files: Paths to YAML files to scan.

    Returns:
        :class:`UrlIndex` mapping each unique URL to its source file(s).
    """
    index = UrlIndex()
    raw: list[str] = []
    for path in yaml_files:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            raw.clear()
            _extract_urls(data, raw)
            for url in raw:
                index.add(url, path)
        except Exception as exc:
            log.warning("Could not parse %s: %s", path, exc)
    return index


def discover_yamls(roots: list[str]) -> list[Path]:
    """Resolve the set of YAML files to scan.

    If *roots* is non-empty, those paths are used directly. Otherwise every
    ``*.yaml`` and ``*.yml`` file under the current working directory is
    returned (recursive).

    Args:
        roots: Explicit file paths supplied on the command line.

    Returns:
        List of :class:`~pathlib.Path` objects for each YAML file.
    """
    if roots:
        return [Path(f) for f in roots]
    base = Path()
    return [*base.rglob("*.yaml"), *base.rglob("*.yml")]


def run(index: UrlIndex, workers: int, fail_fast: bool) -> list[CheckResult]:
    """Probe all URLs in *index* concurrently and return failed results.

    Each result is logged at INFO (ok) or ERROR (fail) as it arrives.
    On ``KeyboardInterrupt`` in-flight futures are cancelled and the process
    exits with code 130.

    Args:
        index: URL index produced by :func:`build_index`.
        workers: Maximum number of threads in the pool.
        fail_fast: If True, stop after the first failure.

    Returns:
        List of :class:`CheckResult` where ``ok`` is False.
    """
    failed: list[CheckResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_url, url): url for url in index.urls()}
        try:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                srcs = ", ".join(str(p) for p in index.sources(result.url))
                if result.ok:
                    log.info(
                        "OK    %s  %s  (%s)", result.detail.rjust(20), result.url, srcs
                    )
                else:
                    log.error(
                        "FAIL  %s  %s  (%s)", result.detail.rjust(20), result.url, srcs
                    )
                    failed.append(result)
                    if fail_fast:
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            sys.exit(130)

    return failed


def _configure_logging(verbose: bool) -> None:
    """Set up root logger with a compact format suitable for CI output.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-5s %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    """CLI entry point — parse arguments, run checks, exit non-zero on failure."""
    parser = argparse.ArgumentParser(
        description="Validate URLs in YAML files via HEAD requests"
    )
    parser.add_argument(
        "files", nargs="*", help="YAML files (default: **/*.yaml **/*.yml)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel workers (default: 4)"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Exit 1 on first failure"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logging"
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    yaml_files = discover_yamls(args.files)
    index = build_index(yaml_files)

    if not index:
        log.info("No URLs found.")
        sys.exit(0)

    log.info("Checking %d unique URLs with %d workers...\n", len(index), args.workers)

    failed = run(index, workers=args.workers, fail_fast=args.fail_fast)

    log.info("=" * 60)
    log.info("Results: %d/%d OK", len(index) - len(failed), len(index))

    if failed:
        log.error("Failed URLs:")
        for r in failed:
            srcs = ", ".join(str(p) for p in index.sources(r.url))
            log.error("  %s\n    reason : %s\n    sources: %s", r.url, r.detail, srcs)
        sys.exit(1)


if __name__ == "__main__":
    main()
