r"""Validate URLs from YAML files using HEAD requests (no download).

Every URL must be unique across the scanned files — a URL referenced by
more than one resource aborts the run before any request is made.

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
#     "aiohttp>=3.14.1",
#     "pyyaml>=6.0.2",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import datetime
import email.utils
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import yaml

_HEADERS: dict[str, str] = {"User-Agent": "url-validator/1.0", "Connection": "close"}
_TIMEOUT = 90
_MAX_RETRIES = 5
_BACKOFF_BASE = 10.0
_BACKOFF_MAX = 600.0
_GET_RANGE_TRIGGER = frozenset({403, 405})
_GET_RANGE_TRIGGER_DETAILS: frozenset[str] = frozenset(
    f"HTTP {code}" for code in _GET_RANGE_TRIGGER
)

log = logging.getLogger("url-validator")


@dataclass(frozen=True, slots=True)
class UrlOccurrence:
    """A single occurrence of a URL inside one YAML file.

    Attributes:
        source: Path of the YAML file containing the URL.
        key_path: Dot-separated path of YAML keys from the file root to the
            URL (e.g. ``Yerkes19.surfaces.43k.sphere.left``).
    """

    source: Path
    key_path: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a single URL probe.

    Attributes:
        url: The URL that was checked.
        ok: True if the server responded with a non-error status.
        detail: Human-readable status string (e.g. ``"200"`` or ``"HTTP 404"``).
        sources: Occurrences of the URL in the YAML files (file + key path).
    """

    url: str
    ok: bool
    detail: str
    sources: list[UrlOccurrence] = field(default_factory=list)


@dataclass(slots=True)
class UrlIndex:
    """Maps each unique URL to every occurrence of it in the YAML files.

    Attributes:
        _data: Internal mapping of URL → list of occurrences.
    """

    _data: dict[str, list[UrlOccurrence]] = field(default_factory=dict)

    def add(self, url: str, source: Path, key_path: str) -> None:
        """Register an occurrence of *url* in *source* at *key_path*.

        Args:
            url: The URL string to index.
            source: Path of the YAML file containing the URL.
            key_path: Dot-separated key path from the YAML root to the URL.
        """
        self._data.setdefault(url, []).append(
            UrlOccurrence(source=source, key_path=key_path)
        )

    def urls(self) -> list[str]:
        """Return all unique URLs in insertion order."""
        return list(self._data)

    def sources(self, url: str) -> list[UrlOccurrence]:
        """Return every occurrence of *url* (file and key path).

        Args:
            url: A URL previously registered via :meth:`add`.
        """
        return self._data[url]

    def duplicate_urls(self) -> dict[str, list[UrlOccurrence]]:
        """Return URLs that appear more than once, with all their occurrences.

        A URL can be duplicated across files, or listed multiple times
        within a single file — both show up as more than one occurrence.

        Returns:
            Mapping of duplicated URL to its occurrences, in index
            insertion order.
        """
        return {url: srcs for url, srcs in self._data.items() if len(srcs) > 1}

    def __len__(self) -> int:
        """Return count of URLs."""
        return len(self._data)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header value into seconds.

    Supports both delay-in-seconds and HTTP-date formats. Returns ``None``
    when the value is missing or unparseable.

    Args:
        value: Raw ``Retry-After`` header value.

    Returns:
        Seconds to wait, or ``None``.
    """
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        dt = email.utils.parsedate_to_datetime(value)
        return max(0.0, (dt - datetime.datetime.now(datetime.UTC)).total_seconds())
    except Exception:
        return None


def _make_headers(*, range_header: bool = False) -> dict[str, str]:
    """Build request headers with shared defaults.

    Args:
        range_header: If True, add ``Range: bytes=0-0`` to avoid body download.

    Returns:
        Header dictionary suitable for :mod:`aiohttp`.
    """
    headers = dict(_HEADERS)
    if range_header:
        headers["Range"] = "bytes=0-0"
    return headers


async def _probe(
    session: aiohttp.ClientSession,
    *,
    url: str,
    method: str,
    sources: list[UrlOccurrence],
    range_header: bool = False,
    fallback: bool = False,
) -> CheckResult:
    """Send one async request, backing off on HTTP 429 rate limits.

    Args:
        session: Shared :class:`aiohttp.ClientSession`.
        url: Target URL.
        method: HTTP method (``"HEAD"`` or ``"GET"``).
        sources: Occurrences of the URL in the YAML files.
        range_header: If True, request only the first byte.
        fallback: If True, annotate a successful status as a GET fallback.

    Returns:
        :class:`CheckResult` with ``ok=True`` on any non-error response.
    """
    headers = _make_headers(range_header=range_header)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT),
                allow_redirects=True,
            ) as response:
                if response.status == 429:
                    retry_after = _parse_retry_after(
                        response.headers.get("Retry-After")
                    )
                    if retry_after is not None:
                        delay = retry_after
                    else:
                        delay = min(_BACKOFF_BASE**attempt, _BACKOFF_MAX)
                    # Apply ±50 % jitter
                    delay *= random.uniform(0.5, 1.5)  # noqa: S311
                    log.warning(
                        "Rate limited on %s, backing off %.1fs (%d/%d)",
                        url,
                        delay,
                        attempt,
                        _MAX_RETRIES,
                    )
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(delay)
                        continue
                    return CheckResult(
                        url,
                        ok=False,
                        detail=f"HTTP 429 after {_MAX_RETRIES} attempts",
                        sources=sources,
                    )

                if response.status < 400:
                    detail = f"{response.status}"
                    if fallback:
                        detail = f"{detail} (GET fallback)"
                    return CheckResult(url, ok=True, detail=detail, sources=sources)

                return CheckResult(
                    url, ok=False, detail=f"HTTP {response.status}", sources=sources
                )

        except TimeoutError:
            return CheckResult(url, ok=False, detail="timeout", sources=sources)
        except aiohttp.ClientError as exc:
            return CheckResult(url, ok=False, detail=str(exc), sources=sources)
        except Exception as exc:
            return CheckResult(url, ok=False, detail=str(exc), sources=sources)

    return CheckResult(url, ok=False, detail="HTTP 429", sources=sources)


async def _get_range(
    session: aiohttp.ClientSession, url: str, sources: list[UrlOccurrence]
) -> CheckResult:
    """Fallback probe: GET with ``Range: bytes=0-0`` to avoid body download.

    Fetches at most one byte. Retries on HTTP 429 like the primary probe.

    Args:
        session: Shared :class:`aiohttp.ClientSession`.
        url: Target URL.
        sources: Occurrences of the URL in the YAML files.

    Returns:
        :class:`CheckResult` with ``ok=True`` on any non-error response.
    """
    return await _probe(
        session,
        url=url,
        method="GET",
        sources=sources,
        range_header=True,
        fallback=True,
    )


def _is_github_blob_url(url: str) -> bool:
    """Return True if *url* is a GitHub ``/blob/`` file-page URL.

    These are human-facing pages (e.g. ``github.com/owner/repo/blob/main/file``)
    that render an HTML view rather than exposing the raw file, so they cannot
    be consumed programmatically. Query strings and fragments are ignored.

    Args:
        url: A candidate URL string.

    Returns:
        True if the URL points at a ``github.com`` blob page, else False.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().removeprefix("www.")
    if host != "github.com":
        return False
    segments = [s for s in parts.path.split("/") if s]
    return len(segments) >= 3 and segments[2] == "blob"


async def check_url(
    session: aiohttp.ClientSession, url: str, sources: list[UrlOccurrence]
) -> CheckResult:
    """Validate *url* without downloading its body.

    Strategy:
        1. Reject unsupported URL types (GitHub ``/blob/`` pages) up front.
        2. Send a HEAD request, retrying on HTTP 429 with backoff.
        3. On 403/405 (server rejects HEAD), fall back to :func:`_get_range`.
        4. Map any other HTTP/network error to a failed :class:`CheckResult`.

    Args:
        session: Shared :class:`aiohttp.ClientSession`.
        url: The URL to validate.
        sources: Occurrences of the URL in the YAML files.

    Returns:
        :class:`CheckResult` describing whether the URL is reachable.
    """
    if _is_github_blob_url(url):
        return CheckResult(
            url,
            ok=False,
            detail="unsupported URL type: GitHub blob URL (use a raw file URL)",
            sources=sources,
        )

    result = await _probe(session, url=url, method="HEAD", sources=sources)
    if not result.ok and result.detail in _GET_RANGE_TRIGGER_DETAILS:
        return await _get_range(session, url, sources)
    return result


def _list_label(item: Any, index: int) -> str:  # noqa: ANN401
    """Return the key-path label for list item *index*.

    Edge entries carry ``from``/``to`` — use them so paths read as
    ``CIVETNMT→D99.surfaces...`` rather than a bare ``[i]``; anything
    else falls back to the index.

    Args:
        item: The list element.
        index: Position of the element in its list.

    Returns:
        One segment of a dot-separated key path.
    """
    if isinstance(item, dict) and "from" in item and "to" in item:
        return f"{item['from']}→{item['to']}"
    return f"[{index}]"


def _extract_urls(node: Any, out: list[tuple[str, str]], path: str = "") -> None:  # noqa: ANN401
    """Recursively extract HTTP/HTTPS strings from an arbitrary YAML node.

    Uses DFS with a caller-supplied *out* buffer to avoid per-call
    allocations. Skip keys named "references" or "notes". Each discovered
    URL is paired with its dot-separated key path from the YAML root.

    Args:
        node: A YAML value — may be a str, dict, list, or scalar.
        out: Mutable list of (url, key_path) tuples to append into.
        path: Key path accumulated above *node* (empty at the root).
    """
    if isinstance(node, str):
        if node.startswith(("http://", "https://")):
            out.append((node, path))
    elif isinstance(node, dict):
        for k, v in node.items():
            if k in ("references", "notes"):
                continue
            _extract_urls(v, out, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            label = _list_label(item, i)
            _extract_urls(item, out, f"{path}.{label}" if path else label)


def build_index(yaml_files: list[Path]) -> UrlIndex:
    """Parse *yaml_files* and return a :class:`UrlIndex` of all URLs found.

    Files that cannot be read or parsed are logged as warnings and skipped.

    Args:
        yaml_files: Paths to YAML files to scan.

    Returns:
        :class:`UrlIndex` mapping each unique URL to its source file(s).
    """
    index = UrlIndex()
    raw: list[tuple[str, str]] = []
    for path in yaml_files:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            raw.clear()
            _extract_urls(data, raw)
            for url, key_path in raw:
                index.add(url, path, key_path)
        except Exception as exc:
            log.warning("Could not parse %s: %s", path, exc)
    return index


class DuplicateURLsError(ValueError):
    """Raised when a URL is referenced by more than one resource.

    Attributes:
        duplicates: Mapping of duplicated URL to its occurrences.
    """

    duplicates: dict[str, list[UrlOccurrence]]

    def __init__(self, duplicates: dict[str, list[UrlOccurrence]]) -> None:
        """Build the error message from *duplicates*.

        Args:
            duplicates: Mapping of duplicated URL to its occurrences.
        """
        self.duplicates = duplicates
        lines = [f"{len(duplicates)} URL(s) referenced more than once:"]
        for url, occurrences in duplicates.items():
            lines.append(f"  {url}")
            lines.extend(f"      {occ.source} [{occ.key_path}]" for occ in occurrences)
        super().__init__("\n".join(lines))


def ensure_unique_urls(index: UrlIndex) -> None:
    """Ensure every URL in *index* is referenced exactly once.

    Args:
        index: URL index produced by :func:`build_index`.

    Raises:
        DuplicateURLsError: If any URL occurs more than once.
    """
    duplicates = index.duplicate_urls()
    if duplicates:
        raise DuplicateURLsError(duplicates)


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


async def _bounded_check(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    sources: list[UrlOccurrence],
) -> CheckResult:
    """Probe *url* while respecting the concurrency semaphore."""
    async with semaphore:
        return await check_url(session, url, sources)


async def _cancel_pending(pending: set[asyncio.Task[CheckResult]]) -> None:
    """Cancel any tasks still in flight and await their cleanup."""
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _format_sources(sources: list[UrlOccurrence]) -> str:
    """Render occurrences as ``file [key.path]`` entries, or 'unknown'."""
    if not sources:
        return "unknown"
    return ", ".join(f"{occ.source} [{occ.key_path}]" for occ in sources)


async def _report_result(result: CheckResult, failed: list[CheckResult]) -> None:
    """Log a single result and append to *failed* if it did not succeed."""
    srcs = _format_sources(result.sources)
    if result.ok:
        log.info("OK    %s  %s  (%s)", result.detail.rjust(20), result.url, srcs)
    else:
        log.error("FAIL  %s  %s  (%s)", result.detail.rjust(20), result.url, srcs)
        failed.append(result)


async def run(index: UrlIndex, *, workers: int, fail_fast: bool) -> list[CheckResult]:
    """Probe all URLs in *index* concurrently and return failed results.

    Uses ``asyncio`` with a bounded semaphore for concurrency. HTTP 429
    responses are retried with exponential backoff (honouring ``Retry-After``
    when provided). Pending work is cancelled on ``fail_fast`` or
    ``KeyboardInterrupt``.

    Args:
        index: URL index produced by :func:`build_index`.
        workers: Maximum number of in-flight requests.
        fail_fast: If True, stop after the first failure.

    Returns:
        List of :class:`CheckResult` where ``ok`` is False.
    """
    failed: list[CheckResult] = []
    semaphore = asyncio.Semaphore(workers)
    connector = aiohttp.TCPConnector(limit=workers)

    async with aiohttp.ClientSession(connector=connector, headers=_HEADERS) as session:
        pending: set[asyncio.Task[CheckResult]] = {
            asyncio.create_task(
                _bounded_check(session, semaphore, url, index.sources(url)), name=url
            )
            for url in index.urls()
        }

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    await _report_result(task.result(), failed)
                    if fail_fast and failed:
                        await _cancel_pending(pending)
                        return failed
        except KeyboardInterrupt:
            await _cancel_pending(pending)
            sys.exit(130)

    return failed


def _configure_logging(*, verbose: bool) -> None:
    """Set up root logger with a compact format suitable for CI output.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(levelname)-5s %(message)s", stream=sys.stdout
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

    _configure_logging(verbose=args.verbose)

    yaml_files = discover_yamls(args.files)
    index = build_index(yaml_files)

    if not index:
        log.info("No URLs found.")
        sys.exit(0)

    try:
        ensure_unique_urls(index)
    except DuplicateURLsError as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info("Checking %d unique URLs with %d workers...\n", len(index), args.workers)

    failed = asyncio.run(run(index, workers=args.workers, fail_fast=args.fail_fast))

    log.info("=" * 60)
    log.info("Results: %d/%d OK", len(index) - len(failed), len(index))

    if failed:
        log.error("Failed URLs:")
        for r in failed:
            srcs = _format_sources(r.sources)
            log.error("  %s\n    reason : %s\n    sources: %s", r.url, r.detail, srcs)
        sys.exit(1)


if __name__ == "__main__":
    main()
