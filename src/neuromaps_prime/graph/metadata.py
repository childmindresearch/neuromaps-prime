"""Metadata helpers for transformation pipelines.

Provides utilities to format structured references into human-readable
strings and print grouped summaries.  Merging of references/notes during
transform composition is done inline to keep the surface/volume ops
modules self-contained.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from neuromaps_prime.graph.models import (
        HopMetadataDict,
        SpaceMetadataDict,
        TransformResult,
    )


def format_reference(ref: str | dict[str, str]) -> str:
    """Combine a structured reference into a single human-readable string.

    String references are returned as-is.  Dict references are joined with
    `` | `` using the ``citation``, ``doi``, and ``url`` keys when present.

    Args:
        ref: Either a plain citation string or a structured dict.

    Returns:
        A single formatted reference string.
    """
    if isinstance(ref, str):
        return ref

    parts = []
    if citation := ref.get("citation"):
        parts.append(citation)
    if doi := ref.get("doi"):
        parts.append(f"DOI: {doi}")
    if url := ref.get("url"):
        parts.append(f"URL: {url}")
    return " | ".join(parts)


def _build_space_path(hops: Sequence[HopMetadataDict]) -> list[str]:
    """Extract ordered space sequence from hop metadata.

    Args:
        hops: Per-hop metadata dicts.

    Returns:
        Ordered list of unique space names forming the transform path.
    """
    space_names: list[str] = []
    for hop in hops:
        src = hop.get("source_space", "")
        if src and (not space_names or space_names[-1] != src):
            space_names.append(src)
        tgt = hop.get("target_space", "")
        if tgt and tgt not in space_names:
            space_names.append(tgt)
    return space_names


def _print_spaces(
    spaces: Sequence[SpaceMetadataDict],
    out: Callable[..., None],
) -> None:
    """Print node-level space references.

    Args:
        spaces: Per-space metadata dicts.
        out: print function configured with the target file.
    """
    out("--- Spaces ---")
    for space in spaces:
        name = space.get("space", "Unknown")
        refs = space.get("references") or []
        out(f"  {name}:")
        for ref in refs:
            out(f"    Ref: {ref}")
    out()


def _print_hop(
    hop: HopMetadataDict,
    out: Callable[..., None],
) -> None:
    """Print a single hop's references and caveats.

    Args:
        hop: Per-hop metadata dict.
        out: print function configured with the target file.
    """
    src = hop.get("source_space", "")
    tgt = hop.get("target_space", "")
    provider = hop.get("provider", "")
    out(f"--- {src} -> {tgt} [{provider}] ---")

    refs = hop.get("references") or []
    if refs:
        out("  References:")
        for ref in refs:
            out(f"    - {ref}")

    notes = hop.get("notes") or []
    if notes:
        out("  Caveats:")
        for note in notes:
            out(f"    - {note}")
    out()


def print_metadata_summary(
    result: TransformResult,
    file: TextIO | None = None,
) -> None:
    """Print a grouped metadata summary to stdout.

    Prints node-level references under a 'Spaces' header, then per-hop
    transform references and caveats. Does nothing if the result has no
    metadata or is falsy.

    Args:
        result: TransformResult carrying structured provenance metadata.
        file: File-like object to write to. Defaults to stdout.
    """
    if not result or result.metadata is None:
        return

    out = partial(print, file=file)
    hops = result.metadata.transforms or []
    out(f"=== Transformation: {' -> '.join(_build_space_path(hops))} ===")
    out()

    spaces = result.metadata.spaces or []
    if spaces:
        _print_spaces(spaces, out)

    for hop in hops:
        _print_hop(hop, out)

    out("=== End ===")
