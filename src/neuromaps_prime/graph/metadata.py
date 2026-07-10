"""Metadata helpers for transformation pipelines.

Provides utilities to format structured references into human-readable
strings.  Merging of references/notes during transform composition is
done inline to keep the surface/volume ops modules self-contained.
"""

from __future__ import annotations


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
