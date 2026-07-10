"""Models for resources in the neuromaps_prime graph."""

import datetime
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from neuromaps_prime.fetcher import download_and_validate

_logger = logging.getLogger(__name__)


class _SerializedEntry(TypedDict, total=False):
    """TypedDict for serialized metadata entry (space or transform hop)."""

    space: str
    source_space: str
    target_space: str
    provider: str
    references: list[str]
    notes: list[str]


class Resource(BaseModel):
    """Base model for resources in the neuromaps_prime graph."""

    name: str
    description: str | None
    file_path: Path
    uri: str | None = None
    references: Sequence[str | dict[str, str]] | None = None
    notes: Sequence[str] | None = None

    def fetch(self) -> Path:
        """Return the path to this resource's file, downloading if necessary.

        Returns:
            Path to the resource file.

        Raises:
            FileNotFoundError: if file cannot be fetched
        """
        if self.file_path.exists():
            return self.file_path
        if self.uri is None:
            raise FileNotFoundError("File does not exist and cannot be fetched.")
        if (local_file := Path(self.uri)).exists():
            self.file_path = local_file
        else:
            _logger.info(f"Fetching {self.file_path.name} from remote server.")
            download_and_validate(uri=self.uri, dest=self.file_path)
            if not self.file_path.exists():
                raise FileNotFoundError("File does not exist.")
        return self.file_path

    def __repr__(self) -> str:
        """Custom string representation for debugging."""
        return self.name  # pragma: nocover


class SurfaceAtlas(Resource):
    """Model for surface atlas resources."""

    space: str
    density: str
    hemisphere: Literal["left", "right"]
    resource_type: str


class SurfaceTransform(Resource):
    """Model for surface transform resources."""

    source_space: str
    target_space: str
    density: str
    hemisphere: Literal["left", "right"]
    resource_type: str
    provider: str
    weight: float = 1.0


class SurfaceAnnotation(Resource):
    """Model for surface annotation."""

    description: str | None = None
    space: str
    label: str
    density: str
    hemisphere: Literal["left", "right"]


class VolumeAtlas(Resource):
    """Model for volume atlas resources."""

    space: str
    resolution: str
    resource_type: str


class VolumeTransform(Resource):
    """Model for volume transform resources."""

    source_space: str
    target_space: str
    resolution: str
    resource_type: str
    provider: str
    weight: float = 1.0


class VolumeAnnotation(Resource):
    """Model for volume annotation resources."""

    description: str | None = None
    space: str
    label: str
    resolution: str


class Node(BaseModel):
    """Node representation in transformation graph."""

    name: str
    species: str
    description: str
    references: Sequence[str | dict[str, str]] | None = None
    surfaces: Sequence[SurfaceAtlas] = Field(default_factory=list)
    volumes: Sequence[VolumeAtlas] = Field(default_factory=list)
    surface_annotations: Sequence[SurfaceAnnotation] = Field(default_factory=list)
    volume_annotations: Sequence[VolumeAnnotation] = Field(default_factory=list)

    def __repr__(self) -> str:
        """String representation."""
        surface_str = "\n".join(s.name for s in self.surfaces)
        volume_str = "\n".join(v.name for v in self.volumes)
        surface_annot_str = "\n".join(s.name for s in self.surface_annotations)
        volume_annot_str = "\n".join(v.name for v in self.volume_annotations)
        return (
            "\nNode:"
            f"\n\tname={self.name},\n"
            f"\tspecies={self.species}\n"
            f"\tdescription={self.description}\n"
            f"\tsurfaces=[{surface_str}]\n"
            f"\tvolumes=[{volume_str}]"
            f"\tsurface annotations=[{surface_annot_str}]\n"
            f"\tvolume annotations=[{volume_annot_str}]\n"
        )


class Edge(BaseModel):
    """Edge representation in transformation graph."""

    surface_transforms: Sequence[SurfaceTransform] = Field(default_factory=list)
    volume_transforms: Sequence[VolumeTransform] = Field(default_factory=list)

    def __repr__(self) -> str:
        """String representation."""
        surface_str = "\n".join(s.name for s in self.surface_transforms)
        volume_str = "\n".join(v.name for v in self.volume_transforms)
        return f"\nEdge:\n\tsurfaces=[{surface_str}],\n\tvolumes=[{volume_str}]"


class HopMetadataDict(TypedDict, total=False):
    """Typed dict for per-hop transform metadata.

    Keys:
        source_space: Source space for this hop.
        target_space: Target space for this hop.
        provider: Provider name of the transform.
        references: Formatted reference strings.
        notes: Caveat/note strings.
    """

    source_space: str
    target_space: str
    provider: str
    references: list[str]
    notes: list[str]


class SpaceMetadataDict(TypedDict, total=False):
    """Typed dict for per-space node metadata.

    Keys:
        space: Space name.
        references: Formatted reference strings for the space.
    """

    space: str
    references: list[str]


class TransformMetadata:
    """Encapsulates provenance metadata for a transform pipeline.

    Groups per-hop transform metadata and per-space node-level references
    so downstream consumers (terminal output, JSON writing) can distinguish
    between transform-level and node-level provenance.

    Attributes:
        transforms: Per-hop metadata (source/target spaces, provider, refs,
            notes). One dict per hop in the transformation chain.
        spaces: Per-space node-level references, deduplicated across the
            transformation path.
    """

    def __init__(
        self,
        transforms: Sequence[HopMetadataDict] | None = None,
        spaces: Sequence[SpaceMetadataDict] | None = None,
    ) -> None:
        """Initialize TransformMetadata.

        Args:
            transforms: Per-hop metadata dicts.
            spaces: Per-space node-level reference dicts.
        """
        self.transforms = transforms
        self.spaces = spaces


class TransformResult:
    """Result of a transformation, including output path and metadata.

    When used without attribute access—printing, f-strings, passing to
    ``os.path.*`` or ``nibabel``—behaves like a :class:`~pathlib.Path`.
    Access :attr:`metadata` for structured provenance, or the backward-
    compatible :attr:`references` and :attr:`notes` properties for
    flattened lists.

    Attributes:
        path: Output file path, or ``None`` if the transform failed.
        metadata: Structured provenance (per-hop + per-space), or
            ``None`` if no transform succeeded.
    """

    def __init__(
        self,
        output_path: Path | None = None,
        metadata: TransformMetadata | None = None,
    ) -> None:
        """Initialize TransformResult.

        Args:
            output_path: Path to the output file, or ``None`` on failure.
            metadata: Structured provenance metadata, or ``None``.
        """
        self._output_path = output_path
        self.path = output_path
        self.metadata = metadata

    # --- Backward-compatible computed properties ---

    @property
    def references(self) -> list[str] | None:
        """Flattened references from all spaces and hops.

        Space-level references come first, then per-hop references.
        Returns ``None`` when no metadata is attached.
        """
        if self.metadata is None:
            return None
        refs: list[str] = []
        for space in self.metadata.spaces or ():
            refs.extend(space.get("references") or ())  # type: ignore[arg-type]
        for hop in self.metadata.transforms or ():
            refs.extend(hop.get("references") or ())  # type: ignore[arg-type]
        return refs or None

    @property
    def notes(self) -> list[str] | None:
        """Flattened notes from all hops.

        Returns ``None`` when no metadata is attached.
        """
        if self.metadata is None:
            return None
        notes: list[str] = []
        for hop in self.metadata.transforms or ():
            notes.extend(hop.get("notes") or ())  # type: ignore[arg-type]
        return notes or None

    # --- Path-like protocol ---

    def __eq__(self, other: object) -> bool:
        """Compare with :class:`~pathlib.Path` or another :class:`TransformResult`.

        When comparing to a :class:`~pathlib.Path`, only the output path is
        considered. When comparing to another :class:`TransformResult`, all
        fields are compared.
        """
        if isinstance(other, TransformResult):
            return (
                self._output_path == other._output_path
                and self.metadata == other.metadata
            )
        return self._output_path == other

    def __fspath__(self) -> str:
        """Return the filesystem path string.

        Raises:
            TypeError: If the output path is ``None``.
        """
        if self._output_path is None:
            raise TypeError("Cannot get filesystem path from None result")
        return str(self._output_path)

    def __bool__(self) -> bool:
        """Return ``True`` if the transform succeeded (path is not ``None``)."""
        return self._output_path is not None

    def __str__(self) -> str:
        """Return the output path string."""
        return str(self._output_path) if self._output_path else "<None>"

    def __repr__(self) -> str:
        """Return the debug representation."""
        meta = self.metadata
        n_hops = len(meta.transforms) if meta and meta.transforms else 0
        n_spaces = len(meta.spaces) if meta and meta.spaces else 0
        return (
            f"TransformResult(path={self._output_path!r}, "
            f"hops={n_hops}, spaces={n_spaces})"
        )

    # --- Path method delegation (test compatibility) ---

    def exists(self) -> bool:
        """Delegate to :meth:`Path.exists`."""
        return self._output_path.exists() if self._output_path else False

    @property
    def parent(self) -> Path | None:
        """Parent directory of the output path."""
        return self._output_path.parent if self._output_path else None

    @property
    def name(self) -> str:
        """Basename of the output file."""
        return self._output_path.name if self._output_path else ""

    def is_file(self) -> bool:
        """Delegate to :meth:`Path.is_file`."""
        return self._output_path.is_file() if self._output_path else False

    # --- Disk serialization ---

    def save_metadata(
        self,
        metadata_path: Path | None = None,
    ) -> Path:
        """Save per-hop and node-level metadata to disk.

        Format is inferred from the file extension:

        - ``.md`` → Markdown (human-readable)
        - ``.json`` → structured JSON

        Args:
            metadata_path: Destination file path. Defaults to
                ``<output>.md``.

        Returns:
            Path to the written metadata file.

        Raises:
            ValueError: If the output path is ``None``.
            NotImplementedError: If the file extension is not
                ``.md`` or ``.json``.
        """
        if self._output_path is None:
            raise ValueError("Cannot save metadata: output path is None")

        if metadata_path is None:
            metadata_path = Path(str(self.path) + ".md")

        ext = metadata_path.suffix.lower()
        if ext not in (".md", ".json"):
            raise NotImplementedError(
                f"Unsupported file extension '{ext}'. "
                "Use '.md' for Markdown or '.json' for JSON."
            )

        spaces_out = self._serialize_spaces()
        transforms_out = self._serialize_transforms()
        transform_path = self._build_space_path()

        if ext == ".json":
            self._write_json(metadata_path, transform_path, spaces_out, transforms_out)
        else:
            content = self._format_markdown(transform_path, spaces_out, transforms_out)
            metadata_path.write_text(content)

        return metadata_path

    def _serialize_spaces(
        self,
    ) -> list[_SerializedEntry] | None:
        """Convert space metadata to serializable dicts."""
        if self.metadata is None or not self.metadata.spaces:
            return None
        result: list[_SerializedEntry] = []
        for space in self.metadata.spaces:
            entry: _SerializedEntry = {}
            name = space.get("space")
            if name:
                entry["space"] = name
            refs = space.get("references")
            if refs:
                entry["references"] = list(refs)
            result.append(entry)
        return result

    def _serialize_transforms(
        self,
    ) -> list[_SerializedEntry] | None:
        """Convert transform hop metadata to serializable dicts."""
        if self.metadata is None or not self.metadata.transforms:
            return None
        result: list[_SerializedEntry] = []
        for hop in self.metadata.transforms:
            entry: _SerializedEntry = {}
            src = hop.get("source_space")
            if src:
                entry["source_space"] = src
            tgt = hop.get("target_space")
            if tgt:
                entry["target_space"] = tgt
            prov = hop.get("provider")
            if prov:
                entry["provider"] = prov
            refs = hop.get("references")
            if refs:
                entry["references"] = list(refs)
            notes = hop.get("notes")
            if notes:
                entry["notes"] = list(notes)
            result.append(entry)
        return result

    def _build_space_path(self) -> list[str]:
        """Extract ordered space sequence from hop metadata."""
        if self.metadata is None or self.metadata.transforms is None:
            return []
        space_names: list[str] = []
        for hop in self.metadata.transforms:
            src = hop.get("source_space", "")
            if src and (not space_names or space_names[-1] != src):
                space_names.append(src)
            tgt = hop.get("target_space", "")
            if tgt and tgt not in space_names:
                space_names.append(tgt)
        return space_names

    def _write_json(
        self,
        metadata_path: Path,
        transform_path: list[str],
        spaces: list[_SerializedEntry] | None,
        transforms: list[_SerializedEntry] | None,
    ) -> None:
        """Write metadata as a JSON file."""
        doc: dict[str, str | list[str] | list[_SerializedEntry]] = {
            "output_file": str(self.path)
        }
        if transform_path:
            doc["transform_path"] = transform_path
        if spaces:
            doc["spaces"] = spaces
        if transforms:
            doc["transforms"] = transforms
        doc["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
        metadata_path.write_text(json.dumps(doc, indent=2) + "\n")

    def _format_markdown(
        self,
        transform_path: list[str],
        spaces: list[_SerializedEntry] | None,
        transforms: list[_SerializedEntry] | None,
    ) -> str:
        """Build a Markdown representation of the metadata."""
        lines: list[str] = []
        self._write_header(lines, transform_path)
        self._write_spaces_section(lines, spaces)
        self._write_transforms_section(lines, transforms)
        lines.append(
            f"*Generated at {datetime.datetime.now(datetime.UTC).isoformat()}*"
        )
        lines.append("")
        return "\n".join(lines)

    def _write_header(self, lines: list[str], transform_path: list[str]) -> None:
        """Write the Markdown transformation header."""
        if transform_path:
            header = " -> ".join(transform_path)
        elif self._output_path:
            header = str(self._output_path.name)
        else:
            header = "Transformation"
        lines.append(f"# Transformation: {header}")
        lines.append("")

    def _write_spaces_section(
        self, lines: list[str], spaces: list[_SerializedEntry] | None
    ) -> None:
        """Write the Markdown spaces section."""
        if not spaces:
            return
        lines.append("## Spaces")
        lines.append("")
        for space in spaces:
            space_name = space.get("space", "Unknown")
            lines.append(f"### {space_name}")
            lines.append("")
            refs = space.get("references")
            if refs:
                lines.extend(f"- {r}" for r in refs)
                lines.append("")

    def _write_transforms_section(
        self, lines: list[str], transforms: list[_SerializedEntry] | None
    ) -> None:
        """Write the Markdown transforms section."""
        if not transforms:
            return
        lines.append("## Transforms")
        lines.append("")
        for hop in transforms:
            src = hop.get("source_space", "?")
            tgt = hop.get("target_space", "?")
            prov = hop.get("provider", "")
            prov_str = f" [{prov}]" if prov else ""
            lines.append(f"### {src} -> {tgt}{prov_str}")
            lines.append("")
            refs = hop.get("references")
            if refs:
                lines.append("**References:**")
                lines.extend(f"- {r}" for r in refs)
                lines.append("")
            notes = hop.get("notes")
            if notes:
                lines.append("**Caveats:**")
                lines.extend(f"- {n}" for n in notes)
                lines.append("")
