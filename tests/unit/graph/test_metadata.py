"""Tests for metadata formatting helpers."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from neuromaps_prime.graph.metadata import format_reference, print_metadata_summary
from neuromaps_prime.graph.models import TransformMetadata, TransformResult

if TYPE_CHECKING:
    from pathlib import Path


class TestFormatReference:
    """Tests for the format_reference helper."""

    def test_string_reference_passthrough(self) -> None:
        """Plain string references are returned as-is."""
        assert format_reference("Smith et al. 2020") == "Smith et al. 2020"

    def test_dict_with_citation_only(self) -> None:
        """Dict with only citation key returns just the citation."""
        result = format_reference({"citation": "Smith et al. 2020"})
        assert result == "Smith et al. 2020"

    def test_dict_with_citation_and_doi(self) -> None:
        """Dict with citation and DOI is joined with pipe."""
        result = format_reference(
            {"citation": "Smith et al. 2020", "doi": "10.1234/abc"}
        )
        assert result == "Smith et al. 2020 | DOI: 10.1234/abc"

    def test_dict_with_all_keys(self) -> None:
        """Dict with all three keys produces three pipe-separated parts."""
        result = format_reference(
            {
                "citation": "Smith et al. 2020",
                "doi": "10.1234/abc",
                "url": "https://example.org",
            }
        )
        assert (
            result == "Smith et al. 2020 | DOI: 10.1234/abc | URL: https://example.org"
        )

    def test_dict_with_missing_citation(self) -> None:
        """Dict without citation key still formats DOI and URL."""
        result = format_reference({"doi": "10.1234/abc", "url": "https://example.org"})
        assert result == "DOI: 10.1234/abc | URL: https://example.org"

    def test_dict_with_empty_values(self) -> None:
        """Dict with empty-string values omits those parts."""
        result = format_reference({"citation": "", "doi": "", "url": ""})
        assert result == ""


class TestPrintMetadataSummary:
    """Tests for the print_metadata_summary function."""

    def test_prints_nothing_when_result_is_none(self) -> None:
        """Does nothing when result is None/falsy."""
        buf = StringIO()
        print_metadata_summary(TransformResult(), file=buf)
        assert buf.getvalue() == ""

    def test_prints_nothing_when_metadata_is_none(self, tmp_path: Path) -> None:
        """Does nothing when metadata is None."""
        buf = StringIO()
        result = TransformResult(output_path=tmp_path / "out.nii")
        print_metadata_summary(result, file=buf)
        assert buf.getvalue() == ""

    def test_prints_spaces_and_hops(self, tmp_path: Path) -> None:
        """Prints grouped spaces and per-hop metadata."""
        meta = TransformMetadata(
            transforms=[
                {
                    "source_space": "A",
                    "target_space": "B",
                    "provider": "RheMap",
                    "references": ["Smith et al. 2020"],
                    "notes": ["Experimental alignment"],
                },
            ],
            spaces=[
                {"space": "A", "references": ["Space A citation"]},
                {"space": "B", "references": ["Space B citation"]},
            ],
        )
        result = TransformResult(output_path=tmp_path / "out.nii", metadata=meta)
        buf = StringIO()
        print_metadata_summary(result, file=buf)
        output = buf.getvalue()

        # Check header
        assert "=== Transformation: A -> B ===" in output
        # Check spaces section
        assert "--- Spaces ---" in output
        assert "A:" in output
        assert "Ref: Space A citation" in output
        assert "B:" in output
        assert "Ref: Space B citation" in output
        # Check hop section
        assert "--- A -> B [RheMap] ---" in output
        assert "References:" in output
        assert "- Smith et al. 2020" in output
        assert "Caveats:" in output
        assert "- Experimental alignment" in output
        # Check footer
        assert "=== End ===" in output

    def test_prints_multiple_hops(self, tmp_path: Path) -> None:
        """Prints multiple hops sequentially."""
        meta = TransformMetadata(
            transforms=[
                {
                    "source_space": "A",
                    "target_space": "B",
                    "provider": "RheMap",
                    "references": ["Ref 1"],
                    "notes": [],
                },
                {
                    "source_space": "B",
                    "target_space": "C",
                    "provider": "CIVET",
                    "references": ["Ref 2"],
                    "notes": ["Note 2"],
                },
            ],
            spaces=[
                {"space": "A", "references": ["A ref"]},
                {"space": "B", "references": ["B ref"]},
                {"space": "C", "references": ["C ref"]},
            ],
        )
        result = TransformResult(output_path=tmp_path / "out.nii", metadata=meta)
        buf = StringIO()
        print_metadata_summary(result, file=buf)
        output = buf.getvalue()

        # Check path shows all spaces
        assert "A -> B -> C" in output
        # Check both hops appear
        assert "--- A -> B [RheMap] ---" in output
        assert "--- B -> C [CIVET] ---" in output
        assert "- Ref 1" in output
        assert "- Ref 2" in output
        assert "- Note 2" in output

    def test_skips_sections_when_empty(self, tmp_path: Path) -> None:
        """Omits spaces and caveats sections when their lists are empty."""
        meta = TransformMetadata(
            transforms=[
                {
                    "source_space": "X",
                    "target_space": "Y",
                    "provider": "Test",
                    "references": ["Only ref"],
                    "notes": [],
                },
            ],
            spaces=[],
        )
        result = TransformResult(output_path=tmp_path / "out.nii", metadata=meta)
        buf = StringIO()
        print_metadata_summary(result, file=buf)
        output = buf.getvalue()

        assert "--- Spaces ---" not in output
        assert "Caveats:" not in output
        assert "- Only ref" in output

    def test_accepts_custom_file_argument(self, tmp_path: Path) -> None:
        """Writes to the provided file-like object."""
        meta = TransformMetadata(
            transforms=[
                {
                    "source_space": "A",
                    "target_space": "B",
                    "provider": "Test",
                    "references": [],
                    "notes": [],
                },
            ],
            spaces=[],
        )
        result = TransformResult(output_path=tmp_path / "out.nii", metadata=meta)
        buf = StringIO()
        print_metadata_summary(result, file=buf)
        assert len(buf.getvalue()) > 0
