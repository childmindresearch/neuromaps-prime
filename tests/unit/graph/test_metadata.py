"""Tests for metadata formatting helpers."""

from __future__ import annotations

from neuromaps_prime.graph.metadata import format_reference


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
