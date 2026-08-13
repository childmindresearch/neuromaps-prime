"""Unit tests for GraphUtils."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime.graph.utils import GraphUtils


class TestCollectSpaceMetadata:
    """Tests for GraphUtils.collect_space_metadata."""

    @pytest.fixture
    def utils(self) -> GraphUtils:
        """Create GraphUtils with a mocked graph and cache."""
        return MagicMock(spec=GraphUtils)  # type: ignore[return-value]

    def _make_node(
        self, references: list[str | dict[str, str]] | None = None
    ) -> MagicMock:
        """Create a mock Node with the given references."""
        return MagicMock(references=references)

    def test_returns_none_when_no_references(self, utils: MagicMock) -> None:
        """Returns None when all spaces have no references."""
        utils.get_node_data.return_value = self._make_node(references=None)

        with patch(
            "neuromaps_prime.graph.utils.GraphUtils.collect_space_metadata",
            wraps=GraphUtils.collect_space_metadata,
        ):
            result = GraphUtils.collect_space_metadata(utils, ["A", "B"])

        assert result is None
        utils.get_node_data.assert_any_call("A")
        utils.get_node_data.assert_any_call("B")

    def test_returns_none_when_references_are_empty(self, utils: MagicMock) -> None:
        """Returns None when all spaces have empty references."""
        utils.get_node_data.return_value = self._make_node(references=[])

        result = GraphUtils.collect_space_metadata(utils, ["A"])
        assert result is None

    def test_returns_formatted_metadata_when_references_exist(
        self, utils: MagicMock
    ) -> None:
        """Returns space metadata dicts when references are present."""
        utils.get_node_data.return_value = self._make_node(
            references=["Smith et al. 2020"]
        )

        result = GraphUtils.collect_space_metadata(utils, ["A", "B"])  # type: ignore[arg-type]

        assert result is not None
        assert len(result) == 2
        assert result[0]["space"] == "A"
        assert result[0]["references"] == ["Smith et al. 2020"]
        assert result[1]["space"] == "B"
        assert result[1]["references"] == ["Smith et al. 2020"]

    def test_deduplicates_spaces(self, utils: MagicMock) -> None:
        """Skips spaces that appear multiple times in the path."""
        utils.get_node_data.return_value = self._make_node(references=["Citation A"])

        result = GraphUtils.collect_space_metadata(  # type: ignore[arg-type]
            utils,
            ["A", "B", "A", "C"],
        )

        assert result is not None
        assert len(result) == 3
        space_names = [entry["space"] for entry in result]
        assert space_names == ["A", "B", "C"]
        # A is only looked up once
        utils.get_node_data.assert_called()
        call_args = [c[0][0] for c in utils.get_node_data.call_args_list]
        assert call_args == ["A", "B", "C"]

    def test_format_reference_called_for_each_raw_ref(self, utils: MagicMock) -> None:
        """format_reference is called for each raw reference."""
        utils.get_node_data.return_value = self._make_node(
            references=[
                "Simple ref",
                {"citation": "Author et al.", "doi": "10.1234/test"},
            ]
        )

        result = GraphUtils.collect_space_metadata(utils, ["A"])  # type: ignore[arg-type]

        assert result is not None
        assert result[0]["space"] == "A"
        assert len(result[0]["references"]) == 2
        assert result[0]["references"][0] == "Simple ref"
        assert "Author et al." in result[0]["references"][1]
        assert "10.1234/test" in result[0]["references"][1]

    def test_skips_spaces_with_no_references(self, utils: MagicMock) -> None:
        """Only includes spaces that have formatted references."""
        node_a = self._make_node(references=["A ref"])
        node_b = self._make_node(references=None)
        node_c = self._make_node(references=["C ref"])

        utils.get_node_data.side_effect = [node_a, node_b, node_c]

        result = GraphUtils.collect_space_metadata(utils, ["A", "B", "C"])  # type: ignore[arg-type]

        assert result is not None
        assert len(result) == 2
        assert result[0]["space"] == "A"
        assert result[1]["space"] == "C"

    def test_empty_path_returns_none(self, utils: MagicMock) -> None:
        """Returns None for an empty space path."""
        result = GraphUtils.collect_space_metadata(utils, [])  # type: ignore[arg-type]
        assert result is None
        utils.get_node_data.assert_not_called()
