"""Unit tests for remote fetcher."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from neuromaps_prime import remote
from neuromaps_prime.fetcher import download_and_validate, id_storage

if TYPE_CHECKING:
    from pathlib import Path


class TestIDStorage:
    """Test suite for identifying storage location."""

    @pytest.mark.parametrize(
        ("storage", "expected"),
        [
            ("https://osf.io/project", "osf"),
            ("https://github.com/owner", "github"),
            (
                "https://raw.githubusercontent.com/owner/repo/refs/tags/v1.0/file.txt",
                "github",
            ),
        ],
    )
    def test_valid(self, storage: str, expected: str) -> None:
        """Test downloading from valid storage options."""
        assert id_storage(storage) == expected

    def test_unknown(self) -> None:
        """Test None returned if unknown uri."""
        assert id_storage("https://google.com") is None

    def test_invalid(self) -> None:
        """Test None returned if invalid."""
        assert id_storage("") is None


class TestDownloadAndValidate:
    """Test suite for fetching and validating from remote uri."""

    def test_unknown_uri_raises(self, tmp_path: Path) -> None:
        """Test unidentifiable uri raises ValueError."""
        with pytest.raises(ValueError, match="Could not identify storage"):
            download_and_validate("https://google.com", tmp_path / "invalid.txt")

    @pytest.mark.parametrize(
        ("storage_cls", "mock_uri"),
        [
            (remote.OSFStorage, "https://files.osf.io/v1/resources/abcde"),
            (remote.GitHubStorage, "https://github.com/owner/repo/blob/v1.0/file.txt"),
            (
                remote.GitHubStorage,
                "https://raw.githubusercontent.com/owner/repo/refs/tags/v1.0/file.txt",
            ),
        ],
    )
    def test_valid_calls_download(
        self, storage_cls: object, mock_uri: str, tmp_path: Path
    ) -> None:
        """Test valid download."""
        dest = tmp_path / "out.surf.gii"
        with patch.object(storage_cls, "download") as mock_download:
            download_and_validate(mock_uri, dest)
        mock_download.assert_called_once_with(mock_uri, dest)
