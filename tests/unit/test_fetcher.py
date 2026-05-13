"""Unit tests for remote fetcher."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime.fetcher import download_and_validate, id_storage

if TYPE_CHECKING:
    from pathlib import Path


class TestIDStorage:
    """Test suite for identifying storage location."""

    def test_osf(self) -> None:
        """Test downloading from osf storage."""
        assert id_storage("https://osf.io/project") == "osf"

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

    @patch("neuromaps_prime.fetcher.OSFStorage")
    def test_osf_calls_download(self, mock_cls: MagicMock, tmp_path: Path) -> None:
        """Test OSF download."""
        mock_osf = mock_cls.return_value
        mock_uri = "https://files.osf.io/v1/resources/abcde"
        dest = tmp_path / "out.surf.gii"
        download_and_validate(mock_uri, dest)
        mock_osf.download.assert_called_once_with(mock_uri, dest)
