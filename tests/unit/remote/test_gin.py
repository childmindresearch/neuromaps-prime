"""Unit tests for GIN remote storage."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime.remote.gin import (
    GINFileMeta,
    GINStorage,
    _filename_from_disposition,
)

if TYPE_CHECKING:
    from pathlib import Path

MOCK_CONTENT = b"surface data"
MOCK_FILENAME = "file.nii.gz"
MOCK_URL = "https://gin.g-node.org/org/user/raw/master/file.nii.gz"
MOCK_DISPOSITION = f'attachment; filename="{MOCK_FILENAME}"'


@pytest.fixture
def mock_meta_response() -> MagicMock:
    """Mock HEAD response carrying a ``Content-Disposition`` filename."""
    resp = MagicMock()
    resp.headers = {"Content-Disposition": MOCK_DISPOSITION}
    return resp


class TestFilenameFromDisposition:
    """Test suite for parsing ``Content-Disposition`` filenames."""

    def test_quoted(self) -> None:
        """Quoted filename is extracted."""
        assert _filename_from_disposition(MOCK_DISPOSITION) == MOCK_FILENAME

    def test_unquoted(self) -> None:
        """Unquoted filename is extracted."""
        assert _filename_from_disposition("attachment; filename=test.txt") == "test.txt"

    def test_none(self) -> None:
        """Missing header returns None."""
        assert _filename_from_disposition(None) is None

    def test_no_filename(self) -> None:
        """A disposition without a filename returns None."""
        assert _filename_from_disposition("attachment") is None


class TestGINFileMeta:
    """Test suite for GIN file metadata."""

    def test_valid(self) -> None:
        """Filename is stored on instantiation."""
        meta = GINFileMeta(name=MOCK_FILENAME)
        assert meta.name == MOCK_FILENAME


class TestGINStorage:
    """Test suite for GINStorage metadata and download behavior."""

    def test_get_meta_parses_filename(self, mock_meta_response: MagicMock) -> None:
        """Filename is parsed from the Content-Disposition header."""
        with patch(
            "neuromaps_prime.remote.gin.requests.head", return_value=mock_meta_response
        ) as mock_head:
            meta = GINStorage().get_meta(MOCK_URL)
        assert meta.name == MOCK_FILENAME
        mock_head.assert_called_once_with(MOCK_URL, timeout=90)

    def test_get_meta_falls_back_to_url_segment(self) -> None:
        """Falls back to the URL's final path segment when no filename header."""
        resp = MagicMock()
        resp.headers = {}
        with patch(
            "neuromaps_prime.remote.gin.requests.head", return_value=resp
        ) as mock_head:
            meta = GINStorage().get_meta(MOCK_URL)
        assert meta.name == MOCK_FILENAME
        mock_head.assert_called_once_with(MOCK_URL, timeout=90)

    @patch("neuromaps_prime.remote.gin.requests.head")
    def test_get_meta_raises_on_http_error(self, mock_head: MagicMock) -> None:
        """HTTP errors from the remote server are propagated."""
        mock_head.return_value.raise_for_status.side_effect = Exception("HTTP Error")
        with pytest.raises(Exception, match="HTTP Error"):
            GINStorage().get_meta(MOCK_URL)

    def test_download_valid(
        self, mock_meta_response: MagicMock, tmp_path: Path
    ) -> None:
        """Streamed bytes are written to dest under the storage-side name."""
        with (
            patch(
                "neuromaps_prime.remote.gin.requests.head",
                return_value=mock_meta_response,
            ),
            patch(
                "neuromaps_prime.remote.gin.requests.get",
                return_value=MagicMock(
                    iter_content=MagicMock(return_value=[MOCK_CONTENT])
                ),
            ),
        ):
            result = GINStorage().download(MOCK_URL, tmp_path)
        assert result == tmp_path / MOCK_FILENAME
        assert result.read_bytes() == MOCK_CONTENT

    def test_download_cache_hit(
        self, mock_meta_response: MagicMock, tmp_path: Path
    ) -> None:
        """A pre-existing file is returned without downloading again."""
        stored = tmp_path / MOCK_FILENAME
        stored.write_bytes(MOCK_CONTENT)
        with (
            patch(
                "neuromaps_prime.remote.gin.requests.head",
                return_value=mock_meta_response,
            ),
            patch("neuromaps_prime.remote.gin.requests.get") as mock_get,
        ):
            result = GINStorage().download(MOCK_URL, tmp_path)
        assert result == stored
        mock_get.assert_not_called()

    @patch("neuromaps_prime.remote.gin.requests.get")
    @patch("neuromaps_prime.remote.gin.requests.head")
    def test_download_cleans_tmp_on_failure(
        self,
        mock_head: MagicMock,
        mock_get: MagicMock,
        mock_meta_response: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A failed download raises and leaves no partial file at the target name."""
        mock_head.return_value = mock_meta_response
        mock_get.return_value.iter_content.side_effect = RuntimeError("network drop")
        with pytest.raises(RuntimeError, match="network drop"):
            GINStorage().download(MOCK_URL, tmp_path)
        assert not (tmp_path / MOCK_FILENAME).exists()
        assert not list(tmp_path.glob("*.part"))

    def test_default_chunk_size(self) -> None:
        """chunk_size defaults to 8192."""
        assert GINStorage().chunk_size == 8192

    def test_custom_chunk_size(self) -> None:
        """chunk_size is correctly set when provided."""
        assert GINStorage(chunk_size=4096).chunk_size == 4096
