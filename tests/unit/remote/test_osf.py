"""Unit tests for OSF remote storage."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime.remote.osf import OSFFileExtra, OSFFileMeta, OSFHashes, OSFStorage

if TYPE_CHECKING:
    from pathlib import Path

MOCK_MD5 = hashlib.md5(b"test content", usedforsecurity=False).hexdigest()

MOCK_META = {
    "data": {
        "attributes": {
            "name": "test.txt",
            "size": 12,
            "extra": {"hashes": {"md5": MOCK_MD5}},
        }
    }
}


@pytest.fixture
def mock_meta_response() -> MagicMock:
    """Mock metadata response object."""
    return MagicMock(json=MagicMock(return_value=MOCK_META))


class TestOSFHashes:
    """Test suite for validating OSF hashes."""

    def test_sha256_optional(self) -> None:
        """Test sha256 defaults to None when not provided."""
        h = OSFHashes(md5=MOCK_MD5)
        assert h.sha256 is None

    def test_both_fields(self) -> None:
        """Test both md5 and sha256 are stored correctly when provided."""
        h = OSFHashes(md5=MOCK_MD5, sha256="abc123")
        assert h.sha256 == "abc123"


class TestOSFFileMeta:
    """Test suite for OSF file metadata."""

    def test_valid(self) -> None:
        """Test metadata fields are correctly assigned on instantiation."""
        meta = OSFFileMeta(
            name="test.txt",
            size=12,
            extra=OSFFileExtra(hashes=OSFHashes(md5=MOCK_MD5)),
        )
        assert meta.name == "test.txt"
        assert meta.extra.hashes.md5 == MOCK_MD5


class TestOSFStorage:
    """Test suite for OSFStorage download and metadata retrieval."""

    @patch("neuromaps_prime.remote.osf.requests.get")
    def test_get_meta(self, mock_get: MagicMock, mock_meta_response: MagicMock) -> None:
        """Test get_meta returns parsed OSFFileMeta and calls correct URL."""
        mock_get.return_value = mock_meta_response
        meta = OSFStorage().get_meta("https://files.osf.io/v1/resources/abcde")
        assert meta.name == "test.txt"
        assert meta.extra.hashes.md5 == MOCK_MD5
        mock_get.assert_called_once_with(
            "https://files.osf.io/v1/resources/abcde",
            params={"meta": ""},
            timeout=90,
        )

    @patch("neuromaps_prime.remote.osf.requests.get")
    def test_get_meta_raises_on_http_error(self, mock_get: MagicMock) -> None:
        """Test get_meta propagates HTTP errors from the remote server."""
        mock_get.return_value.raise_for_status.side_effect = Exception("HTTP Error")
        with pytest.raises(Exception, match="HTTP Error"):
            OSFStorage().get_meta("https://files.osf.io/v1/resources/abcde")

    @patch("neuromaps_prime.remote.osf.requests.get")
    def test_download_valid(
        self, mock_get: MagicMock, mock_meta_response: MagicMock, tmp_path: Path
    ) -> None:
        """Test download writes correct bytes to dest when checksum matches."""
        content = b"test content"
        mock_download = MagicMock(iter_content=MagicMock(return_value=[content]))
        mock_get.side_effect = [mock_meta_response, mock_download]
        dest = tmp_path / "test.txt"
        OSFStorage().download("https://files.osf.io/v1/resources/abcde", dest)
        assert dest.read_bytes() == content

    @patch("neuromaps_prime.remote.osf.requests.get")
    def test_download_checksum_mismatch(
        self, mock_get: MagicMock, mock_meta_response: MagicMock, tmp_path: Path
    ) -> None:
        """Test download raises ValueError when downloaded content fails checksum."""
        mock_download = MagicMock(
            iter_content=MagicMock(return_value=[b"wrong content"])
        )
        mock_get.side_effect = [mock_meta_response, mock_download]
        with pytest.raises(ValueError, match="Checksum mismatch"):
            OSFStorage().download(
                "https://files.osf.io/v1/resources/abcde", tmp_path / "test.txt"
            )

    def test_default_chunk_size(self) -> None:
        """Test chunk_size defaults to 8192."""
        assert OSFStorage().chunk_size == 8192

    def test_custom_chunk_size(self) -> None:
        """Test chunk_size is correctly set when provided."""
        assert OSFStorage(chunk_size=4096).chunk_size == 4096
