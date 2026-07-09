"""Unit tests for GitHub remote storage."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from neuromaps_prime.remote.github import GitHubFileMeta, GitHubStorage

if TYPE_CHECKING:
    from pathlib import Path

MOCK_CONTENT = b"test content"
MOCK_SHA = hashlib.sha1(
    f"blob {len(MOCK_CONTENT)}\0".encode() + MOCK_CONTENT, usedforsecurity=False
).hexdigest()

MOCK_OWNER = "mockowner"
MOCK_REPO = "mockrepo"
MOCK_REF = "v0.1"
MOCK_PATH = "data/test.txt"
MOCK_URL = f"https://github.com/{MOCK_OWNER}/{MOCK_REPO}/blob/{MOCK_REF}/{MOCK_PATH}"
MOCK_DOWNLOAD_URL = (
    f"https://raw.githubusercontent.com/{MOCK_OWNER}/{MOCK_REPO}/{MOCK_REF}/{MOCK_PATH}"
)
MOCK_RAW_URL = (
    f"https://raw.githubusercontent.com/{MOCK_OWNER}/{MOCK_REPO}/{MOCK_REF}/{MOCK_PATH}"
)
MOCK_RAW_TAG_URL = (
    f"https://raw.githubusercontent.com/{MOCK_OWNER}/{MOCK_REPO}/"
    f"refs/tags/{MOCK_REF}/{MOCK_PATH}"
)
MOCK_RAW_HEAD_URL = (
    f"https://raw.githubusercontent.com/{MOCK_OWNER}/{MOCK_REPO}/"
    f"refs/heads/{MOCK_REF}/{MOCK_PATH}"
)

MOCK_META = {
    "name": "test.txt",
    "size": len(MOCK_CONTENT),
    "sha": MOCK_SHA,
    "download_url": MOCK_DOWNLOAD_URL,
}


@pytest.fixture
def mock_meta_response() -> MagicMock:
    """Mock metadata response object."""
    return MagicMock(json=MagicMock(return_value=MOCK_META))


class TestGitHubFileMeta:
    """Test suite for GitHub file metadata."""

    def test_valid(self) -> None:
        """Test metadata fields are correctly assigned on instantiation."""
        meta = GitHubFileMeta(**MOCK_META)
        assert meta.name == "test.txt"
        assert meta.sha == MOCK_SHA
        assert meta.download_url == MOCK_DOWNLOAD_URL


class TestGitHubStorage:
    """Test suite for GitHubStorage download and metadata retrieval."""

    def test_parse_blob_url(self) -> None:
        """Test blob URL is correctly parsed into owner/repo/ref/path."""
        owner, repo, ref, path = GitHubStorage._parse(MOCK_URL)
        assert owner == MOCK_OWNER
        assert repo == MOCK_REPO
        assert ref == MOCK_REF
        assert path == MOCK_PATH

    def test_parse_raw_url(self) -> None:
        """Test bare raw URL is correctly parsed into owner/repo/ref/path."""
        owner, repo, ref, path = GitHubStorage._parse(MOCK_RAW_URL)
        assert owner == MOCK_OWNER
        assert repo == MOCK_REPO
        assert ref == MOCK_REF
        assert path == MOCK_PATH

    def test_parse_raw_tag_url(self) -> None:
        """Test raw URL with refs/tags/ prefix is correctly parsed."""
        owner, repo, ref, path = GitHubStorage._parse(MOCK_RAW_TAG_URL)
        assert owner == MOCK_OWNER
        assert repo == MOCK_REPO
        assert ref == MOCK_REF
        assert path == MOCK_PATH

    def test_parse_raw_head_url(self) -> None:
        """Test raw URL with refs/heads/ prefix is correctly parsed."""
        owner, repo, ref, path = GitHubStorage._parse(MOCK_RAW_HEAD_URL)
        assert owner == MOCK_OWNER
        assert repo == MOCK_REPO
        assert ref == MOCK_REF
        assert path == MOCK_PATH

    def test_parse_invalid_url_raises(self) -> None:
        """Test parsing a non-GitHub URL raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized GitHub URL"):
            GitHubStorage._parse(f"https://github.com/{MOCK_OWNER}/{MOCK_REPO}")

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_get_meta(self, mock_get: MagicMock, mock_meta_response: MagicMock) -> None:
        """Test get_meta returns parsed GitHubFileMeta and calls correct URL."""
        mock_get.return_value = mock_meta_response
        meta = GitHubStorage().get_meta(MOCK_URL)
        assert meta.name == "test.txt"
        assert meta.sha == MOCK_SHA
        mock_get.assert_called_once_with(
            f"https://api.github.com/repos/{MOCK_OWNER}/{MOCK_REPO}/contents/{MOCK_PATH}",
            params={"ref": MOCK_REF},
            timeout=90,
        )

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_get_meta_raises_on_http_error(self, mock_get: MagicMock) -> None:
        """Test get_meta propagates HTTP errors from the remote server."""
        mock_get.return_value.raise_for_status.side_effect = Exception("HTTP Error")
        with pytest.raises(Exception, match="HTTP Error"):
            GitHubStorage().get_meta(MOCK_URL)

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_download_valid(
        self, mock_get: MagicMock, mock_meta_response: MagicMock, tmp_path: Path
    ) -> None:
        """Test download writes correct bytes to dest when checksum matches."""
        mock_download = MagicMock(iter_content=MagicMock(return_value=[MOCK_CONTENT]))
        mock_get.side_effect = [mock_meta_response, mock_download]
        dest = tmp_path / "test.txt"
        GitHubStorage().download(MOCK_URL, dest)
        assert dest.read_bytes() == MOCK_CONTENT

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_download_checksum_mismatch(
        self, mock_get: MagicMock, mock_meta_response: MagicMock, tmp_path: Path
    ) -> None:
        """Test download raises ValueError when downloaded content fails checksum."""
        mock_download = MagicMock(
            iter_content=MagicMock(return_value=[b"wrong content"])
        )
        mock_get.side_effect = [mock_meta_response, mock_download]
        with pytest.raises(ValueError, match="Checksum mismatch"):
            GitHubStorage().download(MOCK_URL, tmp_path / "test.txt")

    def test_default_chunk_size(self) -> None:
        """Test chunk_size defaults to 8192."""
        assert GitHubStorage().chunk_size == 8192

    def test_custom_chunk_size(self) -> None:
        """Test chunk_size is correctly set when provided."""
        assert GitHubStorage(chunk_size=4096).chunk_size == 4096
