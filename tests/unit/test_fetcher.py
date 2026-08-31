"""Unit tests for remote fetcher."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import requests

from neuromaps_prime import remote
from neuromaps_prime.fetcher import download_and_validate, id_storage
from neuromaps_prime.remote.github import GitHubFileMeta
from neuromaps_prime.remote.osf import OSFFileMeta

if TYPE_CHECKING:
    from pathlib import Path


def _osf_meta(payload: bytes, name: str = "file.surf.gii") -> OSFFileMeta:
    """Build an :class:`OSFFileMeta` whose MD5 matches ``payload``."""
    return OSFFileMeta(
        name=name,
        size=len(payload),
        extra={
            "hashes": {"md5": hashlib.md5(payload, usedforsecurity=False).hexdigest()}
        },
    )


def _github_meta(payload: bytes, name: str = "file.surf.gii") -> GitHubFileMeta:
    """Build a :class:`GitHubFileMeta` whose blob SHA matches ``payload``."""
    header = f"blob {len(payload)}\0".encode()
    sha = hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()
    return GitHubFileMeta(
        name=name, size=len(payload), sha=sha, download_url=f"https://raw/{name}"
    )


def _fake_stream(payload: bytes) -> MagicMock:
    """Build a fake ``requests.get`` serving ``payload`` for any url."""
    resp = MagicMock()
    resp.iter_content.return_value = [payload]
    return MagicMock(return_value=resp)


def _http_error(status: int) -> requests.HTTPError:
    """Build an HTTPError with ``status`` set and no Retry-After header.

    ``headers={}`` (not a bare MagicMock) stops ``_backoff`` from reading a
    mock header value; with None it takes the capped-backoff path.
    """
    return requests.HTTPError(response=MagicMock(status_code=status, headers={}))


class TestIDStorage:
    """Test suite for identifying storage location."""

    @pytest.mark.parametrize(
        ("storage", "expected"),
        [
            ("https://osf.io/project", "osf"),
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
            download_and_validate("https://google.com", tmp_path)

    @pytest.mark.parametrize(
        ("storage_cls", "mock_uri"),
        [
            (remote.OSFStorage, "https://files.osf.io/v1/resources/abcde"),
            (
                remote.GitHubStorage,
                "https://raw.githubusercontent.com/owner/repo/refs/tags/v1.0/file.txt",
            ),
        ],
    )
    def test_valid_calls_download(
        self, storage_cls: object, mock_uri: str, tmp_path: Path
    ) -> None:
        """Test valid uri dispatches to the storage backend."""
        with patch.object(storage_cls, "download") as mock_download:
            download_and_validate(mock_uri, tmp_path)
        mock_download.assert_called_once_with(mock_uri, tmp_path)

    def test_returns_stored_path(self, tmp_path: Path) -> None:
        """Test the stored path returned by the backend is passed through."""
        stored = tmp_path / "file.surf.gii"
        with patch.object(remote.OSFStorage, "download", return_value=stored):
            result = download_and_validate(
                "https://files.osf.io/v1/resources/abcde", tmp_path
            )
        assert result == stored


class TestDownloadAndValidateRetry:
    """download_and_validate retries transient errors, else fails fast."""

    def test_retries_transient_then_succeeds(self, tmp_path: Path) -> None:
        """A transient 429 followed by success returns the stored path."""
        stored = tmp_path / "file.surf.gii"
        transient = _http_error(429)
        with (
            patch.object(remote.OSFStorage, "download") as mock_download,
            patch("neuromaps_prime.fetcher.time.sleep"),
        ):
            mock_download.side_effect = [transient, stored]
            result = download_and_validate(
                "https://files.osf.io/v1/resources/abcde", tmp_path
            )
        assert result == stored
        assert mock_download.call_count == 2

    def test_no_retry_on_permanent_error(self, tmp_path: Path) -> None:
        """A non-retryable 404 propagates on the first attempt."""
        not_found = _http_error(404)
        with patch.object(remote.OSFStorage, "download") as mock_download:
            mock_download.side_effect = not_found
            with pytest.raises(requests.HTTPError):
                download_and_validate(
                    "https://files.osf.io/v1/resources/abcde", tmp_path
                )
        assert mock_download.call_count == 1

    def test_gives_up_after_max_attempts(self, tmp_path: Path) -> None:
        """Persistent 503s exhaust the retry budget and raise."""
        server_error = _http_error(503)
        with (
            patch.object(remote.OSFStorage, "download") as mock_download,
            patch("neuromaps_prime.fetcher.time.sleep"),
        ):
            mock_download.side_effect = server_error
            with pytest.raises(requests.HTTPError):
                download_and_validate(
                    "https://files.osf.io/v1/resources/abcde", tmp_path
                )
        assert mock_download.call_count == 5  # == fetcher._MAX_ATTEMPTS

    def test_retries_connection_error_then_succeeds(self, tmp_path: Path) -> None:
        """A dropped connection followed by success returns the stored path."""
        stored = tmp_path / "file.surf.gii"
        with (
            patch.object(remote.OSFStorage, "download") as mock_download,
            patch("neuromaps_prime.fetcher.time.sleep"),
        ):
            mock_download.side_effect = [requests.ConnectionError("dropped"), stored]
            result = download_and_validate(
                "https://files.osf.io/v1/resources/abcde", tmp_path
            )
        assert result == stored
        assert mock_download.call_count == 2

    def test_gives_up_after_max_connection_errors(self, tmp_path: Path) -> None:
        """Persistent connection errors exhaust the retry budget and raise."""
        with (
            patch.object(remote.OSFStorage, "download") as mock_download,
            patch("neuromaps_prime.fetcher.time.sleep"),
        ):
            mock_download.side_effect = requests.ConnectionError("dropped")
            with pytest.raises(requests.ConnectionError):
                download_and_validate(
                    "https://files.osf.io/v1/resources/abcde", tmp_path
                )
        assert mock_download.call_count == 5  # == fetcher._MAX_ATTEMPTS

    def test_no_retry_on_permanent_request_error(self, tmp_path: Path) -> None:
        """A permanent request error propagates on the first attempt."""
        bad_url = requests.exceptions.InvalidURL("not a valid url")
        with patch.object(remote.OSFStorage, "download") as mock_download:
            mock_download.side_effect = bad_url
            with pytest.raises(requests.exceptions.InvalidURL):
                download_and_validate(
                    "https://files.osf.io/v1/resources/abcde", tmp_path
                )
        assert mock_download.call_count == 1


class TestOSFDownload:
    """Test suite for OSFStorage.download naming and cache behavior."""

    def test_stores_under_storage_name(self, tmp_path: Path) -> None:
        """File lands in dest_dir under its exact storage-side name."""
        payload = b"surface data"
        with (
            patch.object(
                remote.OSFStorage, "get_meta", return_value=_osf_meta(payload)
            ),
            patch("neuromaps_prime.remote.osf.requests.get", _fake_stream(payload)),
        ):
            result = remote.OSFStorage().download("https://u1", tmp_path)
        assert result == tmp_path / "file.surf.gii"
        assert result.read_bytes() == payload

    def test_cache_hit_reuses_verified_file(self, tmp_path: Path) -> None:
        """A verifying existing file is returned without downloading."""
        payload = b"surface data"
        stored = tmp_path / "file.surf.gii"
        stored.write_bytes(payload)
        with (
            patch.object(
                remote.OSFStorage, "get_meta", return_value=_osf_meta(payload)
            ),
            patch("neuromaps_prime.remote.osf.requests.get") as mock_get,
        ):
            result = remote.OSFStorage().download("https://u1", tmp_path)
        assert result == stored
        mock_get.assert_not_called()

    def test_mismatch_warns_and_overwrites(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-matching existing file is overwritten, with a warning."""
        old, new = b"stale bytes", b"fresh bytes"
        stored = tmp_path / "file.surf.gii"
        stored.write_bytes(old)
        with (
            caplog.at_level(logging.WARNING),
            patch.object(remote.OSFStorage, "get_meta", return_value=_osf_meta(new)),
            patch("neuromaps_prime.remote.osf.requests.get", _fake_stream(new)),
        ):
            result = remote.OSFStorage().download("https://u1", tmp_path)
        assert result == stored
        assert stored.read_bytes() == new
        assert any(
            "already exists and is being overwritten" in record.getMessage()
            for record in caplog.records
        )

    def test_failed_stream_keeps_existing_file(self, tmp_path: Path) -> None:
        """A failed download stream keeps the existing file and leaves no partial."""
        payload = b"surface data"
        stored = tmp_path / "file.surf.gii"
        stored.write_bytes(payload)  # a previously-good cached copy
        resp = MagicMock()
        resp.iter_content.side_effect = requests.exceptions.ChunkedEncodingError(
            "stream truncated"
        )
        with (
            patch.object(
                remote.OSFStorage, "get_meta", return_value=_osf_meta(b"other")
            ),
            patch("neuromaps_prime.remote.osf.requests.get", return_value=resp),
            pytest.raises(requests.exceptions.ChunkedEncodingError),
        ):
            remote.OSFStorage().download("https://u1", tmp_path)
        assert stored.read_bytes() == payload
        assert list(tmp_path.glob("*.part")) == []


class TestGitHubDownload:
    """Test suite for GitHubStorage.download naming and cache behavior."""

    def test_stores_under_storage_name(self, tmp_path: Path) -> None:
        """File lands in dest_dir under its exact storage-side name."""
        payload = b"surface data"
        meta = _github_meta(payload)
        with (
            patch.object(remote.GitHubStorage, "get_meta", return_value=meta),
            patch("neuromaps_prime.remote.github.requests.get", _fake_stream(payload)),
        ):
            result = remote.GitHubStorage().download("https://u1", tmp_path)
        assert result == tmp_path / "file.surf.gii"
        assert result.read_bytes() == payload

    def test_cache_hit_reuses_verified_file(self, tmp_path: Path) -> None:
        """A verifying existing file is returned without downloading."""
        payload = b"surface data"
        meta = _github_meta(payload)
        stored = tmp_path / "file.surf.gii"
        stored.write_bytes(payload)
        with (
            patch.object(remote.GitHubStorage, "get_meta", return_value=meta),
            patch("neuromaps_prime.remote.github.requests.get") as mock_get,
        ):
            result = remote.GitHubStorage().download("https://u1", tmp_path)
        assert result == stored
        mock_get.assert_not_called()

    def test_mismatch_warns_and_overwrites(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-matching existing file is overwritten, with a warning."""
        old, new = b"stale bytes", b"fresh bytes"
        meta = _github_meta(new)
        stored = tmp_path / "file.surf.gii"
        stored.write_bytes(old)
        with (
            caplog.at_level(logging.WARNING),
            patch.object(remote.GitHubStorage, "get_meta", return_value=meta),
            patch("neuromaps_prime.remote.github.requests.get", _fake_stream(new)),
        ):
            result = remote.GitHubStorage().download("https://u1", tmp_path)
        assert result == stored
        assert stored.read_bytes() == new
        assert any(
            "already exists and is being overwritten" in record.getMessage()
            for record in caplog.records
        )

    def test_failed_stream_keeps_existing_file(self, tmp_path: Path) -> None:
        """A failed download stream keeps the existing file and leaves no partial."""
        payload = b"surface data"
        stored = tmp_path / "file.surf.gii"
        stored.write_bytes(payload)  # a previously-good cached copy
        resp = MagicMock()
        resp.iter_content.side_effect = requests.exceptions.ChunkedEncodingError(
            "stream truncated"
        )
        with (
            patch.object(
                remote.GitHubStorage, "get_meta", return_value=_github_meta(b"other")
            ),
            patch("neuromaps_prime.remote.github.requests.get", return_value=resp),
            pytest.raises(requests.exceptions.ChunkedEncodingError),
        ):
            remote.GitHubStorage().download("https://u1", tmp_path)
        assert stored.read_bytes() == payload
        assert list(tmp_path.glob("*.part")) == []
