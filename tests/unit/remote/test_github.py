"""Exercise direct raw downloads, branch updates, and cache recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import requests

from neuromaps_prime.remote.github import GitHubStorage

if TYPE_CHECKING:
    from pathlib import Path

URL = "https://raw.githubusercontent.com/owner/repo/main/data/test.txt"


def _response(
    payload: bytes = b"", status: int = 200, etag: str | None = '"v1"'
) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    response.headers = {"Content-Length": str(len(payload))}
    if etag is not None:
        response.headers["ETag"] = etag
    response.iter_content.return_value = [payload]
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    return response


class TestGitHubStorage:
    """Verify requests never leave the supplied raw URL."""

    @pytest.mark.parametrize("prefix", ["", "refs/tags/", "refs/heads/"])
    def test_parse_raw_url(self, prefix: str) -> None:
        """Branch, tag, and existing explicit-ref raw URLs remain supported."""
        assert GitHubStorage._parse(
            f"https://raw.githubusercontent.com/owner/repo/{prefix}v1/data/test.txt"
        ) == ("owner", "repo", "v1", "data/test.txt")

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/owner/repo/blob/main/file.txt",
            "https://example.com/raw.githubusercontent.com/owner/repo/main/file.txt",
            "https://raw.githubusercontent.com.evil/owner/repo/main/file.txt",
        ],
    )
    def test_unsupported_url(self, url: str) -> None:
        """Unsupported hosts and file-view pages are rejected before any request."""
        with pytest.raises(ValueError, match="unsupported"):
            GitHubStorage._parse(url)

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_direct_download(self, get: MagicMock, tmp_path: Path) -> None:
        """A cold fetch makes one raw GET and stores the complete bytes."""
        response = _response(b"original")
        get.return_value = response
        target = GitHubStorage().download(URL, tmp_path)
        assert target == tmp_path / "test.txt"
        assert target.read_bytes() == b"original"
        get.assert_called_once_with(
            URL, headers={"Accept-Encoding": "identity"}, stream=True, timeout=90
        )
        response.close.assert_called_once()

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_unchanged_cache(self, get: MagicMock, tmp_path: Path) -> None:
        """A 304 validates the cache without downloading the body again."""
        unchanged = _response(status=304)
        get.side_effect = [_response(b"original"), unchanged]
        storage = GitHubStorage()
        target = storage.download(URL, tmp_path)
        assert storage.download(URL, tmp_path) == target
        assert target.read_bytes() == b"original"
        assert get.call_args.kwargs["headers"]["If-None-Match"] == '"v1"'
        unchanged.iter_content.assert_not_called()
        assert all(call.args[0] == URL for call in get.call_args_list)

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_branch_update(self, get: MagicMock, tmp_path: Path) -> None:
        """A new ETag and body replace an older cached branch version."""
        get.side_effect = [_response(b"old"), _response(b"updated", etag='"v2"')]
        storage = GitHubStorage()
        target = storage.download(URL, tmp_path)
        storage.download(URL, tmp_path)
        assert target.read_bytes() == b"updated"
        assert get.call_args.kwargs["headers"]["If-None-Match"] == '"v1"'

    @pytest.mark.parametrize("corrupt", [b"tampered", b"short"])
    @patch("neuromaps_prime.remote.github.requests.get")
    def test_corrupt_cache_recovers(
        self, get: MagicMock, corrupt: bytes, tmp_path: Path
    ) -> None:
        """Corrupt local bytes force an unconditional raw download."""
        get.side_effect = [_response(b"original"), _response(b"original")]
        storage = GitHubStorage()
        target = storage.download(URL, tmp_path)
        target.write_bytes(corrupt)
        storage.download(URL, tmp_path)
        assert target.read_bytes() == b"original"
        assert "If-None-Match" not in get.call_args.kwargs["headers"]

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_changed_uri_does_not_reuse_validator(
        self, get: MagicMock, tmp_path: Path
    ) -> None:
        """A different URI with the same basename is fetched unconditionally."""
        get.side_effect = [_response(b"main"), _response(b"other")]
        storage = GitHubStorage()
        storage.download(URL, tmp_path)
        storage.download(URL.replace("/main/", "/other/"), tmp_path)
        assert (tmp_path / "test.txt").read_bytes() == b"other"
        assert "If-None-Match" not in get.call_args.kwargs["headers"]

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_old_cache_without_metadata_is_refreshed(
        self, get: MagicMock, tmp_path: Path
    ) -> None:
        """Existing pre-change caches migrate through a complete raw request."""
        (tmp_path / "test.txt").write_bytes(b"legacy")
        get.return_value = _response(b"current")
        GitHubStorage().download(URL, tmp_path)
        assert (tmp_path / "test.txt").read_bytes() == b"current"
        assert "If-None-Match" not in get.call_args.kwargs["headers"]

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_invalid_cache_metadata_is_ignored(
        self, get: MagicMock, tmp_path: Path
    ) -> None:
        """Broken sidecar metadata never prevents an unconditional refresh."""
        (tmp_path / "test.txt").write_bytes(b"old")
        (tmp_path / "test.txt.http.json").write_text("not JSON")
        get.return_value = _response(b"current")
        GitHubStorage().download(URL, tmp_path)
        assert (tmp_path / "test.txt").read_bytes() == b"current"

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_last_modified_fallback(self, get: MagicMock, tmp_path: Path) -> None:
        """Last-Modified can revalidate servers without an ETag."""
        first = _response(b"original", etag=None)
        first.headers["Last-Modified"] = "Mon, 12 Oct 2026 12:00:00 GMT"
        get.side_effect = [first, _response(status=304)]
        storage = GitHubStorage()
        storage.download(URL, tmp_path)
        storage.download(URL, tmp_path)
        assert (
            get.call_args.kwargs["headers"]["If-Modified-Since"]
            == (first.headers["Last-Modified"])
        )

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_failed_stream_preserves_cache(
        self, get: MagicMock, tmp_path: Path
    ) -> None:
        """Interrupted transfers preserve the old complete file and remove partials."""
        failing = _response(b"new")
        failing.iter_content.side_effect = requests.exceptions.ChunkedEncodingError(
            "stream truncated"
        )
        get.side_effect = [_response(b"original"), failing]
        storage = GitHubStorage()
        target = storage.download(URL, tmp_path)
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            storage.download(URL, tmp_path)
        assert target.read_bytes() == b"original"
        assert not list(tmp_path.glob("*.part"))

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_incomplete_response_preserves_cache(
        self, get: MagicMock, tmp_path: Path
    ) -> None:
        """An uncompressed short response does not overwrite good cached bytes."""
        incomplete = _response(b"short")
        incomplete.headers["Content-Length"] = "100"
        get.side_effect = [_response(b"original"), incomplete]
        storage = GitHubStorage()
        target = storage.download(URL, tmp_path)
        with pytest.raises(ValueError, match="Incomplete"):
            storage.download(URL, tmp_path)
        assert target.read_bytes() == b"original"
        assert not list(tmp_path.glob("*.part"))

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_http_error(self, get: MagicMock, tmp_path: Path) -> None:
        """HTTP errors propagate without producing a cached file."""
        get.return_value = _response(status=404)
        with pytest.raises(requests.HTTPError):
            GitHubStorage().download(URL, tmp_path)
        assert not (tmp_path / "test.txt").exists()

    @patch("neuromaps_prime.remote.github.requests.get")
    def test_304_without_cache_is_rejected(
        self, get: MagicMock, tmp_path: Path
    ) -> None:
        """An unsolicited 304 cannot be mistaken for a complete local file."""
        get.return_value = _response(status=304)
        with pytest.raises(ValueError, match="without a valid cached file"):
            GitHubStorage().download(URL, tmp_path)
