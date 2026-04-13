"""Regression tests for download_file atomicity.

Post Plan 1 R1: the underlying transport switched from urlretrieve to
``maxim.utils.http.download_to_file``. Tests mock at the new layer but
exercise the same atomicity guarantees (partial cleanup on failure,
KeyboardInterrupt re-raise, size mismatch rejection, stale .partial
eviction).

Covers:
- HTTPError mid-download leaves no file
- KeyboardInterrupt mid-download leaves no file AND re-raises
- Size mismatch after download rejects and cleans up
- Successful download atomically renames .partial → final
- Stale .partial from a prior crash is removed before the new download starts
- expected_bytes=None skips size verification (legacy profile support)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from maxim.models.download import download_file
from maxim.utils import http as _http


def _make_target(tmp_path: Path, name: str = "test.gguf") -> Path:
    return tmp_path / "subdir" / name


class TestDownloadAtomicity:
    def test_http_error_leaves_no_file(self, tmp_path):
        """An HTTPError during streaming must not leak a file at the final
        path or at .partial. Previously download_file left the partial
        file on disk after a URLError, which would then pass
        profile_has_local_file and crash the spawner on load."""
        dest = _make_target(tmp_path)

        def _boom(url, dest_path, **kwargs):
            # Simulate a partial file being written before the failure
            Path(dest_path).write_bytes(b"partial content")
            raise _http.HTTPConnectionError("_external", fix_hint="network down")

        with patch("maxim.utils.http.download_to_file", side_effect=_boom):
            result = download_file("https://example.invalid/model.gguf", dest)

        assert result is False
        assert not dest.exists()
        assert not dest.with_suffix(dest.suffix + ".partial").exists()

    def test_keyboardinterrupt_cleans_up_and_reraises(self, tmp_path):
        """Ctrl+C mid-download must clean up AND re-raise KeyboardInterrupt
        so the caller's interrupt handling fires. Previously the partial
        file stayed on disk because generic Exception handler didn't catch
        KeyboardInterrupt."""
        dest = _make_target(tmp_path)

        def _interrupted(url, dest_path, **kwargs):
            Path(dest_path).write_bytes(b"halfway there")
            raise KeyboardInterrupt

        with patch("maxim.utils.http.download_to_file", side_effect=_interrupted):
            with pytest.raises(KeyboardInterrupt):
                download_file("https://example.invalid/model.gguf", dest)

        assert not dest.exists()
        assert not dest.with_suffix(dest.suffix + ".partial").exists()

    def test_size_mismatch_rejects_and_cleans_up(self, tmp_path):
        """A truncated download (wrong byte count) must be rejected and
        the partial file deleted. Otherwise future profile_has_local_file
        checks would pass on a corrupted GGUF."""
        dest = _make_target(tmp_path)

        def _write_wrong_size(url, dest_path, **kwargs):
            Path(dest_path).write_bytes(b"only 12 bytes")
            return len(b"only 12 bytes")

        with patch("maxim.utils.http.download_to_file", side_effect=_write_wrong_size):
            result = download_file(
                "https://example.invalid/model.gguf",
                dest,
                expected_bytes=1000,
            )

        assert result is False
        assert not dest.exists()
        assert not dest.with_suffix(dest.suffix + ".partial").exists()

    def test_successful_download_atomic_rename(self, tmp_path):
        """A successful download writes to .partial then os.replace()s to
        the final path. The final path must exist and the partial must not."""
        dest = _make_target(tmp_path)
        payload = b"x" * 1000

        def _write_correct(url, dest_path, **kwargs):
            Path(dest_path).write_bytes(payload)
            return len(payload)

        with patch("maxim.utils.http.download_to_file", side_effect=_write_correct):
            result = download_file(
                "https://example.invalid/model.gguf",
                dest,
                expected_bytes=len(payload),
            )

        assert result is True
        assert dest.exists()
        assert dest.stat().st_size == len(payload)
        assert not dest.with_suffix(dest.suffix + ".partial").exists()

    def test_expected_bytes_none_skips_verification(self, tmp_path):
        """Legacy profiles without verified upstream sizes have
        expected_bytes=None. The download path should skip the size check
        and accept any size."""
        dest = _make_target(tmp_path)

        def _write_anything(url, dest_path, **kwargs):
            Path(dest_path).write_bytes(b"any size is fine")
            return len(b"any size is fine")

        with patch("maxim.utils.http.download_to_file", side_effect=_write_anything):
            result = download_file(
                "https://example.invalid/model.gguf",
                dest,
                expected_bytes=None,
            )

        assert result is True
        assert dest.exists()

    def test_stale_partial_removed_before_new_download(self, tmp_path):
        """If a prior crashed run left a .partial file at the expected tmp
        path, the new download must clear it before starting rather than
        appending to it or refusing."""
        dest = _make_target(tmp_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        stale = dest.with_suffix(dest.suffix + ".partial")
        stale.write_bytes(b"stale garbage from crashed run")
        assert stale.exists()

        def _fresh_download(url, dest_path, **kwargs):
            tmp_partial = Path(dest_path)
            # The stale file should already be gone by the time the downloader
            # is called — verify that inside the mock.
            assert not tmp_partial.exists() or tmp_partial.stat().st_size == 0
            tmp_partial.write_bytes(b"fresh content")
            return len(b"fresh content")

        with patch("maxim.utils.http.download_to_file", side_effect=_fresh_download):
            result = download_file(
                "https://example.invalid/model.gguf",
                dest,
                expected_bytes=len(b"fresh content"),
            )

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"fresh content"
        assert not stale.exists()

    def test_already_exists_short_circuits(self, tmp_path):
        """If the final file already exists, download_file returns True
        without calling the downloader at all."""
        dest = _make_target(tmp_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"already downloaded")

        with patch("maxim.utils.http.download_to_file") as mock_dl:
            result = download_file("https://example.invalid/model.gguf", dest)

        assert result is True
        mock_dl.assert_not_called()
