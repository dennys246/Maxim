"""Regression tests for download_file atomicity.

Covers:
- URLError mid-download leaves no file (was: leaked partial at final path)
- KeyboardInterrupt mid-download leaves no file AND re-raises
- Size mismatch after download rejects and cleans up
- Successful download atomically renames .partial → final
- Stale .partial from a prior crash is removed before the new download starts
- expected_bytes=None skips size verification (legacy profile support)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import pytest

from maxim.models.download import download_file


def _make_target(tmp_path: Path, name: str = "test.gguf") -> Path:
    return tmp_path / "subdir" / name


class TestDownloadAtomicity:
    def test_urlerror_leaves_no_file(self, tmp_path):
        """URLError during urlretrieve must not leak a file at the final path
        or at .partial. Previously download_file left the partial file on
        disk after URLError, which would then pass profile_has_local_file
        and crash the spawner on load."""
        dest = _make_target(tmp_path)

        def _boom(*args, **kwargs):
            # Create the partial file first (simulating urlretrieve opening
            # the output file) then fail, so we can verify cleanup
            tmp_partial = Path(args[1])
            tmp_partial.write_bytes(b"partial content")
            raise URLError("network down")

        with patch("maxim.models.download.urlretrieve", side_effect=_boom):
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

        def _interrupted(*args, **kwargs):
            tmp_partial = Path(args[1])
            tmp_partial.write_bytes(b"halfway there")
            raise KeyboardInterrupt

        with patch("maxim.models.download.urlretrieve", side_effect=_interrupted):
            with pytest.raises(KeyboardInterrupt):
                download_file("https://example.invalid/model.gguf", dest)

        assert not dest.exists()
        assert not dest.with_suffix(dest.suffix + ".partial").exists()

    def test_size_mismatch_rejects_and_cleans_up(self, tmp_path):
        """A truncated download (wrong byte count) must be rejected and
        the partial file deleted. Otherwise future profile_has_local_file
        checks would pass on a corrupted GGUF."""
        dest = _make_target(tmp_path)

        def _write_wrong_size(*args, **kwargs):
            Path(args[1]).write_bytes(b"only 12 bytes")

        with patch("maxim.models.download.urlretrieve", side_effect=_write_wrong_size):
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

        def _write_correct(*args, **kwargs):
            Path(args[1]).write_bytes(payload)

        with patch("maxim.models.download.urlretrieve", side_effect=_write_correct):
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

        def _write_anything(*args, **kwargs):
            Path(args[1]).write_bytes(b"any size is fine")

        with patch("maxim.models.download.urlretrieve", side_effect=_write_anything):
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

        def _fresh_download(*args, **kwargs):
            tmp_partial = Path(args[1])
            # The stale file should already be gone by the time urlretrieve
            # is called — verify that inside the mock
            assert not tmp_partial.exists() or tmp_partial.stat().st_size == 0
            tmp_partial.write_bytes(b"fresh content")

        with patch("maxim.models.download.urlretrieve", side_effect=_fresh_download):
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
        without calling urlretrieve at all."""
        dest = _make_target(tmp_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"already downloaded")

        with patch("maxim.models.download.urlretrieve") as mock_urlretrieve:
            result = download_file("https://example.invalid/model.gguf", dest)

        assert result is True
        mock_urlretrieve.assert_not_called()
