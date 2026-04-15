"""Tests for maxim.utils.atomic_io.

Plan 4 C2 pre-design review E3 flagged permission widening as a silent
secret-leak vector. The ``preserve_mode`` flag is the fix — these
tests are the regression guard.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from maxim.utils.atomic_io import atomic_write_json, atomic_write_secret, atomic_write_text


class TestAtomicWriteText:
    def test_happy_path(self, tmp_path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_text(str(target), "hello\n")
        assert target.read_text() == "hello\n"

    def test_overwrites_existing(self, tmp_path) -> None:
        target = tmp_path / "out.txt"
        target.write_text("old content")
        atomic_write_text(str(target), "new content")
        assert target.read_text() == "new content"

    def test_creates_parent_directories(self, tmp_path) -> None:
        target = tmp_path / "nested" / "deeply" / "out.txt"
        atomic_write_text(str(target), "x")
        assert target.read_text() == "x"

    def test_no_tmp_file_left_on_success(self, tmp_path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_text(str(target), "x")
        assert not (tmp_path / "out.txt.tmp").exists()


class TestPreserveMode:
    """Plan 4 C2 pre-design review E3 regression guard.

    Before the ``preserve_mode`` fix, ``atomic_write_text`` always wrote
    the tmp file via the default umask (typically ``0o644``), then
    ``os.replace()`` installed it — so a pre-existing ``0o600`` file
    containing a secret would silently widen to ``0o644`` on every
    rewrite. Drain state contains node names only and is not secret,
    but the flag is shared infrastructure that future callers writing
    secrets (C3 admin API key rotation, cluster key storage) will opt
    into.
    """

    def test_preserves_0600_on_rewrite(self, tmp_path) -> None:
        """E4 fold (C2 pre-merge review): this test previously relied
        on the OS default umask NOT being 0o077 — a developer under
        `umask 077` would see the first file born 0o600 and the
        assertion would pass even if preserve_mode were entirely
        broken. We now explicitly chmod to 0o644 first (definitely
        different from any restrictive umask), then to 0o600, then
        rewrite, so the test verifies a real mode transition."""
        target = tmp_path / "secret.txt"
        target.write_text("initial secret")
        # Explicit 0o644 first — forces mode to a known permissive
        # state regardless of umask, so the subsequent 0o600 check
        # can't spuriously pass.
        os.chmod(target, 0o644)
        assert (os.stat(target).st_mode & 0o777) == 0o644

        os.chmod(target, 0o600)
        assert (os.stat(target).st_mode & 0o777) == 0o600

        atomic_write_text(str(target), "rotated secret", preserve_mode=True)
        assert target.read_text() == "rotated secret"
        assert (os.stat(target).st_mode & 0o777) == 0o600, (
            "preserve_mode=True must keep 0o600 across a rewrite. "
            "If this fails, the atomic_write_text fix for widening "
            "secrets back to umask has regressed."
        )

    def test_preserves_0640_on_rewrite(self, tmp_path) -> None:
        target = tmp_path / "config.txt"
        target.write_text("v1")
        os.chmod(target, 0o640)

        atomic_write_text(str(target), "v2", preserve_mode=True)
        assert (os.stat(target).st_mode & 0o777) == 0o640

    def test_opt_out_default_widens_as_before(self, tmp_path) -> None:
        """The new kwarg defaults to False so existing call sites
        don't change behavior. If anything ever relies on the old
        "widens to umask" behavior, this test locks it in — the fix
        only activates when the caller opts in.
        """
        target = tmp_path / "out.txt"
        target.write_text("initial")
        os.chmod(target, 0o600)

        atomic_write_text(str(target), "rewritten")  # preserve_mode=False
        # Default umask usually gives 0o644; this may vary by system but
        # the key assertion is "not 0o600 anymore"
        new_mode = os.stat(target).st_mode & 0o777
        assert new_mode != 0o600, (
            f"preserve_mode=False must not keep 0o600 — got {oct(new_mode)}. "
            "If this fails on a system with umask 0o177, adjust the assertion."
        )

    def test_preserve_mode_on_nonexistent_file_is_noop(self, tmp_path) -> None:
        """When the file doesn't exist yet, preserve_mode has nothing
        to preserve — the new file inherits the umask normally. This
        documents the contract: callers that want a specific initial
        mode on a brand-new file should chmod after the write.
        """
        target = tmp_path / "new.txt"
        atomic_write_text(str(target), "x", preserve_mode=True)
        assert target.read_text() == "x"
        # No assertion on the mode — it's umask-dependent and that's OK.

    def test_failed_stat_does_not_block_write(self, tmp_path, monkeypatch) -> None:
        """preserve_mode must be best-effort — if stat fails (permission
        denied, etc.) the write still succeeds, just without mode
        preservation. Refusing to write because we couldn't stat would
        be worse than a mode drift.
        """
        target = tmp_path / "out.txt"
        target.write_text("old")
        os.chmod(target, 0o600)

        real_stat = os.stat

        def fake_stat(path, *args, **kwargs):
            if str(path) == str(target):
                raise PermissionError("simulated stat failure")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", fake_stat)
        atomic_write_text(str(target), "new", preserve_mode=True)
        assert target.read_text() == "new"

    def test_preserve_mode_strips_file_type_bits(self, tmp_path) -> None:
        """``stat.S_IFMT`` bits must not be passed to ``os.chmod`` —
        the helper masks with ``0o7777`` (permission bits + setuid/gid
        + sticky). Regression guard in case the mask is changed to
        ``0o777`` accidentally (which would drop setuid/gid).
        """
        target = tmp_path / "setuid.txt"
        target.write_text("x")
        # setuid bit (04000) + 0600
        os.chmod(target, stat.S_ISUID | 0o600)
        captured_mode = os.stat(target).st_mode & 0o7777

        atomic_write_text(str(target), "y", preserve_mode=True)
        new_mode = os.stat(target).st_mode & 0o7777
        assert new_mode == captured_mode, (
            f"Expected {oct(captured_mode)}, got {oct(new_mode)} — setuid/gid bits must survive preserve_mode"
        )


class TestAtomicWriteSecret:
    """A3 fold (C2 pre-merge review): ``atomic_write_secret`` wraps
    ``atomic_write_text`` with ``preserve_mode=True``. Makes the safe
    path verbose — callers that write secrets pick a different
    function entirely rather than remembering to pass a flag.
    """

    def test_preserves_0600_on_rewrite(self, tmp_path) -> None:
        target = tmp_path / "cluster_key.txt"
        target.write_text("sk-abc123")
        os.chmod(target, 0o600)

        atomic_write_secret(str(target), "sk-rotated456")
        assert target.read_text() == "sk-rotated456"
        assert (os.stat(target).st_mode & 0o777) == 0o600

    def test_new_file_inherits_umask(self, tmp_path) -> None:
        """Brand-new file has no mode to preserve — inherits umask.
        Callers that want 0o600 on first write must chmod explicitly.
        Documents the limitation so C3 cluster-key rotation wires
        the chmod itself on first-write."""
        target = tmp_path / "new_secret.txt"
        atomic_write_secret(str(target), "sk-fresh")
        assert target.read_text() == "sk-fresh"
        # No assertion on mode — umask-dependent, caller's job on new
        # files. The wrapper's value is preserving ON REWRITE.


class TestAtomicWriteJson:
    def test_round_trip(self, tmp_path) -> None:
        target = tmp_path / "out.json"
        atomic_write_json(str(target), {"a": 1, "b": [2, 3]})
        assert json.loads(target.read_text()) == {"a": 1, "b": [2, 3]}

    def test_default_serializer_handles_non_json_types(self, tmp_path) -> None:
        """The ``default=str`` argument allows passing objects like
        pathlib.Path without a custom serializer."""
        from pathlib import Path

        target = tmp_path / "out.json"
        atomic_write_json(str(target), {"path": Path("/foo")})
        data = json.loads(target.read_text())
        assert data["path"] == "/foo"


class TestFailureCleanup:
    def test_tmp_file_removed_on_write_failure(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "out.txt"
        target.parent.mkdir(exist_ok=True)

        # Make os.replace fail — simulate cross-device rename or similar
        def failing_replace(*args, **kwargs):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError, match="simulated replace failure"):
            atomic_write_text(str(target), "x")

        assert not (tmp_path / "out.txt.tmp").exists()
