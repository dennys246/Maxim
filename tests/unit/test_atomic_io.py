"""Tests for maxim.utils.atomic_io.

Plan 4 C2 pre-design review E3 flagged permission widening as a silent
secret-leak vector. The ``preserve_mode`` flag is the fix — these
tests are the regression guard.
"""

from __future__ import annotations

import json
import os
import stat
import threading

import pytest

from maxim.utils.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_secret,
    atomic_write_text,
)


class TestAtomicWriteBytes:
    """The canonical BYTES writer (1.2 P2P Slice B — bundle blobs on disk)."""

    def test_round_trips_binary(self, tmp_path) -> None:
        target = tmp_path / "blob.bin"
        payload = bytes(range(256)) * 8  # includes NULs and high bytes
        atomic_write_bytes(str(target), payload)
        assert target.read_bytes() == payload

    def test_overwrites_existing(self, tmp_path) -> None:
        target = tmp_path / "blob.bin"
        target.write_bytes(b"old")
        atomic_write_bytes(str(target), b"\x00\x01\x02")
        assert target.read_bytes() == b"\x00\x01\x02"

    def test_creates_parent_directories(self, tmp_path) -> None:
        target = tmp_path / "nested" / "deep" / "blob.bin"
        atomic_write_bytes(str(target), b"x")
        assert target.read_bytes() == b"x"

    def test_no_tmp_file_left_on_success(self, tmp_path) -> None:
        target = tmp_path / "blob.bin"
        atomic_write_bytes(str(target), b"x")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["blob.bin"]

    def test_initial_mode_is_applied(self, tmp_path) -> None:
        if os.name == "nt":  # pragma: no cover — POSIX perms only
            pytest.skip("POSIX mode bits")
        target = tmp_path / "blob.bin"
        atomic_write_bytes(str(target), b"secret-blob", initial_mode=0o600)
        assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


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
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_default_mode_is_the_umask_default_not_0600(self, tmp_path) -> None:
        """#1663: the unique temp is made by ``os.open``, not ``open()``; without ``initial_mode`` it must keep
        ``open()``'s 0o666-under-the-umask bits. A ``mkstemp``-style 0600 here would silently narrow every
        persisted file another user or group reads."""
        old_umask = os.umask(0o022)
        try:
            atomic_write_text(str(tmp_path / "out.txt"), "x")
            atomic_write_bytes(str(tmp_path / "out.bin"), b"x")
        finally:
            os.umask(old_umask)
        assert stat.S_IMODE(os.stat(tmp_path / "out.txt").st_mode) == 0o644
        assert stat.S_IMODE(os.stat(tmp_path / "out.bin").st_mode) == 0o644


_LONG_A, _SHORT_B = "A" * 50_000, "B" * 20_000


class TestConcurrentWritersToOnePath:
    """#1663: two writers of one path must not share a temp file.

    With the old fixed ``<path>.tmp`` both writers opened the same temp (``O_TRUNC``), so their bytes could
    interleave in it, and the first ``os.replace`` took it from under the second, which raised
    ``FileNotFoundError`` (seen in #1645's trial run 5 as a swallowed ``_persist_state_json`` failure). The
    interleaving is forced, not hoped for: each writer is held at its ``os.replace`` until both have written,
    which is exactly when the old code had one temp holding both writers' bytes.
    """

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize(
        "write, payloads, installed",
        [
            (atomic_write_text, (_LONG_A, _SHORT_B), (_LONG_A.encode(), _SHORT_B.encode())),
            (atomic_write_bytes, (_LONG_A.encode(), _SHORT_B.encode()), (_LONG_A.encode(), _SHORT_B.encode())),
            (
                atomic_write_json,
                ({"v": _LONG_A}, {"v": _SHORT_B}),
                tuple(json.dumps({"v": s}, indent=2).encode() for s in (_LONG_A, _SHORT_B)),
            ),
        ],
        ids=["text", "bytes", "json"],
    )
    def test_both_writers_install_a_whole_file(self, tmp_path, monkeypatch, write, payloads, installed) -> None:
        target = str(tmp_path / "state.json")
        barrier = threading.Barrier(2, timeout=10)
        real_replace = os.replace
        temps: list[str] = []

        def _held_replace(src, dst):
            temps.append(src)
            barrier.wait()  # both writers have written their temp before either installs it
            real_replace(src, dst)

        monkeypatch.setattr("maxim.utils.atomic_io.os.replace", _held_replace)
        errors: list[BaseException] = []

        def _writer(data):
            try:
                write(target, data)
            except BaseException as e:  # collected and asserted on below
                errors.append(e)

        threads = [threading.Thread(target=_writer, args=(d,)) for d in payloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        assert errors == [], f"a concurrent writer failed: {errors!r}"
        assert len(set(temps)) == 2, f"the two writers shared a temp file: {temps}"
        with open(target, "rb") as f:
            raw = f.read()
        assert raw in installed, "the installed file is not one writer's whole content (interleaved or truncated)"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"], "a temp file was left behind"


class TestTempNames:
    """The temp name is atomic_io's: sweeps ask ``orphan_temp_globs`` / ``orphan_temp_target`` (#1663)."""

    def test_temp_is_unique_hidden_and_beside_the_target(self, tmp_path, monkeypatch) -> None:
        from maxim.utils.atomic_io import orphan_temp_target

        seen: list[str] = []
        real_replace = os.replace

        def _spy(src, dst):
            seen.append(src)
            real_replace(src, dst)

        monkeypatch.setattr("maxim.utils.atomic_io.os.replace", _spy)
        target = tmp_path / "out.txt"
        atomic_write_text(str(target), "1")
        atomic_write_text(str(target), "2")
        assert len(set(seen)) == 2
        for src in seen:
            assert os.path.dirname(src) == str(tmp_path)
            assert os.path.basename(src).startswith(".")
            assert orphan_temp_target(os.path.basename(src)) == "out.txt"

    def test_globs_match_both_forms_and_target_recovers_the_name(self, tmp_path) -> None:
        from maxim.utils.atomic_io import orphan_temp_globs, orphan_temp_target

        unique = tmp_path / ".8100.42.json.0123456789ab.tmp"
        legacy = tmp_path / "8100.42.json.tmp"
        other = tmp_path / "8100.42.json"
        for p in (unique, legacy, other):
            p.write_text("{")
        matched = {p for pattern in orphan_temp_globs("8100.*.json") for p in tmp_path.glob(pattern)}
        assert matched == {unique, legacy}
        assert orphan_temp_target(unique.name) == "8100.42.json"
        assert orphan_temp_target(legacy.name) == "8100.42.json"
        assert orphan_temp_target("8100.42.json") is None
        assert orphan_temp_target(".8100.42.json.nothex000000.tmp") is None
        assert orphan_temp_target(".tmp") is None

    @pytest.mark.skipif(os.name == "nt", reason="atomic_write_json_at needs dir_fd and O_NOFOLLOW")
    def test_the_strict_writer_names_its_temp_the_same_way(self, tmp_path, monkeypatch) -> None:
        """#1663 review fold: ``atomic_write_json_at`` builds its temp through the same ``_temp_name``, so a sweep's
        ``orphan_temp_target`` recognizes it too. A hand-built name could drift from ``_TEMP_RE`` silently."""
        from maxim.utils.atomic_io import atomic_write_json_at, orphan_temp_target

        seen: list[str] = []
        real_replace = os.replace

        def _spy(src, dst, **kwargs):
            seen.append(src)
            real_replace(src, dst, **kwargs)

        monkeypatch.setattr("maxim.utils.atomic_io.os.replace", _spy)
        dir_fd = os.open(tmp_path, os.O_RDONLY)
        try:
            atomic_write_json_at(dir_fd, "report.json", {"a": 1})
        finally:
            os.close(dir_fd)
        (src,) = seen
        assert orphan_temp_target(src) == "report.json"


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

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_new_file_is_0600_even_during_the_write(self, tmp_path, monkeypatch) -> None:
        """Brand-new secret files are 0600 FROM CREATION (2026-09-04
        console-auth review fold, cross-confirmed by both lenses): the old
        contract — inherit the umask, caller chmods afterwards — left the
        credential world-readable in the .tmp window and again between
        os.replace and the chmod, behind an in-file comment claiming the
        window was already closed. The tmp mode is captured AT the replace
        boundary so the whole write path is pinned, not just the end state."""
        target = tmp_path / "new_secret.txt"
        old_umask = os.umask(0o022)  # the umask under which the leak reproduced
        try:
            seen: dict[str, int] = {}
            real_replace = os.replace

            def _spy(src: str, dst: str) -> None:
                seen["tmp_mode"] = os.stat(src).st_mode & 0o777
                real_replace(src, dst)

            monkeypatch.setattr("maxim.utils.atomic_io.os.replace", _spy)
            atomic_write_secret(str(target), "sk-fresh")
        finally:
            os.umask(old_umask)
        assert target.read_text() == "sk-fresh"
        assert seen["tmp_mode"] == 0o600  # never umask-wide, not even mid-write
        assert (os.stat(target).st_mode & 0o777) == 0o600


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

        assert list(tmp_path.iterdir()) == [], "the temp file was left behind"

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit], ids=["KeyboardInterrupt", "SystemExit"])
    @pytest.mark.parametrize(
        "write",
        [lambda p: atomic_write_text(p, "new"), lambda p: atomic_write_bytes(p, b"new")],
        ids=["text", "bytes"],
    )
    def test_an_interrupt_at_replace_leaves_no_temp(self, tmp_path, monkeypatch, interrupt, write) -> None:
        """#1663 review fold: every write's temp is new, so an orphan is never overwritten later. An interrupt
        (not an ``Exception``) at the replace must still clean it up, and still propagate unchanged."""
        target = tmp_path / "out.txt"
        target.write_text("old")

        def interrupted_replace(*args, **kwargs):
            raise interrupt()

        monkeypatch.setattr("maxim.utils.atomic_io.os.replace", interrupted_replace)
        with pytest.raises(interrupt):
            write(str(target))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"], "the interrupted write left its temp"
        assert target.read_text() == "old"


# ── The strict policy: fsync_strict / append_line_fsync (#1315 item 5) ─────


class TestStrictFsync:
    def test_without_full_fsync_it_is_a_plain_fsync(self, tmp_path, monkeypatch):
        import fcntl

        from maxim.utils import atomic_io

        monkeypatch.delattr(fcntl, "F_FULLFSYNC", raising=False)
        synced = []
        monkeypatch.setattr(atomic_io.os, "fsync", lambda fd: synced.append(fd))
        atomic_io.fsync_strict(5)
        assert synced == [5]

    def test_unsupported_full_fsync_falls_back_to_fsync_and_others_raise(self, monkeypatch):
        import errno
        import fcntl

        from maxim.utils import atomic_io

        monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
        failure = {"errno": errno.ENOTSUP}

        def refusing(fd, cmd, arg=0):
            raise OSError(failure["errno"], "injected")

        monkeypatch.setattr(fcntl, "fcntl", refusing)
        synced = []
        monkeypatch.setattr(atomic_io.os, "fsync", lambda fd: synced.append(fd))
        atomic_io.fsync_strict(7)
        assert synced == [7], "an unsupported F_FULLFSYNC gets a plain fsync"
        failure["errno"] = errno.EIO
        with pytest.raises(OSError):
            atomic_io.fsync_strict(7)
        assert synced == [7], "a real I/O error raises; it is never downgraded to fsync"

    @pytest.mark.timeout(10)
    def test_a_write_that_makes_no_progress_raises_instead_of_spinning(self, monkeypatch):
        from maxim.utils import atomic_io

        monkeypatch.setattr(atomic_io.os, "write", lambda fd, data: 0)
        with pytest.raises(OSError, match="wrote 0 bytes"):
            atomic_io.append_line_fsync(99, b"abc\n")
