"""Tests for maxim.peer.drain_state (Plan 4 Stage C2).

These tests are the regression surface for the four CC2 findings C1
deferred:

1. Role detection timing — see ``TestRoleIsolation``
2. Read/write race — see ``TestConcurrency``
3. Orphan validation — see ``TestOrphanValidation``
4. Permission preservation — see ``TestPermissionPreservation``

Plus round-trip + happy path coverage.
"""

from __future__ import annotations

import multiprocessing
import os

import pytest

from maxim.peer.drain_state import (
    DrainError,
    DrainReadResult,
    drain_node,
    drain_state_path,
    read_drained_nodes,
    resume_node,
)


KNOWN = {"leader-desk", "mac-studio", "tablet"}


@pytest.fixture
def leader_home(tmp_path, monkeypatch):
    """Fresh MAXIM_DATA_HOME + MAXIM_ROLE=leader per test."""
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("MAXIM_ROLE", "leader")
    from maxim.utils import paths

    paths._reset_caches()
    yield tmp_path
    paths._reset_caches()


class TestHappyPath:
    def test_empty_drain_state_returns_empty_result(self, leader_home):
        result = read_drained_nodes(KNOWN)
        assert isinstance(result, DrainReadResult)
        assert result.drained == frozenset()
        assert result.orphans == frozenset()
        assert result.active == frozenset()

    def test_drain_then_read_round_trip(self, leader_home):
        drain_node("mac-studio", KNOWN)
        result = read_drained_nodes(KNOWN)
        assert result.drained == {"mac-studio"}
        assert result.orphans == frozenset()
        assert result.active == {"mac-studio"}

    def test_drain_multiple_then_read(self, leader_home):
        drain_node("mac-studio", KNOWN)
        drain_node("tablet", KNOWN)
        result = read_drained_nodes(KNOWN)
        assert result.drained == {"mac-studio", "tablet"}

    def test_drain_idempotent(self, leader_home):
        drain_node("mac-studio", KNOWN)
        drain_node("mac-studio", KNOWN)
        result = read_drained_nodes(KNOWN)
        assert result.drained == {"mac-studio"}

    def test_resume_removes_entry(self, leader_home):
        drain_node("mac-studio", KNOWN)
        drain_node("tablet", KNOWN)
        resume_node("mac-studio", KNOWN)
        result = read_drained_nodes(KNOWN)
        assert result.drained == {"tablet"}

    def test_resume_idempotent_for_not_drained(self, leader_home):
        resume_node("tablet", KNOWN)
        result = read_drained_nodes(KNOWN)
        assert result.drained == frozenset()

    def test_state_file_has_header(self, leader_home):
        drain_node("mac-studio", KNOWN)
        path = drain_state_path()
        content = path.read_text()
        assert content.startswith("# Maxim drain state")
        assert "mac-studio\n" in content

    def test_state_file_is_sorted(self, leader_home):
        drain_node("tablet", KNOWN)
        drain_node("leader-desk", KNOWN)
        drain_node("mac-studio", KNOWN)
        content = drain_state_path().read_text()
        data_lines = [line for line in content.splitlines() if line and not line.startswith("#")]
        assert data_lines == sorted(data_lines)


class TestOrphanValidation:
    """CC2 finding #3: drain must reject unknown names at write time;
    read must report orphans separately without filtering."""

    def test_unknown_node_at_drain_time_rejected(self, leader_home):
        with pytest.raises(DrainError) as exc:
            drain_node("ghost", KNOWN)
        assert "ghost" in str(exc.value)
        assert exc.value.known_nodes == sorted(KNOWN)

    def test_unknown_node_at_resume_time_rejected(self, leader_home):
        with pytest.raises(DrainError):
            resume_node("ghost", KNOWN)

    def test_orphans_surfaced_on_read(self, leader_home):
        """Operator drains a node, then edits mesh.yml to remove it.
        Next `list-nodes` must surface the orphan without hard-fail."""
        drain_node("mac-studio", KNOWN)
        # Simulate mesh.yml edit removing mac-studio
        reduced_known = KNOWN - {"mac-studio"}
        result = read_drained_nodes(reduced_known)
        assert result.drained == {"mac-studio"}
        assert result.orphans == {"mac-studio"}
        assert result.active == frozenset()

    def test_active_excludes_orphans(self, leader_home):
        drain_node("mac-studio", KNOWN)
        drain_node("tablet", KNOWN)
        reduced = KNOWN - {"mac-studio"}
        result = read_drained_nodes(reduced)
        assert result.drained == {"mac-studio", "tablet"}
        assert result.orphans == {"mac-studio"}
        assert result.active == {"tablet"}

    def test_read_without_known_set_reports_no_orphans(self, leader_home):
        """Callers that don't supply a mesh node set opt out of
        orphan detection and see every drained entry as active."""
        drain_node("mac-studio", KNOWN)
        result = read_drained_nodes(None)
        assert result.drained == {"mac-studio"}
        assert result.orphans == frozenset()


class TestRoleIsolation:
    """CC2 finding #1: role-scoped persistence means leader and peer
    drain state never collide on the same machine."""

    def test_leader_and_peer_have_distinct_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        from maxim.utils import paths

        paths._reset_caches()

        monkeypatch.setenv("MAXIM_ROLE", "leader")
        drain_node("mac-studio", KNOWN)

        monkeypatch.setenv("MAXIM_ROLE", "peer")
        result = read_drained_nodes(KNOWN)
        assert result.drained == frozenset()  # peer's file is separate

        monkeypatch.setenv("MAXIM_ROLE", "leader")
        result = read_drained_nodes(KNOWN)
        assert result.drained == {"mac-studio"}

        paths._reset_caches()

    def test_case_and_whitespace_normalized(self, tmp_path, monkeypatch):
        """Plan 2 R2a defines role as lowercase {leader,peer,solo}.
        Case variations and surrounding whitespace normalize to the
        canonical value; empty string falls back to leader.
        """
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        from maxim.utils import paths

        paths._reset_caches()

        for weird in ("LEADER", "  leader  ", "Leader"):
            monkeypatch.setenv("MAXIM_ROLE", weird)
            path = drain_state_path()
            assert path.name == "drained_nodes.leader.txt", f"role={weird!r} produced unexpected path {path.name!r}"

        # Empty string = env var absent = leader default (tests +
        # standalone scripts that skip detect_and_apply_role)
        monkeypatch.setenv("MAXIM_ROLE", "")
        assert drain_state_path().name == "drained_nodes.leader.txt"

        paths._reset_caches()

    def test_unknown_role_raises_drain_error(self, tmp_path, monkeypatch):
        """A1 fold (C2 pre-merge review): silent fallback is a band-aid.
        Unexpected non-empty MAXIM_ROLE values MUST raise DrainError
        with a clear message rather than silently writing to the
        leader bucket on a peer machine."""
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        from maxim.utils import paths

        paths._reset_caches()

        for bad in ("bogus", "LEADER_PEER", "client"):
            monkeypatch.setenv("MAXIM_ROLE", bad)
            with pytest.raises(DrainError, match="unrecognized MAXIM_ROLE"):
                drain_state_path()

        paths._reset_caches()

    def test_solo_role_has_its_own_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        from maxim.utils import paths

        paths._reset_caches()

        assert drain_state_path().name == "drained_nodes.solo.txt"
        paths._reset_caches()


class TestDrainStateIsNotSecret:
    """A3 fold (C2 pre-merge review): drain state is operator-visible
    topology, not a secret. It uses plain ``atomic_write_text`` (umask
    default), NOT ``atomic_write_secret``. Regression test documents
    the intent so a future reviewer doesn't "helpfully" add
    preserve_mode=True to the drain state writer.

    Credential-bearing files (C3 cluster key rotation, etc.) use
    :func:`maxim.utils.atomic_io.atomic_write_secret` directly —
    covered by `test_atomic_io.py::TestPreserveMode`.
    """

    def test_drain_state_write_uses_plain_atomic_write_text(self, leader_home, monkeypatch):
        """Assert the write path calls atomic_write_text WITHOUT
        preserve_mode set (or explicitly False). Catches a future
        regression that over-advertises the secret-handling flag."""
        import maxim.peer.drain_state as ds

        captured_kwargs = []

        def _spy(path, content, **kwargs):
            captured_kwargs.append(kwargs)
            # Delegate to the real write to keep the happy path working
            from maxim.utils.atomic_io import atomic_write_text as _real

            _real(path, content, **kwargs)

        monkeypatch.setattr(ds, "atomic_write_text", _spy)

        drain_node("mac-studio", KNOWN)
        assert len(captured_kwargs) == 1
        # No preserve_mode, or explicitly False
        assert captured_kwargs[0].get("preserve_mode", False) is False


def _concurrent_drain_worker(args):
    """Top-level so multiprocessing.Pool can pickle it."""
    data_home, node_name = args
    os.environ["MAXIM_DATA_HOME"] = data_home
    os.environ["MAXIM_ROLE"] = "leader"

    # Re-import inside the subprocess to pick up the fresh env.
    from maxim.utils import paths

    paths._reset_caches()
    from maxim.peer.drain_state import drain_node as _drain_node

    known = {f"node-{i}" for i in range(20)}
    _drain_node(node_name, known)


class TestConcurrency:
    """CC2 finding #2: concurrent drain operations must not lose
    entries due to RMW races. The ``filelock.FileLock`` around the
    read-modify-write cycle is the fix; this test is the regression
    guard.

    We fire 10 parallel ``drain`` calls via ``multiprocessing.Pool``
    and assert the final state contains all 10 names. Pre-fix
    behavior (no lock) would silently drop most of them.
    """

    def test_ten_parallel_drains_all_land(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        from maxim.utils import paths

        paths._reset_caches()

        nodes = [f"node-{i}" for i in range(10)]
        args = [(str(tmp_path), n) for n in nodes]

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=4) as pool:
            pool.map(_concurrent_drain_worker, args)

        known = {f"node-{i}" for i in range(20)}
        result = read_drained_nodes(known)
        assert result.drained == set(nodes), (
            f"Expected all 10 nodes drained, got {sorted(result.drained)}. "
            "This is the CC2 finding #2 regression — the filelock around "
            "the RMW cycle is not serializing concurrent writes."
        )

        paths._reset_caches()


class TestStateFileFormat:
    """Regression guards for the serialized format so future readers
    (operators editing by hand, or future versions of this module)
    can rely on the shape."""

    def test_header_lines_ignored_on_read(self, leader_home):
        drain_node("mac-studio", KNOWN)
        path = drain_state_path()
        # Add more comment lines by hand
        content = path.read_text()
        path.write_text(content + "# another comment\n# yet another\n")
        result = read_drained_nodes(KNOWN)
        assert result.drained == {"mac-studio"}

    def test_blank_lines_ignored(self, leader_home):
        path = drain_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n\n# header\n\nmac-studio\n\ntablet\n\n")
        result = read_drained_nodes(KNOWN)
        assert result.drained == {"mac-studio", "tablet"}

    def test_hand_edited_unsorted_file_still_works(self, leader_home):
        """Operators may edit the file by hand in a different order.
        Reads must still produce a set-equal result."""
        path = drain_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tablet\nmac-studio\nleader-desk\n")
        result = read_drained_nodes(KNOWN)
        assert result.drained == {"tablet", "mac-studio", "leader-desk"}

    def test_inline_comment_stripped_from_hand_edit(self, leader_home):
        """E2 fold (C2 pre-merge review): the original reader only
        stripped ``#``-prefixed lines and treated
        ``mac-studio  # needs rebuild`` as a literal node name that
        silently became an orphan. The fix splits on ``#`` first."""
        path = drain_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Maxim drain state\nmac-studio  # needs rebuild\ntablet\nleader-desk#note-without-space\n")
        result = read_drained_nodes(KNOWN)
        # All three entries parse to their pure names regardless of
        # inline comment position (with or without preceding space).
        assert result.drained == {"mac-studio", "tablet", "leader-desk"}

    def test_inline_comment_only_line_skipped(self, leader_home):
        """Lines that are nothing but a comment after stripping
        should not produce an empty node name."""
        path = drain_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("  # just a comment\nmac-studio\n")
        result = read_drained_nodes(KNOWN)
        assert result.drained == {"mac-studio"}


class TestSelfDrainStateLayer:
    """A2 fold (C2 pre-merge review): self-drain guard lives in the
    state layer as well as the CLI layer, so any future direct caller
    (C3 admin API, test fixtures, recovery scripts) gets the safety
    check without having to re-implement it.
    """

    def test_drain_self_without_force_raises(self, leader_home):
        with pytest.raises(DrainError, match="refusing to drain self"):
            drain_node("leader-desk", KNOWN, self_name="leader-desk")
        # State should be untouched
        result = read_drained_nodes(KNOWN)
        assert result.drained == frozenset()

    def test_drain_self_with_force_succeeds(self, leader_home):
        result = drain_node(
            "leader-desk",
            KNOWN,
            self_name="leader-desk",
            force_self=True,
        )
        assert "leader-desk" in result

    def test_drain_non_self_with_self_name_still_works(self, leader_home):
        """Passing self_name= shouldn't affect draining a different
        node — only the specific self-match case is guarded."""
        result = drain_node("mac-studio", KNOWN, self_name="leader-desk")
        assert "mac-studio" in result
        assert "leader-desk" not in result

    def test_self_name_none_skips_guard(self, leader_home):
        """Callers that don't know the self name (legacy paths, tests
        that don't care) pass self_name=None and the guard is
        skipped entirely — the CLI layer is expected to have its own
        check in that case."""
        result = drain_node("leader-desk", KNOWN)
        assert "leader-desk" in result


class TestFileLockTimeout:
    """E3 fold (C2 pre-merge review): raw filelock.Timeout tracebacks
    become operator-readable DrainError messages."""

    def test_drain_raises_drain_error_on_lock_timeout(self, leader_home, monkeypatch):
        """Simulate a stuck lock by monkeypatching FileLock.acquire
        to raise Timeout."""
        from filelock import Timeout as _Timeout

        import maxim.peer.drain_state as ds

        def _raise_timeout(self, *args, **kwargs):
            raise _Timeout(str(self.lock_file))

        monkeypatch.setattr(ds.FileLock, "acquire", _raise_timeout)

        with pytest.raises(DrainError, match="drain state locked"):
            drain_node("mac-studio", KNOWN)
        with pytest.raises(DrainError, match="drain state locked"):
            resume_node("mac-studio", KNOWN)
        with pytest.raises(DrainError, match="drain state locked"):
            read_drained_nodes(KNOWN)


def _barrier_drain_worker(args):
    """Threading-test worker. Top-level for pickle safety (though
    threading doesn't need it, this keeps the shape consistent with
    _concurrent_drain_worker)."""
    barrier, node_name = args
    barrier.wait()  # force all threads to enter drain_node simultaneously
    from maxim.peer.drain_state import drain_node as _drain

    known = {f"node-{i}" for i in range(20)}
    _drain(node_name, known)


class TestThreadingConcurrency:
    """E9 fold (C2 pre-merge review): the multiprocessing-pool test
    is weak because spawn overhead (~100-300ms per worker) dwarfs
    the microsecond RMW window, so a broken (no-lock) implementation
    could pass 70%+ of the time. This test uses same-process threads
    + a Barrier to force all workers into the critical section
    simultaneously, which is also the scenario most likely in
    practice (one daemon with multiple internal callers, not
    multiple CLI invocations)."""

    def test_ten_threads_all_drains_land(self, leader_home):
        import threading

        nodes = [f"node-{i}" for i in range(10)]
        barrier = threading.Barrier(len(nodes))
        threads = []
        for name in nodes:
            t = threading.Thread(
                target=_barrier_drain_worker,
                args=((barrier, name),),
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        known = {f"node-{i}" for i in range(20)}
        result = read_drained_nodes(known)
        assert result.drained == set(nodes), (
            f"Expected all 10 nodes drained, got {sorted(result.drained)}. "
            "E9 threading regression — the filelock around the RMW cycle "
            "is not serializing concurrent in-process writes. This is the "
            "scenario a real multi-agent daemon would hit."
        )


class TestEmptyKnownSet:
    """E8 fold (C2 pre-merge review): empty mesh (no nodes) should
    opt out of orphan detection, same as known_node_names=None.
    Previously an empty set would report every drain entry as an
    orphan, which is never useful."""

    def test_empty_known_set_no_orphans(self, leader_home):
        # Seed some drain state
        drain_node("mac-studio", KNOWN)
        drain_node("tablet", KNOWN)

        result = read_drained_nodes(set())
        assert result.drained == {"mac-studio", "tablet"}
        assert result.orphans == frozenset()

    def test_none_known_set_no_orphans(self, leader_home):
        drain_node("mac-studio", KNOWN)
        result = read_drained_nodes(None)
        assert result.orphans == frozenset()
