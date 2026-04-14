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

    def test_unexpected_role_falls_back_to_leader(self, tmp_path, monkeypatch):
        """Plan 2 R2a defines role as lowercase {leader,peer,solo}.
        Case variations, whitespace, and empty string all fall back
        to leader (documented in the module)."""
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        from maxim.utils import paths

        paths._reset_caches()

        for weird in ("LEADER", "  leader  ", "", "bogus"):
            monkeypatch.setenv("MAXIM_ROLE", weird)
            # All should route to the same file (the "leader" fallback)
            path = drain_state_path()
            assert path.name == "drained_nodes.leader.txt", f"role={weird!r} produced unexpected path {path.name!r}"

        paths._reset_caches()

    def test_solo_role_has_its_own_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        from maxim.utils import paths

        paths._reset_caches()

        assert drain_state_path().name == "drained_nodes.solo.txt"
        paths._reset_caches()


class TestPermissionPreservation:
    """CC2 finding #4: drain state writes must preserve the file mode
    bits across round-trips. The E3 fix in ``atomic_write_text``
    gets exercised here."""

    def test_pre_existing_mode_survives_drain_rewrite(self, leader_home):
        drain_node("mac-studio", KNOWN)
        path = drain_state_path()
        os.chmod(path, 0o600)
        drain_node("tablet", KNOWN)  # triggers a rewrite
        assert (os.stat(path).st_mode & 0o777) == 0o600

    def test_pre_existing_mode_survives_resume_rewrite(self, leader_home):
        drain_node("mac-studio", KNOWN)
        drain_node("tablet", KNOWN)
        path = drain_state_path()
        os.chmod(path, 0o640)
        resume_node("tablet", KNOWN)
        assert (os.stat(path).st_mode & 0o777) == 0o640


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
