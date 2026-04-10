"""Unit tests for the stale-process reaper.

The reaper's real job depends on live processes, which we can't safely
spawn in unit tests. We cover:

- Heuristic filtering (`_looks_like_maxim_sim`) — pure function, easy.
- Age parsing from ps etime format — pure function, multiple cases.
- Public ``find_orphans`` returns a list (possibly empty) without raising,
  even when both psutil and ps paths fail.
- ``warn_about_orphans`` is a no-op on empty input.
- ``reap_orphans`` returns 0 on empty input without touching any PIDs.
"""

from __future__ import annotations

from unittest.mock import patch

from maxim.runtime.orphan_reaper import (
    OrphanProcess,
    _looks_like_maxim_sim,
    _parse_ps_etime,
    find_orphans,
    reap_orphans,
    warn_about_orphans,
)


class TestLooksLikeMaximSim:
    def test_matches_direct_sim_invocation(self):
        assert _looks_like_maxim_sim("/usr/bin/python3 /opt/maxim/bin/maxim --sim 'cyberpunk heist'") is True

    def test_matches_module_form(self):
        assert _looks_like_maxim_sim("python -m maxim --sim test --display bio") is True

    def test_rejects_peer_subcommand(self):
        assert _looks_like_maxim_sim("python /opt/maxim/bin/maxim peer update") is False

    def test_rejects_doctor_subcommand(self):
        assert _looks_like_maxim_sim("python /opt/maxim/bin/maxim doctor --retry") is False

    def test_rejects_unrelated_python(self):
        assert _looks_like_maxim_sim("python train.py --epochs 10") is False

    def test_rejects_shell_grep(self):
        assert _looks_like_maxim_sim("grep maxim --sim log.txt") is False


class TestParsePsEtime:
    def test_mm_ss(self):
        assert _parse_ps_etime("05:30") == 5 * 60 + 30

    def test_hh_mm_ss(self):
        assert _parse_ps_etime("02:15:00") == 2 * 3600 + 15 * 60

    def test_dd_hh_mm_ss(self):
        assert _parse_ps_etime("1-02:15:00") == 86400 + 2 * 3600 + 15 * 60

    def test_garbage_returns_zero(self):
        assert _parse_ps_etime("not-a-time") == 0.0

    def test_empty_returns_zero(self):
        assert _parse_ps_etime("") == 0.0


class TestFindOrphans:
    def test_returns_list_on_both_paths_failing(self):
        """Both psutil and ps failing should return [] rather than raise."""
        with (
            patch("maxim.runtime.orphan_reaper._find_orphans_psutil", side_effect=Exception("no psutil")),
            patch("maxim.runtime.orphan_reaper._find_orphans_ps", side_effect=Exception("no ps")),
        ):
            result = find_orphans()
        assert result == []

    def test_uses_psutil_when_it_returns_results(self):
        fake = [OrphanProcess(pid=999, age_seconds=120.0, cmdline="python maxim --sim test")]
        with patch("maxim.runtime.orphan_reaper._find_orphans_psutil", return_value=fake):
            result = find_orphans()
        assert result == fake

    def test_falls_back_to_ps_when_psutil_empty(self):
        fake = [OrphanProcess(pid=111, age_seconds=60.0, cmdline="python maxim --sim other")]
        with (
            patch("maxim.runtime.orphan_reaper._find_orphans_psutil", return_value=[]),
            patch("maxim.runtime.orphan_reaper._find_orphans_ps", return_value=fake),
        ):
            result = find_orphans()
        assert result == fake


class TestWarnAboutOrphans:
    def test_empty_list_is_silent(self, capsys):
        warn_about_orphans([])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_non_empty_prints_count_and_pids(self, capsys):
        orphans = [
            OrphanProcess(pid=1234, age_seconds=300.0, cmdline="python maxim --sim test"),
            OrphanProcess(pid=5678, age_seconds=60.0, cmdline="python maxim --sim other"),
        ]
        warn_about_orphans(orphans)
        captured = capsys.readouterr()
        assert "2 stale maxim sim process" in captured.err
        assert "1234" in captured.err
        assert "5678" in captured.err
        assert "--reap-orphans" in captured.err


class TestReapOrphans:
    def test_empty_list_returns_zero(self):
        assert reap_orphans([]) == 0

    def test_dead_processes_count_as_killed(self):
        """SIGTERM to a non-existent PID should be handled gracefully."""
        # Use a bogus PID guaranteed not to exist
        orphans = [OrphanProcess(pid=999999, age_seconds=0.0, cmdline="python maxim --sim x")]
        killed = reap_orphans(orphans, grace_seconds=0.0)
        # ProcessLookupError during probe => counted as killed
        assert killed == 1
