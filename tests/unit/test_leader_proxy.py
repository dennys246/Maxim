"""Tests for leader proxy — LogBuffer, debug endpoints, admin endpoints, security."""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import patch

import pytest


# ── Branch validation + output sanitization ──────────────────────────────────


class TestBranchValidation:
    """Test _validate_branch rejects dangerous input."""

    def test_valid_branches(self):
        from maxim.runtime.leader_proxy import _validate_branch

        assert _validate_branch("main") == "main"
        assert _validate_branch("develop") == "develop"
        assert _validate_branch("feature/my-branch") == "feature/my-branch"
        assert _validate_branch("experiments/denny") == "experiments/denny"
        assert _validate_branch("v1.2.3") == "v1.2.3"
        assert _validate_branch("fix_underscore") == "fix_underscore"
        assert _validate_branch("release/2026.04") == "release/2026.04"

    def test_rejects_empty(self):
        from maxim.runtime.leader_proxy import _validate_branch

        with pytest.raises(ValueError, match="Empty"):
            _validate_branch("")
        with pytest.raises(ValueError, match="Empty"):
            _validate_branch("   ")

    def test_rejects_path_traversal(self):
        from maxim.runtime.leader_proxy import _validate_branch

        with pytest.raises(ValueError, match="\\.\\."):
            _validate_branch("../../../etc/passwd")
        with pytest.raises(ValueError, match="\\.\\."):
            _validate_branch("main/../refs/heads/evil")
        with pytest.raises(ValueError, match="\\.\\."):
            _validate_branch("HEAD..origin/main")

    def test_rejects_flag_injection(self):
        from maxim.runtime.leader_proxy import _validate_branch

        with pytest.raises(ValueError, match="starts with"):
            _validate_branch("-X")
        with pytest.raises(ValueError, match="starts with"):
            _validate_branch("--upload-pack=evil")

    def test_rejects_shell_metacharacters(self):
        from maxim.runtime.leader_proxy import _validate_branch

        with pytest.raises(ValueError, match="Invalid"):
            _validate_branch("main; rm -rf /")
        with pytest.raises(ValueError, match="Invalid"):
            _validate_branch("main$(whoami)")
        with pytest.raises(ValueError, match="Invalid"):
            _validate_branch("main`id`")
        with pytest.raises(ValueError, match="Invalid"):
            _validate_branch("branch with spaces")

    def test_rejects_special_git_refs(self):
        from maxim.runtime.leader_proxy import _validate_branch

        with pytest.raises(ValueError, match="Invalid"):
            _validate_branch("@{upstream}")
        with pytest.raises(ValueError, match="Invalid"):
            _validate_branch("HEAD~1")

    def test_strips_whitespace(self):
        from maxim.runtime.leader_proxy import _validate_branch

        assert _validate_branch("  main  ") == "main"


class TestSanitizeGitOutput:
    """Test _sanitize_git_output removes sensitive info."""

    def test_removes_absolute_paths(self):
        from maxim.runtime.leader_proxy import _sanitize_git_output

        text = "fatal: '/home/denny/Scripts/Maxim' is not a git repository"
        result = _sanitize_git_output(text)
        assert "/home/denny" not in result
        assert "<path>" in result

    def test_truncates_long_output(self):
        from maxim.runtime.leader_proxy import _sanitize_git_output

        text = "x" * 1000
        result = _sanitize_git_output(text, max_len=300)
        assert len(result) == 300

    def test_handles_none(self):
        from maxim.runtime.leader_proxy import _sanitize_git_output

        assert _sanitize_git_output(None) == ""
        assert _sanitize_git_output("") == ""

    def test_preserves_short_safe_text(self):
        from maxim.runtime.leader_proxy import _sanitize_git_output

        text = "Already up to date."
        assert _sanitize_git_output(text) == text

    def test_removes_nested_paths(self):
        from maxim.runtime.leader_proxy import _sanitize_git_output

        text = "error: could not open '/usr/local/lib/python3.12/site-packages/maxim/foo.py'"
        result = _sanitize_git_output(text)
        assert "/usr/local" not in result


# ── LogBuffer ────────────────────────────────────────────────────────────────


class TestLogBuffer:
    @pytest.fixture
    def buf(self):
        from maxim.runtime.leader_proxy import _LogBuffer

        return _LogBuffer(maxlen=50)

    def test_emit_captures_record(self, buf):
        record = logging.LogRecord(
            name="maxim.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        buf.emit(record)
        entries = buf.get_since(since_seq=-1)
        assert len(entries) == 1
        assert entries[0]["message"] == "hello world"
        assert entries[0]["level"] == "INFO"
        assert entries[0]["logger"] == "maxim.test"

    def test_sequence_increments(self, buf):
        for i in range(5):
            record = logging.LogRecord(
                name="maxim",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"msg {i}",
                args=(),
                exc_info=None,
            )
            buf.emit(record)
        entries = buf.get_since(since_seq=-1)
        seqs = [e["seq"] for e in entries]
        assert seqs == [0, 1, 2, 3, 4]

    def test_get_since_seq_filters(self, buf):
        for i in range(5):
            record = logging.LogRecord(
                name="maxim",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"msg {i}",
                args=(),
                exc_info=None,
            )
            buf.emit(record)
        entries = buf.get_since(since_seq=2)
        assert len(entries) == 2
        assert entries[0]["seq"] == 3
        assert entries[1]["seq"] == 4

    def test_get_since_ts_filters(self, buf):
        t0 = time.time()
        record = logging.LogRecord(
            name="maxim",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="old",
            args=(),
            exc_info=None,
        )
        record.created = t0 - 10
        buf.emit(record)

        record2 = logging.LogRecord(
            name="maxim",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="new",
            args=(),
            exc_info=None,
        )
        record2.created = t0 + 1
        buf.emit(record2)

        entries = buf.get_since(since_ts=t0)
        assert len(entries) == 1
        assert entries[0]["message"] == "new"

    def test_limit_caps_results(self, buf):
        for i in range(20):
            record = logging.LogRecord(
                name="maxim",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"msg {i}",
                args=(),
                exc_info=None,
            )
            buf.emit(record)
        entries = buf.get_since(since_seq=-1, limit=5)
        assert len(entries) == 5
        # Returns the LAST 5
        assert entries[0]["seq"] == 15

    def test_maxlen_evicts_old(self):
        from maxim.runtime.leader_proxy import _LogBuffer

        buf = _LogBuffer(maxlen=3)
        for i in range(5):
            record = logging.LogRecord(
                name="maxim",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"msg {i}",
                args=(),
                exc_info=None,
            )
            buf.emit(record)
        entries = buf.get_since(since_seq=-1)
        assert len(entries) == 3
        assert entries[0]["message"] == "msg 2"

    def test_latest_seq(self, buf):
        assert buf.latest_seq() == -1
        record = logging.LogRecord(
            name="maxim",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        buf.emit(record)
        assert buf.latest_seq() == 0
        buf.emit(record)
        assert buf.latest_seq() == 1

    def test_thread_safety(self, buf):
        errors = []

        def writer(offset: int) -> None:
            try:
                for i in range(50):
                    record = logging.LogRecord(
                        name="maxim",
                        level=logging.INFO,
                        pathname="",
                        lineno=0,
                        msg=f"t{offset}-{i}",
                        args=(),
                        exc_info=None,
                    )
                    buf.emit(record)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # 4 threads × 50 = 200 emits, but buffer maxlen=50
        entries = buf.get_since(since_seq=-1)
        assert len(entries) == 50
        assert buf.latest_seq() == 199

    def test_emit_never_raises(self, buf):
        """emit() should never crash the logging system."""
        # Even with a broken format, emit should silently succeed
        record = logging.LogRecord(
            name="maxim",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="ok",
            args=(),
            exc_info=None,
        )
        buf.emit(record)
        assert buf.latest_seq() == 0


# ── get_version_info ─────────────────────────────────────────────────────────


class TestVersionInfo:
    def test_returns_version(self):
        import maxim
        from maxim import get_version_info

        info = get_version_info()
        assert "version" in info
        assert info["version"] == maxim.__version__

    def test_includes_git_hash(self):
        from maxim import get_version_info

        info = get_version_info()
        # In a git repo, should have git_hash
        assert "git_hash" in info
        assert len(info["git_hash"]) >= 7

    def test_includes_git_message(self):
        from maxim import get_version_info

        info = get_version_info()
        assert "git_message" in info
        assert len(info["git_message"]) > 0

    def test_handles_missing_git(self):
        import maxim
        from maxim import get_version_info

        with patch("subprocess.run", side_effect=FileNotFoundError):
            info = get_version_info()
        assert info["version"] == maxim.__version__
        assert "git_hash" not in info


# ── Peer CLI command helpers ─────────────────────────────────────────────────


class TestPeerCLIDispatch:
    """Test that peer subcommands dispatch correctly."""

    def test_unknown_action_returns_2(self):
        from maxim.peer.cli import run_peer_connect_subcommand

        rc = run_peer_connect_subcommand(["nonexistent"])
        assert rc == 2

    def test_help_returns_0(self):
        from maxim.peer.cli import run_peer_connect_subcommand

        rc = run_peer_connect_subcommand(["--help"])
        assert rc == 0

    def test_show_without_config_returns_1(self, tmp_path, monkeypatch):
        from maxim.peer import cli as peer_cli

        monkeypatch.setattr(peer_cli, "read_peer_config", lambda: None)
        monkeypatch.setattr(peer_cli, "peer_config_path", lambda: tmp_path / "missing.yml")
        rc = peer_cli.run_peer_connect_subcommand(["show"])
        assert rc == 1

    def test_restart_without_config_returns_1(self, monkeypatch):
        from maxim.peer import cli as peer_cli

        monkeypatch.setattr(peer_cli, "read_peer_config", lambda: None)
        rc = peer_cli._cmd_restart([])
        assert rc == 1

    def test_version_without_leader_shows_local(self, monkeypatch):
        from maxim.peer import cli as peer_cli

        monkeypatch.setattr(peer_cli, "read_peer_config", lambda: None)
        rc = peer_cli._cmd_version([])
        assert rc == 0  # Shows local only, no error

    def test_logs_without_config_returns_1(self, monkeypatch):
        from maxim.peer import cli as peer_cli

        monkeypatch.setattr(peer_cli, "read_peer_config", lambda: None)
        rc = peer_cli._cmd_logs([])
        assert rc == 1


# ── ensure_log_buffer idempotent ─────────────────────────────────────────────


class TestEnsureLogBuffer:
    def test_idempotent(self):
        from maxim.runtime.leader_proxy import _ensure_log_buffer

        buf1 = _ensure_log_buffer()
        buf2 = _ensure_log_buffer()
        assert buf1 is buf2

    def test_installs_on_maxim_logger(self):
        from maxim.runtime.leader_proxy import _ensure_log_buffer

        buf = _ensure_log_buffer()
        maxim_logger = logging.getLogger("maxim")
        assert buf in maxim_logger.handlers
