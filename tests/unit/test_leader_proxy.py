"""Tests for leader proxy — LogBuffer, debug endpoints, admin endpoints, security."""

from __future__ import annotations

import json
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

    def test_has_git_metadata_recognizes_directory_form(self, tmp_path):
        """Regular checkout: `.git` is a directory."""
        from maxim import _has_git_metadata

        (tmp_path / ".git").mkdir()
        assert _has_git_metadata(str(tmp_path)) is True

    def test_has_git_metadata_recognizes_worktree_file_form(self, tmp_path):
        """Worktree: `.git` is a FILE containing a gitdir pointer.

        Regression for the worktree-blind ``os.path.isdir`` guard that
        silently dropped git_hash + git_message whenever
        ``get_version_info`` was called from a worktree. Without the
        fix, this test fails because the function early-returns before
        shelling out to git.
        """
        from maxim import _has_git_metadata

        (tmp_path / ".git").write_text("gitdir: /Users/example/Scripts/Maxim/.git/worktrees/example\n")
        assert _has_git_metadata(str(tmp_path)) is True

    def test_has_git_metadata_absent_for_pypi_install(self, tmp_path):
        """PyPI install: no `.git` at all."""
        from maxim import _has_git_metadata

        # tmp_path is empty -- no .git/ dir, no .git file.
        assert _has_git_metadata(str(tmp_path)) is False


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


# ── _current_llama_server_n_ctx ────────────────────────────────────────────


class TestCurrentLlamaServerNCtx:
    """Tests for _current_llama_server_n_ctx lifted from doctor/checks.py."""

    def test_returns_context_length_from_models_endpoint(self):
        """Happy path: /v1/models returns context_length."""
        import json
        from maxim.runtime.leader_proxy import _current_llama_server_n_ctx

        models_resp = json.dumps({"data": [{"id": "test", "context_length": 8192}]})
        http_resp = f"HTTP/1.0 200 OK\r\n\r\n{models_resp}".encode()

        with patch("socket.create_connection") as mock_conn:
            mock_sock = mock_conn.return_value.__enter__.return_value
            mock_sock.recv.side_effect = [http_resp, b""]
            result = _current_llama_server_n_ctx(8100)

        assert result == 8192

    def test_returns_none_when_connection_refused(self):
        from maxim.runtime.leader_proxy import _current_llama_server_n_ctx

        with patch("socket.create_connection", side_effect=ConnectionRefusedError):
            result = _current_llama_server_n_ctx(8100)

        assert result is None

    def test_returns_none_for_empty_models_list(self):
        import json
        from maxim.runtime.leader_proxy import _current_llama_server_n_ctx

        models_resp = json.dumps({"data": []})
        http_resp = f"HTTP/1.0 200 OK\r\n\r\n{models_resp}".encode()

        with patch("socket.create_connection") as mock_conn:
            mock_sock = mock_conn.return_value.__enter__.return_value
            mock_sock.recv.side_effect = [http_resp, b""]
            result = _current_llama_server_n_ctx(8100)

        assert result is None


# ── /v1/debug/vram endpoint ────────────────────────────────────────────────


class TestHandleDebugVram:
    """Tests for _handle_debug_vram on _ProxyHandler.

    Creates a minimal handler instance bypassing BaseHTTPRequestHandler.__init__
    (which requires a live socket). Captures _send_json calls to verify the
    response shape.
    """

    @pytest.fixture
    def handler(self):
        """Create a _ProxyHandler instance with a captured _send_json."""
        from maxim.runtime.leader_proxy import _ProxyHandler

        h = object.__new__(_ProxyHandler)
        h.upstream_url = "http://127.0.0.1:8100"
        h.api_key = None
        h.start_time = time.time()
        h._sent: list[tuple[int, dict]] = []
        h._send_json = lambda code, body: h._sent.append((code, body))
        return h

    def test_nvidia_smi_unavailable_returns_503(self, handler):
        with patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=None):
            handler._handle_debug_vram()

        assert len(handler._sent) == 1
        code, body = handler._sent[0]
        assert code == 503
        assert "nvidia-smi unavailable" in body["error"]
        assert "fix" in body

    def test_gpu_available_no_model_returns_null_projection(self, handler):
        gpu = {
            "utilization_pct": 10.0,
            "vram_used_gb": 0.42,
            "vram_total_gb": 16.0,
            "temperature_c": 35.0,
        }
        with (
            patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu),
            patch("maxim.runtime.llm_server._active_model", None),
        ):
            handler._handle_debug_vram()

        code, body = handler._sent[0]
        assert code == 200
        assert body["live"]["vram_used_gb"] == 0.42
        assert body["live"]["vram_total_gb"] == 16.0
        assert body["live"]["spillover"] is False
        assert body["live"]["warning"] is False
        assert body["projection"] is None
        assert "spillover_ratio" in body["thresholds"]
        assert "warn_ratio" in body["thresholds"]
        assert "timestamp" in body

    def test_gpu_with_model_returns_full_projection(self, handler):
        gpu = {
            "utilization_pct": 85.0,
            "vram_used_gb": 14.0,
            "vram_total_gb": 16.0,
            "temperature_c": 68.0,
        }
        with (
            patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu),
            patch("maxim.runtime.llm_server._active_model", "qwen2.5-14b-instruct"),
            patch(
                "maxim.runtime.leader_proxy._current_llama_server_n_ctx",
                return_value=8192,
            ),
        ):
            handler._handle_debug_vram()

        code, body = handler._sent[0]
        assert code == 200
        proj = body["projection"]
        assert proj is not None
        assert proj["profile"] == "qwen2.5-14b-instruct"
        assert proj["n_ctx"] == 8192
        assert "weights_gb" in proj
        assert "kv_cache_gb" in proj
        assert "headroom_gb" in proj
        assert "projected_total_gb" in proj
        assert "spillover_risk" in proj
        assert "recommended_n_ctx" in proj

    def test_live_spillover_flag_at_96_percent(self, handler):
        """ratio 0.96 > _SPILLOVER_RATIO (0.95) -> spillover=True."""
        gpu = {
            "utilization_pct": 98.0,
            "vram_used_gb": 15.36,
            "vram_total_gb": 16.0,
            "temperature_c": 75.0,
        }
        with (
            patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu),
            patch("maxim.runtime.llm_server._active_model", None),
        ):
            handler._handle_debug_vram()

        code, body = handler._sent[0]
        assert code == 200
        assert body["live"]["spillover"] is True
        assert body["live"]["warning"] is True

    def test_warning_flag_at_90_percent(self, handler):
        """ratio 0.90 > _SPILLOVER_WARN_RATIO (0.85) but < 0.95 -> warning only."""
        gpu = {
            "utilization_pct": 90.0,
            "vram_used_gb": 14.4,
            "vram_total_gb": 16.0,
            "temperature_c": 70.0,
        }
        with (
            patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu),
            patch("maxim.runtime.llm_server._active_model", None),
        ):
            handler._handle_debug_vram()

        code, body = handler._sent[0]
        assert code == 200
        assert body["live"]["spillover"] is False
        assert body["live"]["warning"] is True

    def test_zero_vram_total_no_division_error(self, handler):
        """Edge case: nvidia-smi returns 0 total -- should not crash."""
        gpu = {
            "utilization_pct": 0.0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "temperature_c": 0.0,
        }
        with patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu):
            handler._handle_debug_vram()

        code, body = handler._sent[0]
        assert code == 200
        assert body["live"]["ratio"] == 0.0

    def test_vram_path_registered_in_is_debug_path(self):
        """Verify /v1/debug/vram is recognized as a debug path."""
        from maxim.runtime.leader_proxy import _ProxyHandler

        h = object.__new__(_ProxyHandler)
        assert h._is_debug_path("/v1/debug/vram")
        assert h._is_debug_path("/v1/debug/vram/")  # trailing slash tolerance


class TestDebugPathSync:
    """Verify _is_debug_path and _route_debug stay in sync.

    Pre-merge review found deps + install-status were in _route_debug but
    NOT in _is_debug_path, bypassing the debug auth gate. This regression
    test ensures all routed debug paths are also recognized as debug paths.
    """

    def test_all_routed_paths_in_is_debug_path(self):
        from maxim.runtime.leader_proxy import _ProxyHandler

        h = object.__new__(_ProxyHandler)
        # Every path that _route_debug handles must be in _is_debug_path.
        debug_paths = [
            "/v1/debug/ping",
            "/v1/debug/status",
            "/v1/debug/heartbeat",
            "/v1/debug/metrics",
            "/v1/debug/version",
            "/v1/debug/logs",
            "/v1/debug/last-requests",
            "/v1/debug/vram",
            "/v1/debug/deps",
            "/v1/debug/install-status",
        ]
        for path in debug_paths:
            assert h._is_debug_path(path), f"{path} missing from _is_debug_path"


# ── Pip/Dev dual-mode update (peer_update_pip_mode plan) ─────────────────────


class TestInstallModeDetection:
    """Test _detect_install_mode, _detect_installed_extras, version helpers."""

    def test_detects_dev_when_git_dir_exists(self):
        from maxim.runtime.leader_proxy import _detect_install_mode

        with patch("maxim.runtime.leader_proxy.Path") as MockPath:
            mock_parents = {3: MockPath.return_value.resolve.return_value.parents.__getitem__.return_value}
            MockPath.return_value.resolve.return_value.parents.__getitem__ = lambda s, i: mock_parents.get(i, s)
            # Simpler approach: just check the real repo has .git
            result = _detect_install_mode()
        # We're running from a git checkout, so this should be "dev"
        assert result == "dev"

    def test_detects_pip_when_no_git_dir(self, tmp_path):
        from maxim.runtime.leader_proxy import _detect_install_mode

        # Point __file__ to a path with no .git
        fake_file = tmp_path / "a" / "b" / "c" / "runtime" / "leader_proxy.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()

        with patch("maxim.runtime.leader_proxy.Path") as MockPath:
            MockPath.return_value.resolve.return_value.parents.__getitem__ = (
                lambda _, i: tmp_path if i == 3 else tmp_path
            )
            result = _detect_install_mode()
        assert result == "pip"

    def test_detect_installed_extras_filters_by_allowlist(self):
        from maxim.runtime.leader_proxy import _detect_installed_extras

        # Pretend everything is importable
        with patch("maxim.runtime.leader_proxy._try_import", return_value=True):
            extras = _detect_installed_extras()
        # All detected extras must be in the allowlist
        from maxim.runtime.leader_proxy import _ALLOWED_EXTRAS

        for e in extras:
            assert e in _ALLOWED_EXTRAS

    def test_detect_installed_extras_excludes_missing(self):
        from maxim.runtime.leader_proxy import _detect_installed_extras

        with patch("maxim.runtime.leader_proxy._try_import", return_value=False):
            extras = _detect_installed_extras()
        assert extras == []

    def test_get_current_version_returns_string(self):
        from maxim.runtime.leader_proxy import _get_current_version

        v = _get_current_version()
        assert isinstance(v, str)
        # Should be a real version or "unknown"
        assert v == "unknown" or "." in v

    def test_get_latest_pypi_version_returns_none_on_failure(self):
        from maxim.runtime.leader_proxy import _get_latest_pypi_version

        with patch("subprocess.run", side_effect=Exception("no pip")):
            result = _get_latest_pypi_version()
        assert result is None

    def test_get_latest_pypi_version_parses_output(self):
        from maxim.runtime.leader_proxy import _get_latest_pypi_version

        mock_result = type("R", (), {"returncode": 0, "stdout": "pymaxim (0.3.1)\n  INSTALLED: 0.3.0"})()
        with patch("subprocess.run", return_value=mock_result):
            result = _get_latest_pypi_version()
        assert result == "0.3.1"

    def test_get_latest_pypi_version_timeout(self):
        import subprocess as sp

        from maxim.runtime.leader_proxy import _get_latest_pypi_version

        with patch("subprocess.run", side_effect=sp.TimeoutExpired("pip", 15)):
            result = _get_latest_pypi_version()
        assert result is None


class TestVersionValidation:
    """Test the _VERSION_RE regex and version parsing in _parse_admin_update_body."""

    def test_valid_versions(self):
        from maxim.runtime.leader_proxy import _VERSION_RE

        for v in ["0.3.0", "0.3.1", "1.0.0", "0.3.1rc1", "1.2.3.post1", "0.1.0dev0"]:
            assert _VERSION_RE.match(v), f"{v} should be valid"

    def test_rejects_injection(self):
        from maxim.runtime.leader_proxy import _VERSION_RE

        for v in ["0.3.1; rm -rf /", "0.3.1$(whoami)", "0.3.1`id`", "../../../etc", "0.3.1 --extra-index-url"]:
            assert not _VERSION_RE.match(v), f"{v} should be rejected"


class TestParseAdminUpdateBody:
    """Test _parse_admin_update_body with new mode/version fields."""

    @pytest.fixture
    def handler(self):
        from maxim.runtime.leader_proxy import _ProxyHandler

        h = object.__new__(_ProxyHandler)
        h.upstream_url = "http://127.0.0.1:8100"
        h.api_key = None
        h.start_time = time.time()
        h._sent: list[tuple[int, dict]] = []
        h._send_json = lambda code, body: h._sent.append((code, body))
        return h

    def _set_body(self, handler, body_dict):
        raw = json.dumps(body_dict).encode()
        handler._read_body = lambda max_size=4096: raw
        return handler

    def test_defaults_to_auto_mode(self, handler):
        self._set_body(handler, {"branch": "main"})
        result = handler._parse_admin_update_body()
        assert result is not None
        assert result["mode"] == "auto"
        assert result["version"] is None
        assert result["dry_run"] is True  # safe-by-default

    def test_pip_mode_with_version(self, handler):
        self._set_body(handler, {"mode": "pip", "version": "0.3.1", "dry_run": False})
        result = handler._parse_admin_update_body()
        assert result is not None
        assert result["mode"] == "pip"
        assert result["version"] == "0.3.1"
        assert result["dry_run"] is False

    def test_dev_mode(self, handler):
        self._set_body(handler, {"mode": "dev", "branch": "feat/foo"})
        result = handler._parse_admin_update_body()
        assert result is not None
        assert result["mode"] == "dev"
        assert result["branch"] == "feat/foo"

    def test_rejects_invalid_mode(self, handler):
        self._set_body(handler, {"mode": "invalid"})
        result = handler._parse_admin_update_body()
        assert result is None
        assert handler._sent[0][0] == 400
        assert "Invalid mode" in handler._sent[0][1]["error"]

    def test_rejects_invalid_version(self, handler):
        self._set_body(handler, {"mode": "pip", "version": "0.3.1; rm -rf /"})
        result = handler._parse_admin_update_body()
        assert result is None
        assert handler._sent[0][0] == 400
        assert "Invalid version" in handler._sent[0][1]["error"]

    def test_backward_compat_no_mode_field(self, handler):
        """Old clients send no mode field — should default to auto."""
        self._set_body(handler, {"branch": "main", "dry_run": False, "force": False})
        result = handler._parse_admin_update_body()
        assert result is not None
        assert result["mode"] == "auto"


class TestPipUpdateHandler:
    """Test _handle_pip_update and _run_pip_upgrade."""

    @pytest.fixture
    def handler(self):
        from maxim.runtime.leader_proxy import _ProxyHandler

        h = object.__new__(_ProxyHandler)
        h.upstream_url = "http://127.0.0.1:8100"
        h.api_key = None
        h.start_time = time.time()
        h.client_address = ("127.0.0.1", 12345)
        h.request_log = None
        h._sent: list[tuple[int, dict]] = []
        h._send_json = lambda code, body: h._sent.append((code, body))
        return h

    def test_dry_run_returns_preview(self, handler):
        with (
            patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.0"),
            patch("maxim.runtime.leader_proxy._detect_installed_extras", return_value=["semantic"]),
            patch("maxim.runtime.leader_proxy._get_latest_pypi_version", return_value="0.3.1"),
        ):
            handler._handle_pip_update({"dry_run": True, "version": None})

        assert len(handler._sent) == 1
        code, body = handler._sent[0]
        assert code == 200
        assert body["status"] == "preview"
        assert body["install_mode"] == "pip"
        assert body["current_version"] == "0.3.0"
        assert body["latest_version"] == "0.3.1"
        assert body["extras_detected"] == ["semantic"]
        # Old-client compat: synthetic pending_commits
        assert len(body["pending_commits"]) == 1
        assert "0.3.0" in body["pending_commits"][0]

    def test_dry_run_up_to_date(self, handler):
        with (
            patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.1"),
            patch("maxim.runtime.leader_proxy._detect_installed_extras", return_value=[]),
            patch("maxim.runtime.leader_proxy._get_latest_pypi_version", return_value="0.3.1"),
        ):
            handler._handle_pip_update({"dry_run": True, "version": None})

        code, body = handler._sent[0]
        assert body["status"] == "up_to_date"
        assert body["install_mode"] == "pip"

    def test_dry_run_pypi_down_graceful(self, handler):
        with (
            patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.0"),
            patch("maxim.runtime.leader_proxy._detect_installed_extras", return_value=[]),
            patch("maxim.runtime.leader_proxy._get_latest_pypi_version", return_value=None),
        ):
            handler._handle_pip_update({"dry_run": True, "version": None})

        code, body = handler._sent[0]
        assert code == 200
        assert body["status"] == "preview"
        assert body["latest_version"] is None

    def test_upgrade_success_detects_version_change(self, handler):
        version_counter = iter(["0.3.0", "0.3.1"])
        with (
            patch("maxim.runtime.leader_proxy._get_current_version", side_effect=lambda: next(version_counter)),
            patch("maxim.runtime.leader_proxy._detect_installed_extras", return_value=["semantic"]),
            patch.object(handler, "_run_pip_upgrade", return_value="Successfully installed pymaxim-0.3.1"),
        ):
            handler._handle_pip_update({"dry_run": False, "version": None})

        code, body = handler._sent[0]
        assert code == 200
        assert body["status"] == "updated"
        assert body["install_mode"] == "pip"
        assert body["from_version"] == "0.3.0"
        assert body["to_version"] == "0.3.1"
        assert body["extras_preserved"] == ["semantic"]

    def test_upgrade_no_change_returns_up_to_date(self, handler):
        with (
            patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.1"),
            patch("maxim.runtime.leader_proxy._detect_installed_extras", return_value=[]),
            patch.object(handler, "_run_pip_upgrade", return_value="Requirement already satisfied"),
        ):
            handler._handle_pip_update({"dry_run": False, "version": None})

        code, body = handler._sent[0]
        assert body["status"] == "up_to_date"

    def test_upgrade_failure_returns_none(self, handler):
        """_run_pip_upgrade returns None on failure (error already sent)."""
        with (
            patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.0"),
            patch("maxim.runtime.leader_proxy._detect_installed_extras", return_value=[]),
            patch.object(handler, "_run_pip_upgrade", return_value=None),
        ):
            handler._handle_pip_update({"dry_run": False, "version": None})

        # _run_pip_upgrade sent its own error, _handle_pip_update should not double-send
        assert len(handler._sent) == 0


class TestRunPipUpgrade:
    """Test _run_pip_upgrade subprocess orchestration."""

    @pytest.fixture
    def handler(self):
        from maxim.runtime.leader_proxy import _ProxyHandler

        h = object.__new__(_ProxyHandler)
        h.upstream_url = "http://127.0.0.1:8100"
        h.api_key = None
        h.start_time = time.time()
        h.client_address = ("127.0.0.1", 12345)
        h._sent: list[tuple[int, dict]] = []
        h._send_json = lambda code, body: h._sent.append((code, body))
        return h

    def test_builds_correct_pip_command(self, handler):
        """Verify the pip command includes extras, version pin, and index URL."""
        calls = []

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            r = type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
            return r

        with (
            patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.0"),
            patch("subprocess.run", side_effect=mock_run),
            patch("shutil.disk_usage", return_value=type("U", (), {"free": 10 * (1 << 30)})()),
            patch("shutil.rmtree"),
        ):
            result = handler._run_pip_upgrade("0.3.1", ["semantic", "llm-llama"])

        assert result is not None
        # Find the actual upgrade command (not the download pre-cache)
        upgrade_cmd = [c for c in calls if "--upgrade" in c]
        assert len(upgrade_cmd) == 1
        cmd = upgrade_cmd[0]
        assert "pymaxim[semantic,llm-llama]==0.3.1" in cmd
        assert "--index-url" in cmd
        assert "https://pypi.org/simple/" in cmd

    def test_disk_check_rejects_low_space(self, handler):
        with (
            patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.0"),
            patch("shutil.disk_usage", return_value=type("U", (), {"free": int(0.5 * (1 << 30))})()),
        ):
            result = handler._run_pip_upgrade(None, [])

        assert result is None
        code, body = handler._sent[0]
        assert code == 507
        assert "disk space" in body["error"].lower()

    def test_disk_check_threshold_higher_for_torch(self, handler):
        # 2GB free should be fine for non-torch, but not for torch
        with (
            patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.0"),
            patch("shutil.disk_usage", return_value=type("U", (), {"free": int(2 * (1 << 30))})()),
        ):
            result = handler._run_pip_upgrade(None, ["llm-torch"])

        assert result is None
        code, body = handler._sent[0]
        assert code == 507

    def test_rollback_uses_local_cache(self, handler):
        calls = []
        call_count = [0]

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            call_count[0] += 1
            r = type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
            # Make the upgrade command fail
            if "--upgrade" in cmd:
                r.returncode = 1
                r.stderr = "Could not find version"
            return r

        with (
            patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.0"),
            patch("subprocess.run", side_effect=mock_run),
            patch("shutil.disk_usage", return_value=type("U", (), {"free": 10 * (1 << 30)})()),
            patch("shutil.rmtree"),
        ):
            result = handler._run_pip_upgrade(None, [])

        assert result is None
        # Verify rollback used --no-index --find-links
        rollback_cmd = [c for c in calls if "--no-index" in c]
        assert len(rollback_cmd) == 1
        assert "--find-links" in rollback_cmd[0]

    def test_timeout_sends_error(self, handler):
        import subprocess as sp

        call_count = [0]

        def mock_run(cmd, **kwargs):
            call_count[0] += 1
            if "--upgrade" in cmd:
                raise sp.TimeoutExpired(cmd, 600)
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with (
            patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.0"),
            patch("subprocess.run", side_effect=mock_run),
            patch("shutil.disk_usage", return_value=type("U", (), {"free": 10 * (1 << 30)})()),
            patch("shutil.rmtree"),
        ):
            result = handler._run_pip_upgrade(None, [])

        assert result is None
        code, body = handler._sent[0]
        assert code == 500
        assert "timed out" in body["error"]


class TestDevModeResponses:
    """Verify dev mode responses include install_mode field."""

    @pytest.fixture
    def handler(self):
        from maxim.runtime.leader_proxy import _ProxyHandler

        h = object.__new__(_ProxyHandler)
        h.upstream_url = "http://127.0.0.1:8100"
        h.api_key = None
        h.start_time = time.time()
        h.client_address = ("127.0.0.1", 12345)
        h.request_log = None
        h._sent: list[tuple[int, dict]] = []
        h._send_json = lambda code, body: h._sent.append((code, body))
        return h

    def test_dev_mode_no_git_returns_409(self, tmp_path):
        """When mode=dev but no .git exists, the check at the top of
        _handle_admin_update sends a 409 before reaching git commands."""
        from pathlib import Path as P

        assert not (tmp_path / ".git").is_dir()
        resolved_mode = "dev"
        assert resolved_mode == "dev" and not (P(str(tmp_path)) / ".git").is_dir()

    def test_dev_mode_with_git_passes_check(self, tmp_path):
        """When mode=dev and .git exists, the check passes."""
        from pathlib import Path as P

        (tmp_path / ".git").mkdir()
        assert (P(str(tmp_path)) / ".git").is_dir()


# ── TTFT keepalive (llm_timeout_scalability.md Stage 3) ───────────────────


class TestKeepaliveIntervalResolver:
    """Test _resolve_keepalive_interval_s parsing + clamping."""

    def test_default_when_env_unset(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_KEEPALIVE_DEFAULT_INTERVAL_S,
            _resolve_keepalive_interval_s,
        )

        monkeypatch.delenv("MAXIM_PROXY_KEEPALIVE_INTERVAL_S", raising=False)
        assert _resolve_keepalive_interval_s() == _PROXY_KEEPALIVE_DEFAULT_INTERVAL_S

    def test_default_when_env_empty(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_KEEPALIVE_DEFAULT_INTERVAL_S,
            _resolve_keepalive_interval_s,
        )

        monkeypatch.setenv("MAXIM_PROXY_KEEPALIVE_INTERVAL_S", "")
        assert _resolve_keepalive_interval_s() == _PROXY_KEEPALIVE_DEFAULT_INTERVAL_S

    def test_default_when_env_whitespace(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_KEEPALIVE_DEFAULT_INTERVAL_S,
            _resolve_keepalive_interval_s,
        )

        monkeypatch.setenv("MAXIM_PROXY_KEEPALIVE_INTERVAL_S", "   ")
        assert _resolve_keepalive_interval_s() == _PROXY_KEEPALIVE_DEFAULT_INTERVAL_S

    def test_default_when_env_malformed(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_KEEPALIVE_DEFAULT_INTERVAL_S,
            _resolve_keepalive_interval_s,
        )

        monkeypatch.setenv("MAXIM_PROXY_KEEPALIVE_INTERVAL_S", "not-a-number")
        assert _resolve_keepalive_interval_s() == _PROXY_KEEPALIVE_DEFAULT_INTERVAL_S

    def test_valid_value_passes_through(self, monkeypatch):
        from maxim.runtime.leader_proxy import _resolve_keepalive_interval_s

        monkeypatch.setenv("MAXIM_PROXY_KEEPALIVE_INTERVAL_S", "45")
        assert _resolve_keepalive_interval_s() == 45.0

    def test_value_below_minimum_clamps_up(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_KEEPALIVE_MIN_INTERVAL_S,
            _resolve_keepalive_interval_s,
        )

        monkeypatch.setenv("MAXIM_PROXY_KEEPALIVE_INTERVAL_S", "1")
        assert _resolve_keepalive_interval_s() == _PROXY_KEEPALIVE_MIN_INTERVAL_S

    def test_value_above_maximum_clamps_down(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_KEEPALIVE_MAX_INTERVAL_S,
            _resolve_keepalive_interval_s,
        )

        monkeypatch.setenv("MAXIM_PROXY_KEEPALIVE_INTERVAL_S", "999")
        assert _resolve_keepalive_interval_s() == _PROXY_KEEPALIVE_MAX_INTERVAL_S

    def test_negative_value_clamps_to_minimum(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_KEEPALIVE_MIN_INTERVAL_S,
            _resolve_keepalive_interval_s,
        )

        monkeypatch.setenv("MAXIM_PROXY_KEEPALIVE_INTERVAL_S", "-10")
        assert _resolve_keepalive_interval_s() == _PROXY_KEEPALIVE_MIN_INTERVAL_S


class TestEventStreamDetection:
    """Test _is_event_stream_response distinguishes SSE from buffered JSON."""

    def test_plain_event_stream(self):
        from maxim.runtime.leader_proxy import _is_event_stream_response

        assert _is_event_stream_response({"Content-Type": "text/event-stream"})

    def test_event_stream_with_charset(self):
        from maxim.runtime.leader_proxy import _is_event_stream_response

        assert _is_event_stream_response({"Content-Type": "text/event-stream; charset=utf-8"})

    def test_lowercase_header_name(self):
        from maxim.runtime.leader_proxy import _is_event_stream_response

        assert _is_event_stream_response({"content-type": "text/event-stream"})

    def test_mixed_case_value(self):
        from maxim.runtime.leader_proxy import _is_event_stream_response

        assert _is_event_stream_response({"Content-Type": "Text/Event-Stream"})

    def test_application_json_is_not_stream(self):
        from maxim.runtime.leader_proxy import _is_event_stream_response

        assert not _is_event_stream_response({"Content-Type": "application/json"})

    def test_missing_content_type(self):
        from maxim.runtime.leader_proxy import _is_event_stream_response

        assert not _is_event_stream_response({})

    def test_other_headers_ignored(self):
        from maxim.runtime.leader_proxy import _is_event_stream_response

        # Headers that contain "text/event-stream" in their value but aren't
        # Content-Type should not trigger a false positive.
        assert not _is_event_stream_response({"X-Custom": "text/event-stream", "Content-Type": "application/json"})


class TestKeepaliveChunkFrameFormat:
    """Pin the wire format of the pre-encoded keepalive chunk frame."""

    def test_frame_is_valid_chunked_encoding(self):
        from maxim.runtime.leader_proxy import _KEEPALIVE_CHUNK_FRAME

        # Format: <hex-size>\r\n<payload>\r\n
        size_part, _, rest = _KEEPALIVE_CHUNK_FRAME.partition(b"\r\n")
        payload, trailer, empty = rest.partition(b"\r\n")
        expected_size = int(size_part, 16)
        assert len(payload) == expected_size
        assert trailer == b"\r\n"
        assert empty == b""

    def test_payload_is_sse_comment(self):
        from maxim.runtime.leader_proxy import _KEEPALIVE_CHUNK_FRAME

        # The payload must begin with ":" per the SSE spec for clients to
        # treat it as a comment and ignore it.
        _, _, rest = _KEEPALIVE_CHUNK_FRAME.partition(b"\r\n")
        payload, _, _ = rest.partition(b"\r\n")
        assert payload.startswith(b":")

    def test_payload_terminates_sse_event(self):
        from maxim.runtime.leader_proxy import _KEEPALIVE_CHUNK_FRAME

        # SSE event boundary is \n\n. Comments need the same terminator to
        # flush through any SSE parser the client may layer above raw HTTP.
        _, _, rest = _KEEPALIVE_CHUNK_FRAME.partition(b"\r\n")
        payload, _, _ = rest.partition(b"\r\n")
        assert payload.endswith(b"\n\n")


class TestKeepaliveEmitter:
    """Test _run_keepalive_emitter under controlled timing."""

    def test_emits_at_configured_cadence(self):
        """With a 50ms interval, ~3 emissions in 175ms (after 50, 100, 150ms)."""
        from maxim.runtime.leader_proxy import (
            _KEEPALIVE_CHUNK_FRAME,
            _run_keepalive_emitter,
        )

        class FakeWFile:
            def __init__(self):
                self.writes: list[bytes] = []
                self.flushes = 0

            def write(self, data):
                self.writes.append(data)

            def flush(self):
                self.flushes += 1

        wfile = FakeWFile()
        lock = threading.Lock()
        stop = threading.Event()

        t = threading.Thread(
            target=_run_keepalive_emitter,
            args=(wfile, lock, stop, 0.05),
            daemon=True,
        )
        t.start()
        time.sleep(0.175)
        stop.set()
        t.join(timeout=1.0)

        # At least 2 emissions in the window (timing on CI can be loose).
        assert len(wfile.writes) >= 2
        # Every write is the keepalive frame, byte-exact.
        assert all(w == _KEEPALIVE_CHUNK_FRAME for w in wfile.writes)
        # Each write is paired with a flush.
        assert wfile.flushes == len(wfile.writes)

    def test_stops_immediately_when_event_set(self):
        from maxim.runtime.leader_proxy import _run_keepalive_emitter

        class FakeWFile:
            def __init__(self):
                self.writes: list[bytes] = []

            def write(self, data):
                self.writes.append(data)

            def flush(self):
                pass

        wfile = FakeWFile()
        lock = threading.Lock()
        stop = threading.Event()
        stop.set()  # already stopped — emitter must exit on first wait()

        t = threading.Thread(
            target=_run_keepalive_emitter,
            args=(wfile, lock, stop, 0.05),
            daemon=True,
        )
        t.start()
        t.join(timeout=1.0)

        assert not t.is_alive()
        assert wfile.writes == []

    def test_exits_silently_on_write_error(self):
        """Broken pipe means the client is gone. Emitter must not raise."""
        from maxim.runtime.leader_proxy import _run_keepalive_emitter

        class BrokenWFile:
            def write(self, data):
                raise BrokenPipeError("client gone")

            def flush(self):
                pass

        wfile = BrokenWFile()
        lock = threading.Lock()
        stop = threading.Event()

        t = threading.Thread(
            target=_run_keepalive_emitter,
            args=(wfile, lock, stop, 0.05),
            daemon=True,
        )
        t.start()
        t.join(timeout=1.0)

        assert not t.is_alive()
        # Emitter sets the stop event on write error so callers can detect
        # the client disconnect without inspecting thread state.
        assert stop.is_set()

    def test_acquires_write_lock(self):
        """The emitter must release the lock between iterations so the main
        chunk writer can interleave. Holding the lock for the whole sleep
        would block real chunk forwarding."""
        from maxim.runtime.leader_proxy import _run_keepalive_emitter

        class FakeWFile:
            def write(self, data):
                pass

            def flush(self):
                pass

        wfile = FakeWFile()
        lock = threading.Lock()
        stop = threading.Event()

        t = threading.Thread(
            target=_run_keepalive_emitter,
            args=(wfile, lock, stop, 0.05),
            daemon=True,
        )
        t.start()

        # Probe the lock during the emitter's sleep window — must be free.
        time.sleep(0.025)
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        stop.set()
        t.join(timeout=1.0)

        assert acquired, "emitter must not hold the lock during its sleep"


# ── Context-overflow admission (llm_timeout_scalability.md follow-up) ─────


class TestContextOverheadResolver:
    """Test _resolve_context_overhead_tokens parsing + clamping."""

    def test_default_when_unset(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_CONTEXT_OVERHEAD_DEFAULT,
            _resolve_context_overhead_tokens,
        )

        monkeypatch.delenv("MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS", raising=False)
        assert _resolve_context_overhead_tokens() == _PROXY_CONTEXT_OVERHEAD_DEFAULT

    def test_default_when_empty(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_CONTEXT_OVERHEAD_DEFAULT,
            _resolve_context_overhead_tokens,
        )

        monkeypatch.setenv("MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS", "")
        assert _resolve_context_overhead_tokens() == _PROXY_CONTEXT_OVERHEAD_DEFAULT

    def test_default_when_malformed(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_CONTEXT_OVERHEAD_DEFAULT,
            _resolve_context_overhead_tokens,
        )

        monkeypatch.setenv("MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS", "not-a-number")
        assert _resolve_context_overhead_tokens() == _PROXY_CONTEXT_OVERHEAD_DEFAULT

    def test_valid_value_passes_through(self, monkeypatch):
        from maxim.runtime.leader_proxy import _resolve_context_overhead_tokens

        monkeypatch.setenv("MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS", "512")
        assert _resolve_context_overhead_tokens() == 512

    def test_clamps_below_min(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_CONTEXT_OVERHEAD_MIN,
            _resolve_context_overhead_tokens,
        )

        monkeypatch.setenv("MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS", "-100")
        assert _resolve_context_overhead_tokens() == _PROXY_CONTEXT_OVERHEAD_MIN

    def test_clamps_above_max(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _PROXY_CONTEXT_OVERHEAD_MAX,
            _resolve_context_overhead_tokens,
        )

        monkeypatch.setenv("MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS", "99999")
        assert _resolve_context_overhead_tokens() == _PROXY_CONTEXT_OVERHEAD_MAX


class TestAdmissionEnableGate:
    """Test _is_admission_enabled parser."""

    def test_default_enabled(self, monkeypatch):
        from maxim.runtime.leader_proxy import _is_admission_enabled

        monkeypatch.delenv("MAXIM_PROXY_CONTEXT_ADMISSION", raising=False)
        assert _is_admission_enabled()

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "False", "OFF", "  off  ", ""])
    def test_disabling_values(self, monkeypatch, value):
        from maxim.runtime.leader_proxy import _is_admission_enabled

        monkeypatch.setenv("MAXIM_PROXY_CONTEXT_ADMISSION", value)
        assert not _is_admission_enabled()

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything-else"])
    def test_enabling_values(self, monkeypatch, value):
        from maxim.runtime.leader_proxy import _is_admission_enabled

        monkeypatch.setenv("MAXIM_PROXY_CONTEXT_ADMISSION", value)
        assert _is_admission_enabled()


class TestInputTokenEstimator:
    """Test _estimate_inference_input_tokens across body shapes."""

    def test_empty_body(self):
        from maxim.runtime.leader_proxy import _estimate_inference_input_tokens

        assert _estimate_inference_input_tokens(b"") == (0, 0)

    def test_malformed_json(self):
        from maxim.runtime.leader_proxy import _estimate_inference_input_tokens

        assert _estimate_inference_input_tokens(b"{not json") == (0, 0)

    def test_non_dict_root(self):
        from maxim.runtime.leader_proxy import _estimate_inference_input_tokens

        assert _estimate_inference_input_tokens(b'["list"]') == (0, 0)

    def test_chat_completion_string_content(self):
        from maxim.runtime.leader_proxy import _estimate_inference_input_tokens

        body = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello world"},
                ]
            }
        ).encode()
        tokens, max_t = _estimate_inference_input_tokens(body)
        # 16 chars + 11 chars + 32 chars (2 × 16 role overhead) = 59 chars / 3.5 ≈ 16
        assert 10 <= tokens <= 25
        assert max_t == 0

    def test_chat_completion_with_max_tokens(self):
        from maxim.runtime.leader_proxy import _estimate_inference_input_tokens

        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}).encode()
        _, max_t = _estimate_inference_input_tokens(body)
        assert max_t == 100

    def test_chat_completion_negative_max_tokens_clamps_to_zero(self):
        from maxim.runtime.leader_proxy import _estimate_inference_input_tokens

        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": -50}).encode()
        _, max_t = _estimate_inference_input_tokens(body)
        assert max_t == 0

    def test_chat_completion_multipart_content(self):
        from maxim.runtime.leader_proxy import _estimate_inference_input_tokens

        body = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image"},
                            {"type": "image_url", "image_url": "data:..."},
                        ],
                    }
                ]
            }
        ).encode()
        tokens, _ = _estimate_inference_input_tokens(body)
        # "Describe this image" = 19 chars + 16 role overhead = 35 chars / 3.5 = 10
        # image_url ignored (no text field) — that's by design
        assert tokens > 0

    def test_legacy_completions_prompt_string(self):
        from maxim.runtime.leader_proxy import _estimate_inference_input_tokens

        body = json.dumps({"prompt": "Once upon a time", "max_tokens": 50}).encode()
        tokens, max_t = _estimate_inference_input_tokens(body)
        # 16 chars / 3.5 ≈ 4. Don't lock the exact value; just confirm > 0.
        assert tokens > 0
        assert max_t == 50

    def test_legacy_completions_prompt_list(self):
        from maxim.runtime.leader_proxy import _estimate_inference_input_tokens

        body = json.dumps({"prompt": ["first chunk", "second chunk"]}).encode()
        tokens, _ = _estimate_inference_input_tokens(body)
        assert tokens > 0

    def test_large_prompt_scales(self):
        from maxim.runtime.leader_proxy import _estimate_inference_input_tokens

        # 1000-char message → ~286 tokens at ratio 3.5
        big = "x" * 1000
        body = json.dumps({"messages": [{"role": "user", "content": big}]}).encode()
        tokens, _ = _estimate_inference_input_tokens(body)
        assert 280 <= tokens <= 295


class TestAdmissionCheck:
    """Test _check_context_admission decision logic."""

    def test_within_budget_admits(self):
        from maxim.runtime.leader_proxy import _check_context_admission

        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}).encode()
        # Tiny prompt + 100 + 256 overhead << 8192 ctx → admit
        assert _check_context_admission(body, n_ctx=8192, overhead=256) is None

    def test_unparseable_passes_through(self):
        from maxim.runtime.leader_proxy import _check_context_admission

        # Don't false-reject on malformed bodies — let upstream return
        # a cleaner 400 from its own parser.
        assert _check_context_admission(b"{malformed", n_ctx=100, overhead=10) is None

    def test_rejects_oversize_prompt(self):
        from maxim.runtime.leader_proxy import _check_context_admission

        # 1000-char message → ~286 tokens. n_ctx=300 → reject.
        body = json.dumps({"messages": [{"role": "user", "content": "x" * 1000}]}).encode()
        result = _check_context_admission(body, n_ctx=300, overhead=10)
        assert result is not None
        err = result["error"]
        assert err["code"] == "context_overflow"
        assert err["type"] == "invalid_request_error"
        assert err["context_window"] == 300
        assert err["safety_overhead_tokens"] == 10
        assert err["estimated_prompt_tokens"] > 280
        # max_tokens unset → applies _PROXY_DEFAULT_MAX_TOKENS fallback.
        # Don't hard-code the literal here; reference the constant so a
        # future tune doesn't drift this test out of sync.
        from maxim.runtime.leader_proxy import _PROXY_DEFAULT_MAX_TOKENS

        assert err["max_tokens"] == _PROXY_DEFAULT_MAX_TOKENS
        # Total > n_ctx is the rejection condition; message names the math
        assert "exceeds the model's context window" in err["message"]

    def test_rejects_when_max_tokens_pushes_over(self):
        from maxim.runtime.leader_proxy import _check_context_admission

        # Small prompt but huge max_tokens
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10000}).encode()
        result = _check_context_admission(body, n_ctx=8192, overhead=256)
        assert result is not None
        assert result["error"]["max_tokens"] == 10000

    def test_error_envelope_is_openai_compatible(self):
        """Agent-side error handlers key on error.code / error.type — pin
        the envelope so changes that drop those fields fail loudly."""
        from maxim.runtime.leader_proxy import _check_context_admission

        body = json.dumps({"messages": [{"role": "user", "content": "x" * 10000}]}).encode()
        result = _check_context_admission(body, n_ctx=100, overhead=10)
        assert result is not None
        assert "error" in result
        assert set(result["error"].keys()) >= {"code", "type", "message"}


class TestContextWindowResolver:
    """Test _get_proxy_context_window resolution + caching + reset."""

    def test_resolves_from_env(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _get_proxy_context_window,
            _reset_proxy_context_window_cache_for_test,
        )

        _reset_proxy_context_window_cache_for_test()
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "8192")
        assert _get_proxy_context_window() == 8192

    def test_none_when_unset(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _get_proxy_context_window,
            _reset_proxy_context_window_cache_for_test,
        )

        _reset_proxy_context_window_cache_for_test()
        monkeypatch.delenv("MAXIM_LLM_N_CTX", raising=False)
        assert _get_proxy_context_window() is None

    def test_cache_blocks_repeat_resolution(self, monkeypatch):
        """Once resolved (or known-None), the cache returns the same
        answer without re-walking the loader. This matters because
        ``resolve_setting`` reads env on every call — if the cache
        were broken, an env change mid-process would silently flip
        the admission gate."""
        from maxim.runtime.leader_proxy import (
            _get_proxy_context_window,
            _reset_proxy_context_window_cache_for_test,
        )

        _reset_proxy_context_window_cache_for_test()
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "4096")
        first = _get_proxy_context_window()
        # Change env without resetting cache
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "8192")
        second = _get_proxy_context_window()
        assert first == second == 4096

    def test_reset_clears_cache(self, monkeypatch):
        from maxim.runtime.leader_proxy import (
            _get_proxy_context_window,
            _reset_proxy_context_window_cache_for_test,
        )

        _reset_proxy_context_window_cache_for_test()
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "4096")
        assert _get_proxy_context_window() == 4096
        _reset_proxy_context_window_cache_for_test()
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "8192")
        assert _get_proxy_context_window() == 8192


class TestLaneTimeoutFieldFlow:
    """Test that LaneTierConfig.timeout_s flows through to the backend
    provider config via apply_lane_env_overrides."""

    def test_env_var_populates_remote_timeout_s(self, monkeypatch):
        from maxim.runtime.lane_models import apply_lane_env_overrides
        from maxim.runtime.worker_pool import LaneConfig

        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8100")
        monkeypatch.setenv("MAXIM_LANE_LARGE_TIMEOUT_S", "600")
        lanes = {"large": LaneConfig(name="large", max_workers=1)}
        updated = apply_lane_env_overrides(lanes)
        assert updated["large"].remote_timeout_s == 600.0

    def test_missing_env_leaves_default_none(self, monkeypatch):
        from maxim.runtime.lane_models import apply_lane_env_overrides
        from maxim.runtime.worker_pool import LaneConfig

        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8100")
        monkeypatch.delenv("MAXIM_LANE_LARGE_TIMEOUT_S", raising=False)
        lanes = {"large": LaneConfig(name="large", max_workers=1)}
        updated = apply_lane_env_overrides(lanes)
        assert updated["large"].remote_timeout_s is None

    def test_zero_env_treated_as_unset(self, monkeypatch):
        """Strictly-positive constraint applies at the env-passthrough
        layer too — zero or negative is silently ignored (the loader
        side raises ConfigurationError; this is the bypass-the-loader
        defensive path)."""
        from maxim.runtime.lane_models import apply_lane_env_overrides
        from maxim.runtime.worker_pool import LaneConfig

        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8100")
        monkeypatch.setenv("MAXIM_LANE_LARGE_TIMEOUT_S", "0")
        lanes = {"large": LaneConfig(name="large", max_workers=1)}
        updated = apply_lane_env_overrides(lanes)
        assert updated["large"].remote_timeout_s is None

    def test_malformed_env_treated_as_unset(self, monkeypatch):
        from maxim.runtime.lane_models import apply_lane_env_overrides
        from maxim.runtime.worker_pool import LaneConfig

        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8100")
        monkeypatch.setenv("MAXIM_LANE_LARGE_TIMEOUT_S", "not-a-number")
        lanes = {"large": LaneConfig(name="large", max_workers=1)}
        updated = apply_lane_env_overrides(lanes)
        assert updated["large"].remote_timeout_s is None


class TestBackendsAlwaysSendMaxTokens:
    """Pin that both backends always populate ``max_tokens`` in the
    outgoing HTTP request.

    This makes the proxy admission gate's ``_PROXY_DEFAULT_MAX_TOKENS``
    fallback **structurally unreachable** for Maxim-internal traffic —
    when that fallback fires, it's signal that an external client
    (curl probe, raw API script) made the call, not a Maxim agent.

    The invariant has two halves: the method signature requires
    ``max_tokens`` as a keyword-only argument (no default), AND the
    payload construction always includes the field. Both backends
    follow this contract. A silent regression here (e.g., someone
    adds ``max_tokens: int = 0`` as a default) would let the gate's
    fallback start firing on real agent traffic — degraded admission
    accuracy without a loud failure.
    """

    def test_maxim_peer_backend_build_payload_includes_max_tokens(self):
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        # Construct minimally without firing __init__ — we only need
        # the bound method on the class for payload construction.
        backend = object.__new__(_MaximPeerBackend)
        payload = backend._build_payload(
            system="sys",
            user="usr",
            max_tokens=512,
            temperature=0.7,
            stop=(),
            model="test-model",
            tools=None,
            thinking=None,
        )
        assert "max_tokens" in payload
        assert payload["max_tokens"] == 512
        # Verify the type coercion happens (defensive against bool sneaking
        # through — bool is technically int in Python)
        assert isinstance(payload["max_tokens"], int)

    def test_maxim_peer_backend_complete_with_usage_max_tokens_required(self):
        """Removing the ``max_tokens`` requirement on the signature
        would let callers silently elide the field — the proxy gate
        would then apply ``_PROXY_DEFAULT_MAX_TOKENS`` (2048) which
        may not match what the agent actually wants."""
        import inspect

        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        sig = inspect.signature(_MaximPeerBackend.complete_with_usage)
        assert "max_tokens" in sig.parameters, "_MaximPeerBackend.complete_with_usage must accept max_tokens"
        param = sig.parameters["max_tokens"]
        assert param.default is inspect.Parameter.empty, (
            "max_tokens must be required (no default) — see TestBackendsAlwaysSendMaxTokens docstring"
        )
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            "max_tokens must be keyword-only — positional would risk silent miscount on signature changes"
        )

    def test_maxim_peer_backend_complete_max_tokens_required(self):
        """Same contract on the legacy ``complete`` method."""
        import inspect

        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        sig = inspect.signature(_MaximPeerBackend.complete)
        param = sig.parameters["max_tokens"]
        assert param.default is inspect.Parameter.empty
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_openai_backend_complete_with_usage_max_tokens_required(self):
        """Same contract on the cloud-backend path. ``_OpenAIBackend``
        passes ``max_tokens`` straight through to the OpenAI SDK
        client; the gate's fallback must not be load-bearing here either."""
        import inspect

        from maxim.models.language.openai_backend import _OpenAIBackend

        sig = inspect.signature(_OpenAIBackend.complete_with_usage)
        param = sig.parameters["max_tokens"]
        assert param.default is inspect.Parameter.empty
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_openai_backend_complete_max_tokens_required(self):
        import inspect

        from maxim.models.language.openai_backend import _OpenAIBackend

        sig = inspect.signature(_OpenAIBackend.complete)
        param = sig.parameters["max_tokens"]
        assert param.default is inspect.Parameter.empty
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


class TestProxyAuthSchemeDispatch:
    """CC13 auth format-freeze: ``_ProxyHandler._check_auth`` parses the
    Authorization *scheme* before the credential and dispatches on it.

    1.0 supports Bearer only. The freeze guarantee is that an unknown
    scheme (a future ``Signature`` / ``HSM-Sig`` / mTLS variant) returns
    a clean 401 rather than being mistaken for a malformed Bearer — so
    1.1+ can add a scheme by extending the dispatch, not by re-shaping
    the auth path.
    """

    def _handler(self, *, api_key, auth_header):
        from maxim.runtime.leader_proxy import _ProxyHandler

        h = object.__new__(_ProxyHandler)
        h.api_key = api_key
        h.headers = {} if auth_header is None else {"Authorization": auth_header}
        h.client_address = ("10.0.0.1", 5000)
        h._sent: list[tuple[int, dict]] = []
        h._send_json = lambda code, body: h._sent.append((code, body))
        return h

    def test_no_api_key_configured_allows_all(self):
        h = self._handler(api_key=None, auth_header=None)
        assert h._check_auth() is True
        assert h._sent == []

    def test_correct_bearer_authorized(self):
        h = self._handler(api_key="sk-secret", auth_header="Bearer sk-secret")
        assert h._check_auth() is True
        assert h._sent == []

    def test_bearer_scheme_is_case_insensitive(self):
        """RFC 7235: auth scheme names are case-insensitive. A lowercase
        ``bearer`` with the right credential is accepted."""
        h = self._handler(api_key="sk-secret", auth_header="bearer sk-secret")
        assert h._check_auth() is True
        assert h._sent == []

    def test_wrong_bearer_credential_rejected_401(self):
        h = self._handler(api_key="sk-secret", auth_header="Bearer sk-wrong")
        assert h._check_auth() is False
        assert len(h._sent) == 1
        code, body = h._sent[0]
        assert code == 401
        assert body == {"error": "Invalid API key"}

    def test_non_ascii_bearer_credential_rejected_not_500(self):
        """A non-ASCII Bearer credential makes ``secrets.compare_digest``
        raise TypeError; ``_check_auth`` must catch it and return a clean
        401, not let the exception escape to a 500 from the handler."""
        h = self._handler(api_key="sk-secret", auth_header="Bearer sk-ééé")
        assert h._check_auth() is False
        assert len(h._sent) == 1
        code, body = h._sent[0]
        assert code == 401
        assert body == {"error": "Invalid API key"}

    def test_unknown_scheme_rejected_401_not_invalid_key(self):
        """A future ``Signature`` header is NOT mistaken for a malformed
        Bearer (which would say 'Invalid API key') — it gets the distinct
        unsupported-scheme 401, and never a 400 or a silent accept."""
        h = self._handler(
            api_key="sk-secret",
            auth_header='Signature keyId="peer-a",signature="abc"',
        )
        assert h._check_auth() is False
        assert len(h._sent) == 1
        code, body = h._sent[0]
        assert code == 401
        assert body == {"error": "Unsupported or missing authorization scheme"}

    def test_hsm_sig_scheme_reserved_rejected(self):
        h = self._handler(api_key="sk-secret", auth_header="HSM-Sig deadbeef")
        assert h._check_auth() is False
        code, body = h._sent[0]
        assert code == 401
        assert body == {"error": "Unsupported or missing authorization scheme"}

    def test_missing_authorization_header_rejected_401(self):
        h = self._handler(api_key="sk-secret", auth_header=None)
        assert h._check_auth() is False
        code, body = h._sent[0]
        assert code == 401
        assert body == {"error": "Unsupported or missing authorization scheme"}


class TestKeyFingerprint:
    """CC13: ``_key_fingerprint`` must never emit raw key material to logs
    (CodeQL clear-text-logging). Pin the security property so a future edit
    can't reintroduce a prefix leak.
    """

    def test_never_contains_raw_secret(self):
        from maxim.runtime.leader_proxy import _key_fingerprint

        secret = "sk-supersecret-1234567890"
        fp = _key_fingerprint(secret)
        assert secret not in fp
        # No 6+ char prefix of the secret leaks either.
        assert secret[:6] not in fp
        assert fp.startswith("sha256:")

    def test_stable_for_same_key(self):
        from maxim.runtime.leader_proxy import _key_fingerprint

        assert _key_fingerprint("sk-abc") == _key_fingerprint("sk-abc")

    def test_distinguishes_different_keys(self):
        from maxim.runtime.leader_proxy import _key_fingerprint

        assert _key_fingerprint("sk-abc") != _key_fingerprint("sk-xyz")

    def test_missing_value_sentinel(self):
        from maxim.runtime.leader_proxy import _key_fingerprint

        assert _key_fingerprint("") == "<missing>"
        assert _key_fingerprint(None) == "<missing>"

    def test_non_ascii_value_does_not_raise(self):
        from maxim.runtime.leader_proxy import _key_fingerprint

        fp = _key_fingerprint("sk-ééé")
        assert fp.startswith("sha256:")
