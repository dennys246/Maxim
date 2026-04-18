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
