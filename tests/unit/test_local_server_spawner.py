"""Tests for LocalServerSpawner (multi-LLM Phase 6a).

These tests don't actually launch llama-cpp-server — they verify the
configuration, health-check, and lifecycle logic in isolation via mocks.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch


from maxim.runtime.local_server_spawner import LocalServerSpawner


class TestConstruction:
    def test_defaults(self):
        s = LocalServerSpawner(model_path="/tmp/fake.gguf")
        assert s._port == 8100
        assert s._bind_host == "127.0.0.1"
        assert s._n_ctx == 8192
        assert s._n_gpu_layers == -1
        assert s._chat_format == "chatml"
        assert s.base_url == "http://127.0.0.1:8100/v1"
        assert s.is_running is False

    def test_leader_bind_host(self):
        s = LocalServerSpawner(model_path="/tmp/fake.gguf", bind_host="0.0.0.0")
        # base_url still uses loopback for local client access,
        # even though the bind is public
        assert s.base_url == "http://127.0.0.1:8100/v1"
        assert s._bind_host == "0.0.0.0"


class TestStartFailures:
    def test_missing_model_file_returns_none(self, tmp_path):
        s = LocalServerSpawner(model_path=str(tmp_path / "does-not-exist.gguf"))
        url = s.start(timeout_s=1.0)
        assert url is None
        assert s.is_running is False

    def test_subprocess_launch_failure_returns_none(self, tmp_path):
        model = tmp_path / "fake.gguf"
        model.write_bytes(b"GGUF")
        s = LocalServerSpawner(model_path=str(model))
        with patch("subprocess.Popen", side_effect=OSError("command not found")):
            url = s.start(timeout_s=1.0)
        assert url is None
        assert s.is_running is False


class TestCommandBuild:
    def test_cmd_includes_all_flags(self, tmp_path):
        model = tmp_path / "fake.gguf"
        s = LocalServerSpawner(
            model_path=str(model),
            port=9000,
            bind_host="0.0.0.0",
            n_ctx=4096,
            n_gpu_layers=32,
            chat_format="llama-3",
        )
        cmd = s._build_cmd()
        assert "--model" in cmd and str(model) in cmd
        assert "--port" in cmd and "9000" in cmd
        assert "--host" in cmd and "0.0.0.0" in cmd
        assert "--n_ctx" in cmd and "4096" in cmd
        assert "--n_gpu_layers" in cmd and "32" in cmd
        assert "--chat_format" in cmd and "llama-3" in cmd


class TestApiKey:
    def test_api_key_not_in_cmd_by_default(self, tmp_path):
        model = tmp_path / "fake.gguf"
        s = LocalServerSpawner(model_path=str(model))
        cmd = s._build_cmd()
        assert "--api_key" not in cmd

    def test_api_key_added_when_set(self, tmp_path):
        model = tmp_path / "fake.gguf"
        s = LocalServerSpawner(model_path=str(model), api_key="secret-token-123")
        cmd = s._build_cmd()
        assert "--api_key" in cmd
        idx = cmd.index("--api_key")
        assert cmd[idx + 1] == "secret-token-123"

    def test_empty_api_key_not_added(self, tmp_path):
        model = tmp_path / "fake.gguf"
        s = LocalServerSpawner(model_path=str(model), api_key="")
        assert "--api_key" not in s._build_cmd()


class TestTunnelEcho:
    def test_echo_off_by_default_streams_to_devnull(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MAXIM_TUNNEL_ECHO", raising=False)
        import subprocess as _sp

        model = tmp_path / "fake.gguf"
        model.write_bytes(b"GGUF")
        s = LocalServerSpawner(model_path=str(model))
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 1  # exit immediately
            mock_proc.pid = 99999
            mock_popen.return_value = mock_proc
            s.start(timeout_s=0.1)
        kwargs = mock_popen.call_args.kwargs
        assert kwargs.get("stdout") == _sp.DEVNULL
        assert kwargs.get("stderr") == _sp.DEVNULL

    def test_echo_on_inherits_parent_streams(self, tmp_path, monkeypatch, caplog):
        import logging

        monkeypatch.setenv("MAXIM_TUNNEL_ECHO", "1")
        model = tmp_path / "fake.gguf"
        model.write_bytes(b"GGUF")
        s = LocalServerSpawner(model_path=str(model))
        with (
            patch("subprocess.Popen") as mock_popen,
            caplog.at_level(logging.INFO, logger="maxim.runtime.local_server_spawner"),
        ):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 1
            mock_proc.pid = 99999
            mock_popen.return_value = mock_proc
            s.start(timeout_s=0.1)
        kwargs = mock_popen.call_args.kwargs
        # None → inherits parent's stdout/stderr
        assert kwargs.get("stdout") is None
        assert kwargs.get("stderr") is None
        # And we log a notice (used to print to stderr — now via logger.info)
        assert any("MAXIM_TUNNEL_ECHO" in r.message for r in caplog.records)


class TestSignalIsolation:
    def test_subprocess_started_in_new_session(self, tmp_path):
        """start_new_session=True must be passed so Ctrl+C in the terminal
        doesn't kill the spawned server before Maxim's cleanup completes."""
        model = tmp_path / "fake.gguf"
        model.write_bytes(b"GGUF")
        s = LocalServerSpawner(model_path=str(model))
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 1  # fake "exited" so start returns early
            mock_proc.pid = 99999
            mock_popen.return_value = mock_proc
            s.start(timeout_s=0.1)
        kwargs = mock_popen.call_args.kwargs
        assert kwargs.get("start_new_session") is True


class TestCudaEnvRestoration:
    def test_restores_original_cuda_devices(self, monkeypatch):
        """When parent has CUDA hidden but original value was saved, subprocess gets it."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        with patch("maxim.runtime.local_server_spawner.get_original_cuda_devices") as mock_orig:
            mock_orig.return_value = "0"
            s = LocalServerSpawner(model_path="/tmp/fake.gguf")
            env = s._build_subprocess_env()
        assert env["CUDA_VISIBLE_DEVICES"] == "0"

    def test_leaves_cuda_alone_when_not_hidden(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        with patch("maxim.runtime.local_server_spawner.get_original_cuda_devices") as mock_orig:
            mock_orig.return_value = None
            s = LocalServerSpawner(model_path="/tmp/fake.gguf")
            env = s._build_subprocess_env()
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_leaves_cuda_alone_when_original_not_saved(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        with patch("maxim.runtime.local_server_spawner.get_original_cuda_devices") as mock_orig:
            mock_orig.return_value = None  # Blackwell workaround didn't fire
            s = LocalServerSpawner(model_path="/tmp/fake.gguf")
            env = s._build_subprocess_env()
        assert env["CUDA_VISIBLE_DEVICES"] == ""


class TestStop:
    def test_stop_without_start_is_noop(self):
        s = LocalServerSpawner(model_path="/tmp/fake.gguf")
        s.stop()  # should not raise
        assert s.is_running is False

    @patch("maxim.runtime.local_server_spawner._kill_process_tree")
    def test_stop_terminates_process(self, mock_kill_tree):
        s = LocalServerSpawner(model_path="/tmp/fake.gguf")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # running
        mock_proc.pid = 99999
        s._process = mock_proc
        s.stop()
        mock_proc.terminate.assert_called_once()
        assert s._process is None

    @patch("maxim.runtime.local_server_spawner._kill_process_tree")
    def test_stop_kills_on_timeout(self, mock_kill_tree):
        s = LocalServerSpawner(model_path="/tmp/fake.gguf")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 99999
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="", timeout=5.0), None]
        s._process = mock_proc
        s.stop()
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    @patch("maxim.runtime.local_server_spawner._kill_process_tree")
    def test_stop_swallows_exceptions(self, mock_kill_tree):
        s = LocalServerSpawner(model_path="/tmp/fake.gguf")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 99999
        mock_proc.terminate.side_effect = RuntimeError("oops")
        s._process = mock_proc
        s.stop()  # should not raise


class TestHealthCheck:
    @staticmethod
    def _ok_resp():
        from maxim.utils import http as _http

        return _http.Response(
            status=200,
            headers={},
            content=b"{}",
            elapsed_ms=1.0,
            endpoint=_http._EXTERNAL_ENDPOINT,
            request_id="r",
        )

    def test_healthy_returns_true(self):
        s = LocalServerSpawner(model_path="/tmp/fake.gguf")
        with patch("maxim.utils.http.fetch_url", return_value=self._ok_resp()):
            assert s._health_check() is True

    def test_unhealthy_returns_false_on_exception(self):
        s = LocalServerSpawner(model_path="/tmp/fake.gguf")
        with patch("maxim.utils.http.fetch_url", side_effect=OSError("refused")):
            assert s._health_check() is False

    def test_api_key_sent_as_bearer(self):
        """When api_key is set, health check must include Authorization header
        (llama-cpp-server's /v1/models endpoint requires auth when --api_key
        is passed, so missing header would cause a 401 -> false timeout)."""
        s = LocalServerSpawner(model_path="/tmp/fake.gguf", api_key="secret-token")
        captured: dict = {}

        def _capture(url, *, method="GET", headers=None, **_):
            captured["headers"] = dict(headers or {})
            return self._ok_resp()

        with patch("maxim.utils.http.fetch_url", side_effect=_capture):
            assert s._health_check() is True
            assert captured["headers"].get("Authorization") == "Bearer secret-token"

    def test_no_api_key_no_auth_header(self):
        s = LocalServerSpawner(model_path="/tmp/fake.gguf")  # no api_key
        captured: dict = {}

        def _capture(url, *, method="GET", headers=None, **_):
            captured["headers"] = dict(headers or {})
            return self._ok_resp()

        with patch("maxim.utils.http.fetch_url", side_effect=_capture):
            s._health_check()
            assert "Authorization" not in captured["headers"]

    def test_401_counts_as_up(self):
        """401 means the HTTP listener is alive, just rejected the key.
        We treat this as 'server ready' — the spawner's job is just to
        confirm the listener is up, not to validate the auth itself."""
        from maxim.utils import http as _http

        s = LocalServerSpawner(model_path="/tmp/fake.gguf", api_key="wrong-key")
        with patch(
            "maxim.utils.http.fetch_url",
            side_effect=_http.HTTPAuthError("_external", status=401, fix_hint="Auth rejected (401)"),
        ):
            assert s._health_check() is True

    def test_500_does_not_count_as_up(self):
        """A real server error should be treated as not-ready."""
        from maxim.utils import http as _http

        s = LocalServerSpawner(model_path="/tmp/fake.gguf")
        with patch(
            "maxim.utils.http.fetch_url",
            side_effect=_http.HTTPServerError("_external", status=500, fix_hint="Server error (500)"),
        ):
            assert s._health_check() is False
