"""#1314's follow-ups (#1408): the fail-closed defects left after every spawned llama-cpp-server became keyed.

- Item 2: the n_ctx probe the proxy's ``/v1/debug/vram`` and doctor's VRAM check share sent an unauthenticated
  ``/v1/models`` (a keyed server answers 401), and its cmdline fallback never read ``--n_ctx``, the flag our spawner
  passes. So on a keyed server it never knew the running window. Doctor's ``check_context_window`` held a second,
  fixed copy; there is now one probe.
- Item 5: ``file_lock`` read every ``OSError`` from the lock call as contention, so a filesystem that cannot lock
  (NFS without a lock daemon, ``ENOTSUP``) was reported after 30 s as "another process holds the key-creation lock".
- Item 6: the hot-swap refusal ("no llama-cpp-server started by this process to swap") reached the operator as a
  generic 500.
- Item 9: an end-to-end check that a lane on our own server sends that server's key on a completion.
"""

from __future__ import annotations

import errno
import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

_RED = pytest.mark.xfail(strict=True, reason="red gate: #1408 not fixed yet")

_KEY = "server-key-0123456789abcdef"


class _FakeLlamaServer:
    """A loopback stand-in for a keyed llama-cpp-server: 401 without the key, like the real one."""

    def __init__(self, models_body: dict) -> None:
        seen: list[str | None] = []
        self.seen_auth = seen

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a) -> None:
                return None

            def _reply(self, code: int, body: dict) -> None:
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                seen.append(self.headers.get("Authorization"))
                if self.headers.get("Authorization") != f"Bearer {_KEY}":
                    self._reply(401, {"error": {"message": "Invalid API key", "code": 401}})
                    return
                self._reply(200, models_body)

            def do_POST(self) -> None:  # noqa: N802
                seen.append(self.headers.get("Authorization"))
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                if self.headers.get("Authorization") != f"Bearer {_KEY}":
                    self._reply(401, {"error": {"message": "Invalid API key", "code": 401}})
                    return
                self._reply(
                    200,
                    {
                        "model": "m",
                        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    },
                )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> _FakeLlamaServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


# ─── Item 2: one keyed n_ctx probe ─────────────────────────────────────


@_RED
def test_the_n_ctx_probe_sends_the_key_and_reads_a_keyed_server() -> None:
    from maxim.runtime.leader_proxy import _current_llama_server_n_ctx

    with _FakeLlamaServer({"data": [{"id": "m", "n_ctx": 16384}]}) as srv:
        with patch("maxim.runtime.leader_proxy._llama_cmdline_args", return_value=[]):
            assert _current_llama_server_n_ctx(srv.port, api_key=_KEY) == 16384
    assert srv.seen_auth == [f"Bearer {_KEY}"]


@_RED
def test_the_n_ctx_probe_takes_the_key_as_a_required_keyword() -> None:
    """Forgetting the key must be a TypeError, not a silent 401 (CLAUDE.md: push silent no-ops into types)."""
    from maxim.runtime.leader_proxy import _current_llama_server_n_ctx

    param = inspect.signature(_current_llama_server_n_ctx).parameters["api_key"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


@_RED
@pytest.mark.parametrize(
    "args, expected",
    [
        (["python", "-m", "llama_cpp.server", "--model", "x.gguf", "--n_ctx", "16384"], 16384),
        (["python", "-m", "llama_cpp.server", "--n_ctx=12288"], 12288),
        (["llama-server", "--ctx-size", "8192"], 8192),
        (["llama-server", "-c", "4096"], 4096),
        (["llama-server", "--ctx-size=2048"], 2048),
        (["python", "-m", "llama_cpp.server", "--n_ctx", "lots"], None),
        (["python", "-m", "llama_cpp.server"], None),
    ],
    ids=["n_ctx", "n_ctx=", "ctx-size", "-c", "ctx-size=", "not-a-number", "absent"],
)
def test_the_cmdline_fallback_reads_every_n_ctx_spelling(args: list[str], expected: int | None) -> None:
    from maxim.runtime.leader_proxy import _n_ctx_from_args

    assert _n_ctx_from_args(args) == expected


@_RED
def test_the_cmdline_fallback_reads_the_argv_our_spawner_builds() -> None:
    """The producer's own argv, so a renamed spawn flag fails here rather than silently in /debug/vram."""
    from maxim.runtime.leader_proxy import _n_ctx_from_args
    from maxim.runtime.local_server_spawner import LocalServerSpawner

    for kv in ("f16", "q8_0"):
        spawner = LocalServerSpawner(model_path="/models/x.gguf", port=8100, n_ctx=24576, kv_quant_mode=kv)
        assert _n_ctx_from_args(spawner._build_cmd()) == 24576


@_RED
def test_the_proxy_debug_vram_probe_sends_the_proxys_own_key() -> None:
    import time

    from maxim.runtime.leader_proxy import ProxyAuth, _ProxyHandler

    h = object.__new__(_ProxyHandler)
    h.upstream_url = "http://127.0.0.1:8100"
    h.auth = ProxyAuth.bearer(_KEY)
    h.start_time = time.time()
    h._send_json = lambda code, body: None
    gpu = {"utilization_pct": 1.0, "vram_used_gb": 1.0, "vram_total_gb": 16.0, "temperature_c": 30.0}
    with (
        patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu),
        patch("maxim.runtime.llm_server._active_model", "qwen2.5-14b-instruct"),
        patch("maxim.runtime.leader_proxy._current_llama_server_n_ctx", return_value=8192) as probe,
    ):
        h._handle_debug_vram()
    assert probe.call_args.kwargs.get("api_key") == _KEY


@_RED
def test_doctor_probes_n_ctx_through_the_one_shared_probe_with_the_accepted_key() -> None:
    from maxim.doctor import checks

    with (
        patch.object(checks, "_accepted_server_key", return_value=("llm_server_key", _KEY, "accepted")),
        patch("maxim.models.language.maxim_peer_backend._MaximPeerBackend.health_check", return_value=True),
        patch("maxim.runtime.leader_proxy._current_llama_server_n_ctx", return_value=16384) as probe,
    ):
        result = checks.check_context_window(8100)
        assert probe.call_args.kwargs.get("api_key") == _KEY
        assert "16384" in result.message
        probe.reset_mock()
        checks._current_llama_server_n_ctx(8100)
        assert probe.call_args.kwargs.get("api_key") == _KEY


# ─── Item 5: a filesystem that cannot lock is not contention ──────────


@_RED
@pytest.mark.parametrize("code", [errno.ENOTSUP, errno.ENOLCK, errno.EBADF], ids=["ENOTSUP", "ENOLCK", "EBADF"])
def test_a_lock_the_filesystem_cannot_take_raises_lock_unavailable(tmp_path, code: int) -> None:
    from maxim.utils import process_lock

    with patch.object(process_lock, "_acquire", side_effect=OSError(code, "nope")):
        with pytest.raises(process_lock.LockUnavailable) as excinfo:
            with process_lock.file_lock(tmp_path / "x.lock", timeout_s=5.0):
                pass
    assert isinstance(excinfo.value, OSError)
    assert excinfo.value.errno == code


@pytest.mark.parametrize(
    "exc",
    [BlockingIOError(errno.EWOULDBLOCK, "held"), OSError(errno.EAGAIN, "held"), OSError(errno.EACCES, "held")],
    ids=["BlockingIOError", "EAGAIN", "EACCES"],
)
def test_control_contention_is_still_lock_contended(tmp_path, exc: OSError) -> None:
    from maxim.utils import process_lock

    with patch.object(process_lock, "_acquire", side_effect=exc):
        with pytest.raises(process_lock.LockContended):
            with process_lock.file_lock(tmp_path / "x.lock"):
                pass


@_RED
def test_a_key_whose_lock_cannot_be_taken_names_the_filesystem_not_another_process(tmp_path) -> None:
    from maxim.tunnel import keys
    from maxim.utils import process_lock

    with patch.object(process_lock, "_acquire", side_effect=OSError(errno.ENOTSUP, "Operation not supported")):
        with pytest.raises(keys.KeyFileUnusable) as excinfo:
            with keys._key_creation_lock(tmp_path / "llm_server_key"):
                pass
    assert "another process" not in excinfo.value.reason
    assert "lock" in excinfo.value.reason


def test_control_an_oserror_inside_the_key_lock_is_not_relabelled(tmp_path) -> None:
    from maxim.tunnel import keys

    with pytest.raises(PermissionError):
        with keys._key_creation_lock(tmp_path / "llm_server_key"):
            raise PermissionError(errno.EACCES, "write refused")


# ─── Item 6: the swap refusal is a 409 the operator can act on ────────


@_RED
def test_a_swap_with_no_server_of_ours_is_a_typed_refusal() -> None:
    from maxim.runtime import lane_backends
    from maxim.runtime import llm_server as _server_mod

    with patch.object(_server_mod, "_active_spawner", None):
        with pytest.raises(lane_backends.LLMSwapRefused) as excinfo:
            lane_backends._swap_key()
    assert isinstance(excinfo.value, RuntimeError)
    assert "restart maxim" in str(excinfo.value)


@_RED
def test_the_proxy_answers_a_swap_refusal_with_409_and_its_text() -> None:
    from maxim.runtime import lane_backends
    from maxim.runtime.leader_proxy import _ProxyHandler

    h = object.__new__(_ProxyHandler)
    sent: list[tuple[int, dict]] = []
    h._send_json = lambda code, body: sent.append((code, body))
    h._admin_refused = lambda action: False
    h._read_body = lambda max_size: json.dumps({"model": "qwen2.5-14b"}).encode()
    h.client_address = ("127.0.0.1", 1)
    refusal = lane_backends.LLMSwapRefused("no llama-cpp-server started by this process to swap")
    with patch("maxim.runtime.lane_backends.swap_llm_server", side_effect=refusal):
        h._handle_admin_llm_swap()
    code, body = sent[-1]
    assert code == 409
    assert "no llama-cpp-server started by this process" in body["error"]
    assert body.get("hint")


# ─── Item 9: the key a completion actually sends ──────────────────────


def test_a_lane_on_our_own_server_sends_that_servers_key_on_a_completion(monkeypatch) -> None:
    import dataclasses

    from maxim.models.language.config import LLMConfig
    from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
    from maxim.runtime.lane_backends import LaneBackendManager
    from maxim.runtime.worker_pool import LaneConfig

    monkeypatch.setenv("MAXIM_LANE_LARGE_API_KEY", "stale-or-exported")
    with _FakeLlamaServer({"data": []}) as srv:
        cfg = LaneConfig(
            name="large",
            max_workers=1,
            remote_url=f"http://127.0.0.1:{srv.port}/v1",
            remote_api_key=_KEY,
            remote_model="m",
            remote_api_key_is_server_key=True,
        )
        mgr = LaneBackendManager({"large": cfg}, max_cloud_lanes=1, max_backends=5)
        with (
            patch("maxim.models.language.config.load_llm_config", return_value=LLMConfig()),
            patch("maxim.models.language.router.LLMRouter") as router,
        ):
            mgr.get_backend("large")
        entry = router.call_args[0][0].providers["lane-large"]
        backend = _MaximPeerBackend(dataclasses.replace(LLMConfig(), providers={"lane-large": entry}), "lane-large")
        resp = backend.complete_with_usage(system="", user="hi", max_tokens=4, temperature=0.0)
    assert resp.content == "hi"
    assert srv.seen_auth == [f"Bearer {_KEY}"]
