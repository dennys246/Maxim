"""Auto-spawn a local llama-cpp-server subprocess (multi-LLM Phase 6a).

LocalServerSpawner starts a llama-cpp-python HTTP server as a child process,
waits for readiness, and tears it down on Maxim exit. Removes the "run the
server in a second terminal" step — users get GPU-served inference from a
single `maxim` invocation.

Design:
- Spawner binds to 127.0.0.1 by default (solo-mode) or 0.0.0.0 in leader mode.
- Subprocess inherits CUDA_VISIBLE_DEVICES from the original pre-Blackwell
  value when present (see gpu_compat.get_original_cuda_devices), so the
  server can use the GPU even when the parent process had CUDA hidden.
- atexit handler kills the subprocess on Python interpreter shutdown.
- Health check: poll GET /v1/models until the server responds or timeout.
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import sys
import threading
from typing import Any
import time

logger = logging.getLogger(__name__)
from pathlib import Path

from maxim.exceptions import LeaderRefusal
from maxim.utils.gpu_compat import get_original_cuda_devices


import signal

DEFAULT_PORT = 8100
DEFAULT_N_CTX = 8192
# 120s default — llama-cpp-server loads the model BEFORE binding HTTP, so
# health-check can't succeed until VRAM is populated. 60s is sometimes too
# tight on machines where the parent Maxim process is simultaneously
# initializing TF/PyTorch CUDA contexts, competing for GPU + disk.
# Override with MAXIM_AUTO_SPAWN_TIMEOUT_S.
READINESS_TIMEOUT_S = 120.0
READINESS_POLL_INTERVAL_S = 0.5


def _kill_process_tree(pid: int) -> None:
    """Best-effort kill of a process and all its children.

    Uses psutil when available (handles the full tree reliably on all
    platforms). Falls back to os.kill / taskkill for the common case
    where psutil isn't installed.
    """
    try:
        import psutil

        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        # Kill children first, then parent
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        # Reap zombies
        psutil.wait_procs(children + [parent], timeout=3)
        return
    except ImportError:
        pass

    # Fallback: no psutil — platform-specific best-effort.
    # Use os-level APIs to avoid going through subprocess.Popen (which may
    # be mocked in tests and is heavier than needed here).
    if sys.platform == "win32":
        # taskkill /F /T /PID kills the process tree on Windows.
        # Use os.system to avoid subprocess.Popen (keeps test mocks clean).
        try:
            os.system(f"taskkill /F /T /PID {int(pid)} >nul 2>&1")  # noqa: S605
        except Exception:
            pass
    else:
        # Unix: kill the process group if it was started with start_new_session
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def find_stale_llm_servers(port: int = DEFAULT_PORT) -> list[dict[str, Any]]:
    """Find llama-cpp-server processes that may be holding VRAM.

    Returns a list of dicts with keys: pid, cmdline, port_match, status.
    Works with or without psutil (degrades to port-only check without it).
    """
    results: list[dict[str, Any]] = []
    try:
        import psutil
    except ImportError:
        # No psutil — best-effort: check if something is listening on the port
        try:
            from maxim.utils import http as _http

            resp = _http.fetch_url(
                f"http://127.0.0.1:{port}/v1/models",
                method="GET",
                timeout=_http.TimeoutPolicy(connect_s=0.5, read_s=1.0, total_s=1.5),
            )
            if resp.status == 200:
                results.append(
                    {
                        "pid": None,
                        "cmdline": f"(unknown process on port {port})",
                        "port_match": True,
                        "status": "responding",
                    }
                )
        except Exception:
            pass
        return results

    for proc in psutil.process_iter(["pid", "cmdline", "status"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmd_str = " ".join(cmdline)
            if "llama_cpp.server" in cmd_str or "llama_cpp/server" in cmd_str:
                port_match = "--port" in cmd_str and str(port) in cmd_str
                results.append(
                    {
                        "pid": proc.info["pid"],
                        "cmdline": cmd_str[:200],
                        "port_match": port_match,
                        "status": proc.info.get("status", "unknown"),
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return results


def kill_stale_llm_servers(port: int = DEFAULT_PORT) -> int:
    """Find and kill any stale llama-cpp-server processes holding VRAM.

    Returns the number of processes killed. Safe to call at startup or
    when VRAM appears stuck — only targets llama_cpp.server processes.
    """
    stale = find_stale_llm_servers(port)
    killed = 0
    for entry in stale:
        pid = entry.get("pid")
        if pid is None:
            # No psutil — find PID from port listener, platform-specific
            if sys.platform == "win32":
                try:
                    out = subprocess.check_output(  # noqa: S603, S607
                        ["netstat", "-ano"],
                        text=True,
                        timeout=5,
                    )
                    for line in out.splitlines():
                        if f":{port}" in line and "LISTENING" in line:
                            parts = line.split()
                            try:
                                _kill_process_tree(int(parts[-1]))
                                killed += 1
                            except (ValueError, IndexError):
                                pass
                except Exception:
                    pass
            else:
                try:
                    out = subprocess.check_output(  # noqa: S603, S607
                        ["lsof", "-ti", f":{port}"],
                        text=True,
                        timeout=5,
                    )
                    for lpid_str in out.strip().split():
                        try:
                            _kill_process_tree(int(lpid_str))
                            killed += 1
                        except (ValueError, OSError):
                            pass
                except Exception:
                    pass
        else:
            _kill_process_tree(pid)
            killed += 1
    return killed


_SCRUBBED_ENV = frozenset({"HOST", "PORT", "CONFIG_FILE"})


class LLMServerExposureError(LeaderRefusal):
    """A llama-cpp-server on our port is reachable more widely, or less authenticated, than intended."""


class LLMServerKeyRejected(LLMServerExposureError):
    """The server on our port answers 401 to the key we hold (#1314): it is not ours, or our key never reached it.

    Not "ready": routing to it would 401 every call. The likely causes are named in the message: this process
    resolved a different role from the one that started the server (a leader's server holds the mesh key, any other
    role's the local server key), the mesh key was rotated after the server started, or another server holds the port.
    """

    def __init__(self, port: int) -> None:
        super().__init__(
            f"the llama-cpp-server on 127.0.0.1:{port} rejects this process's key (401). Likely: a role mismatch "
            "(a leader's server holds the mesh key, a solo's the local server key: check MAXIM_ROLE), a mesh key "
            f"rotated since that server started (restart it), or another server on port {port} (stop it)."
        )


def verify_llm_server_exposure(port: int, *, allow_authenticated_lan: bool) -> None:
    """Check the server on ``127.0.0.1:<port>`` from the outside, or raise :class:`LLMServerExposureError`.

    Two observations, not configuration: (1) an unauthenticated ``/v1/models`` must be
    refused with 401, always: every spawned server holds a key since #1314 (a key the server
    ignored, or a server an older maxim started without one, would otherwise fail open
    silently); (2) the port must not accept connections on this host's LAN address unless
    ``allow_authenticated_lan`` (the reuse of a server an older maxim bound to every interface
    with a key: warned, not refused).

    The LAN half is defense in depth: it probes the default-route address only
    (not every interface or a VPN bridge). The real controls are the loopback
    ``--host`` and the scrubbed ``HOST``/``CONFIG_FILE`` env.
    """
    from maxim.utils import http as _http
    from maxim.utils.net import primary_ipv4_address

    try:
        _http.fetch_url(
            f"http://127.0.0.1:{port}/v1/models",
            method="GET",
            timeout=_http.TimeoutPolicy(connect_s=1.0, read_s=3.0, total_s=4.0),
        )
        unauth = "open"
    except _http.HTTPAuthError:
        unauth = "refused"
    except _http.HTTPError as e:
        unauth = f"unknown ({type(e).__name__})"
    if unauth != "refused":
        raise LLMServerExposureError(
            f"the llama-cpp-server on port {port} answered an unauthenticated request ({unauth}): it holds no key "
            "(one an older maxim started?) or ignored the key it was given. Refusing to serve inference without "
            f"auth: stop the server on port {port}; the next start spawns a keyed one."
        )
    lan = primary_ipv4_address()
    if lan is None:
        return
    try:
        with socket.create_connection((lan, port), timeout=0.5):
            reachable = True
    except OSError:
        reachable = False
    if not reachable:
        return
    if not allow_authenticated_lan:
        raise LLMServerExposureError(
            f"llama-cpp-server on port {port} should listen on loopback only but accepts connections on {lan} "
            "(a HOST/CONFIG_FILE env override?)"
        )
    logger.warning(
        "llama-cpp-server on port %d listens on the LAN address %s (it demands auth). It was started by an "
        "older maxim; restart it so it binds loopback only.",
        port,
        lan,
    )


class LocalServerSpawner:
    """Manage a llama-cpp-python server subprocess.

    Lifecycle:
        spawner = LocalServerSpawner(model_path=..., port=8100, bind_host="127.0.0.1")
        url = spawner.start()  # returns base URL or None on failure
        ...
        spawner.stop()         # called manually or via atexit
    """

    def __init__(
        self,
        *,
        model_path: str,
        port: int = DEFAULT_PORT,
        bind_host: str = "127.0.0.1",
        n_ctx: int = DEFAULT_N_CTX,
        n_gpu_layers: int = -1,
        chat_format: str = "chatml",
        api_key: str | None = None,
        kv_quant_mode: str = "f16",
    ) -> None:
        self._model_path = str(model_path)
        self._port = int(port)
        self._bind_host = bind_host
        self._n_ctx = int(n_ctx)
        self._n_gpu_layers = int(n_gpu_layers)
        self._chat_format = chat_format
        self._api_key = api_key
        self._key_rejected = False  # set by _health_check on a 401 to our key (#1314)
        self._kv_quant_mode = kv_quant_mode
        self._process: subprocess.Popen | None = None
        self._atexit_registered = False
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL to use as a lane's remote_url."""
        # Callers connect via loopback even when bind_host=0.0.0.0 (that's
        # for peers). Use 127.0.0.1 for the local client.
        return f"http://127.0.0.1:{self._port}/v1"

    @property
    def n_ctx(self) -> int:
        """The context window this spawner launched (or will launch) the
        server with. Public because the auto-spawn reuse path stamps it as
        the lane's served_n_ctx — a private-attr getattr there would turn a
        rename into a silent no-op (two-lens fold 2026-08-09)."""
        return self._n_ctx

    @property
    def model_path(self) -> str:
        """The GGUF this spawner launches the server with (public for the same reason as :attr:`n_ctx`)."""
        return self._model_path

    @property
    def kv_cache_type(self) -> str:
        """The KV-cache type the spawn command passes: ``q8_0``/``q4_0`` as ``--type_k``/``--type_v``, else
        llama-cpp-server's default ``f16`` (no flag). :meth:`_build_cmd` reads this, so the stamp and the
        command cannot disagree."""
        return self._kv_quant_mode if self._kv_quant_mode in ("q8_0", "q4_0") else "f16"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, *, timeout_s: float = READINESS_TIMEOUT_S) -> str | None:
        """Spawn the server and block until it responds to /v1/models.

        Returns the base URL on success, None on failure (missing model file,
        subprocess crash, readiness timeout). Failures are logged; callers
        should fall back to in-process inference. A server that came up but is
        exposed beyond what was asked (see :func:`verify_llm_server_exposure`)
        is stopped and raises :class:`LLMServerExposureError` instead: that is
        not an availability failure to fall back from.
        """
        with self._lock:
            if self.is_running:
                return self.base_url

            if not Path(self._model_path).is_file():
                logger.warning("LocalServerSpawner: model file not found at %s", self._model_path)
                return None

            env = self._build_subprocess_env()
            cmd = self._build_cmd()
            # Stage A observability: MAXIM_TUNNEL_ECHO=1 keeps llama-cpp-server's
            # stdout/stderr visible so users can see inbound HTTP requests
            # (uvicorn access logs + request bodies) in real time. Default path
            # keeps streams hidden to avoid flooding the terminal.
            echo_enabled = os.environ.get("MAXIM_TUNNEL_ECHO", "").strip().lower() in (
                "1",
                "true",
                "t",
                "yes",
                "y",
                "on",
            )
            stream_target = None if echo_enabled else subprocess.DEVNULL
            if echo_enabled:
                logger.info("MAXIM_TUNNEL_ECHO=1: llama-cpp-server stdout/stderr will stream to this terminal")
            try:
                # start_new_session=True puts the subprocess in its own process
                # group so terminal SIGINT (Ctrl+C) doesn't kill it. We control
                # shutdown explicitly via stop() / atexit, which means the
                # server stays alive long enough for Maxim's cleanup path
                # (e.g. sim-report LLM roundup) to make one last inference call.
                self._process = subprocess.Popen(
                    cmd,
                    stdout=stream_target,
                    stderr=stream_target,
                    env=env,
                    start_new_session=True,
                )
            except (OSError, ValueError) as e:
                logger.warning("LocalServerSpawner: failed to start subprocess: %s", e)
                self._process = None
                return None

            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True

        try:
            ready = self._wait_ready(timeout_s)
            if ready:
                verify_llm_server_exposure(self._port, allow_authenticated_lan=False)
        except LLMServerExposureError:  # LLMServerKeyRejected included: a 401 to our own key is not "ready"
            self.stop()
            raise
        if ready:
            return self.base_url
        # Timed out — kill and report
        self.stop()
        logger.warning(
            "LocalServerSpawner: server did not become ready within %ds (model=%s, port=%d)",
            timeout_s,
            Path(self._model_path).name,
            self._port,
        )
        return None

    def stop(self) -> None:
        """Terminate the subprocess if it's running. Idempotent, swallows errors.

        Kills the entire process tree so child workers (uvicorn, CUDA kernels)
        release GPU memory even when the top-level process exits cleanly but
        its children linger.
        """
        with self._lock:
            proc = self._process
            self._process = None
        if proc is None:
            return
        pid = proc.pid
        try:
            # Phase 1: graceful SIGTERM (or TerminateProcess on Windows)
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Phase 2: SIGKILL the main process
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            pass

        # Phase 3: kill any orphaned children that survived the top-level
        # kill (e.g. uvicorn workers holding CUDA contexts).  Best-effort.
        _kill_process_tree(pid)

    # ─── internals ────────────────────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "llama_cpp.server",
            "--model",
            self._model_path,
            "--n_gpu_layers",
            str(self._n_gpu_layers),
            "--host",
            self._bind_host,
            "--port",
            str(self._port),
            "--n_ctx",
            str(self._n_ctx),
            "--chat_format",
            self._chat_format,
        ]
        # The key travels in the child's env (API_KEY), never argv: argv is
        # world-readable through `ps` on a shared host.
        # KV cache quantization (peer_leader_flexibility_plan P4c). f16 is
        # llama-cpp-server's default, so only emit flags for q8_0/q4_0 — keeps
        # the spawn cmdline backward-compatible with older llama_cpp_python
        # builds that didn't recognize --type_k/--type_v.
        if self.kv_cache_type != "f16":
            cmd.extend(
                [
                    "--type_k",
                    self.kv_cache_type,
                    "--type_v",
                    self.kv_cache_type,
                ]
            )
        return cmd

    def _build_subprocess_env(self) -> dict[str, str]:
        """Inherit the parent env, minus anything that would override where the server listens or its key.

        llama_cpp.server reads ``HOST``/``PORT`` straight from the env over ``--host``/``--port``
        (``uvicorn.run(host=os.getenv("HOST", ...))``), lets ``CONFIG_FILE`` replace every
        setting, and its pydantic settings read ``api_key`` from the env in any case. An
        inherited ``HOST=0.0.0.0`` (common in containers) would silently rebind a loopback
        server to every interface, so those are dropped and ``API_KEY`` is set from ours.
        Also restores ``CUDA_VISIBLE_DEVICES`` if Blackwell hid it.
        """
        env = os.environ.copy()
        for name in list(env):
            if name in _SCRUBBED_ENV or name.lower() == "api_key":
                del env[name]
        if self._api_key:
            env["API_KEY"] = self._api_key
        original_cuda = get_original_cuda_devices()
        if original_cuda is not None and env.get("CUDA_VISIBLE_DEVICES") == "":
            env["CUDA_VISIBLE_DEVICES"] = original_cuda
        return env

    def _wait_ready(self, timeout_s: float) -> bool:
        """Poll the /v1/models endpoint until we get a 200 or run out of time."""
        deadline = time.time() + timeout_s
        self._key_rejected = False
        while time.time() < deadline:
            if not self.is_running:
                return False  # subprocess died
            if self._health_check():
                return True
            if self._key_rejected:
                raise LLMServerKeyRejected(self._port)
            time.sleep(READINESS_POLL_INTERVAL_S)
        return False

    def _health_check(self) -> bool:
        """Probe /v1/models with our key (llama-cpp-server requires it on this endpoint too). Ready only on a 200.
        A 401 is NOT ready (#1314, owner decision 2026-10-20): a listener that rejects the key we spawned it with
        is not the server we started, or never got the key. It is recorded in ``_key_rejected`` so ``_wait_ready``
        stops at once with :class:`LLMServerKeyRejected` rather than polling until the timeout."""
        from maxim.utils import http as _http

        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = _http.fetch_url(
                f"http://127.0.0.1:{self._port}/v1/models",
                method="GET",
                headers=headers,
                timeout=_http.TimeoutPolicy(connect_s=0.5, read_s=1.0, total_s=1.5),
            )
            return resp.status == 200
        except _http.HTTPAuthError:
            self._key_rejected = True
            return False
        except Exception:
            return False


__all__ = [
    "LLMServerExposureError",
    "LLMServerKeyRejected",
    "LocalServerSpawner",
    "verify_llm_server_exposure",
    "DEFAULT_PORT",
    "DEFAULT_N_CTX",
    "find_stale_llm_servers",
    "kill_stale_llm_servers",
]
