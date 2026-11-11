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
import time

logger = logging.getLogger(__name__)
from pathlib import Path

from maxim.exceptions import LeaderRefusal, LLMServerPortBusy
from maxim.runtime.llm_spawn_record import remove_spawn_record, write_spawn_record
from maxim.utils.cancellation import child_groups, spawn_registered, stop_child_group, unregister_child
from maxim.utils.gpu_compat import get_original_cuda_devices


DEFAULT_PORT = 8100
DEFAULT_N_CTX = 8192
# 120s default — llama-cpp-server loads the model BEFORE binding HTTP, so
# health-check can't succeed until VRAM is populated. 60s is sometimes too
# tight on machines where the parent Maxim process is simultaneously
# initializing TF/PyTorch CUDA contexts, competing for GPU + disk.
# Override with MAXIM_AUTO_SPAWN_TIMEOUT_S.
READINESS_TIMEOUT_S = 120.0
READINESS_POLL_INTERVAL_S = 0.5


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
            f"rotated since that server started (restart it), or another server on port {port} (stop it). If it is a "
            f"leftover maxim server, run `maxim llm reap --port {port}` in a terminal."
        )


def probe_keyless(port: int, *, connect_s: float = 0.5, read_s: float = 2.0) -> str:
    """A keyless ``GET /v1/models`` on 127.0.0.1:<port>, classified (#1344 PR 3a; one place for this contract):
    ``"keyed"`` (401: a llama-cpp-server holding a key; llama_cpp authenticates before its generation lock, so this
    costs a busy server nothing), ``"absent"`` (connection refused), or what else the listener did, as a phrase
    ("answers without a key", "answers 403", "did not answer /v1/models (HTTPTimeout)")."""
    from maxim.utils import http as _http

    try:
        _http.fetch_url(
            f"http://127.0.0.1:{port}/v1/models",
            method="GET",
            timeout=_http.TimeoutPolicy(connect_s=connect_s, read_s=read_s, total_s=connect_s + read_s),
        )
    except _http.HTTPAuthError as e:
        status = getattr(e, "status", 401)
        return "keyed" if status == 401 else f"answers {status}"
    except _http.HTTPConnectionError:
        return "absent"
    except _http.HTTPError as e:
        return f"did not answer /v1/models ({type(e).__name__})"
    return "answers without a key"


def describe_port_listener(port: int) -> str:
    """Who listens on 127.0.0.1:<port>, as a phrase for a message: its pid and parent pid when ``lsof``/``ps`` can tell
    (POSIX; each bounded at 5 s; never gates anything), naming a server this process spawned as such."""
    pids: list[int] = []
    if sys.platform != "win32":
        try:
            out = subprocess.run(  # noqa: S603, S607
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            pids = [int(tok) for tok in out.split() if tok.isdigit()]
        except (OSError, subprocess.SubprocessError) as e:
            logger.debug("lsof could not name the listener on port %d: %s", port, e)
    if not pids:
        return f"a listener whose pid is unknown (`lsof -nP -iTCP:{port} -sTCP:LISTEN` names it)"
    ours = {child.proc.pid for child in child_groups("llm_server")}
    parts = []
    for pid in pids:
        if pid in ours:
            parts.append(f"pid {pid} (a llama-cpp-server this process spawned)")
            continue
        ppid = ""
        try:
            ppid = subprocess.run(  # noqa: S603, S607
                ["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            logger.debug("ps could not name pid %d's parent: %s", pid, e)
        orphan = " (an orphan: its parent is gone)" if ppid == "1" else ""
        parts.append(f"pid {pid}" + (f", parent {ppid}{orphan}" if ppid else ""))
    return "a listener, " + "; ".join(parts)


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
    from maxim.utils.net import primary_ipv4_address

    unauth = probe_keyless(port, connect_s=1.0, read_s=3.0)
    if unauth not in ("keyed", "answers 403"):  # a 401 or 403 refuses the keyless request
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


def served_n_ctx(port: int, *, api_key: str | None) -> int | None:
    """The context window of the llama server on 127.0.0.1:``port``, or None: the ONE n_ctx probe (#1408 item 2).

    Doctor's ``check_context_window`` and ``check_vram_pressure`` and the leader proxy's ``/v1/debug/vram`` all
    call it. In order:

    1. this process's own spawner when it serves ``port`` (``LocalServerSpawner.n_ctx``, the value the server was
       launched with: the source of truth);
    2. ``/v1/models`` with ``api_key`` (every spawned server is keyed since #1314, so a keyless probe gets 401 and
       learns nothing; the key is a required keyword so forgetting it is a TypeError, not a silent 401). It reads
       ``context_length``, ``n_ctx``, ``max_context_length`` and vLLM's ``max_model_len``;
    3. the argv of the process that serves ``port``: the pid that owns its listening socket, or a spawn record
       whose argv names ``llama_cpp.server --port <port>`` (``runtime/llm_spawn_record.py``). llama-cpp-python's
       window lives only here, since its ``/v1/models`` carries none. Never any other llama process: a foreign
       server's window would let doctor pass a server that really runs at 4096.

    Under the offline latch, step 2's ``fetch_url`` raises ``OfflineRefusal``; it is not caught here.
    """
    own = _own_spawner_n_ctx(port)
    if own:
        return own
    return _n_ctx_from_models(port, api_key) or _n_ctx_from_server_argv(port)


def _own_spawner_n_ctx(port: int) -> int | None:
    import maxim.runtime.llm_server as _server_mod

    spawner = _server_mod._active_spawner
    # Only while its child runs: a dead child's port may now hold someone else's server.
    if (
        isinstance(spawner, LocalServerSpawner)
        and spawner.port == int(port)
        and spawner.n_ctx > 0
        and spawner.is_running
    ):
        return spawner.n_ctx
    return None


def _n_ctx_from_models(port: int, api_key: str | None) -> int | None:
    from maxim.utils import http as _http

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        data = _http.fetch_url(
            f"http://127.0.0.1:{port}/v1/models",
            method="GET",
            headers=headers,
            timeout=_http.TimeoutPolicy(connect_s=1.0, read_s=2.0, total_s=3.0),
        ).json()
    except (_http.HTTPError, ValueError) as e:
        logger.debug("n_ctx probe: /v1/models on :%d gave nothing (%s)", port, type(e).__name__)
        return None
    models = data.get("data") if isinstance(data, dict) else None
    for model in models if isinstance(models, list) else []:
        if not isinstance(model, dict):
            continue
        for key in ("context_length", "n_ctx", "max_context_length", "max_model_len"):
            ctx = _window(model.get(key))
            if ctx:
                return ctx
    return None


def _window(value: object) -> int | None:
    """A positive window from an int or a decimal string (bounded), else None; a bool is not a window."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 < value < 10**9 else None
    if isinstance(value, str) and value.isascii() and value.isdigit() and len(value) <= 9:
        return int(value) or None
    return None


_N_CTX_FLAGS = ("--n_ctx", "--n-ctx", "--ctx-size", "-c")


def n_ctx_from_args(args: list[str]) -> int | None:
    """The context window a llama server's argv sets, or None: every spelling llama-cpp-python (``--n_ctx``, our
    spawner's) and the llama.cpp server (``--ctx-size``, ``-c``) accept, with a value or ``=value``."""
    for i, arg in enumerate(args):
        value: str | None = None
        if arg in _N_CTX_FLAGS and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith(tuple(f"{flag}=" for flag in _N_CTX_FLAGS if flag.startswith("--"))):
            value = arg.split("=", 1)[1]
        if value is not None:
            return int(value) if value.isascii() and value.isdigit() and len(value) <= 9 else None
    return None


def _server_argvs(port: int) -> list[list[str]]:
    """The argv of each process that serves ``port``: the listener's pid first (its argv is authoritative whatever it
    runs), then any spawn record's pid whose argv names ``llama_cpp.server --port <port>``."""
    from maxim.runtime import llm_spawn_record as _rec

    # Each step degrades to "nothing found", never a raise: the probe's contract is "or None", and doctor builds its
    # whole check list in one expression, so one raise here would end the `maxim doctor` run (a PermissionError on
    # /proc/net/tcp under SELinux or a sandbox, or from lsof, is the usual one).
    listeners: list[int] = []
    try:
        listeners = sorted(_rec.listening_pids(port).pids)
    except (_rec.ListenerScanRefused, OSError) as e:
        logger.debug("n_ctx probe: cannot name the listener on :%d (%s)", port, e)
    try:
        recorded = [r.pid for r in _rec.read_records(port) if r.pid not in listeners]
    except (_rec.RecordsNotTrusted, OSError) as e:
        logger.debug("n_ctx probe: spawn records for :%d not usable (%s)", port, e)
        recorded = []
    argvs: list[list[str]] = []
    for pid in listeners + recorded:
        argv = _rec.process_argv(pid)
        if argv and (pid in listeners or _rec.cmdline_names_server_on_port(argv, port)):
            argvs.append(argv)
    return argvs


def _n_ctx_from_server_argv(port: int) -> int | None:
    for argv in _server_argvs(port):
        n_ctx = n_ctx_from_args(argv)
        if n_ctx:
            return n_ctx
    return None


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
    def port(self) -> int:
        """The port this spawner launches (or launched) the server on (public: ``served_n_ctx`` matches on it)."""
        return self._port

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
                # Registered with the launch, as one step (#1344 PR 3a): a forced stop kills it from here on, its
                # readiness wait included, and a restart's closed registry launches nothing.
                self._process = spawn_registered(
                    lambda: subprocess.Popen(
                        cmd,
                        stdout=stream_target,
                        stderr=stream_target,
                        env=env,
                        start_new_session=True,
                    ),
                    kind="llm_server",
                    stop=self.stop,
                )
            except (OSError, ValueError) as e:
                logger.warning("LocalServerSpawner: failed to start subprocess: %s", e)
                self._process = None
                return None
            if self._process is None:
                logger.warning("LocalServerSpawner: registry closed: a restart is in progress, no server started")
                return None
            spawned_pid = self._process.pid

            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True

        # The spawn record (#1344 PR 3b): what lets an operator's `maxim llm reap` tell this server from a stranger's
        # if this process dies without stopping it. Advisory (a failure is logged, the spawn goes on), and written
        # OUTSIDE the lock: on macOS it runs `ps` (5 s bound each), and stop(), the forced-stop path, needs the lock.
        # A stop() that wins the race leaves one stale record, which the next spawn prunes.
        write_spawn_record(pid=spawned_pid, port=self._port, model_path=self._model_path)

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
        """Stop the server's whole process group and reap it. Idempotent, swallows errors.

        The one claimer of its process (#1344 PR 3a): the process is taken under the lock, so a second stopper finds
        nothing. On POSIX the group gets SIGTERM, then SIGKILL only while the leader is still unreaped (it holds the
        pgid, so the group cannot be a stranger's); nothing is signalled after the reap. Its registry entry goes last,
        so a forced stop landing mid-stop still reaches it. Its own spawn record goes once the leader is reaped (one
        still alive keeps the record that lets an operator reap it, #1344 PR 3b).
        """
        with self._lock:
            proc = self._process
            self._process = None
        if proc is None:
            return
        try:
            stop_child_group(proc)
        except Exception:
            pass
        finally:
            unregister_child(proc)
            if proc.poll() is not None:
                remove_spawn_record(self._port, proc.pid)

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
    "LLMServerPortBusy",
    "describe_port_listener",
]
