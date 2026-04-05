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
import os
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path
from typing import Any

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
    ) -> None:
        self._model_path = str(model_path)
        self._port = int(port)
        self._bind_host = bind_host
        self._n_ctx = int(n_ctx)
        self._n_gpu_layers = int(n_gpu_layers)
        self._chat_format = chat_format
        self._api_key = api_key
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
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, *, timeout_s: float = READINESS_TIMEOUT_S) -> str | None:
        """Spawn the server and block until it responds to /v1/models.

        Returns the base URL on success, None on failure (missing model file,
        subprocess crash, readiness timeout). Failures are logged; callers
        should fall back to in-process inference.
        """
        with self._lock:
            if self.is_running:
                return self.base_url

            if not Path(self._model_path).is_file():
                print(
                    f"[warn] LocalServerSpawner: model file not found at {self._model_path}",
                    file=sys.stderr,
                )
                return None

            env = self._build_subprocess_env()
            cmd = self._build_cmd()
            try:
                # start_new_session=True puts the subprocess in its own process
                # group so terminal SIGINT (Ctrl+C) doesn't kill it. We control
                # shutdown explicitly via stop() / atexit, which means the
                # server stays alive long enough for Maxim's cleanup path
                # (e.g. sim-report LLM roundup) to make one last inference call.
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,
                )
            except (OSError, ValueError) as e:
                print(f"[warn] LocalServerSpawner: failed to start subprocess: {e}", file=sys.stderr)
                self._process = None
                return None

            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True

        if self._wait_ready(timeout_s):
            return self.base_url
        # Timed out — kill and report
        self.stop()
        print(
            f"[warn] LocalServerSpawner: server did not become ready within {timeout_s}s "
            f"(model={Path(self._model_path).name}, port={self._port})",
            file=sys.stderr,
        )
        return None

    def stop(self) -> None:
        """Terminate the subprocess if it's running. Idempotent, swallows errors."""
        with self._lock:
            proc = self._process
            self._process = None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            pass

    # ─── internals ────────────────────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", self._model_path,
            "--n_gpu_layers", str(self._n_gpu_layers),
            "--host", self._bind_host,
            "--port", str(self._port),
            "--n_ctx", str(self._n_ctx),
            "--chat_format", self._chat_format,
        ]
        if self._api_key:
            cmd.extend(["--api_key", self._api_key])
        return cmd

    def _build_subprocess_env(self) -> dict[str, str]:
        """Inherit parent env but restore CUDA_VISIBLE_DEVICES if Blackwell hid it."""
        env = os.environ.copy()
        original_cuda = get_original_cuda_devices()
        if original_cuda is not None and env.get("CUDA_VISIBLE_DEVICES") == "":
            env["CUDA_VISIBLE_DEVICES"] = original_cuda
        return env

    def _wait_ready(self, timeout_s: float) -> bool:
        """Poll the /v1/models endpoint until we get a 200 or run out of time."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not self.is_running:
                return False  # subprocess died
            if self._health_check():
                return True
            time.sleep(READINESS_POLL_INTERVAL_S)
        return False

    def _health_check(self) -> bool:
        """Probe /v1/models. When --api_key is set, llama-cpp-server requires
        Bearer auth on this endpoint too, so we send our token. A 401 also
        counts as 'server up' since it proves the HTTP listener is live
        (just rejected the key, which would be a separate bug)."""
        try:
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{self._port}/v1/models")
            if self._api_key:
                req.add_header("Authorization", f"Bearer {self._api_key}")
            with urllib.request.urlopen(req, timeout=1.0) as resp:  # noqa: S310
                return resp.status == 200
        except urllib.error.HTTPError as e:
            # 401 = server up + listening, just rejected key. Counts as "ready":
            # the spawner only needs to know the HTTP listener is alive.
            return e.code == 401
        except Exception:
            return False


__all__ = ["LocalServerSpawner", "DEFAULT_PORT", "DEFAULT_N_CTX"]
