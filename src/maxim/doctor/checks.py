"""Individual diagnostic checks run by `maxim doctor`.

Each check is a pure function: takes `PlatformInfo`, returns a `CheckResult`.
Results include a status, message, and platform-specific fix hint. The fix
hint is rendered verbatim in the doctor output — it should be copy-pasteable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from maxim.doctor.platform_detect import PlatformInfo, detect_wsl_ip

logger = logging.getLogger(__name__)

Status = Literal["ok", "warn", "fail", "info"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    fix: str | None = None  # platform-specific fix hint
    retry_id: str | None = None  # only set when retry is meaningful

    @property
    def symbol(self) -> str:
        return {"ok": "✓", "warn": "⚠", "fail": "✗", "info": "ℹ"}[self.status]


# ─── environment checks ────────────────────────────────────────────────────


def check_gpu() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="GPU / CUDA",
            status="warn",
            message="torch not installed — CPU-only mode",
            fix="Install GPU stack: pip install -e '.[llm-torch]'",
        )
    if not torch.cuda.is_available():
        import os

        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_vis == "":
            return CheckResult(
                name="GPU / CUDA",
                status="warn",
                message="CUDA hidden (CUDA_VISIBLE_DEVICES=''). Likely the Blackwell workaround.",
                fix=(
                    "If you have a Blackwell GPU (RTX 50xx), current behavior keeps CUDA\n"
                    "  enabled by default. If it's hidden, set: unset MAXIM_BLACKWELL_HIDE_CUDA"
                ),
            )
        return CheckResult(
            name="GPU / CUDA",
            status="warn",
            message="No CUDA device available — inference will be CPU-only",
        )
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    return CheckResult(
        name="GPU / CUDA",
        status="ok",
        message=f"{props.name} ({vram_gb:.1f} GB VRAM)",
    )


def check_tier_detection(caps=None) -> CheckResult:
    """Report which capability tiers are available on this hardware."""
    try:
        from maxim.runtime.capabilities import RuntimeCapabilities, detect_compute_resources
        from maxim.runtime.lane_models import detect_tiers

        if caps is None:
            has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
            caps = RuntimeCapabilities(
                has_gpu=has_gpu,
                gpu_type=gpu_type,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
            )
        tiers = detect_tiers(caps)
    except ImportError as e:
        return CheckResult(
            name="LLM Tiers",
            status="warn",
            message=f"Tier detection unavailable: {e}",
            fix="pip install -e '.[llm-llama]'  # or install pymaxim[llm-llama]",
        )
    except Exception as e:
        return CheckResult(
            name="LLM Tiers",
            status="warn",
            message=f"Tier detection failed: {e}",
        )

    tier_names = sorted(tiers.keys())
    profiles = {name: cfg.model_profile for name, cfg in tiers.items()}

    if "large" in tiers or "medium" in tiers:
        return CheckResult(
            name="LLM Tiers",
            status="ok",
            message=f"Tiers: {', '.join(tier_names)}. Profiles: {profiles}",
        )
    return CheckResult(
        name="LLM Tiers",
        status="warn",
        message=f"Only 'small' tier detected ({caps.ram_gb:.0f}GB RAM, GPU: {caps.gpu_type or 'none'})",
        fix=(
            "Agent inference needs a large or medium tier. Options:\n"
            "  --language-model mistral-7b          # if you have 8+ GB RAM\n"
            "  --cloud-fallback claude-sonnet       # use cloud for inference\n"
            "  --tier-model large=<remote-url>      # point to a remote leader"
        ),
        retry_id="tier_detection",
    )


def check_llama_cpp_server_installed() -> CheckResult:
    try:
        import llama_cpp.server  # noqa: F401
    except ImportError:
        return CheckResult(
            name="llama-cpp-server installed",
            status="fail",
            message="llama_cpp.server not installed — auto-spawn disabled",
            fix="pip install -e '.[llm-server]'",
        )
    return CheckResult(
        name="llama-cpp-server installed",
        status="ok",
        message="ready for auto-spawn",
    )


def check_server_reachable(port: int = 8100) -> CheckResult:
    """Probe the local auto-spawn port.

    Plan 3 R2.6: the standalone ``llm_server_responding_at`` was removed.
    The ``_llm_server_responding_at`` wrapper in lane_backends now
    delegates to ``_MaximPeerBackend.health_check`` with
    ``enable_stage2=False`` — same bool return shape, one probe
    implementation under the hood.
    """
    from maxim.runtime.lane_backends import _llm_server_responding_at

    url = f"http://127.0.0.1:{port}/v1"
    if _llm_server_responding_at(url, timeout_s=2.0):
        return CheckResult(
            name="Auto-spawn server",
            status="ok",
            message=f"responding at {url}",
            retry_id="server",
        )
    return CheckResult(
        name="Auto-spawn server",
        status="warn",
        message=f"nothing responding at {url}",
        fix="Run `maxim` in another terminal — auto-spawn will launch one.",
        retry_id="server",
    )


def check_llm_model_active() -> CheckResult:
    """Check if an LLM model is loaded and report which one."""
    try:
        # Must read `_active_model` through `_server_mod` — importing it by
        # name binds the value at import time and diverges from the live
        # state (CLAUDE.md "Mutable globals + module extraction" lesson).
        # This was a silent bug: the check only ever reported the persisted
        # model name, never the live one.
        import maxim.runtime.llm_server as _server_mod
        from maxim.runtime.lane_backends import _read_persisted_model

        active = _server_mod._active_model
        persisted = _read_persisted_model()

        if active:
            return CheckResult(
                name="LLM model",
                status="ok",
                message=f"active model: {active}",
            )
        if persisted:
            return CheckResult(
                name="LLM model",
                status="warn",
                message=f"persisted model: {persisted} (not yet loaded)",
                fix=f"maxim peer llm {persisted}",
                retry_id="llm_model",
            )
        return CheckResult(
            name="LLM model",
            status="warn",
            message="no model configured — use `maxim peer llm <model>` to set one",
            fix="maxim peer llm qwen2.5-14b",
            retry_id="llm_model",
        )
    except Exception:
        return CheckResult(
            name="LLM model",
            status="info",
            message="model tracking not available (solo mode)",
        )


# ─── environment variable config ───────────────────────────────────────────


def check_env_config(info: PlatformInfo, role: str | None = None) -> list["CheckResult"]:
    """Validate critical Maxim environment variables.

    Returns a list (may be empty if everything is fine) so the caller can
    splice the results into the Environment section without a fixed slot.
    Checks are cross-platform — only uses ``os.environ``, no shell calls.

    Covers:
    - MAXIM_ROLE missing or set to an invalid value
    - MAXIM_LLM_ENABLED not set on a leader/solo machine
    - MAXIM_LLM_PROFILE missing when LLM is enabled
    - MAXIM_LLM_N_CTX not set (context overflow risk on 14B+ models)
    - Stale MAXIM_PEER_PROBE_KEY from pre-R2.5 (now ignored, misleading)
    - MAXIM_SKIP_REMOTE_PROBE left set from debugging (silently disables health probes)
    - MAXIM_ROLE=peer set on a machine running as leader (role_divergence trigger)
    """
    import os

    results: list[CheckResult] = []
    is_peer = (role == "peer") or (os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL", "").strip() != "")

    # ── MAXIM_ROLE ────────────────────────────────────────────────────────────
    maxim_role = os.environ.get("MAXIM_ROLE", "").strip().lower()
    valid_roles = {"leader", "peer", "solo"}
    if not maxim_role:
        # Role is inferred from heuristics — warn so the user knows
        # which heuristic fired and can make it explicit.
        inferred = "peer" if is_peer else "leader"
        # MAXIM_ROLE is normally exported by detect_and_apply_role() at startup —
        # this branch fires only if that path was bypassed (e.g. direct import).
        # Don't suggest .zshrc for peer (auto-detected from peer.yml); for leader
        # the systemd unit Environment= is the right persistent location.
        if info.os == "macos":
            export_cmd = f"export MAXIM_ROLE={inferred}  # current session; auto-detected on normal startup"
        else:
            export_cmd = f"export MAXIM_ROLE={inferred}  # or set in systemd unit Environment="
        results.append(
            CheckResult(
                name="MAXIM_ROLE",
                status="warn",
                message=f"not set — role inferred as '{inferred}' from heuristics. Normally auto-exported at startup; check that cli.py::main ran before this check.",
                fix=export_cmd,
            )
        )
    elif maxim_role not in valid_roles:
        results.append(
            CheckResult(
                name="MAXIM_ROLE",
                status="fail",
                message=f"invalid value '{maxim_role}' — must be leader, peer, or solo",
                fix="export MAXIM_ROLE=leader  # or peer / solo",
            )
        )
    else:
        results.append(
            CheckResult(
                name="MAXIM_ROLE",
                status="ok",
                message=f"MAXIM_ROLE={maxim_role}",
            )
        )

    # ── LLM enablement (leader/solo only) ────────────────────────────────────
    if not is_peer:
        llm_enabled = os.environ.get("MAXIM_LLM_ENABLED", "").strip()
        if llm_enabled not in ("1", "true", "yes"):
            results.append(
                CheckResult(
                    name="MAXIM_LLM_ENABLED",
                    status="warn",
                    message="not set — LLM inference may be disabled",
                    fix="export MAXIM_LLM_ENABLED=1",
                )
            )

        # ── MAXIM_LLM_PROFILE ────────────────────────────────────────────────
        profile = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
        if not profile:
            # Check persisted file before crying foul
            try:
                from maxim.runtime.lane_backends import _read_persisted_model

                persisted = _read_persisted_model()
            except Exception:
                persisted = None
            if persisted:
                results.append(
                    CheckResult(
                        name="MAXIM_LLM_PROFILE",
                        status="info",
                        message=f"not set — using persisted model '{persisted}'",
                        fix=f"export MAXIM_LLM_PROFILE={persisted}  # makes it explicit",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="MAXIM_LLM_PROFILE",
                        status="warn",
                        message="not set and no persisted model — LLM will not start",
                        fix="export MAXIM_LLM_PROFILE=qwen2.5-14b  # or your model name",
                    )
                )

        # ── MAXIM_LLM_N_CTX ──────────────────────────────────────────────────
        n_ctx_raw = os.environ.get("MAXIM_LLM_N_CTX", "").strip()
        if not n_ctx_raw:
            results.append(
                CheckResult(
                    name="MAXIM_LLM_N_CTX",
                    status="warn",
                    message=(
                        "not set — llama-cpp will auto-select context size (often 4096). "
                        "Long prompts on 14B+ models will fill the KV cache and return "
                        "empty choices, causing inference_broken cascades."
                    ),
                    fix="export MAXIM_LLM_N_CTX=16384  # safe for Q4_K_M 14B on 16 GB+ VRAM",
                )
            )
        else:
            try:
                n_ctx = int(n_ctx_raw)
                if n_ctx < 8192:
                    results.append(
                        CheckResult(
                            name="MAXIM_LLM_N_CTX",
                            status="warn",
                            message=f"MAXIM_LLM_N_CTX={n_ctx} — below 8192 risks context overflow on multi-turn sims",
                            fix="export MAXIM_LLM_N_CTX=16384",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            name="MAXIM_LLM_N_CTX",
                            status="ok",
                            message=f"MAXIM_LLM_N_CTX={n_ctx}",
                        )
                    )
            except ValueError:
                results.append(
                    CheckResult(
                        name="MAXIM_LLM_N_CTX",
                        status="fail",
                        message=f"MAXIM_LLM_N_CTX='{n_ctx_raw}' is not a valid integer",
                        fix="export MAXIM_LLM_N_CTX=16384",
                    )
                )

    # ── Stale debugging vars that cause silent failures ───────────────────────
    if os.environ.get("MAXIM_SKIP_REMOTE_PROBE", "").strip().lower() in ("1", "true", "yes"):
        results.append(
            CheckResult(
                name="MAXIM_SKIP_REMOTE_PROBE",
                status="warn",
                message="set — all health probes are bypassed. Inference will fire before the server is ready.",
                fix="unset MAXIM_SKIP_REMOTE_PROBE",
            )
        )

    if os.environ.get("MAXIM_PEER_PROBE_KEY", "").strip():
        results.append(
            CheckResult(
                name="MAXIM_PEER_PROBE_KEY",
                status="warn",
                message="set — this variable was removed in Plan 3 R2.5 (probe key moved to instance level). It is ignored but may indicate a stale environment.",
                fix="unset MAXIM_PEER_PROBE_KEY",
            )
        )

    return results


# ─── context window ─────────────────────────────────────────────────────────


def check_context_window(port: int = 8100) -> CheckResult:
    """Detect the context window actually in use by the running llama-cpp server.

    Queries ``/v1/models`` first (cheap). If that doesn't carry n_ctx, falls
    back to inspecting the process command-line on Linux (``/proc/*/cmdline``)
    and macOS/Windows (``ps``/``wmic``). Returns info if server isn't running.

    This check exists specifically to catch the silent inference_broken cascade
    where llama-cpp runs at n_ctx=4096 and long prompts overflow the KV cache,
    returning empty ``choices`` with HTTP 200.
    """
    import platform
    import subprocess

    url = f"http://127.0.0.1:{port}/v1"

    # Step 1: is the server up at all?
    try:
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        backend = _MaximPeerBackend.for_url(url)
        healthy = backend.health_check(enable_stage2=False)
        if not healthy:
            return CheckResult(
                name="Context window (n_ctx)",
                status="info",
                message="llama-cpp server not reachable — start it first",
            )
    except Exception:
        return CheckResult(
            name="Context window (n_ctx)",
            status="info",
            message="context window check unavailable (server not reachable)",
        )

    # Step 2: check /v1/models for n_ctx in metadata
    n_ctx: int | None = None
    try:
        import socket

        req_bytes = (f"GET /v1/models HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n").encode()
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
            s.sendall(req_bytes)
            raw = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                raw += chunk
        body = raw.split(b"\r\n\r\n", 1)[-1]
        import json as _json

        data = _json.loads(body)
        for model in data.get("data", []):
            ctx = model.get("context_length") or model.get("n_ctx") or model.get("max_context_length")
            if ctx:
                n_ctx = int(ctx)
                break
    except Exception:
        pass

    # Step 3: inspect process args (cross-platform, best-effort)
    if n_ctx is None:
        try:
            system = platform.system().lower()
            if system == "linux":
                # /proc is authoritative and doesn't require ps
                import glob

                for cmdline_path in glob.glob("/proc/*/cmdline"):
                    try:
                        with open(cmdline_path, "rb") as f:
                            args = f.read().split(b"\x00")
                        args_str = [a.decode("utf-8", errors="ignore") for a in args]
                        if any("llama" in a or "llama-server" in a for a in args_str):
                            for i, arg in enumerate(args_str):
                                if arg in ("--ctx-size", "-c", "--n-ctx") and i + 1 < len(args_str):
                                    n_ctx = int(args_str[i + 1])
                                    break
                                if arg.startswith("--ctx-size="):
                                    n_ctx = int(arg.split("=", 1)[1])
                                    break
                    except Exception:
                        continue
                    if n_ctx is not None:
                        break
            elif system == "darwin":
                out = subprocess.check_output(
                    ["ps", "aux"],
                    timeout=3.0,
                    stderr=subprocess.DEVNULL,
                ).decode("utf-8", errors="ignore")
                for line in out.splitlines():
                    if "llama" not in line:
                        continue
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p in ("--ctx-size", "-c", "--n-ctx") and i + 1 < len(parts):
                            n_ctx = int(parts[i + 1])
                            break
                        if p.startswith("--ctx-size="):
                            n_ctx = int(p.split("=", 1)[1])
                            break
                    if n_ctx is not None:
                        break
        except Exception:
            pass

    if n_ctx is None:
        return CheckResult(
            name="Context window (n_ctx)",
            status="warn",
            message=(
                "server is running but n_ctx could not be determined. Set MAXIM_LLM_N_CTX=16384 to make it explicit."
            ),
            fix="export MAXIM_LLM_N_CTX=16384",
        )

    if n_ctx < 8192:
        return CheckResult(
            name="Context window (n_ctx)",
            status="warn",
            message=(
                f"n_ctx={n_ctx} — too small for multi-turn sims with memory summaries. "
                f"Long prompts will overflow the KV cache and return empty choices "
                f"(inference_broken cascade)."
            ),
            fix="export MAXIM_LLM_N_CTX=16384  # then restart: maxim peer restart",
            retry_id="ctx_window",
        )

    return CheckResult(
        name="Context window (n_ctx)",
        status="ok",
        message=f"n_ctx={n_ctx}",
    )


# ─── VRAM spillover (Plan 3.6 R5) ──────────────────────────────────────────


def _current_llama_server_n_ctx(port: int) -> int | None:
    """Return the n_ctx the running llama-cpp-server is configured for, or None.

    Reuses the detection strategy from :func:`check_context_window`: first
    ``/v1/models`` metadata, then process command-line inspection. Kept as a
    separate helper so :func:`check_vram_pressure` doesn't re-run the network
    probe redundantly if the context window check already ran.
    """
    import socket

    try:
        req_bytes = (f"GET /v1/models HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n").encode()
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
            s.sendall(req_bytes)
            raw = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                raw += chunk
        body = raw.split(b"\r\n\r\n", 1)[-1]
        import json as _json

        data = _json.loads(body)
        for model in data.get("data", []):
            ctx = model.get("context_length") or model.get("n_ctx") or model.get("max_context_length")
            if ctx:
                return int(ctx)
    except Exception:
        pass

    # Process args fallback — best-effort, cross-platform
    try:
        import platform as _platform

        system = _platform.system().lower()
        if system == "linux":
            import glob

            for cmdline_path in glob.glob("/proc/*/cmdline"):
                try:
                    with open(cmdline_path, "rb") as f:
                        args = f.read().split(b"\x00")
                    args_str = [a.decode("utf-8", errors="ignore") for a in args]
                    if any("llama" in a for a in args_str):
                        for i, arg in enumerate(args_str):
                            if arg in ("--ctx-size", "-c", "--n-ctx") and i + 1 < len(args_str):
                                return int(args_str[i + 1])
                            if arg.startswith("--ctx-size="):
                                return int(arg.split("=", 1)[1])
                except Exception:
                    continue
    except Exception:
        pass
    return None


def check_vram_pressure(port: int = 8100) -> CheckResult:
    """Detect KV-cache spillover risk — the 2026-04-13 ~125s-latency class bug.

    Two signals:

    1. **Live ratio** from ``nvidia-smi``: ``vram_used / vram_total > 0.95``
       means the UMA driver is already spilling KV pages into shared system
       memory, and inference will run at PCIe speeds (5-30x slower than
       on-GPU). This is the primary signal — it catches the state the user
       actually hit on 2026-04-13.
    2. **Predictive KV-math**: given the currently loaded profile's ``arch``
       + the running server's ``n_ctx``, project weights + KV cache +
       headroom against physical VRAM. Catches misconfiguration BEFORE the
       operator notices a slowdown.

    Either signal alone produces a WARN. Both together → FAIL with a specific
    ``MAXIM_LLM_N_CTX=<recommended>`` fix string, interpolating real numbers
    (not ``<your-vram>`` placeholders).
    """
    # Lazy imports — keep the check fast on systems without a GPU.
    try:
        from maxim.runtime.leader_proxy import _query_nvidia_smi
    except Exception:
        return CheckResult(
            name="VRAM pressure",
            status="info",
            message="VRAM pressure check unavailable (leader_proxy import failed)",
        )

    gpu = _query_nvidia_smi()
    if gpu is None:
        return CheckResult(
            name="VRAM pressure",
            status="info",
            message="nvidia-smi unavailable — skipping VRAM pressure check",
        )

    vram_used_gb = float(gpu.get("vram_used_gb") or 0.0)
    vram_total_gb = float(gpu.get("vram_total_gb") or 0.0)
    if vram_total_gb <= 0:
        return CheckResult(
            name="VRAM pressure",
            status="info",
            message="nvidia-smi reported zero total VRAM — skipping",
        )

    ratio = vram_used_gb / vram_total_gb

    # Resolve active profile (if any) so we can do the predictive projection.
    # Must read through the llm_server module reference — `_active_model` is
    # a mutable global and `from lane_backends import _active_model` binds by
    # value (see CLAUDE.md "Mutable globals + module extraction" lesson).
    active_profile: str | None = None
    try:
        import maxim.runtime.llm_server as _server_mod

        active_profile = _server_mod._active_model
    except Exception:
        active_profile = None

    projection = None
    running_n_ctx: int | None = None
    if active_profile:
        running_n_ctx = _current_llama_server_n_ctx(port)
        if running_n_ctx and running_n_ctx > 0:
            try:
                from maxim.models.language.config import _BUILTIN_PROFILES
                from maxim.runtime.lane_models import project_vram_usage

                profile_meta = _BUILTIN_PROFILES.get(active_profile, {})
                projection = project_vram_usage(
                    active_profile,
                    profile_meta,
                    running_n_ctx,
                    vram_total_gb,
                )
            except Exception:
                projection = None

    live_spillover = ratio > 0.95
    live_warning = ratio > 0.85

    # Build a fix hint that names REAL values — profile, n_ctx, VRAM. The
    # operator should be able to copy-paste it.
    def _build_fix(rec_n_ctx: int | None) -> str:
        lines: list[str] = []
        if rec_n_ctx and rec_n_ctx > 0:
            lines.append(f"export MAXIM_LLM_N_CTX={rec_n_ctx}  # then: maxim peer restart")
        else:
            lines.append("export MAXIM_LLM_N_CTX=4096  # then: maxim peer restart")
        if active_profile and active_profile.startswith("qwen2.5-14b"):
            lines.append("# or switch to a smaller profile: maxim peer llm qwen2-7b-instruct")
        else:
            lines.append("# or switch to a smaller profile via `maxim peer llm <model>`")
        lines.append("# or close other GPU consumers (browser hw accel, extra models)")
        return "\n".join(lines)

    # Case 1: live spillover confirmed — worst case. Inference is already slow.
    if live_spillover:
        msg = (
            f"VRAM {vram_used_gb:.1f}/{vram_total_gb:.0f} GB "
            f"({ratio * 100:.0f}% used) — KV cache likely spilled to shared GPU "
            f"memory. Expect 5-30x slowdown (PCIe-bound inference)."
        )
        rec = projection.recommended_n_ctx if projection else None
        return CheckResult(
            name="VRAM pressure",
            status="fail",
            message=msg,
            fix=_build_fix(rec),
            retry_id="vram_pressure",
        )

    # Case 2: predictive projection says we'll spill even if we're currently fine.
    # Can happen right after spawn before KV cache fills.
    if projection is not None and projection.spillover_risk:
        msg = (
            f"Projected {projection.projected_total_gb:.1f} GB "
            f"(weights {projection.weights_gb:.1f} + KV {projection.kv_cache_gb:.1f} "
            f"+ {projection.headroom_gb:.1f} headroom) exceeds 95% of "
            f"{projection.physical_vram_gb:.0f} GB VRAM at n_ctx={projection.n_ctx}. "
            f"Will spill to shared memory once KV cache fills."
        )
        return CheckResult(
            name="VRAM pressure",
            status="fail",
            message=msg,
            fix=_build_fix(projection.recommended_n_ctx),
            retry_id="vram_pressure",
        )

    # Case 3: in the warning band — no headroom for KV growth.
    if live_warning:
        return CheckResult(
            name="VRAM pressure",
            status="warn",
            message=(
                f"VRAM {vram_used_gb:.1f}/{vram_total_gb:.0f} GB "
                f"({ratio * 100:.0f}% used) — no headroom for KV cache growth. "
                f"Long prompts may spill."
            ),
            fix=_build_fix(projection.recommended_n_ctx if projection else None),
            retry_id="vram_pressure",
        )

    # All clear.
    suffix = f", n_ctx={running_n_ctx}" if running_n_ctx else ""
    return CheckResult(
        name="VRAM pressure",
        status="ok",
        message=f"VRAM {vram_used_gb:.1f}/{vram_total_gb:.0f} GB ({ratio * 100:.0f}% used{suffix})",
    )


# ─── LAN access ────────────────────────────────────────────────────────────


def check_lan_access(info: PlatformInfo, port: int = 8100) -> CheckResult:
    """Platform-specific LAN-access guidance for exposing the local server.

    Doesn't test reachability from an external host (can't, from here). Just
    prints what the user needs to do to make the server LAN-reachable, with
    their actual IPs filled in.
    """
    from maxim.runtime.leader_mode import detect_role

    # Use resolved role (respects both MAXIM_ROLE env var and implicit
    # cloudflared-config detection), not just the raw env var.
    leader = detect_role().is_leader

    if info.runtime == "wsl2":
        wsl_ip = detect_wsl_ip() or "<wsl-ip>"
        host_ip = info.windows_host_ip or "<windows-host-ip>"
        fix = (
            f"WSL2 peers can't reach http://127.0.0.1:{port} directly — the\n"
            f"  Windows host needs to forward the port.\n\n"
            f"  1. In PowerShell on Windows (as admin):\n"
            f"       netsh interface portproxy add v4tov4 listenport={port} listenaddress=0.0.0.0 connectport={port} connectaddress={wsl_ip}\n"
            f'       netsh advfirewall firewall add rule name="Maxim LLM" dir=in action=allow protocol=TCP localport={port}\n\n'
            f"  2. Run Maxim as leader:\n"
            f"       MAXIM_ROLE=leader maxim\n\n"
            f"  3. Peers connect via your Windows LAN IP:\n"
            f"       http://{host_ip}:{port}/v1\n"
        )
        return CheckResult(
            name="LAN access (WSL2)",
            status="warn" if not leader else "ok",
            message=(
                "WSL2 port-forwarding required for LAN peers"
                if not leader
                else f"leader mode + port-forwarded at {host_ip}:{port}"
            ),
            fix=fix if not leader else None,
        )

    if info.runtime in ("wsl1", "docker"):
        return CheckResult(
            name=f"LAN access ({info.runtime})",
            status="warn",
            message=f"LAN access untested on {info.runtime}",
            fix=(
                f"{info.runtime} has unusual networking. Consider switching to WSL2\n"
                f"  (mirrored networking mode) or running Maxim in the native host."
            ),
        )

    if info.os == "linux":
        from maxim.doctor.platform_detect import detect_lan_ip

        lan_ip = detect_lan_ip() or "<your-lan-ip>"
        firewall_hint = {
            "ubuntu": f"sudo ufw allow {port}/tcp",
            "debian": f"sudo ufw allow {port}/tcp",
            "fedora": f"sudo firewall-cmd --add-port={port}/tcp --permanent && sudo firewall-cmd --reload",
            "rhel": f"sudo firewall-cmd --add-port={port}/tcp --permanent && sudo firewall-cmd --reload",
            "arch": f"sudo ufw allow {port}/tcp  # (or your preferred firewall)",
        }.get(info.distro, f"Open TCP port {port} in your firewall")
        fix = (
            f"Linux peers reach you directly via your LAN IP:\n"
            f"  1. Run Maxim as leader:\n"
            f"       MAXIM_ROLE=leader maxim\n\n"
            f"  2. Allow port {port} in your firewall (if active):\n"
            f"       {firewall_hint}\n\n"
            f"  3. Peers connect:\n"
            f"       http://{lan_ip}:{port}/v1\n"
        )
        return CheckResult(
            name="LAN access (Linux)",
            status="warn" if not leader else "ok",
            message="native Linux — peers reach LAN IP directly",
            fix=fix if not leader else None,
        )

    if info.os == "macos":
        from maxim.doctor.platform_detect import detect_lan_ip

        lan_ip = detect_lan_ip() or "<your-lan-ip>"
        fix = (
            f"macOS peers reach you directly via your LAN IP:\n"
            f"  1. Run Maxim as leader:\n"
            f"       MAXIM_ROLE=leader maxim\n\n"
            f"  2. If macOS firewall is enabled, allow Python in:\n"
            f"       System Settings → Network → Firewall → Options\n\n"
            f"  3. Peers connect:\n"
            f"       http://{lan_ip}:{port}/v1\n"
        )
        return CheckResult(
            name="LAN access (macOS)",
            status="warn" if not leader else "ok",
            message="native macOS — peers reach LAN IP directly",
            fix=fix if not leader else None,
        )

    if info.os == "windows":
        fix = (
            f"Windows native:\n"
            f"  1. In PowerShell as admin, allow inbound:\n"
            f'       New-NetFirewallRule -DisplayName "Maxim LLM" -Direction Inbound -Protocol TCP -LocalPort {port} -Action Allow\n\n'
            f"  2. Run Maxim as leader:\n"
            f"       $env:MAXIM_ROLE='leader'\n"
            f"       maxim\n"
        )
        return CheckResult(
            name="LAN access (Windows)",
            status="warn" if not leader else "ok",
            message="native Windows",
            fix=fix if not leader else None,
        )

    return CheckResult(
        name="LAN access",
        status="warn",
        message="unknown platform",
    )


# ─── cloudflared / tunnel ──────────────────────────────────────────────────


def check_cloudflared(info: PlatformInfo) -> CheckResult:
    from maxim.tunnel.cloudflared import cloudflared_version, find_cloudflared

    path = find_cloudflared()
    if path is None:
        install = {
            (
                "linux",
                "ubuntu",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb\n"
            "  sudo dpkg -i cloudflared.deb",
            (
                "linux",
                "debian",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb\n"
            "  sudo dpkg -i cloudflared.deb",
            (
                "linux",
                "fedora",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm -o cloudflared.rpm\n"
            "  sudo rpm -i cloudflared.rpm",
            (
                "linux",
                "rhel",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm -o cloudflared.rpm\n"
            "  sudo rpm -i cloudflared.rpm",
            ("linux", "arch"): "yay -S cloudflared  # or your preferred AUR helper",
            ("macos", "unknown"): "brew install cloudflare/cloudflare/cloudflared",
            (
                "windows",
                "unknown",
            ): "Download https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe\n"
            "  (add to PATH)",
        }
        key = (info.os, info.distro if info.os == "linux" else "unknown")
        install_cmd = install.get(
            key,
            "See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/",
        )
        return CheckResult(
            name="cloudflared",
            status="warn",
            message="not installed — required for external tunnel access",
            fix=f"Install:\n  {install_cmd}\n  Then run: maxim tunnel setup",
            retry_id="cloudflared",
        )
    version = cloudflared_version() or "(unknown version)"
    return CheckResult(
        name="cloudflared",
        status="ok",
        message=f"{path} — {version}",
    )


def check_tunnel_config() -> CheckResult:
    from maxim.tunnel.config import CONFIG_PATH, read_config_summary

    summary = read_config_summary()
    if summary is None:
        return CheckResult(
            name="Tunnel config",
            status="warn",
            message=f"no config at {CONFIG_PATH}",
            fix="Run: maxim tunnel setup",
            retry_id="tunnel-config",
        )
    hostname = summary.get("hostname", "?")
    service = summary.get("service", "")

    # Warn if tunnel points directly at llama-cpp-server (8100) instead of
    # LeaderProxy (8099). Without the proxy, peers bypass auth enforcement,
    # logging, GPU metrics, admission control, and admin endpoints.
    if ":8100" in service:
        return CheckResult(
            name="Tunnel config",
            status="warn",
            message=f"hostname={hostname} — tunnel points at port 8100 (llama-cpp-server directly)",
            fix=(
                "Update your tunnel config to route through the LeaderProxy (port 8099):\n"
                f"  Edit {CONFIG_PATH}\n"
                "  Change: service: http://localhost:8100\n"
                "  To:     service: http://localhost:8099\n"
                "  Then restart cloudflared:\n"
                "    sudo systemctl restart cloudflared   # Linux\n"
                "    cloudflared --config ~/.cloudflared/config.yml tunnel run   # manual"
            ),
        )

    return CheckResult(
        name="Tunnel config",
        status="ok",
        message=f"hostname={hostname} service={service}",
    )


def check_tunnel_config_sync() -> CheckResult:
    """Warn if the systemd cloudflared config differs from the user config.

    When cloudflared is installed as a systemd service, it reads from
    /etc/cloudflared/config.yml — NOT ~/.cloudflared/config.yml. After
    editing or regenerating the user config, the systemd copy must be
    updated too, or the tunnel will use stale settings.
    """
    import platform

    if platform.system() != "Linux":
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="skipped (non-Linux)",
        )

    from pathlib import Path

    user_cfg = Path.home() / ".cloudflared" / "config.yml"
    system_cfg = Path("/etc/cloudflared/config.yml")

    if not user_cfg.is_file():
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="no user config — nothing to compare",
        )
    if not system_cfg.is_file():
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="no systemd config — cloudflared may not be a service",
        )

    try:
        user_text = user_cfg.read_text()
        system_text = system_cfg.read_text()
    except PermissionError:
        return CheckResult(
            name="Tunnel config sync",
            status="warn",
            message="cannot read /etc/cloudflared/config.yml (permission denied)",
            fix="Run: sudo cat /etc/cloudflared/config.yml",
        )

    if user_text.strip() == system_text.strip():
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="~/.cloudflared/config.yml and /etc/cloudflared/config.yml are in sync",
        )

    return CheckResult(
        name="Tunnel config sync",
        status="warn",
        message=(
            "~/.cloudflared/config.yml and /etc/cloudflared/config.yml DIFFER — "
            "systemd service may be using stale settings"
        ),
        fix=(
            "The systemd cloudflared service reads /etc/cloudflared/config.yml,\n"
            "not ~/.cloudflared/config.yml. After editing the user config, sync it:\n"
            "  sudo cp ~/.cloudflared/config.yml /etc/cloudflared/config.yml\n"
            "  sudo systemctl restart cloudflared"
        ),
        retry_id="tunnel-config-sync",
    )


# ─── API key ───────────────────────────────────────────────────────────────


def check_api_key() -> CheckResult:
    from maxim.tunnel.keys import key_exists, key_file_path, truncate_for_display, read_key

    if not key_exists():
        return CheckResult(
            name="API key",
            status="warn",
            message="no key set — server accepts all requests (localhost is fine)",
            fix=(
                "Generate one before exposing to LAN/tunnel:\n"
                "  maxim tunnel key rotate\n"
                "  maxim tunnel key export   # shell snippets for peers"
            ),
        )
    key = read_key() or ""
    return CheckResult(
        name="API key",
        status="ok",
        message=f"{truncate_for_display(key)} at {key_file_path()}",
    )


def check_key_age() -> CheckResult:
    """Warn when the API key file is older than 90 days."""
    import time

    from maxim.tunnel.keys import key_exists, key_file_path

    if not key_exists():
        return CheckResult(
            name="Key age",
            status="info",
            message="no key file — skipped",
        )
    path = key_file_path()
    try:
        age_days = (time.time() - path.stat().st_mtime) / 86400
    except OSError:
        return CheckResult(
            name="Key age",
            status="warn",
            message=f"cannot stat {path}",
        )
    if age_days > 90:
        return CheckResult(
            name="Key age",
            status="warn",
            message=f"key is {age_days:.0f} days old — consider rotating",
            fix="maxim tunnel key rotate && maxim tunnel key export",
            retry_id="key-age",
        )
    return CheckResult(
        name="Key age",
        status="ok",
        message=f"key is {age_days:.0f} days old",
    )


def check_key_permissions() -> CheckResult:
    """Fail if the key file is world-readable (POSIX only)."""
    import platform
    import stat

    if platform.system() == "Windows":
        return CheckResult(
            name="Key permissions",
            status="ok",
            message="skipped (Windows uses ACLs)",
        )

    from maxim.tunnel.keys import key_exists, key_file_path

    if not key_exists():
        return CheckResult(
            name="Key permissions",
            status="info",
            message="no key file — skipped",
        )
    path = key_file_path()
    try:
        mode = path.stat().st_mode
    except OSError:
        return CheckResult(
            name="Key permissions",
            status="warn",
            message=f"cannot stat {path}",
        )
    # Check group/other read bits
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        return CheckResult(
            name="Key permissions",
            status="fail",
            message=f"key file is world-readable (mode {oct(mode & 0o777)})",
            fix=f"chmod 600 {path}",
            retry_id="key-permissions",
        )
    return CheckResult(
        name="Key permissions",
        status="ok",
        message=f"mode {oct(mode & 0o777)}",
    )


def check_key_auth_smoke(port: int = 8100) -> CheckResult:
    """Verify the local server enforces the configured API key."""
    from maxim.tunnel.keys import key_exists, read_key
    from maxim.utils import http as _http

    if not key_exists():
        return CheckResult(
            name="Key auth smoke",
            status="info",
            message="no key configured — skipped",
        )
    key = read_key() or ""
    url = f"http://127.0.0.1:{port}/v1/models"
    fast = _http.TimeoutPolicy(connect_s=1.0, read_s=3.0, total_s=4.0)
    # Test with correct key
    try:
        _http.fetch_url(
            url,
            method="GET",
            headers={"Authorization": f"Bearer {key}"},
            timeout=fast,
        )
    except _http.HTTPAuthError:
        return CheckResult(
            name="Key auth smoke",
            status="fail",
            message="server rejected our key (401)",
            fix="Key mismatch — regenerate: maxim tunnel key rotate",
            retry_id="key-auth",
        )
    except _http.HTTPError as e:
        return CheckResult(
            name="Key auth smoke",
            status="warn",
            message=f"server returned HTTP {e.status}",
        )
    except Exception:
        return CheckResult(
            name="Key auth smoke",
            status="info",
            message="server not reachable — skipped",
        )
    # Test with bogus key — should get 401 if auth is enforced
    try:
        _http.fetch_url(
            url,
            method="GET",
            headers={"Authorization": "Bearer BOGUS"},
            timeout=fast,
        )
        return CheckResult(
            name="Key auth smoke",
            status="warn",
            message="server accepts ANY key — auth not enforced",
            fix=(
                "The server accepts requests without verifying the API key.\n"
                "  Fine for localhost, but insecure if exposed via tunnel.\n"
                "  Route through LeaderProxy (port 8099) to enforce auth."
            ),
        )
    except _http.HTTPAuthError:
        return CheckResult(
            name="Key auth smoke",
            status="ok",
            message="auth enforced — valid key accepted, bogus key rejected",
        )
    except _http.HTTPError as e:
        return CheckResult(
            name="Key auth smoke",
            status="warn",
            message=f"bogus-key probe returned HTTP {e.status} (expected 401)",
        )
    except Exception:
        return CheckResult(
            name="Key auth smoke",
            status="ok",
            message="valid key accepted (bogus-key probe inconclusive)",
        )


# ─── disk + memory ────────────────────────────────────────────────────────


def check_disk_space() -> CheckResult:
    """Warn when free disk space is low."""
    import shutil
    from pathlib import Path

    from maxim.utils.paths import data_home

    try:
        check_path = data_home()
    except Exception:
        check_path = Path.home()
    try:
        usage = shutil.disk_usage(check_path)
    except OSError as e:
        return CheckResult(
            name="Disk space",
            status="warn",
            message=f"cannot check: {e}",
        )
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    if free_gb < 2:
        return CheckResult(
            name="Disk space",
            status="fail",
            message=f"{free_gb:.1f} GB free of {total_gb:.0f} GB — critically low",
            fix="Free disk space. Sim reports, model files, and logs consume storage over time.",
        )
    if free_gb < 10:
        return CheckResult(
            name="Disk space",
            status="warn",
            message=f"{free_gb:.1f} GB free of {total_gb:.0f} GB — model downloads may fail",
        )
    return CheckResult(
        name="Disk space",
        status="ok",
        message=f"{free_gb:.1f} GB free of {total_gb:.0f} GB",
    )


def check_ram_headroom() -> CheckResult:
    """Report available system RAM."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
    except ImportError:
        # Fallback: read /proc/meminfo on Linux, sysctl on macOS
        import platform

        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    info = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            info[parts[0].rstrip(":")] = int(parts[1])
                avail_gb = info.get("MemAvailable", 0) / (1024**2)
                total_gb = info.get("MemTotal", 0) / (1024**2)
            except Exception:
                return CheckResult(name="RAM", status="info", message="cannot read /proc/meminfo")
        elif platform.system() == "Darwin":
            import subprocess

            try:
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5, text=True)
                total_gb = int(out.strip()) / (1024**3)
                # macOS doesn't expose "available" easily without psutil
                avail_gb = -1
            except Exception:
                return CheckResult(name="RAM", status="info", message="cannot read sysctl")
        else:
            return CheckResult(
                name="RAM",
                status="info",
                message="install psutil for RAM check: pip install psutil",
            )

    if avail_gb < 0:
        return CheckResult(
            name="RAM",
            status="ok",
            message=f"{total_gb:.1f} GB total (install psutil for available RAM)",
        )
    if avail_gb < 2:
        return CheckResult(
            name="RAM",
            status="warn",
            message=f"{avail_gb:.1f} GB available of {total_gb:.1f} GB — tight for LLM inference",
            fix="Close other applications or choose a smaller model (e.g., smollm-1.7b)",
        )
    return CheckResult(
        name="RAM",
        status="ok",
        message=f"{avail_gb:.1f} GB available of {total_gb:.1f} GB",
    )


# ─── inference coherence ──────────────────────────────────────────────────


def check_inference_coherence(port: int = 8100) -> CheckResult:
    """Send a fixed prompt and verify the LLM returns a sensible answer."""
    import time

    from maxim.utils import http as _http

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    try:
        t0 = time.time()
        resp = _http.fetch_url(
            url,
            method="POST",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=30.0, total_s=32.0),
        )
        data = resp.json()
        dt_ms = (time.time() - t0) * 1000
    except Exception:
        return CheckResult(
            name="Inference coherence",
            status="info",
            message="server not reachable — skipped",
        )

    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return CheckResult(
            name="Inference coherence",
            status="fail",
            message="response has unexpected structure",
        )

    # Check if response contains "4" somewhere
    if "4" in text:
        return CheckResult(
            name="Inference coherence",
            status="ok",
            message=f"correct ({dt_ms:.0f} ms): {text!r}",
        )
    return CheckResult(
        name="Inference coherence",
        status="warn",
        message=f"unexpected answer ({dt_ms:.0f} ms): {text!r} — model may be misconfigured",
    )


# ─── peer-mode checks ────────────────────────────────────────────────────


def check_peer_url_reachable(url: str) -> CheckResult:
    """Resolve DNS and probe /v1/models on a remote leader."""
    import socket
    import time
    import urllib.parse

    from maxim.utils import http as _http

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if not host:
        return CheckResult(
            name="Remote URL",
            status="fail",
            message=f"no hostname in URL: {url}",
        )
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        return CheckResult(
            name="Remote URL",
            status="fail",
            message=f"DNS resolution failed for {host}",
            fix=f"Check the hostname. Is the leader URL correct?\n  URL: {url}",
            retry_id="peer-url",
        )
    # HTTP probe
    models_url = url.rstrip("/")
    if not models_url.endswith("/v1"):
        models_url += "/v1"
    models_url += "/models"
    try:
        t0 = time.time()
        _http.fetch_url(
            models_url,
            method="GET",
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0),
        )
        dt_ms = (time.time() - t0) * 1000
    except _http.HTTPAuthError:
        return CheckResult(
            name="Remote URL",
            status="ok",
            message=f"{host} reachable (auth required — checked separately)",
        )
    except _http.HTTPError as e:
        return CheckResult(
            name="Remote URL",
            status="warn",
            message=f"{host} returned HTTP {e.status}",
            retry_id="peer-url",
        )
    except Exception as e:
        return CheckResult(
            name="Remote URL",
            status="fail",
            message=f"cannot reach {host}: {e}",
            fix=f"Is the leader running? Try: curl -v {models_url}",
            retry_id="peer-url",
        )
    return CheckResult(
        name="Remote URL",
        status="ok",
        message=f"{host} reachable ({dt_ms:.0f} ms)",
    )


def check_peer_key_set() -> CheckResult:
    """Check that the peer has a remote API key configured."""
    import os

    key = os.environ.get("MAXIM_LANE_LARGE_REMOTE_API_KEY")
    if not key:
        try:
            from maxim.peer.config import read_peer_config

            cfg = read_peer_config()
            if cfg is not None:
                key = cfg.api_key
        except Exception as e:
            logger.debug("Could not read peer config for API key check: %s", e)
    if not key:
        return CheckResult(
            name="Peer API key",
            status="warn",
            message="no API key configured for remote access",
            fix=(
                "Get the key from the leader:\n"
                "  On leader: maxim tunnel key export\n"
                "  Then paste the export snippet here."
            ),
            retry_id="peer-key",
        )
    from maxim.tunnel.keys import truncate_for_display

    return CheckResult(
        name="Peer API key",
        status="ok",
        message=f"key set: {truncate_for_display(key)}",
    )


def check_peer_auth(url: str, key: str | None) -> CheckResult:
    """Send an authenticated request to the leader and verify 200."""
    from maxim.utils import http as _http

    if not key:
        return CheckResult(
            name="Peer auth",
            status="info",
            message="no key — skipped",
        )
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    models_url = f"{base}/models"
    try:
        _http.fetch_url(
            models_url,
            method="GET",
            headers={"Authorization": f"Bearer {key}"},
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0),
        )
    except _http.HTTPAuthError:
        return CheckResult(
            name="Peer auth",
            status="fail",
            message="key rejected by leader (401)",
            fix=(
                "Key mismatch. Ask the leader to regenerate:\n"
                "  On leader: maxim tunnel key rotate\n"
                "  Then: maxim tunnel key export"
            ),
            retry_id="peer-auth",
        )
    except _http.HTTPError as e:
        return CheckResult(
            name="Peer auth",
            status="warn",
            message=f"leader returned HTTP {e.status}",
        )
    except Exception as e:
        return CheckResult(
            name="Peer auth",
            status="warn",
            message=f"connection failed: {e}",
        )
    return CheckResult(
        name="Peer auth",
        status="ok",
        message="authenticated successfully",
    )


def check_peer_model(url: str, key: str | None, expected_model: str | None = None) -> CheckResult:
    """Check if the leader advertises the expected model."""
    from maxim.utils import http as _http

    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    models_url = f"{base}/models"
    headers: dict[str, str] = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        resp = _http.fetch_url(
            models_url,
            method="GET",
            headers=headers,
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0),
        )
        data = resp.json()
        model_ids = [m.get("id", "") for m in data.get("data", [])]
    except Exception:
        return CheckResult(
            name="Remote model",
            status="info",
            message="cannot query /v1/models — skipped",
        )
    if not model_ids:
        return CheckResult(
            name="Remote model",
            status="warn",
            message="leader reports no models loaded",
            fix="On the leader, load a model: maxim peer llm mistral-7b",
        )
    if expected_model and expected_model not in model_ids:
        return CheckResult(
            name="Remote model",
            status="warn",
            message=f"expected {expected_model!r}, leader has: {', '.join(model_ids)}",
            fix=f"On leader: maxim peer llm {expected_model}",
        )
    return CheckResult(
        name="Remote model",
        status="ok",
        message=f"leader model: {', '.join(model_ids)}",
    )


def check_peer_latency(url: str, key: str | None) -> CheckResult:
    """Measure round-trip latency to the leader (5 pings, report p50)."""
    import time

    from maxim.utils import http as _http

    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    ping_url = f"{base}/models"
    headers: dict[str, str] = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    timings: list[float] = []
    fast = _http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0)
    for probe_idx in range(5):
        try:
            t0 = time.time()
            _http.fetch_url(ping_url, method="GET", headers=headers, timeout=fast)
            timings.append((time.time() - t0) * 1000)
        except Exception as e:
            logger.debug("Latency probe %d failed: %s", probe_idx, e)
    if not timings:
        return CheckResult(
            name="Peer latency",
            status="warn",
            message="all latency probes failed",
        )
    timings.sort()
    p50 = timings[len(timings) // 2]
    p95 = timings[int(len(timings) * 0.95)] if len(timings) >= 2 else timings[-1]
    if p50 > 200:
        return CheckResult(
            name="Peer latency",
            status="warn",
            message=f"p50={p50:.0f} ms, p95={p95:.0f} ms — high latency may slow real-time inference",
        )
    return CheckResult(
        name="Peer latency",
        status="ok",
        message=f"p50={p50:.0f} ms, p95={p95:.0f} ms ({len(timings)}/5 probes)",
    )


# ─── leader role ───────────────────────────────────────────────────────────


def check_role() -> CheckResult:
    from maxim.runtime.leader_mode import detect_role as _legacy_detect
    from maxim.runtime.role import detect_role as _new_detect

    decision = _legacy_detect()
    new_role, new_source = _new_detect()

    # Normalise legacy "client" → "peer" for comparison
    legacy_role = "peer" if decision.role == "client" else decision.role

    if legacy_role != new_role:
        # Two role-detection systems disagree — surface this in doctor output
        # rather than burying it as a WARNING-only log event.  Operators need
        # to set MAXIM_ROLE explicitly to resolve it.
        return CheckResult(
            name="Role",
            status="warn",
            message=(
                f"role_divergence: leader_mode says '{legacy_role}' "
                f"({decision.reason}), role.py says '{new_role}' ({new_source}). "
                f"Set MAXIM_ROLE explicitly to resolve."
            ),
            fix=f"export MAXIM_ROLE={new_role}  # or 'leader' / 'solo' as appropriate",
        )

    if decision.is_leader:
        return CheckResult(
            name="Role",
            status="ok",
            message=f"leader (bind={decision.bind_host}) — {decision.reason}",
        )
    return CheckResult(
        name="Role",
        status="ok",
        message=f"{decision.role} (bind={decision.bind_host}) — {decision.reason}",
    )


def _check_lane_metrics() -> list["CheckResult"]:
    """Report per-lane performance metrics if available (Phase 8)."""
    try:
        from maxim.models.language.lane_metrics import get_metrics_registry
    except Exception:
        return []
    registry = get_metrics_registry()
    results: list[CheckResult] = []
    for name, metrics in registry.all_metrics().items():
        snap = metrics.snapshot()
        total = snap["jobs_completed"] + snap["jobs_failed"]
        if total == 0:
            continue
        msg = metrics.format_compact()
        status: Status = "ok"
        if snap["failure_rate"] > 0.2:
            status = "warn"
        if snap["failure_rate"] > 0.5:
            status = "fail"
        results.append(CheckResult(name=f"Lane: {name}", status=status, message=msg))
    return results


def _detect_doctor_role(explicit: str | None = None, peer_url: str | None = None) -> tuple[str, str | None]:
    """Detect whether this machine is peer/leader/solo for doctor purposes.

    Returns ``(role, peer_url)`` where role is ``"peer"``, ``"leader"``,
    ``"solo"``, or ``"auto"`` (fall through to existing behaviour).
    """
    import os
    from urllib.parse import urlparse

    if explicit == "peer":
        url = peer_url or os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL")
        return "peer", url
    if explicit in ("leader", "solo"):
        return explicit, None
    # Auto-detect from env: URL wins (most specific signal)
    url = os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL")
    if url:
        host = urlparse(url).hostname or ""
        if host not in ("127.0.0.1", "localhost", "::1"):
            return "peer", url
    # Fallback: MAXIM_ROLE is set by detect_and_apply_role() before any subcommand
    # runs. If it says "peer", trust it and pull the URL from peer.yml.
    maxim_role = os.environ.get("MAXIM_ROLE", "").strip().lower()
    if maxim_role == "peer":
        try:
            from maxim.peer.config import read_peer_config

            cfg = read_peer_config()
            if cfg is not None:
                return "peer", cfg.url
        except Exception:
            pass
        return "peer", None
    if maxim_role in ("leader", "solo"):
        return maxim_role, None
    return "auto", None


# ─── P8: Routing-decision visibility checks ─────────────────────────────────


def check_tier_effectiveness(caps=None) -> CheckResult:
    """Compare what tier detection actually picked vs. what the hardware allows.

    On a 24 GB Mac that hasn't downloaded qwen2.5-14b yet, ``detect_tiers``
    walks past qwen and lands on the next-best profile that IS on disk
    (e.g. mistral-7b). The tier check still says "ok", which is correct
    but hides the gap. This check runs detect_tiers TWICE — once
    respecting availability, once ignoring it — and reports the gap with
    the exact download command.
    """
    try:
        from maxim.runtime.capabilities import RuntimeCapabilities, detect_compute_resources
        from maxim.runtime.lane_models import detect_tiers
        from maxim.runtime.llm_server import profile_has_local_file
    except ImportError as e:
        return CheckResult(
            name="LLM Tier headroom",
            status="warn",
            message=f"Tier-effectiveness check unavailable: {e}",
        )
    if caps is None:
        try:
            has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
            caps = RuntimeCapabilities(has_gpu=has_gpu, gpu_type=gpu_type, vram_gb=vram_gb, ram_gb=ram_gb)
        except Exception as e:
            return CheckResult(
                name="LLM Tier headroom",
                status="warn",
                message=f"Capability detection failed: {e}",
            )

    try:
        actual_tiers = detect_tiers(caps, profile_available=profile_has_local_file)
        ideal_tiers = detect_tiers(caps, profile_available=lambda _name: True)
    except Exception as e:
        return CheckResult(
            name="LLM Tier headroom",
            status="warn",
            message=f"Tier walk failed: {e}",
        )

    actual_large = actual_tiers.get("large") or actual_tiers.get("medium")
    ideal_large = ideal_tiers.get("large") or ideal_tiers.get("medium")
    if actual_large is None or ideal_large is None:
        return CheckResult(
            name="LLM Tier headroom",
            status="info",
            message="No large/medium tier detected",
        )
    actual_profile = actual_large.model_profile or "(none)"
    ideal_profile = ideal_large.model_profile or "(none)"
    if actual_profile == ideal_profile:
        return CheckResult(
            name="LLM Tier headroom",
            status="ok",
            message=f"Hardware-best profile selected: {actual_profile}",
        )
    return CheckResult(
        name="LLM Tier headroom",
        status="warn",
        message=(f"Running '{actual_profile}' but hardware ({caps.vram_gb:.0f} GB VRAM) could run '{ideal_profile}'."),
        fix=f"python -m maxim.models.download --llm {ideal_profile}",
        retry_id="tier_effectiveness",
    )


def check_peer_vs_local_conflict() -> CheckResult:
    """Inform the user when --llm + peer config will run locally.

    P1's ``_apply_local_llm_override`` clears the large lane's
    ``remote_url`` whenever ``MAXIM_LLM_PROFILE`` names a local
    (non-cloud) profile. This is the right behavior, but invisible — a
    user with a peer config and ``--llm mistral-7b`` may be expecting
    the leader to serve mistral. This check surfaces the override as an
    informational message so they're not surprised.
    """
    import os

    profile = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
    if not profile:
        return CheckResult(
            name="--llm vs peer config",
            status="ok",
            message="no local --llm override active",
        )
    try:
        from maxim.models.language.config import _BUILTIN_PROFILES
        from maxim.peer.config import read_peer_config

        cfg = read_peer_config()
    except Exception:
        return CheckResult(
            name="--llm vs peer config",
            status="info",
            message=f"--llm={profile} active (peer config check unavailable)",
        )
    if cfg is None:
        return CheckResult(
            name="--llm vs peer config",
            status="ok",
            message=f"--llm={profile} (no peer config)",
        )
    profile_data = _BUILTIN_PROFILES.get(profile, {})
    if profile_data.get("cloud"):
        return CheckResult(
            name="--llm vs peer config",
            status="ok",
            message=f"--llm={profile} is cloud (handled by cloud overrides)",
        )
    return CheckResult(
        name="--llm vs peer config",
        status="info",
        message=(
            f"--llm={profile} will run LOCALLY despite peer config at {cfg.url}. "
            f"P1 clears remote_url for local profiles. To use the leader "
            f"instead, run: maxim peer llm {profile}"
        ),
    )


def check_remote_reachability(url: str | None = None, api_key: str | None = None) -> CheckResult:
    """Probe a remote leader URL with the structured probe (P6).

    Reports outcome-specific fix hints. Distinct from
    ``check_peer_url_reachable`` (the existing peer-mode check) which
    just returns ok/fail; this one surfaces the auth_rejected vs
    dns_fail vs timeout distinction so the user gets a directly
    actionable hint.
    """
    if url is None:
        try:
            from maxim.peer.config import read_peer_config

            cfg = read_peer_config()
            if cfg is None:
                return CheckResult(
                    name="Remote leader probe",
                    status="info",
                    message="No peer config — skipping remote probe",
                )
            url = cfg.url
            api_key = api_key or cfg.api_key
        except Exception as e:
            return CheckResult(
                name="Remote leader probe",
                status="warn",
                message=f"Could not load peer config: {e}",
            )
    try:
        from maxim.runtime.llm_server import probe_llm_server
    except ImportError as e:
        return CheckResult(
            name="Remote leader probe",
            status="warn",
            message=f"Probe unavailable: {e}",
        )
    # Plan 3 R2.6: ``probe_llm_server`` is a thin compat shim that
    # delegates to ``_MaximPeerBackend.for_url(url).health_check``.
    # Routing through the shim preserves the existing
    # test_doctor_p8_checks mocking pattern while still funnelling
    # production traffic into the backend's canonical implementation.
    result = probe_llm_server(url, api_key=api_key)
    if result.outcome == "ok":
        return CheckResult(
            name="Remote leader probe",
            status="ok",
            message=f"{url} responding ({result.latency_ms:.0f} ms)",
        )
    if result.outcome == "auth_rejected":
        return CheckResult(
            name="Remote leader probe",
            status="warn",
            message=f"{url} alive but rejected the API key ({result.detail})",
            fix="maxim peer key  # rotate / re-paste from leader",
            retry_id="remote_probe",
        )
    fix_hints = {
        "dns_fail": f"Check the hostname in peer.yml or $MAXIM_LANE_LARGE_REMOTE_URL ({url})",
        "tls_error": "Check the leader's TLS certificate validity",
        "connection_refused": "Leader is not accepting connections — start `maxim` on the leader",
        "timeout": "Leader did not respond in time — is it cold-loading a model?",
        "http_5xx": "Leader returned a server error — check `maxim peer logs`",
        "other": "Unexpected error — check `maxim peer logs`",
    }
    return CheckResult(
        name="Remote leader probe",
        status="fail",
        message=f"{url} unreachable: {result.outcome} ({result.detail})",
        fix=fix_hints.get(result.outcome, "check `maxim peer logs`"),
        retry_id="remote_probe",
    )


def check_storage_footprint() -> CheckResult:
    """Surface ~/.maxim disk usage with a top-N subdir breakdown.

    Distinct from check_disk_space (which is a fs-level free check):
    this one tells the user where their Maxim footprint is going so
    they can decide what to delete. Fails when free space drops below
    10 GB, warns below 20 GB.
    """
    try:
        from maxim.utils.storage import format_report, report_storage
    except ImportError as e:
        return CheckResult(
            name="Storage footprint",
            status="warn",
            message=f"Storage report unavailable: {e}",
        )
    try:
        report = report_storage(force=True)
    except Exception as e:
        return CheckResult(
            name="Storage footprint",
            status="warn",
            message=f"Storage walk failed: {e}",
        )
    summary = format_report(report)
    if report.fs_free_gb == float("inf"):
        return CheckResult(
            name="Storage footprint",
            status="info",
            message=summary,
        )
    if report.fs_free_gb < 10.0:
        return CheckResult(
            name="Storage footprint",
            status="fail",
            message=f"Only {report.fs_free_gb:.1f} GB free on {report.data_home}",
            fix=summary + "\n  Delete unused models: maxim --delete-model NAME",
        )
    if report.fs_free_gb < 20.0:
        return CheckResult(
            name="Storage footprint",
            status="warn",
            message=f"{report.fs_free_gb:.1f} GB free on {report.data_home} (Maxim using {report.total_maxim_gb:.1f} GB)",
            fix=summary,
        )
    return CheckResult(
        name="Storage footprint",
        status="ok",
        message=f"{report.fs_free_gb:.1f} GB free, Maxim using {report.total_maxim_gb:.1f} GB",
    )


def run_all_checks(
    info: PlatformInfo,
    *,
    role: str | None = None,
    peer_url: str | None = None,
    peer_key: str | None = None,
) -> list[tuple[str, list[CheckResult]]]:
    """Return ordered ``[(section_name, results)]`` for ``maxim doctor``.

    Parameters
    ----------
    role
        Explicit role override: ``"peer"``, ``"leader"``, ``"solo"``.
        ``None`` triggers auto-detection.
    peer_url
        Remote leader URL (used when role is ``"peer"``).
    peer_key
        API key for the remote leader.
    """
    detected_role, detected_url = _detect_doctor_role(role, peer_url)
    peer_url = peer_url or detected_url

    # ── environment (always) ──────────────────────────────────────────────
    env_checks: list[CheckResult] = [
        CheckResult(name="Platform", status="ok", message=info.display_name),
        CheckResult(name="Architecture", status="ok", message=info.arch),
        check_gpu(),
        check_tier_detection(),
        check_tier_effectiveness(),
        check_disk_space(),
        check_ram_headroom(),
        check_storage_footprint(),
    ]
    # Splice env-var checks in after the hardware checks so operators see
    # misconfigurations right alongside the hardware context.
    env_checks.extend(check_env_config(info, role=detected_role))
    sections: list[tuple[str, list[CheckResult]]] = [
        ("Environment", env_checks),
    ]

    if detected_role == "peer" and peer_url:
        # ── peer-mode sections ────────────────────────────────────────────
        # Resolve key from arg → env → peer config
        if not peer_key:
            import os

            peer_key = os.environ.get("MAXIM_LANE_LARGE_REMOTE_API_KEY")
        if not peer_key:
            try:
                from maxim.peer.config import read_peer_config

                cfg = read_peer_config()
                if cfg is not None:
                    peer_key = cfg.api_key
            except Exception as e:
                logger.debug("Could not read peer config in run_all_checks: %s", e)
        sections.append(
            (
                "Peer Connectivity",
                [
                    check_peer_url_reachable(peer_url),
                    check_peer_key_set(),
                    check_peer_auth(peer_url, peer_key),
                    check_remote_reachability(peer_url, peer_key),
                    check_peer_vs_local_conflict(),
                    check_peer_model(peer_url, peer_key),
                    check_peer_latency(peer_url, peer_key),
                ],
            )
        )
    else:
        # ── leader / solo sections ────────────────────────────────────────
        sections += [
            (
                "Local LLM",
                [
                    check_llama_cpp_server_installed(),
                    check_server_reachable(),
                    check_llm_model_active(),
                    check_context_window(),
                    check_vram_pressure(),
                    check_inference_coherence(),
                ],
            ),
            (
                "Role & Access",
                [
                    check_role(),
                    check_lan_access(info),
                ],
            ),
            (
                "Tunnel (Cloudflare)",
                [
                    check_cloudflared(info),
                    check_tunnel_config(),
                    check_tunnel_config_sync(),
                ],
            ),
            (
                "API key",
                [
                    check_api_key(),
                    check_key_age(),
                    check_key_permissions(),
                    check_key_auth_smoke(),
                ],
            ),
        ]

    # Lane metrics (only show if any calls have been recorded)
    lane_results = _check_lane_metrics()
    if lane_results:
        sections.append(("Lane Metrics", lane_results))
    return sections


__all__ = [
    "CheckResult",
    "Status",
    "check_gpu",
    "check_tier_detection",
    "check_llama_cpp_server_installed",
    "check_server_reachable",
    "check_llm_model_active",
    "check_env_config",
    "check_context_window",
    "check_vram_pressure",
    "check_lan_access",
    "check_cloudflared",
    "check_tunnel_config",
    "check_tunnel_config_sync",
    "check_api_key",
    "check_key_age",
    "check_key_permissions",
    "check_key_auth_smoke",
    "check_disk_space",
    "check_ram_headroom",
    "check_inference_coherence",
    "check_peer_url_reachable",
    "check_peer_key_set",
    "check_peer_auth",
    "check_peer_model",
    "check_peer_latency",
    "check_role",
    "check_tier_effectiveness",
    "check_peer_vs_local_conflict",
    "check_remote_reachability",
    "check_storage_footprint",
    "run_all_checks",
]
