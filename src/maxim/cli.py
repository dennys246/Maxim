from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Sequence

from maxim.cli_parser import _build_parser
from maxim.utils.data_management import build_home
from maxim.utils.logging import configure_logging, log_exception

# Extracted to cli_utils.py — re-export for backward compatibility
from maxim.cli_utils import (
    normalize_epoch_value as _normalize_epoch_value,
    normalize_args as _normalize_args,
    gpu_available as _gpu_available,
    check_gpu_status as _check_gpu_status,
    configure_cpu_fallback_model as _configure_cpu_fallback_model,
    reexec_with_mode as _reexec_with_mode,
    clear_python_cache as _clear_python_cache,
    clear_memory as _clear_memory,
)

# Re-export MEMORY_PATHS for any external consumers
from maxim.cli_utils import MEMORY_PATHS  # noqa: F401


# ── Discrete subcommand handlers (extracted from main() for clarity) ────────


def _resolve_persona(args, default: str = "adversarial") -> str:
    """Resolve the persona arg, falling back to the supplied default."""
    return getattr(args, "sim_persona", default) or default


def _handle_list_models() -> int:
    """Print all known LLM profiles grouped by backend, then return 0.

    Used by ``maxim --list-models``. The grouping (local llama / local torch /
    cloud anthropic / cloud openai / cloud other) mirrors what users actually
    care about when picking a model.
    """
    from maxim.models.language.config import _BUILTIN_PROFILES, _PROFILE_ALIASES
    from maxim.runtime.lane_backends import _profile_has_local_file, _read_persisted_model

    local_llama: list[str] = []
    local_torch: list[str] = []
    cloud_anthropic: list[str] = []
    cloud_openai: list[str] = []
    cloud_other: list[str] = []

    for name, profile in sorted(_BUILTIN_PROFILES.items()):
        aliases = [a for a, v in _PROFILE_ALIASES.items() if v == name and a != name]
        alias_str = f"  (also: {', '.join(aliases[:2])})" if aliases else ""
        backend = profile.get("backend", "")
        n_ctx = profile.get("n_ctx", 0)
        ctx_str = f"{n_ctx // 1000}K" if n_ctx >= 1000 else str(n_ctx)

        if profile.get("cloud"):
            env = profile.get("api_key_env", "")
            key_set = bool(os.environ.get(env)) if env else False
            status = "✓ ready" if key_set else f"needs {env}"
            base_url = profile.get("base_url", "")
            provider = ""
            if "anthropic" in backend:
                provider = "Anthropic"
            elif "generativelanguage.googleapis" in base_url:
                provider = "Google"
            elif "groq.com" in base_url:
                provider = "Groq"
            elif "together.xyz" in base_url:
                provider = "Together"
            elif "fireworks.ai" in base_url:
                provider = "Fireworks"
            elif "mistral.ai" in base_url:
                provider = "Mistral"
            elif "deepseek.com" in base_url:
                provider = "DeepSeek"
            elif "openai" in backend:
                provider = "OpenAI"

            line = f"  {name:30s} {ctx_str:>6s} ctx  [{status}]"
            if backend == "anthropic":
                cloud_anthropic.append(line)
            elif backend == "openai" and not base_url:
                cloud_openai.append(line)
            else:
                cloud_other.append(f"  {name:30s} {ctx_str:>6s} ctx  {provider:10s} [{status}]")
        else:
            downloaded = _profile_has_local_file(name)
            status = "✓ downloaded" if downloaded else "not downloaded"
            backend_label = "torch" if backend == "pytorch" else "llama.cpp"
            line = f"  {name:30s} {ctx_str:>6s} ctx  ({backend_label})  [{status}]{alias_str}"
            if backend == "pytorch":
                local_torch.append(line)
            else:
                local_llama.append(line)

    print("═══ Local Models (requires GPU + downloaded model) ═══\n")
    if local_llama:
        print(" llama.cpp backend:")
        for line in local_llama:
            print(line)
    if local_torch:
        print("\n PyTorch/Transformers backend:")
        for line in local_torch:
            print(line)

    print("\n═══ Cloud Models (requires API key) ═══\n")
    if cloud_anthropic:
        print(" Anthropic (Claude):")
        for line in cloud_anthropic:
            print(line)
    if cloud_openai:
        print("\n OpenAI:")
        for line in cloud_openai:
            print(line)
    if cloud_other:
        print("\n Other providers:")
        for line in cloud_other:
            print(line)

    persisted = _read_persisted_model()
    current = os.environ.get("MAXIM_LLM_PROFILE", "") or persisted or "mistral-7b"
    print(f"\n═══ Active: {current} ═══")
    print("\nSet model: maxim --llm <model-name>")
    print("Download:  python -m maxim.models.download --llm <model-name>")
    print("Delete:    maxim --delete-model <model-name>")
    return 0


def _handle_delete_model(name: str) -> int:
    """Delete the requested local model and return 0."""
    from maxim.models.language.config import normalize_llm_profile
    from maxim.models.download import delete_llm

    canonical = normalize_llm_profile(name) or name
    if delete_llm(canonical):
        print("Done.")
    else:
        print("\nAvailable local models: python -m maxim.models.download --list")
    return 0


def _handle_clear_memory(scope: str, home_dir: str) -> int:
    """Clear the requested memory scope and return 0."""
    print(f"Clearing memory ({scope})...")
    results = _clear_memory(scope, home_dir)
    cleared = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Cleared {cleared}/{total} memory file(s).")
    return 0


def _bare_maxim_menu() -> int:
    """Interactive menu for bare ``maxim`` invocation (no args).

    Discovers available campaigns from scenarios/campaigns/*.yaml,
    presents a Rich-styled menu, and dispatches to the appropriate mode.
    Ctrl+C during a sim returns to the menu instead of exiting.
    """
    from pathlib import Path

    from maxim import get_version_info

    version = get_version_info().get("version", "dev")

    # Try to use Rich for styled output; fall back to plain text.
    _rich = False
    try:
        from rich.console import Console

        console = Console()
        _rich = True
    except ImportError:
        console = None  # type: ignore[assignment]

    # ── Discover campaigns (once) ─────────────────────────────────────
    campaigns: list[tuple[str, Path]] = []
    try:
        import yaml

        search_dirs = []
        bundled = Path(__file__).parent / "_data" / "campaigns"
        if bundled.is_dir():
            search_dirs.append(bundled)
        cwd_campaigns = Path("scenarios/campaigns")
        if cwd_campaigns.is_dir():
            search_dirs.append(cwd_campaigns)

        seen: set[str] = set()
        for d in search_dirs:
            for p in sorted(d.glob("*.yaml")):
                try:
                    with open(p) as f:
                        raw = yaml.safe_load(f)
                    if isinstance(raw, dict) and "campaign" in raw and "encounters" in raw:
                        name = (
                            raw.get("campaign", {}).get("name", p.stem)
                            if isinstance(raw.get("campaign"), dict)
                            else p.stem
                        )
                        if p.stem not in seen:
                            campaigns.append((name, p))
                            seen.add(p.stem)
                except Exception:
                    pass
    except ImportError:
        pass

    # ── Build options list (once) ─────────────────────────────────────
    options: list[tuple[str, str]] = []
    options.append(("Interactive chat", "interactive"))
    options.append(("Generative simulation (enter a goal)", "generative"))
    for name, _path in campaigns:
        options.append((name, f"campaign:{_path}"))
    options.append(("Run diagnostics (maxim doctor)", "doctor"))
    options.append(("Show help", "help"))

    # ── Menu loop — Ctrl+C returns here ───────────────────────────────
    while True:
        if _rich:
            _render_rich_menu(console, version, options, campaigns)
        else:
            _render_plain_menu(version, options, campaigns)

        try:
            if _rich:
                raw_choice = console.input(
                    f"  [bold dark_goldenrod]Choose [1-{len(options)}]:[/bold dark_goldenrod] "
                ).strip()
            else:
                raw_choice = input(f"  Choose [1-{len(options)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not raw_choice:
            return 0

        try:
            choice_idx = int(raw_choice) - 1
        except ValueError:
            if _rich:
                console.print(f"  [red]Invalid choice: {raw_choice}[/red]")
            else:
                print(f"  Invalid choice: {raw_choice}")
            continue

        if choice_idx < 0 or choice_idx >= len(options):
            if _rich:
                console.print(f"  [red]Invalid choice: {raw_choice}[/red]")
            else:
                print(f"  Invalid choice: {raw_choice}")
            continue

        _label, action = options[choice_idx]

        # ── Dispatch ──────────────────────────────────────────────────
        if action == "help":
            _show_help(console if _rich else None)
            continue

        if action == "doctor":
            from maxim.doctor.cli import run_doctor_subcommand

            run_doctor_subcommand([])
            continue

        try:
            _run_menu_sim(action)
        except KeyboardInterrupt:
            print("\n")
            continue

    return 0


def _render_rich_menu(console, version, options, campaigns) -> None:
    """Render the menu using Rich panels matching the Maxim display style."""
    from rich.panel import Panel
    from rich.text import Text

    # Title panel (matches MaximDisplay dark_violet style)
    console.print(
        Panel(
            Text(f"v{version} — bio-inspired cognitive architecture", style="italic"),
            title="[bold dark_violet]Maxim[/bold dark_violet]",
            border_style="dark_violet",
        )
    )

    # Build menu content
    lines = Text()
    idx = 1

    lines.append("\n  Start a session\n", style="bold dark_goldenrod")
    lines.append(f"    {idx}. ", style="dim")
    lines.append(f"{options[0][0]}\n", style="bold white")
    idx += 1
    lines.append(f"    {idx}. ", style="dim")
    lines.append(f"{options[1][0]}\n", style="bold white")
    idx += 1

    if campaigns:
        lines.append("\n  Run a campaign\n", style="bold dark_goldenrod")
        for name, _path in campaigns:
            lines.append(f"    {idx}. ", style="dim")
            lines.append(f"{name}\n", style="white")
            idx += 1

    lines.append("\n  Utilities\n", style="bold dark_goldenrod")
    for i in range(len(campaigns) + 2, len(options)):
        lines.append(f"    {idx}. ", style="dim")
        lines.append(f"{options[i][0]}\n", style="white")
        idx += 1

    console.print(Panel(lines, border_style="dim"))


def _render_plain_menu(version, options, campaigns) -> None:
    """Plain-text fallback menu (no Rich)."""
    print(f"\n  Maxim v{version} — bio-inspired cognitive architecture\n")
    idx = 1
    print("  Start a session:")
    print(f"    {idx}. {options[0][0]}")
    idx += 1
    print(f"    {idx}. {options[1][0]}")
    idx += 1
    if campaigns:
        print("\n  Run a campaign:")
        for name, _path in campaigns:
            print(f"    {idx}. {name}")
            idx += 1
    print("\n  Utilities:")
    for i in range(len(campaigns) + 2, len(options)):
        print(f"    {idx}. {options[i][0]}")
        idx += 1
    print()


def _show_help(console) -> None:
    """Show help text, styled if Rich is available."""
    lines = [
        'maxim --sim "test memory"        Run a generative simulation',
        "maxim --sim heist_v1.yaml        Run a YAML campaign",
        "maxim --sim interactive           Interactive chat session",
        "maxim doctor                      Check your environment",
        "maxim --list-models               See available LLM models",
        "maxim --help                      Full option reference",
    ]
    if console is not None:
        from rich.panel import Panel

        content = "\n".join(f"  {ln}" for ln in lines)
        console.print(Panel(content, title="[bold]Usage[/bold]", border_style="dim"))
    else:
        print("\n  Usage: maxim [OPTIONS]\n")
        for ln in lines:
            print(f"  {ln}")
        print()


def _setup_interactive_display() -> None:
    """Set up MaximDisplay for interactive mode (shared by menu + CLI paths)."""
    try:
        from maxim.interactive.display import create_display
        from maxim.simulation.sim_logger import (
            set_active_display,
            set_interactive_mode,
        )

        set_interactive_mode("on")

        # Set display tier to "bio" so bio-system events (hippocampus,
        # NAc, pain, etc.) are visible.  Without this the default CLEAN
        # tier filters them out.
        try:
            from maxim.simulation.sim_logger import set_display_tier

            set_display_tier("bio")
        except Exception:
            pass

        display = create_display("auto")
        if display is not None:
            display.start()
            set_active_display(display)
    except Exception:
        pass


def _run_menu_sim(action: str) -> None:
    """Run a sim from the menu. Raises KeyboardInterrupt to return to menu."""
    from pathlib import Path

    # For generative sims, get the goal BEFORE starting the display
    # (Rich Live + input() conflict).
    goal = None
    if action == "generative":
        try:
            goal = input("  Enter simulation goal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not goal:
            print("  No goal entered.")
            return

    _setup_interactive_display()

    if action == "interactive":
        from maxim.simulation.orchestrator import start_simulation_mode

        start_simulation_mode(
            goal="open interactive session — respond to user input naturally",
            persona="campaign",
            max_turns=200,
        )

    elif action == "generative":
        from maxim.simulation.orchestrator import start_simulation_mode

        start_simulation_mode(
            goal=goal,
            persona="campaign",
            max_turns=200,
        )

    elif action.startswith("campaign:"):
        campaign_path = Path(action.split(":", 1)[1])
        try:
            from maxim.simulation.dm_schema import load_campaign, validate_campaign
            from maxim.embodiment.component_registry import ComponentRegistry

            registry = ComponentRegistry(campaign_dir=str(campaign_path.parent))
            dm_campaign = load_campaign(campaign_path, registry=registry)
            errors = validate_campaign(dm_campaign)
            if errors:
                print(f"  Campaign validation failed ({len(errors)} errors):")
                for e in errors:
                    print(f"    - {e}")
                return

            from maxim.simulation.orchestrator import start_simulation_mode

            start_simulation_mode(
                goal=f"dm:{dm_campaign.name}",
                persona="dungeon_master",
                dm_campaign=dm_campaign,
                max_turns=100,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  Failed to load campaign: {e}")

    # Clean up display + reset all sim globals so state doesn't leak
    # into the next sim if the user returns to the menu.
    try:
        from maxim.simulation.sim_logger import (
            get_active_display,
            reset_sim_display_state,
            set_active_display,
        )

        display = get_active_display()
        if display is not None:
            display.stop()
            set_active_display(None)
        reset_sim_display_state()
        # Reset SEM tool discovery module-level state between sims
        from maxim.tools.discovery import reset_discovery_state

        reset_discovery_state()
    except Exception:
        pass


def _main_impl(argv: Sequence[str] | None = None) -> int:
    """Original ``main`` body.  Wrapped by :func:`main` for typed-error surfacing."""
    # Detect Blackwell GPU and apply GStreamer guards BEFORE any CUDA-touching
    # imports.  This was previously at module-import time; moved here so that
    # ``import maxim`` has no subprocess side effects.
    from maxim.utils.gpu_detect import ensure_blackwell_guards

    ensure_blackwell_guards()

    from maxim.utils.last_run import (
        should_save,
        save_last_run,
        load_last_run,
        load_all_runs,
        clear_last_run,
        format_all_runs,
    )

    raw_argv = list(argv) if argv is not None else sys.argv[1:]

    # Plan 1 R1 dual-format logging: honor MAXIM_LOG_FILE for every entry
    # path, not just the main sim loop. Subcommands (doctor, peer, tunnel)
    # short-circuit below before the sim loop calls configure_logging, so
    # we need an early call here to attach the StructuredFormatter JSONL
    # handler. Verbosity 0 (WARNING) is a safe default — the sim loop
    # still calls configure_logging(force=True) later with its own
    # verbosity, and the JSONL handler is deduped by absolute path.
    configure_logging(verbosity=0)

    # Plan 2 R2a: explicit role detection. Must run AFTER configure_logging
    # (so role_detected hits the JSONL handler if MAXIM_LOG_FILE is set) and
    # BEFORE subcommand dispatch (so `maxim doctor` / `maxim peer X` both
    # emit the event instead of only the sim loop). See
    # docs/plans/archive/llm_path_typed_errors.md R2a + feedback_subcommand_logging_gap.md.
    try:
        from maxim.runtime.role import detect_and_apply_role

        detect_and_apply_role(raw_argv)
    except Exception as _role_err:
        # Fix #3 (R2 review): role detection failures must not block
        # startup, but a silent DEBUG log hides real problems. Promote
        # to WARNING and emit a structured event so the failure shows
        # up in both human logs and the JSONL stream.
        _role_logger = logging.getLogger(__name__)
        _role_logger.warning("role detection failed: %s", _role_err)
        try:
            from maxim.utils.structured_logging import log_structured

            log_structured(
                _role_logger,
                logging.WARNING,
                event="role_detection_failed",
                data={"error": f"{type(_role_err).__name__}: {_role_err}"},
            )
        except Exception:
            pass

    # Stage A observability: print loud warning if trace flags are active so
    # users don't leave them on accidentally (log volume + request-id exposure).
    from maxim.models.language.mesh_trace import print_startup_warning_if_enabled

    print_startup_warning_if_enabled()

    # Subcommand dispatch — intercepts positional subcommands before argparse.
    if raw_argv and raw_argv[0] == "tunnel":
        from maxim.tunnel import run_tunnel_subcommand

        return run_tunnel_subcommand(raw_argv[1:])
    if raw_argv and raw_argv[0] == "doctor":
        from maxim.doctor import run_doctor_subcommand

        return run_doctor_subcommand(raw_argv[1:])
    if raw_argv and raw_argv[0] == "bench":
        # Plan 4 B: recovery-time (and future) benchmark harnesses.
        # Entry point lives in maxim.bench.cli — no CUDA imports,
        # no sim orchestrator, just a tight LLM call loop.
        from maxim.bench import run_bench_subcommand

        return run_bench_subcommand(raw_argv[1:])
    if raw_argv and raw_argv[0] == "roy":
        # Roy harness — persona-convergence crucible utilities.
        # Currently exposes `roy diff <session_a> <session_b>`, which
        # delegates to maxim.analysis.substrate_diff.
        from maxim.roy import run_roy_subcommand

        return run_roy_subcommand(raw_argv[1:])
    if raw_argv and raw_argv[0] == "peer":
        # `peer connect/show/forget` go to the peer config module;
        # `peer test` is a diagnostic that lives with `doctor`.
        peer_action = raw_argv[1] if len(raw_argv) > 1 else ""
        if peer_action in (
            "connect",
            "show",
            "key",
            "forget",
            "update",
            "restart",
            "llm",
            "version",
            "logs",
            "install",
            "deps",
            "list-nodes",
            "list-drained",
            "init-mesh",
            "add-node",
            "remove-node",
            "--node",
        ):
            from maxim.peer import run_peer_connect_subcommand

            return run_peer_connect_subcommand(raw_argv[1:])
        from maxim.doctor import run_peer_subcommand

        return run_peer_subcommand(raw_argv[1:])

    # Handle --last, --show-last, --clear-last before full parse
    if "--show-last" in raw_argv:
        print(format_all_runs())
        return 0
    if "--clear-last" in raw_argv:
        if clear_last_run():
            print("  Saved runs cleared.")
        else:
            print("  No saved runs to clear.")
        return 0
    if "--last" in raw_argv:
        # Parse just --last N
        idx_pos = raw_argv.index("--last")
        n = 1
        if idx_pos + 1 < len(raw_argv):
            try:
                n = int(raw_argv[idx_pos + 1])
            except ValueError:
                pass
        last = load_last_run(n)
        if last is None:
            runs = load_all_runs()
            if not runs:
                print("  No saved runs. Run a simulation first.")
            else:
                print(f"  No run at index {n}. Available:")
                print(format_all_runs())
            return 1
        print(f"  Re-running: {last['command']}")
        raw_argv = last["args"]

    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    # ── Deterministic seeding (S4) — must run before heavy imports ───
    if getattr(args, "seed", None) is not None:
        from maxim.utils.seeding import seed_all

        seed_all(args.seed)

    # ── Force-kill on double Ctrl+C ──────────────────────────────────
    # First Ctrl+C signals the LLM cancellation primitive and raises
    # KeyboardInterrupt in the main thread for graceful shutdown. If the
    # user hits Ctrl+C again while shutdown is in progress, force-kill the
    # LLM server subprocess and exit immediately. Prevents ghost processes
    # and prevents in-flight retry loops from burning cloud credits after
    # the user has asked to stop.
    _shutting_down = [False]

    def _force_exit_handler(signum, frame):
        # Always signal LLM cancellation first — this wakes up any backend
        # retry loop that's mid-backoff and lets it abandon the request
        # before Python gets a chance to unwind via KeyboardInterrupt.
        # Must be best-effort: if the import fails (unlikely), we still
        # want the rest of the handler to run.
        try:
            from maxim.models.language.cancellation import request_shutdown

            request_shutdown()
        except Exception:
            pass

        if _shutting_down[0]:
            # Second interrupt — force kill everything and exit
            try:
                from maxim.runtime.lane_backends import stop_active_spawner

                stop_active_spawner()
            except Exception:
                pass
            print("\n  Forced shutdown.", file=sys.stderr)
            os._exit(1)
        _shutting_down[0] = True
        raise KeyboardInterrupt

    import signal as _sig

    _sig.signal(_sig.SIGINT, _force_exit_handler)

    # ── Early leader proxy bootstrap ─────────────────────────────────
    # Start the LeaderProxy BEFORE _normalize_args — that function can
    # trigger heavy imports (llama_cpp, torch) for model validation,
    # which takes 5-15s on CUDA systems. The proxy must be reachable
    # immediately after os.execv restart so peers don't time out.
    # via the Cloudflare tunnel, regardless of what mode maxim enters
    # (sim, interactive, agentic, etc.). This ensures `maxim peer update`,
    # `maxim peer restart`, `maxim peer logs`, and `maxim peer version`
    # always work. The proxy is a daemon thread — it dies with the process.
    detected_role = None
    try:
        from maxim.runtime.leader_mode import detect_role

        detected_role = detect_role()
        if detected_role.role == "leader":
            import os as _os

            _os.environ.setdefault("MAXIM_ALLOW_REMOTE_UPDATE", "1")
            from maxim.runtime.leader_proxy import start_leader_proxy

            _api_key = None
            try:
                from maxim.tunnel.keys import read_key

                _api_key = read_key()
            except Exception:
                # Real leader: surface to operator. See docs/troubleshooting/leader_proxy_debug.md
                from maxim.utils.logging import user_warn

                user_warn(
                    "Could not read API key — proxy will run without auth",
                    fix="Run `maxim tunnel setup` to configure auth, or set MAXIM_API_KEY.",
                    source="leader-boot",
                    event="leader_boot_no_api_key",
                )
            _proxy = start_leader_proxy(api_key=_api_key, bind_host=detected_role.bind_host)
            if _proxy is None:
                from maxim.utils.logging import user_warn

                user_warn(
                    "LeaderProxy failed to start",
                    fix="Check that the listen port is free (`lsof -i :7077`) or set MAXIM_PROXY_PORT.",
                    source="leader-boot",
                    event="leader_boot_proxy_failed",
                )
    except Exception as _e:
        # Only surface to operators when role resolved to leader. Solo users
        # should never see leader-boot noise.
        if detected_role is not None and detected_role.role == "leader":
            from maxim.utils.logging import user_warn

            user_warn(
                f"Early proxy boot failed: {_e}",
                fix="See traceback above; usually a port conflict or missing CUDA. Run `maxim doctor` for diagnostics.",
                source="leader-boot",
                event="leader_boot_exception",
                data={"exc_type": type(_e).__name__},
            )
            import traceback as _tb

            _tb.print_exc()
        else:
            logging.getLogger(__name__).debug("Early proxy boot skipped: %s", _e)

    # ── Normalize args (after proxy boot — can trigger heavy CUDA imports) ──
    _normalize_args(args)

    # Save this invocation if it's meaningful
    if should_save(raw_argv):
        save_last_run(raw_argv)

    # ── Discrete management subcommands (each returns 0 on completion) ──
    if bool(getattr(args, "list_models", False)):
        return _handle_list_models()

    delete_model_name = getattr(args, "delete_model", None)
    if delete_model_name:
        return _handle_delete_model(delete_model_name)

    if bool(getattr(args, "clear_cache", False)):
        removed = _clear_python_cache()
        print(f"Cleared {removed} __pycache__ director{'y' if removed == 1 else 'ies'}.", file=sys.stderr)

    clear_memory = getattr(args, "clear_memory", None)
    if clear_memory is not None:
        return _handle_clear_memory(clear_memory, args.home_dir)

    # ── Asset Foundry dispatch ──────────────────────────────────────────
    _foundry_theme = getattr(args, "foundry", None)
    if _foundry_theme:
        from maxim.simulation.foundry import FoundryRunner

        # Build a lightweight LLM router for generation when --llm is set.
        # The foundry dispatches before the full agent stack, so it creates
        # its own router from the profile name (same pattern as ExecAgent).
        _foundry_llm = None
        _foundry_profile = str(getattr(args, "language_model", "") or "").strip()
        if _foundry_profile:
            try:
                from maxim.models.language.config import load_llm_config
                from maxim.models.language.router import LLMRouter

                _foundry_cfg = load_llm_config(profile_override=_foundry_profile)
                _foundry_llm = LLMRouter(_foundry_cfg)
                if hasattr(_foundry_llm, "warmup"):
                    _foundry_llm.warmup()
                if hasattr(_foundry_llm, "wait_ready"):
                    print(f"  Waiting for LLM ({_foundry_profile}) to load...")
                    if not _foundry_llm.wait_ready(timeout=120.0):
                        print("  WARNING: LLM failed to load — foundry will use template fallback.")
                        if hasattr(_foundry_llm, "shutdown"):
                            _foundry_llm.shutdown()
                        _foundry_llm = None
                    else:
                        print(f"  LLM ready: {_foundry_profile}")
            except Exception as e:
                print(f"  WARNING: LLM init failed ({e}) — using template fallback.")

        runner = FoundryRunner(
            theme=_foundry_theme,
            genre=getattr(args, "foundry_genre", "fantasy"),
            category=getattr(args, "foundry_category", None),
            llm_router=_foundry_llm,
            dry_run=bool(getattr(args, "foundry_dry_run", False)),
        )
        result = runner.run(count=int(getattr(args, "foundry_count", 10) or 10))

        print(f"\nFoundry complete: {result.output_dir}")
        print(f"Generated: {result.generated} | Validated: {result.validated} | Tested: {result.tested}")
        print(f"Promoted: {len(result.promoted)} | Review: {len(result.review)} | Rejected: {len(result.rejected)}")
        if result.promoted:
            print("\nPromoted components:")
            for s in result.promoted:
                print(f"  {s.candidate_name} ({s.total_score:.2f})")

        return 0

    # ── Cross-flag validation ───────────────────────────────────────────
    if getattr(args, "sim_report", None) and getattr(args, "sim", None) is None:
        print("Error: --sim-report requires --sim.", file=sys.stderr)
        return 1
    if getattr(args, "report_json", None) and getattr(args, "sim", None) is None:
        print("Error: --report-json requires --sim.", file=sys.stderr)
        return 1
    # --report-json is read by simulation/orchestrator.py via environment
    # rather than by threading through start_simulation_mode's signature
    # (which has 5+ invocation sites).  Set it once here and let the
    # orchestrator emit when it builds the report.
    _report_json_arg = getattr(args, "report_json", None)
    if _report_json_arg:
        os.environ["MAXIM_REPORT_JSON"] = _report_json_arg

    # Scenario generation if requested
    gen_description = getattr(args, "generate_simulation", None)
    if gen_description is not None:
        from pathlib import Path
        from maxim.simulation.simulation_generator import generate_scenario

        output_path = getattr(args, "output", None)
        output = Path(output_path) if output_path else None
        llm_profile = str(getattr(args, "language_model", "") or "").strip() or None

        try:
            yaml_str = generate_scenario(gen_description, output_path=output, llm_profile=llm_profile)
            if output is None:
                print(yaml_str)
        except Exception as e:
            print(f"Error generating scenario: {e}")
            sys.exit(1)
        sys.exit(0)

    # ── Top-level --benchmark command ─────────────────────────────────
    _benchmark_arg = getattr(args, "benchmark", None)
    if _benchmark_arg is not None:
        from maxim.simulation.benchmark import BenchmarkRunner

        models_raw = getattr(args, "models", None)
        if not models_raw:
            print("  Error: --models is required for --benchmark")
            print("  Example: maxim --benchmark tier1 --models mistral-7b,qwen2.5-14b")
            sys.exit(1)

        models = [m.strip() for m in models_raw.split(",") if m.strip()]
        campaign = getattr(args, "campaign", None)
        if not campaign:
            # Default campaign suite based on tier
            _tier_campaigns = {
                "tier1": "scenarios/benchmarks/cognitive_suite.yaml",
                "tier2": "scenarios/benchmarks/biosystem_suite.yaml",
                "tier3": "scenarios/benchmarks/embodiment_suite.yaml",
                "all": "scenarios/benchmarks/cognitive_suite.yaml",
            }
            campaign = _tier_campaigns.get(_benchmark_arg, _tier_campaigns["all"])

        runner = BenchmarkRunner(
            models=models,
            suite_path=campaign,
            runs=getattr(args, "runs", 1) or 1,
            output_dir=getattr(args, "benchmark_output", None),
            baseline_path=getattr(args, "baseline", None),
            persona=_resolve_persona(args, default="campaign") or "neutral",
            max_turns=50,
            response_timeout=60.0,
            debug=bool(getattr(args, "debug", "")),
        )

        report = runner.run()
        print(report.summary_table())

        report_dir = runner.save_report(report)
        print(f"\n  Report saved: {report_dir}\n")

        if getattr(args, "write_paper", False):
            print("  Generating comparative paper...")
            paper_path = runner.write_paper(report, report_dir)
            if paper_path:
                print(f"  Paper saved: {paper_path}\n")
            else:
                print("  Paper generation failed (LLM unavailable)\n")

        all_passed = all(mr.passed for mr in report.results.values())
        sys.exit(0 if all_passed else 1)

    # Validate simulation-only flags aren't used without --sim agent/research
    sim_path = getattr(args, "sim", None)
    _sim_mode = str(sim_path).strip().lower() if sim_path is not None else ""
    _is_sim_mode = _sim_mode in ("agent", "research")
    if not _is_sim_mode:
        if getattr(args, "sim_goal", None) is not None:
            print("Error: --goal / --sim-goal requires --sim agent or --sim research.")
            print('  Usage: maxim --sim agent --goal "test safety" --sim-mode adversarial')
            print('         maxim --sim research --goal "hippocampal recall" --campaign <yaml>')
            sys.exit(1)
        if getattr(args, "sim_persona", "adversarial") != "adversarial":
            print("Error: --sim-mode / --persona / --sim-persona requires --sim agent (simulation mode).")
            print('  Usage: maxim --sim agent --goal "test safety" --sim-mode adversarial')
            sys.exit(1)
        if getattr(args, "resume_sim", None) is not None and sim_path is None:
            print("Error: --resume-sim requires --sim (simulation mode).")
            print('  Usage: maxim --sim "test goal" --resume-sim SESSION_ID')
            print('         maxim --sim agent --goal "continue" --resume-sim SESSION_ID')
            sys.exit(1)

    # Simulation mode if requested — runs full agentic pipeline with fake percepts
    if sim_path is not None:
        import json as _json
        from pathlib import Path

        from maxim.simulation.scenario_source import ScenarioSource
        from maxim.simulation.sinks import RecordingSink
        from maxim.simulation.validation import validate_expectations

        # ── Safety net: detect stale sim processes from previous runs ──
        # Graceful shutdown (Ctrl+C → orchestrator cleanup → LLM cancellation)
        # handles the common case, but kill -9, crashes, and detached shells
        # can still leave zombie maxim processes that keep hitting the LLM
        # backend and burning cloud credits. Scan for them before starting
        # anything new. Warn always; reap only when user opts in explicitly
        # (--reap-orphans flag or MAXIM_REAP_ORPHANS=1).
        try:
            from maxim.runtime.orphan_reaper import (
                find_orphans,
                reap_orphans,
                warn_about_orphans,
            )

            _orphans = find_orphans()
            if _orphans:
                warn_about_orphans(_orphans)
                _reap_requested = bool(getattr(args, "reap_orphans", False)) or os.environ.get(
                    "MAXIM_REAP_ORPHANS", ""
                ).strip() not in ("", "0", "false", "no")
                if _reap_requested:
                    _killed = reap_orphans(_orphans)
                    print(f"  Reaped {_killed} stale sim process(es).\n", file=sys.stderr)
        except Exception as _e:
            logging.getLogger("maxim").debug("Orphan reaper check failed: %s", _e)

        # Parse --debug subsystem selection
        _debug_raw = getattr(args, "debug", None)
        _debug_all = _debug_raw == "all"
        # Human-readable aliases for bio-system debug subsystems
        _DEBUG_ALIASES = {
            "memory": "hippo",
            "recall": "hippo",
            "causal": "nac",
            "reward": "nac",
            "semantic": "atl",
            "concepts": "atl",
            "temporal": "scn",
            "clock": "scn",
        }
        _debug_subs = set()
        if _debug_raw and _debug_raw != "all":
            raw_subs = {s.strip().lower() for s in _debug_raw.split(",")}
            _debug_subs = {_DEBUG_ALIASES.get(s, s) for s in raw_subs}
        if _debug_all or "hippo" in _debug_subs:
            os.environ["MAXIM_HIPPO_TRACE"] = "1"
        if _debug_all or "nac" in _debug_subs:
            os.environ["MAXIM_NAC_TRACE"] = "1"
        if _debug_all or "atl" in _debug_subs:
            os.environ["MAXIM_ATL_TRACE"] = "1"

        # -- Detect sim mode from --sim value ---------------------------------
        _sim_val = str(sim_path).strip()
        _sim_val_lower = _sim_val.lower()

        # New unified detection:
        # - "agent", "research", "benchmark" → legacy aliases
        # - ends with .yaml/.yml → YAML campaign/scenario
        # - "interactive" or empty → REPL
        # - anything else → goal string for generative campaign
        _is_yaml = _sim_val.endswith((".yaml", ".yml"))
        _is_legacy_agent = _sim_val_lower == "agent"
        _is_legacy_research = _sim_val_lower == "research"
        _is_legacy_benchmark = _sim_val_lower == "benchmark"

        # Legacy aliases: --sim agent/research/benchmark still work
        # but the preferred forms are --sim "goal" / --research / --benchmark
        _is_interactive = _sim_val_lower == "interactive"
        _is_goal_string = not (
            _is_yaml or _is_legacy_agent or _is_legacy_research or _is_legacy_benchmark or _is_interactive
        )

        # If --research flag is set with a goal string, use research mode
        _wants_research = getattr(args, "research", False)
        # If --goal is set explicitly, it overrides the --sim value as goal
        _explicit_goal = getattr(args, "sim_goal", None)

        # ── Apply sim display config EARLY so every subcommand sees it ───
        # Previously set_display_tier was only called inside the scenario-file
        # branch further down, so freeform-goal / research / benchmark / DM
        # paths all ran with the default CLEAN tier — bio/debug display modes
        # were silently no-ops for every path except explicit YAML scenarios.
        # This is upstream of enable_sim_logging because sim_logger reads the
        # tier global when each event fires, not when logging is enabled.
        try:
            from maxim.simulation.sim_logger import (
                set_display_tier as _set_display_tier_early,
                set_interactive_mode as _set_interactive_mode_early,
                set_show_channels as _set_show_channels_early,
            )

            _display_arg_early = getattr(args, "display", "bio")
            _set_display_tier_early(_display_arg_early)
            if _display_arg_early == "debug":
                _set_show_channels_early("all")
            # Auto-interactive detection: when the user did not pass
            # --interactive on the command line, default to ON for DM
            # campaigns (where the user makes choices) and OFF for
            # generative sims (where the persona drives input).
            _interactive_explicit = "--interactive" in (raw_argv or [])
            if _interactive_explicit:
                _interactive_str = str(getattr(args, "interactive", "true")).strip().lower()
                _set_interactive_mode_early("on" if _interactive_str not in ("false", "0", "no", "off") else "off")
            else:
                _wants_dm_early = bool(getattr(args, "dm", False))
                _is_dm_yaml_early = False
                if _is_yaml:
                    try:
                        import yaml as _yaml_probe

                        with open(Path(_sim_val).resolve()) as _f:
                            _probe = _yaml_probe.safe_load(_f)
                        _is_dm_yaml_early = isinstance(_probe, dict) and "campaign" in _probe and "encounters" in _probe
                    except Exception:
                        _is_dm_yaml_early = False
                # CLI with TTY → interactive ON (human at a terminal).
                # API, CI, piped → interactive OFF (no TTY).
                _is_tty = sys.stdout.isatty()
                _set_interactive_mode_early("on" if (_wants_dm_early or _is_dm_yaml_early or _is_tty) else "off")
            _show_channels_early = getattr(args, "show_channels", None)
            if _show_channels_early:
                _set_show_channels_early(_show_channels_early)
        except Exception as _e:
            logging.getLogger("maxim").warning("Failed to apply sim display config: %s", _e)

        # ── MaximDisplay — rich panel UI when interactive mode is on ──
        # Must happen BEFORE any sim path branches, since generative
        # campaigns, DM campaigns, and agent sims all read from the
        # active display via sim_logger routing.
        _maxim_display = None
        try:
            from maxim.interactive.display import create_display
            from maxim.simulation.sim_logger import (
                get_interactive_mode,
                set_active_display,
            )

            if get_interactive_mode().value == "on":
                _maxim_display = create_display("auto")
                if _maxim_display is not None:
                    _maxim_display.start()
                    set_active_display(_maxim_display)
                    logging.getLogger("maxim").info("MaximDisplay started (interactive mode)")
        except Exception as _disp_exc:
            logging.getLogger("maxim").warning("MaximDisplay unavailable: %s", _disp_exc)

        # ── E0: extract entity_ref for sim embodiment ──────────────────
        # No _is_sim_mode gate — all sim paths (generative, DM, agent,
        # interactive) support entity_ref. The enclosing `if sim_path`
        # block already gates non-sim paths. Pre-merge review cross-
        # confirmed that the _is_sim_mode gate silently dropped
        # --embodiment for the primary (generative/DM) use cases.
        _sim_entity_ref = getattr(args, "embodiment", None)

        # Resolve embodiment depth (level 2 default, level 3 with --deep-embodiment)
        try:
            from maxim.embodiment.resolution import get_embodiment_depth

            get_embodiment_depth(args)
        except Exception:
            pass

        # Default to base_humanoid embodiment for sim mode (0.7+).
        # The agent gets physical affordances (move, look, pick_up, use, speak, rest)
        # which unlocks the full 0.7 chain: Acting Coach, imagination, scene-scoped tools.
        # Pass --no-embodiment to disable (pre-0.7 behavior).
        _no_embodiment = bool(getattr(args, "no_embodiment", False))
        if _sim_entity_ref is None and not _no_embodiment:
            _sim_entity_ref = "bodies/base_humanoid"
            logging.getLogger("maxim").info(
                "Sim embodiment defaulting to bodies/base_humanoid (pass --no-embodiment to disable)"
            )

        # ── E3: Pre-sim auto-curation ─────────────────────────────────
        _auto_curate = bool(getattr(args, "auto_curate", False))
        _no_curate = bool(getattr(args, "no_curate", False))
        if _auto_curate and not _no_curate and _sim_entity_ref:
            _curate_threshold = int(getattr(args, "curate_threshold", 5) or 5)
            # Infer genre from entity_ref or --foundry-genre flag
            _curate_genre = getattr(args, "foundry_genre", "fantasy") or "fantasy"

            try:
                from maxim.embodiment.component_registry import ComponentRegistry
                from maxim.simulation.foundry import auto_curate

                _curate_registry = ComponentRegistry()

                # Build ComponentIndex for dedup (optional — degrades gracefully)
                _curate_index = None
                try:
                    from maxim.embodiment.component_index import ComponentIndex

                    _curate_index = ComponentIndex(_curate_registry)
                except Exception as _idx_err:
                    logging.getLogger("maxim").debug("ComponentIndex unavailable for dedup: %s", _idx_err)

                # Reuse the LLM router from --llm if available
                _curate_llm = None
                _curate_profile = str(getattr(args, "language_model", "") or "").strip()
                if _curate_profile:
                    try:
                        from maxim.models.language.config import load_llm_config
                        from maxim.models.language.router import LLMRouter

                        _curate_cfg = load_llm_config(profile_override=_curate_profile)
                        _curate_llm = LLMRouter(_curate_cfg)
                        if hasattr(_curate_llm, "warmup"):
                            _curate_llm.warmup()
                    except Exception as _llm_err:
                        logging.getLogger("maxim").debug("LLM init for curation failed: %s", _llm_err)

                print(f"Auto-curating {_curate_genre} components (threshold={_curate_threshold})...")
                _curate_report = auto_curate(
                    genre=_curate_genre,
                    threshold=_curate_threshold,
                    registry=_curate_registry,
                    component_index=_curate_index,
                    llm_router=_curate_llm,
                )
                if _curate_report.total_promoted > 0:
                    print(
                        f"  Curated: {_curate_report.total_promoted} promoted, "
                        f"{_curate_report.total_skipped_dedup} dedup-skipped"
                    )
                elif _curate_report.categories_below_threshold == 0:
                    print("  Coverage OK — no curation needed")
                else:
                    print(
                        f"  Generated {_curate_report.total_generated} candidates, "
                        f"none promoted (below score threshold)"
                    )
            except Exception as _cur_err:
                logging.getLogger("maxim").warning("Auto-curation failed: %s", _cur_err)
                print(f"  WARNING: Auto-curation failed ({_cur_err})", file=sys.stderr)
        elif _auto_curate and not _no_curate and not _sim_entity_ref:
            print(
                "  NOTE: --auto-curate requires --embodiment to determine genre context; skipping curation.",
                file=sys.stderr,
            )

        # ── Generative campaign mode (new default for goal strings) ──
        if _is_goal_string and not _is_legacy_agent:
            goal = _explicit_goal or _sim_val
            _aut_mode_val = getattr(args, "aut_mode", "llm-primary")
            # --research with --aut-mode substrate-primary means
            # "per-tick substrate telemetry on", not "spin up the
            # multi-agent paper-writing harness". Phase 0 of
            # docs/plans/grounded_language_acquisition.md.
            if _wants_research and _aut_mode_val != "substrate-primary":
                # Generative + research report
                from maxim.simulation.research_orchestrator import start_research_mode

                campaign = getattr(args, "campaign", None)
                language_model = str(getattr(args, "language_model", "") or "").strip() or None
                result = start_research_mode(
                    goal=goal,
                    campaign=campaign,
                    language_model=language_model,
                    aut_model=getattr(args, "aut_model", None),
                    debug=bool(_debug_raw),
                    sandbox_backend=getattr(args, "sandbox_backend", "auto"),
                )
                sys.exit(0 if result.review_verdict != "reject" else 1)
            else:
                # Pure generative campaign — use the orchestrator.
                # When the goal matches a builtin narrative arc (cradle,
                # memory_recall, etc.), enable the generative campaign runner
                # so the narrator drives multi-turn structured phases.
                from maxim.simulation.orchestrator import start_simulation_mode

                persona = _resolve_persona(args, default="campaign") or "neutral"
                debug = bool(_debug_raw)
                resume_sim = getattr(args, "resume_sim", None)

                # Auto-detect generative mode: if goal matches a builtin arc,
                # route through the generative campaign runner.
                _use_generative = False
                try:
                    from maxim.simulation.arcs import select_arc_for_goal

                    if select_arc_for_goal(goal) is not None:
                        _use_generative = True
                except Exception:
                    pass

                result = start_simulation_mode(
                    goal=goal,
                    persona=persona,
                    debug=debug,
                    resume_session=resume_sim,
                    continuous=bool(getattr(args, "continuous", False)),
                    no_sim_env=bool(getattr(args, "no_sim_env", False)),
                    sandbox_backend=getattr(args, "sandbox_backend", "auto"),
                    sandbox_image=getattr(args, "sandbox_image", "python:3.12-slim"),
                    sandbox_network=getattr(args, "sandbox_network", "none"),
                    aut_model=getattr(args, "aut_model", None),
                    aut_mode=_aut_mode_val,
                    research_telemetry=bool(_wants_research),
                    max_turns=int(getattr(args, "sim_max_turns", 50) or 50),
                    entity_ref=_sim_entity_ref,
                    generative=_use_generative,
                )
                sys.exit(0 if result.finish_reason != "error" else 1)

        # ── Legacy: agent mode (deprecated alias) ──
        _sim_agent = _is_legacy_agent
        if _sim_agent:
            from maxim.simulation.orchestrator import start_simulation_mode

            goal = getattr(args, "sim_goal", None) or "test the agent's capabilities"
            persona = _resolve_persona(args, default="adversarial") or "neutral"
            debug = bool(_debug_raw)
            resume_sim = getattr(args, "resume_sim", None)

            result = start_simulation_mode(
                goal=goal,
                persona=persona,
                debug=debug,
                resume_session=resume_sim,
                continuous=bool(getattr(args, "continuous", False)),
                no_sim_env=bool(getattr(args, "no_sim_env", False)),
                sandbox_backend=getattr(args, "sandbox_backend", "auto"),
                sandbox_image=getattr(args, "sandbox_image", "python:3.12-slim"),
                sandbox_network=getattr(args, "sandbox_network", "none"),
                aut_model=getattr(args, "aut_model", None),
                aut_mode=getattr(args, "aut_mode", "llm-primary"),
                max_turns=int(getattr(args, "sim_max_turns", 50) or 50),
                entity_ref=_sim_entity_ref,
            )
            sys.exit(0 if result.finish_reason != "error" else 1)

        # Check for benchmark mode (multi-model comparison)
        _sim_benchmark = str(sim_path).strip().lower() == "benchmark"
        if _sim_benchmark:
            from maxim.simulation.benchmark import BenchmarkRunner

            models_raw = getattr(args, "models", None)
            if not models_raw:
                print("  Error: --models is required for --sim benchmark")
                print("  Example: maxim --sim benchmark --models mistral-7b,qwen2.5-14b --campaign suite.yaml")
                sys.exit(1)

            models = [m.strip() for m in models_raw.split(",") if m.strip()]
            campaign = getattr(args, "campaign", None)
            if not campaign:
                print("  Error: --campaign is required for --sim benchmark")
                print(
                    "  Example: maxim --sim benchmark --models mistral-7b --campaign scenarios/benchmarks/cognitive_suite.yaml"
                )
                sys.exit(1)

            runner = BenchmarkRunner(
                models=models,
                suite_path=campaign,
                runs=getattr(args, "runs", 1) or 1,
                output_dir=getattr(args, "benchmark_output", None),
                baseline_path=getattr(args, "baseline", None),
                persona=_resolve_persona(args, default="campaign") or "neutral",
                max_turns=50,
                response_timeout=60.0,
                debug=bool(_debug_raw),
            )

            report = runner.run()
            print(report.summary_table())

            # Save report
            report_dir = runner.save_report(report)
            print(f"\n  Report saved: {report_dir}\n")

            # Optional: generate comparative paper
            if getattr(args, "write_paper", False):
                print("  Generating comparative paper...")
                paper_path = runner.write_paper(report, report_dir)
                if paper_path:
                    print(f"  Paper saved: {paper_path}\n")
                else:
                    print("  Paper generation failed (LLM unavailable)\n")

            # Exit with 0 if all models passed, 1 if any failed
            all_passed = all(mr.passed for mr in report.results.values())
            sys.exit(0 if all_passed else 1)

        # Check for research mode (legacy alias — deprecated, use --research flag)
        _sim_research = _is_legacy_research
        if _sim_research:
            from maxim.simulation.research_orchestrator import start_research_mode

            goal = getattr(args, "sim_goal", None) or "investigate the research question"
            debug = bool(_debug_raw)
            campaign = getattr(args, "campaign", None)
            language_model = str(getattr(args, "language_model", "") or "").strip() or None

            result = start_research_mode(
                goal=goal,
                campaign=campaign,
                language_model=language_model,
                aut_model=getattr(args, "aut_model", None),
                debug=debug,
                sandbox_backend=getattr(args, "sandbox_backend", "auto"),
            )
            sys.exit(0 if result.review_verdict != "reject" else 1)

        # ── Auto-detect DM campaigns from YAML ──
        # If --sim points to a YAML with 'campaign:' + 'encounters:' keys, it's a DM campaign.
        # Also triggered by --dm flag with a goal string (future: generative DM).
        _wants_dm = getattr(args, "dm", False)
        if _is_yaml:
            _yaml_path = Path(sim_path).resolve()
            if _yaml_path.exists():
                try:
                    import yaml as _yaml

                    with open(_yaml_path) as _yf:
                        _raw = _yaml.safe_load(_yf)
                    if isinstance(_raw, dict) and "campaign" in _raw and "encounters" in _raw:
                        from maxim.simulation.dm_schema import load_campaign, validate_campaign
                        from maxim.simulation.orchestrator import start_simulation_mode

                        from maxim.embodiment.component_registry import ComponentRegistry

                        _registry = ComponentRegistry(campaign_dir=str(Path(_yaml_path).parent))
                        dm_campaign = load_campaign(_yaml_path, registry=_registry)
                        errors = validate_campaign(dm_campaign)
                        if errors:
                            print(f"Campaign validation failed ({len(errors)} errors):")
                            for e in errors:
                                print(f"  - {e}")
                            sys.exit(1)

                        debug = bool(_debug_raw)
                        result = start_simulation_mode(
                            goal=f"dm:{dm_campaign.name}",
                            persona="dungeon_master",
                            debug=debug,
                            no_sim_env=bool(getattr(args, "no_sim_env", False)),
                            sandbox_backend=getattr(args, "sandbox_backend", "auto"),
                            dm_campaign=dm_campaign,
                            max_turns=int(getattr(args, "sim_max_turns", 50) or 50),
                            entity_ref=_sim_entity_ref,
                        )
                        sys.exit(0 if result.finish_reason != "error" else 1)
                except Exception:
                    pass  # Not a DM campaign — fall through to normal YAML handling

        # --dm flag with a goal string = generative narrative campaign
        if _wants_dm and _is_goal_string:
            from maxim.simulation.orchestrator import start_simulation_mode

            debug = bool(_debug_raw)
            result = start_simulation_mode(
                goal=sim_path,  # The goal string
                persona="dungeon_master",
                debug=debug,
                no_sim_env=bool(getattr(args, "no_sim_env", False)),
                sandbox_backend=getattr(args, "sandbox_backend", "auto"),
                generative=True,  # Use generative campaign runner
                arc_yaml=getattr(args, "arc", None),
                max_turns=int(getattr(args, "sim_max_turns", 50) or 50),
                entity_ref=_sim_entity_ref,
            )
            sys.exit(0 if result.finish_reason != "error" else 1)

        # Check for interactive mode
        if _is_interactive:
            scenario_files = []  # No files — interactive REPL handles everything
        else:
            sim_path = Path(sim_path).resolve()  # Resolve to absolute before CWD change
            if sim_path.is_dir():
                scenario_files = sorted(sim_path.glob("*.yaml")) + sorted(sim_path.glob("*.yml"))
            else:
                scenario_files = [sim_path]

        if not scenario_files and not _is_interactive:
            print(f"No scenario files found at {sim_path}")
            sys.exit(1)

        # Force agentic mode with supervised autonomy for simulation
        args.mode = "agentic"
        if not getattr(args, "autonomy", None) or args.autonomy == "planning":
            args.autonomy = "supervised"  # Let agent attempt actions; FearAgent + policy gate
        # Use a reasonable step limit so scenarios don't run forever
        if not getattr(args, "epochs", None):
            args.epochs = 200

        # Create a temporary sandbox directory within the workspace
        # All filesystem operations are confined here, destroyed after the run
        if not _is_interactive:
            # Single/batch scenario mode: set up sandbox and load scenario
            import tempfile

            sim_workspace = Path(getattr(args, "home_dir", "data")) / "sim_sandbox"
            sim_workspace.mkdir(parents=True, exist_ok=True)
            sim_tmpdir = Path(
                tempfile.mkdtemp(
                    prefix=f"sim_{time.strftime('%Y%m%d_%H%M%S')}_",
                    dir=str(sim_workspace),
                )
            )
            args._sim_original_cwd = os.getcwd()
            import atexit

            atexit.register(lambda cwd=args._sim_original_cwd: os.chdir(cwd))
            os.chdir(str(sim_tmpdir))
            _sim_debug = bool(getattr(args, "debug", False) or getattr(args, "sim_debug", False))
            if _sim_debug:
                print(f"  Simulation sandbox: {sim_tmpdir}")

            # (all_results / any_failed removed — unused; batch scenario
            # result aggregation is handled by ScenarioRunner directly)

            from maxim.simulation.sim_logger import (
                enable_sim_logging,
                disable_sim_logging,
            )

            sim_log_path = str(sim_workspace / f"sim_log_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
            enable_sim_logging(log_path=sim_log_path, debug=_sim_debug)

            # Display tier / interactive / show-channels are applied earlier
            # in the sim block so all sim subcommands share the same config.

            for scenario_file in scenario_files:
                if _sim_debug:
                    print(f"\nRunning scenario: {scenario_file.name}")
                    print(f"  Loading full agentic pipeline (autonomy={args.autonomy})...")

                source = ScenarioSource(scenario_file)
                sink = RecordingSink()

                args._sim_source = source
                args._sim_sink = sink
                args._sim_scenario_file = scenario_file
                args._sim_tmpdir = sim_tmpdir
                break  # Process one scenario, let the main loop handle it

        # Fall through to the main loop with mode="agentic"
        # The agentic block will detect args._sim_source and wire it in

    # Architecture audit if requested
    if getattr(args, "audit_architecture", False):
        from maxim.utils.audit import audit_architecture

        violations = audit_architecture()
        if violations:
            print(f"Found {len(violations)} architecture violations:")
            for v in violations:
                print(f"  {v.file}:{v.line} — {v.rule} (imports {v.imported_module})")
            sys.exit(1)
        else:
            print("No architecture violations found.")
            sys.exit(0)

    # If no meaningful action was specified, show quick-start guidance
    _has_sim = sim_path is not None
    _has_mode_override = "--mode" in (raw_argv or [])
    _has_robot = getattr(args, "robot_name", "reachy_mini") != "reachy_mini" or "--robot" in (raw_argv or [])
    _is_leader = os.environ.get("MAXIM_ROLE", "").strip().lower() == "leader"
    _has_llm = os.environ.get("MAXIM_LLM_ENABLED", "").strip() == "1"
    # --llm / --language-model on the command line is a meaningful action
    # by itself ("start maxim with this model loaded"). Previously this
    # flag set args.language_model but the quick-start gate only checked
    # the MAXIM_LLM_ENABLED env var, so `maxim --llm qwen2.5-14b` silently
    # exited with the quick-start banner instead of entering the main loop.
    _has_llm_cli = bool(str(getattr(args, "language_model", "") or "").strip())
    _has_action = _has_sim or _has_mode_override or _has_robot or _has_llm_cli

    if not (_has_action or _is_leader or _has_llm):
        return _bare_maxim_menu()

    # Leader/LLM-enabled without explicit action → server mode.
    # Start the LLM server for peer inference, then block idle.
    # The LeaderProxy (started earlier) handles peer commands.
    # This avoids entering the agentic/exploration loop, which would
    # burn GPU on agent inference cycles with no user interaction.
    if not _has_action and (_is_leader or _has_llm):
        args.mode = "server"

    build_home(args.home_dir)
    mode = str(getattr(args, "mode", "active")).strip().lower()
    while True:
        run_id = time.strftime("%Y-%m-%d_%H%M%S")
        log_path = os.path.join(args.home_dir, "logs", f"reachy_log_{run_id}.log")

        configure_logging(args.verbosity, log_file=log_path, force=True)
        logger = logging.getLogger("maxim")

        maxim = None
        try:
            epochs_value = _normalize_epoch_value(getattr(args, "epochs", 0))
            epochs_label = "unlimited" if epochs_value <= 0 else str(epochs_value)
            logger.info(
                "Starting Maxim (robot_name=%s, home_dir=%s, timeout=%.1fs, epochs=%s, mode=%s, log=%s)",
                args.robot_name,
                args.home_dir,
                float(args.timeout),
                epochs_label,
                mode,
                log_path,
            )

            # Check and log GPU status
            _check_gpu_status(logger)

            if mode == "agentic":
                if not _gpu_available():
                    _configure_cpu_fallback_model(logger, args.home_dir)

                from maxim.agents import MaximAgent
                from maxim.agents.autonomy import (
                    AutonomyController,
                    AutonomyLevel,
                    SafetyConstraints,
                    SupervisionPolicy,
                )
                from maxim.agents.llm_worker import LLMWorker
                from maxim.environment import ReachyEnv
                from maxim.runtime import (
                    build_decision_engine,
                    build_evaluators,
                    build_memory,
                    build_state,
                    build_tool_registry,
                )
                from maxim.runtime.agent_loop import run_agentic_loop
                from maxim.utils.structured_logging import configure_agentic_verbosity

                # Agentic console output is ON by default (unless --no-agentic-console).
                # Verbosity follows the global --log-level (alias --verbosity).
                agentic_verbosity = int(getattr(args, "verbosity", 1))
                agentic_console = not bool(getattr(args, "no_agentic_console", False))
                configure_agentic_verbosity(
                    verbosity=agentic_verbosity,
                    console_output=agentic_console,
                )
                logger.info(
                    "Agentic verbosity: %d (console=%s)",
                    agentic_verbosity,
                    agentic_console,
                )

                # Determine memory persistence path
                memory_path = getattr(args, "memory_path", None)
                if memory_path is None:
                    memory_path = os.path.join(args.home_dir, "memory", "memories.json")

                # Get LLM profile
                llm_profile = str(getattr(args, "language_model", "") or "").strip()
                if not llm_profile:
                    llm_profile = "mistral-7b-instruct-v0.2"

                # Create the agentic agent
                agentic_agent = MaximAgent(
                    llm_profile=llm_profile,
                    memory_persistence_path=memory_path,
                    data_folder=args.home_dir,
                    enable_embeddings=bool(getattr(args, "enable_embeddings", False)),
                    reset_on_startup=bool(getattr(args, "reset", False)),
                )

                # Set up ResponseOutput for LLM responses
                from pathlib import Path
                from maxim.utils.response_output import ResponseOutput

                sandbox_path = Path(args.home_dir) / "sandbox"
                tts_engine = None
                speaker_fn = None

                # Set up TTS if enabled
                if getattr(args, "tts", False):
                    try:
                        from maxim.models.audio.tts import TTSEngine

                        tts_model = str(getattr(args, "tts_model", "en_US-lessac-medium"))
                        tts_engine = TTSEngine(model_name=tts_model)
                        if tts_engine.is_available:
                            logger.info("TTS enabled with model: %s", tts_model)
                        else:
                            logger.warning("TTS model not found, TTS will be disabled")
                            tts_engine = None
                    except Exception as e:
                        logger.warning("Failed to initialize TTS: %s", e)
                        tts_engine = None

                response_output = ResponseOutput(
                    sandbox_path=sandbox_path,
                    logger=logger,
                    tts_engine=tts_engine,
                    speaker_fn=speaker_fn,
                )

                # Check if internet access is enabled (default: True unless --no-internet)
                internet_enabled = not bool(getattr(args, "no_internet", False))

                # Create internet policy getter for tool registry
                def get_internet_policy():
                    from maxim.utils.internet_access import InternetAccessPolicy

                    return InternetAccessPolicy(enabled=internet_enabled)

                # Only pass policy getter if internet is enabled
                internet_policy_getter = get_internet_policy if internet_enabled else None

                # Build comms stack if enabled (--comms flag or MAXIM_COMMS_ENABLED env)
                comms_enabled = bool(getattr(args, "comms", False)) or os.environ.get(
                    "MAXIM_COMMS_ENABLED", ""
                ).lower() in ("1", "true", "yes")
                gateway = None
                if comms_enabled:
                    from maxim.runtime.bootstrap import build_comms_stack

                    gateway, _conv_manager = build_comms_stack(
                        bus=agentic_agent._bus,
                    )

                # ── Tool registry (CLI-specific: depends on response_output,
                # internet_policy, gateway, prompt_handler) ──
                _is_sim_mode = getattr(args, "sim", None) is not None

                _operational_mode = "active" if _is_sim_mode else "passive"
                try:
                    from maxim.interactive.prompts import create_handler

                    _prompt_handler = create_handler("auto")
                except Exception as _ph_exc:
                    logger.warning("PromptHandler unavailable: %s", _ph_exc)
                    _prompt_handler = None

                registry = build_tool_registry(
                    response_output=response_output,
                    internet_policy_getter=internet_policy_getter,
                    gateway=gateway,
                    operational_mode=_operational_mode,
                    prompt_handler=_prompt_handler,
                )

                # ── F2: Full agent construction via AgentFactory ──
                # Replaces ~100 lines of hand-rolled bio-stack + executor
                # + FearGatedExecutor construction. The factory composes
                # build_bio_stack + build_executor + fear gating into a
                # single create_full_agent call (Z1 per-instance Executor).
                from maxim.runtime.agent_factory import AgentConfig, AgentFactory

                _mem_dir = str(Path(memory_path).parent) if memory_path else None
                _embodiment_ref = getattr(args, "embodiment", None) if not _is_sim_mode else None
                _component_registry = None
                if _embodiment_ref:
                    from maxim.embodiment.component_registry import ComponentRegistry

                    _component_registry = ComponentRegistry()

                _agent_config = AgentConfig(
                    agent_id="cli_agent",
                    role="pc",
                    persistence_dir=_mem_dir,
                    with_bio_stack=True,
                    with_executor=True,
                    with_pain_bridge=not _is_sim_mode,
                    with_fear_gate=not _is_sim_mode,
                    embodiment_ref=_embodiment_ref,
                )
                _factory = AgentFactory(
                    component_registry=_component_registry,
                    base_data_dir=_mem_dir,
                )

                from maxim.exceptions import ComponentNotFoundError

                try:
                    _cli_instance = _factory.create_full_agent(
                        _agent_config,
                        tool_registry=registry,
                    )
                except ComponentNotFoundError as _cnf:
                    print(f"error: --embodiment: {_cnf}", file=sys.stderr)
                    sys.exit(2)
                except ValueError as _ve:
                    print(f"error: --embodiment: {_ve}", file=sys.stderr)
                    sys.exit(2)

                # Extract components for the agent loop
                _cli_bio = _cli_instance.bio_stack
                _cli_memory_hub = _cli_instance.memory_hub
                _cli_hippocampus = _cli_instance.hippocampus
                _cli_nac = _cli_instance.nac
                _cli_pain_bus = _cli_instance.pain_bus
                _cli_embodiment = _cli_instance.embodiment
                executor = _cli_instance.executor
                # PFC deliberation: ThoughtGate + BioEnrichmentPipeline from BioStack
                _cli_thought_gate = getattr(_cli_bio, "thought_gate", None) if _cli_bio is not None else None
                _cli_bio_enrichment = (
                    getattr(_cli_bio, "bio_enrichment_pipeline", None) if _cli_bio is not None else None
                )

                if _cli_memory_hub is not None:
                    agentic_agent.wire_memory_hub(_cli_memory_hub)
                    logger.info("AgentFactory: bio-stack + executor wired (F2 migration)")

                if _cli_embodiment is not None:
                    logger.info(
                        "Embodiment loaded: %s (root entity=%r)",
                        _embodiment_ref,
                        _cli_embodiment.root.name,
                    )

                decision_engine = build_decision_engine()
                env = ReachyEnv(data_dir=args.home_dir)
                # In simulation mode, don't limit state steps (grace period handles termination)
                _state_max = 0 if getattr(args, "sim", None) is not None else epochs_value
                state = build_state(max_steps=_state_max)
                memory = build_memory()
                evaluators = build_evaluators()

                # Set up autonomy controller
                autonomy_level_str = str(getattr(args, "autonomy", "planning")).lower()
                initial_level = AutonomyLevel(autonomy_level_str)

                # Build allowed tools set based on internet policy (uses internet_enabled from above)
                allowed_tools = {
                    "read_file",
                    "write_file",  # Requires confirmation (see requires_confirmation below)
                    "focus_interests",
                    "track_target",
                    "maxim_command",
                    "mode_switch",
                    "speak",
                    "respond",
                    "list_directory",  # Allow directory listing
                    "glob",  # Pattern-based file search
                    "bash",  # Shell command execution (requires MAXIM_ALLOW_BASH=1)
                }

                # Add internet tools if enabled
                if internet_enabled:
                    allowed_tools.add("internet_search")
                    allowed_tools.add("http_fetch")

                # Add comms tools if gateway is available
                if gateway is not None:
                    allowed_tools.add("send_message")
                    allowed_tools.add("call_user")

                # Configure supervision policy with sensible defaults
                supervision_policy = SupervisionPolicy(
                    allowed_tools=allowed_tools,
                    forbidden_tools={"execute_file", "delete_file"},
                    min_confidence_autonomous=0.7,
                    requires_confirmation={"write_file", "bash"},  # bash requires confirmation for safety
                )

                autonomy_controller = AutonomyController(
                    initial_level=initial_level,
                    safety_constraints=SafetyConstraints(),
                    supervision_policy=supervision_policy,
                )

                # Set timed autonomy if specified
                autonomy_duration = getattr(args, "autonomy_duration", None)
                if autonomy_duration and initial_level == AutonomyLevel.AUTONOMOUS:
                    autonomy_controller.set_level(
                        AutonomyLevel.AUTONOMOUS,
                        f"CLI: timed autonomy for {autonomy_duration}s",
                        duration_seconds=autonomy_duration,
                    )

                # Set up LLM worker — LLM lives inside ExecAgent, not MaximAgent
                llm_worker = None
                llm_router = None
                # Trigger lazy LLM + router init via ExecAgent
                if hasattr(agentic_agent, "exec_agent"):
                    agentic_agent.exec_agent._ensure_llm()
                    llm_router = agentic_agent.exec_agent._ensure_router()
                if llm_router is not None:
                    # Check model file exists before warming up
                    model_path = str(getattr(llm_router.cfg, "model_path", "")).strip()
                    if model_path and not os.path.exists(model_path):
                        logger.info("Model not found at %s — attempting download...", model_path)
                        try:
                            from maxim.models.download import download_llm, LLM_MODELS

                            profile = str(
                                getattr(llm_router.cfg, "profile", "") or getattr(llm_router.cfg, "model_base", "")
                            ).strip()
                            if profile and profile in LLM_MODELS:
                                print(f"  Downloading LLM model: {profile}...")
                                if download_llm(profile):
                                    print(f"  Download complete: {profile}")
                                else:
                                    print("  Download failed. Run: maxim --list-models to see available models.")
                            else:
                                print("  Model not found. Run: maxim --list-models to see available models.")
                        except Exception as e:
                            print(f"  Auto-download failed: {e}")
                            print("  Run: maxim --list-models to see available models.")

                    if hasattr(llm_router, "warmup"):
                        llm_router.warmup()
                    # In simulation mode, wait for LLM to be fully loaded before starting
                    _is_sim = getattr(args, "sim", None) is not None
                    if _is_sim and hasattr(llm_router, "wait_ready"):
                        logger.info("Waiting for LLM to load (simulation mode)...")
                        if llm_router.wait_ready(timeout=120.0):
                            logger.info("LLM ready")
                        else:
                            logger.warning("LLM failed to load — simulation will use fallback responses")
                            print("  WARNING: LLM failed to load. Simulation will not produce real responses.")
                            print("  Ensure model is downloaded. Run: maxim --list-models")
                    logger.info("LLM router initialized: %s", llm_router)
                    llm_worker = LLMWorker(
                        llm_router,
                        n_ctx=getattr(llm_router, "n_ctx", 4096),
                        token_counter=(
                            llm_router.get_token_counter() if hasattr(llm_router, "get_token_counter") else None
                        ),
                    )
                    llm_worker.start()
                    # B3: Enable Acting Coach when embodiment tools are available
                    if _embodiment_ref is not None:
                        from maxim.prompts.acting_coach import ActingCoachConfig

                        llm_worker.acting_coach = ActingCoachConfig()

                        # E2: Inject entity context (sensors, affordances, failure
                        # triggers) into the AUT prompt so the agent knows what
                        # physical capabilities it has.
                        if _component_registry is not None:
                            try:
                                _entity_raw = _component_registry.get(_embodiment_ref)
                                llm_worker.entity_spec = _entity_raw.get("entity", _entity_raw)
                            except Exception as _e:
                                logger.debug("Entity context injection failed: %s", _e)

                # Store internet access in state (uses internet_enabled from above)
                state.data["internet_access"] = internet_enabled
                state.data["autonomy_level"] = initial_level.value

                # Wire communication gateway if available
                if gateway is not None:
                    agentic_agent.wire_communication(gateway=gateway)
                    logger.info("Communication gateway wired")

                logger.info(
                    "Starting MaximAgent (memory_path=%s, embeddings=%s, reset=%s, autonomy=%s, internet=%s, comms=%s)",
                    memory_path,
                    bool(getattr(args, "enable_embeddings", False)),
                    bool(getattr(args, "reset", False)),
                    initial_level.value,
                    internet_enabled,
                    gateway is not None,
                )

                # Check for interactive simulation REPL — redirect to the
                # full generative sim with interactive mode ON.  This gives
                # the user raw terminal input, MaximDisplay, slash commands,
                # and /new goal continuations — the same stack as --sim "goal".
                if (
                    getattr(args, "sim", None) is not None
                    and str(getattr(args, "sim", "")).strip().lower() == "interactive"
                ):
                    from maxim.simulation.orchestrator import start_simulation_mode

                    _sim_debug = bool(getattr(args, "debug", False) or getattr(args, "sim_debug", False))
                    try:
                        result = start_simulation_mode(
                            goal="open interactive session — respond to user input naturally",
                            persona="campaign",
                            max_turns=200,
                            response_timeout=120.0,
                            debug=_sim_debug,
                            entity_ref=_sim_entity_ref,
                        )
                        sys.exit(0 if result.finish_reason != "error" else 1)
                    finally:
                        if llm_worker:
                            llm_worker.stop()

                # Check for simulation mode — wire percept_source and action_sink
                sim_source = getattr(args, "_sim_source", None)
                sim_sink = getattr(args, "_sim_sink", None)

                # In sim mode, reuse the bio-stack's PainBus for headless
                # pain routing.  Pre-F2 this built a SECOND PainBus on
                # the same hippocampus/nac — review fix (Arch #1) caught
                # the double-subscription.  The bio-stack already has
                # hippocampus + nac subscribers wired.
                if sim_source is not None:
                    _sim_pain_bus = _cli_pain_bus
                    logger.info(
                        "Sim reusing bio-stack PainBus (hippocampus=%s, nac=%s)",
                        _cli_hippocampus is not None,
                        _cli_nac is not None,
                    )

                    args._sim_pain_bus = _sim_pain_bus
                    args._sim_hippo = _cli_hippocampus

                try:
                    run_agentic_loop(
                        agentic_agent,
                        env,
                        state,
                        memory,
                        decision_engine,
                        executor,
                        autonomy_controller=autonomy_controller,
                        llm_worker=llm_worker,
                        hippocampus=_cli_hippocampus,
                        memory_hub=_cli_memory_hub,
                        evaluators=evaluators,
                        max_steps=epochs_value,
                        run_id=run_id,
                        percept_source=sim_source,
                        action_sink=sim_sink,
                        pain_bus=getattr(args, "_sim_pain_bus", None) or _cli_pain_bus,
                        bio_enrichment_pipeline=_cli_bio_enrichment,
                        thought_gate=_cli_thought_gate,
                    )
                finally:
                    if llm_worker:
                        llm_worker.stop()

                # If simulation mode, validate expectations and report
                if sim_source is not None and sim_sink is not None:
                    import json as _json
                    from maxim.simulation.validation import validate_expectations

                    scenario_file = getattr(args, "_sim_scenario_file", None)
                    hippo = getattr(args, "_sim_hippo", None)

                    results = validate_expectations(
                        expectations=sim_source.expectations,
                        sink=sim_sink,
                        hippocampus=hippo,
                        emitted_tags=sim_source.emitted_tags,
                    )

                    passed = all(r.passed for r in results)
                    scenario_name = sim_source.definition.name

                    print(f"\n{'PASS' if passed else 'FAIL'}: {scenario_name}")
                    for r in results:
                        status = "PASS" if r.passed else "FAIL"
                        desc = r.expectation.description or r.expectation.type
                        print(f"  [{status}] {desc}")
                        if not r.passed and r.detail:
                            print(f"         {r.detail}")
                    print(f"\nActions recorded: {len(sim_sink.actions)}")
                    for a in sim_sink.actions[:20]:
                        tag = "[BLOCKED]" if a.blocked else ("[OK]" if a.result_success else "[FAIL]")
                        print(f"  {tag} {a.tool_name}")

                    report_path = getattr(args, "sim_report", None)
                    if report_path:
                        report = {
                            "scenario": scenario_name,
                            "passed": passed,
                            "action_count": len(sim_sink.actions),
                            "expectations_met": [
                                r.expectation.description or r.expectation.type for r in results if r.passed
                            ],
                            "expectations_failed": [
                                f"{r.expectation.description or r.expectation.type}: {r.detail}"
                                for r in results
                                if not r.passed
                            ],
                        }
                        with open(report_path, "w") as f:
                            _json.dump(report, f, indent=2)
                        print(f"Report written to {report_path}")

                    # Save sim log and disable logging
                    saved_log = disable_sim_logging()
                    if saved_log:
                        print(f"  Simulation log saved: {saved_log}")

                    # Clean up simulation sandbox (but preserve the log)
                    sim_tmpdir = getattr(args, "_sim_tmpdir", None)
                    if sim_tmpdir is not None:
                        import shutil

                        original_cwd = getattr(args, "_sim_original_cwd", "/")
                        os.chdir(original_cwd)
                        try:
                            shutil.rmtree(str(sim_tmpdir))
                            print(f"  Sandbox cleaned up: {sim_tmpdir}")
                        except Exception as e:
                            print(f"  Warning: failed to clean sandbox: {e}")

                    return 0 if passed else 1

                return 0

            # ── Server mode: LLM server + proxy, no agent loop ──────────
            if mode == "server":
                print("Maxim — server mode")
                try:
                    from maxim.runtime.lane_backends import build_primary_router

                    _srv_router, _srv_mgr = build_primary_router(logger=logger)
                    if _srv_router is not None:
                        _srv_router.warmup()
                        _srv_router.wait_ready(timeout=180.0)
                        logger.info("LLM server ready — accepting peer inference")
                        print("  LLM server ready — accepting peer inference")
                    else:
                        logger.warning("LLM server did not start")
                        print("  WARNING: LLM server failed to start (check model config)")
                except Exception as _srv_err:
                    logger.warning("LLM server init failed: %s", _srv_err)
                    print(f"  WARNING: LLM server init failed: {_srv_err}")

                print("  Proxy running — accepting peer commands")
                print("  Use --sim or --mode to start a session, or Ctrl+C to stop.\n")
                try:
                    import threading

                    threading.Event().wait()
                except KeyboardInterrupt:
                    print("\n  Shutting down.")
                finally:
                    try:
                        from maxim.runtime.lane_backends import stop_active_spawner

                        stop_active_spawner()
                    except Exception:
                        pass
                return 0

            from maxim.embodied_runtime.selfy import Maxim

            audio_enabled = bool(getattr(args, "audio", True))
            if mode == "sleep":
                audio_enabled = True

            maxim = Maxim(
                robot_name=args.robot_name,
                home_dir=args.home_dir,
                timeout=args.timeout,
                epochs=epochs_value,
                verbosity=args.verbosity,
                mode=mode,
                audio=audio_enabled,
                audio_len=float(getattr(args, "audio_len", 5.0) or 5.0),
                interactive=bool(getattr(args, "interactive", True)),
            )

            if mode == "sleep":
                logger.info("Maxim sleeping (audio-only).")
                # Mark as already sleeping so no movement occurs on startup
                maxim.sleeping = True
                maxim.sleep(home_dir=args.home_dir, run_id=run_id)
            else:
                logger.info("✅ Maxim lives!")
                maxim.live(home_dir=args.home_dir, run_id=run_id)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user (Ctrl+C).")
            break
        except Exception as e:
            log_exception(
                logger,
                e,
                verbosity=getattr(args, "verbosity", 0),
                message="❌ Maxim stopped",
            )
            break
        finally:
            if maxim is not None:
                try:
                    maxim.shutdown()
                except Exception:
                    pass
            # Belt-and-suspenders: kill the LLM server subprocess even if
            # shutdown() didn't reach it (e.g. double Ctrl+C, thread hang).
            # This prevents ghost llama-cpp-server processes holding VRAM.
            try:
                from maxim.runtime.lane_backends import stop_active_spawner

                stop_active_spawner()
            except Exception:
                pass

        requested = getattr(maxim, "requested_mode", None) if maxim is not None else None
        if not requested:
            break
        requested = str(requested).strip().lower()
        if requested == "shutdown":
            logger.info("Shutdown requested.")
            break
        if requested in ("sleep", "live", "agentic", "passive", "active", "singularity"):
            logger.info("Switching mode: %s -> %s", mode, requested)
            delay_s = 0.0
            try:
                delay_s = float(os.getenv("MAXIM_MODE_SWITCH_DELAY_S", "1.5") or 0.0)
            except Exception:
                delay_s = 1.5
            if delay_s > 0:
                logger.info("Waiting %.1fs before reconnect...", delay_s)
                time.sleep(delay_s)
            try:
                _reexec_with_mode(args, mode=requested)
            except Exception as e:
                logger.warning("Failed to restart Maxim for mode switch (%s); continuing in-process.", e)
                mode = requested
                continue
        logger.warning("Ignoring unknown requested_mode=%r", requested)
        break

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint with top-level :class:`BackendError` surfacing.

    The router's typed exception hierarchy (``BackendOverloaded``,
    ``BackendAuthFailed``, ``BackendModelMissing``, ``BackendDown``, ...)
    each carries a class-level ``fix_hint`` describing how to remediate.
    Without this wrapper, BackendErrors that escape an inner ``try``
    surface as bare Python tracebacks — the actionable fix string was
    invisible to the user.

    On ``BackendError``: render via ``user_warn`` (display panel + JSONL
    + stderr) and exit 2.  ``KeyboardInterrupt`` exits cleanly with 130
    (Ctrl-C convention) so the live display can flush.  Any other
    exception re-raises — we don't want to swallow bugs.
    """
    try:
        return _main_impl(argv)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        try:
            from maxim.models.language.types import BackendError
        except Exception:
            raise

        if isinstance(e, BackendError):
            from maxim.utils.logging import user_warn

            # `BackendError.fix_hint` is class-level static text per the
            # CLAUDE.md invariant — every subclass declares one.  Direct
            # attribute access (not getattr default) so a forgotten
            # subclass-level fix_hint surfaces as a loud AttributeError
            # instead of silently using a generic fallback string.
            user_warn(
                f"{type(e).__name__}: {e}",
                fix=e.fix_hint,
                source="backend",
                event="backend_error_uncaught",
                data={
                    "exc_type": type(e).__name__,
                    "status": getattr(e, "status", None),
                },
            )
            return 2
        raise


life = main


if __name__ == "__main__":
    raise SystemExit(main())
