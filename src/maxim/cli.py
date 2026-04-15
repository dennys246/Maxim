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


def main(argv: Sequence[str] | None = None) -> int:
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
                print("[leader-boot] WARNING: could not read API key — proxy will run without auth")
            _proxy = start_leader_proxy(api_key=_api_key, bind_host=detected_role.bind_host)
            if _proxy is None:
                print("[leader-boot] WARNING: LeaderProxy failed to start (port in use?)")
    except Exception as _e:
        import traceback as _tb

        print(f"[leader-boot] WARNING: early proxy boot failed: {_e}")
        _tb.print_exc()

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

    # ── Cross-flag validation ───────────────────────────────────────────
    if getattr(args, "sim_report", None) and getattr(args, "sim", None) is None:
        print("Error: --sim-report requires --sim.", file=sys.stderr)
        return 1

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
            persona=getattr(args, "sim_persona", "campaign") or "campaign",
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
            print('  Usage: maxim --sim agent --goal "test safety" --persona adversarial')
            print('         maxim --sim research --goal "hippocampal recall" --campaign <yaml>')
            sys.exit(1)
        if getattr(args, "sim_persona", "adversarial") != "adversarial":
            print("Error: --persona / --sim-persona requires --sim agent (simulation mode).")
            print('  Usage: maxim --sim agent --goal "test safety" --persona adversarial')
            sys.exit(1)
        if getattr(args, "resume_sim", None) is not None:
            print("Error: --resume-sim requires --sim agent (simulation mode).")
            print('  Usage: maxim --sim agent --goal "continue" --resume-sim SESSION_ID')
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
                _set_interactive_mode_early("on" if (_wants_dm_early or _is_dm_yaml_early) else "off")
            _show_channels_early = getattr(args, "show_channels", None)
            if _show_channels_early:
                _set_show_channels_early(_show_channels_early)
        except Exception as _e:
            logging.getLogger("maxim").warning("Failed to apply sim display config: %s", _e)

        # ── Generative campaign mode (new default for goal strings) ──
        if _is_goal_string and not _is_legacy_agent:
            goal = _explicit_goal or _sim_val
            if _wants_research:
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
                # Pure generative campaign — use the orchestrator with generative runner
                # For now, delegate to start_simulation_mode with the goal
                # The generative runner will be wired in as the default persona
                from maxim.simulation.orchestrator import start_simulation_mode

                persona = getattr(args, "sim_persona", "campaign")
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
                    max_turns=int(getattr(args, "sim_max_turns", 50) or 50),
                )
                sys.exit(0 if result.finish_reason != "error" else 1)

        # ── Legacy: agent mode (deprecated alias) ──
        _sim_agent = _is_legacy_agent
        if _sim_agent:
            from maxim.simulation.orchestrator import start_simulation_mode

            goal = getattr(args, "sim_goal", None) or "test the agent's capabilities"
            persona = getattr(args, "sim_persona", "adversarial")
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
                max_turns=int(getattr(args, "sim_max_turns", 50) or 50),
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
                persona=getattr(args, "sim_persona", "campaign") or "campaign",
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
        print("Maxim — bio-inspired cognitive architecture\n")
        print("Quick start:")
        print('  maxim --sim "test the agent\'s memory"   Run a generative simulation')
        print("  maxim --sim scenarios/my_test.yaml       Run a YAML scenario")
        print("  maxim doctor                             Check your environment")
        print("  maxim --list-models                      See available LLM models")
        print("  maxim --help                             Full option reference\n")
        print("Python API:")
        print("  import maxim")
        print("  maxim.diagnose()                         Check environment from Python")
        print("  maxim.list_models()                      List available models\n")
        return 0

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
                    build_executor,
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

                # Build MemoryHub with Hippocampus for episodic memory.
                # All bio-system handles initialized to None up-front so
                # `build_executor` below sees defined names even if
                # construction fails inside the try block (pre-existing
                # latent UnboundLocalError surfaced in pre-merge review).
                _cli_memory_hub = None
                _cli_hippocampus = None
                _cli_nac = None
                _cli_scn = None
                _cli_ec = None
                try:
                    from maxim.integration.memory_hub import MemoryHub
                    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
                    from maxim.decisions.nac import NAc
                    from maxim.similarity.ec import EntorhinalCortex
                    from maxim.time.scn import SCN

                    _cli_hippocampus = Hippocampus(
                        config=HippocampusConfig(
                            persistence_path=memory_path,
                        )
                    )
                    _cli_nac = NAc()
                    _cli_scn = SCN()
                    _cli_ec = EntorhinalCortex()
                    _cli_memory_hub = MemoryHub(
                        hippocampus=_cli_hippocampus,
                        scn=_cli_scn,
                        nac=_cli_nac,
                        ec=_cli_ec,
                    )
                    agentic_agent.wire_memory_hub(_cli_memory_hub)
                    logger.info("MemoryHub + Hippocampus + NAc + SCN + EC wired to MaximAgent")
                except Exception as e:
                    logger.warning("Failed to create MemoryHub: %s", e)

                # Use active mode in simulation so agent can read/write in sandbox
                _operational_mode = "active" if getattr(args, "sim", None) is not None else "passive"
                # PromptHandler for RequestInteractionTool — console prompts
                # when --interactive is on, respects the global interactive
                # mode gate inside the tool itself otherwise.
                try:
                    from maxim.interactive.prompts import create_handler

                    _prompt_handler = create_handler("auto")
                except Exception as _ph_exc:
                    logger.debug("PromptHandler unavailable: %s", _ph_exc)
                    _prompt_handler = None

                registry = build_tool_registry(
                    response_output=response_output,
                    internet_policy_getter=internet_policy_getter,
                    gateway=gateway,
                    operational_mode=_operational_mode,
                    prompt_handler=_prompt_handler,
                )

                # ── PainBus + Executor construction ──
                # `build_executor` requires an explicit `pain_bus=`
                # decision. Non-sim CLI agent runs construct a real
                # PainBus + subscribe pain-memory capture; sim modes
                # opt out (their own PainBus wiring lives downstream
                # in `run_agentic_loop` / `run_interactive_sim` for
                # now — the sim-path migration is tracked in
                # `docs/plans/sem_execution_hook.md` Stage 2c, which
                # collapses to a one-line build_executor call once
                # this branch lands).
                _cli_embodiment = None
                _cli_pain_bus = None
                _is_sim_mode = getattr(args, "sim", None) is not None
                if _is_sim_mode and getattr(args, "embodiment", None):
                    # Hard error — "warn and ignore" was cross-confirmed
                    # as the wrong UX in the Stage 2 pre-merge review.
                    print(
                        "error: --embodiment is not yet supported with --sim.\n"
                        "  For DM-campaign YAMLs, set `component: <ref>` in the\n"
                        "  encounter spec instead. The standalone --embodiment\n"
                        "  flag for sim modes is tracked in\n"
                        "  docs/plans/sem_execution_hook.md Stage 2c.",
                        file=sys.stderr,
                    )
                    sys.exit(2)

                _embodiment_ref = None
                _component_registry = None
                if not _is_sim_mode:
                    from maxim.proprioception.pain_bus import (
                        PainBus as CliPainBus,
                        create_pain_memory_subscriber,
                    )

                    _cli_pain_bus = CliPainBus()
                    if _cli_hippocampus is not None:
                        _cli_pain_bus.subscribe(create_pain_memory_subscriber(_cli_hippocampus))
                        logger.info("CLI PainBus wired to hippocampus for pain memory capture")

                    _embodiment_ref = getattr(args, "embodiment", None)
                    if _embodiment_ref:
                        from maxim.embodiment.component_registry import ComponentRegistry

                        _component_registry = ComponentRegistry()

                # Narrow exception handling: user-facing errors
                # (ComponentNotFoundError, ValueError from build_executor's
                # precondition checks) propagate as sys.exit(2) with the
                # error message intact. Pre-merge review on the Stage 2
                # helper cross-confirmed that a broad `except Exception`
                # here silently swallowed typos — exactly the silent-no-op
                # mode bootstrap unification exists to eliminate.
                from maxim.exceptions import ComponentNotFoundError

                try:
                    executor = build_executor(
                        registry,
                        pain_bus=_cli_pain_bus,
                        nac=_cli_nac if _cli_pain_bus is not None else None,
                        hippocampus=_cli_hippocampus if _cli_pain_bus is not None else None,
                        scn=_cli_scn if _cli_pain_bus is not None else None,
                        entity_ref=_embodiment_ref,
                        component_registry=_component_registry,
                    )
                except ComponentNotFoundError as _cnf:
                    print(f"error: --embodiment: {_cnf}", file=sys.stderr)
                    sys.exit(2)
                except ValueError as _ve:
                    print(f"error: --embodiment: {_ve}", file=sys.stderr)
                    sys.exit(2)

                _cli_embodiment = getattr(executor, "_embodiment", None)
                if _cli_embodiment is not None:
                    logger.info(
                        "Embodiment loaded: %s (root entity=%r)",
                        _embodiment_ref,
                        _cli_embodiment.root.name,
                    )

                # Wrap executor with FearAgent safety gating (independent of DefaultNetwork)
                from maxim.agents.fear_agent import FearAgent
                from maxim.runtime.fear_gate import FearGatedExecutor

                fear_agent = FearAgent(
                    llm=agentic_agent._llm if hasattr(agentic_agent, "_llm") else None,
                )
                executor = FearGatedExecutor(executor, fear_agent)
                logger.info("FearGatedExecutor active — all tool calls reviewed by FearAgent")

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
                                    print("  Download failed. Run: ./scripts/download_models.sh --llm --enable")
                            else:
                                print("  Model not found. Run: ./scripts/download_models.sh --llm --enable")
                        except Exception as e:
                            print(f"  Auto-download failed: {e}")
                            print("  Run: ./scripts/download_models.sh --llm --enable")

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
                            print("  Ensure model is downloaded: ./scripts/download_models.sh --llm --enable")
                    logger.info("LLM router initialized: %s", llm_router)
                    llm_worker = LLMWorker(
                        llm_router,
                        n_ctx=getattr(llm_router, "n_ctx", 4096),
                        token_counter=(
                            llm_router.get_token_counter() if hasattr(llm_router, "get_token_counter") else None
                        ),
                    )
                    llm_worker.start()

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

                # Check for interactive simulation REPL
                if (
                    getattr(args, "sim", None) is not None
                    and str(getattr(args, "sim", "")).strip().lower() == "interactive"
                ):
                    from maxim.simulation.interactive import run_interactive_sim
                    from maxim.proprioception.pain_bus import (
                        PainBus as SimPainBus,
                        create_pain_memory_subscriber,
                    )

                    _sim_pain_bus = SimPainBus()
                    if _cli_hippocampus is not None:
                        _sim_pain_bus.subscribe(create_pain_memory_subscriber(_cli_hippocampus))

                    try:
                        run_interactive_sim(
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
                            pain_bus=_sim_pain_bus,
                            llm_profile=llm_profile,
                            sim_workspace=Path(getattr(args, "home_dir", "data")) / "sim_sandbox",
                            debug=bool(getattr(args, "debug", False) or getattr(args, "sim_debug", False)),
                        )
                    finally:
                        if llm_worker:
                            llm_worker.stop()
                    return 0

                # Check for simulation mode — wire percept_source and action_sink
                sim_source = getattr(args, "_sim_source", None)
                sim_sink = getattr(args, "_sim_sink", None)

                # In sim mode, create a PainBus for headless pain routing
                # (DefaultNetwork may not exist, so we need our own)
                if sim_source is not None:
                    from maxim.proprioception.pain_bus import (
                        PainBus as SimPainBus,
                        create_pain_memory_subscriber,
                    )

                    _sim_pain_bus = SimPainBus()
                    if _cli_hippocampus is not None:
                        _sim_pain_bus.subscribe(create_pain_memory_subscriber(_cli_hippocampus))
                        logger.info("Sim PainBus wired to hippocampus for pain memory capture")

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


life = main


if __name__ == "__main__":
    raise SystemExit(main())
