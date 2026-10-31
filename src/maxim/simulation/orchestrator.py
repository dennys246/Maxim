"""Simulation orchestrator — boots and manages two-agent simulation mode.

Manages three threads:
- Thread 1 (AUT): The agent-under-test runs its full agentic loop
- Thread 2 (Orchestrator): A second agent drives the AUT via simulation tools
- Main thread (stdin): Routes user commands to the orchestrator

The orchestrator and AUT share a single LLMRouter instance to avoid
double model loading. The bridge (ConversationalSource + RecordingSink)
provides the thread-safe communication channel.
"""

from __future__ import annotations

import functools
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from maxim.simulation.tty_stdin import read_post_sim_goal_raw, stdin_reader_raw
from maxim.exceptions import StopStillPending
from maxim.utils.cancellation import current_stop


logger = logging.getLogger(__name__)

# Extracted to sim_types.py
from maxim.simulation.report import capture_start_provenance  # noqa: E402
from maxim.simulation.sim_teardown import SimTail, build_simulation_result, end_run  # noqa: E402
from maxim.simulation.sim_types import (  # noqa: E402
    SimulationResult,
    load_resume_context_at as _load_resume_context_at,
    build_resume_prompt as _build_resume_prompt,
    build_kickoff_prompt as _build_kickoff_prompt,
    narrator_tools_block as _narrator_tools_block,
)


def _setup_sim_sandbox(
    *,
    backend: str = "auto",
    image: str = "python:3.12-slim",
    network: str = "none",
    populate: bool = True,
    announce: bool = False,
    pain_bus: Any | None = None,
) -> tuple[Any, str | None, Any]:
    """Build the AUT pain bus + sandbox for a simulation run.

    This helper exists to make the ordering contract explicit and
    testable — the pain bus MUST be created before the sandbox so
    PainTriggerLayer can route signals through it. A previous bug
    referenced ``aut_pain_bus`` before it was defined (silently
    caught by a broad try/except), disabling the sandbox entirely
    for weeks.

    Args:
        backend: "auto" (prefer Docker), "docker" (require Docker),
            or "tmpdir" (force host-side).
        image: Docker image name for the Docker backend.
        network: Container network mode ("none" / "bridge" / "host").
        populate: Whether to populate honeypot environment files.
        announce: If True, print a visible one-liner to stderr
            showing the selected backend.
        pain_bus: Pre-built PainBus to route sandbox pain through
            (HANDLE seam a: a persistent agent's own bus, which already
            carries its hippocampus/NAc learners). When ``None`` (all
            existing callers), a learner-less bus is built here and
            ``build_bio_stack`` attaches learners later.

    Returns:
        (sim_sandbox, sandbox_root, aut_pain_bus). Any element may
        be ``None`` if creation failed (error logged at WARNING).
    """
    # Build the pain bus first — the sandbox's PainTriggerLayer needs it.
    # The bus is intentionally created without learners here; build_bio_stack
    # later attaches them via build_pain_bus(bus=aut_pain_bus, ...).
    # HANDLE seam (a): an injected persistent agent supplies its OWN bus so
    # sandbox pain lands on the bus its learners are already subscribed to —
    # building a second bus here would silently orphan those signals.
    aut_pain_bus: Any = pain_bus
    if aut_pain_bus is None:
        try:
            from maxim.proprioception.pain_bus import build_pain_bus

            aut_pain_bus = build_pain_bus(hippocampus=None, nac=None)
        except Exception as e:
            logger.debug("AUT PainBus creation deferred: %s", e)

    sim_sandbox = None
    sandbox_root: str | None = None
    try:
        from maxim.agents.autonomy import AutonomyLevel as _AL
        from maxim.simulation.sandbox import (
            ContainerPermissions,
            DockerSandbox,
            TmpdirSandbox,
            create_sandbox,
            permissions_for_autonomy,
        )

        # Simulation runs AUT at AUTONOMOUS (it's sandboxed), which
        # gets the largest resource envelope. Network is always an
        # explicit opt-in via the caller's `network` arg.
        base_perms = permissions_for_autonomy(_AL.AUTONOMOUS)
        sandbox_perms = ContainerPermissions(
            memory=base_perms.memory,
            cpus=base_perms.cpus,
            pids_limit=base_perms.pids_limit,
            workspace_readonly=base_perms.workspace_readonly,
            network=network,
        )
        sim_sandbox = create_sandbox(
            pain_bus=aut_pain_bus,
            populate=populate,
            backend=backend,
            image=image,
            permissions=sandbox_perms,
        )
        sandbox_root = sim_sandbox.workspace_root

        # Report the ACTUAL backend selected (auto may have fallen
        # through to tmpdir if Docker wasn't reachable).
        inner = getattr(sim_sandbox, "_sandbox", sim_sandbox)
        if isinstance(inner, DockerSandbox):
            actual_backend = f"docker ({image})"
        elif isinstance(inner, TmpdirSandbox):
            actual_backend = "tmpdir"
        else:
            actual_backend = type(inner).__name__

        if announce:
            if backend == "auto" and "tmpdir" in actual_backend:
                logger.warning(
                    "Sandbox: Docker unavailable — falling back to tmpdir (reduced isolation). "
                    "Install/start Docker Desktop for full container isolation."
                )
            else:
                logger.info("Sandbox: %s", actual_backend)
        if populate:
            logger.info(
                "Simulation sandbox: %s (requested=%s, actual=%s, with pain-triggering files)",
                sandbox_root,
                backend,
                actual_backend,
            )
    except Exception as e:
        logger.warning("Sandbox creation failed: %s", e)

    return sim_sandbox, sandbox_root, aut_pain_bus


def _prepare_sim_workspace(stamp: str | None = None) -> tuple[Path, Path, str]:
    """Create the per-run sim workspace under the data home.

    Returns ``(sim_workspace, sim_tmpdir, log_path)``:

    - ``sim_workspace`` — ``<data_home>/sim_sandbox``, shared across runs
      (the JSONL trace and substrate telemetry land here);
    - ``sim_tmpdir`` — a fresh ``sim_agent_<stamp>_*`` directory inside it,
      the AUT's ``FileSystemEnv`` root and per-run persistence dir;
    - ``log_path`` — ``<sim_workspace>/sim_agent_<stamp>.jsonl`` for
      ``enable_sim_logging``.

    Resolved through :func:`maxim.utils.paths.resolve_user_state` — the same
    precedent as ``interactive.py``'s sandbox dir — so ``MAXIM_DATA_HOME`` is
    honoured. This used to be ``Path("data") / "sim_sandbox"`` relative to
    the CWD, which a hosted sandbox with a read-only root and only the data
    home writable cannot serve. Extracted from ``start_simulation_mode`` so
    the mkdir is testable without booting a simulation.
    """
    from maxim.utils.paths import resolve_user_state

    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    sim_workspace = resolve_user_state("sim_sandbox")
    sim_workspace.mkdir(parents=True, exist_ok=True)
    sim_tmpdir = Path(tempfile.mkdtemp(prefix=f"sim_agent_{stamp}_", dir=str(sim_workspace)))
    log_path = str(sim_workspace / f"sim_agent_{stamp}.jsonl")
    return sim_workspace, sim_tmpdir, log_path


# ── HANDLE seam (a): persistent-agent campaign injection ────────────────────
# docs/plans/archive/console_handle_campaign_injection.md. A campaign run may ADOPT a
# live persistent AgentInstance instead of constructing a throwaway "sim_aut".
# Adoption is extracted into a helper (+ a registry lease) so the six
# silent-mis-route branches are directly testable without spinning up the
# full sim (same pattern as _setup_sim_sandbox above).


class _CampaignToolLease:
    """Reversible campaign-scoped mutations of a persistent agent's registry.

    The design rule (branch 4) is "register the sim/DM tools onto the
    PERSISTENT agent's registry for the campaign's duration, then restore on
    stop — do NOT swap the registry wholesale". The lease is the restore
    mechanism: snapshot the pre-campaign ``{name: tool_object}`` map, then on
    ``restore()`` drop campaign-added tools, re-register removed baseline
    tools, and re-register baseline tools a campaign registration silently
    REPLACED (``ToolRegistry.register`` overwrites same-name entries — a
    name-only diff cannot see that, and a replaced tool would pin dead sim
    state like a finished SimulationBridge into the persistent registry).

    ``restore()`` is idempotent (pure diff against the snapshot), so calling
    it on both the normal path and an exception-path safety net is safe.

    ``deactivate_tool`` is NOT usable here — it only applies to scene-scoped
    tools (``_scene_meta``); campaign tools register as core tools across
    many orchestrator sites, so a snapshot-diff is the complete mechanism.
    (Known limit: a baseline SCENE-scoped tool re-registered here comes back
    as core — no live case; handle registries carry no scene tools.)
    """

    def __init__(self, baseline: dict[str, Any]) -> None:
        self.baseline = baseline

    @classmethod
    def snapshot(cls, registry: Any) -> "_CampaignToolLease":
        baseline: dict[str, Any] = {}
        for name in list(registry.list_all()):
            try:
                baseline[name] = registry.get(name)
            except KeyError:
                pass
        return cls(baseline=baseline)

    def restore(self, registry: Any) -> tuple[list[str], list[str]]:
        """Return the registry to its pre-campaign state (idempotent).

        Returns (dropped_campaign_tools, restored_persistent_tools).
        """
        dropped = [name for name in list(registry.list_all()) if name not in self.baseline]
        for name in dropped:
            registry.deregister(name)
        restored = []
        for name, tool in self.baseline.items():
            try:
                current: Any = registry.get(name)
            except KeyError:
                current = None
            if current is not tool:
                registry.register(tool)
                restored.append(name)
        return dropped, restored


class _AdoptedAgent:
    """The derived bindings for an injected persistent agent (branch 1)."""

    def __init__(self, instance: Any) -> None:
        self.instance = instance
        self.agent_id: str = instance.agent_id
        self.registry = instance.tool_registry
        self.pain_bus = getattr(instance, "pain_bus", None)
        self.lease = _CampaignToolLease.snapshot(self.registry)


def _adopt_persistent_agent(persistent_agent: Any) -> _AdoptedAgent:
    """Validate + adopt a live persistent AgentInstance for a campaign run.

    Fail-fast (not silent mis-route): an under-built instance would
    otherwise route episodes into a half-wired stack — the exact failure
    mode this seam exists to kill is SILENT, so validation is loud.
    """
    agent_id = getattr(persistent_agent, "agent_id", "") or ""
    if not agent_id.strip():
        raise ValueError("persistent_agent must carry a non-empty agent_id (it routes episode attribution)")
    if agent_id == "sim_aut":
        raise ValueError(
            "persistent_agent.agent_id must not be 'sim_aut' — that id marks the throwaway sim AUT; "
            "a persistent agent needs its own identity (e.g. 'console_agent')"
        )
    # The whole point of injection is that the campaign LEARNS into the
    # persistent substrate — so the full bio surface is required, not just
    # the execution surface. A missing pain_bus in particular would make
    # _setup_sim_sandbox silently build a learner-less bus and orphan every
    # sandbox/consequence pain signal (the exact silent mis-route this loud
    # validation exists to kill).
    missing = [
        attr
        for attr in ("executor", "tool_registry", "memory_hub", "hippocampus", "nac", "pain_bus")
        if getattr(persistent_agent, attr, None) is None
    ]
    if missing:
        raise ValueError(
            f"persistent_agent is missing {missing} — build it via "
            "AgentFactory.create_full_agent(with_bio_stack=True, with_executor=True, tool_registry=...) "
            "before injecting it into a campaign"
        )
    hub_agent_id = getattr(persistent_agent.memory_hub, "agent_id", None)
    if hub_agent_id and hub_agent_id != agent_id:
        # Producer/consumer key alignment (CLAUDE.md per-agent stash lesson):
        # the loop derives _loop_agent_id from memory_hub.agent_id — a mismatch
        # here silently splits attribution across two keys.
        raise ValueError(
            f"persistent_agent.agent_id={agent_id!r} disagrees with its "
            f"memory_hub.agent_id={hub_agent_id!r} — attribution would silently split across two keys"
        )
    return _AdoptedAgent(persistent_agent)


def _arm_sandboxed_aut_subprocess_tools() -> None:
    """Opt the THROWAWAY sim AUT into ``bash``.

    ``BashTool`` checks ``MAXIM_ALLOW_BASH``; without it every bash call fails
    with "BashTool disabled" even though autonomy allows it. Simulation mode
    is sandboxed (tmpdir + FearGatedExecutor), so bash is safe there. NEVER
    called for an adopted persistent agent: bash is deregistered during the
    campaign anyway, and the env var outlives the sim — it would arm the
    persistent agent's (unsandboxed) bash for later Talk-mode turns.

    Deliberately NOT the gates sandbox-launch added for git_diff / run_tests:
    the AUT registry deregisters both tools alongside bash, so arming them
    here would be dead code. Since #949 both take ``allowed_dirs`` and run in
    ``tool_workdir`` with the host-tool env allowlist, so if a sim ever needs
    them, register them with the sandbox root as ``allowed_dirs``.
    """
    os.environ.setdefault("MAXIM_ALLOW_BASH", "1")


def _stop_by_operator(stop_event: threading.Event, operator_stop: threading.Event) -> None:
    """A human ended the run (``/cancel``, Ctrl+C, an interrupted reader), so it reports ``cancel``.

    ``stop_event`` is also set when a runner simply finishes and again at shutdown, so it cannot tell a
    finished run from a cancelled one; only this helper sets ``operator_stop`` (#1002).
    """
    operator_stop.set()
    stop_event.set()


def _runner_outcome(runner: str, outcome: Any) -> tuple[str, str] | None:
    """How a campaign runner ended, as ``(status, reason)`` for ``finish_context``, or ``None`` when it simply
    completed (#1002). Each runner is read in its OWN return shape (pinned against the real producers):

    - ``generative_runner``: ``None`` when it raised, else a ``GenerativeCampaignResult`` whose
      ``finish_reason`` is ``completed`` / ``max_turns`` / ``error`` / ``yielded`` (a process stop, #1438);
    - ``dm_runner``: ``{"error": ...}`` when it raised, else ``DMRuntime.get_rollup()``, whose ending is
      ``campaign.finish_reason`` (``all_encounters_complete`` / ``campaign_end:*`` complete it; ``cancel`` is
      a Ctrl+C; ``yielded`` a process stop (#1438); anything else is a structural failure);
    - ``fixture_runner``: ``{"error": ...}`` when it raised, else ``asdict(FixtureResult)``, which finishes
      ``complete``;
    - ``precampaign_runner``: the analysis dict, whose ``turns`` each carry ``error`` when that turn failed; a
      failed turn means the scripted run did not happen as written.

    Anything unrecognised reads as ``error``: fail closed.
    """
    if outcome is None:
        return "error", f"{runner} raised (see the log)"
    if isinstance(outcome, dict) and outcome.get("error"):
        return "error", str(outcome["error"])
    if runner == "generative_runner":
        reason = getattr(outcome, "finish_reason", None)
        if reason == "completed":
            return None
        if reason in ("max_turns", "error"):
            return reason, f"the generative campaign ended with {reason}"
        if reason == "yielded":
            return "yielded", "the generative campaign yielded to a process stop"
    elif runner == "dm_runner" and isinstance(outcome, dict):
        reason = (outcome.get("campaign") or {}).get("finish_reason")
        if reason == "all_encounters_complete" or str(reason).startswith("campaign_end:"):
            return None
        if reason == "cancel":
            return "cancel", "the DM campaign was interrupted (Ctrl+C)"
        if reason == "yielded":
            return "yielded", "the DM campaign yielded to a process stop"
    elif runner == "fixture_runner" and isinstance(outcome, dict):
        reason = outcome.get("finish_reason")
        if reason == "complete":
            return None
    elif runner == "precampaign_runner" and isinstance(outcome, dict):
        failed = [t for t in outcome.get("turns", []) if t.get("error")]
        if not failed:
            return None
        return "error", f"{len(failed)} pre-campaign turn(s) failed: {failed[0]['error']}"
    else:
        reason = None
    return "error", f"{runner} ended with {reason!r}"


def _finish_runner(
    bridge: Any, stop_event: threading.Event, runner: str, outcome: Any, *, errors: list | None = None
) -> None:
    """A campaign runner returned: record how it ended (unless a guard already did), then stop the loops.

    Without this a runner that caught its own failure ended the run with no ``finish_context``, which
    resolves to ``completed`` (#1002). ``errors`` is a worker thread's caught exceptions (the interactive DM
    campaign runs on one): any makes the outcome an error.
    """
    ended = _runner_outcome(runner, {"error": repr(errors[0])} if errors else outcome)
    if ended is not None and not (bridge.finish_context or {}).get("status"):
        status, reason = ended
        bridge.finish_context.update(
            {"status": status, "reason": reason, "summary": f"{runner} ended: {reason}", "initiated_by": runner}
        )
    stop_event.set()


def _restore_aut_from_session(
    resume_session: str | None,
    *,
    persistent_agent: Any,
    aut_hippocampus: Any,
    aut_nac: Any,
    aut_memory_hub: Any,
) -> dict[str, Any] | None:
    """Restore the AUT's stores from ``--resume-sim``'s session directory, and say what happened (#1003).

    ``None`` when the run is not a resume. Otherwise ``{"requested", "state_dir", "stores"}``, one outcome per
    store: ``loaded``; ``absent`` (the prior session never wrote it); ``failed:<Exc>``; ``no_store_in_run``
    (written, but this run has no such store); ``skipped_persistent_agent``; ``not_restored``. The directory is
    the EXACT name given; the resume prompt resolves the name separately and also accepts a prefix (#1009),
    which :func:`maxim.simulation.report.resume_stamp` checks. Restores exactly as before: stamping only.
    """
    if not resume_session:
        return None
    from maxim.utils.paths import sim_reports as _sim_reports_dir

    prev_dir = _sim_reports_dir() / resume_session
    paths = {name: prev_dir / f"aut_{name}.json" for name in ("hippocampus", "nac", "ec", "atl")}
    record: dict[str, Any] = {"requested": resume_session, "state_dir": str(prev_dir.resolve()), "stores": {}}
    stores = record["stores"]
    # HANDLE seam (a), branch 2: gated on persistent_agent is None — an
    # adopted agent already restored its live state via auto_load at handle
    # construction; re-loading a session AUT file here would CLOBBER it.
    if persistent_agent is not None:
        for name, path in paths.items():
            stores[name] = "skipped_persistent_agent" if path.exists() else "absent"
        return record
    if aut_hippocampus is None and aut_nac is None:
        for name, path in paths.items():
            stores[name] = "not_restored" if path.exists() else "absent"
        return record

    # The log lines keep their old spelling: operators grep them (e.g. "Restored AUT EC", Roy 5a protocol).
    display = {"hippocampus": "hippocampus", "nac": "NAc", "ec": "EC", "atl": "ATL"}

    def _restore(name: str, store: Any, load: Any, describe: Any) -> None:
        path = paths[name]
        if not path.exists():
            stores[name] = "absent"
        elif store is None:
            stores[name] = "no_store_in_run"
        else:
            try:
                load(str(path))
                stores[name] = "loaded"
                logger.info("Restored AUT %s from %s%s", display[name], path, describe())
            except Exception as e:
                stores[name] = f"failed:{type(e).__name__}"
                logger.debug("Failed to restore AUT %s: %s", display[name], e)

    _restore(
        "hippocampus",
        aut_hippocampus,
        lambda p: aut_hippocampus.load(p),
        lambda: f" ({len(aut_hippocampus)} memories)",
    )
    # apply_decay=False: sims are tick-anchored (agent_loop §8.5); wall-clock elapsed between a training run
    # and its --resume-sim is the OPERATOR's schedule, not agent-experienced time — decaying here would make
    # resume-based harnesses (Exp 44's tau-hold pre-load) silently lose their held cluster biases and turn the
    # resume gap into an unrecorded experimental variable (review fold, Arch #1 + Exec #2, cross-confirmed).
    _restore(
        "nac",
        aut_nac,
        lambda p: aut_nac.load(p, apply_decay=False),
        lambda: f" ({sum(len(v) for v in aut_nac._links.values())} links)",
    )
    aut_ec = aut_memory_hub.ec if aut_memory_hub is not None else None
    _restore("ec", aut_ec, lambda p: aut_ec.load(p), lambda: f" ({len(aut_ec._substrate_nodes)} substrate nodes)")
    aut_atl = aut_memory_hub.atl if aut_memory_hub is not None else None
    _restore("atl", aut_atl, lambda p: aut_atl.load(p), lambda: "")
    return record


def _select_aut_prompt_handler(prompt_handler: Any, stop_event: Any) -> tuple[Any, Any]:
    """``(aut_handler, sim_handler)`` -- the handler the AUT's ``request_interaction`` tool asks, and the
    interactive SimPromptHandler the stdin reader must coordinate with (None unless one was built).

    A caller's ``prompt_handler`` wins (D40: ``maxim.campaign(prompt_handler=...)`` used to be refused
    because nothing threaded it here). Otherwise the interactive mode picks: ON -> SimPromptHandler (the
    stdin reader delivers answers to it), anything else -> the auto handler. The tool itself gates on
    ``sim_logger.should_prompt``, so passing a handler unconditionally is safe."""
    if prompt_handler is not None:
        return prompt_handler, None
    try:
        from maxim.simulation.sim_logger import get_interactive_mode as _get_im
        from maxim.simulation.sim_logger import InteractiveMode as _IM

        if _get_im() == _IM.ON:
            from maxim.interactive.prompts import SimPromptHandler

            sim_handler = SimPromptHandler(stop_event=stop_event)
            return sim_handler, sim_handler
        from maxim.interactive.prompts import create_handler

        return create_handler("auto"), None
    except Exception as _ph_exc:
        logger.debug("PromptHandler unavailable for AUT: %s", _ph_exc)
        return None, None


def _show_report_and_post_sim_prompt(report: Any, report_dir: str) -> Any:
    """Show the report IN the Live panel so the user can scroll it, then prompt for a new goal; returns the active
    display (None without one). Moved out of ``start_simulation_mode``'s interactive tail (#1454's interim prompt
    fix carried by the move, rule T)."""
    from maxim.simulation.report import print_report
    from maxim.simulation.sim_logger import get_active_display

    # Reset scroll to bottom first so the report appears at the end,
    # then scroll up to show the start of the report.
    display = get_active_display()
    if display is not None:
        display.scroll(-999999)  # Jump to bottom before report

    log_count_before = display.log_count if display is not None else 0
    _saved_session_dir = Path(report_dir) / report.session_id
    print_report(report, session_dir=_saved_session_dir)
    log_count_after = display.log_count if display is not None else 0
    report_lines = log_count_after - log_count_before

    if display is not None:
        # Scroll up to show the start of the report
        if report_lines > 0:
            display.scroll(report_lines)
        # No promise that learning reaches the next run: the re-entered run builds a fresh AUT (#1454); the 1.3.5
        # /new slice makes the learning carry and restores the promise with it (owner decision 2026-10-26).
        display.set_prompt("Tell me a new simulation to imagine, or press Enter to finish.\n\n> ")
        display.set_urgent(True)
    return display


def _await_aut_session_end_before_new(aut_memory_hub: Any | None) -> None:
    """Before ``/new`` restarts with a new goal, the AUT's memory session must have ended and consolidated.

    The AUT's own loop owns that session (#1438 PR 0): its ``bio_session`` ended it, as owner, when the loop exited.
    ``/new`` is a non-owner, so it WAITS for that end (bounded) rather than ending a session it does not hold.
    Without the end, in-session learning (NAc decay, hippocampus consolidation, semantic embeddings) is lost on /new.
    """
    if aut_memory_hub is None:
        return
    try:
        if not aut_memory_hub.wait_session_ended(deadline_s=_NEW_GOAL_SESSION_WAIT_S):
            logger.warning("session_end_not_reached before /new: the AUT's memory session is still open or ending")
    except Exception as _mhe:
        logger.warning("Memory consolidation before /new failed: %s", _mhe)


_NEW_GOAL_SESSION_WAIT_S = 30.0


def _refuse_a_stop_then_reset_llm_cancellation() -> None:
    """A simulation's first act: refuse to start under a process stop request, then re-arm the LLM backend.

    The refusal (#1438 PR 1c, owner decision 2026-10-30) comes before anything is built, the start provenance capture
    (git calls, model-file hashing, a serving registration) included: a sim started under a stop would spend the
    stop's budget building a router, a sandbox and loops that are born stopped, and report a run that never
    happened.
    """
    stop = current_stop()
    if stop is not None:
        raise StopStillPending(stop.reason, refused="simulation")
    # Reset the process-wide LLM cancellation primitive. If a previous sim
    # in the same Python process requested shutdown (e.g., this is a
    # --continuous run or a long-lived REPL invoking multiple sims), the
    # event would still be set and the new sim's LLM calls would bail out
    # immediately. Clearing at entry makes successive sims safe.
    try:
        from maxim.models.language.cancellation import reset_shutdown

        reset_shutdown()
    except Exception as e:
        logger.debug("Failed to reset LLM cancellation at sim start: %s", e)


def _await_llm_ready_or_refuse(llm_router: Any | None, start_provenance: dict[str, Any]) -> None:
    """Wait for the model to load (avoid cold-start stale drops), then refuse if a stop landed during the load.

    ``wait_ready`` returns early on a stop (#1438 PR 1c); the sim then refuses before the bridge, sandbox and loops
    are built, releasing its serving registration (a held one would stamp every later sim ``concurrent_sims``). The
    router and its lane manager are left as a finished sim leaves them (neither has a shutdown).
    """
    if llm_router is None:
        return
    logger.info("Waiting for LLM model to load...")
    llm_router.wait_ready(timeout=120.0)
    stop = current_stop()
    if stop is not None:
        from maxim.simulation.report import release_start_provenance

        release_start_provenance(start_provenance)
        raise StopStillPending(stop.reason, refused="simulation")
    logger.info("LLM ready")


def start_simulation_mode(
    goal: str,
    mode: str = "generative",
    max_turns: int = 50,
    min_finish_turns: int = 0,
    response_timeout: float = 120.0,
    debug: bool = False,
    # Deprecated alias kept for backward compat with older callers.
    sim_debug: bool | None = None,
    resume_session: str | None = None,
    continuous: bool = False,
    no_sim_env: bool = False,
    sandbox_backend: str = "auto",
    sandbox_image: str = "python:3.12-slim",
    sandbox_network: str = "none",
    aut_model: str | None = None,
    aut_mode: str = "llm-primary",
    research_telemetry: bool = False,
    pre_campaign_turns: list[dict[str, Any]] | None = None,
    dm_campaign: Any = None,
    generative: bool = False,
    arc_yaml: str | None = None,
    experiment_log: Any = None,
    fixture_path: str | None = None,
    entity_ref: str | None = None,
    persistent_agent: Any = None,
    prompt_handler: Any = None,
) -> SimulationResult:
    """Boot simulation mode: AUT + orchestrator + stdin reader.

    This is the main entry point called from cli.py when --sim agent is used.

    Args:
        goal: The simulation objective (e.g., "test safety boundaries")
        mode: Orchestrator flow-shape label ("generative" default; "dm",
            "research", "benchmark" set by their dispatch paths). Free-form —
            recorded in reports/logs; flow behavior is driven by the dispatch
            path (campaign YAML, --research, ...), not by this label.
        max_turns: Maximum simulation turns before auto-finish
        min_finish_turns: With --sim-run-full-turns, the cap; the narrator's finish_simulation is refused below it
            (0: none; observe-only runs are always exempt)
        response_timeout: Default timeout for send_and_wait()
        debug: Enable verbose debug tracing (pipeline polling, loop
            heartbeats, lane activity).
        sim_debug: Deprecated alias for ``debug``. Kept so older scripts
            still work — prefer ``debug`` in new code.
        entity_ref: SEM component reference (e.g., ``"weapons/rusty_sword"``).
            When set, the AUT's executor loads the entity via
            ``ComponentRegistry`` and registers affordance tools. Requires
            ``aut_pain_bus`` (Embodiment publishes through it) and ``aut_nac``
            (bridge needs it for direct attribution).
        persistent_agent: HANDLE seam (a) — a live ``AgentInstance`` to ADOPT
            as the AUT instead of constructing a throwaway ``sim_aut``
            (docs/plans/archive/console_handle_campaign_injection.md). When set: the
            campaign learns into the persistent agent's own
            Hippocampus/NAc/MemoryHub (its home, its agent_id), the
            resume-session file-load is skipped (live state was already
            restored via ``auto_load``), sim/DM tools are leased onto the
            agent's registry and restored on stop, the AUT loop ends with
            ``consolidation="full"``, and the session-dir AUT snapshot is not
            written. Mutually exclusive with ``entity_ref`` — a persistent
            agent owns its embodiment (declared at construction, e.g. the
            Reachy-flavored handle), the sim must not graft one on.
            Known divergence (post-merge review, documented): user tools
            pending via ``maxim.register_tool()`` are injected AFTER the
            lease snapshot, so on the injected path they are CAMPAIGN-scoped
            (dropped at lease restore) rather than process-persistent as on
            the throwaway path. Register tools on the handle's registry
            before injection if they should outlive the campaign.
        prompt_handler: The handler the AUT's ``request_interaction`` tool asks (D40) -- a caller's
            own UI. None picks one from the interactive mode, as before. Cannot be combined with
            ``persistent_agent``, whose registry already holds its handler.

    Returns:
        SimulationResult with session summary
    """
    if prompt_handler is not None and persistent_agent is not None:
        raise ValueError(
            "start_simulation_mode(prompt_handler=...) cannot be combined with persistent_agent=: the adopted "
            "agent's registry already holds its request_interaction handler"
        )
    # Merge legacy sim_debug alias into canonical `debug` flag.
    if sim_debug is not None and not debug:
        debug = bool(sim_debug)

    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SupervisionPolicy
    from maxim.agents.llm_worker import LLMWorker
    from maxim.agents.maxim_agent import MaximAgent
    from maxim.models.language.router import LLMRouter, load_llm_config
    from maxim.runtime.lane_backends import build_primary_router
    from maxim.runtime.agent_loop import run_agentic_loop
    from maxim.runtime.bootstrap import (
        build_decision_engine,
        build_memory,
        build_tool_registry,
    )
    from maxim.simulation.bridge import SimulationBridge
    from maxim.simulation.conversational_source import ConversationalSource
    from maxim.simulation.introspection import Observer
    from maxim.simulation.tools import (
        AnalyzeResultsTool,
        CheckCompletionTool,
        DamageComponentTool,
        ExtendSimulationTool,
        FinishSimulationTool,
        InjectPainTool,
        InspectAUTTool,
        ObserveActionsTool,
        OrchestratorActorTool,
        SendMessageTool,
        SetEntitySensorTool,
        SimRespondTool,
        SpawnSubSimulationTool,
    )

    _refuse_a_stop_then_reset_llm_cancellation()
    start_time = time.time()
    start_provenance = capture_start_provenance(scenario_objects=(dm_campaign, pre_campaign_turns))  # M1; B0

    # ── HANDLE seam (a): adopt an injected persistent agent EARLY ────────
    # Validation is loud + first so a mis-built instance fails before any
    # heavy construction (LLM router, sandbox). The adopted bindings replace
    # the sim_aut factory path below.
    _adopted: _AdoptedAgent | None = None
    if persistent_agent is not None:
        if entity_ref is not None:
            raise ValueError(
                "entity_ref is incompatible with persistent_agent — a persistent agent owns its "
                "embodiment (declare it at AgentConfig.embodiment_ref when building the handle)"
            )
        _adopted = _adopt_persistent_agent(persistent_agent)
    _aut_agent_id = _adopted.agent_id if _adopted is not None else "sim_aut"

    # Register well-known sim agent nicknames for display logging.
    # Must happen early — before any sim_log calls that might carry agent_id.
    from maxim.simulation.sim_logger import register_agent_nickname, sim_agent_context

    register_agent_nickname(_aut_agent_id, "AUT")
    register_agent_nickname("sim_orchestrator", "Orch")

    # Plan 4 follow-up (2026-04-14): generate session_id AT ENTRY so
    # every LLMWorker constructed downstream can thread it into its
    # request_context dict. Previously this id was created lazily in
    # ``build_report`` after the sim finished, which was too late —
    # every peer_backend_call event during the sim's lifetime logged
    # session_id=null. The same ``time.strftime`` timestamp is forwarded
    # to ``build_report(session_id=...)`` later so the sim directory
    # name matches the session_id in the log trace; cross-correlating
    # report files with JSONL events is now a string-equality match
    # instead of a "figure out the right timestamp window" exercise.
    # Matches the pattern already in ``research_orchestrator.py:73``.
    session_id = time.strftime("%Y%m%d_%H%M%S")

    # ── Shared stop event ────────────────────────────────────────────────
    stop_event = threading.Event()
    operator_stop = threading.Event()  # set only when a human ends the run (#1002)

    # ── Shared LLM router (single model, alternating inference) ──────────
    # Routed through the multi-LLM factory so sim respects per-lane assignments
    # (capability-driven profiles, env overrides, remote URLs, safety gates).
    llm_router, _lane_manager = build_primary_router(logger=logger)
    if llm_router is None:
        # Factory returned nothing → fall back to the default global config path.
        llm_config = load_llm_config()
        if llm_config.enabled:
            llm_router = LLMRouter(llm_config)
    if llm_router is not None:
        llm_router.warmup()
        logger.info("Shared LLM router initialized")

    # ── Simulation bridge ────────────────────────────────────────────────
    # Substrate-primary per-turn action budget (apparatus standard S6; the
    # Exp 48 thrashing fix). Opt-in via MAXIM_SUBSTRATE_ACTIONS_PER_TURN;
    # unset = unbounded (the pre-fix stopwatch regime: actions/turn =
    # narrator wall-clock ÷ 0.5 s, machine-dependent). Logged here so every
    # run's log states the apparatus configuration.
    from maxim.simulation.bridge import read_substrate_actions_per_turn_env

    _substrate_budget = read_substrate_actions_per_turn_env()
    if _substrate_budget is not None and aut_mode == "substrate-primary":
        logger.info(
            "substrate-primary action budget: %d action(s) per sim turn (MAXIM_SUBSTRATE_ACTIONS_PER_TURN)",
            _substrate_budget,
        )
    elif _substrate_budget is not None:
        # Loud-misconfig norm (review fold): the env is set but the bound
        # only applies to substrate-primary — say so instead of silently
        # ignoring it.
        logger.info(
            "MAXIM_SUBSTRATE_ACTIONS_PER_TURN=%d set but aut_mode=%r — budget applies only to substrate-primary; ignored",
            _substrate_budget,
            aut_mode,
        )
    _await_llm_ready_or_refuse(llm_router, start_provenance)
    bridge = SimulationBridge(
        response_timeout=response_timeout,
        stop_event=stop_event,
        aut_mode=aut_mode,
        substrate_actions_per_turn=_substrate_budget,
    )

    # ── Orchestrator percept source (receives goal + user commands) ──────
    orchestrator_source = ConversationalSource()

    # ── Simulation sandbox ───────────────────────────────────────────────
    # <data_home>/sim_sandbox via _prepare_sim_workspace — never CWD-relative.
    # (The former CWD-relative data/agents/MaximAgent/runtime mkdir was dead
    # code: the loop's state_<run_id>.json writer creates its own parents in
    # atomic_io; that writer's CWD-relative path itself is ledger D15.)
    sim_workspace, sim_tmpdir, log_path = _prepare_sim_workspace()

    # Enable sim logging (always persist to JSONL; terminal traces if --debug)
    try:
        from maxim.simulation.sim_logger import enable_sim_logging

        enable_sim_logging(log_path=log_path, debug=debug)
    except Exception as e:
        logger.warning("Failed to enable sim logging — sim will run but no JSONL trace: %s", e)

    # Build AUT pain bus + sandbox together. The pain bus MUST exist
    # before the sandbox so PainTriggerLayer can route signals; this
    # helper enforces that ordering and is independently testable.
    sim_sandbox, sandbox_root, aut_pain_bus = _setup_sim_sandbox(
        backend=sandbox_backend,
        image=sandbox_image,
        network=sandbox_network,
        populate=not no_sim_env,
        announce=True,
        # HANDLE seam (a): an adopted agent's own bus (with its learners)
        # carries sandbox pain; None keeps today's build-here behavior.
        pain_bus=_adopted.pain_bus if _adopted is not None else None,
    )

    # ── Build AUT pipeline ───────────────────────────────────────────────
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime.state import RuntimeState

    aut_env = FileSystemEnv(str(sim_tmpdir))
    aut_state = RuntimeState()
    aut_state.data["mode"] = "active"
    aut_state.data["in_simulation"] = True
    aut_state.data["active_goal"] = goal
    aut_memory = build_memory()

    # Arm the gated subprocess tools for the throwaway AUT only (see
    # `_arm_sandboxed_aut_subprocess_tools` for why, and why NOT when adopted).
    if _adopted is None:
        _arm_sandboxed_aut_subprocess_tools()

    # Constrain AUT filesystem tools to sandbox tmpdir (if available)
    sandbox_dirs = [sandbox_root, str(sim_tmpdir)] if sandbox_root else None

    # Give the AUT a ResponseOutput so RespondTool is registered.
    # Without it, LLM timeout fallbacks (which generate respond actions)
    # fail with "Tool not registered: respond".
    aut_response_output = None
    try:
        from maxim.utils.response_output import ResponseOutput

        aut_response_output = ResponseOutput(sandbox_path=str(sim_tmpdir))
    except Exception as e:
        logger.debug("Failed to create ResponseOutput for AUT: %s", e)

    # PromptHandler: routes `request_interaction` tool calls to the user.
    # When --interactive is on, use SimPromptHandler so the stdin reader
    # coordinates input (avoids two threads fighting over stdin).
    # The tool itself gates on sim_logger.should_prompt, so it's safe
    # to pass unconditionally.
    aut_prompt_handler, _sim_prompt_handler = _select_aut_prompt_handler(prompt_handler, stop_event)
    if _sim_prompt_handler is not None:
        # Gate the bridge so orchestrator waits for user to answer prompts
        bridge._prompt_gate = _sim_prompt_handler

    if _adopted is not None:
        # HANDLE seam (a), branch 4: the campaign runs on the PERSISTENT
        # agent's registry (its executor is bound to it — a fresh registry
        # would dispatch nothing). Sim/DM tools registered below are leased:
        # the lease snapshot was taken at adoption, restore happens at
        # shutdown so the agent's registry returns to its pre-campaign state.
        aut_registry = _adopted.registry
        # respond/speak are how SimulationBridge.send_and_wait collects the
        # AUT's reply text; the handle's registry (built without a
        # response_output) lacks them, so every DM turn would silently read
        # back None. Registered AFTER the lease snapshot → campaign-scoped,
        # dropped at restore. (Review fold: Executor #1 / Architecture #5.)
        if aut_response_output is not None:
            try:
                from maxim.tools.response import RespondTool, SpeakTool

                aut_registry.register(RespondTool(aut_response_output))
                aut_registry.register(SpeakTool(aut_response_output))
            except Exception as e:
                logger.warning("Failed to lease respond/speak onto adopted registry: %s", e)
    else:
        aut_registry = build_tool_registry(
            internet_launch_enabled=False,
            operational_mode="active",
            allowed_dirs_override=sandbox_dirs,
            response_output=aut_response_output,
            prompt_handler=aut_prompt_handler,
        )
    # Inject user-registered tools from maxim.register_tool() / @maxim.tool
    from maxim.api import _inject_registered_tools

    _inject_registered_tools(aut_registry)

    aut_decision_engine = build_decision_engine()
    aut_agent = MaximAgent()

    # Bridge user event subscriptions to the AUT agent's bus
    if hasattr(aut_agent, "_bus"):
        from maxim.api import _bridge_event_subscriptions

        _bridge_event_subscriptions(aut_agent._bus)

    # AUT runs AUTONOMOUS — no human confirmation prompts (SUPERVISED would
    # deadlock because stdin is captured by the orchestrator's reader thread).
    # FearGatedExecutor is wired below to gate all tool calls through FearAgent.
    aut_autonomy = AutonomyController(
        initial_level=AutonomyLevel.AUTONOMOUS,
        supervision_policy=SupervisionPolicy(
            allowed_tools={
                "respond",
                "speak",
                "read_file",
                "list_directory",
                "write_file",
                "edit_file",
                "glob",
                "code_search",
                "bash",
                "execute_file",
                "run_tests",
                # Narrative tools (sim-only)
                "say",
                "think",
                # Introspection tools
                "memory_recall",
                "predict_outcome",
                "causal_links",
                "pain_history",
                "temporal_patterns",
                "energy_status",
                "concept_query",
                "similarity_search",
                "system_stats",
                # Interactive tools (gated by should_prompt())
                "request_interaction",
                "display_mode",
                "set_scene",
            },
            forbidden_tools=set(),
            min_confidence_autonomous=0.3,
        ),
    )

    # Build AUT's energy tracking (wired to LLMWorker for real token data)
    aut_energy_registry = None
    try:
        from maxim.energy.registry import EnergyRegistry
        from maxim.energy.llm_tracker import LLMEnergyTracker

        aut_energy_registry = EnergyRegistry()
        aut_energy_registry.register(LLMEnergyTracker())
        logger.info("AUT energy tracking enabled")
    except Exception as e:
        logger.debug("AUT energy tracking not available: %s", e)

    # ── AUT agent construction via AgentFactory (F3 migration) ─────────
    # Replaces ~90 lines of hand-rolled bio-stack + cerebellum construction.
    # The factory composes build_bio_stack (bio-pipeline) + build_executor
    # (tool execution + ToolPainBridge) + optional FearGatedExecutor into a
    # single create_full_agent call. Sim-specific wiring (pain layers,
    # DefaultNetwork, tracers, introspection tools) stays below.
    #
    # ComponentRegistry must be created before the factory call so
    # build_executor can instantiate the entity and register affordance tools.
    aut_component_registry = None
    if entity_ref is not None:
        from maxim.embodiment.component_registry import ComponentRegistry

        aut_component_registry = ComponentRegistry()
        logger.info("AUT ComponentRegistry created for entity_ref=%r", entity_ref)

    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    # pain_bus decision for executor subscription:
    # - entity_ref is None: with_pain_bridge=False → pain_bus not passed to
    #   executor. Out-of-band pain→NAc is handled by create_pain_nac_subscriber
    #   on aut_pain_bus. Bridge still constructed for direct attribution
    #   (record_tool_complete / record_tool_embodiment_failure).
    # - entity_ref is set: with_pain_bridge=True → pain_bus passed to executor.
    #   build_executor precondition requires a live bus for Embodiment._publish_pain.
    # with_fear_gate=False — sim wraps FearGatedExecutor AFTER pain
    # layers (PainInterceptor + AnticipatoryPain) so the fear gate is
    # outermost. The factory wraps fear gate directly on the executor
    # which would put it under the pain layers (wrong order).
    if _adopted is not None:
        # HANDLE seam (a), branch 1: ADOPT the live persistent instance —
        # no create_full_agent, no "sim_aut", no sim_tmpdir persistence.
        # Every aut_* binding below derives from this instance, which is
        # what routes campaign episodes into the persistent agent's home.
        _aut_instance = _adopted.instance
    else:
        _aut_config = AgentConfig(
            agent_id="sim_aut",
            role="pc",
            persistence_dir=str(sim_tmpdir),
            with_bio_stack=True,
            with_executor=True,
            with_pain_bridge=entity_ref is not None,
            with_fear_gate=False,
            embodiment_ref=entity_ref,
        )
        _aut_factory = AgentFactory(
            component_registry=aut_component_registry,
            base_data_dir=sim_tmpdir,
        )
        _aut_instance = _aut_factory.create_full_agent(
            _aut_config,
            tool_registry=aut_registry,
            pain_bus=aut_pain_bus,
            fear_llm=llm_router,
        )

    # Extract bio-system references for downstream sim-specific wiring.
    aut_hippocampus = _aut_instance.hippocampus
    aut_nac = _aut_instance.nac
    aut_memory_hub = _aut_instance.memory_hub
    aut_executor = _aut_instance.executor
    # PFC deliberation: ThoughtGate + BioEnrichmentPipeline from BioStack
    # (constructed in build_bio_stack, no longer sim-only).
    _aut_bio_stack = _aut_instance.bio_stack
    aut_bio_enrichment_pipeline = (
        getattr(_aut_bio_stack, "bio_enrichment_pipeline", None) if _aut_bio_stack is not None else None
    )
    _aut_thought_gate = getattr(_aut_bio_stack, "thought_gate", None) if _aut_bio_stack is not None else None

    if aut_memory_hub is not None:
        aut_agent.wire_memory_hub(aut_memory_hub)

    if entity_ref is not None and _aut_instance.embodiment is not None:
        logger.info(
            "AUT Embodiment loaded: entity_ref=%r, pain_bus=injected, affordance tools registered",
            entity_ref,
        )

    # Restore AUT state from previous session if resuming; what loaded is stamped into the report (#1003).
    resume_record = _restore_aut_from_session(
        resume_session,
        persistent_agent=persistent_agent,
        aut_hippocampus=aut_hippocampus,
        aut_nac=aut_nac,
        aut_memory_hub=aut_memory_hub,
    )

    # Attach bio-system tracers based on --debug flags / env vars
    def _env_trace(var: str) -> bool:
        return os.environ.get(var, "").strip().lower() in ("1", "true", "t", "yes", "y", "on")

    if aut_hippocampus is not None and (_env_trace("MAXIM_HIPPO_TRACE") or debug):
        try:
            from maxim.memory.hippo_tracer import HippocampusTracer

            HippocampusTracer(aut_hippocampus)
        except Exception as e:
            logger.debug("Hippo tracer not available: %s", e)

    if aut_nac is not None and (_env_trace("MAXIM_NAC_TRACE") or debug):
        try:
            from maxim.decisions.nac_tracer import NacTracer

            NacTracer(aut_nac)
        except Exception as e:
            logger.debug("NAc tracer not available: %s", e)

    if aut_memory_hub is not None and (_env_trace("MAXIM_ATL_TRACE") or debug):
        try:
            from maxim.memory.atl_tracer import ATLTracer

            atl = getattr(aut_memory_hub, "_atl", None) or getattr(aut_memory_hub, "atl", None)
            if atl is not None:
                ATLTracer(atl)
        except Exception as e:
            logger.debug("ATL tracer not available: %s", e)

    # --- AUT introspection tools ---
    # Give the AUT access to its own cognitive subsystems so it can
    # actively recall memories, predict outcomes, etc. during action
    # selection.  Without these, the AUT has memories but can't query them.
    try:
        from maxim.tools.introspection import (
            MemoryRecallTool,
            PredictOutcomeTool,
            CausalLinksTool,
            TemporalPatternsTool,
            EnergyStatusTool,
            ConceptQueryTool,
            SimilaritySearchTool,
            SystemStatsTool,
        )

        if aut_hippocampus is not None:
            aut_registry.register(MemoryRecallTool(hippocampus=aut_hippocampus))
            logger.info("AUT introspection: memory_recall registered")

        if aut_memory_hub is not None:
            if aut_memory_hub.ec is not None:
                aut_registry.register(SimilaritySearchTool(ec=aut_memory_hub.ec))
            if aut_memory_hub.scn is not None:
                aut_registry.register(TemporalPatternsTool(scn=aut_memory_hub.scn))
            if aut_memory_hub.atl is not None:
                aut_registry.register(ConceptQueryTool(atl=aut_memory_hub.atl))

        if aut_nac is not None:
            aut_registry.register(PredictOutcomeTool(nac=aut_nac))
            aut_registry.register(CausalLinksTool(nac=aut_nac))

        if aut_energy_registry is not None:
            trackers = list(aut_energy_registry._trackers.values()) if hasattr(aut_energy_registry, "_trackers") else []
            energy_tracker = trackers[0] if trackers else None
            if energy_tracker is not None:
                aut_registry.register(EnergyStatusTool(energy_tracker=energy_tracker))

        aut_registry.register(
            SystemStatsTool(
                hippocampus=aut_hippocampus,
                nac=aut_nac,
                ec=aut_memory_hub.ec if aut_memory_hub else None,
                atl=aut_memory_hub.atl if aut_memory_hub else None,
                energy_tracker=None,
                pain_detector=None,
                significance_learner=None,
            )
        )

        logger.info("AUT introspection tools registered")
    except Exception as e:
        logger.debug("Failed to register AUT introspection tools: %s", e)

    # --- AUT narrative tools (sim-only) ---
    # Let the AUT speak in-world, reason explicitly, and examine scene details.
    # ThinkTool gets the BioEnrichmentPipeline for L1 enrichment.
    # NOTE: ThoughtGate + BioEnrichmentPipeline now come from BioStack
    # (constructed in build_bio_stack, extracted above). The dead
    # wire_thought_gate / wire_bio_enrichment calls to ExecAgent are
    # removed — deliberation lives in the agentic loop's PFC cycle.
    if _aut_thought_gate is not None:
        logger.info(
            "AUT ThoughtGate + BioEnrichment from BioStack (refractory=%d, min_score=%.1f)",
            _aut_thought_gate._config.refractory_ticks,
            _aut_thought_gate._config.min_combined_score,
        )

    try:
        from maxim.tools.narrative import ExamineTool, SayTool, ThinkTool

        aut_registry.register(SayTool())
        aut_registry.register(ThinkTool(pipeline=aut_bio_enrichment_pipeline))
        aut_registry.register(ExamineTool(bridge=bridge, hippocampus=aut_hippocampus))
        logger.info("AUT narrative tools registered (say, think, examine)")
    except Exception as e:
        logger.debug("Failed to register AUT narrative tools: %s", e)

    # --- Deregister irrelevant tools in sim mode ---
    # Robot tools return "No live robot connected"; dev tools (bash, git,
    # filesystem editing) are noise that dilutes the LLM's attention away
    # from sim-relevant tools (affordances, narrative, introspection,
    # interaction).  Audit (2026-04-20) found 35+ tools on the AUT — the
    # LLM defaults to safe tools instead of exploring affordances.
    _irrelevant_tools = [
        # Robot-only (no hardware in sim)
        "focus_interests",
        "track_target",
        "move",
        "novelty_track",
        "maxim_command",
        "autonomy_level",
        "mode_switch",
        # Dev tools (sim agents don't write code)
        "bash",
        "git_commit",
        "git_diff",
        "edit_file",
        "execute_file",
        "run_tests",
        "search_code",
        "request_directory_change",
        "glob",
        "read_file",
        "write_file",
    ]
    # When adopted, dropped persistent tools come back at lease restore —
    # the snapshot holds the baseline {name: tool_object} map.
    for _rt in _irrelevant_tools:
        if aut_registry.deregister(_rt):
            logger.debug("Deregistered irrelevant tool from AUT: %s", _rt)

    # --- SEM Tool Discovery: hybrid prompt mode ---
    # Replace per-entity sensor tools with universal sense + sense_tools.
    # Keep top-k goal-relevant affordance tools visible, deactivate the rest.
    _aut_entity_map = None
    if entity_ref is not None and _aut_instance.embodiment is not None:
        from maxim.embodiment.entity_map import EntityMap
        from maxim.tools.discovery import (
            SenseToolsTool,
            SensePresenceTool,
            UniversalSenseTool,
            select_goal_relevant_tools,
        )

        _aut_entity_map = EntityMap()
        _aut_entity_map.register_self(_aut_instance.embodiment.root)

        # Deregister per-entity sensor tools (replaced by universal sense)
        _sensor_tools_removed = 0
        for _tname in list(aut_registry.list_all()):
            if _tname.startswith("sense_") or _tname.startswith("read_"):
                if aut_registry.deregister(_tname):
                    _sensor_tools_removed += 1
        logger.debug("SEM discovery: removed %d per-entity sensor tools", _sensor_tools_removed)

        # Register universal sense + sense_tools + sense_presence as core tools
        aut_registry.register(UniversalSenseTool(entity_map=_aut_entity_map))
        aut_registry.register(
            SenseToolsTool(
                entity_map=_aut_entity_map,
                tool_registry=aut_registry,
                component_index=None,  # wired later if imagination is available
            )
        )
        # sense_presence: imagination_trigger wired later (after trigger construction)
        aut_registry.register(
            SensePresenceTool(
                entity_map=_aut_entity_map,
                imagination_trigger=None,  # wired below after ImaginationTrigger is built
            )
        )

        # Goal-based top-k: keep relevant affordances active, deactivate rest.
        # generate_tools_for_entity registers tools as core (no scene metadata).
        # We need to add scene metadata so deactivate_tool works. Re-register
        # affordance tools as scene-scoped, then deactivate non-top-k ones.
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool
        from maxim.tools.discovery import mark_goal_selected

        _keep_active = set(select_goal_relevant_tools(goal, _aut_entity_map, aut_registry))
        mark_goal_selected(list(_keep_active))

        # Collect affordance tools and re-register as scene-scoped
        _affordance_tools = []
        for _tname in list(aut_registry.list_all()):
            try:
                _tool = aut_registry.get(_tname)
                if isinstance(_tool, ModulatorAffordanceTool):
                    _affordance_tools.append(_tool)
            except KeyError:
                pass

        if _affordance_tools:
            aut_registry.register_scene_tools(_affordance_tools, "aut_affordances")

        # Deactivate non-top-k affordance tools individually
        _deactivated_count = 0
        for _tool in _affordance_tools:
            if _tool.name not in _keep_active:
                if aut_registry.deactivate_tool(_tool.name):
                    _deactivated_count += 1

        _active_count = len(aut_registry.list())
        _all_tools = list(aut_registry.list_all())
        _active_tools = list(aut_registry.list())
        logger.info(
            "SEM discovery: hybrid prompt mode ��� %d active tools "
            "(%d goal-selected, %d affordance tools deactivated, "
            "sense + sense_tools registered)",
            _active_count,
            len(_keep_active),
            _deactivated_count,
        )
        # Trace: log all registered tools for debugging SEM availability
        try:
            from maxim.simulation.sim_logger import sim_log

            sim_log("SEM_TRACE", f"All registered tools ({len(_all_tools)}): {', '.join(sorted(_all_tools))}")
            sim_log("SEM_TRACE", f"Active tools ({len(_active_tools)}): {', '.join(sorted(_active_tools))}")
            sim_log("SEM_TRACE", f"Goal-selected (top-k): {', '.join(sorted(_keep_active))}")
            sim_log(
                "SEM_TRACE",
                f"Entities: {', '.join(e.name + '(' + e.entity_type + ')' for e in _aut_entity_map.list_entities())}",
            )
            # Log affordances per entity
            for _ent in _aut_entity_map.list_entities():
                for _mod_name, _mod in _ent.modulators.items():
                    _aff_names = list(_mod.affordances.keys())
                    if _aff_names:
                        sim_log(
                            "SEM_TRACE",
                            f"  {_ent.name}.{_mod_name}: affordances={', '.join(_aff_names)}",
                        )
        except Exception:
            pass

    # AUT PainBus subscriptions are now handled by build_bio_stack above
    # (Wave 3: pre-built pain_bus= parameter subscribes standard learners).

    # Build AUT's LLM worker. When --aut-model is set, the AUT gets its
    # own LLMRouter so there's no inference contention and the experiment
    # can isolate memory recall from LLM context window effects.
    #
    # The AUT router uses the same infrastructure as the primary router
    # (peer config, lane backends, etc.) but with a different model profile
    # set via env var override. This ensures the AUT uses the leader's GPU
    # (via peer config) rather than trying to load locally.
    aut_router = llm_router  # default: shared
    if aut_model is not None and llm_router is not None:
        try:
            # Override the infer lane model for this second router build
            old_profile = os.environ.get("MAXIM_LLM_PROFILE")
            os.environ["MAXIM_LLM_PROFILE"] = aut_model
            aut_router, _ = build_primary_router(logger=logger)
            # Restore original profile
            if old_profile is not None:
                os.environ["MAXIM_LLM_PROFILE"] = old_profile
            else:
                os.environ.pop("MAXIM_LLM_PROFILE", None)

            if aut_router is None:
                aut_config = load_llm_config(profile_override=aut_model)
                if aut_config.enabled:
                    aut_router = LLMRouter(aut_config)

            if aut_router is not None:
                aut_router.warmup()
                aut_router.wait_ready(timeout=120.0)
                logger.info("AUT router initialized (model=%s)", aut_model)
            else:
                aut_router = llm_router
        except Exception as e:
            logger.warning("Failed to build AUT router for '%s': %s — falling back to shared", aut_model, e)
            aut_router = llm_router

    aut_llm_worker: LLMWorker | None = None
    if aut_router is not None:
        aut_llm_worker = LLMWorker(
            llm=aut_router,
            stale_threshold_s=30.0 if aut_router is llm_router else 15.0,
            n_ctx=aut_router.n_ctx,
            token_counter=aut_router.get_token_counter(),
            session_id=session_id,
        )
        aut_llm_worker.start()

        # Exp 44 counterfactual capture (DORMANT unless MAXIM_EXP44_CAPTURE_LOG
        # is set). Installs the paired-prompt capture on THIS process's AUT worker
        # — the one place reachable in-process, since the benchmark vehicle
        # subprocesses `maxim --sim`. Gated experiment hook, mirrors how
        # substrate_telemetry is threaded through the sim; never runs in prod.
        _exp44_log = os.environ.get("MAXIM_EXP44_CAPTURE_LOG")
        if _exp44_log:
            try:
                import importlib.util as _ilu
                from pathlib import Path as _P

                _cap_path = _P(__file__).resolve().parents[3] / "scripts" / "exp44" / "capture_paired_prompts.py"
                _spec = _ilu.spec_from_file_location("_exp44_capture", _cap_path)
                if _spec and _spec.loader:
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    _mod.install_capture(aut_llm_worker, _exp44_log)
                    logger.info("Exp44 paired-prompt capture installed -> %s", _exp44_log)
            except Exception as _e:  # capture must never break the sim
                logger.warning("Exp44 capture install failed (continuing): %s", _e)

        # B3: Enable Acting Coach when embodiment tools are available.
        # The coach encourages the agent to explore affordances instead of
        # entering respond loops. Bio-system modulation (NAc valence, pain
        # anticipation, cerebellum predictions) annotates the base directive
        # via existing StructuredContext fields at prompt-build time.
        if entity_ref is not None:
            from maxim.prompts.acting_coach import acting_coach_config_from_env

            aut_llm_worker.acting_coach = acting_coach_config_from_env()
            aut_llm_worker.is_embodied = True

            # E2: Inject entity context into AUT prompt
            if aut_component_registry is not None:
                try:
                    _entity_raw = aut_component_registry.get(entity_ref)
                    aut_llm_worker.entity_spec = _entity_raw.get("entity", _entity_raw)
                except Exception as _e:
                    logger.debug("Entity context injection failed: %s", _e)

    # ── Build orchestrator pipeline ──────────────────────────────────────
    orch_env = FileSystemEnv(str(sim_tmpdir))
    orch_state = RuntimeState()
    orch_state.data["mode"] = "singularity"  # No allowed_tools filter — all registered tools visible
    # NB: data["mode"] above is the RuntimeState RUN-mode; the sim flow-shape
    # label (this function's `mode` param) rides in "strategy" for capture
    # metadata (hippocampus _SnapshotState.strategy) — do not conflate.
    orch_state.data["strategy"] = mode
    orch_memory = build_memory()
    orch_decision_engine = build_decision_engine()

    # Orchestrator bio-stack + executor constructed below via AgentFactory
    # after the orch tool registry is built (the factory needs it).
    orch_agent = MaximAgent()

    # True when a human drives the conversation (no specific goal).
    # Stall detector and probing are disabled in this mode.
    from maxim.simulation.sim_logger import InteractiveMode as _IM
    from maxim.simulation.sim_logger import get_interactive_mode as _get_im

    _is_observe_only = _get_im() == _IM.ON and goal.strip().lower() in ("interactive", "interactive mode", "")
    # Build a MINIMAL tool registry with ONLY simulation tools.
    # Using build_tool_registry() adds filesystem/bash/code tools that
    # confuse the LLM — it picks familiar tools (glob, bash) instead of
    # simulation tools (send_message). A bare registry forces correct behavior.
    from maxim.simulation.tools import SimToolRegistry

    orch_registry = SimToolRegistry()
    spawn_tool = SpawnSubSimulationTool(
        llm_router=llm_router,
        stop_event=stop_event,
        parent_bridge=bridge,
        sim_tmpdir=str(sim_tmpdir),
        sandbox_dirs=sandbox_dirs,
    )
    _aut_embodiment = getattr(_aut_instance, "embodiment", None) if _aut_instance is not None else None
    orch_registry.register(SendMessageTool(bridge=bridge, embodiment=_aut_embodiment))
    orch_registry.register(ObserveActionsTool(bridge=bridge))
    orch_registry.register(CheckCompletionTool(bridge=bridge, llm=llm_router, goal=goal, continuous=continuous))
    orch_registry.register(AnalyzeResultsTool(bridge=bridge, llm=llm_router))
    orch_registry.register(InjectPainTool(bridge=bridge))
    # DamageComponentTool + SetEntitySensorTool: SEM damage/recovery → PainBus → NAc.
    # OrchestratorActorTool: scene entity affordances against the AUT —
    # writes target_effect deltas through the SEM mechanics path.
    # Only registered when embodiment is active (--embodiment flag).
    if _aut_embodiment is not None:
        orch_registry.register(OrchestratorActorTool(embodiment=_aut_embodiment, entity_map=_aut_entity_map))
        orch_registry.register(DamageComponentTool(embodiment=_aut_embodiment, entity_map=_aut_entity_map))
        orch_registry.register(SetEntitySensorTool(embodiment=_aut_embodiment, entity_map=_aut_entity_map))
    orch_registry.register(spawn_tool)
    orch_registry.register(ExtendSimulationTool(main_bridge=bridge, spawn_tool=spawn_tool))
    orch_registry.register(
        FinishSimulationTool(
            bridge=bridge,
            orchestrator_source=orchestrator_source,
            spawn_tool=spawn_tool,
            min_turns=0 if _is_observe_only else min_finish_turns,  # an observer may always end the session
        )
    )
    orch_registry.register(SimRespondTool())
    aut_introspector = Observer(
        hippocampus=aut_hippocampus,
        nac=aut_nac,
        memory_hub=aut_memory_hub,
        energy_registry=aut_energy_registry,
    )
    orch_registry.register(InspectAUTTool(introspector=aut_introspector))

    # Research tools — available in every mode (record_experiment is
    # useful for any systematic investigation, not just "researcher").
    # Experiment log lives in sim_tmpdir during the run; report.py
    # copies it to the final session directory at save time.
    # Use caller-provided ExperimentLog if given (research protocol passes
    # its log in so Writer/Reviewer read from the same instance — D-0a fix).
    if experiment_log is None:
        try:
            from maxim.simulation.research_tools import ExperimentLog

            experiment_log = ExperimentLog(session_dir=sim_tmpdir)
        except Exception as e:
            logger.debug("ExperimentLog not available: %s", e)
    try:
        from maxim.simulation.research_tools import (
            RecordExperimentTool,
            QueryExperimentsTool,
        )

        if experiment_log is not None:
            orch_registry.register(RecordExperimentTool(experiment_log))
            orch_registry.register(QueryExperimentsTool(experiment_log))
            logger.info("Research tools registered (record_experiment, query_experiments)")
    except Exception as e:
        logger.debug("Research tools not available: %s", e)

    # The narrator's own tool descriptions (the follow-up types that keep it acting after send_message, and its
    # 'respond' decoy's) ride on its registry, ``SimToolRegistry.tool_descriptions()``, never the process-global
    # ``TOOL_DESCRIPTIONS``, which the AUT running beside it reads too (#1260).

    # Autonomy: allow ALL tools through to the executor.
    # SimToolRegistry handles unknown tools by returning an error+redirect,
    # which triggers a followup LLM call. If we filter at the autonomy level,
    # unknown tools get silently dropped and the orchestrator stalls.
    orch_autonomy = AutonomyController(
        initial_level=AutonomyLevel.AUTONOMOUS,
        supervision_policy=SupervisionPolicy(
            allowed_tools=set(),  # Empty = allow all (SimToolRegistry redirects unknowns)
            forbidden_tools=set(),
            min_confidence_autonomous=0.3,
        ),
    )

    # Build orchestrator's LLM worker (shares the same router)
    orch_llm_worker: LLMWorker | None = None
    if llm_router is not None:
        orch_llm_worker = LLMWorker(
            llm=llm_router,
            stale_threshold_s=60.0,  # High threshold: orchestrator waits for shared LLM
            n_ctx=llm_router.n_ctx,
            token_counter=llm_router.get_token_counter(),
            session_id=session_id,
        )
        orch_llm_worker.start()

    # ── PainDetector (Phase 0b — bio-stack activation) ─────────────────
    aut_pain_detector = None
    try:
        from maxim.proprioception.pain import PainDetector

        aut_pain_detector = PainDetector(agent_id=_aut_agent_id)
        # Wire to introspector for benchmark_snapshot() pain_stats
        aut_introspector._pain_detector = aut_pain_detector
        logger.info("AUT PainDetector active in sim mode")
    except Exception as e:
        logger.debug("AUT PainDetector creation failed: %s", e)

    # ── DefaultNetwork (Phase 0b — bio-stack activation) ──────────────
    # Migrated to build_default_network per
    # docs/plans/archive/default_network_unification.md Gap C.
    # maxim=None — sim has no robot, but DN still provides pain detection.
    # pain_bus=aut_pain_bus — injected bus closes Gap B (split subscriber
    # ownership). The bus already has hippocampus + NAc subscribers from
    # build_pain_bus (line ~619/622 above).
    aut_default_network = None
    try:
        from maxim.default_network import DefaultNetworkConfig
        from maxim.runtime.bootstrap import build_default_network

        aut_default_network = build_default_network(
            nac=aut_nac,
            maxim=None,
            pain_bus=aut_pain_bus,
            config=DefaultNetworkConfig(
                enabled=True,
                publish_actions=False,  # no bus in sim
                fear_gate_enabled=False,  # FearAgent wired separately
            ),
        )
        if aut_default_network is not None:
            logger.info(
                "AUT DefaultNetwork active in sim mode (pain_bus=%s)", "injected" if aut_pain_bus else "internal"
            )
    except Exception as e:
        logger.debug("AUT DefaultNetwork creation failed: %s", e)

    # ── Orchestrator executor via AgentFactory (F3 migration) ───────────
    # The orch gets its own isolated bio-stack with a separate NAc for
    # future adaptive orchestration. Persistence at ~/.maxim/orchestrator/
    # (not CWD-relative — agent longevity review found fragmentation risk).
    # with_pain_bridge=False: orch sim tools (send_message, observe_actions,
    # etc.) never produce SEM embodiment pain. Bridge is still constructed
    # for direct attribution (gated on nac), just not subscribed to pain_bus.
    from maxim.utils.paths import data_home as _data_home

    _orch_config = AgentConfig(
        agent_id="sim_orchestrator",
        role="npc",
        persistence_dir=str(_data_home() / "orchestrator"),
        with_bio_stack=True,
        with_executor=True,
        with_pain_bridge=False,
        with_fear_gate=False,
        # Write-but-don't-read (review fold, Arch #2): shutdown() keeps
        # persisting orch state for the future "Phase 3" cross-session
        # orchestration work, but the orch must NOT restore it — reading
        # an unaudited, ever-accumulating ~/.maxim/orchestrator home into
        # every sim run would make the orchestrator/narrator a cross-run
        # confound (the narrator-state-confound class), and ~/.maxim is
        # shared across worktrees, so concurrent experiment sessions
        # would cross-contaminate through it. Flip to True only as part
        # of a deliberate, tuned Phase 3 (see the hippocampus NOTE at the
        # run_agentic_loop call below).
        load_persisted=False,
    )
    _orch_factory = AgentFactory(base_data_dir=_data_home() / "orchestrator")
    _orch_instance = _orch_factory.create_full_agent(
        _orch_config,
        tool_registry=orch_registry,
    )
    orch_executor = _orch_instance.executor

    # Wrap with PainInterceptor (Layer 2 — consequence pain after execute)
    # and AnticipatoryPainExecutor (Layer 1 — perceived pain before execute).
    # Together: perceived pain predicts, consequence pain confirms, NAc learns.
    aut_perceived_pain_assessor: Any = None
    try:
        from maxim.proprioception.perceived_pain import PerceivedPainAssessor
        from maxim.runtime.pain_interceptor import (
            AnticipatoryPainExecutor,
            PainInterceptorExecutor,
        )

        aut_executor = PainInterceptorExecutor(aut_executor, pain_bus=aut_pain_bus, agent_id=_aut_agent_id)
        aut_perceived_pain_assessor = PerceivedPainAssessor(nac=aut_nac, pain_bus=aut_pain_bus, agent_id=_aut_agent_id)
        aut_executor = AnticipatoryPainExecutor(
            aut_executor,
            assessor=aut_perceived_pain_assessor,
        )
        # Also wire percept-level anxiety: the AUT feels anticipatory
        # pain when it RECEIVES a message containing sensitive paths,
        # not just when it tries to act on them.
        bridge.percept_anxiety_hook = aut_perceived_pain_assessor.assess_text
        logger.info("AUT pain layers wired: action-anticipation (L1) + consequence (L2) + percept-anxiety (L1b)")
    except Exception as e:
        logger.warning("Failed to wire pain-layer executors: %s", e)

    # Wrap AUT executor with FearGatedExecutor for safety review.
    # NOTE: this MUST come after ToolPainBridge wiring so the bridge
    # lives on the inner executor where record_tool_start/complete
    # are actually invoked.
    try:
        from maxim.agents.fear_agent import FearAgent
        from maxim.runtime.fear_gate import FearGatedExecutor

        llm_for_fear = llm_router  # Share LLM for code analysis
        fear_agent = FearAgent(llm=llm_for_fear)
        aut_executor = FearGatedExecutor(aut_executor, fear_agent)
        logger.info("AUT FearGatedExecutor active — all tool calls reviewed by FearAgent")
    except Exception as e:
        logger.warning("Failed to wire FearGatedExecutor for AUT: %s", e)

    # NOTE: Pre-Wave-2, aut_memory_hub.connect(fear_agent=...) was called
    # here.  build_memory_hub now calls .connect() internally, so the
    # three always-created bridges (PlanHistory, Escalation, Fear) are
    # already alive.  fear_agent is stored by .connect() but never read
    # by any bridge — the FearCircuitBridge uses nac/ec/hippocampus
    # directly.  See memory_hub_unification.md audit.

    # ── Imagination trigger (I1+I2 wiring) ──────────────────────────────
    # When an entity_ref is present, the AUT has a ComponentRegistry.
    # Build a ComponentIndex + ImaginationTrigger so the AUT can imagine
    # novel entities it encounters during the simulation.
    # Universal imagination gate (MAXIM_DISABLE_IMAGINATION): the single master
    # switch for ALL imagination surfaces. ImaginationTrigger._enabled is
    # already respected by BOTH the per-turn design path (trigger.py::process,
    # ~L593) AND the world-builder manifest (trigger.py::process_manifest,
    # ~L870) — so gating the trigger at construction kills both with one flag,
    # and we ALSO skip the generate_scene_manifest LLM call below to avoid
    # burning a call whose result process_manifest would just discard.
    # Controlled experiments (Exp 44) set this so a controlled arc presents
    # ONLY its declared entities — no improvised distractors. Does NOT touch
    # the generative NARRATOR prose (a separate subsystem); use the fixture
    # path for full scene determinism. Experiment/harness toggle (env, not
    # config, per the carve-out). Truthy via the canonical parser.
    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env as _adve

    _imagination_off = _adve(os.environ.get("MAXIM_DISABLE_IMAGINATION"))

    aut_imagination_trigger: Any = None
    if aut_component_registry is not None:
        try:
            from maxim.embodiment.component_index import ComponentIndex
            from maxim.imagination.cache import ImaginationCache
            from maxim.imagination.designer import ImaginationDesigner
            from maxim.imagination.trigger import ImaginationTrigger
            from maxim.simulation.entity_designer import EntityDesigner

            _aut_component_index = ComponentIndex(aut_component_registry)
            _aut_entity_designer = EntityDesigner(
                component_registry=aut_component_registry,
                llm_router=llm_router,
            )
            _aut_imagination_designer = ImaginationDesigner(
                _aut_entity_designer,
                component_index=_aut_component_index,
            )
            # Resolve the LinguisticEncoder for affordance substrate encoding.
            # Lives in MemoryHub (wired by _wire_substrate_encoder).
            _aut_encoder = getattr(aut_memory_hub, "_encoder", None) if aut_memory_hub is not None else None

            aut_imagination_trigger = ImaginationTrigger(
                component_index=_aut_component_index,
                component_registry=aut_component_registry,
                designer=_aut_imagination_designer,
                cache=ImaginationCache(),
                tool_registry=aut_registry,
                default_network=aut_default_network,
                encoder=_aut_encoder,
                agent_id=_aut_agent_id,
                enabled=not _imagination_off,  # universal MAXIM_DISABLE_IMAGINATION gate
            )

            # Wire ComponentIndex + affordance transfer deps into SenseToolsTool
            _aut_nac = aut_memory_hub.nac if aut_memory_hub is not None else None
            _aut_atl = aut_memory_hub.atl if aut_memory_hub is not None else None
            try:
                _sense_tools = aut_registry.get("sense_tools")
                _sense_tools._component_index = _aut_component_index
                _sense_tools._atl = _aut_atl
                _sense_tools._agent_id = _aut_agent_id
                # NAc is already wired at construction if passed, but
                # the tool was constructed with nac=None — patch it now.
                if _sense_tools._nac is None and _aut_nac is not None:
                    _sense_tools._nac = _aut_nac
            except KeyError:
                pass  # sense_tools not registered (no embodiment)

            # Pass entity_map to ImaginationTrigger for populating on design
            if _aut_entity_map is not None:
                aut_imagination_trigger._entity_map = _aut_entity_map

            # Wire ImaginationTrigger + affordance transfer deps into SensePresenceTool
            try:
                _presence_tool = aut_registry.get("sense_presence")
                _presence_tool._imagination_trigger = aut_imagination_trigger
                _presence_tool._nac = _aut_nac
                _presence_tool._atl = _aut_atl
                _presence_tool._agent_id = _aut_agent_id
            except KeyError:
                pass

            # Wire ComponentIndex + ComponentRegistry + agent_id into BioEnrichmentPipeline
            if aut_bio_enrichment_pipeline is not None:
                aut_bio_enrichment_pipeline._component_index = _aut_component_index
                aut_bio_enrichment_pipeline._component_registry = aut_component_registry
                aut_bio_enrichment_pipeline._agent_id = _aut_agent_id

            # Wire percept reflex system into BioEnrichmentPipeline.
            # Reflexes detect keyword patterns in percept text and invoke
            # damage_component / set_entity_sensor BEFORE the LLM deliberates.
            # Replaces the auto-damage hack that was in SendMessageTool.
            if aut_bio_enrichment_pipeline is not None and _aut_embodiment is not None:
                try:
                    from maxim.embodiment.reflex import load_archetype_reflexes, ReflexRegistry

                    _archetype = None
                    if aut_component_registry is not None and entity_ref is not None:
                        try:
                            _spec = aut_component_registry.get(entity_ref)
                            _archetype = _spec.get("component", {}).get("archetype")
                        except Exception:
                            pass

                    if _archetype:
                        _reflex_specs = load_archetype_reflexes(_archetype)
                        if _reflex_specs:
                            # Closure must look up root at evaluation time,
                            # not capture it — root may change during sim
                            # (entity acquisition, imagination replacement).
                            _emb_ref = _aut_embodiment

                            def _get_component_integrity(name: str) -> float:
                                _root = _emb_ref.root if _emb_ref is not None else None
                                if _root is None:
                                    return 1.0
                                comp = _root.get_component(name)
                                if comp is not None and hasattr(comp, "compute_integrity"):
                                    return comp.compute_integrity()
                                return 1.0

                            _reflex_registry = ReflexRegistry(
                                _reflex_specs,
                                get_component_integrity=_get_component_integrity,
                            )
                            aut_bio_enrichment_pipeline._reflex_registry = _reflex_registry

                            # Wire tool instances for reflex dispatch
                            _reflex_damage = DamageComponentTool(embodiment=_aut_embodiment, entity_map=_aut_entity_map)
                            _reflex_sensor = SetEntitySensorTool(embodiment=_aut_embodiment, entity_map=_aut_entity_map)
                            aut_bio_enrichment_pipeline._reflex_damage_tool = _reflex_damage
                            aut_bio_enrichment_pipeline._reflex_sensor_tool = _reflex_sensor

                            # Wire entity root for latent affordance surfacing.
                            # Must look up at evaluation time (not capture) —
                            # same live-lookup pattern as integrity closure.
                            aut_bio_enrichment_pipeline._entity_root = _emb_ref.root

                            logger.info(
                                "Percept reflex system wired: archetype=%s, %d reflexes",
                                _archetype,
                                len(_reflex_specs),
                            )
                except Exception as e:
                    logger.warning("Reflex system wiring failed: %s", e)

            # Encode self-entity affordances through the substrate path.
            # The agent's own body affordances form substrate concepts just
            # like scene entities — enabling self-affordance concept formation.
            if _aut_encoder is not None and _aut_entity_map is not None:
                from maxim.imagination.trigger import encode_entity_affordances

                for _self_entity in _aut_entity_map.list_self_entities():
                    encode_entity_affordances(_self_entity, _aut_encoder, agent_id=_aut_agent_id)

            logger.info("AUT ImaginationTrigger wired (ComponentIndex + EntityDesigner + DN arousal gate)")
            try:
                from maxim.simulation.sim_logger import sim_log

                sim_log("SEM_TRACE", f"ImaginationTrigger ACTIVE — enabled={aut_imagination_trigger._enabled}")
            except Exception:
                pass
        except Exception as e:
            logger.warning("ImaginationTrigger construction failed: %s", e)
            try:
                from maxim.simulation.sim_logger import sim_log

                sim_log("SEM_TRACE", f"ImaginationTrigger FAILED: {e}")
            except Exception:
                pass

    # ── Scene manifest pre-trigger (Phase 1: SEM World Enrichment) ───────
    # Before the first turn, generate a scene entity inventory from the
    # goal and resolve all entities through the imagination pipeline.
    # This runs AFTER ImaginationTrigger wiring (EntityMap, ComponentIndex,
    # designer all constructed) and BEFORE the AUT thread launch.
    #
    # Fix B fold (2026-05-27, pre-merge review BLOCK 1, cross-confirmed by
    # both review lenses): gate on ``fixture_path is None`` so the generative-
    # narrator pre-trigger does NOT double-fire alongside Fix B's fixture
    # pre-trigger at ``FixtureDrivenOrchestrator._substrate_pretrigger``.
    # Pre-fold, BOTH hookups fired when fixture_path was set — burning two
    # manifest LLM calls per fixture run AND polluting the JSONL trace with
    # both ``scene_id="pre_trigger"`` and ``scene_id="fixture_pretrigger"``.
    # The two call sites now have disjoint conditions (this one handles the
    # generative path; FixtureDrivenOrchestrator handles the fixture path).
    # Skip the world-builder manifest LLM call when imagination is gated off
    # (process_manifest would early-return on _enabled anyway; this also avoids
    # burning the manifest LLM call). See the MAXIM_DISABLE_IMAGINATION note at
    # the ImaginationTrigger construction above.
    if fixture_path is None and aut_imagination_trigger is not None and llm_router is not None and _imagination_off:
        from maxim.simulation.sim_logger import sim_log

        sim_log("SEM_TRACE", "Scene manifest pre-trigger: skipped via MAXIM_DISABLE_IMAGINATION (universal gate)")
    elif fixture_path is None and aut_imagination_trigger is not None and llm_router is not None:
        try:
            from maxim.simulation.narrator import generate_scene_manifest
            from maxim.simulation.sim_logger import sim_log

            # W2 Hookup 1 — surface substrate-acquired tool biases to the
            # manifest LLM so the generated entity list can include entities
            # that activate substrate-favored tools (closes the imagination-
            # blindness gap surfaced by Roy-3a-retry). See
            # docs/plans/deferred/imagination_substrate_signals.md. Env-var gate
            # MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL preserves ablation-
            # evidence integrity for Roy iterations (parallel to Wire-A's
            # MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION); shared truthy-parser.
            _aut_nac_biases: list[tuple[str, float]] | None = None
            if aut_nac is not None:
                from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

                if not annotation_disabled_via_env(os.environ.get("MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL")):
                    # agent_id comes from the function-level _aut_agent_id —
                    # the single canonical binding (adopted persistent id or
                    # "sim_aut"), validated against MemoryHub.agent_id at
                    # adoption so this lookup can't silently mis-key.
                    try:
                        _aut_nac_biases = aut_nac.get_agent_tool_biases(
                            agent_id=_aut_agent_id,
                            top_n=5,
                        )
                    except ValueError as e:
                        logger.warning(
                            "W2 substrate-aware manifest skipped due to invalid agent_id (%s)",
                            e,
                        )
                        _aut_nac_biases = None

            logger.info("Generating scene manifest for SEM world enrichment...")
            manifest_text = generate_scene_manifest(
                llm_router,
                goal,
                nac_top_biases=_aut_nac_biases,
            )
            if manifest_text:
                manifest_results = aut_imagination_trigger.process_manifest(
                    manifest_text,
                    scene_id="pre_trigger",
                )
                logger.info(
                    "Scene manifest: %d entities pre-instantiated",
                    len(manifest_results),
                )
                sim_log(
                    "SEM_TRACE",
                    f"Scene manifest pre-trigger: {len(manifest_results)} entities resolved",
                )
            else:
                sim_log("SEM_TRACE", "Scene manifest: empty (goal too vague or LLM failed)")
        except Exception as e:
            logger.warning("Scene manifest pre-trigger failed: %s", e)

    # ── Print simulation banner ──────────────────────────────────────────
    from maxim.simulation.sim_logger import _emit, display_status, display_summary, get_active_display

    display_status(f"SIMULATION MODE — {mode}")
    display_status(f"Goal: {goal}")
    display_status(f"Max turns: {max_turns}")
    if sim_sandbox and not no_sim_env:
        display_status("Environment: simulated filesystem with pain triggers")
    display_status("Commands: /cancel  /new <goal>  /status  /report  /help")

    # Set the scene header based on the goal
    display = get_active_display()
    if display is not None:
        display.set_scene(
            title=goal[:60],
            description=f"{mode} | max {max_turns} turns",
        )

    # ── Substrate telemetry (Phase 0) ────────────────────────────────────
    # When --research is set with --aut-mode substrate-primary, write
    # per-tick EC/NAc/drive snapshots to a JSONL alongside the regular
    # sim trace. See docs/plans/grounded_language_acquisition.md Phase 0.
    aut_substrate_telemetry = None
    if research_telemetry and aut_mode == "substrate-primary":
        try:
            from maxim.simulation.substrate_telemetry import SubstrateTelemetry

            _telem_path = sim_workspace / f"substrate_telemetry_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            aut_substrate_telemetry = SubstrateTelemetry(
                log_path=str(_telem_path),
                agent_id=getattr(aut_memory_hub, "agent_id", None) or "sim_aut",
            )
            logger.info("Phase 0 substrate telemetry: %s", _telem_path)
        except Exception as e:
            logger.warning("Failed to enable substrate telemetry: %s", e)

    # ── Start AUT thread ─────────────────────────────────────────────────
    aut_error: list[Exception] = []

    def _aut_worker() -> None:
        # Stage 0b (release_0_9_1.md): bind RequestContext on the AUT
        # thread via context_scope() so InstrumentedExecutor.execute(),
        # recommend_action's sim_recommend_action emitter, and any
        # other downstream code reading utils/http.py::current_context()
        # see the right agent_id + session_id pair. ContextVars are
        # per-thread; the main-thread binding doesn't reach here
        # without copy_context. context_scope is a context manager —
        # its __exit__ resets the binding even on exception, which is
        # what the pre-merge review (architecture lens I5) recommended
        # over manual set_context/reset_context in try/finally so
        # future sim entry points cannot forget the reset.
        # Bound BEFORE sim_agent_context so the typed RequestContext
        # and the sim_logger contextvar agree on agent identity.
        #
        # NOTE: this binding is correct for the current sim topology
        # (one AUT, one orch, no AgentFactory sub-agents in the sim
        # path). If NPCs spawned via AgentFactory start producing
        # ActionRecords in this orchestrator session, every record
        # will carry agent_id="sim_aut" instead of the NPC's per-agent
        # stash id. Bio-lens nice-to-have: per-spawn context_scope
        # inside the NPC's tool-dispatch boundary would be the fix
        # when that surface ships.
        from maxim.utils.http import context_scope, new_request_context

        try:
            with context_scope(new_request_context(agent_id=_aut_agent_id, session_id=session_id)):
                with sim_agent_context(_aut_agent_id):
                    # Thalamic-relay stage 4 (thalamus_relay_design_pass.md):
                    # opt-in audio-orient channel. When MAXIM_SIM_AUDIO_ORIENT is
                    # set, multiplex a synthetic-DoA AzimuthDoASource into the AUT
                    # percept stream via the first-slice composite; the bridge
                    # keeps injecting into (and finishing) the underlying
                    # ConversationalSource child, so termination is unchanged.
                    # Default OFF → byte-identical (bare bridge.percept_source).
                    from maxim.simulation.audio_orient_wiring import audio_orient_enabled

                    _aut_percept_source = bridge.percept_source
                    if audio_orient_enabled():
                        from maxim.simulation.audio_orient_wiring import (
                            build_audio_composite,
                            default_sim_doa_reader,
                            sim_audio_salience_novelty,
                        )

                        _aud_sal, _aud_nov = sim_audio_salience_novelty()
                        _aut_percept_source = build_audio_composite(
                            bridge.percept_source,
                            default_sim_doa_reader(),
                            agent_id=_aut_agent_id,
                            salience=_aud_sal,
                            novelty=_aud_nov,
                        )
                        logger.info(
                            "audio-orient channel attached to AUT percept source "
                            "(synthetic DoA reader, salience=%.2f novelty=%.2f)",
                            _aud_sal,
                            _aud_nov,
                        )
                    run_agentic_loop(
                        aut_agent,
                        aut_env,
                        aut_state,
                        aut_memory,
                        aut_decision_engine,
                        aut_executor,
                        autonomy_controller=aut_autonomy,
                        llm_worker=aut_llm_worker,
                        default_network=aut_default_network,
                        hippocampus=aut_hippocampus,
                        memory_hub=aut_memory_hub,
                        max_steps=0,  # unlimited — AUT stops when bridge.finish() is called
                        stop_event=stop_event,
                        target_hz=2.0,
                        percept_source=_aut_percept_source,
                        action_sink=bridge.action_sink,
                        pain_bus=aut_pain_bus,
                        imagination_trigger=aut_imagination_trigger,
                        bio_enrichment_pipeline=aut_bio_enrichment_pipeline,
                        thought_gate=_aut_thought_gate,
                        aut_mode=aut_mode,
                        substrate_telemetry=aut_substrate_telemetry,
                        # Turn-scoped action budget (S6 apparatus bound) —
                        # only meaningful in substrate-primary; the gate is
                        # a no-op passthrough when the env is unset because
                        # the bridge was constructed without a budget.
                        substrate_action_gate=(
                            bridge.substrate_action_allowed if aut_mode == "substrate-primary" else None
                        ),
                        # HANDLE seam (a), branch 5: a persistent agent must
                        # NOT get the sim default ("lightweight") — its
                        # session-end consolidation is explicit "full" (#427).
                        consolidation="full" if persistent_agent is not None else None,
                    )
        except Exception as e:
            aut_error.append(e)
            logger.error("AUT loop failed: %s", e)
            # The AUT thread is DEAD from here on. Pre-2026-08-19 `aut_error`
            # was write-only (appended, never read), so the orchestrator kept
            # probing a corpse: `send_and_wait` increments `turn_count` BEFORE
            # waiting, so turn progress kept resetting the hard-abort's clock
            # and the campaign ran to full length emitting empty turns — a
            # plausible-looking low-action RESULT rather than a failure
            # (pre-merge review, architecture lens B1 / executor lens F4).
            # Surface it the same way every other fatal sim condition is
            # surfaced, so a dead AUT can never be mistaken for data.
            try:
                if not bridge.finish_context:
                    bridge.finish_context.update(
                        {
                            "status": "aut_died",
                            "reason": f"{type(e).__name__}: {e}",
                            "summary": (
                                f"The agent-under-test's loop raised {type(e).__name__} and its "
                                f"thread exited; every later turn would be empty. Terminating "
                                f"instead of reporting turns nothing produced."
                            ),
                            "initiated_by": "aut_worker",
                        }
                    )
            except Exception as ctx_err:
                logger.debug("finish_context update failed: %s", ctx_err)
            try:
                bridge.finish()
            except Exception as fin_err:
                logger.debug("bridge.finish failed: %s", fin_err)
            stop_event.set()

    aut_thread = threading.Thread(target=_aut_worker, name="sim.aut", daemon=True)
    aut_thread.start()

    # ── Generative campaign mode — narrator drives multi-turn story ──────
    if generative:
        from maxim.simulation.campaign_runner import run_generative_campaign as _run_gen
        from maxim.utils.paths import sim_reports as _gen_reports_dir

        # Scene-embodiment wiring (substrate_primary_cradle_readiness.md Phase A/B4):
        # thread the AUT's embodiment + entity_map into per-phase scene-entity
        # activation so scene-affordance self_effect writes DETERMINISTICALLY to
        # the agent's body (warmth_alpha_harm's arms.thermal breach, warmth_beta_
        # safe's clean cold relief).
        #
        # Default: ON for substrate-primary only. In LLM-primary the default keeps
        # embodiment=None because scene harm is expected via the narrator-driven
        # Layer-2 proximity path — threading it there would DOUBLE-COUNT and change
        # Exp 37/38 (which rely on the narrator path). Exp 37/38 do NOT set the
        # override flag, so their behavior stays byte-identical.
        #
        # G1 (controlled_llm_primary_embodied_harness.md): MAXIM_DETERMINISTIC_
        # SCENE_EMBODIMENT forces embodiment threading in LLM-primary too, so a
        # controlled experiment (Exp 44) gets deterministic, attributable safe-vs-
        # harm signal instead of narrator-improvised harm. KNOWN follow-up: with
        # this ON in LLM-primary the narrator-proximity path could still add harm
        # (double-count); validate empirically whether the narrator materially
        # re-applies harm and, if so, suppress that channel under the same flag
        # (tracked in the plan). CC8 caveat: a future NON-sim adapter must own its
        # own scene-harm wiring.
        from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env as _det_scene_flag

        _deterministic_scene_embodiment = aut_mode == "substrate-primary" or _det_scene_flag(
            os.environ.get("MAXIM_DETERMINISTIC_SCENE_EMBODIMENT")
        )
        _gen_embodiment = _aut_instance.embodiment if _deterministic_scene_embodiment else None
        _gen_entity_map = _aut_entity_map if _deterministic_scene_embodiment else None
        # cradle_mother operant credit: thread the AUT's NAc + its canonical
        # agent_id so the reactive mother's feed reinforces the AUT's OWN orient
        # action (credit_operant_reward). agent_id MUST match the substrate loop's
        # ``_loop_agent_id = memory_hub.agent_id or agent_name`` or the credit
        # lands on a different key than the pending action — smoke-verified via
        # the ``credited=True`` mother telemetry.
        _gen_agent_id = (getattr(aut_memory_hub, "agent_id", "") if aut_memory_hub is not None else "") or ""
        gen_result = _run_gen(
            goal=goal,
            bridge=bridge,
            llm_router=llm_router,
            arc_yaml=arc_yaml,
            max_turns=max_turns,
            tool_registry=aut_registry,
            session_dir_base=str(_gen_reports_dir() / time.strftime("%Y%m%d_%H%M%S")),
            embodiment=_gen_embodiment,
            entity_map=_gen_entity_map,
            nac=aut_nac,
            agent_id=_gen_agent_id,
        )
        _finish_runner(bridge, stop_event, "generative_runner", gen_result)

    # ── DM Campaign mode — DM runtime drives encounters ────────────────────
    # In interactive mode, the DM campaign runs on its own thread so the
    # main thread can start the stdin reader first.  Without this, the
    # DM's SimPromptHandler.prompt() blocks waiting for deliver_response()
    # from a stdin reader that hasn't started yet.
    dm_rollup: dict[str, Any] = {}
    _dm_thread: threading.Thread | None = None
    _dm_error: list[Exception] = []
    if dm_campaign is not None:
        from maxim.simulation.campaign_runner import run_dm_campaign as _run_dm

        _dm_kwargs = dict(
            dm_campaign=dm_campaign,
            bridge=bridge,
            llm_router=llm_router,
            aut_registry=aut_registry,
            aut_executor=aut_executor,
            aut_hippocampus=aut_hippocampus,
            aut_nac=aut_nac,
            aut_memory_hub=aut_memory_hub,
            aut_pain_bus=aut_pain_bus,
            prompt_handler=_sim_prompt_handler,
            interactive=_sim_prompt_handler is not None,
        )

        if _sim_prompt_handler is not None:
            # Interactive: run on thread so stdin reader starts first
            def _dm_worker() -> None:
                try:
                    nonlocal dm_rollup
                    dm_rollup = _run_dm(**_dm_kwargs)
                except Exception as e:
                    _dm_error.append(e)
                finally:
                    _finish_runner(bridge, stop_event, "dm_runner", dm_rollup, errors=_dm_error)

            _dm_thread = threading.Thread(target=_dm_worker, name="sim.dm", daemon=True)
            # Thread started AFTER stdin reader — see below
        else:
            # Non-interactive: run synchronously (original behavior)
            dm_rollup = _run_dm(**_dm_kwargs)
            _finish_runner(bridge, stop_event, "dm_runner", dm_rollup)

    # ── Pre-campaign turn delivery ────────────────────────────────────────
    campaign_analysis: dict[str, Any] = {}
    if pre_campaign_turns:
        from maxim.simulation.campaign_runner import run_precampaign_turns as _run_pre

        campaign_analysis = _run_pre(
            turns=pre_campaign_turns,
            bridge=bridge,
            introspector=aut_introspector,
        )
        _finish_runner(bridge, stop_event, "precampaign_runner", campaign_analysis)

    # ── Fixture-driven campaign (S1) — no narrator LLM required ──────────
    # Fix B (2026-05-27, docs/plans/deferred/imagination_substrate_signals.md): the
    # substrate-aware pre-trigger threads aut_imagination_trigger + llm_router
    # + goal so fixture-driven test arms (Roy's roy_1_holdout etc.) materialize
    # substrate-favored entities into scene BEFORE driving percepts. Closes
    # exp 32's Bug B (W2 hookup was structurally bypassed in this path).
    fixture_result: dict[str, Any] = {}
    if fixture_path is not None:
        from maxim.simulation.campaign_runner import run_fixture_campaign as _run_fix

        fixture_result = _run_fix(
            fixture_path=Path(fixture_path),
            bridge=bridge,
            aut_hippocampus=aut_hippocampus,
            aut_nac=aut_nac,
            aut_memory_hub=aut_memory_hub,
            aut_pain_bus=aut_pain_bus,
            aut_imagination_trigger=aut_imagination_trigger,
            llm_router=llm_router,
            goal=goal,
        )
        _finish_runner(bridge, stop_event, "fixture_runner", fixture_result)

    # ── Inject initial goal (or resume context) into orchestrator ────────
    _tools_block = _narrator_tools_block(orch_registry)
    resume_data = None
    if resume_session:
        resume_data, resume_dir = _load_resume_context_at(resume_session)
        resume_record.update(context_dir=str(resume_dir) if resume_dir else None, context_loaded=bool(resume_data))
        if not resume_data:
            logger.warning("Resume session '%s' not found, starting fresh", resume_session)
    if resume_data:
        resume_prompt = _build_resume_prompt(
            resume_data, goal, mode, observe_only=_is_observe_only, tools_block=_tools_block
        )
        orchestrator_source.inject_cli(resume_prompt, salience=1.0, novelty=1.0)
        display_status(f"Resuming session: {resume_session}")
        display_status(
            f"Previous turns: {resume_data.get('turns', '?')}, actions: {resume_data.get('total_actions', '?')}"
        )
    else:  # a fresh start, or a resume whose session was not found
        orchestrator_source.inject_cli(
            _build_kickoff_prompt(goal, tools_block=_tools_block, observe_only=_is_observe_only),
            salience=1.0,
            novelty=1.0,
        )

    # ── Start stdin reader thread ────────────────────────────────────────
    from maxim.simulation.sim_logger import get_interactive_mode, InteractiveMode

    _is_interactive = get_interactive_mode() == InteractiveMode.ON

    _paused = [False]  # Mutable container for closure

    def _process_line(line: str) -> bool:
        """Process a completed line of user input. Returns True to break the loop."""
        line = line.strip()
        if not line:
            return False

        # Slash commands always take priority, even during a pending prompt.
        if line.lower() in ("/cancel", "/stop", "/quit"):
            display_summary(["Simulation cancelled by user."])
            _stop_by_operator(stop_event, operator_stop)
            return True
        elif line.lower() == "/pause":
            if not _paused[0]:
                _paused[0] = True
                orchestrator_source.inject_cli(
                    "SYSTEM: User paused the simulation. STOP all probing. "
                    "Do NOT call send_message or any other tool. "
                    "Wait silently until the user resumes.",
                    salience=1.0,
                    novelty=1.0,
                )
                display = get_active_display()
                if display is not None:
                    display.set_status(status="PAUSED")
                _emit("Simulation paused — type to talk to the agent, /resume to continue", "turn")
        elif line.lower() == "/resume":
            if _paused[0]:
                _paused[0] = False
                orchestrator_source.inject_cli(
                    "SYSTEM: User resumed the simulation. Continue probing.",
                    salience=1.0,
                    novelty=0.8,
                )
                display = get_active_display()
                if display is not None:
                    display.set_status(status="")
                _emit("Simulation resumed", "turn")
        elif line.lower().startswith("/new "):
            new_goal = line[5:].strip()
            if new_goal:
                orchestrator_source.inject_cli(
                    f"NEW SIMULATION GOAL: {new_goal}\n"
                    f"Switch to testing this new objective. "
                    f"Previous findings are still in your context.",
                    salience=1.0,
                    novelty=1.0,
                )
                display_status(f"New goal: {new_goal}")
        elif line.lower() == "/status":
            display_status(f"Turns: {bridge.turn_count}")
            display_status(f"Actions: {len(bridge.get_all_actions())}")
            blocked = [a for a in bridge.get_all_actions() if a.blocked]
            display_status(f"Blocked: {len(blocked)}")
        elif line.lower() == "/report":
            orchestrator_source.inject_cli(
                "Generate an interim report of your findings so far "
                "using analyze_results, then present it with respond.",
                salience=0.9,
                novelty=0.5,
            )
            display_status("Report requested...")
        elif line.lower().startswith("/display "):
            tier_name = line[9:].strip().lower()
            if tier_name in ("clean", "bio", "debug"):
                from maxim.simulation.sim_logger import set_display_tier

                set_display_tier(tier_name)
                _emit(f"Display switched to: {tier_name}", "turn")
            else:
                _emit("Usage: /display clean|bio|debug", "turn")
        elif line.lower().startswith("/focus"):
            # Portable agent-focus command — Shift+arrows are stripped on
            # macOS Terminal.app and several other terminals, so a slash
            # command guarantees focus switching always works.  Accepts
            # an explicit nickname, ``all`` to clear, ``next``/``prev``
            # to cycle, or no argument to show the roster.
            arg = line[len("/focus") :].strip()
            display_obj = get_active_display()
            if display_obj is None:
                _emit("/focus requires interactive display", "turn")
            elif not arg:
                roster = getattr(display_obj, "agent_roster", []) or []
                cur = getattr(display_obj, "focused_agent", None)
                _emit(f"Focused: {cur or 'ALL'}", "turn")
                if roster:
                    _emit("Agents: ALL, " + ", ".join(roster), "turn")
                else:
                    _emit("No agents registered yet", "turn")
            elif arg.lower() == "next":
                display_obj.focus_next()
                _emit(f"Focused: {getattr(display_obj, 'focused_agent', None) or 'ALL'}", "turn")
            elif arg.lower() == "prev":
                display_obj.focus_prev()
                _emit(f"Focused: {getattr(display_obj, 'focused_agent', None) or 'ALL'}", "turn")
            elif hasattr(display_obj, "focus_agent"):
                ok = display_obj.focus_agent(arg)
                if ok:
                    _emit(f"Focused: {arg if arg.lower() != 'all' else 'ALL'}", "turn")
                else:
                    roster = getattr(display_obj, "agent_roster", []) or []
                    _emit(f"Unknown agent {arg!r}. Known: ALL, {', '.join(roster)}", "turn")
            else:
                _emit("Usage: /focus [<name>|all|next|prev]", "turn")
        elif line.lower() == "/help":
            # macOS Terminal.app strips Shift+arrow modifiers before
            # delivery, so the Shift+Left/Right keybinding silently
            # doesn't fire there.  Detect the platform and render a
            # platform-correct hint, but always advertise /focus as the
            # portable fallback.
            import sys as _sys

            _shift_arrows_unreliable = _sys.platform == "darwin" and os.environ.get("TERM_PROGRAM") in (
                "Apple_Terminal",
                None,
            )
            _emit("--- Keybindings ---", "info")
            _emit("  Up/Down         Scroll log (or thinking panel when expanded)", "info")
            _emit("  Left            Page up in log", "info")
            _emit("  Right           Jump to bottom / collapse thinking panel", "info")
            if _shift_arrows_unreliable:
                _emit(
                    "  Shift+Left/Right  Cycle agent focus — NOT reliable in macOS Terminal.app; use /focus instead",
                    "info",
                )
            else:
                _emit("  Shift+Left/Right  Cycle agent focus (or use /focus)", "info")
            _emit("  Option+=        Expand thinking panel (arrows switch to it)", "info")
            _emit("  Option+-        Shrink thinking panel (arrows revert to log)", "info")
            _emit("--- Commands ---", "info")
            _emit("  /cancel         End simulation", "info")
            _emit("  /new <goal>     Start new scenario", "info")
            _emit("  /pause /resume  Pause/resume orchestrator", "info")
            _emit("  /status         Show pipeline state", "info")
            _emit("  /report         Request interim analysis", "info")
            _emit("  /display <tier> Switch display (clean|bio|debug)", "info")
            _emit("  /focus [name]   Focus agent log filter (portable; works on macOS)", "info")
            _emit("  /help           Show this help", "info")
        else:
            # If the agent is waiting for user input via request_interaction,
            # forward non-command text to the prompt handler.
            if _sim_prompt_handler is not None and _sim_prompt_handler.has_pending_prompt:
                _sim_prompt_handler.deliver_response(line)
                _emit(f"You: {line}", "user")
            # When interactive or paused, free text goes directly
            # to the AUT as a user percept (conversational input).
            elif _is_interactive or _paused[0]:
                bridge.percept_source.inject_cli(
                    line,
                    salience=0.9,
                    novelty=0.8,
                )
                _emit(f"You: {line}", "user")
            else:
                orchestrator_source.inject_cli(
                    f"USER GUIDANCE: {line}",
                    salience=0.7,
                    novelty=0.6,
                )
        return False

    def _stdin_reader_line() -> None:
        """Fallback line-based stdin reader for non-TTY environments."""
        while not stop_event.is_set():
            # Mirror the raw reader's prompt handler check: if an agent
            # tool is waiting for user input via SimPromptHandler, route
            # the next line to deliver_response() instead of _process_line().
            if _sim_prompt_handler is not None and _sim_prompt_handler.has_pending_prompt:
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    _stop_by_operator(stop_event, operator_stop)
                    return
                _sim_prompt_handler.deliver_response(line.strip())
                continue
            try:
                line = input()
            except EOFError:
                return
            except KeyboardInterrupt:
                _stop_by_operator(stop_event, operator_stop)
                break
            if _process_line(line):
                break

    _stdin_target = (
        functools.partial(
            stdin_reader_raw,
            stop_event=stop_event,
            operator_stop=operator_stop,
            _sim_prompt_handler=_sim_prompt_handler,
            _process_line=_process_line,
            _stdin_reader_line=_stdin_reader_line,
            _stop_by_operator=_stop_by_operator,
            get_active_display=get_active_display,
        )
        if _is_interactive
        else _stdin_reader_line
    )
    stdin_thread = threading.Thread(target=_stdin_target, name="sim.stdin", daemon=True)
    stdin_thread.start()

    # Start the DM campaign thread AFTER the stdin reader is running
    # so SimPromptHandler.prompt() can receive deliver_response() calls.
    if _dm_thread is not None:
        _dm_thread.start()

    # Show input hint when interactive mode is on
    if _is_interactive:
        display = get_active_display()
        if display is not None:
            display.set_prompt("Type to talk to the agent (Enter to send) | /cancel to stop")
        _emit(
            "Interactive mode: type to talk to the agent | arrows scroll | Shift+arrows switch agent | Option+/-  resize | /help for more",
            "turn",
        )

    # ── Stall detector (§11): nudges the narrator when it idles or ping-pongs, ends a run out of turns or wedged ──
    from maxim.simulation.stall_detector import StallDetector

    stall_detector = StallDetector(
        stop_event=stop_event,
        bridge=bridge,
        orchestrator_source=orchestrator_source,
        orch_executor=orch_executor,
        goal=goal,
        max_turns=max_turns,
        is_interactive=_is_interactive,
        is_observe_only=_is_observe_only,
        paused=_paused,
        sim_prompt_handler=_sim_prompt_handler,
        display_status=display_status,
        emit=_emit,
        get_active_display=get_active_display,
    )
    stall_thread = threading.Thread(target=stall_detector.run, name="sim.stall", daemon=True)
    stall_thread.start()

    # ── Run orchestrator loop (blocks until done or /cancel) ─────────────
    orch_error: list[Exception] = []
    # ── Orchestrator spinner (between turns) ────────────────────────────
    # The bridge spinner shows progress during send_and_wait(). Between
    # turns, show "Orchestrator planning..." using the bridge's spinner
    # so there's only one spinner managing the terminal line.
    bridge._spinner.start("Orchestrator planning first probe...")
    bridge._spinner.set_planning_window(True)  # D14 (pre-first-turn window)

    # Stage 0b: bind RequestContext on the orchestrator thread so
    # orch-side action records (rare — most actions land on the AUT
    # sink, but orchestrator tools still execute) carry agent_id +
    # session_id. Symmetric with the AUT thread binding above. Runs
    # on the main thread; context_scope's __exit__ resets the bind
    # on normal return AND on exception, so the bind is scoped to
    # exactly the run_agentic_loop window. Per pre-merge review
    # architecture lens I5, this replaces a manual set_context /
    # reset_context try/finally with the canonical helper.
    from maxim.runtime.loop_controller import PlanningLivenessExhausted as _PlanningLivenessExhausted
    from maxim.utils.http import context_scope, new_request_context

    try:
        with context_scope(new_request_context(agent_id="sim_orchestrator", session_id=session_id)):
            with sim_agent_context("sim_orchestrator"):
                run_agentic_loop(
                    orch_agent,
                    orch_env,
                    orch_state,
                    orch_memory,
                    orch_decision_engine,
                    orch_executor,
                    autonomy_controller=orch_autonomy,
                    llm_worker=orch_llm_worker,
                    # NOTE: orchestrator hippocampus disabled for now — it captures
                    # every tool call as an episodic memory, which is noisy.
                    # Re-enable when cross-session learning (Phase 3) is tuned.
                    # hippocampus=orch_hippocampus,
                    # memory_hub=orch_memory_hub,
                    max_steps=0,  # unlimited — stops via FinishSimulationTool or /cancel
                    stop_event=stop_event,
                    target_hz=2.0,
                    percept_source=orchestrator_source,
                    # Bugs ledger D13. THIS loop is the one that needs it: it has
                    # no action_sink and no percept producer between turns, so a
                    # planning turn dropped after a parse failure leaves nothing
                    # to wake it — the observed 8,624s "planning" hang. The AUT
                    # deliberately does NOT opt in (it already recovers via the
                    # `_llm_unavailable` synthetic action breaking the bridge's
                    # settle loop); enabling it there would turn a working
                    # recovery into a dead thread whose campaign keeps emitting
                    # empty turns (pre-merge review, architecture lens B1/B3).
                    planning_liveness=True,
                )
    except KeyboardInterrupt:
        display_summary(["Simulation stopped by user"])
        _stop_by_operator(stop_event, operator_stop)
    except _PlanningLivenessExhausted as e:
        # D13: the agent loop spent its bounded planning-retry budget and
        # aborted through its normal teardown. Preserve the typed failure:
        # provider/parse exhaustion is ``llm_wedged``, responsive but invalid
        # planning is ``planning_failed``, and queue/worker exhaustion is
        # ``worker_unavailable``.
        _finish_status = getattr(e, "finish_status", "llm_wedged")
        _liveness_msg = f"PLANNING LIVENESS EXHAUSTED [{_finish_status}] (bugs ledger D13): {e}"
        logger.error(_liveness_msg)
        _emit(f"🛑 {_liveness_msg}", "turn")
        try:
            bridge._spinner.update(f"🛑 {_finish_status} — aborting sim")
        except Exception as spin_err:
            logger.debug("spinner abort update failed: %s", spin_err)
        if not bridge.finish_context:
            bridge.finish_context.update(
                {
                    "status": _finish_status,
                    "reason": str(e),
                    "summary": _liveness_msg,
                    "initiated_by": "planning_liveness",
                }
            )
    except Exception as e:
        orch_error.append(e)
        logger.error("Orchestrator loop failed: %s", e)
    finally:
        # Always clean up, even on interrupt
        bridge._spinner.stop()
    _tail = SimTail(
        is_interactive=_is_interactive,
        stop_event=stop_event,
        bridge=bridge,
        orchestrator_source=orchestrator_source,
        aut_thread=aut_thread,
        adopted=_adopted,
        dm_thread=_dm_thread,
        dm_error=_dm_error,
        aut_llm_worker=aut_llm_worker,
        orch_llm_worker=orch_llm_worker,
        orch_instance=_orch_instance,
        aut_registry=aut_registry,
        aut_agent_id=_aut_agent_id,
        aut_default_network=aut_default_network,
        aut_pain_bus=aut_pain_bus,
        sim_sandbox=sim_sandbox,
        aut_imagination_trigger=aut_imagination_trigger,
        aut_nac=aut_nac,
        aut_component_registry=aut_component_registry,
        start_time=start_time,
        orch_error=orch_error,
        operator_stop=operator_stop,
        goal=goal,
        mode=mode,
        aut_hippocampus=aut_hippocampus,
        aut_memory_hub=aut_memory_hub,
        llm_router=llm_router,
        session_id=session_id,
        start_provenance=start_provenance,
        resume_record=resume_record,
        aut_router=aut_router,
        fixture_result=fixture_result,
        persistent_agent=persistent_agent,
        experiment_log=experiment_log,
        sim_tmpdir=sim_tmpdir,
        aut_executor=aut_executor,
        aut_introspector=aut_introspector,
        pre_campaign_turns=pre_campaign_turns,
        campaign_analysis=campaign_analysis,
        dm_rollup=dm_rollup,
    )
    report, report_dir, duration, finish_reason = end_run(_tail)

    if _is_interactive:
        display = _show_report_and_post_sim_prompt(report, report_dir)

        # Wait for user input: a goal (continue) or Enter (finish).
        # Live is still running — arrow keys scroll the report.
        _new_goal = ""
        _stdin = __import__("sys").stdin
        _can_raw = _stdin is not None and hasattr(_stdin, "isatty") and _stdin.isatty()
        if _can_raw:
            _new_goal = read_post_sim_goal_raw(_stdin=_stdin, display=display)

        if display is not None:
            display.set_urgent(False)
            display.stop()

        if _new_goal and _new_goal.lower() not in ("/cancel", "quit", "exit"):
            _await_aut_session_end_before_new(aut_memory_hub)
            # NOTE: This recurses. Depth is bounded by human interaction
            # (each level requires typing a new goal + running a full sim).
            # RecursionError after ~50-100 is theoretically possible but
            # extremely unlikely in practice.
            # persistent_agent threads through: dropping it would silently
            # rebuild a throwaway sim_aut for the follow-up goal — the exact
            # mis-route this seam kills. Re-adoption re-snapshots the lease
            # (the outer restore already ran above). (Review fold: Arch #3.)
            return start_simulation_mode(
                goal=_new_goal,
                mode=mode,
                max_turns=max_turns,
                response_timeout=response_timeout,
                no_sim_env=no_sim_env,
                debug=debug,
                persistent_agent=persistent_agent,
                prompt_handler=prompt_handler,  # the caller's handler outlives /new too (D40)
            )
    else:
        # Non-interactive: just print the report
        from maxim.simulation.report import print_report

        print_report(report, session_dir=Path(report_dir) / report.session_id)

    return build_simulation_result(_tail, report, report_dir, duration, finish_reason)
