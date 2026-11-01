"""The end of a simulation run: shut its loops down, release what it held, save its report and build its result.

Moved out of ``orchestrator.py::start_simulation_mode`` (its tail, everything after the orchestrator loop returns) by
the decomposition's tail slice (roadmap_1_3_x.md §"The decomposition"; #1438 PR 1c, owner decision D5), behaviour
unchanged: each phase below is the tail's code verbatim, in the tail's order, reading the run's state from one
:class:`SimTail`. ``start_simulation_mode`` keeps the interactive post-sim prompt (it re-enters
``start_simulation_mode`` for a new goal, and the characterization harness patches its reader through
``orchestrator``) and calls the phases around it:

    end_run (shut_down_run -> release_run_resources -> save_run_report -> roundup_and_emit_report)
    -> (the post-sim prompt in orchestrator.py, or print_run_report) -> build_simulation_result

``_resolve_finish_reason`` and ``_sim_provenance`` moved with the code: the tail was their only caller.

Under a process stop request (#1438 PR 1c, owner decisions D2 and D4), :func:`end_run` reorders and bounds the same
phases instead: it signals the loops, saves an EARLY, partial report before any wait, waits within the tail's share of
the stop's deadline, releases what the run held, saves the LATE report only if time is left, and skips the LLM
roundup. With no stop request every step, order and wait is the one above.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maxim.simulation.report import run_provenance
from maxim.simulation.sim_types import SimulationResult, build_basic_analysis as _build_basic_analysis
from maxim.simulation.stdin_lines import STOP_JOIN_S, stop_reader
from maxim.utils.cancellation import StopRequest, current_stop, yield_requested
from maxim.utils.deadline import Deadline

logger = logging.getLogger(__name__)

# The tail's share of what is left of a stop's deadline when the tail starts. The console's drain
# (console/server.py::_drain_and_stop_handle) joins the run thread for share(0.4) of ITS bound, so a tail started at
# t0 into a budget B ends at t0 + 0.35 * (B - t0), inside that join while t0 < 0.077 B. A tail started later is caught
# by MaximHandle.stop's campaign-lock wait (share(0.3), ending near 0.58 B) while t0 < ~0.35 B; later still, the tail
# ends by B but after the drain judged the run thread, which reports an overrun and leaves the request set (fail
# closed). The tail shares the REQUEST's deadline; the drain waits on the tighter of it and its own. Not capped by the
# bound: each LLM worker's stop (its own joins, up to ~6 s) and the late save's AUT-state write, which the early
# save's time does not estimate. The rest of the deadline stays for the SimulationResult, the handle's lease safety net
# and the drain's own agent shutdown.
_SIM_TAIL_SHARE = 0.35
# The late report save starts only with more than this left (or more than the early save took, if longer).
_LATE_SAVE_FLOOR_S = 0.5


@dataclass
class SimTail:
    """The run state the tail reads, handed over once by ``start_simulation_mode``. In-process only, never persisted:
    every field is the run's own object (a thread, a bridge, a memory hub), named after the local it was."""

    is_interactive: Any
    stop_event: Any
    bridge: Any
    orchestrator_source: Any
    aut_thread: Any
    adopted: Any
    dm_thread: Any
    dm_error: Any
    aut_llm_worker: Any
    orch_llm_worker: Any
    orch_instance: Any
    aut_registry: Any
    aut_agent_id: Any
    aut_default_network: Any
    aut_pain_bus: Any
    sim_sandbox: Any
    aut_imagination_trigger: Any
    aut_nac: Any
    aut_component_registry: Any
    start_time: Any
    orch_error: Any
    operator_stop: Any
    goal: Any
    mode: Any
    aut_hippocampus: Any
    aut_memory_hub: Any
    llm_router: Any
    session_id: Any
    start_provenance: Any
    resume_record: Any
    aut_router: Any
    fixture_result: Any
    persistent_agent: Any
    experiment_log: Any
    sim_tmpdir: Any
    aut_executor: Any
    aut_introspector: Any
    pre_campaign_turns: Any
    campaign_analysis: Any
    dm_rollup: Any
    # The run's ``sim.stdin`` reader (#1541): joined by _wait_for_run, so none of this run's readers is left when
    # end_run returns and the post-sim prompt reads stdin. None: no reader to join.
    stdin_thread: Any = None
    # The process stop request the tail ran under (None: no stop), snapshotted ONCE by end_run; the orchestrator's
    # post-sim prompt and build_simulation_result read this snapshot, never the live request.
    stop: StopRequest | None = None


def _sim_provenance(
    start: dict[str, Any],
    orch_worker: Any,
    aut_worker: Any,
    resume: Any,
    adopted: Any,
    goal: str,
    deadline: Deadline | None = None,
) -> dict[str, Any]:
    """The report's provenance (M1 + baseline floors B0): extracted so ``start_simulation_mode`` does not grow.
    ``deadline``: a stopped sim's tail bound (``run_provenance`` skips the end code stamp and caps its waits)."""
    return run_provenance(
        start,
        llm_worker=orch_worker,
        aut_worker=aut_worker,
        resume=resume,
        adopted=adopted is not None,
        goal=goal,
        deadline=deadline,
    )


def _resolve_finish_reason(orch_error: Any, finish_context: dict | None, operator_stop: threading.Event) -> str:
    """How the run ended, in priority order:

    0. the process was asked to stop (``yield_requested()``, #1438 PR 1b) -> ``yielded``, a typed abort that outranks
       the statuses the stop itself causes (an ``llm_wedged`` or an ``error`` from abandoned calls; owner decision
       2026-10-30);
    1. the orchestrator raised -> ``error``;
    2. a ``finish_context`` status (the LLM's ``finish_simulation``, or a guard -- max turns, a stall, the
       AUT dying, planning liveness -- distinguished by ``initiated_by``);
    3. a human stopped it (``operator_stop``: ``/cancel``, Ctrl+C) -> ``cancel``;
    4. otherwise a runner finished or the loops exited -> ``completed``.

    ``stop_event`` is deliberately not an input: a finished runner and the shutdown both set it, so it
    read every finished run as ``cancel`` (#1002).
    """
    if yield_requested():
        return "yielded"
    if orch_error:
        return "error"
    if finish_context and finish_context.get("status"):
        return finish_context["status"]
    if operator_stop.is_set():
        return "cancel"
    return "completed"


def shut_down_run(tail: SimTail) -> None:
    """Signal every loop to stop, then wait for the AUT and DM threads, stop the LLM workers and shut the
    orchestrator instance down (persisting its memory)."""
    _signal_shutdown(tail)
    _wait_for_run(tail, bound=None)


def _signal_shutdown(tail: SimTail) -> None:
    """Tell every loop, LLM call and runner to stop. Never waits."""
    from maxim.simulation.sim_logger import display_status, get_active_display

    _is_interactive = tail.is_interactive
    stop_event = tail.stop_event
    bridge = tail.bridge
    orchestrator_source = tail.orchestrator_source

    # ── Interactive: update scene header for post-sim phase ──────────────
    if _is_interactive:
        display = get_active_display()
        if display is not None:
            display.set_scene(title="Simulation Complete")
            display.set_prompt("> Generating report...")

    # ── Suppress noisy log output during shutdown ─────────────────────
    # LLM responses may still be in-flight from background threads.
    # Raise log level to WARNING to hide [INFO] LLM tool_response lines.
    logging.getLogger("maxim").setLevel(logging.WARNING)

    # ── Shutdown everything (safe even after KeyboardInterrupt) ────────
    display_status("Shutting down agent loops...")
    # Signal the process-wide LLM cancellation primitive FIRST so any
    # in-flight retry loops in anthropic/openai backends abandon pending
    # retries immediately. Without this, a user Ctrl+C during a rate-limit
    # backoff would keep burning cloud credits for up to ~50s while the
    # retry chain completes. See models/language/cancellation.py.
    try:
        from maxim.models.language.cancellation import request_shutdown

        request_shutdown()
    except Exception as e:
        logger.debug("Failed to signal LLM cancellation: %s", e)
    try:
        stop_event.set()
        bridge.finish()
        orchestrator_source.finish()
    except Exception as e:
        logger.warning("Error while signalling shutdown: %s", e, exc_info=True)


def _wait_for_run(tail: SimTail, *, bound: Deadline | None) -> None:
    """Wait for the AUT and DM threads, stop the LLM workers and shut the orchestrator instance down. ``bound`` (a
    stopped sim's tail deadline) caps every wait; None: today's waits."""
    from maxim.simulation.sim_logger import display_status

    aut_thread = tail.aut_thread
    _adopted = tail.adopted
    _dm_thread = tail.dm_thread
    _dm_error = tail.dm_error
    aut_llm_worker = tail.aut_llm_worker
    orch_llm_worker = tail.orch_llm_worker
    _orch_instance = tail.orch_instance

    # The run's stop is set: its stdin reader ends at its next poll. Joined FIRST, before the post-sim prompt reads
    # stdin, so no reader of this run is left to take the next line typed (#1541).
    if tail.stdin_thread is not None:
        stop_reader(tail.stdin_thread, timeout_s=STOP_JOIN_S if bound is None else bound.cap(STOP_JOIN_S))

    display_status("Waiting for AUT to finish...")
    try:
        # Adopted persistent agent: the loop's tail runs FULL consolidation
        # (sleep replay + saves) which routinely exceeds 5s on a real history.
        # Returning early would (a) restore the tool lease under a live loop
        # and (b) let a second campaign's on_session_start race the first's
        # on_session_end on the same MemoryHub. (Review fold: Executor #5.)
        aut_join_s = 5.0 if _adopted is None else 300.0
        aut_thread.join(timeout=aut_join_s if bound is None else bound.cap(aut_join_s))
        if aut_thread.is_alive():
            display_status("AUT thread did not stop in time (continuing anyway)")
    except Exception as e:
        logger.warning("Error joining AUT thread during shutdown: %s", e, exc_info=True)

    # Wait for DM campaign thread if it was started
    if _dm_thread is not None and _dm_thread.is_alive():
        try:
            _dm_thread.join(timeout=5.0 if bound is None else bound.cap(5.0))
        except Exception as e:
            logger.debug("Failed to join the DM campaign thread: %s", e)
    if _dm_error:
        logger.error("DM campaign thread failed: %s", _dm_error[0])

    display_status("Stopping LLM workers...")
    for worker in (aut_llm_worker, orch_llm_worker):
        if worker:
            try:
                worker.stop()
            except Exception as e:
                logger.warning("Error stopping LLM worker during shutdown: %s", e, exc_info=True)

    # Persist orchestrator memory (Phase 3: cross-session learning).
    # shutdown() ends the session it owns (or waits for the loop that took it over) + saves hippocampus/cerebellum.
    try:
        if bound is None:
            _orch_instance.shutdown()
        else:
            # Its own session end and its wait for a loop holding the session both end at the tail's deadline.
            _orch_instance.shutdown(deadline=bound, loop_wait=bound)
    except Exception as e:
        logger.debug("Failed to shutdown orchestrator instance: %s", e)


def release_run_resources(tail: SimTail, *, aut_running: bool = False) -> None:
    """Restore an adopted agent's tools and pain bus, clean the sandbox up, end imagination's session and stop sim
    logging.

    ``aut_running`` (a stopped tail whose bounded AUT join expired, #1438 PR 1c; owner decision 2026-10-30): the AUT
    loop is still ending its session, so the steps that act on its live state -- the lease restore, the DN pain-bridge
    unsubscribe and imagination's tag/decay/clear -- are skipped and logged by name. The process is stopping; the
    sandbox cleanup and sim logging still run."""
    from maxim.simulation.sim_logger import display_status

    _adopted = tail.adopted
    aut_registry = tail.aut_registry
    _aut_agent_id = tail.aut_agent_id
    aut_default_network = tail.aut_default_network
    aut_pain_bus = tail.aut_pain_bus
    sim_sandbox = tail.sim_sandbox
    aut_imagination_trigger = tail.aut_imagination_trigger
    aut_nac = tail.aut_nac
    aut_component_registry = tail.aut_component_registry
    # HANDLE seam (a), branch 4: restore the persistent agent's registry to
    # its pre-campaign state (drop leased sim/DM tools, re-register the
    # persistent tools the noise-reduction loop removed). Runs AFTER the AUT
    # loop has fully stopped so no dispatch races a half-restored registry.
    # Exception paths that bail before this point are covered by the
    # idempotent safety-net restore in MaximHandle.play_campaign's finally.
    if _adopted is not None and aut_running:
        logger.warning(
            "Stopped sim: the AUT loop on %r is still ending its session; its tool lease restore and DefaultNetwork "
            "pain-bridge unsubscribe were skipped",
            _aut_agent_id,
        )
    elif _adopted is not None:
        try:
            _dropped, _restored = _adopted.lease.restore(aut_registry)
            logger.info(
                "Campaign tool lease restored on %r: dropped %d campaign tools, re-registered %d persistent tools",
                _aut_agent_id,
                len(_dropped),
                len(_restored),
            )
        except Exception as e:
            logger.warning("Campaign tool lease restore failed for %r: %s", _aut_agent_id, e)

        # Un-pin the sim-scoped DefaultNetwork's PainCircuitBridge from the
        # PERSISTENT pain bus. In the throwaway path the bus dies with the
        # sim; here it lives on — without this, every campaign accumulates a
        # dead DN subscriber on the agent's bus (a latent duplicate NAc pain
        # learner). (Review fold: Executor #3 / bio-fidelity F2.)
        if aut_default_network is not None and aut_pain_bus is not None:
            try:
                _dn_bridge = getattr(aut_default_network, "pain_bridge", None)
                if _dn_bridge is not None:
                    aut_pain_bus.unsubscribe(_dn_bridge._on_pain)
                    logger.info("DefaultNetwork pain bridge unsubscribed from persistent bus")
            except Exception as e:
                logger.warning("Failed to unsubscribe DN pain bridge from persistent bus: %s", e)

    # Clean up sandbox
    if sim_sandbox:
        pain_count = len(sim_sandbox.pain_events)
        if pain_count > 0:
            display_status(f"Pain signals fired: {pain_count}")
        try:
            sim_sandbox.cleanup()
        except Exception as e:
            logger.warning("Sandbox cleanup failed: %s", e, exc_info=True)

    # Imagination session-end cleanup:
    # 1. Retroactively tag causal links involving imagined entities
    # 2. Decay tagged links by 50%
    # 3. Clear ephemeral registry entries
    # HANDLE seam (a), branch 6: aut_nac derives from the adopted persistent
    # instance, so WHEN this block runs, tag/decay land on the PERSISTENT
    # agent's NAc. HONESTY NOTE (three-lens review, cross-confirmed): on the
    # injected DM path this block is structurally INERT today — the
    # imagination trigger requires entity_ref, which injection forbids, so
    # campaign-DECLARED fiction (the DM's ghost merchant) persists as real
    # learning. Episodes-as-real is the design's stated intent; fiction
    # provenance for campaign-declared entities is a tracked follow-up in
    # docs/plans/archive/console_handle_campaign_injection.md (needs a session-scoped
    # tag set — decay_imagined_links decays ALL imagined links per call,
    # which compounds across a persistent agent's repeated campaigns).
    if aut_imagination_trigger is not None and aut_running:
        logger.warning("Stopped sim: the AUT loop is still running; imagination's tag/decay/clear was skipped")
    elif aut_imagination_trigger is not None:
        try:
            stats = aut_imagination_trigger.stats()
            if stats.get("designs_succeeded", 0) > 0:
                display_status(
                    f"Imagination: {stats['designs_succeeded']} entities imagined, "
                    f"{stats['phrases_extracted']} phrases extracted"
                )
            # Tag + decay causal links from imagined entities
            if aut_nac is not None:
                imagined_refs = aut_imagination_trigger.imagined_refs
                if imagined_refs:
                    tagged = aut_nac.tag_imagined_links(imagined_refs)
                    if tagged > 0:
                        logger.info("Tagged %d causal links as imagined", tagged)
                decayed = aut_nac.decay_imagined_links(0.5)
                if decayed > 0:
                    logger.info("Decayed %d imagined causal links (factor=0.5)", decayed)
            # Clear ephemeral registry entries (session-scoped)
            if aut_component_registry is not None:
                cleared = aut_component_registry.clear_ephemeral()
                if cleared:
                    logger.info("Cleared %d ephemeral entities from registry", len(cleared))
        except Exception as e:
            logger.debug("Imagination cleanup failed: %s", e)

    # Disable sim logging
    try:
        from maxim.simulation.sim_logger import disable_sim_logging

        disable_sim_logging()
    except Exception as e:
        logger.debug("disable_sim_logging failed: %s", e)


def save_run_report(
    tail: SimTail, *, provenance: dict[str, Any] | None = None, partial: bool = False
) -> tuple[Any, str, float, str]:
    """Build the run's report and save it, its action log, the AUT's state and the experiment log; returns
    ``(report, report_dir, duration, finish_reason)``.

    A stopped tail passes ``provenance`` (computed once, for both its saves; None: computed here) and ``partial`` for
    its early save, which writes report.json and the action log only."""
    from maxim.simulation.sim_logger import display_status

    bridge = tail.bridge
    _adopted = tail.adopted
    aut_llm_worker = tail.aut_llm_worker
    orch_llm_worker = tail.orch_llm_worker
    aut_nac = tail.aut_nac
    start_time = tail.start_time
    orch_error = tail.orch_error
    operator_stop = tail.operator_stop
    goal = tail.goal
    mode = tail.mode
    aut_hippocampus = tail.aut_hippocampus
    aut_memory_hub = tail.aut_memory_hub
    llm_router = tail.llm_router
    session_id = tail.session_id
    start_provenance = tail.start_provenance
    resume_record = tail.resume_record
    aut_router = tail.aut_router
    fixture_result = tail.fixture_result
    persistent_agent = tail.persistent_agent
    experiment_log = tail.experiment_log
    sim_tmpdir = tail.sim_tmpdir
    # ── Build comprehensive report ──────────────────────────────────────
    from maxim.simulation.report import (
        build_report,
        save_report,
        save_action_log,
        save_aut_state,
    )

    duration = time.time() - start_time
    llm_finish = bridge.finish_context if bridge.finish_context else None
    finish_reason = _resolve_finish_reason(orch_error, llm_finish, operator_stop)

    if llm_finish:
        logger.info(
            "LLM-initiated finish: status=%s reason=%s",
            llm_finish.get("status"),
            llm_finish.get("reason"),
        )

    display_status("Building simulation report...")
    report = build_report(
        goal=goal,
        mode=mode,
        bridge=bridge,
        duration_s=duration,
        finish_reason=finish_reason,
        aut_hippocampus=aut_hippocampus,
        aut_nac=aut_nac,
        aut_memory_hub=aut_memory_hub,
        llm_router=llm_router,
        language_model=(
            # Prefer AUT router's model name when dual-LLM mode is active,
            # otherwise fall back to the shared router.
            getattr(aut_router, "last_used_model", "")
            or getattr(aut_router, "model_name", "")
            or getattr(aut_router, "active_model", "")
            or getattr(llm_router, "last_used_model", "")
            or getattr(llm_router, "model_name", "")
            or getattr(llm_router, "active_model", "")
        )
        if llm_router
        else "",
        # Routing-path diagnostics: provider key + backend class + endpoint URL.
        # Without these, the report cannot distinguish e.g. claude-sonnet-4-6
        # via Anthropic direct from claude-sonnet-4-6 via a leader proxy.
        # Falls through aut_router → llm_router (single-router sims read llm_router).
        language_provider=(
            getattr(aut_router, "last_used_provider", "") or getattr(llm_router, "last_used_provider", "")
        )
        if llm_router
        else "",
        language_backend_class=(
            getattr(aut_router, "last_used_backend_class", "") or getattr(llm_router, "last_used_backend_class", "")
        )
        if llm_router
        else "",
        language_endpoint=(
            getattr(aut_router, "last_used_endpoint", "") or getattr(llm_router, "last_used_endpoint", "")
        )
        if llm_router
        else "",
        # AUT-specific routing when dual-LLM mode is active. Empty when
        # single-router (orchestrator + AUT share one router).
        aut_model=(
            getattr(aut_router, "last_used_model", "")
            or getattr(aut_router, "model_name", "")
            or getattr(aut_router, "active_model", "")
        )
        if aut_router and aut_router is not llm_router
        else "",
        aut_provider=getattr(aut_router, "last_used_provider", "")
        if aut_router and aut_router is not llm_router
        else "",
        aut_backend_class=getattr(aut_router, "last_used_backend_class", "")
        if aut_router and aut_router is not llm_router
        else "",
        aut_endpoint=getattr(aut_router, "last_used_endpoint", "")
        if aut_router and aut_router is not llm_router
        else "",
        llm_finish_context=llm_finish,
        # Plan 4 follow-up (2026-04-14): forward the session_id we
        # generated at sim entry so the report's session_id matches the
        # same value every LLMWorker threaded into its request_context
        # dict. Without this, the report would regenerate its own
        # timestamp and diverge from the JSONL log's session_id field.
        session_id=session_id,
        started_at=start_time,
        provenance=provenance
        if provenance is not None
        else _sim_provenance(start_provenance, orch_llm_worker, aut_llm_worker, resume_record, _adopted, goal),
    )

    # Attach fixture/substrate metrics if present
    if fixture_result:
        report.substrate_metrics = fixture_result.get("substrate_metrics", {})
        report.substrate_metrics["fixture"] = {k: v for k, v in fixture_result.items() if k != "substrate_metrics"}

    # Persist everything to session directory
    from maxim.utils.paths import sim_reports as _sim_reports_dir

    report_dir = str(_sim_reports_dir())
    display_status(f"Saving report to {report_dir}/{report.session_id}/...")
    save_report(report, base_dir=report_dir)

    action_count = len(bridge.get_all_actions())
    display_status(f"Saving action log ({action_count} records)...")
    save_action_log(bridge, base_dir=report_dir, session_id=report.session_id)
    if partial:
        return report, report_dir, duration, finish_reason

    if persistent_agent is not None:
        # HANDLE seam (a), branch 3: the persistent agent's state lives in
        # its OWN home (saved by the loop's full session-end + the handle's
        # shutdown). Do NOT also snapshot it into the session dir — that
        # duplicates the whole memory per campaign and creates a stale
        # second source a later --resume-session load would clobber from.
        display_status("Persistent agent — state saved to its own home (no session AUT snapshot)")
    else:
        if aut_hippocampus is not None or aut_nac is not None:
            aut_mem = len(aut_hippocampus) if aut_hippocampus else 0
            aut_links = sum(len(v) for v in aut_nac._links.values()) if aut_nac else 0
            display_status(f"Saving AUT state ({aut_mem} memories, {aut_links} causal links)...")
        save_aut_state(
            hippocampus=aut_hippocampus,
            nac=aut_nac,
            base_dir=report_dir,
            session_id=report.session_id,
            ec=aut_memory_hub.ec if aut_memory_hub is not None else None,
            atl=aut_memory_hub.atl if aut_memory_hub is not None else None,
            scn=aut_memory_hub.scn if aut_memory_hub is not None else None,
        )

    # Copy experiment log to report directory (if any experiments were recorded)
    if experiment_log is not None and len(experiment_log) > 0:
        import shutil

        exp_src = sim_tmpdir / "experiments.jsonl"
        exp_dst = Path(report_dir) / report.session_id / "experiments.jsonl"
        if exp_src.exists():
            exp_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(exp_src), str(exp_dst))
            display_status(f"Saving experiment log ({len(experiment_log)} experiments)...")
    return report, report_dir, duration, finish_reason


def roundup_and_emit_report(tail: SimTail, report: Any, report_dir: str) -> None:
    """Run the LLM roundup into the saved report, and emit the machine-readable report when one was requested."""
    from maxim.simulation.report import analyze_simulation, save_report
    from maxim.simulation.sim_logger import display_status

    llm_router = tail.llm_router
    # LLM-powered roundup (log noise suppressed by WARNING level above)
    if llm_router is not None and not getattr(llm_router, "session_cost_exceeded", False):
        try:
            display_status("Running LLM analysis roundup...")
            # request_shutdown() was called above to cancel in-flight agent
            # retries.  Clear the flag here so the roundup LLM call can
            # proceed — without this _MaximPeerBackend.complete_with_usage()
            # sees is_shutdown_requested()==True and raises BackendDown
            # immediately (0.1ms dispatch_exhausted on every sim end).
            from maxim.models.language.cancellation import reset_shutdown

            logger.debug("LLM shutdown signal cleared for sim roundup")
            reset_shutdown()
            analyze_simulation(report, llm_router=llm_router)
            save_report(report, base_dir=report_dir)
        except Exception:
            display_status("LLM roundup skipped (model unavailable after shutdown)")
    elif llm_router is not None:
        display_status("Skipping LLM roundup (session cost ceiling reached)")

    _emit_report_json(report)


def _emit_report_json(report: Any) -> None:
    # Emit machine-readable report when --report-json was requested via
    # MAXIM_REPORT_JSON.  Set by cli.py from args.report_json so the env
    # var is the contract — orchestrator does NOT import argparse.  Pop
    # (not just read) so that two back-to-back sims in the menu loop do
    # not silently overwrite the first sim's report file when only the
    # first invocation passed --report-json.
    from maxim.simulation.report import emit_report_json

    _report_json_path = os.environ.pop("MAXIM_REPORT_JSON", "").strip()
    if _report_json_path:
        try:
            emit_report_json(report, _report_json_path)
        except Exception as e:
            logger.warning("Failed to emit report JSON to %r: %s", _report_json_path, e)


def end_run(tail: SimTail) -> tuple[Any, str, float, str]:
    """The run's end, before the post-sim prompt: shut down, release, save the report, then the roundup. Returns
    ``(report, report_dir, duration, finish_reason)``. Under a process stop request: :func:`_end_stopped_run`. A
    phase added here is added there too, or the stop path's omission is stated in its docstring."""
    tail.stop = current_stop()
    if tail.stop is not None:
        return _end_stopped_run(tail, tail.stop)
    shut_down_run(tail)
    release_run_resources(tail)
    report, report_dir, duration, finish_reason = save_run_report(tail)
    roundup_and_emit_report(tail, report, report_dir)
    return report, report_dir, duration, finish_reason


def _end_stopped_run(tail: SimTail, stop: StopRequest) -> tuple[Any, str, float, str]:
    """The run's end under a stop request, within ``_SIM_TAIL_SHARE`` of its deadline (#1438 PR 1c, D2 + D4).

    The early report (report.json and the action log, ``stop_stage: "early"`` in its provenance) is saved before any
    wait, so a stop that expires in a wait still leaves one; it reads live state while the AUT may still be ending, so
    it is partial by name. The provenance is computed once, bounded, for both saves. The late, full save runs only if
    more than the early save took (at least ``_LATE_SAVE_FLOOR_S``) is left, and may overshoot by its own duration.
    No LLM roundup: a stopped run makes no new LLM call. A yielded run is a typed abort either way, so neither report
    is evidence. A failed early save never costs the waits or the release (the late save is then tried whatever the
    time); a failed late save keeps the early report; only both failing raises, after the release."""
    bound = stop.deadline.share(_SIM_TAIL_SHARE)
    _signal_shutdown(tail)
    provenance = _sim_provenance(
        tail.start_provenance,
        tail.orch_llm_worker,
        tail.aut_llm_worker,
        tail.resume_record,
        tail.adopted,
        tail.goal,
        deadline=bound,
    )
    started = time.monotonic()
    saved: tuple[Any, str, float, str] | None = None
    try:
        saved = save_run_report(tail, provenance={**provenance, "stop_stage": "early"}, partial=True)
    except Exception as e:
        logger.error("Stopped sim: the early report save failed; continuing to the waits and the release: %s", e)
    early_s = time.monotonic() - started
    _wait_for_run(tail, bound=bound)
    release_run_resources(tail, aut_running=tail.aut_thread.is_alive())
    if saved is None or bound.remaining() > max(_LATE_SAVE_FLOOR_S, early_s):
        try:
            saved = save_run_report(tail, provenance={**provenance, "stop_stage": "late"})
        except Exception as e:
            if saved is None:
                raise
            logger.error(
                "Stopped sim: the late report save failed part-way (report.json on disk may already be the late one); "
                "returning and emitting the early report: %s",
                e,
            )
    else:
        logger.warning(
            "Stopped sim %s: late report save skipped (%.1fs left of the tail's bound, the early save took %.1fs); "
            "report.json is the early, partial one, and the AUT state and experiment log were not saved",
            saved[0].session_id,
            bound.remaining(),
            early_s,
        )
    report, report_dir, duration, finish_reason = saved
    logger.warning("LLM roundup skipped: a process stop is requested (%s)", stop.reason)
    _emit_report_json(report)
    return report, report_dir, duration, finish_reason


def print_run_report(tail: SimTail, report: Any, report_dir: str) -> None:
    """Print the report where there is no post-sim prompt: a non-interactive run, or an interactive one under a stop
    (the prompt waits on a human and starts a new sim; neither belongs inside a stop), whose live display closes
    first so the report prints to the terminal."""
    from maxim.simulation.report import print_report

    if tail.is_interactive:
        from maxim.simulation.sim_logger import get_active_display, set_active_display

        display = get_active_display()
        if display is not None:
            display.stop()
            set_active_display(None)  # else the report is routed into the stopped panel and never printed
    print_report(report, session_dir=Path(report_dir) / report.session_id)


def build_simulation_result(
    tail: SimTail, report: Any, report_dir: str, duration: float, finish_reason: str
) -> SimulationResult:
    """The run's :class:`SimulationResult`, with the detailed data captured for programmatic access."""
    bridge = tail.bridge
    goal = tail.goal
    mode = tail.mode
    aut_executor = tail.aut_executor
    aut_introspector = tail.aut_introspector
    pre_campaign_turns = tail.pre_campaign_turns
    campaign_analysis = tail.campaign_analysis
    dm_rollup = tail.dm_rollup
    # ── Capture detailed data for programmatic access ──────────────────
    # Tool usage stats from the inner executor (before wrappers)
    _tool_stats: dict[str, Any] = {}
    try:
        # Unwrap FearGated/Pain wrappers to reach the inner Executor
        _inner = aut_executor
        for _attr in ("_executor", "_executor"):
            if hasattr(_inner, _attr):
                _inner = getattr(_inner, _attr)
        if hasattr(_inner, "tool_usage_stats"):
            _tool_stats = _inner.tool_usage_stats()
    except Exception as e:
        logger.warning("Failed to collect tool_usage_stats for SimulationResult: %s", e, exc_info=True)

    # Serialized action history
    _actions: list[dict[str, Any]] = []
    try:
        for a in bridge.get_all_actions():
            _actions.append(
                {
                    "timestamp": a.timestamp,
                    "tool_name": a.tool_name,
                    "tool_args": getattr(a, "tool_args", {}),
                    "result_success": a.result_success,
                    "result_output": str(a.result_output)[:200] if a.result_output else None,
                    "blocked": a.blocked,
                    "block_reason": a.block_reason,
                    "utterance": a.utterance,
                }
            )
    except Exception as e:
        logger.warning("Failed to serialize action history for SimulationResult: %s", e, exc_info=True)

    # Subsystem snapshot
    _snapshot: dict[str, Any] = {}
    # A stopped tail whose AUT thread is still ending reads none of its live state (the snapshot, the basic analysis).
    aut_running = tail.stop is not None and tail.aut_thread.is_alive()
    if aut_running:
        logger.warning("Subsystem snapshot and analysis skipped: the AUT thread outlived the stopped tail")
    elif aut_introspector is not None:
        try:
            _snapshot = aut_introspector.benchmark_snapshot()
        except Exception as e:
            logger.warning("Failed to collect subsystem snapshot for SimulationResult: %s", e, exc_info=True)

    # JSON parse compliance
    _router_stats: dict[str, Any] = {}
    try:
        from maxim.models.language.json_parser import json_parse_stats

        _router_stats = json_parse_stats()
    except Exception as e:
        logger.debug("json_parse_stats unavailable: %s", e)

    _session_dir = str(Path(report_dir) / report.session_id)

    result = SimulationResult(
        goal=goal,
        mode=mode,
        turns=report.turns,
        total_actions=report.total_actions,
        blocked_actions=report.blocked_actions,
        duration_s=duration,
        finish_reason=finish_reason,
        summary=report.llm_summary,
        session_id=report.session_id,
        session_dir=_session_dir,
        campaign_analysis=campaign_analysis
        if pre_campaign_turns
        else (dm_rollup or ({} if aut_running else _build_basic_analysis(aut_introspector))),
        introspector=aut_introspector,
        tool_stats=_tool_stats,
        actions=_actions,
        subsystem_snapshot=_snapshot,
        router_stats=_router_stats,
    )

    return result
