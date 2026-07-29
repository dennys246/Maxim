"""MaximHandle — one persistent agent behind the Console (HANDLE seam).

The headless flavor of the HANDLE seam ([reachy_app_maxim_seams.md] § HANDLE,
design in [console_handle_campaign_injection.md]): a small wrapper over ONE
persistent ``AgentInstance`` built via ``AgentFactory.create_full_agent``
with ``auto_load=True`` over a ``~/.maxim`` home. Modes are methods:

* ``play_campaign(path)`` — run a DM campaign **as the persistent agent**
  (the campaign-injection surface: ``start_simulation_mode(dm_campaign=…,
  persistent_agent=self.instance)``). Adventure teaches Talk — the episodes
  land in the persistent agent's own Hippocampus/NAc, not a throwaway.
* ``stop(consolidation="full")`` — clean shutdown: full sleep/replay
  consolidation + hippocampus/NAc/cerebellum saves (#427's explicit flavor).

* ``play_premise(text)`` — the same, from a free-text premise the narrator
  improvises (generative flavor) instead of an authored YAML.
* ``talk(text)`` — one conversational turn against a LIVE ``run_agentic_loop``
  over this agent's bio-stack, fed by a ``SimulationBridge``'s
  ``ConversationalSource``. The loop persists between turns; the reply
  reaches the console's ``/ws`` stream as CLEAN-tier ``USER``/``RESPONSE``
  records (the EVENT seam — rides ``sim_log``, not ``api.on()``).

``rest(...)`` remains unimplemented. Talk and an adventure are mutually
exclusive: both drive the same bio-stack, so starting a campaign stops the
talk loop first (the next ``talk()`` rebuilds it lazily).

The constructor is **body-agnostic**: the embodied (Reachy) flavor is the
same interface with ``body="bodies/reachy_mini"`` — ``RunSurface`` drives a
HANDLE, not "a robot".
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


def _extract_reply(turn: dict[str, Any]) -> str | None:
    """Pull the agent's actual words out of a bridge turn.

    The bridge derives its ``response`` by joining respond/speak
    ``result_output``, which works for the sim's ``SimRespondTool`` (its output
    IS the text) but NOT for the production :class:`RespondTool`, whose output
    is a delivery RECEIPT (``{"delivered": True, "mode": "cli"}``). The words
    live in ``tool_args["message"]``. Prefer those; fall back to the bridge's
    own field so both tool flavors work.
    """
    parts: list[str] = []
    for action in turn.get("actions") or []:
        if getattr(action, "tool_name", "") not in ("respond", "speak"):
            continue
        args = getattr(action, "tool_args", None) or {}
        message = args.get("message") or args.get("text")
        if message:
            parts.append(str(message))
    if parts:
        return "\n".join(parts)
    fallback = turn.get("response")
    # Guard against the receipt dict leaking through as a stringified dict.
    if isinstance(fallback, dict) or not fallback:
        return None
    return str(fallback)


class MaximHandle:
    """A live handle on one persistent Maxim agent.

    Args:
        agent_id: The persistent identity episodes attribute to. Must NOT be
            ``"sim_aut"`` (that id marks the throwaway sim AUT).
        home: Persistence directory for the agent. Default: the factory's
            ``~/.maxim/agents/<agent_id>`` home.
        body: Optional SEM component ref (e.g. ``"bodies/reachy_mini"``) for
            the embodied flavor. The handle wires it at CONSTRUCTION
            (``AgentConfig.embodiment_ref``) — never via the sim's
            ``entity_ref``, which is incompatible with injection.
    """

    def __init__(
        self,
        agent_id: str = "console_agent",
        *,
        home: str | Path | None = None,
        body: str | None = None,
    ) -> None:
        from maxim.runtime.agent_factory import AgentConfig, AgentFactory
        from maxim.runtime.bootstrap import build_tool_registry

        if agent_id == "sim_aut":
            raise ValueError("agent_id 'sim_aut' is reserved for the throwaway sim AUT")

        component_registry: Any | None = None
        if body is not None:
            from maxim.embodiment.component_registry import ComponentRegistry

            component_registry = ComponentRegistry()

        config = AgentConfig(
            agent_id=agent_id,
            role="pc",
            persistence_dir=str(home) if home is not None else None,
            with_bio_stack=True,
            with_executor=True,
            # Mirror the sim AUT's pain-bridge decision: subscription only
            # matters when an embodiment can publish SEM pain.
            with_pain_bridge=body is not None,
            with_fear_gate=False,
            embodiment_ref=body,
        )
        factory = AgentFactory(component_registry=component_registry)
        # A ResponseOutput is what makes RespondTool/SpeakTool exist — without
        # it the agent has literally no way to reply, so talk() would always
        # come back empty. Sandbox lands under the agent's own home.
        self._response_output: Any | None = None
        try:
            from maxim.utils.paths import agent_data
            from maxim.utils.response_output import ResponseOutput

            sandbox = Path(home) / "responses" if home is not None else agent_data(agent_id) / "responses"
            sandbox.mkdir(parents=True, exist_ok=True)
            self._response_output = ResponseOutput(sandbox_path=str(sandbox))
        except Exception:
            logger.warning("Console handle: no ResponseOutput — talk() will not be able to reply", exc_info=True)
        tool_registry = build_tool_registry(operational_mode="active", response_output=self._response_output)
        self.instance = factory.create_full_agent(
            config,
            tool_registry=tool_registry,
            auto_load=True,
        )
        self.agent_id = agent_id
        self._stopped = False
        # One campaign at a time — the persistent substrate is not
        # re-entrant across concurrent sims (shared Hippocampus/NAc).
        self._campaign_lock = threading.Lock()
        # Talk-mode live loop (built lazily on the first talk()).
        self._talk_lock = threading.Lock()
        self._talk_bridge: Any = None
        self._talk_thread: threading.Thread | None = None
        self._talk_stop: threading.Event | None = None
        self._talk_worker: Any = None  # LLMWorker owned by the talk loop

    # ── modes ───────────────────────────────────────────────────────────

    def play_campaign(self, path: str | Path, *, max_turns: int = 100) -> Any:
        """Run a DM campaign as the persistent agent; blocks until it ends.

        Returns the ``SimulationResult``. The campaign's episodes are
        recallable from ``self.instance.hippocampus`` afterwards — that is
        the seam's whole point.
        """
        if self._stopped:
            raise RuntimeError("MaximHandle is stopped — build a new handle to play again")
        campaign_path = Path(path)
        if not campaign_path.exists():
            raise FileNotFoundError(f"Campaign not found: {campaign_path}")

        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.simulation.dm_schema import load_campaign, validate_campaign

        registry = ComponentRegistry(campaign_dir=str(campaign_path.parent))
        campaign = load_campaign(campaign_path, registry=registry)
        errors = validate_campaign(campaign)
        if errors:
            raise ValueError(f"Campaign validation failed ({len(errors)} errors): {errors}")

        return self._run_sim(
            goal=f"dm:{campaign.name}",
            persona="dungeon_master",
            dm_campaign=campaign,
            max_turns=max_turns,
        )

    def play_premise(self, premise: str, *, max_turns: int = 20) -> Any:
        """Run a GENERATIVE campaign from a free-text premise, as the agent.

        The "describe an adventure and let Maxim imagine it" path: the
        narrator improvises the arc from ``premise`` (no authored YAML), with
        the same persistent-agent injection as :meth:`play_campaign` — so an
        imagined adventure teaches Talk exactly like an authored one.

        ``max_turns`` defaults lower than the campaign path: a generative arc
        has no authored end condition, so it runs until the turn cap.
        """
        if self._stopped:
            raise RuntimeError("MaximHandle is stopped — build a new handle to play again")
        premise = (premise or "").strip()
        if not premise:
            raise ValueError("premise must be non-empty")
        return self._run_sim(goal=premise, persona="collaborative", generative=True, max_turns=max_turns)

    # ── talk (live loop) ────────────────────────────────────────────────

    def talk(self, text: str, *, timeout: float = 180.0) -> dict[str, Any]:
        """One conversational turn against the LIVE persistent agent.

        The first call starts a real ``run_agentic_loop`` in a background
        thread over THIS agent's bio-stack (hippocampus / NAc / memory hub /
        executor), fed by a :class:`SimulationBridge`'s ``ConversationalSource``.
        The loop stays alive between calls — that is what makes it a live-loop
        mode rather than a one-shot: working memory, deliberation state and the
        bio-pipeline persist across turns the way they do in a sim.

        This deliberately does NOT ride ``AgentPool.run_turn`` — that is the
        lightweight NPC path (personality-only, no bio-pipeline), which would
        make talk look alive while being cognitively hollow.

        Returns the bridge turn dict (``response`` / ``actions`` / ``turn`` /
        ``timed_out`` / ``duration_ms``). The reply ALSO reaches the console's
        ``/ws`` stream as CLEAN-tier records (``USER`` for the utterance,
        ``RESPONSE`` for the reply), which is how the web chat renders it —
        the return value is for programmatic callers.
        """
        if self._stopped:
            raise RuntimeError("MaximHandle is stopped — build a new handle to talk again")
        text = (text or "").strip()
        if not text:
            raise ValueError("talk() requires a non-empty utterance")
        if self._campaign_lock.locked():
            raise RuntimeError("An adventure is running on this handle — talk is unavailable until it ends")

        from maxim.simulation.sim_logger import sim_log

        bridge = self._ensure_talk_loop()
        # The user's own utterance on the wire (CLEAN tier). The web chat
        # echoes locally and dedupes this; a terminal/replay consumer needs it.
        sim_log("USER", text, {"text": text}, agent_id=self.agent_id)
        result = bridge.send_and_wait(text, timeout=timeout)
        reply = _extract_reply(result)
        # Normalize so programmatic callers and the wire agree on the text.
        result["response"] = reply
        if reply:
            sim_log("RESPONSE", str(reply), {"text": str(reply)}, agent_id=self.agent_id)
        else:
            # Say so on the wire rather than leaving the chat silent — a
            # timeout or a turn that produced only non-verbal actions is a
            # real outcome the UI should be able to render.
            sim_log(
                "RESPONSE",
                "(no reply — the turn produced no respond/speak action)",
                {"text": None, "timed_out": bool(result.get("timed_out"))},
                agent_id=self.agent_id,
            )
        return result

    def _ensure_talk_loop(self) -> Any:
        """Build + start the talk loop once; return its bridge."""
        with self._talk_lock:
            if self._talk_bridge is not None and self._talk_thread is not None and self._talk_thread.is_alive():
                return self._talk_bridge
            stale_worker = self._talk_worker
            self._talk_bridge = self._talk_thread = self._talk_stop = self._talk_worker = None
        # A previous loop died (its thread raised and was logged). Its
        # LLMWorker owns a live thread pool — stop it BEFORE building a
        # replacement, or every failed turn leaks another pool (review
        # finding). Done outside the lock: stop() can block.
        if stale_worker is not None:
            try:
                stale_worker.stop()
            except Exception:
                logger.debug("stale talk LLMWorker.stop() raised", exc_info=True)

        with self._talk_lock:
            if self._talk_bridge is not None and self._talk_thread is not None and self._talk_thread.is_alive():
                return self._talk_bridge  # another caller won the race

            from maxim.agents.llm_worker import LLMWorker
            from maxim.runtime.lane_backends import build_primary_router
            from maxim.utils.paths import agent_data

            workspace = agent_data(self.agent_id) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)

            router, _lane_manager = build_primary_router()
            if router is None:
                from maxim.exceptions import ConfigurationError

                raise ConfigurationError(
                    "No LLM backend available for talk — configure one first "
                    "(maxim doctor, or the console's setup page)."
                )
            worker = LLMWorker(llm=router, n_ctx=router.n_ctx, token_counter=router.get_token_counter())
            worker.start()
            try:
                return self._launch_talk_loop(worker, workspace)
            except Exception:
                # A raise between start() and the state assignment would strand
                # a running worker pool with no reference (review finding).
                worker.stop()
                raise

    def _launch_talk_loop(self, worker: Any, workspace: Path) -> Any:
        """Build the loop stack around a started worker and launch it.

        Caller MUST hold ``self._talk_lock`` — this publishes the loop handles.
        """
        from maxim.agents.autonomy import (
            AutonomyController,
            AutonomyLevel,
            SafetyConstraints,
            SupervisionPolicy,
        )
        from maxim.agents.maxim_agent import MaximAgent
        from maxim.environment.filesystem_env import FileSystemEnv
        from maxim.runtime.agent_loop import run_agentic_loop
        from maxim.runtime.bootstrap import build_decision_engine, build_memory
        from maxim.runtime.state import RuntimeState
        from maxim.simulation.bridge import SimulationBridge

        state = RuntimeState()
        state.data["mode"] = "active"
        state.data["active_goal"] = "converse with the user"

        # Conversation-shaped permissions: the agent may speak and read,
        # not mutate the operator's box unprompted. Talk is the DEFAULT
        # surface of a local-first console — a destructive tool firing
        # from small talk is the failure mode to design out.
        #
        # ENFORCEMENT NOTE (cross-confirmed review finding): at
        # AUTONOMOUS, `SupervisionPolicy.allowed_tools` is NEVER consulted
        # — autonomy.py's level branch returns True after safety
        # constraints ("AUTONOMOUS - only safety constraints apply"). An
        # allow-list here would be a silent no-op, and the handle's
        # registry (operational_mode="active") carries bash/write_file/
        # edit_file scoped to the SERVER'S CWD. So the restriction is
        # expressed as SafetyConstraints.forbidden_tools, which IS applied
        # at every level. Keep the supervision policy too: it becomes live
        # if talk is ever run SUPERVISED, and the two agree.
        conversational = {"respond", "speak", "read_file", "list_directory", "glob", "recall", "sense_tools"}
        mutating = {
            t
            for t in (self.instance.tool_registry.list_all() if self.instance.tool_registry else ())
            if t not in conversational
        }
        autonomy = AutonomyController(
            initial_level=AutonomyLevel.AUTONOMOUS,
            supervision_policy=SupervisionPolicy(allowed_tools=set(conversational)),
            safety_constraints=SafetyConstraints(
                # Deny-by-default over the CURRENT registry: anything that
                # is not conversational is forbidden for talk turns. Derived
                # rather than enumerated so a newly-registered tool is
                # denied by construction, not by remembering to list it.
                forbidden_tools=frozenset(mutating | set(SafetyConstraints().forbidden_tools))
            ),
        )

        # spinner_prefix="" keeps the bridge's progress chatter out of a
        # server process's stdout; settle detection still applies.
        # stop_event is SHARED with send_and_wait's poll loop so a turn in
        # flight aborts when the loop dies or a campaign stops it —
        # without it the caller waits the full 180s timeout for a reply
        # that can never come (review finding).
        stop_event = threading.Event()
        bridge = SimulationBridge(response_timeout=180.0, settle_s=2.0, spinner_prefix="", stop_event=stop_event)

        # THE memory-read path. `agent_loop` gates its ENTIRE enrichment
        # block (hippocampal recall, ATL concepts, cerebellum predictions)
        # on `bio_enrichment_pipeline is not None or thought_gate is not
        # None`, and multi-cycle deliberation on the pipeline too. Without
        # these, talk has a hippocampus wired and no automatic way to put a
        # recalled memory into the prompt — "Adventure teaches Talk" would
        # be write-only. They already exist on the instance's bio-stack;
        # this just hands them to the loop (review finding).
        bio_stack = getattr(self.instance, "bio_stack", None)
        enrichment = getattr(bio_stack, "bio_enrichment_pipeline", None)
        thought_gate = getattr(bio_stack, "thought_gate", None)

        agent = MaximAgent()
        if self.instance.memory_hub is not None:
            # Connects MemoryAgent → hippocampus, registers the promotion
            # source + deletion callback. The orchestrator does this for
            # the AUT; talk needs it for the same reason.
            agent.wire_memory_hub(self.instance.memory_hub)

        def _worker() -> None:
            try:
                run_agentic_loop(
                    agent,
                    FileSystemEnv(str(workspace)),
                    state,
                    build_memory(),
                    build_decision_engine(),
                    self.instance.executor,
                    autonomy_controller=autonomy,
                    llm_worker=worker,
                    hippocampus=self.instance.hippocampus,
                    memory_hub=self.instance.memory_hub,
                    bio_enrichment_pipeline=enrichment,
                    thought_gate=thought_gate,
                    max_steps=0,  # unlimited — ends on bridge.finish()/stop_event
                    stop_event=stop_event,
                    target_hz=2.0,
                    percept_source=bridge.percept_source,
                    action_sink=bridge.action_sink,
                    pain_bus=self.instance.pain_bus,
                    # Persistent agent: its session-end is the explicit
                    # full consolidation, never the sim default.
                    consolidation="full",
                )
            except Exception:
                logger.exception("talk loop failed for agent %r", self.agent_id)

        thread = threading.Thread(target=_worker, name=f"console.talk.{self.agent_id}", daemon=True)
        self._talk_bridge, self._talk_thread, self._talk_stop, self._talk_worker = (
            bridge,
            thread,
            stop_event,
            worker,
        )
        thread.start()
        return bridge

    def _stop_talk_loop(self, *, join_s: float = 20.0, required: bool = False) -> None:
        """End the talk loop if one is running (idempotent).

        ``required=True`` RAISES if the loop has not exited within ``join_s``.
        Callers about to start a second loop on the same bio-stack must pass
        it: proceeding anyway is the exact "two agent loops on one substrate"
        hazard this exists to prevent, and the straggler's session-end would
        stop the SHARED hippocampus capture worker mid-campaign (review
        finding). ``stop()`` keeps the proceed-loudly default — a hung loop
        must not make shutdown unbounded.
        """
        with self._talk_lock:
            bridge, thread, stop_event, worker = (
                self._talk_bridge,
                self._talk_thread,
                self._talk_stop,
                self._talk_worker,
            )
            self._talk_bridge = self._talk_thread = self._talk_stop = self._talk_worker = None
        if bridge is not None:
            bridge.finish()  # ConversationalSource is_exhausted() → loop exits
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_s)
            if thread.is_alive():
                if required:
                    raise RuntimeError(
                        f"talk loop did not exit within {join_s:.0f}s — refusing to start a second "
                        "loop on the same bio-stack (try again once the turn settles)"
                    )
                logger.warning("talk loop did not exit within %.0fs — proceeding", join_s)
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                logger.debug("talk LLMWorker.stop() raised", exc_info=True)

    def _run_sim(self, **kwargs: Any) -> Any:
        """Shared sim-invocation body: one-at-a-time lock, non-interactive
        forcing, and the tool-lease safety net. Both adventure flavors
        (authored campaign / generative premise) route through here so the
        lease + stdin discipline cannot drift between them."""
        if not self._campaign_lock.acquire(blocking=False):
            raise RuntimeError("A campaign is already running on this handle (one at a time)")
        try:
            # A live talk loop and a campaign would be TWO agent loops driving
            # one bio-stack (shared Hippocampus/NAc/executor) — the campaign
            # also leases the tool registry out from under the talk loop. Stop
            # talk first; the next talk() lazily rebuilds it. required=True:
            # if it will not stop, fail the run rather than run both.
            self._stop_talk_loop(required=True)
            # Headless surface: force non-interactive so the sim never grabs
            # stdin (the console serves this from a FastAPI worker thread).
            # Prior mode is restored on exit — a library user embedding the
            # handle next to interactive CLI usage keeps their setting.
            from maxim.simulation.sim_logger import (
                InteractiveMode,
                get_interactive_mode,
                set_interactive_mode,
            )

            _prior_mode = get_interactive_mode()
            set_interactive_mode(InteractiveMode.OFF)

            from maxim.simulation.orchestrator import _CampaignToolLease, start_simulation_mode

            # Exception-path safety net (three-lens review, cross-confirmed
            # Executor #2 / Architecture #2): the orchestrator restores its
            # tool lease on the normal path, but a mid-sim raise would
            # otherwise leave the LIVE persistent registry stripped of its
            # own tools + polluted with sim tools for every later mode.
            # restore() is an idempotent snapshot-diff, so running it again
            # after a clean finish is a no-op.
            _safety_lease = _CampaignToolLease.snapshot(self.instance.tool_registry)
            try:
                return start_simulation_mode(persistent_agent=self.instance, **kwargs)
            finally:
                dropped, restored = _safety_lease.restore(self.instance.tool_registry)
                if dropped or restored:
                    logger.warning(
                        "Campaign exited without a clean lease restore — safety net dropped %d "
                        "leaked campaign tools and re-registered %d persistent tools",
                        len(dropped),
                        len(restored),
                    )
                set_interactive_mode(_prior_mode)
        finally:
            self._campaign_lock.release()

    def rest(self, *, cluster: bool = True) -> dict[str, int]:
        """Consolidate memory WITHOUT tearing the agent down (the third mode).

        The distinction from :meth:`stop` is the whole point: ``stop`` ends the
        session and the handle is finished, while ``rest`` is sleep — the agent
        consolidates and remains usable, so a subsequent ``talk`` sees the
        consolidated substrate. That is what makes rest a *mode* rather than an
        alias for shutdown.

        Uses ``sleep_with_clustering`` when the SCN is wired (temporal-cluster
        consolidation — far cheaper on a large store than per-memory
        evaluation), falling back to plain ``sleep``.

        Returns the consolidation counts (compressed / removed / preserved /
        promoted) so a caller can show what rest actually did rather than a
        spinner that claims something happened.
        """
        if self._stopped:
            raise RuntimeError("MaximHandle is stopped — build a new handle to rest")
        if self._campaign_lock.locked():
            raise RuntimeError("An adventure is running on this handle — rest is unavailable until it ends")

        hippocampus = getattr(self.instance, "hippocampus", None)
        if hippocampus is None:
            return {}

        from maxim.simulation.sim_logger import sim_log

        # A live talk loop holds working state over the same substrate;
        # consolidating under it would race the loop's own reads. Stop it —
        # the next talk() rebuilds it lazily against the consolidated store.
        self._stop_talk_loop(required=True)

        sim_log("LEARN", "resting — consolidating memory", {"mode": "rest"}, agent_id=self.agent_id)
        use_clustering = bool(cluster and getattr(hippocampus, "scn", None) is not None)
        results = hippocampus.sleep_with_clustering() if use_clustering else hippocampus.sleep()
        results = dict(results or {})
        sim_log(
            "LEARN",
            "rest complete: " + ", ".join(f"{k}={v}" for k, v in sorted(results.items())) or "rest complete",
            {"mode": "rest", "clustered": use_clustering, **results},
            agent_id=self.agent_id,
        )
        return results

    # ── lifecycle ───────────────────────────────────────────────────────

    def stop(
        self,
        *,
        consolidation: Literal["full", "lightweight"] = "full",
        campaign_wait_s: float = 60.0,
    ) -> None:
        """Shut the persistent agent down with an EXPLICIT consolidation flavor.

        ``"full"`` (default): blocking sleep/replay consolidation +
        hippocampus/NAc/cerebellum saves — the right flavor for a persistent
        agent. ``"lightweight"`` skips the replay (still persists state).
        Idempotent: a second stop is a no-op.

        Waits up to ``campaign_wait_s`` for a live campaign to finish (post-
        merge review, Exec #4 / Arch #1: stopping under a live loop raced its
        own session-end — the MemoryHub's atomic test-and-clear now guarantees
        single consolidation either way, but stopping mid-campaign would still
        steal the loop's consolidation slot, so we wait). On expiry, proceeds
        LOUDLY — a hung campaign must not make stop() unbounded.
        """
        if self._stopped:
            return
        acquired = self._campaign_lock.acquire(timeout=campaign_wait_s)
        if not acquired:
            logger.warning(
                "MaximHandle.stop(): campaign still running after %.0fs wait — "
                "proceeding with shutdown; the campaign loop's own session-end "
                "becomes a no-op (atomic session flag)",
                campaign_wait_s,
            )
        try:
            if self._stopped:  # settled while we waited
                return
            self._stopped = True
            # End the live talk loop BEFORE shutdown so its own session-end
            # settles first (MemoryHub's atomic test-and-clear makes the second
            # consolidation a no-op either way, but a loop still touching the
            # bio-stack during shutdown is a race worth not having).
            self._stop_talk_loop()
            self.instance.shutdown(consolidation=consolidation)
        finally:
            if acquired:
                self._campaign_lock.release()
