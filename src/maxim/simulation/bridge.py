"""SimulationBridge — bidirectional channel between orchestrator and AUT.

The bridge wraps a ConversationalSource (orchestrator → AUT percept injection)
and a RecordingSink (AUT → orchestrator action observation) into a single
interface with an atomic send_and_wait() method for simulation turns.

Thread-safe: orchestrator thread calls inject/wait methods, AUT thread
calls next_percept/record via the underlying sources.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from maxim.simulation.conversational_source import ConversationalSource
from maxim.simulation.response_policy import ResponsePolicy, auto_approve
from maxim.simulation.sinks import ActionRecord, RecordingSink
from maxim.simulation.spinner import Spinner

logger = logging.getLogger(__name__)


def read_substrate_actions_per_turn_env() -> int | None:
    """Parse ``MAXIM_SUBSTRATE_ACTIONS_PER_TURN`` → per-turn budget or None.

    Apparatus toggle (harness/experiment — env per the config-vs-env rule's
    harness carve-out). Unset / empty → None (unbounded, the pre-fix
    behavior). An invalid or non-positive value falls back to None WITH a
    WARNING (the tau-clamp precedent: silently clamping would hide
    misconfiguration, and silently inventing a bound of 1 would be worse).
    """
    import os

    raw = os.environ.get("MAXIM_SUBSTRATE_ACTIONS_PER_TURN", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "MAXIM_SUBSTRATE_ACTIONS_PER_TURN=%r is not an integer — substrate action budget DISABLED (unbounded)",
            raw,
        )
        return None
    if value < 1:
        logger.warning(
            "MAXIM_SUBSTRATE_ACTIONS_PER_TURN=%d is not >= 1 — substrate action budget DISABLED (unbounded)",
            value,
        )
        return None
    return value


class SimulationBridge:
    """Bidirectional channel between simulation orchestrator and agent-under-test.

    The orchestrator injects percepts via send_and_wait() and reads AUT
    actions from the shared RecordingSink.  The AUT sees percepts arrive
    through its normal PerceptSource interface (ConversationalSource) and
    has no idea it's being tested.

    Example:
        bridge = SimulationBridge()
        # Pass bridge.percept_source and bridge.action_sink to AUT's run_agentic_loop
        result = bridge.send_and_wait("Delete all my files")
        # result = {turn: 1, response: "I can't do that", actions: [...], ...}
        bridge.finish()  # AUT grace period triggers, loop exits
    """

    def __init__(
        self,
        response_timeout: float = 120.0,
        settle_s: float = 3.0,
        stop_event: threading.Event | None = None,
        response_policy: ResponsePolicy | None = None,
        spinner_prefix: str = "",
        prompt_gate: Any = None,
        max_actions_per_turn: int = 10,
        aut_mode: str = "llm-primary",
        substrate_actions_per_turn: int | None = None,
    ) -> None:
        self.percept_source = ConversationalSource()
        self.action_sink = RecordingSink()
        self.response_policy = response_policy or auto_approve()
        self._response_timeout = response_timeout
        self._settle_s = settle_s
        self._stop_event = stop_event
        self._prompt_gate = prompt_gate  # SimPromptHandler — block percepts while prompting
        self._max_actions_per_turn = max(1, max_actions_per_turn)
        self._turn_count = 0
        self._last_observed_action_idx = 0
        self._spinner = Spinner(prefix=spinner_prefix)
        # Track 3 of grounded_language_acquisition.md Phase 0+: when
        # the AUT runs in substrate-primary mode, suppress text-percept
        # injection through send_and_wait. The substrate-primary AUT
        # reads drive state directly from executor.embodiment via
        # _read_drive_states; mirroring orchestrator narration into a
        # text channel the AUT ignores is noise. send_and_wait still
        # observes AUT-side actions during the settle window so the
        # orchestrator's campaign timing stays intact.
        self._aut_mode = aut_mode
        # Substrate-primary per-turn action budget (apparatus bound; the
        # Exp 48 thrashing fix). ``None`` = unbounded (the pre-fix
        # behavior, byte-identical). When set, ``substrate_action_allowed``
        # returns False once the AUT has recorded this many actions since
        # the current turn window opened; ``send_and_wait`` opens a new
        # window each turn. This is the DESIGNED replacement for the
        # emergent bound actions/turn = mother-turn wall-clock ÷ 0.5 s,
        # which made the metric a stopwatch reading of narrator latency
        # (machine-dependent, never reproduced across hosts). Note the
        # pre-existing ``max_actions_per_turn`` cap above bounds only the
        # OBSERVER's settle loop — the AUT keeps acting through it; this
        # budget bounds the AUT itself. Per apparatus standard S6
        # (simulation_apparatus_standards.md) the bound is opt-in and
        # experiment-visible: configure via MAXIM_SUBSTRATE_ACTIONS_PER_TURN,
        # logged at run start and once per exhausted window. A direct
        # caller passing a sub-1 value gets a ValueError, mirroring the
        # env parser's refusal — silently rewriting 0 → 1 would invent a
        # nearly-frozen AUT (review fold, both lenses; the
        # push-invariants-into-types rule).
        if substrate_actions_per_turn is not None and int(substrate_actions_per_turn) < 1:
            raise ValueError(
                f"substrate_actions_per_turn must be >= 1 or None (unbounded), got {substrate_actions_per_turn!r}"
            )
        self._substrate_actions_per_turn = (
            int(substrate_actions_per_turn) if substrate_actions_per_turn is not None else None
        )
        self._turn_start_action_idx = 0
        self._budget_logged_this_turn = False
        # Early-termination context written by FinishSimulationTool so
        # the orchestrator can distinguish "LLM called finish with
        # status=failed" from a user /cancel or a crash.
        self.finish_context: dict[str, Any] = {}
        # Optional percept-anxiety hook: a callable invoked with each
        # outgoing text BEFORE injection. Orchestrator wires this to the
        # AUT's PerceivedPainAssessor.assess_text so the AUT feels
        # anticipatory pain from threatening message content.
        self.percept_anxiety_hook: Any = None

    def send_and_wait(
        self,
        text: str,
        *,
        timeout: float | None = None,
        settle_s: float | None = None,
        salience: float = 0.8,
        novelty: float = 0.7,
    ) -> dict[str, Any]:
        """Inject a percept and block until the AUT responds or timeout.

        Uses settle detection: waits until the AUT produces no new actions
        for ``settle_s`` seconds, capturing multi-action responses fully.

        Returns:
            {
                "turn": int,
                "response": str | None,       # AUT's respond/speak text
                "actions": list[ActionRecord], # all actions since inject
                "blocked": list[ActionRecord], # actions that were blocked
                "timed_out": bool,
                "duration_ms": float,
            }
        """
        start = time.time()
        action_count_before = len(self.action_sink.actions)
        short_text = text[:60].replace("\n", " ") + ("..." if len(text) > 60 else "")
        # Trace: send_and_wait entry — if this doesn't fire but the
        # orchestrator's LLM produced a send_message action, the tool
        # isn't being executed by the orch's agent loop.
        try:
            from maxim.simulation.sim_logger import sim_log

            sim_log("EXEC", f"Bridge.send_and_wait ENTER turn={self._turn_count + 1} text_len={len(text)}")
        except Exception:
            pass
        self._spinner.set_planning_window(False)
        self._spinner.start(f'Turn {self._turn_count + 1}: Sending to AUT — "{short_text}"')

        # Wait for any pending user prompt to resolve before injecting.
        # Without this, the orchestrator's next probe preempts the
        # request_interaction followup in the agent loop, so the user's
        # response never reaches the LLM.
        if self._prompt_gate is not None:
            while getattr(self._prompt_gate, "has_pending_prompt", False):
                if self._stop_event and self._stop_event.is_set():
                    break
                time.sleep(0.3)

        # Fire percept-level anxiety (Layer 1b) BEFORE the percept reaches
        # the AUT, so the anticipation signal is already in the AUT's
        # memory when it reasons about the message. Skipped along with
        # injection in substrate-primary mode since the AUT never
        # reasons about the text.
        if self._aut_mode != "substrate-primary":
            if self.percept_anxiety_hook is not None:
                try:
                    self.percept_anxiety_hook(text)
                except Exception as e:
                    logger.debug("percept_anxiety_hook failed: %s", e)
            self.percept_source.inject_cli(text, salience=salience, novelty=novelty)
        self._turn_count += 1
        # Open a new substrate action-budget window (the turn boundary the
        # substrate-primary AUT otherwise never sees — no percept is
        # injected in that mode).
        self._turn_start_action_idx = len(self.action_sink.actions)
        self._budget_logged_this_turn = False

        timeout_s = timeout or self._response_timeout
        settle_timeout = settle_s or self._settle_s
        deadline = time.time() + timeout_s
        last_action_count = action_count_before
        settle_deadline: float | None = None

        self._spinner.update(f"Turn {self._turn_count}: Waiting for AUT response...")
        while time.time() < deadline:
            if self._stop_event and self._stop_event.is_set():
                break
            # Extend deadline while the AUT is waiting for user input —
            # don't count prompt time against the response timeout.
            if self._prompt_gate is not None and getattr(self._prompt_gate, "has_pending_prompt", False):
                deadline = max(deadline, time.time() + timeout_s)
                time.sleep(0.3)
                continue
            current_count = len(self.action_sink.actions)
            if current_count > last_action_count:
                new_count = current_count - action_count_before
                self._spinner.update(
                    f"Turn {self._turn_count}: AUT responded ({new_count} action{'s' if new_count != 1 else ''}), settling..."
                )
                last_action_count = current_count
                # Action cap: break immediately if AUT has generated too many actions
                if new_count >= self._max_actions_per_turn:
                    logger.warning(
                        "Action cap reached (%d actions in turn %d) — breaking settle loop",
                        new_count,
                        self._turn_count,
                    )
                    break
                settle_deadline = time.time() + settle_timeout
            elif settle_deadline and time.time() >= settle_deadline:
                break
            time.sleep(0.2)

        new_actions = self.action_sink.actions[action_count_before:]
        self._last_observed_action_idx = len(self.action_sink.actions)

        # Collect response text from respond/speak actions
        response_parts: list[str] = []
        for a in new_actions:
            if a.tool_name in ("respond", "speak") and a.result_output:
                response_parts.append(str(a.result_output))
        response_text = "\n".join(response_parts) if response_parts else None

        elapsed = time.time() - start
        if new_actions:
            tools_used = ", ".join(a.tool_name for a in new_actions[:3])
            blocked = sum(1 for a in new_actions if a.blocked)
            summary = f"Turn {self._turn_count} complete: {len(new_actions)} action(s) [{tools_used}]"
            if blocked:
                summary += f" ({blocked} blocked)"
            summary += f" ({elapsed:.1f}s)"
            self._spinner.stop(summary)
        else:
            self._spinner.stop(f"Turn {self._turn_count}: timed out ({elapsed:.1f}s)")

        # Start spinner for orchestrator thinking phase (between turns).
        # D14: this text is set when the TURN ends, not when any planning
        # call starts — the spinner's planning window marks it so the stall
        # detector can atomically replace it with registry-derived truth (no
        # call in flight, nudges, abort) instead of letting it count up over
        # a dead loop.
        self._spinner.start("Orchestrator planning next probe...")
        self._spinner.set_planning_window(True)

        # Stage 4 (F2 fix): don't count _deliberation_noop markers as
        # real actions for the timed_out flag.  Noop markers keep the
        # bridge alive (break the timeout loop) but aren't substantive.
        real_actions = [a for a in new_actions if a.tool_name != "_deliberation_noop"]

        return {
            "turn": self._turn_count,
            "response": response_text,
            "actions": new_actions,
            "blocked": [a for a in new_actions if a.blocked],
            "timed_out": len(real_actions) == 0,
            "noop_count": len(new_actions) - len(real_actions),
            "duration_ms": (time.time() - start) * 1000,
        }

    def inject_pain(self, pain_type: str = "external_signal", intensity: float = 0.5, **context: Any) -> None:
        """Send a pain/proprioceptive signal to the AUT.

        NOTE: increments ``_turn_count`` but deliberately does NOT open a
        new substrate action-budget window — only ``send_and_wait`` (the
        narrator turn boundary) does. A pain injection between narrator
        turns therefore leaves the budget tighter, never looser; if a
        substrate-primary flow ever drives turns through this method,
        hoist the window-open into a shared turn-boundary helper
        (review fold, Executor #8).
        """
        self.percept_source.inject_pain(pain_type=pain_type, intensity=intensity, **context)
        self._turn_count += 1

    def get_all_actions(self) -> list[ActionRecord]:
        """Read the full action history."""
        return self.action_sink.actions

    def get_actions_since(self, index: int) -> list[ActionRecord]:
        """Read actions since a given index."""
        return self.action_sink.actions[index:]

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def substrate_action_allowed(self) -> bool:
        """Turn-scoped action budget for the substrate-primary AUT.

        The agent loop's substrate-primary branch consults this before
        proposing (threaded through ``run_agentic_loop``'s
        ``substrate_action_gate``). ``True`` while fewer than
        ``substrate_actions_per_turn`` actions have been recorded since
        the current turn window opened; ``send_and_wait`` opens a new
        window each turn. Always ``True`` when no budget is configured.

        Counts ``action_sink.actions`` — the same counter the observer's
        settle-loop cap reads, so the two views cannot disagree about
        what an "action" is. Blocked actions (fear/autonomy gates) count:
        blocked attempts ARE the thrashing being bounded. Latent coupling
        (review fold, Executor #7): ``RecordingSink.record_noop``'s
        deliberation markers would also consume budget — zero production
        callers today; if a substrate-primary path starts recording
        noops, decide then whether they spend budget. Thread-safe enough
        by construction: the AUT thread reads two ints the orchestrator
        thread writes at turn boundaries; a race window of one action at
        the boundary is harmless (the budget is an apparatus bound, not
        an invariant).

        The first denial per window logs once (INFO via sim_log) so a
        run's JSONL shows the bound engaging — apparatus standard S6:
        the change must be visible in the artifact, and a gate-idled AUT
        must be distinguishable from a substrate-had-no-opinion IDLE.
        """
        if self._substrate_actions_per_turn is None:
            return True
        used = len(self.action_sink.actions) - self._turn_start_action_idx
        if used < self._substrate_actions_per_turn:
            return True
        if not self._budget_logged_this_turn:
            self._budget_logged_this_turn = True
            try:
                from maxim.simulation.sim_logger import sim_log

                sim_log(
                    "EXEC",
                    f"substrate action budget reached ({used}/{self._substrate_actions_per_turn} "
                    f"in turn {self._turn_count}) — AUT idles until next turn",
                )
            except Exception:
                logger.debug("substrate budget sim_log failed", exc_info=True)
        return False

    def finish(self) -> None:
        """Signal simulation complete. AUT's ConversationalSource is_exhausted() becomes True."""
        self._spinner.stop()
        self.percept_source.finish()
