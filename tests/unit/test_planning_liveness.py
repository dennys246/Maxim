"""Planning-liveness guards (bugs ledger D13 + D14).

D13: a planning submit that ends in parse-failure/exhaustion (or whose exact
worker job terminates without a proposal) must LOUDLY reschedule with bounded retries or abort the sim
— never fall through to idle. Pre-fix, a dropped fallback/error/stale
proposal was terminal: ``pending_action_followup`` was already cleared and
the triggering input deduped when the failed request was BUILT, so nothing
ever re-armed the loop (traced live 2026-08-18: narrator returned status 200,
tool-parse failed, zero further backend calls, orchestrator idle forever).

D14: the between-turns spinner must report OBSERVED state, never intent —
pre-fix it showed "planning... (8624s)" over a
loop py-spy proved was doing nothing.

Test strategy follows house style for the 3,000-line loop (see
test_substrate_action_budget.py): pure decision seams get behavioral tests;
the loop wiring gets structural pins on the source; the worker requeue path
gets a real LLMWorker + FakeLLM behavioral test.
"""

from __future__ import annotations

import inspect
import pathlib
import threading
import time
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_context(triggering_input: str):
    """A REAL StructuredContext — the parse-failure path must be reached
    through production prompt building, not short-circuited by an
    incomplete fake raising AttributeError into the catch-all branch."""
    from maxim.agents.bus import StructuredContext

    return StructuredContext(timestamp=time.time(), cli_inputs=[triggering_input])


class NoneLLM:
    """LLM backend whose every response fails to parse (returns None) —
    the exact D13 trigger: HTTP 200, empty/unparseable tool response."""

    def __init__(self):
        self.call_count = 0
        self.prompts: list[str] = []

    def generate_json(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024, **kwargs):
        self.call_count += 1
        self.prompts.append(prompt)
        return None


class BlockingLLM:
    """Valid backend whose release is controlled by the test."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def generate_json(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024, **kwargs):
        self.started.set()
        if not self.release.wait(timeout=4.0):
            raise TimeoutError("test did not release BlockingLLM")
        return {
            "action": {"tool_name": "respond", "params": {"message": "done"}},
            "reasoning": "valid response",
            "confidence": 0.9,
        }


def _make_mode_info():
    from maxim.agents.llm_worker import ModeInfo

    return ModeInfo(
        name="active",
        goal="assist the user",
        context_prompt="You are a helpful assistant.",
        max_response_tokens=512,
        context_window_tokens=2048,
    )


def _submit_test_context(worker, triggering_input: str = "probe the AUT") -> bool:
    from maxim.agents.autonomy import AutonomyLevel

    return worker.submit_context(
        context=_make_context(triggering_input),
        mode=_make_mode_info(),
        autonomy_level=AutonomyLevel.SUPERVISED,
        internet_access=False,
        internet_policy_summary="",
        triggering_input=triggering_input,
        use_tool_prompting=True,
        available_tools={"respond"},
        tool_descriptions={"respond": "Send a message"},
    )


def _wait_for_proposal(worker, timeout_s: float = 4.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proposal = worker.get_latest_proposal()
        if proposal is not None:
            return proposal
        time.sleep(0.05)
    return None


def _wait_for_attempt_state(worker, expected, timeout_s: float = 4.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = worker.latest_attempt_state()
        if state is expected:
            return state
        time.sleep(0.02)
    return worker.latest_attempt_state()


def _make_ctrl():
    """Real LoopController with mocked dependencies — the planning-liveness
    fields and methods are what's under test."""
    from maxim.agents.autonomy import AutonomyController
    from maxim.runtime.loop_controller import LoopController

    return LoopController(
        MagicMock(),  # agent
        MagicMock(),  # environment
        MagicMock(),  # state
        MagicMock(),  # memory
        MagicMock(),  # decision_engine
        MagicMock(),  # executor
        autonomy_controller=MagicMock(spec=AutonomyController),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. LoopController bounded-retry state machine
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordPlanningFailure:
    def test_within_budget_returns_retry(self):
        ctrl = _make_ctrl()
        assert ctrl.planning_retry_limit == 3
        for expected_streak in (1, 2, 3):
            assert ctrl.record_planning_failure(reason="test") == "retry"
            assert ctrl.planning_failure_streak == expected_streak
        assert ctrl.planning_exhausted is False

    def test_budget_spent_returns_exhausted_and_latches(self):
        ctrl = _make_ctrl()
        for _ in range(3):
            ctrl.record_planning_failure(reason="test")
        assert ctrl.record_planning_failure(reason="test") == "exhausted"
        assert ctrl.planning_exhausted is True
        # Latched: further failures never re-log or re-retry.
        assert ctrl.record_planning_failure(reason="test") == "already_exhausted"

    def test_reset_on_executable_proposal(self):
        ctrl = _make_ctrl()
        ctrl.record_planning_failure(reason="test")
        ctrl.record_planning_failure(reason="test")
        ctrl.reset_planning_failures()
        assert ctrl.planning_failure_streak == 0
        assert ctrl.planning_exhausted is False
        # Fresh budget after recovery.
        assert ctrl.record_planning_failure(reason="test") == "retry"

    def test_liveness_fields_default(self):
        ctrl = _make_ctrl()
        assert ctrl.last_proposal_time == 0.0
        assert ctrl.planning_failure_streak == 0
        assert ctrl.planning_transport_failure_streak == 0
        assert ctrl.planning_exhausted is False

    def test_planning_failure_preserves_typed_finish_status(self):
        ctrl = _make_ctrl()
        for _ in range(3):
            ctrl.record_planning_failure(reason="bad_tool", exhausted_status="planning_failed")
        assert ctrl.record_planning_failure(reason="bad_tool", exhausted_status="planning_failed") == "exhausted"
        assert ctrl.planning_exhausted_status == "planning_failed"
        assert ctrl.planning_exhausted_reason == "bad_tool"

    def test_transport_budget_is_separate_and_bounded(self):
        ctrl = _make_ctrl()
        assert ctrl.planning_transport_retry_limit == 3
        for expected_streak in (1, 2, 3):
            assert ctrl.record_planning_transport_failure(reason="queue_full") == "retry"
            assert ctrl.planning_transport_failure_streak == expected_streak
        assert ctrl.planning_failure_streak == 0
        assert ctrl.record_planning_transport_failure(reason="queue_full") == "exhausted"
        assert ctrl.planning_exhausted_status == "worker_unavailable"


# ─────────────────────────────────────────────────────────────────────────────
# 2. _handle_planning_failure — the loop's failure handler
# ─────────────────────────────────────────────────────────────────────────────


class TestHandlePlanningFailure:
    def _sim(self):
        sim = MagicMock()
        sim.is_sim_mode = True
        return sim

    def test_retry_requeues_original_request_and_reopens_window(self):
        from maxim.runtime.agent_loop import _handle_planning_failure

        ctrl = _make_ctrl()
        worker = MagicMock()
        worker.requeue_request.return_value = True
        original = MagicMock(name="original_request")

        before = time.time()
        exhausted = _handle_planning_failure(
            ctrl, worker, self._sim(), reason="fallback_proposal_dropped", original_request=original
        )

        assert exhausted is False
        worker.requeue_request.assert_called_once_with(original)
        worker.requeue_last_request.assert_not_called()
        # The await window MUST re-open or the idle gate closes at 120s
        # and the loop falls back into the pre-fix livelock.
        assert ctrl.last_llm_submit_time >= before

    def test_retry_without_request_uses_last_request_backstop(self):
        from maxim.runtime.agent_loop import _handle_planning_failure

        ctrl = _make_ctrl()
        worker = MagicMock()
        worker.requeue_last_request.return_value = True

        exhausted = _handle_planning_failure(
            ctrl,
            worker,
            self._sim(),
            reason="planning_job_completed_without_proposal",
            original_request=None,
        )

        assert exhausted is False
        worker.requeue_last_request.assert_called_once_with()
        worker.requeue_request.assert_not_called()

    def test_exhaustion_reports_abort_and_stops_requeueing(self):
        from maxim.runtime.agent_loop import _handle_planning_failure

        ctrl = _make_ctrl()
        worker = MagicMock()
        worker.requeue_request.return_value = True
        sim = self._sim()

        results = [
            _handle_planning_failure(
                ctrl, worker, sim, reason="fallback_proposal_dropped", original_request=MagicMock()
            )
            for _ in range(5)
        ]

        # 3 bounded retries, then exhausted (True), then latched (True).
        assert results == [False, False, False, True, True]
        assert worker.requeue_request.call_count == 3
        assert ctrl.planning_exhausted is True

    def test_failed_requeue_still_paces_the_window(self):
        """A rejected requeue still stamps the attempt for exact-state recovery."""
        from maxim.runtime.agent_loop import _handle_planning_failure

        ctrl = _make_ctrl()
        worker = MagicMock()
        worker.requeue_request.return_value = False

        before = time.time()
        exhausted = _handle_planning_failure(
            ctrl, worker, self._sim(), reason="stale_proposal_dropped", original_request=MagicMock()
        )
        assert exhausted is False
        assert ctrl.last_llm_submit_time >= before

    def test_bad_tool_retry_carries_explicit_correction(self):
        from maxim.runtime.agent_loop import _handle_planning_failure

        ctrl = _make_ctrl()
        worker = MagicMock()
        worker.requeue_request.return_value = True
        original = MagicMock(name="original_request")

        assert (
            _handle_planning_failure(
                ctrl,
                worker,
                self._sim(),
                reason="unregistered_tool_proposed",
                original_request=original,
                failed_tool="ghost_tool",
                exhausted_status="planning_failed",
            )
            is False
        )
        worker.requeue_request.assert_called_once_with(original, failed_tool="ghost_tool")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Worker: original_request attachment + requeue path (behavioral,
#    real LLMWorker + pool — the D13 parse-failure trigger end to end)
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkerParseFailurePath:
    def test_invalid_response_proposal_carries_original_request(self):
        from maxim.agents.llm_types import LLMRequest
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=NoneLLM(), stale_threshold_s=10.0)
        worker.start()
        try:
            assert _submit_test_context(worker) is True
            proposal = _wait_for_proposal(worker)
            assert proposal is not None
            # Fallback or error proposal — either way the original request
            # rides along so the loop can requeue instead of dropping.
            assert proposal.reasoning in ("llm_fallback", "LLM returned invalid response")
            assert isinstance(proposal.original_request, LLMRequest)
            assert proposal.original_request.triggering_input == "probe the AUT"
        finally:
            worker.stop()

    def test_requeue_request_produces_second_proposal(self):
        from maxim.agents.llm_worker import LLMWorker

        llm = NoneLLM()
        worker = LLMWorker(llm=llm, stale_threshold_s=10.0)
        worker.start()
        try:
            _submit_test_context(worker)
            first = _wait_for_proposal(worker)
            assert first is not None and first.original_request is not None

            assert worker.requeue_request(first.original_request) is True
            second = _wait_for_proposal(worker)
            assert second is not None
            assert llm.call_count >= 2
            # The retry is byte-identical: same request object resubmitted.
            assert second.request_id == first.request_id
        finally:
            worker.stop()

    def test_requeue_refreshes_timestamp_past_staleness_guard(self):
        """The staleness guard drops requests older than stale_threshold —
        a requeue that keeps the original timestamp would be dropped
        silently, turning the retry into a no-op."""
        from maxim.agents.llm_worker import LLMWorker

        llm = NoneLLM()
        worker = LLMWorker(llm=llm, stale_threshold_s=0.5)
        worker.start()
        try:
            _submit_test_context(worker)
            first = _wait_for_proposal(worker)
            assert first is not None
            time.sleep(0.6)  # original timestamp is now stale
            assert worker.requeue_request(first.original_request) is True
            second = _wait_for_proposal(worker)
            assert second is not None, "stale-guard swallowed the requeue"
        finally:
            worker.stop()

    def test_requeue_last_request_after_submit(self):
        from maxim.agents.llm_worker import LLMWorker

        worker = LLMWorker(llm=NoneLLM(), stale_threshold_s=10.0)
        worker.start()
        try:
            assert worker.requeue_last_request() is False, "nothing submitted yet"
            _submit_test_context(worker)
            _wait_for_proposal(worker)
            assert worker.requeue_last_request() is True
            assert _wait_for_proposal(worker) is not None
        finally:
            worker.stop()

    def test_bad_tool_retry_changes_the_next_prompt(self):
        from maxim.agents.llm_worker import LLMWorker

        llm = NoneLLM()
        worker = LLMWorker(llm=llm, stale_threshold_s=10.0)
        worker.start()
        try:
            assert _submit_test_context(worker) is True
            first = _wait_for_proposal(worker)
            assert first is not None and first.original_request is not None
            assert worker.requeue_request(first.original_request, failed_tool="ghost_tool") is True
            assert _wait_for_proposal(worker) is not None
            assert len(llm.prompts) >= 2
            assert "ghost_tool" not in llm.prompts[0]
            assert "previously called 'ghost_tool'" in llm.prompts[1]
        finally:
            worker.stop()

    def test_latest_attempt_tracks_running_completed_and_consumed(self):
        from maxim.agents.llm_worker import LLMAttemptState, LLMWorker

        llm = BlockingLLM()
        worker = LLMWorker(llm=llm, stale_threshold_s=10.0)
        worker.start()
        try:
            assert _submit_test_context(worker) is True
            assert llm.started.wait(timeout=2.0)
            assert worker.latest_attempt_state() is LLMAttemptState.RUNNING

            llm.release.set()
            assert _wait_for_attempt_state(worker, LLMAttemptState.COMPLETED) is LLMAttemptState.COMPLETED
            proposal = worker.get_latest_proposal()
            assert proposal is not None
            assert proposal.original_request is not None
            assert worker.latest_attempt_state() is LLMAttemptState.CONSUMED
        finally:
            llm.release.set()
            worker.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Slow model != lost turn (the false-abort guard)
# ─────────────────────────────────────────────────────────────────────────────


class TestSlowCallIsNotALostTurn:
    """The await window's 120s is a wall-clock GUESS a big model routinely
    exceeds — qwen2.5-32b measured ~140s/turn in the Exp 37 heartbeat. If
    the liveness backstop keyed on that literal alone it would requeue a
    healthy slow call (doubling load) and abort the campaign after 3 —
    killing exactly the big-model runs the fix is meant to protect. The
    discriminator must be the OBSERVED state of this worker's exact job."""

    @pytest.mark.parametrize("state_name", ["PENDING", "RUNNING", "COMPLETED"])
    def test_active_job_states_hold_the_gate_open(self, state_name):
        from maxim.agents.llm_worker import LLMAttemptState
        from maxim.runtime.agent_loop import _planning_attempt_is_active

        assert _planning_attempt_is_active(LLMAttemptState[state_name]) is True

    @pytest.mark.parametrize("state_name", ["NONE", "FAILED", "CANCELLED", "CONSUMED", "MISSING"])
    def test_terminal_or_absent_job_states_release_the_gate(self, state_name):
        from maxim.agents.llm_worker import LLMAttemptState
        from maxim.runtime.agent_loop import _planning_attempt_is_active

        assert _planning_attempt_is_active(LLMAttemptState[state_name]) is False

    def test_idle_gate_uses_exact_worker_state(self):
        """A running or completed-unconsumed job keeps section 2 reachable."""
        from maxim.runtime import agent_loop

        src = inspect.getsource(agent_loop.run_agentic_loop)
        gate = src.split("if not (", 1)[0]
        assert "llm_worker.latest_attempt_state()" in gate
        assert "_planning_attempt_is_active(_planning_attempt_state)" in gate
        assert "any_call_in_flight" not in gate

    def test_completed_state_is_active_until_proposal_poll(self):
        """The backstop cannot race the result queue publication."""
        from maxim.runtime import agent_loop

        src = inspect.getsource(agent_loop.run_agentic_loop)
        gate_idx = src.index("_planning_attempt_is_active(_planning_attempt_state)")
        backstop_idx = src.index("planning_job_completed_without_proposal")
        assert gate_idx < backstop_idx


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loop wiring pins (house style for run_agentic_loop — see
#    test_substrate_action_budget.py's source pins)
# ─────────────────────────────────────────────────────────────────────────────


class TestLoopWiringPins:
    @pytest.fixture(scope="class")
    def loop_src(self):
        from maxim.runtime import agent_loop

        return inspect.getsource(agent_loop.run_agentic_loop)

    def test_fallback_drop_calls_handler(self, loop_src):
        assert "fallback_proposal_dropped" in loop_src

    def test_error_drop_calls_handler_and_excludes_shutdown(self, loop_src):
        assert "proposal_error:" in loop_src
        assert 'new_proposal.error != "shutdown"' in loop_src

    def test_stale_drop_calls_handler(self, loop_src):
        assert "stale_proposal_dropped" in loop_src

    def test_idle_gate_has_terminal_job_backstop(self, loop_src):
        assert "planning_job_completed_without_proposal" in loop_src
        # The backstop keys on "nothing came back since the last submit".
        assert "ctrl.last_proposal_time < ctrl.last_llm_submit_time" in loop_src

    def test_no_action_no_error_proposal_calls_handler(self, loop_src):
        from maxim.runtime.agent_loop import _proposal_without_action_reason

        assert "_proposal_without_action_reason(new_proposal)" in loop_src
        assert "proposal_without_action" in inspect.getsource(_proposal_without_action_reason)

    def test_single_gate_covers_every_failure_site(self, loop_src):
        """All outcome sites consult ONE precomputed gate, so none can drift
        (arch lens S7: the substrate-primary exclusion was previously only
        incidental — an aut_llm_worker IS built in substrate-primary runs)."""
        assert loop_src.count("_planning_liveness_on") >= 5
        gate_def = loop_src.split("_planning_liveness_on = (", 1)[1].split("\n    )", 1)[0]
        assert "planning_liveness" in gate_def
        assert 'aut_mode != "substrate-primary"' in gate_def
        assert "llm_worker is not None" in gate_def
        # No site may re-derive the gate from is_sim_mode.
        assert "sim.is_sim_mode\n                        and new_proposal.error" not in loop_src

    def test_liveness_is_opt_in_and_defaults_off(self):
        """The mechanism belongs to the ONE loop with no other wake source.
        Defaulting it on would convert the AUT's existing recovery into an
        abort (arch lens B3) and its thread death into fabricated empty
        turns (B1)."""
        from maxim.runtime.agent_loop import run_agentic_loop

        param = inspect.signature(run_agentic_loop).parameters["planning_liveness"]
        assert param.default is False

    def test_orchestrator_opts_in_and_aut_does_not(self):
        from maxim.simulation import orchestrator

        src = inspect.getsource(orchestrator.start_simulation_mode)
        assert src.count("planning_liveness=True") == 1
        # The opt-in must sit on the orchestrator's own loop call, which is
        # the one passing percept_source=orchestrator_source.
        orch_call = src.split("percept_source=orchestrator_source", 1)[1][:1400]
        assert "planning_liveness=True" in orch_call

    def test_exhaustion_raises_after_teardown(self, loop_src):
        # The raise must come AFTER _end_bio_session so state persistence
        # and the bio session end run before the sim aborts.
        teardown_idx = loop_src.rindex("_end_bio_session(")
        raise_idx = loop_src.rindex("raise PlanningLivenessExhausted(")
        assert raise_idx > teardown_idx

    def test_proposal_time_stamped_on_any_proposal(self, loop_src):
        assert "ctrl.last_proposal_time = time.time()" in loop_src

    def test_global_call_registry_does_not_control_loop_liveness(self, loop_src):
        assert "any_call_in_flight" not in loop_src

    def test_streak_reset_is_enforced_by_the_type(self):
        """Reset used to live at ONE install site while three others bypassed
        it (executor lens F2). It is now a consequence of assigning
        pending_proposal, so a future install site cannot forget it."""
        from maxim.runtime.loop_controller import LoopController

        assert isinstance(LoopController.pending_proposal, property)
        ctrl = _make_ctrl()
        ctrl.record_planning_failure(reason="test")
        ctrl.record_planning_failure(reason="test")
        assert ctrl.planning_failure_streak == 2
        ctrl.pending_proposal = MagicMock(name="executable_proposal")
        assert ctrl.planning_failure_streak == 0
        # Clearing must NOT count as recovery.
        ctrl.record_planning_failure(reason="test")
        ctrl.pending_proposal = None
        assert ctrl.planning_failure_streak == 1


class TestProposalWithoutAction:
    def test_action_or_error_is_executable_or_explicit(self):
        from maxim.runtime.agent_loop import _proposal_without_action_reason

        assert _proposal_without_action_reason(MagicMock(action={"tool_name": "respond"}, error=None)) is None
        assert _proposal_without_action_reason(MagicMock(action=None, error="bad response")) is None

    def test_parsed_empty_proposal_is_a_failure(self):
        from maxim.runtime.agent_loop import _proposal_without_action_reason

        proposal = MagicMock(action=None, error=None, ready_to_act=True, mode_goal_achieved=False)
        assert _proposal_without_action_reason(proposal) == "proposal_without_action"

    def test_not_ready_proposal_is_classified(self):
        from maxim.runtime.agent_loop import _proposal_without_action_reason

        proposal = MagicMock(action=None, error=None, ready_to_act=False, mode_goal_achieved=False)
        assert _proposal_without_action_reason(proposal) == "proposal_not_ready_to_act"


# ─────────────────────────────────────────────────────────────────────────────
# 5. D14: spinner truth decision (pure) + wiring pins
# ─────────────────────────────────────────────────────────────────────────────


class TestSpinnerTruthMessage:
    def _call(self, **overrides):
        from maxim.simulation.spinner import spinner_truth_message

        kwargs = dict(
            between_turns=True,
            in_flight_and_silent=False,
            stall_duration_s=200.0,
            threshold_s=30.0,
            nudge_count=0,
            byte_silence_s=None,
            byte_silence_threshold_s=90.0,
        )
        kwargs.update(overrides)
        return spinner_truth_message(**kwargs)

    def test_mid_turn_never_overrides(self):
        assert self._call(between_turns=False) is None

    def test_healthy_in_flight_call_keeps_default_text(self):
        assert self._call(in_flight_and_silent=True, byte_silence_s=5.0) is None

    def test_short_gap_keeps_default_text(self):
        assert self._call(stall_duration_s=10.0) is None

    def test_no_call_in_flight_past_threshold_tells_the_truth(self):
        msg = self._call(stall_duration_s=3286.0)
        assert msg is not None
        assert "no LLM call in flight" in msg
        assert "3286" in msg

    def test_nudges_appear_in_message(self):
        msg = self._call(stall_duration_s=200.0, nudge_count=4)
        assert msg is not None and "4 nudge(s)" in msg

    def test_wedged_in_flight_call_reports_byte_silence(self):
        msg = self._call(in_flight_and_silent=True, byte_silence_s=120.0)
        assert msg is not None
        assert "silent 120s" in msg


class TestSpinnerPlanningWindow:
    """D14 window ownership + the TOCTOU fix (executor lens F7, arch N5):
    the flag lives on the spinner, and the correction is a lock-guarded
    test-and-set so a turn starting mid-decision keeps its own text."""

    def test_default_closed_then_open_after_turn(self):
        from maxim.simulation.bridge import SimulationBridge

        bridge = SimulationBridge(response_timeout=0.2, settle_s=0.05)
        try:
            assert bridge._spinner.planning_window is False
            bridge.send_and_wait("probe")  # times out — no AUT on the other side
            assert bridge._spinner.planning_window is True
        finally:
            bridge.finish()
        assert bridge._spinner.planning_window is False

    def test_send_and_wait_entry_closes_window(self):
        from maxim.simulation import bridge as bridge_mod

        src = inspect.getsource(bridge_mod.SimulationBridge.send_and_wait)
        entry = src.split("self._spinner.start(", 1)[0]
        assert "set_planning_window(False)" in entry

    def test_update_if_planning_respects_the_window(self):
        from maxim.simulation.spinner import Spinner

        sp = Spinner()
        assert sp.update_if_planning("truth") is False, "closed window must reject"
        sp.set_planning_window(True)
        assert sp.update_if_planning("truth") is True
        assert sp._message == "truth"
        sp.set_planning_window(False)
        assert sp.update_if_planning("later") is False
        assert sp._message == "truth", "a closed window must not be overwritten"

    def test_transport_no_longer_carries_the_display_flag(self):
        from maxim.simulation.bridge import SimulationBridge

        bridge = SimulationBridge(response_timeout=0.2, settle_s=0.05)
        assert not hasattr(bridge, "between_turns")


class TestOrchestratorWiringPins:
    @pytest.fixture(scope="class")
    def orch_src(self):
        from maxim.simulation import orchestrator

        return inspect.getsource(orchestrator.start_simulation_mode)

    def test_stall_detector_consults_spinner_truth(self, orch_src):
        assert "spinner_truth_message(" in orch_src

    def test_orchestrator_preserves_typed_liveness_finish(self, orch_src):
        assert "except _PlanningLivenessExhausted" in orch_src
        assert 'getattr(e, "finish_status", "llm_wedged")' in orch_src
        assert '"status": _finish_status' in orch_src
        assert '"initiated_by": "planning_liveness"' in orch_src

    def test_abort_path_corrects_spinner_before_finish(self, orch_src):
        assert '_spinner.update(f"🛑 {_finish_status} — aborting sim")' in orch_src

    def test_519_abort_clock_is_nudge_proof(self, orch_src):
        """Guard-test debt from #519 (scorecard finding): the hard-abort's
        clock is written ONLY on turn advance — a nudge must not be able to
        reset it (pre-#519, nudges reset the clock and made the abort
        structurally unreachable: '>=3 nudges AND >=150s stall' were
        mutually exclusive by construction)."""
        writes = orch_src.count("_last_turn_progress_time[0] = time.time()")
        assert writes == 1, f"expected exactly one abort-clock write site, found {writes}"
        # And that one write is in the turn-advance branch, not the nudge path.
        turn_advance = orch_src.split("if current_turns > _last_turn_count[0]:", 1)[1]
        advance_block = turn_advance.split("continue", 1)[0]
        assert "_last_turn_progress_time[0] = time.time()" in advance_block
        nudge_block = orch_src.split("_nudge_count[0] += 1", 1)[1].split("def _force_exit", 1)[0]
        assert "_last_turn_progress_time" not in nudge_block


# ─────────────────────────────────────────────────────────────────────────────
# 6. Review-round fold guards (findings from the two-lens pre-merge round)
# ─────────────────────────────────────────────────────────────────────────────


class TestParseFailureVsBadToolChoice:
    """Executor F1 / architecture B2 (cross-confirmed): a well-formed proposal
    naming an unregistered tool is a RESPONSIVE model making a bad choice, not
    a lost planning turn. Calling it 'wedged' is a false diagnosis, and a
    byte-identical requeue just reproduces the same name."""

    def test_reasons_are_distinguished_at_the_call_site(self):
        from maxim.runtime import agent_loop

        src = inspect.getsource(agent_loop.run_agentic_loop)
        assert '_is_parse_failure = getattr(new_proposal, "reasoning", "") == "llm_fallback"' in src
        assert "unregistered_tool_proposed" in src
        assert "fallback_proposal_dropped" in src

    def test_bad_tool_name_is_recorded_for_correction(self):
        from maxim.runtime import agent_loop

        src = inspect.getsource(agent_loop.run_agentic_loop)
        block = src.split("_is_parse_failure = ", 1)[1].split("new_proposal = None", 1)[0]
        assert "_tools_hallucinated" in block, (
            "the unregistered-tool case must feed the existing corrective channel, "
            "not just be requeued byte-identically"
        )
        retry_call = src.split("reason=(", 1)[1].split("state.data.pop", 1)[0]
        assert "failed_tool=None if _is_parse_failure else _bad_tool" in retry_call
        assert '"planning_failed"' in retry_call


class TestTransportFailureIsSeparateAndBounded:
    """Executor S2: a requeue rejected because the worker queue is full is a
    TRANSPORT failure. It must not spend more planning strikes, but it also
    must not retry forever."""

    def _sim(self):
        sim = MagicMock()
        sim.is_sim_mode = True
        return sim

    def test_rejected_requeues_exhaust_the_transport_budget(self):
        from maxim.runtime.agent_loop import _handle_planning_failure, _handle_planning_transport_failure

        ctrl = _make_ctrl()
        worker = MagicMock()
        worker.requeue_request.return_value = False
        worker.requeue_last_request.return_value = False

        assert (
            _handle_planning_failure(
                ctrl,
                worker,
                self._sim(),
                reason="fallback_proposal_dropped",
                original_request=MagicMock(),
            )
            is False
        )
        results = [
            _handle_planning_transport_failure(ctrl, worker, self._sim(), reason="worker_job_failed") for _ in range(4)
        ]
        assert results == [False, False, False, True]
        assert ctrl.planning_failure_streak == 1
        assert ctrl.planning_transport_failure_streak == 4
        assert ctrl.planning_exhausted is True
        assert ctrl.planning_exhausted_status == "worker_unavailable"


class TestExactJobStateClosesPublicationRace:
    """Executor F5: provider return and result consumption are not simultaneous.
    COMPLETED stays active until get_latest_proposal consumes the queued job."""

    def test_backstop_has_no_timing_heuristic(self):
        from maxim.runtime import agent_loop

        src = inspect.getsource(agent_loop.run_agentic_loop)
        assert "_lost_turn_observations" not in src
        assert "LLMAttemptState.COMPLETED" in src
        assert "planning_job_completed_without_proposal" in src


class TestRequeueUsesFreshRequest:
    """Executor F6: a submit rejected on queue.Full is itself a lost turn; the
    backstop must requeue THAT request, not the previous answered one."""

    def test_last_request_is_stashed_before_the_submit_attempt(self):
        from maxim.agents import llm_worker as lw

        src = inspect.getsource(lw.LLMWorker.submit_context)
        stash = src.index("self._last_request = request")
        submit = src.index("self._pool.submit(")
        assert stash < submit, "stash must precede the attempt that can raise queue.Full"

    def test_retry_job_ids_are_unique_per_attempt(self):
        from maxim.agents import llm_worker as lw

        src = inspect.getsource(lw.LLMWorker._resubmit)
        assert "_resubmit_seq" in src, "repeated retries must not collide on one job_id"


class TestDeadAutIsSurfaced:
    """Architecture B1 / executor F4 (cross-confirmed): a dead AUT thread used
    to leave `aut_error` write-only while the orchestrator kept probing —
    turn_count still advanced, so the hard-abort clock kept resetting and the
    campaign produced a full run of empty turns that looked like data."""

    def test_aut_worker_terminates_the_sim_on_loop_failure(self):
        from maxim.simulation import orchestrator

        src = inspect.getsource(orchestrator.start_simulation_mode)
        handler = src.split("aut_error.append(e)", 1)[1][:2600]
        assert '"status": "aut_died"' in handler
        assert "stop_event.set()" in handler
        assert "bridge.finish()" in handler


class TestEnvOptOut:
    """Architecture S5: an abort inside the measurement instrument must be
    experiment-visible and disableable (mirrors MAXIM_SIM_HARD_ABORT)."""

    def test_default_on(self, monkeypatch):
        from maxim.runtime.agent_loop import _planning_liveness_enabled_via_env

        monkeypatch.delenv("MAXIM_SIM_PLANNING_LIVENESS", raising=False)
        assert _planning_liveness_enabled_via_env() is True

    def test_opt_out_values(self, monkeypatch):
        from maxim.runtime.agent_loop import _planning_liveness_enabled_via_env

        for raw in ("0", "false", "no", "off", "OFF", " False "):
            monkeypatch.setenv("MAXIM_SIM_PLANNING_LIVENESS", raw)
            assert _planning_liveness_enabled_via_env() is False, raw

    def test_has_an_autouse_scrub(self):
        """CLAUDE.md: a new opt-in env var in a hot startup path must ship its
        conftest scrub in the SAME commit."""
        conftest = pathlib.Path(__file__).resolve().parents[1] / "conftest.py"
        assert "MAXIM_SIM_PLANNING_LIVENESS" in conftest.read_text()
