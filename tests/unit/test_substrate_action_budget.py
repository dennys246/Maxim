"""The Exp 48 thrashing fix — turn-scoped action budget for the substrate-primary AUT.

Pre-fix, actions/turn was an EMERGENT quantity: the substrate-primary AUT
free-runs at the 0.5 s proposal cadence while the narrator thinks, so
actions/turn = mother-turn wall-clock ÷ 0.5 s — a stopwatch reading of
narrator latency that never reproduced across machines (~60/turn observed).
The bridge's pre-existing ``max_actions_per_turn`` cap bounds only the
OBSERVER's settle loop; the AUT keeps acting through it.

The fix: ``SimulationBridge.substrate_action_allowed`` — a turn-scoped
budget the agent loop's substrate-primary branch consults before proposing
(``run_agentic_loop(substrate_action_gate=...)``). Opt-in via
``MAXIM_SUBSTRATE_ACTIONS_PER_TURN`` (apparatus standard S6: the bound is
experiment-visible and default-OFF, preserving pre-fix behavior byte-
identically).
"""

from __future__ import annotations

import inspect
import time

from maxim.simulation.bridge import SimulationBridge, read_substrate_actions_per_turn_env
from maxim.simulation.sinks import ActionRecord


def _record(bridge: SimulationBridge, tool: str = "turn_right") -> None:
    bridge.action_sink.record(
        ActionRecord(
            timestamp=time.time(),
            tool_name=tool,
            result_success=True,
        )
    )


# ─────────────────────────────────────────────────────────────────────
# Env parser
# ─────────────────────────────────────────────────────────────────────


class TestEnvParser:
    def test_unset_means_unbounded(self, monkeypatch):
        monkeypatch.delenv("MAXIM_SUBSTRATE_ACTIONS_PER_TURN", raising=False)
        assert read_substrate_actions_per_turn_env() is None

    def test_empty_means_unbounded(self, monkeypatch):
        monkeypatch.setenv("MAXIM_SUBSTRATE_ACTIONS_PER_TURN", "   ")
        assert read_substrate_actions_per_turn_env() is None

    def test_valid_integer(self, monkeypatch):
        monkeypatch.setenv("MAXIM_SUBSTRATE_ACTIONS_PER_TURN", "6")
        assert read_substrate_actions_per_turn_env() == 6

    def test_garbage_warns_and_disables(self, monkeypatch, caplog):
        """Invalid → unbounded WITH a warning — never silently invent a
        bound, never silently drop a misconfiguration."""
        monkeypatch.setenv("MAXIM_SUBSTRATE_ACTIONS_PER_TURN", "six")
        with caplog.at_level("WARNING"):
            assert read_substrate_actions_per_turn_env() is None
        assert "not an integer" in caplog.text

    def test_nonpositive_warns_and_disables(self, monkeypatch, caplog):
        monkeypatch.setenv("MAXIM_SUBSTRATE_ACTIONS_PER_TURN", "0")
        with caplog.at_level("WARNING"):
            assert read_substrate_actions_per_turn_env() is None
        assert ">= 1" in caplog.text


# ─────────────────────────────────────────────────────────────────────
# Bridge gate behavior
# ─────────────────────────────────────────────────────────────────────


class TestBridgeGate:
    def test_no_budget_always_allowed(self):
        """Default construction = unbounded = pre-fix behavior."""
        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.1)
        for _ in range(50):
            _record(bridge)
        assert bridge.substrate_action_allowed() is True

    def test_budget_denies_after_n_actions(self):
        bridge = SimulationBridge(
            response_timeout=0.5,
            settle_s=0.1,
            aut_mode="substrate-primary",
            substrate_actions_per_turn=3,
        )
        assert bridge.substrate_action_allowed() is True
        _record(bridge)
        _record(bridge)
        assert bridge.substrate_action_allowed() is True  # 2 < 3
        _record(bridge)
        assert bridge.substrate_action_allowed() is False  # 3 >= 3
        _record(bridge)  # actions past the bound still count against the window
        assert bridge.substrate_action_allowed() is False

    def test_send_and_wait_opens_new_window(self):
        """The turn boundary resets the budget — the AUT gets exactly N
        actions per turn regardless of narrator wall-clock."""
        bridge = SimulationBridge(
            response_timeout=0.3,
            settle_s=0.1,
            aut_mode="substrate-primary",
            substrate_actions_per_turn=2,
        )
        _record(bridge)
        _record(bridge)
        assert bridge.substrate_action_allowed() is False
        # Turn boundary (substrate-primary: no percept injected; the call
        # observes the sink briefly and returns on timeout).
        bridge.send_and_wait("narrative", timeout=0.3)
        assert bridge.substrate_action_allowed() is True
        _record(bridge)
        _record(bridge)
        assert bridge.substrate_action_allowed() is False

    def test_budget_clamped_to_one(self):
        """Constructor clamps a sub-1 explicit value (the env parser
        already refuses these; the clamp is the belt for direct callers)."""
        bridge = SimulationBridge(
            response_timeout=0.5,
            settle_s=0.1,
            substrate_actions_per_turn=0,
        )
        assert bridge.substrate_action_allowed() is True
        _record(bridge)
        assert bridge.substrate_action_allowed() is False

    def test_denial_is_stateless_readonly_for_counting(self):
        """Repeated gate calls without new actions don't change the answer
        (the gate is consulted every proposal cadence tick)."""
        bridge = SimulationBridge(
            response_timeout=0.5,
            settle_s=0.1,
            substrate_actions_per_turn=1,
        )
        _record(bridge)
        for _ in range(10):
            assert bridge.substrate_action_allowed() is False


# ─────────────────────────────────────────────────────────────────────
# Wiring pins (the test_audio_orientation.py source-pin precedent):
# the gate must be consulted by the loop's substrate branch and threaded
# by the orchestrator — a copy of the logic in a test cannot detect the
# real branch regressing, so pin the source.
# ─────────────────────────────────────────────────────────────────────


class TestWiringPins:
    def test_agent_loop_signature_accepts_gate(self):
        from maxim.runtime.agent_loop import run_agentic_loop

        assert "substrate_action_gate" in inspect.signature(run_agentic_loop).parameters

    def test_substrate_branch_consults_gate_before_proposing(self):
        """The gate check must sit inside the substrate-primary cadence
        block, BEFORE propose_via_substrate — pinned at source level."""
        import maxim.runtime.agent_loop as al

        src = inspect.getsource(al.run_agentic_loop)
        branch_start = src.index('aut_mode == "substrate-primary" and ctrl.pending_proposal is None')
        gate_pos = src.index("not substrate_action_gate()", branch_start)
        propose_pos = src.index("propose_via_substrate(", branch_start)
        assert gate_pos < propose_pos, (
            "substrate_action_gate must be consulted BEFORE propose_via_substrate "
            "in the substrate-primary branch — the budget bounds proposals, not telemetry"
        )

    def test_orchestrator_threads_gate_into_aut_loop(self):
        """The AUT run_agentic_loop call carries the bridge's gate."""
        from pathlib import Path

        import maxim.simulation.orchestrator as orch

        src = Path(orch.__file__).read_text()
        # Whitespace-insensitive pin (the formatter may rewrap the call).
        collapsed = " ".join(src.split())
        assert "substrate_action_gate=( bridge.substrate_action_allowed" in collapsed, (
            "orchestrator must wire SimulationBridge.substrate_action_allowed into "
            "the AUT's run_agentic_loop as substrate_action_gate"
        )
        assert "read_substrate_actions_per_turn_env" in src, (
            "orchestrator must read MAXIM_SUBSTRATE_ACTIONS_PER_TURN at bridge construction"
        )
