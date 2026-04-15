"""P3.5 Stage 2 — full SessionSnapshot subprocess round-trip.

Tests that the entire 6-system snapshot survives:
    parent capture → write → fresh subprocess → from_file → restore_into → probe

This is the path P4's cross-modal mug test (1.0-gating) will use to prove
vision↔concept edges in CrossLayerGraph survive a real process boundary.
"""

from __future__ import annotations

from maxim.decisions.nac import NAc, NACConfig
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.cross_layer import CrossLayerEdgeType, CrossLayerGraph
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.percept_trace_buffer import PerceptTraceBuffer
from maxim.time.scn import SCN
from tests.substrate.persistence_harness import run_session_round_trip


def _build_empty_systems():
    return {
        "atl": ATL(ATLConfig()),
        "hippocampus": Hippocampus(HippocampusConfig()),
        "nac": NAc(NACConfig()),
        "scn": SCN(),
        "percept_trace_buffer": PerceptTraceBuffer(),
        "cross_layer_graph": CrossLayerGraph(),
    }


def _build_populated_systems():
    """Build six systems with non-trivial state.

    Every populate call uses the public API only — no private field
    pokes — so the populate path is itself a regression guard for the
    bio-systems that round-trip cleanly through SessionSnapshot.
    """
    systems = _build_empty_systems()

    ptb: PerceptTraceBuffer = systems["percept_trace_buffer"]
    for i in range(15):
        ptb.record(agent_id="alice", percept_id=f"a_p{i}")
    for i in range(10):
        ptb.record(agent_id="bob", percept_id=f"b_p{i}")
    for _ in range(7):
        ptb.tick()

    clg: CrossLayerGraph = systems["cross_layer_graph"]
    for i in range(5):
        clg.add_edge(
            source_layer="ATL",
            source_id=f"concept_{i}",
            target_layer="VISION",
            target_id=f"vision_{i}",
            edge_type=CrossLayerEdgeType.DERIVED_FROM,
            weight=0.5 + 0.05 * i,
        )

    return systems


class TestSessionSubprocessRoundTrip:
    def test_empty_session_round_trip(self):
        systems = _build_empty_systems()
        result = run_session_round_trip(
            systems=systems,
            probe="tests.substrate.probes:session_signature",
            timeout_s=60.0,
        )
        assert result.success, f"empty round-trip failed: error={result.error!r} stderr={result.child_stderr!r}"
        assert result.pre_shutdown == result.post_reload

    def test_populated_session_round_trip(self):
        systems = _build_populated_systems()
        result = run_session_round_trip(
            systems=systems,
            probe="tests.substrate.probes:session_signature",
            timeout_s=60.0,
        )
        assert result.success, f"populated round-trip failed: error={result.error!r} stderr={result.child_stderr!r}"
        # Populated state must be reflected in the probe pre-shutdown
        assert result.pre_shutdown["ptb_len"] == 25
        assert result.pre_shutdown["ptb_tick"] == 7
        # And must match exactly post-reload
        assert result.pre_shutdown == result.post_reload

    def test_unknown_kind_rejected(self):
        result = run_session_round_trip(
            systems={"atl": ATL(ATLConfig()), "not_a_real_kind": object()},  # type: ignore[arg-type]
            probe="tests.substrate.probes:session_signature",
        )
        assert not result.success
        assert "unknown bio-system kinds" in result.error

    def test_state_files_field_carries_session_path(self):
        systems = _build_empty_systems()
        result = run_session_round_trip(
            systems=systems,
            probe="tests.substrate.probes:session_signature",
            timeout_s=60.0,
        )
        assert "session" in result.state_files
        assert result.state_files["session"].endswith(".json")
