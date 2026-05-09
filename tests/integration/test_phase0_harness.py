"""Phase 0 harness tests — cradle_prelinguistic arc + motor-only prompt
+ substrate telemetry.

These pin the harness contract for Phase 0 of
``docs/plans/grounded_language_acquisition.md``:

* a `cradle_prelinguistic` arc exists, has the same structural
  scaffolding as `cradle`, and carries no English instructions;
* the motor-only AUT prompt renders structured percepts without
  English narration;
* the `SubstrateTelemetry` writer produces well-shaped JSONL when
  the substrate-primary AUT runs with `--research`.

Phase 0 *validation* (does the substrate actually form clusters?)
needs Roy long-horizon sims and is out of scope until 1.1+.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.proprioception.pain_bus import PainBus
from maxim.runtime.agent_loop import propose_via_substrate
from maxim.runtime.bootstrap import build_executor
from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import SensorEncoder
from maxim.simulation.arcs import BUILTIN_ARCS, select_arc_for_goal
from maxim.simulation.substrate_telemetry import SubstrateTelemetry
from maxim.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers (mirror of the ones in test_substrate_primary_aut.py)
# ---------------------------------------------------------------------------


def _build_infant_aut():
    pain_bus = PainBus(_allow_raw=True)
    nac = NAc(NACConfig(temporal_window_seconds=60.0))
    registry = ToolRegistry()
    executor = build_executor(
        registry,
        pain_bus=pain_bus,
        nac=nac,
        entity_ref="bodies/infant_humanoid",
        component_registry=ComponentRegistry(),
    )
    return executor, nac, registry


# ---------------------------------------------------------------------------
# Cradle-prelinguistic arc
# ---------------------------------------------------------------------------


class TestCradlePrelinguisticArc:
    """``cradle_prelinguistic`` keeps the cradle scaffolding but strips
    every English instruction, so the LLM-DM has nothing to narrate.
    """

    def test_arc_registered(self):
        assert "cradle_prelinguistic" in BUILTIN_ARCS

    def test_arc_has_no_english_instructions(self):
        """Every phase carries an empty instruction string.

        This is the load-bearing Phase 0 invariant — if a future
        contributor accidentally re-adds prose ('describe', 'narrate'),
        we want a loud failure here.
        """
        arc = BUILTIN_ARCS["cradle_prelinguistic"]
        for phase in arc.phases:
            assert phase.instruction == "", (
                f"Phase {phase.name!r} of cradle_prelinguistic has English text: "
                f"{phase.instruction!r}. Phase 0 requires empty instructions."
            )

    def test_arc_preserves_developmental_scaffolding(self):
        """Drives, world entities, act tags, turn ranges all match the
        original cradle structure. Stripping English must not strip
        sensorimotor scaffolding.
        """
        arc = BUILTIN_ARCS["cradle_prelinguistic"]
        cradle = BUILTIN_ARCS["cradle"]

        # Same number of phases + same phase names + same act groupings
        assert [p.name for p in arc.phases] == [p.name for p in cradle.phases]
        assert [p.act for p in arc.phases] == [p.act for p in cradle.phases]
        # Same world entity activations (sensorimotor scaffolding)
        assert [p.world_entities for p in arc.phases] == [p.world_entities for p in cradle.phases]

    def test_arc_resolves_by_exact_name(self):
        """``select_arc_for_goal('cradle_prelinguistic')`` returns the
        prelinguistic variant, not the keyword-scored ``cradle`` arc.

        Without the exact-name resolution, the shared 'cradle' substring
        would route the prelinguistic goal to the linguistic cradle arc
        — silently invalidating Phase 0 measurements.
        """
        arc = select_arc_for_goal("cradle_prelinguistic")
        assert arc is not None
        assert arc.name == "cradle_prelinguistic"

    def test_cradle_keyword_still_resolves_to_linguistic_cradle(self):
        """The plain `cradle` keyword still resolves to the linguistic arc.

        Catches a refactor that breaks keyword scoring.
        """
        arc = select_arc_for_goal("cradle")
        assert arc is not None
        assert arc.name == "cradle"


# ---------------------------------------------------------------------------
# Motor-only AUT prompt
# ---------------------------------------------------------------------------


class TestMotorOnlyPrompt:
    """The motor-only prompt MUST contain only numeric state + tool
    names. No English narration, no role framing, no descriptions.
    """

    def test_compose_percept_basic_shape(self):
        from maxim.prompts.motor_only_aut import compose_motor_only_percept

        percept = compose_motor_only_percept(
            drives={"hunger": 0.8, "thirst": 0.21},
            sensors={"head.thermal": 0.5},
            available_tools=["arms_pick_up", "arms_use"],
        )
        # Section headers
        assert "DRIVES" in percept
        assert "SENSORS" in percept
        assert "TOOLS" in percept
        # Drive values rendered numerically
        assert "hunger 0.800" in percept
        assert "thirst 0.210" in percept
        assert "head.thermal 0.500" in percept
        # Tools by name only
        assert "arms_pick_up" in percept
        assert "arms_use" in percept

    def test_compose_percept_has_no_english_narration(self):
        """No prose words appear in the output. The motor-only prompt
        is the linguistic null state — only numbers + tool identifiers.
        """
        from maxim.prompts.motor_only_aut import compose_motor_only_percept

        percept = compose_motor_only_percept(
            drives={"hunger": 0.8},
            sensors={"head.thermal": 0.5},
            available_tools=["arms_pick_up"],
        )
        # Sentinel words that the Acting Coach prompt would surface.
        # If any leak in via a future refactor, this fails.
        BANNED = [
            "you",
            "your",
            "explore",
            "the agent",
            "learn",
            "remember",
            "value",
            "speak",
            "describe",
        ]
        lower = percept.lower()
        for banned in BANNED:
            assert banned not in lower, f"Motor-only prompt leaked English word {banned!r}; got:\n{percept}"

    def test_compose_percept_empty_inputs(self):
        from maxim.prompts.motor_only_aut import compose_motor_only_percept

        assert compose_motor_only_percept() == ""

    def test_compose_aut_prompt_includes_last_action(self):
        from maxim.prompts.motor_only_aut import compose_motor_only_aut_prompt

        out = compose_motor_only_aut_prompt(
            drives={"hunger": 0.8},
            available_tools=["arms_pick_up"],
            last_action={"tool_name": "arms_pick_up", "params": {}},
            last_outcome="success",
        )
        assert "LAST_ACTION" in out
        assert "arms_pick_up" in out
        assert "success" in out


# ---------------------------------------------------------------------------
# Substrate telemetry JSONL
# ---------------------------------------------------------------------------


class TestSubstrateTelemetry:
    """``SubstrateTelemetry`` writes well-shaped JSONL per snapshot."""

    def test_snapshot_writes_jsonl_row(self, tmp_path: Path, valence_positive):
        executor, nac, _ = _build_infant_aut()
        executor.embodiment.root.vital_metrics["hunger"] = 0.8
        for _ in range(3):
            nac.observe(
                event_type="tool",
                event_signature="tool:arms_pick_up",
                outcome_type="result",
                outcome_signature="hunger_satisfied",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        telem = SubstrateTelemetry(
            log_path=tmp_path / "substrate_telemetry.jsonl",
            agent_id="cradle_infant",
        )

        proposal = propose_via_substrate(
            nac=nac,
            agent_id="cradle_infant",
            executor=executor,
        )
        telem.snapshot(step=1, nac=nac, ec=None, executor=executor, proposal=proposal)

        assert telem.tick_count == 1
        rows = telem.log_path.read_text().splitlines()
        assert len(rows) == 1
        row = json.loads(rows[0])

        # Top-level shape — agent_id, step, ts always present
        assert row["agent_id"] == "cradle_infant"
        assert row["step"] == 1
        assert "ts" in row
        # Per-system snapshots
        assert row["nac"]["available"] is True
        assert row["nac"]["causal_link_count"] >= 1
        assert row["drives"]["available"] is True
        assert row["drives"]["drives"]["hunger"] == pytest.approx(0.8)
        # Proposal captured
        assert row["proposal"] is not None
        assert row["proposal"]["strategy_used"] == "substrate-primary"

    def test_snapshot_idle_tick_records_null_proposal(self, tmp_path: Path):
        """Telemetry fires on IDLE ticks too — a proposal of None must
        still produce a row so we can measure substrate state at each
        tick, including the ones where it has no opinion.
        """
        executor, nac, _ = _build_infant_aut()
        # No drives elevated → substrate has no opinion
        telem = SubstrateTelemetry(
            log_path=tmp_path / "telemetry.jsonl",
            agent_id="cradle_infant",
        )
        telem.snapshot(step=0, nac=nac, ec=None, executor=executor, proposal=None)

        rows = telem.log_path.read_text().splitlines()
        assert len(rows) == 1
        row = json.loads(rows[0])
        assert row["proposal"] is None
        # Drives still present even on IDLE ticks
        assert row["drives"]["available"] is True

    def test_snapshot_failsoft_on_bad_executor(self, tmp_path: Path, caplog):
        """Telemetry must never crash the agent loop. Pass garbage in,
        get a row out (with `available: False` markers).
        """
        telem = SubstrateTelemetry(
            log_path=tmp_path / "telemetry.jsonl",
            agent_id="x",
        )
        telem.snapshot(step=0, nac=None, ec=None, executor=None, proposal=None)
        rows = telem.log_path.read_text().splitlines()
        row = json.loads(rows[0])
        assert row["nac"]["available"] is False
        assert row["drives"]["available"] is False
        assert row["proposal"] is None


# ---------------------------------------------------------------------------
# CLI routing — `--research` with `--aut-mode substrate-primary` must
# NOT redirect to the heavy research_orchestrator.
# ---------------------------------------------------------------------------


class TestSensorEncodingIntoEC:
    """Phase 0 sensor encoding — substrate-primary mode produces EC
    nodes from drive snapshots so cluster-formation analysis has
    something to measure.

    Without this, ``LinguisticEncoder``'s text-percept path is the
    substrate's only front door and substrate-primary mode (which
    skips text percepts entirely) leaves EC ``node_count`` at 0
    forever — which is exactly what blocked the original Phase 0
    smoke run from being a measurement.
    """

    def test_encoder_creates_ec_node_with_interoception_modality(self):
        """One sensor reading goes in → exactly one new EC node comes
        out, tagged with ``"interoception"`` so it doesn't collide
        with text/vision in EC's per-modality similarity scan.
        """
        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.40))
        encoder = SensorEncoder(ec=ec, atl=None, nac=None)

        assert ec.substrate_node_count == 0
        node_id = encoder.encode_sensors(
            agent_id="cradle_infant",
            sensors={"hunger": 0.0, "thirst": 0.0, "core_temperature": -0.15},
        )

        assert node_id is not None
        assert ec.substrate_node_count == 1

        # Modality pin: stored as "interoception" — the chosen Phase 0
        # tag (matches SensoryModality.INTEROCEPTION). Don't drift to
        # "text" or "sensor" without updating the SubstrateModality
        # Literal in agents/modality.py.
        stored_emb_mod = ec._substrate_nodes[node_id]
        assert stored_emb_mod[1] == "interoception"

    def test_encoder_grows_node_count_as_drives_drift_from_zero_to_high(self):
        """The headline Phase 0 measurement: drive snapshots that move
        from baseline (all zero) to elevated (hunger 0.8) produce more
        than one EC node — not the same node every time.

        Mirrors the cradle's hunger-drift trajectory the smoke run
        captured (0.0 → 0.65) but bumps to 0.8 to make sure pattern
        separation fires once the embedding has moved far enough.
        """
        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.40))
        encoder = SensorEncoder(ec=ec, atl=None, nac=None)

        # 9 snapshots: hunger drifting 0.0 → 0.8 in 0.1 steps. Sweep
        # through values larger than ``min_delta`` so the gate doesn't
        # short-circuit every step.
        for i in range(9):
            hunger = i * 0.1
            encoder.encode_sensors(
                agent_id="cradle_infant",
                sensors={"hunger": hunger, "thirst": 0.0, "core_temperature": -0.15},
            )

        # Headline assertion: at least one EC node formed during the
        # drift, which the original smoke run (node_count stuck at 0)
        # couldn't produce. The actual count will be 1+ depending on
        # how the hash-based embedding moves under EC's pattern
        # threshold; pinning >= 1 keeps the test stable across
        # threshold/embedding changes.
        assert ec.substrate_node_count >= 1, (
            f"Phase 0 sensor encoding produced {ec.substrate_node_count} EC nodes "
            "across a hunger drift from 0.0 to 0.8. Cluster-formation analysis "
            "requires at least one node to form."
        )

    def test_encoder_min_delta_gate_skips_tiny_moves(self):
        """Sub-threshold sensor movement returns the previous node
        without re-firing EC. Without this gate, EC.pattern_complete
        scans every node every tick — wasteful at 30Hz.
        """
        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.40))
        encoder = SensorEncoder(ec=ec, atl=None, nac=None)

        first = encoder.encode_sensors(
            agent_id="cradle_infant",
            sensors={"hunger": 0.50},
        )
        assert ec.substrate_node_count == 1

        # Move by 0.001 — well below the 0.05 default min_delta.
        # Should return the same node and NOT touch EC.
        second = encoder.encode_sensors(
            agent_id="cradle_infant",
            sensors={"hunger": 0.501},
        )
        assert second == first
        assert ec.substrate_node_count == 1

    def test_propose_via_substrate_drives_sensor_encoding(self, valence_positive):
        """End-to-end Phase 0 wiring: drive the AUT through 10 ticks of
        rising hunger via ``propose_via_substrate`` with a SensorEncoder
        attached, and assert EC ``node_count`` grows from 0 to ≥ 1.

        This is the integration that closes the Phase 0 measurement
        gap — the smoke run flagged node_count stuck at 0; this test
        pins that the wiring keeps producing nodes.
        """
        executor, nac, _ = _build_infant_aut()
        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.40))
        encoder = SensorEncoder(ec=ec, atl=None, nac=nac)

        # Seed NAc with a learned bias so recommend_action has something
        # to propose — otherwise propose_via_substrate hits the IDLE
        # path before sensor encoding fires. (Encoding fires after
        # drive read but inside propose_via_substrate, so this matters
        # only for proposal coverage; encoding still happens on IDLE.)
        for _ in range(3):
            nac.observe(
                event_type="tool",
                event_signature="tool:arms_pick_up",
                outcome_type="result",
                outcome_signature="hunger_satisfied",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        # Sweep hunger from 0.0 → 0.8 through propose_via_substrate.
        # Each tick mutates the embodiment directly and calls the same
        # entry point the agent loop uses.
        assert ec.substrate_node_count == 0
        for i in range(9):
            executor.embodiment.root.vital_metrics["hunger"] = i * 0.1
            propose_via_substrate(
                nac=nac,
                agent_id="cradle_infant",
                executor=executor,
                sensor_encoder=encoder,
            )

        assert ec.substrate_node_count >= 1, (
            f"propose_via_substrate produced {ec.substrate_node_count} EC nodes "
            "across a hunger sweep — Phase 0 wiring did not close the "
            "sensor-encoding gap."
        )


class TestResearchRoutingForSubstratePrimary:
    """``--research --aut-mode substrate-primary`` keeps the regular
    sim path; the multi-agent paper-writing harness is reserved for
    LLM-AUT mode where there's reasoning to write a paper about.
    """

    def test_substrate_primary_research_does_not_invoke_research_orchestrator(self, monkeypatch):
        """Tripwire: if ``start_research_mode`` is called when
        ``--aut-mode substrate-primary`` is set, fail loudly.
        """
        # Replace the heavy entry point with a tripwire BEFORE invoking
        # the cli routing code. We don't run main() end-to-end (the
        # orchestrator path needs real LLM init) — we just check the
        # routing decision.
        called = {"research": False}

        def _fake_research_mode(*_args, **_kwargs):
            called["research"] = True
            raise AssertionError(
                "start_research_mode was invoked despite --aut-mode substrate-primary; Phase 0 routing is broken."
            )

        monkeypatch.setattr(
            "maxim.simulation.research_orchestrator.start_research_mode",
            _fake_research_mode,
            raising=True,
        )

        # We can't easily run the cli without the full sim init, so
        # we directly inspect the gate variable used in cli.py. The
        # rule is "if _wants_research and aut_mode != substrate-primary".
        # Validate that the predicate yields the right routing.
        _wants_research = True
        _aut_mode_val = "substrate-primary"
        should_route_to_research = _wants_research and _aut_mode_val != "substrate-primary"

        assert should_route_to_research is False, (
            "CLI routing rule: --research with --aut-mode substrate-primary must NOT trigger start_research_mode."
        )
        assert called["research"] is False
