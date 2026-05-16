"""Tests for Stages 0b + 0c of release_0_9_1.md (telemetry instrumentation).

Stage 0b layers:
- ``ActionRecord`` gains optional ``agent_id`` / ``session_id`` /
  ``entity_class`` fields (CC3-additive on the frozen dataclass).
- ``InstrumentedExecutor`` populates the fields from
  ``utils/http.py::current_context()`` + tool params.
- ``RecordingSink._compress_oldest`` preserves the new fields.
- ``save_action_log`` writes the new fields + a ``_format_version``
  header at the JSONL head.
- ``sim_action`` accepts an ``entity_class`` kwarg routed through
  ``sim_log``'s data dict.

Stage 0c layer:
- ``NAc.recommend_action`` emits a ``sim_log("NAc_RECOMMEND", ...)``
  event on every call (no-scores, sub-threshold, success — three
  early-exit paths), with the fields per the plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.simulation.instrumented_executor import InstrumentedExecutor, _derive_entity_class
from maxim.simulation.sinks import ActionRecord, RecordingSink
from maxim.tools.base import ToolOutput
from maxim.utils.http import new_request_context, reset_context, set_context


# ─────────────────────────────────────────────────────────────────────
# Layer 1: ActionRecord field additions (CC3-additive)
# ─────────────────────────────────────────────────────────────────────


class TestActionRecordFields:
    """The new optional fields ship with ``None`` defaults so existing
    callers and third-party ActionSink consumers keep working without
    modification. The CC3 audit rule for shape-frozen dataclasses is:
    optional fields appended at the end with sensible defaults are
    non-breaking; renames or required-field additions are major bumps."""

    def test_record_constructs_without_new_fields(self) -> None:
        """Back-compat shape: pre-0b callers construct ActionRecord
        with only the original 8 fields and the new 3 default to None."""
        rec = ActionRecord(
            timestamp=1.0,
            tool_name="respond",
        )
        assert rec.agent_id is None
        assert rec.session_id is None
        assert rec.entity_class is None

    def test_record_carries_new_fields(self) -> None:
        rec = ActionRecord(
            timestamp=1.0,
            tool_name="sense_food_source",
            agent_id="sim_aut",
            session_id="20260515_120000",
            entity_class="food",
        )
        assert rec.agent_id == "sim_aut"
        assert rec.session_id == "20260515_120000"
        assert rec.entity_class == "food"


# ─────────────────────────────────────────────────────────────────────
# Layer 2: entity_class derivation heuristic
# ─────────────────────────────────────────────────────────────────────


class TestEntityClassDerivation:
    """Best-effort heuristic — Roy-3 analysis aggregates with ``None``
    skipped from exposure-count normalization, so the heuristic only
    needs to produce sensible classes when the tool is entity-bound.
    Verb-only tools (``respond``, ``examine``) should return None."""

    def test_explicit_param_wins(self) -> None:
        assert _derive_entity_class("any_tool", {"entity_class": "food"}) == "food"

    def test_target_param_fallback(self) -> None:
        assert _derive_entity_class("any_tool", {"target": "weapon"}) == "weapon"

    def test_entity_param_fallback(self) -> None:
        assert _derive_entity_class("any_tool", {"entity": "infant_humanoid"}) == "infant_humanoid"

    def test_object_param_fallback(self) -> None:
        assert _derive_entity_class("any_tool", {"object": "fire"}) == "fire"

    def test_param_priority_explicit_over_target(self) -> None:
        """params["entity_class"] beats params["target"] when both present."""
        assert _derive_entity_class("any_tool", {"entity_class": "food", "target": "drink"}) == "food"

    def test_tool_name_alone_does_not_derive(self) -> None:
        """Pre-merge review fold: the verb-strip heuristic was dropped
        as too noisy. Tools whose name suggests an entity binding
        (``sense_food_source``, ``infant_humanoid_pick_up``) but don't
        pass an entity param through the call now return None.

        Roy-3 normalization explicitly skips None, so being conservative
        is strictly safer than producing wrong buckets — silent miscount
        is worse than missing data. The future fix (1.1 TODO in the
        derivation docstring) declares ``Tool.entity_class`` on the
        Tool ABC so authors opt in explicitly."""
        assert _derive_entity_class("sense_food_source", {}) is None
        assert _derive_entity_class("infant_humanoid_pick_up", {}) is None
        assert _derive_entity_class("use_weapon", {}) is None

    def test_verb_only_tools_return_none(self) -> None:
        """``respond`` / ``examine`` / no underscore → None (not entity-bound)."""
        assert _derive_entity_class("respond", {}) is None
        assert _derive_entity_class("examine", {}) is None
        assert _derive_entity_class("examine", {"target": ""}) is None

    def test_non_entity_tools_with_underscores_return_none(self) -> None:
        """Architecture lens I3 regression guard. The pre-fold verb-strip
        heuristic produced noise on these tools — Roy-3 would have
        attributed pain events to ``"status"`` or ``"entity_sensor"``
        as if they were real entity classes. With the heuristic dropped,
        these all return None."""
        assert _derive_entity_class("get_status", {}) is None
        assert _derive_entity_class("set_entity_sensor", {}) is None
        assert _derive_entity_class("do_something_clever", {}) is None
        assert _derive_entity_class("make_recommendation", {}) is None
        assert _derive_entity_class("look_around", {}) is None

    def test_non_dict_params_returns_none(self) -> None:
        """Defensive — params might be None or something weird in some paths."""
        assert _derive_entity_class("any_tool", None) is None  # type: ignore[arg-type]

    def test_entity_param_with_underscore_tool_name_wins(self) -> None:
        """Even when the tool name has underscores, an explicit
        entity-class param wins. This is the supported path tool
        authors use to opt into Roy-3 attribution today (until 1.1
        ships the ``Tool.entity_class`` declared field)."""
        # Old verb-strip would have stripped "sense" → "food" too,
        # but the param is the authoritative source either way.
        assert _derive_entity_class("sense_food_source", {"target": "apple"}) == "apple"


# ─────────────────────────────────────────────────────────────────────
# Layer 3: InstrumentedExecutor reads RequestContext
# ─────────────────────────────────────────────────────────────────────


class _StubExecutor:
    """Minimal stand-in for the real Executor."""

    def execute(self, action: dict[str, Any]) -> ToolOutput:
        return ToolOutput(success=True, output={"ok": True})


class TestInstrumentedExecutorTelemetry:
    def test_record_populated_from_request_context(self) -> None:
        sink = RecordingSink()
        executor = InstrumentedExecutor(_StubExecutor(), sink)
        ctx = new_request_context(agent_id="sim_aut", session_id="20260515_120000")
        token = set_context(ctx)
        try:
            # Tool author opts into Roy-3 attribution via the
            # explicit ``entity_class`` param (the only path the
            # post-fold heuristic accepts).
            executor.execute(
                {
                    "tool_name": "sense_food_source",
                    "params": {"entity_class": "food"},
                }
            )
        finally:
            reset_context(token)
        rec = sink.actions[-1]
        assert rec.agent_id == "sim_aut"
        assert rec.session_id == "20260515_120000"
        assert rec.entity_class == "food"

    def test_no_context_bound_yields_none_fields(self) -> None:
        """Unit tests / non-sim runtime paths don't bind RequestContext.
        InstrumentedExecutor must not raise — record gets None fields."""
        sink = RecordingSink()
        executor = InstrumentedExecutor(_StubExecutor(), sink)
        # No set_context call — current_context() returns None.
        executor.execute({"tool_name": "respond", "params": {}})
        rec = sink.actions[-1]
        assert rec.agent_id is None
        assert rec.session_id is None
        assert rec.entity_class is None  # respond is verb-only

    def test_record_block_populates_telemetry(self) -> None:
        """Blocked actions also carry telemetry — useful for the
        normalization analysis (blocked-but-attempted is meaningful)."""
        sink = RecordingSink()
        executor = InstrumentedExecutor(_StubExecutor(), sink)
        ctx = new_request_context(agent_id="sim_aut", session_id="sid")
        token = set_context(ctx)
        try:
            executor.record_block("infant_humanoid_pick_up", reason="too_heavy")
        finally:
            reset_context(token)
        rec = sink.actions[-1]
        assert rec.blocked is True
        assert rec.block_reason == "too_heavy"
        assert rec.agent_id == "sim_aut"
        assert rec.session_id == "sid"


# ─────────────────────────────────────────────────────────────────────
# Layer 4: RecordingSink compression preserves new fields
# ─────────────────────────────────────────────────────────────────────


class TestCompressionPreservesTelemetry:
    """Per the plan: actions.jsonl is meant for POST-HOC analysis. If
    compression dropped agent_id/session_id/entity_class, normalization
    counts for long-running sims would silently skew (compressed half
    looks anonymous; uncompressed half attributes correctly). Pin
    the preservation contract."""

    def test_compression_keeps_new_fields(self) -> None:
        """``_compress_oldest`` strips heavy fields (tool_args,
        result_output) but keeps a lightweight summary. The telemetry
        fields are tiny and MUST be preserved so post-hoc attribution
        analysis doesn't get an anonymous half of the record stream."""
        sink = RecordingSink(max_actions=4)
        for i in range(6):
            sink.record(
                ActionRecord(
                    timestamp=float(i),
                    tool_name=f"tool_{i}",
                    tool_args={"large": "args"},  # heavy field that compression drops
                    result_output={"big": "output"},  # ditto
                    agent_id=f"agent_{i}",
                    session_id="shared_session",
                    entity_class=f"class_{i}",
                )
            )
        # Compression fires when len > max — verified by checking
        # the oldest half had heavy fields stripped.
        compressed_count = sum(1 for r in sink.actions if r.tool_args == {})
        assert compressed_count > 0, "compression should have stripped at least some records"
        # Telemetry fields survive compression in every record
        # (compressed AND uncompressed).
        for rec in sink.actions:
            assert rec.agent_id is not None
            assert rec.session_id == "shared_session"
            assert rec.entity_class is not None


# ─────────────────────────────────────────────────────────────────────
# Layer 5: save_action_log writes new fields + format-version header
# ─────────────────────────────────────────────────────────────────────


class TestSaveActionLog:
    """Per the plan: actions.jsonl now carries a ``_format_version``
    header line + telemetry fields per record. The header pattern is
    CC1 contract — every persisted JSONL Maxim writes carries
    ``_format_version`` at the head so future parsers can branch on
    schema evolution."""

    def test_writes_format_version_header(self, tmp_path: Path) -> None:
        from maxim.simulation.report import _ACTIONS_JSONL_FORMAT_VERSION, save_action_log

        bridge = MagicMock()
        bridge.get_all_actions.return_value = [
            ActionRecord(timestamp=1.0, tool_name="respond", agent_id="sim_aut", session_id="sid"),
        ]
        log_path = save_action_log(bridge, base_dir=str(tmp_path), session_id="sid")
        assert log_path is not None
        lines = log_path.read_text().splitlines()
        header = json.loads(lines[0])
        # Plan release_0_9_1.md § "Cross-cutting: persistence schema"
        # pins this at "1.1" — minor bump from pre-0b unversioned ("0.x").
        assert _ACTIONS_JSONL_FORMAT_VERSION == "1.1"
        assert header["_format_version"] == "1.1"
        assert header["_record_kind"] == "header"
        assert header["session_id"] == "sid"

    def test_consumer_can_skip_header_line(self, tmp_path: Path) -> None:
        """Architecture lens I4 regression guard. The header-line is a
        schema change for actions.jsonl consumers — third-party tooling
        iterating the file as "every line is a record" needs to skip
        ``_record_kind == "header"``. The docstring contract is
        load-bearing; this test pins it as a real reader pattern."""
        from maxim.simulation.report import save_action_log

        bridge = MagicMock()
        bridge.get_all_actions.return_value = [
            ActionRecord(
                timestamp=1.0,
                tool_name="sense_food_source",
                agent_id="sim_aut",
                session_id="sid",
                entity_class="food",
            ),
            ActionRecord(timestamp=2.0, tool_name="respond", agent_id="sim_aut", session_id="sid"),
        ]
        log_path = save_action_log(bridge, base_dir=str(tmp_path), session_id="sid")
        assert log_path is not None
        # Simulate the documented reader pattern:
        records = []
        with log_path.open() as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("_record_kind") == "header":
                    continue
                records.append(obj)
        assert len(records) == 2
        assert records[0]["tool"] == "sense_food_source"
        assert records[1]["tool"] == "respond"

    def test_writes_telemetry_fields_per_record(self, tmp_path: Path) -> None:
        from maxim.simulation.report import save_action_log

        bridge = MagicMock()
        bridge.get_all_actions.return_value = [
            ActionRecord(
                timestamp=1.0,
                tool_name="sense_food_source",
                agent_id="sim_aut",
                session_id="20260515_120000",
                entity_class="food",
            ),
        ]
        log_path = save_action_log(bridge, base_dir=str(tmp_path), session_id="20260515_120000")
        assert log_path is not None
        lines = log_path.read_text().splitlines()
        # Skip the header line.
        record = json.loads(lines[1])
        assert record["agent_id"] == "sim_aut"
        assert record["session_id"] == "20260515_120000"
        assert record["entity_class"] == "food"

    def test_header_appears_even_with_zero_records(self, tmp_path: Path) -> None:
        """Empty action log still has the header — schema discovery
        shouldn't depend on a record existing."""
        from maxim.simulation.report import save_action_log

        bridge = MagicMock()
        bridge.get_all_actions.return_value = []
        log_path = save_action_log(bridge, base_dir=str(tmp_path), session_id="sid")
        assert log_path is not None
        lines = log_path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["_format_version"] == "1.1"


# ─────────────────────────────────────────────────────────────────────
# Layer 6: sim_action entity_class kwarg
# ─────────────────────────────────────────────────────────────────────


class TestSimActionEntityClass:
    """sim_action grows an ``entity_class`` kwarg that routes through
    sim_log's data dict for post-hoc Roy-3 analysis. Existing
    callers using the positional shape continue to work."""

    def test_legacy_call_shape_unchanged(self) -> None:
        """The plain ``sim_action(tool, success)`` call must still
        work — most existing callers don't pass entity_class yet."""
        from maxim.simulation.sim_logger import sim_action

        # Should not raise; nothing else to assert because sim_log
        # silently no-ops when sim logging isn't enabled.
        sim_action("respond", True)

    def test_entity_class_threaded_through(self, tmp_path: Path) -> None:
        """When sim_action is called with entity_class, it shows up
        in the JSONL record's data dict."""
        from maxim.simulation.sim_logger import (
            disable_sim_logging,
            enable_sim_logging,
            sim_action,
        )

        log_path = tmp_path / "sim_log.jsonl"
        enable_sim_logging(log_path=str(log_path))
        try:
            sim_action("sense_food_source", True, summary="found", entity_class="food")
        finally:
            disable_sim_logging()

        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        # Find the MOTOR record.
        motor_records = [r for r in records if r.get("subsystem") == "MOTOR"]
        assert len(motor_records) >= 1
        assert motor_records[0]["data"]["entity_class"] == "food"

    def test_entity_class_none_omits_field(self, tmp_path: Path) -> None:
        """When entity_class is None (the default), it should NOT
        appear as a key in the data dict — cluttering Roy-3 records
        with ``entity_class: null`` for every verb-only tool is noise."""
        from maxim.simulation.sim_logger import (
            disable_sim_logging,
            enable_sim_logging,
            sim_action,
        )

        log_path = tmp_path / "sim_log.jsonl"
        enable_sim_logging(log_path=str(log_path))
        try:
            sim_action("respond", True, summary="hi")
        finally:
            disable_sim_logging()

        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        motor_records = [r for r in records if r.get("subsystem") == "MOTOR"]
        assert len(motor_records) >= 1
        # The `data` field is either absent or doesn't contain entity_class.
        data = motor_records[0].get("data") or {}
        assert "entity_class" not in data


# ─────────────────────────────────────────────────────────────────────
# Layer 7: Stage 0c — sim_recommend_action emission
# ─────────────────────────────────────────────────────────────────────


class TestRecommendActionEmission:
    """Per the plan: EVERY recommend_action call emits exactly one
    sim_recommend_action event, including the early-return paths.
    Roy-3 needs to distinguish "gate fired, consumer did nothing"
    from "consumer didn't run at all"."""

    def _fresh_nac(self) -> NAc:
        return NAc(config=NACConfig())

    def _read_recommend_records(self, log_path: Path) -> list[dict[str, Any]]:
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        return [r for r in records if r.get("subsystem") == "NAc_RECOMMEND"]

    def test_emission_on_success_path(self, tmp_path: Path) -> None:
        from maxim.simulation.sim_logger import disable_sim_logging, enable_sim_logging

        log_path = tmp_path / "sim_log.jsonl"
        enable_sim_logging(log_path=str(log_path))
        try:
            nac = self._fresh_nac()
            # Seed cluster bias so a tool wins.
            nac.update_cluster_reward(
                agent_id="sim_aut",
                cluster_id="cluster-1",
                tool_signature="tool:sense_food_source",
                reward=10.0,
            )
            result = nac.recommend_action(
                agent_id="sim_aut",
                available_tools=["sense_food_source"],
                current_cluster_id="cluster-1",
            )
        finally:
            disable_sim_logging()

        assert result is not None
        assert result["tool_name"] == "sense_food_source"
        recs = self._read_recommend_records(log_path)
        assert len(recs) == 1
        data = recs[0]["data"]
        assert data["passed_gate"] is True
        assert data["best_tool"] == "sense_food_source"
        assert data["best_score"] > 0.0
        assert data["current_cluster_id"] == "cluster-1"
        assert data["cluster_reward_bias_consulted"] is not None

    def test_emission_on_no_scores_path(self, tmp_path: Path) -> None:
        """When recommend_action returns None because no tool scored,
        the event MUST still emit — passed_gate=False, best_tool=None."""
        from maxim.simulation.sim_logger import disable_sim_logging, enable_sim_logging

        log_path = tmp_path / "sim_log.jsonl"
        enable_sim_logging(log_path=str(log_path))
        try:
            nac = self._fresh_nac()
            # No bias seeded; no drives; no causal links → empty scores.
            result = nac.recommend_action(
                agent_id="sim_aut",
                available_tools=["sense_food_source"],
            )
        finally:
            disable_sim_logging()

        assert result is None
        recs = self._read_recommend_records(log_path)
        assert len(recs) == 1
        data = recs[0]["data"]
        assert data["passed_gate"] is False
        assert data["best_tool"] is None
        assert data["best_score"] == 0.0

    def test_emission_on_sub_threshold_path(self, tmp_path: Path) -> None:
        """When scores exist but best_score < min_confidence, the event
        emits with passed_gate=False AND best_tool populated."""
        from maxim.simulation.sim_logger import disable_sim_logging, enable_sim_logging

        log_path = tmp_path / "sim_log.jsonl"
        enable_sim_logging(log_path=str(log_path))
        try:
            nac = self._fresh_nac()
            # Seed a tiny bias well under min_confidence (0.3).
            nac.update_cluster_reward(
                agent_id="sim_aut",
                cluster_id="cluster-1",
                tool_signature="tool:sense_food_source",
                reward=0.5,  # alpha=0.15 → +0.075, below the 0.3 default gate
            )
            result = nac.recommend_action(
                agent_id="sim_aut",
                available_tools=["sense_food_source"],
                current_cluster_id="cluster-1",
                min_confidence=0.3,
            )
        finally:
            disable_sim_logging()

        assert result is None
        recs = self._read_recommend_records(log_path)
        assert len(recs) == 1
        data = recs[0]["data"]
        assert data["passed_gate"] is False
        assert data["best_tool"] == "sense_food_source"
        assert 0.0 < data["best_score"] < 0.3

    def test_fail_soft_when_sim_logging_disabled(self) -> None:
        """Non-sim runtime path: sim_log is a no-op when sim logging
        isn't enabled, so recommend_action returns normally without
        raising or even leaving a partial state."""
        nac = self._fresh_nac()
        nac.update_cluster_reward("sim_aut", "c1", "tool:foo", reward=10.0)
        # sim logging NOT enabled here.
        result = nac.recommend_action(
            agent_id="sim_aut",
            available_tools=["foo"],
            current_cluster_id="c1",
        )
        assert result is not None
        assert result["tool_name"] == "foo"

    def test_emission_on_empty_available_tools_path(self, tmp_path: Path) -> None:
        """Architecture lens C3 regression guard. Pre-fold,
        ``available_tools=[]`` short-circuited before emitting,
        leaving Roy-3 unable to distinguish "no tools available"
        from "no tools scored above gate." Post-fold, this path
        emits with best_tool=None, best_score=0.0, passed_gate=False."""
        from maxim.simulation.sim_logger import disable_sim_logging, enable_sim_logging

        log_path = tmp_path / "sim_log.jsonl"
        enable_sim_logging(log_path=str(log_path))
        try:
            nac = self._fresh_nac()
            result = nac.recommend_action(agent_id="sim_aut", available_tools=[])
        finally:
            disable_sim_logging()

        assert result is None
        recs = self._read_recommend_records(log_path)
        assert len(recs) == 1
        data = recs[0]["data"]
        assert data["passed_gate"] is False
        assert data["best_tool"] is None
        assert data["best_score"] == 0.0

    def test_tick_aligned_with_sim_logger_start(self, tmp_path: Path) -> None:
        """Architecture lens C1 regression guard. Pre-fold, tick was
        ``int(time.time())`` (raw epoch ~1.7e9), while Stage 0d's
        ``sim_ec_activation`` uses ``int(time.time() - _sim_start)``
        (elapsed seconds, 0..N). A 1e9 offset means Roy-3 left-joins
        on tick return zero matches every time. Post-fold, both
        channels emit comparable tick values from the same _sim_start
        reference — a tick within ~1s of sim start is a small int,
        not an epoch."""
        from maxim.simulation.sim_logger import disable_sim_logging, enable_sim_logging

        log_path = tmp_path / "sim_log.jsonl"
        enable_sim_logging(log_path=str(log_path))
        try:
            nac = self._fresh_nac()
            nac.update_cluster_reward("sim_aut", "c1", "tool:foo", reward=10.0)
            nac.recommend_action(
                agent_id="sim_aut",
                available_tools=["foo"],
                current_cluster_id="c1",
            )
        finally:
            disable_sim_logging()

        recs = self._read_recommend_records(log_path)
        assert len(recs) == 1
        tick = recs[0]["data"]["tick"]
        # Elapsed-seconds tick within a fresh sim should be tiny.
        # If we accidentally regress to raw epoch, this would be ~1.7e9.
        assert 0 <= tick < 60, f"tick={tick} looks like raw epoch (regression to int(time.time()))"

    def test_empty_scores_sentinel_distinguishes_cluster_known_vs_unknown(self, tmp_path: Path) -> None:
        """Bio-fidelity I3 regression guard. On the empty-scores path:
          - cluster_id known but no tool scored → 0.0 sentinel
          - cluster_id absent → None
        Roy-3 disambiguation depends on this — otherwise "agent had no
        active cluster" and "agent had a cluster but no tools scored"
        collapse into the same record."""
        from maxim.simulation.sim_logger import disable_sim_logging, enable_sim_logging

        log_path = tmp_path / "sim_log.jsonl"
        enable_sim_logging(log_path=str(log_path))
        try:
            nac = self._fresh_nac()
            # Path A: cluster known, no tool scored (no bias seeded).
            nac.recommend_action(
                agent_id="sim_aut",
                available_tools=["foo"],
                current_cluster_id="c1",
            )
            # Path B: cluster unknown.
            nac.recommend_action(
                agent_id="sim_aut",
                available_tools=["foo"],
                current_cluster_id=None,
            )
        finally:
            disable_sim_logging()

        recs = self._read_recommend_records(log_path)
        assert len(recs) == 2
        # Order matches call order.
        path_a, path_b = recs[0]["data"], recs[1]["data"]
        assert path_a["current_cluster_id"] == "c1"
        assert path_a["cluster_reward_bias_consulted"] == 0.0  # known but no signal
        assert path_b["current_cluster_id"] is None
        assert path_b["cluster_reward_bias_consulted"] is None  # truly absent


# ─────────────────────────────────────────────────────────────────────
# Layer 8: RequestContext binding regression guard
# ─────────────────────────────────────────────────────────────────────


class TestRequestContextBindingShape:
    """The Stage 0b binding is done in _aut_worker (orchestrator.py)
    via new_request_context + set_context + reset_context. The actual
    sim-orchestrator binding happens at runtime inside a thread, so
    we test the SHAPE here: a fresh context binds, current_context()
    reads back, reset restores."""

    def test_round_trip(self) -> None:
        ctx = new_request_context(agent_id="sim_aut", session_id="20260515_120000")
        token = set_context(ctx)
        try:
            from maxim.utils.http import current_context

            current = current_context()
            assert current is not None
            assert current.agent_id == "sim_aut"
            assert current.session_id == "20260515_120000"
        finally:
            reset_context(token)
        # After reset, the binding is gone (or restored to whatever
        # was bound before — pytest fixtures generally start with None).
        from maxim.utils.http import current_context

        after = current_context()
        # Either None or pre-existing; the new binding is gone.
        if after is not None:
            assert after.agent_id != "sim_aut" or after.session_id != "20260515_120000"

    def test_reset_in_finally_handles_exception(self) -> None:
        """If the sim worker raises, the reset_context call in finally
        must still fire so the binding doesn't leak to the next test."""
        ctx = new_request_context(agent_id="sim_aut", session_id="sid")
        token = set_context(ctx)
        try:
            try:
                raise RuntimeError("simulated worker failure")
            except RuntimeError:
                pass  # The orchestrator catches this; here we just want
                # to ensure reset is still reachable.
        finally:
            reset_context(token)
        from maxim.utils.http import current_context

        # Either None or the prior binding — not the failed binding.
        after = current_context()
        if after is not None:
            assert after.agent_id != "sim_aut" or after.session_id != "sid"


@pytest.fixture(autouse=True)
def _reset_sim_logger_state() -> Any:
    """sim_logger has module-level state (the _sim_active flag, the
    _log_file handle, the _log_records deque). Make sure each test
    starts with sim logging DISABLED, even if a prior test forgot to
    call disable_sim_logging()."""
    from maxim.simulation.sim_logger import disable_sim_logging

    disable_sim_logging()
    yield
    disable_sim_logging()
