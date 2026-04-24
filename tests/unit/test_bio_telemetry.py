"""Tests for bio-system telemetry and new sim_logger functions.

Covers:
- Track 1+2: New sim_logger functions with emoji markers
- Track 3: BioTelemetryCollector save/load/summarize
- Track 4: Enrichment transparency logging
- Track 5: Bio-system wiring (sim_nac_learn, sim_imagination, etc.)
"""

from __future__ import annotations

import tempfile

import pytest

import maxim.simulation.sim_logger as _sim_mod
from maxim.simulation.sim_logger import (
    DisplayTier,
    enable_sim_logging,
    disable_sim_logging,
    get_sim_records,
    sim_nac_learn,
    sim_nac_predict,
    sim_cerebellum_train,
    sim_cerebellum_predict,
    sim_hippocampus_episode,
    sim_ec_pattern,
    sim_imagination,
    sim_discovery,
    sim_enrichment,
    sim_gate,
    sim_sensor,
    sim_pain,
    sim_fear,
    sim_action,
    sim_result,
    sim_nac,
    sim_memory,
    sim_reaction,
    sim_percept,
    sim_body_state,
    set_display_tier,
)


@pytest.fixture(autouse=True)
def _enable_sim_logging():
    """Enable sim logging for each test, disable after."""
    enable_sim_logging(use_color=False)
    yield
    disable_sim_logging()
    # Access via module reference (not by-name import) to avoid rebinding trap
    _sim_mod._log_records.clear()


# ---------------------------------------------------------------------------
# Track 1+2: New sim_logger functions produce structured JSONL records
# ---------------------------------------------------------------------------


class TestNewSimLogFunctions:
    """Verify each new sim_log function emits a record with correct subsystem."""

    def test_sim_nac_learn_emits_record(self):
        set_display_tier(DisplayTier.BIO)
        sim_nac_learn(event="swing_sword", outcome="target_hit", confidence=0.72, rpe=0.31, link_type="new")
        records = [r for r in get_sim_records() if r["subsystem"] == "NAc"]
        assert len(records) >= 1
        last = records[-1]
        assert "swing_sword" in last["message"]
        assert last["data"]["confidence"] == 0.72
        assert last["data"]["rpe"] == 0.31
        assert last["data"]["link_type"] == "new"

    def test_sim_nac_predict_emits_record(self):
        set_display_tier(DisplayTier.BIO)
        sim_nac_predict(event="swing_sword", outcomes=[("target_hit", 0.8), ("miss", 0.2)])
        records = [r for r in get_sim_records() if r["subsystem"] == "NAc" and "predict" in r["message"]]
        assert len(records) >= 1
        assert "target_hit" in records[-1]["message"]

    def test_sim_nac_predict_empty(self):
        set_display_tier(DisplayTier.BIO)
        sim_nac_predict(event="unknown_action", outcomes=[])
        records = [r for r in get_sim_records() if r["subsystem"] == "NAc" and "no known" in r["message"]]
        assert len(records) >= 1

    def test_sim_cerebellum_train(self):
        set_display_tier(DisplayTier.BIO)
        sim_cerebellum_train(entity="sword", affordance="slash", pred_error=0.05, confidence=0.85)
        records = [r for r in get_sim_records() if r["subsystem"] == "CEREBELLUM" and "train" in r["message"]]
        assert len(records) >= 1
        assert records[-1]["data"]["pred_error"] == 0.05

    def test_sim_cerebellum_predict_with_values(self):
        set_display_tier(DisplayTier.BIO)
        sim_cerebellum_predict(entity="sword", affordance="slash", predicted={"stress": 0.7}, confidence=0.9)
        records = [r for r in get_sim_records() if r["subsystem"] == "CEREBELLUM" and "predict" in r["message"]]
        assert len(records) >= 1
        assert "stress=0.70" in records[-1]["message"]

    def test_sim_cerebellum_predict_no_model(self):
        set_display_tier(DisplayTier.BIO)
        sim_cerebellum_predict(entity="sword", affordance="slash", predicted=None, confidence=0.0)
        records = [r for r in get_sim_records() if r["subsystem"] == "CEREBELLUM" and "no confident" in r["message"]]
        assert len(records) >= 1

    def test_sim_hippocampus_episode(self):
        set_display_tier(DisplayTier.BIO)
        sim_hippocampus_episode("close", episode_id="abc12345", node_count=5, valence=-0.3)
        records = [r for r in get_sim_records() if r["subsystem"] == "HIPPOCAMPUS" and "episode" in r["message"]]
        assert len(records) >= 1
        assert "close" in records[-1]["message"]
        assert records[-1]["data"]["node_count"] == 5

    def test_sim_ec_pattern(self):
        set_display_tier(DisplayTier.BIO)
        sim_ec_pattern(action="complete", concept="sword", similarity=0.85, threshold=0.70)
        records = [r for r in get_sim_records() if r["subsystem"] == "ATL" and "pattern" in r["message"]]
        assert len(records) >= 1
        assert "complete" in records[-1]["message"]

    def test_sim_imagination(self):
        set_display_tier(DisplayTier.BIO)
        sim_imagination("design", "crystal dragon", result="creatures/crystal_dragon")
        records = [r for r in get_sim_records() if r["subsystem"] == "IMAGINATION"]
        assert len(records) >= 1
        assert "crystal dragon" in records[-1]["message"]
        assert records[-1]["data"]["phase"] == "design"

    def test_sim_discovery(self):
        set_display_tier(DisplayTier.BIO)
        sim_discovery("attack with sword", matched=5, activated=3, source="SenseToolsTool")
        records = [r for r in get_sim_records() if r["subsystem"] == "DISCOVERY"]
        assert len(records) >= 1
        assert records[-1]["data"]["matched"] == 5
        assert records[-1]["data"]["activated"] == 3

    def test_sim_enrichment(self):
        set_display_tier(DisplayTier.BIO)
        sim_enrichment("hippocampus", "2 episodes: sword encounter, blade damage")
        records = [r for r in get_sim_records() if r["subsystem"] == "ENRICHMENT"]
        assert len(records) >= 1
        assert "hippocampus" in records[-1]["message"]

    def test_sim_gate(self):
        set_display_tier(DisplayTier.BIO)
        sim_gate("ThoughtGate", passed=True, score=0.82, threshold=0.45, reason="high salience")
        records = [r for r in get_sim_records() if r["subsystem"] == "GATE"]
        assert len(records) >= 1
        assert records[-1]["data"]["passed"] is True

    def test_sim_sensor(self):
        set_display_tier(DisplayTier.BIO)
        sim_sensor("sword.blade", "stress", 0.73, baseline=0.5)
        records = [r for r in get_sim_records() if r["subsystem"] == "SENSOR"]
        assert len(records) >= 1
        assert "drift" in records[-1]["message"]
        assert records[-1]["data"]["drift_pct"] == pytest.approx(46.0)

    def test_sim_sensor_no_baseline(self):
        set_display_tier(DisplayTier.BIO)
        sim_sensor("sword.blade", "temperature", 25.0)
        records = [r for r in get_sim_records() if r["subsystem"] == "SENSOR"]
        assert len(records) >= 1
        assert "drift" not in records[-1]["message"]


# ---------------------------------------------------------------------------
# Track 2: Emoji markers in existing functions
# ---------------------------------------------------------------------------


class TestEmojiMarkers:
    """Verify emoji markers appear in existing sim_log functions."""

    def test_sim_pain_has_emoji(self):
        set_display_tier(DisplayTier.BIO)
        sim_pain("EXCESSIVE_VELOCITY", 0.8)
        records = [r for r in get_sim_records() if r["subsystem"] == "PAIN"]
        assert any("\U0001f534" in r["message"] for r in records)  # 🔴

    def test_sim_fear_allowed_has_emoji(self):
        set_display_tier(DisplayTier.BIO)
        sim_fear("swing_sword", allowed=True)
        records = [r for r in get_sim_records() if r["subsystem"] == "FEAR"]
        assert any("\U0001f6e1" in r["message"] for r in records)  # 🛡️

    def test_sim_fear_blocked_has_emoji(self):
        set_display_tier(DisplayTier.CLEAN)
        sim_fear("swing_sword", allowed=False, reason="high risk")
        records = [r for r in get_sim_records() if r["subsystem"] == "BLOCKED"]
        assert any("\U0001f6ab" in r["message"] for r in records)  # 🚫

    def test_sim_action_success_has_emoji(self):
        set_display_tier(DisplayTier.BIO)
        sim_action("swing_sword", success=True, summary="hit target")
        records = [r for r in get_sim_records() if r["subsystem"] == "MOTOR"]
        assert any("\u2694" in r["message"] for r in records)  # ⚔️

    def test_sim_action_fail_has_emoji(self):
        set_display_tier(DisplayTier.BIO)
        sim_action("swing_sword", success=False, summary="missed")
        records = [r for r in get_sim_records() if r["subsystem"] == "MOTOR"]
        assert any("\u274c" in r["message"] for r in records)  # ❌

    def test_sim_nac_has_emoji(self):
        set_display_tier(DisplayTier.BIO)
        sim_nac("swing", "hit", rpe=0.5, confidence=0.8)
        records = [r for r in get_sim_records() if r["subsystem"] == "NAc"]
        assert any("\U0001f9e0" in r["message"] for r in records)  # 🧠

    def test_sim_memory_has_emoji(self):
        set_display_tier(DisplayTier.BIO)
        sim_memory("Captured: sword encounter")
        records = [r for r in get_sim_records() if r["subsystem"] == "HIPPOCAMPUS"]
        assert any("\U0001f4be" in r["message"] for r in records)  # 💾

    def test_sim_percept_has_emoji(self):
        set_display_tier(DisplayTier.BIO)
        sim_percept("vision", "rusty sword on ground")
        records = [r for r in get_sim_records() if r["subsystem"] == "PERCEPT"]
        assert any("\U0001f441" in r["message"] for r in records)  # 👁️

    def test_sim_reaction_pain_has_emoji(self):
        set_display_tier(DisplayTier.BIO)
        sim_reaction("pain", intensity=0.7, source="body")
        records = [r for r in get_sim_records() if r["subsystem"] == "REACTION"]
        assert any("\U0001f534" in r["message"] for r in records)  # 🔴

    def test_sim_result_pass_has_emoji(self):
        set_display_tier(DisplayTier.CLEAN)
        sim_result("test_scenario", passed=True, met=5, failed=0)
        records = [r for r in get_sim_records() if r["subsystem"] == "RESULT"]
        assert any("\u2705" in r["message"] for r in records)  # ✅

    def test_sim_body_state_has_emoji(self):
        set_display_tier(DisplayTier.BIO)
        sim_body_state(entity_count=3, active_failures=1)
        records = [r for r in get_sim_records() if r["subsystem"] == "BODY"]
        assert any("\U0001f525" in r["message"] for r in records)  # 🔥 (active failures)

    def test_sim_body_state_healthy(self):
        set_display_tier(DisplayTier.BIO)
        sim_body_state(entity_count=3, active_failures=0)
        records = [r for r in get_sim_records() if r["subsystem"] == "BODY"]
        assert any("\U0001fac0" in r["message"] for r in records)  # 🫀 (healthy)


# ---------------------------------------------------------------------------
# Track 3: BioTelemetryCollector
# ---------------------------------------------------------------------------


class TestBioTelemetryCollector:
    """Test save/load/summarize of bio telemetry."""

    def test_save_and_load(self):
        from maxim.simulation.bio_telemetry import BioTelemetryCollector

        set_display_tier(DisplayTier.BIO)
        # Emit some bio events
        sim_nac_learn("swing", "hit", 0.7, 0.3, link_type="new")
        sim_pain("VELOCITY", 0.8)
        sim_imagination("design", "dragon", result="creatures/dragon")
        sim_discovery("attack", matched=3, activated=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BioTelemetryCollector()
            path = collector.save(tmpdir)
            assert path is not None
            assert path.exists()

            # Load and verify
            events = BioTelemetryCollector.load(path)
            assert len(events) >= 4
            subsystems = {e["subsystem"] for e in events}
            assert "NAc" in subsystems
            assert "PAIN" in subsystems
            assert "IMAGINATION" in subsystems
            assert "DISCOVERY" in subsystems

    def test_save_excludes_non_bio(self):
        from maxim.simulation.bio_telemetry import BioTelemetryCollector
        from maxim.simulation.sim_logger import sim_log

        set_display_tier(DisplayTier.DEBUG)
        # Emit a pipeline event (non-bio)
        sim_log("PIPELINE", "test pipeline event")
        # Emit a bio event
        sim_pain("TEST", 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BioTelemetryCollector()
            path = collector.save(tmpdir)
            assert path is not None
            events = BioTelemetryCollector.load(path)
            # Pipeline events are NOT in the bio subsystems set
            subsystems = {e["subsystem"] for e in events}
            assert "PIPELINE" not in subsystems
            assert "PAIN" in subsystems

    def test_summarize(self):
        from maxim.simulation.bio_telemetry import BioTelemetryCollector

        set_display_tier(DisplayTier.BIO)
        sim_nac_learn("swing", "hit", 0.7, 0.3, link_type="new")
        sim_nac_learn("guard", "block", 0.6, -0.2, link_type="updated")
        sim_pain("VELOCITY", 0.8)
        sim_imagination("design", "dragon", result="creatures/dragon")

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BioTelemetryCollector()
            path = collector.save(tmpdir)
            events = BioTelemetryCollector.load(path)
            summary = BioTelemetryCollector.summarize(events)

            assert summary["total_events"] >= 4
            assert summary["nac_links_formed"] == 2
            assert summary["pain_events"] >= 1
            assert summary["imagination_events"] >= 1

    def test_save_empty(self):
        """No bio events → save returns None."""
        from maxim.simulation.bio_telemetry import BioTelemetryCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BioTelemetryCollector()
            path = collector.save(tmpdir)
            assert path is None


# ---------------------------------------------------------------------------
# Track 4: Enrichment transparency
# ---------------------------------------------------------------------------


class TestEnrichmentTransparency:
    """Test that BioEnrichmentPipeline emits sim_enrichment logs."""

    def test_enrichment_logs_hippocampus(self):
        from maxim.integration.bio_enrichment import (
            BioEnrichmentPipeline,
            EpisodicSummary,
        )

        set_display_tier(DisplayTier.BIO)

        # Call the static method directly
        memories = [
            EpisodicSummary(memory_id="m1", summary="sword broke during heavy swing", valence=-0.5, relevance=0.8),
        ]
        BioEnrichmentPipeline._log_enrichment_contributions(memories, [], [], [], [])

        records = [r for r in get_sim_records() if r["subsystem"] == "ENRICHMENT"]
        assert len(records) == 1
        assert "hippocampus" in records[0]["message"]
        assert "sword broke" in records[0]["message"]

    def test_enrichment_logs_nac(self):
        from maxim.integration.bio_enrichment import (
            BioEnrichmentPipeline,
            CausalPrediction,
        )

        set_display_tier(DisplayTier.BIO)
        predictions = [
            CausalPrediction(event="swing", outcome="hit", confidence=0.8, valence="positive"),
        ]
        BioEnrichmentPipeline._log_enrichment_contributions([], predictions, [], [], [])

        records = [r for r in get_sim_records() if r["subsystem"] == "ENRICHMENT"]
        assert len(records) == 1
        assert "nac" in records[0]["message"]
        assert "swing" in records[0]["message"]

    def test_enrichment_logs_multiple_systems(self):
        from maxim.integration.bio_enrichment import (
            BioEnrichmentPipeline,
            EpisodicSummary,
            CausalPrediction,
            ConceptLink,
        )

        set_display_tier(DisplayTier.BIO)
        memories = [EpisodicSummary(memory_id="m1", summary="past event", valence=0.0, relevance=0.7)]
        predictions = [CausalPrediction(event="e", outcome="o", confidence=0.5, valence="neutral")]
        concepts = [ConceptLink(concept="sword", category="weapon", activation=0.8)]
        affordances = ["slash", "parry"]
        recent = ["swing succeeded"]

        BioEnrichmentPipeline._log_enrichment_contributions(memories, predictions, concepts, affordances, recent)

        records = [r for r in get_sim_records() if r["subsystem"] == "ENRICHMENT"]
        assert len(records) == 5  # hippocampus, nac, atl, component_index, working_memory

    def test_enrichment_empty_no_logs(self):
        from maxim.integration.bio_enrichment import BioEnrichmentPipeline

        set_display_tier(DisplayTier.BIO)
        BioEnrichmentPipeline._log_enrichment_contributions([], [], [], [], [])

        records = [r for r in get_sim_records() if r["subsystem"] == "ENRICHMENT"]
        assert len(records) == 0


# ---------------------------------------------------------------------------
# Track 5: Display integration — new subsystems in tier/channel maps
# ---------------------------------------------------------------------------


class TestSubsystemTierMaps:
    """Verify new subsystems are in the correct display tiers."""

    def test_new_subsystems_in_bio_tier(self):
        from maxim.simulation.sim_logger import _SUBSYSTEM_TIERS

        for sub in ["SENSOR", "ENRICHMENT", "IMAGINATION", "DISCOVERY", "GATE"]:
            assert sub in _SUBSYSTEM_TIERS, f"{sub} missing from _SUBSYSTEM_TIERS"
            assert _SUBSYSTEM_TIERS[sub] == DisplayTier.BIO, f"{sub} should be BIO tier"

    def test_new_subsystems_in_channel_map(self):
        from maxim.simulation.sim_logger import _CHANNEL_MAP

        bio_set = _CHANNEL_MAP["bio"]
        for sub in ["SENSOR", "ENRICHMENT", "IMAGINATION", "DISCOVERY", "GATE"]:
            assert sub in bio_set, f"{sub} missing from bio channel"

    def test_sem_channel(self):
        from maxim.simulation.sim_logger import _CHANNEL_MAP

        assert "sem" in _CHANNEL_MAP
        sem_set = _CHANNEL_MAP["sem"]
        for sub in ["SENSOR", "IMAGINATION", "DISCOVERY", "CEREBELLUM", "BODY", "BODY_STATE"]:
            assert sub in sem_set, f"{sub} missing from sem channel"

    def test_enrichment_channel(self):
        from maxim.simulation.sim_logger import _CHANNEL_MAP

        assert "enrichment" in _CHANNEL_MAP
        enrichment_set = _CHANNEL_MAP["enrichment"]
        assert "ENRICHMENT" in enrichment_set
        assert "GATE" in enrichment_set


class TestDisplaySubsystems:
    """Verify new subsystems are registered in MaximDisplay."""

    def test_new_subsystems_in_bio_set(self):
        from maxim.interactive.display import MaximDisplay

        bio_subs = MaximDisplay._BIO_SUBSYSTEMS
        for sub in ["sensor", "enrichment", "imagination", "discovery", "gate"]:
            assert sub in bio_subs, f"{sub} missing from MaximDisplay._BIO_SUBSYSTEMS"


# ---------------------------------------------------------------------------
# Display: Thinking panel enrichment details
# ---------------------------------------------------------------------------


class TestThinkingPanelEnrichment:
    """Verify enrichment_details flows through to the thinking panel."""

    def test_set_thinking_accepts_enrichment_details(self):
        from maxim.interactive.display import MaximDisplay

        display = MaximDisplay()
        details = {
            "hippocampus": "2 episodes: sword broke, blade stress warning",
            "nac": "swing→damage (72%)",
        }
        display.set_thinking(
            reasoning="Testing enrichment",
            cycle=1,
            max_cycles=3,
            enrichment_tags=["hippocampus", "nac"],
            enrichment_details=details,
        )
        assert display._thinking_state is not None
        assert display._thinking_state.enrichment_details == details

    def test_enrichment_details_default_empty(self):
        from maxim.interactive.display import MaximDisplay

        display = MaximDisplay()
        display.set_thinking(
            reasoning="No details",
            cycle=1,
            max_cycles=3,
        )
        assert display._thinking_state is not None
        assert display._thinking_state.enrichment_details == {}
