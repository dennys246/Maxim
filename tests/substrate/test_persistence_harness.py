"""Tests for the persistence subprocess harness (S3 — simulator_upgrades_plan)."""

from __future__ import annotations

from tests.substrate.persistence_harness import run_round_trip, _compare_results


# ── Hippocampus round-trip ───────────────────────────────────────────────────


class TestHippocampusRoundTrip:
    def test_empty_hippocampus(self, persistence_round_trip):
        """Empty hippocampus survives round-trip."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        h = Hippocampus(config=HippocampusConfig())
        result = persistence_round_trip(
            state={"hippocampus": h},
            probe="tests.substrate.probes:hippocampus_episode_count",
        )
        assert result.success, f"Round-trip failed: {result.error}"
        assert result.pre_shutdown == 0
        assert result.post_reload == 0

    def test_hippocampus_with_episodes(self, persistence_round_trip):
        """Hippocampus with stored episodes survives round-trip."""
        import time
        from uuid import uuid4

        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
        from maxim.memory.types import EpisodicMemory, Perception

        h = Hippocampus(config=HippocampusConfig())
        for i in range(3):
            ep = EpisodicMemory(
                id=str(uuid4()),
                timestamp=time.time(),
                perception=Perception(cli_input=f"test input {i}"),
            )
            h.store(ep)

        result = persistence_round_trip(
            state={"hippocampus": h},
            probe="tests.substrate.probes:hippocampus_episode_count",
        )
        assert result.success, f"Round-trip failed: {result.error}"
        assert result.pre_shutdown == 3
        assert result.post_reload == 3


# ── NAc round-trip ───────────────────────────────────────────────────────────


class TestNAcRoundTrip:
    def test_empty_nac(self, persistence_round_trip):
        """Empty NAc survives round-trip."""
        from maxim.decisions.nac import NAc

        n = NAc()
        result = persistence_round_trip(
            state={"nac": n},
            probe="tests.substrate.probes:nac_link_count",
        )
        assert result.success, f"Round-trip failed: {result.error}"
        assert result.pre_shutdown == 0
        assert result.post_reload == 0

    def test_nac_with_observations(self, persistence_round_trip):
        """NAc with observed causal links survives round-trip."""
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc

        n = NAc()
        n.observe(
            event_type="action",
            event_signature="touch_hot_stove",
            outcome_type="pain",
            outcome_signature="burn",
            outcome_valence=Valence.NEGATIVE,
            delta_seconds=0.5,
        )
        n.observe(
            event_type="action",
            event_signature="eat_food",
            outcome_type="reward",
            outcome_signature="satiation",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=1.0,
        )

        result = persistence_round_trip(
            state={"nac": n},
            probe="tests.substrate.probes:nac_link_count",
        )
        assert result.success, f"Round-trip failed: {result.error}"
        assert result.pre_shutdown == 2
        assert result.post_reload == 2

    def test_nac_link_signatures_match(self, persistence_round_trip):
        """NAc causal link details are identical after round-trip."""
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc

        n = NAc()
        n.observe(
            event_type="action",
            event_signature="press_button",
            outcome_type="reward",
            outcome_signature="light_on",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=0.2,
        )

        result = persistence_round_trip(
            state={"nac": n},
            probe="tests.substrate.probes:nac_link_signatures",
            tolerance=1e-4,
        )
        assert result.success, f"Round-trip failed: {result.error}"
        assert len(result.pre_shutdown) == 1
        assert result.pre_shutdown[0]["event"] == "press_button"


# ── PerceptTraceBuffer round-trip ────────────────────────────────────────────


class TestPerceptTraceBufferRoundTrip:
    def test_trace_buffer_round_trip(self, persistence_round_trip):
        """PerceptTraceBuffer entries survive manual serialize/deserialize."""
        from maxim.memory.percept_trace_buffer import PerceptTraceBuffer

        ptb = PerceptTraceBuffer()
        ptb.record("agent_0", "percept_1", activation=0.9)
        ptb.record("agent_0", "percept_2", activation=0.7)
        ptb.record("agent_1", "percept_3", activation=0.5)

        result = persistence_round_trip(
            state={"percept_trace_buffer": ptb},
            probe="tests.substrate.probes:percept_trace_entry_count",
        )
        assert result.success, f"Round-trip failed: {result.error}"
        assert result.pre_shutdown == 3
        assert result.post_reload == 3


# ── Combined state round-trip ────────────────────────────────────────────────


class TestCombinedRoundTrip:
    def test_multiple_components(self, persistence_round_trip):
        """Multiple bio-system components survive round-trip together."""
        import time

        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
        from maxim.memory.percept_trace_buffer import PerceptTraceBuffer
        from maxim.memory.types import EpisodicMemory, Perception

        h = Hippocampus(config=HippocampusConfig())
        h.store(EpisodicMemory(id="ep_1", timestamp=time.time(), perception=Perception(cli_input="hello")))

        n = NAc()
        n.observe(
            event_type="action",
            event_signature="greet",
            outcome_type="reward",
            outcome_signature="friendliness",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=0.3,
        )

        ptb = PerceptTraceBuffer()
        ptb.record("agent_0", "p1", activation=0.8)

        result = persistence_round_trip(
            state={
                "hippocampus": h,
                "nac": n,
                "percept_trace_buffer": ptb,
            },
            probe="tests.substrate.probes:combined_state_summary",
        )
        assert result.success, f"Round-trip failed: {result.error}"
        assert result.pre_shutdown["hippocampus_episodes"] == 1
        assert result.pre_shutdown["nac_links"] == 1
        assert result.pre_shutdown["percept_trace_entries"] == 1


# ── Error handling ───────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_bad_probe_ref(self):
        """Invalid probe reference fails with clear error."""
        result = run_round_trip(
            state={"hippocampus": None},
            probe="bad_probe_no_colon",
        )
        assert not result.success
        assert "module.path:function_name" in result.error

    def test_nonexistent_probe(self):
        """Probe pointing to nonexistent function fails clearly."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        result = run_round_trip(
            state={"hippocampus": Hippocampus(config=HippocampusConfig())},
            probe="tests.substrate.probes:nonexistent_function",
        )
        assert not result.success
        assert "not found" in result.error

    def test_no_components_saved(self):
        """Passing all None components fails with clear error."""
        result = run_round_trip(
            state={"hippocampus": None},
            probe="tests.substrate.probes:hippocampus_episode_count",
        )
        assert not result.success
        assert "nothing to round-trip" in result.error.lower()

    def test_child_crash_reports_clearly(self, persistence_round_trip):
        """A child that fails to load state reports the error."""
        # Write a corrupt state file that will cause load failure
        from pathlib import Path
        from unittest.mock import MagicMock

        mock_comp = MagicMock()
        mock_comp.save = MagicMock(side_effect=lambda p: Path(p).write_text("not valid json {{{"))

        result = persistence_round_trip(
            state={"hippocampus": mock_comp},
            probe="tests.substrate.probes:hippocampus_episode_count",
        )
        # Child should fail trying to load corrupt JSON
        assert not result.success
        assert result.child_returncode != 0 or "error" in result.error.lower()


# ── Comparison utilities ─────────────────────────────────────────────────────


class TestCompareResults:
    def test_exact_match(self):
        assert _compare_results(42, 42, 0.0)
        assert not _compare_results(42, 43, 0.0)

    def test_tolerance(self):
        assert _compare_results(1.0, 1.001, 0.01)
        assert not _compare_results(1.0, 1.1, 0.01)

    def test_dict_comparison(self):
        assert _compare_results({"a": 1}, {"a": 1}, 0.0)
        assert not _compare_results({"a": 1}, {"a": 2}, 0.0)
        assert _compare_results({"a": 1.0}, {"a": 1.001}, 0.01)

    def test_list_comparison(self):
        assert _compare_results([1, 2, 3], [1, 2, 3], 0.0)
        assert not _compare_results([1, 2], [1, 2, 3], 0.0)

    def test_nested_comparison(self):
        pre = {"counts": [1, 2], "score": 0.95}
        post = {"counts": [1, 2], "score": 0.951}
        assert _compare_results(pre, post, 0.01)
        assert not _compare_results(pre, post, 0.0)
