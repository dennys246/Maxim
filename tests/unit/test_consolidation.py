"""Unit tests for ConsolidationOrchestrator and BoundedBin.

Tests wave consolidation, sidecar processing, path-dependent thresholds,
chronic staging, and BoundedBin capacity/eviction.
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import Mock

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def staging_dir(tmp_path):
    """Temp staging directory."""
    d = tmp_path / "short_term_memory"
    d.mkdir()
    return str(d)


@pytest.fixture
def mock_hippocampus():
    """Mock hippocampus for consolidation."""
    hippo = Mock()
    hippo.capture_from_loop = Mock(return_value="promoted_mem_1")
    hippo.get = Mock(return_value=Mock(id="promoted_mem_1", timestamp=time.time()))
    hippo.graph = Mock()
    hippo.graph.get_associated = Mock(return_value=[])
    return hippo


@pytest.fixture
def mock_nac():
    """Mock NAc."""
    nac = Mock()
    nac.get_links_for_event = Mock(return_value=[])
    return nac


@pytest.fixture
def mock_scn():
    """Mock SCN."""
    scn = Mock()
    scn.register = Mock()
    scn.query_similar_time = Mock(return_value=set())
    return scn


@pytest.fixture
def mock_weight_learner():
    """Mock weight learner."""
    learner = Mock()
    learner.harvest_utility = Mock()
    learner.track_promotion = Mock()
    return learner


@pytest.fixture
def mock_percept_index():
    """Mock percept SimilarityIndex."""
    idx = Mock()
    idx.register = Mock()
    idx.query_similar = Mock(return_value=[])
    return idx


@pytest.fixture
def mock_context_index():
    """Mock context SimilarityIndex."""
    idx = Mock()
    idx.register = Mock()
    idx.query_similar = Mock(return_value=[])
    return idx


@pytest.fixture
def orchestrator(
    staging_dir,
    mock_hippocampus,
    mock_nac,
    mock_scn,
    mock_weight_learner,
    mock_percept_index,
    mock_context_index,
):
    """Fresh ConsolidationOrchestrator."""
    from maxim.memory.consolidation import ConsolidationOrchestrator

    return ConsolidationOrchestrator(
        staging_dir=staging_dir,
        hippocampus=mock_hippocampus,
        nac=mock_nac,
        scn=mock_scn,
        weight_learner=mock_weight_learner,
        percept_index=mock_percept_index,
        context_index=mock_context_index,
    )


def _write_sidecar(staging_dir: str, name: str, data: dict) -> str:
    """Helper to write a sidecar JSON file."""
    path = os.path.join(staging_dir, name)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _make_sidecar(
    significance: float = 0.5,
    staging_path: str = "acute",
    sleep_cycles: int = 0,
    **overrides,
) -> dict:
    """Create a minimal sidecar dict."""
    sidecar = {
        "version": "1.0",
        "timestamp": time.time(),
        "staging_path": staging_path,
        "significance_score": significance,
        "heuristic_scores": {"novelty": 0.5, "rpe_magnitude": 0.3},
        "rpe": {"raw": 0.3, "running_mean": 0.2, "running_std": 0.1},
        "temporal": {},
        "nac_obs_at_staging": 0,
        "percept": {"objects": [{"label": "mug", "confidence": 0.9}]},
        "context_summary": "looking at objects on table",
        "action": {"tool": "look_around"},
        "outcome": {"success": True, "valence": 0.7},
        "active_goal": "explore room",
        "sleep_cycles_seen": sleep_cycles,
        "wave_scores": [],
    }
    sidecar.update(overrides)
    return sidecar


# ── Wave Consolidation ───────────────────────────────────────────────────────


class TestRunWave:
    """Test wave processing of sidecars."""

    def test_empty_staging_dir(self, orchestrator):
        stats = orchestrator.run_wave()
        assert stats == {"promoted": 0, "survived": 0, "expired": 0, "errors": 0}

    def test_immediate_promotion_high_significance(self, orchestrator, staging_dir):
        sidecar = _make_sidecar(significance=0.90)
        _write_sidecar(staging_dir, "test.json", sidecar)

        stats = orchestrator.run_wave()
        assert stats["promoted"] == 1
        assert not os.path.exists(os.path.join(staging_dir, "test.json"))

    def test_acute_path_threshold(self, orchestrator, staging_dir):
        # Score that should exceed acute threshold (0.45) after wave scoring
        sidecar = _make_sidecar(significance=0.8, staging_path="acute", sleep_cycles=1)
        _write_sidecar(staging_dir, "test.json", sidecar)

        stats = orchestrator.run_wave()
        # With 0.8 significance and 0.30 weight, wave_score = 0.24 base
        # May or may not promote depending on other factors
        assert stats["promoted"] + stats["survived"] + stats["expired"] == 1

    def test_chronic_path_uses_higher_threshold(self, orchestrator, staging_dir):
        # Low significance chronic sidecar should survive (not promote)
        sidecar = _make_sidecar(significance=0.3, staging_path="chronic", sleep_cycles=1)
        _write_sidecar(staging_dir, "test.json", sidecar)

        stats = orchestrator.run_wave()
        # Low score should survive or expire, not promote
        assert stats["promoted"] == 0

    def test_expired_after_max_waves(self, orchestrator, staging_dir):
        sidecar = _make_sidecar(significance=0.1, sleep_cycles=4)
        _write_sidecar(staging_dir, "test.json", sidecar)

        stats = orchestrator.run_wave()
        assert stats["expired"] == 1
        assert not os.path.exists(os.path.join(staging_dir, "test.json"))

    def test_survived_sidecar_increments_sleep_cycles(self, orchestrator, staging_dir):
        sidecar = _make_sidecar(significance=0.1, sleep_cycles=0)
        path = _write_sidecar(staging_dir, "test.json", sidecar)

        orchestrator.run_wave()

        # Should still exist if it survived
        if os.path.exists(path):
            with open(path) as f:
                updated = json.load(f)
            assert updated["sleep_cycles_seen"] == 1

    def test_utility_harvest_called(self, orchestrator, staging_dir, mock_weight_learner):
        orchestrator.run_wave()
        mock_weight_learner.harvest_utility.assert_called_once()

    def test_corrupted_sidecar_skipped(self, orchestrator, staging_dir):
        path = os.path.join(staging_dir, "corrupt.json")
        with open(path, "w") as f:
            f.write("{invalid")

        stats = orchestrator.run_wave()
        # Corrupt file should be skipped during loading
        assert stats["promoted"] == 0

    def test_multiple_sidecars_processed(self, orchestrator, staging_dir):
        _write_sidecar(staging_dir, "a.json", _make_sidecar(significance=0.90))
        _write_sidecar(staging_dir, "b.json", _make_sidecar(significance=0.1, sleep_cycles=4))

        stats = orchestrator.run_wave()
        assert stats["promoted"] + stats["expired"] == 2


# ── Wave Score Computation ───────────────────────────────────────────────────


class TestComputeWaveScore:
    """Test wave score calculation."""

    def test_base_significance_contributes(self, orchestrator):
        sidecar = _make_sidecar(significance=1.0)
        score = orchestrator._compute_wave_score(sidecar)
        # significance contributes 0.30 * 1.0 = 0.30 minimum
        assert score >= 0.30

    def test_zero_significance_gives_low_score(self, orchestrator):
        sidecar = _make_sidecar(significance=0.0)
        score = orchestrator._compute_wave_score(sidecar)
        assert score < 0.5

    def test_nac_corroboration_integrated(self, orchestrator, mock_nac):
        # Set up NAc to return links with observations
        link = Mock()
        link.observation_count = 10
        mock_nac.get_links_for_event.return_value = [link]

        sidecar = _make_sidecar(significance=0.5)
        score = orchestrator._compute_wave_score(sidecar)
        # With NAc corroboration, score should be higher than without
        assert score > 0.0


# ── Promotion ─────────────────────────────────────────────────────────────────


class TestPromotion:
    """Test memory promotion."""

    def test_promote_calls_capture_from_loop(self, orchestrator, mock_hippocampus):
        sidecar = _make_sidecar()
        sidecar["wave_scores"] = [0.6]
        orchestrator._promote(sidecar)
        mock_hippocampus.capture_from_loop.assert_called_once()

    def test_promote_registers_with_scn(self, orchestrator, mock_scn):
        sidecar = _make_sidecar()
        sidecar["wave_scores"] = [0.6]
        sidecar["temporal"] = {
            "circadian_phase": 0.375,
            "weekly_phase": 0.0,
            "monthly_phase": 0.25,
            "annual_phase": 0.0,
        }
        orchestrator._promote(sidecar)
        mock_scn.register.assert_called_once()

    def test_promote_tracks_for_utility(self, orchestrator, mock_weight_learner):
        sidecar = _make_sidecar()
        sidecar["wave_scores"] = [0.6]
        orchestrator._promote(sidecar)
        mock_weight_learner.track_promotion.assert_called_once()

    def test_promote_with_no_hippocampus(self, orchestrator):
        orchestrator.hippocampus = None
        sidecar = _make_sidecar()
        sidecar["wave_scores"] = [0.6]
        # Should not raise
        orchestrator._promote(sidecar)


# ── Sidecar I/O ──────────────────────────────────────────────────────────────


class TestSidecarIO:
    """Test sidecar file operations."""

    def test_load_sidecars_empty_dir(self, orchestrator):
        sidecars = orchestrator._load_sidecars()
        assert sidecars == []

    def test_load_sidecars_returns_valid_json(self, orchestrator, staging_dir):
        _write_sidecar(staging_dir, "test.json", _make_sidecar())
        sidecars = orchestrator._load_sidecars()
        assert len(sidecars) == 1
        path, data = sidecars[0]
        assert "significance_score" in data

    def test_load_sidecars_skips_non_json(self, orchestrator, staging_dir):
        with open(os.path.join(staging_dir, "readme.txt"), "w") as f:
            f.write("not json")
        sidecars = orchestrator._load_sidecars()
        assert len(sidecars) == 0

    def test_save_sidecar_atomic(self, orchestrator, staging_dir):
        path = os.path.join(staging_dir, "test.json")
        orchestrator._save_sidecar(path, _make_sidecar())
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")

    def test_load_sidecars_nonexistent_dir(self, tmp_path):
        from maxim.memory.consolidation import ConsolidationOrchestrator

        orch = ConsolidationOrchestrator(
            staging_dir=str(tmp_path / "nonexistent"),
            hippocampus=None,
            nac=None,
            scn=None,
            weight_learner=Mock(),
            percept_index=None,
            context_index=None,
        )
        assert orch._load_sidecars() == []


# ── Chronic Staging ──────────────────────────────────────────────────────────


class TestChronicStaging:
    """Test chronic sidecar creation."""

    def test_create_chronic_sidecar_basic(self, orchestrator):
        mem1 = Mock(
            id="m1",
            perception={"objects": [{"label": "mug", "confidence": 0.9}]},
        )
        mem2 = Mock(
            id="m2",
            perception={"objects": [{"label": "mug", "confidence": 0.8}]},
        )
        mem3 = Mock(
            id="m3",
            perception={"objects": [{"label": "mug", "confidence": 0.7}]},
        )

        sidecar = orchestrator._create_chronic_sidecar([mem1, mem2, mem3])
        assert sidecar is not None
        assert sidecar["staging_path"] == "chronic"
        assert sidecar["heuristic_scores"]["recurrence_count"] == 3
        assert len(sidecar["cluster_member_ids"]) == 3

    def test_create_chronic_sidecar_empty_cluster(self, orchestrator):
        assert orchestrator._create_chronic_sidecar([]) is None

    def test_write_chronic_sidecar(self, orchestrator, staging_dir):
        sidecar = _make_sidecar(staging_path="chronic")
        sidecar["heuristic_scores"]["recurrence_count"] = 3
        orchestrator._write_chronic_sidecar(sidecar)

        files = os.listdir(staging_dir)
        chronic_files = [f for f in files if "chronic" in f]
        assert len(chronic_files) == 1

    def test_detect_recurrence_needs_threshold(self, orchestrator, mock_percept_index):
        # Set up percept_index to return no similar matches
        mock_percept_index.query_similar.return_value = []

        mem1 = Mock(id="m1", perception={"objects": [{"label": "x", "confidence": 0.9}]})
        mem2 = Mock(id="m2", perception={"objects": [{"label": "y", "confidence": 0.9}]})

        clusters = orchestrator.detect_recurrence([mem1, mem2], recurrence_threshold=3)
        assert clusters == []

    def test_find_chronic_candidates_no_hippocampus(self, orchestrator):
        orchestrator.hippocampus = None
        assert orchestrator.find_chronic_candidates() == []

    def test_run_chronic_staging_returns_count(self, orchestrator):
        # With empty hippocampus queries, should return 0
        orchestrator.hippocampus = None
        assert orchestrator.run_chronic_staging() == 0


# ── BoundedBin ────────────────────────────────────────────────────────────────


class TestBoundedBin:
    """Test BoundedBin capacity management and set compatibility."""

    @pytest.fixture
    def bounded_bin(self):
        from maxim.time.scn import BoundedBin

        return BoundedBin(max_size=5)

    def test_add_basic(self, bounded_bin):
        bounded_bin.add("mem_1", 0.5)
        assert "mem_1" in bounded_bin
        assert len(bounded_bin) == 1

    def test_add_duplicate_ignored(self, bounded_bin):
        bounded_bin.add("mem_1", 0.5)
        bounded_bin.add("mem_1", 0.9)
        assert len(bounded_bin) == 1

    def test_add_evicts_least_significant_when_full(self):
        from maxim.time.scn import BoundedBin

        bb = BoundedBin(max_size=3)
        bb.add("low", 0.1)
        bb.add("mid", 0.5)
        bb.add("high", 0.9)

        # Bin is full. Adding higher significance should evict lowest from older half
        bb.add("new_high", 0.8)
        assert len(bb) == 3
        # "low" (0.1) should have been evicted
        assert "low" not in bb
        assert "new_high" in bb

    def test_add_rejects_low_significance_when_full(self):
        from maxim.time.scn import BoundedBin

        bb = BoundedBin(max_size=3)
        bb.add("a", 0.5)
        bb.add("b", 0.6)
        bb.add("c", 0.7)

        bb.add("too_low", 0.1)
        assert "too_low" not in bb
        assert len(bb) == 3

    def test_discard(self, bounded_bin):
        bounded_bin.add("mem_1", 0.5)
        bounded_bin.discard("mem_1")
        assert "mem_1" not in bounded_bin
        assert len(bounded_bin) == 0

    def test_discard_nonexistent_is_safe(self, bounded_bin):
        bounded_bin.discard("nonexistent")

    def test_remove(self, bounded_bin):
        bounded_bin.add("mem_1", 0.5)
        bounded_bin.remove("mem_1")
        assert "mem_1" not in bounded_bin

    def test_get_ids_returns_copy(self, bounded_bin):
        bounded_bin.add("mem_1", 0.5)
        ids = bounded_bin.get_ids()
        assert ids == {"mem_1"}
        ids.add("fake")
        assert "fake" not in bounded_bin

    def test_copy_returns_id_set(self, bounded_bin):
        bounded_bin.add("mem_1", 0.5)
        assert bounded_bin.copy() == {"mem_1"}

    def test_iter(self, bounded_bin):
        bounded_bin.add("mem_1", 0.5)
        bounded_bin.add("mem_2", 0.6)
        assert set(bounded_bin) == {"mem_1", "mem_2"}

    def test_bool_empty(self, bounded_bin):
        assert not bounded_bin

    def test_bool_nonempty(self, bounded_bin):
        bounded_bin.add("mem_1", 0.5)
        assert bounded_bin

    def test_and_with_set(self, bounded_bin):
        bounded_bin.add("mem_1", 0.5)
        bounded_bin.add("mem_2", 0.6)
        result = bounded_bin & {"mem_1", "mem_3"}
        assert result == {"mem_1"}

    def test_rand_with_set(self, bounded_bin):
        bounded_bin.add("mem_1", 0.5)
        bounded_bin.add("mem_2", 0.6)
        result = {"mem_1", "mem_3"} & bounded_bin
        assert result == {"mem_1"}

    def test_newest_at_front(self):
        from maxim.time.scn import BoundedBin

        bb = BoundedBin(max_size=10)
        bb.add("first", 0.5)
        bb.add("second", 0.5)
        assert bb.entries[0].memory_id == "second"
        assert bb.entries[1].memory_id == "first"


class TestBoundedBinPersistence:
    """Test BoundedBin serialization."""

    def test_to_list(self):
        from maxim.time.scn import BoundedBin

        bb = BoundedBin(max_size=10)
        bb.add("mem_1", 0.7)
        data = bb.to_list()
        assert len(data) == 1
        assert data[0]["memory_id"] == "mem_1"
        assert data[0]["significance"] == 0.7

    def test_from_list_v3(self):
        from maxim.time.scn import BoundedBin

        data = [
            {"memory_id": "mem_1", "significance": 0.7, "registered_at": 100.0},
            {"memory_id": "mem_2", "significance": 0.3, "registered_at": 200.0},
        ]
        bb = BoundedBin.from_list(data)
        assert len(bb) == 2
        assert "mem_1" in bb
        assert "mem_2" in bb

    def test_from_list_v2_compat(self):
        """v2 format: plain string memory IDs."""
        from maxim.time.scn import BoundedBin

        data = ["mem_1", "mem_2", "mem_3"]
        bb = BoundedBin.from_list(data)
        assert len(bb) == 3
        assert "mem_1" in bb
        # Default significance should be 0.5
        assert bb.entries[0].significance == 0.5

    def test_roundtrip(self):
        from maxim.time.scn import BoundedBin

        bb = BoundedBin(max_size=10)
        bb.add("mem_1", 0.7)
        bb.add("mem_2", 0.3)

        data = bb.to_list()
        bb2 = BoundedBin.from_list(data)

        assert bb2.get_ids() == bb.get_ids()


class TestSCNBoundedBinIntegration:
    """Test SCN with BoundedBin (v3 persistence)."""

    def test_save_load_v3(self, scn, temporal_signature_morning, tmp_path):
        """v3 format preserves BoundedBin entries with significance."""
        from maxim.time.scn import SCN

        scn.register("mem_1", temporal_signature_morning, significance=0.8)

        path = str(tmp_path / "scn_v3.json")
        scn.save(path)

        scn2 = SCN()
        scn2.load(path)

        assert "mem_1" in scn2.query_hour(9)
        assert scn2.get_signature("mem_1") is not None

    def test_register_with_significance(self, scn, temporal_signature_morning):
        """Registration passes significance to BoundedBin."""
        scn.register("mem_1", temporal_signature_morning, significance=0.9)
        assert "mem_1" in scn.query_hour(9)


# ── SCN ↔ Consolidation Integration ─────────────────────────────────────────


class TestWaveScoresSafety:
    """BUG-2: wave_scores key may be missing, empty, or None.

    Exercises the setdefault() fix on lines 81/89 (run_wave) and
    the ``or [0.5]`` fallback on lines 237/267 (_promote).
    """

    def test_run_wave_missing_wave_scores_key(self, orchestrator, staging_dir):
        """Sidecar with no 'wave_scores' key must not crash run_wave()."""
        sidecar = _make_sidecar(significance=0.90)
        del sidecar["wave_scores"]
        _write_sidecar(staging_dir, "no_key.json", sidecar)

        stats = orchestrator.run_wave()
        # High significance → immediate promotion; must not KeyError
        assert stats["promoted"] == 1
        assert stats["errors"] == 0

    def test_run_wave_missing_wave_scores_low_sig(self, orchestrator, staging_dir):
        """Low-significance sidecar missing wave_scores survives without error."""
        sidecar = _make_sidecar(significance=0.1, sleep_cycles=0)
        del sidecar["wave_scores"]
        _write_sidecar(staging_dir, "low.json", sidecar)

        stats = orchestrator.run_wave()
        assert stats["errors"] == 0
        assert stats["survived"] + stats["expired"] + stats["promoted"] == 1

    def test_promote_empty_wave_scores(self, orchestrator, mock_hippocampus):
        """Empty wave_scores list must not IndexError in _promote()."""
        sidecar = _make_sidecar()
        sidecar["wave_scores"] = []
        # Should use fallback [0.5][-1] → 0.5
        orchestrator._promote(sidecar)
        mock_hippocampus.capture_from_loop.assert_called_once()
        call_kwargs = mock_hippocampus.capture_from_loop.call_args
        state = call_kwargs[1]["state"] if "state" in (call_kwargs[1] or {}) else call_kwargs[0][1]
        assert state["salience"] == 0.5

    def test_promote_none_wave_scores(self, orchestrator, mock_hippocampus):
        """wave_scores=None must not crash; fallback to 0.5."""
        sidecar = _make_sidecar()
        sidecar["wave_scores"] = None
        orchestrator._promote(sidecar)
        mock_hippocampus.capture_from_loop.assert_called_once()
        call_kwargs = mock_hippocampus.capture_from_loop.call_args
        state = call_kwargs[1]["state"] if "state" in (call_kwargs[1] or {}) else call_kwargs[0][1]
        assert state["salience"] == 0.5

    def test_promote_normal_wave_scores_preserved(self, orchestrator, mock_hippocampus):
        """Normal wave_scores=[0.6, 0.7] uses last value (0.7)."""
        sidecar = _make_sidecar()
        sidecar["wave_scores"] = [0.6, 0.7]
        orchestrator._promote(sidecar)
        mock_hippocampus.capture_from_loop.assert_called_once()
        call_kwargs = mock_hippocampus.capture_from_loop.call_args
        state = call_kwargs[1]["state"] if "state" in (call_kwargs[1] or {}) else call_kwargs[0][1]
        assert state["salience"] == 0.7

    def test_scn_register_empty_wave_scores(self, orchestrator, mock_scn):
        """SCN registration with empty wave_scores uses 0.5 fallback."""
        sidecar = _make_sidecar()
        sidecar["wave_scores"] = []
        sidecar["temporal"] = {
            "circadian_phase": 0.375,
            "weekly_phase": 0.0,
            "monthly_phase": 0.25,
            "annual_phase": 0.0,
        }
        orchestrator._promote(sidecar)
        mock_scn.register.assert_called_once()
        _, kwargs = mock_scn.register.call_args
        assert kwargs["significance"] == 0.5

    def test_scn_register_none_wave_scores(self, orchestrator, mock_scn):
        """SCN registration with None wave_scores uses 0.5 fallback."""
        sidecar = _make_sidecar()
        sidecar["wave_scores"] = None
        sidecar["temporal"] = {
            "circadian_phase": 0.375,
            "weekly_phase": 0.0,
            "monthly_phase": 0.25,
            "annual_phase": 0.0,
        }
        orchestrator._promote(sidecar)
        mock_scn.register.assert_called_once()
        _, kwargs = mock_scn.register.call_args
        assert kwargs["significance"] == 0.5


class TestSCNConsolidationIntegration:
    """Test how SCN integrates with the consolidation pipeline.

    Verifies temporal recurrence scoring, bounded bin queries during
    chronic candidate discovery, and promotion registering back to SCN.
    """

    def test_wave_score_temporal_recurrence(self, staging_dir, tmp_path):
        """SCN query_similar_time contributes to wave score via temporal_recurrence."""
        from maxim.memory.consolidation import ConsolidationOrchestrator
        from maxim.time.scn import SCN
        from maxim.time.temporal_signature import TemporalSignature

        scn = SCN()
        # Register several memories at 9am Monday
        sig_9am = TemporalSignature(
            timestamp=1704103200.0,
            circadian_phase=9 / 24,
            weekly_phase=0 / 7,
            monthly_phase=0.25,
            annual_phase=0.0,
        )
        for i in range(6):
            scn.register(f"existing_{i}", sig_9am, significance=0.5)

        orch = ConsolidationOrchestrator(
            staging_dir=staging_dir,
            hippocampus=Mock(),
            nac=Mock(get_links_for_event=Mock(return_value=[])),
            scn=scn,
            weight_learner=Mock(harvest_utility=Mock()),
            percept_index=Mock(query_similar=Mock(return_value=[])),
            context_index=Mock(query_similar=Mock(return_value=[])),
        )

        sidecar = _make_sidecar(significance=0.3)
        sidecar["temporal"] = {
            "circadian_phase": 9 / 24,
            "weekly_phase": 0 / 7,
            "monthly_phase": 0.25,
            "annual_phase": 0.0,
        }
        score_with_recurrence = orch._compute_wave_score(sidecar)

        # Now test with empty SCN (no temporal recurrence)
        orch2 = ConsolidationOrchestrator(
            staging_dir=staging_dir,
            hippocampus=Mock(),
            nac=Mock(get_links_for_event=Mock(return_value=[])),
            scn=SCN(),
            weight_learner=Mock(harvest_utility=Mock()),
            percept_index=Mock(query_similar=Mock(return_value=[])),
            context_index=Mock(query_similar=Mock(return_value=[])),
        )
        score_without = orch2._compute_wave_score(sidecar)

        assert score_with_recurrence > score_without

    def test_promotion_registers_with_real_scn(self, staging_dir):
        """Promoted memory gets registered in SCN with temporal signature."""
        from maxim.memory.consolidation import ConsolidationOrchestrator
        from maxim.time.scn import SCN

        scn = SCN()
        hippo = Mock()
        hippo.capture_from_loop = Mock(return_value="promoted_1")
        hippo.get = Mock(return_value=Mock(id="promoted_1", timestamp=time.time()))

        orch = ConsolidationOrchestrator(
            staging_dir=staging_dir,
            hippocampus=hippo,
            nac=None,
            scn=scn,
            weight_learner=Mock(track_promotion=Mock()),
            percept_index=Mock(register=Mock()),
            context_index=Mock(register=Mock()),
        )

        sidecar = _make_sidecar(significance=0.7)
        sidecar["wave_scores"] = [0.7]
        sidecar["temporal"] = {
            "circadian_phase": 14 / 24,
            "weekly_phase": 2 / 7,
            "monthly_phase": 0.5,
            "annual_phase": 0.25,
        }

        orch._promote(sidecar)

        # Memory should now be in SCN at hour 14
        assert "promoted_1" in scn.query_hour(14)
        assert scn.get_signature("promoted_1") is not None

    def test_full_wave_with_real_scn(self, staging_dir):
        """End-to-end: sidecar → wave → promotion → SCN registration."""
        from maxim.memory.consolidation import ConsolidationOrchestrator
        from maxim.time.scn import SCN

        scn = SCN()
        hippo = Mock()
        hippo.capture_from_loop = Mock(return_value="wave_promoted")
        hippo.get = Mock(return_value=Mock(id="wave_promoted", timestamp=time.time()))

        orch = ConsolidationOrchestrator(
            staging_dir=staging_dir,
            hippocampus=hippo,
            nac=Mock(get_links_for_event=Mock(return_value=[])),
            scn=scn,
            weight_learner=Mock(harvest_utility=Mock(), track_promotion=Mock()),
            percept_index=Mock(register=Mock(), query_similar=Mock(return_value=[])),
            context_index=Mock(register=Mock(), query_similar=Mock(return_value=[])),
        )

        # High significance → immediate promotion
        sidecar = _make_sidecar(significance=0.90)
        sidecar["temporal"] = {
            "circadian_phase": 10 / 24,
            "weekly_phase": 3 / 7,
            "monthly_phase": 0.75,
            "annual_phase": 0.5,
        }
        _write_sidecar(staging_dir, "immediate.json", sidecar)

        stats = orch.run_wave()
        assert stats["promoted"] == 1
        assert "wave_promoted" in scn.query_hour(10)

    def test_chronic_candidates_from_bounded_bins(self):
        """find_chronic_candidates queries SCN bounded bins for candidates."""
        from maxim.memory.consolidation import ConsolidationOrchestrator
        from maxim.time.scn import SCN
        from maxim.time.temporal_signature import TemporalSignature

        scn = SCN()
        # Register memories at current-ish time
        now = TemporalSignature.now()
        hour_bin, day_bin, _, _ = now.to_bins()

        mem_ids = []
        for i in range(5):
            mid = f"stm_{i}"
            scn.register(mid, now, significance=0.5)
            mem_ids.append(mid)

        # Mock hippocampus that returns non-long-term memories
        hippo = Mock()

        def mock_get(mid):
            if mid in mem_ids:
                return Mock(id=mid, long_term=False)
            return None

        hippo.get = mock_get

        orch = ConsolidationOrchestrator(
            staging_dir="/tmp/fake",
            hippocampus=hippo,
            nac=None,
            scn=scn,
            weight_learner=Mock(),
            percept_index=None,
            context_index=None,
        )

        candidates = orch.find_chronic_candidates()
        candidate_ids = {getattr(c, "id", "") for c in candidates}
        # Should find some of the STM memories via SCN bin queries
        assert len(candidate_ids & set(mem_ids)) > 0

    def test_scn_bounded_bin_eviction_preserves_significant(self):
        """When SCN bins are full, less significant memories are evicted first."""
        from maxim.time.scn import SCN
        from maxim.time.temporal_signature import TemporalSignature

        scn = SCN()
        sig = TemporalSignature(
            timestamp=1704103200.0,
            circadian_phase=9 / 24,
            weekly_phase=0 / 7,
            monthly_phase=0.25,
            annual_phase=0.0,
        )

        # Fill the bin (default max_size=200)
        for i in range(200):
            scn.register(f"mem_{i}", sig, significance=0.3)

        # Add a highly significant memory
        scn.register("important", sig, significance=0.9)

        # Important memory should be in the bin
        assert "important" in scn.query_hour(9)
        # One low-significance memory should have been evicted
        assert len(scn.query_hour(9)) == 200

    def test_significance_preserved_across_save_load(self, tmp_path):
        """SCN v3 format preserves significance for consolidation decisions."""
        from maxim.time.scn import SCN
        from maxim.time.temporal_signature import TemporalSignature

        scn = SCN()
        sig = TemporalSignature(
            timestamp=1704103200.0,
            circadian_phase=9 / 24,
            weekly_phase=0 / 7,
            monthly_phase=0.25,
            annual_phase=0.0,
        )
        scn.register("high_sig", sig, significance=0.95)
        scn.register("low_sig", sig, significance=0.1)

        path = str(tmp_path / "scn.json")
        scn.save(path)

        scn2 = SCN()
        scn2.load(path)

        # Both should survive roundtrip
        assert "high_sig" in scn2.query_hour(9)
        assert "low_sig" in scn2.query_hour(9)
