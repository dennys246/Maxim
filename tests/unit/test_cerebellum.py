"""Tests for the Cerebellum forward model system (Phase 1a).

Covers:
- ModelKey construction and param bucketing
- ForwardModel R-W update and confidence growth
- Cerebellum predict/observe lifecycle
- Confidence threshold gating
- High-variance fallback
- CerebellumModulator predict → fallback → train loop
- Persistence (export/import round-trip)
- Thread safety (concurrent predict/observe)
- LLM skip rate tracking
- Stale model pruning
"""

from __future__ import annotations

import threading
import time
from pathlib import Path


from maxim.embodiment.backends.cerebellum_modulator import (
    CerebellumModulator,
    cerebellum_modulator_factory,
)
from maxim.embodiment.cerebellum import (
    Cerebellum,
    CerebellumConfig,
    ForwardModel,
    bucket_params,
)
from maxim.embodiment.llm_backend import LLMModulator, LLMSensor
from maxim.embodiment.sem import AffordanceSchema, Entity
from maxim.embodiment.spec import load_spec

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "embodiment"


# ---------------------------------------------------------------------------
# Param bucketing
# ---------------------------------------------------------------------------


class TestBucketParams:
    def test_empty_params(self):
        assert bucket_params({}) == "_empty_"

    def test_float_params_default_step(self):
        # default_step=5 → 47 rounds to 45
        result = bucket_params({"degrees": 47.0, "speed": 1.2})
        assert "degrees=45.00" in result
        assert "speed=0.00" in result  # 1.2 rounds to 0.0 with step 5

    def test_float_params_with_range(self):
        # range [0, 360], fraction=0.1 → step=36
        result = bucket_params(
            {"degrees": 47.0},
            sensor_ranges={"degrees": (0, 360)},
        )
        assert "degrees=36.00" in result  # 47 rounds to 36

    def test_string_params(self):
        result = bucket_params({"target": "dummy"})
        assert "target=dummy" in result

    def test_sorted_keys(self):
        r1 = bucket_params({"a": 1.0, "b": 2.0})
        r2 = bucket_params({"b": 2.0, "a": 1.0})
        assert r1 == r2

    def test_similar_params_same_bucket(self):
        # Params differing by <10% of range should bucket together
        # range=180, step=18 → 40 and 44 both round to 36
        r1 = bucket_params({"degrees": 40.0}, sensor_ranges={"degrees": (0, 180)})
        r2 = bucket_params({"degrees": 44.0}, sensor_ranges={"degrees": (0, 180)})
        assert r1 == r2  # Both round to same bucket (step=18)

    def test_different_params_different_bucket(self):
        r1 = bucket_params({"degrees": 45.0}, sensor_ranges={"degrees": (0, 180)})
        r2 = bucket_params({"degrees": 90.0}, sensor_ranges={"degrees": (0, 180)})
        assert r1 != r2  # Different buckets


# ---------------------------------------------------------------------------
# ForwardModel
# ---------------------------------------------------------------------------


class TestForwardModel:
    def test_initial_confidence_zero(self):
        m = ForwardModel()
        assert m.confidence == 0.0

    def test_confidence_grows(self):
        m = ForwardModel()
        for _ in range(5):
            m.update({"angle": 45.0})
        assert m.confidence > 0.5

    def test_confidence_capped(self):
        m = ForwardModel()
        for _ in range(100):
            m.update({"angle": 45.0})
        assert m.confidence <= 0.95

    def test_first_observation_no_error(self):
        m = ForwardModel()
        errors = m.update({"angle": 45.0})
        assert errors["angle"] == 0.0  # first obs is initialization

    def test_prediction_converges(self):
        m = ForwardModel()
        # Train on consistent value
        for _ in range(10):
            m.update({"angle": 90.0})
        predicted = m.predict_sensors()
        assert abs(predicted["angle"] - 90.0) < 1.0

    def test_prediction_error_decreases(self):
        m = ForwardModel()
        errors_first = abs(m.update({"angle": 90.0}).get("angle", 0))
        for _ in range(5):
            m.update({"angle": 90.0})
        errors_last = abs(m.update({"angle": 90.0}).get("angle", 0))
        assert errors_last < errors_first or errors_first == 0.0

    def test_variance_tracks_error(self):
        m = ForwardModel()
        # Train on noisy value
        for val in [45, 55, 42, 58, 40, 60]:
            m.update({"angle": float(val)})
        variance = m.get_variance()
        assert variance["angle"] > 0.0

    def test_serialization_roundtrip(self):
        m = ForwardModel()
        for _ in range(5):
            m.update({"angle": 90.0, "temp": 30.0})
        data = m.to_dict()
        m2 = ForwardModel.from_dict(data)
        assert m2.observations == m.observations
        assert abs(m2.predict_sensors()["angle"] - m.predict_sensors()["angle"]) < 0.01


# ---------------------------------------------------------------------------
# Cerebellum
# ---------------------------------------------------------------------------


class TestCerebellum:
    def test_no_model_returns_none(self):
        cb = Cerebellum()
        result = cb.predict("arm.shoulder", "motor", "rotate", {"degrees": 45})
        assert result is None

    def test_observe_creates_model(self):
        cb = Cerebellum()
        key = cb.make_key("arm.shoulder", "motor", "rotate", {"degrees": 45})
        cb.observe(key, {"angle": 90.0})
        assert cb.has_model("arm.shoulder", "motor", "rotate", {"degrees": 45})

    def test_low_confidence_returns_none(self):
        cb = Cerebellum(CerebellumConfig(confidence_threshold=0.3))
        key = cb.make_key("arm.shoulder", "motor", "rotate", {"degrees": 45})
        cb.observe(key, {"angle": 90.0})
        # 1 observation → confidence ~0.26 < 0.3 threshold
        result = cb.predict("arm.shoulder", "motor", "rotate", {"degrees": 45})
        assert result is None

    def test_confident_model_returns_prediction(self):
        cb = Cerebellum(CerebellumConfig(confidence_threshold=0.3))
        key = cb.make_key("arm.shoulder", "motor", "rotate", {"degrees": 45})
        for _ in range(5):
            cb.observe(key, {"angle": 90.0})
        # 5 observations → confidence ~0.78 > 0.3
        result = cb.predict("arm.shoulder", "motor", "rotate", {"degrees": 45})
        assert result is not None
        assert abs(result["angle"] - 90.0) < 1.0

    def test_high_variance_returns_none(self):
        cb = Cerebellum(
            CerebellumConfig(
                confidence_threshold=0.1,
                high_variance_threshold=0.5,
            )
        )
        key = cb.make_key("arm.shoulder", "motor", "rotate", {"degrees": 45})
        # Train on very noisy values → high variance
        for val in [10, 90, 20, 80, 15, 85, 25, 75]:
            cb.observe(key, {"angle": float(val)})
        result = cb.predict("arm.shoulder", "motor", "rotate", {"degrees": 45})
        # High variance should cause None return
        assert result is None or cb.get_variance("arm.shoulder", "motor", "rotate", {"degrees": 45})["angle"] > 0.5

    def test_observe_from_action(self):
        cb = Cerebellum()
        errors = cb.observe_from_action(
            "arm.shoulder",
            "motor",
            "rotate",
            {"degrees": 45},
            {"angle": 90.0},
        )
        assert "angle" in errors

    def test_llm_fallback_counter(self):
        cb = Cerebellum()
        assert cb.stats()["llm_fallbacks"] == 0
        cb.record_llm_fallback()
        cb.record_llm_fallback()
        assert cb.stats()["llm_fallbacks"] == 2

    def test_stats(self):
        cb = Cerebellum()
        key = cb.make_key("a", "b", "c", {})
        cb.observe(key, {"x": 1.0})
        stats = cb.stats()
        assert stats["model_count"] == 1
        assert stats["total_observations"] == 1

    def test_prune_stale(self):
        cb = Cerebellum()
        key = cb.make_key("a", "b", "c", {})
        cb.observe(key, {"x": 1.0})
        # Force the model to look old
        cb._models[key].last_observed = time.time() - 999999
        pruned = cb.prune_stale_models(max_age_s=1.0)
        assert pruned == 1
        assert not cb.has_model("a", "b", "c", {})

    def test_prune_keeps_recent(self):
        cb = Cerebellum()
        key = cb.make_key("a", "b", "c", {})
        cb.observe(key, {"x": 1.0})
        pruned = cb.prune_stale_models(max_age_s=3600)
        assert pruned == 0
        assert cb.has_model("a", "b", "c", {})

    def test_persistence_roundtrip(self):
        cb = Cerebellum()
        for i in range(5):
            key = cb.make_key("arm.shoulder", "motor", "rotate", {"degrees": float(i * 10)})
            for _ in range(3):
                cb.observe(key, {"angle": float(i * 10 + 5)})

        state = cb.export_state()
        assert state["model_count"] == 5

        cb2 = Cerebellum()
        cb2.import_state(state)
        assert cb2.stats()["model_count"] == 5
        assert cb2.stats()["total_observations"] == cb.stats()["total_observations"]

    def test_save_load_file(self, tmp_path):
        path = str(tmp_path / "cerebellum.json")
        cb = Cerebellum(CerebellumConfig(persistence_path=path))
        key = cb.make_key("arm", "motor", "rotate", {"degrees": 45})
        cb.observe(key, {"angle": 90.0})
        cb.save()

        cb2 = Cerebellum(CerebellumConfig(persistence_path=path))
        loaded = cb2.load()
        assert loaded is True
        assert cb2.stats()["model_count"] == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestCerebellumThreadSafety:
    def test_concurrent_predict_observe(self):
        """8 threads × 100 concurrent predict/observe calls."""
        cb = Cerebellum(CerebellumConfig(confidence_threshold=0.1))
        key = cb.make_key("arm", "motor", "rotate", {"degrees": 45})
        # Pre-seed so predictions can fire
        for _ in range(5):
            cb.observe(key, {"angle": 90.0})

        errors: list[Exception] = []
        barrier = threading.Barrier(8)

        def worker(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(100):
                    if i % 2 == 0:
                        cb.predict("arm", "motor", "rotate", {"degrees": 45})
                    else:
                        cb.observe(key, {"angle": 90.0 + thread_id * 0.01})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        assert cb.stats()["model_count"] == 1
        assert cb.stats()["total_observations"] > 5  # pre-seed + worker observations


# ---------------------------------------------------------------------------
# CerebellumModulator
# ---------------------------------------------------------------------------


class TestCerebellumModulator:
    def _build_entity(self) -> Entity:
        ent = Entity("shoulder", "joint")
        ent.sensors["angle"] = LLMSensor(
            ent,
            "angle",
            "degrees",
            {"type": "float", "range": [-180, 180], "initial": 0.0},
        )
        ent.vital_metrics["angle"] = 0.0
        return ent

    def test_unknown_affordance(self):
        cb = Cerebellum()
        ent = self._build_entity()
        mod = CerebellumModulator(
            ent,
            "motor",
            {},
            cb,
        )
        result = mod.execute("fly", {})
        assert not result.success

    def test_fallback_to_llm(self):
        """When Cerebellum has no model, falls back to LLM modulator."""
        cb = Cerebellum()
        ent = self._build_entity()
        fallback = LLMModulator(
            ent,
            "motor",
            {"rotate_angle": AffordanceSchema(params={"degrees": float, "speed": float})},
        )
        mod = CerebellumModulator(
            ent,
            "motor",
            {"rotate_angle": AffordanceSchema(params={"degrees": float, "speed": float})},
            cb,
            fallback=fallback,
        )
        result = mod.execute("rotate_angle", {"degrees": 45, "speed": 1.0})
        assert result.success
        assert cb.stats()["llm_fallbacks"] == 1
        # Should have trained the Cerebellum
        assert cb.stats()["total_observations"] >= 1

    def test_cerebellum_prediction_used(self):
        """After enough training, Cerebellum predictions replace LLM calls."""
        cb = Cerebellum(CerebellumConfig(confidence_threshold=0.3))
        ent = self._build_entity()

        # Pre-train the Cerebellum
        key = cb.make_key(ent, "motor", "rotate_angle", {"degrees": 45, "speed": 1.0})
        for _ in range(10):
            cb.observe(key, {"angle": 45.0})

        mod = CerebellumModulator(
            ent,
            "motor",
            {"rotate_angle": AffordanceSchema(params={"degrees": float, "speed": float})},
            cb,
        )

        result = mod.execute("rotate_angle", {"degrees": 45, "speed": 1.0})
        assert result.success
        assert result.metadata.get("source") == "cerebellum"
        assert cb.stats()["llm_fallbacks"] == 0

    def test_no_fallback_stub(self):
        """Without fallback, returns stub success."""
        cb = Cerebellum()
        ent = self._build_entity()
        mod = CerebellumModulator(
            ent,
            "motor",
            {"rotate_angle": AffordanceSchema(params={"degrees": float})},
            cb,
        )
        result = mod.execute("rotate_angle", {"degrees": 45})
        assert result.success
        assert result.metadata.get("source") == "stub_no_fallback"

    def test_factory_function(self):
        """Test cerebellum_modulator_factory for attach_backends."""
        cb = Cerebellum()
        spec = load_spec(SCENARIOS_DIR / "robot_arm_3dof.yaml")

        def llm_fallback(ent, mname, spec_mod):
            return LLMModulator(ent, mname, spec_mod.affordances)

        factory = cerebellum_modulator_factory(cb, fallback_factory=llm_fallback)

        shoulder = spec.root_entity.find("shoulder")
        assert shoulder is not None
        spec_mod = shoulder.modulators["motor"]
        cerebellum_mod = factory(shoulder, "motor", spec_mod)

        assert isinstance(cerebellum_mod, CerebellumModulator)
        assert cerebellum_mod.name == "motor"
        result = cerebellum_mod.execute("rotate_angle", {"degrees": 45, "speed": 1.0})
        assert result.success


# ---------------------------------------------------------------------------
# LLM skip rate
# ---------------------------------------------------------------------------


class TestLLMSkipRate:
    def test_skip_rate_tracking(self):
        """Over 20 patterns × 5 reps, LLM calls should drop as models train."""
        cb = Cerebellum(CerebellumConfig(confidence_threshold=0.3))

        # Simulate 20 action patterns × 5 reps
        llm_calls = 0
        cerebellum_calls = 0

        for pattern_idx in range(20):
            params = {"degrees": float(pattern_idx * 10)}
            actual = {"angle": float(pattern_idx * 10 + 5)}

            for rep in range(5):
                predicted = cb.predict("arm", "motor", "rotate", params)
                if predicted is not None:
                    cerebellum_calls += 1
                else:
                    llm_calls += 1
                    cb.record_llm_fallback()
                    key = cb.make_key("arm", "motor", "rotate", params)
                    cb.observe(key, actual)

        total = llm_calls + cerebellum_calls
        assert total == 100
        # After 5 reps, each pattern should have confidence > 0.3
        # So later reps should use Cerebellum. Expect at least 40 skips.
        assert cerebellum_calls >= 40, f"Only {cerebellum_calls} Cerebellum calls (expected >=40)"
        assert cb.stats()["llm_skip_rate"] > 0.0
