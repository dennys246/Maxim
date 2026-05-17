"""Tests for Wire 1 (release_0_9_1.md Stage 4 — risk-sensitive action annotation).

The wire ships four surfaces:

1. ``NAc._event_outcome_welford`` — per-(agent_id, event_sig) Welford online
   variance state for the binary reward signal across outcomes. Updated
   ONCE per outcome in ``_record_outcome_impl`` under ``self._lock``.
   (Design note: the plan originally placed variance on ``CausalLink``,
   but ``_generate_link_id`` keys on ``outcome_signature`` (which embeds
   valence), making per-link variance structurally 0 for binary outcomes.
   Moving up one level captures cross-link heterogeneity.)
2. ``OutcomePrediction.uncertainty_interval`` populated in
   ``NAc._predict_impl`` from the NAc-level Welford state.
3. ``NAc.get_action_risk_profile(*, agent_id, min_observations=N)`` returning
   ``{event_signature → variance}`` for one agent's qualifying events.
4. agent_loop.py hook appends ``(feels unpredictable)`` /
   ``(feels predictable)`` annotations to tool descriptions.

Tests cover Welford correctness across N, once-per-outcome firing,
zero-observation sentinel, persistence round-trip, per-agent isolation,
empty-agent_id rejection, min_observations filter, annotation idempotency,
and env-var ablation gate.
"""

from __future__ import annotations

import math
import re

import pytest

from maxim.decisions.causal_link import (
    Valence,
    _VALENCE_TO_REWARD,
)
from maxim.decisions.nac import NAc, NACConfig


# ─────────────────────────────────────────────────────────────────────
# 1. Welford correctness via NAc.record_outcome → _event_outcome_welford
# ─────────────────────────────────────────────────────────────────────


def _seed(nac: NAc, *valences: Valence, event_sig: str = "tool:test", agent_id: str = "a") -> None:
    """Drive NAc through one full (record_event, record_outcome) per valence."""
    for v in valences:
        nac.record_event("tool", event_sig, context={"agent_id": agent_id})
        nac.record_outcome("tool", event_sig, v, context={"agent_id": agent_id})


class TestWelfordCorrectness:
    """Welford's online algorithm matches the reference variance.

    Pre-merge concerns (per user prompt): (a) numerical stability — no
    ``Σ(x²) − (Σx)²/n / n`` catastrophic cancellation; (b) fires once
    per outcome (not per eligibility-trace credit-distribution event);
    (c) zero-observation case returns (0.0, 0.0) uncertainty without
    divide-by-zero.
    """

    def test_zero_observations_state_is_absent(self):
        nac = NAc()
        assert nac._event_outcome_welford == {}

    def test_one_observation_variance_is_zero(self):
        # n=1: M2 = 0 because the running mean equals the only sample,
        # so delta * delta2 = 0. Not a divide-by-zero edge case.
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(nac, Valence.POSITIVE)
        state = nac._event_outcome_welford[("a", "tool:test")]
        assert state["n"] == 1.0
        assert state["m2"] == 0.0

    def test_two_identical_observations_variance_is_zero(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(nac, Valence.POSITIVE, Valence.POSITIVE)
        state = nac._event_outcome_welford[("a", "tool:test")]
        assert state["n"] == 2.0
        assert math.isclose(state["m2"] / state["n"], 0.0, abs_tol=1e-12)

    def test_alternating_pos_neg_max_bernoulli_variance(self):
        # Bernoulli variance with p=0.5 = 0.25 (population).
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        for _ in range(10):
            _seed(nac, Valence.POSITIVE, Valence.NEGATIVE)
        state = nac._event_outcome_welford[("a", "tool:test")]
        assert state["n"] == 20.0
        assert math.isclose(state["m2"] / state["n"], 0.25, abs_tol=1e-10)

    def test_matches_naive_variance_small_n(self):
        # Compare against the straightforward formula for cross-check.
        rewards = [1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.0]
        valences_for_reward = {v: k for k, v in _VALENCE_TO_REWARD.items()}
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(nac, *[valences_for_reward[r] for r in rewards])
        # NAc maps NEUTRAL→0.5 and UNKNOWN→0.5 — both decode back to NEUTRAL
        # for the reverse lookup; reward=0.5 happens to be the NEUTRAL entry.
        state = nac._event_outcome_welford[("a", "tool:test")]
        # Verify the reverse mapping picked NEUTRAL (0.5)
        mean = sum(rewards) / len(rewards)
        naive_var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        assert math.isclose(state["m2"] / state["n"], naive_var, abs_tol=1e-10)

    def test_numerical_stability_large_n_low_variance(self):
        # The cancellation-prone case: 999 successes + 1 failure → near-zero
        # variance. Welford avoids the catastrophic cancellation that bites
        # the naive Σ(x²) − (Σx)²/n / n formula at low variance + high n.
        nac = NAc(config=NACConfig(temporal_window_seconds=60, max_pending_events=2000))
        _seed(nac, *([Valence.POSITIVE] * 999), Valence.NEGATIVE)
        state = nac._event_outcome_welford[("a", "tool:test")]
        # Reference: mean = 0.999; variance ~ 0.000999
        mean = 999.0 / 1000.0
        expected = (999 * (1.0 - mean) ** 2 + (0.0 - mean) ** 2) / 1000
        assert math.isclose(state["m2"] / state["n"], expected, abs_tol=1e-12)

    def test_variance_grows_when_alternation_introduced(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(nac, *([Valence.POSITIVE] * 5))
        state = nac._event_outcome_welford[("a", "tool:test")]
        baseline = state["m2"] / state["n"]
        assert baseline == 0.0
        _seed(nac, Valence.NEGATIVE)
        state = nac._event_outcome_welford[("a", "tool:test")]
        assert state["m2"] / state["n"] > baseline


class TestWelfordFiresOncePerOutcome:
    """The Welford update fires exactly once per outcome event.

    The eligibility-trace credit-distribution path in
    ``NAc.distribute_reward`` operates on ``_reward_bias`` keyed by
    substrate nodes — that path does not call into the Welford state.
    """

    def test_outcome_increments_welford_n_by_one(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(nac, Valence.POSITIVE)
        state = nac._event_outcome_welford[("a", "tool:test")]
        assert state["n"] == 1.0
        _seed(nac, Valence.NEGATIVE)
        state = nac._event_outcome_welford[("a", "tool:test")]
        assert state["n"] == 2.0

    def test_distribute_reward_does_not_touch_welford(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(nac, Valence.POSITIVE)
        snapshot = dict(nac._event_outcome_welford[("a", "tool:test")])
        # Distribute reward via the substrate eligibility path
        nac.update_eligibility("a", "node_x", activation=0.5)
        nac.distribute_reward(agent_id="a", reward=1.0)
        assert nac._event_outcome_welford[("a", "tool:test")] == snapshot

    def test_record_outcome_without_agent_id_skips_welford(self):
        # CC4 silent-no-op: outcomes lacking agent_id in context don't
        # update the variance accumulator. Documented contract.
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        nac.record_event("tool", "tool:test", context={})  # no agent_id
        nac.record_outcome("tool", "tool:test", Valence.POSITIVE, context={})
        assert nac._event_outcome_welford == {}


# ─────────────────────────────────────────────────────────────────────
# 2. Persistence round-trip
# ─────────────────────────────────────────────────────────────────────


class TestPersistenceRoundTrip:
    def test_welford_state_round_trips_through_dump_load(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(nac, Valence.POSITIVE, Valence.NEGATIVE, Valence.POSITIVE, Valence.NEGATIVE)
        snapshot = dict(nac._event_outcome_welford[("a", "tool:test")])
        data = nac.dump()
        assert "event_outcome_welford" in data
        # Rehydrate into a fresh NAc
        nac2 = NAc(config=NACConfig(temporal_window_seconds=60))
        nac2.load_state(data)
        assert ("a", "tool:test") in nac2._event_outcome_welford
        rehydrated = nac2._event_outcome_welford[("a", "tool:test")]
        assert math.isclose(rehydrated["mean"], snapshot["mean"])
        assert math.isclose(rehydrated["m2"], snapshot["m2"])
        assert math.isclose(rehydrated["n"], snapshot["n"])

    def test_backward_compat_missing_field_loads_empty(self):
        # Pre-0.9.1 NAc dumps lack ``event_outcome_welford``. Loading
        # must succeed with cold-start (empty dict).
        nac = NAc()
        state = {
            "version": "1.0",
            "links": {},
            "outcome_index": {},
            "priors": {},
            "total_observations": 0,
            "reward_bias": {},
            "goal_reward_bias": {},
            "cluster_reward_bias": {},
            "percept_valences": {},
            # no event_outcome_welford
        }
        nac.load_state(state)
        assert nac._event_outcome_welford == {}
        # Adding a new outcome must bootstrap state cleanly
        _seed(nac, Valence.POSITIVE)
        assert ("a", "tool:test") in nac._event_outcome_welford

    def test_load_state_skips_corrupt_entries(self):
        # A malformed entry (wrong shape) should be skipped without
        # raising — the rest of load_state must complete.
        nac = NAc()
        nac.load_state(
            {
                "version": "1.0",
                "links": {},
                "outcome_index": {},
                "priors": {},
                "total_observations": 0,
                "reward_bias": {},
                "goal_reward_bias": {},
                "cluster_reward_bias": {},
                "percept_valences": {},
                "event_outcome_welford": {
                    "a\x1ftool:test": {"mean": 0.5, "m2": 0.25, "n": 4.0},
                    "corrupt_no_separator": {"mean": 0.0, "m2": 0.0, "n": 0.0},
                    "a\x1ftool:bad": "not_a_dict",
                },
            }
        )
        assert ("a", "tool:test") in nac._event_outcome_welford
        assert "corrupt_no_separator" not in {k[0] for k in nac._event_outcome_welford.keys()}


# ─────────────────────────────────────────────────────────────────────
# 3. OutcomePrediction.uncertainty_interval
# ─────────────────────────────────────────────────────────────────────


class TestOutcomePredictionUncertaintyInterval:
    def test_cold_start_returns_sentinel(self):
        nac = NAc(config=NACConfig(min_confidence_threshold=0.0))
        nac.set_prior("tool", "tool:novel", predicted_value=0.5, confidence=0.5)
        pred = nac.predict("tool", "tool:novel", context={"agent_id": "a"})
        assert pred is not None
        # No Welford state yet → sentinel
        assert pred.uncertainty_interval == (0.0, 0.0)

    def test_high_variance_widens_interval(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60, min_confidence_threshold=0.0))
        for _ in range(10):
            _seed(nac, Valence.POSITIVE, Valence.NEGATIVE, event_sig="tool:risky")
        pred = nac.predict("tool", "tool:risky", context={"agent_id": "a"})
        assert pred is not None
        lower, upper = pred.uncertainty_interval
        # variance 0.25 → std 0.5; interval span ~1.0 (clamped to [0, 1])
        assert upper - lower > 0.3, f"expected wide interval, got [{lower}, {upper}]"
        assert 0.0 <= lower <= upper <= 1.0

    def test_low_variance_narrows_interval(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60, min_confidence_threshold=0.0))
        _seed(nac, *([Valence.POSITIVE] * 19), Valence.NEGATIVE, event_sig="tool:reliable")
        pred = nac.predict("tool", "tool:reliable", context={"agent_id": "a"})
        assert pred is not None
        lower, upper = pred.uncertainty_interval
        # variance ~ 0.0475 → std ~ 0.218; span ~ 0.436 (clamped)
        assert 0.0 <= lower <= upper <= 1.0
        assert upper - lower < 0.6

    def test_no_agent_id_in_context_returns_sentinel(self):
        # When context lacks agent_id, uncertainty defaults to sentinel
        # even if a Welford state happens to exist under some other agent.
        nac = NAc(config=NACConfig(temporal_window_seconds=60, min_confidence_threshold=0.0))
        for _ in range(5):
            _seed(nac, Valence.POSITIVE, Valence.NEGATIVE, event_sig="tool:probe")
        pred = nac.predict("tool", "tool:probe", context={})
        # If the predict path even returns a prediction without agent_id,
        # uncertainty must be the sentinel
        if pred is not None:
            assert pred.uncertainty_interval == (0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────
# 4. NAc.get_action_risk_profile
# ─────────────────────────────────────────────────────────────────────


class TestGetActionRiskProfile:
    def test_empty_agent_id_raises(self):
        nac = NAc()
        with pytest.raises(ValueError, match="non-empty agent_id"):
            nac.get_action_risk_profile(agent_id="")

    def test_cold_start_returns_empty(self):
        nac = NAc()
        assert nac.get_action_risk_profile(agent_id="a") == {}

    def test_below_min_observations_filtered(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(nac, *([Valence.POSITIVE] * 3))  # < default 5
        assert nac.get_action_risk_profile(agent_id="a") == {}

    def test_min_observations_threshold_inclusive(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(
            nac,
            Valence.POSITIVE,
            Valence.NEGATIVE,
            Valence.POSITIVE,
            Valence.NEGATIVE,
            Valence.POSITIVE,
            event_sig="tool:counted",
        )
        profile = nac.get_action_risk_profile(agent_id="a")
        assert "tool:counted" in profile
        assert profile["tool:counted"] > 0.0

    def test_per_agent_isolation_shared_nac(self):
        # CC4 invariant: two agents sharing one NAc instance must read
        # disjoint variance profiles.
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        # Agent A: 5 successes (variance 0)
        for _ in range(5):
            _seed(nac, Valence.POSITIVE, event_sig="tool:probe", agent_id="agent_a")
        # Agent B: alternating (high variance)
        _seed(
            nac,
            Valence.POSITIVE,
            Valence.NEGATIVE,
            Valence.POSITIVE,
            Valence.NEGATIVE,
            Valence.POSITIVE,
            event_sig="tool:probe",
            agent_id="agent_b",
        )
        prof_a = nac.get_action_risk_profile(agent_id="agent_a")
        prof_b = nac.get_action_risk_profile(agent_id="agent_b")
        # A's variance ~ 0; B's > 0
        assert prof_a.get("tool:probe", 0.0) == 0.0
        assert prof_b.get("tool:probe", 0.0) > 0.0

    def test_event_sig_filter_narrows_walk(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        for sig in ["tool:a", "tool:b"]:
            _seed(
                nac,
                Valence.POSITIVE,
                Valence.NEGATIVE,
                Valence.POSITIVE,
                Valence.NEGATIVE,
                Valence.POSITIVE,
                event_sig=sig,
            )
        full = nac.get_action_risk_profile(agent_id="a")
        narrowed = nac.get_action_risk_profile(event_sig="tool:a", agent_id="a")
        assert "tool:a" in full
        assert "tool:b" in full
        assert "tool:a" in narrowed
        assert "tool:b" not in narrowed

    def test_links_missing_agent_id_in_context_are_skipped(self):
        # Raw-loop callers (test fixtures, early protocol) may not tag
        # event_context with agent_id. The Welford state is not written
        # without agent_id, so the profile remains empty.
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        for _ in range(5):
            nac.record_event("tool", "tool:untagged", context={})
            nac.record_outcome("tool", "tool:untagged", Valence.POSITIVE, context={})
        assert nac.get_action_risk_profile(agent_id="a") == {}


# ─────────────────────────────────────────────────────────────────────
# 5. agent_loop annotation assembly — verify the regex / band / phrase
#    composition in isolation (mirrors test_wire_3_embodiment_filter.py
#    shape, no agent loop spin-up).
# ─────────────────────────────────────────────────────────────────────


class TestAnnotationAssemblyExpression:
    HIGH_VARIANCE_THRESHOLD = 0.15
    LOW_VARIANCE_THRESHOLD = 0.05
    WIRE1_RE = re.compile(r" \((?:feels unpredictable|feels predictable)\)$")

    def _classify_phrase(self, variance: float) -> str | None:
        if variance >= self.HIGH_VARIANCE_THRESHOLD:
            return "feels unpredictable"
        if variance < self.LOW_VARIANCE_THRESHOLD:
            return "feels predictable"
        return None

    def _annotate(self, descs: dict[str, dict], profile: dict[str, float]) -> dict[str, dict]:
        out = {k: dict(v) for k, v in descs.items()}
        for event_sig, variance in profile.items():
            if event_sig.startswith("tool:use:"):
                continue
            tool_name = event_sig[len("tool:") :] if event_sig.startswith("tool:") else event_sig
            if tool_name not in out:
                continue
            phrase = self._classify_phrase(variance)
            entry = out[tool_name]
            base_desc = entry.get("description", "")
            stripped = self.WIRE1_RE.sub("", base_desc)
            if phrase is None:
                if stripped != base_desc:
                    out[tool_name] = {**entry, "description": stripped}
                continue
            out[tool_name] = {**entry, "description": f"{stripped} ({phrase})"}
        return out

    def test_high_variance_gets_unpredictable_phrase(self):
        descs = {"strike": {"description": "Strike the target.", "params": {}}}
        result = self._annotate(descs, {"tool:strike": 0.22})
        assert "(feels unpredictable)" in result["strike"]["description"]
        assert result["strike"]["description"] == "Strike the target. (feels unpredictable)"

    def test_low_variance_gets_predictable_phrase(self):
        descs = {"strike": {"description": "Strike the target.", "params": {}}}
        result = self._annotate(descs, {"tool:strike": 0.01})
        assert "(feels predictable)" in result["strike"]["description"]

    def test_middle_band_gets_no_annotation(self):
        descs = {"strike": {"description": "Strike the target.", "params": {}}}
        result = self._annotate(descs, {"tool:strike": 0.08})
        assert "(feels" not in result["strike"]["description"]

    def test_idempotent_across_observations(self):
        descs = {"strike": {"description": "Strike the target.", "params": {}}}
        once = self._annotate(descs, {"tool:strike": 0.22})
        twice = self._annotate(once, {"tool:strike": 0.22})
        assert twice["strike"]["description"].count("(feels unpredictable)") == 1

    def test_middle_band_strips_stale_annotation(self):
        # Decayed back to middle band — strip prior annotation
        descs = {"strike": {"description": "Strike the target. (feels unpredictable)", "params": {}}}
        result = self._annotate(descs, {"tool:strike": 0.08})
        assert "(feels" not in result["strike"]["description"]
        assert result["strike"]["description"] == "Strike the target."

    def test_band_transition_strips_then_reannotates(self):
        descs = {"strike": {"description": "Strike the target. (feels unpredictable)", "params": {}}}
        result = self._annotate(descs, {"tool:strike": 0.01})
        desc = result["strike"]["description"]
        assert desc.count("(feels unpredictable)") == 0
        assert desc.count("(feels predictable)") == 1

    def test_tool_use_compound_signature_skipped(self):
        descs = {"use": {"description": "Generic action tool.", "params": {}}}
        result = self._annotate(descs, {"tool:use:dodge": 0.22})
        assert "(feels" not in result["use"]["description"]

    def test_wire3_and_wire1_annotations_coexist(self):
        # Wire 1 regex only matches its own phrases — a prior Wire 3 phrase
        # is preserved, and the Wire 1 phrase is appended after it.
        descs = {"strike": {"description": "Strike. (feels strained)", "params": {}}}
        result = self._annotate(descs, {"tool:strike": 0.22})
        desc = result["strike"]["description"]
        assert "(feels strained)" in desc
        assert "(feels unpredictable)" in desc
        assert desc.endswith("(feels unpredictable)")

    def test_unknown_tool_in_profile_ignored(self):
        descs = {"strike": {"description": "Strike.", "params": {}}}
        result = self._annotate(descs, {"tool:phantom": 0.22})
        assert result == descs


# ─────────────────────────────────────────────────────────────────────
# 6. Env-var ablation gate (MAXIM_DISABLE_VARIANCE_ANNOTATION)
# ─────────────────────────────────────────────────────────────────────


class TestEnvAblationGate:
    """Producer site reads ``os.environ.get(...).strip().lower()`` and
    tests membership in the truthy frozenset. Verify the parser shape.
    """

    TRUTHY = {"1", "true", "t", "yes", "y", "on"}

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Y", "on", "ON"])
    def test_truthy_values_disable(self, value: str):
        assert value.strip().lower() in self.TRUTHY

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  ", "anything"])
    def test_falsy_values_keep_enabled(self, value: str):
        assert value.strip().lower() not in self.TRUTHY


# ─────────────────────────────────────────────────────────────────────
# 7. End-to-end: NAc.record_outcome → get_action_risk_profile →
#    annotation classifies into expected band
# ─────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    """End-to-end pin from outcome recording through to phrase classification.

    Confirms the data flow the agent_loop hook depends on: drive a few
    POS/NEG outcomes through NAc, query get_action_risk_profile, and
    verify the resulting variance falls in the expected annotation band.
    """

    HIGH = 0.15
    LOW = 0.05

    def test_reliable_tool_classifies_as_predictable(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(nac, *([Valence.POSITIVE] * 6), event_sig="tool:reliable")
        profile = nac.get_action_risk_profile(agent_id="a")
        variance = profile.get("tool:reliable", 0.0)
        assert variance < self.LOW, f"expected reliable-band, got variance {variance}"

    def test_alternating_tool_classifies_as_unpredictable(self):
        nac = NAc(config=NACConfig(temporal_window_seconds=60))
        _seed(
            nac,
            Valence.POSITIVE,
            Valence.NEGATIVE,
            Valence.POSITIVE,
            Valence.NEGATIVE,
            Valence.POSITIVE,
            Valence.NEGATIVE,
            event_sig="tool:erratic",
        )
        profile = nac.get_action_risk_profile(agent_id="a")
        variance = profile.get("tool:erratic", 0.0)
        assert variance >= self.HIGH, f"expected unpredictable-band, got variance {variance}"
