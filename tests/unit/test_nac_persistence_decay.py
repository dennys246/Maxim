"""Decay-on-load semantics for NAc cross-session persistence.

Pins the decay decision from docs/plans/archive/nac_cross_session_persistence.md:
NAc decay is tick-anchored in-session, so ``load()`` ages the bias
surfaces by elapsed WALL-CLOCK time using the ``saved_at`` stamp from the
persisted payload. Two half-life schedules:

- ``cluster_bias_wall_decay_half_life_s`` (1 day) — ``cluster_reward_bias``
  (a within-session working signal per its 300-tick tau; persisting it
  unchanged would silently promote it to a cross-session signal).
- ``bias_wall_decay_half_life_s`` (7 days) — ``reward_bias`` /
  ``goal_reward_bias`` / ``percept_valences`` (the designated
  cross-session transfer surfaces).

Causal links + Welford variance are NOT decayed on load; pre-1.3 payloads
(no ``saved_at``) load undecayed; ``load_state`` (the dump-shape entry
used by hivemind merge) never decays.
"""

from __future__ import annotations

import json

import pytest

from maxim.decisions.nac import NAc, NACConfig, Valence

DAY_S = 86400.0
WEEK_S = 604800.0


def _populated_nac(path: str) -> NAc:
    nac = NAc(NACConfig(persistence_path=path))
    nac.observe("tool_call", "tool_a", "result", "success", Valence.POSITIVE, 0.5, context={"agent_id": "a1"})
    with nac._lock:
        nac._reward_bias[("a1", "node_1")] = 0.2
        nac._goal_reward_bias["find warmth"] = 0.4
        nac._percept_valences[("a1", "dragon", "burn")] = -0.5
        nac._cluster_reward_bias[("a1", "cluster_1", "tool:warm")] = 0.8
    return nac


def _rewrite_saved_at(path: str, saved_at) -> None:
    with open(path) as f:
        state = json.load(f)
    if saved_at is None:
        state.pop("saved_at", None)
    else:
        state["saved_at"] = saved_at
    with open(path, "w") as f:
        json.dump(state, f)


class TestSavedAtStamp:
    def test_save_carries_wall_clock_saved_at_but_dump_does_not(self, tmp_path):
        """``saved_at`` is a persistence-envelope concern of the file
        writer — ``dump()`` stays a pure state surface (two dumps of
        identical state compare equal, per test_bio_system_snapshot.py)."""
        import time

        path = str(tmp_path / "nac.json")
        nac = NAc(NACConfig(persistence_path=path))
        assert "saved_at" not in nac.dump()
        before = time.time()
        nac.save()
        with open(path) as f:
            state = json.load(f)
        assert before <= state["saved_at"] <= time.time()

    def test_format_version_is_1_3(self, tmp_path):
        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        with open(path) as f:
            assert json.load(f)["_format_version"] == "1.3"


class TestDecayOnLoad:
    def test_one_day_gap_halves_cluster_bias_and_barely_touches_slow_surfaces(self, tmp_path):
        import time

        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        _rewrite_saved_at(path, time.time() - DAY_S)

        nac2 = NAc(NACConfig(persistence_path=path))
        nac2.load()

        # cluster: half-life 1 day → 0.8 * 0.5 = 0.4
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.4, rel=0.02)
        # slow surfaces: half-life 7 days → factor 0.5^(1/7) ≈ 0.906
        slow = 0.5 ** (1.0 / 7.0)
        assert nac2._reward_bias[("a1", "node_1")] == pytest.approx(0.2 * slow, rel=0.02)
        assert nac2._goal_reward_bias["find warmth"] == pytest.approx(0.4 * slow, rel=0.02)
        assert nac2.get_percept_valence("dragon", "burn", agent_id="a1") == pytest.approx(-0.5 * slow, rel=0.02)

    def test_six_month_gap_prunes_biases_but_keeps_links(self, tmp_path):
        import time

        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        _rewrite_saved_at(path, time.time() - 180 * DAY_S)

        nac2 = NAc(NACConfig(persistence_path=path))
        nac2.load()

        # A 180-day gap: cluster 0.8*0.5^180 → 0; slow 0.5^(180/7) ≈ 1.8e-8 → pruned
        assert nac2._cluster_reward_bias == {}
        assert nac2._reward_bias == {}
        assert nac2._goal_reward_bias == {}
        assert nac2._percept_valences == {}
        # Causal links are accumulated statistics — not activations — and
        # take the flat decay_all(0.95) haircut at session end instead.
        assert len(nac2.get_links_for_event("tool_a")) == 1

    def test_missing_saved_at_loads_undecayed(self, tmp_path):
        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        _rewrite_saved_at(path, None)  # pre-1.3 payload shape

        nac2 = NAc(NACConfig(persistence_path=path))
        nac2.load()
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)
        assert nac2._reward_bias[("a1", "node_1")] == pytest.approx(0.2)

    def test_load_state_never_decays(self, tmp_path):
        """The dump-shape entry point (hivemind nac_merge, snapshot
        restores) must be byte-faithful — decay lives ONLY on the file
        entry point ``load()``."""
        import time

        path = str(tmp_path / "nac.json")
        nac = _populated_nac(path)
        state = nac.dump()
        state["saved_at"] = time.time() - 180 * DAY_S

        nac2 = NAc(NACConfig())
        nac2.load_state(state)
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)
        assert nac2._percept_valences[("a1", "dragon", "burn")] == pytest.approx(-0.5)

    def test_clock_skew_negative_elapsed_is_noop(self, tmp_path):
        import time

        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        _rewrite_saved_at(path, time.time() + 3600.0)  # file "from the future"

        nac2 = NAc(NACConfig(persistence_path=path))
        nac2.load()
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)

    def test_apply_wall_clock_decay_reports_pruned_counts(self):
        nac = NAc(NACConfig())
        with nac._lock:
            nac._cluster_reward_bias[("a1", "c1", "t1")] = 0.8
            nac._reward_bias[("a1", "n1")] = 0.2
        results = nac.apply_wall_clock_decay(180 * DAY_S)
        assert results == {
            "reward_bias_pruned": 1,
            "cluster_reward_bias_pruned": 1,
        }
