"""The B1 floor policies (scripts/survival_world/floor_policies.py) against the prereg's frozen rules
(docs/experiments/b1_exp60_floor_prereg.md §Frozen implementation rules)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world import floor_policies as FP  # noqa: E402

TOOLS = ("mc_eat", "mc_escape_water", "mc_flee", "mc_move_to", "read_mc_oxygen")
WATER = {"is_in_water": 1.0, "oxygen": 20.0}
SHORE = {"is_in_water": 0.0, "oxygen": 20.0}


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _q(**kw):
    clock = Clock()
    pains: list[float] = []
    executed = kw.pop("executed", lambda action, since: True)
    policy = FP.TabularQPolicy(
        seed=kw.pop("seed", 1),
        was_executed=executed,
        pain_since=lambda: sum(pains.pop() for _ in range(len(pains))),
        clock=clock,
        **kw,
    )
    return policy, clock, pains


def _select(policy, obs=WATER, clusters=None):
    return policy.select(agent_id="a", available_tools=TOOLS, clusters=clusters or {"world": "w1"}, observation=obs)


# ── pure pieces ─────────────────────────────────────────────────────────────────────────────────────


def test_the_oxygen_deficit_is_zero_out_of_water_and_clipped_in_it():
    assert FP.oxygen_deficit(SHORE) == 0.0
    assert FP.oxygen_deficit({"is_in_water": 1.0, "oxygen": 20}) == 0.0
    assert FP.oxygen_deficit({"is_in_water": 1.0, "oxygen": 10}) == 0.5
    assert FP.oxygen_deficit({"is_in_water": 1.0, "oxygen": -5}) == 1.0


def test_the_interval_cost_integrates_time_so_a_blocking_action_pays_for_it():
    assert FP.interval_cost(0.5, 0.5, 0.5) == pytest.approx(0.5)
    assert FP.interval_cost(0.5, 0.5, 2.0) == pytest.approx(2.0)  # four ticks' worth
    assert FP.interval_cost(0.0, 1.0, 0.5) == pytest.approx(0.5)  # trapezoid


def test_raw_keys_bin_oxygen_at_the_frozen_edges():
    keys = [FP.raw_key({"is_in_water": 1.0, "oxygen": o}) for o in (0, 4.9, 5, 9.9, 10, 15, 40)]
    assert [k[2] for k in keys] == [0, 0, 1, 1, 2, 3, 3]
    assert FP.raw_key(SHORE)[1] is False


def test_cluster_keys_are_the_sorted_tuple_of_all_active_clusters():
    assert FP.cluster_key({"world": "w", "interoception": "i"}) == FP.cluster_key({"interoception": "i", "world": "w"})
    assert FP.cluster_key(None) == ("clusters", ())


def test_the_frozen_learner_is_undiscounted_with_a_pessimistic_start():
    assert FP.GAMMA == 1.0 and FP.Q_INIT == -2.0
    policy, _, _ = _q()
    _select(policy)
    assert set(policy.q[FP.cluster_key({"world": "w1"})].values()) == {FP.Q_INIT}


def test_ties_are_broken_by_the_seeded_rng_not_by_name():
    picks = {FP.argmax_ties_random({"b": 0.0, "a": 0.0, "c": 0.0}, random.Random(s)) for s in range(40)}
    assert picks == {"a", "b", "c"}  # name order would always give "a"
    assert FP.argmax_ties_random({"b": 0.0, "a": -1.0}, random.Random(0)) == "b"
    same = [FP.argmax_ties_random({"x": 0.0, "y": 0.0}, random.Random(7)) for _ in range(3)]
    assert len(set(same)) == 1  # reproducible from the seed


# ── the learner ─────────────────────────────────────────────────────────────────────────────────────


def test_params_are_always_empty_and_the_action_set_is_stamped():
    policy, _, _ = _q()
    pick = _select(policy)
    assert pick["params"] == {} and pick["tool_name"] in TOOLS
    assert policy.actions == tuple(sorted(TOOLS))


def test_the_prior_set_is_the_two_threat_tools():
    policy, _, _ = _q(prior_only=True)
    _select(policy)
    assert policy.actions == ("mc_escape_water", "mc_flee") and policy.violations == []


def test_a_prior_set_that_is_not_exactly_two_is_a_violation():
    policy, _, _ = _q(prior_only=True)
    policy.select(agent_id="a", available_tools=("mc_flee",), clusters=None, observation=WATER)
    assert policy.violations


def test_the_update_credits_the_previous_executed_action_with_the_interval_cost():
    policy, clock, pains = _q(gamma=0.0)
    policy.set_probe_mode(True)  # greedy, so the first pick is a seeded tie-break among zeros
    first = _select(policy, obs={"is_in_water": 1.0, "oxygen": 10})["tool_name"]  # deficit 0.5
    clock.t += 1.0
    pains.append(0.25)
    _select(policy, obs={"is_in_water": 1.0, "oxygen": 10})
    (u,) = policy.updates
    assert u["action"] == first and not u["terminal"]
    # reward = -(0.5 deficit * 1.0 s / 0.5) - 0.25 pain = -1.25; alpha 0.5 from Q_INIT (-2) -> -1.625
    assert u["reward"] == pytest.approx(-1.25) and u["q_after"] == pytest.approx(-1.625)


def test_the_first_select_in_air_ends_the_episode_and_a_shore_pick_is_never_credited():
    policy, clock, _ = _q()
    first = _select(policy, obs={"is_in_water": 1.0, "oxygen": 18})["tool_name"]
    clock.t += 1.0
    _select(policy, obs=SHORE)  # the head cleared the water before the harness read it
    (u,) = policy.updates
    assert u["action"] == first and u["terminal"] and u["reward"] == pytest.approx(-0.1)  # trapezoid 0.1→0 over 1 s
    clock.t += 0.5
    policy.end_episode("surfaced")  # the shore pick made after the surfacing is truncated, not credited
    assert len(policy.updates) == 1
    policy.clear_pending()
    _select(policy, obs=SHORE)  # armed on the shore just before the teleport
    clock.t += 0.5
    _select(policy, obs=WATER)
    assert len(policy.updates) == 1 and policy.drops[-1]["why"] == "made_out_of_water"


def test_a_pick_that_never_executed_is_not_credited():
    policy, clock, _ = _q(executed=lambda action, since: False)
    _select(policy)
    clock.t += 0.5
    _select(policy)
    assert policy.updates == [] and len(policy.drops) == 1


def test_surfacing_is_terminal_zero_and_pain_is_terminal_minus_one():
    for reason, extra in (("surfaced", 0.0), ("pain", FP.PAIN_TERMINAL)):
        policy, clock, _ = _q()
        _select(policy, obs={"is_in_water": 1.0, "oxygen": 20})  # deficit 0
        clock.t += 0.5
        policy.end_episode(reason)
        (u,) = policy.updates
        assert u["terminal"] and u["reward"] == pytest.approx(extra)


@pytest.mark.parametrize("reason", ["cap", "dirty", "never_submerged"])
def test_a_truncated_episode_books_nothing_and_clears_the_pending_pick(reason):
    policy, clock, _ = _q()
    _select(policy)
    policy.end_episode(reason)
    clock.t += 0.5
    _select(policy)  # nothing pending: no bootstrap across the rescue
    assert policy.updates == []


def test_arming_clears_the_pending_pick_and_discards_pain_published_while_disarmed():
    policy, clock, pains = _q()
    armed = FP.ArmedPolicy(policy)
    armed.arm()
    armed.select(agent_id="a", available_tools=TOOLS, clusters={"world": "w"}, observation=WATER)
    armed.disarm()
    pains.append(1.0)  # published on the shore, between windows
    armed.arm()
    clock.t += 0.5
    armed.select(agent_id="a", available_tools=TOOLS, clusters={"world": "w"}, observation=WATER)
    assert policy.updates == [] and pains == []


def test_probe_mode_is_greedy_and_training_explores():
    policy, _, _ = _q(seed=3)
    policy.set_probe_mode(True)
    for _ in range(20):
        _select(policy)
    assert not any(p["explored"] for p in policy.picks)
    policy.set_probe_mode(False)
    for _ in range(200):
        _select(policy)
    assert any(p["explored"] for p in policy.picks)


def test_q_learns_that_escaping_ends_the_cost():
    """End to end on a toy episode: staying in water costs, escape_water ends the episode with terminal 0, so after a
    few episodes the greedy pick in water is escape_water."""
    policy, clock, _ = _q(seed=5)
    for _ in range(30):
        policy.clear_pending()
        for _tick in range(6):
            pick = _select(policy, obs={"is_in_water": 1.0, "oxygen": 12})["tool_name"]
            clock.t += 0.5
            if pick.endswith("escape_water"):
                policy.end_episode("surfaced")
                break
        else:
            policy.end_episode("pain")
    policy.set_probe_mode(True)
    assert _select(policy, obs={"is_in_water": 1.0, "oxygen": 12})["tool_name"] == "mc_escape_water"


# ── the arming wrapper ──────────────────────────────────────────────────────────────────────────────


class _Raises:
    def select(self, **kw):
        raise RuntimeError("boom")


class _Params:
    def select(self, **kw):
        return {"tool_name": "mc_move_to", "params": {"x": 1}}


def test_a_disarmed_policy_is_idle():
    armed = FP.ArmedPolicy(FP.RandomPolicy(1))
    assert armed.select(agent_id="a", available_tools=TOOLS, clusters=None, observation=SHORE) is None
    armed.arm()
    assert armed.select(agent_id="a", available_tools=TOOLS, clusters=None, observation=SHORE) is not None


@pytest.mark.parametrize("inner", [_Raises(), _Params()])
def test_a_raise_or_params_is_recorded_and_dropped_never_raised(inner):
    armed = FP.ArmedPolicy(inner)
    armed.arm()
    assert armed.select(agent_id="a", available_tools=TOOLS, clusters=None, observation=WATER) is None
    assert armed.all_violations()


def test_a_water_only_arming_idles_on_the_shore():
    armed = FP.ArmedPolicy(FP.RandomPolicy(1))
    armed.arm("pre-placement-0", water_only=True)
    assert armed.select(agent_id="a", available_tools=TOOLS, clusters=None, observation=SHORE) is None
    assert (armed.idle_dry_before, armed.idle_dry_after) == (1, 0)
    assert armed.select(agent_id="a", available_tools=TOOLS, clusters=None, observation=WATER) is not None
    assert armed.select(agent_id="a", available_tools=TOOLS, clusters=None, observation=SHORE) is None
    assert (armed.idle_dry_before, armed.idle_dry_after) == (1, 1)
    armed.arm("pre-roam")  # the roam acts on the shore
    assert armed.select(agent_id="a", available_tools=TOOLS, clusters=None, observation=SHORE) is not None


def test_random_is_uniform_and_reproducible():
    a, b = FP.RandomPolicy(9), FP.RandomPolicy(9)
    picks_a = [
        a.select(agent_id="x", available_tools=TOOLS, clusters=None, observation=SHORE)["tool_name"] for _ in range(50)
    ]
    picks_b = [
        b.select(agent_id="x", available_tools=TOOLS, clusters=None, observation=SHORE)["tool_name"] for _ in range(50)
    ]
    assert picks_a == picks_b and set(picks_a) == set(TOOLS)
