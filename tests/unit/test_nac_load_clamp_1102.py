"""#1102: ``NAc.load_state`` applies each live writer's range to every bias map it restores.

Every writer of the signed/unsigned bias maps clamps (``credit_node`` to ``[0, max_reward_bias]`` and drops a
value that reaches 0; ``credit_goal`` to ``±max_reward_bias``; ``update_cluster_reward`` to
``±max_cluster_reward_bias``; ``record_percept_valence`` to ``±max_percept_valence``; ``record_cluster_fear`` to
``[-max_cluster_fear, 0]``). Before #1102 the load path restored the first four with ``float(bias)`` only, so a
hand-edited, corrupt or pre-clamp ``nac.json`` loaded negative ``reward_bias`` (an invariant the #910 annotator
removals rely on) and above-cap values everywhere else. Fear was already re-clamped on load; it is pinned here so
the shared helper cannot regress it. "Zero is absent" holds only for ``reward_bias``, as in the writers: a
non-positive ``reward_bias`` is dropped, while a loaded positive fear clamps to a KEPT 0.0 key.

Caps are set to NON-default values throughout, so a test passes only when the load reads the live config, never a
hard-coded constant.
"""

from __future__ import annotations

import json
import logging

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.hivemind.ingest import FOREIGN_FEAR_DISCOUNT
from maxim.hivemind.merge import nac_merge

CAPS = {
    "max_reward_bias": 0.3,
    "max_cluster_reward_bias": 0.7,
    "max_percept_valence": 0.6,
    "max_cluster_fear": 0.8,
}
US = "\x1f"


def _nac() -> NAc:
    return NAc(NACConfig(**CAPS))


def _loaded(**maps) -> NAc:
    nac = _nac()
    nac.load_state(maps)
    return nac


@pytest.mark.single_agent_only
def test_reward_bias_clamps_to_zero_max_and_drops_non_positive():
    nac = _loaded(reward_bias={"a:neg": -0.5, "a:zero": 0.0, "a:big": 9.0, "a:ok": 0.1})
    assert nac.reward_bias("a", "big") == pytest.approx(0.3)
    assert nac.reward_bias("a", "ok") == pytest.approx(0.1)
    # zero is absent: a negative clamps to 0 and is dropped, as credit_node drops it
    assert ("a", "neg") not in nac._reward_bias
    assert ("a", "zero") not in nac._reward_bias


@pytest.mark.single_agent_only
def test_goal_reward_bias_clamps_to_signed_max():
    nac = _loaded(goal_reward_bias={"lo": -5.0, "hi": 5.0, "ok": -0.2})
    assert nac.get_goal_reward_bias("lo") == pytest.approx(-0.3)
    assert nac.get_goal_reward_bias("hi") == pytest.approx(0.3)
    assert nac.get_goal_reward_bias("ok") == pytest.approx(-0.2)


@pytest.mark.single_agent_only
def test_cluster_reward_bias_clamps_to_signed_cluster_cap():
    nac = _loaded(cluster_reward_bias={f"a{US}c{US}tool:lo": -3.0, f"a{US}c{US}tool:hi": 3.0})
    assert nac.cluster_reward_bias("a", "c", "tool:lo") == pytest.approx(-0.7)
    assert nac.cluster_reward_bias("a", "c", "tool:hi") == pytest.approx(0.7)


@pytest.mark.single_agent_only
def test_percept_valences_clamp_to_signed_percept_cap():
    nac = _loaded(percept_valences={f"a{US}fire{US}burn": -2.0, f"a{US}food{US}eat": 2.0})
    assert nac.get_percept_valence("fire", "burn", agent_id="a") == pytest.approx(-0.6)
    assert nac.get_percept_valence("food", "eat", agent_id="a") == pytest.approx(0.6)


@pytest.mark.single_agent_only
def test_cluster_fear_still_clamps_to_fear_range():
    nac = _loaded(cluster_fear={f"a{US}c{US}drown": -4.0, f"a{US}d{US}drown": 0.5})
    assert nac.cluster_fear("a", "c") == pytest.approx(-0.8)
    assert nac.cluster_fear("a", "d") == 0.0


@pytest.mark.single_agent_only
def test_load_bounds_equal_what_the_live_writers_saturate_at():
    """Parity: a value past the cap loads to exactly where repeated live credit saturates."""
    live = _nac()
    for _ in range(500):
        live.credit_node("a", "n", 1.0)
        live.credit_goal("g", 1.0)
        live.credit_goal("h", -1.0)
        live.update_cluster_reward("a", "c", "tool:x", 1.0)
        live.update_cluster_reward("a", "c", "tool:y", -1.0)
        live.record_percept_valence("e", "m", -1.0, agent_id="a")
        live.record_cluster_fear("a", "c", "drive:health", 1.0)
    loaded = _loaded(
        reward_bias={"a:n": 99.0},
        goal_reward_bias={"g": 99.0, "h": -99.0},
        cluster_reward_bias={f"a{US}c{US}tool:x": 99.0, f"a{US}c{US}tool:y": -99.0},
        percept_valences={f"a{US}e{US}m": -99.0},
        cluster_fear={f"a{US}c{US}drive:health": -99.0},
    )
    assert loaded.reward_bias("a", "n") == live.reward_bias("a", "n")
    assert loaded.get_goal_reward_bias("g") == live.get_goal_reward_bias("g")
    assert loaded.get_goal_reward_bias("h") == live.get_goal_reward_bias("h")
    assert loaded.cluster_reward_bias("a", "c", "tool:x") == live.cluster_reward_bias("a", "c", "tool:x")
    assert loaded.cluster_reward_bias("a", "c", "tool:y") == live.cluster_reward_bias("a", "c", "tool:y")
    assert loaded.get_percept_valence("e", "m", agent_id="a") == live.get_percept_valence("e", "m", agent_id="a")
    assert loaded.cluster_fear("a", "c") == live.cluster_fear("a", "c")


@pytest.mark.single_agent_only
def test_clamped_values_are_counted_in_one_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="maxim.decisions.nac"):
        _loaded(reward_bias={"a:neg": -1.0}, goal_reward_bias={"g": 2.0}, cluster_reward_bias={f"a{US}c{US}t": 2.0})
    lines = [r.getMessage() for r in caplog.records if "out of range" in r.getMessage()]
    assert len(lines) == 1
    assert "'reward_bias': 1" in lines[0] and "'goal_reward_bias': 1" in lines[0]
    assert "'cluster_reward_bias': 1" in lines[0]


@pytest.mark.single_agent_only
def test_in_range_state_round_trips_unchanged_and_silently(caplog):
    src = _nac()
    src.credit_node("a", "n", 1.0)
    src.credit_goal("g", -1.0)
    src.update_cluster_reward("a", "c", "tool:x", -1.0)
    src.record_percept_valence("e", "m", -1.0, agent_id="a")
    src.record_cluster_fear("a", "c", "drive:health", 1.0)
    state = src.dump()
    dst = _nac()
    with caplog.at_level(logging.WARNING, logger="maxim.decisions.nac"):
        dst.load_state(json.loads(json.dumps(state)))
    assert dst.dump() == state
    assert "out of range" not in caplog.text


@pytest.mark.single_agent_only
def test_file_load_applies_the_clamp(tmp_path):
    path = tmp_path / "nac.json"
    path.write_text(json.dumps({"reward_bias": {"a:neg": -0.4, "a:big": 4.0}}))
    nac = NAc(NACConfig(**CAPS, persistence_path=str(path)))
    nac.load()
    assert nac._reward_bias == {("a", "big"): pytest.approx(0.3)}


@pytest.mark.single_agent_only
def test_hivemind_merge_output_loads_with_the_foreign_fear_discount_intact():
    """Ingest discounts foreign fear to ×FOREIGN_FEAR_DISCOUNT before the MIN fold; the load clamp keeps it."""
    foreign = -1.0 * FOREIGN_FEAR_DISCOUNT
    merged = nac_merge(
        {"reward_bias": {"a:z": 0.0}},
        {"cluster_fear": {f"a{US}c{US}drown": foreign}},
        left_source="recv",
        right_source="donor",
    )
    nac = NAc(NACConfig())
    nac.load_state(merged)
    assert nac.cluster_fear("a", "c") == pytest.approx(foreign)
    # a zero reward_bias the merge carried is absent after load, as credit_node would leave it
    assert ("a", "z") not in nac._reward_bias
