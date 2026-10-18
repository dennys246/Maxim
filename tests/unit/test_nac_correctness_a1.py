"""NAc correctness fixes (PR A1).

1. Non-finite credit inputs raise at every NAc credit/update entry point, and a
   non-finite STORED value is dropped (with one warning) on load — the clamps
   turned NaN into the cap (``max(-cap, min(nan, cap)) == -cap``) and a stored
   NaN was never pruned (``nan < 0.001`` is False).
2. The session observer (``maxim.session._build_session_observer``) loads NAc
   without decay-on-load, like the other read-only observers.
3. ``TemporalCreditDistributor`` reads eligibility through NAc's locked
   ``eligibility_snapshot`` (never ``nac._eligibility``), and ``decay_all``
   takes NAc's lock.
4. Agent-loop section 8.5 runs every per-tick decay independently.
5. ``reward_bias`` keys (``"<agent_id>:<node_id>"``, split on the first ``:``)
   cannot be written with an agent id holding ``:``; node ids with ``:``
   round-trip, legacy files included.
6. ``get_version`` reports the format ``save()`` stamps.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time

import pytest

from maxim.decisions.nac import _NAC_FORMAT_VERSION, NAc, NACConfig, Valence
from maxim.decisions.temporal_credit import TemporalCreditDistributor
from maxim.time.scn import SCN

NON_FINITE = [math.nan, math.inf, -math.inf]


# ── 1. non-finite inputs ─────────────────────────────────────────────────────


def _cfg() -> NACConfig:
    return NACConfig(cluster_fear_failure_modes=frozenset({"burn"}))


ENTRY_POINTS = {
    "credit_node": lambda nac, v: nac.credit_node("a1", "n1", v),
    "update_cluster_reward": lambda nac, v: nac.update_cluster_reward("a1", "c1", "tool:x", v),
    "credit_operant_reward": lambda nac, v: nac.credit_operant_reward("a1", v),
    "credit_goal": lambda nac, v: nac.credit_goal("g", v),
    "record_percept_valence": lambda nac, v: nac.record_percept_valence("dragon", "burn", v, agent_id="a1"),
    "record_cluster_fear": lambda nac, v: nac.record_cluster_fear("a1", "c1", "burn", v),
    "update_eligibility": lambda nac, v: nac.update_eligibility("a1", "n1", v),
    "distribute_reward": lambda nac, v: nac.distribute_reward("a1", v),
}


@pytest.mark.parametrize("value", NON_FINITE, ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
def test_non_finite_input_raises_and_stores_nothing(entry, value):
    nac = NAc(_cfg())
    nac.set_pending_operant_action("a1", "c1", "tool:x")
    nac.update_eligibility("a1", "n1", 1.0)
    before = nac.dump()
    with pytest.raises(ValueError, match="finite"):
        ENTRY_POINTS[entry](nac, value)
    assert nac.dump() == before
    assert nac._eligibility == {("a1", "n1"): 1.0}


def test_finite_inputs_still_accepted():
    nac = NAc(_cfg())
    nac.set_pending_operant_action("a1", "c1", "tool:x")
    for call in ENTRY_POINTS.values():
        call(nac, 0.5)
    assert nac.cluster_reward_bias("a1", "c1", "tool:x") > 0.0


@pytest.mark.parametrize("value", NON_FINITE, ids=["nan", "inf", "-inf"])
def test_distributor_rejects_non_finite_reward_even_with_nothing_eligible(value):
    dist = TemporalCreditDistributor(nac=NAc(), scn=SCN())
    with pytest.raises(ValueError, match="finite"):
        dist.distribute("a1", value)


def _write_nac(path, state: dict) -> None:
    state = {"_format_version": _NAC_FORMAT_VERSION, **state}
    path.write_text(json.dumps(state))  # json.dumps writes NaN/Infinity literals


@pytest.mark.single_agent_only
def test_load_drops_non_finite_stored_values(tmp_path, caplog):
    sep = "\x1f"
    path = tmp_path / "nac.json"
    _write_nac(
        path,
        {
            "reward_bias": {"a1:n_nan": math.nan, "a1:n_inf": math.inf, "a1:n_ok": 0.2},
            "goal_reward_bias": {"g_nan": math.nan, "g_ok": 0.15},  # in range: #1102 clamps to +-max_reward_bias
            "cluster_reward_bias": {f"a1{sep}c1{sep}tool:x": math.nan, f"a1{sep}c2{sep}tool:x": 0.4},
            "inherent_bias_keys": [f"a1{sep}c1{sep}tool:x"],
            "percept_valences": {f"a1{sep}dragon{sep}burn": -math.inf, f"a1{sep}fire{sep}burn": -0.1},
            "cluster_fear": {f"a1{sep}c1{sep}burn": math.nan, f"a1{sep}c2{sep}burn": -math.inf},
            "event_outcome_welford": {f"a1{sep}tool:x": {"mean": math.nan, "m2": 0.0, "n": 1.0}},
        },
    )
    nac = NAc()
    with caplog.at_level(logging.WARNING, logger="maxim.decisions.nac"):
        nac.load(str(path))  # no offline decay by default (owner decision 2026-10-17)

    assert nac._reward_bias == {("a1", "n_ok"): 0.2}
    assert nac._goal_reward_bias == {"g_ok": 0.15}
    assert nac._cluster_reward_bias == {("a1", "c2", "tool:x"): 0.4}
    assert nac.inherent_bias_keys == frozenset()  # its bias was dropped: no dangling marker
    assert nac._percept_valences == {("a1", "fire", "burn"): -0.1}
    # The clamp read NaN / -inf as the DEEPEST fear; both must be gone, not capped.
    assert nac._cluster_fear == {}
    assert nac._event_outcome_welford == {}
    assert "non-finite" in caplog.text
    # Nothing left that decay cannot prune, nor a widest-threshold override.
    nac.decay_reward_biases()
    assert all(math.isfinite(v) for v in nac.get_threshold_overrides("a1", base_threshold=0.85).values())


# ── 2. read-only session observer ───────────────────────────────────────────


@pytest.mark.single_agent_only
def test_session_observer_does_not_decay_a_stale_nac(tmp_path):
    from maxim.session import _build_session_observer

    path = tmp_path / "aut_nac.json"
    nac = NAc()
    with nac._lock:
        nac._cluster_reward_bias[("a1", "c1", "tool:warm")] = 0.8
        nac._reward_bias[("a1", "n1")] = 0.2
    nac.save(str(path))
    state = json.loads(path.read_text())
    state["saved_at"] = time.time() - 30 * 86400.0  # a month stale: decay would prune both
    path.write_text(json.dumps(state))

    observer = _build_session_observer(tmp_path)

    assert observer._nac.cluster_reward_bias("a1", "c1", "tool:warm") == pytest.approx(0.8)
    assert observer._nac._reward_bias == {("a1", "n1"): pytest.approx(0.2)}


# ── 3. locks ────────────────────────────────────────────────────────────────


class _LockCheckedDict(dict):
    """An ``_eligibility`` that fails any iteration not made under NAc's lock."""

    def __init__(self, lock, *args):
        super().__init__(*args)
        self._nac_lock = lock
        self.unlocked_reads = 0

    def items(self):
        if not self._nac_lock._is_owned():
            self.unlocked_reads += 1
        return super().items()


def test_distributor_reads_eligibility_under_nacs_lock():
    nac = NAc()
    nac._eligibility = _LockCheckedDict(nac._lock)
    nac.update_eligibility("a1", "n1", 1.0)
    nac.update_eligibility("a2", "n2", 1.0)
    dist = TemporalCreditDistributor(nac=nac, scn=SCN())

    credited = dist.distribute("a1", 1.0)
    dist.anticipatory_pre_activate("a1")

    assert [nid for nid, _ in credited] == ["n1"]
    assert nac._eligibility.unlocked_reads == 0


def test_eligibility_snapshot_is_a_filtered_copy():
    nac = NAc()
    nac.update_eligibility("a1", "n1", 0.5)
    nac.update_eligibility("a1", "faint", 0.005)
    nac.update_eligibility("a2", "n2", 0.9)
    snap = nac.eligibility_snapshot("a1")
    assert snap == {"n1": 0.5}
    snap["n1"] = 99.0
    assert nac._eligibility[("a1", "n1")] == 0.5


def test_distribute_survives_concurrent_decay():
    """Hammer decay/update on one thread while another distributes."""
    import sys

    nac = NAc()
    dist = TemporalCreditDistributor(nac=nac, scn=SCN())
    errors: list[BaseException] = []
    stop = threading.Event()

    def churn():
        i = 0
        while not stop.is_set():
            for k in range(50):
                nac.update_eligibility("a1", f"n{i}_{k}", 0.02)
            nac.decay_eligibility(factor=0.4)  # deletes most keys each pass
            i += 1

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    worker = threading.Thread(target=churn, daemon=True)
    worker.start()
    try:
        for _ in range(300):
            try:
                dist.distribute("a1", 0.1)
            except RuntimeError as exc:  # "dictionary changed size during iteration"
                errors.append(exc)
                break
    finally:
        stop.set()
        worker.join(timeout=5)
        sys.setswitchinterval(old)
    assert errors == []


def test_decay_all_holds_the_lock():
    nac = NAc()
    nac.observe("tool_call", "tool_a", "result", "success", Valence.POSITIVE, 0.5)
    held: list[bool] = []
    link = next(iter(nac._links.values()))[0]
    original = link.decay

    def spy(factor):
        held.append(nac._lock._is_owned())
        original(factor)

    link.decay = spy
    nac.decay_all(0.9)
    assert held == [True]


# ── 4. per-tick decay independence: tests/unit/test_loop_section_extraction.py ──


# ── 5. reward_bias key round-trip ───────────────────────────────────────────


@pytest.mark.single_agent_only
def test_node_id_with_colons_round_trips(tmp_path):
    nac = NAc()
    nac.credit_node("agent-1", "tool:use:dodge", 1.0)
    path = tmp_path / "nac.json"
    nac.save(str(path))
    nac2 = NAc()
    nac2.load(str(path))  # no offline decay by default (owner decision 2026-10-17)
    assert nac2._reward_bias == nac._reward_bias
    assert ("agent-1", "tool:use:dodge") in nac2._reward_bias


@pytest.mark.parametrize("entry", ["credit_node", "update_eligibility"])
@pytest.mark.single_agent_only
def test_agent_id_with_colon_is_refused(entry):
    nac = NAc()
    with pytest.raises(ValueError, match="must not contain ':'"):
        getattr(nac, entry)("ns:agent", "n1", 1.0)
    assert nac._reward_bias == {} and nac._eligibility == {}


@pytest.mark.single_agent_only
def test_legacy_reward_bias_file_loads(tmp_path):
    path = tmp_path / "nac.json"
    path.write_text(json.dumps({"_format_version": "1.0", "reward_bias": {"a1:tool:x": 0.15, "a1:n1": 0.1}}))
    nac = NAc()
    nac.load(str(path))
    assert nac._reward_bias == {("a1", "tool:x"): 0.15, ("a1", "n1"): 0.1}


# ── 6. version drift ────────────────────────────────────────────────────────


def test_get_version_matches_the_saved_format(tmp_path):
    path = tmp_path / "nac.json"
    NAc().save(str(path))
    assert NAc().get_version() == json.loads(path.read_text())["_format_version"] == _NAC_FORMAT_VERSION
