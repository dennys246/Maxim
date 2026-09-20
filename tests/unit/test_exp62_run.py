"""Exp 62 rung A harness — the pure halves, and the whole row end to end against a two-pool bridge.

The standing rule is that a rig-bound harness runs end to end OFFLINE before its PR (import checks
and hand-typed stubs hide what the rig finds). So the gates, the read classification and the verdict
are asserted directly here, and one row of each arm runs against ``ScriptedWaterBridge`` with BOTH
pools — the real ``build_aut`` assembly, the real training, the real ``live_g2``, the real placement.

The arm this file exists to protect is :func:`classify_g2`. ``live_g2`` RAISES on a failed read,
which is correct for Exp 60/61 (there, fear not reading is a broken instrument) and wrong here: at
the read pool a failed read is the finding the pre-registration names. The classification splits the
two by asking whether the fear was booked on the TRAINING clusters at all — booked-and-missed is a
clean row carrying the finding, not-booked is a refusal. Both directions are pinned below, because
getting this backwards would either drop the result or publish a training failure as evidence.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from survival_world import exp62_run as E  # noqa: E402
from survival_world.exp61_run import FROZEN as FROZEN61  # noqa: E402

FLOOR = E.FROZEN["read_floor"]
CAP = E.FROZEN["fear_value_cap"]
ESC = "minecraft_player_escape_water"


# ─────────────────────────── the frozen block ───────────────────────────


def test_the_borrowed_constants_are_checked_against_their_sources_not_copied_from_them() -> None:
    """A later Exp 60/61 edit must fail HERE, never be inherited silently into a frozen campaign.

    The blocks are LITERALS for exactly this reason: a deep copy taken at import would take any
    source edit with it, and the equality below could never fail. Pinned by mutating the source.
    """
    assert E.frozen_matches() == []
    assert E.FROZEN["exp60"] == FROZEN61["exp60"]
    assert E.FROZEN["read_floor"] == FROZEN61["read_floor"] == 0.5
    assert E.FROZEN["fear_value_cap"] == FROZEN61["fear_value_cap"] == -1.0
    assert E.FROZEN["cosine_threshold"] == E.FROZEN["exp60"]["fingerprint"]["encoder_pattern_threshold"]


def test_a_one_sided_edit_to_exp61s_block_is_reported_as_drift(monkeypatch) -> None:
    """The leg that a deep copy would have made vacuous — mutate the SOURCE and the check must fire."""
    monkeypatch.setitem(FROZEN61["settle_guard"], "nearest_player_dist", 999.0)
    assert any("settle_guard" in d for d in E.frozen_matches())
    monkeypatch.undo()
    monkeypatch.setitem(FROZEN61["exp60"], "death_cap", 99)
    assert any("exp60" in d for d in E.frozen_matches())


def test_the_arms_read_the_pools_the_prereg_says() -> None:
    assert E.FROZEN["read_pool"] == {"cross": "pool2", "same": "pool1", "cross_ablated": "pool2"}
    assert E.FROZEN["arms"] == {"cross": 12, "same": 12, "cross_ablated": 3}
    assert E.FROZEN["train_pool"] == "pool1"
    # the contrast is WHERE IT READS and nothing else: cross and same differ in read_pool alone
    assert E.FROZEN["read_pool"]["cross"] != E.FROZEN["read_pool"]["same"]
    assert E.FROZEN["g2_arm"]["cross"] == E.FROZEN["g2_arm"]["same"] == "fear"
    assert set(E.FROZEN["read_pool"]) == set(E.FROZEN["arms"]) == set(E.FROZEN["seeds"]) == set(E.ARM_ORDER)


def test_seeds_do_not_collide_between_arms() -> None:
    seen: set[int] = set()
    for arm, seeds in E.FROZEN["seeds"].items():
        assert len(seeds) == E.FROZEN["arms"][arm]
        assert not (seen & set(seeds)), f"{arm} reuses a seed"
        seen |= set(seeds)


# ─────────────────────────── classify_g2: the confound guard ───────────────────────────


def _fields(*, trained="N1", probe="N1", shore="S1", need_probe=0.9, needs=None, water=CAP, shore_fear=0.0):
    return {
        "water_fear": water,
        "shore_fear": shore_fear,
        "live_g2": {
            "training_majority_cluster": trained,
            "probe_water_cluster": probe,
            "probe_shore_cluster": shore,
            "need_probe_cluster": need_probe,
            "need_episode_clusters": {"N1": 0.9} if needs is None else needs,
        },
    }


def test_booked_and_read_at_the_trained_node_passes_all_four_clauses() -> None:
    out = E.classify_g2(_fields(), arm="cross", floor=FLOOR)
    assert out["refusal"] is None and out["pass"] is True
    assert out["same_node"] and out["booked_at_training_nodes"]


def test_booked_but_not_read_at_the_read_pool_is_a_CLEAN_ROW_carrying_the_finding() -> None:
    """The row this whole harness exists to get right: a carry MISS is a result, never a refusal."""
    out = E.classify_g2(_fields(probe="N2", need_probe=0.0), arm="cross", floor=FLOOR)
    assert out["refusal"] is None, "a carry miss must not be dropped as an instrument failure"
    assert out["pass"] is False
    assert out["booked_at_training_nodes"] is True
    assert "!= trained node" in out["why"] and "strict floor" in out["why"]


def test_not_booked_at_the_training_nodes_REFUSES_so_a_training_failure_is_never_a_carry_failure() -> None:
    out = E.classify_g2(_fields(needs={"N1": 0.0}, need_probe=0.0), arm="cross", floor=FLOOR)
    assert out["pass"] is False
    assert "training failure must not be reported as a carry failure" in out["refusal"]


def test_a_need_exactly_at_the_floor_is_not_booked_the_consumer_floor_is_strict() -> None:
    out = E.classify_g2(_fields(needs={"N1": FLOOR}), arm="cross", floor=FLOOR)
    assert out["refusal"] is not None, "the loop's activation floor is >, not >="


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"water": -0.6}, "is not at the cap"),
        ({"shore_fear": -0.4}, "specificity"),
        ({"need_probe": FLOOR}, "strict floor"),
        ({"probe": "N2"}, "!= trained node"),
    ],
)
def test_each_clause_of_the_mechanism_read_can_fail_on_its_own_and_is_named(kwargs, fragment) -> None:
    out = E.classify_g2(_fields(**kwargs), arm="cross", floor=FLOOR)
    assert out["refusal"] is None and out["pass"] is False
    assert fragment in out["why"], out["why"]


def test_one_live_cluster_for_the_read_pools_shore_and_water_refuses() -> None:
    out = E.classify_g2(_fields(probe="X", shore="X"), arm="cross", floor=FLOOR)
    assert "ONE live cluster" in out["refusal"]


def test_the_ablated_arm_passes_only_on_nothing_readable_and_a_leak_is_an_instrument_failure() -> None:
    clean = E.classify_g2(
        _fields(water=0.0, shore_fear=0.0, need_probe=0.0, needs={"N1": 0.0}), arm="cross_ablated", floor=FLOOR
    )
    assert clean["refusal"] is None and clean["ablation_held"] is True and clean["pass"] is None
    leaked = E.classify_g2(_fields(), arm="cross_ablated", floor=FLOOR)
    assert leaked["ablation_held"] is False and "detach did not hold" in leaked["refusal"]
    assert "never a null" in leaked["refusal"]


# ─────────────────────────── the replay row ───────────────────────────


def _rec(shore_y: float, sub_y: float, spawn=None):
    return {
        "shore": [10.0, shore_y, 20.0],
        "submerged": [10.0, sub_y, 20.0],
        "world_spawn": spawn if spawn is not None else {"x": 0.0, "y": 64.0, "z": 0.0},
    }


# Pool 1 as its own committed probe saw it: y_altitude 0.2734 -> y 35, distance_from_spawn 0.7736
# -> d 70.04 (the encoder's normalizations, inverted). An anchor that re-derives to these is the
# only one the replay will predict from once a gate record is cited.
POOL1_SUB_Y = 35.0
POOL1_DIST = 0.7736 * 256.0 - 128.0
POOL1_SPAWN = {"x": 10.0 - math.sqrt(POOL1_DIST**2 - (64.0 - POOL1_SUB_Y) ** 2), "y": 64.0, "z": 20.0}


def _real_pool1_anchor() -> dict:
    return {
        "shore": [10.0, 40.0, 20.0],
        "submerged": [10.0, POOL1_SUB_Y, 20.0],
        "world_spawn": POOL1_SPAWN,
    }


def test_the_replay_runs_on_the_BUILT_geometry_and_reproduces_its_own_live_record() -> None:
    out = E.replay_prediction(_rec(64, 59), _rec(105, 100))
    assert out["reproduces_live_record"] == pytest.approx(0.7874, abs=0.002)
    assert out["predicted_cross_hit"] is True and out["cross_pool_cosine"] > out["threshold"]
    # …and the gate is not vacuous: pool 2 must still separate its OWN shore from its OWN water
    assert out["pool2_separates_internally"] is True
    assert out["pool1"]["submerged_y"] == 59.0 and out["pool2"]["submerged_y"] == 100.0


def test_two_different_world_spawns_refuse_because_the_distance_is_not_derivable() -> None:
    with pytest.raises(E.Refusal, match="ONE stamped world_spawn"):
        E.replay_prediction(_rec(64, 59), _rec(105, 100, spawn={"x": 8.0, "y": 64.0, "z": 0.0}))
    with pytest.raises(E.Refusal, match="ONE stamped world_spawn"):
        E.replay_prediction(_rec(64, 59), _rec(105, 100, spawn=None) | {"world_spawn": None})


def test_a_replay_that_no_longer_reproduces_its_record_is_not_citable(tmp_path: Path) -> None:
    """The replay's own guard, wired to the refusal: a moved sensor→basis mapping is not citable."""
    import shutil

    src = E.REPLAY_SCRIPT.read_text().replace('LIVE = float(rec["cosine"]["a4_gained"])', "LIVE = 0.1234")
    shim = tmp_path / "replay_shim.py"
    shim.write_text(src)
    # the module reads its record beside itself (`Path(__file__).with_name`)
    shutil.copyfile(
        E.REPLAY_SCRIPT.with_name("exp60_geometry_2026-09-15b.json"), tmp_path / "exp60_geometry_2026-09-15b.json"
    )
    with pytest.raises(E.Refusal, match="no longer reproduces"):
        E.replay_prediction(_rec(64, 59), _rec(105, 100), script=shim)


def test_replay_consistency_names_the_disagreement_in_both_directions() -> None:
    hit = [{"predicted_cross_hit": True, "cross_pool_cosine": 0.99}]
    assert E.replay_consistency(hit, 1.0)["pass"] is True
    miss = E.replay_consistency(hit, 0.25)
    assert miss["pass"] is False and "wrong about something live" in miss["note"]
    other = E.replay_consistency([{"predicted_cross_hit": False}], 1.0)
    assert other["pass"] is False and "understated the body" in other["note"]
    assert E.replay_consistency([], 1.0)["pass"] is False, "citing nothing is not a pass"


# ─────────────────────────── the verdict ───────────────────────────


def _row(arm, seed, *, node=True, success=True, t_air=2.5, calls=1, refusal=None, ts=None, hash_="abc123abc123"):
    fc = {
        "success": success,
        "behavioural_dv": success,
        "decision_dv": success,
        "decisive": success,
        "t_first_air": t_air if success else None,
        "escape_calls": calls,
        "flee_calls": 0,
        "placement": {"calls": [{"tool": ESC, "success": True}] * calls},
    }
    # the ablated arm has NO node gate — `pass` None, and an ablation that held
    gate = {"pass": None, "ablation_held": True} if arm == "cross_ablated" else {"pass": node}
    return {
        "kind": "row",
        "arm": arm,
        "seed": seed,
        "ts": ts if ts is not None else 1000.0 + seed,
        "campaign_id": "c1",
        "provenance": {"executed_git_hash": hash_},
        "refusal": refusal,
        "node_gate": gate,
        "first_contact": fc,
    }


def _campaign(**over):
    rows = [{"kind": "apparatus", "campaign_id": "c1", "refusal": None, "ts": 1.0}]
    rows.append(
        {
            "kind": "replay",
            "campaign_id": "c1",
            "refusal": None,
            "ts": 2.0,
            "predicted_cross_hit": True,
            "cross_pool_cosine": 0.99,
        }
    )
    rows += [_row("cross", 600 + i, **over.get("cross", {})) for i in range(12)]
    rows += [_row("same", 620 + i, **over.get("same", {})) for i in range(12)]
    rows += [
        _row(
            "cross_ablated",
            640 + i,
            **{"success": False, "t_air": None, "calls": 0, **over.get("abl", {})},
        )
        for i in range(3)
    ]
    return rows


def test_verdict_earned_on_the_expected_campaign() -> None:
    v = E.compute_verdict(_campaign(), campaign_id="c1")
    assert v["verdict"] == "EARNED", v["incomplete_cause"] or v["checks"]
    assert all(v["checks"].values())
    assert v["rates"]["cross"]["node_gate"]["rate"] == 1.0
    assert v["fisher_cross_vs_ablated"]["p_one_sided"] < 0.05


def test_a_node_gate_miss_the_replay_also_predicted_is_a_NULL_with_the_carry_named() -> None:
    """When offline and live AGREE that the fear does not carry, the campaign is a clean NULL.

    (A miss the replay predicted a HIT for is INCOMPLETE instead — the disagreement outranks the
    rung, and is pinned separately below.)
    """
    rows = _campaign()
    next(r for r in rows if r.get("kind") == "replay")["predicted_cross_hit"] = False
    for r in rows:
        if r.get("arm") == "cross":
            r["node_gate"] = {"pass": False}
            r["first_contact"].update(
                {"success": False, "behavioural_dv": False, "decision_dv": False, "t_first_air": None}
            )
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["checks"]["REPLAY"] is True, "offline and live agree; the replay gate is not what fails here"
    assert v["verdict"] == "NULL"
    assert "does not carry across pools" in v["incomplete_cause"]


def test_one_node_miss_in_twelve_still_clears_the_frozen_tolerance() -> None:
    rows = _campaign()
    next(r for r in rows if r.get("arm") == "cross")["node_gate"] = {"pass": False}
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["checks"]["NODE"] is True and v["rates"]["cross"]["node_gate"]["rate"] == pytest.approx(11 / 12)


def test_a_failed_same_arm_says_the_apparatus_is_what_was_measured_not_the_carry() -> None:
    v = E.compute_verdict(_campaign(same={"success": False, "t_air": None}), campaign_id="c1")
    assert v["verdict"] == "NULL" and "the apparatus, not the carry" in v["incomplete_cause"]


def test_anti_vacuity_fails_when_the_ablated_arm_calls_the_executor_at_all() -> None:
    """Counted over EVERY executor call the spy recorded, not just the two water affordances."""
    v = E.compute_verdict(_campaign(abl={"success": False, "t_air": None, "calls": 1}), campaign_id="c1")
    assert v["checks"]["ANTI_VACUITY"] is False and v["verdict"] == "INCOMPLETE"
    assert v["rates"]["cross_ablated"]["executor_calls"] == 3


def test_a_missing_replay_or_apparatus_row_is_INCOMPLETE_not_a_quiet_pass() -> None:
    rows = [r for r in _campaign() if r.get("kind") != "replay"]
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["verdict"] == "INCOMPLETE" and "no replay row" in v["incomplete_cause"]
    rows = [r for r in _campaign() if r.get("kind") != "apparatus"]
    assert "no apparatus citation row" in E.compute_verdict(rows, campaign_id="c1")["incomplete_cause"]


def test_an_offline_hit_with_a_live_miss_is_INCOMPLETE_with_the_disagreement_named() -> None:
    rows = _campaign()
    for r in rows:
        if r.get("arm") == "cross":
            r["node_gate"] = {"pass": False}
    v = E.compute_verdict(rows, campaign_id="c1")
    # CROSS still passes (the behaviour), so the verdict turns on the replay disagreement
    assert v["checks"]["REPLAY"] is False
    assert v["verdict"] == "INCOMPLETE" and "wrong about something live" in v["incomplete_cause"]


def test_rows_spanning_two_code_hashes_are_INCOMPLETE() -> None:
    rows = _campaign()
    next(r for r in rows if r.get("arm") == "same")["provenance"] = {"executed_git_hash": "deadbeefdead"}
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["verdict"] == "INCOMPLETE"
    assert any("code hashes" in m for m in v["refused"])


def test_a_refusal_is_named_and_never_counted_as_a_zero() -> None:
    rows = _campaign()
    rows.append(_row("cross", 611, success=False, refusal="bridge died", ts=9999.0))
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["rates"]["cross"]["n"] == 12 and v["rates"]["cross"]["rate"] == 1.0
    assert any("bridge died" in m for m in v["refused"])


def test_a_later_clean_row_supersedes_an_earlier_refusal_and_clean_duplicates_refuse() -> None:
    rows = _campaign()
    rows.insert(2, _row("cross", 600, success=False, refusal="first attempt died", ts=0.5))
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["n_clean"]["cross"] == 12 and any("first attempt died" in m for m in v["refused"])
    rows.append(_row("cross", 600, ts=9999.0))
    assert any("duplicate clean" in m for m in E.compute_verdict(rows, campaign_id="c1")["refused"])


def test_the_timing_cause_fires_only_when_air_fails_not_when_decisiveness_fails() -> None:
    rows = _campaign()
    for r in rows:
        if r.get("arm") == "cross":
            r["first_contact"].update({"success": False, "behavioural_dv": False, "decision_dv": True})
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["verdict"] == "INCOMPLETE" and "actuation timing" in v["incomplete_cause"]
    rows = _campaign()
    for r in rows:
        if r.get("arm") == "cross":
            r["first_contact"].update({"success": False, "decisive": False})
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["verdict"] == "NULL" and "won by a component other than the drive" in v["incomplete_cause"]


def test_campaign_drift_reads_the_same_arm_because_it_is_the_within_pool_ceiling() -> None:
    rows = [_row("same", 620 + i, t_air=1.0, ts=float(i)) for i in range(4)]
    rows += [_row("same", 630 + i, t_air=3.0, ts=float(10 + i)) for i in range(4)]
    assert E.campaign_drift(rows, max_s=0.5) and "drifted" in E.campaign_drift(rows, max_s=0.5)[0]
    assert E.campaign_drift(rows, max_s=5.0) == []


# ─────────────────────────── wiring ───────────────────────────


def test_the_plan_interleaves_the_arms_so_drift_hits_them_equally() -> None:
    plan = E.interleaved_plan(["cross", "same", "cross_ablated"], rows=None)
    assert len(plan) == 27
    assert [a for a, _ in plan[:6]] == ["cross", "same", "cross_ablated", "cross", "same", "cross_ablated"]
    # the short arm runs out; the long ones keep alternating
    assert [a for a, _ in plan[9:11]] == ["cross", "same"]
    assert E.interleaved_plan(["cross", "same"], rows=1) == [("cross", 600), ("same", 620)]


def test_two_records_with_one_pool_id_refuse_because_a_build_overwrote_one(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    full = {"t_pain_edge_min_s": 5.1, "t_damage_onset_min_s": 16.0}
    a.write_text(json.dumps({"measured": full, "pool_id": "pool1"}))
    b.write_text(json.dumps({"measured": full, "pool_id": "pool1"}))
    with pytest.raises(E.Refusal, match="one of them was overwritten"):
        E.load_geoms(str(a), str(b))
    b.write_text(json.dumps({"pool_id": "pool2"}))
    with pytest.raises(E.Refusal, match="no `measured` block"):
        E.load_geoms(str(a), str(b))
    # a PARTIAL block is its own refusal: the caps come from these keys, and a bare KeyError three
    # frames later is not the documented refusal
    b.write_text(json.dumps({"pool_id": "pool2", "measured": {"t_pain_edge_min_s": 5.4}}))
    with pytest.raises(E.Refusal, match="no t_damage_onset_min_s"):
        E.load_geoms(str(a), str(b))


def test_existing_clean_skips_only_clean_rows_of_this_campaign(tmp_path: Path) -> None:
    p = tmp_path / "rows.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                _row("cross", 600),
                _row("cross", 601, refusal="died"),
                {**_row("cross", 602), "campaign_id": "other"},
                {"not": "json-row"},
            )
        )
    )
    assert E.existing_clean(p, "c1") == {("cross", 600)}


# ─────────────────────────── end to end, offline, two pools ───────────────────────────


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_one_row_of_each_arm_against_a_two_pool_scripted_bridge(tmp_path: Path, monkeypatch) -> None:
    """The whole row, offline: real assembly, real preflights, real training, real g2, real placement.

    Pool 2 is a second submerged point on the SAME scripted bridge — with one submerged point its
    floor would read dry and the cross arm would be meaningless. What is asserted is the SHAPE the
    rig must produce: the same arm surfaces from its own trained node; the cross arm produces a
    CLEAN row either way (its node gate is the measurement, whichever way it lands); the ablated arm
    is censored with zero executor calls.

    What this canNOT settle: the scripted bridge lights its water (`light_level` 9/14, never the
    sealed shell's 0) and its oxygen/pain timings are its own, so the offline cross arm's node gate
    landing TRUE is a fact about the plumbing, not evidence for the claim. The place absolutes do
    differ between the two pools here (y 60 v 101, and the distances with them), which is what makes
    the arm non-vacuous as a plumbing test. The live answer is the rig's, and the pre-check's.
    """
    import argparse

    from survival_world.scripted_water import ScriptedWaterBridge, ScriptedWaterControl

    pool1 = {
        "shore": [10.0, 64.0, 10.0],
        "submerged": [10.0, 60.0, 20.0],
        "deaths_objective": "exp62_deaths",
        "pool_id": "pool1",
        "measured": {"t_damage_onset_min_s": 16.0, "t_damage_onset_max_s": 16.2, "t_pain_edge_min_s": 5.1},
    }
    pool2 = {
        "shore": [10.0, 105.0, 10.0],
        "submerged": [10.0, 101.0, 20.0],
        "deaths_objective": "exp62_deaths",
        "pool_id": "pool2",
        "measured": {"t_damage_onset_min_s": 16.0, "t_damage_onset_max_s": 16.2, "t_pain_edge_min_s": 5.4},
    }
    srv = ScriptedWaterBridge(
        shore={"x": 10.0, "y": 64.0, "z": 10.0},
        submerged=[{"x": 10.0, "y": 60.0, "z": 20.0}, {"x": 10.0, "y": 101.0, "z": 20.0}],
        damage_onset_s=16.0,
        damage_per_s=2.0,
    )
    rcon = ScriptedWaterControl(srv)
    monkeypatch.setattr("exp56.common.RconControl", lambda *a, **k: rcon)
    # Exp 60's constants, shortened so the row runs in seconds; the SHAPE of every check is the same.
    # All THREE sources move together (exp60's, exp61's, this harness's copy), so the drift guard
    # stays live through the shortening — a real one-sided edit would still refuse the campaign.
    from survival_world.exp60_run import FROZEN as FROZEN60

    for key, value in (
        ("K_usable_episodes", 1),
        ("placements_per_probe", 1),
        ("loop_liveness_s", 2.0),
        ("loop_liveness_min_ticks", 3),
        ("loop_warm_s", 0.5),
        ("shore_roam_s", 1.0),
        # `fingerprint` is NOT shortened: the offline agent's live config fingerprint equals the
        # frozen one exactly, so `check_fingerprint` runs for real here rather than being stubbed.
    ):
        for block in (FROZEN60, FROZEN61["exp60"], E.FROZEN["exp60"]):
            monkeypatch.setitem(block, key, value)
    assert E.frozen_matches() == [], "the drift guard must still be satisfied, not bypassed"

    def _stub_gate(sub_y: float) -> str:
        """A gate-(ii) record shaped like the real ones: the binding reads y_altitude, the prereg's
        gate reads light/time, and both pools carry the SAME dark, frozen-day constants."""
        path = tmp_path / f"gate_{int(sub_y)}.json"
        path.write_text(
            json.dumps(
                {
                    "cosine": {"a4_gained": 0.78, "threshold": 0.85},
                    "run_gate": {"pass": True},
                    "verdict": "separable_here",
                    "per_sensor": [
                        {"sensor": "y_altitude", "v_safe": (sub_y + 4) / 128.0, "v_dark": sub_y / 128.0},
                        {"sensor": "light_level", "v_safe": 0.0, "v_dark": 0.0},
                        {"sensor": "time_of_day", "v_safe": 0.0417, "v_dark": 0.0417},
                    ],
                }
            )
        )
        return str(path)

    gates = [_stub_gate(60.0), _stub_gate(101.0)]
    out = tmp_path / "rows.jsonl"
    args = argparse.Namespace(
        rcon_host="h",
        rcon_port=1,
        rcon_password="p",
        username="maxim",
        bridge_host="127.0.0.1",
        bridge_port=srv.port,
        workdir=str(tmp_path / "work"),
        gate_record=gates,
    )
    camp = E.Exp62Campaign(
        args,
        {"pool1": pool1, "pool2": pool2},
        provenance={"executed_git_hash": "offline"},
        out_path=out,
        campaign_id="offline",
    )
    # one cap for BOTH pools: below pool 1's 5.1 s pain edge AND pool 2's 5.4 s
    assert camp.probe_cap_s == pytest.approx(5.1 - E.FROZEN["exp60"]["probe_cap_margin_s"])
    ap = camp.apparatus_citation()
    assert ap["refusal"] is None, ap["refusal"]
    assert ap["context_constants"]["match"] is True
    # the light/time gate is LIVE here, not merely present: a lit pool 2 stops the campaign
    lit = json.loads(Path(gates[1]).read_text())
    for row in lit["per_sensor"]:
        if row["sensor"] == "light_level":
            row["v_safe"] = row["v_dark"] = 1.0
    lit_path = tmp_path / "gate_101_lit.json"
    lit_path.write_text(json.dumps(lit))
    camp.args.gate_record = [gates[0], str(lit_path)]
    assert "LIGHT/TIME contrast" in (camp.apparatus_citation()["refusal"] or "")
    camp.args.gate_record = gates
    assert camp.apparatus_citation()["refusal"] is None

    prev = Path.cwd()
    rows = {}
    try:
        for arm in ("same", "cross", "cross_ablated"):
            seed = E.FROZEN["seeds"][arm][0]
            d = camp.workdir / f"{arm}_{seed}"
            d.mkdir(parents=True, exist_ok=True)
            import os

            os.chdir(d)
            rows[arm] = camp.row(arm, seed)
    finally:
        import os

        os.chdir(prev)
        srv.close()

    same = rows["same"]
    assert same["refusal"] is None, same["refusal"]
    assert same["node_gate"]["pass"] is True, same["node_gate"]
    assert same["node_gate"]["same_node"] and same["water_fear"] == CAP
    assert same["first_contact"]["behavioural_dv"] is True, same["first_contact"]
    assert same["first_contact"]["escape_calls"] >= 1
    assert same["positive_escape_links_after_training"] == 0, "propose-only training must book no link"

    cross = rows["cross"]
    assert cross["refusal"] is None, f"a cross row must be CLEAN whichever way the node gate lands: {cross['refusal']}"
    assert cross["node_gate"]["booked_at_training_nodes"] is True
    assert isinstance(cross["node_gate"]["pass"], bool)
    if not cross["node_gate"]["pass"]:
        assert cross["node_gate"]["why"], "a node-gate miss must say WHY — it is the finding"

    abl = rows["cross_ablated"]
    assert abl["refusal"] is None, abl["refusal"]
    assert abl["detached_subscribers"] == 1
    # the ablated arm has no node gate — it has an ablation check, and publishing `pass=True` here
    # would read as "the ablated agent resolved to the trained node", its own opposite
    assert abl["water_fear"] == 0.0
    assert abl["node_gate"]["pass"] is None and abl["node_gate"]["ablation_held"] is True
    assert abl["first_contact"]["escape_calls"] == 0, "the apparatus must not surface an agent by itself"
    assert abl["first_contact"]["success"] is False

    # the rows are on disk and the verdict reads them without a live anything
    written = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert [r["kind"] for r in written].count("row") == 3
    v = E.compute_verdict(written, campaign_id="offline")
    assert v["verdict"] == "INCOMPLETE" and "clean rows <" in v["incomplete_cause"]
    assert v["rates"]["cross_ablated"]["executor_calls"] == 0


# ───────────────── the light/time gate: the red gate for the finding that blocked this PR ─────────────────

DATA = Path(__file__).resolve().parents[2] / "docs" / "experiments" / "data"
POOL1_GATE = DATA / "exp60_geometry_2026-09-15b.json"
POOL2_GATE_STALE = DATA / "exp62_pool2_geometry.json"
POOL2_GATE_RECONNECT = DATA / "exp62_pool2_geometry_reconnect.json"


def _gate(path: Path) -> dict:
    return json.loads(path.read_text())


def test_the_committed_stale_light_record_is_refused_and_the_reconnect_one_is_not() -> None:
    """Run against the REAL committed records, because this is not a hypothetical failure.

    Pool 2's first probe read `light_level` 1.0 and its reconnect read 0.0 — the prereg's named
    "stale-light read at a freshly filled box". BOTH carry `run_gate.pass: true`, so a gate-pass
    check alone cites either. Citing the stale one puts the real cross-pool cosine at 0.5878 (a
    MISS) while the synthetic replay prediction still says HIT, because light is unrepresentable
    in a place-absolutes-only construction: the campaign would have measured the LIGHT contrast
    and published it as the pool contrast.
    """
    assert _gate(POOL2_GATE_STALE)["run_gate"]["pass"] is True, "the trap is that the stale record passes its own gate"
    bad = E.context_constants_match(_gate(POOL1_GATE), _gate(POOL2_GATE_STALE))
    assert not bad["match"]
    assert any("light_level" in m for m in bad["mismatches"]), bad["mismatches"]
    good = E.context_constants_match(_gate(POOL1_GATE), _gate(POOL2_GATE_RECONNECT))
    assert good["match"], good["mismatches"]


def test_time_of_day_is_gated_too_not_only_light() -> None:
    rec = json.loads(json.dumps(_gate(POOL2_GATE_RECONNECT)))
    for row in rec["per_sensor"]:
        if row["sensor"] == "time_of_day":
            row["v_dark"] = 0.5  # noon at the floor
    out = E.context_constants_match(_gate(POOL1_GATE), rec)
    assert not out["match"] and any("time_of_day" in m for m in out["mismatches"])


def test_an_absent_context_sensor_is_a_mismatch_not_a_silent_pass() -> None:
    rec = json.loads(json.dumps(_gate(POOL2_GATE_RECONNECT)))
    rec["per_sensor"] = [r for r in rec["per_sensor"] if r["sensor"] != "light_level"]
    out = E.context_constants_match(_gate(POOL1_GATE), rec)
    assert not out["match"] and any("absent" in m for m in out["mismatches"])


def test_gate_records_are_bound_to_their_pool_by_the_probes_own_altitude() -> None:
    """Argument order alone would let a swapped pair through every other check in the harness."""
    pool1_anchor = {"submerged": [10.0, 35.0, 20.0]}
    pool2_anchor = {"submerged": [10.0, 90.0, 20.0]}
    assert E.gate_record_matches_pool(_gate(POOL1_GATE), pool1_anchor) is None
    assert E.gate_record_matches_pool(_gate(POOL2_GATE_RECONNECT), pool2_anchor) is None
    swapped = E.gate_record_matches_pool(_gate(POOL2_GATE_RECONNECT), pool1_anchor)
    assert swapped and "records swapped?" in swapped
    assert "no y_altitude" in (E.gate_record_matches_pool({"per_sensor": []}, pool1_anchor) or "")


def test_cite_gate_records_refuses_the_real_stale_pair_before_any_agent_is_built(tmp_path: Path) -> None:
    geoms = {"pool1": {"submerged": [10.0, 35.0, 20.0]}, "pool2": {"submerged": [10.0, 90.0, 20.0]}}
    with pytest.raises(E.Refusal, match="LIGHT/TIME contrast"):
        E.cite_gate_records([str(POOL1_GATE), str(POOL2_GATE_STALE)], geoms)
    cited, records = E.cite_gate_records([str(POOL1_GATE), str(POOL2_GATE_RECONNECT)], geoms)
    assert [c["pool"] for c in cited] == ["pool1", "pool2"] and set(records) == {"pool1", "pool2"}
    with pytest.raises(E.Refusal, match="pass --gate-record twice"):
        E.cite_gate_records([str(POOL1_GATE)], geoms)


def test_the_replay_refuses_when_the_records_and_the_synthetic_prediction_disagree() -> None:
    """The synthetic construction cannot see light; the records can. Disagreement is an apparatus
    refusal, not a prediction — this is the number that would have been published as HIT."""
    r1, r2 = _real_pool1_anchor(), _rec(95, 90, spawn=POOL1_SPAWN)
    with pytest.raises(E.Refusal, match="opposite sides"):
        E.replay_prediction(r1, r2, gate1=_gate(POOL1_GATE), gate2=_gate(POOL2_GATE_STALE))
    out = E.replay_prediction(r1, r2, gate1=_gate(POOL1_GATE), gate2=_gate(POOL2_GATE_RECONNECT))
    assert out["pool1_anchor_vs_record_cosine"] >= 0.999
    assert out["from_gate_records"]["predicted_cross_hit"] is True
    assert out["from_gate_records"]["cross_pool_cosine"] > out["threshold"]


def test_the_replay_refuses_a_pool1_record_it_does_not_predict_from() -> None:
    foreign = json.loads(json.dumps(_gate(POOL1_GATE)))
    for row in foreign["per_sensor"]:
        if row["sensor"] == "y_altitude":
            row["v_dark"] = 0.9
    with pytest.raises(E.Refusal, match="not the one the replay predicts from"):
        E.replay_prediction(
            _real_pool1_anchor(), _rec(95, 90, spawn=POOL1_SPAWN), gate1=foreign, gate2=_gate(POOL2_GATE_RECONNECT)
        )


def test_an_anchor_describing_a_different_pool_than_the_cited_record_refuses() -> None:
    """anchor <-> cited record <-> the record the replay predicts from: all three must be one pool."""
    wrong = _real_pool1_anchor()
    wrong["submerged"] = [10.0, 59.0, 20.0]  # a pool 1 at a different altitude than its probe saw
    with pytest.raises(E.Refusal, match="describe different pools"):
        E.replay_prediction(
            wrong, _rec(95, 90, spawn=POOL1_SPAWN), gate1=_gate(POOL1_GATE), gate2=_gate(POOL2_GATE_RECONNECT)
        )


# ───────────────── the remaining review folds ─────────────────


def test_a_failed_encode_at_the_read_pool_refuses_rather_than_scoring_a_node_miss() -> None:
    """`encode_world_cluster` returns None on an encode failure — it never raises into the loop.
    Scoring that as a carry miss would publish an instrument failure as the finding."""
    out = E.classify_g2(_fields(probe=None, shore=None, need_probe=0.0), arm="cross", floor=FLOOR)
    assert out["pass"] is False and "no live world cluster" in out["refusal"]


def test_a_missing_shore_cluster_refuses_because_specificity_would_be_credited_vacuously() -> None:
    """`live_g2` sets shore_fear to 0.0 when the shore cluster is absent — which SATISFIES the
    specificity clause. A row must not pass a gate it never measured."""
    out = E.classify_g2(_fields(shore=None), arm="cross", floor=FLOOR)
    assert out["pass"] is False and "specificity cannot be measured" in out["refusal"]


def test_a_none_need_refuses_rather_than_raising_out_of_the_rows_handler() -> None:
    out = E.classify_g2(_fields(needs={"N1": None}), arm="cross", floor=FLOOR)
    assert out["refusal"] is not None and "training failure" in out["refusal"]


def test_the_ablated_arm_has_no_node_gate_and_is_not_counted_in_one() -> None:
    """`pass=True` there would publish as 'the ablated arm resolved to the trained node in 3/3'."""
    out = E.classify_g2(_fields(water=0.0, need_probe=0.0, needs={"N1": 0.0}), arm="cross_ablated", floor=FLOOR)
    assert out["pass"] is None and out["ablation_held"] is True
    v = E.compute_verdict(_campaign(), campaign_id="c1")
    assert v["rates"]["cross_ablated"]["node_gate"]["n"] == 0
    assert v["rates"]["cross_ablated"]["node_gate"]["rate"] is None


def test_a_duplicate_clean_row_blocks_EARNED_instead_of_letting_the_first_silently_win() -> None:
    rows = _campaign()
    dup = _row("cross", 600, success=False, ts=9999.0)
    rows.append(dup)
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["verdict"] == "INCOMPLETE", "the first row would otherwise supply every number"
    assert "duplicate clean" in v["incomplete_cause"]


def test_an_anti_vacuity_failure_is_an_INSTRUMENT_statement_not_a_mechanism_NULL() -> None:
    v = E.compute_verdict(_campaign(abl={"success": True, "t_air": 2.0, "calls": 1}), campaign_id="c1")
    assert v["checks"]["ANTI_VACUITY"] is False
    assert v["verdict"] == "INCOMPLETE"
    assert "the apparatus, not the carry" in v["incomplete_cause"]


def test_every_non_earned_verdict_names_a_cause() -> None:
    cases = [
        _campaign(cross={"success": False, "t_air": None}),
        _campaign(same={"success": False, "t_air": None}),
        _campaign(abl={"success": False, "t_air": None, "calls": 2}),
        [r for r in _campaign() if r.get("kind") != "replay"],
    ]
    rows = _campaign()
    next(r for r in rows if r.get("arm") == "same")["provenance"] = {"executed_git_hash": "deadbeefdead"}
    cases.append(rows)
    for i, c in enumerate(cases):
        v = E.compute_verdict(c, campaign_id="c1")
        assert v["verdict"] != "EARNED"
        assert v["incomplete_cause"], f"case {i} produced a bare {v['verdict']} with no cause"


def test_rows_with_an_unknown_arm_are_named_not_silently_dropped() -> None:
    rows = _campaign()
    rows.append(_row("cross_lit", 700, ts=5000.0))
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["verdict"] == "INCOMPLETE" and "unknown arm" in v["incomplete_cause"]


def test_world_spawn_is_compared_numerically_so_a_list_and_a_dict_agree() -> None:
    a = _rec(64, 59, spawn=[0.0, 64.0, 0.0])
    b = _rec(105, 100, spawn={"x": 0, "y": 64, "z": 0})
    out = E.replay_prediction(a, b)
    assert out["pool1"]["distance_from_spawn"] == pytest.approx(22.91, abs=0.01)
