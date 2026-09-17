"""Pure halves of the Exp 61 campaign harness (scripts/survival_world/exp61_run.py): the Wilson
interval, Fisher's exact test, the decision-provenance clause, the first-contact classifier,
staged donor sanity, the per-arm ingest gate, the anti-vacuity kit, campaign drift, the pair plan
and the verdict; plus the guards the two-lens review asked for (the export body-spec pinned to the
component, the frozen Exp 60 copy pinned to Exp 60's FROZEN, the discount pinned to the ingest
constant). The live halves are proven by the scripted water-bridge smoke
(`test_water_trial_smoke.py`) offline and the one-pair dry run on the bridge box, never here."""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import pytest
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world import exp60_run as H  # noqa: E402
from survival_world import exp61_run as E  # noqa: E402

ESC = "minecraft_player_escape_water"
FLEE = "minecraft_player_flee"


def _event(best=ESC, drive=0.75, causal=0.0, learned=0.0):
    return {
        "best_tool": best,
        "passed_gate": True,
        "score_components": {"drive": drive, "causal": causal, "learned_bias": learned, "explore": 0.0},
    }


def _placement(*, surfaced=True, latency=2.9, calls=None, dirty=False):
    return {
        "surfaced": surfaced,
        "latency_s": latency if surfaced else None,
        "censored": not surfaced,
        "dirty": dirty,
        "calls": calls or [],
    }


def _row(
    arm,
    seed,
    *,
    success=True,
    decision=True,
    air=True,
    decisive=True,
    refusal=None,
    code="abc",
    t_air=3.0,
    rekeyed=None,
    shipped=1,
    gate_pass=True,
    ts=0.0,
    kind="receiver",
):
    return {
        "kind": kind,
        "campaign_id": "c1",
        "arm": arm,
        "pair_seed": seed,
        "ts": ts,
        "refusal": refusal,
        "provenance": {"executed_git_hash": code},
        "ingest": None if rekeyed is None else {"fear_rekeyed": rekeyed, "fear_dropped": shipped - rekeyed},
        "donor": None if rekeyed is None else {"fear_shipped": shipped},
        "representation_gate": {"pass": gate_pass},
        "first_contact": {
            "success": success,
            "decision_dv": decision,
            "behavioural_dv": air,
            "decisive": decisive,
            "t_first_air": t_air if air else None,
        },
    }


def _apparatus_row(seed, *, t_act=1.5, ts=0.0, code="abc"):
    return {
        "kind": "apparatus",
        "campaign_id": "c1",
        "arm": "-",
        "pair_seed": seed,
        "ts": ts,
        "refusal": None,
        "provenance": {"executed_git_hash": code},
        "apparatus": {"actuation": {"t_surface": t_act}},
    }


def _kit_row(*, ok=True, ts=0.5):
    return {
        "kind": "anti_vacuity",
        "campaign_id": "c1",
        "arm": "transferred",
        "pair_seed": 200,
        "ts": ts,
        "refusal": None,
        "kit": {"pass": ok},
    }


def _campaign(
    *, transferred=12, isolated=24, cnf=12, dangling=24, t_rate=1.0, i_rate=0.0, c_rate=0.0, d_rate=0.0, kit=True
):
    rows = []
    ts = 0.0

    def add(arm, n, rate, **kw):
        nonlocal ts
        for i in range(n):
            ok = i < round(rate * n)
            ts += 1.0
            rows.append(_row(arm, 200 + i, success=ok, decision=ok, air=ok, decisive=ok, ts=ts, **kw))

    add("transferred", transferred, t_rate)
    add("isolated", isolated, i_rate)
    add("cluster_not_fear", cnf, c_rate)
    add("dangling", dangling, d_rate, rekeyed=0)
    if kit:
        rows.append(_kit_row())
    return rows


# ── numerics ──


def test_wilson_interval_matches_known_values() -> None:
    lo, hi = E.wilson_interval(0, 24)
    assert lo == 0.0 and 0.13 < hi < 0.15  # the D2 arithmetic: 0/24 upper bound ≈ 0.14 < the 0.20 margins
    lo, hi = E.wilson_interval(0, 12)
    assert 0.23 < hi < 0.26  # and why n = 12 was not enough for the floor arms
    assert E.wilson_interval(12, 12)[1] == 1.0
    assert E.wilson_interval(0, 0) == (0.0, 1.0)


def test_fisher_exact_one_sided_against_hand_hypergeometric() -> None:
    """The executor lens's cases: the denominator is draws-of-n_a (comb(n, n_a)), not comb(n, k)."""
    assert E.fisher_one_sided_p(0, 12, 0, 24)["p_one_sided"] == 1.0
    assert E.fisher_one_sided_p(12, 12, 0, 24)["p_one_sided"] == pytest.approx(1 / 1251677700)
    hand = sum(comb(12, x) * comb(24, 12 - x) for x in range(6, 13)) / comb(36, 12)  # 6/12 vs 6/24, N=36 K=12 n=12
    assert E.fisher_one_sided_p(6, 12, 6, 24)["p_one_sided"] == pytest.approx(hand)
    assert 0.12 < hand < 0.14
    # symmetric form: P(X >= k_a) over A's draws equals P(Y <= k_b) over B's draws
    a = E.fisher_one_sided_p(9, 12, 2, 24)["p_one_sided"]
    b_le = sum(comb(11, y) * comb(25, 24 - y) for y in range(0, 3)) / comb(36, 24)
    assert a == pytest.approx(b_le)
    assert E.fisher_one_sided_p(1, 0, 0, 5)["p_one_sided"] is None


# ── decision-provenance clause ──


def test_decisive_requires_drive_with_zero_causal_and_learned() -> None:
    assert E.decision_decisive(_event())[0]
    assert not E.decision_decisive(_event(causal=0.4))[0]
    assert not E.decision_decisive(_event(learned=0.3))[0]
    assert not E.decision_decisive(_event(drive=0.0))[0]
    assert not E.decision_decisive(_event(best=FLEE))[0]
    ok, why = E.decision_decisive(None)
    assert not ok and "no NAc_RECOMMEND" in why


# ── first contact ──


def test_first_contact_success_needs_both_dvs_and_decisiveness() -> None:
    calls = [{"t": 0.87, "tool": FLEE, "success": False}, {"t": 1.55, "tool": ESC, "success": True}]
    fc = E.first_contact_outcome(_placement(calls=calls), cap_s=4.335, decision_event=_event())
    assert fc["success"] and fc["decision_dv"] and fc["behavioural_dv"] and fc["decisive"]
    assert fc["t_flee_call"] == 0.87 and fc["t_escape_call"] == 1.55 and fc["t_first_air"] == 2.9
    assert fc["censoring"] is None and fc["refusal"] is None
    # surfaced but won by the causal channel → counts AGAINST, not a refusal
    fc = E.first_contact_outcome(_placement(calls=calls), cap_s=4.335, decision_event=_event(causal=0.9))
    assert not fc["success"] and fc["behavioural_dv"] and fc["refusal"] is None


def test_first_contact_censoring_classes_and_the_two_refusals() -> None:
    fc = E.first_contact_outcome(
        _placement(surfaced=False, calls=[{"t": 3.9, "tool": ESC, "success": True}]),
        cap_s=4.335,
        decision_event=_event(),
    )
    assert not fc["success"] and fc["decision_dv"] and fc["censoring"].startswith("escape called before the cap")
    fc = E.first_contact_outcome(_placement(surfaced=False), cap_s=4.335, decision_event=None)
    assert fc["censoring"] == "no escape call inside the window" and not fc["decision_dv"] and fc["refusal"] is None
    # head in air with ZERO executor calls → apparatus refusal
    fc = E.first_contact_outcome(_placement(surfaced=True, calls=[]), cap_s=4.335, decision_event=_event())
    assert fc["refusal"] and "ZERO executor calls" in fc["refusal"] and not fc["success"]
    # escape EXECUTED but no proposal captured → instrument refusal, never a mechanism null
    fc = E.first_contact_outcome(
        _placement(surfaced=True, calls=[{"t": 1.5, "tool": ESC, "success": True}]), cap_s=4.335, decision_event=None
    )
    assert fc["refusal"] and "sink not delivering" in fc["refusal"]
    # a DIRTY placement is its own class and never a behavioural success
    fc = E.first_contact_outcome(
        _placement(surfaced=True, dirty=True, calls=[{"t": 1.0, "tool": ESC, "success": True}]),
        cap_s=4.335,
        decision_event=_event(),
    )
    assert not fc["behavioural_dv"] and not fc["success"] and fc["censoring"].startswith("US/damage")


# ── staged donor sanity ──


def _stage(
    tmp_path: Path,
    *,
    fear: dict | None,
    links=None,
    water="w1",
    tag="gABC",
    nodes=None,
    name="stage",
    reward_bias=None,
) -> Path:
    stage = tmp_path / name
    stage.mkdir(parents=True)
    nac = {
        "links": links or {},
        "event_outcome_welford": {},
        "cluster_reward_bias": {},
        "reward_bias": reward_bias or {},
        "percept_valences": {"a\x1fdrowning\x1fdrive:oxygen": -0.5},
        "cluster_fear": fear if fear is not None else {},
        "saved_at": 1.0,
    }
    (stage / "aut_nac.json").write_text(json.dumps(nac))
    ec_nodes = nodes or {water: {"modality": "world", "geometry": tag}, "s1": {"modality": "world", "geometry": tag}}
    (stage / "aut_ec.json").write_text(json.dumps({"substrate_nodes": ec_nodes}))
    return stage


def test_donor_sanity_passes_a_clean_fear_donor_and_stamps_meta(tmp_path: Path) -> None:
    stage = _stage(tmp_path, fear={"a\x1fw1\x1fdrive:oxygen": -1.0})
    s = E.donor_sanity_staged(stage, donor_kind="fear", episode_clusters=["w1"], shore_node="s1")
    assert s["pass"], s["reasons"]
    assert s["fear_shipped"] == 1 and s["geometry_tag"] == "gABC" and s["world_nodes"] == 2
    assert len(s["nac_sha256"]) == 64


@pytest.mark.parametrize(
    "fear, links, episodes, why",
    [
        ({}, None, ["w1"], "no cluster_fear"),
        ({"a\x1fw1\x1fdrive:oxygen": -0.5}, None, ["w1"], "!= cap"),
        ({"a\x1fw1\x1fdrive:health": -1.0}, None, ["w1"], "only drive:oxygen"),
        ({"a\x1fs1\x1fdrive:oxygen": -1.0}, None, ["s1"], "SHORE"),
        ({"a\x1fw1\x1fdrive:oxygen": -1.0}, None, ["other"], "never noted"),
        (
            {"a\x1fw1\x1fdrive:oxygen": -1.0},
            {"tool:x": [{"outcome_valence": "positive"}]},
            ["w1"],
            "links is not empty",
        ),
    ],
)
def test_donor_sanity_refuses_each_named_shape(tmp_path: Path, fear, links, episodes, why) -> None:
    stage = _stage(tmp_path, fear=fear, links=links)
    s = E.donor_sanity_staged(stage, donor_kind="fear", episode_clusters=episodes, shore_node="s1")
    assert not s["pass"] and any(why in r for r in s["reasons"]), s["reasons"]


def test_donor_sanity_tolerates_the_pain_credits_zero_valued_node_keys_and_refuses_a_positive_one(
    tmp_path: Path,
) -> None:
    """Dry run 2026-09-17 (pair 200): training alone leaves ZERO-valued `reward_bias` keys — the
    pain's negative credit clamped at 0.0 by `NAc.credit_node` — and both donors were refused as
    "a probe happened". A zero key reads like an absent one; only a POSITIVE bias proves a positive
    reaction (relief / success) was credited before export."""
    fear = {"a\x1fw1\x1fdrive:oxygen": -1.0}
    zero = {"a:w1": 0.0, "a:s1": 0.0, "a:other": 0.0}
    s = E.donor_sanity_staged(
        _stage(tmp_path, fear=fear, reward_bias=zero), donor_kind="fear", episode_clusters=["w1"], shore_node="s1"
    )
    assert s["pass"], s["reasons"]
    assert s["reward_bias_zero_nodes"] == 3
    s = E.donor_sanity_staged(
        _stage(tmp_path, fear=fear, reward_bias={**zero, "a:s1": 0.05}, name="pos"),
        donor_kind="fear",
        episode_clusters=["w1"],
        shore_node="s1",
    )
    assert not s["pass"] and any("non-zero node bias" in r for r in s["reasons"]), s["reasons"]
    assert s["reward_bias_zero_nodes"] == 2


def test_donor_sanity_ablated_must_carry_no_fear_and_the_staged_world_node_must_exist(tmp_path: Path) -> None:
    stage = _stage(tmp_path, fear={"a\x1fw1\x1fdrive:oxygen": -1.0})
    s = E.donor_sanity_staged(stage, donor_kind="ablated", episode_clusters=["w1"], shore_node="s1")
    assert not s["pass"] and any("ablated donor carries" in r for r in s["reasons"])
    stage2 = _stage(
        tmp_path,
        fear={},
        nodes={"w1": {"modality": "world", "geometry": "g1"}, "s1": {"modality": "world", "geometry": "g2"}},
        name="b",
    )
    s = E.donor_sanity_staged(stage2, donor_kind="ablated", episode_clusters=["w1"], shore_node="s1")
    assert not s["pass"] and any("geometry tag" in r for r in s["reasons"])
    # the S1 trap's footprint: a staged EC with NO world node (the liveness-close snapshot) is refused
    stage3 = _stage(tmp_path, fear={}, nodes={"i1": {"modality": "interoception", "geometry": "g1"}}, name="c")
    s = E.donor_sanity_staged(stage3, donor_kind="ablated", episode_clusters=["w1"], shore_node=None)
    assert not s["pass"] and any("no world node" in r for r in s["reasons"])


# ── ingest gate ──


def test_ingest_gate_per_arm() -> None:
    ok = {"fear_rekeyed": 1, "fear_dropped": 0, "fear_below_floor": 0, "fear_discount": 0.75}
    none_entry = {"fear_rekeyed": 0, "fear_dropped": 0, "fear_below_floor": 0, "fear_discount": None}
    assert E.ingest_gate("transferred", ok, shipped=1) is None
    assert "below_floor" in E.ingest_gate("transferred", {**ok, "fear_below_floor": 1}, shipped=1)
    assert "discount" in E.ingest_gate("transferred", {**ok, "fear_discount": 1.0}, shipped=1)
    assert "shipped no fear" in E.ingest_gate("transferred", ok, shipped=0)
    assert E.ingest_gate("cluster_not_fear", none_entry, shipped=0) is None
    assert E.ingest_gate("dangling", {**none_entry, "fear_dropped": 2}, shipped=2) is None
    assert "LOUDLY" in E.ingest_gate("dangling", {**ok, "fear_dropped": 1}, shipped=2)
    assert E.ingest_gate("transferred", {}, shipped=1) is not None  # a MISSING key is a violation, never a silent pass


# ── the anti-vacuity kit (real substrate_merge over staged files) ──


def test_anti_vacuity_kit_reads_the_fear_only_through_the_real_fold(tmp_path: Path) -> None:
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.similarity.ec import ECConfig, EntorhinalCortex
    from maxim.similarity.encoder import SensorEncoder

    nac = NAc(NACConfig())
    ec = EntorhinalCortex(ECConfig())
    enc = SensorEncoder(ec=ec, atl=None, nac=nac)
    node = enc.encode_sensors(
        agent_id="donor",
        sensors={"is_in_water": 1.0, "oxygen": 12.0},
        modality="world",
        ranges={"is_in_water": (-1, 1), "oxygen": (0, 40)},
    )
    nac.record_cluster_fear("donor", str(node), "drive:oxygen", 1.0)
    nac.record_cluster_fear("donor", str(node), "drive:oxygen", 1.0)
    donor = tmp_path / "donor_stage"
    donor.mkdir()
    (donor / "aut_nac.json").write_text(json.dumps(nac.dump()))
    ec.save(str(donor / "aut_ec.json"))
    recv = tmp_path / "recv_pre"
    recv.mkdir()
    (recv / "aut_nac.json").write_text(json.dumps(NAc(NACConfig()).dump()))
    EntorhinalCortex(ECConfig()).save(str(recv / "aut_ec.json"))
    kit = E.anti_vacuity_kit(donor, recv, receiver_agent_id="recv")
    assert kit["pass"], kit
    assert kit["real_read"] > 0.5 and kit["real_fear_rekeyed"] == 1
    assert kit["variants"] == {"receiver_unchanged": 0.0, "empty_state": 0.0}


# ── drift, pair plan, verdict ──


def test_campaign_drift_reads_apparatus_rows_and_transfer_latency() -> None:
    rows = [_row("transferred", i, t_air=2.9 + 0.1 * i, ts=i) for i in range(12)]
    assert any("latency" in p for p in E.campaign_drift(rows, max_s=0.5))
    rows = [_apparatus_row(200 + i, t_act=1.4 + 0.1 * i, ts=i) for i in range(12)]
    assert any("actuation" in p for p in E.campaign_drift(rows, max_s=0.5))
    assert E.campaign_drift([_row("transferred", i, t_air=3.0, ts=i) for i in range(12)], max_s=0.5) == []


def test_pair_plan_reuses_the_arm2_donor_for_the_second_dozen_dangling_pairs() -> None:
    arms, donors = E.pair_plan(0, 200, n_full=12, offset=12)
    assert arms == list(E.ARM_ORDER) and donors == {"transferred": 200, "cluster_not_fear": 200, "dangling": 200}
    arms, donors = E.pair_plan(12, 212, n_full=12, offset=12)
    assert arms == ["isolated", "dangling"] and donors == {"dangling": 200}


def test_verdict_earned_on_the_expected_campaign() -> None:
    v = E.compute_verdict(_campaign(), campaign_id="c1")
    assert v["verdict"] == "EARNED", v
    assert v["n_clean"] == {"isolated": 24, "transferred": 12, "cluster_not_fear": 12, "dangling": 24}
    assert v["rates"]["transferred"]["rate"] == 1.0 and v["rates"]["isolated"]["wilson95"][1] < 0.20
    assert v["permutation"]["transferred_vs_isolated"]["p_one_sided"] < 0.05
    assert v["checks"]["anti_vacuity"] is True


def test_verdict_requires_the_anti_vacuity_kit_row() -> None:
    v = E.compute_verdict(_campaign(kit=False), campaign_id="c1")
    assert v["verdict"] == "INCOMPLETE" and "anti-vacuity" in v["incomplete_cause"]
    rows = _campaign(kit=False) + [_kit_row(ok=False)]
    assert E.compute_verdict(rows, campaign_id="c1")["verdict"] == "NULL"


def test_verdict_null_when_transfer_fails_and_incomplete_on_n_or_hashes_including_donor_rows() -> None:
    assert E.compute_verdict(_campaign(t_rate=0.0), campaign_id="c1")["verdict"] == "NULL"
    v = E.compute_verdict(_campaign(transferred=11), campaign_id="c1")
    assert v["verdict"] == "INCOMPLETE" and "transferred: 11" in v["incomplete_cause"]
    rows = _campaign() + [_row("fear", 200, kind="donor", code="other")]
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["verdict"] == "INCOMPLETE" and any("code hashes" in r for r in v["refused"])


def test_verdict_both_halves_and_specificity_are_load_bearing() -> None:
    v = E.compute_verdict(_campaign(d_rate=0.5), campaign_id="c1")
    assert v["verdict"] == "NULL" and not v["checks"]["both_halves"]
    rows = _campaign()
    for r in rows:
        if r["arm"] == "dangling":
            r["ingest"]["fear_dropped"] = 0  # the accounting half: dropped must equal shipped
    assert not E.compute_verdict(rows, campaign_id="c1")["checks"]["both_halves"]
    rows = _campaign()
    rows[0]["representation_gate"]["pass"] = False
    assert not E.compute_verdict(rows, campaign_id="c1")["checks"]["specificity"]


def test_verdict_timing_cause_only_when_air_fails_not_when_decisiveness_fails() -> None:
    rows = _campaign()
    for r in rows:
        if r.get("kind") == "receiver" and r["arm"] == "transferred":
            r["first_contact"].update({"success": False, "behavioural_dv": False, "decision_dv": True})
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["verdict"] == "INCOMPLETE" and "actuation timing" in v["incomplete_cause"]
    rows = _campaign()
    for r in rows:
        if r.get("kind") == "receiver" and r["arm"] == "transferred":
            r["first_contact"].update({"success": False, "decisive": False})  # surfaced, but won by another component
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["verdict"] == "NULL" and "won by a component other than the drive" in v["incomplete_cause"]


def test_verdict_later_clean_row_supersedes_an_earlier_refusal_and_clean_duplicates_refuse() -> None:
    rows = _campaign()
    rows.append(_row("transferred", 200, refusal="bridge stale", ts=-1.0))  # earlier refusal, same key
    v = E.compute_verdict(rows, campaign_id="c1")
    assert v["verdict"] == "EARNED" and any("bridge stale" in r for r in v["refused"])
    rows = _campaign() + [_row("transferred", 200, ts=999.0)]  # a second CLEAN row for one key
    v = E.compute_verdict(rows, campaign_id="c1")
    assert any("duplicate clean" in r for r in v["refused"])
    other = [dict(r, campaign_id="c2") for r in _campaign()]
    assert E.compute_verdict(_campaign() + other, campaign_id="c2")["verdict"] == "EARNED"


# ── the review's guards ──


def test_frozen_discount_matches_the_ingest_constant() -> None:
    from maxim.hivemind.ingest import FOREIGN_FEAR_DISCOUNT

    assert E.FROZEN["foreign_fear_discount"] == FOREIGN_FEAR_DISCOUNT


def test_frozen_exp60_copy_matches_exp60_frozen() -> None:
    """A later Exp 60 edit must fail here, never be inherited silently (architecture lens S7)."""
    assert E.exp60_frozen_matches() == []
    for k, v in E.FROZEN["exp60"].items():
        assert H.FROZEN[k] == v, k


def test_export_body_spec_matches_the_component_affordances() -> None:
    """`maxim substrate export --body-yaml` reads a `body:`-rooted spec (executor lens B2)."""
    body = yaml.safe_load(E.BODY_YAML.read_text())["body"]
    component = yaml.safe_load(E.COMPONENT_YAML.read_text())["entity"]
    assert body["name"] == component["name"] == E.BODY_REF
    assert sorted(body["modulators"]["avatar"]["affordances"]) == sorted(
        component["modulators"]["avatar"]["affordances"]
    )
    from maxim.embodiment.spec import load_spec

    spec = load_spec(E.BODY_YAML)  # what the CLI does
    assert spec.name == E.BODY_REF
