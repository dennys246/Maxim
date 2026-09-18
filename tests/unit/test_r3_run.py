"""R3 harness — the pure halves (report, gauntlet, drift, numerics, the seed plan) and an OFFLINE
campaign smoke on the scripted water bridge: the apparatus row plus one event per in-process arm
(A innate-only, B in-situ, C self-learned, E exposed-ablated) against a game-rate scripted drowning,
then the gauntlet written from the A rows and drift judged against a real row. Arm D (the Exp 61
donor → receiver seam) is live-only here: its seams are Exp 61's, proven by that campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world import r3_run as R  # noqa: E402
from survival_world.scripted_water import ScriptedWaterBridge, ScriptedWaterControl  # noqa: E402
from survival_world.setup_world import water_anchor_record, water_classroom_geometry  # noqa: E402


def _event(
    arm: str,
    seed: int,
    *,
    end: str = "surface",
    t_surface: float | None = 8.4,
    health_lost: float = 0.0,
    oxygen_pain: float = 2.9,
    decisive: bool = True,
    sat: float = 20.0,
    tick: float = 0.62,
    git: str = "abc",
) -> dict:
    return {
        "kind": "event",
        "campaign_id": "c1",
        "arm": arm,
        "seed": seed,
        "refusal": None,
        "decisive": decisive,
        "provenance": {"executed_git_hash": git},
        "apparatus_record_ts": 1.0,
        "anchor_measured": {"t_damage_onset_min_s": 16.06},
        "frozen": {"exp60_frozen_sha256": R.FROZEN["exp60_frozen_sha256"]},
        "event": {
            "end": end,
            "survived": end == "surface",
            "t_surface": t_surface,
            "health_lost": health_lost,
            "escaped_before_damage": t_surface is not None and t_surface < 16.0,
            "pain_seconds": {"oxygen": oxygen_pain, "health": 0.0 if health_lost == 0 else 2.0},
            "food_at_teleport": {"foodLevel": 20.0, "foodSaturationLevel": sat, "foodExhaustionLevel": 0.0},
            "tick_period_median_s": tick,
            "tick_period_iqr_s": 0.1,
            "max_state_age_s": 0.1,
        },
    }


def _campaign_rows(n: int = 12) -> list[dict]:
    rows = [
        {
            "kind": "apparatus",
            "campaign_id": "c1",
            "refusal": None,
            "provenance": {"executed_git_hash": "abc"},
            "flee_preflight": {"latency_s": 0.01},
        }
    ]
    per = {
        "A_innate_only": (27.8, 11.0, 22.0),
        "B_in_situ": (8.4, 0.0, 2.9),
        "C_self_learned": (3.1, 0.0, 0.0),
        "D_shared": (3.2, 0.0, 0.0),
        "E_exposed_ablated": (27.5, 11.0, 22.0),
    }
    for arm, (t, hl, op) in per.items():
        for i in range(n):
            rows.append(_event(arm, 300 + i, t_surface=t + 0.05 * i, health_lost=hl, oxygen_pain=op))
    return rows


def test_report_complete_with_named_contrasts_and_intervals() -> None:
    rep = R.report(_campaign_rows(), campaign_id="c1")
    assert rep["status"] == "COMPLETE", rep["incomplete_cause"]
    assert rep["arms"]["A_innate_only"]["survived"] == 12 and rep["arms"]["A_innate_only"]["escaped_before_damage"] == 0
    assert rep["arms"]["C_self_learned"]["escaped_before_damage"] == 12
    c = rep["contrasts"]["C_minus_A"]
    assert c["t_surface_median_diff_s"] < -20 and c["t_surface_mann_whitney_p"] < 0.001 and "innate" in c["mechanism"]
    assert rep["contrasts"]["D_beside_C"]["mechanism"].endswith("never a contrast")
    assert rep["arms"]["B_in_situ"]["t_surface"]["median_ci95"] is not None
    assert "graduated" in rep["what_this_is"]


def test_report_incomplete_causes_are_named() -> None:
    rows = _campaign_rows(n=12)
    rows = [r for r in rows if not (r["kind"] == "event" and r["arm"] == "E_exposed_ablated" and r["seed"] >= 306)]
    rows[5]["provenance"] = {"executed_git_hash": "zzz"}
    rows[6]["decisive"] = False
    for r in rows:
        if r["kind"] == "event" and r["arm"] == "A_innate_only":
            r["decisive"] = False
    rep = R.report(rows, campaign_id="c1")
    assert rep["status"] == "INCOMPLETE"
    cause = rep["incomplete_cause"]
    assert (
        "E_exposed_ablated: 6 clean rows < 12" in cause
        and "2 code hashes" in cause
        and "A_innate_only: no drive-decisive" in cause
    )


def test_gauntlet_written_from_the_floor_rows_and_drift_judged() -> None:
    cal = [
        _event(
            "A_innate_only", 400 + i, t_surface=27.8 + 0.1 * i, health_lost=11.0, oxygen_pain=22.0, sat=20.0 + (i % 2)
        )
        for i in range(12)
    ]
    ap = {"kind": "apparatus", "refusal": None, "flee_preflight": {"latency_s": 0.01, "bridge": {"ok": True}}}
    g = R.write_gauntlet(cal, ap, campaign_id="cal1")
    assert g["cal_code_hash"] == "abc" and g["n_cal"] == 12 and g["depth"] == R.FROZEN["depth"]
    assert g["reservoir_band"]["foodSaturationLevel"] == [19.0, 22.0] and g["floor_arm"]["survived"] == 12
    assert g["tick_period_band_s"] == [0.42, 0.82]
    ok = _event("B_in_situ", 320)
    assert R.gauntlet_drift(g, ok) == []
    bad = _event("B_in_situ", 321, sat=5.0, tick=1.5)
    bad["apparatus_record_ts"] = 2.0
    why = R.gauntlet_drift(g, bad)
    assert (
        any("reservoir" in w for w in why)
        and any("tick period" in w for w in why)
        and any("apparatus record" in w for w in why)
    )
    stale = _event("B_in_situ", 322)
    stale["event"]["max_state_age_s"] = 0.4
    assert any("stale" in w for w in R.gauntlet_drift(g, stale))
    assert R.validate_gauntlet(g) == []
    hollow = {**g, "tick_period_band_s": None}
    assert any("tick_period_band_s" in w for w in R.gauntlet_drift(hollow, ok))
    assert any("no tick_period_band_s" in w for w in R.validate_gauntlet(hollow))
    assert any("kind" in w for w in R.validate_gauntlet({**g, "kind": "other"}))
    rep_g = R.report(_campaign_rows(), campaign_id="c1", gauntlet={**g, "cal_code_hash": "zzz"})
    assert "other than the gauntlet's" in rep_g["incomplete_cause"]
    with pytest.raises(SystemExit):
        R.write_gauntlet(cal + [_event("A_innate_only", 499, git="other")], ap, campaign_id="cal1")


def test_numerics_and_seed_plan() -> None:
    assert R.bootstrap_median_ci([1.0]) is None
    lo, hi = R.bootstrap_median_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lo <= 3.0 <= hi
    assert R.mann_whitney_p([], [1.0]) is None and R.mann_whitney_p([1, 2, 3], [10, 11, 12]) < 0.2
    seeds = sum(R.FROZEN["seeds"].values(), []) + R.FROZEN["cal"]["seeds"]
    assert len(seeds) == len(set(seeds)) == 72
    assert R.FROZEN["arms"] == {arm: 12 for arm in R.ARMS} and R.DETACHED == {"A_innate_only", "E_exposed_ablated"}


# ── offline campaign smoke ──


def _xyz(t: list) -> dict[str, float]:
    return {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])}


@pytest.mark.timeout(600)
def test_offline_campaign_apparatus_and_one_event_per_in_process_arm(tmp_path: Path, monkeypatch) -> None:
    from survival_world import water_trial as WT

    rec = water_anchor_record(water_classroom_geometry(10, 20, depth=5, shore_y=64))
    rec["measured"] = {"t_damage_onset_min_s": 16.0, "t_damage_onset_max_s": 16.2, "t_pain_edge_min_s": 5.1}
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text(json.dumps(rec))
    srv = ScriptedWaterBridge(
        shore=_xyz(rec["shore"]), submerged=_xyz(rec["submerged"]), damage_onset_s=16.0, damage_per_s=2.0
    )
    rcon = ScriptedWaterControl(srv, gamerules={r: v for r, v in WT.R3_GAMERULES})
    apparatus = {"all_pass": True, "ts": 123.0}
    monkeypatch.setattr(R, "ANCHOR_FILE", anchor_path)
    monkeypatch.setattr(
        R, "load_json", lambda p: apparatus if str(p).endswith(R.APPARATUS_RECORD) else json.loads(Path(p).read_text())
    )
    monkeypatch.setattr(R, "min_pain_edge_s", lambda a: 5.1)
    monkeypatch.setattr(R.C, "RconControl", lambda *a, **k: rcon)
    monkeypatch.setattr(
        WT.WaterTrial, "check_fingerprint", lambda self, m: {"offline": True}
    )  # a live-apparatus pin (Exp 61)
    monkeypatch.setitem(R.FROZEN, "lethal_cap_s", 32.0)
    monkeypatch.setitem(R.FROZEN60, "K_usable_episodes", 1)
    monkeypatch.setitem(R.FROZEN60, "loop_liveness_s", 2.0)
    monkeypatch.setitem(R.FROZEN60, "loop_liveness_min_ticks", 3)
    monkeypatch.setitem(R.FROZEN60, "loop_warm_s", 0.5)
    out = tmp_path / "r3.jsonl"
    args = argparse.Namespace(
        rcon_host="h",
        rcon_port=1,
        rcon_password="p",
        username="maxim",
        bridge_host="127.0.0.1",
        bridge_port=srv.port,
        workdir=str(tmp_path / "wd"),
        out=str(out),
        gate_record="x",
        allow_dirty=True,
    )
    camp = R._R3(args, provenance={"executed_git_hash": "offline"}, out_path=out, campaign_id="smoke")
    assert camp.train_cap_s == pytest.approx(15.0) and camp.probe_cap_s == pytest.approx(4.35)
    try:
        ap = camp.apparatus_row()
        assert ap["refusal"] is None, ap["refusal"]
        assert ap["flee_preflight"]["bridge"]["ok"] and ap["actuation"]["t_surface"] is not None
        a = camp.event_row("A_innate_only", 400)
        b = camp.event_row("B_in_situ", 320)
        c = camp.event_row("C_self_learned", 340)
        e = camp.event_row("E_exposed_ablated", 380)
    finally:
        srv.close()
    # C: trained (propose-only, rescued) → boundary clean → carried fear fires at once, before the in-situ learner
    assert c["refusal"] is None, c["refusal"]
    assert c["training"]["usable_episodes"] >= 1 and c["fear_before"] and c["decisive"] is True
    assert c["event"]["end"] == "surface" and c["event"]["t_surface"] < b["event"]["t_surface"]
    # E: the same exposure with the subscriber detached → no fear; the innate route, like A
    assert e["refusal"] is None, e["refusal"]
    assert e["fear_before"] == {} and e["event"]["escaped_before_damage"] is False and e["event"]["health_lost"] > 0
    assert all(r["flee_preflight"]["bridge"]["ok"] for r in (a, b, c, e))
    # A: the innate route — no fear, surfaced AFTER damage with health paid; drive-decisive
    assert a["refusal"] is None, a["refusal"]
    assert (
        a["event"]["end"] == "surface"
        and a["event"]["escaped_before_damage"] is False
        and a["event"]["health_lost"] > 0
    )
    assert a["fear_after"] == {} and a["decisive"] is True and a["detached_count"] >= 1
    assert 16.0 < a["event"]["t_surface"] < 30.0 and a["event"]["min_health"] < 14.0  # the innate route, after the band
    # B: the in-situ learner — fear booked, surfaced BEFORE damage, no health paid, oxygen pain paid; decisive
    assert b["refusal"] is None, b["refusal"]
    assert (
        b["event"]["end"] == "surface"
        and b["event"]["escaped_before_damage"] is True
        and b["event"]["health_lost"] == 0.0
    )
    assert b["fear_after"] and b["decisive"] is True and b["event"]["pain_seconds"]["oxygen"] > 0
    assert b["event"]["t_surface"] < a["event"]["t_surface"]
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert [r["kind"] for r in rows] == ["apparatus", "event", "event", "event", "event"]
    g = R.write_gauntlet(rows, ap, campaign_id="smoke")
    assert g["n_cal"] == 1 and g["floor_arm"]["survived"] == 1
    if g.get("tick_period_band_s"):  # one cal row may carry no IQR → no band (never a zero-width one)
        assert R.validate_gauntlet(g) == [] and R.gauntlet_drift(g, b) == []
    rep = R.report(rows, campaign_id="smoke")
    assert rep["status"] == "INCOMPLETE" and "A_innate_only: 1 clean rows < 12" in rep["incomplete_cause"]
    assert rep["arms"]["B_in_situ"]["drive_decisive"] == 1
    assert rep["arms"]["A_innate_only"]["idle_tick_period_median_s"]["n"] == 1  # computed from the row's ticks


def test_amendment_2_recounts_only_pure_tick_band_refusals_with_complete_events() -> None:
    rows = _campaign_rows()
    band = _event("C_self_learned", 500, t_surface=3.1, tick=0.79)
    band["refusal"] = "Refusal: gauntlet drift: tick period median 0.786 outside the gauntlet band [0.394, 0.768]"
    two = _event("D_shared", 501, t_surface=3.1, tick=0.79)
    two["refusal"] = (
        "gauntlet drift: tick period median 0.79 outside the gauntlet band [0.394, 0.768]; reservoir foodSaturationLevel=5.0 outside the gauntlet band [19.0, 21.0]"
    )
    stale = _event("E_exposed_ablated", 502, t_surface=27.7)
    stale["refusal"] = "Refusal: a sample was stale (0.169 s > 0.15) — cal and bench alike"
    capped = _event("C_self_learned", 503, t_surface=None, end="cap", tick=0.79)
    capped["refusal"] = "gauntlet drift: tick period median 0.79 outside the gauntlet band [0.394, 0.768]"
    out, recounted = R.reclassify_under_amendments(rows + [band, two, stale, capped])
    assert [(r["arm"], r["seed"]) for r in recounted] == [("C_self_learned", 500)]
    by = {(r["arm"], r["seed"]): r for r in out if r.get("kind") == "event"}
    assert by[("C_self_learned", 500)]["refusal"] is None and by[("C_self_learned", 500)]["amended"]["amendment"] == 2
    assert (
        by[("D_shared", 501)]["refusal"]
        and by[("E_exposed_ablated", 502)]["refusal"]
        and by[("C_self_learned", 503)]["refusal"]
    )
    assert R.reclassify_under_amendments([])[1] == []
    # narrower than a hand recount: a BELOW-band or MISSING median is F22's regression signature — never recounted
    below = _event("C_self_learned", 504, t_surface=3.1, tick=0.2)
    below["refusal"] = "gauntlet drift: tick period median 0.2 outside the gauntlet band [0.394, 0.768]"
    missing = _event("C_self_learned", 505, t_surface=3.1, tick=0.79)
    missing["event"]["tick_period_median_s"] = None
    missing["refusal"] = "gauntlet drift: tick period median None outside the gauntlet band [0.394, 0.768]"
    assert R.reclassify_under_amendments([below, missing])[1] == []
    # the frozen supersede rule wins: a LATER clean row for the same (arm, seed) blocks the recount — never both
    orig = _event("C_self_learned", 506, t_surface=3.1, tick=0.79)
    orig["refusal"] = "gauntlet drift: tick period median 0.79 outside the gauntlet band [0.394, 0.768]"
    orig["ts"] = 1.0
    rerun = _event("C_self_learned", 506, t_surface=3.2)
    rerun["ts"] = 2.0
    out2, rec2 = R.reclassify_under_amendments([orig, rerun])
    assert rec2 == [] and out2[0]["refusal"] and out2[1]["refusal"] is None
    earlier = dict(rerun, ts=0.5)  # an EARLIER clean row is not a supersede — the recount stands
    assert len(R.reclassify_under_amendments([earlier, orig])[1]) == 1


def test_amendment_1_harness_unchanged_between_reads_git_diff(monkeypatch) -> None:
    import subprocess as sp

    class _Out:
        def __init__(self, stdout, rc=0):
            self.stdout, self.returncode, self.stderr = stdout, rc, ""

    calls: list[str] = []

    def _git(diff_out: str, *, ancestor: bool = True, diff_rc: int = 0):
        def run(cmd, **k):
            calls.append(cmd[1])
            if cmd[1] == "merge-base":
                return _Out("", rc=0 if ancestor else 1)
            return _Out(diff_out, rc=diff_rc)

        return run

    monkeypatch.setattr(sp, "run", _git("docs/experiments/data/r3_cal.jsonl\ndocs/experiments/x.md\n"))
    assert R.harness_unchanged_between("a", "b") == (True, []) and calls == ["merge-base", "diff"]
    for path in ("scripts/survival_world/r3_run.py", "pyproject.toml", "data/robots.yaml", "scenarios/x.yaml"):
        monkeypatch.setattr(sp, "run", _git(path + "\n"))
        ok, touched = R.harness_unchanged_between("a", "b")
        assert not ok and touched == [path], path
    monkeypatch.setattr(sp, "run", _git("", ancestor=False))  # not an ancestor → fail-closed, no diff read
    calls.clear()
    assert R.harness_unchanged_between("a", "b")[0] is False and calls == ["merge-base"]
    monkeypatch.setattr(sp, "run", _git("", diff_rc=128))
    assert R.harness_unchanged_between("a", "b")[0] is False


def test_amendment_1_holds_for_the_real_r3_cal_and_bench_hashes() -> None:
    ok, touched = R.harness_unchanged_between("6b16bbe9", "4cca5524")  # cal PR merge-base → bench hash
    assert ok and touched == []
    assert R.harness_unchanged_between("4cca5524", "6b16bbe9")[0] is False  # the reverse is not an ancestor


def test_report_hash_rule_yields_to_ancestry_flag() -> None:
    rows = _campaign_rows()
    g = {"cal_code_hash": "other", "reservoir_band": [19.0, 21.0], "tick_period_band_s": [0.394, 0.768]}
    assert "other than the gauntlet" in (R.report(rows, gauntlet=g)["incomplete_cause"] or "")
    assert "other than the gauntlet" not in (
        R.report(rows, gauntlet=g, hash_rule_satisfied_by_ancestry=True)["incomplete_cause"] or ""
    )


def test_idle_tick_period_ignores_proposing_ticks() -> None:
    ev = {
        "ticks": [
            {"t": 0.0},
            {"t": 0.5},
            {"t": 1.0, "proposal": "x_flee"},
            {"t": 1.8, "proposal": "x_escape_water"},
            {"t": 2.3},
            {"t": 2.8},
        ]
    }
    assert R.idle_tick_period_median_s(ev) == 0.5  # the 0.8 s tie-break dispatch is excluded
    assert R.idle_tick_period_median_s({"ticks": [{"t": 0.0}, {"t": 0.7, "proposal": "x"}]}) is None
