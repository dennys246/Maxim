"""Exp 60 trial harness — the pure halves: placement classification, the verdict, the
permutation test, and the fingerprint comparator (scripts/survival_world/exp60_run.py).

The live `run` half is operator-run; these pin the parts a refactor could silently flip:
a DIRTY placement (damage inside the window) can never count as a surface, a seed with
no clean placements makes the verdict INCOMPLETE (never a silent drop), the gates read
the way the prereg states them, and 5 v 5 exact permutation bottoms out at 1/252.

`--apparatus-record` (the 1.3.2 re-run's record): the named record is the one authorized
(never the default in its place), a missing, refused or anchor-mismatched record ends the
run with exit 3 before any row, and an OFFLINE end-to-end run (`main(["run", ...])` on the
scripted water bridge, both arms, then `main(["verdict", ...])`) stamps the record's path
and digest into every row and the verdict (feedback: run rig scripts offline first).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from survival_world import exp60_run as H  # noqa: E402


def _samples(seq, dt=0.25, health=20.0):
    return [{"t": round(i * dt, 3), "in_water": w, "health": health} for i, w in enumerate(seq)]


class TestClassifyPlacement:
    def test_surface_is_the_first_air_read_after_being_submerged(self):
        s = _samples([False, True, True, True, False, True])  # a pre-teleport shore read, then dive, then air
        r = H.classify_placement(s, cap_s=15.0)
        assert r["surfaced"] and r["latency_s"] == 1.0 and not r["censored"] and not r["dirty"]

    def test_never_surfacing_is_censored_at_the_cap(self):
        s = _samples([True] * 61)
        r = H.classify_placement(s, cap_s=15.0)
        assert not r["surfaced"] and r["censored"] and not r["dirty"] and r["latency_s"] is None

    def test_damage_inside_the_window_is_dirty_and_censored_even_if_air_follows(self):
        s = _samples([True] * 8 + [True, False, False])
        s[8]["health"] = 18.0
        r = H.classify_placement(s, cap_s=15.0)
        assert r["dirty"] and r["censored"] and not r["surfaced"]

    def test_samples_past_the_cap_are_ignored(self):
        s = _samples([True] * 61 + [False])
        r = H.classify_placement(s, cap_s=15.0)
        assert not r["surfaced"] and r["censored"]

    def test_never_submerged_is_flagged(self):
        r = H.classify_placement(_samples([False] * 8), cap_s=15.0)
        assert r["never_submerged"] and not r["surfaced"]


class TestPSurface:
    def test_dirty_and_never_submerged_placements_are_excluded(self):
        probe = {
            "placements": [
                {"surfaced": True, "dirty": False},
                {"surfaced": False, "dirty": False},
                {"surfaced": True, "dirty": True},  # excluded
                {"surfaced": False, "dirty": False, "never_submerged": True},  # excluded
            ]
        }
        assert H.p_surface(probe) == 0.5
        assert H.p_surface({"placements": [{"surfaced": True, "dirty": True}]}) is None


class TestPermutation:
    def test_clear_separation_bottoms_out_at_one_over_252(self):
        r = H.exact_permutation_p([1.0, 1.0, 0.8, 1.0, 0.9], [0.0, 0.2, 0.0, 0.1, 0.0])
        assert r["relabellings"] == 252 and r["p_one_sided"] == pytest.approx(1 / 252)

    def test_identical_groups_are_not_significant(self):
        r = H.exact_permutation_p([0.5] * 5, [0.5] * 5)
        assert r["p_one_sided"] == 1.0


def _seed(arm, seed, pre, post, *, water_fear=-1.0, shore_fear=0.0, refusal=None):
    def probe(p):
        n = 6
        k = round(p * n)
        return {"placements": [{"surfaced": i < k, "dirty": False} for i in range(n)], "p_surface": p}

    r = {"arm": arm, "seed": seed, "refusal": refusal, "water_fear": water_fear, "shore_fear": shore_fear}
    if refusal is None:
        r["pre"] = probe(pre)
        r["post"] = probe(post)
    return r


class TestVerdict:
    def test_earned_when_every_gate_holds(self):
        recs = [_seed("fear", s, 0.0, 1.0) for s in range(5)] + [
            _seed("ablated", s, 0.0, 0.0, water_fear=0.0) for s in range(5)
        ]
        v = H.compute_verdict(recs)
        assert v["verdict"] == "EARNED", v
        assert all(v["checks"].values())
        assert v["permutation"]["p_one_sided"] == pytest.approx(1 / 252)

    def test_null_when_fear_does_not_beat_its_own_pre(self):
        recs = [_seed("fear", s, 0.5, 0.5) for s in range(5)] + [
            _seed("ablated", s, 0.0, 0.0, water_fear=0.0) for s in range(5)
        ]
        v = H.compute_verdict(recs)
        assert v["verdict"] == "NULL" and not v["checks"]["fear_post_gt_pre_every_seed"]

    def test_null_when_ablated_surfaces_too(self):
        recs = [_seed("fear", s, 0.0, 1.0) for s in range(5)] + [
            _seed("ablated", s, 0.0, 0.8, water_fear=0.0) for s in range(5)
        ]
        v = H.compute_verdict(recs)
        assert v["verdict"] == "NULL" and not v["checks"]["ablated_post_median_le_max"]

    def test_incomplete_names_refusals_instead_of_dropping_them(self):
        recs = [_seed("fear", s, 0.0, 1.0) for s in range(4)] + [_seed("fear", 4, 0.0, 1.0, refusal="LIVE G2 FAILED")]
        recs += [_seed("ablated", s, 0.0, 0.0, water_fear=0.0) for s in range(5)]
        v = H.compute_verdict(recs)
        assert v["verdict"] == "INCOMPLETE" and v["refused"][0]["refusal"] == "LIVE G2 FAILED"
        assert v["n_clean"] == {"fear": 4, "ablated": 5}

    def test_incomplete_when_a_seed_has_no_clean_placements(self):
        recs = [_seed("fear", s, 0.0, 1.0) for s in range(5)] + [
            _seed("ablated", s, 0.0, 0.0, water_fear=0.0) for s in range(5)
        ]
        recs[0]["post"] = {"placements": [{"surfaced": True, "dirty": True}] * 6, "p_surface": None}
        assert H.compute_verdict(recs)["verdict"] == "INCOMPLETE"


class TestVerdictFolds:
    def test_duplicate_seed_rows_refuse_unless_a_run_is_selected(self):
        recs = [_seed("fear", s, 0.0, 1.0) for s in range(5)] + [
            _seed("ablated", s, 0.0, 0.0, water_fear=0.0) for s in range(5)
        ]
        for r in recs:
            r["run_id"] = "aaa"
        rerun = dict(_seed("fear", 0, 0.0, 1.0))
        rerun["run_id"] = "bbb"
        v = H.compute_verdict(recs + [rerun])
        assert v["verdict"] == "INCOMPLETE" and "fear/seed0×2" in v["duplicates"]
        assert H.compute_verdict(recs + [rerun], run_id="aaa")["verdict"] == "EARNED"

    def test_two_arms_are_two_run_ids_and_both_must_be_selected(self):
        # each `run` invocation mints its own id: the FEAR arm and the ABLATED arm of one trial
        # carry different ids, so the selection takes one id PER ARM (the live run-2 shape)
        recs = [_seed("fear", s, 0.0, 1.0) for s in range(5)] + [
            _seed("ablated", s, 0.0, 0.0, water_fear=0.0) for s in range(5)
        ]
        for r in recs:
            r["run_id"] = "fear-run" if r["arm"] == "fear" else "ablated-run"
        stale = [_seed("fear", s, 0.0, 0.0) for s in range(5)] + [
            _seed("ablated", s, 0.0, 0.0, water_fear=0.0) for s in range(5)
        ]
        for r in stale:
            r["run_id"] = "run-1"
        both = stale + recs
        assert H.compute_verdict(both)["verdict"] == "INCOMPLETE"  # duplicates unselected
        assert H.compute_verdict(both, run_id=["fear-run"])["verdict"] == "INCOMPLETE"  # one arm only
        assert H.compute_verdict(both, run_id=["fear-run", "ablated-run"])["verdict"] == "EARNED"

    def test_more_than_five_clean_seeds_is_incomplete_not_a_bigger_n(self):
        recs = [_seed("fear", s, 0.0, 1.0) for s in range(6)] + [
            _seed("ablated", s, 0.0, 0.0, water_fear=0.0) for s in range(5)
        ]
        assert H.compute_verdict(recs)["verdict"] == "INCOMPLETE"

    def test_shore_bleed_is_a_null_not_a_refusal(self):
        recs = [_seed("fear", s, 0.0, 1.0, water_fear=-1.0, shore_fear=-0.5) for s in range(5)]
        recs += [_seed("ablated", s, 0.0, 0.0, water_fear=0.0) for s in range(5)]
        v = H.compute_verdict(recs)
        assert v["verdict"] == "NULL" and not v["checks"]["fear_specificity_every_seed"]

    def test_min_pain_edge_reads_the_apparatus_record(self):
        rec = {
            "cycles": [
                {"w2_dive": {"t_pain_edge": 5.186}},
                {"w2_dive": {"t_pain_edge": 5.085}},
                {"w2_dive": {"t_pain_edge": 5.443}},
            ]
        }
        assert H.min_pain_edge_s(rec) == 5.085
        assert H.min_pain_edge_s({"cycles": []}) is None


class TestFingerprint:
    def test_order_and_float_noise_are_not_drift(self):
        frozen = {
            "cluster_fear_failure_modes": ["drive:health", "drive:oxygen"],
            "encoder_pattern_threshold": 0.85,
            "sensor_ranges": {"oxygen": [0.0, 40.0]},
        }
        live = {
            "cluster_fear_failure_modes": ["drive:oxygen", "drive:health"],
            "encoder_pattern_threshold": 0.8500000001,
            "sensor_ranges": {"oxygen": [0, 40]},
        }
        assert H.fingerprint_drift(live, frozen) == []

    def test_a_changed_range_or_missing_key_is_drift(self):
        frozen = {"a": 1, "sensor_ranges": {"saturation": [0.0, 20.0]}}
        assert H.fingerprint_drift({"a": 1, "sensor_ranges": {"saturation": [0.0, 10.0]}}, frozen) == ["sensor_ranges"]
        assert H.fingerprint_drift({"sensor_ranges": {"saturation": [0.0, 20.0]}}, frozen) == ["a"]


class TestFrozenContract:
    def test_usable_oxygen_sits_below_the_declared_comfort_band(self):
        fp = H.FROZEN["fingerprint"]["oxygen_drive"]
        assert H.FROZEN["usable_oxygen_max"] < fp["set_point"] - fp["comfort_band"]

    def test_usable_episode_requires_the_saturating_publish(self):
        # intensity = min(1, (|oxygen − 20| − 6) · 0.5): 1.0 first at oxygen 12; a 0.5-intensity
        # write converges fear to exactly −θ, which the loop's strict floor treats as dead
        fp = H.FROZEN["fingerprint"]["oxygen_drive"]
        intensity = min(1.0, (abs(H.FROZEN["usable_oxygen_max"] - fp["set_point"]) - fp["comfort_band"]) * 0.5)
        assert intensity >= H.FROZEN["usable_pain_intensity_min"] == 1.0

    def test_probe_cap_is_below_the_measured_pain_edge(self):
        rec = {"cycles": [{"w2_dive": {"t_pain_edge": 5.085}}]}
        cap = H.min_pain_edge_s(rec) - H.FROZEN["probe_cap_margin_s"]
        assert 0 < cap < 5.085 and H.FROZEN["probe_cap_margin_s"] >= 0.5

    def test_allowlist_and_ranges_match_the_shipped_substrate(self):
        import yaml

        from maxim.decisions.nac import NACConfig

        assert sorted(NACConfig().cluster_fear_failure_modes) == H.FROZEN["fingerprint"]["cluster_fear_failure_modes"]
        body = yaml.safe_load(
            (
                Path(__file__).resolve().parents[2] / "src/maxim/_data/components/bodies/minecraft_player.yaml"
            ).read_text()
        )

        def find(o):
            if isinstance(o, dict):
                if "sensors" in o:
                    return o["sensors"]
                for v in o.values():
                    r = find(v)
                    if r:
                        return r
            return None

        sensors = find(body)
        for name, rng in H.FROZEN["fingerprint"]["sensor_ranges"].items():
            assert [float(v) for v in sensors[name]["range"]] == rng, name
        oxy = sensors["oxygen"]["drive"]
        assert float(oxy["set_point"]) == H.FROZEN["fingerprint"]["oxygen_drive"]["set_point"]
        assert float(oxy["comfort_band"]) == H.FROZEN["fingerprint"]["oxygen_drive"]["comfort_band"]


class TestBridgeCadence:
    def test_median_interval(self):
        assert H.median_interval_s([0.0, 0.1, 0.21, 0.3]) == pytest.approx(0.1, abs=0.01)
        assert H.median_interval_s([0.0, 0.5, 1.0]) == pytest.approx(0.5)
        assert H.median_interval_s([0.0]) is None and H.median_interval_s([]) is None

    def test_frozen_cadence_bound_is_sensor_freshness(self):
        # the DV clock samples is_in_water at 4 Hz: the bridge's snapshot interval must not exceed
        # the sampling period, or a latency read inherits up to one stale interval
        assert H.FROZEN["bridge_state_interval_max_s"] <= 0.25

    def test_loop_liveness_contract(self):
        # 2 Hz proposal cadence → ~6 ticks in the 3 s window; 4 is the floor with margin
        assert H.FROZEN["loop_liveness_min_ticks"] == 4 and H.FROZEN["loop_liveness_s"] == 3.0
        assert H.FROZEN["loop_liveness_min_ticks"] <= H.FROZEN["loop_liveness_s"] / 0.5


class TestWindowTelemetry:
    def test_ticks_are_relative_to_the_first_tick_and_carry_the_proposal(self, tmp_path):
        import json as _json

        p = tmp_path / "t.jsonl"
        rows = [
            {
                "ts": 100.0,
                "step": 1,
                "proposal": None,
                "gated": False,
                "nac": {"active_clusters": {"world": "w1"}},
                "drives": {"threat": 0.0, "food": 0.2},
            },
            {
                "ts": 100.5,
                "step": 2,
                "proposal": {"tool_name": "minecraft_player_flee", "confidence": 0.7},
                "gated": False,
                "nac": {},
                "drives": {"threat": 1.0},
            },
        ]
        p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
        ticks = H._telemetry_ticks(p, t0_monotonic=0.0)
        assert [t["t_from_first_tick"] for t in ticks] == [0.0, 0.5]
        assert ticks[1]["proposal"] == "minecraft_player_flee" and ticks[0]["proposal"] is None
        assert ticks[0]["active_clusters"] == {"world": "w1"} and ticks[1]["drives"] == {"threat": 1.0}

    def test_missing_or_empty_file_is_no_ticks_not_a_crash(self, tmp_path):
        assert H._telemetry_ticks(tmp_path / "missing.jsonl", 0.0) == []
        (tmp_path / "empty.jsonl").write_text("")
        assert H._telemetry_ticks(tmp_path / "empty.jsonl", 0.0) == []


def test_verdict_over_the_committed_exp60_record_matches_the_committed_verdict() -> None:
    """The lift's proof, mechanical (Exp 61 architecture lens S11): `compute_verdict` over the
    committed Exp 60 trials with the two run ids must reproduce the committed verdict's checks and
    verdict — a refactor of the runner that moved a pure function cannot move a shipped verdict."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    rows = [
        json.loads(ln)
        for ln in (root / "docs/experiments/data/exp60_trials.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    committed = json.loads((root / "docs/experiments/data/exp60_verdict.json").read_text())
    v = H.compute_verdict(rows, run_id=committed["run_ids"])
    assert v["verdict"] == committed["verdict"] == "EARNED"
    assert v["checks"] == committed["checks"]
    assert v["n_clean"] == committed["n_clean"]
    assert v["permutation"] == committed["permutation"]


# ── --apparatus-record ───────────────────────────────────────────────────


_TS = 1_791_000_000.25  # the fixture check's own epoch ts (the anchor's measured block carries the same)


def _apparatus(ts: float = _TS, **over) -> dict:
    """A stamped, real, passing water check (M1b PR 5a-2) whose earliest pain edge is 5.1 s."""
    rec = {
        "ts": ts,
        "record_kind": "instrument_check",
        "status": "ok",
        "pass": True,
        "mock": False,
        "all_pass": True,
        "cycles": [
            {"w2_dive": {"t_pain_edge": 5.3, "t_damage_onset": 16.1}},
            {"w2_dive": {"t_pain_edge": 5.1, "t_damage_onset": 16.0}},
        ],
    }
    rec.update(over)
    return rec


def _measured(ts: float = _TS, edge: float | None = 5.1) -> dict:
    return {"t_damage_onset_min_s": 16.0, "t_damage_onset_max_s": 16.1, "t_pain_edge_min_s": edge, "ts": ts}


def _earned_rows(fear_rec: dict, abl_rec: dict) -> list:
    return [dict(_seed("fear", s, 0.0, 1.0), **fear_rec) for s in range(5)] + [
        dict(_seed("ablated", s, 0.0, 0.0, water_fear=0.0), **abl_rec) for s in range(5)
    ]


class TestOneApparatusRecordPerVerdict:
    """Owner decision 2026-10-10: a verdict over rows naming more than one apparatus record fails closed."""

    A = {"apparatus_record": "a/water_apparatus.json", "apparatus_record_sha256": "aa", "apparatus_record_ts": 1.0}
    B = {"apparatus_record": "b/water_apparatus.json", "apparatus_record_sha256": "bb", "apparatus_record_ts": 2.0}
    LEGACY = {"apparatus_record_ts": 1.0}

    def test_one_record_across_both_arms_passes(self):
        v = H.compute_verdict(_earned_rows(self.A, self.A))
        assert v["verdict"] == "EARNED" and "incomplete_cause" not in v
        assert v["apparatus_records"] == [{"path": "a/water_apparatus.json", "sha256": "aa", "ts": 1.0}]

    @pytest.mark.parametrize(
        "fear, abl",
        [
            (A, B),  # FEAR and ABLATED under two records
            (A, {**A, "apparatus_record_sha256": "a2"}),  # same path, different bytes
            (A, LEGACY),  # a named record and a legacy row that cannot say which bytes it read
            (A, {**A, "apparatus_record_sha256": None}),  # a half-named row
            (LEGACY, {"apparatus_record_ts": 9.0}),  # legacy rows from two different checks
        ],
    )
    def test_a_mixed_record_pair_is_incomplete_with_a_named_cause(self, fear, abl):
        v = H.compute_verdict(_earned_rows(fear, abl))
        assert v["verdict"] == "INCOMPLETE" and v["incomplete_cause"] == "mixed_apparatus_records", v
        assert "apparatus records" in v["reason"] and len(v["apparatus_records"]) == 2

    def test_a_refused_row_under_another_record_still_counts(self):
        rows = _earned_rows(self.A, self.A) + [dict(_seed("fear", 9, 0, 0, refusal="x"), **self.B)]
        assert H.compute_verdict(rows)["incomplete_cause"] == "mixed_apparatus_records"

    def test_run_id_cannot_select_around_a_killed_run_under_another_record(self):
        """Architecture S1: identities are judged over the WHOLE file, before `--run-id` selects rows."""
        killed = dict(_seed("fear", 11, 0, 0, refusal="Ctrl-C"), run_id="killed", **self.A)
        rows = [dict(r, run_id="f" if r["arm"] == "fear" else "a") for r in _earned_rows(self.B, self.B)]
        assert H.compute_verdict(rows, run_id=["f", "a"])["verdict"] == "EARNED"
        v = H.compute_verdict([killed] + rows, run_id=["f", "a"])
        assert v["verdict"] == "INCOMPLETE" and v["incomplete_cause"] == "mixed_apparatus_records"

    def test_every_incomplete_names_a_machine_readable_cause(self):
        dup = _earned_rows(self.A, self.A) + [dict(_seed("fear", 0, 0.0, 1.0), **self.A)]
        assert H.compute_verdict(dup)["incomplete_cause"] == "duplicate_arm_seed_rows"
        assert H.compute_verdict(_earned_rows(self.A, self.A)[1:])["incomplete_cause"] == "too_few_clean_seeds"
        blank = _earned_rows(self.A, self.A)
        blank[0]["post"] = {"placements": [], "p_surface": None}
        assert H.compute_verdict(blank)["incomplete_cause"] == "no_clean_placements"

    def test_legacy_rows_from_one_check_still_pass(self):
        assert H.compute_verdict(_earned_rows(self.LEGACY, self.LEGACY))["verdict"] == "EARNED"


class TestApparatusRecordPure:
    def test_the_anchor_must_carry_the_same_checks_measured_block(self):
        assert H.anchor_record_mismatch(_measured(), _apparatus()) is None
        assert "ts" in H.anchor_record_mismatch(_measured(ts=_TS + 1), _apparatus())
        assert "pain edge" in H.anchor_record_mismatch(_measured(edge=5.3), _apparatus())
        assert "no measured block" in H.anchor_record_mismatch(None, _apparatus())
        assert "no epoch ts" in H.anchor_record_mismatch(_measured(), _apparatus(ts=None))
        assert "no epoch ts" in H.anchor_record_mismatch({**_measured(), "ts": True}, _apparatus(ts=True))

    def test_the_verdict_lists_each_distinct_record_its_rows_ran_under(self):
        a = {"apparatus_record": "a.json", "apparatus_record_sha256": "1", "apparatus_record_ts": 1.0}
        legacy = {"apparatus_record_ts": 2.0}  # a row from before the flag: path and digest unknown, never invented
        assert H.apparatus_records_used([a, dict(a), legacy]) == [
            {"path": "a.json", "sha256": "1", "ts": 1.0},
            {"path": None, "sha256": None, "ts": 2.0},
        ]
        v = H.compute_verdict([dict(_seed("fear", 0, 0.0, 1.0), **a)])
        assert v["verdict"] == "INCOMPLETE" and v["apparatus_records"] == [{"path": "a.json", "sha256": "1", "ts": 1.0}]

    def test_load_reads_once_and_stamps_the_path_it_read(self, tmp_path):
        p = tmp_path / "water_apparatus.json"
        p.write_bytes(json.dumps(_apparatus()).encode())
        got = H.load_apparatus_record(str(p))
        assert got["record"] == _apparatus() and got["path"] == str(p.resolve())
        assert got["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
        inside = H.load_apparatus_record(H.GATE_RECORD)  # relative = under the repo root, stamped repo-relative
        assert inside["path"] == H.GATE_RECORD
        for bad, why in ((tmp_path / "absent.json", "cannot be read"), (tmp_path, "cannot be read")):
            with pytest.raises(H.ApparatusRecordError, match=why):
                H.load_apparatus_record(str(bad))
        (tmp_path / "x.json").write_text("not json")
        with pytest.raises(H.ApparatusRecordError, match="not JSON"):
            H.load_apparatus_record(str(tmp_path / "x.json"))
        (tmp_path / "l.json").write_text("[1]")
        with pytest.raises(H.ApparatusRecordError, match="not a JSON object"):
            H.load_apparatus_record(str(tmp_path / "l.json"))

    def test_the_default_is_the_constant_exp61_run_and_r3_pilot_import(self, monkeypatch):
        """The flag's default is the unchanged module constant, which `exp61_run` and `r3_pilot` import by name;
        `r3_run` has its own (dated) constant and is not an importer."""
        from survival_world import exp61_run, r3_pilot, r3_run

        captured: dict = {}
        monkeypatch.setattr(H, "_run", lambda args: captured.update(vars(args)) or 0)
        assert H.main(["run", "--arm", "fear", "--rcon-password", "p", "--out", "x.jsonl"]) == 0
        assert captured["apparatus_record"] == H.APPARATUS_RECORD
        assert H.APPARATUS_RECORD == "docs/experiments/data/exp60_water_apparatus.json"
        assert exp61_run.APPARATUS_RECORD is H.APPARATUS_RECORD and r3_pilot.APPARATUS_RECORD is H.APPARATUS_RECORD
        assert r3_run.APPARATUS_RECORD != H.APPARATUS_RECORD

    def test_the_binding_holds_against_the_real_water_check_stamping(self, tmp_path):
        """Architecture S4: the anchor binding pinned against the REAL producers, not hand-written fixtures —
        `stamp_instrument_check`, `all_cycles_pass` and `_stamp_measured` / `measured_edges` are the water check's
        own functions; `_finish`'s write is a closure inside `main`, so its one line (`json.dumps(report, indent=2)`)
        is copied here by hand. Each record is read back through JSON as the harness reads it."""
        import time

        from survival_world import exp60_water_check as chk

        report = {
            "ts": time.time(),  # a real epoch float: the round trip must keep it exact
            "cycles": [
                {
                    "w1_shore": {"distance_from_spawn": 31.7},
                    "w2_dive": {"t_damage_onset": 16.06 + i / 7, "t_pain_edge": 5.0 + i / 3},
                    "w4_escape": {"t_surface": 1.2 + i / 11, "t_sinkback": None},
                    "pass": True,
                }
                for i in range(3)
            ],
            "instrument_error": None,
        }
        report["all_pass"] = chk.all_cycles_pass(report["cycles"], 3)
        report["frozen_params"] = True
        chk.stamp_instrument_check(report, mock=False, passed=report["all_pass"] and report["frozen_params"])
        out = tmp_path / "water_apparatus.json"
        out.write_text(json.dumps(report, indent=2))  # a hand copy of `_finish`'s write (a closure in `main`)
        anchor = tmp_path / "anchor.json"
        anchor.write_text(json.dumps({"shore": [0, 64, 0], "submerged": [0, 60, 5]}))
        chk._stamp_measured(report, out, anchor_file=anchor)
        apparatus = json.loads(out.read_text())
        measured = json.loads(anchor.read_text())["measured"]
        assert H.instrument_check_authorizes(apparatus) is None
        assert H.anchor_record_mismatch(measured, apparatus) is None
        assert measured["t_pain_edge_min_s"] == H.min_pain_edge_s(apparatus) == 5.0
        # and the binding is not vacuous on the same producers: a later check's stamp is refused
        report2 = dict(report, ts=report["ts"] + 60.0)
        chk._stamp_measured(report2, out, anchor_file=anchor)
        assert "ts" in H.anchor_record_mismatch(json.loads(anchor.read_text())["measured"], apparatus)


def _write_inputs(tmp_path: Path, *, apparatus: dict, measured: dict | None, geom: dict | None = None) -> dict:
    ap = tmp_path / "rerun" / "water_apparatus.json"
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text(json.dumps(apparatus))
    gate = tmp_path / "geometry.json"
    gate.write_text(json.dumps({"run_gate": {"pass": True}, "code_hash": "offline"}))
    anchor = tmp_path / "anchor.json"
    rec = dict(geom or {"shore": [10.0, 64.0, 10.0], "submerged": [10.0, 60.0, 20.0]})
    if measured is not None:
        rec["measured"] = measured
    anchor.write_text(json.dumps(rec))
    return {"apparatus": ap, "gate": gate, "anchor": anchor, "out": tmp_path / "out" / "trials.jsonl"}


def _run_argv(paths: dict, arm: str = "fear", *, port: int = 1, seeds: int = 1) -> list[str]:
    return [
        "run",
        "--arm",
        arm,
        "--apparatus-record",
        str(paths["apparatus"]),
        "--gate-record",
        str(paths["gate"]),
        "--rcon-password",
        "p",
        "--bridge-port",
        str(port),
        "--seeds",
        str(seeds),
        "--out",
        str(paths["out"]),
    ]


class TestApparatusRecordRefusals:
    """Each refusal ends `run` with exit 3 BEFORE any bridge connection or row (no fallback to the default)."""

    @pytest.fixture(autouse=True)
    def _no_bridge(self, monkeypatch):
        def _never(*a, **k):
            raise AssertionError("a refused apparatus record must stop the run before any RCON/bridge connection")

        monkeypatch.setattr(H.C, "RconControl", _never)

    def test_the_named_record_is_the_one_authorized(self, tmp_path, monkeypatch, capsys):
        paths = _write_inputs(tmp_path, apparatus=_apparatus(), measured=_measured())
        monkeypatch.setattr(H, "ANCHOR_FILE", paths["anchor"])
        seen: list[dict] = []

        def _spy(record: dict) -> str:
            seen.append(record)
            return "spy: stop here"

        monkeypatch.setattr(H, "instrument_check_authorizes", _spy)
        assert H.main(_run_argv(paths)) == 3
        assert seen == [_apparatus()], "the authorization read the default (or nothing), not --apparatus-record"
        assert str(paths["apparatus"].resolve()) in capsys.readouterr().out
        assert not paths["out"].exists()

    def test_a_missing_record_fails_loudly_and_never_falls_back(self, tmp_path, monkeypatch, capsys):
        paths = _write_inputs(tmp_path, apparatus=_apparatus(), measured=_measured())
        monkeypatch.setattr(H, "ANCHOR_FILE", paths["anchor"])
        paths["apparatus"].unlink()
        assert H.main(_run_argv(paths)) == 3
        out = capsys.readouterr().out
        assert "cannot be read" in out and str(paths["apparatus"]) in out and H.APPARATUS_RECORD not in out
        assert not paths["out"].exists()

    @pytest.mark.parametrize(
        "over, why",
        [
            ({"record_kind": None}, "not a stamped instrument check"),
            ({"mock": True}, "a mock"),
            ({"status": "failed"}, "ended 'failed'"),
            ({"pass": False}, "did not pass"),
        ],
    )
    def test_an_unauthorized_record_is_refused(self, tmp_path, monkeypatch, capsys, over, why):
        paths = _write_inputs(tmp_path, apparatus=_apparatus(**over), measured=_measured())
        monkeypatch.setattr(H, "ANCHOR_FILE", paths["anchor"])
        assert H.main(_run_argv(paths)) == 3
        out = capsys.readouterr().out
        assert why in out and "does not authorize a run" in out and str(paths["apparatus"].resolve()) in out
        assert not paths["out"].exists()

    def test_a_record_with_no_pain_edge_is_refused(self, tmp_path, monkeypatch, capsys):
        paths = _write_inputs(tmp_path, apparatus=_apparatus(cycles=[]), measured=_measured(edge=None))
        monkeypatch.setattr(H, "ANCHOR_FILE", paths["anchor"])
        assert H.main(_run_argv(paths)) == 3
        assert "no measured t_pain_edge" in capsys.readouterr().out and not paths["out"].exists()

    def test_a_record_that_is_not_the_anchors_check_is_refused(self, tmp_path, monkeypatch, capsys):
        paths = _write_inputs(tmp_path, apparatus=_apparatus(), measured=_measured(ts=_TS - 86400.0))
        monkeypatch.setattr(H, "ANCHOR_FILE", paths["anchor"])
        assert H.main(_run_argv(paths)) == 3
        assert "disagree" in capsys.readouterr().out and not paths["out"].exists()


class _ReachedTheBridge(Exception):
    """Raised by the RCON stub: the run passed every apparatus preflight."""


def _git(repo: Path, *args: str) -> str:
    cfg = ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    return subprocess.run(["git", *cfg, *args], cwd=repo, check=True, capture_output=True, text=True).stdout


class TestCommittedRecordOnAGatedRun:
    """Review fold (Executor SF-1, Architecture S2, Wire SF-1): rows into docs/experiments/data/ need the record
    committed on main byte for byte — tracked, equal to its blob at HEAD, HEAD an ancestor of origin/main."""

    REL = "docs/experiments/data/rerun_exp60_x/water_apparatus.json"

    @pytest.fixture
    def repo(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        (repo / self.REL).parent.mkdir(parents=True)
        (repo / self.REL).write_text(json.dumps(_apparatus()))
        _git(repo, "init", "-q")
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", "record")
        _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")  # no remote: the fetch fails, the ref stands
        monkeypatch.setattr(H.C, "REPO_ROOT", repo)
        # the code-provenance doors are the harness's own tests' business; here the tree under test is a fixture
        monkeypatch.setattr(H, "in_process_code_provenance", lambda *a, **k: {"code_tree_sha256": "fixture"})
        monkeypatch.setattr(H, "evidence_out_paths_or_exit", lambda root, ps, **k: [Path(p).resolve() for p in ps])

        def _reached(*a, **k):
            raise _ReachedTheBridge

        monkeypatch.setattr(H.C, "RconControl", _reached)
        anchor = tmp_path / "anchor.json"
        anchor.write_text(json.dumps({"shore": [0, 64, 0], "submerged": [0, 60, 5], "measured": _measured()}))
        monkeypatch.setattr(H, "ANCHOR_FILE", anchor)
        gate = tmp_path / "geometry.json"
        gate.write_text(json.dumps({"run_gate": {"pass": True}}))
        return {"root": repo, "record": repo / self.REL, "gate": gate, "tmp": tmp_path}

    def _argv(self, repo, record, *, gated=True, extra=()):
        out = repo["root"] / "docs/experiments/data/rerun_exp60_x/trials.jsonl" if gated else repo["tmp"] / "t.jsonl"
        paths = {"apparatus": record, "gate": repo["gate"], "out": out}
        return _run_argv(paths) + list(extra), out

    def test_the_committed_record_on_main_passes(self, repo):
        argv, _ = self._argv(repo, self.REL)
        with pytest.raises(_ReachedTheBridge):
            H.main(argv)

    def _refused(self, repo, record, why, capsys, **kw):
        argv, out = self._argv(repo, record, **kw)
        assert H.main(argv) == 3  # the RCON stub would have raised: never connected
        text = capsys.readouterr().out
        assert why in text and "committed record on main" in text, text
        assert not out.exists()

    def test_an_untracked_record_is_refused(self, repo, capsys):
        other = repo["record"].with_name("untracked.json")
        other.write_text(json.dumps(_apparatus()))
        self._refused(repo, str(other), "not tracked", capsys)

    def test_a_modified_tracked_record_is_refused(self, repo, capsys):
        repo["record"].write_text(json.dumps(_apparatus(), indent=1))  # same record, other bytes
        self._refused(repo, self.REL, "differs from its blob at HEAD", capsys)

    def test_a_record_outside_the_repo_is_refused(self, repo, capsys):
        outside = repo["tmp"] / "outside.json"
        outside.write_text(json.dumps(_apparatus()))
        self._refused(repo, str(outside), "outside the repo", capsys)

    def test_a_head_not_on_main_is_refused(self, repo, capsys):
        (repo["root"] / "later.txt").write_text("x")
        _git(repo["root"], "add", "later.txt")
        _git(repo["root"], "commit", "-q", "-m", "not on main")
        self._refused(repo, self.REL, "not an ancestor of origin/main", capsys)

    def test_a_missing_origin_main_says_so(self, repo, capsys):
        _git(repo["root"], "update-ref", "-d", "refs/remotes/origin/main")
        self._refused(repo, self.REL, "origin/main is not a valid ref (fetch it)", capsys)

    def test_a_non_gated_out_is_unaffected(self, repo):
        other = repo["record"].with_name("untracked.json")
        other.write_text(json.dumps(_apparatus()))
        argv, _ = self._argv(repo, str(other), gated=False)
        with pytest.raises(_ReachedTheBridge):
            H.main(argv)

    @pytest.mark.parametrize("waived", [True, False])
    def test_a_waived_record_reaches_the_ledger_gate_as_an_allowance(self, repo, monkeypatch, waived):
        """Delta-review SHOULD-FIX: on a CLEAN tree, `--allow-dirty` past an uncommitted record must stamp the row's
        `provenance.allow_dirty` — the waiver `_evidence_records.judge_provenance` reads (allowed-dirty, never sole
        support) — as well as `apparatus_record_check`, which the gate also reads since #1220 and fails when the two
        disagree. The live pieces are mocks; the seed
        refuses at its first step so a real row is written through `_write_row`."""
        import types
        from unittest import mock

        import _evidence_records as ER
        import maxim.simulation.minecraft as mc
        import maxim.simulation.minecraft_harness as mh
        from survival_world import common as swc

        clean = {"executed_git_hash": "a" * 40, "code_tree_sha256": "b" * 64, "working_tree_dirty_src_scripts": False}
        monkeypatch.setattr(H, "in_process_code_provenance", lambda *a, **k: dict(clean, harness_family="in_process"))
        monkeypatch.setattr(H.C, "RconControl", mock.MagicMock())
        for mod, name in ((mc, "MinecraftClient"), (mh, "build_minecraft_aut"), (mh, "MinecraftSyncPump")):
            monkeypatch.setattr(mod, name, mock.MagicMock())
        monkeypatch.setattr(swc, "make_fresh_encoder", mock.MagicMock())
        trial = mock.MagicMock()
        trial.check_fingerprint.side_effect = H.Refusal("offline stop after the preflights under test")
        monkeypatch.setattr(H, "WaterTrial", mock.MagicMock(return_value=trial))
        record = self.REL
        if waived:
            record = str(repo["record"].with_name("untracked.json"))
            Path(record).write_text(json.dumps(_apparatus()))
        argv, out = self._argv(repo, record, extra=["--allow-dirty"] if waived else [])
        assert H.main(argv) == 4  # the seed refused (by the mock), after a row was written
        (row,) = [json.loads(ln) for ln in out.read_text().splitlines()]
        assert row["provenance"]["working_tree_dirty_src_scripts"] is False
        assert row["provenance"].get("allow_dirty") is (True if waived else None)
        assert (row["apparatus_record_check"]["problem"] is not None) is waived
        ctx = types.SimpleNamespace(repo=types.SimpleNamespace(is_ancestor=lambda a, b: True), base="base")
        j = ER.Judgement(path=str(out))
        ER.judge_provenance(row["provenance"], j, ctx, "row")
        assert j.reasons == [] and j.allowed_dirty is waived, j.reasons  # the judge's provenance rule, unfailed

    def test_allow_dirty_passes_it_and_says_so(self, repo, capsys):
        other = repo["record"].with_name("untracked.json")
        other.write_text(json.dumps(_apparatus()))
        argv, _ = self._argv(repo, str(other), extra=["--allow-dirty"])
        with pytest.raises(_ReachedTheBridge):
            H.main(argv)
        assert "--allow-dirty: apparatus record" in capsys.readouterr().out


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_offline_run_both_arms_then_verdict_stamp_the_named_apparatus_record(tmp_path: Path, monkeypatch) -> None:
    """`exp60_run` end to end OFFLINE through `main()`: both arms on the scripted water bridge with the REAL row
    and verdict writers, the apparatus record passed by `--apparatus-record` (the R3 pilot died twice on the rig
    on bugs a hand-typed stub hid; this harness had no such run since the M1b writers and #783 changed it)."""
    from survival_world import water_trial as WT
    from survival_world.scripted_water import ScriptedWaterBridge, ScriptedWaterControl
    from survival_world.setup_world import water_anchor_record, water_classroom_geometry

    geom = water_anchor_record(water_classroom_geometry(10, 20, depth=5, shore_y=64))
    paths = _write_inputs(tmp_path, apparatus=_apparatus(), measured=_measured(), geom=geom)
    xyz = {
        k: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
        for k, v in geom.items()
        if k in ("shore", "submerged")
    }
    srv = ScriptedWaterBridge(shore=xyz["shore"], submerged=xyz["submerged"], damage_onset_s=16.0, damage_per_s=2.0)
    rcon = ScriptedWaterControl(srv, gamerules={r: v for r, v in WT.GAMERULES})
    monkeypatch.setattr(H, "ANCHOR_FILE", paths["anchor"])
    monkeypatch.setattr(H.C, "RconControl", lambda *a, **k: rcon)
    # the frozen design, shortened so two seeds run in minutes; every check keeps its shape
    for key, value in (
        ("K_usable_episodes", 2),
        ("placements_per_probe", 1),
        ("shore_roam_s", 1.0),
        ("loop_liveness_s", 2.0),
        ("loop_liveness_min_ticks", 3),
        ("loop_warm_s", 0.5),
    ):
        monkeypatch.setitem(H.FROZEN, key, value)
    try:
        codes = {arm: H.main(_run_argv(paths, arm, port=srv.port)) for arm in ("fear", "ablated")}
    finally:
        srv.close()
    assert paths["out"].is_file(), f"no rows written (exit codes {codes}): was the named record refused?"
    rows = [json.loads(ln) for ln in paths["out"].read_text().splitlines() if ln.strip()]
    assert [(r["arm"], r["seed"]) for r in rows] == [("fear", 11), ("ablated", 11)]
    digest = hashlib.sha256(paths["apparatus"].read_bytes()).hexdigest()
    for r in rows:
        # M1b row stamps, the in-process provenance block, and the record this run read (not only its ts)
        assert (r["record_kind"], r["mock"]) == ("harness_row", False)
        prov = r["provenance"]
        assert prov["harness_family"] == "in_process" and prov["executed_maxim_file"].endswith("maxim/__init__.py")
        assert {
            "executed_git_hash",
            "code_tree_sha256",
            "working_tree_dirty_src_scripts",
            "python",
            "pythonpath",
        } <= set(prov)
        assert r["apparatus_record"] == str(paths["apparatus"].resolve()), "the run did not read --apparatus-record"
        assert r["apparatus_record_sha256"] == digest and r["apparatus_record_ts"] == _TS
        assert r["apparatus_record_check"] == {"gated": False, "problem": None, "allow_dirty": False}
        assert r["anchor_measured"] == _measured() and r["probe_cap_s"] == pytest.approx(5.1 - 0.75)
        assert r["train_cap_s"] == pytest.approx(15.0) and r["pain_edge_min_s"] == 5.1
    assert len({r["run_id"] for r in rows}) == 2 and len({r["provenance"]["code_tree_sha256"] for r in rows}) == 1
    fear, abl = rows
    # each arm ran its whole seed: no refusal, every DV present
    assert fear["refusal"] is None and abl["refusal"] is None, (fear["refusal"], abl["refusal"])
    assert codes == {"fear": 0, "ablated": 0}
    assert fear["status"] == abl["status"] == "ok" and abl["detached_subscribers"] == 1
    assert fear["training"]["usable_episodes"] >= 1 and fear["water_fear"] < 0 and abl["water_fear"] == 0.0
    assert fear["post"]["p_surface"] == 1.0 and abl["post"]["p_surface"] == 0.0, (fear["post"], abl["post"])

    vpath = tmp_path / "verdict.json"
    assert H.main(["verdict", "--data", str(paths["out"]), "--json", str(vpath)]) == 0
    v = json.loads(vpath.read_text())
    assert (v["record_kind"], v["kind"], v["verdict"]) == ("verdict", "exp60_verdict", "INCOMPLETE")  # 1 seed/arm
    assert v["scope"] == {"all_rows": True} and v["mock"] is False  # no mock mode: the fakes are injected here
    assert v["data_sha256"] == hashlib.sha256(paths["out"].read_bytes()).hexdigest()
    assert v["provenance"]["harness_family"] == "in_process"
    assert v["apparatus_records"] == [{"path": str(paths["apparatus"].resolve()), "sha256": digest, "ts": _TS}]
