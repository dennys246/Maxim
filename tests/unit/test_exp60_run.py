"""Exp 60 trial harness — the pure halves: placement classification, the verdict, the
permutation test, and the fingerprint comparator (scripts/survival_world/exp60_run.py).

The live `run` half is operator-run; these pin the parts a refactor could silently flip:
a DIRTY placement (damage inside the window) can never count as a surface, a seed with
no clean placements makes the verdict INCOMPLETE (never a silent drop), the gates read
the way the prereg states them, and 5 v 5 exact permutation bottoms out at 1/252.
"""

from __future__ import annotations

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
