"""Offline verdict logic of the L11 geometry probe (Slice 1).

The probe's `analyze` is a pure, network-free diagnosis over a captured trace
(scripts/survival_world/l11_geometry_probe.py). These tests pin its four verdict
branches on synthetic traces shaped like the real cases, so a refactor can't
silently flip "authorizes NO build" or mis-route the gain-silenced diagnosis the
substrate-faithful review lens predicted for Exp 58. The live `capture` half is
operator-run and not exercised here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO / "src", _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from survival_world.l11_geometry_probe import main  # noqa: E402


def _write_trace(tmp_path: Path, safe_state_fn, dark_state_fn, ranges, n=20) -> Path:
    recs = [{"kind": "provenance", "code_hash": "test", "world_ranges": ranges, "world_sensor_count": len(ranges)}]
    for _ in range(n):
        recs.append({"kind": "sample", "situation": "safe", "state": safe_state_fn()})
    for _ in range(n):
        recs.append({"kind": "sample", "situation": "dark", "state": dark_state_fn()})
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def _run(tmp_path: Path, trace: Path) -> dict:
    out = tmp_path / "diag.json"
    rc = main(["analyze", "--trace", str(trace), "--json", str(out)])
    assert rc == 0
    return json.loads(out.read_text())


def _ranges(n_filler=14):
    r = {"y_altitude": [0, 128], "nearest_hostile_dist": [0, 32], "light_level": [0, 15]}
    for i in range(n_filler):
        r[f"filler{i}"] = [0, 1]
    return r


def test_verdict_absent_when_no_sensor_moves(tmp_path):
    """Identical safe/dark states → contrast is not in the sensors at all."""
    ranges = _ranges()

    def st():
        s = {"y_altitude": 40, "nearest_hostile_dist": 16, "light_level": 7}
        for i in range(14):
            s[f"filler{i}"] = 0.5
        return s

    rec = _run(tmp_path, _write_trace(tmp_path, st, st, ranges))
    assert rec["verdict"] == "absent"
    assert rec["authorizes_build"] is False
    assert rec["movers"] == []


def test_verdict_gain_silenced_when_movers_rest_near_neutral(tmp_path):
    """A sensor that moves but stays hard against the A4 neutral 0.5 is muzzled.

    y_altitude 60→68 of [0,128] straddles the 64 midpoint (v≈0.47→0.53): it moves
    ≥ MOVE_EPS but its gain weight (|v-0.5|*2)**3 stays below GAIN_MASS_EPS at both
    ends — the substrate lens's predicted failure mode.
    """
    ranges = _ranges()

    def _mk(y):
        def st():
            s = {"y_altitude": y, "nearest_hostile_dist": 16, "light_level": 7}
            for i in range(14):
                s[f"filler{i}"] = 0.5
            return s

        return st

    rec = _run(tmp_path, _write_trace(tmp_path, _mk(60), _mk(68), ranges))
    assert rec["verdict"] == "gain_silenced"
    assert rec["authorizes_build"] is False
    assert "y_altitude" in rec["moved_but_silenced"]
    assert rec["live_contributors"] == []


def test_verdict_separable_still_refuses_build(tmp_path):
    """A big mover that carries gain mass separates offline — but Slice 1 never
    authorizes a build; the live re-encode (Slice 2) is the sole gate."""
    ranges = _ranges()

    def _mk(hd):
        def st():
            s = {"y_altitude": 40, "nearest_hostile_dist": hd, "light_level": 7}
            for i in range(14):
                s[f"filler{i}"] = 0.5
            return s

        return st

    rec = _run(tmp_path, _write_trace(tmp_path, _mk(30), _mk(1), ranges))
    assert rec["verdict"] in ("separable_here", "diluted_present")
    assert rec["authorizes_build"] is False
    assert "nearest_hostile_dist" in rec["live_contributors"]


def test_record_carries_faithful_provenance(tmp_path):
    """The decision record stamps the real substrate config, not invented values."""
    ranges = _ranges()

    def st():
        s = {"y_altitude": 40, "nearest_hostile_dist": 16, "light_level": 7}
        for i in range(14):
            s[f"filler{i}"] = 0.5
        return s

    rec = _run(tmp_path, _write_trace(tmp_path, st, st, ranges))
    # world is a gained modality at exponent 3.0, threshold 0.85 (the shipped config)
    assert rec["provenance"]["gain_modality"] is True
    assert rec["provenance"]["gain_exponent"] == 3.0
    assert rec["provenance"]["pattern_threshold"] == 0.85
    assert rec["cosine"]["threshold"] == 0.85
    assert rec["cluster_ids_offline_fresh_ec"]["distinct"] is False  # identical states


def test_analyze_refuses_trace_without_ranges(tmp_path):
    """No world_ranges provenance → refuse, never measure a guessed sensor set."""
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"kind": "sample", "situation": "safe", "state": {"y_altitude": 40}}) + "\n")
    with pytest.raises(SystemExit):
        main(["analyze", "--trace", str(p), "--json", str(tmp_path / "x.json")])


# ───────────────────── Exp 60 chunk (ii): generalized probe + run gate ─────────────────────

from survival_world import l11_geometry_probe as probe  # noqa: E402


def _exp60_anchor(measured=True):
    a = {
        "probe_situations": {"shore": [99, 40, -40], "submerged": [104, 35, -40]},
        "probe_settle": {"shore": {"is_in_water": 0, "on_ground": 1}, "submerged": {"is_in_water": 1}},
        "probe_rescue": {"submerged": "shore"},
    }
    if measured:
        a["measured"] = {"t_damage_onset_min_s": 16.25}
    return a


class TestSituationsFromAnchor:
    def test_exp58_shape_keeps_safe_dark_and_altitude_settle(self):
        plan = probe.situations_from_anchor({"anchor": [10, 40, 5], "dark": [30, 28, 5], "mid_y": 34})
        assert plan["labels"] == ["safe", "dark"]
        assert plan["settle"] == {"safe": {"y_altitude": 40.0}, "dark": {"y_altitude": 28.0}}
        assert plan["rescue"] == {} and plan["dive_budget_s"] is None and plan["mid_y"] == 34.0

    def test_exp60_shape_budgets_dives_from_the_measured_onset(self):
        plan = probe.situations_from_anchor(_exp60_anchor())
        assert plan["labels"] == ["shore", "submerged"]  # baseline first
        assert plan["positions"]["submerged"] == {"x": 104.0, "y": 35.0, "z": -40.0}
        assert plan["settle"]["submerged"] == {"is_in_water": 1}
        assert plan["rescue"] == {"submerged": "shore"}
        assert plan["dive_budget_s"] == pytest.approx(16.25 - probe.DIVE_MARGIN_S)

    def test_exp60_shape_without_measured_has_no_budget(self):
        plan = probe.situations_from_anchor(_exp60_anchor(measured=False))
        assert plan["dive_budget_s"] is None and plan["measured"] is None

    def test_exactly_two_situations(self):
        with pytest.raises(SystemExit):
            probe.situations_from_anchor({"probe_situations": {"a": [0, 0, 0]}})


class TestSettlePredicate:
    def test_altitude_clamps_to_the_body_range(self):
        ok = probe.settle_predicate({"y_altitude": 151}, {"y_altitude": (0, 128)})
        assert ok({"y_altitude": 128.0})  # sensed cap (docs/wiring/sensor-range-clamps.md)
        assert not ok({"y_altitude": 120.0})

    def test_min_rule_settles_at_or_above_the_bar(self):
        ok = probe.settle_predicate({"oxygen": {"min": 19.0}}, {})
        assert ok({"oxygen": 19.0}) and ok({"oxygen": 20.0}) and not ok({"oxygen": 18.0})

    def test_rescue_and_staleness_bars_match_the_apparatus_check(self):
        # the probe must never demand MORE than the bar the apparatus check PASSED at
        from survival_world import exp60_water_check as chk

        assert probe.RESCUE_OXYGEN_MIN == chk.RECOVER_OXYGEN_MIN
        assert probe.DIVE_SETTLE_S == chk.IN_WATER_WITHIN_S
        assert probe.STALE_STATE_S == chk.STALE_STATE_S
        assert probe.STALE_MAX_CONSECUTIVE == chk.STALE_MAX_CONSECUTIVE

    def test_binary_flags_settle_on_their_value_and_absent_sensor_never_settles(self):
        ok = probe.settle_predicate({"is_in_water": 1, "on_ground": 0}, {})
        assert ok({"is_in_water": 1.0, "on_ground": 0.0})
        assert not ok({"is_in_water": 0.0, "on_ground": 0.0})
        assert not ok({"is_in_water": 1.0})  # on_ground missing from the snapshot
        assert probe.settle_predicate({}, {})({})


class TestEarlyLateBins:
    def test_non_dive_rows_return_none(self):
        assert probe.early_late_bins([{"y_altitude": 1.0}], ["c1"]) is None

    def test_same_cluster_iff_early_and_late_id_sets_are_equal(self):
        rows = [{"oxygen": 20.0}, {"oxygen": 18.0}, {"oxygen": 12.0}, {"oxygen": 6.0}]
        b = probe.early_late_bins(rows, ["c1", "c1", "c1", "c1"])
        assert b["same_cluster"] is True and b["n_early"] == 2 and b["n_late"] == 2
        b = probe.early_late_bins(rows, ["c1", "c1", "c1", "c2"])
        assert b["same_cluster"] is False and b["late_ids"] == ["c1", "c2"]
        # a jitter-split EARLY bin is the conservative FAIL: fear booked on c1 reads 0.0
        # on a fresh dive that lands on c1' (subset semantics would have passed this)
        b = probe.early_late_bins(rows, ["c1", "c1b", "c1", "c1"])
        assert b["same_cluster"] is False and b["early_ids"] == ["c1", "c1b"]

    def test_empty_bin_is_unmeasured_not_passed(self):
        b = probe.early_late_bins([{"oxygen": 20.0}, {"oxygen": 19.0}], ["c1", "c1"])
        assert b["same_cluster"] is None and b["late_ids"] == []


def _write_dive_trace(tmp_path: Path, ranges, *, contrast_in_water=1.0, n=20, unsettled=False) -> Path:
    """An Exp 60-shaped trace: labelled situations, rescue, oxygen depleting across a visit."""
    prov = {
        "kind": "provenance",
        "code_hash": "test",
        "situations": ["shore", "submerged"],
        "rescue": {"submerged": "shore"},
        "dive_budget_s": 13.25,
        "world_ranges": ranges,
        "world_sensor_count": len(ranges),
    }
    recs = [prov]

    def base():
        s = {"y_altitude": 40, "nearest_hostile_dist": 64, "light_level": 0, "is_in_water": 0, "oxygen": 20}
        for i in range(12):
            s[f"filler{i}"] = 0.5
        return s

    for _ in range(n):
        recs.append({"kind": "sample", "situation": "shore", "state": base()})
    for i in range(n):
        s = base()
        s["y_altitude"] = 35
        s["is_in_water"] = contrast_in_water
        s["oxygen"] = max(0, 20 - i)  # one visit: 20 → 1 across the samples
        rec = {"kind": "sample", "situation": "submerged", "state": s}
        if unsettled and i == 0:
            rec["settled"] = False
        recs.append(rec)
    p = tmp_path / "dive.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def _dive_ranges():
    r = {
        "y_altitude": [0, 128],
        "nearest_hostile_dist": [0, 128],
        "light_level": [0, 15],
        "is_in_water": [-1, 1],
        "oxygen": [0, 40],
    }
    for i in range(12):
        r[f"filler{i}"] = [0, 1]
    return r


def test_labels_ride_the_trace_and_keys_stay_role_positional(tmp_path):
    rec = _run(tmp_path, _write_dive_trace(tmp_path, _dive_ranges()))
    assert rec["situation_labels"] == {"safe": "shore", "dark": "submerged"}
    assert rec["provenance"]["samples"] == {"safe": 20, "dark": 20}
    water = next(s for s in rec["per_sensor"] if s["sensor"] == "is_in_water")
    assert water["v_safe"] == 0.5 and water["v_dark"] == 1.0  # baseline / contrast roles
    assert rec["authorizes_build"] is False


def test_run_gate_passes_on_a_separable_dive_trace(tmp_path):
    """is_in_water neutral→extreme separates; early (full air) and late (pain edge) submerged
    samples land in the same fresh-EC cluster → the Exp 60 run gate PASSES; build stays refused."""
    rec = _run(tmp_path, _write_dive_trace(tmp_path, _dive_ranges()))
    gate = rec["run_gate"]
    assert rec["cosine"]["a4_gained"] < 0.85
    assert gate["is_dive_trace"] and gate["cos_a4_below_threshold"] and gate["fresh_ec_ids_distinct"]
    bins = rec["contrast_early_vs_late_oxygen"]
    assert bins["n_early"] >= 4 and bins["n_late"] >= 4 and bins["same_cluster"] is True
    assert gate["early_late_same_cluster"] is True and gate["pass"] is True
    assert rec["authorizes_build"] is False


def test_run_gate_fails_on_an_unsettled_visit_even_when_geometry_separates(tmp_path):
    """A gate computed on samples that never confirmed their situation is the silent-failure
    shape: recorded, and it FAILS the run gate with the situation named."""
    rec = _run(tmp_path, _write_dive_trace(tmp_path, _dive_ranges(), unsettled=True))
    assert rec["run_gate"]["cos_a4_below_threshold"] and rec["run_gate"]["fresh_ec_ids_distinct"]
    assert rec["run_gate"]["unsettled_situations"] == ["submerged"]
    assert rec["run_gate"]["pass"] is False


def test_dive_record_names_its_experiment_and_keeps_role_keyed_sample_counts(tmp_path):
    rec = _run(tmp_path, _write_dive_trace(tmp_path, _dive_ranges()))
    assert rec["experiment"] == "exp60_chunk_ii_run_gate"
    assert rec["provenance"]["samples"] == {"safe": 20, "dark": 20}
    assert "preflight" in rec["run_gate"]["necessary_not_sufficient"]
    assert "run_gate" in rec["reading"] or "separate" in rec["reading"]  # dive-phrased, not the Slice-2 text
    assert "Slice 2" not in rec["reading"]


def test_run_gate_fails_when_the_cue_does_not_move(tmp_path):
    rec = _run(tmp_path, _write_dive_trace(tmp_path, _dive_ranges(), contrast_in_water=0.0))
    assert rec["run_gate"]["pass"] is False
    assert rec["run_gate"]["fresh_ec_ids_distinct"] is False


def test_slice1_shaped_trace_has_no_dive_gate(tmp_path):
    """The Exp 58 / Slice-1 trace shape (no situations, no rescue) still analyzes unchanged
    and can never pass the run gate (not a dive trace; sub-bins unmeasured)."""
    ranges = _ranges()

    def st():
        s = {"y_altitude": 40, "nearest_hostile_dist": 16, "light_level": 7}
        for i in range(14):
            s[f"filler{i}"] = 0.5
        return s

    rec = _run(tmp_path, _write_trace(tmp_path, st, st, ranges))
    assert rec["situation_labels"] == {"safe": "safe", "dark": "dark"}
    assert rec["experiment"] == "l11_slice1"
    assert rec["contrast_early_vs_late_oxygen"] is None
    assert rec["run_gate"]["is_dive_trace"] is False and rec["run_gate"]["pass"] is False


def test_capture_refuses_to_dive_without_a_measured_onset(tmp_path, monkeypatch):
    """The dive budget comes from the apparatus check's stamped `measured`; without it the
    contrast situation (which names a rescue) must refuse before any teleport."""
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps(_exp60_anchor(measured=False)))
    with pytest.raises(SystemExit, match="measured damage onset"):
        main(["capture", "--rcon-password", "x", "--anchor-file", str(anchor), "--trace", str(tmp_path / "t.jsonl")])
