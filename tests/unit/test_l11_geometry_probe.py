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
