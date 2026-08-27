"""Exp 54 additions to the Exp 53 cross-context readout harness
(scripts/orient_backbone/exp53_cross_context_readout.py) — the pieces that
turn a number into a verdict, pinned offline:

- direction-only correctness admits the ``_big`` pair (the Exp 52/54 rule);
- the sweep's target procedure (bins → strongest-bias bins → two magnitudes per
  direction, front hemisphere; wrong-way region → exploratory) reproduces the
  Exp 53 amendment-1 placements from the Exp 53 cluster map;
- Gate C (the user path) passes/fails on consulted audio bias + direction;
- ``--targets`` refuses placements outside the front hemisphere;
- the production factory read binds the Reachy nursery body's four deltas and
  refuses the infant body (no ``head_yaw`` self-effect → no δ map anywhere);
- ``--delta`` is refused with ``--factory``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "orient_backbone" / "exp53_cross_context_readout.py"


@pytest.fixture(scope="module")
def h():
    sys.path.insert(0, str(_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("exp53_cross_context_readout", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_direction_only_correctness_admits_big_turns(h):
    assert h._correct_for(-0.5, "turn_left") is True
    assert h._correct_for(-0.5, "turn_left_big") is True
    assert h._correct_for(-0.5, "turn_right_big") is False
    assert h._correct_for(+0.5, "turn_right_big") is True
    assert h._correct_for(+0.5, "turn_left_big") is False
    assert h._correct_for(+0.5, None) is None


def _exp53_rows(h, far_left_bias=0.006, centre_left=0.62, right_bias=0.90):
    """The Exp 53 amendment-1 map: FAR-LEFT (≤ −0.5) / CENTRE (−0.4…+0.3) / RIGHT (≥ +0.4)."""
    rows = []
    for az in h._sweep_values():
        if az <= -0.5:
            cid, biases, aff = "farleft", {"turn_left": far_left_bias}, "turn_left"
        elif az <= 0.3:
            cid, biases, aff = "centre", {"turn_left": centre_left}, "turn_left"
        else:
            cid, biases, aff = "right", {"turn_right": right_bias}, "turn_right"
        margin = max(biases.values())
        rows.append(
            {
                "az": az,
                "audio_cluster": cid,
                "completed": True,
                "affordance": aff,
                "correct": h._correct_for(az, aff),
                "learned_margin": margin,
                "consulted_audio": margin,
                "biases": biases,
            }
        )
    return rows


def test_sweep_procedure_reproduces_exp53_targets(h):
    per_agent = {}
    for seed in (42, 43, 44):
        rows = _exp53_rows(h)
        per_agent[f"taught_seed{seed}"] = {"rows": rows, "bins": h._bins_from_rows(rows)}
    bins = per_agent["taught_seed42"]["bins"]
    assert [b["cluster"] for b in bins] == ["farleft", "centre", "right"]
    assert h._strongest(bins, "left_strength")["cluster"] == "centre"
    assert h._strongest(bins, "right_strength")["cluster"] == "right"
    decl = h._declare_targets(per_agent, majority=2)
    assert decl["gated_targets"] == [-0.3, -0.2, 0.5, 0.6]
    # centre bin's right half (0.1…0.3): the learned turn_left fires the wrong way with margin
    assert decl["predicted_wrong_way_region"] == [0.1, 0.2, 0.3]
    assert decl["exploratory_targets"] == [0.2]
    assert decl["flags"] == []


def test_sweep_procedure_flags_a_direction_with_no_bias(h):
    per_agent = {}
    for seed in (42, 43, 44):
        rows = _exp53_rows(h, centre_left=0.0, far_left_bias=0.0)
        per_agent[f"taught_seed{seed}"] = {"rows": rows, "bins": h._bins_from_rows(rows)}
    decl = h._declare_targets(per_agent, majority=2)
    assert decl["gated_targets"] == [0.5, 0.6]
    assert any("no bin with a left bias" in f for f in decl["flags"])


def test_big_turn_keys_count_toward_bin_strength(h):
    rows = _exp53_rows(h)
    for r in rows:
        if r["audio_cluster"] == "centre":
            r["biases"] = {"turn_left_big": 0.7, "turn_left": 0.1}
    bins = h._bins_from_rows(rows)
    assert h._strongest(bins, "left_strength")["left_strength"] == 0.7


def _probe(correct, audio):
    return {"exploratory": False, "correct": correct, "consulted_bias_by_modality": {"audio": audio}, "tool_name": "x"}


def test_gate_C_pass_and_fail(h):
    agents = [
        {"label": "taught_seed42", "arm": "taught", "exploratory": False},
        {"label": "taught_seed43", "arm": "taught", "exploratory": False},
        {"label": "taught_seed44", "arm": "taught", "exploratory": False},
        {"label": "taught_seed48", "arm": "taught", "exploratory": True},
        {"label": "satiated_seed42", "arm": "satiated", "exploratory": False},
    ]
    good = [_probe(True, 0.6)] * 4
    results = {
        "taught_seed42": good,
        "taught_seed43": good,
        "taught_seed44": [_probe(False, 0.6)] * 4,  # consulted but wrong: fails this seed
        "taught_seed48": [_probe(False, 0.3)] * 4,  # exploratory: never gated
        "satiated_seed42": [_probe(None, 0.0)] * 4,
    }
    v = h._gate_C(agents, results)
    assert v["verdict"] == "PASS" and v["taught_seeds_passing"] == 2 and v["controls_consulted_zero"] is True
    # A control that CONSULTS a bias fails the gate regardless of the taught seeds.
    results["satiated_seed42"] = [_probe(True, 0.2)] * 4
    assert h._gate_C(agents, results)["verdict"] == "FAIL"
    # Correct direction WITHOUT a consulted audio bias (the innate drive choosing) does not count.
    results["satiated_seed42"] = [_probe(None, 0.0)] * 4
    results["taught_seed43"] = [_probe(True, 0.0)] * 4
    assert h._gate_C(agents, results)["verdict"] == "FAIL"


def test_apply_targets_refuses_outside_front_hemisphere(h, tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"gated_targets": [-0.3, 0.8], "exploratory_targets": []}))
    with pytest.raises(RuntimeError, match="front hemisphere"):
        h._apply_targets(str(p))
    p.write_text(json.dumps({"gated_targets": [-0.3, 0.6], "exploratory_targets": [0.2]}))
    h._apply_targets(str(p))
    assert h.TARGETS == (-0.3, 0.6) and h.EXPLORATORY_TARGETS == (0.2,)
    h.TARGETS, h.EXPLORATORY_TARGETS = (-0.3, -0.2, 0.5, 0.6), (-0.6, 0.2)  # restore the Exp 53 constants


def test_factory_deltas_bind_reachy_infant_and_refuse_infant_operant(h):
    from maxim.embodiment.component_registry import ComponentRegistry

    reg = ComponentRegistry()
    deltas = h._factory_deltas(reg.instantiate("bodies/reachy_mini_infant"))
    assert deltas == pytest.approx({"turn_left": 0.3, "turn_right": -0.3, "turn_left_big": 0.9, "turn_right_big": -0.9})
    with pytest.raises(RuntimeError, match="head_yaw"):
        h._factory_deltas(reg.instantiate("bodies/infant_operant"))


def test_dry_rig_factory_mode_builds_the_reachy_body_with_four_tools(h):
    rig = h.DryReadoutRig(body_ref="bodies/reachy_mini_infant", factory_mode=True)
    assert rig.entity.name == "reachy_mini"
    assert set(rig.deltas) == {"turn_left", "turn_right", "turn_left_big", "turn_right_big"}
    for aff in rig.deltas:
        assert f"reachy_mini_{aff}" in rig.registry.list()
    res = rig.execute("turn_left_big")
    assert res.success and res.entity_name == "reachy_mini"
    # The Exp 53 default is byte-for-byte the old rig: infant body, explicit δ.
    old = h.DryReadoutRig()
    assert old.entity.name == "infant_operant" and old.deltas == h.DELTAS


def test_delta_is_refused_with_factory(h, capsys):
    rc = h.main(
        ["run", "--dry-run", "--factory", "--delta", "0.3", "--manifest", "x.json", "--phase", "1", "--out", "y"]
    )
    assert rc == 2
    assert "refused" in capsys.readouterr().out
