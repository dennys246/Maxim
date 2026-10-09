"""Unit tests for scripts/grounding_census.py (GL1): the collision logic on synthetic fixtures, and the production
encode path with an INJECTED fake model (no sentence-transformers, no network). The real census runs offline only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import grounding_census as G  # noqa: E402


def _rec(ref: str, name: str, **self_effect: float) -> G.AffordanceRecord:
    return G.AffordanceRecord(
        ref=ref,
        entity=ref.split("/")[-1],
        category=ref.split("/")[0],
        modulator="m",
        name=name,
        self_effect={k.replace("__", "."): v for k, v in self_effect.items()},
    )


def _walk(assignments: list[list[tuple[str, str]]], n_nodes: int, threshold: float = 0.44) -> dict:
    return {
        "per_record": dict(enumerate(assignments)),
        "node_first_text": {},
        "n_nodes": n_nodes,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Harm classification
# ---------------------------------------------------------------------------


def _drive_table() -> dict:
    from maxim.embodiment.sem import EntropicDriveSpec, HomeostaticDriveSpec

    return {
        "bodies/infant": {
            "arms.thermal": HomeostaticDriveSpec(set_point=0.0, drift_rate=0.0, comfort_band=0.5, pain_scale=0.4),
            "hunger": EntropicDriveSpec(
                drift_direction="up",
                drift_rate=0.0,
                deprivation_threshold=0.7,
                deprivation_pain=0.3,
                satisfaction_threshold=0.3,
            ),
        },
    }


def test_harm_class_homeostatic_band_and_entropic_direction():
    table = _drive_table()
    assert G.classify_harm(_rec("items/blanket", "touch", arms__thermal=0.1), table)["class"] == "safe"
    burn = G.classify_harm(_rec("items/fire_pit", "touch", arms__thermal=0.6), table)
    assert burn["class"] == "harmful"
    assert burn["sensors"]["self:arms.thermal"]["harmful_in"] == ["bodies/infant"]
    assert G.classify_harm(_rec("items/food", "eat", hunger=-0.5), table)["class"] == "safe"
    assert G.classify_harm(_rec("items/salt", "lick", hunger=0.8), table)["class"] == "harmful"
    assert G.classify_harm(_rec("weapons/sword", "slash", hp=-10.0), table)["class"] == "unclassified"
    assert G.classify_harm(_rec("items/rock", "observe"), table)["class"] == "no_effect"


def test_harm_class_owner_rule_prefers_the_owning_entity():
    from maxim.embodiment.sem import HomeostaticDriveSpec

    table = _drive_table()
    table["bodies/robot"] = {"arms.thermal": HomeostaticDriveSpec(set_point=0.0, drift_rate=0.0, comfort_band=0.9)}
    owned = G.classify_harm(_rec("bodies/robot", "vent", arms__thermal=0.6), table)
    assert owned["class"] == "safe" and owned["sensors"]["self:arms.thermal"]["rule"] == "owner"


# ---------------------------------------------------------------------------
# Collision analysis on injected node assignments
# ---------------------------------------------------------------------------


def _fixture():
    records = [
        _rec("items/blanket", "touch", arms__thermal=0.1),  # 0 safe
        _rec("items/fire_pit", "touch", arms__thermal=0.6),  # 1 harmful, same node as 0
        _rec("bodies/a", "turn_left", azimuth=0.3),  # 2
        _rec("bodies/a", "turn_right", azimuth=-0.3),  # 3 opposite sign, same node as 2
        _rec("items/hearth", "warm_hands", arms__thermal=0.2),  # 4 same consequence as 0, other node
        _rec("items/fire_pit", "observe"),  # 5 no effect
    ]
    walk = _walk(
        [
            [("touch", "N0")],
            [("touch", "N0")],
            [("turn left", "N1"), ("turn", "N1"), ("left", "N2")],
            [("turn right", "N1"), ("turn", "N1"), ("right", "N3")],
            [("warm hands", "N4"), ("warm", "N5"), ("hands", "N6")],
            [("observe", "N7")],
        ],
        n_nodes=8,
    )
    harm = [G.classify_harm(r, _drive_table()) for r in records]
    return records, walk, harm


def test_same_node_collisions_harm_flip_and_opposite_sign():
    records, walk, harm = _fixture()
    out = G.analyse(records, walk, harm)
    pairs = {(c["a"], c["b"]): c for c in out["collisions"]}
    touch = pairs[("touch[self:arms.thermal=+0.1]", "touch[self:arms.thermal=+0.6]")]
    assert touch["harm_flip"] and touch["same_name"] and touch["node"] == "N0" and not touch["opposite_sensors"]
    turn = pairs[("turn_left[self:azimuth=+0.3]", "turn_right[self:azimuth=-0.3]")]
    assert turn["opposite_sensors"] == ["self:azimuth"] and not turn["same_name"]
    assert turn["opposite_magnitude"] == pytest.approx(0.6)
    assert len(out["collisions"]) == 2
    # Harm flips sort first.
    assert out["collisions"][0]["harm_flip"]


def test_same_consequence_on_different_nodes_and_no_collision_across_nodes():
    records, walk, harm = _fixture()
    out = G.analyse(records, walk, harm)
    same = {(x["a"], x["b"]) for x in out["same_consequence_different_node"]}
    assert ("touch[self:arms.thermal=+0.1]", "warm_hands[self:arms.thermal=+0.2]") in same
    # The harmful touch and warm_hands differ in harm class: not "same consequence".
    assert ("touch[self:arms.thermal=+0.6]", "warm_hands[self:arms.thermal=+0.2]") not in same


def test_shared_word_links_and_component_placement():
    records = [_rec("c/x", "flame_jet"), _rec("c/y", "water_jet"), _rec("c/z", "fire_breath")]
    walk = _walk(
        [
            [("flame jet", "N0"), ("flame", "N0"), ("jet", "N1")],
            [("water jet", "N2"), ("water", "N2"), ("jet", "N1")],
            [("fire breath", "N3"), ("fire", "N0"), ("breath", "N3")],
        ],
        n_nodes=4,
    )
    out = G.analyse(records, walk, [{"class": "no_effect", "sensors": {}}] * 3)
    links = {(x["a"], x["b"], x["word_node"]) for x in out["shared_word_links"]}
    assert ("flame_jet", "water_jet", "N1") in links
    assert ("fire_breath", "flame_jet", "N0") in links  # `fire` landed on flame jet's compound node
    comp = out["components"]
    assert "flame_jet:flame" in comp["absorbed_into_own_compound"]
    assert "fire_breath:fire->N0" in comp["landed_on_another_compound"]
    assert "flame_jet:jet" in comp["own_node"]


def test_drift_lists_use_the_threshold():
    records = [_rec("c/a", "alpha"), _rec("c/b", "beta"), _rec("c/c", "gamma")]
    walk = _walk([[("alpha", "N0")], [("beta", "N0")], [("gamma", "N1")]], n_nodes=2)
    cos = {("alpha", "beta"): 0.30, ("alpha", "gamma"): 0.50, ("beta", "gamma"): 0.10}
    out = G.analyse(records, walk, [{"class": "no_effect", "sensors": {}}] * 3, cos)
    assert out["drift"]["co_noded_below_threshold"] == [{"a": "alpha", "b": "beta", "cosine": 0.3}]
    assert out["drift"]["separated_at_or_above_threshold"] == [{"a": "alpha", "b": "gamma", "cosine": 0.5}]


def test_known_answer_and_order_dependence():
    records = [
        _rec("items/cradle_blanket", "touch", arms__thermal=0.1),
        _rec("items/cradle_fire_pit", "touch", arms__thermal=0.6),
    ]
    harm = [G.classify_harm(r, _drive_table()) for r in records]
    same = _walk([[("touch", "N0")], [("touch", "N0")]], n_nodes=1)
    split = _walk([[("touch", "N0")], [("touch", "N1")]], n_nodes=2)
    a, b = G.analyse(records, same, harm), G.analyse(records, split, harm)
    assert G.known_answer(records, same, a)["ok"]
    assert not G.known_answer(records, split, b)["ok"]
    od = G.order_dependence({"sorted": a, "other": b})
    assert od["collisions_in_every_order"] == 0 and od["collisions_in_any_order"] == 1
    assert od["order_sensitive_collisions"] == a["collision_keys"]


def test_walk_orders_are_deterministic_permutations():
    records = [_rec("c/a", n) for n in ("b", "a", "c", "d")]
    o1, o2 = G.walk_orders(4, records, 7), G.walk_orders(4, records, 7)
    assert o1 == o2
    assert o1["sorted"] == [1, 0, 2, 3] and o1["reverse_sorted"] == [3, 2, 0, 1]
    assert sorted(o1["shuffle_seed_7"]) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# The production encode path, with an injected model
# ---------------------------------------------------------------------------


class _FakeModel:
    """Stands in for the SentenceTransformer: fixed vectors per string, unit-normalised."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._v = vectors

    def encode(self, text: str, convert_to_numpy: bool = True):
        v = np.asarray(self._v[text], dtype=float)
        return v / np.linalg.norm(v)


def test_encode_walk_through_the_production_affordance_path(monkeypatch):
    import maxim.similarity.encoder as enc

    vectors = {
        "touch": [1, 0, 0, 0, 0, 0],
        "turn left": [0, 1, 0.2, 0, 0, 0],
        "turn right": [0, 1, -0.2, 0, 0, 0],  # cos ~0.92 to turn left: binds at 0.44
        "turn": [0, 1, 0, 0, 0, 0],
        "left": [0, 0, 0, 1, 0, 0],
        "right": [0, 0, 0, 0, 1, 0],
    }
    monkeypatch.setattr(enc, "_get_encoder", lambda *a, **k: _FakeModel(vectors))
    records = [
        _rec("items/cradle_blanket", "touch", arms__thermal=0.1),
        _rec("items/cradle_fire_pit", "touch", arms__thermal=0.6),
        _rec("bodies/a", "turn_left", azimuth=0.3),
        _rec("bodies/a", "turn_right", azimuth=-0.3),
    ]
    walk = G.encode_walk(records, [0, 1, 2, 3], G.build_aff_encoder)
    per = walk["per_record"]
    assert per[0] == [("touch", "N0")] and per[1] == [("touch", "N0")]
    # Compound first, then each word: the decomposer's chunk order, relabelled in creation order.
    assert [t for t, _ in per[2]] == ["turn left", "turn", "left"]
    assert per[3][0][1] == per[2][0][1]  # turn right completes into turn left's node
    assert per[2][1][1] == per[2][0][1]  # `turn` is absorbed into its compound
    assert walk["n_nodes"] == 4  # touch, turn left, left, right
    walk["threshold"] = 0.44
    harm = [G.classify_harm(r, _drive_table()) for r in records]
    out = G.analyse(records, walk, harm)
    assert G.known_answer(records, walk, out)["ok"]
    assert out["opposite_sign_collisions"] == 1 and out["harm_flip_collisions"] == 1


def test_encode_walk_refuses_the_hash_fallback(monkeypatch):
    import maxim.similarity.encoder as enc

    monkeypatch.setattr(enc, "_get_encoder", lambda *a, **k: None)
    with pytest.raises(G.Refusal) as exc:
        G.encode_walk([_rec("c/a", "touch")], [0], G.build_aff_encoder)
    assert exc.value.reason == "encoder_fallback"


def test_importing_the_script_imports_nothing_heavy_and_changes_no_environment():
    code = (
        "import json, os, sys; before = dict(os.environ); "
        f"sys.path.insert(0, {str(REPO / 'scripts')!r}); "
        "import grounding_census; "
        "heavy = sorted(m for m in ('maxim', 'torch', 'sentence_transformers') if m in sys.modules); "
        "print(json.dumps({'heavy': heavy, 'env_same': dict(os.environ) == before}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"heavy": [], "env_same": True}


def test_compact_links_keeps_count_and_fanout():
    result = {
        "shared_word_links": [
            {"a": "x", "b": "y", "word_node": "N1", "words": ["jet"], "cosine": None},
            {"a": "x", "b": "z", "word_node": "N1", "words": ["jet"], "cosine": None},
            {"a": "y", "b": "z", "word_node": "N2", "words": ["self"], "cosine": None},
        ]
    }
    out = G.compact_links(result, {"N1": "flame jet"})
    assert "shared_word_links" not in out and out["shared_word_links_count"] == 3
    assert out["shared_word_fanout"][0] == {
        "word_node": "N1",
        "first_text": "flame jet",
        "compounds": 3,
        "words": ["jet"],
    }


# ---------------------------------------------------------------------------
# GL1 code-review folds: entropic rest, deterministic EC config, refusal siblings, headline facts
# ---------------------------------------------------------------------------


def test_entropic_harm_is_measured_from_rest_not_from_zero():
    from maxim.embodiment.sem import EntropicDriveSpec

    food = EntropicDriveSpec(
        drift_direction="down",
        drift_rate=0.0,
        deprivation_threshold=6.0,
        deprivation_pain=0.3,
        satisfaction_threshold=16.0,
    )
    # From rest 20 a -8 drain lands at 12: not deprived. Measured from 0 (the old rule) it would have been harmful.
    assert G.sensor_harm(-8.0, food, rest=20.0) is False
    assert G.sensor_harm(-15.0, food, rest=20.0) is True
    assert G.sensor_harm(+4.0, food, rest=20.0) is False  # against the drift: never harmful
    assert G.sensor_harm(-15.0, food, rest=None) is None  # a down-drift with no rest is unclassifiable
    cold = EntropicDriveSpec(
        drift_direction="up",
        drift_rate=0.0,
        deprivation_threshold=0.5,
        deprivation_pain=0.15,
        satisfaction_threshold=0.3,
    )
    assert G.sensor_harm(0.1, cold, rest=0.6) is True  # already past the threshold: deepening is harm
    assert G.sensor_harm(-0.3, cold, rest=0.6) is False
    assert G.sensor_harm(0.4, cold, rest=None) is False  # up-drift default rest 0.0
    table = {"bodies/mc": {"food": food}}
    drain = _rec("items/x", "sprint", food=-8.0)
    assert G.classify_harm(drain, table, {"bodies/mc": {"food": 20.0}})["class"] == "safe"
    assert G.classify_harm(drain, table)["class"] == "unclassified"


def test_sensor_rest_reads_the_declared_rest_before_initial():
    from types import SimpleNamespace

    entity = SimpleNamespace(
        sensors={"food": SimpleNamespace(_initial=5), "light": SimpleNamespace(_initial=7)},
        modulators={"arms": SimpleNamespace(vital_metrics={"thermal": 0.25})},
    )
    spec = {
        "entity": {
            "sensors": {"food": {"rest": 20, "initial": 5}, "light": {"rest": None, "initial": 7}},
            "modulators": {"arms": {"sensors": {"thermal": {"initial": 0.25}}}},
        }
    }
    assert G.sensor_rest(entity, "food", spec) == 20.0  # declared rest wins over initial
    assert G.sensor_rest(entity, "light", spec) is None  # `rest: null` = no rest, never a fallback
    assert G.sensor_rest(entity, "arms.thermal", spec) == 0.25  # no rest declared: the initial state
    assert G.sensor_rest(entity, "food") == 5.0  # no spec: the initial state


def test_ec_config_record_is_deterministic_across_hash_seeds():
    code = (
        f"import json, sys; sys.path.insert(0, {str(REPO / 'scripts')!r}); "
        "import grounding_census as G; from maxim.similarity.ec import ECConfig; "
        "print(json.dumps(G.ec_config_record(ECConfig()), sort_keys=True))"
    )
    outs = []
    for seed in ("1", "2", "12345"):
        env = {**__import__("os").environ, "PYTHONHASHSEED": seed}
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
        outs.append(res.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1] == outs[2]
    rec = json.loads(outs[0])
    assert rec["pattern_complete_threshold"] == 0.44
    assert rec["frozen_centroid_modalities"] == sorted(rec["frozen_centroid_modalities"])


def _stub_main(monkeypatch, tmp_path):
    """``main`` with its paths and provenance stubbed. ``_provenance.stamp_diagnosis`` stays REAL (it is pure): it
    derives ``status`` from ``report["refusal"]``, so a stamp taken before the refusal is recorded reads ``ok``."""
    import _provenance

    json_path, md_path = tmp_path / "census.json", tmp_path / "summary.md"
    monkeypatch.setattr(G, "pin_environment", lambda: None)
    monkeypatch.setattr(_provenance, "evidence_out_paths_or_exit", lambda *a, **k: (json_path, md_path))
    monkeypatch.setattr(_provenance, "in_process_code_provenance", lambda *a, **k: {"executed_git_hash": "0" * 40})
    return json_path, md_path


def _refusing_main(monkeypatch, tmp_path):
    json_path, md_path = _stub_main(monkeypatch, tmp_path)

    def refuse(seed):
        raise G.Refusal("known_answer", "synthetic")

    monkeypatch.setattr(G, "measure", refuse)
    return json_path, md_path


def test_a_refusal_never_overwrites_the_record(monkeypatch, tmp_path):
    json_path, md_path = _refusing_main(monkeypatch, tmp_path)
    json_path.write_text('{"good": true}')
    md_path.write_text("good summary")
    assert G.main(["--write-experiment-results"]) == 4
    assert json.loads(json_path.read_text()) == {"good": True}
    assert md_path.read_text() == "good summary"
    failed_json, failed_md = G.failed_paths(json_path)
    assert failed_json.name == "census.failed.json" and failed_md.name == "census.failed.md"
    failed = json.loads(failed_json.read_text())
    assert failed["refusal"] == "known_answer" and failed["status"] == "failed"
    assert failed["record_kind"] == "diagnosis"
    assert "REFUSED" in failed_md.read_text()


def test_a_passing_run_removes_stale_refusal_siblings(monkeypatch, tmp_path):
    json_path, md_path = _stub_main(monkeypatch, tmp_path)
    passing = {
        "counts": {"affordances": 1},
        "results": {"sorted": {"collisions": [], "harm_flip_collisions": 0, "opposite_sign_collisions": 0}},
        "known_answer": {"ok": True},
    }
    monkeypatch.setattr(G, "measure", lambda seed: dict(passing))
    monkeypatch.setattr(G, "_write", lambda j, m, report: (j.write_text(json.dumps(report)), m.write_text("ok")))
    failed_json, failed_md = G.failed_paths(json_path)
    failed_json.write_text('{"refusal": "known_answer"}')
    failed_md.write_text("REFUSED")
    # Default (temp) mode never touches the siblings: they belong to the committed record's directory.
    assert G.main([]) == 0
    assert failed_json.exists() and failed_md.exists()
    assert G.main(["--write-experiment-results"]) == 0
    assert not failed_json.exists() and not failed_md.exists()
    assert json.loads(json_path.read_text())["status"] == "ok"


def _col(a, b, flip, same, cos=1.0, opp=(), harm=("safe", "harmful")):
    return {
        "a": a,
        "b": b,
        "harm_flip": flip,
        "same_name": same,
        "cosine": cos,
        "opposite_sensors": list(opp),
        "harm": list(harm),
    }


def _headline_report(sorted_collisions, azimuth_modes=("homeostatic",)):
    return {
        "results": {
            "primary_order": "sorted",
            "sorted": {"collisions": sorted_collisions},
            "reverse_sorted": {"collisions": [_col("feel[x=-0.1]", "touch[x=+0.6]", True, False, 0.51)]},
        },
        "affordances": [
            {
                "name": "turn_left",
                "harm": {
                    "class": "harmful",
                    "sensors": {"self:azimuth": {"class": "harmful", "harmful_drive_modes": list(azimuth_modes)}},
                },
            },
            {"name": "touch", "harm": {"class": "harmful", "sensors": {"self:arms.thermal": {"class": "harmful"}}}},
            {"name": "look", "harm": {"class": "no_effect", "sensors": {}}},
        ],
    }


_TOUCH_FLIP = _col("touch[x=+0.1]", "touch[x=+0.6]", True, True)
_TURN_PAIR = _col(
    "turn_left[a=+0.3]", "turn_right[a=-0.3]", False, False, 0.87, ["self:azimuth"], ["harmful", "harmful"]
)
_FEEL_TOUCH_FLIP = _col("feel[x=-0.1]", "touch[x=+0.6]", True, False, 0.51)


def test_headline_facts_read_the_collision_lists():
    hf = G.headline_facts(_headline_report([_TOUCH_FLIP, _TURN_PAIR]))
    assert hf["harm_flips"] == hf["same_name_harm_flips"] == 1 and hf["cross_name_harm_flips"] == 0
    assert hf["same_name_harm_flip_names"] == ["touch"] and hf["same_name_harm_flip_cosines"] == [1.0]
    assert hf["cross_name_collision_pairs"] == ["turn_left ↔ turn_right"]
    assert hf["cross_name_harm_flips_other_orders"] == {"reverse_sorted": {"feel ↔ touch": 1}}
    assert (hf["harmful_instances"], hf["harmful_instances_azimuth"]) == (2, 1)
    assert hf["all_harm_flips_same_name"] and hf["cross_name_all_azimuth"]
    assert hf["cross_name_all_opposite_and_both_harmful"] and hf["azimuth_harm_all_comfort_band"]
    text = G.headline_bullet(hf, "")
    assert "Every primary-walk harm flip is a SAME-name variant pair" in text
    assert "all orient (azimuth) pairs" in text and "opposite signs, both harmful" in text
    assert "exceeds the orienting drive's comfort band" in text

    # A cross-name harm flip in the PRIMARY walk: the same-name / cross-name split must bite, and every
    # qualitative sentence it falsifies must disappear from the headline.
    hf = G.headline_facts(_headline_report([_TOUCH_FLIP, _TURN_PAIR, _FEEL_TOUCH_FLIP], azimuth_modes=["entropic"]))
    assert (hf["harm_flips"], hf["same_name_harm_flips"], hf["cross_name_harm_flips"]) == (2, 1, 1)
    assert hf["cross_name_collisions"] == 2
    assert hf["cross_name_collision_pairs"] == ["feel ↔ touch", "turn_left ↔ turn_right"]
    assert not hf["all_harm_flips_same_name"]
    assert not hf["cross_name_all_azimuth"] and not hf["cross_name_all_opposite_and_both_harmful"]
    assert not hf["azimuth_harm_all_comfort_band"]
    text = G.headline_bullet(hf, "")
    assert "Every primary-walk harm flip" not in text and "**1 of 2** primary-walk harm flips" in text
    assert "not all orient (azimuth) pairs" in text and "NOT all opposite-signed" in text
    assert "because one turn exceeds" not in text
