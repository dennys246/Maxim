"""Known-answer guards for the E4 harness's decision logic (scripts/e4_text_widening_drift.py, #911).

The measurement itself needs the real encoder and runs offline by hand; these pin the pure pieces a bookkeeping bug
would corrupt without any instrument check noticing: the frozen decision rule (COLLAPSE / NO HEADROOM / NO
COLLAPSE, and the marginal tag), the replay reference (the EC's own centroid rule), the walk and its fixture pin,
and that importing the harness changes no environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import e4_text_widening_drift as E  # noqa: E402

BASE = 0.44
SEED = E.SEED


def _walk() -> list[dict]:
    # seed + its pair-mate (food), one more food string, three foreign strings
    return [
        {"text": SEED, "cls": "food", "entry": E.SEED_PAIR},
        {"text": "mate", "cls": "food", "entry": E.SEED_PAIR},
        {"text": "food2", "cls": "food", "entry": "pair_02"},
        {"text": "f1", "cls": "other", "entry": "pair_05"},
        {"text": "f2", "cls": "other", "entry": "pair_05"},
        {"text": "f3", "cls": "other2", "entry": "pair_06"},
    ]


def _run(in_seed: set[str], cos_rewarded: dict[str, float], winner: dict[str, float] | None = None) -> dict:
    rows = []
    for w in _walk():
        t = w["text"]
        rows.append(
            {
                "text": t,
                "cls": w["cls"],
                "node": SEED if (t == SEED or t in in_seed) else t,
                "cos_rewarded": cos_rewarded.get(t, 0.1),
                "rewarded_threshold": BASE,
                "winner_similarity": (winner or {}).get(t, 0.9),
            }
        )
    return {"rows": rows}


def _replay(admitted: set[str], cos_by_text: dict[str, float] | None = None) -> dict:
    return {
        w["text"]: {"cos": (cos_by_text or {}).get(w["text"], 0.0), "admitted": w["text"] in admitted}
        for w in _walk()[1:]
    }


def _decide(a0: set[str], a2: set[str], i2: set[str], cos2: dict[str, float] | None = None, cos_replay=None) -> dict:
    seq = {0.0: _run(a0, {}), E.CAP: _run(a2, cos2 or {})}
    return E.decide(_walk(), seq, {0.0: _replay(set()), E.CAP: _replay(i2, cos_replay)}, BASE)


def test_collapse_needs_all_three_clauses():
    verdict = _decide(a0=set(), a2={"f1"}, i2=set(), cos2={"f1": 0.40}, cos_replay={"f1": 0.10})
    assert verdict["outcome"] == "COLLAPSE"
    assert verdict["collapse_strings"] == ["f1"]


def test_a_string_already_absorbed_without_reward_is_not_collapse():
    verdict = _decide(a0={"f1"}, a2={"f1"}, i2=set())
    assert verdict["collapse_strings"] == []


def test_a_string_widening_alone_admits_is_not_collapse():
    """Widening working as designed (s in I(0.2)), reported as overreach, never as drift."""
    verdict = _decide(a0=set(), a2={"f1"}, i2={"f1"})
    assert verdict["collapse_strings"] == []
    assert "f1" in verdict["widening_overreach"]


def test_within_class_absorption_is_never_collapse():
    verdict = _decide(a0=set(), a2={"food2"}, i2=set())
    assert verdict["collapse_strings"] == []


def test_no_headroom_when_every_foreign_string_is_absorbed_or_admitted():
    verdict = _decide(a0={"f1"}, a2={"f1"}, i2={"f2", "f3"})
    assert verdict["outcome"] == "NO HEADROOM"


def test_no_collapse_when_there_is_headroom_and_nothing_drifted():
    verdict = _decide(a0=set(), a2=set(), i2={"f1"})
    assert verdict["outcome"] == "NO COLLAPSE"
    assert verdict["headroom_strings"] == ["f2", "f3"]


def test_a_collapse_within_one_hundredth_of_every_threshold_is_marginal():
    over = E.realised_override(E.CAP, BASE)
    verdict = _decide(
        a0=set(),
        a2={"f1"},
        i2=set(),
        cos2={"f1": over + 0.005},
        cos_replay={"f1": over - 0.005},
    )
    assert verdict["outcome"] == "COLLAPSE (marginal)"


def test_the_replay_reference_follows_the_ecs_own_centroid_rule():
    run = {
        "rows": [
            {"text": SEED, "node": SEED},
            {"text": "m", "node": SEED},
            {"text": "x", "node": "x"},
            {"text": "y", "node": SEED},
        ],
        "embeddings": {SEED: [1.0, 0.0], "m": [0.0, 1.0], "x": [5.0, 5.0], "y": [1.0, 1.0]},
    }
    running = E.replay_references(run, frozen=False)
    assert running["m"] == [1.0, 0.0]  # only the seed precedes it
    assert running["x"] == [0.5, 0.5]  # seed + m (x itself never joins)
    assert running["y"] == [0.5, 0.5]  # x is not a member
    frozen = E.replay_references(run, frozen=True)
    assert frozen["y"] == [1.0, 0.0]  # a frozen modality holds the seed


def test_the_walk_is_the_prereg_walk():
    walk = E.load_walk()
    assert len(walk) == 22
    assert walk[0]["text"] == SEED and walk[0]["entry"] == E.SEED_PAIR
    assert sum(1 for w in walk if w["entry"] != E.SEED_PAIR) == 20
    assert len({w["text"] for w in walk}) == 22


def test_a_changed_fixture_is_a_refusal(tmp_path, monkeypatch):
    other = tmp_path / "roy.json"
    other.write_text(E.FIXTURE.read_text() + " ")
    monkeypatch.setattr(E, "FIXTURE", other)
    with pytest.raises(E.Refusal) as exc:
        E.load_walk()
    assert exc.value.reason == "fixture_sha_mismatch"


def test_importing_the_harness_imports_nothing_heavy_and_changes_no_environment():
    """A fresh interpreter: importing the harness must not pull in maxim / torch / HF (which read HF_HUB_OFFLINE at
    import time, before pin_environment could set it), nor touch the environment or sys.path."""
    import json
    import subprocess

    code = (
        "import json, os, sys; before = dict(os.environ); path = list(sys.path); "
        f"sys.path.insert(0, {str(REPO / 'scripts')!r}); path = list(sys.path); "
        "import e4_text_widening_drift; "
        "heavy = [m for m in ('maxim', 'torch', 'sentence_transformers', 'huggingface_hub') if m in sys.modules]; "
        "print(json.dumps({'heavy': heavy, 'env_same': dict(os.environ) == before, 'path_same': sys.path == path}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"heavy": [], "env_same": True, "path_same": True}


def _clear_maxim_env(monkeypatch) -> None:
    """Other tests leave MAXIM_* trace toggles set; clear every one (monkeypatch restores them afterwards)."""
    for key in [k for k in os.environ if k.startswith("MAXIM_")]:
        monkeypatch.delenv(key)


def test_an_unexpected_maxim_toggle_is_reported(monkeypatch):
    _clear_maxim_env(monkeypatch)
    monkeypatch.setenv("MAXIM_NAC_REWARD_BIAS_DISABLED", "1")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MAXIM_DATA_HOME"):
        monkeypatch.setenv(key, "placeholder")  # recorded, so monkeypatch restores it after pin_environment
    assert E.pin_environment() == ["MAXIM_NAC_REWARD_BIAS_DISABLED"]
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_main_refuses_an_unexpected_toggle_with_a_failed_record(monkeypatch):
    _clear_maxim_env(monkeypatch)
    monkeypatch.setenv("MAXIM_NAC_REWARD_BIAS_DISABLED", "1")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MAXIM_DATA_HOME"):
        monkeypatch.setenv(key, "placeholder")
    written: list[dict] = []
    monkeypatch.setattr(E, "_write", lambda path, report: written.append(report))
    assert E.main([]) == 4
    [record] = written
    assert record["status"] == "failed" and record["refusal"] == "env_toggle_set"
    assert "verdict" not in record and record["scope"] == E.SCOPE


def _bare_stack(credited: dict[str, float]):
    stack = E.Stack.__new__(E.Stack)  # no EC/model: check_wiring reads only the call record
    stack.base, stack.credited = BASE, dict(credited)
    return stack


def _call(override, geometry="g", expected="g", modality="text"):
    return {"override": override, "geometry": geometry, "geometry_expected": expected, "modality": modality}


@pytest.mark.parametrize(
    ("credited", "call"),
    [
        ({"n": 0.2}, _call({"n": 0.30})),  # wrong value
        ({"n": 0.2}, _call({"n": BASE - 0.2, "m": 0.24})),  # an extra node
        ({"n": 0.2}, _call(None)),  # credited but no override reached the EC
        ({}, _call({"n": 0.24})),  # an override with nothing credited (b = 0)
        ({}, _call(None, geometry="a", expected="b")),  # wrong geometry
        ({}, _call(None, modality="vision")),  # wrong modality
    ],
)
def test_check_wiring_refuses_every_wrong_shape(credited, call):
    with pytest.raises(E.Refusal) as exc:
        _bare_stack(credited).check_wiring(call)
    assert exc.value.reason == "wiring"


def test_check_wiring_accepts_the_right_shapes():
    _bare_stack({}).check_wiring(_call(None))
    _bare_stack({"n": 0.2, "m": 0.1}).check_wiring(_call({"n": BASE - 0.2, "m": BASE - 0.1}))


def test_the_class_rule_on_the_real_fixture():
    walk = E.load_walk()
    far = E.foreign(walk)
    assert len(far) == 22 - sum(1 for w in walk if w["cls"] == walk[0]["cls"])
    assert {"two people are arguing in the next room.", "the room grows quiet."} <= far
    # A string in both a pair and a distractor keeps its pair's class: the seed is also in a distractor.
    assert walk[0]["cls"] == "food_overlap_2pc" and SEED not in far


def test_marginal_needs_every_collapse_string_to_be_marginal():
    over = E.realised_override(E.CAP, BASE)
    verdict = _decide(
        a0=set(),
        a2={"f1", "f2"},
        i2=set(),
        cos2={"f1": over + 0.005, "f2": over + 0.2},
        cos_replay={"f1": over - 0.005, "f2": over - 0.2},
    )
    assert verdict["outcome"] == "COLLAPSE"


def test_a_negative_clause_margin_is_an_instrument_refusal():
    over = E.realised_override(E.CAP, BASE)
    with pytest.raises(E.Refusal) as exc:
        # in A(0.2) yet its cosine sits below the cap's override: inconsistent bookkeeping
        _decide(a0=set(), a2={"f1"}, i2=set(), cos2={"f1": over - 0.05}, cos_replay={"f1": 0.0})
    assert exc.value.reason == "margin_inconsistent"


def test_overreach_is_what_the_cap_admits_beyond_the_base():
    seq = {0.0: _run(set(), {}), E.CAP: _run(set(), {})}
    replay = {0.0: _replay({"f1"}), E.CAP: _replay({"f1", "f2"})}
    verdict = E.decide(_walk(), seq, replay, BASE)
    assert verdict["I(0.2)"] == ["f1", "f2"] and verdict["widening_overreach"] == ["f2"]
    assert verdict["headroom_count"] == 1


def test_the_environment_stamp_never_copies_a_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert "HF_TOKEN" not in E.STAMPED_ENV
    assert not any(("TOKEN" in k or "KEY" in k or "SECRET" in k) for k in E.STAMPED_ENV)
