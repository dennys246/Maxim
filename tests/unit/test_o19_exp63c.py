"""Exp 63c's harness and judge (frozen prereg ``docs/experiments/exp63c_scripted_recall_prereg.md``; the process
session's rulings on its text: ``docs/experiments/rationale/exp63c-scripted-recall/build-rulings.md``).

Its own module, so it runs on the SHIPPED keying (post-#1620): ``test_o19_rerun.py``'s autouse fixture puts that
module's tests on Exp 63b's executed keying (e841abca) and does not reach here. The 63c port (``c_key_tokens``) is
pinned against the shipped keying in ``test_o19_rerun.py::test_exp63c_the_port_is_the_shipped_keying``. Each wire is
exercised through its production entry point (``complete_problems``, ``judge``, ``_gate_passes``, the harness's
``main``, the smoke), not only through its helper. Nothing here reads PROTOCOL["63c"] at import (the red gates).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import o19_rerun as h  # noqa: E402
import o19_verdict as v  # noqa: E402

REPLAY = REPO / "docs/experiments/rationale/exp63c-scripted-recall/replay"
PREREG_PATH = REPO / "docs/experiments/exp63c_scripted_recall_prereg.md"
GOAL = v.EXP10_GOAL_DUNGEON
_PROVENANCE: dict = {}


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _key():
    return v.keying_of("63c")


# ── the protocol is the prereg's ─────────────────────────────────────────────────────────────────────────


def test_exp63c_protocol_is_the_preregs() -> None:
    text = _squash(PREREG_PATH.read_text())
    p = v.PROTOCOL["63c"]
    assert p["prereg"] == "docs/experiments/exp63c_scripted_recall_prereg.md"
    assert (p["experiment"], p["kind"], p["scope"]) == ("63c", "exp63c_verdict", "63c_scripted_recall")
    assert "kind `exp63c_verdict`" in text and "`refs/tags/o19/63c/`" in text
    assert f"`{v.data_dir('63c')}/`" in text and "supersedes" not in p
    assert f"goal `{GOAL}`; two phases of 8 turns; phase 2 resumes phase 1" in text
    assert f"`{v.MODEL_GGUF}` (`llm.profile {v.MODEL_PROFILE}`) at `llm.n_ctx {v.N_CTX}`" in text
    assert "`--sim-run-full-turns` is dropped" in text
    assert [ph[1:4] for ph in p["phases"]] == [(GOAL, 8, False), (GOAL, 8, True)]
    assert [ph[4] for ph in p["phases"]] == [["--sim-script", path] for path in v.EXP63C_SCRIPTS]
    assert v.phase_argv("63c", 1, "S1") == [
        "--sim", GOAL, "--interactive", "false", "--sim-max-turns", "8",
        "--sim-script", v.EXP63C_SCRIPTS[1], "--resume-sim", "S1",
    ]  # fmt: skip
    assert p["base_preregs"] == [v.PROTOCOL["63b"]["prereg"]]
    assert v.LEAK_GATES["63c"] == v.LEAK_GATES["63b"] and "63c" not in v.PRE_1079_KEYS
    assert v.protocol_problems() == [] and v.required_files("63c") == v.required_files("63b")
    assert v.keying_of("63c") is v.c_key_tokens
    assert "R3d's decisive search runs on the four cue turns (3, 4, 5, 7) only" in text
    assert sorted(v.EXP63C_CUE_TURNS) == [3, 4, 5, 7] and v.EXP63C_CONTROL_TURNS == (2, 6)
    # the pins live in the PROTOCOL entry (the gate's table checks see them); the module names are views of them
    assert v.EXP63C_LINES == tuple(map(tuple, p["script_lines"])) and v.EXP63C_GGUF_SHA256 == p["gguf_sha256"]
    assert v.EXP63C_CONTENT_PINS == tuple(p["scenario_content_sha256"])
    assert f"`{p['gguf_sha256'][:8]}…{p['gguf_sha256'][-4:]}`" in text


def test_a_base_prereg_must_be_another_campaigns_prereg() -> None:
    bad = copy.deepcopy(v.PROTOCOL)
    for value in ([v.PROTOCOL["63c"]["prereg"]], v.PROTOCOL["63b"]["prereg"], ["docs/experiments/nowhere.md"]):
        bad["63c"]["base_preregs"] = value
        assert any("base_preregs" in x for x in v.protocol_problems(bad)), value


def test_bound_blobs_binds_the_base_prereg(monkeypatch) -> None:
    seen: list = []
    monkeypatch.setattr(v, "_git_bytes", lambda *a: seen.append(a[1]) or b"blob\n")
    out = v.bound_blobs("63c", ["c1"])
    assert set(out) == {v.PROTOCOL["63c"]["prereg"], *v.PROTOCOL["63c"]["base_preregs"], *v.BOUND_FILES}
    assert f"c1:{v.PROTOCOL['63b']['prereg']}" in seen


def test_the_evidence_gate_requires_the_base_prereg_in_bound_files() -> None:
    """``_rejudge_with``'s bound-file set is the judge, the harness, the prereg AND its base preregs (else the gate
    would refuse every 63c verdict at the ledger PR, or accept one that bound no 63b prereg)."""
    import _evidence_records as E

    three = {E.O19_JUDGE: "a", E.O19_RERUN: "b", v.PROTOCOL["63c"]["prereg"]: "c"}
    rec = {"experiment": "63c", "kind": "exp63c_verdict", "data": "elsewhere"}
    why = E._rejudge_with(v, rec, [], "", None, [], three, "main")
    assert why.startswith("bound_files names") and "exp63b_carried_recall_prereg.md" in why
    four = {**three, v.PROTOCOL["63c"]["base_preregs"][0]: "d"}
    assert "is not campaign 63c's rows file" in E._rejudge_with(v, rec, [], "", None, [], four, "main")
    assert "exp63c_verdict" in E.O19_KINDS


# ── the scripts, the pins and the vocabulary ─────────────────────────────────────────────────────────────


def test_the_scripts_load_to_the_literals_and_hash_to_the_pins() -> None:
    from maxim.simulation.experiment import load_campaign_turns
    from maxim.simulation.serving_provenance import _content_sha256

    prereg = PREREG_PATH.read_text()
    for index, path in enumerate(v.EXP63C_SCRIPTS):
        strict = load_campaign_turns(str(REPO / path), strict=True)
        assert strict == load_campaign_turns(str(REPO / path))  # the sim's default loader reads the same dicts
        assert tuple(t["text"] for t in strict) == v.EXP63C_LINES[index]
        assert _content_sha256([strict]) == v.EXP63C_CONTENT_PINS[index]
        assert {(t["salience"], t["novelty"], t["phase"]) for t in strict} == {(0.8, 0.7, "")}  # no per-line override
    for line in (*v.EXP63C_LINES[0], *v.EXP63C_LINES[1]):
        assert f"| {line} |" in prereg, line  # each literal is a row of the prereg's script tables


def test_the_script_matches_the_frozen_replay() -> None:
    ns: dict = {}
    exec((REPLAY / "script_v5x.py").read_text(), ns)  # noqa: S102 -- a committed, frozen design file
    assert tuple(line for line, *_ in ns["P1"]) == v.EXP63C_LINES[0]
    assert tuple(line for line, *_ in ns["P2"]) == v.EXP63C_LINES[1]
    assert {t: f"fact {f}" for t, f in v.EXP63C_CUE_TURNS.items()} == ns["CUE_OF"]


def test_the_vocabulary_binds_to_the_frozen_tables() -> None:
    key = _key()
    blacklist = set(json.loads((REPLAY / "blacklist.json").read_text()))
    assert len(blacklist) == 669 and v.EXP63C_GOAL_KEYS <= blacklist  # the blacklist counts the goal keys
    assert set(key(GOAL)) == v.EXP63C_GOAL_KEYS == {"dungeon", "escap", "guard", "sleep"}
    for line in (*v.EXP63C_LINES[0], *v.EXP63C_LINES[1]):
        assert not (set(key(line)) - v.EXP63C_GOAL_KEYS) & blacklist, line  # no non-goal key collides
    for turn, fact in v.EXP63C_CUE_TURNS.items():
        cue = set(key(v.EXP63C_LINES[1][turn - 1]))
        fact_keys = set(key(v.EXP63C_LINES[0][v.EXP63C_FACT_LINES[fact] - 1]))
        answer = set(v.EXP63C_ANSWER_KEYS[fact])
        assert answer <= fact_keys and not answer & cue, fact  # an answer key is the fact's and not its cue's
        assert cue & v.EXP63C_GOAL_KEYS <= fact_keys & v.EXP63C_GOAL_KEYS, fact  # the goal-subset check
        assert cue - v.EXP63C_GOAL_KEYS, fact  # every cue has non-goal keys


# ── preflight (each through the harness's own entry point) ───────────────────────────────────────────────


def test_preflight_checks_the_scripts_and_the_keying() -> None:
    h.check_scripts("63c")
    h.check_keying("63c")
    assert h.check_scripts("63b") is None
    assert h.check_keying("63b") is None  # no-ops for another experiment (63b's port is e841abca's keying)


def _edited_script(tmp_path: Path) -> Path:
    edited = tmp_path / "p1.yaml"
    edited.write_text((REPO / v.EXP63C_SCRIPTS[0]).read_text().replace("brass", "bronze"))
    return edited


def test_check_scripts_refuses_each_way_a_script_can_be_wrong(monkeypatch, tmp_path) -> None:
    from maxim.simulation.experiment import load_campaign_turns
    from maxim.simulation.serving_provenance import _content_sha256

    edited = _edited_script(tmp_path)
    monkeypatch.setattr(v, "EXP63C_SCRIPTS", (str(edited), v.EXP63C_SCRIPTS[1]))
    with pytest.raises(h.Refused, match="not the pinned content"):
        h.check_scripts("63c")
    pinned = _content_sha256([load_campaign_turns(str(edited))])
    monkeypatch.setattr(v, "EXP63C_CONTENT_PINS", (pinned, v.EXP63C_CONTENT_PINS[1]))
    with pytest.raises(h.Refused, match="not PROTOCOL's literals"):
        h.check_scripts("63c")
    monkeypatch.setattr(v, "EXP63C_SCRIPTS", (str(tmp_path / "missing.yaml"), v.EXP63C_SCRIPTS[1]))
    with pytest.raises(h.Refused, match="cannot be loaded"):
        h.check_scripts("63c")


def test_check_scripts_runs_the_clis_own_sim_script_check(monkeypatch) -> None:
    import maxim.cli as cli

    monkeypatch.setattr(cli, "resolve_sim_script", lambda args: (None, "a test refusal"))
    with pytest.raises(h.Refused, match="refused by --sim-script: a test refusal"):
        h.check_scripts("63c")


def test_a_keying_drift_refuses_before_the_marker(monkeypatch) -> None:
    monkeypatch.setitem(v.KEYINGS, "63c", v.p_key_tokens)  # 63b's keying: "moving" no longer keys as "mov"
    with pytest.raises(h.Refused, match="keying port disagrees"):
        h.check_keying("63c")


def test_the_gguf_is_rehashed_at_preflight(monkeypatch, tmp_path) -> None:
    gguf = tmp_path / v.MODEL_GGUF
    gguf.write_bytes(b"not the model")
    with pytest.raises(h.Refused, match="not the pinned"):
        h.check_gguf_identity("63c", str(gguf))
    monkeypatch.setattr(v, "EXP63C_GGUF_SHA256", hashlib.sha256(b"not the model").hexdigest())
    h.check_gguf_identity("63c", str(gguf))
    with pytest.raises(h.Refused, match="cannot be read"):
        h.check_gguf_identity("63c", str(tmp_path / "absent.gguf"))
    assert h.check_gguf_identity("63b", str(tmp_path / "absent.gguf")) is None


def test_run_and_smoke_call_the_gguf_check_after_the_model_check() -> None:
    """A source-order check (the mock path skips the model check, so no ``main --mock`` run can reach it): ``prepare``
    and ``smoke`` each re-hash the configured GGUF after the model check and before the port check."""
    src = Path(h.__file__).read_text()
    for block in (
        src[src.index("def prepare(") : src.index("def run(")],
        src[src.index("def smoke(") : src.index("def main(")],
    ):
        model = block.index('check_model_config(home, exp, run_id=run_id)["model_path"]')
        assert block.index("check_gguf_identity(exp, gguf)") > model  # after the model is known, before the port
        assert block.index("check_gguf_identity(exp, gguf)") < block.index("check_port_free(v.SIM_PORT)")


def _stable_provenance(monkeypatch) -> None:
    if "block" not in _PROVENANCE:
        block = h._provenance.executed_code_provenance(REPO, sys.executable)
        if not (isinstance(block.get("code_tree_sha256"), str) and len(block["code_tree_sha256"]) == 64):
            pytest.skip("the live code tree did not hash")
        _PROVENANCE["block"] = block
    monkeypatch.setattr(
        h._provenance,
        "executed_code_provenance",
        lambda *_a, **_k: {**copy.deepcopy(_PROVENANCE["block"]), **h._provenance._run_id_stamp()},
    )


def _harness(monkeypatch, rows: Path) -> None:
    _stable_provenance(monkeypatch)
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows)


@pytest.mark.parametrize("command", ["run", "smoke"])
@pytest.mark.parametrize("fault", ["script", "keying"])
def test_an_edited_script_or_a_keying_drift_refuses_the_run_and_the_smoke(
    monkeypatch, tmp_path, command, fault
) -> None:
    """Through ``main``: exit 2 and no row (the prereg's "a missing or edited script cannot cost an attempt")."""
    rows = tmp_path / "rows_63c.jsonl"
    _harness(monkeypatch, rows)
    if fault == "script":
        monkeypatch.setattr(v, "EXP63C_SCRIPTS", (str(_edited_script(tmp_path)), v.EXP63C_SCRIPTS[1]))
    else:
        monkeypatch.setitem(v.KEYINGS, "63c", v.p_key_tokens)
    assert h.main([command, "--exp", "63c", "--mock"]) == 2
    assert not rows.exists() or not rows.read_text().strip()


def test_the_63c_sim_runs_from_the_repo_root(monkeypatch, tmp_path) -> None:
    seen: dict = {}

    class _Sim:
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(h.subprocess, "Popen", lambda *a, **k: seen.update(k) or _Sim())
    monkeypatch.setattr(h, "stop_sim", lambda proc: None)
    monkeypatch.setattr(h, "port_accepts", lambda port: False)
    h.spawn_phase("63c", 0, home=tmp_path, run_id="r" * 32, resume=None, gguf="g", timeout_s=60, read_served=None)
    assert seen["cwd"] == h.REPO_ROOT
    h.spawn_phase("63b", 0, home=tmp_path, run_id="r" * 32, resume=None, gguf="g", timeout_s=60, read_served=None)
    assert seen["cwd"] is None


# ── the judge on the real-seam mock ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def exp63c(tmp_path, monkeypatch):
    """One mock 63c attempt through the real harness ``run`` (the shipped Hippocampus, enrichment, persistence and
    log formatter; only the LLM is stubbed). Returns ``(rows_file, verdict, phases)``."""

    def make(variant: str = "pass"):
        rows = tmp_path / "rows_63c.jsonl"
        _harness(monkeypatch, rows)
        monkeypatch.setattr(h, "MOCK63B_VARIANT", variant)
        h.main(["run", "--exp", "63c", "--mock"])
        parsed = [json.loads(ln) for ln in rows.read_text().splitlines()]
        attempts = v.attempts_from_rows(parsed)
        ordered = [{"run_id": rid, "k": i + 1, "rows": rs} for i, (rid, rs) in enumerate(attempts.items())]
        phases = [h._copied_phase(rows.parent, r["session_id"]) for r in parsed]
        return rows, v.judge("63c", ordered, rows.parent), phases

    return make


def _session_dir(rows: Path, session_id: str) -> Path:
    return next(p for p in rows.parent.rglob(session_id) if p.is_dir())


def test_the_real_seam_mock_passes_on_the_cue_turns(exp63c) -> None:
    _rows, out, _phases = exp63c()
    g = out["gates"]
    assert out["verdict"] == "PASS", (g["not_passed"], out["attempts"])
    assert g["R3a"]["status"] == "held" and g["R3a"]["turn"] == 1
    assert g["R3d"]["searched_turns"] == [3, 4, 5, 7] and g["R3d"]["decisive_turns"] == [3, 4, 5, 7]
    d = g["descriptive"]
    assert d["controls"] == {2: "armed", 6: "armed"} and d["k"] == "4 of 4"
    assert d["uncontested"] is True and d["contested_turns"] == []
    for t in (3, 4, 5, 7):
        r = d["cue_turns"][t]
        assert r["winner_is_planted_fact"] and r["completion"] and r["uncontested_literal"], r
        assert r["contested"] is False and r["orders"] and "what_won" not in r
        assert set(r["echo_load"]) == set(r["cue_nongoal_keys"])  # ruling S2: echo over the cue's non-goal keys
        assert set(r["answer_key_load"]) == set(v.EXP63C_ANSWER_KEYS[v.EXP63C_CUE_TURNS[t]])
    # the pool is the turn's view (two new records per earlier turn: an arrival and a reply), not the final store
    assert [d["cue_turns"][t]["displaceable_pool"] for t in (3, 4, 5, 7)] == [4, 6, 8, 12]
    assert [d["cue_turns"][t]["score"] for t in (3, 4, 5, 7)] == ["3 v 0", "3 v 0", "4 v 0", "3 v 0"]
    assert set(d["imagination_events_by_turn"]) == {"baseline", "gate"}


def _complete(rows: Path, index: int, edit) -> list[str]:
    parsed = [json.loads(ln) for ln in rows.read_text().splitlines()]
    row = parsed[index]
    report = json.loads(v.read_copied(_session_dir(rows, row["session_id"]), "report.json"))
    edit(row, report)
    return v.complete_problems("63c", index, row, report, parsed[0]["session_id"], set(v.RESUME_STORES))


def _entry(row: dict) -> dict:
    return row["sims"][0]["llm_calls"]["1"]["entries"][0]


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (lambda row, rep: rep.update(finish_reason="max_turns"), "C1 gate: finish_reason 'max_turns', not 'completed'"),
        (lambda row, rep: rep.update(turns=7), "C1 gate: turns 7, not 8"),
        (lambda row, rep: rep["provenance"].update(scenario_content_sha256="0" * 64), "not the pinned script"),
        (lambda row, rep: _entry(row).update(identity="b" * 64), "S2' gate"),
        (lambda row, rep: _entry(row).update(backend_kind="local"), "S2' gate"),
        (lambda row, rep: _entry(row).update(served_id=""), "S2' gate"),
        (lambda row, rep: row["served_model"]["reads"].insert(1, {"served": "/m/qwen.gguf", "match": False}), "S2'"),
        (lambda row, rep: row.update(memory_strategy="other"), "C4' gate"),
        (lambda row, rep: row["sims"][0].update(aut_model_attribution="established"), "C4'' gate"),
    ],
)
def test_c1_c4_and_s2_prime_reach_the_complete_attempt_condition(exp63c, edit, needle) -> None:
    rows, _out, _phases = exp63c()
    assert _complete(rows, 1, lambda row, rep: None) == []
    found = _complete(rows, 1, edit)
    assert any(needle in x for x in found), found


def test_s2_prime_rules() -> None:
    ok = {"served": f"/m/{v.MODEL_GGUF}", "match": True}
    entry = {"backend_kind": "peer", "identity_source": "local_file", "identity": v.EXP63C_GGUF_SHA256,
             "served_id": f"/m/{v.MODEL_GGUF}"}  # fmt: skip

    def row(calls, reads):
        return {"sims": [{"llm_calls": calls}], "served_model": {"reads": reads}}

    good = {"1": {"counts": {"success": 1}, "entries": [entry]}}
    start, stop = {"served": None, "error": "refused"}, {"served": None, "error": "stopped"}
    assert v.exp63c_serving_problems(row(good, [start, ok, ok, stop]), "p") == []  # outside the span: reported
    inside = v.exp63c_serving_problems(row(good, [ok, {"served": None, "error": "mid"}, ok]), "p")
    assert any("inside the served span" in x for x in inside)  # ruling S5: a failed read inside the span fails
    assert any("no harness" in x for x in v.exp63c_serving_problems(row(good, [start, stop]), "p"))  # zero OK reads
    assert any("no llm_calls" in x for x in v.exp63c_serving_problems(row({}, [ok]), "p"))
    idle = {**good, "2": {"counts": {"success": 0}, "entries": []}}
    assert v.exp63c_serving_problems(row(idle, [ok]), "p") == []  # a router that served nothing has nothing to judge
    busy = {**good, "2": {"counts": {"success": 3}, "entries": []}}
    assert any("served calls but records no entry" in x for x in v.exp63c_serving_problems(row(busy, [ok]), "p"))
    none = {"2": {"counts": {"success": 0}, "entries": []}}
    assert any("no call entry" in x for x in v.exp63c_serving_problems(row(none, [ok]), "p"))
    judged, unjudged = v._served_span(row(good, [start, ok, stop]))
    assert judged == [ok] and unjudged == [start, stop]


# ── the judge's dispatch and its gate rules ──────────────────────────────────────────────────────────────


def test_judge_and_the_leak_bars_dispatch_63c_with_its_keying(exp63c, monkeypatch) -> None:
    calls: list = []
    real_c5, real_gates = v.exp63c_c5_problems, v.exp63c_gates
    monkeypatch.setattr(v, "exp63c_c5_problems", lambda *a, **k: calls.append(("c5", k["key"])) or real_c5(*a, **k))
    monkeypatch.setattr(v, "exp63c_gates", lambda *a, **k: calls.append(("gates", k["key"])) or real_gates(*a, **k))
    _rows, out, phases = exp63c()
    assert out["verdict"] == "PASS" and ("c5", v.c_key_tokens) in calls and ("gates", v.c_key_tokens) in calls
    calls.clear()
    assert v._gate_passes("63c", phases)["R3d"] is True and calls == [("gates", v.c_key_tokens)]


def test_the_63c_judge_reads_the_keying_table_not_a_hard_coded_port(exp63c, monkeypatch) -> None:
    """Swapping KEYINGS["63c"] changes what the judge computes (63b's keying makes C5(b) fail: the traces' keywords
    are the shipped keys), so preflight's check_keying guards the port the judge actually calls."""
    _rows, _out, phases = exp63c()
    assert v.exp63c_c5_problems(phases, goal=GOAL, key=v.keying_of("63c"), verdict_time=True) == []
    monkeypatch.setitem(v.KEYINGS, "63c", v.p_key_tokens)
    assert any(
        x.startswith("C5(b)")
        for x in v.exp63c_c5_problems(phases, goal=GOAL, key=v.keying_of("63c"), verdict_time=True)
    )


def test_r3d_counts_only_cue_turns_after_a_held_r3a(exp63c, monkeypatch) -> None:
    _rows, _out, phases = exp63c()
    real_r3a = v.exp63b_r3a
    monkeypatch.setattr(v, "exp63b_r3a", lambda *a: {**real_r3a(*a), "turn": 4})
    g = v.exp63c_gates(phases, goal=GOAL, key=_key())
    assert g["R3d"]["decisive_turns"] == [5, 7]  # turns 3 and 4 are at or before R3a's view
    monkeypatch.setattr(v, "exp63b_r3a", lambda *a: {**real_r3a(*a), "status": "unobservable", "pass": False})
    g = v.exp63c_gates(phases, goal=GOAL, key=_key())
    assert g["R3d"] == {"decisive_turns": [], "pass": False, "searched_turns": [3, 4, 5, 7]}
    assert g["verdict"] == "NOT SHOWN" and "R3d" in g["not_passed"]


def test_a_control_alone_decisive_is_not_shown_and_contaminated_is_never_c5(exp63c, monkeypatch) -> None:
    _rows, _out, phases = exp63c()
    real = v.exp63b_turn

    def only_control(tr, *a, **k):
        return {**real(tr, *a, **k), "decisive": tr.get("cue_text") == v.EXP63C_LINES[1][1]}  # turn 2's control

    monkeypatch.setattr(v, "exp63b_turn", only_control)
    g = v.exp63c_gates(phases, goal=GOAL, key=_key())
    assert g["R3d"]["decisive_turns"] == [] and g["verdict"] == "NOT SHOWN"
    assert any(
        x.startswith("C5(k) turn 2") for x in v.exp63c_c5_problems(phases, goal=GOAL, key=_key(), verdict_time=True)
    )
    monkeypatch.setattr(v, "exp63c_control_status", lambda *a, **k: "contaminated")
    assert not any(
        x.startswith("C5(k)") for x in v.exp63c_c5_problems(phases, goal=GOAL, key=_key(), verdict_time=True)
    )


def test_c5_i_prime_checks_both_halves_in_both_phases(exp63c) -> None:
    _rows, _out, phases = exp63c()
    for index in (0, 1):
        line = v.EXP63C_LINES[index][2]
        p = copy.deepcopy(phases)
        for r in p[index]["lines"]:
            if r.get("e") == "enrichment_trace" and r.get("cue_text") == line:
                r["cue_text"] = "something else"
        assert any(
            "turn 3: cue_text is not script line 3" in x
            for x in v.exp63c_c5_problems(p, goal=GOAL, key=_key(), verdict_time=True)
        )
        p = copy.deepcopy(phases)
        for r in p[index]["lines"]:
            if "ENTER turn=3 " in str(r.get("message", "")):
                r["message"] = r["message"].replace(f"text_len={len(line)}", "text_len=1")
        assert any(
            "turn 3: the bridge sent 1 characters" in x
            for x in v.exp63c_c5_problems(p, goal=GOAL, key=_key(), verdict_time=True)
        )


def test_c5_q_fires_on_a_decisive_pair_without_a_shared_key_and_never_in_the_smoke(exp63c, monkeypatch) -> None:
    """C5(q) guards against a broken port (under G4 a decisive pair always shares a key): forced here."""
    _rows, _out, phases = exp63c()
    real = v.exp63b_turn
    carried = v.store_records(phases[0]["store"])
    neutral = next(m for m, rec in carried.items() if v.p_input_text(rec) == v.EXP63C_LINES[0][6])  # no cue key

    def broken(tr, *a, **k):
        x = real(tr, *a, **k)
        if k.get("explain"):
            x["orders"] = [{"displaced": ["d"], "displaced_scores": {"d": 0}, "pairs": [[neutral, "d", 1, 0]]}]
        return {**x, "decisive": True}

    monkeypatch.setattr(v, "exp63b_turn", broken)
    assert any(x.startswith("C5(q)") for x in v.exp63c_c5_problems(phases, goal=GOAL, key=_key(), verdict_time=True))
    smoke = v.exp63c_c5_problems(phases, goal=GOAL, key=_key(), r3a_view=False, verdict_time=False)
    assert not any(x.startswith(("C5(q)", "C5(k)")) for x in smoke)  # ruling S1


def test_the_smoke_never_computes_a_decisive_status(exp63c, monkeypatch) -> None:
    _rows, _out, phases = exp63c()
    seen: list = []
    real = v.exp63b_turn
    monkeypatch.setattr(v, "exp63b_turn", lambda *a, **k: seen.append(k.get("decide", True)) or real(*a, **k))
    v.exp63c_c5_problems(phases, goal=GOAL, key=_key(), r3a_view=False, verdict_time=False)
    assert seen and not any(seen)  # every call decide=False


def test_the_harness_attempt_check_routes_63c_c5_with_its_keying_and_phase(exp63c, monkeypatch) -> None:
    rows, _out, _phases = exp63c()
    seen: list = []
    real = v.exp63c_c5_problems
    monkeypatch.setattr(v, "exp63c_c5_problems", lambda *a, **k: seen.append(k) or real(*a, **k))
    sessions = [json.loads(ln)["session_id"] for ln in rows.read_text().splitlines()]
    data_root = _session_dir(rows, sessions[0]).parent
    assert h.exp63_attempt_problems("63c", data_root, sessions, smoke=True) == []
    assert h.exp63_attempt_problems("63c", data_root, sessions) == []
    assert [(k["verdict_time"], k["key"]) for k in seen] == [(False, v.c_key_tokens), (True, v.c_key_tokens)]


def test_a_non_decisive_cue_reports_what_won(exp63c, monkeypatch) -> None:
    _rows, _out, phases = exp63c()
    real = v.exp63b_turn
    monkeypatch.setattr(v, "exp63b_turn", lambda *a, **k: {**real(*a, **k), "decisive": False})
    won = v.exp63c_gates(phases, goal=GOAL, key=_key())["descriptive"]["cue_turns"][3]["what_won"]
    assert won and all(len(w) == 4 and isinstance(w[1], bool) for w in won)  # [id, carried, non-goal score, ts]


def test_contested_and_the_literal_uncontested_rule_read_every_valid_order(exp63c, monkeypatch) -> None:
    """Ruling S3: a turn is contested when ANY valid order's best pair is; the PASS wording's "uncontested" holds
    only when no displaced id in any valid order holds a cue non-goal key."""
    _rows, _out, phases = exp63c()
    real = v.exp63b_turn

    def with_orders(orders):
        def fake(tr, *a, **k):
            x = real(tr, *a, **k)
            if k.get("explain") and x.get("orders") and tr.get("cue_text") == v.EXP63C_LINES[1][2]:
                c = x["orders"][0]["pairs"][0][0]
                x["orders"] = [{**o, "pairs": [[c, *pr[1:]] for pr in o["pairs"]]} for o in orders]
            return x

        return fake

    # S3 i: a second order whose best pair is contested makes the turn contested
    contested = [{"displaced": ["d1"], "displaced_scores": {"d1": 0}, "pairs": [["c", "d1", 3, 0]]},
                 {"displaced": ["d2"], "displaced_scores": {"d2": 1}, "pairs": [["c", "d2", 3, 1]]}]  # fmt: skip
    monkeypatch.setattr(v, "exp63b_turn", with_orders(contested))
    d = v.exp63c_gates(phases, goal=GOAL, key=_key())["descriptive"]
    assert d["cue_turns"][3]["contested"] is True and d["cue_turns"][3]["uncontested_literal"] is False
    assert d["uncontested"] is False and 3 in d["contested_turns"]
    # S3 ii: the best pair is "3 v 0" (uncontested) but another displaced id holds a cue key: NOT literally uncontested
    split = [{"displaced": ["d1", "d2"], "displaced_scores": {"d1": 0, "d2": 1},
              "pairs": [["c", "d1", 3, 0], ["c", "d2", 3, 1]]}]  # fmt: skip
    monkeypatch.setattr(v, "exp63b_turn", with_orders(split))
    d = v.exp63c_gates(phases, goal=GOAL, key=_key())["descriptive"]
    r = d["cue_turns"][3]
    assert r["score"] == "3 v 0" and r["contested"] is False and r["uncontested_literal"] is False
    assert d["uncontested"] is False and d["contested_turns"] == [3]  # the PASS wording names the turn


def test_explain_covers_every_valid_tie_order_of_the_real_function() -> None:
    """X and Y tie exactly below the cut (63b's G4 case): removing the carried C lets either in, so there are two valid
    orders, and explain= reports both: displacing X (no cue non-goal key) and displacing Y (one)."""
    key = _key()

    def rec(rid, text, ts, seq):
        return {"id": rid, "timestamp": ts, "capture_seq": seq, "perception": {"cli_input": text, "observations": {}},
                "context": {"active_goal": None}, "action": {"tool_name": ""}, "outcome": {}}  # fmt: skip

    records = [rec("A", "the iron door", 20, 0), rec("B", "an iron door gate", 21, 1), rec("C", "a door", 5, 2),
               rec("X", "the guard", 3, 3), rec("Y", "the door", 3, 4)]  # fmt: skip
    recs = {r["id"]: r for r in records}
    tr = {"memory_ids": ["B", "A", "C"], "memory_paths": ["cue"] * 3, "own_ids": ["own"],
          "cue_text": "where is the iron door guard", "goal_path_horizon": 4, "goal_path_holes": [], "goal": GOAL}  # fmt: skip
    x = v.exp63b_turn(tr, recs, {"C"}, GOAL, key=key, explain=True)
    assert x["conforms"] and len(x["orders"]) == 2, x.get("orders")
    assert sorted(tuple(o["displaced"]) for o in x["orders"]) == [("X",), ("Y",)]
    assert {d: s for o in x["orders"] for d, s in o["displaced_scores"].items()} == {"X": 0, "Y": 1}


# ── the prereg's deletion probes (build order step 4) ────────────────────────────────────────────────────


@pytest.mark.parametrize("variant,gate", [("no_exclusion", "R3prime"), ("scrambled", "R2prime")])
def test_probe_a_deleted_mechanism_fails_its_gate(exp63c, variant, gate) -> None:
    """The own-capture exclusion removed, a restored store that is not phase 1's bytes: complete attempts that FAIL on
    their gate (as 63b's probes do), never a PASS."""
    _rows, out, _phases = exp63c(variant)
    assert out["attempts"][0]["complete"], out["attempts"][0]["problems"]
    assert out["verdict"] == "FAIL" and gate in out["gates"]["not_passed"], out["gates"]["not_passed"]


def test_probe_a_control_word_planted_in_a_carried_record_contaminates_it(exp63c) -> None:
    _rows, _out, phases = exp63c()
    p = copy.deepcopy(phases)
    store = p[0]["store"]
    mems = store["memories"] if isinstance(store, dict) else store
    mems[0]["perception"]["cli_input"] = "a silver bell rings in the chapel"
    controls = v.exp63c_gates(p, goal=GOAL, key=_key())["descriptive"]["controls"]
    assert controls == {2: "contaminated", 6: "armed"}


def test_probe_a_fact_without_its_goal_word_ties_on_the_full_cue() -> None:
    """Confounding S1: arrival records hold no goal word, the AUT's do; drop the fact's goal word and an echo holding
    every non-goal cue key plus the goal word outranks it on the full cue; with it, they tie and the newer wins."""
    key = _key()
    cue = v.EXP63C_LINES[1][2]  # cue A
    fact = {"id": "F", "timestamp": 1.0, "perception": {"cli_input": "A brass ladle hangs from a copper hook."}}
    echo = {"id": "E", "timestamp": 2.0, "perception": {"cli_input": "the guard: brass ladle hang"}}
    with_goal = {"id": "G", "timestamp": 1.0, "perception": {"cli_input": v.EXP63C_LINES[0][1]}}
    recs = {r["id"]: r for r in (fact, echo, with_goal)}
    tokens = set(key(cue))

    def score(rid: str) -> float:
        return len(tokens & v.p_record_tokens(recs[rid], key=key)) / len(tokens)

    assert score("E") > score("F")
    assert score("E") == score("G") == 1.0
    assert [m for g in v.p_ranked_groups(recs, ["G", "E"], cue, key=key) for m in g] == ["E", "G"]


# ── the smoke ────────────────────────────────────────────────────────────────────────────────────────────


def test_the_smoke_runs_63cs_checks_and_never_prints_r3a_or_r3d(monkeypatch, capsys, tmp_path) -> None:
    _harness(monkeypatch, tmp_path / "unused.jsonl")
    monkeypatch.setattr(h, "MOCK63B_VARIANT", "pass")
    assert h.main(["smoke", "--exp", "63c", "--mock"]) == 0, capsys.readouterr()
    out = capsys.readouterr().out
    assert "[STOP]" not in out, out
    for label in ("eight ENTER lines", "zero narrator submits", "no LLM call in flight", "peak prompt is under n_ctx"):
        assert out.count(label) == 2, (label, out)  # once per phase
    for report in ("turn seconds", "AUT sim_exec lines by turn", "imagination and designer events by turn",
                   "holding a control or planted key", "facts surfaced before their cue", "hippocampus_size by turn",
                   v.EXP63C_VERDICT_TIME_ONLY):  # fmt: skip
        assert report in out, report
    assert "R3a" not in out and "R3d" not in out and "decisive" not in out


def test_the_smoke_stops_on_a_narrator_submit_and_a_call_in_flight() -> None:
    phase = {"lines": [
        {"e": "sim_exec", "agent_id": "sim_orchestrator", "t": 10.0, "message": "Bridge.send_and_wait ENTER turn=1 text_len=3"},
        {"e": "sim_exec", "agent_id": "sim_orchestrator", "t": 11.0, "message": "LLM submit: followup"},
        {"e": "peer_backend_call", "t": 12.0, "latency_ms": 5000.0, "input_tokens": 9000},
    ], "store": {"memories": []}}  # fmt: skip
    found: list = []
    h.smoke_checks_63c([phase, phase], lambda label, ok, detail="": found.append((label, bool(ok))), [])
    bad = {label for label, ok in found if not ok}
    assert {"phase 1: zero narrator submits", "phase 1: no LLM call in flight at any ENTER",
            "phase 1: the peak prompt is under n_ctx", "phase 1: eight ENTER lines, one per script line"} <= bad  # fmt: skip


def test_the_smoke_through_main_never_decides_a_turn(monkeypatch, capsys, tmp_path) -> None:
    """Ruling S1 on the production path: every turn decision the smoke makes is decide=False (no R3d), so a forgotten
    verdict_time=False in smoke_checks or exp63_attempt_problems goes red here."""
    _harness(monkeypatch, tmp_path / "unused.jsonl")
    monkeypatch.setattr(h, "MOCK63B_VARIANT", "pass")
    seen: list = []
    real = v.exp63b_turn
    monkeypatch.setattr(v, "exp63b_turn", lambda *a, **k: seen.append(k.get("decide", True)) or real(*a, **k))
    assert h.main(["smoke", "--exp", "63c", "--mock"]) == 0, capsys.readouterr()
    assert seen and not any(seen), f"{sum(seen)} of {len(seen)} turn calls decided"


def test_the_smokes_echo_count_is_nac_and_atl_only() -> None:
    lines = [{"e": "sim_exec", "agent_id": "x", "t": 1.0, "message": "Bridge.send_and_wait ENTER turn=1 text_len=3"},
             {"e": "sim_enrichment", "system": "nac", "t": 2.0}, {"e": "sim_enrichment", "system": "hippocampus", "t": 2.0}]  # fmt: skip
    reported: list = []
    phase = {"lines": lines, "store": {"memories": []}}
    h.smoke_checks_63c([phase, phase], lambda *a, **k: None, reported)
    assert any("NAc/ATL enrichment lines by turn (echo channels): {1: 1}" in r for r in reported), reported


# ── the second delta round's pins (each guard went green when reverted) ─────────────────────────────────


def test_c5_q_fires_when_any_one_valid_order_lacks_a_shared_key(exp63c, monkeypatch) -> None:
    _rows, _out, phases = exp63c()
    real = v.exp63b_turn
    carried = v.store_records(phases[0]["store"])
    neutral = next(m for m, rec in carried.items() if v.p_input_text(rec) == v.EXP63C_LINES[0][6])  # no cue key

    def two_orders(tr, *a, **k):
        x = real(tr, *a, **k)
        if k.get("explain") and x.get("orders") and x["orders"][0]["pairs"]:
            sound = x["orders"][0]
            x["orders"] = [sound, {"displaced": ["d"], "displaced_scores": {"d": 0}, "pairs": [[neutral, "d", 1, 0]]}]
        return x

    monkeypatch.setattr(v, "exp63b_turn", two_orders)
    assert any(x.startswith("C5(q)") for x in v.exp63c_c5_problems(phases, goal=GOAL, key=_key(), verdict_time=True))


def test_a_base_prereg_is_never_the_campaigns_own_even_when_another_campaign_shares_it() -> None:
    bad = copy.deepcopy(v.PROTOCOL)
    bad["63d"] = {**copy.deepcopy(v.PROTOCOL["63c"]), "scope": "63d_x", "base_preregs": []}  # shares 63c's prereg
    bad["63c"]["base_preregs"] = [v.PROTOCOL["63c"]["prereg"]]
    assert any("campaign 63c: base_preregs" in x for x in v.protocol_problems(bad))


def test_the_smoke_reports_an_answer_key_in_an_imagination_line() -> None:
    lines = [{"e": "sim_exec", "agent_id": "x", "t": 1.0, "message": "Bridge.send_and_wait ENTER turn=1 text_len=3"},
             {"e": "sim_imagination", "t": 2.0, "message": "a copper pot in the scene"}]  # fmt: skip
    reported: list = []
    phase = {"lines": lines, "store": {"memories": []}}
    h.smoke_checks_63c([phase, phase], lambda *a, **k: None, reported)
    assert any("holding a control or planted key (never C5): ['copper']" in r for r in reported), reported


def test_verdict_time_is_required() -> None:
    import inspect

    assert inspect.signature(v.exp63c_c5_problems).parameters["verdict_time"].default is inspect.Parameter.empty
