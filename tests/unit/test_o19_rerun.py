"""O19 re-run harness + verdict (scripts/o19_rerun.py, scripts/o19_verdict.py).

The verdict decides T1-1 (Exp 10) and T3-9 (Exp 09) from committed bytes, so each rule the preregs state is pinned
here against a perturbation that must flip it, and the log readers against the committed 2026-09-24 / 09-27 logs.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import o19_rerun as h  # noqa: E402
import o19_verdict as v  # noqa: E402

PREREG = {exp: (REPO / v.PROTOCOL[exp]["prereg"]).read_text() for exp in v.PROTOCOL}
DATA_ROOT = REPO / "docs/experiments/data"  # every campaign's committed data directory (the leaked-gate bar reads it)
_EVIDENCE_OUT_PATH = (
    h._provenance.evidence_out_path
)  # the fixture redirects the harness's rows; the verdict needs the real one
_REAL_PROVENANCE = h._provenance.executed_code_provenance
_PROVENANCE: dict = {}  # one real provenance block per test session (#1081 item 6)


def stable_provenance(monkeypatch) -> None:
    """Make a mock attempt independent of concurrent writes to the live worktree (#1081 item 6): the harness's
    provenance hashes the live tree (``code_tree_sha256``), so a file created or removed mid-hash makes it
    "unknown" and the mock run exits 2. The block is captured ONCE per session from the real function (retried,
    and the test skipped if the tree never hashes), then served with a live run id. The provenance tests themselves
    stay on the live tree."""
    if "block" not in _PROVENANCE:
        for _ in range(3):
            block = _REAL_PROVENANCE(REPO, sys.executable)
            if isinstance(block.get("code_tree_sha256"), str) and len(block["code_tree_sha256"]) == 64:
                _PROVENANCE["block"] = block
                break
        else:
            pytest.skip(f"the live code tree never hashed ({block.get('code_tree_sha256')!r})")

    def fixed(*_a, **_k) -> dict:
        return {**copy.deepcopy(_PROVENANCE["block"]), **h._provenance._run_id_stamp()}

    monkeypatch.setattr(h._provenance, "executed_code_provenance", fixed)


# ── the protocol is the prereg's ─────────────────────────────────────────────────────────────────────────


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_goals_caps_and_scope_are_the_preregs() -> None:
    p09 = _squash(PREREG["09"])
    for key in ("10", "10c2"):
        p10 = _squash(PREREG[key])
        assert f"`{v.EXP10_GOAL_DUNGEON}` | 8 | none" in p10 and f"`{v.EXP10_GOAL_DUNGEON}` | 8 | phase 1" in p10
        assert f"`{v.EXP10_GOAL_GARDEN}` | 5 | phase 1" in p10
    assert f'*"{v.EXP09_GOAL}"*' in p09
    assert "`--embodiment bodies/base_humanoid`, `--sim-max-turns 8`, `--sim-run-full-turns`" in p09
    assert v.PROTOCOL["09"]["phases"][0][4] == ["--embodiment", "bodies/base_humanoid", "--sim-run-full-turns"]
    for exp in (k for k in v.PROTOCOL if v.experiment_of(k) != "63"):  # Exp 63's prereg: test_exp63_protocol_*
        assert f"**Scope:** `{v.PROTOCOL[exp]['scope']}`" in PREREG[exp]
        assert f"--data {v.rows_path(exp)}" in _squash(PREREG[exp]).replace("\\ ", "")
        assert f"`{v.MODEL_GGUF}`" in PREREG[exp] and f"`llm.profile {v.MODEL_PROFILE}`" in PREREG[exp]
        assert f"`llm.n_ctx {v.N_CTX}`" in PREREG[exp]
        assert f"`{v.RULESET_INCLUDE}`" in PREREG[exp] and f"`{v.MARKER_NAMESPACE}/<exp>/attempt-<k>-" in PREREG[exp]
        assert "At most 3 attempts" in PREREG[exp] and v.MAX_ATTEMPTS == 3
        assert (
            f"`{v.MODEL_PROFILE_STAMPED}`" in PREREG[exp]
            and "**The rig stays at the first attempt's commit**" in PREREG[exp]
        )


def test_phase_argv_is_the_original_command() -> None:
    assert v.phase_argv("10", 1, "S1") == [
        "--sim", v.EXP10_GOAL_DUNGEON, "--interactive", "false", "--sim-max-turns", "8", "--resume-sim", "S1"
    ]  # fmt: skip
    assert v.phase_argv("09", 0, None)[-3:] == ["--embodiment", "bodies/base_humanoid", "--sim-run-full-turns"]
    with pytest.raises(ValueError):
        v.phase_argv("10", 2, None)


def test_resume_stores_are_the_reports() -> None:
    from maxim.simulation.report import RESUME_STORES

    assert v.RESUME_STORES == RESUME_STORES


# ── the log readers, on committed logs (known answers) ───────────────────────────────────────────────────

E09 = REPO / "docs/experiments/data/rerun_exp09_2026-09-24/20260924_095451"
E10 = REPO / "docs/experiments/data/rerun_exp10_2026-09-27"


def test_exp09_gates_on_the_2026_09_24_log() -> None:
    data = v.read_copied(E09, "run_log.jsonl")
    g = v.exp09_gates({"lines": v.log_lines(data)}, data)
    assert g["H1"] == {"status": "PASS", "attack_flinch_records": 4}
    assert g["H2"] == {"status": "NOT MET", "components": ["torso"]}  # as the 09-24 README recorded
    assert g["H3"]["status"] == "NOT MEASURED"
    assert g["H4"]["attack_flinch_intensities"] == [0.15, 0.133, 0.115, 0.107] and g["H4"]["status"] == "PASS"
    assert g["H5"]["status"] == "PASS" and g["H5"]["at"] == {"reflex": "fire_burn", "n": 2}
    assert g["H6"] == {"status": "PASS", "occurrences": 0} and g["H7"]["status"] == "PASS"
    assert g["verdict"] == "PARTIAL" and g["not_passed"] == ["H2", "H3"]


def test_turn_windows_and_traces_on_the_2026_09_27_logs() -> None:
    lines = v.log_lines(v.read_copied(E10 / "20260927_113807", "run_log.jsonl"))
    assert sorted(v.turn_windows(lines)) == [1, 2, 3]
    traces = [(r["memories"], r["hippocampus_size"]) for r in lines if r.get("e") == "enrichment_trace"]
    assert traces == [(0, 0), (3, 13), (3, 55)]
    store = json.loads(v.read_copied(E10 / "20260927_121056", "aut_hippocampus.json"))
    assert len(v.hippocampus_ids(store)) == 136


def test_h5_bound_is_exact_and_conservative() -> None:
    # (0.150 - 0.0005)(1 + 0.3*0) / 0.15 < 1: a factor that only rounding could put above 1 does not pass.
    assert v.h5_lower_bound(0.15, 0.15, 0) == Fraction(1495, 1500)
    assert v.h5_lower_bound(0.161, 0.15, 0) > 1


# ── markers, ruleset, ordering ───────────────────────────────────────────────────────────────────────────

RID = ["a" * 32, "b" * 32, "c" * 32, "d" * 32]


def _ls(*entries: tuple[int, str, bool]) -> str:
    out = []
    for k, rid, annotated in entries:
        ref = f"refs/tags/o19/10/attempt-{k}-{rid}"
        out.append(f"{'1' * 40}\t{ref}")
        if annotated:
            out.append(f"{'2' * 40}\t{ref}^{{}}")
    return "\n".join(out)


def test_markers_parse_and_refuse() -> None:
    got = v.parse_markers("10", _ls((1, RID[0], True), (2, RID[1], True)))
    assert {r: m["k"] for r, m in got.items()} == {RID[0]: 1, RID[1]: 2}
    assert v.parse_markers("09", _ls((1, RID[0], True))) == {}  # another exp's namespace
    for bad in (
        _ls((1, RID[0], False)),  # lightweight
        _ls((1, RID[0], True), (3, RID[1], True)),  # gap
        _ls((1, RID[0], True), (1, RID[1], True)),  # duplicate k
        _ls(*[(i + 1, RID[i], True) for i in range(4)]),  # more than 3
        "1" * 40 + "\trefs/tags/o19/10/attempt-1-xyz\n" + "2" * 40 + "\trefs/tags/o19/10/attempt-1-xyz^{}",
    ):
        with pytest.raises(v.Refusal):
            v.parse_markers("10", bad)


GOOD_RULESET = {
    "id": 7,
    "target": "tag",
    "enforcement": "active",
    "conditions": {"ref_name": {"include": [v.RULESET_INCLUDE], "exclude": []}},
    "rules": [{"type": "deletion"}, {"type": "update"}],
    "bypass_actors": [],
    "current_user_can_bypass": "never",
    "created_at": "2026-10-01T00:00:00Z",
    "updated_at": "2026-10-01T00:00:00Z",
}
FIRST_MARKER = v._iso("2026-10-02T00:00:00Z")


def _ruleset(change: dict | None = None, history: list[dict] | None = None, extra: list[dict] | None = None):
    d = {**copy.deepcopy(GOOD_RULESET), **(change or {})}
    rulesets = [{"id": d["id"]}] + [{"id": e["id"]} for e in extra or []]
    details = {d["id"]: d, **{e["id"]: e for e in extra or []}}
    return v.ruleset_problems(rulesets, details, {d["id"]: history or []}, FIRST_MARKER, v._iso)


def test_ruleset_accepts_the_owner_ruleset() -> None:
    assert _ruleset() == []
    assert _ruleset(history=[{"version_id": 1, "updated_at": "2026-10-01T00:00:00Z"}]) == []


@pytest.mark.parametrize(
    "change,history,extra",
    [
        ({"enforcement": "disabled"}, None, None),
        ({"rules": [{"type": "deletion"}]}, None, None),
        ({"bypass_actors": [{"actor_id": 5}]}, None, None),
        ({"current_user_can_bypass": "always"}, None, None),
        ({"created_at": "2026-10-03T00:00:00Z"}, None, None),  # created after the first marker (re-created)
        ({"updated_at": "2026-10-03T00:00:00Z"}, None, None),
        (None, [{"version_id": 2, "updated_at": "2026-10-02T00:00:01Z"}], None),  # edited after the first marker
        ({"conditions": {"ref_name": {"include": ["refs/tags/**"], "exclude": []}}}, None, None),
        (None, None, [{**GOOD_RULESET, "id": 8}]),  # two matching rulesets
        ({"target": "branch"}, None, None),
    ],
)
def test_ruleset_refuses(change, history, extra) -> None:
    assert _ruleset(change, history, extra)


def test_ruleset_missing_bypass_field_refuses() -> None:
    d = copy.deepcopy(GOOD_RULESET)
    del d["current_user_can_bypass"]
    assert v.ruleset_problems([{"id": 7}], {7: d}, {7: []}, FIRST_MARKER, v._iso)


def _order(*attempts: tuple) -> list[dict]:
    return [
        {"run_id": rid, "start": start, "landed": landed, "has_rows": first is not None, "first_ts": first}
        for rid, start, landed, first in attempts
    ]


def test_ordering() -> None:
    assert v.ordering_problems(_order(("a", 1.0, 10.0, 5.0), ("b", 11.0, 20.0, 12.0))) == []
    assert v.ordering_problems(_order(("a", 1.0, 12.0, 5.0), ("b", 11.0, 20.0, 12.0)))  # landed after b's marker
    assert v.ordering_problems(_order(("a", 1.0, None, 5.0), ("b", 11.0, 20.0, 12.0)))  # a's rows never reached main
    assert v.ordering_problems(_order(("a", 6.0, 10.0, 5.0)))  # a row older than its own marker
    # A marker with no rows (the harness died after the push) is an aborted attempt, not a refusal (#review 4a).
    assert v.ordering_problems(_order(("a", 1.0, 10.0, 5.0), ("b", 11.0, None, None), ("c", 30.0, 40.0, 31.0))) == []


def test_history_is_append_only() -> None:
    assert v.history_problems([b"a\n", b"a\nb\n"], b"a\nb\n") == []
    assert v.history_problems([b"a\n", b"b\n"], b"b\n")  # attempt a dropped (#review 1)
    assert v.history_problems([b"a\n", b"", b"a\nb\n"], b"a\nb\n")  # deleted, then re-added
    assert v.history_problems([b"a\n"], b"a\nb\n")  # judged bytes not on main
    assert v.history_problems([], b"")


# ── the complete-attempt condition and the judge, end to end on a mock attempt ───────────────────────────


@pytest.fixture
def mock_attempt(tmp_path, monkeypatch):
    """One mock attempt through the real harness ``run`` (rows file redirected into tmp_path)."""

    def make(exp: str) -> Path:
        stable_provenance(monkeypatch)
        monkeypatch.setattr(h._provenance, "_RUN_ID", {})
        rows = tmp_path / f"rows_{exp}.jsonl"
        monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows)
        assert h.main(["run", "--exp", exp, "--mock"]) == 0
        return rows

    return make


def _judge(exp: str, rows_file: Path) -> dict:
    rows = [json.loads(ln) for ln in rows_file.read_text().splitlines()]
    attempts = v.attempts_from_rows(rows)
    ordered = [{"run_id": rid, "k": i + 1, "rows": rs} for i, (rid, rs) in enumerate(attempts.items())]
    return v.judge(exp, ordered, rows_file.parent)


def test_mock_attempts_pass_end_to_end(mock_attempt) -> None:
    assert _judge("10", mock_attempt("10"))["verdict"] == "PASS"
    out = _judge("09", mock_attempt("09"))
    assert out["verdict"] == "PARTIAL" and out["gates"]["not_passed"] == ["H3"]  # the pre-registered ceiling


def test_a_tampered_copy_refuses(mock_attempt) -> None:
    rows = mock_attempt("10")
    session = json.loads(rows.read_text().splitlines()[1])["session_id"]
    report = rows.parent / session / "report.json"
    report.write_text(report.read_text().replace('"max_turns"', '"max_turns" '))
    with pytest.raises(v.Refusal):
        _judge("10", rows)


def test_an_attempt_after_the_complete_one_refuses(mock_attempt) -> None:
    rows = mock_attempt("09")
    attempt = [json.loads(ln) for ln in rows.read_text().splitlines()]
    later = copy.deepcopy(attempt)
    for r in later:
        r["provenance"]["harness_run_id"] = "f" * 32
    with pytest.raises(v.Refusal):
        v.judge(
            "09", [{"run_id": "x", "k": 1, "rows": attempt}, {"run_id": "f" * 32, "k": 2, "rows": later}], rows.parent
        )


def test_aborted_attempts_give_abort(mock_attempt) -> None:
    rows = mock_attempt("10")
    attempt = [json.loads(ln) for ln in rows.read_text().splitlines()]
    attempt[1]["status"] = "failed"
    out = v.judge(
        "10", [{"run_id": "x", "k": 1, "rows": attempt[:2]}, {"run_id": "y", "k": 2, "rows": []}], rows.parent
    )
    assert out["verdict"] == "ABORT" and [a["complete"] for a in out["attempts"]] == [False, False]


def _p1(rows_file: Path) -> str:
    return json.loads(rows_file.read_text().splitlines()[0])["session_id"]


def _phase(rows_file: Path, index: int) -> tuple[dict, dict]:
    row = json.loads(rows_file.read_text().splitlines()[index])
    return row, json.loads((rows_file.parent / row["session_id"] / "report.json").read_text())


PERTURB = [
    ("C1", lambda row, rep: rep.update(finish_reason="planning_failed")),
    ("C1", lambda row, rep: rep.update(turns=7)),
    ("C2", lambda row, rep: rep["provenance"]["resume"].update(resume_loaded=False)),
    ("C2", lambda row, rep: rep["provenance"]["resume"]["stores"].update(nac="failed:ValueError")),
    ("C2", lambda row, rep: rep["provenance"]["resume"].update(resumed_from_session="other")),
    ("C3", lambda row, rep: rep["provenance"].update(working_tree_dirty_src_scripts=True)),
    ("C3", lambda row, rep: rep["provenance"].update(code_changed_during_run=True)),
    ("C3", lambda row, rep: rep["provenance"].update(end_code_tree_sha256="0" * 64)),
    ("C3", lambda row, rep: row["provenance"].update(code_tree_sha256="unknown")),
    ("C4", lambda row, rep: rep["provenance"].update(aut_profile="qwen2.5-32b")),
    ("C4", lambda row, rep: rep["provenance"].update(language_router_n_ctx=4096)),
    ("C4", lambda row, rep: rep["provenance"].update(configured_n_ctx_source="env")),
    ("C4", lambda row, rep: rep.update(goal="escape a dungeon")),
    ("C4", lambda row, rep: row["sim_argv"].append("--seed")),
    ("C4", lambda row, rep: row["served_model"]["reads"][0].update(match=None)),
    ("C4", lambda row, rep: row["served_model"].update(reads=[])),
    ("C4", lambda row, rep: row["served_model"]["reads"].append({"served": "qwen.gguf", "match": False, "at": 1.0})),
    ("C4", lambda row, rep: rep.update(language_endpoint="")),
    ("C4", lambda row, rep: row["sim_env"].update(MAXIM_SUBSTRATE_PATH="1")),
    ("C1", lambda row, rep: row["files"].pop("run_log.jsonl")),
    ("C2", lambda row, rep: row["files"].pop("aut_hippocampus.json")),
    ("C4", lambda row, rep: rep.update(language_endpoint="http://10.0.0.2:8100/v1")),
]


def test_the_mock_gate_phase_is_complete(mock_attempt) -> None:
    rows = mock_attempt("10")
    row, rep = _phase(rows, 1)
    assert v.complete_problems("10", 1, row, rep, _p1(rows), set(v.RESUME_STORES)) == []


@pytest.mark.parametrize("label,perturb", PERTURB)
def test_each_condition_flips(mock_attempt, label, perturb) -> None:
    rows = mock_attempt("10")
    row, rep = _phase(rows, 1)
    perturb(row, rep)
    problems = v.complete_problems("10", 1, row, rep, _p1(rows), set(v.RESUME_STORES))
    assert problems and all(p.startswith(label) for p in problems), problems


def test_a_store_phase_1_saved_must_load_and_a_store_it_did_not_need_not(mock_attempt) -> None:
    rows = mock_attempt("10")
    row, rep = _phase(rows, 1)
    rep["provenance"]["resume"]["stores"]["ec"] = "absent"
    assert v.complete_problems("10", 1, row, rep, _p1(rows), {"hippocampus", "ec"})
    assert v.complete_problems("10", 1, row, rep, _p1(rows), {"hippocampus"}) == []


def test_phase_one_resumes_nothing(mock_attempt) -> None:
    rows = mock_attempt("10")
    row, rep = _phase(rows, 0)
    rep["provenance"]["resume"] = {"resume_loaded": True}
    assert any(p.startswith("C2") for p in v.complete_problems("10", 0, row, rep, row["session_id"], set()))


# ── Exp 10 gates ─────────────────────────────────────────────────────────────────────────────────────────


def _exp10_phases(rows_file: Path) -> list[dict]:
    out = []
    for line in rows_file.read_text().splitlines():
        row = json.loads(line)
        sdir = rows_file.parent / row["session_id"]
        out.append({
            "report": json.loads(v.read_copied(sdir, "report.json")),
            "store": json.loads(v.read_copied(sdir, "aut_hippocampus.json")),
            "lines": v.log_lines(v.read_copied(sdir, v.RUN_LOG)),
        })  # fmt: skip
    return out


def test_exp10_gates_flip(mock_attempt) -> None:
    phases = _exp10_phases(mock_attempt("10"))
    assert v.exp10_gates(phases)["verdict"] == "PASS"
    assert v.exp10_gates(phases)["carried_recall_turn1"] == "carried recall shown on the first resume turn"

    p = copy.deepcopy(phases)
    p[0]["store"]["memories"] = p[0]["store"]["memories"][:2]  # P0: N1 < 3
    assert not v.exp10_gates(p)["P0"]["pass"]
    p = copy.deepcopy(phases)
    p[1]["store"]["memories"] = p[1]["store"]["memories"][1:]  # P2: a carried id missing
    assert not v.exp10_gates(p)["P2"]["pass"]
    p = copy.deepcopy(phases)
    next(r for r in p[1]["lines"] if r.get("e") == "enrichment_trace")["hippocampus_size"] = 1  # P1
    assert not v.exp10_gates(p)["P1"]["pass"]
    p = copy.deepcopy(phases)
    p[1]["lines"] = [
        r for r in p[1]["lines"] if not (r.get("e") == "enrichment_trace" and r["t"] >= 1070)
    ]  # R1: turn 8
    assert not v.exp10_gates(p)["R1"]["pass"]
    p = copy.deepcopy(phases)
    [r for r in p[1]["lines"] if r.get("e") == "enrichment_trace"][3]["memories"] = 2  # R2
    out = v.exp10_gates(p)
    assert not out["R2"]["pass"] and out["verdict"] == "FAIL"


def _damage(component: str, source: str, agent: str = "sim_aut") -> dict:
    msg = f"component damage: {component}.integrity → 0.8 (source={source}, amount=0.1)"
    return {"e": "sim_sem_damage", "agent_id": agent, "message": msg}


def test_h2_counts_only_reflex_damage_in_the_agents_context() -> None:
    def h2(lines: list[dict]) -> dict:
        return v.exp09_gates({"lines": lines}, b"")["H2"]

    assert h2([_damage("torso", "reflex_attack"), _damage("legs", "reflex_impact")])["status"] == "PASS"
    # The narrator's own damage_component calls, or a reflex logged outside the agent, do not count.
    assert h2([_damage("torso", "reflex_attack"), _damage("legs", "damage_component")])["status"] == "NOT MET"
    assert h2([_damage("torso", "reflex_attack"), _damage("legs", "reflex_impact", "sim_orchestrator")])["status"] == (
        "NOT MET"
    )
    assert h2([_damage("torso", "reflex_attack"), _damage("arms", "reflex_attack")])["status"] == "NOT MET"  # no legs


# ── the harness's refusals (none of them is an attempt) ──────────────────────────────────────────────────


def test_harness_refuses_before_any_marker(capsys) -> None:
    assert h.main(["run", "--exp", "10", "--mock", "--write-experiment-results"]) == 2  # a mock never writes evidence
    assert h.main(["run", "--exp", "10"]) == 2  # a real attempt must write committed evidence
    assert "REFUSED" in capsys.readouterr().err


def test_a_complete_attempt_closes_the_file(mock_attempt, monkeypatch, capsys) -> None:
    rows = mock_attempt("09")
    assert h.attempt_complete("09", [json.loads(ln) for ln in rows.read_text().splitlines()])
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    assert h.main(["run", "--exp", "09", "--mock"]) == 2  # same rows file (redirected by the fixture)
    assert "already holds a complete attempt" in capsys.readouterr().err


def test_a_fourth_attempt_is_refused(mock_attempt, monkeypatch, capsys) -> None:
    rows = mock_attempt("10")
    lines = [json.loads(ln) for ln in rows.read_text().splitlines()]
    aborted = []
    for i, rid in enumerate(("1" * 32, "2" * 32, "3" * 32)):
        row = copy.deepcopy(lines[0])
        row["provenance"]["harness_run_id"] = rid
        row["status"] = "failed"
        aborted.append(json.dumps(row))
    rows.write_text("\n".join(aborted) + "\n")
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    assert h.main(["run", "--exp", "10", "--mock"]) == 2
    assert "at most 3" in capsys.readouterr().err


def test_an_incomplete_phase_is_a_failed_row_and_stops_the_attempt(tmp_path, monkeypatch) -> None:
    real = h.mock_phase

    def aborting(exp, index, **kw):
        sdir, report, *rest = real(exp, index, **kw)
        if index == 1:
            report["finish_reason"] = "planning_failed"  # the typed abort that made O19 necessary
            (sdir / "report.json").write_text(json.dumps(report))
        return (sdir, report, *rest)

    rows = tmp_path / "rows.jsonl"
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows)
    monkeypatch.setattr(h, "mock_phase", aborting)
    assert h.main(["run", "--exp", "10", "--mock"]) == 1
    written = [json.loads(ln) for ln in rows.read_text().splitlines()]
    assert [r["status"] for r in written] == ["ok", "failed"]  # phase 3 never ran
    assert written[1]["complete_problems"][0].startswith("C1 gate: finish_reason 'planning_failed'")
    assert _judge("10", rows)["verdict"] == "ABORT"


# ── check_apparatus against a real (temporary) git origin ────────────────────────────────────────────────

import subprocess  # noqa: E402


def _git(cwd: Path, *args: str, date: str | None = None) -> str:
    env = None
    if date:
        import os

        env = {**os.environ, "GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date}
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgSign=false", "-c", "tag.gpgSign=false",
         *args], cwd=cwd, env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip


class Rig:
    """A bare origin + a clone standing in for the repo: code commit, markers, rows landing by --no-ff merges."""

    def __init__(self, tmp: Path, key: str = "10"):
        self.key, self.REL = key, v.rows_path(key)
        self.origin, self.work = tmp / "origin.git", tmp / "work"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)], check=True)
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.work)], check=True, capture_output=True)
        _git(self.work, "checkout", "-q", "-b", "main")
        (self.work / "code.py").write_text("x = 1\n")
        _git(self.work, "add", ".")
        _git(self.work, "commit", "-q", "-m", "code", date="2026-10-01T10:00:00Z")
        self.code = _git(self.work, "rev-parse", "HEAD")
        _git(self.work, "push", "-q", "origin", "main")
        self.rows: list[str] = []

    def marker(self, k: int, rid: str, date: str, commit: str | None = None) -> None:
        name = f"o19/{self.key}/attempt-{k}-{rid}"
        _git(self.work, "tag", "-a", name, "-m", "m", commit or self.code, date=date)
        _git(self.work, "push", "-q", "origin", f"refs/tags/{name}")

    def land(self, rid: str, ts: float, date: str, executed: str | None = None, replace: bool = False) -> None:
        row = {"record_kind": "harness_row", "status": "failed", "ts": ts, "phase_index": 0,
               "provenance": {"harness_run_id": rid, "executed_git_hash": executed or self.code}}  # fmt: skip
        self.rows = [json.dumps(row)] if replace else [*self.rows, json.dumps(row)]
        branch = f"data-{rid[:4]}-{len(self.rows)}"
        _git(self.work, "checkout", "-q", "-b", branch)
        path = self.work / self.REL
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.rows) + "\n")
        _git(self.work, "add", ".")
        _git(self.work, "commit", "-q", "-m", "rows", date=date)
        _git(self.work, "checkout", "-q", "main")
        _git(self.work, "merge", "-q", "--no-ff", "-m", "data PR", branch, date=date)
        _git(self.work, "push", "-q", "origin", "main")

    def check(self, monkeypatch):
        monkeypatch.setattr(v, "REPO_ROOT", self.work)
        ruleset = {**GOOD_RULESET, "created_at": "2026-10-01T09:00:00Z", "updated_at": "2026-10-01T09:00:00Z"}
        monkeypatch.setattr(v, "_gh_json", lambda path: {"full_name": "o/r"} if path.endswith("{repo}") else ruleset)
        monkeypatch.setattr(v, "_gh_list", lambda path: [] if path.endswith("/history") else [{"id": 7}])
        data = (self.work / self.REL).read_bytes()
        rows = [json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]
        return v.check_apparatus(self.key, self.REL, data, v.attempts_from_rows(rows))

    def put(self, rel: str, data: bytes, date: str) -> None:
        """Commit ``rel`` straight onto main (its first-parent landing time is ``date``)."""
        path = self.work / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        _git(self.work, "add", ".")
        _git(self.work, "commit", "-q", "-m", f"add {rel}", date=date)
        _git(self.work, "push", "-q", "origin", "main")


def _epoch(iso: str) -> float:
    return v._iso(iso)


def test_apparatus_accepts_two_ordered_attempts(tmp_path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    rig.marker(1, RID[0], "2026-10-01T10:05:00Z")
    rig.land(RID[0], _epoch("2026-10-01T10:06:00Z"), "2026-10-01T11:00:00Z")
    rig.marker(2, RID[1], "2026-10-01T12:00:00Z")
    rig.land(RID[1], _epoch("2026-10-01T12:01:00Z"), "2026-10-01T13:00:00Z")
    ordered, record = rig.check(monkeypatch)
    assert [a["run_id"] for a in ordered] == RID[:2]
    assert [m["tagger_date"] for m in record["markers"]] == [
        _epoch("2026-10-01T10:05:00Z"),
        _epoch("2026-10-01T12:00:00Z"),
    ]
    assert record["landed"][RID[0]] == _epoch("2026-10-01T11:00:00Z")


def test_apparatus_refuses_a_dropped_attempt(tmp_path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    rig.marker(1, RID[0], "2026-10-01T10:05:00Z")
    rig.land(RID[0], _epoch("2026-10-01T10:06:00Z"), "2026-10-01T11:00:00Z")
    rig.marker(2, RID[1], "2026-10-01T12:00:00Z")
    rig.land(RID[1], _epoch("2026-10-01T12:01:00Z"), "2026-10-01T13:00:00Z", replace=True)  # attempt 1's rows gone
    with pytest.raises(v.Refusal, match="not a prefix"):
        rig.check(monkeypatch)


def test_apparatus_refuses_an_attempt_started_before_the_last_landed(tmp_path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    rig.marker(1, RID[0], "2026-10-01T10:05:00Z")
    rig.marker(2, RID[1], "2026-10-01T10:30:00Z")
    rig.land(RID[0], _epoch("2026-10-01T10:06:00Z"), "2026-10-01T11:00:00Z")
    rig.land(RID[1], _epoch("2026-10-01T10:31:00Z"), "2026-10-01T13:00:00Z")
    with pytest.raises(v.Refusal, match="not before attempt"):
        rig.check(monkeypatch)


def test_apparatus_refuses_code_off_main_or_not_the_markers(tmp_path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    _git(rig.work, "checkout", "-q", "-b", "side")
    (rig.work / "code.py").write_text("x = 2\n")
    _git(rig.work, "commit", "-q", "-am", "unreviewed", date="2026-10-01T10:01:00Z")
    side = _git(rig.work, "rev-parse", "HEAD")
    _git(rig.work, "checkout", "-q", "main")
    rig.marker(1, RID[0], "2026-10-01T10:05:00Z", commit=side)
    rig.land(RID[0], _epoch("2026-10-01T10:06:00Z"), "2026-10-01T11:00:00Z", executed=side)
    with pytest.raises(v.Refusal, match="not on origin/main"):
        rig.check(monkeypatch)


def test_apparatus_refuses_rows_from_another_commit_than_the_marker(tmp_path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    rig.marker(1, RID[0], "2026-10-01T10:05:00Z")
    rig.land(RID[0], _epoch("2026-10-01T10:06:00Z"), "2026-10-01T11:00:00Z", executed="0" * 40)
    with pytest.raises(v.Refusal, match="not its marker's commit"):
        rig.check(monkeypatch)


def test_apparatus_refuses_rows_without_a_marker(tmp_path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    rig.marker(1, RID[0], "2026-10-01T10:05:00Z")
    rig.land(RID[0], _epoch("2026-10-01T10:06:00Z"), "2026-10-01T11:00:00Z")
    rig.land(RID[1], _epoch("2026-10-01T11:06:00Z"), "2026-10-01T12:00:00Z")
    with pytest.raises(v.Refusal, match="no start marker"):
        rig.check(monkeypatch)


def test_the_stamped_profile_is_what_the_config_normalizes_to() -> None:
    from maxim.models.language.config import normalize_llm_profile

    assert normalize_llm_profile(v.MODEL_PROFILE) == v.MODEL_PROFILE_STAMPED


# ── the verdict's main, the harness's environment and interruption ───────────────────────────────────────


def test_verdict_main_offline_rules(mock_attempt, tmp_path, capsys, monkeypatch) -> None:
    rows = mock_attempt("10")
    monkeypatch.setattr(h._provenance, "evidence_out_path", _EVIDENCE_OUT_PATH)
    out = tmp_path / "verdict.json"
    assert (
        v.main(["--exp", "10", "--data", str(rows), "--json", str(out), "--offline", "--write-experiment-results"]) == 2
    )
    assert v.main(["--exp", "10", "--data", str(rows), "--json", str(out), "--offline"]) == 0
    verdict = json.loads(out.read_text())
    assert verdict["mock"] is True and verdict["kind"] == "exp10_verdict" and verdict["apparatus_checked"] is False
    assert verdict["scope"] == {"all_rows": True} and verdict["data_sha256"] == v.sha256_bytes(rows.read_bytes())
    real = [json.loads(ln) for ln in rows.read_text().splitlines()]
    for r in real:
        r["mock"] = False
    rows.write_text("".join(json.dumps(r) + "\n" for r in real))
    assert v.main(["--exp", "10", "--data", str(rows), "--json", str(out), "--offline"]) == 2  # offline is mock-only
    assert v.main(["--exp", "10", "--data", str(rows), "--json", str(out)]) == 2  # not the prereg's rows path
    assert "REFUSED" in capsys.readouterr().err


def test_the_sims_get_no_operator_maxim_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAXIM_SUBSTRATE_PATH", "1")  # left over from an Exp 09 shell
    monkeypatch.setenv("MAXIM_SUBSTRATE_ACTIONS_PER_TURN", "3")
    env = h.sim_environment("10", 0, home=tmp_path, run_id="r", run_log=tmp_path / "log")
    assert {k for k in env if k.startswith("MAXIM_")} == {
        "MAXIM_DATA_HOME", "MAXIM_HARNESS_RUN_ID", "MAXIM_LOG_FILE", "MAXIM_LOG_FILE_MAX_BYTES"
    }  # fmt: skip
    env09 = h.sim_environment("09", 0, home=tmp_path, run_id="r", run_log=tmp_path / "log")
    assert env09["MAXIM_SUBSTRATE_PATH"] == "1" and env09["MAXIM_BACKEND_TRACE"] == "1"
    assert {"MAXIM_SUBSTRATE_ACTIONS_PER_TURN", "MAXIM_SUBSTRATE_PATH"} <= set(h.base_env()[1])  # all are dropped


def test_the_preregs_command_parses(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(h, "run", lambda args: seen.setdefault("args", args) and 0)
    h.main(["--exp", "10", "--write-experiment-results"])
    assert seen["args"].command == "run"


def test_an_interrupted_phase_still_writes_its_row(tmp_path, monkeypatch) -> None:
    real = h.mock_phase

    def interrupted(exp, index, **kw):
        if index == 1:
            raise KeyboardInterrupt  # Ctrl-C, an SSH drop: the attempt still counts (its marker is pushed)
        return real(exp, index, **kw)

    rows = tmp_path / "rows.jsonl"
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows)
    monkeypatch.setattr(h, "mock_phase", interrupted)
    with pytest.raises(KeyboardInterrupt):
        h.main(["run", "--exp", "10", "--mock"])
    written = [json.loads(ln) for ln in rows.read_text().splitlines()]
    assert [r["status"] for r in written] == ["ok", "failed"]
    assert written[1]["reason"].startswith("harness interrupted: KeyboardInterrupt")


def test_a_marker_without_a_tagger_date_refuses_not_crashes(tmp_path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    rig.marker(1, RID[0], "2026-10-01T10:05:00Z")
    rig.land(RID[0], _epoch("2026-10-01T10:06:00Z"), "2026-10-01T11:00:00Z")
    real_git = v._git

    def no_tagger(*args: str) -> str:
        out = real_git(*args)
        return " ".join(out.split()[:2]) if args[0] == "for-each-ref" else out

    monkeypatch.setattr(v, "_git", no_tagger)
    with pytest.raises(v.Refusal, match="not the annotated tag"):
        rig.check(monkeypatch)


def test_duplicated_phase_rows_refuse(mock_attempt) -> None:
    rows = mock_attempt("10")
    attempt = [json.loads(ln) for ln in rows.read_text().splitlines()]
    with pytest.raises(v.Refusal, match="0..n-1"):
        v.judge("10", [{"run_id": "x", "k": 1, "rows": [attempt[0], attempt[0], attempt[2]]}], rows.parent)


# ── the remaining guards (each pinned; the delta round showed they were not) ─────────────────────────────


def test_strict_log_parse_refuses_a_bad_line_but_not_a_torn_tail() -> None:
    good, bad = b'{"e": "a"}\n', b"not json\n"
    assert len(v.log_lines(good + b'{"e": "b"', strict=True)) == 1  # a torn tail line is tolerated
    with pytest.raises(v.Refusal):
        v.log_lines(good + bad + good, strict=True)
    assert len(v.log_lines(good + bad + good)) == 2


def _bound_files(rig: Rig, text: str = "v1") -> None:
    for path in (v.PROTOCOL["10"]["prereg"], *v.BOUND_FILES):
        f = rig.work / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)


def test_check_bound(tmp_path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    _bound_files(rig)
    _git(rig.work, "add", ".")
    _git(rig.work, "commit", "-q", "-m", "o19", date="2026-10-01T10:01:00Z")
    _git(rig.work, "push", "-q", "origin", "main")
    executed = _git(rig.work, "rev-parse", "HEAD")
    monkeypatch.setattr(v, "REPO_ROOT", rig.work)
    _git(rig.work, "fetch", "-q", "origin")
    rows = [{"provenance": {"executed_git_hash": executed}}]
    assert v.check_bound("10", rows)["verdict_commit"] == executed
    _bound_files(rig, "v2")  # the gate changed after the data
    _git(rig.work, "commit", "-q", "-am", "edit gate", date="2026-10-01T12:00:00Z")
    _git(rig.work, "push", "-q", "origin", "main")
    _git(rig.work, "fetch", "-q", "origin")
    with pytest.raises(v.Refusal, match="differs"):
        v.check_bound("10", rows)
    _git(rig.work, "checkout", "-q", "-b", "side", executed)
    (rig.work / "other.txt").write_text("x")
    _git(rig.work, "add", ".")
    _git(rig.work, "commit", "-q", "-m", "off main", date="2026-10-01T12:30:00Z")
    with pytest.raises(v.Refusal, match="not on origin/main"):
        v.check_bound("10", rows)


def test_check_bound_needs_the_first_parent_history(tmp_path, monkeypatch) -> None:
    """A branch tip merged into main is an ancestor of main but never landed on it: the gate refuses a verdict
    written there (#1050), so the writer does too."""
    rig = Rig(tmp_path)
    _bound_files(rig)
    _git(rig.work, "add", ".")
    _git(rig.work, "commit", "-q", "-m", "o19", date="2026-10-01T10:01:00Z")
    executed = _git(rig.work, "rev-parse", "HEAD")
    _git(rig.work, "checkout", "-q", "-b", "data")
    (rig.work / "rows.txt").write_text("x")
    _git(rig.work, "add", ".")
    _git(rig.work, "commit", "-q", "-m", "the data", date="2026-10-01T11:00:00Z")
    tip = _git(rig.work, "rev-parse", "HEAD")
    _git(rig.work, "checkout", "-q", "main")
    _git(rig.work, "merge", "-q", "--no-ff", "-m", "merge the data", "data", date="2026-10-01T11:30:00Z")
    _git(rig.work, "push", "-q", "origin", "main")
    monkeypatch.setattr(v, "REPO_ROOT", rig.work)
    _git(rig.work, "fetch", "-q", "origin")
    rows = [{"provenance": {"executed_git_hash": executed}}]
    assert v.check_bound("10", rows)["verdict_commit"] != tip  # on main's merge commit: accepted
    _git(rig.work, "checkout", "-q", tip)
    with pytest.raises(v.Refusal, match="first-parent history"):
        v.check_bound("10", rows)


def test_a_full_page_refuses(monkeypatch) -> None:
    monkeypatch.setattr(v, "_gh_json", lambda path: [{"id": i} for i in range(100)])
    with pytest.raises(v.Refusal, match="unpaginated"):
        v._gh_list("repos/o/r/rulesets")
    monkeypatch.setattr(v, "_gh_json", lambda path: [{"id": 1}])
    assert v._gh_list("repos/o/r/rulesets?includes_parents=false") == [{"id": 1}]


def test_harness_check_on_main(tmp_path, monkeypatch) -> None:
    rig = Rig(tmp_path)
    monkeypatch.setattr(v, "REPO_ROOT", rig.work)
    monkeypatch.setattr(h, "REPO_ROOT", rig.work)
    rows_file = rig.work / rig.REL
    h.check_on_main(rows_file)  # the first attempt: no rows anywhere
    rig.land(RID[0], 1.0, "2026-10-01T11:00:00Z")
    h.check_on_main(rows_file)  # on main, here too
    rows_file.write_text(rows_file.read_text() + '{"local": "not on main"}\n')
    with pytest.raises(h.Refused, match="not origin/main's"):
        h.check_on_main(rows_file)
    rows_file.unlink()
    with pytest.raises(h.Refused, match="restore it"):
        h.check_on_main(rows_file)
    _git(rig.work, "checkout", "-q", "-b", "side")
    _git(rig.work, "checkout", "-q", "--", ".")
    (rig.work / "x.txt").write_text("x")
    _git(rig.work, "add", "x.txt")
    _git(rig.work, "commit", "-q", "-m", "off main", date="2026-10-01T12:00:00Z")
    with pytest.raises(h.Refused, match="not on origin/main"):
        h.check_on_main(rows_file)


def test_lock_and_port(tmp_path) -> None:
    import socket

    held = h.take_lock("10")
    try:
        with pytest.raises(h.Refused, match="another o19 harness"):
            h.take_lock("10")
    finally:
        held.close()
    h.take_lock("10").close()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        with pytest.raises(h.Refused, match="already listens"):
            h.check_port_free(s.getsockname()[1])


def _probe_result(monkeypatch, gguf: Path, **got) -> None:
    import subprocess as sp

    payload = {"profile": "mistral-7b", "profile_source": "config", "n_ctx": 8192, "n_ctx_source": "config",
               "stamped_profile": v.MODEL_PROFILE_STAMPED, "model_path": str(gguf), **got}  # fmt: skip
    monkeypatch.setattr(
        h.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 0, stdout=json.dumps(payload) + "\n", stderr="")
    )


@pytest.mark.parametrize(
    "got",
    [
        {"profile": "claude-sonnet", "profile_source": "env"},  # C7a switched a solo run to a cloud model
        {"profile_source": "env"},
        {"n_ctx": 4096},
        {"n_ctx_source": "env"},
        {"stamped_profile": "qwen2.5-32b"},
    ],
)
def test_model_preflight_refuses(monkeypatch, tmp_path, got) -> None:
    gguf = tmp_path / v.MODEL_GGUF
    gguf.write_text("x")
    _probe_result(monkeypatch, gguf)
    assert h.check_model_config(tmp_path, "10")["model_path"] == str(gguf)  # the baseline passes
    _probe_result(monkeypatch, gguf, **got)
    with pytest.raises(h.Refused, match="not the prereg's"):
        h.check_model_config(tmp_path, "10")


def test_model_preflight_mirrors_the_sims_startup(monkeypatch, tmp_path) -> None:
    gguf = tmp_path / v.MODEL_GGUF
    gguf.write_text("x")
    seen = {}

    def fake_run(cmd, **kw):
        import subprocess as sp

        seen["probe"] = cmd[-1]
        seen["env"] = kw["env"]
        payload = {"profile": "mistral-7b", "profile_source": "config", "n_ctx": 8192, "n_ctx_source": "config",
                   "stamped_profile": v.MODEL_PROFILE_STAMPED, "model_path": str(gguf)}  # fmt: skip
        return sp.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(h.subprocess, "run", fake_run)
    monkeypatch.setenv("MAXIM_LLM_PROFILE", "claude-sonnet")
    assert h.check_model_config(tmp_path, "10")["model_path"] == str(gguf)
    probe = seen["probe"]
    assert (
        probe.index("detect_and_apply_role(")
        < probe.index("configure_cloud_solo_auto_detect(logging")
        < probe.index("resolve_setting('llm.profile')")
    )
    assert "MAXIM_LLM_PROFILE" not in seen["env"]  # the sims' environment, not the operator's


def test_stop_sim_terms_before_it_kills() -> None:
    import subprocess as sp

    calls = []

    class Proc:
        def poll(self):
            return None

        def terminate(self):
            calls.append("term")

        def kill(self):
            calls.append("kill")

        def wait(self, timeout=None):
            calls.append(f"wait({timeout})")
            if timeout is not None:
                raise sp.TimeoutExpired("sim", timeout)

    h.stop_sim(Proc())
    assert calls == ["term", f"wait({h.TERM_GRACE_S})", "kill", "wait(None)"]


def test_recorded_env_is_what_was_passed(tmp_path) -> None:
    env = h.sim_environment("09", 0, home=tmp_path, run_id="r", run_log=tmp_path / "l")
    assert h.recorded_env(env) == v.expected_env("09", 0)
    env["MAXIM_ROLE"] = "solo"
    del env["MAXIM_LOG_FILE"]
    rec = h.recorded_env(env)
    assert rec["MAXIM_ROLE"] == "solo" and rec["_missing_run_local"] == "MAXIM_LOG_FILE"


def test_the_temp_home_is_removed(tmp_path, monkeypatch) -> None:
    made = []
    real = h.tempfile.mkdtemp

    def record(*a, **k):
        made.append(real(*a, **k))
        return made[-1]

    monkeypatch.setattr(h.tempfile, "mkdtemp", record)
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: tmp_path / "rows.jsonl")
    assert h.main(["run", "--exp", "09", "--mock"]) == 0
    homes = [d for d in made if Path(d).name.startswith("o19-home-")]
    assert homes and not any(Path(d).exists() for d in homes)


def test_sigterm_to_the_harness_still_writes_the_row(tmp_path, monkeypatch) -> None:
    import os
    import signal

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    real = h.mock_phase

    def terminated(exp, index, **kw):
        if index == 1:
            os.kill(os.getpid(), signal.SIGTERM)  # the operator's `kill`, a closing SSH session
        return real(exp, index, **kw)

    rows = tmp_path / "rows.jsonl"
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows)
    monkeypatch.setattr(h, "mock_phase", terminated)
    try:
        with pytest.raises(h.HarnessSignal):
            h.main(["run", "--exp", "10", "--mock"])
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    written = [json.loads(ln) for ln in rows.read_text().splitlines()]
    assert [r["status"] for r in written] == ["ok", "failed"]
    assert written[1]["reason"].startswith("harness interrupted: HarnessSignal")


class _FakeSim:
    def __init__(self, polls):
        self.polls, self.calls, self.returncode = list(polls), [], None

    def poll(self):
        if "term" in self.calls:
            self.returncode = -15
        elif self.polls:
            self.returncode = self.polls.pop(0)
        return self.returncode

    def terminate(self):
        self.calls.append("term")

    def kill(self):
        self.calls.append("kill")

    def wait(self, timeout=None):
        return self.returncode


def _spawn(monkeypatch, tmp_path, sim, read_served):
    monkeypatch.setattr(h.subprocess, "Popen", lambda *a, **k: sim)
    monkeypatch.setattr(h, "POLL_S", 0.0)
    return h.spawn_phase(
        "10", 0, home=tmp_path, run_id="r" * 32, resume=None, gguf="g", timeout_s=60, read_served=read_served
    )


def test_an_interrupted_poll_stops_the_sim(monkeypatch, tmp_path) -> None:
    def interrupt(*a):
        raise KeyboardInterrupt

    sim = _FakeSim([None, None])
    with pytest.raises(KeyboardInterrupt):
        _spawn(monkeypatch, tmp_path, sim, interrupt)
    assert sim.calls[0] == "term"  # the sim (and the server it spawned) does not outlive the harness


def test_the_row_records_the_env_actually_passed(monkeypatch, tmp_path) -> None:
    real = h.sim_environment
    monkeypatch.setattr(h, "sim_environment", lambda *a, **k: {**real(*a, **k), "MAXIM_ROLE": "solo"})
    _sdir, _report, fields, *_rest = _spawn(monkeypatch, tmp_path, _FakeSim([0]), lambda *a: {})
    assert fields["sim_env"]["MAXIM_ROLE"] == "solo"  # C4 then refuses it: not the protocol's env


def test_the_harness_restores_the_signal_handlers(mock_attempt) -> None:
    import signal

    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    mock_attempt("09")
    assert {sig: signal.getsignal(sig) for sig in before} == before


# ── campaigns (owner decisions 2026-10-02: a successor only after a pinned ABORT; campaign 2 is the last) ─────


def test_the_campaign_table_is_sound_and_10_is_closed() -> None:
    assert v.protocol_problems() == []
    assert v.closed_keys() == {"10": "10c2"}
    assert v.experiment_of("10c2") == "10" and v.PROTOCOL["10c2"]["phases"] == v.PROTOCOL["10"]["phases"]
    sup = v.PROTOCOL["10c2"]["supersedes"]
    assert (sup["owner_decision"], sup["cause_issue"], sup["key"]) == ("2026-10-01", 1042, "10")
    assert sup["verdict_sha256"] in PREREG["10c2"] and "**second and last**" in PREREG["10c2"]


def test_the_pinned_closure_is_campaign_1s_real_abort() -> None:
    sup = v.PROTOCOL["10c2"]["supersedes"]
    closure = (REPO / sup["verdict"]).read_bytes()
    assert v.successor_problems("10c2", closure, 1.0, 1.0, 2.0, data_root=DATA_ROOT) == []


def _table(**changes) -> dict:
    import copy

    table = copy.deepcopy(v.PROTOCOL)
    for key, value in changes.items():
        if value is None:
            table.pop(key)
        else:
            table[key] = value
    return table


def test_the_campaign_table_refuses() -> None:
    c2 = v.PROTOCOL["10c2"]
    third = {**c2, "scope": "rerun_exp10_o19c3", "supersedes": {**c2["supersedes"], "key": "10c2",
             "verdict": "docs/experiments/data/rerun_exp10_o19c2/verdict.json"}}  # fmt: skip
    cases = {
        "more than 2": _table(**{"10c3": third}),
        "is not [0-9a-z]+": _table(**{"10/c2": c2, "10c2": None}),
        "or is reserved": _table(**{"preflight": c2, "10c2": None}),
        "superseded more than once": _table(**{"10c3": {**c2, "scope": "rerun_exp10_o19c3"}}),
        "2 open campaigns": _table(**{"10c2": {k: x for k, x in c2.items() if k != "supersedes"}}),
        "is not 10, 09 or 63": _table(**{"09": {**v.PROTOCOL["09"], "experiment": "11"}}),
        "another experiment or kind": _table(**{"10c2": {**c2, "kind": "exp09_verdict"}}),
        "is not 10's own": _table(**{"10c2": {**c2, "supersedes": {**c2["supersedes"], "verdict": "docs/x.json"}}}),
        "owner_decision is missing": _table(
            **{"10c2": {**c2, "supersedes": {**c2["supersedes"], "owner_decision": ""}}}
        ),
        "share a scope": _table(**{"10c2": {**c2, "scope": "rerun_exp10_o19"}}),
        "share a prereg": _table(**{"10c2": {**c2, "prereg": v.PROTOCOL["10"]["prereg"]}}),
    }
    for expected, table in cases.items():
        problems = v.protocol_problems(table)
        assert any(expected in p for p in problems), (expected, problems)


def test_the_verdict_refuses_an_unsound_campaign_table(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(v, "protocol_problems", lambda: ["broken"])
    out = tmp_path / "v.json"
    assert v.main(["--exp", "10c2", "--data", v.rows_path("10c2"), "--json", str(out)]) == 2
    assert "campaign table is unsound" in capsys.readouterr().err and not out.exists()


def test_the_harness_checks_the_campaign_before_any_marker(monkeypatch, capsys) -> None:
    """The real (non-mock) path runs check_campaign: campaign 10 is refused as closed before anything is pushed."""
    monkeypatch.setattr(h._provenance, "assert_repo_interpreter", lambda *a, **k: None)
    monkeypatch.setattr(h._provenance, "executed_code_provenance", lambda *a, **k: {})
    monkeypatch.setattr(h._provenance, "append_refusal", lambda *a, **k: None)
    monkeypatch.setattr(h, "take_lock", lambda exp: None)
    monkeypatch.setattr(h, "check_on_main", lambda rows_file: None)
    pushed = []
    monkeypatch.setattr(h, "push_marker", lambda *a, **k: pushed.append(a))
    assert h.main(["run", "--exp", "10", "--write-experiment-results"]) == 2
    assert "closed" in capsys.readouterr().err and pushed == []


def test_the_apparatus_checks_a_successors_closure(tmp_path, monkeypatch) -> None:
    sup = v.PROTOCOL["10c2"]["supersedes"]
    closure, prereg = (REPO / sup["verdict"]).read_bytes(), (REPO / v.PROTOCOL["10c2"]["prereg"]).read_bytes()
    c1_rows = (
        REPO / v.rows_path("10")
    ).read_bytes()  # the leaked-gate bar reads them (one failed phase: nothing leaked)
    rig = Rig(tmp_path, key="10c2")
    rig.put(v.rows_path("10"), c1_rows, "2026-10-02T08:00:00Z")
    rig.put(sup["verdict"], closure, "2026-10-02T09:00:00Z")
    rig.put(v.PROTOCOL["10c2"]["prereg"], prereg, "2026-10-02T09:30:00Z")
    rig.marker(1, RID[0], "2026-10-02T10:05:00Z")
    rig.land(RID[0], _epoch("2026-10-02T10:06:00Z"), "2026-10-02T11:00:00Z")
    _ordered, record = rig.check(monkeypatch)
    assert record["succession"]["key"] == "10" and record["succession"]["closure_landed"] < _epoch(
        "2026-10-02T10:05:00Z"
    )
    late = Rig(tmp_path / "late", key="10c2")
    late.put(v.rows_path("10"), c1_rows, "2026-10-02T08:00:00Z")
    late.put(v.PROTOCOL["10c2"]["prereg"], prereg, "2026-10-02T09:30:00Z")
    late.marker(1, RID[0], "2026-10-02T10:05:00Z")
    late.put(sup["verdict"], closure, "2026-10-02T10:30:00Z")  # the closure landed after the first marker
    late.land(RID[0], _epoch("2026-10-02T10:06:00Z"), "2026-10-02T11:00:00Z")
    with pytest.raises(v.Refusal, match="closure verdict reached origin/main after its first marker"):
        late.check(monkeypatch)
    swapped = Rig(tmp_path / "swapped", key="10c2")
    swapped.put(v.rows_path("10"), c1_rows, "2026-10-02T08:00:00Z")
    swapped.put(sup["verdict"], b"{}", "2026-10-02T09:00:00Z")  # another file at the path, early
    swapped.put(v.PROTOCOL["10c2"]["prereg"], prereg, "2026-10-02T09:30:00Z")
    swapped.marker(1, RID[0], "2026-10-02T10:05:00Z")
    swapped.put(sup["verdict"], closure, "2026-10-02T10:30:00Z")  # the pinned bytes, after the first marker
    swapped.land(RID[0], _epoch("2026-10-02T10:06:00Z"), "2026-10-02T11:00:00Z")
    with pytest.raises(v.Refusal, match="closure verdict reached origin/main after its first marker"):
        swapped.check(monkeypatch)


def test_only_a_pinned_real_abort_landed_before_the_first_marker_may_be_succeeded() -> None:
    sup = v.PROTOCOL["10c2"]["supersedes"]
    real = (REPO / sup["verdict"]).read_bytes()
    record = json.loads(real)

    def edited(**fields) -> bytes:
        return json.dumps({**record, **fields}).encode()

    assert v.successor_problems("10c2", None, None, 1.0, 2.0, data_root=DATA_ROOT)  # not on main
    assert v.successor_problems("10c2", real + b" ", 1.0, 1.0, 2.0, data_root=DATA_ROOT)  # not the pinned bytes
    for change in ({"verdict": "PASS"}, {"verdict": "FAIL"}, {"verdict": "NOT SHOWN"}, {"mock": True}, {"experiment": "10c2"},
                   {"apparatus_checked": False}):  # fmt: skip
        problems = v.successor_problems("10c2", edited(**change), 1.0, 1.0, 2.0, data_root=DATA_ROOT)
        assert any("not a real ABORT" in p or "did not check" in p for p in problems), change
    assert v.successor_problems(
        "10c2", real, 3.0, 1.0, 2.0, data_root=DATA_ROOT
    )  # the closure landed after the first marker
    assert v.successor_problems("10c2", real, 1.0, None, 2.0, data_root=DATA_ROOT)  # the prereg never reached main
    assert v.successor_problems("10c2", real, 1.0, 3.0, 2.0, data_root=DATA_ROOT)  # ...or after the first marker
    assert v.successor_problems("09", None, None, None, 2.0, data_root=DATA_ROOT) == []  # supersedes nothing


def test_the_harness_refuses_a_closed_or_unclosed_campaign(monkeypatch) -> None:
    with pytest.raises(h.Refused, match="closed"):
        h.check_campaign("10")
    # The leaked-gate bar reads origin/main (absent in CI's unit-test checkout): its own tests below.
    monkeypatch.setattr(v, "materialize", lambda ref, keys, root: root)
    monkeypatch.setattr(v, "chain_leaked_gate_problems", lambda key, data_root: [])
    monkeypatch.setattr(v, "_git_bytes", lambda *a: None)
    monkeypatch.setattr(v, "landed_on_main", lambda path, want=None: None)
    with pytest.raises(h.Refused, match="not on origin/main"):
        h.check_campaign("10c2")
    real = (REPO / v.PROTOCOL["10c2"]["supersedes"]["verdict"]).read_bytes()
    monkeypatch.setattr(v, "_git_bytes", lambda *a: real)
    monkeypatch.setattr(v, "landed_on_main", lambda path, want=None: 1.0)
    monkeypatch.setattr(v, "subject_problems", lambda key, head: [])  # its own test below
    h.check_campaign("10c2")  # landed before now: may start
    h.check_campaign("09")


def test_every_gate_choice_goes_through_the_experiment_not_the_key() -> None:
    for script in ("o19_verdict.py", "o19_rerun.py"):
        text = (REPO / "scripts" / script).read_text()
        assert not re.search(r"\bexp\s*[!=]=\s*[\"']\d", text), script


def test_a_campaign_2_attempt_is_judged_as_exp_10(mock_attempt) -> None:
    out = _judge("10c2", mock_attempt("10c2"))
    assert out["verdict"] == "PASS" and out["experiment"] == "10c2" and "P0" in json.dumps(out["gates"])


def test_a_closed_campaigns_verdict_is_not_recomputable_after_a_script_change() -> None:
    """By design (#1050): campaign 1 ran on 17ca6c56, and this change alters the bound verdict script, so
    ``check_bound`` refuses ``--exp 10`` from now on. Pinned on the real blobs (skipped on a shallow clone)."""
    closure = json.loads((REPO / v.PROTOCOL["10c2"]["supersedes"]["verdict"]).read_text())
    then = subprocess.run(
        ["git", "rev-parse", f"17ca6c56:{v.BOUND_FILES[0]}"], cwd=REPO, capture_output=True, text=True
    )
    if then.returncode != 0:
        pytest.skip("campaign 1's commit is not in this clone")
    now = subprocess.run(["git", "hash-object", v.BOUND_FILES[0]], cwd=REPO, capture_output=True, text=True)
    assert then.stdout.strip() != now.stdout.strip()
    assert closure["bound_files"][v.BOUND_FILES[0]] == then.stdout.strip()  # the closure bound the old script


# ── the served-model reader ──────────────────────────────────────────────────────────────────────────────


def test_the_served_reader_waits_out_a_completion_and_records_why_a_read_failed(monkeypatch) -> None:
    from maxim.utils import http as _http

    calls: list = []

    class _Resp:
        content = json.dumps({"data": [{"id": v.MODEL_GGUF}]}).encode()

    def fetch(url, **kw):
        calls.append((url, kw["timeout"], kw["headers"]))
        if len(calls) == 2:
            raise TimeoutError("read timed out")  # a read still queued behind the model lock
        return _Resp()

    monkeypatch.setattr(_http, "fetch_url", fetch)
    monkeypatch.setattr("maxim.tunnel.keys.read_key", lambda: "k")
    read = h.served_reader()
    ok, failed = read(v.SIM_PORT, v.MODEL_GGUF), read(v.SIM_PORT, v.MODEL_GGUF)
    assert ok["served"] == v.MODEL_GGUF and ok["match"] is True and "error" not in ok
    assert failed["served"] is None and failed["match"] is None and failed["error"] == "TimeoutError: read timed out"
    url, timeout, headers = calls[0]
    assert url == f"http://127.0.0.1:{v.SIM_PORT}/v1/models" and headers == {"Authorization": "Bearer k"}
    # campaigns 1-2's completions took up to 24 s and the narrator and the AUT share the server
    assert timeout.read_s >= 48 and timeout.total_s >= 48


@pytest.mark.parametrize("exp", sorted(v.PROTOCOL))
def test_every_phase_argv_parses_on_this_build(exp) -> None:
    h.check_argv_parses(exp)  # refuses before the marker when a protocol needs a flag this build lacks


def test_an_unparseable_phase_argv_refuses_before_the_marker(monkeypatch) -> None:
    phases = [
        (n, g, cap, r, [*extra, "--no-such-flag"], env) for n, g, cap, r, extra, env in v.PROTOCOL["09"]["phases"]
    ]
    monkeypatch.setitem(v.PROTOCOL, "09", {**v.PROTOCOL["09"], "phases": phases})
    with pytest.raises(h.Refused, match="does not parse"):
        h.check_argv_parses("09")


def test_the_harness_refuses_an_unparseable_argv_before_any_attempt(tmp_path, monkeypatch) -> None:
    phases = [
        (n, g, cap, r, [*extra, "--no-such-flag"], env) for n, g, cap, r, extra, env in v.PROTOCOL["09"]["phases"]
    ]
    monkeypatch.setitem(v.PROTOCOL, "09", {**v.PROTOCOL["09"], "phases": phases})
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    rows = tmp_path / "rows_09.jsonl"
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows)
    assert h.main(["run", "--exp", "09", "--mock"]) != 0
    assert not rows.exists() or not rows.read_text().strip(), "a refusal before the marker writes no attempt"


# ── Exp 63 (T1-16): carried memory takes part in recall ──────────────────────────────────────────────────


def test_exp63_protocol_is_the_preregs() -> None:
    text = _squash(PREREG["63"])
    goal = v.PROTOCOL["63"]["phases"][1][1]
    assert f"| 1, baseline | `{goal}` | 8 | none (a fresh data home) |" in text
    assert f"| 2, the gate | `{goal}` | 8 | phase 1 |" in text
    assert "`--interactive false`, `--sim-max-turns 8` and `--sim-run-full-turns`" in text
    assert [ph[4] for ph in v.PROTOCOL["63"]["phases"]] == [["--sim-run-full-turns"]] * 2
    assert [ph[3] for ph in v.PROTOCOL["63"]["phases"]] == [False, True]
    assert f"`{v.MODEL_GGUF}` (`llm.profile {v.MODEL_PROFILE}`) at `llm.n_ctx {v.N_CTX}`" in text
    assert f"`{v.PROTOCOL['63']['kind']}` → `{{PASS}}`" in text and "at most 3 attempts" in text
    assert f"it must read `{v.EXP63_MEMORY_STRATEGY}`" in text
    assert f"The AUT's agent id is `{v.SIM_AUT_AGENT_ID}`" in text
    for name in v.EXP63_MUTABLE_FIELDS:
        assert f"`{name}`" in text.split("**P3 (content identity).**")[1].split("Every other field")[0], name
    assert "`recall(query=goal, limit=5)`" in text and v.EXP63_GOAL_LIMIT == 5
    assert v.required_files("63") == v.required_files("10") and "supersedes" not in v.PROTOCOL["63"]
    assert v.EXIT["NOT SHOWN"] == 4 and v.EXIT["ABORT"] == 4 and v.EXIT["PASS"] == 0 and v.EXIT["FAIL"] == 1


def test_exp63_constants_are_the_codes() -> None:
    from maxim.runtime.config_loader import SIM_AUT_AGENT_ID, resolve_memory_strategy

    assert v.SIM_AUT_AGENT_ID == SIM_AUT_AGENT_ID
    assert v.EXP63_MEMORY_STRATEGY == "access_based"
    assert "resolve_memory_strategy" in h.check_memory_strategy.__doc__ and callable(resolve_memory_strategy)


def test_exp63_mock_passes_end_to_end(mock_attempt) -> None:
    rows = mock_attempt("63")
    written = [json.loads(ln) for ln in rows.read_text().splitlines()]
    assert [r["memory_strategy"] for r in written] == ["access_based"] * 2  # read in the fresh home, every row
    out = _judge("63", rows)
    assert out["verdict"] == "PASS" and out["gates"]["not_passed"] == [], out
    assert out["gates"]["R3a"]["turn"] == 1 and out["gates"]["R3d"]["decisive_turns"] == list(range(2, 9))


def test_exp63_not_shown(mock_attempt, monkeypatch, tmp_path) -> None:
    """Complete, conforming, but no decisive turn after R3a's: NOT SHOWN, exit 4 (no status change)."""
    monkeypatch.setattr(h, "MOCK63_VARIANT", "not_shown")
    rows = mock_attempt("63")
    out = _judge("63", rows)
    assert out["verdict"] == "NOT SHOWN" and out["gates"]["not_passed"] == ["R3d"], out["gates"]["not_passed"]
    monkeypatch.setattr(h._provenance, "evidence_out_path", _EVIDENCE_OUT_PATH)
    verdict = tmp_path / "verdict.json"
    assert v.main(["--exp", "63", "--data", str(rows), "--json", str(verdict), "--offline"]) == 4
    assert json.loads(verdict.read_text())["kind"] == "exp63_verdict"


def test_exp63_an_instrument_fault_aborts_the_attempt(tmp_path, monkeypatch) -> None:
    """C5 is the harness's reading too: the attempt's last row fails, so the next attempt may start."""
    monkeypatch.setattr(h, "MOCK63_VARIANT", "unrendered")
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    rows = tmp_path / "rows.jsonl"
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows)
    assert h.main(["run", "--exp", "63", "--mock"]) == 1
    written = [json.loads(ln) for ln in rows.read_text().splitlines()]
    assert written[1]["status"] == "failed" and written[1]["complete_problems"][0].startswith("C5(c)")
    assert not h.attempt_complete("63", written)
    out = _judge("63", rows)
    assert out["verdict"] == "ABORT"


def test_exp63_c5_makes_the_verdicts_attempt_incomplete(mock_attempt) -> None:
    """The judge decides C5 itself, from the committed bytes (a row the harness called ok is not enough)."""
    rows = mock_attempt("63")
    attempt = [json.loads(ln) for ln in rows.read_text().splitlines()]
    sdir = rows.parent / attempt[1]["session_id"]
    store = json.loads((sdir / "aut_hippocampus.json").read_text())
    store["memories"][0]["activation_sources"]["enrichment"] += 1
    data = json.dumps(store).encode()
    (sdir / "aut_hippocampus.json").write_bytes(data)
    attempt[1]["files"]["aut_hippocampus.json"] = v.sha256_bytes(data)
    out = v.judge("63", [{"run_id": "x", "k": 1, "rows": attempt}], rows.parent)
    assert out["verdict"] == "ABORT" and out["attempts"][0]["problems"][0].startswith("C5(c)")


def test_exp63_c4_prime(mock_attempt, monkeypatch) -> None:
    rows = mock_attempt("63")
    row, rep = _phase(rows, 1)
    assert v.complete_problems("63", 1, row, rep, _p1(rows), set(v.RESUME_STORES)) == []
    for bad in ("strength", None):
        row["memory_strategy"] = bad
        problems = v.complete_problems("63", 1, row, rep, _p1(rows), set(v.RESUME_STORES))
        assert problems and all(p.startswith("C4'") for p in problems), problems
    row10, rep10 = _phase(mock_attempt("10"), 1)
    assert "memory_strategy" not in row10  # only Exp 63's rows carry the stamp


def test_exp63_preflight_refuses_another_retention_model(monkeypatch, tmp_path) -> None:
    import subprocess as sp

    def probe(strategy):
        out = json.dumps({"memory_strategy": strategy}) + "\n"
        monkeypatch.setattr(h.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 0, stdout=out, stderr=""))

    probe("access_based")
    assert h.check_memory_strategy(tmp_path) == "access_based"
    probe("strength")
    with pytest.raises(h.Refused, match="C4'"):
        h.check_memory_strategy(tmp_path)


def test_exp63_preflight_reads_the_fresh_home_without_operator_env(monkeypatch, tmp_path) -> None:
    """A real probe (this repo's maxim): MAXIM_MEMORY_STRATEGY in the operator's shell cannot reach it."""
    monkeypatch.setenv("MAXIM_MEMORY_STRATEGY", "strength")
    assert h.check_memory_strategy(tmp_path) == "access_based"


def _exp63(mock_attempt, monkeypatch=None, variant: str = "pass") -> list[dict]:
    if monkeypatch is not None:
        monkeypatch.setattr(h, "MOCK63_VARIANT", variant)
    return _exp10_phases(mock_attempt("63"))  # the same per-phase reader: report, store, lines


GOAL63 = v.PROTOCOL["63"]["phases"][1][1]


def _aut(phases, n: int = 0) -> dict:
    return [r for r in phases[1]["lines"] if r.get("e") == "enrichment_trace" and r.get("agent_id") == "sim_aut"][n]


def _new_id(phases) -> str:
    carried = {m["id"] for m in phases[0]["store"]["memories"]}
    return next(m["id"] for m in phases[1]["store"]["memories"] if m["id"] not in carried)


def _rec(phases, phase: int, mid: str) -> dict:
    return next(m for m in phases[phase]["store"]["memories"] if m["id"] == mid)


def _c5_no_agent(p):
    _aut(p).pop("agent_id")


def _c5_no_deliberation(p):
    p[1]["lines"] = [r for r in p[1]["lines"] if r.get("e") != "sim_deliberation"]


def _c5_ids_not_a_list(p):
    _aut(p, 2)["memory_ids"] = None


def _c5_paths_short(p):
    _aut(p, 2)["memory_paths"].pop()


def _c5_unknown_id(p):
    _aut(p, 2)["memory_ids"][0] = "not-in-the-store"


def _c5_no_horizon(p):
    _aut(p, 2).pop("goal_path_horizon")


def _c5_bad_holes(p):
    _aut(p, 2)["goal_path_holes"] = [[5, 3]]


def _c5_other_goal(p):
    _aut(p, 2)["goal"] = "escape a dungeon"


def _c5_liveness(p):
    _aut(p, 3)["hippocampus_size"] = 2


def _c5_compressed(p):
    _rec(p, 1, _new_id(p))["_compressed"] = True


def _c5_observation_text(p):
    _rec(p, 1, _new_id(p))["perception"]["observations"]["text"] = ["not", "a", "string"]


def _c5_render_count(p):
    _rec(p, 1, _aut(p, 4)["memory_ids"][0])["activation_sources"]["enrichment"] += 1


def _c5_graph_without_objects(p):
    tr = _aut(p, 1)
    tr["memory_paths"][0] = "graph"
    _rec(p, 1, tr["memory_ids"][0])["perception"]["detected_objects"] = []


def _c5_graph_not_visible(p):
    tr = _aut(p, 1)
    late = max(p[1]["store"]["memories"], key=lambda m: m["capture_seq"])
    late["perception"]["detected_objects"] = ["door"]
    tr["memory_ids"][0], tr["memory_paths"][0] = late["id"], "graph"
    _rec(p, 1, late["id"])["activation_sources"]["enrichment"] = (
        _rec(p, 1, late["id"])["activation_sources"].get("enrichment", 0) + 1
    )


def _c5_substring(p):
    _aut(p, 2)["memory_paths"][2] = "substring"


def _c5_carried_without_seq(p):
    p[0]["store"]["memories"][0]["capture_seq"] = None  # Amendment 1: strict


def _c5_r3a_unobservable(p):
    for r in p[1]["lines"]:
        if r.get("e") == "enrichment_trace" and r.get("agent_id") == "sim_aut":
            r["goal_path_holes"] = [[0, 0]]


def _c5_r3a_contaminated(p):
    """A record landed between R3a's horizon read and its recall: its view held only carried records."""
    seq_c = v._seq_c({m["id"]: m for m in p[0]["store"]["memories"]})
    _i, trace = v.exp63_r3a_trace(p[1]["lines"], seq_c)
    trace["memory_ids"][0] = _new_id(p)


C5_PERTURB = [
    ("C5(a)", _c5_no_agent),
    ("C5(a)", _c5_no_deliberation),
    ("C5(b)", _c5_ids_not_a_list),
    ("C5(b)", _c5_paths_short),
    ("C5(b)", _c5_unknown_id),
    ("C5(b)", _c5_no_horizon),
    ("C5(b)", _c5_bad_holes),
    ("C5(b)", _c5_other_goal),
    ("C5(L)", _c5_liveness),
    ("C5(d)", _c5_compressed),
    ("C5(d)", _c5_observation_text),
    ("C5(c)", _c5_render_count),
    ("C5(g)", _c5_graph_without_objects),
    ("C5(d)", _c5_carried_without_seq),
    ("C5(s)", _c5_substring),
    ("C5(R3a)", _c5_r3a_unobservable),
    ("C5(R3a)", _c5_r3a_contaminated),
]


@pytest.fixture
def exp63_phases(mock_attempt):
    return _exp63(mock_attempt)


def test_exp63_the_mock_is_c5_clean(exp63_phases) -> None:
    assert v.exp63_c5_problems(exp63_phases, goal=GOAL63) == []


@pytest.mark.parametrize("label,perturb", C5_PERTURB, ids=[f.__name__ for _l, f in C5_PERTURB])
def test_exp63_each_c5_check_flips(exp63_phases, label, perturb) -> None:
    p = copy.deepcopy(exp63_phases)
    perturb(p)
    problems = v.exp63_c5_problems(p, goal=GOAL63)
    assert any(x.startswith(label) for x in problems), problems


def test_exp63_c5_reads_only_the_auts_traces(exp63_phases) -> None:
    """The narrator's trace (another agent's pipeline) is neither shape-checked nor counted as a render."""
    narr = [r for r in exp63_phases[1]["lines"] if r.get("agent_id") == "sim_orchestrator" and "memory_ids" in r]
    assert narr and all(r["memory_paths"] == ["substring"] for r in narr)
    assert v.exp63_c5_problems(exp63_phases, goal=GOAL63) == []


def _g5_p0(p):
    p[0]["store"]["memories"] = p[0]["store"]["memories"][:2]


def _g5_p1(p):
    _aut(p, 0)["hippocampus_size"] = 3


def _g5_p2(p):
    gone = p[0]["store"]["memories"][0]["id"]
    p[1]["store"]["memories"] = [m for m in p[1]["store"]["memories"] if m["id"] != gone]


def _g5_p3_content(p):
    _rec(p, 1, p[0]["store"]["memories"][0]["id"])["outcome"]["success"] = True


def _g5_p3_strength_field(p):
    _rec(p, 1, p[0]["store"]["memories"][0]["id"])["encoding_tag"] = 0.5


def _g5_p3_new_before_seq_c(p):
    _rec(p, 1, _new_id(p))["capture_seq"] = 3


def _g5_r1(p):
    trs = [r for r in p[1]["lines"] if r.get("e") == "enrichment_trace" and r.get("agent_id") == "sim_aut"]
    p[1]["lines"].remove(trs[-1])  # turn 8 has no trace


def _g5_r3a(p):
    _aut(p, 0)["memory_ids"][2] = _new_id(p)


def _g5_r3prime(p):
    tr = _aut(p, 4)  # a lower-ranked visible record in place of the third
    carried = {m["id"] for m in p[0]["store"]["memories"]}
    tr["memory_ids"][2] = next(m["id"] for m in p[1]["store"]["memories"] if m["id"] not in carried)


GATE_PERTURB = [
    ("P0", _g5_p0),
    ("P1", _g5_p1),
    ("P2", _g5_p2),
    ("P3", _g5_p3_content),
    ("P3", _g5_p3_strength_field),
    ("P3", _g5_p3_new_before_seq_c),
    ("R1", _g5_r1),
    ("R3a", _g5_r3a),
    ("R3prime", _g5_r3prime),
]


def test_exp63_the_mock_passes_every_gate(exp63_phases) -> None:
    out = v.exp63_gates(exp63_phases, goal=GOAL63)
    assert out["verdict"] == "PASS" and out["not_passed"] == []
    desc = out["descriptive"]["turns"][1]
    assert desc["observed_carried"] == 3 and desc["best_carried_score"] == 1.0
    # D1: on turn 1 no new record is visible, so the carried ids only fill slots; from turn 2 they outrank one
    assert desc["decisive"] is False and out["descriptive"]["turns"][2]["decisive"] is True


@pytest.mark.parametrize("gate,perturb", GATE_PERTURB, ids=[f.__name__ for _g, f in GATE_PERTURB])
def test_exp63_each_gate_flips(exp63_phases, gate, perturb) -> None:
    p = copy.deepcopy(exp63_phases)
    perturb(p)
    out = v.exp63_gates(p, goal=GOAL63)
    assert not out[gate]["pass"] and out["verdict"] == "FAIL", (gate, out["not_passed"])


def test_exp63_p3_ignores_the_access_bookkeeping(exp63_phases) -> None:
    p = copy.deepcopy(exp63_phases)
    rec = _rec(p, 1, p[0]["store"]["memories"][0]["id"])
    for name in v.EXP63_MUTABLE_FIELDS:
        rec[name] = {"moved": name}
    assert v.exp63_gates(p, goal=GOAL63)["P3"]["pass"]


def test_exp63_r3prime_accepts_any_order_within_an_exact_tie_only(exp63_phases) -> None:
    """Turn 1's top 3: the newest carried record, then two that share a timestamp (an exact tie)."""
    tr = _aut(exp63_phases, 0)
    first, a, b = tr["memory_ids"]
    recs = {m: _rec(exp63_phases, 1, m) for m in (first, a, b)}
    assert recs[a]["timestamp"] == recs[b]["timestamp"] != recs[first]["timestamp"]
    p = copy.deepcopy(exp63_phases)
    _aut(p, 0)["memory_ids"] = [first, b, a]
    assert v.exp63_gates(p, goal=GOAL63)["R3prime"]["pass"]
    p = copy.deepcopy(exp63_phases)
    _aut(p, 0)["memory_ids"] = [a, first, b]
    assert not v.exp63_gates(p, goal=GOAL63)["R3prime"]["pass"]


def test_exp63_r3d_not_shown(mock_attempt, monkeypatch) -> None:
    out = v.exp63_gates(_exp63(mock_attempt, monkeypatch, "not_shown"), goal=GOAL63)
    assert out["verdict"] == "NOT SHOWN" and out["R3d"]["decisive_turns"] == [] and out["R3a"]["pass"]
    assert out["descriptive"]["turns"][2]["decisive"] is False


def test_exp63_turns_are_attributed_by_line_order() -> None:
    """The log's ``t`` is rounded to 0.01 s: a trace logged in the same hundredth as the next turn's ENTER, but
    BEFORE it, belongs to the earlier turn."""
    lines = [
        {"t": 1.0, "e": "sim_exec", "message": "Bridge.send_and_wait ENTER turn=1 text_len=3"},
        {"t": 2.0, "e": "enrichment_trace", "agent_id": "sim_aut", "goal": GOAL63, "n": "a"},
        {"t": 2.0, "e": "enrichment_trace", "agent_id": "sim_aut", "goal": GOAL63, "n": "b"},
        {"t": 2.0, "e": "sim_exec", "message": "Bridge.send_and_wait ENTER turn=2 text_len=3"},
        {"t": 2.0, "e": "enrichment_trace", "agent_id": "sim_orchestrator", "goal": GOAL63, "n": "narrator"},
        {"t": 2.0, "e": "enrichment_trace", "agent_id": "sim_aut", "goal": "", "n": "no goal"},
        {"t": 2.0, "e": "enrichment_trace", "agent_id": "sim_aut", "goal": GOAL63, "n": "c"},
    ]
    per_turn = v.turn_traces(lines)
    assert per_turn[1][1]["n"] == "a" and per_turn[2][1]["n"] == "c" and per_turn[3] is None


def test_goal_ids_conform_over_tie_groups() -> None:
    groups = [["a"], ["b", "c", "d"]]
    for ok in (["a", "b", "c"], ["a", "d", "b"], ["a", "c", "d"]):
        assert v.goal_ids_conform(ok, groups, set(), 3), ok
    for bad in (["a", "b"], ["b", "a", "c"], ["a", "b", "b"], ["a", "b", "x"], ["a", "b", "c", "d"]):
        assert not v.goal_ids_conform(bad, groups, set(), 3), bad
    # one graph id ahead (``seen``): the goal path skips it in its top 3 and keeps 2
    assert v.goal_ids_conform(["a", "c"], groups, {"b"}, 2) and v.goal_ids_conform(["a", "d"], groups, {"b"}, 2)
    assert not v.goal_ids_conform(["a"], groups, {"b"}, 2) and not v.goal_ids_conform(["a", "b"], groups, {"b"}, 2)
    assert v.goal_ids_conform(["a"], [["a"], ["b"]], {"b"}, 2)  # only two candidates, one of them the graph id
    assert not v.goal_ids_conform(["a"], [["a"], ["b", "c"]], set(), 3)  # fewer than the store offers


def test_forces_carried_needs_a_carried_id_in_every_top3() -> None:
    assert v.forces_carried([["c1"], ["n1", "n2"]], {"c1"})
    assert v.forces_carried([["n1"], ["c1", "n2"]], {"c1"})  # the whole tie group lies inside the top 3
    assert not v.forces_carried([["n1"], ["c1", "n2", "n3"]], {"c1"})  # the boundary may skip it
    assert v.forces_carried([["n1"], ["c1", "c2", "n2"]], {"c1", "c2"})  # 2 slots, 1 non-carried candidate
    assert not v.forces_carried([["n1"], ["n2"], ["n3"], ["c1"]], {"c1"})


def _random_store(rng, n: int) -> list[dict]:
    """Saved-store dicts as the real writer produces them, with exact (score, timestamp) ties, compressed records,
    non-string observation text and every token source the ranker reads."""
    from maxim.memory.types import Action, CompressedMemory, Context, EpisodicMemory, Perception

    words = ["escape", "a", "dungeon", "with", "sleeping", "guard", "Guard.", "door", "key", "the", "", "DUNGEON"]

    def phrase(k):
        return " ".join(rng.choice(words) for _ in range(rng.randint(0, k)))

    out = []
    stamps = [1000.0 + rng.randint(0, 4) for _ in range(n)]  # few distinct timestamps: many exact ties
    for i in range(n):
        if rng.random() < 0.2:
            rec = CompressedMemory(
                id=f"c{i}", timestamp=stamps[i], goal=phrase(5) or None, tool_name=rng.choice(["", "look", "key"])
            )
        else:
            text = rng.choice([phrase(8), phrase(60), None, ["guard"]])
            obs = {} if text is None else {"text": text}
            rec = EpisodicMemory(
                id=f"m{i}",
                timestamp=stamps[i],
                perception=Perception(
                    observations=obs,
                    detected_objects=[phrase(2) for _ in range(rng.randint(0, 2))],
                    detected_people=[phrase(2) for _ in range(rng.randint(0, 1))],
                    decision_rationale=phrase(4),
                ),
                context=Context(active_goal=rng.choice([None, "", phrase(7), GOAL63])),
                action=Action(tool_name=rng.choice(["", "examine", "guard", "Dungeon"])),
                capture_seq=i,
            )
        out.append(json.loads(json.dumps(rec.to_dict(), default=str)))  # through JSON, as the store writes it
    return out


def test_the_frozen_ranker_is_the_shipped_ranker() -> None:
    """The judge's frozen copy, pinned BEHAVIOURALLY to ``hippocampus_retrieval._rank_by_relevance`` (over records
    rebuilt by the store's own ``from_dict``): the same order for the same input order, exact ties included, and
    every shipped output is a linearisation of the judge's tie groups. When #1064 changes the shipped ranker, this
    pin is re-pointed at the executed commit's ranker (``git show``), never deleted."""
    import random

    from maxim.memory.hippocampus_retrieval import _rank_by_relevance
    from maxim.memory.types import CompressedMemory, EpisodicMemory

    def load(d):
        return CompressedMemory.from_dict(d) if d.get("_compressed", False) else EpisodicMemory.from_dict(d)

    rng = random.Random(63)
    queries = [GOAL63, "the guard", "Guard. DUNGEON key", "", "zzz", "a a a"]
    ties = 0
    for _ in range(60):
        store = _random_store(rng, rng.randint(1, 25))
        for query in queries:
            for _shuffle in range(3):
                rng.shuffle(store)
                for limit in (3, 5, 100):
                    shipped = [m.id for m in _rank_by_relevance([load(d) for d in store], query, limit)]
                    assert [d["id"] for d in v.rank_by_relevance(store, query, limit)] == shipped, (query, limit)
                groups = v.ranked_groups(store, query)
                ties += sum(len(g) > 1 for g in groups)
                shipped = [m.id for m in _rank_by_relevance([load(d) for d in store], query, len(store))]
                assert [sorted(g) for g in groups] == [
                    sorted(shipped[sum(map(len, groups[:i])) : sum(map(len, groups[: i + 1]))])
                    for i in range(len(groups))
                ]
    assert ties > 100  # the generated stores do exercise exact ties


# ── the 2026-10-03 review folds (owner decisions D1-D3; A2, A8; E1-E5) ──────────────────────────────────


def test_exp63_a_graph_id_that_landed_late_is_allowed(exp63_phases) -> None:
    """E4: a graph id outside V0 but in the saved store landed between the horizon read and the recall, as a goal id
    may: C5(g) accepts it, and it joins L for R3' (so a recall it pushed down still conforms)."""
    p = copy.deepcopy(exp63_phases)
    _c5_graph_not_visible(p)
    assert not any(x.startswith("C5(g)") for x in v.exp63_c5_problems(p, goal=GOAL63))


def test_a_late_graph_id_joins_l_as_a_real_competitor() -> None:
    """Exp 2 / E4: the late graph id g (outside V0, in the saved store) joins L. R3' conforms either way: g is a graph
    id, so the goal path skips it, and the first ``need`` non-graph ids in rank order are the same with or without
    it. What L changes is R3d: g is the only visible new record, and the carried c outranks it, so the turn is
    decisive only because g is a competitor (late arrivals were in recall's view)."""
    records = _keys({"c": (6, 1.0), "g": (1, 9.0)})
    records["g"].update(capture_seq=5, perception={"detected_objects": ["door"]})
    trace = {"memory_ids": ["g", "c"], "memory_paths": ["graph", "goal"], "goal_path_horizon": 0, "goal_path_holes": []}
    out = v.exp63_turn(trace, records, {"c"}, GOAL63)
    assert out["late"] == ["g"] and out["visible"] == 1 and out["conforms"] and out["decisive"]


def test_exp63_every_aut_trace_needs_an_integer_hippocampus_size(exp63_phases) -> None:
    """Arch SF2: P1 reads the FIRST AUT trace, which need not be a turn's (an empty-goal trace): an unreadable size
    there aborts (C5(b)), and cannot turn a restore that worked into a P1 FAIL."""
    p = copy.deepcopy(exp63_phases)
    first = _aut(p, 0)
    blank = {**copy.deepcopy(first), "goal": "", "memory_ids": [], "memory_paths": [], "memories": 0}
    blank.pop("hippocampus_size")
    p[1]["lines"].insert(p[1]["lines"].index(first), blank)
    assert any(x.startswith("C5(b)") and "hippocampus_size" in x for x in v.exp63_c5_problems(p, goal=GOAL63))


def test_exp63_r3a_is_never_an_empty_goal_trace(exp63_phases) -> None:
    """E3: an empty-goal trace with horizon seq_C (it never ran the goal path) is not R3a's view."""
    p = copy.deepcopy(exp63_phases)
    first = _aut(p, 0)
    blank = {**copy.deepcopy(first), "goal": "", "memory_ids": [], "memory_paths": [], "memories": 0}
    p[1]["lines"].insert(p[1]["lines"].index(first), blank)
    seq_c = v._seq_c(v.store_records(p[0]["store"]))
    assert v.exp63_r3a_trace(p[1]["lines"], seq_c)[1] is first


def test_exp63_a_total_restore_failure_is_a_fail_not_an_abort(exp63_phases) -> None:
    """A2: nothing restored, so no trace can see seq_C. R3a's unobservability aborts only when P1 and P2 hold;
    here they do not (independent bytes), so the attempt is complete and the verdict is FAIL."""
    p = copy.deepcopy(exp63_phases)
    carried = {m["id"] for m in p[0]["store"]["memories"]}
    p[1]["store"]["memories"] = [m for m in p[1]["store"]["memories"] if m["id"] not in carried]
    for r in p[1]["lines"]:
        if r.get("e") == "enrichment_trace" and r.get("agent_id") == "sim_aut":
            r.update(memory_ids=[], memory_paths=[], memories=0, hippocampus_size=0, goal_path_horizon=-1)
    assert not any("(R3a)" in x for x in v.exp63_c5_problems(p, goal=GOAL63))
    out = v.exp63_gates(p, goal=GOAL63)
    assert out["verdict"] == "FAIL" and {"P1", "P2"} <= set(out["not_passed"])
    q = copy.deepcopy(exp63_phases)  # with P1 and P2 holding, the same unobservability still aborts
    _c5_r3a_unobservable(q)
    assert any(x.startswith("C5(R3a)") for x in v.exp63_c5_problems(q, goal=GOAL63))


CORRUPT = [
    lambda tr: tr.update(memory_ids=7),
    lambda tr: tr.update(memory_ids=[1, 2, 3]),
    lambda tr: tr.update(memory_paths=[None, 3, "goal"]),
    lambda tr: tr.update(goal_path_holes="0-3"),
    lambda tr: tr.update(goal_path_holes=[["a", 1]]),
    lambda tr: tr.update(goal_path_horizon=True),
    lambda tr: tr.update(goal=["escape"]),
    lambda tr: tr.update(memories=None, hippocampus_size="8"),
]


@pytest.mark.parametrize("corrupt", CORRUPT)
def test_exp63_a_corrupt_trace_is_incomplete_never_a_crash(exp63_phases, corrupt) -> None:
    """E5: every trace R3a or the turns read, corrupted: C5 records it; the judge does not raise."""
    p = copy.deepcopy(exp63_phases)
    for n in (0, 3):
        corrupt(_aut(p, n))
    assert v.exp63_c5_problems(p, goal=GOAL63)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda p: p[1]["store"].update(memories=None),
        lambda p: p[1]["store"]["memories"][0].update(perception="seen"),
        lambda p: p[1]["store"]["memories"][0].update(timestamp="noon"),
        lambda p: p[1]["store"]["memories"][0].update(activation_sources=["enrichment"]),
        lambda p: p[0].update(store=[]),
    ],
)
def test_exp63_a_corrupt_store_is_incomplete_never_a_crash(exp63_phases, corrupt) -> None:
    p = copy.deepcopy(exp63_phases)
    corrupt(p)
    assert v.exp63_c5_problems(p, goal=GOAL63)


def _keys(scores: dict[str, tuple]) -> dict[str, dict]:
    """Records whose ranker key is exactly the given (score in sixths, timestamp)."""
    words = GOAL63.split()  # 6 distinct tokens: k of them gives score k/6
    return {m: {"id": m, "timestamp": ts, "capture_seq": 0, "context": {"active_goal": " ".join(words[:k])}}
            for m, (k, ts) in scores.items()}  # fmt: skip


def _brute_decisive(records: dict, graph: list[str], carried: set[str]) -> bool:
    """D1 by enumeration: every ordering consistent with the ranking (ties in any order) shows a carried goal id that
    ranks strictly above a visible non-carried record."""
    import itertools

    if len(graph) >= 3:
        return False
    groups = v.ranked_groups(list(records.values()), GOAL63)
    new_keys = [v.ranker_key(records[m], GOAL63) for m in records if m not in carried]
    q = {m for m in carried if new_keys and v.ranker_key(records[m], GOAL63) > min(new_keys)}
    for order in itertools.product(*(itertools.permutations(g) for g in groups)):
        top3 = [m for g in order for m in g][:3]
        shown = [m for m in top3 if m not in graph][: 3 - len(graph)]
        if not set(shown) & q:
            return False
    return True


def test_r3d_is_decided_on_the_memories_shown(exp63_phases) -> None:
    """D1, against enumeration over every tie ordering on random small stores with heavy exact ties."""
    import random

    rng = random.Random(1063)
    seen = {True: 0, False: 0}
    for _ in range(400):
        n = rng.randint(1, 7)
        records = _keys({f"r{i}": (rng.randint(0, 2), float(rng.randint(0, 2))) for i in range(n)})
        carried = {m for m in records if rng.random() < 0.5}
        graph = rng.sample(sorted(records), rng.randint(0, min(3, n)))
        groups = v.ranked_groups(list(records.values()), GOAL63)
        got = v.exp63_decisive(groups, graph, carried, records, GOAL63)
        assert got == _brute_decisive(records, graph, carried), (records, carried, graph)
        seen[got] += 1
    assert min(seen.values()) > 20


def test_r3d_named_cases() -> None:
    def decisive(scores, carried, graph=()):
        records = _keys(scores)
        groups = v.ranked_groups(list(records.values()), GOAL63)
        return v.exp63_decisive(groups, list(graph), set(carried), records, GOAL63)

    # carried above a visible new record, shown: decisive
    assert decisive({"c": (6, 1.0), "n": (1, 2.0)}, {"c"})
    # carried only fills a slot: fewer than 3 new records are visible, and every one outranks it
    assert not decisive({"n1": (6, 1.0), "n2": (6, 1.0), "c": (1, 9.0)}, {"c"})
    assert not decisive({"c": (6, 1.0)}, {"c"})  # no visible new record at all
    # graph ids fill the slots: the goal path never ran
    assert not decisive({"c": (6, 1.0), "n": (1, 2.0)}, {"c"}, graph=["g1", "g2", "g3"])
    # a tie the ordering may resolve against the carried id at the boundary
    assert not decisive({"n0": (6, 5.0), "n1": (6, 4.0), "c": (3, 1.0), "n2": (3, 1.0), "n3": (0, 0.0)}, {"c"})
    # Exp 1: a carried record tied EXACTLY on (score, timestamp) with the lowest visible new record does not rank
    # above it, so it does not qualify (the brute force shares the rule; this case pins the strict comparison)
    assert not decisive({"c": (3, 1.0), "n": (3, 1.0)}, {"c"})
    assert not decisive({"c": (3, 1.0), "n1": (6, 9.0), "n2": (3, 1.0)}, {"c"})
    # a graph id dedups a slot, which pulls the carried id into the shown three
    assert decisive({"n0": (6, 5.0), "n1": (6, 4.0), "c": (3, 1.0), "n3": (0, 0.0)}, {"c"}, graph=["n0"])


def test_exp63_carried_leaving_the_top3_is_reported_forced_and_possible(mock_attempt, monkeypatch) -> None:
    """A8: under ties, the turn the goal top 3 stops HAVING to hold a carried id, and the turn it no longer CAN."""
    out = v.exp63_gates(_exp63(mock_attempt, monkeypatch, "not_shown"), goal=GOAL63)
    assert out["descriptive"]["carried_leave_goal_top3_turn"] == {"forced": 2, "possible": 2}
    assert out["descriptive"]["turns"][1]["carried_in_goal_top3"] == "forced"


def test_a_carried_id_at_a_tie_boundary_is_possible_not_forced() -> None:
    """A8: two new records outrank; the carried record ties a new one for the last slot."""

    def label(scores, ids):
        records = _keys(scores)
        trace = {"memory_ids": ids, "memory_paths": ["goal"] * 3, "goal_path_horizon": 0, "goal_path_holes": []}
        return v.exp63_turn(trace, records, {"c"}, GOAL63)["carried_in_goal_top3"]

    assert label({"n0": (6, 5.0), "n1": (6, 4.0), "c": (3, 1.0), "n2": (3, 1.0)}, ["n0", "n1", "n2"]) == "possible"
    assert label({"n0": (6, 5.0), "n1": (6, 4.0), "n2": (4, 1.0), "c": (3, 1.0)}, ["n0", "n1", "n2"]) == "none"
    assert label({"n0": (6, 5.0), "c": (6, 4.0), "n2": (4, 1.0)}, ["n0", "c", "n2"]) == "forced"


def test_exp63_carried_never_leave_the_top3_in_the_pass_mock(exp63_phases) -> None:
    leave = v.exp63_gates(exp63_phases, goal=GOAL63)["descriptive"]["carried_leave_goal_top3_turn"]
    assert leave == {"forced": None, "possible": None}


def test_the_frozen_ranker_caps_observation_text_at_50_words() -> None:
    """E1: a query word that appears only after word 50 of the observation text does not score (both rankers)."""
    from maxim.memory.hippocampus_retrieval import _rank_by_relevance
    from maxim.memory.types import EpisodicMemory, Perception

    filler = " ".join(f"w{i}" for i in range(50))
    late = EpisodicMemory(id="late", timestamp=2.0, perception=Perception(observations={"text": f"{filler} guard"}))
    early = EpisodicMemory(id="early", timestamp=1.0, perception=Perception(observations={"text": f"guard {filler}"}))
    dicts = [json.loads(json.dumps(m.to_dict())) for m in (late, early)]
    shipped = [m.id for m in _rank_by_relevance([EpisodicMemory.from_dict(d) for d in dicts], "guard", 2)]
    assert shipped == ["early", "late"] == [d["id"] for d in v.rank_by_relevance(dicts, "guard", 2)]
    assert v.ranker_key(dicts[0], "guard") == (0.0, 2.0) and v.ranker_key(dicts[1], "guard") == (1.0, 1.0)


def _refused_probe(monkeypatch, **behaviour):
    import subprocess as sp

    def run(*a, **k):
        if behaviour.get("timeout"):
            raise sp.TimeoutExpired(a[0], 120)
        return sp.CompletedProcess(a, 0, stdout=behaviour.get("stdout", ""), stderr="")

    monkeypatch.setattr(h.subprocess, "run", run)


@pytest.mark.parametrize(
    "behaviour", [{"timeout": True}, {"stdout": ""}, {"stdout": "not json\n"}, {"stdout": "[1]\n"}]
)
def test_a_probe_that_fails_any_way_is_a_refusal(monkeypatch, tmp_path, behaviour) -> None:
    """E2: both preflight probes (memory strategy and model config) refuse; nothing crashes past the marker."""
    _refused_probe(monkeypatch, **behaviour)
    with pytest.raises(h.Refused):
        h.check_memory_strategy(tmp_path)
    with pytest.raises(h.Refused):
        h.check_model_config(tmp_path, "10")


def test_a_refusal_before_the_marker_removes_the_fresh_home(tmp_path, monkeypatch, capsys) -> None:
    """E2: the attempt's fresh home is removed on any refusal (and any failure) before the marker."""
    made = []
    real_mkdtemp = h.tempfile.mkdtemp

    def mkdtemp(*a, **k):
        made.append(real_mkdtemp(*a, **k))
        return made[-1]

    monkeypatch.setattr(h.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: tmp_path / "rows.jsonl")
    for failure in (h.Refused("C4'"), RuntimeError("probe crashed")):

        def boom(home, failure=failure):
            raise failure

        monkeypatch.setattr(h, "check_memory_strategy", boom)
        with pytest.raises(RuntimeError) if isinstance(failure, RuntimeError) else _nullcontext():
            assert h.main(["run", "--exp", "63", "--mock"]) == 2
    homes = [m for m in made if "o19-home-" in m]
    assert len(homes) == 2 and not any(Path(m).exists() for m in homes)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ── #1059: the leaked-gate bar and the subject preflight (the evidence gate is the authority) ────────────────


def _mock_campaign(tmp_path: Path, monkeypatch, exp: str, *, failed_from: int | None = None, edit=None) -> list[dict]:
    """One mock attempt of ``exp`` with its data directory at ``tmp_path/<scope>`` (as a data root holds it); phases
    from ``failed_from`` on are marked failed (an aborted attempt), then ``edit(rows, data_dir)``."""
    data_dir = tmp_path / v.PROTOCOL[exp]["scope"]
    data_dir.mkdir(parents=True, exist_ok=True)
    rows_file = data_dir / "rows.jsonl"
    stable_provenance(monkeypatch)
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows_file)
    import contextlib
    import io

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = h.main(["run", "--exp", exp, "--mock"])
    assert code == 0, f"the mock harness exited {code}: {err.getvalue()}"  # a flaky exit 2 shows its refusal here
    rows = [json.loads(ln) for ln in rows_file.read_text().splitlines()]
    for r in rows[failed_from:] if failed_from is not None else []:
        r["status"] = "failed"
    if edit:
        edit(rows, data_dir)
    rows_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def _thin_phase_0(rows: list[dict], data_dir: Path) -> None:
    """Phase 1's saved store holds 2 memories: Exp 10's P0 (N1 >= 3) fails on it."""
    sdir = data_dir / rows[0]["session_id"]
    data = json.dumps({"memories": [{"id": "a"}, {"id": "b"}]}).encode()
    (sdir / "aut_hippocampus.json").write_bytes(data)
    rows[0]["files"]["aut_hippocampus.json"] = v.sha256_bytes(data)


def test_a_clean_aborted_prefix_leaks_nothing(tmp_path, monkeypatch) -> None:
    for failed_from in (None, 2, 1, 0):
        rows = _mock_campaign(tmp_path / str(failed_from), monkeypatch, "10", failed_from=failed_from)
        assert v.leaked_gate_problems("10", rows, tmp_path / str(failed_from) / "rerun_exp10_o19") == []


@pytest.mark.parametrize("failed_from, gates", [(1, ["P0"]), (2, ["P0", "P2.gate"]), (None, ["P0", "P2"])])
def test_a_failed_gate_in_committed_phases_is_a_leak(tmp_path, monkeypatch, failed_from, gates) -> None:
    rows = _mock_campaign(tmp_path, monkeypatch, "10", failed_from=failed_from, edit=_thin_phase_0)
    problems = v.leaked_gate_problems("10", rows, tmp_path / "rerun_exp10_o19")
    assert len(problems) == 1 and f"a FAILED gate leaked into its committed phases: {gates}" in problems[0], problems


def test_an_exp09_not_met_hypothesis_is_a_leak(tmp_path, monkeypatch) -> None:
    def no_flinch(rows, data_dir):
        import gzip

        sdir = data_dir / rows[0]["session_id"]
        log = gzip.decompress((sdir / f"{v.RUN_LOG}.gz").read_bytes()).replace(b"attack_flinch", b"other_reflex")
        (sdir / f"{v.RUN_LOG}.gz").write_bytes(gzip.compress(log))
        rows[0]["files"][v.RUN_LOG] = v.sha256_bytes(log)
        rows[0]["status"] = "ok"

    rows = _mock_campaign(tmp_path, monkeypatch, "09", edit=no_flinch)
    problems = v.leaked_gate_problems("09", rows, tmp_path / "rerun_exp09_o19")
    assert problems and "'H1'" in problems[0] and "'H3'" not in problems[0], problems


@pytest.mark.parametrize("edit, expected", [
    (lambda rows, d: rows[0]["files"].update({"report.json": "0" * 64}), "cannot be judged"),
    (lambda rows, d: rows[0].update(session_id="../x"), "cannot be judged"),
    (lambda rows, d: (d / rows[0]["session_id"] / "report.json").unlink(), "cannot be judged"),
])  # fmt: skip
def test_a_leaked_phase_that_cannot_be_judged_bars(tmp_path, monkeypatch, edit, expected) -> None:
    rows = _mock_campaign(tmp_path, monkeypatch, "10", failed_from=2, edit=edit)
    problems = v.leaked_gate_problems("10", rows, tmp_path / "rerun_exp10_o19")
    assert problems and expected in problems[0], problems


def test_a_prefix_with_no_known_gate_set_bars(tmp_path, monkeypatch) -> None:
    rows = _mock_campaign(tmp_path, monkeypatch, "10", failed_from=2)
    monkeypatch.setitem(v.LEAK_GATES, "10", {1: ("P0",), 3: ("P0",)})
    problems = v.leaked_gate_problems("10", rows, tmp_path / "rerun_exp10_o19")
    assert problems and "decide no known gate set" in problems[0], problems


def test_the_leak_bar_covers_every_experiment_and_phase_count() -> None:
    for key, p in v.PROTOCOL.items():
        assert set(v.LEAK_GATES[v.experiment_of(key)]) == set(range(1, len(p["phases"]) + 1)), key


def test_a_successor_with_unreadable_predecessor_rows_is_barred(tmp_path) -> None:
    problems = v.chain_leaked_gate_problems("10c2", tmp_path)
    assert problems and "cannot be ruled out" in problems[0], problems


def _pin_closure(data_root: Path, monkeypatch) -> bytes:
    """Close the mock campaign 1 under ``data_root`` with a verdict binding its rows, pinned in campaign 2's entry."""
    import copy as _copy

    pdir = data_root / v.PROTOCOL["10"]["scope"]
    closure = json.dumps({"data": v.rows_path("10"), "data_sha256": v.sha256_bytes((pdir / "rows.jsonl").read_bytes())})
    (pdir / "verdict.json").write_text(closure)
    table = _copy.deepcopy(v.PROTOCOL)
    table["10c2"]["supersedes"]["verdict_sha256"] = v.sha256_bytes(closure.encode())
    monkeypatch.setattr(v, "PROTOCOL", table)
    return closure.encode()


def test_successor_problems_and_the_judge_carry_the_bar(tmp_path, monkeypatch) -> None:
    real = (REPO / v.PROTOCOL["10c2"]["supersedes"]["verdict"]).read_bytes()
    _mock_campaign(tmp_path, monkeypatch, "10", failed_from=2, edit=_thin_phase_0)
    _pin_closure(tmp_path, monkeypatch)
    problems = v.successor_problems("10c2", real, 1.0, 1.0, 2.0, data_root=tmp_path)
    assert any("FAILED gate leaked" in p for p in problems), problems
    rows = _mock_campaign(tmp_path, monkeypatch, "10c2")
    attempts = v.attempts_from_rows(rows)
    rid = next(iter(attempts))
    out = v.judge("10c2", [{"run_id": rid, "k": 1, "rows": attempts[rid]}], tmp_path / "rerun_exp10_o19c2")
    assert out["verdict"] == "PASS" and any("FAILED gate leaked" in p for p in out["leaked_gates"]), out
    root = v.judge("10", [{"run_id": rid, "k": 1, "rows": attempts[rid]}], tmp_path / "rerun_exp10_o19c2")
    assert "leaked_gates" not in root  # a root campaign's output is unchanged (half B)


def test_the_harness_refuses_a_successor_on_a_leak_or_another_subject(monkeypatch) -> None:
    real = (REPO / v.PROTOCOL["10c2"]["supersedes"]["verdict"]).read_bytes()
    monkeypatch.setattr(v, "_git_bytes", lambda *a: real)
    monkeypatch.setattr(v, "landed_on_main", lambda path, want=None: 1.0)
    monkeypatch.setattr(v, "materialize", lambda ref, keys, dest: dest)  # an empty data root: nothing readable
    monkeypatch.setattr(v, "subject_problems", lambda key, head: [])
    with pytest.raises(h.Refused, match="cannot be ruled out"):
        h.check_campaign("10c2")
    monkeypatch.setattr(v, "chain_leaked_gate_problems", lambda key, root: [])
    h.check_campaign("10c2")
    monkeypatch.setattr(v, "subject_problems", lambda key, head: ["the subject differs"])
    with pytest.raises(h.Refused, match="the subject differs"):
        h.check_campaign("10c2")


def test_subject_problems_compare_every_predecessor_commit_with_head(tmp_path, monkeypatch) -> None:
    import copy as _copy

    rig = Rig(tmp_path)
    rows = json.dumps({"record_kind": "harness_row", "provenance": {"executed_git_hash": rig.code}}) + "\n"
    # A header is not an attempt (the gate's _executed reads harness rows only): its commit is never compared.
    rows += json.dumps({"record_kind": "harness_header", "provenance": {"executed_git_hash": "f" * 40}}) + "\n"
    closure = {"data": v.rows_path("10"), "data_sha256": v.sha256_bytes(rows.encode()),
               "apparatus": {"markers": [{"peeled": rig.code}]}}  # fmt: skip
    rig.put(v.rows_path("10"), rows.encode(), "2026-10-01T11:00:00Z")
    raw = json.dumps(closure).encode()
    rig.put(v.PROTOCOL["10c2"]["supersedes"]["verdict"], raw, "2026-10-01T12:00:00Z")
    table = _copy.deepcopy(v.PROTOCOL)
    table["10c2"]["supersedes"]["verdict_sha256"] = v.sha256_bytes(raw)
    monkeypatch.setattr(v, "PROTOCOL", table)
    monkeypatch.setattr(v, "REPO_ROOT", rig.work)
    head = _git(rig.work, "rev-parse", "HEAD")
    assert v.subject_problems("10c2", head) == []  # only data and the closure changed: not subject
    assert v.subject_problems("10", head) == []  # a root supersedes nothing
    rig.put("src/maxim/utils/function_length_baseline.json", b"{}", "2026-10-01T13:00:00Z")
    assert v.subject_problems("10c2", _git(rig.work, "rev-parse", "HEAD")) == []  # the standing exclusion
    rig.put("src/maxim/agent.py", b"x = 2\n", "2026-10-01T14:00:00Z")
    problems = v.subject_problems("10c2", _git(rig.work, "rev-parse", "HEAD"))
    assert problems and "src/maxim/agent.py" in problems[0], problems
    table["10c2"]["supersedes"]["verdict_sha256"] = "0" * 64
    assert any("pinned closure verdict is not on origin/main" in p for p in v.subject_problems("10c2", head))


def test_the_campaign_table_refuses_a_successor_with_other_phases_or_a_shared_kind() -> None:
    c2 = v.PROTOCOL["10c2"]
    other = [*c2["phases"][:2], ("negative_transfer", v.EXP10_GOAL_GARDEN, 5, True, ["--x"], {})]
    assert any(
        "phases (argv, env) are not 10's" in p for p in v.protocol_problems(_table(**{"10c2": {**c2, "phases": other}}))
    )
    shared = _table(**{"63": {**v.PROTOCOL["63"], "kind": "exp09_verdict"}})
    assert any("belongs to more than one experiment" in p for p in v.protocol_problems(shared))


@pytest.mark.parametrize("tamper, expected", [
    ("unpinned", "is not the pinned closure"),
    ("rows", "are not the bytes its pinned closure judged"),
    ("no_closure", "cannot be read as its pinned closure judged"),
])  # fmt: skip
def test_the_bar_reads_a_predecessor_only_as_its_pinned_closure_judged_it(tmp_path, monkeypatch, tamper, expected):
    """#1059 review: rows edited after the closure (here: phase 1 marked failed, which would empty the leaked prefix)
    cannot launder a leak; the bar refuses instead."""
    _mock_campaign(tmp_path, monkeypatch, "10", failed_from=2, edit=_thin_phase_0)
    _pin_closure(tmp_path, monkeypatch)
    assert any("FAILED gate leaked" in p for p in v.chain_leaked_gate_problems("10c2", tmp_path))
    pdir = tmp_path / v.PROTOCOL["10"]["scope"]
    if tamper == "unpinned":
        (pdir / "verdict.json").write_text((pdir / "verdict.json").read_text() + " ")
    elif tamper == "rows":
        rows = [json.loads(ln) for ln in (pdir / "rows.jsonl").read_text().splitlines()]
        rows[0]["status"] = "failed"
        (pdir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        (pdir / "verdict.json").unlink()
    problems = v.chain_leaked_gate_problems("10c2", tmp_path)
    assert len(problems) == 1 and expected in problems[0] and "cannot be ruled out" in problems[0], problems


def test_a_scope_is_one_path_component() -> None:
    """#1081 item 3: the gate places a campaign's directory under its last path component and the leaked-gate bar
    reads ``data_root/<scope>``, so a scope with a ``/`` would refuse forever: the table refuses it up front."""
    assert v.protocol_problems() == []
    for bad in ("rerun/exp10", "..", "Rerun_exp10", "", None):
        problems = v.protocol_problems(_table(**{"09": {**v.PROTOCOL["09"], "scope": bad}}))
        assert any("is not one path component" in p for p in problems), (bad, problems)


def test_the_harness_and_the_gate_read_one_set_of_executed_commits(tmp_path, monkeypatch) -> None:
    """#1081 item 4: ``campaign_commits_on_main`` (harness) and ``_campaign_record`` + ``_executed`` (gate) are two
    readers of the same closures; on one fixture they agree, per predecessor (a harness header's decoy commit is read
    by neither; a marker with no rows is read by both)."""
    import copy as _copy

    import _evidence_records as R

    rig = Rig(tmp_path)
    rig.put("marker_only.py", b"x = 2\n", "2026-10-01T10:30:00Z")
    marker_only = _git(rig.work, "rev-parse", "HEAD")
    rig.put("c2.py", b"x = 3\n", "2026-10-01T10:40:00Z")
    c2 = _git(rig.work, "rev-parse", "HEAD")
    table = _copy.deepcopy(v.PROTOCOL)

    def close(key: str, ran: str, markers: list[str], date: str) -> str:
        rows = json.dumps({"record_kind": "harness_row", "provenance": {"executed_git_hash": ran}}) + "\n"
        rows += json.dumps({"record_kind": "harness_header", "provenance": {"executed_git_hash": "f" * 40}}) + "\n"
        rig.put(v.rows_path(key), rows.encode(), date)
        closure = {"data": v.rows_path(key), "data_sha256": v.sha256_bytes(rows.encode()),
                   "apparatus": {"markers": [{"peeled": m} for m in markers]}}  # fmt: skip
        raw = json.dumps(closure).encode()
        rig.put(f"{v.data_dir(key)}/verdict.json", raw, date)
        return v.sha256_bytes(raw)

    table["10c2"]["supersedes"]["verdict_sha256"] = close(
        "10", rig.code, [rig.code, marker_only], "2026-10-01T11:00:00Z"
    )
    pin_c2 = close("10c2", c2, [c2], "2026-10-01T12:00:00Z")
    table["10c3"] = {**table["10c2"], "scope": "rerun_exp10_o19c3", "prereg": "c3.md",
                     "supersedes": {**table["10c2"]["supersedes"], "key": "10c2", "verdict_sha256": pin_c2,
                                    "verdict": f"{v.data_dir('10c2')}/verdict.json"}}  # fmt: skip
    monkeypatch.setattr(v, "PROTOCOL", table)
    monkeypatch.setattr(v, "REPO_ROOT", rig.work)
    base = _git(rig.work, "rev-parse", "origin/main")
    ctx = R.Ctx(repo=R.Repo(rig.work), base=base, ref=base, legacy={}, prereg={}, table={}, retired={})

    def gate_reads(succ: str) -> set:
        found: set = set()
        for pred in v.predecessors(succ):
            sup = table[{"10": "10c2", "10c2": "10c3"}[pred]]["supersedes"]
            found |= R._executed(*R._campaign_record(ctx, sup["verdict"], sup["verdict_sha256"]))
        return found

    assert v.campaign_commits_on_main("10c2") == ({rig.code, marker_only}, [])
    assert v.campaign_commits_on_main("10c2")[0] == gate_reads("10c2")
    assert v.campaign_commits_on_main("10c3") == ({rig.code, marker_only, c2}, [])
    assert v.campaign_commits_on_main("10c3")[0] == gate_reads("10c3")


def test_a_mock_attempt_does_not_read_the_live_tree(tmp_path, monkeypatch) -> None:
    """#1081 item 6: the mock helpers serve one captured provenance block, so a live tree that cannot be hashed (a
    file created and removed mid-hash makes ``code_tree_sha256`` "unknown") no longer makes a mock attempt exit 2."""
    stable_provenance(monkeypatch)  # captured while the tree hashes
    monkeypatch.setattr(h._provenance, "code_tree_sha256", lambda *a, **k: "unknown: a file vanished mid-hash")
    rows = _mock_campaign(tmp_path, monkeypatch, "10")
    assert rows and all(r["provenance"]["code_tree_sha256"] == _PROVENANCE["block"]["code_tree_sha256"] for r in rows)
    assert len({r["provenance"]["harness_run_id"] for r in rows}) == 1  # the live run id, minted per attempt


# ── #1079: the leaked-gate bar WITHIN one campaign (new campaign keys only; owner decision 2026-10-08, REFUSE) ─────


def _two_attempts(tmp_path: Path, monkeypatch, exp: str, *, failed_from=None, edit=None, rowless: bool = False):
    """Campaign ``exp`` with an aborted attempt 1 (its phases from ``failed_from`` on failed, then ``edit(rows,
    data_dir)``; ``rowless``: a start marker whose rows never landed) and a complete mock attempt 2, as the judge
    reads them: ``([{run_id, k, rows}] in k order, the campaign's data directory)``."""
    data_dir = tmp_path / v.PROTOCOL[exp]["scope"]
    if rowless:
        _mock_campaign(tmp_path, monkeypatch, exp)
        rows = [json.loads(ln) for ln in (data_dir / "rows.jsonl").read_text().splitlines()]
        rid = rows[0]["provenance"]["harness_run_id"]
        return [{"run_id": "0" * 32, "k": 1, "rows": []}, {"run_id": rid, "k": 2, "rows": rows}], data_dir
    _mock_campaign(tmp_path, monkeypatch, exp, failed_from=failed_from, edit=edit)
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    first = (data_dir / "rows.jsonl").read_text()
    (data_dir / "rows.jsonl").write_text("")  # the harness's own reading may call an incomplete all-ok attempt complete
    assert h.main(["run", "--exp", exp, "--mock"]) == 0  # attempt 2
    (data_dir / "rows.jsonl").write_text(first + (data_dir / "rows.jsonl").read_text())
    attempts = v.attempts_from_rows([json.loads(ln) for ln in (data_dir / "rows.jsonl").read_text().splitlines()])
    assert len(attempts) == 2
    ordered = [{"run_id": rid, "k": i + 1, "rows": rs} for i, (rid, rs) in enumerate(attempts.items())]
    return ordered, data_dir


@pytest.fixture
def new_keys(monkeypatch):
    """Every campaign judged as a NEW key: the within-campaign bar applies (no real campaign is new yet)."""
    monkeypatch.setattr(v, "PRE_1079_KEYS", frozenset())


def _served_never_read(rows, data_dir):
    """Attempt 1's only phase is committed ok but INCOMPLETE (C4: the served model was never read): N1."""
    rows[0]["served_model"] = {**(rows[0].get("served_model") or {}), "reads": []}


def _no_flinch(rows, data_dir):
    import gzip

    sdir = data_dir / rows[0]["session_id"]
    log = gzip.decompress((sdir / f"{v.RUN_LOG}.gz").read_bytes()).replace(b"attack_flinch", b"other_reflex")
    (sdir / f"{v.RUN_LOG}.gz").write_bytes(gzip.compress(log))
    rows[0]["files"][v.RUN_LOG] = v.sha256_bytes(log)


def _list_store(rows, data_dir):
    """Phase 1's saved store is a JSON list: the judge reads it, but no gate can (an unjudgeable prefix)."""
    data = b"[]"
    (data_dir / rows[0]["session_id"] / "aut_hippocampus.json").write_bytes(data)
    rows[0]["files"]["aut_hippocampus.json"] = v.sha256_bytes(data)


def test_exp10_a_leak_in_an_earlier_attempt_is_reported(tmp_path, monkeypatch, new_keys) -> None:
    """The known answer (#1079 section 1): attempt 1 committed phases 1-2 ok with a 2-memory store (P0 FAILED), then
    died; attempt 2 PASSes. The verdict and its deciding attempt are unchanged; the bar names attempt 1's P0."""
    ordered, data_dir = _two_attempts(tmp_path, monkeypatch, "10", failed_from=2, edit=_thin_phase_0)
    out = v.judge("10", ordered, data_dir)
    assert out["verdict"] == "PASS" and out["deciding_attempt"] == ordered[1]["run_id"]
    assert out["within_campaign_leaks"] == [[ordered[0]["run_id"], 1, ["P0", "P2.gate"]]]
    assert len(out["within_campaign_leak_notes"]) == 1 and "'P0'" in out["within_campaign_leak_notes"][0]


def test_a_clean_earlier_attempt_leaks_nothing(tmp_path, monkeypatch, new_keys) -> None:
    for failed_from in (2, 1, 0):
        ordered, data_dir = _two_attempts(tmp_path / str(failed_from), monkeypatch, "10", failed_from=failed_from)
        out = v.judge("10", ordered, data_dir)
        assert out["verdict"] == "PASS" and out["within_campaign_leaks"] == [], (failed_from, out)


def test_an_abort_lists_every_attempts_leak(tmp_path, monkeypatch, new_keys) -> None:
    ordered, data_dir = _two_attempts(tmp_path, monkeypatch, "10", failed_from=2, edit=_thin_phase_0)
    ordered[1]["rows"][-1]["status"] = "failed"  # attempt 2 aborts too, in its garden phase, after a clean prefix
    out = v.judge("10", ordered, data_dir)
    assert out["verdict"] == "ABORT" and [a for _r, a, _s in out["within_campaign_leaks"]] == [1]
    ordered[1]["rows"][0], ordered[1]["rows"][1:] = (
        ordered[0]["rows"][0] | {"provenance": ordered[1]["rows"][0]["provenance"]},
        [],
    )
    out = v.judge("10", ordered, data_dir)
    assert [a for _r, a, _s in out["within_campaign_leaks"]] == [1, 2], out["within_campaign_leaks"]


def test_exp63_an_earlier_phase_1_with_fewer_than_3_records_leaks_p0(tmp_path, monkeypatch, new_keys) -> None:
    ordered, data_dir = _two_attempts(tmp_path, monkeypatch, "63", failed_from=1, edit=_thin_phase_0)
    out = v.judge("63", ordered, data_dir)
    assert out["deciding_attempt"] == ordered[1]["run_id"]
    assert out["within_campaign_leaks"] == [[ordered[0]["run_id"], 1, ["P0"]]]


def test_exp09_an_incomplete_earlier_attempt_is_still_judged(tmp_path, monkeypatch, new_keys) -> None:
    """N1 (strict): an earlier attempt the judge found C4-invalid still had its committed phases seen."""
    ordered, data_dir = _two_attempts(tmp_path / "clean", monkeypatch, "09", edit=_served_never_read)
    out = v.judge("09", ordered, data_dir)
    assert [a["complete"] for a in out["attempts"]] == [False, True] and out["verdict"] == "PARTIAL"
    assert out["within_campaign_leaks"] == []  # incomplete, but every decided gate passed: nothing leaked

    def both(rows, d):
        _served_never_read(rows, d)
        _no_flinch(rows, d)

    ordered, data_dir = _two_attempts(tmp_path / "leak", monkeypatch, "09", edit=both)
    out = v.judge("09", ordered, data_dir)
    (leak,) = out["within_campaign_leaks"]
    assert leak[:2] == [ordered[0]["run_id"], 1] and "H1" in leak[2] and "H3" not in leak[2], leak


def test_an_earlier_prefix_that_cannot_be_judged_bars(tmp_path, monkeypatch, new_keys) -> None:
    ordered, data_dir = _two_attempts(tmp_path / "store", monkeypatch, "10", failed_from=2, edit=_list_store)
    out = v.judge("10", ordered, data_dir)
    assert out["within_campaign_leaks"] == [[ordered[0]["run_id"], 1, "unjudgeable"]]
    assert "cannot be judged" in out["within_campaign_leak_notes"][0]
    ordered, data_dir = _two_attempts(tmp_path / "set", monkeypatch, "10", failed_from=2)
    monkeypatch.setitem(v.LEAK_GATES, "10", {1: ("P0",), 3: ("P0",)})
    out = v.judge("10", ordered, data_dir)
    assert out["within_campaign_leaks"] == [[ordered[0]["run_id"], 1, "unjudgeable"]]


def test_a_rowless_earlier_marker_bars(tmp_path, monkeypatch, new_keys) -> None:
    """D1 (owner decision 2026-10-08): an earlier start marker whose rows never reached main cannot be judged, so it
    bars: otherwise resetting the local rows file before the next attempt would launder a leak."""
    ordered, data_dir = _two_attempts(tmp_path, monkeypatch, "10", rowless=True)
    out = v.judge("10", ordered, data_dir)
    assert out["verdict"] == "PASS" and out["within_campaign_leaks"] == [["0" * 32, 1, "rowless"]]


def test_the_deciding_attempt_and_later_ones_are_never_read(tmp_path, monkeypatch, new_keys) -> None:
    """The deciding attempt's own FAILED gate (Exp 09: H1 not met, so a FAIL) is the verdict's, never a leak."""
    rows = _mock_campaign(tmp_path, monkeypatch, "09", edit=_no_flinch)
    data_dir = tmp_path / v.PROTOCOL["09"]["scope"]
    rid = rows[0]["provenance"]["harness_run_id"]
    assert v._attempt_leak("09", "x", rows, data_dir) is not None  # it HAS a failed decided gate (non-vacuous)
    out = v.judge("09", [{"run_id": rid, "k": 1, "rows": rows}], data_dir)
    assert out["deciding_attempt"] == rid and out["verdict"] == "FAIL", out
    assert out["within_campaign_leaks"] == [] and out["within_campaign_leak_notes"] == []
    # judge() refuses an attempt after the complete one, so the "later" half is pinned on the function itself: an
    # attempt after the deciding one (the same failed-gate bytes under another run id) is never read either.
    later = {"run_id": "f" * 32, "k": 2, "rows": rows}
    assert v.within_campaign_leaks("09", [{"run_id": rid, "k": 1, "rows": rows}, later], rid, data_dir) == ([], [])


def test_a_pre_1079_campaigns_output_is_unchanged(tmp_path, monkeypatch) -> None:
    ordered, data_dir = _two_attempts(tmp_path, monkeypatch, "10", failed_from=2, edit=_thin_phase_0)
    out = v.judge("10", ordered, data_dir)
    assert "within_campaign_leaks" not in out and "within_campaign_leak_notes" not in out


def _main_judge():
    """``origin/main``'s judge, loaded as the gate loads one (None when this clone has no origin/main)."""
    import subprocess

    import _evidence_records as R

    src = subprocess.run(["git", "show", f"origin/main:{R.O19_JUDGE}"], cwd=REPO, capture_output=True).stdout
    mod = R.load_o19_judge(src) if src else None
    return None if isinstance(mod, str) else mod


@pytest.mark.skipif(_main_judge() is None, reason="needs origin/main's judge")
@pytest.mark.parametrize("exp", ["10", "10c2", "09", "63"])
def test_every_pre_1079_campaign_judges_exactly_as_mains_judge(tmp_path, monkeypatch, exp) -> None:
    """The four campaigns' FULL judge() output is main's, on a multi-attempt fixture (not only the compared fields)."""
    edit = _served_never_read if v.experiment_of(exp) == "09" else _thin_phase_0
    ordered, data_dir = _two_attempts(tmp_path, monkeypatch, exp, failed_from=None if exp == "09" else 1, edit=edit)
    old = _main_judge()
    assert v.judge(exp, copy.deepcopy(ordered), data_dir) == old.judge(exp, copy.deepcopy(ordered), data_dir)


@pytest.mark.skipif(_main_judge() is None, reason="needs origin/main's judge")
@pytest.mark.parametrize("failed_from, edit", [
    (None, None), (2, None), (1, _thin_phase_0), (2, _thin_phase_0), (None, _thin_phase_0), (2, _list_store),
    (2, lambda rows, d: rows[0]["files"].update({"report.json": "0" * 64})),
    (2, lambda rows, d: rows[0].update(session_id="../x")),
])  # fmt: skip
def test_the_extracted_leak_reading_is_byte_equal_to_mains(tmp_path, monkeypatch, failed_from, edit) -> None:
    rows = _mock_campaign(tmp_path, monkeypatch, "10", failed_from=failed_from, edit=edit)
    old = _main_judge()
    for gates in (None, {1: ("P0",), 3: ("P0",)}):
        if gates is not None:
            monkeypatch.setitem(v.LEAK_GATES, "10", gates)
            monkeypatch.setitem(old.LEAK_GATES, "10", gates)
        new = v.leaked_gate_problems("10", rows, tmp_path / "rerun_exp10_o19")
        assert new == old.leaked_gate_problems("10", rows, tmp_path / "rerun_exp10_o19")


def test_pre_1079_keys_are_the_four_campaigns_with_a_verdict_on_main() -> None:
    assert v.PRE_1079_KEYS == {"10", "10c2", "09", "63"}
    for key in v.PRE_1079_KEYS:
        assert (REPO / v.data_dir(key) / "verdict.json").is_file(), key


def test_the_campaign_table_refuses_a_campaign_the_leak_bar_cannot_read() -> None:
    table = _table()
    table["09"]["phases"] = [*table["09"]["phases"], table["09"]["phases"][0]]  # a second phase: no gate set for 2
    assert any("LEAK_GATES does not cover" in p for p in v.protocol_problems(table))


def _materialized(src: Path):
    """A ``materialize`` stand-in: the campaign's directory as ``src`` holds it (the harness reads origin/main's)."""
    import shutil

    def materialize(ref, keys, dest):
        assert ref == "origin/main"
        for key in keys:
            shutil.copytree(src, dest / v.PROTOCOL[key]["scope"])
        return dest

    return materialize


def _markers(ordered: list[dict]) -> dict:
    return {a["run_id"]: {"k": a["k"]} for a in ordered}


def test_the_harness_refuses_a_campaign_whose_earlier_attempt_leaked(tmp_path, monkeypatch, new_keys) -> None:
    ordered, data_dir = _two_attempts(tmp_path, monkeypatch, "10", failed_from=2, edit=_thin_phase_0)
    earlier = ordered[:1]  # attempt 2 is about to start: only attempt 1 is on main
    (data_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in earlier[0]["rows"]))
    monkeypatch.setattr(v, "materialize", _materialized(data_dir))
    with pytest.raises(h.Refused, match="#1079"):
        h.check_within_campaign("10", _markers(earlier))
    rowless = {**_markers(earlier), "0" * 32: {"k": 2}}  # D1 + N4: a marker origin lists with no rows on main
    (data_dir / "rows.jsonl").write_text("")
    with pytest.raises(h.Refused, match="rows never reached main"):
        h.check_within_campaign("10", rowless)


def test_the_harness_lets_a_clean_campaign_or_a_pre_1079_key_continue(tmp_path, monkeypatch) -> None:
    ordered, data_dir = _two_attempts(tmp_path, monkeypatch, "10", failed_from=2)
    (data_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ordered[0]["rows"]))
    monkeypatch.setattr(v, "materialize", _materialized(data_dir))
    monkeypatch.setattr(v, "PRE_1079_KEYS", frozenset())
    h.check_within_campaign("10", _markers(ordered[:1]))
    monkeypatch.setattr(v, "PRE_1079_KEYS", frozenset({"10"}))
    monkeypatch.setattr(v, "materialize", lambda *a: pytest.fail("a pre-#1079 key reads nothing"))
    h.check_within_campaign("10", {"0" * 32: {"k": 1}})


def test_the_harness_runs_the_within_campaign_bar_before_any_marker(monkeypatch, capsys) -> None:
    monkeypatch.setattr(h._provenance, "assert_repo_interpreter", lambda *a, **k: None)
    monkeypatch.setattr(h._provenance, "executed_code_provenance", lambda *a, **k: {})
    monkeypatch.setattr(h._provenance, "append_refusal", lambda *a, **k: None)
    monkeypatch.setattr(h, "take_lock", lambda exp: None)
    monkeypatch.setattr(h, "check_on_main", lambda rows_file: None)
    monkeypatch.setattr(h, "check_campaign", lambda exp: None)
    monkeypatch.setattr(h, "read_rows", lambda path: [])
    monkeypatch.setattr(h, "remote_markers", lambda exp: {"0" * 32: {"k": 1}})
    seen = []

    def refuse(exp, markers):
        seen.append((exp, markers))
        raise h.Refused("an earlier attempt leaked (#1079)")

    monkeypatch.setattr(h, "check_within_campaign", refuse)
    pushed = []
    monkeypatch.setattr(h, "push_marker", lambda *a, **k: pushed.append(a))
    assert h.main(["run", "--exp", "09", "--write-experiment-results"]) == 2
    assert "#1079" in capsys.readouterr().err and pushed == [] and seen == [("09", {"0" * 32: {"k": 1}})]


def test_the_verdict_writer_prints_a_within_campaign_leak(monkeypatch, capsys, tmp_path) -> None:
    """N3: the verdict reports (stderr), the gate refuses; the exit code stays the verdict's."""
    monkeypatch.setattr(v, "judge", lambda exp, ordered, root: {
        "verdict": "PASS", "deciding_attempt": "b", "attempts": [], "experiment": exp,
        "within_campaign_leaks": [["a", 1, ["P0"]]], "within_campaign_leak_notes": ["attempt 1: P0 leaked"]})  # fmt: skip
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"record_kind": "harness_row", "mock": True, "ts": 1.0,
                                "provenance": {"harness_run_id": "a"}}) + "\n")  # fmt: skip
    assert v.main(["--exp", "09", "--data", str(rows), "--json", str(tmp_path / "v.json"), "--offline"]) == 0
    assert "WITHIN-CAMPAIGN LEAK (#1079" in capsys.readouterr().err
    written = json.loads((tmp_path / "v.json").read_text())  # S1: the prose (host paths, exception text) stays off disk
    assert written["within_campaign_leaks"] == [["a", 1, ["P0"]]] and "within_campaign_leak_notes" not in written


def test_the_harness_refuses_when_mains_rows_cannot_be_read(tmp_path, monkeypatch, capsys) -> None:
    """A malformed rows line on origin/main is a refusal (exit 2), never a ValueError traceback."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "rows.jsonl").write_text('{"record_kind": "harness_row"\n')
    monkeypatch.setattr(v, "PRE_1079_KEYS", frozenset())
    monkeypatch.setattr(v, "materialize", _materialized(src))
    with pytest.raises(h.Refused, match="cannot be read"):
        h.check_within_campaign("09", {"0" * 32: {"k": 1}})
    monkeypatch.setattr(h._provenance, "assert_repo_interpreter", lambda *a, **k: None)
    monkeypatch.setattr(h._provenance, "executed_code_provenance", lambda *a, **k: {})
    monkeypatch.setattr(h._provenance, "append_refusal", lambda *a, **k: None)
    monkeypatch.setattr(h, "take_lock", lambda exp: None)
    monkeypatch.setattr(h, "check_on_main", lambda rows_file: None)
    monkeypatch.setattr(h, "check_campaign", lambda exp: None)
    real_read_rows = h.read_rows  # the materialized copy of main is read for real; the local rows file is empty
    monkeypatch.setattr(h, "read_rows", lambda path: [] if Path(path).is_relative_to(REPO) else real_read_rows(path))
    monkeypatch.setattr(h, "remote_markers", lambda exp: {"0" * 32: {"k": 1}})
    pushed = []
    monkeypatch.setattr(h, "push_marker", lambda *a, **k: pushed.append(a))
    assert h.main(["run", "--exp", "09", "--write-experiment-results"]) == 2
    assert "REFUSED" in capsys.readouterr().err and pushed == []


# ── #1166: the host identity stays out of the rows and the copies (synthetic names, TEST-NET addresses) ──────────

import socket  # noqa: E402

FQDN = "box.example-isp.net"


def _isp_host(monkeypatch, name: str = FQDN, address: str = "203.0.113.7") -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: name)
    monkeypatch.setattr(socket, "gethostbyname", lambda _n: address)  # no real DNS lookup


def _with_log_line(monkeypatch, line: str, on_index: int = 0) -> None:
    """The mock phase's run log gains ``line`` (a heartbeat or a message the sim might log)."""
    real = h.mock_phase

    def phase(exp, index, **kw):
        sdir, report, fields, log, console, error = real(exp, index, **kw)
        if index == on_index:
            with log.open("a") as f:
                f.write(line + "\n")
        return sdir, report, fields, log, console, error

    monkeypatch.setattr(h, "mock_phase", phase)


def _host_run(tmp_path, monkeypatch) -> tuple[int, list[dict]]:
    stable_provenance(monkeypatch)
    rows = tmp_path / "rows.jsonl"
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows)
    code = h.main(["run", "--exp", "10", "--mock"])
    return code, [json.loads(ln) for ln in rows.read_text().splitlines()] if rows.exists() else []


def _heartbeat(hostname: str) -> str:
    return json.dumps(
        {
            "t": 1.0,
            "l": "DEBUG",
            "s": "heartbeat",
            "e": "heartbeat",
            "network": {"hostname": hostname, "local_ip": "10.0.0.5"},
        }
    )


def test_the_row_stamps_the_short_hostname(monkeypatch, tmp_path) -> None:
    _isp_host(monkeypatch)
    _sdir, _report, fields, *_rest = _spawn(monkeypatch, tmp_path, _FakeSim([0]), lambda *a: {})
    assert fields["hostname"] == "box"


def test_a_clean_attempt_under_an_isp_hostname_still_passes(tmp_path, monkeypatch) -> None:
    """The check does not fire on the mock's own bytes, and the copies still verify (the hash binds what is committed)."""
    _isp_host(monkeypatch)
    code, rows = _host_run(tmp_path, monkeypatch)
    assert code == 0 and [r["status"] for r in rows] == ["ok", "ok", "ok"]
    assert _judge("10", tmp_path / "rows.jsonl")["verdict"] == "PASS"


@pytest.mark.parametrize(
    "line, check",
    [
        (_heartbeat(FQDN), "known value"),
        (_heartbeat("lab-7.example-isp.org"), "heartbeat network.hostname"),  # no known value: the structured half
        (json.dumps({"e": "log", "msg": "connect to BOX.Example-ISP.net. refused"}), "known value"),  # case, message
    ],
)
def test_a_copy_carrying_the_full_hostname_is_refused_and_writes_nothing(tmp_path, monkeypatch, line, check) -> None:
    _isp_host(monkeypatch)
    _with_log_line(monkeypatch, line)
    code, rows = _host_run(tmp_path, monkeypatch)
    assert code == 1 and len(rows) == 1  # phase 2 never ran
    row = rows[0]
    assert row["status"] == "failed" and row["reason"] == f"{h.HOST_IN_COPY}: {v.RUN_LOG} ({check})"
    assert "files" not in row and not (tmp_path / row["session_id"]).exists()  # scanned before anything was written
    assert "example-isp" not in json.dumps(rows).lower()  # the reason names the file, never the value


def test_the_phase_start_sample_is_a_needle(tmp_path, monkeypatch) -> None:
    """macOS renames the host on a DHCP renew: a name seen only at phase start is still searched for at copy time."""
    _isp_host(monkeypatch)
    real = h.mock_phase

    def renew(exp, index, **kw):
        out = real(exp, index, **kw)
        with out[3].open("a") as f:
            f.write(json.dumps({"e": "log", "msg": f"resolved {FQDN}"}) + "\n")
        monkeypatch.setattr(socket, "gethostname", lambda: "box")  # the renew: copy time sees the bare name
        return out

    monkeypatch.setattr(h, "mock_phase", renew)
    code, rows = _host_run(tmp_path, monkeypatch)
    assert code == 1 and rows[0]["reason"].endswith("(known value)")


@pytest.mark.parametrize(
    "text",
    [
        "http://127.0.0.1:8100/v1 localhost",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "https://api.anthropic.com/v1 https://huggingface.co/x",
        "docs at https://docs.example.org/maxim",
        "box.local box.lan box.home.arpa",
        "pymaxim 1.3.1",
        "example-isp.net",  # a bare suffix is never a needle
    ],
)
def test_ordinary_bytes_are_copied(tmp_path, text) -> None:
    sdir = tmp_path / "s"
    sdir.mkdir()
    (sdir / "actions.jsonl").write_text(json.dumps({"e": "log", "msg": text}) + "\n" + _heartbeat("box") + "\n")
    (sdir / "report.json").write_text(json.dumps({"note": text}))
    files, problem = h.copy_session(
        sdir, None, None, tmp_path / "out", hostnames=(FQDN, "box", "localhost", "box.local")
    )
    assert problem is None and set(files) == {"actions.jsonl", "report.json"}


def test_only_full_public_hostnames_are_needles() -> None:
    assert h.host_needles(["box", "", None, "box.lan", "box.local", "box.home.arpa", "x.ec2.internal", "box."]) == []
    assert h.host_needles(["BOX.Example-ISP.net."]) == [FQDN.encode()]


@pytest.mark.parametrize(
    "patch_name, value",
    [
        ("short_hostname", lambda name=None: socket.gethostname()),  # a regressed reducer
        ("collect_network_interfaces", lambda: {"hostname": "box", "local_ip": "203.0.113.7"}),
        ("collect_network_interfaces", lambda: {"hostname": "c-203-0-113-7", "local_ip": "10.0.0.5"}),
        ("collect_network_interfaces", lambda: {"hostname": FQDN, "local_ip": "10.0.0.5"}),
        ("collect_network_interfaces", lambda: {"hostname": "cb-00-71-07", "local_ip": "10.0.0.5"}),
        ("recordable_ip", lambda raw: raw),  # a regressed address reducer: the check states its own rule
    ],
)
def test_an_unreduced_host_identity_is_refused_before_the_marker(tmp_path, monkeypatch, capsys, patch_name, value):
    from maxim.runtime import system_metrics

    _isp_host(monkeypatch)
    monkeypatch.setattr(system_metrics, patch_name, value)
    code, rows = _host_run(tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 2 and rows == [] and "#1166" in err and "example-isp" not in err and "203.0.113" not in err


@pytest.mark.parametrize("name", ["p5b0c1d2e", "cb007107", "cb-00-71-07", "box-203-0-113", "c-203-0-113-7"])
def test_an_identity_reducer_cannot_pass_the_pre_marker_check(tmp_path, monkeypatch, capsys, name) -> None:
    """Deletion probe kept as a test: with ``short_hostname`` the identity, a bare address-encoded name (no dot for
    the old rule to catch) is still refused, because the check's rule is not the reducer's."""
    from maxim.runtime import system_metrics

    _isp_host(monkeypatch, name=name, address="10.0.0.5")
    monkeypatch.setattr(system_metrics, "short_hostname", lambda name=None: socket.gethostname())
    code, rows = _host_run(tmp_path, monkeypatch)
    assert code == 2 and rows == [] and "#1166" in capsys.readouterr().err


@pytest.mark.parametrize(
    "value, unreduced",
    [
        ("box", False),
        ("big-mac-mini", False),
        ("raspberrypi", False),
        ("dennys-mbp", False),
        ("ip-encoded", False),
        ("unknown", False),
        ("mac-mini-m4", False),
        (FQDN, True),
        ("box.", True),
        ("2001:db8::7", True),
        ("203.0.113.7", True),
        ("c-203-0-113-7", True),
        ("ip-10-0-0-5", True),
        ("box-203-0-113", True),
        ("p5b0c1d2e", True),
        ("cb007107", True),
        ("cb-00-71-07", True),
        ("deadbeef", True),
        ("mac-mini-2024", True),  # the stated cost of the stricter rule: refused before any marker
        ("", True),
        (None, True),
    ],
)
def test_the_pre_marker_hostname_rule(value, unreduced) -> None:
    assert h.unreduced_hostname(value) is unreduced


@pytest.mark.parametrize(
    "value, private",
    [
        ("10.0.0.5", True),
        ("172.16.4.4", True),
        ("192.168.1.20", True),
        ("127.0.0.1", True),
        ("169.254.3.4", True),
        ("fd00::5", True),
        ("fe80::1%en0", True),
        ("unknown", True),
        ("non-private", True),
        ("203.0.113.7", False),
        ("198.51.100.9", False),
        ("100.64.0.9", False),
        ("2001:db8::7", False),
        ("box", False),
        (None, False),
    ],
)
def test_the_pre_marker_address_rule(value, private) -> None:
    assert h.recorded_ip_is_private(value) is private


@pytest.mark.parametrize("data", ['"x"', "[1]", "null", "7"])
def test_a_heartbeat_with_a_non_dict_data_is_not_an_error(data) -> None:
    line = f'{{"e": "heartbeat", "data": {data}}}'.encode()
    assert h.heartbeat_hostname_dotted(line) is False
