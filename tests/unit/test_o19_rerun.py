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
_EVIDENCE_OUT_PATH = (
    h._provenance.evidence_out_path
)  # the fixture redirects the harness's rows; the verdict needs the real one


# ── the protocol is the prereg's ─────────────────────────────────────────────────────────────────────────


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_goals_caps_and_scope_are_the_preregs() -> None:
    p10, p09 = _squash(PREREG["10"]), _squash(PREREG["09"])
    assert f"`{v.EXP10_GOAL_DUNGEON}` | 8 | none" in p10 and f"`{v.EXP10_GOAL_DUNGEON}` | 8 | phase 1" in p10
    assert f"`{v.EXP10_GOAL_GARDEN}` | 5 | phase 1" in p10
    assert f'*"{v.EXP09_GOAL}"*' in p09
    assert "`--embodiment bodies/base_humanoid`, `--sim-max-turns 8`, `MAXIM_SUBSTRATE_PATH=1`" in p09
    for exp in v.PROTOCOL:
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
    assert v.phase_argv("09", 0, None)[-2:] == ["--embodiment", "bodies/base_humanoid"]
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

    REL = v.rows_path("10")

    def __init__(self, tmp: Path):
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
        name = f"o19/10/attempt-{k}-{rid}"
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
        return v.check_apparatus("10", self.REL, data, v.attempts_from_rows(rows))


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
    assert h.base_env()[1] == ["MAXIM_SUBSTRATE_ACTIONS_PER_TURN", "MAXIM_SUBSTRATE_PATH"]


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
    rows_file = rig.work / Rig.REL
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
