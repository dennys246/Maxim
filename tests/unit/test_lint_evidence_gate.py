"""The ledger evidence gate, M1b PR 5b-1 (scripts/lint_evidence_gate.py; spec: docs/plans/m1b_ledger_evidence_gate.md,
"PR 5b build spec").

Each test builds a throwaway git repo: a BASE commit (the ledger, the pass table, the legacy snapshot, the O19 judge) and
a HEAD commit that changes a row, then runs ``gate()`` on it. The O19 records are a real attempt written by the
harness's mock path and judged by the real ``o19_verdict.judge``, then re-stamped as a non-mock, apparatus-checked
verdict, so the gate's re-judge exercises the shipped code.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import _evidence_records as R  # noqa: E402
import lint_evidence_gate as G  # noqa: E402

LEDGER = "docs/plans/behavioral_graduation_candidates.md"
PREREG_CLAUSE = {"id": "x1", "kind": "prereg", "path": "docs/experiments/data/legacy_old.jsonl", "sha256": "s",
                 "owner": "o", "reason": "r", "date": "2026-10-01"}  # fmt: skip
T1 = "| ID | Claim | Bio-mechanism | Status |\n|---|---|---|---|\n"
T3 = "| ID | CLAUDE.md ref | Mechanism | Bio-claim | Graduation predicate | Status |\n|---|---|---|---|---|---|\n"
BASE_DATE = "2026-10-01T00:00:00+00:00"
HEAD_DATE = "2026-10-02T00:00:00+00:00"
DATA = "docs/experiments/data"


def _git(root: Path, *args: str, date: str = BASE_DATE) -> str:
    env = {**os.environ, "GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date}
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgSign=false", *args],
        cwd=root, env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip


def ledger(t1_rows: list[str], t3_rows: list[str]) -> str:
    return "# Ledger\n\n" + T1 + "\n".join(t1_rows) + "\n\n## Tier 3\n\n" + T3 + "\n".join(t3_rows) + "\n"


def t1(row_id: str, status: str) -> str:
    return f"| {row_id} | claim {row_id} | mech | {status} Regression guard: x. |"


def t3(row_id: str, status: str) -> str:
    return f"| {row_id} | ref | mech | bio {row_id} | pred | {status} Regression guard: x. |"


class Rig:
    """A temp repo: ``base()`` commits the BASE, ``head()`` the change; ``run()`` gates HEAD against BASE."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True)
        _git(root, "init", "-q", "-b", "main")
        (root / "scripts").mkdir()
        shutil.copy(REPO / "scripts" / "o19_verdict.py", root / "scripts" / "o19_verdict.py")
        shutil.copy(REPO / "scripts" / "o19_rerun.py", root / "scripts" / "o19_rerun.py")
        import o19_verdict as v

        for p in v.PROTOCOL.values():  # every campaign's prereg: a verdict binds its own
            self.write(p["prereg"], (REPO / p["prereg"]).read_bytes())
        self.write(G.EXCEPTIONS, "[]\n")
        self.write(G.LEGACY_SNAPSHOT, "{}\n")
        self.write(G.PASS_TABLE, (REPO / G.PASS_TABLE).read_text())
        self.write(f"{DATA}/legacy_old.jsonl", '{"x": 1}\n')
        # Every rig data entry is PASS unless a test says otherwise (the gate refuses a record with no status).
        self.prereg = {DATA: "PASS", f"{DATA}/rerun_exp09_o19": "PASS", f"{DATA}/rerun_exp10_o19": "PASS"}

    def write(self, rel: str, text: str | bytes) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text if isinstance(text, bytes) else text.encode())

    def commit(self, msg: str, date: str) -> str:
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "-m", msg, date=date)
        return _git(self.root, "rev-parse", "HEAD")

    def base(self, text: str) -> str:
        self.write(LEDGER, text)
        self.base_sha = self.commit("base", BASE_DATE)
        return self.base_sha

    def head(self, text: str) -> str:
        self.write(LEDGER, text)
        return self.commit("head", HEAD_DATE)

    def run(self) -> tuple[list[str], list[str]]:
        failures, notes, _ = G.gate(self.root, base=self.base_sha, prereg=self.prereg)
        return failures, notes


# ── an O19 attempt as the harness writes it, stamped as the verdict writer would ─────────────────────────


def o19_attempt(rig: Rig, exp: str, executed: str, *, monkeypatch, rows_edit=None, side_branch: bool = False) -> dict:
    """Write a complete O19 attempt for ``exp`` (run on ``executed``), land its rows and session files on main's
    first-parent history (``rig.base_sha`` moves to the landing), and write (uncommitted) the verdict the writer
    stamps AT that landing commit: ``verdict_commit`` = its executed commit = the landing, each bound blob read there.
    ``rows_edit(rig)`` changes the data before it lands; ``side_branch`` lands it on a branch merged ``--no-ff``."""
    import o19_rerun as h
    import o19_verdict as v
    from _provenance import stamp_verdict

    data_dir = rig.root / v.data_dir(exp)
    rows_path = data_dir / "rows.jsonl"
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows_path)
    assert h.main(["run", "--exp", exp, "--mock"]) == 0
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    for r in rows:  # as a real attempt on the rig: not mock, run on a commit on main, its sims stamped likewise
        r["mock"] = False
        r["provenance"].update(executed_git_hash=executed, working_tree_dirty_src_scripts=False)
        for sim in r.get("sims") or []:
            sim.update(executed_git_hash=executed, working_tree_dirty_src_scripts=False, ts=sim.get("ts") or r["ts"])
    rows_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    if rows_edit:
        rows_edit(rig)
    if side_branch:
        _git(rig.root, "checkout", "-q", "-b", "side")
    landed = rig.commit(f"o19 {exp}: the attempt's rows land", BASE_DATE)
    rig.base_sha = landed
    if side_branch:
        _git(rig.root, "checkout", "-q", "main")
        _git(rig.root, "merge", "-q", "--no-ff", "-m", "merge the attempt", "side")
        rig.base_sha = _git(rig.root, "rev-parse", "HEAD")
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    attempts = v.attempts_from_rows(rows)
    rid = next(iter(attempts))
    out = v.judge(exp, [{"run_id": rid, "k": 1, "rows": attempts[rid]}], data_dir)
    bound = {
        p: _git(rig.root, "rev-parse", f"{landed}:{p}") for p in (R.O19_JUDGE, R.O19_RERUN, v.PROTOCOL[exp]["prereg"])
    }
    marker = {"run_id": rid, "k": 1, "ref": f"{v.MARKER_NAMESPACE}/{exp}/attempt-1-{rid}", "peeled": executed}
    out.update(
        apparatus_checked=True,
        apparatus={"markers": [marker]},
        bound_files=bound,
        verdict_commit=landed,
        verdict_source_sha256=hashlib.sha256(_git_bytes(rig.root, f"{landed}:{R.O19_JUDGE}")).hexdigest(),
        provenance={
            "executed_git_hash": landed,
            "code_tree_sha256": "t" * 64,
            "working_tree_dirty_src_scripts": False,
        },
    )
    data_bytes = rows_path.read_bytes()
    stamp_verdict(out, repo_root=rig.root, kind=v.PROTOCOL[exp]["kind"], data=rows_path, data_bytes=data_bytes,
                  scope={"all_rows": True}, mock=False)  # fmt: skip
    out = json.loads(json.dumps(out))  # as the writer's JSON file holds it
    (data_dir / "verdict.json").write_text(json.dumps(out, indent=1))
    return out


def _git_bytes(root: Path, spec: str) -> bytes:
    return subprocess.run(["git", "cat-file", "blob", spec], cwd=root, capture_output=True, check=True).stdout


@pytest.fixture
def rig(tmp_path) -> Rig:
    return Rig(tmp_path / "repo")


def _t39_move(
    rig: Rig,
    monkeypatch,
    *,
    verdict_edit=None,
    rows_edit=None,
    head_edit=None,
    ledger_token="PARTIAL",
    row="T3-9",
    side_branch=False,
):
    """BASE: T3-9 STALE, then an O19 Exp 09 attempt's rows landed (``rows_edit`` before they land). HEAD: its
    verdict (``verdict_edit``), any ``head_edit`` to the tree, and ``row`` moved to ``ledger_token`` citing it."""
    rig.base(ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]))
    record = o19_attempt(rig, "09", rig.base_sha, monkeypatch=monkeypatch, rows_edit=rows_edit, side_branch=side_branch)
    if head_edit:
        head_edit(rig)
    if verdict_edit:
        verdict_edit(record)
        (rig.root / DATA / "rerun_exp09_o19" / "verdict.json").write_text(json.dumps(record, indent=1))
    cite = f"**Evidence:** `{DATA}/rerun_exp09_o19/verdict.json`."
    status = f"**Status: {ledger_token} 2026-10-02**. {cite}"
    t1_row = t1("T1-1", status if row == "T1-1" else "**Status: STALE 2026-09-30**.")
    t3_row = t3("T3-9", status if row == "T3-9" else "**Status: STALE 2026-09-30**.")
    rig.head(ledger([t1_row], [t3_row]))
    return record


# ── the O19 path: T3-9 STALE -> PARTIAL on an Exp 09 PARTIAL verdict ─────────────────────────────────────


def test_an_o19_partial_verdict_moves_t3_9_to_partial(rig, monkeypatch) -> None:
    record = _t39_move(rig, monkeypatch)
    assert record["verdict"] == "PARTIAL"  # Exp 09's pre-registered ceiling (H3 not measured)
    failures, _ = rig.run()
    assert failures == []


def test_a_partial_verdict_does_not_support_maintained(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, ledger_token="MAINTAINED")
    failures, _ = rig.run()
    assert any("does not support MAINTAINED" in f for f in failures), failures


def test_a_verdict_supports_only_its_own_rows(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, row="T1-1")
    failures, _ = rig.run()
    assert any("may not support T1-1" in f for f in failures), failures


def test_a_pass_table_added_in_the_same_pr_supports_nothing(rig, monkeypatch) -> None:
    rig.write(G.PASS_TABLE, "{}\n")  # empty on main
    _t39_move(rig, monkeypatch)
    rig.write(G.PASS_TABLE, (REPO / G.PASS_TABLE).read_text())
    rig.commit("table in the same PR", HEAD_DATE)
    failures, _ = rig.run()
    assert any("not in the merge-base pass table" in f for f in failures), failures


def test_the_apparatus_must_have_been_checked(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, verdict_edit=lambda r: r.update(apparatus_checked=False))
    failures, _ = rig.run()
    # The record itself is NOT-ESTABLISHED (the pass table's `require` would refuse it too, as a second layer).
    assert any("apparatus (markers, ruleset, landing) was not checked" in f for f in failures), failures


def test_a_tampered_session_file_is_refused(rig, monkeypatch) -> None:
    def tamper(rig):
        sessions = [p for p in (rig.root / DATA / "rerun_exp09_o19").iterdir() if p.is_dir()]
        report = sessions[0] / "report.json"
        report.write_text(report.read_text() + " ")

    _t39_move(rig, monkeypatch, head_edit=tamper)  # in the PR, after the verdict was written
    failures, _ = rig.run()
    assert any("differs from its row's SHA-256" in f for f in failures), failures


def test_a_session_file_present_plain_and_gz_is_refused(rig, monkeypatch) -> None:
    def both(rig):
        sessions = [p for p in (rig.root / DATA / "rerun_exp09_o19").iterdir() if p.is_dir()]
        gz = sessions[0] / "run_log.jsonl.gz"
        import gzip

        (sessions[0] / "run_log.jsonl").write_bytes(gzip.decompress(gz.read_bytes()))

    _t39_move(rig, monkeypatch, head_edit=both)
    failures, _ = rig.run()
    assert any("present both plain and .gz" in f for f in failures), failures


def test_the_judge_must_be_the_one_that_wrote_the_verdict(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, verdict_edit=lambda r: r.update(verdict_source_sha256="0" * 64))
    failures, _ = rig.run()
    assert any("verdict_source_sha256 is not the bound judge's" in f for f in failures), failures


def test_a_verdict_the_judge_does_not_reproduce_is_refused(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, verdict_edit=lambda r: r.update(verdict="PASS"))
    failures, _ = rig.run()
    assert any("bound judge gives a different verdict" in f for f in failures), failures


def test_rows_without_a_marker_are_refused(rig, monkeypatch) -> None:
    marker = {"run_id": "f" * 32, "k": 1, "ref": f"refs/tags/o19/09/attempt-1-{'f' * 32}"}  # well formed, not the rows'

    def swap(r):
        r["apparatus"] = {"markers": [{**marker, "peeled": r["apparatus"]["markers"][0]["peeled"]}]}

    _t39_move(rig, monkeypatch, verdict_edit=swap)
    failures, _ = rig.run()
    assert any("no start marker" in f for f in failures), failures


def test_mock_rows_sink_the_verdict(rig, monkeypatch) -> None:
    def mock(rig):
        p = rig.root / DATA / "rerun_exp09_o19" / "rows.jsonl"
        rows = [json.loads(ln) for ln in p.read_text().splitlines()]
        rows[0]["mock"] = True
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    _t39_move(rig, monkeypatch, rows_edit=mock)  # mock rows landed and were judged: data_sha256 is honest
    failures, _ = rig.run()
    assert any("verdict over mock" in f for f in failures), failures


def test_new_support_must_be_pre_registered(rig, monkeypatch) -> None:
    rig.prereg = {f"{DATA}/rerun_exp09_o19": "OUT_OF_SCOPE"}
    _t39_move(rig, monkeypatch)
    failures, _ = rig.run()
    assert any("prereg status OUT_OF_SCOPE" in f for f in failures), failures


# ── legacy evidence, exceptions, the snapshot ────────────────────────────────────────────────────────────


def _legacy_snapshot(rig: Rig) -> None:
    data = (rig.root / DATA / "legacy_old.jsonl").read_bytes()
    rig.write(G.LEGACY_SNAPSHOT, json.dumps({f"{DATA}/legacy_old.jsonl": hashlib.sha256(data).hexdigest()}))


def test_a_date_advance_on_legacy_evidence_needs_new_support(rig, monkeypatch) -> None:
    monkeypatch.setattr(G, "M1A_CUTOFF", 4_000_000_000)  # the rig's files count as pre-M1a
    _legacy_snapshot(rig)
    cite = f"**Evidence:** `{DATA}/legacy_old.jsonl`."
    rig.base(
        ledger([t1("T1-13", f"**Status: EARNED 2026-09-16**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    )
    rig.head(
        ledger([t1("T1-13", f"**Status: EARNED 2026-10-02**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    )
    failures, notes = rig.run()
    assert any("no NEW support" in f for f in failures), failures
    assert any("LEGACY" in n for n in notes)


def test_an_untouched_positive_row_is_not_judged(rig, monkeypatch) -> None:
    monkeypatch.setattr(G, "M1A_CUTOFF", 4_000_000_000)
    _legacy_snapshot(rig)
    cite = f"**Evidence:** `{DATA}/legacy_old.jsonl`."
    text = ledger(
        [t1("T1-13", f"**Status: EARNED 2026-09-16**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]
    )
    rig.base(text)
    rig.head(text.replace("# Ledger", "# The ledger"))
    assert rig.run()[0] == []


def test_an_active_exception_may_supply_the_support(rig, monkeypatch) -> None:
    monkeypatch.setattr(G, "M1A_CUTOFF", 4_000_000_000)
    _legacy_snapshot(rig)
    cite = f"**Evidence:** `{DATA}/legacy_old.jsonl`."
    digest = hashlib.sha256((rig.root / DATA / "legacy_old.jsonl").read_bytes()).hexdigest()
    rig.write(G.EXCEPTIONS, json.dumps([{
        "id": "x1", "kind": "ledger", "row": "T1-13", "from": "EARNED", "to": "EARNED", "to_date": "2026-10-02",
        "path": f"{DATA}/legacy_old.jsonl", "sha256": digest, "owner": "owner", "reason": "r", "date": "2026-10-01",
    }]))  # fmt: skip
    rig.base(
        ledger([t1("T1-13", f"**Status: EARNED 2026-09-16**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    )
    rig.head(
        ledger([t1("T1-13", f"**Status: EARNED 2026-10-02**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    )
    assert rig.run()[0] == []


def test_the_exceptions_file_is_append_only(rig) -> None:
    entry = PREREG_CLAUSE
    rig.write(G.EXCEPTIONS, json.dumps([entry]))
    text = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    rig.write(G.EXCEPTIONS, "[]")
    rig.head(text)
    assert any("append-only" in f for f in rig.run()[0])


def test_the_legacy_snapshot_only_shrinks_and_must_stay_true(rig, monkeypatch) -> None:
    monkeypatch.setattr(G, "M1A_CUTOFF", 4_000_000_000)
    text = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    _legacy_snapshot(rig)  # a key added after main
    rig.head(text)
    assert any("only shrinks" in f for f in rig.run()[0])


def test_a_changed_legacy_record_is_stale(rig, monkeypatch) -> None:
    monkeypatch.setattr(G, "M1A_CUTOFF", 4_000_000_000)
    _legacy_snapshot(rig)
    text = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    rig.write(f"{DATA}/legacy_old.jsonl", '{"x": 2}\n')
    rig.head(text)
    assert any("changed since it was snapshotted" in f for f in rig.run()[0])


# ── units ────────────────────────────────────────────────────────────────────────────────────────────────


def test_require_is_dotted_strict_and_missing_is_unmet() -> None:
    assert G.require_met({"noop_kit": {"kit_pass": True}}, {"noop_kit.kit_pass": True})
    assert not G.require_met({"noop_kit": {"kit_pass": 1}}, {"noop_kit.kit_pass": True})  # 1 == True, but not the type
    assert not G.require_met({"noop_kit": None}, {"noop_kit.kit_pass": True})
    assert not G.require_met({}, {"apparatus_checked": True})


def test_finish_reasons_never_admit_a_failure() -> None:
    from maxim.simulation.sim_types import SIMULATION_FAILURE_FINISH_REASONS

    assert not (R.FINISH_OK & SIMULATION_FAILURE_FINISH_REASONS)
    assert all(not R.finish_ok(r) for r in SIMULATION_FAILURE_FINISH_REASONS)


def test_the_real_pass_table_names_the_o19_rows() -> None:
    table = json.loads((REPO / G.PASS_TABLE).read_text())
    assert table["exp10_verdict"]["rows"] == ["T1-1"] and table["exp09_verdict"]["rows"] == ["T3-9"]
    assert "MAINTAINED" not in table["exp09_verdict"]["targets"] or table["exp09_verdict"]["targets"]["MAINTAINED"] == [
        "PASS"
    ]
    assert table["exp09_verdict"]["targets"]["PARTIAL"] == ["PARTIAL"]
    assert not any(k.startswith(("exp54", "gate6")) for k in table)  # no ledger row / never support


def test_the_real_pass_table_supports_exp63_earned_from_pass_only() -> None:
    """Exp 63's prereg: ``exp63_verdict`` -> {PASS} supports EARNED on the new row T1-16 (NOT SHOWN supports nothing),
    and only from an apparatus-checked verdict; it is an O19 kind, so its bound judge is re-run. REPRODUCED (#1059)
    is for a successor campaign's PASS only: Exp 63 has none, and the gate refuses it from its root campaign."""
    table = json.loads((REPO / G.PASS_TABLE).read_text())
    assert table["exp63_verdict"] == {
        "rows": ["T1-16"],
        "targets": {"EARNED": ["PASS"], "REPRODUCED": ["PASS"]},
        "require": {"apparatus_checked": True},
    }
    assert G.pass_table_problems(table, "HEAD") == [] and "exp63_verdict" in R.O19_KINDS
    import o19_verdict as v

    assert {p["kind"] for p in v.PROTOCOL.values()} <= R.O19_KINDS  # every campaign's verdicts are re-judged


def test_unknown_digests_never_match() -> None:
    assert R.unknown("unknown") and R.unknown("unknown: OSError") and R.unknown(None) and R.unknown("")
    assert not R.unknown("a" * 64)


# ── more rules: time, code on main, sims, the removal ratchet, triggers, record kinds ────────────────────


def _edit_rows(rig: Rig, exp: str, fn) -> None:
    import o19_verdict as v

    p = rig.root / v.data_dir(exp) / "rows.jsonl"
    rows = [json.loads(ln) for ln in p.read_text().splitlines()]
    for r in rows:
        fn(r)
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def _restamp(rig: Rig, exp: str):
    import o19_verdict as v

    def edit(record):
        record["data_sha256"] = hashlib.sha256((rig.root / v.data_dir(exp) / "rows.jsonl").read_bytes()).hexdigest()

    return edit


def test_runs_older_than_the_previous_status_are_not_new_support(rig, monkeypatch) -> None:
    # Inside the clock-skew window of the commit it ran on, but not after the base set the row's STALE status.
    ts = rig_epoch(BASE_DATE) - 100
    _t39_move(rig, monkeypatch, rows_edit=lambda rg: _edit_rows(rg, "09", lambda r: r.update(ts=ts)))
    failures, _ = rig.run()
    assert any("not after the previous status was set" in f for f in failures), failures


def test_code_not_on_main_is_refused(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, rows_edit=lambda rg: _edit_rows(rg, "09", lambda r: r["provenance"].update(
        executed_git_hash="f" * 40)))  # fmt: skip
    failures, _ = rig.run()
    assert any("is not on main" in f for f in failures), failures


def test_an_o19_sim_whose_code_changed_is_refused(rig, monkeypatch) -> None:
    def changed(r):
        for sim in r.get("sims") or []:
            sim["code_changed_during_run"] = True

    _t39_move(rig, monkeypatch, rows_edit=lambda rg: _edit_rows(rg, "09", changed))
    failures, _ = rig.run()
    assert any("code_changed_during_run" in f for f in failures), failures


def test_removing_established_evidence_keeps_one(rig, monkeypatch) -> None:
    """BASE: T1-1 MAINTAINED on an Exp 10 O19 verdict (ESTABLISHED). HEAD: the verdict dropped for a legacy file."""
    monkeypatch.setattr(G, "M1A_CUTOFF", 4_000_000_000)
    _legacy_snapshot(rig)
    _git(rig.root, "init", "-q")  # (already a repo; keeps the fixture shape explicit)
    rig.write(
        LEDGER, ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    )
    first = rig.commit("main before the run", BASE_DATE)
    o19_attempt(rig, "10", first, monkeypatch=monkeypatch)  # lands the rows; the base below commits the verdict
    cite_v = f"**Evidence:** `{DATA}/rerun_exp10_o19/verdict.json`."
    rig.base(
        ledger(
            [t1("T1-1", f"**Status: MAINTAINED 2026-10-01**. {cite_v}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]
        )
    )
    cite_l = f"**Evidence:** `{DATA}/legacy_old.jsonl`."
    rig.head(
        ledger(
            [t1("T1-1", f"**Status: MAINTAINED 2026-10-01**. {cite_l}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]
        )
    )
    failures, _ = rig.run()
    assert any("keeps none" in f for f in failures), failures


def test_a_rewritten_qualifier_triggers_and_an_added_one_does_not() -> None:
    import _ledger as L

    def row(q):
        r = L.Row(id="T1-1", table="T1", line=1, cells={})
        r.token, r.date, r.qualifier = "MAINTAINED", "2026-10-01", q
        return r

    assert "qualifier removed or rewritten" in G.triggers_for(row("broad"), row("narrow"), set(), [])
    assert G.triggers_for(row("narrow, one model"), row("narrow"), set(), []) == []


def test_uncommitted_gated_files_fail(rig) -> None:
    text = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    rig.head(text.replace("# Ledger", "# The ledger"))
    rig.write(f"{DATA}/new.jsonl", "{}\n")
    assert any("uncommitted" in f for f in rig.run()[0])


def rig_epoch(iso: str) -> int:
    import datetime as dt

    return int(dt.datetime.fromisoformat(iso).timestamp())


def _ctx(rig: Rig) -> G.Ctx:
    return G.Ctx(repo=G.Repo(rig.root), base=rig.base_sha, ref="HEAD", legacy={}, prereg={DATA: "PASS"}, table={})


def test_event_log_groups(rig) -> None:
    sys.path.insert(0, str(REPO / "scripts" / "orient_backbone"))
    from live_common import provenance_digest

    text = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    block = {"executed_git_hash": rig.base_sha, "code_tree_sha256": "a" * 64,
             "working_tree_dirty_src_scripts": False, "harness_family": "in_process"}  # fmt: skip
    d = provenance_digest(block)

    def line(kind, gid, **kw):
        return json.dumps({"record_kind": kind, "log_run_id": gid, "mock": False, "provenance_sha256": d, "ts": 2e9,
                           **kw})  # fmt: skip

    ok = [line("harness_event", "g1", provenance=block), line("harness_run_end", "g1", status="ok",
          provenance=block, end_code_tree_sha256="a" * 64)]  # fmt: skip
    failed = [line("harness_event", "g2", provenance=block), line("harness_run_end", "g2", status="failed",
              provenance=block, end_code_tree_sha256="a" * 64)]  # fmt: skip
    rig.write(f"{DATA}/ev_ok.jsonl", "\n".join(ok + failed) + "\n")
    rig.write(f"{DATA}/ev_none.jsonl", "\n".join(failed) + "\n")
    rig.write(f"{DATA}/ev_mock.jsonl", "\n".join(ok).replace('"mock": false', '"mock": true', 1) + "\n")
    rig.commit("logs", HEAD_DATE)
    ctx = _ctx(rig)
    assert G.judge_entry(f"{DATA}/ev_ok.jsonl", ctx).status == G.ESTABLISHED  # the failed group is excluded
    assert G.judge_entry(f"{DATA}/ev_none.jsonl", ctx).status == G.NOT_ESTABLISHED  # no group ended ok
    assert G.judge_entry(f"{DATA}/ev_mock.jsonl", ctx).status == G.NOT_ESTABLISHED  # a mock line sinks the file


def test_sim_reports_and_instrument_checks(rig) -> None:
    text = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    prov = {"executed_git_hash": rig.base_sha, "code_tree_sha256": "a" * 64, "end_code_tree_sha256": "a" * 64,
            "working_tree_dirty_src_scripts": False, "code_changed_during_run": False, "configured_n_ctx": 8192,
            "language_profile": "mistral-7b-instruct-v0.2", "language_router_n_ctx": 8192}  # fmt: skip
    report = {"record_kind": "sim_report", "finish_reason": "max_turns", "ts": 2e9, "provenance": prov}
    rig.write(f"{DATA}/s_ok/report.json", json.dumps(report))
    rig.write(f"{DATA}/s_abort/report.json", json.dumps({**report, "finish_reason": "planning_failed"}))
    check = {"record_kind": "instrument_check", "status": "ok", "pass": False, "mock": False, "provenance": prov}
    rig.write(f"{DATA}/check.json", json.dumps(check))
    rig.commit("records", HEAD_DATE)
    ctx = _ctx(rig)
    assert G.judge_entry(f"{DATA}/s_ok", ctx).status == G.ESTABLISHED
    abort = G.judge_entry(f"{DATA}/s_abort", ctx)
    assert abort.status == G.NOT_ESTABLISHED and any("planning_failed" in r for r in abort.reasons)
    assert G.judge_entry(f"{DATA}/check.json", ctx).status == G.NOT_ESTABLISHED  # did not pass


def test_a_mock_row_sinks_a_cited_rows_file(rig, monkeypatch) -> None:
    text = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    prov = {"executed_git_hash": rig.base_sha, "code_tree_sha256": "a" * 64, "working_tree_dirty_src_scripts": False,
            "harness_family": "in_process"}  # fmt: skip
    row = {"record_kind": "harness_row", "status": "ok", "mock": False, "ts": 2e9, "provenance": prov}
    rig.write(f"{DATA}/rows_ok.jsonl", json.dumps(row) + "\n")
    rig.write(f"{DATA}/rows_mock.jsonl", json.dumps(row) + "\n" + json.dumps({**row, "mock": True}) + "\n")
    rig.commit("rows", HEAD_DATE)
    ctx = _ctx(rig)
    assert G.judge_entry(f"{DATA}/rows_ok.jsonl", ctx).status == G.ESTABLISHED
    j = G.judge_entry(f"{DATA}/rows_mock.jsonl", ctx)
    assert j.status == G.NOT_ESTABLISHED and any("a line is mock" in r for r in j.reasons)


def test_a_one_line_rows_file_is_judged_by_the_rows_rules(rig) -> None:
    """A one-line rows file is valid single-document JSON too; it must still meet the rows-file rules."""
    text = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    prov = {"executed_git_hash": rig.base_sha, "code_tree_sha256": "a" * 64, "working_tree_dirty_src_scripts": False,
            "harness_family": "spawning"}  # fmt: skip
    row = {"record_kind": "harness_row", "status": "ok", "mock": False, "ts": 2e9, "provenance": prov}
    rig.write(f"{DATA}/one_row.jsonl", json.dumps(row) + "\n")
    rig.commit("one row", HEAD_DATE)
    j = G.judge_entry(f"{DATA}/one_row.jsonl", _ctx(rig))
    assert j.kind == "harness_row" and any("names no sims" in r for r in j.reasons), (j.kind, j.reasons)


# ── positive controls for each rule (each proven by deleting its mechanism) ──────────────────────────────

STALE_BOTH = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])


def _prov(rig: Rig, **kw) -> dict:
    return {"executed_git_hash": rig.base_sha, "code_tree_sha256": "a" * 64, "working_tree_dirty_src_scripts": False,
            "harness_family": "in_process", **kw}  # fmt: skip


def _row(rig: Rig, **kw) -> dict:
    return {"record_kind": "harness_row", "status": "ok", "mock": False, "ts": 2e9, "provenance": _prov(rig), **kw}


def _sim_report(rig: Rig, **prov) -> dict:
    p = {"executed_git_hash": rig.base_sha, "code_tree_sha256": "a" * 64, "end_code_tree_sha256": "a" * 64,
         "working_tree_dirty_src_scripts": False, "code_changed_during_run": False, "configured_n_ctx": 8192,
         "language_profile": "mistral-7b-instruct-v0.2", "language_router_n_ctx": 8192, **prov}  # fmt: skip
    return {"record_kind": "sim_report", "finish_reason": "max_turns", "ts": 2e9, "provenance": p}


def _judge(rig: Rig, path: str, **ctx_kw) -> R.Judgement:
    ctx = _ctx(rig)
    for k, v in ctx_kw.items():
        setattr(ctx, k, v)
    return G.judge_entry(path, ctx)


def test_status_set_time_is_when_the_base_first_set_the_status(rig) -> None:
    set_at = "2026-09-30T00:00:00+00:00"
    rig.write(LEDGER, STALE_BOTH)
    rig.commit("STALE set", set_at)
    rig.base(STALE_BOTH.replace("claim T1-1", "claim T1-1 reworded"))  # a later ledger commit, T3-9 untouched
    assert G.status_set_time(G.Repo(rig.root), rig.base_sha, "T3-9", "STALE", "2026-09-30") == rig_epoch(set_at)


def test_a_change_to_cited_data_triggers_its_row(rig) -> None:
    rig.base_sha = rig.commit("main before the run", BASE_DATE)
    rig.write(f"{DATA}/r.jsonl", json.dumps(_row(rig)) + "\n")
    cite = f"**Evidence:** `{DATA}/r.jsonl`."
    text = ledger(
        [t1("T1-13", f"**Status: MAINTAINED 2026-09-30**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]
    )
    rig.base(text)
    rig.write(f"{DATA}/r.jsonl", json.dumps(_row(rig, mock=True)) + "\n")  # the ledger is untouched
    rig.head(text)
    failures, _ = rig.run()
    assert any("T1-13" in f and "a line is mock" in f for f in failures), failures


def _exception_case(rig, monkeypatch, *, where: str, **override) -> list[str]:
    monkeypatch.setattr(G, "M1A_CUTOFF", 4_000_000_000)
    _legacy_snapshot(rig)
    cite = f"**Evidence:** `{DATA}/legacy_old.jsonl`."
    digest = hashlib.sha256((rig.root / DATA / "legacy_old.jsonl").read_bytes()).hexdigest()
    entry = {"id": "x1", "kind": "ledger", "row": "T1-13", "from": "EARNED", "to": "EARNED", "to_date": "2026-10-02",
             "path": f"{DATA}/legacy_old.jsonl", "sha256": digest, "owner": "owner", "reason": "r",
             "date": "2026-10-01", **override}  # fmt: skip
    if where == "base":
        rig.write(G.EXCEPTIONS, json.dumps([entry]))
    rig.base(
        ledger([t1("T1-13", f"**Status: EARNED 2026-09-16**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    )
    if where == "head":
        rig.write(G.EXCEPTIONS, json.dumps([entry]))
    rig.head(
        ledger([t1("T1-13", f"**Status: EARNED 2026-10-02**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    )
    return rig.run()[0]


def test_an_exception_added_in_the_same_change_does_not_act(rig, monkeypatch) -> None:
    failures = _exception_case(rig, monkeypatch, where="head")
    assert any("no NEW support" in f for f in failures), failures


def test_an_exception_pinned_to_other_bytes_does_not_act(rig, monkeypatch) -> None:
    failures = _exception_case(rig, monkeypatch, where="base", sha256="0" * 64)
    assert any("no NEW support" in f for f in failures), failures


def test_an_exception_for_another_transition_is_inert(rig, monkeypatch) -> None:
    failures = _exception_case(rig, monkeypatch, where="base", to_date="2026-10-03")
    assert any("no NEW support" in f for f in failures), failures


def test_an_exception_may_name_a_new_row_and_must_say_from() -> None:
    entry = {"id": "x", "kind": "ledger", "row": "T1-1", "from": None, "to": "EARNED", "to_date": "2026-10-02",
             "path": "p", "sha256": "s", "owner": "o", "reason": "r", "date": "2026-10-01"}  # fmt: skip
    assert G.exceptions_problems([], [entry]) == []
    del entry["from"]
    assert any("lacks a required field" in p for p in G.exceptions_problems([], [entry]))


def test_legacy_needs_a_pre_m1a_add_and_no_record_kind(rig) -> None:
    rig.write(f"{DATA}/early_kind.jsonl", '{"record_kind": "harness_row"}\n')
    rig.commit("before M1a", "2026-09-01T00:00:00+00:00")
    rig.write(f"{DATA}/late.jsonl", '{"x": 3}\n')
    rig.base(STALE_BOTH)
    repo = G.Repo(rig.root)
    assert set(G.generate_legacy(repo)) == {f"{DATA}/legacy_old.jsonl"}
    blob = {p: hashlib.sha256((rig.root / p).read_bytes()).hexdigest()
            for p in (f"{DATA}/legacy_old.jsonl", f"{DATA}/early_kind.jsonl", f"{DATA}/late.jsonl")}  # fmt: skip
    problems = G.legacy_problems(repo, blob, blob)
    assert any("early_kind.jsonl carries a record_kind" in p for p in problems), problems
    assert any("late.jsonl was first committed after M1a" in p for p in problems), problems
    assert not any("legacy_old" in p for p in problems), problems


def _event_lines(rig, *, digest=None, end_tree="a" * 64, ts=2e9, gid="g1", status="ok") -> list[str]:
    from _provenance import provenance_digest

    block = _prov(rig)
    d = digest or provenance_digest(block)

    def line(kind, **kw):
        return json.dumps({"record_kind": kind, "log_run_id": gid, "mock": False, "provenance_sha256": d, "ts": ts,
                           "provenance": block, **kw})  # fmt: skip

    return [line("harness_event"), line("harness_run_end", status=status, end_code_tree_sha256=end_tree)]


def test_an_event_log_binds_each_line_to_its_block_and_its_end_tree(rig) -> None:
    rig.base(STALE_BOTH)
    rig.write(f"{DATA}/ev_digest.jsonl", "\n".join(_event_lines(rig, digest="0" * 64)) + "\n")
    rig.write(f"{DATA}/ev_end.jsonl", "\n".join(_event_lines(rig, end_tree="b" * 64)) + "\n")
    rig.commit("logs", HEAD_DATE)
    j = _judge(rig, f"{DATA}/ev_digest.jsonl")
    assert j.status == G.NOT_ESTABLISHED and any("provenance_sha256 does not match" in r for r in j.reasons)
    j = _judge(rig, f"{DATA}/ev_end.jsonl")
    assert j.status == G.NOT_ESTABLISHED and any("ended on another code tree" in r for r in j.reasons)


def test_a_verdict_over_an_event_log_times_only_its_counted_groups(rig) -> None:
    rig.base(STALE_BOTH)
    lines = _event_lines(rig, ts=2e9) + _event_lines(rig, ts=1.95e9, gid="g2", status="failed")
    rig.write(f"{DATA}/ev.jsonl", "\n".join(lines) + "\n")
    data = (rig.root / DATA / "ev.jsonl").read_bytes()
    verdict = {"record_kind": "verdict", "kind": "exp57_verdict", "verdict": "PASS", "mock": False,
               "data": f"{DATA}/ev.jsonl", "data_sha256": hashlib.sha256(data).hexdigest(),
               "scope": {"all_rows": True}, "provenance": _prov(rig)}  # fmt: skip
    rig.write(f"{DATA}/v.json", json.dumps(verdict))
    rig.commit("verdict", HEAD_DATE)
    j = _judge(rig, f"{DATA}/v.json")
    assert j.status == G.ESTABLISHED, j.reasons
    assert j.time == 2e9  # the failed group's earlier events are not a unit


def test_one_code_tree_per_rows_file(rig) -> None:
    rig.base(STALE_BOTH)
    other = _row(rig, provenance=_prov(rig, code_tree_sha256="b" * 64))
    rig.write(f"{DATA}/two.jsonl", json.dumps(_row(rig)) + "\n" + json.dumps(other) + "\n")
    rig.commit("rows", HEAD_DATE)
    j = _judge(rig, f"{DATA}/two.jsonl")
    assert j.status == G.NOT_ESTABLISHED and any("2 code trees" in r for r in j.reasons), j.reasons


def test_a_verdicts_own_provenance_is_judged(rig, monkeypatch) -> None:
    def off_main(record):
        record["provenance"]["executed_git_hash"] = "f" * 40

    _t39_move(rig, monkeypatch, verdict_edit=off_main)
    failures, _ = rig.run()
    assert any("verdict provenance: executed" in f for f in failures), failures


def test_a_verdict_never_carries_an_allowance(rig, monkeypatch) -> None:
    def allowed(record):
        record["provenance"].update(working_tree_dirty_src_scripts=True, allow_dirty=True)

    _t39_move(rig, monkeypatch, verdict_edit=allowed)
    failures, _ = rig.run()
    assert any("verdict provenance: working_tree_dirty_src_scripts" in f for f in failures), failures


def test_a_verdict_is_bound_to_its_data_bytes(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, verdict_edit=lambda record: record.update(data_sha256="0" * 64))
    failures, _ = rig.run()
    assert any("differs from data_sha256" in f for f in failures), failures


def test_verdict_data_lives_under_the_data_root(rig) -> None:
    rig.base(STALE_BOTH)
    verdict = {"record_kind": "verdict", "kind": "exp57_verdict", "verdict": "PASS", "mock": False,
               "data": R.O19_JUDGE, "data_sha256": "0" * 64, "scope": {"all_rows": True},
               "provenance": _prov(rig)}  # fmt: skip
    rig.write(f"{DATA}/v.json", json.dumps(verdict))
    rig.commit("verdict", HEAD_DATE)
    j = _judge(rig, f"{DATA}/v.json")
    assert j.status == G.NOT_ESTABLISHED and any("is not a path under" in r for r in j.reasons), j.reasons


def test_new_supports_data_must_be_pre_registered_too(rig, monkeypatch) -> None:
    rig.prereg = {f"{DATA}/rerun_exp09_o19": "PASS", f"{DATA}/rerun_exp09_o19/rows.jsonl": "OUT_OF_SCOPE"}
    _t39_move(rig, monkeypatch)
    failures, _ = rig.run()
    assert any("its data's OUT_OF_SCOPE" in f for f in failures), failures


def test_a_symlink_is_not_evidence(rig) -> None:
    rig.base(STALE_BOTH)
    os.symlink("legacy_old.jsonl", rig.root / DATA / "link.jsonl")
    rig.commit("link", HEAD_DATE)
    j = _judge(rig, f"{DATA}/link.jsonl")
    assert j.status == G.NOT_ESTABLISHED and any("symlink" in r for r in j.reasons), j.reasons


def test_an_allowance_establishes_but_never_supports(rig) -> None:
    rig.base(STALE_BOTH)
    granted = _row(rig, provenance=_prov(rig, working_tree_dirty_src_scripts=True, allow_dirty=True))
    refused = _row(rig, provenance=_prov(rig, working_tree_dirty_src_scripts=True))
    rig.write(f"{DATA}/granted.jsonl", json.dumps(granted) + "\n")
    rig.write(f"{DATA}/refused.jsonl", json.dumps(refused) + "\n")
    rig.commit("rows", HEAD_DATE)
    j = _judge(rig, f"{DATA}/granted.jsonl")
    assert j.status == G.ESTABLISHED and j.allowed_dirty
    assert _judge(rig, f"{DATA}/refused.jsonl").status == G.NOT_ESTABLISHED
    candidate = R.Judgement(path="v", status=G.ESTABLISHED, kind="verdict", time=3e9, allowed_dirty=True,
                            prereg="PASS", data_prereg="PASS",
                            record={"kind": "exp57_verdict", "verdict": "PASS"})  # fmt: skip
    table = {"exp57_verdict": {"rows": ["T1-12"], "targets": {"EARNED": ["PASS"]}}}
    assert "allowed-dirty" in (G.support_problem(candidate, "T1-12", "EARNED", table, 1.0, ctx=_ctx(rig)) or "")


def test_runs_cannot_predate_the_commit_they_ran_on(rig) -> None:
    rig.base(STALE_BOTH)
    rig.write(f"{DATA}/old.jsonl", json.dumps(_row(rig, ts=1.0)) + "\n")
    rig.commit("rows", HEAD_DATE)
    j = _judge(rig, f"{DATA}/old.jsonl")
    assert j.status == G.NOT_ESTABLISHED and any("before the commit it claims" in r for r in j.reasons), j.reasons


def test_triggers_for_a_new_row_and_a_claim_change() -> None:
    import _ledger as L

    def row(claim):
        r = L.Row(id="T1-1", table="T1", line=1, cells={"Claim": claim})
        r.token, r.date = "MAINTAINED", "2026-10-01"
        return r

    assert G.triggers_for(row("a"), None, set(), []) == ["new row"]
    assert "claim changed" in G.triggers_for(row("b"), row("a"), set(), [])
    assert "d/ changed" in G.triggers_for(row("a"), row("a"), {"d/x.jsonl"}, ["d/"])


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        ({"resume": {"resume_loaded": False}}, "resume_loaded is not true"),
        ({"configured_n_ctx": None}, "no configured_n_ctx"),
        ({"language_profile": None}, "no language_profile"),
        ({"language_router_n_ctx": None}, "language ran without a stamped language_router_n_ctx"),
        ({"aut_profile": "qwen", "aut_router_n_ctx": None}, "aut ran without a stamped aut_router_n_ctx"),
    ],
)
def test_a_sim_report_must_stamp_its_model_context_and_resume(rig, edit, why) -> None:
    rig.base(STALE_BOTH)
    rig.write(f"{DATA}/s/report.json", json.dumps(_sim_report(rig, **edit)))
    rig.commit("report", HEAD_DATE)
    j = _judge(rig, f"{DATA}/s")
    assert j.status == G.NOT_ESTABLISHED and any(why in r for r in j.reasons), j.reasons


def test_a_sim_report_needs_a_ts(rig) -> None:
    rig.base(STALE_BOTH)
    report = _sim_report(rig)
    del report["ts"]
    rig.write(f"{DATA}/s/report.json", json.dumps(report))
    rig.commit("report", HEAD_DATE)
    j = _judge(rig, f"{DATA}/s")
    assert j.status == G.NOT_ESTABLISHED and any("no ts" in r for r in j.reasons), j.reasons


def test_a_row_names_its_family_and_only_a_spawner_carries_sims(rig) -> None:
    rig.base(STALE_BOTH)
    sims = _row(rig, sims=[{"session_id": "s"}])
    nameless = _row(rig, provenance={k: v for k, v in _prov(rig).items() if k != "harness_family"})
    rig.write(f"{DATA}/sims.jsonl", json.dumps(sims) + "\n")
    rig.write(f"{DATA}/nameless.jsonl", json.dumps(nameless) + "\n")
    rig.commit("rows", HEAD_DATE)
    j = _judge(rig, f"{DATA}/sims.jsonl")
    assert j.status == G.NOT_ESTABLISHED and any("in-process row carries sims" in r for r in j.reasons), j.reasons
    j = _judge(rig, f"{DATA}/nameless.jsonl")
    assert j.status == G.NOT_ESTABLISHED and any("neither spawning nor in_process" in r for r in j.reasons)


def test_partial_rows_are_judged_on_every_change(rig) -> None:
    rig.base_sha = rig.commit("main before the run", BASE_DATE)
    rig.write(f"{DATA}/r.jsonl", json.dumps(_row(rig, mock=True)) + "\n")
    cite = f"**Evidence:** `{DATA}/r.jsonl`."
    text = ledger(
        [t1("T1-13", f"**Status: PARTIAL 2026-09-30**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]
    )
    rig.base(text)
    rig.head(text.replace("claim T1-13", "claim T1-13 widened"))
    failures, _ = rig.run()
    assert any("T1-13" in f and "a line is mock" in f for f in failures), failures


def test_leaving_stale_for_a_non_positive_token_is_noted(rig) -> None:
    rig.base(STALE_BOTH)
    rig.head(STALE_BOTH.replace("**Status: STALE 2026-09-30**", "**Status: DROPPED 2026-10-02**", 1))
    failures, notes = rig.run()
    assert failures == [] and any("leaves STALE for DROPPED" in n for n in notes), (failures, notes)


def test_legacy_still_obeys_the_prereg(rig) -> None:
    rig.base(STALE_BOTH)
    path = f"{DATA}/legacy_old.jsonl"
    legacy = {path: hashlib.sha256((rig.root / path).read_bytes()).hexdigest()}
    assert _judge(rig, path, legacy=legacy).status == R.LEGACY
    j = _judge(rig, path, legacy=legacy, prereg={path: "FAIL"})
    assert j.status == G.NOT_ESTABLISHED and any("prereg status FAIL" in r for r in j.reasons)


def test_malformed_records_are_refused_not_raised(rig) -> None:
    rig.base(STALE_BOTH)
    rig.write(f"{DATA}/bad.jsonl.gz", b"not gzip")
    rig.write(f"{DATA}/sims_str.jsonl", json.dumps(_row(rig, provenance=_prov(rig, harness_family="spawning"),
                                                          sims="s")) + "\n")  # fmt: skip
    rig.write(f"{DATA}/rows_ok.jsonl", json.dumps(_row(rig)) + "\n")
    data = (rig.root / DATA / "rows_ok.jsonl").read_bytes()
    verdict = {"record_kind": "verdict", "kind": "exp57_verdict", "verdict": "PASS", "mock": False,
               "data": f"{DATA}/rows_ok.jsonl", "data_sha256": hashlib.sha256(data).hexdigest(),
               "scope": {"run_ids": "r1"}, "provenance": _prov(rig)}  # fmt: skip
    rig.write(f"{DATA}/v_scope.json", json.dumps(verdict))
    rig.write(f"{DATA}/s/report.json", json.dumps({**_sim_report(rig), "provenance": ["x"]}))
    rig.commit("malformed", HEAD_DATE)
    for path in ("bad.jsonl.gz", "sims_str.jsonl", "v_scope.json", "s"):
        assert _judge(rig, f"{DATA}/{path}").status == G.NOT_ESTABLISHED, path


def test_every_demo_line_is_judged(rig) -> None:
    rig.base(STALE_BOTH)
    demo = {"record_kind": "harness_demo", "mock": False, "provenance": _prov(rig)}
    rig.write(f"{DATA}/demo.jsonl", json.dumps(demo) + "\n" + json.dumps({**demo, "mock": True}) + "\n")
    rig.commit("demo", HEAD_DATE)
    assert _judge(rig, f"{DATA}/demo.jsonl").status == G.NOT_ESTABLISHED


def test_a_malformed_pass_table_fails_at_head(rig) -> None:
    rig.base(STALE_BOTH)
    rig.write(G.PASS_TABLE, json.dumps({"exp09_verdict": {"rows": "T3-9", "targets": {}}}))
    rig.head(STALE_BOTH)
    assert any("entry 'exp09_verdict' is malformed" in f for f in rig.run()[0])


def test_changed_sees_both_names_of_a_rename(rig) -> None:
    rig.base(STALE_BOTH)
    _git(rig.root, "mv", f"{DATA}/legacy_old.jsonl", f"{DATA}/renamed.jsonl")
    rig.commit("rename", HEAD_DATE)
    assert {f"{DATA}/legacy_old.jsonl", f"{DATA}/renamed.jsonl"} <= G.Repo(rig.root).changed(rig.base_sha)


def _assert_stdlib_only(text: str) -> None:
    import ast

    tree = ast.parse(text)
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    in_main = {id(n) for n in ast.walk(main)} if main else set()
    for node in ast.walk(tree):
        if id(node) in in_main:
            continue
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            top = name.split(".")[0]
            assert top == "__future__" or top in sys.stdlib_module_names, name


def test_the_o19_judge_imports_only_the_standard_library() -> None:
    """The gate executes the BOUND ``o19_verdict.judge`` (#1050): outside the CLI's ``main`` it imports nothing but
    the standard library, so re-judging cannot run repo code beyond the bound file (N3). Statically, and by loading
    the bytes through the gate's own loader: every module the load adds is standard library, and sys.path is kept."""
    source = (REPO / R.O19_JUDGE).read_bytes()
    _assert_stdlib_only(source.decode())
    before_mods, before_path = set(sys.modules), list(sys.path)
    mod = R.load_o19_judge(source)
    assert not isinstance(mod, str), mod
    assert sys.path == before_path
    added = {m.split(".")[0] for m in set(sys.modules) - before_mods} - {"_o19_bound"}
    assert added <= set(sys.stdlib_module_names), added


class _WorkingTreeJudge(R.Repo):
    """The real repository, with HEAD's ``o19_verdict.py`` read from the working tree (the edit under test)."""

    def blob(self, ref: str, path: str) -> bytes | None:
        if ref == "HEAD" and path == R.O19_JUDGE:
            return (self.root / path).read_bytes()
        return super().blob(ref, path)


def test_half_b_on_the_real_repo_every_o19_verdict_rejudges_the_same() -> None:
    """#1050 half B on the REAL evidence: the judge in this tree (Exp 63's key added) re-judges every O19 verdict the
    repository holds exactly as it was written. Exp 63's behaviour is scoped to its own campaign key."""
    repo = _WorkingTreeJudge(REPO)
    verdicts = [p for p in (REPO / DATA).glob("*/verdict.json") if '"kind": "exp' in p.read_text()]
    if repo.kind("HEAD", f"{DATA}/rerun_exp10_o19c2/verdict.json") != "blob":
        pytest.skip("this clone's HEAD holds no O19 verdict")
    assert verdicts
    ctx = R.Ctx(repo=repo, base="HEAD", ref="HEAD", legacy={}, prereg={}, table={})
    assert R.o19_judge_edit_problems(ctx) == []


@pytest.mark.parametrize("pin", ["file", "other_bytes"])
def test_a_settled_pinned_clause_excepts_a_failing_record(rig, pin) -> None:
    """BASE: T1-13 MAINTAINED on a mock rows file, with a clause on main for that transition. HEAD edits the claim:
    the row is judged, and only a clause pinned to the cited bytes turns the refusal into EXCEPTED."""
    rig.base_sha = rig.commit("main before the run", BASE_DATE)
    path = f"{DATA}/r.jsonl"
    rig.write(path, json.dumps(_row(rig, mock=True)) + "\n")
    digest = hashlib.sha256((rig.root / path).read_bytes()).hexdigest() if pin == "file" else "0" * 64
    rig.write(G.EXCEPTIONS, json.dumps([{
        "id": "x1", "kind": "ledger", "row": "T1-13", "from": "STALE", "to": "MAINTAINED", "to_date": "2026-09-30",
        "path": path, "sha256": digest, "owner": "owner", "reason": "r", "date": "2026-09-30",
    }]))  # fmt: skip
    cite = f"**Evidence:** `{path}`."
    text = ledger(
        [t1("T1-13", f"**Status: MAINTAINED 2026-09-30**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]
    )
    rig.base(text)
    rig.head(text.replace("claim T1-13", "claim T1-13 widened"))
    failures, notes = rig.run()
    if pin == "file":
        assert failures == [] and any("EXCEPTED" in n for n in notes), (failures, notes)
    else:
        assert any("a line is mock" in f for f in failures), failures
        assert any("does not pin the cited bytes" in n for n in notes), notes


def test_a_clause_naming_a_session_directory_never_pins_and_never_crashes(rig) -> None:
    rig.base_sha = rig.commit("main before the run", BASE_DATE)
    session = f"{DATA}/s"
    rig.write(f"{session}/report.json", json.dumps({**_sim_report(rig), "finish_reason": "planning_failed"}))
    rig.write(G.EXCEPTIONS, json.dumps([{
        "id": "x1", "kind": "ledger", "row": "T1-13", "from": "STALE", "to": "MAINTAINED", "to_date": "2026-09-30",
        "path": session, "sha256": hashlib.sha256(b"").hexdigest(), "owner": "o", "reason": "r", "date": "2026-09-30",
    }]))  # fmt: skip
    cite = f"**Evidence:** `{session}`."
    text = ledger(
        [t1("T1-13", f"**Status: MAINTAINED 2026-09-30**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]
    )
    rig.base(text)
    rig.head(text.replace("claim T1-13", "claim T1-13 widened"))
    failures, _ = rig.run()
    assert any("planning_failed" in f for f in failures), failures


def test_a_verdict_written_on_a_later_commit_than_its_runs_is_established(rig) -> None:
    ran_at = "2026-09-20T00:00:00+00:00"
    rig.write(LEDGER, STALE_BOTH)
    ran_on = rig.commit("the commit the runs ran on", ran_at)
    rig.base(STALE_BOTH.replace("claim T1-1", "claim T1-1 reworded"))  # main moves on 11 days
    row = _row(rig, ts=float(rig_epoch(ran_at) + 10), provenance={**_prov(rig), "executed_git_hash": ran_on})
    rig.write(f"{DATA}/r.jsonl", json.dumps(row) + "\n")
    data = (rig.root / DATA / "r.jsonl").read_bytes()
    verdict = {
        "record_kind": "verdict",
        "kind": "exp57_verdict",
        "verdict": "PASS",
        "mock": False,
        "data": f"{DATA}/r.jsonl",
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "scope": {"all_rows": True},
        "provenance": _prov(rig),
    }  # fmt: skip  (written on the later base)
    rig.write(f"{DATA}/v.json", json.dumps(verdict))
    rig.commit("verdict", HEAD_DATE)
    j = _judge(rig, f"{DATA}/v.json")
    assert j.status == G.ESTABLISHED, j.reasons


@pytest.mark.parametrize("cause", ["unknown_kinds", "bad_gzip", "bad_shape", "judge_defect"])
def test_the_removal_ratchet_and_a_base_record_that_cannot_be_judged(rig, monkeypatch, cause) -> None:
    """A corrupt base record was never ESTABLISHED: dropping it is free. A judge-code error on main's record keeps
    the ratchet on (fail closed), with a NOTE saying why."""
    monkeypatch.setattr(G, "M1A_CUTOFF", 4_000_000_000)
    _legacy_snapshot(rig)
    odd = {"bad_gzip": "odd.jsonl.gz", "bad_shape": "odd.json"}.get(cause, "odd.jsonl")
    body = {"bad_gzip": b"not gzip", "bad_shape": b'{"record_kind": ["verdict"]}'}.get(cause, b'{"x": 1}\n{"x": 2}\n')
    rig.write(f"{DATA}/{odd}", body)
    if cause == "judge_defect":
        monkeypatch.setattr(R, "json_lines", lambda data: {}["boom"])  # a KeyError inside the judges
    legacy = f"`{DATA}/legacy_old.jsonl`"
    both = f"**Evidence:** {legacy}, `{DATA}/{odd}`."
    rig.base(
        ledger([t1("T1-13", f"**Status: EARNED 2026-09-16**. {both}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    )
    rig.head(
        ledger([t1("T1-13", f"**Status: EARNED 2026-09-16**. **Evidence:** {legacy}.")],
               [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    )  # fmt: skip
    failures, notes = rig.run()
    if cause != "judge_defect":  # never established on main: dropping it is free
        assert failures == [], failures
    else:
        assert any("while a base record could not be judged" in f for f in failures), failures
        assert any("could not be judged" in n for n in notes), notes


def test_an_echoed_sim_cannot_predate_its_own_commit(rig) -> None:
    rig.base(STALE_BOTH)
    sim = {k: v for k, v in _sim_report(rig)["provenance"].items()}
    sim.update(finish_reason="max_turns", ts=1.0)
    row = _row(rig, provenance=_prov(rig, harness_family="spawning"), sims=[sim])  # the row's own ts is valid
    rig.write(f"{DATA}/spawn.jsonl", json.dumps(row) + "\n")
    rig.commit("rows", HEAD_DATE)
    j = _judge(rig, f"{DATA}/spawn.jsonl")
    assert any("row 1 sim 0: ran at ts 1" in r for r in j.reasons), j.reasons


def test_a_sim_report_cannot_predate_its_commit(rig) -> None:
    rig.base(STALE_BOTH)
    rig.write(f"{DATA}/s/report.json", json.dumps({**_sim_report(rig), "ts": 1.0}))
    rig.commit("report", HEAD_DATE)
    j = _judge(rig, f"{DATA}/s")
    assert any("sim_report: ran at ts 1" in r for r in j.reasons), j.reasons


def test_an_event_group_cannot_predate_its_commit(rig) -> None:
    rig.base(STALE_BOTH)
    rig.write(f"{DATA}/ev.jsonl", "\n".join(_event_lines(rig, ts=1.0)) + "\n")
    rig.commit("log", HEAD_DATE)
    j = _judge(rig, f"{DATA}/ev.jsonl")
    assert any("group g1: ran at ts 1" in r for r in j.reasons), j.reasons


def test_entering_by_tests_from_another_token_is_noted(rig) -> None:
    rig.write("tests/unit/test_x.py", "def test_x():\n    pass\n")
    text = ledger([t1("T1-2", "**Status: PARTIAL 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    rig.head(text.replace("**Status: PARTIAL 2026-09-30**.", "**Status: RE-VALIDATED-BY-TESTS 2026-10-02**. "
                          "**Evidence:** `tests/unit/test_x.py`."))  # fmt: skip
    _, notes = rig.run()
    assert any("T1-2: enters RE-VALIDATED-BY-TESTS from PARTIAL" in n for n in notes), notes
    assert any("T1-2: RE-VALIDATED-BY-TESTS: named, not checked" in n for n in notes), notes


def test_exception_ids_are_unique_strings() -> None:
    entry = {**PREREG_CLAUSE, "id": "x"}
    assert any("unique string id" in p for p in G.exceptions_problems([], [entry, dict(entry)]))
    assert any("unique string id" in p for p in G.exceptions_problems([], [{**entry, "id": ["x"]}]))
    assert G.exceptions_problems([], [entry]) == []


def test_an_evidence_less_judged_row_cannot_change(rig) -> None:
    text = ledger([t1("T1-2", "**Status: PARTIAL 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    rig.head(text.replace("claim T1-2", "claim T1-2 widened"))
    failures, _ = rig.run()
    assert any("T1-2" in f and "cites no ESTABLISHED" in f for f in failures), failures


def test_a_same_date_move_into_by_tests_is_noted_without_a_trigger(rig) -> None:
    rig.write("tests/unit/test_x.py", "def test_x():\n    pass\n")
    cite = "**Evidence:** `tests/unit/test_x.py`."
    text = ledger([t1("T1-2", f"**Status: EARNED 2026-09-30**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    rig.head(text.replace("EARNED 2026-09-30", "RE-VALIDATED-BY-TESTS 2026-09-30"))
    _, notes = rig.run()
    assert any("T1-2: enters RE-VALIDATED-BY-TESTS from EARNED" in n for n in notes), notes


def test_a_new_by_tests_row_is_noted(rig) -> None:
    rig.write("tests/unit/test_x.py", "def test_x():\n    pass\n")
    rig.base(STALE_BOTH)
    row = t1("T1-20", "**Status: RE-VALIDATED-BY-TESTS 2026-10-02**. **Evidence:** `tests/unit/test_x.py`.")
    rig.head(ledger([t1("T1-1", "**Status: STALE 2026-09-30**."), row], [t3("T3-9", "**Status: STALE 2026-09-30**.")]))
    _, notes = rig.run()
    assert any("T1-20: enters RE-VALIDATED-BY-TESTS as a new row" in n for n in notes), notes


def test_unhashable_record_and_clause_values_are_refused_not_raised(rig) -> None:
    """A verdict whose ``kind`` is a list, cited by a judged row, and a clause on main whose ``path`` is a list:
    plain refusals, never a traceback."""
    rig.base_sha = rig.commit("main before the run", BASE_DATE)
    rig.write(G.EXCEPTIONS, json.dumps([{
        "id": "x1", "kind": "ledger", "row": "T1-13", "from": "STALE", "to": "EARNED", "to_date": "2026-10-02",
        "path": [f"{DATA}/v.json"], "sha256": "0" * 64, "owner": "o", "reason": "r", "date": "2026-09-30",
    }]))  # fmt: skip
    rig.base(ledger([t1("T1-13", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]))
    rig.write(f"{DATA}/rows.jsonl", json.dumps(_row(rig)) + "\n")
    data = (rig.root / DATA / "rows.jsonl").read_bytes()
    verdict = {"record_kind": "verdict", "kind": ["exp10_verdict"], "verdict": "PASS", "mock": False,
               "data": f"{DATA}/rows.jsonl", "data_sha256": hashlib.sha256(data).hexdigest(),
               "scope": {"all_rows": True}, "provenance": _prov(rig)}  # fmt: skip
    rig.write(f"{DATA}/v.json", json.dumps(verdict))
    cite = f"**Evidence:** `{DATA}/v.json`."
    rig.head(
        ledger([t1("T1-13", f"**Status: EARNED 2026-10-02**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    )
    failures, _ = rig.run()
    assert any("T1-13: no NEW support" in f and "not in the merge-base pass table" in f for f in failures), failures


def test_a_verdict_over_an_event_log_with_a_list_run_id_is_a_plain_refusal(rig) -> None:
    rig.base(STALE_BOTH)
    lines = [json.loads(ln) for ln in _event_lines(rig)]
    lines[0]["log_run_id"] = ["g1"]
    rig.write(f"{DATA}/ev.jsonl", "".join(json.dumps(ln) + "\n" for ln in lines))
    data = (rig.root / DATA / "ev.jsonl").read_bytes()
    verdict = {"record_kind": "verdict", "kind": "exp57_verdict", "verdict": "PASS", "mock": False,
               "data": f"{DATA}/ev.jsonl", "data_sha256": hashlib.sha256(data).hexdigest(),
               "scope": {"all_rows": True}, "provenance": _prov(rig)}  # fmt: skip
    rig.write(f"{DATA}/v.json", json.dumps(verdict))
    rig.commit("verdict", HEAD_DATE)
    j = _judge(rig, f"{DATA}/v.json")
    assert j.status == G.NOT_ESTABLISHED and not R.unjudged(j), j.reasons


# ── M1b PR 5b-2: complete-run rules ──────────────────────────────────────────────────────────────────────

REAL_TABLE = json.loads((REPO / G.PASS_TABLE).read_text())


def _verdict_over(rig: Rig, name: str, lines: list[dict], kind: str, scope: dict, **extra) -> str:
    rig.write(f"{DATA}/{name}", "".join(json.dumps(ln) + "\n" for ln in lines))
    data = (rig.root / DATA / name).read_bytes()
    verdict = {"record_kind": "verdict", "kind": kind, "verdict": "EARNED", "mock": False, "data": f"{DATA}/{name}",
               "data_sha256": hashlib.sha256(data).hexdigest(), "scope": scope, "provenance": _prov(rig),
               **extra}  # fmt: skip
    rig.write(f"{DATA}/v_{name}.json", json.dumps(verdict))
    rig.commit(f"verdict over {name}", HEAD_DATE)
    return f"{DATA}/v_{name}.json"


def _complete(rig: Rig, path: str) -> R.Judgement:
    return _judge(rig, path, table=REAL_TABLE)


def _assert_refused(j: R.Judgement, why: str) -> None:
    assert j.status == G.NOT_ESTABLISHED and any(why in r for r in j.reasons), j.reasons


# Exp 60: one complete run per arm in the FILE, and the scope is exactly those runs.


def _run60(rig: Rig, rid: str, arm: str, seeds=range(11, 16), refused=()) -> list[dict]:
    return [_row(rig, run_id=rid, arm=arm, seed=s, refusal="r" if s in refused else None) for s in seeds]


def test_exp60_one_complete_run_per_arm_is_established(rig) -> None:
    rig.base(STALE_BOTH)
    lines = _run60(rig, "A", "fear") + _run60(rig, "B", "ablated")
    assert (
        _complete(rig, _verdict_over(rig, "e60.jsonl", lines, "exp60_verdict", {"run_ids": ["A", "B"]})).status
        == G.ESTABLISHED
    )
    assert (
        _complete(rig, _verdict_over(rig, "e60b.jsonl", lines, "exp60_verdict", {"all_rows": True})).status
        == G.ESTABLISHED
    )


def test_exp60_a_second_complete_run_of_an_arm_cannot_be_left_out(rig) -> None:
    """Re-run the fear arm into the same file after a refusal, then scope the clean run: refused (the committed
    Exp 60 data has exactly this shape)."""
    rig.base(STALE_BOTH)
    lines = _run60(rig, "OLD", "fear", refused=(13,)) + _run60(rig, "A", "fear") + _run60(rig, "B", "ablated")
    _assert_refused(_complete(rig, _verdict_over(rig, "e60.jsonl", lines, "exp60_verdict", {"run_ids": ["A", "B"]})),
                    "2 complete runs of arm fear")  # fmt: skip


def test_exp60_a_run_killed_mid_arm_is_excludable_but_not_scopable(rig) -> None:
    rig.base(STALE_BOTH)
    lines = _run60(rig, "K", "fear", seeds=(11, 12)) + _run60(rig, "A", "fear") + _run60(rig, "B", "ablated")
    assert (
        _complete(rig, _verdict_over(rig, "e60.jsonl", lines, "exp60_verdict", {"run_ids": ["A", "B"]})).status
        == G.ESTABLISHED
    )
    _assert_refused(_complete(rig, _verdict_over(rig, "e60b.jsonl", lines, "exp60_verdict", {"all_rows": True})),
                    "not exactly the file's complete runs")  # fmt: skip


@pytest.mark.parametrize(("bad", "why"), [(16, "outside the frozen sets"), (True, "outside the frozen sets")])
def test_exp60_a_seed_off_the_frozen_list_is_refused(rig, bad, why) -> None:
    rig.base(STALE_BOTH)
    lines = _run60(rig, "A", "fear") + _run60(rig, "B", "ablated")
    lines[0]["seed"] = bad
    _assert_refused(_complete(rig, _verdict_over(rig, "e60.jsonl", lines, "exp60_verdict", {"all_rows": True})), why)


def test_exp60_a_repeated_scope_run_id_is_refused(rig) -> None:
    rig.base(STALE_BOTH)
    lines = _run60(rig, "A", "fear") + _run60(rig, "B", "ablated")
    _assert_refused(_complete(rig, _verdict_over(rig, "e60.jsonl", lines, "exp60_verdict", {"run_ids": ["A", "B", "B"]})),
                    "repeats a run_id")  # fmt: skip


# Exp 61 / 62: one campaign per file, the keyed rows equal the frozen sets.


def _campaign61(rig: Rig, campaign="c1") -> list[dict]:
    sets = REAL_TABLE["exp61_verdict"]["complete"]["sets"]
    rows = [_row(rig, kind="receiver", campaign_id=campaign, arm=a, pair_seed=s, refusal=None)
            for a, seeds in sets.items() for s in seeds]  # fmt: skip
    return rows + [_row(rig, kind="anti_vacuity", campaign_id=campaign, arm="transferred", pair_seed=200, refusal=None)]


def _v61(rig, lines, name="e61.jsonl", scope=None, campaign="c1"):
    return _complete(rig, _verdict_over(rig, name, lines, "exp61_verdict", scope or {"campaign_id": "c1"},
                                        campaign_id=campaign))  # fmt: skip


def test_exp61_a_whole_campaign_is_established(rig) -> None:
    rig.base(STALE_BOTH)
    assert _v61(rig, _campaign61(rig)).status == G.ESTABLISHED
    assert _v61(rig, _campaign61(rig), name="e61b.jsonl", scope={"all_rows": True}).status == G.ESTABLISHED


def test_exp61_a_refusal_superseded_on_resume_is_established(rig) -> None:
    rig.base(STALE_BOTH)
    lines = _campaign61(rig)
    lines.insert(0, {**lines[0], "refusal": "bridge stale"})
    assert _v61(rig, lines).status == G.ESTABLISHED


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        (lambda ls: ls.pop(0), "frozen key(s) missing"),
        (lambda ls: ls.append({**ls[0], "pair_seed": 224}), "outside the frozen sets"),
        (lambda ls: ls.append({**ls[0]}), "two clean rows"),
        (lambda ls: ls.append({**ls[-1], "campaign_id": "c2"}), "2 campaigns"),
        (lambda ls: ls.append({**ls[-1], "kind": "mystery"}), "is not one of"),
        (lambda ls: ls.pop(), "no ['anti_vacuity'] row"),
    ],
)
def test_exp61_campaign_structure_refusals(rig, edit, why) -> None:
    rig.base(STALE_BOTH)
    lines = _campaign61(rig)
    edit(lines)
    _assert_refused(_v61(rig, lines), why)


def test_exp61_the_verdict_must_name_the_files_campaign(rig) -> None:
    rig.base(STALE_BOTH)
    _assert_refused(_v61(rig, _campaign61(rig), scope={"all_rows": True}, campaign="other"), "not the file's campaign")


def test_exp62_a_campaign_needs_its_replay_and_apparatus_rows(rig) -> None:
    rig.base(STALE_BOTH)
    sets = REAL_TABLE["exp62_verdict"]["complete"]["sets"]
    rows = [_row(rig, kind="row", campaign_id="c1", arm=a, seed=s, refusal=None) for a, ss in sets.items() for s in ss]
    extra = [_row(rig, kind=k, campaign_id="c1", arm="cross", seed=600, refusal=None) for k in ("replay", "apparatus")]
    ok = _verdict_over(rig, "e62.jsonl", rows + extra, "exp62_verdict", {"campaign_id": "c1"}, campaign_id="c1")
    assert _complete(rig, ok).status == G.ESTABLISHED
    bad = _verdict_over(rig, "e62b.jsonl", rows + extra[1:], "exp62_verdict", {"campaign_id": "c1"}, campaign_id="c1")
    _assert_refused(_complete(rig, bad), "no ['replay'] row")


# Exp 53: one gate-I phase-1 run and one complete primary run per file, pinned and alone in ok log groups.

MANIFEST53 = REAL_TABLE["exp53_verdict"]["complete"]["manifest"]
AGENTS53 = [  # (label, arm, seed, exploratory, nac, ec)
    ("taught_seed42", "taught", 42, False, "n1", "e1"),
    ("no_feed_seed42", "no_feed", 42, False, "n2", "e2"),
    ("taught_seed48", "taught", 48, True, "n3", "e3"),
]


def _manifest53(rig: Rig) -> None:
    agents = [{"label": lb, "arm": a, "seed": s, "exploratory": x, "nac_sha256": n, "ec_sha256": e}
              for lb, a, s, x, n, e in AGENTS53]  # fmt: skip
    rig.write(MANIFEST53, json.dumps({"experiment": "53_cross_context_readout", "agents": agents}))


def _group53(rig: Rig, rid: str, phase: int, ts: float, *, start=None, gate_i="PASS", terminal="ok",
             status="complete", relabel=None, dry_run=False) -> list[dict]:  # fmt: skip
    from _provenance import provenance_digest

    block, gid = _prov(rig), f"g-{rid}"
    d = provenance_digest(block)

    def ev(event, **kw):
        return {"record_kind": "harness_event", "log_run_id": gid, "mock": False, "provenance": block,
                "provenance_sha256": d, "ts": ts, "run_id": rid, "phase": phase, "dry_run": dry_run, "event": event,
                **kw}  # fmt: skip

    lines = [ev("start", **{**REAL_TABLE["exp53_verdict"]["complete"]["start"], **(start or {})})]
    cond = {} if phase == 1 else {"condition": "primary"}
    for lb, a, s, x, n, e in AGENTS53:
        arm = relabel if relabel and lb == "taught_seed42" else a
        lines.append(
            ev("agent_load", agent=lb, arm=arm, seed=s, exploratory_agent=x, nac_sha256=n, ec_sha256=e, **cond)
        )
        lines.append(ev("probe" if phase == 1 else "trial", agent=lb, arm=arm, exploratory_agent=x, **cond))
    if phase == 1:
        lines.append(ev("gate_I", verdict=gate_i))
    lines.append(ev("run_end", status=status))
    term = {"record_kind": "harness_run_end", "log_run_id": gid, "status": terminal, "mock": False, "provenance": block,
            "provenance_sha256": d, "end_code_tree_sha256": "a" * 64, "ts": ts + 1}  # fmt: skip
    return lines + [term]


def _v53(rig, lines, runs_used=None, name="e53.jsonl"):
    used = runs_used or {"phase1": "P1", "primary": "P2", "secondary": None}
    scope = {"run_ids": sorted(v for v in used.values() if v)}
    return _complete(rig, _verdict_over(rig, name, lines, "exp53_verdict", scope, gate="T", verdict="PASS",
                                        scoped_lines_stamped=True, runs_used=used))  # fmt: skip


def _base53(rig: Rig) -> None:
    _manifest53(rig)
    rig.base(STALE_BOTH)


def test_exp53_a_pinned_pair_of_runs_is_established(rig) -> None:
    _base53(rig)
    j = _v53(rig, _group53(rig, "P1", 1, 2e9) + _group53(rig, "P2", 2, 2e9 + 100))
    assert j.status == G.ESTABLISHED, j.reasons


@pytest.mark.parametrize(
    ("extra", "why"),
    [
        ("second_primary", "2 complete primary run(s)"),
        ("debug", "debug (--only) run"),
        ("phase1_fail_then_pass", "2 gate-I phase-1 run(s)"),
    ],
)
def test_exp53_the_file_cannot_hide_another_attempt(rig, extra, why) -> None:
    _base53(rig)
    lines = _group53(rig, "P1", 1, 2e9) + _group53(rig, "P2", 2, 2e9 + 100)
    if extra == "second_primary":
        lines += _group53(rig, "P3", 2, 2e9 + 200)
    elif extra == "debug":
        lines += _group53(rig, "D", 2, 2e9 + 200, start={"only": ["taught_seed42"]})
    else:
        lines = _group53(rig, "F", 1, 2e9 - 100, gate_i="FAIL", status="stopped") + lines
    _assert_refused(_v53(rig, lines), why)


@pytest.mark.parametrize(
    ("kw", "why"),
    [
        ({"start": {"deltas": {"turn_left": 0.55, "turn_right": -0.55}}}, "start.deltas"),
        ({"relabel": "no_feed"}, "unlike its merge-base manifest entry"),
        ({"terminal": "failed"}, "exactly one log group that ended ok"),
        ({"dry_run": True}, "dry-run line"),
    ],
)
def test_exp53_each_scoped_run_is_pinned(rig, kw, why) -> None:
    _base53(rig)
    _assert_refused(_v53(rig, _group53(rig, "P1", 1, 2e9) + _group53(rig, "P2", 2, 2e9 + 100, **kw)), why)


def test_exp53_runs_used_must_name_the_files_runs(rig) -> None:
    _base53(rig)
    lines = _group53(rig, "P1", 1, 2e9) + _group53(rig, "P2", 2, 2e9 + 100)
    _assert_refused(_v53(rig, lines, runs_used={"phase1": "P1", "primary": "P1", "secondary": None}),
                    "does not name the file's phase-1 and primary runs")  # fmt: skip


def test_a_ruled_kind_without_its_complete_block_is_refused(rig) -> None:
    rig.base(STALE_BOTH)
    lines = _run60(rig, "A", "fear") + _run60(rig, "B", "ablated")
    path = _verdict_over(rig, "e60.jsonl", lines, "exp60_verdict", {"all_rows": True})
    table = {**REAL_TABLE, "exp60_verdict": {k: v for k, v in REAL_TABLE["exp60_verdict"].items() if k != "complete"}}
    _assert_refused(_judge(rig, path, table=table), "carries no `seeds_per_run_arm` rule")


# The table: shape, and pinned to each harness's FROZEN.


@pytest.mark.parametrize(
    ("kind", "complete", "why"),
    [
        ("exp60_verdict", None, "needs a `complete` block"),
        ("exp57_verdict", {"rule": "seeds_per_run_arm", "sets": {"a": [1]}}, "no complete-run rule exists"),
        ("exp60_verdict", {"rule": "seeds_per_run_arm", "sets": {"fear": [True]}}, "distinct integers"),
        ("exp60_verdict", {"rule": "seeds_per_run_arm", "sets": {"fear": [1, 1]}}, "distinct integers"),
        ("exp53_verdict", {"rule": "exp53_runs", "start": {}, "manifest": "/etc/x"}, "under the data root"),
    ],
)
def test_the_pass_table_complete_block_shape(kind, complete, why) -> None:
    entry = {"rows": ["T1-1"], "targets": {"EARNED": ["EARNED"]}}
    if complete is not None:
        entry["complete"] = complete
    assert any(why in p for p in G.pass_table_problems({kind: entry}, "HEAD")), G.pass_table_problems(
        {kind: entry}, "HEAD"
    )


def test_the_real_pass_table_is_well_formed_and_pinned_to_each_harness() -> None:
    sys.path.insert(0, str(REPO / "scripts" / "survival_world"))
    from survival_world import exp60_run, exp61_run, exp62_run

    assert G.pass_table_problems(REAL_TABLE, "HEAD") == []
    assert set(R.COMPLETE_RULES) <= set(REAL_TABLE)
    sets = {k: REAL_TABLE[k]["complete"].get("sets") for k in R.COMPLETE_RULES}
    assert sets["exp60_verdict"] == {
        "fear": list(exp60_run.FROZEN["seeds"]),
        "ablated": list(exp60_run.FROZEN["seeds"]),
    }
    f61 = exp61_run.FROZEN
    want61: dict[str, list[int]] = {a: [] for a in f61["arms"]}
    for idx, seed in enumerate(f61["pair_seeds"]):
        arms, _ = exp61_run.pair_plan(idx, seed, n_full=f61["arms"]["transferred"], offset=f61["dangling_donor_offset"])
        for a in arms:
            want61[a].append(seed)
    assert sets["exp61_verdict"] == want61 and all(len(v) == f61["arms"][a] for a, v in want61.items())
    assert sets["exp62_verdict"] == {a: list(v) for a, v in exp62_run.FROZEN["seeds"].items()}
    pins = REAL_TABLE["exp53_verdict"]["complete"]
    manifest = json.loads((REPO / pins["manifest"]).read_text())
    assert pins["start"]["experiment"] == manifest["experiment"]
    assert pins["start"]["body_ref"] == manifest["frozen"]["body_ref"]
    assert pins["start"]["deltas"] == {
        "turn_left": 0.3,
        "turn_right": -0.3,
    }  # 53b's declared change (T1-10 rests on it)


# Prereg exceptions: fields and pins.


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        (lambda c: c.pop("sha256"), "exactly one pin"),
        (lambda c: c.update(tree="t"), "exactly one pin"),
        (lambda c: c.update(path="docs/experiments/data/a/b.jsonl"), "top-level data entry"),
        (lambda c: c.pop("owner"), "lacks a required field"),
        (lambda c: c.update(extra=1), "unknown fields"),
    ],
)
def test_prereg_exception_clause_shape(edit, why) -> None:
    clause = dict(PREREG_CLAUSE)
    edit(clause)
    assert any(why in p for p in G.exceptions_problems([], [clause])), G.exceptions_problems([], [clause])


def test_prereg_exception_pin_form_must_match_the_entry(rig) -> None:
    rig.base(STALE_BOTH)
    tree_pin = {**{k: v for k, v in PREREG_CLAUSE.items() if k != "sha256"}, "tree": "t"}
    assert any("pins a tree" in p for p in G.exceptions_problems([], [tree_pin], G.Repo(rig.root)))
    assert G.exceptions_problems([], [PREREG_CLAUSE], G.Repo(rig.root)) == []


@pytest.mark.parametrize("status", ["NON_GATED", "NOT_GOVERNED", "FAIL", "SOMETHING_NEW", None])
def test_the_prereg_status_is_an_allow_list(rig, status) -> None:
    rig.base(STALE_BOTH)
    rig.write(f"{DATA}/r.jsonl", json.dumps(_row(rig)) + "\n")
    rig.commit("rows", HEAD_DATE)
    prereg = {} if status is None else {f"{DATA}/r.jsonl": status}
    assert _judge(rig, f"{DATA}/r.jsonl", prereg=prereg).status == G.NOT_ESTABLISHED
    assert _judge(rig, f"{DATA}/r.jsonl", prereg={f"{DATA}/r.jsonl": "EXCEPTED"}).status == G.ESTABLISHED


def test_an_exp60_verdict_moves_its_row_end_to_end(rig) -> None:
    """Through gate(): the merge-base table reaches the record judges, so a complete Exp 60 verdict supports T1-13."""
    text = ledger([t1("T1-13", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    lines = _run60(rig, "A", "fear") + _run60(rig, "B", "ablated")
    path = _verdict_over(rig, "e60.jsonl", lines, "exp60_verdict", {"run_ids": ["A", "B"]})
    rig.head(
        text.replace("**Status: STALE 2026-09-30**.", f"**Status: MAINTAINED 2026-10-02**. **Evidence:** `{path}`.", 1)
    )
    failures, _ = rig.run()
    assert failures == [], failures


def test_exp53_a_gate_c_phase1_run_is_not_a_gate_i_attempt(rig) -> None:
    """A `--gate C` phase-1 run writes an `informative: true` gate_I event: it is not a gate-I attempt, so it does
    not count against the one-phase-1-run rule (it stays unscoped)."""
    _base53(rig)
    gate_c = _group53(rig, "C", 1, 2e9 - 100, start={"gate": "C"})
    for ln in gate_c:
        if ln.get("event") == "gate_I":
            ln["informative"] = True
    j = _v53(rig, gate_c + _group53(rig, "P1", 1, 2e9) + _group53(rig, "P2", 2, 2e9 + 100))
    assert j.status == G.ESTABLISHED, j.reasons


def test_a_complete_block_with_another_kinds_rule_is_refused(rig) -> None:
    rig.base(STALE_BOTH)
    lines = _run60(rig, "A", "fear") + _run60(rig, "B", "ablated")
    path = _verdict_over(rig, "e60.jsonl", lines, "exp60_verdict", {"all_rows": True})
    wrong = {**REAL_TABLE["exp60_verdict"], "complete": {"rule": "campaign_seeds", "sets": {"fear": [11]}}}
    _assert_refused(
        _judge(rig, path, table={**REAL_TABLE, "exp60_verdict": wrong}), "carries no `seeds_per_run_arm` rule"
    )


# ── 5b-2 review folds ─────────────────────────────────────────────────────────────────────────────────────


def test_exp61_a_default_verdict_names_no_campaign_over_all_rows(rig) -> None:
    """`verdict` with no --campaign-id writes campaign_id null over all_rows: the file's single campaign is proven."""
    rig.base(STALE_BOTH)
    assert _v61(rig, _campaign61(rig), scope={"all_rows": True}, campaign=None).status == G.ESTABLISHED
    _assert_refused(_v61(rig, _campaign61(rig), name="e61b.jsonl", campaign=None), "not the file's campaign")


def _with_invalid(lines: list[dict], agent: str) -> list[dict]:
    """Insert an invalid placement as the harness logs it: no arm, no exploratory flag."""
    probe = next(ln for ln in lines if ln.get("event") in ("probe", "trial"))
    bad = {k: v for k, v in probe.items() if k not in ("arm", "exploratory_agent")}
    bad.update(agent=agent, invalid=True)
    return lines[:-2] + [bad] + lines[-2:]


def test_exp53_an_invalid_placement_without_arm_is_fine_for_a_loaded_agent(rig) -> None:
    _base53(rig)
    p1 = _with_invalid(_group53(rig, "P1", 1, 2e9), "taught_seed42")
    assert _v53(rig, p1 + _group53(rig, "P2", 2, 2e9 + 100)).status == G.ESTABLISHED
    p1 = _with_invalid(_group53(rig, "P1", 1, 2e9), "stranger_seed99")
    _assert_refused(_v53(rig, p1 + _group53(rig, "P2", 2, 2e9 + 100), name="e53b.jsonl"), "unlike its load")


def test_exp53_a_scoped_run_must_load_agents(rig) -> None:
    _base53(rig)
    p1 = [ln for ln in _group53(rig, "P1", 1, 2e9) if ln.get("event") not in ("agent_load", "probe")]
    _assert_refused(_v53(rig, p1 + _group53(rig, "P2", 2, 2e9 + 100)), "loads no agent")


def test_exp53_the_manifest_must_be_at_the_merge_base(rig) -> None:
    rig.base(STALE_BOTH)  # no manifest on main
    _manifest53(rig)
    _assert_refused(_v53(rig, _group53(rig, "P1", 1, 2e9) + _group53(rig, "P2", 2, 2e9 + 100)), "not at the merge-base")


def test_exp53_runs_used_secondary_must_be_a_complete_secondary_run(rig) -> None:
    _base53(rig)
    lines = _group53(rig, "P1", 1, 2e9) + _group53(rig, "P2", 2, 2e9 + 100)
    _assert_refused(_v53(rig, lines, runs_used={"phase1": "P1", "primary": "P2", "secondary": "P1"}),
                    "runs_used.secondary is not a complete phase-2 run")  # fmt: skip


def test_a_ruled_kind_may_lack_its_block_at_the_merge_base_only() -> None:
    entry = {"rows": ["T1-11"], "targets": {"RE-VALIDATED": ["PASS"]}}
    assert G.pass_table_problems({"exp60_verdict": entry}, "the merge-base", at_base=True) == []
    assert G.pass_table_problems({"exp60_verdict": entry}, "HEAD") != []
    stray = {**entry, "complete": {"rule": "seeds_per_run_arm", "sets": {"a": [1]}}}
    assert G.pass_table_problems({"exp57_verdict": stray}, "the merge-base", at_base=True) != []  # a dropped rule


def test_an_inherited_prereg_clause_whose_entry_changed_shape_never_blocks(rig) -> None:
    rig.base(STALE_BOTH)
    tree_pin = {**{k: v for k, v in PREREG_CLAUSE.items() if k != "sha256"}, "tree": "t"}
    assert G.exceptions_problems([tree_pin], [tree_pin], G.Repo(rig.root)) == []  # on main already: inert, not fatal
    assert G.exceptions_problems([], [tree_pin], G.Repo(rig.root)) != []  # new: refused


def test_the_exp53_pins_match_the_harness_constants(monkeypatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_e53", REPO / "scripts/orient_backbone/exp53_cross_context_readout.py"
    )
    monkeypatch.syspath_prepend(str(REPO / "scripts/orient_backbone"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    start = REAL_TABLE["exp53_verdict"]["complete"]["start"]
    assert start["targets"] == list(mod.TARGETS) and start["exploratory_targets"] == list(mod.EXPLORATORY_TARGETS)


# ── campaigns: a verdict is bound to its campaign, and a successor to its pinned closure ─────────────────


@pytest.mark.parametrize(
    "edit, expected",
    [
        (lambda r: r.update(experiment="zz"), "not in the judge's campaign table"),
        (lambda r: r.update(experiment="10c2"), "is not of kind"),
        (lambda r: r["apparatus"]["markers"][0].update(ref="refs/tags/o19/10/attempt-1-x"), "is not campaign 09's"),
        (
            lambda r: r["apparatus"].update(
                markers=[
                    {
                        "run_id": f"{k:032x}",
                        "k": k,
                        "ref": f"refs/tags/o19/09/attempt-{k}-{k:032x}",
                        "peeled": r["apparatus"]["markers"][0]["peeled"],
                    }
                    for k in range(1, 5)
                ]  # fmt: skip
            ),
            "more than 3 attempts",
        ),
    ],
)
def test_an_o19_verdict_is_bound_to_its_campaign(rig, monkeypatch, edit, expected) -> None:
    _t39_move(rig, monkeypatch, verdict_edit=edit)
    failures, _ = rig.run()
    assert any(expected in f for f in failures), failures


def _t11_from_campaign_2(rig: Rig, monkeypatch, *, closure: bool = True, succession: bool = True) -> list[str]:
    """BASE: T1-1 STALE (+ campaign 1's closure on main). HEAD: a campaign-2 Exp 10 verdict moves T1-1 to MAINTAINED."""
    import o19_verdict as v

    sup = v.PROTOCOL["10c2"]["supersedes"]
    if closure:
        rig.write(sup["verdict"], (REPO / sup["verdict"]).read_bytes())
    rig.prereg[f"{DATA}/rerun_exp10_o19c2"] = "PASS"
    rig.base(ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]))
    record = o19_attempt(rig, "10c2", rig.base_sha, monkeypatch=monkeypatch)
    if succession:
        record["apparatus"]["succession"] = {**sup, "closure_landed": 1.0, "prereg_landed": 1.0}
    (rig.root / DATA / "rerun_exp10_o19c2" / "verdict.json").write_text(json.dumps(record, indent=1))
    cite = f"**Evidence:** `{DATA}/rerun_exp10_o19c2/verdict.json`."
    rig.head(
        ledger(
            [t1("T1-1", f"**Status: MAINTAINED 2026-10-02**. {cite}")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]
        )
    )
    failures, _ = rig.run()
    return failures


def test_a_campaign_2_pass_never_supports_maintained(rig, monkeypatch) -> None:
    """#1059 (owner decision 2026-10-02): a successor campaign's verdict supports REPRODUCED, never MAINTAINED."""
    failures = _t11_from_campaign_2(rig, monkeypatch)
    assert any("campaign 10c2 is a successor campaign" in f and "never MAINTAINED" in f for f in failures), failures


def test_a_campaign_2_verdict_without_its_pinned_closure_is_refused(rig, monkeypatch) -> None:
    failures = _t11_from_campaign_2(rig, monkeypatch, closure=False)
    assert any("is not the pinned verdict" in f for f in failures), failures


def test_a_campaign_2_verdict_that_records_no_succession_is_refused(rig, monkeypatch) -> None:
    failures = _t11_from_campaign_2(rig, monkeypatch, succession=False)
    assert any("does not record campaign 10c2's succession" in f for f in failures), failures


def test_an_o19_verdict_naming_another_rows_file_is_refused(rig, monkeypatch) -> None:
    """Same bytes, matching data_sha256, another path: the hash check passes, the campaign binding refuses."""
    other = f"{DATA}/rerun_exp09_o19/rows_copy.jsonl"  # beside the verdict, so only the campaign binding refuses

    def copy_rows(rg):
        rg.write(other, (rg.root / DATA / "rerun_exp09_o19" / "rows.jsonl").read_bytes())

    _t39_move(rig, monkeypatch, rows_edit=copy_rows, verdict_edit=lambda r: r.update(data=other))
    failures, _ = rig.run()
    assert any("is not campaign 09's rows file" in f for f in failures), failures


# ── #1050: the judge is the BOUND blob, bound to the data; a judge edit keeps every old verdict ──────────


def _o19_reasons(rig: Rig) -> list[str]:
    return rig.run()[0]


def test_a_verdict_commit_off_the_first_parent_history_is_refused(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, side_branch=True)  # vc is reachable from main, but only through a merged branch
    assert any("is not on the merge-base's first-parent history" in f for f in _o19_reasons(rig))


def test_a_verdict_commit_that_is_not_its_executed_commit_is_refused(rig, monkeypatch) -> None:
    def other(r):  # the commit the attempt ran on: on main, but not where the verdict was written
        r["provenance"]["executed_git_hash"] = r["apparatus"]["markers"][0]["peeled"]

    _t39_move(rig, monkeypatch, verdict_edit=other)
    assert any("is not the verdict's own executed commit" in f for f in _o19_reasons(rig))


@pytest.mark.parametrize("path", ["scripts/o19_verdict.py", "scripts/o19_rerun.py", "prereg"])
def test_a_bound_file_that_changed_after_the_run_is_refused(rig, monkeypatch, path) -> None:
    import o19_verdict as v

    rel = v.PROTOCOL["09"]["prereg"] if path == "prereg" else path

    def change(rg):  # lands with the rows: the verdict commit's blob is not the one the attempt ran with
        rg.write(rel, (rg.root / rel).read_bytes() + b"\n# changed after the run\n")

    _t39_move(rig, monkeypatch, rows_edit=change)
    assert any(f"bound {rel} is" in f and "at executed commit" in f for f in _o19_reasons(rig))


def test_a_bound_blob_that_is_not_the_verdict_commits_is_refused(rig, monkeypatch) -> None:
    other = "1" * 40  # a blob id no commit holds at that path

    _t39_move(rig, monkeypatch, verdict_edit=lambda r: r["bound_files"].update({R.O19_RERUN: other}))
    failures = _o19_reasons(rig)
    assert any(f"bound {R.O19_RERUN} is" in f and "at verdict_commit" in f for f in failures), failures


def test_rows_at_the_verdict_commit_must_be_the_judged_bytes(rig, monkeypatch) -> None:
    def append(rg):  # in the PR: a blank line (the rows parse the same), restamped below
        p = rg.root / DATA / "rerun_exp09_o19" / "rows.jsonl"
        p.write_bytes(p.read_bytes() + b"\n")

    _t39_move(rig, monkeypatch, head_edit=append, verdict_edit=_restamp(rig, "09"))
    failures = _o19_reasons(rig)
    assert any("at verdict_commit" in f and "is not the data_sha256 bytes" in f for f in failures), failures


def test_gates_the_bound_judge_does_not_reproduce_are_refused(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, verdict_edit=lambda r: r["gates"]["H1"].update(status="FAIL"))
    assert any("bound judge gives a different gates" in f for f in _o19_reasons(rig))


@pytest.mark.parametrize("edit", ["missing", "extra"])
def test_bound_files_are_exactly_the_judge_the_harness_and_the_prereg(rig, monkeypatch, edit) -> None:
    import o19_verdict as v

    def change(r):
        if edit == "missing":
            del r["bound_files"][v.PROTOCOL["09"]["prereg"]]
        else:  # a real, unchanged file: its blob checks pass, the set does not
            r["bound_files"][G.PASS_TABLE] = _git(rig.root, "rev-parse", f"{r['verdict_commit']}:{G.PASS_TABLE}")

    _t39_move(rig, monkeypatch, verdict_edit=change)
    assert any("bound_files names" in f and "not exactly" in f for f in _o19_reasons(rig))


def test_an_older_judge_chosen_by_hand_is_refused(rig, monkeypatch) -> None:
    """A1: main once held a looser judge (it says PASS). The attempt ran after the fix; the author points
    verdict_commit / executed / bound_files / the source hash at the old commit so the old judge re-judges."""
    current = (rig.root / R.O19_JUDGE).read_text()
    old = current.replace('out["verdict"] = gates["verdict"]', 'out["verdict"] = "PASS"')
    assert old != current
    rig.write(R.O19_JUDGE, old)
    rig.base(ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]))
    old_commit = rig.base_sha
    rig.write(R.O19_JUDGE, current)
    fixed = rig.commit("the judge is fixed", BASE_DATE)
    record = o19_attempt(rig, "09", fixed, monkeypatch=monkeypatch)
    record.update(
        verdict="PASS",
        verdict_commit=old_commit,
        verdict_source_sha256=hashlib.sha256(old.encode()).hexdigest(),
    )
    record["provenance"]["executed_git_hash"] = old_commit
    record["bound_files"][R.O19_JUDGE] = _git(rig.root, "rev-parse", f"{old_commit}:{R.O19_JUDGE}")
    (rig.root / DATA / "rerun_exp09_o19" / "verdict.json").write_text(json.dumps(record, indent=1))
    cite = f"**Evidence:** `{DATA}/rerun_exp09_o19/verdict.json`."
    rig.head(
        ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", f"**Status: PARTIAL 2026-10-02**. {cite}")])
    )
    failures = _o19_reasons(rig)
    assert any(f"bound {R.O19_JUDGE} is" in f and "at executed commit" in f for f in failures), failures
    assert any("at verdict_commit" in f and "not the data_sha256 bytes" in f for f in failures), failures
    assert not any("different verdict" in f for f in failures), failures  # the binding refuses, not the re-judge


def _verdict_on_main_then(rig: Rig, monkeypatch, edit) -> list[str]:
    """BASE: an Exp 09 O19 verdict on main (no row cites it). HEAD: only ``edit`` to the judge."""
    text = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])
    rig.base(text)
    o19_attempt(rig, "09", rig.base_sha, monkeypatch=monkeypatch)
    rig.base(text)  # the verdict lands
    edit(rig)
    rig.commit("edit the judge", HEAD_DATE)
    return rig.run()[0]


def _judge_text(rig: Rig, old: str, new: str) -> None:
    text = (rig.root / R.O19_JUDGE).read_text()
    assert old in text
    rig.write(R.O19_JUDGE, text.replace(old, new))


def test_a_judge_edit_that_keeps_every_verdict_passes(rig, monkeypatch) -> None:
    assert (
        _verdict_on_main_then(rig, monkeypatch, lambda rg: _judge_text(rg, "MAX_ATTEMPTS = 3", "MAX_ATTEMPTS = 3  # x"))
        == []
    )


def test_a_judge_edit_that_flips_an_existing_verdict_fails(rig, monkeypatch) -> None:
    failures = _verdict_on_main_then(
        rig, monkeypatch, lambda rg: _judge_text(rg, 'out["verdict"] = gates["verdict"]', 'out["verdict"] = "FAIL"')
    )
    assert any("a judge edit changes an existing O19 verdict" in f and "verdict was 'PARTIAL'" in f for f in failures)


def test_deleting_the_judge_while_verdicts_exist_fails(rig, monkeypatch) -> None:
    failures = _verdict_on_main_then(rig, monkeypatch, lambda rg: (rg.root / R.O19_JUDGE).unlink())
    assert any("is deleted while O19 verdicts exist" in f for f in failures), failures


def _history_blobs() -> list[tuple[str, bytes]]:
    """Every ``o19_verdict.py`` blob on the real ``origin/main``'s first-parent history."""

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True).stdout

    if git("rev-parse", "--is-shallow-repository").strip() != b"false" or not git(
        "rev-parse", "--verify", "-q", "origin/main"
    ):
        return []
    commits = git("log", "--first-parent", "--format=%H", "origin/main", "--", R.O19_JUDGE).decode().split()
    return [
        (c, git("show", f"{c}:{R.O19_JUDGE}")) for c in commits if git("cat-file", "-e", f"{c}:{R.O19_JUDGE}") == b""
    ]


@pytest.mark.skipif(not _history_blobs(), reason="needs origin/main's full history (the unit-test job is depth 1)")
def test_every_judge_on_mains_history_loads_through_the_gate() -> None:
    """N2: the gate re-judges a verdict with the judge that wrote it, so every judge main ever held must still load
    through ``load_o19_judge`` with the interface ``rejudge_o19`` calls (N3: and import only the standard library)."""
    for commit, source in _history_blobs():
        mod = R.load_o19_judge(source)
        assert not isinstance(mod, str), (commit[:12], mod)
        assert all(hasattr(mod, name) for name in R.O19_INTERFACE), commit[:12]
        _assert_stdlib_only(source.decode())


def test_a_marker_without_its_commit_is_refused(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, verdict_edit=lambda r: r["apparatus"]["markers"][0].pop("peeled"))
    assert any("records no peeled commit" in f for f in _o19_reasons(rig))


def test_an_attempt_that_ran_off_its_markers_commit_is_refused(rig, monkeypatch) -> None:
    other = "1" * 40
    _t39_move(rig, monkeypatch, verdict_edit=lambda r: r["apparatus"]["markers"][0].update(peeled=other))
    assert any("not its marker's commit" in f for f in _o19_reasons(rig))


def test_an_o19_verdict_cited_under_another_name_is_refused(rig, monkeypatch) -> None:
    """Half B finds O19 verdicts by their place beside the rows, so half A accepts them only there."""
    _t39_move(rig, monkeypatch)
    src = rig.root / DATA / "rerun_exp09_o19" / "verdict.json"
    (src.parent / "verdict_v2.json").write_text(src.read_text())
    cite = f"**Evidence:** `{DATA}/rerun_exp09_o19/verdict_v2.json`."
    rig.head(
        ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", f"**Status: PARTIAL 2026-10-02**. {cite}")])
    )
    assert any("is not its rows' verdict.json" in f for f in _o19_reasons(rig))


def test_a_judge_edit_cannot_rewrite_a_landed_verdict_to_match(rig, monkeypatch) -> None:
    """Half B re-judges the verdict as MAIN holds it: rewriting it in the same diff hides nothing."""

    def edit(rg):
        _judge_text(rg, 'out["verdict"] = gates["verdict"]', 'out["verdict"] = "FAIL"')
        path = rg.root / DATA / "rerun_exp09_o19" / "verdict.json"
        rec = json.loads(path.read_text())
        rec["verdict"] = "FAIL"
        path.write_text(json.dumps(rec, indent=1))

    failures = _verdict_on_main_then(rig, monkeypatch, edit)
    assert any("a judge edit changes an existing O19 verdict" in f and "verdict was" in f for f in failures), failures


def test_a_judge_main_once_held_must_still_load_through_the_gate(rig, monkeypatch) -> None:
    """N2, enforced on every gate run (the lint job has the full history): an old judge the gate can no longer load
    would strand its verdicts."""
    good = (rig.root / R.O19_JUDGE).read_text()
    rig.write(R.O19_JUDGE, good.replace("MARKER_NAMESPACE = ", "OLD_NAMESPACE = ", 1))
    rig.base(ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]))
    rig.write(R.O19_JUDGE, good)
    rig.base(ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]))
    rig.write("README.md", "an unrelated change\n")
    rig.head(ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]))
    assert any("no longer loads through the gate" in f and "MARKER_NAMESPACE" in f for f in rig.run()[0])


def test_a_judge_edit_cannot_delete_a_landed_verdict_to_escape(rig, monkeypatch) -> None:
    def edit(rg):
        _judge_text(rg, 'out["verdict"] = gates["verdict"]', 'out["verdict"] = "FAIL"')
        (rg.root / DATA / "rerun_exp09_o19" / "verdict.json").unlink()

    failures = _verdict_on_main_then(rig, monkeypatch, edit)
    assert any("a judge edit changes an existing O19 verdict" in f and "verdict was" in f for f in failures), failures


# ── #1059: a successor campaign supports REPRODUCED only on byte-identical subject code ─────────────────────
# The rig closes its own Exp 10 campaign 1 (an ABORT on the garden phase, phases 1-2 committed ok), pins that closure
# in its judge's campaign-2 entry, then runs a campaign-2 attempt; HEAD moves T1-1 citing campaign 2's verdict.

REAL_C1_PIN = "5da2128968bb4516350aa26680294d9f79966424a518fc37b3dd68c0bf826fbe"
STALE_BOTH_ROWS = ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")])


def _stored(session_dir: Path, name: str) -> Path:
    return next(p for p in (session_dir / name, session_dir / f"{name}.gz") if p.exists())


def _close_campaign_1(rig: Rig, monkeypatch, *, rows_edit=None, record_edit=None) -> str:
    """Land campaign 1's aborted attempt (garden phase failed) and its stamped ABORT closure; returns its SHA-256."""
    import o19_rerun as h
    import o19_verdict as v
    from _provenance import stamp_verdict

    data_dir = rig.root / v.data_dir("10")
    rows_path = data_dir / "rows.jsonl"
    monkeypatch.setattr(h._provenance, "_RUN_ID", {})
    monkeypatch.setattr(h._provenance, "evidence_out_path", lambda *a, **k: rows_path)
    assert h.main(["run", "--exp", "10", "--mock"]) == 0
    executed = rig.base_sha
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    for r in rows:
        r["mock"] = False
        r["provenance"].update(executed_git_hash=executed, working_tree_dirty_src_scripts=False)
    rows[-1].update(status="failed", reason="SimRunFailed: planning_failed")
    if rows_edit:
        rows_edit(rows, data_dir)
    rows_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    landed = rig.commit("campaign 1's attempt lands", BASE_DATE)
    attempts = v.attempts_from_rows(rows)
    rid = next(iter(attempts))
    out = v.judge("10", [{"run_id": rid, "k": 1, "rows": attempts[rid]}], data_dir)
    assert out["verdict"] == "ABORT"
    out.update(
        apparatus_checked=True,
        apparatus={
            "markers": [{"run_id": rid, "k": 1, "ref": f"{v.MARKER_NAMESPACE}/10/attempt-1-{rid}", "peeled": executed}]
        },  # fmt: skip
        bound_files={
            p: _git(rig.root, "rev-parse", f"{landed}:{p}")
            for p in (R.O19_JUDGE, R.O19_RERUN, v.PROTOCOL["10"]["prereg"])
        },  # fmt: skip
        verdict_commit=landed,
    )
    stamp_verdict(out, repo_root=rig.root, kind="exp10_verdict", data=rows_path, data_bytes=rows_path.read_bytes(),
                  scope={"all_rows": True}, mock=False)  # fmt: skip
    if record_edit:
        record_edit(out)
    raw = json.dumps(out, indent=1).encode()
    (data_dir / "verdict.json").write_bytes(raw)
    rig.commit("campaign 1 closes", BASE_DATE)
    return hashlib.sha256(raw).hexdigest()


def _reproduce(rig: Rig, monkeypatch, *, token="REPRODUCED", rows_edit=None, record_edit=None, before_c2=None,
               after_c2=None) -> list[str]:  # fmt: skip
    """BASE: T1-1 STALE, campaign 1 closed, the judge pinning that closure, ``before_c2(rig)`` (lands with the pin),
    a campaign-2 attempt landed, then ``after_c2(rig)`` on main. HEAD: T1-1 -> ``token`` citing campaign 2."""
    import o19_verdict as v

    rig.prereg[f"{DATA}/rerun_exp10_o19c2"] = "PASS"
    rig.base(STALE_BOTH_ROWS)
    pin = _close_campaign_1(rig, monkeypatch, rows_edit=rows_edit, record_edit=record_edit)
    _judge_text(rig, REAL_C1_PIN, pin)
    if before_c2:
        before_c2(rig)
    rig.base_sha = rig.commit("the judge pins campaign 1's closure", BASE_DATE)
    record = o19_attempt(rig, "10c2", rig.base_sha, monkeypatch=monkeypatch)
    record["apparatus"]["succession"] = {**v.PROTOCOL["10c2"]["supersedes"], "verdict_sha256": pin,
                                         "closure_landed": 1.0, "prereg_landed": 1.0}  # fmt: skip
    verdict = rig.root / DATA / "rerun_exp10_o19c2" / "verdict.json"
    verdict.unlink()  # o19_attempt wrote it uncommitted: the verdict arrives in HEAD, after anything main does next
    if after_c2:
        after_c2(rig)
    verdict.write_text(json.dumps(record, indent=1))
    cite = f"**Evidence:** `{DATA}/rerun_exp10_o19c2/verdict.json`."
    rig.head(ledger([t1("T1-1", f"**Status: {token} 2026-10-02**. {cite}")],
                    [t3("T3-9", "**Status: STALE 2026-09-30**.")]))  # fmt: skip
    return rig.run()[0]


def test_a_successor_on_identical_subject_code_supports_reproduced(rig, monkeypatch) -> None:
    assert _reproduce(rig, monkeypatch) == []


def test_a_successor_never_supports_maintained_even_on_identical_code(rig, monkeypatch) -> None:
    failures = _reproduce(rig, monkeypatch, token="MAINTAINED")
    assert any("supports REPRODUCED, never MAINTAINED" in f for f in failures), failures


def test_a_root_campaign_never_supports_reproduced(rig, monkeypatch) -> None:
    rig.base(STALE_BOTH_ROWS)
    o19_attempt(rig, "10", rig.base_sha, monkeypatch=monkeypatch)  # campaign 1 as a root: a PASS
    cite = f"**Evidence:** `{DATA}/rerun_exp10_o19/verdict.json`."
    for token, refused in (("REPRODUCED", True), ("MAINTAINED", False)):
        rig.head(ledger([t1("T1-1", f"**Status: {token} 2026-10-02**. {cite}")],
                        [t3("T3-9", "**Status: STALE 2026-09-30**.")]))  # fmt: skip
        failures = rig.run()[0]
        if refused:
            assert any("is a root campaign: only a successor" in f for f in failures), failures
        else:
            assert failures == [], failures


def _src_change(path: str):
    def change(rg: Rig) -> None:
        rg.write(path, "changed = True\n")

    return change


def test_a_successor_on_changed_subject_code_supports_nothing(rig, monkeypatch) -> None:
    failures = _reproduce(rig, monkeypatch, before_c2=_src_change("src/maxim/runtime/agent_loop.py"))
    assert any("the subject differs between executed commits" in f and "src/maxim/runtime/agent_loop.py" in f
               for f in failures), failures  # fmt: skip


@pytest.mark.parametrize("path", ["pyproject.toml", "scenarios/campaigns/x.yaml", "data/motion/x.json", "uv.lock"])
def test_the_subject_is_more_than_src(rig, monkeypatch, path) -> None:
    failures = _reproduce(rig, monkeypatch, before_c2=_src_change(path))
    assert any("the subject differs" in f and path in f for f in failures), failures


def test_the_standing_exclusion_is_not_subject(rig, monkeypatch) -> None:
    assert _reproduce(rig, monkeypatch, before_c2=_src_change("src/maxim/utils/function_length_baseline.json")) == []


def test_a_mode_change_is_a_subject_change(rig, monkeypatch) -> None:
    def exec_bit(rg: Rig) -> None:
        os.chmod(rg.root / "src/maxim/run.py", 0o755)

    rig.write("src/maxim/run.py", "x = 1\n")
    failures = _reproduce(rig, monkeypatch, before_c2=exec_bit)
    assert any("the subject differs" in f and "src/maxim/run.py" in f for f in failures), failures


def test_a_successor_after_a_leaked_failed_gate_supports_nothing(rig, monkeypatch) -> None:
    """Campaign 1 committed phases 1-2 ok; phase 1's store held 2 memories, so P0 FAILED there (D2)."""

    def thin_store(rows, data_dir):
        sdir = data_dir / rows[0]["session_id"]
        store = _stored(sdir, "aut_hippocampus.json")
        data = json.dumps({"memories": [{"id": "a"}, {"id": "b"}]}).encode()
        store.write_bytes(data)
        rows[0]["files"]["aut_hippocampus.json"] = hashlib.sha256(data).hexdigest()

    failures = _reproduce(rig, monkeypatch, rows_edit=thin_store)
    assert any("a FAILED gate leaked into a predecessor" in f and "P0" in f for f in failures), failures


def test_a_successor_whose_bound_judge_lacks_the_bar_supports_nothing(rig, monkeypatch) -> None:
    def no_bar(rg: Rig) -> None:
        _judge_text(rg, 'out["leaked_gates"] = chain_leaked_gate_problems(exp, data_root.parent)', "pass")

    failures = _reproduce(rig, monkeypatch, before_c2=no_bar)
    assert any("does not compute the leaked-gate bar" in f for f in failures), failures


def test_a_successor_with_another_argv_supports_nothing(rig, monkeypatch) -> None:
    def flag(rows, _data_dir):
        rows[0]["sim_argv"] = [*rows[0]["sim_argv"], "--some-mechanism-flag"]

    failures = _reproduce(rig, monkeypatch, rows_edit=flag)
    assert any("phase 0: the recorded sim argv or MAXIM_* env differs" in f for f in failures), failures


def test_a_successor_with_another_env_supports_nothing(rig, monkeypatch) -> None:
    def env(rows, _data_dir):
        rows[1]["sim_env"] = {**rows[1]["sim_env"], "MAXIM_SUBSTRATE_PATH": "1"}

    failures = _reproduce(rig, monkeypatch, rows_edit=env)
    assert any("phase 1: the recorded sim argv or MAXIM_* env differs" in f for f in failures), failures


def test_a_successor_whose_table_phases_differ_from_the_roots_supports_nothing(rig, monkeypatch) -> None:
    """S6 on the bound tables: campaign 1's own (bound) judge names other phases than campaign 2's judge does."""
    text = (rig.root / R.O19_JUDGE).read_text()
    old = '    ("baseline", EXP10_GOAL_DUNGEON, 8, False, [], {}),'
    assert old in text
    alt = rig.root.parent / "alt_judge.py"
    alt.write_text(text.replace(old, '    ("baseline", EXP10_GOAL_DUNGEON, 8, False, [], {"MAXIM_X": "1"}),'))
    oid = _git(rig.root, "hash-object", "-w", str(alt))

    def other_judge(record):
        record["bound_files"][R.O19_JUDGE] = oid

    failures = _reproduce(rig, monkeypatch, record_edit=other_judge)
    assert any("bound phases, HARNESS_ENV or model pins are not root campaign 10's" in f for f in failures), failures


@pytest.mark.parametrize(
    "old, new",
    [
        ("N_CTX = 8192", "N_CTX = 4096"),
        ('MODEL_GGUF = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"', 'MODEL_GGUF = "other.gguf"'),
    ],
)
def test_a_successor_whose_model_pins_differ_from_the_roots_supports_nothing(rig, monkeypatch, old, new) -> None:
    """S6 (review): the model is configured through `maxim config`, so argv and env never show it; the bound judges'
    model pins are compared across the chain."""
    text = (rig.root / R.O19_JUDGE).read_text()
    assert old in text
    alt = rig.root.parent / "alt_judge.py"
    alt.write_text(text.replace(old, new))
    oid = _git(rig.root, "hash-object", "-w", str(alt))

    def other_judge(record):
        record["bound_files"][R.O19_JUDGE] = oid

    failures = _reproduce(rig, monkeypatch, record_edit=other_judge)
    assert any("model pins are not root campaign 10's" in f for f in failures), failures
    assert R.MODEL_FIELDS == ("MODEL_PROFILE", "MODEL_PROFILE_STAMPED", "MODEL_GGUF", "N_CTX")


def _launder(rg: Rig) -> None:
    """Mark campaign 1's committed phases failed in its rows: the ok prefix empties, so nothing would leak."""
    import o19_verdict as v

    path = rg.root / v.rows_path("10")
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    for r in rows:
        r["status"] = "failed"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def _thin_store(rows, data_dir):
    sdir = data_dir / rows[0]["session_id"]
    data = json.dumps({"memories": [{"id": "a"}, {"id": "b"}]}).encode()
    _stored(sdir, "aut_hippocampus.json").write_bytes(data)
    rows[0]["files"]["aut_hippocampus.json"] = hashlib.sha256(data).hexdigest()


def test_a_leak_cannot_be_laundered_by_editing_a_predecessors_rows_in_the_pr(rig, monkeypatch) -> None:
    """#1059 review (DO-NOT-MERGE probe): the PR edits campaign 1's rows; the bar reads main's copy, pinned."""
    failures = _reproduce(rig, monkeypatch, rows_edit=_thin_store, after_c2=_launder)
    assert any("FAILED gate leaked" in f or "cannot be ruled out" in f for f in failures), failures


def test_a_leak_cannot_be_laundered_by_editing_a_predecessors_rows_on_main(rig, monkeypatch) -> None:
    """The same edit landed on main after the closure: the rows are not the bytes the pinned closure judged."""

    def on_main(rg: Rig) -> None:
        _launder(rg)
        rg.base_sha = rg.commit("campaign 1's rows edited on main", BASE_DATE)

    failures = _reproduce(rig, monkeypatch, rows_edit=_thin_store, after_c2=on_main)
    assert any("are not the bytes its pinned closure judged" in f for f in failures), failures


def test_a_predecessor_is_read_as_main_holds_it(rig, monkeypatch) -> None:
    """A PR-only change to a predecessor's rows (a byte that parses the same) and closure (a trailing space) reaches
    neither the bar nor the closure pin (both read main's copy); the PR fails only because a closed campaign's data
    directory is immutable."""

    def touch(rg: Rig) -> None:
        import o19_verdict as v

        for rel in (v.rows_path("10"), v.data_dir("10") + "/verdict.json"):
            path = rg.root / rel
            path.write_bytes(path.read_bytes() + b"\n")

    failures = _reproduce(rig, monkeypatch, after_c2=touch)
    assert failures and all("campaign 10 is closed (its verdict is on main)" in f for f in failures), failures


def _pr_x(rg: Rig) -> None:
    """PR-X (#1059 delta review): mark campaign 1's rows failed, rewrite its closure's data_sha256 to match and re-pin
    campaign 2's `supersedes`: main stays self-consistent, and the leaked-gate bar would read nothing."""
    import o19_verdict as v

    _launder(rg)
    vpath = rg.root / v.data_dir("10") / "verdict.json"
    old_raw = vpath.read_bytes()
    rec = json.loads(old_raw)
    rec["data_sha256"] = hashlib.sha256((rg.root / v.rows_path("10")).read_bytes()).hexdigest()
    raw = json.dumps(rec, indent=1).encode()
    vpath.write_bytes(raw)
    _judge_text(rg, hashlib.sha256(old_raw).hexdigest(), hashlib.sha256(raw).hexdigest())


def test_pr_x_rewriting_a_closed_campaign_and_its_pin_fails_the_gate(rig, monkeypatch) -> None:
    """Each half catches PR-X alone: the frozen `supersedes` pin, and the closed campaign's immutable data."""
    rig.prereg[f"{DATA}/rerun_exp10_o19c2"] = "PASS"
    rig.base(STALE_BOTH_ROWS)
    pin = _close_campaign_1(rig, monkeypatch, rows_edit=_thin_store)
    _judge_text(rig, REAL_C1_PIN, pin)
    rig.base_sha = rig.commit("the judge pins campaign 1's closure", BASE_DATE)
    _pr_x(rig)
    rig.head(STALE_BOTH_ROWS)
    failures = rig.run()[0]
    assert any("campaign 10c2's `supersedes`" in f and "was removed or edited" in f for f in failures), failures
    assert any("campaign 10 is closed (its verdict is on main)" in f for f in failures), failures


def test_a_two_pr_launder_already_on_main_supports_nothing(rig, monkeypatch) -> None:
    """PR-X reached main before campaign 2 ran (here: before these rules existed): the successor's gate reads main's
    history, and a closed campaign's directory touched after its closure landed supports nothing."""
    failures = _reproduce(rig, monkeypatch, rows_edit=_thin_store, before_c2=_pr_x)
    assert any("changed on main after its closure landed" in f for f in failures), failures


def test_the_control_leak_is_detected(rig, monkeypatch) -> None:
    failures = _reproduce(rig, monkeypatch, rows_edit=_thin_store)
    assert any("FAILED gate leaked" in f for f in failures), failures


def test_a_pinned_closure_outside_the_predecessors_own_directory_supports_nothing(rig, monkeypatch) -> None:
    """#1059 delta review 2: the gate itself requires the pinned closure to be the predecessor's own data directory's
    (the directory the freeze, the history check and the bar read), not only the judge's ``protocol_problems``."""
    real = f"{DATA}/rerun_exp10_o19/verdict.json"
    elsewhere = "docs/elsewhere/verdict.json"

    def relocate(rg: Rig) -> None:
        rg.write(elsewhere, (rg.root / real).read_text())
        _judge_text(rg, f'"verdict": "{real}"', f'"verdict": "{elsewhere}"')
        _judge_text(rg, "problems.append(f\"campaign {key}: the closure verdict is not {sup['key']}'s own\")", "pass")

    failures = _reproduce(rig, monkeypatch, before_c2=relocate)
    assert any("is not its own data directory's" in f for f in failures), failures


def test_amending_a_successor_entry_other_than_its_supersedes_is_not_the_pin_rule(rig, monkeypatch) -> None:
    """Only `supersedes` is frozen by the pin rule: a not-yet-run successor's prereg may still be amended."""
    rig.base(STALE_BOTH_ROWS)
    _judge_text(rig, '"scope": "rerun_exp10_o19c2",', '"scope": "rerun_exp10_o19c2b",')  # an entry field, not the pin
    rig.write("README.md", "x\n")
    rig.head(STALE_BOTH_ROWS)
    assert not any("`supersedes`" in f for f in rig.run()[0])


def test_a_missing_executed_commit_refuses(rig, monkeypatch) -> None:
    def ghost(record):
        record["apparatus"]["markers"][0]["peeled"] = "1" * 40

    failures = _reproduce(rig, monkeypatch, record_edit=ghost)
    assert any("executed commit 111111111111 does not exist here" in f for f in failures), failures


def test_has_a_predecessor_is_read_from_the_merge_base_table(rig, monkeypatch) -> None:
    """S2: after the verdict, main's table edits campaign 2's entry: the bound judge and main disagree."""

    def edit(rg: Rig) -> None:
        _judge_text(rg, '"cause_issue": 1042,', '"cause_issue": 1043,')
        rg.base_sha = rg.commit("main edits the campaign-2 entry", BASE_DATE)

    failures = _reproduce(rig, monkeypatch, after_c2=edit)
    assert any("is not its bound judge's (S2)" in f for f in failures), failures


def test_a_successor_partial_obeys_the_identity_check_too(rig, monkeypatch) -> None:
    """PARTIAL is not refused by the token rule, but the identity check applies (owner decision 2026-10-02)."""
    import o19_verdict as v

    sup = v.PROTOCOL["10c2"]["supersedes"]
    assert sup["key"] == "10"
    rig.write(G.PASS_TABLE, json.dumps({**json.loads((REPO / G.PASS_TABLE).read_text()),
              "exp10_verdict": {"rows": ["T1-1"], "targets": {"PARTIAL": ["PASS"]},
                                "require": {"apparatus_checked": True}}}))  # fmt: skip
    failures = _reproduce(rig, monkeypatch, token="PARTIAL", before_c2=_src_change("src/maxim/x.py"))
    assert any("the subject differs" in f for f in failures), failures


def test_the_gate_and_the_harness_name_one_subject() -> None:
    import o19_verdict as v

    assert v.SUBJECT_PATHS == R.SUBJECT_PATHS and v.SUBJECT_EXCLUDED == R.SUBJECT_EXCLUDED
    assert "src/maxim" in R.SUBJECT_PATHS and "src/maxim/utils/function_length_baseline.json" in R.SUBJECT_EXCLUDED


def test_reproduced_is_an_o19_kinds_target_only() -> None:
    entry = {"rows": ["T1-12"], "targets": {"REPRODUCED": ["PASS"]}}
    assert any("only an O19 kind may" in p for p in G.pass_table_problems({"exp57_verdict": entry}, "HEAD"))
    assert G.pass_table_problems({"exp10_verdict": entry | {"rows": ["T1-1"]}}, "HEAD") == []


def _edit_table_in_pr(rig: Rig, monkeypatch, old: str, new: str) -> list[str]:
    rig.base(STALE_BOTH_ROWS)
    _judge_text(rig, old, new)
    rig.write("README.md", "x\n")
    rig.head(STALE_BOTH_ROWS)
    return rig.run()[0]


def test_a_named_predecessor_entry_is_frozen(rig, monkeypatch) -> None:
    """S3: campaign 10 is named by 10c2's `supersedes`: editing its entry (here its scope) fails."""
    failures = _edit_table_in_pr(rig, monkeypatch, '"scope": "rerun_exp10_o19",', '"scope": "rerun_exp10_o19x",')
    assert any("campaign 10 is frozen" in f for f in failures), failures


def test_a_campaign_with_data_on_main_is_frozen(rig, monkeypatch) -> None:
    """S3: rows on main freeze an entry nobody names (here Exp 09's)."""
    rig.write(f"{DATA}/rerun_exp09_o19/rows.jsonl", "\n")
    failures = _edit_table_in_pr(rig, monkeypatch, '"scope": "rerun_exp09_o19",', '"scope": "rerun_exp09_o19x",')
    assert any("campaign 09 is frozen" in f for f in failures), failures


def test_an_unfrozen_entry_may_still_be_edited(rig, monkeypatch) -> None:
    failures = _edit_table_in_pr(rig, monkeypatch, '"scope": "rerun_exp09_o19",', '"scope": "rerun_exp09_o19x",')
    assert not any("frozen" in f for f in failures), failures


def test_a_verdict_kind_belongs_to_one_experiment(rig, monkeypatch) -> None:
    """D1: Exp 63's campaign may not take Exp 10's kind (a fresh root for a claim another experiment owns)."""
    failures = _edit_table_in_pr(rig, monkeypatch, '"kind": "exp63_verdict",', '"kind": "exp10_verdict",')
    assert any("belongs to 2 experiments (D1" in f for f in failures), failures
