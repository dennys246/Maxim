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

import lint_evidence_gate as G  # noqa: E402

LEDGER = "docs/plans/behavioral_graduation_candidates.md"
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
        self.write(G.EXCEPTIONS, "[]\n")
        self.write(G.LEGACY_SNAPSHOT, "{}\n")
        self.write(G.PASS_TABLE, (REPO / G.PASS_TABLE).read_text())
        self.write(f"{DATA}/legacy_old.jsonl", '{"x": 1}\n')
        self.prereg = {f"{DATA}/rerun_exp09_o19": "PASS", f"{DATA}/rerun_exp10_o19": "PASS"}

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


def o19_attempt(rig: Rig, exp: str, executed: str, *, monkeypatch) -> dict:
    """Write a complete O19 attempt for ``exp`` under the rig's data dir and return its verdict record."""
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
            sim.update(executed_git_hash=executed, working_tree_dirty_src_scripts=False)
    rows_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    attempts = v.attempts_from_rows(rows)
    rid = next(iter(attempts))
    out = v.judge(exp, [{"run_id": rid, "k": 1, "rows": attempts[rid]}], data_dir)
    source = (rig.root / G.O19_JUDGE).read_bytes()
    blob = _git(rig.root, "hash-object", G.O19_JUDGE)
    out.update(
        apparatus_checked=True,
        apparatus={"markers": [{"run_id": rid, "k": 1}]},
        bound_files={G.O19_JUDGE: blob},
        verdict_source_sha256=hashlib.sha256(source).hexdigest(),
        provenance={
            "executed_git_hash": executed,
            "code_tree_sha256": "t" * 64,
            "working_tree_dirty_src_scripts": False,
        },
    )
    data_bytes = rows_path.read_bytes()
    stamp_verdict(out, repo_root=rig.root, kind=v.PROTOCOL[exp]["kind"], data=rows_path, data_bytes=data_bytes,
                  scope={"all_rows": True}, mock=False)  # fmt: skip
    (data_dir / "verdict.json").write_text(json.dumps(out, indent=1))
    return out


@pytest.fixture
def rig(tmp_path) -> Rig:
    return Rig(tmp_path / "repo")


def _t39_move(rig: Rig, monkeypatch, *, verdict_edit=None, rows_edit=None, ledger_token="PARTIAL", row="T3-9"):
    """BASE: T3-9 STALE. HEAD: an O19 Exp 09 attempt + verdict, and ``row`` moved to ``ledger_token`` citing it."""
    rig.base(ledger([t1("T1-1", "**Status: STALE 2026-09-30**.")], [t3("T3-9", "**Status: STALE 2026-09-30**.")]))
    record = o19_attempt(rig, "09", rig.base_sha, monkeypatch=monkeypatch)
    if rows_edit:
        rows_edit(rig)
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

    _t39_move(rig, monkeypatch, rows_edit=tamper)
    failures, _ = rig.run()
    assert any("differs from its row's SHA-256" in f for f in failures), failures


def test_a_session_file_present_plain_and_gz_is_refused(rig, monkeypatch) -> None:
    def both(rig):
        sessions = [p for p in (rig.root / DATA / "rerun_exp09_o19").iterdir() if p.is_dir()]
        gz = sessions[0] / "run_log.jsonl.gz"
        import gzip

        (sessions[0] / "run_log.jsonl").write_bytes(gzip.decompress(gz.read_bytes()))

    _t39_move(rig, monkeypatch, rows_edit=both)
    failures, _ = rig.run()
    assert any("present both plain and .gz" in f for f in failures), failures


def test_the_judge_must_be_the_one_that_wrote_the_verdict(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, verdict_edit=lambda r: r.update(verdict_source_sha256="0" * 64))
    failures, _ = rig.run()
    assert any("merge-base judge is not the one" in f for f in failures), failures


def test_a_verdict_the_judge_does_not_reproduce_is_refused(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, verdict_edit=lambda r: r.update(verdict="PASS"))
    failures, _ = rig.run()
    assert any("gives a different result" in f for f in failures), failures


def test_rows_without_a_marker_are_refused(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, verdict_edit=lambda r: r.update(apparatus={"markers": [{"run_id": "f" * 32, "k": 1}]}))
    failures, _ = rig.run()
    assert any("no start marker" in f for f in failures), failures


def test_mock_rows_sink_the_verdict(rig, monkeypatch) -> None:
    def mock(rig):
        p = rig.root / DATA / "rerun_exp09_o19" / "rows.jsonl"
        rows = [json.loads(ln) for ln in p.read_text().splitlines()]
        rows[0]["mock"] = True
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    record_holder = {}

    def restamp(r):  # keep data_sha256 honest so the mock rule is what refuses
        p = rig.root / DATA / "rerun_exp09_o19" / "rows.jsonl"
        r["data_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        record_holder["r"] = r

    _t39_move(rig, monkeypatch, rows_edit=mock, verdict_edit=restamp)
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
    entry = {"id": "x1", "kind": "prereg", "path": "p", "sha256": "s"}
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

    assert not (G.FINISH_OK & SIMULATION_FAILURE_FINISH_REASONS)
    assert all(not G.finish_ok(r) for r in SIMULATION_FAILURE_FINISH_REASONS)


def test_the_real_pass_table_names_the_o19_rows() -> None:
    table = json.loads((REPO / G.PASS_TABLE).read_text())
    assert table["exp10_verdict"]["rows"] == ["T1-1"] and table["exp09_verdict"]["rows"] == ["T3-9"]
    assert "MAINTAINED" not in table["exp09_verdict"]["targets"] or table["exp09_verdict"]["targets"]["MAINTAINED"] == [
        "PASS"
    ]
    assert table["exp09_verdict"]["targets"]["PARTIAL"] == ["PARTIAL"]
    assert not any(k.startswith(("exp53", "exp54", "exp60", "exp61", "exp62")) for k in table)  # 5b-2


def test_unknown_digests_never_match() -> None:
    assert G.unknown("unknown") and G.unknown("unknown: OSError") and G.unknown(None) and G.unknown("")
    assert not G.unknown("a" * 64)


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
    _t39_move(rig, monkeypatch, rows_edit=lambda rg: _edit_rows(rg, "09", lambda r: r.update(ts=1.0)),
              verdict_edit=_restamp(rig, "09"))  # fmt: skip
    failures, _ = rig.run()
    assert any("not after the previous status was set" in f for f in failures), failures


def test_code_not_on_main_is_refused(rig, monkeypatch) -> None:
    _t39_move(rig, monkeypatch, rows_edit=lambda rg: _edit_rows(rg, "09", lambda r: r["provenance"].update(
        executed_git_hash="f" * 40)), verdict_edit=_restamp(rig, "09"))  # fmt: skip
    failures, _ = rig.run()
    assert any("is not on main" in f for f in failures), failures


def test_an_o19_sim_whose_code_changed_is_refused(rig, monkeypatch) -> None:
    def changed(r):
        for sim in r.get("sims") or []:
            sim["code_changed_during_run"] = True

    _t39_move(rig, monkeypatch, rows_edit=lambda rg: _edit_rows(rg, "09", changed), verdict_edit=_restamp(rig, "09"))
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
    o19_attempt(rig, "10", first, monkeypatch=monkeypatch)
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


def _ctx(rig: Rig) -> G.Ctx:
    return G.Ctx(repo=G.Repo(rig.root), base=rig.base_sha, ref="HEAD", legacy={}, prereg={})


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
            "language_profile": "mistral-7b-instruct-v0.2"}  # fmt: skip
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
