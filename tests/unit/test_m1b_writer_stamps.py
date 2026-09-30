"""M1b PR 5a -- every experiment record says what it is, in the shape the evidence gate (PR 5b) will read.

A harness row carries ``record_kind``, ``status`` (a refusal is ``failed``) and an explicit ``mock``; a
verdict carries its ``kind``, the rows file it judged (repo-relative, with its sha256) and the scope that
selects its rows; the provenance block names its ``harness_family``, stamped by ``_provenance`` itself so a
writer cannot choose how the gate judges it. Each test reads a writer's REAL output (its write path, its CLI
or its report), never a hand-built dict.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import maxim

REPO = Path(maxim.__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
DATA = REPO / "docs" / "experiments" / "data"
sys.path.insert(0, str(SCRIPTS))
import _provenance as P  # noqa: E402
import lint_harness_provenance as L  # noqa: E402
from survival_world import exp60_run as E60  # noqa: E402
from survival_world import exp61_run as E61  # noqa: E402
from survival_world import exp62_run as E62  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── the two stamps ───────────────────────────────────────────────────────


def test_a_harness_row_is_failed_when_refused_or_already_failed_and_always_says_mock() -> None:
    assert P.stamp_harness_row({"refusal": None}, mock=False) == {
        "refusal": None,
        "record_kind": "harness_row",
        "status": "ok",
        "mock": False,
    }
    assert P.stamp_harness_row({"refusal": "no clean donor"}, mock=False)["status"] == "failed"
    assert P.stamp_harness_row({"status": "failed"}, mock=True)["status"] == "failed"
    assert P.stamp_harness_row({"status": "ok"}, mock=True)["mock"] is True
    # the verdicts' own reading of a refusal: an exception with an empty message is still one
    assert P.stamp_harness_row({"refusal": ""}, mock=False)["status"] == "failed"
    with pytest.raises(ValueError, match="neither"):
        P.stamp_harness_row({"status": "incomplete"}, mock=False)  # never silently rewritten to ok


def test_a_verdict_names_its_data_repo_relative_with_the_hash_of_the_bytes_it_judged(tmp_path: Path) -> None:
    rows = tmp_path / "docs" / "rows.jsonl"
    rows.parent.mkdir()
    rows.write_text('{"a": 1}\n')
    judged = rows.read_bytes()
    rows.write_text('{"a": 1}\n{"appended": "after the verdict read"}\n')
    v = P.stamp_verdict(
        {"verdict": "PASS"},
        repo_root=tmp_path,
        kind="k",
        data=rows,
        data_bytes=judged,
        scope={"campaign_id": "c"},
        mock=False,
    )
    assert v["record_kind"] == "verdict" and v["kind"] == "k" and v["scope"] == {"campaign_id": "c"}
    assert v["data"] == "docs/rows.jsonl" and v["data_sha256"] == hashlib.sha256(judged).hexdigest() != _sha(rows)
    outside = P.stamp_verdict(
        {},
        repo_root=tmp_path / "docs",
        kind="k",
        data=tmp_path / "x.jsonl",
        data_bytes=b"",
        scope={"all_rows": True},
        mock=True,
    )
    assert outside["data"] == str(tmp_path / "x.jsonl") and outside["mock"] is True
    # a verdict over a smoke, or over rows that do not say, is a smoke (M1b PR 5a-2: unknown is mock)
    assert P.any_not_stamped_real([{"mock": False}]) is False
    assert P.any_not_stamped_real([{"mock": False}, {}]) is True and P.any_not_stamped_real([{"mock": True}])
    assert P.any_not_stamped_real([]) is True, "no rows is not a real run"


@pytest.mark.parametrize("scope", [{}, {"campaign_id": None}, {"run_ids": None}, {"run_ids": []}, {"campaign_id": ""}])
def test_a_verdict_scope_is_never_empty_or_a_none_selector(scope: dict, tmp_path: Path) -> None:
    """ "Every row" must be said ({"all_rows": True}), never be what a forgotten selector reads as."""
    with pytest.raises(ValueError, match="all_rows"):
        P.stamp_verdict({}, repo_root=tmp_path, kind="k", data=tmp_path / "r", data_bytes=b"", scope=scope, mock=False)


def test_the_provenance_block_names_its_family(monkeypatch) -> None:
    assert P.in_process_code_provenance(REPO, maxim.__file__)["harness_family"] == "in_process"
    monkeypatch.setattr(P, "resolved_maxim_file", lambda binary, timeout=60.0: maxim.__file__)
    assert P.executed_code_provenance(REPO, sys.executable)["harness_family"] == "spawning"


# ── survival writers: the write path each campaign appends through ───────


@pytest.mark.parametrize(
    "write",
    [
        lambda out, row: E60._write_row(out, row),
        lambda out, row: E61._Campaign.write(SimpleNamespace(out_path=out), row),
        lambda out, row: E62.Exp62Campaign.write(SimpleNamespace(out_path=out), row),
    ],
    ids=["exp60", "exp61", "exp62"],
)
def test_survival_rows_are_stamped_where_they_are_written(write, tmp_path: Path) -> None:
    out = tmp_path / "rows.jsonl"
    write(out, {"seed": 1, "refusal": None})
    write(out, {"seed": 2, "refusal": "InstrumentError: no surface"})
    assert [(r["seed"], r["record_kind"], r["status"], r["mock"]) for r in _rows(out)] == [
        (1, "harness_row", "ok", False),
        (2, "harness_row", "failed", False),
    ]


# ── verdict CLIs over the committed records ──────────────────────────────


def _committed(name: str) -> dict:
    return json.loads((DATA / name).read_text())


@pytest.mark.parametrize(
    ("module", "verdict_file", "kind", "selector"),
    [
        (E60, "exp60_verdict.json", "exp60_verdict", "run_ids"),
        (E61, "exp61_verdict.json", "exp61_verdict", "campaign_id"),
        (E62, "exp62_verdict.json", "exp62_verdict", "campaign_id"),
    ],
    ids=["exp60", "exp61", "exp62"],
)
def test_a_survival_verdict_binds_its_rows_file_scope_and_code(
    module, verdict_file: str, kind: str, selector: str, tmp_path: Path, monkeypatch
) -> None:
    committed = _committed(verdict_file)
    data = committed["data"]  # repo-relative, as the committed verdict names it
    if selector == "run_ids":
        select = [a for rid in committed["run_ids"] for a in ("--run-id", rid)]
        scope = {"run_ids": committed["run_ids"]}
    else:
        select = ["--campaign-id", committed["campaign_id"]]
        scope = {"campaign_id": committed["campaign_id"]}
    monkeypatch.chdir(REPO)
    out = tmp_path / "verdict.json"
    assert module.main(["verdict", "--data", data, *select, "--json", str(out)]) == 0
    v = json.loads(out.read_text())
    assert v["verdict"] == committed["verdict"] == "EARNED"
    assert v["record_kind"] == "verdict" and v["kind"] == kind
    assert v["data"] == data and v["data_sha256"] == _sha(REPO / data)
    assert v["scope"] == scope, "the scope names every row the verdict read (a campaign: all of its kinds)"
    # the committed rows predate the stamps: a re-verdict over them is mock (unknown is mock, M1b PR 5a-2)
    assert v["mock"] is True
    assert v["provenance"]["harness_family"] == "in_process"


_EXP57_ROW = {"rung": 1, "condition": "creche", "cohort": 0, "tau": 3, "t": 6, "k_max": 6, "coverage": 0.5}


@pytest.mark.parametrize(
    ("analyzer", "kind", "row"),
    [
        ("analyze_exp56.py", "exp56_verdict", {"arm": "taught"}),
        ("analyze_exp57.py", "exp57_verdict", _EXP57_ROW),
    ],
)
def test_the_exp56_and_exp57_analyzers_stamp_their_verdicts(
    analyzer: str, kind: str, row: dict, tmp_path: Path, monkeypatch, capsys
) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({**row, "mock": True}) + "\n")
    mod = _load(f"m1b5a_{kind}", analyzer)
    monkeypatch.setattr(sys, "argv", [analyzer, "--in", str(rows)])
    mod.main()
    report = json.loads(capsys.readouterr().out.split("\nVERDICT", 1)[0])
    assert report["record_kind"] == "verdict" and report["kind"] == kind
    assert report["data"] == str(rows) and report["data_sha256"] == _sha(rows)  # outside the repo: named as given
    assert report["provenance"]["harness_family"] == "in_process" and report["scope"] == {"all_rows": True}
    assert report["verdict"] == "NO-VERDICT", "a mock row never yields a verdict"
    assert report["mock"] is True, "a verdict over a smoke is a smoke (M1b PR 5a-2)"


@pytest.mark.parametrize("analyzer", ["analyze_exp56.py", "analyze_exp57.py"])
def test_an_analyzer_refuses_with_3_when_its_code_cannot_be_established(
    analyzer: str, tmp_path: Path, monkeypatch
) -> None:
    """3 is the house refusal; 1 is FAIL, which a provenance failure must never read as."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(_EXP57_ROW) + "\n")
    mod = _load(f"m1b5a_refuse_{analyzer}", analyzer)

    def not_this_repo(*a, **k):
        raise P.ProvenanceError("imported maxim is not this repo")

    monkeypatch.setattr(P, "in_process_code_provenance", not_this_repo)
    monkeypatch.setattr(sys, "argv", [analyzer, "--in", str(rows)])
    assert mod.main() == 3


# ── smokes say they are smokes ───────────────────────────────────────────


def test_an_exp44_dry_run_manifest_is_mock(tmp_path: Path, monkeypatch) -> None:
    exp44 = _load("m1b5a_exp44", "exp44/campaign.py")
    cfg = json.loads((SCRIPTS / "exp44" / "campaign_44b.json").read_text())
    work = tmp_path / "campaign"
    argv = ["campaign.py", "--config", str(SCRIPTS / "exp44" / "campaign_44b.json"), "--workdir", str(work)]
    argv += ["--dry-run", "--arms", cfg["arms"][0]["name"], "--seeds", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    assert exp44.main() == 0
    rows = _rows(work / "manifest.jsonl")
    assert rows and all(r["mock"] is True for r in rows)
    # the opening row is a header (config + provenance, never a run: M1b PR 5a-2), the rest are harness rows
    assert rows[0]["record_kind"] == "harness_header" and "status" not in rows[0]
    assert all(r["record_kind"] == "harness_row" for r in rows[1:])


def test_the_exp49_scripted_arm_is_mock(tmp_path: Path, monkeypatch) -> None:
    exp49 = _load("m1b5a_exp49", "exp49/run_trials.py")
    monkeypatch.setattr(
        sys, "argv", ["run_trials.py", "--arm", "scripted", "--out", str(tmp_path), "--limit-trials", "1"]
    )
    assert exp49.main() == 0
    (row,) = _rows(tmp_path / "trials_scripted.jsonl")
    assert row["record_kind"] == "harness_row" and row["status"] == "ok" and row["mock"] is True


def test_an_exp49_trial_whose_maxim_crashed_is_failed(tmp_path: Path, monkeypatch) -> None:
    exp49 = _load("m1b5a_exp49_spawn", "exp49/run_trials.py")

    class Exited:
        def __init__(self, code: int) -> None:
            self.returncode = code

        def poll(self):
            return self.returncode

    monkeypatch.setattr(exp49._provenance, "executed_code_provenance", lambda *a, **k: {})
    monkeypatch.setattr(exp49.time, "sleep", lambda s: None)
    for code, status in ((1, "failed"), (0, "ok")):
        monkeypatch.setattr(exp49.subprocess, "Popen", lambda *a, _c=code, **k: Exited(_c))
        rec = exp49.run_spawned_trial("B", 30.0, 1, tmp_path / f"t{code}", "maxim")
        assert rec["metrics"]["end_reason"] == f"process_exited_{code}"
        assert P.stamp_harness_row(rec, mock=False)["status"] == status


def test_an_exp37_failed_row_says_whether_it_was_a_smoke(tmp_path: Path, monkeypatch) -> None:
    exp37 = _load("m1b5a_exp37", "benchmark_cross_session.py")

    def fails(**kwargs):
        raise exp37._provenance.SimRunFailed("no report", sims=[])

    monkeypatch.setattr(exp37, "run_one_sim", fails)
    out = tmp_path / "37.jsonl"
    with pytest.raises(exp37._provenance.SimRunFailed):
        exp37.run_benchmark(
            out_path=out,
            workdir=tmp_path / "work",
            arms=("A",),
            scenarios=("fire_pit",),
            trials=1,
            model="claude-sonnet",
            cost_cap=100.0,
            max_turns=3,
            seed_base=42,
            mock=True,
        )
    (row,) = _rows(out)
    assert row["status"] == "failed" and row["mock"] is True


# ── the lint ─────────────────────────────────────────────────────────────

_SPAWNER = """import subprocess
assert_repo_interpreter(ROOT, b)
prov = executed_code_provenance(ROOT, resolve(), out_path=out, allow_dirty=a)
run_id = harness_run_id()
env["MAXIM_HARNESS_RUN_ID"] = run_id
subprocess.run(["maxim", "--sim", goal], env=env)
session, report = spawn_evidence(home, run_id, before, returncode=0)
row = {"record_kind": "harness_row", "sims": [sim_evidence(session, report)]}
"""


def _tree(tmp_path: Path, harness: str) -> Path:
    (tmp_path / "scripts/orient_backbone").mkdir(parents=True)
    (tmp_path / "scripts/orient_backbone/live_common.py").write_text(
        "class JsonlLog:\n    def __init__(self, path):\n        preflight_gated_record_or_exit(ROOT, path)\n"
    )
    (tmp_path / "scripts/bench.py").write_text(harness)
    return tmp_path


def test_the_lint_refuses_a_sim_spawner_that_claims_the_in_process_family(tmp_path: Path) -> None:
    assert L.lint(_tree(tmp_path, _SPAWNER)) == []
    claims = _SPAWNER + "row['provenance'] = in_process_code_provenance(ROOT, maxim.__file__)\n"
    (fail,) = L.lint(_tree(tmp_path / "b", claims))
    assert fail.startswith("scripts/bench.py") and "in_process_code_provenance" in fail


def test_the_lint_refuses_in_process_provenance_in_any_maxim_spawner(tmp_path: Path) -> None:
    """The Exp 49 shape: `maxim --mode live`, no `--sim`, still judged by the runtime it spawned."""
    live = """import subprocess
assert_repo_interpreter(ROOT, b)
prov = executed_code_provenance(ROOT, resolve(), out_path=out, allow_dirty=a)
subprocess.Popen(["maxim", "--mode", "live"])
row = {"record_kind": "harness_row"}
"""
    assert L.lint(_tree(tmp_path, live)) == []
    (fail,) = L.lint(_tree(tmp_path / "b", live + "p = in_process_code_provenance(ROOT, f)\n"))
    assert "in_process_code_provenance" in fail


def test_the_lint_refuses_a_writer_that_names_the_harness_family(tmp_path: Path) -> None:
    root = _tree(tmp_path, _SPAWNER)
    (root / "scripts/analysis.py").write_text('prov["harness_family"] = "in_process"\n')
    (fail,) = L.lint(root)
    assert fail.startswith("scripts/analysis.py") and "harness_family" in fail


@pytest.mark.parametrize(
    "rel",
    ["exp56/run_campaign.py", "exp56/instrument_check.py", "exp57/run_ladder.py", "exp57/instrument_check.py"],
)
def test_the_exp56_and_exp57_harnesses_stamp_in_process_provenance(rel: str) -> None:
    """They IMPORT maxim and never spawn it, so their code is the imported package, not whatever the `maxim`
    console script resolves. Structural: their mock runs take 30 s to 6 min (verified live for PR 5a)."""
    import ast

    calls = L._names_called(ast.parse((SCRIPTS / rel).read_text()))
    assert "in_process_code_provenance" in calls and "executed_code_provenance" not in calls
    stamp = "stamp_instrument_check" if rel.endswith("instrument_check.py") else "stamp_harness_row"
    assert stamp in calls, "its records are stamped where they are written"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("rel", "out_name"), [("exp56/run_campaign.py", "rows.jsonl"), ("exp56/instrument_check.py", "phase0.json")]
)
def test_the_exp56_mock_runs_write_stamped_records(rel: str, out_name: str, tmp_path: Path) -> None:
    """The real output of the ScriptedBridge smokes (30-45 s each, so the nightly slow lane)."""
    import os
    import subprocess

    out = tmp_path / out_name
    args = [sys.executable, str(SCRIPTS / rel), "--mock", "--out", str(out), "--allow-dirty"]
    if rel.endswith("run_campaign.py"):
        args += ["--pairs", "1", "--workdir", str(tmp_path / "work")]
    env = dict(os.environ, MAXIM_OPERANT_ONLY_CREDIT="1", PYTHONPATH=str(REPO / "src"))
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    records = _rows(out) if out_name.endswith(".jsonl") else [json.loads(out.read_text())]
    assert records
    kind = "instrument_check" if rel.endswith("instrument_check.py") else "harness_row"
    for r in records:
        assert (r["record_kind"], r["status"], r["mock"]) == (kind, "ok", True)
        if kind == "instrument_check":
            assert r["pass"] is True and r["pass"] == r["all_pass"]  # no pass-relevant flag (settle_s exempt)
        assert r["provenance"]["harness_family"] == "in_process"
