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


def test_a_verdict_names_its_data_repo_relative_with_the_hash_of_the_bytes_it_judged(tmp_path: Path) -> None:
    rows = tmp_path / "docs" / "rows.jsonl"
    rows.parent.mkdir()
    rows.write_text('{"a": 1}\n')
    v = P.stamp_verdict({"verdict": "PASS"}, repo_root=tmp_path, kind="k", data=rows, scope={"campaign_id": "c"})
    assert v["record_kind"] == "verdict" and v["kind"] == "k" and v["scope"] == {"campaign_id": "c"}
    assert v["data"] == "docs/rows.jsonl" and v["data_sha256"] == _sha(rows)
    outside = P.stamp_verdict({}, repo_root=tmp_path / "docs", kind="k", data=tmp_path / "missing.jsonl", scope={})
    assert outside["data"] == str(tmp_path / "missing.jsonl") and outside["data_sha256"] is None


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
    assert {k: v["scope"][k] for k in scope} == scope
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
    assert report["provenance"]["harness_family"] == "in_process"
    assert report["verdict"] == "NO-VERDICT", "a mock row never yields a verdict"


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
    assert rows and all(r["record_kind"] == "harness_row" and r["mock"] is True for r in rows)


def test_the_exp49_scripted_arm_is_mock(tmp_path: Path, monkeypatch) -> None:
    exp49 = _load("m1b5a_exp49", "exp49/run_trials.py")
    monkeypatch.setattr(
        sys, "argv", ["run_trials.py", "--arm", "scripted", "--out", str(tmp_path), "--limit-trials", "1"]
    )
    assert exp49.main() == 0
    (row,) = _rows(tmp_path / "trials_scripted.jsonl")
    assert row["record_kind"] == "harness_row" and row["status"] == "ok" and row["mock"] is True


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
