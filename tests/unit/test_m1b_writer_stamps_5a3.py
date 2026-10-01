"""M1b PR 5a-3 -- the last writer fixes before the evidence gate.

One campaign per file (#1022): the survival harnesses default to a file named by the campaign, never a committed
legacy file, and every one refuses to append to a file that is unstamped or ran on another code tree. The Exp 53/54
verdict kind comes from what its runs ran under. Companions get a kind. The Exp 56 analyzer never renders a PASS
without its no-op kit. A harness row's time is its writer's, never a default.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import maxim

REPO = Path(maxim.__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
ORIENT = SCRIPTS / "orient_backbone"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ORIENT))
import _provenance as P  # noqa: E402
from survival_world import exp61_run as E61  # noqa: E402
from survival_world import exp62_run as E62  # noqa: E402
from survival_world import r3_run as R3  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prov() -> dict:
    return P.in_process_code_provenance(REPO, maxim.__file__)


# ── the append refusal ───────────────────────────────────────────────────


def test_a_file_takes_appends_only_from_its_own_code_tree(tmp_path: Path) -> None:
    prov = _prov()
    out = tmp_path / "rows.jsonl"
    assert P.append_refusal(out, prov) is None, "a new file"
    out.write_text(json.dumps({"record_kind": "harness_row", "provenance": prov}) + "\n")
    assert P.append_refusal(out, prov) is None, "a resume on the same tree"
    other = {**prov, "code_tree_sha256": "f" * 64}
    assert "another code tree" in P.append_refusal(out, other)
    out.write_text(json.dumps({"seed": 1}) + "\n")
    assert "unstamped" in P.append_refusal(out, prov), "a pre-M1b file never takes a stamped run"
    out.write_text("not json\n")
    assert "not JSON" in P.append_refusal(out, prov)


def test_the_default_rows_file_is_named_by_its_campaign() -> None:
    assert P.campaign_out_path("exp61_pairs", "c1") == "docs/experiments/data/exp61_pairs_c1.jsonl"
    assert R3.gauntlet_path_for("c1") == "docs/experiments/data/r3_gauntlet_c1.json"


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(write_experiment_results=False, allow_dirty=False, **kw)


def test_exp61_refuses_a_resume_of_nothing_and_an_append_onto_a_legacy_file(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "pairs.jsonl"
    assert E61._run(_ns(campaign_id="c1", resume=True, out=str(missing))) == 2
    assert "does not exist" in capsys.readouterr().out
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps({"kind": "receiver", "campaign_id": "old"}) + "\n")
    assert E61._run(_ns(campaign_id="c1", resume=False, out=str(legacy))) == 2
    assert "unstamped" in capsys.readouterr().out
    assert legacy.read_text().count("\n") == 1, "nothing appended"


def test_exp62_run_needs_its_campaigns_replay_row(tmp_path: Path, capsys) -> None:
    out = tmp_path / "rows.jsonl"
    assert E62.cmd_run(_ns(campaign_id="c1", out=str(out), resume=False)) == 2
    assert "no unrefused replay row for campaign c1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("cmd", "campaign_id", "resume", "why"),
    [
        ("bench", None, False, "need --campaign-id"),
        ("cal", None, True, "need --campaign-id"),
        ("cal", "c1", True, "does not exist"),
    ],
)
def test_r3_keys_cal_and_bench_by_campaign(cmd: str, campaign_id, resume: bool, why: str, capsys) -> None:
    args = _ns(cmd=cmd, campaign_id=campaign_id, resume=resume, out=None, gauntlet=None)
    assert R3._setup(args) == 2
    assert why in capsys.readouterr().out
    if campaign_id:
        assert args.gauntlet == R3.gauntlet_path_for(campaign_id), "bench reads its cal campaign's gauntlet"


@pytest.mark.parametrize("rel", ["exp58_run.py", "exp60_run.py", "exp61_run.py", "exp62_run.py", "r3_run.py"])
def test_every_survival_row_writer_refuses_an_unsafe_append(rel: str) -> None:
    tree = ast.parse((SCRIPTS / "survival_world" / rel).read_text())
    called = {getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "append_refusal" in called
    defaults = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    legacy = ("exp58_claim_b.jsonl", "exp60_trials.jsonl", "exp61_pairs.jsonl", "exp62_rows.jsonl", "r3_cal.jsonl")
    assert not any(n.value.endswith(legacy) for n in defaults), "no committed legacy file as a default --out"


@pytest.mark.parametrize("rel", ["exp58_run.py", "exp60_run.py"])
def test_exp58_and_exp60_need_an_explicit_out(rel: str, capsys) -> None:
    mod = _load(f"m1b5a3_{rel}", SCRIPTS / "survival_world" / rel)
    argv = ["--arm", "fear", "--rcon-password", "x"] if rel == "exp58_run.py" else ["run", "--arm", "fear"]
    with pytest.raises(SystemExit):
        mod.main(argv)
    required = [ln for ln in capsys.readouterr().err.splitlines() if "arguments are required" in ln]
    assert required and "--out" in required[0], "--out is named as missing, not merely shown in the usage"


# ── the Exp 53/54 verdict kind ───────────────────────────────────────────


def _exp53_records(tmp_path: Path, experiments: list[str | None]) -> Path:
    rows = []
    for rid, phase, experiment in zip(("A", "Q"), (1, 2), experiments):
        start = {"event": "start", "run_id": rid, "phase": phase, "only": None}
        if experiment is not None:
            start["experiment"] = experiment
        rows.append(start)
        for a in ("taught_seed42", "satiated_seed42", "no_feed_seed42"):
            arm = a.split("_seed")[0]
            rows.append({"event": "agent_load", "run_id": rid, "phase": phase, "agent": a})
            rows.append(
                {
                    "event": "trial" if phase == 2 else "probe",
                    "run_id": rid,
                    "phase": phase,
                    "agent": a,
                    "arm": arm,
                    "seed": 42,
                    "condition": "primary",
                    "exploratory": False,
                    "exploratory_agent": False,
                    "toward": arm == "taught",
                    "affordance": "turn_left",
                    "sign_rule_correct": True,
                    "target_az": -0.3,
                }
            )
            rows.append({"event": "agent_done", "run_id": rid, "phase": phase, "agent": a})
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.mark.parametrize(
    ("experiments", "kind"),
    [
        (["53_cross_context_readout"] * 2, "exp53_verdict"),
        (["54_nurture_reachy_body"] * 2, "exp54_verdict"),
        (["gate6_merged_gauntlet"] * 2, "gate6_exp53_verdict"),
        ([None, None], "exp53_unlabelled_verdict"),
        (["53_cross_context_readout", "54_nurture_reachy_body"], None),
        (["53_cross_context_readout", None], None),
        (["99_unknown"] * 2, None),
    ],
)
def test_the_verdict_kind_is_what_its_runs_ran_under(experiments, kind, tmp_path: Path, capsys) -> None:
    """Never a verdict-time flag: an Exp 54 run can no longer be stamped `exp53_verdict` (the counted kind)."""
    h = _load("m1b5a3_exp53", ORIENT / "exp53_cross_context_readout.py")
    records = _exp53_records(tmp_path, experiments)
    rc = h.main(["verdict", "--records", str(records)])
    verdict_file = tmp_path / "records_verdict.json"
    if kind is None:
        assert rc == 2 and not verdict_file.exists() and "REFUSED" in capsys.readouterr().out
    else:
        assert rc in (0, 1) and json.loads(verdict_file.read_text())["kind"] == kind


# ── companions and analyzers ─────────────────────────────────────────────


def test_the_exp56_analyzer_never_renders_a_pass_without_its_no_op_kit(tmp_path: Path, monkeypatch, capsys) -> None:
    mod = _load("m1b5a3_analyze_exp56", SCRIPTS / "analyze_exp56.py")
    passing = {"stats": {}, "gates": {"TRANSFERRED": True}, "problems": [], "verdict": "PASS", "constants": {}}
    monkeypatch.setattr(mod, "analyze", lambda rows, min_pairs: dict(passing, problems=[]))
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"arm": "taught", "mock": False}) + "\n")
    monkeypatch.setattr(sys, "argv", ["analyze_exp56.py", "--in", str(rows)])
    assert mod.main() == 4
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "NO-VERDICT" and any("no-op kit did not run" in p for p in report["problems"])


def test_the_r3_report_is_a_diagnosis_with_its_code(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "bench.jsonl"
    data.write_text("")
    out = tmp_path / "report.json"
    args = argparse.Namespace(data=str(data), json=str(out), campaign_id="c1", gauntlet=None, amended=False)
    assert R3._report(args) == 0
    rep = json.loads(out.read_text())
    assert rep["record_kind"] == "diagnosis" and rep["code_provenance"]["harness_family"] == "in_process"
    assert rep["r3_status"] == "INCOMPLETE" and rep["status"] == "ok", "R3's completion vs how the run ended"
    assert rep["_format_version"] == "1.1", "the rename is marked: a pre-1.1 report reads `status` as R3's own"


@pytest.mark.parametrize(
    ("rel", "stamp"),
    [
        ("orient_backbone/exp53_cross_context_readout.py", "stamp_harness_header"),  # manifest + targets
        ("orient_backbone/gate6_merged_gauntlet.py", "stamp_diagnosis"),
        ("survival_world/r3_run.py", "stamp_diagnosis"),  # gauntlet + report
    ],
)
def test_the_companion_writers_stamp_a_kind(rel: str, stamp: str) -> None:
    """Live-rig / archive-bound writers: structural (the R3 report is driven for real above)."""
    tree = ast.parse((SCRIPTS / rel).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) == stamp
    ]
    assert len(calls) >= (2 if rel != "orient_backbone/gate6_merged_gauntlet.py" else 1)


def test_an_unknown_code_tree_matches_no_file_not_even_its_own(tmp_path: Path) -> None:
    unknown = {**_prov(), "code_tree_sha256": "unknown"}
    assert "unknown" in P.append_refusal(tmp_path / "new.jsonl", unknown)
    assert "unknown" in P.append_refusal(tmp_path / "new.jsonl", {})


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "c 1"])
def test_a_campaign_id_names_a_plain_path_segment(bad: str) -> None:
    with pytest.raises(ValueError, match="campaign id"):
        P.campaign_out_path("exp61_pairs", bad)
    with pytest.raises(ValueError, match="campaign id"):
        R3.gauntlet_path_for(bad)


def test_a_non_dict_line_never_counts_as_a_replay_row(tmp_path: Path) -> None:
    out = tmp_path / "rows.jsonl"
    out.write_text("[1]\n" + json.dumps({"kind": "replay", "campaign_id": "c1"}) + "\n")
    assert E62._has_replay_row(out, "c1") and not E62._has_replay_row(out, "c2")


def test_the_exp57_ladder_stamps_rows_with_their_time(tmp_path: Path) -> None:
    """The real stamp order (a tiny ScriptedBridge mock, ~2 s): `ts` is set before the stamp requires it."""
    import os
    import subprocess

    out = tmp_path / "rows.jsonl"
    args = [sys.executable, str(SCRIPTS / "exp57/run_ladder.py"), "--mock", "--out", str(out), "--allow-dirty"]
    args += [
        "--rungs",
        "1",
        "--cohorts",
        "1",
        "--conditions",
        "creche",
        "--k-max",
        "2",
        "--workdir",
        str(tmp_path / "w"),
    ]
    env = dict(os.environ, MAXIM_OPERANT_ONLY_CREDIT="1", PYTHONPATH=str(REPO / "src"))
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert rows and all(r["record_kind"] == "harness_row" and isinstance(r["ts"], float) for r in rows)


def test_a_refused_replay_row_does_not_open_a_campaign(tmp_path: Path) -> None:
    """The verdict counts only an unrefused replay row, so `run` must not start on a refused one."""
    out = tmp_path / "rows.jsonl"
    out.write_text(json.dumps({"kind": "replay", "campaign_id": "c1", "refusal": "geometry not derivable"}) + "\n")
    assert not E62._has_replay_row(out, "c1")


def test_a_bad_campaign_id_is_a_refusal_not_a_traceback(tmp_path: Path, capsys) -> None:
    assert E61._run(_ns(campaign_id="../x", resume=False, out=None)) == 2
    assert R3._setup(_ns(cmd="cal", campaign_id="../x", resume=False, out=None, gauntlet=None)) == 2
    assert "campaign id" in capsys.readouterr().out


def test_a_same_tree_resume_appends_to_rows_its_harness_wrote(tmp_path: Path) -> None:
    """Positive control: rows written through the real write path, with the run's provenance, take a resume."""
    from types import SimpleNamespace

    prov = _prov()
    out = tmp_path / "pairs.jsonl"
    E61._Campaign.write(SimpleNamespace(out_path=out), {"kind": "receiver", "ts": 1.0, "provenance": prov})
    assert P.append_refusal(out, prov) is None
