"""#1003 (M1b PR 2) -- the harnesses that spawn ``maxim --sim`` write rows a gate can judge on their own.

Each row names the session its spawn wrote (found by the harness run id, never the newest directory),
echoes that report's evidence, and carries the harness's own provenance block (Exp 37 and Exp 41 used to
discard it). A failed run is a row too, never a silently dropped trial, and readers exclude it; a legacy row
(no ``status``) is still a trial. The lint keeps it that way for every sim-spawning harness.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import maxim

REPO = Path(maxim.__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
DATA = REPO / "docs" / "experiments" / "data"
sys.path.insert(0, str(SCRIPTS))
import _provenance  # noqa: E402  (exceptions only; a harness's own module is `harness._provenance`)
import lint_harness_provenance as L  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp41():
    return _load("exp41_harness_1003", "benchmark_exp41_exploration.py")


@pytest.fixture(scope="module")
def exp42():
    return _load("exp42_harness_1003", "benchmark_exp42_preference.py")


@pytest.fixture(scope="module")
def exp37():
    return _load("exp37_harness_1003", "benchmark_cross_session.py")


@pytest.fixture(scope="module")
def exp44():
    return _load("exp44_campaign_1003", "exp44/campaign.py")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fake_sim(*, returncode: int = 0, run_id: str | None = None, finish_reason: str = "completed"):
    """A stand-in for ``subprocess.run`` of ``maxim --sim``: writes a session into the spawn's
    MAXIM_DATA_HOME the way a sim does, stamping the run id it was handed (or ``run_id`` to impersonate
    another harness)."""
    seen: dict = {}

    def run(cmd, env=None, **kwargs):
        home = Path(env["MAXIM_DATA_HOME"])
        seen["env"] = dict(env)
        session = (
            home
            / "sim_reports"
            / f"sess_{len(list((home / 'sim_reports').glob('*'))) if (home / 'sim_reports').exists() else 0}"
        )
        session.mkdir(parents=True)
        stamped = run_id if run_id is not None else env["MAXIM_HARNESS_RUN_ID"]
        report = {"record_kind": "sim_report", "finish_reason": finish_reason, "ts": 1.0}
        report["provenance"] = {"harness_run_id": stamped, "executed_git_hash": "c" * 40}
        (session / "report.json").write_text(json.dumps(report))
        (session / "actions.jsonl").write_text(
            '{"_record_kind": "header"}\n{"tool": "hearth_warm_self"}\n{"tool": "blanket_wrap"}\n'
        )
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="")

    return run, seen


# ── a spawn reads back ITS report ────────────────────────────────────────


def test_exp41_reads_back_the_report_its_spawn_wrote(exp41, tmp_path: Path, monkeypatch) -> None:
    stale = tmp_path / "A_dec_seed42" / "sim_reports" / "from_an_earlier_launch"
    stale.mkdir(parents=True)  # the home is reused across launches unless wiped
    run, seen = _fake_sim()
    monkeypatch.setattr(exp41.subprocess, "run", run)
    tools, evidence = exp41._run_real(
        "A_dec", 42, model="m", embodiment="e", max_turns=3, explore_weight=1.5, timeout_s=5, workdir=tmp_path
    )
    assert tools == ["hearth_warm_self", "blanket_wrap"]
    run_id = exp41._provenance.harness_run_id()  # the module the harness imported (others may reload it)
    assert seen["env"]["MAXIM_HARNESS_RUN_ID"] == run_id
    assert evidence["harness_run_id"] == run_id and evidence["finish_reason"] == "completed"
    assert not stale.exists(), "each run starts from a fresh home (#1003; the Exp 42 rule)"


@pytest.mark.parametrize("harness", ["exp41", "exp42"])
def test_a_spawn_that_cannot_be_established_fails(harness, request, tmp_path: Path, monkeypatch) -> None:
    mod = request.getfixturevalue(harness)
    arm = "A_dec" if harness == "exp41" else "cradle_pref_a"
    kwargs = dict(model="m", embodiment="e", max_turns=3, explore_weight=1.5, timeout_s=5, workdir=tmp_path)
    run, _ = _fake_sim(returncode=4, finish_reason="planning_failed")
    monkeypatch.setattr(mod.subprocess, "run", run)
    with pytest.raises(mod._provenance.SimRunFailed) as err:
        mod._run_real(arm, 42, **kwargs)
    assert err.value.sims[0]["finish_reason"] == "planning_failed"
    run, _ = _fake_sim(run_id="another-harness")
    monkeypatch.setattr(mod.subprocess, "run", run)
    with pytest.raises(mod._provenance.OwnReportError):
        mod._run_real(arm, 43, **kwargs)


# ── a failed run is a row, and readers exclude it ────────────────────────


def test_exp41_records_a_failed_run_and_resumes_it(exp41, tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "41.jsonl"

    def boom(arm, seed):
        raise _provenance.SimRunFailed("sub-sim exited 4", sims=[{"session_id": "s", "finish_reason": "error"}])

    monkeypatch.setattr(exp41, "_mock_tools", boom)
    kwargs = dict(
        arms=("A_dec",), trials=1, seed_base=42, out_path=out, mock=True, model="m", embodiment="e", max_turns=3
    )
    exp41.run_benchmark(**kwargs, explore_weight=1.5, timeout_s=5, resume=False)
    (row,) = _rows(out)
    assert row["status"] == "failed" and row["record_kind"] == "harness_row" and row["arm"] == "A_dec"
    assert row["sims"] == [{"session_id": "s", "finish_reason": "error"}]
    assert exp41._existing_keys(out) == set(), "a failed run is not done: a resume re-runs it"
    monkeypatch.undo()
    exp41.run_benchmark(**kwargs, explore_weight=1.5, timeout_s=5, resume=True)
    rows = _rows(out)
    assert [r["status"] for r in rows] == ["failed", "ok"]
    assert rows[1]["record_kind"] == "harness_row" and rows[1]["harness_schema"] == 2 and "provenance" in rows[1]
    analyzer = _load("exp41_analyzer_1003", "analyze_exp41_exploration.py")
    assert [r["status"] for r in analyzer.load_records(out)] == ["ok"]
    assert analyzer.count_failed_runs(out) == 1


@pytest.mark.parametrize(
    "analyzer_rel, data",
    [
        ("analyze_exp41_exploration.py", "41_results.jsonl"),
        ("analyze_exp42_preference.py", "42_results.jsonl"),
    ],
)
def test_committed_legacy_rows_are_all_still_trials(analyzer_rel: str, data: str) -> None:
    """R1: every committed row predates `status`; excluding failed rows must not drop one of them."""
    analyzer = _load(f"analyzer_{data}", analyzer_rel)
    path = DATA / data
    arm_rows = [r for r in _rows(path) if r.get("arm") in analyzer.ARMS]
    assert arm_rows and all("status" not in r for r in arm_rows)
    assert len(analyzer.load_records(path)) == len(arm_rows)
    assert analyzer.count_failed_runs(path) == 0


def test_the_exp37_analyzer_still_reads_its_committed_file() -> None:
    analyzer = _load("exp37_analyzer_1003", "analyze_exp37.py")
    path = DATA / "37_results.jsonl"
    rows = _rows(path)
    assert all("status" not in r for r in rows)
    assert len(analyzer.load_records(path, expected_schema="1.0")) == len(rows)


# ── Exp 37: an abort is on the record, and the resume retries it ─────────


def test_exp37_records_the_failed_sim_before_aborting_and_retries_it(exp37, tmp_path: Path, monkeypatch) -> None:
    out, work = tmp_path / "37.jsonl", tmp_path / "work"
    real = exp37.run_one_sim

    def b_fails(**kwargs):
        if kwargs.get("resume_session") is not None:
            raise exp37._provenance.SimRunFailed("--resume-sim did not load", sims=[{"session_id": "b"}])
        return real(**kwargs)

    kwargs = dict(
        out_path=out,
        workdir=work,
        arms=("A", "B"),
        scenarios=("fire_pit",),
        trials=1,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=3,
        seed_base=42,
        mock=True,
    )
    monkeypatch.setattr(exp37, "run_one_sim", b_fails)
    with pytest.raises(exp37._provenance.SimRunFailed):
        exp37.run_benchmark(**kwargs)
    first = _rows(out)
    assert [(r["arm"], r["status"]) for r in first] == [("A", "ok"), ("B", "failed")]
    monkeypatch.setattr(exp37, "run_one_sim", real)
    exp37.run_benchmark(**kwargs, resume=True)
    rows = _rows(out)
    assert [(r["arm"], r["status"]) for r in rows] == [("B", "failed"), ("A", "ok"), ("B", "ok")]
    a, b = rows[1], rows[2]
    assert a["sims"][0]["session_id"] == a["session_id"] and a["depends_on"] == []
    assert a["mock"] is True and b["mock"] is True, "a synthetic report's rows say so"
    assert [d["session_id"] for d in b["depends_on"]] == [a["session_id"]], "B inherits A's home"
    assert b["sims"][0]["resume"]["resume_loaded"] is True
    analyzer = _load("exp37_analyzer_1003b", "analyze_exp37.py")
    assert [r["arm"] for r in analyzer.load_records(out, expected_schema="1.0")] == ["A", "B"]


def test_exp37_refuses_a_resume_that_did_not_load(exp37, tmp_path: Path) -> None:
    home = tmp_path / "home"
    session = home / "sim_reports" / "s1"
    session.mkdir(parents=True)
    report = {"provenance": {"harness_run_id": "rid", "resume": {"resume_loaded": False, "requested": "p"}}}
    (session / "report.json").write_text(json.dumps(report))
    with pytest.raises(exp37._provenance.SimRunFailed, match="did not load"):
        exp37._own_session(home, "rid", set(), returncode=0, resume_session="p")
    assert exp37._own_session(home, "rid", set(), returncode=0, resume_session=None).session_id == "s1"


# ── Exp 44: a stage row names its session or says why not ───────────────


def test_exp44_stage_evidence(exp44, tmp_path: Path) -> None:
    home = tmp_path / "seed42"
    (home / "sim_reports" / "prior").mkdir(parents=True)
    (home / "sim_reports" / "prior" / "report.json").write_text(json.dumps({"provenance": {}}))
    session, fields = exp44._own_session(home, {"prior"}, 0)
    assert session is None and fields["failure"].startswith("OwnReportError")
    assert [d["session_id"] for d in fields["depends_on"]] == ["prior"]
    mine = home / "sim_reports" / "mine"
    mine.mkdir()
    run_id = exp44._provenance.harness_run_id()  # the module the campaign imported
    (mine / "report.json").write_text(json.dumps({"provenance": {"harness_run_id": run_id}}))
    session, fields = exp44._own_session(home, {"prior"}, 0)
    assert session == mine and fields["sims"][0]["session_id"] == "mine"


# ── the lint ─────────────────────────────────────────────────────────────

_WRITER = "class JsonlLog:\n    def __init__(self, path):\n        preflight_gated_record_or_exit(ROOT, path)\n"
_GOOD = """import subprocess
assert_repo_interpreter(ROOT, b)
prov = executed_code_provenance(ROOT, resolve(), out_path=out, allow_dirty=a)
run_id = harness_run_id()
env["MAXIM_HARNESS_RUN_ID"] = run_id
subprocess.run(["maxim", "--sim", goal], env=env)
session, report = spawn_evidence(home, run_id, before, returncode=0)
row = {"record_kind": "harness_row", "sims": [sim_evidence(session, report)]}
"""
_BREAKS = {
    "preflight_only": (
        "prov = executed_code_provenance(ROOT, resolve(), out_path=out, allow_dirty=a)",
        "preflight_gated_record_or_exit(ROOT, out)",
    ),
    "no_record_kind": ('"record_kind": "harness_row", ', ""),
    "no_env": ('env["MAXIM_HARNESS_RUN_ID"] = run_id', "pass"),
    "no_mint": ("run_id = harness_run_id()", "run_id = 'x'"),
    "newest_dir": ("session, report = spawn_evidence(home, run_id, before, returncode=0)", "session = newest(home)"),
    "no_echo": ("sim_evidence(session, report)", "session"),
}


def _tree(tmp_path: Path, harness: str) -> Path:
    (tmp_path / "scripts/orient_backbone").mkdir(parents=True)
    (tmp_path / "scripts/orient_backbone/live_common.py").write_text(_WRITER)
    (tmp_path / "scripts/bench.py").write_text(harness)
    return tmp_path


def test_the_lint_accepts_a_harness_that_binds_its_sims(tmp_path: Path) -> None:
    assert L.lint(_tree(tmp_path, _GOOD)) == []


@pytest.mark.parametrize("case", sorted(_BREAKS))
def test_the_lint_catches_each_missing_piece(tmp_path: Path, case: str) -> None:
    old, new = _BREAKS[case]
    assert old in _GOOD
    fails = L.lint(_tree(tmp_path, _GOOD.replace(old, new)))
    assert len(fails) == 1 and fails[0].startswith("scripts/bench.py"), fails


def test_exp42_records_a_failed_run_and_resumes_it(exp42, tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "42.jsonl"

    def boom(arm, seed):
        raise _provenance.SimRunFailed("sub-sim exited 4", sims=[])

    monkeypatch.setattr(exp42, "_mock_tools", boom)
    kwargs = dict(
        arms=("cradle_pref_a",),
        trials=1,
        seed_base=42,
        out_path=out,
        mock=True,
        model="m",
        embodiment="e",
        max_turns=3,
        explore_weight=1.5,
        timeout_s=5,
        workdir=str(tmp_path / "w"),
    )
    exp42.run_benchmark(**kwargs, resume=False)
    assert [r["status"] for r in _rows(out)] == ["failed"] and exp42._existing_keys(out) == set()
    monkeypatch.undo()
    exp42.run_benchmark(**kwargs, resume=True)
    rows = _rows(out)
    assert [r["status"] for r in rows] == ["failed", "ok"]
    assert rows[1]["provenance"] == {} and "executed_git_hash" not in rows[1], "provenance is nested, not flat"
    analyzer = _load("exp42_analyzer_1003", "analyze_exp42_preference.py")
    assert len(analyzer.load_records(out)) == 1 and analyzer.count_failed_runs(out) == 1


def test_cradle_records_a_failed_run_and_its_analyzer_skips_it(tmp_path: Path, monkeypatch) -> None:
    cradle = _load("cradle_harness_1003", "benchmark_cradle_mother.py")
    out = tmp_path / "cradle.jsonl"
    real = cradle._mock_fade

    def fade(arm, seed):
        if arm == "taught" and seed == 43:
            raise RuntimeError("incomplete fade — sub-sim ended early")
        return real(arm, seed)

    monkeypatch.setattr(cradle, "_mock_fade", fade)
    argv = [
        "x",
        "--mock",
        "--arms",
        "taught,no_feed",
        "--trials",
        "2",
        "--out",
        str(out),
        "--workdir",
        str(tmp_path / "w"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert cradle.main() == 0
    rows = _rows(out)
    statuses = [(r["arm"], r["seed"], r["status"]) for r in rows]
    assert statuses == [
        ("taught", 42, "ok"),
        ("taught", 43, "failed"),
        ("no_feed", 42, "ok"),
        ("no_feed", 43, "ok"),
    ]
    assert all(r["record_kind"] == "harness_row" and "provenance" in r for r in rows)
    assert "executed_git_hash" not in rows[0], "provenance is nested, not flat"
    only_ok = tmp_path / "only_ok.jsonl"
    only_ok.write_text("".join(json.dumps(r) + "\n" for r in rows if r["status"] == "ok"))
    analyzer = SCRIPTS / "analyze_cradle_mother.py"

    def analyze(path: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(analyzer), "--in", str(path)], capture_output=True, text=True)

    with_failed, without = analyze(out), analyze(only_ok)
    assert "1 failed run(s)" in with_failed.stderr
    # A failed row counted as a trial would halve taught's turns/seed in the exposure check.
    assert "exposure:" in without.stdout
    assert with_failed.stdout == without.stdout, "a failed row must not count as a trial"


def test_a_failed_row_never_completes_a_group(exp37) -> None:
    ok = {"trial_pair_id": 1, "scenario": "fire_pit", "total_input_tokens": 10}
    rows = [{**ok, "arm": "A"}, {**ok, "arm": "B"}, {**ok, "arm": "B", "status": "failed", "total_input_tokens": 0}]
    assert exp37._complete_clean_groups(rows, {"A", "B"}) == {(1, "fire_pit")}
    assert exp37._complete_clean_groups(rows[:1] + rows[2:], {"A", "B"}) == set()


@pytest.mark.parametrize("loaded", [True, False])
def test_exp44_capture_fails_when_its_resume_did_not_load(exp44, tmp_path: Path, monkeypatch, loaded: bool) -> None:
    """A capture that asked to resume the learned substrate must have loaded it (#1003, #1009): otherwise the
    arm measures a run with no substrate, and the stage row is failed, naming the session that ran."""
    run_id = exp44._provenance.harness_run_id()

    def fake_run_sim(cmd, env, log_path, timeout_s):
        home = Path(env["MAXIM_DATA_HOME"])
        assert env["MAXIM_HARNESS_RUN_ID"] == run_id
        pair = {"prompt_full": "f", "prompt_ablated": "a", "world_state": {"has_cluster_bias": True}}
        Path(env["MAXIM_EXP44_CAPTURE_LOG"]).write_text((json.dumps(pair) + "\n") * 6)
        session = home / "sim_reports" / "capture_session"
        session.mkdir(parents=True)
        resume = {"requested": "learn_session", "resume_loaded": loaded}
        (session / "report.json").write_text(json.dumps({"provenance": {"harness_run_id": run_id, "resume": resume}}))
        return 0

    monkeypatch.setattr(exp44, "_resolve_maxim_binary", lambda: "maxim")
    monkeypatch.setattr(exp44, "_run_sim", fake_run_sim)
    arm = {"name": "learn_arm", "arc": "arc", "substrate": "learn", "capture": {"min_pairs": 5}}
    learned = tmp_path / "arms" / "learn_arm" / "seed42" / "sim_reports" / "learn_session"
    learned.mkdir(parents=True)
    out = exp44.stage_capture(arm, 42, tmp_path / "arms" / "learn_arm", learned, {}, tmp_path, {}, False)
    (row,) = [r for r in _rows(tmp_path / "manifest.jsonl") if r.get("stage") == "capture"]
    assert row["resume_loaded"] is loaded and row["status"] == ("ok" if loaded else "failed")
    assert row["sims"][0]["session_id"] == "capture_session"
    assert [d["session_id"] for d in row["depends_on"]] == ["learn_session"]
    assert (out is not None) is loaded


def test_a_failure_after_the_report_names_the_session(exp41, tmp_path: Path, monkeypatch) -> None:
    run, _ = _fake_sim()

    def no_actions(cmd, env=None, **kwargs):
        result = run(cmd, env=env, **kwargs)
        for actions in Path(env["MAXIM_DATA_HOME"]).glob("sim_reports/*/actions.jsonl"):
            actions.unlink()
        return result

    monkeypatch.setattr(exp41.subprocess, "run", no_actions)
    with pytest.raises(exp41._provenance.SimRunFailed) as err:
        exp41._run_real(
            "A_dec", 42, model="m", embodiment="e", max_turns=3, explore_weight=1.5, timeout_s=5, workdir=tmp_path
        )
    assert err.value.sims[0]["session_id"] == "sess_0" and "no actions.jsonl" in str(err.value)
