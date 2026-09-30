"""M1b PR 5a-2 -- every writer the ledger cites stamps what its record is, and an event log says how its run ended.

The orient event logs (``live_common.JsonlLog``) declare evidence vs non-support. An evidence run ends in exactly
ONE terminal line, ``ok`` only through an explicit ``finish("ok")`` that no abort latched; an exception, an early
return or a caught-and-continued abort ends it ``failed``, so the data lines of an aborted run never count. Instrument
checks pass only at their frozen parameters; diagnoses and headers are never support; a verdict over a smoke, or over
lines that do not say, is a smoke. Each test reads a writer's real output or drives its real code path.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
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
import live_common as lc  # noqa: E402

EVIDENCE_LOGS = {"doa_sweep.py", "delivered_shift_block.py", "live_3_learn.py", "exp53_cross_context_readout.py"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _terminals(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["record_kind"] == "harness_run_end"]


# ── the JsonlLog run protocol ────────────────────────────────────────────


def test_an_evidence_run_ends_ok_only_through_finish(tmp_path: Path) -> None:
    out = tmp_path / "log.jsonl"
    with lc.JsonlLog(str(out), mock=False, evidence=True) as log:
        log.write("start", provenance=log.provenance)
        log.write("trial", i=1)
        log.finish("ok")
    rows = _rows(out)
    assert [r["record_kind"] for r in rows] == ["harness_event", "harness_event", "harness_run_end"]
    assert {r["log_run_id"] for r in rows} == {log.log_run_id} and all(r["mock"] is False for r in rows)
    assert rows[0]["provenance"]["harness_family"] == "in_process"
    # the full block once per run (first + terminal line); every line carries its digest (owner decision 2026-09-30)
    assert "provenance" not in rows[1] and {r["provenance_sha256"] for r in rows} == {log.provenance_sha256}
    (end,) = _terminals(rows)
    assert end["status"] == "ok" and end["end_code_tree_sha256"] == rows[0]["provenance"]["code_tree_sha256"]
    assert end["provenance"] == rows[0]["provenance"]


def test_every_log_mints_its_own_run_id(tmp_path: Path) -> None:
    """gate6 runs several Exp 53 phases in ONE process: a process-wide id would merge their runs."""
    out = tmp_path / "log.jsonl"
    for _ in range(2):
        with lc.JsonlLog(str(out), mock=False, evidence=True) as log:
            log.write("start")
            log.finish("ok")
    assert len({r["log_run_id"] for r in _rows(out)}) == 2


@pytest.mark.parametrize("how", ["exception", "no_finish", "close_then_exit", "latched_abort", "mark_aborted"])
def test_every_other_ending_is_exactly_one_failed_terminal(how: str, tmp_path: Path) -> None:
    out = tmp_path / "log.jsonl"

    def body(log) -> None:
        log.write("trial", i=1)
        if how == "exception":
            raise KeyboardInterrupt
        if how == "close_then_exit":
            log.close()
            log.close()  # idempotent
        if how in ("latched_abort", "mark_aborted"):
            log.write("block_aborted", reason="x") if how == "latched_abort" else log.mark_aborted("robot lost")
            with pytest.raises(RuntimeError, match="aborted"):
                log.finish("ok")  # a caught-and-continued abort can never end ok

    with pytest.raises(KeyboardInterrupt) if how == "exception" else _no_raise():
        with lc.JsonlLog(str(out), mock=False, evidence=True) as log:
            body(log)
    (end,) = _terminals(_rows(out))
    assert end["status"] == "failed"
    if how == "exception":
        assert end["reason"] == "exception: KeyboardInterrupt", "the terminal line says why the run failed"


class _no_raise:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_the_terminal_line_is_final(tmp_path: Path) -> None:
    with lc.JsonlLog(str(tmp_path / "log.jsonl"), mock=False, evidence=True) as log:
        with pytest.raises(ValueError):
            log.finish("stopped")  # a terminal status is ok or failed, nothing else
        log.finish("ok")
        with pytest.raises(RuntimeError):
            log.finish("ok")
        with pytest.raises(RuntimeError):
            log.write("late")


def test_a_non_support_log_has_no_terminal_and_never_says_evidence(tmp_path: Path) -> None:
    out = tmp_path / "log.jsonl"
    with lc.JsonlLog(str(out), mock=True, evidence=False) as log:
        log.write("demo_decision", evidence=False)  # a caller field of that name keeps its meaning
        with pytest.raises(RuntimeError, match="non-support"):
            log.finish("ok")
    (row,) = _rows(out)
    assert row["record_kind"] == "harness_demo" and row["mock"] is True and row["evidence"] is False


def test_a_caller_cannot_override_the_stamps(tmp_path: Path) -> None:
    out = tmp_path / "log.jsonl"
    with lc.JsonlLog(str(out), mock=True, evidence=False) as log:
        log.write("x", mock=False, record_kind="harness_row", log_run_id="forged")
    (row,) = _rows(out)
    assert (row["mock"], row["record_kind"], row["log_run_id"]) == (True, "harness_demo", log.log_run_id)


def test_a_caller_provenance_must_be_the_logs_own(tmp_path: Path) -> None:
    with lc.JsonlLog(str(tmp_path / "log.jsonl"), mock=False, evidence=False) as log:
        with pytest.raises(ValueError, match="log.provenance"):
            log.write("start", provenance={"executed_git_hash": "other"})


def test_a_log_refuses_when_its_maxim_is_not_this_repos(tmp_path: Path, monkeypatch) -> None:
    """Owner decision 2026-09-30: for ANY path, scratch logs too."""
    monkeypatch.setattr(lc, "_maxim_file", lambda: str(tmp_path / "elsewhere" / "maxim" / "__init__.py"))
    with pytest.raises(SystemExit) as ei:
        lc.JsonlLog(str(tmp_path / "scratch.jsonl"), mock=False, evidence=False)
    assert ei.value.code == 3 and not (tmp_path / "scratch.jsonl").exists()


# ── the call shape, pinned (the design pass's positive rule) ─────────────


def _jsonllog_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) == "JsonlLog" or getattr(n.func, "attr", None) == "JsonlLog")
    ]


def _kw(call: ast.Call, name: str):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _finish_positions_ok(tree: ast.AST) -> list[str]:
    """Every ``.finish(`` is a statement directly in an evidence ``with JsonlLog(...)`` body (an ``if`` wrapper
    allowed), in the body's last statement; no bare ``.finish`` attribute (an alias)."""
    allowed: set[int] = set()
    for w in (n for n in ast.walk(tree) if isinstance(n, ast.With)):
        if not any(isinstance(i.context_expr, ast.Call) and i.context_expr in _jsonllog_calls(w) for i in w.items):
            continue
        last = w.body[-1]
        stmts = [last, *(last.body if isinstance(last, ast.If) else [])]
        for st in stmts:
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                allowed.add(id(st.value))
    problems = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "finish" and id(n) not in allowed:
            problems.append(f"finish() at line {n.lineno} is not the last statement of an evidence `with` body")
    finish_calls = {
        id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "finish"
    }
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "finish" and id(n) not in finish_calls:
            problems.append(f"`.finish` aliased at line {n.lineno}")
    return problems


def _abort_class_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
    if name == "mark_aborted":
        return True
    first = node.args[0] if node.args else None
    return (
        name in ("emit", "write")
        and isinstance(first, ast.Constant)
        and isinstance(first.value, str)
        and (first.value == "abort" or first.value.endswith("_aborted"))
    )


def _handler_problems(tree: ast.AST) -> list[str]:
    """v6 SF-1, applied where the run's code now lives: in every function that receives the evidence log (a
    parameter annotated ``JsonlLog``), each ``except`` handler re-raises, writes an abort-class event
    (``abort`` / ``*_aborted`` — which latches the run failed) or calls ``mark_aborted``, or sits inside a handler
    that does. So no caught-and-continued failure can reach ``finish("ok")`` from inside these functions."""
    problems = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        if not any(a.annotation is not None and "JsonlLog" in ast.unparse(a.annotation) for a in fn.args.args):
            continue
        handlers = [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]
        ok = {id(h) for h in handlers if any(isinstance(n, ast.Raise) or _abort_class_call(n) for n in ast.walk(h))}
        covered = {
            id(inner) for h in handlers if id(h) in ok for inner in ast.walk(h) if isinstance(inner, ast.ExceptHandler)
        }
        for h in handlers:
            if id(h) not in ok and id(h) not in covered:
                problems.append(f"{fn.name}: except at line {h.lineno} neither re-raises nor marks the run aborted")
    return problems


def test_every_orient_log_declares_evidence_and_evidence_logs_run_in_a_with() -> None:
    seen: set[str] = set()
    for path in sorted(ORIENT.glob("*.py")):
        if path.name == "live_common.py":
            continue
        tree = ast.parse(path.read_text())
        for call in _jsonllog_calls(tree):
            ev, mock = _kw(call, "evidence"), _kw(call, "mock")
            assert isinstance(ev, ast.Constant) and isinstance(ev.value, bool), f"{path.name}:{call.lineno}"
            assert mock is not None, f"{path.name}:{call.lineno}: mock= is required"
            if ev.value:
                seen.add(path.name)
                withs = [i.context_expr for w in ast.walk(tree) if isinstance(w, ast.With) for i in w.items]
                assert call in withs, f"{path.name}:{call.lineno}: an evidence log is a `with` item"
        assert _finish_positions_ok(tree) == [], path.name
        assert _handler_problems(tree) == [], path.name
    assert seen == EVIDENCE_LOGS


def test_the_call_shape_guard_catches_a_finish_in_finally_and_an_alias() -> None:
    bad_finally = (
        "with JsonlLog(p, mock=False, evidence=True) as log:\n"
        "    try:\n        run(log)\n    finally:\n        log.finish('ok')\n"
    )
    bad_alias = "with JsonlLog(p, mock=False, evidence=True) as log:\n    f = log.finish\n    f('ok')\n"
    good = "with JsonlLog(p, mock=False, evidence=True) as log:\n    rc = run(log)\n    if rc == 0:\n        log.finish('ok')\n"
    assert _finish_positions_ok(ast.parse(good)) == []
    assert _finish_positions_ok(ast.parse(bad_finally)) and _finish_positions_ok(ast.parse(bad_alias))
    swallow = "def _run(args, log: JsonlLog) -> int:\n    try:\n        go()\n    except ConnectionError:\n        pass\n    return 0\n"
    latched = swallow.replace("        pass\n", "        log.write('abort', reason='x')\n")
    assert _handler_problems(ast.parse(swallow)) and _handler_problems(ast.parse(latched)) == []


# ── the evidence callers, driven through their real paths ────────────────


def test_a_doa_sweep_dry_run_is_one_mock_run_ending_ok(tmp_path: Path, monkeypatch) -> None:
    sweep = _load("m1b5a2_doa_sweep", ORIENT / "doa_sweep.py")
    out = tmp_path / "sweep.jsonl"
    monkeypatch.setattr(sys, "argv", ["doa_sweep.py", "--dry-run", "--log", str(out), "--step", "0.7", "--reads", "2"])
    assert sweep.main() == 0
    rows = _rows(out)
    (end,) = _terminals(rows)
    assert end["status"] == "ok" and all(r["mock"] is True for r in rows)


def test_a_ctrl_c_in_a_delivered_shift_block_ends_failed(tmp_path: Path, monkeypatch) -> None:
    """The handler writes block_aborted and returns 0 with a partial summary: the latch makes that run failed."""
    dsb = _load("m1b5a2_dsb", ORIENT / "delivered_shift_block.py")

    def interrupted(self, affordance):
        raise KeyboardInterrupt

    monkeypatch.setattr(dsb.DryBlockRig, "execute", interrupted)
    out = tmp_path / "block.jsonl"
    argv = ["delivered_shift_block.py", "--dry-run", "--log", str(out), "--reps", "1", "--settle", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    assert dsb.main() == 0  # the harness's own rc is unchanged
    rows = _rows(out)
    assert "block_aborted" in [r.get("event") for r in rows]
    (end,) = _terminals(rows)
    assert end["status"] == "failed" and end["reason"] == "block_aborted"


def test_a_lost_robot_in_live_3_learn_ends_failed(tmp_path: Path, monkeypatch) -> None:
    """ConnectionError → `abort` → break → the summary is still written: the latch makes that run failed."""
    learn = _load("m1b5a2_live_3_learn", ORIENT / "live_3_learn.py")
    calls = {"n": 0}
    real = learn.DryRig.goto_body_yaw

    def drops(self, *a, **k):
        calls["n"] += 1
        if calls["n"] > 3:
            raise ConnectionError("backend not ready")
        return real(self, *a, **k)

    monkeypatch.setattr(learn.DryRig, "goto_body_yaw", drops)
    out = tmp_path / "learn.jsonl"
    argv = [
        "live_3_learn.py",
        "--dry-run",
        "--log",
        str(out),
        "--nac-path",
        str(tmp_path / "nac.json"),
        "--trials",
        "6",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert learn.main() == 0
    rows = _rows(out)
    assert "abort" in [r.get("event") for r in rows] and "summary" in [r.get("event") for r in rows]
    (end,) = _terminals(rows)
    assert end["status"] == "failed"


# ── the Exp 53 verdict: its own record, a smoke when its lines are ───────


def _exp53():
    return _load("m1b5a2_exp53", ORIENT / "exp53_cross_context_readout.py")


def _runs(stamped: dict | None) -> list[dict]:
    agents = ("taught_seed42", "satiated_seed42", "no_feed_seed42")
    recs: list[dict] = []
    for rid, phase in (("A", 1), ("Q", 2)):
        recs.append({"event": "start", "run_id": rid, "phase": phase, "only": None})
        for a in agents:
            arm = a.split("_seed")[0]
            recs.append({"event": "agent_load", "run_id": rid, "phase": phase, "agent": a})
            for i in range(2):
                recs.append(
                    {
                        "ts": 3.0 + i,
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
            recs.append({"event": "agent_done", "run_id": rid, "phase": phase, "agent": a})
    return [{**r, **(stamped or {})} for r in recs]


@pytest.mark.parametrize(
    ("stamp", "mock", "stamped"),
    [
        (None, True, False),  # legacy lines: unknown is mock, never support
        ({"log_run_id": "x", "mock": True, "provenance": {}}, True, True),  # a dry run
        ({"log_run_id": "x", "mock": False, "provenance": {}}, False, True),
    ],
)
def test_the_exp53_verdict_is_its_own_record_and_a_smoke_when_its_lines_are(
    stamp, mock: bool, stamped: bool, tmp_path: Path
) -> None:
    h = _exp53()
    records = tmp_path / "records.jsonl"
    records.write_text("\n".join(json.dumps(r) for r in _runs(stamp)) + "\n")
    before = records.read_bytes()
    assert h.main(["verdict", "--records", str(records)]) in (0, 1)
    assert records.read_bytes() == before, "the verdict never appends to the records it judged"
    v = json.loads((tmp_path / "records_verdict.json").read_text())
    assert (v["record_kind"], v["kind"], v["gate"]) == ("verdict", "exp53_verdict", "T")
    assert v["scope"] == {"run_ids": ["A", "Q"]} and v["mock"] is mock and v["scoped_lines_stamped"] is stamped
    assert v["provenance"]["harness_family"] == "in_process"
    assert h.main(["verdict", "--records", str(records)]) == 2, "an existing verdict is not silently replaced"


@pytest.mark.parametrize(("rc", "status"), [(0, "ok"), (6, "ok"), (5, "failed"), (7, "failed"), (2, "failed")])
def test_exp53_ends_a_run_ok_only_on_a_computed_outcome(rc: int, status: str, tmp_path: Path, monkeypatch) -> None:
    """rc 0 (PASS) and 6 (a Gate-I / Gate-C FAIL) are results; a stop rule (5, 7) or a refusal is not."""
    import types

    h = _exp53()
    (tmp_path / "manifest.json").write_text("{}")

    def fake_run(args, log, **kw):
        log.write("start", run_id="r", phase=1)
        return rc

    monkeypatch.setattr(h, "_run_logged", fake_run)
    out = tmp_path / "records.jsonl"
    args = types.SimpleNamespace(
        gate="T",
        whitelist=False,
        delta=None,
        factory=False,
        targets=None,
        allow_incomplete_targets=False,
        manifest=str(tmp_path / "manifest.json"),
        out=str(out),
        phase=1,
        dry_run=True,
        allow_dirty=False,
    )
    assert h.cmd_run(args) == rc
    (end,) = _terminals(_rows(out))
    assert end["status"] == status


def test_one_mock_line_anywhere_makes_the_exp53_verdict_mock(tmp_path: Path) -> None:
    """Owner decision 2026-09-30: a verdict's mock is judged over the whole file; smokes go in their own files."""
    h = _exp53()
    real = {"log_run_id": "x", "mock": False, "provenance": {}}
    records = tmp_path / "records.jsonl"
    rows = _runs(real) + [{"event": "start", "run_id": "Z", "phase": 2, **real, "mock": True}]
    records.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert h.main(["verdict", "--records", str(records), "--run-id", "A", "--run-id", "Q"]) in (0, 1)
    v = json.loads((tmp_path / "records_verdict.json").read_text())
    assert v["scope"] == {"run_ids": ["A", "Q"]} and v["mock"] is True


def test_gate6_reads_only_a_verdict_bound_to_the_current_records(tmp_path: Path) -> None:
    g6 = _load("m1b5a2_gate6", ORIENT / "gate6_merged_gauntlet.py")
    records = tmp_path / "records_A.jsonl"
    records.write_text(json.dumps({"event": "gate_I", "verdict": "PASS"}) + "\n")
    verdict = tmp_path / "records_A_verdict.json"
    verdict.write_text(json.dumps({"gate": "T", "verdict": "PASS", "data_sha256": "stale"}))
    assert "gate_T" not in g6._gate_records(records), "a verdict from an earlier run of these records"
    import hashlib

    verdict.write_text(
        json.dumps({"gate": "T", "verdict": "PASS", "data_sha256": hashlib.sha256(records.read_bytes()).hexdigest()})
    )
    assert g6._gate_records(records)["gate_T"]["verdict"] == "PASS"


@pytest.mark.parametrize(
    ("record", "why"),
    [
        ({"all_pass": True}, "pre-M1b"),
        ({"record_kind": "instrument_check", "status": "ok", "pass": True, "mock": True}, "mock"),
        ({"record_kind": "instrument_check", "status": "failed", "pass": False, "mock": False}, "ended"),
        ({"record_kind": "instrument_check", "status": "ok", "pass": False, "mock": False}, "frozen"),
    ],
)
def test_only_a_stamped_passing_real_check_authorizes_a_live_run(record: dict, why: str) -> None:
    """Exp 58/60/61 and R3 used to authorize on `all_pass`, which a `--cycles 1` check also sets."""
    assert why in P.instrument_check_authorizes(record)
    passing = {"record_kind": "instrument_check", "status": "ok", "pass": True, "mock": False}
    assert P.instrument_check_authorizes(passing) is None


@pytest.mark.parametrize("rel", ["exp60_run.py", "exp61_run.py", "r3_run.py", "exp58_run.py"])
def test_the_live_harnesses_authorize_through_the_one_rule(rel: str) -> None:
    src = (SCRIPTS / "survival_world" / rel).read_text()
    assert "instrument_check_authorizes(" in src and '.get("all_pass")' not in src


def test_a_void_exp52_phase_a_run_is_failed(tmp_path: Path, monkeypatch) -> None:
    """The credit path misbehaved (no feed credited): VOID is an apparatus failure, never a trial."""
    phase_a = _load("m1b5a2_exp52a", SCRIPTS / "orient_substrate/9_hunger_relief_orient.py")
    tel = {"fed": 0, "credits": 0, "credit_rewards": [], "fed_ticks": [], "hunger_at_feed_median": None}
    monkeypatch.setattr(phase_a, "run", lambda arm, **kw: ([0.5] * 12, [0] * 600, dict(tel)))
    out = tmp_path / "void.json"
    monkeypatch.setattr(sys, "argv", ["9_hunger_relief_orient.py", "--json", str(out)])
    assert phase_a.main() == 4
    r = json.loads(out.read_text())
    assert (r["verdict"], r["status"], r["record_kind"]) == ("VOID", "failed", "harness_row")


# ── single-record writers ─────────────────────────────────────────────────


def test_the_stamps_for_checks_diagnoses_and_headers() -> None:
    ok = P.stamp_instrument_check({"instrument_error": None}, mock=False, passed=True)
    assert (ok["record_kind"], ok["status"], ok["pass"], ok["mock"]) == ("instrument_check", "ok", True, False)
    broken = P.stamp_instrument_check({"instrument_error": "lost bridge"}, mock=False, passed=True)
    assert broken["status"] == "failed" and broken["pass"] is False
    d = P.stamp_diagnosis({"provenance": {"anchor_file": "a"}}, mock=False, code_provenance={"code": 1})
    assert d["record_kind"] == "diagnosis" and d["provenance"] == {"anchor_file": "a"} and d["code_provenance"]
    assert P.stamp_harness_header({"stage": "campaign_start"}, mock=True)["record_kind"] == "harness_header"
    with pytest.raises(ValueError, match="no status"):
        P.stamp_harness_header({"status": "ok"}, mock=False)


def test_the_exp58_offline_gates_are_a_passing_instrument_check(tmp_path: Path) -> None:
    out = tmp_path / "g.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "survival_world/exp58_offline_gates.py"), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    r = json.loads(out.read_text())
    assert (r["record_kind"], r["status"], r["pass"], r["mock"]) == ("instrument_check", "ok", True, False)


@pytest.mark.parametrize("rel", ["survival_world/instrument_check.py", "survival_world/exp60_water_check.py"])
def test_a_live_check_passes_only_at_its_frozen_cycles(rel: str) -> None:
    """Live-only (the rig), so structural: `pass` requires `frozen_params`, and that is `args.cycles == CYCLES`."""
    src = (SCRIPTS / rel).read_text()
    assert 'report["frozen_params"] = args.cycles == CYCLES' in src
    assert 'passed=report["all_pass"] and report["frozen_params"]' in src


def test_exp52_phase_a_passes_only_at_its_frozen_parameters(tmp_path: Path) -> None:
    script = SCRIPTS / "orient_substrate/9_hunger_relief_orient.py"
    verdicts = {}
    for name, extra in (("frozen", []), ("tuned", ["--seeds", "2"])):
        out = tmp_path / f"{name}.json"
        proc = subprocess.run([sys.executable, str(script), "--json", str(out), *extra], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
        verdicts[name] = json.loads(out.read_text())
    frozen, tuned = verdicts["frozen"], verdicts["tuned"]
    assert (frozen["record_kind"], frozen["status"], frozen["mock"]) == ("harness_row", "ok", False)
    assert frozen["verdict"] == "PASS" and frozen["frozen_params"] is True and "ts" in frozen
    assert tuned["verdict"] == "NOT_FROZEN" and tuned["frozen_params"] is False


def test_the_exp62_precheck_records_an_instrument_error(tmp_path: Path, monkeypatch) -> None:
    pre = _load("m1b5a2_exp62_precheck", SCRIPTS / "survival_world/exp62_precheck.py")
    for name in ("p1.json", "p2.json"):
        (tmp_path / name).write_text(json.dumps({"measured": {}, "shore": [0, 64, 0]}))

    def no_world(*a, **k):
        raise pre.InstrumentError("bridge not reachable")

    monkeypatch.setattr(pre, "build_trial", no_world)
    out = tmp_path / "pre.json"
    argv = ["--pool1-anchor", str(tmp_path / "p1.json"), "--pool2-anchor", str(tmp_path / "p2.json")]
    assert pre.main([*argv, "--out", str(out), "--rcon-password", "x"]) == 4
    r = json.loads(out.read_text())
    assert (r["record_kind"], r["status"]) == ("diagnosis", "failed") and r["instrument_error"]
    assert r["code_provenance"]["harness_family"] == "in_process"
    # a later failed attempt must not erase that record
    assert pre.main([*argv, "--out", str(out), "--rcon-password", "x"]) == 2
