"""Run bookkeeping in the Exp 53 harness (bugs ledger D34 / D36) and the run-aware verdict.

Verified to fail on the pre-fix harness: `runs_of` / `select_run` / `Run` did not exist, and
`cmd_verdict` pooled every trial in the file across run_ids (no `runs_used` in the gate
record; a file with two complete primary runs returned 0 instead of refusing).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts/orient_backbone/exp53_cross_context_readout.py"
R1 = REPO / "docs/experiments/data/53b_cross_context_readout_replication_2026-08-28.jsonl"


@pytest.fixture(scope="module")
def h():
    sys.path.insert(0, str(HARNESS.parent))
    spec = importlib.util.spec_from_file_location("exp53_under_test", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(
    rid: str,
    phase: int,
    agents: tuple[str, ...],
    *,
    done: tuple[str, ...] | None = None,
    condition="primary",
    trials_per_agent: int = 2,
    run_end: str | None = None,
    only=None,
    toward=True,
) -> list[dict]:
    recs = [{"ts": 1.0, "event": "start", "run_id": rid, "phase": phase, "only": only}]
    for a in agents:
        recs.append({"ts": 2.0, "event": "agent_load", "run_id": rid, "phase": phase, "agent": a})
        arm = a.split("_seed")[0]
        for i in range(trials_per_agent):
            recs.append(
                {
                    "ts": 3.0 + i,
                    "event": "trial" if phase == 2 else "probe",
                    "run_id": rid,
                    "phase": phase,
                    "agent": a,
                    "arm": arm,
                    "seed": int(a[-2:]),
                    "condition": condition,
                    "exploratory": False,
                    "exploratory_agent": False,
                    "toward": toward if arm == "taught" else False,
                    "affordance": "turn_left",
                    "sign_rule_correct": toward,
                    "target_az": -0.3,
                }
            )
    for a in done if done is not None else agents:
        recs.append({"ts": 9.0, "event": "agent_done", "run_id": rid, "phase": phase, "agent": a})
    if run_end:
        recs.append({"ts": 10.0, "event": "run_end", "run_id": rid, "phase": phase, "status": run_end})
    return recs


AGENTS = ("taught_seed42", "satiated_seed42", "no_feed_seed42")


def test_runs_of_groups_by_run_id_and_status_rules(h) -> None:
    recs = (
        _run("A", 1, AGENTS)
        + _run("B", 2, AGENTS, done=("taught_seed42",))  # legacy partial: last agents never finished
        + _run("C", 2, AGENTS, run_end="complete")
        + _run("D", 2, AGENTS, run_end="interrupted")
        + _run("E", 2, ("taught_seed42",), only=["taught_seed42"])
        + [{"event": "gate_T", "verdict": "PASS"}]  # no run_id — ignored
    )
    runs = {r.run_id: r for r in h.runs_of(recs)}
    assert set(runs) == {"A", "B", "C", "D", "E"}
    assert runs["A"].status == "complete"  # legacy file: all loaded agents done
    assert runs["B"].status == "partial"  # interrupted-vs-crashed unknowable pre-run_end (D36)
    assert runs["C"].status == "complete" and runs["D"].status == "interrupted"
    assert runs["E"].status == "debug"  # --only subset is never a result
    assert runs["C"].conditions == {"primary"} and runs["C"].n_trials == 6


def test_select_run_picks_the_one_complete_run_and_refuses_ambiguity(h) -> None:
    runs = h.runs_of(_run("P", 2, AGENTS, done=()) + _run("Q", 2, AGENTS))
    assert h.select_run(runs, phase=2, condition="primary").run_id == "Q"
    assert h.select_run(runs, phase=2, condition="secondary") is None
    runs = h.runs_of(_run("Q", 2, AGENTS) + _run("R", 2, AGENTS))
    with pytest.raises(h.VerdictError, match="2 complete phase 2 / primary runs"):
        h.select_run(runs, phase=2, condition="primary")
    assert h.select_run(runs, phase=2, condition="primary", pinned=("R",)).run_id == "R"
    with pytest.raises(h.VerdictError, match="not complete"):
        h.select_run(h.runs_of(_run("P", 2, AGENTS, done=())), phase=2, condition="primary", pinned=("P",))


def _write(tmp_path: Path, recs: list[dict]) -> Path:
    p = tmp_path / "records.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def _gate_record(p: Path, event: str) -> dict:
    recs = [json.loads(line) for line in p.read_text().splitlines()]
    return [r for r in recs if r.get("event") == event][-1]


def test_verdict_refuses_two_complete_primary_runs_without_run_id(h, tmp_path: Path, capsys) -> None:
    p = _write(tmp_path, _run("A", 1, AGENTS) + _run("Q", 2, AGENTS) + _run("R", 2, AGENTS, toward=False))
    assert h.main(["verdict", "--records", str(p)]) == 2
    assert "REFUSED" in capsys.readouterr().out
    rc = h.main(["verdict", "--records", str(p), "--run-id", "Q"])
    gate = _gate_record(p, "gate_T")
    assert gate["runs_used"]["primary"] == "Q" and rc in (0, 1)
    assert [r["run_id"] for r in gate["runs_excluded"]] == ["R"]


def test_verdict_excludes_partial_runs(h, tmp_path: Path) -> None:
    p = _write(tmp_path, _run("A", 1, AGENTS) + _run("P", 2, AGENTS, done=()) + _run("Q", 2, AGENTS))
    h.main(["verdict", "--records", str(p)])
    gate = _gate_record(p, "gate_T")
    assert gate["runs_used"] == {"phase1": "A", "primary": "Q", "secondary": None}
    assert [(r["run_id"], r["status"]) for r in gate["runs_excluded"]] == [("P", "partial")]


@pytest.mark.skipif(not R1.exists(), reason="R1 data file not present")
def test_r1_file_selects_the_complete_runs_and_reproduces_the_numbers(h, tmp_path: Path) -> None:
    p = tmp_path / "r1.jsonl"
    shutil.copy(R1, p)
    assert h.main(["verdict", "--records", str(p)]) == 0
    gate = _gate_record(p, "gate_T")
    assert gate["runs_used"] == {
        "phase1": "20260828T144251Z-55302",
        "primary": "20260829T011136Z-22574",
        "secondary": None,
    }
    assert sorted(r["run_id"] for r in gate["runs_excluded"]) == [
        "20260828T143101Z-42201",
        "20260828T145949Z-73973",
        "20260828T181211Z-28228",
    ]
    assert gate["primary_directedness_by_arm"] == {"taught": 1.0, "satiated": 0.0, "no_feed": 0.5}
    assert gate["exploratory_placements_taught"]["-0.6"]["n"] == 9  # the complete run alone, not 15 pooled
