"""#1003 (M1b PR 2) -- a harness row and the sim reports it spawned can be joined, and each stamp says what
actually happened.

The sim reports live under the gitignored ``data/``, so the committed harness row must carry what the ledger
gate will judge: which session the spawn wrote (found by the harness's run id, never the newest directory),
that report's evidence, and -- for a resume -- whether the prior state really loaded. Hashes are the full
commit id on both sides, so they compare byte for byte.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import maxim
from maxim.simulation import report as report_mod
from maxim.simulation.orchestrator import _restore_aut_from_session
from maxim.simulation.report import SimulationReport, resume_stamp, run_provenance
from maxim.simulation.sim_types import load_resume_context_at
from maxim.utils import code_tree

REPO = Path(maxim.__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import _provenance  # noqa: E402

ORCHESTRATOR = REPO / "src" / "maxim" / "simulation" / "orchestrator.py"


@pytest.fixture(autouse=True)
def _fresh_run_id():
    """The conftest scrub clears the ``_provenance`` in ``sys.modules``; other tests re-load the script under
    that name, so the object imported above may be a different one. Clear its run-id cache too."""
    _provenance._RUN_ID.clear()
    yield
    _provenance._RUN_ID.clear()


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


# ── one hash length ──────────────────────────────────────────────────────


def test_every_stamp_names_the_full_commit() -> None:
    head = _head()
    assert len(head) == 40
    assert code_tree.head_commit(REPO) == head
    assert _provenance._git_hash(REPO) == head
    assert report_mod.capture_start_provenance()["executed_git_hash"] == head
    in_process = _provenance.in_process_code_provenance(REPO, maxim.__file__)
    assert in_process["executed_git_hash"] == head


def test_head_hash_keeps_its_abbreviating_meaning() -> None:
    """``head_hash(repo, None)`` stays git's default abbreviation; the full id has its own name."""
    assert _head().startswith(code_tree.head_hash(REPO))
    assert len(code_tree.head_hash(REPO, 12)) == 12


# ── the run id ───────────────────────────────────────────────────────────


def test_both_sides_name_the_same_variable() -> None:
    assert _provenance.HARNESS_RUN_ID_ENV == report_mod.HARNESS_RUN_ID_ENV == "MAXIM_HARNESS_RUN_ID"


def test_the_run_id_is_minted_once_and_never_inherited(monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_HARNESS_RUN_ID", "a-parent-or-a-stray-export")
    first = _provenance.harness_run_id()
    assert first != "a-parent-or-a-stray-export" and len(first) == 32
    assert _provenance.harness_run_id() == first
    stamp = _provenance.in_process_code_provenance(REPO, maxim.__file__)
    assert stamp["harness_run_id"] == first
    assert stamp["parent_harness_run_id"] == "a-parent-or-a-stray-export"


def test_no_parent_is_stamped_when_none_was_inherited() -> None:
    stamp = _provenance.in_process_code_provenance(REPO, maxim.__file__)
    assert stamp["harness_run_id"] == _provenance.harness_run_id()
    assert "parent_harness_run_id" not in stamp


def test_a_sim_stamps_the_run_id_its_harness_set(monkeypatch) -> None:
    assert "harness_run_id" not in report_mod.capture_start_provenance()
    monkeypatch.setenv("MAXIM_HARNESS_RUN_ID", "rid-1003")
    assert report_mod.capture_start_provenance()["harness_run_id"] == "rid-1003"


def test_every_report_says_what_it_is() -> None:
    fields = {f.name: f.default for f in dataclasses.fields(SimulationReport)}
    assert fields["record_kind"] == "sim_report"


# ── finding the harness's own report ─────────────────────────────────────


def _session(home: Path, name: str, run_id: str | None, **extra) -> Path:
    d = home / "sim_reports" / name
    d.mkdir(parents=True)
    prov = {"harness_run_id": run_id} if run_id is not None else {}
    (d / "report.json").write_text(json.dumps({"provenance": prov, **extra}))
    return d


def test_the_report_is_found_by_run_id_not_by_age(tmp_path: Path) -> None:
    _session(tmp_path, "20260929_100000", "mine", finish_reason="completed")
    _session(tmp_path, "20260929_100001", "someone-else")  # newer, and not ours
    session, report = _provenance.find_own_report(tmp_path, "mine", set())
    assert session.name == "20260929_100000" and report["finish_reason"] == "completed"


def test_a_session_the_home_already_held_is_never_the_spawns(tmp_path: Path) -> None:
    """A copied home keeps the prior's report, carrying the SAME run id; only the snapshot excludes it."""
    _session(tmp_path, "prior", "mine")
    with pytest.raises(_provenance.OwnReportError) as err:
        _provenance.find_own_report(tmp_path, "mine", {"prior"})
    assert err.value.detail == {"before": ["prior"], "after": ["prior"]}


def test_two_candidates_are_refused_with_both_echoed(tmp_path: Path) -> None:
    _session(tmp_path, "a", "mine")
    _session(tmp_path, "b", "mine")
    with pytest.raises(_provenance.OwnReportError) as err:
        _provenance.find_own_report(tmp_path, "mine", set())
    assert [s["session_id"] for s in err.value.sims] == ["a", "b"]


def test_a_failed_exit_is_a_failed_run_with_its_evidence(tmp_path: Path) -> None:
    _session(tmp_path, "s", "mine", finish_reason="planning_failed")
    with pytest.raises(_provenance.SimRunFailed) as err:
        _provenance.spawn_evidence(tmp_path, "mine", set(), returncode=4)
    assert err.value.sims[0]["finish_reason"] == "planning_failed"
    with pytest.raises(_provenance.SimRunFailed, match="exited 4 and no report"):
        _provenance.spawn_evidence(tmp_path / "empty", "mine", set(), returncode=4)


def test_the_evidence_projection_copies_what_the_report_stamped(tmp_path: Path) -> None:
    prov = {
        "harness_run_id": "mine",
        "executed_git_hash": "a" * 40,
        "code_tree_sha256": "b" * 64,
        "working_tree_dirty_src_scripts": False,
        "code_changed_during_run": False,
        "configured_n_ctx": 8192,
        "aut_profile": "qwen",
        "aut_router_n_ctx": 8192,
        "aut_budget_n_ctx": 8000,
        "python": "not echoed",
    }
    d = tmp_path / "sim_reports" / "s"
    d.mkdir(parents=True)
    report = {"record_kind": "sim_report", "finish_reason": "completed", "ts": 1.5, "provenance": prov}
    ev = _provenance.sim_evidence(d, report)
    assert ev["session_id"] == "s" and ev["report_found"] is True
    assert ev["finish_reason"] == "completed" and ev["ts"] == 1.5 and ev["record_kind"] == "sim_report"
    for key in ("harness_run_id", "executed_git_hash", "code_tree_sha256", "aut_profile", "aut_budget_n_ctx"):
        assert ev[key] == prov[key]
    assert ev["end_code_tree_sha256"] is None and "python" not in ev
    assert _provenance.sim_evidence(d, None)["report_found"] is False


def test_a_failed_row_says_so_and_legacy_rows_are_not_failed() -> None:
    exc = _provenance.SimRunFailed("boom", sims=[{"session_id": "s"}], detail={"before": []})
    row = _provenance.failed_row(exc)
    assert row["status"] == "failed" and row["record_kind"] == "harness_row" and row["sims"] == [{"session_id": "s"}]
    assert _provenance.is_failed_row(row)
    assert not _provenance.is_failed_row({"arm": "A"})  # a legacy row: no status, a trial


# ── what a resume actually restored ──────────────────────────────────────


class _Store:
    def __init__(self, *, fails: bool = False) -> None:
        self.fails, self.loaded = fails, []
        self._links: dict = {}
        self._substrate_nodes: list = []

    def load(self, path: str, **kwargs) -> None:
        if self.fails:
            raise ValueError("corrupt")
        self.loaded.append(path)

    def __len__(self) -> int:
        return 0


def _prior(home: Path, name: str, *stores: str) -> Path:
    d = home / "sim_reports" / name
    d.mkdir(parents=True)
    (d / "report.json").write_text('{"goal": "g"}')
    for s in stores:
        (d / f"aut_{s}.json").write_text("{}")
    return d


def test_the_restore_reports_each_store(tmp_path: Path, monkeypatch, caplog) -> None:
    caplog.set_level("DEBUG", logger="maxim.simulation.orchestrator")
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    prior = _prior(tmp_path, "20260929_120000", "hippocampus", "nac", "ec")
    hippo, nac, ec = _Store(), _Store(fails=True), _Store()
    hub = SimpleNamespace(ec=ec, atl=None)
    record = _restore_aut_from_session(
        "20260929_120000", persistent_agent=None, aut_hippocampus=hippo, aut_nac=nac, aut_memory_hub=hub
    )
    assert record["state_dir"] == str(prior.resolve())
    assert record["stores"] == {"hippocampus": "loaded", "nac": "failed:ValueError", "ec": "loaded", "atl": "absent"}
    assert hippo.loaded == [str(prior / "aut_hippocampus.json")]
    # The log lines keep their spelling: operators grep "Restored AUT EC" (Roy 5a protocol).
    assert "Restored AUT EC from" in caplog.text and "Failed to restore AUT NAc" in caplog.text


def test_an_adopted_agent_skips_the_restore_and_says_so(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    _prior(tmp_path, "p", "nac")
    hippo = _Store()
    record = _restore_aut_from_session(
        "p", persistent_agent=object(), aut_hippocampus=hippo, aut_nac=_Store(), aut_memory_hub=None
    )
    assert record["stores"]["nac"] == "skipped_persistent_agent" and record["stores"]["hippocampus"] == "absent"
    assert hippo.loaded == []


def test_a_store_the_run_lacks_is_not_loaded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    _prior(tmp_path, "p", "hippocampus", "atl")
    record = _restore_aut_from_session(
        "p", persistent_agent=None, aut_hippocampus=_Store(), aut_nac=None, aut_memory_hub=None
    )
    assert record["stores"]["atl"] == "no_store_in_run"
    assert (
        _restore_aut_from_session(
            None, persistent_agent=None, aut_hippocampus=_Store(), aut_nac=None, aut_memory_hub=None
        )
        is None
    )


def _resume(state_dir: Path | None, context_dir: Path | None, stores: dict, *, loaded: bool = True) -> dict:
    return {
        "requested": "x",
        "state_dir": str(state_dir) if state_dir else None,
        "context_dir": str(context_dir) if context_dir else None,
        "context_loaded": loaded,
        "stores": stores,
    }


ALL_LOADED = {"hippocampus": "loaded", "nac": "loaded", "ec": "absent", "atl": "absent"}


def test_a_resume_that_loaded_names_the_session(tmp_path: Path) -> None:
    d = _prior(tmp_path, "20260929_120000")
    stamp = resume_stamp(_resume(d, d, ALL_LOADED))
    assert stamp["resume_loaded"] is True and stamp["resumed_from_session"] == "20260929_120000"


@pytest.mark.parametrize(
    "case",
    ["prefix_resolved_elsewhere", "store_failed", "store_skipped", "not_restored", "context_missing", "no_dir"],
)
def test_a_resume_that_did_not_load_says_so(tmp_path: Path, case: str) -> None:
    d = _prior(tmp_path, "20260929_120000")
    other = _prior(tmp_path, "20260929_120000_b")
    stores, state, context, loaded = dict(ALL_LOADED), d, d, True
    if case == "prefix_resolved_elsewhere":  # #1009: the prompt took a prefix match, the stores the exact dir
        context = other
    elif case == "store_failed":
        stores["nac"] = "failed:ValueError"
    elif case == "store_skipped":
        stores["nac"] = "skipped_persistent_agent"
    elif case == "not_restored":
        del stores["hippocampus"]
    elif case == "context_missing":
        context, loaded = None, False
    elif case == "no_dir":
        state = context = tmp_path / "gone"
    stamp = resume_stamp(_resume(state, context, stores, loaded=loaded))
    assert stamp["resume_loaded"] is False and "resumed_from_session" not in stamp


def test_the_resume_context_names_the_directory_it_resolved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    d = _prior(tmp_path, "20260929_120000_full")
    data, resolved = load_resume_context_at("20260929_120000")  # a prefix
    assert data == {"goal": "g"} and resolved == d.resolve()
    assert load_resume_context_at("nope") == (None, None)


def test_a_prefix_resume_is_stamped_as_not_loaded(tmp_path: Path, monkeypatch) -> None:
    """The #1009 composition end to end: the restore reads the exact name (no such dir), the prompt resolves
    the prefix, and the stamp says the resume did not load."""
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    _prior(tmp_path, "20260929_120000_full", "hippocampus")
    record = _restore_aut_from_session(
        "20260929_120000", persistent_agent=None, aut_hippocampus=_Store(), aut_nac=_Store(), aut_memory_hub=None
    )
    data, resolved = load_resume_context_at("20260929_120000")
    record.update(context_dir=str(resolved), context_loaded=bool(data))
    stamp = resume_stamp(record)
    assert stamp["context_loaded"] is True and stamp["resume_loaded"] is False


def test_the_report_carries_the_resume_only_for_a_resume() -> None:
    start = report_mod.capture_start_provenance()
    assert "resume" not in run_provenance(start, llm_worker=None, aut_worker=None, resume=None)
    prov = run_provenance(start, llm_worker=None, aut_worker=None, resume=_resume(None, None, {}, loaded=False))
    assert prov["resume"]["resume_loaded"] is False
    assert inspect.signature(run_provenance).parameters["resume"].default is inspect.Parameter.empty


# ── the orchestrator's wiring ────────────────────────────────────────────


def _start_simulation_mode() -> ast.FunctionDef:
    tree = ast.parse(ORCHESTRATOR.read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "start_simulation_mode")


def test_the_sim_records_what_its_resume_did() -> None:
    fn = _start_simulation_mode()
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    (restore,) = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "_restore_aut_from_session"]
    assert ast.unparse(restore.args[0]) == "resume_session"
    (context,) = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "_load_resume_context_at"]
    updates = [
        c
        for c in calls
        if isinstance(c.func, ast.Attribute)
        and c.func.attr == "update"
        and ast.unparse(c.func.value) == "resume_record"
    ]
    assert {k.arg for k in updates[0].keywords} == {"context_dir", "context_loaded"}
    assert not any(isinstance(c.func, ast.Name) and c.func.id == "_load_resume_context" for c in calls), (
        "the prompt must use the path-returning loader, so the two resolutions can be compared"
    )
