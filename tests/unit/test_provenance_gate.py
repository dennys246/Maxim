"""scripts/_provenance.py gated-record refusal + the guarded JsonlLog (roadmap 1.1.x item 16.7).

Verified to fail on the pre-fix tree: `preflight_gated_record` / `in_process_code_provenance`
did not exist (AttributeError) and `JsonlLog` opened any path unconditionally.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import _provenance as P


def _git(root: Path, *args: str) -> None:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "src/maxim").mkdir(parents=True)
    (tmp_path / "src/maxim/__init__.py").write_text("")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/h.py").write_text("")
    (tmp_path / "docs/experiments/data").mkdir(parents=True)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "clean")
    return tmp_path


def _dirty(repo: Path) -> None:
    (repo / "scripts/h.py").write_text("# edited, uncommitted\n")


def test_clean_tree_gated_write_is_allowed(repo: Path) -> None:
    gate = P.preflight_gated_record(repo, repo / "docs/experiments/data/x.jsonl")
    assert gate == {"gated": True, "working_tree_dirty_src_scripts": False, "allow_dirty": False}


def test_dirty_tree_gated_write_is_refused(repo: Path) -> None:
    _dirty(repo)
    with pytest.raises(P.DirtyTreeError, match="refusing to write a GATED record"):
        P.preflight_gated_record(repo, repo / "docs/experiments/data/x.jsonl")


def test_dirty_tree_non_gated_write_is_reported_not_refused(repo: Path, tmp_path_factory) -> None:
    _dirty(repo)
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "log.jsonl"
    gate = P.preflight_gated_record(repo, elsewhere)
    assert gate == {"gated": False, "working_tree_dirty_src_scripts": True, "allow_dirty": False}


def test_allow_dirty_grants_and_is_reported_only_when_needed(repo: Path) -> None:
    _dirty(repo)
    gate = P.preflight_gated_record(repo, repo / "docs/experiments/data/x.jsonl", allow_dirty=True)
    assert gate["allow_dirty"] is True and gate["working_tree_dirty_src_scripts"] is True
    # A clean tree needs no allowance — the flag must not claim one it never used.
    _git(repo, "checkout", "--", "scripts/h.py")
    gate = P.preflight_gated_record(repo, repo / "docs/experiments/data/x.jsonl", allow_dirty=True)
    assert gate["allow_dirty"] is False


def test_or_exit_variant_exits_3(repo: Path) -> None:
    _dirty(repo)
    with pytest.raises(SystemExit) as ei:
        P.preflight_gated_record_or_exit(repo, repo / "docs/experiments/data/x.jsonl")
    assert ei.value.code == 3


def test_dirty_scope_is_src_and_scripts_only(repo: Path) -> None:
    (repo / "docs/notes.md").write_text("docs edits do not make the code-under-test unestablishable\n")
    assert P.working_tree_dirty(repo) is False


def test_git_failure_counts_as_dirty(tmp_path: Path) -> None:
    assert P.working_tree_dirty(tmp_path / "not-a-repo") is True


def test_in_process_provenance_refuses_foreign_maxim(repo: Path, tmp_path_factory) -> None:
    foreign = tmp_path_factory.mktemp("other") / "maxim/__init__.py"
    with pytest.raises(P.ProvenanceError, match="not this repo's src"):
        P.in_process_code_provenance(repo, str(foreign))


def test_in_process_provenance_stamps_dirty_flag_and_allowance(repo: Path) -> None:
    mf = str(repo / "src/maxim/__init__.py")
    prov = P.in_process_code_provenance(repo, mf)
    assert prov["working_tree_dirty_src_scripts"] is False and "allow_dirty" not in prov
    assert prov["executed_maxim_file"] == str((repo / "src/maxim/__init__.py").resolve())
    _dirty(repo)
    with pytest.raises(P.DirtyTreeError):
        P.in_process_code_provenance(repo, mf, out_path=repo / "docs/experiments/data/x.jsonl")
    prov = P.in_process_code_provenance(repo, mf, out_path=repo / "docs/experiments/data/x.jsonl", allow_dirty=True)
    assert prov["allow_dirty"] is True and prov["working_tree_dirty_src_scripts"] is True


# ── the guarded writer ────────────────────────────────────────────────────────


def _load_live_common():
    path = P.__file__.replace("_provenance.py", "orient_backbone/live_common.py")
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location("live_common_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_jsonl_log_refuses_gated_path_from_dirty_tree_exit_3(repo: Path, monkeypatch) -> None:
    lc = _load_live_common()
    monkeypatch.setattr(lc, "_REPO_ROOT", str(repo))
    _dirty(repo)
    with pytest.raises(SystemExit) as ei:
        lc.JsonlLog(str(repo / "docs/experiments/data/x.jsonl"))
    assert ei.value.code == 3
    assert not (repo / "docs/experiments/data/x.jsonl").exists()


def test_jsonl_log_allow_dirty_stamps_every_record(repo: Path, monkeypatch) -> None:
    lc = _load_live_common()
    monkeypatch.setattr(lc, "_REPO_ROOT", str(repo))
    _dirty(repo)
    out = repo / "docs/experiments/data/x.jsonl"
    log = lc.JsonlLog(str(out), allow_dirty=True)
    log.write("start", a=1)
    log.write("trial", b=2)
    log.close()
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["allow_dirty"] for r in recs] == [True, True]
    assert all("ts" in r for r in recs)


def test_jsonl_log_non_gated_path_never_refuses_or_stamps(repo: Path, tmp_path_factory, monkeypatch) -> None:
    lc = _load_live_common()
    monkeypatch.setattr(lc, "_REPO_ROOT", str(repo))
    _dirty(repo)
    out = tmp_path_factory.mktemp("tmp") / "scratch.jsonl"
    log = lc.JsonlLog(str(out))
    log.write("start")
    log.close()
    rec = json.loads(out.read_text().splitlines()[0])
    assert "allow_dirty" not in rec and log.gated is False


def test_jsonl_log_truncate_mode(repo: Path, tmp_path_factory, monkeypatch) -> None:
    lc = _load_live_common()
    monkeypatch.setattr(lc, "_REPO_ROOT", str(repo))
    out = tmp_path_factory.mktemp("tmp") / "w.jsonl"
    out.write_text("stale\n")
    log = lc.JsonlLog(str(out), mode="w")
    log.write("header")
    log.close()
    assert [json.loads(line)["event"] for line in out.read_text().splitlines()] == ["header"]
    with pytest.raises(ValueError):
        lc.JsonlLog(str(out), mode="r+")


def test_executed_code_provenance_stamps_dirty_flag_and_refuses_gated_dirty_write(repo: Path, monkeypatch) -> None:
    monkeypatch.setattr(P, "resolved_maxim_file", lambda binary, timeout=60.0: str(repo / "src/maxim/__init__.py"))
    prov = P.executed_code_provenance(repo, sys.executable)
    assert prov["working_tree_dirty_src_scripts"] is False and "allow_dirty" not in prov
    _dirty(repo)
    with pytest.raises(P.DirtyTreeError):
        P.executed_code_provenance(repo, sys.executable, out_path=repo / "docs/experiments/data/x.jsonl")
    prov = P.executed_code_provenance(
        repo, sys.executable, out_path=repo / "docs/experiments/data/x.jsonl", allow_dirty=True
    )
    assert prov["allow_dirty"] is True and prov["working_tree_dirty_src_scripts"] is True
