"""M1 -- a sim report carries the provenance an operator used to attest by hand.

``report.json`` stamps which code the run imported (commit, clean tree over ``src/``+``scripts/``,
``maxim.__file__``, interpreter), the configured context window, and per LLM role the router's profile,
its configured window and the worker's clamped prompt budget; plus ``ts``, the run's start in epoch
seconds. The keys follow ``scripts/_provenance.py``, so the prereg lint reads them as it reads a
harness's own ``provenance`` block.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import maxim
from maxim.agents.llm_worker import LLMWorker
from maxim.simulation import report as report_mod
from maxim.simulation.report import build_report, capture_start_provenance, run_provenance, save_report

REPO = Path(maxim.__file__).resolve().parents[2]
ORCHESTRATOR = REPO / "src" / "maxim" / "simulation" / "orchestrator.py"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ── the code stamp ───────────────────────────────────────────────────────


def test_the_code_stamp_agrees_with_the_harness_provenance_helper() -> None:
    """Same vocabulary, scope and answers as ``scripts/_provenance.py`` for this checkout."""
    prov = _load_script("_provenance")
    assert report_mod._DIRTY_SCOPE == tuple(prov.DIRTY_SCOPE)
    stamp = capture_start_provenance()
    head = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    assert stamp["executed_maxim_file"] == str(Path(maxim.__file__).resolve())
    assert stamp["executed_git_hash"] == head
    assert stamp["working_tree_dirty_src_scripts"] is prov.working_tree_dirty(REPO)
    assert stamp["python"] == sys.executable
    assert stamp["maxim_version"] == maxim.__version__


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_a_maxim_installed_inside_some_other_repo_claims_no_commit(tmp_path, monkeypatch) -> None:
    """The dangerous layout: a venv's site-packages inside a clean, committed repo. Asking git there would
    stamp that repo's commit as a clean run of code that never came from it."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "README").write_text("x")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "-q", "-m", "init")
    fake = tmp_path / "lib" / "maxim" / "__init__.py"  # parents[2] is the repo root, with a .git
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    monkeypatch.setattr(maxim, "__file__", str(fake))
    stamp = capture_start_provenance()
    assert stamp["executed_git_hash"] == "unknown"
    assert stamp["working_tree_dirty_src_scripts"] is True


def test_a_git_failure_is_unknown_and_dirty(monkeypatch) -> None:
    """Unknown must never read as clean (``scripts/_provenance.py``'s rule)."""

    def failing(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 128, stdout="deadbeef\n M src/x.py\n", stderr="fatal")

    from maxim.utils import code_tree

    monkeypatch.setattr(code_tree.subprocess, "run", failing)
    stamp = capture_start_provenance()
    assert stamp["executed_git_hash"] == "unknown"
    assert stamp["working_tree_dirty_src_scripts"] is True


def test_the_start_stamp_records_the_configured_context(monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_LLM_N_CTX", "8192")
    stamp = capture_start_provenance()
    assert (stamp["configured_n_ctx"], stamp["configured_n_ctx_source"]) == (8192, "env")
    monkeypatch.setenv("MAXIM_LLM_N_CTX", "not-a-number")
    stamp = capture_start_provenance()  # a malformed value must not cost the run its start
    assert stamp["configured_n_ctx"] is None
    assert stamp["configured_n_ctx_source"].startswith("unresolved")


def test_the_code_digest_is_the_same_on_both_sides_and_names_the_exact_code(tmp_path) -> None:
    """The binding between a dirty sim report and the harness record that allowed it (M1) compares this
    digest, so its two implementations must agree byte for byte, and it must move with the code."""
    prov = _load_script("_provenance")
    for d in ("src", "scripts", "docs"):
        (tmp_path / d).mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    clean = report_mod._code_tree_sha256(tmp_path)
    assert clean == prov.code_tree_sha256(tmp_path)

    (tmp_path / "src" / "a.py").write_text("x = 2\n")  # tracked edit
    (tmp_path / "scripts" / "new.py").write_bytes(b"\x00binary\xff")  # untracked, binary
    (tmp_path / "docs" / "note.md").write_text("outside the scope")
    dirty = report_mod._code_tree_sha256(tmp_path)
    assert dirty == prov.code_tree_sha256(tmp_path)
    assert dirty != clean

    (tmp_path / "docs" / "note.md").write_text("still outside the scope")
    assert report_mod._code_tree_sha256(tmp_path) == dirty  # docs/ is not code
    (tmp_path / "scripts" / "new.py").write_bytes(b"\x00binary\xfe")
    assert report_mod._code_tree_sha256(tmp_path) != dirty  # the untracked file's bytes count
    assert report_mod._code_tree_sha256(tmp_path / "docs" / "missing") == "unknown"


def _dirty_fixture(tmp_path: Path) -> Path:
    for d in ("src", "scripts"):
        (tmp_path / d).mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "src" / "gone.py").write_text("y = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    (tmp_path / "src" / "gone.py").unlink()
    return tmp_path


def test_no_git_config_moves_the_digest_or_hides_dirty_code(tmp_path) -> None:
    """Round-2 blocker: a diff-TEXT digest let ``diff.external`` hash every dirty tree as clean."""
    repo = _dirty_fixture(tmp_path)
    plain = report_mod._code_tree_sha256(repo)
    for key, value in (
        ("diff.external", "true"),
        ("core.abbrev", "4"),
        ("color.ui", "always"),
        ("diff.noprefix", "true"),
        ("diff.renames", "false"),
    ):
        _git(repo, "config", key, value)
    assert report_mod._code_tree_sha256(repo) == plain
    _git(repo, "checkout", "-q", "--", "src/gone.py")  # the deletion alone must count
    assert report_mod._code_tree_sha256(repo) != plain
    _git(repo, "stash", "-u", "-q")  # now clean, under the same config
    assert report_mod._code_tree_sha256(repo) != plain


@pytest.mark.parametrize("hide", ["excludes_file", "assume_unchanged", "skip_worktree", "file_mode_off"])
def test_no_git_config_or_index_state_hides_a_change(tmp_path, hide) -> None:
    """Round 3: asking git WHAT changed let per-user excludes, index flags and core.fileMode hide code."""
    import os

    repo = _dirty_fixture(tmp_path)
    _git(repo, "checkout", "-q", "--", ".")  # clean
    before = report_mod._code_tree_sha256(repo)
    if hide == "excludes_file":
        (tmp_path / "global_ignore").write_text("*.py\n")
        _git(repo, "config", "core.excludesFile", str(tmp_path / "global_ignore"))
        (repo / "src" / "new.py").write_text("print('hidden?')\n")
    elif hide in ("assume_unchanged", "skip_worktree"):
        _git(repo, "update-index", f"--{hide.replace('_', '-')}", "src/a.py")
        (repo / "src" / "a.py").write_text("x = 3\n")
    else:
        _git(repo, "config", "core.fileMode", "false")
        os.chmod(repo / "src" / "a.py", 0o755)
    assert report_mod._code_tree_sha256(repo) != before


def test_an_untracked_gitignore_that_ignores_itself_still_moves_the_digest(tmp_path) -> None:
    """Round 4: ``src/pkg/.gitignore`` holding ``*`` hid itself and the code beside it."""
    repo = _dirty_fixture(tmp_path)
    before = report_mod._code_tree_sha256(repo)
    (repo / "src" / "pkg").mkdir()
    (repo / "src" / "pkg" / ".gitignore").write_text("*\n")
    (repo / "src" / "pkg" / "evil.py").write_text("print('hidden')\n")
    assert report_mod._code_tree_sha256(repo) != before


def test_the_root_gitignore_is_hashed_with_the_code(tmp_path) -> None:
    """Round 5: ignoring new code from the root .gitignore hid it one level up."""
    repo = _dirty_fixture(tmp_path)
    (repo / ".gitignore").write_text("")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "gitignore")
    before = report_mod._code_tree_sha256(repo)
    (repo / ".gitignore").write_text("src/evil.py\n")
    (repo / "src" / "evil.py").write_text("print('hidden')\n")
    assert report_mod._code_tree_sha256(repo) != before


def test_an_untracked_root_gitignore_that_ignores_itself_still_moves_the_digest(tmp_path) -> None:
    """Round 6: with no root .gitignore in HEAD, an untracked one holding ``*`` hid itself and new code."""
    repo = _dirty_fixture(tmp_path)
    before = report_mod._code_tree_sha256(repo)
    (repo / ".gitignore").write_text("*\n")
    (repo / "src" / "evil.py").write_text("print('hidden')\n")
    assert report_mod._code_tree_sha256(repo) != before


def test_an_inherited_pathspec_variable_cannot_blind_a_listing(tmp_path, monkeypatch) -> None:
    """``GIT_LITERAL_PATHSPECS=1`` turned the untracked-.gitignore glob into a literal that matches nothing."""
    repo = _dirty_fixture(tmp_path)
    before = report_mod._code_tree_sha256(repo)
    (repo / "src" / "pkg").mkdir()
    (repo / "src" / "pkg" / ".gitignore").write_text("*\n")
    (repo / "src" / "pkg" / "evil.py").write_text("print('hidden')\n")
    monkeypatch.setenv("GIT_LITERAL_PATHSPECS", "1")
    assert report_mod._code_tree_sha256(repo) != before


def test_a_leaked_git_location_variable_cannot_point_the_digest_elsewhere(tmp_path, monkeypatch) -> None:
    (tmp_path / "repo").mkdir()
    (tmp_path / "other").mkdir()
    repo = _dirty_fixture(tmp_path / "repo")
    other = _dirty_fixture(tmp_path / "other")
    (other / "src" / "a.py").write_text("something else\n")
    (other / "src" / "extra.py").write_text("only there\n")  # a different path set
    own = report_mod._code_tree_sha256(repo)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert report_mod._code_tree_sha256(repo) == own


def test_a_touch_or_a_restored_edit_leaves_the_digest_alone(tmp_path) -> None:
    import os
    import time

    repo = _dirty_fixture(tmp_path)
    before = report_mod._code_tree_sha256(repo)
    later = time.time() + 5
    os.utime(repo / "src" / "a.py", (later, later))
    assert report_mod._code_tree_sha256(repo) == before
    (repo / "src" / "a.py").write_text("x = 99\n")
    (repo / "src" / "a.py").write_text("x = 2\n")  # back to what it was
    assert report_mod._code_tree_sha256(repo) == before


def test_a_removed_file_counts_as_deleted_not_as_never_there(tmp_path) -> None:
    """``git rm`` takes a file out of the index too; only HEAD still names it."""
    trees = []
    for name, committed in (("had", True), ("never", False)):
        (tmp_path / name).mkdir()
        repo = tmp_path / name
        (repo / "src").mkdir()
        (repo / "src" / "a.py").write_text("x = 1\n")
        if committed:
            (repo / "src" / "gone.py").write_text("y = 1\n")
        _git(repo, "init", "-q")
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", "init")
        if committed:
            _git(repo, "rm", "-q", "src/gone.py")
        trees.append(report_mod._code_tree_sha256(repo))
    assert trees[0] != trees[1]


def test_code_reached_through_a_symlinked_directory_or_a_nested_repo_is_unknown(tmp_path) -> None:
    (tmp_path / "repo").mkdir()
    repo = _dirty_fixture(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_text("x = 1\n")
    moved = tmp_path / "moved_src"
    (repo / "src").rename(moved)
    (repo / "src").symlink_to(outside)  # tracked src/a.py now resolves outside the repo
    assert report_mod._code_tree_sha256(repo) == "unknown"
    (repo / "src").unlink()
    moved.rename(repo / "src")
    nested = repo / "scripts" / "nested"
    nested.mkdir()
    _git(nested, "init", "-q")
    assert report_mod._code_tree_sha256(repo) == "unknown"


def test_an_untracked_symlink_is_hashed_by_its_target_not_followed(tmp_path) -> None:
    repo = _dirty_fixture(tmp_path)
    (repo / "scripts" / "zero").symlink_to("/dev/zero")  # following it would never return
    first = report_mod._code_tree_sha256(repo)
    assert first != "unknown"
    (repo / "scripts" / "zero").unlink()
    (repo / "scripts" / "zero").symlink_to("/dev/null")
    assert report_mod._code_tree_sha256(repo) != first


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="no FIFOs here")
def test_a_path_that_is_neither_file_nor_symlink_is_unknown(tmp_path) -> None:
    import os

    repo = _dirty_fixture(tmp_path)
    os.mkfifo(repo / "scripts" / "pipe")  # git never lists an untracked FIFO: nothing to read
    assert report_mod._code_tree_sha256(repo) != "unknown"
    (repo / "src" / "a.py").unlink()
    os.mkfifo(repo / "src" / "a.py")  # a TRACKED path turned into a FIFO is listed; reading it would block
    assert report_mod._code_tree_sha256(repo) == "unknown"


def test_the_digest_is_framed_so_two_trees_cannot_share_a_stream(tmp_path) -> None:
    """Unframed, a symlink whose target spells out the next record would hash like the two paths:
    ``a -> X`` plus file ``b`` versus ``a -> X + "scripts/b" + "file" + sha256(b)``."""
    import hashlib
    import os

    content = next(c for c in (bytes([i]) for i in range(65, 91)) if 0 not in hashlib.sha256(c).digest())
    repo = _dirty_fixture(tmp_path)
    (repo / "scripts" / "a").symlink_to("X")
    (repo / "scripts" / "b").write_bytes(content)
    two = report_mod._code_tree_sha256(repo)
    (repo / "scripts" / "b").unlink()
    (repo / "scripts" / "a").unlink()
    target = b"X" + b"scripts/b" + b"file" + hashlib.sha256(content).digest()
    os.symlink(target, repo / "scripts" / "a")
    assert report_mod._code_tree_sha256(repo) != two


def test_the_harness_and_the_sim_read_the_same_code_tree_module() -> None:
    """One implementation (#998): ``scripts/_provenance.py`` loads ``utils/code_tree.py`` by path from its
    own tree, and ``report.py`` imports it; the M1 twin pin became this identity."""
    from maxim.utils import code_tree

    prov = _load_script("_provenance")
    assert Path(prov._code_tree.__file__).resolve() == Path(code_tree.__file__).resolve()
    assert report_mod._code_tree_sha256 is code_tree.code_tree_sha256
    assert report_mod._tree_dirty is code_tree.tree_dirty


def test_a_harness_whose_sub_sims_import_another_tree_stamps_no_digest(tmp_path, monkeypatch) -> None:
    """An allowance judged on one tree must never bind to another tree's code."""
    prov = _load_script("_provenance")
    other = tmp_path / "other" / "src" / "maxim" / "__init__.py"
    monkeypatch.setattr(prov, "resolved_maxim_file", lambda binary: str(other))
    stamp = prov.executed_code_provenance(REPO, "maxim")
    assert stamp["code_tree_sha256"] == "unknown"
    monkeypatch.setattr(prov, "resolved_maxim_file", lambda binary: str(REPO / "src" / "maxim" / "__init__.py"))
    assert prov.executed_code_provenance(REPO, "maxim")["code_tree_sha256"] != "unknown"


def test_the_harness_stamp_carries_the_same_digest() -> None:
    prov = _load_script("_provenance")
    harness = prov.in_process_code_provenance(REPO, maxim.__file__)
    assert harness["code_tree_sha256"] == capture_start_provenance()["code_tree_sha256"]


# ── per role, and the end-of-run check ───────────────────────────────────


class _Router:
    def __init__(self, profile: str, n_ctx: int, provider_ctx: int) -> None:
        self.cfg = SimpleNamespace(profile=profile)
        self.n_ctx = n_ctx
        self._provider_ctx = provider_ctx

    def get_provider_configs(self):
        return {"local": {"n_ctx": self._provider_ctx}}


def test_each_role_stamps_the_profile_and_the_budget_it_ran_with() -> None:
    """The budget is the worker's, after the clamp to the smallest declared provider context."""
    language = LLMWorker(llm=_Router("qwen-32b", 32768, 16384), n_ctx=32768)
    aut = LLMWorker(llm=_Router("mistral-7b", 8192, 8192), n_ctx=8192)
    start = capture_start_provenance()
    prov = run_provenance(start, llm_worker=language, aut_worker=aut)
    assert (prov["language_profile"], prov["language_router_n_ctx"], prov["language_budget_n_ctx"]) == (
        "qwen-32b",
        32768,
        16384,
    )
    assert (prov["aut_profile"], prov["aut_budget_n_ctx"]) == ("mistral-7b", 8192)
    assert prov["code_changed_during_run"] is False
    none = run_provenance(start, llm_worker=None, aut_worker=None)
    assert (none["aut_profile"], none["aut_router_n_ctx"], none["aut_budget_n_ctx"]) == (None, None, None)


@pytest.mark.parametrize("key, value", [("executed_git_hash", "000000000000"), ("code_tree_sha256", "0" * 64)])
def test_code_that_moved_during_the_run_is_flagged(key, value) -> None:
    start = {**capture_start_provenance(), key: value}
    assert run_provenance(start, llm_worker=None, aut_worker=None)["code_changed_during_run"] is True


def test_the_run_is_dirty_if_either_stamp_saw_a_dirty_tree(monkeypatch) -> None:
    """Code imported lazily after the start stamp is code too."""
    start = {**capture_start_provenance(), "working_tree_dirty_src_scripts": False}
    monkeypatch.setattr(report_mod, "_tree_dirty", lambda repo: True)  # dirtied during the run
    assert run_provenance(start, llm_worker=None, aut_worker=None)["working_tree_dirty_src_scripts"] is True
    monkeypatch.setattr(report_mod, "_tree_dirty", lambda repo: False)
    dirty_start = {**start, "working_tree_dirty_src_scripts": True}
    assert run_provenance(dirty_start, llm_worker=None, aut_worker=None)["working_tree_dirty_src_scripts"] is True
    assert run_provenance(start, llm_worker=None, aut_worker=None)["working_tree_dirty_src_scripts"] is False
    unknown_start = {k: v for k, v in start.items() if k != "working_tree_dirty_src_scripts"}
    assert run_provenance(unknown_start, llm_worker=None, aut_worker=None)["working_tree_dirty_src_scripts"] is True


def test_the_report_keeps_the_end_stamp_beside_the_start_one() -> None:
    start = {**capture_start_provenance(), "code_tree_sha256": "0" * 64}
    prov = run_provenance(start, llm_worker=None, aut_worker=None)
    assert prov["code_tree_sha256"] == "0" * 64  # the start digest: what a harness stamped at ITS start
    assert prov["end_code_tree_sha256"] == capture_start_provenance()["code_tree_sha256"]
    assert prov["end_executed_git_hash"] == start["executed_git_hash"]


# ── the report ───────────────────────────────────────────────────────────


def test_a_caller_cannot_forget_the_stamps() -> None:
    """Required keyword-only: forgetting them is a TypeError. (Passing empty ones is possible; the
    ledger lint treats an empty stamp as unestablished.)"""
    from maxim.simulation.bridge import SimulationBridge

    params = inspect.signature(build_report).parameters
    for name in ("started_at", "provenance"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        build_report(goal="g", mode="m", bridge=SimulationBridge(), duration_s=1.0, finish_reason="completed")


def test_the_saved_report_carries_the_stamps_the_prereg_lint_reads(tmp_path) -> None:
    from maxim.simulation.bridge import SimulationBridge

    lint = _load_script("lint_prereg_precedes_data")
    stamp = {**capture_start_provenance(), "working_tree_dirty_src_scripts": True}
    report = build_report(
        goal="g",
        mode="m",
        bridge=SimulationBridge(),
        duration_s=1.0,
        finish_reason="completed",
        started_at=1790000000.5,
        provenance=stamp,
    )
    saved = json.loads(Path(save_report(report, base_dir=str(tmp_path))).read_text())
    assert saved["provenance"]["executed_git_hash"] == stamp["executed_git_hash"]
    assert lint._parse_ts(saved["ts"]) == (1790000000.5, False)  # epoch: not naive, the lint's data time
    assert lint._dirty_flag(saved) is True  # the lint sees a dirty run


# ── the sim's wiring (no test runs a sim to its report) ──────────────────


def _start_simulation_mode() -> ast.FunctionDef:
    tree = ast.parse(ORCHESTRATOR.read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "start_simulation_mode")


def _calls(fn: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def test_the_sim_takes_its_code_stamp_before_building_any_router() -> None:
    fn = _start_simulation_mode()
    (capture,) = _calls(fn, "capture_start_provenance")
    routers = _calls(fn, "build_primary_router")
    assert routers, "start_simulation_mode no longer builds a router: re-pin this ordering"
    assert capture.lineno < min(c.lineno for c in routers)
    assigned = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and n.value is capture
        and [ast.unparse(t) for t in n.targets] == ["start_provenance"]
    ]
    assert assigned, "the start stamp must be the value run_provenance receives"


def test_the_sim_passes_its_start_and_its_workers_to_the_report() -> None:
    (call,) = _calls(_start_simulation_mode(), "build_report")
    keywords = {kw.arg: kw.value for kw in call.keywords}
    assert ast.unparse(keywords["started_at"]) == "start_time"
    assert ast.unparse(keywords["provenance"]) == (
        "run_provenance(start_provenance, llm_worker=orch_llm_worker, aut_worker=aut_llm_worker)"
    )
