"""Positive control for ``scripts/lint_loop_stack.py`` (docs/plans/loop_stack_builder.md, "The guard").

A fixture per row of the note's evasion table, the six ledger rules, the exit-2 cases, the pinned constants, and a
deletion probe on the real tree (a direct loop call re-added to talk fails).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import lint_loop_stack as L  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

_AGENT_LOOP = """
def run_agentic_loop(*a, **k):
    _run_agentic_loop_body(*a, **k)


def _run_agentic_loop_body(*a, **k):
    pass


def run_agent_loop():
    pass
"""
_BOOTSTRAP = """
LOOP_PARAMETERS = {"agent": "stack", "pain_bus": "stack", "thought_gate": "stack", "max_steps": "run"}


def run_loop_stack(**k):
    from maxim.runtime import agent_loop

    agent_loop.run_agentic_loop()
"""


def _tree(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    root = tmp_path / "repo"
    (root / "src/maxim/runtime").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "src/maxim/runtime/agent_loop.py").write_text(_AGENT_LOOP)
    (root / "src/maxim/runtime/bootstrap.py").write_text(_BOOTSTRAP)
    (root / "scripts/tool.py").write_text("x = 1\n")
    for rel, text in (files or {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
    return root


def _sites(tmp_path: Path, src: str, rel: str = "src/maxim/site.py") -> tuple[list[str], list[str]]:
    sites, builder = L.measure(_tree(tmp_path, {rel: src}))
    return [s.key for s in sites], [b.key for b in builder]


# ── counted sites: the evasion table ─────────────────────────────────────────────────────────────────────


def test_the_minimal_tree_has_no_site_and_no_finding(tmp_path: Path) -> None:
    assert L.measure(_tree(tmp_path)) == ([], [])


@pytest.mark.parametrize(
    ("label", "src"),
    [
        ("direct call", "from maxim.runtime.agent_loop import run_agentic_loop\ndef f():\n    run_agentic_loop()\n"),
        ("alias", "from maxim.runtime.agent_loop import run_agentic_loop as ral\ndef f():\n    ral()\n"),
        ("attribute", "from maxim.runtime import agent_loop as al\ndef f():\n    al.run_agentic_loop()\n"),
        (
            "thread target",
            "import threading\nfrom x import run_agentic_loop\ndef f():\n    threading.Thread(target=run_agentic_loop)\n",
        ),
        (
            "partial",
            "import functools\nfrom x import run_agentic_loop\ndef f():\n    functools.partial(run_agentic_loop)\n",
        ),
        ("dict of callables", "from x import run_agentic_loop\ndef f():\n    return {'go': run_agentic_loop}\n"),
        ("getattr string", "import x\ndef f():\n    getattr(x, 'run_agentic_loop')()\n"),
        ("importlib / _LAZY key", "def f():\n    return {'run_agent_loop': 'maxim.runtime.agent_loop'}\n"),
        ("the body", "from x import _run_agentic_loop_body\ndef f():\n    _run_agentic_loop_body()\n"),
        ("the old loop", "from x import run_agent_loop\ndef f():\n    run_agent_loop()\n"),
        ("a new wrapper", "def my_loop(**kw):\n    from x import run_agentic_loop\n    run_agentic_loop(**kw)\n"),
    ],
)
def test_each_way_to_reach_the_loop_is_a_counted_site(tmp_path: Path, label: str, src: str) -> None:
    sites, _ = _sites(tmp_path, src)
    assert sites, f"{label}: not counted"
    assert all(s.startswith("src/maxim/site.py::") for s in sites)


def test_a_lambda_counts_toward_its_enclosing_def_and_a_closure_toward_the_inner_def(tmp_path: Path) -> None:
    src = """
    from x import run_agentic_loop
    def outer():
        cb = lambda: run_agentic_loop()
        def inner():
            run_agentic_loop()
    """
    sites, _ = _sites(tmp_path, src)
    assert sorted(sites) == ["src/maxim/site.py::outer", "src/maxim/site.py::outer.<locals>.inner"]


def test_a_docstring_naming_the_loop_and_an_import_alone_are_not_sites(tmp_path: Path) -> None:
    src = '''
    """run_agentic_loop"""
    from x import run_agentic_loop
    def f():
        """run_agentic_loop"""
        return "run_agentic_loop(...) is a sim: pass response_policy="
    '''
    assert _sites(tmp_path, src) == ([], [])


def test_scripts_are_in_scope_and_tests_and_docs_are_not(tmp_path: Path) -> None:
    call = "from x import run_agentic_loop\ndef f():\n    run_agentic_loop()\n"
    root = _tree(tmp_path, {"scripts/harness.py": call, "tests/unit/test_x.py": call, "docs/ex.py": call})
    sites, _ = L.measure(root)
    assert [s.key for s in sites] == ["scripts/harness.py::f"]


def test_the_sanctioned_caller_and_the_wrappers_body_call_are_exempt_and_nothing_else_in_agent_loop(
    tmp_path: Path,
) -> None:
    extra = _AGENT_LOOP + "\ndef helper():\n    run_agentic_loop()\n"
    root = _tree(tmp_path, {"src/maxim/runtime/agent_loop.py": extra})
    sites, _ = L.measure(root)
    assert [s.key for s in sites] == ["src/maxim/runtime/agent_loop.py::helper"]
    more = _BOOTSTRAP + "\ndef other():\n    from maxim.runtime import agent_loop\n    agent_loop.run_agentic_loop()\n"
    root2 = _tree(tmp_path / "b", {"src/maxim/runtime/bootstrap.py": more})
    assert [s.key for s in L.measure(root2)[0]] == ["src/maxim/runtime/bootstrap.py::other"]


def test_the_lints_own_name_constants_are_exempt_only_at_module_level(tmp_path: Path) -> None:
    ok = "NAMES = ('run_agentic_loop', '_run_agentic_loop_body', 'run_agent_loop')\nBUILDER_NAME = 'run_loop_stack'\n"
    assert L.measure(_tree(tmp_path, {L.SELF: ok})) == ([], [])
    bad = ok + "def f():\n    NAMES = ('run_agentic_loop',)\n    cb = lambda: 0\n"
    sites, _ = L.measure(_tree(tmp_path / "b", {L.SELF: bad}))
    assert [s.key for s in sites] == [f"{L.SELF}::f"]
    main = ok + "if __name__ == '__main__':\n    from x import run_agentic_loop\n    run_agentic_loop()\n"
    sites, _ = L.measure(_tree(tmp_path / "c", {L.SELF: main}))
    assert [s.key for s in sites] == [f"{L.SELF}::<module>"]
    # the same constant names in ANOTHER file are not exempt
    sites, _ = L.measure(_tree(tmp_path / "d", {"scripts/other.py": ok}))
    assert len(sites) == 3


# ── run_loop_stack: direct, splat-free calls only ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "src",
    [
        "from maxim.runtime.bootstrap import run_loop_stack\ndef f():\n    run_loop_stack(agent=1)\n",
        "from maxim.runtime.bootstrap import run_loop_stack as rls\ndef f():\n    rls(agent=1)\n",
        "from maxim.runtime import bootstrap\ndef f():\n    bootstrap.run_loop_stack(agent=1)\n",
    ],
)
def test_a_splat_free_direct_call_is_allowed_by_name_alias_or_attribute(tmp_path: Path, src: str) -> None:
    assert _sites(tmp_path, src) == ([], [])


@pytest.mark.parametrize(
    ("label", "src"),
    [
        ("a ** splat", "from b import run_loop_stack\ndef f(kw):\n    run_loop_stack(**kw)\n"),
        ("a * splat", "from b import run_loop_stack\ndef f(a):\n    run_loop_stack(*a)\n"),
        ("an alias", "from b import run_loop_stack\ndef f(kw):\n    g = run_loop_stack\n    g(**kw)\n"),
        (
            "partial",
            "import functools\nfrom b import run_loop_stack\ndef f(kw):\n    functools.partial(run_loop_stack, **kw)()\n",
        ),
        (
            "partial on an import alias",
            "import functools\nfrom b import run_loop_stack as rls\ndef f(kw):\n    functools.partial(rls, **kw)()\n",
        ),
        (
            "partial on an attribute",
            "import functools\nfrom maxim.runtime import bootstrap\ndef f(kw):\n    functools.partial(bootstrap.run_loop_stack, **kw)()\n",
        ),
        (
            "a thread target",
            "import threading\nfrom b import run_loop_stack\ndef f(kw):\n    threading.Thread(target=run_loop_stack, kwargs=kw)\n",
        ),
        ("a dict of callables", "from b import run_loop_stack\ndef f():\n    return {'go': run_loop_stack}\n"),
        ("a getattr string", "import b\ndef f():\n    getattr(b, 'run_loop_stack')(agent=1)\n"),
    ],
)
def test_any_other_use_of_run_loop_stack_is_a_finding(tmp_path: Path, label: str, src: str) -> None:
    sites, builder = _sites(tmp_path, src)
    assert builder == ["src/maxim/site.py::f"], label
    assert sites == []


# ── a wrapper that defaults a stack component (review round, exec SF) ─────────────────────────────────


_CALL = "    run_loop_stack(agent=1, pain_bus=pain_bus, max_steps=3)\n"


@pytest.mark.parametrize(
    ("label", "src"),
    [
        ("a positional default", "from b import run_loop_stack\ndef f(a, pain_bus=None):\n" + _CALL),
        ("a keyword-only default", "from b import run_loop_stack\ndef f(*, pain_bus=None):\n" + _CALL),
        (
            "a nested def's outer default",
            "from b import run_loop_stack\ndef f(pain_bus=None):\n    def inner():\n    " + _CALL + "    inner()\n",
        ),
        ("a positional-only default", "from b import run_loop_stack\ndef f(pain_bus=None, /):\n" + _CALL),
    ],
)
def test_a_stack_value_from_a_defaulted_parameter_is_a_finding(tmp_path: Path, label: str, src: str) -> None:
    sites, builder = _sites(tmp_path, src)
    assert sites == []
    assert builder, f"{label}: a wrapper defaulting pain_bus passed"


@pytest.mark.parametrize(
    ("label", "src"),
    [
        ("no default: the requirement moves up", "from b import run_loop_stack\ndef f(pain_bus):\n" + _CALL),
        (
            "a RUN keyword from a default",
            "from b import run_loop_stack\ndef f(pain_bus, max_steps=3):\n    run_loop_stack(agent=1, pain_bus=pain_bus, max_steps=max_steps)\n",
        ),
        (
            "an inner rebinding parameter without a default shadows the outer default",
            "from b import run_loop_stack\ndef f(pain_bus=None):\n    def inner(pain_bus):\n    " + _CALL,
        ),
        ("a local, not a parameter", "from b import run_loop_stack\ndef f():\n    pain_bus = None\n" + _CALL),
    ],
)
def test_a_stack_value_not_from_a_default_is_allowed(tmp_path: Path, label: str, src: str) -> None:
    assert _sites(tmp_path, src) == ([], []), label


def test_the_stack_names_are_read_from_bootstraps_loop_parameters() -> None:
    from maxim.runtime.bootstrap import LOOP_PARAMETERS

    bootstrap = L._parse_rel(REPO, L.BOOTSTRAP)
    expected = {k for k, v in LOOP_PARAMETERS.items() if v == "stack"} | {"acting_coach", "is_embodied"}
    assert L.stack_names(bootstrap) == expected
    assert L._STATEMENTS == ("acting_coach", "is_embodied")


# ── the ledger: rules 1, 2 and 6 (HEAD) ──────────────────────────────────────────────────────────────────


def _entry(file: str = "src/maxim/site.py", qualname: str = "f", count: int = 1, **over: Any) -> dict[str, Any]:
    e: dict[str, Any] = {"file": file, "qualname": qualname, "count": count, "ref": "#610", "reason": "r"}
    e["permanent"] = False
    e.update(over)
    return e


def _site(qualname: str = "f", file: str = "src/maxim/site.py") -> Any:
    return L.Site(file, qualname, 1, "a load of run_agentic_loop")


def test_rule_1_measured_equality_and_the_orphan() -> None:
    assert L.head_rules([_entry()], [_site()])[0] == []
    [f] = L.head_rules([_entry(count=2)], [_site()])[0]
    assert "strict equality" in f
    [f] = L.head_rules([_entry()], [])[0]
    assert "orphan" in f


def test_rule_2_an_unlisted_site_fails() -> None:
    [f] = L.head_rules([], [_site()])[0]
    assert "unlisted" in f and "run_loop_stack" in f


def test_the_report_prints_every_entry_and_marks_permanent_ones() -> None:
    _, report = L.head_rules([_entry(permanent=True)], [_site()])
    assert any("non-shrinking" in line for line in report)


@pytest.mark.parametrize(
    "bad",
    [
        "{not json",
        "{}",
        json.dumps([_entry(count=0)]),
        json.dumps([_entry(count=True)]),
        json.dumps([_entry(ref="issue 610")]),
        json.dumps([_entry(reason=" ")]),
        json.dumps([_entry(permanent="no")]),
        json.dumps([_entry(moved_from={"file": "a"})]),
        json.dumps([{k: v for k, v in _entry().items() if k != "reason"}]),
        json.dumps([_entry(extra=1)]),
        json.dumps([_entry(), _entry()]),
    ],
)
def test_rule_6_a_malformed_ledger_is_a_ledger_error(bad: str) -> None:
    with pytest.raises(L.LedgerError):
        L.parse_ledger(bad, where="t")


# ── rules 3 and 4 (against the base) ─────────────────────────────────────────────────────────────────────


def test_rule_3_shrink_only() -> None:
    assert L.base_rules([_entry(), _entry(qualname="g")], [_entry()]) == []  # a migrated site's entry is gone
    [f] = L.base_rules([_entry()], [_entry(count=2)])
    assert "rises" in f
    [f] = L.base_rules([_entry()], [_entry(), _entry(qualname="new")])
    assert "no moved_from" in f
    [f] = L.base_rules([_entry()], [_entry(permanent=True)])
    assert "permanent" in f
    assert L.base_rules([], []) == []


def test_rule_4_a_valid_move() -> None:
    src = {"file": "src/maxim/site.py", "qualname": "f"}
    assert L.base_rules([_entry()], [_entry(file="src/maxim/new.py", qualname="g", moved_from=src)]) == []
    # an entry that landed as a move keeps its moved_from on the next branch: it is at base now, so it is judged
    # by rule 3 alone
    moved = _entry(file="src/maxim/new.py", qualname="g", moved_from=src)
    assert L.base_rules([moved], [moved]) == []


@pytest.mark.parametrize(
    ("base", "head", "needle"),
    [
        (
            [_entry()],
            [_entry(), _entry(qualname="g", moved_from={"file": "src/maxim/site.py", "qualname": "f"})],
            "still present",
        ),
        (
            [_entry()],
            [_entry(qualname="g", moved_from={"file": "src/maxim/x.py", "qualname": "nope"})],
            "not a base entry",
        ),
        (
            [_entry()],
            [_entry(qualname="g", count=2, moved_from={"file": "src/maxim/site.py", "qualname": "f"})],
            "above its source",
        ),
        (
            [_entry(permanent=True)],
            [_entry(qualname="g", moved_from={"file": "src/maxim/site.py", "qualname": "f"})],
            "keeps its source's permanent",
        ),
        (
            [_entry(count=2)],
            [
                _entry(qualname="g", moved_from={"file": "src/maxim/site.py", "qualname": "f"}),
                _entry(qualname="h", moved_from={"file": "src/maxim/site.py", "qualname": "f"}),
            ],
            "no splits",
        ),
    ],
)
def test_rule_4_every_bad_move_fails(base: list[Any], head: list[Any], needle: str) -> None:
    failures = L.base_rules(base, head)
    assert any(needle in f for f in failures), failures


# ── rule 5, the base ref, and exit 2: end to end in a git repo ───────────────────────────────────────────


def _git(root: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _repo(tmp_path: Path, *, ledger_at_base: list[Any] | None) -> Path:
    root = _tree(tmp_path)
    _git(root, "init", "-q", "-b", "main")
    if ledger_at_base is not None:
        (root / L.LEDGER).write_text(json.dumps(ledger_at_base))
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "checkout", "-q", "-b", "feature")
    return root


@pytest.fixture(autouse=True)
def _no_ci_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)


def test_rule_5_a_ledger_added_by_the_branch_passes(tmp_path: Path) -> None:
    root = _repo(tmp_path, ledger_at_base=None)
    (root / L.LEDGER).write_text("[]")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add the ledger")
    failures, report = L.check(root)
    assert failures == [] and any("added by this branch" in r for r in report)


def test_rule_5_a_ledger_absent_at_base_and_not_added_fails(tmp_path: Path) -> None:
    root = _repo(tmp_path, ledger_at_base=None)
    (root / L.LEDGER).write_text("[]")  # present in the work tree, never committed
    failures, _ = L.check(root)
    assert any("rule 5" in f for f in failures)


def test_the_base_rules_run_end_to_end(tmp_path: Path) -> None:
    root = _repo(tmp_path, ledger_at_base=[])
    site = "from x import run_agentic_loop\ndef f():\n    run_agentic_loop()\n"
    (root / "src/maxim/site.py").write_text(site)
    (root / L.LEDGER).write_text(json.dumps([_entry()]))
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "a new hand-assembled site")
    failures, _ = L.check(root)
    assert any("no moved_from" in f for f in failures), failures


def test_no_base_ref_skips_locally_and_fails_on_a_pull_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _tree(tmp_path)
    (root / L.LEDGER).write_text("[]")
    _git(root, "init", "-q", "-b", "lonely")  # no main, no origin/main: base_ref raises
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "x")
    failures, report = L.check(root)
    assert failures == [] and any("skipped locally" in r for r in report)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    with pytest.raises(L.NoVerdict):  # no verdict: exit 2, as the sibling diff-scoped lints (review round, arch SF)
        L.check(root)
    monkeypatch.setattr(L, "REPO_ROOT", root)
    assert L.main() == 2


def test_git_failing_in_rule_5_is_no_verdict_on_a_pull_request_and_a_skip_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path, ledger_at_base=None)
    (root / L.LEDGER).write_text("[]")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add the ledger")

    def _boom(*_a: Any, **_k: Any) -> str:
        raise L.GitUnavailable("diff failed")

    monkeypatch.setattr(L, "git", _boom)
    failures, report = L.check(root)
    assert failures == [] and any("skipped locally" in r for r in report)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    with pytest.raises(L.NoVerdict):
        L.check(root)
    monkeypatch.setattr(L, "REPO_ROOT", root)
    assert L.main() == 2  # never a traceback


def test_the_head_rules_run_even_without_a_base(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"src/maxim/site.py": "from x import run_agentic_loop\ndef f():\n    run_agentic_loop()\n"})
    (root / L.LEDGER).write_text("[]")
    _git(root, "init", "-q", "-b", "lonely")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "x")
    failures, _ = L.check(root)
    assert any("unlisted" in f for f in failures)


@pytest.mark.parametrize(
    "breakage", ["no ledger", "bad ledger", "no agent_loop", "no bootstrap def", "no scripts", "no LOOP_PARAMETERS"]
)
def test_exit_2_cases(tmp_path: Path, breakage: str) -> None:
    root = _tree(tmp_path)
    (root / L.LEDGER).write_text("[]")
    if breakage == "no ledger":
        (root / L.LEDGER).unlink()
    elif breakage == "bad ledger":
        (root / L.LEDGER).write_text("{")
    elif breakage == "no agent_loop":
        (root / "src/maxim/runtime/agent_loop.py").unlink()
    elif breakage == "no bootstrap def":
        (root / "src/maxim/runtime/bootstrap.py").write_text('LOOP_PARAMETERS = {"agent": "stack"}\n')
    elif breakage == "no LOOP_PARAMETERS":
        (root / "src/maxim/runtime/bootstrap.py").write_text("def run_loop_stack(**k):\n    pass\n")
    else:
        shutil.rmtree(root / "scripts")
    with pytest.raises((L.ScopeError, L.LedgerError)):
        L.check(root)


def test_main_returns_2_on_a_broken_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    assert L.main() == 2


# ── the pinned constants and the real tree ───────────────────────────────────────────────────────────────


def test_the_lints_constants_are_pinned() -> None:
    assert L.NAMES == ("run_agentic_loop", "_run_agentic_loop_body", "run_agent_loop")
    assert L.BUILDER_NAME == "run_loop_stack"
    assert L.SANCTIONED_QUALNAME == "src/maxim/runtime/bootstrap.py::run_loop_stack"
    assert L.ROOTS == ("src/maxim", "scripts")
    assert L.LEDGER == "scripts/loop_stack_ledger.json"
    assert L._SELF_CONSTANTS == frozenset({"NAMES", "BUILDER_NAME", "SANCTIONED_QUALNAME"})


def test_the_real_tree_measures_exactly_its_ledger() -> None:
    sites, builder = L.measure(REPO)
    head = L.parse_ledger((REPO / L.LEDGER).read_text(encoding="utf-8"), where=L.LEDGER)
    failures, _ = L.head_rules(head, sites)
    assert failures == [] and builder == []
    assert "src/maxim/console/handle.py" not in {s.file for s in sites}, "talk is migrated: it holds no direct site"


def test_deletion_probe_a_direct_loop_call_re_added_to_talk_fails(tmp_path: Path) -> None:
    root = tmp_path / "copy"
    for top in ("src/maxim", "scripts"):
        shutil.copytree(REPO / top, root / top, ignore=shutil.ignore_patterns("__pycache__"))
    handle = root / "src/maxim/console/handle.py"
    handle.write_text(
        handle.read_text(encoding="utf-8")
        + "\n\ndef _probe():\n    from maxim.runtime.agent_loop import run_agentic_loop\n\n    run_agentic_loop()\n"
    )
    sites, _ = L.measure(root)
    head = L.parse_ledger((root / L.LEDGER).read_text(encoding="utf-8"), where=L.LEDGER)
    failures, _ = L.head_rules(head, sites)
    assert any("console/handle.py" in f and "unlisted" in f for f in failures), failures
