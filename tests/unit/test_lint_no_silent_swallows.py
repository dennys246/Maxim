"""scripts/lint_no_silent_swallows.py check 3 — no de-instrumentation on the measurement path.

The guard it replaces compared the instrumented-site COUNT to a frozen number. A count cannot tell a
swallow that was DELETED from one that was de-instrumented, so every swallow burn-down (#863) failed
it and had to be re-baselined past — a guard that fires on the fix it protects gets clicked through.
Check 3 counts the UNREPORTED broad swallows instead: deleting a swallow leaves that unchanged,
de-instrumenting one raises it.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from scripts import _lint_git
from scripts import lint_no_silent_swallows as L

_REPORTED = "def f():\n    try:\n        g()\n    except Exception:\n        log_swallowed_exception()\n"
_DEBUG_ONLY = "def f():\n    try:\n        g()\n    except Exception:\n        logger.debug('x')\n"
_DELETED = "def f():\n    g()\n"
_EXPLICIT = (
    "def f():\n    try:\n        g()\n    except Exception as e:\n        log_swallowed_exception(e, operation='x')\n"
)


# ── the counter ──────────────────────────────────────────────────────────────


def test_a_reporting_handler_is_not_counted():
    # Only the Stage-1 forms report. The first version of this test ALSO asserted that the explicit
    # form `log_swallowed_exception(e, operation=...)` reports — it encoded the very bug the review
    # round found; see test_the_explicit_form_is_NOT_a_report.
    assert L.unreported_swallow_hits(_REPORTED) == []
    supplied = _REPORTED.replace("log_swallowed_exception()", "log_swallowed_exception(site='f:g:1')")
    assert L.unreported_swallow_hits(supplied) == []


@pytest.mark.parametrize(
    "handler",
    ["logger.debug('x')", "pass", "return None", "logger.warning('x')"],
)
def test_a_broad_handler_that_keeps_swallowing_without_reporting_is_counted(handler):
    src = f"def f():\n    try:\n        g()\n    except Exception:\n        {handler}\n"
    assert L.unreported_swallow_hits(src) == [4]


def test_bare_except_and_broad_tuples_are_broad():
    assert L.unreported_swallow_hits("try:\n    g()\nexcept:\n    pass\n") == [3]
    assert L.unreported_swallow_hits("try:\n    g()\nexcept (ValueError, Exception):\n    pass\n") == [3]
    assert L.unreported_swallow_hits("try:\n    g()\nexcept BaseException:\n    pass\n") == [3]


def test_a_narrow_handler_is_not_this_checks_business():
    assert L.unreported_swallow_hits("try:\n    g()\nexcept KeyError:\n    pass\n") == []


def test_a_reraising_handler_does_not_swallow():
    assert L.unreported_swallow_hits("try:\n    g()\nexcept Exception:\n    cleanup()\n    raise\n") == []


def test_a_syntax_error_counts_zero():
    assert L.unreported_swallow_hits("def (:\n") == []


# ── the ratchet, on a real git history ───────────────────────────────────────


def _git(root: Path, *args: str) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True).stdout


_PATH = "src/maxim/decisions/nac.py"  # any measurement-path file; the ratchet is scoped to one file


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / _PATH).parent.mkdir(parents=True)
    (tmp_path / _PATH).write_text(_REPORTED)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path, _git(tmp_path, "rev-parse", "HEAD").strip()


def _commit(root: Path, text: str) -> None:
    (root / _PATH).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "change")


def _ratchet(root: Path, base: str) -> list[str]:
    return _lint_git.count_ratchet(root, base, _PATH, L.unreported_swallow_hits, what="unreported broad-swallow count")


def test_de_instrumenting_a_site_FAILS(repo):
    """The documented case: `log_swallowed_exception()` rewritten to `logger.debug(...)`. Checks 1 and
    2 cannot see it (it is handled-and-logged); the site keeps swallowing while its firings stop."""
    root, base = repo
    _commit(root, _DEBUG_ONLY)
    fails = _ratchet(root, base)
    assert len(fails) == 1 and "rose 0 → 1" in fails[0], fails


def test_DELETING_a_swallow_is_free(repo):
    """The case the old count guard got wrong, and the reason for this check: removing the whole
    try/except is what #863's burn-down does, and it must not trip anything."""
    root, base = repo
    _commit(root, _DELETED)
    assert _ratchet(root, base) == []


def test_instrumenting_a_previously_silent_swallow_is_free(repo):
    root, base = repo
    _commit(root, _DEBUG_ONLY)
    base2 = _git(root, "rev-parse", "HEAD").strip()
    _commit(root, _REPORTED)
    assert _ratchet(root, base2) == []


def test_the_real_measurement_path_is_what_is_ratcheted():
    """The scope must be the lint's own MEASUREMENT_PATH — one source of truth, not a second list —
    and every file in it must exist, or check 3 silently stops covering it."""
    root = Path(L.REPO_ROOT)
    assert len(L.MEASUREMENT_PATH) >= 16
    assert all((root / rel).exists() for rel in L.MEASUREMENT_PATH)


def test_the_real_tree_reports_a_nonzero_unreported_count():
    """Anti-vacuity: if the counter regressed to always-empty, every ratchet above would pass."""
    root = Path(L.REPO_ROOT)
    total = sum(len(L.unreported_swallow_hits((root / r).read_text())) for r in L.MEASUREMENT_PATH)
    assert total > 50, total


# ── review round: ONE definition of "reported" ───────────────────────────────


def _call(src: str) -> ast.Call:
    return ast.parse(src).body[0].value


@pytest.mark.parametrize(
    "src,expected",
    [
        ("log_swallowed_exception()", True),  # zero-arg: site from the frame
        ("log_swallowed_exception(site='f.py:g:1')", True),  # supplied-site Stage-1 form
        ("log_swallowed_exception(context={'k': 1})", True),  # still stage1_form in the helper
        ("utils.log_swallowed_exception()", True),  # attribute spelling
        ("log_swallowed_exception(e)", False),  # explicit exc -> plain DEBUG, no event
        ("log_swallowed_exception(e, operation='x')", False),
        ("log_swallowed_exception(operation='x')", False),
        ("log_swallowed_exception(**kw)", False),  # cannot prove it is Stage-1
        ("log_swallowed_exception(e, operation='x', site=None)", False),  # a literal None is no site
        ("logger.warning('x')", False),
    ],
)
def test_is_stage1_report_mirrors_the_helpers_own_stage1_form(src, expected):
    assert L.is_stage1_report(_call(src)) is expected


def test_the_explicit_form_is_NOT_a_report():
    """The hole both lenses found: `log_swallowed_exception()` -> `log_swallowed_exception(e,
    operation="x")` keeps swallowing, stops emitting the event, and used to pass check 3."""
    src = "def f():\n    try:\n        g()\n    except Exception as e:\n        log_swallowed_exception(e, operation='x')\n"
    assert L.unreported_swallow_hits(src) == [4]


def test_the_stage2_inventory_uses_the_same_definition():
    """Two definitions that disagree were the root cause; pin that there is one."""
    import scripts.fail_loud_stage2 as S

    # Same SOURCE, not the same object: fail_loud_stage2 imports the lint under its bare module name.
    assert S.is_stage1_report.__code__.co_filename == L.is_stage1_report.__code__.co_filename
    assert S.is_stage1_report.__code__.co_firstlineno == L.is_stage1_report.__code__.co_firstlineno


def test_nested_scopes_neither_exempt_nor_report():
    """A `raise` or a report inside a nested def/lambda does not run when the handler does."""
    raise_in_def = "try:\n    g()\nexcept Exception:\n    def h():\n        raise ValueError\n"
    report_in_lambda = "try:\n    g()\nexcept Exception:\n    cb = lambda: log_swallowed_exception()\n"
    assert L.unreported_swallow_hits(raise_in_def) == [3]
    assert L.unreported_swallow_hits(report_in_lambda) == [3]


# ── review round: main() as WIRED, not the pieces (the #590/D43 shape) ───────


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A tmp repo with a `main` base and a feature branch, with the lint pointed at it."""
    mp = "src/maxim/decisions/nac.py"
    legacy = "src/maxim/runtime/legacy.py"  # a listed file with a GRANDFATHERED unreported swallow
    other = "src/maxim/runtime/extracted.py"
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / mp).parent.mkdir(parents=True)
    (tmp_path / other).parent.mkdir(parents=True)
    (tmp_path / mp).write_text(_REPORTED)
    (tmp_path / legacy).write_text(_DEBUG_ONLY)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "checkout", "-q", "-b", "feature")  # HEAD must not be main, or the diff is empty
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(L, "MEASUREMENT_PATH", [mp, legacy])
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)

    def commit(files: dict[str, str | None]) -> None:
        for rel, text in files.items():
            p = tmp_path / rel
            if text is None:
                _git(tmp_path, "rm", "-q", rel)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-q", "-m", "change")

    return tmp_path, mp, other, commit


def test_main_is_clean_on_an_untouched_branch(wired):
    assert L.main() == 0


def test_main_fails_a_de_instrumentation(wired, capsys):
    root, mp, _, commit = wired
    commit({mp: _DEBUG_ONLY})
    assert L.main() == 1
    assert "unreported broad-swallow count rose" in capsys.readouterr().err


def test_main_fails_the_explicit_form_rewrite(wired, capsys):
    root, mp, _, commit = wired
    commit({mp: _EXPLICIT})
    assert L.main() == 1


def test_main_passes_a_deletion(wired):
    """What #863's burn-down does. Must never trip anything."""
    _, mp, _, commit = wired
    commit({mp: _DELETED})
    assert L.main() == 0


def test_main_passes_a_pure_rename_of_a_listed_file(wired, monkeypatch):
    """A single-file pathspec broke `-M` pairing, so a MOVE read as a file whose every site is new.

    The moved file must carry an UNREPORTED swallow for this to mean anything: the first version
    moved a file with only a reporting handler, zero before and zero after, and passed with the
    rename bug restored (found by mutation, this fold)."""
    root, mp, _, commit = wired
    legacy = "src/maxim/runtime/legacy.py"
    moved = "src/maxim/runtime/legacy_core.py"
    commit({legacy: None, moved: _DEBUG_ONLY})
    monkeypatch.setattr(L, "MEASUREMENT_PATH", [mp, moved])
    assert L.main() == 0


def test_check_4_catches_a_handler_de_instrumented_while_MOVED_out(wired, capsys):
    """The coverage hole check 3 alone had: a FUNCTION extracted into an UNLISTED module and
    de-instrumented on the way. `f` leaves the listed file (check 3 sees a fall and passes); only the
    repo-wide check, pooling functions that move, sees the report disappear without its handler.

    The function must actually MOVE. If `f` stayed put and only its handler went elsewhere, the diff
    would be indistinguishable from "delete a reporting swallow in f, add an unrelated logged handler
    in g" — which must PASS (S1, review round 2) — so no count can call it; that shape is the stated
    blind spot. Real extractions move functions, which is the case this pins."""
    root, mp, other, commit = wired
    commit({mp: "def unrelated():\n    return 1\n", other: _DEBUG_ONLY})
    assert L.main() == 1
    assert "check 4" in capsys.readouterr().err


def _rebase(root: Path, files: dict[str, str]) -> None:
    """Commit `files` to MAIN and fast-forward the feature branch, so they become the BASE."""
    _git(root, "checkout", "-q", "main")
    for rel, text in files.items():
        (root / rel).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "new base")
    _git(root, "checkout", "-q", "feature")
    _git(root, "merge", "-q", "--ff-only", "main")


_TWO = (
    "def f():\n    try:\n        g()\n    except Exception:\n        log_swallowed_exception()\n\n\n"
    "def h():\n    try:\n        g()\n    except Exception:\n        logger.debug('x')\n"
)


def test_a_burn_down_cannot_mask_a_de_instrumentation_in_another_function(wired, capsys):
    """S2, review round 2: deleting a silent swallow in `h` and de-instrumenting `f` in the same diff
    netted to zero under per-FILE counting and passed both checks — on exactly the burn-down PRs
    these exist for. Per function, `f` is caught."""
    root, mp, _, commit = wired
    _rebase(root, {mp: _TWO})
    commit(
        {
            mp: "def f():\n    try:\n        g()\n    except Exception:\n        logger.debug('y')\n\n\ndef h():\n    g()\n"
        }
    )
    assert L.main() == 1
    err = capsys.readouterr().err
    assert "::f" in err, err


def test_deleting_a_report_here_and_adding_a_logged_handler_there_passes(wired):
    """S1, review round 2: this failed check 4 under per-REPO netting, while its docstring promised
    handle-and-log outside the measurement path was untouched. A realistic burn-down shape."""
    root, mp, other, commit = wired
    commit({mp: _DELETED, other: _DEBUG_ONLY.replace("logger.debug('x')", "logger.warning('caller guard')")})
    assert L.main() == 0


def test_the_stage2_inventory_BEHAVES_like_the_shared_predicate(tmp_path):
    """S3, review round 2: the old version checked the predicate was IMPORTED, and passed with the
    inline zero-arg test restored inside `inventory_sites`. This checks what the inventory counts."""
    import scripts.fail_loud_stage2 as S

    (tmp_path / "m.py").write_text(
        "def a():\n    try:\n        g()\n    except Exception:\n        log_swallowed_exception()\n"
        "def b():\n    try:\n        g()\n    except Exception:\n        log_swallowed_exception(site='m.py:b:1')\n"
        "def c():\n    try:\n        g()\n    except Exception as e:\n        log_swallowed_exception(e, operation='x')\n"
    )
    assert sorted(s["function"] for s in S.inventory_sites(tmp_path)) == ["a", "b"]


def test_check_4_passes_a_pure_extraction(wired):
    root, mp, other, commit = wired
    commit({mp: _DELETED, other: _REPORTED})
    assert L.main() == 0
