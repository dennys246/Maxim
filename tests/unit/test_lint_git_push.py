"""The diff-scoped lints on a push to main (#1089; owner decisions 2026-10-04).

``scripts/_lint_git.py::base_ref`` returned ``merge-base(origin/main, HEAD)``, which on a push to main is HEAD
itself, so every diff-scoped lint compared HEAD with HEAD and passed. On a push it now returns ``push_base``: the
newest first-parent ancestor whose ``lint`` job succeeded on a push run (not the event's ``before``: GitHub cancels
a PENDING run when a newer push queues). Each test below is one row of the approach note's "how could this input
lie" table, or one lint's push path.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import _lint_git
from tests.unit._push_event_helpers import fake_push, rev


def _git(root: Path, *args: str) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True).stdout


def _commit(root: Path, rel: str, text: str, msg: str = "change") -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)
    return rev(root, "HEAD")


@pytest.fixture
def main_repo(tmp_path: Path) -> Path:
    """A `main` with three linear commits c1 <- c2 <- c3 (HEAD)."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "commit.gpgsign", "false")
    for i in (1, 2, 3):
        _commit(root, "f.txt", f"{i}\n", f"c{i}")
    return root


def _shas(root: Path) -> list[str]:
    return _git(root, "rev-list", "--reverse", "HEAD").split()


# ── push_base: the base is the last green push ───────────────────────────────


def test_the_base_is_before_when_its_lint_passed(main_repo, monkeypatch):
    c1, c2, _c3 = _shas(main_repo)
    fake_push(monkeypatch, main_repo, before=c2, green={c2, c1})
    assert _lint_git.base_ref(main_repo) == c2


def test_a_cancelled_or_red_before_walks_back_to_the_last_green_push(main_repo, monkeypatch):
    """Pushes c2 then c3 queued: c2's run was cancelled while pending, so c1..c2 was never diffed."""
    c1, c2, _c3 = _shas(main_repo)
    fake_push(monkeypatch, main_repo, before=c2, green={c1})
    assert _lint_git.base_ref(main_repo) == c1


def test_no_green_push_in_reach_fails_closed(main_repo, monkeypatch):
    _c1, c2, _c3 = _shas(main_repo)
    fake_push(monkeypatch, main_repo, before=c2, green=set())
    with pytest.raises(_lint_git.GitUnavailable, match="no push run with a green `lint` job"):
        _lint_git.base_ref(main_repo)


def test_a_red_lint_job_is_not_green(main_repo, monkeypatch):
    c1, c2, _c3 = _shas(main_repo)
    calls = fake_push(monkeypatch, main_repo, before=c2, green={c1, c2})
    real = _lint_git.gh_api

    def api(path):
        if path.endswith(f"/runs/{1000 + sorted({c1, c2}).index(c2)}/jobs?per_page=100"):
            calls.append(path)
            return {"jobs": [{"name": "lint", "conclusion": "failure"}, {"name": "other", "conclusion": "success"}]}
        return real(path)

    import sys

    for name in ("_lint_git", "scripts._lint_git"):
        if name in sys.modules:
            monkeypatch.setattr(sys.modules[name], "gh_api", api)
    assert _lint_git.base_ref(main_repo) == c1


# ── push_base: every input that could lie fails closed ───────────────────────


@pytest.mark.parametrize(
    ("event", "match"),
    [
        ({"after": "x"}, "no valid `before`"),
        ({"before": "abc", "after": "x"}, "no valid `before`"),
        ({"before": "0" * 40, "after": None}, "no valid `after`"),
    ],
)
def test_a_malformed_event_fails_closed(main_repo, monkeypatch, event, match):
    fake_push(monkeypatch, main_repo)
    Path(os.environ["GITHUB_EVENT_PATH"]).write_text(json.dumps(event))
    with pytest.raises(_lint_git.GitUnavailable, match=match):
        _lint_git.base_ref(main_repo)


def test_an_unreadable_event_file_fails_closed(main_repo, monkeypatch):
    fake_push(monkeypatch, main_repo)
    Path(os.environ["GITHUB_EVENT_PATH"]).write_text("{not json")
    with pytest.raises(_lint_git.GitUnavailable, match="unreadable"):
        _lint_git.base_ref(main_repo)


def test_a_branch_creating_push_fails_closed(main_repo, monkeypatch):
    fake_push(monkeypatch, main_repo, before="0" * 40)
    with pytest.raises(_lint_git.GitUnavailable, match="created the branch"):
        _lint_git.base_ref(main_repo)


def test_a_checkout_that_is_not_the_pushed_commit_fails_closed(main_repo, monkeypatch):
    _c1, c2, _c3 = _shas(main_repo)
    fake_push(monkeypatch, main_repo, before=_shas(main_repo)[0], after=c2)
    with pytest.raises(_lint_git.GitUnavailable, match="is not the pushed commit"):
        _lint_git.base_ref(main_repo)


def test_a_before_missing_from_the_clone_fails_closed(main_repo, monkeypatch):
    fake_push(monkeypatch, main_repo, before="1" * 40)
    with pytest.raises(_lint_git.GitUnavailable):
        _lint_git.base_ref(main_repo)


def test_a_non_fast_forward_push_fails_closed(main_repo, monkeypatch):
    c1, _c2, _c3 = _shas(main_repo)
    _git(main_repo, "checkout", "-q", "-b", "side", c1)
    stray = _commit(main_repo, "g.txt", "x\n", "stray")
    _git(main_repo, "checkout", "-q", "main")
    fake_push(monkeypatch, main_repo, before=stray, green={stray})
    with pytest.raises(_lint_git.GitUnavailable, match="non-fast-forward"):
        _lint_git.base_ref(main_repo)


@pytest.mark.parametrize("missing", ["GITHUB_EVENT_PATH", "GITHUB_REPOSITORY", "GITHUB_JOB", "GITHUB_WORKFLOW_REF"])
def test_a_missing_push_variable_fails_closed(main_repo, monkeypatch, missing):
    fake_push(monkeypatch, main_repo)
    monkeypatch.delenv(missing)
    with pytest.raises(_lint_git.GitUnavailable, match=missing):
        _lint_git.base_ref(main_repo)


def test_an_api_failure_fails_closed(main_repo, monkeypatch):
    fake_push(monkeypatch, main_repo)

    def broken(path):
        raise _lint_git.GitUnavailable(f"gh api {path}: HTTP 403")

    import sys

    for name in ("_lint_git", "scripts._lint_git"):
        if name in sys.modules:
            monkeypatch.setattr(sys.modules[name], "gh_api", broken)
    with pytest.raises(_lint_git.GitUnavailable, match="403"):
        _lint_git.base_ref(main_repo)


def test_locally_the_base_is_still_the_merge_base(main_repo):
    _git(main_repo, "checkout", "-q", "-b", "feature")
    _commit(main_repo, "f.txt", "4\n")
    assert _lint_git.base_ref(main_repo) == _shas(main_repo)[2]


# ── push_units: what landed, and its merged PR ───────────────────────────────


def test_units_split_a_range_into_merges_squashes_and_direct_pushes(main_repo, monkeypatch):
    c1, _c2, c3 = _shas(main_repo)
    _git(main_repo, "checkout", "-q", "-b", "pr7")
    b1 = _commit(main_repo, "a.txt", "1\n", "branch 1")
    b2 = _commit(main_repo, "a.txt", "2\n", "branch 2")
    _git(main_repo, "checkout", "-q", "main")
    _git(main_repo, "merge", "-q", "--no-ff", "-m", "Merge pull request #7", "pr7")
    merge = rev(main_repo, "HEAD")
    squash = _commit(main_repo, "b.txt", "1\n", "feat: squashed (#8)")
    direct = _commit(main_repo, "c.txt", "1\n", "direct push")
    prs = {merge: {"number": 7, "title": "t7", "body": ""}, squash: {"number": 8, "title": "t8", "body": ""}}
    fake_push(
        monkeypatch,
        main_repo,
        before=squash,
        green={c3},
        prs=prs,
        pr_commit_dates={8: ["2026-01-02T00:00:00Z", "2026-03-01T00:00:00Z"]},
    )
    base = _lint_git.base_ref(main_repo)
    assert base == c3
    units = _lint_git.push_units(main_repo, base)
    assert [u.sha for u in units] == [merge, squash, direct]
    assert units[0].pr["number"] == 7 and units[0].commits == (b1, b2)
    assert units[1].pr["number"] == 8 and units[1].commits == (squash,)
    assert units[1].fork_epoch == 1767312000  # 2026-01-02: the PR's first commit, earlier than its parent
    assert units[2].pr is None and units[2].commits == (direct,)
    del c1


# ── each lint's push path ────────────────────────────────────────────────────


def _fix_repo(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "commit.gpgsign", "false")
    base = _commit(root, "src/maxim/x.py", "a = 1\n", "init")
    return root, base


def test_fix_touches_tests_on_push_honours_the_merged_prs_body(tmp_path, monkeypatch, capsys):
    from scripts import lint_fix_touches_tests as F

    root, base = _fix_repo(tmp_path, monkeypatch)
    squash = _commit(root, "src/maxim/x.py", "a = 2\n", "fix: a thing (#9)")
    monkeypatch.setattr(F, "REPO_ROOT", root)
    fake_push(monkeypatch, root, green={base}, prs={squash: {"number": 9, "title": "fix: a thing", "body": ""}})
    assert F.main() == 1
    assert "touches src/" in capsys.readouterr().err
    fake_push(
        monkeypatch,
        root,
        green={base},
        prs={squash: {"number": 9, "title": "fix: a thing", "body": "[no-tests: config only]"}},
    )
    assert F.main() == 0
    assert "declared: config only" in capsys.readouterr().out


def test_fix_touches_tests_on_a_direct_push_has_no_pr_opt_out(tmp_path, monkeypatch, capsys):
    from scripts import lint_fix_touches_tests as F

    root, base = _fix_repo(tmp_path, monkeypatch)
    _commit(root, "src/maxim/x.py", "a = 2\n", "fix: pushed straight to main")
    monkeypatch.setattr(F, "REPO_ROOT", root)
    fake_push(monkeypatch, root, green={base})
    assert F.main() == 1


def test_fix_touches_tests_mid_run_git_failure_is_an_error_on_push(tmp_path, monkeypatch, capsys):
    """Was a silent `return 0` ("skipped mid-run"): a fail-open path the push gate made reachable."""
    from scripts import lint_fix_touches_tests as F

    root, base = _fix_repo(tmp_path, monkeypatch)
    _commit(root, "src/maxim/x.py", "a = 2\n", "fix: x")
    monkeypatch.setattr(F, "REPO_ROOT", root)
    fake_push(monkeypatch, root, green={base})

    def boom(*_a, **_k):
        raise F.GitUnavailable("git log: broken")

    monkeypatch.setattr(F, "push_violations", boom)
    assert F.main() == 2


def test_multi_agent_marker_uses_the_shared_resolver_on_push(main_repo, monkeypatch):
    from scripts import lint_multi_agent_marker as M

    c1, c2, _c3 = _shas(main_repo)
    monkeypatch.setattr(M, "REPO_ROOT", main_repo)
    fake_push(monkeypatch, main_repo, before=c2, green={c1})
    assert M._resolve_base_ref() == (c1, "")


def test_the_ledger_branch_point_on_a_squash_push_is_the_prs_first_commit(main_repo, monkeypatch):
    """A squash keeps no fork point in git, so `_branch_point` fell back to the push base's date and a row dated
    when the branch began read as backdated (the adversarial pass's false red). On push the PR's first commit says."""
    from scripts import lint_ledger_format as LF

    _c1, _c2, c3 = _shas(main_repo)
    squash = _commit(main_repo, "b.txt", "1\n", "feat: squashed (#8)")
    fake_push(
        monkeypatch,
        main_repo,
        before=c3,
        green={c3},
        prs={squash: {"number": 8, "title": "t", "body": ""}},
        pr_commit_dates={8: ["2026-01-02T00:00:00Z", "2026-03-01T00:00:00Z"]},
    )
    assert LF._branch_epoch(main_repo, c3) == 1767312000
    monkeypatch.delenv("GITHUB_EVENT_NAME")  # off a push: the git-only branch point (the base commit itself)
    assert LF._branch_epoch(main_repo, c3) == int(_git(main_repo, "show", "-s", "--format=%ct", c3).strip())


def test_a_test_never_inherits_the_runners_push_event(tmp_path):
    """tests/conftest.py::_scrub_ci_event_env: in the unit-tests job of a push run, the runner's real
    GITHUB_EVENT_NAME=push would otherwise make every lint test judge its fixture repo against the real push."""
    import sys

    probe = tmp_path / "test_sees_no_event.py"
    probe.write_text(
        "import os\n\ndef test_clean():\n"
        "    for name in ('GITHUB_EVENT_NAME', 'GITHUB_EVENT_PATH', 'GITHUB_BASE_REF', 'PR_TITLE', 'PR_BODY'):\n"
        "        assert os.environ.get(name) is None, name\n"
    )
    env = {**os.environ, "GITHUB_EVENT_NAME": "push", "GITHUB_EVENT_PATH": "/x", "PR_TITLE": "fix: t", "PR_BODY": "b"}
    repo = Path(__file__).resolve().parents[2]
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "tests.conftest", "-p", "no:cacheprovider", str(probe)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_green_run_may_be_on_a_later_page(main_repo, monkeypatch):
    c1, c2, _c3 = _shas(main_repo)
    fake_push(monkeypatch, main_repo, before=c2, green={c1}, paginate=True)
    assert _lint_git.base_ref(main_repo) == c1


def test_fix_touches_tests_reads_a_rebase_merged_prs_aggregate_diff(tmp_path, monkeypatch, capsys):
    """A rebase merge lands PR #10 as two first-parent commits: src first, its test second. The PR title rule
    reads the PR's whole diff, so the fix is not red for the commit that only touched src."""
    from scripts import lint_fix_touches_tests as F

    root, base = _fix_repo(tmp_path, monkeypatch)
    one = _commit(root, "src/maxim/x.py", "a = 2\n", "refactor: the change")
    two = _commit(root, "tests/test_x.py", "def test_x(): pass\n", "test: its test")
    pr = {"number": 10, "title": "fix: x", "body": ""}
    monkeypatch.setattr(F, "REPO_ROOT", root)
    fake_push(monkeypatch, root, green={base}, prs={one: pr, two: pr})
    assert F.main() == 0, capsys.readouterr().err


# ── the acceptance record (owner decision 2026-10-04) ─────────────────────────


def _accept(root: Path, sha: str, msg: str = "accept") -> str:
    entry = {"sha": sha, "reason": "the #1090 format migration", "owner": "owner", "date": "2026-10-04"}
    return _commit(root, "scripts/push_base_accepts.json", json.dumps([entry]), msg)


def test_an_accepted_commit_is_a_push_base_when_no_push_is_green(main_repo, monkeypatch):
    c1, c2, _c3 = _shas(main_repo)
    adder = _accept(main_repo, c2)
    fake_push(
        monkeypatch,
        main_repo,
        before=_shas(main_repo)[2],
        green=set(),
        prs={adder: {"number": 5, "title": "t", "body": ""}},
    )
    assert _lint_git.base_ref(main_repo) == c2
    del c1


def test_a_direct_push_cannot_accept_itself(main_repo, monkeypatch):
    _c1, c2, c3 = _shas(main_repo)
    _accept(main_repo, c2)
    fake_push(monkeypatch, main_repo, before=c3, green=set())
    with pytest.raises(_lint_git.GitUnavailable, match="did not arrive through a merged PR"):
        _lint_git.base_ref(main_repo)


@pytest.mark.parametrize("second", ["re-add", "rewrite"])
def test_a_direct_push_cannot_rewrite_or_re_add_an_accepted_entry(main_repo, monkeypatch, second):
    """A merged PR added the entry first; a later direct push re-adds it after removal, or rewrites its reason.
    Checking only the commit that introduced the sha honoured both (architecture delta round)."""
    _c1, c2, _c3 = _shas(main_repo)
    adder = _accept(main_repo, c2)
    pr = {adder: {"number": 5, "title": "t", "body": ""}}
    if second == "re-add":
        remover = _commit(main_repo, "scripts/push_base_accepts.json", "[]", "remove")
        pr[remover] = {"number": 6, "title": "t", "body": ""}
        _accept(main_repo, c2, "direct re-add")
    else:
        entry = {"sha": c2, "reason": "forged", "owner": "someone", "date": "2026-10-05"}
        _commit(main_repo, "scripts/push_base_accepts.json", json.dumps([entry]), "direct rewrite")
    fake_push(monkeypatch, main_repo, green=set(), prs=pr)
    with pytest.raises(_lint_git.GitUnavailable, match="did not arrive through a merged PR"):
        _lint_git.base_ref(main_repo)


def test_an_accepted_sha_off_mains_first_parent_chain_fails_closed(main_repo, monkeypatch):
    _c1, _c2, c3 = _shas(main_repo)
    adder = _accept(main_repo, "2" * 40)
    fake_push(monkeypatch, main_repo, before=c3, green={c3}, prs={adder: {"number": 5, "title": "t", "body": ""}})
    with pytest.raises(_lint_git.GitUnavailable, match="not a first-parent commit"):
        _lint_git.base_ref(main_repo)


_GOOD = {"reason": "r", "owner": "o", "date": "2026-10-04"}


@pytest.mark.parametrize(
    "entry",
    [
        {"sha": "x"},
        {"sha": "SHA", **_GOOD, "reason": ""},
        {"sha": "SHA", **_GOOD, "date": "4 Oct 2026"},  # a valid sha and keys: only the date check catches it
        {"sha": "SHA", **_GOOD, "approved_by": "o"},  # an extra key: only the exact-keys check catches it
    ],
)
def test_a_malformed_acceptance_fails_closed(main_repo, monkeypatch, entry):
    _c1, c2, c3 = _shas(main_repo)
    entry = {k: (c2 if v == "SHA" else v) for k, v in entry.items()}
    adder = _commit(main_repo, "scripts/push_base_accepts.json", json.dumps([entry]))
    fake_push(monkeypatch, main_repo, before=c3, green={c3}, prs={adder: {"number": 5, "title": "t", "body": ""}})
    with pytest.raises(_lint_git.GitUnavailable, match="malformed entry"):
        _lint_git.base_ref(main_repo)


def test_an_acceptance_merged_by_a_merge_commit_is_honoured(main_repo, monkeypatch):
    """House rule (3) merge-commits data/protocol PRs. `git log --diff-merges=first-parent --format=%H` prints the
    merge's PATCH too, and its tokens reached the PR lookup as shas (architecture delta round 2)."""
    _c1, c2, _c3 = _shas(main_repo)
    _git(main_repo, "checkout", "-q", "-b", "accept")
    _accept(main_repo, c2)
    _git(main_repo, "checkout", "-q", "main")
    _git(main_repo, "merge", "-q", "--no-ff", "-m", "Merge pull request #11", "accept")
    merge = rev(main_repo, "HEAD")
    fake_push(monkeypatch, main_repo, green=set(), prs={merge: {"number": 11, "title": "t", "body": ""}})
    assert _lint_git.base_ref(main_repo) == c2
