"""scripts/pr_overlap.py: which open PRs collide, and in what order to merge them (the merge-train stopgap, M83)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import pr_overlap as O  # noqa: E402

LOG = "# Changelog\n\n## [Unreleased]\n\n- base\n"


def _git(root: Path, *args: str) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True).stdout.strip()


def _branch(root: Path, name: str, edits: dict[str, str], *, start: str = "main") -> str:
    _git(root, "checkout", "-q", "-b", name, start)
    for path, text in edits.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", name)
    sha = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", "main")
    return sha


def _advance_main(root: Path, edits: dict[str, str]) -> None:
    for path, text in edits.items():
        (root / path).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "main moves")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "CHANGELOG.md").write_text(LOG)
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def _pr(n, head="h", files=(), *, ci="success", state="BEHIND", draft=False, main_conflicts=()):
    return O.PR(n, f"pr {n}", head, set(files), state, ci, f"2026-10-27T00:00:0{n}Z", draft, list(main_conflicts))


# ── the merge predictions, on real git ──────────────────────────────────────────────────────────────────────────


def test_merge_tree_finds_a_real_conflict_and_a_clean_merge_and_writes_nothing(repo):
    a = _branch(repo, "a", {"CHANGELOG.md": LOG + "- from A\n"})
    b = _branch(repo, "b", {"CHANGELOG.md": LOG + "- from B\n"})
    c = _branch(repo, "c", {"a.py": "x = 2\n"})
    assert O.merge_tree(a, b, repo)[1] == ["CHANGELOG.md"]
    assert O.merge_tree(a, c, repo)[1] == []
    assert _git(repo, "status", "--porcelain") == "" and _git(repo, "branch", "--show-current") == "main"


def test_an_unknown_commit_is_a_tool_error_never_a_conflict_list(repo):
    """git merge-tree exits 1 for an unknown commit too; that must not read as "conflicts"."""
    with pytest.raises(O.ToolError):
        O.merge_tree("0" * 40, "1" * 40, repo)


def test_a_conflict_with_main_is_the_prs_own_and_never_blamed_on_another_pr(repo):
    """Executor E1: #1 branched from old main and already conflicts with main in CHANGELOG.md; #2 also edits it and
    lands cleanly. Landing #2 adds NOTHING to what #1 owes, and #1's own landing cannot be simulated."""
    stale = _branch(repo, "stale", {"CHANGELOG.md": LOG + "- stale entry\n", "a.py": "x = 3\n"})
    _advance_main(repo, {"CHANGELOG.md": LOG + "- main moved\n"})
    fresh = _branch(repo, "fresh", {"CHANGELOG.md": LOG + "- main moved\n- fresh\n", "a.py": "x = 1\ny = 1\n"})
    prs = [_pr(1, stale), _pr(2, fresh)]
    (pair,) = O.analyse(prs, repo, base="main")
    assert prs[0].main_conflicts == ["CHANGELOG.md"] and prs[1].main_conflicts == []
    assert pair.owed[1] is None  # #1 cannot land on main as it stands
    assert "a.py" in pair.owed[2]  # landing #2 adds a.py, which #1 did not already owe main
    assert "CHANGELOG.md" not in pair.owed[2]  # #1's own conflict with main is not blamed on #2
    text = O.report(prs, [pair])
    assert "#1 pr 1 — BEHIND, unit-tests success; CONFLICTS WITH MAIN NOW in CHANGELOG.md" in text
    assert "#1 conflicts with main itself, so its landing cannot be simulated" in text


def test_a_conflict_that_only_appears_after_the_other_lands_is_predicted(repo):
    """Both PRs append to CHANGELOG.md from current main: each is clean against main, and whichever lands first
    leaves the other a CHANGELOG conflict."""
    a = _branch(repo, "a", {"CHANGELOG.md": LOG + "- from A\n"})
    b = _branch(repo, "b", {"CHANGELOG.md": LOG + "- from B\n"})
    (pair,) = O.analyse([_pr(1, a), _pr(2, b)], repo, base="main")
    assert pair.owed == {1: ["CHANGELOG.md"], 2: ["CHANGELOG.md"]}
    assert pair.shared == {"CHANGELOG.md"}


def test_file_sets_come_from_git_not_gh(repo):
    """Executor E2: gh returns at most 100 files per PR; git has no cap."""
    many = {f"pkg/f{i}.py": "" for i in range(150)}
    head = _branch(repo, "big", many)
    assert O.changed_files(head, repo, base="main") == set(many)


def test_fetch_heads_points_at_the_fetched_commit_and_drop_refs_cleans_up(repo, tmp_path_factory):
    """Wire W1: the PR heads land in a per-run namespace, and nothing is left behind afterwards."""
    origin = tmp_path_factory.mktemp("origin") / "origin.git"  # outside the work tree
    _git(origin.parent, "init", "-q", "--bare", str(origin))
    head = _branch(repo, "feature", {"a.py": "x = 9\n"})
    _git(repo, "push", "-q", str(origin), "main:refs/heads/main", f"{head}:refs/pull/7/head")
    _git(repo, "remote", "add", "origin", str(origin))
    pr = _pr(7, "stale-sha-from-gh")
    ns = O.fetch_heads([pr], repo, ns="refs/pr-overlap/test")
    assert pr.head == head
    assert _git(repo, "for-each-ref", "--format=%(refname)", ns + "/") == f"{ns}/7"
    O.drop_refs(ns, repo)
    assert _git(repo, "for-each-ref", "--format=%(refname)", "refs/pr-overlap/") == ""


def test_main_drops_its_refs_even_when_the_analysis_fails(monkeypatch, capsys):
    dropped = []
    monkeypatch.setattr(O, "list_prs", lambda: [_pr(1)])
    monkeypatch.setattr(O, "fetch_heads", lambda prs, ns: ns)
    monkeypatch.setattr(O, "drop_refs", lambda ns: dropped.append(ns))

    def boom(prs):
        raise O.ToolError("git merge-base: bad object")

    monkeypatch.setattr(O, "analyse", boom)
    assert O.main([]) == 2 and dropped == [f"refs/pr-overlap/{os.getpid()}"]
    assert "cannot answer" in capsys.readouterr().err


def test_main_drops_its_refs_when_the_fetch_itself_fails_part_way(monkeypatch):
    """Delta S2: a fetch that wrote some refs then failed must still be cleaned up."""
    dropped = []
    monkeypatch.setattr(O, "list_prs", lambda: [_pr(1)])

    def half_fetch(prs, ns):
        raise O.ToolError("git fetch: cannot lock ref")

    monkeypatch.setattr(O, "fetch_heads", half_fetch)
    monkeypatch.setattr(O, "drop_refs", lambda ns: dropped.append(ns))
    assert O.main([]) == 2 and dropped == [f"refs/pr-overlap/{os.getpid()}"]


# ── reading gh, ordering, wording ───────────────────────────────────────────────────────────────────────────────


def test_the_newest_unit_tests_run_is_the_prs_state():
    """Wire W2 / #1442: a superseded cancelled run is not the PR's state; the newest one is, and a run not started
    yet counts as newest. A StatusContext (context/state) is read too."""
    old_cancel = {
        "name": "unit-tests",
        "status": "COMPLETED",
        "conclusion": "CANCELLED",
        "startedAt": "2026-10-27T01:00:00Z",
    }
    new_green = {
        "name": "unit-tests",
        "status": "COMPLETED",
        "conclusion": "SUCCESS",
        "startedAt": "2026-10-27T02:00:00Z",
    }
    queued = {"name": "unit-tests", "status": "QUEUED", "conclusion": "", "startedAt": None}
    assert O._unit_tests([new_green, old_cancel]) == "success"
    assert O._unit_tests([old_cancel, new_green]) == "success"
    assert O._unit_tests([new_green, queued]) == "pending"
    assert O._unit_tests([{"context": "unit-tests", "state": "FAILURE"}]) == "failure"
    assert O._unit_tests([{"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"}]) == "absent"


def test_parse_prs_keeps_main_based_prs_only():
    raw = [
        {"number": 7, "title": "t", "headRefOid": "abc", "baseRefName": "main", "mergeStateStatus": "BEHIND"},
        {"number": 8, "headRefOid": "d", "baseRefName": "release"},
    ]
    (pr,) = O.parse_prs(raw)
    assert (pr.number, pr.head, pr.merge_state, pr.unit_tests) == (7, "abc", "BEHIND", "absent")


@pytest.mark.parametrize(
    "case, expected",
    [
        ("draft_last", [2, 1]),
        ("blocked_after_ready", [2, 1]),
        ("isolated_first", [2, 1]),
        ("fewer_caused_conflicts_first", [2, 1]),
        ("fewer_hot_overlaps_first", [2, 1]),
    ],
)
def test_the_merge_order_key(case, expected):
    """Each clause of the order key, alone (Executor E3: every clause is pinned by a case)."""
    p1, p2 = _pr(1), _pr(2)
    ps: list[O.Pair] = []
    if case == "draft_last":
        p1.draft = True
    elif case == "blocked_after_ready":
        p1.main_conflicts = ["CHANGELOG.md"]
    elif case == "isolated_first":
        p3 = _pr(3)
        ps = [O.Pair(1, 3, {"x.py"}, {1: [], 3: []})]
        assert [p.number for p in O.merge_order([p1, p2, p3], ps)][0] == 2
        return
    elif case == "fewer_caused_conflicts_first":
        ps = [O.Pair(1, 2, {"x.py"}, {1: ["x.py"], 2: []})]
    elif case == "fewer_hot_overlaps_first":
        p3 = _pr(3)
        ps = [O.Pair(1, 3, {"CHANGELOG.md"}, {1: [], 3: []}), O.Pair(2, 3, {"x.py"}, {2: [], 3: []})]
        assert [p.number for p in O.merge_order([p1, p2, p3], ps)][:2] == [2, 1]
        return
    assert [p.number for p in O.merge_order([p1, p2], ps)] == expected


def test_a_dirty_pr_is_blocked():
    """Delta S1: GitHub's DIRTY alone (no main conflict computed) sorts the PR after a ready one."""
    assert [p.number for p in O.merge_order([_pr(1, state="DIRTY"), _pr(2)], [])] == [2, 1]


def test_red_ci_is_blocked_but_pending_is_not():
    assert [p.number for p in O.merge_order([_pr(1, ci="failure"), _pr(2, ci="pending")], [])] == [2, 1]


def test_owed_text_names_conflicts_and_a_floors_repin_with_advice_that_exists_today():
    """Arch A2 / Wire W3: --pins-from-run is not on main yet, and main's lint_coverage.py ignores an unknown flag,
    so the advice names only what exists: the values the lint prints."""
    pair = O.Pair(1, 2, {O.FLOORS, "CHANGELOG.md"}, {1: ["CHANGELOG.md"], 2: []})
    text = O.owed_text(1, 2, pair)
    assert text.startswith("if #1 lands first, #2 needs: merge main")
    assert "resolve CHANGELOG.md" in text and "the values lint_coverage.py prints" in text
    assert "--pins-from-run" not in text
    assert "resolve" not in O.owed_text(2, 1, pair)


def test_as_json_carries_the_order_and_the_owed_lists():
    prs = [_pr(1, files={"x.py"}), _pr(2, files={"x.py"}, draft=True)]
    ps = [O.Pair(1, 2, {"x.py"}, {1: ["x.py"], 2: None})]
    doc = O.as_json(prs, ps)
    assert doc["order"] == [1, 2]
    assert doc["pairs"] == [{"a": 1, "b": 2, "shared": ["x.py"], "owed": {"1": ["x.py"], "2": None}}]
