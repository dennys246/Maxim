"""scripts/lint_function_length.py — the ONE function-length ratchet (roadmap 1.3.2 item 6; #940 item 1).

Every gate test drives ``main()`` on a fixture git repo (base on ``main``, HEAD on ``feature``) with
a tiny ``src/maxim`` package and its own baseline, at the real THRESHOLD of 200. The last test runs
rules 1-3 on THIS checkout, with the root taken from this file's path — never ``maxim.__file__``,
which in a worktree without ``PYTHONPATH`` would measure another tree.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import lint_function_length as L

_BASELINE = "src/maxim/utils/function_length_baseline.json"


def fn(name: str, n: int, tag: str = "x", indent: str = "") -> str:
    """A def whose span is exactly ``n`` lines; ``tag`` changes the body without changing the span."""
    body = "".join(f"{indent}    {tag}{i} = {i}\n" for i in range(n - 1))
    return f"{indent}def {name}():\n{body}"


def exc(qualname: str, frm, to: int, *, file: str = "src/maxim/a.py", ref: str = "#1", reason: str = "r", **opt):
    return {
        "file": file,
        "qualname": qualname,
        "from": frm,
        "to": to,
        "date": "2026-10-04",
        "ref": ref,
        "reason": reason,
        **opt,
    }


def baseline(entries: dict, exceptions: list | None = None, *, threshold: int = 200, version: int = 2) -> str:
    return json.dumps(
        {
            "_comment": "fixture",
            "baseline_format_version": version,
            "threshold": threshold,
            "entries": [{"file": f, "qualname": q, "lines": n} for (f, q), n in entries.items()],
            "exceptions": exceptions or [],
            "history": [],
        }
    )


def _git(root: Path, *args: str) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True).stdout


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _commit(root: Path) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "change")


A = ("src/maxim/a.py", "big")
_BASE_A = fn("big", 300) + "\n\n" + fn("small", 10)


def make_repo(root: Path, monkeypatch, files: dict[str, str], base_baseline: str) -> Path:
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "commit.gpgsign", "false")
    _write(root, "src/maxim/__init__.py", "")
    for rel, text in files.items():
        _write(root, rel, text)
    _write(root, _BASELINE, base_baseline)
    _commit(root)
    _git(root, "checkout", "-q", "-b", "feature")
    monkeypatch.setattr(L, "REPO_ROOT", root)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    return root


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    """Base: a.py::big is 300 lines and pinned at 300, with no exceptions."""
    return make_repo(tmp_path, monkeypatch, {"src/maxim/a.py": _BASE_A}, baseline({A: 300}))


def run(root: Path, capsys, *, commit: bool = True) -> tuple[int, str]:
    if commit:
        _commit(root)
    rc = L.main()
    out = capsys.readouterr()
    return rc, out.out + out.err


def head(root: Path, a: str | None = None, entries: dict | None = None, exceptions: list | None = None, **kw) -> None:
    if a is not None:
        _write(root, "src/maxim/a.py", a)
    _write(root, _BASELINE, baseline(entries if entries is not None else {A: 300}, exceptions, **kw))


# ── every run: rules 1-3 ─────────────────────────────────────────────────────


def test_untouched_branch_is_clean_and_prints_every_pin_and_the_count(repo, capsys):
    rc, out = run(repo, capsys, commit=False)
    assert rc == 0, out
    assert "src/maxim/a.py::big = 300 (pin 300)" in out
    assert "1 pinned over 200" in out


def test_growth_FAILS(repo, capsys):
    head(repo, fn("big", 301))
    rc, out = run(repo, capsys)
    assert rc == 1 and "grew to 301" in out


def test_shrink_FAILS_until_the_pin_is_lowered(repo, capsys):
    head(repo, fn("big", 299))
    rc, out = run(repo, capsys)
    assert rc == 1 and "shrank to 299" in out and "lower the pin" in out
    head(repo, fn("big", 299), {A: 299})
    assert run(repo, capsys)[0] == 0


def test_dropping_below_the_threshold_requires_removing_the_entry(repo, capsys):
    head(repo, fn("big", 150), {A: 150})
    rc, out = run(repo, capsys)
    assert rc == 1 and "remove the entry" in out
    head(repo, fn("big", 150), {})
    assert run(repo, capsys)[0] == 0


def test_a_new_function_over_200_FAILS_as_unpinned(repo, capsys):
    head(repo, _BASE_A + "\n\n" + fn("fresh", 201))
    rc, out = run(repo, capsys)
    assert rc == 1 and "a.py::fresh: 201 lines" in out and "unpinned" in out


def test_deleting_the_entry_of_a_still_long_function_FAILS(repo, capsys):
    head(repo, entries={})
    rc, out = run(repo, capsys)
    assert rc == 1 and "a.py::big: 300 lines, over 200 and unpinned" in out


def test_an_orphan_entry_FAILS(repo, capsys):
    head(repo, entries={A: 300, ("src/maxim/a.py", "gone"): 250})
    rc, out = run(repo, capsys)
    assert rc == 1 and "a.py::gone: orphan entry" in out


def test_an_ambiguous_qualname_over_200_FAILS(repo, capsys):
    cond = "import os\nif os.name:\n" + fn("cond", 250, indent="    ") + "else:\n" + fn("cond", 3, indent="    ")
    head(repo, _BASE_A + "\n\n" + cond)
    rc, out = run(repo, capsys)
    assert rc == 1 and "a.py::cond: ambiguous" in out


def test_a_threshold_change_FAILS(repo, capsys):
    head(repo, threshold=400)
    rc, out = run(repo, capsys)
    assert rc == 1 and "THRESHOLD is 200" in out and "threshold changed 200 -> 400" in out


@pytest.mark.parametrize("bad", ['"version 1"', "1", "3"])
def test_only_format_version_2_is_accepted(repo, capsys, bad):
    _write(
        repo, _BASELINE, baseline({A: 300}).replace('"baseline_format_version": 2', f'"baseline_format_version": {bad}')
    )
    rc, out = run(repo, capsys)
    assert rc == 1 and "baseline_format_version must be 2" in out


def test_an_unknown_baseline_key_FAILS(repo, capsys):
    data = json.loads(baseline({A: 300}))
    data["allow"] = ["everything"]
    _write(repo, _BASELINE, json.dumps(data))
    rc, out = run(repo, capsys)
    assert rc == 1 and "unknown: ['allow']" in out


# ── diff-scoped: raises and new entries need a NEW exception ─────────────────


def test_a_raise_without_an_exception_FAILS(repo, capsys):
    head(repo, fn("big", 310), {A: 310})
    rc, out = run(repo, capsys)
    assert rc == 1 and "pin raised 300 -> 310 without a new exception" in out


def test_a_raise_with_a_new_matching_exception_passes(repo, capsys):
    head(repo, fn("big", 310), {A: 310}, [exc("big", 300, 310)])
    rc, out = run(repo, capsys)
    assert rc == 0, out


def test_a_raise_whose_exception_has_the_wrong_from_FAILS(repo, capsys):
    head(repo, fn("big", 310), {A: 310}, [exc("big", 290, 310)])
    rc, out = run(repo, capsys)
    assert rc == 1 and "pin raised" in out and "unused exception" in out


def test_a_new_function_over_200_with_a_from_null_exception_passes(repo, capsys):
    head(
        repo, _BASE_A + "\n\n" + fn("fresh", 210), {A: 300, ("src/maxim/a.py", "fresh"): 210}, [exc("fresh", None, 210)]
    )
    assert run(repo, capsys)[0] == 0
    head(repo, _BASE_A + "\n\n" + fn("fresh", 210), {A: 300, ("src/maxim/a.py", "fresh"): 210})
    rc, out = run(repo, capsys)
    assert rc == 1 and "a.py::fresh: new entry at 210" in out


def test_an_unused_new_exception_FAILS(repo, capsys):
    head(repo, exceptions=[exc("big", 300, 320)])  # pre-approving a later raise
    rc, out = run(repo, capsys)
    assert rc == 1 and "unused exception for src/maxim/a.py::big (300 -> 320)" in out


@pytest.mark.parametrize(
    ("ref", "reason"), [("PR 12", "r"), ("#", "r"), ("https://example.com/pull/1", "r"), ("#12", "  ")]
)
def test_a_bad_ref_or_empty_reason_FAILS(repo, capsys, ref, reason):
    head(repo, fn("big", 310), {A: 310}, [exc("big", 300, 310, ref=ref, reason=reason)])
    rc, out = run(repo, capsys)
    assert rc == 1 and ("`ref` must be" in out or "`reason` must be non-empty" in out)


def test_a_github_url_ref_is_accepted(repo, capsys):
    url = "https://github.com/dennys246/Maxim/pull/1088"
    head(repo, fn("big", 310), {A: 310}, [exc("big", 300, 310, ref=url)])
    assert run(repo, capsys)[0] == 0


# ── exceptions are append-only ───────────────────────────────────────────────

_OLD = [exc("big", None, 300, reason="first"), exc("big", None, 300, reason="second")]


@pytest.fixture
def repo_with_exceptions(tmp_path: Path, monkeypatch) -> Path:
    return make_repo(tmp_path, monkeypatch, {"src/maxim/a.py": _BASE_A}, baseline({A: 300}, _OLD))


@pytest.mark.parametrize(
    "mutated",
    [
        [dict(_OLD[0], reason="edited"), _OLD[1]],  # edited
        [_OLD[1], _OLD[0]],  # reordered
        [_OLD[0]],  # removed
        [_OLD[1], exc("big", 300, 310)],  # removed + a new one appended
    ],
    ids=["edited", "reordered", "removed", "removed-plus-appended"],
)
def test_an_old_exception_edited_reordered_or_removed_FAILS(repo_with_exceptions, capsys, mutated):
    head(repo_with_exceptions, exceptions=mutated)
    rc, out = run(repo_with_exceptions, capsys)
    assert rc == 1 and "exceptions are append-only" in out


def test_appending_after_the_old_exceptions_passes(repo_with_exceptions, capsys):
    head(repo_with_exceptions, fn("big", 310), {A: 310}, [*_OLD, exc("big", 300, 310)])
    rc, out = run(repo_with_exceptions, capsys)
    assert rc == 0, out


# ── moves and splits ─────────────────────────────────────────────────────────

B = ("src/maxim/b.py", "big")


def test_an_identical_body_move_is_free(repo, capsys):
    _write(repo, "src/maxim/b.py", fn("big", 300))
    head(repo, fn("small", 10), {B: 300})
    rc, out = run(repo, capsys)
    assert rc == 0, out


def test_an_identical_body_rename_is_free(repo, capsys):
    head(repo, fn("renamed", 300) + "\n\n" + fn("small", 10), {("src/maxim/a.py", "renamed"): 300})
    assert run(repo, capsys)[0] == 0


def test_a_changed_body_move_FAILS_without_a_moved_from_exception(repo, capsys):
    _write(repo, "src/maxim/b.py", fn("big", 300, tag="y"))  # same span, different body
    head(repo, fn("small", 10), {B: 300})
    rc, out = run(repo, capsys)
    assert rc == 1 and "b.py::big: new entry at 300" in out
    head(
        repo,
        fn("small", 10),
        {B: 300},
        [exc("big", 300, 300, file="src/maxim/b.py", moved_from={"file": A[0], "qualname": "big"})],
    )
    rc, out = run(repo, capsys)
    assert rc == 0, out


def test_a_split_with_split_from_passes_and_without_FAILS(repo, capsys):
    split = fn("big", 150) + "\n\n" + fn("piece", 250, tag="p") + "\n\n" + fn("small", 10)
    piece = ("src/maxim/a.py", "piece")
    head(repo, split, {piece: 250})
    rc, out = run(repo, capsys)
    assert rc == 1 and "a.py::piece: new entry at 250" in out
    head(repo, split, {piece: 250}, [exc("piece", None, 250, split_from={"file": A[0], "qualname": "big"})])
    rc, out = run(repo, capsys)
    assert rc == 0, out


def test_split_from_a_base_entry_whose_pin_did_not_drop_FAILS(repo, capsys):
    piece = ("src/maxim/a.py", "piece")
    head(
        repo,
        _BASE_A + "\n\n" + fn("piece", 250, tag="p"),
        {A: 300, piece: 250},
        [exc("piece", None, 250, split_from={"file": A[0], "qualname": "big"})],
    )
    rc, out = run(repo, capsys)
    assert rc == 1 and "split_from src/maxim/a.py::big, but that base entry's pin did not drop" in out


# ── when the diff rules run ──────────────────────────────────────────────────


def test_no_merge_base_on_a_pull_request_FAILS(repo, monkeypatch, capsys):
    _git(repo, "branch", "-q", "-D", "main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    rc, out = run(repo, capsys, commit=False)
    assert rc == 2 and "cannot run on a pull request" in out


def test_without_a_merge_base_locally_only_rules_1_to_3_run(repo, capsys):
    _git(repo, "branch", "-q", "-D", "main")
    head(repo, fn("big", 310), {A: 310})  # an unrecorded raise: only the diff rules see it
    rc, out = run(repo, capsys)
    assert rc == 0 and "diff-scoped rules skipped" in out
    head(repo, fn("big", 311), {A: 310})  # growth: rule 3 still fires
    assert run(repo, capsys)[0] == 1


def test_a_push_event_runs_rules_1_to_3_only(repo, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    head(repo, fn("big", 310), {A: 310})
    rc, out = run(repo, capsys)
    assert rc == 0 and "push event — rules 1-3 only" in out


# ── the merge-base baseline must be format 2 (#1089 item 1) ───────────────────

_V1 = json.dumps({"baseline_format_version": 1, "entries": [{"file": "maxim/a.py", "function": "big", "lines": 300}]})


def test_a_v1_baseline_at_the_merge_base_FAILS(tmp_path, monkeypatch, capsys):
    root = make_repo(tmp_path, monkeypatch, {"src/maxim/a.py": _BASE_A}, _V1)
    head(root)
    rc, out = run(root, capsys)
    assert rc == 1 and "merge-base baseline unreadable (baseline_format_version must be 2" in out


def test_a_merge_base_without_the_baseline_FAILS(tmp_path, monkeypatch, capsys):
    root = make_repo(tmp_path, monkeypatch, {"src/maxim/a.py": _BASE_A}, baseline({A: 300}))
    _git(root, "checkout", "-q", "main")
    _git(root, "rm", "-q", _BASELINE)
    _commit(root)
    _git(root, "checkout", "-q", "-b", "feature2")
    head(root)
    rc, out = run(root, capsys)
    assert rc == 1 and "absent at the merge-base (moved or deleted?)" in out


def test_head_may_not_be_v1(repo, capsys):
    _write(repo, _BASELINE, _V1)
    rc, out = run(repo, capsys, commit=False)
    assert rc == 1 and "baseline_format_version must be 2" in out


# ── measurement ──────────────────────────────────────────────────────────────


def test_span_excludes_decorators_and_counts_def_to_last_line():
    defs = L.defs_in("@dec\ndef f():\n    a = 1\n\n    return a\n\n\ndef g():\n    pass\n")
    assert L.span(defs["f"][0]) == 4
    assert L.span(defs["g"][0]) == 2


def test_qualnames_follow_python():
    src = (
        "class C:\n    def m(self):\n        def inner():\n            pass\n"
        "def outer():\n    class K:\n        def km(self):\n            pass\n    async def a():\n        pass\n"
    )
    assert set(L.defs_in(src)) == {"C.m", "C.m.<locals>.inner", "outer", "outer.<locals>.K.km", "outer.<locals>.a"}


def test_normalized_ignores_name_and_position_but_not_body():
    a = L.defs_in("def f():\n    return 1\n")["f"][0]
    b = L.defs_in("\n\n\ndef g():\n    return 1\n")["g"][0]
    c = L.defs_in("def f():\n    return 2\n")["f"][0]
    assert L.normalized(a) == L.normalized(b) != L.normalized(c)


def test_a_syntax_error_in_scope_FAILS_rather_than_skips(repo, capsys):
    _write(repo, "src/maxim/broken.py", "def (:\n")
    rc, out = run(repo, capsys)
    assert rc == 1 and "src/maxim/broken.py: cannot parse" in out


# ── this checkout ────────────────────────────────────────────────────────────


def test_this_checkout_is_clean():
    """Rules 1-3 on the repo this test file lives in (not ``maxim.__file__``'s)."""
    root = Path(__file__).resolve().parents[2]
    failures, measure, base = L.check_head(root)
    assert failures == []
    assert base is not None and base.threshold == L.THRESHOLD == 200
    assert measure is not None and len(base.entries) == sum(1 for s in measure.spans.values() if s > 200)


def test_a_threshold_change_FAILS_against_base_even_when_the_constant_moves_with_it(repo, monkeypatch, capsys):
    """A PR that edits THRESHOLD and the field together passes the every-run check; the base comparison fails it."""
    monkeypatch.setattr(L, "THRESHOLD", 400)
    head(repo, entries={}, threshold=400)
    rc, out = run(repo, capsys)
    assert rc == 1 and "threshold changed 200 -> 400" in out


# ── review folds (executor + architecture lens, 2026-10-04) ──────────────────

_TWO_G = "import os\nif os.name:\n" + fn("g", 10, indent="    ") + "else:\n" + fn("g", 10, indent="    ")
G = ("src/maxim/a.py", "g")


def test_P1_a_pin_on_a_short_ambiguous_qualname_FAILS(repo, capsys):
    """A pin + from-null exception on two short conditional defs would pre-approve a later long def."""
    head(repo, _BASE_A + "\n\n" + _TWO_G, {A: 300, G: 250}, [exc("g", None, 250)])
    rc, out = run(repo, capsys)
    assert rc == 1 and "a.py::g: cannot pin an ambiguous qualname" in out


def test_P1b_a_pinned_function_that_becomes_two_short_defs_FAILS(repo, capsys):
    head(repo, "import os\nif os.name:\n" + fn("big", 10, indent="    ") + "else:\n" + fn("big", 10, indent="    "))
    rc, out = run(repo, capsys)
    assert rc == 1 and "a.py::big: cannot pin an ambiguous qualname" in out


def test_P1c_a_kept_pin_that_matched_no_single_def_at_base_is_held_to_the_new_entry_rule(tmp_path, monkeypatch, capsys):
    """The cash-out of a poisoned base (P1 merged before the every-run check): g becomes one 250-line def."""
    poisoned = baseline({A: 300, G: 250}, [exc("g", None, 250)])
    root = make_repo(tmp_path, monkeypatch, {"src/maxim/a.py": _BASE_A + "\n\n" + _TWO_G}, poisoned)
    _write(root, "src/maxim/a.py", _BASE_A + "\n\n" + fn("g", 250))
    rc, out = run(root, capsys)
    assert rc == 1 and "a.py::g: its merge-base pin 250 did not match a single def" in out


def test_P2_one_removed_entry_pays_for_one_free_move_only(repo, capsys):
    (repo / "src/maxim/a.py").unlink()
    _write(repo, "src/maxim/b.py", fn("big", 300))
    _write(repo, "src/maxim/c.py", fn("big", 300))
    head(repo, entries={B: 300, ("src/maxim/c.py", "big"): 300})
    rc, out = run(repo, capsys)
    assert rc == 1 and "c.py::big: new entry at 300" in out


def test_P2b_a_copy_is_not_a_free_move_while_the_original_still_exists(repo, capsys):
    _write(repo, "src/maxim/c.py", fn("bigcopy", 300))
    head(repo, fn("big", 150), {("src/maxim/c.py", "bigcopy"): 300})
    rc, out = run(repo, capsys)
    assert rc == 1 and "c.py::bigcopy: new entry at 300" in out


def test_a_bare_from_null_beside_a_pin_drop_FAILS(repo, capsys):
    """A decomposition must say where a piece's debt came from: split_from/moved_from, not a bare from-null."""
    piece = ("src/maxim/a.py", "piece")
    split = fn("big", 250) + "\n\n" + fn("piece", 240, tag="p")
    head(repo, split, {A: 250, piece: 240}, [exc("piece", None, 240)])
    rc, out = run(repo, capsys)
    assert rc == 1 and "bare from-null exception in a diff that lowers or removes pins" in out
    head(repo, split, {A: 250, piece: 240}, [exc("piece", None, 240, split_from={"file": A[0], "qualname": "big"})])
    assert run(repo, capsys)[0] == 0


def test_a_git_failure_inside_the_diff_rules_exits_2_not_a_traceback(repo, monkeypatch, capsys):
    def boom(*a, **k):
        raise L.GitUnavailable("simulated")

    monkeypatch.setattr(L, "measure_at_base", boom)
    head(repo, fn("big", 310), {A: 310}, [exc("big", 300, 310)])
    rc, out = run(repo, capsys)
    assert rc == 2 and "git could not read the merge-base" in out


def test_P2c_a_copy_is_not_free_while_the_original_survives_as_short_conditional_defs(repo, capsys):
    _write(repo, "src/maxim/c.py", fn("bigcopy", 300))
    head(
        repo,
        "import os\nif os.name:\n" + fn("big", 10, indent="    ") + "else:\n" + fn("big", 10, indent="    "),
        {("src/maxim/c.py", "bigcopy"): 300},
    )
    rc, out = run(repo, capsys)
    assert rc == 1 and "c.py::bigcopy: new entry at 300" in out


# ── delta-review folds (2026-10-04) ──────────────────────────────────────────

_S = {"file": "src/maxim/a.py", "qualname": "big"}
_P, _Q = ("src/maxim/a.py", "part"), ("src/maxim/a.py", "part2")


@pytest.fixture
def repo600(tmp_path: Path, monkeypatch) -> Path:
    return make_repo(
        tmp_path, monkeypatch, {"src/maxim/a.py": fn("big", 600)}, baseline({("src/maxim/a.py", "big"): 600})
    )


def test_G1_a_recorded_move_whose_source_survives_FAILS(repo600, capsys):
    """A split labelled as a move (big stays at 150) must not unlock a bare from-null sibling."""
    head(
        repo600,
        fn("big", 150) + "\n\n" + fn("part", 300) + "\n\n" + fn("part2", 300, tag="y"),
        {_P: 300, _Q: 300},
        [exc("part", 600, 300, moved_from=_S), exc("part2", None, 300)],
    )
    rc, out = run(repo600, capsys)
    assert rc == 1 and "unused exception for src/maxim/a.py::part (600 -> 300)" in out


def test_G1b_a_shrinking_recorded_move_counts_as_a_drop(repo600, capsys):
    """big deleted; part 'moved' 600->300 is really a split, so part2's bare from-null is refused."""
    head(
        repo600,
        fn("part", 300) + "\n\n" + fn("part2", 300, tag="y"),
        {_P: 300, _Q: 300},
        [exc("part", 600, 300, moved_from=_S), exc("part2", None, 300)],
    )
    rc, out = run(repo600, capsys)
    assert rc == 1 and "a.py::part2: new entry at 300 with a bare from-null exception" in out


def test_G3_a_free_move_plus_a_genuinely_new_function_passes(repo600, capsys):
    """A pure move is not a drop: a new function beside it may carry a bare from-null."""
    (repo600 / "src/maxim/a.py").unlink()
    _write(repo600, "src/maxim/b.py", fn("big", 600))
    _write(repo600, "src/maxim/c.py", fn("newf", 250, tag="n"))
    head(
        repo600,
        entries={B: 600, ("src/maxim/c.py", "newf"): 250},
        exceptions=[exc("newf", None, 250, file="src/maxim/c.py")],
    )
    rc, out = run(repo600, capsys)
    assert rc == 0, out


def test_K2_repairing_a_drifted_pin_needs_no_exception(tmp_path, monkeypatch, capsys):
    root = make_repo(
        tmp_path, monkeypatch, {"src/maxim/a.py": fn("big", 600)}, baseline({("src/maxim/a.py", "big"): 610})
    )
    head(root, entries={("src/maxim/a.py", "big"): 600})
    rc, out = run(root, capsys)
    assert rc == 0, out


def test_K2_a_raise_on_a_drifted_base_is_judged_against_the_measured_span(tmp_path, monkeypatch, capsys):
    root = make_repo(
        tmp_path, monkeypatch, {"src/maxim/a.py": fn("big", 600)}, baseline({("src/maxim/a.py", "big"): 610})
    )
    head(root, fn("big", 605), {("src/maxim/a.py", "big"): 605})
    rc, out = run(root, capsys)
    assert rc == 1 and "pin raised 600 -> 605 without a new exception" in out


def test_U1_a_pin_that_drifted_up_on_main_is_not_raised_without_an_exception(tmp_path, monkeypatch, capsys):
    """Main's pin (600) is below the function's span (610): keeping the drift as the new pin is a raise, judged
    from min(pin, span), so it needs an exception (``old = bs`` alone would accept it)."""
    root = make_repo(
        tmp_path, monkeypatch, {"src/maxim/a.py": fn("big", 610)}, baseline({("src/maxim/a.py", "big"): 600})
    )
    head(root, fn("big", 610), {("src/maxim/a.py", "big"): 610})
    rc, out = run(root, capsys)
    assert rc == 1 and "pin raised 600 -> 610 without a new exception" in out


def test_G2_one_source_cannot_be_both_moved_and_split(repo600, capsys):
    head(
        repo600,
        fn("part", 300) + "\n\n" + fn("part2", 300, tag="y"),
        {_P: 300, _Q: 300},
        [exc("part", 600, 300, moved_from=_S), exc("part2", None, 300, split_from=_S)],
    )
    rc, out = run(repo600, capsys)
    assert rc == 1 and "split_from src/maxim/a.py::big, but that base entry's pin did not drop" in out


def test_K3_repairing_a_drifted_pin_is_not_a_drop_a_split_can_claim(tmp_path, monkeypatch, capsys):
    """Main's pin (610) drifted above the function's span (600). Lowering it to 600 repairs the pin; the
    function lost no lines, so a new piece may not claim ``split_from`` against it."""
    root = make_repo(
        tmp_path, monkeypatch, {"src/maxim/a.py": fn("big", 600)}, baseline({("src/maxim/a.py", "big"): 610})
    )
    head(
        root,
        fn("big", 600) + "\n\n" + fn("piece", 250),
        {("src/maxim/a.py", "big"): 600, ("src/maxim/a.py", "piece"): 250},
        [exc("piece", None, 250, split_from={"file": "src/maxim/a.py", "qualname": "big"})],
    )
    rc, out = run(root, capsys)
    assert rc == 1 and "split_from src/maxim/a.py::big, but that base entry's pin did not drop" in out
