"""scripts/lint_coverage.py — coverage as a ratchet (roadmap 1.3.2 item 5, the gate half).

Every gate test drives ``main()`` on a fixture git repo (a tiny ``src/maxim`` with two floored
packages and one excluded file) plus a synthetic ``coverage.json`` in coverage 7's JSON shape.
Base is ``main``; HEAD is a ``feature`` branch, so the diff-scoped rules run as on a pull request.
"""

from __future__ import annotations

import json
import re
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import lint_coverage as L

N = 4  # min_package_statements in the fixture (the real value, 200, needs real packages)
A = "src/maxim/alpha/a.py"
B = "src/maxim/beta/b.py"
HW = "src/maxim/hw/robot.py"
# Built, not written out: the lint scans tests/ for these and must not trip on its own test.
IMPORT_COV = "import " + "coverage"
NO_COV = "--no-" + "cov"
ENV_COV = "COVERAGE" + "_FILE"


WORKFLOW = (
    "on: push\njobs:\n"
    "  unit-tests:\n"
    "    steps:\n"
    "      - uses: actions/checkout@v4\n"
    "        with:\n"
    "          fetch-depth: 0\n"
    "      - name: deps\n"
    f'        run: pip install "coverage=={L.COVERAGE_VERSION}"\n'
    "      - name: suite\n"
    "        run: |\n"
    f"          {L.CI_CLEAN}\n"
    "          python -m pytest tests/ -x \\\n"
    f"            {L.CI_COV_ARGS}\n"
    "      - name: Coverage ratchet\n"
    f"        {L.CI_LINT_RUN}\n"
    "  lint:\n"
    "    steps: []\n"
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


def _pyproject(omit: list[str] | None = None, extra: str = "") -> str:
    omit = L.FIXED_OMIT + ([HW] if omit is None else omit)
    return (
        '[project]\nname = "fixture"\n\n'
        '[tool.pytest.ini_options]\naddopts = "-v"\n\n'
        f'[tool.coverage.run]\nsource = ["src/maxim"]\nomit = {json.dumps(omit)}\nbranch = true\n\n'
        f"[tool.coverage.report]\nexclude_lines = {json.dumps(L.CANONICAL_EXCLUDE_LINES)}\nshow_missing = true\n"
        + extra
    )


def _floors(
    overall=(65.0, 7),
    alpha=(80.0, 2),
    beta=(50.0, 5),
    band=1.0,
    n=N,
    *,
    files: dict | None = None,
    retired: list | None = None,
    version: int = 2,
    **more,
) -> str:
    def f(v):
        return {"percent": v[0], "missing": v[1]} if v is not None else {"percent": None, "missing": None}

    pk = {"maxim/alpha": alpha, "maxim/beta": beta, **more}
    doc = {
        "floors_format_version": version,
        "band": band,
        "min_package_statements": n,
        "overall": f(overall),
        "packages": {k: f(v) for k, v in pk.items() if v != "drop"},
    }
    if version != 1:
        doc["files"] = {k: f(v) for k, v in (files or {}).items()}
        doc["retired_files"] = retired or []
    return json.dumps(doc, indent=1)


def _ledger(kind: str, file: str, count: int, ref: str | None = None, reason: str = "r") -> dict:
    return {"kind": kind, "file": file, "count": count, "reason": reason, "ref": ref}


BASE_LEDGER = [_ledger("excluded_statements", HW, 3), _ledger("pragma", B, 1)]
HW_ENTRY = {"path": HW, "reason": "needs a robot", "covered_by": None, "ref": "#1"}


def _exclusions(entries=None, ledger=None) -> str:
    return json.dumps(
        {
            "exclusions_format_version": 1,
            "exclusions": [HW_ENTRY] if entries is None else entries,
            "ledger": BASE_LEDGER if ledger is None else ledger,
        },
        indent=1,
    )


def _lines(n: int, prefix: str) -> str:
    return "".join(f"{prefix}{i} = {i}\n" for i in range(n))


def _auto(root: Path, rel: str, missing: set[int]) -> dict:
    """Every non-blank, non-comment line a statement; a line matching an exclude regex is excluded."""
    stmts, excluded = [], []
    for i, ln in enumerate((root / rel).read_text().splitlines(), 1):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if any(re.search(rx, ln) for rx in L.CANONICAL_EXCLUDE_LINES):
            excluded.append(i)
        else:
            stmts.append(i)
    return {
        "executed": [i for i in stmts if i not in missing],
        "missing": [i for i in stmts if i in missing],
        "excluded": excluded,
    }


def _cov(root: Path, overrides: dict | None = None, *, version=L.COVERAGE_VERSION, branch=True, missing=None) -> None:
    missing = {A: {9, 10}, B: {6, 7, 8, 9, 10}} if missing is None else missing
    files = {}
    for p in sorted((root / "src/maxim").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel == HW or rel in (overrides or {}) and overrides[rel] is None:
            continue
        spec = (overrides or {}).get(rel) or _auto(root, rel, missing.get(rel, set()))
        ex, mi = spec["executed"], spec["missing"]
        files[rel] = {
            "executed_lines": ex,
            "missing_lines": mi,
            "excluded_lines": spec.get("excluded", []),
            "summary": {
                "covered_lines": len(ex),
                "num_statements": len(ex) + len(mi),
                "num_branches": 0,
                "covered_branches": 0,
            },
        }
    meta = {"format": 3, "version": version, "timestamp": "t", "branch_coverage": branch, "show_contexts": False}
    (root / "coverage.json").write_text(json.dumps({"meta": meta, "files": files, "totals": {}}))


@pytest.fixture
def repo(tmp_path: Path, monkeypatch):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _write(tmp_path, ".gitignore", "coverage.json\n")
    _write(tmp_path, "pyproject.toml", _pyproject())
    _write(tmp_path, "src/maxim/__init__.py", "")
    _write(tmp_path, "src/maxim/alpha/__init__.py", "")
    _write(tmp_path, A, _lines(10, "a"))
    _write(tmp_path, "src/maxim/beta/__init__.py", "")
    _write(tmp_path, B, _lines(10, "b") + "zz = 0  # pragma: no cover\n")
    _write(tmp_path, HW, _lines(3, "h"))
    _write(tmp_path, "tests/test_x.py", "def test_x():\n    assert True\n")
    _write(tmp_path, ".github/workflows/test.yml", WORKFLOW)
    _write(tmp_path, L.FLOORS_REL, _floors())
    _write(tmp_path, L.EXCLUSIONS_REL, _exclusions())
    _commit(tmp_path)
    _git(tmp_path, "checkout", "-q", "-b", "feature")
    _cov(tmp_path)
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(L, "MIN_PACKAGE_STATEMENTS", N)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)  # CI's suite runs with it set; --local refuses it
    return tmp_path


def _run(capsys) -> tuple[int, str, str]:
    rc = L.main()
    c = capsys.readouterr()
    return rc, c.out, c.err


# ── baseline ─────────────────────────────────────────────────────────────────


def test_untouched_branch_is_clean_and_prints_totals(repo, capsys):
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert "overall: 65.00% of 20 statements, missing 7" in out
    assert "maxim/alpha: 80.00% of 10 statements" in out and "no executable src/maxim change" in out


# ── floors ───────────────────────────────────────────────────────────────────


def test_measured_below_a_floor_FAILS(repo, capsys):
    _cov(repo, missing={A: set(range(2, 11)), B: {6, 7, 8, 9, 10}})  # 7 statements under: beyond k = K_MIN = 5
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/alpha: 10.00% is below its floor 80.0% by more than the 5-statement" in err


def test_a_floor_more_than_the_band_below_the_measurement_FAILS(repo, capsys):
    _cov(repo, missing={A: set(), B: {6, 7, 8, 9, 10}})
    rc, _, err = _run(capsys)
    assert (
        rc == 1
        and 'maxim/alpha: 100.00% is more than 1.0 pt above its floor 80.0% (`python scripts/lint_coverage.py --pins-from-run <run-id>` writes it) — raise it to {"percent": 100.0'
        in err
    )


def test_raising_a_floor_passes(repo, capsys):
    _cov(repo, missing={A: set(), B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(75.0, 5), alpha=(100.0, 0)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_changed_floor_cannot_pin_missing_above_the_measurement(repo, capsys):
    _cov(repo, missing={A: set(), B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(75.0, 5), alpha=(100.0, 3)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/alpha: floor pins missing 3 above min(measured 0 + 5 of noise, base 2)" in err


@pytest.mark.parametrize("kw,needle", [({"band": 2.0}, "band"), ({"n": 9}, "min_package_statements")])
def test_band_or_N_change_FAILS(repo, capsys, kw, needle):
    _write(repo, L.FLOORS_REL, _floors(**kw))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{needle} must be" in err and f"{needle} changed" in err


def test_lowering_a_floor_after_DELETING_covered_code_passes(repo, capsys):
    _write(repo, A, _lines(8, "a"))  # drop two covered lines: 6/8 = 75%, missing still 2
    _cov(repo, missing={A: {7, 8}, B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(61.1, 7), alpha=(75.0, 2)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_lowering_a_floor_after_ADDING_uncovered_code_FAILS(repo, capsys):
    _write(repo, A, _lines(12, "a"))
    _cov(repo, missing={A: {9, 10, 11, 12}, B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(59.0, 9), alpha=(66.6, 4)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "floor lowered 80.0% -> 66.6% but measured missing 4 > the base's pinned missing 2" in err


def test_a_deleted_package_may_drop_its_floor(repo, capsys):
    _git(repo, "rm", "-q", "-r", "src/maxim/beta")
    _write(repo, L.FLOORS_REL, _floors(overall=(80.0, 2), beta="drop"))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}})
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_floor_removed_while_its_package_exists_FAILS_hysteresis(repo, capsys):
    _write(repo, L.FLOORS_REL, _floors(beta="drop"))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/beta: floor removed while its directory still exists" in err


def test_a_package_under_N_keeps_its_floor_hysteresis(repo, capsys):
    _write(repo, B, _lines(2, "b") + "zz = 0  # pragma: no cover\n")  # 2 statements < N
    _cov(repo, missing={A: {9, 10}, B: set()})
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/beta: 100.00% is more than 1.0 pt above its floor 50.0%" in err


def test_an_orphan_floor_FAILS(repo, capsys):
    _write(repo, L.FLOORS_REL, _floors(**{"maxim/ghost": (10.0, 1)}))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/ghost: floored package has no directory at HEAD (orphan)" in err


def test_a_new_package_over_N_needs_a_floor(repo, capsys):
    _write(repo, "src/maxim/gamma/g.py", _lines(5, "g"))
    _commit(repo)
    _cov(repo)
    rc, _, err = _run(capsys)
    assert (
        rc == 1
        and 'maxim/gamma: 5 statements (>= 4) and no floor — add "maxim/gamma": {"percent": 100.0, "missing": 0}' in err
    )


def test_a_package_rename_carries_its_floor(repo, capsys):
    _git(repo, "mv", "src/maxim/beta", "src/maxim/gamma")
    # moved code counts as changed (S6), so it arrives covered; the pragma moves with a ref-less entry (no rise)
    _write(repo, L.FLOORS_REL, _floors(overall=(90.0, 2), beta="drop", **{"maxim/gamma": (100.0, 0)}))
    _write(repo, L.EXCLUSIONS_REL, _exclusions(ledger=BASE_LEDGER + [_ledger("pragma", "src/maxim/gamma/b.py", 1)]))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}})
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_package_rename_cannot_lower_the_carried_floor(repo, capsys):
    _git(repo, "mv", "src/maxim/beta", "src/maxim/gamma")
    _write(repo, "src/maxim/gamma/b.py", _lines(10, "b") + "zz = 0  # pragma: no cover\nc0 = 0\nc1 = 1\n")
    _write(repo, L.FLOORS_REL, _floors(overall=(60.8, 9), beta="drop", **{"maxim/gamma": (41.6, 7)}))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, "src/maxim/gamma/b.py": {6, 7, 8, 9, 10, 13, 14}})
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/gamma (carrying maxim/beta): floor lowered 50.0% -> 41.6%" in err


def test_a_null_floor_fails_closed_and_prints_the_measurement(repo, capsys):
    _write(repo, L.FLOORS_REL, _floors(overall=None, alpha=None, beta=None))
    _commit(repo)
    rc, out, err = _run(capsys)
    assert rc == 1
    assert 'floor missing: overall measured 65.00% (missing 7) — pin {"percent": 65.0, "missing": 7}' in err
    assert '"maxim/beta": {\n      "percent": 50.0' in out  # the paste-ready measured file


def test_bootstrap_without_a_base_floors_file_checks_every_floor_as_new(tmp_path, monkeypatch, capsys):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _write(tmp_path, ".gitignore", "coverage.json\n")
    _write(tmp_path, "pyproject.toml", _pyproject(omit=["*/hw/robot.py"]))
    _write(tmp_path, ".github/workflows/test.yml", WORKFLOW)
    for rel, text in ((A, _lines(10, "a")), (B, _lines(10, "b") + "zz = 0  # pragma: no cover\n"), (HW, "h = 1\n")):
        _write(tmp_path, rel, text)
    _commit(tmp_path)
    _git(tmp_path, "checkout", "-q", "-b", "feature")
    _write(tmp_path, "pyproject.toml", _pyproject())
    _write(tmp_path, L.FLOORS_REL, _floors(overall=(65.0, 9)))
    _write(
        tmp_path,
        L.EXCLUSIONS_REL,
        _exclusions(
            entries=[dict(HW_ENTRY, ref=None)], ledger=[_ledger("excluded_statements", HW, 1), _ledger("pragma", B, 1)]
        ),
    )
    _commit(tmp_path)
    _cov(tmp_path)
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(L, "MIN_PACKAGE_STATEMENTS", N)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    rc, out, err = _run(capsys)
    # the grandfathered omit needs no ref; the overall floor's pinned missing (9) exceeds the measured 7
    assert rc == 1 and "NEW EXCLUSION" not in out
    assert err.count("  - ") == 1 and "overall: floor pins missing 9 above the measured 7" in err


# ── diff coverage ────────────────────────────────────────────────────────────


def test_diff_coverage_below_80_FAILS_and_lists_uncovered_lines(repo, capsys):
    _write(repo, A, _lines(10, "a") + "n0 = 0\nn1 = 1\nn2 = 2\n")
    _commit(repo)
    _cov(repo, missing={A: {9, 10, 12, 13}, B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(56.5, 9), alpha=(69.2, 4)))
    _commit(repo)
    rc, out, err = _run(capsys)
    assert rc == 1 and "diff coverage 33.3% is below 80%" in err
    assert f"UNCOVERED {A}:12: n1 = 1" in out and f"UNCOVERED {A}:13: n2 = 2" in out


def test_diff_coverage_at_80_passes(repo, capsys):
    _write(repo, A, _lines(10, "a") + "".join(f"n{i} = {i}\n" for i in range(5)))
    _write(repo, L.FLOORS_REL, _floors(overall=(68.0, 7), alpha=(80.0, 2)))  # the missing pin never rises
    _commit(repo)
    _cov(repo, missing={A: {9, 10, 15}, B: {6, 7, 8, 9, 10}})
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert "diff coverage 4/5 changed executable statements = 80.0%" in out


def test_a_multi_line_statement_maps_to_its_first_line(repo, capsys):
    _write(repo, A, _lines(10, "a") + "big = (\n    1\n    + 2\n)\n")
    _commit(repo)
    spec = _auto(repo, A, {9, 10})
    spec["executed"] = [ln for ln in spec["executed"] if ln <= 11]  # coverage lists only line 11
    _cov(repo, {A: spec})
    _write(repo, L.FLOORS_REL, _floors(overall=(66.6, 7), alpha=(81.8, 2)))
    _commit(repo)
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert "diff coverage 1/1 changed executable statements = 100.0%" in out


def test_comments_blank_lines_and_docstrings_are_not_executable(repo, capsys):
    _write(repo, A, '"""doc\nmore doc"""\n' + _lines(10, "a") + "# a comment\n\n")
    _commit(repo)
    spec = _auto(repo, A, set())
    spec["executed"] = [ln for ln in spec["executed"] if ln > 2]  # coverage does not list a docstring
    spec["missing"] = [11, 12]
    spec["executed"] = [ln for ln in spec["executed"] if ln not in (11, 12)]
    _cov(repo, {A: spec})
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert "no executable src/maxim change" in out


def test_a_changed_continuation_line_of_an_uncovered_statement_counts(repo, capsys):
    """S1: coverage lists only a statement's first line, so a changed line 2 must map back to it."""
    _write(repo, A, _lines(9, "a") + "big = (\n    1\n)\n")
    _commit(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _write(repo, A, _lines(9, "a") + "big = (\n    2\n)\n")
    _commit(repo)
    _cov(repo, {A: {"executed": [1, 2, 3, 4, 5, 6, 7, 8], "missing": [9, 10]}})
    rc, out, err = _run(capsys)
    assert rc == 1 and "diff coverage 0.0% is below 80%" in err and f"UNCOVERED {A}:10: big = (" in out


def test_a_comment_added_inside_a_function_body_is_not_executable(repo, capsys):
    """Without the code-line filter the comment maps to its enclosing `def` (executed) and counts."""
    body = _lines(8, "a") + "def g():\n    return 1\n"
    _write(repo, A, body)
    _commit(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _write(repo, A, _lines(8, "a") + "def g():\n    # why\n    return 1\n")
    _commit(repo)
    _cov(repo, {A: {"executed": [1, 2, 3, 4, 5, 6, 7, 9], "missing": [8, 11]}})
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert "no executable src/maxim change" in out


def test_a_changed_file_absent_from_coverage_json_FAILS(repo, capsys):
    _write(repo, "src/maxim/alpha/unseen.py", "u = 1\n")
    _commit(repo)
    _cov(repo, {"src/maxim/alpha/unseen.py": None})
    rc, _, err = _run(capsys)
    assert rc == 1 and "src/maxim/alpha/unseen.py: changed but absent from coverage.json" in err


def test_a_changed_line_coverage_EXCLUDES_counts_as_uncovered(repo, capsys):
    _write(repo, A, _lines(10, "a") + "def f():  # pragma: no cover\n    return 1\n")
    _commit(repo)
    spec = _auto(repo, A, {9, 10})
    spec["excluded"] = [11, 12]
    spec["executed"] = [ln for ln in spec["executed"] if ln < 11]
    _cov(repo, {A: spec})
    _write(repo, L.EXCLUSIONS_REL, _exclusions(ledger=BASE_LEDGER + [_ledger("pragma", A, 1, "#7")]))
    _commit(repo)
    rc, out, err = _run(capsys)
    assert rc == 1 and "diff coverage 0.0% is below 80%" in err and f"UNCOVERED {A}:11" in out


def test_a_TYPE_CHECKING_imports_only_block_is_not_counted(repo, capsys):
    text = _lines(10, "a") + "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from os import path\n"
    _write(repo, A, text)
    _commit(repo)
    spec = _auto(repo, A, {9, 10})
    spec["excluded"] = [12, 13]
    spec["executed"] = [ln for ln in spec["executed"] if ln != 13]
    _cov(repo, {A: spec})
    _write(repo, L.FLOORS_REL, _floors(overall=(66.6, 7), alpha=(81.8, 2)))
    _commit(repo)
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert "diff coverage 1/1" in out  # the `from typing` line only


def test_a_TYPE_CHECKING_block_with_code_counts_as_uncovered(repo, capsys):
    text = _lines(10, "a") + "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    x = 1\n"
    _write(repo, A, text)
    _commit(repo)
    spec = _auto(repo, A, {9, 10})
    spec["excluded"] = [12, 13]
    spec["executed"] = [ln for ln in spec["executed"] if ln != 13]
    _cov(repo, {A: spec})
    _write(repo, L.FLOORS_REL, _floors(overall=(66.6, 7), alpha=(81.8, 2)))
    _write(repo, L.EXCLUSIONS_REL, _exclusions(ledger=BASE_LEDGER + [_ledger("pragma", A, 1, "#7")]))
    _commit(repo)
    rc, out, err = _run(capsys)
    assert rc == 1 and f"UNCOVERED {A}:12" in out and f"UNCOVERED {A}:13" in out


def test_a_moved_line_counts_as_changed(repo, capsys):
    """S6: rename detection off — a pure file move makes every line changed."""
    _git(repo, "mv", A, "src/maxim/alpha/moved.py")
    _commit(repo)
    _cov(repo, missing={"src/maxim/alpha/moved.py": {9, 10}, B: {6, 7, 8, 9, 10}})
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert "diff coverage 8/10 changed executable statements = 80.0%" in out


# ── configuration (S4) ───────────────────────────────────────────────────────


def test_a_tampered_coverage_table_FAILS(repo, capsys):
    _write(repo, "pyproject.toml", _pyproject().replace("branch = true", "branch = false"))
    rc, _, err = _run(capsys)
    assert rc == 1 and "[tool.coverage] differs from the canonical table" in err


def test_an_extra_coverage_subtable_FAILS(repo, capsys):
    _write(repo, "pyproject.toml", _pyproject(extra='\n[tool.coverage.paths]\nsource = ["x"]\n'))
    rc, _, err = _run(capsys)
    assert rc == 1 and "[tool.coverage] differs" in err


def test_omit_not_equal_to_the_exclusion_list_FAILS(repo, capsys):
    _write(repo, "pyproject.toml", _pyproject(omit=[HW, B]))
    rc, _, err = _run(capsys)
    assert rc == 1 and "[tool.coverage] differs" in err


def test_a_fixed_glob_matching_a_src_file_FAILS(repo, capsys):
    _write(repo, "src/maxim/alpha/tests/t.py", "t = 1\n")
    rc, _, err = _run(capsys)
    assert rc == 1 and "src/maxim/alpha/tests/t.py: matched by the fixed omit glob '*/tests/*'" in err


@pytest.mark.parametrize(
    "rel,text",
    [
        (".coveragerc", "[run]\nomit = *\n"),
        ("setup.cfg", "[coverage:run]\nomit = *\n"),
        ("tox.ini", "[coverage:report]\n"),
    ],
)
def test_a_second_coverage_config_file_FAILS(repo, capsys, rel, text):
    _write(repo, rel, text)
    rc, _, err = _run(capsys)
    assert rc == 1 and "coverage configuration lives only in pyproject.toml" in err


@pytest.mark.parametrize(
    "rel,text,needle",
    [
        ("pyproject.toml", _pyproject().replace('addopts = "-v"', f'addopts = "-v {NO_COV}"'), NO_COV),
        ("pytest.ini", "[pytest]\naddopts = --cov-" + "config=x\n", "--cov-" + "config"),
        (
            ".github/workflows/test.yml",
            WORKFLOW + "env:\n  PYTEST_ADDOPTS: --cov-" + "fail-under=1\n",
            "--cov-" + "fail-under",
        ),
        (".github/workflows/test.yml", WORKFLOW + f"env:\n  {ENV_COV}: x\n", ENV_COV),
        # a composite action runs inside a job, so it is read too (#1117 review)
        (".github/actions/setup/action.yml", f"runs:\n  using: composite\nenv:\n  {ENV_COV}: x\n", ENV_COV),
        (".github/actions/setup/action.yml", "runs:\n  steps:\n    - run: pytest --no-" + "cov\n", "--no-" + "cov"),
        ("tests/test_y.py", f"{IMPORT_COV}\n", "imports or drives the coverage API"),
        ("src/maxim/alpha/sneaky.py", "def f(c):\n    return c.Coverage" + ".current()\n", "imports or drives"),
    ],
)
def test_banned_options_env_and_imports_FAIL(repo, capsys, rel, text, needle):
    _write(repo, rel, text)
    _cov(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and needle in err


# ── coverage.json staleness (S8) ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "kw,needle",
    [({"version": "7.0.0"}, "was written by coverage '7.0.0'"), ({"branch": False}, "branch coverage")],
)
def test_a_stale_coverage_json_fails_closed(repo, capsys, kw, needle):
    _cov(repo, **kw)
    rc, _, err = _run(capsys)
    assert rc == 2 and needle in err


def test_a_coverage_json_naming_a_missing_file_fails_closed(repo, capsys):
    _cov(repo)
    (repo / B).unlink()
    rc, _, err = _run(capsys)
    assert rc == 2 and "does not exist (stale file?)" in err


def test_no_coverage_json_fails_closed(repo, capsys):
    (repo / "coverage.json").unlink()
    rc, _, err = _run(capsys)
    assert rc == 2 and "coverage.json not found" in err


# ── exclusions + ledger (S3, S5) ─────────────────────────────────────────────


def test_a_new_exclusion_without_a_ref_FAILS(repo, capsys):
    new = {"path": B, "reason": "hard", "covered_by": None, "ref": None}
    _write(repo, "pyproject.toml", _pyproject(omit=[HW, B]))
    _write(
        repo,
        L.EXCLUSIONS_REL,
        _exclusions([HW_ENTRY, new], BASE_LEDGER + [_ledger("excluded_statements", B, 10, "#9")]),
    )
    _write(repo, L.FLOORS_REL, _floors(overall=(80.0, 2), beta="drop"))
    _commit(repo)
    _cov(repo, {B: None})
    rc, out, err = _run(capsys)
    assert rc == 1 and f"NEW EXCLUSION {B}" in out and f"exclusion {B}: a new exclusion needs a ref" in err


def test_a_new_exclusion_with_refs_passes(repo, capsys):
    new = {"path": "src/maxim/alpha/hw2.py", "reason": "camera", "covered_by": None, "ref": "#9"}
    _write(repo, new["path"], "c = 1\n")
    _write(repo, "pyproject.toml", _pyproject(omit=[HW, new["path"]]))
    _write(
        repo,
        L.EXCLUSIONS_REL,
        _exclusions([HW_ENTRY, new], BASE_LEDGER + [_ledger("excluded_statements", new["path"], 1, "#9")]),
    )
    _commit(repo)
    _cov(repo, {new["path"]: None})
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert "NEW EXCLUSION src/maxim/alpha/hw2.py (ref #9)" in out


def test_an_excluded_file_growing_without_a_ledger_entry_FAILS(repo, capsys):
    _write(repo, HW, _lines(5, "h"))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"exclusion {HW}: 5 statements exceed its ledger allowance 3" in err
    assert f"{HW}: excluded_statements count rose 3 -> 5" in err


def test_an_excluded_file_growing_with_a_refd_entry_passes(repo, capsys):
    _write(repo, HW, _lines(5, "h"))
    _write(repo, L.EXCLUSIONS_REL, _exclusions(ledger=BASE_LEDGER + [_ledger("excluded_statements", HW, 5, "#12")]))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_ledger_rise_without_a_ref_FAILS(repo, capsys):
    _write(repo, HW, _lines(5, "h"))
    _write(repo, L.EXCLUSIONS_REL, _exclusions(ledger=BASE_LEDGER + [_ledger("excluded_statements", HW, 5)]))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "raises the count: it needs a ref" in err


def test_a_pragma_rise_without_a_ledger_entry_FAILS(repo, capsys):
    _write(repo, B, _lines(10, "b") + "zz = 0  # pragma: no cover\nyy = 0  # pragma: no cover\n")
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{B}: 2 coverage-exclusion matches exceed its ledger allowance 1" in err
    assert f"{B}: pragma count rose 1 -> 2" in err


def test_every_exclude_regex_counts_not_only_the_pragma(repo, capsys):
    _write(repo, A, _lines(10, "a") + "if __name__ == '__main__':\n    pass\n")
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{A}: pragma count rose 0 -> 1" in err


def test_a_pragma_rise_with_a_refd_ledger_entry_passes(repo, capsys):
    _write(repo, B, _lines(10, "b") + "zz = 0  # pragma: no cover\nyy = 0  # pragma: no cover\n")
    _write(
        repo,
        L.EXCLUSIONS_REL,
        _exclusions(ledger=BASE_LEDGER + [_ledger("pragma", B, 2, "https://github.com/o/r/pull/5")]),
    )
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_dropped_then_re_raised_pragma_count_needs_a_new_entry(repo, capsys):
    """The allowance is an upper bound; the gate compares against the merge-base's MEASURED count."""
    _write(repo, B, _lines(10, "b"))
    _commit(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _write(repo, B, _lines(10, "b") + "zz = 0  # pragma: no cover\n")  # back to the ledger's 1
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{B}: pragma count rose 0 -> 1" in err


def test_a_new_ledger_entry_must_equal_the_measured_count(repo, capsys):
    _write(repo, L.EXCLUSIONS_REL, _exclusions(ledger=BASE_LEDGER + [_ledger("pragma", A, 5, "#3")]))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "must equal the measured count (0) — no pre-approving a later rise" in err


def test_the_ledger_is_append_only(repo, capsys):
    _write(
        repo,
        L.EXCLUSIONS_REL,
        _exclusions(ledger=[_ledger("excluded_statements", HW, 3, reason="edited"), BASE_LEDGER[1]]),
    )
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "ledger is append-only" in err


def test_an_exclusion_without_a_reason_FAILS(repo, capsys):
    _write(repo, L.EXCLUSIONS_REL, _exclusions([dict(HW_ENTRY, reason=" ")]))
    rc, _, err = _run(capsys)
    assert rc == 1 and "exclusion needs a non-empty reason" in err


# ── git / events ─────────────────────────────────────────────────────────────


def test_no_merge_base_on_a_pull_request_FAILS(repo, capsys, monkeypatch):
    _git(repo, "branch", "-m", "main", "elsewhere")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    rc, _, err = _run(capsys)
    assert rc == 2 and "diff-scoped check cannot run on a pull request" in err


def test_a_push_is_judged_against_the_last_green_push(repo, capsys, monkeypatch):
    """Was head rules only on push (#1089): uncovered code that reached main without a PR passed."""
    from tests.unit._push_event_helpers import fake_push

    _write(repo, A, _lines(10, "a") + "n0 = 0\nn1 = 1\nn2 = 2\n")  # fails diff coverage
    _commit(repo)
    _cov(repo, missing={A: {9, 10, 11, 12, 13}, B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(56.5, 10), alpha=(61.5, 5)))
    fake_push(monkeypatch, repo)
    rc, out, err = _run(capsys)
    assert rc == 1 and "diff coverage" in err and "head rules only" not in out, (out, err)


def test_another_ci_event_runs_the_head_rules_only(repo, capsys, monkeypatch):
    _write(repo, A, _lines(10, "a") + "n0 = 0\nn1 = 1\nn2 = 2\n")  # would fail diff coverage
    _commit(repo)
    _cov(repo, missing={A: {9, 10, 11, 12, 13}, B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(56.5, 10), alpha=(61.5, 5)))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert "schedule event — head rules only" in out


def test_the_committed_files_parse_and_match_pyproject():
    """The real files: the exclusion list equals pyproject's omit tail; every ledger entry names a real file."""
    root = Path(__file__).resolve().parents[2]
    ex = L.parse_exclusions((root / L.EXCLUSIONS_REL).read_text())
    fl = L.parse_floors((root / L.FLOORS_REL).read_text())
    assert fl.band == L.BAND and fl.min_statements == L.MIN_PACKAGE_STATEMENTS
    assert L.config_rules(root, ex.paths) == []
    assert all((root / x["file"]).is_file() for x in ex.ledger)


# ── review folds (2026-10-04): verified floors, clause headers, ledger exemptions, CI wiring ─


@pytest.mark.parametrize("overall", [(64.0, 7), (64.9, 7), (59.9, 7)])
def test_a_floors_only_PR_cannot_lower_a_floor_at_all(repo, capsys, overall):
    """The review's DO-NOT-MERGE probes: no code change, a floor lowered with missing unchanged. A lowered floor
    goes exactly to round_down(measured), and without a code change that is where it already is."""
    _write(repo, L.FLOORS_REL, _floors(overall=overall))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"overall: floor lowered to {overall[0]}%, below the measurement 65.00%" in err


def test_a_package_rename_cannot_carry_a_floor_lowered_by_a_fraction(repo, capsys):
    _git(repo, "mv", "src/maxim/beta", "src/maxim/gamma")
    _write(repo, L.FLOORS_REL, _floors(overall=(90.0, 2), beta="drop", **{"maxim/gamma": (49.9, 0)}))
    _write(repo, L.EXCLUSIONS_REL, _exclusions(ledger=BASE_LEDGER + [_ledger("pragma", "src/maxim/gamma/b.py", 1)]))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}})
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/gamma (carrying maxim/beta): floor lowered to 49.9%, below the measurement 100.00%" in err


def test_a_new_floor_cannot_start_low(repo, capsys):
    _write(repo, "src/maxim/gamma/g.py", _lines(1000, "g"))  # k = 5 statements = 0.5 pt
    _write(repo, L.FLOORS_REL, _floors(overall=(99.3, 7), **{"maxim/gamma": (99.4, 0)}))
    _commit(repo)
    _cov(repo)
    rc, _, err = _run(capsys)
    assert (
        rc == 1
        and "maxim/gamma: new floor 99.4% is below the measured 100.00% less 5 statement(s) of run-to-run noise (99.5%)"
        in err
    )


def test_merge_floors_takes_the_per_scope_minimum(tmp_path, capsys):
    a = {"overall": {"percent": 65.9, "missing": 100}, "packages": {"maxim/x": {"percent": 50.0, "missing": 9}}}
    b = {"overall": {"percent": 65.8, "missing": 101}, "packages": {"maxim/x": {"percent": 50.1, "missing": 8}}}
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))
    assert L.main(["--merge-floors", "--all-scopes", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["overall"] == {"percent": 65.8, "missing": 100}
    assert out["packages"]["maxim/x"] == {"percent": 50.0, "missing": 8}


_CLAUSE_BASE = (
    "def f(x):\n"  # 1
    "    try:\n"  # 2
    "        y = 1\n"  # 3
    "    except ValueError:\n"  # 4
    "        y = 2\n"  # 5
    "    else:\n"  # 6
    "        y = 3\n"  # 7
    "    if x:\n"  # 8
    "        a = 1\n"  # 9
    "    else:\n"  # 10
    "        a = 2\n"  # 11
    "    return y\n"  # 12
)


@pytest.mark.parametrize(
    "old,new,uncovered",
    [
        ("    except ValueError:\n", "    except (ValueError, KeyError):\n", 5),
        ("    else:\n        y = 3\n", "    else:  # widened\n        y = 3\n", 7),
        ("    else:\n        a = 2\n", "    else:  # widened\n        a = 2\n", 11),
    ],
)
def test_a_changed_clause_header_maps_to_its_block_not_the_enclosing_statement(repo, capsys, old, new, uncovered):
    """The review's except/else probe: the header used to map to the enclosing try/if, executed, so 1/1."""
    _write(repo, A, _lines(10, "a") + _CLAUSE_BASE)
    _commit(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _write(repo, A, _lines(10, "a") + _CLAUSE_BASE.replace(old, new))
    _commit(repo)
    # try/if executed; the handler, the try-else and the if-else bodies did not run
    executed = [*range(1, 9), 11, 12, 13, 18, 19, 22]
    _cov(repo, {A: {"executed": executed, "missing": [14, 15, 17, 21]}})
    rc, out, err = _run(capsys)
    assert f"UNCOVERED {A}:{10 + uncovered}" in out, out
    assert "diff coverage 0/1" in out


@pytest.mark.parametrize(
    "src,count",
    [
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import os\n", 0),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    x = 1\n", 1),
        ("def f():\n    raise NotImplementedError\n", 0),
        ("def f():\n    raise NotImplementedError('x')\n", 0),
        ("def f(x):\n    if x: raise NotImplementedError\n", 1),
        ("if __name__ == '__main__':\n    pass\n", 1),
        ("x = 1  # pragma: no cover\n", 1),
    ],
)
def test_the_pragma_count_exempts_only_shapes_that_hide_nothing(src, count):
    assert L.exclusion_matches(src) == count


def test_a_new_imports_only_TYPE_CHECKING_block_and_a_stub_need_no_ledger_entry(repo, capsys):
    _write(
        repo,
        "src/maxim/alpha/split.py",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import os\n\n\ndef f():\n    raise NotImplementedError\n",
    )
    _commit(repo)
    _cov(repo, {"src/maxim/alpha/split.py": {"executed": [1, 2, 6], "missing": [], "excluded": [2, 3, 7]}})
    _write(repo, L.FLOORS_REL, _floors(overall=(69.5, 7), alpha=(84.6, 2)))
    _commit(repo)
    rc, _, err = _run(capsys)
    # no ledger failure; the stub's excluded raise still counts as an uncovered changed line (S3)
    assert rc == 1 and "ledger" not in err and "coverage-exclusion matches" not in err
    assert err.count("  - ") == 1 and "diff coverage 75.0%" in err


def test_a_covered_by_claim_is_rejected_while_no_lane_measures_it(repo, capsys):
    _write(repo, L.EXCLUSIONS_REL, _exclusions([dict(HW_ENTRY, covered_by="nightly-model-cache")]))
    rc, _, err = _run(capsys)
    assert rc == 1 and "covered_by must be null" in err


_HUB = "      - name: MemoryHub\n        run: python -m pytest tests/integration/test_memory_hub.py -q\n"
_GATE = f"      - name: Coverage ratchet\n        {L.CI_LINT_RUN}\n"


@pytest.mark.parametrize(
    "old,new,needle",
    [
        (_GATE, "", "needs exactly one step whose run is"),
        (f"        {L.CI_LINT_RUN}\n", f"        {L.CI_LINT_RUN} || true\n", "needs exactly one step whose run is"),
        (_GATE, _GATE + "      - name: again\n" + f"        {L.CI_LINT_RUN}\n", "needs exactly one step whose run is"),
        (_GATE, _GATE.replace("ratchet\n", "ratchet\n        if: false\n"), "may carry only `name` and `run`"),
        (_GATE, _GATE.replace("ratchet\n", "ratchet\n        shell: bash -c 'true' {0}\n"), "may carry only"),
        (_GATE, _GATE.replace("ratchet\n", "ratchet\n        env:\n          X: y\n"), "may carry only"),
        (
            _GATE,
            _GATE.replace("ratchet\n", "ratchet\n        continue-on-error: true\n"),
            "continue-on-error is banned",
        ),
        ("  unit-tests:\n", "  unit-tests:\n    continue-on-error: true\n", "continue-on-error is banned"),
        (f"            {L.CI_COV_ARGS}\n", "            --cov=src/maxim\n", "needs exactly one measuring step"),
        (f"          {L.CI_CLEAN}\n", "", "needs exactly one measuring step"),
        ("          fetch-depth: 0\n", "          fetch-depth: 1\n", "fetch-depth: 0"),
        ("        with:\n          fetch-depth: 0\n", "        # fetch-depth: 0\n", "fetch-depth: 0"),
        (f'"coverage=={L.COVERAGE_VERSION}"', '"coverage"', "exact coverage pin"),
        (_GATE, "      - name: rewrite\n        run: echo '{}' > coverage.json\n" + _GATE, "sits between"),
        ("on: push\njobs:\n", "on: push\ndefaults:\n  run:\n    shell: bash\njobs:\n", "`defaults`"),
        ("  unit-tests:\n", "  unit-tests:\n    defaults:\n      run:\n        shell: bash\n", "`defaults`"),
        ("  unit-tests:\n", "  unit-tests:\n    if: false\n", "job-level `if:`"),
        (
            _GATE,
            _GATE.replace("        run:", "        run: >-\n          ").replace("\n          python", " python"),
            None,
        ),
    ],
)
def test_the_gate_checks_its_own_CI_step(repo, capsys, old, new, needle):
    assert old in WORKFLOW
    _write(repo, ".github/workflows/test.yml", WORKFLOW.replace(old, new))
    rc, _, err = _run(capsys)
    if needle is None:
        assert rc == 0, err  # the parsed run, not its YAML spelling, is what counts
    else:
        assert rc == 1 and needle in err


def test_the_MemoryHub_step_may_sit_between_the_suite_and_the_gate(repo, capsys):
    _write(repo, ".github/workflows/test.yml", WORKFLOW.replace(_GATE, _HUB + _GATE))
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_the_gate_step_must_follow_the_measuring_suite(repo, capsys):
    moved = WORKFLOW.replace(_GATE, "").replace("      - name: suite\n", _GATE + "      - name: suite\n")
    _write(repo, ".github/workflows/test.yml", moved)
    rc, _, err = _run(capsys)
    assert rc == 1 and "must come after the fast suite" in err


def test_a_floor_cannot_walk_down_across_a_chain_of_PRs(repo, capsys):
    """The delta review's walk: pin missing a tick HIGH, then drop that many covered statements and lower a
    tick, re-pinning high each time. Each link exited 0 under the two-sided tolerance."""
    # link 1: a floors-only re-pin with missing above the measurement
    _write(repo, L.FLOORS_REL, _floors(alpha=(80.0, 3)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/alpha: floor pins missing 3 above min(measured 2 + 5 of noise, base 2)" in err
    # suppose link 1 had merged anyway (the old rule passed it): link 2 must still fail
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _cov(repo, missing={A: {8, 9, 10}, B: {6, 7, 8, 9, 10}})  # one covered statement's test dropped
    _write(repo, L.FLOORS_REL, _floors(overall=(60.0, 9), alpha=(70.0, 4)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/alpha: floor pins missing 4 above min(measured 3 + 5 of noise, base 3)" in err
    # link 3: the same drop pinned honestly still fails — the drop rule holds against the honest overall pin
    _write(repo, L.FLOORS_REL, _floors(overall=(60.0, 8), alpha=(70.0, 3)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert (
        rc == 1 and "overall: floor lowered 65.0% -> 60.0% but measured missing 8 > the base's pinned missing 7" in err
    )


def _printed_floors(out: str) -> dict:
    i = out.index("review before committing):\n") + len("review before committing):\n")
    return json.JSONDecoder().raw_decode(out[i:])[0]


def test_the_two_run_minimum_of_a_small_package_passes_against_either_run(repo, capsys, tmp_path):
    """The `analysis` case: 658 statements, runs one line apart (86.93% / 86.78%). With a flat 0.1-pt
    percent tolerance the merged 86.7 was rejected against the 86.93% run."""
    big = "src/maxim/gamma/g.py"
    _write(repo, big, _lines(658, "g"))
    _write(repo, L.FLOORS_REL, _floors(overall=None, alpha=None, beta=None))
    _commit(repo)
    _git(repo, "checkout", "-q", "main")  # the bootstrap shape: the base carries null floors
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    runs = []
    for covered in (572, 571):
        _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, big: set(range(covered + 1, 659))})
        _write(repo, L.FLOORS_REL, _floors(overall=None, alpha=None, beta=None))
        _, out, _ = _run(capsys)
        f = tmp_path / f"run{covered}.json"
        f.write_text(json.dumps(_printed_floors(out)))
        runs.append(f)
    assert L.main(["--merge-floors", *map(str, runs)]) == 0
    merged = json.loads(capsys.readouterr().out)
    assert merged["packages"]["maxim/gamma"] == {"percent": 86.7, "missing": 86}
    assert set(merged) == {"overall", "packages"}  # only the new scopes (SF-A4): copy only the new entries
    doc = json.loads((repo / L.FLOORS_REL).read_text())
    doc["overall"] = merged["overall"]
    doc["packages"].update(merged["packages"])
    _write(repo, L.FLOORS_REL, json.dumps(doc))
    _commit(repo)
    for covered in (572, 571):
        _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, big: set(range(covered + 1, 659))})
        rc, _, err = _run(capsys)
        assert rc == 0, (covered, err)


def test_a_changed_path_with_a_space_is_read(repo, capsys):
    rel = "src/maxim/alpha/sp ace.py"
    _write(repo, rel, "s0 = 0\ns1 = 1\n")
    _commit(repo)
    _cov(repo, {rel: {"executed": [1], "missing": [2]}})
    rc, out, err = _run(capsys)
    assert f"UNCOVERED {rel}:2: s1 = 1" in out and "diff coverage 50.0%" in err


def test_the_lower_then_repin_walk_fails_at_every_link(repo, capsys):
    """walk2 (delta round 3): link A lowers a floor with no code change to the k-statement tolerance, keeping
    missing; link B drops tests for k statements and re-pins missing up to the measurement. Each passed alone."""
    big = "src/maxim/gamma/g.py"
    _write(repo, big, _lines(1000, "g"))
    _write(repo, L.FLOORS_REL, _floors(overall=(98.3, 17), **{"maxim/gamma": (99.0, 10)}))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, big: set(range(991, 1001))})
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    rc, _, err = _run(capsys)
    assert rc == 0, err
    # link A: lower 99.0 -> 98.9 with no code change (k = 1 statement for 1000)
    _write(repo, L.FLOORS_REL, _floors(overall=(98.3, 17), **{"maxim/gamma": (98.9, 10)}))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/gamma: floor lowered to 98.9%, below the measurement 99.00%" in err
    # link B, as if A had merged: drop one covered statement's test and re-pin missing up
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, big: set(range(990, 1001))})
    _write(repo, L.FLOORS_REL, _floors(overall=(98.3, 17), **{"maxim/gamma": (98.9, 11)}))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/gamma: floor pins missing 11 above min(measured 11 + 5 of noise, base 10)" in err
    # and the lower half again, from there: below round_down(measured) with measured missing above base
    _write(repo, L.FLOORS_REL, _floors(overall=(98.3, 17), **{"maxim/gamma": (98.8, 10)}))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "floor lowered 98.9% -> 98.8% but measured missing 11 > the base's pinned missing 10" in err


def test_deleting_covered_code_lowers_a_floor_to_round_down_of_the_measurement(repo, capsys):
    _write(repo, A, _lines(8, "a"))  # 6/8 = 75.0%, missing unchanged at 2
    _cov(repo, missing={A: {7, 8}, B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(61.1, 7), alpha=(74.9, 2)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/alpha: floor lowered to 74.9%, below the measurement 75.00%" in err
    _write(repo, L.FLOORS_REL, _floors(overall=(61.1, 7), alpha=(75.0, 2)))
    _commit(repo)
    assert _run(capsys)[0] == 0


def test_raising_a_floor_cannot_raise_its_missing_pin(repo, capsys):
    _write(repo, A, _lines(10, "a") + "".join(f"n{i} = {i}\n" for i in range(5)))
    _commit(repo)
    _cov(repo, missing={A: {9, 10, 15}, B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(68.0, 8), alpha=(80.0, 3)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "overall: floor pins missing 8 above min(measured 8 + 5 of noise, base 7)" in err


def test_following_the_raise_hint_passes_the_next_run(repo, capsys):
    """The ratchet's two halves must agree (#1409, 2026-10-20): a PR whose package rose more than the band above its
    floor while adding statements (measured missing 3 > base pin 2) was told to pin the MEASURED missing, and the
    next run refused that pin ("a pinned missing never rises"). With the floor unchanged in the PR (the CI-fix case),
    following the raise hints verbatim must pass. Rename-carried targets are M60's (the redesign of this bookkeeping)."""
    import re

    _write(repo, A, _lines(50, "a"))
    _commit(repo)
    _cov(repo, missing={A: {9, 10, 15}, B: {6, 7, 8, 9, 10}})
    rc, _, err = _run(capsys)
    hints = dict(re.findall(r"^\s*(?:- )?(\S+): .* raise it to (\{.*\})$", err, re.M))
    assert rc == 1 and {"overall", "maxim/alpha"} <= set(hints), err
    pin = {k: tuple(json.loads(v).values()) for k, v in hints.items()}
    _write(repo, L.FLOORS_REL, _floors(overall=pin["overall"], alpha=pin["maxim/alpha"]))
    _commit(repo)
    _cov(repo, missing={A: {9, 10, 15}, B: {6, 7, 8, 9, 10}})
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_the_local_pin_rise_hint_prints_the_cap_the_ci_rule_enforces(repo, capsys):
    """``pin_rises`` gates under ``--local``; with a measurement it prints min(measured, base), the CI rule's cap,
    not the base pin a lower measurement would refuse."""
    from types import SimpleNamespace

    base = L.parse_floors(_floors(alpha=(80.0, 5)))
    head = L.parse_floors(_floors(alpha=(85.0, 9)))
    assert L.pin_rises(base, head)[0].endswith("pin 5")  # no measurement: the base pin
    measured = {"maxim/alpha": SimpleNamespace(missing=3)}
    assert L.pin_rises(base, head, measured.get)[0].endswith("pin 3")


def _split_base(repo: Path) -> None:
    """Base on main: a third package `sp` (two files, 9/10 covered each) with its own floor."""
    _write(repo, "src/maxim/sp/c.py", _lines(10, "c"))
    _write(repo, "src/maxim/sp/d.py", _lines(10, "d"))
    _write(repo, L.FLOORS_REL, _floors(overall=(77.5, 9), **{"maxim/sp": (90.0, 2)}))
    _commit(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    (repo / "src/maxim/sa").mkdir()
    (repo / "src/maxim/sb").mkdir()
    _git(repo, "mv", "src/maxim/sp/c.py", "src/maxim/sa/c.py")
    _git(repo, "mv", "src/maxim/sp/d.py", "src/maxim/sb/d.py")


def test_a_clean_split_passes(repo, capsys):
    _split_base(repo)
    _write(repo, L.FLOORS_REL, _floors(overall=(77.5, 9), **{"maxim/sa": (90.0, 1), "maxim/sb": (90.0, 1)}))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, "src/maxim/sa/c.py": {10}, "src/maxim/sb/d.py": {10}})
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_split_may_not_lose_the_sum_of_its_missing(repo, capsys):
    """Delta round 4 (split1): each piece alone is under the carried pin (2), together they are 3."""
    _split_base(repo)
    _write(repo, L.FLOORS_REL, _floors(overall=(77.5, 9), **{"maxim/sa": (90.0, 1), "maxim/sb": (80.0, 2)}))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, "src/maxim/sa/c.py": {10}, "src/maxim/sb/d.py": {9, 10}})
    rc, _, err = _run(capsys)
    assert rc == 1
    assert (
        "maxim/sa + maxim/sb: received maxim/sp; together they measure missing 3 > the carried pinned missing 2 "
        "— cover 1 statement(s)" in err
    )


def test_a_blocked_lowering_names_the_statements_to_cover(repo, capsys):
    _cov(repo, missing={A: {7, 8, 9, 10}, B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(55.0, 7), alpha=(60.0, 2)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "measured missing 4 > the base's pinned missing 2 — cover 2 statement(s) first" in err


def _gamma_1000(repo: Path) -> None:
    """A 1000-statement package at 99.0% (k = K_MIN = 5 statements = 0.5 pt), committed on main as the base."""
    _write(repo, "src/maxim/gamma/g.py", _lines(1000, "g"))
    _write(repo, L.FLOORS_REL, _floors(overall=(98.3, 17), **{"maxim/gamma": (99.0, 10)}))
    _commit(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")


def _gamma_cov(repo: Path, covered: int) -> None:
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, "src/maxim/gamma/g.py": set(range(covered + 1, 1001))})


def test_a_drop_of_k_statements_below_a_floor_passes(repo, capsys):
    """Owner decision 2026-10-04 ("tolerance below floor"): CI noise of a few statements is not a failure."""
    _gamma_1000(repo)
    _gamma_cov(repo, 985)  # 98.5% under a 99.0 floor: exactly k = 5 statements
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_drop_of_k_plus_1_statements_below_a_floor_FAILS(repo, capsys):
    _gamma_1000(repo)
    _gamma_cov(repo, 984)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/gamma: 98.40% is below its floor 99.0% by more than the 5-statement" in err


def test_the_below_floor_tolerance_does_not_accumulate_across_PRs(repo, capsys):
    """Each PR drops statements and changes no pin: link 1 uses up k = 5, link 2's one more is k + 1 under."""
    _gamma_1000(repo)
    _gamma_cov(repo, 985)
    _write(repo, "src/maxim/alpha/note.py", "")  # a no-op change so the PR has a commit
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 0, err
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _gamma_cov(repo, 984)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/gamma: 98.40% is below its floor 99.0%" in err


def test_a_small_scope_gets_the_K_MIN_tolerance(repo, capsys):
    """Owner decision 2026-10-05: CI noise is absolute (~±2 statements; maxim/retrieval, 296 statements, measured
    59 then 61 missing). ceil(296 × 0.001) = 1 would fail this 5-statement drop; K_MIN = 5 admits it."""
    small = "src/maxim/gamma/r.py"
    _write(repo, small, _lines(296, "r"))
    _write(repo, L.FLOORS_REL, _floors(overall=(78.4, 68), **{"maxim/gamma": (79.3, 61)}))
    _commit(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, small: set(range(231, 297))})  # 230 covered: 77.70%
    rc, _, err = _run(capsys)
    assert rc == 0, err
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, small: set(range(229, 297))})  # 228: beyond k
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/gamma: 77.03% is below its floor 79.3% by more than the 5-statement" in err


# ── --local: the pre-push check scripts/suite_at_commit.sh runs ─────────────────────────────────────


def _run_local(capsys) -> tuple[int, str, str]:
    rc = L.main(["--local"])
    c = capsys.readouterr()
    return rc, c.out, c.err


def test_local_mode_only_notes_a_floor_of_a_package_the_diff_does_not_touch(repo, capsys):
    """Floors are measured in CI's environment, so a local run gates no floor (M60a); it prints the finding instead.
    The default (CI) run still fails on it."""
    _write(repo, B, _lines(10, "b") + "zz = 0  # pragma: no cover\n# touched\n")  # touches beta: a comment, no code
    _commit(repo)
    _cov(repo, missing={A: set(range(2, 11)), B: {6, 7, 8, 9, 10}})
    rc_ci, _, _ = _run(capsys)
    rc, out, err = _run_local(capsys)
    assert rc_ci == 1
    assert rc == 0, err
    assert "maxim/alpha: 10.00% is below its floor" in out and "not gated" in out


def test_local_mode_only_notes_a_touched_package_rise(repo, capsys):
    """The #1240 shape: the PR's own tests lift a touched package above its band. M60a (owner decision 2026-10-27): a
    macOS number cannot pin a Linux floor, so locally it is a note; CI still fails it and prints the pin."""
    _write(repo, A, _lines(10, "a") + "na = 1\n")
    _commit(repo)
    _cov(repo, missing={A: set(), B: {6, 7, 8, 9, 10}})
    rc_ci, _, err_ci = _run(capsys)
    rc, out, err = _run_local(capsys)
    assert rc_ci == 1 and "maxim/alpha: 100.00% is more than 1.0 pt above its floor" in err_ci
    assert rc == 0, err
    assert "not gated): maxim/alpha: 100.00% is more than 1.0 pt above its floor" in out


def test_local_mode_still_gates_diff_coverage(repo, capsys):
    """The #1232 shape: changed lines no test executes."""
    _write(repo, A, _lines(10, "a") + "n0 = 0\nn1 = 1\nn2 = 2\n")
    _commit(repo)
    _cov(repo, missing={A: {9, 10, 12, 13}, B: {6, 7, 8, 9, 10}})
    _write(repo, L.FLOORS_REL, _floors(overall=(56.5, 9), alpha=(69.2, 4)))
    _commit(repo)
    rc, _, err = _run_local(capsys)
    assert rc == 1 and "diff coverage 33.3% is below 80%" in err


def test_local_mode_is_refused_inside_github_actions(repo, capsys, monkeypatch):
    """A belt beside ci_step_rules (which already pins the CI step to the bare command): --local can never soften
    the CI gate."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    rc, _, err = _run_local(capsys)
    assert rc == 2 and "--local" in err


def test_the_suite_script_measures_with_cis_own_command():
    """M47: scripts/suite_at_commit.sh is the pre-push check only if it measures exactly as CI does and then runs the
    local ratchet; a drifted copy of the command would measure something CI does not."""
    text = (Path(__file__).resolve().parents[2] / "scripts" / "suite_at_commit.sh").read_text(encoding="utf-8")
    flat = " ".join(text.replace("\\\n", " ").split())
    clean, measure, local = (
        flat.find(L.CI_CLEAN),
        flat.find(L.CI_COV_ARGS),
        flat.find("python scripts/lint_coverage.py --local"),
    )
    assert -1 < clean < measure < local, (clean, measure, local)  # clean, then measure, then the local ratchet
    assert '-m "not slow" --ignore=tests/integration/test_memory_hub.py' in flat


def test_local_mode_only_notes_a_touched_package_fall(repo, capsys):
    """Both directions are notes locally (M60a): a touched package that falls below its floor is printed, and CI, the
    only floor authority, fails it."""
    _write(repo, A, _lines(10, "a") + "# touched\n")
    _commit(repo)
    _cov(repo, missing={A: set(range(2, 11)), B: {6, 7, 8, 9, 10}})
    rc_ci, _, err_ci = _run(capsys)
    rc, out, err = _run_local(capsys)
    assert rc_ci == 1 and "maxim/alpha: 10.00% is below its floor" in err_ci
    assert rc == 0, err
    assert "not gated): maxim/alpha: 10.00% is below its floor" in out


def test_local_mode_only_notes_a_floor_pin_edit(repo, capsys):
    """A PR's floor edits are CI measurements; locally they are printed, not verified."""
    _cov(repo, missing={A: set(), B: {6, 7, 8, 9, 10}})
    # lowers alpha 80 -> 70 below the measured 100% (CI refuses: a lowered floor goes exactly to the measurement);
    # its missing pin 0 is under its base 2, so no measurement-free rule fires
    _write(repo, L.FLOORS_REL, _floors(overall=(75.0, 5), alpha=(70.0, 0)))
    _commit(repo)
    rc, out, err = _run_local(capsys)
    rc_ci, _, _ = _run(capsys)
    assert rc_ci == 1
    assert rc == 0, err
    assert "floor pin rule, verified in CI, not gated" in out


@pytest.mark.parametrize("scope", ["package", "file", "overall"])
def test_local_mode_gates_a_missing_pin_that_rises(repo, capsys, scope):
    """The #1268 shape: new code adds uncovered statements, and the PR raises the package's ``missing`` pin to match
    (console 173 -> 207). "A pinned missing never rises" compares the two floor files, no measurement, so the local run
    gates it exactly as CI does."""
    _cov(repo, missing={A: {8, 9, 10}, B: {6, 7, 8, 9, 10}})  # alpha measured missing 3, one above its base pin 2
    if scope == "package":
        edit = _floors(alpha=(70.0, 3))
    elif scope == "file":
        _write(repo, L.FLOORS_REL, _floors(files={A: (70.0, 2)}))
        _commit(repo)
        _git(repo, "checkout", "-q", "main")
        _git(repo, "merge", "-q", "--ff-only", "feature")
        _git(repo, "checkout", "-q", "feature")
        edit = _floors(files={A: (70.0, 3)})
    else:
        edit = _floors(overall=(65.0, 8))
    _write(repo, L.FLOORS_REL, edit)
    _commit(repo)
    rc_ci, _, err_ci = _run(capsys)
    rc, out, err = _run_local(capsys)
    assert rc_ci == 1 and "a pinned missing never rises" in err_ci
    assert rc == 1 and "a pinned missing never rises" in err, err
    key = {"package": "maxim/alpha", "file": A, "overall": "overall"}[scope]
    assert f"not gated): {key}:" not in out, out  # gated once: not also printed as a contradicting note


def test_pin_rises_fires_only_on_a_same_key_pin_above_its_base():
    """The guards that keep --local from failing what CI accepts (#1270 review): a NEW key, a rename carried to a new
    key, and a null base pin are never a "rise" (CI judges them against measurements); a same-key rise always is."""
    base = L.parse_floors(_floors(alpha=(None, None), files={A: (80.0, 2)}))
    head = L.parse_floors(
        _floors(alpha=(80.0, 9), beta="drop", files={A: (80.0, 2), B: (50.0, 9)}, **{"maxim/gamma": (50.0, 9)})
    )
    assert L.pin_rises(base, head) == []  # alpha's base pin null; gamma new (beta's rename target); file B new
    rose = L.parse_floors(_floors(overall=(65.0, 8), beta=(50.0, 6), files={A: (80.0, 3)}))
    assert [m.split(":", 1)[0] for m in L.pin_rises(base, rose)] == ["overall", "maxim/beta", A]


def test_floor_head_rules_only_skips_an_untouched_package_without_a_floor(repo):
    """The second ``only`` filter: a measured package with no floor, outside the touched set, is not a failure."""
    fl = L.parse_floors(_floors())
    overall = L.ScopeCov(statements=20, covered=13)
    pkgs = {
        "maxim/alpha": L.ScopeCov(statements=10, covered=8),
        "maxim/beta": L.ScopeCov(statements=10, covered=5),
        "maxim/gamma": L.ScopeCov(statements=9, covered=9),
    }
    assert any("maxim/gamma" in m for m in L.floor_head_rules(repo, fl, overall, pkgs, files={}))
    assert not any(
        "maxim/gamma" in m for m in L.floor_head_rules(repo, fl, overall, pkgs, only={"maxim/alpha"}, files={})
    )


def test_the_suite_scripts_version_check_runs():
    """The pre-suite coverage-version check is an inline heredoc; run it exactly as the script holds it (a first cut
    loaded lint_coverage unregistered, so its dataclasses failed and every default run exited before pytest)."""
    import subprocess
    import sys as _sys

    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "suite_at_commit.sh").read_text(encoding="utf-8")
    body = text.split("<<'PY'", 1)[1].split("\n", 1)[1].split("\nPY\n", 1)[0]  # from the line after the opener
    out = subprocess.run([_sys.executable, "-c", body], cwd=root, capture_output=True, text=True)
    from importlib.metadata import version  # not `import coverage`: the gate bans the coverage API in tests (S4)

    if version("coverage") == L.COVERAGE_VERSION:
        assert out.returncode == 0, out.stderr
    else:
        assert out.returncode != 0 and L.COVERAGE_VERSION in out.stderr


def test_the_suite_script_is_parsed_whole_before_it_runs():
    """Bash reads a script as it runs: an edit during a run made the tail parse at the wrong offset, skip the
    ratchet and still exit 0. The body is one function, invoked (with its exit) on the file's last line."""
    lines = (Path(__file__).resolve().parents[2] / "scripts" / "suite_at_commit.sh").read_text().rstrip().splitlines()
    assert lines[-1] == 'main "$@"; exit "$?"'
    assert "main() {" in lines and lines[-2] == "}"


# ── per-file floors (format v2, owner decisions 2026-10-13: SF-1, SF-2, SF-3) ──────────────────────────────

G = "src/maxim/gamma/g.py"  # a 100-statement floored file: k = K_MIN = 5 statements = 5.0 pt
G2 = "src/maxim/gamma/g2.py"


def _to_base(repo: Path) -> None:
    """Make the current feature commit the base (main), as if it had merged."""
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")


def _g_cov(repo: Path, g_missing: set[int], **more: set[int]) -> None:
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, G: g_missing, **more})


def _g_floors(g=(90.0, 10), gamma=(90.0, 10), overall=(85.8, 17), files=None, **kw) -> str:
    return _floors(overall=overall, files={G: g} if files is None else files, **{"maxim/gamma": gamma}, **kw)


def _g_base(repo: Path, **kw) -> None:
    """Base on main: G (100 statements, 90 covered) with a file floor (90.0, 10) and its package floor."""
    _write(repo, G, _lines(100, "g"))
    _write(repo, L.FLOORS_REL, _g_floors(**kw))
    _commit(repo)
    _to_base(repo)
    _g_cov(repo, set(range(91, 101)))


def test_a_file_floor_within_k_passes_and_prints_its_line(repo, capsys):
    _g_base(repo)
    _g_cov(repo, set(range(86, 101)))  # 85 covered: exactly k = 5 under the 90.0 floor
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert f"  {G}: 85.00% of 100 statements, missing 15; branches 100.00% of 0 floor 90.0% (missing 10)" in out


def test_a_file_floor_k_plus_1_under_FAILS(repo, capsys):
    _g_base(repo, gamma=(85.0, 15), overall=(81.6, 22))  # the package has headroom: only the file fails
    _g_cov(repo, set(range(85, 101)))  # 84 covered
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{G}: 84.00% is below its floor 90.0% by more than the 5-statement noise tolerance" in err
    assert err.count("  - ") == 1, err


def test_a_file_floor_more_than_the_band_under_the_measurement_FAILS(repo, capsys):
    _g_base(repo, gamma=(91.0, 9), overall=(86.3, 16))
    _g_cov(repo, set(range(93, 101)))  # 92 covered: more than 1.0 pt over the 90.0 floor
    rc, _, err = _run(capsys)
    assert (
        rc == 1
        and f'{G}: 92.00% is more than 1.0 pt above its floor 90.0% (`python scripts/lint_coverage.py --pins-from-run <run-id>` writes it) — raise it to {{"percent": 92.0'
        in err
    )


def test_only_the_package_floor_fails_while_the_file_floor_holds(repo, capsys):
    """The two scopes are enforced independently: unfloored g2.py loses coverage, g.py holds."""
    _write(repo, G2, _lines(100, "q"))
    _g_base(repo, gamma=(90.0, 20), overall=(86.8, 27))
    _g_cov(repo, set(range(91, 101)), **{G2: set(range(81, 101))})  # g2: 80 covered; package 170/200 = 85%
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/gamma: 85.00% is below its floor 90.0%" in err
    assert G not in err, err


def test_only_the_file_floor_fails_while_the_package_floor_holds(repo, capsys):
    _write(repo, G2, _lines(100, "q"))
    _g_base(repo, gamma=(89.1, 20), overall=(86.8, 27))
    _g_cov(repo, set(range(85, 101)), **{G2: set(range(91, 101))})  # g: 84 covered; package 174/200 = 87%
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{G}: 84.00% is below its floor 90.0%" in err
    assert "maxim/gamma:" not in err and "overall:" not in err, err


def test_a_null_file_floor_fails_closed_and_prints_the_pin(repo, capsys):
    _g_base(repo, g=None)
    rc, out, err = _run(capsys)
    assert (
        rc == 1 and f'floor missing: {G} measured 90.00% (missing 10) — pin {{"percent": 90.0, "missing": 10}}' in err
    )
    assert _printed_floors(out)["files"] == {G: {"percent": 90.0, "missing": 10}}


def test_an_orphan_file_floor_FAILS(repo, capsys):
    _write(repo, L.FLOORS_REL, _floors(files={"src/maxim/alpha/ghost.py": (50.0, 1)}))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "src/maxim/alpha/ghost.py: floored file not found at HEAD (case-exact) — orphan" in err


def test_a_case_variant_file_key_FAILS(repo, capsys):
    """APFS answers is_file() True for `A.py`; the case-exact listing does not (NIT-1)."""
    _write(repo, L.FLOORS_REL, _floors(files={"src/maxim/alpha/A.py": (80.0, 2)}))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "src/maxim/alpha/A.py: floored file not found at HEAD (case-exact)" in err


@pytest.mark.parametrize("link,target", [("src/maxim/alpha/link.py", "a.py"), ("src/maxim/linked", "alpha")])
def test_a_symlink_file_key_FAILS(repo, capsys, link, target):
    os.symlink(target, repo / link)
    key = link if link.endswith(".py") else f"{link}/a.py"
    _write(repo, L.FLOORS_REL, _floors(files={key: (80.0, 2)}))
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{key}: a file floor's key may not be (or pass through) a symlink" in err


def test_an_excluded_floored_file_FAILS(repo, capsys):
    _write(repo, L.FLOORS_REL, _floors(files={HW: (50.0, 1)}))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{HW}: floored file has no measured statements" in err


@pytest.mark.parametrize(
    "key",
    [
        "./src/maxim/alpha/a.py",
        "/src/maxim/alpha/a.py",
        "src/maxim//alpha/a.py",
        "src/maxim/beta/../alpha/a.py",
        "src\\maxim\\alpha\\a.py",
        "src/maxim/alpha/a.pyc",
        "scripts/x.py",
    ],
)
def test_a_malformed_file_key_is_a_format_error(repo, capsys, key):
    _write(repo, L.FLOORS_REL, _floors(files={key: (80.0, 2)}))
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{L.FLOORS_REL}: file key {key!r} must be a normalized repo-relative path" in err


@pytest.mark.parametrize("rel", [L.FLOORS_REL, L.EXCLUSIONS_REL])
def test_a_duplicate_json_key_is_a_format_error(repo, capsys, rel):
    """NIT-2: json.loads keeps the LAST value; a reviewer reads the first."""
    text = (repo / rel).read_text()
    dup = (
        text.replace('"band": 1.0,', '"band": 1.0,\n "band": 1.0,')
        if rel == L.FLOORS_REL
        else text.replace('"ledger":', '"exclusions": [],\n "ledger":')
    )
    assert dup != text
    _write(repo, rel, dup)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{rel}: duplicate JSON key" in err


def test_a_duplicate_nested_key_is_a_format_error():
    text = _floors().replace('"maxim/alpha": {', '"maxim/alpha": {"percent": 1.0,', 1)
    with pytest.raises(L.FileFormatError, match="duplicate JSON key 'percent'"):
        L.parse_floors(text)


# PR rules (a file floor is verified data, the same rules as packages)


def test_a_new_file_floor_pinned_from_the_measurement_passes(repo, capsys):
    _write(repo, L.FLOORS_REL, _floors(files={A: (80.0, 2)}))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 0, err


@pytest.mark.parametrize(
    "pin,needle",
    [
        ((84.9, 15), "new floor 84.9% is below the measured 90.00% less 5 statement(s) of run-to-run noise (85.0%)"),
        ((90.0, 11), "floor pins missing 11 above the measured 10"),
    ],
)
def test_a_new_file_floor_is_verified_against_the_measurement(repo, capsys, pin, needle):
    _write(repo, G, _lines(100, "g"))
    _write(repo, L.FLOORS_REL, _g_floors(g=pin))
    _commit(repo)
    _g_cov(repo, set(range(91, 101)))
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{G}: {needle}" in err


def test_a_files_only_PR_cannot_lower_a_file_floor(repo, capsys):
    _g_base(repo)
    _write(repo, L.FLOORS_REL, _g_floors(g=(89.0, 10)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{G}: floor lowered to 89.0%, below the measurement 90.00%" in err


def test_a_file_floor_cannot_raise_its_missing_pin(repo, capsys):
    _g_base(repo)
    _write(repo, L.FLOORS_REL, _g_floors(g=(90.0, 11)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{G}: floor pins missing 11 above min(measured 10 + 5 of noise, base 10)" in err


def test_lowering_a_file_floor_after_adding_uncovered_code_FAILS(repo, capsys):
    _g_base(repo)
    _write(repo, G, _lines(110, "g"))
    _g_cov(repo, set(range(91, 111)))  # 90/110 = 81.8%, missing 20
    _write(repo, L.FLOORS_REL, _g_floors(g=(81.8, 20), gamma=(81.8, 20), overall=(79.6, 27)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert (
        rc == 1 and f"{G}: floor lowered 90.0% -> 81.8% but measured missing 20 > the base's pinned missing 10" in err
    )


def test_lowering_a_file_floor_after_moving_covered_code_out_passes_at_round_down(repo, capsys):
    """Covered statements leave (deleted here; a move to an unfloored EXISTING file is the documented residual)."""
    _g_base(repo)
    _write(repo, G, _lines(80, "g"))
    _g_cov(repo, set(range(71, 81)))  # 70/80 = 87.5%, missing still 10
    _write(repo, L.FLOORS_REL, _g_floors(g=(87.4, 10), gamma=(87.5, 10), overall=(83.0, 17)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{G}: floor lowered to 87.4%, below the measurement 87.50%" in err
    _write(repo, L.FLOORS_REL, _g_floors(g=(87.5, 10), gamma=(87.5, 10), overall=(83.0, 17)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_file_floor_removed_while_its_file_exists_FAILS(repo, capsys):
    _g_base(repo)
    _write(repo, L.FLOORS_REL, _g_floors(files={}))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{G}: floor removed while its file still exists" in err


def test_a_file_floor_set_back_to_null_FAILS(repo, capsys):
    _g_base(repo)
    _write(repo, L.FLOORS_REL, _g_floors(g=None))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{G}: floor set back to null" in err


def test_a_file_rename_carries_its_floor(repo, capsys):
    _g_base(repo)
    moved = "src/maxim/gamma/moved.py"
    _git(repo, "mv", G, moved)
    _write(repo, L.FLOORS_REL, _g_floors(files={moved: (90.0, 10)}))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, moved: set(range(91, 101))})
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_file_rename_cannot_lower_the_carried_floor(repo, capsys):
    _g_base(repo)
    moved = "src/maxim/gamma/moved.py"
    _git(repo, "mv", G, moved)
    _write(repo, L.FLOORS_REL, _g_floors(files={moved: (89.0, 10)}))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, moved: set(range(91, 101))})
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{moved} (carrying {G}): floor lowered to 89.0%, below the measurement 90.00%" in err


def test_a_file_rename_target_without_a_floor_FAILS(repo, capsys):
    _g_base(repo)
    moved = "src/maxim/gamma/moved.py"
    _git(repo, "mv", G, moved)
    _write(repo, L.FLOORS_REL, _g_floors(files={}))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, moved: set(range(91, 101))})
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{moved}: received {G}'s renamed files; the floor carries over" in err


def test_a_case_only_rename_carries_the_floor_case_exactly(repo, capsys):
    """NIT-1 on the removal side: on APFS `g.py` still "is_file()" after `git mv g.py G.py`; the listing says gone."""
    _g_base(repo)
    upper = "src/maxim/gamma/G.py"
    _git(repo, "mv", G, upper)
    _write(repo, L.FLOORS_REL, _g_floors(files={upper: (90.0, 10)}))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, upper: set(range(91, 101))})
    rc, _, err = _run(capsys)
    assert rc == 0, err


# SF-1: a floored file does not vanish


def _delete_g(repo: Path, retired=None) -> None:
    _git(repo, "rm", "-q", G)
    _write(repo, L.FLOORS_REL, _g_floors(files={}, gamma="drop", overall=(65.0, 7), retired=retired))
    _commit(repo)
    _cov(repo)


def test_a_floored_file_gone_without_a_rename_FAILS_closed(repo, capsys):
    _g_base(repo)
    _delete_g(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{G}: floored file is gone at HEAD with no detected rename" in err


def test_a_refd_retired_files_entry_retires_a_gone_floor(repo, capsys):
    _g_base(repo)
    _delete_g(repo, retired=[{"path": G, "ref": "#77", "reason": "dead code"}])
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert f"RETIRED FILE FLOOR {G} (ref #77): dead code" in out


def test_a_retired_files_entry_needs_a_ref(repo, capsys):
    _g_base(repo)
    _delete_g(repo, retired=[{"path": G, "ref": None, "reason": "dead code"}])
    rc, _, err = _run(capsys)
    assert rc == 1 and "retired_files entry needs a ref" in err


def test_retired_files_is_append_only(repo, capsys):
    _g_base(repo)
    _delete_g(repo, retired=[{"path": G, "ref": "#77", "reason": "dead code"}])
    _to_base(repo)
    _write(repo, L.FLOORS_REL, _g_floors(files={}, gamma="drop", overall=(65.0, 7), retired=[]))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "retired_files is append-only" in err


@pytest.mark.parametrize("path", [G, A])
def test_a_retired_entry_must_name_a_floored_file_that_is_gone(repo, capsys, path):
    _g_base(repo)
    _write(repo, L.FLOORS_REL, _g_floors(files={}, retired=[{"path": path, "ref": "#7", "reason": "r"}]))
    _commit(repo)
    rc, _, err = _run(capsys)
    needle = "the file or its floor is still here at HEAD" if path == G else "no file floor at the base to retire"
    assert rc == 1 and f"retired_files {path}: {needle}" in err


# SF-2: the split rule (S = 25)


def _split(repo: Path, kept: int, g_missing: set[int], successors: dict[str, tuple[int, set[int]]], files) -> None:
    """Shrink G to its first ``kept`` lines and ADD ``successors`` ({path: (lines, missing)})."""
    _write(repo, G, _lines(kept, "g"))
    for rel, (n, _) in successors.items():
        _write(repo, rel, _lines(n, "s" + Path(rel).stem))
    _write(repo, L.FLOORS_REL, _g_floors(files=files))
    _commit(repo)
    _g_cov(repo, g_missing, **{rel: miss for rel, (_, miss) in successors.items()})


CORE = "src/maxim/gamma/core.py"
SMALL = "src/maxim/gamma/small.py"


def test_the_shim_trick_FAILS_a_big_successor_needs_its_own_floor(repo, capsys):
    """A15: move everything to a new module, leave a 5-line re-export shim: the floor would stay on the shim."""
    _g_base(repo)
    _split(repo, 5, set(), {CORE: (95, set(range(86, 96)))}, files={G: (100.0, 0)})
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{CORE}: 95 statements (added) while this diff removed 95 of floored {G}'s statements" in err
    assert f'needs its own file floor (split rule): add "{CORE}": {{"percent": 89.4, "missing": 10}}' in err


def test_a_clean_split_with_a_floored_successor_passes(repo, capsys):
    _g_base(repo)
    _split(repo, 5, set(), {CORE: (95, set(range(86, 96)))}, files={G: (100.0, 0), CORE: (89.4, 10)})
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_file_split_may_not_lose_the_sum_of_its_missing(repo, capsys):
    """A17: each piece pinned honestly, but together they carry more uncovered statements than G's base pin."""
    _g_base(repo, gamma=(88.0, 12), overall=(84.1, 19))
    _split(repo, 5, set(), {CORE: (95, set(range(85, 96)))}, files={G: (100.0, 0), CORE: (88.4, 11)})
    rc, _, err = _run(capsys)
    assert rc == 1 and (
        f"{CORE} + {G}: received {G}; together they measure missing 11 > the carried pinned missing 10 — cover 1 "
        "statement(s) (split rule" in err
    )


def test_a_successor_under_25_statements_only_joins_the_sum(repo, capsys):
    _g_base(repo)
    _split(repo, 70, set(range(61, 71)), {SMALL: (10, set())}, files=None)  # g: 60/70, small: 10/10
    rc, _, err = _run(capsys)
    assert rc == 0, err
    _g_cov(repo, set(range(61, 71)), **{SMALL: {10}})  # one uncovered small statement
    rc, _, err = _run(capsys)
    assert (
        rc == 1
        and f"{G} + {SMALL}: received {G}; together they measure missing 11 > the carried pinned missing 10" in err
    )


def test_an_in_place_rewrite_with_no_successor_is_held_by_the_floor_not_the_sum(repo, capsys):
    """30 statements rewritten in place fire the trigger, but no file was added or grew: the no-tolerance sum is
    skipped (owner decision 2026-10-13), so missing may rise within k; the HEAD floor rule still holds G."""
    _g_base(repo)
    _write(repo, G, _lines(10, "g") + _lines(30, "r") + _lines(100, "g")[len(_lines(40, "g")) :])
    _commit(repo)
    _g_cov(repo, set(range(91, 101)) | {38, 39, 40})  # missing 13 > G's pin 10, within k = 5
    rc, _, err = _run(capsys)
    assert rc == 0, err
    _g_cov(repo, set(range(91, 101)) | set(range(35, 41)))  # missing 16: beyond k under the 90.0 floor
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{G}: 84.00% is below its floor 90.0% by more than the 5-statement noise tolerance" in err
    assert "split rule" not in err, err


@pytest.mark.parametrize("kept", [76, 75])
def test_the_split_rule_fires_at_a_shrink_of_25_statements(repo, capsys, kept):
    """S = 25: 24 removed statements have no successors; at 25 the unfloored 30-statement added file needs a floor."""
    _g_base(repo)
    _split(repo, kept, set(range(kept - 9, kept + 1)), {CORE: (30, set())}, files=None)
    rc, _, err = _run(capsys)
    if kept == 76:
        assert rc == 0, err
    else:
        assert (
            rc == 1 and f"{CORE}: 30 statements (added) while this diff removed 25 of floored {G}'s statements" in err
        )


def test_an_excluded_added_file_is_not_a_successor(repo, capsys):
    """The TTY route: a whole-file exclusion with a ref'd `excluded_statements` ledger entry (the existing model)."""
    _g_base(repo)
    tty = "src/maxim/gamma/tty_input.py"
    entry = {"path": tty, "reason": "termios read loops only", "covered_by": None, "ref": "#5"}
    _write(repo, "pyproject.toml", _pyproject(omit=[HW, tty]))
    _write(
        repo,
        L.EXCLUSIONS_REL,
        _exclusions([HW_ENTRY, entry], BASE_LEDGER + [_ledger("excluded_statements", tty, 30, "#5")]),
    )
    _write(repo, tty, _lines(30, "t"))
    _split(repo, 70, set(range(61, 71)), {}, files=None)
    _cov(repo, {tty: None}, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, G: set(range(61, 71))})
    rc, out, err = _run(capsys)
    assert rc == 0, err
    assert f"NEW EXCLUSION {tty} (ref #5)" in out


def test_a_retired_floor_still_hands_its_code_to_added_files(repo, capsys):
    """Gone counts as shrunk to 0: retiring G while adding its code elsewhere does not drop the floor."""
    _g_base(repo)
    _git(repo, "rm", "-q", G)
    _write(repo, CORE, _lines(100, "c"))
    _write(repo, L.FLOORS_REL, _g_floors(files={}, retired=[{"path": G, "ref": "#7", "reason": "dead"}]))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, CORE: set(range(91, 101))})
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{CORE}: 100 statements (added) while this diff removed 100 of floored {G}'s statements" in err


def test_a_pre_created_successor_FAILS_an_existing_file_that_grew_is_a_successor(repo, capsys):
    """Bypass 1 (review probe P1): PR1 adds a one-statement core.py; PR2 moves 95 statements into it, leaving a
    5-line shim. core.py is EXISTING, so before the fold only ADDED files were successors and this passed."""
    _g_base(repo)
    _write(repo, CORE, "c0 = 0\n")
    _commit(repo)
    _to_base(repo)
    _write(repo, G, _lines(5, "g"))
    _write(repo, CORE, "c0 = 0\n" + _lines(95, "s"))
    _write(repo, L.FLOORS_REL, _g_floors(files={G: (100.0, 0)}))
    _commit(repo)
    _g_cov(repo, set(), **{CORE: set(range(87, 97))})
    rc, _, err = _run(capsys)
    assert rc == 1 and (
        f"{CORE}: 96 statements (grew by 95 statements) while this diff removed 95 of floored {G}'s statements — a "
        f'successor of >= 25 statements needs its own file floor (split rule): add "{CORE}": {{"percent": 89.5, '
        '"missing": 10}' in err
    )
    _write(repo, L.FLOORS_REL, _g_floors(files={G: (100.0, 0), CORE: (89.5, 10)}))  # NEW at the base: NEW rules
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 0, err


@pytest.mark.parametrize("removed", [24, 25])
def test_padding_does_not_hide_removed_statements(repo, capsys, removed):
    """Bypass 2 (review probe P3): statements moved out and as many trivial ones added back, so the NET count is
    unchanged. The trigger counts base statements REMOVED: 24 does not fire, 25 does."""
    _g_base(repo)
    _write(repo, G, _lines(10, "g") + _lines(removed, "pad") + _lines(100, "g")[len(_lines(10 + removed, "g")) :])
    _write(repo, CORE, _lines(30, "c"))
    _write(repo, L.FLOORS_REL, _g_floors(gamma=(92.3, 10), overall=(88.6, 17)))  # core.py lifts both scopes
    _commit(repo)
    _g_cov(repo, set(range(91, 101)), **{CORE: set()})
    assert L.statement_count((repo / G).read_text()) == 100
    rc, _, err = _run(capsys)
    if removed == 24:
        assert rc == 0, err
    else:
        assert (
            rc == 1 and f"{CORE}: 30 statements (added) while this diff removed 25 of floored {G}'s statements" in err
        )


def test_the_review_probes_padding_shape_FAILS(repo, capsys):
    """P3 verbatim: 60 statements leave for a NEW file and 60 trivial ones are added back (net change 0)."""
    _g_base(repo)
    _write(repo, G, _lines(40, "g") + _lines(60, "pad"))
    _write(repo, CORE, _lines(60, "c"))
    _commit(repo)
    _g_cov(repo, set(range(91, 101)), **{CORE: set(range(50, 61))})
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{CORE}: 60 statements (added) while this diff removed 60 of floored {G}'s statements" in err


def test_a_symlink_left_at_the_floored_path_counts_as_gone(repo, capsys):
    """N4 (review probe P4): G retired and replaced by a symlink to its new home; the symlink read 100 statements, so
    nothing looked removed. A symlinked path is gone: all 100 base statements are removed."""
    _g_base(repo)
    _write(repo, CORE, _lines(100, "g"))
    _git(repo, "rm", "-q", G)
    os.symlink("core.py", repo / G)
    _write(repo, L.FLOORS_REL, _g_floors(files={}, retired=[{"path": G, "ref": "#9", "reason": "r"}]))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, CORE: set(range(91, 101))})
    d = json.loads((repo / "coverage.json").read_text())
    d["files"].pop(G, None)  # coverage canonicalizes: the symlink is never a coverage.json key
    (repo / "coverage.json").write_text(json.dumps(d))
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{CORE}: 100 statements (added) while this diff removed 100 of floored {G}'s statements" in err


def test_split_rule_reads_a_symlinked_path_as_gone_whatever_git_reports(tmp_path):
    """N4 on its own: git records a file -> symlink typechange as a full deletion, so the test above also passes via
    the hunks. Here git reports nothing for G (no hunks), and only N4 sees that its path is now a symlink."""
    _write(tmp_path, CORE, _lines(100, "g"))
    os.symlink("core.py", tmp_path / G)
    fd = L.FileDiff({}, {CORE}, lambda rel: _lines(100, "g") if rel == G else "")
    base_fl, fl = L.parse_floors(_g_floors()), L.parse_floors(_g_floors(files={}))
    out = L.split_rule(tmp_path, base_fl, fl, {}, fd, set(), set(L._src_files(tmp_path)))
    assert any(f"{CORE}: 100 statements (added) while this diff removed 100 of floored {G}'s" in o for o in out), out


def _gamma_with_core(repo: Path, core_floor=None) -> None:
    """Base: G (100 statements, 10 missing, floored) and an EXISTING core.py (40 statements, its last 10 uncovered),
    floored at the base only when ``core_floor`` is given."""
    _write(repo, CORE, _lines(40, "c"))
    _g_base(
        repo,
        gamma=(85.7, 20),
        overall=(83.1, 27),
        files={G: (90.0, 10), **({CORE: core_floor} if core_floor else {})},
    )
    _g_cov(repo, set(range(91, 101)), **{CORE: set(range(31, 41))})


def _move_30_into_core(repo: Path, files: dict, core_missing: set[int]) -> None:
    """Move G's first 30 (covered) statements to the end of core.py: core grows by 30."""
    _write(repo, G, _lines(100, "g")[len(_lines(30, "g")) :])
    _write(repo, CORE, _lines(40, "c") + _lines(30, "g"))
    _write(repo, L.FLOORS_REL, _g_floors(files=files, gamma=(85.7, 20), overall=(83.1, 27)))
    _commit(repo)
    _g_cov(repo, set(range(61, 71)), **{CORE: core_missing})


def test_an_existing_successor_already_floored_follows_the_changed_rules_and_brings_its_pin(repo, capsys):
    _gamma_with_core(repo, core_floor=(75.0, 10))
    _move_30_into_core(repo, {G: (85.7, 10), CORE: (85.7, 10)}, set(range(31, 41)))
    rc, _, err = _run(capsys)
    assert rc == 0, err  # G 10 + core 10 <= G's pin 10 + core's base pin 10
    _move_30_into_core(repo, {G: (85.7, 10), CORE: (85.7, 11)}, set(range(31, 42)))  # CHANGED: missing never rises
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{CORE}: floor pins missing 11 above min(measured 11 + 5 of noise, base 10)" in err
    _move_30_into_core(repo, {G: (85.7, 10), CORE: (84.2, 10)}, set(range(31, 42)))  # one moved statement uncovered
    rc, _, err = _run(capsys)
    assert rc == 1 and (
        f"{CORE} + {G}: received {G}; together they measure missing 21 > the carried pinned missing 20 — cover 1" in err
    )


def test_an_unfloored_existing_successor_is_charged_its_whole_missing(repo, capsys):
    """The strict charge: core.py's 10 OLD uncovered statements count against G's pin, because core.py brings no
    base pin. The documented remedy, moving into a NEW module, passes."""
    _gamma_with_core(repo)
    _move_30_into_core(repo, {G: (85.7, 10), CORE: (85.7, 10)}, set(range(31, 41)))
    rc, _, err = _run(capsys)
    assert rc == 1 and (
        f"{CORE} + {G}: received {G}; together they measure missing 20 > the carried pinned missing 10 — cover 10"
        in err
    )
    assert "move code into NEW modules" in err and err.count("  - ") == 1, err
    _write(repo, CORE, _lines(40, "c"))  # instead: the same 30 statements into a new module
    _write(repo, SMALL, _lines(30, "g"))
    _write(
        repo, L.FLOORS_REL, _g_floors(files={G: (85.7, 10), SMALL: (100.0, 0)}, gamma=(85.7, 20), overall=(83.1, 27))
    )
    _commit(repo)
    _g_cov(repo, set(range(61, 71)), **{CORE: set(range(31, 41)), SMALL: set()})
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_an_existing_file_that_grew_by_fewer_than_25_is_not_a_successor(repo, capsys):
    """The thin-slicing residual's other shape: 30 statements leave G, core.py grows by 24 (the rest deleted)."""
    _gamma_with_core(repo)
    _write(repo, G, _lines(100, "g")[len(_lines(30, "g")) :])
    _write(repo, CORE, _lines(40, "c") + _lines(24, "g"))
    _commit(repo)
    _g_cov(repo, set(range(61, 71)), **{CORE: set(range(31, 41))})
    rc, _, err = _run(capsys)
    assert rc == 0, err


_REMOVE_BASE = (
    "def f(x):\n"  # 1
    "    # a comment\n"  # 2
    "    y = call(\n"  # 3
    "        x,\n"  # 4
    "    )\n"  # 5
    "    try:\n"  # 6
    "        z = 1\n"  # 7
    "    except ValueError:\n"  # 8
    "        z = 2\n"  # 9
    "    return y\n"  # 10
)


@pytest.mark.parametrize(
    "deleted,head,removed",
    [
        ({2}, _REMOVE_BASE.replace("    # a comment\n", ""), 0),  # a comment is no statement
        ({4}, _REMOVE_BASE.replace("        x,\n", ""), 1),  # a continuation line maps to its statement
        ({8}, _REMOVE_BASE.replace("except ValueError", "except KeyError"), 1),  # a header maps to its block
        ({3, 4, 5}, _REMOVE_BASE.replace("    y = call(\n        x,\n    )\n", "    y = 0\n"), 1),
    ],
)
def test_removed_statements_uses_the_diff_coverage_mapping(deleted, head, removed):
    lo, n = min(deleted), len(deleted)
    new_n = len(head.splitlines()) - len(_REMOVE_BASE.splitlines()) + n
    assert L.removed_statements(_REMOVE_BASE, [(lo, n, lo, new_n)], head) == removed


def test_code_wrapped_in_a_string_is_removed_though_no_line_is_deleted():
    base = "a = 1\nb = 2\nc = 3\n"
    assert L.removed_statements(base, [(0, 0, 1, 1)], "z = 0\n" + base) == 0  # an insertion only shifts lines
    wrapped = 'x = """\n' + base + '"""\n'
    assert L.removed_statements(base, [(0, 0, 1, 1), (3, 0, 5, 1)], wrapped) == 3


def test_the_file_diff_runs_git_only_when_a_file_floor_exists_on_either_side(repo, capsys, monkeypatch):
    """No file floor at the base or HEAD: no git call (no new exit-2 path). The first PR that adds one runs it."""
    calls = []
    real = L.file_diff
    monkeypatch.setattr(L, "file_diff", lambda root, base: calls.append(base) or real(root, base))
    rc, _, err = _run(capsys)
    assert rc == 0 and calls == [], err
    _write(repo, L.FLOORS_REL, _floors(files={A: (80.0, 2)}))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 0 and len(calls) == 1, err


@pytest.mark.parametrize("first", ["q = 1\n", "-- 1\n"])
def test_a_deleted_line_that_looks_like_a_diff_header_does_not_hide_later_hunks(repo, capsys, first):
    """Delta review (2026-10-13): deleting a column-0 ``-- 1`` prints ``--- 1``; read as a file header it dropped
    every later hunk of the file, so the 60-statement move below read as one removed statement and passed."""
    ext = "src/maxim/delta/core.py"
    _write(repo, G, first + _lines(99, "g"))
    _write(repo, L.FLOORS_REL, _g_floors())
    _commit(repo)
    _to_base(repo)
    _write(repo, G, _lines(39, "g") + _lines(61, "pad"))
    _write(repo, "src/maxim/delta/__init__.py", "")
    _write(repo, ext, "".join(f"g{i} = {i}\n" for i in range(39, 99)))
    floors = _g_floors(g=(100.0, 0), gamma=(100.0, 0), overall=(90.5, 17), **{"maxim/delta": (83.3, 10)})
    _write(repo, L.FLOORS_REL, floors)
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, G: set(), ext: set(range(51, 61))})
    rc, _, err = _run(capsys)
    assert rc == 1 and f"{ext}: 60 statements (added) while this diff removed 61 of floored {G}'s statements" in err


def test_an_added_line_that_looks_like_a_diff_header_does_not_hide_later_hunks(repo, capsys):
    """The same shape in diff coverage (an existing rule, hardened): an added ``++ a0`` prints ``+++ a0``, which read
    as a header and dropped the file's later hunk — its two uncovered lines went uncounted."""
    _write(repo, A, "++ a0\n" + _lines(10, "a") + "z1 = 1\nz2 = 2\n")
    _commit(repo)
    assert L.changed_lines(repo, "main") == {A: {1, 12, 13}}
    _cov(repo, missing={A: {10, 11, 12, 13}, B: {6, 7, 8, 9, 10}})
    rc, _, err = _run(capsys)
    assert rc == 1 and "diff coverage 33.3% is below 80% (1/3)" in err


_ALPHA_TXT = "src/maxim/alpha/notes.txt"


@pytest.mark.parametrize(
    "rel,text,lines",
    [
        (A, _lines(10, "a") + "\x0c\nz = 1\n", {11, 12}),  # a form-feed line (legal Python whitespace)
        (A, _lines(10, "a") + 's = "a\u2028b"\n', {11}),  # U+2028 inside a string
        (_ALPHA_TXT, "y = 1\rz\n", None),  # a lone CR, in a non-.py file under src/maxim
        (A, _lines(10, "a") + "# caf\x85e\nz = 1\n", {11, 12}),  # \x85 in a comment
        (A, "".join(f"a{i} = {i}\r\n" for i in range(10)) + "z = 1\r\n", set(range(1, 12))),  # a CRLF file
    ],
    ids=["form-feed", "u2028", "lone-cr-txt", "x85", "crlf"],
)
def test_line_separators_inside_a_diff_line_do_not_break_the_diff_reader(repo, capsys, rel, text, lines):
    """Delta review (2026-10-13): splitlines() and text-mode git split one diff line at these characters, and the
    count-based reader then refused the diff (exit 2) on every PR. Real git, both readers, correct line numbers."""
    _write(repo, rel, text)
    _commit(repo)
    assert L.changed_lines(repo, "main") == ({A: lines} if lines is not None else {})
    hunks, modified = L._diff_hunks(repo, "main")
    if lines is None:  # an added file: no base path, so no hunks keyed by one
        assert hunks == {} and modified == set()
    else:
        assert modified == {A} and hunks[A]
    _cov(repo)
    rc, _, err = _run(capsys)
    assert rc != 2 and "could not answer" not in err, err


_DIFF_TWO_FILES = (
    "diff --git a/src/maxim/x.py b/src/maxim/x.py\n"
    "index 1..2 100644\n"
    "--- a/src/maxim/x.py\n"
    "+++ b/src/maxim/x.py\n"
    "@@ -1,2 +1,2 @@ def f():\n"
    "--- 1\n"  # a deleted ``-- 1``
    "-x = 1\n"
    "+++ y\n"  # an added ``++ y``
    "+diff --git a/z b/z\n"  # an added line that looks like the next file's header
    "@@ -5 +5 @@\n"  # no counts: one line each
    "-a\n"
    "\\ No newline at end of file\n"
    "+b\n"
    "\\ No newline at end of file\n"
    "diff --git a/src/maxim/gone.py b/src/maxim/gone.py\n"
    "deleted file mode 100644\n"
    "--- a/src/maxim/gone.py\n"
    "+++ /dev/null\n"
    "@@ -1,2 +0,0 @@\n"
    "-p = 1\n"
    "-q = 2\n"
    "diff --git a/src/maxim/new.py b/src/maxim/new.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/src/maxim/new.py\n"
    "@@ -0,0 +1 @@\n"
    "++++ r\n"
)


def test_the_diff_parser_consumes_hunk_bodies_by_their_counts():
    files = L.parse_unified_diff(_DIFF_TWO_FILES)
    assert [(f.old, f.new, f.hunks) for f in files] == [
        ("src/maxim/x.py", "src/maxim/x.py", [(1, 2, 1, 2), (5, 1, 5, 1)]),
        ("src/maxim/gone.py", None, [(1, 2, 0, 0)]),
        (None, "src/maxim/new.py", [(0, 0, 1, 1)]),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1,2 +1 @@\n-a\n",  # the body ends early
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n-b\n",  # more than the header says
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\nstray\n",
    ],
)
def test_the_diff_parser_refuses_output_it_cannot_account_for(text):
    with pytest.raises(L.GitUnavailable):
        L.parse_unified_diff(text)


def test_a_non_ascii_rename_carries_its_file_floor(repo, capsys):
    """core.quotePath=false on the file rules' rename map: quoted, the target read as a path that does not exist."""
    _g_base(repo)
    moved = "src/maxim/gamma/gé.py"
    _git(repo, "config", "core.quotePath", "true")  # git's default, pinned against a developer's global config
    _git(repo, "mv", G, moved)
    _write(repo, L.FLOORS_REL, _g_floors(files={moved: (90.0, 10)}))
    _commit(repo)
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}, moved: set(range(91, 101))})
    rc, _, err = _run(capsys)
    assert rc == 0, err


# format v1 -> v2


def test_a_v1_base_may_become_v2(repo, capsys):
    _write(repo, L.FLOORS_REL, _floors(version=1))
    _commit(repo)
    _to_base(repo)
    _write(repo, L.FLOORS_REL, _floors(files={A: (80.0, 2)}))  # every file floor is NEW against a v1 base
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 0, err


def test_a_v1_floors_file_at_HEAD_FAILS(repo, capsys):
    _write(repo, L.FLOORS_REL, _floors(version=1))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "floors_format_version must be 2, got 1" in err and "changed 2 -> 1" in err


def test_a_version_past_2_FAILS(repo, capsys):
    _write(repo, L.FLOORS_REL, _floors(version=3))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "floors_format_version must be 2, got 3" in err and "changed 2 -> 3" in err


def test_a_v2_file_without_files_is_a_format_error():
    with pytest.raises(L.FileFormatError, match="top-level keys must be exactly"):
        L.parse_floors(_floors(version=1).replace('"floors_format_version": 1', '"floors_format_version": 2'))


# --local and --merge-floors


def test_local_mode_only_notes_a_file_floor_even_when_its_package_is_touched(repo, capsys):
    _g_base(repo, gamma=(85.0, 15), overall=(81.6, 22))
    _g_cov(repo, set(range(85, 101)))  # the file is 6 statements under: beyond k
    _write(repo, "src/maxim/alpha/note.py", "")  # touches alpha only
    _commit(repo)
    rc, out, err = _run_local(capsys)
    assert rc == 0, err
    assert f"measured in CI's environment, not gated): {G}: 84.00% is below its floor" in out
    _write(repo, "src/maxim/gamma/note.py", "")  # now gamma is touched: still a note (M60a); CI fails it
    _commit(repo)
    rc, out, err = _run_local(capsys)
    assert rc == 0, err
    assert f"not gated): {G}: 84.00% is below its floor 90.0%" in out
    assert _run(capsys)[0] == 1


def _run_doc(**files) -> str:
    return json.dumps(
        {
            "overall": {"percent": 80.0, "missing": 100},
            "packages": {},
            "files": {k: {"percent": v[0], "missing": v[1]} for k, v in files.items()},
            "retired_files": [],
        }
    )


def _merge(tmp_path: Path, capsys, *docs: str, flags: tuple[str, ...] = ()) -> tuple[int, str, str]:
    paths = []
    for i, d in enumerate(docs):
        (tmp_path / f"r{i}.json").write_text(d)
        paths.append(str(tmp_path / f"r{i}.json"))
    rc = L.main(["--merge-floors", *flags, *paths])
    c = capsys.readouterr()
    return rc, c.out, c.err


@pytest.fixture
def committed(tmp_path: Path, monkeypatch):
    """The committed floors file --merge-floors reads (SF-A4): every scope pinned, no file floor, unless rewritten."""

    def write(**kw) -> None:
        _write(tmp_path, L.FLOORS_REL, _floors(**kw))

    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    write()
    return write


def test_merge_floors_merges_files(tmp_path, capsys, committed):
    rc, out, err = _merge(tmp_path, capsys, _run_doc(**{G: (90.0, 10)}), _run_doc(**{G: (89.9, 12)}))
    assert rc == 0, err
    assert json.loads(out) == {"files": {G: {"percent": 89.9, "missing": 10}}}  # the new entry only


def test_merge_floors_refuses_a_spread_wider_than_k(tmp_path, capsys, committed):
    """SF-3: 100 statements (k = 5); the runs differ by 6."""
    rc, out, err = _merge(tmp_path, capsys, _run_doc(**{G: (90.0, 10)}), _run_doc(**{G: (84.0, 16)}))
    assert rc == 1 and out == ""
    assert f"{G}: the runs' missing differ by 6 statements, more than its k = 5" in err


def test_merge_floors_refuses_a_scope_measured_in_one_run(tmp_path, capsys, committed):
    rc, _, err = _merge(tmp_path, capsys, _run_doc(**{G: (90.0, 10)}), _run_doc())
    assert rc == 1 and f"{G}: measured in 1 of 2 runs" in err


def test_merge_floors_k_comes_from_a_lower_bound_on_statements(tmp_path, capsys, committed):
    """10,000 statements at 70.0% (k = 10): a spread of 10 passes, 11 is refused."""
    a = _run_doc(**{G: (70.0, 3000)})
    assert _merge(tmp_path, capsys, a, _run_doc(**{G: (69.9, 3010)}))[0] == 0
    rc, _, err = _merge(tmp_path, capsys, a, _run_doc(**{G: (69.8, 3011)}))
    assert rc == 1 and "differ by 11 statements, more than its k = 10" in err


def test_merge_floors_refuses_a_duplicate_key(tmp_path, capsys, committed):
    dup = _run_doc().replace('"packages": {}', '"packages": {}, "packages": {}')
    rc, _, err = _merge(tmp_path, capsys, dup, _run_doc())
    assert rc == 2 and "duplicate JSON key 'packages'" in err


def test_merge_floors_merges_only_the_scopes_null_in_the_committed_file(tmp_path, capsys, committed):
    """SF-A4: the overall scope's runs differ by far more than its k, but overall is pinned in the committed file,
    so it is neither merged nor allowed to block the new file floor's pin. --all-scopes merges (and refuses) it."""
    committed(files={G: None})
    a = _run_doc(**{G: (90.0, 10)})
    b = _run_doc(**{G: (89.9, 11)}).replace('"missing": 100', '"missing": 140', 1)
    rc, out, err = _merge(tmp_path, capsys, a, b)
    assert rc == 0, err
    assert json.loads(out) == {"files": {G: {"percent": 89.9, "missing": 10}}}
    rc, out, err = _merge(tmp_path, capsys, a, b, flags=("--all-scopes",))
    assert rc == 1 and out == "" and "overall: the runs' missing differ by 40 statements" in err


def test_merge_floors_merges_a_null_overall_and_a_package_absent_from_the_committed_file(tmp_path, capsys, committed):
    committed(overall=None)
    a = _run_doc().replace('"packages": {}', '"packages": {"maxim/new": {"percent": 50.0, "missing": 9}}')
    b = _run_doc().replace('"packages": {}', '"packages": {"maxim/new": {"percent": 50.1, "missing": 8}}')
    rc, out, err = _merge(tmp_path, capsys, a, b)
    assert rc == 0, err
    assert json.loads(out) == {
        "overall": {"percent": 80.0, "missing": 100},
        "packages": {"maxim/new": {"percent": 50.0, "missing": 8}},
    }


def test_merge_floors_with_nothing_new_is_refused(tmp_path, capsys, committed):
    rc, out, err = _merge(tmp_path, capsys, _run_doc(), _run_doc().replace("80.0", "80.1"))
    assert rc == 1 and out == "" and "nothing new to pin" in err


def test_merge_floors_refuses_byte_identical_inputs(tmp_path, capsys, committed):
    """SF-A1's cheap hardening: the same printed block given twice is one run, not two."""
    doc = _run_doc(**{G: (90.0, 10)})
    rc, out, err = _merge(tmp_path, capsys, doc, doc)
    assert rc == 1 and out == "" and "runs 1 and 2 are byte-identical" in err


@pytest.mark.parametrize(
    "entry,needle",
    [
        ({"percent": 80.0, "missing": None}, "percent and missing are pinned together"),
        ({"percent": 80.0}, "must be exactly {percent, missing}"),
        ({"percent": "80", "missing": 1}, "percent must be a number"),
    ],
)
def test_merge_floors_validates_every_entry_as_a_floor(tmp_path, capsys, committed, entry, needle):
    """N1: a null `missing` or a missing key is a clean format error (exit 2), not a traceback."""
    bad = json.dumps({"overall": entry, "packages": {}, "files": {}, "retired_files": []})
    rc, out, err = _merge(tmp_path, capsys, bad, _run_doc())
    assert rc == 2 and out == "" and needle in err


def test_the_committed_file_floors_are_case_exact_and_measured():
    """Every committed `files` key names a real, case-exact, non-symlink, non-excluded src/maxim file."""
    root = Path(__file__).resolve().parents[2]
    fl = L.parse_floors((root / L.FLOORS_REL).read_text())
    ex = L.parse_exclusions((root / L.EXCLUSIONS_REL).read_text())
    assert fl.version == L.FLOORS_FORMAT_VERSION == 2
    listing = set(L._src_files(root))
    for rel in fl.files:
        assert L.file_present(root, rel, listing) and rel not in ex.paths, rel


# ── --pins-from-run (M60a): CI's pins from the run's own artifact, in one command ───────────────────────────────────


def _ci_artifact(repo: Path, dest: Path, *, absolute: bool = False, **cov) -> Path:
    """The repo's measurement as CI writes it: relative ``src/maxim/...`` keys (run 40549504653), or, with
    ``absolute``, keys under a runner's checkout."""
    _cov(repo, **cov)
    data = json.loads((repo / "coverage.json").read_text())
    if absolute:
        data["files"] = {f"/home/runner/work/Maxim/Maxim/{k}": v for k, v in data["files"].items()}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "coverage.json").write_text(json.dumps(data))
    return dest / "coverage.json"


def _api(head: str, *, attempt=1, suite="success", artifacts=("coverage-json-1",), job="unit-tests"):
    def api(path: str):
        if path.endswith("/jobs?per_page=100"):
            assert f"/attempts/{attempt}/" in path  # the LATEST attempt's jobs, never attempt 1's by default
            steps = [] if suite is None else [{"name": L.RUN_SUITE_STEP, "conclusion": suite}]
            return {"jobs": [{"name": job, "steps": steps}]}
        if path.endswith("/artifacts?per_page=100"):
            return {"artifacts": [{"name": n, "expired": False} for n in artifacts]}
        return {"head_sha": head, "run_attempt": attempt, "path": ".github/workflows/test.yml", "event": "pull_request"}

    return api


def _measure(repo, tmp_path, api, **cov):
    art = _ci_artifact(repo, tmp_path / "ci", **cov)
    got: list[str] = []

    def download(run_id, name, dest):
        got.append(name)
        return art

    return L.run_measurement(repo, "123", api=api, download=download, tmp=tmp_path / "dl"), got


def test_pins_from_run_reads_the_attempt_bound_artifact_and_reroots_runner_paths(repo, tmp_path):
    head = _git(repo, "rev-parse", "HEAD").strip()
    cov, got = _measure(repo, tmp_path, _api(head, attempt=2, artifacts=("coverage-json-1", "coverage-json-2")))
    assert got == ["coverage-json-2"]  # attempt 1's (possibly partial, -x) upload is never read
    assert set(cov) >= {A, B}


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"head": "0" * 40}, "not this checkout's HEAD"),
        ({"suite": "failure"}, "is failure, not success"),
        ({"suite": None}, "is absent, not success"),
        ({"job": "lint"}, "has no unit-tests job"),
        ({"attempt": 2, "artifacts": ("coverage-json",)}, "no unexpired artifact 'coverage-json-2'"),
    ],
)
def test_pins_from_run_refuses_a_run_it_cannot_bind(repo, tmp_path, kw, match):
    head = kw.pop("head", _git(repo, "rev-parse", "HEAD").strip())
    with pytest.raises(L.MeasurementError, match=match):
        _measure(repo, tmp_path, _api(head, **kw))


def test_pins_from_run_accepts_the_legacy_artifact_name_on_attempt_one_only(repo, tmp_path):
    head = _git(repo, "rev-parse", "HEAD").strip()
    _, got = _measure(repo, tmp_path, _api(head, artifacts=("coverage-json",)))
    assert got == ["coverage-json"]


def test_pins_from_run_accepts_a_measured_file_the_checkout_lacks(repo, tmp_path):
    """A PR run measures the merge with main, which may hold a file this branch lacks; the pins are for that merge,
    and CI re-measures it on the pin commit (delta2 P2: "merge main first" may never converge)."""
    head = _git(repo, "rev-parse", "HEAD").strip()
    art = _ci_artifact(repo, tmp_path / "ci")
    data = json.loads(art.read_text())
    data["files"]["src/maxim/alpha/new.py"] = next(iter(data["files"].values()))
    art.write_text(json.dumps(data))
    cov = L.run_measurement(repo, "1", api=_api(head), download=lambda *a: art, tmp=tmp_path)
    assert "src/maxim/alpha/new.py" in cov
    with pytest.raises(L.MeasurementError, match="does not exist"):  # the local loader still refuses a stale file
        _cov(repo)
        data = json.loads((repo / "coverage.json").read_text())
        data["files"]["src/maxim/alpha/new.py"] = next(iter(data["files"].values()))
        (repo / "coverage.json").write_text(json.dumps(data))
        L.load_coverage(repo)


def test_pins_are_the_ci_hint_and_falls_nulls_are_left_alone(repo):
    fl = L.parse_floors(_floors(alpha=(80.0, 2), beta=(70.0, 3)))
    _cov(repo, missing={A: set(), B: set(range(2, 11))})  # alpha 100% (a rise), beta 10% (a fall beyond k)
    cov = L.load_coverage(repo)
    overall, pkgs = L.measure_scopes(cov)
    pins, notes = L.pins_from_measurement(fl, overall, pkgs, L.file_scopes(cov))
    # round_down(measured), missing min(measured, pin): alpha 100%/0; overall (55%) is within its band, left alone
    assert pins == {"maxim/alpha": L.Floor(100.0, 0)}
    assert "maxim/beta" not in pins and any("maxim/beta" in n and "a pin cannot fix a fall" in n for n in notes)
    fl_null = L.parse_floors(_floors(alpha=None))
    _, notes = L.pins_from_measurement(fl_null, overall, pkgs, {})
    assert any("maxim/alpha: null floor" in n for n in notes)


def test_a_pinned_missing_from_the_run_never_rises(repo):
    """The pin keeps the current missing when the measurement adds uncovered statements (#1409)."""
    fl = L.parse_floors(_floors(alpha=(50.0, 1)))
    _cov(repo, missing={A: {9, 10}, B: {6, 7, 8, 9, 10}})  # alpha 80%, missing 2 > pinned 1
    cov = L.load_coverage(repo)
    overall, pkgs = L.measure_scopes(cov)
    pins, _ = L.pins_from_measurement(fl, overall, pkgs, {})
    assert pins["maxim/alpha"] == L.Floor(80.0, 1)


def test_apply_pins_rewrites_only_the_pinned_entries(repo):
    text = (repo / L.FLOORS_REL).read_text()
    out = L.apply_pins(text, {"maxim/alpha": L.Floor(90.0, 1), A: L.Floor(95.0, 0)})
    doc = json.loads(out)
    assert doc["packages"]["maxim/alpha"] == {"percent": 90.0, "missing": 1}
    assert doc["files"][A] == {"percent": 95.0, "missing": 0}
    assert doc["packages"]["maxim/beta"] == json.loads(text)["packages"]["maxim/beta"]
    assert out.endswith("}\n")


def test_the_committed_floors_file_round_trips_through_apply_pins():
    """--pins-from-run rewrites the real file with json.dumps(indent=2): a no-op pin set must change no byte."""
    text = (Path(__file__).resolve().parents[2] / L.FLOORS_REL).read_text(encoding="utf-8")
    assert L.apply_pins(text, {}) == text


def test_pins_from_run_main_writes_the_file_and_ci_accepts_it(repo, tmp_path, monkeypatch, capsys):
    head = _git(repo, "rev-parse", "HEAD").strip()
    art = _ci_artifact(repo, tmp_path / "ci", missing={A: set(), B: {6, 7, 8, 9, 10}})
    monkeypatch.setattr(L, "gh_api", _api(head))
    monkeypatch.setattr(L, "_download_artifact", lambda run_id, name, dest: art)
    monkeypatch.setattr(L.run_measurement, "__kwdefaults__", {"api": _api(head), "download": lambda *a: art})
    rc = L.main(["--pins-from-run", "123"])
    out = capsys.readouterr().out
    assert rc == 0 and "pin maxim/alpha: 80.0%/2 -> 100.0%/0" in out
    assert json.loads((repo / L.FLOORS_REL).read_text())["packages"]["maxim/alpha"] == {"percent": 100.0, "missing": 0}
    _commit(repo)
    _cov(repo, missing={A: set(), B: {6, 7, 8, 9, 10}})
    rc_ci, _, err = _run(capsys)
    assert "maxim/alpha" not in err, err  # the pinned scope now passes CI's own rules


def test_pins_from_run_main_refuses_a_non_numeric_id(capsys):
    assert L.main(["--pins-from-run", "abc"]) == 2
    assert L.main(["--pins-from-run"]) == 2


@pytest.mark.parametrize("over, fails", [(0, False), (L.K_MIN, False), (L.K_MIN + 1, True)])
def test_a_changed_pin_may_sit_k_min_above_the_measurement_never_above_its_base(repo, capsys, over, fails):
    """M60a / delta2 P1: a pin taken from one CI run is verified on the pin commit's run (+-2 statements of noise), so a
    CHANGED pin may sit up to K_MIN (the absolute noise, not the size-scaled k) above the measured missing; one more
    fails. Measured alpha: 80%, missing 2; the base pin is 80%/20 so the base side never binds here."""
    _write(repo, L.FLOORS_REL, _floors(alpha=(70.0, 20)))
    _commit(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _write(repo, L.FLOORS_REL, _floors(alpha=(79.0, 2 + over)))
    _commit(repo)
    _cov(repo)
    rc, _, err = _run(capsys)
    assert ("a pinned missing never rises" in err) is fails, err


def test_the_k_min_slack_never_lets_a_pin_rise_above_its_base(repo, capsys):
    """The base side stays exact: measured 2 + K_MIN = 7 would allow 7, but the base pin is 3, so 4 fails."""
    _write(repo, L.FLOORS_REL, _floors(alpha=(70.0, 3)))
    _commit(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _write(repo, L.FLOORS_REL, _floors(alpha=(79.0, 4)))
    _commit(repo)
    _cov(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/alpha: floor pins missing 4 above" in err


def test_pins_from_run_refuses_a_run_of_another_workflow_or_event(repo, tmp_path):
    head = _git(repo, "rev-parse", "HEAD").strip()
    base = _api(head)

    def api(path):
        r = base(path)
        return {**r, "event": "schedule"} if "head_sha" in r else r

    with pytest.raises(L.MeasurementError, match="pin from a Tests run"):
        _measure(repo, tmp_path, api)


def test_the_pin_noise_slack_is_k_min_not_the_size_scaled_k(repo):
    """delta2 P1: on a 20,000-statement scope k is 20 statements, but CI's noise is ~2 at any size. The slack is the
    absolute K_MIN: measured missing 1000, base pin 2000, so a pin of 1000 + K_MIN passes and 1000 + K_MIN + 1 fails
    (with k it would pass up to 1020). Deleting the fix (``+ missing_slack(m)``) fails this test."""
    big = L.ScopeCov(statements=20000, covered=19000)
    assert L.missing_slack(big) == 20 > L.K_MIN

    def rules(pin_missing: int) -> list[str]:
        base = L.parse_floors(_floors(overall=(90.0, 2000), alpha=None, beta=None))
        head = L.parse_floors(_floors(overall=(95.0, pin_missing), alpha=None, beta=None))
        return L.floor_diff_rules(repo, base, head, big, {}, {}, files={}, fd=L.NO_FILE_DIFF, excluded=set())

    assert not [r for r in rules(1000 + L.K_MIN) if r.startswith("overall")]
    assert any("overall: floor pins missing" in r for r in rules(1000 + L.K_MIN + 1))


def test_pins_from_run_reroots_only_an_absolute_runner_key(repo, tmp_path):
    """CI writes relative keys; an absolute runner key is re-rooted, and a relative key is never rewritten (code review
    E1: a relative key holding ``/src/maxim/`` twice used to be mangled)."""
    head = _git(repo, "rev-parse", "HEAD").strip()
    art = _ci_artifact(repo, tmp_path / "ci", absolute=True)
    cov = L.run_measurement(repo, "1", api=_api(head), download=lambda *a: art, tmp=tmp_path)
    assert A in cov and not any(k.startswith("/") for k in cov)
    data = json.loads(art.read_text())
    data["files"] = {"src/maxim/alpha/src/maxim/x.py": next(iter(data["files"].values()))}
    art.write_text(json.dumps(data))
    assert set(L.run_measurement(repo, "1", api=_api(head), download=lambda *a: art, tmp=tmp_path)) == {
        "src/maxim/alpha/src/maxim/x.py"
    }


def test_pins_from_run_is_refused_inside_github_actions(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert L.main(["--pins-from-run", "123"]) == 2
    assert "cannot run inside GitHub Actions" in capsys.readouterr().err


def test_the_pinner_names_match_the_real_workflow():
    """Wire W1: the pinner hard-codes the unit-tests job, its measuring step's name and the attempt-bound artifact; a
    rename in test.yml would make every --pins-from-run refuse while CI stays green, so the real workflow is read."""
    wf = yaml.safe_load((Path(__file__).resolve().parents[2] / ".github/workflows/test.yml").read_text())
    job = wf["jobs"]["unit-tests"]
    assert "name" not in job and "strategy" not in job  # the jobs API reports it as exactly "unit-tests"
    suite = [s for s in job["steps"] if s.get("name") == L.RUN_SUITE_STEP]
    assert len(suite) == 1 and L.CI_COV_ARGS in " ".join(suite[0]["run"].split())
    up = [s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert [s["with"]["name"] for s in up] == [L.COVERAGE_ARTIFACT + "-${{ github.run_attempt }}"]
    assert up[0]["with"]["path"] == L.COVERAGE_JSON and up[0]["if"] == "always()"


def test_gh_runs_from_the_checkout_it_pins(repo, tmp_path, monkeypatch):
    """Code delta NIT: ``gh`` resolves ``{owner}/{repo}`` from its working directory, while HEAD is read from the
    checkout, so both gh calls run from the checkout, not the caller's shell directory."""
    seen: list[object] = []

    class R:
        returncode, stdout, stderr = 0, "{}", ""

    def run(cmd, **kw):
        seen.append(kw.get("cwd"))
        return R()

    monkeypatch.setattr(L.subprocess, "run", run)
    monkeypatch.setattr(L, "REPO_ROOT", repo)
    L._download_artifact("1", "coverage-json-1", tmp_path)
    import _lint_git

    monkeypatch.setattr(_lint_git.subprocess, "run", run)
    _lint_git.gh_api("repos/{owner}/{repo}/actions/runs/1", cwd=repo)
    assert seen == [repo, repo]
    with pytest.raises(L.MeasurementError):  # run_measurement's default api passes the checkout as cwd
        L.run_measurement(repo, "1", tmp=tmp_path)
    assert seen[-1] == repo
