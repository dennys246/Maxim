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


def _floors(overall=(65.0, 7), alpha=(80.0, 2), beta=(50.0, 5), band=1.0, n=N, **more) -> str:
    def f(v):
        return {"percent": v[0], "missing": v[1]} if v is not None else {"percent": None, "missing": None}

    pk = {"maxim/alpha": alpha, "maxim/beta": beta, **more}
    return json.dumps(
        {
            "floors_format_version": 1,
            "band": band,
            "min_package_statements": n,
            "overall": f(overall),
            "packages": {k: f(v) for k, v in pk.items() if v != "drop"},
        },
        indent=1,
    )


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
        and 'maxim/alpha: 100.00% is more than 1.0 pt above its floor 80.0% — raise it to {"percent": 100.0' in err
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
    assert rc == 1 and "maxim/alpha: floor pins missing 3 above min(measured 0, base 2)" in err


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
    assert L.main(["--merge-floors", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0
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
    assert rc == 1 and "maxim/alpha: floor pins missing 3 above min(measured 2, base 2)" in err
    # suppose link 1 had merged anyway (the old rule passed it): link 2 must still fail
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")
    _git(repo, "checkout", "-q", "feature")
    _cov(repo, missing={A: {8, 9, 10}, B: {6, 7, 8, 9, 10}})  # one covered statement's test dropped
    _write(repo, L.FLOORS_REL, _floors(overall=(60.0, 9), alpha=(70.0, 4)))
    _commit(repo)
    rc, _, err = _run(capsys)
    assert rc == 1 and "maxim/alpha: floor pins missing 4 above min(measured 3, base 3)" in err
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
    merged = capsys.readouterr().out
    assert json.loads(merged)["packages"]["maxim/gamma"] == {"percent": 86.7, "missing": 86}
    _write(repo, L.FLOORS_REL, merged)
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
    assert rc == 1 and "maxim/gamma: floor pins missing 11 above min(measured 11, base 10)" in err
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
    assert rc == 1 and "overall: floor pins missing 8 above min(measured 8, base 7)" in err


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
