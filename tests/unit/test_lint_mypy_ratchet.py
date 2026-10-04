"""scripts/lint_mypy_ratchet.py — the repo-wide mypy per-file ratchet (roadmap 1.3.2 item 4, part B).

Every gate test drives ``main()`` on a fixture git repo with a tiny ``src/maxim`` package and the REAL
pinned mypy (about 0.5 s per run on these trees), because the parse of mypy's output is half the gate:
a lint that reads mypy's output wrong passes everything.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts import lint_mypy_ratchet as L

_ERR = 'x: int = "a"\n'  # exactly one mypy error
_CLEAN = "y = 1\n"
_IGNORED = 'z: int = "q"  # type: ignore[assignment]\n'  # one inline suppression, zero errors


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


@pytest.fixture
def wired(tmp_path: Path, monkeypatch):
    """Base on `main`: a.py has one error, b.py one suppressed line; HEAD is a `feature` branch."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _write(tmp_path, "src/maxim/__init__.py", "")
    _write(tmp_path, "src/maxim/a.py", _ERR)
    _write(tmp_path, "src/maxim/b.py", _CLEAN + _IGNORED)
    _commit(tmp_path)
    _git(tmp_path, "checkout", "-q", "-b", "feature")
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    return tmp_path


# ── main(), as wired ─────────────────────────────────────────────────────────


def test_untouched_branch_is_clean_and_prints_both_totals(wired, capsys):
    assert L.main() == 0
    out = capsys.readouterr().out
    assert "HEAD: 1 error(s) in 1 file(s), 1 suppression(s)" in out
    assert "base " in out and "no per-file deltas" in out


def test_a_new_error_FAILS(wired, capsys):
    _write(wired, "src/maxim/b.py", _CLEAN + _IGNORED + "w: str = 1\n")
    _commit(wired)
    assert L.main() == 1
    assert "src/maxim/b.py: mypy error count rose 0 → 1" in capsys.readouterr().err


def test_a_fixed_error_passes(wired, capsys):
    _write(wired, "src/maxim/a.py", _CLEAN)
    _commit(wired)
    assert L.main() == 0
    assert "src/maxim/a.py: errors 1 → 0" in capsys.readouterr().out


def test_an_error_moved_to_a_new_file_FAILS(wired, capsys):
    _write(wired, "src/maxim/a.py", _CLEAN)
    _write(wired, "src/maxim/c.py", "def f() -> None:\n    " + _ERR)
    _commit(wired)
    assert L.main() == 1
    assert "src/maxim/c.py (new file): mypy error count rose 0 → 1" in capsys.readouterr().err


def test_a_pure_rename_keeps_its_base_count(wired, capsys):
    _git(wired, "mv", "src/maxim/a.py", "src/maxim/renamed.py")
    _git(wired, "mv", "src/maxim/b.py", "src/maxim/b2.py")
    _commit(wired)
    assert L.main() == 0, capsys.readouterr().err


def test_an_added_type_ignore_FAILS_even_when_errors_fall(wired, capsys):
    _write(wired, "src/maxim/a.py", _CLEAN + 'v: int = "b"  # type: ignore\n')  # 1 → 0 errors, 0 → 1 suppressions
    _commit(wired)
    assert L.main() == 1
    err = capsys.readouterr().err
    assert "suppression count rose 0 → 1" in err and "error count rose" not in err


def test_a_type_ignore_inside_a_string_and_a_non_column_0_directive_do_not_count():
    # A `type: ignore` is a tokenizer comment, so one inside a string is not one. The `# mypy:` text here
    # does not count ONLY because it is not at column 0 of a raw line — a column-0 directive inside a
    # string DOES count (mypy honours it): see test_a_mypy_directive_inside_a_string_FAILS.
    assert L.suppression_count('s = "# type: ignore"\nt = """# mypy: ignore-errors"""\n') == 0


@pytest.mark.parametrize(
    "src",
    [
        "from typing import no_type_check\n@no_type_check\ndef f(): pass\n",
        "import typing\n@typing.no_type_check\ndef f(): pass\n",
        "from typing import TYPE_CHECKING\nif not TYPE_CHECKING:\n    x = 1\n",
        "import typing\nif typing.TYPE_CHECKING:\n    x = 1\nelse:\n    x = 2\n",
        "# mypy: disable-error-code=assignment\nx = 1\n",
    ],
)
def test_every_suppression_shape_counts(src):
    assert L.suppression_count(src) >= 1


def test_a_new_file_level_mypy_ignore_errors_FAILS(wired, capsys):
    # Trades b.py's one inline ignore for a whole-module one: the suppression COUNT is unchanged,
    # so only the file-level check can catch it.
    _write(wired, "src/maxim/b.py", "# mypy: ignore-errors\n" + _CLEAN)
    _commit(wired)
    assert L.main() == 1
    err = capsys.readouterr().err
    assert "NEW file-level suppression" in err and "suppression count rose" not in err


def test_a_new_leading_type_ignore_FAILS(wired, capsys):
    _write(wired, "src/maxim/b.py", "# type: ignore\n" + _CLEAN)
    _commit(wired)
    assert L.main() == 1
    err = capsys.readouterr().err
    assert "NEW file-level suppression" in err and "suppression count rose" not in err


def test_a_new_pyi_FAILS(wired, capsys):
    # A stub beside its source REPLACES it: mypy then never checks a.py, its one error vanishes, and the
    # checked-module count is unchanged — only the .pyi check catches this.
    _write(wired, "src/maxim/a.pyi", "x: int\n")
    _commit(wired)
    assert L.main() == 1
    assert "src/maxim/a.pyi: NEW .pyi stub" in capsys.readouterr().err


def test_a_syntax_error_in_a_file_with_errors_FAILS_closed(wired, capsys):
    # The blocking error hides every other error in the tree (a.py's count would read 1 → 1).
    _write(wired, "src/maxim/a.py", _ERR + "def (:\n")
    _commit(wired)
    assert L.main() == 2
    assert "cannot be trusted" in capsys.readouterr().err


def test_unrecognised_mypy_output_FAILS_closed(wired, monkeypatch, capsys):
    real = L.run_mypy

    def odd(tree):
        rc, out, err = real(tree)
        return rc, "some new mypy chatter\n" + out, err

    monkeypatch.setattr(L, "run_mypy", odd)
    assert L.main() == 2
    assert "unrecognised mypy output line" in capsys.readouterr().err


def test_a_config_file_added_in_the_pr_has_no_effect(wired, capsys):
    _write(wired, "mypy.ini", "[mypy]\nignore_errors = True\n")
    _write(wired, "setup.cfg", "[mypy]\nignore_errors = True\n")
    _write(wired, "pyproject.toml", "[tool.mypy]\nignore_errors = true\n")
    _write(wired, "src/maxim/b.py", _CLEAN + _IGNORED + "w: str = 1\n")
    _commit(wired)
    assert L.main() == 1
    assert "src/maxim/b.py: mypy error count rose 0 → 1" in capsys.readouterr().err


def test_missing_merge_base_on_a_pull_request_FAILS(wired, monkeypatch, capsys):
    _git(wired, "branch", "-q", "-D", "main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    assert L.main() == 2
    assert "cannot run on a pull request" in capsys.readouterr().err


def test_missing_merge_base_locally_skips_with_totals(wired, capsys):
    _git(wired, "branch", "-q", "-D", "main")
    assert L.main() == 0
    out = capsys.readouterr().out
    assert "HEAD: 1 error(s)" in out and "INFO: no base ref" in out


def test_a_push_is_judged_against_the_last_green_push(wired, monkeypatch, capsys):
    """Was totals-only on push (#1089): a change that reached main without a PR went unjudged."""
    from tests.unit._push_event_helpers import fake_push

    _write(wired, "src/maxim/b.py", "w: str = 1\n")  # a new error in b.py
    _commit(wired)
    fake_push(monkeypatch, wired)
    assert L.main() == 1
    out = capsys.readouterr()
    assert "totals only" not in out.out and "b.py" in out.err


def test_another_ci_event_prints_head_totals_only(wired, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    _write(wired, "src/maxim/b.py", "w: str = 1\n")  # would fail a PR
    _commit(wired)
    assert L.main() == 0
    out = capsys.readouterr().out
    assert "HEAD: 2 error(s)" in out and "totals only" in out and "base " not in out


# ── the fail-closed parse, directly ──────────────────────────────────────────

_OK = "src/maxim/a.py:1: error: Bad  [assignment]\nFound 1 error in 1 file (checked 3 source files)\n"


def test_parse_reads_a_well_formed_run():
    errors, checked = L.parse_mypy(1, _OK, "", n_modules=3)
    assert errors == {"src/maxim/a.py": 1} and checked == 3


@pytest.mark.parametrize(
    ("rc", "out", "err", "n", "why"),
    [
        (2, _OK, "", 3, "exited 2"),
        (
            1,
            "src/maxim/a.py:1: error: Invalid syntax  [syntax]\n"
            "Found 1 error in 1 file (errors prevented further checking)\n",
            "",
            3,
            "blocking error",
        ),
        (1, _OK, "warning: x\n", 3, "stderr"),
        (1, _OK, "", 4, "checked 3 source files but the tree has 4"),
        (1, _OK.replace("Found 1 error", "Found 2 errors"), "", 3, "summary says 2"),
        (1, "src/maxim/a.py:1: error: Bad\n", "", 3, "no mypy summary"),
        (0, _OK, "", 3, "disagrees"),
    ],
)
def test_parse_fails_closed(rc, out, err, n, why):
    with pytest.raises(L.InstrumentError, match=why):
        L.parse_mypy(rc, out, err, n_modules=n)


# ── review-round folds ───────────────────────────────────────────────────────


@pytest.mark.parametrize("directive", ['disable-error-code="assignment"', "ignore-errors"])
def test_a_mypy_directive_inside_a_string_FAILS(wired, capsys, directive):
    """mypy reads `# mypy:` from RAW lines (util.py::get_mypy_comments), string literal or not, so a
    docstring-embedded directive silences a.py's error: 1 → 0 errors, and it must cost a suppression."""
    _write(wired, "src/maxim/a.py", f'_DOC = """\n# mypy: {directive}\n"""\n' + _ERR)
    _commit(wired)
    assert L.main() == 1
    err = capsys.readouterr().err
    assert "suppression count rose 0 → 1" in err
    if directive == "ignore-errors":
        assert "NEW file-level suppression" in err


def test_mypy_directive_rule_matches_mypys_own():
    from mypy.util import get_mypy_comments

    src = '# mypy: a\n  # mypy: indented\n#mypy: nospace\ns = """\n# mypy: in-string\n"""\nx = 1  # mypy: trailing\n'
    assert len(L.mypy_directives(src)) == len(get_mypy_comments(src)) == 2


@pytest.mark.parametrize(
    "raw",
    [
        b"x = 1\r\n# mypy: ignore-errors\r\n",  # CRLF: split on "\n" keeps the "\r", the prefix still matches
        b"\xef\xbb\xbf# mypy: ignore-errors\nx = 1\n",  # BOM: mypy strips it before reading directives
        b"# -*- coding: latin-1 -*-\n# mypy: ignore-errors\ns = '\xe9'\n",  # PEP 263
        b"x = 1\r# mypy: ignore-errors\r",  # lone CR: one raw line to mypy, so no directive
    ],
)
def test_source_decoding_and_directives_match_mypy(raw):
    from mypy.util import decode_python_encoding, get_mypy_comments

    assert L.read_source(raw) == decode_python_encoding(raw)
    assert len(L.mypy_directives(L.read_source(raw))) == len(get_mypy_comments(decode_python_encoding(raw)))


@pytest.mark.parametrize(
    "src",
    [
        "import typing\nif typing.TYPE_CHECKING or False:\n    pass\nelif True:\n    x = 1\n",
        "from typing import TYPE_CHECKING\nif not TYPE_CHECKING and f():\n    x = 1\n",
        "import typing\ndef g() -> int:\n    assert not typing.TYPE_CHECKING\n    return 1\n",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    x = 1\nelif True:\n    x = 2\n",
    ],
)
def test_a_compound_type_checking_test_counts(src):
    assert L.suppression_count(src) == 1


def test_the_plain_import_guard_is_not_a_suppression():
    assert L.suppression_count("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import os\n") == 0


def test_a_compound_type_checking_test_FAILS(wired, capsys):
    _write(
        wired,
        "src/maxim/b.py",
        _CLEAN + _IGNORED + "import typing\nif typing.TYPE_CHECKING or False:\n    pass\n"
        "elif True:\n    def bad() -> int:\n        return 'x'\n",
    )
    _commit(wired)
    assert L.main() == 1
    assert "suppression count rose 1 → 2" in capsys.readouterr().err


def test_installed_packages_are_invisible(tmp_path):
    """`--no-site-packages`: an installed, typed package (mypy itself ships py.typed) must read as Any,
    so a developer venv and the bare lint-job venv count the same. With site-packages visible this
    assignment is an error."""
    _write(tmp_path, "src/maxim/__init__.py", "")
    _write(tmp_path, "src/maxim/a.py", "import mypy.version\nv: int = mypy.version.__version__\n")
    assert sum(L.measure(tmp_path).errors.values()) == 0


def test_a_directive_after_a_utf8_bom_FAILS(wired, capsys):
    """mypy strips a BOM before reading directives, so the lint must read the file the same way."""
    (wired / "src/maxim/a.py").write_bytes(b"\xef\xbb\xbf# mypy: ignore-errors\n" + _ERR.encode())
    _commit(wired)
    assert L.main() == 1
    assert "NEW file-level suppression" in capsys.readouterr().err


@pytest.mark.parametrize(
    "exit_stmt",
    ["return 1", "raise RuntimeError", "assert False", "sys.exit(0)", "os._exit(0)"],
)
def test_an_early_exit_inside_the_plain_guard_counts(exit_stmt):
    """`if TYPE_CHECKING: return 1` makes the rest of the block unreachable to mypy — not an import guard."""
    src = f"import os, sys\nfrom typing import TYPE_CHECKING\ndef f() -> int:\n    if TYPE_CHECKING:\n        {exit_stmt}\n    return 2\n"
    assert L.suppression_count(src) == 1


@pytest.mark.parametrize("exit_stmt", ["continue", "break"])
def test_a_loop_exit_inside_the_plain_guard_counts(exit_stmt):
    src = (
        f"from typing import TYPE_CHECKING\nfor i in range(3):\n    if TYPE_CHECKING:\n        {exit_stmt}\n    x = i\n"
    )
    assert L.suppression_count(src) == 1


def test_an_early_exit_inside_the_plain_guard_FAILS(wired, capsys):
    _write(
        wired,
        "src/maxim/b.py",
        _CLEAN + _IGNORED + "from typing import TYPE_CHECKING\ndef f() -> int:\n    if TYPE_CHECKING:\n"
        "        return 1\n    x: int = 'bad'\n    return x\n",
    )
    _commit(wired)
    assert L.main() == 1
    assert "suppression count rose 1 → 2" in capsys.readouterr().err
