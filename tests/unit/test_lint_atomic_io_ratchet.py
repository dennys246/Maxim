"""scripts/lint_atomic_io_ratchet.py — counting all four rename spellings + the ratchet (item 16.3).

Verified against the first draft, which matched only the bare `os.replace` name: it read 6
where the truth is 12 — missing the `import os as _os` alias (models/download.py), the
`Path.replace` on the provenance decision log, and `os.rename` in inference/transcribe_audio.py.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts import _lint_git
from scripts import lint_atomic_io_ratchet as L


def _lines(text: str) -> list[int]:
    return [ln for ln, _ in L.rename_call_sites(text)]


def test_counts_call_sites_not_mentions() -> None:
    src = (
        "import os\n"
        "# comment: os.replace(a, b)\n"
        "doc = '''os.replace in a string'''\n"
        "def f():\n"
        "    os.replace('a', 'b')\n"
        "    return os.replace\n"
    )
    assert L.rename_call_sites(src) == [(5, "os.replace")]


def test_resolves_import_alias_and_os_rename() -> None:
    assert L.rename_call_sites("import os as _os\n_os.replace('a', 'b')\n_os.rename('c', 'd')\n") == [
        (2, "os.replace"),
        (3, "os.rename"),
    ]


def test_matches_path_replace_and_rename() -> None:
    """The spellings the first draft missed: decision_log.py's `tmp.replace(path)` and
    bio_stack.py's `_old.rename(_new)`."""
    assert L.rename_call_sites("tmp.replace(path)\nold.rename(new)\n") == [(1, "Path.replace"), (2, "Path.rename")]


def test_ignores_str_replace_and_dataclasses_replace() -> None:
    src = "import dataclasses\ns = 'x'.replace('a', 'b')\ncfg = dataclasses.replace(other)\n"
    assert L.rename_call_sites(src) == []
    assert L.rename_call_sites("from dataclasses import replace\ncfg = replace(other)\n") == []


def test_syntax_error_counts_zero() -> None:
    assert L.rename_call_sites("def (:\n") == []


def _git(root: Path, *args: str) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "src/maxim/utils").mkdir(parents=True)
    (tmp_path / "src/maxim/utils/atomic_io.py").write_text("import os\nos.replace('t', 'p')\n")
    (tmp_path / "src/maxim/old.py").write_text("import os\nos.replace('a', 'b')\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path, _git(tmp_path, "rev-parse", "HEAD").strip()


def _ratchet(root: Path, base: str) -> list[str]:
    return _lint_git.count_ratchet(
        root, base, L.SRC.as_posix(), L.rename_call_sites, exclude=frozenset({L.CANONICAL.as_posix()})
    )


def test_canonical_writer_is_excluded_and_counts_are_per_file(repo) -> None:
    root, _ = repo
    assert {k: _lines_of(v) for k, v in L.counts(root).items()} == {"src/maxim/old.py": [2]}


def _lines_of(sites: list[tuple[int, str]]) -> list[int]:
    return [ln for ln, _ in sites]


def test_ratchet_fails_when_a_file_count_rises(repo) -> None:
    root, base = repo
    (root / "src/maxim/old.py").write_text("import os\nos.replace('a', 'b')\ntmp.replace(dest)\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "more")
    fails = _ratchet(root, base)
    assert len(fails) == 1 and "rose 1 → 2" in fails[0]


def test_ratchet_fails_on_a_new_file_with_a_site_and_allows_burn_down(repo) -> None:
    root, base = repo
    (root / "src/maxim/new.py").write_text("import os\nos.rename('a', 'b')\n")
    (root / "src/maxim/old.py").write_text("import os\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "shuffle")
    assert [f.split(":")[0] for f in _ratchet(root, base)] == ["src/maxim/new.py"]


def test_moving_a_site_within_a_file_is_free(repo) -> None:
    root, base = repo
    (root / "src/maxim/old.py").write_text("import os\n\n\ndef g():\n    os.replace('a', 'b')\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "move")
    assert _ratchet(root, base) == []


def test_real_tree_counts_every_spelling() -> None:
    """The spellings the first draft missed are present in the real tree — if this
    shrinks to os.replace-only, the matcher regressed."""
    spellings = {s for sites in L.counts().values() for _, s in sites}
    assert {"os.replace", "os.rename", "Path.replace", "Path.rename"} <= spellings


# ── the vacuous-guard blocker: a skipped diff-scoped check on a PR is an ERROR ──


def test_must_not_skip_is_true_only_on_a_pull_request(monkeypatch, capsys) -> None:
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    assert _lint_git.must_not_skip("no base") is False
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    assert _lint_git.must_not_skip("no base") is False
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    assert _lint_git.must_not_skip("no base") is True
    assert "fetch-depth: 0" in capsys.readouterr().err


def test_ratchet_follows_renames(repo) -> None:
    """A pure move must not read as 2 new sites — item 7's decomposition is all moves."""
    root, base = repo
    _git(root, "mv", "src/maxim/old.py", "src/maxim/moved.py")
    _git(root, "commit", "-q", "-m", "move file")
    assert _ratchet(root, base) == []
    (root / "src/maxim/moved.py").write_text("import os\nos.replace('a', 'b')\nos.rename('c', 'd')\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "move and add")
    fails = _ratchet(root, base)
    assert len(fails) == 1 and "renamed from src/maxim/old.py" in fails[0] and "rose 1 → 2" in fails[0]


def test_show_distinguishes_absent_from_failed(repo) -> None:
    root, base = repo
    assert _lint_git.show(root, base, "src/maxim/old.py").startswith("import os")
    assert _lint_git.show(root, base, "src/maxim/never.py") == ""
    with pytest.raises(_lint_git.GitUnavailable):
        _lint_git.show(root, "not-a-ref", "src/maxim/old.py")
