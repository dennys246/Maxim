"""scripts/lint_atomic_io_ratchet.py — counting + the diff-scoped ratchet (roadmap 1.1.x item 16.3).

Verified: the alias shape (`import os as _os; _os.replace(...)`, models/download.py) was
missed by the first draft — the count read 6 where the truth is 7.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts import lint_atomic_io_ratchet as L


def test_counts_call_sites_not_mentions() -> None:
    src = (
        "import os\n"
        "# comment: os.replace(a, b)\n"
        "doc = '''os.replace in a string'''\n"
        "def f():\n"
        "    os.replace('a', 'b')\n"
        "    os.path.replace_nothing\n"
        "    return os.replace\n"
    )
    assert L.replace_call_lines(src) == [5]


def test_resolves_import_alias() -> None:
    assert L.replace_call_lines("import os as _os\n_os.replace('a', 'b')\n") == [2]


def test_ignores_other_replace_calls() -> None:
    assert L.replace_call_lines("s = 'x'.replace('a', 'b')\nfrom pathlib import Path\nPath('a').replace('b')\n") == []


def test_syntax_error_counts_zero() -> None:
    assert L.replace_call_lines("def (:\n") == []


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


def test_canonical_writer_is_excluded_and_counts_are_per_file(repo) -> None:
    root, _ = repo
    assert L.counts(root) == {"src/maxim/old.py": [2]}


def test_ratchet_fails_when_a_file_count_rises(repo) -> None:
    root, base = repo
    (root / "src/maxim/old.py").write_text("import os\nos.replace('a', 'b')\nos.replace('c', 'd')\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "more")
    fails = L.ratchet_violations(root, base, L.counts(root))
    assert len(fails) == 1 and "rose 1 → 2" in fails[0]


def test_ratchet_fails_on_a_new_file_with_a_site_and_allows_burn_down(repo) -> None:
    root, base = repo
    (root / "src/maxim/new.py").write_text("import os\nos.replace('a', 'b')\n")
    (root / "src/maxim/old.py").write_text("import os\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "shuffle")
    fails = L.ratchet_violations(root, base, L.counts(root))
    assert [f.split(":")[0] for f in fails] == ["src/maxim/new.py"]


def test_moving_a_site_within_a_file_is_free(repo) -> None:
    root, base = repo
    (root / "src/maxim/old.py").write_text("import os\n\n\ndef g():\n    os.replace('a', 'b')\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "move")
    assert L.ratchet_violations(root, base, L.counts(root)) == []
