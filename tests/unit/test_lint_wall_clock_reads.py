"""M72's wall-clock lint (scripts/lint_wall_clock_reads.py; world_clock_a_pre_guard_note.md §Proof by deletion).

Every planted case runs in a temp git tree, so the lint's own fixtures never touch the real scan; the last test is
the positive control on this checkout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import lint_wall_clock_reads as lint  # noqa: E402

IN_SCOPE = "memory/planted.py"  # machinery scope (memory/)
PROMPT = "agents/prompt_builder.py"  # prompt scope


def _tree(tmp_path: Path, files: dict[str, str], entries: list[dict] | None = None, *, track: bool = True) -> Path:
    """A git tree with *files* under src/maxim and *entries* as the allowlist."""
    for rel, text in files.items():
        path = tmp_path / "src" / "maxim" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    allow = tmp_path / lint.ALLOWLIST_REL
    allow.parent.mkdir(parents=True, exist_ok=True)
    allow.write_text(json.dumps({"_format_version": "1.0", "entries": entries or []}), encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    if track:
        subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    return tmp_path


def _entry(site: str, reads: int = 1, category: str = "process", reason: str = "planted") -> dict:
    return {"site": site, "category": category, "reads": reads, "reason": reason}


def _violations(tmp_path: Path, files: dict[str, str], entries: list[dict] | None = None) -> list[str]:
    return lint.check(_tree(tmp_path, files, entries)).violations


@pytest.mark.parametrize(
    "src",
    [
        "import time\n\ndef f():\n    return time.time()\n",
        "import time as _t\n\ndef f():\n    return _t.time()\n",
        "from time import time as now\n\ndef f():\n    return now()\n",
        "from datetime import datetime\n\ndef f():\n    return datetime.now()\n",
        "import datetime as dt\n\ndef f():\n    return dt.datetime.now()\n",
        "from datetime import datetime as DT\n\ndef f():\n    return DT.now()\n",
        "import time\n\ndef f():\n    return time.monotonic()\n",
        "import time\n\ndef f():\n    return time.strftime('%H')\n",
        "from datetime import datetime\n\ndef f(s):\n    return datetime.fromtimestamp(s)\n",
        "from datetime import date\n\ndef f():\n    return date.today()\n",
    ],
    ids=[
        "time.time",
        "import-as",
        "from-import-as",
        "from-datetime",
        "datetime-as",
        "from-datetime-as",
        "monotonic",
        "strftime",
        "fromtimestamp",
        "date.today",
    ],
)
def test_each_banned_form_without_an_entry_fails(tmp_path: Path, src: str) -> None:
    found = _violations(tmp_path, {IN_SCOPE: src})
    assert len(found) == 1 and found[0].startswith(f"{IN_SCOPE}::f: 1 wall-clock read(s) with no allowlist entry"), (
        found
    )


def test_the_same_reads_with_their_entry_pass(tmp_path: Path) -> None:
    assert (
        _violations(
            tmp_path, {IN_SCOPE: "import time\n\ndef f():\n    return time.time()\n"}, [_entry(f"{IN_SCOPE}::f")]
        )
        == []
    )


def test_a_reference_not_only_a_call_counts(tmp_path: Path) -> None:
    """``default_factory=time.time`` binds at class definition: G-long cannot see it, so the static half must."""
    src = (
        "import time\nfrom dataclasses import dataclass, field\n\n"
        "@dataclass\nclass Rec:\n    ts: float = field(default_factory=time.time)\n"
    )
    found = _violations(tmp_path, {IN_SCOPE: src})
    assert len(found) == 1 and found[0].startswith(f"{IN_SCOPE}::Rec:"), found


def test_a_chain_counts_once(tmp_path: Path) -> None:
    src = "import datetime\n\ndef f():\n    return datetime.datetime.now()\n"
    assert _violations(tmp_path, {IN_SCOPE: src}, [_entry(f"{IN_SCOPE}::f", reads=1)]) == []


def test_no_counted_name_nests_in_another() -> None:
    """The scanner counts each counted link of a chain; one read stays one count only while no counted name is a
    dotted prefix of another. A banned name added to ``_clock_refs`` that nests would double-count: fail here."""
    from _clock_refs import CALENDAR, MONOTONIC, WALL

    counted = WALL | MONOTONIC | CALENDAR
    assert not [(a, b) for a in counted for b in counted if b.startswith(a + ".")]


def test_a_read_added_to_an_allowlisted_function_fails(tmp_path: Path) -> None:
    """The count pin: a key alone would let a new leak hide inside an allowlisted function."""
    src = "import time\n\ndef f():\n    a = time.monotonic()\n    return a, time.time()\n"
    found = _violations(tmp_path, {IN_SCOPE: src}, [_entry(f"{IN_SCOPE}::f", reads=1)])
    assert len(found) == 1 and "reads rose 1 → 2" in found[0], found


def test_a_read_removed_from_an_allowlisted_function_fails(tmp_path: Path) -> None:
    src = "import time\n\ndef f():\n    return time.time()\n"
    found = _violations(tmp_path, {IN_SCOPE: src}, [_entry(f"{IN_SCOPE}::f", reads=2)])
    assert len(found) == 1 and "reads fell 2 → 1" in found[0], found


def test_an_entry_whose_function_reads_nothing_is_stale(tmp_path: Path) -> None:
    found = _violations(tmp_path, {IN_SCOPE: "def f():\n    return 1\n"}, [_entry(f"{IN_SCOPE}::f")])
    assert len(found) == 1 and "stale entry (a function with no read)" in found[0], found


def test_an_entry_for_a_file_outside_the_scopes_is_stale(tmp_path: Path) -> None:
    src = "import time\n\ndef f():\n    return time.time()\n"
    found = _violations(tmp_path, {"cli_helpers/x.py": src}, [_entry("cli_helpers/x.py::f")])
    assert len(found) == 1 and "outside M72's scopes" in found[0], found


def test_a_file_outside_both_scopes_is_not_scanned(tmp_path: Path) -> None:
    assert _violations(tmp_path, {"cli_helpers/x.py": "import time\n\ndef f():\n    return time.time()\n"}) == []


def test_the_prompt_scope_and_focus_learner_are_scanned(tmp_path: Path) -> None:
    src = "from datetime import datetime\n\ndef build():\n    return datetime.now()\n"
    found = _violations(tmp_path, {PROMPT: src, "proprioception/focus_learner.py": src})
    assert sorted(v.split(":", 1)[0] for v in found) == ["agents/prompt_builder.py", "proprioception/focus_learner.py"]


def test_an_untracked_file_is_not_scanned(tmp_path: Path) -> None:
    root = _tree(tmp_path, {IN_SCOPE: "def f():\n    return 1\n"})
    (root / "src" / "maxim" / "memory" / "scratch.py").write_text("import time\nx = time.time()\n", encoding="utf-8")
    assert lint.check(root).violations == []


def test_uuid4_warns_and_does_not_fail(tmp_path: Path) -> None:
    result = lint.check(_tree(tmp_path, {IN_SCOPE: "import uuid\n\ndef f():\n    return uuid.uuid4()\n"}))
    assert result.violations == []
    assert len(result.warnings) == 1 and "uuid.uuid4" in result.warnings[0]


def test_random_draws_are_not_this_lints(tmp_path: Path) -> None:
    assert _violations(tmp_path, {IN_SCOPE: "import random\n\ndef f():\n    return random.random()\n"}) == []


@pytest.mark.parametrize(
    "src",
    [
        "import datetime\n\ndef f():\n    from datetime import datetime\n    return datetime\n",
        "from time import *\n",
    ],
    ids=["two-targets", "star-import"],
)
def test_an_ambiguous_binding_fails_closed(tmp_path: Path, src: str) -> None:
    found = _violations(tmp_path, {IN_SCOPE: src})
    assert any("fails closed" in v for v in found), found


@pytest.mark.parametrize(
    ("entry", "expect"),
    [
        (_entry(f"{IN_SCOPE}::f", category="convenience"), "unknown category"),
        (_entry(f"{IN_SCOPE}::f", category="pending:"), "unknown category"),
        (_entry(f"{IN_SCOPE}::f", reason="  "), "no reason"),
        (_entry(f"{IN_SCOPE}::f", reads=0), "positive integer"),
        ({"site": f"{IN_SCOPE}::f", "category": "process", "reads": 1}, "exactly the keys"),
        (_entry(f"{IN_SCOPE}.f"), "file::function"),
    ],
)
def test_a_malformed_entry_fails(tmp_path: Path, entry: dict, expect: str) -> None:
    found = _violations(tmp_path, {IN_SCOPE: "import time\n\ndef f():\n    return time.time()\n"}, [entry])
    assert any(expect in v for v in found), found


def test_a_pending_entry_names_where_it_goes(tmp_path: Path) -> None:
    entry = _entry(f"{IN_SCOPE}::f", category="pending:PR 4 (OD-A5)")
    assert _violations(tmp_path, {IN_SCOPE: "import time\n\ndef f():\n    return time.time()\n"}, [entry]) == []


def test_a_duplicate_entry_fails(tmp_path: Path) -> None:
    src = "import time\n\ndef f():\n    return time.time()\n"
    found = _violations(tmp_path, {IN_SCOPE: src}, [_entry(f"{IN_SCOPE}::f"), _entry(f"{IN_SCOPE}::f")])
    assert any("listed twice" in v for v in found), found


def test_a_file_that_does_not_parse_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _tree(tmp_path, {IN_SCOPE: "def f(:\n"})
    assert lint.main(["--root", str(root)]) == 2
    assert "does not parse" in capsys.readouterr().err


def test_a_missing_allowlist_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _tree(tmp_path, {IN_SCOPE: "def f():\n    return 1\n"})
    (root / lint.ALLOWLIST_REL).unlink()
    assert lint.main(["--root", str(root)]) == 2
    assert "unreadable allowlist" in capsys.readouterr().err


def test_an_untracked_tree_exits_2(tmp_path: Path) -> None:
    root = _tree(tmp_path, {IN_SCOPE: "import time\nx = time.time()\n"}, track=False)
    assert lint.main(["--root", str(root)]) == 2


def test_violations_exit_1(tmp_path: Path) -> None:
    root = _tree(tmp_path, {IN_SCOPE: "import time\nx = time.time()\n"})
    assert lint.main(["--root", str(root)]) == 1


def test_this_checkout_is_clean() -> None:
    """Positive control: every in-scope read on the real tree is placed, with its exact count."""
    result = lint.check(lint.REPO_ROOT)
    assert result.violations == [], "\n".join(result.violations)
    assert result.refs, "the real scan found no read at all: the scan itself is broken"
