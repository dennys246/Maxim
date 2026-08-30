"""Guard for the fail-loud Stage-2 measurement tool (scripts/fail_loud_stage2.py).

THE BUG THIS GUARDS (found 2026-08-30 while running Stage 2)
------------------------------------------------------------
``log_swallowed_exception()`` emits its event via
``extra={"event": "swallowed_exception", "data": {...}}``. Reading the call
site and writing a parser against that shape is the obvious move, and it is
wrong: the ``MAXIM_LOG_FILE`` handler formats with ``StructuredFormatter``,
which calls ``LogRecord.to_compact()`` — and that keys the event as ``"e"``
and FLATTENS ``data`` to the top level. A parser matching only ``"event"``
reads a real capture full of firings and reports **zero**, which is
indistinguishable from "nothing fired".

That is the same failure class PR #487's review already caught once at the
other end of the pipe (the `extra` pair was missing, so nothing was written).
Stage 2's entire output is a count of firings; a parser that silently reads
zero makes the baseline a lie and makes the extraction gate that cites it
vacuous.

``test_reads_compact_shape`` fails if the compact branch is reverted.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "fail_loud_stage2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fail_loud_stage2", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stage2():
    return _load_module()


def _write(tmp_path: Path, name: str, records: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_reads_compact_shape(stage2, tmp_path):
    """The shape the JSONL handler actually writes (StructuredFormatter/to_compact)."""
    capture = _write(
        tmp_path,
        "compact.jsonl",
        [
            {"t": 1.0, "l": "I", "s": "maxim", "e": "log"},
            {
                "t": 2.0,
                "l": "W",
                "s": "maxim",
                "e": "swallowed_exception",
                "site": "nac.py:credit_node:2382",
                "exc_type": "TypeError",
                "exc": "boom",
            },
        ],
    )
    firings, meta = stage2._parse_capture(capture)
    assert len(firings) == 1, "compact-shape firing must be detected"
    assert firings[0]["basename"] == "nac.py"
    assert firings[0]["function"] == "credit_node"
    assert firings[0]["exc_type"] == "TypeError"
    assert meta["lines"] == 2
    assert meta["unparsable_lines"] == 0


def test_reads_verbose_shape(stage2, tmp_path):
    """LogRecord.to_verbose() — the other real shape, kept working."""
    capture = _write(
        tmp_path,
        "verbose.jsonl",
        [
            {
                "event": "swallowed_exception",
                "data": {"site": "body.py:tick:120", "exc_type": "KeyError", "exc": "k"},
            }
        ],
    )
    firings, _meta = stage2._parse_capture(capture)
    assert len(firings) == 1
    assert firings[0]["basename"] == "body.py"
    assert firings[0]["exc_type"] == "KeyError"


def test_unparsable_lines_are_counted_not_hidden(stage2, tmp_path):
    """A capture we cannot fully parse must say so, not quietly under-report."""
    path = tmp_path / "mixed.jsonl"
    path.write_text('{"e":"log"}\nnot json at all\n', encoding="utf-8")
    _firings, meta = stage2._parse_capture(path)
    assert meta["unparsable_lines"] == 1


def test_check_fails_on_a_new_pair(stage2, tmp_path):
    """The extraction gate's whole job: a new (file, exception) pair fails."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"fired_pairs": [{"basename": "nac.py", "exc_type": "TypeError", "count": 1}]}),
        encoding="utf-8",
    )
    candidate = _write(
        tmp_path,
        "cand.jsonl",
        [
            {"e": "swallowed_exception", "site": "nac.py:f:1", "exc_type": "TypeError"},
            {"e": "swallowed_exception", "site": "agent_loop.py:g:2", "exc_type": "ValueError"},
        ],
    )
    rc = stage2.main(["check", "--capture", f"m={candidate}", "--baseline", str(baseline)])
    assert rc == 1, "a new (file, exc) pair must fail the gate"


def test_check_tolerates_renamed_functions_and_moved_lines(stage2, tmp_path):
    """The reason the key is (basename, exc_type) and not the raw site string.

    A mechanical extraction renames the enclosing function and moves every
    line. That must NOT read as a new firing, or the gate reports 100% noise
    on exactly the refactor it exists to protect.
    """
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"fired_pairs": [{"basename": "agent_loop.py", "exc_type": "TypeError", "count": 1}]}),
        encoding="utf-8",
    )
    moved = _write(
        tmp_path,
        "moved.jsonl",
        [
            {
                "e": "swallowed_exception",
                "site": "agent_loop.py:_loop_perception:87",  # was run_agentic_loop:2410
                "exc_type": "TypeError",
            }
        ],
    )
    rc = stage2.main(["check", "--capture", f"m={moved}", "--baseline", str(baseline)])
    assert rc == 0, "a moved/renamed site with the same (file, exc) must pass"


def test_check_refuses_when_baseline_is_missing(stage2, tmp_path):
    """A gate must never pass by citing an artifact that does not exist."""
    capture = _write(tmp_path, "c.jsonl", [{"e": "log"}])
    rc = stage2.main(["check", "--capture", f"m={capture}", "--baseline", str(tmp_path / "nope.json")])
    assert rc == 2


def test_inventory_finds_only_zero_arg_sites(stage2, tmp_path):
    """The Stage-1 form is the zero-arg call; the legacy explicit form is not."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "m.py").write_text(
        "from maxim.utils.logging import log_swallowed_exception\n"
        "def outer():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:\n"
        "        log_swallowed_exception()\n"
        "def legacy(e):\n"
        "    log_swallowed_exception(e, operation='x')\n",
        encoding="utf-8",
    )
    sites = stage2.inventory_sites(pkg)
    assert len(sites) == 1
    assert sites[0]["function"] == "outer"


def test_repo_inventory_is_nonempty(stage2):
    """If this hits zero, the helper was renamed and Stage 2 is measuring nothing."""
    assert len(stage2.inventory_sites()) > 0
