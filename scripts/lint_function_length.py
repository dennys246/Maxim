#!/usr/bin/env python3
"""Ratchet lint: the three god-functions may only get SHORTER, never longer.

The 2026-08-27 score card's standing Maintainability complaint: "CI does not bound any
function's length" — and all three of the repo's largest functions had GROWN since the
prior cut. This is the missing mechanization. It pins each named function at its current
line span and fails CI if any EXCEEDS its pin (a ratchet, like
``lint_atomic_io_ratchet.py``): decomposition lowers the pin, regrowth fails.

It also prints every function's current span each run (the count lives in CI output, not
in a number that rots in prose), and fails LOUD if a pinned function cannot be found —
a rename/move must update the baseline deliberately, not silently disable the guard.

Baselines are the SPAN (inclusive line count) measured on the commit that introduced this
lint. To lower one after a real extraction: run this script, copy the printed span in.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# (relative path, function name) -> max allowed inclusive line span.
# Ratchet-DOWN only: never raise a number here to make a growing function pass.
_BASELINES: dict[tuple[str, str], int] = {
    ("src/maxim/runtime/agent_loop.py", "run_agentic_loop"): 3488,
    ("src/maxim/simulation/orchestrator.py", "start_simulation_mode"): 3324,
    ("src/maxim/cli.py", "_main_impl"): 1747,
}


def _span_of(path: Path, func_name: str) -> int | None:
    """Largest inclusive line span of any def named ``func_name`` in ``path``.

    Takes the max across matches so a god-function is measured even if a small
    same-named nested helper exists; returns None if no such def is found.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    spans = [
        node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == func_name
        and node.end_lineno is not None
    ]
    return max(spans) if spans else None


def check() -> int:
    failures: list[str] = []
    print("function-length ratchet (span = inclusive line count; ratchet-down only):")
    for (rel, name), pinned in sorted(_BASELINES.items()):
        path = _REPO / rel
        if not path.exists():
            failures.append(f"MISSING FILE: {rel} (pinned function {name!r})")
            continue
        span = _span_of(path, name)
        if span is None:
            failures.append(
                f"NOT FOUND: {name!r} in {rel} — if it was renamed/moved, update the "
                f"baseline in scripts/lint_function_length.py deliberately."
            )
            continue
        status = "OK" if span <= pinned else "GREW"
        slack = pinned - span
        note = f" (pin {pinned}; {slack:+d} — lower the pin)" if 0 < slack else f" (pin {pinned})"
        print(f"  [{status}] {rel}::{name} = {span}{note}")
        if span > pinned:
            failures.append(
                f"{rel}::{name} grew to {span} lines (pin {pinned}). This function is a "
                f"known god-function; add code by EXTRACTING, not growing it. See "
                f"docs/plans/god_function_decomposition.md."
            )

    if failures:
        print("\nfunction-length ratchet: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("function-length ratchet: clean")
    return 0


if __name__ == "__main__":
    sys.exit(check())
