#!/usr/bin/env python3
"""Stage 4 of measurement_path_fail_loud.md — the no-silent-swallows lock.

Two checks, both comment-tolerant (the PR #487 review found the comment-blind
pattern missed 10 ``pass  # best-effort`` swallows):

1. **Zero-total over the measurement path.** The 16 scoped files from the
   plan's inventory were purged in Stage 1 (48 sites instrumented via
   ``log_swallowed_exception``); this lock keeps them at zero bare
   ``except Exception:`` → ``pass``/``continue`` swallows forever.

2. **No-new-swallows, diff-scoped, repo-wide.** Every other ``src/maxim/``
   file is grandfathered at its origin/main swallow COUNT; a branch may not
   increase any file's count. (Count-based, so moving code within a file
   stays free; adding a swallow anywhere fails.) The motivating incident:
   the SCN drive path was dead for months behind exactly one
   bare-except-swallowed TypeError.

Exits: 0 clean; 1 violations (details on stderr); 2 git/repo state error.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The plan's scope table (measurement_path_fail_loud.md §Scope).
MEASUREMENT_PATH = [
    "src/maxim/decisions/nac.py",
    "src/maxim/decisions/temporal_credit.py",
    "src/maxim/runtime/tool_dispatch.py",
    "src/maxim/runtime/bio_integration.py",
    "src/maxim/runtime/agent_loop.py",
    "src/maxim/similarity/encoder.py",
    "src/maxim/similarity/ec.py",
    "src/maxim/bridges/tool_pain_bridge.py",
    "src/maxim/proprioception/pain_bus.py",
    "src/maxim/embodiment/body.py",
    "src/maxim/embodiment/tool_bridge.py",
    "src/maxim/simulation/sim_logger.py",
    "src/maxim/memory/hippocampus.py",
    "src/maxim/memory/hippocampus_consolidation.py",
    "src/maxim/integration/memory_hub.py",
    "src/maxim/decisions/causal_link.py",
]

_EXCEPT_RE = re.compile(r"except (Exception|BaseException)(\s+as\s+\w+)?\s*:\s*(#.*)?$")
_SWALLOW_RE = re.compile(r"^\s+(pass|continue)\s*(#.*)?$")


def swallow_hits(text: str) -> list[int]:
    """1-indexed line numbers of bare-swallow ``pass``/``continue`` lines."""
    lines = text.splitlines()
    hits: list[int] = []
    for i, line in enumerate(lines):
        if _EXCEPT_RE.search(line) and i + 1 < len(lines) and _SWALLOW_RE.match(lines[i + 1]):
            hits.append(i + 2)
    return hits


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def _base_ref() -> str:
    for ref in ("origin/main", "main"):
        try:
            return _git("merge-base", ref, "HEAD").strip()
        except RuntimeError:
            continue
    raise RuntimeError("no origin/main or main to diff against")


def main() -> int:
    failures: list[str] = []

    # Check 1 — zero-total over the measurement path.
    for rel in MEASUREMENT_PATH:
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"{rel}: scoped file missing — update the lint's scope table")
            continue
        for ln in swallow_hits(path.read_text()):
            failures.append(
                f"{rel}:{ln}: silent swallow in the measurement path — "
                "use log_swallowed_exception() or narrow/propagate "
                "(measurement_path_fail_loud.md policy)"
            )

    # Check 2 — diff-scoped no-new-swallows across src/maxim/.
    try:
        base = _base_ref()
        changed = [
            f
            for f in _git("diff", "--name-only", base, "HEAD", "--", "src/maxim/").split()
            if f.endswith(".py") and f not in MEASUREMENT_PATH
        ]
        for rel in changed:
            path = REPO_ROOT / rel
            new_count = len(swallow_hits(path.read_text())) if path.exists() else 0
            try:
                old_text = _git("show", f"{base}:{rel}")
            except RuntimeError:
                old_text = ""  # new file — grandfathered at zero
            old_count = len(swallow_hits(old_text))
            if new_count > old_count:
                failures.append(
                    f"{rel}: swallow count rose {old_count} → {new_count} on this branch — "
                    "no NEW bare `except Exception: pass/continue`; "
                    "use log_swallowed_exception() or narrow the exception type"
                )
    except RuntimeError as e:
        print(f"lint_no_silent_swallows: git error: {e}", file=sys.stderr)
        return 2

    if failures:
        print("no-silent-swallows lint FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("no-silent-swallows lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
