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

This lint catches FORGETTING, not evasion: known-unmatched shapes include a
comment line between the `except` and the `pass`, `except (X, Exception):`,
and same-line `except Exception: pass` (zero instances of any exist in
src/maxim/ today — verified 2026-08-13). Extend `swallow_hits` if one of
these ever appears in review.

Exits: 0 clean; 1 violations (details on stderr); 2 unexpected error.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_git import GitUnavailable, base_ref, count_ratchet, must_not_skip  # noqa: E402

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

    # Check 0 — PRINT THE TOTALS every run. The 2026-08-27 score card's
    # Documentation-honesty condition is that this lint and the atomic_io
    # ratchet print their totals in CI so CLAUDE.md can cite the OUTPUT
    # instead of a number that rots in the file (added 2026-08-29).
    repo_total = 0
    repo_files = 0
    for path in sorted((REPO_ROOT / "src" / "maxim").rglob("*.py")):
        n = len(swallow_hits(path.read_text(errors="replace")))
        if n:
            repo_total += n
            repo_files += 1
    print(
        f"no-silent-swallows: {repo_total} bare `except Exception: pass/continue` site(s) in {repo_files} "
        f"file(s) across src/maxim/ ({len(MEASUREMENT_PATH)} measurement-path files held at zero; "
        "every other file grandfathered at its origin/main count)"
    )

    # Check 2 — diff-scoped no-new-swallows across src/maxim/, on the shared
    # ratchet (scripts/_lint_git.py). A shallow CI clone can lack a merge-base
    # entirely; check 1 needs no git and its results must never be discarded
    # for a git failure, so a missing base ref SKIPS check 2 with an INFO.
    # (The pre-fold version returned 2 here, which made every PR red in CI and
    # threw away check 1's findings unprinted — caught by the #508 review.)
    try:
        base = base_ref(REPO_ROOT)
    except GitUnavailable as e:
        if must_not_skip(str(e)):
            return 2
        print(f"INFO: no base ref available; skipping diff-scoped check 2 ({e})")
        base = None
    if base is not None:
        try:
            failures.extend(
                count_ratchet(
                    REPO_ROOT,
                    base,
                    "src/maxim/",
                    swallow_hits,
                    exclude=frozenset(MEASUREMENT_PATH),
                    what="bare-swallow count",
                    advice=(
                        "no NEW bare `except Exception: pass/continue`; use log_swallowed_exception() "
                        "or narrow the exception type"
                    ),
                )
            )
        except GitUnavailable as e:
            print(f"INFO: diff-scoped check 2 skipped mid-run ({e})")

    if failures:
        print("no-silent-swallows lint FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("no-silent-swallows lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
