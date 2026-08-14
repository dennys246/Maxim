#!/usr/bin/env python3
"""Every harness that spawns `maxim` sub-sims must run the provenance guard.

The Exp 42b retraction (CLAUDE.md, first lesson): a 40-sub-sim behavioural
re-validation was retracted because the sub-sims imported a DIFFERENT checkout
than the one under test — silently, with authoritative-looking JSONL.
``scripts/_provenance.py::assert_repo_interpreter`` exists to make that
impossible, but a guard only guards the harnesses that call it.

This lint finds ``scripts/**/*.py`` that spawn the maxim runtime (subprocess +
a maxim-invocation pattern) and requires each to reference
``assert_repo_interpreter``. False positives (a script whose "maxim" match is
not a sub-sim spawn) opt out with a line containing ``# provenance-exempt:``
followed by the reason.

This lint catches FORGETTING, not evasion (house convention for heuristic
lints): a docstring mention of ``assert_repo_interpreter`` counts as
compliance, one exempt marker exempts the whole file, and ``shell=True``
string spawns escape the regex. It is a forcing function for the honest
author, not a security boundary.

Exits: 0 clean; 1 violations (stderr).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# A spawn of the maxim runtime, in list-arg or console-script form:
#   [sys.executable, "-m", "maxim", ...]   /  ["maxim", "--sim", ...]
#   shutil.which("maxim") / .venv/bin/maxim used as argv[0]
_MAXIM_SPAWN = re.compile(
    r"""(
        -m['"],\s*['"]maxim['"]            # [..., "-m", "maxim", ...]
      | ['"]maxim['"],\s*['"]--            # ["maxim", "--sim"/"--goal"...]
      | which\(['"]maxim['"]\)             # shutil.which("maxim")
      | bin/maxim['"]                      # .venv/bin/maxim path literal
    )""",
    re.VERBOSE,
)


def main() -> int:
    failures: list[str] = []
    for path in sorted((REPO_ROOT / "scripts").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        text = path.read_text(errors="replace")
        if "subprocess" not in text or not _MAXIM_SPAWN.search(text):
            continue
        if "assert_repo_interpreter" in text or "# provenance-exempt:" in text:
            continue
        failures.append(
            f"{rel}: spawns maxim sub-sims without the provenance preflight — "
            "call scripts/_provenance.py::assert_repo_interpreter before the "
            "first spawn (exit 3 on mismatch) and stamp executed_code_provenance "
            "into every run record, or mark a false positive with "
            "'# provenance-exempt: <reason>' (Exp 42b lesson)"
        )

    if failures:
        print("harness-provenance lint FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("harness-provenance lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
