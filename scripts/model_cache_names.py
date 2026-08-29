#!/usr/bin/env python3
"""The sentence-transformers model names the nightly model-cache lane must warm — DERIVED, not hand-kept.

The lane (`.github/workflows/test.yml` → `model-cache-tests`) runs offline against a warmed
HuggingFace cache. Its warm list was a comment-maintained tuple that omitted `all-MiniLM-L6-v2`
(loaded by `tests/substrate/baselines/test_baselines.py` and `memory_agent.py`), so every
scheduled run since 2026-08-21 failed on `LocalEntryNotFoundError` — the lane built to stop
vacuous guards was itself red under a stale list (score card 2026-08-27, Test/CI truthfulness).

This script is the single source: it scans `src/maxim/` and every test file carrying the
`requires_model_cache` marker (plus the helpers they import from `tests/`) for
sentence-transformers model-name literals and prints one per line. CI warms exactly this
output; `tests/unit/test_model_cache_names.py` pins that the production defaults and the
marked tests' literals are in it.

Catches forgetting, not evasion: a model name built at runtime (f-string, config file) is
not a literal and will not be found — name it in a literal next to the loader.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_LITERAL = re.compile(
    r"""["'](?:sentence-transformers/)?((?:all|paraphrase|clip|multi-qa|msmarco)-[A-Za-z0-9.-]+)["']"""
)
MARKER = "requires_model_cache"


def _literals(path: Path) -> set[str]:
    return set(_MODEL_LITERAL.findall(path.read_text(errors="replace")))


def marked_test_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    return sorted(
        p
        for p in (repo_root / "tests").rglob("*.py")
        if MARKER in p.read_text(errors="replace") and p.name != "conftest.py"
    )


def _local_test_imports(path: Path, repo_root: Path) -> list[Path]:
    """Helper modules under tests/ that a marked test imports (e.g. baselines/embedding_baseline.py)."""
    out = []
    text = path.read_text(errors="replace")
    for m in re.finditer(r"^\s*from\s+(tests(?:\.[\w]+)+)\s+import|^\s*import\s+(tests(?:\.[\w]+)+)", text, re.M):
        mod = m.group(1) or m.group(2)
        cand = repo_root / (mod.replace(".", "/") + ".py")
        if cand.exists():
            out.append(cand)
    for m in re.finditer(r"^\s*from\s+\.(\w+)\s+import", text, re.M):
        cand = path.parent / f"{m.group(1)}.py"
        if cand.exists():
            out.append(cand)
    return out


def model_names(repo_root: Path = REPO_ROOT) -> list[str]:
    names: set[str] = set()
    for p in sorted((repo_root / "src" / "maxim").rglob("*.py")):
        names |= _literals(p)
    for p in marked_test_files(repo_root):
        names |= _literals(p)
        for helper in _local_test_imports(p, repo_root):
            names |= _literals(helper)
    return sorted(names)


def main() -> int:
    for name in model_names():
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
