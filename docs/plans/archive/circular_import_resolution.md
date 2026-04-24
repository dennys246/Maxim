# Circular Import Resolution Plan

**Status:** RESOLVED. ToolErrorKind lives in tools/base.py, circular import chain broken.
**Scope:** Pre-1.0 tech debt — prevents test isolation failures and future refactoring pain
**Priority:** Ship before 1.0 (user decision 2026-04-20)
**Estimated LOC:** ~80 (mostly moving imports, not new code)

---

## Problem

A circular import chain prevents any module under `maxim.tools` from being imported in isolation:

```
tools/__init__.py → tools/base.py → agents/bus.py (ToolErrorKind)
→ agents/__init__.py → agents/exec_agent.py → default_network/messages.py
→ default_network/__init__.py → default_network/gate.py
→ runtime/gating.py → runtime/__init__.py → runtime/bootstrap.py
→ runtime/executor.py → tools/base.py  ← CYCLE
```

**Consequences today:**
- Tests importing from `maxim.tools.*` cannot run standalone — they only work when the full package is pre-loaded via other imports in the test session
- The `tests/experiments/conftest.py` workaround (`import maxim.agents.bus`) is a band-aid
- Any new test file or script that imports a tool in isolation hits `ImportError`

**Consequences at scale:**
- Refactoring any of the 4 packages (`tools`, `agents`, `default_network`, `runtime`) requires understanding the full cycle
- IDE static analysis tools (mypy, pyright, pylint) may produce incorrect results
- New contributors (human or AI) will hit this on their first test run
- The cycle will grow as more cross-package features ship

## Root Cause Analysis

The cycle exists because **5 `__init__.py` files eagerly re-export symbols** that drag in the entire package tree at import time. The problematic imports:

| File | Import | Actually needed at load time? |
|------|--------|-------------------------------|
| `tools/base.py:7` | `from maxim.agents.bus import ToolErrorKind` | **NO** — only used as type annotation |
| `agents/__init__.py:34` | `from maxim.agents.exec_agent import ExecAgent` | **NO** — convenience re-export |
| `agents/exec_agent.py:32` | `from maxim.default_network.messages import FilteredPercept` | **YES** — used in `__init__()` |
| `default_network/__init__.py:32-36` | `from maxim.default_network.gate import ...` | **NO** — convenience re-export |
| `runtime/__init__.py:11` | `from maxim.runtime.bootstrap import ...` | **NO** — convenience re-export |

**Key insight:** Only ONE import in the chain is genuinely needed at load time (`FilteredPercept` in `exec_agent.__init__()`). Everything else is either a type annotation or a convenience re-export.

## Fix Strategy

Break the cycle at its **cheapest point** — where the import is used only as a type annotation. This requires changing exactly ONE line and produces zero runtime behavior change.

### Stage 1: Break the cycle (1 file, 3 lines)

**File:** `src/maxim/tools/base.py`

```python
# BEFORE (line 7):
from maxim.agents.bus import ToolErrorKind

# AFTER:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maxim.agents.bus import ToolErrorKind
```

Then change the runtime usage of `ToolErrorKind` (if any beyond type annotations) to use a string literal or lazy import. Audit required.

**Why this is the right break point:**
- `ToolErrorKind` is an enum used as a type on `ToolOutput.error_kind: ToolErrorKind | None`
- At runtime, the field is set to an enum VALUE (e.g., `ToolErrorKind.TIMEOUT`), which happens inside tool execution — AFTER all modules are loaded
- The type annotation itself only needs the symbol at type-checking time (mypy/pyright), not runtime

**Risk:** If any code does `isinstance(x, ToolErrorKind)` or references the enum at module-load time in `tools/base.py`, this breaks. Audit first.

### Stage 2: Audit ToolErrorKind runtime usage (validation)

```bash
# Find all runtime uses of ToolErrorKind in tools/base.py
grep -n "ToolErrorKind" src/maxim/tools/base.py
```

Expected: it appears in a dataclass field annotation and nowhere else at module level. If it's used in a default value or class body (not just type hints), we need the deferred approach from Stage 3 instead.

### Stage 3: Fallback — lazy import in `tools/base.py` (if Stage 1 insufficient)

If `ToolErrorKind` is needed at runtime in `base.py` (e.g., in a class method body), use a function-level import:

```python
# In tools/base.py — remove the top-level import entirely.
# Where ToolErrorKind is actually used at runtime:
def _classify_error(self, error: Exception) -> "ToolErrorKind":
    from maxim.agents.bus import ToolErrorKind
    if isinstance(error, TimeoutError):
        return ToolErrorKind.TIMEOUT
    ...
```

### Stage 4: Harden — prevent `__init__.py` re-export creep

Add a CI check (or pre-commit hook) that catches new eager imports in `__init__.py` files that create cross-package dependencies:

```bash
# .github/workflows/test.yml addition:
# Verify tools can be imported standalone (cycle detection)
- run: python -c "from maxim.tools.narrative import ThinkTool; print('OK')"
```

This serves as a canary — if anyone adds an eager import that recreates the cycle, CI fails.

### Stage 5 (optional): Clean up other re-exports

Once Stage 1 breaks the cycle, these are non-blocking but reduce import latency:

| File | Change | Impact |
|------|--------|--------|
| `agents/__init__.py:34` | Move `ExecAgent` to TYPE_CHECKING or remove re-export | Faster `import maxim.agents` |
| `default_network/__init__.py:32-36` | Lazy-load gate exports via `__getattr__` | Faster DN import |
| `runtime/__init__.py:11` | Lazy-load bootstrap exports | Faster runtime import |

These are independent improvements. Each can ship separately.

## Testing

After Stage 1:
```bash
# This must work (currently fails):
python -c "from maxim.tools.narrative import ThinkTool; print('OK')"

# This must still work:
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# Type checking still passes:
mypy src/maxim/tools/base.py --ignore-missing-imports
```

After Stage 4:
```bash
# CI canary:
python -c "from maxim.tools.narrative import ThinkTool; print('OK')"
python -c "from maxim.tools.base import Tool, ToolResult; print('OK')"
```

## Sequencing

| Stage | Effort | Risk | Ship when |
|-------|--------|------|-----------|
| 1 | 5 min | Very low (type annotation move) | Immediately |
| 2 | 10 min | None (read-only audit) | With Stage 1 |
| 3 | 15 min | Low (only if Stage 1 insufficient) | Only if needed |
| 4 | 10 min | None (CI-only) | After Stage 1 validates |
| 5 | 30 min | Low (each independent) | Anytime, not blocking |

**Total critical path: Stage 1+2 = 15 minutes.** The cycle breaks with a 3-line change if the audit confirms `ToolErrorKind` is annotation-only in `base.py`.

## Why not bigger refactors?

Alternative approaches considered and rejected:

1. **Lazy `__init__.py` via `__getattr__`** — Correct but high-risk: changes import behavior for the entire package. Every `from maxim.tools import X` call path changes semantics. Not worth the risk for a pre-1.0 fix.

2. **Move `ToolErrorKind` out of `agents/bus.py`** — Moving the enum to a standalone `tools/types.py` would also break the cycle, but violates the principle of co-locating error kinds with the bus that dispatches them. The enum belongs with the bus.

3. **Remove all `__init__.py` re-exports** — Nuclear option. Would break every existing `from maxim.agents import X` usage across the codebase + downstream users. Not appropriate.

The TYPE_CHECKING approach is the standard Python solution for this class of problem. It's well-understood, zero-risk for runtime behavior, and tooling (mypy, pyright) handles it natively.
