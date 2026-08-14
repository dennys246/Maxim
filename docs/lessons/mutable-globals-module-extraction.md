# Mutable globals + module extraction

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] Mutable globals + module extraction:** When extracting module-level mutable globals (like `_active_spawner`) into a new file, do NOT re-import them by name (`from new_module import _active_spawner`). Python binds by value at import time — assignments in the importing module diverge from the source. Use module reference instead: `import new_module as _mod; _mod._active_spawner = value`. Functions are safe to re-import (they close over their own module's namespace). Regression guard: pattern-lesson, no test enforces; rely on reviewer attention during module extraction PRs.
