# Dead code accumulates silently

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] Dead code accumulates silently:** Before publishing or after major refactors, grep for orphan modules: `.py` files whose basename doesn't appear in any `import` statement. We found 15 dead modules (~8,500 LOC) shipping in the wheel. Regression guard: process invariant — periodic grep before publish; no automated test enforces.
