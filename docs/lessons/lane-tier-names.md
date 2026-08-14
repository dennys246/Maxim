# Lane tier names

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] Lane tier names:** The canonical tier names are `"large"`, `"medium"`, `"small"`. The old names `"infer"`, `"review"`, `"record"` have been fully removed. Do not re-introduce them. Regression guard: CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" step) — quoted `"infer"`/`"review"`/`"record"` literals in [src/maxim/runtime/lane_models.py](src/maxim/runtime/lane_models.py) fail CI.
