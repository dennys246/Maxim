# NAc class name

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] NAc class name:** The class is `NAc` (in `decisions/nac.py`), NOT `NucleusAccumbens`. Old code may reference the wrong name — always grep for `NucleusAccumbens` after touching NAc-related code. Regression guard: CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" step) — `grep -rn "NucleusAccumbens" src/maxim/` must return zero matches; class definition lives in [src/maxim/decisions/nac.py](src/maxim/decisions/nac.py).
