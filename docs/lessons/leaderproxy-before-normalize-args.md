# Startup ordering in cli.py

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] Startup ordering in cli.py:** The LeaderProxy MUST start BEFORE `_normalize_args()` because arg normalization can trigger heavy CUDA imports (5-15s on GPU systems). Peers polling for the proxy during restart will time out if the proxy starts after these imports. Regression guard: [src/maxim/cli.py::main](src/maxim/cli.py) — LeaderProxy startup must precede `_normalize_args()` call; ordering enforced by code structure at the top of `main()`.
