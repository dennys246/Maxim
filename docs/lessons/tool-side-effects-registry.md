# `ToolOutput.side_effects` is the typed channel for bio-pipeline signals

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

- **[engineering] `ToolOutput.side_effects` is the typed channel for bio-pipeline signals.** A `dict[str, Any] | None` field on `ToolOutput`. The append-only registry of well-known keys lives at [docs/user/tool_side_effects.md](docs/user/tool_side_effects.md) — that page is the authoritative source third-party tool authors read. Current keys: `embodiment_failures` (list of SEM failure event dicts), `entity_acquired` / `entity_released` (entity name str — Mechanism B contact sensation), `affordance_blocked` (precondition-fail metadata, informational — no internal consumer, used by replay tooling and external observers), `drive_potential_diff` (float signed value-progress toward comfort; `tool_dispatch` uses its SIGN as the ±1 cluster reward — see the motor-credit invariant above). Add new keys via PR that updates the registry table AND wires the consumer (or marks the key `informational` per the doc's rule #4). Do NOT hijack `metadata` (caller-facing extras) or `output` (main result) — those serve different audiences and collapsing them silently couples the tools layer to bio concepts. Regression guard: [src/maxim/tools/base.py::ToolOutput](src/maxim/tools/base.py) (dataclass + docstring) + [docs/user/tool_side_effects.md](docs/user/tool_side_effects.md) (key registry).
