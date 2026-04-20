# Game NPC integration — FOLDED into agent_factory_canonicalization.md

**Status:** Folded (2026-04-18). Content merged into [agent_factory_canonicalization.md](agent_factory_canonicalization.md) as **Wave G (Game/External Host)**.

## Why folded

The execution + architecture review (2026-04-18) found that game_npc_integration Stage 1 is the same deliverable as agent_factory_canonicalization Stage F1 — both extend `AgentFactory` to produce fully-capable agents via `build_bio_stack()` + `build_executor()`. Shipping them as separate plans means touching `agent_factory.py` twice with potentially conflicting Executor lifetime decisions (Z1/Z2/Z3).

The G-wave adds ~940 LOC to the F-wave's ~1500-2500 LOC. Same architectural arc, one design pass, no conflicting decisions.

## What moved where

| Original stage | New location |
|---|---|
| Stage 1 — Wire Executor + bio-pipeline | Wave G Stage G1 |
| Stage 2 — HostContext protocol | Wave G Stage G2 |
| Stage 3 — Async tool dispatch | Wave G Stage G4 |
| Stage 4 — Emotional state readout | Wave G Stage G3 (moved up — simpler) |
| Stage 5 — Memory backend protocol | Wave G Stage G5 |

See [agent_factory_canonicalization.md](agent_factory_canonicalization.md) for the full plan.
