# Affordance names are encoded through a SEPARATE `LinguisticEncoder` from the percept encoder

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

- **[engineering] Affordance names are encoded through a SEPARATE `LinguisticEncoder` from the percept encoder.** Both share the same EC/ATL/NAc backing, but the affordance encoder uses `AffordanceDecompositionStrategy` (splits on underscores: `fire_breath` → `["fire breath", "fire", "breath"]`) while the percept encoder uses `SpaCyNounChunkStrategy`. The module-level singleton `AFFORDANCE_STRATEGY` in `similarity/decomposer.py` is used by bio_enrichment, discovery tools, and trigger for annotation lookups. `ImaginationTrigger._aff_encoder` and the standalone `encode_entity_affordances()` are constructed via the shared `_make_aff_encoder()` factory. Regression guard: [src/maxim/similarity/decomposer.py](src/maxim/similarity/decomposer.py) (`AFFORDANCE_STRATEGY` singleton) + [src/maxim/imagination/trigger.py](src/maxim/imagination/trigger.py) (`_make_aff_encoder` factory).
