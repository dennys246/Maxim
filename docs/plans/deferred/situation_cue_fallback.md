# Situation cue — rebuilding from the records' own situation (deferred)

> **DEFERRED 2026-09-25 (owner).** Memory-strength 2S-d
> ([memory_2s_d_situation_cue.md](../memory_2s_d_situation_cue.md)) nominates candidates through the
> ATL concepts' `memory_refs['hippocampus']`, a DERIVED, lossy index. The durable key is
> `EpisodicMemory.situation`.

**Limit:** a memory whose link was lost is never cued, even though its own `situation` matches:
- a compressed ATL concept (`CompressedSemantic`) holds no `memory_refs`, so consolidation drops every
  link on it and new ones cannot form (until #816 makes compression reversible);
- `Concept.MAX_REFS_PER_LAYER` evicts in insertion order, which after a reload is uuid order, not age.
- when the ATL evicts a substrate concept, `activate_substrate_node` later re-creates it with the SAME
  id and no refs, so every link to it is gone.
- a COMPRESSED Hippocampus record (`CompressedMemory`) carries no `situation`, so even with its link
  intact it can never qualify.

**Would lift it:** rebuilding the refs from the records' `situation` on load, or a Hippocampus-side
`(modality, cluster) → record ids` index derived from `situation` (rebuilt on load, never persisted).

**Revive when:** a measured run shows cue candidates missing for memories whose `situation` matches
(compare the cue's candidate count against a scan of `situation` on a saved store), OR a situation
concept is compressed while its memories are still stored (ConceptExtractor already warns once), OR
2S-e's consumer needs completion to be complete rather than best-effort.

## Also deferred: cueing on the llm-primary and real-hardware paths

**Limit:** the cue fires only inside `propose_via_substrate`. llm-primary and real-hardware (Reachy)
passes take their situation from `_attach_live_situation`, so their captures carry a situation that
nothing cues.
**Revive when:** 2S-e's consumer is to run on an llm-primary or hardware path, OR a measurement of
situation recall is planned on one.
