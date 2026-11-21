# Foreign advice layer — advice held apart from the agent's own values

> **PROPOSED 2026-11-12 — plan stub only, no code. A shared mechanism, split out by owner decision on 2026-11-12.**
> It is the one foreign-advice layer for two consumers:
> - [social_referencing.md](social_referencing.md), where the source is an Oasis mirror;
> - [hybrid_arbitration.md](hybrid_arbitration.md) HA-O3's L→S cells, where the source is the LLM.
>
> This plan owns the layer. Neither consumer builds its own, and neither waits on the other's gate:
> social_referencing keeps its public_oasis Phase 0 gate for the mirror, but not for the layer. **No `src/` before
> the fence lifts** (the fence lifts when 1.3.6's decomposition slices land, which is the point at which T0 can pass).
> The layer reads in `NAc.recommend_action`'s selection path, which E1's mechanism uses, so this line has no `src/`
> before the 1.4.0 cut either. It needs its own two-lens review before build.

## What it is

A table of advice entries `(source, situation key, tool) → bias`. It adds to `recommend_action`'s scores, weighted by
`trust(source) × (1 − own confidence)`, and decays. **Credit never writes it.** This is the design in
social_referencing.md §2 and §6, lifted out so that both lines can use one copy.

## Source-specific rules (owner decisions 2026-11-12, from the hybrid line's final review)

For an **LLM** source:
- **Never promoted** into the agent's own values. social_referencing §6's sleep-consolidation promotion is OFF.
- **Never persisted.** The table is runtime-only, like `_noted_active_clusters`: it is excluded from `NAc.dump()` and
  `load()`, from substrate and Hivemind/Oasis bundles, and from peer exchange.
- **Trust is a frozen constant** fixed in each prereg, never learned from outcomes. There is a per-episode cap on
  total advice weight.
- **Zero weight on danger-selected and protective actions** (hybrid_arbitration §A.1d's safety floor).
- **An action the advice changed is stamped `substrate_advised`**, never as substrate-chosen.
  - "Changed" means the outcome with advice differs from the outcome without it, and that includes gate-crossing: a
    recommendation that is `None` without advice, but whose advised tool clears `min_confidence` with it.
  - It is credited under the LLM rule (drive relief only).
  - Without this, advice reaches the agent's own values through dispatch credit, which is the echo path.
- **Decay runs on the experience clock** ([experience_clock_decay.md](experience_clock_decay.md)), not on loop
  passes. That way advice influence lasts the same in every arm, whatever the LLM latency or host load.
  - That plan is still PROPOSED, so it is a prerequisite.
  - Until it lands, the interim clock is the decision count, frozen in the prereg.

For an **Oasis-mirror** source, social_referencing's own rules apply: learned trust, promotion only if the advice
demonstrably worked, and the consult budget.

## Guards (built with the layer; "enforced, or on the backlog")

- A test that the advice table is absent from `nac.json`, with an **export → ingest round trip**: bundle export reads
  the runtime table, so the test must show that nothing from it survives. Absence does not depend on hashing, so a
  single process suffices.
- A deletion-proven test that credit never writes the table, and that a `substrate_advised` action is credited under
  drive relief only. The test includes the `None`-to-tool gate-crossing case.
- "One foreign layer": until a lint exists, this is mechanization backlog M89.

## Open questions for its review

- Does social_referencing's limit L6 (no LLM prose turned into a valence on the action path) apply to an LLM
  *proposal*, which is an action rather than prose?
- The situation key: the cluster id, given Exp 44b's F4 finding that `get_agent_tool_biases` drops it.
- Thread ownership: every write happens on the loop thread at proposal consumption, with an `RLock`.
