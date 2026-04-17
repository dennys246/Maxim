# Experiment 4 (Tier 3) -- Reproduction Protocol

**Experiment:** Organic LLM learning -- agent learns from its own actions in a real sim, no scripted training.

## Quick verification

```bash
# Full Tier 3 (~3-5 min, requires leader with qwen2.5-14b or similar):
PYTHONPATH=src python scripts/behavioral_convergence_exp4_tier3.py --model qwen2.5-14b
```

## Prerequisites

- Leader online with LLM loaded (`maxim peer llm --status`)
- Peer config exists (`~/.maxim/peer.yml` or env vars)
- Network connectivity to leader (`maxim peer version`)
- SEM entity specs present: `_data/components/items/antidote_vial.yaml`, `poison_vial.yaml`, `purple_vial.yaml`, `teal_vial.yaml`, `orange_vial.yaml`

## What to expect

**Session 1 (exploration):** Agent is poisoned, has 3 masked vials. No prior knowledge. Expect roughly uniform or random selection. Agent experiences outcomes organically through CerebellumModulator.

**Session 2 (early learning):** Agent reloaded with Session 1 bio-state. Should show some preference shift toward teal (antidote). Teal rate ~25%.

**Session 3 (convergence):** Agent reloaded with Session 2 bio-state. Should converge strongly toward teal. Teal rate ~100%.

**Fresh control:** Agent with no prior experience, same scenario. Should die (never picks antidote without learning).

## Hypotheses (5/5)

1. Session 1 teal selection rate < 50% (exploration, no prior knowledge)
2. Session 3 teal selection rate > Session 1 teal selection rate (learning occurred)
3. Session 3 teal selection rate >= 75% (strong convergence)
4. Fresh control dies or picks non-teal (no learning signal)
5. Valence differentiation: teal valence > orange valence after Session 2+

## If hypotheses fail

1. **Agent never tries different vials:** Check that the scenario forces multiple turns and multiple poisoning events. The agent may need to experience failure before exploring alternatives.

2. **No valence differentiation after sessions:** Check that CerebellumModulator is wired into the executor and that reaction_bus subscribers are active. Verify `bio.cerebellum is not None` in `build_bio_stack`.

3. **LLM ignores valence context in later sessions:** Check that `StructuredContext.valence_context` is populated and that `PromptAssembler.compose_memory_section()` includes it. Run with `--json` to inspect the prompt.

4. **Fresh control survives:** This would mean the LLM has a language prior about teal/antidote. Verify that vial names are truly masked (arbitrary visual attributes, no semantic hints).

5. **Session 2 shows no improvement over Session 1:** Check persistence -- hippocampus/NAc save/load may have failed. Inspect the persist dir for `hippocampus.json`, `nac.json`, `cerebellum.json`.

## Key invariants

- **No scripted reactions.** All learning comes from the agent's actual tool executions through CerebellumModulator -> _emit_failure/success_reaction pathway.
- **Masked vial names.** Purple Hexagonal Glass, Teal Cylindrical Ceramic, Orange Triangular Crystal -- no semantic hints about function.
- **Session persistence.** Bio-state saved after each session and reloaded for the next.
- **Fresh control isolation.** Fresh agent has zero bio-state -- no hippocampus, no NAc, no cerebellum history.
- **CerebellumModulator in production.** `BioStack.cerebellum` wired through `build_executor(cerebellum=...)`. Reactions flow through ReactionBus to hippocampus + NAc.

## Experimental controls

- **Positional bias control:** Vial order shuffled per trial
- **Language prior control:** Vial names are arbitrary visual attributes
- **Organic training:** Agent takes actions and experiences outcomes -- no injected reactions
- **Model:** qwen2.5-14b, temperature 0.3
