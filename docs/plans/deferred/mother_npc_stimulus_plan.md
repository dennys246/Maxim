# Mother NPC Stimulus Plan — Deferred Research Infrastructure

**Status:** Deferred. Not on the critical path to 1.0.
**⚠ Needs heavy refinement.** This plan captures the *framing* and *known rough edges*, not a finished design. The isolation audit list in particular is a starting point — not a contract. Before reviving, expect 1–2 days of design work to turn the rough edges into concrete code-level checks against whatever the runtime looks like at revive time.

**Revive when:** [behavioral_convergence_practice.md](../behavioral_convergence_practice.md) has logged at least two successful experiment entries on hand-authored fixtures AND a third experiment is blocked on "we need more stimulus variety than we can author by hand." **Do not revive before this trigger.** Mother NPC is the correct answer to a problem that isn't load-bearing yet, and building her before the trigger would dilute substrate mechanism work with stimulus-infrastructure work.

**Prerequisites (from [reaction_abstraction_plan.md](../reaction_abstraction_plan.md)):**
- **Phase 4 (runtime unification)** — Mother (AgentInstance-shaped) must produce Percept objects that Baby (MaximAgent-shaped) consumes. Phase 4's `make_text_percept` factory + AgentPool integration is a hard prereq.
- **Reaction isolation rule** — Reactions are Baby-internal: Baby generates them from her own bio-stack, not from Mother's intent. The isolation-hygiene rule in `ReactionContext` (Phase 1) is the formal contract; the leak-vector audit below must verify no Reaction back-channels exist at revive time.

**Kin:**
- [bio_system_plugin_plan.md](bio_system_plugin_plan.md) — similar deferred-plan shape, also conditional on platform/research trigger
- [mother_maxim_plan.md](mother_maxim_plan.md) — adjacent but different (that plan is about persistent collective memory across sessions; this plan is about a stimulus-producing NPC that never shares state with the AUT)
- [../reaction_abstraction_plan.md](../reaction_abstraction_plan.md) — provides the Percept/Reaction information barrier this plan's isolation requirements depend on
- [../behavioral_convergence_practice.md](../behavioral_convergence_practice.md) — this plan is the likely infrastructure for H1–H5 experiments at scale
- [unified_event_bus_plan.md](unified_event_bus_plan.md) — tangential; if that plan revives first, Mother NPC uses the unified bus; if this plan revives first, Mother NPC subscribes to the existing percept pipeline

## The framing

**"Baby Maxim"** (the AUT): a substrate-driven agent whose bio-stack (ATL, Hippocampus, NAc, PerceptTraceBuffer, SCN) is the thing we're trying to demonstrate cross-session learning in. Baby's LLM is frozen, off-the-shelf, and should be small enough to be cost-effective across many sessions. Baby's behavior change across sessions has to come from its bio-stack state evolving, not from any other source.

**"Mother NPC"**: a separate agent instance whose job is to produce realistic, varied percepts that Baby experiences. Mother provides the *environment* Baby learns from. Mother has her own LLM (potentially different from Baby's, but see the "same-class LLM discipline" warning below), her own memory if she has any, her own agent_id, and absolutely no access to Baby's internal state.

**The interaction surface: percepts only.** Mother emits speech / scene events / reward signals through the existing percept pipeline. Baby consumes them as sensor input. Neither agent sees the other's internals. The information flow between them is exactly the bandwidth of the percept stream — which is the same bandwidth a real biological child has into its parent's mind, which is to say: only what's externally observable.

## Why this sharpens the research claim instead of weakening it

The substrate plan's 1.0 banner is *cross-session learning without LLM fine-tuning*. A naive concern is that adding a generative Mother risks "emergence" coming from Mother's LLM being clever rather than from Baby's substrate actually learning. That concern is real, but the isolation requirements below prevent it.

What Mother NPC *gives* the research program is **scalable, realistic stimulus variety** without the combinatorial explosion of hand-authored fixtures. By the time you're at P4 / P6 / P8 / behavioral convergence experiments, hand-authored YAML starts to feel artificial and fixture authoring becomes the bottleneck. Mother gives you open-ended variation from a scenario brief, and you can re-run a scenario with different Mother seeds to get different-but-plausible sessions — which is exactly what robust behavioral convergence claims need.

**The key is that Baby's LLM stays frozen.** Mother's LLM produces stimuli; Baby's LLM processes language with whatever context the substrate recalls. The research claim — "Baby gets better across sessions without fine-tuning Baby's LLM" — remains exactly as testable, and actually more credible because the environment is now realistic.

## Known rough edges — information leak vectors

The isolation requirement is the whole claim. **Zero information leak between Mother and Baby except through the percept stream.** Here are the leak vectors identified so far, each with a starting-point mitigation. **These need verification against actual runtime code at revive time** — several of these assume runtime structures that may have evolved by then.

### Leaks that would break the claim

**1. Shared LLM context.**
Mother and Baby share the same LLM instance or conversation history. Baby's "learning" would just be Mother's prompt context leaking through the shared model state.

*Mitigation:* separate LLM instances, separate conversation threads, separate backend processes. Ideally Mother and Baby route through different `LLMWorker` instances or even different routers. Audit for shared prompt caches at the backend level (llama.cpp can reuse KV cache across calls on the same instance — this is a real leak vector if both agents hit the same llama-cpp-server).

**2. Shared memory stores.**
Mother and Baby both have bio-stack instances that share a MemoryHub, a persistence directory, or a cross-layer event callback. Baby's "recall" would be Mother's reasoning leaking in through shared memory.

*Mitigation:* per-agent memory stores (already provided by AgentFactory's fresh-instance pattern per iceberg sweep verification), per-agent `~/.maxim/sessions/{agent_id}/` directories, separate agent_id (F0.5 already threads this), zero cross-reads between agents' memory stores. Assert this with integration tests.

**3. Shared random seed.**
Mother and Baby share an RNG and their "independent" decisions end up correlated in a way that mimics information transfer. Subtle and dangerous — the correlation might look like Baby learning Mother's patterns when it's really just RNG coupling.

*Mitigation:* per-agent RNG streams derived from different seeds. F0.5 (agent_id threading) + simulator_upgrades S4 (deterministic seeding) should be extended with explicit per-agent RNG streams at the simulator level. **This needs a concrete code change in S4** — see the hygiene note in archive/simulator_upgrades_plan.md.

**4. Percept metadata back-channel.**
Mother's Percept emissions carry metadata fields (sender_id, scene_id, intent annotations) and Baby's substrate reads those fields as stronger signal than the plain content. This isn't inherently wrong — real children get intent from tone and context — but it becomes a back-channel if Mother encodes her goals in metadata Baby reads mechanically.

*Mitigation:* audit `PerceptContext` (F0.4) carefully. Mother cannot set fields that real sensors can't produce. **No `mother_intent_hint`, no `mother_reward_for_baby`, no `mother_scenario_goal` fields.** If Mother wants to signal intent, she has to do it through speech content that Baby parses like any other text. Add this as an explicit rule in F0.4's schema documentation.

**5. Orchestrator cheating.**
The simulator runtime has to know both Mother's and Baby's state to run the sim at all. Anything the orchestrator does *based on* that knowledge is a leak. This is the subtlest vector because it's partially unavoidable.

*Mitigation:* the orchestrator can schedule tick order and route percepts, but it cannot:
- Produce reward signals from Mother's state (reward comes from Mother-as-percept or from explicit scenario rules, never from the orchestrator peeking)
- Modulate Mother's LLM output based on Baby's internal state
- Delay or drop Mother's output based on what Baby is "about to do"
- Include any cross-agent state in either agent's log stream

This needs a **dedicated isolation contract test** that enumerates orchestrator behaviors and asserts which ones are allowed vs. forbidden.

**6. Co-located filesystem writes.**
Both agents write to the same session log file and reloads accidentally mingle state. Boring bug but it actually happens.

*Mitigation:* per-agent subdirectory, per-agent log files, assert directory isolation at test time. F0.5 + existing AgentFactory patterns get most of the way there but worth explicit verification.

### Leaks that are *not* leaks (allowed)

- **Mother seeing what Baby said aloud.** That's how conversation works. Baby's utterances are percepts Mother can respond to, and that's the whole point.
- **Mother having a scenario brief.** "Teach the concept of mugs across three sessions" is the experimenter setting up the environment, not a leak from Baby.
- **Baby's substrate storing facts about Mother.** Who Mother is, what she said last time, reward history. That's what the substrate is *for* — the point is Baby learns Mother across sessions. As long as that knowledge comes through the percept pipeline, it's legitimate.
- **Wall-clock time.** Both agents see the same clock. Fine.
- **Shared global constants.** Both agents know what month it is in the simulated world, what the laws of physics are, etc. Fine.

## The same-class-LLM discipline

If Mother is Claude Opus and Baby is a local 7B model, a critic will correctly say "Baby's learning is just Baby getting better at parroting what Opus said." That's a distillation experiment dressed up as learning.

**Mother and Baby should use the same LLM, or at least comparable-class LLMs.** A 7B Mother teaching a 7B Baby is a fair experiment. A 14B Mother teaching a 14B Baby is fine. Opus teaching a 7B Baby is not.

The research claim is about substrate, not LLM gradient. Keep the LLMs comparable. Document the choice in every experiment entry.

## Two modes

### Deterministic replay mode

For mechanism tests (P1, P2, P3a convergence sims) where determinism matters, Mother runs once in "record mode" to generate a percept stream, which is persisted. Subsequent runs replay the recorded stream. Baby sees identical percepts across runs. This gives the reproducibility mechanism tests need while still using Mother to produce realistic variation at record time.

### Live seeded mode

For behavioral convergence experiments where variety matters, Mother runs live with an explicit seed. Different seeds produce different-but-plausible sessions. Baby sees fresh stimulus each run. Reproducibility comes from seed-pinning, not output-pinning.

Both modes should exist. Mechanism tests use replay mode; behavioral convergence uses live mode. Be explicit about which mode is in use for any given experiment in the practice doc entry.

## Failure modes (things that could go wrong beyond leaks)

**Scenario drift.** Mother is supposed to be teaching Baby about mugs; she gets distracted and spends three sessions talking about birds. Baby's substrate fills up with bird concepts and the experiment is now measuring something else.

*Mitigation:* structured scenario brief, not free rein. Check transcripts periodically. Consider a "scenario rail" that can redirect Mother if she drifts too far from the intended teaching.

**Reward signal cleanliness.** Who decides when Baby gets a reward?
- *If Mother decides:* you're measuring "Baby's ability to satisfy Mother's judgment." Interesting but not the same claim as "Baby learns." Also introduces a leak vector because Mother's judgment is internal state.
- *If a rigid scenario rule decides* ("Baby said 'mug' → reward"): you're back to hand-authored ground truth. Clean but limited.

*Honest answer:* probably both, at different scales. Hand-authored rules for mechanism tests (P1, P2). Mother-as-judge for behavioral convergence where the metric is subjective. **Be explicit about which is in use for every experiment.** A mixed-mode experiment where reward comes partly from Mother and partly from rules is fine as long as the split is documented.

**LLM capability asymmetry.** Covered above under "same-class LLM discipline" but worth re-stating as a failure mode — this is the one most likely to make a critic dismiss the results.

**Reproducibility under live Mother.** Live mode is non-deterministic by nature. Any claim based on live-mode results needs high-N seeding and statistical reporting (mean ± std over ≥10 seeds, same as substrate phase pass criteria).

**Isolation audit gaps.** The six leak vectors above are a starting point. There are almost certainly others specific to whatever runtime exists at revive time. **Before running any Mother+Baby experiment, audit against the current runtime — do not trust this document's list to be complete.**

## Where Mother lives in the codebase

Mother is an orchestration artifact, not a bio-stack component. She lives in the simulator layer:

- Likely home: `src/maxim/simulation/mother_npc.py` as an extension of `generative_runner.py` or as a parallel runner
- Uses the existing `AgentFactory.create_agent()` pattern to get her own bio-stack (which she may or may not actually use — she's primarily an LLM-driven stimulus source, not a memory-accumulating agent)
- Does NOT touch `src/maxim/memory/`, `src/maxim/decisions/`, or the `agents/bus.py` percept dataclass
- Subscribes to (or publishes through) the percept pipeline the same way `ConversationalSource` does today — see simulator_upgrades S1 and F0.8 for the `inject_sensor` API she'd use

The existing runtime already supports two-agent setups (per the iceberg sweep — `AgentFactory` creates per-agent isolated bio-stacks). What's missing is the **harness** that wires "agent A is Baby, agent B is Mother, the percept pipeline only flows Mother→Baby at the stimulus layer."

## Phased build when revived

### Phase 1 — Isolation contract test (~150 LOC)

**Before** any other Mother code lands, write the isolation audit tests:
- Two-agent sim construction with explicit Baby and Mother roles
- Assert separate LLM instances (and separate backend processes if feasible)
- Assert separate memory directories, zero cross-read
- Assert separate RNG streams, no seed coupling
- Assert percept metadata schema forbids any Mother-intent fields
- Assert orchestrator doesn't pass cross-agent state
- Assert separate log files

If these tests can't be written against the current runtime, the runtime needs isolation fixes *first*. Phase 1 is the gate.

### Phase 2 — Minimal Mother implementation (~200 LOC)

- `MotherNPC` class with a scenario brief, an LLM, and a percept emission method
- Record mode: generates a percept stream to disk
- Live mode: generates percepts on-demand during a sim run
- Integration with `FixtureDrivenOrchestrator` (S1) as a percept source alongside fixture YAML

### Phase 3 — First experiment (~100 LOC + authoring)

A single behavioral convergence experiment from [behavioral_convergence_practice.md](../behavioral_convergence_practice.md) ported to use Mother NPC. Compare: same scenario, hand-authored fixture vs. Mother-generated. Results go in the practice doc.

### Phase 4 — Scenario library (~ongoing)

As behavioral convergence experiments accumulate, build a library of scenario briefs that can be re-used. Each brief specifies: teaching goal, Mother persona, reward rules, duration, expected Baby behavior changes.

### Phase 5 — Revisit after first-experiment findings

Phase 3 will reveal leak vectors and failure modes I haven't thought of. Phase 5 is explicit "re-audit the isolation contract based on what Phase 3 taught us." Do not skip this — Phase 1's audit will have blind spots, and Phase 3 is where they'll surface.

**Total scope when revived: ~500–700 LOC plus ongoing scenario authoring.** 2–3 weeks of focused work.

## Non-goals

- **No Mother with persistent cross-session memory of Baby.** That turns this plan into a variant of `mother_maxim_plan.md` (persistent collective memory) and the research claims diverge. Mother is stateless across session boundaries by default, or at most has her own memory that's fully isolated from Baby's. A persistent-memory Mother is a separate plan.
- **No Mother-as-teacher-with-gradients.** Mother does not produce learning signal for Baby's LLM — the LLM stays frozen. Mother produces stimulus. If you want gradient-based learning, that's a different research program.
- **No multi-Mother setups** in Phase 1–2. Start with one Mother teaching one Baby. Multiple Mothers is a later phase that adds complexity.
- **No cross-session Mother continuity** without explicit isolation audit. If you want Mother to "remember" Baby across sessions, that's a new leak surface and needs its own audit round.
- **No Mother-generated fixtures replacing P0/P1 hand-authored fixtures.** The substrate phase fixtures stay hand-authored because mechanism tests need ground truth. Mother is for behavioral convergence experiments where the metric is behavior change, not mechanism correctness.

## Cross-plan interactions

- **substrate_plan.md:** Mother doesn't affect any P-phase directly. She's stimulus infrastructure that downstream experiments can optionally use. P1–P8 still run with hand-authored fixtures. B-phases (Track B) might benefit from Mother for blind A/B NPC coherence tests in B3.
- **foundations_plan.md F0.4:** percept context schema must forbid Mother-intent back-channel fields. Add to F0.4 documentation when this plan is revived.
- **foundations_plan.md F0.5:** agent_id threading is load-bearing for Mother+Baby isolation. Already covered.
- **archive/simulator_upgrades_plan.md S4:** per-agent RNG streams are load-bearing for Mother+Baby isolation. Already noted in S4.
- **archive/simulator_upgrades_plan.md S1:** Mother is a percept source that plugs into the fixture-driven orchestrator, reusing the same pipeline. No new orchestrator.
- **behavioral_convergence_practice.md:** H1–H5 experiments become tractable at scale once Mother exists. Each experiment entry that uses Mother must document the mode (replay/live), the LLM class, the isolation audit status, and the reward signal source.

## If you're reading this cold

You found this plan because behavioral convergence experiments are bottlenecked on hand-authored fixture variety. Before you start building:

1. **Confirm the revive trigger fired.** Has behavioral_convergence_practice.md actually accumulated the two successful experiments + one blocked experiment? If not, the need isn't real yet. Close this file.
2. **Re-audit the isolation leak vectors against the current runtime.** The list in this document is from a specific point in time. Things may have changed. Verify every vector against current code before trusting it.
3. **Commit to Phase 1 first, no shortcuts.** The temptation will be to write `MotherNPC` first and "come back to the isolation tests." Resist. Phase 1 is the gate for the whole research claim.
4. **Pick the LLM class deliberately.** Mother and Baby using the same model is the cleanest starting point. Document the choice.
5. **Pick ONE experiment to port first.** Don't port all of H1–H5 simultaneously. One experiment, one commit, one result in the practice doc. Then evaluate whether the isolation held.
6. **Budget time for Phase 5.** The first experiment will teach you things that invalidate parts of the isolation audit. That's a feature, not a failure — just budget for the follow-up audit round.

## Summary of rough edges that need refinement before build

- **Leak vector list is provisional**, not a contract
- **Orchestrator contract** (what it can and can't do based on cross-agent knowledge) needs concrete enumeration against current runtime
- **Reward signal source policy** needs explicit decision per experiment (Mother-as-judge vs. rigid rules vs. mixed)
- **LLM backend separation** needs audit at the backend-process level, not just at the Python object level (KV cache leaks are real)
- **Scenario drift guardrails** are undefined
- **Cross-session Mother continuity** policy is undefined (default: stateless, but needs explicit decision)
- **Phase 5 re-audit process** is sketched but not concrete
- **Integration with F0.8 sensor contract** needs concrete mapping of which sensor schemas Mother emits
- **Isolation tests** don't exist yet and are the gate for everything else

None of these are blockers to reviving the plan. All of them are real design work that happens at revive time, not speculatively now.
