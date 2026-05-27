# Minecraft Live Demo + Harness Benchmark

**Status:** Design exploration, post-1.0 (1.1 splash launch). Stub — implementation plan pending.
**Target version:** 1.1
**Concurrent with:** 1.0 stabilization (this work proceeds in parallel without gating 1.0).
**Depends on:** B4 Cradle ([archive/cradle_sensorimotor_development.md](archive/cradle_sensorimotor_development.md)) shipped — provides embodied learning foundation. Also benefits from [scene_actor_affordances.md](scene_actor_affordances.md) (1.1 track) for hostile mob mechanics.

---

## Front-gate scope pressure (retroactive)

Added 2026-05-27 per CLAUDE.md Principle 3.

**Question:** does this need to be its own mechanism, or can it ride on existing infrastructure?

**Existing infrastructure surveyed:**

| Candidate | Why insufficient (or sufficient) |
|---|---|
| `--embodiment` flag + `bodies/*.yaml` SEM templates | **Already the right entry point** for embodied agent simulation. Minecraft player body is one more body archetype — rides on existing pattern |
| `ToolBridge` + `AffordanceSchema` (`requires`, `self_effect`, `target_effect`) | **Already the right model** for "concrete embodied actions" (the plan's preferred approach over high-level intents) — Minecraft actions (move_to, mine_block, place_block, attack_entity) become SEM affordances |
| `scene_actor_affordances` `target_effect` (1.1 work) | **Already the right model** for hostile mob mechanics — zombie attacking player uses the same `target_effect` shape as Maxim-internal dragon |
| Cradle B4 (shipped) — cross-session evidence | **Already provides the foundational claim** — Minecraft demonstrates the claim viscerally, doesn't enable it |
| Voyager / GITM / SPRING comparison harness | External benchmark infrastructure — Maxim does not replace these; just runs alongside on same seeds |
| Custom benchmarking harness (would need to build) | The orchestrator + sim_log + report.json infrastructure already exists; Minecraft adds a new PerceptSource adapter + an external-process bridge, not a new benchmark mechanism |

**Verdict:** could-ride-on-existing for the **substrate side** (embodiment + affordances + sim infrastructure all exist). The new pieces are **external-integration adapters** — a Mineflayer/WebSocket bridge to drive a real Minecraft client and a comparison-protocol harness for fairness.

**Specific reason:** Minecraft is a *demonstration*, not a new mechanism. The bio-pipeline differentiation comes from existing infrastructure (NAc reward biases persisting across runs); Minecraft surfaces this in a third-party-recognizable environment. The new code is bridge + protocol + scoring — not new substrate mechanism. The plan's own framing ("post-1.0 splash launch") matches this — it's a marketable proof point of *existing* claims, not a load-bearing claim of its own.

## Goal

Demonstrate Maxim playing Minecraft against published LLM-Minecraft harnesses (Voyager, GITM, SPRING, others). The differentiator is the bio-pipeline: Maxim's NAc reward biases, hippocampal episodes, substrate-encoded concepts, and SCN circadian patterns persist across sessions and shape behavior in ways prompt-loop agents cannot match.

The cross-session learning claim (1.0 banner) becomes visceral in Minecraft: the agent that died to a creeper yesterday avoids them today, without prompt engineering, without skill-library curation.

---

## Why this is post-1.0, not gating

1. **Comparison protocol design is research-grade work.** Same seeds, same turn budgets, same embodiment, same Minecraft version, same world generation seed. Doing it wrong produces a worse demo than not doing it. This benefits from time and iteration, not a 1.0 deadline.
2. **1.0 = interface freeze.** Adding Minecraft to 1.0 either rushes the demo or delays the freeze. Splitting them lets each release have its own narrative: 1.0 is "the API is stable," 1.1 is "and here's why you should care."
3. **Cradle (B4) already provides cross-session evidence.** Experiment 11 validates the foundational claim. Minecraft strengthens the story; it doesn't enable it.
4. **Two news cycles.** 1.0 announces the stable release. 1.1 announces the benchmark results. Each gets focused attention.

---

## Open questions (to resolve before drafting full plan)

### Q1. Harness API

Which Minecraft integration surface? Options:
- **Mineflayer / mcp protocol** — proven, used by Voyager. JS-based, would need a Python bridge.
- **Minestudio / native gym envs** — Python-native but might lock in a specific Minecraft version.
- **Custom Forge mod with HTTP/WebSocket bridge** — full control, more work to maintain.
- **Minecraft Education Edition Code Connection** — official, simpler API surface.

Lean: Mineflayer with a WebSocket/JSON-RPC bridge to Python. Voyager's integration is well-documented; we can mirror it for fairness in the comparison protocol.

### Q2. Action space mapping

Voyager generates JS code that runs in Mineflayer. Maxim's tools-as-LLM-output is similar but the action space is a discrete set of game actions (move, look, place, mine, attack, eat, craft). Two approaches:
- **Affordance tools per Minecraft action** — `move_to`, `mine_block`, `place_block`, `attack_entity`. Each becomes a SEM affordance with `requires` preconditions and `target_effect` declarations.
- **High-level intent tools** — "explore," "gather wood," "find shelter." Cerebellum decomposes into low-level actions.

The bio-pipeline differentiation argues for option 1 — let the bio-systems learn from concrete embodied actions, not abstracted intents. Concrete affordances also let scene_actor_affordances apply naturally to mobs (zombie attacks the player using the same `target_effect` mechanism a Maxim-internal dragon would).

### Q3. Cross-session learning demonstration

What's the experimental design that *shows* cross-session learning?
- **Run 1:** fresh Maxim, hostile mob biome. Likely dies to creepers.
- **Run 2 (resume):** same Maxim, same biome. Should avoid creeper proximity, prefer ranged combat. Measure: time-to-death, kill/death ratio, action-class diversity, prompt-injection frequency (should NOT rise — the avoidance comes from NAc, not from extended prompts).
- **Comparison runs:** Voyager / GITM / SPRING on same seeds. Same metrics.

The novel metric: "behavioral change explained by NAc reward biases" — quantify how much of the run-2 vs run-1 delta is attributable to learned biology rather than prompt evolution. Voyager's skill library is the comparable mechanism on the other side.

### Q4. Comparison fairness

Voyager has a curated skill library that grows across runs. Maxim has NAc reward biases that grow. These are different mechanisms. Apples-to-apples is hard:
- Same total inference budget?
- Same wall-clock time?
- Same number of in-game actions?
- Same total prompt tokens?

This is a research design question, not an engineering question. Resolving it well is what makes the demo credible. Likely answer: report all four metrics + per-token efficiency curves so reviewers can see the trade-offs.

### Q5. Live demo UX

Streaming a Maxim Minecraft session to an audience needs:
- Game view (Minecraft client, third-person)
- Bio-state overlay (current pain signals, recent NAc predictions, hippocampus retrievals)
- Inner-monologue stream (thinking panel content)
- Tool call stream (which affordance fired, with what parameters)

Builds on the existing display infrastructure but with a different presentation target — viewer-friendly, not operator-friendly. This is its own ~50-100 LOC of streaming output formatters; can ship as part of the v1_refinement docs cleanup (D1-D3 area) if it's worth the cross-cut.

### Q6. Skill-library equivalence

Voyager / GITM publish their skill libraries. Maxim's analog is the NAc causal-link graph + ATL semantic concepts + Cerebellum forward models. For a fair side-by-side, we need to be able to export, visualize, and explain Maxim's learned "skills." This is a research artifact, not gating engineering, but worth flagging.

Likely deliverable: a `maxim cast inspect` or `maxim agent inspect` verb that exports the agent's learned NAc top-K, ATL concepts, and cerebellum predictions in a comparable format. Aligns with [scene_actor_affordances.md](scene_actor_affordances.md) Stage 5 instrumentation.

### Q7. Tick-rate alignment

Maxim's discrete tick model (~2-30Hz agent loop) needs to align with Minecraft's 20Hz tick rate. Options: sub-sample Maxim ticks (lossy), buffer Minecraft state and present per-Maxim-tick (more accurate, more latency), or run Maxim at 20Hz (forces low-latency LLM calls — feasible only with small/medium-tier models).

Lean: per-Maxim-tick state buffering. Minecraft state changes within a Maxim tick are summarized as percept deltas. Accept the latency cost; the bio-pipeline doesn't need 20Hz reactivity.

---

## What 1.0 needs to NOT preclude

For Minecraft work to slot in cleanly without 2.0-requiring refactors:

- **Percept source / action sink contracts** must be plug-replaceable without modifying the agent loop. Today the sim orchestrator owns these; the Minecraft adapter would be a parallel implementation. Verify [bridge.py](../../src/maxim/simulation/bridge.py) protocols are general enough. **This is exactly [v1_refinement.md CC8](v1_refinement.md) — sim adapter contract audit.** CC8 ships in 1.0 specifically to prevent this from being a 2.0 trap.
- **Embodiment must accept non-sim sensor sources.** The three-layer sensation model (contact / proximity / narrative) needs to extend to "real game state" as a fourth layer. The pipeline `sensor change → evaluate_failures → PainBus → NAc` should not assume a specific orchestrator.
- **Tool registration must support runtime add/remove.** Minecraft's available actions vary by inventory and world state. The scene-scoped tool window (I3, shipped 0.7) already supports this — verify it works without a sim orchestrator.
- **Persistence file formats need version markers.** A Minecraft session in 1.1 should be able to load 1.0-era persisted bio-state (and migrate or warn). [v1_refinement.md CC1](v1_refinement.md) shipped this in PR #203 — `_format_version` is now contract.
- **Tool dual-format schema** for any external tool integrations. [v1_refinement.md CC9](v1_refinement.md) shipped JSONSchema bridge in PR #204 — Minecraft adapter tools can declare schemas in either format.
- **MCP compatibility** as a parallel goal — Minecraft players who use Claude Desktop or other MCP clients should be able to configure Maxim as an MCP server providing Minecraft-aware tools. See [mcp_compatibility.md](mcp_compatibility.md).

---

## Stages (TBD when full plan drafted)

A reasonable shape:

### Stage M0 — Research design + comparison protocol (~1 week, no LOC)

- Decide on world seeds, turn budgets, success criteria, comparison metrics.
- Reproduce Voyager / GITM / SPRING baselines on chosen seeds.
- Document the protocol in `docs/experiments/12_minecraft_protocol.md`.

### Stage M1 — Mineflayer adapter (~300-500 LOC)

- Python ↔ Node.js bridge via WebSocket.
- `MinecraftPerceptSource` + `MinecraftActionSink` implementing Maxim simulation bridge protocols.
- Affordance tool generation per Minecraft action class.

### Stage M2 — Cross-session experiment harness (~150 LOC)

- `maxim --bench minecraft --runs N --seeds S1,S2,S3` runs paired sessions (fresh + resume).
- Metric collection: time-to-death, action diversity, NAc top-K growth, etc.
- Output format compatible with Voyager's published metric tables.

### Stage M3 — Comparison runs + paper draft (~ongoing)

- Run paired sessions on Maxim vs Voyager vs GITM vs SPRING.
- Write up findings.
- Public seeds + Docker images for reproducibility.

### Stage M4 — Live demo polish (~50-100 LOC)

- Streaming output formatter for viewer-friendly bio-state overlay.
- Recording/replay tooling for showcase content.

---

## Risks and tradeoffs

**R1. Voyager runs on GPT-4o; Maxim's research story is local-first.** Direct comparison on a hosted model masks the local-first story. Mitigation: run Maxim on multiple backends (claude-sonnet, mistral-7b, qwen2.5-14b) so the comparison shows backend-independence.

**R2. Mineflayer bridges have latency overhead.** WebSocket round-trips per action could push tick latency above acceptable bounds. Mitigation: batch actions per Maxim tick; accept that Maxim's tick rate is lower than Minecraft's.

**R3. Reproducibility for an LLM-driven harness is hard.** Same prompt + same seed doesn't guarantee same output. Mitigation: report distribution metrics (median + IQR) over N runs per seed, not single-run results.

**R4. Skill library comparison is rhetorically loaded.** Voyager's skill library is impressive; calling NAc the "Maxim equivalent" without showing visible artifacts undersells the claim. Mitigation: ship the inspection tooling (Q6) so the bio-state is browseable and explainable.

**R5. Minecraft scope is unbounded.** "Get diamond" is the standard goal but "play creatively" is what makes Minecraft Minecraft. Mitigation: scope tightly to the published benchmarks (Voyager's task suite); leave open-ended creative play to a follow-up demo.

---

## Cross-references

- [v1_refinement.md](v1_refinement.md) Section 8 — 1.1 track index.
- [v1_refinement.md](v1_refinement.md) CC8 — sim adapter contract audit (1.0 prereq for clean Minecraft adapter integration).
- [archive/cradle_sensorimotor_development.md](archive/cradle_sensorimotor_development.md) — embodied learning foundation this builds on.
- [scene_actor_affordances.md](scene_actor_affordances.md) — embodied hostile entities; relevant for Minecraft mobs (zombies, creepers, skeletons) using the same `target_effect` pattern.
- [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md) — revives if Minecraft demo exposes a cognition gap (zombies pathfinding, villagers remembering trades, recurring antagonists).
- [mcp_compatibility.md](mcp_compatibility.md) — sister 1.1 concern; MCP server mode could expose Minecraft tools to other agents.
- Voyager paper: https://arxiv.org/abs/2305.16291
- GITM paper: https://arxiv.org/abs/2305.17144
- SPRING paper: https://arxiv.org/abs/2305.15486
