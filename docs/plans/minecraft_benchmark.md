# Minecraft as the sharing apparatus (1.1.4 seam → 1.2 benchmark)

> **REVIVED 2026-08-30 from `deferred/` by the 1.2 scoping dive.** The deferral banner
> below is kept as the record. What changed: the 1.2 scoping pass established that
> Minecraft is not a demo or a splash launch — it is the **instrument** for the 1.2
> sharing claim, and the only place the "find something seen previously" coordination
> task is achievable at all. The Reachy Mini **cannot translate** (body yaw is rotation;
> no odometry, no depth, no SLAM — a repo-wide grep returns one docstring hit), so
> allocentric spatial tasks are physically unavailable on the robot. Minecraft supplies
> ground-truth coordinates for free. The robot becomes the *replication*, not the
> experiment.
>
> Scope changed accordingly: **not** a Voyager/GITM comparison harness, **not** a splash
> demo. One contingency, two agents, four arms, a pre-registered gate. Everything else in
> the original stub is out of scope until that lands.

> **DEFERRED (2026-07-15 plans audit):** Stub, zero implementation (Minecraft appears only in CC8 adapter-contract docstrings). Its 1.0 must-not-preclude prerequisites (CC8 plug-replaceable adapters, CC1 `_format_version`, CC9 dual-format schema) all shipped, so nothing is decaying. **Revive when:** 1.1 splash-launch work is greenlit AND someone commits to the M0 comparison-protocol research (or a second external-world adapter consumer appears).


**Status:** ACTIVE (revived 2026-08-30). Seam lands in **1.1.4**; the benchmark it exists for is the **1.2** headline.
**Target version:** 1.1.4 (infrastructure, no claim) → 1.2 (the pre-registered result).
**Target version:** 1.1
**Concurrent with:** 1.0 stabilization (this work proceeds in parallel without gating 1.0).
**Depends on:** B4 Cradle ([archive/cradle_sensorimotor_development.md](archive/cradle_sensorimotor_development.md)) shipped — provides embodied learning foundation. Also benefits from [scene_actor_affordances.md](deferred/scene_actor_affordances.md) (1.1 track) for hostile mob mechanics.

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

Lean: Mineflayer with a WebSocket/JSON-RPC bridge to Python. Voyager's integration is well-documented; we can mirror it for fairness in the comparison protocol. *(SHIPPED DIFFERENTLY 2026-09-04, PR 3: NDJSON over plain TCP — same JSON-RPC-shaped messages, zero new dependencies; protocol authority is `src/maxim/simulation/minecraft.py`. "WS" here named a transport family, and the family's browser-facing member is a bridge-process add if ever needed.)*

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

Likely deliverable: a `maxim cast inspect` or `maxim agent inspect` verb that exports the agent's learned NAc top-K, ATL concepts, and cerebellum predictions in a comparable format. Aligns with [scene_actor_affordances.md](deferred/scene_actor_affordances.md) Stage 5 instrumentation.

### Q7. Tick-rate alignment

Maxim's discrete tick model (~2-30Hz agent loop) needs to align with Minecraft's 20Hz tick rate. Options: sub-sample Maxim ticks (lossy), buffer Minecraft state and present per-Maxim-tick (more accurate, more latency), or run Maxim at 20Hz (forces low-latency LLM calls — feasible only with small/medium-tier models).

Lean: per-Maxim-tick state buffering. Minecraft state changes within a Maxim tick are summarized as percept deltas. Accept the latency cost; the bio-pipeline doesn't need 20Hz reactivity.

---

## What 1.0 needs to NOT preclude

For Minecraft work to slot in cleanly without 2.0-requiring refactors:

- **Percept source / action sink contracts** must be plug-replaceable without modifying the agent loop. Today the sim orchestrator owns these; the Minecraft adapter would be a parallel implementation. Verify [bridge.py](../../src/maxim/simulation/bridge.py) protocols are general enough. **This is exactly [v1_refinement.md CC8](archive/v1_refinement.md) — sim adapter contract audit.** CC8 ships in 1.0 specifically to prevent this from being a 2.0 trap.
- **Embodiment must accept non-sim sensor sources.** The three-layer sensation model (contact / proximity / narrative) needs to extend to "real game state" as a fourth layer. The pipeline `sensor change → evaluate_failures → PainBus → NAc` should not assume a specific orchestrator.
- **Tool registration must support runtime add/remove.** Minecraft's available actions vary by inventory and world state. The scene-scoped tool window (I3, shipped 0.7) already supports this — verify it works without a sim orchestrator.
- **Persistence file formats need version markers.** A Minecraft session in 1.1 should be able to load 1.0-era persisted bio-state (and migrate or warn). [v1_refinement.md CC1](archive/v1_refinement.md) shipped this in PR #203 — `_format_version` is now contract.
- **Tool dual-format schema** for any external tool integrations. [v1_refinement.md CC9](archive/v1_refinement.md) shipped JSONSchema bridge in PR #204 — Minecraft adapter tools can declare schemas in either format.
- **MCP compatibility** as a parallel goal — Minecraft players who use Claude Desktop or other MCP clients should be able to configure Maxim as an MCP server providing Minecraft-aware tools. See [mcp_compatibility.md](deferred/mcp_compatibility.md).

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

- [v1_refinement.md](archive/v1_refinement.md) Section 8 — 1.1 track index.
- [v1_refinement.md](archive/v1_refinement.md) CC8 — sim adapter contract audit (1.0 prereq for clean Minecraft adapter integration).
- [archive/cradle_sensorimotor_development.md](archive/cradle_sensorimotor_development.md) — embodied learning foundation this builds on.
- [scene_actor_affordances.md](deferred/scene_actor_affordances.md) — embodied hostile entities; relevant for Minecraft mobs (zombies, creepers, skeletons) using the same `target_effect` pattern.
- [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md) — revives if Minecraft demo exposes a cognition gap (zombies pathfinding, villagers remembering trades, recurring antagonists).
- [mcp_compatibility.md](deferred/mcp_compatibility.md) — sister 1.1 concern; MCP server mode could expose Minecraft tools to other agents.
- Voyager paper: https://arxiv.org/abs/2305.16291
- GITM paper: https://arxiv.org/abs/2305.17144
- SPRING paper: https://arxiv.org/abs/2305.15486


---

# The 1.2 scoping pass (2026-08-30)

Everything above this line is the original 2026-07 stub, kept because its front-gate survey
is still correct: the substrate side rides on existing infrastructure. Everything below is
the scope the 1.2 dive established.

## Why Minecraft rather than two robots

Three findings decided this, each verified against the code:

1. **The Reachy Mini cannot translate.** `body_yaw` is rotation; there is no odometry,
   depth, or SLAM anywhere in `src/` (one docstring hit, repo-wide). A coordination task of
   the form "A saw X somewhere; A and B relocate X" needs a world location, and the robot
   cannot produce one at any budget. Minecraft supplies ground-truth coordinates from the
   engine.
2. **The seams were designed for exactly this.** `simulation/sources.py::PerceptSource` and
   `simulation/sinks.py::ActionSink` were frozen as CC8 contracts **naming Mineflayer as the
   target**, and `runtime/sim_adapter.py` documents that a Minecraft adapter reuses
   `is_sim_mode`. A player body is a declarative YAML: `embodiment/component_registry.py`
   discovers `~/.maxim/components/**/*.yaml` before bundled data, and
   `embodiment/tool_bridge.py::generate_tools_for_entity` generates the tools.
3. **Cost per trial.** The sharing claim needs arms and n. A game gives unlimited trials,
   full ground truth, no customs delays, and no motor damage. The hardware replication then
   answers a *different* question — does it survive a real body — at n = 12, which is what
   hardware n has always been.

## What to build (1.1.4, infrastructure only, NO claim)

| Piece | Shape | Notes |
|---|---|---|
| Bridge process | JS (Mineflayer) + ~~WS~~ **NDJSON/TCP** (corrected 2026-09-04, PR 3 — protocol authority `src/maxim/simulation/minecraft.py`) | The only non-Python component. |
| `MinecraftPerceptSource` | implements the frozen `PerceptSource` | Percepts must be text-shaped or take the numeric route below — `MemoryHub.on_percept_received` **returns early unless `transcript_chunk` or `content` is non-empty text**. |
| Affordance backend seam | injected reader callable + a writer into `Entity.vital_metrics` | Copy the Reachy pattern (`embodiment/audio_localization.py::AzimuthDoASource(doa_reader=...)`). Affordance effects today are *declarative deltas*; a Minecraft action must call the game and read back truth. `embodiment/backends/` currently holds exactly one file. |
| `bodies/minecraft_player.yaml` | sensors (health, hunger, light_level, y, nearest_hostile_dist), modulators → `move`/`look`/`mine`/`place` | Drop-in file, zero code. Any sensor the world owns must have a null `drive`, or `body.py`'s drift loop fights the writer. |
| World modality channel | one entry in `runtime/agent_loop.py::_SUBSTRATE_CHANNELS` | **Not free:** `_EXTEROCEPTIVE_ROOT_SENSORS` is a hardcoded `("azimuth",)` tuple, and the `recommend_action` cluster-bias sum scales with channel count — adding one is a **selection-dynamics recalibration**, and must be re-baselined, not assumed. |
| Two-AUT-one-world harness | two `run_agentic_loop` threads, separate `percept_source`/`action_sink` | No shipped code runs two full AUTs against one world (`AgentPool.run_turn` says in its own docstring it does not run the full loop). Mechanically straightforward; new. |

Estimate: **800–1500 LOC plus the bridge.** An adapter over designed seams, not a subsystem.

### The sensor ceiling is a THRESHOLD artifact, not an information limit (measured 2026-09-01)

> **Tracked as [L11](../limits/README.md#l11--sensor-count-dilution-and-the-discrimination-ceiling-behind-it--mitigated)** in the instrument ledger, with its `Re-measure on:` trigger. The measurements below are the source; L11 is the durable entry a future pre-registration author reads at design time.

The extero/intero dilution finding caps a modality channel at roughly **14 informative
scalars**: `_sensor_embed` sums `(1-v)·basis_low + v·basis_high` over sorted sensor names, so
each contributes 1/N, and at N ≥ 15 a full single-sensor swing no longer clears
`SensorEncoderConfig.pattern_threshold = 0.85`. Read literally that caps a Minecraft body at
~8 hand-projected scalars, which would be a hard constraint on "maximally embody."

**Measured, it is not a hard constraint.** Three results:

1. **The law is `cos ≈ 1 − 0.57/N`** — clean 1/N, confirmed from N=1 to N=200.
2. **Signal degrades but noise does not.** A meaningful single-sensor swing falls as 1/N
   (0.119 → 0.006 across N=4→100) while an all-sensor jitter stays flat at ~0.0008. SNR falls
   185:1 → 7:1 — real degradation, but **~15:1 headroom still remains at N=50**.
3. **An N-scaled threshold fully recovers separability** — *superseded 2026-09-01 by the
   bake-off; see the correction under §Bake-off in
   [l11_sensor_dilution.md](../limits/l11_sensor_dilution.md).* The result below is a PAIRWISE
   synthetic measurement and it is reproducible, but on the full frozen metric
   (`min(separation, stability, discrimination)`) against the REAL `EntorhinalCortex` the scaled
   threshold scores **0.70–0.84 and degrades with N**, while the nonlinear gain scores
   **1.00/1.00/1.00 from N=30 up**. Pairwise measurement did not reveal that; only the full
   metric did. Kept as written because it is what was measured then. With
   `threshold = 1 − 0.30/N`: **100% signal separation AND 100% noise rejection at every N from
   6 to 80.** The fixed 0.85 separates 0% of half-swings even at N=6 — it is calibrated for
   ~6 drives and nothing else.

**Dimension is not a lever.** 8× more embedding dimensions (384 → 3072) changes the cosine by
< 0.001 at every N, and marginally for the worse. Dilution is an *averaging* problem, not a
*capacity* problem: the sensors are not running out of room, they are being summed. More room
does not un-average them. `dim` matters only as a floor (it must stay ≫ N for the bases to be
near-orthogonal); 384 is ample at N=50.

**Can a different equation help? Yes — but not via randomness.** Four encodings measured,
40 random resting states each, signal = one sensor to an extreme, noise = all-sensor 2% jitter:

| encoding | signal N=6 → N=100 | noise | SNR N=6 → N=100 |
|---|---|---|---|
| current: `(1-v)·lo + v·hi`, summed | 0.074 → **0.006** | 0.0008 | 93 → **7.5** |
| sparse (each sensor writes k=16 dims) | **identical to current** | — | — |
| deviation: `(v - set_point)·(hi - lo)` | 0.591 → 0.107 | 0.015 | 29 → 7.0 |
| **sharpened: gain ∝ `(|v-0.5|·2)^3`** | **0.794 → 0.727** | 0.057 | **18 → 12.7** |

**Sparse hashing is exactly as diluted as the plain sum** — the 1/N law is inherent to
summing N contributions and comparing the sum by cosine, and no basis trick escapes it.
**Sharpening does escape it**: weighting each sensor by how far it sits from its set point
keeps the signal essentially FLAT across a 16× sensor increase (0.79 → 0.73), at the cost of a
higher noise floor, and holds SNR roughly constant where the plain sum's collapses. The
mechanism is "a sensor resting at its set point should not be shouting," which is the same
principle the comfort-band drive design already encodes.

Note the crossover: the plain sum has the **better** SNR at small N (93 vs 18) and they cross
around N≈30–50. So sharpening is the right choice for a many-sensor body and the wrong one for
a six-drive infant — it is a trade, not a free win.

**Distributional moments help DETECTION and hurt DISCRIMINATION.** Adding permutation-invariant
shape statistics of the sensor vector (mean, sd, skew, excess kurtosis, max-deviation) to the
embedding gives an N-*independent* "something changed" signal — at equal weight, cos after a
single-sensor spike is **0.63 at N=6 and 0.27 at N=100**, against identity-only's 0.93 → 0.995.
That is the one thing measured here that does not dilute.

But it is blind to *which* sensor moved, by construction:

| | detection (rest vs one spike) | discrimination (sensor A spike vs sensor B spike) |
|---|---|---|
| identity only, N=100 | 0.995 | 0.990 |
| + moments (w=1.0), N=100 | **0.270** | **0.995** |
| moments only, N=100 | −0.503 | **0.999** |

**Discrimination is the real ceiling, and moments make it worse at every weight.** No choice of
moments fixes this — permutation invariance is what a moment *is*. And note identity-only is
already at 0.990 discrimination at N=100: two entirely different sensors spiking are 99% alike
*before* any moment block. For learning "turn_left helps when the sound is left," discrimination
is the whole game, so a moment-heavy encoding would be actively harmful.

**The structural fix is to stop compressing N sensors into ONE cluster id.** Splitting 50 sensors
across G per-type channels puts each channel back in the small-N regime:

| channels G | sensors/channel | detection | discrimination |
|---|---|---|---|
| 1 | 50 | 0.991 | 0.980 |
| 2 | 25 | 0.979 | 0.955 |
| 5 | 10 | 0.949 | 0.897 |
| 10 | 5 | 0.909 | **0.831** |

Grouping alone does **not** clear the fixed 0.85 bar — it is grouping **and** the scaled
threshold, not either alone.

**This rides existing machinery and is the natural completion of a half-written intent.**
`_SUBSTRATE_CHANNELS` is already a tuple of `ModalityChannel(tag, reader, ranger)`, and
`_EXTEROCEPTIVE_ROOT_SENSORS = ("azimuth",)` carries the comment *"Kept a named set so a future
exteroceptive sensor is one entry, not a code change at the read site."* The missing piece is
that **a sensor cannot declare its own modality**: the YAML schema accepts `unit`, `range`,
`initial`, `drive` and nothing else, so channel membership lives in hardcoded name tuples rather
than in the body. Adding a `modality:` field to the sensor schema and deriving channels from it
would: recover discrimination by grouping; remove the hardcoded `("azimuth",)` that 1.1.4 has to
generalise anyway; and make "does this body have audio?" a property of the body — which is where
[roadmap_1_3_path.md](roadmap_1_3_path.md) §Stage B already says it belongs.

**Cost, and it is the known one:** `recommend_action` sums `cluster_reward_bias` additively
across the active channel set, so the term's range grows with channel count (±2 today, ±5 at
G=5) while `min_confidence` stays 0.3. Every added channel is a selection-dynamics
recalibration, and nothing in CI catches it.

**A measurement caution, recorded because it nearly produced a false result here.** A first
pass appeared to show deviation- and sharpened-encoding achieving perfect separation at every
N. It was an artifact: with every sensor resting at exactly 0.5, both encodings produce the
**zero vector**, and the cosine helper returned 0.0 for it. The numbers above use a scattered
resting state and return `nan` on a degenerate vector instead of a flattering zero.

**So maximal embodiment is achievable**, by either of two routes.

> **DECIDED 2026-09-01 by bake-off, and it went the other way from the hedge below.** The
> nonlinear gain (arm A4) wins at every N and is perfect from N=30 up; the scaled threshold is
> second; and the two **combined** are WORSE than the threshold alone, because grouping shrinks
> per-channel N, which loosens `1 − k/N` and lets noise separate. "Cheap vs durable" was the
> right axis and the wrong conclusion: the cheap one is also the weaker one. See
> [l11_sensor_dilution.md](../limits/l11_sensor_dilution.md) §Bake-off. The caveats below still
> stand in full — they are about A4, and A4 is what ships.

The cheap one is a threshold
scaled with sensor count — one line, fixes the symptom, does not improve SNR (it moves the bar
to where the signal actually is). The durable one is the nonlinear gain above — it fixes the
cause by keeping signal magnitude constant, but changes the representation every EARNED row was
measured on. Three caveats before anyone does it:

- **It is a selection-dynamics change touching every existing result.** Every EARNED row ran
  at 0.85. This needs a re-baseline and would re-stale graduation rows — treat it like adding
  a modality channel, not like a config tweak.
- **These are synthetic measurements** with uncorrelated SHA bases and iid noise. Real sensors
  correlate (hunger and fatigue drift together), which changes the geometry. Confirm on a real
  body before relying on the numbers.
- **It makes the EC scan load-bearing** *(corrected 2026-09-03: initially filed as "makes
  D51 load-bearing", but `LSHIndex` — D51 proper — is not on the `pattern_complete_or_separate`
  path; the cost lands on the unindexed `_substrate_nodes` scan, an exact O(N_nodes · d)
  Python loop with no cap and no pruning (2.7 ms @ 100 nodes, 136 ms @ 5,000 — per encode,
  per channel, per tick). Measured at A4's allocation rate 2026-09-03: verdict
  index-prerequisite; see `scripts/ec_scan_cost.py` +
  [world_seam_1_1_4.md](world_seam_1_1_4.md) §PR 0 result)*. Raising the sensor count
  without addressing cluster-count growth trades a representation ceiling for a latency one.

### Two traps to design against, both verified

- **`is_sim_mode` takes the lightweight session close.** A long Minecraft run would silently
  skip consolidation. Set `is_sim_mode=False` for benchmark runs, or the thing being measured
  never persists. (1.1.1 made the un-opened case loud; this case is a *correctly* opened
  session closed on the wrong path.)
- **The LLM already knows what lava is.** Any "learned aversion" in Minecraft is confounded
  by model priors unless the arm runs **substrate-primary** — the Exp 37/38/40 Goldilocks
  finding applies directly, and substrate-primary has an EARNED row.

## The 1.2 benchmark — four arms

> **Pre-registered 2026-09-05:**
> [exp56_four_arm_sharing_preregistration.md](../experiments/protocols/exp56_four_arm_sharing_preregistration.md)
> operationalizes this section (gates carried verbatim; the contingency is a
> TEACHER-DELIVERED reward — the Exp 52 operant mechanism on the world channel, the
> only shipped path that keys an exteroceptive cluster with positive credit; donors run
> a balanced action schedule so the causal-link channel is preference-neutral by
> construction; live-bridge apparatus; Phase-0 instrument checks against L11/L12/L1/L2).
> Its pre-freeze two-lens review round is designed against in §Links/§Selector of the
> prereg itself. The D43/D44 blockers named below are DISCHARGED (1.1.3 + the 1.2
> ingestion adapter, PR #637) — ingestion in the arms runs `maxim substrate ingest`,
> never bare `nac_merge`.

**Claim under test:** *agent A's learned representation changes agent B's behaviour, where A
and B are genuinely independent agents* — different `agent_id`, independently encoded EC.
Independence is the whole point: the two existing federation experiments pass only because
every infant shares one `agent_id` and one encoder (bugs ledger **D44**).

One contingency only A can experience (a hazard that costs health, or a reward source).

| Arm | Setup | Purpose |
|---|---|---|
| 1 **isolated** | B alone, never exposed | Floor |
| 2 **merged-taught** | B + A's bundle | The claim |
| 3 **merged-satiated** | B + a bundle from an agent that learned nothing | Controls for "a bundle arrived", not "a want arrived" |
| 4 **dangling-half** | `nac.json` merged without `ec.json` | Must reproduce the D43 silent-zero; proves the effect needs both halves |

**Dependent measure:** B's **first-contact** action choice (and approach latency) on A's
contingency, never having experienced it.

**Pre-registered gates:**

- merged-taught ≥ **0.70**
- merged-taught − isolated ≥ **0.20**
- merged-taught − merged-satiated ≥ **0.20**
- dangling-half ≈ isolated (within the isolated arm's spread) — the falsifier

**Power:** n ≥ 50 per arm in Minecraft; hardware replication at n = 12 on two Reachy Minis.

**Blocked on:** bugs ledger **D43** (the re-keyed merge, both halves) and **D44** (a merge
test that asserts behaviour). Until those land, arm 2 is *guaranteed* to read out as arm 1 —
running the benchmark first would produce a confident null with a known cause.

## The follow-on question worth its own experiment

Arms 2 and 3 above both share *memory*. The genuinely novel contrast is **perception-sharing
vs memory-sharing**: fan A's raw percepts into B live (≈50 LOC in-process, no mesh transport
needed) and ask whether streamed experience buys anything the merged substrate does not — a
faster acquisition curve, a different asymptote, or nothing at all. That is the question the
two-robot framing was reaching for, and it is affordable here and nowhere else.

Note that live percept transport does not exist at any layer today: `MeshMessageType.PERCEPT_PUSH`
is an enum slot with zero producers, `mesh/bus.py::LocalMessageBus` is in-process only, and a
docstring in `simulation/sources.py` names a file (`agents/remote_percept_source.py`) that was
never written (bugs ledger **D46**). In-process fan-out sidesteps all of it.


## The dose–response ladder — does collective learning *scale*? (added 2026-08-30)

The four-arm benchmark above asks **does it transfer**. This asks **does it scale**, which
is the claim Oasis actually rests on and the harder one to obtain by accident.

### The precedent, and exactly what it did and did not establish

`scripts/orient_substrate/5_operant_creche_federation.py` and its graded sibling
`7_graded_creche_federation.py` already run this design against the real
`hivemind/merge.py::nac_merge`, with committed records under
`docs/experiments/data/scripted_rederivation_2026-08-24/`. Four arms, and critically the
right control:

| Arm | Meaning |
|---|---|
| `single_partial` | one infant, K ticks — the per-agent floor |
| `single_full` | one infant, **N×K ticks** — "all the experience in one agent" |
| `creche_taught` | N infants × K ticks, merged — federation |
| `creche_none` | N infants × K ticks, no mother, merged — pooling noise ≠ signal |

Results: **0.73 / 1.00 / 1.00 / 0.51** (exp 5) and **0.59 / 1.00 / 1.00 / 0.16** (exp 7).

**What this establishes:** pooling sample-limited learners recovers the full-experience
policy, and the merge pools *learning* rather than noise. `single_full` is the control that
makes the claim non-trivial — without it, "N agents did better" is just "N× more experience
happened", which nobody doubts. That control was there from the start; keep it.

**What it does NOT establish, and why the ladder is a new experiment rather than a re-run:**

1. **It measures endpoint, not rate — and the endpoint is saturated.** `creche_taught` and
   `single_full` are both exactly **1.00**. A saturated measure cannot say whether the crèche
   learned *faster*, nor detect a moderate regression. This repo has been bitten by this
   twice: Exp 41 went VOID on a floored harm-rate metric, and Exp 42b's `safe_pref`
   saturated at SD 0.000 with the explicit caveat that it "cannot detect a moderate
   regression". **Choose the statistic before the arms.**
2. **The infants share a perceptual encoder by construction.** The script says so and
   defends it (same cochlea → same clustering), then states the limit itself: *"Fully
   independent agents encode to DIFFERENT uuid clusters and need `ec_merge` alignment
   first."* So it proves pooling works *where cluster ids already align* — the one
   configuration in which bugs ledger **D43** cannot fire.
3. **It is scripted, not the runtime.** Hand-built `NAc` + `Embodiment`, no agent loop, no
   LLM. A proof of concept for the merge arithmetic, not for the system.
4. **N is fixed.** One rung is a contrast; a curve is a mechanism.

### The design

**Independent variable:** N ∈ {1, 2, 4, 8} agents, each with a **fixed per-agent budget K**,
merged after their K trials. K is chosen so a single agent at K sits **partway up the curve**
(the original script's `single_partial` logic) and the criterion sits **below ceiling** — if
either saturates, the ladder measures nothing.

**Dependent measure: trials-to-criterion.** The first trial index at which a sliding window
of W trials sustains directedness ≥ the criterion. Endpoint directedness is reported but is
*not* the gate.

**Per rung, three conditions:**

| Condition | Purpose |
|---|---|
| `creche(N)` | N agents × K trials, merged |
| `single_matched(N)` | one agent given **N×K** trials — is pooling at least as efficient as one agent seeing everything? |
| `creche_none(N)` | N agents, no contingency, merged — noise floor at that rung |

**Pre-registered gates:**

- **Monotonicity (primary):** median trials-to-criterion strictly decreases across
  N = 1 → 8; negative rank correlation, p < 0.05.
- **Not-just-more-data:** at each rung, `creche(N)` trials-to-criterion ≤ `single_matched(N)`
  within a declared margin. If pooling is *worse*, that is an honest finding about merge
  cost and ships as one.
- **Noise floor:** `creche_none(N)` stays at chance at every rung.
- **Falsifier:** trials-to-criterion flat in N means collective learning buys nothing here.
  A flat curve is a result and ships as a result.

**Where it runs:** Minecraft, generalizing 1.1.4's two-AUT harness to **N-AUT**. N = 8 is
affordable there and will never be affordable on hardware; the robot replicates **one rung**,
not the ladder.

**Blocked on the same thing as everything else:** D43. Until the re-keyed merge lands, every
rung above N = 1 reads out as N = 1, and the ladder would produce a confident flat curve with
a known cause.

### Scope discipline: this is not an arm on every experiment

Adding an N-arm to an unrelated experiment tests a *different* hypothesis than that
experiment's pre-registered one, dilutes both, and multiplies cost — against the
one-frozen-confirmatory-test rule this project runs on. The ladder belongs where the
mechanism under test **is** collective learning. That is 1.2, and only 1.2.
