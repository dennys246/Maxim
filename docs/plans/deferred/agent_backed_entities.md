# Agent-backed entities — three-tier cognition, Cradle-trained cast, mesh-pressure budget

**Status:** **DEFERRED** (2026-04-28). Design preserved for post-1.0 revival.
**Scope:** ~600-900 LOC net new across `embodiment/`, `runtime/`, `imagination/`, `simulation/`, `peer/`, `interactive/`, plus a new `_data/cast/` directory and Cradle role-arc extensions.
**Depends on:** Cradle sensorimotor development ([../archive/cradle_sensorimotor_development.md](../archive/cradle_sensorimotor_development.md)) — shipped 2026-04-25 ([exp 11](../../experiments/11_cradle_sensorimotor_poc.md)); reactive peer mesh ([../reactive_peer_mesh_roadmap.md](../reactive_peer_mesh_roadmap.md)) C3.4+ for VRAM/lane telemetry; Wave 1 P4 multi-agent attribution (PR #202, shipped) for the per-agent isolation guarantees this plan depends on.
**Gates:** None.
**Branch:** TBD

---

## Why deferred

The motivating symptom — "sims feel one-way, narrator struggles with dragon embodiment" — has a simpler root cause that the much smaller [scene_actor_affordances.md](scene_actor_affordances.md) plan addresses directly. Imagination already generates dragon entities with `breathe_fire` affordances; the gap is that scene-entity affordances are declarative-only by design (per the entity ownership ship). Letting the orchestrator invoke them with the AUT as target — plus a `target_effect` field on `AffordanceSchema` — should produce embodied-feeling adversaries at ~110 LOC, leveraging all existing imagination work.

**Revive when:** scene_actor_affordances ships and the lived-experience gap remains. The remaining gap would be entity *agency* — entities deciding when to act, learning from outcomes, accumulating cross-session identity. That's what this plan addresses, and at that point the cost is justified.

**Other revival triggers:**
- Minecraft live demo benchmarking against Voyager / GITM / SPRING shows that scripted-but-embodied entities are insufficient for competitive comparison (e.g., zombies need to pathfind, villagers need to remember player trades).
- Variety ceiling becomes quantifiably measurable (behavioral repertoire metric, see `deferred/mother_npc_stimulus_plan.md`) and shipped scripted-actor refinements aren't enough.
- A user-facing recurring-character demand emerges (DM campaigns where consistent antagonists matter; storytelling sims with named cast).

---

## Vision

Three layered claims:

1. **Not every "alive" entity should be a Maxim.** Most NPCs (a guard at a door, a peasant in a market, a random goblin) are fine as orchestrator-narrated SEM data. A few — recurring antagonists, named characters, creatures whose multi-turn behavior should learn from outcomes — earn a full bio-stack. We make the decision explicit at spec time via a `cognition` field.

2. **Recurring characters carry persistent identity, trained in the Cradle.** Today's Cradle ([../archive/cradle_sensorimotor_development.md](../archive/cradle_sensorimotor_development.md)) trains an infant body through developmental acts. Extend it: role-specific Cradles produce *cast members* — Maxim agents with frozen baseline bio-state (Hippocampus, NAc, ATL, EC, SCN), persona, body, and learned behaviors. Loaded into sims as agent-backed scene entities. A "Dragon Cradle" trains Vorthax: thermal-immune body, gold-acquisition reinforcement → hoarding behavior emerges from NAc reward bias, not prompt engineering.

3. **Mesh capacity is the binding constraint, not RAM.** Agent-backed entities all queue on the same `LLMRouter`. The mesh already publishes drain state, VRAM headroom, and dispatch latency. We expose a `MeshPressure` signal that ImaginationTrigger consults before promoting an entity to `cognition: agent` — when the mesh is hot, novel imagined creatures stay narrated; when the mesh is cool, they can become full agents. This makes the cognition budget an emergent property of mesh state, not a hard-coded limit.

**Research framing for 1.0+:** "Cross-session character identity emerges from cradle-trained bio-state, not prompt engineering. A dragon trained to hoard treasure does so because its NAc reward biases were shaped during development, the same way the AUT learns to avoid fire."

---

## Current state — concrete findings (2026-04-28 deep-dive)

A four-agent parallel investigation produced these facts. Summarized here.

### What works

- **Bio-system isolation per agent is real.** Each `AgentFactory.create_full_agent()` produces own Hippocampus, NAc, ATL, EC, SCN, Cerebellum, MemoryHub, PainBus, ReactionBus, persisted at `~/.maxim/agents/{agent_id}/` ([agent_factory.py:495-585](../../../src/maxim/runtime/agent_factory.py#L495-L585)).
- **Multi-agent attribution gaps were closed by Wave 1 P4** (PR #202). The two `bio_integration.py` global stashes (`_latest_substrate_nodes`, `_latest_pain_intensity`) are now per-agent dicts; `tool_dispatch.record_outcome()` carries `agent_id`. This was Stage 0 of the original draft — extracted and shipped independently.
- **Agent identity infrastructure is ~90% complete.** `register_agent_nickname()` registers human-readable names. `sim_log` records carry both `agent_id` and `agent` (nickname). `RequestContext` propagates `X-Maxim-Agent-Id` on internal endpoints. Display `_LogEntry` tracks per-line agent attribution; focus cycling exists.
- **Cradle infrastructure already trains Maxim through developmental acts.** `BUILTIN_ARCS["cradle"]`, `bodies/infant_humanoid.yaml`, drives, reflexes, 7 stages — all shipped 2026-04-25. The output (persisted bio-state) is exactly the format a "cast template" needs.
- **Mesh telemetry exists.** `/v1/debug/vram` (C3.4), drain state (C2/C4/C4.5), per-call `peer_backend_call` JSONL trace (Plan 3 R2.5), `dispatch_exhausted_all_drained` event. Reading these into a pressure signal is plumbing, not new instrumentation.

### Gaps

| Gap | Why it matters | Est. LOC |
|---|---|---|
| **G1. No `cognition` field on Entity spec.** | Today every entity is implicitly "passive" (data + AUT-callable affordances). No way to declare "this should be an agent." | ~10 LOC + YAML schema |
| **G2. No NPC agent runtime.** | Only AUT and orchestrator have agent loops today. Spawning a third+Nth `run_agentic_loop` thread per agent-backed entity has no scaffolding. | ~150 LOC |
| **G3. AUT prompt does not include registered nickname.** | [prompt_builder.py:1205](../../../src/maxim/agents/prompt_builder.py#L1205) uses `state.name` (persona-derived), not the registered nickname. So "You are Vorthax" is missing — agent doesn't know its own identity. | ~10 LOC |
| **G4. No cast template format or load path.** | No `_data/cast/` directory; AgentFactory does not load from frozen template; copy-on-write semantics undefined. | ~200 LOC + cast YAML |
| **G5. No mesh pressure aggregator.** | Telemetry exists but no `MeshPressure` object reading it and producing a 0.0–1.0 pressure scalar. | ~120 LOC |
| **G6. ImaginationTrigger has no cognition-budget gate.** | Today it always registers as scene entity ([trigger.py:755-783](../../../src/maxim/imagination/trigger.py#L755-L783)). Branching on `spec.cognition` + mesh pressure does not exist. | ~50 LOC |
| **G7. Display has no per-agent color or visual lane.** | Filtering exists; rendering is single-pane. Per-agent color prefix is the smallest visual delta worth shipping. | ~40 LOC |
| **G8. No Dragon Cradle arc.** | Cradle today is single-arc (`infant_humanoid` → developmental acts). Need extensible role-arc registration so each cast member has its own training Cradle. | ~150 LOC + YAML |

**Total: ~730 LOC + cast YAML + Cradle role-arcs.** Most stages are independently shippable. Wave 1 P4 already shipped the multi-agent correctness foundation, removing it from this plan's critical path.

---

## Design

### D1. Three-tier cognition spec

Add `cognition: "passive" | "narrated" | "agent"` to entity YAML, default `"passive"`. Concrete semantics:

```yaml
# _data/components/items/sword.yaml
name: rusty_sword
entity_type: weapons
cognition: passive          # default — pure SEM data, AUT calls affordances as tools

# _data/components/npcs/guard.yaml
name: guard
cognition: narrated         # SEM data + LLM-narrated dialogue (today's NPC behavior, made explicit)

# _data/cast/dragons/vorthax.yaml
name: vorthax
entity_type: creatures
cognition: agent            # full Maxim with cradle-trained baseline
cast_template: dragons/vorthax  # frozen bio-state location
```

`SpecModulator._parse_cognition` sets `Entity.cognition` (new field). EntityMap gains a third ownership category: `register_agent_backed(entity, npc_agent)` alongside `register_self`/`register_scene`. Tool registration for agent-backed entities is *not* shared with the AUT — the AUT can `sense` and `examine` an agent-backed dragon, but cannot call its affordances directly. Affordances fire via the dragon's own loop.

**Default for imagined entities:** `cognition: narrated`. Designer.imagine returns `cognition: narrated` unless the prompt explicitly requests an agent (rare; typically reserved for hand-curated cast).

### D2. Cast template format

A cast template is a directory under `_data/cast/{type}/{name}/`:

```
_data/cast/dragons/vorthax/
  manifest.yaml          # name, persona, body_ref, cradle_arc, version, hash
  body.yaml              # SEM Entity spec (or ref to existing component)
  hippocampus.json       # frozen bio-state from Cradle training
  nac.json               # NAc reward biases — the load-bearing learned behavior
  atl.json               # semantic concepts learned during Cradle
  scn.json               # circadian phase patterns
  cerebellum.json        # forward models
  prompt.md              # persona/voice prompt (overlay on agent system prompt)
```

**Persistence model:** templates are **immutable**. When a sim spawns Vorthax, AgentFactory copies the template into a session-scoped agent dir (`~/.maxim/sessions/{session_id}/agents/vorthax_{instance}/`) and the instance learns within that copy. Sim-end → instance dir is preserved (for analysis) but does NOT promote back to the template.

**Promotion workflow (deferred to Stage 7):** an explicit `maxim cast promote --from <session>/<agent> --to dragons/vorthax` verb to merge a session's learning back into the template. Manual gate prevents identity drift across uncoordinated sims.

### D3. Cradle role training — produce the cast member

Generalize the existing Cradle so it can train *any* role, not just `infant_humanoid`. The current Cradle is `(arc=BUILTIN_ARCS["cradle"], body="infant_humanoid")`. Extend to:

```bash
maxim --sim cradle --embodiment bodies/infant_humanoid    # today (general infant)
maxim --sim cradle --cradle-role dragon --embodiment creatures/dragon_hatchling  # new
```

A role-arc is a tuple `(NarrativeArc, body_template, reinforcement_overlay)`. The reinforcement overlay shapes what gets reward/pain during training:

```yaml
# _data/cradle_roles/dragon.yaml
name: dragon
body: creatures/dragon_hatchling
arc:
  - act: "hatching"        # learn body, find warmth (heat = pleasure, not pain — overlay)
    phases: [...]
  - act: "first_flight"    # motor calibration via cerebellum
    phases: [...]
  - act: "first_hoard"     # gold acquisition triggers strong positive valence
    phases: [...]
  - act: "territorial"     # intruders trigger negative valence
    phases: [...]
reinforcement_overlay:
  body_modifications:
    - failure_mode: thermal_overload
      action: omit            # dragons don't experience fire damage
  reward_overlay:
    pick_up:
      - if: "entity.metadata.valuable == true"
        valence: +0.8         # strong positive — drives hoarding
    proximity:
      - if: "entity.entity_type == 'intruders'"
        valence: -0.5         # territorial avoidance
```

The reinforcement overlay is applied at the body construction layer (`failure_mode.action: omit` removes the spec) and at the reward layer (NAc receives the overlay's valence, not just the default body-derived signal). End of cradle → freeze bio-state → save to `_data/cast/dragons/{name}/`.

**Why the Cradle, not handcrafted NAc state:** the user's framing is exactly right — hand-writing reward biases produces brittle, context-free preferences. Cradle training produces *grounded* biases tied to specific concept embeddings (gold → bright + small + cold → reward), which transfer to novel-but-similar percepts via the substrate path (silver, gems, jewels) without re-training. This is the same affordance-concept-transfer mechanism shipped 2026-04-24, applied to character training.

### D4. Mesh pressure budget

A single function: `MeshPressure.cognition_budget() -> int` returns the maximum number of agent-backed entities a sim can host *right now*.

```python
# src/maxim/peer/mesh_pressure.py (new)
@dataclass(frozen=True)
class PressureSnapshot:
    inflight_ratio: float       # in_flight_requests / lane_capacity
    latency_ratio: float        # recent_p95 / target_p95
    vram_ratio: float           # 1 - vram_headroom_fraction (from /v1/debug/vram)
    drain_ratio: float          # drained_nodes / total_nodes
    # composite
    pressure: float             # max of the above, clamped [0.0, 1.0]

class MeshPressure:
    def snapshot(self) -> PressureSnapshot: ...
    def cognition_budget(self, base: int = 4) -> int:
        """Returns max concurrent agent-backed entities. base * (1 - pressure), floor 1."""
```

Pressure inputs come from existing telemetry (no new instrumentation):
- `inflight_ratio`: `LLMRouter._inflight_count` / lane capacity (per-tier).
- `latency_ratio`: rolling p95 from the existing `peer_backend_call` JSONL trace.
- `vram_ratio`: `/v1/debug/vram` ratio aggregated across reachable nodes.
- `drain_ratio`: `drained_nodes` / `mesh.yml::nodes` count.

`ImaginationTrigger._resolve_phrase` consults `MeshPressure.cognition_budget()` before promoting a designed entity to `cognition: agent`. If the budget is full or the design's tier requires more capacity than the mesh can provide, it downgrades to `cognition: narrated`. Logged as `cognition_downgraded` event with `reason: pressure_exceeded`.

**Per-tier sizing:** large-tier agents (full 14B Maxims) are expensive; small-tier agents (cradle-trained 7B Maxims) are cheap. `cognition_budget` returns a per-tier breakdown:

```python
{
    "large": 1,   # AUT only when pressure is high
    "medium": 2,  # one agent-backed antagonist + AUT
    "small": 4,   # several lightweight NPCs
}
```

Cast manifest declares its tier; designer always defaults imagined agents to small-tier first.

### D5. NPC agent runtime

A new `runtime/npc_pool.py` that runs N agent-backed entities alongside the AUT and orchestrator. Each NPC runs its own `run_agentic_loop` on a thread, with its own:

- AgentFactory-built bio-stack (loaded from cast template if present)
- Percept source: `EntityProximityPerceptSource` — feeds the NPC percepts about the AUT and other entities in scene proximity (analogous to ConversationalSource)
- Action sink: outputs feed back into the orchestrator's action stream (so the orchestrator can narrate what the dragon does, plus any direct effects on AUT body via shared embodiment)
- LLM call cadence: NPCs do not tick every AUT turn. They tick on **trigger** (AUT enters scene, AUT acts on NPC, drive crosses threshold) or **slow timer** (every 30 sec idle). This is critical for LLM router contention.

**Tick scheduling:** an `NPCScheduler` queues NPC ticks and drains them between AUT turns, respecting `MeshPressure`. A dragon under low pressure ticks every AUT turn; under high pressure, it ticks only when triggered. The scheduler emits `npc_tick_skipped` events when pressure forces a skip.

**Prompt injection (G3 fix):** AUT and NPC system prompts both include the registered nickname:

```
You are Vorthax, an ancient red dragon.
{persona_overlay from cast template}
Your body: {body_state_summary}
Recent memories: {bio_enrichment}
```

The single line `prompt_builder.py` change is to read `register_agent_nickname.get(agent_id)` instead of `state.name` when both are present.

### D6. Display + identity wiring

Smallest-shippable visual delta:

1. **Per-agent color in log prefix.** `_LogEntry.agent` already exists. Add a deterministic `agent_color(nickname)` mapping (hash → palette of ~12 colors). Render `[{color}][{agent:>8}][/]` instead of the current `bright_cyan`-only prefix.
2. **Roster panel.** Tiny new panel in the display's status bar listing active agents (`AUT (red), Orch (gray), Vorthax (green)`).
3. **Agent focus already cycles** — confirmed working ([display.py:515-557](../../../src/maxim/interactive/display.py#L515-L557)). Just needs the roster panel to make discoverable.
4. **Thinking panel attribution.** Already supports `agent: str | None` ([display.py:430-485](../../../src/maxim/interactive/display.py#L430-L485)). Verify it renders the focused agent's deliberation correctly with multiple agents active.

Per-agent visual lanes (split panels) are deferred — the color+filter combo gives most of the readability win at 10% of the implementation cost.

---

## Stages

### Stage 0 — Pre-req plumbing — **EXTRACTED to [v1_refinement.md P4](../archive/v1_refinement.md), SHIPPED in PR #202**

The two correctness gaps (`bio_integration.py` globals + `tool_dispatch.record_outcome` agent_id) were independently necessary regardless of cast direction, so they shipped as part of the 1.0 cleanup pipeline rather than blocking on this plan's revival. Done.

### Stage 1 — Cognition spec + EntityMap agent-backed category

- **1.1.** Add `cognition: Literal["passive", "narrated", "agent"]` to `Entity` dataclass in `sem.py`. Default `"passive"`. Update `SpecModulator._parse_*` to read the field.
- **1.2.** Add `EntityMap.register_agent_backed(entity, agent_id)` alongside `register_self`/`register_scene`. Self-tools and scene-tools logic unchanged; agent-backed entities expose `sense` + `examine` to the AUT but their affordances are NOT registered to the AUT's tool registry.
- **1.3.** Add `Entity.metadata["agent_id"]` populated when registered as agent-backed.

**Exit:** YAML with `cognition: agent` is parseable and routable through EntityMap; AUT can still observe the entity but cannot call its affordances. ~80 LOC + tests.

### Stage 2 — NPC agent runtime + name-aware prompts

- **2.1.** New `runtime/npc_pool.py` with `NPCAgent` (wraps `MaximAgent`) and `NPCScheduler`. NPC runs `run_agentic_loop` on its own thread. Reuses existing `build_bio_stack`, `AgentFactory.create_full_agent` machinery. **No new bio-system construction code** — just composition.
- **2.2.** New `EntityProximityPerceptSource` — feeds the NPC percepts about scene entities and AUT actions within proximity. Mirrors `ConversationalSource`'s contract.
- **2.3.** NPC action sink routes back to orchestrator; orchestrator narrates NPC actions in the AUT's percept stream.
- **2.4.** Fix G3: `prompt_builder.py` reads registered nickname when present, falls back to `state.name`. Adds "You are {nickname}." line to the system prompt.
- **2.5.** EntityMap.register_agent_backed wires the NPC into the orchestrator's tick loop.

**Exit:** a hand-curated test cast member (no Cradle yet) loads, ticks, observes the AUT, and produces actions visible in the orchestrator log. ~200 LOC + tests.

### Stage 3 — Mesh pressure cognition budget

- **3.1.** New `peer/mesh_pressure.py` with `PressureSnapshot` + `MeshPressure.snapshot()` + `cognition_budget(base, tier)`. Reads existing telemetry: `LLMRouter._inflight_count`, `peer_backend_call` JSONL p95, `/v1/debug/vram`, drain state.
- **3.2.** Wire `MeshPressure` into `ImaginationTrigger._resolve_phrase`: when designer returns `cognition: agent`, check budget; if over, downgrade to `narrated` and log `cognition_downgraded`.
- **3.3.** `NPCScheduler` consults `MeshPressure` to decide tick cadence — ticks defer when pressure is high.
- **3.4.** Add `maxim doctor mesh-pressure` subcommand that prints current snapshot for operator visibility.

**Exit:** under simulated load (e.g., second peer making concurrent calls), imagined dragons downgrade to `narrated`; under cool mesh, they upgrade to `agent`. Verified by inspecting `cognition_downgraded` event count. ~150 LOC + tests.

### Stage 4 — Cast template format + load path

- **4.1.** Define `_data/cast/{type}/{name}/manifest.yaml` schema. New `cast_loader.py` parses it.
- **4.2.** `AgentFactory.create_full_agent` accepts `cast_template: Path | None`. When set: copy frozen bio-state files from template into session agent dir, then construct as usual (existing JSON deserialization paths handle the rest — Hippocampus, NAc, ATL, EC, SCN, Cerebellum already load from JSON).
- **4.3.** Persona/voice overlay: cast manifest's `prompt.md` is appended to the agent system prompt under a clear heading.
- **4.4.** Hand-curate a minimal test cast member (no Cradle training yet — just frozen empty bio-state + persona) to validate the load path.

**Exit:** `maxim cast list` shows available cast; spawning a sim with `--cast dragons/test_dragon` loads the agent with persona prompt active. ~200 LOC + tests + first cast YAML.

### Stage 5 — Dragon Cradle (the proof point)

- **5.1.** Generalize Cradle: `maxim --sim cradle --cradle-role <name>` reads `_data/cradle_roles/{name}.yaml` instead of hardcoded `infant_humanoid` arc.
- **5.2.** Implement reinforcement overlay parser: `body_modifications` (omit failure modes, override sensors), `reward_overlay` (NAc valence overrides at specific affordance/entity matches).
- **5.3.** Author `_data/cradle_roles/dragon.yaml`: 4 acts (hatching, first_flight, first_hoard, territorial). Author the supporting scene entities (warm_nest, gold_pile, intruder_npc).
- **5.4.** Run the Dragon Cradle end-to-end. Validate: thermal failure modes are omitted, gold-pickup produces strong positive NAc bias, intruder-proximity produces negative bias. Save bio-state to `_data/cast/dragons/vorthax/`.
- **5.5.** Spawn Vorthax in a non-cradle sim (e.g., `--sim "explore the dragon's lair" --cast dragons/vorthax`). Verify: dragon's behavior reflects Cradle-learned biases (approaches gold, avoids intruders, ignores fire) WITHOUT prompt engineering for those behaviors.

**Exit:** Vorthax demonstrates Cradle-trained character identity in a fresh sim, evidenced by NAc-driven action selection diverging from a control "untrained dragon" agent given the same prompt. This is the research demo. ~250 LOC + cradle YAMLs + cast.

### Stage 6 — Display polish

- **6.1.** Per-agent deterministic color (`hash(nickname) % palette_size`).
- **6.2.** Roster panel in display status bar.
- **6.3.** Multi-agent thinking panel rendering verification.

**Exit:** running a sim with AUT + Vorthax + Orch shows three distinct color-prefixed log streams; cycling focus filters cleanly. ~50 LOC + visual verification.

### Stage 7 — Cross-session learning policy (deferred)

- **7.1.** `maxim cast promote --from <session>/<agent> --to <type>/<name> [--review]` verb. Manual gate.
- **7.2.** Diff visualization (NAc reward biases changed, new ATL concepts, new hippocampus episodes) so the operator can review before promoting.
- **7.3.** Versioning: cast templates carry a hash; promoted versions are appended (`v1`, `v2`) not overwritten.

Deferred until Stages 1-6 ship and we have actual session-learning to evaluate.

---

## Load-bearing invariants

(In the spirit of the lessons in CLAUDE.md — these are the rules to push DOWN into types when forgetting causes silent bugs.)

1. **Cast templates are immutable.** Any code path that writes to `_data/cast/` must go through the explicit `cast promote` verb. Filesystem-level write attempts during sims are a `RuntimeError`. Enforce via path check in `cast_loader.py`.

2. **`build_npc_agent` requires `cast_template` keyword-only OR `agent_spec` keyword-only — XOR.** Forgetting which path is intended is a `TypeError`. Pattern matches the `build_executor(pain_bus=...)` and `build_pain_bus(hippocampus=..., nac=...)` precedents.

3. **`MeshPressure.cognition_budget()` returns 0 when telemetry is unavailable, not None.** Callers default to "no agent-backed spawning" rather than "unbounded." Fail safe.

4. **NPC tick deferral is logged, not silent.** Skipped ticks emit `npc_tick_skipped {reason}`. A silent skip is a debug nightmare; per the no-band-aid rule, surface the pressure signal so operators see it.

5. **AUT cannot call agent-backed entity affordances directly.** `EntityMap.register_agent_backed` does NOT register affordances to the AUT's tool registry. AUT interacts via observation (`sense`, `examine`) or via shared scene effects (damage, proximity). Enforces the "agent-backed entities have agency" semantic.

6. **Cradle role overlays apply at construction, not runtime.** A dragon's omitted thermal failure mode is not in the body spec at all — not "spec'd but suppressed." `evaluate_failures` cannot accidentally re-fire it. Mirrors the Plan 3 single-call rule: structurally impossible vs. happens-not-to-fire.

7. **Cast manifests carry an explicit `tier: "small" | "medium" | "large"` field.** Mesh pressure budget is per-tier. Defaulting to `"large"` would let one heavy cast member starve the AUT. Forgetting the field is a `KeyError` at load time.

---

## Risks and tradeoffs

**R1. LLM router contention is the binding constraint.** Even with `MeshPressure`, four agent-backed entities + AUT + orchestrator on a single 14B model is genuinely expensive. Mitigation: NPCs default to small tier; large-tier agent-backing is reserved for hand-curated boss-tier cast. The `cognition_budget(tier)` per-tier breakdown enforces this.

**R2. Cradle training cost is not free.** Producing one cast member requires running a full Cradle sim — minutes to hours. Mitigation: cast templates are reusable across sims; one Vorthax serves N campaigns. Promote-to-template is manual (Stage 7) so we don't accidentally pollute templates.

**R3. Identity bleed across concurrent sims.** If two sims fork Vorthax simultaneously and both try to promote, last-write-wins. Mitigation: Stage 7 promote is manual + diff-reviewed; we explicitly accept that session-learning is sandboxed unless promoted.

**R4. Persistence sprawl in `~/.maxim/sessions/`.** Each session-scoped agent dir is ~50-200 MB. Sessions accumulate. Mitigation: session GC policy (already part of v1_refinement docs cleanup); cast templates live in repo, not user dir.

**R5. The framing "alive ⇒ Maxim" is a category error.** This was the original pushback — most NPCs should be `narrated`, not `agent`. The plan addresses this by making `cognition` a 3-tier explicit field with `passive` as default, not by gating on metadata flags like `alive`. Worth restating in the plan exit checklist.

**R6. Mesh pressure aggregation is global, but cognition decisions are per-sim.** Two sims running concurrently both consult `MeshPressure.cognition_budget()` and both think "I have budget for 2 agents" — total is 4, oversubscribed. Mitigation: budget tracks cluster-wide reservations via the existing `~/.maxim/util/` mutable state pattern (per Plan 4 C2 invariant). Reserve at sim start; release at sim end.

**R7. Cradle role overlays are a new spec dialect.** Reinforcement overlays could grow to a complex DSL. Mitigation: keep it small in v1 (omit failure modes, scalar valence overrides on affordance matches). Resist DSL growth — match the `mesh.yml` discipline (the parser dialect is FROZEN).

---

## Open questions

1. **Should `cognition: narrated` be a real distinct tier, or just "passive but the orchestrator describes its actions"?** The latter is what we do today. The former implies a structured per-NPC prompt slot in the orchestrator's narration. Lean: keep `narrated` as today's behavior made explicit, no new infra. If we find we need structured per-NPC narration prompts, promote to a real tier later.

2. **Does an agent-backed entity get its own `aut_pain_bus`-equivalent, or share the AUT's?** Lean: each agent-backed NPC gets its own PainBus (same per-agent bio-system isolation principle). Shared scene damage events (e.g., AUT swings sword at dragon) propagate via the orchestrator's existing `DamageComponentTool` path.

3. **How does an agent-backed NPC perceive the AUT's actions — through orchestrator narration, or through a structured percept feed?** Lean: structured. `EntityProximityPerceptSource` feeds typed percepts (AUT entered scene, AUT used `pick_up` on `gold_pile`, etc.) rather than re-narrated text. Cleaner for bio-pipeline learning; orchestrator only narrates for the AUT's view.

4. **Cradle role training — how many acts is enough?** The infant Cradle has 4. Dragon-specific behaviors might need 6 (more hoard-accumulation reps). No hard answer; iterate empirically per role.

5. **Should cast templates version with the Maxim software version?** A bio-state file produced under 0.8 may not deserialize cleanly under 1.5. Lean: yes, cast manifest carries `maxim_version` + `compatible_versions` range; loader rejects incompatible. Composes with [v1_refinement.md CC1](../archive/v1_refinement.md) `_format_version` (shipped PR #203) — cast manifest's compatibility check uses the same versioning scheme.

6. **Mesh pressure when no peers are configured (solo mode)?** Telemetry is local-only. Lean: pressure is computed against local-lane capacity; solo mode just has lower budget defaults.

7. **Per-agent attribution under shared scene damage** (Wave 1 P4 composability): When the AUT swings a sword at Vorthax, the damage flows through `DamageComponentTool` → Vorthax's body. Does Vorthax's NAc record this with the correct `agent_id` (Vorthax's, not the AUT's)? Stage 5 verification: confirm `record_outcome` plumbs the resolved target's `agent_id` correctly.

---

## Why this is the right time (when revived)

- **0.8 shipped** entity ownership (self vs scene), affordance concept transfer, scene-scoped tool windows, imagination trigger. The cognition tier is the natural next layer above ownership.
- **Cradle infrastructure is built and proven** (PR #200, [exp 11](../../experiments/11_cradle_sensorimotor_poc.md)). Role-arc extension is a small delta vs building from scratch.
- **Mesh telemetry C3.4 + drain wiring is shipped.** `MeshPressure` is a reader, not new instrumentation.
- **Wave 1 P4 closed multi-agent attribution gaps** (PR #202). Per-agent bio-system isolation is now load-bearing rather than aspirational.
- **The research-claim ceiling for 1.0 is "cross-session learning without fine-tuning."** Cast members trained in the Cradle and re-loaded in non-cradle sims demonstrate this in the most dramatic form: a *character* trained once, deployed many times, retains identity across deployments.

This unblocks recurring-character sims (DM campaigns with consistent antagonists), variety in behavioral convergence experiments (the Mother NPC stimulus pattern in `deferred/mother_npc_stimulus_plan.md` becomes trivial — Mother is just a cast member), and a clean answer to "how do you build an NPC with personality?" — you train one in the Cradle.

---

## Cross-references

- [../scene_actor_affordances.md](scene_actor_affordances.md) — diagnostic for whether this plan is needed; revives this plan if it doesn't close the dragon-narration gap.
- [../archive/cradle_sensorimotor_development.md](../archive/cradle_sensorimotor_development.md) — Cradle infrastructure this plan extends. Shipped 2026-04-25.
- [../reactive_peer_mesh_roadmap.md](../reactive_peer_mesh_roadmap.md) — Mesh telemetry this plan reads.
- [../v1_refinement.md](../archive/v1_refinement.md) P4 — multi-agent attribution shipped PR #202; load-bearing for this plan's per-agent semantics.
- [../v1_refinement.md](../archive/v1_refinement.md) CC1 — `_format_version` shipped PR #203; cast manifest compatibility checks compose with this scheme.
- [mother_npc_stimulus_plan.md](mother_npc_stimulus_plan.md) — Stimulus pattern subsumed by cast members.
- [mother_maxim_plan.md](../archive/mother_maxim_plan.md) — Persistent collective memory; related but distinct (Mother Maxim is shared infrastructure; cast members are individuals).
- [llm_path_multi_peer_dispatch.md](llm_path_multi_peer_dispatch.md) — Multi-peer load distribution. Sister concern to `MeshPressure`.
- [../minecraft_benchmark.md](minecraft_benchmark.md) — revival trigger if Minecraft mobs need pathfinding / villagers need trade memory.
- CLAUDE.md "Plan review round runs BEFORE PR merge" — applies. Two-lens review (Executor + Architecture) before each stage's PR.
