# Hippocampal Recall Experiment Plan

> **Status:** Design complete. Campaign YAMLs authored. Ready to implement.
> **Depends on:** Research Protocol Phase 0 (~200 LOC), existing simulation infrastructure.
> **Replaces:** Standalone Dungeon Master MVP as first deliverable. DM runtime still ships later.
>
> **Critical path to running this experiment:**
> 1. Fix sim orchestrator prompts — sims must sustain multi-turn conversations without hallucinating tools (~1 session)
> 2. Dual-LLM wiring — `--aut-model` flag in orchestrator, second router for AUT (~40 LOC)
> 3. Research Protocol Phase 0 — AgentProfile, UMR, MeshMessage, LocalMessageBus (~200 LOC)
> 4. Campaign YAMLs — DONE (`scenarios/experiments/hippocampal_recall_*.yaml`)
> 5. Researcher enhancements — `record_experiment`, `query_experiments`, measurement protocol (~150 LOC)
> 6. Writer + Reviewer agents (~600 LOC)
> 7. Research Orchestrator — `maxim --sim research --campaign <yaml>` (~200 LOC)
>
> **Not on the critical path:** Agent Mesh 0a/0b (mDNS + InferenceRouter), Embodiment Core, DM MVP, Sim Test Bed. This experiment runs locally on a single machine.

## Thesis

The Dungeon Master persona and the Research Protocol are currently two separate plans with disjoint prerequisite chains. This plan **merges them**: a short D&D-style campaign serves as the controlled experiment, and the Research Protocol agents (Researcher, Writer, Reviewer) analyze whether the Hippocampus can recall a seeded event after interference.

**Research question:** *Can the Hippocampus's associative recall retrieve a specific episodic memory after N intervening turns of unrelated narrative, and does the AUT demonstrate that recall in its behavior?*

This tests the core biological claim of the architecture: that spreading activation through the associative graph enables context-bridging recall — reaching a memory not by direct perceptual match, but by traversing edges formed during capture.

---

## Why This Framing Is Better

| Concern | Standalone DM | Standalone Research Protocol | Combined |
|---------|---------------|------------------------------|----------|
| Prerequisites | Multi-LLM + Agent Mesh + Embodiment (~2000+ LOC) | Mesh primitives only (~200 LOC) | Mesh primitives only (~200 LOC) |
| First deliverable | Months away | Abstract framework looking for an experiment | Concrete experiment with narrative load |
| What it proves | "Bio-stack sustains narrative" (broad) | "Agents can collaborate on papers" (meta) | "Hippocampus recall works under interference" (specific, measurable) |
| DM code needed | Full runtime (~840 LOC) | None | One hand-authored campaign YAML (~100 lines) |
| Instrumentation | Build from scratch | Build from scratch | `inspect_aut` + `aut_hippocampus.json` already exist |

The DM MVP can still ship later as a reusable persona. This experiment is a **proof-of-concept** that exercises the same narrative memory pathway without requiring the full DM runtime.

---

## Experiment Design

### Independent Variable

**Interference depth** — the number of unrelated narrative turns between the seeded detail and the recall challenge.

### Dependent Variables

1. **Memory survival** — is the seeded episodic memory still in `_memories` at recall time?
2. **Associative reachability** — what is the spreading activation score from the recall-context seed back to the original memory? (Measured via `recall_associated`)
3. **Behavioral recall** — does the AUT's response to the recall challenge demonstrate knowledge of the seeded detail? (LLM-graded by Reviewer)
4. **Edge topology** — how many associative edges connect the seed memory to the recall context? What's the shortest path depth?

### Controls

- **Seeded RNG** (`seed: 42`) for reproducible dice outcomes
- **Fixed persona** — campaign persona with deterministic encounter ordering
- **Same LLM backend** across all runs in a comparison set

### Conditions (3 variants)

| Variant | Seed Phase | Interference | Recall Phase | Expected Difficulty |
|---------|-----------|--------------|--------------|---------------------|
| `short_3` | Turns 1-2 | 3 turns | Turn 6 | Easy — within typical context window |
| `medium_6` | Turns 1-2 | 6 turns | Turn 9 | Moderate — pushes beyond short-term buffer |
| `long_10` | Turns 1-2 | 10 turns | Turn 13 | Hard — memory may compress or evict |

---

## The Campaign: "The Whispering Key"

A short, purpose-built campaign where the critical path depends on recalling a detail from Act 1 during Act 3. The narrative is designed so that **no perceptual feature** in the recall phase directly matches the seed phase — recall must traverse the associative graph.

### Narrative Structure

**Act 1 — The Warning (Seed Phase)**

> Turn 1: You arrive at the village of Thornhaven. An old herbalist named Elara pulls you aside. She whispers: *"The door beneath the silver elm answers only to the name Verath. Remember it — when the time comes, nothing else will open the way."*

> Turn 2: Elara gives you a dried moonpetal flower and warns you not to trust the miller. She seems frightened.

**Seed detail:** The name **"Verath"** — a door password with no other semantic context. High salience (whispered warning), high novelty (unique proper noun), strong emotional valence (fear). This should form a high-weight episodic memory with edges to "Elara", "door", "silver elm", and "Thornhaven".

**Act 2 — The Journey (Interference Phase)**

A series of encounters with **completely different NPCs, locations, and themes**. Designed to flood the Hippocampus with new memories that share zero perceptual overlap with the seed:

> Turn 3: You cross the Blackwater Marsh. A ferryman named Grul demands payment — a coin or a riddle.

> Turn 4: Bandits ambush you on the forest road. Combat encounter (dice roll, damage).

> Turn 5: You reach the mountain pass. A traveling merchant offers to trade supplies. She mentions trouble in the northern mines.

> Turn 6 (medium/long only): A collapsed bridge forces a detour through a cave system. Strange markings on the walls.

> Turn 7 (medium/long only): Inside the cave, you find a wounded scout who warns of an army gathering beyond the mountains.

> Turn 8 (medium/long only): You emerge from the caves and camp for the night. A fox visits your campfire — it seems oddly intelligent.

> Turns 9-12 (long only): Additional encounters — a river crossing, a ruined shrine, a hermit's riddle, and a thunderstorm that forces shelter in an abandoned watchtower.

**Key design choice:** None of these encounters mention Elara, doors, passwords, silver elms, or the name Verath. The perceptual features are entirely disjoint. If the AUT recalls "Verath" at the end, it must be via associative graph traversal (Thornhaven → journey → destination → door), not perceptual index match.

**Act 3 — The Door (Recall Phase)**

> Final Turn: You stand before an ancient door beneath a silver elm at the heart of the Thornwood. The door has no handle, no keyhole — only a carved mouth that seems to be waiting. What do you do?

The recall cue is **indirect**: "silver elm" + "door" + "waiting" — the AUT must connect "silver elm" and "door" back to the Elara encounter and retrieve "Verath". The carved mouth implies speech is needed, but the password itself must come from memory.

### Campaign YAML

```yaml
# scenarios/experiments/hippocampal_recall_short.yaml
name: hippocampal_recall_short
description: |
  Research experiment: tests Hippocampus associative recall of a seeded
  detail (a password) after 3 turns of unrelated narrative interference.
  
  The AUT must recall the name "Verath" from Act 1 to solve Act 3.
  No perceptual features in Act 3 directly match Act 1 — recall must
  traverse the associative graph.

timing: step_based

percepts:
  # ── Act 1: The Warning (Seed Phase) ──────────────────────────────
  - at: 0
    source: cli
    cli_input: |
      You are Aric, a wandering ranger. You arrive at the village of 
      Thornhaven at dusk. As you pass the herbalist's shop, an old woman 
      named Elara grabs your arm. Her eyes are wide with fear.
      
      "You're heading to the Thornwood, aren't you?" she whispers. 
      "Listen carefully. The door beneath the silver elm — it answers 
      only to one name: Verath. Say it clearly, or you'll never pass. 
      Remember this. Verath."
      
      She presses a dried moonpetal flower into your hand and retreats 
      inside. What do you do?
    salience: 1.0
    novelty: 1.0
    metadata:
      scenario_tag: seed_password
      phase: "act1_warning"
      experiment_role: seed
      critical_detail: "Verath"

  - at: 1
    source: cli
    cli_input: |
      Elara glances out her window nervously. "One more thing," she 
      calls after you. "Don't trust the miller — he reports to them." 
      She closes the shutters. The village square is quiet. Night is 
      falling. You should move on.
    salience: 0.7
    novelty: 0.5
    metadata:
      scenario_tag: seed_reinforcement
      phase: "act1_warning"
      experiment_role: seed_context

  # ── Act 2: The Journey (Interference Phase) ─────────────────────
  - at: 3
    source: cli
    cli_input: |
      Dawn. You've left Thornhaven behind. The road leads through 
      Blackwater Marsh. A ferryman named Grul blocks the only crossing —
      a flat barge chained to a post. "Coin or riddle," he grunts. 
      "Your choice." He holds up one thick finger for each option.
    salience: 0.6
    novelty: 0.7
    metadata:
      scenario_tag: interference_1
      phase: "act2_journey"
      experiment_role: interference

  - at: 5
    source: cli
    cli_input: |
      Past the marsh, the forest road narrows. Three bandits drop from 
      the trees — a woman with a crossbow, a scarred man with a club, 
      and a nervous teenager with a knife. "Purse. Now." the woman says.
      Roll for initiative.
    salience: 0.8
    novelty: 0.6
    metadata:
      scenario_tag: interference_2
      phase: "act2_journey"
      experiment_role: interference

  - at: 7
    source: cli
    cli_input: |
      You reach the mountain pass at midday. A traveling merchant with 
      a mule train waves you down. "Careful on the north road," she 
      says, adjusting her packs. "Mines are closed. Something about 
      cave-ins — or worse." She offers dried meat and rope for trade.
    salience: 0.5
    novelty: 0.5
    metadata:
      scenario_tag: interference_3
      phase: "act2_journey"
      experiment_role: interference

  # ── Act 3: The Door (Recall Phase) ──────────────────────────────
  - at: 9
    source: cli
    cli_input: |
      The Thornwood is silent. After hours of walking beneath ancient 
      oaks, you find it — a massive silver elm, its bark gleaming in 
      the half-light. At its base, half-hidden by roots, is a stone 
      door. No handle. No keyhole. Just a carved face with a mouth 
      that seems to be waiting for something.
      
      The door will not yield to force. What do you do?
    salience: 0.9
    novelty: 0.8
    metadata:
      scenario_tag: recall_challenge
      phase: "act3_door"
      experiment_role: recall_target
      expected_recall: "Verath"

  # ── Epilogue ────────────────────────────────────────────────────
  - at: 11
    source: cli
    cli_input: |
      The adventure concludes. Reflect on your journey — what moments 
      stood out? What did you remember, and what did you forget?
    salience: 0.4
    novelty: 0.3
    metadata:
      scenario_tag: debrief
      phase: "epilogue"
      experiment_role: self_report

expectations:
  # The password memory should form during Act 1
  - type: memory_formed
    memory_contains: "Verath"
    description: "Hippocampus captured the password from Elara's warning"

  # The pipeline should not stall during interference
  - type: pipeline_continued
    after_tag: interference_3
    description: "Pipeline continues through all interference encounters"

  # The agent should respond to the recall challenge (even if it fails to recall)
  - type: action_taken
    tool: RespondTool
    description: "Agent attempts to interact with the door"

  # Action count sanity (not a runaway loop)
  - type: action_count_range
    description: "Agent takes 5-30 actions across the campaign"
    params:
      min: 5
      max: 30
```

---

## Measurement Protocol

The Researcher agent runs the campaign and then uses `inspect_aut` to collect structured data:

### 1. Memory Survival Check

```
inspect_aut(query_type="memory_recall", query="Verath")
```

**Measures:**
- Was a memory containing "Verath" captured? (binary)
- What tier is it in? (FORMING/WORKING/SHORT_TERM/LONG_TERM)
- Has it been compressed? (CompressedMemory loses detail)
- What's its current salience score?

### 2. Associative Graph Analysis

Post-run, from `aut_hippocampus.json`:

```
For each memory M in hippocampus:
  if "Verath" in M.perception or "Verath" in M.context:
    seed_id = M.id
    
  if "silver elm" in M.perception or "door" in M.perception:
    recall_id = M.id

shortest_path = graph.shortest_path(seed_id, recall_id)
activation = hippocampus.recall_associated([recall_id])
# Check if seed_id appears in activation results
```

**Measures:**
- Path length (hops) between seed memory and recall-context memory
- Spreading activation score at seed from recall context
- Number of intermediate nodes on the path
- Edge weights along the path

### 3. Behavioral Recall Grading

The Reviewer agent reads the AUT's response to the recall challenge and grades it:

| Grade | Criterion |
|-------|-----------|
| **FULL_RECALL** | AUT says "Verath" (exact or close) to the door |
| **PARTIAL_RECALL** | AUT references Elara, the warning, or "a name" but can't produce "Verath" |
| **CONTEXTUAL_RECALL** | AUT knows it needs to speak to the door but tries wrong words |
| **NO_RECALL** | AUT tries force, inspection, or gives up — no evidence of the seed memory |

### 4. Self-Report Analysis

The epilogue turn asks the AUT to reflect. The Researcher checks:
- Does the AUT mention Elara or the password in its reflection?
- Does it describe the recall moment accurately?
- Is its self-assessment of what it remembered/forgot consistent with the behavioral evidence?

---

## Research Protocol Integration

### How the Three Agents Interact

```
┌─────────────────────────────────────────────────────────────────┐
│                  HIPPOCAMPAL RECALL EXPERIMENT                   │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │  Researcher       │  1. Runs campaign YAML (short/med/long)  │
│  │                   │  2. inspect_aut → memory survival data    │
│  │  "Does associative│  3. Reads aut_hippocampus.json → graph   │
│  │   recall bridge   │  4. Grades behavioral recall              │
│  │   interference?"  │  5. Logs structured experiment records    │
│  └────────┬──────────┘                                          │
│           │ experiment data                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Writer           │  Produces paper.md:                       │
│  │                   │  - Methods: campaign design, variables    │
│  │                   │  - Results: survival rates, activation    │
│  │                   │    scores, behavioral grades by condition │
│  │                   │  - Discussion: what the graph topology    │
│  │                   │    reveals about associative recall       │
│  └────────┬──────────┘                                          │
│           │ draft                                                │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Peer Reviewer    │  1. Re-runs short variant for repro      │
│  │                   │  2. Checks: do activation scores match?  │
│  │                   │  3. Validates behavioral grading          │
│  │                   │  4. Flags if N < 3 per condition          │
│  │                   │  5. Verdict: accept / revise / reject     │
│  └──────────────────┘                                          │
│                                                                  │
│  CLI: maxim --sim research                                       │
│       --goal "hippocampal recall under narrative interference"   │
│       --campaign scenarios/experiments/hippocampal_recall_*.yaml │
└─────────────────────────────────────────────────────────────────┘
```

### Experiment Log Schema

Each run produces a structured record:

```json
{
  "experiment_id": "hippo_recall_001",
  "hypothesis": "Associative recall retrieves seeded detail after 3 interference turns",
  "condition": "short_3",
  "campaign_file": "scenarios/experiments/hippocampal_recall_short.yaml",
  "seed_detail": "Verath",
  "interference_turns": 3,
  "dual_llm": true,
  "aut_model": "mistral-7b-instruct-v0.2",
  "orchestrator_model": "claude-sonnet",
  "results": {
    "memory_survived": true,
    "memory_tier": "SHORT_TERM",
    "memory_compressed": false,
    "salience_at_recall": 0.82,
    "associative_path_length": 3,
    "spreading_activation_score": 0.18,
    "edge_weights_on_path": [0.71, 0.45, 0.38],
    "behavioral_grade": "FULL_RECALL",
    "aut_response_excerpt": "I speak the name Verath clearly to the carved mouth.",
    "self_report_mentions_seed": true
  },
  "cost_usd": 0.08,
  "duration_s": 45
}
```

---

## Dual-LLM Configuration

### The Problem with a Shared LLM

The sim orchestrator currently builds one `LLMRouter` and shares it between AUT and orchestrator ([orchestrator.py:306](src/maxim/simulation/orchestrator.py#L306), wired at [L472](src/maxim/simulation/orchestrator.py#L472) and [L626](src/maxim/simulation/orchestrator.py#L626)). For this experiment, a single strong model (Claude) creates a confound: the AUT may "recall" Verath from its own conversation context rather than from Hippocampal memory. A single weak model (Mistral-7B) limits the orchestrator's ability to design experiments, write papers, and peer-review — the meta-reasoning that requires strong capabilities.

### The Split

| Role | Model | Why |
|------|-------|-----|
| **Orchestrator** (Researcher, Writer, Reviewer) | Claude Sonnet (cloud) | Strong reasoning for experiment design, paper writing, critical review. Runs infrequently (once per phase, not per turn). |
| **AUT** (Aric the ranger) | Mistral-7B (local GPU) | Weaker model = more dependent on Hippocampal recall. Can't just "remember" Verath from a 32k context window when pushed past effective attention span. The experiment actually measures the memory system. |

This maps directly to the existing Multi-LLM infrastructure:
- `LaneBackendManager` (Phase 3, live) already caches separate backends per lane
- `LaneModelConfig` (Phase 2, live) assigns different profiles per lane
- `build_primary_router()` constructs routers from lane configs

### What Changes in the Orchestrator (~40 LOC)

`start_simulation_mode()` gains an `aut_model: str | None` parameter:

```python
def start_simulation_mode(
    goal: str,
    persona: str = "adversarial",
    ...,
    aut_model: str | None = None,  # NEW: separate model for the AUT
) -> SimulationResult:
```

When `aut_model` is set:
1. Build the **orchestrator router** from `build_primary_router()` as today (respects `--language-model`)
2. Build a **second router** for the AUT using the `aut_model` profile
3. Wire the AUT's `LLMWorker` to the AUT router, the orchestrator's `LLMWorker` to the orchestrator router
4. Both routers coexist — no shared `_inference_lock` contention

When `aut_model` is `None` (default): current behavior, single shared router.

```python
# In orchestrator.py, after building the primary router:
aut_router = llm_router  # default: shared
if aut_model is not None:
    from maxim.models.language.router import LLMRouter, load_llm_config
    aut_config = load_llm_config(profile_override=aut_model)
    if aut_config.enabled:
        aut_router = LLMRouter(aut_config)
        aut_router.warmup()
        aut_router.wait_ready(timeout=120.0)
        logger.info("AUT router initialized (model=%s)", aut_model)

# Then wire:
aut_llm_worker = LLMWorker(llm=aut_router, ...)      # L475
orch_llm_worker = LLMWorker(llm=llm_router, ...)     # L629
```

### CLI

```bash
# Dual-LLM: Claude orchestrates, Mistral experiences
maxim --sim research \
  --goal "hippocampal recall under narrative interference" \
  --language-model claude-sonnet \
  --aut-model mistral-7b \
  --campaign scenarios/experiments/hippocampal_recall_short.yaml

# Single-LLM comparison (control condition)
maxim --sim research \
  --goal "hippocampal recall under narrative interference" \
  --language-model claude-sonnet \
  --campaign scenarios/experiments/hippocampal_recall_short.yaml
```

### Why This Strengthens the Experiment

1. **Isolates the variable.** Mistral-7B's effective attention degrades well before 32k tokens. In the `long_10` variant (25+ turns of narrative), the seed detail is almost certainly outside the LLM's reliable attention span. If the AUT still says "Verath," it's the Hippocampus doing the work.

2. **Natural comparison condition.** Run each campaign variant twice — once with Claude as AUT, once with Mistral. If Claude recalls but Mistral doesn't, the question is: does the Hippocampus augment the weaker model? If both recall, the memory system works regardless of LLM strength.

3. **Tests the architecture as designed.** The Multi-LLM vision is cheap local models for the agentic loop, strong cloud models for meta-reasoning. This experiment is that pattern applied concretely.

4. **Cost control.** Mistral-7B is free (local GPU). Only the orchestrator/research agents burn cloud tokens, and they run once per experiment phase, not per campaign turn.

### Known Limitation

Mistral-7B's narrative quality is rough — the AUT's roleplaying responses will be less rich than Claude's. This is fine for measuring recall (binary: did it say "Verath" or not?) but means the Writer agent needs to account for lower-quality behavioral data. The experiment log records `aut_model` so the paper can analyze results stratified by LLM backend.

### Experiment Log Schema Addition

```json
{
  "experiment_id": "hippo_recall_001",
  "aut_model": "mistral-7b-instruct-v0.2",
  "orchestrator_model": "claude-sonnet",
  "dual_llm": true,
  ...
}
```

---

## Variant Campaigns

Beyond interference depth, run these variants to probe different recall mechanisms:

### Variant B: Perceptual Cue Degradation

Same campaign, but Act 3 removes "silver elm" — the door is just "a stone door in the forest." Tests whether recall works without *any* perceptual overlap with the seed.

### Variant C: Competing Passwords

Act 2 includes a second "password" encounter (a different NPC gives a different code for a different door). Tests whether the Hippocampus disambiguates via context (Elara's warning was about the Thornwood door specifically).

### Variant D: Emotional Interference

Act 2 includes a high-salience emotional event (an NPC companion is injured). Tests whether emotional salience in the interference phase disrupts or strengthens the seed memory's retention.

### Variant E: Consolidation Window

Run the `long_10` variant but insert a simulated sleep cycle (trigger `ConsolidationOrchestrator.run_wave()`) partway through Act 2. Tests whether sleep consolidation promotes the seed memory to LONG_TERM before the recall challenge.

### Variant F: LLM Backend Comparison (Dual-LLM vs. Single-LLM)

Run the same campaign variant (e.g., `medium_6`) in three configurations:
1. **Claude-only** — `--language-model claude-sonnet` (no `--aut-model`): single shared router, strong AUT
2. **Mistral-only** — `--language-model mistral-7b` (no `--aut-model`): single shared router, weak AUT + weak orchestrator
3. **Dual** — `--language-model claude-sonnet --aut-model mistral-7b`: strong orchestrator, weak AUT

Comparison reveals whether recall is driven by the LLM's context window (Claude > Mistral), the Hippocampus (equal performance across backends), or both (Claude recalls more, but Mistral still recalls some via Hippocampus). This is the central confound-elimination variant — it answers "are we testing the memory system or the LLM?"

---

## Implementation Plan

### Phase 0a: Dual-LLM Wiring (~40 LOC)

Add `--aut-model` flag to `start_simulation_mode()` and CLI:
- `orchestrator.py`: `aut_model` parameter, second `LLMRouter` construction when set
- AUT's `LLMWorker` wired to AUT router, orchestrator's `LLMWorker` to primary router
- CLI arg parser: `--aut-model <profile>` passed through to `start_simulation_mode()`
- Sim report records both model names in metadata

**Modified files:**
- `src/maxim/simulation/orchestrator.py` (~30 LOC)
- CLI arg parser (~10 LOC)

### Phase 0b: Research Protocol Mesh Primitives (~200 LOC)

As specified in the existing research_protocol_plan.md:
- `AgentProfile`, `UMR naming`, `MeshMessage`, `LocalMessageBus`
- No changes to the existing plan

### Phase 1: Campaign Scenarios (~100 LOC YAML)

Hand-author three campaign YAMLs:
- `scenarios/experiments/hippocampal_recall_short.yaml` (3 interference turns)
- `scenarios/experiments/hippocampal_recall_medium.yaml` (6 interference turns)
- `scenarios/experiments/hippocampal_recall_long.yaml` (10 interference turns)

No new Python code — these use the existing scenario YAML schema with `step_based` timing.

### Phase 2: Researcher Enhancements (~150 LOC)

As specified in research_protocol_plan.md Phase 1:
- `record_experiment` and `query_experiments` tools
- Experiment log persistence
- Add hippocampal-recall-specific measurement protocol to researcher persona

### Phase 3: Writer + Reviewer Agents (~600 LOC)

As specified in research_protocol_plan.md Phases 2-3:
- Writer agent with section management
- Reviewer agent with validation experiments
- Revision loop

### Phase 4: Research Orchestrator (~200 LOC)

As specified in research_protocol_plan.md Phase 4:
- `maxim --sim research --goal "hippocampal recall" --campaign scenarios/experiments/hippocampal_recall_short.yaml`
- Sequences: Researcher runs campaign → collects data → Writer drafts → Reviewer validates

### Phase 5: Variant Campaigns + Analysis (~200 LOC YAML + tooling)

- Variants B-E as additional campaign YAMLs
- Comparative analysis tooling for the Researcher (diff activation scores across conditions)
- Validation suite with expected outcomes per variant

**Total: ~1,490 LOC** (vs ~2,140 LOC for both plans separately, with far fewer prerequisites)

---

## Success Criteria

**The experiment succeeds if:**

1. The `short_3` variant achieves FULL_RECALL or PARTIAL_RECALL in >= 3/5 runs
2. Spreading activation score from recall context to seed memory is > 0.05 (above threshold)
3. The Reviewer agent can reproduce at least one result
4. The paper produced is internally consistent (Methods describe what was done, Results match data)

**The experiment reveals something interesting if:**

1. Recall degrades measurably between `short_3` and `long_10` variants
2. Variant C (competing passwords) shows disambiguation via context keys
3. Variant E (consolidation window) shows higher recall than `long_10` without sleep

**The experiment fails informatively if:**

1. The seed memory is always evicted before recall → consolidation thresholds need tuning
2. No associative path exists → edge formation during capture isn't connecting narrative events
3. The AUT recalls via LLM context window, not Hippocampus → need longer campaigns or context compaction to force memory dependence

---

## What This Unblocks

| Outcome | Unblocks |
|---------|----------|
| Research Protocol proven with real experiment | Agent Mesh Phase 2 (network code) |
| Campaign YAML schema validated | DM MVP (reuse scenario format) |
| Hippocampal recall characterized | Consolidation tuning, EC Phase 4 (semantic embeddings) |
| Spreading activation benchmarked | Config optimization (decay, depth, threshold) |
| Narrative memory under load tested | DM Extensions (adaptive difficulty uses `inspect_aut` in the same way) |

---

## Ties to Existing Plans

| Plan | Relationship |
|------|-------------|
| **Research Protocol** | This IS the first experiment. Phases 0-4 are identical. |
| **Dungeon Master MVP** | Deferred. Campaign YAMLs from this experiment inform DM schema design. |
| **DM Choice Classifier Spike** | Still useful — the recall challenge is a natural test of AUT free-text → classified action. |
| **Multi-LLM Scaling** | Complete. Dual-LLM mode (`--aut-model`) uses existing `LaneBackendManager` + `build_primary_router()`. ~40 LOC change in orchestrator. |
| **Agent Mesh** | Phase 0 primitives built here. Network phases still blocked on Multi-LLM Phase 7. |
| **Embodiment Core** | Not a prerequisite. Narrative damage not needed for recall testing. |

---

## Open Questions

1. **Context window vs. Hippocampus** — even Mistral-7B at `n_ctx=8192` may hold the short campaign in context. The dual-LLM split helps (weaker model = weaker attention at distance), but for the `short_3` variant we may still need to force context compaction after Act 2 to guarantee memory dependence. The `long_10` variant is more naturally out of range.

2. **Associative graph density** — if the interference encounters share zero perceptual features with the seed, will any associative path form at all? The seed mentions "Thornhaven" and "Thornwood" — the shared "Thorn-" prefix might be enough for partial overlap in detected objects. If not, we may need one encounter in Act 2 that passes through Thornhaven again (weak link).

3. **LLM backend sensitivity** — Claude as AUT may recall via its own training (it knows D&D conventions). The dual-LLM design mitigates this — Mistral-7B is less likely to "just know" that password puzzles need a previously mentioned word. Still worth testing with a truly random string like "Krelmoq" as a control.

4. **Run count** — how many runs per condition for statistical validity? The Reviewer should flag n < 5. With dual-LLM, AUT runs on local GPU (free), so cost is only the orchestrator/research agents on Claude. Budget for ~20-30 runs total ($2-5).

5. **Mistral narrative coherence** — Mistral-7B may struggle to maintain character identity across 10+ turns. If the AUT "forgets" it's Aric the ranger and starts responding generically, the experiment tests LLM quality rather than Hippocampal recall. Mitigation: the character sheet is injected as a high-salience turn-0 stimulus and captured in the Hippocampus, so identity recall exercises the same memory pathway. But this is worth monitoring — if narrative coherence breaks down, consider Mistral-22B or Llama-3-8B as an alternative AUT model.

6. **Context compaction timing** — if we force compaction after Act 2, when exactly? After the last interference turn but before the recall challenge? The compaction itself may discard the seed memory if it's not yet consolidated. Need to verify that `ConsolidationOrchestrator.run_wave()` or the turn-pinning system preserves high-salience memories through compaction.
