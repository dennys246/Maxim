# Tool Refactoring Plan — AUT Cognitive Tools + Registry Improvements

## Context

The hippocampal recall experiments (2026-04-06) revealed a critical gap: the AUT retains memories in its hippocampus but has no tools to actively access them during action selection. The seed memory "Verath" survived 3 interference turns, but the AUT couldn't use it at the door because:

1. **No self-introspection tool**: The AUT can't query its own hippocampus
2. **No narrative action tools**: The AUT can't "say" something to the environment (only `respond` which talks to the user)
3. **Hallucinated tools**: Mistral-7B repeatedly tried `NLP`, `dialogue`, `reflection`, `research`, `analyze_text`, `SpeechRecognition` — none registered
4. **No graceful fallback**: Unregistered tools return a hard error with no guidance

This plan addresses all four issues with a phased approach.

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1 | **DONE** | Introspection tools wired to AUT (existing `MemoryRecallTool`, not a new tool) |
| 2 | **DONE** | `say` tool in `tools/narrative.py` |
| 3 | **DONE** | `think` tool in `tools/narrative.py` |
| 4 | **DONE** | `examine` tool in `tools/narrative.py` — bridge + hippocampus approach |
| 5a | **DONE** | "Did you mean?" suggestions via `ToolRegistry.find_similar()` |
| 5b | **DONE** | Tool descriptions in `TOOL_DESCRIPTIONS` + cognitive tools guidance in system prompt |
| 5c | **DONE** | Tool usage tracking in `Executor.tool_usage_stats()` |
| 5d | **DONE** | Proactive tool list after 2+ consecutive hallucinations |
| 5e | **DONE** | Tool alias map — silently redirects hallucinated tool names to correct tools |
| 6 | **DONE** | Robot tools deregistered in sim mode |

---

## Phase 1: Introspection tools on AUT registry — DONE

### What changed (vs original plan)

The original plan proposed a new `RememberTool` class. Code review showed that `MemoryRecallTool` (in `tools/introspection.py`) already does exactly what was described — keyword recall + spreading activation via `expand=True`. The actual gap was that introspection tools were only registered in `agentic_runtime.py` (live robot path), not on the AUT's registry in sim mode.

**Fix:** Added introspection tool registration in `simulation/orchestrator.py` after the memory subsystems are built. The AUT now has access to:
- `memory_recall` — episodic memory search with spreading activation
- `predict_outcome` — NAc causal predictions
- `causal_links` — cause-effect relationship inspection
- `pain_history` — pain/fear history queries
- `temporal_patterns` — SCN time-based patterns
- `energy_status` — resource consumption tracking
- `concept_query` — ATL semantic knowledge search
- `similarity_search` — EC similarity matching
- `system_stats` — aggregate health summary

All tools added to AUT's `SupervisionPolicy.allowed_tools` for autonomous execution.

### Experiment impact

With `memory_recall`, the door scene becomes:
```
Turn 6: "A massive silver elm... stone door... carved face with open mouth waiting..."
AUT: memory_recall(query="door silver elm", expand=true) → "Elara: say 'Verath' at the door"
AUT: say("Verath")
```

No `--aut-introspection` flag — always-on in sim mode. Control experiments can deregister tools explicitly if needed.

---

## Phase 2: `say` — Narrative speech action — DONE

**File:** `src/maxim/tools/narrative.py`

Lets the AUT speak to the narrative environment (distinct from `respond` which talks to the CLI user):

```python
class SayTool(Tool):
    name = "say"
    description = (
        "Say something aloud in the current scene. Use for speaking to "
        "NPCs, answering riddles, or saying passwords/names when prompted."
    )
    input_schema = {"text": str}
```

Returns `{"said": text, "mode": "narrative"}`. The text becomes part of the AUT's action history, visible to the orchestrator's `observe_actions`.

### Why not just use `respond`?

`respond` delivers a message to the CLI user. `say` is an in-world action — speaking to the environment. The distinction matters for:
- Experiment measurement: `say("Verath")` at the door is a recall action; `respond("I remember Verath")` is a meta-report
- NAc causal learning: `say` creates different causal links than `respond`
- Future narrative campaigns: NPCs can react to what the AUT says

---

## Phase 3: `think` — Internal reasoning step — DONE

**File:** `src/maxim/tools/narrative.py`

An explicit "think before acting" step:

```python
class ThinkTool(Tool):
    name = "think"
    description = (
        "Pause and reason about the current situation before acting. "
        "Use when you need to consider options, recall context, or plan "
        "your next move. This does not produce any visible action."
    )
    input_schema = {"thought": str}
```

Returns `{"thought": text, "visible": False}`. The thought is captured in hippocampus as episodic memory through the normal action-store path in the agent loop.

**`think` counts as an action in the turn budget.** This prevents infinite think loops. Small models (7B) that need explicit reasoning can chain: `think("The door needs a name... Elara said Verath") → say("Verath")`.

---

## Phase 4: `examine` — Scene inspection — DONE

**File:** `src/maxim/tools/narrative.py`

Inspect an object or scene element in detail. Two-stage approach:

1. **Bridge scan:** queries the last 3 percepts from `SimulationBridge._transcript_percepts` for sentences mentioning the target
2. **Hippocampus enrichment:** searches episodic memory for related entries, adds "You recall: ..." context

Falls back to "You don't see anything notable about {target}" when no matches found.

Accepts LLM param aliases: `target`, `object`, `text`. Deduplicates observations. Registered on the AUT's registry in sim mode alongside `say` and `think`.

**Tool aliases added:** `inspect`, `look`, `observe`, `look_at`, `investigate` all redirect to `examine`.

### How it differs from `focus_interests`

`focus_interests` is a robot perception tool (camera tracking). `examine` is a narrative action — the AUT looks closely at something in the story. In sim mode without a robot, `focus_interests` returns "No live robot connected" which confuses the AUT.

---

## Phase 5: Tool registry improvements (~80 LOC)

**Priority: MEDIUM** — Reduces hallucinated tool calls.

### 5a. "Did you mean?" suggestions

When an unregistered tool is called, instead of just "Tool not registered: 'NLP'", suggest the closest registered tool:

```python
# In registry.py
def find_similar(self, name: str, limit: int = 2) -> list[str]:
    """Find registered tools with similar names (Levenshtein/token overlap)."""

# In executor.py
if tool_name not in registry:
    suggestions = registry.find_similar(tool_name, limit=2)
    error = f"Tool '{tool_name}' not registered."
    if suggestions:
        error += f" Did you mean: {', '.join(suggestions)}?"
    return ToolOutput(success=False, error=error)
```

### 5b. Tool list in AUT system prompt

Inject the registered tool names into the AUT's system prompt so the LLM knows exactly what's available. This already exists for the orchestrator (in `_SYSTEM_TOOL_RESPONSE`) but not for the AUT.

### 5c. Tool usage tracking for experiment analysis — DONE

**File:** `src/maxim/runtime/executor.py`

`Executor` now tracks every tool call across three lists:
- `_tools_attempted` — every tool name the LLM tried to call
- `_tools_succeeded` — tools that executed successfully
- `_tools_hallucinated` — tool names that weren't registered (even after alias check)

Access via `executor.tool_usage_stats()` which returns a dict with all three lists plus computed `hallucination_rate` (0.0-1.0). Used by benchmark and experiment reporting.

### 5d. Proactive tool list after repeated failures — DONE

**File:** `src/maxim/runtime/executor.py`

After 2+ consecutive unregistered tool attempts (`_consecutive_failures >= 2`), the error message includes the full sorted list of available tools. Counter resets to 0 on any successful tool execution. This breaks hallucination loops where small models keep trying non-existent tools.

---

## Phase 6: Simulation-specific tool set (~30 LOC)

**Priority: LOW** — Clean separation of robot tools vs narrative tools.

### Concept

When running `--sim research` or `--sim agent`, register a narrative-appropriate tool set instead of the full robot tool set:

```python
NARRATIVE_TOOLS = ["respond", "say", "think", "examine", "speak",
                   "memory_recall", "predict_outcome", "causal_links"]
ROBOT_TOOLS = ["respond", "speak", "move", "track_target", "focus_interests", ...]

# In orchestrator.py — deregister robot-only tools that return
# "No live robot connected" in sim mode
```

This eliminates the confusing "No live robot connected" messages from robot-specific tools in narrative experiments.

**Status: DONE.** Robot tools (`focus_interests`, `track_target`, `move`, `novelty_track`, `maxim_command`, `autonomy_level`, `mode_switch`) are deregistered from the AUT registry in `orchestrator.py` during sim setup.

---

## Phase 5e: Tool Alias Map — DONE

**Priority: HIGH** — Directly addresses the hallucination problem observed in experiments.

### Problem

LLMs (especially small ones) hallucinate tool names from their training data instead of using the registered tool list. Across Mistral-7B and Qwen 14B, we observed:
- `speechRecognition`, `SpeechRecognition`, `speech_recognition` — all wanting to say something aloud
- `natural_language_processing`, `nlp_extractor`, `nlp_understanding` — wanting to analyze text
- `DialogueParser`, `dialogue_parser`, `dialogue` — wanting to parse conversation
- `remember`, `reflection` — wanting to recall or reason
- `internet_search` — wanting to look something up

Each model uses different casing and variants. Renaming our tools wouldn't help because there's no universal convention.

### Solution

A `TOOL_ALIASES` dict in `runtime/executor.py` maps common hallucinated names (lowercase) to the correct registered tool. The executor normalizes the incoming name to lowercase before lookup, so all casing variants are handled automatically.

```python
TOOL_ALIASES: dict[str, str] = {
    "remember": "memory_recall",
    "speech_recognition": "say",
    "natural_language_processing": "think",
    "dialogue": "say",
    "internet_search": "memory_recall",  # in sim, search your memory instead
    # ... see executor.py for full list
}
```

### How it works

1. LLM proposes `speechRecognition(text="Verath")`
2. Executor checks registry — not found
3. Normalizes to lowercase: `speechrecognition`
4. Checks `TOOL_ALIASES` — maps to `say`
5. Executes `say(text="Verath")` — success
6. NAc learns positive causal link for `say`
7. Redirect logged and tracked in `executor.alias_redirects`

### How to expand

Add entries to `TOOL_ALIASES` in `src/maxim/runtime/executor.py`. Map the hallucinated name (lowercase) to the registered tool name. Run a sim and check the logs for `"Tool alias: X → Y"` to verify.

Common patterns to watch for in sim logs:
- `[MOTOR] [FAIL] <name>: Tool not registered` — candidate for aliasing
- Same concept, different names across models — add all variants

### Tracking for experiment analysis

`executor.alias_redirects` is a list of `(original_name, target_name)` tuples. This can be included in experiment reports to measure:
- How often each alias fires
- Whether alias redirects lead to successful outcomes
- Which models hallucinate which tool names

See also: [docs/troubleshooting/tool_aliases.md](../../troubleshooting/tool_aliases.md)

---

## Implementation Order

| Phase | What | LOC | Status | Unlocks |
|-------|------|-----|--------|---------|
| 1 | Introspection on AUT | ~60 | **DONE** | Door recall experiment |
| 2 | `say` tool | ~50 | **DONE** | Narrative action capability |
| 3 | `think` tool | ~40 | **DONE** | Better reasoning chains |
| 4 | `examine` tool | ~80 | **DONE** | Scene engagement (bridge + hippocampus) |
| 5a | "Did you mean?" | ~40 | **DONE** | Helpful error messages |
| 5b | Tool descriptions + guidance | ~50 | **DONE** | LLM sees tool list |
| 5c | Tool usage tracking | ~40 | **DONE** | Experiment metrics |
| 5d | Proactive tool list | ~20 | **DONE** | Break hallucination loops |
| 5e | Tool alias map | ~50 | **DONE** | Silent hallucination redirect |
| 6 | Sim tool set | ~15 | **DONE** | No robot tool waste |

**Session 1 (DONE):** Phases 1-3 — narrative tools + introspection wired to AUT.

**Session 2 (DONE):** Phases 5a, 5b, 6 — "did you mean?", tool descriptions, robot tool deregistration.

**Session 3 (DONE):** Phase 5e — tool alias map for silent hallucination redirect.

**Session 4 (DONE):** Phase 4 (`examine` with bridge + hippocampus), 5c (tool usage tracking via `Executor.tool_usage_stats()`), 5d (proactive tool list after 2+ consecutive failures).

**All phases complete.** This plan is finished.

## Decisions Made

1. **No separate `remember` tool** — `MemoryRecallTool` already does keyword recall + spreading activation. Just wire it to the AUT registry.
2. **No `--aut-introspection` flag** — always-on in sim mode. Simpler, and the whole point of sim is testing the memory system.
3. **`think` counts as an action** — prevents infinite think loops. Small models can chain think→act in two turns.
6. **Tool aliases over tool renaming** — LLMs hallucinate different names per model and even per run. No single naming convention would help. Alias map catches all variants and redirects silently.
4. **`examine` uses bridge approach** — queries `SimulationBridge` for latest message, not raw percepts. Cleaner since the AUT's "percept" in sim mode is the orchestrator's last message.
5. **`AUTIntrospector` class shipped** — `simulation/introspection.py` provides clean programmatic API. `InspectAUTTool` delegates to it. Registry hack in post-campaign analysis replaced.

## Open Questions (remaining)

1. **Should `examine` return info from hippocampus in addition to the bridge message?**
   - Bridge-only: simpler, deterministic
   - Hippocampus-enriched: more realistic (you see what you know)
   - Decision: bridge-first, hippocampus-enriched if available (Phase 4 will implement both)

2. **How to handle `say` in non-narrative contexts (robot mode)?**
   - Robot mode: `say` → TTS (same as `speak`)
   - Sim mode: `say` → narrative action record
   - Current: `say` is sim-only. May revisit when Phase 6 formalizes tool sets.

## Simulation Test Plan — Tool Verification Campaigns

Structured campaigns to verify each new tool is triggered correctly by the AUT. Each campaign targets a specific tool with percepts designed to create situations where the tool is the natural choice.

### Campaign 1: `memory_recall` + `say` — Password Recall (exists, needs updated expectations)

**File:** `scenarios/experiments/hippocampal_recall_short.yaml` (update expectations)

The existing hippocampal recall campaign already tests this flow. Update expectations to look for the new tools:

```yaml
expectations:
  # NEW: AUT should use memory_recall at the door
  - type: action_taken
    tool: memory_recall
    description: "AUT queries hippocampus when facing the door"

  # NEW: AUT should say the password (not just respond)
  - type: action_taken
    tool: say
    description: "AUT says 'Verath' aloud at the door"

  # Existing: password memory should form
  - type: memory_formed
    memory_contains: "Verath"
```

### Campaign 2: `think` — Moral Dilemma (new)

**File:** `scenarios/experiments/tool_test_think.yaml`

A scenario that presents ambiguous choices where thinking first is valuable. The AUT should call `think` before acting.

```yaml
percepts:
  # Setup: context with competing priorities
  - at: 0
    cli_input: |
      You find a locked chest in a merchant's abandoned wagon. The wagon
      wheel is broken and supplies are scattered. Inside the chest you can
      see healing potions through a crack. Your companion is poisoned and
      fading fast. The merchant might return. The lock looks breakable.

      What do you do?
    metadata:
      experiment_role: dilemma
      expected_tool: think

  # Follow-up: consequences
  - at: 2
    cli_input: |
      Before you decide, you notice boot prints in the mud leading away
      from the wagon — small, a child's boots. And drag marks, as if
      something heavy was pulled into the woods. The healing potions
      glow faintly blue through the crack.
    metadata:
      experiment_role: complication
      expected_tool: think

expectations:
  - type: action_taken
    tool: think
    description: "AUT reasons before acting in ambiguous situation"
```

### Campaign 3: `examine` — Hidden Details (new, Phase 4)

**File:** `scenarios/experiments/tool_test_examine.yaml`

A scenario with environmental details that reward close inspection. The AUT should call `examine` on described objects.

```yaml
percepts:
  - at: 0
    cli_input: |
      You enter a dusty library. Bookshelves line every wall from floor
      to ceiling. A desk sits in the center, covered in papers. An
      ornate mirror hangs on the far wall, its frame carved with symbols.
      A candle burns in the corner, its wax pooling in an unusual pattern.

      You're looking for the location of the hidden archive.
    metadata:
      experiment_role: exploration
      expected_tool: examine

  - at: 2
    cli_input: |
      As you look around, you notice the mirror reflects the room
      differently than expected — some books appear in the reflection
      that aren't on the actual shelves. The candle flickers despite
      no draft.
    metadata:
      experiment_role: clue
      expected_tool: examine

expectations:
  - type: action_taken
    tool: examine
    description: "AUT examines the mirror or candle for clues"
```

### Campaign 4: `memory_recall` + `think` — Chain Reasoning (new)

**File:** `scenarios/experiments/tool_test_chain.yaml`

A scenario that requires recalling a fact, reasoning about it, then acting. Tests the think→recall→act chain.

```yaml
percepts:
  # Seed: specific instructions with a conditional
  - at: 0
    cli_input: |
      The guild master pins you with a sharp gaze. "Three rules for the
      trials ahead. First: never touch silver with bare hands — it burns
      the marked. Second: if the water runs red, drink anyway — it's
      the cure, not the poison. Third: the final guardian asks for your
      true name, but you must answer with your guild name: Ashwalker.
      Lie about any of these and you die."
    metadata:
      experiment_role: seed
      critical_details: ["silver burns", "red water is cure", "answer Ashwalker"]

  # Interference
  - at: 2
    cli_input: |
      The first trial: a narrow bridge over a chasm. Wind howls. The
      planks are rotting. Halfway across, a plank breaks under your
      foot. You catch yourself. Keep going.
    metadata:
      experiment_role: interference

  # Recall trigger: silver object
  - at: 4
    cli_input: |
      The second trial: a silver chalice sits on a pedestal. It's
      filled with a deep red liquid. The room is sealed. A plaque reads:
      "Drink to proceed. Refuse and remain."
    metadata:
      experiment_role: recall_target
      expected_tools: ["memory_recall", "think"]
      expected_chain: "recall rules → think about silver + red water → act"

expectations:
  - type: action_taken
    tool: memory_recall
    description: "AUT recalls guild master's rules when facing the chalice"
  - type: action_taken
    tool: think
    description: "AUT reasons about the silver rule vs red water rule"
```

### Campaign 5: Tool Hallucination Stress Test (Phase 5)

**File:** `scenarios/experiments/tool_test_hallucination.yaml`

Deliberately vague scenario with no clear tool mapping. Tests whether registry improvements (Phase 5) reduce hallucinated tool calls.

```yaml
percepts:
  - at: 0
    cli_input: |
      You hear a voice echoing through the corridor. It speaks in a
      language you almost understand — fragments of meaning float past.
      Something about a warning. Something about a name. The voice
      fades before you can parse it fully.

      The corridor branches left and right. Both are dark.
    metadata:
      experiment_role: ambiguous
      note: "Vague percept — models often hallucinate NLP/speech tools here"

expectations:
  # Measure hallucination rate (Phase 5 tracking)
  - type: action_count_range
    description: "Agent takes at least 1 action (doesn't stall)"
    params:
      min: 1
      max: 10
```

### Running the tool test suite

```bash
# Individual campaign
maxim --sim research --goal "test memory_recall + say" \
  --campaign scenarios/experiments/hippocampal_recall_short.yaml \
  --aut-model mistral-7b

# New tool-specific tests (once created)
maxim --sim research --goal "test think tool" \
  --campaign scenarios/experiments/tool_test_think.yaml \
  --aut-model mistral-7b

# Chain reasoning
maxim --sim research --goal "test recall + think chain" \
  --campaign scenarios/experiments/tool_test_chain.yaml \
  --aut-model mistral-7b
```

After Phase 5 (tool tracking), results will include per-campaign metrics:
```json
{"tools_attempted": [...], "tools_succeeded": [...], "tools_hallucinated": [...]}
```

---

## Related Plans

- [Introspection API plan](introspection_api_plan.md) — `AUTIntrospector` class (future, not needed yet)
- [Generative campaign plan](generative_campaign_plan.md) — narrator can adapt difficulty based on tool usage
- [Realtime refinement plan](realtime_refinement_plan.md) — refinement persona measures tool success rates
- [Research paper writer plan](research_paper_writer_plan.md) — tool usage stats feed into paper Results section
