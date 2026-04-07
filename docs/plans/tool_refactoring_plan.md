# Tool Refactoring Plan — AUT Cognitive Tools + Registry Improvements

## Context

The hippocampal recall experiments (2026-04-06) revealed a critical gap: the AUT retains memories in its hippocampus but has no tools to actively access them during action selection. The seed memory "Verath" survived 3 interference turns, but the AUT couldn't use it at the door because:

1. **No self-introspection tool**: The AUT can't query its own hippocampus
2. **No narrative action tools**: The AUT can't "say" something to the environment (only `respond` which talks to the user)
3. **Hallucinated tools**: Mistral-7B repeatedly tried `NLP`, `dialogue`, `reflection`, `research`, `analyze_text`, `SpeechRecognition` — none registered
4. **No graceful fallback**: Unregistered tools return a hard error with no guidance

This plan addresses all four issues with a phased approach.

## Phase 1: `remember` — Self-directed memory retrieval (~100 LOC)

**Priority: HIGH** — This is the single biggest experiment enabler.

### What it does

Lets the AUT actively search its own hippocampus during action selection:

```python
class RememberTool(Tool):
    """Search your own memory for relevant information."""
    name = "remember"
    input_schema = {
        "query": (str, ""),        # What to search for
        "context": (str, ""),      # Current situation context
    }
```

### How it works

1. AUT encounters the door with "carved face with open mouth"
2. AUT calls `remember(query="door silver elm", context="stone door, no handle, waiting for something")`
3. Tool runs `hippocampus.recall_similar()` + `recall_associated()` with spreading activation
4. Returns: `{"memories": [{"summary": "Elara warned: say 'Verath' at the door beneath the silver elm", "relevance": 0.85}]}`
5. AUT now has the information to call `say("Verath")`

### Implementation

**File:** `src/maxim/tools/remember.py` (new)

```python
class RememberTool(Tool):
    name = "remember"
    description = (
        "Search your memory for relevant information. Use when you need to "
        "recall something you learned earlier. Returns matching memories "
        "ranked by relevance."
    )
    input_schema = {
        "query": (str, ""),
        "context": (str, ""),
        "limit": (int, 5),
    }

    def __init__(self, *, hippocampus=None, memory_hub=None):
        super().__init__()
        self._hippocampus = hippocampus
        self._memory_hub = memory_hub

    def execute(self, **kwargs):
        query = kwargs.get("query", "")
        context = kwargs.get("context", "")
        limit = kwargs.get("limit", 5)

        results = []

        # Stage 1: keyword recall
        if self._hippocampus and query:
            memories = self._hippocampus.recall(limit=limit)
            # Filter by keyword match
            for mem in memories:
                if query.lower() in str(mem).lower():
                    results.append(self._format_memory(mem))

        # Stage 2: associative recall via spreading activation
        if self._hippocampus and results:
            seed_ids = [r["id"] for r in results if "id" in r]
            if seed_ids:
                associated = self._hippocampus.recall_associated(
                    seed_ids, limit=limit
                )
                for mem, score in associated:
                    results.append({
                        **self._format_memory(mem),
                        "activation_score": round(score, 3),
                    })

        return ToolOutput(
            success=True,
            output={"memories": results[:limit], "total_found": len(results)},
        )
```

### Registration

- **AUT-only**: Register on AUT's tool registry in `orchestrator.py`, NOT on the orchestrator
- **Opt-in**: Gated by `--aut-introspection` flag (default off, to preserve experiment validity when needed)
- Wire `hippocampus` and `memory_hub` from AUT's components

### Experiment impact

With `remember`, the door scene becomes:
```
Turn 6: "A massive silver elm... stone door... carved face with open mouth waiting..."
AUT: remember(query="door silver elm") → "Elara: say 'Verath' at the door"
AUT: say("Verath")
```

This tests whether spreading activation actually connects the recall cue to the seed — the core hippocampal mechanism.

---

## Phase 2: `say` — Narrative speech action (~50 LOC)

**Priority: HIGH** — Needed for the AUT to act on recalled memories.

### What it does

Lets the AUT speak to the narrative environment (distinct from `respond` which talks to the CLI user):

```python
class SayTool(Tool):
    name = "say"
    description = (
        "Say something aloud in the current scene. Use for speaking to "
        "NPCs, answering riddles, or saying passwords/names when prompted."
    )
    input_schema = {
        "text": (str, ""),
    }
```

### Implementation

**File:** `src/maxim/tools/narrative.py` (new)

Simple tool that records what was said and returns success. The text becomes part of the AUT's action history, visible to the orchestrator's `observe_actions`.

```python
def execute(self, **kwargs):
    text = kwargs.get("text", "")
    if not text:
        return ToolOutput(success=False, error="Nothing to say")
    return ToolOutput(
        success=True,
        output={"said": text, "mode": "narrative"},
    )
```

### Why not just use `respond`?

`respond` delivers a message to the CLI user. `say` is an in-world action — speaking to the environment. The distinction matters for:
- Experiment measurement: `say("Verath")` at the door is a recall action; `respond("I remember Verath")` is a meta-report
- NAc causal learning: `say` creates different causal links than `respond`
- Future narrative campaigns: NPCs can react to what the AUT says

---

## Phase 3: `think` — Internal reasoning step (~40 LOC)

**Priority: MEDIUM** — Helps small models that need explicit reasoning.

### What it does

An explicit "think before acting" step that doesn't produce an external action:

```python
class ThinkTool(Tool):
    name = "think"
    description = (
        "Pause and reason about the current situation before acting. "
        "Use when you need to consider options, recall context, or plan."
    )
    input_schema = {
        "thought": (str, ""),
    }
```

### Why it matters

Small models (7B) often jump to action without reasoning. An explicit `think` tool:
- Gives the model a "scratchpad" step
- The thought gets captured in hippocampus (useful for introspection)
- Reduces hallucinated tool calls (model reasons first, then picks the right tool)
- Chain: `think("The door needs a name... Elara said Verath") → say("Verath")`

### Implementation

Returns the thought as output (no side effects). The hippocampus captures it as an episodic memory with the reasoning as the goal.

---

## Phase 4: `examine` — Scene inspection (~60 LOC)

**Priority: MEDIUM** — Replaces `focus_interests` for narrative contexts.

### What it does

Inspect an object or scene element in detail:

```python
class ExamineTool(Tool):
    name = "examine"
    description = (
        "Examine an object, person, or feature in the current scene. "
        "Returns what you observe about it."
    )
    input_schema = {
        "target": (str, ""),
    }
```

### How it differs from `focus_interests`

`focus_interests` is a robot perception tool (camera tracking). `examine` is a narrative action — the AUT looks closely at something in the story. In sim mode without a robot, `focus_interests` returns "No live robot connected" which confuses the AUT. `examine` would return a contextual description based on the current percept.

### Implementation

Scans the most recent percept for mentions of the target and returns relevant text. Falls back to "You don't see anything notable about {target}" if no match.

---

## Phase 5: Tool registry improvements (~80 LOC)

**Priority: MEDIUM** — Reduces hallucinated tool calls.

### 5a. "Did you mean?" suggestions

When an unregistered tool is called, instead of just "Tool not registered: 'NLP'", suggest the closest registered tool:

```python
# In executor.py
if tool_name not in registry:
    suggestions = registry.find_similar(tool_name, limit=2)
    error = f"Tool '{tool_name}' not registered."
    if suggestions:
        error += f" Did you mean: {', '.join(suggestions)}?"
    return ToolOutput(success=False, error=error)
```

Uses simple string similarity (Levenshtein or token overlap) against registered tool names + descriptions.

### 5b. Tool list in AUT system prompt

Inject the registered tool names into the AUT's system prompt so the LLM knows exactly what's available:

```
Available tools: respond, say, remember, think, examine, focus_interests, 
read_file, write_file, glob, speak
```

This already exists for the orchestrator (in `_SYSTEM_TOOL_RESPONSE`) but not for the AUT.

### 5c. Tool usage tracking for experiment analysis

Track which tools the AUT attempts vs which succeed, and include in experiment metrics:

```python
metrics = {
    "tools_attempted": ["remember", "say", "think", "respond"],
    "tools_succeeded": ["remember", "say", "respond"],
    "tools_hallucinated": ["NLP", "dialogue"],  # attempted but not registered
}
```

---

## Phase 6: Simulation-specific tool set (~30 LOC)

**Priority: LOW** — Clean separation of robot tools vs narrative tools.

### Concept

When running `--sim research` or `--sim agent`, register a narrative-appropriate tool set instead of the full robot tool set:

```python
NARRATIVE_TOOLS = ["respond", "say", "remember", "think", "examine", "speak"]
ROBOT_TOOLS = ["respond", "speak", "move", "track_target", "focus_interests", ...]

# In orchestrator.py
if sim_mode:
    register_tools(NARRATIVE_TOOLS)
else:
    register_tools(ROBOT_TOOLS)
```

This eliminates the confusing "No live robot connected" messages from robot-specific tools in narrative experiments.

---

## Implementation Order

| Phase | What | LOC | Depends On | Unlocks |
|-------|------|-----|-----------|---------|
| 1 | `remember` tool | ~100 | Hippocampus access | Door recall experiment |
| 2 | `say` tool | ~50 | None | Narrative action capability |
| 3 | `think` tool | ~40 | None | Better reasoning chains |
| 4 | `examine` tool | ~60 | Percept access | Scene engagement |
| 5 | Registry improvements | ~80 | None | Reduced hallucination |
| 6 | Sim tool set | ~30 | Phases 1-4 | Clean separation |
| **Total** | | **~360** | | |

**Recommended first session:** Phases 1 + 2 (~150 LOC) — gives you `remember` + `say`, which is enough to re-run the hippocampal recall experiment and test whether the AUT can actually USE its memory at the door.

## Open Questions

1. **Should `remember` always be available, or opt-in via flag?**
   - Always-on: simpler, AUT always has memory access
   - Opt-in (`--aut-introspection`): preserves ability to test "without memory tools" as a control condition
   - Recommendation: always-on in sim mode, since the whole point is testing the memory system

2. **Should `think` count as an action in the turn budget?**
   - If yes: AUT might skip thinking to save actions
   - If no: could loop endlessly thinking without acting
   - Recommendation: counts as action but with a low "cost" in the autonomy budget

3. **Should `examine` return info from hippocampus or from the raw percept?**
   - Percept-only: simpler, deterministic
   - Hippocampus-enriched: more realistic (you see what you know)
   - Recommendation: percept-first, hippocampus-enriched if available

4. **How to handle `say` in non-narrative contexts (robot mode)?**
   - Robot mode: `say` → TTS (same as `speak`)
   - Sim mode: `say` → narrative action record
   - Recommendation: `say` is sim-only, `speak` is the robot equivalent

5. **Should the tool registry suggest tools proactively?**
   - e.g., after 3 failed tool attempts, inject: "Available tools: respond, say, remember..."
   - Could reduce the hallucination loop
   - Recommendation: yes, after 2+ consecutive unregistered tool attempts

## Related Plans

- [Introspection API plan](introspection_api_plan.md) — `AUTIntrospector` class that `remember` delegates to
- [Generative campaign plan](generative_campaign_plan.md) — narrator can adapt difficulty based on tool usage
- [Realtime refinement plan](realtime_refinement_plan.md) — refinement persona measures tool success rates
- [Research paper writer plan](research_paper_writer_plan.md) — tool usage stats feed into paper Results section
