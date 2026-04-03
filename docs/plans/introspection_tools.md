# Introspection Tools Plan

> **Status:** Not started. Depends on ToolRegistry (implemented), MemoryHub (implemented), all biological subsystems (implemented).

Expose Maxim's internal biological subsystems as LLM-callable tools so the agent can introspect on its own memories, predictions, pain history, temporal patterns, and energy state. Each tool is a thin read-only wrapper around existing query methods — no new computation, just access.

---

## Motivation

The LLM currently receives memory context passively via StructuredContext (built by MemoryAgent). It cannot:
- Search its own episodic memories ("what happened last time I tried this?")
- Query causal predictions explicitly ("what does NAc predict for this tool?")
- Check its pain/fear history ("have I been hurt by this action before?")
- Discover temporal patterns ("do I fail more in the morning?")
- Assess its own energy budget ("how many tokens have I burned?")
- Explore semantic relationships ("what concepts relate to 'kitchen'?")

These are all read-only queries against existing subsystems with existing methods. The tools are thin wrappers with input validation and output formatting.

---

## Tool Design Principles

1. **Read-only.** No tool modifies agent state. Queries only.
2. **Formatted for LLM consumption.** Return structured dicts that the LLM can reason about, not raw dataclass dumps.
3. **Bounded output.** All tools accept a `limit` parameter and cap output to avoid flooding the context window.
4. **Graceful degradation.** If a subsystem isn't available (e.g., no robot → no SalienceNetwork), return `{"available": false, "reason": "..."}`.
5. **Keyword-friendly.** Each tool declares manual keywords for the LearnedToolIndex so it surfaces when relevant.

---

## Tools

### Tool 1: `memory_recall` — Search Episodic Memory

Query the hippocampus for past experiences. Supports filtering by goal, tool, success, object, person, mode, and time range. Optionally expands results via spreading activation through ASSOCIATES and CAUSES edges.

```python
class MemoryRecallTool(Tool):
    name = "memory_recall"
    description = (
        "Search your episodic memories. Filter by goal, tool, success/failure, "
        "detected objects, or time range. Use 'expand=true' to find associated "
        "memories via spreading activation through the associative graph."
    )
    input_schema = {
        "query": (str, None),           # Free-text goal/description to match
        "tool_name": (str, None),       # Filter by tool used
        "success": (bool, None),        # Filter by outcome
        "object": (str, None),          # Filter by detected object
        "person": (str, None),          # Filter by detected person
        "mode": (str, None),            # Filter by operational mode
        "time_after": (str, None),      # ISO timestamp lower bound
        "time_before": (str, None),     # ISO timestamp upper bound
        "expand": (bool, False),        # Use spreading activation to find related memories
        "limit": (int, 5),              # Max results
    }
    MANUAL_KEYWORDS = {"remember", "recall", "memory", "past", "history",
                       "previous", "before", "last", "when", "episode"}
```

**Returns:** List of memory summaries with id, timestamp, goal, tool, success, outcome preview, salience, novelty. If `expand=true`, includes activation score.

### Tool 2: `predict_outcome` — Query Causal Predictions

Ask NAc what it predicts will happen if a specific action is taken. Returns the Rescorla-Wagner predicted value, expected delay, confidence, and all possible outcomes.

```python
class PredictOutcomeTool(Tool):
    name = "predict_outcome"
    description = (
        "Query what the causal learning system (NAc) predicts will happen "
        "if you execute a tool. Returns predicted outcome, confidence, "
        "expected delay, and all possible outcomes with their valences."
    )
    input_schema = {
        "tool_name": str,                # Tool to predict for
        "context": (dict, None),         # Optional context (goal, mode, etc.)
        "include_all_outcomes": (bool, True),  # Show all possible outcomes
    }
    MANUAL_KEYWORDS = {"predict", "outcome", "expect", "happen", "result",
                       "likely", "probability", "confidence", "risk"}
```

**Returns:** Primary prediction (value, valence, delay, delay_bounds, confidence, context_match) + all outcomes sorted by confidence.

### Tool 3: `pain_history` — Query Pain and Fear

Access the pain detector's history and the fear agent's current safety assessments. Shows recent pain signals, tool error escalation state, and which actions the fear system would block.

```python
class PainHistoryTool(Tool):
    name = "pain_history"
    description = (
        "Check your pain and fear history. See recent pain signals from "
        "tool failures, movement errors, and timeouts. Query whether the "
        "fear system would block a specific action."
    )
    input_schema = {
        "check_action": (str, None),     # Tool name to check fear gate for
        "action_params": (dict, None),   # Params for fear check
        "limit": (int, 10),              # Max pain signals to return
    }
    MANUAL_KEYWORDS = {"pain", "hurt", "fear", "danger", "safe", "risk",
                       "error", "failure", "avoid", "block", "afraid"}
```

**Returns:** Recent pain signals (type, intensity, tool, timestamp), pain stats (counts per type), and optionally a fear gate check result (allow/block + reason).

### Tool 4: `temporal_patterns` — Query Time-Based Patterns

Access the SCN's temporal binning to discover circadian, weekly, and monthly patterns in the agent's activity. Find what happens at specific times or discover rhythmic patterns.

```python
class TemporalPatternsTool(Tool):
    name = "temporal_patterns"
    description = (
        "Discover time-based patterns in your experience. Find memories "
        "from specific times of day, days of week, or discover rhythmic "
        "patterns (e.g., failures cluster on Monday mornings)."
    )
    input_schema = {
        "hour": (int, None),             # Hour of day (0-23)
        "day": (int, None),              # Day of week (0=Mon, 6=Sun)
        "discover_rhythms": (bool, False),  # Find rhythmic patterns
        "limit": (int, 10),              # Max memories to return
    }
    MANUAL_KEYWORDS = {"time", "when", "morning", "evening", "monday",
                       "pattern", "rhythm", "schedule", "circadian", "daily", "weekly"}
```

**Returns:** Memory IDs matching temporal filter, or discovered rhythmic patterns with occurrence counts.

### Tool 5: `energy_status` — Self-Awareness of Resource Consumption

Query the energy tracker for recent and total computational budget usage.

```python
class EnergyStatusTool(Tool):
    name = "energy_status"
    description = (
        "Check your energy and resource consumption. See recent token usage, "
        "inference costs, and overall energy budget status."
    )
    input_schema = {
        "window_seconds": (float, 300.0),  # Time window for recent stats
    }
    MANUAL_KEYWORDS = {"energy", "budget", "cost", "token", "resource",
                       "usage", "consumption", "efficiency", "expensive"}
```

**Returns:** Window stats (count, total_energy, rate_per_second, average) + total stats (lifetime counts).

### Tool 6: `concept_query` — Search Semantic Knowledge

Query the ATL for semantic concepts, their relationships, and properties. Explore the concept graph via typed relationships (IS_A, PART_OF, EXECUTES_WITH, etc.).

```python
class ConceptQueryTool(Tool):
    name = "concept_query"
    description = (
        "Search your semantic knowledge base for concepts, facts, and "
        "relationships. Query by name, category, or explore relationships "
        "between concepts (IS_A, PART_OF, CAUSES, EXECUTES_WITH, etc.)."
    )
    input_schema = {
        "name": (str, None),             # Concept name to search
        "category": (str, None),         # Filter by category
        "concept_id": (str, None),       # Get relationships for specific concept
        "relationship_type": (str, None),  # Filter relationships (IS_A, PART_OF, etc.)
        "limit": (int, 10),
    }
    MANUAL_KEYWORDS = {"concept", "know", "knowledge", "what", "define",
                       "relationship", "related", "category", "semantic", "meaning"}
```

**Returns:** Concept entries with name, category, confidence, episode_count, properties, and optionally relationships to other concepts.

### Tool 7: `scene_summary` — Current Visual Scene

Query the salience and attention networks for what's currently visible and interesting.

```python
class SceneSummaryTool(Tool):
    name = "scene_summary"
    description = (
        "Get a summary of the current visual scene. Shows the most salient "
        "objects, where you're currently looking, and what's novel or familiar."
    )
    input_schema = {
        "top_n": (int, 5),               # Number of top salient objects
        "include_attention": (bool, True),  # Include gaze/focus info
    }
    MANUAL_KEYWORDS = {"see", "look", "scene", "visible", "object",
                       "detect", "salient", "novel", "attention", "focus", "gaze"}
```

**Returns:** Top salient objects (label, salience, novelty, position), current focus point, dwell time, next suggested target.

### Tool 8: `similarity_search` — Find Similar Situations

Query the EC for situations similar to a description or similar to a past memory. Uses LSH approximate nearest neighbor + structural/temporal matching.

```python
class SimilaritySearchTool(Tool):
    name = "similarity_search"
    description = (
        "Find situations similar to a description or to a past memory. "
        "Uses multi-modal similarity (structural, temporal, semantic) to "
        "find related experiences across your memory."
    )
    input_schema = {
        "tool_name": (str, None),        # Find similar uses of this tool
        "memory_id": (str, None),        # Find memories similar to this one
        "context": (dict, None),         # Context for tool-based search
        "limit": (int, 5),
    }
    MANUAL_KEYWORDS = {"similar", "like", "related", "comparable", "same",
                       "analogy", "pattern", "match", "familiar"}
```

**Returns:** Similar situation memory IDs with similarity scores and signatures.

### Tool 9: `causal_links` — Inspect Learned Cause-Effect Relationships

Directly inspect NAc's causal link database. See what events cause what outcomes, with confidence, observation counts, and temporal delay distributions.

```python
class CausalLinksTool(Tool):
    name = "causal_links"
    description = (
        "Inspect your learned cause-effect relationships. See what events "
        "lead to what outcomes, filter by positive/negative valence, and "
        "trace which episodic memories contributed to each causal link."
    )
    input_schema = {
        "event": (str, None),            # Event signature (e.g., "tool:grab")
        "outcome": (str, None),          # Outcome signature to reverse-query
        "memory_id": (str, None),        # Find links referencing this memory
        "valence": (str, None),          # "positive", "negative", or None for all
        "limit": (int, 10),
    }
    MANUAL_KEYWORDS = {"cause", "effect", "causal", "link", "learn",
                       "outcome", "because", "why", "consequence", "lead"}
```

**Returns:** CausalLink summaries with event/outcome signatures, valence, predicted_value, confidence, observation_count, temporal_delta (mean delay), and contributing memory IDs.

### Tool 10: `system_stats` — Overall System Health

Aggregate statistics from all biological subsystems in one query. Quick health check.

```python
class SystemStatsTool(Tool):
    name = "system_stats"
    description = (
        "Get a health summary of all your biological subsystems: memory counts, "
        "causal link stats, energy usage, pain history, and learning progress."
    )
    input_schema = {}  # No parameters — returns everything
    MANUAL_KEYWORDS = {"status", "health", "stats", "system", "summary",
                       "overview", "diagnostic", "how", "doing"}
```

**Returns:** Aggregated stats from hippocampus, NAc, EC, ATL, energy tracker, pain detector, and significance learner.

---

## Implementation Sequencing

| Phase | What | Effort | Tools |
|-------|------|--------|-------|
| **1** | Core memory + prediction tools | Medium | memory_recall, predict_outcome, causal_links |
| **2** | Pain, fear, and safety tools | Small | pain_history |
| **3** | Temporal and similarity tools | Small | temporal_patterns, similarity_search |
| **4** | Concept and scene tools | Small | concept_query, scene_summary |
| **5** | Energy and system health | Small | energy_status, system_stats |
| **6** | Wire into registry + documentation + tests | Medium | All tools |

Phase 1 is the highest value. Phases 2-5 are independent of each other.

---

## Wiring

All tools registered conditionally in `agentic_runtime.py` based on available subsystems:

```python
# After memory_hub, nac, pain_detector, etc. are created:
from maxim.tools.introspection import (
    MemoryRecallTool, PredictOutcomeTool, PainHistoryTool,
    TemporalPatternsTool, EnergyStatusTool, ConceptQueryTool,
    SceneSummaryTool, SimilaritySearchTool, CausalLinksTool,
    SystemStatsTool,
)

if memory_hub is not None:
    registry.register(MemoryRecallTool(hippocampus=memory_hub.hippocampus))
    registry.register(SimilaritySearchTool(ec=memory_hub.ec))
    registry.register(TemporalPatternsTool(scn=memory_hub.scn))
    if memory_hub.atl is not None:
        registry.register(ConceptQueryTool(atl=memory_hub.atl))

if nac is not None:
    registry.register(PredictOutcomeTool(nac=nac))
    registry.register(CausalLinksTool(nac=nac))

if pain_detector is not None:
    registry.register(PainHistoryTool(
        pain_detector=pain_detector,
        fear_agent=fear_agent,
    ))

if energy_tracker is not None:
    registry.register(EnergyStatusTool(energy_tracker=energy_tracker))

# Scene tools only when vision available
if default_network is not None:
    salience = getattr(default_network, "_salience_network", None)
    attention = getattr(default_network, "_attention_network", None)
    if salience or attention:
        registry.register(SceneSummaryTool(
            salience_network=salience,
            attention_network=attention,
        ))

# Always register system stats (works with whatever is available)
registry.register(SystemStatsTool(
    hippocampus=memory_hub.hippocampus if memory_hub else None,
    nac=nac,
    ec=memory_hub.ec if memory_hub else None,
    atl=memory_hub.atl if memory_hub else None,
    energy_tracker=energy_tracker,
    pain_detector=pain_detector,
    significance_learner=significance_learner,
))
```

---

## Interaction with Other Plans

| Plan | Interaction |
|------|-------------|
| **Learned Tool Index** | Each tool declares MANUAL_KEYWORDS for the index. Goal text like "what happened last time" triggers memory_recall; "is this safe" triggers pain_history |
| **Decision Engine** | AdaptivePlanner queries memory systems automatically. These tools let the LLM do targeted follow-up queries during reasoning |
| **Agent Mesh** | Peer agents could proxy introspection queries: "ask Agent-B what it knows about grab failures" → remote causal_links query |
| **Provenance** | ExplainTool already handles "why did I decide X?". These tools handle "what do I know about X?" — complementary, not overlapping |

---

## Risks

1. **Token cost of tool results.** Memory recalls can be verbose. **Mitigation:** All tools accept `limit` parameter, output is pre-formatted as concise summaries (not raw dataclass dumps), and the LearnedToolIndex ensures only relevant tools get full schemas.

2. **Circular reasoning.** LLM queries memory, gets a memory about a past query, queries about that... **Mitigation:** Introspection tool results are NOT stored as episodic memories. They're ephemeral tool results like any other.

3. **Stale scene data.** Scene summary reflects the last perception cycle, not real-time. **Mitigation:** Include timestamp in the response so the LLM knows how fresh the data is.

4. **Subsystem unavailability.** In headless mode, scene_summary and pain (movement types) return nothing useful. **Mitigation:** Each tool checks its subsystem reference and returns `{"available": false}` gracefully.
