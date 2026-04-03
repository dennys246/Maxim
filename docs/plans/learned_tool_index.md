# Learned Tool Index Plan

> **Status:** Not started. Depends on ToolRegistry (implemented), NAc RPE flow (implemented), ToolPainBridge (implemented), PromptBudgeter section priorities (implemented).

Keyword-weighted hashtable that surfaces relevant tools preferentially in LLM prompts. Auto-extracted keywords bootstrap the index at startup; NAc outcome signals continuously refine weights so the index learns which words actually predict which tools.

---

## Motivation

The full tool registry gets dumped into every LLM prompt regardless of relevance. With 20+ tools, that's hundreds of tokens spent on tools the LLM will never pick for a given goal. The prompt budgeter may drop important context sections (conversation history, memory, plan progress) to make room for irrelevant tool schemas.

**Current prompt cost:** ~15-30 tokens per tool (name + description + params). 20 tools = 300-600 tokens of tool context per prompt. At 4096 context window on a 7B model, that's 7-15% of the budget.

**With learned index:** Only matched tools get full schemas (CRITICAL priority). Unmatched tools get a one-line name (NICE_TO_HAVE, dropped first under pressure). Typical goal matches 2-4 tools → full schemas cost ~60-120 tokens. Saves 200-400 tokens per prompt for actual reasoning context.

---

## Design

### The Index

```
Per-tool keyword dictionaries, inverted into a shared lookup table.

Registration:
  GrabTool.keywords = {"grab": 0.5, "pick": 0.5, "hold": 0.5, "object": 0.5, "cup": 0.5}
  ScanTool.keywords = {"scan": 0.5, "look": 0.5, "search": 0.5, "find": 0.5}

Inverted index (hashtable):
  "grab"   → [(GrabTool, 0.5)]
  "pick"   → [(GrabTool, 0.5)]
  "scan"   → [(ScanTool, 0.5)]
  "look"   → [(ScanTool, 0.5)]
  "find"   → [(ScanTool, 0.5)]
  ...

After learning (50 sessions):
  "pick"   → [(GrabTool, 0.92)]          # Strong — always leads to grab
  "cup"    → [(GrabTool, 0.78)]          # Strong — object-tool association learned
  "red"    → [(ScanTool, 0.08)]          # Decayed — color rarely predicts scan
  "file"   → [(ReadFileTool, 0.95), (EditFileTool, 0.88), (WriteFileTool, 0.71)]
```

### Weight Update Rule (Rescorla-Wagner inspired)

```python
LEARNING_RATE = 0.1
DECAY_RATE = 0.2

# On tool execution SUCCESS:
# Strengthen keywords from the goal that matched this tool
for keyword in goal_tokens & tool.keyword_set:
    entry.weight += LEARNING_RATE * (1.0 - entry.weight)
    entry.observations += 1

# On tool execution FAILURE:
# Weaken keywords — this tool wasn't the right choice
for keyword in goal_tokens & tool.keyword_set:
    entry.weight -= LEARNING_RATE * entry.weight

# On prompt build (tool surfaced but NOT used):
# Gentle decay — keyword triggered the tool but agent chose something else
for keyword in goal_tokens & tool.keyword_set:
    entry.weight -= LEARNING_RATE * DECAY_RATE * entry.weight
```

**Convergence:** With α=0.1, a consistently-correct keyword reaches weight 0.9 in ~22 observations. A consistently-wrong keyword decays to 0.1 in ~22 observations. Mixed signals converge to the empirical probability of the keyword predicting the tool.

### Prompt Integration

At prompt build time, tokenize the current goal and query the index:

```python
goal_tokens = tokenize("pick up the red cup")
# Lookup: {"pick", "up", "the", "red", "cup"} against index

tool_scores: dict[str, float] = {}
for token in goal_tokens:
    for tool, weight in index.lookup(token):
        tool_scores[tool.name] = tool_scores.get(tool.name, 0) + weight

# Result: {"grab": 1.7, "scan": 0.08, "read_file": 0.0, ...}

# Partition by score:
#   score > threshold (0.3) → CRITICAL: full schema in prompt
#   score ≤ threshold       → NICE_TO_HAVE: name only, dropped under pressure
```

---

## Implementation

### Phase 1: ToolKeywordEntry + Auto-Extraction

```python
@dataclass
class ToolKeywordEntry:
    """A keyword associated with a tool, with learned weight."""
    word: str
    weight: float = 0.5      # 0.0 = never relevant, 1.0 = always relevant
    source: str = "auto"     # "auto" (extracted) or "manual" (declared)
    observations: int = 0    # Times this word co-occurred with tool execution


class LearnedToolIndex:
    """Keyword-weighted hashtable for tool relevance scoring.

    Auto-extracts keywords from tool metadata at registration time.
    Learns from NAc outcome signals to refine which keywords predict
    which tools.
    """

    STOPWORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "nor", "not", "no", "so", "if", "than", "that",
        "this", "it", "its", "use", "used", "using",
    })

    LEARNING_RATE = 0.1
    DECAY_RATE = 0.2
    RELEVANCE_THRESHOLD = 0.3  # Minimum score to surface full schema

    def __init__(self) -> None:
        # Per-tool keyword dictionaries
        self._tool_keywords: dict[str, dict[str, ToolKeywordEntry]] = {}
        # Inverted index: word → [(tool_name, ToolKeywordEntry)]
        self._index: dict[str, list[tuple[str, ToolKeywordEntry]]] = {}

    def register_tool(self, tool) -> None:
        """Auto-extract keywords from tool metadata and register."""
        name = tool.name
        keywords = self._extract_keywords(tool)
        self._tool_keywords[name] = {}

        for word in keywords:
            entry = ToolKeywordEntry(word=word, weight=0.5, source="auto")
            self._tool_keywords[name][word] = entry
            self._index.setdefault(word, []).append((name, entry))

    def register_manual_keywords(self, tool_name: str, keywords: set[str]) -> None:
        """Add manually declared keywords (higher initial weight)."""
        for word in keywords:
            word = word.lower()
            entry = ToolKeywordEntry(word=word, weight=0.7, source="manual")
            self._tool_keywords.setdefault(tool_name, {})[word] = entry
            self._index.setdefault(word, []).append((tool_name, entry))

    def score_tools(self, goal_text: str) -> dict[str, float]:
        """Score all tools against a goal string.

        Returns {tool_name: relevance_score} sorted descending.
        Scores are sum of matched keyword weights.
        """
        tokens = self._tokenize(goal_text)
        scores: dict[str, float] = {}
        for token in tokens:
            entries = self._index.get(token, [])
            for tool_name, entry in entries:
                scores[tool_name] = scores.get(tool_name, 0.0) + entry.weight
        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    def get_relevant_tools(self, goal_text: str) -> tuple[list[str], list[str]]:
        """Partition tools into relevant (full schema) and background (name only).

        Returns (relevant_tool_names, background_tool_names).
        """
        scores = self.score_tools(goal_text)
        relevant = [name for name, score in scores.items() if score >= self.RELEVANCE_THRESHOLD]
        background = [name for name in self._tool_keywords if name not in set(relevant)]
        return relevant, background

    # ── Learning ──────────────────────────────────────────────

    def record_outcome(
        self,
        goal_text: str,
        tool_name: str,
        success: bool,
    ) -> None:
        """Update keyword weights based on tool execution outcome.

        Called after every tool execution (hook into ToolPainBridge
        or agent_loop on_step callback).
        """
        tokens = self._tokenize(goal_text)
        tool_kw = self._tool_keywords.get(tool_name)
        if not tool_kw:
            return

        matched_keywords = tokens & set(tool_kw.keys())
        for word in matched_keywords:
            entry = tool_kw[word]
            entry.observations += 1
            if success:
                # Strengthen: keyword correctly predicted this tool
                entry.weight += self.LEARNING_RATE * (1.0 - entry.weight)
            else:
                # Weaken: keyword led to this tool but it failed
                entry.weight -= self.LEARNING_RATE * entry.weight
            # Clamp to [0.01, 1.0] — never fully zero (allow recovery)
            entry.weight = max(0.01, min(1.0, entry.weight))

    def record_surfaced_but_unused(
        self,
        goal_text: str,
        surfaced_tools: list[str],
        used_tool: str,
    ) -> None:
        """Gently decay keywords for tools that were surfaced but not chosen.

        Called after tool execution for all tools that scored above
        threshold but weren't the one actually executed.
        """
        tokens = self._tokenize(goal_text)
        for tool_name in surfaced_tools:
            if tool_name == used_tool:
                continue
            tool_kw = self._tool_keywords.get(tool_name)
            if not tool_kw:
                continue
            matched_keywords = tokens & set(tool_kw.keys())
            for word in matched_keywords:
                entry = tool_kw[word]
                entry.weight -= self.LEARNING_RATE * self.DECAY_RATE * entry.weight
                entry.weight = max(0.01, min(1.0, entry.weight))

    # ── Keyword extraction ────────────────────────────────────

    def _extract_keywords(self, tool) -> set[str]:
        """Auto-extract keywords from tool name, description, and params."""
        text_parts = [tool.name]
        if hasattr(tool, "description") and tool.description:
            text_parts.append(tool.description)
        # Parameter names are often good keywords
        if hasattr(tool, "input_schema") and isinstance(tool.input_schema, dict):
            text_parts.extend(tool.input_schema.keys())

        raw = " ".join(text_parts).lower()
        # Split on non-alphanumeric, filter stopwords and short tokens
        tokens = set()
        for token in re.split(r"[^a-z0-9]+", raw):
            if len(token) > 2 and token not in self.STOPWORDS:
                tokens.add(token)
        return tokens

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize goal text for index lookup."""
        raw = text.lower()
        tokens = set()
        for token in re.split(r"[^a-z0-9]+", raw):
            if len(token) > 2 and token not in self.STOPWORDS:
                tokens.add(token)
        return tokens

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist learned weights to JSON."""
        data = {}
        for tool_name, keywords in self._tool_keywords.items():
            data[tool_name] = {
                word: {
                    "weight": entry.weight,
                    "source": entry.source,
                    "observations": entry.observations,
                }
                for word, entry in keywords.items()
            }
        import json, os
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        """Load learned weights from JSON.

        Only updates weights for tools that are currently registered.
        New tools get auto-extracted defaults. Removed tools are ignored.
        """
        import json
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        for tool_name, keywords in data.items():
            if tool_name not in self._tool_keywords:
                continue  # Tool no longer registered
            for word, kw_data in keywords.items():
                if word in self._tool_keywords[tool_name]:
                    entry = self._tool_keywords[tool_name][word]
                    entry.weight = kw_data.get("weight", entry.weight)
                    entry.observations = kw_data.get("observations", 0)
                    entry.source = kw_data.get("source", entry.source)
```

### Phase 2: Hook into ToolPainBridge + Agent Loop

Two integration points for the weight update signals:

#### 2a. Tool outcome → keyword weight update

```python
# In ToolPainBridge.record_tool_complete() and _on_pain(),
# after NAc outcome recording:

if self._tool_index is not None:
    goal_text = context.get("goal", "") if context else ""
    if goal_text:
        self._tool_index.record_outcome(
            goal_text=goal_text,
            tool_name=tool_name,
            success=success,
        )
```

**Wiring:** `ToolPainBridge.__init__()` gets optional `tool_index: LearnedToolIndex | None = None`.

#### 2b. Surfaced-but-unused decay

```python
# In run_agentic_loop(), after tool execution in the on_step callback:

if tool_index is not None:
    goal_text = str(intent.get("goal", ""))
    used_tool = action.get("tool_name", "")
    # Get which tools were surfaced in the prompt for this goal
    surfaced = state.data.get("_surfaced_tools", [])
    if surfaced and used_tool:
        tool_index.record_surfaced_but_unused(goal_text, surfaced, used_tool)
```

### Phase 3: Prompt Integration

The prompt builder uses tool scores to partition tools into priority tiers:

```python
# In prompt building (wherever tool schemas are injected into the prompt):

def build_tool_context(
    tool_index: LearnedToolIndex,
    tool_registry: ToolRegistry,
    goal_text: str,
) -> tuple[str, list[str]]:
    """Build tool context for LLM prompt with learned relevance.

    Returns (prompt_section, surfaced_tool_names).
    """
    relevant, background = tool_index.get_relevant_tools(goal_text)

    parts = []

    # CRITICAL: Full schema for relevant tools
    if relevant:
        parts.append("## Available Tools (relevant to current goal)")
        for name in relevant:
            tool = tool_registry.get(name)
            if tool:
                parts.append(f"\n### {tool.name}")
                parts.append(f"{tool.description}")
                if hasattr(tool, "input_schema"):
                    parts.append(f"Parameters: {tool.input_schema}")

    # NICE_TO_HAVE: Name-only list for background tools
    if background:
        parts.append(f"\nOther tools available: {', '.join(background)}")

    return "\n".join(parts), relevant
```

The `surfaced_tool_names` list gets stored in `state.data["_surfaced_tools"]` so Phase 2b can compute the unused-decay signal.

### Phase 4: Persistence + Session Lifecycle

```python
# At startup (in agentic_runtime.py):
tool_index = LearnedToolIndex()
for tool in registry.all_tools():
    tool_index.register_tool(tool)
tool_index.load(f"{data_dir}/memory/tool_index.json")

# At shutdown (in _stop_agentic_runtime):
tool_index.save(f"{data_dir}/memory/tool_index.json")

# Periodic auto-save (every 60s, alongside hippocampus auto-save):
# Same atomic tmp+replace pattern used by Hippocampus and NAc.
```

---

## Implementation Sequencing

| Phase | What | Effort | Dependencies |
|-------|------|--------|-------------|
| **1** | `ToolKeywordEntry` + `LearnedToolIndex` with auto-extraction | Small | ToolRegistry |
| **2** | Hook into ToolPainBridge + agent loop for weight updates | Small | Phase 1, ToolPainBridge |
| **3** | Prompt integration (tool relevance partitioning) | Medium | Phase 1, PromptBudgeter |
| **4** | Persistence (save/load learned weights) | Small | Phase 1 |

Phases 1 and 4 can be done together. Phase 2 and 3 are independent of each other.

---

## Interaction with Other Systems

| System | Interaction |
|--------|-------------|
| **NAc / ToolPainBridge** | Outcome signals (success/failure) drive keyword weight updates |
| **AdaptivePlanner** | Planner's tool scoring (NAc predictions) is separate from keyword relevance. Both inform tool choice but at different stages: index filters the prompt, planner scores the candidates |
| **ConceptContextBuilder** | `rank_available_skills()` uses ATL graph edges (semantic). The keyword index uses lexical matching (syntactic). They complement each other — semantic catches "grasp"→GrabTool, lexical catches "cup"→GrabTool |
| **PromptBudgeter** | Tool sections get priority based on index scores. Relevant tools are CRITICAL, background tools are NICE_TO_HAVE |
| **Agent Mesh** | Learned keyword weights could be shared between peers as part of knowledge exchange (lightweight — just the weight dict, not the full index) |

---

## Risks

1. **Cold start.** New tools have all keywords at weight 0.5, so everything looks equally relevant. **Mitigation:** First few sessions use all tools at CRITICAL priority (existing behavior). Weights diverge within 5-10 tool executions.

2. **Keyword drift.** If the agent's tasks shift (e.g., from robot control to coding), old keyword weights may be stale. **Mitigation:** The decay mechanism on surfaced-but-unused tools gradually lowers irrelevant weights. Additionally, auto-extracted keywords ensure new tools are always discoverable.

3. **Over-filtering.** Aggressive threshold could hide a tool the LLM would have chosen. **Mitigation:** Background tools are still listed by name in the prompt — the LLM can request any tool, it just doesn't see the full schema. If it picks a background tool, that tool's keywords get strengthened on the next success.

4. **Tokenization mismatch.** Simple regex tokenization may miss compound terms or split them wrong ("read_file" → "read", "file"). **Mitigation:** Also index the full tool name as a keyword. Underscore-split names generate both the full name and components.

5. **Memory cost.** 20 tools × 10 keywords each = 200 entries. Negligible. Even 100 tools × 50 keywords = 5000 entries fits in a few KB.

---

## Expected Savings

| Scenario | Tools in prompt | Tokens used | Savings |
|----------|----------------|-------------|---------|
| Current (all tools, full schema) | 20 | ~500 | — |
| With index (3 relevant + 17 names) | 3 full + 17 names | ~130 | ~370 tokens (74%) |
| Under token pressure (3 relevant only) | 3 full | ~90 | ~410 tokens (82%) |

At 4096 context window, saving 370 tokens means ~9% more room for conversation history, memory context, and plan progress — the sections that actually help the LLM make better decisions.
