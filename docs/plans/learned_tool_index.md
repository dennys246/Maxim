# Learned Tool Index Plan

> **Status:** Complete (all phases implemented). Phases 1-4 landed on `refactor/tool-learning` branch.

Keyword-weighted hashtable that surfaces relevant tools preferentially in LLM prompts. Auto-extracted keywords bootstrap the index at startup; NAc outcome signals continuously refine weights and discover new keyword associations so the index learns which words actually predict which tools.

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

Registration (auto-extracted + manual):
  GrabTool.keywords = {"grab": 0.5, "pick": 0.5, "hold": 0.5, "object": 0.5}
  ScanTool.keywords = {"scan": 0.5, "look": 0.5, "search": 0.5, "find": 0.5}

Inverted index (hashtable):
  "grab"   → [(GrabTool, 0.5)]
  "pick"   → [(GrabTool, 0.5)]
  "scan"   → [(ScanTool, 0.5)]
  "look"   → [(ScanTool, 0.5)]
  ...

After learning (50 sessions):
  "pick"   → [(GrabTool, 0.92)]          # Strong — always leads to grab
  "cup"    → [(GrabTool, 0.78)]          # LEARNED — not in description, discovered via use
  "mug"    → [(GrabTool, 0.65)]          # LEARNED — synonym discovered from goal text
  "red"    → [(ScanTool, 0.08)]          # Decayed — color rarely predicts scan
  "file"   → [(ReadFileTool, 0.95), (EditFileTool, 0.88), (WriteFileTool, 0.71)]
```

### Weight Update Rule (Rescorla-Wagner inspired)

Three signals drive learning, each with a different semantic meaning:

```python
LEARNING_RATE = 0.1
DECAY_RATE = 0.2
NEW_KEYWORD_INITIAL_WEIGHT = 0.2   # Discovered keywords start low
NEW_KEYWORD_MAX_PER_OUTCOME = 2    # Cap new keywords per tool outcome

# Signal 1: Tool execution SUCCESS
# Keyword correctly predicted this tool — strengthen association.
# Also discover new keywords: goal tokens NOT in the tool's keyword set
# that co-occurred with a successful execution become new learned entries.
for keyword in goal_tokens & tool.keyword_set:
    entry.weight += LEARNING_RATE * (1.0 - entry.weight)
    entry.observations += 1

new_tokens = goal_tokens - tool.keyword_set - STOPWORDS
for token in list(new_tokens)[:NEW_KEYWORD_MAX_PER_OUTCOME]:
    create_entry(token, weight=NEW_KEYWORD_INITIAL_WEIGHT, source="learned")

# Signal 2: Tool execution FAILURE
# The tool was the right CHOICE (the keyword association is correct),
# the execution just failed. This is NAc's domain, not the index's.
# Very weak signal — only count the observation, don't weaken.
for keyword in goal_tokens & tool.keyword_set:
    entry.observations += 1
    # No weight change: "pick"→GrabTool is still correct even if grab failed

# Signal 3: Tool surfaced in prompt but agent chose a DIFFERENT tool
# This is the real signal that the keyword-tool association is wrong.
# The LLM saw the tool and rejected it — keyword is misleading.
for keyword in goal_tokens & tool.keyword_set:
    entry.weight -= LEARNING_RATE * DECAY_RATE * entry.weight
```

**Why failure doesn't weaken keywords:** If the goal is "pick up the cup" and GrabTool fails because the servo overheated, "pick"→GrabTool is still a correct association. Weakening it would teach the index to stop surfacing GrabTool for pick-up tasks. The failure signal belongs to NAc (which learns "GrabTool fails in this context"), not the keyword index (which learns "the word pick means use GrabTool").

**Convergence:** With α=0.1, a consistently-correct keyword reaches weight 0.9 in ~22 observations. A consistently-unused keyword decays to near-zero in ~22 observations. Discovered keywords start at 0.2 and climb to full strength if they keep co-occurring with successful tool use.

### Score Normalization

Raw scores are the sum of matched keyword weights. This favors tools with verbose descriptions (more keywords = more chances to match). Normalize by keyword count:

```python
# Raw: sum of matched keyword weights
raw_score = sum(entry.weight for entry in matched_entries)

# Normalized: average weight of matched keywords, scaled by match count
# This rewards tools where MOST matched keywords are strong,
# not tools that happen to have many weak keywords.
if matched_count > 0:
    avg_weight = raw_score / matched_count
    # Boost slightly for multiple matches (1 match = 1x, 3 matches = ~1.4x)
    score = avg_weight * (1.0 + 0.2 * min(matched_count - 1, 3))
else:
    score = 0.0
```

### Prompt Integration

At prompt build time, tokenize the current goal and query the index. **Always surface at least MIN_TOOLS tools** regardless of score, so the LLM always has something to work with:

```python
MIN_TOOLS = 3               # Always surface at least this many
RELEVANCE_THRESHOLD = 0.3   # Additional tools above this score

goal_tokens = tokenize("pick up the red cup")
tool_scores = index.score_tools("pick up the red cup")
# Result: {"grab": 0.85, "move": 0.31, "scan": 0.08, "read_file": 0.0, ...}

# Partition: top MIN_TOOLS always get full schema,
# plus any additional tools above threshold
sorted_tools = sorted(tool_scores.items(), key=lambda x: -x[1])
relevant = set()
for i, (name, score) in enumerate(sorted_tools):
    if i < MIN_TOOLS or score >= RELEVANCE_THRESHOLD:
        relevant.add(name)
    else:
        break
background = [name for name in all_tools if name not in relevant]
```

---

## Implementation

### Phase 1: ToolKeywordEntry + Auto-Extraction

```python
import re
import threading
from dataclasses import dataclass


@dataclass
class ToolKeywordEntry:
    """A keyword associated with a tool, with learned weight."""
    word: str
    weight: float = 0.5      # 0.0 = never relevant, 1.0 = always relevant
    source: str = "auto"     # "auto" (extracted), "manual" (declared), or "learned" (discovered)
    observations: int = 0    # Times this word co-occurred with tool execution


class LearnedToolIndex:
    """Keyword-weighted hashtable for tool relevance scoring.

    Auto-extracts keywords from tool metadata at registration time.
    Learns from tool execution outcomes to refine weights and discover
    new keyword associations.

    Thread-safe: all mutations to _index and _tool_keywords are
    protected by a lock.
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
    RELEVANCE_THRESHOLD = 0.3
    MIN_TOOLS = 3                    # Always surface at least this many
    NEW_KEYWORD_INITIAL_WEIGHT = 0.2
    NEW_KEYWORD_MAX_PER_OUTCOME = 2

    def __init__(self) -> None:
        # Per-tool keyword dictionaries
        self._tool_keywords: dict[str, dict[str, ToolKeywordEntry]] = {}
        # Inverted index: word → [(tool_name, ToolKeywordEntry)]
        self._index: dict[str, list[tuple[str, ToolKeywordEntry]]] = {}
        self._lock = threading.Lock()

    def register_tool(self, tool) -> None:
        """Auto-extract keywords from tool metadata and register.

        Safe to call multiple times for the same tool — skips if
        already registered (preserves learned weights).
        """
        name = tool.name
        with self._lock:
            if name in self._tool_keywords:
                return  # Already registered — don't clobber learned weights
            keywords = self._extract_keywords(tool)
            self._tool_keywords[name] = {}
            for word in keywords:
                entry = ToolKeywordEntry(word=word, weight=0.5, source="auto")
                self._tool_keywords[name][word] = entry
                self._index.setdefault(word, []).append((name, entry))

    def register_manual_keywords(self, tool_name: str, keywords: set[str]) -> None:
        """Add manually declared keywords (higher initial weight)."""
        with self._lock:
            for word in keywords:
                word = word.lower()
                if tool_name in self._tool_keywords and word in self._tool_keywords[tool_name]:
                    continue  # Already exists
                entry = ToolKeywordEntry(word=word, weight=0.7, source="manual")
                self._tool_keywords.setdefault(tool_name, {})[word] = entry
                self._index.setdefault(word, []).append((tool_name, entry))

    def score_tools(self, goal_text: str) -> dict[str, float]:
        """Score all tools against a goal string.

        Returns {tool_name: normalized_relevance_score} sorted descending.
        Scores are normalized by keyword count to avoid favoring verbose tools.
        """
        tokens = self._tokenize(goal_text)
        # Collect matched entries per tool
        tool_matches: dict[str, list[float]] = {}
        with self._lock:
            for token in tokens:
                entries = self._index.get(token, [])
                for tool_name, entry in entries:
                    tool_matches.setdefault(tool_name, []).append(entry.weight)

        # Normalize: avg weight * match count boost
        scores: dict[str, float] = {}
        for tool_name, weights in tool_matches.items():
            avg_weight = sum(weights) / len(weights)
            match_boost = 1.0 + 0.2 * min(len(weights) - 1, 3)
            scores[tool_name] = avg_weight * match_boost

        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    def get_relevant_tools(self, goal_text: str) -> tuple[list[str], list[str]]:
        """Partition tools into relevant (full schema) and background (name only).

        Always returns at least MIN_TOOLS relevant tools regardless of score.
        Returns (relevant_tool_names, background_tool_names).
        """
        scores = self.score_tools(goal_text)
        sorted_tools = sorted(scores.items(), key=lambda x: -x[1])

        relevant = []
        for i, (name, score) in enumerate(sorted_tools):
            if i < self.MIN_TOOLS or score >= self.RELEVANCE_THRESHOLD:
                relevant.append(name)
            else:
                break

        # If fewer tools scored than MIN_TOOLS, pad with unscored tools
        if len(relevant) < self.MIN_TOOLS:
            with self._lock:
                remaining = [n for n in self._tool_keywords if n not in set(relevant)]
            relevant.extend(remaining[: self.MIN_TOOLS - len(relevant)])

        relevant_set = set(relevant)
        with self._lock:
            background = [n for n in self._tool_keywords if n not in relevant_set]
        return relevant, background

    # ── Learning ──────────────────────────────────────────────

    def record_outcome(
        self,
        goal_text: str,
        tool_name: str,
        success: bool,
    ) -> None:
        """Update keyword weights based on tool execution outcome.

        On success: strengthen existing keyword associations AND discover
        new keywords from goal tokens not yet in the tool's keyword set.

        On failure: count the observation but DON'T weaken the weight.
        Tool failure means the execution failed, not that the keyword
        association is wrong. Keyword weakening is handled by
        record_surfaced_but_unused (agent chose a different tool).
        """
        tokens = self._tokenize(goal_text)
        with self._lock:
            tool_kw = self._tool_keywords.get(tool_name)
            if not tool_kw:
                return

            matched_keywords = tokens & set(tool_kw.keys())
            for word in matched_keywords:
                entry = tool_kw[word]
                entry.observations += 1
                if success:
                    entry.weight += self.LEARNING_RATE * (1.0 - entry.weight)
                # Failure: observation counted, no weight change
                entry.weight = max(0.01, min(1.0, entry.weight))

            # Discover new keywords from successful executions
            if success:
                new_tokens = tokens - set(tool_kw.keys())
                created = 0
                for word in new_tokens:
                    if created >= self.NEW_KEYWORD_MAX_PER_OUTCOME:
                        break
                    if len(word) <= 2 or word in self.STOPWORDS:
                        continue
                    entry = ToolKeywordEntry(
                        word=word,
                        weight=self.NEW_KEYWORD_INITIAL_WEIGHT,
                        source="learned",
                        observations=1,
                    )
                    tool_kw[word] = entry
                    self._index.setdefault(word, []).append((tool_name, entry))
                    created += 1

    def record_surfaced_but_unused(
        self,
        goal_text: str,
        surfaced_tools: list[str],
        used_tool: str,
    ) -> None:
        """Decay keywords for tools that were surfaced but the agent chose differently.

        This is the primary negative signal: the LLM saw the tool schema
        and decided not to use it. The keyword association is misleading.
        """
        tokens = self._tokenize(goal_text)
        with self._lock:
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
        """Auto-extract keywords from tool name, description, and params.

        Indexes both the full tool name (for exact matches) and
        underscore/camelCase-split components.
        """
        text_parts = []
        # Full tool name (e.g., "edit_file")
        if hasattr(tool, "name") and tool.name:
            text_parts.append(tool.name)
            # Also split compound names: "edit_file" → "edit", "file"
            text_parts.extend(tool.name.replace("_", " ").replace("-", " ").split())
        if hasattr(tool, "description") and tool.description:
            text_parts.append(tool.description)
        # Parameter names are often good keywords
        if hasattr(tool, "input_schema") and isinstance(tool.input_schema, dict):
            text_parts.extend(tool.input_schema.keys())

        raw = " ".join(text_parts).lower()
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
        """Persist learned weights to JSON (atomic write)."""
        import json
        import os

        with self._lock:
            data = {}
            for tool_name, keywords in self._tool_keywords.items():
                data[tool_name] = {
                    word: {
                        "weight": round(entry.weight, 4),
                        "source": entry.source,
                        "observations": entry.observations,
                    }
                    for word, entry in keywords.items()
                }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        """Load learned weights from JSON.

        Only updates weights for tools that are currently registered.
        Restores learned keywords (source="learned") that were discovered
        in previous sessions. New tools get auto-extracted defaults.
        Removed tools are ignored.
        """
        import json

        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        with self._lock:
            for tool_name, keywords in data.items():
                if tool_name not in self._tool_keywords:
                    continue  # Tool no longer registered
                for word, kw_data in keywords.items():
                    if word in self._tool_keywords[tool_name]:
                        # Update existing entry
                        entry = self._tool_keywords[tool_name][word]
                        entry.weight = kw_data.get("weight", entry.weight)
                        entry.observations = kw_data.get("observations", 0)
                        entry.source = kw_data.get("source", entry.source)
                    elif kw_data.get("source") == "learned":
                        # Restore learned keyword from previous session
                        entry = ToolKeywordEntry(
                            word=word,
                            weight=kw_data.get("weight", self.NEW_KEYWORD_INITIAL_WEIGHT),
                            source="learned",
                            observations=kw_data.get("observations", 0),
                        )
                        self._tool_keywords[tool_name][word] = entry
                        self._index.setdefault(word, []).append((tool_name, entry))
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

The surfaced tools list is passed through the decision pipeline via the `decision` dict returned by `DecisionEngine.decide()`, avoiding fragile shared state:

```python
# In DecisionEngine.decide(), add surfaced tools to the return dict:
return {
    "action": next_action,
    "plan": best_plan,
    "score": best_score,
    "surfaced_tools": [name for name in relevant_tool_names],  # From prompt builder
}

# In run_agentic_loop(), in the on_step callback:
if tool_index is not None:
    goal_text = str(intent.get("goal", ""))
    used_tool = action.get("tool_name", "")
    surfaced = decision.get("surfaced_tools", [])
    if surfaced and used_tool:
        tool_index.record_surfaced_but_unused(goal_text, surfaced, used_tool)
```

### Phase 3: Prompt Integration (Remaining Work)

Wire the learned index into the prompt building pipeline. Five sub-steps identified from code audit:

#### 3a. Split tool section by priority in prompt builder

**File:** `agents/prompt_builder.py`, method `_build_tool_aware_prompt()`, line ~701

Currently all tools are added as one CRITICAL section. Replace with:

```python
if self._tool_index is not None:
    goal_text = question_text  # Already extracted on line 669
    relevant, background = self._tool_index.get_relevant_tools(goal_text)
    request.surfaced_tools = relevant

    relevant_section = build_tools_section_filtered(request, relevant, mode_name)
    budgeter.add("tools", relevant_section, SectionPriority.CRITICAL,
                  truncatable=True, min_tokens=50,
                  truncate_fn=lambda c, m: _truncate_tool_guidance(c, m, counter))

    if background:
        bg_section = f"Other tools available: {', '.join(background)}"
        budgeter.add("tools_background", bg_section, SectionPriority.NICE_TO_HAVE)
else:
    # Fallback: existing behavior
    budgeter.add("tools", build_tools_section(request, mode_name=mode_name),
                  SectionPriority.CRITICAL, truncatable=True, min_tokens=50,
                  truncate_fn=lambda c, m: _truncate_tool_guidance(c, m, counter))
```

#### 3b. New filtered tool formatter

**File:** `agents/prompt_builder.py`, new function after `build_tools_section()` (line ~177)

```python
def build_tools_section_filtered(
    request: LLMRequest, tool_names: list[str], mode_name: str = "passive",
) -> str:
    """Build tools section for only the specified tool names."""
    lines = ["=== Available Tools ==="]
    for tool_name in sorted(tool_names):
        tool_info = request.tool_descriptions.get(tool_name, {})
        if isinstance(tool_info, dict) and tool_info:
            desc = tool_info.get("description", "")
            params = tool_info.get("params", {})
            lines.append(f"- {tool_name}: {desc}")
            if params:
                for p_name, p_info in params.items():
                    required = "(required)" if p_info.get("required") else ""
                    lines.append(f"    {p_name}: {p_info.get('type', '?')} {required}")
    return "\n".join(lines)
```

#### 3c. Add surfaced_tools field to LLMRequest

**File:** `agents/llm_types.py`, class `LLMRequest` (line ~108)

```python
surfaced_tools: list[str] = field(default_factory=list)
```

#### 3d. Pass tool_index through LLMWorker to PromptBuilder

**Files:** `agents/prompt_builder.py` (`__init__`), `agents/llm_worker.py` (`__init__`), `conscience/agentic_runtime.py` (LLMWorker creation)

```python
# prompt_builder.py:
class PromptBuilder:
    def __init__(self, ..., tool_index=None):
        self._tool_index = tool_index

# llm_worker.py:
class LLMWorker:
    def __init__(self, ..., tool_index=None):
        self._tool_index = tool_index
        self._prompt_builder = PromptBuilder(..., tool_index=tool_index)

# agentic_runtime.py (where LLMWorker is created):
llm_worker = LLMWorker(..., tool_index=tool_index)
```

#### 3e. Surfaced-but-unused decay signal in agent loop

**File:** `runtime/agent_loop.py`, in on_step callback after tool execution

```python
if tool_index is not None:
    goal_text = str(intent.get("goal", ""))
    used_tool = action.get("tool_name", "")
    surfaced = decision.get("surfaced_tools", [])
    if surfaced and used_tool and goal_text:
        tool_index.record_surfaced_but_unused(goal_text, surfaced, used_tool)
```

#### Integration map

| Step | File | What changes |
|------|------|-------------|
| 3a | prompt_builder.py:~701 | Split tool section into CRITICAL (relevant) + NICE_TO_HAVE (background) |
| 3b | prompt_builder.py:~177 | New `build_tools_section_filtered()` function |
| 3c | llm_types.py:~108 | Add `surfaced_tools` field to `LLMRequest` |
| 3d | prompt_builder.py, llm_worker.py, agentic_runtime.py | Pass `tool_index` through the chain |
| 3e | agent_loop.py | Call `record_surfaced_but_unused()` in on_step callback |

### Phase 4: Persistence + Session Lifecycle — IMPLEMENTED

```python
# At startup (in agentic_runtime.py) — DONE:
tool_index = LearnedToolIndex()
for tool in registry.all_tools():
    tool_index.register_tool(tool)
tool_index.load(f"{data_dir}/memory/tool_index.json")

# At shutdown (in _stop_agentic_runtime) — DONE:
tool_index.save(f"{data_dir}/memory/tool_index.json")
```

---

## Implementation Sequencing

| Phase | What | Effort | Dependencies |
|-------|------|--------|-------------|
| **1** | `ToolKeywordEntry` + `LearnedToolIndex` with auto-extraction, normalization, thread safety | Small | ToolRegistry |
| **2** | Hook into ToolPainBridge + agent loop for weight updates + keyword discovery | Small | Phase 1, ToolPainBridge |
| **3** | Prompt integration (tool relevance partitioning, surfaced list passthrough) | Medium | Phase 1, PromptBudgeter |
| **4** | Persistence (save/load with learned keyword restoration) | Small | Phase 1 |

Phases 1 and 4 can be done together. Phase 2 and 3 are independent of each other.

---

## Interaction with Other Systems

| System | Interaction |
|--------|-------------|
| **NAc / ToolPainBridge** | Outcome signals (success/unused) drive keyword weight updates. Failure signals are NAc's domain, not the index's |
| **AdaptivePlanner** | Planner's tool scoring (NAc predictions) is separate from keyword relevance. Both inform tool choice but at different stages: index filters the prompt, planner scores the candidates |
| **ConceptContextBuilder** | `rank_available_skills()` uses ATL graph edges (semantic). The keyword index uses lexical matching (syntactic). They complement each other — semantic catches "grasp"→GrabTool, lexical catches "cup"→GrabTool |
| **PromptBudgeter** | Tool sections get priority based on index scores. Relevant tools are CRITICAL, background tools are NICE_TO_HAVE |
| **Agent Mesh** | Learned keyword weights could be shared between peers as part of knowledge exchange (lightweight — just the weight dict, not the full index) |

---

## Risks

1. **Cold start.** New tools have all keywords at weight 0.5, so everything looks equally relevant. **Mitigation:** `MIN_TOOLS=3` guarantees the top 3 always get full schemas. Weights diverge within 5-10 tool executions. First few sessions behave like existing behavior (all tools visible).

2. **Keyword drift.** If the agent's tasks shift (e.g., from robot control to coding), old keyword weights may be stale. **Mitigation:** The surfaced-but-unused decay gradually lowers irrelevant weights. Auto-extracted keywords ensure new tools are always discoverable. Learned keywords from prior sessions are restored at load time but start decaying if unused.

3. **Over-filtering.** Aggressive threshold could hide a tool the LLM would have chosen. **Mitigation:** `MIN_TOOLS=3` ensures the LLM always sees at least 3 full schemas. Background tools are still listed by name — the LLM can request any tool, it just doesn't see the full schema. If it picks a background tool, that tool's keywords get strengthened on the next success.

4. **Tokenization mismatch.** Simple regex tokenization may miss compound terms or split them wrong. **Mitigation:** `_extract_keywords` indexes both the full tool name and underscore/dash-split components ("edit_file" → {"edit_file", "edit", "file"}). Goal tokenization uses the same regex. Learned keyword discovery (Phase 2) catches synonyms and domain-specific terms that auto-extraction misses.

5. **Memory cost.** 20 tools × 10 keywords each = 200 entries. With learned keyword discovery adding ~2 per outcome, after 100 sessions that's ~400 additional entries. Total ~600 entries = a few KB. Negligible.

6. **Learned keyword noise.** Discovery creates entries for goal tokens that happen to co-occur with tool success but aren't semantically meaningful (e.g., "the"→GrabTool if "the" survived stopword filtering). **Mitigation:** Stopword filter catches common words. `NEW_KEYWORD_MAX_PER_OUTCOME=2` caps per-outcome creation. Learned keywords start at 0.2 weight — they only climb if they consistently co-occur. The surfaced-but-unused decay cleans up noise that gets surfaced but never chosen.

7. **Thread contention under heavy load.** The lock is held during `score_tools` (read path, called on prompt build) and `record_outcome` (write path, called on tool complete). **Mitigation:** Both operations are O(tokens × entries per token), which is microseconds for typical inputs. If contention becomes measurable, upgrade to a RWLock (readers don't block each other).

---

## Expected Savings

| Scenario | Tools in prompt | Tokens used | Savings |
|----------|----------------|-------------|---------|
| Current (all tools, full schema) | 20 | ~500 | — |
| With index (3 relevant + 17 names) | 3 full + 17 names | ~130 | ~370 tokens (74%) |
| Under token pressure (3 relevant only) | 3 full | ~90 | ~410 tokens (82%) |

At 4096 context window, saving 370 tokens means ~9% more room for conversation history, memory context, and plan progress — the sections that actually help the LLM make better decisions.
