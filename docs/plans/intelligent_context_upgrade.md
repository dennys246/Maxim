# Intelligent Context Upgrade Plan

Incremental improvements to how Maxim's LLM manages context: disambiguating edits, pinning important turns, and surviving context compaction without losing critical information. Split from the claw-code upgrade (now implemented and documented in ARCHITECTURE.md/DECISIONS.md) because these features require iterative prompt tuning and behavioral validation that can't be verified by code review alone.

---

## Motivation

Two context problems surface during long-horizon coding tasks:

1. **Edit disambiguation:** `old_text` like `return None` appears 5 times in a file. The LLM needs a way to specify WHICH one. Without it, edits fail or hit the wrong location.

2. **Context loss during compaction:** Over a 20-phase plan, the prompt budgeter drops conversation history to fit the token budget. The LLM loses the original goal, early constraints, and foundational decisions. It starts contradicting itself.

Both require LLM behavioral tuning — the mechanism is simple but getting the LLM to USE it well requires iteration.

---

## Part 1: Edit Disambiguation (context_before / context_after)

### The Problem

EditFileTool (#11a from claw-code plan) uses `old_text` / `new_text` anchors. Common patterns appear many times:

```python
return None      # 5 occurrences
pass             # 12 occurrences
self.x = x       # 3 occurrences
```

Without disambiguation, the LLM must guess `expected_count` or the edit hits the wrong location.

### The Mechanism

Add optional `context_before` and `context_after` params to EditFileTool:

```python
input_schema = {
    "path": str,
    "old_text": str,
    "new_text": str,
    "expected_count": (int, 1),
    "context_before": (str, None),   # Text that must appear before old_text
    "context_after": (str, None),    # Text that must appear after old_text
}
```

When provided, build a full search pattern `context_before + old_text + context_after` and replace only the `old_text` portion within the matched context.

### Error Messages That Teach

When disambiguation fails, return actionable error information:

```python
# No match:
"old_text not found in file.\nNear matches:\n  line 12: def foo(): return None\n  line 47: if x: return None"

# Multiple matches:
"Expected 1 occurrence, found 3 at lines [12, 47, 89]. Use context_before/context_after to disambiguate."
```

The LLM gets line numbers and surrounding code so it can refine on retry.

### Incremental Rollout

| Step | What | Validation |
|------|------|-----------|
| v1 | EditFileTool with `old_text`/`new_text` only, rich error messages | **DONE** — EditFileTool already returns near-matches with line numbers on failure. |
| v2 | Add `context_before`/`context_after` params | Does the LLM use them? Does it extract correct context from its memory of the file? |
| v3 | Prompt tuning — add instructions for when to use context params | Does usage rate improve? Does disambiguation accuracy improve? |
| v4 | Auto-suggest context in error messages | When 3 matches found, include `"suggestion": {"context_before": "def foo():\n    ", "context_after": "\n    x += 1"}` |

### Metrics to Track

- **Retry rate:** How often does the first edit attempt fail?
- **Disambiguation usage rate:** How often does the LLM use context_before/context_after?
- **Disambiguation accuracy:** When used, does it select the right match?
- **Self-correction rate:** After an error message, does the retry succeed?

### Prompt Instructions (iterate on these)

Start minimal and refine:

```
v1 (no context params):
"edit_file replaces exact text. If old_text appears multiple times, set expected_count. Read the file first."

v2 (with context params):
"If old_text matches multiple locations, use context_before and/or context_after to specify which one.
These are the surrounding text — enough to uniquely identify the edit location."

v3 (refined after testing):
"For common patterns (return None, pass, self.x = x), ALWAYS include context_before with the
function/method signature or the preceding line. This prevents ambiguous matches."
```

---

## Part 2: Turn Pinning for Context Compaction

### The Problem

Over a long-horizon plan (10+ phases), the prompt budgeter's `_truncate_conversation()` drops oldest turns first. This loses:
- The original user instruction (turn 1)
- Early constraints ("don't modify the API surface")
- Foundational decisions ("we're using the factory pattern for this")

The LLM starts contradicting earlier decisions because it literally can't see them anymore.

### The Mechanism

Add `pin_turns` to the LLM's JSON output schema:

```json
{
    "action": {"tool_name": "edit_file", "params": {...}},
    "confidence": 0.9,
    "reasoning": "...",
    "pin_turns": [1, 3]
}
```

The LLM parser (llm_worker.py:920-960) is already permissive — `.get()` style, no strict schema. Adding `pin_turns` won't break parsing.

Compaction algorithm keeps: first turn (always pinned) + explicitly pinned turns + last N turns.

### Relative Pins with Compaction Remapping

Pins use relative indices (what the LLM sees in its prompt, not absolute IDs). When compaction removes turns, pins remap:

```python
def _compact_conversation(content, max_turns, pinned=None):
    turns = _split_into_turns(content)
    if len(turns) <= max_turns:
        return content, pinned or set()

    pinned = pinned or set()
    keep_indices = sorted({0} | (pinned & set(range(len(turns)))))
    remaining = max_turns - len(keep_indices)
    for i in range(len(turns) - 1, -1, -1):
        if i not in keep_indices and remaining > 0:
            keep_indices.append(i)
            remaining -= 1
    keep_indices = sorted(set(keep_indices))

    # Remap pins to new indices
    new_pins = {new_idx for new_idx, old_idx in enumerate(keep_indices) if old_idx in pinned}
    return "\n".join(turns[i] for i in keep_indices), new_pins
```

Session stores the current pin set. Each compaction remaps it. The LLM always sees contiguous turn numbers.

### Incremental Rollout

| Step | What | Validation |
|------|------|-----------|
| v1 | Always pin turn 1 (original goal). No LLM pinning yet. | **DONE** — `_compact_conversation()` already pins first turn. |
| v2 | Add `pin_turns` to LLMProposal. Basic prompt instruction. | Does the LLM pin at all? How often? What does it pin? |
| v3 | Refine prompt instruction based on v2 observations. | Does pinning accuracy improve? Is over-pinning controlled? |
| v4 | Add pin rate monitoring (warn if >50% turns pinned). | Catches over-pinning before it degrades compaction. |

### Metrics to Track

- **Pin rate:** Percentage of turns pinned per session
- **Pin survival rate:** How many pinned turns survive to the end of a session
- **Contradiction rate:** Does the LLM contradict earlier decisions less with pinning?
- **Over-pinning rate:** Sessions where >50% of turns are pinned (instruction needs tuning)

### Prompt Instructions (iterate on these)

```
v2 (initial):
"Optional: pin_turns: [1, 3] — turn numbers to keep during context compaction.
Pin turns that contain critical constraints or decisions that later phases depend on."

v3 (after observing over-pinning):
"Optional: pin_turns: [1, 3] — pin ONLY turns with information that would be lost
and cannot be re-derived. User instructions, architectural decisions, constraints.
Do NOT pin routine exchanges, tool results, or status updates."
```

---

## Part 3: Dropped Context Notice

> **Status:** DONE — Implemented as PERF-4 in the claw-code upgrade. The dropped context notice is already prepended to prompts in prompt_builder.py.

### The Problem

When the prompt budgeter drops sections due to token budget, the LLM doesn't know what's missing. It may make decisions based on incomplete context without awareness.

### The Mechanism

`PromptBudgeter.build()` already returns `(prompt_text, dropped: list[str])`. The caller (prompt_builder.py:835) only logs this. Surface it to the LLM:

```python
prompt_text, dropped = budgeter.build()
if dropped:
    notice = f"[Context note: these sections were omitted to fit token budget: {', '.join(dropped)}]"
    prompt_text = notice + "\n\n" + prompt_text
```

### Incremental Rollout

| Step | What | Validation |
|------|------|-----------|
| v1 | Add notice listing dropped section names | Does the LLM acknowledge missing context? Does it ask for re-reads? |
| v2 | Add detail level ("conversation_history kept last 3 of 15 turns") | Does more detail help or just waste tokens? |

### Risks

- The notice itself consumes tokens from the budget. Keep it short.
- The LLM might over-react ("I don't have enough context to proceed") instead of working with what it has. Prompt tuning needed.

---

## Implementation Dependencies

| Item | Depends On | From Plan |
|------|-----------|-----------|
| Edit disambiguation (Part 1) | EditFileTool (#11a) | claw-code upgrade (implemented) |
| Turn pinning (Part 2) | Context compaction (#10), LLMProposal field, Session persistence (#8) | claw-code upgrade (implemented) |
| Dropped context notice (Part 3) | None — can be done independently | claw-code upgrade (implemented) PERF-4 |

### Sequencing

1. **Part 3 (dropped context notice)** — **DONE**
2. **Part 1 v1 (EditFileTool with rich errors)** — **DONE**
3. **Part 1 v2-v4 (context_before/after)** — NEXT: add params after observing v1 retry rates
4. **Part 2 v1 (always pin turn 1)** — **DONE**
5. **Part 2 v2-v4 (LLM-driven pinning)** — add after observing v1 contradiction rates

Each step is independently deployable and reversible. Prompt instructions are treated as tunable parameters, not fixed code.
