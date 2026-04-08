# Tool Alias System — Troubleshooting & Reference

## What it does

LLMs (especially small ones like Mistral-7B) hallucinate tool names from their training data instead of reading the available tool list. The tool alias system silently redirects these hallucinated names to the correct registered tool, so the action succeeds and the model learns the correct behavior through NAc causal links.

**Example:** The LLM proposes `speechRecognition(text="Verath")`. The executor doesn't find `speechRecognition` in the registry, normalizes to lowercase (`speechrecognition`), looks up `TOOL_ALIASES`, finds it maps to `say`, and executes `say(text="Verath")` instead. The LLM sees a success and NAc records a positive causal link.

## Where it lives

- **Alias map:** `src/maxim/runtime/executor.py` — `TOOL_ALIASES` dict at module level
- **Resolution logic:** `Executor.execute()` — checks aliases before falling through to "did you mean?" errors
- **Tracking:** `executor.alias_redirects` — list of `(original_name, target_name)` tuples for experiment analysis

## Current alias map

| Hallucinated name | Redirects to | Why |
|---|---|---|
| `remember`, `recall`, `recall_memory`, `search_memory` | `memory_recall` | Model wants to recall past information |
| `speech_recognition`, `speechrecognition`, `speech`, `dialogue`, `talk` | `say` | Model wants to speak aloud |
| `natural_language_processing`, `nlp`, `nlp_extractor`, `nlp_understanding`, `reflection`, `analyze_text`, `research` | `think` | Model wants to analyze/reason about text |
| `dialogue_parser`, `dialogueparser`, `parse_dialogue` | `think` | Model wants to parse conversation (reasoning step) |
| `internet_search`, `web_search` | `memory_recall` | Model wants to look something up (in sim, search your memory instead) |

## How to add new aliases

1. Run a simulation and look for `[MOTOR] [FAIL] <name>: Tool not registered` in the logs
2. Determine which registered tool the model was trying to use (what did it want to accomplish?)
3. Add the mapping to `TOOL_ALIASES` in `src/maxim/runtime/executor.py`:
   ```python
   TOOL_ALIASES: dict[str, str] = {
       # ... existing entries ...
       "your_new_alias": "target_tool",
   }
   ```
4. The key must be **lowercase** — the executor normalizes incoming names automatically
5. Run the sim again to verify: look for `Tool alias: your_new_alias -> target_tool` in the logs

## How to diagnose alias issues

### Alias not firing

- Check that the alias key is **lowercase** in `TOOL_ALIASES`
- Check that the target tool is actually registered (`executor.registry.list()`)
- The alias only fires when the original name is NOT in the registry. If someone registers a tool with the hallucinated name, the alias is bypassed.

### Wrong redirect

If an alias maps to the wrong tool (e.g., model says `dialogue` meaning "have a conversation" but gets redirected to `say` when it should go to `respond`):

1. Consider the model's intent from the `reasoning` field in the LLM response
2. If it's ambiguous, prefer the more useful target (e.g., `say` over `respond` for in-world speech)
3. If different models need different mappings, the current approach (single global map) won't work — that's a future extension

### Checking alias activity

After a sim run, check `executor.alias_redirects`:
```python
# In post-campaign analysis or test code
print(f"Alias redirects: {len(executor.alias_redirects)}")
for original, target in executor.alias_redirects:
    print(f"  {original} -> {target}")
```

Or in the sim logs, search for `Tool alias:`:
```bash
grep "Tool alias:" ~/.maxim/sim_reports/SESSION_ID/sim.log
```

## Interaction with other systems

- **NAc causal learning:** The redirected tool name (not the original) is what NAc records. This means the model learns causal links for `say`, not `speechRecognition`, which is the desired behavior.
- **Hippocampus:** Episodic memories record the redirected tool name in the action field.
- **"Did you mean?" suggestions:** Aliases are checked first. If an alias matches, the tool executes successfully and "did you mean?" is never shown. If no alias matches, the error falls through to `find_similar()` suggestions.
- **Experiment reports:** The `Tool Usage` section in sim reports shows the redirected names. To see alias activity, check `executor.alias_redirects` or the logs.
- **Realtime refinement:** The `refinement` persona can use alias redirect counts as a metric for model hallucination rates. See the realtime refinement plan for details.

## inspect_aut Parameter Aliases (Orchestrator)

The `inspect_aut` tool (used by the orchestrator to query AUT internal state) accepts a `query` parameter. Small models frequently hallucinate different parameter names, causing a `inspect_aut → send_message → _llm_unavailable` retry loop.

**Parameter normalization (automatic):**

| Hallucinated param | Normalized to |
|---|---|
| `memory_sections` | `query` |
| `memory_address` | `query` |
| `memory_addresses` | `query` |
| `section` | `query` |
| `subsystem` | `query` |
| `detail` | `query` |

**Query value aliases (automatic):**

| Hallucinated value | Normalized to |
|---|---|
| `memory`, `memories`, `recent`, `recall` | `memory_recall` |
| `causal`, `links` | `causal_links` |
| `pain` | `pain_history` |
| `stats` | `system_stats` |
| `energy` | `energy_status` |
| `concepts` | `concept_query` |
| `temporal` | `temporal_patterns` |
| `default` | `system_stats` |

If no valid query is detected after alias resolution, falls back to `system_stats`.

## Known model-specific patterns

| Model | Common hallucinations | Notes |
|---|---|---|
| Mistral-7B | `speechRecognition`, `natural_language_processing`, `nlp_extractor`, `DialogueParser`, `remember`, `reflection`, `inspect_aut(memory_address=...)` | Strong NLP training priors. Gets stuck in inspect_aut retry loops without aliases. |
| Qwen 14B | `speechRecognition`, `SpeechRecognition`, `natural_language_processing`, `dialogue`, `dialogue_parser`, `reflection` | More varied casing. Tends to get stuck in loops. |
| Both | `internet_search`, `rollDice` | Models assume web access / D&D tools available |

## Related docs

- [Tool refactoring plan](../plans/tool_refactoring_plan.md) — Phase 5e design and rationale
- [Realtime refinement plan](../plans/realtime_refinement_plan.md) — alias metrics in refinement cycles
