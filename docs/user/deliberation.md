# Deliberation Guide

Maxim's PFC deliberation system gives the agent an inner monologue — a cycle of thinking, consulting bio-system memories and predictions, and refining reasoning before acting. Instead of reacting immediately to each percept, the agent pauses to reflect when the situation warrants it.

## Quick Start

Deliberation is automatic when bio-enrichment is active. No flags needed:

```bash
# Interactive sim — watch the thinking panel build
maxim --sim "test stealth mission" --interactive

# Non-interactive — deliberation still runs, visible in JSONL
MAXIM_LOG_FILE=/tmp/maxim.jsonl maxim --sim "test combat" --interactive false --sim-max-turns 10
```

## How It Works

### The Deliberation Cycle

When a percept arrives, the ThoughtGate decides whether the agent should deliberate. If the gate fires:

1. **Cycle 1:** The percept text is enriched through the bio-systems (hippocampus, NAc, EC, cerebellum, SCN). The enrichment — recalled memories, causal predictions, active concepts — is injected into the LLM prompt alongside the percept.

2. **The LLM responds** with a JSON proposal containing `ready_to_act` (true/false), an `action`, `confidence`, and `reasoning`.

3. **If `ready_to_act: false`:** The LLM's reasoning is fed back through bio-enrichment for another round. The enrichment result may surface different memories or predictions based on the new reasoning text. This is cycles 2+.

4. **If `ready_to_act: true`:** The agent acts on the proposal.

5. **Convergence detection:** If two consecutive cycles produce nearly identical reasoning (Jaccard similarity >= 0.8), deliberation stops early — the agent has converged on a conclusion.

6. **Max cycles:** Hard cap (3 for sim, 2 for interactive) prevents infinite loops.

### What the Agent Sees

The LLM receives a **deliberation transcript** — an accumulating record of its prior reasoning paired with the bio-system responses it triggered:

```
=== Your inner deliberation (private — not speech) ===

[Cycle 1]
You thought: The guard is sleeping and I notice keys on his belt...
Your experience responded:
- Memory: Last time you reached for something near a sleeping NPC,
  the noise check succeeded (salience=0.71)
- Prediction: stealth actions near sleeping entities have 72% success rate

[Cycle 2]
You thought: Given the memory of success with stealth near sleeping
NPCs, and the prediction of 72% success, I'll reach for the keys slowly...
Your experience responded:
- Memory: Slow movements reduce noise check difficulty by one tier
- Prediction: combined stealth + slow movement → 89% success estimate
```

Each cycle genuinely builds on the previous — the LLM in cycle 3 sees cycles 1 and 2's reasoning and the bio-system responses they triggered.

### Inner Monologue

The agent's reasoning during deliberation is framed as **private inner thought**, not speech. The prompt instructs the LLM to think in first person:

- **Correct:** "I notice the dragon is attacking from the east. I recall that fire dragons are weak to water."
- **Wrong:** "What do you plan to do?" (narrator voice) or "Would you like me to try stealth?" (seeking guidance)

The agent explores autonomously — using tools, memories, and predictions — before asking the user for input.

## Computed Salience

Not all thoughts are equally important. Each THOUGHT entry in working memory receives a computed salience score based on the bio-system response it triggered:

| Component | Weight | Signal |
|-----------|--------|--------|
| Section count | 0.3 | More bio-systems activated = more cross-system relevance |
| Recall depth | 0.3 | More memories recalled = stronger associative resonance |
| Novelty | 0.4 | Novel thoughts (low Jaccard with previous) are more informative |

Range: 0.0-1.0. Typical cycle 1: ~0.5-0.7 (novel, moderate enrichment). Converging later cycles: ~0.2-0.4 (high Jaccard, less novelty).

The `top_by_salience` query on WorkingMemorySet returns the highest-salience thoughts first, so the most important reasoning surfaces in the prompt when token budget is tight.

## The Thinking Panel

In interactive mode, the thinking panel (left side of the display) shows a **continuous stream** of the agent's inner thoughts:

```
╭─────────────────────────── Thinking ───────────────────────────╮
│ Cycle 1/3  8.2s  enriched: hippocampus, nac                    │
│ I notice the dragon is approaching from the east. My memory    │
│ recalls that fire dragons are vulnerable to water attacks.     │
│── hippocampus, nac s=0.69                                      │
│ Given my earlier success with the water spell, I should try    │
│ to find the river to the north before engaging directly.       │
│── hippocampus s=0.54                                           │
│ The villagers mentioned a well near the town square. I recall  │
│ wells can be used as water sources too — closer than the river.│
╰────────────────────────────────────────────────────────────────╯
```

- **Thoughts accumulate** across turns — the full session thought stream, scrollable with arrow keys
- **Salience shown** per entry (`s=0.69`) so you can see which thoughts the bio-systems responded to most strongly
- **Enrichment tags** show which bio-systems contributed (hippocampus, nac, ec, cerebellum, scn)
- **Left arrow** enters the thinking panel for scrolling; **right arrow** exits and auto-follows new thoughts

## ThoughtGate

The ThoughtGate decides when the agent should deliberate. It composes four checks in a short-circuit cascade:

1. **Refractory:** Don't re-fire within N ticks of the last deliberation (default: 2 ticks)
2. **Energy:** Don't think below 15% of the token budget
3. **Salience score:** Run the SalienceScorer over the working memory head
4. **Adaptive threshold:** Score vs. AdaptiveThresholdController (adjusts based on whether past deliberations were useful)

If all checks pass, deliberation proceeds. If any fails, the agent acts on the percept without deliberating.

## Token Budget

The deliberation transcript uses a **proportional** token budget to avoid starving other prompt sections:

```
budget = min(2000, int((n_ctx - response_reserve - overhead) * 0.3))
```

| Model context | Budget |
|---------------|--------|
| 4K (local 14B) | ~891 tokens |
| 8K (Claude) | ~2000 tokens |
| 16K+ | 2000 tokens (cap) |

When the transcript exceeds the budget, the oldest cycle entries are dropped first — most recent reasoning is most valuable.

When a transcript is present, the separate bio-enrichment section is suppressed to avoid rendering the current cycle's enrichment twice.

## Debugging

### JSONL Logging

```bash
MAXIM_LOG_FILE=/tmp/maxim.jsonl maxim --sim "test recall" --interactive false --sim-max-turns 5
```

Look for these entries in the log panel or JSONL:
- `[THOUGHT] deliberation approved` — ThoughtGate fired
- `[THOUGHT] cycle 1: salience=0.69, sections=2, memories=3` — computed salience
- `[DELIBERATION] cycle 2: 3 enrichment section(s)` — multi-cycle enrichment
- `[DELIBERATION] deliberation converged (Jaccard)` — convergence detection

### SEM Diagnostics

When running with embodiment (default in sim mode), `[SEM_TRACE]` entries appear in the log showing:
- Registered entities and their affordances
- `discover_tools` query results (what matched and what didn't)
- Imagination trigger extraction (what entity phrases were found in percepts)
- Entity instantiation from seed components

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No thoughts appearing | ThoughtGate rejecting (refractory, energy, or threshold) | Check `[THOUGHT]` log for rejection reason |
| All salience=0.40 | Cold start — hippocampus has no memories to recall | Run more turns; enrichment improves with memory |
| Agent asks user instead of acting | Interactive mode prompt encouraging guidance-seeking | The "ACT FIRST, ASK SECOND" instruction should prevent this; check prompt assembly |
| Transcript too long for 4K model | Proportional budget should handle this | Check `n_ctx` is set correctly |
| discover_tools returns 0 matches | Entity not instantiated or no matching affordances | Check `[SEM_TRACE]` for entity registration |
| Agent uses dragon's tools as its own | SEM entity ownership gap (tracked in sem_entity_ownership.md) | Known issue — entity tools currently shared |
| Repetitive thoughts in panel | Novelty gate threshold may need tuning (default 0.40 Jaccard) | Thoughts with >= 60% word overlap are suppressed |

## Entity Discovery

The agent has three tools for understanding its environment:

- **`sense_presence`** — Scan surroundings for interactive entities. Shows name, type, modulators, affordances, and sensor state. Also triggers imagination for entity phrases in the context query.
- **`discover_tools(query)`** — Find specific actions matching an intent. Activates matched tools for the next turn.
- **`sense(entity_name)`** — Read detailed sensor state of a specific entity.

The recommended flow: `sense_presence` (what exists?) → `discover_tools` (what can I do?) → use discovered tool.

## Architecture

```
ThoughtGate.should_think()
  │ (refractory → energy → salience → adaptive threshold)
  ▼
BioEnrichmentPipeline.enrich(percept_text)
  │ (hippocampus recall, NAc predictions, EC concepts, cerebellum motor programs)
  ▼
_compute_thought_salience(n_sections, n_memories, jaccard)
  │
  ▼
WMS.add(THOUGHT, salience=computed, content={enrichment})
  │
  ▼
LLM prompt ← deliberation transcript + bio-enrichment sections
  │
  ▼
LLM response: {ready_to_act, action, reasoning, confidence}
  │
  ├─ ready_to_act=true → execute action
  └─ ready_to_act=false → feed reasoning back → cycle 2+
```

### Key Files

| File | Role |
|------|------|
| `runtime/agent_loop.py` | `_run_deliberation_cycles`, `_compute_thought_salience`, `_jaccard_similarity` |
| `runtime/thought_gate.py` | ThoughtGate composite gate |
| `integration/bio_enrichment.py` | BioEnrichmentPipeline |
| `agents/prompt_builder.py` | `_add_deliberation_transcript_section`, bio-enrichment suppression |
| `agents/exec_prompts.py` | PFC_PREAMBLE (inner monologue framing) |
| `agents/working_memory.py` | `WorkingMemorySet.top_by_salience` |
| `simulation/sim_logger.py` | `sim_deliberation_update`, `sim_deliberation_end` |
| `interactive/display.py` | `_ThinkingState`, `set_thinking`, `_build_thinking_panel` |
