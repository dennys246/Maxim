# Experiment 09: PFC Deliberation Cycle (Stage 1+2)

**Date:** 2026-04-22
**Version:** 0.8.x (post-PFC Stage 1+2 ship)
**Status:** PROTOCOL DEFINED — awaiting live LLM run
**Depends on:** [pfc_deliberation_cycle.md](../plans/pfc_deliberation_cycle.md)

---

## What was shipped

Stages 1 and 2 of the PFC deliberation cycle:

### Stage 1 — Non-optional wiring
- `ThoughtGate` and `BioEnrichmentPipeline` now constructed inside `build_bio_stack()` (not sim-only)
- Both fields added to `BioStack` frozen dataclass
- Orchestrator reads from BioStack instead of constructing them manually; dead `wire_thought_gate()` / `wire_bio_enrichment()` calls removed
- CLI non-sim path passes both to `run_agentic_loop`
- `run_agentic_loop` accepts new `thought_gate` parameter

### Stage 2 — Enrichment integration
- `ready_to_act: bool = True` added to `LLMProposal` (backward compatible default)
- LLM response parsing extracts `ready_to_act` from JSON
- JSON schema in prompt updated to include `ready_to_act` field with concrete trigger checklist
- `PFC_PREAMBLE` added to `exec_prompts.py` — injected at `SectionPriority.IMPORTANT` when bio-enrichment is active
- `_add_working_memory_section()` renders recent THOUGHT entries in prompt (truncatable, IMPORTANT priority)
- `working_memory_thoughts: list[str] | None` added to `StructuredContext`
- `reset_refractory(tick)` added to `ThoughtGate` — refractory counts from enrichment completion, not start
- Inline bio-enrichment block (section 1.2) replaced with gated enrichment: ThoughtGate check → enriched percept → THOUGHT WMS entry → sim logging
- Draft `_run_deliberation_cycle()` function defined for future multi-cycle LLM calls (cycles 2+)

### What is NOT yet shipped
- Multi-cycle LLM calls (cycles 2+) — the `_run_deliberation_cycle` function is defined but not wired into the loop. The agentic loop's async submit/poll model requires restructuring the LLM submission point to support blocking multi-call sequences. Currently, the gate-fired minimum (one enriched round) is the active path.

## Test results

5680 tests passed, 0 failed, 2 skipped (pre-existing), 25 deselected (slow markers).

High-risk test files verified first:
- `test_bio_enrichment_wiring.py` — 27/27 pass
- `test_thought_gate.py` — 12/12 pass
- `test_bio_enrichment.py` — 17/17 pass
- `test_prompt_builder.py` — 57/57 pass
- `test_llm_types.py` — 37/37 pass

## Pre-implementation review findings

Two-lens parallel review (concurrency/state + prompt/LLM-behavior) caught 3 CRITICAL + 4 HIGH issues. All resolved before ship:

| Finding | Severity | Resolution |
|---------|----------|------------|
| `budgeter.add()` called with unsupported `max_tokens` param | CRITICAL | Removed `max_tokens` (budgeter handles truncation via `truncatable=True`) |
| PFC preamble gate `is not None` on `""` default | HIGH | Changed to truthy check `if request.context.bio_enrichment_context:` |
| `_run_deliberation_cycle` calls `submit_context(context)` with wrong signature | HIGH | Marked as DRAFT — function not wired into loop yet |
| LLMWorker is async submit/poll, cycle assumed sync | CRITICAL (plan) | Added `_wait_for_proposal` blocking poll; deferred to multi-cycle wiring stage |
| JSON examples used wrong schema (`tool_name` vs `action.tool_name`) | CRITICAL (plan) | Fixed in plan doc |
| THOUGHT entries not rendered in prompt | CRITICAL (plan) | Added `_add_working_memory_section()` |
| ThoughtGate refractory penalizes post-deliberation percepts | HIGH (plan) | Added `reset_refractory(tick)` |

## Reproduction protocol

### Sim-mode verification (requires local LLM)

```bash
# Verify THOUGHT + DELIBERATION log lines appear on novel percepts
PYTHONPATH=src python -m maxim \
  --sim "You are trapped in a dungeon with a sleeping guard. Escape quietly." \
  --interactive false \
  --sim-max-turns 6 \
  --display bio \
  --llm mistral-7b

# Expected output at --display bio:
#   [THOUGHT     ] [AUT] pre-deliberation: gate passed (score=X.XX >= X.XX), N enrichment section(s)
#   [DELIBERATION] [AUT] deliberation converged after 1 cycles (score=X.XX)
```

### CLI non-sim verification (requires local or cloud LLM)

```bash
# Verify enrichment fires in non-sim CLI mode
PYTHONPATH=src MAXIM_LOG_FILE=/tmp/maxim.jsonl python -m maxim --llm mistral-7b
# Send a message, then:
grep -c THOUGHT /tmp/maxim.jsonl  # Should be > 0
```

### JSONL trace analysis

```bash
# Capture full trace for analysis
PYTHONPATH=src MAXIM_LOG_FILE=/tmp/pfc_trace.jsonl python -m maxim \
  --sim "explore a haunted castle" \
  --interactive false \
  --sim-max-turns 4

# Analyze:
python -c "
import json
with open('/tmp/pfc_trace.jsonl') as f:
    events = [json.loads(l) for l in f if l.strip()]
thoughts = [e for e in events if e.get('subsystem') == 'THOUGHT']
delibs = [e for e in events if e.get('subsystem') == 'DELIBERATION']
print(f'THOUGHT events: {len(thoughts)}')
print(f'DELIBERATION events: {len(delibs)}')
for t in thoughts[:5]:
    print(f'  {t.get(\"message\", \"\")[:100]}')
"
```

## Hypotheses for live validation

| # | Hypothesis | Pass criteria |
|---|-----------|---------------|
| H1 | ThoughtGate fires on first novel percept in sim | At least 1 `[THOUGHT] pre-deliberation: gate passed` log line |
| H2 | Enrichment produces bio-system sections | `enrichment section(s)` count > 0 on gate-passed lines |
| H3 | THOUGHT entries accumulate in working memory | WMS THOUGHT count increases across turns |
| H4 | PFC preamble appears in prompt when enrichment is active | `ready_to_act` appears in LLM request trace |
| H5 | Refractory prevents immediate re-firing | No back-to-back gate passes on consecutive ticks |
| H6 | CLI non-sim path constructs ThoughtGate + BioEnrichment | THOUGHT subsystem events in JSONL log |
| H7 | `ready_to_act` defaults to `true` for local models | LLMProposal.ready_to_act == True on all responses (local 14B models don't produce the field) |

## Architecture notes

The PFC deliberation cycle lives in `runtime/agent_loop.py` as the enrichment block at section 1.2. It replaces the previous inline bio-enrichment that ran without ThoughtGate gating.

Key invariants:
- **Gate-fired minimum:** When ThoughtGate fires, cycle 1 always enriches (`bypass_gate=True`)
- **Refractory reset:** `thought_gate.reset_refractory(tick)` after enrichment completes
- **WMS accumulation:** THOUGHT entries with `source=pfc_deliberation` and `cycle=1`
- **Prompt injection:** `bio_enrichment_context` on StructuredContext + `working_memory_thoughts` for prior cycles
- **Backward compatible:** `ready_to_act` defaults to `True` — old models/prompts work as one-shot
