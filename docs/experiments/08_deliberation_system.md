# Experiment 08: Deliberation System Stress Test (L1+L2)

**Date:** 2026-04-20
**Version:** 0.7.0 (post-L1+L2 ship)
**Status:** PASS — all 18 stress tests + 25 unit tests green

---

## What was tested

The L1+L2 deliberation system adds bio-enrichment and active deliberation control to the `think` tool. This experiment stress-tests the critical paths to validate safety invariants before shipping.

### L1 — Bio-enrichment wiring
- ThinkTool calls `BioEnrichmentPipeline.enrich(thought, bypass_gate=True)` on every think
- Enriched response (memories, predictions, affordances, valence) included in ToolOutput
- Novel percepts run through the pipeline and inject into `StructuredContext.bio_enrichment_context`
- Pipeline failures are gracefully degraded (tool still succeeds)

### L2 — Active deliberation
- Consecutive think counter + convergence detection (Jaccard keyword similarity >= 0.8)
- Hard cap at 3 hops (configurable via `max_deliberation_hops`)
- NAc-gated termination: declining enrichment valence biases LLM toward action
- `reset_deliberation()` called when any non-think tool fires

## Stress test results

| Test | Result | Notes |
|------|--------|-------|
| Hard cap fires at exact hop (1,2,3,5,10) | PASS | Parametrized across 5 cap values |
| Hard cap on 100 consecutive thinks | PASS | Every hop >= 3 has termination signal |
| Realistic sword deliberation sequence | PASS | Progressive refinement, no false convergence |
| Looping deliberation detected | PASS | Agent rephrasing same idea caught within 3 iterations |
| Progressive refinement not flagged | PASS | Genuinely different thoughts don't trigger convergence |
| NAc declining valence sequence | PASS | Valence [0.6, 0.3, 0.1, 0.0] correctly exposed to LLM |
| Reset clears all state | PASS | Counter + history fully zeroed |
| Reset prevents stale convergence | PASS | Same thought after reset: no false positive |
| Interleaved action/think cycles | PASS | 50 cycles, clean resets |
| Pipeline always-failing | PASS | RuntimeError on every call — deliberation still tracks |
| Pipeline intermittent failures | PASS | Alternating success/failure — hop counter correct |
| 100 thinks memory bounded | PASS | History stays at 5 entries, counter at 100 |
| 1000 calls < 1s | PASS | **2-3 μs/call** (no pipeline), well under budget |
| 50 reset cycles | PASS | No state leakage |

### Performance

```
1000 calls (no pipeline): 2.1ms total, 2μs/call
1000 calls (with convergence tracking): 2.8ms total, 3μs/call
Memory: keyword history bounded at 5 entries regardless of call count
```

The deliberation system adds ~1μs overhead per call (convergence check + keyword extraction). This is negligible against the ~26ms bio-enrichment budget and the ~500ms+ LLM call latency.

## Safety invariants validated

1. **Hard cap is absolute.** The termination signal fires at exactly `max_deliberation_hops` regardless of thought content, convergence state, or pipeline failures. There is no code path that skips the cap check.

2. **Convergence is a nudge, not a block.** The convergence signal ("Your thoughts are converging...") appears in the output but does NOT prevent further thinking. Only the hard cap contains the word "time to act." The LLM decides whether to heed the nudge.

3. **Reset isolation is complete.** After `reset_deliberation()`, both the counter and keyword history are zeroed. There is no cross-sequence contamination. The reset is wired into both tool dispatch paths in `agent_loop.py` (single-action and parallel-action).

4. **Pipeline failures never break deliberation.** All enrichment calls are wrapped in `try/except`. Counter tracking, convergence detection, and hard cap operate independently of enrichment. A completely broken pipeline produces a ThinkTool that still tracks deliberation correctly.

5. **Memory is bounded.** Keyword history is capped at 5 entries regardless of call volume. No unbounded growth.

## How to reproduce

```bash
# Run the stress tests (standalone, ~0.06s)
python -m pytest tests/experiments/test_deliberation_stress.py -v

# Run the full L1+L2 unit tests (within full suite collection)
python -m pytest tests/ -v -m "not slow" --ignore=tests/integration/test_memory_hub.py -k "bio_enrichment"

# Run the full test suite (confirms no regressions)
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
```

### Live verification (with LLM)

To verify the deliberation system works end-to-end with a real LLM:

```bash
# Run a sim with JSONL logging to capture think tool calls
MAXIM_LOG_FILE=/tmp/deliberation_test.jsonl \
  maxim --sim "test deliberative thinking about a locked gate" \
  --embodiment weapons/rusty_sword \
  --interactive false \
  --sim-max-turns 10

# Inspect think tool calls and their outputs
cat /tmp/deliberation_test.jsonl | python -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    if e.get('event') == 'tool_called' and e.get('data', {}).get('tool') == 'think':
        print(json.dumps(e['data'], indent=2))
    if e.get('event') == 'tool_result' and 'bio_response' in str(e.get('data', {})):
        print('--- ENRICHED RESULT ---')
        print(json.dumps(e['data'], indent=2))
"
```

**What to look for:**
- `bio_response` field present in think tool output (L1 enrichment working)
- `deliberation_hop` incrementing across consecutive thinks
- `deliberation_signal` appearing after convergence or cap
- Agent choosing an action tool after receiving the termination signal
- `deliberation_hop` resetting to 1 after a non-think action fires

## Architecture reference

```
LLM calls think("how do I get past the gate?")
    ↓
ThinkTool.execute()
    ├── _consecutive_thinks += 1
    ├── _extract_thought_keywords(thought)
    ├── pipeline.enrich(thought, bypass_gate=True)  [L1]
    │   └── returns: memories, predictions, affordances, valence
    ├── _check_convergence(keywords)  [L2]
    │   └── Jaccard(current, recent[-3:]) >= 0.8 → nudge
    ├── hard cap check  [L2]
    │   └── consecutive_thinks >= max_hops → "time to act"
    └── returns ToolResult with bio_response + deliberation_signal + hop
            ↓
ACTION_FOLLOWUP (followup_type="process")
    ↓
LLM sees enriched result, decides: think again or act?
    ↓
If act → any non-think tool → reset_deliberation()
```

## Files modified

| File | Change |
|------|--------|
| `src/maxim/tools/narrative.py` | L1: pipeline param + enrichment. L2: deliberation state, convergence, cap |
| `src/maxim/integration/bio_enrichment.py` | L0 (pre-existing): pipeline core |
| `src/maxim/runtime/agent_loop.py` | L1: percept enrichment hook. L2: deliberation reset on non-think tools |
| `src/maxim/simulation/orchestrator.py` | Pipeline construction + wiring to ThinkTool and agent loop |
| `src/maxim/agents/bus.py` | `bio_enrichment_context` field on StructuredContext |
| `src/maxim/agents/prompt_builder.py` | Render bio_enrichment_context as prompt section |
| `tests/unit/test_bio_enrichment_wiring.py` | 25 unit tests (L1 enrichment + L2 deliberation) |
| `tests/experiments/test_deliberation_stress.py` | 18 stress tests (this experiment) |
