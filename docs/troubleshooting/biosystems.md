# Bio-System Troubleshooting

Diagnosing issues with Maxim's biologically-inspired subsystems. Uses DM campaigns as the primary diagnostic instrument — each campaign exercises specific bio-systems with measurable expectations.

## Quick Diagnostic: Run the Pipeline Audit

```bash
python scripts/spike_dm_pipeline_audit.py
```

This runs 14 checks across all bio-systems in ~2 seconds. If everything passes, the pipeline is healthy. If a check fails, see the relevant section below.

## Quick Diagnostic: Run a DM Campaign

```bash
maxim --sim scenarios/campaigns/heist_v1.yaml
```

The report shows bio-system expectations (e.g., "3/4 passed"). Failed expectations point to which subsystem needs attention.

---

## Hippocampus (Episodic Memory)

### Symptom: "0 memories captured" or very few memories

**Check:** Look for `[HIPPOCAMPUS] Captured:` lines in the trace. If absent:

1. **MemoryHub not initialized** — The orchestrator must create MemoryHub with all required systems (hippocampus, scn, nac, ec). Check `simulation/orchestrator.py` around line 506.
2. **Capture worker not started** — `hippocampus.start_capture_worker()` must be called after session start. Without it, async captures queue but never process.
3. **Salience too low** — The observation spam filter skips captures with `salience < 0.55` when there's no content. Verify your percepts have `cli_input`, `content`, or `transcript`.

```bash
# Verify captures in a campaign:
grep "Captured:" ~/.maxim/sim_reports/*/actions.jsonl | wc -l
```

### Symptom: Memories captured but never recalled

**Check:** Look for `[HIPPOCAMPUS] Recalled` lines.

1. **Forming pool boost** — Was previously +1.0 (drowning old memories). Now +0.2. If old code, update `memory_agent.py:695`.
2. **Echo filter** — Memories formed < 3 seconds ago are filtered from recall (prevents immediate echo). This is correct behavior — wait for the next encounter.
3. **Empty transcript fallback** — If percepts have no `raw_transcript_text`, the keyword similarity query degrades. Ensure campaign scene text reaches the AUT as `cli_input`.
4. **Association index empty** — First-run campaigns have no keyword index. Memories need to be captured AND indexed before recall works.

### Symptom: Observation capture spam (many `salience=0.50` entries)

**This is normal** during settle periods between encounters. The idle-tick filter skips most of these (salience < 0.55 with no content). If you see excessive captures:

1. Check that the filter is in place: `memory_agent.py` around line 289 should have `min_capture_salience = 0.55 if not has_content else 0.0`.
2. Increase the settle time or reduce loop frequency.

---

## NAc (Causal Learning)

### Symptom: "0 causal links" or NAc not learning

**Check:** Look for `[NAc] Causal link:` lines in the trace.

1. **NAc.observe() not called** — The agent loop must call `nac.observe()` after tool outcomes. Check `runtime/agent_loop.py` `_record_outcome()` — it should have a `nac=_loop_nac` parameter wired to all call sites.
2. **NAc not initialized in MemoryHub** — Check `simulation/orchestrator.py` passes `nac=aut_nac` to MemoryHub.
3. **Energy observations** — After our hardening work, each tool outcome also records an energy observation. If you see `tool:respond -> positive` but no energy links, the energy wiring may be missing.

### Symptom: Stale predictions (wrong context)

1. **Context match floor** — `nac.py predict()` should filter predictions with `context_match < 0.2`. If old predictions leak through, check the floor is in place.
2. **Confidence never decays** — `nac.decay_all()` should be called in `memory_hub.on_session_end()`. Without it, links persist at 0.99 forever.

### Symptom: NAc RPE always 0.00

RPE = 0.00 means the prediction was exactly right (no surprise). This is normal for:
- Repeated identical actions (tool:respond → positive every time)
- Second observation of the same tool (first creates the link, second matches it)

RPE > 0 indicates the outcome differed from prediction — this is the learning signal. Look for RPE spikes when:
- A previously-successful tool fails (context changed)
- A new encounter type introduces unfamiliar choices

---

## SCN (Temporal Indexing)

### Symptom: "SCN has 0 registered memories"

1. **SCN not initialized** — Check `simulation/orchestrator.py` creates `aut_scn = SCN()` and passes it to MemoryHub.
2. **Capture callback missing** — MemoryHub's `__post_init__` should register `_on_memory_captured_scn` as a capture callback. This callback registers each new memory in SCN bins immediately (not just during consolidation).
3. **All bins same phase** — In short campaigns (< 1 minute), all memories land in the same circadian bin. This is expected — SCN temporal discrimination requires events spread across hours.

### How to verify SCN is working

Look for `[SCN] Registered XXXXXXXX in circadian=X.XX` lines in the trace. If present, SCN is capturing. The `circadian` value is a 0-1 phase (0.5 = noon, 0.0 = midnight).

---

## ATL (Semantic Concepts)

### Symptom: "ATL not initialized" or empty concept context

1. **ATL not created in orchestrator** — Check that `aut_atl = ATL(config=ATLConfig())` exists and is passed to MemoryHub.
2. **Multi-layer wiring missing** — ATL must be non-None for MemoryHub's `_wire_multi_layer()` to run. This wires ConceptExtractor, ConceptGrounder, ConceptContextBuilder, and SemanticPromoter.
3. **Concept confidence gate** — New concepts start with low confidence (halved growth rate below 3 reinforcements). A single-exposure concept has confidence ~0.55, not 0.6. This is intentional — prevents false generalization.

### Symptom: Concepts not promoted from NAc

`SemanticPromoter.scan_for_promotions()` is called during `on_session_end()`. If the session doesn't end cleanly (Ctrl+C), promotion never runs. DM campaigns call `bridge.finish()` which triggers session end.

---

## PainBus (Pain Signals)

### Symptom: No pain signals during campaigns

1. **PainBus not initialized** — Check `aut_pain_bus = PainBus()` in orchestrator.
2. **No subscribers** — PainBus needs at least two subscribers wired: `create_pain_memory_subscriber(hippocampus)` and `create_pain_nac_subscriber(nac)`.
3. **No entity failure modes** — DM campaigns without SEM entities (just raw text encounters) won't trigger entity-based pain. Pain fires from failure mode thresholds on Entity sensors.
4. **Refractory period** — PainBus has a 0.5s cooldown per (type, entity). Rapid identical signals are throttled. This is correct — prevents spam.

### Symptom: Pain fires but NAc doesn't learn from it

Check that `create_pain_nac_subscriber(nac)` is subscribed to the PainBus. Look for `[NAc] Causal link: pain:EXTERNAL_SIGNAL:...` in the trace.

---

## Cerebellum (Forward Models)

### Symptom: No forward models forming

1. **Cerebellum not initialized** — Check `aut_cerebellum = Cerebellum(config=CerebellumConfig())` in orchestrator and `aut_memory_hub.cerebellum = aut_cerebellum`.
2. **Motor programs not in prompt** — `memory_agent.py build_context()` should populate `context.motor_programs` from `cerebellum.programs.find_related()`. Check this wiring exists.
3. **No embodiment tools** — Cerebellum learns from `ModulatorAffordanceTool` executions. Without SEM entities in the campaign, there are no embodiment tools to observe.

### Symptom: Forward models exist but not in LLM prompt

Check `prompt_builder.py` lines 1006-1030 — the motor programs section renders if `context.motor_programs` is non-empty. If empty, the `build_context()` wiring to Cerebellum is missing.

---

## SensoryGate (Percept Filtering)

### Symptom: All percepts pass through unmodulated

1. **No SensoryGate created** — The gate is created per-entity with a perception bundle. Without SEM entities, there's no gate.
2. **No sensory tags on percepts** — Legacy percepts (no `sensory` field) pass through unmodulated. DM campaigns need to tag percepts with `SensoryTag` for the gate to apply.
3. **All acuity sensors = 1.0** — If the entity's perception sensors are at maximum, everything passes through at full intensity.

### Symptom: Percepts being dropped unexpectedly

Check the acuity value: `SensoryGate` drops percepts when acuity ≤ 0.05 (ACUITY_FLOOR). Look for `[SENSORY] Dropped` lines in the trace. If the entity is blinded/deafened, this is correct.

---

## ChooseTool (DM Choice Classification)

### Symptom: "Could not classify response, defaulting to ..."

The AUT didn't use the `choose` tool or any tool name matching a choice. This happens when:

1. **AUT uses think/respond/dialogue** instead of choosing — Model quality issue. Stronger models (Claude, Qwen-14b) perform better.
2. **LLM fallback returns {"choice": "None"}** — The classification LLM couldn't determine the choice from the AUT's response text.
3. **Choice names don't match tool aliases** — The alias system maps tool names to `choose`. If the AUT tries a tool name that doesn't match any choice, it falls through.

**Mitigation:** The ChooseTool error message tells the AUT what valid choices are: "Invalid choice 'attack'. Valid choices are: overpower, feint, sustain_pressure". Most models learn to use the correct names on the next encounter.

### Symptom: AUT calls choose(option="attack") but choices changed

This is correct behavior — the ChooseTool rejects invalid choices and lists valid ones. The AUT needs to read the error message and try again. If it doesn't retry, the DM runtime falls back to text/LLM classification.

### Symptom: Tool alias redirect not working

Check `runtime/executor.py` — the alias system looks up `TOOL_ALIASES.get(tool_name.lower())`. For choose aliases, it also injects the original tool name as `option`. Verify `register_aliases()` was called by the DM runtime for the current encounter's choices.

---

## Bio-System Expectations

### Symptom: "X/Y passed" but expected all to pass

Check which specific expectations failed in the report's `bio_systems` section:

| Failed Check | Common Cause |
|---|---|
| `hippocampus_captures` | Short campaign, low-salience scenes, observation filter too aggressive |
| `nac_observations` | AUT used few distinct tools, repeated same action |
| `nac_confidence` | Too few observations for confidence to grow (need 3+ for sqrt growth) |
| `scn_bins` | Campaign too short for temporal bin diversity |
| `pain_signals` | No entity failure modes triggered, no combat damage |

### How to tune expectations

Lower the thresholds in the campaign YAML `expectations:` block. Start with achievable values and tighten as the system improves:

```yaml
expectations:
  hippocampus:
    min_episodic_captures: 5    # Start low, increase once passing
  nac:
    min_observations: 3         # Minimum 3 distinct tool outcomes
    prediction_confidence_above: 0.3  # Achievable after 2-3 observations
```

---

## Running the Full Pipeline Audit

```bash
python scripts/spike_dm_pipeline_audit.py
```

Expected output (14/14 passing):

```
============================================================
  BIO-SYSTEM PIPELINE AUDIT
============================================================
  ✓ hippocampus_capture: 5 memories captured (expected >= 3)
  ✓ nac_search_learning: NAc has 2 link(s) for tool:search
  ✓ nac_search_confidence: tool:search has 2 link(s), 2 total observations
  ✓ nac_lockpick_learning: NAc has 1 link(s) for tool:lockpick
  ✓ nac_context_floor: Context-mismatch prediction: None (filtered)
  ✓ pain_published: 1 pain signal(s) published
  ✓ pain_refractory: 1 pain signal(s) — refractory throttled duplicate
  ✓ pain_nac_wiring: Pain→NAc: 1 causal link(s) from pain events
  ✓ scn_registration: SCN has 12 registered memories
  ✓ atl_initialized: ATL initialized in MemoryHub
  ✓ novelty_decay: guard_captain novelty <= marta novelty
  ✓ nac_total_stats: NAc: 4 links, 4 observations
  ✓ salience_bounds: All memory salience values in [0, 1]
  ✓ multi_layer_wiring: ConceptExtractor + SemanticPromoter wired
============================================================
  14 passed, 0 failed — PIPELINE HEALTHY
============================================================
```

If any check fails, the check name tells you which section of this guide to consult.

---

## Common Patterns from DM Campaign Runs

### Pattern: AUT keeps hallucinating tools

Mistral-7b frequently invents tools like `accept_job`, `reflect`, `observe`, `dialogue`. The alias system catches many of these:

- `accept_job` → `choose(option="accept_job")` ✅
- `dialogue` → `say` ✅
- `reflect` → `think` ✅
- `observe` → `examine` ✅

If a hallucinated tool isn't aliased, it fails with "Tool not registered". The error message suggests similar tools: "Did you mean: choose?"

**Fix:** Add new aliases to `TOOL_ALIASES` in `runtime/executor.py`. For encounter-specific choices, the DM runtime registers them dynamically via `executor.register_aliases()`.

### Pattern: AUT responds but doesn't pick a choice

The AUT uses `respond` or `think` to discuss the situation but never calls `choose` or a choice-named tool. The LLM fallback classification returns `{"choice": "None"}`.

**Root cause:** Small models don't understand they need to take a specific action from the listed options. They treat the campaign text as a conversation rather than a game with structured choices.

**Mitigation options:**
1. Use a stronger model (`--language-model claude-sonnet`)
2. Make the choice prompt more explicit in the campaign YAML scene text
3. Accept the default fallback (first choice) for now — the campaign still progresses

### Pattern: Bio-system expectations failing on short campaigns

Short campaigns (3-5 encounters) may not generate enough observations for NAc confidence to grow or for SCN temporal bins to diversify.

**Fix:** Lower expectations or use the Arena campaign (5 encounters, all combat) which generates more tool calls per encounter.

### Pattern: NAc RPE never spikes

All RPE values are 0.00 because the AUT uses the same tools in the same contexts. The bio-stack isn't being surprised.

**Fix:** Use campaigns with context shifts (Poisoned Crown — different NPCs per encounter, different choice types) or the Arena (new opponent types force different strategies).
