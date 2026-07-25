# Exp 44 — substrate counterfactual harness (trajectory-matched)

**Question:** does the learned orient / substrate change the LLM-primary agent's
*decision*, holding the world-state fixed? Isolate the substrate's contribution
to the LLM's **input** (prompt), since it never touches the LLM's output.

**Non-negotiable framing (the correction):** you cannot hold LLM *outputs* fixed —
they are the dependent variable. Hold the **world-state at each decision + the
sampling seed** fixed; let only the substrate annotation in the prompt differ;
measure whether the **action** changes.

---

## Design: offline reconstruction (least new infra, faithful, trajectory-matched)

No replay mode exists, so gold-standard teacher-forcing = new loop infra. This
design gets trajectory-matching *for free* by reconstructing the ablated prompt
**offline from the logged state** of a single status-quo run — both arms see the
identical world-state at every decision because they replay the same log.

### Pass 1 — live capture (status-quo, flags ON), ZERO core edits

A harness monkeypatch wraps `LLMWorker._prompt_builder.build_prompt`
([llm_worker.py:1190](../../src/maxim/agents/llm_worker.py#L1190)). At each LLM
submission it logs one JSONL row:

```
{ decision_id, step, world_state_digest,      # percepts / scene / az / drive states
  prompt_full,                                 # arm A — the prompt that actually ran
  context_blob }                               # serialized StructuredContext + the
                                               # request fields needed to re-render
```

`prompt_full` is exactly what the live run used (arm A). `context_blob` is what
lets the offline pass re-render the ablated variant. The executed action is
already in `actions.jsonl` (keyed by step) — join on it later.

### The ablation function — `ablate_context(ctx, request)` — WITH a validity guard

Null **every** substrate carrier the sanctioned env flags null. Enumerated set
(verify each against the producer before trusting):

| Flag it mirrors | Carrier to null |
|---|---|
| `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION` | `ctx.cluster_bias_annotations = None` (+ the goal-suggest reuse at agent_loop ~3416-3441) |
| `MAXIM_ENABLE_BODY_STATE_PROMPT` (→ off) | `ctx.body_state = None` |
| `MAXIM_DISABLE_COACH_BODY_LAYERS` | acting-coach Layers 2+4 off (`acting_coach_config_from_env` with body layers disabled, or drop `request.acting_coach`) |
| `MAXIM_DISABLE_VARIANCE_ANNOTATION` | **RESOLVED — out of scope.** The Wire-1 `_risk_profile` bakes into `request.tool_descriptions` (not a clean context field), and it is a risk-sensitivity mechanism, NOT orient. Held CONSTANT across both arms so the ablation isolates the *learned orient* (cluster-bias + body_state + coach body layers) alone. |

**Validity guard (this is what makes field-nulling safe instead of a band-aid):**
after `build_prompt(ablated)`, assert the rendered prompt contains **none** of the
substrate section markers:

```python
_SUBSTRATE_MARKERS = [
    CLUSTER_BIAS_SECTION_HEADER,   # from prompts/cluster_bias_annotation
    BODY_STATE_SECTION_HEADER,     # from the body_state formatter
    COACH_BODY_LAYER_MARKER,       # Acting Coach Layer 2/4 text
    VARIANCE_BAND_MARKER,          # Wire-1 felt-sensation phrase
]
def assert_fully_ablated(prompt_ablated: str) -> None:
    hit = [m for m in _SUBSTRATE_MARKERS if m in prompt_ablated]
    assert not hit, f"ablation leaked substrate markers: {hit}"
```

If a marker survives, the run FAILS LOUD — no silent mis-ablation. Ship this as a
unit test on a fixture context too, so the carrier list can't rot.

### Pass 2 — offline re-query (both arms, temp 0)

For each logged decision:
1. `prompt_ablated = build_prompt(ablate_context(ctx, request))`; `assert_fully_ablated`.
2. Query the LLM at **temp 0** on `prompt_full` AND `prompt_ablated`
   (route through the same backend the run used; parse the tool call).
3. Record `{decision_id, action_full, action_ablated, flipped, world_state_digest}`.

Re-querying **both** offline at temp 0 (rather than reusing arm A's live action)
puts both arms on identical decoding, so a flip is attributable to the prompt
delta alone, not sampling.

### Prior-strength slice (the interpretation guard — Goldilocks)

Substrate should matter only where the LLM prior is weak (Exp 37/38/40:
prior-agreement is the gating variable). So don't report a bare flip rate — slice
it. Per decision, sample the **ablated** prompt (= the LLM's own prior, substrate
removed) N≈8 times at temp≈0.7; the action-distribution entropy is the
prior-strength proxy.

- **Headline metric:** flip rate *among decisions where prior entropy is high*
  (ambiguous / counter-prior states).
- Report the low-entropy flip rate too — it should be ≈0 (substrate correctly
  stays quiet where the prior is confident). If it's high, the substrate is
  fighting a confident prior = a red flag, not a win.

### Directional metric

A flip is only good if it moves toward correct behavior. For the orient/safe-vs-
harm arcs, label each `action_full` vs `action_ablated` by whether it's the
correct orient direction / the safe warmth source. Report flips **toward** vs
**away** — a substrate that flips decisions *away* from correct is worse than one
that does nothing.

---

## Deliverables

- `scripts/exp44/capture_paired_prompts.py` — monkeypatch + JSONL logger (pass 1).
- `scripts/exp44/rerun_ablated_offline.py` — `ablate_context` + `assert_fully_ablated`
  + temp-0 re-query + prior-entropy slice + directional rollup (pass 2).
- `tests/unit/test_exp44_ablation.py` — pins the carrier list via `assert_fully_ablated`
  on a fixture context (guards against carrier rot).

## The one open decision (validity vs cost)

**Offline reconstruction (above)** vs **teacher-forced second pass** (new loop
infra: capture arm-B's live LLM proposal but execute arm-A's action to pin the
trajectory). Offline reconstruction is cheaper, needs no core edits, and is
trajectory-matched *by construction* — its only risk is the carrier enumeration,
which `assert_fully_ablated` neutralizes. Recommend offline reconstruction unless
a reviewer wants the live counterfactual. The free-running paired-seed arms
(flags on vs off, N seeds) remain the separate end-to-end confirmation.

## Wiring reality (2026-07-25) — direct `maxim --sim`, not the benchmark

`benchmark_exp42_preference.py` **subprocesses `maxim --sim`** and forces
`MAXIM_AUTO_SPAWN_LLM_SERVER=0` (it was built for substrate-primary, LLM-free
AUT). Both break the counterfactual: (1) a class-level patch in the parent never
reaches the AUT in the child; (2) an llm-primary AUT has no backend → "No
eligible LLM providers" → no report. **Vehicle = a direct `maxim --sim` run**
(auto-spawn ON, single process). Capture is installed by a DORMANT gated hook in
`orchestrator.py` (after `aut_llm_worker.start()`) that fires only when
`MAXIM_EXP44_CAPTURE_LOG` is set and installs `install_capture` on the AUT worker
in-process. The narrator uses a separate worker and is not captured.
