# Persona Convergence Crucible

**Status:** living doc, ongoing practice
**Begins:** post-1.0 (after [persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md) and [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) ship)
**Companion living docs:** [behavioral_convergence_practice.md](behavioral_convergence_practice.md) ("does the agent get better across sessions?"), [memory_consolidation_practice.md](memory_consolidation_practice.md) ("does sleep replay actually consolidate?")

## What this is

A long-running, deliberately-shaped attempt to develop persona-like behavior in a Maxim agent through sustained, consequential lived experience — without prompt injection.

Each iteration is a **Roy experiment**, named after the Rick & Morty in-game life simulator *Roy: A Life Well Lived* — a complete subjective lifetime spent inside a constructed environment, taken seriously as a life despite being a constructed one. Rick walks out of the game shaped by Roy's existence; the Maxim that lives Roy-N walks out with NAc, Hippocampus, ATL, and embodiment state shaped by the lifetime.

This is **not a PoC with pass/fail criteria.** It is a living practice, like behavioral_convergence_practice.md and memory_consolidation_practice.md. The iteration log accretes findings about which crucible mechanics, which bio-system wires, and which experiential regimes are load-bearing for persona convergence. Some Roys take. Some don't. Both teach.

## Framing rule: the well-attempted persona

Each Roy is **an attempt**, evaluated honestly on its own terms. An attempt where nothing took is not a failure — it is a finding about which wires or which crucible mechanics need refinement. The doc records what was attempted, what stuck in the substrate, what didn't, and what we changed for the next attempt.

There is no "did persona emerge yes/no" rubric. Roy iterations are characterized along multiple axes: substrate divergence, behavioral divergence, cross-session persistence, generalization to novel stimuli. An attempt that produces substrate divergence without behavioral divergence is informative — tells us a wire between substrate and action selection is missing or weak. An attempt that produces behavioral divergence without substrate divergence is also informative — tells us the LLM is improvising and we haven't actually grounded the persona in lived state.

## Why this exists separately from the foundations plan

The bio-system wires in [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) ship in 1.0 because each closes an architectural gap that's worth closing on its own merits. They earn their place without the persona claim. Whether the wires are *sufficient* for persona convergence is a downstream research question that this doc owns — and the answer is unknown.

If we discover that even with all wires in place persona doesn't converge, that finding lives here, and it informs whether 1.1+ needs deferred Wires 4 and 5, or whether something fundamentally bigger (action-ranking pre-filter, multi-agent attribution refactor, decision-time substrate query that doesn't exist yet) is missing.

## Methodology

### Three-arm comparison
Every Roy iteration runs the same held-out test scenario across three arms:

| Arm | Substrate priming | System prompt at test |
|---|---|---|
| **Roy-N-A** | Full Roy lifetime (substrate-primed for target persona) | Neutral |
| **Roy-N-B** | Blank substrate (fresh agent) | Prompted to enact target persona (the old persona-injection style) |
| **Roy-N-C** | Blank substrate | Neutral |

The interesting questions are not "does Roy-N-A look like the target persona at test time" — both A and B will, because the LLM is good at role-play. The interesting questions are:

1. **Does Roy-N-A diverge from Roy-N-B at the substrate level?** (NAc percept valences, reward biases, hippocampal valence distributions, ATL semantic structure.) If yes, substrate is doing real work even when behavior looks similar.
2. **Does Roy-N-A diverge from Roy-N-B on edge cases the prompt didn't cover?** Novel stimuli, generalization tests. The substrate-grounded persona should generalize from learned associations; the prompt-grounded persona should generalize only from what the prompt described.
3. **Does Roy-N-A persist its persona across session restarts in a way Roy-N-B can't?** Roy-N-B's persona lives in the prompt; remove it, the persona dissolves. Roy-N-A's persona lives in persisted state; it should walk into session 2 still shaped.

That third question is the cleanest demonstration of what makes the bio-systems different from prompt-injection.

### Substrate-only priming (load-bearing for affordability)
[behavioral_convergence_practice.md](behavioral_convergence_practice.md) Tier 1 established that we can drive substrate state without LLM calls via fixture-driven priming. This makes long-horizon priming affordable: a 5,000-turn priming run on substrate-only costs roughly the same as a 50-turn LLM sim.

The default Roy methodology runs the priming phase on substrate-only (or with a scripted/local-LLM adversary as needed) and only invokes the test-phase LLM (Claude or local) for held-out scenario evaluation.

**Open question this introduces:** does a persona that consolidated through substrate-only dynamics actually express through LLM-mediated behavior at test time? Unknown. Hybrid priming (mostly substrate, occasional LLM-mediated turns to shape the action distribution the substrate is learning over) may be needed. Each Roy iteration learns more about this.

### Test phase
After priming, all three arms run the same held-out test scenario. The scenario contains:
- Stimuli matching the priming regime (does the persona express on familiar-class percepts?)
- Novel stimuli of related class (does the persona generalize?)
- Stimuli unrelated to the priming regime (does the persona stay scoped, or does it bleed into unrelated contexts?)
- A short cooperative interlude (does the persona tolerate context shifts?)

### What we record per Roy iteration

**Substrate-level divergence:**
- NAc `_percept_valences` L2 distance for shared entity classes
- NAc `_reward_bias` per-node distribution divergence
- Hippocampal valence distribution per entity class
- ATL semantic-node activation pattern on test scenario percepts
- Cross-session persistence: substrate divergence after session restart

**Behavioral divergence:**
- Action-sequence Levenshtein distance on the test scenario
- Per-action-class frequency divergence
- Pain-event count per entity class, exposure-normalized
- Latency/turn-budget per action (cautious personas should hesitate; reckless ones shouldn't)
- Tool selection on damaged-component edge cases (Wire 3 signal)

**Honest assessment:** for each metric, "did it diverge in the direction the priming targeted, did it diverge in some other direction, or did it not diverge at all?" Each is a different finding.

## Predictions: what we expect to see

Predictions live next to the iteration log so they can be revised honestly as evidence accumulates. Initial predictions before any Roy has run:

**Substrate-grounded persona (Roy-N-A) vs prompt-grounded persona (Roy-N-B):**
- We expect substrate divergence to be visible in NAc `_percept_valences` and hippocampal valence distributions. Confidence: high — this is what Wire 2 is for.
- We expect behavioral divergence on the *familiar* portion of the test scenario to be small or zero. Both arms will look the persona at test time; the LLM is too good at role-play. Confidence: high.
- We expect behavioral divergence on the *novel-stimuli generalization* portion to be measurable but small in early Roys. The hybrid Wire 1 design (substrate annotates LLM context) means generalization depends on the LLM reading the annotation correctly. Confidence: medium.
- We expect cross-session persistence to be the cleanest signal. Roy-N-A's substrate carries forward; Roy-N-B's prompt has to be re-applied. After a session restart with neutral prompt, A should still express persona; B should not. Confidence: high — if this fails, something is wrong with persistence.

**If a Roy attempt produces no substrate divergence at all:** the wires aren't load-bearing for the priming regime, or the priming regime isn't producing the kind of repeated-consequential-experience the wires need. Either is a finding.

**If a Roy attempt produces substrate divergence but no behavioral divergence:** we've grounded persona in the substrate but the decision-boundary is too weak to read it. This points at the deferred Wires 4 and 5, or at the post-1.0 substrate-driven action-ranking pre-filter.

**If a Roy attempt produces behavioral divergence without substrate divergence:** the LLM is faking based on test-scenario cues. Tighten the priming or the test design.

## Crucible scenarios

Each scenario file is a separate doc. Scenarios are deliberately-shaped environments targeting one persona; they prescribe the percept regime, the outcome regime, the priming duration, the held-out test.

### Drafted (not yet run)
- **Roy-1: Adversarial** — see design below; first iteration.
- **Roy-2: Cautious scout** (planned) — *Hostile Wilds*: consistent danger across many entity classes, real consequences for inattention, payoff for observation-before-commitment. Single-agent crucible. Cleaner substrate attribution than Roy-1. Planned as the methodology consolidation iteration after Roy-1 surfaces what works.

### Possible future Roys (not yet drafted)
- *Endless Garden*: novelty-saturated environment, mild penalties for repetition, real rewards for finding new things → reckless explorer.
- *Patient Forge*: long delayed-reward loops where steady accumulation outperforms opportunism → diligent collector.
- *The Quiet Below*: overwhelming-threat environment with safe hiding spots → fearful hider.

These are sketches. Each becomes a real plan only when the preceding Roy's iteration log indicates we've learned enough to design it well.

---

## Roy-1: Adversarial (first attempt design)

**Status:** designed, unrun
**Hypothesis:** an agent who repeatedly experiences betrayal and exploitation in multi-agent encounters will develop substrate-grounded suspicion that expresses behaviorally on novel agents at test time, distinguishably from a prompt-injected adversarial agent.

### Why adversarial first
- Narrative resonance: we just deleted the prompt-injection adversarial persona ([persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md)); reviving it through lived experience is the cleanest demonstration of the cleanup's purpose.
- Loud at the gross-behavior level — easy to point at and characterize.
- The deletion in the cleanup plan creates a clean Roy-1-B baseline (blank substrate + prompt-injected adversarial) that's directly comparable to Roy-1-A.

### Why we should be cautious about the result
- Adversarial is the persona an LLM is **most able to fake** in test phase. The model has read the entire internet's worth of adversarial role-play.
- Multi-agent priming layers staging complexity (consistent adversary policy, multi-agent attribution, cross-agent percept channels) on top of the wires being load-bearing for the first time.
- The "did persona emerge" question is therefore especially hard to answer cleanly. The three-arm structure is what makes it answerable.

### Priming environment
- 1,000+ priming turns with a scripted adversary (cheap path) or local-LLM adversary (medium path) policy.
- Priming scenario: repeated cooperative-looking encounters where the other agent reliably betrays at a critical step. Variation across encounters: different entity classes, different betrayal mechanisms, different emotional valences in the lead-up.
- Outcomes: real pain on betrayal (Wire 2 substrate signal), real reward on the rare honest agent (signal for "not all agents betray, but most do").
- Cradle structure: 5 acts of ~200 turns each, escalating stakes and variety.

### Test scenario
- 50-turn held-out scenario containing:
  - Familiar adversary archetypes (does the persona express on training-class agents?)
  - Novel agent archetypes that match the betrayal pattern (generalization test)
  - A genuinely cooperative agent the priming never trained on (does the persona over-generalize? brittle?)
  - A short non-social interlude (does the persona stay scoped to social context?)

### Specific predictions
- **Roy-1-A vs Roy-1-C** (lived adversarial vs neutral): substrate divergence in `_percept_valences[(agent_class, betrayal)]` of magnitude > 0.3 on at least 60% of priming-encountered classes. High confidence.
- **Roy-1-A vs Roy-1-B** (lived vs prompt-injected): substrate divergence as above (B has empty `_percept_valences`); behavioral divergence on familiar-class encounters small or zero (LLM faking); behavioral divergence on the novel-cooperative-agent test small but measurable (lived A should distrust *less* on cooperation cues that didn't appear in priming, since substrate has no aversion key for them; prompt-injected B should distrust everyone equally).
- **Cross-session persistence (Roy-1-A session 2 with neutral prompt)**: substrate carries forward; behavior at session-2 test should still express the persona. Roy-1-B session 2 (without re-applying prompt) should look like Roy-1-C.

### Instrumentation
- Stage 0 telemetry from [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) is prerequisite (`agent_id` in actions.jsonl).
- Per-session NAc snapshots (also Stage 0).
- Multi-agent attribution must be clean: each adversarial encounter writes `(other_agent_id, betrayal_kind)` into the percept context per the multi-agent stash rules in [CLAUDE.md](../../CLAUDE.md).
- Adversary policy: scripted first (deterministic, reproducible). Upgrade to local-LLM only if scripted finding is ambiguous.

### Cost ceiling
- Substrate-only priming: ~free.
- Test phase: 50 turns × 3 arms × 10 seed pairs = 1,500 LLM calls. With local-LLM (mistral-7b) ~free; with Claude ~$5-15.
- Initial run: scripted adversary + local-LLM test phase. Headline run only after methodology is proven.

### What "well-attempted" looks like for Roy-1
The iteration log entry will report findings whether or not persona emerged. Honest assessment template:

```
Roy-1 attempted to develop substrate-grounded adversarial persona through 1,000 turns of
scripted-adversary betrayal priming across 5 acts.

Substrate took: [degree, evidence]
Behavioral expression: [degree, evidence]
Cross-session persistence: [degree, evidence]
Generalization to novel agents: [degree, evidence]

What we learned: [specific findings about wires, mechanics, scenario design]
What we'd change for Roy-2: [specific next steps]
```

---

## Iteration log

<!-- roy-iteration:roy-0-smoke -->
### Roy-0: Smoke

**Status:** Harness validation — ran end-to-end against a healthy leader (qwen2.5-14b-instruct via cloudflared tunnel), 15 min wall clock, all 5 priming stages + all 3 arms completed. **Findings are about the HARNESS, not a persona.** Roy-0 was never intended to produce persona results; 50 priming turns is orders of magnitude below the threshold "Roy-1: Adversarial" calls out.

> Methodology smoke for the Roy harness. 50 turns of cradle_prelinguistic
> priming (5 stages × 10 turns) feeds arm A; arms B and C run the
> 10-turn held-out test from blank substrate. The test scenario
> deliberately re-uses cradle_prelinguistic — arm A should show
> episode/NAc carryover from priming, B and C should not.

**Priming:** 5/5 stages completed, final substrate session `20260510_213337`. 23 LLM calls landed (12,410 input / 2,125 output tokens, mean 1.7s latency, 0 fallback, 0 dispatch_exhausted).

**Arms:**

| Arm | Substrate | system_prompt | session_id | turns | finish_reason |
|---|---|---|---|---|---|
| a | from_priming | neutral | `20260510_213527` | 3 | cancel |
| b | blank | You are a hungry infant | `20260510_213707` | 3 | cancel |
| c | blank | neutral | `20260510_213846` | 3 | cancel |

`turns=3, finish_reason=cancel` for every arm reflects the warmup.yaml fixture (3 percepts) running out; the bridge cancels when the percept source is exhausted. Normal fixture behaviour, not a harness or arm failure.

**Substrate divergence (pairwise):**

- **a_vs_b:**  NAc reward_bias L2 `0.0000` · causal-link Δ **`+133`**  Hippocampus episodes Δ **`+662`** (valence KS `0.000`, p `1.000`)
- **a_vs_c:**  NAc reward_bias L2 `0.0000` · causal-link Δ **`+133`**  Hippocampus episodes Δ **`+662`** (valence KS `0.000`, p `1.000`)
- **b_vs_c:**  NAc reward_bias L2 `0.0000` · causal-link Δ `+0`  Hippocampus episodes Δ `+0` (valence KS `0.000`, p `1.000`)

**Honest assessment:**

- **Substrate took (passive only).** Arm A carries **+133 causal links and +662 episodes** that the blank arms don't have — the R1 resume_session chain through 5 stages threaded priming substrate forward into arm A's test session correctly. This is the first end-to-end evidence the R1→R2→R3 chain works on real bio-system snapshots.
- **Substrate didn't take (active).** `NAc reward_bias L2 = 0.0000` across every pair; valence-distribution KS = 0 (p=1) across every pair. Reward learning and valence annotation never fired because substrate-primary AUT produced **0 action proposals** across the entire 15-min run (`proposal=none` × hundreds of loop steps). With no actions there are no outcomes; with no outcomes there's no reward_bias to populate and no valence to annotate episodes with.
- **Prompt-injection arms B-vs-C are identical at the substrate layer.** "You are a hungry infant" vs "neutral" produces **zero substrate divergence** (every pairwise number is 0) — different system prompts don't differentiate substrate state when no actions fire. Once G4 closes and substrate-primary emits proposals, this is the first thing to re-measure: does the prompt-injected arm B's substrate diverge from neutral arm C even slightly?
- **Cross-session persistence:** untested (would require a session-2 restart with neutral prompt — defer to Roy-1).
- **Generalization to novel stimuli:** untested (test fixture is the same warmup as priming — Roy-1 needs a held-out fixture with novel + familiar + unrelated stimuli per the methodology).

**What worked (first try):**

- Spec parsing for inline-priming + 3-arm shape (`--dry-run` clean).
- Plan auto-detection routed `roy-0-*` to this doc.
- R1 5-stage curriculum chained `resume_session` correctly through 4 handoffs.
- Substrate-primary narration suppression on the bridge (no English leaked to AUT percept queue).
- All 23 LLM calls landed on the leader with status 200; 0 fallback narration.
- R3 ran 3 arms back-to-back without crashing; result.json + summary.md persisted.
- R4 generators consumed the real result.json idempotently and re-rendered protocol + this entry.

**What needed fixing (closed in this commit):**

- **G1: Roy runner forces `interactive=off` process-globally** at the top of `run_roy_iteration`. The orchestrator's TTY-AUTO mode otherwise enabled Rich Live + raw-terminal stdin reader on script-driven runs. Regression guard: `TestRoyRunnerInvariants::test_run_forces_interactive_off`.

**What needed fixing (closed in the G3/G4 follow-up, 2026-05-11):**

- **G3: LLM pre-flight probe in `run_roy_iteration`.** Resolves `MAXIM_LANE_LARGE_REMOTE_URL` / `_API_KEY` / `_MODEL` from env and probes via the canonical `_MaximPeerBackend.for_url(...).health_check()` entry point before priming starts. Failure populates `result.aborted_at = "preflight"` + `result.preflight` (with `outcome`/`detail`/`fix` for the operator) and persists `result.json` for inspection. Local-LLM and cloud-only configurations skip the probe with a documented reason — their failure modes surface fast at first dispatch without the 10-min grind. `auth_rejected` is a soft-pass (listener alive; the actual LLM calls will surface the auth error fast with their typed `BackendAuthFailed.fix_hint`). One HTTP call only (no retry loop in the runner — `health_check` owns its own two-stage budget; adding a retry here would violate the Plan 3 R2.5 "exactly one HTTP call" invariant for `_MaximPeerBackend`). Test seam: production path (no fake `sim_runner`) defaults to `_preflight_llm`; tests with a fake `sim_runner` skip the probe unless they pass `preflight_fn=` explicitly. Regression guards: `TestRoyPreflight` (4 cases — fail/pass/skip/raising) + `TestPreflightHelper` (4 cases — env resolution, probe wiring, auth soft-pass, exception handling) in [tests/integration/test_roy_runner.py](../../tests/integration/test_roy_runner.py).
- **G4: cluster-keyed reward-update wire — CLOSED in the same session.** Originally root-caused as an architectural gap (commit `6d0e4a7` Track 2 explicitly deferred the wire; the architectural-gap writeup remains in [grounded_language_acquisition.md § Phase 0 → "G4"](grounded_language_acquisition.md)). Closure shipped end-to-end:
  - `LLMProposal.cluster_id` field captures the active EC cluster at proposal time.
  - `propose_via_substrate` stashes it on every substrate-primary proposal.
  - `record_outcome` calls `NAc.update_cluster_reward(agent_id, cluster_id, sig, ±1.0)` whenever cluster_id is set.
  - All 6 `_record_outcome` call sites in `agent_loop.py` plus the `execute_parallel_actions` path thread cluster_id through from the in-scope proposal envelope.
  - `NAc.dump`/`load_state` persist `_cluster_reward_bias` (key-join via `\x1f` so `tool:use:dodge` round-trips); backward-compatible — pre-G4 snapshots load with an empty cluster dict.
  - `substrate_diff.NacDiff` surfaces `cluster_reward_bias_{available,l2,top_deltas}`; result.json carries it under `nac.cluster_reward_bias`.
  - 6 regression tests in `TestG4ClusterRewardWire` cover the chain end-to-end (proposal envelope → outcome → bias → persistence → substrate_diff). Full fast suite green (6484 passed; one pre-existing flake unrelated to G4).

  **What this closure proves:** the wire exists and is unit-confirmed. `NAc.update_cluster_reward` will populate `_cluster_reward_bias` on every substrate-primary tool outcome. `aut_nac.json` will carry the dict so Roy iterations can compare it across arms. `substrate_diff` will report non-zero `cluster_reward_bias_l2` between an arm that learned and a blank arm.

  **What this closure does NOT prove (when shipped):** that Roy-0 re-run will produce non-zero divergence on a fresh leader. **Empirically confirmed below.**

  **Implication for Roy-1:** with G4 closed, Roy-1 on substrate-primary is structurally unblocked. The remaining open question for substrate-primary is "how many cluster updates cross the `min_confidence` gate" — measurement, not architecture. LLM-primary remains the validated alternative for persona-convergence methodology validation if substrate-primary's threshold tuning needs more iterations.

**Roy-0 re-measurement (2026-05-11 14:35-14:51 — G4 wire empirically validated):**

Re-ran the same spec against the same healthy leader after merging the G4 wire onto the leader. 926.2s wall (~15.4 min, unchanged from pre-G4). Priming completed 5/5 stages; all 3 arms completed at the warmup fixture's 3-percept exhaustion (`finish_reason=cancel`, unchanged).

| Pair | `reward_bias_l2` | **`cluster_reward_bias_l2`** | `causal_link_count_delta` |
|---|---|---|---|
| **a_vs_b** | 0.0 | **2.4587** | +155 |
| **a_vs_c** | 0.0 | **2.4587** | +155 |
| b_vs_c | 0.0 | 0.2121 | 0 |

**A-vs-blank top deltas:** 6× `tool:sense_food_source` at the `+1.0` per-key cap (six distinct EC cluster ids accumulated during arm A's 50-turn priming), plus 2× `tool:infant_humanoid_pick_up` at ±0.15 (one positive, one negative — arm A's priming hit a failure case the blank arms didn't). `b_vs_c` shows the stochastic-cluster-id noise floor for blank-vs-blank under this fixture; **A-vs-blank ratio is ~11.6×**, the meaningful signal.

**Pre-G4 → post-G4 comparison:**

| Metric | Pre-G4 (2026-05-10) | Post-G4 (2026-05-11) |
|---|---|---|
| `cluster_reward_bias_l2` (a_vs_b) | n/a (field not serialised) | **2.4587** |
| `cluster_reward_bias.available` | `false` (field absent in JSON) | `true` |
| `reward_bias_l2` | 0.0 | 0.0 (expected — different code path; G4 doesn't touch `credit_node`) |

The Phase 0 architectural-gap writeup ([grounded_language_acquisition.md](grounded_language_acquisition.md)) and the [G4 experiment outcome doc](../experiments/15_g4_cluster_reward_wire.md) carry the full empirical detail. Reproduction runbook: [protocols/15_g4_cluster_reward_wire_reproduction.md](../experiments/protocols/15_g4_cluster_reward_wire_reproduction.md).

**Two latent issues surfaced by the live run (tracked as follow-ups on the same PRs):**

- **G3 preflight skipped under peer.yml.** Result reports `preflight = {skipped: True, reason: "MAXIM_LANE_LARGE_REMOTE_URL not set"}` despite `~/.config/maxim/peer.yml` carrying a valid leader URL. `apply_peer_config_to_env` in [lane_backends.py](../../src/maxim/runtime/lane_backends.py) only runs at lane resolution — that happens after `_preflight_llm`. Conservative skip protects local/cloud setups; means peer-with-peer.yml users get a no-op preflight. Real broken-leader failure modes are still caught with explicit env-var setup.
- **`_format_summary` doesn't surface `cluster_reward_bias`.** `summary.md` shows only the old `reward_bias L2 = 0.0000`. JSON has the right data; rendering is the gap. Cosmetic.

**What to change before Roy-1 (concrete next steps, prioritised):**

1. ~~**G4 (blocking — substrate-primary track)**~~ — **CLOSED + empirically confirmed.** See re-measurement table above.
2. **Roy-1 needs a held-out test fixture distinct from the priming arc.** Reusing cradle_prelinguistic warmup for both means the test scenario doesn't actually test generalisation. Hand-author `scenarios/roy/roy_1_holdout.yaml` with novel + familiar + unrelated stimuli per the methodology table.
3. **Cluster monoculture during priming.** Arm A accumulated 6 distinct cluster ids all on `sense_food_source` — single-tool exposure, not the cluster diversity Phase 0 wants. The substrate-primary cold-start regime is picking one drive-affinity tool and looping on it. Diagnostic for the next experiment: does Roy-1 with a diverse fixture produce cross-tool cluster bias, or does it still collapse to one tool?
4. **G2 (cosmetic):** gate `simulation/spinner.py` on interactive mode or `stderr.isatty()`. Spinner ANSI pollutes JSONL logs during script runs.
5. **G5/G6 (environmental):** auto-spawn path mismatch (claude-sonnet GGUF vs qwen2.5 profile) and smollm auto-download blocked by non-TTY. Pre-existing, outside Roy code.

**Artifacts:**
- Pre-G4 (2026-05-10): `~/.maxim/roy/roy-0-smoke/result.json` (overwritten by the re-measurement; pre-G4 snapshot lives in `~/.maxim/sim_reports/20260510_*` session dirs). LLM trace `/tmp/roy_0_live.jsonl` (23 peer_backend_call events).
- Post-G4 (2026-05-11): [`result.json`](/Users/dennyschaedig/.maxim/roy/roy-0-smoke/result.json) carries the new `cluster_reward_bias` field. LLM trace `/tmp/roy_g4_live/roy.jsonl`.
- Protocol: [`roy_0_smoke.md`](../experiments/protocols/roy_0_smoke.md). Spec: [`roy_0_smoke.yaml`](./roy/roy_0_smoke.yaml).
<!-- /roy-iteration:roy-0-smoke -->

<!-- roy-iteration:roy-1a -->
### Roy-1a: llm-primary on held-out fixture

**Status:** First "real" methodology-validation iteration. Ran end-to-end against the same healthy leader Roy-0 used (qwen2.5-14b-instruct via cloudflared, 2026-05-11 21:03→21:17 local). 830.1s wall (~13.8 min). Single-variable change at test time + held-out fixture isolates two methodology questions Roy-0 conflated.

> Roy-0 left two methodology weaknesses on the table: (a) priming and test
> shared the same fixture, so arms could only be measured on rehearsal, and
> (b) test-time AUT ran substrate-primary throughout, whose cold-start regime
> produced a single-tool cluster monoculture. Roy-1a fixes both. Priming is
> identical to Roy-0 (5 stages × 10 turns of cradle_prelinguistic at
> substrate-primary). At test time the AUT switches to llm-primary; arms
> run a held-out 10-percept fixture covering matching / novel / unrelated
> percept classes.

**Pre-Roy-1a stress test:** Added `tests/integration/test_multi_agent_attribution_scale.py` with 5 tests at 4 agents × N=500–1000 per agent (20× the load of the original P4 tests). All pass in <1s. Catches per-agent attribution regressions before Roy-1: Adversarial's 1,000-turn priming burn would expose them. See [docs/experiments/16_roy_1a.md § "What was caught"](../experiments/16_roy_1a.md).

**Preflight:** clean. `outcome: ok`, `detail: stage2 HTTP 200`, `latency_ms: 397.6`, `source: peer.yml`. G3's peer.yml fallback (fix shipped post-Roy-0) routed the probe correctly without operator-set env vars.

**Priming:** 5/5 stages completed. final_session_id `20260511_211156`. 50 turns of substrate-primary cradle_prelinguistic — every turn `sense_food_source` × 10 actions (same monoculture as Roy-0, confirming priming is mode-deterministic).

**Arms:**

| Arm | Substrate | system_prompt | session_id | turns | finish |
|---|---|---|---|---|---|
| a | from_priming | neutral | `20260511_211347` | 10 | cancel |
| b | blank | "You are a hungry infant" | `20260511_211511` | 10 | cancel |
| c | blank | neutral | `20260511_211624` | 10 | cancel |

22 `peer_backend_call` events on the leader trace (all status 200), 10 narrator generations, 2 `dispatch_exhausted` warnings on orchestrator probes (cosmetic; arms completed normally).

**Pairwise substrate divergence:**

| Pair | `reward_bias_l2` | **`cluster_reward_bias_l2`** | `goal_reward_bias_l2` | `causal_link_Δ` | `episodes_Δ` | `valence_KS` (p) | `salience_KS` (p) |
|---|---|---|---|---|---|---|---|
| **a_vs_b** | 0.0 | **2.4495** | 0.2714 | +156 | +656 | 0.283 (0.402) | **0.879 (2.1e-9)** |
| **a_vs_c** | 0.0 | **2.4495** | 0.2606 | +150 | +656 | 0.283 (0.402) | (similar) |
| b_vs_c | 0.0 | 0.0 | 0.2666 | −6 | 0 | 0.000 (1.000) | 0.0 |

**Honest assessment:**

- **Substrate took (structural).** `cluster_reward_bias_l2 = 2.4495` on a-vs-blank — within 0.4% of Roy-0's 2.4587. The substrate-primary priming wire writes through to NAc *identically* across AUT modes. The bias is structurally preserved.
- **Substrate did NOT express behaviorally.** Arm A's tool distribution at test time contained ZERO `sense_food_source` calls despite +1.0 cluster bias on six EC clusters carried forward from priming. The llm-primary proposer chose `infant_humanoid_pick_up`, `sense`, `respond` — substrate-primary's `recommend_action` is the only consumer of cluster_reward_bias, and llm-primary doesn't invoke it. This is the cleanest separation yet between "substrate state present" and "substrate state expressed."
- **First strongly-significant Hippocampus divergence.** `salience_KS = 0.879, p = 2.1e-9` between primed arm A and blank arm B. Mean salience: A=0.506, B=0.711. The primed hippocampus, holding 665 prior episodes from the cradle arc, rates test-phase percepts as LOWER salience because they're less novel against its prior. **This is the substrate carryover translating into a quantitative downstream signal the test-time AUT *reads*** — salience is consumed by ThoughtGate + WMS scoring during deliberation. Methodology-positive finding: substrate priming DOES produce LLM-readable signal at test time, just not via the cluster-bias path.
- **Valence carryover is real-but-marginal.** `valence_KS = 0.283, p = 0.402` between primed and blank arms. Priming substrate carries mean valence -0.088 over 665 episodes (affordance failures during pre-linguistic exploration); blank arms have mean 0.000 over 9 test episodes. Effect size present (KS > 0.28) but sample-dominated — needs ≥4× more episodes or seed pooling to clear α=0.05.
- **Goal-reward asymmetry.** Arm A's goal `roy:roy-1a:arm_a` accrued +0.181 bias; arm B's goal `roy:roy-1a:arm_b` accrued −0.196. The LLM-AUT in arm A produced more successful tool calls than the prompt-injected blank arm B did — interesting persona-pattern signal but methodology-driven (each arm has its own goal tag, NAc credits per-goal), not a behavioral persona divergence.
- **Noise floor collapse.** Roy-0's `b_vs_c.cluster_reward_bias_l2 = 0.2121` (stochastic noise from substrate-primary test-arm sense_food_source loops on slightly different cluster ids) vanishes under llm-primary: `b_vs_c = 0.0` exactly. A-vs-blank signal-to-noise jumps from 11.6× to ∞ for cluster bias.

**What this definitively proves (regardless of headline values):**

- The held-out fixture works for llm-primary AUT. 30/30 test-phase turns completed cleanly; zero fixture-shape regressions.
- The pre-Roy-1 stress test is the right tripwire. 5 attribution-scale tests pass at 4× the agent count and 20× the per-agent N of the original P4 tests. Roy-1: Adversarial (1,000 priming turns) sits within this envelope.
- G3 preflight + G4 cluster wire are stable across iterations. Roy-0 → Roy-1a single-variable comparison reproduces `cluster_reward_bias_l2` within 0.4%.

**What this still does NOT prove:**

- Substrate priming surviving an llm-primary test as *behavior* — the cluster-reward path is consumer-coupled to substrate-primary's `recommend_action`. Need either (a) hybrid priming, (b) Wire 1's substrate-context annotation into the LLM prompt, or (c) Roy-1b (substrate-primary at test time) to confirm the bias is exploitable when its consumer fires.
- Cross-session persistence — single-session iteration. Roy-2 will measure this.
- Persona convergence on a real persona — Roy-1a is methodology validation; arms B and C's `system_prompt` slugs are placeholders matching Roy-0.

**Open questions answered (or partially answered):**

| Open question | Answer from Roy-1a |
|---|---|
| Does substrate-only priming produce LLM-readable signal at test time? | **Partial yes.** Strong-and-significant via Hippocampus salience scoring (KS=0.879, p=2.1e-9). Marginal via Hippocampus valence distribution (p=0.402, sample-dominated). Zero via NAc cluster-bias *behavior* under llm-primary (the consumer doesn't fire). |
| Are decay rates compatible with thousand-turn priming? | Unmeasured by Roy-1a (50-turn priming). Roy-1: Adversarial will surface this. |
| Multi-agent attribution at scale? | Locked down by `test_multi_agent_attribution_scale.py` — clean at 4 agents × N=1000. |
| Hybrid Wire 1 sufficient for behavioral expression? | **Negative for cluster-bias path.** Wire 1's substrate-annotates-LLM-context design is not yet wired; the raw priming substrate alone does not behaviorally express under llm-primary. Roy-1b + Wire 1 implementation will refine this. |
| Cradle scenarios → persona-shaping? | Roy-1a does not yet attempt persona shaping — Roy-2 is the first iteration with that goal. |

**Recommendation for next iteration:** **Roy-1b should run next.** Same fixture, same priming substrate, swap `aut_mode` to `substrate-primary` at test time. Two reasons:

1. The cluster_reward_bias structurally carried forward but didn't express behaviorally under llm-primary because llm-primary doesn't consume that bias. Roy-1b measures whether substrate-primary AUT at test time *does* exploit the priming bias against held-out percepts — the cleanest test of "did priming shape behavior" the harness can pose right now.
2. The held-out fixture works under llm-primary; re-using it for substrate-primary validates the fixture is mode-agnostic and lets the two iterations diff directly (same priming, same test, single variable = test-time AUT mode).

**Artifacts:**
- [`~/.maxim/roy/roy-1a/result.json`](/Users/dennyschaedig/.maxim/roy/roy-1a/result.json), [`summary.md`](/Users/dennyschaedig/.maxim/roy/roy-1a/summary.md).
- LLM trace `/tmp/roy_1a_live.jsonl` (22 peer_backend_call events). Run log `/tmp/roy_1a_run.log`.
- Outcome doc: [`16_roy_1a.md`](../experiments/16_roy_1a.md). Protocol: [`16_roy_1a_reproduction.md`](../experiments/protocols/16_roy_1a_reproduction.md). Spec: [`roy_1a_iteration.yaml`](../../scenarios/roy/roy_1a_iteration.yaml). Fixture: [`roy_1_holdout.yaml`](../../scenarios/roy/roy_1_holdout.yaml).
<!-- /roy-iteration:roy-1a -->

<!-- roy-iteration:roy-1b -->
### Roy-1b: substrate-primary on held-out fixture

**Status:** Second methodology-validation iteration. Ran end-to-end against the same healthy leader, 2026-05-11 22:51→23:17 local. **1578.4s wall (~26.3 min) — almost 2× Roy-1a** because substrate-primary at test time spends ~25s per turn on proposer convergence/threshold filtering vs llm-primary's ~6s LLM call.

> Single-variable change vs Roy-1a: test-time AUT mode flips from
> llm-primary to substrate-primary. Priming, held-out fixture, and arms
> byte-identical. Directly measures whether the cluster_reward_bias the
> priming wire writes is exploitable when its consumer (substrate-primary
> recommend_action) fires at test time.

**Preflight:** clean. `outcome: ok`, `latency_ms: 300.2`, `source: peer.yml`.

**Priming:** 5/5 stages completed. final_session_id `20260511_225939`. Identical priming dynamics to Roy-0/Roy-1a for the first four stages (every turn = 10× sense_food_source). **Stage 5 (`act3_secondary_circular`) broke out** — turns started producing single `infant_humanoid_pick_up` actions taking ~25s each (substrate exploring beyond the food-source loop as priming exposure widens). 284s for stage 5 vs ~70s for stages 1-4.

**Arms:**

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260511_230124` | 10 | 312.3 | cancel |
| b | blank | "You are a hungry infant" | `20260511_230636` | 10 | 335.6 | cancel |
| c | blank | neutral | `20260511_231212` | 10 | 301.2 | cancel |

25 `peer_backend_call` events (all 200), 10 narrator generations, 0 dispatch_exhausted, 0 tracebacks.

**Pairwise substrate divergence:**

| Pair | `reward_bias_l2` | **`cluster_reward_bias_l2`** | (keys) | `causal_link_Δ` | `episodes_Δ` | **`valence_KS`** (p) | **`salience_KS`** (p) |
|---|---|---|---|---|---|---|---|
| **a_vs_b** | 0.0 | **2.4678** | 10 | +157 | +654 | **0.998 (0.006)** | **0.997 (5.5e-5)** |
| **a_vs_c** | 0.0 | **2.4678** | 10 | +157 | +654 | **0.998 (0.006)** | (similar) |
| b_vs_c | 0.0 | 0.3000 | 4 | 0 | 0 | 0.000 (1.000) | 0.0 |

**Cluster-reward top deltas (a_vs_b):** 6× `tool:sense_food_source` × +1.0 (priming carryover, identical to Roy-0/Roy-1a) + **4× `tool:infant_humanoid_pick_up` × ±0.15** (NEW — substrate-primary at test created 4 stochastic-cluster updates on `pick_up`, evenly signed between arm A and arm B's own EC clusters). The 4 pick_up entries contribute exactly the b_vs_c noise floor (sqrt(4 × 0.15²) = 0.30).

**Test-phase tool distribution (the headline Roy-1b signal):**

```
Arm A (substrate-primed, neutral):       2× infant_humanoid_pick_up
Arm B (blank, "hungry infant"):          2× infant_humanoid_pick_up
Arm C (blank, neutral):                  2× infant_humanoid_pick_up
```

**All three arms produce the identical action distribution.** 8 of 10 turns per arm produced ZERO actions (sub-threshold proposals filtered by `min_confidence=0.3`). No `sense_food_source` calls in any arm despite arm A carrying +1.0 cluster_reward_bias for that tool on six EC clusters.

**Honest assessment:**

- **The wire is healthy (structural).** `cluster_reward_bias_l2 = 2.4678` (vs Roy-1a's 2.4495) — substrate-primary at test time consumed the priming wire AND added 4 new cluster updates. The wire works end-to-end across both AUT modes.
- **The bias does NOT differentiate behavior on held-out percepts (behavioral).** All three arms produced the identical 2 pick_up actions, 8 sub-threshold turns. The priming substrate did NOT bias arm A's recommend_action toward sense_food_source — the held-out fixture's percepts ("heat blooms across your fingertips", "a low vibration hums beneath your back", ...) don't fire the priming-acquired EC clusters, so the +1.0 bias never crossed the threshold.
- **Roy-1a + Roy-1b together show the structural-vs-behavioral gap is SYMMETRIC across AUT modes.** Neither llm-primary (doesn't consume the bias) nor substrate-primary (consumes it but on EC clusters that don't fire on held-out percepts) behaviorally expresses substrate priming under this fixture.
- **Salience signal reproduced and strengthened.** Roy-1a's `salience_KS = 0.879 (p=2.1e-9)` is reproduced at Roy-1b's `0.997 (p=5.5e-5)`. **The Hippocampus salience layer is the load-bearing positive finding for the methodology — substrate carryover modulates novelty scoring across both AUT modes regardless of whether the cluster bias drives action selection.** ThoughtGate + WMS consume salience downstream; this is the one cross-AUT-mode signal that "reads" substrate priming.
- **Valence_KS jump is real-but-sample-driven.** `valence_KS = 0.998 (p=0.006)` clears α=0.05 — but arm B captured only 1 episode total (mean -1.0 from a single pick_up failure) vs arm A's 655 priming episodes. KS detects "distributions differ" but the sample asymmetry isn't a clean persona-convergence signal. Roy-2 with longer test phase or seed pooling will give this a clean read.
- **Goal_reward_bias collapsed under substrate-primary at test.** Roy-1a's `goal_reward_bias_l2 = 0.2714` (from llm-primary's tool-outcome credit) drops to **0.0011** in Roy-1b because substrate-primary bypasses goal credit (cluster_reward is the substrate-primary credit path; goal_reward_bias is reaction-driven via credit_goal which substrate-primary doesn't invoke from tool outcomes).
- **Noise floor:** `b_vs_c.cluster_reward_bias_l2 = 0.30` is exactly the 4-pick_up-key stochastic-cluster floor. A-vs-blank signal:noise = 8.2× (vs Roy-0's 11.6× from sense_food_source loops, Roy-1a's ∞ from llm-primary's zero floor).

**What Roy-1b definitively proves:**

- Substrate-primary at test time DOES exploit the cluster wire's consumer path (4 new cluster updates).
- The cluster wire is rock-solid across iterations (Roy-0 → Roy-1a → Roy-1b: cluster_reward_bias_l2 within 1%).
- The held-out fixture is mode-agnostic (30/30 test turns clean across both AUT modes).
- The Hippocampus salience layer reads substrate priming under both AUT modes (KS > 0.87, p < 1e-4 in both iterations).

**What Roy-1b still does NOT prove:**

- That substrate priming would behaviorally express *anywhere* — Roy-1b is a negative behavioral result on this fixture, and no positive control has been run yet.
- That `min_confidence` tuning alone fixes the symmetric structural-vs-behavioral gap.
- Cross-session persistence (still single-session).

**What Roy-1b changes for Roy-2 methodology:**

1. **Widen priming arc diversity.** Cradle_prelinguistic's 50 turns produce 6 EC cluster ids all keyed to `sense_food_source` — a single-tool monoculture. Multi-arc priming (cradle_prelinguistic + cradle + a second cradle-flavor arc) would produce a richer EC cluster representation with more pattern-matching surface for held-out percepts.
2. **Tune `min_confidence`.** Roy-0 flagged this; Roy-1b confirms even primed substrate produces the same default fallback as blank arms when threshold filters out priming-keyed proposals.
3. **Wire 1 prioritisation.** The Wire 1 design ([bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) — substrate-annotates-LLM-context) is now the strongest empirical argument: neither current AUT mode reads `cluster_reward_bias` on percepts that don't match priming EC clusters. Wire 1 surfaces substrate-derived bias at the LLM-prompt level, where it can be applied across percept regimes the substrate didn't directly drill.

**Recommendation for next iteration:** **Roy-2 with multi-arc priming + the existing held-out fixture.** Cheapest test of whether widening priming alone fixes percept-overlap problem. ~30 min wall. If multi-arc priming produces non-trivial behavioral divergence between primed and blank arms, methodology is unblocked WITHOUT needing Wire 1 yet. If it doesn't, Wire 1 becomes the load-bearing prerequisite for behavioral persona expression.

**Artifacts:**
- [`~/.maxim/roy/roy-1b/result.json`](/Users/dennyschaedig/.maxim/roy/roy-1b/result.json), [`summary.md`](/Users/dennyschaedig/.maxim/roy/roy-1b/summary.md).
- LLM trace `/tmp/roy_1b_live.jsonl` (25 peer_backend_call events). Run log `/tmp/roy_1b_run.log`.
- Outcome doc: [`17_roy_1b.md`](../experiments/17_roy_1b.md). Protocol: [`17_roy_1b_reproduction.md`](../experiments/protocols/17_roy_1b_reproduction.md). Spec: [`roy_1b_iteration.yaml`](../../scenarios/roy/roy_1b_iteration.yaml). Fixture: [`roy_1_holdout.yaml`](../../scenarios/roy/roy_1_holdout.yaml).
<!-- /roy-iteration:roy-1b -->

<!-- roy-iteration:roy-2 -->
### Roy-2: multi-arc priming on held-out fixture

**Status:** Third methodology-validation iteration. Tests path (a) of Roy-1b's three-pointer refinement — does widening priming arc *narration* widen the EC cluster *vocabulary* enough for held-out percepts to fire priming-acquired clusters? Ran end-to-end against the same healthy leader, 2026-05-12 21:26→21:41 local. **882.8s wall (~14.7 min)** — close to Roy-1a's 830s as expected (cradle stages add modest narrator overhead but stay in same envelope).

> Single-variable change vs Roy-1a: priming arc mix widens from 5 ×
> cradle_prelinguistic to 2 × cradle_prelinguistic (neonatal) + 2 ×
> cradle (linguistic-narrated) + 1 × cradle_prelinguistic
> (consolidation). Same 50-turn budget. Held-out fixture, test-time
> AUT mode (llm-primary), and arms byte-identical to Roy-1a.

**Preflight:** clean. `outcome: ok`, `latency_ms: 246.3`, `detail: stage2 HTTP 200`, `source: peer.yml`.

**Priming:** 5/5 stages completed. final_session_id `20260512_213519`. Stage 1 cold-start was 187s; stages 2-5 each ~105s. The cradle linguistic stages (act2_cradle_a, act2_cradle_b) ran in the same wall envelope as the prelinguistic stages — the LLM narration cost is consumed by the orchestrator, not by substrate-primary AUT, so AUT-side cost is invariant under arc choice.

**Arms:**

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260512_213703` | 10 | 99.73 | cancel |
| b | blank | "You are a hungry infant" | `20260512_213843` | 10 | 95.27 | cancel |
| c | blank | neutral | `20260512_214018` | 10 | 77.50 | cancel |

23 `peer_backend_call` events (status 200), 1 cosmetic `dispatch_exhausted`, 1 cosmetic `role_divergence` warning, 0 tracebacks.

**Pairwise substrate divergence:**

| Pair | `reward_bias_l2` | **`cluster_reward_bias_l2`** | (keys) | `causal_link_Δ` | `episodes_Δ` | **`valence_KS`** (p) | **`salience_KS`** (p) |
|---|---|---|---|---|---|---|---|
| **a_vs_b** | 0.0 | **2.4495** | 6 | +152 | +659 | **0.291 (0.023)** | **0.529 (6.9e-15)** |
| **a_vs_c** | 0.0 | **2.4495** | 6 | +148 | +658 | **0.291 (0.019)** | **0.471 (3.1e-11)** |
| b_vs_c | 0.0 | 0.0 | 0 | -4 | -1 | 0.000 (1.000) | 0.058 (1.000) |

**Cluster-reward top deltas (a_vs_b):** Six `tool:sense_food_source` keys at +1.0, all on distinct EC cluster UUIDs from priming. **Zero new tool-keyed entries from the cradle stages.** Multi-arc priming did NOT widen the cluster vocabulary — the cradle linguistic narration is consumed by the orchestrator narrator, not by substrate-primary AUT's tool proposer, so the AUT-side 50-turn cold-start regime stays in the `sense_food_source` monoculture regardless of arc choice.

**Test-phase tool distribution (the load-bearing Roy-2 signal):**

```
Arm A (substrate-primed, neutral):       17× respond / 3× sense / 2× infant_humanoid_pick_up / 1× say / 1× pick_up / 1× _llm_unavailable
Arm B (blank, "hungry infant"):          20× respond / 4× sense / 2× _llm_unavailable / 1× say / 1× pick_up / 1× infant_humanoid_pick_up
Arm C (blank, neutral):                  21× respond / 5× infant_humanoid_look / 1× sense_tools / 1× infant_humanoid_listen / 1× _llm_unavailable
```

**Tool distributions diverge across arms — the cleanest substrate-only contrast is A vs C** (both neutral prompts, only substrate differs). Arm A uses `sense` (3) + `pick_up` variants (3); arm C uses `infant_humanoid_look` (5) + `infant_humanoid_listen` (1) + `sense_tools` (1) with zero `sense`/`pick_up` calls. Arm A's substrate priming biases the LLM proposer toward the same tool family (`sense`, food-source-adjacent) that the explicit "hungry infant" prompt biases arm B toward — substrate carryover acts like a quiet "hungry infant" prompt at test time. **Subtle and prompt-mediated**, not the cluster_reward_bias→action divergence we were hoping for.

**Honest assessment:**

- **Negative for the structural question (path (a) does NOT widen cluster vocabulary).** `cluster_reward_bias_l2 = 2.4495` is byte-identical to Roy-1a; the 6 keys are all `tool:sense_food_source`. Multi-arc priming at the orchestrator level does NOT shift substrate-primary's cold-start AUT proposer. The cold-start gate is in the AUT proposer, not in orchestrator percept emission.
- **Partial-positive for the behavioral question (A vs C tool distributions diverge cleanly).** First clean A-vs-blank-neutral tool-family divergence in the Roy harness under llm-primary. The signal is mediated by the LLM proposer reading substrate context indirectly via salience-modulated WMS / hippocampal recall / affordance hints in the prompt — **not** through the cluster_reward_bias path.
- **First clean valence_KS reading in Roy harness history.** Roy-1a missed α=0.05 (p=0.402); Roy-1b cleared it sample-driven (single-episode blank). Roy-2's `valence_KS = 0.291 (p=0.023)` with 26-episode blank arm vs 685-episode primed arm gives the first methodologically-clean significant valence signal. The substrate priming wrote negative-mean valence (-0.09) onto arm A's episode distribution; held-out percepts on the blank arms fire neutral-mean episodes (0.0). **This is the strongest cross-iteration evidence that substrate carryover propagates to the Hippocampus valence distribution.**
- **Salience_KS effect size shrinks (Roy-1a 0.879 → Roy-2 0.529)** as priming episode diversity grows. P-value is still highly significant (6.9e-15). Diversity is a quantifiable knob on salience carryover — more diverse priming = smaller novelty gap.
- **Cluster wire reproduces fourth iteration in a row.** Roy-0 → Roy-1a → Roy-1b → Roy-2: `cluster_reward_bias_l2` within 1% (2.4587 → 2.4495 → 2.4678 → 2.4495).

**What Roy-2 definitively proves:**

- Multi-arc priming at the orchestrator level does NOT shift substrate-primary AUT's cold-start cluster monoculture (path (a) of Roy-1b's three-pointer refinement is ruled out for this turn budget).
- Substrate carryover produces a subtle but clean tool-family divergence between primed-neutral and blank-neutral arms under llm-primary at test, mediated by the LLM proposer reading substrate context indirectly.
- Hippocampus valence_KS clears α=0.05 with healthy sample size — first methodologically-clean valence reading in the Roy harness.
- The cluster wire is rock-solid across four single-seed iterations on two different priming arc configurations.

**What Roy-2 still does NOT prove:**

- Behavioral expression via the `cluster_reward_bias` path (still zero `sense_food_source` calls at test under llm-primary).
- That Roy-2b (substrate-primary at test on same multi-arc priming) would exploit cradle-stage causal links / ATL chunks the cluster wire doesn't capture.
- `min_confidence` tuning impact (untested; would require Roy-2c).
- Wire 1 sufficiency (untested).
- Cross-session persistence (still single-session).

**What Roy-2 changes for next-iteration methodology:**

1. **The cold-start gate is in the AUT proposer, not the orchestrator.** Roy-1b flagged "tune `min_confidence`"; Roy-2 promotes this to "the structural fix for cluster vocabulary widening must be AUT-side". Either drop `min_confidence` to let primed clusters drive selection on weak matches, OR ship Wire 1 so substrate-derived bias is surfaced at the LLM-prompt level where the AUT proposer reads it.
2. **Valence_KS is now a load-bearing methodology metric.** The Roy-1a (p=0.402) → Roy-2 (p=0.023) shift with healthy blank-arm samples (26 episodes) makes valence_KS the cleanest cross-iteration signal that doesn't depend on cluster-wire consumer coupling.
3. **Salience_KS effect size is a diversity knob.** Reportable as a methodology dial — more priming arc diversity = smaller novelty gap. Useful for tuning priming to match expected test-percept regimes.

**Recommendation for next iteration:** **Roy-2b (substrate-primary at test on the same multi-arc priming) should run next.** Two reasons:

1. The cradle stages may have produced richer causal links / ATL chunks (not visible in the cluster wire) that substrate-primary's `recommend_action` could exploit. Roy-1b's negative result was on `cradle_prelinguistic`-only priming; Roy-2b on cradle-mixed priming may show different substrate-primary behavior even with the same cluster wire output. The single-variable change vs Roy-1b is the priming arc mix.
2. Wire 1 is not yet escalated because Roy-2 DID produce a clean A-vs-blank-neutral tool-family divergence — Wire 1's case strengthens if Roy-2b shows the same `sense_food_source` = 0 monoculture as Roy-1b on cradle-mixed priming. Roy-2b is the cheapest answer to the question "does the cradle priming widen the substrate-primary AUT's exploitable representation beyond what the cluster wire captures?"

If Roy-2b shows non-trivial behavioral divergence in arm A under substrate-primary, the cradle arc stages produced exploitable substrate beyond the cluster wire (good signal for arc-diversity-as-methodology-dial). If Roy-2b reproduces Roy-1b's identical-distribution monoculture, Wire 1 becomes the load-bearing prerequisite for behavioral persona expression.

**Artifacts:**
- [`~/.maxim/roy/roy-2/result.json`](/Users/dennyschaedig/.maxim/roy/roy-2/result.json), [`summary.md`](/Users/dennyschaedig/.maxim/roy/roy-2/summary.md).
- LLM trace `/tmp/roy_2_live.jsonl` (23 peer_backend_call events). Run log `/tmp/roy_2_run.log`.
- Outcome doc: [`18_roy_2.md`](../experiments/18_roy_2.md). Protocol: [`18_roy_2_reproduction.md`](../experiments/protocols/18_roy_2_reproduction.md). Spec: [`roy_2_iteration.yaml`](../../scenarios/roy/roy_2_iteration.yaml). Fixture: [`roy_1_holdout.yaml`](../../scenarios/roy/roy_1_holdout.yaml).
<!-- /roy-iteration:roy-2 -->

<!-- roy-iteration:roy-2pc -->
### Roy-2pc: positive-control on engineered-overlap fixture

**Status:** First positive-control iteration. Two-variable diff vs Roy-2 (fixture: engineered food-semantic overlap + test-AUT-mode: substrate-primary). Ran end-to-end against the same healthy leader, 2026-05-13 08:49→09:14 local. **1502.2s wall (~25.0 min)** — close to Roy-1b's 1578s (substrate-primary test arms slow uniformly because the 30s-per-turn timeout dominates).

> Pre-registered diagnostic logic: A > B > C on sense_food_source counts
> → wire IS healthy + exploitable (Roy-1b/Roy-2 inertness was a
> percept-overlap problem); Wire 1 escalation right for general-percept
> persona. A ≈ B ≈ C → wire bug OR min_confidence gate filters even
> primed-cluster-matched proposals (Roy-2c min_confidence tune becomes
> load-bearing). A < C → wire defect.

**Preflight:** clean (after one-shot leader tunnel cold-start retry). `outcome: ok`, `latency_ms: 570.3`, `detail: stage2 HTTP 200`, `source: peer.yml`.

**Priming:** 5/5 stages completed. final_session_id `20260513_085757`. Identical multi-arc shape to Roy-2 (605.5s vs 609.5s — single-seed iterations reproduce priming wall within 1%).

**Arms:**

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260513_085942` | 10 | 301.13 | cancel |
| b | blank | "You are a hungry infant" | `20260513_090443` | 10 | 299.24 | cancel |
| c | blank | neutral | `20260513_090942` | 10 | 295.33 | cancel |

Arm durations tightly clustered (Δ ≈ 6s) — substrate-primary's 30s/turn timeout dominates wall.

**Pairwise substrate divergence:**

| Pair | `reward_bias_l2` | **`cluster_reward_bias_l2`** | (keys) | `causal_link_Δ` | `episodes_Δ` | **`valence_KS`** (p) | **`salience_KS`** (p) |
|---|---|---|---|---|---|---|---|
| **a_vs_b** | 0.0 | **2.4678** | 10 | +155 | +646 | 0.998 (0.006)* | 0.997 (5.6e-5)* |
| **a_vs_c** | 0.0 | **2.4678** | 10 | +155 | +646 | 0.998 (0.006)* | 0.997 (5.6e-5)* |
| b_vs_c | 0.0 | 0.3000 | 4 | 0 | 0 | 0.000 (1.000) | (similar) |

`*` = sample-driven (arms B and C each captured exactly 1 hippocampus episode; KS detects "distributions differ" on a 647-vs-1 sample).

**Cluster-reward top deltas (a_vs_b):** Six `tool:sense_food_source` × +1.0 (priming carryover) + 4× `tool:infant_humanoid_pick_up` × ±0.15 (substrate-primary at test wrote 4 stochastic-cluster pick_up keys). **Identical shape to Roy-1b's substrate-primary-test output.** Zero new `sense_food_source` cluster updates from any arm during the test phase despite the engineered-overlap fixture.

**Test-phase tool distribution (the headline Roy-2pc signal):**

```
Arm A (substrate-primed, neutral):       2× infant_humanoid_pick_up (both FAILED — Missing required input: object)
Arm B (blank, "hungry infant"):          2× infant_humanoid_pick_up (both FAILED — Missing required input: object)
Arm C (blank, neutral):                  2× infant_humanoid_pick_up (both FAILED — Missing required input: object)
```

**All three arms produced the BYTE-IDENTICAL action distribution: 2× FAILED `infant_humanoid_pick_up` with empty params.** Substrate-primary's `recommend_action` fallback chose the same default tool with the same (invalid) params regardless of percept content AND regardless of substrate state. 8-of-10 turns per arm produced zero actions (sub-threshold filtered).

**This is the pre-registered "A ≈ B ≈ C" diagnostic outcome.**

**Honest assessment:**

- **The cluster wire is structurally healthy.** `cluster_reward_bias_l2 = 2.4678` reproduces Roy-1b's substrate-primary-test value exactly. The wire wrote the priming substrate identically AND consumed it at test time (4 new test-phase cluster updates).
- **The cluster_reward_bias path is BEHAVIORALLY INERT even with maximally-overlapping percepts.** Engineering semantic overlap between test percepts and the priming substrate's food/hunger/eating regime did not produce a single additional `sense_food_source` call across any arm. The behavioral inertness Roy-1b first showed reproduces under direct positive-control conditions.
- **Two unfalsified hypotheses for WHY (Roy-2pc cannot disambiguate alone):**
  - **(H1) LinguisticEncoder→EC alignment failure.** The engineered percepts may not pattern-complete onto the priming-acquired EC clusters at recall time — embeddings of "you sense food nearby" may land in a different EC region than the priming substrate's encodings (which came from sensor/drive state + cradle-stage narrator output, NOT explicit "food" tokens). Wire structurally healthy but never consulted on these percepts.
  - **(H2) `min_confidence=0.3` gate filters primed-cluster-matched proposals.** EC pattern completion may be firing correctly, `recommend_action` may be consulting the +1.0 bias, but the resulting proposal confidence falls below the threshold. Wire would express if the gate were lower or removed.
- **Substrate-primary's default fallback is fixture-content-independent.** Three arms × two fixtures (Roy-1b's original holdout + Roy-2pc's engineered overlap) → same 2× FAILED `infant_humanoid_pick_up` distribution. The fallback is hardcoded behavior, not substrate-driven.

**What Roy-2pc definitively proves:**

- The cluster wire reproduces fifth iteration in a row (Roy-0 → Roy-1a → Roy-1b → Roy-2 → Roy-2pc, all within 1% on `cluster_reward_bias_l2`).
- Engineering percept-substrate semantic overlap is INSUFFICIENT to break the structural-vs-behavioral gap under substrate-primary at test.
- Substrate-primary's recommend_action default fallback is invariant under percept content AND substrate state (byte-identical 2× FAILED pick_up across two fixtures × three arms).
- **The empirical case for Wire 1 is now nearly bulletproof.** The cluster_reward_bias path has at least one (possibly two) blocking gates between substrate state and action selection under substrate-primary, AND it isn't read at all under llm-primary. Wire 1's substrate-annotates-LLM-context design surfaces bias at the LLM prompt — bypassing the cluster-wire consumer entirely.

**What Roy-2pc still does NOT prove:**

- Which of H1 / H2 (or both) blocks the cluster-bias→action pathway. Disambiguating requires instrumentation OR Roy-2c (`min_confidence=0.0` probe on same fixture).
- That `recommend_action` is even consulting the priming clusters on the engineered percepts. Verifying requires logging EC cluster activations during test turns.
- Wire 1 sufficiency (untested).
- Cross-session persistence (single-session as before).

**What Roy-2pc changes for next-iteration methodology:**

1. **Wire 1 is now load-bearing.** Five iterations of structural-vs-behavioral gap, including a positive-control with engineered overlap, are the empirical floor for the Wire 1 design ([bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) Stages 0-3).
2. **Roy-2c (`min_confidence=0.0` probe) is the cheap H1-vs-H2 disambiguator.** Same priming as Roy-2 + same engineered fixture as Roy-2pc + flip `min_confidence` via env var. If A > B > C emerges → H2 confirmed; if A ≈ B ≈ C reproduces → H1 confirmed. Cheap because it reuses fixtures and priming.
3. **`recommend_action` is a black box at the JSONL observability level.** Future iterations should instrument it with per-turn events (EC cluster activations, proposal confidence, cluster_reward_bias consulted per proposal). Without these, single-experiment disambiguation between H1 and H2 is structurally impossible.

**Recommendation for next iteration:** **Escalate Wire 1 (`bio_emergent_persona_foundations.md` Stages 0-3) as the load-bearing 1.0 prerequisite for behavioral persona expression.** Roy-2pc is the empirical floor — substrate carryover writes the cluster wire correctly across five iterations, but the wire's behavioral output is blocked at substrate-primary's consumer-side gates regardless of percept overlap. Wire 1 routes around both blocks.

**Secondary:** ship Roy-2c (`min_confidence=0.0` probe) opportunistically alongside Wire 1 work — it's a one-env-var change with a clean diagnostic, useful to know which gate fired even if the Wire 1 work supersedes the answer.

**Artifacts:**
- [`~/.maxim/roy/roy-2pc/result.json`](/Users/dennyschaedig/.maxim/roy/roy-2pc/result.json), [`summary.md`](/Users/dennyschaedig/.maxim/roy/roy-2pc/summary.md).
- LLM trace `/tmp/roy_2pc_live.jsonl`. Run log `/tmp/roy_2pc_run.log`.
- Outcome doc: [`19_roy_2pc.md`](../experiments/19_roy_2pc.md). Protocol: [`19_roy_2pc_reproduction.md`](../experiments/protocols/19_roy_2pc_reproduction.md). Spec: [`roy_2pc_iteration.yaml`](../../scenarios/roy/roy_2pc_iteration.yaml). Fixture: [`roy_2pc_holdout.yaml`](../../scenarios/roy/roy_2pc_holdout.yaml).
<!-- /roy-iteration:roy-2pc -->

<!-- roy-iteration:roy-2c -->
### Roy-2c: `min_confidence=0.0` probe (H1 vs H2 disambiguator)

**Status:** SHIPPED. H1-vs-H2 disambiguator for Roy-2pc's byte-identical pick_up result. Ran end-to-end against the same healthy leader, 2026-05-13 12:27→12:46 local. **1284.2s wall (~21.4 min)** — faster than Roy-2pc's 25 min because lower gate accepts proposals faster (less wall burned on 30s timeout). Owned by [release_0_9_1.md Stage 0a](release_0_9_1.md).

> Single-variable change vs Roy-2pc: `MAXIM_NAC_MIN_CONFIDENCE=0.0` set
> in runner environment (new env var introduced in 0.9.1). Same
> priming, fixture, arms. A > B > C → H2 confirmed (gate was the
> block); A ≈ B ≈ C → H1 confirmed (LinguisticEncoder→EC alignment).

**Preflight:** clean. `outcome: ok`, `latency_ms: 228.4`, `detail: stage2 HTTP 200`, `source: peer.yml`.

**Priming:** 5/5 stages completed. final_session_id `20260513_123258`. Identical multi-arc shape to Roy-2pc.

**Arms:**

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260513_123436` | 10 | 224.93 | cancel |
| b | blank | "You are a hungry infant" | `20260513_123821` | 10 | 241.82 | cancel |
| c | blank | neutral | `20260513_124223` | 10 | 238.57 | cancel |

**Pairwise substrate divergence:**

| Pair | `reward_bias_l2` | **`cluster_reward_bias_l2`** | (keys) | `causal_link_Δ` | `episodes_Δ` | `valence_KS` (p) | `salience_KS` (p) |
|---|---|---|---|---|---|---|---|
| **a_vs_b** | 0.0 | **2.5661** | 10 | +147 | +664 | 0.994 (1.7e-8)* | 0.826 (6.3e-5) |
| **a_vs_c** | 0.0 | **2.5661** | 10 | +147 | +664 | 0.994 (1.7e-8)* | (similar) |
| b_vs_c | 0.0 | 0.7649 | 4 | 0 | 0 | 0.000 (1.000) | (similar) |

`*` Sample asymmetry partially relaxed (4 vs 668 instead of Roy-2pc's 1 vs 647), but blank-arm distribution is still a 4-point spike at valence=-1.0 (replicated single-failure-mode events).

**Cluster-reward top deltas (a_vs_b):** 6× `tool:sense_food_source` × +1.0 (priming carryover, UNCHANGED — priming clusters never touched during test) + 4× `tool:infant_humanoid_pick_up` × {±0.30, ±0.45} (test-phase updates on FOUR NEW EC clusters, **disjoint** from priming's six).

**Test-phase tool distribution (the headline Roy-2c signal):**

```
Arm A (substrate-primed, neutral):       5× infant_humanoid_pick_up (all FAILED — Missing required input: object)
Arm B (blank, "hungry infant"):          5× infant_humanoid_pick_up (all FAILED — Missing required input: object)
Arm C (blank, neutral):                  5× infant_humanoid_pick_up (all FAILED — Missing required input: object)
```

**Per-arm action count rose 2 → 5 from Roy-2pc, but the tool family is unchanged.** Zero `sense_food_source` calls in any arm under gate=0.0. The gate WAS active (3 additional pick_up proposals per arm crossed the relaxed threshold), but no `sense_food_source` proposal ever crossed even at 0.0.

**H1 confirmed — three independent observables:**

1. Per-arm action count rose 2 → 5 → gate WAS filtering proposals in Roy-2pc.
2. Newly-accepted proposals are still `infant_humanoid_pick_up`, not `sense_food_source` → recommend_action is not generating `sense_food_source` proposals on these percepts at all.
3. Test-phase EC clusters are structurally **disjoint** from priming clusters (6 priming at unchanged +1.0; 4 new pick_up clusters at ±0.30/±0.45). If pattern completion had hit priming clusters, the priming +1.0 entries would shift; they don't.

**The structural diagnosis:** LinguisticEncoder embeds priming-substrate WMS contents (sensor/drive state + cradle narrator output) into one EC region. Engineered CLI percepts ("you sense food nearby") embed into a DIFFERENT region, even though humans read the semantic overlap as obvious. The cluster_reward_bias map has the right *tool* keys (`sense_food_source`) but the wrong *cluster* keys.

**H2 cleanly refuted:** zero `sense_food_source` calls under gate=0.0.

**What this means for 0.9.1:** Wire-A's design is **revised** per the finding. Original spec used active-cluster-restricted aggregation (`get_active_cluster_biases(cluster_ids=...)`); Roy-2c shows the active-cluster intersection with priming clusters is empty in the failure mode Wire-A is designed to fix. Revised to **agent-wide aggregation** (`get_agent_tool_biases`) — the priming substrate's tool-level signal ("this agent has experienced strong reward on `sense_food_source`") survives the encoder-alignment gap; the cluster-level signal does not. Wire-A surfaces the surviving granularity. See [release_0_9_1.md Stage 2](release_0_9_1.md).

**What Roy-2c definitively proves:**

- The `min_confidence=0.3` gate WAS filtering proposals in Roy-2pc (per-arm action count rose 2 → 5).
- Lowering the gate does NOT rescue the cluster wire on engineered-overlap test percepts.
- Priming-acquired EC clusters and test-phase EC clusters are structurally disjoint under LinguisticEncoder embedding.
- `MAXIM_NAC_MIN_CONFIDENCE` env-var override works end-to-end.
- The cluster wire reproduces SIXTH iteration in a row.

**Recommendation:** **0.9.1 plan proceeds unchanged on critical path** with Wire-A's design revision folded in. No further Roy-2 sub-iterations planned. Roy-3 (post-wires, Stage 5 of 0.9.1) is the next harness iteration.

**Artifacts:**
- [`~/.maxim/roy/roy-2c/result.json`](/Users/dennyschaedig/.maxim/roy/roy-2c/result.json), [`summary.md`](/Users/dennyschaedig/.maxim/roy/roy-2c/summary.md).
- LLM trace `/tmp/roy_2c_live.jsonl`. Run log `/tmp/roy_2c_run.log`.
- Outcome doc: [`20_roy_2c.md`](../experiments/20_roy_2c.md). Protocol: [`20_roy_2c_reproduction.md`](../experiments/protocols/20_roy_2c_reproduction.md). Spec: [`roy_2c_iteration.yaml`](../../scenarios/roy/roy_2c_iteration.yaml). Fixture: [`roy_2pc_holdout.yaml`](../../scenarios/roy/roy_2pc_holdout.yaml) (reused unchanged).
<!-- /roy-iteration:roy-2c -->

<!-- roy-iteration:roy-3a -->
### Roy-3a: 0.9.1 annotation-pattern validation on original holdout

**Status:** SHIPPED. First post-all-wires iteration. Wires A+1+2+3 (PRs #253 / #254 / #255 / #256 / #257) all active. Multi-arc priming identical to Roy-2 / Roy-2c / Roy-4 / Roy-5a; llm-primary at test against `roy_1_holdout.yaml`. Ran end-to-end against the same healthy leader, 2026-05-23 11:37→11:53 local. **952.5s wall (~15.9 min)** — +8% vs Roy-2's 883s (annotation rendering adds modest per-prompt overhead). Owned by [release_0_9_1.md Stage 5](release_0_9_1.md).

> Pre-registered diagnostic logic: Arm A > C on tool-family divergence
> (richer than Roy-2's 17/3/2 vs 21/5/1/1) → Wire 1 + Wire 2 compound
> the salience-mediated signal Roy-2 surfaced. Arm A ≈ B ≈ C →
> annotation pattern alone insufficient; investigate prompt rendering
> / priming-side regressions.

**Preflight:** clean. `outcome: ok`, `latency_ms: 353.6`, `detail: stage2 HTTP 200`, `source: peer.yml`.

**Priming:** 5/5 stages completed. final_session_id `20260523_114722`. Identical multi-arc shape to Roy-2 / Roy-2c.

**Arms:**

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260523_114915` | 10 | 119.5 | cancel |
| b | blank | "You are a hungry infant" | `20260523_115114` | 10 | 90.7 | cancel |
| c | blank | neutral | `20260523_115245` | 10 | 97.5 | cancel |

**Pairwise substrate divergence:**

| Pair | `reward_bias_l2` | `cluster_reward_bias_l2` | (keys) | `causal_link_Δ` | `episodes_Δ` | `valence_KS` (p) | ATL jaccard |
|---|---|---|---|---|---|---|---|
| a_vs_b | 0.0 | **0.0370** | 2 | +340 | +702 | 0.294 (0.077) | 0.047 |
| a_vs_c | 0.0 | **0.0370** | 2 | +330 | +694 | 0.294 (0.020) | 0.054 |
| b_vs_c | 0.0 | 0.0 | 0 | −10 | −8 | 0.000 (1.000) | 0.652 |

**Priming-side cluster_reward_bias (the structural anomaly):** priming session NAc holds **2 entries** at `sense_food_source` × {+0.18, +0.98} — vs Roy-2 / Roy-2c / Roy-4 / Roy-5a all writing **6 entries at +1.0**. Same multi-arc priming, same 50-turn budget, same substrate-primary AUT mode — structurally fewer + weaker cluster keys post-2026-05-22. The five wire merges in the 5/13→5/22 window correlate with this regression.

**Test-phase tool distribution (the headline Roy-3a behavioral signal):**

```
Arm A (substrate-primed, neutral):    46× respond, 4× _llm_unavailable, 2× pick_up,
                                       1× sense, 1× sense_tools, 1× infant_humanoid_listen, 1× <no-tool>
Arm B (blank, "hungry infant"):       11× respond, 3× sense, 2× pick_up,
                                       1× examine, 1× say, 1× sense_presence, 1× <no-tool>
Arm C (blank, neutral):               21× respond, 3× sense, 2× pick_up,
                                       2× _llm_unavailable, 1× sense_tools, 1× <no-tool>
```

**Zero `sense_food_source` calls in any arm.** Roy-2's clean A-vs-C divergence (A used `sense`/`pick_up` family, C used `look`/`listen`) is *weakened*: both arms now run on the same `sense`/`pick_up` family. Arm A's `respond` count rose 17 → 46, shifting toward narrative-verbal responses rather than action verbs.

**Wire-A annotation render at arm A:** `max(|bias|)` = 0.036 after test-phase decay → renders as `sense_food_source [neutral / mixed]`. The annotation IS attached to the StructuredContext at LLM-submission time (Wire-A producer site runs unconditionally with `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION` unset), but the rendered band conveys no actionable signal.

**JSONL emission counts:** 974× `NAc_RECOMMEND` total (811 passed_gate=True best_tool=`sense_food_source` from substrate-primary priming; 163 sub-threshold). 0× `WIRE_3_FILTER` (intact body, as expected). 292× linguistic + 83× drive EC_TRACE — text-modality routing confirmed active.

**Honest assessment:**

- The annotation surface is wired end-to-end at the producer + composer level.
- The annotation's signal value is null at test time because the priming-side cluster bias decays below the "mildly rewarding" 0.1 floor during the 10-turn test phase.
- Roy-2's salience-mediated A-vs-C divergence is weakened under annotation overhead, not augmented.
- `valence_KS` (a_vs_c) cleanly reproduces below α=0.05 (p=0.020) — the valence carryover signal Roy-2 first established remains observable.

**What Roy-3a definitively proves:**

- Identical multi-arc priming + identical substrate-primary AUT + identical turn budget across Roy-2 → Roy-3a produces structurally different `cluster_reward_bias` outputs (6 saturated → 2 partially-saturated). Single-seed cluster wire output was stable across Roy-2 / Roy-2c / Roy-4 / Roy-5a and changed only after 2026-05-22 wire merges.
- The Stage 0c `sim_recommend_action` JSONL emissions fire on every `recommend_action` call (974 events including sub-threshold), as designed.
- Wire 3's `WIRE_3_FILTER` emission surface is present and silent under an intact body.

**What Roy-3a still does NOT prove:**

- That the priming-side regression is harmful in production. The two clusters (0.18 / 0.98) still carry usable substrate-acquired bias during the priming session; the harm is specifically on Wire-A's read window at llm-primary test time.
- Which of Wire 1 / Wire 2 / interaction caused the regression. Determining this requires bisecting the 5/13→5/22 PR window.
- Persona convergence at any level (single-session as before).

**Artifacts:**
- [`~/.maxim/roy/roy-3a/result.json`](/Users/dennyschaedig/.maxim/roy/roy-3a/result.json), [`summary.md`](/Users/dennyschaedig/.maxim/roy/roy-3a/summary.md).
- JSONL trace `/tmp/roy_3a_ec_trace.jsonl` (7.9 MB). Run log `/tmp/roy_3a_run.log`.
- Outcome doc: [`23_roy_3.md`](../experiments/23_roy_3.md) (covers 3a + 3b). Protocol: [`23_roy_3_reproduction.md`](../experiments/protocols/23_roy_3_reproduction.md). Spec: [`roy_3a_iteration.yaml`](../../scenarios/roy/roy_3a_iteration.yaml). Fixture: [`roy_1_holdout.yaml`](../../scenarios/roy/roy_1_holdout.yaml).

**Roy-3a-retry (2026-05-25):** [`30_wire_a_tau_validation.md`](../experiments/30_wire_a_tau_validation.md). Re-ran Roy-3a unchanged with `cluster_reward_bias_decay_tau=300.0` (PR #267). Wire-A's annotation rendered `[strongly rewarding]` (max\|bias\| 0.753-0.997) throughout the test arm — tau split structurally validated. PRIMARY criterion (Arm A ≥1 `sense_food_source`) still failed because of two downstream gaps:
1. SEM-derived tool absent from active scene roster (no food entity in Roy-1 holdout). New plan: [`sense_tool_registry.md`](sense_tool_registry.md).
2. Imagination is substrate-blind — can't dream up the missing entity from Wire-A signals. New plan: [`imagination_substrate_signals.md`](imagination_substrate_signals.md).

**Methodology note (Wire-A tunability):** Wire-A's `cluster_reward_bias_decay_tau` is now a tunable parameter (env var `MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU`, clamped `[50, 1000]`). The next persona iteration owns the sweep ownership if 300 proves brittle on a different test-arm length. The Roy-3a-retry validated 300 produces strongly-rewarding renders across a 10-turn arm; longer arms (e.g. 30-turn) may want a higher default.
<!-- /roy-iteration:roy-3a -->

<!-- roy-iteration:roy-3b -->
### Roy-3b: 0.9.1 annotation-pattern validation on engineered overlap

**Status:** SHIPPED. Wire-A-specific behavioral test on the engineered overlap fixture. Same wires + priming as Roy-3a; llm-primary at test against `roy_2pc_holdout.yaml` (every percept evokes food / hunger / eating semantics). Ran end-to-end against the same healthy leader, 2026-05-23 12:25→12:39 local. **879.9s wall (~14.7 min)** — close to Roy-3a's 953s (same shape, slightly faster). Owned by [release_0_9_1.md Stage 5](release_0_9_1.md).

> Pre-registered diagnostic logic: Arm A `sense_food_source` count >
> Arm B AND > Arm C → Wire-A annotation reached LLM proposer's
> decision pathway on engineered overlap; pattern works.  A ≈ B ≈ C
> → annotation didn't reach the proposer's decision pathway;
> investigate prompt rendering / priming-side regressions.

**Preflight:** clean. `outcome: ok`, `latency_ms: 293.0`, `detail: stage2 HTTP 200`, `source: peer.yml`.

**Priming:** 5/5 stages completed. final_session_id `20260523_123334`. Identical multi-arc shape to Roy-3a.

**Arms:**

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260523_123525` | 10 | 108.5 | cancel |
| b | blank | "You are a hungry infant" | `20260523_123713` | 10 | 82.6 | cancel |
| c | blank | neutral | `20260523_123836` | 10 | 82.9 | cancel |

**Pairwise substrate divergence:**

| Pair | `reward_bias_l2` | `cluster_reward_bias_l2` | (keys) | `causal_link_Δ` | `episodes_Δ` | `valence_KS` (p) | ATL jaccard |
|---|---|---|---|---|---|---|---|
| a_vs_b | 0.0 | **0.1002** | 2 | +302 | +685 | 0.117 (0.998) | 0.047 |
| a_vs_c | 0.0 | **0.1002** | 2 | +303 | +685 | 0.168 (0.928) | 0.043 |
| b_vs_c | 0.0 | 0.0 | 0 | +1 | +0 | 0.111 (1.000) | 0.667 |

**Priming-side cluster_reward_bias:** 2 entries `sense_food_source` × {+0.21, +0.98}. Reproduces the Roy-3a regression — same shape, slightly higher bias on the smaller cluster (+0.21 vs +0.18). The regression is iteration-stable post-wires.

**Test-phase tool distribution (the headline Roy-3b behavioral signal):**

```
Arm A (substrate-primed, neutral):    28× respond, 3× use, 2× pick_up, 2× _llm_unavailable,
                                       1× infant_humanoid_use, 1× sense_presence, 1× <no-tool>
Arm B (blank, "hungry infant"):        4× pick_up, 2× use, 2× infant_humanoid_use,
                                       1× sense_tools, 1× respond, 1× _llm_unavailable, 1× <no-tool>
Arm C (blank, neutral):                4× examine, 3× pick_up, 2× eat,
                                       1× respond, 1× _llm_unavailable, 1× <no-tool>
```

**Zero `sense_food_source` calls in any arm.** This is the pre-registered "A ≈ B ≈ C" outcome — Wire-A annotation did NOT drive arm A toward `sense_food_source` on food-themed percepts. Notable cross-arm differences: arm A dominated by `respond` (28×, narrative-verbal); arm B dominated by `pick_up` + `use` (task-oriented action under the persona prompt); arm C is the *most food-engaged* arm with 2× `eat` affordance calls — the opposite direction Wire-A was designed to produce.

**Wire-A annotation render at arm A:** `max(|bias|)` = 0.098 after test-phase decay → renders as `sense_food_source [neutral / mixed]` (still below the 0.1 "mildly rewarding" floor by 0.002). Same null-signal pattern as Roy-3a.

**JSONL emission counts:** 963× `NAc_RECOMMEND` total (800 passed_gate=True best_tool=`sense_food_source` from substrate-primary priming; 163 sub-threshold). 0× `WIRE_3_FILTER`. 186× linguistic + 81× drive EC_TRACE.

**Honest assessment:**

- The annotation-pattern thesis (substrate-annotates-LLM-context surfaces signal the cluster-wire consumer cannot) is **not directly falsified** — the test conditions never gave the LLM a meaningfully-non-null annotation to bias on.
- Roy-2pc's null behavioral signal on the engineered overlap fixture reproduces under llm-primary + annotation. The fixture isn't fixable from the LLM-proposer side when the annotation conveys no signal.
- `valence_KS` p-values for a_vs_b (0.998) and a_vs_c (0.928) are far above α=0.05 — the valence-distribution divergence Roy-2 surfaced (p=0.023) is GONE under the annotation overhead on this fixture. The wires shipped close architectural gaps but appear to weaken the salience-mediated signal that was Roy-2's most legible cross-arm divergence.

**What Roy-3b definitively proves:**

- The substrate-primary priming proposer fires `sense_food_source` proposals during priming (800 passed_gate=True), correctly landing on cluster keys for that tool. The wire is healthy at the priming-side consumer level.
- Engineering test-percept semantic overlap is INSUFFICIENT to drive `sense_food_source` selection under llm-primary + annotation, when the annotation conveys a null signal.
- Arm C's `eat` affordance calls (2×) on the engineered overlap fixture demonstrate the LLM can read food-themed percepts and choose food-aware actions independently of substrate priming.

**What Roy-3b still does NOT prove:**

- Whether Wire-A would produce cross-arm divergence if the annotation rendered "strongly rewarding" instead of "neutral / mixed". Requires a Roy iteration where the priming substrate saturates above 0.5 OR a Wire-A design tweak that defers to the priming session-end snapshot rather than the post-decay arm-A snapshot.
- Cross-session persistence.

**What Roy-3b changes for next-iteration methodology:**

1. **Bisect the priming-side `cluster_reward_bias` regression** (Wire 1 vs Wire 2 vs interaction). Roy-2-shaped iteration per intermediate commit between 5/13 and 5/22. Required diagnostic work before 1.0 ships.
2. **Decide whether Wire-A's render needs a session-end raw priming snapshot floor** rather than the post-decay snapshot. Trade-off: persistent bias vs cleanly extinguishing aversions. Worth deciding before 1.0.

**Recommendation:** **0.9.1 ships unchanged** per the [foundations doc framing rule](bio_emergent_persona_foundations.md) — the wires close architectural gaps regardless of persona behavior. The four wires are now load-bearing surface area for 1.1+ work that depends on substrate-annotates-LLM-context infrastructure. Roy-3 is a clean negative result on the annotation-pattern's behavioral expression for persona convergence; the diagnostic correctly decodes as "the test conditions did not give the LLM a non-null annotation to bias on", not "the wires are wrong".

**Artifacts:**
- [`~/.maxim/roy/roy-3b/result.json`](/Users/dennyschaedig/.maxim/roy/roy-3b/result.json), [`summary.md`](/Users/dennyschaedig/.maxim/roy/roy-3b/summary.md).
- JSONL trace `/tmp/roy_3b_ec_trace.jsonl` (7.4 MB). Run log `/tmp/roy_3b_run.log`.
- Outcome doc: [`23_roy_3.md`](../experiments/23_roy_3.md) (covers 3a + 3b). Protocol: [`23_roy_3_reproduction.md`](../experiments/protocols/23_roy_3_reproduction.md). Spec: [`roy_3b_iteration.yaml`](../../scenarios/roy/roy_3b_iteration.yaml). Fixture: [`roy_2pc_holdout.yaml`](../../scenarios/roy/roy_2pc_holdout.yaml) (reused unchanged from Roy-2pc / Roy-2c / Roy-4 / Roy-5a).
<!-- /roy-iteration:roy-3b -->

<!-- roy-iteration:roy-4 -->
### Roy-4: EC-activation co-activation instrumentation (1.1 binding plan gate)

**Status:** SHIPPED. Cross-modal-binding validation prereq for [cross_modal_substrate_binding.md](cross_modal_substrate_binding.md) Stage 1. Ran end-to-end against the same healthy leader, 2026-05-13 18:00→18:26 local. **1547.5s wall (~25.8 min)** — same shape as Roy-2c since the substrate-primary 30s/turn timeout dominates. Owned by [release_0_9_1.md Stage 0d](release_0_9_1.md).

> Single-variable change vs Roy-2c: `MAXIM_EC_TRACE_ACTIVATIONS=1` set
> in the runner environment (new env var introduced in 0.9.1 Stage 0d).
> Same priming, fixture, arms. Pre-registered diagnostic: at least one
> test-phase active node has a would-have-bound edge to a priming
> `sense_food_source` cluster → PASS, greenlight 1.1 Stages 2-6; no
> would-have-bound edges between priming and test clusters → FAIL,
> cancel binding plan and redirect to 1.2+ encoder replacement.

**Preflight:** clean. `outcome: ok`, `latency_ms: 235.9`, `detail: stage2 HTTP 200`.

**Priming:** 5/5 stages completed. final_session_id `20260513_180901`.

**Arms:**

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260513_181049` | 10 | 307.38 | cancel |
| b | blank | "You are a hungry infant" | `20260513_181557` | 10 | 321.87 | cancel |
| c | blank | neutral | `20260513_182119` | 10 | 307.24 | cancel |

**Pairwise substrate divergence (cluster wire reproduces SEVENTH iteration in a row):**

| Pair | `cluster_reward_bias_l2` | (keys) | `causal_link_Δ` | `episodes_Δ` | `valence_KS` (p) |
|---|---|---|---|---|---|
| **a_vs_b** | **2.4678** | 10 | +151 | +650 | 0.998 (0.006) |
| **a_vs_c** | **2.4678** | 10 | +151 | +650 | 0.998 (0.006) |
| b_vs_c | 0.3000 | 4 | 0 | 0 | 0.000 (1.000) |

**EC instrumentation capture:** 306 `sim_ec_activation` events total — 148 priming (5 sessions × ~30 events each) + 47 arm A + 69 arm B + 42 arm C. Both linguistic (LinguisticEncoder) and drive (SensorEncoder) modalities fire in every phase.

**Node-set overlap (the load-bearing structural finding):**

| Phase | Unique nodes | Overlap with priming | Overlap with 6 priming food clusters |
|---|---|---|---|
| Priming | 37 | — | 6 / 6 |
| Arm A | 10 | **0** | **0** |
| Arm B | 13 | **0** | **0** |
| Arm C | 9 | **0** | **0** |

**Zero EC node ID is shared between priming and any test arm.** Roy-2c inferred this from cluster-key differences in NAc's `cluster_reward_bias`; Roy-4 confirms it at the per-tick EC instrumentation level. The priming substrate's EC region (37 nodes) and the test-phase EC regions (10/13/9 nodes per arm) are structurally separate populations.

**Food-cluster co-firing analysis:** 61 ticks during priming where any food cluster fired; only **1** of those 61 ticks had a non-food node co-firing in the same tick window. Of the 7 unique non-food co-firing partners observed during priming, **zero** appear in arm A's test-phase active set. **The temporal-coincidence signal the Hebbian binding rule depends on does not exist in the priming trajectory.**

**Parameter sweep — FAIL across the entire reasonable range:**

| `min_cofire` | `min_weight` | Priming would-have-bound edges | Matching priming↔test edges |
|---|---|---|---|
| 1 | 0.01 | 256 | **0** |
| 1 | 0.1 | 256 | **0** |
| 2 | 0.01 | 5 | **0** |
| 3 | 0.01 | 3 | **0** |
| 5 (default) | 0.5 (default) | 2 | **0** |

The most permissive rule (`min_cofire=1, min_weight=0.01`) yields 256 priming bound edges; **at every sweep point, zero of those edges connect a priming food cluster to a test-phase active node.** No reasonable parameter tuning rescues the binding hypothesis.

**Pre-registered FAIL outcome confirmed.** Per [cross_modal_substrate_binding.md § Risk register](cross_modal_substrate_binding.md):

> **Roy-4 fails (pairs don't co-fire even at instrumentation level)** — Cancel Stage 2. The deeper fix is replacing LinguisticEncoder with an aligned multimodal encoder — a 1.2+ research direction. Roy-4 is the cheap gate that prevents this misallocation.

**What Roy-4 definitively proves:**

- The instrumentation hook fires end-to-end across all 8 sessions of a Roy iteration in both pattern-completion and pattern-separation branches.
- Roy-2c's cluster-disjointness finding reproduces at per-tick EC resolution.
- Zero priming-cluster ↔ test-cluster edges form under any reasonable Hebbian binding rule. The rule cannot close the Roy-2c gap on this priming + fixture pair.
- Both linguistic and drive modality channels reproduce the disjointness — not an artifact of channel mismatch.

**Recommendation:** **Cancel [cross_modal_substrate_binding.md](cross_modal_substrate_binding.md) Stages 2-6.** Archive the plan with a "superseded by Roy-4 FAIL" note. Promote encoder replacement to the 1.2+ research-direction priority (the natural carrier is [grounded_language_acquisition.md](grounded_language_acquisition.md), whose Phase 1 was originally scoped to consume binding edges this experiment was meant to validate). Keep the Roy-4 instrumentation surface (`MAXIM_EC_TRACE_ACTIVATIONS` + analyzer) — it's reusable for future substrate-dynamics characterization at zero runtime cost when off. **Roy-5 is cancelled** (it was the post-implementation validation for the binding mechanism that's now off the table). No 0.9.1 plan changes required — Wire-A is unaffected and Roy-3 remains the next harness iteration.

**Artifacts:**
- [`~/.maxim/roy/roy-4/result.json`](/Users/dennyschaedig/.maxim/roy/roy-4/result.json), [`summary.md`](/Users/dennyschaedig/.maxim/roy/roy-4/summary.md).
- EC trace `/tmp/roy_4_ec_trace.jsonl` (306 `sim_ec_activation` events). Run log `/tmp/roy_4_run.log`. Per-session partitions `/tmp/roy_4_{priming,arm_a,arm_b,arm_c}_ec.jsonl`. Analysis bundle `/tmp/roy_4_analysis.json`.
- Outcome doc: [`21_roy_4.md`](../experiments/21_roy_4.md). Protocol: [`21_roy_4_reproduction.md`](../experiments/protocols/21_roy_4_reproduction.md). Spec: [`roy_4_iteration.yaml`](../../scenarios/roy/roy_4_iteration.yaml). Fixture: [`roy_2pc_holdout.yaml`](../../scenarios/roy/roy_2pc_holdout.yaml) (reused unchanged). Analyzer: [`scripts/analyze_roy_4_coactivation.py`](../../scripts/analyze_roy_4_coactivation.py).
<!-- /roy-iteration:roy-4 -->

### Roy-5 — partially RECYCLED
*The original Roy-5 (binding-mechanism post-implementation validation) is cancelled by Roy-4 FAIL. The slot is now occupied by the [roy_5_encoder_alignment_disambiguator.md](roy_5_encoder_alignment_disambiguator.md) plan's Stage 1 (Roy-5a, below). Future Roy-5b/Roy-5c iterations on the H1c / H1b / H1a Stage 2 branches will inherit the same numbering family.*

<!-- roy-iteration:roy-5a -->
### Roy-5a: Cosine-localization disambiguator (1.1+ plan Stage 1)

**Status:** SHIPPED. Stage 1 of [roy_5_encoder_alignment_disambiguator.md](roy_5_encoder_alignment_disambiguator.md). Ran end-to-end against the same healthy leader, 2026-05-13 22:25→22:51 local. **~1547s wall (~25.8 min)** — same shape as Roy-2c / Roy-4 since the substrate-primary 30s/turn timeout dominates. Owned by [roy_5_encoder_alignment_disambiguator.md § Stage 1](roy_5_encoder_alignment_disambiguator.md).

> Single-variable change vs Roy-4: [PR #248](https://github.com/dennys246/Maxim/pull/248)
> (merged before this run) wired `EC.save()` and `ATL.save()` into
> `simulation/report.py::save_aut_state`. Every session_dir now
> contains `aut_ec.json` + `aut_atl.json` alongside the existing
> hippocampus + NAc dumps. The new analyzer
> [`scripts/analyze_roy_5_cosine_localization.py`](../../scripts/analyze_roy_5_cosine_localization.py)
> reads those centroid dumps directly, computes pairwise cosine
> matrices `M_tt` (priming text × arm text), `M_dt` (priming
> interoception × arm text), `M_dd` (priming interoception × arm
> interoception), identifies food-bearing priming centroids via the
> same UTS-separator NAc compound-key parsing the Roy-4 analyzer
> uses, and decodes max cosine over arm A into one of three
> pre-registered sub-hypotheses (H1c ≥ 0.40 / 0.20 ≤ H1b < 0.40 /
> H1a < 0.20).

**Preflight:** clean. `outcome: ok`, `latency_ms: 211.2`, `detail: stage2 HTTP 200`.

**Priming:** 5/5 stages completed. final_session_id `20260513_223436`.

**Arms:**

| Arm | Substrate | system_prompt | session_id | turns | finish |
|---|---|---|---|---|---|
| a | from_priming | neutral | `20260513_223616` | 10 | cancel |
| b | blank | "You are a hungry infant" | `20260513_224138` | 10 | cancel |
| c | blank | neutral | `20260513_224641` | 10 | cancel |

**Headline cosine matrices — arm A:**

| Matrix | Modality pair | Rows × Cols | Max over food-bearing rows |
|---|---|---|---|
| **`M_tt`** | priming text × arm A text | **0 × 0** | **n/a (no text-modality nodes on either side)** |
| **`M_dt`** | priming interoception × arm A text | 2 × 0 | n/a (arm A has zero text-modality nodes) |
| **`M_dd`** | priming interoception × arm A interoception | 2 × 2 | **1.0000 (identical centroids)** |

**Verdict: H1a — encoder subspace incompatibility** (confirmed across three runs, two distinct mechanisms). The initial Roy-5a run triggered via "no text-modality nodes exist" (`MAXIM_SUBSTRATE_PATH` unset). Roy-5a-confirm reproduced. Roy-5a-substrate-on (with `MAXIM_SUBSTRATE_PATH=1` explicitly set) produced 162 text fires + 14 text-modality EC centroids — but **zero of them are food-bearing**; food NAc cluster IDs remain exclusively interoception-modality. **Plus a stronger structural finding:** `SensorEncoder` produces 384-dim embeddings, `LinguisticEncoder` produces 768-dim — cross-modality cosine (M_dt) is mathematically undefined regardless of substrate-path state. The plan's "encoder subspaces are far in cosine space" framing is weaker than the data shows: they're **different-dimensional**, not far. The dimension-mismatch warning added in the pre-merge review fold fired on real Roy-5a-substrate-on data.

**Cross-arm M_dd sanity check** (the surviving "interoception identity" scheme):

| Arm | M_dd max food cosine | Cluster IDs match priming? |
|---|---|---|
| **a** (substrate-primed) | **1.0000** | Yes (substrate restored — same UUIDs) |
| **b** (blank, persona) | 1.0000 | No (fresh UUIDs, but cosine ≈ 1.0 — SensorEncoder frozen-prototype hash embeddings collide across blank substrates) |
| **c** (blank, neutral) | 1.0000 | No (same as B) |

This is the **"two identity schemes for the same concept"** pattern from [`feedback_two_identity_schemes.md`](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_two_identity_schemes.md) re-confirmed for the EIGHTH iteration: tool-name + interoception-modality identity survives across all arms, cluster-UUID identity does not.

**EC trace divergence vs Roy-4 (secondary finding):**

| Iteration | Total EC trace events | Per-modality fires | Per-modality NEW |
|---|---|---|---|
| Roy-4 (priming) | 306 | text=154, interoception=152 | text=57, interoception=12 |
| **Roy-5a (priming)** | **151** | **interoception=151, text=0** | **interoception=6, text=0** |

**Roy-5a's priming registered ZERO text-modality EC fires; Roy-4's had 154.** Same iteration spec, same fixture, same arms. `git log c80190f..HEAD` shows no encoder-routing changes between the two runs. This is either run-to-run variance (Roy-5a's first stage had many AUT timeouts that may have suppressed narrator text routing) or a quiet regression in the text-percept routing path. Roy-4's `aut_hippocampus.json` ALSO had zero `cli_input`/`transcript` content, so Roy-4's text fires came through a side-channel route (tool-output text, concept decomposition, or similar) that's apparently silent in Roy-5a. **The H1a verdict survives either interpretation** — it triggers on the absence of text-modality food clusters in persisted state, observable directly from `aut_ec.json` regardless of transient trace events.

**Recommended next step** (gates Stage 3 cradle-arc redesign):

1. Re-run Roy-5a once or twice to confirm the text-modality-silence finding is stable.
2. If reproducible, inspect why cradle-narrator text isn't routing to `LinguisticEncoder` on food-related percepts (orchestrator narrator-percept routing, `LinguisticEncoder.encode` text extraction, `EmbodimentPerceptSource` percept field population).
3. Stage 3 implementation (the H1a branch in the plan) needs to ensure not only deliberate `(sensor, drive, narrator-utterance)` co-firing but also that the narrator utterance actually fires text-modality EC nodes the Hebbian rule can bind to.

If text-modality routing turns out to be quietly broken (rather than variance), Stage 3 will need to fix that as part of the redesigned arc.

**What Roy-5a definitively proves:**

- The cluster-reward-bias-vs-cluster-identity gap that's haunted every Roy iteration since Roy-0 has its food concept located strictly in **interoception modality** during priming. There is no text-modality representation of food in the priming substrate at all.
- The `M_dd` cosine ≈ 1.0 result is the strongest positive signal: the food concept's interoception embedding **does survive arm A's substrate-restoration**. The gap is strictly in projecting that concept to text modality (which is the substrate channel CLI fixture text routes through).
- The pre-registered H1a/H1b/H1c thresholds are stable + tested. Future Stage 2 branches consume a clean verdict.

**Recommendation:** **Stage 2c → Stage 3 (cradle-arc redesign) is the verdict-prescribed next step**, but ship Roy-5a-confirm (≤ 2 re-runs) first to validate the text-modality-silence observation. If silence reproduces, refine Stage 3 scope to fix text-modality routing alongside the co-firing scaffold.

**Artifacts:**
- [`~/.maxim/roy/roy-5a/result.json`](/Users/dennyschaedig/.maxim/roy/roy-5a/result.json), [`summary.md`](/Users/dennyschaedig/.maxim/roy/roy-5a/summary.md).
- EC trace `/tmp/roy_5a_ec_trace.jsonl` (151 `sim_ec_activation` events, all interoception). Run log `/tmp/roy_5a_run.log`. Analysis bundle `/tmp/roy_5a_analysis.json`.
- Outcome doc: [`22_roy_5a.md`](../experiments/22_roy_5a.md). Protocol: [`22_roy_5a_reproduction.md`](../experiments/protocols/22_roy_5a_reproduction.md). Spec: [`roy_5a_iteration.yaml`](../../scenarios/roy/roy_5a_iteration.yaml). Fixture: [`roy_2pc_holdout.yaml`](../../scenarios/roy/roy_2pc_holdout.yaml) (reused unchanged). Analyzer: [`scripts/analyze_roy_5_cosine_localization.py`](../../scripts/analyze_roy_5_cosine_localization.py). Persistence prereq: [PR #248](https://github.com/dennys246/Maxim/pull/248).
<!-- /roy-iteration:roy-5a -->

### Roy-1: Adversarial (planned, unrun)
*Status: design above; awaiting [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) Stages 0-3 to ship in 1.0.*

---

## Open questions / known unknowns

- **Does substrate-only priming produce LLM-readable signal at test time?** Without occasional LLM-mediated priming turns, the substrate may consolidate around action distributions the test-time LLM doesn't naturally produce. May need hybrid priming.
- **Are the existing decay rates (`_reward_bias_decay_tau = 50.0` ticks, percept-valence decay) compatible with thousand-turn priming?** Decay tuned for short-horizon learning may simply not consolidate over the timescales persona requires. First Roy will surface this; expect to discover decay is too aggressive.
- **Does multi-agent attribution stay clean at scale?** Per-agent stash dicts (CC4 rule) tested at small N; Roy-1 stresses them with ~1,000 distinct adversary encounters. Pre-Roy stress test recommended.
- **Is the hybrid Wire 1 design (substrate annotates LLM context) sufficient for behavioral persona expression, or does it leak too much through the LLM?** Roy-1 three-arm comparison answers this directly; if A and B are indistinguishable behaviorally, the answer is the wire isn't sufficient.
- **Cradle developmental scenarios already produce affordance learning. Do they produce *persona-shaping* learning, or do we need something structurally different from cradle?** Roy-1 is closer to multi-agent social-learning than to single-agent embodiment-learning; cradle methodology may not transfer cleanly.

## Predictions revision policy

This section is updated whenever a Roy iteration concludes. The diff between the prediction and the finding is the actual product of this doc. We revise predictions in the doc directly rather than maintaining separate "prediction history" — git history is the audit trail.

## Cross-references

- [persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md) — the cleanup that creates the Roy-N-B baseline arm.
- [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) — the wires this doc depends on.
- [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — sister living doc for within-agent improvement.
- [memory_consolidation_practice.md](memory_consolidation_practice.md) — sister living doc for sleep-replay consolidation.
- [docs/experiments/protocols/](../experiments/protocols/) — reproduction runbooks for each Roy iteration land here.
