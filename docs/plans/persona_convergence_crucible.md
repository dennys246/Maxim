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

### Roy-2 (planned, methodology refinement first)
*Status: Roy-1b ships a methodology decision: widen priming arc diversity, tune min_confidence, OR ship Wire 1 substrate-annotates-LLM-context. Roy-2 awaits a decision among the three.*

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
