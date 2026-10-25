# Re-assessment: narrator-driven evidence recorded before #1261 and #1260

> **Status: RECORDED 2026-10-25** ([#1301](https://github.com/dennys246/Maxim/issues/1301), Session E, paper only).
> **This re-assessment moves no row's status**: no row changes status without its own evidence. Each claim gets an
> outcome:
> - **unaffected**, with the structural reason;
> - **bounded**, with the new caveat;
> - **withdrawn**: not supported until it is re-run on the fixed path (owner decision 2026-10-25).
>
> The facts below were read from the committed records and from the pre-fix code at `main` `7a6a5b56`.

## The two defects, and the code paths they reached

Both defects shaped every narrator-driven sim: `maxim --sim` and every caller of
`simulation/orchestrator.py::start_simulation_mode`.

**#1261: the narrator saw delivery receipts, not the AUT's words.**
- Fixed by PR #1289, merged 2026-10-15 as `be0aefc5`; the pre-fix code is its first parent, `75f1b4ab`.
- `simulation/bridge.py::SimulationBridge._send_and_wait` built the per-turn `response` by joining `str(result_output)`
  for `respond` and `speak`.
  - For `respond`, that output is `RespondTool`'s receipt, `{'delivered': True, 'mode': 'cli'}`.
  - For `speak`, it is `SpeakTool`'s TTS receipt.
  - `say` was never joined.
- The damage depended on which narrator was running:
  - **The plain `--sim` narrator** (`simulation/tools.py::SendMessageTool`, `ObserveActionsTool`) saw receipts for
    `respond` and `speak`. It did see `say`'s words in its per-action summary: `tools/narrative.py::SayTool` returns
    `{"said": text}`.
  - **The generative-campaign narrator** (the cradle arcs, Exp 11/37/38/40, and the Roy priming arcs) read only
    `response` (`simulation/generative_runner.py::run_generative_campaign`, `last_aut_response`). It saw a receipt
    on a `respond` turn and nothing otherwise. It was **AUT-blind**, running open-loop, before and after the fix
    alike, on every turn without a `respond`.
  - **The DM runtime** (`simulation/dm_runtime.py::DMRuntime._classify_choice`) was also reached. No committed claim
    rests on a DM run.
- `report.json` `response_texts` holds receipts: all 209 committed entries.
- The words were always in `actions.jsonl` `params`, which the defect never touched.

**#1260: the AUT was told `respond: NOT AVAILABLE`.**
- Fixed by PR #1299, merged 2026-10-16 UTC as `5a54c197`; the pre-fix code is `8af3ce49`. Present since `794ee046`
  (2026-04-03).
- `start_simulation_mode` §5 wrote `respond: NOT AVAILABLE. Use send_message instead.` with `followup_type: process`
  into the process-global `modes.definitions.TOOL_DESCRIPTIONS`. This happened before the AUT thread and before the
  generative, DM and fixture branches.
- So **every LLM-primary AUT** read it through `runtime/agent_loop.py::_describe_tools_for_prompt`:
  - plain, generative, research and Roy fixture runs;
  - `api.py` campaigns;
  - embodied AUTs (`is_embodied` changes only identity text).
- Each `respond` drew a follow-up: an extra LLM call and an extra action. `respond` stopped ending the turn.
- **Not reached:**
  - **Substrate-primary AUTs.** `runtime/substrate_proposal.py::propose_via_substrate` builds no prompt. A `respond`
    it selects supplies no message and fails ("Message cannot be empty"), so a substrate AUT never produced a
    receipt or words. The `cradle_mother` runs (Exp 48/52/54) also offer only `turn_left` and `turn_right`.
  - **Harnesses that never call `start_simulation_mode`:** the Minecraft harness and `water_trial`, the robot live
    mode, `scripts/orient_substrate/`, `behavioral_convergence_*`, and pytest.
- No committed record contains an AUT prompt ("NOT AVAILABLE" appears 0 times in the committed run logs). So the
  leak is established from the code, not from a record.

### Facts the issue did not state

- **12 of the 13 heartbeat Sim-Short `respond` receipts are the AUT's LLM-outage fallback, not speech.** Decoded from
  `docs/experiments/data/heartbeat_1_1/sim_short_2026-08-18/*/actions.jsonl.gz` `params`, each reads "I heard your
  question about '…', but I'm having trouble processing it right now":
  - all 6 in `20260818_004321`, which is T3-9's surviving copy;
  - all 6 in `20260818_010010`;
  - the 1 in `20260818_233844`.

  `20260818_010010`'s single `speak` ("Wake up, guard. We need to escape.") is real, and the narrator received its
  TTS receipt instead. Under #1261 the narrator also could not see that the AUT's LLM was down.
- **The committed Exp 37/38/40 per-trial records show `respond` volume uneven across arms and models.** These are
  the `tool_usage` totals per arm, recomputed from `docs/experiments/data/`:

  | Fire | Total | Per arm |
  |---|---|---|
  | Exp 37 Qwen14B | 0 | — |
  | Exp 37 Mistral24B | 0 | — |
  | Exp 37 Qwen32B | 6 | A 2, B-nac-bias-off 2, C 2 |
  | Exp 37 R1 | 27 | A 9, B 3, wire-a-off 3, nac-off 3, C 9 |
  | Exp 37 Sonnet | 81 | A 20, B 0, wire-a-off 37, nac-off 24, C 0 |
  | Exp 37 heartbeat Qwen32B (2026-08-21) | 32 | A 3, B 9, wire-a-off 3, wire-1-off 3, nac-off 8, C 6 |
  | Exp 38 Sonnet | 84 | A 33, wire-1-off 21, nac-off 27, C 3 |
  | Exp 38 GPT-4o | 21 | A only |
  | Exp 38 DeepSeek | 0 | — |
  | Exp 38 R1 | 37 | spread over every arm |
  | Exp 40 Qwen32B | 45 | A 6, B 6, wire-a-off 3, wire-1-off 9, nac-off 9, C 12 |

  `scripts/benchmark_cross_session.py::TURN_BOUNDARY_TOOLS = {"say", "respond"}` splits the per-turn bins on
  these calls.

## Outcomes, one entry per claim

Each outcome is decided from the claim's scoring path (the field or file it was scored from) and the run's
structure, never from its result.

### Ledger rows ([behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md))

| Claim | Run, path | Scored from | Outcome |
|---|---|---|---|
| **T1-16** (EARNED 2026-10-04, narrow): carried memory takes part in recall | Exp 63, `63_carried_recall/` (2026-10-03, `52653848`), plain `--sim` | `scripts/o19_verdict.py::exp63_gates`: `aut_hippocampus.json` round-trip and field equality; surfaced ids against a frozen-ranker recomputation | **Unaffected.** 0 `respond`/`speak`/`say` in either session, so #1261 never fired. Every gate compares the run with itself (store round-trip, ranker equality), which neither defect can change. **Caveat:** #1260 was in the prompt, so N1 = 84 and the turn content are not comparable with a post-fix run. |
| **T1-1** (SUPERSEDED by T1-16), history: the 2026-04 Exp 10 "3 memories/turn" | Exp 10, raw LOST | enrichment traces | **Unaffected** (retired). It read surfaced memories, not words. |
| **T1-1** history: the 2026-08-18/19 heartbeat re-validation (hippocampus 108 → 535 → 295) | `heartbeat_1_1/sim_short_2026-08-18/` | store counts from `aut_hippocampus.json` | **Bounded.** Persistence stands. The counts include the episodes the `respond` follow-ups created, and the AUT's LLM was intermittently down (7 of 7 receipts are outage fallbacks). Not comparable across the fix. |
| **T1-1** history: MAINTAINED 2026-09-27 (narrow), and the O19 campaigns 1–2 | `rerun_exp10_2026-09-27/`, `rerun_exp10_o19*/` | store round-trip, enrichment traces | **Unaffected.** No speech in any session. All were already typed aborts that cannot gate. |
| **T1-2** (STALE): the June Exp 37 fires and the Goldilocks map | `37_results*.jsonl`, generative `cradle` | `scripts/analyze_exp37.py`: `positive_approach_engagement_fraction` from affordance actions | **Bounded.** The primary metric reads no words, and the narrator was open-loop either way. But #1260 changed every arm's prompt. The `respond` bursts, each with its follow-up, used the action budget unevenly across arms (Sonnet A 20 / B 0 / wire-a-off 37) and moved the turn bins. These are confounds on top of L8. The owed re-fire, B4 ([#1354](https://github.com/dennys246/Maxim/issues/1354)), runs on the fixed path. |
| **T1-2**: Exp 38 counter-prior dominance (4 models; R1 interaction +2.25 SD) | `38_results_*.jsonl`, generative `cradle_deceptive` | `warm_self_engagement_fraction`, `first_contact_*` | **Bounded.** As above. `first_contact_action_index` also shifts by one action per follow-up before the first contact. |
| **T1-2**: the 2026-08-21 heartbeat re-fire (INCONCLUSIVE) | `37_heartbeat_1.1_qwen32b.jsonl` | as above | **Bounded.** Already INCONCLUSIVE. **New for L8:** `respond` volume on Qwen32B rose from 6 (June) to 32 (August) under the same prompt leak, so the defect's effect was not constant between the two fires L8 compares. |
| **T3-9** (PARTIAL 2026-10-03, narrow): the Exp 09 O19 re-run | `rerun_exp09_o19/20261002_230206`, plain `--sim` | `o19_verdict.py::exp09_gates`: reflex lines keyed on the narration | **Unaffected by #1261:** 0 speech, so no receipt. **Bounded by #1260:** the prompt shifts the AUT's own deliberation, which the row already says can produce H1/H4/H5 firings. |
| **T3-9**: its surviving Exp 09 copy, Sim-Short `20260818_004321` | `heartbeat_1_1/…/20260818_004321` | reflex events | **Bounded.** The reflexes fire on narration, so the count stands. But all 6 of this run's `respond` calls are LLM-outage fallbacks, so it is an outage-dominated record that #1261 hid from the narrator. It is weak as a "surviving record". |
| **T3-9** history: EARNED 2026-04-25; MAINTAINED 2026-09-24 | raw LOST; `rerun_exp09_2026-09-24/` | reflex lines | **Unaffected.** No words read. The 2026-09-24 run is already a typed abort. |
| **T3-11** (DROPPED 2026-08-30): Exp 12 Phase C−A, "3 respond hits vs 1" | Exp 12, raw `actions.jsonl` LOST | a hand count of `respond` actions | **Withdrawn.** The quantity is `respond` calls made against "NOT AVAILABLE", each drawing a follow-up. The row is already DROPPED, with the number called confounded. |
| **T1-3, T1-6, T1-8, T1-9, T1-10** (and Exp 13/39/41/42b/54) | narrator present, **substrate-primary** AUT | substrate telemetry, `safe_pref`, the mother telemetry | **Unaffected.** No AUT prompt, and no possible words: a substrate `respond` fails for want of a message, and `cradle_mother` offers no speech tool. The narrator's content never fed the scoring. |
| **T3-2, T3-3, T3-10, T3-15, T3-16, T3-4/T3-5/T1-5** | various, mostly lost | non-word primaries, or no recorded result | **Unaffected.** Each is DROPPED, DORMANT or superseded on grounds that do not depend on the defects. |

The other 18 rows are out of scope, with no narrator: the Minecraft rows (T1-11 to T1-15), the robot row (T1-7), the
test-only T1-4, and the tier-2, dormant and never-run T3 rows.

### Experiment-doc claims with no live ledger row

| Claim | Scored from | Outcome |
|---|---|---|
| **Exp 12** "CLEAN PASS, 7/7 phases recalled `BLUE-7-DAWN`" ([12_v1_phased_attribution.md](12_v1_phased_attribution.md); the [index](README.md); the 1.0 flag-removal decision) | `v1_recall_success`: did session 2 emit the token in **any** tool action, read post hoc from `actions.jsonl` (LOST) | **Bounded.** "Emitted anywhere" means each `respond` follow-up was one more chance to emit the token. The recall narrator probed blind, and the memory and link counts include follow-up episodes. The word-independent half stands: `memory_recall` output carried the token, so the substrate is the data path. Between-phase comparisons (C−A, memory counts) are confounded. It cannot be re-derived from git. |
| **Hippocampal recall** ([run notes](hippocampal_recall_run_notes.md), 2026-04-06): "Verath survives in memory; behavioural recall at the door fails" | memory survival via `inspect_aut(memory_recall)`; behavioural recall = whether the AUT used the password | **Memory half unaffected. Behavioural half withdrawn.** Passing the door needs speech, and the AUT's prompt said `respond` was unavailable. The observed failure (it called `research("miller")` and ignored the door) may be independent of that, so the run neither supports nor refutes behavioural recall. |
| **Exp 32**: "Arm A's `respond` count tripled (47 vs 11): it leaned into verbalizing" ([32_wire_a_post_w1_w2.md](32_wire_a_post_w1_w2.md)) | per-arm `respond` counts (Roy `result.json`, LOST) | **Withdrawn.** The finding *is* a `respond` count, and #1260 shaped it through both the prompt and the follow-ups. The wiring-bug findings in the same doc are structural and stand. |
| **Exp 18 Roy-2**: the A-vs-C tool-family shift, "`respond` 17 vs 20/21: the primed substrate redirects turns away from narration" ([18_roy_2.md](18_roy_2.md)) | per-arm tool counts (LOST) | **Withdrawn** (the `respond` half). The `substrate_diff` priming result is unaffected. |
| **Roy-1a/3, Exp 30/33/34** NULLs, and "Wire-A reaches the prompt" | `substrate_diff`, `sense_food_source` counts | **Bounded.** The nulls rest on non-`respond` primaries and on the priming-side cluster bias. Don't cite the salience KS, the episode deltas or the `respond` counts reported beside them. |
| **Exp 44** counterfactual flip rate (EXPLORATORY) and the 44b pilot; [paper outline](../paper_outline.md) Contribution 2 | `analyze_counterfactual.py`: full vs ablated prompt | **Bounded.** The paired design cancels the shared leaked line, but the base rates were set under it. The 44b confirmatory must run on the fixed path. |
| **Exp 40** counter-prior Goldilocks, Qwen32B ([40_counter_prior_goldilocks.md](40_counter_prior_goldilocks.md); paper Contribution 3; the 1.0 release) | `warm_self_engagement_fraction`, `first_contact_action_index` | **Bounded.** A between-arm contrast with the leak in every arm and 45 `respond` calls spread unevenly. First-contact index and turn bins shift with the follow-ups. |
| **Exp 11** cradle PoC; **Exp 08** H1/H3/H4; **e0** and **sem_tool_discovery_s1** ("the AUT uses affordances from turn 1") | narrator output and tool-call greps (LOST) | **Bounded.** The infrastructure findings stand. s1's "affordances from turn 1" ran under a prompt that steered away from `respond`, toward affordances: a direction that favours the finding. Re-run it before citing. |

### Public and release surfaces

Each surface inherits its outcome from the run it cites:

| Surface | Outcome |
|---|---|
| README (T1-16), CHANGELOG 1.3.2, `release_1_3_2.md` | **Unaffected** (T1-16, above) |
| CHANGELOG 1.3.1, `release_1_3_1.md`, `10_cross_session_enrichment.md`: the Exp 10 re-run | **Unaffected.** The row it names is already SUPERSEDED. |
| CHANGELOG 1.0.0, `maxim_1_0_release.md`, `docs/plans/README.md`: the Goldilocks band, the four-model dominance, Exp 40 | **Bounded**, as for T1-2 and Exp 40 above |
| [limits](../limits/README.md) L6, L7, L8 | **Bounded.** L8's drift comparison spans the 6 → 32 change in `respond` volume above. |
| limits L1, L2, L4, L5, L11, L12; the Exp 42/48/52/53b/54 surfaces; CHANGELOG 0.9.x Roy | **Unaffected.** A substrate AUT, or a narrator used only for timing. |
| [paper outline](../paper_outline.md) Contribution 2 (Exp 44b) and Contribution 3 (Exp 38/40) | **Bounded** |
| [experiments index](README.md) (Exp 12); [docs/index.md](../index.md) (hippocampal recall) | **Bounded** (Exp 12); **withdrawn** (the behavioural half) |

Released CHANGELOG sections are left as published. The `[Unreleased]` section records this re-assessment (owner
decision 2026-10-25).

## What is owed

- **The withdrawn observations** are marked where they are stated, and none backs a live ledger row:
  - T3-11's "3 respond hits vs 1";
  - Exp 32's "respond tripled";
  - Exp 18's `respond` half;
  - the hippocampal-recall behavioural half.

  **Trigger:** any new citation of one of them. **Re-run:** on the fixed path (after PR #1289 and PR #1299) before it
  is cited, tracked in [#1461](https://github.com/dennys246/Maxim/issues/1461). No sim runs now.
- **The bounded LLM-AUT claims** (T1-2, Exp 38/40, Exp 44) are already owed re-fires on the fixed path: B4 (#1354)
  for T1-2, and the 44b confirmatory. Each must record the AUT prompt, or a hash of it, so that a leak like #1260 can
  be seen in the record next time.
- **No row changes status.** Every ledger row named above carries a dated note pointing here.
