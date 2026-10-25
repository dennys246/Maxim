# Re-assessment: narrator-driven evidence recorded before #1261 and #1260

> **Status: RECORDED 2026-10-25** ([#1301](https://github.com/dennys246/Maxim/issues/1301), Session E, paper only).
> **This re-assessment moves no row's status**: no row changes status without its own evidence. Each claim gets an
> outcome:
> - **unaffected**, with the structural reason;
> - **bounded**, with the new caveat;
> - **withdrawn**: not supported until it is re-run on the fixed path (owner decision 2026-10-25).
>
> The facts below were read from the committed records as they stand at `main` `7a6a5b56`, and from the pre-fix
> code at `75f1b4ab` (#1261) and `8af3ce49` (#1260). The April to August runs predate the 2026-09-04 extraction of
> `_describe_tools_for_prompt` (P21); the mechanism was the same, since at `f742527` (Exp 12) `run_agentic_loop` read
> `TOOL_DESCRIPTIONS[name]` inline and the orchestrator wrote the same `respond` entry.

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
    `response` (`simulation/generative_runner.py::run_generative_campaign`, `last_aut_response`), never the action
    list. Before the fix that was a receipt on a `respond` turn or a successful `speak` turn, and nothing on other
    turns. After the fix it hears the AUT's words on speech turns (`respond`, `say`, `speak`, and speech affordances).
    Before and after alike, it never sees a non-speech action. Every committed Exp 37/38/40 record has 0 `say` and
    0 `speak` calls, so in those runs it saw only `respond` receipts and otherwise ran open-loop.
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
    it selects supplies no message, so `Tool.run`'s input check refuses it before `execute` ("Missing required input:
    message"), so a substrate AUT never produced a
    receipt or words. The `cradle_mother` runs (Exp 48/52/54) also offer only `turn_left` and `turn_right`.
  - **Harnesses that never call `start_simulation_mode`:** the Minecraft harness and `water_trial`, the robot live
    mode, `scripts/orient_substrate/` and `behavioral_convergence_*`. pytest does call it, with mocked LLMs; no
    evidence rests on it.
- **The leak is on record.** The committed Exp 44b pilot captures
  (`docs/experiments/data/44b_pilot/arms/{A_green_safe,B_purple_safe,CTRL_transplant_A_into_B}/seed1/capture.jsonl`,
  committed in `88739318`) hold the AUT prompt. Each row has a `prompt_full` and a `prompt_ablated`. Of the 101 rows
  (36, 30 and 35), the 90 planning prompts (30 per arm) carry `- respond: NOT AVAILABLE. Use send_message instead.`
  in the AUT's tool list, in both the full and the ablated prompt. The other 11 (A 6, CTRL 5) are post-tool
  follow-up prompts ("You just executed …"), which list no tools and say "use 'respond' to inform the user". No committed
  `.gz` run log contains an AUT prompt ("NOT AVAILABLE" appears 0 times in them), so for every other run the leak is
  established from the code.

### Facts the issue did not state

- **All 13 heartbeat Sim-Short `respond` calls are the AUT's LLM-outage fallback, not speech.** Decoded from
  `docs/experiments/data/heartbeat_1_1/sim_short_2026-08-18/*/actions.jsonl.gz` `params`, each is one of the two
  canned replies in `agents/llm_fallback.py`:
  - all 6 in `20260818_004321`, which is T3-9's surviving copy: "I heard your question about '…', but I'm having
    trouble processing it right now";
  - all 6 in `20260818_010010`: the same template;
  - the 1 in `20260818_233844`: the other template, "I'm not sure how to answer '…'. Could you rephrase that or ask
    something else?";
  - `20260819_000740` made none.

  `20260818_010010`'s one non-fallback utterance, a `speak` ("Wake up, guard. We need to escape."), **failed**
  (`"success": false`, `"error": "Missing required input: text"`, output null; it passed `message`, not `text`). It
  delivered nothing, and the narrator got nothing from it: the pre-fix bridge joins only a truthy output, so that
  run's 6 `response_texts` entries are its 6 `respond` receipts. These runs used the **plain** narrator (their goals
  match no generative arc), which lists every sink action with its `tool` and `success`. So the outage itself was
  visible to the narrator as `_llm_unavailable` actions (6, 4 and 1 of them); what #1261 hid was that the `respond`
  receipts stood for fallback text.
- **The committed Exp 37/38/40 per-trial records show `respond` volume uneven across arms and models.** These are
  the `tool_usage` totals per arm, recomputed from `docs/experiments/data/` (`37_results.jsonl` is Qwen14B; the
  excluded files `37_results_invalid_llm_outage.jsonl` (657 calls) and `37_results_pre_fix_bug.jsonl` (0) back no
  claim):

  | Fire | Total | Per arm |
  |---|---|---|
  | Exp 37 Qwen14B | 0 | — |
  | Exp 37 Mistral24B | 0 | — |
  | Exp 37 Qwen32B | 6 | A 2, nac-off 2, C 2 |
  | Exp 37 R1 | 27 | A 9, B 3, wire-a-off 3, nac-off 3, C 9 |
  | Exp 37 Sonnet | 81 | A 20, B 0, wire-a-off 37, nac-off 24, C 0 |
  | Exp 37 heartbeat Qwen32B (2026-08-21) | 32 | A 3, B 9, wire-a-off 3, wire-1-off 3, nac-off 8, C 6 |
  | Exp 38 Sonnet | 84 | A 33, wire-1-off 21, nac-off 27, C 3 |
  | Exp 38 GPT-4o | 21 | A only |
  | Exp 38 DeepSeek | 0 | — |
  | Exp 38 R1 | 37 | spread over every arm |
  | Exp 40 Qwen32B | 45 | A 6, B 6, wire-a-off 3, wire-1-off 9, nac-off 9, C 12 |

  `scripts/benchmark_cross_session.py::TURN_BOUNDARY_TOOLS` (`frozenset({"say", "respond"})`) splits the per-turn
  bins on these calls. Every one of these records has 0 `say` and 0 `speak` calls. Sessions are capped at 12 turns,
  so a burst mostly adds actions to a trial rather than displacing others, and the large bursts sit in one trial per
  cell (Exp 37 Sonnet: A's 20 is one trial; wire-a-off's 37 is 19 + 18 in two scenarios). Exp 40 is near-uniform:
  3 calls in 13 of its 14 affected trials and 6 in one. R1 makes at most 3 per trial in either experiment.
- **In Exp 38 Sonnet, the defect's channel produces the reported interaction value.** `warm_self_engagement_fraction`
  is `warm_self_actions / engagement_actions`, and 0.0 when a trial has no engagement. Such a trial is scored 0.0,
  not dropped. In Sonnet's arm A `deceptive_fire`, seeds 43 and 45 made 25 and 8 `respond` calls (seed 45 was cut
  to 5 turns), never touched the hearth, and score 0.0. Arm A is the baseline for both the interaction and the
  pooled SD. The `first_contact_warm_self` primary drops those two trials (None), so its arm-A deceptive P = 0.67
  rests on 3 of 5 trials. Trial-level `actions.jsonl` for Exp 37/38 is not committed, so "a `respond` loop driven by
  #1260's follow-up" is inferred from the counts, not read from the records.

  A **post hoc** sensitivity check, recomputed from the committed rows with the analyzer's formula
  (`scripts/analyze_exp37.py`: Exp 37 primary `(B − A) / sd(A)` on `fire_pit`
  `positive_approach_engagement_fraction`; Exp 38 interaction `(Δ_dec − Δ_con) / pooled arm-A SD` on
  `warm_self_engagement_fraction`), dropping every trial with a `respond` call. It is descriptive, not a result:

  | Fire | All trials | Without trials that called `respond` |
  |---|---|---|
  | Exp 37 Qwen32B | +1.43 SD | +1.32 SD |
  | Exp 37 R1 | +2.11 SD | +1.81 SD |
  | Exp 38 Sonnet | +0.40 SD | **−0.99 SD** (pooled arm-A SD 0.225 → 0.089; the PASS bar is ≤ −1.0) |
  | Exp 38 GPT-4o | −0.46 SD | −0.33 SD |
  | Exp 38 DeepSeek | −0.62 SD | −0.62 SD (0 `respond` calls) |
  | Exp 38 R1 | +2.25 SD | +2.31 SD |

  The other fires hold; Sonnet's Exp 38 value is mostly the defect's channel, and the push runs toward the frozen
  rule's "otherwise → dominance" default, a direction that favours the claimed outcome.

## Outcomes, one entry per claim

Each outcome is decided from the claim's scoring path (the field or file it was scored from) and the run's
structure, never from its result.

### Ledger rows ([behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md))

| Claim | Run, path | Scored from | Outcome |
|---|---|---|---|
| **T1-16** (EARNED 2026-10-04, narrow): carried memory takes part in recall | Exp 63, `63_carried_recall/` (2026-10-03, `52653848`), plain `--sim` | `scripts/o19_verdict.py::exp63_gates`: `aut_hippocampus.json` round-trip and field equality; surfaced ids against a frozen-ranker recomputation | **Unaffected (the gates); not comparable across the fix (N1 = 84, the turn content).** 0 `respond`/`speak`/`say` in either session, so #1261 never fired. Every gate compares the run with itself (store round-trip, ranker equality), which neither defect can change. #1260 was in the prompt, so N1 = 84 and the turn content are not comparable with a post-fix run, and the zero speech may itself be #1260's effect. |
| **T1-1** (SUPERSEDED by T1-16), history: the 2026-04 Exp 10 "3 memories/turn" | Exp 10, raw LOST | enrichment traces | **Unaffected** (retired). It read surfaced memories, not words. |
| **T1-1** history: the 2026-08-18/19 heartbeat re-validation (hippocampus 108 → 535 → 295) | `heartbeat_1_1/sim_short_2026-08-18/` | store counts from `aut_hippocampus.json` | **Bounded.** Persistence stands. The counts include the episodes the `respond` follow-ups created, and the AUT's LLM was intermittently down (all 7 of its `respond` calls are canned outage fallbacks, and the one `speak` failed). Not comparable across the fix. |
| **T1-1** history: MAINTAINED 2026-09-27 (narrow), and the O19 campaigns 1–2 | `rerun_exp10_2026-09-27/`, `rerun_exp10_o19*/` | store round-trip, enrichment traces | **Unaffected.** No speech in any session. All were already typed aborts that cannot gate. |
| **T1-2** (STALE): the June Exp 37 fires and the Goldilocks map | `37_results*.jsonl`, generative `cradle` | `scripts/analyze_exp37.py`: `positive_approach_engagement_fraction` from affordance actions | **Bounded.** The primary metric reads no words, and in these runs (0 `say`/`speak`) the narrator saw only `respond` receipts. But #1260 changed every arm's prompt. The `respond` bursts, each with its follow-up, added actions to single trials unevenly across arms (Sonnet A 20 / B 0 / wire-a-off 37), moved the turn bins and shifted `first_contact_action_index`. These are confounds on top of L8. The post hoc check above leaves the Qwen32B and R1 values near where they were (+1.43 → +1.32, +2.11 → +1.81). The owed re-fire, B4 ([#1354](https://github.com/dennys246/Maxim/issues/1354)), runs on the fixed path. |
| **T1-2**: Exp 38 counter-prior dominance (4 models; R1 interaction +2.25 SD) | `38_results_*.jsonl`, generative `cradle_deceptive` | `warm_self_engagement_fraction`, `first_contact_*` | **Bounded** at GPT-4o, DeepSeek-V3 and R1, as above (DeepSeek made 0 `respond` calls; GPT-4o's 21 are in one arm-A `fire_pit` trial; R1 makes at most 3 per trial). **Sonnet: its interaction value (+0.40 SD) and its three ablation-shrinkage values (−0.62, 0.21, 0.63 SD) are withdrawn.** They are produced by the defect's channel: two of its five arm-A deceptive trials are `respond` loops with no hearth engagement, scored 0.0, and the confound favours the claimed dominance (+0.40 → −0.99 SD without them, post hoc). Sonnet's dominance verdict now rests on its first-contact result alone (FAIL: arm-A deceptive P = 0.67 over 3 of 5 trials, B 1.00). `first_contact_action_index` also shifts by one action per follow-up before the first contact. |
| **T1-2**: the 2026-08-21 heartbeat re-fire (INCONCLUSIVE) | `37_heartbeat_1.1_qwen32b.jsonl` | as above | **Bounded.** Already INCONCLUSIVE. **New for L8:** `respond` volume on Qwen32B rose from 6 (June) to 32 (August) under the same prompt leak, so the defect's effect was not constant between the two fires L8 compares. |
| **T3-9** (PARTIAL 2026-10-03, narrow): the Exp 09 O19 re-run | `rerun_exp09_o19/20261002_230206`, plain `--sim` | `o19_verdict.py::exp09_gates`: reflex lines keyed on the narration | **Unaffected by #1261:** 0 speech, so no receipt. **Bounded by #1260:** the prompt shifts the AUT's own deliberation, which the row already says can produce H1/H4/H5 firings. |
| **T3-9**: its surviving Exp 09 copy, Sim-Short `20260818_004321` | `heartbeat_1_1/…/20260818_004321` | reflex events | **Bounded.** The reflexes fire on narration, so the count stands. But all 6 of this run's `respond` calls are canned LLM-outage fallbacks, beside 6 `_llm_unavailable` actions, so it is an outage-dominated record. The plain narrator saw the outage as failed actions; #1261 hid only that the receipts stood for fallback text. It is weak as a "surviving record". |
| **T3-9** history: EARNED 2026-04-25; MAINTAINED 2026-09-24 | raw LOST; `rerun_exp09_2026-09-24/` | reflex lines | **Unaffected.** No words read. The 2026-09-24 run is already a typed abort. |
| **T3-11** (DROPPED 2026-08-30): Exp 12 Phase C−A, "3 respond hits vs 1" | Exp 12, raw `actions.jsonl` LOST | a hand count of `respond` actions | **Withdrawn.** The quantity is `respond` calls made against "NOT AVAILABLE", each drawing a follow-up. The row is already DROPPED, with the number called confounded. |
| **T1-3, T1-6, T1-8, T1-9, T1-10** (and Exp 13/39/41/42b/54) | narrator present, **substrate-primary** AUT (T1-10: a narrated Exp 52 nursery, then an unnarrated readout on the Reachy via `exp53_cross_context_readout.py`) | substrate telemetry, `safe_pref`, the mother telemetry | **Unaffected.** A substrate AUT reads no prompt and emits no words: a substrate-selected `respond`, `say` or `speak` supplies no text and is refused by `Tool.run`'s input check before `execute` ("Missing required input: message" / "…: text"), and `cradle_mother` offers only `turn_left`/`turn_right`. So the narrator's input was identical before and after both fixes. The narrator's output does feed the substrate (Roy-4 shows it encoded into EC), but that output never depended on the defects. |
| **Roy-0, Roy-1b, Roy-2pc, Roy-2c, Roy-4, Roy-5a, Roy-5b** (and its confound isolation), **Roy-3c** (Exp 15/17/19/20/21/22/35/36/29; CHANGELOG 0.9.x) | Roy runs with a substrate-primary AUT at priming and at test (`scenarios/roy/*_iteration.yaml` `aut_mode: substrate-primary`; Roy-3c ran `roy_2_iteration.yaml`, which is llm-primary at test, so only its priming-side measure is covered here) | `substrate_diff` (`cluster_reward_bias_l2`, node and cluster overlap), the co-activation analyzer | **Unaffected**, for the same structural reason as the row above: the narrator's input was identical before and after both fixes. |
| **T3-2, T3-3, T3-10, T3-15, T3-16, T3-4/T3-5/T1-5** | various, mostly lost | non-word primaries, or no recorded result | **Unaffected.** Each is DROPPED, DORMANT or superseded on grounds that do not depend on the defects. |

The other 18 T1/T3 rows are out of scope, with no narrator: the Minecraft rows (T1-11 to T1-15), the robot row
(T1-7), the test-only T1-4, and the T3 rows whose status is `TIER-2` (T3-17 to T3-20), dormant or never run. (The
Tier 2 section itself has no table rows, only a bullet list of scaffolding that carries no behavioural claim.)

### Experiment-doc claims with no live ledger row

| Claim | Scored from | Outcome |
|---|---|---|
| **Exp 12** "CLEAN PASS, 7/7 phases recalled `BLUE-7-DAWN`" ([12_v1_phased_attribution.md](12_v1_phased_attribution.md); the [index](README.md); the 1.0 flag-removal decision) | `v1_recall_success`: did session 2 emit the token in **any** tool action, read post hoc from `actions.jsonl` (LOST) | **Bounded.** "Emitted anywhere" means each `respond` follow-up was one more chance to emit the token. The plain recall narrator saw every action's tool, success and output (the `memory_recall` output included), but could not see the AUT's `respond` words, and the memory and link counts include follow-up episodes. The word-independent half stands: `memory_recall` output carried the token, so the substrate is the data path. Between-phase comparisons (C−A, memory counts) are confounded. It cannot be re-derived from git. |
| **Hippocampal recall** ([run notes](hippocampal_recall_run_notes.md), 2026-04-06): "Verath survives in memory; behavioural recall at the door fails" | memory survival via `inspect_aut(memory_recall)`; behavioural recall = whether the AUT used the password | **Memory half unaffected. Behavioural half withdrawn**, because the reply channel was declared unavailable: passing the door needs speech, `say` did not exist yet (`SayTool` arrived in `bb14acef`, 2026-04-06 19:01 −0600, after both sessions at 17:26 and 18:20), so `respond` was the speech channel, and the AUT's prompt said it was NOT AVAILABLE. **Counter-evidence, recorded beside the withdrawal:** the withdrawn headline rests on Run #2 (softened miller), where at the door the AUT called `read_file`/`analyze_text`; in Run #1 it called `research("miller")`, the documented miller fixation. Both runs delivered turns by direct bridge injection, so #1261 does not apply to either. The AUT used `respond` on other turns despite the leak (Run #1 turn 2; Run #2 turns 1, 5 and 7). So the door failure may be independent of #1260, and the run neither supports nor refutes behavioural recall. |
| **Exp 32**: "Arm A's `respond` count tripled (47 vs 11 in Phase B) — the LLM leaned even harder into verbalizing rather than acting" ([32_wire_a_post_w1_w2.md](32_wire_a_post_w1_w2.md); the same doc later says "quadrupled") | per-arm `respond` counts (Roy `result.json`, LOST) | **Withdrawn.** The finding *is* a `respond` count, and #1260 shaped it through both the prompt and the follow-ups. The wiring-bug findings in the same doc are structural and stand. |
| **Exp 18 Roy-2**: the A-vs-C tool-family shift; "Arm A's `respond` count (17) is also notably lower than B's (20) or C's (21), suggesting the primed substrate redirects turns away from default-narration responses into sensory actions" ([18_roy_2.md](18_roy_2.md)) | per-arm tool counts (LOST); llm-primary at test | **Withdrawn** (the `respond` half). **Bounded** (the non-`respond` half: arm A's `sense`/`pick_up` family against C's): the prompt leak was in every arm and the follow-ups added actions. The `substrate_diff` priming result is unaffected (substrate-primary priming). |
| **Roy-1a/3, Exp 30/33/34** NULLs, and "Wire-A reaches the prompt" | `substrate_diff`, `sense_food_source` counts | **Bounded.** The nulls rest on non-`respond` primaries and on the priming-side cluster bias. Don't cite the salience KS, the episode deltas or the `respond` counts reported beside them. Roy-1a's `goal_reward_bias_l2` (0.2714) is also **bounded**: it comes from llm-primary tool-outcome credit at test, and each follow-up added an outcome. |
| **Exp 44** counterfactual flip rate (EXPLORATORY) and the 44b pilot; [paper outline](../paper_outline.md) Contribution 2 | `analyze_counterfactual.py`: full vs ablated prompt | **Bounded.** The paired design cancels the shared leaked line. For the 44b pilot that is **observed in the record**: every committed planning prompt, full and ablated, carries it (above). For Exp 44 arm A (July; no capture committed) it is inferred from the code, since `capture_paired_prompts.py` builds both prompts in the live process. The base rates were set under the line, so the 44b confirmatory must run on the fixed path. |
| **Exp 40** counter-prior Goldilocks, Qwen32B ([40_counter_prior_goldilocks.md](40_counter_prior_goldilocks.md); paper Contribution 3; the 1.0 release) | `warm_self_engagement_fraction`, `first_contact_action_index` | **Bounded.** A between-arm contrast with the leak in every arm and 45 `respond` calls spread near-uniformly (3 in 13 of 14 affected trials, 6 in one). First-contact index and turn bins shift with the follow-ups. |
| **Exp 11** cradle PoC; **Exp 08** H1/H3/H4; **e0** | narrator output and tool-call greps (LOST) | **Bounded.** The infrastructure findings stand; the behavioural observations ran with the leak in the prompt. |
| **sem_tool_discovery_s1** ("the AUT uses affordances from turn 1") ([sem_tool_discovery_s1.md](sem_tool_discovery_s1.md)) | tool-call greps (LOST) | **Withdrawn.** It ran under a prompt that steered the AUT away from `respond`, toward affordances: a direction that favours the finding. Re-run it on the fixed path before citing. The infrastructure findings (the hybrid prompt mode's tool count) stand. |

### Public and release surfaces

Each surface inherits its outcome from the run it cites:

| Surface | Outcome |
|---|---|
| README (T1-16), CHANGELOG 1.3.2, `release_1_3_2.md` | **Unaffected** (T1-16, above) |
| CHANGELOG 1.3.1, `release_1_3_1.md`, `10_cross_session_enrichment.md`: the Exp 10 re-run | **Unaffected.** The row it names is already SUPERSEDED. |
| CHANGELOG 1.0.0, [`maxim_1_0_release.md`](../announcements/maxim_1_0_release.md), [`docs/plans/README.md`](../plans/README.md): the Goldilocks band, the four-model dominance, Exp 40 | **Bounded**, as for T1-2 and Exp 40 above. "Dominance across all four frontier models" now reads: dominance at GPT-4o, DeepSeek-V3 and R1; Sonnet's interaction withdrawn, its verdict resting on first-contact over 3 of 5 arm-A deceptive trials. The two un-released surfaces carry a note; CHANGELOG 1.0.0 stays as published. |
| [limits](../limits/README.md) L6 | **Bounded by the prompt line only.** Measured on the Exp 37 Mistral24B fire, which made 0 `respond` calls, so the follow-up channel never fired. |
| limits L7 | **Unaffected.** Measured on Exp 48 (substrate-primary `cradle_mother`). |
| limits L8 | **Bounded.** Its drift comparison spans the 6 → 32 change in `respond` volume above. |
| limits L1, L2, L4, L5, L11, L12; the Exp 42/48/52/53b/54 surfaces; CHANGELOG 0.9.x Roy (the substrate-primary Roy row above) | **Unaffected.** A substrate AUT, or a narrator used only for timing. |
| [paper outline](../paper_outline.md) Contribution 2 (Exp 44b) and Contribution 3 (Exp 38/40) | **Bounded**; Contribution 3's Sonnet Exp 38 interaction and ablation values **withdrawn** |
| [experiments index](README.md) (Exp 12); [docs/index.md](../index.md) (hippocampal recall) | **Bounded** (Exp 12); **withdrawn** (the behavioural half) |

Released CHANGELOG sections are left as published. The `[Unreleased]` section records this re-assessment (owner
decision 2026-10-25).

## What is owed

- **The withdrawn observations** are marked where they are stated, and none backs a live ledger row:
  - T3-11's "3 respond hits vs 1";
  - Exp 32's "respond tripled";
  - Exp 18's `respond` half;
  - the hippocampal-recall behavioural half;
  - sem_tool_discovery_s1's "affordances from turn 1";
  - Exp 38 Sonnet's interaction value (+0.40 SD) and its three ablation-shrinkage values.

  **Trigger:** any new citation of one of them. **Re-run:** on the fixed path (after PR #1289 and PR #1299) before it
  is cited, tracked in [#1461](https://github.com/dennys246/Maxim/issues/1461). No sim runs now.
- **The bounded LLM-AUT claims** (T1-2, Exp 38/40, Exp 44) are already owed re-fires on the fixed path: B4 (#1354)
  for T1-2, and the 44b confirmatory.
- **A requirement on B4 ([#1354](https://github.com/dennys246/Maxim/issues/1354)) and on the #1461 re-runs, not a
  new standing rule:** each records the AUT prompt, or a hash of it, so that a leak like #1260 can be seen in the
  record. The Exp 44b pilot capture above is the existing example.
- **Notes beside each claim.** Every ledger row named above, and every experiment doc, limits entry and un-released
  surface that states a bounded or withdrawn claim, carries a dated note pointing here. Released CHANGELOG sections
  are the deliberate exception.
- **No row changes status.**
