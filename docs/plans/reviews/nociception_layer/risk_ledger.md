# Risk / ledger / sequencing lens — nociception_layer.md

**Verdict: ADOPT the deferral, with SHOULD-FIXes.** Deferral is right for E1. The ledger list is
incomplete, though: step 4's DRIVE decision reaches the survival rows. And one revive trigger is
missing. Phase 5's R4 credit routing turns F1 from inert to load-bearing.

## SHOULD-FIX

**S1. Step 4 trips Exp 60/61/62, not only 52/56.** `body.py` publishes drive breaches with
`agent_id`. `pain_signal_to_reaction` carries that id, and `_distribute_reward_from_reaction`
(`bio_stack.py`) pays `-intensity` for every negative reaction that has an agent. So `drive:oxygen`
pain is distributed as negative reward in the survival loop today (`water_trial.py` runs the full
`aut.bio`). `MAXIM_OPERANT_ONLY_CREDIT` gates only the tool-success floor (`tool_dispatch.py`), not
this path. The step's open question, "whatever the owner decides for DRIVE", is a ledger decision,
not a free parameter:
- Excluding DRIVE changes the training inputs of Exp 60, 61 and 62, Exp 52 (hunger), Exp 56 and 57,
  Exp 42 and Exp 45 (azimuth drive; hardware re-run).
- It also fires R3's §Outcome clause, since it is a credit-path write (roadmap §Bodies).

The step should name all of these rows, or pin DRIVE as included and say so.

**S2. Step 3's cluster-fear wording invites a widening that would break Exp 60–62.**
`{NOCICEPTIVE(health), DRIVE(oxygen)}` is not a set of kinds; it is kind plus name. A
`subscribe(kinds=...)` filter cannot express it. The obvious migration, `kinds={DRIVE, NOCICEPTIVE}`,
admits hunger, saturation and every world hit as fear's US. In E1, where the agent is hungry at the
food underwater, that would write hunger fear on the very cluster the rung measures. Rewrite the
step: the `kinds=` filter is a coarse pre-filter, and `DEFAULT_CLUSTER_FEAR_FAILURE_MODES` stays the
authority, unchanged. Changing it is a literal Exp 60 and Exp 61 trigger ("Wire-4 allowlist").

**S3. Two ledger rows are wrong or missing in "What gates it".**
- "Exp 42 (tool attribution, step 6)" is mislabelled. The row that names ToolPainBridge attribution
  and `record_outcome` is **SEM pain → NAc cascade** (Tier 1). Its trigger fires on step 3 (the
  ToolPainBridge migration), step 4 (`record_outcome` / reward) and step 6.
- The trigger table's generic "PainBus / ReactionBus / NAc reward pipeline change" row fires on
  every step from 2 to 6. That means the SEM cascade row plus **row 9** (which also reads the Wire-2
  percept-valence consumer, via H3).
- Add row 219 (Exp 48) and a row 212 note to step 4.

**S4. Missing revive trigger (e): Phase 5's R4 credit-routing audit** (roadmap_1_4 §Phase 5). F1's
reward reaches `credit_node` and lands on `_reward_bias`, which is clamped to `[0, 0.20]`
(`nac.py::credit_node`) and which R4 says the selection surface does not read. F1 is therefore
near-inert today. Once R4 routes that trace credit to selection, it becomes live:
- Anticipated pain and DRIVE reward start moving choices on E1–E3.
- E3's graded-predictor audit names `anticipatory_pre_activate`, whose pre-activated traces
  `TemporalCreditDistributor.distribute` credits. A prediction would then pay its own reward.
- E2's relief store, if it rides the eligibility trace, inherits the same path.

This is the one route by which the plan could silently power E1–E3, against the Parallel-lines
guardrail. Step 4 must land before, or together with, any R4 routing PR.

**S5. Trigger (c) has arguably already fired.** It needs a date anchor. ToolPainBridge attribution
(tool-failure `pain_type`), Wire-4 (`failure_mode`), and `_reaction_to_pain_signal` (which re-parses
`source`) all re-derive meaning today. Reword it to: "a consumer *added after 2026-09-24*…".

**S6. F1 should be a GitHub issue now.** It meets outstanding.md's rule: a verified live defect with
a definite done state (reward distribution never pays ANTICIPATORY; a guard test proven by deletion).
Scope is wider than the plan says. `PerceivedPainAssessor` is wired into every orchestrator sim
(`orchestrator.py`, both action-anticipation and `percept_anxiety_hook`), not only the sandbox.
"Impact unmeasured" is honest, but it can be bounded cheaply:
one `--interactive false` sim with `MAXIM_LOG_FILE`, counting percept_anxiety events against
`pain_chain.distribute_returned` credits, plus a unit replay showing that only the clamped recognition
bias moves. Link step 2 as the fix; parsing `source` would itself be trigger (c).

**S7. Step 1 misclassifies pain arriving from ReactionBus, and the plan does not say so.**
`_reaction_to_pain_signal` drops `context["source"]`. Two consequences:
- A bridged Reaction is classified by a reconstructed `pain_type` alone. `cerebellum_modulator`'s
  execution-failure pain (`source="cerebellum:…"`) becomes EXTERNAL_SIGNAL, which classifies as
  NOCICEPTIVE, which 2S-c records as pain. That covers both memory capture and ToolPainBridge
  `_note_felt_pain`.
- `pain_signal_to_reaction` rewrites `drive:oxygen` as `pain_detector:external_signal`, so no
  ReactionBus consumer can tell air hunger from a hit.

Harmless while strength is write-only. Add trigger: "before memory 2c-3 reads a pain-derived `S`".

## NIT

- **N1.** Step 2 needs an **isolation review**, which is what `Reaction`'s SHAPE-FROZEN marker
  actually requires. A "hivemind/persistence check" is not enough.
- **N2.** F2 omits the deferred shell `pain_bus_bridge_subscriber_unification.md`. That shell
  documents `create_pain_nac_subscriber` × ToolPainBridge overlap, whose correctness rests on
  context similarity being 0.0. Counting `_distribute_reward_from_reaction`, there are four pain
  paths into NAc. Link the shell and fold it into step 6. The `transition_based_drive_pain` B8
  disposition (Phase 5) touches the same attribution path.
- **N3.** The 2S-c PR should state that SEM-cascade discharge holds: the additions are record-only
  and attribution is unchanged.

## Guard spec for step 3 (needed; "byte-identical guard on its inputs" is underspecified)

Per consumer, commit a **golden table generated from pre-migration code** before the migration
commit. It covers the cartesian product of:
- every `PainType`;
- the sources `drive:{oxygen,health,hunger,saturation}`, `""`, a tool source, `cerebellum:*` and
  `perceived_pain:anticipated`;
- origin: PainBus-direct or ReactionBus-bridged;
- `agent_id`: set, `WORLD_AGENT_ID` or `None`.

For each combination, record the accepted flag and the exact delivered value. The migration must
reproduce the table exactly. Each guard must fail when its filter is deleted (the
prove-guard-by-deletion rule).

## Steps → rows tripped

| Step | Rows (graduation ledger) |
|---|---|
| 1 (2S-c) | None, if output-identical. SEM cascade discharge must be stated (N3). |
| 2 Reaction kind | Generic pipeline trigger: SEM cascade, row 9. Isolation review. |
| 3 consumers declare | Exp 60, 61, 62 (Wire-4 allowlist); SEM cascade (ToolPainBridge); row 9 (percept valence); Exp 10 capture (memory consumer, discharge by golden table) |
| 4 F1 / DRIVE | SEM cascade; Exp 42, 45, 52, 219/48, 56, 57; Exp 60/61/62 if DRIVE is excluded; R3 §Outcome; row 212 note |
| 5 adaptation | Everything above that reads intensity; row 9; Exp 60/61/62 (oxygen US magnitude vs θ) |
| 6 bridges | SEM cascade; Exp 42 |

## Verified fine

- `classify_pain` totality is guarded: every `PainType` is iterated in
  `tests/unit/test_memory_2sc_signals.py`. `kind` is a computed property, so it adds no persisted
  shape.
- E1 does not need kind filtering or F1: its argmax reads hunger × fear only. Deferral is correct
  until R4, E2's store or E3's predictor opens.
- There is no conflict with adaptive_nociception or reflex_layering. Both place gain at the producer
  and both name this plan.
