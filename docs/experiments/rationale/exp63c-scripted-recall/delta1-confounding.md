# Exp 63c: confounding lens, delta round 1 (DRAFT v1 → v3, 2026-11-09)

Reviewer: confounding lens ([DESIGN_REVIEW.md](../../DESIGN_REVIEW.md)), a different reader from round 1. Scope: what
changed since v1, and whether round 1's confounding findings (D1, D2, S1, S2, S3) are folded correctly. Read: the v3
prereg, the four round-1 reviews, 63b's prereg (§G1, §G4), `scripts/o19_verdict.py::exp63b_turn`, and the preliminary
replay (`replay63c/`: `simcore.py`, `replay.py`, `validate.py`, `blacklist.py`, `script_v5.py`, `intents_v4.py` and
their outputs). My probes are extensions of that replay, copied into
`C:\Users\denny\AppData\Local\Temp\claude\5f0e565c4ebe3bec\296036da-cc5c-4ebb-8a45-b9fe0292b7c8\scratchpad\delta_confounding\`
(`replay.py` patched with the env switches `R1MODE`, `R2MODE`, `Z1` and `NAUT`, plus the policies `b4`, `cpart` and
`cpart2`; `script_v5x.py`; outputs `*.out`). All runs are offline on today's tokenizer: no LLM, no sim.

## Verdict

**SOUND WITH CONDITIONS: four SHOULD-FIX, no DO-NOT-BUILD.** No stop line: F1 and F2 change the script, and F1 also
changes the forecast's wording. Neither is fail-closed in a class the design already covers. The round still owed can be
**delta-scoped to F1–F4 and run together with the freeze-time replay re-run** that OD-c2 already requires. It does not
need all four lenses.

The control rule and the cue-turn restriction are folded correctly, and I found no way for them to yield a PASS for the
wrong reason (§What I verified). The problems are in the replay's **evidence for its forecast**. Phase 1 of the replay
uses hand-written AUT intents that restate each fact's words. Cue A's win at turn 2 depends on that restatement. If the
phase-1 AUT text is neutral, cue A is never decisive under any policy, so the stated failure mechanism ("a cue loses
only when ≥3 new records echo it") is false for cue A. Even k = 0 is reachable at one AUT record per turn. Moving cue A
off turn 2 fixes this (F1). Cue C also breaks round 1's S1 fold (F2).

## SHOULD-FIX

### F1. Cue A at turn 2 can only win with help from the replay's assumed phase-1 echo, so the forecast's failure mechanism is wrong

**Mechanism.** At phase-2 turn 2, the new pool (own arrival excluded) holds two records: P2.1 (the R3a-view arrival)
and P2.1.aut. Both hold `guard`, so both reach the counterfactual's cue path. G4 condition 2 needs a new cue-path id
that the actual top 3 lacks. With only two new candidates, that happens only when **two** carried records outrank them.
The planted fact is one. The other has to be a second carried record that holds a cue-A non-goal key. In the replay,
that record is P1.2.aut, whose hand-written intent `intents_v4.R1[1]` is "I'll locate the guard by the chair before
reaching for the brass ladle". `replay_v5.out`, policy r, t2: shown = P1.2, **P1.2.aut**, P2.1.aut; without carried =
P2.1.aut, P2.1; displaced = P2.1. The "3 v 0" winner is credited to P1.2, but P1.2.aut is the record that did the
displacing.

**Probe** (`delta_confounding/neutral.out`, `bare.out`, `neutral_nogoal.out`). The phase-1 AUT intents are replaced
with fact-free text ("I'll look around for a way to escape past the sleeping guard.", or "I'll look around."):

| Variant (script v5, 1 AUT record/turn) | r | p | stuckB/C/D | e2 / e3 | b4 (all four facts verbatim, every record) |
|---|---|---|---|---|---|
| replay as committed | 4/4 | 4/4 | 3/4 | 2/4 | 1/4 |
| neutral phase-1 intents | **3/4** | 3/4 | **2/4** | **1/4** | **0/4** |
| neutral, and phase-2 intents with no goal word | **2/4** | 3/4 | 2/4 | 1/4 | **0/4** |
| neutral, 0 AUT records in phase-2 turn 1 | 3/4 | 3/4 | 2–3/4 | 1/4 | 1/4 |

With neutral phase-1 text, cue A is lost under **every** policy, with no echo at all (t2: shown = P1.2, P2.1.aut, P2.1;
nothing is displaced). With two AUT records per turn (`naut2.out`, `neutral_naut2.out`), cue A comes back.

What this does to the v3 text:
- "A cue loses R3d only when at least 3 new records each hold every cue key its fact holds" is false. A cue also loses
  when **too few new records hold the cue's goal word to be displaced** (condition-2 starvation). Turn 2 always suffers
  this unless phase 1 echoes. Cue D at turn 7 suffers it when the AUT's records omit `dungeon` (`neutral_nogoal.out` r,
  and `base.out` cpart t7: the counterfactual's third slot is a goal-path fill, so nothing is displaced).
- "k = 0 is out of reach at about one AUT record per turn" is false: neutral + b4 gives 0/4 at one record per turn.
  b4 is round 1's D2 fold (d) ("a phase-1 plan naming every fact, restated in every phase-2 tool record"), and the v3
  replay never ran it. Its e-policies restate only the 1–3 *upcoming* facts, and in an inflected form that is weaker for
  fact B ("under", not "beneath"; `intents_v4.INFL`).
- "NOT SHOWN ⇒ ≥3 echoing records per turn, or a fault" leaves out the third cause, so an interpretation PR would be
  forced to pick between two wrong options.

**Fold.**
1. **Move cue A off turn 2.** Put a control at turn 2 and cue A at turn 3, so R3d searches turns 3, 4, 5 and 7. Probe
   (`script_v5x.py`, `x_*.out`): neutral phase-1 then gives r/p/e1 4/4, stuckX 3/4, e2 3/4, e3 2/4, b4 1/4, with no
   armed control decisive. Cue A survives every variant, so k ≥ 1 holds under every policy I tried at one record per
   turn.
2. **Add to the go rule** a neutral-phase-1 variant and a goal-free-phase-2 variant of every policy, plus b4. Phase-1
   AUT text is LLM output that the replay cannot know, so the forecast must not depend on how it was written.
3. **Restate the forecast** with both mechanisms: echo (≥3 new records at least tying the fact) and condition-2
   starvation (fewer than 3 − c new records hold any cue key, where c = carried records above them). Add per cue to
   §Report: **the displaceable pool**, meaning new records holding a cue goal word, and the counterfactual's goal-fill
   count. The NOT SHOWN attribution then names one of three causes: echo, starvation, or fault.

### F2. Cue C breaks round 1's S1 fold: it carries `sleep`, and fact C does not

S1's fold was "give each planted fact line the goal word(s) **of its cue**". v3 weakened it to "every fact line carries
a goal word". Cue C ("…squeak while the guard **sleeps**?") has the keys {cell, hatch, padlock, squeak, guard, sleep}.
Fact C holds {…, guard} and not `sleep`, so it scores 5/6. The AUT parrots "the sleeping guard" in almost every intent
(63b's records, and `intents_v4.R2`). So an echo that holds only **3 of the 4** non-goal keys plus both goal words also
scores 5/6, ties the fact, and wins on recency. That is the exact asymmetry S1 was written to remove. Probe (`base.out`):
`cpart` ("I remember the padlock on the cell hatch near the sleeping guard.", every record) loses cue C, k = 2/4.
`cpart2`, the same line without "sleeping", keeps it, k = 3/4. With cue C rewritten as "Does the padlock on the cell
hatch by the guard squeak?" (`x_base.out`), cpart keeps cue C.
**Fold:** make each cue's goal-key set a subset of its fact's (drop "while the guard sleeps" as above, or add `sleep`
to fact C). Add that subset check to `tables.py` and to the harness tests.

### F3. The replay's fidelity claim is narrower than v3 states

- "Matched the shipped `_rank_by_relevance` on 276 calls with 0 mismatches" compares a stdlib port against the shipped
  function. `simcore.ranked` *returns the shipped function's result*, so this checks the port, not the simulator. It is
  not evidence that `simulate` is faithful.
- The simulator's real fidelity evidence is the 8/8 reproduction of 63b's logged traces (`validate.out`, re-run here:
  8/8). **All 8 are cue-only** (`memory_paths` = cue, cue, cue on every trace). So the goal-fill branch is never checked
  against a log, and it is the branch that decides F1's starvation (the counterfactual's third slot).
- `simcore.r3d` re-implements G4. It is not `o19_verdict.py::exp63b_turn`. It uses one ordering, where the judge
  requires every valid tie order, and it treats the graph path as inert, where the judge carries `graph_r`. The ties are
  0 here only because the synthetic timestamps are 20 s apart.

**Fold:** at freeze, drive the replay's decisions through the **shipped judge** (`exp63b_turn`, or its 63c port), on
traces produced by the shipped `bio_enrichment.py::_query_hippocampus_traced` over a real `Hippocampus` filled with the
synthetic records, not through simcore. Then a judge bug and a simulator bug cannot cancel out. Reword the 276-call line
to "port self-check". State that the goal-fill branch is validated by a unit fixture, not by 63b's log.

### F4. Every replayed win is "3 v 0" against goal-word-only filler, so the report must say whether a win was contested

In every decisive cue turn of the committed replay (policies r, p, e1, w, pX), the displaced id holds **0** of the cue's
non-goal keys. It reached the cue path on `guard` or `dungeon` alone. Contested cases (displaced ng ≥ 1, for example
`naut2.out` e1 t7 at "3 v 3") are the ones that lose. So a PASS, as forecast, shows that a carried record holding 3–4 cue
words outranks new records holding only the goal word, which is close to what R3′ (ranker conformance) already
establishes. Round 1's S3 asked for this to be stated, and v3's "near-certain" does so. But a reader who sees the ledger's
"displacing a new memory" will picture a contest that did not happen.
**Fold:** add to §Report per decisive turn: "contested" = the displaced id's ng score ≥ 1. Add to the PASS ledger
wording: "(the displaced new record(s) shared only the goal word)" whenever no decisive turn was contested. This is
descriptive. It does not change OD-c3.

## NIT

- **N1. Control 6's word `hung` is the past tense of fact A's `hang`.** `key_tokens("The ladle hung from a copper hook")`
  holds `hung`, so a phase-1 AUT that paraphrases fact A in the past tense contaminates control 6, and the contaminating
  record is fact A itself. This is fail-closed (contaminated → descriptive), but it likely costs a control. Choose a word
  outside every planted word's inflection family, and check it on the #1620 tokenizer. (`draped` → `drap` today, so
  check any replacement too.)
- **N2. Contaminated controls are classified at store level.** An armed control cannot be decisive on correct code, so
  the rule never fires a false C5. That part is correct. But on a contaminated control, a judge fault that credits a
  carried record with *no* control key goes unseen. A cheap universal assertion closes this: on **every** decisive pair
  (cue or control), the carried record's ng overlap must be ≥ 1, recomputed independently from the saved bytes;
  otherwise C5.
- **N3. The run's own manifest is unchecked against the cues.** The blacklist covers 63b's manifest entities, and 63c's
  manifest is LLM output (identical in 63b's two phases, but not pinned). Pre-commit a rule: if the smoke's manifest
  phrases share a key with a cue line, the script is revised before the run (there are no data yet).
- **N4. Completion is near-tautological.** When the planted fact surfaces, it holds its answer keys by construction.
  Completion is informative only when the decisive record is not the planted line. Say so. Also, "planted fact or AUT
  record" is ambiguous when both are carried and on the cue path (F1's P1.2/P1.2.aut). Report every carried id on the
  cue path and each decisive pair.
- **N5. The tokenizer bends echo strength.** `dripped` → `dripp` today, so a past-tense echo of fact D is weaker than a
  present-tense one. The freeze re-run on #1620 should include past-tense variants of the sweeps.
- **N6. Text drift.** The title says DRAFT v3 but the status block says DRAFT v2. The heading "The script (v2; …)"
  sits over script v5. The forecast's "two or three facts per record cost two" holds only with echoing phase-1 text
  (neutral: they cost three). stuckA is missing from the policy table (its cue is at turn 2, with one echo record before
  it).

## Round-1 folds: status

- **D1 (controls):** (1) The control words are against the blacklist: bell, chapel, ring, silver, banner, hung and
  purple are all absent from the committed 686-key list (checked). (2) Classify by the store: folded, in the coarser
  store-level form (N2). (3) R3d on cue turns only: folded, and correct. **Folded.**
- **D2 (replay channel):** real tool records with `active_goal` rewritten: folded. Persistent echo (stuckX) and echo-all
  (e1–e3): folded. Margins and new pool reported: folded. **(d) as specified (every fact, every record) was not run**:
  that is b4 (F1). The phase-1 channel is hand-written, and the result depends on it (F1).
- **S1 (goal words):** partly folded. Cue C breaks it (F2).
- **S2 (manifest/imagination):** folded through OD-c7, the blacklist and the smoke logging. N3 is the remaining gap.
- **S3 (forecast):** folded in spirit, but the mechanism and the k = 0 bound are wrong (F1), and the uncontested nature
  of the wins is not reported (F4).

## What I verified

- (a) **Controls and the cue-turn restriction.** On correct code, an armed control has every carried record at ng 0,
  so condition 3 cannot hold, and armed ⇒ never decisive. I confirmed this over every variant I ran: 0 armed-decisive
  across 12 output files, 16 policies each. Controls and turn 8 never count toward R3d, so a contaminated control cannot
  produce a PASS. I found no path by which the restriction yields a NOT SHOWN except a cue turn's own loss (F1).
- (b) **Fidelity.** `validate.py` re-run: 8/8 traces reproduced, 0 port mismatches, all of them cue-only (F3). The judge
  logic was compared by hand with `o19_verdict.py::exp63b_turn::decisive_under` (all valid orders, `graph_r`). simcore
  matches it when ties are 0 and the graph path is inert. AUT templates: 63b's real tool records, as claimed. The stuck
  and sweep policies are fair stressors of echo, but not of the phase-1 channel or of starvation (F1).
- (c) **Script v5.** Key tables and answer keys match `tables_v5.out`. No cue shares a non-goal key with another
  phase-1 line. R3d is not vacuous: it can fail under b4 or under starvation. But its wins are uncontested (F4), and cue
  A's slot was an artifact of the replay (F1). Script v5x (control at 2, cue A at 3, cue C without "sleeps", control 6
  without "hung") keeps k ≥ 1 under every policy and variant I ran.
- Not verified: the #1620 tokenizer (undesigned); the real AUT's phase-1 text and actions per turn under a scripted
  stimulus; the run's manifest.
