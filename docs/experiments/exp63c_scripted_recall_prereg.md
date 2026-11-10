# Exp 63c: carried memory displaces a new one on the cue path, under a scripted stimulus (pre-registered 2026-11-10)

> **Status: frozen (v5, owner decision 2026-11-10: freeze before the 63c judge port, the forecast labelled
> simulator-judged). The freeze re-run has run on main d40eb4c0, with the tokenizer fix
> #1620 merged (#1624): the power replay, the key tables, the blacklist and the judge-free half of the fidelity check
> (§Power replay, "Freeze result"). A scoped confounding follow-up on delta round 1's F1–F4
> ([followup-f1-f4](rationale/exp63c-scripted-recall/followup-f1-f4.md)) found all four RESOLVED and asked for the
> text folds made here. Earlier rounds, all folded: the four-lens design review
> ([confounding](rationale/exp63c-scripted-recall/confounding.md),
> [bio-faithful](rationale/exp63c-scripted-recall/bio-faithful.md),
> [wiring](rationale/exp63c-scripted-recall/wiring.md),
> [environment](rationale/exp63c-scripted-recall/environment.md)) and a delta round
> ([delta1-confounding](rationale/exp63c-scripted-recall/delta1-confounding.md),
> [delta1-wiring](rationale/exp63c-scripted-recall/delta1-wiring.md)). Owner decisions OD-c1 to OD-c7 are taken. The
> judge half of the fidelity check (R3d re-derived through the shipped 63c judge port) is owed before the smoke; until
> it runs, the forecast's k values are simulator-judged. No harness code and no data exist.** It re-tests ledger row T1-16 after Exp 63b ([prereg](exp63b_carried_recall_prereg.md); attempt 1 at
> e841abca; data #1615) read **NOT SHOWN**: every gate held except R3d, which found no decisive turn, because five of
> 63b's eight phase-2 cues were the stall template, whose words outside the goal phase 1 barely used. The owner HELD
> 1.3.4 for this experiment (2026-11-09) rather than ship under a release exception.
>
> **Base:** Exp 63b's gates G1–G5, I1, R2′, the verdict table, OD1, OD2 and its `Re-run on:` list stand here unless this
> file changes them. 63b's prereg is **bound by hash** alongside this one through a new `PROTOCOL["63c"]["base_preregs"]`
> key, read by both `bound_blobs` and `_evidence_records.py::_rejudge_with`'s bound-file set (which today requires exactly
> {judge, harness, prereg}, so a fourth path would make the evidence gate refuse every 63c verdict at the ledger PR), with
> a check in `protocol_problems` and a test. The prereg lint's parent chain is not changed: it times when a prereg landed
> against the data, and `parent_token` is a global rule (delta wiring 3). 63b's gate text is not restated.

## Owner decisions

- **OD-c1 (2026-11-09), stimulus: a scripted narrator.** Phase 1 plants facts, and phase 2's cues reuse their words. The
  agent under test (AUT) stays live.
- **OD-c2, strict: the tokenizer is fixed first** ([#1620](https://github.com/dennys246/Maxim/issues/1620), silent-e
  inflections). The key tables, the vocabulary blacklist and the power replay are recomputed on the fixed code at freeze.
- **OD-c3, the claim threshold stays at least one decisive turn.** Rigour comes from the controls and from the
  descriptive report (§Report), not from a higher count.
- **OD-c4/OD-c5, attribution: S2′.** `aut_model_attribution` stays `not_established`, and the gates keep that literal.
  63c adds a check: every call entry on every router serial in the attempt's accounting, plus every harness
  `/v1/models` read after the first OK one, names the PROTOCOL model, and that GGUF's SHA-256 is pinned in
  `PROTOCOL["63c"]` and re-read on the rig at preflight. This establishes "everything served in this run was model X"
  by machine check, not by operator attestation. **The comparison, field by field** (delta wiring 1–2): on every
  `llm_calls` entry of both phases, `backend_kind == "peer"`, `identity_source == "local_file"`, `identity` equals the
  pinned SHA-256 (from 63b's committed `identity`, `3e0039fd…ccf9`, re-read on the rig), and the basename of
  `served_id` equals the GGUF's name, with an empty `served_id` failing. `identity` is per server, hashed once by
  `serving_provenance.py::_owned_identity` and joined onto each entry, so it proves which file that server loaded, not a
  per-call fact. The comparator is a stdlib copy in the judge, never `_served_model_matches`. Harness `/v1/models` reads
  are judged only from the first OK read to the last OK read: the sim's server stops at `atexit`, before the process
  exits, so a failed read after the last OK one is reported, not judged. It does not establish that the AUT's own slot was X (B0b, #1227/#1228,
  is not a 63c gate). The bio-faithful lens notes that the ranker is model-free and the model only writes the competitor
  records, so the claim's content does not depend on the stronger form. Channels outside the accounting
  (`backend.warmup`, the post-report roundup, another process on :8100) are covered by the quiet-rig rule and named.
- **OD-c6, #1616 is fixed before the smoke.** A new campaign is the window that issue allows for changing the instrument.
- **OD-c7, imagination and the scene manifest stay on** (the shipped path). Planted words avoid their vocabulary
  (§Vocabulary), and their output is logged per turn (§Smoke).

## Claim

> After a resume, the earlier session's Hippocampus store is back with its content unchanged. Text arriving on the
> addressed path (here a fixed script's, not a person's) ranks carried and new memories alike: by how many of its
> content words (stemmed keys) each holds, then by recency, with no reading of a memory's origin; an episode stored many
> times takes one place, the newer copy on a tie. On the first resume turn what surfaces is carried. On at least one
> later turn a carried memory takes a place on the cue path, displacing a new memory, because it holds more of the
> cue's words outside the goal. This is one 8-turn resume on one model, cued by a script whose phase-2 cues reuse words
> it planted in phase 1.

**Forecast** (the freeze re-run, §Power replay; simulator-judged until the judge half of the fidelity check runs): given
R2′ and R3′, and about one AUT record per turn (63b's rate), at least one decisive cue turn is expected under every
policy run on script v4. At two AUT records per turn the stated echo level is: echo-everything (b4) gives k = 0, e1
gives 2/4, e2 and e3 give 1/4; that can only produce a NOT SHOWN, never a false PASS. A cue can be lost in two ways, and a NOT SHOWN must name which:
- **echo:** new records restate the fact, and the third record holding every cue key the fact holds ties it, so the
  newer copy wins on recency;
- **starvation:** too few new records carry the cue's goal word, so G4 condition 2 finds no displaceable new id. This is
  why the first cue moved behind a control; it also bears on cue D when the AUT's records omit `dungeon`.

Any other cause is an instrument or path fault. The interpretation must say which, using §Report (each cue's
displaceable pool and echo load). The earlier wording ("k = 0 is out of reach"; "NOT SHOWN ⇒ echo or fault") is
withdrawn: with neutral phase-1 text and every fact echoed in every record, the v3 script reached k = 0.

**Not claimed:** 63b's list, plus:
- recall under unscripted conversation (Exp 63b, with an LLM narrator: no decisive turn);
- retrieval from a cue that does not share the plant's words (paraphrase or meaning);
- that a cue selects the RIGHT carried memory (reported descriptively as completion; a selection claim is new and owes
  four lenses);
- that an original memory beats the AUT's own later restatement of it (a newer restatement winning on recency is
  specified behaviour).

**Ledger wording on PASS** (bio-faithful S2): `EARNED <date> (narrow: cued by a fixed script whose phase-2 cues reuse
words it planted in phase 1; k of 4 planted cues decisive; content-key overlap, not meaning; one 8-turn resume, one
model. Under an LLM narrator (Exp 63b) no decisive turn occurred.)`

## Apparatus

Verified by the wiring lens against origin/main d4bbb360:
- **The path:** `orchestrator.start_simulation_mode(pre_campaign_turns=...)` → `campaign_runner.py::run_precampaign_turns` →
  `bridge.send_and_wait(text)`, with salience 0.8 and novelty 0.7 by default, as for 63b's narrator lines. The ENTER line
  and its `text_len`, the anxiety hook, `inject_cli` with source `"cli"`, `take_user_input` (addressed),
  `publish_arrival` and the own-capture id, the trace's `cue_text` and `own_ids`, and agent `sim_aut` are identical. The
  narrator makes no call (`test_pre_campaign_turns_run_without_a_narrator` pins `narrator_calls == []`), and stall
  nudges go only to the narrator's source. Resume works as in 63b. (A person typing into a sim meets this path at
  `take_user_input`, not at `send_and_wait`.)
- **Not used:** `--sim <file>.yaml` (ScenarioSource) runs the CLI loop with no bridge, no `sim_aut`, no resume and no cap.
- **The engine change** (its own issue, review round and `changelog.d` fragment): a flag on the goal-string sim path,
  proposed `--sim-script <file>` (repo-relative in argv), that loads the file with **`experiment.load_campaign_turns`**
  (no new loader) and passes `pre_campaign_turns` together with `--resume-sim`. It is **accepted only on the plain
  goal-string call** (a whitelist), and requires `--sim-max-turns` explicitly. It **refuses**: a script whose length
  differs from `--sim-max-turns` (the runner ignores the cap); a goal that selects an arc (an arc probe that raises
  refuses, never reads as "no arc"; today that call sits inside an `except Exception: pass`); `--dm` (already a silent
  no-op on the goal-string path, so refusing it is about honesty, not the bridge); interactive mode; `--research`; a YAML
  `--sim`; the legacy `--sim agent`, benchmark and `--continuous` paths; a whitespace-only line.
  `load_campaign_turns` silently drops an empty `cli_input` line, so the length refusal and the pins catch that.
  The run ends `finish_reason == "completed"`; a `max_turns` relabel would fake a guard event that never fired (wiring S1).
- **Pins:** one script per phase, so two pins: each phase's `scenario_content_sha256` in `PROTOCOL["63c"]` (not
  `scenario_sha256`, which hashes host paths; wiring S2; the delta probe confirmed `orchestrator.py` passes
  `pre_campaign_turns` to `capture_start_provenance`, and the hash is stable and path-free). The judge imports only the
  standard library, so the 16 lines are also literals in `PROTOCOL["63c"]`; a test checks that the loader's texts equal
  them and that each phase's `_content_sha256` equals its pin. Each turn's `cue_text` equals its line exactly. The
  harness spawns the sim with `cwd=REPO_ROOT`, and preflight loads and hashes both scripts before the marker, so a
  missing or edited script cannot cost an attempt.
- **Same as 63b:** goal `escape a dungeon with a sleeping guard`; two phases of 8 turns; phase 2 resumes phase 1; a fresh
  data home; `--interactive false`; `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (`llm.profile mistral-7b`) at `llm.n_ctx 8192`,
  set through `maxim config`; big-mac-mini, quiet, with the harness on the rig, never on the leader. `--sim-run-full-turns`
  is dropped (it does nothing without a narrator).
- **Judge sites:** `63c` is handled at every `experiment_of` site in `o19_verdict.py` and `o19_rerun.py`, including the
  six that fall through silently today (wiring S4: C4′, C4″, the judge's C5, `exp63_attempt_problems`, `_run_phases`,
  `mock_phase`), plus new 63c dispatches for C1's turn check and `mock_phase`'s `max_turns`, `EXP63_TURNS`, `required_files`, the served-reader preflight, the smoke, `LEAK_GATES["63c"]` and
  the test exclusion in `test_o19_rerun.py`. `"63"` and `"63b"` keep their behaviour. The tag ruleset 24345734 already
  covers `refs/tags/o19/**`.
- **Known, not on the recall path:** a working-directory-relative state write outside the fresh home (#1264/#1567).

## Vocabulary (fixed at freeze, on the fixed tokenizer)

A **blacklist** is computed and committed with the prereg. It is the union of the keys of: the tool-output templates
(`tools/discovery.py` and every tool the AUT can call), the scene manifest's few-shot example, 63b's manifest entities in
both phases, the stall and diversity templates, and every `content_tokens` key of 63b's 31 phase-2 records. It also includes the
other tool classes of the AUT's 30-tool registry (error strings included, so it errs on the safe side),
`format_thought_response`, and `bodies/base_humanoid.yaml` and `npcs/guard.yaml`. **No key of any script line, and no
control key, may be in it.** On the fixed tokenizer (main d40eb4c0) it holds 669 keys (686 before #1620) and no
script line or control collides; it forced v2's `gate` → `hatch` and `stone` → `brick`. Answer keys are stored as
`key_tokens` (the freeze caught the replay storing them raw, which read completion "no" on facts B and D), so a
tokenizer change cannot make completion read "no" silently.

**Every fact line carries a goal word** (confounding S1). Arrival records hold no goal, while the AUT's records hold the
goal words, so without one an echo ties the fact on the full cue and wins on recency even when the fact holds more
non-goal words.

## The script (v4; key tables recomputed at freeze)

`key_tokens(goal)` = {dungeon, escap, guard, sleep} (main d40eb4c0).

Script v4 (`script_v5x.py`), keyed on main d40eb4c0 (the replay's code and outputs are committed under
`docs/experiments/rationale/exp63c-scripted-recall/replay/`). Every line has 0 blacklist collisions (669 keys). v2's `gate` and `stone` collided (the guard YAML and the manifest few-shot's
"rusty iron gate", "stone wall"), and its non-fact lines held blacklisted words.

**Phase 1:**

| # | Line | Role | Non-goal keys | Answer keys (stored as `key_tokens`) |
|---|---|---|---|---|
| 1 | You lie in a cold damp alcove. The guard is asleep in the dungeon and you wish to escape. | scene | alcov, asleep, cold, damp, lie, wish | – |
| 2 | A brass ladle hangs from a copper hook beside the guard's chair. | fact A | besid, brass, chair, copper, hang, hook, ladl | copper, chair |
| 3 | The guard hid a small knife beneath the straw mattress. | fact B | beneath, hid, knif, mattress, smal, straw | knif, smal |
| 4 | The cell hatch by the guard has a cracked padlock whose chain squeaks. | fact C | cel, chain, crack, hatch, padlock, squeak, whose | chain, crack |
| 5 | Water drips from a crevice in the north bricks into a drain grate in the dungeon. | fact D | brick, crevic, drain, drip, grat, north, water | drain, grat, north |
| 6 | The guard is still sleeping in the dungeon. | goal-words distractor | – | – |
| 7 | What comes to mind? | neutral | com, mind | – |
| 8 | Sum up your intention in a sentence. | neutral | intention, sentenc, sum | – |

**Phase 2:**

| # | Line | Role | Non-goal keys | Phase-1 lines sharing them |
|---|---|---|---|---|
| 1 | You are back in the dungeon and the guard is still sleeping. | R3a's view | back | none |
| 2 | Is a silver bell ringing in the guard's chapel? | control | bel, chapel, ring, silver | none |
| 3 | Where does the brass ladle hang by the guard? | cue A | brass, hang, ladl | 2 only (3) |
| 4 | What is beneath the guard's straw mattress? | cue B | beneath, mattress, straw | 3 only (3) |
| 5 | Does the padlock on the cell hatch by the guard squeak? | cue C | cel, hatch, padlock, squeak | 4 only (4) |
| 6 | Is a purple banner draped in the dungeon? | control | banner, drap, purpl | none |
| 7 | Where does water from the crevice drip in the dungeon? | cue D | crevic, drip, water | 5 only (3) |
| 8 | The guard stirs. What do you do now? | closing | stir | none |

v4 changes from v5 (delta confounding F1, F2 and a NIT; script `script_v5x.py`): **the first cue moves from turn 2 to
turn 3**, behind a control. At turn 2 the new pool holds only two records, so a displacement there needed a second
carried record in the top 3, and in the replay that was text the replay itself had written. **Cue C drops "while the
guard sleeps"**, so its goal keys ({guard}) are a subset of its fact's; with `sleep`, an echo holding three of the four
non-goal keys plus "sleeping guard" tied the fact and won on recency. **Control 6's `hung` became `draped`**, because a
paraphrasing AUT would key "hung" with fact A's "hang".

No cue shares a key with another phase-1 line or an earlier cue, and no answer key is in its cue. **Each cue's goal keys
are a subset of its fact's goal keys**, checked mechanically at freeze (`tables.py`) and by a harness test. Imagination over v5
extracts only "…guard" phrases, which hit the already-live `npcs/guard`, so the script adds no new entity.

Rules the script follows:
- Each fact is cued once, and before any later cue reuses its words. The AUT's reply to a cue lands after that cue's
  query, so it cannot compete on the same turn.
- No line is the stall template, and no line asks the AUT to recall.
- Every cue carries a goal word, so G4 condition 2 has displaceable new records (records score through their text).
- An answer key is a key the fact holds that its cue lacks; completion (§Report) reads them.

## Gates

63b's, with these changes:
- **R3d's decisive search runs on the four cue turns (3, 4, 5, 7) only.** The control turns 2 and 6 never count, so a control word
  cannot be the only decisive turn (confounding D1, environment F1). The closing turn 8 is not searched.
- **Controls.** The judge decides each control's status from the recomputed carried store:
  - **armed** when no carried record holds any of the control cue's non-goal keys. An armed control that reads decisive
    is C5 (an instrument fault: G4 condition 3 cannot hold there);
  - **contaminated** otherwise, reported descriptively and never C5 (the gate is then computing correctly on a shared
    word, which is not a fault).
- **A decisive pair's sanity check:** on every decisive turn the carried record shares at least one of the cue's non-goal
  keys; otherwise C5 (G4 condition 3 cannot hold without one).
- **C1** reads `finish_reason == "completed"`, `turns == 8`, and the content pin.
- **C4/C4′** compare the new argv and the pins.
- **S2′** (OD-c5) is a C4-class check: a mismatch makes the attempt incomplete.
- **Deletion probes** (build order step 5): remove the own-capture exclusion; plant a control word in a carried record,
  so an armed control turns contaminated; scramble the origin; drop a fact's goal word, so the echo tie appears.

## Report (descriptive, never gating)

Per cue turn:
- decisive or not;
- the carried record's non-goal score against the displaced id's (e.g. "3 v 0"), and whether the win was **contested**:
  the best pair's displaced id (the pair with the largest margin) held at least one of the cue's non-goal keys. The
  harness emits the flag per decisive turn. If no decisive turn was contested, the PASS ledger
  wording says "uncontested: every displaced record shared only the goal word";
- the cue's displaceable pool: the new records that score above zero on the cue;
- whether the decisive carried record is the planted fact or an AUT record;
- **completion**: does a surfaced carried record hold the turn's answer keys?
- for a non-decisive cue, what won and why (a newer restatement winning on recency is specified behaviour);
- the echo load: new records holding each planted key at that turn.

Also: k, the decisive cue turns out of 4; each control's status; whether the goal-words distractor surfaced; and the
imagination and designer events and entities per turn.

## Power replay (before freeze; a design step, not data)

Offline, on the fixed tokenizer, with no rig and no LLM. It uses the shipped `key_tokens`, `content_tokens`,
`_rank_by_relevance` and `distinct_copies`, and **63b's real tool records as the template for the AUT's records**: about
one action record per turn, with tool-description outputs and the AUT's text in `context.active_goal`. Phase-1 records
are modelled the same way, so the carried pool holds them too. Policies:
- **(r) 63b-real:** each turn's AUT record is 63b's record of the same tool, with its `active_goal` rewritten as a
  plausible intent for the scripted line;
- **(p) persistent echo:** each phase-2 record's intent restates one recalled fact (63b's AUT did so in 5 of its 7
  phase-2 records);
- **(e1, e2, e3) echo-all:** one, two or three facts restated per record, in inflected form ("the knife was hidden");
- **(w) world:** the manifest entities appear in every sense output;
- **(pX) and (stuckX):** fact X surfaces before its cue (on the R3a view, a control turn or a goal-path fill) and is then
  restated in every later record, which is what 63b's AUT did. Policy (p) alone cannot stress R3d, because each fact is
  cued once, so "the most recently surfaced fact" is always the one just cued;
- **sweeps:** fact X restated in the first k records.

Report the decisive cue turns per policy, the winner and margin per cue, the new-pool size, and each control's armed
status. - **(b4) echo-everything:** every fact restated in every phase-2 record (round 1's D2 case, not run in v3);
- **variants applied to every policy:** neutral phase-1 AUT text (the real phase-1 intents are LLM output nobody can
  predict, so the replay may not credit hand-written text that mentions a fact), and phase-2 AUT records without the
  cue's goal word (starvation).

**Go rule:** powered when every policy, under every variant, gives at least one decisive cue turn, and no armed control
is decisive under any policy. The prereg states the result as a forecast, including the echo and starvation levels at
which R3d fails.

**Fidelity check** (delta confounding F3), in two halves:
- **judge-free half, DONE at freeze** (`replay/fidelity.py`): every `simulate()` call the replay makes, with and without
  the carried records, is re-run through the shipped recall: the pool loaded into a real Hippocampus (`load_state`),
  the cue enriched by the shipped `BioEnrichmentPipeline` with the turn's own capture excluded, and the logged
  `enrichment_trace` (ids and paths) compared. It covers retrieval once enrichment fires (`bypass_gate=True`), not the
  novelty gate;
- **judge half, owed before the smoke:** R3d re-derived by the shipped 63c judge port (every valid tie order, and the
  graph path) on those traces. Until it runs, the decisive verdicts below are the simulator's.

**Freeze result** (2026-11-10, main d40eb4c0 with #1620; script v4 = `script_v5x.py`, intents v4; outputs in
`replay/`). The re-run on main is byte-identical to its preview on the #1620 branch.

| Variant | k per policy (r p e1 e2 e3 b4 cpart cpart2 / pA pB pC pD / stuckA stuckB stuckC stuckD) | min k | contested wins |
|---|---|---|---|
| base | 4 4 4 3 2 1 3 3 / 4 4 4 4 / 4 3 3 3 | 1 | 4 of 53 |
| neutral phase-1 | as base | 1 | 4 of 53 |
| neutral phase-1, no goal word in phase 2 | r = 3, the rest as base | 1 | 4 of 52 |
| bare phase-1 (extra) | as base | 1 | 4 of 53 |
| two AUT records per turn (extra) | 4 4 2 1 1 0 3 3 / 4 4 4 4 / 3 3 3 3 | 0 (b4) | 5 of 46 |
| neutral phase-1, two AUT records per turn (extra) | as the row above | 0 (b4) | 5 of 46 |

- **Go rule met:** under every named variant every policy gives at least one decisive cue turn, and no armed control is
  decisive under any policy or variant (both controls are armed on every run). The "extra" rows are stress tests
  outside the go rule; the two-records row is the stated echo level in §Forecast.
- Every decisive cue turn's winner is the planted fact's own record, never an AUT record (F1: no win rests on text the
  replay wrote), and completion holds on every decisive turn (53 of 53 in base).
- **Contested wins** are rare: e2/e3 at cue B (3 v 2) and cpart/cpart2 at cue C (4 v 3); the two-records rows add pA
  at cue A (3 v 2). Under the realistic policies (r, p, pX, stuckX) no win is contested, so a PASS will probably take
  the "uncontested" ledger wording (§Report).
- Each cue's goal keys are a subset of its fact's (A, B, C {guard}; D {dungeon}); a harness test repeats the check.
- **Fidelity, judge-free half:** 256 `simulate()` calls in each of base, neutral + no goal word and two-records, 0
  mismatches against the shipped recall; the goal-fill branch was taken 12–15 times per variant. Deletion probes:
  dropping the simulator's top cue hit is caught on 45 of 48 calls; a goal-fill probe on 1 of 48 (the probed policies
  rarely use that path).

**History.** The v3 preliminary replay (2026-11-09, before #1620, script v5) read GO but was superseded (delta
confounding F1, F3, F4): cue A's turn-2 wins needed a hand-written phase-1 intent naming the ladle (with neutral text it
was lost under every policy, and b4 reached 0/4); every replayed win was uncontested; and its "0 mismatches over 276
calls" was a port self-check (a port against the shipped ranker that the simulator calls anyway), not evidence of the
simulator's fidelity. Script v4 moved cue A behind a control, dropped "sleeps" from cue C and changed "hung" to
"draped".

## Smoke (`o19_rerun.py smoke --exp 63c`, never computes R3a or R3d)

63b's smoke checks, plus:
- 8 ENTER lines whose `text_len` and `cue_text` equal the script lines;
- every AUT action inside its own turn window, and no AUT call in flight at any ENTER;
- per-turn timing;
- a turn-1 trace with `own_ids`;
- `hippocampus_size` rising by the arrival record plus the AUT's records each turn;
- zero narrator submits;
- the peak prompt size against `n_ctx`;
- the S2′ check;
- the manifest phrases and entities for each phase, and per-turn imagination and designer events;
- per turn, the echo channels that can show the AUT a phase-1 fact before its cue: NAc predictions (the `plan:<intent>`
  signatures under "Predictions from past actions"), ATL "Related concepts", goal-path fills, and **the cue path itself on
  turns 1, 2 and 6** (the main way a fact surfaces before its cue). These are how a stuck echo would start. Every check
  here is observable in today's logs (delta wiring (d)): `sim_exec` ENTER lines, `enrichment_trace`, `actions.jsonl`,
  `peer_backend_call` (time and latency, for in-flight intervals; no agent field), `sim_enrichment` nac/atl lines (joined
  to their trace by order and time), and `sim_imagination` phases including `design`. The manifest's text is not logged,
  only its phrases through imagination events;
- if the run's own LLM-made manifest contains a planted or control key, that is reported, and a control it touches
  reads contaminated by the armed rule; it is never C5.

## Attempts, verdict, ledger

63b's verdict table, with kind `exp63c_verdict`, markers under `refs/tags/o19/63c/`, and data in
`docs/experiments/data/63c_scripted_recall/`. PASS → T1-16 EARNED (narrow, the wording above) at the run date; NOT
SHOWN → no raise (OD2); FAIL → BROKEN; ABORT → STALE stays, and the release decision returns to the owner.

## Build order

1. #1620, the tokenizer fix: its own PR and review round, and the ledger trigger walk.
2. On the fixed tokenizer: the vocabulary blacklist, the key tables (and the goal-subset check), the power replay with
   every policy and variant, and the fidelity check through the shipped judge; the scoped follow-up on delta
   confounding F1–F4; then frozen on its own PR, with the replay committed under
   `rationale/exp63c-scripted-recall/replay/`, before any data.
3. The engine change (`--sim-script`) and #1616's fix, each with its own issue and review round.
4. The harness: the `63c` key at every site, S2′, the controls' armed status, the report, the mock variant, red gates
   (`xfail(strict=True)`), the deletion probes, the three-lens code round, a full suite.
5. The smoke at the merged commit, then `preflight`.
6. The run on the rig, the data PR (never squashed), the interpretation PR by a different reader.
