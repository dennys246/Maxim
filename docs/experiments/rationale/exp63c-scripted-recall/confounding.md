# Exp 63c: confounding lens (design review, 2026-11-09)

Reviewer: confounding lens ([DESIGN_REVIEW.md](../../DESIGN_REVIEW.md)). Read: the 63c DRAFT v1, the 63b prereg and its
three rationale files, `memory/text.py`, `memory/types.py::EpisodicMemory.content_tokens` / `copy_key`,
`hippocampus_retrieval.py::_rank_by_relevance` / `distinct_copies`, `bio_enrichment.py::_query_hippocampus_traced`,
`o19_verdict.py::exp63b_turn`, `simulation/orchestrator.py` (scene manifest pre-trigger), `simulation/narrator.py::
generate_scene_manifest`, `tools/discovery.py`, and 63b's committed stores under `data/63b_carried_recall/`.

## Verdict

**DO-NOT-BUILD as drafted, on two cheap folds.** The scripted design does fix 63b's power problem: in every replay
I ran, R3d finds two to five decisive cue turns unless the AUT repeats the planted facts many times. But the two
**negative controls do not control what they claim**, and the draft classifies a control hit as an instrument fault (C5).
On real tool outputs, turn 3 goes decisive in every replay (6 of 6). The reason is a carried `sense_tools` record that
holds the word "above" from a fixed tool template. The instrument would be right, the control would be contaminated,
and the attempt would read incomplete, which risks an ABORT on correct code. The **power replay** cannot see this: its
"quiet" policy assumes that sense outputs hold no planted word, and they do. Its "worst case" (c) is not the worst
case either. A finding about the opposite direction also matters: under the script, R3d is close to determined by
R2′ + R3′ + the script itself, so a PASS adds little beyond those gates, and the prereg should say so.

The probes are offline replays on the real tokenizer and ranker (`_rank_by_relevance`, `distinct_copies`,
`EpisodicMemory.content_tokens`). The 63c script lines are built as arrival records from 63b's own arrival template.
63b's real phase-1 and phase-2 tool records stand in for the AUT's tool records. Each turn uses the cue path (top 5
after `distinct_copies`, stop at 3) and its carried-removed counterfactual, and G4 condition 3 is computed on non-goal
keys. Scripts: `C:\Users\denny\AppData\Local\Temp\claude\5f0e565c4ebe3bec\296036da-cc5c-4ebb-8a45-b9fe0292b7c8\scratchpad\lens_confounding\p4.py`
(policies), `...\p5.py` (vocabulary overlap), `...\p6.py` (persistent echo). These are design numbers. The proxies are
63b's AUT, not 63c's.

## DO-NOT-BUILD

### D1. The negative controls are contaminated by carried text the script does not control, and a hit is misclassified as C5

**What is carried.** The draft says the control words appear "in no phase-1 line". But the carried pool is not the eight
script lines. It is every phase-1 record, and that includes the AUT's tool records, whose `content_tokens` take in the
tool outcome (whole `outcome.result`), the LLM's free text in `context.active_goal`, and the tool name
(`types.py::EpisodicMemory.content_tokens`). In 63b's store, 15 of 31 records are tool records, and their keys include
fixed template text and world vocabulary.

**Evidence.**
- `tools/discovery.py` (the `sense_tools` "Found N capabilities" output) ends with "pick the most relevant tool
  **above** and call it". So every such carried record holds the key `above`, and `above` is one of turn 3's control
  words. 63b's store holds it on three `sense_tools` records (seq 1, 21, 27).
- Probe p4, with world/tool proxies, turn 3 ("Can you see a window of blue glass above the guard?"): **decisive in 6 of
  6 policies** (quiet, echo-cue, echo-recall ×1 and ×3, echo-all ×1 and ×3). In each case a carried `sense_tools`
  record (non-goal score 1, `above`) displaces a new record with score 0. Without the tool proxies (arrivals only), turn
  3 is never decisive. Turn 6 was not decisive in any run. But "hear" is the opening of
  `embodiment/audio_localization.py`'s listen text ("You hear a sound ..."), so a body that renders listen as text
  would contaminate it the same way.
- `see`, `hear` and `play` are perception and action verbs the AUT's own LLM text uses freely. 63b's narrator line 10
  held `see`. "window" is a plausible entry in an LLM-made dungeon manifest (see S2).

**Why the classification is wrong.** In such a hit, all three G4 conditions are computed correctly over a carried record
that truly shares a non-goal word. That is not an instrument fault. A *real* instrument fault (crediting a goal word,
or a word the records do not share) breaks condition 3 by definition, and a deterministic test catches it better (the
build order already has the deletion probe "a negative-control cue that shares a planted word must turn decisive"). As
drafted, the live control can mostly raise a false alarm. Because C5 makes the attempt incomplete, and the next
attempt re-runs the same script against the same templates, the likely end is an ABORT on correct code. That strands
T1-16, and the owner is holding 1.3.4 for it.

**Fold.**
1. **Pick control words against a measured blacklist at freeze.** Compute `key_tokens` of the control cues and require
   them to be disjoint from (a) every carried and new tool record in 63b's two stores (`content_tokens`), (b) the tool
   templates in `tools/discovery.py` and `sense_presence`'s renderer, (c) `narrator.py::_MANIFEST_SYSTEM`'s few-shot
   example, and (d) 63c's own smoke store once it exists. Drop perception verbs (`see`, `hear`) and anything a dungeon
   manifest plausibly names (`window`, `glass`). Use nouns that no dungeon world or template would produce. Pin the
   check in the harness tests.
2. **Classify by the store, not by the script.** On a control turn, the judge computes `W` = the control's non-goal keys
   ∩ the union of the carried records' `content_tokens`, using the saved phase-1 bytes and G2's port. A decisive control
   turn is **C5 only if the decisive carried id's non-goal overlap is empty**, which is impossible on correct code. If
   `W` is non-empty, it reads **"control contaminated: <words> in <record kinds>"**: descriptive, never C5, never FAIL.
3. **Control turns never count toward R3d.** As drafted, the search runs on turns 2–8 *including* 3 and 6. Once (2) stops
   a contaminated control from being C5, a contaminated control could become the *only* decisive turn and produce a
   PASS on the word `above`. Restrict the R3d search to the cue turns (2, 4, 5, 7, and 8 if kept).

### D2. The power replay's policies model the wrong channel and miss the real worst case, so its go rule can pass a design that fails

**(a) "Quiet" is counterfactual.** The rule "sense-type tool calls only, with outcomes holding no planted word" is false
for this world. 63b's `sense_presence` output names `rusty_key` and `heavy_wooden_door`, which give the keys `key`,
`rusty` and `door`: cue A's `key` and cue C's `door` and `rusty` (p5). The template gives `above` (D1). A replay under
(a) leaves out the very competitors that cause D1.

**(b) The echo channel is not the "reply".** In 63b's store, the AUT's `respond` record holds only
`{cli, fallback, llm, respond}` (seq 15). The text the AUT actually writes into memory is each tool record's
`context.active_goal` (free LLM text), plus its outcomes. A mock that stubs "replies" puts the echo in a field that, in
the real store, carries none.

**(c) Policy (c) is not the worst case.** The cue-each-fact-once rule makes echo-recall structurally harmless. Phase 2
turn 1 surfaces the goal-only distractor and carried tool records, not facts (p4, turn 1, every run). Each fact is
echoed only *after* its own cue, and later cues use other words. So (c) passes almost by construction: echo-recall ×3
still gives 4 decisive cue turns. The real threat is **persistent echo**. 63b's AUT did exactly this: 5 of its 7
phase-2 tool records have an `active_goal` that restates one recalled memory ("I remember waking someone up required
a loud noise", "My experience suggests ..."), turn after turn. The 63c analogue is a phase-1 plan ("take the brass key,
grab the knife under the straw, open the rusty cell door") that surfaces early and is restated in every later tool
call. Probe p6: with one or two such records per turn, **cue C (turn 5) loses**. The top 3 are all new echoes at score
5/7, tied with planted line 4's 5/7, and recency wins the tie (see S1 for why they tie). With an "echo-all" AUT that
restates all four facts, R3d falls to turn 2 only (×1) or to none at all (×3). In that last case only the contaminated
turn-3 control stays decisive.

**Fold.** Build the replay from the real channels. Stub the AUT's text into `context.active_goal` of tool records, and
use **real tool outputs** (63b's recorded `sense_tools`, `sense_presence` and `sense` results, or the real tools run
offline against the real world). Add two policies: (d) **persistent echo** (a phase-1 plan naming every fact, restated
in every phase-2 tool record) and (e) **echo-all** at 1 and 3 records per turn. The go rule: the controls are clean
(D1's W is empty) under every policy, and at least one cue turn is decisive under (d). Report (e) as the stated failure
envelope. Also report, per cue, the size of the new pool and the margin (carried score minus best new score), not only
decisive yes or no.

## SHOULD-FIX

### S1. Goal words handicap the planted lines on the ranker, but not in G4, so a planted line can lose a tie it "should" win

Arrival records carry `context.active_goal = None`: all 16 of them in 63b's store. A planted fact line therefore holds a
goal word only if its *text* does, and lines 2–5 hold none. New AUT tool records hold goal words routinely, through the
LLM's `active_goal` text ("... sneak past the sleeping guard") and through outputs that list the guard. The ranker
scores *all* the cue's keys (`_rank_by_relevance`), while G4 condition 3 scores only the non-goal ones. So the goal word
that every cue carries by design adds to new records and not to the planted line. Probe p6, cue C: the echo holds
`cell, door, rusty, guard, sleep` = 5/7, and planted line 4 holds `cell, door, lock, rusty, squeak` = 5/7. They tie,
the newer record wins, and three such echoes push line 4 out, although line 4 holds 5 non-goal keys to their 3.
**Fold:** give each planted fact line the goal word(s) of its cue (for example, "Beside the sleeping guard, a brass key
hangs ..."). Under the same p6 policies, this keeps cue C decisive (line 4 at 7/7 against 5/7) and leaves every other
cue's result unchanged. Then ranker order and condition 3 agree on what a fact line holds over the goal.

### S2. The stimulus is not fully scripted: an LLM builds the world, its words overlap the planted ones, and the "narrator router is silent" premise is false

`orchestrator.py` (the "Scene manifest pre-trigger" block) calls `narrator.py::generate_scene_manifest(llm_router,
goal)`, and `EntityDesigner(llm_router=llm_router)` designs the entities. Both run on **`llm_router`, the
shared/narrator router**, before the AUT starts, in each phase, and `pre_campaign_turns` does not skip them. 63b's logs:
"Manifest pre-trigger: 4 phrases extracted: ['heavy wooden door', 'snoring guard', 'bed a rusty key', 'exit a
flickering lantern']", then "(1 from index, 3 designed)", in both phases. These words reach the carried and new pools
through every `sense_presence` record. Such records hear nothing, so `copy_key` is None and each call takes its own
place. The manifest's few-shot example ("a rusty iron gate", "a flickering torch mounted on the stone wall", "loose
stones") contains five of the script's planted words: `iron`, `torch` (fact A), `loose` (fact B), `rusty` (fact C) and
`wall` (fact D) (p5). The world also contradicts the script: a rusty key against a brass key, a heavy wooden door
against the cell door. That muddies the descriptive "right memory" measure, and phase 2's manifest can differ from
phase 1's (its prompt takes the NAc biases phase 1 earned).
**Fold:** either (i) choose planted words disjoint from the measured world and template vocabulary (D1's blacklist, but
this needs a margin, not disjointness), and log and compare the manifest phrases of both phases as a descriptive, or (ii)
pin `MAXIM_DISABLE_IMAGINATION=1` in `PROTOCOL["63c"]` as a stated protocol difference from 63b (the environment lens
should rule on what that does to the world). **For OD-c5:** S2's "the narrator's router made zero calls" is false as
the code stands. The manifest and design calls go through that router, so a sole-consumer stamp as written would be
untrue or would refuse every run. The wiring lens should confirm this.

### S3. Under the script, R3d is nearly a consequence of R2′ + R3′; say so, and say what a NOT SHOWN would mean

Every planted line holds 3–5 of its cue's non-goal keys, against 0–2 for any world or template record. So whenever the
carried store loads intact (R2′) and the ranker conforms (R3′), a carried line wins its cue unless at least three new
distinct records match it. The draft's "displacement possible without being guaranteed" is true only in that narrow
sense. That is acceptable for a non-vacuity condition (63b's own S5). But the information in a PASS lies almost
entirely in R2′, R3′ and R3a, and the prereg should state the forecast as near-certain rather than present R3d as an
open test. Conversely, a NOT SHOWN would be evidence about **AUT echo volume**, not about persistence. The
interpretation PR should be required to attribute it, using the echo-load descriptive the draft already adds, before
reading anything into it. Note too that a decisive turn can come from a carried *AUT* record instead of the planted line
(p6, turn 2: the carried phase-1 plan record). The claim permits that. Report it under "right memory", and never
upgrade it to "the planted fact was recalled".

## NIT

- **N1. Fraction against count: no confound.** `_rank_by_relevance` divides by `len(query_tokens)`, a constant within
  one query, so fraction order equals count order. A short carried line cannot win "on fraction", and G4 condition 3
  uses raw counts (`o19_verdict.py::_nongoal_score`). Say this in the prereg to head off the question.
- **N2. The tokenizer fix should be a no-op on the script.** No script word ends in -ed; the -s and -ing strips
  ("hangs" → hang, "hides" → hide, "drips" → drip, "sleeps"/"sleeping" → sleep, "hinges" → hinge) are unaffected by
  silent-e. p4's key tables match the draft's exactly. The fix *does* change AUT and world keys (63b's store holds
  `locat`, `requir`, `discu`, `bas`), so recompute the blacklist (D1) on the fixed code.
- **N3. The goal-only distractor's check is vacuous, and its real effect is on the AUT.** G4 condition 3 excludes goal
  words by definition, so line 6 cannot be decisive unless the judge has a bug. Test that by deletion, not live. Its live
  effect is that it is what surfaces at phase 2 turn 1 (rank 1–2 in every p4 run), so it shapes the AUT's first-turn
  context toward goal-only text, which is part of why echo-recall is benign (D2c). State that as a design effect.
- **N4. Early cues need enough new records.** With no tool records (p4, arrivals only), cue A at turn 2 is *not*
  decisive: the counterfactual holds only the single turn-1 arrival, so no new id is displaced. Real tool records fill
  this gap, but the replay should report the new pool at each cue (D2 fold).
- **N5. The stated reason for "every cue carries a goal word" is off.** Phase-2 records score above zero through their
  *text* (cue lines, the AUT's `active_goal` text, outputs naming the guard), not through a goal field: arrivals have
  none. The rule holds in practice; correct the reason.
- **N6. Shared non-goal words across cues.** `near` is in cues A and D, and `cell` is in line 1 and cue C. After turn 2,
  the new arrival "Where does the brass key hang, near the guard?" is a cue-D competitor (2 against line 5's 4). It is
  harmless at this margin. Prefer disjoint non-goal words across cues so each fight is measured against the world and the
  AUT alone.

## What I verified

- Script key tables: all 16 lines, recomputed with `key_tokens` (p4), match the draft.
- Ranking: `hippocampus_retrieval.py::_rank_by_relevance` (fraction of query keys, recency tiebreak, zero excluded,
  `distinct_copies` before the cut), and `bio_enrichment.py::_query_hippocampus_traced` (cue `limit=5`, stop at 3, goal
  path only under 3, own ids excluded before every limit).
- `o19_verdict.py::exp63b_turn` / `decisive_under`: `carried_cue` is *any* carried id shown by the cue (not only script
  lines), and the counterfactual reruns without the goal path.
- 63b's store (31 records): every arrival record has `active_goal` None. Tool records' `active_goal` is free LLM text,
  and five of seven phase-2 tool records restate one recalled memory. `respond` records carry no reply text.
  `sense_presence` lists the LLM-designed world entities. `above` comes from `tools/discovery.py`'s template.
- The scene manifest and EntityDesigner use `llm_router`, run in both phases, and produced identical phrases in 63b's two
  phases (not pinned).
- Replays p4 and p6: decisive turns per policy, as cited above. The negative control at turn 3 is decisive in 6 of 6
  runs with world proxies and 0 of 6 without them, and turn 6 in 0 of 12.
- Not verified: the 63c mock (`mock63b_session`'s variant does not exist yet), the real AUT's tool mix under a scripted
  stimulus, and whether `MAXIM_DISABLE_IMAGINATION` leaves a usable world.
