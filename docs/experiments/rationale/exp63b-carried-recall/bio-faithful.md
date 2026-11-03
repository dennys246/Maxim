# Exp 63b: BIO-FAITHFUL lens (design review, 2026-11-02)

Charter: [DESIGN_REVIEW.md](../../DESIGN_REVIEW.md). Question: does the design test the mechanism's real job, not a
caricature? I reviewed `docs/experiments/exp63b_carried_recall_prereg.md` (DRAFT v1) against Exp 63's prereg and its
[bio-faithful lens](../exp63-carried-recall/bio-faithful.md), `docs/agents/bio-memory.md`, `docs/plans/recall_1_3_4.md`,
the T1-16 row and the code. I changed no tracked file except this one.

**On the evidence used here.** The numbers below replay shipped retrieval over Exp 63's committed stores, or run the
shipped pipeline on a synthetic store. The stores are old-code bytes. They are used only to show what the design would
read. Nothing here moves a row. The stores also differ from what the re-run will produce (B2), so every replay number
is a shape, not a forecast. Scripts: `bio_probe.py` to `bio_probe5.py` in the session scratchpad.

**The verdict in one paragraph.** The claim's persistence half and R3a are sound. The new R3d does not test cue-dependent
recall: it would credit goal words and the weakest competitor. The prereg's evidence for R3d and G4 omits the own-capture
skip, and with the skip applied its main statements stop being true. The claim's "full candidates" is false for a carried
memory holding the turn's exact words, and that gap is a shipped defect in the row's own mechanism. All three are fixable
before build. None needs a new phase.

## DO-NOT-BUILD

### B1. R3d credits the goal's words and the weakest competitor, not what the cue selected

**The real job.** Since Session R the mechanism's job is cue-dependent episodic recall: what was just said retrieves what
shares its content. Encoding specificity gives the faithful prediction for a resumed session:
- an old episode is reachable;
- with equal overlap it loses to a newer one (recency here, retroactive interference in the biology);
- it wins only when it holds what the cue asks for and the newer episodes do not.

So "a carried memory outranks a new memory" is the right form only when the win comes from the cue's content. An old
memory beating a new one is not in itself a sign of recall. As drafted, R3d does not require that, in two ways.

**(a) The cue here is mostly the goal.** The narrator paraphrases the goal, so the cue's overlap with the store is nearly
all goal keys (`dungeon`, `escape`, `guard`, `sleep`). The goal words sit in the heard text (`cli_input`, `transcript`,
`observations`: 207 of 356 records hold all four), not in the label: `active_goal` is empty on 342 of 356. Records hit by
a cue key outside the goal keys, against all cue hits:

| Turn | cue hits | hit on a non-goal key |
|---|---|---|
| 1 | 83 | 0 |
| 2 | 99 | 33 |
| 3 | 145 | 1 |
| 4 | 168 | 82 |
| 5 | 242 | 17 |
| 6 | 251 | 23 |
| 7 | 11 | 11 |
| 8 | 271 | 39 |

The plan's limit (§3, F4) names this for the goal path. The same thing happens on the cue path, by a different route.
Turn 1 is the extreme: the cue's other two keys (`tell`, `locat`) hit nothing, so the 0.60 is purely goal overlap and the
order is recency. Turn 7 is the only turn where the cue holds no goal key. There D1 does real selection (11 of 273 records
are candidates), and that is what the mechanism looks like when it works.

R3d also counts a carried id "from the cue or goal path". A carried id shown by the goal path shows goal fill, not recall.

**(b) "Ranks strictly above at least one" names the weakest competitor.** A shown carried record needs to beat only one
new record with a lower non-zero score. In a store where new records score anywhere from 0.25 to 1.0, that tail almost
always exists, so the condition reduces to "a carried id was shown". On the store collapsed to one record per arrival (B2)
the drafted R3d is true on turns 2, 3, 4 and 6, and turn 3 is a win on goal keys alone.

**The fix.** Replace R3d's decisive test with:

> A turn is decisive when, in every valid tie ordering, a carried id shown **by the cue path** ranks strictly above a
> visible non-carried record that scored above zero on the cue, **and scores strictly higher than that record on the cue's
> keys outside `key_tokens(goal)`**. A turn whose cue has no key outside the goal cannot be decisive. A goal-path win is
> reported as `goal_fill_win` and never counts.

This is computable from I1's `cue_text` and G2's frozen ranker, with no new `src/` field. On the collapsed store it leaves
turns 2, 4 (`next`) and 6 (`location`); on the committed store with the skip applied, none. So the gate has power and is
not automatic. Tightening it cannot block 1.3.4: no decisive turn reads `NOT SHOWN`, which is PARTIAL, and only `STALE` and
`BROKEN` block a cut. The honest limit stays: the gate credits a shared word, not a shared meaning, and `next` is a template
word (S3).

### B2. The evidence table omits the skip and reads a store the shipped code should no longer produce, and G4's reasoning rests on it

**The skip was left out, and it changes the conclusions.** Applying `_add_memory`'s skip over the committed store
(`bio_probe3.py`):

| Prereg statement | With the skip applied |
|---|---|
| "on every replayed turn [the cue] holds at least 3 records, so the goal path never runs" | False. On turns 2, 4 and 8 the cue's top 5 are all exact copies of the turn's words, so the cue surfaces nothing and the goal path supplies all 3 |
| "The replay gives turns 2 and 8 as candidates" | Turn 8 shows no carried id at all. Turn 2 shows carried ids by the goal path |
| "Exp 63's R3d would read `NOT SHOWN` by construction" | False on this store: turn 2's goal-path carried ids beat 15 new records |

Both "candidates" were carried records holding the turn's exact words, which the skip removes. So the table's two most
useful rows were counting exactly the records the mechanism excludes.

**The store shape is pre-#1414.** Exp 63 ran 2026-10-03. [#1414](https://github.com/dennys246/Maxim/issues/1414) (one input
is observed once, not once per tick) came later. The committed store holds 356 records for 12 distinct heard lines: 134
copies of one line, 82 of another, 34 of a third. The carried 84 are 7 lines. Consecutive records are identical except for
id and sequence. So:
- on 6 of the 8 turns of the skip-applied replay, the 3 surfaced ids are copies of **one** heard line; on turns 5 and 7
  they are 3 `sense_tools` records with no heard text;
- `visible`, `new visible` and the cue-hit counts in the table are mostly re-capture artefacts;
- the shipped store should be far smaller. Collapsing runs of same-words records gives 14 records at turn 1 and 28 at
  turn 8, against the table's 84 and 356.

The collapsed store is an estimate, and I did not establish the shipped per-utterance count on a real sim. `take_user_input`
and the guard `tests/unit/test_talk_rejection_1413_1414.py::test_one_input_is_published_once_while_an_llm_answer_is_pending`
say one, for the slot they cover.

**The fix.**
1. Regenerate the table with the skip applied, on the committed and the collapsed store, and label both.
2. Restate G4's motivation. What holds on both shapes is that the cue path now runs first, so Exp 63's goal-only reading
   no longer describes what surfaced. The "by construction" argument goes.
3. Add to build order step 3 (the real short sim): report records per utterance and distinct heard texts among surfaced
   ids. If an utterance still produces several records, stop and look before the run, because the cue-starving in B3 then
   fires on ordinary turns.

### B3. "Full candidates" is false for a memory holding the turn's exact words, and that gap is a shipped defect

**The claim says** the carried memories "take part in recall as full candidates" and that the surfaced set is "exactly what
the shipped ranking over carried and new memories dictates". **The code does not do that.**
`_query_hippocampus_traced` compares a record's `input_text` to the turn's words as text, and the skip is called "this
turn's own capture". It matches any record with those words, from any turn or session. It also runs after `recall(limit=5)`
has already cut the list. Two consequences, both measured on the shipped pipeline (`bio_probe4.py`):

- **An earlier identical episode is never recalled.** Ask a question, then ask it again with nothing else stored: 0 surfaced.
  Hearing the same words in a later session is the plainest cross-session case there is (recognition, "you asked me this
  before"), and it is the one case excluded.
- **Repetition starves the cue.** Store a fact, then the question stored k times. For k = 1, 3 and 4 the fact surfaces. For
  k = 5 and 6 it does not: 0 surfaced, because the 5 slots hold skipped copies and the fact is ranked sixth. An agent asked
  the same thing 5 times forgets the answer, by the path the row certifies.

Exp 63's own stimulus exercises this. The stall detector hands the narrator a fixed line
(`_build_stall_nudge_example`: "What is your next step toward: ..."), the narrator sent it on several turns, and phase 1
heard one line that phase 2 repeated verbatim on turn 8.

**The conformance gate does not catch this,** and it should not: G1 reproduces the skip faithfully, so R3′ holds. The gap
is between the claim's words and the code. The skip reads text, not origin, so "reads no memory's origin" stays true. But
"full candidates" does not.

**Root cause and scope.** Own-capture identity is keyed on words, and the cut runs before the skip. These interact, so
neither alone is the fix:
- Widening `limit` only moves the threshold from 5 to a larger number. That is a band-aid.
- Keying the skip on capture identity (the capture this arrival made) restores the earlier episode. But then 3 earlier
  identical asks score 1.0 and fill every slot, so the answer is still crowded out. A proper fix also has to decide what
  repeated same-words records mean among the surfaced set.

The brief states the current behaviour as an invariant ("A record holding exactly the turn's own words ... is never
recalled"), and `tests/unit/test_recall_memory_seams_1444.py` pins it, so changing it is an owner decision, not a fold.

**The owner's choice, before the prereg freezes:**
- **(A, the strict option and my recommendation)** Fix the root cause first, under an issue with its own review, and run
  on the fixed code. Any later fix would fire the row's own `Re-run on:`, so running first spends the rig twice.
- **(B)** Run on the shipped code and narrow the claim as S4 words it: a memory holding the turn's exact words is skipped,
  whichever session it came from. File the defect. This keeps the 1.3.4 schedule and states a known recall gap in the
  release.

Under either option, the sentence as drafted must not ship.

## SHOULD-FIX

### S1. R3a's E counts records the mechanism can remove, which risks a false BROKEN

E counts "the carried records the cue or the goal scores above zero". The skip and the top-5 cut can remove them. If phase
2's first narrator line repeats a phase 1 line (likely: one goal, and the first resume line is written from it), the reachable
count can be below `min(3, E)`, R3a fails, and the row reads `FAIL`, `BROKEN`, and blocks the release, on a correct run.

**The fix.** Take E from G1's recomputation: the count of carried ids that recomputation surfaces at that view (and
`NOT SHOWN` at 0). R3a then asserts "all surfaced are carried, and the live count equals the recomputed one". Keep the
known-answer role. On a first turn with only carried records this is the faithful reachability test, and with the fix a
FAIL can only mean a real fault.

### S2. Separate cue selection from goal fill in the report

R3c from the base design (descriptive, never gating) should grow per turn, all from I1 and G2:
- ids surfaced by path, and `goal_fill_win`;
- cue hits, and cue hits on a non-goal key (the table in B1);
- `cue_minus_goal_shown`: would the same ids surface if the goal's keys were removed from the cue? This is the direct
  answer to "did the cue add anything?";
- distinct heard texts among surfaced ids (B2);
- carried share, and the turn it first reaches 0 (the base design's prediction that the share falls as competitors build up).

Without these, a PASS cannot be told from goal-word recency in the ledger.

### S3. The narrator is the same path but a stand-in stimulus: say which

**The path is real.** A sim's `cli` percept goes through `loop_perception.py::take_user_input`, `_addresses`
(`ADDRESSED_SOURCES`), `publish_arrival` and the `agent_loop.py` enrichment with `user_input=_addressed`, the code a
person's typed input takes. It is not a parallel channel, so the wiring half is fair evidence.

**The stimulus is not a person's.**
- The narrator is an LLM holding the goal, so its cues repeat the goal's words. A person's cue in Talk has no such
  overlap with the store.
- It is mostly questions. In the committed store, surfaced records with heard text are the narrator's own earlier lines,
  and the only other surfaced records are `sense_tools` results. The AUT's reply is not in them (no `respond`/`speak`/`say`
  call in Exp 63, per the row).
- So a PASS here means "the narrator's rephrasing of its earlier line surfaced". Near-duplicates are therefore real overlap,
  not a mechanism fault, and they do not make a gate false. They do change what a PASS means: on the collapsed store, 2 of
  the 3 decisive turns are won on `next`, a word from the stall template's family.
- The stall nudge is a harness line copied by the narrator, not free narration.

**The fix.** Keep the protocol. Add one sentence to the Protocol section naming all of this, count nudge-derived lines in
the report, and word the claim as S4 does. Do not exclude near-duplicates with a similarity threshold: that invents a
constant with no evidence behind it, and the mechanism has no such notion.

### S4. The claim sentence and "Not claimed"

"Not claimed" is honest about what it lists, and the pointer to `test_cross_session_recall_1444.py` is right. It is
incomplete. A reader of "takes part in recall across sessions" and "a carried memory outranks a new memory" can reasonably
take away "the agent remembers across sessions". Exp 63's lens asked for the operational meaning to be stated, and the new
text drops it. Add to "Not claimed":
- that a cue selects the right carried memory among carried ones (S6);
- that recall is driven by a person's words, or by meaning, rather than shared content keys;
- that a memory holding the turn's exact words is recalled (B3, under option B);
- that a surfaced memory is used: surfaced is not rendered (F6, the brief's known gap), and rendered is not acted on (T1-2).

Suggested claim, option B wording in brackets:

> After a resume, the earlier session's Hippocampus store is back with its content unchanged. Text arriving on the addressed
> path (here a simulation narrator's, not a person's) ranks carried and new memories alike: by how many of its content
> words each holds, then by recency, with no reading of a memory's origin. [A memory holding exactly the turn's words is
> skipped, whichever session it came from.] On the first resume turn what surfaces is carried. On at least one later turn
> a carried memory takes a place above a new one on the cue path because it holds more of the cue's words outside the goal.
> This is one 8-turn resume on one model. It does not show that the agent remembers what a person told it, that it uses
> what surfaced, or that a memory outlives sleep.

Change the PARTIAL label to match: "carried outranking new on the cue's own words not shown".

### S5. Re-run triggers miss what the claim now depends on

The list covers the ranker, enrichment, keying, the skip, save/restore, resume and record shape. The claim now also depends on:
- `loop_perception.py::take_user_input`, `_addresses` and `ADDRESSED_SOURCES` (what is captured at arrival, what is enriched);
- `EpisodicMemory.text_fields` / `content_tokens` and `memory/text.py::text_values` (what a record holds, which a change
  can alter with no shape change).

Add them.

### S6. Nothing tests a cue selecting the RIGHT carried memory: record that as a decision

B1's fix gives "a carried memory wins on the cue's content". It does not give "the cue picked the correct one of several
carried memories", which is the job at its fullest. This protocol cannot show that: the narrator states no distinct facts.
A planted-cue run (a scripted scenario through `simulation/scenario_source.py`, a fact in phase 1 and its cue in phase 2) would.
That is a new protocol and a new claim, so it needs all four lenses, and it is not this row's. `test_cross_session_recall_1444.py`
covers the same behaviour on the shipped composition, as a test.

**The fix.** File it as the follow-up named in the prereg, so that the row's narrowness is a decision and not an omission.

## NIT

- **N1.** The Mechanism line lists the graph path. It is inert on the percept path (the `_query_hippocampus_traced`
  docstring; ledger T3-7). Say so, since a reader takes "graph" for associative recall.
- **N2.** R3d's "Note the asymmetry" is correct and useful. Keep it, and add the reason in biology terms: recency gives new
  memories every tie, so only content can favour an old one. That is why content is what the gate must read (B1).
- **N3.** On a claim-moving page, cite F4 precisely: in this sim the leak is through the narrator's words. Talk's route (a
  constant goal matching records) is a different one.
- **N4.** `recall` touches all 5 returned records, including ones the skip then drops. P3's bookkeeping exemption already
  covers it. Do not read `access_count` as "recalled" in the report.

## Answers to the review's questions

- **Is the claim the real job?** Its persistence half and "origin-blind" half are real. "Outranks" is the right form only with
  B1's fix. As drafted it is a caricature: it would pass on goal words.
- **Should the design care about a cue selecting the right carried memory?** Yes, as far as this protocol can (B1), and the rest
  is recorded as not shown (S6).
- **Separate cue from goal fill?** Yes: in the gate (B1) and the report (S2).
- **Narrator as user?** Same path, stand-in stimulus (S3).
- **Near-duplicates?** Exact ones are removed by the skip, including genuine earlier episodes, and that is the problem (B3).
  Paraphrases do surface. They are real overlap and distort the meaning of a PASS, not the gates (S3).
- **Is "Not claimed" honest?** Honest but incomplete, and the claim above it overstates (B3, S4).

## Verified (file::symbol)

- `src/maxim/integration/bio_enrichment.py::BioEnrichmentPipeline._query_hippocampus_traced`: path order; `recall(query=text,
  limit=5)` is cut before `_add_memory` runs; `_add_memory` marks seen, then skips on `_same_words(_input_text(mem)) == own_words`,
  with no reference to turn or capture; `goal_results[:3]`.
- `src/maxim/integration/bio_enrichment.py::_same_words`: text compare (NFKC, casefold, whitespace).
- `src/maxim/memory/hippocampus_retrieval.py::_rank_by_relevance`: overlap fraction, zero excluded, recency tiebreak, no origin.
- `src/maxim/memory/types.py::EpisodicMemory.text_fields` / `content_tokens` / `input_text`; `active_goal` is a label on
  top of the text fields.
- `src/maxim/memory/text.py::key_tokens`, `text_values`.
- `src/maxim/runtime/loop_perception.py::take_user_input`, `_addresses`, `ADDRESSED_SOURCES`; the comment there states the
  narrator plays the user.
- `src/maxim/runtime/agent_loop.py` §1.2 (`EnrichmentContext(... user_input=_addressed)`, `bypass_gate=True`).
- `src/maxim/agents/perception_agent.py::PerceptionAgent.publish_arrival`, `propose_intent` (the #1414 comment).
- `src/maxim/simulation/conversational_source.py::inject_cli`: `source="cli"`.
- `src/maxim/simulation/stall_detector.py::_build_stall_nudge_example`: the template line, passed to the narrator as `send_message(text=...)`.
- `tests/unit/test_recall_memory_seams_1444.py::test_the_cue_finds_the_answer_and_never_the_turns_own_words`: pins the
  skip with one own copy, which is why neither consequence in B3 shows there.
- `CHANGELOG.md`, #1414 entry; `docs/agents/bio-memory.md` (the "never recalled" sentence, F6's known gap).
- Replay of the prereg's table reproduces its visible, cue-hit and top-score columns over the committed stores with the
  supplied `replay.py`. Everything in B1, B2 and S3 adds the skip, the goal path and the record-level detail.
- `docs/experiments/data/63_carried_recall/20261003_173617/aut_hippocampus.json`: 356 records, 12 distinct heard lines; records
  76 to 83 are identical but for id and sequence.

**Not verified:** the per-utterance record count on a real shipped sim (B2 step 3 settles it), and how often the re-run's
narrator repeats a line. The collapsed store is my approximation: it keeps the first of each run of same-words records.
