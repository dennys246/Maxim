# Exp 63c: BIO-FAITHFUL lens (design review, 2026-11-09)

Charter: [DESIGN_REVIEW.md](../../DESIGN_REVIEW.md). The question: does the design test the mechanism's real job, not a
caricature, and does the metric measure its purpose? I reviewed `docs/experiments/exp63c_scripted_recall_prereg.md`
(DRAFT v1) against its base ([Exp 63b](../../exp63b_carried_recall_prereg.md)), 63b's
[bio-faithful lens](../exp63b-carried-recall/bio-faithful.md), `docs/agents/bio-memory.md`, ledger row T1-16 and the code
(`memory/hippocampus_retrieval.py`, `integration/bio_enrichment.py`, `memory/text.py`, `memory/types.py`). I ran no probes.
Every statement about keying below comes from reading `normalize_tokens` / `_lemmatize` / `_STOP_WORDS` by hand.

**Verdict.** BUILD, with the folds below. The scripted plant-then-cue design is not a caricature of the mechanism. It is
the textbook cued-recall paradigm (study items, intervening material, a cue sharing part of an item), and that paradigm is
what 63b's lens asked for (its S6). It is also more faithful to the row's job than 63b's narrator was. The honest limit is
that the shipped mechanism *is* a stemmed-keyword index with a recency tiebreak. So a PASS shows that index working on
the live, composed, resumed loop. It cannot tell hippocampal recall apart from any keyword index, and the ledger must not
word it as if it could. I found no DO-NOT-BUILD. Four SHOULD-FIXes: the negative controls are defined over the script
when they must be defined over the carried store; the ledger wording has to carry 63b's NOT SHOWN and the planted
stimulus; "at least one" needs reporting as k of 4; and the "Not claimed" list is missing three items.

## DO-NOT-BUILD

None.

## SHOULD-FIX

### S1. The negative controls are defined over the script lines, but the carried store also holds the AUT's own episodes, and recalling those is the mechanism working

**What the draft says.** On turns 3 and 6 the cue's non-goal words "appear in no phase-1 line", so a decisive turn there
"would mean the gate credits something other than shared non-goal words" and is C5 (an instrument fault, so the attempt
is incomplete).

**Why this is wrong for the mechanism.** The carried store is not the script. It also holds every phase-1 record the
AUT's own turns wrote: its replies, its tool calls, their rationale and their outcomes (`content_tokens` reads what it
heard, its goals, tool, objects, rationale and outcome, per `_rank_by_relevance`'s docstring). An agent's own actions and
utterances are its most autobiographical episodes. A cue that shares a word with one of them recalls it correctly.
G4 condition 3 also makes the draft's reading impossible as stated. A decisive turn needs the carried id to score
strictly higher on the cue's non-goal keys, so it needs a carried record that HOLDS one. A decisive turn on a control
can therefore only be either (a) a judge fault, or (b) a carried AUT or tool record that holds a control word. (b) is
correct recall, and the draft would mark a correct run incomplete. Under the leak bar, an incomplete attempt showing a
not-passed gate ends the campaign, and an ABORT hands 1.3.4 back to the owner.

**This is likely, not hypothetical.** Turn 3 has the keys `see`, `above`, `blue`, `glass`, `window`. Turn 6 has `hear`,
`play`, `somewhere`, `flute`. `see` and `hear` are not stop words, and they are among the words a 7B model most often
uses in phase 1 ("I look around and see a torch", "I listen and hear dripping"). Simulated `sense`-type outcomes can hold
them too.

**Fold.**
1. Replace the two high-frequency verbs. Suggested lines: *"Is there a window of blue glass above the guard?"* (keys
   `window`, `blue`, `glass`, `above`, plus `guard`; `is`/`there` are stop words) and *"Is a flute playing somewhere in
   the dungeon?"* (`flute`, `play`, `somewhere`, plus `dungeon`). Recompute both on the fixed tokenizer at freeze.
2. Make the control's validity a judge computation, not a script property. A control turn is **armed** only when no
   carried record in that trace's pool holds any of the cue's non-goal keys. On an armed control, a decisive turn is C5:
   condition 3 cannot hold, so only the judge can be wrong. On an unarmed control, the turn is reported as
   `control_unarmed` with the carried record(s) and their origin (script line, AUT reply, tool outcome). It is excluded
   from the decisive search, which is effectively what the draft does already, and it is never C5.
3. Pre-register that both controls are excluded from the decisive search whatever happens. Then the claimable turns are
   exactly the planted cues plus the closing turn, fixed in advance (see S3).

### S2. A PASS here follows a NOT SHOWN under a different stimulus: the ledger has to say both, and say "planted"

**What a PASS would license.** On the live loop, after a resume, a carried episode takes a cue-path place over a newer
episode when it holds more of the cue's non-goal content keys, and the AUT's own intervening records do not prevent
this. That is the row's mechanism doing its job end to end (persistence, restore, arrival enrichment, the ranker, the
own-record exclusion, copy collapse) on a real resume, with live competitors. The power replay cannot show that, and
neither can `test_cross_session_recall_1444.py`.

**What it would not license.**
- That this happens in conversation the designer did not write. The cue words are planted copies of the plant words, and
  the script was chosen after 63b's narrator produced no decisive turn in 8 turns. That change is a legitimate power fix:
  the diagnosis was a design defect, the stall template. It does mean the two results answer different questions.
  63b's NOT SHOWN is the evidence about narrator-driven conversation, and 63c does not overturn it.
- Anything that separates this from a keyword index. There is no partial or degraded cue, no paraphrase, and no
  competition between similar carried episodes (pattern separation). Interference is resolved by recency only, because
  that is what the mechanism is.

**Fold: the ledger head and the claim's closing sentence.** Suggested:

> EARNED \<date\> (narrow: cued by a fixed script whose phase-2 cues reuse words it planted in phase 1; k of 4 planted
> cues decisive; content-key overlap, not meaning; one 8-turn resume, one model. Under an LLM narrator (Exp 63b) no
> decisive turn occurred.)

Add one sentence to the claim's end: "The cues reuse words the script planted; this shows the ranker's content
preference survives a live resume, not that conversation produces such cues." Also add the same pairing to the T1-16
History when the interpretation PR lands, so 63b's NOT SHOWN is never dropped from the row.

### S3. "At least one decisive turn" is a weak bar under four planted cues: bound it by reporting k of 4 and what the carried id beat

I am not re-arguing OD-c3, only how a PASS is worded.

**Why it is weak here.** Under 63b's narrator, a decisive turn was a non-vacuity event the design could not arrange.
Under the script, a quiet AUT makes one nearly certain. Each fact record holds 3 to 5 of its cue's non-goal keys. The
competitor it must beat (condition 2: the non-carried id that would show with the carried ids removed) will often be a
goal-only record such as phase 2's turn-1 capture. So condition 3 frequently reduces to "the carried record holds at
least one non-goal key and the competitor holds none". That is 63b B1(b)'s "weakest competitor" in a milder form. The
real uncertainty in this run is whether live AUT records (echoes, restatements, tool outcomes) and the live wiring defeat
a design that should win. One decisive turn of four and four of four are different findings.

**Fold.** Keep the gate. Add these to the descriptive per-turn report, never gating, pre-registered so they cannot be
chosen after the fact:
- **k**, the number of the four planted-cue turns (2, 4, 5, 7) that are decisive, and the ledger's "k of 4";
- for each decisive turn, the displaced competitor's non-goal score beside the carried id's (for example "3 v 0"), and
  whether the competitor was goal-only;
- for each non-decisive cue turn, what won (a newer restatement, a tool outcome, a copy) and on what score. This is the
  interference account. A newer restatement winning is the mechanism behaving as specified (recency on a tie), not a
  failure, and the report should say so.

Then turn 8 ("stir") can still be decisive only if some carried record holds `stir`. Say whether a turn-8 decisive
counts toward k. I suggest it does not, so k stays out of 4.

### S4. "Not claimed" misses three things the design now makes salient; the "right memory" descriptive should measure completion

Add to "Not claimed":
1. **That recall occurs under unscripted conversation.** Under Exp 63b's LLM narrator no decisive turn occurred (S2).
2. **Retrieval from a cue that does not share the plant's words.** Every cue reuses the plant's word forms, and the claim
   is lexical (stemmed keys). Paraphrase, synonym and meaning-level cues are untested ("credits a shared word, not a shared
   meaning", which 63b already states for G4, belongs in the list too).
3. **That an original beats its own rehearsal.** An AUT restatement of a recalled fact is a newer episode with the same
   keys and wins on recency. That is specified behaviour, so it says nothing either way about the carried original's
   strength.

**On "the right memory" (kept descriptive, which is correct).** The script makes a stronger, more faithful descriptive
measure available at no cost: **completion**. Does the surfaced carried record hold the answer key the cue lacked? Cue B
(`hide`, `mattress`, `straw`) should surface a record holding `knife` and `flagstone`. Cue A (`brass`, `key`, `hang`)
should surface `hook`, `iron` and `torch`. Cue D should surface `drain`, `grate` and `north`. Cue C is a yes/no question
whose plant adds only `hinge` and `loudly`, so its completion is weak, and the report should say so. Cue-to-whole
retrieval of content the cue did not carry is the operational form of the hippocampal job (pattern completion). It keeps
the planted design from reading as a pure echo of the cue's own words. Report it per cue turn, never gating. Pre-register
the expected answer keys in the key table at freeze. A decisive turn won by the WRONG carried record (for example, phase
1 line 1 on cue C via `cell`) still counts for R3d, but the ledger must not then say "recalled the planted fact".

## NIT

- **N1. Tokenizer dependence (#1620) is mechanical, not biological, and the script barely touches it.** By inspection,
  every plant and cue word keys through the `-s` rule or is uninflected (`hangs`→`hang`, `hides`→`hide`,
  `drips`→`drip`, `sleeps`/`sleeping`→`sleep`, `hinges`→`hinge`). No silent-e `-ed`/`-ing` form is present, so #1620
  should not move the script's key tables. It moves how the AUT's replies key (`hiding` stays `hiding` today; `placed`,
  `moved`, `hidden`). A fix that unifies those forms makes restatements compete *more* with the carried fact, raising
  the echo load. That is the direction the power replay must see. **Fold:** write the replay's echo policies (b) and (c)
  in inflected past and progressive forms ("the knife was hidden", "the key is hanging"), not verbatim cue copies.
  Otherwise the replay never exercises the stemming the fix changes. Also state in the prereg that the claim's "content
  words" means stemmed word forms through `key_tokens`. That keying is lexical either way, and no biological claim rests
  on which stemmer is used.
- **N2. Removing the narrator's reactivity is legitimate for this claim.** The claim is source-blind and origin-blind: the
  ranker reads the arrival's text, not who sent it. The draft's apparatus says a scripted line yields the same `cli`
  arrival record (the wiring lens verifies that). Reactivity mattered only to ecological validity ("a person's
  conversation"), which 63b already disclaimed. It also removes a confound: a reactive narrator could plant a cue's words
  by echoing the AUT, and a script cannot. Keep the draft's sentence, and add that the AUT's replies still enter the store
  as competitors, so the stimulus is fixed but the competitor population is not.
- **N3. R3d's role changed.** 63b's "Not claimed" called R3d "a non-vacuity condition on R3′, not a test of persistence".
  Under planted cues it is closer to the cued-recall event itself. The persistence half still rests on P0–P3 and R2′.
  Add a clause: "R3d remains non-vacuity for R3′, here under cues designed to give it the opportunity."
- **N4. Phase-1 line 6 (the goal-only distractor) is well chosen.** It gives the gate a carried record that can win
  places but never a decisive turn. Report whether it surfaced on each cue turn: a goal-only carried record surfacing
  above a planted fact would be a ranker finding.
- **N5. Attribution S2 is honest from this lens, with one condition.** The claim names a model only because the AUT's
  model generates the competitor records. The ranker itself is model-free, so "everything served in this run came from X,
  and the narrator made no calls" is a sufficient and honest statement of what this claim needs. Its wording must keep
  the scope the draft gives it ("not the AUT's own slot"). It is honest only if the accounting window shows AUT-side calls
  on that server: zero total calls would make "sole consumer" vacuous. Require at least one AUT-attributed call in
  `llm_calls`. Whether S1's slot-level guarantee is still owed under OD-c4 is the owner's call. From this lens it adds
  nothing to the claim's content.

## What I verified

- `src/maxim/memory/hippocampus_retrieval.py::_rank_by_relevance`: the score is |query keys ∩ record `content_tokens`| /
  |query keys|; a zero score is excluded (D1); recency breaks ties; `distinct_copies` runs before `limit` when collapsing;
  no origin is read. The mechanism is a stemmed-keyword index with a recency tiebreak, which is the basis of S2 and S3.
- `src/maxim/memory/text.py::_STOP_WORDS`, `normalize_tokens`, `_lemmatize`, `key_tokens`: `see`, `hear`, `near`,
  `first`, `try`, `back` and `above` are not stop words (`where`, `what`, `does`, `can`, `now`, `still`, `there` and
  `under` are). The `-ing` rule keeps a non-doubled stem only at 4 or more characters (`hiding` stays `hiding`); the `-ed`
  rule produces the silent-e stems #1620 targets (`moved`→`mov`). I hand-keyed the inflections cited in N1.
- `src/maxim/memory/types.py`: `CompressedMemory.content_tokens` keys `heard`, `goal`, `active_goal` and `tool_name`;
  `record_active_goal`. A record's goal fields are keyed, which is why condition 3's goal exclusion matters.
- `src/maxim/integration/bio_enrichment.py::BioEnrichmentPipeline._query_hippocampus_traced` (signature and path order,
  as 63b's lens verified; the #1573 exclusion is described in the brief).
- `docs/agents/bio-memory.md`: the #1573 invariant (own records excluded by identity before every limit; an earlier
  record of the same words is an earlier episode; copy collapse is an innate prior) and F6 (surfaced is not rendered).
- `docs/plans/behavioral_graduation_candidates.md`, row T1-16: STALE since 2026-10-25, its claim text, its `Re-run on:`
  list and the 2026-11-02 walk (#1573).
- 63b's frozen G4 (conditions 1–3) and its "Not claimed" list; 63b's lens B1 (weakest competitor) and S6 (the planted-cue
  follow-up, which this design is).

**Not verified:** what a 7B AUT actually says in phase 1 (S1's likelihood argument is a judgment, which is why the fold
makes the control a judge computation), and the sense tools' outcome text under the scripted path.
