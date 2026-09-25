# Bio-fidelity review — grounded word binding + social referencing (plan v2)

**Verdict: ADOPT WITH CHANGES.** The biology supports the plan's central decision rule: consult only when
your own read of the situation is ambiguous, weight your own experience above the social cue, and expect
indiscriminate copying to fail when the world changes. Most citations are real and correctly attributed.
Two statements attributed to specific papers are wrong, though, and they need fixing before any public
line cites them: the Sorce "unambiguous drop" claim, and the stickleback "copy when uncertain" claim.
Two design points also need changing. First, sensory preconditioning (SPC) alone does not separate
reference from mediated conditioning; an extinction arm is needed. Second, the trigger takes its "stakes"
from the same learned fear that defines "certain". There is one DO-NOT-BUILD, and it applies to Exp A
only.

Reviewer lens: bio-faithful (DESIGN_REVIEW.md). What I read: the plan in full, `docs/agents/bio-memory.md`,
`DESIGN_REVIEW.md`, `hivemind/ingest.py::FOREIGN_FEAR_DISCOUNT`, and the fear min-fold in
`hivemind/merge.py`. Sources: the Sorce et al. chapter reprint, read page by page from the scan (Table 16.1
and Study 4); web-verified abstracts for the rest.

---

## DO-NOT-BUILD

### B1. Exp A: sensory preconditioning cannot, on its own, separate "reference" from mediated conditioning. Add a post-conditioning extinction arm.

**Evidence.** In SPC, fear to S1 (the word) after S2 (the situation) → US conditioning can come from two
sources. (i) An **associative chain at test**: word → situation → fear. This is the plan's "reference". (ii)
**Mediated conditioning during phase 2**: the situation retrieves the word's representation, and that
retrieved representation is conditioned directly. The word then holds its own fear. Both produce the same
result on the plan's DV. The classic separator is Rizley & Rescorla (1972, *J Comp Physiol Psychol*
81:1–11). After higher-order training, extinguish the first-order stimulus. Responding to the SPC stimulus
falls with it (S-S chain). Second-order conditioning does not fall (S-R / direct).

This matters for Maxim specifically. If `NAMES` is traversed in both directions, or if phase-2 pain credit
spreads to concepts linked from the active situation, then the text cluster can acquire `cluster_fear`
directly while the word is absent. That is route (ii), the exact "conditioning" the experiment is meant to
exclude. The binding-ablated arm cannot catch it, because removing `NAMES` removes routes (i) and (ii)
together.

**Fix.** Add a pre-registered **situation-extinction arm** to Exp A. After phase 2, extinguish the
situation's fear by re-exposing the agent to the situation without harm (this is active re-learning, which
bio-memory Wire 4(a) says is the only way `cluster_fear` extinguishes). Then test the word alone.
Reference predicts that fear to the word falls with the situation's. Mediated or direct conditioning
predicts that it persists. Also add an instrument check: after phase 2, before any test, assert that the
text cluster's own `cluster_fear` and `cluster_reward_bias` are **absent**. A non-zero value means route
(ii) fired, and the arm should be recorded as such.

## SHOULD-FIX

### S1. Sorce et al. 1985: "at an unambiguous drop they mostly do not look" is not what the paper shows.

**Evidence (from the paper's text).** Study 4, "Fear signal without uncertainty", used **no drop**: two
shallow sides, N = 23. The "unambiguous" condition was unambiguously **safe**. The result: "Seventeen of
the infants tested in this condition did not look to the mother at all." The 4 who looked and received a
fear face "crossed … in spite of her fear pose" (Fisher exact p = .0004 vs the ambiguous condition).

For a deeper, unambiguously dangerous drop, the authors report only pilot testing without numbers: "If the
'deep side' … was deeper, infants showed fear and avoidance; if it was shallower, they showed no
uncertainty and crossing … occurred without referencing." They report no referencing rate at the deep drop.

The crossing numbers in the plan are **correct**. In Study 1, 0 of 17 crossed under fear and 14 of 19 (74%)
crossed under joy. For completeness: interest 73% (11/15), anger 11% (2/18), and sadness 33% (Table 16.1;
the text says 6/18).

**Fix.** Replace the sentence with: *"when the drop was removed (no ambiguity), 17 of 23 did not look at
the mother, and those who saw a fear face crossed anyway; pilot work reports that a deeper drop produced
avoidance without the need to reference."* For the "clearly dangerous → own perception wins" half, cite
the properly measured result: **Tamis-LeMonda, Adolph et al. 2008** (*Dev Psychol* 44:734–746; 18-month-olds
on slopes). Infants walked safe slopes about 91% of the time regardless of the mother's advice. They
attempted risky slopes only about 12% of the time even when encouraged. They followed advice only on
borderline slopes: 74% attempted when encouraged, 27% when discouraged. This one study supports the plan's
gate on both sides, and "own experience wins" as well.

Add one honest caveat. Sorce et al. **excluded infants who did not reference** (21%), and 40% of all
recruited infants were excluded overall. "12-month-olds consult" describes a selected sample.

### S2. Coolen et al. 2003 is not "copy when uncertain". It is "copy when asocial learning is costly". Re-cite.

**Evidence.** Coolen, van Bergen, Day & Laland 2003 (*Proc R Soc B* 270:2413–2419, "Species difference in
adaptive use of public information in sticklebacks") found that nine-spined sticklebacks use public
information about patch quality and three-spined sticklebacks do not. The authors explain the difference
by nine-spines' weaker armour and higher predation risk, which make personal sampling costly. The
"uncertain / unreliable private information" result is a **different paper**: **van Bergen, Coolen & Laland
2004** (*Proc R Soc B* 271:957–962). In that study the fish ignored public information when their private
information was reliable. They switched to public information as private information aged, and did so
after 7 days without an update.

**Fix.** Cite van Bergen et al. 2004 for *copy when uncertain*. Keep Coolen et al. 2003, but for *copy when
asocial learning is costly*. That is the biological justification for the **stakes** conjunct, which the
plan currently leaves uncited. As a bonus, van Bergen 2004 also supports **staleness** as a trigger: own
information decays in reliability with time since it was last updated. That is a trigger term the plan
does not have.

### S3. "Stakes" and "certainty" read the same signal. Referencing happens in the ambiguous zone, not the acute-threat zone.

**Evidence.** The trigger is stakes (drive pressure, **`anticipatory_threat_need`**, pain just felt) AND
(unfamiliar OR uncertain OR failing). But `anticipatory_threat_need` is produced by the agent's **own
learned `cluster_fear`** (Wire 4). When it is high, the agent is *certain* the situation is dangerous. In
both Sorce (pilot) and Tamis-LeMonda/Adolph, a clearly dangerous situation produced avoidance, **not**
referencing. As written, the conjunction can fire exactly when the biology says not to consult. If the
waiting behaviour is a "cautious hold", it would also delay a learned escape (for example, the drowning
flee that Exp 60 earned).

**Fix.** Take stakes from signals that are **independent of the agent's own learned valence for this
situation**: drive pressure, recent nociceptive pain, and the Coolen-style cost of trial-and-error (a body
near a lethal margin). Keep `anticipatory_threat_need` out of the stakes term, or use it as a
**suppressor**: high own threat means act, don't ask. State in the prereg that a confident own appraisal,
of either sign, blocks the consult. That is the Sorce Study 4 result plus Tamis-LeMonda's safe and risky
slopes.

### S4. "Uncertain" mixes three different quantities. Consult only on reducible uncertainty.

**Evidence.** The plan's "uncertain" branch lists three signals: no history, `recommend_action` below
`min_confidence`, and high outcome variance. These are not the same thing.
- *No history* is **estimation uncertainty** (ignorance). Others can reduce it. It is close to novelty, but
  it is a separate thing: novelty is "I don't recognise this" (EC margin), and no-history is "I recognise
  this but have no valence for it".
- *High outcome variance* over many samples is **risk** (irreducible). Yu & Dayan's (2005) "expected
  uncertainty" is roughly this. A consult cannot reduce it, because another agent's record of a coin-flip
  situation is also a coin flip.
- *"Own information failing"* (recent negative outcomes where the agent used to succeed) is Yu & Dayan's
  **unexpected uncertainty**: a contingency change, signalled by noradrenaline. This is the correct home
  for the bee rule, and the plan already has it as its own branch.

**Fix.** Drop "high outcome variance" as a trigger by itself, or gate it on small n (variance that comes
from few samples is still estimation uncertainty). In the prereg, map each branch explicitly: novelty →
EC margin (recognition; hippocampal/perirhinal familiarity); estimation uncertainty → no or low-n history;
unexpected uncertainty → own-information-failing. Remove "uncertainty drives information seeking" from
the Yu & Dayan bullet. Their paper is about how uncertainty weights top-down vs bottom-up inference
(attention), not information seeking. Cite the information-seeking claim to a review of curiosity and
information seeking (for example Kidd & Hayden 2015, *Neuron* 88:449–460), or state it without a citation.

### S5. Rogers' paradox: Enquist et al.'s "critical" learner is social-first, which is the reverse of the plan's gate.

**Evidence.** Enquist, Eriksson & Ghirlanda 2007 (*Am Anthropol* 109:727–734) define the critical social
learner as one who tries **social learning first** and falls back to individual learning when the copied
behaviour proves unsatisfactory. The plan's gate is **individual-first** (consult only when own
information is lacking). In the literature that is the *conditional* social learner (Boyd & Richerson
1995; Kameda & Nakanishi 2003). Both resolve Rogers' paradox in models, and Rendell, Fogarty & Laland
2010 (*Evolution* 64:534–548, "Rogers' paradox recast and resolved") shows it more generally. Rogers 1988
is cited correctly: it models a changing environment, and at equilibrium social learners give no net
fitness gain.

The plan combines both mechanisms: an individual-first gate, plus an own-experience override, which is a
critical-style check. Exp C's gate-disabled arm removes only the gate, so it tests the conditional half.

**Fix.** Reword the bullet: *"copying pays when it is conditional (individual-first, copy when your own
information is insufficient; Boyd & Richerson 1995) or critical (copy, then check against your own
experience; Enquist et al. 2007)."* Say which half Exp C tests. If the claim is "the gate protects", the
arms must separate the gate from the override. Suggested arms: gate-on/override-off and
gate-off/override-on.

### S6. Owner Q2: only the *stale* corruption arm is Rogers' prediction. The *inverted* arm is a deception test.

**Evidence.** Rogers' mechanism is environmental change: copied information goes out of date. Inverted
valence is misinformation, which is closer to the reliability and deception literature (selective trust,
Koenig & Harris 2005, *Child Dev* 76:1261–1277) and to the security review.

**Fix.** Pre-register **stale** as the Rogers test that carries the claim. Run **inverted** as a separate,
labelled arm. Do not pool the two.

### S7. The biology behind "own experience wins" is confidence-weighted, not absolute.

**Evidence.** Grüter, Czaczkes & Ratnieks 2011 (*Behav Ecol Sociobiol* 65:141–148): when memory and
pheromone conflicted, *Lasius niger* foragers prioritised memory after only a few visits. Later work by
the same group shows that pheromone use rises when memory is weak. Van Bergen 2004 shows private
information losing weight with staleness. Tamis-LeMonda/Adolph 2008 show advice being used exactly where
perception is borderline. In each case the switch is **graded by the reliability of the agent's own
information**, not a fixed "own wins".

**Fix.** Define the merge-time precedence as own weight rising with own confidence (count and recency).
The foreign discount stays as the floor. Otherwise a single bad own trial permanently outranks a
well-supported Oasis entry, which none of the cited animals do. This also makes "own experience wins"
something Exp C can measure.

### S8. The "learned trust" follow-up is cited to the wrong rule, and the missing biology is source reliability.

**Evidence.** The "bee rule" (Grüter & Ratnieks 2011, *Anim Behav* 81:949–954) is about *own* information
failing. It does not describe updating trust in a *source*. Tracking source reliability is a separate,
well-supported strategy: children selectively trust accurate informants (Koenig & Harris 2005), and
payoff- and prestige-biased copying are both in Kendal et al. 2018.

There is also a **many-donor aggregation** question the plan does not address. Fear merges by **min**
(`hivemind/merge.py`: "deepest fear survives"), so one fearful or corrupted donor dominates regardless of
how many others disagree. That is the opposite of conformist transmission (for example Pike & Laland 2010
in nine-spines). The literature supports a negativity bias (infants weight negative signals more heavily),
which argues for min over mean. It does not support one voice over any number of others.

**Fix.** Re-cite learned trust to reliability- and payoff-biased learning. In the consult design, state
the aggregation rule across matching entries: min, mean, count-weighted, or agreement-gated. Declare it an
innate prior (behaviour-tier rule). Also state that Exp C's corrupted arm must corrupt a **minority** of
donors if it is to test conformist robustness.

### S9. Put the binding in the Hippocampus first and consolidate it to the ATL. The hub-and-spoke framing needs a sentence.

**Evidence.** The ATL as a transmodal semantic hub is well supported (Patterson, Nestor & Rogers 2007,
*Nat Rev Neurosci* 8:976–987; Lambon Ralph et al. 2017, *Nat Rev Neurosci* 18:42–55). Two parts of the
plan are not.
- In hub-and-spoke, the **word form is a spoke** and the hub holds **one** concept shared across spokes.
  Two hub nodes joined by a `NAMES` edge is not the model; it is an associative link between two
  concepts, which is closer to a hub-internal association.
- New word–referent bindings are learned **episodically first** (hippocampus: one-shot co-occurrence) and
  integrate into neocortical, lexical-semantic memory **slowly** (complementary learning systems:
  McClelland, McNaughton & O'Reilly 1995; lexical integration after sleep: Davis & Gaskell 2009, *Phil
  Trans R Soc B* 364:3773–3800).

Maxim already has this shape. The 2S-b `capture_from_loop(situation=…)` stores the co-present
`{modality: cluster}` map. Once Stage 1 puts `text` in the situation, **every episode is a word–situation
co-occurrence record**.

**Fix.** State `NAMES` as the **consolidated** form. It is strengthened from hippocampal co-occurrence
episodes, either at the look-back or at `sleep()`/consolidation. It is not a relation written directly to
the ATL at hearing time. At minimum, add a sentence saying the direct ATL write is an engineering
shortcut and that the bio-faithful route is Hippocampus → consolidation. This is also scope pressure in
the plan's own terms: it rides on 2S-b.

### S10. "Refers" overclaims what sensory preconditioning shows.

**Evidence.** SPC demonstrates a **stimulus–stimulus association**. It has been shown in dogs (Brogden
1939, *J Exp Psychol* 25:323–332) and routinely in rats, and nobody takes it as evidence of reference in
the linguistic sense.

**Fix.** Name Exp A's positive outcome *"the word acts through the situation it co-occurred with (an S-S
association), not through its own conditioned value"*. Reserve "refers" for plan prose, with that gloss.
The demo claim sentence already says "reactivates the situation it named", and that wording is fine.

## NIT

- **N1. SPC parameters.** SPC is small and fragile. It is sensitive to the number of pre-exposure pairings
  (few is better; many weaken it). Hoffeld, Kendall, Thompson & Brogden 1960 found an optimum of roughly a
  handful of pairings. Many harmless pre-exposures of the situation also risk **latent inhibition**, which
  would slow the situation's own phase-2 fear acquisition. The prereg should fix the phase-1 pairing count
  and include a check that phase-2 fear to the situation reached criterion.
- **N2. Phase ordering in a live world.** If the teacher ever says the word *after* the situation becomes
  harmful, that pairing is second-order conditioning, not SPC. The prereg should forbid the word from
  being heard in phase 2 or later, except at test.
- **N3. Stigmergy.** Grassé 1959 (*Insectes Sociaux* 6:41–81) coined the term for termite nest
  construction, where the work in progress cues more work. Pheromone trails were folded in later (for
  example Theraulaz & Bonabeau 1999). "Read locally, on need, and fading" fits trails better than
  Grassé's case. Cite trails separately or soften to "stigmergy (Grassé 1959), as in pheromone trails".
- **N4. `FOREIGN_FEAR_DISCOUNT` 0.75 is engineering, not biology.** Observational fear learning in humans
  can match direct conditioning in magnitude (Olsson & Phelps 2004, *Psychol Sci* 15:822–828). Label the
  discount as a prior, not a biological fact. Related: a teacher's word signalling danger is closer to
  **instructed** fear (Olsson & Phelps 2004; Phelps et al. 2001) than to co-occurrence binding. The
  curated teacher design should make sure Stage 2 measures co-occurrence binding and not instruction.
- **N5. Waiting behaviour (owner Q1).** Sorce's pilot describes "infant pauses at the edge and frequent
  looks", so a **bounded cautious hold** is the observed behaviour. Per S3, though, the hold must never
  suppress an escape that the agent's own confident fear selects.
- **N6. What transfers is valence.** Sorce et al. themselves note that fear and anger may act through
  negative hedonic tone, not discrete content. That supports the plan's "a learned valence transfers"
  framing. The answer should stay **situation-scoped** (referential), not a global mood shift. The plan
  already does this.
- **N7. The stated limit is correct.** "No animal queries a remote store" is right. The plan should keep
  the transport labelled as engineering, as it does.

## Citation table

| Claim in plan | Verified? | Correction |
|---|---|---|
| Sorce, Emde, Campos & Klinnert 1985, *Dev Psychol* 21:195–200; 12-month-olds at an ambiguous drop reference the mother | **Yes** | The deep side was tuned by pilot testing to about 30 cm to produce ambiguity. |
| "fear → almost none cross; joy → most do" | **Yes** (understated) | Fear 0/17; joy 14/19 (74%); interest 73%; anger 11%; sadness 33%. "None" is exact. |
| "at an unambiguous drop they mostly do not look" | **No** | Study 4 had *no* drop (unambiguously safe): 17/23 did not look, and the fear-signalled infants crossed anyway. The deeper-drop outcome is pilot-only and says "fear and avoidance", with no referencing data. See S1. |
| Laland 2004 social learning strategies | **Yes** | *Learning & Behavior* 32:4–14. |
| Kendal et al. 2018 | **Yes** | "Social learning strategies: bridge-building between fields", *Trends Cogn Sci* 22:651–665. |
| Coolen et al. 2003 = copy when uncertain | **Misattributed** | Coolen, van Bergen, Day & Laland 2003 (*Proc R Soc B* 270:2413) is a species difference explained by the cost of asocial learning. Copy-when-private-info-unreliable is van Bergen, Coolen & Laland 2004 (*Proc R Soc B* 271:957). See S2. |
| Grüter & Ratnieks 2011, bees follow dances more when their feeder fails | **Yes** | *Anim Behav* 81:949–954. It is about own information becoming unrewarding, not source trust (S8). |
| Grüter, Czaczkes & Ratnieks 2011, *L. niger* prefer memory over conflicting trail | **Yes** | *Behav Ecol Sociobiol* 65:141–148. The preference depends on memory strength (S7). |
| Rogers 1988 | **Yes** | "Does biology constrain culture?", *Am Anthropol* 90:819–831. It depends on environmental change. |
| Enquist, Eriksson & Ghirlanda 2007 | **Yes (citation)**; **framing off** | *Am Anthropol* 109:727–734. The critical learner is social-first with individual fallback; the plan's gate is the conditional (individual-first) strategy. See S5. |
| Grassé 1959 stigmergy "(pheromone trails)" | **Partly** | *Insectes Sociaux* 6:41–81, about termite building. Trails are a later extension (N3). |
| Yu & Dayan 2005 ACh = expected, NE = unexpected uncertainty | **Yes** | *Neuron* 46:681–692. "Uncertainty drives information seeking" is not this paper's claim (S4). |
| Brogden 1939 sensory preconditioning | **Yes** | *J Exp Psychol* 25:323–332 (dogs). Does not by itself separate a test-time chain from mediated conditioning (B1). |
| Karst et al. 2023, wood-wide-web overstated | **Yes** | Karst, Jones & Hoeksema, *Nat Ecol Evol* 7:501–511 (positive citation bias about common mycorrhizal networks). |
| Vogel & Dussutour 2016, *Physarum* habituation transfers on fusion | **Yes** | *Proc R Soc B* 283:20162382. |
| (implicit) ATL as semantic hub | **Defensible** | Patterson et al. 2007; Lambon Ralph et al. 2017. The word is a spoke; binding is hippocampal first (S9). |

## Verified fine

- The core decision rule (consult only in ambiguity, adopt the other's appraisal, own perception wins when
  it is clear) is faithful. Tamis-LeMonda/Adolph 2008 supports it more directly than Sorce does.
- "What transfers is a valence", scoped to the situation, matches the social-referencing account and
  continues Exp 56/61.
- The stated limit (the rule is biological, the transport is engineering) is honest and correct.
- Per-situation refractory: this is engineering. That is fine and is not presented as biology.
- Declaring the trigger an **innate prior**, with a learned-trust follow-up, is the right tier.
- The Exp C design (gated / always / never × clean / corrupted, with a genuine may-fail) is the right
  shape for testing Rogers, once S5 and S6 are folded in.
- The Exp A control set (shuffled, unbound, ablated, no-word) is standard and appropriate. It needs the
  extinction arm from B1 added.
- Excluding the mycorrhizal "warning" story and limiting *Physarum* to merge precedent is well judged.
- Harmless-then-harmful ordering is the correct SPC order (not second-order conditioning), subject to N2.
- A negativity bias in merging fear (tighten-only) has developmental support. The open question is only
  how it aggregates across many donors (S8).
