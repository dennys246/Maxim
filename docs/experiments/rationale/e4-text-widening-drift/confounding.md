# E4 design review — CONFOUNDING lens

Reviewed: `docs/experiments/protocols/e4_text_widening_drift_preregistration.md` (draft, 2026-10-05),
against `DESIGN_REVIEW.md`, `docs/lessons/ec-centroid-drift.md`, Exp 24 (`24_roy_paraphrase_diagnostic.md`
+ `scripts/diagnose_roy_paraphrase_collapse.py::run_cell`), `data/roy_paraphrase_pairs.json`,
`similarity/ec.py::pattern_complete_or_separate` + `_NodeMatrix.scan`, `decisions/nac.py::credit_node` /
`get_threshold_overrides`.

Question this lens asks: can COLLAPSE (`s ∈ A(0.2) ∧ s ∉ A(0.0) ∧ s ∉ I(0.2)`) fire, or NO COLLAPSE
be returned, for a reason other than "reward widening admitted foreign strings whose averaging moved
the rewarded centroid"?

## What the rule gets right (verified, not findings)

- **Under a frozen centroid the triple is structurally unsatisfiable.** `scan` admits a node iff
  `sim(s, stored_row) >= override`; the isolated arm has exactly one node, so with a frozen prototype
  `s ∈ A(0.2) ⇒ sim(s, prototype) >= 0.24 ⇒ s ∈ I(0.2)`. So any COLLAPSE string implies the rewarded
  centroid MOVED. The rule does isolate centroid movement. The confounds below are about WHAT moved it.
- **Competition cannot manufacture a COLLAPSE**: `scan` takes argmax over eligible rows, so other
  nodes can only take strings away from the rewarded node in the sequential arm. (It can manufacture a
  false NULL — finding 4.)
- **"b=0 already absorbing" is handled for membership**: `s ∉ A(0.0)` excludes every string the node
  takes without reward. Exp 24 suggests this set is large (food strings matched at 0.68/0.70/0.57/0.62
  even at 0.40), which matters for finding 1, not for this exclusion.
- **Ordering of credit is not a confound here**: the rewarded node is the first string, credited
  before any other string is encoded, so no string is ever encoded against it un-widened.

---

## DO-NOT-BUILD

### 1. The isolated arm is not the fair counterfactual: it lacks the node's LEGITIMATE (b=0) members, so COLLAPSE can fire from baseline running-mean movement + widening, with no reward-admitted member involved

**Reason.** `I(0.2)` tests `s` against the bare first embedding (`"you sense food nearby."`). The
sequential rewarded node's centroid at the moment `s` arrives also contains members that join at
**b = 0** — at minimum the pair-mate `"the smell of food fills the air."` (pre-EC cosine 0.754, joins
at any b), and per Exp 24 most of pairs 02–04 (cosines 0.57–0.70 against the food centroid, above
0.44). So a string `s` with `sim(s, mean(baseline members so far)) ∈ [0.24, 0.44)` and
`sim(s, first embedding) < 0.24` satisfies all three clauses — yet no reward-admitted string touched
the centroid. The node's centroid was moved by un-rewarded, already-tolerated accumulation (the 0.44
production behaviour the lesson accepted); reward only lowered the gate onto that moved centroid.
That is "widening applied to an already-averaged concept prototype", not "widening-driven drift",
and the rule's consequence (an EC change to `pattern_complete_or_separate`, firing the EC completion
row's trigger) would be bought by a confound. The pair-mate half of this is near-certain to exist;
whether it decides depends on unseen cosines, which is exactly why it must be designed out, not hoped
away.

**Proposed prereg fix (keeps the owner's absorption-beyond-baseline rule; redefines `I` fairly).**
Replace the ISOLATED definition with a *baseline-anchored* counterfactual:

> **R1, ISOLATED (decides)**: for every string `s` not in pair_01, the reference centroid is the
> running mean of the embeddings, in walk order, of the strings that sit in the rewarded node in
> **R1 SEQUENTIAL at b = 0** and precede `s` in the walk (always including both pair_01 strings).
> `s ∈ I(b)` iff `cos(s, reference centroid) >= base − b` (the same override value, computed the same
> way). This is widening applied to the node as it would be without any reward-admitted member.

This is computable from the recorded embeddings and the b=0 assignment, needs no EC modification,
and makes `s ∉ I(0.2)` mean precisely "admitted only because a reward-admitted member moved the
centroid". Keep the current bare-prototype isolated arm as a reported (non-deciding) column so the
pair-mate / baseline contribution is visible. State in the rule: "the counterfactual differs from the
sequential arm ONLY in the centroid contributions of strings admitted under the override".

(An equivalent alternative: a sequential arm where override-admitted matches are counted but not
averaged — i.e. the very fix the COLLAPSE branch would ship. That needs a harness-side EC variant and
is a wiring-lens question; the replay above avoids it.)

### 2. The positive control does not exercise THIS meter — it passes on pure widening, which the rule explicitly classifies as non-drift

**Reason.** The control requires RA at b=0.2 to put "at least one foreign string into a node it does
not reach at b=0". At an effective threshold of 0.24, with fixture cross-class distractor cosines up
to 0.297, that is satisfied by widening alone — the `I(0.2)` case the rule says is "widening working
as designed". It shows the override reaches the EC (useful: it would catch a leaked
`MAXIM_NAC_REWARD_BIAS_DISABLED`, which empties the overrides), but it does not show the meter can
see the decisive asymmetry `A \ I`. A NO COLLAPSE could still be a blind meter (e.g. a harness that
records the isolated arm wrongly, canonicalises node ids wrongly, or reads assignments before the
running-mean update).

**Proposed prereg fix.** Keep the current check, renamed "override-reaches-EC check". Add:

- **Drift positive control (must fire, else exit 4):** run the same meter — sequential vs the
  isolated/replay counterfactual, absorption into the first-formed node — at Exp 24's known-drift
  setting (`pattern_complete_threshold = 0.40`, no reward). Exp 24 recorded 19/20 sequential strings
  in node 1 with isolated pairs 10/10 clean, so ≥1 string must land in `A_seq \ I` there. If it does
  not, the meter cannot see drift and no verdict is issued.
- **Drift negative control (must be empty, else exit 4):** R1 at b=0.2 with `"text"` added to
  `frozen_centroid_modalities`. As shown above, the triple is structurally empty under a frozen
  centroid; a non-empty result means the harness's `I` or assignment bookkeeping is wrong (or a
  competition/ordering artefact is leaking into the meter).
- **Cross-arm identity check:** R1 and RA at b=0 are the same run (no credit) — assert identical
  assignments. Cheap, catches an accidental credit at b=0.

---

## SHOULD-FIX

### 3. "Foreign" is not defined at a pre-declared granularity; same-class food strings can fire COLLAPSE

**Reason.** "Strings that belong to other concepts" is operationalised as "strings outside pair_01",
but pairs 02–04 share pair_01's fixture class (`food_overlap_2pc`) and Exp 24 shows them sitting
0.57–0.70 from the food centroid. A COLLAPSE carried only by, say, `"your stomach feels satisfied."`
is arguably concept-level generalisation, not cross-concept contamination; a COLLAPSE carried by a
thermal/pressure/social string is unambiguous drift. Deciding the granularity after seeing which
string fired is the post-hoc reframing the house rules forbid.

**Proposed prereg fix.** Add to Metrics and Decision rule: "Each COLLAPSE string is reported with its
fixture class. The verdict is labelled **COLLAPSE (cross-class)** if any COLLAPSE string's class
differs from `food_overlap_2pc`, else **COLLAPSE (within-class only)**. Concept granularity for the
rule is the PAIR (as the fixture defines it), so both labels are COLLAPSE; the label travels into the
engram_formation.md E4 entry so the fix's urgency is read correctly." (If the owner prefers the rule
itself to require a cross-class string, that is their call; this lens only requires it be fixed
before data.)

### 4. Competition can hide drift (false NULL); absorption under-measures centroid movement

**Reason.** In the sequential arm a string that the drifted rewarded centroid WOULD admit may go to a
closer non-rewarded node (`scan` argmax). The isolated arm has no competitor. So a drifted centroid
can produce NO COLLAPSE. The NO COLLAPSE branch closes #911 "with the numbers; no code".

**Proposed prereg fix.** Report, per string per arm, `sim(s, rewarded centroid at encode time)`, its
effective threshold, and the winning node. Add a reported (non-deciding) set
`E(b) = {s : rewarded node eligible for s in R1 SEQUENTIAL, s ∉ I(b), s ∉ A(0.0)}`. Add to "What this
does not claim": "A NO COLLAPSE with non-empty `E(0.2)` means the centroid drifted far enough to admit
foreign strings that a competing node took instead; it is reported as such and #911's closing comment
must say so."

### 5. The drift metric has no baseline and is dominated by legitimate averaging

**Reason.** "Cosine between the rewarded node's first embedding and its final centroid" moves for the
pair-mate alone (two strings at 0.754 → centroid ≈ 0.94 to the first) and for every b=0 member. As
specified it cannot distinguish reward-driven drift from the accepted baseline running mean.

**Proposed prereg fix.** Report `drift(b)` alongside `drift(0.0)` and the drift of the baseline-anchored
reference centroid of finding 1 at end-of-walk, and state that only `drift(b) − drift(0.0)` is read as
reward-attributable movement. Still never decides.

### 6. One string can decide; margins to the threshold are not recorded, so a threshold tie decides silently

**Reason.** The rule fires on a single string. `scan` uses `sims >= thresholds` on float64; the
decisive comparisons are at 0.24 (sequential) and 0.24 / 0.44 (counterfactual, baseline). A string at
0.2401 vs 0.2399 flips with a sentence-transformers / torch / device change, and the in-process
determinism check (two runs, one process) cannot see cross-environment variation.

**Proposed prereg fix.** Record, for every string in `A(0.2) \ A(0.0)`, its margin to each of the three
clause thresholds; pin the encoder device (CPU) and stamp it with the torch version. Add: "A COLLAPSE
whose every deciding string has a clause margin < 0.01 is reported as **COLLAPSE (marginal)**; the
outcome addendum must say the verdict rests on a sub-0.01 margin." (Reporting rule only; does not move
the decision boundary.)

### 7. "What this does not claim" omits the order/position bound

**Reason.** The rewarded node is the FIRST node formed, encoded with no competitors, followed
immediately by its six same-class food strings — the maximum-exposure, food-first ordering. That is a
plausible worst case for absorption but also a specific case: a node rewarded mid-stream, or a
non-food node, or a walk with distractors interleaved, is untested. Order matters by construction
(`run_cell`'s own comment: "Order matters because pattern completion is centroid-dependent").

**Proposed prereg fix.** Extend the bound: "It is one walk order (Exp 24's, pairs then distractors), and
the rewarded node is the first-formed node with no competitor at formation and its same-class strings
immediately following. It does not say how a node rewarded later in a stream, or a non-food node,
behaves; it does not say anything about reward that rises during the stream rather than being held at
`b` from formation." Optional (reported, never deciding): the reversed walk.

---

## NIT

### 8. Bias exactness at b = 0.1

`credit_node` stores `alpha · reward` clamped to `[0, 0.20]`; b=0.2 is reachable exactly by
over-crediting into the clamp, but b=0.1 depends on `reward = 0.1/alpha` and `0.44 − 0.1` is
`0.33999999999999997` in float64. Non-deciding, but the record should stamp the stored
`reward_bias` and the override value actually passed, not the nominal `b`.

### 9. Pair purity / distractor collapse need the b=0 reference to be readable

Every distractor string except two is a pair string, so distractor collapse is fully determined by
pair assignments; and pair_01 is impure at b=0 already (food strings join it). Report both metrics
as deltas from b=0 so a reader does not read baseline impurity as a reward effect.

### 10. Positive control's "foreign" vs RA's "every node rewarded"

In RA every node is rewarded, so "a node it does not reach at b=0" is ambiguous when nodes are not the
same across b (canonicalised by first-formation order, which itself changes with b). Define it as
"a string whose node's first-formed member differs from its b=0 node's first-formed member".
