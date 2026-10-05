# E4 design review — bio-faithful lens

Reviewed: `docs/experiments/protocols/e4_text_widening_drift_preregistration.md` (draft, 2026-10-05).
Question asked of this lens: does the setup reproduce how reward actually reaches a text node in
production (credit cadence, decay, which nodes get rewarded, whether the 0.20 cap is reached, per-agent
keying), so the result transfers? Is "bias held constant, no decay" a fair worst case, and is that said?

**Verdict: no DO-NOT-BUILD. Four SHOULD-FIX, three NIT.** The apparatus is a faithful copy of the EC
mechanics. What it does not copy is how reward reaches a node in production, and today **no live path
gives a text node a positive bias at all**. The measurement can still answer its own question (does the
mechanism drift if a node is widened?), but the prereg has to say the hazard is latent and that the arm
is an upper bound. Otherwise a COLLAPSE result reads as a live defect.

## What I verified (at the symbol)

- **Override semantics.** `NAc.get_threshold_overrides` computes `base − bias`, clamped to `[0.1, base]`,
  and the encoder passes `base_threshold=self.ec.config.pattern_complete_threshold` on both paths
  (`similarity/encoder.py::LinguisticEncoder._get_reward_overrides`, `encode_decomposed`). The EC matrix
  scan (`similarity/ec.py`, the `scan` method) applies per-row thresholds and returns the
  highest-similarity *eligible* node. So widening only wins when no other node is closer, which matches
  the prereg's apparatus.
- **Centroid.** Text is not in `frozen_centroid_modalities`. The running mean is
  `(c·n + e)/(n+1)` (`EntorhinalCortex.pattern_complete_or_separate`).
- **Decomposition is off by default** (`MAXIM_CONCEPT_DECOMPOSITION` is opt-in, `integration/memory_hub.py`),
  so whole-string encoding is the faithful regime, matching Exp 24.
- **The text path itself is opt-in.** `LinguisticEncoder` is only constructed in production when
  `MAXIM_SUBSTRATE_PATH=1` (`MemoryHub` substrate wiring). That covers the Roy/O19 regime. Default runs
  do not encode text into the EC at all.
- **Credit path.** The only live writer of `_reward_bias` is `NAc.credit_node`, which is called from
  `TemporalCreditDistributor.distribute`, which is called from `bio_stack._distribute_reward_from_reaction`
  (a `reaction_bus.subscribe_all` subscriber). `NAc.distribute_reward` has no production caller. The
  other writer is `load_state` (persisted rows).
- **Decay.** `decay_reward_biases` runs every loop tick (`agent_loop` §8.5): ×(1 − 1/50) per tick, pruned
  below 0.001. The wall-clock half-life on load is 7 days.

## Findings

### SHOULD-FIX 1 — No live producer of positive node credit, so the hazard is latent; the prereg must say so

**Reason.** Every Reaction published to the reaction bus is `Valence.NEGATIVE` pain:
`proprioception/pain_bus.py` (via `reactions/compat.py`), `perceived_pain.py`, `pain_interceptor.py`,
`sim_adapter.py`, `simulation/sandbox.py` and `conversational_source.py`. The one positive emitter is
`CerebellumModulator` (`kind="reward"`), which has been **Dormant since 2026-05-26** and has no live
factory. `credit_node` clamps at 0 and removes a zero key, so negative credit never creates a bias.
`focus_learner`, `tool_pain_bridge`, `planning_bridge` and `pain_bridge` do produce POSITIVE values, but
they record causal-link outcomes, not Reactions, so they never reach `credit_node`.

The committed record agrees. The O19 substrate run
(`docs/experiments/data/rerun_exp09_o19/20261002_230206/`, `MAXIM_SUBSTRATE_PATH=1`) logs 12 credit
distributions, all negative (−0.15 to −0.25), each split across 34 nodes. Its persisted `aut_nac.json`
has `reward_bias == {}`. The four heartbeat sim-short `aut_nac.json` files are also empty.

So under the current code, no text node in any production run ever has an override, and the widening the
prereg measures never fires. The issue's framing, "reward can lower the effective threshold … for the
nodes that matter most", describes a mechanism that has no live input.

**Why this matters for the decision.** As written, COLLAPSE leads directly to a change in
`pattern_complete_or_separate`, which fires the EC completion ledger row's re-run. That would be a
`src/` change to the encode path to guard against a hazard nothing can currently trigger. The
fix-ships-with-a-caller rule applies in reverse here: there is no live caller of the hazard either.

**Proposed fix to the prereg text.**
- In **Question**, add: "Today no live path writes a positive `_reward_bias`: the reaction bus carries
  only pain, apart from the Dormant `CerebellumModulator`, and the committed O19 substrate run ends with
  `reward_bias == {}`. Text widening is also opt-in (`MAXIM_SUBSTRATE_PATH=1`). This measures a latent
  hazard: what the mechanism would do once a positive node-credit producer exists (e.g. R4's credit
  routing)."
- In **Decision rule → COLLAPSE**, add: "the design entry is written now. The
  `pattern_complete_or_separate` change lands with, or before, the first live producer of positive node
  credit, and that producer's PR cites this record." This does not change the decision rule; it states
  when the consequence takes effect. If the owner prefers to fix it preventively now, the prereg should
  say "preventive" explicitly.
- In **What this does not claim**, add: "nor that any current run widens a text node."

### SHOULD-FIX 2 — "b held at 0.20 with no decay" is an upper bound, and the prereg does not say so

**Reason.** In production a node reaches bias `b` only through
`credit_node(reward × share)`, where `alpha = 0.15`:
- A single Reaction has intensity ≤ 1.0, so even with the whole reward on one node, one credit adds at
  most **0.15**. Reaching the 0.20 cap needs a cumulative reward of at least 1.33 on that one node
  before decay. The real share is `strength / Σ strengths` over every eligible node; in the O19 run that
  was 1/34, so roughly 0.004 of bias per credit.
- Decay at τ = 50 ticks takes 0.20 down to about 0.11 in 30 ticks, and a lone credit drops below the
  0.001 prune in about 75 ticks (if it was 0.005 to begin with).
- The rewarded node in the arm is newly formed (n = 1–2) when it is widened, so each absorbed string
  moves its centroid by 1/(n+1), the largest possible step. In the O19 run, long-lived text nodes already
  had 25, 18 and 14 members at b = 0, and a centroid that old barely moves.

All of this makes arm R1 a worst case: maximal bias, held indefinitely, on the most movable (freshest)
centroid. That is a legitimate design: a NO COLLAPSE under the worst case is strong.
But it has to be stated, or a COLLAPSE will be read as typical.

**Proposed fix to the prereg text.** Add a short **Fidelity** paragraph under Apparatus:
"Arm R1 is an upper bound on bias exposure, not a production replay. Production credit is split across
every eligible node (1/34 per node in the O19 run). One Reaction adds at most `alpha` = 0.15, so
b = 0.20 needs a cumulative reward of 1.33 or more on one node. Per-tick decay (τ = 50) halves a bias in
about 34 ticks. The rewarded centroid is widened while n is 1–2. So NO COLLAPSE here bounds production;
COLLAPSE here does not show production reaches the condition."

Also state the exact call: `credit_node(agent, node, reward=b / alpha)`, i.e. 0.667 for b = 0.1 and
1.334 for b = 0.2 (or 2.0, which the clamp caps at 0.20). Note that 1.334 exceeds what any single
production Reaction can deliver.

NO COLLAPSE at b = 0.2 transfers to lower and decaying biases only if absorption is monotone in the
threshold. A sequential walk is path-dependent, so that is not guaranteed. Report whether
`A(0.1) ⊆ A(0.2)` holds; it costs nothing, and it is reported, not decided on.

### SHOULD-FIX 3 — "Foreign" is defined by pair id, but three other pairs are the same concept as the rewarded node

**Reason.** `A(b)` counts every string outside pair_01. But pair_02, pair_03 and pair_04 share
pair_01's fixture class `food_overlap_2pc` ("a portion of food rests within reach.", "warm food rises in
your belly.", "fullness eases the pull of hunger.", …). Reward widening exists so that a rewarded concept
recognises looser variants of itself. A food-rewarded node taking in another food string is that
mechanism doing its job (generalization), not drift. Under the current definition, COLLAPSE could fire
on a `food_overlap_2pc` string reached through a drifted food centroid. The satiety strings are the most
likely candidates, because they share no lexical "food" anchor with the seed.

**Proposed fix to the prereg text.** Under **Metrics**, report `A(b)` and `I(b)` broken down by fixture
`class`. Under **Decision rule**, define foreign as "a string whose fixture `class` differs from
pair_01's (`food_overlap_2pc`)", or keep the pair-id definition and add a sentence saying that a COLLAPSE
carried only by `food_overlap_2pc` strings is reported as within-concept generalization. Either way,
decide it before data. This is a definition, not a change to the owner's absorption-beyond-baseline rule.

### SHOULD-FIX 4 — The I(0.2) exclusion may leave no headroom, and it labels cross-concept admission "as designed"

**Reason.** At an effective threshold of 0.24, the isolated arm may already admit most strings written
in the fixture's register. Exp 24 (`24_roy_paraphrase_diagnostic.md`) shows cross-class strings matching
a food centroid at 0.42–0.53, and distractor direct cosines as high as 0.297. Both are above 0.24. If
`I(0.2)` covers almost every foreign string, the COLLAPSE condition `s ∉ I(0.2)` cannot be met, and
NO COLLAPSE would mean "no headroom", not "no drift".

Biologically, the radius rather than the drift is then the finding. A reward that makes the food node
complete "an abrupt chill grips your shoulders." on its own has widened past the concept, and the rule
files that as "widening working as designed". The owner's rule stands, but the record has to show it.

**Proposed fix to the prereg text.**
- Under **Metrics**, add each string's pre-EC cosine to the seed embedding (cheap, and it explains every
  `I` membership).
- Under **Instrument checks**, add a headroom declaration: "If no string lies outside both `A(0.0)` and
  `I(0.2)` (outside pair_01 / outside the rewarded class), the verdict is NO HEADROOM: no verdict, and it
  is reported as such, never as NO COLLAPSE."
- Under **NO COLLAPSE**, add: "the close note on #911 lists every cross-class string in `I(0.2)` as
  widening overreach (the radius at the cap spans concepts), a separate input to whoever builds the first
  positive node-credit producer."

### NIT 5 — Per-agent keying and the encode path

**Reason.** In production, `agent_id` comes from `percept.context.agent_id` (`""` when absent), and the
same id keys the credit (the Reaction's `context.agent_id`), the override lookup and `update_eligibility`.
The prereg says the override is applied "through the same `threshold_override=` the encoder passes". That
reads as a direct EC call.

**Proposed fix.** State that the strings go through `LinguisticEncoder.encode(percept)` (decomposer
`None`) with a percept whose `context.agent_id` is one fixed non-empty id, and that `credit_node` uses
that same id. Keying, eligibility and ATL side effects then match production. Overlaps the wiring lens.

### NIT 6 — The self-reinforcing loop is absent (state it)

**Reason.** In production, a string that completes into the rewarded node refreshes that node's
eligibility (`activation = similarity`). The widened node therefore claims a larger share of the next
credit, which pushes it toward the cap: widening feeds itself. Holding b constant leaves this loop out,
and the cap bounds it, so the worst case still holds. Say so in the Fidelity paragraph from SHOULD-FIX 2.

### NIT 7 — The fixture register versus the production text stream

**Reason.** The O19 run's text nodes are short, templated labels ("turn left", "attack", "claw strike"),
not Roy sensation sentences, and they repeat a lot. The fixture is the Exp 24 register (Roy campaigns),
which is a legitimate choice and the one with known drift history. Add to **What this does not claim**:
"the fixture is the Roy sensation register (Exp 24), not a capture of a run's text stream."
