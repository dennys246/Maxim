# Confounding-lens review — grounded word binding demo (plan v2)

**Verdict: ADOPT WITH CHANGES.** The plan as a *plan* is sound: every experiment is routed through
its own prereg and four-lens review, and Exp A's move to a sensory-preconditioning design is the
right instinct. But all three experiments, as sketched, can produce their headline result for a
reason other than the claim. Two findings are DO-NOT-BUILD for the Exp A and Exp C preregs: Exp A's
"conditioning" route does not exist in today's write path, so the discrimination is vacuous; and
Exp C has no arm that separates *consulting less* from *consulting when uncertain*. Neither blocks
Stages 0–2 or 4.

Scope: Experiments A (Stage 3), B (Stage 5), C (Stage 6), plus the claim sentence and the Stage 1/2
choices that feed them. Read against `docs/experiments/DESIGN_REVIEW.md`,
`docs/experiments/56_four_arm_sharing.md`, `docs/experiments/exp61_shared_fear_prereg.md`,
`docs/experiments/paired_data_audit_2026-09-20.md`, `docs/experiments/paired_data_audit_reaudit_2026-09-21.md`,
`docs/wiring/substrate-learning-channels.md` and `docs/wiring/cluster-dilution-blocks-situation-fear.md`,
plus the code cited below.

---

## DO-NOT-BUILD

### DNB-1 — Exp A: the "conditioning" route it claims to rule out cannot happen with today's code, so "fear to the word ⇒ reference" holds by construction

**Evidence.** The plan's Stage 3: *"(a) conditioning — the text cluster in the situation acquires
fear/want directly, as any cue would; (b) reference — via `NAMES` … Fear to the word ⇒ reference."*
The only fear writer, `proprioception/pain_bus.py` (the `_on_pain` subscriber that calls
`nac.record_cluster_fear`), books fear on **the world cluster only**:
`world_cluster = nac.active_clusters(agent_id).get("world")`. Its docstring says *"WORLD cluster
only"*. A `text` cluster in the situation key can never receive fear, so route (a) is structurally
closed for fear. The want side is similar. `runtime/tool_dispatch.py` routes cluster reward to one
`credit_cluster` chosen by operant routing (world/audio/interoception), and `NAc.credit_operant_reward`
credits a pending `(cluster_id, tool)` set by the substrate path. Neither routes to `text` unless
Stage 1 changes it, and the plan does not say whether it will.

As written, any fear to the word *must* come through `NAMES`. The design would call that
"reference", but the result would come from the write path's allowlist, not from anything measured.
The binding-ablated arm then only shows that the only route is the only route. The opposite failure
is just as possible. If Stage 1 quietly makes the pain subscriber book fear on every active cluster,
a leftover `text` cluster from phase 1 can take fear directly in phase 2, and the design cannot
detect that (see SF-1).

**Fix.**
1. The Exp A prereg must state, per valence surface (`cluster_fear`, `cluster_reward_bias`), whether
   a `text` cluster can be written, citing the writer by `file::symbol`.
2. Add a **positive-control arm, "conditioned word"**: the word is present *during* the harmful
   phase and absent in phase 1, with `NAMES` ablated. It must show fear to the word through route
   (a). Otherwise route (a) does not exist and the claim becomes "the word's effect runs through
   `NAMES`", which is a wiring statement and not a reference-vs-conditioning result. If route (a) is
   deliberately closed (fear stays world-only), say so and retitle Exp A. Its real question is then
   "does a bound word reactivate its situation's valence, beyond the novelty of hearing any word?"
   Keep the SP design for when route (a) exists.
3. Pin both routes with a known-answer check before the harness runs. Route (a) must write
   `cluster_fear(text_cid) < 0` in the conditioned-word arm, and `cluster_fear(text_cid) == 0` must
   hold on disk in every SP arm at test time.

### DNB-2 — Exp C: consulting less and consulting when it matters are not separated, and "own experience overrides" is a property of the merge that every arm shares

**Evidence.** The plan's Stage 6: *"the gated arm limits the damage because it consults less and its
own experience overrides."* Both parts of that explanation undercut the claim:
- **"Consults less".** Against a corrupted Oasis, any policy that ingests fewer entries takes in
  fewer corrupted ones. Gated < always on damage is guaranteed by the dose, whatever the gate
  decides. The claim ("the gate protects", and in the headline "*when and only when the receiver is
  uncertain*") is about **when** it consults, and no arm tests timing apart from frequency.
- **"Own experience overrides".** Foreign discount, own-experience-wins and the tighten-only fear
  fold (`hivemind/merge.py::substrate_merge`, the Exp 61 `FOREIGN_FEAR_DISCOUNT` at
  `hivemind/ingest.py::_validate_nac_payload`) are the **same** in all three arms. The override
  cannot explain a difference between arms. It only sets how much any consult can hurt.

**Fix.**
1. Add a **yoked-random consult** arm. Each yoked agent consults exactly as many times as a paired
   gated agent, at times drawn from the gated agent's inter-consult distribution but **not**
   triggered by its own uncertainty or stakes. Gated vs yoked is the test of the gate; gated vs
   always is only the dose. Without this arm the prereg may not claim "the gate protects".
2. Add a **content-null (empty-slice) consult** arm, or at minimum an always-consult arm that gets an
   empty answer. This separates the *waiting behaviour* (Open question 1; a "cautious hold" makes
   always-consult agents more cautious whatever the answer says) from the *answer's content*.
3. Record the dose as a mediator: consults, entries ingested, and corrupted entries ingested per
   agent. Report **damage per corrupted entry ingested**. If gated and yoked match on that measure,
   the gate protects only through frequency, and the record must say so.

---

## SHOULD-FIX

### SF-1 — Exp A: text persistence and the causal link can leak the word into phase 2, or the outcome into the test

**Evidence.** Stage 1: *"a heard word stays in the situation for a window, then decays."* Stage 3:
*"then let the situation acquire its fear without the word present."* `NAc.note_active_clusters`
stashes whatever the tick encoded. A phase-1 word still inside its window at a phase-2 pain tick is
**present**, so the preconditioning becomes simple conditioning if route (a) is live. Separately,
`docs/wiring/substrate-learning-channels.md` shows the **causal link is state-blind**. Any
*executed* escape or avoidance in phase 2 gives that action a positive link, which drives it at test
in every arm, so the word would not be needed (Exp 61 Trap 1, same shape).
**Fix.** (a) The gap between phases must be at least the persistence window plus a margin, and the
harness must **assert from the logs** that no pain booking in phase 2 had a `text` cluster in its
noted active set. (b) Phase 2 is propose-only, or the donor is rescued with no executed avoidance,
as in Exp 61, and `links == {}` for the DV tool is asserted on the staged `nac.json` before test.
(c) Use Exp 56/61's decision-provenance clause: a test success counts only when the fear/drive
component is decisive with `causal == 0` and `learned_bias == 0` at the executed proposal.

### SF-2 — Exp A: "sensors do not yet discriminate" must be a measured identity check, and the teacher is a world sensor

**Evidence.** Stage 3's DV is *"the executed choice at a timepoint where the sensors do not yet
discriminate."* No check is named. The world channel also carries `nearest_player_dist`
(`_data/components/bodies/minecraft_player.yaml`, rest 64; the bridge reports it). A **scripted
teacher player** who comes close to speak changes the world vector exactly when the word arrives. The
word and the teacher's approach are then confounded in binding **and** at test: "fear to the word"
could be fear to a cluster that the teacher's approach moved. Other sidebands include chat arriving
through the `chat` event rather than `systemChat`, and the harness's `/tp` settle.
**Fix.** (a) The teacher speaks from off-world (RCON `/tellraw`, or a player held at ≥ 64 blocks),
and every settle asserts `nearest_player_dist == 64`, the Exp 61 settle guard reused. (b) At the test
tick, assert that the world cluster the agent is in is **the same id** in every arm (the neutral
situation) and differs from the feared situation's id. Known-answer rule: the no-word arm reads
`anticipatory_threat_need == 0` at that tick. (c) Fix the test word's timing on the clock, never on
the agent's state (no "speak when it nears water"), so the teacher's timing cannot carry information
about the agent.

### SF-3 — Exp A: sensory preconditioning cannot tell reference from mediated conditioning; decide which reading the ablation arm tests

**Evidence.** In the animal literature SP has two accounts. One is **retrieval at test**, which is
the plan's "reference". The other is **mediated conditioning**: during phase 2 the situation
reactivates the word's representation and the outcome conditions *that*. Behaviour at test is the
same under both. If `NAMES` is traversed in both directions, or the reactivated text concept enters
the active set during phase 2, the plan's positive result is the mediated one.
**Fix.** Split the ablation arm by timing. **Ablate before phase 2** vs **ablate after phase 2, before
test.** Retrieval-at-test predicts that both ablations abolish the effect. Mediated conditioning
predicts that the after-phase-2 ablation leaves the effect intact, because the valence is already on
the word. Make `NAMES` retrieval one-way (word → situation) and say so, or measure the direction.
Together with DNB-1(3) this makes "reactivates the situation it named" a result instead of an
interpretation.

### SF-4 — Exp A: controls for word novelty and for how reactivation enters selection

**Evidence.** The arms: *"word (bound) · no word · shuffled binding · unbound word · binding
ablated."* An "unbound word" that was never heard in phase 1 is **novel** at test, and novelty
drives the Stage 6 trigger (EC-margin novelty, `LLMProposal.cluster_margins`), so it differs from
the bound word in two ways. The plan also does not say how a reactivated world concept enters
`NAc.recommend_action` / `anticipatory_threat_need`. If it **replaces** the perceived `world` slot,
the agent acts as though it perceives water, which is a masked percept rather than anticipation.
Pain at test would also book onto the *recalled* cluster through `pain_bus._on_pain`.
**Fix.** (a) Make **shuffled binding** (a word equally familiar but bound to a neutral situation) the
primary control. Keep "unbound" as a novelty check that is reported but not gated. (b) The prereg
fixes reactivation as **additive**: the recalled cluster is read by `anticipatory_threat_need` under
its own key and never overwrites `active_clusters["world"]`. It asserts that pain during test cannot
write onto the recalled id.

### SF-5 — Exp A: harmless pre-exposure is a latent-inhibition confound; hold it constant across arms

**Evidence.** SP needs phase-1 exposure to the situation "while it is still harmless". Pre-exposure
by itself slows the later fear acquisition (latent inhibition), and more so the more exposure there
is. If arms differ in phase-1 exposure (for example, no-word agents skip the teacher episodes), the
phase-2 fear differs between arms and the test difference is a difference in fear strength, not in
reference.
**Fix.** Yoke phase 1. Every arm gets the same situation exposures on the same schedule, and only the
word (present, absent, or a shuffled mapping) varies. Before test, gate on the **phase-2 fear value
on the world node** being equal across arms (read from the staged file, as Exp 61's
post-discount-value check does).

### SF-6 — Exp B: arms missing the want-not-file and cluster-not-fear analogues, and pre-boot vs live is confounded

**Evidence.** The arms: *"taught · isolated (same budget, learns alone) · dangling (bindings without
the world EC nodes — must fail) · naive."* The load-bearing control of Exp 56 (satiated) and Exp 61
(arm 3, cluster-not-fear) has no analogue here. When the word alone moves B, it is unclear whether
the imported **binding** did it or the imported **fear on the world node**. B hears the word on
shore, where the imported fear does not fire unless something reactivates the node, and that
"something" is exactly what the missing arm would test. "Isolated (same budget, learns alone)" does
not say what B learns alone from.
Pre-boot vs live also differ in more than the ingest path. A reboot clears transient state (the
active-cluster stash, text persistence, `PerceptTraceBuffer`). A live receiver already has its own
world nodes, so `substrate_merge`'s aligned re-key can map a donor node onto one of B's own nodes,
which cannot happen on a fresh pre-boot receiver.
**Fix.** (a) Add **binding-stripped** (bundle with EC and fear, `NAMES` removed; must be at floor for
the word-alone DV) and **fear-stripped** (bundle with EC and `NAMES` from an ablated donor, where
the word reactivates the node but nothing is feared; must be at floor). These show that the
transfer needs *both* halves. (b) Before export, assert that the donor's text cluster has
`cluster_fear == 0` and `cluster_reward_bias == {}` on the staged files. Otherwise B could react to
fear conditioned straight onto the donor's text node (DNB-1). (c) State that pre-boot vs live is an
**equivalence / non-inferiority** comparison with a pre-registered margin (for example live ≥
pre-boot − 0.15, one-sided). Give both arms the same pre-ingest world exposure, and record the
id-map on both as a mechanism DV. (d) The text-merge threshold (Stage 4) is calibrated on data
separate from Exp B's receivers.

### SF-7 — Exp C: the corrupted Oasis decides the result by where the corruption sits

**Evidence.** Stage 6 corrupts *"stale or wrong entries for some situations."* The gated trigger
fires only in unfamiliar or uncertain situations. If the corrupted entries sit on situations the
agent already knows, the gate **never reads them**, and "the gate protects" is true by the choice of
placement. If they sit only on novel situations, gated and always read them equally often. The
tighten-only fear fold also makes the harm asymmetric. Wrong *fear* can only add caution (a
lost-opportunity cost), while wrong *want* can pull the agent toward harm. So valence inversion
means something different on each surface.
**Fix.** Pre-register placement as a factor, or at least a stated mix: a fixed share of corrupted
entries on **novel** situations (where the gate fires) and a share on **familiar** ones. Report
damage separately for each. Corrupt **fear and want separately**, and choose the corruption type
(Open question 2) before any pilot. Use a primary DV that covers both costs of caution and costs of
harm, for example R3's survival score or time-alive plus reward, not a count of deaths alone.

### SF-8 — Exp C: trigger thresholds and the statistic must be fixed on data disjoint from the confirmatory run

**Evidence.** *"Thresholds start hard-coded"*, and the trigger is an AND/OR over about seven signals
(drive pressure, anticipatory threat, pain just felt, novelty, missing history, `min_confidence`,
outcome variance, recent negative outcomes, refractory). That leaves a large set of analysis choices
that pilots on the same world would tune.
**Fix.** Calibrate thresholds on a **separate seed set or world layout**, freeze them in
`FROZEN`-style constants that the analyser checks for drift, and report a threshold sweep only as
secondary. The prereg's claim is about *this frozen gate*, not "the gate". Statistic: a 2 (Oasis) ×
4 (gated / yoked / always / never, plus content-null if adopted) design with the agent-session as the
replication unit. The primary contrast is the **interaction** (gated − yoked under corrupted) minus
(gated − yoked under clean). Use a permutation test on the interaction, with Holm across at most
three pre-named contrasts. n comes from a counted-out pilot's variance, set before the confirmatory
run. As a floor, n ≥ 20 per cell for a continuous DV, because live Minecraft timing noise is real
(Exp 61 environment S3).

### SF-9 — Adding the text modality changes the baseline; every arm needs text on

**Evidence.** Stage 1: *"adding a modality changes the situation key every EARNED survival result was
measured on."* The plan handles this for Exp 60/61/62 re-runs but not inside A/B/C. If "no word" or
"never consult" arms run with the text channel **off** while the word arms run with it **on**, the
channel's presence (a new `{modality: cluster}` entry scored by `recommend_action`) confounds the
contrast.
**Fix.** Every arm of A, B and C runs with the text channel on. Silence is represented the same way
in every arm (an absent `text` key, stated), and a Phase-0 check confirms that a text-on agent in
silence matches a text-off agent on the neutral-situation proposal.

### SF-10 — The claim sentence is stronger than A/B/C can support

**Evidence.** *"…reaches another on demand, **when and only when** the receiver is uncertain in a
situation that matters; and that gate **protects** it from a wrong Oasis."* "Only when" is a design
property of the trigger, true by construction, not a finding. "Protects" needs DNB-2's yoked arm.
"Reactivates the situation it named" needs DNB-1 and SF-3. Stage 6 also names Exp C "Rogers'
prediction". Rogers' paradox concerns population-level fitness in a *changing* environment. One
agent against a fixed corrupted store tests the **critical-learner** rule (Enquist et al. 2007), not
the paradox.
**Fix.** Rewrite: *"…reaches another on demand, through a consult its own uncertainty and stakes
trigger; against a corrupted Oasis the triggered consult costs less than an equally frequent
untriggered one."* The second clause is claimable only if the yoked contrast passes. Replace "when
and only when" with a **trigger-validity** DV: the share of consults that fire in situations with
novelty above the threshold and stakes above the threshold, versus the yoked arm. Rename the Exp C
prediction "critical-learner prediction (after Rogers 1988)".

---

## NIT

- **N1 — Stage 2's binding accuracy measures the teacher.** With a scripted teacher speaking at
  pre-registered moments, "binding accuracy on held-out pairings" mostly reads back where the
  teacher was placed. Jitter the teacher's lag by sampling from Stage 0's measured natural-lag
  distribution, so the look-back window is actually exercised.
- **N2 — Exp A statistic and n.** One binary DV per agent at the **first** word-alone test. Later
  tests are reported and not gated, because hearing the word without the outcome is an extinction
  trial. With a deterministic selector and a structurally-`None` floor, use Fisher's exact test,
  one-sided, as in Exp 61. Primary contrasts are bound vs shuffled and bound vs ablated-before-phase-2,
  with Holm correction. n = 12 for the claim arm and n = 24 for the floor arms (the Exp 61 D2
  reasoning: the Wilson upper bound for 0/24 is 0.14).
- **N3 — The `dark` row.** Exp 62's night pool (0.799) says fear misses at night. Any Exp A/B/C trial
  run at night will fail for a known reason. Keep night trials out of the gated arms, or make them a
  declared secondary.
- **N4 — "Taught" in Exp B** is a borrowed name. In Exp 56 it meant teacher credit. Here the donor
  *learned* from pain. Name the arm "transferred-binding" to avoid a false parallel.

---

## Verified fine

- **Raw, no bundle, for Exp A** (Stage 3) keeps the transfer mechanism out of the reference claim.
  Correct separation.
- **The binding-ablated arm as the guard-by-deletion arm** follows the #859 lesson and is the right
  kind of arm. SF-3 only asks for it to be split by timing.
- **The dangling arm in Exp B** carries over Exp 56/61's representation-half falsifier. Keep the
  loud-drop accounting (`*_dropped == shipped`, `*_rekeyed == 0`) for `NAMES` endpoints (the Stage 4
  dangling rule).
- **Signature assertion in the Exp B harness** closes a real Exp 61 gap (a prereg that said signed,
  a harness that passed no `--sign`).
- **"A genuine may-fail"** is stated for Exp C, and the null is recorded as a result.
- **Web text as TEST only** (Stage 8) correctly keeps unpaired data out of training.
- **The Stage 0 window is fixed from a natural-death capture, not the staged re-audit.** This honours
  the re-audit's own EXPLORATORY caveats (window chosen post hoc, identical staging, n = 5).
- **Fear transfer through `substrate_merge` is already min-fold and discounted** (Exp 61 D1), so
  Exp B inherits a measured transport. The confound is the missing arms (SF-6), not the transport.
