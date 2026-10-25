# GL3 design review — BIO-FAITHFUL lens (thalamic_relay.md)

I read all of `docs/plans/thalamic_relay.md` in worktree `.worktrees/e-t8b` at origin/main `d9dffe6e`, along with
`docs/experiments/DESIGN_REVIEW.md`, `docs/plans/grounding.md` (owner decisions G1–G20, the track table, and the
GL1 fold table), `docs/plans/reviews/grounding_gl1/bio-faithful.md`, `docs/plans/autonomic_layer.md` §3.1.2–§3.2
(where GL3 touches it), `docs/plans/deferred/reflex_layering.md`, and `docs/wiring/` (README,
`body-and-word-worlds.md`, `cluster-dilution-blocks-situation-fear.md`). I checked every code claim I rely on
against the code in this worktree (see §Verified). Owner decisions G1–G20 and TR1/TR2 are not re-opened. Where
a finding touches one of them, it only asks how the decided item is labelled or routed.

The lens asks whether each track, receptor and gate does the biological job its name claims, and whether each
manipulation respects how Maxim's body, PainBus and Wire-4 fear actually work. The plan's code names
(`ReceptorRegistry`, `TrackScheduler`) are neutral. Most of the bio over-claim sits in the track table and
the stage gates, not in the class names.

**Verdict: 2 DO-NOT-BUILD / 8 SHOULD-FIX / 6 NIT.**

Recommended reading order: D1 (it breaks the earned Exp 60 mechanism under the GL3.B3 flag), then D2 (GL3.B4
has nothing to preempt with), then S1 (the `reflex` track is mapped onto a module that inflicts damage), then
S2, S3 and S4 (what the slow track and provenance really are), then the rest.

---

## DO-NOT-BUILD

### D1 (GL3.B3): track routing of the PainBus publish delays air-hunger fear by 2 passes and writes health fear twice; the learning sinks must get each pid once, at transduction

**Issue.** GL3.B3 routes "NOCICEPTIVE fans out to fast + slow, DRIVE goes to slow" (§6 GL3.B3 Build). §5.8 makes
PainBus "the `pain_bus` destination of `nociceptive_fast` (NOCICEPTIVE) and `affective_slow` (DRIVE)", with
track-delivered publishes replacing the wall refractory. §5.4 gives `affective_slow` latency 2 and the
destinations "autonomic, credit (cluster fear, memory tag), prompt". `nociceptive_fast` keeps "pain_bus (the
learning subscribers, unchanged)". Put that against the code:

1. **Air hunger is DRIVE and is on the Wire-4 allowlist.** `proprioception/pain.py::classify_pain` returns
   `PainKind.DRIVE` for every `drive:` source except `TISSUE_DAMAGE_DRIVES = {"drive:health"}`, so
   `drive:oxygen` is DRIVE. `decisions/nac.py::DEFAULT_CLUSTER_FEAR_FAILURE_MODES = {"drive:health",
   "drive:oxygen"}`. The Wire-4 subscriber (`proprioception/pain_bus.py::create_pain_cluster_fear_subscriber`)
   books fear on `nac.active_clusters(agent_id)["world"]` **at publish time**. Its docstring says that is
   correct only because "the loop runs [the encode] BEFORE its pain tick, so pain sees THIS tick's situation".
   With DRIVE on `affective_slow`, the `drive:oxygen` publish moves from H3 (in place, inside
   `propose_via_substrate`) to H4 two passes later. The fear then lands on whatever world cluster was noted
   last. That is exactly the Exp 58 W-4 hazard the plan calls load-bearing (§3.3, §5.5 H3). The plan's claim
   that "the W-4 order … stays byte-identical" holds only for latency-0 tracks, and drowning is not on one.
   The frozen characterization records `drive:oxygen` publish at **+0 passes** on the drowning arm. Exp 60's
   learned water fear rides on that number.
2. **Health damage is NOCICEPTIVE and reaches fear twice.** `drive:health` fans out to fast and slow.
   Fast → PainBus → the pain-memory, NAc-outcome, Wire-2 and Wire-4 subscribers. Slow → "credit (cluster fear,
   memory tag)". So one pid writes cluster fear and tags memory twice, two passes apart, and the second write
   can land on a different cluster. The plan's own invariant is "one physical event, one pid". Under it the
   learner should see one event, not two.
3. **Biology.** Calling air hunger "slow affective" gets the physiology wrong. Dyspnea and air hunger come
   from central and carotid chemoreceptors. They are among the fastest and most aversive interoceptive alarms
   (the CO₂ / suffocation-alarm literature; insula and amygdala activation within the breath). The
   homeostatic-afferent reading folded from GL1 (N2) justifies a slow *route label*. It does not justify
   delaying the teaching signal. On the fast/slow split itself: first and second pain both reach the
   structures that learn, and the split exists so the *fast* signal can act (localize, withdraw, interrupt)
   before the slow one adds affective weight. Neither pathway is a second, independent copy of the
   prediction-error update. A design that teaches the learner twice from one event, or teaches it late on the
   wrong context, does not reproduce either job.

Flag-off goldens do not catch this, because nothing routes. GL3.B3's flag-on gate 5 pins a *new* golden
without asserting the W-4 property, so the regression would be recorded as the expected answer.

**Fold (plan edits, strict first):**
- (a) **Strict, recommended.** Separate *transduction* (the PainBus publish to the learning subscribers) from
  *track delivery*. The PainBus publish of **every** kind (NOCICEPTIVE and DRIVE) stays at H3, in place, at the
  transduction pass, exactly as today. That is the only point where Wire-4, pain memory and NAc outcome see a
  pid. `affective_slow` destinations are restricted to sinks that are **not** already PainBus subscribers:
  autonomic, prompt, and a *pid-keyed* consequence record. Delete "credit (cluster fear, memory tag)" from the
  `affective_slow` row, and delete the "`affective_slow` (DRIVE)" half of §5.8's PainBus row. Add a §5.10
  invariant: "each learning sink receives a given pid at most once, keyed on the situation noted at its
  transduction pass", guarded by a GL3.B3 test that counts `record_cluster_fear` calls and pain-memory tags per
  pid on a scripted health hit and a scripted oxygen breach (one each) and asserts that the oxygen-breach fear
  key equals the cluster noted at the breach pass.
- (b) Alternative. If the owner wants the slow track to carry credit, any slow-delivered learning write must
  carry the **cluster snapshot taken at the pid's transduction** (in `AfferentEvent.extra` or in a field) and
  write against it, never against `active_clusters()` at delivery. The fast track must then drop the same
  sink, so the count stays one. This still needs the count test above. It also changes what Exp 58/60's fear
  write reads, so it fires T1-13/14/15 under the flag.
- Either way, add to GL3.B3's falsifiable gate: "flag on, drowning arm: `drive:oxygen` publish latency in
  passes equals the frozen characterization's (+0), and its Wire-4 key equals the breach-pass cluster".

### D2 (GL3.B4): in substrate-primary nothing transduces a breach before H2, so the preemption check has no event to see; receptors must transduce every pass, not when selection polls

**Issue.** GL3.B4's build list is `basis_seq`, H2 (`preempted_since` before §4), the preemption right, and the
handler (§6 GL3.B4). In substrate-primary, the only per-pass body transduction is inside `propose_via_substrate`
(`runtime/substrate_proposal.py`: encode → `note_active_clusters` → `embodiment.evaluate_failures()` →
`anticipatory_threat_need` → `recommend_action`). That function runs only when
`runtime/loop_state.py::_substrate_tick_due` holds, which requires `ctrl.pending_proposal is None`. The
llm-primary per-pass tick, `runtime/loop_gates.py::tick_embodiment_drift`, is a no-op on substrate-primary by
its own docstring. So while a proposal is pending, which is exactly the L1 window GL3.B4 exists to close, no
`nociceptive_fast` event is emitted before H2. The hardware-edge inbox that could carry the pump's write is
GL3.B8. The frozen record shows this: damage arm, health 20 → 8 at pass 1, the stale `mine_block` dispatched
at pass 1, and pain published at **pass 3** (+2). At H2 in pass 1 (or 2), `preempted_since(basis_seq, …)` has
nothing with `seq > basis_seq`. Gates 2 and 3 cannot pass with GL3.B4's build list alone. They would fail
loudly, so this is not a silent problem, but building GL3.B4 as written cannot do its job.

**Biology.** This is the core caricature. A nociceptor transduces when the stimulus arrives, independent of
whether the cortex is deliberating. A "fast" afferent that is sampled only when the selector next asks for
context is not fast. The latency gain the stage measures would then come from moving transduction, not from
preemption. The plan already names this seam for L2 ("separating transduction from selection inside the
substrate tick, so the turn budget gates action only"), but it never makes per-pass transduction a GL3.B4
prerequisite, and L2's wording keeps transduction "inside the substrate tick".

**Fold (plan edit; owner pick at GL3.B4 start, strict first):**
- (a) **Strict, recommended.** Make "body transduction runs once per pass, independent of selection, before
  H2" an explicit GL3.B4 **prerequisite**, delivered by the L1 issue (#1176) or by a named GL3 stage (a
  per-pass `registry.tick` for pull receptors at H0, after `_loop_live_tick`). It must keep the W-4 order:
  encode → `note_active_clusters` → pain in the same tick, so per-pass transduction must also encode that
  pass. State this tradeoff and its blast radius (per-pass encodes move NAc's per-pass inputs; check
  T1-13/14/15 wording). Restate gate 3 as two numbers against the frozen record: (i) breach → publish latency
  (owed to per-pass transduction) and (ii) breach → protective action **given** a publish before §4 (owed to
  preemption). The deletion probe in gate 5 must then show that (ii), not (i), goes red without H2.
  Otherwise the stage can claim credit for the transduction fix.
- (b) Alternative. Leave GL3.B4 unbuildable until GL3.B8's pump edge exists. This defers the arm past 1.4.

---

## SHOULD-FIX

### S1 (GL3.B6, §5.4 `reflex` row, §5.4 fan-out example, §3.4 L5): the `reflex` track is mapped to the "spinal withdrawal arc", but the module it re-houses inflicts damage on the body; it is narrated transduction, not a motor reflex

**Issue.** Every shipped `_data/reflexes/*.yaml` response is `damage_component` (7) or `set_entity_sensor` (3).
`attack_flinch` matches "attacks / strikes / hits …" in percept text and **damages the torso**.
`thermal_contact` matches "touch fire / searing heat …" and damages the arms. The module's own docstring
(`embodiment/reflex.py`, "Known layering debt") says that "habituation, sensitization and pre-emption scale the
damage the world inflicts, not the felt pain or the body's response". `deferred/reflex_layering.md` tables the
five stages and finds that stage 4 (withdraw/flinch, the stage that habituates) is "never executed". The plan
nevertheless (i) maps `reflex` to "spinal reflex arc (withdrawal, flexor)", (ii) moves `ReflexRegistry`
evaluation onto a priority-100, latency-0 track that bypasses ThoughtGate, (iii) keeps "`ReflexRegistry`'s own
habituation maths", and (iv) shows the burn delivered on "`reflex` (withdraw)", a motor program that does not
exist. L5's framing, "an innate reflex depends on the token budget. That inverts the biology", is right, but
the inverted thing is **nociceptive transduction of a narrated injury** gated by the cortex, not a spinal reflex.

**Why it matters.** Under the plan's mapping, GL3.B6 makes word-triggered damage, which is narrator physics
stamped `narrated` (G6), the highest-priority "spinal" track, and keeps a habituation that makes injury
habituate. Nociceptors sensitize; they do not habituate. Any later claim about "reflex" timing would describe
stage 2 under a stage-4 name.

**Fold (plan edit):** relabel the `reflex` track's current emitters honestly. Keyword reflexes are a
**narrated nociceptive receptor** (`receptor_class="nociceptive"`, `provenance_kinds={"narrated"}`) whose output
is a body write that the body then transduces onto `nociceptive_fast`. They are not a `reflex`-track motor
program. Keep `reflex` (spinal withdrawal / collicular orienting) only for emitters whose response is a motor
act. Today that is the sim-only §1.16 audio orienting reflex. Strike "(withdraw)" from the fan-out example, or
add "once `reflex_layering.md` route 3 ships a withdrawal act". Make GL3.B6 a revive trigger of
`reflex_layering.md` (it already lists GL3's two-track receptor) rather than a re-housing of the
damage-dealing maths. Restate L5 as "narrated injury transduction is gated by ThoughtGate".

### S2 (GL3.B3, §5.4 `affective_slow` row): "diffuse, lingering, summing" is claimed, but the slow track is a delayed copy of one latched phasic event

**Issue.** Channel 2 of `embodiment/body.py::evaluate_failures` is **latched**. It publishes on band entry and
again only when the breach "materially DEEPENS" (the latch comment calls this "the bio-faithful shape …
steady-state and recovery are silent"). A standing burn therefore yields one event, and `affective_slow` (latency
2, refractory 0, coalesce `sum`) delivers that one event two passes later. Biological second pain is C-fibre
**tonic** activity. It outlasts the stimulus, shows temporal summation (wind-up; Price et al. 1977), and its
affective weight tracks duration. A delay line cannot "linger". The prompt destination "it still hurts" would
fire once, two passes after the hit, and never again. Separately, `coalesce: sum` merges several physical events
into one delivery, which conflicts with "the scheduler delivers the SAME object on every track" (§5.2): it is
not clear which pid a summed delivery carries, and the sum is unbounded above 1.0. At GL3.B3 the slow delivery
also has no reader that acts differently from the fast one. The `autonomic` destination is GL2's
`InteroceptiveOutcome`, which the executor's tool path produces, not a track. So the track's bio job is served
by nothing.

**Fold (plan edit):** pick one, and say it in §5.4 and the GL3.B3 docstring.
- (a) **Recommended.** Describe `affective_slow` as "FUNCTIONAL: a delayed phasic copy; tonic persistence is
  carried by drive pressure and the latch severity, not by this track". Drop "lingering" and the "it still
  hurts" prompt until a tonic source exists.
- (b) Give the slow track a tonic input: one delivery per pass (or per k passes) while the latch is held, with
  intensity = latched severity. That is a real second-pain shape, but it is a new behaviour. It needs an owner
  decision, and it fires T3-9.

Either way, define a coalesced delivery as carrying the **set** of member pids (or the first pid, with the rest
as `caused_by` links), clamp the sum at 1.0, and name the GL3.B3 consumer that reads the slow delivery. If
there is none, say that the track is capability with no caller until GL2's out-of-band producer or GL4 joins
it.

### S3 (§5.9, §5.7, TR12): "imagined creates no eligibility" is tiered **invariant**, and the narrated discount **innate prior**; both are engineering priors over an oracle source label

**Issue.** Biology does not forbid learning from imagination. Imagined extinction attenuates threat responses
much as real extinction does (Reddan, Wager & Schiller 2018). Mental practice changes motor performance.
Hippocampal replay and preplay feed model-based value. Imagined reward shifts valuation (Gershman et al.,
"imaginative RL"). What biology has is **reality monitoring** (Johnson & Raye 1981), an *inferred* source
judgement that fails in known ways (confabulation, hallucination). In this plan the agent never infers
provenance: the producer stamps it as ground truth. That is the same oracle-attribution pattern GL1 SF-6 re-tiered for
`CauseRef`. Likewise `narrated` vs `experienced` is a fact about who wrote the sensor. Inside `--sim` the agent
cannot sense it, and no organism has an analogue of "the world's author" discounting its own pain.

**Fold (plan edit; no decision re-opened):** in §5.9, re-tier "imagined creates no eligibility" as an
**engineering prior (contamination safeguard; oracle source label)**. Keep TR12's strict *refuse* as the
recommendation. The tier changes, not the default. Re-tier "narrated at the declared discount" from innate prior
to **engineering prior (experimenter-side provenance; G6)**. Add one sentence: provenance kinds are
oracle-assigned, so no GL5 or later claim may cite them as learned source monitoring. If the owner ever wants
reality monitoring, it is a separate mechanism with its own trigger, for example a learned
`imagined`-vs-`experienced` discriminator scored against the oracle.

### S4 (GL3.B1, §5.7 item 1, red gate (g)): the strict mixed-state rule makes a whole body channel `narrated` once any one of its sensors was last written by the narrator

**Issue.** The strict default for a pull read of a channel that the narrator partly wrote is "`narrated`
whenever any declared sensor's last writer was the narrator". Red gate (g) already assumes this. The body
channels are sums over all declared sensors (the interoception channel: every drive; the world channel: 17
sensors, per `docs/wiring/cluster-dilution-blocks-situation-fear.md`). In an arc `--sim`, the narrator writes
`arms.thermal` once, and from then on every interoception encode is `narrated` until something else writes
that sensor. That includes the hunger drift, which is the body's own dynamics, so experienced hunger credit is
discounted for the rest of the session. Biologically, provenance belongs to an afferent, not to the percept it
contributes to. One narrated input does not turn a whole bodily state into narration.

**Fold (owner decision at GL3.B1, strict option first):**
- (a) **Strict, as drafted.** Keep the rule, but record its scope in §5.7. Make GL3.B1's gate 4 report the
  share of interoception-channel encodes stamped `narrated` on a cradle arc trace, so the discount's reach is
  measured, not discovered.
- (b) **Recommended on bio grounds.** Track last-writer provenance per sensor. Have the node's set record the
  kinds of the sensors that differed from the prior read (the sensors that moved), and decide explicitly
  whether a drift step by the body's own `tick_vital_drift` resets that sensor's last writer to `experienced`.
  Red gate (g)'s read surface assumes (a), so choosing (b) is the "owner-approved change" the GL3.B0 note
  requires, recorded before GL3.B1 flips it.

### S5 (§5.4 `mechano_proprio` row; GL3.B0 census): GL1 SF-7's per-sensor track mapping was folded into grounding.md but not into this plan or the census

**Issue.** `grounding.md`'s track table maps Minecraft sensors per sensor: vitals → `affective_slow`, distance
senses → `extero_detail`, `time_of_day` → its own clock, `speed` / `on_ground` / `look_pitch` →
`mechano_proprio`. It says this is "mapped per sensor in GL3.B0's census". This plan's §5.4 row still says
"Minecraft position/contact world sensors via the pump". `tests/unit/test_receptor_census.py` has no track
column (grep: no `mechano`, `affective_slow`, `extero_detail` or `time_of_day`), so the fold exists only in the
umbrella. Three sensors are not mapped anywhere: `y_altitude` and `distance_from_spawn` (allocentric position:
hippocampal / path-integration, not dorsal-column proprioception) and `is_in_water` (immersion: cutaneous plus
vestibular). Position is not proprioception.

**Fold (plan edit):** copy grounding.md's per-sensor mapping into §5.4's `mechano_proprio` and `extero_detail`
rows. Add `y_altitude` / `distance_from_spawn` → `extero_detail` (spatial) or a named spatial input, and
`is_in_water` → `mechano_proprio` (contact). Correct grounding.md's sentence to say the mapping lives in this
plan's table and is enforced by GL3.B3's track-membership test, not by the GL3.B0 census (which checks sites,
not membership).

### S6 (§5.4 `threat_low_road` row): the "low road" is fed by a learned, post-encoding EC cluster, which is the high (contextual) road; and it cannot be faster than the encode

**Issue.** LeDoux's low road is a thalamo-amygdala projection of **coarse, pre-cortical sensory features**. It
is fast because it skips cortical (and hippocampal) processing. Its existence in primates is also disputed
(Pessoa & Adolphs 2010). This row's emitter is "a situation-cluster change whose learned fear clears θ". That
is a hippocampal / EC context representation (the encoded situation) projecting to an amygdala-like store,
which is the *contextual* high road. Mechanically, a "cluster change" exists only after the summed channel is
encoded, which happens inside the substrate tick. So the track cannot deliver earlier than the
`anticipatory_threat_need` read it would replace. The "coarse, fast" property is named, but nothing in the
design produces it.

**Fold (plan edit):** rename the row's bio map to "contextual fear (hippocampus → amygdala)", or keep the
low-road name only for an emitter fed by raw pre-encode sensor features (for example a thresholded single
sensor such as `nearest_hostile_dist`, read before the summed encode), and note that the low road is contested
in primates. Since the bio-map string is recorded in the run fingerprint (§5.2 `AfferentTrackSpec` docstring),
fix it before any spec persists.

### S7 (§5.2 `ReceptorClass`, §5.3 stamp, TR16): `world.state` carries interoceptive vitals under one persisted `receptor_class`; `efference` and `linguistic` are not receptor classes

**Issue.** The `world.state` receptor reads 17 world sensors, including `health`, `food`, `saturation` and
`oxygen` (the TR16 double entry, kept on purpose), plus spatial and circadian ones. Its stamp,
`receptor_class` included, is persisted into `ec.json`'s `encoder_provenance` and the bundle manifest (§5.3). A
single `"exteroceptive"` there is a durable false label on a persisted wire. Two members of `ReceptorClass`
also misname their concept. **Efference** is a corollary discharge of a motor command. It is not afferent at
all, and its job is to predict and cancel *reafference*. **Linguistic** input arrives through auditory or
visual receptors, so it is an encoding, not a receptor class.

**Fold (plan edit):** allow `receptor_class` to be declared per sensor in the derived spec (it is derived from
YAML anyway, TR5) and stamp the per-sensor classes, or add an honest `"composite"` class for game-state
channels. Move `efference` out of `ReceptorClass` into a separate `SignalOrigin` (or `ProvenanceKind`-adjacent)
value with a reafference-prediction note, or drop it until a producer exists (no row in §5.8 emits it). Rename
`linguistic` to an encoding attribute, which `EncoderKind` already covers.

### S8 (§5.4 `nociceptive_fast` row; §5.5 refractory): the track's 1-pass refractory could drop the latch's "deepening" publish, adding a second, unbiological adaptation on top of the producer's

**Issue.** The body latch already does the bio-correct thing: it re-publishes on material deepening ("repeated
noxious stimulus of already-injured tissue sensitizes, it does not go silent", `embodiment/body.py` latch
comment). `nociceptive_fast` adds "1 pass per (receptor, locus) … / max" on top. The plan does not say whether a
publish inside the refractory window is **coalesced and delivered** or **dropped**. If it is dropped, a
re-injury on the next pass is silenced. Nociceptors sensitize; they do not adapt. Habituation and
sensitization belong at the producer (`adaptive_nociception.md`), as the autonomic plan already says ("never in
a consumer").

**Fold (plan edit):** define track refractory as "coalesce (max) and deliver at the window's end, never drop".
Add a GL3.B3 test: a deepening publish one pass after onset is delivered, with the deepened intensity. State in
§5.9 that the track adds no adaptation of its own.

---

## NIT

- **N1 (title and §1 framing).** The registry also routes imagination, language and body drives. Much
  interoception reaches the hypothalamus and insula through NTS / parabrachial without a thalamic relay, a
  spinal reflex never reaches the thalamus, and imagery is not afferent at all. The code names are already
  neutral. Add one sentence: "thalamic by analogy for the exteroceptive and nociceptive relay only; the
  registry is otherwise a registration and provenance point". Real thalamic gating is state-dependent
  (thalamic reticular nucleus, burst vs tonic mode). A static `enabled` plus a fixed `relay_gain` is a thin
  relay, which is fine, but it should not borrow the gating name.
- **N2 (§5.2 `AfferentEvent.salience` / `novelty`).** Neither field has a named producer or reader. This is the
  same dead-attention-field pattern the plan itself flags in gap 7 (`perceived_intensity`, #1120 lesson). Drop
  them until a producer exists (they can go in `extra`), or name one.
- **N3 (§5.4 / GL3.B3 docstring).** GL1 N4's fold, "FUNCTIONAL: one threshold for both fibre classes", is in
  grounding.md and autonomic_layer.md but not in this plan's track table or GL3.B3 build text, which is where
  the docstring is written. Copy it.
- **N4 (§5.4 `extero_detail` emitters).** "Minecraft events" are game-log **text** (chat / damage / death).
  They belong on `language`, as narrated or `reported` text (TR11), with `caused_by` to the sensor event. They
  are not cortical exteroception.
- **N5 (§5.4 `mechano_proprio` destinations).** "Cerebellum forward-model error" is fed in biology by the
  spinocerebellar tracts, not the dorsal column / medial lemniscus. Either the destination is fine and the bio
  map should name both, or say "FUNCTIONAL".
- **N6 (§5.4 latency note).** Second-pain delay depends on conduction distance (arm vs foot), and passes can
  stretch to 300 s in llm-primary (L3). "Ordinal, not metric" covers this. Add "and not distance-dependent" so
  no reader infers a fixed first/second pain interval.

---

## Verified

At `d9dffe6e` in `.worktrees/e-t8b`, by code read (read-only):
- `proprioception/pain.py::classify_pain`: `drive:` sources other than `TISSUE_DAMAGE_DRIVES =
  frozenset({"drive:health"})` → `PainKind.DRIVE`, so `drive:oxygen` is DRIVE. `::failure_pain_kind` does the same
  for FailureEvent names. The `PainKind` docstring: "DRIVE … air hunger, hunger, cold".
- `decisions/nac.py::DEFAULT_CLUSTER_FEAR_FAILURE_MODES = frozenset({"drive:health", "drive:oxygen"})`.
  `NAc.record_cluster_fear` is allowlist-filtered, accumulates, and clamps. `NAc.note_active_clusters` stores a
  per-agent dict.
- `proprioception/pain_bus.py::create_pain_cluster_fear_subscriber`: reads
  `nac.active_clusters(agent_id).get("world")` at publish time, and its docstring gives the W-4 "encode BEFORE
  pain tick" rationale.
- `runtime/loop_state.py::_substrate_tick_due`: requires `ctrl.pending_proposal is None`.
  `runtime/loop_gates.py::tick_embodiment_drift`: a no-op on substrate-primary (docstring).
  `runtime/substrate_proposal.py`: `note_active_clusters` precedes `embodiment.evaluate_failures()`, which
  precedes `anticipatory_threat_need` and `recommend_action`.
- `embodiment/body.py::evaluate_failures`: channel-2 drive pain is severity-latched (onset plus material
  deepening; hysteresis clear). Channel-1 FailureEvents are per call.
- `tests/fixtures/afferent_latency_characterization_v1.json`: damage arm publish +2 passes, `flee` +3, stale
  `mine_block` executed at pass 1. Drowning arm `drive:oxygen` publish +0, `escape_water` +25.
- `_data/reflexes/{humanoid,infant,quadruped}.yaml`: every response is `damage_component` (7) or
  `set_entity_sensor` (3). `embodiment/reflex.py` module docstring, "Known layering debt".
- `_data/components/bodies/minecraft_player.yaml`: the sensor list used in S5 (`health`, `food`,
  `light_level`, `y_altitude`, `nearest_hostile_dist`, `time_of_day`, `saturation`, `oxygen`,
  `hostile_count`, `distance_from_spawn`, `speed`, `on_ground`, `is_raining`, `is_in_water`, `xp_level`,
  `nearest_player_dist`, `look_pitch`).
- `tests/unit/test_receptor_census.py`: has no track-membership column (grep).
- grounding.md track table (per-sensor Minecraft mapping; "FUNCTIONAL: one threshold for both fibre classes")
  and `deferred/reflex_layering.md` (the five-stage table; stage 4 never executed).

**UNVERIFIED:** the status of the L1 issue (#1176), that is, whether it already plans per-pass transduction
(D2's option (a) depends on it; I did not query GitHub). Whether `ScriptedWorld` health reaches the body only
through the backend sync on the substrate tick or action path (D2 relies on `evaluate_failures` being the only
transducer, which I did verify). The literature points (Price 1977 wind-up; Reddan et al. 2018; Johnson & Raye
1981; LeDoux and Pessoa & Adolphs 2010; Gershman imaginative RL; chemoreceptive air hunger) are from my own
knowledge and were not checked in this session.
