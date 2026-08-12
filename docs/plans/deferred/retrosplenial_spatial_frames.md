# Retrosplenial complex — egocentric ↔ allocentric frame translation

**Status:** DEFERRED PLAN (2026-08-11). Design + trigger registration only; **do not
build yet**. Registered in [../bio_faithful_roadmap.md](../bio_faithful_roadmap.md) as an
OPEN gap on the spatial axis.
**Front-gate answer:** this needs to be its own mechanism *only if* the existing
sensor/EC path cannot represent "the same place from a different vantage" — §2 argues it
cannot, and names the falsifiable check that would prove otherwise.

## 1. Why the RSC is the right structure to name

The retrosplenial cortex is the brain's **frame translator**: it sits between
hippocampal/entorhinal allocentric representations (place and grid coding — where things
are *in the world*) and parietal egocentric ones (where things are *relative to me right
now*), and converts between them. Damage to it produces a specific, recognisable deficit:
patients can describe landmarks but cannot use them to navigate — they lose the mapping
from "what I see from here" to "where that is on the map."

**Naming caveat (bio-lens review):** the clinical syndrome is *heading disorientation*
(Takahashi 1997; Aguirre & D'Esposito 1999) — losing *direction* from recognised
landmarks, not landmark recognition itself (that is parahippocampal). And Maxim does NOT
have this deficit: a heading-disoriented patient *has* both frames and has lost the link,
whereas Maxim lacks both. Also note **postrhinal cortex** (LaChance/Taube 2019, an
explicit egocentric bearing map) is the closer homologue for the stage actually proposed
in §3; RSC earns its name only once an allocentric map exists to translate into.

What Maxim actually lacks is any *persisted* spatial frame, and that already costs us:

- **`azimuth` is explicitly head-relative** (`reachy_mini.yaml`: "Exteroceptive
  localization: head-relative azimuth of the current target"). It is a pure **egocentric**
  quantity. Turn the body and the same world-fixed sound source takes a different azimuth
  value — with no representation anywhere that it is the *same source*.
- **The head-frame incident** (CLAUDE.md invariant; Exp 45) is motivating colour, NOT
  evidence: it was a coordinate-convention bug in the *actuation API*
  (`goto_target(head=None)` re-solving IK against a retained world pose), strictly below
  the cognitive substrate. A cognitive frame module would not have caught it. It shows
  frame confusion is easy here; it does not show this module is the fix.
- **`memory/spatial.py` is type-only** ("No integration with Perception yet"), a
  reservation made for exactly this and never filled.

So the gap is real and already load-bearing, not speculative.

## 2. What today's architecture cannot represent (the front-gate test)

EC clusters percepts by embedding similarity per modality. An egocentric azimuth reading
of 0.3 clusters with other 0.3 readings — **regardless of where the body was pointing**.
Two consequences follow, and both are checkable:

- **Same place, different vantage → different clusters.** A sound at world-bearing 90°
  encountered while facing 0° (azimuth +90°) and while facing 90° (azimuth 0°) produces
  two unrelated EC nodes. Nothing binds them.
- **Different places, same vantage → one cluster.** Two genuinely different sources both
  at azimuth 0° from different body poses collapse into one node.

**Falsifiable pre-check — RUN 2026-08-11, then CORRECTED by a four-lens review
2026-08-11.** The first run used a modality that does not exist in production and asserted
a circular topology the sensor does not have. Corrected findings below; the superseded
numbers are kept struck-through because the correction is the point.

**Corrected measurement** (production path: `modality="audio"`, frozen centroids,
`SensorEncoderConfig.pattern_threshold = 0.85`):

| probe | ~~first run~~ | corrected | reading |
|---|---|---|---|
| H-C azimuth sweep, 21 readings | ~~2 nodes~~ | **3 nodes** | `-1.0…-0.2` / `-0.1…+0.5` / `+0.6…+1.0` |
| H-A one source, 5 vantages | ~~2 nodes~~ | **RETIRED** | 100% wrap artifact (see below) |
| H-B 5 places, identical reading | ~~1 node~~ | **STRUCK** | tautology (see below) |

**What actually holds — and it is much weaker than the first write-up claimed:**

1. **The partition is left / centre / right.** Applying this plan's own R3 criterion
   (boundary *placement*, not just count): 3 buckets split at roughly the two most
   meaningful points on this variable. Coarse but well-placed. This substantially softens
   the finding and was unreported in the first pass.
2. **~1.6 bits, not "one bit"** — and measured on the production path, not a
   non-existent `modality="sensor"` with centroid drift (drift *costs* resolution;
   frozen is consistently better by 1-2 buckets).
3. **The count is order-dependent (2-3 across shuffles).** The original single monotone
   sweep is the drift-maximising traversal. Any re-run must report mean ± range over
   ≥8 shuffles plus the isolated pairwise arc — this plan mandated that discipline in §5
   and then failed to apply it to its own headline.

**STRUCK — H-B (aliasing) was a tautology, twice over.** It fed five *bit-identical*
floats to a deterministic hash encoder; one node is the only possible outcome at any
threshold under any encoder. Worse, the `SensorEncoder` delta gate (`min_delta = 0.05`)
short-circuits calls 2-5 and returns the cached node — EC was never invoked
(`substrate_node_count == 1`). The claim "the real deficit is ALIASING" rested entirely
on this and is withdrawn. That a head-relative sensor cannot distinguish places is true
*a priori from its definition*; stating it as a measured finding launders a definition
into data. The version worth running is pose-conditioned: encode
`{azimuth, body_yaw, head_yaw}` and ask whether five distinct poses yield five nodes —
that tests the *proposed transform*, and is the argument this plan should rest on.

**STRUCK — the circularity claim, which was WRONG and would have BROKEN A GRADUATED
RESULT.** The first write-up said azimuth is circular and needs a ring code. It is not.
`doa_to_azimuth` (`embodiment/audio_localization.py`): *"−1 = left, 0 = centered (front),
+1 = right"* — **±1 are 180° apart, the two most behaviourally distinct directions on
this robot.** The linear encoding placing them at maximum cosine distance is *correct*.
A sin/cos ring code would identify hard-left with hard-right and destroy exactly the
left/right discrimination Exp 45's orient policy and **Exp 48's EARNED operant row**
depend on. H-A's 2-node result came from `wrap_norm` teleporting a hard-right source to
hard-left; with the sensor's real clamp it is 1 node, i.e. H-A was 100% artifact.

**The real topology:** the world is a circle; a linear mic array projects it **2:1 onto a
line segment** via front/back folding. The genuine degeneracy is at the **centre** of the
range (az ≈ 0 is front *or* back — a documented hardware limit), the exact inverse of the
original claim. **Blocking consequence for §3: an allocentric bearing is not achievable
from this sensor alone.** It needs dynamic-cue disambiguation across head movements or a
second modality. (The bio-story here is free and better-earned than the RSC one: mammals
resolve the same front/back cone of confusion by *moving the head* — Wallach 1940 — and
Maxim's orient loop already performs that movement.)

**Also corrected:**

4. **Not first measurement.** `scripts/orient_substrate/6_graded_orient_curve.py` (merged
   2026-07-22, PR #410) already recorded *"a single azimuth scalar folds into just 2 EC
   clusters at every threshold"* AND validated the fix — Gaussian direction-tuned cells,
   **6/6 distinct at width ≤ 0.15**. This pre-check re-derived known work under a less
   production-faithful configuration. Cite probe 6 as the prior art.
5. **The "no representation anywhere" framing in §1 is false.**
   `hardware/reachy/motor_backend.py::_frame_corrected_before` composes
   `cap_head + cap_body` and re-expresses capture-frame azimuth into the turn-entry
   frame, refusing (returns `None`) rather than guessing when stamps are missing. The
   ego→allo transform *ships*, for the within-turn credit path. The genuine gap is
   narrower: **no persisted, allocentric memory key** — the transform exists transiently
   and is discarded. Also: Maxim does not have the RSC deficit (a heading-disoriented
   patient *has* both frames and has lost the link); it lacks both frames.
6. **Scope correction — orient is NOT resolution-free.** The claim that Exp 45/49 "ride
   the drive VALUE, not cluster identity" is wrong: `recommend_action(current_cluster_id=…)`
   is cluster-keyed, and `drive_potential_diff` supplies only the reward *sign*, which is
   then booked *onto a cluster*. Orient needs ≥2 clusters splitting at az≈0 — one bit —
   and gets it (measured sign purity 1.00). Exp 48 is likewise cluster-keyed (that is the
   whole point of the extero/intero seam) and survives on boundary placement. Corrected,
   this *strengthens* the architecture: 3 well-placed buckets are why the seam worked.

The plan therefore STANDS but is **re-sequenced**: resolution before frames (§6).

## 3. Proposed shape (thin, riding existing seams)

Not a new bio-system with its own store. A **transform layer** between percept
production and substrate encoding:

- **Input:** an egocentric reading (azimuth, later: visual bearing, range) **plus** the
  pose it was taken from (`body_yaw`, head yaw — already available via
  `ReachyMiniController.get_current_pose()` and the retained-axes stash).
- **Output:** an allocentric bearing (world-frame) alongside the egocentric one — both
  carried, neither replacing the other. The egocentric value stays the control signal
  (orienting is inherently egocentric: "turn 30° right"); the allocentric value becomes
  the **memory key**.
- **Wiring:** a new `ModalityChannel` (`space`) so allocentric bearings cluster in EC
  independently of interoception and audio — the extero/intero split precedent
  (`record_outcome(clusters=…)`, Exp 48) is the template, and this is the third channel it
  was designed to admit.
- **Hippocampus:** allocentric bearing joins `Perception.location` via the already-reserved
  `memory/spatial.py` types, so episodes become *placed* and recall can be
  place-conditioned.
- **What it is NOT:** not SLAM, not a metric map, not grid cells. One angular frame
  transform plus a modality channel. Range/position come later, if ever.

**Bio-mapping tag at birth: FUNCTIONAL** — shares the RSC's translation role; implements
a rotation composition, not retrosplenial circuitry. (Per the roadmap's standing rule.)

## 4. Trigger — when this stops being deferred

Build when **any** of these fires (each is a task the system provably cannot do without
frame translation):

- **T1 — Place memory across motion.** A task requires remembering *where* something was
  after the body has moved (e.g. "the warm source is behind you now"). Today's substrate
  cannot express it.
- **T2 — Multi-vantage identity.** The cross-modal perception fabric (1.3) needs one
  identity for a source seen/heard from several poses; the §2 pre-check's fragmentation
  count is the blocker it hits.
- **T3 — Return-to-place.** Any navigation/foraging arc where reward depends on returning
  to a previously-rewarded location rather than reacting to a present cue.
- **T4 — A second frame-confusion incident** in hardware work (the head-frame class
  recurring is direct evidence the representation gap is operational, not theoretical).

Explicit non-trigger: better orienting performance. Orienting is egocentric and works;
this buys nothing there, and the front-gate rule says do not build it for elegance.

## 5. Re-validation registry (the operator's point, and the reason to register NOW)

Adding a `space` modality changes **EC cluster assignment**, which is upstream of
cluster-keyed reward bias — so it can perturb every result that rides on clustering, even
though none of them are "spatial" experiments. Registering the list before the work makes
that auditable rather than discovered afterwards:

| Experiment | Why it is at risk | Re-run requirement |
|---|---|---|
| Exp 42 / 42b (safe-vs-harm preference) | cluster-keyed `cluster_reward_bias`; a new modality changes which cluster credit lands in | full re-run, both counterbalanced arms |
| Exp 48 (cradle_mother operant) | already extero/intero split-dependent; a third channel changes routing | full re-run |
| Exp 45 / 45e (orient Layer 1) | azimuth is the input this plan reframes | re-run; ALSO re-baseline the DoA gain |
| Exp 49 / 50 (two-joint, re-adaptation) | pose-conditioned motor credit | re-run after 45 re-baselines |
| Exp 44b / 44c (counterfactual) | annotation is cluster-derived; §S2 context-aware rendering interacts directly | re-capture, do NOT re-score old captures |
| Exp 46 (operant orient) | scripted-substrate orient chain | re-run |

**Rule:** this work does not land on `main` until the table above is executed or each row
is explicitly waived with a written reason. Any Earned graduation row touched by it moves
to `Stale` on merge and blocks the next release until re-run — the existing
graduation-ledger discipline, applied in advance.

## 6. Sequencing with what exists

1. ~~§2 pre-check~~ **DONE 2026-08-11 — result re-sequenced everything below.**
2. **ANGULAR RESOLUTION — IF AND ONLY IF A CONSUMER NEEDS IT (see §4 non-trigger).**
   NOT a ring code (wrong topology — see §2). The validated option is the Gaussian
   direction-tuned population code already prototyped in
   `scripts/orient_substrate/6_graded_orient_curve.py` (6/6 buckets at width ≤ 0.15),
   promoted from script into `_read_exteroceptive_states`.
   Acceptance test: re-run `scripts/rsc_precheck.py`; H-C must resolve the range into
   many nodes and H-B must separate distinct places. Until this passes, every later step
   is decoration. NOTE this touches the EC drift lesson's territory — thresholds are
   pre-registered, never tuned on the outcome.
3. [cross_modal_perception_fabric.md](../cross_modal_perception_fabric.md) — its
   orient-windowed binding is where multi-vantage identity would consume this, and its
   suspected "2-cluster ceiling" is the same measurement (see §2 item 3).
4. RSC transform + `space` channel behind a flag, default OFF.
5. Re-validation table (§5) — note step 2 alone probably triggers most of it, since
   changing azimuth encoding changes cluster assignment.
6. Only then: place-conditioned recall in hippocampus.

## Open questions (decide at build time, not now)

- Whose pose is authoritative when head and body disagree — and does the answer change
  under `automatic_body_yaw`? (The retained-axes invariant says commanded, not read-back;
  a *memory* key may want achieved instead.)
- Does the allocentric bearing need a world anchor (a landmark, a compass) or is
  session-relative sufficient? Session-relative is cheaper and probably enough for
  T1/T3; T2 across sessions likely needs an anchor.
- Sim parity: `SimulatedController` must model the same frame composition, or sim and
  hardware will diverge exactly where this plan is supposed to help (the "your sim must
  model what the hardware does" lesson).
