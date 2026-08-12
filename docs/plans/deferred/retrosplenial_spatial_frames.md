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

Maxim has exactly that deficit today, and it has already cost us:

- **`azimuth` is explicitly head-relative** (`reachy_mini.yaml`: "Exteroceptive
  localization: head-relative azimuth of the current target"). It is a pure **egocentric**
  quantity. Turn the body and the same world-fixed sound source takes a different azimuth
  value — with no representation anywhere that it is the *same source*.
- **The head-frame incident** (CLAUDE.md invariant; Exp 45) was an ego↔allo confusion in
  hardware form: head pose is world-frame, `body_yaw` is another frame, and reading a
  head-mounted sensor while assuming the frames composed produced six successive false
  sensor pathologies. The system had no representation that could have caught it.
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

**Falsifiable pre-check (cheap, do before any build):** drive the existing orient stack
through a scripted sequence that visits one world-fixed source from ≥3 body yaws, dump
`sim_ec_activation` (`MAXIM_EC_TRACE_ACTIVATIONS=1`), and count distinct `active_node_id`s
for that source. If they collapse to one node, the sensor/EC path already generalises
across vantage and this whole plan is unnecessary — record that and close it. If they
fragment (expected), the fragmentation count is the baseline the RSC layer must reduce.

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

1. §2 pre-check (cheap, scripted, no build) — may close this plan outright.
2. [cross_modal_perception_fabric.md](../cross_modal_perception_fabric.md) lands first;
   its orient-windowed binding is where multi-vantage identity would consume this.
3. RSC transform + `space` channel behind a flag, default OFF.
4. Re-validation table (§5).
5. Only then: place-conditioned recall in hippocampus.

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
