# L11 world-channel diagnostic — MEASUREMENT-VALIDITY / CONFOUNDING lens

**Reviewer lens:** the ways this DIAGNOSTIC could return a WRONG or MISLEADING answer.
**Target:** `docs/plans/l11_world_channel_diagnostic.md` (PLAN DRAFT 2026-09-14).
**Context read:** exp58 prereg §Outcome + Addenda 3/4/5, `cluster-dilution-blocks-situation-fear.md`,
`docs/limits/l11_sensor_dilution.md`, `docs/agents/bio-memory.md`, `similarity/ec.py`.

The bar this instrument must clear is set by its own predecessors: `survival_phase0` and the
exp58 offline gates reported **1.0/1.0** separability by swinging light AND altitude across their
FULL ranges together — they validated an easier problem than the live single-axis classroom and
gave false confidence. Every finding below is a way this diagnostic repeats that failure one level
up (offline math on captured vectors is itself an "easier problem" than the live sequential
classroom) or a way its decision output can be read to bless a remedy the existing evidence already
rejected.

---

## DO-NOT-BUILD

### DNB-1 — Offline pairwise-cosine replay does not model the live clustering mechanism; the plan even names the wrong mechanism

**Weakness.** The plan's core metric (§3, §4) is "pairwise cosines safe↔dark vs the 0.85
threshold" recomputed offline on captured vectors, and it justifies the isolated/sequential dual
measurement by citing **running-mean centroid drift** (§3, open-q#3, Discipline). That is the wrong
mechanism for this channel. `world` is a **frozen-centroid modality**
(`ec.py::ECConfig.frozen_centroid_modalities = {"interoception","audio","world"}`, line 428).
Live clustering is therefore:

1. **First-touch prototype allocation, not running-mean.** The first vector to reach a node fixes
   the prototype; there is no running-mean update (`pattern_complete_or_separate` returns early at
   line 720 for frozen modalities). So the order-dependence is real but it is *first-touch*, not
   *drift*. A remedy validated against the drift concern is validated against a hazard this channel
   does not have, while the hazard it DOES have (which vector lands first sets the cluster every
   later vector is tested against) goes unmodelled.

2. **Completion is vector-vs-centroid over ALL stored same-modality nodes, never pairwise.**
   `matrix.scan(embedding, threshold, ...)` compares each new vector to every existing world node
   (line 698-699). Whether a dark vector merges with safe is NOT `cos(dark, safe) > 0.85`; it is
   "does dark complete onto ANY existing world cluster — safe, a gradient/mid-stairs cluster, or a
   walking-around/combat cluster from elsewhere in the session — before it separates." A pairwise
   safe↔dark cosine below 0.85 is **necessary but not sufficient** for a live split, and can be
   wrong in BOTH directions:
   - **False positive:** safe↔dark pairwise reads separable, but live the dark vector completes
     onto an intermediate gradient cluster (or safe completes onto dark's neighbourhood) → merged
     live, "separates" on the diagnostic.
   - **False negative:** pairwise reads merged on one low-contrast sample, but the assigned-id
     SETS are actually disjoint across the jitter distribution → "infeasible" reported when a
     remedy would work.

3. **Offline replay on the captured sample omits cross-traffic crowding.** The live world channel
   is shared across the whole session and accumulates clusters from every world state the agent
   visits (walking, combat, mid-stairs). A fresh-EC or capture-only-sample replay scans against a
   near-empty store and so UNDER-estimates cluster crowding — exactly the "easier problem" pattern.
   (Also note the `l11_sensor_dilution.md` harness note: `pattern_complete_or_separate` allocates
   an id WITHOUT registering it — a replay that skips `register_substrate_node` measures a
   stateless EC that "never remembers anything." The first bake-off run hit this and reported every
   arm at stability 0.00.)

**Failure it causes.** The diagnostic's headline decision ("which remedy separates / infeasible")
is computed on a proxy that is not the production clustering rule. It can green-light a remedy that
merges live, or kill one that works live — the same false-confidence class that produced the
Phase-0 1.0/1.0 and cost the whole Exp 58 build.

**Minimal fix.**
- Replay must reproduce the **production sequential first-touch protocol through the real EC**
  (`pattern_complete_or_separate` + `register_substrate_node`, frozen-world semantics, the same
  `geometry` tag), not offline pairwise cosine, and must include representative **cross-traffic**
  world vectors, not only the safe/dark contrast pair. The metric is **cluster-id ASSIGNMENT**
  (are the id-sets for safe vs dark disjoint across the jittered sample), not a pairwise margin.
- The isolated/sequential dual measurement stays MANDATORY but is re-justified as first-touch
  order-dependence; additionally, any remedy that introduces NEW **unfrozen** sub-channels (a
  channel-split could) reintroduces TRUE running-mean drift and must be measured for it too.
- **A live re-encode of the top candidate is MANDATORY, not optional** (resolves the plan's
  open-q#3 in the strong direction). The winning remedy must be re-encoded in the full live
  classroom, under real jitter and cross-traffic, and must PASS the exact `cluster-distinct`
  preflight that blocked Exp 58 (prereg Addendum 4/5) before B recommends BUILD. Offline replay
  alone selects the candidate to live-test; it never issues the BUILD verdict.

---

### DNB-2 — No pre-registered, objective "separates" criterion; and separation-only re-selects the arm the existing bake-off already rejected

**Weakness.** The deliverable is "which remedy separates … or infeasible," but nowhere does the
plan pre-register **what "separates" means numerically**, **how many samples**, or **which
composite metric**. Two compounding hazards:

1. **Post-hoc threshold.** With the criterion unfixed, the remedy comparison table can be read
   after the fact to make channel-split (the plan's pre-named "leading candidate," per exp58
   §Outcome) look good. That is the D43/D44 "measured a possibility, presented as proof" shape,
   and it is exactly how a 1.0/1.0 got celebrated once.

2. **Separation-only is a WEAKER instrument than the one that already exists.** `l11_sensor_dilution.md`'s
   bake-off already scored these remedies on **PRIMARY = min(separation, stability, discrimination)**
   and found: grouping-alone (A2) ≈ 0.00 at N=100; grouping+scaled-threshold (A3) is **worse** than
   threshold-alone and unstable (stability 0.56–0.62) because shrinking per-channel N loosens
   `1−k/N` and lets NOISE separate; gain+threshold (A5) collapses stability to 0.00; A4 gain is the
   selected mitigation and is ALREADY the world default. A diagnostic that ranks candidates on
   separation alone will happily re-pick A3-shaped channel-split (tiny per-channel N "separates" the
   captured pair) — the arm the full metric already rejected. It would ship a remedy whose apparent
   separation is noise finding room in a small-N channel.

**Failure it causes.** The instrument endorses a remedy that separates the captured pair but is
unstable / non-discriminating live — merging under jitter (the very failure mode Exp 58 Addendum 4
observed: one spot → a neighbourhood of ids). "Separates" on the diagnostic, merges in production.

**Minimal fix.** Pre-register, BEFORE any capture: (a) the exact situations and their real
magnitudes; (b) the sample size + the stability fraction that counts as a pass; (c) the composite
criterion **min(separation, stability, discrimination)** carried over verbatim from the bake-off
(reuse `encoding_bakeoff.py`'s metric, don't invent a looser one); (d) the numeric bar for each.
Stability and discrimination are non-negotiable — an id-set that is disjoint on average but flips
under jitter is a FAIL, not a pass.

---

### DNB-3 — The Exp 56/57 non-regression check is the load-bearing half of "B GATES C," yet it is admitted-uncertain and effectively punted to C

**Weakness.** The plan asserts "**B GATES C**: no substrate change is built until this diagnostic
proves … it does not break the separations Exp 56/57 rely on" (lines 8-10), but §5 + open-q#4 admit
it may "need the Exp 56 apparatus standing (a scope/sequencing question for B vs C)," and the
Discipline section downgrades it to "a re-baseline the C-plan must own explicitly." You cannot have
it both ways: if the non-regression measurement is not concretely specified and MANDATORY inside B,
then B does not gate C on its load-bearing criterion, and the plan's central promise is hollow.

**Failure it causes.** Channel-splitting the `world` modality changes world cluster ids globally.
If a taught-want (Exp 56/57) is keyed on world clusters, splitting re-keys them and **silently
breaks an EARNED behavioral result** (Exp 56 is the 1.2 headline). A remedy chosen on survival
separation alone, with the preservation check deferred, can ship that breakage — and "B gated C"
will have been asserted while the gate's second half never ran. (Same family as the frozen-modality
merge hazard in bio-memory: `ec_merge` must respect `frozen_centroid_modalities` — a schema change
here interacts with that invariant.)

**Minimal fix.** Make the non-regression capture CONCRETE and MANDATORY in B: capture Exp 56/57
taught-want situation vectors through the same production encoder, and require every candidate
remedy to preserve their cluster assignments (id-set stability, same composite metric) as a
hard pass/fail — ranked identically to the survival separation. If capturing Exp 56/57 vectors
genuinely requires their apparatus, then that capture is IN SCOPE for B and the plan must say so;
it may not be relabelled a C-owned re-baseline while B still claims to gate C. State which modality
Exp 56/57 wants are keyed on (world vs interoception) up front — if world, the interaction is
direct and this is DO-NOT-BUILD-as-specified until it's mandatory.

---

## SHOULD-FIX

### SF-1 — Situation sample: no count, boundary-straddling jitter, and apparatus-dependence that C can silently invalidate

- **No sample count / no boundary-crossing statistic.** Exp 58 Addendum 4 already observed the
  same physical spot re-completing to **2 different cluster ids (6/4 split)** as the 17-sensor
  vector jitters across the 0.85 boundary. A handful of samples per situation can miss the merge
  tail entirely. Pre-register N samples and report the boundary-crossing RATE with a confidence
  interval, not single points or an unquantified "multiple."
- **hostile_count counts-all bug bakes an apparatus artefact into the vectors.** The bridge counts
  ALL loaded hostiles, so the persistent `clustermob` (Addendum 5) shows at *safe* too and
  `hostile_count` is identical at both situations (documented in the wiring doc). The captured
  vectors therefore encode a specific, buggy apparatus configuration. If the diagnostic captures on
  the Addendum-5 apparatus but a revived Exp 58 (C) uses a different one, B validated a different
  problem — the exact trap again.
- **Circularity: the remedy shapes the apparatus, which shapes the vectors B should have measured.**
  B picks the remedy; the remedy shapes C's classroom; C's classroom shapes the real contrast. Pin,
  in B, the exact apparatus/contrast the diagnostic assumes, and make it a stop-rule that C may not
  change the sensor set / ranges / apparatus without re-running B.

### SF-2 — D1 tuning-to-the-apparatus risk is live and named in the wiring doc but not guarded in the plan

`cluster-dilution-blocks-situation-fear.md` explicitly warns: "Don't force separation by tuning a
sensor's range to the apparatus (e.g. narrowing `y_altitude` so 12 blocks reads as a big swing) …
it is D1-adjacent." The remedy replay includes "gain exponent variants" and channel-split with
small per-channel N — both are knobs that can manufacture separation on the captured sample without
a real contrast (A3/A5's noise-separation, measured). The plan must forbid range-narrowing-to-force
and any exponent/threshold tuning chosen to pass THIS sample, and must require that any range or
gain change be justified against the drive comfort-band semantics (A4's actual principle), not
against "it made safe/dark split." The stability+discrimination criterion from DNB-2 is the
mechanical guard for this.

### SF-3 — "Recompute over the same code" is stated as a goal but the failure mode is unstated

Open-q#1 asks whether the telemetry is non-perturbing and re-implementation-free. The concrete
risk to name: the read-only telemetry compares against "current centroids," which for frozen-world
ARE the accumulated live first-touch prototypes — so what it reports depends on the EC's
accumulated session state at capture time. A fresh-EC offline replay reports a DIFFERENT geometry
than the live EC that also saw all the session's other world states (the DNB-1 cross-traffic point).
The plan should require: telemetry runs the production `pattern_complete_readonly` for pure
observation (never the mutating `_or_separate`, per the D8 bio-memory invariant), the replay uses
the real register protocol, and the report states explicitly whether numbers are against a
fresh EC, the capture-time live EC, or a cross-traffic-seeded EC — because those are three
different problems and only the last is the live one.

---

## NIT

### NIT-1 — Reframe the isolated/sequential rationale from "running-mean drift" to "first-touch order-dependence"

Folded into DNB-1's fix; recorded separately because the plan repeats the "running-mean centroid
drift" phrase in three places (§3, open-q#3, Discipline) and a reader will otherwise validate the
wrong hazard. `world` is frozen; the order-dependence is which vector lands first + which prior
clusters exist. Keep the dual measurement; fix the reason. (And re-flag: a channel-split that
mints new UNFROZEN sub-channels reintroduces real running-mean drift — so the old phrasing becomes
correct precisely for the remedy under test, which is a reason to measure both, not to drop it.)

---

## Bottom line

The diagnostic's INSTINCT is right (measure the real contrast at real magnitude, don't trust the
synthetic law). But as drafted its MEASUREMENT is offline pairwise cosine over a proxy that is not
the production clustering rule (frozen first-touch, whole-store scan, cross-traffic), its DECISION
criterion is unregistered and weaker than the bake-off instrument that already exists, and its
load-bearing non-regression gate is admitted-uncertain. Any of the three DNBs alone lets the
diagnostic hand back a confident wrong answer — the specific failure it was created to prevent.
The through-line fix: replay through the real sequential EC protocol on cluster-ID assignment with
the bake-off's composite min(sep,stab,disc), pre-register the bar + sample before capture, make
Exp 56/57 preservation a mandatory in-B pass/fail, and make a live re-encode past the
cluster-distinct preflight the ONLY thing that issues a BUILD verdict.
