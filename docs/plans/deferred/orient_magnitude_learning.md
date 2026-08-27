# Orient magnitude learning — and the parietal/cerebellar division of labor

> **⏸ DEFERRED 2026-08-27 — S0/S1 shipped 2026-07-16 and **S4 completed via Exp 45e (2026-07-27, magnitude 1.00, readout-independent)**; only S3 (AG abstraction of the learned table) is open and nothing owns it. **Revive when** a held-out-bin generalization claim is pre-registered or a cross-state deriver exists.**

**Status:** **S0 DONE; S1 SHIPPED but NOT sufficient (replication correction below).**
S0 (magnitude action set) + S1 (`--flip-bins`) both passed on hardware 2026-07-16, and 45c
scored magnitude 1.00 — but that was **n=1**. The [Exp 45d](../../experiments/45d_magnitude_replication.md)
replication (2026-07-23, 3 clean seeds) lands magnitude at **0.75–1.00, mode 0.75**: direction
unanimous 1.00, but the 1.00-magnitude draw does not reliably recur. **S1 is necessary but not
sufficient** — the residual is a **single-far-bin big-turn-cell starvation** (per seed, exactly
one far bin never gets a positive exploration sample of its big turn), which is a coverage limit
of per-cell tabular argmax. That residual is now S3's concrete motivation and the new S4's target
(below). S2 motivation weakened by measurement. Original draft 2026-07-16; replication +
S3/S4 update 2026-07-23. Follow-on to [substrate_native_orienting.md](../substrate_native_orienting.md)
Layer 1 / [Exp 45](../../experiments/45_reachy_orient_live.md) (all arms EARNED). Scope: the
**magnitude** half of the orient policy, and — as its natural test case — whether the
bio-regions that *should* own continuous magnitude (IPS, Angular Gyrus, Cerebellum) are
wired to carry this kind of learning.

**Trigger (live observation, 2026-07-16):** watching the learned policy run
(`orient_demo.py`), the robot turns the right *way* but never learns how *far* — it
takes the same step from 0.15 and from 0.65 azimuth.

**Root cause — NOT bin resolution (the first hypothesis, and it was wrong).** The
policy can't express magnitude: the action set is exactly two affordances,
`turn_left`/`turn_right`, both hard-fixed at ±0.25 rad in the dispatch. Finer bins
would teach the same fixed-step policy in more states. **Bins are what the robot can
sense; actions are what it can express.** The gap is on the expression side. (Phase 0b's
sim backbone had 5 magnitude-varying actions; the hardware bring-up simplified to 2.)

**The credit signal already discriminates magnitude for free.** `potential_diff`
punishes a big step from `near_*` (overshoot → small/negative relief) and rewards it
from `far_*` (more relief per action). Nothing new is needed to *teach* magnitude —
only to *offer* it.

---

## Audit before building (what is actually wired)

| Region | Exists? | Wired to what | Shape |
|---|---|---|---|
| **IPS** ([math/ips.py](../../../src/maxim/math/ips.py)) | ✅ | `memory_hub` (constructs `IPS()`), `maxim_agent`, `statistician_agent`, `math_tool`, `concept_grounder` | **Stateless** Approximate Number System — Weber-Fechner log compression; `compare` / `categorize` / `detect_trend`. Cannot store; it is magnitude *intuition*, not magnitude *memory*. |
| **Angular Gyrus** ([math/angular_gyrus.py](../../../src/maxim/math/angular_gyrus.py)) | ✅ | `build_bio_stack` (persisted `angular_gyrus.json`), `memory_hub`, `maxim_agent`, `create.py` | MemoryLayer **+** compute engine: math facts, methods, **patterns**; dual verbal/code. The one region here that could *hold* a relationship. |
| **IPS → AG escalation** ([agents/statistician_agent.py](../../../src/maxim/agents/statistician_agent.py)) | ✅ | — | Already implemented: "dual number system (IPS fast/approximate + AG slow/precise)"; escalates to AG when IPS confidence is 0.3–0.65. **The abstraction pathway the magnitude idea wants already exists.** |
| **Cerebellum** ([embodiment/cerebellum.py](../../../src/maxim/embodiment/cerebellum.py)) | ✅ | `build_bio_stack` | Learned **forward** models (command → predicted sensory consequence) via Rescorla-Wagner prediction error, keyed `(entity, modulator, affordance, param_bucket)`. |

**Gap 1 (load-bearing): the orient loop is wired to NONE of them.** The Exp 45 scripts
construct a bare `NAc(NACConfig())` — no `build_bio_stack`, no MemoryHub, no
Statistician, no IPS/AG/Cerebellum. So "is this wired properly?" has an honest answer
today: *the regions are wired to each other; the orient task talks to none of them.*
This line is therefore as much **wire-it-up** as **test-it** — and that is the more
interesting version of the exercise, because the orient task is the cleanest possible
integration probe (1-D, deterministic reward, real hardware, no LLM).

**Gap 2 (shapes the AG step): the Statistician detects patterns in per-metric TIME
SERIES** (`_metric_series: dict[str, deque[float]]`, PatternDetector FSM, fed by
`_on_tool_result` / `_on_goal_completed`). What AG-abstraction of the orient policy
needs is a **cross-state function shape** ("preferred magnitude scales with |az|"), not
a trend within one metric's history. The existing escalation pathway does not have that
shape. S3 below is therefore genuinely new mechanism, not a wiring exercise — and that
is why it ranks last.

---

## The four candidate steps (ranked; independently pre-registered)

### S0 — magnitude action set (Exp 45b). **DONE — PASSED on hardware 2026-07-16.**

**Result: direction 1.00, magnitude 0.75** (the pre-registered bar, and the exact
predicted value), stable across 8 consecutive probes. `near_left` learned `turn_left`
(+0.185) over `turn_left_big` (+0.072) — the substrate declining to overshoot, from
relief alone. Full record: **[Exp 45b](../../experiments/45b_orient_magnitude.md)**.
**Prerequisite discovered en route:** the `head=None` counter-rotation bug (the mics
never turned); mag1 was incoherent, mag2 passed immediately after the fix.

`bodies/reachy_mini.yaml` `orient` now declares 2×2: `turn_left`/`turn_right` (±0.3 rad,
names+values unchanged → **queen-mind v0.1 still loads**) plus `turn_left_big`/
`turn_right_big` (±0.9). Scripts read YAML magnitudes directly (`--step` → `--step-scale`).
Full design + metrics + diagnostics: **[Exp 45b](../../experiments/45b_orient_magnitude.md)**.
- **Front-gate:** rides existing infra (YAML + NAc + `potential_diff`). No new mechanism.
- **CORRECTION to this plan's first sketch (±0.12/±0.4 would have FAILED):**
  `potential_diff` has no cost for large moves, so big wins everywhere *unless it
  overshoots near center*. At the gain assumed then (0.58 — from the contaminated pre-headfix sweep; it happened to land within noise of the true post-fix 0.55-0.57), ±0.12/±0.4 never overshoots
  → the policy would learn "always big". **0.3 normal / 0.9 big** is what makes magnitude
  learnable (big: −0.16 relief from az 0.18, +0.52 from az 0.60).
- **Sim predicted 0.75, hardware delivered 0.75.** Sim (seeds 0–2) said direction 1.00
  robust, magnitude seed-dependent ~0.75–1.00; the pre-registration committed to "expect
  0.75, not 1.00" and that is exactly what the robot did.
- **Walked-motion fold (rationale RETRACTED):** the ≤0.3 rad walk was added for "DoA
  lock safety" per a tracking-estimator finding that turned out to be the head-frame
  bug. Behaviour kept (harmless), rationale withdrawn.
- **Note:** trips Exp 45's "orient-affordance YAML change" re-run rule → fresh NAc,
  queen-mind **v0.2**.

### S1 — bin boundaries at the DECISION BOUNDARY (was: "Weber-scaled bins"). **SHIPPED, necessary-not-sufficient ([Exp 45c](../../experiments/45c_flip_bins.md) n=1 got 1.00; [Exp 45d](../../experiments/45d_magnitude_replication.md) replication says 0.75–1.00, mode 0.75).**

**Original result (45c, n=1): magnitude 0.75 → 1.00**, direction 1.00, stable across 13
consecutive probes; greedy turned-toward 0.286 → 1.000 (below chance → perfect). Every bin
decisive; `near_right` learned the big step is *harmful* there (−0.570). Two derived constants, no
new mechanism, no effort cost, no exploration change. Sim predicted 0.92; hardware gave
1.00. Implemented as `--flip-bins` (`decision_boundary()` + `placement_ranges()` in
live_common, both derived from the robot's own measured gain).

**REPLICATION CORRECTION (Exp 45d, 2026-07-23, 3 clean seeds, boundary frozen at the derived
0.330 via `--az-gain 0.55`):** direction stays **unanimous 1.00**, but magnitude lands
**{0.75, 1.00, 0.75}** — the 45c 1.00 was the lucky draw (the sim already flagged 0.75 on 2 of 6
seeds). So the flip-point boundary **removes the straddle** (the near bins now resolve cleanly —
that half of the fix is real and replicated) but does **not** by itself deliver reliable 1.00
magnitude. The residual is a *different* failure from the straddle: **single-far-bin big-turn-cell
starvation** — both far bins need the big turn, they are learned independently, and 40 trials at
ε=0.25 doesn't guarantee every (far bin × big) cell draws a positive sample (seed 3: far_right
learned `turn_right_big` at +0.585 while far_left stayed exactly 0.0). Seed 2 proves 1.00 is
reachable when both far cells happen to be covered → **coverage limit of per-cell tabular argmax,
not a capability limit.** This is what promotes S3 from speculative to motivated and defines the
new S4 target below.

**Reframed 2026-07-16 by the post-headfix sweep.** The problem is not uniform-vs-log
bins — it is that the `near` bin **straddles the decision boundary**, so it holds two
opposite correct answers. The boundary is *derived*, not guessed: a step of shift
`S = |delta| * gain` takes `|az| → |az - S|`, so step A beats B exactly when az is
NEARER A's shift — the boundary is their midpoint:

    az_boundary = gain * (|delta_big| + |delta_normal|) / 2   # Reachy: 0.546 * 1.2/2 = 0.328

*(Correction: first written as `|delta_big| * gain / 2` = 0.246 — that is where big's
relief crosses zero, which decides nothing. Caught by deriving instead of reasoning.)*

Current `az_bin`: center ≤0.1, near 0.1-0.5, far >0.5 — **near straddles 0.328**, which
is why Exp 45b measured 0.75 (`near_right` drew 0.44/0.49, *above* the boundary, so
learning big there was CORRECT; `near_left` drew lower ones and learned normal — same
bin, opposite lessons, both right).
- **The fix (implemented, `--flip-bins`):** boundary at the derived value; placements
  near [0.16, 0.27], far [0.39, 0.80] (the cap widens from 0.65 — the post-headfix sweep
  is monotonic to |az| ≈ 0.87, so the endfire cap was chasing an artifact). Each bin then
  holds ONE correct magnitude.
- **Sim result (6 seeds, stationary source):** legacy **0.375** vs derived **0.92**
  (4/6 perfect). Pre-registered on hardware as [Exp 45c](../../experiments/45c_flip_bins.md).
- **Front-gate:** this is a boundary constant, not a mechanism — ~2 lines. **IPS
  routing is NOT needed** and would be bio-naming theater on arithmetic; the honest
  version of the Weber intuition is "put the boundary where the physics changes," and
  the physics tells us where.
- **Ties to S2:** `az_flip` depends on the robot's *measured* gain, so a robot that
  calibrates its own gain also derives its own bin boundaries. That is a stronger
  portability argument for S2 than the (retracted) drift claim.
- **Pre-registered metric:** magnitude appropriateness 0.75 → **1.00** at equal trial
  count, with the flip-point boundary and nothing else changed.

### S2 — cerebellar gain calibration (inverse model). **Motivation WEAKENED — read this before building.**

**Honest correction (2026-07-16):** this step's original motivation was "the gain is not
a constant — it drifted 0.58 → 0.39 between sessions, so the robot must learn its own."
**That drift was the head-frame bug, not physics.** Post-fix the gain is *stable and
reproducible* (0.562 / 0.574 / 0.549 / 0.58 across independent measurements). The
"sensor is unstable, therefore adaptive calibration" argument is **retracted**.

**Update (2026-07-16): this weaker argument just became LOAD-BEARING.** The runtime
plan ([orient_runtime_integration.md](../archive/orient_runtime_integration.md)) gates
"learning ON in production" on exactly it: the decision boundary derives from the gain,
so a robot learning in a stranger's room with an inherited gain learns the *wrong
magnitudes*, confidently. S2 is no longer a nice-to-have research step — it is the gate
for shipping a learning reflex. Cheapest form: passive EMA over the reflex's own
`(yaw_delta, Δaz)` trials — free, no boot ritual.

What survives is weaker but real: gain genuinely varies across *rooms, source distances,
mounts, and robots* (and the eared-shell mod will change it deliberately), so a robot
that measures its own transfer function is more portable than one inheriting a constant
from a doc. Biologically, adaptive sensorimotor gain is *the* cerebellar job (VOR gain
adaptation is the textbook case), and the module already learns from prediction error.
But this is now a **portability** argument, not a **the-sensor-drifts** argument — and it
should be front-gated on that weaker basis.
- **Front-gate:** the existing cerebellum learns **forward** models; orienting needs the
  **inverse** (error → command). That is a genuine extension, not a wiring change — name
  it as such.
- **Pre-registered claim:** the robot learns its own gain from experience (converges to
  ≈0.55-0.57 (post-headfix; the retracted 0.58 was named here originally)), and **re-calibrates** when the gain changes
  (verifiable by changing `--step` or the sweep-measured mount/acoustics — the eared
  shell will change it for real).
- **Rigor caution:** with one parameter, "learned the gain" is a thin claim on its own;
  it earns its weight from the *re-calibration* arm, not the initial fit.

### S3 — AG abstraction of the learned table. **Most interesting — and now empirically motivated (Exp 45d).**

**Concrete motivation added 2026-07-23:** S3 was "most speculative" until 45d's replication
handed it a *measured, reproducing* failure to fix. The far-bin starvation IS this step's
falsifiable test case in the flesh: far_left×big is a bin the tabular policy **never positively
visited**, so tabular NAc leaves it at 0.0 and mis-picks `turn_left` there — exactly "predict the
right action in bins NAc never visited." An AG abstraction that observes "far → big" learned
decisively at far_right should fill far_left from the *symmetry of its own learned table*, with no
new far_left×big sample. So S3's held-out-bin claim is no longer a thought experiment; 45d says
which held-out bin, how often (~⅔ of seeds at 40 trials), and gives the baseline it must beat
(tabular = 0.75). (S4 below attacks the same failure with a lighter, non-declarative mechanism —
they are alternatives, not a stack; whichever earns it, the *other* becomes the comparison arm.)

**Not** "fit a polynomial to (azimuth → turn)" — that regresses the analytic servo from
sensor data with the substrate bypassed, and collapses the learned-vs-servo distinction
Exp 45 exists to defend (the optimal policy here is `turn = -az/gain` (~0.55-0.57 post-headfix), a straight line;
fitting it *is* writing the controller). **Instead:** after training, NAc's bias table
*is* a sampled function — `(bin, action) → value` over bins that carry magnitude. A
derivation step observes the regularity **across the substrate's own learned table**
and promotes the compact relationship into AG's pattern memory. The substrate learns
procedurally from relief first; AG abstracts declaratively from *its learning*. That is
the AG's actual biological job.
- **Front-gate:** needs a deriver that does not exist (Gap 2 — the Statistician's shape
  is time-series, not cross-state). New mechanism; must argue why the existing
  IPS→AG escalation cannot be reshaped to carry it.
- **Pre-registered claim (falsifiable, and the reason this is worth doing):** does the
  AG-abstracted relationship correctly predict the right action in **bins NAc never
  visited**? Tabular NAc cannot; an abstraction can. Held-out bins are the test.

### S4 — population-vector readout (graded selection over the tabular biases). **NEW (Exp 45d-motivated).**

The starvation is a *readout* problem before it is an abstraction problem: the learned biases
are hard-`argmax`'d over a hard `az_bin`. Two hard steps. S4 softens the **selection** without
learning any new model: read out the commanded turn as a `cluster_reward_bias`-weighted average
of the action deltas across the active + *neighbouring* bins (optionally with graded EC membership
instead of a hard bin). This is the superior-colliculus population-vector: discrete substrate
(unchanged — still merges via `nac_merge`, still transfers cross-session), continuous output. It
attacks the exact 45d failure — the two symmetric far bins **share** the "far → big" evidence
through the weighted blend, so far_left borrows what far_right learned decisively instead of
needing its own big-cell sample.
- **Distinct from S2 and S3:** S2 learns a new (inverse) model; S3 derives a declarative
  abstraction; S4 changes only how the *existing* tabular values are combined at selection time.
  Cheapest of the three, keeps the learning primitive intact (learn discrete, act continuous).
- **Front-gate:** rides the existing `NAc.recommend_action` chokepoint + `cluster_reward_bias`
  surface — a readout change, not a new mechanism. Keeps the discrete probe available (argmax is
  still computable) so the learned-vs-servo rigor bar survives.
- **Honest IPS caveat (self-correction, 2026-07-23):** S1 already, correctly, rejected
  IPS/Weber-scaled **bins** as "bio-naming theater on arithmetic" — the *boundary* is derivable
  from physics, so log-spacing the boundary buys nothing. That verdict stands and S4 does **not**
  reopen it. Where the Weber/log-magnitude intuition *might* still apply is the **readout
  geometry** (the SC amplitude map is roughly log-scaled), and whether that beats a linear blend is
  an *arm to test*, not an assumption — the default S4 is a flat weighted average; the log-scaled
  readout is a secondary arm that must earn its coupling to `math/ips.py`, same bar S1 held it to.
- **Pre-registered metric:** the RIGHT yardstick here is NOT discrete argmax-correctness (it can't
  see continuous improvement) but a **continuous residual** — mean `|az_after|` per trial (or
  steps-to-center). Claim: S4 pushes residual `|az|` below the tabular quantizer's floor AND
  dissolves the far-bin starvation (no per-far-cell coverage requirement, boundary-tuning-free).
- **Pre-registered risk (the make-or-break):** sample efficiency. Argmax commits on one bin's data;
  a population vector needs enough of the map populated to average well. Does it still converge in
  ~40 hardware trials, or need ~80? If the continuous residual gain is marginal at 2–3× the
  hardware cost, the pragmatic ship stays tabular-with-`--flip-bins` and S4 is a research result —
  which is why the tabular baseline (45d) is kept as the comparison arm.

---

## Sequencing

1. ~~**S0**~~ **DONE** (Exp 45b) — magnitude action set; strengthened the Exp 45 claim
   ("learned which way **and how far**, from relief alone"). Gated queen-mind v0.2.
2. ~~**S1**~~ **SHIPPED** (`--flip-bins`, Exp 45c) — removed the near-bin straddle. Replicated
   (Exp 45d): necessary-not-sufficient; residual = far-bin starvation.
3. **S4 next (the cheapest attack on the 45d residual)** — population-vector readout. Rides the
   existing `recommend_action` chokepoint, keeps the substrate mergeable/transferable, and its
   continuous-residual metric measures the thing S1's argmax-correctness can't see. Pairs with
   **Gap 1** (run the orient loop through `build_bio_stack` rather than a bare `NAc`) since a graded
   EC-membership readout wants the real substrate; converge with the 1.1 `--embodiment`
   hardware-runtime work, do it once.
4. **S3** as the S4 alternative / comparison arm — AG declarative abstraction of the learned table.
   Same held-out-bin target as S4, heavier mechanism (needs a cross-state deriver, Gap 2). Run it
   *against* S4's result, not before: whichever earns the starvation fix, the other is its baseline.
5. **S2** independently on the **portability/ship-gate** track (not the starvation track) — the
   inverse-model gain calibration that [orient_runtime_integration.md](../archive/orient_runtime_integration.md)
   gates "learning ON in production" on; natural pairing with the eared-shell experiment, which
   changes the gain for real. Needs no bio-stack.

**Do not blob these.** Separate pre-registrations, separable claims — the cradle-cascade lesson
(don't stack unverifiable layers) applies exactly. S3 and S4 are **alternatives** for the same
failure, not a stack: run one, keep the tabular 45d result and the other mechanism as comparison
arms.

## Open questions

1. **S1 front-gate:** IPS routing vs 3 lines of log-spacing — what does IPS's Weber
   calibration buy that justifies the coupling?
2. **S3 deriver:** reshape the Statistician (cross-state pattern support) vs a new
   derivation step? Front-gate argument required either way.
3. ~~**Does S0 make S1 moot?**~~ **ANSWERED (2026-07-16, sim): no — S0's residual noise
   IS S1's motivation.** The `near` bin (|az| 0.1–0.5) *spans the flip point*: big is
   wrong at az 0.18 (−0.16 relief) and right at 0.42 (+0.32). A bin whose correct action
   changes inside it is under-resolved by construction — bin-averaging works but is noisy
   and exploration-fragile, exactly as the seed sweep shows. Weber bins put the flip
   *between* bins. Now evidence-backed rather than speculative.
4. **Integration-test framing:** once Gap 1 closes, the orient task becomes the
   project's cleanest bio-region integration probe. Worth a standing "does the whole
   stack still carry a real sensorimotor policy" regression run?
5. **S3 vs S4 for the starvation (NEW, Exp 45d):** both target the same held-out-bin failure —
   S4 shares evidence at *readout* time (population vector, no new model), S3 abstracts it
   *declaratively* (AG pattern memory, needs a deriver). Front-gate favours S4 first (cheaper,
   rides existing infra); S3 earns its place only if the graded readout still can't fill a bin the
   learned table has no neighbouring evidence for. Which mechanism the substrate *should* own this
   with is itself the interesting question — run S4, then decide whether S3 adds anything.
6. **S4 readout geometry (NEW):** flat weighted average vs log/Weber-scaled (SC-map-like). Default
   flat; the log arm must clear the same "what does IPS buy that arithmetic doesn't" bar S1 held.
