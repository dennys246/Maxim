# Orient magnitude learning — and the parietal/cerebellar division of labor

**Status:** DRAFT pre-registration (2026-07-16). Follow-on to
[substrate_native_orienting.md](substrate_native_orienting.md) Layer 1 /
[Exp 45](../experiments/45_reachy_orient_live.md) (all arms EARNED). Scope: the
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
| **IPS** ([math/ips.py](../../src/maxim/math/ips.py)) | ✅ | `memory_hub` (constructs `IPS()`), `maxim_agent`, `statistician_agent`, `math_tool`, `concept_grounder` | **Stateless** Approximate Number System — Weber-Fechner log compression; `compare` / `categorize` / `detect_trend`. Cannot store; it is magnitude *intuition*, not magnitude *memory*. |
| **Angular Gyrus** ([math/angular_gyrus.py](../../src/maxim/math/angular_gyrus.py)) | ✅ | `build_bio_stack` (persisted `angular_gyrus.json`), `memory_hub`, `maxim_agent`, `create.py` | MemoryLayer **+** compute engine: math facts, methods, **patterns**; dual verbal/code. The one region here that could *hold* a relationship. |
| **IPS → AG escalation** ([agents/statistician_agent.py](../../src/maxim/agents/statistician_agent.py)) | ✅ | — | Already implemented: "dual number system (IPS fast/approximate + AG slow/precise)"; escalates to AG when IPS confidence is 0.3–0.65. **The abstraction pathway the magnitude idea wants already exists.** |
| **Cerebellum** ([embodiment/cerebellum.py](../../src/maxim/embodiment/cerebellum.py)) | ✅ | `build_bio_stack` | Learned **forward** models (command → predicted sensory consequence) via Rescorla-Wagner prediction error, keyed `(entity, modulator, affordance, param_bucket)`. |

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

### S0 — magnitude action set (Exp 45b). **Do this first; it is the actual fix.**

Add `turn_{left,right}_{small,big}` to
[bodies/reachy_mini.yaml](../../src/maxim/_data/components/bodies/reachy_mini.yaml)
(≈ ±0.12 / ±0.4 rad `self_effect`); scripts read each affordance's **actual magnitude**
instead of sign × fixed `--step` (~10 lines; `orient_demo.py` inherits it). Retrain
fresh (~60–80 trials, 16 state-action pairs).
- **Front-gate:** rides existing infra (YAML + NAc + `potential_diff`). No new mechanism.
- **Pre-registered metric:** *magnitude-appropriateness* — far bins prefer `big`, near
  bins prefer `small` — alongside direction correctness. Learned-vs-servo separation is
  unchanged (probe curve from empty + cross-session).
- **Note:** this trips Exp 45's own re-run rule ("orient-affordance YAML change") → new
  pre-registration, fresh NAc, new bundle version (queen-mind v0.2).

### S1 — Weber-scaled bins (IPS). **Nearly free; the honest answer to "finer bins".**

Carve the azimuth axis log-spaced (fine near center, coarse far out) instead of
uniformly — matching both the psychophysics and the task (precision matters near 0;
"it's way over there" suffices far out).
- **Front-gate (be honest):** log-spaced bins are ~3 lines of `math.log`. Routing
  through `IPS.categorize` is justified **only** if we want its Weber calibration +
  confidence values, and only once the orient loop has a bio-stack to reach it through
  (Gap 1). Otherwise this is bio-naming theater on a log function. **Decide explicitly.**
- **Pre-registered metric:** does Weber binning beat uniform binning at equal bin count
  (faster convergence, or finer terminal centering)? A/B on the same hardware protocol.

### S2 — cerebellar gain calibration (inverse model). **Real, and the right organ.**

The `0.58 az/rad` tracked gain is currently a hand-measured constant in the apparatus.
Biologically, adaptive sensorimotor gain is *the* cerebellar job (VOR gain adaptation is
the textbook case), and the module already learns from prediction error.
- **Front-gate:** the existing cerebellum learns **forward** models; orienting needs the
  **inverse** (error → command). That is a genuine extension, not a wiring change — name
  it as such.
- **Pre-registered claim:** the robot learns its own gain from experience (converges to
  ≈0.58 with no hand-measured constant), and **re-calibrates** when the gain changes
  (verifiable by changing `--step` or the sweep-measured mount/acoustics — the eared
  shell will change it for real).
- **Rigor caution:** with one parameter, "learned the gain" is a thin claim on its own;
  it earns its weight from the *re-calibration* arm, not the initial fit.

### S3 — AG abstraction of the learned table. **Most interesting, most speculative.**

**Not** "fit a polynomial to (azimuth → turn)" — that regresses the analytic servo from
sensor data with the substrate bypassed, and collapses the learned-vs-servo distinction
Exp 45 exists to defend (the optimal policy here is `turn = -az/0.58`, a straight line;
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

---

## Sequencing

1. **S0 now** — it fixes the observed gap, is substrate-native, costs ~25 min of robot
   time, and *strengthens* the Exp 45 claim ("learned which way **and how far**, from
   relief alone"). Gate the queen-mind v0.2 release on it.
2. **Gap 1 next (prerequisite for S1/S3)** — run the orient loop through
   `build_bio_stack` instead of a bare NAc. Converges with the 1.1
   `--embodiment` hardware-runtime work; do it there, not twice.
3. **S1** after Gap 1, with the front-gate decision made explicitly.
4. **S2** independently (needs no bio-stack — the cerebellum sits next to the
   embodiment layer); natural pairing with the eared-shell experiment, which changes the
   gain for real.
5. **S3** last, only if S0/S1 leave magnitude structure the tabular policy can't hold.

**Do not blob these.** Four steps, four separate pre-registrations, four separable
claims — the cradle-cascade lesson (don't stack unverifiable layers) applies exactly.

## Open questions

1. **S1 front-gate:** IPS routing vs 3 lines of log-spacing — what does IPS's Weber
   calibration buy that justifies the coupling?
2. **S3 deriver:** reshape the Statistician (cross-state pattern support) vs a new
   derivation step? Front-gate argument required either way.
3. **Does S0 make S1 moot?** If near/far × small/big already produces
   magnitude-appropriate behavior, finer/Weber bins may add nothing measurable. Check
   before building.
4. **Integration-test framing:** once Gap 1 closes, the orient task becomes the
   project's cleanest bio-region integration probe. Worth a standing "does the whole
   stack still carry a real sensorimotor policy" regression run?
