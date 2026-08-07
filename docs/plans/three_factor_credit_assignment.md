# Three-Factor Credit Assignment (architectural stance + calibration learning)

**Status:** DESIGN NOTE (2026-08-06). Zero code. Owner-initiated, arising from the
[cross_modal_perception_fabric.md](cross_modal_perception_fabric.md) design pass.
Mostly **names what already exists**; proposes exactly one small new learnable.
**Target version:** 1.3 (the stance applies immediately and retroactively; the
calibration learner is 1.3 work).
**Owns (proposed):** the learning-rule stance, the per-learnable teacher-signal
rule, and the consolidation-window update discipline.
**Companion plans:** [cross_modal_perception_fabric.md](cross_modal_perception_fabric.md)
(Layer-2 calibration is this note's motivating learnable) ·
[sem_motor_binding.md](sem_motor_binding.md) ·
[deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md)
(the thesis-boundary question this note sharpens).

---

## The stance

**Maxim does not backpropagate, and does not update pretrained weights.** It
does not thereby forswear gradients.

This replaces the looser framing "learning without fine-tuning," which is heard
as "no gradient descent" and is both weaker and less accurate.

The objection is specifically to **backpropagation**, for the reason it has
always been suspect biologically — the **weight transport problem** (Grossberg,
1987): backprop requires the feedback pathway to carry the *transpose of the
forward weights*, and no synapse has access to the downstream weights it would
need. Feedback alignment (Lillicrap et al., 2016) shows random fixed feedback
works surprisingly well, which softens the problem without making backprop
plausible as a mechanism.

The brain plainly does credit assignment. The best-supported account is the
**three-factor rule**:

```
Δw  =  pre  ×  post  ×  M
```

— Hebbian coincidence gated by a third, often neuromodulatory factor, with
**eligibility traces** bridging the delay between the coincidence and the
teaching signal (Frémaux & Gerstner, 2016). The trace is not a convenience:
Yagishita et al. (2014) measured a ~1–2 s window in which dopamine must arrive
for a dendritic spine's activity to potentiate. Credit assignment over delay is
solved by *decaying memory of what was just active*, not by propagating error
backwards.

**Why this is a stronger claim than "no gradient descent":** three-factor
learning is a live, well-supported position with a literature, not a limitation
being worked around. "We don't backprop" invites the interesting argument; "we
don't do gradients" invites a shrug.

---

## What is already three-factor in Maxim (audit, not aspiration)

| Factor | Shipped surface |
|---|---|
| pre × post (coincidence) | EC node co-activation; the Hebbian binding edges proposed in the perception-fabric plan |
| M (the third factor) | `NAc` reward / `cluster_reward_bias`; the salience-novelty plasticity gate (perception fabric §C) |
| Eligibility trace | `NAc.update_eligibility` (fast decay), `_temporal_anchors` + `TemporalSignature`, and `TemporalCreditDistributor.distribute`, which credits phase-matching nodes whose fast trace has already expired |

**All three factors and the trace already exist.** This note does not add them;
it names them, so future work stops re-deriving the rule and starts citing it.

Two consequences worth stating explicitly:

- The perception fabric's two-level attention convention **is** a three-factor
  rule — level 2 (association surprise) is the third factor gating Hebbian edge
  plasticity. It was designed on its own terms and landed on the canonical shape.
- `TemporalCreditDistributor` is the eligibility mechanism, and the substrate
  review flagged it as **under-used**. It is the natural home for any delayed
  teaching signal, and should be the first thing consulted before anyone builds
  a bespoke delay-bridging path.

---

## Depth is the axis that matters

Backprop is required only for **hidden layers**. For a single layer, the gradient
is local by construction — the delta rule is gradient descent with no backward
pass. So the question is never "gradients: yes or no," it is *how deep is the
learnable*:

| Learnable | Depth | Verdict |
|---|---|---|
| Hebbian binding edges | 1 — local | Already three-factor. No change. |
| **Layer-2 calibration** (gain, zero offset, asymmetry, saturation shape) | 1 — local | **The one genuinely new learnable.** See below. |
| Population tuning curves (preferred bearing, width) | 1 — local | Same treatment; likely a later increment |
| NAc reward biases | 1 — local | Already a delta-rule update (`current + alpha * reward`) |
| Any network with hidden layers | deep | **Excluded.** The stance holds. |

**Corollary for reviewers:** "is this gradient descent?" is the wrong question to
ask of a proposal here. Ask **"how many layers deep is the credit assignment?"**
One layer is local and admissible. More than one requires either backprop
(excluded) or an explicit bio-plausible alternative (target propagation,
predictive coding, forward-forward) argued on its own merits.

---

## The new learnable: calibration by local gradient

The perception fabric splits shareable **semantic** structure (Layer 1) from
per-body **calibration** (Layer 2 — gain, zero offset, asymmetry, saturation).
Layer 2 is a handful of continuous parameters mapping raw sensor readings to a
corrected axis. It is exactly the shape that local gradient learning fits, and it
has a direct, well-studied biological analog: **cerebellar gain adaptation** —
the vestibulo-ocular reflex recalibrates continuously via climbing-fiber error
signals onto parallel-fiber synapses. Local error, local update, no backward pass.

### Rule 1 — teacher signal per learnable: prediction error for calibration, reward for policy

Reward is sparse, delayed, and says *what to do*. Calibration needs a dense,
immediate signal saying *how the sensor maps to the world* — and that is
**prediction error from the forward model**, not reward.

Maxim already has the producer: `Cerebellum.observe_from_action(...)` fires on
every affordance and has been accumulating forward models. (Its `predict` read
path is **dormant** — one caller, constructed only by tests — so un-dormanting
carries a Principle-2 earning obligation, which the perception fabric's Stage 1
already owns.)

Concretely, for the orient axis: the body commands a turn of δ; the forward model
predicts the resulting azimuth change; the sensor reports the actual change; the
residual updates the gain. That is VOR adaptation with different hardware.

**Do not** wire NAc reward into calibration. It would make the sensor map a
function of whether the agent got what it wanted, which is a different and worse
thing.

### Rule 2 — update during consolidation, never online

If calibration shifts while EC is clustering on calibrated values, the clusters
move underneath the policy — the same failure class as centroid drift, and as
"changing the encoded state space orphans the trained policy."

So: **accumulate the residuals online, apply the update during the consolidation
phase.** This is the engineering answer (a stable input distribution within a
session) and the biological one (recalibration and consolidation are sleep-phase
operations). Maxim already runs consolidation at session end
(`MemoryHub.on_session_end` → hippocampal sleep/replay), so this rides an
existing phase rather than adding a scheduler.

### Rule 3 — a calibration change invalidates the frame of everything learned under it

**Open problem, flagged not solved.** Nodes encoded under calibration *v1* are in
a different frame than nodes encoded under *v2*. Three options:

- (a) accept the drift and let stale clusters decay — simple, lossy, and it
  quietly degrades cross-session claims;
- (b) re-encode stored nodes on recalibration — expensive, and it rewrites
  memory, which has its own honesty problem;
- (c) **stamp the calibration version on the node and refuse cross-calibration
  matching** — nodes from different frames simply do not compare.

(c) is preferred, and it *unifies with the perception fabric's artifact
contract*: the calibration version is the same compatibility stamp the Layer-1
sharing rule needs. One concept, two uses. It does mean a recalibration
effectively starts a new cluster generation, which is a real cost and an argument
for recalibrating **rarely and decisively** rather than continuously.

---

## Developmental timescale — a stated design stance

**Maxim is expected to learn at the speed of an infant: slowly.** Sample
efficiency is explicitly *not* an optimization target.

This is bio-faithful (reliable audio-visual binding and sound localization take
human infants months, not trials) and it neutralizes the standard objection to
reward-modulated learning without backprop — that global-signal gradient
estimation is high-variance and needs many samples. It does need many samples.
That is acceptable here in a way it would not be in a benchmark setting.

But the stance has costs, and they should be named rather than discovered:

- **Experiments get expensive in wall-clock.** A learning claim needs a horizon
  long enough for slow learning to show. Hardware sessions are the bottleneck.
- **Null results become harder to interpret.** "No learning at n=10 trials" is
  uninformative under this stance. **Pre-register the learning *curve*, not just
  an endpoint** — a trajectory with a slope is falsifiable; a single endpoint at
  a short horizon is not. This interacts directly with the pre-registration
  discipline and should be reflected in future experiment designs.
- **It is only viable because cross-session persistence works.** Slow learning
  accumulates only if state survives sessions — NAc/EC persistence (#446) is the
  load-bearing prerequisite. Any regression there silently invalidates the whole
  stance.
- **It does not fix credit assignment over long *delays*.** Rate and delay are
  different axes: eligibility traces decay, so a teaching signal arriving long
  after the coincidence teaches nothing regardless of how patient the system is.

---

## Front-gate scope pressure (CLAUDE.md Principle 3)

| Need | Existing infrastructure | Verdict |
|---|---|---|
| Coincidence detection | EC co-activation / binding edges | **RIDES** |
| Third factor (modulator) | NAc reward, salience/novelty gate | **RIDES** |
| Delay bridging | `NAc.update_eligibility` + `TemporalCreditDistributor` | **RIDES** (and is under-used) |
| Dense error signal | `Cerebellum` forward models (write path live, read path dormant) | **RIDES** + Principle-2 earning obligation |
| Update scheduling | Consolidation phase (`MemoryHub.on_session_end`) | **RIDES** |
| Calibration parameters + their local update | Nothing | **GENUINELY NEW** — but one layer, a handful of parameters, delta rule |

**Count: one small new learnable.** Everything else is naming and wiring what
ships today.

### Deliberately NOT doing

- **Backpropagation, anywhere.** Weight transport; see the stance.
- **Updating pretrained encoder weights.** Fixed encoders at the sensory boundary
  stay fixed — that is what makes them admissible.
- **Reward as the calibration teacher.** See Rule 1.
- **Online calibration updates.** See Rule 2.

---

## What would falsify what

- **Calibration residuals do not converge** (gain estimate wanders across
  consolidation cycles) → the forward model is not a usable teacher for this axis;
  suspect the model's coverage before suspecting the rule.
- **Calibration converges but behavior does not improve** → calibration was not
  the binding constraint; look at cluster resolution (perception fabric Stage 0b)
  instead.
- **Cluster generations proliferate** (frequent recalibration → many
  incompatible frames) → Rule 3(c) is too strict in practice, or recalibration is
  firing too often; consider a hysteresis band before a new generation is minted.
- **Learning shows no slope over a long horizon** → under the developmental
  stance this is the *real* negative result, and it is only interpretable if the
  curve was pre-registered.

---

## Open decisions

**A. Does the thesis statement change?** Recommended wording: *"cross-session
learning with no backpropagation and no pretrained-weight updates."* More
accurate, more defensible, and it names a position in the literature rather than
an absence. Needs an owner call, since it touches how the work is described
externally.

**B. Do population tuning curves become learnable, or stay hand-specified?**
Hand-specified is simpler and sufficient for Stage 0b; learnable is the natural
increment once calibration learning works. *Recommendation: hand-specify first.*

**C. Recalibration cadence.** Rule 3 makes recalibration a generation boundary,
which argues for rarely-and-decisively. What triggers it — a residual threshold,
a session count, an operator verb? Undecided.
