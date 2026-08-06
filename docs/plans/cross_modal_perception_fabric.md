# Cross-Modal Perception Fabric (1.3 design direction)

**Status:** DESIGN DRAFT (2026-08-06). Zero code. Owner-initiated after Exp 49
("thoughts and language don't drive movement — they run in parallel"), grounded
in three things Exp 49 + the 2026-08-05 hardware session MEASURED.
**Target version:** 1.3 (design may start earlier; nothing here is on the 1.1
critical path).
**Owns (proposed):** the *artifact contract* + the extero/intero channel
extension + the sharing rule. Projection TRAINING internals stay owned by
[jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) — this
plan does not duplicate them; it supplies the artifact shape that plan leaves
open and re-evaluates its revival trigger.
**Companion plans:** [sem_motor_binding.md](sem_motor_binding.md) (Phase 3 is
Stage 1 here) · [jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md)
(Stage 2's training half) · [maxim_hivemind.md](maxim_hivemind.md) (Stage 4's
sharing surface) · [exteroception_interoception_seam.md](exteroception_interoception_seam.md)
(the shipped channel split this extends) · [perception_pipeline_placement.md](perception_pipeline_placement.md)
(where perception stages run).

---

## Motivation — three measured facts, not an aesthetic

1. **Language is the bottleneck in the motor loop.** Exp 49: the substrate
   centered a sound in **4.65 s** vs the LLM's **86.8 s** (~20×), and every
   LLM-arm failure was clock-bound, not capability-bound. The architecture
   already declares three orient vocabularies (deliberative / substrate /
   reflex); only the deliberative one is fully wired on live hardware. **The
   gap is wiring, not design.**
2. **A single azimuth reading cannot resolve the task.** The linear array's
   front/back fold creates a **180° false equilibrium**: a rear source reads as
   its front mirror, so honest credit *punishes* correct turns behind the fold
   (Exp 49: 77 fold-divergent credits; arm C trapped on every far bin while the
   non-credit-following LLM crossed it). No threshold tunes this away.
3. **The sensor cannot express graded magnitude.** The 2026-08-05 sweep
   measured a **compressed staircase** — proportional only within ~±30°
   (~0.19/rad vs 0.637 geometric), plateaus at ±12–14° equivalent, ~13°
   discrete sectors, L/R asymmetry, ~6–11° zero offset. Off-center there are
   ~3 distinguishable input levels. No function approximator recovers
   information the sensor never encoded.

Facts 2 and 3 have the **same escape**: use the *difference across a probe
turn*, not the single reading. Fact 2 additionally has a second escape:
**another modality** (vision resolves front from back). Both are what this plan
builds toward.

---

## Front-gate scope pressure (CLAUDE.md Principle 3)

**Question:** does this need to be its own mechanism, or can it ride on
existing infrastructure?

| Need | Existing infrastructure | Verdict |
|---|---|---|
| Per-modality compartmentalized encoding | `ModalityChannel` registry (`_SUBSTRATE_CHANNELS`, agent_loop.py) — a 3-tuple of `(tag, read_values, read_ranges)`; audio + interoception already live | **RIDES.** Vision is one more entry + a state reader. No new type. |
| Action selection from sensory state | `NAc.recommend_action` + `cluster_reward_bias` summed additively across the active `{modality: cluster}` set | **RIDES.** Explicitly binding-free by design ("late convergence at selection, no arbitration"). |
| Credit for a movement | `drive_comfort_progress` + the Phase-2 measured before/after pairs (`measured_drive_transitions`) | **RIDES.** Exp 49 validated the sign accuracy (1.00 / 0.969) with **zero confirmed honesty-gate leaks**, twice audited. |
| Learned movement prediction | `embodiment/cerebellum.py` forward models (`predict`, `observe_from_action`) — the designated, currently-underused slot | **RIDES.** If a learned predictor is wanted, this is its home. |
| Per-body calibration artifact | `spatial/bounds_learner.py` persists to `resolve_user_state("util/learned_bounds.json")` | **RIDES the pattern** (mutable-state layer, per-unit). New artifact, established shape. |
| Cross-modal alignment between different-dim encoders | *Nothing.* `SensorEncoder` is 384-dim, `LinguisticEncoder` 768-dim; cross-modal cosine is **mathematically undefined** between them, and `cross_modal_substrate_binding.md` was CANCELLED by Roy-4 for exactly this reason | **GENUINELY NEW** — and already designed as JEPA. This plan does not re-solve it. |
| Sharable perception artifacts | `hivemind/bundle.py` + `merge.py` (contract + migration seam + reserved `_*` namespace) | **RIDES**, with one new rule (below). |

**Conclusion:** exactly ONE genuinely-new mechanism (the projection), and it
already has a plan. Everything else is wiring existing surfaces. This plan's
real deliverable is **the artifact contract and the layer split** — the thing
that decides whether perception artifacts are shareable at all.

### Deliberately NOT doing

- **A conv net (or any gradient-trained policy) mapping sensors → movement.**
  Four independent disqualifiers: (a) wrong data shape — DoA is one scalar,
  nothing to convolve, and the az→turn map is ~10 parameters; (b) wrong
  sample regime — a hardware turn costs 1.5–3 s + settle, so a session yields
  low hundreds of labeled turns against the 10⁴–10⁶ a net wants, while
  EC/NAc prototype matching learns from 10–100; (c) it duplicates NAc as a
  second learning system claiming the same job; (d) fact 3 above — the sensor
  can't feed it anyway. **If a learned predictor is ever wanted, it goes in
  the Cerebellum slot, not a parallel net.**
- **A learned space→space transformer for cross-*species* transfer** (robot A's
  vision encoding → robot B's). It needs paired data *spanning both robots*,
  which is strictly harder to obtain than each robot's own alignment — the
  "align the aligners" regress. Deferred until two robots and real paired data
  exist; the contract stamp (below) means it slots in later as a declared
  adapter between two *named* contracts rather than an untyped blob conversion.

---

## Architecture: three layers, deliberately separable

### Layer 0 — substrate-primary sensorimotor (no new mechanism)

Orienting without the LLM in the action path, using the **act-and-compare**
primitive: probe (small turn) → re-listen → use the *sign of the change*.
This is the biologically canonical answer (animals resolve the cone of
confusion with head movements — active sensing, not better ears) and it
dissolves facts 2 and 3 together: the differential carries direction
information the fold destroys AND magnitude information the staircase
plateaus destroy.

The perception half already exists — Phase-2 measurement computes exactly the
before/after pairs act-and-compare consumes. What is missing is a **policy
layer that consumes differentials rather than single readings**.

### Layer 1 — semantic projection (shared, CONTRACT-keyed)

Encoder-space → shared latent, so two modalities can be compared at all. Its
identity is a **contract**, never a robot or a component name: which encoder
produced the input, which modality/axis, which convention and range. Any body
whose declaration satisfies the contract can load it — compatibility by
*construction*, not by conversion. Training internals: JEPA plan.

### Layer 2 — body calibration (per-unit, NEVER shared)

Gain, asymmetry, zero offset, saturation shape, motor scaling. **The 2026-08-05
staircase IS this artifact.** Per-robot, plausibly per-installation (the
room-vs-sensor question is open pending the second-position sweep). Sharing it
between two units of the same model could actively poison the receiver.

**Why the split is the load-bearing idea:** it makes the hivemind rule crisp —
**share Layer 1, never Layer 2** — instead of "share projections but be
careful."

---

## The artifact contract

Weights live in the **mutable state layer** (`resolve_user_state`, `.npy` +
`.json` sidecar per the no-pickle rule), never in component YAML. Components
are declarative operator config — the same layer as `mesh.yml` / `robots.yaml`
/ `config.json`, where **runtime writes are forbidden**. A component may
*declare which contract it satisfies*; it never carries weights or paths.

```json
{
  "_format_version": "1.0",
  "kind": "projection",
  "layer": "semantic",
  "contract": {
    "modality": "audio",
    "axis": "azimuth",
    "range": [-1.0, 1.0],
    "convention": "doa-xvf3800-v1",
    "source_encoder": "sensor-hash-384-v1",
    "target_encoder": "paraphrase-mpnet-768"
  },
  "weights_ref": "audio_azimuth_v1.npy"
}
```

Load rule: **contract match → load; mismatch → refuse LOUDLY.** Never silently
transform. This is the same shape as `check_format_version` and the
`hash_scheme: "stable-sha256-v1"` marker (#446), for the same reason.

**The precedent that matters most:** NAc biases key on EC node ids, so NAc and
EC persist as a **pair** — restoring one without the other leaves biases
dangling on nodes a fresh EC never re-allocates. Projection weights key on an
encoder space in exactly the same way. Same invariant class → reuse the same
solution: paired persistence, a compatibility stamp, refuse-don't-guess.

**Component inheritance comes free:** `ComponentRegistry` already supports
`extends` with deep-merge, so `bodies/reachy_mini_v2 extends bodies/reachy_mini`
inherits the sensor declarations → the contract → projection compatibility,
legibly and with zero coupling between the declarative and learned layers.

---

## Stages

| Stage | Content | Gate to proceed |
|---|---|---|
| **1** | **Act-and-compare orienting, substrate-primary** (Layer 0). Three-lens design review FIRST (bio / wiring / substrate-credit) — it may change what "orienting" means architecturally (a policy over *probes*, not over readings). | A pre-registered sim experiment on the staircase scenario shows differential policy beats single-reading policy on far bins. |
| **2** | **Hardware-faithful sim scenario**: measured-staircase option for `SimulatedDoAScenario` (piecewise transfer + ~13° quantization + asymmetry + zero offset, fit from sweep JSONL). Ideal-linear stays the default. | Sim-on-staircase reproduces observed hardware behavior. |
| **3** | **Vision as a third `ModalityChannel`** + the JEPA Stage-0 paired-data audit (~50 LOC). | Audit confirms sufficient training pairs; otherwise Layer 1 stays deferred and vision remains an independent channel. |
| **4** | **Artifact contract + sharing rule** wired into hivemind (Layer 1 shareable, Layer 2 refused at the bundle boundary). | Requires Stage 3's audit to have passed. |

**Stage ordering is a dependency chain, not a preference:** Stage 1 needs no
projection at all, and may reduce how much Layer 1 has to carry.

---

## Does JEPA's revival trigger fire?

Its registered trigger: *"a 1.1+ iteration surfaces a problem that is
structurally cross-modal AND unsolvable by threshold tuning, AND the Stage 0
paired-data audit confirms sufficient training pairs."*

**Half fired.** Exp 49's fold is structurally cross-modal (vision resolves
audio's front/back ambiguity — the canonical disambiguation) and provably not
threshold-tunable. The **second conjunct is untested** — hence Stage 3's audit
before any promotion. Recording the half-fire here so the trigger is evaluated
deliberately rather than drifted into.

**Thesis boundary to decide explicitly (do not drift):** the headline claim is
cross-session learning *without fine-tuning*. A **fixed pretrained encoder as
input** is already accepted practice (sentence-transformers for text). A
**projection trained on the robot's own paired experience** is a judgment call
— defensible as learned-from-experience, but it *is* gradient descent, and the
JEPA plan currently rejects imported alignment on thesis grounds. A
**gradient-trained policy** clearly breaks it and is excluded above. Write the
decision down before Stage 4.

---

## What would falsify what

- Stage 1 differential policy ≈ single-reading policy on far bins → act-and-
  compare is not the answer; the fold needs the second modality (raises Stage 3
  priority) or a reflex scaffold.
- Stage 2 sim-on-staircase does NOT reproduce hardware → the transfer model is
  wrong (more likely room-dependent than sensor-intrinsic; the second-position
  sweep is the discriminator).
- Stage 3 audit finds too few pairs → Layer 1 stays deferred; vision still ships
  as an independent channel (additive cluster bias needs no binding).
- Any Layer-2 artifact appearing in a hivemind bundle → the sharing rule leaked;
  treat as a poison-resistance defect, not a convenience.
