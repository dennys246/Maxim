# The 1.3 path — baseline, engines, fabric

**Status:** DRAFT 2026-09-01, written immediately after 1.1.2 published. **This is the
SEQUENCING plan for 1.3.** It does not restate the designs it orders; each stage names the
plan that owns it.

**Companion plans, unchanged and still authoritative in their own areas:**
[microduck_intent_layer.md](microduck_intent_layer.md) (the duck design exploration, rev 2
post-review) · [cross_modal_perception_fabric.md](cross_modal_perception_fabric.md) (the
fabric, Stage 0 gates Stage 1) · [three_factor_credit_assignment.md](three_factor_credit_assignment.md) ·
[sem_motor_binding.md](sem_motor_binding.md) ·
[deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) (deferred;
off the near-term path).

---

## Why this document exists

1.3 is the first release on the roadmap that **contains a pivotal may-fail experiment** — the
roadmap's own risk column says "High". Everything through 1.2 is plumbing and claims hygiene.
That changes what good planning looks like: the job is not to reduce the chance of failure, it
is to make a failure **informative** rather than ambiguous.

Three separate workstreams currently point at 1.3 — the duck, the robot-factory build-out, and
the perception fabric — and they were scoped independently. This document is the order they go
in, and the reason for that order.

## The organising fact

**Every EARNED behavioural result this project has is sound-orienting** — Exp 45, 52, 53b, 54.
One validated behaviour, one modality.

`microduck_intent_layer.md` §1.1 then establishes, from two independent enumerations of the
duck's sensing, that **no microphone appears in either**. The duck's supplied sensor suite is
proprioceptive and locomotor: 48 proprioception + command channels, pose, contacts, fall
state, projected gravity, per-joint commanded-vs-measured, servo temperature, battery.

So the duck is not a port of the orient result onto a second body. It is a **new behaviour
class on a new modality**, and it needs its own pre-registered experiment to earn anything.

That has been read as a risk to plan around. **It is better read as the thing to measure
first.** A baseline resolves it empirically in less time than it takes to argue about, and
`microduck_intent_layer.md` §8 already names the mic-array question as blocking design. Stage
A is that measurement.

---

## Stage A — the duck baseline (unblocks everything else)

**Purpose:** answer, with data rather than a spec sheet, what the duck can sense and what it
can be made to do. Stage A is not an attempt to earn a behavioural claim.

**It resolves, empirically:**

- the mic-array question — absent, presence-only, or directional. This is the single
  highest-value fact and it selects between two very different first experiments (§8 item 1).
- what the 48 proprioception + command channels actually contain, at what rate, with what
  noise — the input to any later dimensionality decision (§5.4 warns 48 dims would dilute).
- whether commanded-vs-measured per joint is a usable discrepancy signal, which is the
  duck-side analogue of the clamp/reach problem in D53.
- the real throughput ceiling for a headless episode loop (§6.3: real time is disqualifying).

**Ship gate:** a committed baseline artifact under `docs/experiments/data/`, produced by a
harness that runs `preflight_gated_record` (the gated-path rule — `lint_harness_provenance.py`
Family 3 now enforces it by WHERE records land). A baseline whose provenance cannot be
established is not a baseline.

**Explicitly not in Stage A:** any behavioural claim, any graduation row, any comparison to
the Reachy results. It is an instrument-characterisation pass.

## Stage B — the engine seam (the robot factory, driven by N=2)

**The factory already exists and is better than its reputation.** `src/maxim/hardware/`
carries `controller.py::RobotController` (ABC), `registry.py` with a `maxim.robots`
entry-point group, `capabilities.py` (`MotionCapability`, `StreamCapability`,
`RobotConnectionState`), and two backends — `reachy/` and `simulation/`. This is a
**build-out, not a build**, which is what the front-gate principle wants.

**The work is the leak at the edges.** Files outside `hardware/` that name the Reachy SDK,
measured 2026-09-01:

| file | refs | note |
|---|---|---|
| `embodied_runtime/selfy.py` | 13 | the big one; the roadmap already calls this "the gate on any second robot" |
| `api.py` | 6 | public surface naming one robot |
| `cli_utils.py`, `cli_parser.py` | 2 each | |
| `cli.py`, `tools/reachy.py` | 1 each | |

`hardware/reachy/*` naming Reachy is correct by construction and is out of scope.

**Why Stage B follows Stage A rather than preceding it: N=2 is when an abstraction stops
being speculative.** Designing the seam from the duck's spec sheet would be guessing at a
shape; designing it against a characterised second body is engineering. The 1.1.2 cycle's own
lesson applies — a proxy that agrees with the truth sometimes is worse than one that never
does.

**The modality question moves into the type system here.** "Does this body have directional
audio?" belongs in `StreamCapability` as a query, not in a roadmap as an assumption. Stage A
supplies the answer; Stage B encodes it. That is the mechanism by which the organising fact
above stops being a planning risk.

**Ship gate:** a second real engine registered through `maxim.robots`, with the runtime
reaching it through `RobotController` only — `grep -c "reachy_mini\|ReachyMini"` outside
`src/maxim/hardware/reachy/` trends to zero, with any remainder named and justified. Motion
safety is touched, so a two-lens round is required (the CLAUDE.md hardware-safety rule).

## Stage C — the perception fabric

Unchanged and still owned by [cross_modal_perception_fabric.md](cross_modal_perception_fabric.md).
Its own rule stands: **do not open Stage 1 until the Stage 0 preconditions pass.** Stage 0a is
COMPLETE (rev 4); 0b gained an expressibility criterion and 0c a decisiveness clause. 0c is
the experiment that earns the mechanism.

Stage A feeds Stage C directly: if the duck has no directional audio, the fabric's second
modality is proprioceptive/locomotor rather than auditory, and the binding convention must be
designed against that. Running Stage C first would design against an assumption.

## What is NOT in this path

- **JEPA.** Its blocker is the absence of paired cross-modal data, and a second body is that
  pairing stated physically — but the plan is DEFERRED and the fabric plan puts it "no longer
  on the near-term path". It stays a hypothesis to test, and it *raises* JEPA's bar: the pairs
  must be collected under this repo's evidence standard, not assumed. Revisit when Stage A has
  reported.
- **Re-deciding anything 1.1.3 owns.** The two-lens review withdrew both of rev 1's headline
  recommendations precisely because they converted a 1.3 exploration into decisions binding
  1.1.3. That correction holds. Gate 7 and the bias-key namespace belong to
  [1.1.3](roadmap_1_1_to_1_3.md); the duck is an INPUT to them, not a decision about them.
- **A schedule change.** The duck stays in 1.3 (`microduck_intent_layer.md` §7).

## Prerequisites worth doing regardless

Both are independent of whether the duck ever arrives.

1. **D51 — the LSH index is a linear scan over one bucket** (four byte-identical tables;
   `_hashers` built and never read). Fix or mark dormant. Perception work leans on similarity
   retrieval, so entering a may-fail experiment with a known-degenerate index manufactures an
   ambiguous null. Now carries `tests/unit/test_lsh_degeneracy_d51.py`, which fails loudly
   whichever way the decision goes.
2. **D53 — a clamped or unreached motion is credited to the substrate as a SUCCESS.**
   `tools/reachy.py::focus_on_sound` returns `success=True` even when `clamped` is true or
   `reached` is false. This is not 1.3 prep; it sits directly beneath the only EARNED claim
   the project has, and Stage A is a measurement that would run through it.

## Sequencing summary

```
1.1.3 merge correctness  →  1.1.4 Minecraft  →  1.2 Oasis  →  1.3:
        Stage A  duck baseline        (unblocks the mic question; gated artifact)
        Stage B  engine seam / factory (N=2 makes the abstraction honest)
        Stage C  perception fabric     (Stage 0 gates Stage 1; 0c is the pivotal experiment)
```

D51 and D53 can start now and are not on the critical path of anything above.
