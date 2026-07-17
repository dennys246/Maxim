# Orienting in the runtime — sound as a percept, and a learned reflex, on every live Reachy

**Status:** DRAFT plan (2026-07-16). Turns [Layer 1](substrate_native_orienting.md) —
complete on hardware ([Exp 45](../experiments/45_reachy_orient_live.md) direction /
[45c](../experiments/45c_flip_bins.md) magnitude 1.00) — from a bring-up script into
runtime behavior every embodied Reachy gets. Target: **1.1**, alongside the
`--embodiment` hardware work.

**One-line thesis:** the orient loop should not become a runtime *feature*; it should
become **one more DN behavior** in an arbiter that already exists — and the percept it
consumes is worth shipping on its own, before any motor behavior at all.

---

## Front-gate: audit first (and it moved the plan a long way)

Per CLAUDE.md Principle 3 + the audit-before-building rule. **Almost everything this
needs already exists**, which is why this plan is small:

| Need | Already in `src/maxim`? |
|---|---|
| Fast reactive tick, decoupled from LLM cognition | ✅ `DefaultNetwork`, threaded, `update_hz=30` |
| A reflex module contract | ✅ `default_network/behaviors/base.py::Behavior` (evaluate → `ActionProposal`, <10 ms) |
| **Motor arbitration between competing reflexes** | ✅ `default_network/arbiter.py::PriorityArbiter` — "publishes winning actions to motor queue" |
| **Voluntary action suppressing the reflex** | ✅ `default_network/inhibition.py::InhibitionMixin` + `BehaviorState.inhibited_behaviors` / `priority_modifiers` |
| "Don't chase the same sound twice" | ✅ `SalienceMap` + `ThreadSafeNoveltyTracker` + inhibition-of-return |
| Percept intake | ✅ `DefaultNetwork.on_percept()` (stashes `_latest_percept`, extracts detections) |
| Body rotation as a reflex action | ✅ `action_type="turn_around"` → `self._maxim.turn_around(angle, duration)` (network.py:1275) |
| An orienting behavior | ✅ `behaviors/orienting.py::OrientingResponse` — **visual-only, hardcoded heuristic** |
| DoA → azimuth percept source | ✅ `embodiment/audio_localization.py::AzimuthDoASource` — **built, never wired** |
| The learned policy + its state space | ✅ Exp 45c NAc + `.meta.json` sidecar |

**Verdict: rides existing infrastructure.** The genuinely new surface is small and
named below. **We are not building a reflex layer; we are adding a behavior to one.**

**The nice consequence:** `OrientingResponse` currently hardcodes its policy (novelty
threshold, movement weight, priority scaling). An audio sibling driven by
`NAc.recommend_action` makes DN's reflex layer **substrate-driven rather than
hand-tuned** — the project's thesis applied to its own reactive layer.

---

## Landing 1 — the percept (do this first; it is separable and independently valuable)

Wire `AzimuthDoASource` into the Reachy runtime's percept path so `DefaultNetwork.on_percept`
and the agent loop both see it. **No motor behavior.**

What it buys with zero arbitration risk:
- The substrate encodes azimuth (`"audio"` modality, frozen-centroid) → sound direction
  becomes part of what the agent *is*, not just what it could do.
- The centeredness drive fires → off-centre sound becomes an interoceptive state.
- The prompt can say *"a voice, to your left."*
- **It is the Layer-2 prerequisite**: spatial co-activation of sound + sight + word at a
  shared pose needs the percept, not the reflex. The Roy-4-style co-activation
  measurement ([`MAXIM_EC_TRACE_ACTIVATIONS=1`](substrate_native_orienting.md) Phase 3b)
  becomes runnable the day this lands.

**New surface:** a non-blocking reader (the REST DoA path for off-robot; local
`get_DoA()` onboard), and the `PerceptSource`-conformant wiring at the runtime's percept
seam. Both already have shapes (`AzimuthDoASource`, `make_reachy_doa_reader`).

**Gate:** none. Ship it.

---

## Landing 2 — the reflex (one real problem: the credit clock)

Add `behaviors/audio_orienting.py::AudioOrienting(Behavior)`:

```
evaluate(state) -> ActionProposal | None      # <10 ms, per DN tick
    az = latest audio percept's azimuth       # None/stale -> propose nothing
    if |az| <= band: return None              # centred: nothing to do
    state_bin = az_bin(az, band, boundary)    # boundary from the POLICY's sidecar
    action   = nac.recommend_action(bin)      # the learned policy, not a heuristic
    return ActionProposal(
        behavior_name="audio_orienting",
        action_type="turn_body",              # NEW: signed yaw delta in metadata
        target=None,                          # NOT pixels — see below
        priority=f(|az|, novelty, gate),
        metadata={"yaw_delta": deltas[action], "action": action, "az_before": az},
    )
```

**Design notes, each grounded in what the audit found:**

- **`ActionProposal.target` is (u,v) PIXELS** — a camera-frame concept. An azimuth action
  has no pixels. Precedent exists: `turn_around` carries `metadata["turn_angle"]` and
  ignores `target`. So a `turn_body` action_type with a signed `yaw_delta` in metadata
  follows the existing pattern rather than bending the pixel one. It dispatches through
  the same `_maxim` motor seam.
- **Priority is where the reflex earns arbitration.** It must scale with |az| *and* be
  gated by novelty — otherwise the robot chases every cough forever. DN already has
  novelty + inhibition-of-return; use them rather than adding a refractory.
- **Suppression is free.** `BehaviorState.inhibited_behaviors` already lets the
  deliberative layer say "I'm doing a `look_at`, sit down." That is the biological story
  (voluntary gaze suppresses the orienting reflex) and it is already modeled.

### The one genuine impedance mismatch: act now, credit later

`Behavior.evaluate()` is **<10 ms, stateless-ish, 30 Hz**. Proposing fits perfectly.
**Credit does not**: our trial is act → wait ~1.1 s for the motion to settle → re-read →
`potential_diff`. That spans many ticks.

**Do not** put a sleep in a behavior. **Do not** invent a mechanism. The codebase already
solved exactly this shape — `ToolPainBridge` keeps a `_pending_tools` map, records the
action at dispatch, and credits it on a **later** call by direct key lookup (never
context similarity; see the CLAUDE.md invariant). Mirror it:

```
propose  -> stash {proposal_id: (state_bin, action, az_before, t_dispatch)}
later tick -> for each pending whose motion has settled (t > t_dispatch + settle):
                az_after = latest azimuth percept
                nac.update_cluster_reward(bin, action, |az_before| - |az_after|)
                drop it
```

This lives in the behavior (or a small `OrientCreditBridge` beside it, if a second
consumer appears). **Discard, never guess**: a pending entry whose `az_after` read is
ungated or stale is dropped uncredited — the same "never fabricate a direction" rule the
percept source already enforces.

**Gate:** the extraction the porting doc scoped (`OrientRig` + `embodiment/orient_loop.py`)
— production code cannot live in `scripts/`. This *is* the second consumer that triggers
it.

---

## Landing 3 — learning ON (gated on gain calibration; do not ship first)

Frozen policy = safe. **Continuous learning in a stranger's room is not**, and 45c says
exactly why:

> the decision boundary is **derived from the gain** (`gain × (dᵢ + dᵢ₊₁) / 2`), and the
> gain depends on the room, the mount, the source distance and the robot.

Ship learning-on with a default gain and the boundary is wrong for that room → the reflex
confidently learns **the wrong magnitudes**. So:

**S2 (online gain estimation) is the gate for Landing 3** — and this is the argument that
survived its honest weakening (the "gain drifts between sessions" premise was the head
bug; the *portability* premise is real and now load-bearing). Options, cheapest first:
1. **Passive estimation from the reflex's own trials**: every credited trial is a
   `(yaw_delta, Δaz)` sample. An EMA over those *is* the gain — free, no ritual, and it
   is what the `--perturb` apparatus already does. Re-derive the boundary when the
   estimate moves materially, and **re-key or reset the policy when it does** (a changed
   boundary means a changed state space — see the sidecar rule).
2. Startup micro-sweep (a few walked poses) — a ~20 s boot ritual; more accurate,
   worse UX.
3. Cerebellar inverse model (the S3-flavoured version) — the interesting one, not the
   necessary one.

**Config:** per [feedback_prefer_config_over_new_env_vars], this is `config.json`
(`embodiment.orient.{enabled,learn,gain,bin_boundary}`), not new `MAXIM_*` vars.

---

## Load-bearing constraint: the policy's state space must reach the runtime

A `cluster_reward_bias` table is keyed on bin **names**; a bin name means nothing without
the boundary that produced it. Today this bit us three times in one day (head frame;
learner-vs-metric `az_bin`; the demo replaying a 45c policy at the legacy 0.5 boundary —
**silently**, which is what "why is it only doing small increments?" turned out to be).

The scripts fixed it with a `<nac>.meta.json` sidecar. **The runtime needs the same
fields somewhere durable** — the body YAML (`orient.bin_boundary`, `orient.gain`) or the
substrate-bundle manifest. Until then a runtime that loads a queen-mind policy will
mis-bin it exactly like the demo did. **This is a prerequisite for Landing 2, not a
nicety.**

---

## Sequencing + gates

| | Landing | Gate | Risk |
|---|---|---|---|
| 1 | Audio percept (`AzimuthDoASource` → runtime) | none | none — no motor path touched |
| 2 | `AudioOrienting` behavior + pending-credit | `OrientRig` extraction; state-space in YAML/manifest | motor arbitration (mitigated: arbiter + inhibition already exist) |
| 3 | Learning ON | **S2 online gain** | wrong-gain → wrong boundary → confidently wrong magnitudes |

**Do not blob these.** Landing 1 is independently valuable and unlocks Layer 2 whether or
not the reflex ever ships. Landing 3 without its gate is worse than not shipping.

## Deliberately NOT in this plan

- **Fusing audio + visual orienting into one drive** (umbrella Phase 3a). Right
  eventually; **do not refactor a working visual behavior to add audio**. Ship
  `AudioOrienting` as a sibling; fuse when there is evidence, not before.
- **Cross-modal binding** (Phase 3b) — gated on the co-activation measurement, which
  Landing 1 makes possible. Measure before building.
- **Vision on the same backbone** (Phase 2) — needs the P1 vision-encoder check first.

## Open questions

1. **Sibling vs extend:** `AudioOrienting` alongside `OrientingResponse`, or one
   bearing-agnostic behavior? (Recommend sibling; revisit at 3a.)
2. **`turn_body` action_type vs reusing `turn_around`:** `turn_around` means "rotate to
   see behind you" and takes degrees; ours is a signed servo delta. Probably a sibling
   action_type, but the `_maxim.turn_around()` motor seam may be reusable as-is.
3. **Does the reflex need the head to ride along?** Our loop commands `body_yaw` with an
   explicit head matrix (the CLAUDE.md invariant). `_maxim.turn_around()` predates that —
   **audit whether it counter-rotates the head** before trusting it. If it does, every
   DN body rotation has been moving the camera/mics less than it thinks.
4. **Startle vs orient:** `behaviors/startle.py` exists. A loud sound should probably
   startle *and* orient — do they compete in the arbiter, or compose?
5. Where exactly does the Reachy runtime's percept loop live for Landing 1's wiring —
   `embodied_runtime/agentic_runtime.py` shows no `PerceptSource` usage; DN has
   `on_percept()`. Confirm the seam before writing the adapter.

## Pointers

- Layer 1 result: [substrate_native_orienting.md](substrate_native_orienting.md) ·
  [Exp 45](../experiments/45_reachy_orient_live.md) / [45c](../experiments/45c_flip_bins.md)
- Porting contract + the derived-constants law:
  [porting_orient_loop.md](../embodiment/porting_orient_loop.md)
- Sensor truth: [audio_localization.md](../embodiment/reachy_mini/audio_localization.md)
- The head-frame invariant: [CLAUDE.md](../../CLAUDE.md)
- Gain calibration (S2) + bins line: [orient_magnitude_learning.md](orient_magnitude_learning.md)
