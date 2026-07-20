# Productive orienting affordance — make attending-to-sound pay off

**Status:** Design draft (2026-07-19). Follows the audio/DoA recognition work in PR #402
(`thalamus_relay_design_pass.md` stage 4). Recognition is done — a DoA percept reaches the agent's
prompt. This draft addresses the next gap the first live run exposed: **the agent hears the sound and
*wants* to attend to it, but there is no affordance that makes attending a productive, rewarded
action** — so the substrate learns to stop.

Still `[engineering]`. The earning experiment (does attending change behavior, does scaling it matter)
is what graduates the channel; this is the mechanism that lets attending *complete*.

---

## The observation (Qwen-32B run, big-mac-mini, 2026-07-19)

With the audio channel on and salience scaled up, the agent — hearing the DoA percepts — chose the
`listen` affordance and reasoned *"I should listen for any sounds that might draw my attention while
sitting still."* Exactly the right instinct. But the run recorded:

```
NAc: tool:base_humanoid_listen → plan:"I should listen…":failure  (valence=negative)
```

So the agent's attempt to attend to sound booked as a **negative** outcome — teaching it, via NAc, to
*avoid* listening. That is backwards for an orienting behavior.

## Root cause (traced)

The `listen` **tool succeeds mechanically** — `SpecModulator.execute("listen", {})` returns
`ModulatorResult(success=True)` (valid affordance, no preconditions). The negative valence comes from
the **plan-outcome** layer: `bridges/planning_bridge.py:384` writes
`tool:<name> → plan:<goal>:failure` with `success = (tool_success AND not embodiment_failed)`, and the
agent-loop `success` flag came through false for that action.

`base_humanoid`'s `listen` is a **pure passive affordance** — `params: {}`, **no `self_effect`, no
output that advances the sim's goal-scoring**. Two candidate reasons it books non-success at the
agent-loop layer:

1. **Passive-non-advancing** — it executes but produces no state change / goal progress, so it is
   scored as a non-success. (The primary hypothesis.)
2. **Not active at dispatch** — the scene-scoped tool activation / active-tool cap didn't have `listen`
   live, so the executor treated it as a failed/invalid call (consistent with the orchestrator's
   "your previous tool call may have failed or used an invalid tool name" nudge).

Disambiguating (1) vs (2) needs the session's `actions.jsonl` (on big-mac-mini, not on the shared
mount). **The sensor-route fix below addresses (1) regardless**; if (2) is also in play it is a
separate tool-activation fix.

---

## The seam — and why it's the *same* seam as live DoA on the robot

Investigation of the write/read paths:

- **Sensor write path exists:** `embodiment/tool_bridge.py:40` (`_apply_effect`-style helper) writes
  scalar sensor values into `body.vital_metrics[...]` (with range clamping + `sim_sensor` logging).
  A world-set value is a direct assignment, not a delta.
- **Sense read-back exists:** `ModulatorAffordanceTool.execute` already **reads back entity sensor
  state after the action** into its output (`tool_bridge.py:~406`); `EntitySenseTool.execute` returns
  `read_all_sensors()`. So an affordance on a body part that owns a sensor surfaces that sensor's value
  in its output *for free*.
- **The reachy body already has the sensor:** `reachy_mini.yaml` declares an `azimuth` sensor
  (`[-1,1]`, world-set, `drift_rate: 0`) — the head-relative sound direction, exactly the thing
  `listen` should report.

So the clean, bio-honest design is:

> **The audio percept world-sets a body `azimuth` sensor; `listen` (a sense affordance on the head)
> reads it back.** Attending to sound then returns the current direction as productive output → the
> action advances "report what draws your attention" → it books as a *success* → NAc learns attending
> pays off.

**Key architectural point:** writing the DoA azimuth into the body's `azimuth` sensor is *exactly what
the real robot does* — Track 2 Layer 2 is "feed live DoA into the azimuth sensor." So building this for
the sim (synthetic reader → sensor) is the **offline dry-run of the hardware path**, not a sim-only
hack. Sim and hardware converge on one seam. This also means the write **simultaneously activates the
Decision-4 drive-pain route** (the centeredness drive reads the same `azimuth` sensor) — so one sensor
write unifies *attention* (listen reads it) and *motivation* (drive-pain), on the same value. Elegant,
and it's why the sensor route beats a percept-only back-channel.

**Anti-back-channel note:** this deliberately does NOT use a `state.data` stash or a module global for
"last heard azimuth" — it uses the body sensor, the declared surface the sense system already reads.
Per the codebase's per-agent-state and no-back-channel rules, the sensor is the right carrier.

---

## The body-choice decision (needs the operator)

The design needs the AUT body to *have* an `azimuth` sensor. Options:

- **(A, recommended) Run the experiment on `bodies/reachy_mini`.** It already declares the `azimuth`
  sensor + the centeredness drive, so the sensor route + the drive-pain route both light up with no new
  sensor invention, and it's the body the hardware path targets. Cost: the generative sim must accept a
  reachy body (it took `base_humanoid` by default in the runs so far).
- **(B) Add an `azimuth` head sensor to `base_humanoid`.** Keeps the current default body but *invents*
  a sensor on a generic humanoid that only exists to carry sound direction — a small YAML add, but it
  duplicates the reachy declaration and drifts the two bodies.

**Recommendation: (A).** It's bio-honest, rides the reachy declaration, exercises the Decision-4 drive,
and keeps sim and hardware on one body model. (B) is the fallback if a reachy-bodied generative sim
proves awkward.

---

## Minimal first implementation (once the body is chosen)

1. **World-set the azimuth sensor from the percept.** In `agent_loop.py` §1.16 (which already has
   `executor` in scope and already extracts `metadata["azimuth"]`), when an audio percept arrives and
   the AUT body owns an `azimuth` sensor, assign it (world-set, not delta) via the sensor-write helper.
   Gate on the sensor existing (fail-soft on bodies without it). This is the sim mirror of Layer 2.
2. **Verify `listen` reads it back.** Confirm the head's `listen`/sense affordance surfaces `azimuth`
   in its output via the existing read-back; if `listen` is scoped to a body part that doesn't own the
   sensor, either move the sensor to that part or point the affordance's read-back at it.
3. **Confirm the plan-outcome flips positive.** With `listen` returning a real reading, the
   agent-loop `success` should be true → `planning_bridge` books `plan:…:success` (positive valence) →
   NAc learns attending pays off. Add a substrate test asserting the positive link.
4. **(If root cause (2) applies)** ensure `listen` is in the active scene-tool set for the AUT.

**Deferred (Track 2, motor-gated):** the *directional turn* affordance — "turn head toward the
azimuth" — which nulls the centeredness drive and is the actual orienting reflex. This draft only makes
*attending* (sensing the direction) productive; *acting on* the direction (turning) is the reflex.

## Interactions / constraints

- **Decision-4 preconditions still apply** to the substrate-primary route: `_normalize_value`'s
  `-1.0`≡`0.0` aliasing must be fixed before the azimuth EC encoding is meaningful, and azimuth must be
  de-bundled from the interoception sweep. The *llm-primary* listen-read-back does not depend on those
  (it reads the raw sensor scalar), but the drive-pain route does.
- **World-set discipline:** `drift_rate: 0` on the azimuth sensor is load-bearing — the sensor must not
  auto-recenter between DoA writes (the `tick_vital_drift` fabrication hazard).
- **Front/back ambiguity** (linear-array DoA) is unchanged — `listen` reports left/right cleanly, not
  front/back.

## Open questions for the operator

1. Body choice — **(A) reachy_mini** or (B) add azimuth to base_humanoid? (Recommend A.)
2. Should this land in PR #402 (extends the audio branch) or a fresh PR stacked on it? (Given #402 is
   already large and under review, lean fresh PR.)
3. Do we want the *turn* affordance (Track 2) scoped now as a follow-up, or held until the motor repair?

## Dimensionality: 1-D azimuth today, and the clean path to 2-D (elevation / altitude)

**State (Phase 1+2):** orientation is deliberately **1-D — horizontal azimuth only** — because that is
what the hardware gives and what the experiment needs. Two hardware ceilings are baked in, not chosen:

- **Azimuth-only** — reachy's XVF3800 mic array yields a horizontal bearing, no elevation. A robot with
  a spherical / 3-D array could resolve elevation; reachy cannot.
- **Front/back ambiguous** — a linear array reads a sound *behind* the same as *in front* (both ≈
  centered), so the azimuth is really a half-plane. A circular/3-D array resolves this too.

Magnitude is 1-D in both senses: *off-center magnitude* = `|azimuth|` (the centeredness drive's
deviation from set-point 0); *turn magnitude* = discrete step sizes (base_humanoid a fixed 0.3;
reachy's calibrated 0.17 / 0.50 set with learned step-selection — the Exp 45 orient-magnitude line).
The orient loop turns only on azimuth (`head_yaw`), even though reachy physically *has* `head_pitch` /
`body_yaw` (its motor could orient in elevation — nothing drives it from sound because sound gives no
elevation).

**Why we are NOT generalizing to N-D now:** we have exactly one axis. A multi-axis "orienting axes"
framework built before a second axis exists is the *abstract-at-N=1* mistake this codebase has scars
from — and it would force guessing the 2-D magnitude model (below) with no body to validate against.
The 1-D design is correct for current hardware and is **not painted into a corner**.

**The extension is clean and mechanical when a 2-axis body arrives** (elevation-capable localization, or
a vision-driven pitch-orient use case — reachy's `head_pitch` supports the latter *today* if ever
wired). It requires, and only requires:

1. The DoA reader / percept producing **(azimuth, elevation)** — the percept `metadata` dict is already
   extensible, so `metadata["elevation"]` needs no schema change; only `make_audio_percept` grows an
   optional `elevation=` param.
2. **Axis-parameterizing the ~6 azimuth-named surfaces** — `doa_to_azimuth`, `world_set_azimuth`,
   `reflex_oriented_azimuth`, `OrientingProfile.max_orient_azimuth`, the `metadata["azimuth"]` reads in
   agent_loop §1.16, and the turn `self_effect` — e.g. `world_set_axis(emb, "elevation", v)` alongside
   the azimuth one. `turn_up`/`turn_down` affordances (self_effect on the elevation sensor) join
   `turn_left`/`turn_right`, capability-declared per body.
3. **One real design decision that does not exist yet** — is 2-D "off-center magnitude" *two
   independent drives* (azimuth centeredness + elevation centeredness, orient each axis separately) or
   *one combined angular distance* (`√(az² + el²)`, orient toward the point)? Bio-honestly, orienting is
   toward a *point*, so a combined-magnitude drive is more faithful; independent axes are simpler and
   more learnable. **Make this choice when a 2-axis body forces it, not by guessing now.**

Everything else — capability-driven declaration, the per-entity `OrientingProfile`, the three-tier gate,
the physical-reach clamp — is axis-agnostic and carries over unchanged. So: build 2-D the day a body
actually has the second axis; until then, 1-D is the honest and correct shape.

## Related

- [thalamus_relay_design_pass.md](thalamus_relay_design_pass.md) — the audio recognition this builds on.
- [thalamus_hypothalamus_framing.md](thalamus_hypothalamus_framing.md) — dual-organ frame (attention vs drive).
- `hybrid_substrate_reflex_runtime.md` / Track 2 — the directional-turn reflex this defers to.
