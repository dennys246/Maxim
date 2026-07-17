# Hybrid substrate-reflex runtime — the learned orient policy as a DN reflex under LLM cognition

**Status:** DRAFT plan (2026-07-17). Track 2 of
[embodiment_runtime_wiring.md](embodiment_runtime_wiring.md) — the piece that actually
**runs the Exp 45 orient policy on a live robot**, which Track 1 (body wiring) does not.
Resolves the operator's decision (HYBRID) and the two problems the five-track audit named:
**P1** (the Reachy runtime is LLM-primary; the learned policy is substrate-primary) and
**P2** (a live-wired body encodes azimuth as EC-UUID interoception, incompatible with the
policy's az_bin-string state space). Grounded in a focused audit of the substrate-primary
path + DN; front-gate below.

**One-line thesis:** the hybrid is not a new agent-loop mode — the orienting reflex is a
**Default Network `Behavior`** that calls `NAc.recommend_action` with the policy's own
az_bin string, dispatches through DN's existing arbiter, and is suppressed by the
deliberative layer's existing inhibition. LLM cognition stays exactly as it is. The
"hybrid" is DN (fast reflex loop) coexisting with the agent loop (slow deliberation) —
which is what DN was built to be.

---

## Front-gate: hybrid rides DN; it does NOT need a new loop mode (audit-confirmed)

The tempting design — add an `aut_mode="hybrid"` to `run_agentic_loop` that interleaves
substrate and LLM action selection — is **wrong and unnecessary**. The audit found DN
already is the fast substrate-driven action layer:

| Hybrid needs | Already in DN? |
|---|---|
| A fast loop separate from LLM cognition | ✅ DN runs its own ~30 Hz thread |
| A NAc reference for the reflex to select on | ✅ `DefaultNetwork.__init__(nac=…)` (network.py:194) |
| Motor arbitration between competing actions | ✅ `PriorityArbiter` (network.py:222) |
| Voluntary (LLM) action suppressing the reflex | ✅ `InhibitionMixin` + `BehaviorState.inhibited_behaviors` |
| A reflex module contract | ✅ `Behavior` ABC → `ActionProposal` |
| Body-rotation dispatch | ✅ `action_type="turn_around"` → `_maxim.turn_around()` |
| `recommend_action(current_cluster_id: str)` — az_bin as a first-class key | ✅ nac.py:1659 — **string**, no bypass needed |

**Verdict: no new loop mode, no new mechanism.** The reflex is one `Behavior`; the hybrid
is DN-coexists-with-cognition, which already happens for the *visual* `OrientingResponse`.
This is the same conclusion the [orient_runtime_integration.md](orient_runtime_integration.md)
Landing 2 reached — **P1 and Landing 2 are the same work.** This plan merges them.

**Why NOT `aut_mode="hybrid"` in the agent loop** (naming the reason, per Principle 3):
substrate-primary mode (`propose_via_substrate`, agent_loop.py:789) *replaces* LLM action
selection for the whole agent — it's an either/or at the cognition layer. Hybrid is not
"sometimes substrate, sometimes LLM cognition"; it's "a reflex runs *underneath*
continuous LLM cognition." That is a two-layer architecture (DN under the loop), not a
mode switch. Forcing it into `aut_mode` would fight the arbiter/inhibition that already
model the two layers correctly.

---

## P2 — azimuth keys on the az_bin string (settled)

The five-track audit (Track D) proved the runtime's `SensorEncoder → EC interoception →
UUID` path is a *different, incompatible* state space from the Exp 45 policy's hand-binned
az_bin strings, and that `_normalize_value` folds away the left/right sign the policy
depends on. The resolution is direct and needs no new encoder:

**The reflex behavior computes `az_bin(az, band, boundary)` itself and passes that string
as `current_cluster_id`** — exactly as `live_3_learn.py` does. `recommend_action` already
accepts a string key (nac.py:1659), so the trained `cluster_reward_bias[(agent, "near_left",
tool)]` is looked up directly. The queen-mind policy is **usable as-is** — no retrain, no
SensorEncoder involvement for azimuth.

**The boundary + band travel with the policy** via the `.meta.json` sidecar (shipped in
#399): the reflex reads `bin_boundary`, `band`, `action_deltas` from the sidecar so it bins
the live azimuth the same way the policy learned it. This is the fourth application of
"state space travels with the policy" — and here it's load-bearing, because a wrong
boundary silently mis-bins (the demo bug, on the robot this time).

**The `"audio"` EC modality is explicitly NOT this plan.** It's the principled long-term
exteroceptive encoder (frozen-centroid, reserved in ec.py), but wiring it means a *new*
representation the current policy wasn't trained on — retrain-from-scratch, and Track D's
sign-collapse caution would have to be designed out first. Deferred; the az_bin path ships
the existing result. (A future plan may earn the `"audio"` modality if the reflex needs
finer-than-5-bin resolution or cross-modal binding.)

---

## The reflex behavior — `AudioOrienting(Behavior)`

`[generic where possible; the Reachy specifics are the reader + calibration, already
isolated]`. Lives in `default_network/behaviors/audio_orienting.py`, a sibling to the
visual `OrientingResponse` (NOT a rewrite of it — umbrella Phase 3a fuses them later, with
evidence). Per DN tick (<10 ms, stateless-ish):

```
evaluate(state) -> ActionProposal | None
    az = latest gated azimuth        # from the built AzimuthDoASource (Steps 1/2, #399)
    if az is None or |az| <= band: return None
    bin = az_bin(az, band, boundary)                 # boundary from the policy sidecar
    action = nac.recommend_action(agent_id, tools, current_cluster_id=bin, ...)
    return ActionProposal(
        behavior_name="audio_orienting",
        action_type="turn_body",                     # signed yaw_delta in metadata (turn_around precedent)
        priority=f(|az|, novelty, gate),             # novelty + inhibition-of-return from DN's SalienceMap
        metadata={"yaw_delta": action_deltas[action] * sign_mult, "bin": bin, "az_before": az},
    )
```

**Credit (the one genuinely-new piece)** — act now, credit later, spanning DN ticks. Do
NOT sleep in a behavior; mirror `ToolPainBridge._pending_tools`: stash `(bin, action,
az_before, t_dispatch)` on dispatch; on a later tick whose motion has settled, read
`az_after`, `nac.update_cluster_reward(bin, action, |az_before| - |az_after|)`, drop it.
Discard uncredited rather than fabricate (the never-fabricate-a-direction rule). This is
the reflex's learning loop, and it means the policy keeps improving live (gated — see
Track 3 of the runtime plan; learning-on needs online gain, which the reflex's own
`(yaw_delta, Δaz)` trials supply for free).

**Arbitration + suppression (both free):** priority scales with `|az|` and is gated by DN's
novelty / inhibition-of-return so the robot doesn't chase every cough; the deliberative
layer suppresses via `BehaviorState.inhibited_behaviors` when the LLM is doing a voluntary
`look_at` (the biology: voluntary gaze suppresses the orienting reflex).

---

## Motor arbitration: the two paths sharing the head

The reflex (DN) and LLM cognition (agent loop, via robot tools) can both command the head.
The audit found the seam is already there, but the coexistence needs one explicit rule:

- **Within DN**, `PriorityArbiter` already picks one winner across behaviors (audio-orient
  vs visual-orient vs startle vs return-to-center). No new code.
- **Between DN and the LLM**, `InhibitionMixin` lets the deliberative layer suppress DN
  output. The rule to make explicit: when the LLM issues a voluntary head/motion tool, the
  reflex is inhibited for its duration; when cognition is idle, the reflex owns the head.
  **Open question:** is that inhibition wired both ways today, or only DN-suppressed-by-
  deliberation? (Audit gap — verify before implementing.)
- **The `head=None` invariant applies here too**: whatever dispatches `turn_body` must ship
  an explicit head matrix so the mics ride along (CLAUDE.md Reachy head-frame invariant).
  `_maxim.turn_around()` predates that fix — **audit it before the reflex dispatches
  through it** (flagged in the runtime plan; still open).

---

## What's `[generic]` vs `[declaration]` (the abstraction discipline from Track 1)

- `[generic]` — `AudioOrienting` is bearing-agnostic in shape (any robot with an azimuth
  reader + orient affordances + a trained policy uses it); the `Behavior`/arbiter/inhibition
  contract is DN's, shared. The `az_bin`-string keying + sidecar convention is generic.
- `[declaration]` — a robot declares "I have an audio-orient reflex over sensor `azimuth`
  driving affordance `orient`, policy at `<path/bundle>`" via the minimal reflex
  declaration (Track 1's `[declaration]` seam). Reachy is the first filler.
- `[reachy-specific]` — the DoA reader (REST/onboard) and the calibration (boundary/gain/
  sign) — already isolated in the library + sidecar. A new robot swaps the reader and
  re-runs the calibration protocol (porting doc); the reflex code is untouched.

---

## Sequencing + gates

1. **Prereq: Track 1 (body wiring) lands** — the reflex needs `executor.embodiment` for the
   affordances + the tick, and the runtime needs the body declared. (Track 1 is safe/small.)
2. **`AudioOrienting` behavior** + the pending-credit bridge, `[generic]`, offline-testable
   with a fake reader + a fake NAc policy: assert it proposes the policy's action for a
   given az, credits on the delayed tick, and is suppressed when inhibited.
3. **Register it in DN** gated on `has_audio` + the reflex declaration + a default-off flag.
4. **Motor-arbitration rule** (DN↔LLM inhibition) + the `_maxim.turn_around()` head-frame
   audit — **gate: this must not fight for the head or counter-rotate the mics.**
5. **Learning-on** — gated on online gain (the reflex's own trials); default frozen-policy.

**Gates:** (a) the `turn_around` head-frame audit before any live dispatch; (b) hardware
validation of the whole reflex loop once the motor is repaired — the offline tests prove
the logic, the robot proves the loop (the session's standing rule).

## Open questions

1. **DN↔LLM inhibition bidirectionality** — is voluntary-suppresses-reflex wired, or only
   the reverse? Audit before implementing (§Motor arbitration).
2. **`turn_body` vs reuse `turn_around`** — `turn_around` means "rotate to see behind you"
   (degrees); ours is a signed servo delta. Sibling action_type, but the `_maxim.turn_around()`
   motor seam may be reusable — pending the head-frame audit.
3. **Startle vs orient** — a loud sound should startle *and* orient; do they compose in the
   arbiter or compete? (`behaviors/startle.py` exists.)
4. **Does the reflex want the LLM to *know* it fired?** ("I turned toward a voice on my
   left") — a percept/annotation back to cognition, or silent reflex? Bio: you're aware you
   oriented. Small, but a design choice.

## Pointers

- Track 1 (prereq): [embodiment_runtime_wiring.md](embodiment_runtime_wiring.md)
- The reflex-as-DN-behavior Landing this merges with: [orient_runtime_integration.md](orient_runtime_integration.md) Landing 2
- The policy + az_bin/sidecar convention: [substrate_native_orienting.md](substrate_native_orienting.md), [porting_orient_loop.md](../embodiment/porting_orient_loop.md)
- The head-frame invariant that gates live dispatch: [CLAUDE.md](../../CLAUDE.md)
- The `"audio"` EC modality (deferred alternative): [perception_pipeline_placement.md](perception_pipeline_placement.md)
