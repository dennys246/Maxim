# Hybrid substrate-reflex runtime — the learned orient policy as a DN reflex under LLM cognition

**Status:** DRAFT plan, REVISED after three-lens review (2026-07-17 —
[reviews/hybrid_runtime_two_lens_review.md](reviews/hybrid_runtime_two_lens_review.md)).
Track 2 of [embodiment_runtime_wiring.md](embodiment_runtime_wiring.md) — the piece that actually
**runs the Exp 45 orient policy on a live robot**, which Track 1 (body wiring) does not.

> **Review correction (load-bearing):** the first draft's thesis — "just one DN `Behavior`, no new
> mechanism" — was **too strong**. All three lenses cross-confirmed that DN's action loop is built
> entirely around *visual detections*, so an audio reflex must **own five generic-DN additions**
> (BL-1..BL-5 below), not zero. The reflex-rides-DN *architecture* is right (no new `aut_mode`); the
> "no new mechanism" claim was wrong. The part the draft worried most about — the az_bin/
> `recommend_action` string keying — is the part that checks out cleanly.
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

## Front-gate: hybrid rides DN — but it is NOT free (revised after review)

The `aut_mode="hybrid"` design (interleave substrate + LLM selection in `run_agentic_loop`)
is still **wrong and unnecessary** — DN is the right home, and that part of the front-gate holds.
But the first draft's table asserted "already in DN ✅" for rows that are only *partially* there.
Corrected against the code (spot-verified):

| Hybrid needs | Reality (post-review) |
|---|---|
| A fast loop separate from LLM cognition | ✅ DN runs its own ~30 Hz thread |
| A NAc reference for the reflex to select on | ✅ `DefaultNetwork.__init__(nac=…)` (network.py:194) — but SHARED with the agent loop; see **BL-4** |
| `recommend_action(current_cluster_id: str)` — az_bin as a first-class key | ✅ nac.py:1653 — keyword-only, **string** key; the queen-mind policy is usable as-is |
| A reflex module contract | ✅ `Behavior` ABC → `ActionProposal` … | 
| Behaviors evaluated on an audio-only tick | ❌ **BL-2** — `_process_tick` early-returns `if not detections` (network.py:912); `BehaviorState` has no azimuth field; no seam pushes azimuth/`nac` into a `Behavior` |
| Motor arbitration between competing actions | ✅ `PriorityArbiter` (network.py:222) — *within* DN |
| Novelty / inhibition-of-return gating the reflex | ❌ **BL-5** — the salience/IOR gates `return True` on target-less proposals; the audio reflex has no `target`, so gating silently no-ops |
| Voluntary (LLM) action suppressing the reflex | ⚠️ **SF-1** — only `turn_around` self-inhibits DN today; voluntary `look_at` does not; `inhibited_behaviors` comes from *mode config*, not live per-tool suppression |
| Body-rotation dispatch that turns the mics | ❌ **BL-1** — no `turn_body` branch; `turn_around` ships **no head matrix** (head=None counter-rotation → mics don't turn) AND blocks the DN thread 5-8 s |
| A `src/` home for `az_bin`/sidecar/orient-action loading | ❌ **BL-3** — those live only in `scripts/`; `src/` can't import them |

**Corrected verdict: no new loop mode, but five owned generic-DN additions.** The reflex-under-
continuous-cognition *architecture* is right and matches the *visual* `OrientingResponse` coexistence;
this is still the same work as [orient_runtime_integration.md](orient_runtime_integration.md) Landing 2
(**P1 and Landing 2 merge**). What changed is the honest cost: the additions in the ❌/⚠️ rows are
this plan's real scope. Each is spelled out in **§Owned generic-DN additions** below.

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

> **BL-3 (blocking prerequisite):** `az_bin`, `decision_boundary`, `save/load_policy_meta`,
> and `load_orient_actions` currently live **only in `scripts/orient_backbone/`** (`grep` of
> `src/maxim/` returns zero matches), and `src/` cannot import from `scripts/`. The keying
> *logic* is correct; it's in the wrong place for the runtime. **Step 0 of this plan is to
> give these a `src/maxim/` home** (e.g. `src/maxim/embodiment/orient_policy.py`), leaving thin
> re-exports in `scripts/` so the training loop is unchanged. Without this the reflex has no way
> to bin a live azimuth or load a sidecar.

**The `"audio"` EC modality is explicitly NOT this plan.** It's the principled long-term
exteroceptive encoder (frozen-centroid, reserved in ec.py), but wiring it means a *new*
representation the current policy wasn't trained on — retrain-from-scratch, and Track D's
sign-collapse caution would have to be designed out first. Deferred; the az_bin path ships
the existing result. (A future plan may earn the `"audio"` modality if the reflex needs
finer-than-5-bin resolution or cross-modal binding.)

---

## Owned generic-DN additions (the real scope — BL-1..BL-5)

These are `[generic]` DN changes the reflex depends on. They are the honest cost the first
draft's front-gate hid. Each benefits *every* future non-visual reflex, not just audio-orient —
so they belong in the generic DN layer, per Track 1's "fix in the generic layer" discipline.

- **A0 (BL-3) — `src/` home for the policy helpers.** As above. Prerequisite for everything else.
- **A1 (BL-2) — a non-visual input channel into the tick.** Un-gate `_evaluate_behaviors` on
  audio-only ticks (today `_process_tick` returns at `if not detections`, network.py:912), and
  carry the latest gated azimuth into behaviors. Precedent: DN pushes external state into behaviors
  via setters (`TurnAround.set_head_yaw`). Mirror it — DN pulls `AzimuthDoASource.next_percept()`
  once per tick (the source's inbox is already non-blocking) and pushes az + a `nac` ref into the
  behavior via setters. The behavior must NOT call the blocking REST reader inside `evaluate`
  (the `<10 ms` contract). Generic shape: a `NonVisualCue` channel on `BehaviorState`, so future
  reflexes (thermal, tactile) reuse it.
- **A2 (BL-1) — a non-blocking body-rotation dispatch that ships a head matrix.** Add
  `action_type="turn_body"` to `_dispatch_action_to_motor` routing to a **new** controller method
  (e.g. `ReachyMiniController.orient_body(yaw_delta)`) that issues one `goto_target(body_yaw=…,
  head=<world-yaw = body-yaw matrix>)` and **returns immediately** (fire-and-forget on the fast
  loop; the DN thread must not sleep). Do NOT reuse `turn_around` — it is head=None (mics don't
  turn) AND blocks 5-8 s AND self-inhibits DN. The head matrix satisfies the CLAUDE.md head-frame
  invariant at the point of dispatch.
- **A3 (BL-4) — isolate the reflex's NAc learning.** The reflex uses a **dedicated `agent_id`**
  (as `live_3_learn.py` does) so `recommend_action`'s causal-link/reward-bias components aren't
  perturbed by the LLM agent's history, AND its `update_cluster_reward` writes must not leak into
  the LLM's Wire-A prompt: either a **separate NAc instance** for the DN reflex, or a reserved
  cluster/tool namespace `get_agent_tool_biases` (nac.py:2091) filters out. Pick separate-instance
  unless there's a reason the reflex needs the shared causal graph (there isn't, for orienting).
  This is the load-bearing correctness fix — same family as the Wire-1 / EC-drift silent-pollution
  bugs.
- **A4 (BL-5) — register the bearing so IOR actually gates.** A target-less proposal skips the
  salience/IOR gates (they `return True` when `proposal.target` is falsy). Give the audio proposal
  a **spatial target in a bearing/azimuth IOR map** (bio: the SC holds registered visual *and*
  auditory maps) so "don't chase every cough" is real. Until A4 lands, the plan must NOT describe
  novelty/IOR gating as free — it is absent.

## The reflex behavior — `AudioOrienting(Behavior)`

`[generic where possible; the Reachy specifics are the reader + calibration, already
isolated]`. Lives in `default_network/behaviors/audio_orienting.py`, a sibling to the
visual `OrientingResponse` (NOT a rewrite of it — umbrella Phase 3a fuses them later, with
evidence). Per DN tick (<10 ms, stateless-ish):

```
# az + nac are PUSHED in by DN each tick via setters (A1) — NOT pulled here
# (the REST reader is blocking; evaluate() must stay <10 ms).
evaluate(detections, state) -> ActionProposal | None
    az = self._latest_az                             # set by DN from AzimuthDoASource (A1)
    if az is None or |az| <= self._band: return None
    bin = az_bin(az, self._band, self._boundary)     # helpers imported from src/ (A0); boundary from sidecar
    rec = self._nac.recommend_action(               # dedicated reflex NAc / agent_id (A3); keyword-only
        agent_id=self._reflex_agent_id,
        available_tools=self._orient_affordances,    # e.g. ["turn_left","turn_right",...] — REQUIRED, or lookup misses (NH-1)
        current_cluster_id=bin,
    )
    if rec is None: return None
    action = rec["tool"]                             # keyed tool:{action}; deltas from loaded sidecar
    return ActionProposal(
        behavior_name="audio_orienting",
        action_type="turn_body",                     # new non-blocking dispatch (A2), ships head matrix
        target=self._bearing_target(az),             # spatial target so IOR/salience gate fires (A4)
        priority=f(|az|),                            # scaled by eccentricity; gating now real via target
        metadata={"yaw_delta": self._deltas[action] * sign, "bin": bin, "az_before": az},
    )
```

**Credit (genuinely new — front-gated).** Act now, credit later, spanning DN ticks. Do NOT sleep
in a behavior; mirror `ToolPainBridge._pending_tools`: stash `(bin, action, az_before, t_dispatch)`
on dispatch; on a later settled tick read `az_after`, `nac.update_cluster_reward(reflex_agent_id,
bin, "tool:"+action, |az_before| - |az_after|)`, drop it. Discard uncredited rather than fabricate.
*Why a new mechanism (Principle 3):* DN's existing delayed-outcome learners — `FocusLearner`
(movement-gain from focus feedback) and `PainCircuitBridge` (movement→pain) — carry gain/pain
signals, not a NAc cluster-reward keyed on an az_bin; neither fits, so the pending-map is warranted.

> **SF-3 caveat — this credit is error-correction, not reward.** `|az_before|−|az_after|` is
> saccadic/orienting *adaptation*, whose bio-correct organ is the Cerebellum
> (`embodiment/cerebellum.py`, a Rescorla-Wagner forward model), not the dopaminergic NAc. Routing
> it through NAc reward-bias is acceptable for the **discrete 5-bin** policy (a tabular bandit and a
> forward model pick the same bin), but NAc can only re-rank fixed `action_deltas` — it **cannot
> recalibrate gain** ("the same action now yields a different Δaz"), which is exactly the head=None
> regime. Continuous-gain adaptation is deferred to a Cerebellum plan; this plan ships the discrete
> result and names the boundary.

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
- **Between DN and the LLM**, inhibition is **one-way (SF-1, bio-confirmed):** voluntary gaze
  suppresses the reflex (cortex → colliculus; the antisaccade direction), **never** the reverse.
  When the LLM issues a voluntary head/motion tool, `audio_orienting` is inhibited for its
  duration; when cognition is idle, the reflex owns the head. The one legitimate bottom-up path is
  **salience escalation** through the thalamic gate (`gate.py` already escalates on salience/speech),
  NOT the reflex suppressing the LLM's motor output. Symmetric mutual inhibition would freeze or
  oscillate the head — do not wire it.
  **What does not exist yet (must be built, not deferred):** today only `turn_around` self-inhibits
  DN; voluntary `look_at` does not, and `inhibited_behaviors` is populated from *mode config*, not
  live per-tool LLM suppression. So the "voluntary tool inhibits the reflex" rule is **net-new
  wiring** — a hook that, when the executor dispatches a voluntary head tool, sets/clears the
  reflex's inhibition.
- **The `head=None` invariant applies here too (BL-1):** the new `turn_body` dispatch (A2) ships an
  explicit head matrix so the mics ride along. `_maxim.turn_around()` is a **confirmed** violation
  (movement.py:1220-1224 dispatches `body_yaw` with no `head=`) — which is exactly why the reflex
  gets its own non-blocking dispatch instead of reusing it.

**Efference copy (SF-2) — mandatory, not optional.** The reflex is silent as to the *decision*
(orienting is pre-attentive; you don't consciously choose to orient), but it MUST emit a
*self-motion signal* — "the body turned Δyaw; the source is now ~centered" — into cognition
(via Track 1's `body_state` post-orient azimuth). Without it, when the reflex turns the robot the
LLM sees the whole scene shift and misattributes its own turn as the world rotating (a corollary-
discharge failure). Silent as to intent; loud as to self-motion.

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

**Track 1 coupling, reframed (SF-5):** the reflex does NOT need Track 1's interoceptive
`evaluate_failures` tick or `executor.embodiment` — by its own P2 design it bypasses
`SensorEncoder`/interoception and reads DoA directly. It shares only Track 1's **`[declaration]`
seam** (robots.yaml body/reflex fields) and needs A0's `src/` policy-helper home. So Track 1 is a
*parallel* enabler for the declaration, not a hard interoception prerequisite.

0. **A0 (BL-3): `src/maxim/` home for `az_bin`/`decision_boundary`/sidecar-load/orient-action
   load**, with `scripts/` re-exports so training is unchanged. *Prerequisite for all below.*
1. **A1 (BL-2): non-visual cue channel into the DN tick** — un-gate `_evaluate_behaviors` on
   audio-only ticks + push az/`nac` into behaviors via setters. `[generic]`.
2. **A2 (BL-1): non-blocking `turn_body` dispatch** shipping a head matrix (`orient_body(yaw_delta)`).
   `[reachy-specific]` controller method + `[generic]` `_dispatch_action_to_motor` branch.
3. **A3 (BL-4): isolate the reflex NAc** (dedicated `agent_id` + separate instance or filtered
   namespace). `[generic]`.
4. **A4 (BL-5): bearing IOR map** so novelty/IOR gating actually fires. `[generic]`.
5. **`AudioOrienting` behavior + pending-credit bridge** — `[generic]`, offline-testable on the
   A0/A1 seams with a fake reader + standalone NAc: asserts it proposes the policy's action for a
   given az, credits on the delayed tick, is suppressed when inhibited, and gates on a repeated
   bearing (A4).
6. **Register in DN** gated on `has_audio` (evaluated **post-`connect()`**, NH-4) + the reflex
   declaration + a default-off `config.json` flag (concrete `*ConfigSection`/`resolve_setting` path,
   NH-3).
7. **DN↔LLM inhibition hook (SF-1)** — voluntary head tool inhibits the reflex; one-way only.
8. **Efference copy (SF-2)** — post-orient azimuth into `body_state`.
9. **Learning-on** — gated on online gain (the reflex's own `(yaw_delta, Δaz)` trials); default
   frozen-policy.

**Gates:** (a) the A2 dispatch ships a head matrix (BL-1) — no live dispatch through any head=None
path; (b) reflex NAc isolation (A3) verified before wiring to the shared bio-stack; (c) hardware
validation of the whole reflex loop once the motor is repaired — offline tests prove the logic, the
robot proves the loop.

## Open questions (post-review)

1. ~~DN↔LLM inhibition bidirectionality~~ **ANSWERED (SF-1):** one-way, voluntary-suppresses-reflex,
   plus a salience-escalation path. The wiring is net-new (see §Motor arbitration).
2. ~~`turn_body` vs reuse `turn_around`~~ **ANSWERED (BL-1):** do NOT reuse `turn_around` (head=None,
   blocking, self-inhibits). New non-blocking `orient_body` dispatch (A2).
3. **Startle vs orient (still open, SF-3):** biologically they **compose sequentially** (protective
   startle → investigative orient toward the same bearing), not pure winner-take-all. But
   `behaviors/startle.py` is **vision-only** — there is no audio startle to compose with today. If
   startle matters here it's net-new work (an audio startle + priority habituation, NH-5); out of
   scope for the first reflex, flagged for a follow-up.
4. ~~Does the reflex want the LLM to know it fired?~~ **ANSWERED (SF-2):** silent as to the
   *decision*, mandatory as to the *self-motion signal* (efference copy → `body_state`).
5. **NAc isolation shape (A3):** separate NAc instance vs reserved namespace filtered from
   `get_agent_tool_biases` — decide at implementation; separate-instance is the default.

## Pointers

- Behavior authoring guide (the `Behavior` contract this reflex implements): [../behaviors/README.md](../behaviors/README.md) + [../behaviors/audio_behaviors.md](../behaviors/audio_behaviors.md)

- Track 1 (prereq): [embodiment_runtime_wiring.md](embodiment_runtime_wiring.md)
- The reflex-as-DN-behavior Landing this merges with: [orient_runtime_integration.md](orient_runtime_integration.md) Landing 2
- The policy + az_bin/sidecar convention: [substrate_native_orienting.md](substrate_native_orienting.md), [porting_orient_loop.md](../embodiment/porting_orient_loop.md)
- The head-frame invariant that gates live dispatch: [CLAUDE.md](../../CLAUDE.md)
- The `"audio"` EC modality (deferred alternative): [perception_pipeline_placement.md](perception_pipeline_placement.md)
