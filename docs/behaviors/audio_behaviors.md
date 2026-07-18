# Audio behaviors — the orient-to-sound path

**Status: not yet buildable as "just one Behavior."** Unlike a vision behavior (see
[vision_behaviors.md](vision_behaviors.md)), an audio-driven behavior needs **five generic-DN
additions** the visual path already has for free. This doc records exactly what's missing, why, and
the intended shape — so the audio orienting reflex (running the Exp 45 learned turn-toward-sound
policy) can be built without re-discovering the seams. It is the companion to the plan
[../plans/hybrid_substrate_reflex_runtime.md](../plans/hybrid_substrate_reflex_runtime.md) and the
review [../plans/reviews/hybrid_runtime_two_lens_review.md](../plans/reviews/hybrid_runtime_two_lens_review.md);
keep those three in sync as the work lands.

## Why audio is not symmetric with vision

The DN action loop was built entirely around **visual detections**. Three structural facts make an
audio reflex different from a vision behavior:

1. **The tick is gated on visual detections.** `_process_tick` returns at `if not detections: return`
   ([`network.py:912`](../../src/maxim/default_network/network.py)) *before* behaviors are evaluated.
   Orienting to a sound is precisely the *no-visual-detection* case (a voice off to the side, out of
   frame) — so an audio behavior would essentially never run.
2. **`BehaviorState` has no audio field** ([`base.py:39-46`](../../src/maxim/default_network/behaviors/base.py)),
   and `evaluate(detections, state)` is handed YOLO detections only. There is no channel carrying a
   direction-of-arrival (DoA) azimuth into a behavior.
3. **The motor + gate machinery is pixel-shaped.** `_dispatch_action_to_motor` has no body-turn branch
   that ships a head matrix; the salience/IOR gates work in `(u, v)` pixel space and no-op on
   target-less proposals.

None of these is a flaw in the audio design — they're the cost of being the first non-visual reflex,
and each addition below benefits *every* future non-visual behavior (thermal, tactile, …).

## The five additions (A0–A4)

These mirror the blocking findings BL-1..BL-5 from the review. Detail + code anchors live in the plan;
summary here so this doc stands alone:

| # | addition | why | layer |
|---|---|---|---|
| **A0** | a `src/maxim/` home for `az_bin` / `decision_boundary` / the `.meta.json` sidecar loader / orient-action loading | those helpers live only in `scripts/orient_backbone/`, which `src/` can't import; the reflex has no way to bin a live azimuth the way the policy learned it | `[generic]` |
| **A1** | a non-visual cue channel into the tick — un-gate behavior eval on audio-only ticks + push the gated azimuth (and a `NAc` ref) into behaviors via setters | the tick discards no-detection frames; there's no seam for azimuth to reach a behavior; the DoA reader is a **blocking** HTTP call so it can't be pulled inside `evaluate()` | `[generic]` |
| **A2** | a non-blocking `turn_body` motor branch that ships a head matrix | no `turn_body` branch exists; reusing `turn_around` counter-rotates the head (mics don't turn) **and** blocks the DN thread 5-8 s | `[generic]` dispatch branch + `[reachy-specific]` controller method |
| **A3** | isolate the reflex's NAc learning (dedicated `agent_id` + separate NAc instance or a namespace filtered from `get_agent_tool_biases`) | the reflex crediting at DN rate on the shared NAc would flood the LLM's Wire-A tool-value prompt, and the LLM's history would perturb the reflex's action scores | `[generic]` |
| **A4** | register the bearing into an azimuth IOR map so novelty/IOR gating fires | target-less proposals skip the salience/IOR gates — "don't chase every cough" is silently absent otherwise | `[generic]` |

## The intended behavior shape

Once A0–A4 exist, the behavior itself is small — a sibling to the visual `OrientingResponse`, living at
`default_network/behaviors/audio_orienting.py`:

```python
class AudioOrienting(Behavior):
    name = "audio_orienting"
    base_priority = 0.8

    # az + nac are PUSHED in by DN each tick (A1) — never pulled here (<10 ms, non-blocking)
    def evaluate(self, detections, state):
        az = self._latest_az                      # set by DN from the DoA source (A1)
        if az is None or abs(az) <= self._band:
            return None
        bin = az_bin(az, self._band, self._boundary)   # helpers from src/ (A0); boundary from sidecar
        rec = self._nac.recommend_action(         # dedicated reflex NAc / agent_id (A3); keyword-only
            agent_id=self._reflex_agent_id,
            available_tools=self._orient_affordances,   # e.g. ["turn_left","turn_right",...] — REQUIRED or lookup misses
            current_cluster_id=bin,
        )
        if rec is None:
            return None
        action = rec["tool"]
        return self._create_proposal(
            action_type="turn_body",              # non-blocking dispatch, ships a head matrix (A2)
            target=self._bearing_target(az),      # spatial target so IOR gates fire (A4)
            yaw_delta=self._deltas[action] * sign,
            bin=bin, az_before=az,
        )
```

### Learning (credit spans DN ticks)

Orienting *adaptation* — learning how far to turn for a given bearing — uses an act-now-credit-later
pending map (mirror `ToolPainBridge._pending_tools`): stash `(bin, action, az_before)` on dispatch;
on a later settled tick read `az_after` and call `update_cluster_reward(..., |az_before| − |az_after|)`;
discard uncredited rather than fabricate a direction. **Caveat (bio-fidelity):** this credit signal is
error-correction (cerebellar territory), not dopaminergic reward. It's fine for the **discrete** binned
policy, but NAc reward-bias can only re-rank fixed action deltas — it cannot recalibrate *gain* ("the
same action now yields a different Δaz"), which is exactly the head-frame failure regime. Continuous-gain
adaptation belongs in the Cerebellum ([`embodiment/cerebellum.py`](../../src/maxim/embodiment/cerebellum.py))
and is deferred.

### The percept source already exists

The off-robot DoA reader and the source builder are shipped:
[`build_reachy_audio_orienting_source(...)`](../../src/maxim/embodiment/audio_localization.py) and
[`make_reachy_rest_doa_reader(...)`](../../src/maxim/embodiment/audio_localization.py) (which routes
through `maxim.utils.http`, satisfying the single-HTTP-surface invariant, and accepts a `fetch` seam
for offline tests). What's missing is A1 — the wiring that pulls from that source once per tick and
pushes the azimuth into the behavior.

## Cross-modal note: the bearing is exteroceptive

Azimuth is an **allocentric world-bearing** (parietal/collicular), not interoceptive homeostatic state
(insula). The reflex deliberately keys on the hand-binned `az_bin` **string** via
`recommend_action(current_cluster_id=...)` rather than routing azimuth through the interoception
`SensorEncoder` (which would fold away the left/right sign). This is the bio-correct call and it means
the Exp 45 policy is usable **as-is** — no retrain. A future `"audio"` EC modality (a learned
exteroceptive encoder) is the principled long-term path but is out of scope here.

## Inhibition & efference copy

- **Inhibition is one-way:** a voluntary LLM-driven head move suppresses the reflex, never the reverse
  (the antisaccade direction). That per-tool suppression hook does **not** exist yet — today only
  `turn_around` self-inhibits DN — so it's net-new wiring.
- **Efference copy is mandatory:** the reflex should be silent as to its *decision* (orienting is
  pre-attentive) but must emit a *self-motion signal* ("the body turned Δyaw; the source is now
  centered") into cognition via `body_state`, or the LLM misreads its own reflexive turn as the world
  moving.

## Where this connects

- Plan (owns A0–A4 as steps): [../plans/hybrid_substrate_reflex_runtime.md](../plans/hybrid_substrate_reflex_runtime.md)
- Review (why each addition is needed): [../plans/reviews/hybrid_runtime_two_lens_review.md](../plans/reviews/hybrid_runtime_two_lens_review.md)
- Body wiring prerequisite (declaration seam): [../plans/embodiment_runtime_wiring.md](../plans/embodiment_runtime_wiring.md)
- The policy + az_bin/sidecar convention: [../embodiment/porting_orient_loop.md](../embodiment/porting_orient_loop.md)
- The head-frame invariant gating any body dispatch: [CLAUDE.md](../../CLAUDE.md)

> **Keep this doc live.** As A0–A4 land, replace "not yet buildable" / "does not exist yet" with the
> real APIs and file:line anchors, and move settled items from "intended shape" to "shipped." This is
> the durable record of *why* the audio path is shaped the way it is.
