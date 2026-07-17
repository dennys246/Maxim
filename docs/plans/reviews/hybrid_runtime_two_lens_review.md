# Three-lens design review — embodiment_runtime_wiring.md (Track 1) + hybrid_substrate_reflex_runtime.md (Track 2)

**Date:** 2026-07-17. **Lenses:** Architecture, Executor, Bio-fidelity (three parallel reviewers,
per the pre-merge review-round discipline; bio-fidelity added because it repeatedly surfaces the
top load-bearing finding on embodiment/DN work). **Reviewed at:** commit on
`feat/audio-percept-step3-runtime` after the Track 2 draft.

**Headline:** Track 1 (body wiring) is sound and implementable as written — all three lenses agree.
**Track 2's central thesis — "the orient reflex is just one DN `Behavior`, no new mechanism" — does
not hold.** The DN action loop is built entirely around *visual detections*; an audio-driven reflex
that turns toward a sound *not in view* hits generic-DN seams the front-gate table asserted were free.
Track 2 must be revised to own those additions explicitly before implementation. The one part the plan
worried most about — the az_bin / `recommend_action(current_cluster_id=…)` string keying (P2) — is the
part that checks out cleanly.

Findings ranked; cross-confirmed findings (≥2 independent lenses) are trusted most and marked.

---

## BLOCKING

**BL-1 — Motor path is confirmed defective, and `turn_body` has no dispatch branch.** *[cross-confirmed:
Architecture B2 + Executor B2; spot-verified]* `_dispatch_action_to_motor` (network.py:1241-1281) handles
only `look_at`/`scan`/`track`/`turn_around`; the plan's proposed `action_type="turn_body"` would be a
silent no-op. Reusing `turn_around` is worse: it (a) dispatches `goto_target(body_yaw=…)` with **no `head=`
matrix** (movement.py:1220-1224) → the CLAUDE.md head-frame counter-rotation bug, **fatal for an audio
reflex because the mics wouldn't turn**; (b) **blocks the DN loop thread** with `sleep` for the full 5-8 s
turn (movement.py:1208-1233); (c) self-inhibits DN. The reflex needs a **new, non-blocking body-rotation
dispatch that ships an explicit head matrix**. This is net-new generic-DN motor wiring, not "free."

**BL-2 — DN's tick never evaluates behaviors on an audio-only input, and no seam carries azimuth (or
`nac`) into a `Behavior`.** *[cross-confirmed: Architecture B1 + Executor S1; spot-verified network.py:912]*
`_process_tick` early-returns `if not detections: return` (network.py:912) *before* `_evaluate_behaviors`;
`BehaviorState` has no azimuth field; `Behavior.evaluate(detections, state)` receives YOLO detections only
and must be `<10 ms`/non-blocking. The whole point of audio orienting is the *no-visual-detection* case,
which the tick discards — so `AudioOrienting.evaluate` would essentially never fire. Servicing audio needs
(a) un-gating behavior evaluation on audio-only ticks and (b) a channel that pushes the latest gated
azimuth + a `nac` ref into the behavior (DN-pushed setter precedent: `TurnAround.set_head_yaw`). The
reader can't be pulled inside `evaluate` — `make_reachy_rest_doa_reader` does a blocking HTTP call.

**BL-3 — The policy/binning helpers live only in `scripts/`; nothing promotes them to `src/`.** *[Executor
B1; unique but decisive]* `az_bin`, `decision_boundary`, `save/load_policy_meta` (the `.meta.json` sidecar),
and `load_orient_actions` live in `scripts/orient_backbone/live_common.py` + `live_3_learn.py`; `grep` of
`src/maxim/` returns zero matches. `src/` cannot import from `scripts/`. The reflex (specified under
`src/maxim/default_network/behaviors/`) therefore has **no code path to bin a live azimuth the way the
policy learned it, nor to load the sidecar**. A `src/maxim/` home for these helpers is a required, currently
missing, first step. (P2's keying logic is correct — it just lives in the wrong place for the runtime.)

**BL-4 — Reflex learning on the shared NAc silently pollutes the LLM's tool-value surface (and vice
versa).** *[cross-confirmed from two directions: Bio-fidelity B1 + Architecture N1]* `DefaultNetwork` holds
the *same* NAc the agent loop + Wire-A use. The reflex calling `update_cluster_reward` at DN rate would
accumulate into `get_agent_tool_biases` (nac.py:2091, aggregates per-tool across all clusters, agent-wide),
which Wire-A surfaces into the LLM prompt — the deliberative layer would silently read "you've been
rewarded for turn_body" from reflexive motor activity it never chose. Symmetrically, `recommend_action`
folds in causal-link confidence + `reward_bias` on a shared `agent_id`, so the LLM's history would perturb
the reflex's action scores. **Fix:** isolate the reflex — dedicated `agent_id` + either a separate NAc
instance or a reserved cluster/tool namespace excluded from `get_agent_tool_biases`. Same failure family as
the Wire-1 key-embedded-statistic and EC centroid-drift bugs (right signal, wrong aggregation level, fails
silently, tests don't cover it).

**BL-5 — IOR / novelty gating is claimed "free" but silently no-ops for target-less audio proposals.**
*[cross-confirmed: Bio-fidelity B2 + Architecture]* The salience/fear/IOR gates short-circuit `return True`
when `proposal.target` is falsy, and IOR runs in visual (u,v) pixel space via `_gaze_history`. The audio
reflex proposes a bare `yaw_delta` with no `target`, so it sails through untouched — "so the robot doesn't
chase every cough" is **false as designed**. Bio: IOR's home in the DN/SC-analog is correct, but an auditory
bearing must enter a spatially-registered map for IOR to fire. **Fix:** register the bearing into an azimuth
IOR map (or a spatial target the SalienceMap understands), or stop describing the gating as free.

---

## SHOULD-FIX

**SF-1 — DN↔LLM inhibition: wire ONE-WAY, and it isn't wired the way the plan assumes.** *[cross-confirmed:
Bio-fidelity S1 + Architecture S1]* Biology is asymmetric (antisaccade: cortex/FEF suppresses the collicular
reflex, not the reverse) — so voluntary-gaze-suppresses-reflex is the load-bearing direction; the reflex must
NOT inhibit cognition as a peer. The one legitimate bottom-up path is salience *escalation* through the
thalamic gate (gate.py already escalates), not reflex-suppresses-LLM. Symmetric mutual inhibition would
freeze/oscillate the head. Also: today only `turn_around` self-inhibits DN; voluntary `look_at` does **not**,
and `inhibited_behaviors` is populated from *mode config*, not live per-tool LLM suppression — so Open Q1's
assumed mechanism does not exist yet and needs concrete design, not a deferral.

**SF-2 — Efference copy / corollary discharge is mandatory, not "small" (Open Q4).** *[Bio-fidelity S2]* The
reflex should be silent as to the *decision* (orienting is pre-attentive) but MUST emit a *self-motion signal*
so cognition can distinguish self-generated turning from world motion — otherwise the LLM sees the whole scene
shift after a reflexive turn and misattributes it. Flows into Track 1's `body_state` (post-orient azimuth →
prompt); state the requirement explicitly.

**SF-3 — Credit signal is error-correction (cerebellar), not reward (dopaminergic); name it and defer gain
adaptation to the Cerebellum.** *[Bio-fidelity B1b]* `|az_before| − |az_after|` is saccadic/orienting
adaptation — `embodiment/cerebellum.py` (Rescorla-Wagner forward model) is the bio-correct organ. Acceptable
to ship the *discrete* 5-bin policy on NAc as a tabular bandit, but NAc reward-bias can only re-rank fixed
`action_deltas`; it cannot represent "the same action now yields a different Δaz" — i.e. it silently tolerates
a wrong *gain*, which is exactly the head=None regime. Name the caveat; route continuous-gain recalibration to
the Cerebellum in a future plan.

**SF-4 — The Acting Coach centeredness fix re-commits the smell it fixes.** *[cross-confirmed: Architecture
S4 + Bio-fidelity N1]* Track 1 Gap 2 proposes teaching `_compose_drive_modulation` the centeredness case —
but that bakes a *second* drive-name-specific hardcoded branch into the generic coach (the first is the
thermal "seek warmth" misfire). The robot-agnostic fix is a **data-driven drive→guidance mapping** (each drive
declares its own modulation text), not another `if drive == "centeredness"`. Otherwise robot #3's new drive
re-triggers the identical bug. Also a drive-semantics category error: rendering an exteroceptive bearing as a
homeostatic instruction.

**SF-5 — Track 2's dependency on Track 1 is overstated.** *[cross-confirmed: Architecture S2 + Executor]* By
its own P2 design the reflex bypasses `SensorEncoder`/interoception entirely — it needs Track 1's
`[declaration]` seam (robots.yaml body/reflex fields) but **not** the interoceptive per-iteration
`evaluate_failures` tick or `executor.embodiment`. Reframe "prereq: Track 1" as "shares Track 1's declaration
seam + the `src/` policy-helper home (BL-3)"; the interoception coupling is not real.

**SF-6 — Front-gate the pending-map credit mechanism.** *[Architecture S3]* DN already ships `FocusLearner`
and `PainCircuitBridge` delayed-outcome learners; the plan asserts the pending-map is "the one genuinely-new
piece" without naming why neither fits (Principle 3). It likely *is* new (NAc cluster-reward, not gain/pain),
but say so.

**SF-7 — Startle vs orient compose sequentially; there is no audio startle today (Open Q3).** *[Bio-fidelity
S3]* Startle (protective, fastest-habituating) and orient (investigative) are distinct co-occurring circuits;
`PriorityArbiter` winner-take-all models only the instant. `behaviors/startle.py` is vision-only, so there is
no audio startle to compose with — if it matters here it's net-new work + habituation/priority-decay.

---

## NICE-TO-HAVE / execution details

- **NH-1** *[Executor N1]* `recommend_action` is keyword-only; learned entries key on `tool_signature=f"tool:{action}"`, and lookup re-prefixes `tool:` — so the reflex MUST pass `available_tools=[orient affordance names]` (e.g. `turn_left`/`turn_right`) or the lookup misses. Call this out in the pseudocode.
- **NH-2** *[Executor S2]* `RobotConfig` has no `body:` field (only `robot_id/robot_type/primary/config`) — either add a typed field or ride the existing free-form `config["body"]`; the plan should pick one.
- **NH-3** *[Executor S3]* The default-off flag has no concrete `*ConfigSection`/`resolve_setting` path yet; per the config-over-env-vars standard, name it.
- **NH-4** *[Executor N3]* `has_audio` is only populated post-`connect()`; evaluate the gate post-connect or audio reads as absent.
- **NH-5** *[Bio-fidelity N2]* If an audio startle is added, its priority (0.95 > orient 0.8, winner-take-all) must habituate/decay or it starves orient.

---

## AFFIRMATIONS (keep these — validated by the review)

- **Track 1 is implementable as written**; the Track A tick fix calls the public `evaluate_failures()` (not raw `tick_vital_drift(`), so it does not trip the body.py-only CI grep *[Executor N2]*.
- **P2 az_bin string keying is correct** — `recommend_action` genuinely takes a string cluster key; the queen-mind policy is usable as-is, no retrain, no SensorEncoder for azimuth *[all three lenses]*.
- **The exteroceptive az_bin bypass is the bio-correct call** — a world-bearing is allocentric (parietal/collicular), not interoceptive; routing it around the sign-folding interoception encoder is right *[Bio-fidelity N3]*.
- **DN = fast subcortical orienting / LLM = cortical voluntary control** holds; reflex-under-continuous-cognition is a two-layer architecture, correctly NOT an `aut_mode` switch *[Bio-fidelity, Architecture]*.
- **`make_reachy_rest_doa_reader` routes through `maxim.utils.http`** — no HTTP-surface invariant violation *[Executor]*.
- **The offline-test seams exist** (reader injection, standalone NAc, pure-Python pending map) — the offline test is achievable once BL-2 + BL-3 land *[Executor]*.

---

## Disposition

Track 1 → cleared to implement, with SF-4 folded (data-driven drive→guidance mapping, not a second hardcoded
branch). Track 2 → **revise before code**: fold BL-1..BL-5 as owned steps, SF-1/SF-2/SF-3/SF-5/SF-6 into the
design, and reframe the front-gate table honestly (what DN provides vs what this must add). Both remain gated
on the `turn_around` head-frame audit before any live dispatch and on hardware validation once the motor is
repaired.
