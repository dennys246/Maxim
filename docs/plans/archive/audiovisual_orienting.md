# Audio-Visual Orienting — unified embodied attention via a shared orient-to-center substrate

**Status:** Coordination plan, drafted 2026-06-28. Unifies two in-flight tracks; does NOT re-do either.
**Scope:** define the shared orient-to-center backbone both tracks converge on, the cross-track
sequencing, and the fusion endgame. The deliverable is the *unification + sequence*, not new mechanism.
**Branch state (must reconcile):** the two tracks currently live on separate branches with no common
base — audio on `feat/reachy-doa-audio-source`, visual on `feat/gaze-substrate-probes`. Phase 0 below
requires getting both tracks' foundations onto one base first.

## The two tracks

| | Audio (DoA) | Visual (substrate gaze) |
|---|---|---|
| Plan / record | [`perception_pipeline_placement.md`](../perception_pipeline_placement.md) | [`docs/experiments/43_gaze_operant_substrate.md`](../../experiments/43_gaze_operant_substrate.md) |
| Code | [`embodiment/audio_localization.py`](../../src/maxim/embodiment/audio_localization.py) | [`scripts/gaze_substrate/`](../../scripts/gaze_substrate/) |
| State | commits 1–4 done (DoA front-end landed); commit 5 pending hardware | sim-validated (operant redirection, cross-session persistence, visual-category transfer to novel instances); P1 pending |
| Maturity | perception **solved** (chip gives azimuth); motor/credit pending | motor loop **sim-validated**; perception (vision-encoder on real images) **unvalidated** |

## The convergence (why this plan exists)

Both reduce to **one control loop**: azimuth/bearing error → orient affordance → credit for reducing
|error| → centered. The audio module's own docstring notes its azimuth ([-1,1], 0=center) is "exactly
the centeredness set-point shape the homeostatic drive wants." **Audio commit 5 and the visual motor
loop are the same commit — build it once.**

They are also **complementary at the hardware-limitation level**, which is what makes co-running
valuable rather than redundant:
- **Audio:** 360° coverage, densely/continuously sensed, but coarse and front/back-ambiguous (linear array).
- **Vision:** precise and identity-bearing ("it's a *person*"), but FOV-limited.

Co-localization = each covers the other's blind spot: **vision resolves audio's front/back ambiguity;
audio pulls out-of-FOV targets into view for vision to confirm.** This is the superior-colliculus
audio-visual orienting map — strong bio-grounding, and the *easiest* instance of cross-modal binding
because both modalities collapse to one shared spatial variable (head-relative azimuth).

## Central clarification: TWO learning signals, not one

The tensions between the tracks dissolve once you see the system has two distinct, non-conflicting signals:

1. **Motor centering (SHARED, dim-trivial).** Orient to reduce azimuth error. One
   `HomeostaticDriveSpec` centeredness drive (set_point 0) → drive-pain → NAc credit on discrete
   `turn_left`/`turn_right`. **Both audio and visual feed their scalar azimuth/bearing into this ONE
   drive** (scalar → 384-d `SensorEncoder` space → dim-consistent across modalities).
2. **Target valuation (MODALITY-SPECIFIC).** "Is this worth orienting to?" Audio: the hardware
   `is_speech_detected` gate (no learning). Visual: EC **category recognition** (person vs distractor),
   `cluster_reward` on orient/ignore — the substrate-generalization payoff that transfers to *novel*
   individuals ([Exp 43 Probe 4](../../experiments/43_gaze_operant_substrate.md); high-dim `"vision"` modality).

Keeping these separate is the load-bearing design move: the shared motor loop is scalar and bio-faithful;
the valuation is visual-specific and high-dimensional. They never compete.

## Reconciled decisions (where the tracks differed)

- **Action path = substrate-primary, NOT LLM-gated. [CONFIRMED 2026-06-28]** The orient loop selects via
  `recommend_action` (no LLM in the action path), per the Exp 43 probes. This **removes the audio plan's
  "sustained-sound / slow-convergence" constraint**, which assumed one LLM cognition cycle per orient
  step. The visual track's substrate-primary validation is a genuine upgrade to the audio latency story.
- **Motor credit [RESOLVED 2026-06-28, Phase 0b]: credit drive-pain REDUCTION per step (relief), as a
  state-conditioned positive (`potential_diff` on `cluster_reward`) — NOT the off-center pain state.**
  Phase 0b ([`scripts/orient_backbone/phase0_affect_study.py`](../../scripts/orient_backbone/phase0_affect_study.py))
  found **pain alone cannot drive `recommend_action`**: it is positive-gated (negative-only pain has
  nothing to select on → defaults to `stay`), the causal/`record_outcome` surface is per-tool
  **state-blind**, and off-center pain misinforms mid-approach (it punishes correct-but-incomplete
  actions). Reward (state-conditioned `cluster_reward`) drives a perfect orient policy; `potential_diff`
  (dense per-step |az|-reduction credit) additionally closes the far-state gap (far-dir 1.0 vs
  terminal-reward 0.94). **The reconciliation:** `potential_diff` *is* drive-pain *reduction* = relief =
  negative-reinforcement — bio-faithful AND mechanically selectable — so it folds the audio (drive-pain)
  and visual (reward) credit approaches into one signal. Ties to
  [`transition_based_drive_pain.md`](../deferred/transition_based_drive_pain.md) (credit transitions / relief, not
  the per-tick pain state). Consistent with the existing NAc invariant (reward_bias clamps [0,max]; pain
  is a modulator handled by edge valence, not a selection driver) — Phase 0b makes that concrete for orienting.
- **EC modalities stay separate:** `"audio"` (frozen, [-1,1], 384-d; RESOLVED Q5 in the audio plan) and
  `"vision"` (frozen, vision-encoder dim) for category. The **spatial azimuth** from both is scalar →
  same 384-d space → dim-consistent (relevant only to binding, Phase 3b).
- **Plan-drift fix (audio plan hygiene):** `perception_pipeline_placement.md` still describes commit 4
  as "4-mic ITD-TDOA"; the XVF3800 (2-ch processed only, no raw 4-mic) forced **DoA-consumption**, which
  is what shipped. Update that plan's commit-4 description to match reality.

## Phased sequence (cross-track)

**Phase 0 — sim, now, no hardware: build the shared backbone + characterize the credit signals.**
- **0a — backbone:** a modality-agnostic centeredness drive + discrete orient affordances +
  substrate-primary NAc credit, consuming an azimuth-error from *any* `PerceptSource`. This is Exp 43
  Probe 1/2 generalized AND audio commit 5 generalized — both tracks' "commit 5" collapse into this.
  Validate in sim (world-coupling rule for the sim; on hardware the mic/camera physically re-measure).
- **0b — affect-signal study [DONE 2026-06-28]:** ran pain / positive / both / `potential_diff` on the
  0a backbone. Verdict (see Motor-credit decision above): orienting is **reward-driven**; credit
  drive-pain **reduction** (`potential_diff`, state-conditioned); **pain-state alone does not orient**.
  A relative-gain sweep would only refine magnitudes — the qualitative verdict is settled, so it is not
  a 0a blocker.
- **Prereq:** both branches' foundations on one base (see branch coordination below).

**Phase 1 — hardware, AUDIO first.**
Wire `AzimuthDoASource` → shared backbone, live. Audio leads because its perception is *solved* → the
cleanest path to a working live learning loop, and it **validates the shared motor/drive/credit substrate
on-device** (which visual then inherits for free). Azimuth is densely sensed → ideal testbed for the
potential-difference reward shaping. Deliverable: *Reachy turns toward sounds and learns.* (= audio
commit 5, now on the shared backbone.)

**Phase 2 — sim → hardware, VISUAL.**
First clear **P1** (vision-encoder category clustering on real skier-vs-object *images* — Exp 43
prerequisite, no robot; the make-or-break for the substrate payoff). If it holds, wire the visual
`PerceptSource` (bearing → motor loop; category → valuation) into the *same* backbone. Deliverable:
*Reachy turns toward people and generalizes to novel people.*

**Phase 3 — hardware, fusion (two tiers).**
- **3a — drive-level azimuth fusion (cheap, no binding).** Both modalities' azimuth → one centeredness
  error; arbitration by confidence/recency; vision resolves audio front/back; audio extends vision FOV.
  Dim-agnostic. **This is the practical co-localization win** and needs none of the binding machinery.
- **3b — substrate cross-modal binding (GATED research).** Audio-azimuth node ↔ visual node binding into
  a multimodal object. **Gated on a Roy-4-style co-activation measurement** (`MAXIM_EC_TRACE_ACTIVATIONS`)
  per the audio plan: `cross_modal_substrate_binding` was cancelled because text clusters never co-fired
  in time. **Spatial AV has a structural reason to co-fire that the text case lacked** — a
  simultaneously-seen-and-heard target produces both percepts in the *same* tick — so it is the tractable
  case, but **measure before building.** Do not resurrect the cancelled mechanism; fresh plan if it passes.

## Branch coordination [DECIDED 2026-06-28]

Both tracks land on `main` via PRs, then Phase 0 happens in one branch/session on top:
1. **PR — visual side** (`feat/gaze-substrate-probes` → main): Exp 43 probes + writeup + this cohesive plan.
2. **PR — audio side** (`feat/reachy-doa-audio-source` → main): commit 4 (DoA front-end) + the commit-4
   plan-drift fix (ITD-TDOA → DoA-consumption) in `perception_pipeline_placement.md`.
3. After both merge: branch `feat/orient-backbone` off main; do Phase 0 (0a + 0b) there in a single session.

## Open questions (pre-implementation review)

1. **Empty-state arbitration** — audio-empty (no sound → wait) vs vision-empty (no person → Layer-0 search)
   differ; the fused drive needs a policy for "audio says turn, vision sees nothing yet."
2. **Shared coordinate frame** — reconcile audio `(doa−π/2)/(π/2)` sign against the camera bearing sign
   *explicitly*, or the two modalities pull the head opposite directions.
3. **Shared body YAML** — [`reachy_mini.yaml`](../../src/maxim/_data/components/bodies/reachy_mini.yaml)
   already exists (head_yaw); add the camera/vision sensor + orient affordances there (one body, both tracks).

*(Resolved: motor-credit pain-vs-positive → Phase 0b experiment; action-path latency → substrate-primary, confirmed.)*

## Pointers
- Audio: [`perception_pipeline_placement.md`](../perception_pipeline_placement.md), [`audio_localization.py`](../../src/maxim/embodiment/audio_localization.py)
- Visual: [`docs/experiments/43_gaze_operant_substrate.md`](../../experiments/43_gaze_operant_substrate.md), [`scripts/gaze_substrate/`](../../scripts/gaze_substrate/)
- Binding (cancelled / gated): [`cross_modal_substrate_binding.md`](cross_modal_substrate_binding.md), [`jepa_cross_modal_alignment.md`](../deferred/jepa_cross_modal_alignment.md)
- Drive-pain refinement: [`transition_based_drive_pain.md`](../deferred/transition_based_drive_pain.md)
