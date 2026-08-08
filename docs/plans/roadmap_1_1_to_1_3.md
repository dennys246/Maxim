# Roadmap 1.1 → 1.3 (scoped 2026-08-07 by four-lens review)

**Status:** ACTIVE. Supersedes [release_1_1_checklist.md](release_1_1_checklist.md)
(which was itself stale on half its items) as the 1.1 scope authority.
**Method:** four parallel review lenses — audio front-end, reflex wiring,
bio-fidelity, release scope — each grounded in code rather than plan docs.
**Headline finding:** *the roadmap was being drafted from the plans' ambitions
rather than from the audits' findings.* Every blocker below was already written
down somewhere in this repo by someone being careful.

---

## The three versions

| | Theme | Contains | Risk |
|---|---|---|---|
| **1.1** | **"Sensorimotor"** — *the substrate leaves the simulator* | Already-merged embodiment work + the truth/hygiene debt it incurred. **Zero new mechanisms.** | Low. One hardware session, currently unblocked. |
| **1.2** | **Oasis + Hivemind** | ~1,400 LOC of de-risked engineering with a cleared gate | Low-medium. Known shape. |
| **1.3** | **Perception fabric + reflex tier** | Cochlear front-end, vision encoder, binding, three-factor calibration, DN-canonical orienting reflex | **High — contains a pivotal may-fail experiment.** |

### Why this ordering (two corrections to an earlier draft)

An earlier sketch put the perception fabric at 1.2 and Oasis at 1.3. Both moved back:

1. **Put the predictable deliverable near, the may-fail research far.** 1.3's Stage 0c
   is described by its own plan as *"the single most informative outcome"* — a pivotal
   test whose failure means emergent identity needs a different mechanism. A re-plan
   costs least in the far slot. Oasis is de-risked engineering with a known shape.
2. **Respect the plans' own stated targets and published commitments.**
   `cross_modal_perception_fabric.md` and `three_factor_credit_assignment.md` both
   self-target **1.3**; pulling them to 1.2 was against their headers with no new
   evidence. Conversely `maxim_hivemind.md` and this README both commit Oasis to
   **1.1** — a one-minor slip (→1.2) is an honest re-scope; two minors would be a
   quiet abandonment of a published commitment.

The earlier "build the thing worth sharing before the sharing" argument still holds
directionally — Oasis's near-term payload (direction policy) is per-body calibration,
the *least* transferable thing — but it is answered by landing **artifact stamping in
1.1** (below) so the calibration leak can't bake into circulating bundles, rather than
by delaying Oasis two versions.

---

## 1.1 cut line

> **Ship the work that is already merged, plus the truth and hygiene debt that work
> incurred. Zero new mechanisms. One hardware session.**

| # | Item | Est. |
|---|---|---|
| 1 | **PR #467** — `_cosine` dimension guards (hivemind **and** EC), frozen-modality parity, CLAUDE.md correction | open, green |
| 2 | **CHANGELOG + git tags** for 1.0.0→1.0.6 — **88 PRs, ~14,000 `src/` lines, zero entries, no tags past `v1.0.0`** | 3–5 d |
| 3 | **n_ctx clamp** (`llm_worker.py:299`) — `max()` twice where CLAUDE.md says `min`, inside a bare `except`, gated on `_has_cloud_providers()` so local runs never clamp | 1 d |
| 4 | **Persona hard-remove** — "removed in 1.1" promised in 7 shipped user-facing files; needs the Option A/B/C decision blocked since April | ~1 wk |
| 5 | **Graduation heartbeat walk** (9 rows) — the **first ever**; no row has ever been marked `Maintained` | 1.5–2.5 wk |
| 6 | **H1 hardware session** — version-verified DoA re-sweep (≥2 geometries, 07-16 protocol) + `yaw_verify` + motor-bound delivered-shift measurement | 2–3 d |
| 7 | **Artifact stamping** — `embedding_dim`, `using_fallback`, sensor-name set, normalization mode. Pulled forward from the fabric plan's Stage 4 | 80–150 LOC |
| 8 | **Orient-vocabulary audit + workspace-limit bypass fix** — ⚠️ **SAFETY**: the two `goto_target`-bypassing paths are the likely cause of the motor destruction (see hardware note). Highest priority after #467 | 3–5 d |
| 9 | Doc-truth pass — this file, README, `perception_placement.py` disposition, Exp 09 citation | 2–3 d |

**Total: 5–7 weeks**, one external dependency (H1), currently unblocked.

### H1 buys three things at once — now four

The DoA re-sweep resolves **Exp 45's staleness**, *is* **1.3's Stage 0a**, and provides
**motor-binding Phase 3's gain calibration**. Buy once. Fold the contingent H2 branch
(magnitude re-probe, only if H1 moves the ≈0.33 decision boundary) into the
pre-registration **before** H1 runs, so its outcome is decided in advance.

### Hardware note (2026-08-07): motors 2 and 3 were broken for the ENTIRE 1.0+ era

**Operator report:** motors in Stewart positions **2 and 3** were broken and have now
been replaced and reflashed from the motor-1 config. All six legs confirmed healthy;
the platform moves cleanly. **The breakage spans essentially all of 1.0+**, repaired
~2026-08-05.

**Root cause, and it is ours:** an earlier Maxim iteration commanded a pose *beyond its
physical capability*; the motors glitched, the head snapped violently to the opposite
extreme, and the robot rotated itself off the table. This is a **workspace-limit
enforcement failure**, and it connects directly to the orient-vocabulary audit (item 8):
**two paths bypass `ReachyMiniController.goto_target` entirely** — `MoveTool` gaze
without a `robot_id`, and `turn_around`, which hand-rolls its own centering. A path that
bypasses the controller plausibly bypasses workspace clamping with it. That elevates
item 8 from a correctness fix to a **safety** fix, and it should be treated as the
highest-priority item in the 1.1 cut line after PR #467.

**Data-quality consequence — the important part.** Every live-hardware measurement in
the 1.0+ era was taken on a degraded platform, including:

- the 2026-07-16 "TRUE characterization" (0.57 az/rad, R²=0.9982) — **also degraded**,
  contrary to the earlier hypothesis that it was the healthy baseline
- the 2026-08-05 contested sweep (~0.19 az/rad)
- Exp 45 / 45b / 45c / 45d / 45e (orient direction + magnitude)
- Exp 46 / 48 (operant orienting)
- every live orient session and smoke test

**This yields a better hypothesis for the contested curve than either previous one:
progressive mechanical degradation.** If motors 2 and 3 were failing *gradually*, then
0.57 (July) and 0.19 (August) are both real measurements of a platform in two different
states of decline — not one good run and one instrument artifact. That predicts
**healthy hardware should now measure ≥0.57, plausibly nearer the geometric 0.637.**
H1 tests it directly.

**What survives and what does not.** Direction findings are likely robust — if the body
turns at all, which way it turned is preserved. **Magnitude findings are not**: delivered
shift is exactly what a degraded platform corrupts, and the magnitude line already
rested on n=1 sessions. Treat every magnitude claim (the ≈0.33 decision boundary, Exp
45b/45c/45d/45e magnitude arms) as **provisional pending re-measurement**, and say so in
the graduation walk rather than assuming clean.

**H1 is therefore the first honest hardware measurement in the project's 1.0+ history**,
and its value is much higher than originally scoped. Run it early.

**Operational:** stock multiple spares — this incident needed more than the one held.
And two failures from one root cause argues for fixing the *cause* (workspace-limit
bypass) rather than only the symptom.

---

## Why the reflex tier is NOT in 1.1

The verb "enable" was doing enormous work. Verified against code:

- **The `sim.is_sim_mode` gate does not gate a motor command.** It gates a *modeled
  sensor write* (`world_set_azimuth`). Removing it enables a live **fabrication**, not
  a live reflex — the head-frame failure class that cost a full session and six
  falsified hypotheses. SEM motor binding gave real motion to the *affordance* path;
  the §1.16 reflex has no dispatcher of its own.
- **`world_set_axis` has no `live_world_set_sensors` check.** That guard exists only
  inside `ModulatorAffordanceTool`, so a live reflex write would bypass phantom-credit
  protection entirely and break `DoAFeed`'s single-writer claim.
- **A fabricated centering would null the drive that motivates real orienting.**
  `azimuth → 0.0` sits inside `comfort_band: 0.1` → no breach → no drive pain → the
  policy loses its reason to turn. With `drift_rate: 0` and silence-writes-nothing,
  that persists **indefinitely** in a quiet room.
- **The trigger predicate has no measurement behind it.** `salience`/`novelty` are
  constructor constants from `robots.yaml`, so `is_orienting_reflex` fires *never*
  (0.5 vs 0.9) or *on every speech-gated reading*. Worse: `AzimuthDoASource` gates on
  `is_speech_detected`, so **a clap or a bang never produces a reading at all** — the
  transient stimulus class is filtered upstream.
- **Real scope is ~900–1,400 production LOC**, with an unresolved design decision
  (reflex NAc isolation) and a *mandatory unbuilt prerequisite* (`sem_motor_binding.md`:
  the pending map is required before any non-blocking dispatcher ships).
- ~~**Sign-off is motor-blocked.**~~ **2026-08-07 — a spare motor was swapped in.**
  Four plan docs gated on this. **TWO motors were found broken**, so whether the gate
  is fully open depends on whether both are now healthy — see the hardware note, which
  also raises a more interesting question.

---

## Reflex canonicalization — the decision (lands in 1.1, the behavior does not)

**Decision: the DefaultNetwork is the canonical home for all reflexes.** It inherits
`PriorityArbiter` (arbitration), `BehaviorState.inhibited_behaviors` (one-way
voluntary→reflex suppression — the bio-correct direction; symmetric inhibition
oscillates), and `Behavior.cooldown_seconds` (refractory). The §1.16 inline path
reaches none of these and is invisible to the DN — two controllers, one actuator, the
`head=None` failure class.

**The contract any future reflex must satisfy:**

1. Subclass `Behavior`; propose through `PriorityArbiter`.
2. **Never write sensors directly** — enforced structurally, not by convention.
3. Declare a cooldown (refractory) and take habituation/sensitization **gain** from the
   multiplier, never by mutating a declarative threshold.
4. **Emit efference copy.** A reflex that turns the body without telling higher systems
   makes the LLM read the whole scene shifting and attribute it to the world — a
   corollary-discharge failure, direct descendant of the `head=None` lesson.

**What ships in 1.1** is only the structural half — move the `live_world_set_sensors`
check into `world_set_axis` with an explicit `owner=` opt-out for legitimate writers
(`DoAFeed`, the backend's measured readback), so forgetting becomes a refusal rather
than a silent fabrication. Keep §1.16's branch (dormancy over deletion) but make it
*unable* to lie if someone un-gates it.

---

## The orient-vocabulary audit (1.1 item 8)

There are **seven** paths that command orientation, not three. Two are live bugs:

- **`MoveTool` gaze and `turn_around` bypass `ReachyMiniController.goto_target`** — and
  therefore the head-frame composition, the invariant earned at the cost of a full
  session and a three-doc retraction.
- **SEM `motion` affordances (`look_at`, `goto_pose`, `recenter`, `nod`, `shake_head`)
  are advertised to the LLM but are motorless no-ops** — `make_reachy_orient_factory`
  binds only `mod_name == "orient"`. Same dishonesty class PR #459 fixed for
  `focus_on_sound`.

Also designed-in and uncoordinated: `focus_on_sound` **recommends the SEM turn tool by
name** when clamped, with no coordinator, refractory, or mutual exclusion — a single
~45° sound can produce ~97° of world rotation. Only two of the seven paths call DN
inhibit.

**`ReachyMiniController.goto_target` has no lock**, and reads `get_current_pose()`
before composing the head matrix — a TOCTOU on live kinematic state. Two overlapping
callers each compose against a stale body yaw. A single `RLock` spanning read→compose→
dispatch is ~10 LOC and the highest-value fix in this list.

---

## Loudness — blocked outside this repo

`DoAReading = tuple[float, bool]`. The daemon serves `{"angle", "speech_detected"}`.
There is no level anywhere. Two paths, both outside our control or against current
config:

1. **Daemon-side** — a new endpoint or extra fields. That is Pollen's code. Unbounded
   calendar, and the 2026-08-05 SDK/daemon version-skew incident is a fresh reminder of
   what that dependency costs.
2. **Local PCM** — `mini.media.get_audio_sample()` is onboard-only and needs
   `media_backend: default`, abandoning the `no_media` config live orient sessions run
   under specifically to kill the frame thief (#456).

**Two bench tests answer this in under a day and must precede any plan:** (a) does
`media_backend: default` yield non-empty audio samples on the live rig, or is GStreamer
broken there? (b) does the XVF3800's **AGC** flatten RMS to near-constant? If (b) fails,
"loud" is unmeasurable on this hardware and the tier is onset-only forever.

**Consequence for 1.3:** the reflex tier's own trigger is gated on this. Until a
sound's intensity is measured, "loud and sudden" is a config constant wearing a
predicate's clothes.

---

## Bio-fidelity corrections (fold into 1.3's plan before implementing)

1. **Drop "startle" entirely — this is ORIENTING.** Startle is PnC (~5–10 ms,
   non-directional, protective bracing). Orienting is superior colliculus (~70–200 ms,
   directional, information-seeking). The code implements orienting.
   `behaviors/startle.py` is *also* actually orienting (vision-only, proposes
   look-toward) — a third mislabel would compound two existing ones.
2. **A sensitization experiment designed on startle literature would falsify for the
   right biological reason.** Repeated aversive pairing does not produce larger turns
   *toward* — in the defense literature it produces freezing or avoidance.
3. **Do not publish pain on startle.** An ambient sound has no entity, so the publish
   either no-ops or accumulates junk into `_percept_valences` →
   `get_percept_aversions` → `TextSalienceScorer`, boosting every future percept
   containing that token as aversive. Use `Reaction(kind="surprise")` — **declared in
   `ReactionKind` with zero producers today** (every Reaction in the codebase emits
   `kind="pain"`). For genuine nociception, declare a `sound_level` sensor with a
   homeostatic comfort band and let the existing severity-latched drive-pain channel
   handle it.
4. **Habituation ships with or before sensitization, never after.** Dual-process theory:
   response = S-R decrement + state increment. Sensitization alone escalates
   monotonically — the repo already warned: *"NH-5: an audio startle at priority 0.95
   must habituate or it starves orient."*
5. **Front-gate answer: ride `NoveltyTracker`.** It already implements dual-process for
   vision — habituation decay, spontaneous recovery, novelty floor, ceiling-clamped
   sensitization, with an explicit VTA rationale in `MemoryHub._wire_sensitization`.
   The work is generalizing it off COCO class keys, not building from scratch.
6. **Threshold stays declarative; gain is the plastic part.** CeA potentiates PnC's
   *response*, it does not lower its threshold. So:
   `effective = measured × habituation × sensitization_gain` tested against the
   **fixed** YAML threshold. No learned thresholds, no mutable YAML, no new state file.
7. **PPI explicitly out of scope** — the loop tick (33–500 ms) exceeds the PPI lead
   window, there is no sub-threshold acoustic channel, and no graded amplitude.
8. **Halo risk:** graduation Tier-3 #9 reads "Reflex system … EARNED — Experiment 09,"
   but that covers the keyword-matched NARRATIVE percept reflexes only (Exp 09 ran
   `bodies/base_humanoid` against dragon attack/fire narration — this sentence
   originally repeated the row's own "infant thermal contact" mis-description;
   corrected in the 2026-08-07 doc-truth pass). A different modality, trigger,
   circuit, and output **does not inherit it.**
9. **Better framing available:** SC is a *multisensory integration* structure, so
   orienting is arguably the oldest **instance** of binding rather than a layer beneath
   it. That makes 1.1→1.3 continuous rather than stacked.

---

## Free findings (fix during the 1.1 doc-truth pass)

- ✅ **RESOLVED (doc-truth PR, 2026-08-07):** **Graduation Tier-3 #9 cites a file that
  does not exist** (`09_percept_reflex.md`; the real file is `09_percept_reflex_poc.md`)
  **and mis-describes the experiment** (says infant thermal contact; Exp 09 ran
  `bodies/base_humanoid` against dragon/fire narration). A Principle-5 defect on an
  EARNED row. — Row corrected + halo caveat added.
- **`CaptureManager` is constructed only when `has_vision`** — an audio-only robot gets
  no audio thread at all.
- **`get_audio_stream()` returns non-`None` under `no_media`** — a capability lie at the
  stream surface. Gate on `_capabilities.has_audio`, never on stream presence.
- **`inhibit_during_tool_execution` is dead code** — zero callers, config field read by
  nothing, while a plan doc claims it "covers half the race." It covers none.
- **`Reaction(kind="reward")` is published** from `cerebellum_modulator.py` and is not a
  member of `ReactionKind`. The taxonomy is drifting unenforced.
- ✅ **RESOLVED (doc-truth PR, 2026-08-07):** **`perception_placement.py`** — 267 LOC,
  zero `src/` callers, claimed "✅ shipped." Wire it or mark it Dormant per Principle 2.
  — Marked Dormant (module docstring, resurrection trigger = the 1.3 fabric actually
  placing stages); plans README claim corrected to PARTIALLY landed.

---

## Enthusiasm-to-evidence flags (recorded so they don't recur)

1. **Loudness scoped as a 1.1 addition without checking the wire format.** It is
   `(float, bool)`; the fix lives in a vendor's daemon.
2. **"Enable the reflex tier"** — the verb implied a flag; the reality is ~1,000 LOC
   against a deliberate correctness fold.
3. **Fabric pulled 1.3 → 1.2** against its own header, with no new evidence.
4. **"Sensorimotor" as a headline** — the *direction* result is robust; the *magnitude*
   line rests on n=1 sessions on a robot with a known motor asymmetry, atop a DoA gain
   contested by 3×. The docs are honest about all of this; the roadmap headline must
   inherit that honesty and claim the loop, not the sensorium.
