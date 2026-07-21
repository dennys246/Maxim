# Cradle: the mother-scaffolded orient experiment (build plan)

**Status:** Building (2026-07-20). The current, buildable spec — supersedes the closed #404 draft (whose specifics predated GAP-1/P1/the motor-credit value-progress fix). Rides the now-complete orient substrate foundation on `main`: the **value-progress motor-credit** (drive-relief → cluster reward) + the **range-aware fold fix** (signed azimuth separates left/center/right). No LLM in the infant's action path (substrate-primary).

## The claim
A hungry infant, and a mother who calls from a direction, physically turns the infant's head toward her (a fading scaffold), speaks motherese, and feeds it when it faces her. Does the infant **learn to orient toward the mother's voice itself** as the scaffold fades — cross-session, no fine-tuning? The fraction of the orient the infant produces *itself* per act is the measured learning curve.

## The per-turn loop (load-bearing order)
Each turn the world (reactive mother) acts on the PASSIVE infant, then the infant takes its substrate-primary turn. Order inside the mother tick:

1. **Feed if oriented** — reads the azimuth the infant's *prior* turn left (`|azimuth| ≤ oriented_threshold`) and, if facing the mother, relieves hunger/thirst. **This must come first**: it rewards the infant's own prior orient. If feeding read the post-stimulus/post-guide azimuth instead, each new stimulus would overwrite the infant's orient before it could be rewarded → the infant never learns (the silent-break this ordering avoids). Feed-then-stimulus also gives the temporal structure *orient this turn → fed next turn*.
2. **Place the stimulus** — world-set the infant's `azimuth` to the mother's direction for this turn (a deterministic per-seed sequence, e.g. alternating/spread left–right). This is "the mother calls from over there." In substrate-primary the #403 §1.16 audio world-set is gated out, so the mother tick sets it directly (world-driven, not the AUT path).
3. **Guide (the fading scaffold)** — world-set `azimuth` toward center by `guide_strength` (Act 1 fully centers; Act 3 not at all). The caregiver can center beyond the infant's own motor reach.
4. **Speak** — inject motherese as a text percept (the non-gated `bridge.percept_source.inject_cli`, since `send_and_wait` is suppressed in substrate-primary). The audio-visual + language-grounding channel.

Then the infant's substrate-primary turn runs (`propose_via_substrate` → orient action). Because azimuth is now range-encoded (P1) and relief is value-progress-credited (motor-credit), turning toward the sound is a positively-credited, direction-conditioned action.

## The fade curriculum (4-act arc `cradle_mother`)
`MotherScaffold` per act (`guide_strength`, `feed_amount`, `oriented_threshold`, `stimulus_azimuths`, `speech`):

| Act | guide_strength | infant must | fed when |
|---|---|---|---|
| 1 fully guided | 1.0 | nothing (passive) | mother centers it → fed |
| 2 co-active | 0.5 | complete the turn | it finishes orienting |
| 3 autonomous | 0.0 | orient itself | it orients |
| 4 autonomous (voice) | 0.0 | orient to the voice | it orients |

**Measured signal:** per act, the fraction of turns the infant is oriented *by its own action* (vs by the mother's guide) — the fade curve. Plus latency-to-fed and cross-session (`aut_nac.json`).

## 3-arm ablation (taught vs driven)
- **A (taught):** full scenario (fading guide + feed + speak + stimulus).
- **B (drive-only):** stimulus + hunger/centeredness drives, no mother guide/feed.
- **C (scaffold-only):** guide + feed but learning reward disabled (`MAXIM_NAC_REWARD_BIAS_DISABLED=1`).
A reaches autonomous orienting and C does not → learned, not innate or hand-fed; A ≫ B → the taught signal beats the built-in drive.

## Build pieces
1. **`reactive_mother_tick` redesign** (feed-prior → stimulus → guide → speak; `MotherScaffold.stimulus_azimuths`) + unit tests. ← *this step*
2. **`NarrativePhase.mother_scaffold` field** + the generative-runner per-turn hook (call the tick before `send_and_wait`, embodiment in scope) with a substrate-safe inject.
3. **`cradle_mother` 4-act arc** in `BUILTIN_ARCS` with the fade schedule.
4. **Harness + analyzer** (mirror `benchmark_exp42_preference.py`) measuring the fade curve.
5. Validate (offline mechanism + a real substrate-primary run) → two-lens review → land → the mac-mini behavioral run.

## Connections (why this is foundational)
Act 4 (orient to the *voice*) is the cross-modal step JEPA sets up + feeds (voice ↔ face ↔ food paired data); the motherese grounds first symbols (grounded-language); the gaze machinery + the value-progress motor-credit are the proven substrate this rides. The smallest task exercising action-conditioned prediction, cross-modal binding, grounded symbols, and reward-driven policy at once, on a real reward.