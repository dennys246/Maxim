# Substrate-primary orient learning — does the agent *learn* to orient (Decision-4 + the experiment)

> **ADOPTED INTO THE ROADMAP 2026-08-13 (historical design record).** This doc was written
> 2026-07-19 on `docs/substrate-primary-orient-plan` and never landed; it is the design
> ancestor of work that has since SHIPPED, so it lives in `archive/` with this map rather
> than as an active plan:
> - **P1 (range-blind `_normalize_value` fold)** → shipped 2026-07-20; now the
>   "Signed sensors MUST be encoded WITH their range" CLAUDE.md invariant
>   (`tests/unit/test_normalize_value_range_aware.py`).
> - **P2 (azimuth bundled into interoception)** → shipped as the extero/intero
>   ModalityChannel split (PR #411, `exteroception_interoception_seam.md`).
> - **P3 (audio percept routing gap)** → shipped as the extero/intero seam that lifted the
>   embodied cradle_mother sim off chance (PR #412, Exp 48).
> - **The experiment** → realized as Exp 46 (scripted operant orient, COMPLETE) and Exp 48
>   (embodied seam, GRADUATE → CONTESTED 2026-08-11, re-baseline pending).
> - **The residual live ask** — orient learning in the full production loop on live
>   hardware — is owned by `orient_runtime_integration.md` (1.2 unless explicitly pulled).

**Status:** Plan (2026-07-19). Follows the productive-orienting work (PR #403, Phase 1+2):
[productive_orienting_affordance.md](../deferred/productive_orienting_affordance.md). That shipped the orient
behavior in **llm-primary** — the agent *decides* to `listen`/`turn`. This plan takes it to
**substrate-primary**, where the question is the project's actual 1.0 thesis: **does the agent *learn*
to orient across sessions via NAc reward, with no LLM and no fine-tuning?**

Still `[engineering]`. This is the run that could *earn* the audio-orient line a `[behavioral]`
graduation — a clean, self-contained cross-session-learning testbed.

---

## Why this is the load-bearing next step

Everything shipped so far is llm-primary: the agent hears the sound, `listen`s, and *chooses* to turn.
That proves the *mechanism* (delivery, salience, the orient loop closing) but not the *thesis*. The
whole bio-inspired bet is **cross-session learning without fine-tuning** — and orienting is an unusually
clean testbed for it:

- The reward is intrinsic and unambiguous — turning toward the sound reduces `|azimuth|`, which relieves
  the centeredness-drive pain (`potential_diff`). No LLM judge, no hand-labeled reward.
- The correct policy is simple and measurable — "turn toward the sound" — so *learning* it (vs.
  deciding it in-context) is directly observable: direction-correctness, `|az|` reduction per session,
  and improvement across sessions.
- It rides the substrate machinery already in place: `SensorEncoder.encode_sensors` → EC cluster →
  `NAc.recommend_action(cluster_id=…)` → `cluster_reward_bias` (the exact path Exp 37/42 used for
  cross-session graduation).

If the agent orients *better across sessions* under substrate-primary (no LLM in the action path), that
is the thesis validated on a fresh behavior.

---

## Three preconditions (the Decision-4 fixes + the routing gap)

### P1 — `_normalize_value` is range-blind and aliases `-1.0` ≡ `0.0` (confirmed)

`similarity/encoder.py::_normalize_value` maps a scalar to `[0,1]` for basis interpolation, but it does
so **without the sensor's declared range**:

```python
if v < 0.0:  return (v + 1.0) * 0.5   # assumes [-1, 1]
return v                               # assumes [0, 1]
```

So a `[-1,1]` sensor's **center `0.0`** falls through the positive branch → `0.0`, *the same embedding
contribution as `-1.0` (hard left)*. For azimuth this is fatal: **"centered" (the orient success state)
is embedding-indistinguishable from "hard left"** — the negative half is aliased, and EC cannot
represent the orient state at all.

**Fix:** make normalization **range-aware** — `(v - lo) / (hi - lo)`. A `[-1,1]` azimuth then maps
`-1 → 0, 0 → 0.5, +1 → 1`; a `[0,1]` hunger maps `0 → 0`. The range must be threaded from the sensor
schema into the encode path (`encode_sensors` already has the sensor names; it needs their ranges).
Regression: pin `_normalize_value(0.0, range=[-1,1]) == 0.5` and `!= _normalize_value(-1.0, …)`.

### P2 — azimuth is bundled into the `interoception` modality (confirmed)

Substrate-primary encodes at `agent_loop.py:871`:
`sensor_encoder.encode_sensors(agent_id=…, sensors=drives)` — **default `modality="interoception"`** —
where `drives = _read_drive_states(executor)` sweeps *every* drive-spec, azimuth included. So the
exteroceptive sound direction is smeared into the same EC cluster as hunger/thermal. The orient reward
then can't cleanly key on an azimuth cluster, and `encode_sensors`' own docstring says exteroceptive
azimuth must pass `modality="audio"` to form a *separate* within-modality cluster space.

**Fix:** split the encode by modality — encode the genuine interoceptive drives under
`"interoception"` and azimuth (any exteroceptive/localization sensor) under `"audio"`, so the orient
loop gets its own cluster. This is the "de-bundle azimuth" precondition. Design choice to settle in
implementation: a declared per-sensor `modality` tag on the drive-spec vs. a hardcoded
exteroceptive-sensor set — prefer the **declared tag** (extend-by-DATA; the same body YAML already
carries the sensor).

### P3 — the audio percept never reaches the substrate-primary body (the routing gap)

Phase 1's `world_set_azimuth` runs in `agent_loop.py` **§1.16, which is gated `aut_mode !=
"substrate-primary"`** (the S1 review fix — auto_sense is an llm-primary channel). So in
substrate-primary the azimuth sensor is *never world-set from the percept* today — the whole audio
channel is inert there.

**Fix:** in the substrate-primary tick (`propose_via_substrate`, where `drives` is read at
`agent_loop.py:858`), world-set the body's azimuth sensor from `sim.current_percept` **before**
`_read_drive_states`/`encode_sensors`, so the azimuth cluster reflects the current sound. Capability-
gated + fail-soft, same as §1.16. (The salience/reflex *tiers* are llm-primary concepts — reflex
bypasses the LLM, which substrate-primary already does; in substrate-primary the sound simply *is* the
percept the substrate acts on, so no escalation gate is needed. The reflex-tier's motor-reflex is a
separate DN concern, Track 2.)

---

## The reward loop (already in place once P1–P3 land)

1. Sound arrives → azimuth world-set → encoded as an `"audio"` EC cluster (P2/P3), correctly (P1).
2. `NAc.recommend_action(cluster_id=<audio cluster>)` selects a turn — cold-start via drive-affinity,
   then increasingly via `cluster_reward_bias` as it learns.
3. The turn's `self_effect` reduces `|azimuth|` → the centeredness drive's pain drops → `potential_diff`
   relief → **NAc reward** credited to the turn, keyed on the audio cluster.
4. Over ticks/sessions the audio cluster's `cluster_reward_bias` favors the *correct* turn (toward the
   sound) → the agent orients faster and more reliably. This is the learned policy.

Note the **transition-based-drive-pain** interaction: the centeredness drive currently fires state-based
(pain every off-center tick). For the *reward* (relief on `|az|` reduction) the `potential_diff`
formulation is what matters, but the transition-based-drive-pain fix (fire on band entry) should land
first or in tandem so the reward signal is clean — it is already near-path.

---

## The experiment

**Body:** `reachy_mini` (genuine azimuth sensor + centeredness drive + calibrated turn magnitudes from
Exp 45), matching the hardware path — or `base_humanoid` (now has azimuth + turns + drive) for a purely
abstract testbed. Run both if cheap; reachy is the one that transfers to hardware.

**Stimulus:** the synthetic `default_sim_doa_reader` emitting sounds at known azimuths (a fixed schedule
of L/R/near-center events per session, `--seed`-pinned, `MAXIM_DISABLE_IMAGINATION=1` so the sound is
the only stimulus). `--aut-mode substrate-primary --embodiment bodies/reachy_mini`.

**Measure (port the orient_backbone / M3 metrics into the orchestrator telemetry):**
- **direction-correctness** — fraction of turns that reduce `|azimuth|` (turned toward, not away).
- **|az| reduction per session** — summed `potential_diff` relief; latency-to-center (ticks to within
  comfort band).
- **cross-session improvement** — the load-bearing metric: does session N+1 orient faster / more
  correctly than session N, with `aut_nac.json` `cluster_reward_bias` persisted between runs (the Exp 37
  cross-session harness pattern).

**Ablation arms** (Exp-37-style): (a) full substrate; (b) `MAXIM_NAC_REWARD_BIAS_DISABLED=1` — does the
cross-session improvement vanish without reward bias (proving the learning is substrate-carried, not an
artifact)? (c) shuffled-reward control.

**Success / graduation:** cross-session orient improvement in arm (a) that is absent in arm (b), with
direction-correctness rising toward ceiling. That earns the audio-orient line a `[behavioral]` entry in
[behavioral_graduation_candidates.md](behavioral_graduation_candidates.md) — cross-session learning of a
*new* behavior, no LLM in the action path, no fine-tuning.

---

## Sequencing

1. **P1** (`_normalize_value` range-aware) — small, high-value, unblocks any substrate azimuth use;
   ships with a unit regression. *(Also benefits every `[-1,1]` sensor, not just azimuth.)*
2. **P2** (de-bundle azimuth → `"audio"` modality, declared-tag) — the substrate cluster hygiene.
3. **P3** (world-set azimuth in the substrate-primary tick) — the routing.
4. **transition-based-drive-pain** (near-path) — clean the reward signal.
5. **M3 telemetry** — port the orient metrics into the orchestrator so the run is *measurable*.
6. **The run** — cross-session, ablation arms, on reachy_mini.

Steps 1–3 are the Decision-4 close-out; 4–6 are the experiment. Each of 1–3 is independently
useful and independently testable.

## Risks / caveats

- **P1 blast radius:** `_normalize_value` feeds every substrate sensor encode. A range-aware fix changes
  the embedding of *every* `[-1,1]` sensor (thermal, core_temperature). Re-run the substrate regression
  + any Roy/Exp arm keyed on those embeddings; the change is *correct* (fixes a latent aliasing) but it
  moves numbers — treat like an encoder change (behavioral_graduation re-validation trigger).
- **Cold-start:** substrate-primary needs the drive-affinity heuristic to seed the first correct turns
  before `cluster_reward_bias` takes over; `MAXIM_NAC_MIN_CONFIDENCE=0.0` may be needed to bypass the
  cold-start gate for the first session (as Roy-2c did).
- **Body choice couples to hardware:** reachy keeps sim + hardware on one model but its `pain_scale 1.0`
  azimuth mis-scale (the YAML TODO) should drop toward 0.2–0.3 first so the centeredness pain doesn't
  dominate genuine noxious modes.

## Related

- [cradle_orient_learning.md](cradle_orient_learning.md) — **the developmental reframing that SUPERSETS this plan.** P1–P3 here are shared prerequisites; this plan's drive-relief run becomes *Arm B* (the built-in-reward control) of the three-arm cradle study, where orienting is *taught* (caregiver + cross-modal feedback) rather than assumed innately rewarding. Prefer the cradle framing; keep this doc as the P1–P3 mechanics + the drive-only arm.
- [productive_orienting_affordance.md](../deferred/productive_orienting_affordance.md) — the llm-primary orient this extends (incl. the 2-D elevation extension path).
- [thalamus_relay_design_pass.md](thalamus_relay_design_pass.md) — Decision-4 first named these two preconditions.
- [grounded_language_acquisition.md](grounded_language_acquisition.md) — the substrate-primary sensor-encode + cluster-reward-bias path this rides.
- [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md) — where a successful run graduates.
