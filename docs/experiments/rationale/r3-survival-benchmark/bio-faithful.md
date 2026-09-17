# R3 survival benchmark — bio-faithful lens (design review of DRAFT v1, 2026-09-17)

Charter: does the design test the mechanism's REAL job, not a caricature? Does the manipulation
respect how the substrate / body / drives actually work, and does the metric measure the
mechanism's actual purpose? Read against `docs/agents/bio-memory.md` (Wire-4, NAc, drives, decay),
`docs/agents/simulation-experiments.md`, the survival body
(`src/maxim/_data/components/bodies/minecraft_player.yaml`), `src/maxim/decisions/nac.py`,
`src/maxim/runtime/agent_loop.py::_read_drive_states` / `propose_via_substrate`,
`src/maxim/embodiment/sem.py`, `src/maxim/proprioception/pain_bus.py`, the Exp 60/61 preregs and
outcomes, `docs/plans/survival_world_1_3.md`, `docs/plans/intrinsic_motivation_1_3.md`, and the
Exp 60/61 apparatus (`scripts/survival_world/water_trial.py`, `setup_world.py`).

The short version: the draft's central premise — *"the one hazard that kills is the one nothing but
the learned fear answers"* (§R3-cal, "The calibration problem") — is false on the mechanism as
shipped. Two paths answer an UNRESCUED drowning in every arm, including the naive one, and both
are the substrate doing exactly its job. Everything downstream of that premise (the naive floor
≈ 0, option (c), the survival DV, the anti-vacuity row) inherits the error.

---

## DO-NOT-BUILD

### DNB-1 — The naive agent learns the fear INSIDE its first event; the "naive floor ≈ 0" is a US-free measurement extrapolated to a US-present gauntlet

**Failure scenario.** R3-bench reports arm 2 − arm 1 ≈ 0 s survival and the number is read as
"the learned drive buys nothing," when the naive agent simply acquired the same −1.0 fear at
event 1 (one-trial acquisition under a prolonged US — the mechanism's actual job) and is
indistinguishable from arm 2 from event 2 on.

**Evidence.**
- The naive arm runs the production loop, and the production loop has Wire-4 attached BY DEFAULT:
  `pain_bus.py::build_pain_bus` subscribes `create_pain_cluster_fear_subscriber(nac)` whenever
  `nac is not None`, with the comment that a harness-level attach "would … make a fear-ablated
  agent the accidental default everywhere." Nothing in the draft detaches it for arm 1.
- Timeline, from the code and the Exp 60 apparatus record: `drive:oxygen` pain edge 5.09 s
  (`oxygen` homeostatic, set_point 20, comfort_band 6, pain_scale 0.5 →
  `sem.py::drive_pain_for_value` = min(1, (20 − oxygen − 6) × 0.5): 0.5 at 13 bubbles, **1.0 at
  ≤ 12 bubbles ≈ 6 s**). PainBus refractory 0.5 s per (entity, failure_mode)
  (`PainBus.DEFAULT_PAIN_REFRACTORY_S`). `NAc.record_cluster_fear` does
  `valence −= 0.5 × intensity`, cap −1.0, on the world cluster the loop noted THIS tick
  (`agent_loop::propose_via_substrate` calls `nac.note_active_clusters` before
  `embodiment.evaluate_failures`, so the pain sees the `is_in_water=1` cluster). Publishes at
  ≈ 5.1 s (−0.25), ≈ 5.6 s (−0.5 — reads as need 0.5, dead at the strict `> 0.5` floor), ≈ 6.1 s
  (**−1.0 → `anticipatory_threat_need` = 1.0**). Exp 60's own training record confirms the
  arithmetic: "the SATURATING publish (~6 s in)", 20 pain signals over 10 episodes = 2 per episode
  to the cap.
- From ≈ 6.1 s the naive agent's `drives["threat"]` = 1.0 → `recommend_action`: `flee` and
  `escape_water` both score 0.7 via `_DRIVE_TOOL_AFFINITIES["threat"]` ("escape") > min_confidence
  0.3; `flee` wins the name tie-break, fails fast (one tick, one negative link — Exp 60 §Outcome),
  then `escape_water` executes; measured escape 1.3–3.3 s → **head in air by ≈ 8–11 s, damage onset
  16.07 s.** The naive agent surfaces with zero damage, having paid ≈ 1–5 s of air-hunger pain once.
- Why Exp 60/61 saw 0/30 and 0/60: those windows were **US-FREE by construction** (cap = pain edge
  − 0.75 s = 4.3 s; "the probe cap is the measured pain edge … a longer window would CONDITION the
  FEAR arm"). No pain landed in the window, so nothing could be learned in it. The draft cites
  those counts as evidence that a naive agent "never surfaces" in a window that CONTAINS the pain.
  That is the extrapolation the charter exists to catch.

**What it means for the design.** After event 1, arms 1 and 2 carry identical fear on the same
cluster; a survival DV integrates over events and therefore cannot separate them except by the
first event's few seconds of pain. The learned drive's WORTH, on this mechanism, is precisely
"event 1's pain and risk, not paid" — a per-event quantity, not a survival-time quantity.

### DNB-2 — Removing the rescue admits the INNATE `health→threat` reflex into the DV, in every arm including ABLATED

**Failure scenario.** Even a fully lesioned agent (fear subscriber detached throughout) surfaces
at event 1 through the innate health reflex with ≈ 8–10 hp of damage; "survival" at event 1 is
then near-universal, and the arm 2 − arm 4 contrast measures reflex + regen mechanics (DNB-3),
not "drive vs exposure".

**Evidence.**
- `agent_loop.py::_DRIVE_CORRECTIVE_NEEDS` maps `health` → `threat`;
  `sem.py::corrective_need_intensity` (homeostatic) returns min(1, |deviation|) once
  deviation < −comfort_band, i.e. **need = 1.0 the moment health < 14 hp**.
  `_DRIVE_TOOL_AFFINITIES["threat"]` contains "escape" → `escape_water` scores 0.7 with no
  learning at all.
- Drowning damage 2 hp/s from 16.07 s (draft's own number) → health 13.x at ≈ 19–20 s → escape at
  ≈ 21–23 s → death at ≈ 26 s. A 3–5 s margin on a 1.3–3.3 s escape: the reflex wins most events.
- This reflex is exactly what Exp 60 kept OUT of its measurement on purpose. The body YAML on
  `escape_water`: "Surfacing is DELIBERATELY a LEARNED response only: `oxygen` has NO innate
  corrective need … unlike `health`. An innate oxygen→threat reflex would confound Exp 60's
  anticipatory-LEARNING claim." The US-free cap was the fence. R3's "no rescue" — presented as
  "the whole difference between a probe and a benchmark" — is also the removal of that fence.
- Bio reading: this is not a bug. A health-deficit → flight prior IS the reactive tier the
  roadmap's Phase 1b wants; but the draft says Phase 1b "is not in any arm here," and it is — via
  `health→threat`, in all four. An innate-only floor is a legitimate control, but it must be
  NAMED as the floor, not mistaken for "nothing learned, nothing carried."

**Combined consequence of DNB-1 + DNB-2.** §"The calibration problem" solves a non-problem; its
premise "a naive agent's survival on this gauntlet is not a distribution but a constant" is the
opposite of what the mechanism predicts (near-ceiling, by two independent routes). Option (c) then
calibrates the food supply around a floor that does not exist (DNB-3 on why the food axis is not
calibratable either). The draft's anti-vacuity row — "one naive agent's ZERO calls through its first
event" — would FAIL at the first dry run, which is the one good thing here: it is a red gate that
would have caught this. Re-point it to "zero calls before the first pain publish" (the US-free half),
and add "≥ 1 `escape_water` call after the saturating publish" as the naive arm's acquisition read.

---

## SHOULD-FIX

### SF-1 — "Survival time on a teleport gauntlet" is a treadmill, not the drives' job; re-cut the DV per event

The drives' real job in this architecture (bio-memory brief §Wire-4 / R2 record;
`docs/wiring/substrate-learning-channels.md`): interoceptive deficit → corrective need →
**state-conditioned action selection**, and for Wire-4 specifically **anticipatory avoidance of a
situation before the US**. Teleport-forced submersion removes the avoidance half by construction
(the agent cannot decline the water), so the fear can only express as ESCAPE — the thing Exp 60/61
already measure. Between events the substrate-primary agent does nothing: no need clears 0.5 on the
shore, explore weight is 0.0, there is no locomotor drive ("structurally near 0 v 0 — no drive fires
on the shore", Exp 60). "Survival time" therefore equals Σ(per-event escape outcomes) + the hunger
arithmetic (SF-2) + the innate reflex (DNB-2). Nothing in it is a WORLD the agent lives in.

**Mechanism-faithful DV (keeps the learned drive load-bearing at every event, per Q5):**
- per event: P(escape before the US) and latency from the teleport (Exp 60's DV, now with the
  consequence unrescued); time-below-comfort / integrated `drive_pain_for_value` for `oxygen` and
  `health` in the event; hp lost in the event; the executed tool sequence (SF-3);
- event 1 marked as the FEAR-ONLY read in every arm (after it, `escape_water` carries a positive
  link — Exp 60 caveat, 47–51 links per seed — and later events read fear + habit, SF-4);
- for arm 1 specifically, the **acquisition curve**: pain publishes until `anticipatory_threat_need`
  clears the floor, and the latency of the first escape — the number R3 can newly contribute: what
  a carried drive SAVES is exactly this;
- survival (death/censoring) becomes a reported outcome per episode, never the calibrated axis.

A bio-faithful survival WORLD under D1 would let the hazard be entered by the agent's own movement
so avoidance can express — but nothing moves this agent on the shore today (no locomotor prior;
curiosity unbuilt, SF-6), so that world is not reachable on the current substrate. Name it as the
gap rather than simulating it with a teleport schedule.

### SF-2 — Hunger is neither independent nor equal across arms; it is coupled to drowning damage through the game's regen mechanic, and the eat prior sits below the regen threshold

**Failure scenario.** R3-cal sweeps H to tens of minutes and never lands the feared arm in
[0.2, 0.8] on hunger; or lands it via `/effect hunger` — the D1 violation `setup_world.py` already
reserves for WIRING tests only ("the real learning run lets hunger drain naturally").

**Evidence.**
- Eat by prior fires only when the derived `hunger` need > 0.5: `corrective_need_intensity`
  (entropic down) = (16 − food)/10 ⇒ **food ≤ 10**. Natural regeneration needs food ≥ 18 (game
  mechanic; environment lens to re-verify on 1.20.4). Bread (+5) lifts 10 → 15: **still no regen**,
  and the prior does not fire again until ≤ 10. So an agent eating by prior lives in a
  (10, 18) band where hp lost in an event is never recovered. This decides survival for any arm
  that takes damage (the innate-reflex escapes of DNB-2), not the learned drive.
- Hunger DRAIN is dominated by regen exhaustion (healing costs food) and swimming; an idle,
  undamaged agent on the shore drains ~nothing. So fear-carrying arms (no damage, no regen) will
  not approach hunger within any H measured in minutes, while damaged arms burn food. "Hunger
  presses every arm the same" is false — it presses the un-feared arms harder, as a CONSEQUENCE
  of the water outcome. That is bio-faithful (healing costs energy), but it is the opposite of an
  independent background pressure, and it makes option (c) a measurement of bread count × drain
  in the feared arm → ceiling.
- On `difficulty=normal` (`setup_world.py`) starvation stops at 1 hp and cannot kill (Hard only).
  "An agent that starves between submersions has not survived" is not a death on this server;
  hunger's only route to survival here is the regen coupling above. (Environment lens to confirm.)

**Fix.** In v1 report hunger, never calibrate on it: log food/saturation per event and the eat
calls, and state the regen coupling. If a hunger axis is wanted later, it needs a game-native
drain the agent itself causes (locomotion) — which is the same locomotor gap as SF-1.

### SF-3 — Drive interaction can make `eat` pre-empt `escape_water` underwater; deaths would be misattributed to the fear

**Failure scenario.** A hungry arm-3 receiver (fear discounted to 0.75 → escape score 0.525)
drowns while calling `eat`, and the row is scored as "the discount's price" (arm 2 − arm 3) when the
cause is hunger's state-blind link out-arguing a discounted fear.

**Evidence.** `recommend_action` is a flat argmax with no drive arbitration: `drive_gate_enabled`
defaults False (and Exp 60's fingerprint does not pin it); when on, the gate takes the UNION of
drive-relevant tools across all intense drives (`recommend_action` NOTE), so it does not choose
between hunger and threat either. `eat` and `escape_water` each get 0.7 × their need, but `eat`
additionally carries (i) a state-BLIND `causal_pos` link (up to ~1.0) after any successful shore
eat, and (ii) the break-3 relief credit, which `tool_dispatch.py` routes to the **INTEROCEPTION**
cluster only ("drive-relief AND generic tool-success write the INTEROCEPTION cluster only") — and
the hungry interoception cluster is the same underwater, so that bias rides into the pool. `bot.consume()`
works submerged; each call costs ≈ 1.6 s plus a same-tool-cap cycle. R2's wiring doc already
warns that in a single-corrective-action world the eat causal link saturates. Ties fall to the
larger name (`max(..., key=(score, name))`): `flee` > `escape_water` > `eat`, so ties favour
escape, but any accrued eat link breaks the tie the other way.

**Fix.** Record the executed tool per tick inside every event and attribute each death to its last
executed action; report eat-in-water counts per arm; consider pinning `drive_gate_enabled` in the
fingerprint (documented as NOT an arbitration). The per-event DV (SF-1) makes the misattribution
visible instead of averaged away.

### SF-4 — Repeated events measure habit, not fear; and there is no extinction, so state it

`_cluster_fear` has no positive writer (`record_cluster_fear` only subtracts, clamp [−1, 0]), no
tick decay (Wire-4 design: "extinction is active re-learning, not a timer"), and wall decay only in
`apply_wall_clock_decay` on `load()` (7-day class; irrelevant within an episode). Repeated forced
exposure with self-rescue neither extinguishes nor sensitises beyond the cap; what it DOES do is
grow `escape_water`'s positive causal link on every success, so events ≥ 2 read fear + habit in
every arm (and, per DNB-1, in the naive arm too). Per-event latency will fall over events for
reasons that are not fear. Repeated events are meaningful only if reported per event with event 1
as the fear-only read (SF-1); otherwise the event count is a multiplier on habit. Bio note: a fear
that never habituates under repeated pain-free self-rescue overstates the drive's worth relative to
an animal's — Exp 61 §"does NOT claim" states the gap correctly; R3 should carry the same sentence.

### SF-5 — Arm 4 must declare the subscriber's state in the GAUNTLET, and what it carries

After ABLATED training the agent carries: the underwater world cluster in EC (representation);
Wire-2 `percept_valences` on (agent, entity_class, `drive:oxygen`) — situation-blind and read only by
the salience scorer / `get_percept_aversions`, NOT by `propose_via_substrate`; `reward_bias` with
its pain-clamped zero keys removed (#746); NO `cluster_fear`; NO escape link (propose-only, same as
arm 2). Two different controls hide in the draft's one row: (a) subscriber re-attached at the
gauntlet (the production default) → arm 4 = arm 1 + a pre-formed cluster, learns at event 1;
(b) subscriber detached throughout → a lesion arm that survives by the innate reflex (DNB-2) and
dies of the regen dead band (SF-2) around event 2–3. (b) is the honest "innate-only floor"; (a) is
"does a pre-formed representation speed acquisition" — both legitimate, different questions. Name
which. Note also that for a SURVIVAL claim (b) is the only real floor, and it is the reflex, not
"nothing".

### SF-6 — Drive integrity `min(health, oxygen, food)/set_point` collapses three different dynamics into the slowest one

`food` has no `set_point` (entropic: rest 20, satisfaction 16, deprivation 6), so the formula is
undefined for it as written. Dynamics differ by two orders of magnitude: `oxygen` 20 → 0 → 20 in
seconds; `health` −2 hp/s down, ≈ +0.25 hp/s up; `food` moves over minutes. `min` then tracks
whichever drive is currently most depleted and the time-integral is dominated by `food` — i.e. the
secondary DV measures the hunger arithmetic of SF-2. The game-faithful homeostatic summary already
exists in code: per-drive `sem.drive_pain_for_value` (or time outside comfort band / past
deprivation threshold), integrated per event and reported PER DRIVE. No collapse.

---

## NIT

### N-1 — "Curiosity OFF by declaration" needs a named pin, not a declaration
The intrinsic curiosity REWARD is unbuilt (`intrinsic_motivation_1_3.md`: "The curiosity *reward*
is the thing to build"; no writer in `src/`), so "OFF" is trivially true. The thing that CAN move an
otherwise-idle agent is `substrate_explore_bonus_weight` — default 0.0 in `NACConfig` and
`config_loader.py`, but resolved from `config.json::sim.substrate_explore_bonus_weight` (env
`MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT`) at `bio_stack.build_bio_stack`. `water_trial.py`'s
fingerprint already pins it at 0.0; the prereg should cite that pin (and add `drive_gate_enabled`).
`agents/bus.py::exploration_curiosity` is an LLM-path field and irrelevant to the substrate path.

### N-2 — `flee` tie-break tax
Every arm pays ≈ 1.5 s + one negative link on its first fear-driven pick (Exp 60 §Outcome). Fine,
but with an unrescued event it is part of the margin arithmetic in DNB-1/2 — record it per event.

### N-3 — Wording
"Phase 1b … is not in any arm here" — see DNB-2: the innate `health→threat` prior is in every arm.
"Tier-0 damage avoidance is UNLEARNED on this substrate" — true of the LEARNED path; the innate
health reflex is a prior that will act in R3 the moment damage lands.

---

## What a bio-faithful R3 looks like on today's substrate (for the fold)

1. Keep the water classroom, the apparatus, and the four agent kinds. Keep "no rescue" — but as
   the CONSEQUENCE that makes the per-event outcome real, not as the thing that makes survival the DV.
2. Primary DVs per event: escape-before-US, latency, integrated oxygen/health pain, hp lost,
   executed-tool sequence. Event 1 = fear-only read. Arm 1 additionally reports its acquisition
   curve (publishes-to-threshold, first-escape latency). Contrasts are per-event differences with
   intervals; survival/censoring is reported, never calibrated.
3. Floor arm = innate-only (subscriber detached at the gauntlet); say so. If a "fresh, production
   default" arm is also wanted, it is the ACQUISITION arm, not the floor.
4. Hunger: report, don't calibrate. Log food/saturation and eat calls; state the regen coupling and
   the (10, 18) dead band.
5. Drop `min(...)/set_point`; report per-drive time-in-pain.
6. State the no-extinction limit as Exp 61 does; keep the event count small (the information is in
   event 1 and the acquisition curve; events ≥ 2 mostly add habit).
7. The Goldilocks calibration then has a real axis: the per-event DV is bounded [0, 1] per event
   with a floor that is the innate-only arm (escape at ≈ 21–23 s with ≈ 8–10 hp lost) and a
   ceiling that is "escape before the pain edge with 0 hp lost" — arms CAN sit between them.

---

## What I verified

- `docs/experiments/DESIGN_REVIEW.md` (charter); `docs/agents/bio-memory.md` §Wire-4 / drives /
  decay; `docs/agents/simulation-experiments.md`.
- `src/maxim/proprioception/pain_bus.py::build_pain_bus` — the fear subscriber is auto-wired whenever
  `nac is not None` (naive arm has Wire-4 live); `create_pain_cluster_fear_subscriber` intensity
  threshold 0.3, world-cluster keyed; `PainBus` refractory 0.5 s per (entity, failure_mode).
- `src/maxim/decisions/nac.py`: `NACConfig` (`cluster_fear_alpha` 0.5, `max_cluster_fear` 1.0,
  θ 0.5, allowlist {drive:health, drive:oxygen}, `drive_gate_enabled` False,
  `substrate_explore_bonus_weight` 0.0); `record_cluster_fear` (subtract-only, clamp [−1, 0]);
  `cluster_fear` / `anticipatory_threat_need` (deepest fear, ≥ θ else 0); `recommend_action`
  (flat argmax; causal_pos / reward_bias / cluster_bias summed over active clusters; strict
  `drive_value <= 0.5: continue` floor; affinity 0.7 × need; tie → larger name; drive gate = union);
  `_DRIVE_TOOL_AFFINITIES["threat"]` contains "escape"; `apply_wall_clock_decay` is the ONLY
  decay path for `_cluster_fear` (7-day class, on `load()` only); no per-tick decay caller exists.
- Grep across `src/maxim/` for `cluster_fear` writers: only `record_cluster_fear` (pain) and the
  hivemind ingest/merge (Exp 61 transport, min-fold + 0.75 discount). No positive writer.
- `src/maxim/runtime/agent_loop.py::_DRIVE_CORRECTIVE_NEEDS` (`health` → `threat`, `food` →
  `hunger`; no `oxygen` entry), `_read_drive_states`, and `propose_via_substrate` (note clusters →
  `evaluate_failures` → re-read drives → `threat = max(innate, fear)` → `recommend_action`).
- `src/maxim/embodiment/sem.py::drive_pain_for_value` and `corrective_need_intensity` (homeostatic:
  min(1, |dev|) past the band → 1.0 at health < 14 / oxygen < 14; entropic down: (16 − food)/10).
- `src/maxim/runtime/tool_dispatch.py` seam routing: relief and tool-success credit write the
  INTEROCEPTION cluster only.
- `src/maxim/_data/components/bodies/minecraft_player.yaml`: `health` / `oxygen` homeostatic
  (set_point 20, band 6, scale 0.5), `food` entropic (satisfaction 16, deprivation 6, no set_point),
  `escape_water` comment on the deliberate absence of an innate oxygen reflex.
- `scripts/survival_world/water_trial.py` (GAMERULES frozen: doMobSpawning/doDaylightCycle/
  doWeatherCycle false, doImmediateRespawn/keepInventory true; fingerprint pins explore weight;
  `rescue()` heals + saturates via `/effect`; `deaths()` reads the scoreboard objective);
  `setup_world.py` (`difficulty=normal`, bread × 64 seeded, `--induce-hunger` reserved for wiring
  tests); `scripts/minecraft_bridge/index.js` (`eat` = `bot.consume()`, no water guard;
  `is_in_water` head-block).
- Exp 60 prereg §Design (iii) + §Outcome (US-free cap 4.3 s; training = 2 pain publishes to the
  cap per episode; naive 0/30 in US-free windows; positive escape links 47–51 after the first
  success; `flee` tie-break); Exp 61 prereg §Arms, §Receiver lifecycle, §Does NOT claim
  (no extinction, no positive writer, wall decay only).
- `docs/plans/survival_world_1_3.md` (D1; the R2 breaks; Tier ladder), `docs/plans/roadmap_1_3.md`
  §Phase 1b / 3 / 6, `docs/plans/intrinsic_motivation_1_3.md` (reward unbuilt; guardrail),
  `docs/plans/minecraft_benchmark.md` §R3 (original scope: DV chosen at calibration;
  drive-integrity or time-to-death), `docs/wiring/substrate-learning-channels.md` (eat link
  saturation trap).
- Not verified live (environment lens): the 1.20.4 regen threshold (food ≥ 18), regen exhaustion
  cost, bread's +5, drowning 2 hp/s, starvation floor on Normal. The DNB-1 timeline uses the
  Exp 60 apparatus record's measured pain edge / damage onset and the shipped code constants.

## Verdict

**DO-NOT-BUILD (as drafted).** The calibration section rests on a premise the mechanism refutes
(DNB-1: the naive agent acquires the fear inside event 1 with Wire-4 live by default; DNB-2: the
innate `health→threat` reflex answers an unrescued drowning in every arm), and the survival DV
does not measure the drives' job (SF-1). Folding to a per-event DV with an innate-only floor and
hunger reported-not-calibrated (the list above) turns this into FIX-THEN-BUILD; the apparatus,
arms and protocols reused from Exp 60/61 are sound.
