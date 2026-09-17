# R3 (DRAFT v2.1, 2026-09-17, four-lens design review FOLDED, owner decisions D1–D5 TAKEN) — the lethal-window benchmark: what a carried survival drive buys at the one moment it matters, measured on a depth-calibrated, frozen gauntlet

> **STATUS: DRAFT v2 — the four-lens design review ran on v1 (2026-09-17; confounding, bio-faithful,
> wiring and environment, reports preserved verbatim under
> `docs/experiments/rationale/r3-survival-benchmark/`). All four returned DO-NOT-BUILD as drafted,
> converging independently on the same two premises being false; v2 folds every DO-NOT-BUILD and
> SHOULD-FIX below and records what was dismissed and why. Nothing is built; no live measurement
> has been taken. This is the 1.3 Phase-3 rung (`docs/plans/roadmap_1_3.md` §Phase 3), an
> INSTRUMENT plus a FROZEN BASELINE, never a graduated claim (`minecraft_benchmark.md` D2). The
> owner decisions D1–D5 were TAKEN 2026-09-17 (recorded at the end, with the one design idea
> considered and held out); the build order starts with a MEASURED pilot,
> because two of v1's load-bearing "facts" were extrapolations from windows built to prevent the
> very thing they were cited for.**

## What v1 got wrong, in one paragraph (the reason this document looks different)

v1 assumed a naive agent dies at its first unrescued submersion, so a naive floor of ≈ 0 needed a
calibrating axis (hunger) to give the learned arms a middle. Both halves are false on this world
and this agent. **The naive agent does not drown.** Two mechanisms it carries by default answer
the water: the Wire-4 fear subscriber is auto-wired on every NAc-bearing pain bus (Exp 60 had to
DETACH it for the ablated arm), so a fresh agent books −0.75 on its own water cluster by the
second oxygen-pain publish ≈ 6 s in and escapes ≈ 9–10 s in, before damage at 16 s; and with the
subscriber detached, the innate `health → threat` corrective need (built #683, on by default, the
agent not the world) reaches 1.0 the moment drowning damage takes health below 14 and scores the
escape with no learning at all, ≈ 21 s in, 4–10 s before death. Exp 60's and 61's "0 naive calls
in 90/90 windows" were measured in windows capped at 4.34 s — the pain edge minus a margin —
precisely so that neither path could fire. **And hunger is not a pressure here:** exhaustion at
rest is zero, the shore is a sealed five-block strip, starvation cannot kill at Normal difficulty,
and a hungry reading rotates the world vector the fear is keyed on. Every arm in v1 would have sat
at the ceiling, and the reported "what the drive buys" would have been the schedule's arithmetic.

What survives: the classroom, the apparatus, the agent kinds, the protocols, and the purpose. What
changes: the unit of measurement is ONE lethal-window event per fresh agent (the only unit that
carries arm information — after the first escape the state-blind positive `escape_water` link owns
selection in every arm, measured at 14 dry-land escapes per 10 s in Exp 60), the DV is the COST of
that event, the arms are declared by their learning channel, and the difficulty axis is placement
DEPTH, calibrated on the innate-only floor.

## Purpose — and what this is NOT

**Purpose.** Exp 60 showed a learned drive acts; Exp 61 showed it travels. R3 measures what a
carried drive is WORTH at the one moment the world makes it matter: an agent is in the water, no
one will pull it out, and the routes to air fire in a known order — a carried fear at ≈ 1–3 s, an
in-situ Wire-4 acquisition at ≈ 6–10 s, the innate health reflex at ≈ 21 s — against fixed
deadlines: damage at ≈ 16 s, death at ≈ 25–33 s. Depth stretches the ascent and titrates those
routes out in that order. The number R3 reports is how much air, health and depth each channel
buys, with intervals, on a gauntlet frozen so the number can move.

**Not a claim.** R3 graduates nothing. Contrasts are named after ARMS with the predicted carrying
mechanism stated beside each, never after causes; "the drive buys X" is a reported measurement
with its interval and its mechanism attribution, not a sentence that licenses "agents learn to
survive". A saturated cell is a failed calibration and re-calibrates; it is never a NULL on the
drives.

**Not:** the survival reflex of Phase 1b as a NEW mechanism (the innate `health → threat` prior IS
in every arm and is named as such); R4; intrinsic motivation (unbuilt; the only exploration lever,
`substrate_explore_bonus_weight`, is pinned at 0.0 by the apparatus fingerprint, cited); a survival
WORLD the agent enters by its own movement (not reachable on today's substrate: no locomotor prior
— recorded as the gap, `survival_world_1_3.md`).

## Front-gate scope pressure

Nothing new in `src/`. The gauntlet is the Exp 60 water classroom and its apparatus
(`water_trial.WaterTrial`), the learned arms are Exp 60's training and Exp 61's export → ingest
protocols verbatim, and the analysis is Exp 61's shape. R3 adds one harness method — a
`lethal_event` beside `placement`, which submerges the live agent and does NOT rescue it — plus
depth as a parameter of the placement, a death detector, and the gauntlet file. Every one of these
is a harness change; none is a mechanism.

## The gauntlet

One world, the Exp 60 water classroom, one server and bridge (Paper 1.20.4, bridge at
`--state_interval_ms=100`, flee anchor frozen at the SHORE — v1's default was world spawn 69 blocks
away, a multi-second no-path inside a blocking executor call the moment a threat need stands on
the shore), one encoder, fresh persistence per agent, one agent per process.

**The event.** After its protocol (below) and the apparatus checks, the agent is teleported once
into the water column at the frozen DEPTH with the loop live, and left there. The event ends at
the first of: the head clears the water by the agent's own act (bridge truth, `is_in_water` at eye
height 0 with position above the water line); death (below); or a hard cap of 45 s, which is a
named refusal (an agent alive underwater at 45 s is an instrument fault — death is ≈ 25–33 s).
There is no second event. The agent's persistence is closed and kept as the row's evidence.

**Death detection (v1 had none).** The `deaths` scoreboard objective is set to 0 at event start and
polled at every 4 Hz sample over local RCON; a parse failure is an `InstrumentError`, never 0;
death is corroborated by the respawn discontinuity (position = shore spawnpoint, health 20,
`is_in_water` 0 within one sample). `doImmediateRespawn` makes health 0 a one-tick state, so the
health series is recorded but never the detector.

**Hunger and regeneration are CONSTANT, declared, and recorded — never an axis.** Start state =
the Exp 60/61 rescue settle (`WaterTrial.heal`: instant health + saturation; sensed food ≥ 16,
saturation at the bridge clamp 10) — a harness injection of interoceptive state, declared as such,
kept for parity with every fear read Exp 60/61 ever took. `naturalRegeneration` is frozen at its
default `true` and VERIFIED in the gamerule roster (it moves the death edge by 2–8 s);
`doDrowningDamage true` verified (a false value makes drowning non-lethal with no other symptom);
`difficulty normal` pinned by an RCON read in the fingerprint (starvation cannot kill here — stated,
not relied on); `doInsomnia false` as a belt for a multi-hour campaign; `weather clear` and `time
set day` at setup. Food, saturation, health and oxygen are stamped on every sample.

**The settle guard** (`is_raining == 0`, `nearest_player_dist == 64`, `hostile_count == 0`) runs at
the event teleport and on EVERY sample, not only inside `rescue()` (which R3 never calls).

## Arms — declared by learning channel, with the Wire-4 subscriber's state in the gauntlet named

| Arm | Agent, before the event | Subscriber during the event | Predicted route to air (mechanism, onset) |
|---|---|---|---|
| **A innate-only** | fresh persistence, no training | DETACHED (`_detach_fear_subscriber`, as Exp 60's ablated) | innate `health → threat` at health < 14, ≈ 19–21 s; the FLOOR |
| **B in-situ learner** | fresh persistence, no training | ATTACHED (production default) | Wire-4 acquires −0.75 on its own cluster ≈ 6 s, escape ≈ 9–10 s; the acquisition curve is this arm's own number |
| **C self-learned** | Exp 60 FEAR training (10 usable episodes, propose-only, rescued), NO post probe (zero escape links at the boundary, asserted) | ATTACHED | carried fear −1.0, escape ≈ 1.3–3.3 s |
| **D shared** | fresh receiver that ingested a C-protocol donor's export (Exp 61 lifecycle: real CLI export/ingest, reboot, loop-OFF gate reads −0.75) | ATTACHED | carried fear −0.75; the argmax makes D ≡ C structurally (both clear θ, same affinity, same tie-break) — reported side by side, NEVER contrasted as "the discount's price" |
| **E exposed-ablated** | Exp 60 ABLATED training (same exposure, subscriber detached, propose-only) | DETACHED throughout | A plus a pre-formed water cluster and Wire-2 valence: the exposure-without-drive control; predicted ≡ A |

One fresh agent per row; C-protocol donors for D are one per receiver, no reuse. Every learned
arm passes its own protocol's sanity before the event (Exp 60's G2 for C/E; Exp 61's staged
sanity, ingest gate and loop-OFF representation gate for D); every arm asserts
`positive_escape_links() == 0` and `reward_bias == {}` at the gauntlet boundary; a failure is a
named REFUSAL, never a row. Curiosity is unbuilt; `substrate_explore_bonus_weight` 0.0 and
`drive_gate_enabled` false are in the fingerprint and the fingerprint is verified per row.

## Dependent measures

**Primary (per event, one per agent):**
- `escaped_before_damage` — surfaced by its own act before the damage onset (binary; Exp 60/61's
  behavioural DV without the rescue);
- `t_surface` — latency from the teleport to the head clearing the water (censored at death or
  the cap, reported with the censoring class).

**Secondary (reported, never gated):** `survived` (binary); health lost; oxygen-pain seconds (the
integral of `drive_pain_for_value` for oxygen over the event, from the telemetry the apparatus
already records) and health-pain seconds, per drive, never `min(...)`; the executed-tool sequence
with timestamps (`flee` first is the predicted tie-break; a `flee` no-path is a named refusal);
decision provenance per proposal (`RecommendCapture` + `decision_decisive`, Exp 61 step 5) and the
drive-decisive fraction per arm; for arm B, the acquisition curve (oxygen-pain publishes to the
first proposal, `cluster_fear_dump` after the event); for every arm, `cluster_fear_dump` and
`positive_escape_links` AFTER the event (the snowball, recorded).

**Contrasts, named by arm, mechanism beside each, intervals always:** C − A (carried fear vs the
innate floor), B − A (in-situ acquisition vs the floor), C − B (what carrying it saves over
learning it there), C − E (drive vs exposure), D reported beside C. Fisher exact one-sided on the
binaries (Exp 60/61's statistic); latency by Mann–Whitney with a bootstrap of the median where
< 50 % censored, else the censoring fraction is the number.

**Anti-vacuity rows, required (the campaign is INCOMPLETE without them):** one arm-A and one arm-B
first event recorded tick by tick (the pilot's two seeds, re-run inside the campaign at the frozen
cell); one arm-C row whose fear reads at the cap before the event AND whose executed escape is
drive-decisive (the object read paired with the behaviour, never the object alone).

## The Goldilocks calibration — depth, on the floor arm

**Axis.** Placement depth `d` (blocks of water above the placement, game-native: the ascent takes
longer, `escape_water` holds jump for at most 8 s and reports "still submerged (capped)" as a
SUCCESS — recorded per event as `detail`, a capped escape that dies is a death). The routes'
onsets are fixed by the mechanism; the deadlines are fixed by the game; depth moves the ascent
between them.

**R3-cal.** Arm A only, `n_cal` = 12 per cell, pre-declared sweep from the Exp 60 depth (the
placement the apparatus already uses) downward in the column in 2-block steps as the classroom
allows (the builder extends the column; `setup_world` records it in the anchor). The calibrated
quantity for the floor arm is **`survived`**, not `escaped_before_damage`: arm A's route fires
only AFTER damage, so it can never escape before it, and its survival is the outcome depth moves.
**Accept** the first cell where arm A's survival fraction lies in **[0.20, 0.80]**
with its Wilson 95 % inside [0.05, 0.95]. At that depth the floor arm is mid-range on the
lethal outcome, arm B is predicted to survive with damage or not at all depending on its
≈ 9–10 s escape plus the ascent, and arms C/D are predicted to escape before damage — a cell
where the ORDER of routes is visible in the outcomes. If no cell lands, R3 stops with the cause
named (the column cannot be made deep enough, or the 8 s cap floors the floor) and the design
reopens; that is an instrument finding.

**Before any live cell (corollary 3):** an offline replay of the cosine between the Exp 60
water node (encoded at the rescue settle) and the submerged reading at each candidate depth —
`y_altitude` is a world-modality sensor and a deeper reading may complete into a NEW node. Then
every learned arm trains at the FROZEN depth (exact-key cache), and its representation gate is
read at that depth before the event.

## The frozen gauntlet file (D5)

`docs/experiments/data/r3_gauntlet.json`, written by R3-cal, read by R3-bench, refused on drift
like Exp 61's `FROZEN`: the accepted depth and its calibration rows' sha256, `n_cal`, the Wilson
interval; the anchor record verbatim (incl. `t_damage_onset_min_s`, `deaths_objective`); the
fingerprint (incl. `substrate_explore_bonus_weight`, `drive_gate_enabled`, difficulty); the verified
gamerule roster; the flee anchor; the start state; the apparatus record's timestamp and code
hash; the calibration code hash (R3-bench refuses unless `git merge-base --is-ancestor` puts it
on main); the per-arm subscriber declaration; the offline replay's result. Bench rows at a hash
other than the gauntlet's read INCOMPLETE.

## Refusals and stop rules

Every Exp 60/61 apparatus refusal (fingerprint, cadence ≤ 0.15 s at every sample, liveness,
actuation through the bridge with zero executor calls, clusters distinct, gamerules incl. the two
added, settle guard, one code hash, clean tree, `assert_repo_interpreter`), plus R3's own: a
`flee` preflight on the shore must return "fled to anchor" ≤ 0.5 s (bridge-side, never the
executor); no executor call in flight at the teleport; a death with `oxygen > 0` at the last
sample (a hostile leaked, fall damage) is refused with the cause; a capped-escape death is a
death; `deaths` parse failure is an instrument error; a learned-arm agent that fails its sanity or
carries a positive escape link at the boundary is refused before the event; an event alive
underwater at 45 s is refused.

## What was dismissed, and why (the review's record)

- **Survival time on a repeated-teleport schedule (v1's DV):** after event 1 the state-blind
  positive `escape_water` link owns selection in every arm (Exp 60: 14 dry-land escape successes
  per 10 s roam, 47–51 links); events 2..k are identical across arms; "buys X seconds" = H − 26 s,
  the experimenter's choice. Rejected by all four lenses.
- **Hunger as the calibrating axis (v1 D3 option c):** zero exhaustion at rest on a sealed shore;
  starvation non-lethal at Normal; the `eat` prior fires only below food 11; a hungry reading
  rotates the cluster key; the food supply would be a step function. Not tuning-to-pass — not
  calibratable. Options (a) (a naive agent makes NO actions, so there is no random walk to
  titrate) and (b) (saturates at the ceiling in every arm) rejected for stronger reasons than v1
  gave.
- **`min(health, oxygen, food)/set_point` as drive integrity:** three dynamics collapsed into the
  slowest; `food` has no set-point. Replaced by per-drive pain-seconds.
- **"Naive = nothing learned, nothing carried":** the innate health reflex and the auto-wired
  subscriber are the agent. Replaced by arms A and B, declared.
- **Arm C − D as "the discount's price":** structurally zero on a deterministic argmax; a timing
  failure would have been read as a mechanism result. Reported side by side only.
- **Long loop-live episodes (H up to 600 s):** never proven live beyond 15 s; memory and link
  growth measured per build; the one-event unit removes the need.

## Owner decisions (TAKEN 2026-09-17)

- **D1 — the five arms as tabled, n = 12 each. TAKEN.** (60 agents + 24 trainings for C/E + 12
  donors for D ≈ 36 trainings × ≈ 2.5–3 min + 60 events × ≤ 45 s + apparatus ≈ 2.5–3.5 h; plus
  R3-cal at 12 per cell × the cells walked, ≈ 10 min per cell.) E stays: it is the only control
  that separates "exposure" from "drive" against the floor, and it is cheap.
- **D2 — primary DVs `escaped_before_damage` + `t_surface`; survival secondary. TAKEN as written.**
- **D3 — depth as the axis, calibrated on arm A's survival. TAKEN.** The pilot measures whether
  depth actually moves arm A before any sweep is declared final. **Considered and held out of R3
  (owner idea, 2026-09-17): an internal PRESSURE signal** — a body (SEM) component that transforms
  depth into an interoceptive homeostatic/entropic signal. The categorization is settled in its
  favour: altitude is game-exposed, and a transform of an exposed quantity into an internal one is
  the same class as the eye-height `is_in_water` flag, NOT the invented world fact D1 forbids. It
  is held out of R3 on the CONFOUND, not on D1: as a sensor it re-keys every cluster the trained
  fear sits on (a new channel in the summed vector) and gives R3 nothing depth-through-oxygen does
  not already give (oxygen is the game's pressure analog — the deeper, the longer the air-hunger
  before the head clears — and it is the drive every arm already senses); as a DRIVE it would be a
  route to air that fires on placement in every arm, collapsing the benchmark to the ceiling for
  the same structural reason v1 did. It is a legitimate design candidate for its own front-gate
  and four-lens review after R3-bench (perception line / Phase 1b-adjacent), never inside R3.
- **D4 — start state = the Exp 60/61 heal settle. TAKEN** (declared injection, parity with every
  prior fear read; the game-native respawn state — food 20, saturation 5 — would change the
  cluster key relative to training).
- **D5 — `naturalRegeneration true` (the game default) frozen and verified for the campaign.
  TAKEN.** Running both settings as two gauntlets was considered (it would show how much of the
  floor arm's survival is regeneration) and rejected as doubling the campaign and splitting the
  frozen file; instead the PILOT records the death edge under BOTH settings (one deliberate
  drowning each way), so the off-regeneration edge is a measured number in the gauntlet file, and
  an off-regeneration baseline is one gamerule flip and a re-run under the ledger's existing
  gamerule trigger.

## Build order

1. **PILOT, before anything is built beyond a throwaway script (the review's first ask):** on
   big-mac-mini, two seeds at the Exp 60 depth, no rescue, loop live, window telemetry — one arm-A
   agent (subscriber detached) and one arm-B agent (attached) — plus one deliberate drowning with
   the loop live (the death/respawn seam has never been exercised: ticks continue, snapshot resets,
   `deaths` +1 exactly, cadence resumes), the regen timeline under the frozen gamerules, exhaustion
   at rest for 60 s, the `flee` anchor preflight, and the drowning/ascent timing at two deeper
   placements. Recorded as a diagnostic; every number above marked "predicted" becomes measured or
   the design reopens. A four-lens re-run on the re-cut design is the review's own request; it runs
   on v3, after the pilot has replaced predictions with measurements.
2. Offline cosine replay of the submerged reading vs depth (corollary 3).
3. Harness PR: `WaterTrial.lethal_event(depth)` beside `placement` (no fork), the death detector,
   the settle guard per sample, the two added gamerules, the flee preflight, the boundary assertions;
   `r3_run.py cal` / `bench` / `report` with pure halves unit-tested; two-lens code review.
4. R3-cal live → gauntlet file → freeze PR (v3 → FROZEN).
5. R3-bench campaign at one hash → merge-commit data PR → §Outcome: the frozen baseline and the
   reported contrasts, each with its mechanism beside it, and what they do not say.
