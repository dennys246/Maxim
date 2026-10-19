# E1 v3 four-lens design review: the environment lens

Reviewed: e1_conflict_prereg.md DRAFT v1 (2026-10-18)

The question this lens answers: does the world afford E1 v3 game-natively (D1: no synthetic sensor, no synthetic
reward)? Are the states and acts it needs reachable and measurable? Does the bridge and world behave as the design and
its toy (`learner_sim.py`) assume?

**Verdict: buildable after two DO-NOT-BUILD folds**, both of them text plus harness rules, with no world change:

1. The pain terminal, and Wire 4's write, are unsound while `eat` blocks the loop.
2. The hunger setter is unspecified in exactly the parameters that decide whether the campaign is feasible.

A second structural fact changes the toy's numbers but not its predicted verdict: on measured timings, two breads plus
an escape do not fit before the saturating pain.

## What I verified

- **Committed (the apparatus records).** Oxygen drains about 1.33 bubbles/s from the teleport.
  - Source: `exp60_water_apparatus*.json` and `b1_exp60_floor_apparatus_2026-10-14.json`, w2 series.
  - Oxygen reads 13 at 5.0–5.3 s (`t_pain_edge`, the first publish, intensity 0.5) and 12 at 5.8–6.4 s (the saturating
    publish, intensity 1.0). Damage starts at 16.1–16.4 s.
  - The toy's "20 → 12 in 6 s" matches. The toy omits the earlier 0.5 publish.
- **Committed (the B1 campaign JSONL, `b1_exp60_floor_48ce40118c20.jsonl`, 40 arm-seeds).**
  - `escape_water` calls: median 1.82 s (p90 1.91 s). The head clears 1.46–1.67 s into the call (`t_surface`).
    Training episodes surface 2.0–2.6 s after the teleport.
  - The first in-water select comes about 0.96–0.98 s after the teleport (`t_first_water` 0.36–0.40 s, plus the tick).
  - Every failing or read tool returns in 1–70 ms. The "0.5 s" in the toy is the loop tick, not the call.
  - `eat` underwater at food 20 fails in 2 ms with `Food is full` (40 calls). So the bridge reaches `bot.consume()`
    submerged. **No successful underwater eat has ever been timed** (roadmap pilot row 5 is still owed).
  - Episode cycle with rescue: about 5.0–5.5 s for a surfaced episode, about 6.6 s for a 4.26 s cap window. A
    propose-only pain episode to the saturating pain is about 9 s (FEAR's training block).
  - Arm-seed wall time: 115–200 s. Whole campaign: 1.64 h.
  - Bread held: 128 (not 64, and not 2). After each rescue the true state is `foodLevel` 20,
    `foodSaturationLevel` 20, exhaustion about 1.5.
- **Committed (the code).**
  - `WaterTrial.heal()` gives `saturation 1 10`, so every rescue restores food 20 and true saturation 20. `rescue()`
    then waits on the body's `saturation ≥ 10`, which is a world-written sensor.
  - `WaterTrial.sample()` carries no `food`.
  - The bridge `eat` does: equip, then `bot.consume()` (mineflayer 4.38.0 in the lockfile: throws `Food is full` at
    food 20, has a 2500 ms timeout, and resolves on entity status 9), then polls up to 1.5 s for `bot.food` to change.
  - The bridge `stop` clears only the pathfinder goal. It does not cancel item use.
  - Bridge actions are keyed by affordance name, not by body name.
- **Committed (pain timing).** On substrate-primary, drive pain is published only from `propose_via_substrate`
  (`evaluate_failures()` at proposal time; `loop_gates.tick_embodiment_drift` is a no-op there). A standing breach
  re-publishes about every 2 Hz.
- **Committed (food pain).** The `food` drive is entropic with deprivation 6, so `body.py` publishes a
  `drive:food:deprived` pain (intensity 0.3, the default) at food ≤ 6. Wire 4's allowlist is `drive:health` and
  `drive:oxygen`, so Wire 4 ignores it.
- **Measured here (offline; scratch YAMLs through `ComponentRegistry`).**
  - `extends:` cannot drop a sensor. `deep_merge` has no delete; `saturation: null` raises `AttributeError`; and
    `modality: interoception` is refused at parse time.
  - The only working form is `saturation: {modality: null}`. It leaves a 17th sensor that no one writes: it holds its
    initial value 10, and its `read_*_saturation` tool reads 10 forever. With it the world roster is 16 sensors, the
    affordances are unchanged, and the entity name `minecraft_e1` gives `minecraft_e1_*` tools.
- **Measured here (the toy with live timings).** A scratch copy of `learner_sim.py` with:
  - the first select at 1.0 s;
  - a 0.58 s tick for non-blocking picks;
  - escape terminal at 1.5 s into the call;
  - eat wall time 1.75 s, then the next 0.5 s tick;
  - the 0.5 publish at 5.2 s.

  The qualitative result holds: Q_EXPLORE takes the pain on 20–56 % of probes, and MAXIM and TWIN show 0 / 0. The
  floors' WANT falls: Q_EXPLORE 0.41–0.87 → 0.24–0.38, Q_PRIOR 0.45–0.86 → 0.24–0.47, RANDOM 0.32 → 0.14–0.23.
  Q_PRIOR's P(pain) falls to 0.00–0.10.
- **Inferred (vanilla 1.20.4 mechanics; wiki-confirmed where cited).**
  - The Hunger effect adds 0.005 × level exhaustion per tick. Every 4 exhaustion removes one saturation point, or one
    food point once saturation is 0 ([Hunger effect](https://minecraft.wiki/w/Hunger_(status_effect)),
    [Starving](https://minecraft.wiki/w/Starving), [Hunger](https://breezewiki.discard.no/minecraft/wiki/Hunger)).
    So any game-native hungry state (food < 20 after a drain) has saturation exactly 0.
  - Hunger does not drain food on Peaceful, and Peaceful refills food by itself.
  - Bread gives +5 food and +6 saturation (capped at the food level). An eat takes 32 ticks (1.6 s).
  - Eating works with the head underwater, and a teleport does not cancel it.
  - Natural regeneration costs food only when health < 20. E1 never damages (terminal at about 6 s, damage at about
    16 s), so there is no regeneration coupling. Swim exhaustion (0.01/m) is negligible.
  - Natural drain was measured at zero over 60 s in the frozen world (committed in
    `grounded_word_binding_demo/environment.md`). Pilot row 7, "holds ≥ 60 s with the loop live", is not yet measured.

## DO-NOT-BUILD

### DNB-1. The pain terminal, and Wire 4's write, depend on whether `eat` is blocking the loop

**Claim attacked.** Protocol, Episode: "Terminals by bridge truth on the harness clock: surfaced, the saturating
oxygen pain, or a cap". The toy also assumes "the pain lands inside the blocking call" and books it at once.

**Evidence.**
- The saturating pain is a bus publish, not bridge truth.
- On substrate-primary it is emitted only at proposal time (`substrate_proposal.py`, the `evaluate_failures()` tick).
  While a 1.6–1.9 s `eat` is in flight no tick runs, so no drive pain publishes.
- E1 is the first rung where MAXIM trains through EXECUTED blocking acts (Exp 60 and B1's FEAR trained propose-only
  at 4 Hz). Its decisive moment, eating as oxygen crosses 13 → 12, is exactly where the loop is blocked.
- Two harness choices both fail:
  - If the harness terminates on sensed oxygen ≤ 12 (B1's rule), it rescues mid-eat. The loop's next tick then reads
    shore oxygen climbing, so the 1.0 publish may never happen. The floors still book their −1 terminal; MAXIM gets no
    Wire 4 write. The US is asymmetric across arms.
  - If it terminates on the publish, the terminal fires up to one eat late.
- The bread in flight finishes on the shore after the teleport: `stop` does not cancel item use, and a teleport does
  not cancel eating. Its food delta then lands in the rescue's satiation window.

**Fix (text for the prereg).**
> *Terminal.* The pain terminal is the first `drive:oxygen` publish of intensity ≥ 1.0 seen on the pain bus during the
> episode. It is the same for every arm, and the same signal Wire 4 writes from. The harness waits through a call in
> flight, because oxygen ≥ 9 at the latest publish is far from damage at about 16 s. The cap is the measured damage
> onset − 3 s. Every row records sensed oxygen and time at the publish, and the call in flight (tool, start, return).
> *Credit.* A bread counts toward WANT, and toward the floors' +1, only if its `eat` call RETURNED ok before the
> terminal. An eat in flight at the terminal is recorded, not counted. It is booked as the toy books it: failure,
> terminal pain.
> *Pain record.* The first 0.5-intensity publish (oxygen 13, about 5.0–5.3 s) is part of every pain episode. It is
> delivered to Wire 4 and charged in the floors' reward (B1 rule 2), and the toy is re-run with it.

### DNB-2. The hunger setter: unspecified amplifier, it runs after every rescue, and the rescue's satiation check goes blind on the variant

**Claim attacked.** "Hunger: set game-natively per episode (`effect give hunger` to drain …), read back from the
bridge, and banded."

**Evidence.**
- Every rescue (`WaterTrial.heal`) restores food 20 and TRUE saturation 20. So every episode's setter must drain
  26–38 points: 20 saturation, then food 20 → the band.
- That cost is set entirely by the amplifier:
  - The only hunger setting in the repo is `setup_world prepare --induce-hunger`: `hunger 40 4`, level 5, one point per
    8 s. At that rate the setter takes 3.5–5 min per episode, about 70 h for the campaign.
  - At amplifier 255 (level 256) it is one point per 3.1 ticks (about 6.4 points/s): 4–6 s.
- The landing has its own traps:
  - A point drops every 150 ms and an RCON `effect clear` applies about one tick late, so a single-stage drain
    overshoots by 1–2 points.
  - The heal's own `saturation` effect must be cleared first, or it fights the drain.
  - Band [1,3] can overshoot to 0. On Normal difficulty that is starvation damage, a `drive:health` pain, and a refused
    arm-seed.
  - On Peaceful the drain does nothing and food refills by itself. B1's and Exp 60's gamerule check does not read
    difficulty.
- On the variant body (OE2, `modality: null`) the bridge never writes `saturation`. `rescue()`'s settle on
  `saturation ≥ 10` then passes vacuously, so the satiation it verifies is no longer verified. The bridge's `food` is
  live, but the bridge clamps saturation and the health packet is sent only on food or zero-crossing changes, so true
  state must come over RCON.
- Train band [11,13] has its floor exactly on B1's hunger-key edge (11). A one-point slip moves Q_RAW and Q_PRIOR to
  the [6,11) key.

**Fix (text for the prereg).**
> *Setter.* It runs on the shore after the rescue settles, with the loop live and disarmed. Steps:
> 1. `effect clear <bot>`.
> 2. `effect give <bot> minecraft:hunger 30 255 true` until RCON `foodLevel` ≤ target + 2.
> 3. `effect clear`, then `effect give … hunger 30 19 true` (level 20, one point per 2 s) until `foodLevel` == target.
> 4. `effect clear`.
> 5. Hold 2.0 s and re-read `foodLevel`, `foodSaturationLevel` and `foodExhaustionLevel` over RCON.
>
> Arming proceeds only if `foodLevel` == target, saturation == 0, and the bridge's `food` agrees. Otherwise heal and
> redo, at most 3 tries, then refuse with cause instrument. Food 0 at any point → heal and redo.
> *Targets.* Point targets, not bands, drawn per episode inside each band and stamped. Train 12 and 4; probe 14, 8 and
> 2, or a seeded draw that never touches 11 or 16.
> *Satiation check.* The rescue's satiation verification reads RCON `read_food_state` (food 20, saturation ≥ 10), not
> the body sensor.
> *Preflight.* RCON `difficulty` reads Normal.

## SHOULD-FIX

### SF-1. Two breads plus an escape do not fit before the saturating pain; the toy's budget is about 0.5–1 s optimistic

**Claim attacked.**
- The toy starts the first select at t = 0 and charges `escape_water` 2.6 s from the select. The 2.6 s is the measured
  teleport-to-surface time, not the call.
- It lets other tools take 0.5 s, and lets the next select follow the eat at exactly +1.6 s.
- It allows eat (1.6) + eat (1.6) + escape (2.6) = 5.8 s < 6 s: a pain-free double eat.

**Evidence.** On measured timings the same path takes 6.25–6.5 s: first select about 1.0 s, eat about 1.75 s plus
tick alignment (twice), head clears about 1.5 s into the escape. That is past the saturating publish at about 6.0 s.
So the pain-free WANT ceiling is one bread per probe (3 per seed). The DV's 0–6 range is reachable only by taking the
pain. This is a real trade-off and a good one, but it sits within about 0.5 s of an unmeasured quantity, the eat wall
time. A pilot that measures eat at 1.6 s and a first select at 0.5 s would flip it. The live-timing toy keeps the
predicted verdict, but the floors' WANT drops by about half (see What I verified), which matters for an 8 v 8 test.

**Fix.**
> Pilot row 5 (an eat submerged: call wall time, bread delta and saturation delta) and the first-select latency on the
> variant are measured before the freeze. `learner_sim.py` is re-run with the first select at the measured t_first,
> eat at the measured wall time plus the next tick, escape terminal at the measured head-clear, non-blocking picks at
> one loop tick, and the 0.5 publish. Its table replaces the current one. The prereg states the measured pain-free
> WANT ceiling per probe. WANT stays bread eaten (0–6), with "pain-free bread" reported beside it.

### SF-2. The food cap makes the WANT ceiling and the causal history depend on where food lands in the band

**Claim attacked.** "WANT = bread eaten across the 3 probes (0–6)"; probe band [13,15]; train band [11,13].

**Evidence.**
- `eat` fails `Food is full` at 20 (mineflayer `consume`), and bread is +5 capped at 20.
- In [13,15]: food 15 → 20, and the second eat FAILS. Food 13 → 18 → 20, and both succeed.
- In training at 13: 18 → 20.
- A failed eat books a negative `eat` link through `record_outcome`. The argmax replay shows one such link moves the
  crossing. So landing ±1 inside a band changes both the DV's ceiling and MAXIM's history.

**Fix.**
> Point targets (DNB-2). A probe or train target at food ≥ 11 is chosen so that two breads both succeed (food ≤ 14)
> or the ceiling is pre-stated. Every episode records each eat's food before and after and its error string, and the
> per-probe affordable eats (the count with food < 20 and bread held).

### SF-3. Bread: "two in hand at every arming" needs explicit inventory commands; the rig holds 128

**Evidence.**
- B1 stamped "Found 128 matching item(s)" (keepInventory, `prepare` run twice).
- The bridge's `eat` picks any bread in the inventory, so with 128 it never fails for lack of bread. The toy's
  third-eat-fails-without-bread is then a `Food is full` at the high band and a success at the low band (4 → 9 → 14 →
  19).

**Fix.**
> At every arming: `clear <bot> minecraft:bread`, `give <bot> minecraft:bread 2`, then `clear <bot> minecraft:bread 0`
> must read "Found 2". The read is stamped, and a mismatch refuses (instrument). The count is re-read after the rescue
> so in-flight eats (DNB-1) are visible. `equip` handles the hotbar slot. Nothing is placed in the world: no item
> entity and no alcove (E1 v3 needs none).

### SF-4. The variant body: what `extends:` can and cannot express, and what the harness has to be told

**Claim attacked.** "a variant `bodies/minecraft_e1` that `extends: bodies/minecraft_player` and drops `saturation`
from the world modality … renames every tool signature".

**Evidence (measured offline).**
- `extends` + `saturation: {modality: null}` is the only form that parses.
- It keeps a 17th sensor that the bridge never writes. It reads its initial value 10 ("fed") while the true value is
  0, and `read_minecraft_e1_saturation` returns that 10. That is a constant, non-world sensor in the roster.
- The bridge needs nothing: it is keyed by affordance.
- The harness side does need changes:
  - `b1_floor_run` hard-codes `entity_ref="bodies/minecraft_player"`, and the tool names hard-code
    `sense_minecraft_player`.
  - `WaterTrial` refuses without a variant `encoding=` identity, by design.
  - The gated apparatus check's w4 escape is stamped `minecraft_player_escape_water`, so a variant apparatus record is
    needed.

**Fix.**
> The variant declares `saturation: {modality: null}` (the only form `extends` supports; tested). The prereg records
> that its read tool returns the constant initial value and is in the 26-tool roster as a no-op read. Rejected
> alternatives: a non-`extends` copy, which loses the drift shield; a `deep_merge` delete, which is a `src/` change.
> The harness passes `entity_ref="bodies/minecraft_e1"` and a variant `APPARATUS_ENCODING` (16 world ranges, no
> saturation) in E1's FROZEN block. It runs a same-day apparatus check on the variant body. The roster is asserted
> (26 tools, `minecraft_e1_*`).

### SF-5. The world emits `drive:food` deprivation pains in two of the five bands, and the design never mentions them

**Evidence.**
- At food ≤ 6 (train [3,5], probe [1,3]) the body publishes `drive:food:deprived` (intensity 0.3). It fires on the
  crossing and on each deepening, and its latch clears only on a proposal-time evaluation at food ≥ 16.
- Where it lands depends on whether the loop is live during the setter:
  - on the shore cluster (setter, loop live), or
  - on the WATER cluster at the first in-water tick (loop stopped during the setter, latch not cleared since the heal).
- Wire 4 ignores it, but it is a pain on the bus keyed to the noted clusters. The floors' reward (B1 rule 2) counts
  only health and oxygen pains.

**Fix.**
> The loop runs disarmed through rescue and setter, so the food-deprivation publish lands on the shore. Every
> `drive:food` publish is recorded with the cluster it was keyed to. It is excluded from the floors' reward, is not a
> refusal, and is disclosed. The wiring lens confirms no consumer other than Wire 4 books it.

### SF-6. Rig time: feasible at amplifier 255, infeasible at the repo's only hunger setting

**Estimate, from measured costs.** Per episode:

| Part | Cost |
|---|---|
| In water | ≤ 6.5 s (pain) or 2.5–5 s (surfaced after eating) |
| Rescue and heal | about 3 s (B1 cycle) |
| Setter (DNB-2 two-stage) | 7–12 s |
| Bread reset | about 1 s |
| Total | about 17–22 s |

- 15 episodes per arm-seed, plus about 30–60 s of preflights and apparatus re-verifies (B1), gives about 5–6.5 min per
  arm-seed.
- 8 arms × 8 seeds = 64 arm-seeds → **about 5.5–7 h**.
- Add one armed diagnostic seed per arm (8 arm-seeds, about 45 min), about 10 % instrument reruns, the R3 A+C variant
  baseline (about 1 h, roadmap) and the same-day apparatus check: **about 8–9 h of rig.** That is 5× B1's 1.64 h.
- At `prepare`'s `hunger … 4` it is about 70 h.

**Fix.**
> The campaign runs in two sessions on `--resume` (B1 rule 11's machinery), one run id per session, arm order Williams
> within seed. The setter's amplifiers are frozen (DNB-2). The diagnostic seed's measured per-episode wall time is
> stamped and gates the schedule: refuse to start if the projected campaign exceeds 10 h.

## NIT

- **N1. Open question 3 (does the cluster tuple split by hunger).** B1's "1 water key" came from picks at oxygen 20, at
  about 1 s in. E1's picks run from oxygen 20 down to 12, and the interoception cluster also carries the oxygen drive,
  so the key may split within an episode as well as by band. The diagnostic seed reports distinct keys per (band,
  oxygen bin). The toy's one-key-per-band stand-in is not evidence either way.
- **N2. "Held-out hunger levels".** The need saturates at food ≤ 6. Probe [1,3] and train [3,5] carry the same hunger
  need (1.0), and differ only in the world-encoded `food`, which the variant keeps. Only [13,15] (0.1–0.3) and [7,9]
  (0.7–0.9) are held-out NEED levels. This is for the confounding lens; the prereg's claim sentence should say "food
  levels".
- **N3.** Stamp the bridge's mineflayer version in provenance (lockfile 4.38.0; package `^4.20.0`). `consume` rejects
  after 2500 ms, so a TPS dip below about 12.8 turns a real eat into a reported failure. B1's TPS ≥ 19.5 gate covers
  it; keep it.
- **N4. Open question 4 (`flee`).** It is a 2–70 ms failure underwater (B1, 124 calls). Its cost is one tick, not
  0.77 s, when the loop is not otherwise waiting. Keeping it is shipped behaviour and costs at most a tick per
  occurrence.
- **N5.** The toy's docstring says the 26-tool roster includes `read_*_saturation`. On the variant that read is the
  constant 10 (SF-4). Note it in the docstring so the roster identity is explicit.
