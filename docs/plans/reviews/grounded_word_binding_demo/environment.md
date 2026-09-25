# Environment lens: grounded word binding demo (v2)

**Verdict: ADOPT WITH CHANGES.** The world can host a curated teacher: player chat already reaches
the agent as a percept, with no flag needed. One of the five situations (water) is fully usable
today. Exp A has a concrete, already-measured affordance: submerged dips kept short of the US.
Two instrument defects block building their stages: the teacher's physical presence moves a world
sensor, and the lag capture cannot resolve the lag it exists to measure. Three of the five
situations (fire, cave, dark) are not discriminable, not repeatable, or not harmful as specified.
The consult cannot be anticipatory on a first exposure, because its stakes signal arrives with the
pain. Each of these is fixable without new sensors, so D1 holds.

Scope: the plan (v2 at `6ebcf9b1`), DESIGN_REVIEW.md, the simulation-experiments brief, the
minecraft_benchmark Part II decisions, roadmap_1_4 (§The classroom, Phase 1), `scripts/minecraft_bridge/index.js`,
`src/maxim/simulation/minecraft.py`, `bodies/minecraft_player.yaml`, `scripts/survival_world/*`,
`scripts/l11_real_trace_remeasure.py`, the Exp 60/61/62 preregs, and both paired-data audits.
I ran nothing against the rig or any server.

---

## DO-NOT-BUILD

### E1. The teacher is a logged-in player, so its physical presence moves the world situation
`bodies/minecraft_player.yaml` declares `nearest_player_dist` as a `modality: world` sensor: rest
64 (the bridge cap, which the range centers so it reads as silent), range [0,128], "a nearing
player descends loud". `index.js::snapshot` computes it from `bot.players[*].entity`. A scripted
teacher standing near the agent when it speaks therefore changes the world cluster at the moment
the word is heard. The word becomes confounded with "a player is near", and Exp A/B would bind a
word to a teacher-present situation that the probe (teacher absent, or present) may not
reproduce. The brief already records this hazard in the opposite direction: a stray bridge
"logs its bot into the new world as a second player — a confound for any body sensing
`nearest_player_dist`" (simulation-experiments §5, rig facts). There is a second leak:
join and leave lines ("teacher joined the game") are system messages, so they become text
percepts once `--system_messages` is on.
**Fix:** park the teacher well outside the server's player entity-tracking range (Paper's
`entity-tracking-range.players`, default 48; the bridge reads 64 = rest when `p.entity` is
untracked). Log it in before the campaign's first window and keep it online for the whole
campaign. Make `nearest_player_dist == 64` in 100% of window snapshots a per-row refusal
condition, as Exp 60's `hostile_count` is recorded. Record the teacher's join time so no
join line falls inside a window. Chat is server-wide, so distance costs nothing.

### E2. The Stage 0 lag capture cannot resolve a lag of about 1 s
`MinecraftClient._handle_line` appends events as `{"kind", "text"}` with **no arrival time**.
`l11_real_trace_remeasure.py capture` stamps every event with the **poll** time of a 0.5 s loop
(`SNAPSHOT_CADENCE_S = 0.5`; the committed trace shows a `damage` and a `system` event sharing one
`ts`). Its docstring's claim that "event records also carry the bridge-arrival wall time" is false
in the code. The measurement is therefore quantized to two samples at exactly the scale it must
resolve (the re-audit's t−2 ≈ 1 s), and the window it fixes would inherit a ±0.5 s floor.
**Fix, needed before the capture:**
- the bridge stamps `t_ms = Date.now()` on every `state` and `event` line;
- the client records `time.monotonic()` on receipt;
- the capture runs the bridge at `--state_interval_ms=100` (Exp 60 Amendment 4's cadence);
- the capture writes both times;
- correct the docstring, or add the field it describes.

This is a protocol addition, and `minecraft.py` is the protocol authority, so pin it in the
FakeBridge lockstep test.

---

## SHOULD-FIX

### E3. Teacher transport: chat works today, but the plan names the wrong channel and no teacher process exists
- **Player chat is forwarded unconditionally.** The handler is `bot.on("chat")` →
  `event("chat", "<username> says: <msg>")`, with no flag. The `messagestr`/`position === "system"`
  path, gated by `--system_messages`, carries only system lines. A teacher that speaks as a player
  arrives as `[minecraft:chat] teacher says: water`. Stage 1's guard names a
  `[minecraft:system]` percept, which is the channel for death, advancement and `tellraw` lines.
  **Pick one channel and pin the exact percept string in the prereg.** Player chat is the
  D1-cleanest choice (another player speaking is a game-native multiplayer affordance). `tellraw`
  from RCON would arrive as `system`, indistinguishable by kind from death messages. RCON
  `/say` in 1.20.4 is a disguised chat message, and I have not verified how mineflayer classifies
  it; make that a pilot row if it is used.
- **D1 constrains the path, not just the channel.** The teacher's text must travel the real server
  (bot chat → server → the AUT's bridge). Injecting it into `MinecraftClient._events` or a
  FakeBridge is a synthetic sensor, and it would also erase the real delivery lag.
- **The bridge serves one client** ("bridge busy: one client at a time") and has **no chat
  action**. A separate teacher process therefore cannot read the AUT's state from the AUT's
  bridge. **Moments must be scheduled by the harness that owns the AUT client.** In the Exp 60
  pattern the harness causes the situation (teleport-in), so the teacher fires at a pre-registered
  offset from the harness's own placement, which is reliable by construction. Validity is then
  checked on receipt: the AUT's own snapshot at the percept's arrival must show the situation
  (e.g. `is_in_water == 1`), or the row is refused. The teacher is its own minimal mineflayer
  process (it needs no bridge), and its send time is stamped.
- **Rate:** vanilla/Paper kicks for chat spam at about 10 lines/s sustained. The teacher is sparse,
  so this is only a note for any "repeat the word" variant. The rig world is `online-mode=false`,
  so signed-chat enforcement should not apply. Verify that in the pilot.

### E4. Situation-by-situation: only water is ready, and three situations are not ready as written
The water classroom's only live world discriminator is `is_in_water` (roadmap_1_4 §Phase 1 bound:
`live_contributors == ["is_in_water"]`). Every other situation must be separated by a sensor that
departs from rest.

| Word | Reachable | Discriminable by a world sensor | Safe to repeat | Reuse | Disposition |
|---|---|---|---|---|---|
| `water` (drowning) | Yes: teleport-in (`setup_world.py water_classroom`, Exp 60/61/62 pools) | Yes: `is_in_water` at EYE height (index.js) | Yes: US-free cap 4.40 s, pain edge 5.15–5.28 s, damage onset ≈ 16 s, all measured | Exp 60/61/62 apparatus, `WaterTrial`, preflights | **Ready**; the anchor situation |
| `fire` / lava edge | New classroom | **Not as specified.** No fire/lava/on-fire sensor exists. Before damage, the only discriminator is `light_level` (lava, fire and campfire emit block light 15; the sealed shell reads 0). After damage, `health` separates, but it is shared with every damage cause (the re-audit measured drowned–slain at 0.71). Light is a known constant-mass contributor with `rest: null` and "patchy across restarts" (roadmap §The classroom). | **Poor.** Standing in fire/lava sets the player burning for seconds after exit, so the harm outlasts the situation and books onto the NEXT (shore) cluster. `doFireTick` is not in `_GAMERULES`, so spread is unguarded. Every hit emits the bridge's `damage` text (not gated), which puts text at the harm. | None | Re-scope to a **lit campfire cell in a sealed dark shell**: the key is light 15 vs 0, the harm is contact damage with no burn-over (pilot must verify both), plus `doFireTick false`. The word may then bind to "bright place". Say so. Or drop it from the lead-up. |
| `food` (hungry) | Yes: `effect give hunger` → `effect clear` (natural drain was measured at zero over 60 s in the frozen world); bread from `prepare`; `foodLevel ≤ 15` | Yes, marginally: fed vs hungry measured cos 0.826 (saturation's constant mass) | Yes | Exp 58/60 prepare + R3 | Usable. Pilot regen–hunger coupling (roadmap pilot row 9). |
| `cave` (never seen) | Teleport | **Only if high-gain sensors depart.** Exp 62 showed that relocating a situation is invariant to the low-gain place absolutes (`offset_x/z`, `distance_from_spawn`). A new place at the same altitude, light and water state is the SAME cluster, so it is not novel and the consult trigger never sees novelty. | **No, per agent.** Novelty (`1 − best similarity`) is consumed by the first visit. Each novelty trial needs a fresh agent or a new high-gain situation. | Exp 62's lit pond (cos 0.588) and night pool (0.799) are measured novel-to-a-dark-pool situations | Define "never seen" as a cluster the agent has no EC node within 0.85 of, and verify it on committed vectors (Exp 62's replay method) before the prereg. |
| `dark` (night) | `time set night` via RCON (the classroom freezes day) | Only through `time_of_day` (`rest: null`). In the sealed shell, sky light is 0 at any hour, so the dark classroom is ALWAYS dark: the word `dark` names every classroom situation and has no contrast. Night on the open surface also changes light, but `doMobSpawning false` removes the harm, and summoned hostiles key the fear onto `nearest_hostile_dist`/`hostile_count`, not onto night. | Yes | Exp 62's night pool | Rename it `night`. The only harmful night situation available is the **night pool**, which shows the 0.799 boundary honestly, as the plan intends. |

### E5. Exp A (sensory preconditioning): the world affords it in water, and only in water
The plan's worry is correct: wading reads `is_in_water = 0`, because the sensor reads the EYE
block, not the feet. Shallow water therefore falls in the shore cluster and cannot serve as the
harmless phase. Drowning does have a harmless phase already, and it is measured:

- **Phase 1 (preconditioning, word present, harmless):** teleport-in submerged, teacher says
  `water` at a fixed offset (e.g. +1.0 s), and the harness removes the agent at ≤ the US-free cap
  (4.40 s, roadmap §The classroom). Oxygen stays above the comfort band (pain below 14 bubbles), so
  no pain is published and no fear is booked. Exp 61's representation gate already runs "one
  loop-OFF submersion, US-free, ≈ 2 s".
- **Phase 2 (conditioning, word absent):** Exp 60's training dives. Its US is **oxygen pain before
  any health damage**, so no `damage`/`death` text is emitted during conditioning. Keep every dive
  capped before the damage onset (≈ 16 s), or `maxim drowned` (under `--system_messages`) and
  `took damage` land co-present with the harm.
- **Phase 3 (probe):** the word alone on a **dry cell displaced from the flee anchor**
  (`is_in_water = 0`, shore cluster, no fear).

Required checks, and a limit:
- **Cluster identity** between the Phase 1 dip state (oxygen ≈ 15) and the Phase 2 training
  state, as a per-row gate on the live cluster ids.
- **Zero-text windows:** every text event in every Phase 2 window is recorded, and any
  non-teacher text refuses the row.
- **The DV can only be a decision.** On land, the fear consumers are `flee` (valid only when
  displaced from its anchor; a goto to where the agent already stands "books success for doing
  nothing") and `escape_water`, which on dry land returns `"already at surface"` as a *success*.
  The name tie-break sorts `escape_water` first. So the executed behaviour is a no-op, and the DV
  must be the executed/proposed fear-consumer choice with its provenance, stated as a decision DV
  (Exp 61's decision-vs-behavioural split). No other situation affords the design:
  - fire leaks harm across the exit;
  - night's harm is a different cluster;
  - food is a want, not a harmless-then-harmful valence flip.

Number of Phase 1 dips: repeated harmless exposure is the classic latent-inhibition setup. This is
a flag for the bio lens; environmentally, each dip costs about 20 s including settle.

### E6. "Natural, unstaged deaths" are feasible, but not in the classroom world and not for most of the vocabulary
- **Gamerules are world-wide.** The classroom builds set `doMobSpawning false` and
  `_GAMERULES` freezes day and weather. A natural-death capture in the survival world would
  either find no hazards or unfreeze the world that every classroom depends on. **Use a dedicated
  world (its own server dir) with the day cycle and mob spawning on.** Run one server at a time
  (both templates bind 25565/25575; brief §5).
- **A substrate AUT does not die naturally.** It has no locomotion affinity, and "a shore start
  never descends in any arm". A **mineflayer wanderer** does: random `GoalNearXZ` targets. Its
  default Movements avoid lethal drops and lava and are dead in water, so its natural deaths will
  be almost entirely **mob kills at night**. To get falls, raise `maxDropDown`. Drowning and fire
  will essentially never happen unstaged.
- **Honest scope:** the death message is emitted at the death tick whatever the cause, and
  `doImmediateRespawn` makes the at-event snapshot post-respawn. So one cause family can measure
  the *delivery lag* if the prereg says the lag is cause-independent and checks it on the causes
  obtained. The capture cannot supply natural examples for `water`/`fire`, which is what the
  curated teacher is for. Run a 30-minute pilot to measure the death rate before fixing *n*.

### E7. The consult's stakes gate fires only at pain on a first exposure, so it cannot be anticipatory
A naive receiver placed in the pool has no anticipatory threat and no fear history, and oxygen has
no innate need (by design, Exp 60). Its first stakes signal is the oxygen pain at ≈ 5.15 s. A
consult triggered then must fetch, merge at a tick boundary, and drive `escape_water` (1.5–1.9 s
measured) before health damage at ≈ 16 s. That window is feasible, about 9 s. Behaviour without
help is a known floor: Exp 60's pre-probe was censored 30/30 with zero calls, so the agent drowns.
**Fix:** Exp C's DV is P(surface before damage onset | pain onset). It is not "before the US",
and the prereg says so. Exp 61's pool is the reusable testbed:
- the donor's dark-pool fear is on the Oasis;
- the receiver is novel to that pool and matches it at ≥ 0.85.

The lit pond (0.588) is the no-match control, where a consult should return nothing.

### E8. Latency budget: measure the consult on the loop, do not assume it fits
- **The loop, measured:**
  - Exp 60's liveness preflight read 6 ticks in 3 s;
  - a primitive costs a 0.58 s tick plus its hold, with a 0.77 s `flee` tie-break tax
    (roadmap §The classroom);
  - the bridge runs at 100 ms.

  `scripts/survival_world/loop_tick_probe.py` is the instrument (cProfile on the loop thread plus
  substrate tick times). No consult latency exists to cite, because search is absent.
- **Three new costs land on or near the loop thread:**
  1. **The text encode:** `LinguisticEncoder` loads `paraphrase-mpnet-base-v2` lazily through
     `_get_encoder`. The first call loads the model (seconds) on whichever thread calls it. Any
     caller asking for a different model name, such as `_get_encoder`'s own default
     `all-mpnet-base-v2`, **reloads the single global model**.
  2. **The off-thread fetch**, which is network time.
  3. **The tick-boundary `load_state()` of the merged NAc/EC**, which is on the loop.

  **Fix:** add three preflights:
  - encoder warm-up before the first window;
  - `require_semantic_encoder(...)` as a refusal;
  - a `loop_tick_probe` run with text events ON and a synthetic merge injected (fake bridge,
    offline), reporting tick gaps with and without a merge.

  Refuse if liveness drops below Exp 60's ≥ 4 ticks / 3 s.
- **The silent fallback:** if `sentence_transformers` is missing, `LinguisticEncoder` falls back
  to **384-d bag-of-words hashes** with one warning. That silently breaks the plan's "768-d" text
  modality and any calibrated text threshold. Make the preflight refuse.

### E9. The text percept's string shape may collapse the text clusters
Every teacher percept is `[minecraft:chat] teacher says: <word>`: five strings that share all but
one token. Bridge templates (`[minecraft:damage] took damage (health N)`, `the player died`) are
also text percepts. Stage 1 must say which kinds enter the situation key. A shared prefix can
put two teacher words above a calibrated text threshold, or put a word and a damage template in
one text cluster. The second case would put "the word" into the conditioning phase by proxy and
void Exp A.
**Fix, offline and cheap:** before the prereg, encode the exact strings through the shipped
encoder and report the pairwise-cosine matrix. Compare it with the candidate threshold. Decide
whether the harness strips the `[minecraft:chat] <name> says:` prefix. That is a representational
choice, so record it.

### E10. Apparatus text under `--system_messages`
Stage 0/1 turn `--system_messages` on, so every system line becomes a percept. RCON apparatus
commands (`tp`, `effect`, `clear`, `give`) send their feedback to the RCON source. The bot sees
them only if it is op (`broadcastConsoleToOps`). The survival world does not op the bot, but
`scripts/exp56/run_campaign.py` does op a spectator. **Pin that neither the AUT nor the teacher
is op.** Verify in the pilot that a full placement cycle produces zero system lines at the AUT.
Otherwise a teleport line becomes a perfect predictor of submersion.

### E11. Rig and run-time budget (big-mac-mini, Paper 1.20.4)
- **Constraints:**
  - Paper 1.20.4 needs Java 17 (`java_home -v 17`);
  - the survival server is tmux `minecraft`;
  - both server templates bind 25565/25575, so one server at a time, and stop the other *bridge*;
  - one client per bridge, with the flee anchor fixed at bridge start (restart the bridge per
    classroom, and after any pull: bridge-restart memory);
  - `max-players=4` covers the AUT, the teacher, and the Exp 61 second AUT;
  - the box must be quiet (no pytest, no qwen32b server during timing windows);
  - the Oasis server should NOT run on the leader's :8100 box during a run. Declare its host:
    loopback on the rig gives a latency floor; a second machine gives the real transport.
- **Estimates** from measured per-row costs (Exp 61: apparatus ≈ 1 min, training 2.5–3 min,
  receiver 60–90 s; Exp 60: 10 seeds in 56 min):

  | Stage | Rows | Estimate |
  |---|---|---|
  | Stage 0 | 30-min rate pilot, then capture to *n* ≥ 30 deaths | 1–3 h (rate unknown) |
  | Exp A | 5 arms × n; per row ≈ 1 min apparatus + ~5 dips × 20 s + training 3 min + probe ~1 min ≈ 6–7 min | n = 12: ≈ 6–7 h; n = 5: ≈ 3 h |
  | Exp B | Exp A donors reused; 4 arms × 2 transports × 12 receivers × ~75 s, plus a signed publish/pull per donor | ≈ 2–2.5 h |
  | Exp C | 3 gates × 2 Oasis × 12, per row ~1.5–2 min; "own experience wins" rows add a training (~3 min) | ≈ 3–5 h |

  Split Exp A and C across sessions at one code hash each. The budget should be in each prereg
  before freeze.

---

## NIT

- **N1.** "Fire / lava edge": a lava edge harms nothing without contact. Keep the word only if the
  harm is contact.
- **N2.** Exp 62's night pool was measured at `time_of_day` 0.99, which is just before dawn
  (vanilla midnight is 18000 = 0.75). Name the time the `night` teacher situation uses. The body's
  own comment notes that `time_of_day` wraps at midnight.
- **N3.** Stage 0's "vocabulary count" is trivial for a five-word curated teacher. Report the
  natural-capture vocabulary and the teacher vocabulary separately.
- **N4.** The damage one-hit lag (16/39 at the paired snapshot, 38/39 paired-or-previous) still
  affects any `fire` binding keyed on `damage` text. Fix it or document it (re-audit's last line).

## Verified fine

- Player chat already reaches the agent as a text percept on every run (`bot.on("chat")`,
  not gated); the percept source wakes the loop on events (`has_pending`); and text is excluded
  from `_SUBSTRATE_CHANNELS` today, so Stage 1 is genuinely an opt-in arm.
- A teacher player speaking in chat is game-native under D1: no synthetic sensor or reward,
  provided it travels the server (E3) and its presence is silent (E1).
- The water situation is reachable, discriminable, and safe to repeat. The US-free cap (4.40 s),
  pain edge (5.15–5.28 s), escape time (1.5–1.9 s), bridge cadence (0.101 s) and liveness (6 ticks
  / 3 s) are all measured, and the Exp 60/61/62 classroom, `WaterTrial` and preflights are reusable
  as they are.
- `keepInventory`, `doImmediateRespawn`, frozen day and weather, and `mobGriefing false` make
  repeated drowning and hunger safe. `online-mode=false` suits scripted bots.
- The existing Exp 62 context-wall measurements (lit pond 0.588, night pool 0.799) give the
  consult and "never seen" work ready-made novel-but-comparable situations.
- Exp C's "corrupted Oasis" is affordable for want. For fear it is structurally limited: fear
  merges are MIN-fold, tighten-only (Exp 61), so an inverted-fear entry cannot loosen a receiver's
  fear. The prereg should pick corruption types knowing that.
