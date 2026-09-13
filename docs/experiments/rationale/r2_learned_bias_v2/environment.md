# Environment lens — R2 learned-bias v2 design review

**VERDICT: DO-NOT-BUILD as written.** The competitor pool `[move_to, mine_block, turn]` is not
game-natively always-executable/always-successful on the DEFAULT survival world, and this body
exposes only ~1 reliable competitor tool — so the balance premise (the very confound v2 exists to
kill) re-breaks, and K=3 cannot be built from distinct reliable tools. Compounded by two more
live-world gaps: probe food levels 11/18 are not reliably settable to exact integers game-natively
(and rung-1 forbids injection), and the DEFAULT+mobs world drifts across a long round-robin. All
three are fixable — adopt Exp 56's opaque-turn roster, a confirm-food-landing loop, and
`spawn-monsters=false` — but the fixes change the apparatus enough that the prereg must be re-frozen
before any build. Verified: `turn`, `eat`, `mine_block`, `move_to` handlers exist in the live bridge;
`keepInventory`/`doImmediateRespawn` gamerules are set; RCON works; the world hunger mechanic is
game-native (D1-clean).

Reviewed against: `scripts/minecraft_bridge/index.js` (live bridge handlers),
`scripts/survival_world/setup_world.py` (the live world config), `scripts/survival_world/break3_smoke.py`
(the only live loop exercised so far), `scripts/exp56/common.py` (the proven turn-roster apparatus),
`src/maxim/_data/components/bodies/minecraft_player.yaml` (the roster + drives),
`docs/wiring/substrate-learning-channels.md`, `docs/experiments/r2_drive_premise_check.md`, and
DECISIONS.md 2026-09-12 (the game-native / injected-signal lanes).

---

## DO-NOT-BUILD

### D1. `move_to` and `mine_block` are NOT always-executable/always-successful on the DEFAULT survival world — and only ~1 reliable competitor tool exists on this body.

The prereg's competitor requirement is explicit: *"non-corrective, always-executable,
always-successful tools that build their own causal links"*, with the candidate pool
`[move_to, mine_block, turn]`, and the balance premise (`gap_LEARNING − gap_NO-CREDIT`) rests on each
competitor accruing a causal link comparable to eat's. The causal link forms **only on success**
(`nac.observe` on tool success — `docs/wiring/substrate-learning-channels.md`). Against the live
bridge on a `level-type=DEFAULT`, mob-populated world this premise fails:

- **`move_to`** (`index.js` `case "move_to"`) runs `bot.pathfinder.goto(GoalNearXZ(x, z, 1))` with
  **no timeout**. On DEFAULT terrain (hills, water, trees, ravines, caves) pathfinding to an
  arbitrary XZ can throw `NoPath`/`GoalChanged` (→ `action_result ok:false`) or search for a very
  long time. It also **moves the bot**, so every subsequent `move_to`/`mine_block` coordinate shifts
  meaning, and it perturbs the position world-sensors (`distance_from_spawn`, `offset_x/z`, `speed`,
  `y_altitude`).
- **`mine_block`** (`case "mine_block"`) throws `"no block there"` on air, and `bot.dig` can hang or
  fail on unbreakable/tool-gated blocks. Decisively: mining is **destructive** — a fixed coordinate
  that is solid in episode 1 is **air in episode 2** (`keepInventory` restores items, not terrain),
  so the SAME coordinate throws `"no block there"` on re-mine. mine_block cannot be repeated at one
  spot, and moving the spot re-introduces reachability failures.

**Failure scenario:** K=2 arm, c2 = `mine_block(x,y,z)`. Episode 1 mines it (success, causal link
+1). Episode 2: coordinate is now air → fail → no link increment. Over the round-robin `mine_block`
accrues far fewer successes than eat → **eat's causal link dominates → forced null in LEARNING** —
indistinguishable from a true drive-relief null. That is exactly the causal-link re-saturation v1
died of and v2 is built to isolate; an intermittently-failing competitor re-creates it.

Worse, this is **structural, not tunable**: the `minecraft_player` roster has exactly one each of
`move_to`, `turn`, `mine_block`, `place_block`, `eat`, `attack_nearest`. `eat` is the corrective
action under test; `attack_nearest` needs a hostile; `place_block` needs a held placeable + valid
reference; `move_to`/`mine_block` are the two just shown unreliable. The **only** always-executable,
non-perturbing competitor is `turn`. So the body affords **one** reliable distinct competitor tool —
you cannot assemble K=2 or K=3 distinct always-successful competitor **tools** from it. The titration
axis (K distinct causal-linked competitors) has no game-native substrate here as designed.

**Fix (proven precedent):** adopt Exp 56's roster pattern (`scripts/exp56/common.py`): mint K opaque
affordances `aff_1..aff_K`, each mapped by a per-run `TranslatingClient` to a distinct `turn(degrees)`
bridge action. `scripts/exp56/common.py::BRIDGE_ACTIONS` is 8 turn variants chosen for exactly this
reason — *"ALL always-executable turns … turns also never perturb the situation's location sensors
mid-trial."* This gives K **distinct tool names** (→ K distinct `tool:aff_i` causal links, real
titration), all physically `turn` (game-native, D1-clean), all always-successful, all
position-non-perturbing. `turn` is verified reliable in `index.js` (`bot.look` never throws; changes
yaw only — `look_pitch`, position, and `speed` sensors are untouched). If instead you insist on the
named tools, the prereg must justify per-competitor reliability against the live DEFAULT world and
solve the mine-destructiveness/pathfind-failure problems — which reduces back to "use turns."

---

## SHOULD-FIX

### S1. Probe food levels 11 (hungry) and 18 (satiated) are not reliably settable to exact integers game-natively — and rung-1 forbids injection.

The prereg pins hungry = **food 11** because *"food 11 is the only None-band state SHARING the
food≤4 training cluster fd0ae83c; food 12/13 are a different cluster"* — i.e. the probe must land on
**exactly 11**, not 10 or 12, or the cluster-sharing invariant (the whole isolation argument) silently
fails. There is **no vanilla/Paper command to set a player's food to an exact value** (players can't
be `/data`-targeted; there is no food attribute). The only game-native levers are:

- **drain** via `effect give <p> minecraft:hunger` (what `break3_smoke.py::_drain_until_hungry` uses,
  polling every 1.5 s to `food ≤ target`) — **overshoots**: food can drop 12→10 between polls,
  skipping 11 entirely;
- **raise** by eating bread (+~5 food, discrete jumps — can't land 11 or 18) or
  `effect give … saturation` (fills to 20 — overshoots 18).

So neither 11 nor 18 is reliably reachable by drain-and-hope. `break3_smoke.py` only ever drove food
to `≤3` and ate a few times; it has **never** established food==11 or ==18, so this precision has no
live evidence. DECISIONS.md 2026-09-12 rule #4 bars the injected-signal lane for rung-1 game-native
claims (this experiment is rung 1), so you **cannot** just set an exact synthetic food value — the
landing must be game-native.

**Failure scenario:** harness drains toward 11, lands on 10, encodes a *different* cluster than
fd0ae83c; the hungry probe silently measures a non-training cluster; gap collapses; NULL — a
measurement artifact reported as a science result.

**Fix:** specify a **confirm-landing loop** (the same confirm-don't-assume discipline as Exp 56's
`settle_until_reflected`): drain/eat, `sync_world_sensors()`, read `vital_metrics["food"]`, repeat
until it equals the target exactly, and **refuse the seed (REFUSED-UNVERIFIED)** if the exact value
isn't reached within N tries. Then make the **pilot gate** this explicitly: demonstrate that food==11
and food==18 can be landed + confirmed repeatably before freeze. If exact-11 can't be hit
game-natively, the two-adjacent-cluster design (11 in / 12,13 out) is unbuildable on rung 1 and must
be redesigned (e.g. a wider hungry band that is robustly one cluster).

### S2. The DEFAULT + `spawn-monsters=true` world drifts across a long round-robin — mobs can damage/kill→respawn and (if the credit cluster is multi-sensor) move the probe cluster.

`scripts/survival_world/setup_world.py` sets `level-type=DEFAULT`, `difficulty=normal`,
**`spawn-monsters=true`** — whereas the proven `scripts/exp56/setup_world.py` uses
`spawn-monsters=false`. Over a long round-robin (K × 3 arms × seeds × N × episodes) this world does
not hold still:

- **Mob spawns → damage → health drive / respawn.** A hostile attacking drops `health`, which carries
  a homeostatic drive (`minecraft_player.yaml`), perturbing `_read_drive_states` and the prior; a kill
  triggers `doImmediateRespawn` **to spawn**, teleporting the bot and invalidating any
  coordinate-based competitor (compounds D1). `time set day` + frozen daylight reduces surface spawns
  but caves/shadows still spawn; only `spawn-monsters=false` removes the class.
- **If the credit-keying cluster is computed from the multi-sensor `world` channel** (as Exp 56's
  `encode_clusters` encodes the *whole* world modality into one node), then the "food 11 → cluster
  fd0ae83c" identity is a function of health, `nearest_hostile_dist`, position, etc. — all of which
  drift with mobs and a moving bot — so the probe's cluster is **non-stationary** and the
  cluster-sharing invariant can break mid-run. The v1 "food 11 shares fd0ae83c" diagnostic was an
  offline/quiescent measurement; it is not established on a live, moving, mob-populated world.
- **Terrain/position accumulation** from `mine_block` craters and `move_to` wandering degrades
  reproducibility across arms/seeds even absent mobs (ties to D1's turn-only fix, which keeps the bot
  planted — Exp 56's stated rationale).

**Fix:** (a) set `spawn-monsters=false` in the survival `server.properties` — hunger still drains
(difficulty ≥ easy, monster-independent), so this stays game-native/D1-clean; (b) use the turn-only
competitor roster (D1 fix) so the bot never moves and position/terrain never drift; (c) the prereg
must **state exactly which sensors key the cluster the drive-relief credit books to** — if it is the
multi-sensor world channel, the probe cluster's stability under any residual drift must be
demonstrated in the pilot, not assumed.

### S3. The cited reuse harness `scripts/survival_world/r2_learned_bias.py` does not exist — v1 was offline-only, so none of the live-world risks above are de-risked by prior work.

Both the prereg ("Apparatus… Reuses `scripts/survival_world/r2_learned_bias.py` machinery + all
guards") and `docs/wiring/substrate-learning-channels.md` ("Established… `scripts/survival_world/
r2_learned_bias.py`") cite a harness that is **absent from the repo** (`grep -rl r2_learned_bias
--include=*.py` returns nothing; `scripts/survival_world/` holds only `setup_world.py` and
`break3_smoke.py`). The wiring doc's own provenance is *"dry-run + offline replication"* — v1 proved
the causal-link confound **offline**. Consequently the live-survival-world machinery v2 depends on
(food-landing at exact integers, a long live round-robin, live competitor training, the
frozen-apparatus fingerprint + bridge-connect-retry guards the prereg says "carry over") **has never
been built or run** except the print-only `break3_smoke.py`, which drains to ≤3 and eats ~4 times.
The "all v1 guards carry over" claim is therefore unbacked. **Fix:** locate or state plainly that the
live harness is net-new; treat S1/S2 as unproven-until-piloted (they are); scope the pilot to
exercise the full live round-robin at small N before freeze, not just "confirm the effect."

### S4. Bread depletion and eat-from-deficit across the round-robin are unspecified.

`eat` (`index.js`) consumes bread and only produces measured relief (break-2 credit) when `bot.food`
actually rises — near the cap it rises little/none. Over a long drain→eat→competitor round-robin the
seeded 64 bread depletes → `eat` throws `"no food in inventory"` → eat's causal link stops growing
and eat probes fail; and eat episodes run from too-high a food level produce no relief signal to
credit. **Fix:** the harness must (a) re-`give` bread when low (game-native, fine) and (b) ensure
each eat-training episode starts from a genuine deficit so relief is measurable — specify both, or
the "eat trained identically across all K" balance claim is not met.

---

## NIT

### N1. One-client bridge over a long run.
The bridge is one-client (`index.js`) and has no reconnect logic of its own; a mid-run
kick/disconnect (`bot.on("kicked")`) or an idle-watchdog on a very long run would drop the session.
The prereg says bridge-connect retry "carries over" — confirm the retry re-establishes cleanly and
that a reconnect doesn't reset any world-state assumption the harness holds.

---

## Verified working / D1-positive (no action needed)

- `turn`, `eat`, `mine_block`, `move_to`, `place_block`, `attack_nearest` handlers all exist in the
  live bridge; `turn` is always-successful and position-non-perturbing (the one reliable competitor).
- `setup_world.py` sets `keepInventory=true` (food not stripped on death) and
  `doImmediateRespawn=true` (no respawn-screen stall) — both correctly anticipate death; the eat
  food-update poll (`index.js`, the fixed eat-lag) correctly waits for the real food packet so the
  action_result snapshot is not stale.
- **D1 compliance of the mechanics is sound:** hunger drain, RCON `effect give`/`give`, and `eat` are
  game-native game mechanics, not synthetic sensors or bespoke reward — the rung-1 game-native
  posture holds *provided* the food-landing fix (S1) stays game-native (drain/eat, never an injected
  exact value, which would cross into the DECISIONS.md 2026-09-12 injected-signal lane forbidden for
  this rung).
