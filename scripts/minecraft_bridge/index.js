#!/usr/bin/env node
/**
 * Maxim Minecraft bridge — the JS half of the 1.1.4 world seam (PR 3).
 *
 * Owns the Mineflayer bot and speaks the frozen NDJSON-over-TCP protocol
 * documented in src/maxim/simulation/minecraft.py (the Python side is the
 * protocol authority — keep the two in sync):
 *
 *   JS -> PY  {"type":"state","data":{health,food,light_level,y_altitude,
 *              nearest_hostile_dist,time_of_day}}
 *   JS -> PY  {"type":"event","kind":"chat|damage|death|block|spawn|info",
 *              "text":"..."}
 *   PY -> JS  {"type":"action","id":N,"name":"move_to|turn|mine_block|
 *              place_block|eat|attack_nearest","params":{...}}
 *   JS -> PY  {"type":"action_result","id":N,"ok":bool,"detail":"...",
 *              "state":{...}}
 *
 * Not packaged, not CI-run (CI has no Minecraft server): a dev-side tool,
 * like everything under scripts/. See README.md here for setup.
 */

"use strict";

const net = require("net");
const mineflayer = require("mineflayer");
const { pathfinder, Movements, goals } = require("mineflayer-pathfinder");

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, "").split("=");
    return [k, v ?? true];
  })
);

const MC_HOST = args.mc_host || "127.0.0.1";
const MC_PORT = parseInt(args.mc_port || "25565", 10);
const BRIDGE_PORT = parseInt(args.bridge_port || "25567", 10);
const USERNAME = args.username || "maxim";
const STATE_INTERVAL_MS = parseInt(args.state_interval_ms || "500", 10);
const FLEE_X = args.flee_x !== undefined ? parseFloat(args.flee_x) : null;
const FLEE_Z = args.flee_z !== undefined ? parseFloat(args.flee_z) : null;

const bot = mineflayer.createBot({ host: MC_HOST, port: MC_PORT, username: USERNAME });
bot.loadPlugin(pathfinder);

let client = null; // one Maxim at a time (the two-AUT harness runs two bridges)

function send(obj) {
  if (client && !client.destroyed) client.write(JSON.stringify(obj) + "\n");
}

// Perceived brightness at the agent — the debug-screen quantity, NOT raw block light.
// Block light alone reads 0 on a sunlit surface (the sun feeds SKY light; block light
// counts only torches/lava/etc.), which is why the sensor read "dead 0 everywhere" in
// Exp 56: broad daylight and a lethal cave are the same 0. Effective light =
// max(block, sky - darkness(time)): full sun ~15 at noon, moonlit surface ~4, cave 0.
// Both halves are game-exposed (D1-clean). Sky darkness ramps over dusk (12000-13800)
// and dawn (22200-24000), 11 through the night — the vanilla brightness ramp, linearized.
function skyDarkness() {
  const t = (bot.time?.timeOfDay ?? 0) % 24000;
  if (t < 12000) return 0;
  if (t < 13800) return (11 * (t - 12000)) / 1800;
  if (t < 22200) return 11;
  return (11 * (24000 - t)) / 1800;
}
function perceivedLight(me) {
  if (!bot.world || !me) return 7;
  const block = bot.world.getBlockLight?.(me.position) ?? 0;
  const sky = bot.world.getSkyLight?.(me.position) ?? 0;
  return Math.max(block, Math.max(0, sky - skyDarkness()));
}

function snapshot() {
  // Emits EVERY modality:world sensor bodies/minecraft_player.yaml declares
  // (16 — the L11 re-measure needs the channel above the ~12 safe band);
  // the lockstep is test-pinned on the Python side via FakeBridgeServer.
  const me = bot.entity;
  const hostiles = Object.values(bot.entities).filter(
    (e) => e.kind === "Hostile mobs" && e.position && me && me.position
  );
  let nearest = 64;
  for (const h of hostiles) {
    const d = me.position.distanceTo(h.position);
    if (d < nearest) nearest = d;
  }
  let nearestPlayer = 64;
  for (const p of Object.values(bot.players)) {
    if (p.username !== bot.username && p.entity && p.entity.position && me && me.position) {
      const d = me.position.distanceTo(p.entity.position);
      if (d < nearestPlayer) nearestPlayer = d;
    }
  }
  const spawn = bot.spawnPoint || (me ? me.position : null);
  const distSpawn = me && spawn ? Math.min(128, me.position.distanceTo(spawn)) : 0;
  // SIGNED horizontal offset from spawn — what the game already exposes
  // (distance_from_spawn is the HORIZONTAL-plane magnitude of this pair; the 3D
  // distanceTo also includes the y term, so they coincide only at equal
  // altitude — offsets are still strictly less lossy than the magnitude).
  // Direction-bearing, so situations that differ only in bearing (the Exp 57
  // contingency slots) separate; the direction-blind distance alone collapses
  // them. Same spawn basis as distSpawn; clamped to the +-128 body-declared range.
  const clampOff = (v) => Math.max(-128, Math.min(128, v));
  const offsetX = me && spawn ? clampOff(me.position.x - spawn.x) : 0;
  const offsetZ = me && spawn ? clampOff(me.position.z - spawn.z) : 0;
  const vel = me ? me.velocity : null;
  const speed = vel ? Math.min(1, Math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)) : 0;
  // Exp 60: is_in_water — the stable underwater cue. Primary signal is the block
  // at the HEAD (position + 1): head-submerged is exactly the condition under
  // which oxygen depletes, so it tracks the drowning situation, not just feet
  // getting wet. me.isInWater is an OR fallback for versions/edge cases.
  // HEAD-block only (NOT me.isInWater): oxygen depletes exactly when the head
  // is submerged, so this tracks the DROWNING situation. me.isInWater is true
  // for any water contact incl. feet-wet wading (head in air, oxygen full, not
  // drowning) — including it would inject non-drowning states into the
  // underwater cluster and disagree with escape_water's head-only check (S1).
  const headBlock = me ? bot.blockAt(me.position.offset(0, 1, 0)) : null;
  const inWater = headBlock && (headBlock.name === "water" || headBlock.name === "bubble_column") ? 1 : 0;
  return {
    health: bot.health ?? 20,
    food: bot.food ?? 20,
    saturation: Math.min(10, bot.foodSaturation ?? 5),
    oxygen: bot.oxygenLevel ?? 20,
    light_level: perceivedLight(me),
    y_altitude: me ? me.position.y : 64,
    nearest_hostile_dist: nearest,
    hostile_count: Math.min(32, hostiles.length),
    nearest_player_dist: nearestPlayer,
    distance_from_spawn: distSpawn,
    offset_x: offsetX,
    offset_z: offsetZ,
    speed,
    on_ground: me && me.onGround ? 1 : 0,
    is_raining: bot.isRaining ? 1 : 0,
    is_in_water: inWater,
    xp_level: Math.min(50, bot.experience ? bot.experience.level : 0),
    look_pitch: me ? me.pitch : 0,
    time_of_day: bot.time ? (bot.time.timeOfDay % 24000) / 24000 : 0,
  };
}

function event(kind, text) {
  send({ type: "event", kind, text });
}

// ── game -> events ─────────────────────────────────────────────────────────
bot.on("chat", (username, message) => {
  if (username !== bot.username) event("chat", `${username} says: ${message}`);
});
bot.on("entityHurt", (entity) => {
  if (bot.entity && entity.id === bot.entity.id) event("damage", `took damage (health ${bot.health})`);
});
bot.on("death", () => event("death", "the player died"));
bot.on("entitySpawn", (entity) => {
  if (entity.kind === "Hostile mobs" && bot.entity && entity.position.distanceTo(bot.entity.position) < 16) {
    event("spawn", `a ${entity.name} appeared nearby`);
  }
});
bot.on("kicked", (reason) => event("info", `kicked: ${reason}`));
bot.on("error", (err) => event("info", `bot error: ${err.message}`));

// ── actions ────────────────────────────────────────────────────────────────
async function runAction(name, params) {
  switch (name) {
    case "move_to": {
      const m = new Movements(bot);
      bot.pathfinder.setMovements(m);
      await bot.pathfinder.goto(new goals.GoalNearXZ(params.x, params.z, 1));
      return "arrived";
    }
    case "turn": {
      const yaw = bot.entity.yaw + (params.degrees * Math.PI) / 180;
      await bot.look(yaw, bot.entity.pitch, true);
      return `turned to yaw ${yaw.toFixed(2)}`;
    }
    case "mine_block": {
      const block = bot.blockAt(new (require("vec3").Vec3)(params.x, params.y, params.z));
      if (!block || block.name === "air") throw new Error("no block there");
      await bot.dig(block);
      return `mined ${block.name}`;
    }
    case "place_block": {
      const ref = bot.blockAt(new (require("vec3").Vec3)(params.x, params.y - 1, params.z));
      if (!ref) throw new Error("no reference block below target");
      await bot.placeBlock(ref, new (require("vec3").Vec3)(0, 1, 0));
      return "placed";
    }
    case "stop": {
      // Clear any residual pathfinder goal. A goto promise PERSISTS across
      // Python-side timeouts and /tp (executor-lens finding 1) — a stale goal
      // self-moves the bot during later placements/training. The harness
      // calls this at phase boundaries.
      bot.pathfinder.setGoal(null);
      return "stopped";
    }
    case "flee": {
      // Wire 4 (Exp 58): species-typical flight — retreat to the SAFE anchor.
      // Param-free (substrate-selectable); canDig=false so the bot cannot
      // tunnel through classroom walls (env lens SF-8).
      // ANCHOR: explicit --flee_x/--flee_z bridge args (the classroom's safe
      // platform, printed by setup_world classroom). bot.spawnPoint is the
      // WORLD spawn from the login packet — /spawnpoint never updates it and
      // it points wherever the world was created (run-harness review DNR-1);
      // it remains only a last-resort fallback for non-classroom use.
      // No position fallback: a goto-to-where-you-stand resolves instantly and
      // would book flight SUCCESS for doing nothing (executor-lens catch).
      const anchor =
        FLEE_X !== null && FLEE_Z !== null ? { x: FLEE_X, z: FLEE_Z } : bot.spawnPoint;
      if (!anchor) throw new Error("no flee anchor (pass --flee_x/--flee_z)");
      const fm = new Movements(bot);
      fm.canDig = false;        // no tunnelling through classroom walls (env SF-8)
      fm.canOpenDoors = true;   // open the dark-chamber door to flee (keeps it
                                // closed otherwise, so the dark chamber stays sealed)
      bot.pathfinder.setMovements(fm);
      await bot.pathfinder.goto(new goals.GoalNearXZ(anchor.x, anchor.z, 2));
      return "fled to anchor";
    }
    case "escape_water": {
      // Exp 60: escape drowning by swimming UP. The pathfinder is DEAD in water
      // (mineflayer-pathfinder move generators hard-return on liquid nodes, so
      // `flee`/goto throws NoPath from a submerged start — env-lens E1). This
      // bypasses the pathfinder entirely: hold the `jump` control (which is
      // swim-up while submerged) until the HEAD block is air again, then
      // release. Oxygen recovers game-natively once surfaced. Param-free.
      const headWater = () => {
        const e = bot.entity;
        if (!e) return false;
        const b = bot.blockAt(e.position.offset(0, 1, 0));
        return b && (b.name === "water" || b.name === "bubble_column");
      };
      if (!headWater()) return "already at surface";
      // A surface is a BREATH, not a tick: keep swimming up for SURFACE_HOLD_MS
      // after the head first clears so the surfaced state persists for at
      // least one STATE_INTERVAL_MS snapshot (the body can observe air; the
      // game refills air only while the eyes are out) — releasing jump the
      // same tick the head clears sank the bot back within a few ticks, an
      // unobservable "surface" (Exp 60 chunk-i executor-lens fold).
      const SURFACE_HOLD_MS = 600;
      bot.setControlState("jump", true);
      try {
        await new Promise((res) => {
          let waited = 0;
          let clearedAt = null;
          const iv = setInterval(() => {
            waited += 100;
            if (clearedAt === null && !headWater()) clearedAt = waited;
            // stop after the post-clear hold, or a hard 8s cap (never hang the
            // action loop — a walled column with no reachable air would hang)
            if ((clearedAt !== null && waited - clearedAt >= SURFACE_HOLD_MS) || waited >= 8000) {
              clearInterval(iv);
              res();
            }
          }, 100);
        });
      } finally {
        bot.setControlState("jump", false);
      }
      return headWater() ? "surface: still submerged (capped)" : "surfaced";
    }
    case "eat": {
      const item = bot.inventory.items().find((i) => i.name.includes("bread") || i.foodPoints);
      if (!item) throw new Error("no food in inventory");
      const before = bot.food;
      await bot.equip(item, "hand");
      await bot.consume();
      // bot.food is updated by a server packet that lands a tick or two AFTER
      // consume() resolves, so the action_result snapshot() below would otherwise
      // read stale food and the relief would surface on the NEXT action — which
      // mis-attributes the eat's measured-relief credit (break 2) to whatever the
      // agent did next in a multi-action loop. Poll until bot.food ACTUALLY changes
      // (a fixed-delay fallback flickered near the food cap when the packet was slow),
      // capped at 1.5s so we never hang. (Eat-local; other affordances and Exp 56's
      // roster untouched.)
      await new Promise((res) => {
        if (bot.food !== before) return res();
        let waited = 0;
        const iv = setInterval(() => {
          waited += 50;
          if (bot.food !== before || waited >= 1500) {
            clearInterval(iv);
            res();
          }
        }, 50);
      });
      return `ate ${item.name}`;
    }
    case "attack_nearest": {
      const target = bot.nearestEntity((e) => e.kind === "Hostile mobs");
      if (!target) throw new Error("no hostile nearby");
      await bot.attack(target);
      return `attacked ${target.name}`;
    }
    default:
      throw new Error(`unknown action: ${name}`);
  }
}

// ── bridge server ──────────────────────────────────────────────────────────
const server = net.createServer((sock) => {
  if (client && !client.destroyed) {
    sock.end(JSON.stringify({ type: "event", kind: "error", text: "bridge busy: one client at a time" }) + "\n");
    return;
  }
  client = sock;
  console.log("bridge client connected");
  let buffer = "";
  sock.on("data", (chunk) => {
    buffer += chunk.toString("utf8");
    let idx;
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 1);
      if (!line.trim()) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue;
      }
      if (msg.type === "action") {
        runAction(msg.name, msg.params || {})
          .then((detail) =>
            send({ type: "action_result", id: msg.id, ok: true, detail, state: snapshot() })
          )
          .catch((err) =>
            send({ type: "action_result", id: msg.id, ok: false, detail: err.message, state: snapshot() })
          );
      }
    }
  });
  sock.on("close", () => {
    if (client === sock) {
      client = null;
      console.log("bridge client disconnected");
    }
  });
  sock.on("error", () => {});
});

bot.once("spawn", () => {
  server.listen(BRIDGE_PORT, "127.0.0.1", () => {
    console.log(`maxim minecraft bridge: game ${MC_HOST}:${MC_PORT} <-> tcp 127.0.0.1:${BRIDGE_PORT}`);
  });
  // One state line to stdout (after chunks/light settle) so an operator can see the bot
  // is alive and sensors read without attaching a TCP client; state otherwise flows
  // only over the bridge socket.
  setTimeout(() => console.log("spawn state: " + JSON.stringify(snapshot())), 2000);
  setInterval(() => send({ type: "state", data: snapshot() }), STATE_INTERVAL_MS);
  event("info", "player spawned");
});
