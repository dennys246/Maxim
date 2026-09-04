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

const bot = mineflayer.createBot({ host: MC_HOST, port: MC_PORT, username: USERNAME });
bot.loadPlugin(pathfinder);

let client = null; // one Maxim at a time (the two-AUT harness runs two bridges)

function send(obj) {
  if (client && !client.destroyed) client.write(JSON.stringify(obj) + "\n");
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
  const vel = me ? me.velocity : null;
  const speed = vel ? Math.min(1, Math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)) : 0;
  return {
    health: bot.health ?? 20,
    food: bot.food ?? 20,
    saturation: Math.min(10, bot.foodSaturation ?? 5),
    oxygen: bot.oxygenLevel ?? 20,
    light_level: bot.world && me ? (bot.world.getBlockLight?.(me.position) ?? 7) : 7,
    y_altitude: me ? me.position.y : 64,
    nearest_hostile_dist: nearest,
    hostile_count: Math.min(32, hostiles.length),
    nearest_player_dist: nearestPlayer,
    distance_from_spawn: distSpawn,
    speed,
    on_ground: me && me.onGround ? 1 : 0,
    is_raining: bot.isRaining ? 1 : 0,
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
    case "eat": {
      const food = bot.inventory.items().find((i) => i.name.includes("bread") || i.foodPoints);
      if (!food) throw new Error("no food in inventory");
      await bot.equip(food, "hand");
      await bot.consume();
      return `ate ${food.name}`;
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
    if (client === sock) client = null;
  });
  sock.on("error", () => {});
});

bot.once("spawn", () => {
  server.listen(BRIDGE_PORT, "127.0.0.1", () => {
    console.log(`maxim minecraft bridge: game ${MC_HOST}:${MC_PORT} <-> tcp 127.0.0.1:${BRIDGE_PORT}`);
  });
  setInterval(() => send({ type: "state", data: snapshot() }), STATE_INTERVAL_MS);
  event("info", "player spawned");
});
