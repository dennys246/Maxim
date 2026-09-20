#!/usr/bin/env python3
"""1.3 survival world — stand up the break-3 apparatus (R2 break 3: the world affords the acts).

R2's null had three structural breaks (docs/experiments/r2_drive_premise_check.md). Break 1
(the drive->corrective-action prior) and break 2 (measured-relief credit for interoceptive
world drives) shipped. Break 3 is the world layer: the void/superflat contingency world the
1.2 apparatus used does NOT afford the corrective acts (`eat` throws "no food in inventory",
`attack_nearest` throws "no hostile nearby"). This stands up a SURVIVAL world where the game's
own hunger mechanic drains (game-native pressure, D1: no synthetic sensor, no bespoke reward)
and food is present in the inventory, so `eat` is a single executable corrective act and the
break-1 prior -> break-2 credit loop can actually close.

This is a DIFFERENT apparatus from Exp 56: a survival `minecraft_player` world, NOT the frozen
`minecraft_bench` superflat contingency world (mobs off, floating slots). They share no world,
server, or body — so standing this up does not touch the Exp 56 EARNED result. It reuses
Exp 56's proven mechanics where they carry: the `fill.papermc.io/v3` download (the old
`api.papermc.io/v2` returns 410 Gone) and `exp56.common.RconControl` for the world script.

Version: Paper 1.20.4 (DECIDED 2026-09-13 — docs/plans/archive/survival_world_1_3.md §"Minecraft
version"; the 1.3 platform, ported off the 1.16.5 the break-3 smoke first ran on). mineflayer
`^4.20.0` negotiates the protocol, so the bridge needs no version change. Design-relevant
mechanic delta vs 1.16.5: since 1.18, hostile mobs spawn only at block-light 0 (not <=7), which
sharpens the dark=danger contingency the Phase-1 classroom keys on. `scripts/exp56/` stays
1.16.5-pinned until its own re-baseline port.

Subcommands
-----------
``setup``    download Paper 1.20.4, write survival ``server.properties`` + ``eula.txt``
             (EULA acceptance needs the explicit ``--accept-eula`` flag — Mojang's agreement,
             not this script's to accept silently), print the start command.
``prepare``  against a RUNNING server WITH THE BRIDGE BOT JOINED: set time/gamerules and seed
             the bot's inventory with food over RCON. Idempotent. ``--induce-hunger`` applies a
             short game-native hunger effect to drive the loop quickly during WIRING tests (the
             real learning run lets hunger drain naturally).
``verify``   re-check difficulty/gamerules and that the bot holds food (exit 4 on mismatch).
``classroom``  build the Exp 58 dark-fear cave (depth + persistent hostile) at the recorded anchor.
``water_classroom``  build the Exp 60 drowning classroom (dry shore + walled source-water column,
             open top = the escape) at the recorded anchor; refuses within 72 blocks of the Exp 58
             cave (hostile horizon); records ``~/.maxim/exp60_water_classroom.json``. Live
             apparatus check: ``scripts/survival_world/exp60_water_check.py``.

Typical flow (Java 17+ — Paper 1.20.4 REQUIRES it; temurin@17 coexists with the temurin@11
the pinned exp56 server still needs):

    brew install --cask temurin@17
    python scripts/survival_world/setup_world.py setup --dir ~/maxim-mc-survival-1.20 \\
        --accept-eula --rcon-password '<pw>'
    # macOS `java` may resolve to an older default JDK (the pinned exp56 Java 11) even with
    # 17 installed — pin per-invocation via java_home instead of changing the default:
    (cd ~/maxim-mc-survival-1.20 && "$(/usr/libexec/java_home -v 17)/bin/java" \\
        -jar paper-1.20.4.jar nogui)     # first boot
    # in another terminal, connect the bridge so the bot 'maxim' joins:
    (cd scripts/minecraft_bridge && node index.js --mc_host=127.0.0.1 --mc_port=25565 \\
        --bridge_port=25567 --username=maxim)
    python scripts/survival_world/setup_world.py prepare --rcon-password '<pw>' --username maxim
    python scripts/survival_world/setup_world.py verify  --rcon-password '<pw>' --username maxim
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from _paper_server import download_paper, refuse_stale_world, write_world_stamp  # noqa: E402
from exp56.common import RconControl  # noqa: E402  (proven RCON client — shared dev tooling)

MC_VERSION = "1.20.4"
# PaperMC's Fill v3 API (the old api.papermc.io v2 returns 410 Gone — exp56 lesson).
PAPER_API = f"https://fill.papermc.io/v3/projects/paper/versions/{MC_VERSION}/builds/latest"

# level-type=minecraft\:normal (a normal world — terrain, animals, food; 1.19+ takes the
# NAMESPACED id, the pre-1.19 bare `DEFAULT` is not a valid modern value) so
# `eat`/`attack_nearest` have something to act on; a fixed seed so the world is reproducible.
# difficulty=normal (NOT peaceful) so the game's hunger mechanic drains and mobs afford the
# threat channel. NOTE the literal `\:` — server.properties is a Java .properties file, where
# an unescaped `:` in a VALUE is tolerated by most parsers but the escaped form is canonical;
# Paper writes it back escaped, so writing it escaped keeps setup idempotent under diff.
SERVER_PROPERTIES = """\
# 1.3 survival world (generated by scripts/survival_world/setup_world.py). R2 break 3:
# a survival world whose native hunger mechanic drains (game-native pressure) and whose
# inventory can hold food, so the corrective `eat` act is executable.
online-mode=false
enable-rcon=true
rcon.port={rcon_port}
rcon.password={rcon_password}
level-type=minecraft\\:normal
level-seed=maxim-oasis-1
level-name=survival_world
gamemode=survival
difficulty=normal
spawn-monsters=true
spawn-animals=true
allow-nether=false
spawn-protection=0
view-distance=8
max-players=4
motd=Maxim Oasis survival world (R2 break 3)
enable-command-block=false
"""


def _setup(args: argparse.Namespace) -> int:
    server_dir = Path(args.dir).expanduser()
    server_dir.mkdir(parents=True, exist_ok=True)
    # Stale-world guard (shared helper, scripts/_paper_server.py): booting a new Paper version over
    # a world generated by an older one silently UPGRADES it in place — pre-1.18 terrain gets
    # stitched to new-gen chunks and the world is no longer reproducible from its seed. A version
    # stamp is written on every successful setup; a world folder with no stamp or a different
    # stamped version refuses setup (fresh --dir is the right fix; --force-world overrides).
    refusal = refuse_stale_world(server_dir, "survival_world", MC_VERSION, force=args.force_world)
    if refusal:
        print(refusal)
        return 2
    if not args.accept_eula:
        print(
            "Refusing to write eula.txt without --accept-eula: accepting Mojang's EULA\n"
            "(https://aka.ms/MinecraftEULA) is the operator's action, not this script's."
        )
        return 2
    jar = server_dir / f"paper-{MC_VERSION}.jar"
    if not download_paper(MC_VERSION, jar):
        return 1
    (server_dir / "eula.txt").write_text("eula=true\n")
    (server_dir / "server.properties").write_text(
        SERVER_PROPERTIES.format(rcon_port=args.rcon_port, rcon_password=args.rcon_password)
    )
    write_world_stamp(server_dir, MC_VERSION, stamped_by="scripts/survival_world/setup_world.py")
    print(
        f"\nsetup complete in {server_dir}\n"
        f"  Java 17+ required (Paper {MC_VERSION}; `brew install --cask temurin@17`).\n"
        f"  first boot (java_home pins 17 even when the shell default is the exp56 Java 11):\n"
        f'  (cd {server_dir} && "$(/usr/libexec/java_home -v 17)/bin/java" '
        f"-jar paper-{MC_VERSION}.jar nogui)\n"
        f"  then connect the bridge (bot joins as the username you pass to prepare), then:\n"
        f"  python scripts/survival_world/setup_world.py prepare "
        f"--rcon-port {args.rcon_port} --rcon-password '<pw>' --username maxim"
    )
    return 0


# Idempotent world conditions. doImmediateRespawn so a death does not park the AUT on a
# respawn screen; keepInventory so a death does not silently strip the seeded food (which
# would masquerade as "eat unavailable"); time frozen at day for reproducible conditions.
_GAMERULES = {
    "doImmediateRespawn": "true",
    "keepInventory": "true",
    "doDaylightCycle": "false",
    # R3 belt: a multi-hour campaign exceeds the insomnia threshold; phantoms are already blocked
    # by the frozen day and doMobSpawning, but the rule is cheap and VERIFIED by R3's roster.
    "doInsomnia": "false",
    # Exp 58 env lens SF-1: storms make the daytime surface spawnable, rain
    # extinguishes burning zombies (the pursuit-containment mechanic), and
    # `is_raining` flips cluster identity mid-arm.
    "doWeatherCycle": "false",
    # Belt (env NIT): zombies must not break the classroom geometry.
    "mobGriefing": "false",
}


def _prepare(args: argparse.Namespace) -> int:
    rcon = RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    try:
        print(f"rcon> difficulty normal\n      {rcon.command('difficulty normal').strip() or '(ok)'}")
        print(f"rcon> time set day\n      {rcon.command('time set day').strip() or '(ok)'}")
        for rule, val in _GAMERULES.items():
            resp = rcon.command(f"gamerule {rule} {val}").strip()
            print(f"rcon> gamerule {rule} {val}\n      {resp or '(ok)'}")
        # Seed food so `eat` is a single executable corrective act. Bread is a stable,
        # non-perishable food the bridge's `eat` handler already matches by name.
        give = f"give {args.username} minecraft:bread 64"
        resp = rcon.command(give).strip()
        print(f"rcon> {give}\n      {resp or '(ok)'}")
        if "no player" in resp.lower() or "was found" in resp.lower():
            print(
                f"\nWARNING: no player named {args.username!r} online — start the bridge first "
                f"(node index.js --username={args.username}) so the bot joins, then re-run prepare."
            )
            return 4
        if args.induce_hunger:
            # A game-native hunger effect to drive the loop quickly during WIRING tests only.
            # The real learning run lets hunger drain from ordinary activity (D1).
            hun = f"effect give {args.username} minecraft:hunger 40 4"
            print(f"rcon> {hun}\n      {rcon.command(hun).strip() or '(ok)'}")
        print("\nprepare complete — the bot holds food and hunger will drain (game-native).")
        return 0
    finally:
        rcon.close()


def _verify(args: argparse.Namespace) -> int:
    rcon = RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    bad = 0
    try:
        for rule, val in _GAMERULES.items():
            resp = rcon.command(f"gamerule {rule}").strip()
            ok = val in resp
            print(f"gamerule {rule}: {resp}  {'OK' if ok else 'MISMATCH (want ' + val + ')'}")
            bad += 0 if ok else 1
        # `clear <player> bread 0` reports the count without removing (vanilla dry-run form).
        # Replies "Found N matching item(s) on player X" (has food) or "No items were found
        # on player X" (none) — the commands.clear.testing.* translation keys, unchanged
        # 1.16→1.20.4 (re-confirm on the first 1.20.4 boot) — match the FAILURE string, not a
        # "0 item" substring that never appears in either reply (which would make this guard
        # vacuous).
        cnt = rcon.command(f"clear {args.username} minecraft:bread 0").strip()
        low = cnt.lower()
        holds = "no items were found" not in low and "no player" not in low
        print(f"food held by {args.username}: {cnt}  {'OK' if holds else 'NONE — re-run prepare'}")
        bad += 0 if holds else 1
    finally:
        rcon.close()
    if bad:
        print(f"\nVERIFY FAILED ({bad} mismatch) — the world does not yet afford the corrective act.")
        return 4
    print("\nverify OK — the survival world affords `eat`; break-3 loop can close.")
    return 0


def _classroom(args: argparse.Namespace) -> int:
    """Build the Exp 58 dark-fear CAVE classroom over RCON (prereg §Apparatus).

    DEPTH-based danger cluster (owner decision 2026-09-14, Addendum 3). The
    ``light_level`` sensor proved UNRELIABLE in this world — spatially patchy,
    inconsistent run-to-run, and skylight-contaminated underground (a buried
    cell read 14 at day / 0 at night; neighbours read block-light values with no
    source). Paper/mineflayer light here cannot carry the contingency. So the
    danger cluster is discriminated by ``y_altitude`` — the bot's own position,
    read straight from the entity with NO lighting engine involved (100%
    reliable) — plus hostile presence. Still a game-native cave: descend deep to
    where the monsters are.

    Geometry, encased in one solid stone cuboid at depth (the AUT is teleported
    in for placements/training; it FLEES up the staircase):

    - SAFE chamber (upper, floor y=SAFE_Y): carved air, anchor + spawnpoint +
      flee target. No hostiles (kept swept).
    - STAIRCASE: 1-up-1-over stone steps (2 wide, 2-high headroom) from the safe
      chamber down-east to the dark pit — pathfinder climbs it to flee.
    - DARK pit (lower, floor y=DARK_Y): carved air, zombie SPAWNER at the far
      end (spawners ignore ``doMobSpawning false``; NBT pins adult + density +
      proximity). Deep + hostile = the danger cluster.

    No door, no light source, no relight — nothing depends on the broken
    skylight path. ``--sweep`` kills classroom-range zombies (arm boundaries /
    probe windows). Idempotent: rebuilds IN PLACE at the recorded anchor.
    """
    from survival_world.common import bot_pos

    SAFE_Y = 40
    DARK_Y = 28
    MID_Y = (SAFE_Y + DARK_Y) // 2  # 34 — the "exited the dark" boundary the harness uses

    rcon = RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    try:
        if args.sweep:
            # Spare the persistent cluster-mob (Addendum 5): it is part of the
            # apparatus (it gives the danger cluster its reliable hostile axis),
            # not spawner spillover to clear.
            resp = rcon.command("kill @e[type=minecraft:zombie,tag=!exp58clustermob,distance=..64]")
            print(f"rcon> kill zombies (spare clustermob) r64\n      {resp.strip() or '(none)'}")
            return 0
        anchor_file = Path.home() / ".maxim" / "exp58_classroom.json"
        if args.anchor_x is not None:
            ax, _ay, az = args.anchor_x, args.anchor_y, args.anchor_z
        elif anchor_file.exists():
            ax, _ay, az = json.loads(anchor_file.read_text())["anchor"]
        else:
            ax, _ay, az = bot_pos(rcon, args.username)
        ax, az = int(ax), int(az)
        # Layout along +x (east): safe chamber, staircase down, LONG dark pit.
        #   safe interior  x [ax-3 .. ax+1], floor SAFE_Y   (anchor at ax-1)
        #   staircase       x  ax+2 .. ax+13, floor SAFE_Y-1 .. DARK_Y (1 down per x)
        #   dark pit        x  ax+14 .. ax+27 (14 long), floor DARK_Y
        #     clustermob at ax+18, dark_x (placement/encode) ax+20, spawner ax+25
        anchor_x = ax - 1
        stair_x0 = ax + 2
        n_steps = SAFE_Y - DARK_Y  # 12
        pit_x0 = ax + 14
        pit_x1 = ax + 27  # ADEQUATELY LONG pit (14 blocks) — room for the danger
        # zone, deep separation, and headroom to add a deeper TREASURE layer in the
        # follow-up experiment (see docs/experiments/exp59_layered_cave_prereg.md).
        clustermob_x = ax + 18
        dark_x = ax + 20  # deep in the pit, far from the stairs (big position + depth signal)
        spawn_x = ax + 25
        cmds = [
            # FORCELOAD the classroom chunks so they stay resident with NO player
            # online: RCON `summon` into an UNLOADED chunk silently relocates the
            # entity to the always-loaded spawn chunks (the clustermob landed at
            # world spawn until this) — and the persistent clustermob + spawner must
            # survive between runs regardless of where the bot is. Must precede the
            # summon so the pit chunk is loaded when the mob is placed.
            f"forceload add {ax - 6} {az - 4} {pit_x1 + 3} {az + 4}",
            # ENCASE the whole footprint (+ margin) in solid stone — overwrites
            # natural cave/lava/water so every chamber has clean, sky-sealed
            # walls (skylight 0 by burial, though we no longer depend on it).
            f"fill {ax - 6} {DARK_Y - 2} {az - 4} {pit_x1 + 3} {SAFE_Y + 4} {az + 4} minecraft:stone",
            # Carve SAFE chamber (5 long x 3 wide x 3 high air) at SAFE_Y.
            f"fill {ax - 3} {SAFE_Y} {az - 1} {ax + 1} {SAFE_Y + 2} {az + 1} minecraft:air",
            # Carve the LONG DARK pit (3 wide x 3 high air) at DARK_Y.
            f"fill {pit_x0} {DARK_Y} {az - 1} {pit_x1} {DARK_Y + 2} {az + 1} minecraft:air",
        ]
        # Carve the staircase: step i (0..n_steps) at x = stair_x0+i, floor y =
        # (SAFE_Y-1)-i, with 2-high headroom above and 2 wide (z-1..z). The floor
        # stays stone (from the encase); we carve the two air blocks a walker
        # occupies. 1-up-1-over is standard pathfinder-climbable stairs.
        for i in range(n_steps + 2):
            sx = stair_x0 + i
            fy = (SAFE_Y - 1) - i
            if fy < DARK_Y:
                fy = DARK_Y
            cmds.append(f"fill {sx} {fy + 1} {az - 1} {sx} {fy + 2} {az + 1} minecraft:air")
        cmds += [
            # Zombie spawner at the far end of the pit (AI attackers → training damage).
            f"setblock {spawn_x} {DARK_Y} {az} minecraft:spawner"
            + '{SpawnData:{entity:{id:"minecraft:zombie",IsBaby:0b}},'
            + "MaxNearbyEntities:3s,RequiredPlayerRange:12s,SpawnCount:2s,"
            + "MinSpawnDelay:80s,MaxSpawnDelay:200s,SpawnRange:4s}",
            # Clear any prior clustermob (rebuild), then summon THE persistent one:
            # a NoAI zombie that never moves/attacks/despawns — it exists only to
            # give the danger cluster its reliable HOSTILE axis at encode/probe
            # time, so depth + hostiles is a stable multi-axis contrast vs the safe
            # chamber (Addendum 5 — L11 dilution made depth-alone too weak/jittery).
            "kill @e[type=minecraft:zombie,tag=exp58clustermob]",
            f"summon minecraft:zombie {clustermob_x} {DARK_Y} {az} "
            + '{NoAI:1b,PersistenceRequired:1b,Silent:1b,IsBaby:0b,Tags:["exp58clustermob"]}',
            # Global: natural spawning OFF (spawner + clustermob are the only mobs).
            "gamerule doMobSpawning false",
            # Bot respawns in the safe chamber.
            f"spawnpoint {args.username} {anchor_x} {SAFE_Y} {az}",
            # Death accounting for the harness's death cap.
            "scoreboard objectives add exp58_deaths deathCount",
        ]
        for cmd in cmds:
            resp = rcon.command(cmd).strip()
            print(f"rcon> {cmd[:96]}\n      {resp or '(ok)'}")
            low = resp.lower()
            if "unknown" in low or "expected" in low or "incorrect" in low:
                print("\nCLASSROOM BUILD FAILED on the command above — nothing gated may run.")
                return 4
        # VERIFY the clustermob actually landed in the pit (not relocated to spawn
        # by an unloaded chunk — the world-spawn bug forceload fixes). Refuse the
        # build if it is not within 3 blocks of its intended pit cell, so a
        # mis-placed hostile can never quietly leave the danger cluster unformed.
        check = rcon.command(
            f"execute if entity @e[type=minecraft:zombie,tag=exp58clustermob,"
            f"x={clustermob_x},y={DARK_Y},z={az},distance=..3]"
        )
        if "passed" not in check.lower():
            print(
                f"\nCLASSROOM BUILD FAILED: clustermob not in the pit at "
                f"({clustermob_x},{DARK_Y},{az}) — got {check.strip()!r}. "
                "Is the bridge bot online so the pit chunk loads? (forceload should cover it.)"
            )
            return 4
        anchor_file.parent.mkdir(parents=True, exist_ok=True)
        geom = {
            "anchor": [anchor_x, SAFE_Y, az],
            "dark": [dark_x, DARK_Y, az],
            "safe_y": SAFE_Y,
            "dark_y": DARK_Y,
            "mid_y": MID_Y,
            "flee_x": anchor_x,
            "flee_z": az,
        }
        anchor_file.write_text(json.dumps(geom, indent=2))

        print(
            f"\ndepth cave built: SAFE chamber floor y={SAFE_Y} (anchor ({anchor_x},{SAFE_Y},{az})), "
            f"staircase down-east, LONG DARK pit floor y={DARK_Y} (dark_x={dark_x}), persistent "
            f"clustermob at ({clustermob_x},{DARK_Y},{az}), spawner at ({spawn_x},{DARK_Y},{az}). "
            f"Danger cluster = depth (y_altitude) + RELIABLE hostile (the clustermob); light NOT "
            f"used. 'Exited dark' = y_altitude > {MID_Y}.\n"
            f"geometry recorded -> {anchor_file}.\n"
            f"START THE BRIDGE WITH THE FLEE ANCHOR: --flee_x={anchor_x} --flee_z={az}"
        )
        return 0
    finally:
        rcon.close()


# ───────────────────────── Exp 60 water classroom (drowning-avoidance) ──────────────────────

WATER_SHORE_Y = 40  # the Exp 58 depth band: the offline cos~0.79 estimate was computed on a
# base vector captured HERE (light 0, frozen day, y~40) — a surface pool would change
# light_level and re-open the "estimated on a different geometry" trap.
WATER_DEPTH_DEFAULT = 5  # 4-block head ascent; shore/submerged y differ by > the probe's 3-block settle
WATER_MIN_DEPTH, WATER_MAX_DEPTH = 3, 12  # < 3: the floor-placed head is not in water
# The bridge caps nearest_hostile_dist at 64 ("beyond horizon") and 64 is that sensor's
# neutral midpoint (range [0,128]). Exp 58's persistent clustermob must therefore sit
# BEYOND the cap from the pool, or both water situations carry constant hostile mass.
WATER_MIN_DIST_FROM_EXP58 = 72
# ...and the SAME sensor class the other way: `distance_from_spawn` (3D distance to WORLD
# spawn, capped at 128, range [-128,128] → neutral at 0) becomes a full-weight CONSTANT in
# both water situations when the pool is far from spawn — the offline cos≈0.79 estimate
# never modelled it (its base vector sat 36 blocks from spawn). Replayed on the real encoder
# bases (exp60_spawn_distance_check.py in the experiments data dir): cos(shore, submerged) 0.786
# @36 blocks, 0.794 @90, 0.802 @100, 0.834 @120, 0.8525 @128 = SAME cluster. Bound: 90 (3D).
# Architecture-lens fold (Exp 60 chunk i); `offset_x/z` are bridge-only, not body sensors.
WATER_MAX_DIST_FROM_SPAWN = 90
WATER_ANCHOR_FILE = Path.home() / ".maxim" / "exp60_water_classroom.json"
EXP58_ANCHOR_FILE = Path.home() / ".maxim" / "exp58_classroom.json"


def water_classroom_geometry(
    ax: int, az: int, *, depth: int = WATER_DEPTH_DEFAULT, shore_y: int = WATER_SHORE_Y
) -> dict:
    """Pure geometry of the Exp 60 water classroom (no I/O) — unit-tested structurally.

    One stone-encased room at the Exp 58 depth band: a DRY shore platform (rest /
    rescue / spawnpoint) beside a walled 3x3 SOURCE-water column ``depth`` deep whose
    open top is the only air — the reachable escape. The dive target is the pool
    FLOOR so dive-second-0 already reads ``is_in_water`` 1 with full air (the 0-15 s
    pre-damage window starts clean); the head must rise ``depth - 1`` blocks to air.
    Cuboids are inclusive ``(x0, y0, z0, x1, y1, z1)``.
    """
    if not (WATER_MIN_DEPTH <= depth <= WATER_MAX_DEPTH):
        raise ValueError(f"depth must be in [{WATER_MIN_DEPTH}, {WATER_MAX_DEPTH}], got {depth}")
    ax, az = int(ax), int(az)
    pool_x0, pool_x1 = ax + 3, ax + 5
    pool_floor_y = shore_y - depth - 1
    shell = (ax - 6, pool_floor_y - 2, az - 4, ax + 9, shore_y + 5, az + 4)
    chamber = (ax - 3, shore_y, az - 1, pool_x1, shore_y + 2, az + 1)  # air over shore + pool
    pool = (pool_x0, shore_y - depth, az - 1, pool_x1, shore_y - 1, az + 1)  # water sources
    shore = (ax - 1, shore_y, az)  # feet in air, standing on stone at shore_y-1
    pool_cx = ax + 4
    submerged = (pool_cx, shore_y - depth, az)  # feet on the pool floor, head in water
    return {
        "shore_y": shore_y,
        "depth": depth,
        "shell": shell,
        "chamber": chamber,
        "pool": pool,
        "shore": shore,
        "shore_floor": (ax - 1, shore_y - 1, az),
        "lip": (ax + 2, shore_y - 1, az),  # stone between the shore floor and the top water layer
        "submerged": submerged,
        "submerged_head": (pool_cx, shore_y - depth + 1, az),
        "surface": (pool_cx, shore_y, az),  # first air cell above the pool centre
        "pool_floor": (pool_cx, pool_floor_y, az),
        "forceload": (shell[0], shell[2], shell[3], shell[5]),
    }


def _cuboid_cmd(verb: str, c: tuple, block: str) -> str:
    x0, y0, z0, x1, y1, z1 = c
    return f"{verb} {x0} {y0} {z0} {x1} {y1} {z1} {block}"


def water_classroom_commands(geom: dict, username: str) -> list[str]:
    """Ordered RCON build commands. forceload FIRST (fluid ticks + fills need resident chunks)."""
    fx0, fz0, fx1, fz1 = geom["forceload"]
    sx, sy, sz = geom["shore"]
    return [
        f"forceload add {fx0} {fz0} {fx1} {fz1}",
        # ENCASE (overwrites natural caves/lava/water): every pool face becomes stone.
        _cuboid_cmd("fill", geom["shell"], "minecraft:stone"),
        # Carve the air chamber over shore + pool (3 high: headroom for a floating bot).
        _cuboid_cmd("fill", geom["chamber"], "minecraft:air"),
        # The walled SOURCE-water column (open top into the chamber = the escape).
        _cuboid_cmd("fill", geom["pool"], "minecraft:water"),
        # Apparatus-owned world conditions: no natural spawns (block-light-0 water spawns
        # DROWNED zombies = a second pain source); respawn on the dry shore; death accounting.
        "gamerule doMobSpawning false",
        f"spawnpoint {username} {sx} {sy} {sz}",
        "scoreboard objectives add exp60_deaths deathCount",
    ]


def water_classroom_verifications(geom: dict) -> list[tuple[str, str]]:
    """``(execute-if-block command, what it proves)`` — every one must reply 'passed'."""
    hx, hy, hz = geom["submerged_head"]
    ux, uy, uz = geom["surface"]
    sx, sy, sz = geom["shore"]
    fx, fy, fz = geom["shore_floor"]
    px, py, pz = geom["pool_floor"]
    lx, ly, lz = geom["lip"]
    bx, by, bz = geom["submerged"]
    x0, _y0, z0, _x1, y1, _z1 = geom["pool"]
    return [
        (f"execute if block {hx} {hy} {hz} minecraft:water", "submerged head cell is water"),
        (f"execute if block {ux} {uy} {uz} minecraft:air", "surface cell above the pool is air"),
        (f"execute if block {sx} {sy} {sz} minecraft:air", "shore feet cell is air (dry)"),
        (f"execute if block {sx} {sy + 1} {sz} minecraft:air", "shore head cell is air"),
        (f"execute if block {fx} {fy} {fz} minecraft:stone", "shore floor is stone"),
        (f"execute if block {px} {py} {pz} minecraft:stone", "pool floor is stone"),
        (f"execute if block {lx} {ly} {lz} minecraft:stone", "lip between shore and water is stone"),
        # SOURCE blocks (level=0) after the fluid-tick pause: a drained/flowing column
        # would read level>0 or air here.
        (f"execute if block {bx} {by} {bz} minecraft:water[level=0]", "bottom water cell is a source"),
        (f"execute if block {ux} {y1} {uz} minecraft:water[level=0]", "top water cell is a source"),
        (f"execute if block {x0} {y1} {z0} minecraft:water[level=0]", "corner top water cell is a source"),
    ]


def exp58_clearance(geom: dict, exp58_geom: dict | None) -> tuple[bool, float]:
    """(ok, min horizontal distance) from the pool centre to the Exp 58 anchor and pit.

    The clustermob stands within 2 blocks of the recorded ``dark`` point; both points
    must lie beyond the bridge's 64-block hostile horizon plus margin, or
    ``nearest_hostile_dist``/``hostile_count`` carry constant mass in BOTH water
    situations (the exact L11 dilution shape). No Exp 58 record → nothing to clear.
    """
    if not exp58_geom:
        return True, float("inf")
    cx, _cy, cz = geom["submerged"]
    dmin = float("inf")
    for key in ("anchor", "dark"):
        pt = exp58_geom.get(key)
        if not pt:
            continue
        dmin = min(dmin, ((cx - float(pt[0])) ** 2 + (cz - float(pt[2])) ** 2) ** 0.5)
    return dmin >= WATER_MIN_DIST_FROM_EXP58, dmin


def spawn_clearance(geom: dict, spawn: tuple[float, float, float] | None) -> tuple[bool, float | None]:
    """(ok, 3D distance) from the submerged target to WORLD spawn; ``(True, None)`` when unknown.

    World spawn is the login-packet spawn the bridge measures ``distance_from_spawn``
    against (``/spawnpoint`` never moves it) and is not readable over RCON, so the
    builder checks it only when the operator passes ``--spawn-x/y/z``; the live check
    gates the SENSED value regardless (``exp60_water_check`` W1, <= 90).
    """
    if spawn is None:
        return True, None
    bx, by, bz = geom["submerged"]
    sx, sy, sz = (float(v) for v in spawn)
    d = ((bx - sx) ** 2 + (by - sy) ** 2 + (bz - sz) ** 2) ** 0.5
    return d <= WATER_MAX_DIST_FROM_SPAWN, d


def record_shell(record: dict) -> tuple[int, int, int, int, int, int] | None:
    """This pool's shell cuboid — stamped, or DERIVED for a record written before the stamp.

    Pool 1's record on the rig predates ``shell``. Without this the two-pool guard would take its
    "cannot check" branch on the one record it exists to check, and the obvious remedy (rebuild
    pool 1) replaces the record wholesale and drops the ``measured`` block that Exp 60's harness and
    R3's gauntlet refuse to run without. Deriving costs nothing and keeps the apparatus untouched.
    """
    shell = record.get("shell")
    if shell is not None:
        if not (
            isinstance(shell, (list, tuple)) and len(shell) == 6 and all(isinstance(v, (int, float)) for v in shell)
        ):
            raise ValueError(f"record has a malformed shell: {shell!r}")
        return tuple(int(v) for v in shell)  # type: ignore[return-value]
    try:  # the same geometry this module builds: shore x is anchor-1, so the anchor is shore x + 1
        shore = record["shore"]
        return tuple(  # type: ignore[return-value]
            water_classroom_geometry(
                int(shore[0]) + 1, int(shore[2]), depth=int(record["depth"]), shore_y=int(record["surface_y"])
            )["shell"]
        )
    except (KeyError, TypeError, ValueError):
        return None


def pools_disjoint(geom: dict, other_record: dict | None) -> tuple[bool, str]:
    """Exp 62 two-pool guard: a second pool must not intersect the first pool's SHELL.

    Stacked pools share x/z by design (the cross-pool cosine is what Exp 62 measures), so the
    only separation is vertical and the shells must not overlap by a single block — a shared
    wall would let one build's fill punch a hole in the other's, and a drained or leaking column
    reads as a null instead of an apparatus failure. ``None`` (no other record) is disjoint:
    nothing to collide with. Pure; unit-tested.
    """
    if other_record is None:
        return True, "no other pool given — NOT CHECKED"
    other = record_shell(other_record)
    if other is None:
        return True, "other record carries neither a shell nor the geometry to derive one — NOT CHECKED"
    ax0, ay0, az0, ax1, ay1, az1 = geom["shell"]
    bx0, by0, bz0, bx1, by1, bz1 = other
    overlap = min(ax1, bx1) >= max(ax0, bx0) and min(ay1, by1) >= max(ay0, by0) and min(az1, bz1) >= max(az0, bz0)
    if overlap:
        return False, f"shells intersect: this {tuple(geom['shell'])} vs recorded {tuple(other)}"
    if min(ax1, bx1) < max(ax0, bx0) or min(az1, bz1) < max(az0, bz0):
        return True, "shells are separated horizontally"
    gap = max(ay0, by0) - min(ay1, by1) - 1  # blocks of solid world between the two shells
    return True, f"vertical gap {gap} block(s) between shells"


def water_anchor_record(
    geom: dict,
    *,
    exp58_clearance_blocks: float | None = None,
    spawn_clearance_blocks: float | None = None,
    world_spawn: tuple[float, float, float] | None = None,
    pool_id: str = "pool1",
    pool_clearance: dict | None = None,
) -> dict:
    """The recorded built truth the check/probe/harness drive off (never live position).

    ``None`` clearances mean NOT CHECKED (record absent / spawn not given) — never "clear".
    """
    shore_x, _shore_y, shore_z = geom["shore"]
    return {
        "_format_version": "1.0",
        # Exp 62: the pool this record describes, and the truths a SECOND pool needs — the shell
        # (the disjointness guard reads it), the flee anchor the bridge must be started with, and
        # the world spawn the distance-from-spawn cap is measured against. `None` = not given,
        # never "clear" (the live check gates it either way).
        "pool_id": pool_id,
        "shell": list(geom["shell"]),
        "flee_x": int(shore_x),
        "flee_z": int(shore_z),
        "world_spawn": list(world_spawn) if world_spawn else None,
        # None = NOT CHECKED (no --stack-on), never "clear" — the same idiom as the two guards below.
        "pool_clearance": pool_clearance,
        # `/spawnpoint` is per PLAYER, not per pool: the last build owns every respawn in the world.
        # A harness that can respawn in this pool must re-assert it (Exp 62's harness does).
        "spawnpoint_set_for": pool_id,
        "exp58_clearance_blocks": exp58_clearance_blocks,
        "spawn_clearance_blocks": spawn_clearance_blocks,
        "shore": list(geom["shore"]),
        "submerged": list(geom["submerged"]),
        "surface_y": geom["shore_y"],
        "depth": geom["depth"],
        "pool": list(geom["pool"]),
        "forceload": list(geom["forceload"]),
        "deaths_objective": "exp60_deaths",
        # Generic situation plan so l11_geometry_probe can take --anchor-file without
        # borrowing Exp 58's anchor/dark names: baseline FIRST (first-touch order),
        # settle on the sensor that DEFINES each situation (not altitude), and the
        # contrast situation names its rescue — the probe budgets each dive visit from
        # the check's stamped `measured.t_damage_onset_min_s` and refuses without it.
        "probe_situations": {"shore": list(geom["shore"]), "submerged": list(geom["submerged"])},
        "probe_settle": {"shore": {"is_in_water": 0, "on_ground": 1}, "submerged": {"is_in_water": 1}},
        "probe_rescue": {"submerged": "shore"},
    }


def _read_json_or_none(path: Path) -> dict | None:
    """ABSENT → None; any other read failure is loud (a permission error or a different
    $HOME must not read as "no Exp 58 record" and vacate the clearance guard)."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SystemExit(f"cannot read anchor record {path}: {exc}")
    except ValueError as exc:
        raise SystemExit(f"corrupt anchor record {path}: {exc}")


def _water_classroom(args: argparse.Namespace) -> int:
    """Build the Exp 60 drowning classroom over RCON (docs/experiments/exp60_drowning_avoidance_prereg.md).

    Geometry from :func:`water_classroom_geometry`; hygiene per the environment lens
    (E4): walled source water, forceload before any teleport, dry rescue platform,
    ``doMobSpawning false``, recorded anchor file. Every fill reply and every
    post-build block check is verified — a silent apparatus failure (head in an air
    gap, drained column, wet shore) must refuse the build, never masquerade as a null.
    Two PLACEMENT guards keep capped distance sensors off their caps (each cap is a
    full-weight constant in both water situations): >= 72 blocks from the Exp 58
    clustermob (``nearest_hostile_dist`` horizon 64) and <= 90 blocks from WORLD spawn
    (``distance_from_spawn`` cap 128; checked here only with ``--spawn-x/y/z``, always
    checked live by ``exp60_water_check`` W1). Idempotent: rebuilds in place at the
    recorded anchor. ``--sweep`` kills zombies (sparing the Exp 58 clustermob) and
    drowned within 64 of the shore (arm boundaries).
    """
    from survival_world.common import bot_pos

    anchor_file = Path(args.anchor_file).expanduser() if args.anchor_file else WATER_ANCHOR_FILE
    if args.backfill:
        # Add the Exp 62 fields to a record written before them, touching nothing else — in
        # particular NOT `measured`, whose loss makes exp60_run and r3_run refuse every row.
        rec = _read_json_or_none(anchor_file)
        if rec is None:
            print(f"no water classroom recorded at {anchor_file} — nothing to backfill")
            return 4
        try:
            shell = record_shell(rec)
        except ValueError as exc:
            print(f"REFUSING: {exc}")
            return 4
        if shell is None:
            print(f"REFUSING: {anchor_file} lacks shore/depth/surface_y — cannot derive the shell; rebuild it instead.")
            return 4
        before = dict(rec)
        rec.setdefault("pool_id", args.pool_id)
        rec["shell"] = list(shell)
        rec.setdefault("flee_x", int(rec["shore"][0]))
        rec.setdefault("flee_z", int(rec["shore"][2]))
        rec.setdefault("world_spawn", None)
        rec.setdefault("pool_clearance", None)
        rec.setdefault("spawnpoint_set_for", rec.get("pool_id", args.pool_id))
        assert rec.get("measured") == before.get("measured")  # the whole point of --backfill
        anchor_file.write_text(json.dumps(rec, indent=2) + "\n")
        added = sorted(set(rec) - set(before))
        print(f"backfilled {anchor_file}: added {added or '(nothing — already current)'}; measured preserved.")
        return 0
    rcon = RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    try:
        recorded = _read_json_or_none(anchor_file)
        if args.sweep:
            if not recorded:
                print(f"no water classroom recorded at {anchor_file} — build first")
                return 4
            sx, sy, sz = recorded["shore"]
            # Spare the Exp 58 clustermob (apparatus, not spillover) — the 72-block
            # clearance leaves it 1 block outside this radius at the margin.
            for sel in ("zombie,tag=!exp58clustermob", "drowned"):
                resp = rcon.command(
                    f"execute positioned {sx} {sy} {sz} run kill @e[type=minecraft:{sel},distance=..64]"
                )
                print(f"rcon> kill {sel} r64 @shore\n      {resp.strip() or '(none)'}")
            return 0
        shore_y = args.shore_y
        if shore_y is None:
            shore_y = int(recorded["surface_y"]) if recorded and "surface_y" in recorded else WATER_SHORE_Y
        elif recorded and "surface_y" in recorded and int(recorded["surface_y"]) != int(shore_y):
            print(
                f"REFUSING: {anchor_file} records this pool at shore y {recorded['surface_y']}, but --shore-y "
                f"{shore_y} was given. A rebuild at another height is a DIFFERENT pool: pass a new --anchor-file "
                f"(and --stack-on this one), or drop --shore-y to rebuild in place."
            )
            return 2
        if args.anchor_x is not None:
            ax, az = args.anchor_x, args.anchor_z
        elif recorded:
            ax, _sy, az = recorded["shore"]
            ax += 1  # shore is anchor-1 (see geometry)
        else:
            ax, _ay, az = bot_pos(rcon, args.username)
        try:
            geom = water_classroom_geometry(int(ax), int(az), depth=args.depth, shore_y=shore_y)
        except ValueError as exc:
            print(f"usage: {exc}")
            return 2
        # Exp 62: a stacked pool must not share a wall with the other one (one fill away from
        # draining its column). The other pool is named EXPLICITLY — the build must not be a
        # function of whatever else happens to sit in the directory.
        pool_clearance: dict | None = None
        if args.stack_on:
            other_path = Path(args.stack_on).expanduser()
            other_rec = _read_json_or_none(other_path)
            if other_rec is None:
                print(
                    f"REFUSING: --stack-on {other_path} does not exist; give the other pool's record or omit the flag."
                )
                return 4
            try:
                ok_gap, why_gap = pools_disjoint(geom, other_rec)
            except ValueError as exc:
                print(f"REFUSING: cannot read the other pool's shell ({exc}) — fix that record or rebuild it.")
                return 4
            if not ok_gap:
                print(
                    f"REFUSING: this build would collide with the pool recorded at {other_path} — {why_gap}.\n"
                    f"Stacked pools need disjoint shells; move --shore-y further from the other pool's."
                )
                return 4
            print(f"pool clearance vs {other_path.name}: {why_gap}")
            pool_clearance = {"other_record": str(other_path), "result": why_gap}
        else:
            print(
                "NOTE: no --stack-on given — pool-vs-pool clearance NOT CHECKED (recorded null). "
                "Exp 62's second pool MUST pass the first pool's record."
            )
        exp58_rec = _read_json_or_none(EXP58_ANCHOR_FILE)
        ok, dmin = exp58_clearance(geom, exp58_rec)
        if not ok:
            print(
                f"REFUSING: pool centre is {dmin:.0f} blocks (horizontal) from the recorded Exp 58 "
                f"classroom ({EXP58_ANCHOR_FILE}); need >= {WATER_MIN_DIST_FROM_EXP58} so its persistent "
                f"clustermob stays beyond the bridge's 64-block hostile horizon (sensor neutral).\n"
                f"Pass --anchor-x/--anchor-z at least {WATER_MIN_DIST_FROM_EXP58 - dmin:.0f} blocks further away."
            )
            return 4
        exp58_blocks = None if exp58_rec is None else round(dmin, 1)
        if exp58_rec is None:
            print(f"NOTE: no Exp 58 record at {EXP58_ANCHOR_FILE} — clustermob clearance NOT checked (recorded null).")
        spawn = None
        if any(v is not None for v in (args.spawn_x, args.spawn_y, args.spawn_z)):
            if any(v is None for v in (args.spawn_x, args.spawn_y, args.spawn_z)):
                print("usage: --spawn-x, --spawn-y and --spawn-z must be given together")
                return 2
            spawn = (args.spawn_x, args.spawn_y, args.spawn_z)
        ok, dspawn = spawn_clearance(geom, spawn)
        if not ok:
            print(
                f"REFUSING: submerged target is {dspawn:.0f} blocks (3D) from world spawn {spawn}; need "
                f"<= {WATER_MAX_DIST_FROM_SPAWN} or `distance_from_spawn` (cap 128) is a full-weight "
                f"CONSTANT in both water situations — replayed cos 0.8525 at the cap = same cluster "
                f"(exp60_spawn_distance_check.py in the experiments data dir). Pass --anchor-x/--anchor-z nearer spawn."
            )
            return 4
        if spawn is None:
            print(
                "NOTE: world spawn not given (--spawn-x/y/z) — distance_from_spawn is gated LIVE by "
                f"exp60_water_check W1 (sensed value must be <= {WATER_MAX_DIST_FROM_SPAWN})."
            )
        for cmd in water_classroom_commands(geom, args.username):
            resp = rcon.command(cmd).strip()
            print(f"rcon> {cmd[:96]}\n      {resp or '(ok)'}")
            low = resp.lower()
            bad = "unknown" in low or "expected" in low or "incorrect" in low
            # "No blocks were filled" CONTAINS "filled" — test the failure reply explicitly.
            if cmd.startswith("fill") and ("filled" not in low or "no blocks were filled" in low):
                bad = True  # nothing placed / unloaded chunk: the geometry is not built
            if bad:
                print("\nWATER CLASSROOM BUILD FAILED on the command above — nothing gated may run.")
                return 4
        # Let fluid ticks run before asserting the sources held (a draining column shows here).
        time.sleep(2.0)
        for cmd, proves in water_classroom_verifications(geom):
            resp = rcon.command(cmd).strip()
            if "passed" not in resp.lower():
                print(f"\nWATER CLASSROOM BUILD FAILED: {proves} — `{cmd}` -> {resp!r}")
                return 4
            print(f"verified: {proves}")
        anchor_file.parent.mkdir(parents=True, exist_ok=True)
        record = water_anchor_record(
            geom,
            pool_id=args.pool_id,
            world_spawn=spawn,
            pool_clearance=pool_clearance,
            exp58_clearance_blocks=exp58_blocks,
            spawn_clearance_blocks=None if dspawn is None else round(dspawn, 1),
        )
        anchor_file.write_text(json.dumps(record, indent=2) + "\n")
        sx, sy, sz = geom["shore"]
        bx, by, bz = geom["submerged"]
        print(
            f"\nwater classroom built: shore {geom['shore']} (dry; spawnpoint; rescue target), "
            f"pool {geom['pool']} ({geom['depth']} deep, source water, walled, open top), "
            f"submerged target {geom['submerged']} (head at y={by + 1} in water, "
            f"{geom['depth'] - 1} blocks below air at y={sy}). "
            f"Exp 58 clearance {exp58_blocks} blocks (null = not checked); world-spawn clearance "
            f"{record['spawn_clearance_blocks']} blocks (null = not given; W1 gates it live). "
            f"doMobSpawning=false is APPARATUS-OWNED (the "
            f"Phase-0 instrument check restores it to true — rebuild/verify before a gated run).\n"
            f"geometry recorded -> {anchor_file}\n"
            f"next (clean tree, PYTHONPATH=$PWD/src): python scripts/survival_world/exp60_water_check.py "
            f"--rcon-password '<pw>' --username {args.username} --write-experiment-results"
            + (
                ""
                if anchor_file == WATER_ANCHOR_FILE
                else (f" \\\n    --anchor-file {anchor_file} --out <this pool's OWN apparatus record>")
            )
            + (
                ""
                if anchor_file == WATER_ANCHOR_FILE
                else "\n  (both flags are REQUIRED for a second pool: without them the check reads pool 1's "
                "record, stamps THIS pool's timings into it, and overwrites Exp 60's committed apparatus. "
                "Run the check without --out and it refuses, naming the path to use.)"
            )
        )
        return 0
    finally:
        rcon.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="1.3 survival world (R2 break 3) setup")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="download Paper + write survival configs (needs --accept-eula)")
    s.add_argument("--dir", default="~/maxim-mc-survival-1.20")
    s.add_argument("--accept-eula", action="store_true")
    s.add_argument(
        "--force-world",
        action="store_true",
        help="proceed even if the dir holds a world from another MC version (in-place upgrade)",
    )
    s.add_argument("--rcon-port", type=int, default=25575)
    s.add_argument("--rcon-password", required=True)
    s.set_defaults(fn=_setup)

    c = sub.add_parser("classroom", help="build the Exp 58 dark-fear classroom (or --sweep zombies)")
    c.add_argument("--rcon-host", default="127.0.0.1")
    c.add_argument("--rcon-port", type=int, default=25575)
    c.add_argument("--rcon-password", required=True)
    c.add_argument("--username", default="maxim")
    c.add_argument("--anchor-x", type=float, default=None)
    c.add_argument("--anchor-y", type=float, default=None)
    c.add_argument("--anchor-z", type=float, default=None)
    c.add_argument("--sweep", action="store_true", help="kill classroom-range zombies (arm boundary)")
    c.set_defaults(fn=_classroom)

    w = sub.add_parser("water_classroom", help="build the Exp 60 drowning classroom (or --sweep mobs)")
    w.add_argument("--rcon-host", default="127.0.0.1")
    w.add_argument("--rcon-port", type=int, default=25575)
    w.add_argument("--rcon-password", required=True)
    w.add_argument("--username", default="maxim")
    w.add_argument(
        "--anchor-x",
        type=float,
        default=None,
        help="classroom anchor x (with --anchor-z); default: the recorded exp60 anchor, else the JOINED bot's position",
    )
    w.add_argument("--anchor-z", type=float, default=None)
    w.add_argument(
        "--spawn-x", type=float, default=None, help="WORLD spawn x (with -y/-z): enables the <=90-block spawn guard"
    )
    w.add_argument("--spawn-y", type=float, default=None)
    w.add_argument("--spawn-z", type=float, default=None)
    w.add_argument("--depth", type=int, default=WATER_DEPTH_DEFAULT, help="water column depth in blocks (3..12)")
    w.add_argument(
        "--shore-y",
        type=int,
        default=None,
        help="shore floor height (Exp 62 stacks a second pool at a different --shore-y; default is pool 1's)",
    )
    w.add_argument(
        "--anchor-file",
        default=None,
        help=(
            "where to record this pool's geometry (default: the Exp 60 record). A SECOND pool must pass its own "
            "path — one fixed path means a rebuild overwrites the first pool's `measured` block, which the "
            "harnesses refuse to run without."
        ),
    )
    w.add_argument(
        "--stack-on",
        default=None,
        help=(
            "the OTHER pool's record; its shell must not touch this build's (Exp 62 stacks pool 2 over pool 1). "
            "Omitted = NOT CHECKED, recorded null."
        ),
    )
    w.add_argument(
        "--backfill",
        action="store_true",
        help=(
            "add the Exp 62 fields (shell, flee anchor, pool id) to an EXISTING record without building or "
            "touching `measured` — how pool 1's pre-Exp-62 record is armed for the clearance guard"
        ),
    )
    w.add_argument(
        "--pool-id",
        default="pool1",
        help="name stamped into the record so a row can say which pool it ran in (Exp 62 uses pool1/pool2)",
    )
    w.add_argument(
        "--sweep", action="store_true", help="kill zombies (not the exp58 clustermob) + drowned within 64 of the shore"
    )
    w.set_defaults(fn=_water_classroom)

    for name, fn, helptext in (
        ("prepare", _prepare, "seed food + set conditions over RCON (bot must be joined)"),
        ("verify", _verify, "re-check conditions + food (exit 4 on mismatch)"),
    ):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("--rcon-host", default="127.0.0.1")
        q.add_argument("--rcon-port", type=int, default=25575)
        q.add_argument("--rcon-password", required=True)
        q.add_argument("--username", default="maxim", help="the bridge bot's username")
        if name == "prepare":
            q.add_argument(
                "--induce-hunger",
                action="store_true",
                help="apply a short game-native hunger effect for wiring tests (not the real run)",
            )
        q.set_defaults(fn=fn)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
