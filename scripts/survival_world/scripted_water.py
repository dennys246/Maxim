"""A DETERMINISTIC scripted water classroom speaking the frozen NDJSON bridge protocol — the
offline instrument that proves the shared ``WaterTrial`` seed context can tick and act
(``docs/wiring/harness-loop-must-be-proven-live.md``: a harness loop must be PROVEN able to tick
and act; Exp 61 architecture lens S10 — the 700 lifted lines were covered by nothing, and the
hub-session trap the review found would have shown here).

Dev/smoke/guard-test instrument ONLY, never a confirmatory record (the campaigns run the live
bridge). Modelled on ``exp56.common.ScriptedBridgeServer`` (anchor-following snapshots) with the
PLAYER roster: the paired :class:`ScriptedWaterControl`'s ``teleport`` sets the anchor; the served
world sensors derive from it — at the ``submerged`` anchor ``is_in_water`` is 1, ``on_ground`` 0,
``oxygen`` drains at a scripted rate; at the ``shore`` anchor the bot is dry, grounded and refilled.
``escape_water`` moves the anchor to the shore after a scripted delay (the head clears the water);
``flee`` fails fast when submerged (the live bridge's contract); every action is confirmed with a
post-action snapshot. RCON: ``teleport`` moves the anchor; frozen gamerules echo their wanted
value; the deaths scoreboard reads 0; effects are no-ops (the shore refills by construction).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

PLAYER_ROSTER = (
    "health",
    "food",
    "light_level",
    "y_altitude",
    "nearest_hostile_dist",
    "time_of_day",
    "saturation",
    "oxygen",
    "hostile_count",
    "distance_from_spawn",
    "speed",
    "on_ground",
    "is_raining",
    "is_in_water",
    "xp_level",
    "nearest_player_dist",
    "look_pitch",
)


class ScriptedWaterBridge:
    def __init__(
        self,
        *,
        shore: dict[str, float],
        submerged: dict[str, float],
        state_interval_s: float = 0.05,
        oxygen_drain_per_s: float = 1.0,
        escape_delay_s: float = 0.3,
        spawn_distance: float = 30.0,
        damage_onset_s: float | None = None,
        damage_per_s: float = 2.0,
        respawn_saturation: float = 5.0,
        respawn_lag_s: float | None = None,
    ) -> None:
        """``damage_onset_s`` (default None = no damage, the Exp 60/61 smoke) turns on a SCRIPTED
        DROWNING: submerged past that many seconds the bot loses ``damage_per_s`` health per second
        (the game's 2 hp/s), and at 0 it DIES — ``deaths`` +1, respawn at the shore with health 20,
        oxygen 20 and the game-native respawn saturation (5) — so a harness's death branch and its
        detector are red-gated OFFLINE (R3 wiring lens SF-C)."""
        self.shore = dict(shore)
        self.submerged = dict(submerged)
        self.anchor: dict[str, float] = dict(shore)
        self._interval = state_interval_s
        self._drain = oxygen_drain_per_s
        self._escape_delay = escape_delay_s
        self._spawn_distance = spawn_distance
        self._damage_onset = damage_onset_s
        self._damage_per_s = damage_per_s
        self._respawn_saturation = respawn_saturation
        # how long the `deaths` objective LEADS the respawn snapshot (the live race); default two state intervals
        self._respawn_lag = respawn_lag_s if respawn_lag_s is not None else 2.0 * state_interval_s
        self._health = 20.0
        self._saturation = 10.0  # the SENSED value (the bridge clamps at 10)
        self._sat_true = 20.0  # the game's true reservoir: 20 after the apparatus heal, 5 after a respawn
        self.deaths = 0
        self.died_at: list[float] = []
        self._respawn_at: float | None = None  # the scoreboard leads the respawn snapshot (the live race)
        self.surface_air = True  # what `execute if block <surface> minecraft:air` answers
        self._submerged_since: float | None = None
        self.actions: list[str] = []  # every action name the bridge received, in order
        self._lock = threading.Lock()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(4)
        self.port: int = self._server.getsockname()[1]
        self._stop = threading.Event()
        self._accept = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass

    # ── the scripted world ──

    def set_anchor(self, pos: dict[str, float]) -> None:
        with self._lock:
            self.anchor = dict(pos)
            if self._in_water_locked():
                self._submerged_since = self._submerged_since or time.monotonic()
            else:
                self._submerged_since = None

    def _in_water_locked(self) -> bool:
        a, s = self.anchor, self.submerged
        return abs(a["x"] - s["x"]) < 0.5 and abs(a["y"] - s["y"]) < 0.5 and abs(a["z"] - s["z"]) < 0.5

    def _snapshot(self) -> dict[str, float]:
        with self._lock:
            in_water = self._in_water_locked()
            if in_water:
                since = self._submerged_since or time.monotonic()
                elapsed = time.monotonic() - since
                oxygen = max(0.0, 20.0 - self._drain * elapsed)
                if self._damage_onset is not None and elapsed > self._damage_onset and self._respawn_at is None:
                    self._health = max(0.0, 20.0 - self._damage_per_s * (elapsed - self._damage_onset))
                    if self._health <= 0.0:
                        # DEATH: the `deaths` objective rises NOW; the respawn snapshot follows two state
                        # intervals later — the live race a harness's corroboration must survive
                        self.deaths += 1
                        self.died_at.append(time.monotonic())
                        self._respawn_at = time.monotonic() + self._respawn_lag
                if self._respawn_at is not None and time.monotonic() >= self._respawn_at:
                    # RESPAWN at the shore (doImmediateRespawn), sensors reset
                    self._respawn_at = None
                    self.anchor = dict(self.shore)
                    self._submerged_since = None
                    self._health = 20.0
                    self._saturation = self._respawn_saturation
                    self._sat_true = self._respawn_saturation
                    in_water = False
                    oxygen = 20.0
            else:
                oxygen = 20.0
            y = float(self.anchor["y"])
            health = self._health
            saturation = self._saturation
        return {
            "health": health,
            "food": 20.0,
            "light_level": 9.0 if in_water else 14.0,
            "y_altitude": y,
            "nearest_hostile_dist": 64.0,
            "time_of_day": 0.25,
            "saturation": saturation,
            "oxygen": oxygen,
            "hostile_count": 0.0,
            "distance_from_spawn": self._spawn_distance,
            "speed": 0.0,
            "on_ground": 0.0 if in_water else 1.0,
            "is_raining": 0.0,
            "is_in_water": 1.0 if in_water else 0.0,
            "xp_level": 0.0,
            "nearest_player_dist": 64.0,
            "look_pitch": 0.0,
        }

    def _do_action(self, name: str) -> tuple[bool, str]:
        self.actions.append(name)
        with self._lock:
            in_water = self._in_water_locked()
        if name == "escape_water":
            if not in_water:
                return True, "already in air"

            def _surface() -> None:
                time.sleep(self._escape_delay)
                self.set_anchor(self.shore)

            threading.Thread(target=_surface, daemon=True).start()
            return True, "surfaced"
        if name == "flee":
            if in_water:
                return False, "flee: submerged — the pathfinder is dead in water; escape_water is the water actuator"
            return True, "fled"
        return True, f"did {name}"

    # ── the wire ──

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sock, _addr = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(sock,), daemon=True).start()

    def _serve(self, sock: socket.socket) -> None:
        def send(obj: dict[str, Any]) -> None:
            try:
                sock.sendall(json.dumps(obj).encode() + b"\n")
            except OSError:
                pass

        sock.settimeout(self._interval)
        buffer = b""
        while not self._stop.is_set():
            send({"type": "state", "data": self._snapshot()})
            try:
                chunk = sock.recv(65536)
                if not chunk:
                    return
                buffer += chunk
            except TimeoutError:
                continue
            except OSError:
                return
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                if not raw.strip():
                    continue
                try:
                    msg = json.loads(raw.decode())
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if msg.get("type") == "action":
                    ok, detail = self._do_action(str(msg.get("name")))
                    send(
                        {
                            "type": "action_result",
                            "id": msg.get("id"),
                            "ok": ok,
                            "detail": detail,
                            "state": self._snapshot(),
                        }
                    )


_KNOWN_GAMERULES = {"naturalRegeneration", "drowningDamage", "doInsomnia"}  # 1.20.4 names (camelCase era)


class ScriptedWaterControl:
    """The RCON half, paired with :class:`ScriptedWaterBridge`."""

    def __init__(self, server: ScriptedWaterBridge, *, gamerules: dict[str, str] | None = None) -> None:
        self._server = server
        self._gamerules = dict(
            gamerules
            or {
                "doMobSpawning": "false",
                "doDaylightCycle": "false",
                "doWeatherCycle": "false",
                "doImmediateRespawn": "true",
                "keepInventory": "true",
            }
        )
        self.commands: list[str] = []

    def teleport(self, bot_name: str, pos: dict[str, float]) -> None:
        # logged like the real client's `tp` (which goes through `command`), so a test can assert ORDER
        self.commands.append(f"tp {bot_name} {pos['x']} {pos['y']} {pos['z']}")
        self._server.set_anchor(pos)

    def command(self, cmd: str) -> str:
        """The RCON verbs the water harnesses issue, answered in the server's own reply shapes."""
        self.commands.append(cmd)
        parts = cmd.split()
        srv = self._server
        if parts and parts[0] == "gamerule" and len(parts) >= 2:
            if len(parts) >= 3:  # a SET: update and echo the server's set reply
                if parts[1] not in self._gamerules and parts[1] not in _KNOWN_GAMERULES:
                    return f"Incorrect argument for command\ngamerule {parts[1]}<--[HERE]"
                self._gamerules[parts[1]] = parts[2]
                return f"Gamerule {parts[1]} is now set to: {parts[2]}"
            if parts[1] not in self._gamerules and parts[1] not in _KNOWN_GAMERULES:
                return f"Incorrect argument for command\ngamerule {parts[1]}<--[HERE]"
            return f"Gamerule {parts[1]} is currently set to: {self._gamerules.get(parts[1], 'true')}"
        if parts and parts[0] == "scoreboard":
            if parts[1:3] == ["objectives", "list"]:
                return "There are 1 objective(s): [exp60_deaths]"
            if parts[1:3] == ["players", "set"] and len(parts) >= 6:
                srv.deaths = int(parts[5])
                return f"Set [{parts[4]}] for {parts[3]} to {parts[5]}"
            return f"{parts[3] if len(parts) > 3 else 'maxim'} has {srv.deaths} [{parts[4] if len(parts) > 4 else 'exp60_deaths'}]"
        if parts and parts[0] == "tp" and len(parts) >= 5:
            srv.set_anchor({"x": float(parts[2]), "y": float(parts[3]), "z": float(parts[4])})
            return f"Teleported {parts[1]}"
        if parts and parts[0] == "execute" and "minecraft:air" in cmd:
            return "Test passed" if srv.surface_air else "Test failed"
        if parts and parts[:3] == ["data", "get", "entity"] and len(parts) >= 5:
            field = parts[4]
            vals = {"foodLevel": "20", "foodSaturationLevel": f"{srv._sat_true:.1f}f", "foodExhaustionLevel": "0.0f"}
            if field in vals:
                return f"{parts[3]} has the following entity data: {vals[field]}"
            return f"Found no elements matching {field}"
        if parts and parts[0] == "fill":
            return "Successfully filled 9 block(s)"
        if parts and parts[:2] == ["effect", "give"] and len(parts) >= 4:
            # the apparatus heal: instant health restores the scripted damage; saturation fills the TRUE
            # reservoir to 20 while the sensed value sits at the clamp (the R3 reservoir finding)
            with srv._lock:
                if "instant_health" in parts[3]:
                    srv._health = 20.0
                elif "saturation" in parts[3]:
                    srv._sat_true = 20.0
                    srv._saturation = 10.0
            return f"Applied effect {parts[3]} to {parts[2]}"
        return ""

    def close(self) -> None:
        return
