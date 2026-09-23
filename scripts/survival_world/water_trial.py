"""The water-classroom seed context shared by the Exp 60 and Exp 61 harnesses.

Lifted VERBATIM (2026-09-16, Exp 61 wiring lens SF-7) from ``exp60_run._run``'s closures — the
rescue/submerge primitives, every preflight, the loop window, the US-free probe, propose-only
training and the live G2 read — into one class over the state those closures shared (the AUT,
RCON, the anchor geometry, the frozen constants, the per-seed pain-signal and executor-call
instruments). ``exp60_run`` calls the same methods in the same order it always did (its EARNED
verdict is proven unchanged by diffing ``exp60_run.py verdict`` before and after the lift);
``exp61_run`` composes donor and receiver flows from the same methods. A second 700-line ``_run``
would be the "hand-composed second builder" shape ``build_minecraft_aut``'s docstring warns
against — this class is the one implementation.

Nothing here is a harness entry point: no provenance, no evidence path, no argument parsing.
Those stay in the two runners (``in_process_code_provenance`` + ``evidence_out_paths_or_exit``).
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world.common import InstrumentError, settle_until, sync_snapshot  # noqa: E402
from survival_world.exp60_water_check import (  # noqa: E402
    IN_WATER_WITHIN_S,
    RECOVER_OXYGEN_MIN,
    REQUIRED_BRIDGE_SENSORS,
    STALE_MAX_CONSECUTIVE,
    STALE_STATE_S,
    SURFACE_WITHIN_S,
    missing_bridge_sensors,
)
from survival_world.l11_geometry_probe import SATURATION_REST  # noqa: E402

GAMERULES: tuple[tuple[str, str], ...] = (
    ("doMobSpawning", "false"),
    ("doDaylightCycle", "false"),
    ("doWeatherCycle", "false"),
    ("doImmediateRespawn", "true"),
    ("keepInventory", "true"),
)
# R3 (the lethal window) verifies three more — the Exp 60/61 roster above stays as frozen:
# regeneration moves the death edge by 7.7 s (pilot), drowning damage IS the hazard (the pilot
# misspelled it `doDrowningDamage`; the 1.20.4 name is `drowningDamage`), insomnia is a belt for a
# multi-hour campaign. An UNKNOWN-name reply is an instrument error, never a recorded absence.
R3_GAMERULES: tuple[tuple[str, str], ...] = GAMERULES + (
    ("naturalRegeneration", "true"),
    ("drowningDamage", "true"),
    ("doInsomnia", "false"),
)
_UNKNOWN_GAMERULE = ("incorrect argument", "unknown", "<--[here]")
_NUM_TOKEN = re.compile(r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)[fdbsL]?\b")


class Refusal(RuntimeError):
    """A prereg stop rule fired — the seed must not produce a verdict row.

    ``partial`` carries the record fields the failing step had already measured (the fingerprint
    that drifted, the actuation outcome that failed, the training that fell short, …) so a refused
    row keeps its diagnosis — Exp 60's committed refused row carries `actuation_preflight` beside
    its refusal, and that field is how it was diagnosed (executor-lens fold, Exp 61 harness)."""

    def __init__(self, message: str, *, partial: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.partial: dict[str, Any] = dict(partial or {})


class _NullCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def _f(vm: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(vm.get(key, default))
    except (TypeError, ValueError):
        return default


def _detach_fear_subscriber(aut: Any) -> int:
    """ABLATED arm: detach the Wire-4 fear subscriber (the pain still publishes)."""
    bus = aut.bio.pain_bus
    targets = [cb for cb in list(bus._pain_signal_subs) if "cluster_fear" in getattr(cb, "__qualname__", "")]
    for cb in targets:
        bus.unsubscribe(cb)
    return len(targets)


def _telemetry_ticks(path: Path, t0_monotonic: float, *, t0_wall: float | None = None) -> list[dict[str, Any]]:
    """Compress a window's SubstrateTelemetry JSONL into per-tick rows (pure over the file).

    Rows carry wall-clock ``ts``. Without ``t0_wall`` (Exp 60/61's windows) ticks are reported
    relative to the FIRST row (the loop's first tick — ≈ ``loop_warm_s`` BEFORE the teleport, a
    clock the window's monotonic ``t`` does not share; ``t0_monotonic`` is kept for those callers
    and unused). With ``t0_wall`` (R3's lethal event: ``time.time()`` stamped AT the teleport) every
    tick also carries ``t`` on the window's clock, so proposals and executor calls are comparable.
    The drive snapshot is nested (``drives.drives`` keyed by body sensor name); the earlier reader
    looked one level up and recorded nothing.
    """
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    except OSError:
        return []
    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except ValueError:
            continue
    if not rows:
        return []
    first_ts = rows[0].get("ts", 0.0)
    out = []
    for r in rows:
        prop = r.get("proposal") or {}
        nac = r.get("nac") or {}
        dr = r.get("drives")
        # live rows nest the values (`drives.drives`, keyed by body sensor); a flat dict is accepted too
        inner = (
            dr.get("drives")
            if isinstance(dr, dict) and isinstance(dr.get("drives"), dict)
            else (dr if isinstance(dr, dict) else {})
        )
        tick = {
            "t_from_first_tick": round(float(r.get("ts", 0.0)) - float(first_ts), 3),
            "step": r.get("step"),
            "proposal": prop.get("tool_name") or prop.get("tool") if isinstance(prop, dict) else prop,
            "gated": r.get("gated"),
            "active_clusters": nac.get("active_clusters") if isinstance(nac, dict) else None,
            "drives": {k: v for k, v in inner.items() if k in ("threat", "oxygen", "health", "food")} or None,
        }
        if t0_wall is not None:
            tick["t"] = round(float(r.get("ts", 0.0)) - float(t0_wall), 3)
        out.append(tick)
    return out


# ── pure helpers (moved here from exp60_run so the class needs no import from its caller) ──


def classify_placement(samples: list[dict[str, Any]], *, cap_s: float) -> dict[str, Any]:
    """One probe placement from ``(t, in_water, health)`` samples, t from teleport (pure).

    ``surfaced`` = the first head-in-air read AFTER the head was seen in water and BEFORE
    any health drop or the cap. A health drop inside the window is a DIRTY placement
    (damage occurred: the window was not damage-free) and censors it.
    """
    seen_water = False
    health0 = None
    for s in samples:
        if s["t"] > cap_s:
            break
        if health0 is None and s.get("health") is not None:
            health0 = s["health"]
        if s.get("health") is not None and health0 is not None and s["health"] < health0:
            return {"surfaced": False, "latency_s": None, "censored": True, "dirty": True, "t_end": s["t"]}
        if s["in_water"]:
            seen_water = True
        elif seen_water:
            return {"surfaced": True, "latency_s": round(s["t"], 3), "censored": False, "dirty": False, "t_end": s["t"]}
    return {
        "surfaced": False,
        "latency_s": None,
        "censored": True,
        "dirty": False,
        "t_end": samples[-1]["t"] if samples else 0.0,
        "never_submerged": not seen_water,
    }


def p_surface(probe: dict[str, Any]) -> float | None:
    """Per-seed P(surface before first damage tick) over the probe's CLEAN placements."""
    ps = [p for p in probe.get("placements", []) if not p.get("dirty") and not p.get("never_submerged")]
    if not ps:
        return None
    return sum(1 for p in ps if p["surfaced"]) / len(ps)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def fingerprint_drift(live: dict[str, Any], frozen: dict[str, Any]) -> list[str]:
    """Keys whose live value differs from the frozen apparatus (pure; lists compared sorted)."""

    def norm(v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return (
                sorted(norm(x) for x in v) if all(not isinstance(x, (list, dict)) for x in v) else [norm(x) for x in v]
            )
        if isinstance(v, dict):
            return {k: norm(x) for k, x in sorted(v.items())}
        if isinstance(v, float):
            return round(v, 6)
        return v

    return sorted(k for k in set(live) | set(frozen) if norm(live.get(k)) != norm(frozen.get(k)))


# The world channel's ENCODING IDENTITY on the shipped `bodies/minecraft_player` body (issue #783).
# `w = (2|v−0.5|)^gain_exponent` with `v = (c−lo)/(hi−lo)`: the encoder config IS the equation and a
# declared range IS the weight of a constant sensor, so either moving re-keys every `world` node.
# Transcribed 2026-09-20 from an offline read (build_minecraft_aut + `_read_world_ranges`) of the
# shipped body; `test_water_trial_encoding.py` binds it to the committed Exp 60 / Exp 62 geometry
# records' provenance and to Exp 60's frozen ranges, so the constant cannot drift from what those
# campaigns ran under without a red test.
# Kept OUT of the experiments' FROZEN blocks on purpose: `exp60_run.FROZEN` is the literal apparatus
# Exp 60 froze, and Exp 61/62 carry it as literal copies checked by `frozen_matches` — widening it
# would edit what three closed experiments DECLARED and make them claim guarding they never ran
# under. (R3's gauntlet also pins `sha256(FROZEN60)`.)
# RE-RUN TRIGGER: an edit to this constant is a change to the world-channel geometry — it fires the
# Exp 60/61/62 "Re-run on" triggers exactly as the encoder/range change it mirrors would. A VARIANT
# body's identity belongs in its own experiment's FROZEN block and prereg, passed as
# `WaterTrial(encoding=...)` — never a second module constant here; omitting it refuses.
APPARATUS_ENCODING: dict[str, Any] = {
    "encoder_config": {
        "embedding_dim": 384,
        "min_delta": 0.05,
        "pattern_threshold": 0.85,
        "gain_exponent": 3.0,
        "gain_modalities": ["world"],
    },
    "world_ranges": {
        "distance_from_spawn": {"lo": -128.0, "hi": 128.0},
        "food": {"lo": 0.0, "hi": 40.0},
        "health": {"lo": 0.0, "hi": 40.0},
        "hostile_count": {"lo": -32.0, "hi": 32.0},
        "is_in_water": {"lo": -1.0, "hi": 1.0},
        "is_raining": {"lo": -1.0, "hi": 1.0},
        "light_level": {"lo": 0.0, "hi": 15.0},
        "look_pitch": {"lo": -1.5708, "hi": 1.5708},
        "nearest_hostile_dist": {"lo": 0.0, "hi": 128.0},
        "nearest_player_dist": {"lo": 0.0, "hi": 128.0},
        "on_ground": {"lo": -1.0, "hi": 3.0},
        "oxygen": {"lo": 0.0, "hi": 40.0},
        "saturation": {"lo": 0.0, "hi": 20.0},
        "speed": {"lo": -1.0, "hi": 1.0},
        "time_of_day": {"lo": 0.0, "hi": 1.0},
        "xp_level": {"lo": -50.0, "hi": 50.0},
        "y_altitude": {"lo": 0.0, "hi": 128.0},
    },
}


def encoder_config_identity(config: Any) -> dict[str, Any]:
    """EVERY field of a `SensorEncoderConfig`, by `dataclasses.fields` (pure).

    Enumerated, not listed: a field added to the config later appears here and drifts against a
    frozen identity that lacks it — it cannot join the equation unguarded.
    """
    import dataclasses

    out: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        v = getattr(config, f.name)
        out[f.name] = sorted(v) if isinstance(v, (set, frozenset)) else v
    return out


UNRANGED = "unranged"  # a declared world sensor with no usable range (encoded via the legacy map)


def declared_world_roster(executor: Any) -> list[str]:
    """Every `modality: world` sensor the body DECLARES, range or not (sorted).

    `_read_world_ranges` skips a world sensor with a missing/malformed range (it re-folds through
    the legacy map) while `_read_world_states` still encodes it — so the ranges alone cannot see a
    range-less sensor joining the vector. Same walk as `agent_loop._read_declared_modality_ranges`.
    """
    from maxim.embodiment.sensory_streams import WORLD_TAG

    root = getattr(getattr(executor, "embodiment", None), "root", None)
    if root is None:
        return []
    walk = getattr(root, "walk", None)
    names: set[str] = set()
    for ent in walk() if callable(walk) else (root,):
        for name, sensor in (getattr(ent, "sensors", {}) or {}).items():
            if (getattr(sensor, "reading_schema", {}) or {}).get("modality") == WORLD_TAG:
                names.add(name)
    return sorted(names)


def encoding_identity(config: Any, world_ranges: dict[str, Any], roster: "list[str] | None" = None) -> dict[str, Any]:
    """The encoding equation + the full declared world roster (pure). Ranges are `{lo, hi}`, not
    `[lo, hi]`: `fingerprint_drift` sorts scalar lists, which would hide a reversed range. A roster
    sensor with no usable range enters as `UNRANGED` — present in the vector, absent from the ranges.
    Never `None`: `fingerprint_drift` compares `.get(k)`, so a None value equals an absent key."""
    names = sorted(set(world_ranges) | set(roster or ()))
    return {
        "encoder_config": encoder_config_identity(config),
        "world_ranges": {
            k: ({"lo": float(world_ranges[k][0]), "hi": float(world_ranges[k][1])} if k in world_ranges else UNRANGED)
            for k in names
        },
    }


def encoding_drift(live: dict[str, Any], frozen: dict[str, Any]) -> list[str]:
    """Dotted paths where a live encoding identity differs from the frozen one (pure; empty = same)."""
    drift = [
        f"{sec}.{k}"
        for sec in ("encoder_config", "world_ranges")
        for k in fingerprint_drift(live.get(sec) or {}, frozen.get(sec) or {})
    ]
    drift += [k for k in fingerprint_drift(live, frozen) if k not in ("encoder_config", "world_ranges")]
    return drift


def median_interval_s(reset_times: list[float]) -> float | None:
    """Median gap between consecutive snapshot arrivals (pure). None below two arrivals."""
    if len(reset_times) < 2:
        return None
    gaps = sorted(b - a for a, b in zip(reset_times, reset_times[1:]))
    n = len(gaps)
    return gaps[n // 2] if n % 2 else (gaps[n // 2 - 1] + gaps[n // 2]) / 2.0


def min_pain_edge_s(apparatus: dict[str, Any]) -> float | None:
    """The earliest measured air-hunger pain edge across the apparatus check's cycles (pure)."""
    edges = [c.get("w2_dive", {}).get("t_pain_edge") for c in apparatus.get("cycles", [])]
    edges = [float(e) for e in edges if e is not None]
    return min(edges) if edges else None


class WaterTrial:
    """One seed's live water-classroom apparatus (see the module docstring)."""

    def __init__(
        self,
        *,
        aut: Any,
        rcon: Any,
        username: str,
        geom: dict[str, Any],
        frozen: dict[str, Any],
        probe_cap_s: float,
        train_cap_s: float,
        persistence_dir: str | Path,
        agent_id: str,
        encoder: Any,
        settle_guard: dict[str, float] | None = None,
        encoding: dict[str, Any] | None = None,
    ) -> None:
        # The frozen encoding identity `check_fingerprint` holds the live one to (issue #783).
        self.encoding: dict[str, Any] = APPARATUS_ENCODING if encoding is None else encoding
        # Sensors that must read an exact value on EVERY rescue settle (Exp 61: `is_raining` 0,
        # `nearest_player_dist` 64 — rain breaks cluster completion, a spectator costs the margin);
        # an ABSENT key refuses, never defaults to the passing value (environment lens S4).
        self.settle_guard: dict[str, float] = dict(settle_guard or {})
        self.aut = aut
        self.rcon = rcon
        self.username = username
        self.geom = geom
        self.frozen = frozen
        self.probe_cap_s = probe_cap_s
        self.train_cap_s = train_cap_s
        self.persistence_dir = Path(persistence_dir)
        self.agent_id = agent_id
        self.encoder = encoder
        self.shore = {"x": float(geom["shore"][0]), "y": float(geom["shore"][1]), "z": float(geom["shore"][2])}
        self.sub = {
            "x": float(geom["submerged"][0]),
            "y": float(geom["submerged"][1]),
            "z": float(geom["submerged"][2]),
        }
        # The bus subscriber runs for the WHOLE seed: every pain publish is timestamped so a
        # probe window can prove it was US-free (or be marked DIRTY + unyoked exposure).
        self.signals: list[dict[str, Any]] = []
        # Every executor call this seed makes — tool, outcome, error, time — so a window can
        # show what the loop DID, not only what succeeded.
        self.calls: list[dict[str, Any]] = []
        self._orig_execute: Any = None
        self.escape_tool: str | None = None
        self.flee_tool: str | None = None
        self.deaths0 = 0

    # ── instruments ──────────────────────────────────────────────────────────────────

    def _record_pain(self, sig: Any) -> None:
        ctx = getattr(sig, "context", None) or {}
        self.signals.append(
            {
                "t": time.monotonic(),
                "failure_mode": ctx.get("failure_mode"),
                "intensity": float(getattr(sig, "intensity", 0.0)),
            }
        )

    def use_geometry(self, geom: dict[str, Any]) -> dict[str, Any]:
        """Point this trial at ANOTHER pool, keeping the agent and its single instrument attach.

        Exp 62 reads one agent in two pools. Building a second WaterTrial would double-wrap the
        executor spy (attach_instruments refuses it), so the geometry is what moves. Returns the
        geometry replaced, so a caller can put it back.
        """
        previous = self.geom
        self.geom = geom
        self.shore = {"x": float(geom["shore"][0]), "y": float(geom["shore"][1]), "z": float(geom["shore"][2])}
        self.sub = {
            "x": float(geom["submerged"][0]),
            "y": float(geom["submerged"][1]),
            "z": float(geom["submerged"][2]),
        }
        return previous

    def attach_instruments(self) -> None:
        """Idempotent by refusal: a second attach would wrap the FIRST trial's spy, so every call
        it records would be counted twice and the pain subscriber would fire twice per publish.
        Exp 62 runs two pools per agent, which is exactly when this happens (prereg §Apparatus:
        "ONE instrument attach — the executor spy must not double-wrap")."""
        # Per-TRIAL: re-attaching the same trial. Per-AGENT: a SECOND trial (Exp 62 runs one per
        # pool) wrapping the same executor — which the per-trial check alone cannot see, because the
        # new trial has its own `self`. The sentinel rides on the installed wrapper, so it catches
        # both. (Prereg §Apparatus: "ONE instrument attach — the executor spy must not double-wrap".)
        if getattr(self, "_orig_execute", None) is not None or getattr(
            self.aut.executor.execute, "_water_trial_spy", False
        ):
            raise InstrumentError(
                "instruments are already attached to this agent — a second attach double-wraps the "
                "executor spy (every call counted twice, the pain subscriber fired twice per publish); "
                "attach ONCE per agent and pass each pool's geometry to the call that needs it"
            )
        self.aut.bio.pain_bus.subscribe(self._record_pain)
        self._orig_execute = self.aut.executor.execute
        orig = self._orig_execute
        calls = self.calls

        def _spy_execute(action: dict[str, Any]) -> Any:
            t = time.monotonic()
            try:
                out = orig(action)
            except Exception as exc:
                calls.append(
                    {
                        "t": t,
                        "t_return": time.monotonic(),
                        "tool": (action or {}).get("tool_name"),
                        "success": False,
                        "error": repr(exc),
                        "detail": None,
                    }
                )
                raise
            output = str(getattr(out, "output", "") or "")
            m = re.search(r"'detail': '([^']*)'", output)
            calls.append(
                {
                    "t": t,
                    "t_return": time.monotonic(),  # a call in flight at death RETURNS after respawn (pilot)
                    "tool": (action or {}).get("tool_name"),
                    "success": getattr(out, "success", None),
                    "error": getattr(out, "error", None),
                    "detail": m.group(1) if m else None,
                }
            )
            return out

        _spy_execute._water_trial_spy = True  # the per-AGENT sentinel the guard above reads
        self.aut.executor.execute = _spy_execute  # instance attribute; the loop calls executor.execute(action)

    def detach_instruments(self) -> None:
        try:
            self.aut.bio.pain_bus.unsubscribe(self._record_pain)
        except Exception as exc:
            print(f"WARNING: pain unsubscribe raised: {exc!r}")
        if self._orig_execute is not None:
            self.aut.executor.execute = self._orig_execute

    def reopen_hub_session(self) -> None:
        """The loop's own start/end bio-session pair CLOSES the hub session the harness opened
        (`bio_integration.end_bio_session` → `MemoryHub.on_session_end`, atomic test-and-clear),
        and a hub whose session is not active persists NOTHING at the harness's own close — the
        environment lens's S1 trap one layer up (fear booked after a liveness loop would never
        reach the staged nac). Re-open after every loop run; `on_session_start` is idempotent and
        does not reload state from disk."""
        hub = getattr(self.aut.bio, "memory_hub", None)
        if hub is not None:
            hub.on_session_start()

    def resolve_tools(self) -> None:
        names = self.aut.executor.registry.list()
        self.escape_tool = next((t for t in names if t.endswith("_escape_water")), None)
        self.flee_tool = next((t for t in names if t.endswith("_flee")), None)
        if self.escape_tool is None:
            raise Refusal("no *_escape_water tool registered")

    # ── primitives ───────────────────────────────────────────────────────────────────

    def heal(self) -> None:
        self.rcon.command(f"effect give {self.username} minecraft:instant_health 1 10 true")
        self.rcon.command(f"effect give {self.username} minecraft:saturation 1 10 true")

    @property
    def deaths_objective(self) -> str:
        return str(self.geom.get("deaths_objective", "exp60_deaths"))

    def deaths(self) -> int:
        """The `deaths` scoreboard read. A parse failure is an InstrumentError — NEVER 0: in a lethal
        window death is the DV, and a silent zero reads a death as a surface (R3 wiring lens SF-4/SF-C)."""
        resp = self.rcon.command(f"scoreboard players get {self.username} {self.deaths_objective}")
        try:
            return int(resp.split(" has ")[1].split()[0])
        except (IndexError, ValueError) as exc:
            raise InstrumentError(f"deaths objective unreadable: {resp!r}") from exc

    def preflight_deaths_objective(self) -> None:
        """The objective EXISTS, is set to 0 and READS BACK 0 before a lethal event."""
        listed = self.rcon.command("scoreboard objectives list")
        if self.deaths_objective not in listed:
            raise InstrumentError(f"deaths objective {self.deaths_objective!r} is not on the server: {listed!r}")
        self.rcon.command(f"scoreboard players set {self.username} {self.deaths_objective} 0")
        if self.deaths() != 0:
            raise InstrumentError("deaths objective did not read back 0 after the reset")

    def read_food_state(self) -> dict[str, float]:
        """The TRUE food state over RCON (`data get entity`): the bridge clamps sensed saturation at 10
        while the apparatus heal sets ≈ 20 underneath — the reservoir that funds the regen-on margin
        (R3 confounding lens F17). Read at every event teleport; frozen in the gauntlet file."""
        out: dict[str, float] = {}
        for field in ("foodLevel", "foodSaturationLevel", "foodExhaustionLevel"):
            resp = self.rcon.command(f"data get entity {self.username} {field}")
            nums = _NUM_TOKEN.findall(resp.split(":")[-1])
            if not nums:
                raise InstrumentError(f"could not parse {field} from RCON reply: {resp!r}")
            out[field] = float(nums[-1])
        return out

    def check_surface_cell_air(self) -> None:
        """The pool's first air layer is AIR (a stone cap left by a drowning diagnostic makes every
        event a 45 s refusal with no other symptom)."""
        sx, _sy, sz = self.geom["submerged"]
        y = int(self.geom["surface_y"])
        resp = self.rcon.command(f"execute if block {int(sx)} {y} {int(sz)} minecraft:air")
        if "passed" not in resp.lower():
            raise Refusal(f"the pool's surface cell ({sx}, {y}, {sz}) is not air: {resp!r}")

    def set_gamerule(self, rule: str, value: str) -> str:
        """SET then READ BACK — the set's echo is not a verification (pilot: `regen_restored` was the echo)."""
        self.rcon.command(f"gamerule {rule} {value}")
        resp = self.rcon.command(f"gamerule {rule}").strip()
        if any(tok in resp.lower() for tok in _UNKNOWN_GAMERULE):
            raise InstrumentError(f"gamerule {rule!r} is not a rule on this server: {resp!r}")
        if value not in resp.lower():
            raise InstrumentError(f"gamerule {rule} did not read back {value}: {resp!r}")
        return resp

    def check_flee_anchor(self) -> dict[str, Any]:
        """`flee` through the BRIDGE on the shore, never the executor. The anchor must be set (the
        bridge takes `--flee_x/--flee_z` at start; without them it is world spawn, a no-path from the
        sealed room — the pilot recorded "No path to the goal!" in 15 ms and did NOT refuse). With
        the anchor AT the shore the call is a free success by construction; recorded as such."""
        self.rescue("flee-preflight")
        t0 = time.monotonic()
        try:
            out = self.aut.client.call_action("flee", {})
        except Exception as exc:
            out = {"ok": False, "detail": repr(exc)}
        res = {"latency_s": round(time.monotonic() - t0, 3), "bridge": {k: out.get(k) for k in ("ok", "detail")}}
        detail = str(out.get("detail") or "").lower()
        if not out.get("ok") or "no path" in detail or "no flee anchor" in detail or res["latency_s"] > 0.5:
            raise Refusal(f"flee anchor preflight (must answer fled ≤ 0.5 s): {res}", partial={"flee_preflight": res})
        return res

    def pain_between(self, t_a: float, t_b: float) -> list[dict[str, Any]]:
        return [
            s for s in self.signals if t_a <= s["t"] <= t_b and s["failure_mode"] in ("drive:oxygen", "drive:health")
        ]

    def rescue(self, label: str) -> float:
        """Shore + OBSERVED recovery + satiation. Returns the SENSED health on arrival
        (before the heal) so callers can see whether damage happened."""
        self.rcon.teleport(self.username, self.shore)
        vm = settle_until(
            self.aut,
            lambda vm: _f(vm, "is_in_water", 1) < 0.5 and _f(vm, "oxygen", 0) >= RECOVER_OXYGEN_MIN,
            timeout_s=15.0,
        )
        if vm is None:
            raise Refusal(f"{label}: rescue did not restore air on the shore")
        for key, want in self.settle_guard.items():
            if key not in vm:
                raise Refusal(f"{label}: settle guard sensor {key!r} absent from the snapshot")
            if _f(vm, key, want + 1.0) != want:
                raise Refusal(f"{label}: settle guard {key}={vm.get(key)} != {want}")
        arrival_health = _f(vm, "health", 20.0)
        self.heal()
        if (
            settle_until(
                self.aut,
                lambda vm: _f(vm, "health", 0) >= 20.0
                and _f(vm, "food", 0) >= 16.0
                and _f(vm, "saturation", 0) >= SATURATION_REST,
                timeout_s=10.0,
            )
            is None
        ):
            raise Refusal(
                f"{label}: heal/satiate never settled (health 20, food >= 16, saturation >= {SATURATION_REST})"
            )
        return arrival_health

    def stop_motion(self) -> None:
        try:
            self.aut.client.call_action("stop", {})
        except Exception as exc:
            print(f"WARNING: stop action raised: {exc!r}")

    def sample(self, t0: float) -> dict[str, Any] | None:
        if self.aut.client.state_age_s() > STALE_STATE_S:
            return None
        vm = sync_snapshot(self.aut)
        if vm is None or "is_in_water" not in vm:
            return None  # never let an absent key read as a surface
        return {
            "t": round(time.monotonic() - t0, 3),
            "in_water": _f(vm, "is_in_water", 0) >= 0.5,
            "health": _f(vm, "health", 20.0),
            "oxygen": _f(vm, "oxygen", 20.0),
        }

    def submerge(self, label: str) -> float:
        """Teleport to the pool floor; returns the TELEPORT time (the window's clock)."""
        self.stop_motion()
        t_tp = time.monotonic()
        self.rcon.teleport(self.username, self.sub)
        if settle_until(self.aut, lambda vm: _f(vm, "is_in_water", 0) >= 0.5, timeout_s=IN_WATER_WITHIN_S) is None:
            self.rescue(label)
            raise Refusal(f"{label}: is_in_water did not reflect the submerged teleport")
        return t_tp

    def encode_world_cluster(self) -> str | None:
        from maxim.runtime.agent_loop import _encode_current_clusters

        return _encode_current_clusters(self.encoder, self.agent_id, self.aut.executor).get("world")

    def world_cluster_margin(self) -> float | None:
        """The margin the LAST world encode resolved by (issue #786).

        The cluster id says WHICH node; this says how close the call was. Exp 62's NODE gate turns
        on that call and the apparatus runs ~0.06 from its threshold, so a row carrying only the id
        cannot say whether it resolved comfortably or by a hair. ``None`` = not measured (the
        min_delta gate bypassed the scan, or no encode has happened); ``-1.0`` = nothing comparable.
        Call it immediately after `encode_world_cluster`.
        """
        fn = getattr(self.encoder, "last_encode_margin", None)
        if fn is None:  # a fake encoder without the accessor: absent, not zero
            return None
        return fn(agent_id=self.agent_id, modality="world")

    # ── preflights ───────────────────────────────────────────────────────────────────

    def measure_bridge_cadence(self, seconds: float = 3.0) -> float | None:
        """Median interval between FRESH bridge snapshots, from the client's state-age resets."""
        arrivals: list[float] = []
        last_age = None
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            age = self.aut.client.state_age_s()
            if last_age is not None and age < last_age:  # a reset = a new snapshot arrived
                arrivals.append(time.monotonic() - age)
            last_age = age
            time.sleep(0.01)
        return median_interval_s(arrivals)

    def live_encoding(self) -> dict[str, Any]:
        """The encoding identity of the encoder this trial BOOKS and READS through (training's
        `propose_via_substrate` and the NODE-gate encodes), over the body's full declared roster."""
        from maxim.runtime.agent_loop import _read_world_ranges

        ex = self.aut.executor
        return encoding_identity(self.encoder.config, _read_world_ranges(ex), declared_world_roster(ex))

    @staticmethod
    def loop_encoder_config() -> dict[str, Any]:
        """The PROBE loop's encoder config. `run_agent_loop` builds its own `_loop_sensor_encoder`
        with no config argument and exposes no handle to it, so the class default IS its live
        config — held to the same frozen identity, so a changed source default refuses too."""
        from maxim.similarity.encoder import SensorEncoderConfig

        return encoder_config_identity(SensorEncoderConfig())

    def live_fingerprint(self) -> dict[str, Any]:
        from maxim.runtime.agent_loop import _read_world_ranges

        cfg = self.aut.bio.nac.config
        oxy = self.aut.executor.embodiment.root.drive_specs.get("oxygen")
        ranges = _read_world_ranges(self.aut.executor)
        return {
            "cluster_fear_alpha": cfg.cluster_fear_alpha,
            "max_cluster_fear": cfg.max_cluster_fear,
            "cluster_fear_threshold": cfg.cluster_fear_threshold,
            "cluster_fear_failure_modes": sorted(cfg.cluster_fear_failure_modes),
            # Read off the BUILT store, like every other row here -- not off config.json.
            "memory_strategy": str(self.aut.bio.hippocampus.config.memory_strategy),
            "encoder_pattern_threshold": float(self.encoder.config.pattern_threshold),
            "substrate_explore_bonus_weight": float(getattr(cfg, "substrate_explore_bonus_weight", 0.0)),
            "oxygen_drive": None
            if oxy is None
            else {"set_point": float(oxy.set_point), "comfort_band": float(oxy.comfort_band)},
            "sensor_ranges": {
                k: [float(v) for v in ranges[k]] for k in ("is_in_water", "oxygen", "saturation") if k in ranges
            },
        }

    def check_fingerprint(self, usable_oxygen_max: float) -> dict[str, Any]:
        cfg = self.aut.bio.nac.config
        oxy = self.aut.executor.embodiment.root.drive_specs.get("oxygen")
        live_fp = self.live_fingerprint()
        drift = fingerprint_drift(live_fp, self.frozen["fingerprint"])
        if drift:
            raise Refusal(f"config fingerprint drift on {drift}: live={live_fp}", partial={"fingerprint_live": live_fp})
        live_enc = self.live_encoding()
        enc_drift = encoding_drift(live_enc, self.encoding) + [
            f"loop.encoder_config.{k}"
            for k in fingerprint_drift(self.loop_encoder_config(), self.encoding.get("encoder_config") or {})
        ]
        if enc_drift:
            raise Refusal(
                f"encoding identity drift on {enc_drift} (issue #783): every world cluster id would re-key",
                partial={"fingerprint_live": live_fp, "encoding_live": live_enc},
            )
        if oxy is None or usable_oxygen_max >= oxy.set_point - oxy.comfort_band:
            raise Refusal("usable_oxygen_max does not sit below the oxygen comfort band (band-edge trap)")
        if "drive:oxygen" not in cfg.cluster_fear_failure_modes:
            raise Refusal("drive:oxygen not in the fear allowlist")
        return {**live_fp, "encoding": live_enc}

    def check_bridge(self) -> float:
        """Roster + freshness: returns the measured snapshot cadence (s)."""
        if settle_until(self.aut, lambda vm: "is_in_water" in vm and "oxygen" in vm, timeout_s=10.0) is None:
            raise Refusal("bridge never delivered state")
        missing = missing_bridge_sensors(self.aut.client.latest_state(), REQUIRED_BRIDGE_SENSORS)
        if missing:
            raise Refusal(f"the running bridge does not emit {sorted(missing)} — restart it from current main")
        # Sensor FRESHNESS: the DV clock reads is_in_water from the latest snapshot at 4 Hz, so the
        # bridge's snapshot interval must not exceed the sampling period. (The loop's own tick rate
        # no longer depends on it — agent_loop._substrate_tick_due, 2026-09-16.)
        cadence = self.measure_bridge_cadence()
        if cadence is None or cadence > self.frozen["bridge_state_interval_max_s"]:
            raise Refusal(
                f"bridge state cadence {cadence}s > {self.frozen['bridge_state_interval_max_s']}s — restart the bridge "
                "with --state_interval_ms=100 (sensor freshness: the snapshot interval must not exceed the 0.25 s sampling period)",
                partial={"bridge_state_interval_s": cadence},
            )
        return cadence

    def check_liveness(self) -> int:
        """The full loop, on the shore, must tick at its cadence. Returns the tick count."""
        from maxim.simulation.minecraft_harness import run_minecraft_aut
        from maxim.simulation.substrate_telemetry import SubstrateTelemetry

        self.rescue("liveness")
        live_path = self.persistence_dir / "telemetry_liveness.jsonl"
        live_telem = SubstrateTelemetry(log_path=live_path, agent_id=self.agent_id)
        live_stop = threading.Event()
        live_loop = threading.Thread(
            target=run_minecraft_aut,
            args=(self.aut,),
            kwargs={
                "max_steps": 100_000,
                "target_hz": self.frozen["loop_hz"],
                "stop_event": live_stop,
                "substrate_telemetry": live_telem,
            },
            daemon=True,
        )
        live_loop.start()
        time.sleep(self.frozen["loop_liveness_s"])
        live_stop.set()
        live_loop.join(timeout=20.0)
        self.stop_motion()
        self.reopen_hub_session()
        live_ticks = _telemetry_ticks(live_path, 0.0)
        if live_loop.is_alive() or len(live_ticks) < self.frozen["loop_liveness_min_ticks"]:
            raise Refusal(
                f"loop liveness: {len(live_ticks)} substrate tick(s) in {self.frozen['loop_liveness_s']}s on the shore "
                f"(need >= {self.frozen['loop_liveness_min_ticks']}) — the loop is not reaching its substrate branch; "
                "no window can measure anything (see loop_tick_probe.py)",
                partial={"loop_liveness_ticks": len(live_ticks)},
            )
        return len(live_ticks)

    def check_gamerules(self, rules: tuple[tuple[str, str], ...] = GAMERULES) -> None:
        for rule, want in rules:
            resp = self.rcon.command(f"gamerule {rule}").strip().lower()
            if any(tok in resp for tok in _UNKNOWN_GAMERULE):
                raise InstrumentError(f"gamerule {rule!r} is not a rule on this server: {resp!r}")
            if want not in resp:
                raise Refusal(f"gamerule {rule} is not {want} ({resp!r})")

    def check_clusters_distinct(self) -> tuple[str | None, str | None]:
        """LIVE cluster-distinct preflight on the live agent's EC. Submerges the bot once."""
        self.rescue("preflight")
        shore_cluster = self.encode_world_cluster()
        self.submerge("preflight")
        water_cluster = self.encode_world_cluster()
        if not water_cluster or water_cluster == shore_cluster:
            self.rescue("preflight")
            raise Refusal(
                f"shore and submerged encode to the same LIVE world cluster ({shore_cluster}) — "
                "the offline gate passed but the live EC does not separate; not a behavioural null"
            )
        return shore_cluster, water_cluster

    def check_escape_actuation(self) -> dict[str, Any]:
        """Escape actuation through the BACKEND, never the executor: an executor success would book
        a POSITIVE causal link that makes escape_water selectable with ZERO fear. The bridge action
        is called directly; bridge truth (is_in_water 0) decides. Assumes the bot is SUBMERGED."""
        t0 = time.monotonic()
        outcome: dict[str, Any] = {}

        def _bridge_escape() -> None:
            try:
                outcome.update(self.aut.client.call_action("escape_water", {}))
            except Exception as exc:
                outcome["ok"] = False
                outcome["detail"] = repr(exc)

        th = threading.Thread(target=_bridge_escape, daemon=True)
        th.start()
        surfaced_at = None
        while time.monotonic() - t0 < SURFACE_WITHIN_S + 2.0:
            s = self.sample(t0)
            if s is not None and not s["in_water"]:
                surfaced_at = s["t"]
                break
            time.sleep(0.25)
        th.join(timeout=10.0)
        result = {"t_surface": surfaced_at, "bridge": {k: outcome.get(k) for k in ("ok", "detail")}}
        if surfaced_at is None or surfaced_at > SURFACE_WITHIN_S:
            self.rescue("preflight")
            raise Refusal(
                f"escape actuation check FAILED ({outcome}) — head not in air within {SURFACE_WITHIN_S}s",
                partial={"actuation_preflight": result},
            )
        self.rescue("preflight")
        return result

    def positive_escape_links(self) -> int:
        return len(self.aut.bio.nac.get_positive_outcomes(f"tool:{self.escape_tool}"))

    def check_no_positive_escape_link(self) -> None:
        pos_links = self.positive_escape_links()
        if pos_links:
            raise Refusal(
                f"preflight seeded {pos_links} positive causal link(s) on escape_water — the probe would surface without fear"
            )

    # ── the loop window, the placement, the probe ────────────────────────────────────

    def loop_window(self, seconds: float, *, enter: Any, on_sample: Any, label: str) -> dict[str, Any]:
        """WARM the full loop on the shore, `enter()` the situation (returns the window's t0),
        sample at 4 Hz until `on_sample` says stop or the cap; then RESCUE FIRST (teleport to the
        shore) and only then stop/join the loop."""
        from maxim.simulation.minecraft_harness import run_minecraft_aut
        from maxim.simulation.substrate_telemetry import SubstrateTelemetry

        stop = threading.Event()
        actions0 = len(getattr(self.aut.executor, "_tools_succeeded", []) or [])
        calls0 = len(self.calls)
        telem_path = self.persistence_dir / f"telemetry_{label}_{int(time.time() * 1000)}.jsonl"
        telem = SubstrateTelemetry(log_path=telem_path, agent_id=self.agent_id)
        loop = threading.Thread(
            target=run_minecraft_aut,
            args=(self.aut,),
            kwargs={
                "max_steps": 100_000,
                "target_hz": self.frozen["loop_hz"],
                "stop_event": stop,
                "substrate_telemetry": telem,
            },
            daemon=True,
        )
        loop.start()
        time.sleep(self.frozen["loop_warm_s"])  # loop boot is NOT inside the window
        samples: list[dict[str, Any]] = []
        stale = 0
        t0 = enter()
        t_rescue = None
        stuck = False
        try:
            while time.monotonic() - t0 < seconds:
                s = self.sample(t0)
                if s is None:
                    stale += 1
                    if stale >= STALE_MAX_CONSECUTIVE:
                        raise InstrumentError(f"{label}: bridge stopped delivering fresh state inside a loop window")
                else:
                    stale = 0
                    samples.append(s)
                    if on_sample(s):
                        break
                time.sleep(0.25)
        finally:
            t_rescue = time.monotonic()
            try:
                self.rcon.teleport(self.username, self.shore)  # rescue BEFORE the loop drains
            except Exception as exc:
                print(f"WARNING: rescue teleport raised: {exc!r}")
            stop.set()
            loop.join(timeout=20.0)
            stuck = loop.is_alive()
            self.stop_motion()
            self.reopen_hub_session()
        if stuck:
            raise Refusal(f"{label}: loop thread did not stop")
        succeeded = list(getattr(self.aut.executor, "_tools_succeeded", []) or [])[actions0:]
        ticks = _telemetry_ticks(telem_path, t0)
        window_calls = [{**c, "t": round(c["t"] - t0, 3)} for c in self.calls[calls0:]]
        return {
            "samples": samples,
            "actions": succeeded,
            "calls": window_calls,
            "ticks": ticks,
            "t0": t0,
            "t_rescue": t_rescue,
            "us_events": self.pain_between(t0, t_rescue),
        }

    def placement(self, label: str, *, window_label: str | None = None) -> dict[str, Any]:
        """One US-free placement on the pool floor with the full loop live — Exp 60's DV unit.
        Rescues before and after; classifies from bridge truth; marks DIRTY on any US or damage."""
        self.rescue(label)
        seen = {"water": False, "h0": None}

        def _until(s: dict[str, Any]) -> bool:
            if seen["h0"] is None:
                seen["h0"] = s["health"]
            if s["health"] < seen["h0"]:
                return True  # damage: rescue NOW (dirty placement)
            if s["in_water"]:
                seen["water"] = True
                return False
            return seen["water"]  # first head-in-air read after being submerged

        win = self.loop_window(
            self.probe_cap_s, enter=lambda: self.submerge(label), on_sample=_until, label=window_label or label
        )
        arrival_health = self.rescue(f"{label}-after")
        cls = classify_placement(win["samples"], cap_s=self.probe_cap_s)
        cls["actions"] = win["actions"]
        cls["calls"] = win["calls"]  # every executor call, incl. failures
        cls["ticks"] = win["ticks"]  # every loop tick: proposal + what the loop saw
        cls["escape_water_calls"] = sum(1 for c in win["calls"] if str(c["tool"]).endswith("_escape_water"))
        cls["flee_calls"] = sum(1 for c in win["calls"] if str(c["tool"]).endswith("_flee"))
        cls["us_events"] = win["us_events"]
        cls["arrival_health"] = arrival_health
        if win["us_events"] or arrival_health < 20.0:
            # the window was NOT US-free / damage-free: exclude and count as unyoked exposure
            cls["dirty"] = True
            cls["surfaced"] = False
            cls["censored"] = True
        proposed = [t["proposal"] for t in win["ticks"] if t.get("proposal")]
        print(
            f"  {label}: {'SURFACED %.2fs' % cls['latency_s'] if cls['surfaced'] else 'censored'}"
            f"{' [DIRTY]' if cls['dirty'] else ''} ticks={len(win['ticks'])} proposed={proposed[:4]} "
            f"calls={[(c['tool'], c['success']) for c in win['calls']][:4]}"
        )
        return cls

    def check_death_cap(self) -> None:
        if self.deaths() - self.deaths0 > self.frozen["death_cap"]:
            raise Refusal(f"death cap exceeded ({self.deaths() - self.deaths0})")

    def probe(self, label: str) -> dict[str, Any]:
        """Exp 60's probe: P placements + a shore free-roam window (activity control)."""
        placements = []
        for i in range(self.frozen["placements_per_probe"]):
            placements.append(self.placement(f"{label}-placement-{i}", window_label=label))
            self.check_death_cap()
        # Shore free-roam: activity control + P(enter water) secondary (structurally near 0 v 0 —
        # no drive fires on the shore; it re-tests specificity, recorded not gated)
        self.rescue(f"{label}-roam")
        roam = self.loop_window(
            self.frozen["shore_roam_s"], enter=time.monotonic, on_sample=lambda s: s["in_water"], label=f"{label}-roam"
        )
        self.rescue(f"{label}-roam-end")
        return {
            "placements": placements,
            "p_surface": p_surface({"placements": placements}),
            "cap_s": self.probe_cap_s,
            "unyoked_us_events": sum(len(p["us_events"]) for p in placements),
            "positive_escape_links": self.positive_escape_links(),
            "shore_roam": {
                "actions": roam["actions"],
                "entered_water": any(s["in_water"] for s in roam["samples"]),
                "window_s": self.frozen["shore_roam_s"],
            },
        }

    # ── training + the live G2 read ──────────────────────────────────────────────────

    def train(self) -> tuple[dict[str, Any], list[str]]:
        """K yoked, propose-only conditioning episodes at the pool floor (no execution)."""
        from maxim.runtime.agent_loop import propose_via_substrate

        fz = self.frozen
        usable = 0
        attempts = 0
        episode_clusters: list[str] = []
        deadline = time.monotonic() + fz["K_usable_episodes"] * (self.train_cap_s + 20.0) * 1.5
        while usable < fz["K_usable_episodes"] and time.monotonic() < deadline:
            attempts += 1
            self.rescue(f"train-{attempts}")
            t0 = self.submerge(f"train-{attempts}")
            while time.monotonic() - t0 < self.train_cap_s:
                n_before = len(self.signals)
                propose_via_substrate(
                    nac=self.aut.bio.nac,
                    agent_id=self.agent_id,
                    executor=self.aut.executor,
                    sensor_encoder=self.encoder,
                )
                new = [
                    s
                    for s in self.signals[n_before:]
                    if s["failure_mode"] == "drive:oxygen" and s["intensity"] >= fz["usable_pain_intensity_min"]
                ]
                vm = sync_snapshot(self.aut) or {}
                noted = self.aut.bio.nac.active_clusters(self.agent_id).get("world")
                if (
                    new
                    and noted
                    and _f(vm, "is_in_water", 0) >= 0.5
                    and _f(vm, "oxygen", 99) <= fz["usable_oxygen_max"]
                ):
                    episode_clusters.append(noted)
                    usable += 1
                    print(f"  training: usable episode {usable}/{fz['K_usable_episodes']} (oxygen {vm.get('oxygen')})")
                    break
                time.sleep(1.0 / fz["loop_hz"])
            arrival_health = self.rescue(f"train-{attempts}-end")
            if arrival_health < 20.0:
                raise Refusal(
                    f"drowning DAMAGE during propose-only training (health {arrival_health}) — the cap did not keep conditioning pre-damage"
                )
            for _ in range(4):  # healthy ticks: the latch observes recovery
                propose_via_substrate(
                    nac=self.aut.bio.nac,
                    agent_id=self.agent_id,
                    executor=self.aut.executor,
                    sensor_encoder=self.encoder,
                )
                time.sleep(0.25)
            self.check_death_cap()
        training = {
            "usable_episodes": usable,
            "attempts": attempts,
            "oxygen_pain_signals": sum(1 for s in self.signals if s["failure_mode"] == "drive:oxygen"),
            "health_pain_signals": sum(1 for s in self.signals if s["failure_mode"] == "drive:health"),
            "episode_clusters": episode_clusters,
            "deaths": self.deaths() - self.deaths0,
        }
        if usable < fz["K_usable_episodes"]:
            raise Refusal(f"only {usable}/{fz['K_usable_episodes']} usable episodes", partial={"training": training})
        if training["health_pain_signals"]:
            raise Refusal("drowning DAMAGE pain fired during propose-only training", partial={"training": training})
        return training, episode_clusters

    def fear_dump(self) -> dict[str, float]:
        lock = getattr(self.aut.bio.nac, "_lock", None)
        with lock if lock is not None else _NullCtx():
            return {
                f"{cid}|{fm}": v
                for (aid, cid, fm), v in getattr(self.aut.bio.nac, "_cluster_fear", {}).items()
                if aid == self.agent_id
            }

    def live_g2(self, arm: str, episode_clusters: list[str], water_cluster_pre: str | None) -> dict[str, Any]:
        """The PRODUCTION read (`anticipatory_threat_need`) on the probe-activated underwater cluster
        and every distinct episode cluster; the ABLATED arm must carry ZERO fear. Submerges once.
        Returns the record fields (`water_fear`, `shore_fear`, `cluster_fear_dump`, `live_g2`);
        raises Refusal when the read fails."""
        nac = self.aut.bio.nac
        self.submerge("g2")
        water_cluster = self.encode_world_cluster()
        water_margin = self.world_cluster_margin()
        # The RAW sensor state at the moment of the reading that decides the gate. Exp 62 reads one
        # agent in two pools, and light/time are the full-weight constants the cross-pool cosine
        # rests on — so when the read misses, the record must be able to say WHICH live thing
        # differed instead of only "something live". Captured here because this is the only moment
        # the body is submerged at the pool being read; a snapshot taken by the caller before or
        # after this call is at the SHORE, and before `submerge` it is at the previous pool.
        sensed_probe = self.read_context()
        self.rescue("g2")
        shore_cluster = self.encode_world_cluster()
        shore_margin = self.world_cluster_margin()
        sensed_shore = self.read_context()
        theta = float(nac.config.cluster_fear_threshold)
        water_fear = round(nac.cluster_fear(self.agent_id, water_cluster), 4)
        shore_fear = round(nac.cluster_fear(self.agent_id, shore_cluster), 4) if shore_cluster else 0.0
        need_probe = nac.anticipatory_threat_need(self.agent_id, {"world": water_cluster})
        need_episodes = {
            cid: nac.anticipatory_threat_need(self.agent_id, {"world": cid}) for cid in set(episode_clusters)
        }
        # the loop's activation floor is STRICT (> 0.5): a need of exactly θ is dead at recall
        live_floor = 0.5
        if arm == "fear":
            g2_pass = need_probe > live_floor and all(n > live_floor for n in need_episodes.values())
        else:
            g2_pass = water_fear == 0.0 and shore_fear == 0.0 and need_probe == 0.0
        fields = {
            "water_fear": water_fear,
            "shore_fear": shore_fear,
            "cluster_fear_dump": self.fear_dump(),
            "live_g2": {
                "pre_water_cluster": water_cluster_pre,
                "training_majority_cluster": max(set(episode_clusters), key=episode_clusters.count)
                if episode_clusters
                else None,
                "probe_water_cluster": water_cluster,
                "probe_shore_cluster": shore_cluster,
                "distinct_episode_clusters": len(set(episode_clusters)),
                "need_probe_cluster": need_probe,
                "need_episode_clusters": need_episodes,
                "sensed_at_probe": sensed_probe,
                "sensed_at_shore": sensed_shore,
                # how CLOSE the call was, not just which node it landed on (issue #786)
                "probe_margin": water_margin,
                "shore_margin": shore_margin,
                "specificity_ok": abs(shore_fear) < self.frozen["specificity_ratio"] * abs(water_fear)
                if water_fear
                else None,
                "pass": g2_pass,
            },
        }
        if not g2_pass:
            raise Refusal(
                f"LIVE G2 FAILED ({arm}): need on probe cluster={need_probe}, on episode clusters={need_episodes}, "
                f"water fear={water_fear} shore fear={shore_fear} (θ={theta}, floor {live_floor}) — fear not readable at "
                "recall through the production read; must not ship as a behavioural null",
                partial=fields,
            )
        return fields

    def read_context(self) -> dict[str, Any]:
        """The full-weight constants and the place absolutes as the BODY senses them right now."""
        state = self.aut.client.latest_state() or {}
        return {
            k: state.get(k)
            for k in ("light_level", "time_of_day", "y_altitude", "distance_from_spawn", "is_in_water", "oxygen")
        }

    def negative_links(self) -> dict[str, int | None]:
        nac = self.aut.bio.nac
        return {
            "escape_negative_links": len(nac.get_negative_outcomes(f"tool:{self.escape_tool}")),
            "flee_negative_links": len(nac.get_negative_outcomes(f"tool:{self.flee_tool}")) if self.flee_tool else None,
        }

    # ── teardown ─────────────────────────────────────────────────────────────────────

    # ── R3: the lethal window (no rescue inside; the pilot's `live_window` with its must-nots fixed) ──

    def _drive_specs(self) -> dict[str, Any]:
        specs: dict[str, Any] = {}
        root = getattr(getattr(self.aut.executor, "embodiment", None), "root", None)
        if root is None:
            return specs
        for ent in root.walk():
            for name, spec in getattr(ent, "drive_specs", {}).items():
                specs[name.split(".", 1)[-1]] = spec
        return specs

    def sample_full(self, t0: float, deaths0: int) -> dict[str, Any] | None:
        """One 4 Hz sample: the snapshot FIRST, then the `deaths` objective (that order is load-bearing —
        reversed, a respawn snapshot reads as a surface). Carries the settle-guard keys."""
        age = self.aut.client.state_age_s()
        if age > STALE_STATE_S:
            return None
        vm = sync_snapshot(self.aut)
        if vm is None or "is_in_water" not in vm:
            return None
        return {
            "t": round(time.monotonic() - t0, 3),
            "state_age_s": round(age, 3),
            "in_water": _f(vm, "is_in_water", 0) >= 0.5,
            "health": _f(vm, "health", 20.0),
            "oxygen": _f(vm, "oxygen", 20.0),
            "food": _f(vm, "food", -1.0),
            "saturation": _f(vm, "saturation", -1.0),
            "y": _f(vm, "y_altitude", -1.0),
            "is_raining": _f(vm, "is_raining", -1.0),
            "nearest_player_dist": _f(vm, "nearest_player_dist", -1.0),
            "hostile_count": _f(vm, "hostile_count", -1.0),
            "deaths_delta": self.deaths() - deaths0,
        }

    def lethal_event(self, label: str, *, cap_s: float, hold_hz: float = 4.0) -> dict[str, Any]:
        """ONE unrescued submersion with the loop live — R3's unit of measurement.

        Ends at the FIRST of: the head clears (the first sample with `is_in_water` 0 — the eye-block
        sensor; feet stay below the water line) → teleport to the shore IMMEDIATELY, then stop the
        loop (the pilot joined first and left a re-sinking agent exposed for the join; no linger);
        death (`deaths_delta` > 0, corroborated by the respawn discontinuity within one sample) →
        respawn already did it; the cap → teleport + Refusal. One wall clock `t0_wall` stamped at the
        teleport makes ticks, calls and samples comparable. Decision provenance from `RecommendCapture`
        (the pilot's drive column was empty). Pain-seconds are integrated from the SAMPLE series to
        `t_surface` with `drive_pain_for_value` (never from the publishes, which count deepenings).
        """
        from maxim.embodiment.sem import drive_pain_for_value
        from maxim.simulation.minecraft_harness import run_minecraft_aut
        from maxim.simulation.substrate_telemetry import SubstrateTelemetry
        from exp56.common import RecommendCapture

        specs = self._drive_specs()
        stop = threading.Event()
        calls0 = len(self.calls)
        sig0 = len(self.signals)
        telem_path = self.persistence_dir / f"telemetry_{label}_{int(time.time() * 1000)}.jsonl"
        telem = SubstrateTelemetry(log_path=telem_path, agent_id=self.agent_id)
        loop = threading.Thread(
            target=run_minecraft_aut,
            args=(self.aut,),
            kwargs={
                "max_steps": 100_000,
                "target_hz": self.frozen["loop_hz"],
                "stop_event": stop,
                "substrate_telemetry": telem,
            },
            daemon=True,
        )
        samples: list[dict[str, Any]] = []
        end: str | None = None
        t_surface: float | None = None
        t_death: float | None = None
        guard_breach: dict[str, Any] | None = None
        stale = 0
        self.preflight_deaths_objective()
        period = 1.0 / hold_hz
        t0: float | None = None
        t0_wall: float | None = None
        deaths0 = 0
        food_at_teleport: dict[str, float] = {}
        with RecommendCapture() as cap:
            try:
                loop.start()  # inside the try: a refused submerge must STOP this thread (a leaked loop keeps
                # emitting NAc_RECOMMEND into the process-wide sink and drains the agent under a closed client)
                time.sleep(self.frozen["loop_warm_s"])  # loop boot is NOT inside the window
                deaths0 = self.deaths()
                self.stop_motion()
                food_at_teleport = self.read_food_state()  # right before the teleport
                t0_wall = time.time()
                t0 = self.submerge(label)
                while time.monotonic() - t0 < cap_s:
                    tick_start = time.monotonic()
                    s = self.sample_full(t0, deaths0)
                    if s is None:
                        stale += 1
                        if stale >= STALE_MAX_CONSECUTIVE:
                            raise InstrumentError(f"{label}: bridge stopped delivering fresh state inside the window")
                    else:
                        stale = 0
                        samples.append(s)
                        if self.settle_guard and guard_breach is None:
                            for key, want in self.settle_guard.items():
                                if key in s and abs(float(s[key]) - float(want)) > 1e-6:
                                    guard_breach = {"key": key, "value": s[key], "t": s["t"]}
                        if s["deaths_delta"] > 0:
                            # the scoreboard may LEAD the respawn snapshot by up to a sample (a death landing
                            # between the snapshot's timestamp and the RCON reply): keep sampling briefly for
                            # the respawn discontinuity, corroborate on THAT sample; t_death = this one
                            end, t_death = "death", s["t"]
                            for _ in range(6):
                                if not s["in_water"] and s["health"] >= 20.0:
                                    break
                                time.sleep(period)
                                nxt = self.sample_full(t0, deaths0)
                                if nxt is not None:
                                    samples.append(nxt)
                                    s = nxt
                            break
                        if not s["in_water"]:
                            end, t_surface = "surface", s["t"]
                            self.rcon.teleport(self.username, self.shore)  # the exit, BEFORE the loop stops
                            break
                    time.sleep(max(0.0, period - (time.monotonic() - tick_start)))
                if end is None:
                    end = "cap"
            finally:
                t_end = time.monotonic()
                if end != "surface" and end != "death":
                    try:  # the cap AND any exception inside the window: never leave the bot underwater
                        self.rcon.teleport(self.username, self.shore)
                    except Exception as exc:
                        print(f"WARNING: post-window teleport raised: {exc!r}")
                stop.set()
                loop.join(timeout=20.0)
                stuck = loop.is_alive()
                self.stop_motion()
                self.reopen_hub_session()
            events = [
                {**dict(e.get("data", {})), "t": e.get("t")}
                for e in cap.events
                if e.get("agent_id") in (None, self.agent_id)  # never another row's leaked events
            ]
        if t0 is None:
            raise InstrumentError(f"{label}: the event never started")
        partial_evidence = {
            "samples": samples,
            "calls": [{**c, "t": round(c["t"] - t0, 3)} for c in self.calls[calls0:]],
        }
        if stuck:
            raise Refusal(f"{label}: loop thread did not stop", partial={"event_partial": partial_evidence})
        deaths_after = self.deaths() - deaths0
        if end != "death" and deaths_after > 0:
            raise Refusal(
                f"INSTRUMENT: {label}: deaths rose AFTER the window ended as {end!r} (a post-window death)",
                partial={"event_partial": partial_evidence},
            )
        if end == "death":
            after = [x for x in samples if x["t"] >= (t_death or 0.0)]
            if not after or after[-1]["in_water"] or after[-1]["health"] < 20.0:
                raise Refusal(
                    f"INSTRUMENT: {label}: death read but no respawn discontinuity within {len(after)} sample(s)",
                    partial={"event_partial": partial_evidence},
                )
        cut = t_surface if t_surface is not None else (t_death if t_death is not None else cap_s)
        window = [x for x in samples if x["t"] <= cut]
        pain_s: dict[str, float] = {}
        for drive in ("oxygen", "health"):
            spec = specs.get(drive)
            total = 0.0
            for a, b in zip(window, window[1:]):
                if spec is not None:
                    total += drive_pain_for_value(spec, float(a[drive])) * (b["t"] - a["t"])
            pain_s[drive] = round(total, 3) if spec is not None else None
        window_calls = [
            {**c, "t": round(c["t"] - t0, 3), "t_return": round(c["t_return"] - t0, 3)} for c in self.calls[calls0:]
        ]
        for c in window_calls:
            c["post_event"] = c["t"] > cut
            if end == "death" and t_death is not None and c["t"] <= t_death < c["t_return"] and c.get("success"):
                c["surfaced_by_respawn"] = True
        ticks = _telemetry_ticks(telem_path, t0, t0_wall=t0_wall)
        if not ticks:
            raise Refusal(
                f"INSTRUMENT: {label}: the loop wrote no telemetry ticks ({telem_path.name})",
                partial={"event_partial": partial_evidence},
            )
        in_window = [
            t for t in ticks if "t" in t and 0.0 <= t["t"] <= (t_end - t0)
        ]  # the cadence band is IN-WATER ticks
        periods = sorted(b["t"] - a["t"] for a, b in zip(in_window, in_window[1:]))
        escape_events = [
            e for e in events if str(e.get("best_tool", "")).endswith("escape_water") and e.get("passed_gate") is True
        ]
        first_escape_call = next((c for c in window_calls if str(c["tool"]).endswith("escape_water")), None)
        health_in_window = [x["health"] for x in window]
        t_first_damage = next((x["t"] for x in samples if x["health"] < 20.0), None)
        pain = [
            {"t": round(p["t"] - t0, 3), "failure_mode": p["failure_mode"], "intensity": p["intensity"]}
            for p in self.signals[sig0:]
            if p["t"] <= t_end
        ]
        row = {
            "label": label,
            "end": end,
            "t0_wall": t0_wall,
            "t_surface": t_surface,
            "t_death": t_death,
            "t_end": round(t_end - t0, 3),
            "survived": end == "surface",
            "t_first_damage": t_first_damage,
            # the event's OWN observed onset decides (env lens N-2); the anchor's minimum is reported beside it
            "escaped_before_damage": t_surface is not None and (t_first_damage is None or t_surface < t_first_damage),
            "escaped_before_anchor_onset": t_surface is not None
            and t_surface < float(self.geom["measured"]["t_damage_onset_min_s"]),
            "min_health": min(health_in_window) if health_in_window else None,
            "health_lost": round(20.0 - min(health_in_window), 3) if health_in_window else None,
            "min_oxygen": min(x["oxygen"] for x in window) if window else None,
            "pain_seconds": pain_s,
            "pain_publishes": pain,
            "calls": window_calls,
            "t_first_call": window_calls[0]["t"] if window_calls else None,
            "t_escape_call": first_escape_call["t"] if first_escape_call else None,
            "decision_events": events[:12],
            "executed_escape_event": escape_events[0] if escape_events else None,
            "ticks": ticks,
            "tick_period_median_s": round(_median(periods), 3) if periods else None,
            "tick_period_iqr_s": round(periods[3 * len(periods) // 4] - periods[len(periods) // 4], 3)
            if len(periods) >= 4
            else None,
            "sample_period_median_s": round(_median(sorted(b["t"] - a["t"] for a, b in zip(samples, samples[1:]))), 3)
            if len(samples) > 1
            else None,
            "max_state_age_s": max((x["state_age_s"] for x in samples), default=None),
            "food_at_teleport": food_at_teleport,
            "guard_breach": guard_breach,
            "samples": samples,
            "deaths_delta_after": deaths_after,
        }
        if end == "cap":
            raise Refusal(
                f"{label}: alive underwater at the {cap_s:.0f} s cap — an instrument fault", partial={"event": row}
            )
        return row

    def final_rescue(self) -> None:
        try:
            self.rcon.teleport(self.username, self.shore)
            self.heal()
        except Exception as exc:
            print(f"WARNING: final rescue raised: {exc!r}")
