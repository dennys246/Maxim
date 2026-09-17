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


def _telemetry_ticks(path: Path, t0_monotonic: float) -> list[dict[str, Any]]:
    """Compress a window's SubstrateTelemetry JSONL into per-tick rows (pure over the file).

    Rows carry wall-clock ``ts``; the window's clock is monotonic, so ticks are reported
    relative to the FIRST row (the loop's first tick) and the proposal/tool per tick.
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
        out.append(
            {
                "t_from_first_tick": round(float(r.get("ts", 0.0)) - float(first_ts), 3),
                "step": r.get("step"),
                "proposal": prop.get("tool_name") or prop.get("tool") if isinstance(prop, dict) else prop,
                "gated": r.get("gated"),
                "active_clusters": nac.get("active_clusters") if isinstance(nac, dict) else None,
                "drives": {k: v for k, v in (r.get("drives") or {}).items() if k in ("threat", "oxygen", "health")}
                if isinstance(r.get("drives"), dict)
                else None,
            }
        )
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
    ) -> None:
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

    def attach_instruments(self) -> None:
        self.aut.bio.pain_bus.subscribe(self._record_pain)
        self._orig_execute = self.aut.executor.execute
        orig = self._orig_execute
        calls = self.calls

        def _spy_execute(action: dict[str, Any]) -> Any:
            t = time.monotonic()
            try:
                out = orig(action)
            except Exception as exc:
                calls.append({"t": t, "tool": (action or {}).get("tool_name"), "success": False, "error": repr(exc)})
                raise
            calls.append(
                {
                    "t": t,
                    "tool": (action or {}).get("tool_name"),
                    "success": getattr(out, "success", None),
                    "error": getattr(out, "error", None),
                }
            )
            return out

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

    def deaths(self) -> int:
        resp = self.rcon.command(
            f"scoreboard players get {self.username} {self.geom.get('deaths_objective', 'exp60_deaths')}"
        )
        try:
            return int(resp.split(" has ")[1].split()[0])
        except (IndexError, ValueError):
            return 0

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

    def live_fingerprint(self) -> dict[str, Any]:
        from maxim.runtime.agent_loop import _read_world_ranges
        from maxim.similarity.encoder import SensorEncoderConfig

        cfg = self.aut.bio.nac.config
        oxy = self.aut.executor.embodiment.root.drive_specs.get("oxygen")
        ranges = _read_world_ranges(self.aut.executor)
        return {
            "cluster_fear_alpha": cfg.cluster_fear_alpha,
            "max_cluster_fear": cfg.max_cluster_fear,
            "cluster_fear_threshold": cfg.cluster_fear_threshold,
            "cluster_fear_failure_modes": sorted(cfg.cluster_fear_failure_modes),
            "encoder_pattern_threshold": float(SensorEncoderConfig().pattern_threshold),
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
        if oxy is None or usable_oxygen_max >= oxy.set_point - oxy.comfort_band:
            raise Refusal("usable_oxygen_max does not sit below the oxygen comfort band (band-edge trap)")
        if "drive:oxygen" not in cfg.cluster_fear_failure_modes:
            raise Refusal("drive:oxygen not in the fear allowlist")
        return live_fp

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

    def check_gamerules(self) -> None:
        for rule, want in GAMERULES:
            resp = self.rcon.command(f"gamerule {rule}").strip().lower()
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
        self.rescue("g2")
        shore_cluster = self.encode_world_cluster()
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

    def negative_links(self) -> dict[str, int | None]:
        nac = self.aut.bio.nac
        return {
            "escape_negative_links": len(nac.get_negative_outcomes(f"tool:{self.escape_tool}")),
            "flee_negative_links": len(nac.get_negative_outcomes(f"tool:{self.flee_tool}")) if self.flee_tool else None,
        }

    # ── teardown ─────────────────────────────────────────────────────────────────────

    def final_rescue(self) -> None:
        try:
            self.rcon.teleport(self.username, self.shore)
            self.heal()
        except Exception as exc:
            print(f"WARNING: final rescue raised: {exc!r}")
