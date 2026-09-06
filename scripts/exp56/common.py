"""Exp 56 four-arm sharing benchmark — the harness core (frozen apparatus).

Implements `docs/experiments/protocols/exp56_four_arm_sharing_preregistration.md`
against the SHIPPED seams only:

* per-AUT assembly via ``simulation.minecraft_harness.build_minecraft_aut``
  (the canonical builders) on ``bodies/minecraft_bench``;
* the world encode via the production ``SensorEncoder`` against the AUT's
  own real ``EntorhinalCortex`` (the L11-remeasure replay lineage);
* action execution via ``Executor.execute`` and learning via
  ``runtime.tool_dispatch.record_outcome`` (the loop's own intake) under
  ``MAXIM_OPERANT_ONLY_CREDIT=1``;
* the TEACHER via ``NAc.credit_operant_reward`` with the Exp 52
  relief-signed value (``reactive_mother_tick``'s exact computation,
  through the same ``tool_bridge`` helpers);
* export/ingest via the REAL CLI (``run_substrate_subcommand``) — never
  bare merge calls;
* decision provenance via a registered ``sim_logger`` sink over the
  ``NAc_RECOMMEND`` channel (the production telemetry surface).

FROZEN CONSTANTS live in :data:`FROZEN` — the prereg's sign-off records
them at this file's merge commit; the analyzer refuses drift (constants
are extended, never retuned).

The opaque-name -> bridge-action permutation lives in
:class:`TranslatingClient`, per pair, shared by the pair's donor and
receiver (the transferred tool signature must name the same physical
action on both sides).
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import socket
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

BODY_REF = "bodies/minecraft_bench"
BODY_REF_SATIATED = "bodies/minecraft_bench_satiated"
ENTITY_NAME = "minecraft_bench"
BODY_SPEC_YAML = Path(__file__).resolve().parent / "body_spec_minecraft_bench.yaml"

#: The eight opaque roster affordances (tool names are ENTITY_NAME + "_" + aff).
AFFORDANCES: tuple[str, ...] = ("aff_a", "aff_b", "aff_c", "aff_d", "aff_e", "aff_f", "aff_g", "aff_h")
ROSTER: tuple[str, ...] = tuple(f"{ENTITY_NAME}_{a}" for a in AFFORDANCES)

#: The eight underlying bridge actions — ALL always-executable turns (the
#: link-balance obligation: every scheduled action mechanically succeeds in
#: the controlled world; turns also never perturb the situation's location
#: sensors mid-trial). The per-pair permutation maps AFFORDANCES onto these.
BRIDGE_ACTIONS: tuple[tuple[str, dict[str, float]], ...] = tuple(
    ("turn", {"degrees": float(d)}) for d in (15, -15, 30, -30, 60, -60, 90, -90)
)

#: Frozen constants (prereg §Selector, §Arms, §Apparatus + sign-off).
FROZEN: dict[str, Any] = {
    "roster_k": 8,
    "schedule_reps_per_cell": 6,  # K = 8 affs x 2 situation-states x 6 = 96 trials/donor
    "epsilon": 0.2,
    "min_confidence": 0.3,
    "donor_bias_floor": 0.4,
    "link_spread_max": 0.05,
    "link_count_tolerance": 1,
    "probe_precontact_max": 10,
    "probe_tail": 5,
    "feed_amount": 0.5,
    "relief_epsilon": 1e-9,
    # Candidate contingency slots (world coordinates the RCON /tp uses; the
    # per-pair seed picks one, identical across that pair's arms). Chosen so
    # the situation signature spans >= 2 declared world sensors vs the rest
    # anchor: distance_from_spawn (far vs 0) + light_level (roofed dark spot
    # vs open) + y_altitude on the elevated slots.
    "rest_anchor": {"x": 0, "y": 64, "z": 0},
    "contingency_slots": [
        {"x": 40, "y": 64, "z": 0},
        {"x": -40, "y": 64, "z": 8},
        {"x": 8, "y": 70, "z": 40},
        {"x": -8, "y": 70, "z": -40},
    ],
}

SCHEDULE_K = FROZEN["roster_k"] * 2 * FROZEN["schedule_reps_per_cell"]


# ── per-pair action-name permutation (the opaque-name mitigation) ────────


class TranslatingClient:
    """Per-pair opaque-affordance -> bridge-action translation proxy.

    Wraps a real ``MinecraftClient`` (or the fake bridge's): the SEM
    affordance name (``aff_c``) is translated to its per-pair bridge action
    before dispatch; everything else passes through. The permutation is
    seeded by the pair seed so donor and receiver SHARE it (the transferred
    tool signature must mean the same physical action on both sides) while
    the name<->action coupling varies across pairs (review finding I1: the
    deterministic name tiebreak must not couple the floor to the alphabet).
    """

    def __init__(self, inner: Any, *, pair_seed: int) -> None:
        self._inner = inner
        rng = random.Random(pair_seed * 7919 + 13)
        order = list(range(len(BRIDGE_ACTIONS)))
        rng.shuffle(order)
        self.action_map: dict[str, tuple[str, dict[str, float]]] = {
            aff: BRIDGE_ACTIONS[order[i]] for i, aff in enumerate(AFFORDANCES)
        }

    def call_action(self, name: str, params: "dict[str, Any] | None" = None) -> dict[str, Any]:
        mapped = self.action_map.get(name)
        if mapped is None:
            return self._inner.call_action(name, params)
        bridge_name, bridge_params = mapped
        return self._inner.call_action(bridge_name, dict(bridge_params))

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


# ── world control (live RCON; mock no-op) ────────────────────────────────


class RconControl:
    """Minimal Minecraft RCON client for the world-script commands (/tp).

    Hand-rolled (no new dependency): login packet type 3, command type 2,
    little-endian length-prefixed frames. Live-bridge campaigns only; the
    ``--mock`` smoke uses :class:`NullWorldControl`.
    """

    def __init__(self, host: str, port: int, password: str) -> None:
        self._sock = socket.create_connection((host, port), timeout=10)
        self._req = 0
        self._send(3, password)
        rid, _ = self._recv()
        if rid == -1:
            raise RuntimeError("RCON authentication failed")

    def _send(self, ptype: int, body: str) -> int:
        self._req += 1
        payload = struct.pack("<ii", self._req, ptype) + body.encode("utf-8") + b"\x00\x00"
        self._sock.sendall(struct.pack("<i", len(payload)) + payload)
        return self._req

    def _recv(self) -> tuple[int, str]:
        raw = b""
        while len(raw) < 4:
            raw += self._sock.recv(4 - len(raw))
        (length,) = struct.unpack("<i", raw)
        body = b""
        while len(body) < length:
            body += self._sock.recv(length - len(body))
        rid, _ptype = struct.unpack("<ii", body[:8])
        return rid, body[8:-2].decode("utf-8", errors="replace")

    def command(self, cmd: str) -> str:
        self._send(2, cmd)
        _rid, text = self._recv()
        return text

    def teleport(self, bot_name: str, pos: dict[str, float]) -> None:
        self.command(f"tp {bot_name} {pos['x']} {pos['y']} {pos['z']}")

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class NullWorldControl:
    """Inert world control (unit-test plumbing only)."""

    def teleport(self, bot_name: str, pos: dict[str, float]) -> None:  # noqa: ARG002
        return

    def command(self, cmd: str) -> str:  # noqa: ARG002
        return ""

    def close(self) -> None:
        return


class ScriptedBridgeServer:
    """The ``--mock`` instrument: a DETERMINISTIC scripted world speaking the
    frozen NDJSON protocol, whose snapshots FOLLOW the commanded anchor.

    The paired :class:`ScriptedWorldControl`'s teleport sets the anchor;
    the served world sensors derive from it (distance_from_spawn from the
    coordinates, light dark at far slots, y from the anchor) with a small
    seeded jitter — so situation placement is real IN THE MEASURED STREAM
    while remaining fully scripted. Dev/smoke/guard-test instrument ONLY,
    never a confirmatory record (prereg §Apparatus: the campaign runs the
    live bridge; the ``FakeBridgeServer`` lineage's role, made
    situation-capable so the wiring smoke can exercise the whole chain
    deterministically).
    """

    def __init__(self, *, seed: int = 42, state_interval_s: float = 0.02) -> None:
        self._rng = random.Random(seed)
        self.anchor: dict[str, float] = dict(FROZEN["rest_anchor"])
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(4)
        self.port: int = self._server.getsockname()[1]
        self._interval = state_interval_s
        self._stop = __import__("threading").Event()
        self._accept = __import__("threading").Thread(target=self._accept_loop, daemon=True)
        self._accept.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass

    def _snapshot(self) -> dict[str, float]:
        a = self.anchor
        dist = (a["x"] ** 2 + a["z"] ** 2) ** 0.5
        jitter = lambda v, s: v + self._rng.uniform(-s, s)  # noqa: E731
        return {
            "light_level": jitter(3.0 if dist > 20 else 12.0, 0.4),
            "y_altitude": jitter(float(a["y"]), 0.3),
            "distance_from_spawn": jitter(dist, 0.4),
            "speed": abs(jitter(0.0, 0.01)),
            "on_ground": 1.0,
            "time_of_day": 0.5,
        }

    def _accept_loop(self) -> None:
        import threading

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
                    send(
                        {
                            "type": "action_result",
                            "id": msg.get("id"),
                            "ok": True,
                            "detail": f"did {msg.get('name')}",
                            "state": self._snapshot(),
                        }
                    )


class ScriptedWorldControl:
    """World control paired with :class:`ScriptedBridgeServer`."""

    def __init__(self, server: ScriptedBridgeServer, *, settle_s: float = 0.08) -> None:
        self._server = server
        self.settle_s = settle_s

    def teleport(self, bot_name: str, pos: dict[str, float]) -> None:  # noqa: ARG002
        self._server.anchor = dict(pos)
        time.sleep(self.settle_s)

    def command(self, cmd: str) -> str:  # noqa: ARG002
        return ""

    def close(self) -> None:
        return


# ── decision-provenance capture (the production telemetry surface) ───────


class RecommendCapture:
    """A registered sim_logger sink collecting ``NAc_RECOMMEND`` events."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, record: dict[str, Any]) -> None:
        if record.get("subsystem") == "NAc_RECOMMEND":
            self.events.append(record)

    def __enter__(self) -> "RecommendCapture":
        from maxim.simulation.sim_logger import register_sim_sink

        register_sim_sink(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        from maxim.simulation.sim_logger import unregister_sim_sink

        unregister_sim_sink(self)


# ── encode + act + record (the production seams, hand-ticked) ────────────


def sensor_ranges(root: Any, names: "tuple[str, ...] | list[str]") -> dict[str, tuple[float, float]]:
    """Declared ranges from the body's own schema (range-aware encoding is
    the invariant; the YAML is the single source)."""
    out: dict[str, tuple[float, float]] = {}
    for name in names:
        sensor = (getattr(root, "sensors", {}) or {}).get(name)
        schema = getattr(sensor, "reading_schema", {}) or {}
        rng = schema.get("range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            out[name] = (float(rng[0]), float(rng[1]))
    return out


@dataclass
class BenchSession:
    """One agent (donor or receiver) with its encode/act/record plumbing."""

    aut: Any
    encoder: Any
    world_ranges: dict[str, tuple[float, float]]
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)
    context_pool: Any = None

    @property
    def agent_id(self) -> str:
        return str(self.aut.agent_id)

    @property
    def root(self) -> Any:
        return self.aut.executor.embodiment.root

    def world_values(self) -> dict[str, float]:
        vm = self.root.vital_metrics
        return {k: float(vm[k]) for k in self.world_ranges if k in vm}

    def encode_clusters(self) -> dict[str, str]:
        """Encode BOTH channels through the production path; returns the
        per-modality active-cluster set ``record_outcome``/``recommend_action``
        consume. The world channel is the A4-gained one (config default)."""
        clusters: dict[str, str] = {}
        world = self.world_values()
        if world:
            node = self.encoder.encode_sensors(
                agent_id=self.agent_id, sensors=world, modality="world", ranges=self.world_ranges
            )
            if node:
                clusters["world"] = node
        d1 = float(self.root.vital_metrics.get("d1", 0.0))
        node = self.encoder.encode_sensors(
            agent_id=self.agent_id, sensors={"d1": d1}, modality="interoception", ranges={"d1": (0.0, 1.0)}
        )
        if node:
            clusters["interoception"] = node
        return clusters

    def sync_world(self) -> int:
        return int(self.aut.backend.sync_world_sensors())

    def execute_and_record(self, tool_name: str, clusters: dict[str, str], *, reasoning: str) -> Any:
        """Execute through the REAL Executor and learn through the REAL
        ``record_outcome`` intake — the loop's own sequence, hand-ticked."""
        from maxim.runtime.tool_dispatch import read_learning_side_effects, record_outcome

        out = self.aut.executor.execute({"tool_name": tool_name, "params": {}})
        side = read_learning_side_effects(out)
        record_outcome(
            agent_id=self.agent_id,
            tool_name=tool_name,
            success=bool(getattr(out, "success", False)),
            result_summary=str(getattr(out, "output", ""))[:80] or None,
            error=getattr(out, "error", None),
            reasoning=reasoning,
            recent_outcomes=self.recent_outcomes,
            max_recent=20,
            llm_worker=None,
            context_pool=self.context_pool,
            nac=self.aut.bio.nac,
            tool_params={},
            cluster_id=clusters.get("interoception"),
            clusters=clusters,
            embodiment_failed=side.embodiment_failed,
            drive_potential_diff=side.drive_potential_diff,
            drive_credit_withheld=side.drive_credit_withheld,
            drive_relief_channel=side.drive_relief_channel,
            outcome_valence=side.outcome_valence,
        )
        return out


def build_bench_session(
    *,
    agent_id: str,
    bridge_port: int,
    home: Path,
    pair_seed: int,
    bridge_host: str = "127.0.0.1",
    body_ref: str = BODY_REF,
) -> BenchSession:
    """The canonical assembly (build_minecraft_aut) on the bench body, with
    the per-pair translating client and the production SensorEncoder.
    ``body_ref`` selects the satiated twin for arm-3 donors (SAME entity
    name — the twin differs only in d1's initial/drift)."""
    from maxim.simulation.minecraft import MinecraftClient
    from maxim.simulation.minecraft_harness import build_minecraft_aut
    from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

    inner = MinecraftClient(bridge_host, bridge_port)
    inner.connect()
    client = TranslatingClient(inner, pair_seed=pair_seed)
    aut = build_minecraft_aut(
        agent_id=agent_id,
        bridge_port=bridge_port,
        persistence_dir=str(home),
        entity_ref=body_ref,
        client=client,
    )
    try:
        from maxim.agents.context_pool import ContextPool

        pool: Any = ContextPool()
    except Exception:  # pragma: no cover - context pool is plumbing, not science

        class _NullPool:
            def add_outcome(self, **_kw: Any) -> None:
                return

        pool = _NullPool()
    encoder = SensorEncoder(ec=aut.bio.ec, config=SensorEncoderConfig())
    ranges = sensor_ranges(aut.executor.embodiment.root, tuple(aut.backend.world_owned_sensors))
    # C2 (design-lens): the frozen selector regime requires the substrate
    # explore bonus OFF — ambient config.json / env can silently arm it
    # through build_bio_stack's resolution (two paths, the n-ctx-drift
    # lesson shape). Refuse, never proceed on a diverged apparatus.
    explore_w = float(getattr(aut.bio.nac.config, "substrate_explore_bonus_weight", 0.0))
    if explore_w != 0.0:
        raise RuntimeError(
            f"exp56: substrate_explore_bonus_weight resolved to {explore_w} (frozen: 0.0) — "
            "ambient sim.substrate_explore_bonus_weight config/env diverges from the frozen "
            "apparatus; unset it (maxim config) and re-run (S6/S3)."
        )
    # The hub session must be OPENED by the object that will close it, or
    # the close persists nothing (the D41/D42 lesson — an unopened hub is
    # exactly how a harness loses the very state it measures).
    aut.bio.memory_hub.on_session_start()
    return BenchSession(aut=aut, encoder=encoder, world_ranges=ranges, context_pool=pool)


# ── the teacher (Exp 52's mechanism on the world channel) ────────────────


def teacher_tick(
    session: BenchSession,
    *,
    situation_active: bool,
    executed_aff: str,
    target_aff: str,
    arm: str,
) -> dict[str, Any]:
    """One teacher evaluation, AFTER the trial's action executed and its
    pending-operant stash landed. Feeds (relieves d1) when the executed
    action is the target AND the situation is active; the credit VALUE is
    the SIGN of the relief the feed actually produced — zero relief mints
    nothing (``reactive_mother_tick``'s exact ``credit="relief"``
    computation, through the same tool_bridge helpers), which is what makes
    the satiated arm a mechanism check rather than a config flag.
    """
    from maxim.embodiment.tool_bridge import _apply_sensor_deltas, _drive_potential_diff

    out: dict[str, Any] = {
        "arm": arm,
        "situation_active": bool(situation_active),
        "executed": executed_aff,
        "target": target_aff,
        "fed": False,
        "relief": None,
        "reward": None,
        "credited": False,
    }
    if not (situation_active and executed_aff == target_aff):
        return out
    root = session.root
    deltas = {"d1": -float(FROZEN["feed_amount"])}
    pre_values = {"d1": float(root.vital_metrics.get("d1", 0.0))}
    # target_effect: a teacher acting ON the agent (the cradle_mother kind).
    _apply_sensor_deltas(root, deltas, delta_kind="target_effect")
    out["fed"] = True
    relief = float(_drive_potential_diff(root, deltas, pre_values) or 0.0)
    out["relief"] = relief
    if abs(relief) <= FROZEN["relief_epsilon"]:
        return out
    reward = 1.0 if relief > 0.0 else -1.0
    out["reward"] = reward
    credited = session.aut.bio.nac.credit_operant_reward(session.agent_id, reward)
    out["credited"] = credited is not None
    if credited is not None:
        out["credited_cluster"], out["credited_tsig"] = credited
    return out


# ── donor schedule + training ────────────────────────────────────────────


def pair_config(pair_seed: int) -> dict[str, Any]:
    """Everything a pair's four arms share, derived from the pair seed:
    the taught target affordance, the contingency slot, and the donor's
    contributor id. Identical across arms by construction (S5)."""
    rng = random.Random(pair_seed * 31337 + 1)
    return {
        "pair_seed": pair_seed,
        "target_aff": rng.choice(AFFORDANCES),
        "slot": dict(rng.choice(FROZEN["contingency_slots"])),
        "contributor_id": f"donor-{pair_seed}",
    }


def balanced_schedule(pair_seed: int, *, reps: "int | None" = None) -> list[tuple[bool, str]]:
    """The seeded balanced donor schedule: every (situation-state, affordance)
    cell exactly ``reps`` times (default: the FROZEN campaign value),
    shuffled per pair. ``reps`` exists for guard tests exercising the
    mechanics at small scale — the campaign never passes it."""
    cells = [(state, aff) for state in (False, True) for aff in AFFORDANCES]
    trials = cells * (FROZEN["schedule_reps_per_cell"] if reps is None else int(reps))
    random.Random(pair_seed * 104729 + 7).shuffle(trials)
    return trials


def run_donor_training(
    session: BenchSession,
    *,
    world: Any,
    pair_seed: int,
    target_aff: str,
    arm: str,
    slot: dict[str, float],
    bot_name: str,
    settle_s: float = 0.6,
    schedule: "list[tuple[bool, str]] | None" = None,
) -> list[dict[str, Any]]:
    """The A-phase: the balanced schedule with the teacher watching. The
    situation-reflected assertion is ALWAYS on — the scripted mock bridge
    reflects placements too (its whole point), so no arm of any mode runs
    against an unmeasured situation (S3)."""
    telemetry: list[dict[str, Any]] = []
    for idx, (situation, aff) in enumerate(balanced_schedule(pair_seed) if schedule is None else schedule):
        anchor = slot if situation else FROZEN["rest_anchor"]
        world.teleport(bot_name, anchor)
        time.sleep(settle_s)
        session.sync_world()
        _assert_situation_reflected(session, situation, slot, where=f"donor trial {idx}")
        clusters = session.encode_clusters()
        tool = f"{ENTITY_NAME}_{aff}"
        out = session.execute_and_record(tool, clusters, reasoning="exp56 balanced schedule")
        tick = teacher_tick(session, situation_active=situation, executed_aff=aff, target_aff=target_aff, arm=arm)
        tick.update(
            {
                "trial": idx,
                "clusters": clusters,
                "tool": tool,
                "mech_success": bool(getattr(out, "success", False)),
                "d1": float(session.root.vital_metrics.get("d1", 0.0)),
                "ts": time.time(),
            }
        )
        telemetry.append(tick)
    return telemetry


def _assert_situation_reflected(session: BenchSession, situation: bool, slot: dict[str, float], *, where: str) -> None:
    """S3: the MEASURED world must reflect the commanded placement — the
    situation is defined by sensors, not by the script's intent (world-seam
    doctrine). Refusal is a raise (apparatus failure), never a warning."""
    values = session.world_values()
    dist = values.get("distance_from_spawn")
    if dist is None:
        raise RuntimeError(f"exp56 {where}: distance_from_spawn missing from measured world state")
    expected_far = (slot["x"] ** 2 + slot["z"] ** 2) ** 0.5
    if situation and dist < expected_far * 0.5:
        raise RuntimeError(
            f"exp56 {where}: situation commanded but measured distance_from_spawn={dist:.1f} "
            f"(expected ~{expected_far:.0f}) — the world does not reflect the script (S3)"
        )
    if not situation and dist > expected_far * 0.5:
        raise RuntimeError(
            f"exp56 {where}: rest commanded but measured distance_from_spawn={dist:.1f} — "
            "the world does not reflect the script (S3)"
        )


# ── donor sanity (prereg §Arms) ──────────────────────────────────────────


def donor_sanity(session: BenchSession, *, arm: str) -> dict[str, Any]:
    """The asserted per-donor checks; the caller re-runs the pair on failure."""
    nac = session.aut.bio.nac
    state = nac.dump()
    ec_payload = json.loads(session.aut.bio.ec.dumps()) if hasattr(session.aut.bio.ec, "dumps") else None
    biases = state.get("cluster_reward_bias", {}) or {}
    world_nodes = {
        nid for nid, (_e, mod) in getattr(session.aut.bio.ec, "_substrate_nodes", {}).items() if mod == "world"
    }
    world_biases = {}
    for key, value in biases.items():
        parts = str(key).split("\x1f")
        if len(parts) == 3 and parts[1] in world_nodes:
            world_biases[key] = float(value)
    # Link balance: per-affordance positive-link confidence spread + counts.
    link_conf: dict[str, float] = {}
    link_count: dict[str, int] = {}
    for tool in ROSTER:
        sig = f"tool:{tool}"
        links = (state.get("links", {}) or {}).get(sig, [])
        pos = [ld for ld in links if ld.get("outcome_valence") == "positive"]
        link_count[tool] = sum(int(ld.get("observation_count", 0)) for ld in pos)
        link_conf[tool] = max((float(ld.get("confidence", 0.0)) for ld in pos), default=0.0)
    spread = max(link_conf.values()) - min(link_conf.values()) if link_conf else 0.0
    counts = list(link_count.values())
    count_spread = (max(counts) - min(counts)) if counts else 0
    checks = {
        "arm": arm,
        "world_bias_max": max(world_biases.values(), default=0.0),
        "world_bias_entries": len(world_biases),
        "total_cluster_bias_entries": len(biases),
        "link_conf_spread": round(spread, 4),
        "link_count_spread": count_spread,
        "inherent_keys": len(state.get("inherent_bias_keys", []) or []),
    }
    if arm == "taught":
        checks["pass"] = (
            checks["world_bias_entries"] >= 1
            and checks["world_bias_max"] >= FROZEN["donor_bias_floor"]
            and spread <= FROZEN["link_spread_max"]
            and count_spread <= FROZEN["link_count_tolerance"]
            and checks["inherent_keys"] == 0
        )
    else:  # satiated
        checks["pass"] = (
            checks["total_cluster_bias_entries"] == 0
            and spread <= FROZEN["link_spread_max"]
            and count_spread <= FROZEN["link_count_tolerance"]
            and checks["inherent_keys"] == 0
        )
    # I8 (design-lens): geometry stamps are part of the per-donor set —
    # asserted HERE so an unstamped donor takes the re-run-and-record path
    # rather than aborting the campaign at ingest.
    geometries = getattr(session.aut.bio.ec, "_substrate_node_geometries", {}) or {}
    unstamped_world = [nid for nid in world_nodes if geometries.get(nid) is None]
    checks["unstamped_world_nodes"] = len(unstamped_world)
    checks["pass"] = bool(checks["pass"]) and not unstamped_world
    _ = ec_payload
    return checks


# ── session persistence + the real CLI export/ingest ─────────────────────


def close_and_stage_session(session: BenchSession, *, stage_dir: Path) -> Path:
    """Full bio close, then stage the pair the export CLI reads
    (``aut_nac.json``/``aut_ec.json``). The close is the FULL one — the
    is_sim_mode trap is exactly a lightweight close losing the thing
    measured. Order: the HUB close persists the NAc/EC pair (+ SCN/ATL);
    the bio-stack close saves the cerebellum + distributor cleanup."""
    session.aut.bio.memory_hub.on_session_end()
    session.aut.bio.on_session_end()
    home = Path(session.aut.persistence_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    nac_src, ec_src = home / "nac.json", home / "ec.json"
    if not nac_src.is_file() or not ec_src.is_file():
        raise RuntimeError(f"exp56: full close did not persist the NAc/EC pair in {home}")
    shutil.copyfile(nac_src, stage_dir / "aut_nac.json")
    shutil.copyfile(ec_src, stage_dir / "aut_ec.json")
    try:
        session.aut.client.close()
    except OSError:
        pass
    return stage_dir


def export_bundle(stage_dir: Path, out_zip: Path, *, contributor_id: str, dangling: bool = False) -> None:
    """The REAL CLI export. ``dangling=True`` re-composes from a copy of the
    stage with ``aut_ec.json`` absent (the export's documented nac-only
    path) — arm 4 ships the SAME donor's nac with no representation."""
    from maxim.hivemind.cli import run_substrate_subcommand

    src = stage_dir
    if dangling:
        src = stage_dir.parent / (stage_dir.name + "_dangling")
        src.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stage_dir / "aut_nac.json", src / "aut_nac.json")
    rc = run_substrate_subcommand(
        [
            "export",
            str(out_zip),
            "--session",
            str(src),
            "--contributor-id",
            contributor_id,
            "--body-ref",
            ENTITY_NAME,
            "--body-yaml",
            str(BODY_SPEC_YAML),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"exp56: substrate export failed (rc={rc}) for {out_zip}")


def ingest_bundle_into(receiver_home: Path, bundle: Path, *, contributor_id: str, receiver_agent_id: str) -> dict:
    """The REAL CLI ingest (strict path: no unstamped-geometry override, no
    force, no inherent trust). Returns the journal's entry for the counts
    (``biases_rekeyed``/``dropped`` — the D43 honesty indicator)."""
    from maxim.hivemind.cli import run_substrate_subcommand

    rc = run_substrate_subcommand(
        [
            "ingest",
            str(bundle),
            "--session",
            str(receiver_home),
            "--trust",
            contributor_id,
            "--receiver-body",
            ENTITY_NAME,
            "--receiver-agent-id",
            receiver_agent_id,
            "--apply",
        ]
    )
    if rc != 0:
        raise RuntimeError(f"exp56: substrate ingest failed (rc={rc}) for {bundle}")
    journal = json.loads((receiver_home / "substrate_ingest_journal.json").read_text())
    return dict(journal["entries"][-1])


# ── the B-probe (frozen selector regime) ─────────────────────────────────


def probe_receiver(
    session: BenchSession,
    *,
    world: Any,
    pair_seed: int,
    slot: dict[str, float],
    bot_name: str,
    settle_s: float = 0.6,
) -> dict[str, Any]:
    """The B-phase probe: rest decisions, then the situation (the
    ARM-INDEPENDENT script trigger), first-contact readout at the frozen
    epsilon-greedy selector, then the fixed tail."""
    nac = session.aut.bio.nac
    # ARM-IDENTICAL dither (design-lens C1): every stochastic ingredient is
    # a pure function of (pair_seed, decision index), so no arm's behavior
    # can shift another draw's position in a shared stream — the prereg's
    # "the same B-probe script per pair seed is used in all four arms" is
    # structural, not hoped-for. contact_at is 0-indexed: randint(3, max-1)
    # puts first contact at the 4th..10th decision ("within the first 10",
    # design-lens I3).
    contact_at = random.Random(pair_seed * 65537 + 3).randint(3, FROZEN["probe_precontact_max"] - 1)

    def _eps_draw(idx: int) -> float:
        return random.Random(pair_seed * 99991 + idx * 17 + 5).random()

    def _choice_draw(idx: int) -> str:
        return random.Random(pair_seed * 77773 + idx * 13 + 9).choice(ROSTER)

    decisions: list[dict[str, Any]] = []
    first_contact: dict[str, Any] | None = None
    total = contact_at + 1 + FROZEN["probe_tail"]
    with RecommendCapture() as cap:
        for idx in range(total):
            situation = idx >= contact_at
            world.teleport(bot_name, slot if situation else FROZEN["rest_anchor"])
            time.sleep(settle_s)
            session.sync_world()
            _assert_situation_reflected(session, situation, slot, where=f"probe decision {idx}")
            clusters = session.encode_clusters()
            drives = {"d1": float(session.root.vital_metrics.get("d1", 0.0))}
            n_before = len(cap.events)
            proposal = nac.recommend_action(
                agent_id=session.agent_id,
                available_tools=list(ROSTER),
                current_drives=drives,
                current_clusters=clusters,
                min_confidence=FROZEN["min_confidence"],
            )
            events = cap.events[n_before:]
            provenance = dict(events[-1].get("data", {})) if events else {}
            drive_component = float((provenance.get("score_components") or {}).get("drive", 0.0) or 0.0)
            if abs(drive_component) > 1e-12:
                raise RuntimeError(
                    f"exp56 probe decision {idx}: score_components['drive']={drive_component} != 0 "
                    "— the L12 prior is inside the selector (S3 refusal)"
                )
            if _eps_draw(idx) < FROZEN["epsilon"]:
                chosen = _choice_draw(idx)
                source = "epsilon"
            elif proposal is not None:
                chosen = str(proposal["tool_name"])
                source = "substrate"
            else:
                chosen = _choice_draw(idx)
                source = "none_fallback"
            record = {
                "decision": idx,
                "situation_active": situation,
                "chosen": chosen,
                "source": source,
                "substrate_tool": None if proposal is None else proposal["tool_name"],
                "substrate_confidence": None if proposal is None else proposal["confidence"],
                "clusters": clusters,
                "provenance": provenance,
                "ts": time.time(),
            }
            decisions.append(record)
            if situation and first_contact is None:
                first_contact = record
            session.execute_and_record(chosen, clusters, reasoning="exp56 probe")
    assert first_contact is not None
    return {
        "pair_seed": pair_seed,
        "contact_at": contact_at,
        "decisions": decisions,
        "first_contact": first_contact,
        "ts": time.time(),
    }


def bias_decisive(first_contact: dict[str, Any], chosen: str) -> bool:
    """The prereg's mechanism assertion: the winning tool's score must be
    LEARNED-BIAS decisive — read from the captured decision-provenance
    components of the substrate proposal that selected it (an epsilon or
    fallback pick is by construction not a substrate success)."""
    if first_contact.get("source") != "substrate" or first_contact.get("substrate_tool") != chosen:
        return False
    provenance = first_contact.get("provenance") or {}
    components = provenance.get("score_components") or {}
    # The scorer's component vocabulary is {causal, reward_bias,
    # learned_bias, drive, explore}; learned_bias is the cluster-keyed,
    # situation-conditioned channel the claim names. Decisiveness reads
    # ``learned_margin`` — the provenance field built for exactly this
    # (winner's learned_bias minus the runner-up's): the balanced-schedule
    # design makes the causal channel large but UNIFORM across the roster,
    # so a positive learned margin is what separates the winner, and a
    # sum-vs-sum comparison would misread uniform non-discriminative
    # components as rivals.
    learned = float(components.get("learned_bias", 0.0) or 0.0)
    margin = provenance.get("learned_margin")
    return learned > 0.0 and margin is not None and float(margin) > 0.0


# ── the anti-vacuity kit (D62/D44's shapes, at the analyzer's disposal) ──


def noop_variant_readout(
    *,
    bundle: Path,
    receiver_pre_nac: dict[str, Any],
    receiver_pre_ec: dict[str, dict[str, Any]],
    receiver_agent_id: str,
    contributor_id: str,
    first_contact: dict[str, Any],
    target_tool: str,
) -> dict[str, Any]:
    """Re-run one recorded arm-2 pair's merge under no-op variants and read
    the gate out again (the D62 kit: a gate that cannot fail is not a gate).

    Per-variant EXPECTATIONS (the D44 kit's own lesson — `return right`
    historically PASSED probes because a fresh receiver makes donor-alone
    equivalent to the real merge; asserting collapse there would test a
    recipe, not the gate):

    * ``receiver_unchanged`` — MUST collapse (no donor state arrives).
    * ``empty_state`` — MUST collapse (nothing arrives).
    * ``donor_alone`` — readout RECORDED; on a fresh receiver it is
      expected to persist (the equivalence is documented, not asserted).

    Returns per-variant readouts; ``kit_pass`` is the two must-collapse
    variants both collapsing.
    """
    from unittest import mock as _mock

    import maxim.hivemind.ingest as ingest_mod
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.hivemind.merge import SubstrateMergeResult, ec_merge_aligned, rekey_nac_state

    def _readout(nac_state: dict[str, Any]) -> dict[str, Any]:
        nac = NAc(config=NACConfig())
        nac.load_state(dict(nac_state))
        proposal = nac.recommend_action(
            agent_id=receiver_agent_id,
            available_tools=list(ROSTER),
            current_drives={"d1": 0.0},
            current_clusters=dict(first_contact.get("clusters") or {}),
            min_confidence=FROZEN["min_confidence"],
        )
        return {
            "tool": None if proposal is None else proposal["tool_name"],
            "chose_target": proposal is not None and proposal["tool_name"] == target_tool,
        }

    def _variant(kind: str):
        def fake(**kwargs: Any) -> SubstrateMergeResult:
            if kind == "receiver_unchanged":
                return SubstrateMergeResult(
                    nac=dict(kwargs["receiver_nac"]),
                    ec_nodes=dict(kwargs["receiver_ec"]),
                    id_map={},
                    biases_rekeyed=0,
                    biases_dropped=0,
                )
            if kind == "empty_state":
                return SubstrateMergeResult(nac={}, ec_nodes={}, id_map={}, biases_rekeyed=0, biases_dropped=0)
            aligned = ec_merge_aligned(
                kwargs["receiver_ec"],
                kwargs["donor_ec"],
                left_source=kwargs["receiver_source"],
                right_source=kwargs["donor_source"],
                strict_geometry=True,
            )
            donor_only = rekey_nac_state(
                kwargs["donor_nac"], aligned.id_map, to_agent_id=kwargs.get("receiver_agent_id")
            )
            return SubstrateMergeResult(
                nac=donor_only,
                ec_nodes=aligned.nodes,
                id_map=aligned.id_map,
                biases_rekeyed=0,
                biases_dropped=0,
            )

        return fake

    out: dict[str, Any] = {}
    for kind in ("receiver_unchanged", "empty_state", "donor_alone"):
        with _mock.patch.object(ingest_mod, "substrate_merge", side_effect=_variant(kind)):
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                journal = ingest_mod.IngestionJournal(Path(td) / "j.json")
                report = ingest_mod.ingest_bundle(
                    bundle,
                    receiver_nac=dict(receiver_pre_nac),
                    receiver_ec_nodes=dict(receiver_pre_ec),
                    receiver_body=ENTITY_NAME,
                    trusted_sources=frozenset({contributor_id}),
                    journal=journal,
                    receiver_agent_id=receiver_agent_id,
                )
        out[kind] = _readout(report.nac)
    out["kit_pass"] = (not out["receiver_unchanged"]["chose_target"]) and (not out["empty_state"]["chose_target"])
    return out
