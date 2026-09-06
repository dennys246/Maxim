"""Two-AUT-one-world harness glue — the 1.1.4 PR 4 assembly (world seam).

The importable half of the harness (`scripts/minecraft_two_aut.py` is the
thin CLI): per-AUT construction on the canonical builders, the per-tick
world-sensor pump with the STALENESS GATE, the deterministic
:class:`FakeBridgeServer` (the harness's own instrument — a scripted world
speaking the frozen protocol so the smoke gate is runnable without a
Minecraft server), and the NON-VACUOUS smoke verdict.

Design notes, load-bearing:

* **One AUT = one bridge = one client = one agent home.** Two full
  ``run_agentic_loop`` threads share no BIO state — separate
  ``persistence_dir`` (never a shared ``~/.maxim``), separate percept
  source/sink, separate bio-stack. (`AgentPool.run_turn` is explicitly not
  the full loop; this is.) Known shared residue, diagnostics only: the
  loop's CWD-relative ``data/agents/<name>/runtime/state_*.json`` — both
  AUTs are MaximAgents with second-resolution run ids, so those overlap;
  run the harness from a scratch CWD (the CLI and the test both do).
* **Substrate-primary, no LLM in the action path**: the loop builds its own
  ``SensorEncoder`` from ``memory_hub.ec`` and proposes via
  ``propose_via_substrate`` — the world channel's A4-gained encode IS the
  action-selection input, which is what makes the smoke non-vacuous.
* **``consolidation="full"`` always** (the ship gate's second half): a
  benchmark session that closes lightweight silently loses the very
  consolidation being measured (`runtime/sim_adapter.py`'s own warning).
  Pinned by `tests/unit/test_minecraft_harness.py` against `_loop_kwargs`.
* **Live-run coherence (PR 3 obligation):** `items/minecraft_bread.yaml` is
  a SEM-only finite-resource item — on live-bridge runs it is a phantom
  (its portions are not game inventory) and MUST NOT be spawned; the fake
  bridge and SEM-only arms may use it freely.
* **Staleness gate (PR 3 obligation):** the pump consults
  ``MinecraftClient.state_age_s()`` and refuses to write snapshots older
  than ``max_state_age_s`` (warn-once) — a dead bridge must not keep
  feeding the substrate its last snapshot as if fresh.

The smoke verdict (the roadmap's 1.1.4 gate, executable): world-modality EC
nodes > 0 on the LIVE store AND persisted by the FULL close into the agent
home's ``ec.json`` (D64's lesson: "clean" must not mean "never reached" —
the assertion is on the artifact the session leaves behind).
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MINECRAFT_BODY_REF = "bodies/minecraft_player"
_DEFAULT_MAX_STATE_AGE_S = 5.0


# ── fake bridge (the harness's own instrument) ───────────────────────────


class FakeBridgeServer:
    """A seeded scripted world speaking the frozen NDJSON protocol.

    Dev/test support: lets the WHOLE two-AUT loop run without Minecraft.
    SEEDED, not deterministic across runs (executor-lens correction): one
    shared rng drawn from per-connection threads makes each client's stream
    interleaving-dependent, and two clients get two INDEPENDENT worlds from
    one entropy stream — "one port = two emulated bridges". Fine for a
    wiring smoke; 1.2's sharing arms need a fake with genuinely SHARED
    per-tick state (recorded in the plan). Every action is confirmed with a
    post-action snapshot (the action_result-carries-state contract). Note
    the real JS bridge refuses a second client; this fake accepts them by
    design.
    """

    def __init__(self, *, seed: int = 42, state_interval_s: float = 0.1) -> None:
        import random

        self._rng = random.Random(seed)
        self._interval = state_interval_s
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(4)
        self.port: int = self._server.getsockname()[1]
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._accept = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass

    def _snapshot(self) -> dict[str, float]:
        # Covers EVERY declared modality:world sensor (lockstep test-pinned
        # against the body YAML — the fake must not drift under it).
        r = self._rng
        return {
            "health": max(0.0, min(20.0, 14.0 + r.uniform(-4, 6))),
            "food": max(0.0, min(20.0, 12.0 + r.uniform(-6, 6))),
            "saturation": r.uniform(0, 5),
            "oxygen": max(0.0, min(20.0, 18.0 + r.uniform(-4, 2))),
            "light_level": float(r.randint(0, 15)),
            "y_altitude": 64.0 + r.uniform(-8, 8),
            "nearest_hostile_dist": max(0.0, min(64.0, r.uniform(2, 64))),
            "hostile_count": float(r.randint(0, 6)),
            "nearest_player_dist": max(0.0, min(64.0, r.uniform(1, 64))),
            "distance_from_spawn": r.uniform(0, 64),
            "speed": r.uniform(0, 0.4),
            "on_ground": float(r.random() > 0.2),
            "is_raining": float(r.random() > 0.9),
            "xp_level": float(r.randint(0, 10)),
            "look_pitch": r.uniform(-1.2, 1.2),
            "time_of_day": r.random(),
        }

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sock, _addr = self._server.accept()
            except OSError:
                return
            t = threading.Thread(target=self._serve, args=(sock,), daemon=True)
            t.start()
            self._threads.append(t)

    def _serve(self, sock: socket.socket) -> None:
        def send(obj: dict[str, Any]) -> None:
            try:
                sock.sendall(json.dumps(obj).encode() + b"\n")
            except OSError:
                pass

        sock.settimeout(self._interval)
        buffer = b""
        next_event = 0
        while not self._stop.is_set():
            send({"type": "state", "data": self._snapshot()})
            next_event += 1
            if next_event % 5 == 0:
                send({"type": "event", "kind": "info", "text": f"the wind shifts (tick {next_event})"})
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


# ── per-AUT assembly ─────────────────────────────────────────────────────


class MinecraftAut:
    """Everything one AUT owns — built by :func:`build_minecraft_aut`."""

    def __init__(self, **parts: Any) -> None:
        self.__dict__.update(parts)


def build_minecraft_aut(
    *,
    agent_id: str,
    bridge_port: int,
    persistence_dir: str,
    bridge_host: str = "127.0.0.1",
    action_timeout_s: float = 15.0,
    entity_ref: str = MINECRAFT_BODY_REF,
    client: Any | None = None,
) -> MinecraftAut:
    """Build one full AUT against one bridge, on the canonical builders.

    Raises on any missing piece (a harness that degrades into an
    unembodied or world-less agent measures nothing — D64's shape).

    ``entity_ref`` selects the body (default: the production player body;
    Exp 56 passes ``bodies/minecraft_bench``) and ``client`` accepts a
    pre-built/wrapping client (Exp 56's per-pair translating proxy) — both
    defaulted so every existing caller is byte-identical. The assembly
    stays HERE either way: a second hand-composed builder in scripts/
    would be the D43 composition lesson re-armed.
    """
    from maxim.embodiment.backends.minecraft import minecraft_modulator_factory
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.runtime.bio_stack import build_bio_stack
    from maxim.runtime.bootstrap import build_executor
    from maxim.simulation.minecraft import MinecraftClient, MinecraftPerceptSource
    from maxim.tools.registry import ToolRegistry

    if client is None:
        client = MinecraftClient(bridge_host, bridge_port, action_timeout_s=action_timeout_s)
        client.connect()

    component_registry = ComponentRegistry()
    bio = build_bio_stack(agent_id=agent_id, persistence_dir=persistence_dir)
    registry = ToolRegistry()
    executor = build_executor(
        tool_registry=registry,
        # Explicit opt-out (the required-keyword contract): the AUT runs
        # unrestricted inside its OWN isolated sandbox home.
        permissions=None,
        # agent_id MUST reach the Embodiment (architecture-lens review):
        # an agent_id-unaware body publishes PainSignals the reward
        # distributor SILENTLY SKIPS — embodied credit as a wired-in no-op,
        # in the very two-AUT instrument 1.2's attribution rides on.
        agent_id=agent_id,
        pain_bus=bio.pain_bus,
        nac=bio.nac,
        # Mirror the canonical factory's FULL kwarg set (executor-lens
        # review: an under-wired harness agent would make the live-bridge
        # L11 re-measure a verify-the-instrument failure — no cerebellum
        # training, no SCN temporal credit).
        hippocampus=bio.hippocampus,
        scn=bio.scn,
        cerebellum=bio.cerebellum,
        distributor=bio.distributor,
        entity_ref=entity_ref,
        component_registry=component_registry,
        # The factory derives world_owned_sensors from the ATTACHED entity's
        # own modality: world declarations — no probe parse, no drift.
        modulator_factory=minecraft_modulator_factory(client),
    )
    backend = None
    embodiment = getattr(executor, "embodiment", None)
    root = getattr(embodiment, "root", None)
    for mod in (getattr(root, "modulators", None) or {}).values():
        candidate = getattr(mod, "_backend", None)
        if candidate is not None:
            backend = candidate
            break
    if backend is None:
        raise RuntimeError("minecraft modulator backend did not attach — the world seam is not wired")
    if not backend.world_owned_sensors:
        raise RuntimeError(f"{entity_ref} declares no modality: world sensors — nothing to measure")

    return MinecraftAut(
        agent_id=agent_id,
        client=client,
        executor=executor,
        backend=backend,
        bio=bio,
        percept_source=MinecraftPerceptSource(client),
        persistence_dir=persistence_dir,
    )


class MinecraftSyncPump:
    """Per-tick world-sensor pump with the STALENESS GATE (PR 3 obligation)."""

    def __init__(
        self, aut: MinecraftAut, *, interval_s: float = 0.5, max_state_age_s: float = _DEFAULT_MAX_STATE_AGE_S
    ) -> None:
        self._aut = aut
        self._interval = interval_s
        self._max_age = max_state_age_s
        self._stop = threading.Event()
        self._warned_stale = False
        self._thread: threading.Thread | None = None
        self.writes = 0
        self.stale_skips = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"mc-sync-{self._aut.agent_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                age = self._aut.client.state_age_s()
                if age > self._max_age:
                    self.stale_skips += 1
                    if not self._warned_stale:
                        self._warned_stale = True
                        logger.warning(
                            "minecraft sync pump (%s): snapshot is %.1fs old (> %.1fs) — refusing to feed "
                            "stale world state to the substrate; will resume when fresh data arrives",
                            self._aut.agent_id,
                            age,
                            self._max_age,
                        )
                    continue
                if self._warned_stale:
                    self._warned_stale = False
                    logger.info("minecraft sync pump (%s): fresh snapshots resumed", self._aut.agent_id)
                self.writes += self._aut.backend.sync_world_sensors()
            except Exception:
                # A pump that dies silently is indistinguishable from a
                # healthy one (executor-lens review) — log LOUDLY and keep
                # ticking; the verdict's writes-gate catches a pump that
                # never writes.
                logger.warning("minecraft sync pump (%s): tick raised", self._aut.agent_id, exc_info=True)


def _loop_kwargs(aut: MinecraftAut, *, max_steps: int, stop_event: threading.Event, target_hz: float) -> dict[str, Any]:
    """The `run_agentic_loop` keyword set — a PURE function so the ship
    gate's consolidation/aut-mode pins test the exact kwargs the harness
    passes (not a hand-composed sequence; the D43 §5 lesson)."""
    return {
        "aut_mode": "substrate-primary",
        "percept_source": aut.percept_source,
        "pain_bus": aut.bio.pain_bus,
        "memory_hub": aut.bio.memory_hub,
        "hippocampus": aut.bio.hippocampus,
        "max_steps": max_steps,
        "stop_event": stop_event,
        "target_hz": target_hz,
        # THE GATE'S SECOND HALF: a benchmark session must take the FULL
        # close, or the consolidation being measured never persists.
        "consolidation": "full",
    }


def run_minecraft_aut(
    aut: MinecraftAut,
    *,
    max_steps: int = 60,
    target_hz: float = 4.0,
    stop_event: "threading.Event | None" = None,
) -> None:
    """Run one AUT's full agent loop (blocking; run each AUT on a thread)."""
    from maxim.agents.maxim_agent import MaximAgent
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime.agent_loop import run_agentic_loop
    from maxim.runtime.bootstrap import build_decision_engine, build_memory
    from maxim.runtime.state import RuntimeState

    workspace = Path(aut.persistence_dir) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    agent = MaximAgent()
    if aut.bio.memory_hub is not None:
        agent.wire_memory_hub(aut.bio.memory_hub)
    state = RuntimeState()
    state.data["mode"] = "active"
    state.data["active_goal"] = "survive in the world"
    try:
        run_agentic_loop(
            agent,
            FileSystemEnv(str(workspace)),
            state,
            build_memory(),
            build_decision_engine(),
            aut.executor,
            **_loop_kwargs(aut, max_steps=max_steps, stop_event=stop_event or threading.Event(), target_hz=target_hz),
        )
    finally:
        # The bio-stack's OWN session end (cerebellum save + distributor
        # cleanup) — the loop's close covers hub + hippocampus only, and
        # the brief's invariant says save_cerebellum() must run at session
        # end (executor-lens review).
        try:
            aut.bio.on_session_end()
        except Exception:
            logger.warning("bio-stack session end raised for %s", aut.agent_id, exc_info=True)


def smoke_verdict(aut: MinecraftAut, pump: "MinecraftSyncPump | None" = None) -> dict[str, Any]:
    """The NON-VACUOUS gate readout for one AUT.

    Asserts nothing itself — returns the facts; the caller (test / script)
    gates on ALL of: ``world_feed_writes > 0`` (the world feed was ALIVE —
    the executor-lens review DEMONSTRATED the node counts alone go green
    with the pumps never started, because the body's non-neutral initials
    mint one static encode), ``world_nodes_live >= 2`` (change-driven
    encodes beyond that static first one), and ``world_nodes_persisted >=
    2`` (the close saved them). The close FLAVOR is not decidable from
    files (the lightweight closer also writes ec.json) — the test's
    runtime close-flavor discriminator carries that half.
    """
    ec = aut.bio.ec
    live = sum(1 for _nid, (_e, mod) in getattr(ec, "_substrate_nodes", {}).items() if mod == "world")
    persisted = 0
    ec_path = Path(aut.persistence_dir) / "ec.json"
    if ec_path.exists():
        try:
            data = json.loads(ec_path.read_text())
            persisted = sum(
                1 for _nid, nd in (data.get("substrate_nodes") or {}).items() if nd.get("modality") == "world"
            )
        except (OSError, json.JSONDecodeError):
            persisted = -1  # a corrupt artifact is a FINDING, not a zero
    return {
        "agent_id": aut.agent_id,
        "world_nodes_live": live,
        "world_nodes_persisted": persisted,
        "ec_json_exists": ec_path.exists(),
        "state_age_s": aut.client.state_age_s(),
        "world_feed_writes": pump.writes if pump is not None else None,
    }


def verdict_is_green(verdict: dict[str, Any]) -> bool:
    """THE gate, one place (test and CLI both call this — no hand-composed
    variant can drift): feed alive AND change-driven encodes AND persisted."""
    writes = verdict.get("world_feed_writes")
    return (
        writes is not None
        and writes > 0
        and verdict.get("world_nodes_live", 0) >= 2
        and verdict.get("world_nodes_persisted", 0) >= 2
    )
