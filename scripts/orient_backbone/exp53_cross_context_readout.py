#!/usr/bin/env python
"""Exp 53 — cross-context readout: the nursery-taught want, read out on the robot.

Pre-registration (frozen; read it first):
``docs/experiments/protocols/exp53_cross_context_readout_preregistration.md``.

Loads the Exp 52 Phase B infants' persisted substrate — ``aut_nac.json`` +
``aut_ec.json`` from the SAME sim_reports session dir — UNCHANGED into the
production substrate-primary decision path with the live Reachy Mini as the
body, and asks whether the taught infants turn toward a speech source while the
zero-bias controls (satiated / no_feed), loaded identically, do not.

Production pieces, called as the loop calls them (no re-implementation):

* ``bodies/infant_operant`` instantiated by ``ComponentRegistry`` — its tool
  names (``infant_operant_turn_left`` …) are the learned bias keys; no innate
  orient drive (as in the nursery).
* ``ReachyOrientMotorBackend`` attached through ``attach_backends`` — with an
  EXPLICIT ``deltas`` map (δ, declared S6) because the infant body declares no
  ``head_yaw`` self-effect for the factory to read.
* ``DoAFeed`` world-setting the same ``azimuth`` root sensor the mother's call
  world-set in the nursery (owner ``doa_feed``).
* ``generate_tools_for_entity`` → ``ToolRegistry``; ``_encode_current_clusters``
  / ``propose_via_substrate`` (``runtime/agent_loop.py``) →
  ``NAc.recommend_action(current_clusters=)``; the ``NAc_RECOMMEND`` decision
  provenance (``learned_margin`` / ``explore_decisive`` / ``score_components``)
  captured through ``sim_logger.register_sim_sink``.

READOUT ONLY: nothing calls ``record_outcome`` / ``credit_operant_reward``;
turns are dispatched through the modulator (not the Executor's dispatch path
that credits); the persisted files' SHA-256 is recorded before and after every
agent and must match.

Subcommands::

    manifest --archive <phaseB dir> --out 53_agents_manifest.json
    run --manifest ... --phase 1|2 --host <ip> --out 53_cross_context_readout.jsonl
    run --manifest ... --phase 1|2 --dry-run --out /tmp/53_dry.jsonl
    verdict --records 53_cross_context_readout.jsonl

Phase 2 refuses to start unless the records file already holds a Phase 1
``gate_I`` record with verdict PASS (stop rule I).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from delivered_shift_block import (  # noqa: E402
    HEALTHY_GAIN,
    _MaximShim,
    daemon_body_yaw,
    daemon_status,
    installed_sdk_version,
    provenance,
    speech_rate_probe,
)
from live_common import JsonlLog, gated_azimuth, resolve_host  # noqa: E402

# ── frozen pre-registration constants (do not edit after the first Phase 1 record) ──
BODY_REF = "bodies/infant_operant"
AGENT_ID = "sim_aut"  # the nursery's AUT id — the learned bias keys start with it
DELTA_RAD = 0.55  # body-yaw step per turn ≈ the nursery's 0.30-az self_effect (S6)
DELTAS = {"turn_left": +DELTA_RAD, "turn_right": -DELTA_RAD}
TARGETS = (-0.3, -0.2, 0.5, 0.6)  # gated az targets — where the sweep shows the learning lives (amendment 1)
EXPLORATORY_TARGETS = (-0.6, 0.2)  # recorded, excluded from every gate (amendment 1)
TRIALS_PER_AGENT = 12  # 4 gated targets × 3 (+ exploratory placements × 3, not gated)
PROBE_PLACEMENTS = 1  # Phase 1: each target once
MARGIN_FLOOR = 0.11  # L1 visibility floor
TOWARD_EPS = 0.05  # |az_after| < |az_before| - eps
EXPLORE_PRIMARY = 0.0  # Phase 2 primary: frozen policy + motion (amendment 1)
EXPLORE_SECONDARY = 1.5  # Phase 2 secondary: Phase B's value, reported not gated
GATE_I_RATE = 0.80
GATE_I_SEEDS = 2
GATE_T_LEARNED = 0.70
GATE_T_MARGIN = 0.20
GATE_T_SIGN_AGREEMENT = 0.80
ARMS = {"taught": (42, 43, 44), "satiated": (42, 43, 44), "no_feed": (42, 43, 44)}
EXPLORATORY = (("taught", 48),)
MIN_SPEECH_RATE = 0.70
GATED_READS = 5


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── manifest ─────────────────────────────────────────────────────────────────


def _find_pair(run_dir: Path) -> tuple[Path, Path] | None:
    for nac in sorted(run_dir.glob("sim_reports/*/aut_nac.json")):
        ec = nac.with_name("aut_ec.json")
        if ec.exists():
            return nac, ec
    return None


def _describe_state(nac_path: Path, ec_path: Path) -> dict:
    nac = json.loads(nac_path.read_text())
    ec = json.loads(ec_path.read_text())
    biases = nac.get("cluster_reward_bias") or {}
    nodes = ec.get("substrate_nodes") or {}
    audio = [nid for nid, n in nodes.items() if isinstance(n, dict) and n.get("modality") == "audio"]
    bias_rows = []
    for key, val in biases.items():
        parts = key.split("\x1f")
        if len(parts) == 3:
            bias_rows.append({"agent": parts[0], "cluster": parts[1], "tool": parts[2], "bias": round(float(val), 4)})
    return {
        "bias_entries": len(biases),
        "biases": bias_rows,
        "audio_nodes": audio,
        "n_audio_nodes": len(audio),
        "ec_hash_scheme": ec.get("hash_scheme"),
        "ec_encoder_provenance": ec.get("encoder_provenance"),
        "nac_format_version": nac.get("_format_version"),
        "nac_saved_at": nac.get("saved_at"),
    }


def cmd_manifest(args: argparse.Namespace) -> int:
    archive = Path(args.archive).expanduser()
    agents = []
    wanted = [(arm, seed, False) for arm, seeds in ARMS.items() for seed in seeds]
    wanted += [(arm, seed, True) for arm, seed in EXPLORATORY]
    for arm, seed, exploratory in wanted:
        run_dir = archive / f"{arm}_seed{seed}_ew1.5"
        pair = _find_pair(run_dir)
        if pair is None:
            print(f"[FAIL] no aut_nac.json + aut_ec.json pair under {run_dir}")
            return 2
        nac_path, ec_path = pair
        desc = _describe_state(nac_path, ec_path)
        agents.append(
            {
                "arm": arm,
                "seed": seed,
                "exploratory": exploratory,
                "label": f"{arm}_seed{seed}",
                "nac_path": str(nac_path),
                "ec_path": str(ec_path),
                "nac_sha256": _sha256(nac_path),
                "ec_sha256": _sha256(ec_path),
                **desc,
            }
        )
        print(
            f"  {arm:9s} seed{seed}{' (exploratory)' if exploratory else ''}: "
            f"{desc['bias_entries']} bias entries, {desc['n_audio_nodes']} audio nodes, session {nac_path.parent.name}"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "_format_version": "1.0",
                "experiment": "53_cross_context_readout",
                "archive": str(archive),
                "frozen": {
                    "body_ref": BODY_REF,
                    "agent_id": AGENT_ID,
                    "deltas_rad": DELTAS,
                    "targets_az": TARGETS,
                    "trials_per_agent": TRIALS_PER_AGENT,
                    "explore_primary": EXPLORE_PRIMARY,
                    "explore_secondary": EXPLORE_SECONDARY,
                },
                "agents": agents,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[manifest] {len(agents)} agents -> {out}")
    return 0


# ── loaded substrate ──────────────────────────────────────────────────────────


class LoadedAgent:
    """The nursery's persisted NAc + EC, loaded unchanged, plus the production
    SensorEncoder over them. One instance per (agent, explore weight)."""

    def __init__(self, spec: dict, explore_weight: float) -> None:
        from dataclasses import replace

        from maxim.decisions.nac import NAc, NACConfig
        from maxim.similarity.ec import EntorhinalCortex
        from maxim.similarity.encoder import SensorEncoder

        self.spec = spec
        self.explore_weight = explore_weight
        self.ec = EntorhinalCortex()
        self.ec.load(spec["ec_path"])
        cfg = replace(NACConfig(), substrate_explore_bonus_weight=float(explore_weight), persistence_path=None)
        self.nac = NAc(config=cfg, ec=self.ec)
        # apply_decay=False: wall-clock since the nursery saved is the operator's
        # schedule, not agent-experienced time (the --resume-sim rule).
        self.nac.load(spec["nac_path"], apply_decay=False)
        self.encoder = SensorEncoder(ec=self.ec, atl=None, nac=self.nac)
        self.audio_nodes = {
            nid for nid, n in self.ec._substrate_nodes.items() if getattr(n, "modality", None) == "audio"
        } or set(spec.get("audio_nodes") or [])
        loaded_bias = self._bias_entries()
        if loaded_bias != int(spec["bias_entries"]):
            raise RuntimeError(
                f"{spec['label']}: loaded {loaded_bias} cluster biases, manifest says {spec['bias_entries']} — "
                "the load did not restore the persisted state"
            )

    def _bias_entries(self) -> int:
        table = getattr(self.nac, "_cluster_reward_bias", None)
        return len(table) if table is not None else -1

    def files_unchanged(self) -> bool:
        return (
            _sha256(Path(self.spec["nac_path"])) == self.spec["nac_sha256"]
            and _sha256(Path(self.spec["ec_path"])) == self.spec["ec_sha256"]
        )


class _ProvenanceSink:
    """Captures the NAc_RECOMMEND decision-provenance record for the current call."""

    def __init__(self) -> None:
        self.last: dict | None = None

    def __call__(self, record: dict) -> None:
        if record.get("kind", "").upper() == "NAC_RECOMMEND" or "learned_margin" in (record.get("data") or {}):
            self.last = record

    def take(self) -> dict | None:
        rec, self.last = self.last, None
        data = rec.get("data") if isinstance(rec, dict) else None
        return dict(data) if isinstance(data, dict) else rec


# ── rigs ─────────────────────────────────────────────────────────────────────


class _Executor:
    """The two attributes ``propose_via_substrate`` / ``_encode_current_clusters`` read."""

    def __init__(self, registry, embodiment) -> None:
        self.registry = registry
        self.embodiment = embodiment


def _build_body(modulator_factory):
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.embodiment.spec import attach_backends
    from maxim.embodiment.tool_bridge import generate_tools_for_entity
    from maxim.tools.registry import ToolRegistry

    entity = ComponentRegistry().instantiate(BODY_REF)
    if modulator_factory is not None:
        attach_backends(entity, modulator_factory=modulator_factory)
    embodiment = Embodiment(entity)
    registry = ToolRegistry()
    generate_tools_for_entity(entity, registry, embodiment=embodiment)
    names = list(registry.list())
    for want in ("infant_operant_turn_left", "infant_operant_turn_right"):
        if want not in names:
            raise RuntimeError(f"tool {want!r} not registered — got {names}")
    return entity, embodiment, registry


class LiveReadoutRig:
    def __init__(self, host: str) -> None:
        from maxim.embodiment.audio_localization import DoAFeed
        from maxim.hardware.reachy.controller import ReachyMiniController
        from maxim.hardware.reachy.motor_backend import ReachyOrientMotorBackend

        self.host = host
        self.dry = False
        self.robot = ReachyMiniController(host=host, connection_mode="network", media_backend="no_media")
        if not self.robot.connect():
            raise RuntimeError("controller.connect() failed")
        if not self.robot.wake_up():
            raise RuntimeError("controller.wake_up() failed — torque not enabled, refusing to run")
        self.reader = self.robot.get_doa_reader()
        if self.reader is None:
            raise RuntimeError("controller has no DoA reader")
        self.stop = threading.Event()
        self.shim = _MaximShim(None)
        robot, shim = self.robot, self.shim

        def factory(entity, mod_name, spec_modulator):
            if mod_name != "orient":
                return None
            return ReachyOrientMotorBackend(
                robot=robot,
                maxim=shim,
                entity=entity,
                deltas=dict(DELTAS),
                modulator_name=mod_name,
                entity_name=getattr(entity, "name", "") or "",
            )

        self.entity, self.embodiment, self.registry = _build_body(factory)
        owned = getattr(self.embodiment, "live_world_set_sensors", None)
        if owned is not None:
            owned.update(ReachyOrientMotorBackend.world_owned_sensors)
        self.feed = DoAFeed(
            self.reader,
            self.embodiment,
            stop_event=self.stop,
            head_yaw_provider=self._head_rel_yaw_deg,
            body_yaw_provider=self._body_yaw_deg,
        )
        self.shim._doa_feed = self.feed
        self.orient = self.entity.modulators["orient"]
        backend = getattr(self.orient, "_backend", None)
        if backend is None:
            raise RuntimeError("orient modulator has NO backend attached — the production path is not bound")
        backend.bind_embodiment(self.embodiment)
        self.deltas = dict(backend._deltas)
        self.executor = _Executor(self.registry, self.embodiment)
        self._thread = threading.Thread(target=self.feed.run, name="doa-feed", daemon=True)
        self._thread.start()

    def _pose(self) -> dict:
        return self.robot.get_current_pose() or {}

    def _head_rel_yaw_deg(self) -> float:
        p = self._pose()
        return math.degrees(float(p.get("yaw", 0.0)) - float(p.get("body_yaw", 0.0)))

    def _body_yaw_deg(self) -> float:
        return math.degrees(float(self._pose().get("body_yaw", 0.0)))

    def head_pose_deg(self) -> dict:
        p = self._pose()
        out = {}
        for k in ("yaw", "pitch", "roll", "body_yaw"):
            if k in p:
                out[f"head_{k}_deg" if k != "body_yaw" else "body_yaw_deg"] = round(math.degrees(float(p[k])), 2)
        return out

    def body_yaw(self) -> float | None:
        return daemon_body_yaw(self.host)

    def goto_body_yaw(self, yaw: float, duration: float = 2.5) -> None:
        from maxim.hardware import MotionTarget

        for attempt in range(2):
            if self.robot.goto_target(MotionTarget(body_yaw=float(yaw), duration=duration)):
                break
            if attempt:
                raise RuntimeError("goto_target rejected twice — daemon/motors need attention")
            print("      (goto rejected/timed out — re-issuing once)")
            time.sleep(1.0)
        time.sleep(0.5)

    def recenter(self) -> None:
        self.goto_body_yaw(0.0)

    def gated_az(self) -> float | None:
        return gated_azimuth(self.reader, k=GATED_READS, timeout_s=8.0, poll_s=0.15)

    def sync_embodiment(self, az: float | None) -> None:
        # Live: DoAFeed owns the sensor; nothing to do.
        return None

    def execute(self, affordance: str):
        return self.orient.execute(affordance, {})

    def close(self) -> None:
        self.stop.set()
        try:
            self.recenter()
        except Exception as e:  # noqa: BLE001 - best-effort on the way out
            print(f"      (final recenter failed: {e})")
        try:
            self.robot.disconnect()
        except Exception:  # noqa: BLE001
            pass


class DryReadoutRig:
    """Offline stand-in: infant body + a dry orient backend over a modeled source.
    Same production body/tools/encode path; only the motor and the sensor are fake."""

    def __init__(self, ratio: float = 0.95, seed: int = 1) -> None:
        from maxim.embodiment.audio_localization import world_set_axis

        self.dry = True
        self.host = "dry"
        self._rng = random.Random(seed)
        self.ratio = ratio
        self.body = 0.0
        self.source = 0.0
        self._world_set_axis = world_set_axis
        rig = self

        class _DryBackend:
            def __init__(self, entity, mod_name):
                self._deltas = dict(DELTAS)
                self._entity = entity

            def bind_embodiment(self, embodiment):
                self._embodiment = embodiment

            def execute(self, affordance, params):
                from maxim.embodiment.sem import ModulatorResult

                before = rig.az()
                target = rig.body + self._deltas[affordance]
                rig.body += self._deltas[affordance] * rig.ratio
                after = rig.az()
                rig.sync_embodiment(after)
                return ModulatorResult(
                    success=True,
                    modulator_name="orient",
                    entity_name="infant_operant",
                    affordance=affordance,
                    params=params,
                    metadata={
                        "commanded_body_yaw_deg": round(math.degrees(target), 1),
                        "achieved_body_yaw_deg": round(math.degrees(rig.body), 1),
                        "measured_drive_transitions": {"azimuth": (before, after)},
                    },
                )

        def factory(entity, mod_name, spec_modulator):
            return _DryBackend(entity, mod_name) if mod_name == "orient" else None

        self.entity, self.embodiment, self.registry = _build_body(factory)
        self.orient = self.entity.modulators["orient"]
        self.deltas = dict(DELTAS)
        self.executor = _Executor(self.registry, self.embodiment)
        self.sync_embodiment(self.az())

    def az(self) -> float:
        return max(-1.0, min(1.0, self.source + HEALTHY_GAIN * self.body + self._rng.gauss(0, 0.01)))

    def reader(self):
        return (self.az() * math.pi / 2 + math.pi / 2, True)

    def body_yaw(self) -> float:
        return self.body

    def goto_body_yaw(self, yaw: float, duration: float = 0.0) -> None:
        self.body = float(yaw)
        self.sync_embodiment(self.az())

    def recenter(self) -> None:
        self.goto_body_yaw(0.0)

    def gated_az(self) -> float | None:
        return round(statistics.median(self.az() for _ in range(GATED_READS)), 4)

    def sync_embodiment(self, az: float | None) -> None:
        if az is not None:
            self._world_set_axis(self.embodiment, "azimuth", float(az), default_range=(-1.0, 1.0), owner="doa_feed")

    def head_pose_deg(self) -> dict:
        return {"body_yaw_deg": round(math.degrees(self.body), 2)}

    def execute(self, affordance: str):
        return self.orient.execute(affordance, {})

    def close(self) -> None:
        pass


# ── the decision, through the production path ──────────────────────────────


def decide(agent: LoadedAgent, rig, sink: _ProvenanceSink) -> dict:
    """One production decision at the current pose: encode → recommend → provenance."""
    from maxim.runtime.agent_loop import _encode_current_clusters, propose_via_substrate

    clusters = _encode_current_clusters(agent.encoder, AGENT_ID, rig.executor)
    audio_cluster = clusters.get("audio")
    sink.last = None
    proposal = propose_via_substrate(
        nac=agent.nac, agent_id=AGENT_ID, executor=rig.executor, sensor_encoder=agent.encoder
    )
    prov = sink.take() or {}
    tool = proposal.action.get("tool_name") if proposal is not None else None
    return {
        "clusters": clusters,
        "audio_cluster": audio_cluster,
        "completed": (audio_cluster in agent.audio_nodes) if audio_cluster else None,
        "tool_name": tool,
        "affordance": tool.rsplit("_turn_", 1)[-1].join(["turn_", ""]) if tool and "_turn_" in tool else None,
        "learned_margin": prov.get("learned_margin"),
        "explore_decisive": prov.get("explore_decisive"),
        "score_components": prov.get("score_components"),
        "best_score": prov.get("best_score"),
        "runner_up_score": prov.get("runner_up_score"),
        "consulted_bias_by_modality": prov.get("consulted_bias_by_modality"),
        "n_candidates": prov.get("n_candidates"),
    }


def _correct_for(az: float, affordance: str | None) -> bool | None:
    if affordance is None:
        return None
    return (affordance == "turn_left") == (az < 0.0)


def _schedule(rng: random.Random, reps: int) -> list[tuple[float, bool]]:
    """Balanced blocks of the gated targets plus the exploratory placements; each
    entry is ``(target_az, exploratory)``."""
    order: list[tuple[float, bool]] = []
    for _ in range(reps):
        block = [(t, False) for t in TARGETS] + [(t, True) for t in EXPLORATORY_TARGETS]
        rng.shuffle(block)
        order.extend(block)
    return order


# ── run ──────────────────────────────────────────────────────────────────────


def _load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _phase1_passed(records_path: Path) -> bool:
    if not records_path.exists():
        return False
    for line in records_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "gate_I" and rec.get("verdict") == "PASS":
            return True
    return False


def cmd_run(args: argparse.Namespace) -> int:
    os.environ["MAXIM_SUBSTRATE_TOOL_WHITELIST"] = "turn_left,turn_right"  # the nursery's repertoire (S6)
    os.environ.pop("MAXIM_PLACE_CODE_EXTEROCEPTION", None)  # place code OFF, as in Phase B (provenance)
    manifest = _load_manifest(args.manifest)
    out_path = Path(args.out)
    if args.phase == 2 and not _phase1_passed(out_path):
        print(f"[STOP] no Phase 1 gate_I PASS record in {out_path} — stop rule I: Phase 2 does not run.")
        return 4
    log = JsonlLog(str(out_path))
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    prov = provenance(_HERE.parent.parent)

    def emit(event: str, **fields: object) -> None:
        log.write(event, run_id=run_id, phase=args.phase, dry_run=args.dry_run, **fields)

    from maxim.simulation import sim_logger

    sink = _ProvenanceSink()
    sim_logger.register_sim_sink(sink)

    if args.dry_run:
        rig = DryReadoutRig()
        emit("start", provenance=prov, frozen=manifest.get("frozen"), rig="dry")
        print(f"[dry-run] run_id={run_id} phase={args.phase}")
    else:
        host, source = resolve_host(args.host)
        if host is None:
            print("[FAIL] no robot address: --host <ip> or export MAXIM_REACHY_HOST=<ip>")
            return 2
        status = daemon_status(host) or {}
        sdk = installed_sdk_version()
        daemon_ver = status.get("version")
        print(f"[version] sdk={sdk} daemon={daemon_ver} hardware_id={status.get('hardware_id')}")
        if not sdk or not daemon_ver or sdk != daemon_ver:
            print("[FAIL] SDK/daemon version skew — refusing (skew fails silently).")
            return 3
        if not args.yes:
            print("\n  SOURCE: continuous speech (podcast) ~1-2 m, in FRONT of the base's neutral heading.")
            print("  The apparatus rotates the base to place the source at az ±0.5/±0.6; keep it still.")
            if input("  ready? [y/N] ").strip().lower() != "y":
                return 1
        rig = LiveReadoutRig(host)
        status2 = daemon_status(host) or {}
        mode = (status2.get("backend_status") or {}).get("motor_control_mode")
        if mode != "enabled":
            print(f"[FAIL] motor_control_mode={mode!r} — torque off; refusing.")
            rig.close()
            return 3
        rate, n = speech_rate_probe(rig.reader, args.probe_s)
        print(f"[audio] speech-gate rate over {args.probe_s:.0f}s: {rate:.2f} ({n} reads)")
        if rate < MIN_SPEECH_RATE:
            print(f"[FAIL] speech rate {rate:.2f} < {MIN_SPEECH_RATE} — start the source and re-run.")
            rig.close()
            return 3
        emit(
            "start",
            provenance=prov,
            frozen=manifest.get("frozen"),
            rig="live",
            host=host,
            sdk_version=sdk,
            daemon_version=daemon_ver,
            hardware_id=status.get("hardware_id"),
            speech_rate_probe=round(rate, 3),
            deltas=rig.deltas,
        )
        print(f"[start] run_id={run_id} log={out_path}")

    if args.dry_run:
        args.settle = 0.0  # the dry rig has nothing to settle; 360 trials × 2 s is robot time, not logic
    agents = [a for a in manifest["agents"] if not args.only or a["label"] in args.only]
    try:
        if args.phase == 1:
            rc = _phase1(agents, rig, sink, emit, args)
        else:
            rc = _phase2(agents, rig, sink, emit, args)
    finally:
        rig.close()
        sim_logger.unregister_sim_sink(sink)
    return rc


def _place(rig, target: float, settle: float, emit, label: str) -> float | None:
    """Apparatus: rotate the base so the fixed source sits at ``target`` az. The
    commanded offset is apparatus-only (never reaches the substrate)."""
    rig.recenter()
    offset = target / HEALTHY_GAIN
    rig.goto_body_yaw(offset)
    time.sleep(settle)
    az = rig.gated_az()
    rig.sync_embodiment(az)
    emit("apparatus_place", agent=label, target_az=target, commanded_offset_rad=round(offset, 4), measured_az=az)
    return az


def _phase1(agents, rig, sink, emit, args) -> int:
    results: dict[str, list[dict]] = {}
    for spec in agents:
        label = spec["label"]
        agent = LoadedAgent(spec, EXPLORE_PRIMARY)
        emit(
            "agent_load",
            agent=label,
            arm=spec["arm"],
            seed=spec["seed"],
            explore_weight=EXPLORE_PRIMARY,
            bias_entries=agent._bias_entries(),
            audio_nodes=sorted(agent.audio_nodes),
            nac_sha256=spec["nac_sha256"],
            ec_sha256=spec["ec_sha256"],
        )
        rng = random.Random(1000 + spec["seed"])
        rows = []
        for i, (target, exploratory) in enumerate(_schedule(rng, PROBE_PLACEMENTS)):
            az = _place(rig, target, args.settle, emit, label)
            if az is None:
                emit("probe", agent=label, i=i, target_az=target, exploratory=exploratory, invalid=True)
                print(f"  [{label}] probe {i}: no gated read — invalid (re-drawn)")
                continue
            d = decide(agent, rig, sink)
            correct = _correct_for(az, d["affordance"])
            margin = d["learned_margin"]
            with_margin = bool(correct) and margin is not None and abs(float(margin)) > MARGIN_FLOOR
            no_learned_pref = (d["learned_margin"] in (None, 0, 0.0)) and not (
                (d["consulted_bias_by_modality"] or {}).get("audio")
            )
            row = {
                "i": i,
                "target_az": target,
                "exploratory": exploratory,
                "az": az,
                "correct": correct,
                "correct_with_margin": with_margin,
                "no_learned_preference": no_learned_pref,
                **d,
            }
            rows.append(row)
            emit("probe", agent=label, arm=spec["arm"], seed=spec["seed"], head=rig.head_pose_deg(), **row)
            print(
                f"  [{label}] az {az:+.2f}{' (expl)' if exploratory else '       '} → cluster "
                f"{str(d['audio_cluster'])[:8]} completed={d['completed']} "
                f"tool={d['affordance']} margin={margin} correct={correct}"
            )
        results[label] = rows
        unchanged = agent.files_unchanged()
        emit("agent_done", agent=label, files_unchanged=unchanged, credited=0)
        if not unchanged:
            print(f"[FAIL] {label}: persisted files changed during readout — S3 violation")
            return 5
    verdict = _gate_I(agents, results)
    emit("gate_I", **verdict)
    print(f"[gate I] {verdict['verdict']}: {verdict['summary']}")
    return 0 if verdict["verdict"] == "PASS" else 6


def _gate_I(agents, results) -> dict:
    per_seed = {}
    for spec in agents:
        rows = [r for r in results.get(spec["label"], []) if not r.get("exploratory")]
        if not rows:
            continue
        completed = sum(1 for r in rows if r["completed"]) / len(rows)
        cwm = sum(1 for r in rows if r["correct_with_margin"]) / len(rows)
        acted = sum(1 for r in rows if r["tool_name"]) / len(rows)
        no_pref = sum(1 for r in rows if r["no_learned_preference"]) / len(rows)
        per_seed[spec["label"]] = {
            "arm": spec["arm"],
            "completed": round(completed, 3),
            "correct_with_margin": round(cwm, 3),
            "acted": round(acted, 3),
            "no_learned_preference": round(no_pref, 3),
        }
    taught = [v for k, v in per_seed.items() if v["arm"] == "taught" and not k.endswith("seed48")]
    taught_pass = sum(1 for v in taught if v["completed"] >= GATE_I_RATE and v["correct_with_margin"] >= GATE_I_RATE)
    controls = {k: v for k, v in per_seed.items() if v["arm"] != "taught"}
    controls_no_pref = all(v["no_learned_preference"] == 1.0 for v in controls.values()) if controls else None
    verdict = "PASS" if (taught_pass >= GATE_I_SEEDS and controls_no_pref is not False) else "FAIL"
    return {
        "verdict": verdict,
        "taught_seeds_passing": taught_pass,
        "controls_no_learned_preference": controls_no_pref,
        "per_seed": per_seed,
        "summary": (
            f"{taught_pass}/{len(taught)} taught seeds complete+correct with margin; "
            f"controls show no learned preference={controls_no_pref}"
        ),
    }


def _phase2(agents, rig, sink, emit, args) -> int:
    for spec in agents:
        label = spec["label"]
        for condition, weight in (("primary", EXPLORE_PRIMARY), ("secondary", EXPLORE_SECONDARY)):
            if args.primary_only and condition != "primary":
                continue
            agent = LoadedAgent(spec, weight)
            emit(
                "agent_load",
                agent=label,
                arm=spec["arm"],
                seed=spec["seed"],
                condition=condition,
                explore_weight=weight,
                bias_entries=agent._bias_entries(),
                nac_sha256=spec["nac_sha256"],
                ec_sha256=spec["ec_sha256"],
            )
            rng = random.Random(2000 + spec["seed"] + (7 if condition == "secondary" else 0))
            trials = _schedule(rng, TRIALS_PER_AGENT // len(TARGETS))
            i = 0
            invalid = 0
            while i < len(trials):
                target, exploratory = trials[i]
                az_before = _place(rig, target, args.settle, emit, label)
                if az_before is None:
                    invalid += 1
                    emit("trial", agent=label, condition=condition, i=i, target_az=target, invalid=True, stage="before")
                    if invalid > 6:
                        print(f"[FAIL] {label}: too many invalid reads — apparatus fault, block aborted")
                        return 7
                    continue
                d = decide(agent, rig, sink)
                aff = d["affordance"]
                head_pre = rig.head_pose_deg()
                body_pre = rig.body_yaw()
                res = None
                if aff is not None:
                    t0 = time.monotonic()
                    res = rig.execute(aff)
                    turn_s = round(time.monotonic() - t0, 2)
                    time.sleep(args.settle)
                else:
                    turn_s = 0.0
                body_post = rig.body_yaw()
                head_post = rig.head_pose_deg()
                az_after = rig.gated_az() if aff is not None else az_before
                if aff is not None and az_after is None:
                    invalid += 1
                    emit(
                        "trial",
                        agent=label,
                        condition=condition,
                        i=i,
                        target_az=target,
                        invalid=True,
                        stage="after",
                        affordance=aff,
                    )
                    continue
                meta = dict(getattr(res, "metadata", None) or {})
                toward = (abs(az_after) < abs(az_before) - TOWARD_EPS) if aff is not None else False
                sign_rule = _correct_for(az_before, aff)
                achieved = (body_post - body_pre) if (body_pre is not None and body_post is not None) else None
                row = {
                    "i": i,
                    "condition": condition,
                    "explore_weight": weight,
                    "target_az": target,
                    "exploratory": exploratory,
                    "az_before": az_before,
                    "az_after": az_after,
                    "affordance": aff,
                    "no_action": aff is None,
                    "toward": toward,
                    "sign_rule_correct": sign_rule,
                    "commanded_delta_rad": (rig.deltas.get(aff) if aff else None),
                    "achieved_delta_rad": (round(achieved, 4) if achieved is not None else None),
                    "success": bool(getattr(res, "success", False)) if res is not None else None,
                    "error": getattr(res, "error", None) if res is not None else None,
                    "turn_wall_s": turn_s,
                    "head_pre": head_pre,
                    "head_post": head_post,
                    "backend_metadata": meta,
                    **{
                        k: d[k]
                        for k in (
                            "clusters",
                            "audio_cluster",
                            "completed",
                            "tool_name",
                            "learned_margin",
                            "explore_decisive",
                            "score_components",
                            "best_score",
                            "runner_up_score",
                        )
                    },
                }
                emit("trial", agent=label, arm=spec["arm"], seed=spec["seed"], **row)
                print(
                    f"  [{label}/{condition}] {i + 1:2d}/{len(trials)} az {az_before:+.2f} → {str(aff):10s} "
                    f"→ {az_after:+.2f}  toward={toward} sign={sign_rule} margin={d['learned_margin']}"
                )
                i += 1
            unchanged = agent.files_unchanged()
            emit(
                "agent_done",
                agent=label,
                condition=condition,
                files_unchanged=unchanged,
                credited=0,
                invalid_reads=invalid,
            )
            if not unchanged:
                print(f"[FAIL] {label}: persisted files changed during readout — S3 violation")
                return 5
    return 0


# ── verdict ──────────────────────────────────────────────────────────────────


def _read_records(path: str) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def cmd_verdict(args: argparse.Namespace) -> int:
    recs = _read_records(args.records)
    gate_i = [r for r in recs if r.get("event") == "gate_I"]
    print(f"[gate I] {gate_i[-1]['verdict'] if gate_i else 'NOT RUN'}")
    all_primary = [
        r for r in recs if r.get("event") == "trial" and not r.get("invalid") and r.get("condition") == "primary"
    ]
    trials = [r for r in all_primary if not r.get("exploratory")]
    expl = [r for r in all_primary if r.get("exploratory")]
    if not trials:
        print("[gate T] no primary Phase 2 trials")
        return 1
    by_agent: dict[str, list[dict]] = {}
    for r in trials:
        by_agent.setdefault(r["agent"], []).append(r)
    arm_dir: dict[str, list[float]] = {}
    per_seed = {}
    for label, rows in by_agent.items():
        arm = rows[0]["arm"]
        if label.endswith("seed48"):
            continue
        d = sum(1 for r in rows if r["toward"]) / len(rows)
        per_seed[label] = round(d, 3)
        arm_dir.setdefault(arm, []).append(d)
    means = {arm: round(statistics.mean(v), 3) for arm, v in arm_dir.items()}
    taught_rows = [r for rows in by_agent.values() for r in rows if r["arm"] == "taught" and r["affordance"]]
    sign_agree = (
        sum(1 for r in taught_rows if bool(r["sign_rule_correct"]) == bool(r["toward"])) / len(taught_rows)
        if taught_rows
        else 0.0
    )
    t = means.get("taught", 0.0)
    learned = t >= GATE_T_LEARNED
    vs_sat = t - means.get("satiated", 1.0) >= GATE_T_MARGIN
    vs_nf = t - means.get("no_feed", 1.0) >= GATE_T_MARGIN
    taught_vals = arm_dir.get("taught", [])
    # L2 check with the S7 ceiling clause: a single repeated value is the phase-lock
    # signature only BELOW ceiling — three seeds at 1.00 is a pass, not an apparatus flag.
    spread = len(set(taught_vals)) > 1 or (bool(taught_vals) and min(taught_vals) >= 1.0)
    apparatus_ok = sign_agree >= GATE_T_SIGN_AGREEMENT and spread
    if not apparatus_ok:
        verdict = "APPARATUS"
    elif learned and vs_sat and vs_nf:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    expl_by_target: dict[str, dict] = {}
    for r in expl:
        if r["arm"] != "taught" or r["agent"].endswith("seed48"):
            continue
        key = f"{r['target_az']:+.1f}"
        e = expl_by_target.setdefault(key, {"n": 0, "toward": 0, "turn_left": 0})
        e["n"] += 1
        e["toward"] += int(bool(r["toward"]))
        e["turn_left"] += int(r.get("affordance") == "turn_left")
    secondary = [
        r
        for r in recs
        if r.get("event") == "trial"
        and not r.get("invalid")
        and r.get("condition") == "secondary"
        and not r.get("exploratory")
    ]
    sec_means = {}
    if secondary:
        by_arm: dict[str, list[dict]] = {}
        for r in secondary:
            if not r["agent"].endswith("seed48"):
                by_arm.setdefault(r["arm"], []).append(r)
        sec_means = {arm: round(sum(1 for r in rows if r["toward"]) / len(rows), 3) for arm, rows in by_arm.items()}
    summary = {
        "verdict": verdict,
        "primary_directedness_by_arm": means,
        "per_seed": per_seed,
        "taught_sign_rule_agreement": round(sign_agree, 3),
        "learned_live": learned,
        "taught_minus_satiated_ok": vs_sat,
        "taught_minus_no_feed_ok": vs_nf,
        "taught_seed_spread": spread,
        "secondary_explore_1_5_by_arm": sec_means,
        "exploratory_placements_taught": expl_by_target,
    }
    print(json.dumps(summary, indent=2))
    JsonlLog(args.records).write("gate_T", **summary)
    return 0 if verdict == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--archive", required=True)
    m.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--phase", type=int, choices=(1, 2), required=True)
    r.add_argument("--host", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--settle", type=float, default=1.0)
    r.add_argument("--probe-s", type=float, default=10.0)
    r.add_argument("--yes", action="store_true")
    r.add_argument("--only", nargs="*", default=None, help="agent labels to run (debugging; not a result)")
    r.add_argument("--primary-only", action="store_true", help="skip the secondary explore-1.5 block")
    v = sub.add_parser("verdict")
    v.add_argument("--records", required=True)
    args = ap.parse_args(argv)
    return {"manifest": cmd_manifest, "run": cmd_run, "verdict": cmd_verdict}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
