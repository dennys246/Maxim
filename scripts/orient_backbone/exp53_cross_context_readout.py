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

Exp 54 additions (``exp54_nurture_reachy_body_preregistration.md``; the Exp 53
defaults are untouched, so the 53 records/verdict shape and the demo script
keep working):

* ``run --body-ref bodies/reachy_mini_infant --factory`` — the nursery body is
  the ROBOT'S OWN (tool names ``reachy_mini_turn_left`` … ``_big``) and the
  orient backend is attached through the production
  ``make_reachy_orient_factory`` (deltas read from the YAML's ``head_yaw``
  self-effects). ``--delta`` is refused with ``--factory``: no δ map anywhere.
* ``sweep`` — az ∈ [−1, 1] step 0.1 through each taught seed's loaded EC (a
  fresh load per value, nothing saved) → ``54_targets.json`` with the bins, the
  strongest-bias bins, the gated targets by the declared procedure and the
  predicted wrong-way region as exploratory placements (declared BEFORE Phase B).
* ``run --targets 54_targets.json`` — the gated/exploratory placements from
  the sweep instead of the Exp 53 constants.
* ``manifest --experiment 54 --archive <phaseA workdir> --phase-a-records …`` —
  reads Phase A's per-run ``sim_reports/*/aut_{nac,ec}.json``; the exploratory
  agent is the weakest taught seed by late-bin directedness.
* Phase C = ``run --phase 1 --body-ref bodies/reachy_mini --factory --gate C``
  (the user's body, innate azimuth drive present, probe only) + ``verdict
  --gate C``: consulted audio bias ≠ 0 AND correct direction at ≥ 80 % of the
  gated placements for ≥ 2/3 taught seeds; controls consulted audio == 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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
import live_common  # noqa: E402
from live_common import JsonlLog, gated_azimuth, resolve_host  # noqa: E402

# ── frozen pre-registration constants (do not edit after the first Phase 1 record) ──
BODY_REF = "bodies/infant_operant"
AGENT_ID = "sim_aut"  # the nursery's AUT id — the learned bias keys start with it
DELTA_RAD = 0.55  # body-yaw step per turn ≈ the nursery's 0.30-az self_effect (S6)
DELTAS = {"turn_left": +DELTA_RAD, "turn_right": -DELTA_RAD}
TARGETS = (-0.3, -0.2, 0.5, 0.6)  # gated az targets — where the sweep shows the learning lives (amendment 1)
EXPLORATORY_TARGETS = (-0.6, 0.2)  # recorded, excluded from every gate (amendment 1)
TRIALS_PER_AGENT = 12  # 4 gated targets × 3 (+ exploratory placements × 3, not gated)
TRIALS_PER_TARGET = 3  # Phase 2: each gated target (and each exploratory placement) three times
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
MIN_SPEECH_RATE = 0.50  # H1's floor (amendment 2: the VAD flag under-reports speech energy by ~0.25)
GATED_READS = 5

# ── Exp 54 (frozen in exp54_nurture_reachy_body_preregistration.md) ──
EXP54_BODY_REF = "bodies/reachy_mini_infant"  # the robot's own body, azimuth drive removed, hunger/thirst added
EXP54_USER_BODY_REF = "bodies/reachy_mini"  # Phase C: the body a user's agentic_runtime instantiates
SWEEP_STEP = 0.1  # az grid for the sweep
FRONT_HEMISPHERE_MAX = 0.6  # gated targets are clamped to |az| ≤ 0.6
GATE_C_RATE = 0.80
GATE_C_SEEDS = 2


def _affordance_of(tool_name: str | None, entity_name: str | None = None) -> str | None:
    """Tool name → orient affordance, single-sourced. Tool names are
    ``f"{entity.name}_{affordance}"`` (tool_bridge); strip the KNOWN entity prefix
    when given (immune to an entity name containing ``_turn_``), else fall back to
    the ``_turn_`` split the Exp 53 records were parsed with. ``tool:``-prefixed
    NAc signatures are accepted."""
    if not tool_name:
        return None
    name = tool_name[5:] if tool_name.startswith("tool:") else tool_name
    if entity_name and name.startswith(entity_name + "_"):
        return name[len(entity_name) + 1 :]
    if "_turn_" in name:
        return "turn_" + name.rsplit("_turn_", 1)[-1]
    return None


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


def _weakest_taught_seed(records_path: str, exclude: tuple[int, ...]) -> tuple[int, float]:
    """The exploratory agent for Exp 54: the taught seed with the LOWEST late-bin
    (act3+act4) directedness in the Phase A campaign record, outside the gated
    seeds (the Exp 53 rule — seed 48 was the weak learner — made a procedure)."""
    late = ("act3_consolidating", "act4_autonomous")
    best: tuple[int, float] | None = None
    skipped = 0
    for line in Path(records_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if rec.get("arm") != "taught" or int(rec.get("seed", -1)) in exclude:
            continue
        fade = rec.get("fade") or {}
        vals = [float(fade[a].get("directedness", 0.0)) for a in late if a in fade]
        if not vals:
            continue
        d = sum(vals) / len(vals)
        if best is None or d < best[1]:
            best = (int(rec["seed"]), d)
    if skipped:
        print(f"  [warn] {records_path}: {skipped} undecodable line(s) skipped while picking the weakest seed")
    if best is None:
        raise RuntimeError(f"no taught seed outside {exclude} in {records_path}")
    return best


def _factory_deltas(entity) -> dict[str, float]:
    """The production factory's own read of the body: ``make_reachy_orient_factory``
    over the orient modulator, robot-less (the backend constructor is offline-safe).
    Raises when the body declares no ``head_yaw`` self-effect — i.e. the factory
    path would attach nothing (the infant_operant case Exp 53 needed δ for)."""
    from maxim.hardware.reachy.motor_backend import make_reachy_orient_factory

    factory = make_reachy_orient_factory(robot=None)
    for ent in entity.walk():
        mod = ent.modulators.get("orient")
        if mod is None:
            continue
        backend = factory(ent, "orient", mod)
        if backend is None:
            raise RuntimeError(
                f"{getattr(ent, 'name', '?')}: orient affordances declare no head_yaw self-effect — "
                "the production factory attaches nothing (use --delta on that body, not --factory)"
            )
        return dict(backend._deltas)
    raise RuntimeError("body has no orient modulator")


def cmd_manifest(args: argparse.Namespace) -> int:
    archive = Path(args.archive).expanduser()
    agents = []
    wanted = [(arm, seed, False) for arm, seeds in ARMS.items() for seed in seeds]
    if args.experiment == "54":
        if not args.phase_a_records:
            print("[FAIL] --experiment 54 needs --phase-a-records <54_phaseA_nursery.jsonl> (the weakest taught seed)")
            return 2
        weak_seed, weak_late = _weakest_taught_seed(args.phase_a_records, ARMS["taught"])
        print(f"  exploratory agent: taught seed{weak_seed} (Phase A late-bin directedness {weak_late:.3f})")
        wanted += [("taught", weak_seed, True)]
    else:
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
    if args.experiment == "54":
        from maxim.embodiment.component_registry import ComponentRegistry

        body = ComponentRegistry().instantiate(EXP54_BODY_REF)
        frozen = {
            "body_ref": EXP54_BODY_REF,
            "user_body_ref": EXP54_USER_BODY_REF,
            "agent_id": AGENT_ID,
            "factory": True,
            "deltas_rad": _factory_deltas(body),  # read by the production factory from the YAML — no δ map
            "targets_az": "declared by the sweep procedure (54_targets.json), before Phase B",
            "trials_per_target": TRIALS_PER_TARGET,
            "explore_primary": EXPLORE_PRIMARY,
            "explore_secondary": EXPLORE_SECONDARY,
            "phase_a_records": args.phase_a_records,
        }
        experiment = "54_nurture_reachy_body"
    else:
        frozen = {
            "body_ref": BODY_REF,
            "agent_id": AGENT_ID,
            "deltas_rad": DELTAS,
            "targets_az": TARGETS,
            "trials_per_agent": TRIALS_PER_AGENT,
            "explore_primary": EXPLORE_PRIMARY,
            "explore_secondary": EXPLORE_SECONDARY,
        }
        experiment = "53_cross_context_readout"
    out = Path(args.out)
    live_common._provenance.preflight_gated_record_or_exit(_HERE.parent.parent, out, allow_dirty=args.allow_dirty)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "_format_version": "1.0",
                "experiment": experiment,
                "archive": str(archive),
                "frozen": frozen,
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


def _orient_affordances(entity) -> list[str]:
    """The body's orient affordances the ``turn_left,turn_right`` whitelist admits
    (substring match — on the Reachy bodies that is the 4-tool repertoire, S6)."""
    for ent in entity.walk():
        mod = ent.modulators.get("orient")
        if mod is not None:
            affs = list(getattr(mod, "_affordances", None) or {})
            return [a for a in affs if "turn_left" in a or "turn_right" in a]
    raise RuntimeError(f"{getattr(entity, 'name', '?')}: no orient modulator")


def _build_body(modulator_factory, body_ref: str = BODY_REF):
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.embodiment.spec import attach_backends
    from maxim.embodiment.tool_bridge import generate_tools_for_entity
    from maxim.tools.registry import ToolRegistry

    entity = ComponentRegistry().instantiate(body_ref)
    if modulator_factory is not None:
        attach_backends(entity, modulator_factory=modulator_factory)
    embodiment = Embodiment(entity)
    registry = ToolRegistry()
    generate_tools_for_entity(entity, registry, embodiment=embodiment)
    names = list(registry.list())
    # The learned bias keys are tool:<entity.name>_<affordance>: the registered tool
    # names ARE the namespace the nursery wrote — check every orient tool is there.
    for aff in _orient_affordances(entity):
        want = f"{entity.name}_{aff}"
        if want not in names:
            raise RuntimeError(f"tool {want!r} not registered — got {names}")
    return entity, embodiment, registry


class LiveReadoutRig:
    def __init__(self, host: str, body_ref: str = BODY_REF, factory_mode: bool = False) -> None:
        from maxim.embodiment.audio_localization import DoAFeed
        from maxim.hardware.reachy.controller import ReachyMiniController
        from maxim.hardware.reachy.motor_backend import ReachyOrientMotorBackend, make_reachy_orient_factory

        self.host = host
        self.dry = False
        self.body_ref = body_ref
        self.factory_mode = factory_mode
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

        if factory_mode:
            # Exp 54: the PRODUCTION factory — deltas read from the body's own
            # head_yaw self-effects; the harness declares no step size at all.
            factory = make_reachy_orient_factory(robot, maxim=shim)
        else:

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

        self.entity, self.embodiment, self.registry = _build_body(factory, body_ref)
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
            raise RuntimeError(
                "orient modulator has NO backend attached — the production path is not bound"
                + (" (does this body declare head_yaw self-effects? --factory needs them)" if factory_mode else "")
            )
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
    """Offline stand-in: the body + a dry orient backend over a modeled source.
    Same production body/tools/encode path; only the motor and the sensor are fake.
    ``factory_mode`` takes the dry backend's deltas from the PRODUCTION factory's
    own read of the body (``_factory_deltas``) — so a body the factory cannot bind
    fails here, offline, exactly as it would on the robot."""

    def __init__(
        self, ratio: float = 0.95, seed: int = 1, body_ref: str = BODY_REF, factory_mode: bool = False
    ) -> None:
        from maxim.embodiment.audio_localization import world_set_axis

        self.dry = True
        self.host = "dry"
        self.body_ref = body_ref
        self.factory_mode = factory_mode
        self._rng = random.Random(seed)
        self.ratio = ratio
        self.body = 0.0
        self.source = 0.0
        self._world_set_axis = world_set_axis
        rig = self

        class _DryBackend:
            def __init__(self, entity, mod_name):
                self._deltas = _factory_deltas(entity) if factory_mode else dict(DELTAS)
                self._entity = entity
                self._entity_name = getattr(entity, "name", "") or ""

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
                    entity_name=self._entity_name,
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

        self.entity, self.embodiment, self.registry = _build_body(factory, body_ref)
        self.orient = self.entity.modulators["orient"]
        self.deltas = dict(self.orient._backend._deltas)
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
        "affordance": _affordance_of(tool, getattr(getattr(rig, "entity", None), "name", None)),
        "learned_margin": prov.get("learned_margin"),
        "explore_decisive": prov.get("explore_decisive"),
        "score_components": prov.get("score_components"),
        "best_score": prov.get("best_score"),
        "runner_up_score": prov.get("runner_up_score"),
        "consulted_bias_by_modality": prov.get("consulted_bias_by_modality"),
        "n_candidates": prov.get("n_candidates"),
    }


def _correct_for(az: float, affordance: str | None) -> bool | None:
    """Direction-only: any leftward affordance (``turn_left`` OR ``turn_left_big``)
    is correct for a source on the left — the Exp 52/54 directedness rule."""
    if affordance is None:
        return None
    return affordance.startswith("turn_left") == (az < 0.0)


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


def _apply_targets(path: str, allow_incomplete: bool = False) -> dict:
    """Replace the Exp 53 target constants with the sweep-declared placements.

    The frozen procedure is "two magnitudes each" side: a one-sided or 3-target
    set would let an always-one-way policy satisfy the direction rule most of the
    time, so an incomplete declaration is REFUSED unless the operator passes
    ``--allow-incomplete-targets`` (recorded in the start record)."""
    global TARGETS, EXPLORATORY_TARGETS
    data = json.loads(Path(path).read_text())
    gated = [float(t) for t in data.get("gated_targets") or []]
    expl = [float(t) for t in data.get("exploratory_targets") or []]
    if not gated:
        raise RuntimeError(f"{path}: no gated_targets declared")
    for t in gated:
        if t == 0.0 or abs(t) > FRONT_HEMISPHERE_MAX + 1e-9:
            raise RuntimeError(f"{path}: gated target {t:+.2f} is not in the front hemisphere (0 < |az| ≤ 0.6)")
    left = sorted(t for t in gated if t < 0)
    right = sorted(t for t in gated if t > 0)
    complete = len(left) == 2 and len(right) == 2 and not data.get("incomplete")
    if not complete and not allow_incomplete:
        raise RuntimeError(
            f"{path}: incomplete gated targets (left {left}, right {right}, flags {data.get('flags')}) — "
            "the procedure declares two magnitudes per direction; pass --allow-incomplete-targets to run anyway"
        )
    TARGETS = tuple(gated)
    EXPLORATORY_TARGETS = tuple(expl)
    return data


def cmd_run(args: argparse.Namespace) -> int:
    if args.gate == "C" and args.phase != 1:
        print("[FAIL] --gate C is the Phase C probe (no motion) — it runs with --phase 1 only.")
        return 2
    if args.gate == "C" and not args.whitelist:
        # Phase C = the USER's tool space: a user's agentic_runtime on bodies/reachy_mini
        # offers listen / look_at / recenter / nod … to recommend_action, and the
        # sentence being measured is "consulted on a user's robot with no remap".
        # The S6 nursery whitelist would shrink that to a 4-candidate contest.
        os.environ.pop("MAXIM_SUBSTRATE_TOOL_WHITELIST", None)
    else:
        os.environ["MAXIM_SUBSTRATE_TOOL_WHITELIST"] = "turn_left,turn_right"  # the nursery's repertoire (S6)
    if args.delta is not None and args.factory:
        print("[FAIL] --delta is refused with --factory: the step size is the body's own (Exp 54, no δ map).")
        return 2
    if args.delta is not None:
        # Exp 53b: the declared step size is the one pre-registered change; stamped in every start record.
        DELTAS.update({"turn_left": +float(args.delta), "turn_right": -float(args.delta)})
    targets_decl = (
        _apply_targets(args.targets, allow_incomplete=args.allow_incomplete_targets) if args.targets else None
    )
    os.environ.pop("MAXIM_PLACE_CODE_EXTEROCEPTION", None)  # place code OFF, as in Phase B (provenance)
    manifest = _load_manifest(args.manifest)
    out_path = Path(args.out)
    if args.phase == 2 and not _phase1_passed(out_path):
        print(f"[STOP] no Phase 1 gate_I PASS record in {out_path} — stop rule I: Phase 2 does not run.")
        return 4
    prov = provenance(
        _HERE.parent.parent, out_path=out_path, allow_dirty=args.allow_dirty
    )  # refuses before any file exists
    log = JsonlLog(str(out_path), allow_dirty=args.allow_dirty)
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"

    def emit(event: str, **fields: object) -> None:
        log.write(event, run_id=run_id, phase=args.phase, dry_run=args.dry_run, **fields)

    from maxim.simulation import sim_logger

    sink = _ProvenanceSink()
    sim_logger.register_sim_sink(sink)
    # D35: the controller's WARNINGs (the H1 F1 achieved-vs-commanded divergence early
    # warning among them) were console-only; R1 could not archive its one body_yaw
    # divergence. They are records now, joined to trials by ts.
    warn_handler = _WarningsToJsonl(emit)
    reachy_logger = logging.getLogger("maxim.hardware.reachy")
    reachy_logger.addHandler(warn_handler)

    common = dict(
        experiment=manifest.get("experiment"),
        only=list(args.only) if args.only else None,  # a subset run is debugging, never a result (D34)
        body_ref=args.body_ref,
        factory=bool(args.factory),
        gate=args.gate,
        tool_whitelist=os.environ.get("MAXIM_SUBSTRATE_TOOL_WHITELIST"),
        allow_incomplete_targets=bool(args.allow_incomplete_targets),
        targets=list(TARGETS),
        exploratory_targets=list(EXPLORATORY_TARGETS),
        targets_file=args.targets,
        targets_declaration=(targets_decl or {}).get("procedure"),
    )
    if args.dry_run:
        rig = DryReadoutRig(body_ref=args.body_ref, factory_mode=args.factory)
        emit("start", provenance=prov, frozen=manifest.get("frozen"), rig="dry", deltas=rig.deltas, **common)
        print(
            f"[dry-run] run_id={run_id} phase={args.phase} body={args.body_ref} factory={args.factory} deltas={rig.deltas}"
        )
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
        rig = LiveReadoutRig(host, body_ref=args.body_ref, factory_mode=args.factory)
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
            **common,
        )
        print(f"[start] run_id={run_id} log={out_path} body={args.body_ref} factory={args.factory} deltas={rig.deltas}")

    if args.dry_run:
        args.settle = 0.0  # the dry rig has nothing to settle; 360 trials × 2 s is robot time, not logic
    agents = [a for a in manifest["agents"] if not args.only or a["label"] in args.only]
    # D36: every run closes with a `run_end` record — complete (rc 0), stopped (a stop
    # rule returned non-zero), interrupted (Ctrl-C) or error (traceback) — so a file
    # can say whether a partial run was interrupted or crashed. A run without one
    # died before the harness could write it.
    status, rc, err = "error", None, None
    try:
        if args.phase == 1:
            rc = _phase1(agents, rig, sink, emit, args)
        else:
            rc = _phase2(agents, rig, sink, emit, args)
        status = "complete" if rc == 0 else "stopped"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException as exc:
        err = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            emit(
                "run_end",
                status=status,
                rc=rc,
                error=err,
                # The dirty check ran at open; a 5 h robot session on a live branch is
                # the incident's shape, so the tree is re-read at close and stamped here.
                working_tree_dirty_src_scripts=live_common._provenance.working_tree_dirty(_HERE.parent.parent),
            )
        finally:
            reachy_logger.removeHandler(warn_handler)
            rig.close()
            sim_logger.unregister_sim_sink(sink)
    return rc


class _WarningsToJsonl(logging.Handler):
    """WARNING+ log records from the controller become `controller_warning` records (D35)."""

    def __init__(self, emit) -> None:
        super().__init__(level=logging.WARNING)
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit("controller_warning", logger=record.name, level=record.levelname, message=record.getMessage())
        except Exception:  # noqa: BLE001 — logging handlers must never raise into the caller
            self.handleError(record)


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
            exploratory_agent=bool(spec.get("exploratory")),
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
            emit(
                "probe",
                agent=label,
                arm=spec["arm"],
                seed=spec["seed"],
                exploratory_agent=bool(spec.get("exploratory")),
                head=rig.head_pose_deg(),
                **row,
            )
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
    if args.gate == "C":
        # Informative: the frozen text's "Gate I only" — instrument completion is asserted,
        # not assumed, even though Phase C's verdict is Gate C.
        gate_i = _gate_I(agents, results)
        emit("gate_I", informative=True, **gate_i)
        print(f"[gate I, informative] {gate_i['verdict']}: {gate_i['summary']}")
        verdict = _gate_C(agents, results)
        emit("gate_C", **verdict)
        print(f"[gate C] {verdict['verdict']}: {verdict['summary']}")
    else:
        verdict = _gate_I(agents, results)
        emit("gate_I", **verdict)
        print(f"[gate I] {verdict['verdict']}: {verdict['summary']}")
    return 0 if verdict["verdict"] == "PASS" else 6


def _is_exploratory_agent(spec: dict) -> bool:
    return bool(spec.get("exploratory")) or str(spec.get("label", "")).endswith("seed48")


def _consulted_audio(row: dict) -> float | None:
    """The consulted audio bias, or ``None`` when the provenance record is malformed —
    never 0.0, which would read as a control PASS (review fold)."""
    cbm = row.get("consulted_bias_by_modality")
    if cbm is None:
        return 0.0  # no modality consulted at all (e.g. no candidate tools): a genuine zero
    if not isinstance(cbm, dict):
        return None
    val = cbm.get("audio")
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _gate_C(agents, results) -> dict:
    """Exp 54 Phase C — the user path: under plain ``bodies/reachy_mini`` (innate
    azimuth drive present), is the nursery-taught audio bias CONSULTED and does it
    pick the correct direction? Taught: ≥ 80 % of gated placements with a non-zero
    consulted audio bias AND the correct direction, for ≥ 2 of 3 seeds. Controls:
    consulted audio bias 0 at every placement."""
    per_seed = {}
    for spec in agents:
        rows = [r for r in results.get(spec["label"], []) if not r.get("exploratory")]
        if not rows:
            continue
        # Unknown (malformed provenance) counts AGAINST both sides: as "consulted" for the
        # control zero-check and as not-consulted-correct for the taught seeds.
        vals = [_consulted_audio(r) for r in rows]
        consulted_n = sum(1 for v in vals if v is None or v != 0.0)
        consulted_correct = sum(1 for r, v in zip(rows, vals) if v is not None and v != 0.0 and r.get("correct"))
        per_seed[spec["label"]] = {
            "arm": spec["arm"],
            "exploratory": _is_exploratory_agent(spec),
            "consulted": round(consulted_n / len(rows), 3),
            "consulted_and_correct": round(consulted_correct / len(rows), 3),
            "malformed": sum(1 for v in vals if v is None),
            "acted": round(sum(1 for r in rows if r.get("tool_name")) / len(rows), 3),
        }
    taught = [v for v in per_seed.values() if v["arm"] == "taught" and not v["exploratory"]]
    taught_pass = sum(1 for v in taught if v["consulted_and_correct"] >= GATE_C_RATE)
    controls = {k: v for k, v in per_seed.items() if v["arm"] != "taught"}
    controls_zero = all(v["consulted"] == 0.0 for v in controls.values()) if controls else None
    verdict = "PASS" if (taught_pass >= GATE_C_SEEDS and controls_zero is not False) else "FAIL"
    return {
        "verdict": verdict,
        "taught_seeds_passing": taught_pass,
        "controls_consulted_zero": controls_zero,
        "per_seed": per_seed,
        "summary": (
            f"{taught_pass}/{len(taught)} taught seeds consulted+correct at ≥ {GATE_C_RATE:.0%} of placements; "
            f"controls consulted audio bias 0 = {controls_zero}"
        ),
    }


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
            "exploratory": _is_exploratory_agent(spec),
            "completed": round(completed, 3),
            "correct_with_margin": round(cwm, 3),
            "acted": round(acted, 3),
            "no_learned_preference": round(no_pref, 3),
        }
    taught = [v for v in per_seed.values() if v["arm"] == "taught" and not v["exploratory"]]
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
            if args.condition != "both" and condition != args.condition:
                continue
            agent = LoadedAgent(spec, weight)
            emit(
                "agent_load",
                agent=label,
                arm=spec["arm"],
                seed=spec["seed"],
                exploratory_agent=bool(spec.get("exploratory")),
                condition=condition,
                explore_weight=weight,
                bias_entries=agent._bias_entries(),
                nac_sha256=spec["nac_sha256"],
                ec_sha256=spec["ec_sha256"],
            )
            rng = random.Random(2000 + spec["seed"] + (7 if condition == "secondary" else 0))
            trials = _schedule(rng, TRIALS_PER_TARGET)
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
                emit(
                    "trial",
                    agent=label,
                    arm=spec["arm"],
                    seed=spec["seed"],
                    exploratory_agent=bool(spec.get("exploratory")),
                    **row,
                )
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


def _record_is_exploratory_agent(r: dict) -> bool:
    if "exploratory_agent" in r:
        return bool(r["exploratory_agent"])
    return str(r.get("agent", "")).endswith("seed48")  # Exp 53 records predate the flag


class VerdictError(RuntimeError):
    """`verdict` refuses rather than guess which run a file's records belong to (D34)."""


class Run:
    """One `start` record and everything written under its run_id."""

    def __init__(self, run_id: str, phase: int | None) -> None:
        self.run_id = run_id
        self.phase = phase
        self.start: dict | None = None
        self.loaded: set[str] = set()
        self.done: set[str] = set()
        self.conditions: set[str] = set()
        self.run_end: dict | None = None
        self.n_trials = 0
        self.n_probes = 0

    @property
    def status(self) -> str:
        """complete | stopped | interrupted | error | debug | partial.

        `run_end` (D36) is authoritative. A `--only` subset run is `debug` — never a
        result. Files that predate `run_end` are `complete` iff every agent that
        loaded also finished, else `partial` (interrupted-vs-crashed unknowable).
        Caveat for files that also predate the `only` key on `start` (before
        2026-08-29): a legacy `--only` debugging run whose few agents all finished
        reads as `complete` — pin the intended run with `--run-id` for those.
        """
        if self.start is not None and self.start.get("only"):
            return "debug"
        if self.run_end is not None:
            return str(self.run_end.get("status"))
        if self.loaded and self.loaded <= self.done:
            return "complete"
        return "partial"

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "status": self.status,
            "conditions": sorted(c for c in self.conditions if c),
            "agents_loaded": len(self.loaded),
            "agents_done": len(self.done),
            "probes": self.n_probes,
            "trials": self.n_trials,
        }


def runs_of(recs: list[dict]) -> list[Run]:
    """Group a records file by run_id, in first-seen order (gate summaries carry none)."""
    runs: dict[str, Run] = {}
    order: list[str] = []
    for r in recs:
        rid = r.get("run_id")
        if rid is None:
            continue
        run = runs.get(rid)
        if run is None:
            run = runs[rid] = Run(rid, r.get("phase"))
            order.append(rid)
        ev = r.get("event")
        if ev == "start":
            run.start = r
        elif ev == "agent_load":
            run.loaded.add(str(r.get("agent")))
        elif ev == "agent_done":
            run.done.add(str(r.get("agent")))
        elif ev == "trial":
            run.n_trials += 1
            run.conditions.add(str(r.get("condition")))
        elif ev == "probe":
            run.n_probes += 1
        elif ev == "run_end":
            run.run_end = r
    return [runs[rid] for rid in order]


def select_run(
    runs: list[Run], *, phase: int, condition: str | None = None, pinned: tuple[str, ...] = ()
) -> Run | None:
    """The ONE complete run for (phase, condition), or None when there is none.

    D34: `verdict` used to pool every trial in the file across run_ids — partial
    starts, re-runs and debugging subsets included. Now a run is eligible only when
    it is COMPLETE (every loaded agent has an `agent_done`; `run_end.status ==
    complete` when present); `--run-id` restricts the eligible set; two eligible
    runs for the same (phase, condition) is a refusal, never a silent "last one".
    """
    cands = [r for r in runs if r.phase == phase and (condition is None or condition in r.conditions)]
    pinned_here = [r for r in cands if r.run_id in pinned]
    if pinned_here:
        # --run-id disambiguates the block(s) it names; other blocks select normally.
        cands = pinned_here
        not_complete = [r for r in cands if r.status != "complete"]
        if not_complete:
            raise VerdictError(
                "pinned run(s) are not complete: "
                + ", ".join(f"{r.run_id} ({r.status})" for r in not_complete)
                + " — a partial run cannot back a verdict"
            )
    cands = [r for r in cands if r.status == "complete"]
    if not cands:
        return None
    if len(cands) > 1:
        label = f"phase {phase}" + (f" / {condition}" if condition else "")
        raise VerdictError(
            f"{len(cands)} complete {label} runs in the file ({', '.join(r.run_id for r in cands)}) — "
            "pass --run-id <id> to say which one the verdict is about"
        )
    return cands[0]


def cmd_verdict(args: argparse.Namespace) -> int:
    # The gate record is appended to the records file — refuse a dirty-tree write
    # BEFORE computing, so the refusal is the first thing printed, not the last.
    out_log = JsonlLog(args.records, allow_dirty=args.allow_dirty)
    recs = _read_records(args.records)
    runs = runs_of(recs)
    pinned = tuple(args.run_id or ())
    unknown = sorted(set(pinned) - {r.run_id for r in runs})
    if unknown:
        print(f"[verdict] REFUSED: --run-id names run(s) not in the file: {', '.join(unknown)}")
        return 2
    try:
        run_p1 = select_run(runs, phase=1, pinned=pinned)
        run_primary = select_run(runs, phase=2, condition="primary", pinned=pinned)
        run_secondary = select_run(runs, phase=2, condition="secondary", pinned=pinned)
    except VerdictError as exc:
        print(f"[verdict] REFUSED: {exc}")
        return 2
    used = {r for r in (run_p1, run_primary, run_secondary) if r is not None}
    runs_used = {
        "phase1": run_p1.run_id if run_p1 else None,
        "primary": run_primary.run_id if run_primary else None,
        "secondary": run_secondary.run_id if run_secondary else None,
    }
    runs_excluded = [r.summary() for r in runs if r not in used]
    for r in runs_excluded:
        print(f"[verdict] excluded run {r['run_id']} phase={r['phase']} status={r['status']} trials={r['trials']}")

    def _in(run: Run | None):
        rid = run.run_id if run else None
        return lambda r: r.get("run_id") == rid

    if args.gate == "C":
        probes = [r for r in recs if r.get("event") == "probe" and not r.get("invalid") and _in(run_p1)(r)]
        if not probes:
            print("[gate C] no probe records from a complete Phase 1 run")
            return 1
        by_agent: dict[str, list[dict]] = {}
        specs: dict[str, dict] = {}
        for r in probes:
            by_agent.setdefault(r["agent"], []).append(r)
            specs.setdefault(
                r["agent"],
                {
                    "label": r["agent"],
                    "arm": r.get("arm"),
                    "seed": r.get("seed"),
                    "exploratory": _record_is_exploratory_agent(r),
                },
            )
        verdict = _gate_C(list(specs.values()), by_agent)
        verdict = {**verdict, "runs_used": runs_used, "runs_excluded": runs_excluded}
        print(json.dumps(verdict, indent=2))
        out_log.write("gate_C", **verdict)
        return 0 if verdict["verdict"] == "PASS" else 1
    gate_i = [r for r in recs if r.get("event") == "gate_I" and _in(run_p1)(r)]
    print(f"[gate I] {gate_i[-1]['verdict'] if gate_i else 'NOT RUN'}")
    all_primary = [
        r
        for r in recs
        if r.get("event") == "trial"
        and not r.get("invalid")
        and r.get("condition") == "primary"
        and _in(run_primary)(r)
    ]
    trials = [r for r in all_primary if not r.get("exploratory")]
    expl = [r for r in all_primary if r.get("exploratory")]
    if not trials:
        print("[gate T] no primary Phase 2 trials from a complete run")
        return 1
    by_agent: dict[str, list[dict]] = {}
    for r in trials:
        by_agent.setdefault(r["agent"], []).append(r)
    arm_dir: dict[str, list[float]] = {}
    per_seed = {}
    for label, rows in by_agent.items():
        arm = rows[0]["arm"]
        if _record_is_exploratory_agent(rows[0]):
            continue
        d = sum(1 for r in rows if r["toward"]) / len(rows)
        per_seed[label] = round(d, 3)
        arm_dir.setdefault(arm, []).append(d)
    means = {arm: round(statistics.mean(v), 3) for arm, v in arm_dir.items()}
    taught_rows = [
        r
        for label, rows in by_agent.items()
        for r in rows
        if r["arm"] == "taught" and r["affordance"] and not _record_is_exploratory_agent(r)
    ]
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
        if r["arm"] != "taught" or _record_is_exploratory_agent(r):
            continue
        key = f"{r['target_az']:+.1f}"
        e = expl_by_target.setdefault(key, {"n": 0, "toward": 0, "turn_left": 0})
        e["n"] += 1
        e["toward"] += int(bool(r["toward"]))
        e["turn_left"] += int(str(r.get("affordance") or "").startswith("turn_left"))  # any leftward incl. _big
    secondary = [
        r
        for r in recs
        if r.get("event") == "trial"
        and not r.get("invalid")
        and r.get("condition") == "secondary"
        and not r.get("exploratory")
        and _in(run_secondary)(r)
    ]
    sec_means = {}
    if secondary:
        by_arm: dict[str, list[dict]] = {}
        for r in secondary:
            if not _record_is_exploratory_agent(r):
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
        "runs_used": runs_used,
        "runs_excluded": runs_excluded,
    }
    print(json.dumps(summary, indent=2))
    out_log.write("gate_T", **summary)
    return 0 if verdict == "PASS" else 1


# ── sweep (Exp 54: targets by declared procedure, not by number) ────────────

SWEEP_PROCEDURE = (
    "az in [-1, 1] step 0.1 through each gated taught seed's loaded EC (fresh load per value, "
    "explore 0, nothing saved). Bins = maximal runs of consecutive az values completing into the "
    "same audio cluster; a bin's left/right strength = the max persisted bias among its turn_left*/"
    "turn_right* keys; the LEFT (RIGHT) bin = the bin with the strongest left (right) strength. "
    "Eligible left targets = az values in the LEFT bin of a majority of seeds with az < 0 and "
    "|az| <= 0.6 (right: az > 0); two magnitudes per direction = the grid value nearest the "
    "eligible centroid and its neighbour one step further from centre (one step closer if the outer "
    "neighbour is not eligible). Exploratory placements = the grid value nearest the centroid of the "
    "predicted wrong-way region (values where a majority of seeds' frozen probe picks the wrong "
    "direction with |learned_margin| > 0.11; az = 0 has no direction and is excluded), if any. "
    "Grid ties resolve toward centre; an exact strength tie between bins resolves to the bin nearer centre."
)


def _sweep_values(step: float = SWEEP_STEP) -> list[float]:
    n = int(2.0 / step + 1e-9)
    return [round(-1.0 + i * step, 4) for i in range(n + 1) if -1.0 + i * step <= 1.0 + 1e-9]


def _cluster_biases(agent: LoadedAgent, cluster_id: str | None) -> dict[str, float]:
    """Persisted ``cluster_reward_bias`` entries for one cluster, keyed by affordance."""
    out: dict[str, float] = {}
    if not cluster_id:
        return out
    table = getattr(agent.nac, "_cluster_reward_bias", None) or {}
    for key, val in table.items():
        if not (isinstance(key, tuple) and len(key) == 3 and key[1] == cluster_id):
            continue
        tsig = str(key[2])
        aff = _affordance_of(tsig) or tsig
        out[aff] = round(float(val), 4)
    return out


def _bins_from_rows(rows: list[dict]) -> list[dict]:
    bins: list[dict] = []
    for r in rows:
        cid = r.get("audio_cluster")
        if bins and bins[-1]["cluster"] == cid:
            bins[-1]["az"].append(r["az"])
        else:
            bins.append({"cluster": cid, "az": [r["az"]], "biases": dict(r.get("biases") or {})})
    for b in bins:
        b["az_min"], b["az_max"] = min(b["az"]), max(b["az"])
        b["centroid"] = round(sum(b["az"]) / len(b["az"]), 3)
        b["left_strength"] = max([v for a, v in b["biases"].items() if a.startswith("turn_left")] or [0.0])
        b["right_strength"] = max([v for a, v in b["biases"].items() if a.startswith("turn_right")] or [0.0])
    return bins


def _strongest(bins: list[dict], key: str) -> dict | None:
    """The bin with the largest ``key`` strength; an exact tie resolves to the bin
    whose centroid is nearer centre (declared, amendment 1)."""
    cands = [b for b in bins if b[key] > 0.0]
    return max(cands, key=lambda b: (b[key], -abs(b["centroid"]))) if cands else None


def _nearest_grid(x: float, grid: list[float]) -> float:
    return min(grid, key=lambda g: (abs(g - x), abs(g)))


def _declare_targets(per_agent: dict[str, dict], majority: int, step: float = SWEEP_STEP) -> dict:
    """The declared procedure over the gated taught seeds' sweeps → gated + exploratory targets."""
    values = _sweep_values(step)
    gated: list[float] = []
    flags: list[str] = []
    per_direction: dict[str, dict] = {}
    for direction, key, sign in (("left", "left_strength", -1.0), ("right", "right_strength", +1.0)):
        votes: dict[float, int] = {v: 0 for v in values}
        seeds_with_bin = 0
        for label, res in per_agent.items():
            b = _strongest(res["bins"], key)
            if b is None:
                flags.append(f"{label}: no bin with a {direction} bias")
                continue
            seeds_with_bin += 1
            for v in b["az"]:
                votes[v] += 1
        eligible = [
            v for v in values if votes[v] >= majority and v * sign > 0.0 and abs(v) <= FRONT_HEMISPHERE_MAX + 1e-9
        ]
        info: dict = {"seeds_with_bin": seeds_with_bin, "eligible": eligible, "targets": []}
        if not eligible:
            flags.append(f"{direction}: no eligible placement (majority {majority}) — gated targets incomplete")
            per_direction[direction] = info
            continue
        centroid = sum(eligible) / len(eligible)
        t1 = _nearest_grid(centroid, eligible)
        outer = round(t1 + sign * step, 4)
        inner = round(t1 - sign * step, 4)
        if outer in eligible:
            t2 = outer
        elif inner in eligible:
            t2 = inner
            flags.append(f"{direction}: outer neighbour of {t1:+.1f} not eligible; inner {inner:+.1f} used")
        else:
            t2 = None
            flags.append(f"{direction}: a single eligible placement {t1:+.1f} — one magnitude only")
        info.update({"centroid": round(centroid, 3), "targets": [t for t in (t1, t2) if t is not None]})
        gated.extend(info["targets"])
        per_direction[direction] = info
    wrong_votes: dict[float, int] = {v: 0 for v in values}
    for res in per_agent.values():
        for r in res["rows"]:
            m = r.get("learned_margin")
            if r["az"] == 0.0:
                continue  # dead ahead has no direction — neither correct nor wrong
            if r.get("correct") is False and m is not None and abs(float(m)) > MARGIN_FLOOR:
                wrong_votes[r["az"]] += 1
    wrong_region = [v for v in values if wrong_votes[v] >= majority]
    exploratory: list[float] = []
    if wrong_region:
        exploratory.append(_nearest_grid(sum(wrong_region) / len(wrong_region), wrong_region))
    return {
        "gated_targets": sorted(set(gated)),
        "exploratory_targets": exploratory,
        "per_direction": per_direction,
        "predicted_wrong_way_region": wrong_region,
        "flags": flags,
    }


def cmd_sweep(args: argparse.Namespace) -> int:
    os.environ["MAXIM_SUBSTRATE_TOOL_WHITELIST"] = "turn_left,turn_right"
    os.environ.pop("MAXIM_PLACE_CODE_EXTEROCEPTION", None)
    manifest = _load_manifest(args.manifest)
    from maxim.simulation import sim_logger

    sink = _ProvenanceSink()
    sim_logger.register_sim_sink(sink)
    values = _sweep_values(args.step)
    per_agent: dict[str, dict] = {}
    all_agents: dict[str, dict] = {}
    try:
        for spec in manifest["agents"]:
            if spec["arm"] != "taught":
                continue
            label = spec["label"]
            rows = []
            for az in values:
                # Fresh per value: a new load of the persisted files and a new body, so
                # nothing (drift, a separated node, encoder stash) carries between values.
                agent = LoadedAgent(spec, EXPLORE_PRIMARY)
                rig = DryReadoutRig(body_ref=args.body_ref, factory_mode=args.factory)
                rig.sync_embodiment(az)
                d = decide(agent, rig, sink)
                rows.append(
                    {
                        "az": az,
                        "audio_cluster": d["audio_cluster"],
                        "completed": d["completed"],
                        "affordance": d["affordance"],
                        "correct": _correct_for(az, d["affordance"]),
                        "learned_margin": d["learned_margin"],
                        "consulted_audio": _consulted_audio(d),  # None = malformed provenance
                        "biases": _cluster_biases(agent, d["audio_cluster"]),
                    }
                )
                if not agent.files_unchanged():
                    print(f"[FAIL] {label}: persisted files changed during the sweep — S3 violation")
                    return 5
            bins = _bins_from_rows(rows)
            res = {
                "arm": spec["arm"],
                "seed": spec["seed"],
                "exploratory": _is_exploratory_agent(spec),
                "rows": rows,
                "bins": bins,
            }
            all_agents[label] = res
            if not res["exploratory"]:
                per_agent[label] = res
            print(f"  [{label}]{' (exploratory)' if res['exploratory'] else ''}")
            for b in bins:
                print(
                    f"      {b['az_min']:+.1f} … {b['az_max']:+.1f}  cluster {str(b['cluster'])[:8]}  "
                    f"L {b['left_strength']:.3f}  R {b['right_strength']:.3f}  biases {b['biases']}"
                )
    finally:
        sim_logger.unregister_sim_sink(sink)
    majority = args.majority or (len(per_agent) // 2 + 1)
    decl = _declare_targets(per_agent, majority, args.step)
    out = {
        "_format_version": "1.0",
        "experiment": manifest.get("experiment"),
        # Incomplete = a direction with fewer than two magnitudes (the procedure's
        # inner-neighbour fallback is a COMPLETE set; its flag is informative only).
        "incomplete": any(len(v.get("targets") or []) < 2 for v in decl["per_direction"].values())
        or len(decl["gated_targets"]) != 4,
        "manifest": args.manifest,
        "body_ref": args.body_ref,
        "factory": bool(args.factory),
        "step": args.step,
        "front_hemisphere_max": FRONT_HEMISPHERE_MAX,
        "margin_floor": MARGIN_FLOOR,
        "majority": majority,
        "procedure": SWEEP_PROCEDURE,
        "provenance": provenance(_HERE.parent.parent, out_path=args.out, allow_dirty=args.allow_dirty),
        **decl,
        "agents": all_agents,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[sweep] gated targets {decl['gated_targets']}  exploratory {decl['exploratory_targets']}")
    for f in decl["flags"]:
        print(f"  [flag] {f}")
    print(f"[sweep] -> {out_path}")
    return 0 if len(decl["gated_targets"]) == 4 and not out["incomplete"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--allow-dirty",
        action="store_true",
        help="write a GATED record (docs/experiments/data/) from a dirty src/scripts tree; stamps allow_dirty: true "
        "into every record (default: refuse, exit 3 — docs/lessons/experiment-prereg-precedes-data.md)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", parents=[common])
    m.add_argument("--archive", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--experiment", choices=("53", "54"), default="53")
    m.add_argument("--phase-a-records", default=None, help="Exp 54: the Phase A campaign JSONL (weakest taught seed)")
    sw = sub.add_parser(
        "sweep", parents=[common], help="Exp 54: az sweep through each taught seed's loaded EC → targets JSON"
    )
    sw.add_argument("--manifest", required=True)
    sw.add_argument("--out", required=True)
    sw.add_argument("--body-ref", default=EXP54_BODY_REF)
    sw.add_argument("--factory", action="store_true", default=True)
    sw.add_argument("--no-factory", dest="factory", action="store_false", help="dry deltas from DELTAS (Exp 53 bodies)")
    sw.add_argument("--step", type=float, default=SWEEP_STEP)
    sw.add_argument("--majority", type=int, default=None, help="seeds that must agree (default: > half)")
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--manifest", required=True)
    r.add_argument("--phase", type=int, choices=(1, 2), required=True)
    r.add_argument("--host", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument(
        "--body-ref", default=BODY_REF, help=f"body component (Exp 53: {BODY_REF}; Exp 54: {EXP54_BODY_REF})"
    )
    r.add_argument(
        "--factory",
        action="store_true",
        help="attach the orient backend through the production make_reachy_orient_factory (Exp 54; refuses --delta)",
    )
    r.add_argument("--targets", default=None, help="Exp 54: the sweep's targets JSON (gated + exploratory placements)")
    r.add_argument(
        "--gate", choices=("I", "C"), default="I", help="Phase 1 gate: I (instrument) or C (Exp 54 user path)"
    )
    r.add_argument(
        "--whitelist",
        action="store_true",
        help="keep the S6 nursery whitelist ON under --gate C (default: Phase C runs in the user's full tool space)",
    )
    r.add_argument(
        "--allow-incomplete-targets",
        action="store_true",
        help="run with a --targets file the sweep marked incomplete (fewer than two magnitudes per direction); recorded",
    )
    r.add_argument("--settle", type=float, default=1.0)
    r.add_argument("--probe-s", type=float, default=30.0)
    r.add_argument("--yes", action="store_true")
    r.add_argument("--only", nargs="*", default=None, help="agent labels to run (debugging; not a result)")
    r.add_argument(
        "--delta",
        type=float,
        default=None,
        help="body-yaw step per turn in rad (default: the frozen DELTA_RAD; Exp 53b pre-registers 0.30)",
    )
    r.add_argument(
        "--condition",
        choices=("both", "primary", "secondary"),
        default="both",
        help="Phase 2 block(s) to run — the two blocks may be run as separate invocations of one session",
    )
    v = sub.add_parser("verdict", parents=[common])
    v.add_argument("--records", required=True)
    v.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="restrict the verdict to these run_id(s); required when a file holds two complete runs for one block",
    )
    v.add_argument(
        "--gate", choices=("T", "C"), default="T", help="T = Phase 2 transfer (Exp 53); C = Exp 54 user path"
    )
    args = ap.parse_args(argv)
    return {"manifest": cmd_manifest, "run": cmd_run, "sweep": cmd_sweep, "verdict": cmd_verdict}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
