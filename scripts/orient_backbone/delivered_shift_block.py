#!/usr/bin/env python
"""H1 Part C follow-up — the ``_big`` delivered-shift block (n reps per side).

Session 2 of the H1 campaign
(docs/experiments/protocols/h1_healthy_hardware_doa_preregistration.md, Part C)
measured the normal orient arms at n=37 but fired ``turn_left_big`` /
``turn_right_big`` only ONCE each. The ``_big`` YAML magnitudes
(``self_effect: {head_yaw: ±0.9, azimuth: ±0.50}`` in ``bodies/reachy_mini.yaml``)
stay frozen until a dedicated block measures them. This is that block.

What it measures, per turn, from a CENTERED pose:

* ``commanded_delta_rad`` — the affordance's declared body-yaw delta (the YAML
  ``self_effect["head_yaw"]``, read by the production backend);
* ``achieved_delta_rad`` — the body rotation the DAEMON reports
  (``GET /api/state/full`` body_yaw, post − pre) — the ground truth the
  controller readback is checked against, never the commanded value echoed;
* ``delivered_ratio`` = achieved / commanded;
* ``d_az`` — the azimuth transition (head-relative, [-1, 1]) across the turn,
  from (a) the production backend's own frame-corrected before/after pair
  (``metadata["measured_drive_transitions"]``) and (b) this script's
  speech-gated median-of-k before/after (a second, independent estimate with a
  longer window);
* ``implied_gain`` = d_az / achieved_delta_rad (az per rad of ACTUAL rotation).

It drives the PRODUCTION affordance path — ``SpecModulator.execute`` →
``ReachyOrientMotorBackend.execute`` → ``ReachyMiniController.goto_target`` —
so the number is the one the runtime will deliver, not a raw-SDK proxy. (The
runbook's sweep/verify scripts deliberately bypass the controller; this one
deliberately does not, because the question IS the production path.)

Preflight (all fail loud, exit 3): SDK version == daemon version (skew fails
silently on sensing AND control), daemon backend ready after wake, and a
CONTINUOUS speech source — the speech-gate rate over a short probe must clear
``--min-speech-rate`` or the block refuses to start (a silent room yields no
transitions and the run would be waste).

Records are append-only JSONL, one ``run_id`` per invocation; group by
``run_id``, never by label (L9/L10). ``dry_run`` rides on every record.

Usage::

    python scripts/orient_backbone/yaw_verify.py                      # gate step 1 FIRST
    python scripts/orient_backbone/delivered_shift_block.py --dry-run  # logic check, no robot
    python scripts/orient_backbone/delivered_shift_block.py --reps 8 --log <durable path>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from live_common import JsonlLog, gated_azimuth, resolve_host  # noqa: E402

BODY_REF = "bodies/reachy_mini"
DEFAULT_AFFORDANCES = ("turn_left_big", "turn_right_big")
HEALTHY_GAIN = 0.578  # H1 full-range gain (L9-admitted), az per rad


# ── daemon-side helpers ──────────────────────────────────────────────────────


def _get_json(url: str, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001 - diagnostic path
        print(f"      (GET {url} failed: {e})")
        return None


def daemon_status(host: str) -> dict | None:
    return _get_json(f"http://{host}:8000/api/daemon/status")


def daemon_body_yaw(host: str) -> float | None:
    """The daemon's own body_yaw from /api/state/full (shape varies by version)."""
    state = _get_json(f"http://{host}:8000/api/state/full")
    if not isinstance(state, dict):
        return None
    if isinstance(state.get("body_yaw"), (int, float)):
        return float(state["body_yaw"])
    for v in state.values():
        if isinstance(v, dict) and isinstance(v.get("body_yaw"), (int, float)):
            return float(v["body_yaw"])
    return None


def installed_sdk_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("reachy-mini")
    except Exception:  # noqa: BLE001
        return None


def provenance(repo_root: Path) -> dict:
    """Which code actually ran — the harness-provenance discipline, in-process form."""
    import maxim

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=repo_root, text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--short", "src", "scripts"], cwd=repo_root, text=True).strip()
        )
    except Exception:  # noqa: BLE001
        git_hash, dirty = "unknown", True
    executed = Path(getattr(maxim, "__file__", "") or "").resolve()
    if not executed.is_relative_to((repo_root / "src").resolve()):
        # The harness-provenance lesson, in-process form: a result whose
        # code-under-test cannot be established is not a measurement.
        print(f"[FAIL] the imported maxim is {executed}, not this repo's src/ — fix PYTHONPATH (absolute) and re-run.")
        raise SystemExit(3)
    return {
        "executed_maxim_file": str(executed),
        "executed_git_hash": git_hash,
        "working_tree_dirty_src_scripts": dirty,
        "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }


# ── the rig: production controller + production backend + production feed ──


class _MaximShim:
    """The minimal ``maxim`` surface the backend reads: ``_doa_feed`` only."""

    def __init__(self, feed) -> None:
        self._doa_feed = feed


class LiveBlockRig:
    def __init__(self, host: str) -> None:
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.embodiment.spec import attach_backends
        from maxim.hardware.reachy.controller import ReachyMiniController
        from maxim.hardware.reachy.motor_backend import ReachyOrientMotorBackend, make_reachy_orient_factory
        from maxim.embodiment.audio_localization import DoAFeed

        self.host = host
        self.robot = ReachyMiniController(host=host, connection_mode="network", media_backend="no_media")
        if not self.robot.connect():
            raise RuntimeError("controller.connect() failed")
        if not self.robot.wake_up():  # enable_motors() + wake + automatic_body_yaw=False
            raise RuntimeError("controller.wake_up() failed — torque not enabled, refusing to run")
        self.reader = self.robot.get_doa_reader()
        if self.reader is None:
            raise RuntimeError("controller has no DoA reader (audio_localization opt-out?)")

        # Production body entity + production orient backend, bound the way
        # build_executor binds them (backend attaches to the raw entity, then
        # the Embodiment wrapper is handed to it).
        self.entity = ComponentRegistry().instantiate(BODY_REF)
        self.stop = threading.Event()
        # Backend first (it attaches to the raw entity, as build_executor does),
        # then the Embodiment wrapper, then the feed with that wrapper.
        self.shim = _MaximShim(None)
        attach_backends(self.entity, modulator_factory=make_reachy_orient_factory(self.robot, maxim=self.shim))
        self.embodiment = Embodiment(self.entity)
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
        self._thread = threading.Thread(target=self.feed.run, name="doa-feed", daemon=True)
        self._thread.start()

    # providers for the feed's capture-frame stamps (degrees, as the runtime wires them)
    def _pose(self) -> dict:
        return self.robot.get_current_pose() or {}

    def head_pose_deg(self) -> dict:
        """World head yaw/pitch/roll + body yaw in degrees, from the controller readback
        (D30 / the Exp 45 "mics must actually rotate" trigger — recorded per turn)."""
        p = self._pose()
        out = {}
        for k in ("yaw", "pitch", "roll", "body_yaw"):
            if k in p:
                out[f"head_{k}_deg" if k != "body_yaw" else "body_yaw_deg"] = round(math.degrees(float(p[k])), 2)
        if "yaw" in p and "body_yaw" in p:
            out["head_rel_yaw_deg"] = round(math.degrees(float(p["yaw"]) - float(p["body_yaw"])), 2)
        return out

    def _head_rel_yaw_deg(self) -> float:
        p = self._pose()
        return math.degrees(float(p.get("yaw", 0.0)) - float(p.get("body_yaw", 0.0)))

    def _body_yaw_deg(self) -> float:
        return math.degrees(float(self._pose().get("body_yaw", 0.0)))

    def recenter(self, duration: float = 2.5) -> None:
        from maxim.hardware import MotionTarget

        # One retry: the daemon occasionally drops a task-completion ack
        # ("Task did not complete in time") on a command that landed. The
        # target is ABSOLUTE, so re-issuing is idempotent (live_common._goto).
        for attempt in range(2):
            if self.robot.goto_target(MotionTarget(body_yaw=0.0, duration=duration)):
                break
            if attempt:
                raise RuntimeError("recenter goto_target rejected twice — daemon/motors need attention")
            print("      (recenter goto rejected/timed out — re-issuing once)")
            time.sleep(1.0)
        time.sleep(0.5)

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


class DryBlockRig:
    """Offline stand-in: a body that delivers ``ratio`` of the command with a
    ``HEALTHY_GAIN`` sensor, so the loop/summary logic is checked before robot time."""

    def __init__(self, ratio: float = 0.95, source_az: float = 0.20, seed: int = 1) -> None:
        import random

        self._rng = random.Random(seed)
        self.ratio = ratio
        self.body = 0.0
        self.source = source_az  # fixed source bearing, in az units at body_yaw 0
        self.deltas = {"turn_left": 0.3, "turn_right": -0.3, "turn_left_big": 0.9, "turn_right_big": -0.9}
        self.host = "dry"

    def az(self) -> float:
        # YAML convention: turning LEFT (+body_yaw) moves a fixed source's head-relative
        # bearing toward + (self_effect azimuth has the same sign as head_yaw).
        return max(-1.0, min(1.0, self.source + HEALTHY_GAIN * self.body + self._rng.gauss(0, 0.01)))

    def reader(self):
        # (doa_rad, is_speech) — inverse of doa_to_azimuth
        return (self.az() * math.pi / 2 + math.pi / 2, True)

    def recenter(self, duration: float = 0.0) -> None:
        self.body = 0.0

    def execute(self, affordance: str):
        from maxim.embodiment.sem import ModulatorResult

        before = self.az()
        target = self.body + self.deltas[affordance]
        self.body += self.deltas[affordance] * self.ratio
        after = self.az()
        return ModulatorResult(
            success=True,
            modulator_name="orient",
            entity_name="reachy_mini",
            affordance=affordance,
            params={},
            metadata={
                "commanded_body_yaw_deg": round(math.degrees(target), 1),
                "achieved_body_yaw_deg": round(math.degrees(self.body), 1),
                "reached": abs(self.body - target) <= math.radians(5.0),
                "clamped_to_body_limit": False,
                "measured_drive_transitions": {"azimuth": (before, after)},
            },
        )

    def head_pose_deg(self) -> dict:
        return {"body_yaw_deg": round(math.degrees(self.body), 2), "head_rel_yaw_deg": 0.0, "head_roll_deg": 0.0}

    def close(self) -> None:
        pass


# ── measurement ──────────────────────────────────────────────────────────────


def _sd(xs: list[float]) -> float | None:
    return round(statistics.pstdev(xs), 4) if len(xs) > 1 else None


def speech_rate_probe(reader, seconds: float, poll_s: float = 0.15) -> tuple[float, int]:
    """Fraction of reads with the speech flag set over ``seconds``."""
    hits = total = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        total += 1
        try:
            r = reader()
        except Exception:  # noqa: BLE001
            r = None
        if r is not None and r[1]:
            hits += 1
        time.sleep(poll_s)
    return (hits / total if total else 0.0), total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default=None, help="Reachy daemon IP (or $MAXIM_REACHY_HOST)")
    ap.add_argument("--reps", type=int, default=8, help="turns per affordance")
    ap.add_argument("--affordances", default=",".join(DEFAULT_AFFORDANCES))
    ap.add_argument("--reads", type=int, default=5, help="gated samples per before/after median")
    ap.add_argument("--settle", type=float, default=1.0, help="settle after recenter / after turn (s)")
    ap.add_argument("--probe-s", type=float, default=6.0, help="speech-rate preflight window (s)")
    ap.add_argument("--min-speech-rate", type=float, default=0.5, help="refuse to start below this")
    ap.add_argument("--label", default="partc_big_block")
    ap.add_argument(
        "--log", default="/tmp/delivered_shift_block.jsonl", help="append-only JSONL (put it on DURABLE storage)"
    )
    ap.add_argument("--yes", action="store_true", help="skip the source-placement confirm prompt")
    ap.add_argument("--dry-run", action="store_true", help="offline logic check (no robot)")
    args = ap.parse_args()

    affordances = [a.strip() for a in args.affordances.split(",") if a.strip()]
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    log = JsonlLog(args.log)

    def emit(event: str, **fields: object) -> None:
        log.write(event, run_id=run_id, label=args.label, dry_run=args.dry_run, **fields)

    repo_root = _HERE.parent.parent
    prov = provenance(repo_root)

    if args.dry_run:
        rig = DryBlockRig()
        reader = rig.reader
        emit("block_start", affordances=affordances, reps=args.reps, deltas=rig.deltas, provenance=prov)
        print(f"[dry-run] run_id={run_id} — no robot; verifying loop + summary logic")
    else:
        host, source = resolve_host(args.host)
        if host is None:
            print("[FAIL] no robot address: --host <ip> or export MAXIM_REACHY_HOST=<ip>")
            return 2
        print(f"[host] {host} (source: {source})")
        status = daemon_status(host) or {}
        sdk = installed_sdk_version()
        daemon_ver = status.get("version")
        print(f"[version] sdk={sdk} daemon={daemon_ver} hardware_id={status.get('hardware_id')}")
        if not sdk or not daemon_ver or sdk != daemon_ver:
            print("[FAIL] SDK/daemon version skew — refusing: skew fails SILENTLY on sensing and control.")
            return 3
        if not args.yes:
            print("\n  SOURCE: continuous speech (podcast/radio) at a FIXED bearing, roughly 30-60° to one")
            print("  side, ~1 m from the head, loud enough that the speech gate fires most of the time.")
            print("  The robot will turn ±52° from center on every rep; keep the source still.")
            if input("  ready? [y/N] ").strip().lower() != "y":
                return 1
        rig = LiveBlockRig(host)
        reader = rig.reader
        status2 = daemon_status(host) or {}
        backend_ready = (status2.get("backend_status") or {}).get("ready")
        mode = (status2.get("backend_status") or {}).get("motor_control_mode")
        print(f"[daemon] backend ready={backend_ready} motor_control_mode={mode}")
        # Torque is the gate that matters: with motor_control_mode != enabled every
        # goto is silently ignored while reads keep working. `ready` is logged but
        # not gated on — 2026-08-24 it read False on a daemon that moved the base
        # through yaw_verify minutes earlier; the first recenter below is the
        # actuation check (daemon body_yaw is read back before every turn).
        if mode != "enabled":
            print(
                f"[FAIL] motor_control_mode={mode!r} (torque off) — wake_up() did not enable motors; refusing to run."
            )
            rig.close()
            return 3
        rate, n = speech_rate_probe(reader, args.probe_s)
        print(f"[audio] speech-gate rate over {args.probe_s:.0f}s: {rate:.2f} ({n} reads)")
        if rate < args.min_speech_rate:
            print(f"[FAIL] speech rate {rate:.2f} < {args.min_speech_rate} — start the continuous source and re-run.")
            rig.close()
            return 3
        emit(
            "block_start",
            affordances=affordances,
            reps=args.reps,
            deltas=rig.deltas,
            host=host,
            sdk_version=sdk,
            daemon_version=daemon_ver,
            hardware_id=status.get("hardware_id"),
            speech_rate_probe=round(rate, 3),
            provenance=prov,
        )
        print(f"[start] run_id={run_id} log={args.log}")

    def gated(k: int) -> float | None:
        return gated_azimuth(reader, k=k, timeout_s=8.0, poll_s=0.15)

    results: dict[str, list[dict]] = {a: [] for a in affordances}
    try:
        # Alternate sides so any slow drift (thermal, source creep) spreads evenly.
        order = [a for _ in range(args.reps) for a in affordances]
        for i, aff in enumerate(order):
            rig.recenter()
            time.sleep(args.settle)
            body_pre = rig.body if args.dry_run else daemon_body_yaw(rig.host)
            head_pre = rig.head_pose_deg()
            az_pre = gated(args.reads)
            t0 = time.monotonic()
            res = rig.execute(aff)
            t_turn = round(time.monotonic() - t0, 2)
            time.sleep(args.settle)
            body_post = rig.body if args.dry_run else daemon_body_yaw(rig.host)
            head_post = rig.head_pose_deg()
            az_post = gated(args.reads)
            meta = dict(getattr(res, "metadata", None) or {})
            commanded = float(rig.deltas[aff])
            achieved = (body_post - body_pre) if (body_pre is not None and body_post is not None) else None
            ratio = (achieved / commanded) if (achieved is not None and abs(commanded) > 1e-9) else None
            trans = (meta.get("measured_drive_transitions") or {}).get("azimuth")
            d_az_backend = round(trans[1] - trans[0], 4) if trans else None
            d_az_script = round(az_post - az_pre, 4) if (az_pre is not None and az_post is not None) else None
            d_az = d_az_backend if d_az_backend is not None else d_az_script
            gain = round(d_az / achieved, 4) if (d_az is not None and achieved and abs(achieved) > 1e-6) else None
            rec = {
                "i": i,
                "affordance": aff,
                "success": bool(getattr(res, "success", False)),
                "error": getattr(res, "error", None),
                "commanded_delta_rad": commanded,
                "body_pre_rad": body_pre,
                "body_post_rad": body_post,
                "achieved_delta_rad": (round(achieved, 4) if achieved is not None else None),
                "delivered_ratio": (round(ratio, 4) if ratio is not None else None),
                "az_pre": az_pre,
                "az_post": az_post,
                "d_az_script": d_az_script,
                "d_az_backend": d_az_backend,
                "d_az": d_az,
                "implied_gain": gain,
                "turn_wall_s": t_turn,
                "head_pre": head_pre,
                "head_post": head_post,
                "backend_metadata": meta,
            }
            emit("turn", **rec)
            results[aff].append(rec)
            print(
                f"  [{i + 1:2d}/{len(order)}] {aff:15s} cmd {math.degrees(commanded):+6.1f}°"
                f"  ach {('%+6.1f' % math.degrees(achieved)) if achieved is not None else '   n/a'}°"
                f"  ratio {('%.3f' % ratio) if ratio is not None else '  n/a'}"
                f"  d_az {('%+.3f' % d_az) if d_az is not None else '  n/a'}"
                f"{'' if d_az_backend is not None else ' (script est.)'}"
                f"  gain {('%.3f' % gain) if gain is not None else ' n/a'}"
                f"{'' if rec['success'] else '  !! ' + str(rec['error'])}"
            )
    except KeyboardInterrupt:
        print("\n[abort] Ctrl-C — partial block recorded; run_id is marked block_aborted")
        emit("block_aborted", reason="KeyboardInterrupt")
    except Exception as e:  # noqa: BLE001 - the record must say the run died, then re-raise
        print(f"\n[abort] {type(e).__name__}: {e} — partial block recorded; run_id is marked block_aborted")
        emit("block_aborted", reason=f"{type(e).__name__}: {e}")
        raise
    finally:
        rig.close()

    # ── summary ──────────────────────────────────────────────────────────
    print("\n[summary] per affordance (daemon-reported achieved rotation; d_az prefers the backend's measured pair)")
    summary: dict[str, dict] = {}
    for aff, recs in results.items():
        ratios = [r["delivered_ratio"] for r in recs if r["delivered_ratio"] is not None]
        d_azs = [r["d_az"] for r in recs if r["d_az"] is not None]
        gains = [r["implied_gain"] for r in recs if r["implied_gain"] is not None]
        row = {
            "n_turns": len(recs),
            "n_ratio": len(ratios),
            "delivered_ratio_mean": (round(statistics.mean(ratios), 4) if ratios else None),
            "delivered_ratio_sd": _sd(ratios),
            "n_d_az": len(d_azs),
            "d_az_mean": (round(statistics.mean(d_azs), 4) if d_azs else None),
            "d_az_sd": _sd(d_azs),
            "implied_gain_mean": (round(statistics.mean(gains), 4) if gains else None),
            "implied_gain_sd": _sd(gains),
            "yaml_azimuth_self_effect": None,
        }
        summary[aff] = row
        print(
            f"  {aff:15s} n={row['n_turns']:2d}  ratio {row['delivered_ratio_mean']} ± {row['delivered_ratio_sd']} (n={row['n_ratio']})"
            f"  d_az {row['d_az_mean']} ± {row['d_az_sd']} (n={row['n_d_az']})"
            f"  gain {row['implied_gain_mean']} ± {row['implied_gain_sd']}"
        )
    print(
        f"  reference: healthy full-range sensor gain {HEALTHY_GAIN} az/rad; YAML _big self_effect azimuth ±0.50 for ±0.9 rad"
    )
    emit("block_done", summary=summary)
    log.close()
    print(f"\n[done] run_id={run_id} → {args.log}  (group by run_id, not label)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
