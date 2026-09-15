#!/usr/bin/env python3
"""L11 world-channel diagnostic — SLICE 1: the live geometry probe.

Step B of the L11 approach (docs/plans/l11_world_channel_diagnostic.md); Slice 1
of that plan. Motivated by Exp 58's block: safe and dark situations would not
encode to distinct world clusters on the ~17-sensor `world` channel
(docs/wiring/cluster-dilution-blocks-situation-fear.md).

WHAT THIS ANSWERS (and what it does NOT). This probe measures the cheapest,
highest-information question first: for the real safe/dark situations, WHERE does
the contrast die — is the signal diluted, GAIN-SILENCED, or genuinely absent? The
substrate-faithful review lens predicted the discriminating sensors (`y_altitude`
Δ≈0.09, `nearest_hostile_dist`) swing only PARTIALLY and sit near the A4 gain's
neutral 0.5, where `(|v−0.5|·2)**p` crushes them to nearly nothing — i.e. the
mitigation is the muzzle. This probe tests that prediction on real vectors.

DIAGNOSIS ONLY — it NOMINATES a remedy direction and authorizes NO build. The
build gate is Slice 2's live re-encode past the exact cluster-distinct preflight
that blocked Exp 58 (per the plan, offline geometry only nominates). A
separation-only reading here never authorizes a substrate change.

Faithfulness (the whole point — don't repeat the false-confidence trap):
  * capture reads the encoder INPUT via production `_read_world_states` /
    `_read_world_ranges` (NOT the raw bridge dict, which carries drive keys and is
    unclamped), through the canonical `build_minecraft_aut`;
  * analyze replays through the SHIPPED `SensorEncoder.encode_sensors(
    modality="world")` against a real frozen-centroid `EntorhinalCortex` — never a
    mirror (the `encoding_bakeoff.py` hand-copy + stride grouping is the thing the
    review told us NOT to build on);
  * the per-sensor contribution is GAIN-WEIGHTED with the real config exponent, or
    a sensor that moved but rests near neutral is mis-attributed as carrying the
    contrast it is actually silenced on;
  * `world` is FROZEN-CENTROID (ec.py `frozen_centroid_modalities`): clustering is
    first-touch prototype allocation, not running-mean drift, so the replay stays
    frozen and reports cluster-ID assignment over the sample distribution (the live
    jitter Exp 58 Addendum 4 saw as a 6/4 id split), not a single pairwise cosine.

capture is live (operator-run: server + bridge + `setup_world.py classroom`);
analyze is pure and offline — reproducible from the captured trace, and unit-
testable without a network.

Usage
-----
    # LIVE (operator), against the standing Exp 58 classroom + bridge:
    python scripts/survival_world/l11_geometry_probe.py capture \
        --rcon-password <pw> --samples 30 \
        --trace ~/.maxim/l11_geometry_trace.jsonl
    # (defaults: --bridge-port 25567 --rcon-port 25575 --username maxim)

    # OFFLINE (anywhere), on the captured trace:
    python scripts/survival_world/l11_geometry_probe.py analyze \
        --trace ~/.maxim/l11_geometry_trace.jsonl \
        --json docs/experiments/data/l11_geometry_<date>.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# ── Diagnostic classification thresholds (DESCRIPTIVE, not build-authorizing) ──
# These label WHERE the signal is; they gate no build. Kept explicit so a reader
# can see they were fixed by intent, not tuned until the diagnosis read the way we
# hoped (the D1 tune-to-apparatus guard applies to Slice 2's real bar, but the
# spirit is honoured here too).
MOVE_EPS = 0.05  # normalized safe↔dark delta above which a sensor "moved"
GAIN_MASS_EPS = 0.05  # gain weight above which a sensor "carries mass" under A4
PATTERN_THRESHOLD = 0.85  # the world pattern-completion cosine (reported, from config)


# ─────────────────────────────────────── shared ───────────────────────────────


def _classroom_geometry() -> dict[str, Any]:
    anchor = Path.home() / ".maxim" / "exp58_classroom.json"
    try:
        return json.loads(anchor.read_text())
    except OSError as exc:  # apparatus failure, name it — never measure a guess
        raise SystemExit(
            f"[FAIL] classroom geometry not found: {anchor} — run "
            f"`setup_world.py classroom` on the live server first ({exc})"
        )


def _code_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


# ─────────────────────────────────────── capture ──────────────────────────────


def capture(args: argparse.Namespace) -> int:
    """Stage the bot at each situation and record production world-sensor reads.

    Read-only w.r.t. the substrate: teleports the live bot and reads
    `_read_world_states` — it does NOT encode into the live agent's EC (the
    cluster-ID replay is done offline in analyze on a fresh EC), so a concurrent
    measurement is never perturbed.
    """
    from exp56 import common as C  # noqa: E402  (rcon + repo root, as exp58_run)
    from maxim.runtime.agent_loop import _read_world_ranges, _read_world_states
    from maxim.simulation.minecraft_harness import MinecraftSyncPump, build_minecraft_aut
    from survival_world.common import settle_until

    geom = _classroom_geometry()
    sx, sy, sz = (float(v) for v in geom["anchor"])
    dx, dy, dz = (float(v) for v in geom["dark"])
    mid_y = float(geom["mid_y"])
    situations = {
        "safe": {"x": sx, "y": sy, "z": sz},
        "dark": {"x": dx, "y": dy, "z": dz},
    }

    rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    import tempfile

    persistence_dir = tempfile.mkdtemp(prefix="l11_geom_")
    aut = build_minecraft_aut(
        agent_id="l11_geometry_probe",
        bridge_port=args.bridge_port,
        bridge_host=args.bridge_host,
        persistence_dir=persistence_dir,
    )
    pump = MinecraftSyncPump(aut)
    pump.start()

    ranges = {k: list(v) for k, v in _read_world_ranges(aut.executor).items()}
    if not ranges:
        pump.stop()
        raise SystemExit("[FAIL] no declared world ranges — body/executor mis-wired")

    records: list[dict[str, Any]] = []
    provenance = {
        "kind": "provenance",
        "code_hash": _code_hash(),
        "classroom_anchor": {"safe": [sx, sy, sz], "dark": [dx, dy, dz], "mid_y": mid_y},
        "world_ranges": ranges,
        "world_sensor_count": len(ranges),
        "samples_requested": args.samples,
        "bridge": {"host": args.bridge_host, "port": args.bridge_port},
    }
    records.append(provenance)
    print(f"[capture] {len(ranges)} declared world sensors; {args.samples} samples/situation")

    for label, pos in situations.items():
        rcon.teleport(args.username, pos)
        target_y = pos["y"]
        # Settle on the SENSED y (clamped [0,128] per body YAML) reaching the target
        # depth band — the same settle discipline exp58_run/instrument_check use.
        clamped_y = max(0.0, min(128.0, target_y))

        def _at_depth(vm: dict[str, Any], ty: float = clamped_y) -> bool:
            try:
                return abs(float(vm.get("y_altitude", -999.0)) - ty) <= 3.0
            except (TypeError, ValueError):
                return False

        if settle_until(aut, _at_depth, timeout_s=20.0) is None:
            print(f"[capture] WARN settle at {label} did not confirm depth — sampling anyway")
        n = 0
        while n < args.samples:
            if aut.backend.sync_world_sensors() <= 0:
                time.sleep(args.cadence_s)
                continue
            state = _read_world_states(aut.executor)
            # keep only declared world sensors (production encodes exactly these)
            state = {k: float(v) for k, v in state.items() if k in ranges}
            records.append({"kind": "sample", "situation": label, "state": state})
            n += 1
            time.sleep(args.cadence_s)
        print(f"[capture] {label}: recorded {n} samples")

    pump.stop()
    out = Path(args.trace).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"[capture] wrote {len(records) - 1} samples + provenance -> {out}")
    return 0


# ─────────────────────────────────────── analyze ──────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def analyze(args: argparse.Namespace) -> int:
    """Pure offline diagnosis from a captured trace. No network, no live substrate."""
    from maxim.similarity.ec import ECConfig, EntorhinalCortex
    from maxim.similarity.encoder import (
        SensorEncoder,
        SensorEncoderConfig,
        _normalize_value,
        _sensor_embed,
    )

    recs = [json.loads(ln) for ln in Path(args.trace).expanduser().read_text().splitlines() if ln.strip()]
    prov = next((r for r in recs if r.get("kind") == "provenance"), {})
    ranges = {k: tuple(v) for k, v in prov.get("world_ranges", {}).items()}
    if not ranges:
        raise SystemExit("[FAIL] trace carries no world_ranges provenance — recapture")
    samples: dict[str, list[dict[str, float]]] = {}
    for r in recs:
        if r.get("kind") == "sample":
            samples.setdefault(r["situation"], []).append({k: float(v) for k, v in r["state"].items()})
    for label in ("safe", "dark"):
        if not samples.get(label):
            raise SystemExit(f"[FAIL] trace has no '{label}' samples")

    cfg = SensorEncoderConfig()
    gained = "world" in cfg.gain_modalities
    p = cfg.gain_exponent if gained else None

    # ── Per-situation mean RAW dict (over the declared sensor set) ──
    def _mean_raw(rows: list[dict[str, float]]) -> dict[str, float]:
        keys = set(ranges)
        return {k: sum(row.get(k, 0.0) for row in rows) / len(rows) for k in keys}

    mean = {label: _mean_raw(rows) for label, rows in samples.items()}

    # ── Gain-weighted per-sensor contribution + raw normalized delta ──
    def _norm(name: str, val: float) -> float:
        return _normalize_value(val, ranges.get(name))

    def _gain_w(v: float) -> float:
        return 1.0 if p is None else (abs(v - 0.5) * 2.0) ** p

    per_sensor = []
    for name in sorted(ranges):
        vs = _norm(name, mean["safe"][name])
        vd = _norm(name, mean["dark"][name])
        delta = abs(vd - vs)
        w_safe, w_dark = _gain_w(vs), _gain_w(vd)
        moved = delta >= MOVE_EPS
        carries = max(w_safe, w_dark) >= GAIN_MASS_EPS
        per_sensor.append(
            {
                "sensor": name,
                "v_safe": round(vs, 4),
                "v_dark": round(vd, 4),
                "norm_delta": round(delta, 4),
                "gain_w_safe": round(w_safe, 4),
                "gain_w_dark": round(w_dark, 4),
                "moved": moved,
                "carries_mass": carries,
                # the load-bearing category: moved but silenced by the A4 gain
                "moved_but_silenced": bool(moved and not carries),
            }
        )
    per_sensor.sort(key=lambda d: d["norm_delta"], reverse=True)

    # ── Mean-vector cosine, gained (A4, shipped) vs ungained (A0, control) ──
    def _cos_between(g: float | None) -> float:
        es = _sensor_embed(mean["safe"], ranges=ranges, gain_exponent=g)
        ed = _sensor_embed(mean["dark"], ranges=ranges, gain_exponent=g)
        return _cosine(es, ed)

    cos_a4 = _cos_between(p)
    cos_a0 = _cos_between(None)

    # ── Frozen-centroid cluster-ID over the sample distribution (fresh EC,
    #    register protocol — the shipped path, never the live substrate). Order is
    #    pre-registered safe-first then dark (matches the exp58 preflight) and IS
    #    first-touch order-sensitive: reported, not hidden. ──
    def _cluster_ids() -> dict[str, Any]:
        ec = EntorhinalCortex(ECConfig())
        enc = SensorEncoder(ec=ec, config=cfg)
        ids: dict[str, list[str]] = {"safe": [], "dark": []}
        for label in ("safe", "dark"):
            for row in samples[label]:
                state = {k: v for k, v in row.items() if k in ranges}
                node = enc.encode_sensors(agent_id="l11", sensors=state, modality="world", ranges=ranges)
                ids[label].append(node or "none")
        safe_set, dark_set = set(ids["safe"]), set(ids["dark"])
        return {
            "safe_ids": sorted(safe_set),
            "dark_ids": sorted(dark_set),
            "shared_ids": sorted(safe_set & dark_set),
            "distinct": len(safe_set & dark_set) == 0,
            "safe_id_count": len(safe_set),
            "dark_id_count": len(dark_set),
        }

    clusters = _cluster_ids()

    # ── Diagnosis (nominates a direction; authorizes no build) ──
    moved = [s for s in per_sensor if s["moved"]]
    silenced = [s for s in moved if s["moved_but_silenced"]]
    live_contrib = [s for s in moved if s["carries_mass"]]
    if not moved:
        verdict = "absent"
        reading = (
            "No world sensor moves ≥ MOVE_EPS between safe and dark. The contrast is not "
            "in the sensors at all — this is a world/apparatus problem, and NO substrate "
            "remedy (gain, threshold, channel-split) can manufacture a signal that isn't "
            "sensed. Slice 2 is not warranted; the classroom/body must expose the contrast."
        )
    elif silenced and not live_contrib:
        verdict = "gain_silenced"
        reading = (
            "Sensors DO move but every mover rests near the A4 neutral 0.5, where "
            "(|v−0.5|·2)**p crushes its weight below GAIN_MASS_EPS. The mitigation is the "
            "muzzle (the substrate lens's prediction). A gain-exponent sweep and stride/"
            "grouping are the WRONG direction (bake-off already scored them worse); the "
            "nominated Slice-2 arm is a per-type sub-channel isolating the depth/threat "
            "movers, or a set-point-aware neutral (deferred by encoder D1 — a C question)."
        )
    elif live_contrib and not clusters["distinct"]:
        verdict = "diluted_present"
        reading = (
            "Movers carry gain mass, yet safe/dark still share a cluster id — dilution "
            "proper: real contrast summed across many sensors stays above the "
            f"{PATTERN_THRESHOLD} threshold. Nominated Slice-2 arms: per-type channel-split "
            "(each channel back in the small-N regime) and/or a 1−k/N scaled threshold, "
            "each replayed through production code and confirmed by a LIVE re-encode."
        )
    else:
        verdict = "separable_here"
        reading = (
            "Movers carry mass AND safe/dark land in distinct cluster ids on this offline "
            "fresh-EC replay. This is NOT a build authorization — it disagrees with the "
            "Exp 58 LIVE block (cross-traffic, order, delta-gate, jitter differ), so Slice 2 "
            "must reproduce it via a live re-encode past the exact cluster-distinct preflight "
            "before anything is concluded. Report the discrepancy; do not declare victory."
        )

    record = {
        "_format_version": "1.0",
        "kind": "l11_geometry_diagnosis",
        "slice": 1,
        "authorizes_build": False,
        "code_hash": prov.get("code_hash", _code_hash()),
        "provenance": {
            "classroom_anchor": prov.get("classroom_anchor"),
            "world_sensor_count": len(ranges),
            "samples": {k: len(v) for k, v in samples.items()},
            "gain_modality": gained,
            "gain_exponent": p,
            "pattern_threshold": cfg.pattern_threshold,
        },
        "cosine": {
            "a4_gained": round(cos_a4, 4),
            "a0_ungained": round(cos_a0, 4),
            "threshold": PATTERN_THRESHOLD,
            "a4_above_threshold": cos_a4 > PATTERN_THRESHOLD,
        },
        "cluster_ids_offline_fresh_ec": clusters,
        "per_sensor": per_sensor,
        "movers": [s["sensor"] for s in moved],
        "moved_but_silenced": [s["sensor"] for s in silenced],
        "live_contributors": [s["sensor"] for s in live_contrib],
        "verdict": verdict,
        "reading": reading,
    }

    # Human summary
    print("\n=== L11 geometry diagnosis (Slice 1 — authorizes NO build) ===")
    print(
        f"world sensors: {len(ranges)}  gain: {'A4 p=%s' % p if gained else 'OFF'}  "
        f"samples: safe={len(samples['safe'])} dark={len(samples['dark'])}"
    )
    print(f"cos(safe,dark)  A4={cos_a4:.4f}  A0={cos_a0:.4f}  (threshold {PATTERN_THRESHOLD})")
    print(
        f"offline fresh-EC cluster ids: safe={clusters['safe_ids']} dark={clusters['dark_ids']} "
        f"distinct={clusters['distinct']}"
    )
    print(f"\n{'sensor':<22}{'Δnorm':>8}{'w_safe':>9}{'w_dark':>9}  flags")
    for s in per_sensor:
        flags = []
        if s["moved_but_silenced"]:
            flags.append("MOVED-BUT-SILENCED")
        elif s["moved"] and s["carries_mass"]:
            flags.append("live-contributor")
        elif not s["moved"]:
            flags.append("static")
        print(
            f"{s['sensor']:<22}{s['norm_delta']:>8.4f}{s['gain_w_safe']:>9.4f}"
            f"{s['gain_w_dark']:>9.4f}  {','.join(flags)}"
        )
    print(f"\nVERDICT: {verdict}\n{reading}\n")

    if args.json:
        out = Path(args.json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"[analyze] wrote diagnosis -> {out}")
    return 0


# ─────────────────────────────────────── cli ──────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="LIVE: stage bot at safe/dark, record world-sensor reads")
    # Defaults match survival_world/exp58_run.py + setup_world.py (one convention).
    cap.add_argument("--bridge-host", default="127.0.0.1")
    cap.add_argument("--bridge-port", type=int, default=25567)
    cap.add_argument("--rcon-host", default="127.0.0.1")
    cap.add_argument("--rcon-port", type=int, default=25575)
    cap.add_argument("--rcon-password", required=True)
    cap.add_argument("--username", default="maxim")
    cap.add_argument("--samples", type=int, default=30, help="samples per situation")
    cap.add_argument("--cadence-s", type=float, default=0.5)
    cap.add_argument("--trace", required=True, help="output JSONL trace path")
    cap.set_defaults(func=capture)

    an = sub.add_parser("analyze", help="OFFLINE: diagnose geometry from a captured trace")
    an.add_argument("--trace", required=True)
    an.add_argument("--json", default=None, help="write the diagnosis record here")
    an.set_defaults(func=analyze)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
