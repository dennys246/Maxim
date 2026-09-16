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
separation-only reading here never authorizes a substrate change. (Exp 60 chunk
(ii), below, adds a RUN gate — authorization for a harness to run under a frozen
prereg, which is a different thing from a substrate build; both stay distinct in
the record: ``authorizes_build`` is always False, ``run_gate`` is its own block.)

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

Exp 60 chunk (ii) — the RUN-AUTHORIZING gate on the water classroom. The same probe,
generalized (docs/experiments/exp60_drowning_avoidance_prereg.md §Gate (ii)): ``capture
--anchor-file ~/.maxim/exp60_water_classroom.json`` reads the situations (``probe_situations``:
shore = baseline, submerged = contrast), settles each on the sensor that DEFINES it
(``probe_settle``: is_in_water, not altitude), and — because the contrast situation drowns the
bot — dives in VISITS budgeted from the apparatus check's MEASURED damage onset
(``measured.t_damage_onset_min_s`` − 3 s; refuses to dive without it), rescuing to the shore
(``probe_rescue``) and settling oxygen between visits. Labels travel in the trace provenance so
``analyze`` stays reproducible on the Slice-1 trace unchanged; record keys stay ROLE-positional
(``v_safe``/``safe_ids`` = baseline, ``v_dark``/``dark_ids`` = contrast) with a
``situation_labels`` map. Two additions in ``analyze``: an early-vs-late oxygen SUB-BIN of the
contrast samples (dive-second-0 full air vs the ≤13-bubble pain edge where Wire-4 books fear —
the bio-faithful lens's "conditioning-moment cluster == recall-moment cluster" preflight,
measured) and an explicit ``run_gate`` block (A4 cosine below threshold, fresh-EC ids distinct,
early/late same cluster). ``authorizes_build`` stays False — this authorizes a RUN of the frozen
prereg's harness, never a substrate change.

Usage
-----
    # LIVE (operator), against the standing Exp 58 classroom + bridge:
    python scripts/survival_world/l11_geometry_probe.py capture \
        --anchor-file ~/.maxim/exp58_classroom.json --rcon-password <pw> --samples 30 \
        --trace ~/.maxim/l11_geometry_trace.jsonl
    # (defaults: --bridge-port 25567 --rcon-port 25575 --username maxim)
    # Exp 60 water classroom (after exp60_water_check PASSED and stamped `measured`):
    python scripts/survival_world/l11_geometry_probe.py capture \
        --anchor-file ~/.maxim/exp60_water_classroom.json --rcon-password <pw> \
        --samples 30 --trace ~/.maxim/exp60_geometry_trace.jsonl

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
DEFAULT_LABELS = ("safe", "dark")  # Slice-1 / Exp 58 trace roles: (baseline, contrast)
Y_SETTLE_TOLERANCE = 3.0  # altitude settle band (the Exp 58 shape; clamped per body range)
# Exp 60 dive visits: budget from the apparatus check's MEASURED damage onset, minus margin.
DIVE_MARGIN_S = 3.0
EARLY_OXYGEN_MIN = 16.0  # dive-second-0 bin: full/near-full air (bridge oxygenLevel of 20)
LATE_OXYGEN_MAX = 13.0  # the oxygen drive's pain edge (set_point 20 − comfort_band 6)
DEFAULT_ANCHOR = Path.home() / ".maxim" / "exp58_classroom.json"
DIVE_SETTLE_S = 3.0  # a dive must reflect within this (exp60_water_check.IN_WATER_WITHIN_S) — else apparatus fault
RESCUE_OXYGEN_MIN = 19.0  # == exp60_water_check.RECOVER_OXYGEN_MIN (the bar the apparatus PASSED at; test-pinned)
SATURATION_REST = 10.0  # the bridge clamp == the body's declared rest (minecraft_player.yaml saturation midpoint)
STALE_STATE_S = 1.5  # == exp60_water_check.STALE_STATE_S (3x the 500 ms bridge cadence)
STALE_MAX_CONSECUTIVE = 8  # == exp60_water_check.STALE_MAX_CONSECUTIVE (~2 s of stale polls)


# ─────────────────────────────────────── shared ───────────────────────────────


def _classroom_geometry(anchor: Path = DEFAULT_ANCHOR) -> dict[str, Any]:
    try:
        return json.loads(anchor.read_text())
    except OSError as exc:  # apparatus failure, name it — never measure a guess
        raise SystemExit(
            f"[FAIL] classroom geometry not found: {anchor} — run "
            f"`setup_world.py classroom` / `water_classroom` on the live server first ({exc})"
        )


def situations_from_anchor(geom: dict[str, Any]) -> dict[str, Any]:
    """Normalize EITHER anchor shape into one situation plan (pure; unit-tested).

    Exp 58 (``anchor``/``dark``/``mid_y``): labels safe/dark, settle on altitude.
    Exp 60 (``probe_situations`` + ``probe_settle`` + ``probe_rescue`` + ``measured``):
    labels in file order (baseline first), settle on the DEFINING sensor, and a dive
    budget for every situation that names a rescue — ``None`` when the apparatus check
    has not stamped ``measured`` yet (capture REFUSES to dive on None: an unmeasured
    onset would drown the bot).
    """
    if "probe_situations" in geom:
        labels = list(geom["probe_situations"])
        if len(labels) != 2:
            raise SystemExit(f"[FAIL] anchor probe_situations must name exactly 2 situations, got {labels}")
        positions = {
            lab: {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for lab, p in geom["probe_situations"].items()
        }
        measured = geom.get("measured") or {}
        onset = measured.get("t_damage_onset_min_s")
        return {
            "labels": labels,
            "positions": positions,
            "settle": {lab: dict(rule) for lab, rule in (geom.get("probe_settle") or {}).items()},
            "rescue": dict(geom.get("probe_rescue") or {}),
            "dive_budget_s": None if onset is None else max(0.0, float(onset) - DIVE_MARGIN_S),
            "measured": measured or None,
        }
    sx, sy, sz = (float(v) for v in geom["anchor"])
    dx, dy, dz = (float(v) for v in geom["dark"])
    return {
        "labels": list(DEFAULT_LABELS),
        "positions": {"safe": {"x": sx, "y": sy, "z": sz}, "dark": {"x": dx, "y": dy, "z": dz}},
        "settle": {"safe": {"y_altitude": sy}, "dark": {"y_altitude": dy}},
        "rescue": {},
        "dive_budget_s": None,
        "measured": None,
        "mid_y": float(geom.get("mid_y", (sy + dy) / 2)),
    }


def settle_predicate(rule: dict[str, float], ranges: dict[str, Any]):
    """Predicate over vital_metrics for one situation's settle rule (pure).

    ``y_altitude`` settles within Y_SETTLE_TOLERANCE of the target CLAMPED to the body
    range (docs/wiring/sensor-range-clamps.md); a ``{"min": m}`` rule settles at ``v >= m``;
    every other sensor (the binary state-flags) must sit within 0.5 of the wanted value.
    An empty rule settles at once.
    """

    def _ok(vm: dict[str, Any]) -> bool:
        for name, want in rule.items():
            try:
                v = float(vm.get(name, float("nan")))
            except (TypeError, ValueError):
                return False
            if v != v:  # NaN: sensor absent from the snapshot
                return False
            if isinstance(want, dict):
                if v < float(want["min"]):
                    return False
            elif name == "y_altitude":
                lo, hi = ranges.get(name, (float("-inf"), float("inf")))
                if abs(v - min(max(float(want), lo), hi)) > Y_SETTLE_TOLERANCE:
                    return False
            elif abs(v - float(want)) > 0.5:
                return False
        return True

    return _ok


def early_late_bins(rows: list[dict[str, float]], ids: list[str]) -> dict[str, Any] | None:
    """Split the CONTRAST samples by oxygen into dive-second-0 vs pain-edge bins (pure).

    Returns None when the samples carry no oxygen (a non-dive trace). ``same_cluster``
    is True iff the early and late id SETS are EQUAL — the cluster fear is booked on
    (late, at the pain edge) is exactly the cluster active at re-submersion (early). A
    jitter-split early bin ({A, A'}) with late {A} is the conservative FAIL: fear booked
    on A reads 0.0 on a fresh dive that lands on A' (architecture-lens fold; the Exp 58
    Addendum-4 6/4 split is the precedent). None when a bin is empty (unmeasured, never
    "passed").
    """
    if not rows or not any("oxygen" in r for r in rows):
        return None
    early = sorted({i for r, i in zip(rows, ids) if r.get("oxygen", 0.0) >= EARLY_OXYGEN_MIN})
    late = sorted({i for r, i in zip(rows, ids) if r.get("oxygen", 99.0) <= LATE_OXYGEN_MAX})
    same = None if (not early or not late) else set(late) == set(early)
    return {
        "early_oxygen_min": EARLY_OXYGEN_MIN,
        "late_oxygen_max": LATE_OXYGEN_MAX,
        "early_ids": early,
        "late_ids": late,
        "n_early": sum(1 for r in rows if r.get("oxygen", 0.0) >= EARLY_OXYGEN_MIN),
        "n_late": sum(1 for r in rows if r.get("oxygen", 99.0) <= LATE_OXYGEN_MAX),
        "same_cluster": same,
    }


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

    anchor_path = Path(args.anchor_file).expanduser()
    plan = situations_from_anchor(_classroom_geometry(anchor_path))
    labels = plan["labels"]
    situations = plan["positions"]
    for lab in labels:
        if lab in plan["rescue"] and plan["dive_budget_s"] is None:
            raise SystemExit(
                f"[FAIL] situation {lab!r} needs a rescue but the anchor carries no measured damage "
                "onset — run exp60_water_check.py to PASS first (it stamps `measured`); an unmeasured "
                "dive budget would drown the bot."
            )

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
    # The settle rules must read sensors the RUNNING bridge actually emits (raw roster) — the
    # body carries every declared sensor at its initial value whether or not the bridge writes
    # it, so a stale bridge would make `is_in_water` a constant 0 and every dive settle refuse
    # (or, worse, a rest-state rule pass on a value nobody measured). Wait for the first raw
    # snapshot, then check the keys.
    needed = {name for rule in plan["settle"].values() for name in rule} | (
        {"oxygen", "health"} if plan["rescue"] else set()
    )
    deadline = time.monotonic() + 8.0
    while not aut.client.latest_state() and time.monotonic() < deadline:
        time.sleep(0.25)
    raw = aut.client.latest_state()
    missing = sorted(needed - set(raw or {}))
    if not raw or missing:
        pump.stop()
        raise SystemExit(
            f"[FAIL] the running bridge does not emit {missing or 'any state'} — restart `node index.js` "
            "from current main (the Exp 60 sensors shipped in #719) before capturing"
        )

    records: list[dict[str, Any]] = []
    provenance = {
        "kind": "provenance",
        "code_hash": _code_hash(),
        "anchor_file": str(anchor_path),
        "situations": labels,  # [baseline, contrast] — analyze reads the roles from here
        "classroom_anchor": {lab: [situations[lab]["x"], situations[lab]["y"], situations[lab]["z"]] for lab in labels},
        "settle": plan["settle"],
        "rescue": plan["rescue"],
        "dive_budget_s": plan["dive_budget_s"],
        "measured": plan["measured"],
        "world_ranges": ranges,
        "world_sensor_count": len(ranges),
        "samples_requested": args.samples,
        "cadence_s": args.cadence_s,
        "bridge": {"host": args.bridge_host, "port": args.bridge_port},
    }
    if "mid_y" in plan:
        provenance["classroom_anchor"]["mid_y"] = plan["mid_y"]
    records.append(provenance)
    print(f"[capture] {len(ranges)} declared world sensors; {args.samples} samples/situation; situations {labels}")

    def _settle(label: str, timeout_s: float = 20.0) -> bool:
        rule = plan["settle"].get(label, {})
        return settle_until(aut, settle_predicate(rule, ranges), timeout_s=timeout_s) is not None

    def _one_sample(label: str, visit: int, t_in: float, settled: bool) -> bool | None:
        """True = recorded; False = nothing synced yet; None = the snapshot is STALE.

        `sync_world_sensors` re-syncs the client's LAST snapshot, which persists after the
        bridge dies — a dead bridge would otherwise be recorded as 30 identical samples
        (the exp60_water_check `_sample` discipline, mirrored).
        """
        if aut.client.state_age_s() > STALE_STATE_S:
            return None
        if aut.backend.sync_world_sensors() <= 0:
            return False
        state = _read_world_states(aut.executor)
        # keep only declared world sensors (production encodes exactly these)
        state = {k: float(v) for k, v in state.items() if k in ranges}
        records.append(
            {
                "kind": "sample",
                "situation": label,
                "visit": visit,
                "t_in_situation": round(t_in, 3),
                "settled": settled,
                "state": state,
            }
        )
        return True

    def _heal() -> None:
        # Satiate as well as heal (confounding-lens SF-3, and what exp60_water_check's rescue does): the
        # bot must sit at its RESTING interoceptive state between visits, or a drifting
        # food/saturation carries constant mass into every sample.
        rcon.command(f"effect give {args.username} minecraft:instant_health 1 10 true")
        rcon.command(f"effect give {args.username} minecraft:saturation 1 10 true")

    def _refuse(msg: str, rescue_to: str | None) -> None:
        # Apparatus fault, not data: rescue FIRST (never leave the bot underwater), write NO
        # trace (a partial trace must never be analyzable), then refuse.
        if rescue_to is not None:
            try:
                rcon.teleport(args.username, situations[rescue_to])
            except Exception as exc:  # the refusal below is the primary error; name this one
                print(f"[capture] WARNING: rescue teleport raised during refusal: {exc!r}")
        raise SystemExit(f"[FAIL] {msg} — no trace written")

    # Order is pre-registered baseline-first then contrast (first-touch frozen-centroid
    # allocation is order-sensitive; analyze reports it, never hides it).
    try:
        # Satiate + heal BEFORE the baseline visit, for BOTH anchor shapes: the first live gate
        # run started satiated only because the apparatus check had just run (an accident of
        # ordering); a fresh/respawned bot carries the game's default saturation 5 = a constant
        # off-rest value in every sample (executor-lens fold).
        _heal()
        for label in labels:
            pos = situations[label]
            rescue_to = plan["rescue"].get(label)
            budget = plan["dive_budget_s"]
            n = 0
            visit = 0
            while n < args.samples:
                n_at_start = n
                rcon.teleport(args.username, pos)
                t0 = time.monotonic()  # the check's frame: t from teleport (settle time counts)
                if rescue_to is not None:
                    # A dive settle must confirm FAST (the check's IN_WATER_WITHIN_S) and inside the
                    # budget; a miss is an apparatus fault (drained head cell, stalled bridge) — refuse,
                    # never "sample anyway" underwater past the measured damage onset.
                    if not _settle(label, timeout_s=min(DIVE_SETTLE_S, budget)):
                        _refuse(
                            f"{label!r} settle rule {plan['settle'].get(label, {})} did not confirm within {min(DIVE_SETTLE_S, budget):.1f}s",
                            rescue_to,
                        )
                    settled = True
                else:
                    settled = _settle(label)
                    if not settled:
                        print(
                            f"[capture] WARN settle at {label} (rule {plan['settle'].get(label, {})}) did not confirm — sampling anyway, recorded UNSETTLED"
                        )
                health0 = None
                stale = 0
                while n < args.samples:
                    t_in = time.monotonic() - t0
                    if rescue_to is not None and t_in >= budget:
                        break  # budget exhausted: rescue, then another visit
                    got = _one_sample(label, visit, t_in, settled)
                    if got is None:
                        stale += 1
                        if stale >= STALE_MAX_CONSECUTIVE:
                            _refuse(
                                f"bridge stopped delivering fresh state at {label!r} ({stale} stale polls)", rescue_to
                            )
                    elif got:
                        stale = 0
                        n += 1
                        h = records[-1]["state"].get("health")
                        if rescue_to is not None and h is not None:
                            health0 = h if health0 is None else health0
                            if h < health0:
                                records[-1]["health_drop"] = True
                                print(
                                    f"[capture] {label}: health dropped ({health0}->{h}) inside the budget — rescuing early"
                                )
                                break
                    time.sleep(args.cadence_s)
                if rescue_to is None:
                    break
                rcon.teleport(args.username, situations[rescue_to])
                rule = dict(plan["settle"].get(rescue_to, {}))
                # a real breath between visits (the latch clears on OBSERVED recovery); the bar is
                # the apparatus check's own RECOVER bar, never stricter than what it PASSED at
                rule.setdefault("oxygen", {"min": RESCUE_OXYGEN_MIN})
                # the satiate is OBSERVED, not assumed: sensed saturation back at the bridge clamp
                # (= the body's declared rest) before the next visit
                rule.setdefault("saturation", {"min": SATURATION_REST})
                if settle_until(aut, settle_predicate(rule, ranges), timeout_s=20.0) is None:
                    _refuse(f"rescue to {rescue_to!r} did not restore the rest state (rule {rule})", None)
                _heal()
                visit += 1
                if n == n_at_start:
                    _refuse(f"{label!r} visit {visit} produced zero samples — the budget/settle cannot progress", None)
            visits = visit if rescue_to is not None else 1
            print(f"[capture] {label}: recorded {n} samples over {visits} visit(s)")
    finally:
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
    raw: dict[str, list[dict[str, float]]] = {}
    for r in recs:
        if r.get("kind") == "sample":
            raw.setdefault(r["situation"], []).append({k: float(v) for k, v in r["state"].items()})
    base_label, contrast_label = list(prov.get("situations")) if prov.get("situations") else list(DEFAULT_LABELS)
    for label in (base_label, contrast_label):
        if not raw.get(label):
            raise SystemExit(f"[FAIL] trace has no '{label}' samples (situations {[base_label, contrast_label]})")
    # ROLE-positional keys: "safe" = baseline, "dark" = contrast — the Slice-1 record
    # shape is a cited instrument; labels ride alongside in `situation_labels`.
    samples: dict[str, list[dict[str, float]]] = {"safe": raw[base_label], "dark": raw[contrast_label]}
    situation_labels = {"safe": base_label, "dark": contrast_label}
    is_dive = bool(prov.get("rescue"))
    # A visit that never confirmed its situation is recorded, never hidden: on a dive
    # trace it FAILS the run gate (a gate computed on unconfirmed samples is the
    # silent-failure shape; architecture-lens fold).
    unsettled = sorted({r["situation"] for r in recs if r.get("kind") == "sample" and r.get("settled") is False})

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
            # per-sample contrast ids (sample order) for the early/late sub-bin
            "_dark_ids_by_sample": ids["dark"],
        }

    clusters = _cluster_ids()
    dark_ids_by_sample = clusters.pop("_dark_ids_by_sample")
    # Exp 60 chunk (ii): conditioning-moment (late, pain edge) vs recall-moment (early,
    # dive-second-0) cluster — only meaningful on a dive trace.
    sub_bins = early_late_bins(samples["dark"], dark_ids_by_sample) if is_dive else None
    run_gate = {
        "applies_to": "situation-fear RUN authorization on this apparatus (Exp 60 chunk ii) — never a substrate build",
        "cos_a4_below_threshold": cos_a4 < PATTERN_THRESHOLD,
        "fresh_ec_ids_distinct": bool(clusters["distinct"]),
        "early_late_same_cluster": None if sub_bins is None else sub_bins["same_cluster"],
        "is_dive_trace": is_dive,
    }
    run_gate["unsettled_situations"] = unsettled
    run_gate["pass"] = bool(
        is_dive
        and not unsettled
        and run_gate["cos_a4_below_threshold"]
        and run_gate["fresh_ec_ids_distinct"]
        and run_gate["early_late_same_cluster"] is True  # None (unmeasured bin) never passes
    )
    run_gate["necessary_not_sufficient"] = (
        "offline fresh-EC replay of live-captured vectors; chunk (iii)'s harness MUST still refuse on "
        "its own LIVE cluster-distinct preflight (the live agent's EC, exp58_run pattern) before trial 1"
    )

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

    if is_dive:
        # Gate-phrased readings: the Exp 58/Slice-2 text below would contradict the run gate.
        bl, cl = base_label, contrast_label
        reading = {
            "absent": f"No world sensor moves between {bl} and {cl}: the cue is not sensed — apparatus fault.",
            "gain_silenced": f"{bl}/{cl} movers rest near the A4 neutral — the cue is not a neutral→extreme swing.",
            "diluted_present": f"{bl}/{cl} movers carry mass but share a cluster id — diluted; run gate FAILS.",
            "separable_here": f"{bl}/{cl} separate on the offline fresh-EC replay; the run_gate block decides "
            "(with the early/late sub-bin), authorizes_build stays False, and chunk (iii)'s live preflight is "
            "still required.",
        }[verdict]

    record = {
        "_format_version": "1.0",
        "kind": "l11_geometry_diagnosis",
        "slice": 1,
        "experiment": "exp60_chunk_ii_run_gate" if is_dive else "l11_slice1",
        "authorizes_build": False,
        "code_hash": prov.get("code_hash", _code_hash()),
        "situation_labels": situation_labels,
        "provenance": {
            "anchor_file": prov.get("anchor_file"),
            "classroom_anchor": prov.get("classroom_anchor"),
            "dive_budget_s": prov.get("dive_budget_s"),
            "measured": prov.get("measured"),
            "world_sensor_count": len(ranges),
            "samples": {k: len(v) for k, v in samples.items()},  # role-keyed (safe=baseline, dark=contrast)
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
        "contrast_early_vs_late_oxygen": sub_bins,
        "run_gate": run_gate,
        "per_sensor": per_sensor,
        "movers": [s["sensor"] for s in moved],
        "moved_but_silenced": [s["sensor"] for s in silenced],
        "live_contributors": [s["sensor"] for s in live_contrib],
        "verdict": verdict,
        "reading": reading,
    }

    # Human summary
    bl, cl = base_label, contrast_label
    print("\n=== L11 geometry diagnosis (Slice 1 — authorizes NO build) ===")
    print(
        f"world sensors: {len(ranges)}  gain: {'A4 p=%s' % p if gained else 'OFF'}  "
        f"samples: {bl}={len(samples['safe'])} {cl}={len(samples['dark'])}"
    )
    print(f"cos({bl},{cl})  A4={cos_a4:.4f}  A0={cos_a0:.4f}  (threshold {PATTERN_THRESHOLD})")
    print(
        f"offline fresh-EC cluster ids: {bl}={clusters['safe_ids']} {cl}={clusters['dark_ids']} "
        f"distinct={clusters['distinct']}"
    )
    if sub_bins is not None:
        print(
            f"{cl} early(oxygen>={EARLY_OXYGEN_MIN:.0f}) ids={sub_bins['early_ids']} n={sub_bins['n_early']}  "
            f"late(oxygen<={LATE_OXYGEN_MAX:.0f}) ids={sub_bins['late_ids']} n={sub_bins['n_late']}  "
            f"same_cluster={sub_bins['same_cluster']}"
        )
    print(f"RUN GATE (Exp 60 chunk ii): {'PASS' if run_gate['pass'] else 'no'}  {run_gate}")
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
        # Stamp code provenance into the decision record (plan discipline:
        # "provenance-stamped decision record") AND satisfy the gated-record
        # preflight — the diagnosis names docs/experiments/data/. Diagnostic,
        # not gated behavioural data, so --allow-dirty is offered (stamped).
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        import maxim  # noqa: PLC0415  (deferred: keep _provenance the maxim-free surface)
        from _provenance import in_process_code_provenance  # noqa: PLC0415

        record["provenance"]["code"] = in_process_code_provenance(
            _REPO_ROOT, maxim.__file__, out_path=out, allow_dirty=args.allow_dirty
        )
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
    cap.add_argument(
        "--anchor-file",
        required=True,
        help=f"classroom anchor record: the Exp 58 shape ({DEFAULT_ANCHOR}) or the Exp 60 `probe_situations` shape "
        "(~/.maxim/exp60_water_classroom.json, with the water check's measured dive budget). Required so a pool "
        "probe can never silently probe the cave.",
    )
    cap.add_argument("--trace", required=True, help="output JSONL trace path")
    cap.set_defaults(func=capture)

    an = sub.add_parser("analyze", help="OFFLINE: diagnose geometry from a captured trace")
    an.add_argument("--trace", required=True)
    an.add_argument("--json", default=None, help="write the diagnosis record here")
    an.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dirty tree when writing the record (stamps allow_dirty; diagnostic only)",
    )
    an.set_defaults(func=analyze)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
