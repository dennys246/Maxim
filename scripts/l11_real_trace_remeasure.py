#!/usr/bin/env python3
"""L11 re-measure on REAL world traces — the retirement-grade measurement (pre-registered).

L11's retirement condition (`docs/limits/l11_sensor_dilution.md`): the A4
mitigation must be *shipped AND re-measured on a real body at a per-channel
count above the current safe band (~12)*. Shipped ✓ (1.1.4 PR 1–4);
`bodies/minecraft_player.yaml` declares **16** world sensors with RANGES
RE-CENTERED so resting values sit at the A4 neutral (the review round
measured rest-at-extreme ranges failing structurally: event cos 0.926 vs
0.747 re-centered — D1's lever, applied and annotated in the YAML). The
canonical prose pre-registration is
`docs/experiments/protocols/l11_remeasure_preregistration.md` (the lint-
governed authority); this docstring mirrors it and the code implements it.
Both merged to main BEFORE any capture.

Two phases
----------
``capture`` — a bare :class:`MinecraftClient` (no agent) against one live
bridge records snapshots at a fixed cadence plus the bridge's event stream,
each stamped ``ts`` (event records also carry the bridge-arrival wall time;
snapshot records carry ``state_age_s`` so the analyzer can report the
duplicate fraction). The FIRST snapshot must cover every declared world
sensor or the capture refuses immediately (apparatus, second one — not
after ten minutes). The world must be DYNAMIC (night + hostiles; a static
world is an apparatus failure the analyzer will refuse).

``analyze`` — replays a captured trace through the SHIPPED production path
(`SensorEncoder.encode_sensors(modality="world")` against a real
`EntorhinalCortex`, register-on-separate protocol — never a mirror), TWICE:
arm **A4** (the shipped default) and arm **A0** (`gain_modalities=
frozenset()`, the ungained control). Real-data analogs of the bake-off's
frozen trio, computed identically for both arms:

  stability       over QUIET pairs — consecutive snapshots with (a) no
                  onset-kind event within ±QUIET_GUARD_S of either and (b)
                  max normalized state delta ≤ QUIET_MAX_DELTA (un-evented
                  world drift is real change, not noise — without (b) this
                  analog would punish exactly the sensitivity A4 provides)
                  — the fraction resolving to the SAME cluster. The
                  delta-gate counts as same-cluster (it is arm-independent:
                  it keys on RAW values before any gain).
  separation      over EVENT ONSETS — for each damage/spawn/death event,
                  the first snapshot STRICTLY AFTER the event ts (equal
                  timestamps are the pre-event push: the capture loop
                  stamps the drain with the same clock as the snapshot it
                  just wrote, whose state predates the event by up to one
                  bridge interval) — deduplicated by snapshot index — the
                  fraction resolving to a DIFFERENT cluster than the
                  preceding snapshot.
  discrimination  over pairs of onsets of DIFFERENT event kinds at
                  DIFFERENT snapshot indices — the fraction landing on
                  different clusters.
  economy         clusters per 100 snapshots (a COST, reported, never
                  folded in), the same-cluster fraction, the raw-delta
                  gate-eligible fraction, and the duplicate-snapshot
                  fraction.

  PRIMARY = min(separation, stability, discrimination) — the bake-off's
  frozen weakest-link metric.

Decision rule (frozen)
----------------------
1. **Mitigation confirmed on a real body** iff A4 PRIMARY ≥ A0 PRIMARY,
   AND A4 PRIMARY > 0, AND A4 separation > 0, at N ≥ 13 encodable sensors.
   (Without the >0 clauses, two blind arms tie at 0.0 and the rule would
   print success exactly when nothing worked — the D43 shape, caught by
   the review round's demonstration.)
2. **RETIRED-eligible** iff additionally A4 PRIMARY ≥ 0.70 (the bake-off's
   next-best-arm band; synthetic A4 at N=12 scored 0.94 and real
   correlated sensors are expected to cost something).
3. **Both arms at PRIMARY 0 → verdict "refuted-blind"** — an apparatus or
   body-design refutation, ITS OWN OUTCOME, never "confirmed".
4. Anything else: "not-confirmed" — L11 stays MITIGATED with the numbers
   recorded and the next step named. No post-hoc threshold motion.

Validity requirements (S3 — explicit refusals, exit 4, never bare asserts):
≥ MIN_SNAPSHOTS snapshots; ≥ MIN_EVENTS resolved ONSETS of ≥ 2 kinds (an
event with no strictly-later snapshot resolves to no onset and counts for
nothing); > 0 quiet pairs; > 0 discrimination pairs; ≥ 13 ENCODABLE sensors
(state ∩ declared ranges) in ≥ 95% of snapshots; the declared world set and
the declared ranges must be identical sets (a range-less world sensor would
make the analyzer measure a different set than production encodes).

Usage
-----
    python scripts/l11_real_trace_remeasure.py capture --bridge-port 25567 --minutes 10 \
        --trace docs/experiments/data/l11_world_trace_<date>.jsonl
    python scripts/l11_real_trace_remeasure.py analyze \
        --trace docs/experiments/data/l11_world_trace_<date>.jsonl \
        --json docs/experiments/data/l11_remeasure_<date>.json

Operator recipe for a dynamic world: night (`time set night`), daylight
cycle off, 6+ hostiles summoned near the bot across the session, difficulty
normal. The capture bot does not act; the world acts on it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

SNAPSHOT_CADENCE_S = 0.5
QUIET_GUARD_S = 3.0
QUIET_MAX_DELTA = 0.10
MIN_SNAPSHOTS = 600
MIN_EVENTS = 8
MIN_SENSORS = 13
RETIRE_PRIMARY_BAR = 0.70
ONSET_KINDS = ("damage", "spawn", "death")


def _refuse(msg: str) -> None:
    print(f"[REFUSED — apparatus] {msg}", file=sys.stderr)
    raise SystemExit(4)


def _preflight(json_path: str, allow_dirty: bool):
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import maxim
    from _provenance import DirtyTreeError, ProvenanceError, in_process_code_provenance

    try:
        return in_process_code_provenance(_REPO_ROOT, maxim.__file__, out_path=json_path, allow_dirty=allow_dirty)
    except (ProvenanceError, DirtyTreeError) as exc:
        print(f"[FAIL] gated-record preflight: {exc}", file=sys.stderr)
        raise SystemExit(3)


def _declared_world_ranges() -> dict[str, tuple[float, float]]:
    import yaml

    from maxim.utils.paths import bundled_data

    spec = yaml.safe_load((bundled_data() / "components" / "bodies" / "minecraft_player.yaml").read_text())
    sensors = spec["entity"]["sensors"]
    world = {n for n, sd in sensors.items() if isinstance(sd, dict) and sd.get("modality") == "world"}
    ranges = {
        n: (float(sd["range"][0]), float(sd["range"][1]))
        for n, sd in sensors.items()
        if isinstance(sd, dict) and sd.get("modality") == "world" and "range" in sd
    }
    if set(ranges) != world:
        _refuse(f"declared world set != ranged set (range-less world sensors: {world - set(ranges)})")
    return ranges


def capture(args) -> int:
    provenance = _preflight(args.trace, args.allow_dirty)
    from maxim.simulation.minecraft import MinecraftClient

    declared = set(_declared_world_ranges())
    client = MinecraftClient(port=args.bridge_port)
    client.connect()
    # Apparatus check in second one, not minute ten: the first snapshot must
    # cover every declared world sensor (a drifted bridge fails HERE).
    deadline0 = time.monotonic() + 10.0
    first: dict[str, float] = {}
    while time.monotonic() < deadline0:
        first = client.latest_state()
        if first:
            break
        time.sleep(0.2)
    if not first:
        client.close()
        _refuse("no snapshot from the bridge within 10s")
    missing = declared - set(first)
    if missing:
        client.close()
        _refuse(f"bridge snapshot lacks declared world sensors: {sorted(missing)} — bridge/body drift")

    out = Path(args.trace)
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.minutes * 60.0
    n_snap = n_ev = 0
    with out.open("w") as f:
        f.write(
            json.dumps(
                {
                    "kind": "header",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "provenance": provenance,
                    "cadence_s": SNAPSHOT_CADENCE_S,
                    "bridge_port": args.bridge_port,
                    "declared_world_sensors": sorted(declared),
                }
            )
            + "\n"
        )
        while time.monotonic() < deadline:
            time.sleep(SNAPSHOT_CADENCE_S)
            state = client.latest_state()
            now = datetime.now(timezone.utc).isoformat()
            if state:
                f.write(
                    json.dumps(
                        {
                            "kind": "snapshot",
                            "ts": now,
                            "state": state,
                            "state_age_s": round(client.state_age_s(), 4),
                        }
                    )
                    + "\n"
                )
                n_snap += 1
            while True:
                event = client.pop_event()
                if event is None:
                    break
                f.write(
                    json.dumps({"kind": "event", "ts": now, "event_kind": event["kind"], "text": event["text"]}) + "\n"
                )
                n_ev += 1
            if n_snap % 120 == 0 and n_snap:
                print(f"  {n_snap} snapshots, {n_ev} events, state_age {client.state_age_s():.2f}s")
    client.close()
    print(f"captured: {n_snap} snapshots, {n_ev} events -> {out}")
    return 0


def _normalized(state: dict[str, float], ranges: dict[str, tuple[float, float]]) -> dict[str, float]:
    out = {}
    for k, (lo, hi) in ranges.items():
        if k in state and hi > lo:
            out[k] = max(0.0, min(1.0, (float(state[k]) - lo) / (hi - lo)))
    return out


def _resolve_stream(
    snaps: list[dict], ranges: dict[str, tuple[float, float]], gained: bool
) -> tuple[list[str], dict[str, Any]]:
    """Replay snapshots through the SHIPPED path; per-snapshot cluster ids."""
    from maxim.similarity.ec import ECConfig, EntorhinalCortex
    from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

    config = SensorEncoderConfig() if gained else SensorEncoderConfig(gain_modalities=frozenset())
    ec = EntorhinalCortex(ECConfig())
    encoder = SensorEncoder(ec=ec, config=config)
    ids: list[str] = []
    last_id: str | None = None
    same = 0
    gate_eligible = 0
    prev_raw: dict[str, float] | None = None
    for rec in snaps:
        state = {k: float(v) for k, v in rec["state"].items() if k in ranges}
        if prev_raw is not None:
            max_delta = max(
                (abs(state.get(k, 0.0) - prev_raw.get(k, 0.0)) for k in set(state) | set(prev_raw)), default=0.0
            )
            if max_delta < config.min_delta:
                gate_eligible += 1
        prev_raw = state
        node = encoder.encode_sensors(agent_id="l11", sensors=state, modality="world", ranges=ranges)
        if node is None:
            node = last_id or "none"
        if node == last_id:
            same += 1
        ids.append(node)
        last_id = node
    live_nodes = sum(1 for _n, (_e, m) in ec._substrate_nodes.items() if m == "world")
    n = max(1, len(ids))
    return ids, {
        "clusters": live_nodes,
        "clusters_per_100_snapshots": round(100.0 * live_nodes / n, 2),
        "same_cluster_fraction": round(same / n, 4),
        "gate_eligible_fraction": round(gate_eligible / n, 4),
    }


def analyze(args) -> int:
    provenance = _preflight(args.json, args.allow_dirty)
    ranges = _declared_world_ranges()
    records = [json.loads(line) for line in Path(args.trace).read_text().splitlines() if line.strip()]
    all_snaps = [r for r in records if r.get("kind") == "snapshot"]
    events = [r for r in records if r.get("kind") == "event" and r.get("event_kind") in ONSET_KINDS]

    # Dedupe identical consecutive snapshots (the 0.5s poll can read one
    # bridge push twice); the duplicate fraction is reported.
    snaps: list[dict] = []
    dups = 0
    for s in all_snaps:
        if snaps and s["state"] == snaps[-1]["state"]:
            dups += 1
            continue
        snaps.append(s)
    duplicate_fraction = round(dups / max(1, len(all_snaps)), 4)

    if len(snaps) < MIN_SNAPSHOTS:
        _refuse(f"{len(snaps)} deduped snapshots < {MIN_SNAPSHOTS}")
    counts = [len(set(s["state"]) & set(ranges)) for s in snaps]
    covered = sum(1 for c in counts if c >= MIN_SENSORS) / len(snaps)
    if covered < 0.95:
        _refuse(f"only {covered:.0%} of snapshots carry >= {MIN_SENSORS} ENCODABLE sensors")

    def _ts(r):
        return datetime.fromisoformat(r["ts"])

    event_times = [_ts(e) for e in events]
    onset_by_index: dict[int, set[str]] = {}
    for e in events:
        et = _ts(e)
        for i, s in enumerate(snaps):
            if _ts(s) > et:  # STRICTLY after: equal ts is the pre-event push
                onset_by_index.setdefault(i, set()).add(e["event_kind"])
                break
    onsets = sorted(onset_by_index)
    onset_kinds = set().union(*onset_by_index.values()) if onset_by_index else set()
    if len(onsets) < MIN_EVENTS or len(onset_kinds) < 2:
        _refuse(f"{len(onsets)} resolved onsets (kinds {sorted(onset_kinds)}) — need >= {MIN_EVENTS} of >= 2 kinds")

    norm = [_normalized(s["state"], ranges) for s in snaps]

    def quiet(i: int) -> bool:
        t = _ts(snaps[i])
        return all(abs((t - et).total_seconds()) > QUIET_GUARD_S for et in event_times)

    stab_pairs = []
    for i in range(len(snaps) - 1):
        if not (quiet(i) and quiet(i + 1)):
            continue
        keys = set(norm[i]) | set(norm[i + 1])
        delta = max((abs(norm[i].get(k, 0.0) - norm[i + 1].get(k, 0.0)) for k in keys), default=0.0)
        if delta <= QUIET_MAX_DELTA:
            stab_pairs.append((i, i + 1))
    if not stab_pairs:
        _refuse("0 quiet pairs — the trace has no measurable rest; capture longer or calmer")

    disc_pairs = [
        (a, b)
        for x, a in enumerate(onsets)
        for b in onsets[x + 1 :]
        if a != b and onset_by_index[a] != onset_by_index[b]
    ]
    if not disc_pairs:
        _refuse("0 discrimination pairs (different kinds at different indices) — need more varied events")

    result: dict[str, Any] = {
        "apparatus": {
            "snapshots": len(snaps),
            "duplicate_fraction": duplicate_fraction,
            "resolved_onsets": len(onsets),
            "onset_kinds": sorted(onset_kinds),
            "quiet_pairs": len(stab_pairs),
            "discrimination_pairs": len(disc_pairs),
            "n_sensors_median": sorted(counts)[len(counts) // 2],
        }
    }
    for arm, gained in (("A4", True), ("A0", False)):
        ids, econ = _resolve_stream(snaps, ranges, gained)
        stability = sum(1 for a, b in stab_pairs if ids[a] == ids[b]) / len(stab_pairs)
        separation = sum(1 for i in onsets if i > 0 and ids[i] != ids[i - 1]) / max(1, sum(1 for i in onsets if i > 0))
        discrimination = sum(1 for a, b in disc_pairs if ids[a] != ids[b]) / len(disc_pairs)
        result[arm] = {
            "stability": round(stability, 4),
            "separation": round(separation, 4),
            "discrimination": round(discrimination, 4),
            "primary_min": round(min(stability, separation, discrimination), 4),
            **econ,
        }

    a4, a0 = result["A4"]["primary_min"], result["A0"]["primary_min"]
    if a4 == 0 and a0 == 0:
        verdict = "refuted-blind"
    elif a4 >= a0 and a4 > 0 and result["A4"]["separation"] > 0:
        verdict = "retired-eligible" if a4 >= RETIRE_PRIMARY_BAR else "mitigation-confirmed"
    else:
        verdict = "not-confirmed"
    result["decision"] = {
        "rule": (
            "confirmed iff A4>=A0 primary AND A4 primary>0 AND A4 separation>0 at N>=13; "
            f"retired-eligible iff also A4>={RETIRE_PRIMARY_BAR}; both-zero => refuted-blind"
        ),
        "verdict": verdict,
    }
    print(f"  apparatus: {result['apparatus']}")
    for arm in ("A4", "A0"):
        print(f"  {arm}: {result[arm]}")
    print(f"  decision: {result['decision']}")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "harness": "scripts/l11_real_trace_remeasure.py",
                "ts": datetime.now(timezone.utc).isoformat(),
                "provenance": provenance,
                "trace": str(args.trace),
                "prereg": "docs/experiments/protocols/l11_remeasure_preregistration.md",
                "result": result,
            },
            indent=2,
        )
    )
    print(f"written: {out}")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--bridge-port", type=int, default=25567)
    c.add_argument("--minutes", type=float, default=10.0)
    c.add_argument("--trace", required=True)
    c.add_argument("--allow-dirty", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--trace", required=True)
    a.add_argument("--json", required=True)
    a.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args(argv)
    return capture(args) if args.cmd == "capture" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
