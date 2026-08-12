"""RSC pre-check — does the current sensor→EC path generalise across VANTAGE?

The falsifiable gate from
[docs/plans/deferred/retrosplenial_spatial_frames.md](../docs/plans/deferred/retrosplenial_spatial_frames.md) §2.
If the existing path already binds one world-fixed source seen from several body
yaws into ONE EC node, the retrosplenial plan is unnecessary and closes. If it
fragments, the fragmentation count is the baseline any fix must beat.

Runs entirely OFFLINE — no robot, no sim, no LLM. ``azimuth`` on the Reachy body is
declared head-relative (``reachy_mini.yaml``: "head-relative azimuth of the current
target … WORLD-COUPLED: overwritten every tick by the percept"), so the egocentric
reading for a world-fixed source at world-bearing ``W`` from body yaw ``B`` is simply
``W − B`` wrapped into the sensor's ``[-1, 1]`` normalized range. That is all the
geometry needed to ask the question of the REAL encoder + REAL EC.

Two halves, mirroring the plan's two predictions:

  H-A  SAME source, DIFFERENT vantages  → should be ONE node if the path generalises.
  H-B  DIFFERENT sources, SAME reading  → should be DIFFERENT nodes if the path
                                          distinguishes places (it cannot: the
                                          reading is identical, so this measures
                                          aliasing, not a defect of EC).

Usage::

    PYTHONPATH=src python scripts/rsc_precheck.py [--source-bearing 0.5] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AZIMUTH_RANGE = (-1.0, 1.0)  # reachy_mini.yaml: normalized, -1 full left … +1 full right


def wrap_norm(x: float) -> float:
    """Wrap a normalized bearing into [-1, 1) the way a circular sensor would."""
    return (x + 1.0) % 2.0 - 1.0


def egocentric_azimuth(world_bearing: float, body_yaw: float) -> float:
    """Head-relative reading of a world-fixed source. Pure geometry, normalized units."""
    return wrap_norm(world_bearing - body_yaw)


def _fresh_encoder():
    """A real LinguisticEncoder over a fresh EC — production encode path."""
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.similarity.encoder import SensorEncoder

    ec = EntorhinalCortex()
    return SensorEncoder(ec=ec), ec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-bearing", type=float, default=0.5, help="world bearing of the fixed source")
    ap.add_argument(
        "--body-yaws",
        default="-0.5,-0.25,0.0,0.25,0.5",
        help="comma-separated body yaws (normalized units) to view the source from",
    )
    ap.add_argument("--json", default="", help="write the report as JSON")
    args = ap.parse_args()

    yaws = [float(y) for y in args.body_yaws.split(",") if y.strip()]
    report: dict[str, object] = {"source_bearing": args.source_bearing, "body_yaws": yaws}

    # ── H-A: one world-fixed source, many vantages ──────────────────────────
    enc, _ec = _fresh_encoder()
    rows = []
    for yaw in yaws:
        az = egocentric_azimuth(args.source_bearing, yaw)
        node = enc.encode_sensors(
            agent_id="rsc_precheck",
            sensors={"azimuth": az},
            modality="sensor",
            ranges={"azimuth": AZIMUTH_RANGE},
        )
        rows.append({"body_yaw": yaw, "azimuth": round(az, 4), "node": node})
    nodes_a = {r["node"] for r in rows if r["node"] is not None}

    print(f"H-A  ONE world-fixed source at bearing {args.source_bearing:+.2f}, viewed from {len(yaws)} vantages")
    print(f"{'body_yaw':>10}{'egocentric az':>16}   node")
    for r in rows:
        n = r["node"]
        print(f"{r['body_yaw']:>10.2f}{r['azimuth']:>16.3f}   {str(n)[:8] if n else 'None'}")
    print(f"\n  distinct EC nodes for ONE source: {len(nodes_a)} (ideal = 1)")

    # ── H-B: different sources that happen to give the same reading ─────────
    enc_b, _ = _fresh_encoder()
    rows_b = []
    for yaw in yaws:
        world = wrap_norm(yaw)  # a DIFFERENT source per vantage, each dead ahead
        az = egocentric_azimuth(world, yaw)  # ⇒ identically 0.0 every time
        node = enc_b.encode_sensors(
            agent_id="rsc_precheck_b",
            sensors={"azimuth": az},
            modality="sensor",
            ranges={"azimuth": AZIMUTH_RANGE},
        )
        rows_b.append({"world_bearing": round(world, 4), "azimuth": round(az, 4), "node": node})
    nodes_b = {r["node"] for r in rows_b if r["node"] is not None}
    print(f"\nH-B  {len(yaws)} DIFFERENT sources, each dead ahead (azimuth ≡ 0.0)")
    print(f"  distinct EC nodes for DIFFERENT places: {len(nodes_b)} (ideal = {len(yaws)})")

    # ── H-C: angular RESOLUTION — how many distinct nodes does the whole
    # azimuth range even produce? H-A can "pass" for the wrong reason: if the
    # encoding cannot resolve direction at all, every vantage collapses to one
    # node and that looks like generalisation when it is blindness.
    enc_c, _ = _fresh_encoder()
    sweep = [-1.0 + i * 0.1 for i in range(21)]
    seen: list[tuple[float, str | None]] = []
    for az in sweep:
        node = enc_c.encode_sensors(
            agent_id="rsc_precheck_c",
            sensors={"azimuth": round(az, 3)},
            modality="sensor",
            ranges={"azimuth": AZIMUTH_RANGE},
        )
        seen.append((round(az, 2), node))
    nodes_c = {n for _, n in seen if n is not None}
    print(f"\nH-C  angular RESOLUTION: swept azimuth {sweep[0]:+.1f}…{sweep[-1]:+.1f} in 0.1 steps")
    print(f"  distinct EC nodes across the FULL range: {len(nodes_c)} of {len(sweep)} readings")
    groups: dict[str, list[float]] = {}
    for az, n in seen:
        if n is not None:
            groups.setdefault(n, []).append(az)
    for n, azs in list(groups.items())[:6]:
        print(f"    node {n[:8]}: az {min(azs):+.1f} … {max(azs):+.1f}  ({len(azs)} readings)")

    report["h_a"] = {"rows": rows, "distinct_nodes": len(nodes_a), "ideal": 1}
    report["h_b"] = {"rows": rows_b, "distinct_nodes": len(nodes_b), "ideal": len(yaws)}
    report["h_c"] = {
        "n_readings": len(sweep),
        "distinct_nodes": len(nodes_c),
        "groups": {n[:8]: [min(a), max(a), len(a)] for n, a in groups.items()},
    }

    # ── Verdict ─────────────────────────────────────────────────────────────
    generalises = len(nodes_a) == 1
    distinguishes = len(nodes_b) == len(yaws)
    # Resolution is the PRECONDITION for reading H-A/H-B at all: a channel that
    # cannot resolve direction collapses every vantage into one node, which LOOKS
    # like generalisation but is blindness. Judge resolution first.
    low_res = len(nodes_c) <= 3
    print("\n=== VERDICT ===")
    if low_res:
        print(f"  RESOLUTION-BOUND: the whole azimuth range yields only {len(nodes_c)} EC node(s)")
        print("  — roughly one bit of direction. At this resolution H-A 'passing' would mean")
        print("  blindness, not binding, and H-B cannot separate places even in principle.")
        print("  CONSEQUENCE FOR THE PLAN: frame translation alone buys NOTHING here. An")
        print("  allocentric bearing encoded through this same path collapses into the same")
        print("  few buckets. Encoding resolution is the PREREQUISITE, not a follow-up.")
        verdict = "resolution_bound"
    elif generalises and distinguishes:
        print("  PLAN CLOSES: the existing path already binds one source across vantages")
        print("  AND separates distinct places. No frame translation needed.")
        verdict = "plan_closes"
    elif generalises:
        print("  Partial: binds across vantages but ALIASES distinct places.")
        verdict = "aliases_places"
    else:
        print(f"  PLAN STANDS: one world-fixed source fragments into {len(nodes_a)} EC nodes")
        print("  across vantage — nothing binds them as the same place. Baseline for any")
        print(f"  frame-translation fix: reduce {len(nodes_a)} → 1.")
        if not distinguishes:
            print(f"  AND distinct places collapse to {len(nodes_b)} node(s) — the aliasing half.")
        verdict = "plan_stands"
    print(
        f"\n  measured: H-A {len(nodes_a)} node(s) / H-B {len(nodes_b)} node(s) / "
        f"H-C {len(nodes_c)} node(s) over {len(sweep)} readings"
    )
    report["verdict"] = verdict
    print(
        "\n  NOTE: this measures the SENSOR->EC path only. It is the necessary condition —\n"
        "  if the reading itself cannot distinguish vantage, nothing downstream can."
    )

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nreport written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
