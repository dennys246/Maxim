"""Exp 44b S4 — is the substrate annotation stationary across a capture run?

Pilot finding F6 ([docs/experiments/44b_pilot.md]): cluster bias fell ~0.997 → 0.059
WITHIN a single capture despite the decay-tau hold, while the annotation's bands are
ABSOLUTE (>=0.5 "strongly rewarding", >=0.1 "mildly rewarding", else "neutral / mixed",
and tools can drop out of the top-N entirely). If that is so, early decisions receive a
strong treatment and late ones a weak or absent one — they are not the same experiment,
and a confirmatory campaign that pools them is pooling two doses.

This script answers that from captures ALREADY ON DISK. No LLM, no sims, no new runs.

It parses the annotation block out of each captured ``prompt_full``, tracks band strength
by decision index, and — when the matching re-query results are supplied — reports the
flip rate by band tier and by run half.

Usage::

    python scripts/exp44/analyze_nonstationarity.py \
        --capture ~/exp44b/pilot/arms/A_green_safe/seed1/capture.jsonl \
        [--results ~/exp44b/pilot/arms/A_green_safe/seed1/requery/<model>__<hash>__e8t0.7.jsonl] \
        [--tools green_flame_warm_self,green_flame_touch] [--json out.json]

Reads the REAL rendering contract from ``prompts/cluster_bias_annotation.py``:
``=== Substrate associations from prior experience ===`` then
``  <tool>  [<band> from prior experience]`` / ``  <tool>  [neutral / mixed]``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HEADER = "=== Substrate associations from prior experience ==="
# ``  <name padded>  [band…]`` — band text is everything up to the closing bracket;
# "from prior experience" is a suffix on every band except "neutral / mixed".
_ROW = re.compile(r"^\s{2,}(\S+)\s{2,}\[([^\]]+)\]\s*$")

# Ordinal tiers — higher = stronger positive treatment. Mirrors bias_to_band's five
# bands; anything unrecognised scores None (counted, never silently bucketed).
_TIER = {
    "strongly rewarding": 2,
    "mildly rewarding": 1,
    "neutral / mixed": 0,
    "mildly aversive": -1,
    "strongly aversive": -2,
}


def parse_annotation(prompt: str) -> dict[str, str] | None:
    """{tool_signature: band} from a prompt's annotation block, or None if absent."""
    if HEADER not in prompt:
        return None
    block = prompt.split(HEADER, 1)[1]
    out: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip():
            if out:
                break  # blank line ends the block once rows have started
            continue
        m = _ROW.match(line)
        if not m:
            break  # first non-row line ends the block
        tool, band_text = m.group(1), m.group(2).strip()
        band = band_text.replace(" from prior experience", "").strip()
        out[tool] = band
    return out or None


def band_tier(band: str | None) -> int | None:
    return _TIER.get(band) if band is not None else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="paired-prompt capture.jsonl")
    ap.add_argument("--results", default="", help="matching re-query results JSONL (optional)")
    ap.add_argument("--tools", default="", help="comma-separated tools to track (default: all seen)")
    ap.add_argument("--json", default="", help="write the full report as JSON")
    args = ap.parse_args()

    rows = []
    for line in Path(args.capture).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "prompt_full" not in r:
            continue  # capture_error row
        rows.append(r)
    if not rows:
        print("no usable capture rows", args.capture)
        return 1

    per_decision: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        ann = parse_annotation(r["prompt_full"])
        per_decision.append(
            {
                "decision_id": r.get("decision_id", idx),
                "index": idx,
                "has_annotation": ann is not None,
                "n_tools": len(ann or {}),
                "bands": ann or {},
            }
        )

    tracked = [t for t in args.tools.split(",") if t] or sorted({t for d in per_decision for t in d["bands"]})
    report_head = {
        "n_annotated": sum(1 for d in per_decision if d["has_annotation"]),
        "last_annotated_index": max((d["index"] for d in per_decision if d["has_annotation"]), default=None),
    }

    n_ann = sum(1 for d in per_decision if d["has_annotation"])
    print(f"[S4] {len(per_decision)} captured decisions; {n_ann} carry an annotation")
    print(f"     tracking {len(tracked)} tool(s)")

    # Annotation DISAPPEARANCE is the sharp form of F6. Production suppresses the
    # whole section when every bias has decayed into the neutral band
    # (compose_cluster_bias_annotation_section: "an all-neutral block adds tokens
    # without signal"), so the treatment does not merely weaken across a run — it
    # switches OFF at a point. Locate that point; decisions after it are untreated
    # and must not be pooled with treated ones as if they were the same condition.
    annotated_idx = [d["index"] for d in per_decision if d["has_annotation"]]
    if annotated_idx and len(annotated_idx) < len(per_decision):
        last_i = max(annotated_idx)
        gap_after = [d["index"] for d in per_decision if d["index"] > last_i]
        print(
            f"     ANNOTATION VANISHES: last annotated decision index {last_i}; "
            f"{len(gap_after)} later decision(s) are UNTREATED"
        )
    elif not annotated_idx:
        print("     NO decision carries an annotation (substrate absent or fully decayed)")
    print()

    # ── Per-tool trajectory: presence + mean tier, first vs second half ──────
    half = len(per_decision) // 2 or 1
    report: dict[str, Any] = {
        "capture": args.capture,
        "n_decisions": len(per_decision),
        **report_head,
        "tools": {},
    }
    print(f"{'tool':<34}{'present':>9}{'tier 1st':>10}{'tier 2nd':>10}{'drift':>8}")
    for tool in tracked:
        tiers = [(d["index"], band_tier(d["bands"].get(tool))) for d in per_decision]
        present = [i for i, t in tiers if t is not None]
        first = [t for i, t in tiers if t is not None and i < half]
        second = [t for i, t in tiers if t is not None and i >= half]
        m1 = sum(first) / len(first) if first else None
        m2 = sum(second) / len(second) if second else None
        drift = (m2 - m1) if (m1 is not None and m2 is not None) else None
        report["tools"][tool] = {
            "present_count": len(present),
            "mean_tier_first_half": m1,
            "mean_tier_second_half": m2,
            "tier_drift": drift,
            "last_present_index": max(present) if present else None,
        }
        f1 = f"{m1:+.2f}" if m1 is not None else "  n/a"
        f2 = f"{m2:+.2f}" if m2 is not None else "  n/a"
        fd = f"{drift:+.2f}" if drift is not None else "  n/a"
        print(f"{tool:<34}{len(present):>9}{f1:>10}{f2:>10}{fd:>8}")

    # ── Optional: join re-query results and report flip rate by treatment ────
    if args.results:
        res = {}
        for line in Path(args.results).read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            res[rec.get("decision_id")] = rec

        joined = [(d, res[d["decision_id"]]) for d in per_decision if d["decision_id"] in res]
        scored = [(d, r) for d, r in joined if r.get("action_full") is not None and r.get("action_ablated") is not None]
        if not scored:
            print("\n[S4] results supplied but no decision_ids joined — check the pairing")
            return 1

        def rate(subset):
            return (sum(1 for _, r in subset if r.get("flipped")) / len(subset)) if subset else float("nan")

        first_h = [(d, r) for d, r in scored if d["index"] < half]
        second_h = [(d, r) for d, r in scored if d["index"] >= half]
        print(f"\n[S4] flip rate by RUN HALF (n={len(scored)} scored)")
        print(f"     first half : {rate(first_h):.3f}  (n={len(first_h)})")
        print(f"     second half: {rate(second_h):.3f}  (n={len(second_h)})")

        # By max tier present in that decision's annotation — the treatment strength.
        buckets: dict[Any, list] = {}
        for d, r in scored:
            tiers = [band_tier(d["bands"].get(t)) for t in tracked]
            tiers = [t for t in tiers if t is not None]
            key = max(tiers) if tiers else None
            buckets.setdefault(key, []).append((d, r))
        print("\n[S4] flip rate by STRONGEST tracked band in the prompt")
        for key in sorted(buckets, key=lambda k: (k is None, k)):
            label = {
                2: "strongly rewarding",
                1: "mildly rewarding",
                0: "neutral / mixed",
                -1: "mildly aversive",
                -2: "strongly aversive",
            }.get(key, "absent")
            print(f"     {label:<20} {rate(buckets[key]):.3f}  (n={len(buckets[key])})")
        report["flip_by_half"] = {
            "first": rate(first_h),
            "second": rate(second_h),
            "n_first": len(first_h),
            "n_second": len(second_h),
        }
        report["flip_by_tier"] = {str(k): {"rate": rate(v), "n": len(v)} for k, v in buckets.items()}
        print(
            "\n     READ: a large half-to-half gap, or a monotone flip rate across tiers,"
            "\n     means the treatment is NON-STATIONARY within a run — pooling early and"
            "\n     late decisions pools two doses. Report it; do not average it away."
        )

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nreport written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
