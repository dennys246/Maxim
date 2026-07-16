#!/usr/bin/env python
"""Hivemind merge arm — fleet-learning mechanics on real orient-policy NAcs.

Runbook Follow-up A (docs/plans/reachy_orient_live.md) + gauntlet #1 of the
Queen-tier trust topology (docs/plans/maxim_hivemind.md "Trust topology").
No robot needed — pure substrate operations on saved NAc files.

What it does:
  1. Loads two independently-trained orient NAcs (e.g. sessions s1b and m1,
     trained with --fresh + different seeds / days / rooms / units).
  2. Runs the frozen-policy PROBE GAUNTLET on each input AND on their
     ``hivemind.merge.nac_merge`` result (mean-merged cluster_reward_bias,
     provenance-tagged). A poisoned/flipped-calibration contribution fails
     here in milliseconds — this probe IS the Queen tier's promotion gate.
  3. Optionally persists the merged NAc (loadable by live_3_learn.py — a
     robot bootstrapped from it should probe 1.00 at trial 0), and exports
     a substrate bundle zip via ``hivemind.bundle.compose_bundle`` — the
     "queen-mind" release artifact (NAc slice only; episodes are never in
     bundles by construction, and orient NAcs contain nothing else anyway).

Usage (after two independent trainings):
    ~/Envs/maxim-env/bin/python scripts/orient_backbone/orient_merge_arm.py \\
        --left  ~/.maxim/orient_live/nac_reachy.json \\
        --right ~/.maxim/orient_live/nac_reachy_b.json \\
        --left-source dennys-reachy-01-s1b --right-source dennys-reachy-01-m1 \\
        --save-merged ~/.maxim/orient_live/nac_merged.json \\
        --bundle /tmp/queen_mind_orient_v0_1.zip \\
        --contributor-id dennys-reachy-01 --domain robotics-orient

Claim this earns when the merged probe passes: the substrate's learned
sensorimotor policy is a mergeable, distributable artifact — two robots'
(or two sessions') learning combine into one policy at least as correct as
either input, with provenance preserved. Cross-UNIT transfer (import on a
physically different robot, probe 1.00 at trial 0) is the follow-on when a
second unit exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from live_3_learn import load_orient_actions, probe_policy

if os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None) is not None:
    print("[env] MAXIM_NAC_REWARD_BIAS_DISABLED was set — cleared for this run (ablation flag does not apply here)")
from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.hivemind.merge import nac_merge  # noqa: E402


def load_nac_state(path: str) -> dict:
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return json.load(f)


def gauntlet(state: dict, names: list[str], action_signs: dict[str, float], label: str) -> float:
    """Probe a NAc state dict's orient policy; print the per-bin report."""
    nac = NAc(NACConfig())
    nac.load_state(state)
    p = probe_policy(nac, names, action_signs)
    print(f"\n[{label}] probe correctness = {p['correctness']:.2f}")
    for b, v in p["bins"].items():
        print(f"    {b:<11} argmax={str(v['argmax']):<11} correct={v['correct']}  biases={v['biases']}")
    return p["correctness"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="first trained NAc json (e.g. s1b)")
    ap.add_argument("--right", required=True, help="second independently-trained NAc json")
    ap.add_argument("--left-source", default="contributor-left", help="provenance tag for --left")
    ap.add_argument("--right-source", default="contributor-right", help="provenance tag for --right")
    ap.add_argument("--gauntlet-threshold", type=float, default=1.0, help="min merged probe correctness to PASS")
    ap.add_argument("--save-merged", default=None, help="write the merged NAc here (loadable by live_3_learn)")
    ap.add_argument("--bundle", default=None, help="export the merged NAc as a substrate bundle zip")
    ap.add_argument("--contributor-id", default=None, help="bundle manifest contributor (required with --bundle)")
    ap.add_argument("--domain", default="robotics-orient", help="bundle substrate-domain tag")
    args = ap.parse_args()

    action_signs, _band = load_orient_actions()
    names = list(action_signs)

    left = load_nac_state(args.left)
    right = load_nac_state(args.right)
    c_left = gauntlet(left, names, action_signs, f"left  ({args.left_source})")
    c_right = gauntlet(right, names, action_signs, f"right ({args.right_source})")

    merged = nac_merge(left, right, left_source=args.left_source, right_source=args.right_source)
    c_merged = gauntlet(merged, names, action_signs, "MERGED")

    print(
        f"\n[verdict] left={c_left:.2f} right={c_right:.2f} merged={c_merged:.2f} "
        f"(gauntlet threshold {args.gauntlet_threshold:.2f})"
    )
    passed = c_merged >= args.gauntlet_threshold and c_merged >= max(c_left, c_right) - 1e-9
    if passed:
        print("[verdict] PASS — merged policy is correct and at least as good as either input.")
    else:
        print("[verdict] FAIL — do NOT promote this merge (Queen-tier gate).")

    if args.save_merged:
        if not passed:
            # Persisting a do-not-promote merge invites bootstrapping from it.
            print("[saved] SKIPPED — gauntlet failed; not persisting a merge marked do-not-promote.")
        else:
            out = os.path.expanduser(args.save_merged)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            nac = NAc(NACConfig(persistence_path=out))
            nac.load_state(merged)
            nac.save(out)
            print(f"[saved] merged NAc -> {out}  (bootstrap a robot with --nac-path {out})")

    if args.bundle:
        if not passed:
            print("[bundle] REFUSED — gauntlet failed; a queen-mind bundle ships only gauntlet-passed substrate.")
            return 1
        if not args.contributor_id:
            print("[bundle] --contributor-id is required with --bundle")
            return 2
        from maxim.hivemind.bundle import compose_bundle

        bundle_path = os.path.expanduser(args.bundle)
        manifest = compose_bundle(
            nac_state=merged,
            ec_substrate_nodes=None,  # orient NAcs carry no EC state
            output_path=bundle_path,
            contributor_id=args.contributor_id,
            domain=args.domain,
        )
        print(f"[bundle] wrote {bundle_path}")
        print(f"[bundle] manifest: {json.dumps(dict(sorted(manifest.items())), default=str)}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
