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

from live_3_learn import LEGACY_PLACEMENT_RANGES, load_orient_actions, probe_policy
from live_common import load_policy_meta, save_policy_meta

if os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None) is not None:
    print("[env] MAXIM_NAC_REWARD_BIAS_DISABLED was set — cleared for this run (ablation flag does not apply here)")
from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.hivemind.merge import NAC_KEY_SEP, nac_merge  # noqa: E402


def load_nac_state(path: str) -> dict:
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return json.load(f)


def gauntlet(state: dict, names: list[str], action_deltas: dict[str, float], label: str, **probe_kw) -> dict:
    """Probe a NAc state dict's orient policy; print the per-bin report."""
    nac = NAc(NACConfig())
    nac.load_state(state)
    p = probe_policy(nac, names, action_deltas, **probe_kw)
    print(f"\n[{label}] probe correctness = {p['correctness']:.2f} magnitude = {p['magnitude_appropriateness']:.2f}")
    for b, v in p["bins"].items():
        print(
            f"    {b:<11} argmax={str(v['argmax']):<15} dir={v['correct']!s:<5} mag={v['magnitude_ok']!s:<5} "
            f"biases={v['biases']}"
        )
    return p


def verdict(p_merged: dict, p_left: dict, p_right: dict, threshold: float) -> tuple[bool, str]:
    """The promotion gate, on BOTH axes.

    D62: this gate read ``correctness`` only, so ``nac_merge`` replaced by
    ``return right`` PASSED — argmax-wrong in both far bins (magnitude 0.50)
    but direction-correct everywhere, and direction was all the gate looked
    at. ``magnitude_appropriateness`` was computed and PRINTED on every run
    and never consulted. Gating it is most of the fix.

    It is not all of it: when a parent is already perfect on both axes,
    ``return left`` passes too, and no threshold can change that — the
    merged policy and that parent are argmax-identical. Use
    ``--assert-noop-fails`` to find out whether the gate has any
    discriminating power on the inputs you gave it.
    """
    eps = 1e-9
    c, m = p_merged["correctness"], p_merged["magnitude_appropriateness"]
    c_best = max(p_left["correctness"], p_right["correctness"])
    m_best = max(p_left["magnitude_appropriateness"], p_right["magnitude_appropriateness"])
    if c < threshold:
        return False, f"correctness {c:.2f} < threshold {threshold:.2f}"
    if c < c_best - eps:
        return False, f"correctness {c:.2f} below the best parent's {c_best:.2f}"
    if m < m_best - eps:
        return False, f"magnitude {m:.2f} below the best parent's {m_best:.2f}"
    return True, f"correctness {c:.2f}, magnitude {m:.2f}, both >= the best parent"


# The no-op stubs `--assert-noop-fails` substitutes for `nac_merge`. Each is a
# function a broken merge could plausibly degenerate into; a gate that cannot
# tell them from a real merge is not measuring the merge.
NOOP_MERGES: dict[str, object] = {
    "return left": lambda left, right, **kw: dict(left),
    "return right": lambda left, right, **kw: dict(right),
    "return empty": lambda left, right, **kw: {},
    # The stub that matters most, and the one the first D62 pass omitted: a
    # plain dict update reproduces a DISJOINT fold exactly, so a gauntlet that
    # cannot tell it from `nac_merge` is watching the key union, not the fold.
    "naive dict update": lambda left, right, **kw: {
        **left,
        **right,
        "cluster_reward_bias": {
            **(left.get("cluster_reward_bias") or {}),
            **(right.get("cluster_reward_bias") or {}),
        },
    },
}


def split_complementary(state: dict, bins: list[str]) -> tuple[dict, dict]:
    """Split ONE policy into two parents that each know half the bins.

    D62's root cause is not the gate's thresholds — it is the INPUTS. Both
    recorded parents already probe correctness 1.00, so the
    ``merged >= max(parents)`` clause is evaluated at ceiling and carries zero
    information, and any stub returning a whole parent passes. A gauntlet run
    on two already-correct policies cannot observe a merge.

    This derives inputs the gauntlet CAN observe: parent A keeps only the
    left-hand bins' biases, parent B only the right-hand ones. Neither alone
    scores above ~0.50 correctness; only a fold that actually unions them
    recovers 1.00. That restores the row's `Re-run on: nac_merge semantics
    change` trigger, which today cannot fire.

    It tests the FOLD, not the ALIGNMENT — both halves keep their original
    `agent_id` and bin names, so keys match by construction and D43 is out of
    scope here by design (a real cross-agent arm is `test_d44_merge_
    behavioural_delta.py`, which needs D43's aligned path). This is derived
    from recorded data, not a new measurement, and earns no behavioural claim.
    """
    half = len(bins) // 2
    left_bins, right_bins = set(bins[:half]), set(bins[half:])

    def keep(which: set[str]) -> dict:
        """This half's bins keep their learned value; the others go to 0.0.

        The halves OVERLAP on every key, and that is the point. A disjoint
        split — each half holding only its own bins — makes the union the whole
        policy, so a plain ``{**left, **right}`` reproduces ``nac_merge``'s
        output BIT-IDENTICALLY (measured), and the mean-fold on colliding keys
        — the semantics the module docstring advertises and the
        ``Re-run on: nac_merge semantics change`` trigger is supposed to watch
        — is never exercised at all. That was this guard's REMAINING vacuity
        after the first D62 pass: it caught a merge that returns a parent, and
        not one that ignores the fold.

        With overlap, a colliding key must be MEAN-folded to recover the
        argmax: 0.0 against a learned +1.0 averages to +0.5 and still wins,
        while a dict update lets the other half's 0.0 clobber it outright.
        """
        out = dict(state)
        for field in ("cluster_reward_bias", "cluster_reward_source"):
            src = state.get(field)
            if not isinstance(src, dict):
                continue
            if field == "cluster_reward_source":
                out[field] = dict(src)
                continue
            out[field] = {
                k: (v if (k.split(NAC_KEY_SEP)[1:2] and k.split(NAC_KEY_SEP)[1] in which) else 0.0)
                for k, v in src.items()
            }
        return out

    return keep(left_bins), keep(right_bins)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="first trained NAc json (e.g. s1b)")
    ap.add_argument("--right", required=True, help="second independently-trained NAc json")
    ap.add_argument("--left-source", default="contributor-left", help="provenance tag for --left")
    ap.add_argument("--right-source", default="contributor-right", help="provenance tag for --right")
    ap.add_argument("--gauntlet-threshold", type=float, default=1.0, help="min merged probe correctness to PASS")
    ap.add_argument(
        "--assert-noop-fails",
        action="store_true",
        help="guard on the guard (D62): re-run the gauntlet with nac_merge replaced by "
        "each no-op stub and exit non-zero if ANY of them still PASSES. A gate that "
        "cannot distinguish a real merge from `return left` is not measuring the merge.",
    )
    ap.add_argument(
        "--complementary-split",
        action="store_true",
        help="derive the two parents by splitting --left's bins in half instead of using "
        "--right (D62). Neither derived parent is correct alone, so the gauntlet can "
        "observe the fold; --right is then ignored.",
    )
    ap.add_argument("--save-merged", default=None, help="write the merged NAc here (loadable by live_3_learn)")
    ap.add_argument("--bundle", default=None, help="export the merged NAc as a substrate bundle zip")
    ap.add_argument("--contributor-id", default=None, help="bundle manifest contributor (required with --bundle)")
    ap.add_argument("--domain", default="robotics-orient", help="bundle substrate-domain tag")
    ap.add_argument(
        "--body-ref",
        default=None,
        help=(
            "gate 7: the body the merged policy's keys were learned on (e.g. 'reachy_mini'). "
            "Omitted -> the bundle ships body_ref null and is REFUSED by body-checking receivers."
        ),
    )
    args = ap.parse_args()

    action_deltas, _band = load_orient_actions()
    names = list(action_deltas)

    left = load_nac_state(args.left)
    right = load_nac_state(args.right)
    if args.complementary_split:
        bins = sorted(
            {k.split(NAC_KEY_SEP)[1] for k in (left.get("cluster_reward_bias") or {}) if k.count(NAC_KEY_SEP) == 2}
        )
        if len(bins) < 2:
            print(f"[FAIL] --complementary-split needs >=2 bins in --left, found {len(bins)}: {bins}")
            return 2
        left, right = split_complementary(left, bins)
        args.left_source = f"{args.left_source}-half1"
        args.right_source = f"{args.right_source}-half2"
        print(f"[split] derived two complementary parents from --left over bins {bins}")
        print("        neither half is correct alone — only a real fold recovers the whole policy")

    # A bin NAME is identical across state spaces, so parents that learned at
    # different boundaries CANNOT be merged — their `near_left` are different
    # regions and the mismatch is SILENT. Refuse rather than produce nonsense.
    # Under --complementary-split both parents derive from --left, so --right's
    # sidecar describes a file that is no longer an input: it could abort on a
    # bin_boundary "disagreement" that no longer exists, or — worse now that
    # D62 promoted `magnitude_appropriateness` to a GATE — fall back to
    # LEGACY_PLACEMENT_RANGES and produce a wrong PASS/FAIL.
    _meta_right = args.left if args.complementary_split else args.right
    ml, mr = load_policy_meta(args.left), load_policy_meta(_meta_right)
    merged_meta: dict | None = None
    probe_kw: dict = {}
    if ml is not None and mr is not None:
        for key in ("bin_boundary", "band", "action_deltas"):
            if ml.get(key) != mr.get(key):
                print(f"[FAIL] parents disagree on '{key}': {ml.get(key)!r} vs {mr.get(key)!r}")
                print("       Their bin names mean different things — this is a category")
                print("       error, not a merge. Retrain one to match the other.")
                return 2
        merged_meta = dict(ml)
        merged_meta["session"] = f"merge({args.left_source}+{args.right_source})"
        merged_meta.pop("backfilled", None)
        probe_kw = {
            "gain": float(merged_meta.get("gain", 0.55)),
            "ranges": {k: tuple(v) for k, v in merged_meta.get("placements", {}).items()} or LEGACY_PLACEMENT_RANGES,
        }
        print(f"[meta] parents agree: boundary |az|={merged_meta.get('bin_boundary')}, gain {merged_meta.get('gain')}")
    else:
        missing = [n for n, m in ((args.left, ml), (args.right, mr)) if m is None]
        print(f"[meta] no sidecar for: {', '.join(missing)} — the merge will carry none, so")
        print("       consumers must assume the legacy boundary. The printed `magnitude` is")
        print("       graded in the legacy state space and may be meaningless. Retrain to fix.")

    p_left = gauntlet(left, names, action_deltas, f"left  ({args.left_source})", **probe_kw)
    p_right = gauntlet(right, names, action_deltas, f"right ({args.right_source})", **probe_kw)

    merged = nac_merge(left, right, left_source=args.left_source, right_source=args.right_source)
    p_merged = gauntlet(merged, names, action_deltas, "MERGED", **probe_kw)

    print(
        f"\n[verdict] correctness left={p_left['correctness']:.2f} right={p_right['correctness']:.2f} "
        f"merged={p_merged['correctness']:.2f} | magnitude left={p_left['magnitude_appropriateness']:.2f} "
        f"right={p_right['magnitude_appropriateness']:.2f} merged={p_merged['magnitude_appropriateness']:.2f} "
        f"(threshold {args.gauntlet_threshold:.2f})"
    )
    passed, why = verdict(p_merged, p_left, p_right, args.gauntlet_threshold)
    if passed:
        print(f"[verdict] PASS — {why}.")
    else:
        print(f"[verdict] FAIL — do NOT promote this merge (Queen-tier gate): {why}.")

    if args.assert_noop_fails:
        print("\n[noop-guard] re-running the gauntlet with nac_merge replaced by each stub.")
        print("[noop-guard] a stub that PASSES means the gate is not measuring the merge (D62).")
        survivors = []
        for name, stub in NOOP_MERGES.items():
            p_stub = gauntlet(
                stub(left, right, left_source=args.left_source, right_source=args.right_source),
                names,
                action_deltas,
                f"NOOP {name}",
                **probe_kw,
            )
            stub_passed, stub_why = verdict(p_stub, p_left, p_right, args.gauntlet_threshold)
            print(f"[noop-guard] {name:<14} -> {'PASS (BAD)' if stub_passed else 'fail (good)'}: {stub_why}")
            if stub_passed:
                survivors.append(name)
        if survivors:
            print(
                f"\n[noop-guard] VACUOUS — {len(survivors)} no-op(s) pass this gauntlet: {survivors}.\n"
                "             This arm cannot observe the merge on these inputs, so a\n"
                "             `Re-run on: nac_merge semantics change` trigger citing it CANNOT FIRE.\n"
                "             Root cause is the INPUTS, not the thresholds: a parent that is already\n"
                "             correct on both axes is, by itself, a passing policy. Re-run with\n"
                "             --complementary-split for inputs the gauntlet can actually observe."
            )
            return 3
        print("\n[noop-guard] OK — every no-op fails; the gauntlet has discriminating power here.")

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
            # Carry the state space forward. A merged policy with no sidecar is
            # precisely the bug this pipeline exists to prevent: the demo would
            # assume the legacy boundary and mis-bin a 0.33-trained merge, silently.
            if merged_meta is not None:
                save_policy_meta(out, **merged_meta)
                print(f"[saved] merged NAc + sidecar -> {out}  (boundary {merged_meta['bin_boundary']})")
            else:
                print(f"[saved] merged NAc -> {out}  (NO sidecar — parents had none)")

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
            body_ref=args.body_ref,  # gate 7 — None ships honestly unverifiable
        )
        if args.body_ref is None:
            print("[bundle] note: no --body-ref — body-checking receivers will REFUSE this bundle")
        print(f"[bundle] wrote {bundle_path}")
        print(f"[bundle] manifest: {json.dumps(dict(sorted(manifest.items())), default=str)}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
