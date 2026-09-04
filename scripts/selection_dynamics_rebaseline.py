#!/usr/bin/env python3
"""Selection-dynamics re-baseline for the third substrate channel (1.1.4 PR 2).

`recommend_action` sums `cluster_reward_bias` ADDITIVELY across the active
per-modality cluster set — ±1 per cluster (post-clamp) against a fixed
`min_confidence = 0.3` — so the world channel widens the summed term from
±2 to ±3. The channel registry's own comment says adding a channel "is a
selection-dynamics change; re-check gate calibration" and the roadmap row
says the addition "must be re-baselined, not assumed". This harness IS that
re-baseline: a CHARACTERIZATION of what the third channel does to selection,
against the real `NAc`, recorded before any body feeds the channel.

**This is a baseline, not a gate** — stated so the output is not over-read.
The recorded decision (plan `docs/plans/world_seam_1_1_4.md` §PR 2):
`min_confidence` stays 0.3 for 1.1.4 (infrastructure, no behavioral claim;
the channel is inert for every existing body), and THIS record is the
reference against which 1.2's calibration — measured on real Minecraft
bias distributions, which these synthetic ones are not — is compared.

**Metrics, frozen before the run.** Per seeded trial: 4 tools; per-cluster
per-tool biases drawn uniform in [−1, 1] (the post-clamp range) and written
through the real `update_cluster_reward`; the same interoception + audio
biases in both arms, the world biases added only in arm 2. Reported over
2,000 trials:

  flip_fraction        arm-2 recommendation differs from arm-1's (incl.
                       None→Some and Some→None transitions, broken out)
  gate_pass_fraction   trials where each arm recommends at all (≥ 0.3)
  cluster_term_range   observed min/max of the summed cluster term per arm

Usage
-----
    python scripts/selection_dynamics_rebaseline.py --json docs/experiments/data/selection_dynamics_rebaseline_2026-09-03.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from maxim.decisions.nac import NAc  # noqa: E402

TOOLS = ("tool_a", "tool_b", "tool_c", "tool_d")
TRIALS = 2000
MIN_CONFIDENCE = 0.3


def _teach_exact_bias(nac: NAc, agent: str, cluster: str, tool: str, bias: float) -> None:
    """One update landing exactly at `bias` (reward = bias/alpha, clamp does the rest)."""
    alpha = nac.config.reward_bias_alpha
    nac.update_cluster_reward(
        agent_id=agent,
        cluster_id=cluster,
        tool_signature=f"tool:{tool}",
        reward=bias / alpha,
        source="operant",
    )


def run(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    flips = none_to_some = some_to_none = 0
    passes = {"two": 0, "three": 0}
    term_min = {"two": 0.0, "three": 0.0}
    term_max = {"two": 0.0, "three": 0.0}

    for trial in range(TRIALS):
        agent = f"t{trial}"
        nac = NAc()
        clusters = {"interoception": "ic", "audio": "au", "world": "wo"}
        biases: dict[tuple[str, str], float] = {}
        for cl in clusters.values():
            for tool in TOOLS:
                b = rng.uniform(-1.0, 1.0)
                biases[(cl, tool)] = b
                _teach_exact_bias(nac, agent, cl, tool, b)

        for arm, active in (("two", ["interoception", "audio"]), ("three", ["interoception", "audio", "world"])):
            for tool in TOOLS:
                term = sum(biases[(clusters[m], tool)] for m in active)
                term_min[arm] = min(term_min[arm], term)
                term_max[arm] = max(term_max[arm], term)

        rec2 = nac.recommend_action(
            agent_id=agent,
            available_tools=list(TOOLS),
            current_clusters={m: clusters[m] for m in ("interoception", "audio")},
            min_confidence=MIN_CONFIDENCE,
        )
        rec3 = nac.recommend_action(
            agent_id=agent,
            available_tools=list(TOOLS),
            current_clusters=dict(clusters),
            min_confidence=MIN_CONFIDENCE,
        )
        t2 = rec2["tool_name"] if rec2 else None
        t3 = rec3["tool_name"] if rec3 else None
        passes["two"] += t2 is not None
        passes["three"] += t3 is not None
        if t2 != t3:
            flips += 1
            if t2 is None:
                none_to_some += 1
            elif t3 is None:
                some_to_none += 1

    return {
        "trials": TRIALS,
        "tools": len(TOOLS),
        "min_confidence": MIN_CONFIDENCE,
        "flip_fraction": round(flips / TRIALS, 4),
        "none_to_some_fraction": round(none_to_some / TRIALS, 4),
        "some_to_none_fraction": round(some_to_none / TRIALS, 4),
        "gate_pass_fraction": {k: round(v / TRIALS, 4) for k, v in passes.items()},
        "cluster_term_observed_range": {k: (round(term_min[k], 3), round(term_max[k], 3)) for k in term_min},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", default="", help="write results here")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args(argv)

    provenance = None
    if args.json:
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        import maxim
        from _provenance import DirtyTreeError, ProvenanceError, in_process_code_provenance

        try:
            provenance = in_process_code_provenance(
                _REPO_ROOT, maxim.__file__, out_path=args.json, allow_dirty=args.allow_dirty
            )
        except (ProvenanceError, DirtyTreeError) as exc:
            print(f"[FAIL] gated-record preflight: {exc}", file=sys.stderr)
            return 3

    result = run(args.seed)
    for key, value in result.items():
        print(f"  {key}: {value}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "harness": "scripts/selection_dynamics_rebaseline.py",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "provenance": provenance,
                    "metric": "flip/gate-pass/term-range characterization, 2 vs 3 channels, real NAc",
                    "decision": "min_confidence stays 0.3 for 1.1.4 (baseline, not a gate); 1.2 calibrates on real distributions",
                    "seed": args.seed,
                    "result": result,
                },
                indent=2,
            )
        )
        print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
