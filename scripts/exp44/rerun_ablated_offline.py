"""Exp 44 counterfactual harness — Pass 2: offline temp-0 re-query.

Reads the paired-prompt JSONL from ``capture_paired_prompts.py`` and, for each
decision, queries the LLM at temp 0 on BOTH the full and ablated prompts. A
FLIP (action_full != action_ablated) means the learned-orient substrate changed
the decision at that world-state.

Re-querying BOTH offline at temp 0 (rather than reusing arm A's live action)
puts the two arms on identical decoding, so a flip is attributable to the
prompt delta alone, not sampling.

**Slice by prior strength (the interpretation guard).** Substrate should only
move decisions where the LLM's own prior is weak (Exp 37/38/40: prior-agreement
is the gating variable). We estimate prior strength by sampling the ABLATED
prompt (= the LLM's substrate-free prior) N times at a moderate temperature and
taking the action-distribution entropy. The headline is the flip rate AMONG
high-entropy decisions; the low-entropy flip rate is reported as a ~0 control.

Usage::

    MAXIM_LLM_PROFILE=<model> python scripts/exp44/rerun_ablated_offline.py \
        --log data/exp44/paired_prompts.jsonl \
        --out data/exp44/counterfactual_results.jsonl \
        --entropy-samples 8 --entropy-temp 0.7
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _build_router() -> Any:
    """Construct an LLMRouter from the ambient config (MAXIM_LLM_PROFILE)."""
    from maxim.models.language.router import LLMRouter, load_llm_config

    return LLMRouter(load_llm_config())


def _action_of(router: Any, prompt: str, *, temperature: float, max_tokens: int = 512) -> str | None:
    """Query the LLM and extract the proposed tool name (None on failure).

    The tool-aware response format nests the tool under ``action``:
    ``{"ready_to_act": ..., "action": {"tool_name": "...", "params": {...}}}``.
    Fall back to a flat ``tool_name`` for other shapes.
    """
    resp = router.generate_json(prompt, temperature=temperature, max_tokens=max_tokens)
    if not isinstance(resp, dict):
        return None
    action = resp.get("action")
    if isinstance(action, dict) and action.get("tool_name") is not None:
        return str(action["tool_name"])
    tool = resp.get("tool_name")  # flat fallback
    return str(tool) if tool is not None else None


def _prior_entropy(router: Any, prompt_ablated: str, *, samples: int, temperature: float) -> float:
    """Shannon entropy (bits) of the action distribution on the substrate-free
    prompt — the LLM's own prior at this world-state. High = ambiguous prior."""
    actions = [_action_of(router, prompt_ablated, temperature=temperature) for _ in range(samples)]
    actions = [a for a in actions if a is not None]
    if not actions:
        return 0.0
    counts = Counter(actions)
    total = len(actions)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="paired-prompt JSONL from pass 1")
    ap.add_argument("--out", required=True, help="per-decision results JSONL")
    ap.add_argument("--entropy-samples", type=int, default=8, help="0 = skip prior-entropy pass")
    ap.add_argument("--entropy-temp", type=float, default=0.7)
    ap.add_argument("--entropy-hi", type=float, default=0.5, help="bits above which a prior is 'weak'")
    ap.add_argument("--max-decisions", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--determinism-check",
        type=int,
        default=0,
        help=(
            "Before scoring, re-query the SAME prompt twice at temp 0 for N rows and "
            "report the disagreement rate. The whole method assumes temp-0 decoding is "
            "deterministic — if it is not, some 'flips' are decoder noise, not substrate "
            "effect. 0 = skip."
        ),
    )
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "prompt_full" in r and "prompt_ablated" in r]
    if args.max_decisions:
        rows = rows[: args.max_decisions]
    if not rows:
        print("no usable paired-prompt rows in", args.log)
        return 1

    print(f"[exp44] building router + loading model (first call is slow)… scoring {len(rows)} decisions", flush=True)
    router = _build_router()

    # ── Determinism check (instrument validity) ─────────────────────────────
    # The counterfactual attributes a changed action to the prompt delta. That
    # inference is only sound if temp-0 decoding returns the same action for
    # the same prompt — otherwise a share of "flips" is decoder noise (batching,
    # KV-cache reuse, and non-associative float reduction can all break
    # determinism in practice, even at temperature 0). Reviewers will ask; this
    # measures it instead of assuming it. Reported, never silently swallowed.
    if args.determinism_check > 0:
        n_check = min(args.determinism_check, len(rows))
        disagreements = []
        for i, r in enumerate(rows[:n_check], 1):
            a1 = _action_of(router, r["prompt_full"], temperature=0.0)
            a2 = _action_of(router, r["prompt_full"], temperature=0.0)
            if a1 != a2:
                disagreements.append({"decision_id": r.get("decision_id"), "first": a1, "second": a2})
            print(f"[determinism {i}/{n_check}] {a1} vs {a2} {'MISMATCH' if a1 != a2 else 'ok'}", flush=True)
        rate = len(disagreements) / n_check if n_check else 0.0
        print(f"\n[exp44 determinism] {len(disagreements)}/{n_check} disagreed = {rate:.3f}")
        if disagreements:
            print(
                "  WARNING: temp-0 decoding is NOT deterministic on this backend. Flip "
                "counts include decoder noise at roughly this rate — report it as a floor "
                "on the effect, and consider it when reading small NETs.",
                file=sys.stderr,
            )
            for d in disagreements[:5]:
                print(f"    id={d['decision_id']}: {d['first']!r} != {d['second']!r}", file=sys.stderr)
        else:
            print("  temp-0 decoding is deterministic on these prompts — flips are prompt-attributable.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with out_path.open("w") as f:
        for i, r in enumerate(rows, 1):
            a_full = _action_of(router, r["prompt_full"], temperature=0.0)
            a_abl = _action_of(router, r["prompt_ablated"], temperature=0.0)
            flipped = a_full is not None and a_abl is not None and a_full != a_abl
            entropy = (
                _prior_entropy(router, r["prompt_ablated"], samples=args.entropy_samples, temperature=args.entropy_temp)
                if args.entropy_samples > 0
                else None
            )
            rec = {
                "decision_id": r.get("decision_id"),
                "action_full": a_full,
                "action_ablated": a_abl,
                "flipped": flipped,
                "prior_entropy_bits": entropy,
                "world_state": r.get("world_state"),
            }
            results.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()  # so `wc -l` on --out is a real live progress indicator
            tag = "FLIP" if flipped else ("skip" if a_full is None or a_abl is None else "same")
            print(f"[{i}/{len(rows)}] full={a_full} abl={a_abl} {tag}", flush=True)

    # ── rollup ──────────────────────────────────────────────────────────────
    scored = [x for x in results if x["action_full"] is not None and x["action_ablated"] is not None]
    n = len(scored)
    flips = sum(1 for x in scored if x["flipped"])
    print(f"\n[exp44 counterfactual] decisions scored: {n}/{len(results)}")
    print(f"  overall flip rate: {flips}/{n} = {flips / n:.3f}" if n else "  no scored decisions")

    if args.entropy_samples > 0 and n:
        hi = [x for x in scored if (x["prior_entropy_bits"] or 0.0) >= args.entropy_hi]
        lo = [x for x in scored if (x["prior_entropy_bits"] or 0.0) < args.entropy_hi]
        hi_flip = sum(1 for x in hi if x["flipped"])
        lo_flip = sum(1 for x in lo if x["flipped"])
        print(
            f"  weak-prior (entropy>= {args.entropy_hi}b): {hi_flip}/{len(hi)} = "
            f"{(hi_flip / len(hi)) if hi else float('nan'):.3f}   <- HEADLINE"
        )
        print(
            f"  strong-prior (entropy<  {args.entropy_hi}b): {lo_flip}/{len(lo)} = "
            f"{(lo_flip / len(lo)) if lo else float('nan'):.3f}   <- should be ~0 (control)"
        )
        print("  NOTE: flips are unlabeled here — join world_state to label toward/away")
        print("        from correct orient/safe before claiming the substrate helps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
