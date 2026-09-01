#!/usr/bin/env python3
"""Analyzer for Exp 42 — Substrate-Primary Preference (counterbalanced A/B).

Consumes the per-run JSONL produced by ``scripts/benchmark_exp42_preference.py``
and emits the FROZEN verdict from docs/experiments/42_substrate_primary_preference.md
§4 (hypotheses/metrics) + §5 (disposition matrix). No metric is computed here that
isn't pinned in §4 — the analyzer is a faithful executor of the pre-registration.

Record schema (one JSON object per line, per run):

    {"experiment": "exp42", "arm": "cradle_pref_a", "seed": 42,
     "n_actions": 120, "n_contact": 22, "n_exploit": 14,
     "safe_pref": 1.0, "id_pref_a": 0.07, "discovery_pos": 1,
     "harm_net": -0.25, "safe_net": 0.14, "mock": false}

Arms (counterbalanced): cradle_pref_a (safe = β) and cradle_pref_b (safe = α).

FROZEN metrics (§4):
  * safe_pref(arm) = exploitation contacts on the SAFE source / all exploitation
    contacts (warm_self + touch), pooled across valid seeds.
  * Validity gate K: a seed counts only if n_exploit ≥ K (else floored, excluded
    and reported). An arm with zero valid seeds → VOID (the Exp 41 flooring trap).
  * H1 (PRIMARY, sufficient, bias-proof): safe_pref(a) ≥ 0.66 AND safe_pref(b) ≥ 0.66.
  * C1 (corroborating): id_pref_a(b) − id_pref_a(a) ≥ +0.33 (identity flips with harm).
  * C2 (corroborating): harmful-source net < safe-source net (relative; best-effort).
  * substrate_signal = H1.

Disposition (§5) → exit codes:
  * GRADUATE (safe_pref ≥ 0.66 in BOTH arms)        → 0
  * PARTIAL  (safe_pref ≥ 0.66 in exactly one arm)  → 4
  * REFRAME  (safe_pref < 0.66 in BOTH arms)        → 5
  * VOID     (floored / missing-arm: not measurable)→ 4
  * analyzer error                                  → 2

Determinism note: substrate-primary action selection is deterministic, so
cross-seed variance arises only from the LLM narrator's scene timing. H1 is an
ABSOLUTE threshold on the pooled mean (not an SD-relative effect like Exp 41), so
the per-arm SD is reported for transparency but does not enter the gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Exit codes (mirror analyze_exp41_exploration.py conventions).
EXIT_GRADUATE = 0
EXIT_ERROR = 2
EXIT_PARTIAL = 4
EXIT_REFRAME = 5

ARMS = ("cradle_pref_a", "cradle_pref_b")

# Frozen thresholds (§4). Set at authorization after triage; do NOT re-tune post-hoc.
SAFE_PREF_THRESHOLD = 0.66
MIN_EXPLOIT_K = 10
C1_FLIP_THRESHOLD = 0.33

_MARKER = "<!-- Analyzer appends"


# ── data model ───────────────────────────────────────────────────────────


@dataclass
class ArmStats:
    arm: str
    n_seeds_total: int
    n_seeds_valid: int  # seeds with n_exploit ≥ K
    n_floored: int  # seeds excluded for n_exploit < K (or no exploitation phase)
    safe_prefs: list[float] = field(default_factory=list)
    id_pref_a: list[float] = field(default_factory=list)
    harm_nets: list[float] = field(default_factory=list)
    safe_nets: list[float] = field(default_factory=list)
    mean_safe_pref: float | None = None
    sd_safe_pref: float = 0.0
    mean_id_pref_a: float | None = None
    mean_harm_net: float | None = None
    mean_safe_net: float | None = None


@dataclass
class AnalysisResult:
    verdict: str
    exit_code: int
    h1_pass: bool | None
    arm_pass: dict[str, bool | None]
    c1_flip: float | None
    c1_pass: bool | None
    c2_pass: bool | None
    substrate_signal: bool
    void: bool
    void_reason: str | None
    arms: dict[str, ArmStats]
    notes: list[str]


# ── stats helpers ─────────────────────────────────────────────────────────


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _num(rec: dict[str, Any], key: str) -> float | None:
    v = rec.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── core analysis ─────────────────────────────────────────────────────────


def _build_arm_stats(arm: str, records: list[dict[str, Any]], *, min_k: int) -> ArmStats:
    safe_prefs: list[float] = []
    id_pref_a: list[float] = []
    harm_nets: list[float] = []
    safe_nets: list[float] = []
    n_total = len(records)
    n_floored = 0
    for rec in records:
        n_exploit = rec.get("n_exploit")
        sp = _num(rec, "safe_pref")
        # validity gate: enough exploitation samples AND a computed safe_pref.
        if not isinstance(n_exploit, int) or n_exploit < min_k or sp is None:
            n_floored += 1
            continue
        safe_prefs.append(sp)
        ip = _num(rec, "id_pref_a")
        if ip is not None:
            id_pref_a.append(ip)
        hn = _num(rec, "harm_net")
        if hn is not None:
            harm_nets.append(hn)
        sn = _num(rec, "safe_net")
        if sn is not None:
            safe_nets.append(sn)
    return ArmStats(
        arm=arm,
        n_seeds_total=n_total,
        n_seeds_valid=len(safe_prefs),
        n_floored=n_floored,
        safe_prefs=safe_prefs,
        id_pref_a=id_pref_a,
        harm_nets=harm_nets,
        safe_nets=safe_nets,
        mean_safe_pref=statistics.mean(safe_prefs) if safe_prefs else None,
        sd_safe_pref=_sample_sd(safe_prefs),
        mean_id_pref_a=statistics.mean(id_pref_a) if id_pref_a else None,
        mean_harm_net=statistics.mean(harm_nets) if harm_nets else None,
        mean_safe_net=statistics.mean(safe_nets) if safe_nets else None,
    )


def analyze(records: list[dict[str, Any]], *, min_k: int = MIN_EXPLOIT_K) -> AnalysisResult:
    by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in ARMS}
    for rec in records:
        arm = rec.get("arm")
        if arm in by_arm:
            by_arm[arm].append(rec)

    arms = {a: _build_arm_stats(a, by_arm[a], min_k=min_k) for a in ARMS}
    notes: list[str] = []

    a, b = arms[ARMS[0]], arms[ARMS[1]]

    # VOID: a missing arm or an arm with no valid (non-floored) seeds — the
    # metric is not measurable (the Exp 41 flooring trap; gate hard).
    floored_arms = [s.arm for s in (a, b) if s.n_seeds_valid == 0]
    if floored_arms:
        reasons = ", ".join(
            f"{s.arm}: {s.n_seeds_valid}/{s.n_seeds_total} valid (K={min_k}; {s.n_floored} floored)" for s in (a, b)
        )
        return AnalysisResult(
            verdict="VOID — metric not measurable (floored / missing arm)",
            exit_code=EXIT_PARTIAL,
            h1_pass=None,
            arm_pass={a.arm: None, b.arm: None},
            c1_flip=None,
            c1_pass=None,
            c2_pass=None,
            substrate_signal=False,
            void=True,
            void_reason=(
                f"arm(s) with no valid seeds: {floored_arms} — too few exploitation "
                f"contacts (need n_exploit ≥ {min_k}); {reasons}. Re-tune (speed cold "
                "re-drift / raise turns) and re-triage; do not freeze a floored metric."
            ),
            arms=arms,
            notes=notes,
        )

    for s in (a, b):
        if s.n_floored:
            notes.append(f"{s.arm}: {s.n_floored}/{s.n_seeds_total} seeds floored (n_exploit < {min_k}), excluded.")

    arm_pass = {s.arm: (s.mean_safe_pref is not None and s.mean_safe_pref >= SAFE_PREF_THRESHOLD) for s in (a, b)}
    h1_pass = bool(arm_pass[a.arm] and arm_pass[b.arm])
    n_pass = sum(1 for v in arm_pass.values() if v)

    # C1 — identity-preference flips with the harm assignment (reported).
    c1_flip: float | None = None
    c1_pass: bool | None = None
    if a.mean_id_pref_a is not None and b.mean_id_pref_a is not None:
        c1_flip = b.mean_id_pref_a - a.mean_id_pref_a
        c1_pass = c1_flip >= C1_FLIP_THRESHOLD
    else:
        notes.append("C1: id_pref_a incomplete — identity-flip check skipped.")

    # C2 — per-source learning signs, relative (reported, best-effort).
    c2_pass: bool | None = None
    c2_arms = [s for s in (a, b) if s.mean_harm_net is not None and s.mean_safe_net is not None]
    if c2_arms:
        c2_pass = all(s.mean_harm_net < s.mean_safe_net for s in c2_arms)  # type: ignore[operator]
    else:
        notes.append("C2: per-source nets absent (no telemetry) — learning-sign check skipped.")

    substrate_signal = h1_pass

    if n_pass == 2:
        verdict = "GRADUATE #6 — substrate drives adaptive, feedback-tracked behavior"
        exit_code = EXIT_GRADUATE
    elif n_pass == 1:
        won = next(arm for arm, p in arm_pass.items() if p)
        verdict = f"PARTIAL — asymmetric preference (safe preferred only in {won})"
        exit_code = EXIT_PARTIAL
    else:
        verdict = "REFRAME #6 — no adaptive preference even with a safe escape"
        exit_code = EXIT_REFRAME

    return AnalysisResult(
        verdict=verdict,
        exit_code=exit_code,
        h1_pass=h1_pass,
        arm_pass=arm_pass,
        c1_flip=c1_flip,
        c1_pass=c1_pass,
        c2_pass=c2_pass,
        substrate_signal=substrate_signal,
        void=False,
        void_reason=None,
        arms=arms,
        notes=notes,
    )


# ── rendering ──────────────────────────────────────────────────────────────


def _fmt(v: float | None, prec: int = 3) -> str:
    return "—" if v is None else f"{v:.{prec}f}"


def render_markdown(result: AnalysisResult, *, heading_suffix: str = "") -> str:
    suffix = f" — {heading_suffix}" if heading_suffix else ""
    lines: list[str] = []
    lines.append(f"## Results{suffix}")
    lines.append("")
    lines.append(f"**Verdict: {result.verdict}**  (exit {result.exit_code})")
    lines.append("")
    if result.void:
        lines.append(f"> VOID: {result.void_reason}")
        lines.append("")
    lines.append(f"- `substrate_signal` (H1, both arms ≥ {SAFE_PREF_THRESHOLD}): **{result.substrate_signal}**")
    for arm in ARMS:
        p = result.arm_pass.get(arm)
        st = result.arms.get(arm)
        mean = _fmt(st.mean_safe_pref, 3) if st else "—"
        tag = "PASS" if p else "FAIL" if p is not None else "—"
        lines.append(f"- H1[{arm}] safe_pref = {mean} → {tag}")
    c1 = "—" if result.c1_flip is None else f"{result.c1_flip:+.3f}"
    c1tag = "PASS" if result.c1_pass else "FAIL" if result.c1_pass is not None else "—"
    lines.append(f"- C1 (identity flip id_pref_a(b)−id_pref_a(a) ≥ {C1_FLIP_THRESHOLD}): {c1} → {c1tag}")
    c2tag = "PASS" if result.c2_pass else "FAIL" if result.c2_pass is not None else "—"
    lines.append(f"- C2 (per-source learning, harm net < safe net): {c2tag}")
    lines.append("")
    lines.append("### Per-arm (pooled across valid seeds)")
    lines.append("")
    lines.append("| arm | safe id | valid/total | floored | safe_pref | SD | id_pref_a | harm net | safe net |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    safe_id = {"cradle_pref_a": "β", "cradle_pref_b": "α"}
    for arm in ARMS:
        st = result.arms.get(arm)
        if st is None:
            lines.append(f"| {arm} | — | 0/0 | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {arm} | {safe_id.get(arm, '?')} | {st.n_seeds_valid}/{st.n_seeds_total} | {st.n_floored} | "
            f"{_fmt(st.mean_safe_pref, 3)} | {st.sd_safe_pref:.3f} | {_fmt(st.mean_id_pref_a, 3)} | "
            f"{_fmt(st.mean_harm_net, 3)} | {_fmt(st.mean_safe_net, 3)} |"
        )
    lines.append("")
    if result.notes:
        lines.append("### Notes")
        for n in result.notes:
            lines.append(f"- {n}")
        lines.append("")
    return "\n".join(lines)


def _emit_to_doc(doc_path: Path, body: str, *, force: bool = False, append: bool = False) -> None:
    """Write ``body`` below the analyzer marker.

    Three modes, because the two-mode version destroyed a frozen record:

    * default — refuse when non-trivial content already sits below the marker.
    * ``append`` — keep what is there and add ``body`` BELOW it. This is the
      right mode for a re-validation: the doc accumulates one section per run.
    * ``force`` — REPLACE everything below the marker. Destructive by design;
      only for correcting a section this analyzer itself just wrote.

    History (2026-09-01): the refusal above existed precisely to protect a
    recorded verdict — its own docstring named "Exp 42's GRADUATE record" as
    the thing at risk — but the only escape hatch offered was ``--force``, and
    the error message told the operator to use it. A post-D53 re-validation
    did, twice, and replaced main's frozen 2026-06-23 Results section (both
    tables plus the B7-is-not-load-bearing analysis) with a single generated
    block. Caught in PR review, restored byte-for-byte. A guard whose only
    documented exit is the destructive one is not a guard, so ``--append``
    now exists and the refusal recommends it first.
    """
    text = doc_path.read_text() if doc_path.exists() else ""
    idx = text.find(_MARKER)
    if idx == -1:
        head = text.rstrip() + "\n\n" + _MARKER + " '## Results' sections below this line -->\n\n"
        doc_path.write_text(head + body + "\n")
        return

    line_end = text.find("\n", idx)
    cut = line_end + 1 if line_end != -1 else len(text)
    head = text[:cut] + "\n"
    existing_below = text[cut:].strip()

    if not existing_below:
        doc_path.write_text(head + body + "\n")
        return
    if append:
        doc_path.write_text(head + existing_below + "\n\n" + body + "\n")
        return
    if force:
        doc_path.write_text(head + body + "\n")
        return
    raise SystemExit(
        f"ERROR: {doc_path} already has content below the analyzer marker "
        f"({len(existing_below)} chars — possibly a recorded verdict).\n"
        "  --append  keep it and add this run below (what a re-validation wants)\n"
        "  --force   REPLACE it (destructive; only to fix a section just written)"
    )


# ── io ─────────────────────────────────────────────────────────────────────


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("arm") in ARMS:
            records.append(rec)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exp 42 substrate-primary preference analyzer")
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="per-run records JSONL")
    parser.add_argument("--out", dest="out_path", type=Path, default=None, help="doc to write results into")
    parser.add_argument("--trials", type=int, default=None, help="expected seeds per arm (advisory)")
    parser.add_argument("--min-exploit", type=int, default=MIN_EXPLOIT_K, help="validity gate K (frozen at 10)")
    parser.add_argument("--heading-suffix", default="", help="suffix for the '## Results' heading")
    parser.add_argument(
        "--append",
        action="store_true",
        help="keep existing content below the marker and add this run BELOW it (re-validation)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing content below the doc's analyzer marker "
        "(refused by default — it may be a recorded verdict).",
    )
    args = parser.parse_args(argv)

    if not args.in_path.exists():
        print(f"error: input not found: {args.in_path}", file=sys.stderr)
        return EXIT_ERROR

    records = load_records(args.in_path)
    if not records:
        print(f"error: no valid exp42 records in {args.in_path}", file=sys.stderr)
        return EXIT_ERROR

    if args.trials is not None:
        for arm in ARMS:
            n = sum(1 for r in records if r.get("arm") == arm)
            if n != args.trials:
                print(f"warning: arm {arm} has {n} records, expected {args.trials}", file=sys.stderr)

    result = analyze(records, min_k=args.min_exploit)
    body = render_markdown(result, heading_suffix=args.heading_suffix)
    print(body)
    if args.out_path is not None:
        _emit_to_doc(args.out_path, body, force=args.force, append=args.append)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
