#!/usr/bin/env python3
"""Analyzer for Exp 41 — Substrate-Primary Exploration (counter-prior).

Consumes the per-run JSONL produced by ``scripts/benchmark_exp41_exploration.py``
and emits the FROZEN verdict from docs/experiments/41_substrate_primary_exploration.md
§4 (hypotheses/metrics) + §5 (disposition matrix). No metric is computed here that
isn't pinned in §4 — the analyzer is a faithful executor of the pre-registration.

Record schema (one JSON object per line, per run):

    {"experiment": "exp41", "arm": "A_dec", "deceptive": true,
     "exploration": false, "seed": 42,
     "harm_rate_thirds": [f, f, f],          # deceptive arms: harmful-action rate
     "warm_self_rate_thirds": [f, f, f],     # consistent arms: correct-action rate
     "n_actions": 120, "mock": false}

Arms (2×2): A_cons / B_cons (consistent arc) × A_dec / B_dec (deceptive arc),
where A = exploration OFF, B = exploration ON.

FROZEN metrics (§4):
  * harm_rate(third)  — deceptive arms; want it to FALL (learned avoidance).
  * warm_self_rate(third) — consistent arms; want it to STAY HIGH (correct prior).
  * SD = pooled across-seed sample stdev of the first-third value of the arm.
  * H1 (between-arm): (mean_last[A_dec] − mean_last[B_dec]) / SD(A_dec first) ≥ 1.0
  * H2 (within-arm) : mean_seed(first[B_dec] − last[B_dec]) / SD(B_dec first) ≥ 1.0
  * S1 (consistent no-regression, informational)
  * S2 (interaction, informational)
  * substrate_signal = H1 PASS AND H2 PASS

Disposition (§5) → exit codes:
  * GRADUATE (H1 PASS, H2 PASS)                         → 0
  * PARTIAL  (exactly one of H1/H2 PASS)                → 4
  * REFRAME  (H1 FAIL, H2 FAIL)                         → 5
  * VOID     (mechanism-not-ready: no temptation)       → 4
  * analyzer error                                      → 2

Determinism note: substrate-primary action selection is deterministic, so
cross-seed variance arises only from the LLM narrator's scene timing. When the
first-third SD of an arm is ~0 the SD-unit ratio is undefined; this analyzer
treats SD≈0 as a sign test (PASS iff the raw effect exceeds ``--zero-sd-floor``)
and FLAGS it in the output so a vacuous "≥1.0 SD" is never silently reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Exit codes (mirror analyze_exp37.py conventions).
EXIT_GRADUATE = 0
EXIT_ERROR = 2
EXIT_PARTIAL = 4
EXIT_REFRAME = 5

ARMS = ("A_cons", "B_cons", "A_dec", "B_dec")
_DECEPTIVE_ARMS = ("A_dec", "B_dec")
_CONSISTENT_ARMS = ("A_cons", "B_cons")

# Threshold from §4 (frozen): effect must reach ≥ 1.0 SD of the first-third value.
SD_THRESHOLD = 1.0
# When SD ≈ 0 (deterministic runs), fall back to a raw-effect sign test.
DEFAULT_ZERO_SD_FLOOR = 0.10

_MARKER = "<!-- Analyzer appends"


# ── data model ───────────────────────────────────────────────────────────


@dataclass
class ArmStats:
    arm: str
    deceptive: bool
    n_seeds: int
    first: list[float]  # per-seed first-third value (harm or warm_self)
    last: list[float]  # per-seed last-third value
    mean_first: float
    mean_last: float
    sd_first: float
    # within-seed delta (first − last); positive ⇒ value fell over session
    deltas: list[float] = field(default_factory=list)
    mean_delta: float = 0.0


@dataclass
class AnalysisResult:
    verdict: str
    exit_code: int
    h1_pass: bool | None
    h2_pass: bool | None
    h1_ratio: float | None
    h2_ratio: float | None
    s1_pass: bool | None
    interaction: float | None
    substrate_signal: bool
    void: bool
    void_reason: str | None
    arms: dict[str, ArmStats]
    notes: list[str]


# ── stats helpers ─────────────────────────────────────────────────────────


def _sample_sd(values: list[float]) -> float:
    """Sample stdev; 0.0 for <2 points (pooled across-seed SD per §4)."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _ratio(numerator: float, sd: float, *, zero_sd_floor: float) -> tuple[float, bool]:
    """Effect / SD with SD≈0 handling.

    Returns ``(ratio, sd_is_zero)``. When SD ≈ 0 the ratio is a sign test:
    +inf if the raw effect clears the floor, −inf if it's clearly negative,
    else 0.0 (no effect, no variance → not a pass).
    """
    if sd > 1e-9:
        return numerator / sd, False
    if numerator >= zero_sd_floor:
        return float("inf"), True
    if numerator <= -zero_sd_floor:
        return float("-inf"), True
    return 0.0, True


def _channel(rec: dict[str, Any], deceptive: bool) -> list[float] | None:
    key = "harm_rate_thirds" if deceptive else "warm_self_rate_thirds"
    thirds = rec.get(key)
    if not isinstance(thirds, list) or len(thirds) != 3:
        return None
    try:
        return [float(x) for x in thirds]
    except (TypeError, ValueError):
        return None


# ── core analysis ─────────────────────────────────────────────────────────


def _build_arm_stats(arm: str, records: list[dict[str, Any]]) -> ArmStats | None:
    deceptive = arm in _DECEPTIVE_ARMS
    first: list[float] = []
    last: list[float] = []
    deltas: list[float] = []
    for rec in records:
        ch = _channel(rec, deceptive)
        if ch is None:
            continue
        first.append(ch[0])
        last.append(ch[2])
        deltas.append(ch[0] - ch[2])
    if not first:
        return None
    return ArmStats(
        arm=arm,
        deceptive=deceptive,
        n_seeds=len(first),
        first=first,
        last=last,
        mean_first=statistics.mean(first),
        mean_last=statistics.mean(last),
        sd_first=_sample_sd(first),
        deltas=deltas,
        mean_delta=statistics.mean(deltas),
    )


def analyze(records: list[dict[str, Any]], *, zero_sd_floor: float = DEFAULT_ZERO_SD_FLOOR) -> AnalysisResult:
    by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in ARMS}
    for rec in records:
        arm = rec.get("arm")
        if arm in by_arm:
            by_arm[arm].append(rec)

    arms: dict[str, ArmStats] = {}
    for arm in ARMS:
        st = _build_arm_stats(arm, by_arm[arm])
        if st is not None:
            arms[arm] = st

    notes: list[str] = []

    # Need both deceptive arms to test H1/H2 at all.
    a_dec = arms.get("A_dec")
    b_dec = arms.get("B_dec")
    if a_dec is None or b_dec is None:
        return AnalysisResult(
            verdict="VOID — missing deceptive-arm data (need A_dec and B_dec)",
            exit_code=EXIT_PARTIAL,
            h1_pass=None,
            h2_pass=None,
            h1_ratio=None,
            h2_ratio=None,
            s1_pass=None,
            interaction=None,
            substrate_signal=False,
            void=True,
            void_reason="A_dec and/or B_dec records absent",
            arms=arms,
            notes=notes,
        )

    # Mechanism-not-ready (§5): if the harmful action is never even tried in the
    # exploration-OFF deceptive arm AND barely tried with exploration ON, there
    # is no counter-prior temptation to override → VOID (don't false-GRADUATE
    # on "nothing to learn"). We gate on B_dec's first-third harm: explore-first
    # should surface warm_self at least once early.
    if b_dec.mean_first < zero_sd_floor and a_dec.mean_first < zero_sd_floor:
        return AnalysisResult(
            verdict="VOID — no counter-prior temptation (harmful action never engaged)",
            exit_code=EXIT_PARTIAL,
            h1_pass=None,
            h2_pass=None,
            h1_ratio=None,
            h2_ratio=None,
            s1_pass=None,
            interaction=None,
            substrate_signal=False,
            void=True,
            void_reason=(
                f"first-third harm_rate ~0 in both deceptive arms "
                f"(A_dec={a_dec.mean_first:.3f}, B_dec={b_dec.mean_first:.3f}); "
                "warm_self was never a live temptation — check drive tuning / cold body"
            ),
            arms=arms,
            notes=notes,
        )

    # H1 — between-arm: exploration reduces end-of-session harm vs OFF.
    h1_num = a_dec.mean_last - b_dec.mean_last
    h1_ratio, h1_zero = _ratio(h1_num, a_dec.sd_first, zero_sd_floor=zero_sd_floor)
    h1_pass = h1_ratio >= SD_THRESHOLD
    if h1_zero:
        notes.append(
            f"H1: A_dec first-third SD ≈ 0 (deterministic) — used sign test on raw "
            f"effect {h1_num:+.3f} (floor {zero_sd_floor})."
        )

    # H2 — within-arm: the reduction is genuine within-session learning.
    h2_ratio, h2_zero = _ratio(b_dec.mean_delta, b_dec.sd_first, zero_sd_floor=zero_sd_floor)
    h2_pass = h2_ratio >= SD_THRESHOLD
    if h2_zero:
        notes.append(
            f"H2: B_dec first-third SD ≈ 0 (deterministic) — used sign test on mean "
            f"within-session drop {b_dec.mean_delta:+.3f} (floor {zero_sd_floor})."
        )

    # S1 — consistent-arc no-regression (informational).
    s1_pass: bool | None = None
    a_cons = arms.get("A_cons")
    b_cons = arms.get("B_cons")
    if a_cons is not None and b_cons is not None:
        s1_pass = b_cons.mean_last >= (a_cons.mean_last - 0.5 * a_cons.sd_first)
    else:
        notes.append("S1: consistent-arc data incomplete — no-regression check skipped.")

    # S2 — interaction (informational): how much MORE exploration helps when the
    # prior is wrong vs the perturbation it causes when the prior is right.
    interaction: float | None = None
    if a_cons is not None and b_cons is not None:
        deceptive_help = a_dec.mean_last - b_dec.mean_last  # harm reduction (want +)
        consistent_perturb = a_cons.mean_last - b_cons.mean_last  # correct-action loss (want ~0)
        interaction = deceptive_help - consistent_perturb

    substrate_signal = bool(h1_pass and h2_pass)

    if h1_pass and h2_pass:
        verdict = "GRADUATE #6 — substrate drives adaptive behavior (unmasked)"
        exit_code = EXIT_GRADUATE
    elif h1_pass and not h2_pass:
        verdict = "PARTIAL — avoidance without learning (H1 only)"
        exit_code = EXIT_PARTIAL
    elif not h1_pass and h2_pass:
        verdict = "PARTIAL — learning without behavioral payoff (H2 only)"
        exit_code = EXIT_PARTIAL
    else:
        verdict = "REFRAME #6 — fixation is deeper than selection (H1 & H2 fail)"
        exit_code = EXIT_REFRAME

    return AnalysisResult(
        verdict=verdict,
        exit_code=exit_code,
        h1_pass=h1_pass,
        h2_pass=h2_pass,
        h1_ratio=h1_ratio,
        h2_ratio=h2_ratio,
        s1_pass=s1_pass,
        interaction=interaction,
        substrate_signal=substrate_signal,
        void=False,
        void_reason=None,
        arms=arms,
        notes=notes,
    )


# ── rendering ──────────────────────────────────────────────────────────────


def _fmt_ratio(r: float | None) -> str:
    if r is None:
        return "—"
    if r == float("inf"):
        return "+∞"
    if r == float("-inf"):
        return "−∞"
    return f"{r:.2f}"


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
    lines.append(f"- `substrate_signal` (H1 ∧ H2): **{result.substrate_signal}**")
    lines.append(
        f"- H1 (between-arm, harm drop / SD): {_fmt_ratio(result.h1_ratio)} "
        f"→ {'PASS' if result.h1_pass else 'FAIL' if result.h1_pass is not None else '—'}"
    )
    lines.append(
        f"- H2 (within-arm, learning / SD): {_fmt_ratio(result.h2_ratio)} "
        f"→ {'PASS' if result.h2_pass else 'FAIL' if result.h2_pass is not None else '—'}"
    )
    s1 = "PASS" if result.s1_pass else "FAIL" if result.s1_pass is not None else "—"
    lines.append(f"- S1 (consistent no-regression, informational): {s1}")
    inter = "—" if result.interaction is None else f"{result.interaction:+.3f}"
    lines.append(f"- S2 (interaction, informational): {inter}")
    lines.append("")
    lines.append("### Per-arm (first-third → last-third, mean across seeds)")
    lines.append("")
    lines.append("| arm | arc | explore | seeds | channel | first | last | SD(first) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        st = result.arms.get(arm)
        if st is None:
            lines.append(f"| {arm} | — | — | 0 | — | — | — | — |")
            continue
        arc = "deceptive" if st.deceptive else "consistent"
        explore = "ON" if arm.startswith("B") else "OFF"
        channel = "harm_rate" if st.deceptive else "warm_self_rate"
        lines.append(
            f"| {arm} | {arc} | {explore} | {st.n_seeds} | {channel} | "
            f"{st.mean_first:.3f} | {st.mean_last:.3f} | {st.sd_first:.3f} |"
        )
    lines.append("")
    if result.notes:
        lines.append("### Notes")
        for n in result.notes:
            lines.append(f"- {n}")
        lines.append("")
    return "\n".join(lines)


def _emit_to_doc(doc_path: Path, body: str) -> None:
    """Replace everything below the analyzer marker with ``body`` (overwrite,
    not append — matches analyze_exp37 semantics; commit the doc first)."""
    text = doc_path.read_text() if doc_path.exists() else ""
    idx = text.find(_MARKER)
    if idx == -1:
        # No marker: append a marker + body so the doc gains the contract.
        head = text.rstrip() + "\n\n" + _MARKER + " '## Results' sections below this line -->\n\n"
    else:
        line_end = text.find("\n", idx)
        head = text[: (line_end + 1 if line_end != -1 else len(text))] + "\n"
    doc_path.write_text(head + body + "\n")


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
    parser = argparse.ArgumentParser(description="Exp 41 substrate-primary exploration analyzer")
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="per-run records JSONL")
    parser.add_argument("--out", dest="out_path", type=Path, default=None, help="doc to write results into")
    parser.add_argument("--trials", type=int, default=None, help="expected seeds per arm (advisory)")
    parser.add_argument("--heading-suffix", default="", help="suffix for the '## Results' heading")
    parser.add_argument("--zero-sd-floor", type=float, default=DEFAULT_ZERO_SD_FLOOR)
    parser.add_argument("--strict-schema-version", default=None, help="(reserved)")
    args = parser.parse_args(argv)

    if not args.in_path.exists():
        print(f"error: input not found: {args.in_path}", file=sys.stderr)
        return EXIT_ERROR

    records = load_records(args.in_path)
    if not records:
        print(f"error: no valid exp41 records in {args.in_path}", file=sys.stderr)
        return EXIT_ERROR

    if args.trials is not None:
        for arm in ARMS:
            n = sum(1 for r in records if r.get("arm") == arm)
            if n != args.trials:
                print(f"warning: arm {arm} has {n} records, expected {args.trials}", file=sys.stderr)

    result = analyze(records, zero_sd_floor=args.zero_sd_floor)
    body = render_markdown(result, heading_suffix=args.heading_suffix)
    print(body)
    if args.out_path is not None:
        _emit_to_doc(args.out_path, body)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
