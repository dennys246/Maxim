#!/usr/bin/env python3
"""Exp 37 cross-session graduation analyzer (PR #4 of 6).

Reads `37_results.jsonl` emitted by ``scripts/benchmark_cross_session.py``
and computes the pre-registered acceptance criteria for the Exp 37
graduation gate. Emits a Markdown block ready to append to
``docs/experiments/37_cross_session_graduation.md`` as the Results section.

Rules implemented (verbatim from
[37_cross_session_graduation.md](docs/experiments/37_cross_session_graduation.md);
pre-reg amendments 2026-05-31 (per-action rate) and 2026-06-XX
(exp37_metric_pivot.md Path 2 → positive-approach-engagement-fraction)):

  - **Primary (mean-shift-in-SD-units, pre-reg amendment 2026-06-05):**
    ``(B - A) / A.sd ≥ +1`` SD for positive_approach_engagement_fraction
    (predicted direction is increase). Replaces the legacy
    "B mean OUTSIDE A's empirical percentile band" rule because bounded
    metrics that frequently hit their ceiling/floor (LLM-AUT positive-
    approach piles up at 1.0) make the percentile-band predicate
    structurally impossible. The SD-shift is the same statistical shape
    already used for corroborating metrics. Zero-SD fallback: pass on
    directional sign + non-zero shift.
  - **Robustness (legacy primary, decrease):** ``(B - A) / A.sd ≤ -1`` SD
    on per_action_failure_rate (same SD-shift test, opposite direction).
    Divergence between primary and robustness flags potential substrate
    weirdness (warm_self bias without touch reduction, or vice versa).
  - **Isolation:** Arm C mean WITHIN Arm A's [2.5%, 97.5%] band. If Arm C
    also shows shrinkage, primary FAILS with the "general caution" confound.
  - **Corroborating (≥1 of 3 must hit):** affordance_safe_fraction,
    tool_class_diversity, time_to_safe_steady_state_turns. Each must shift
    in the predicted direction by ≥1 SD of Arm A baseline.
  - **Secondary (≥1 ablation):** B-wire-a-off, B-wire-1-off, B-nac-bias-off.
    Each must shrink Arm B's delta toward Arm A by ≥1 SD of Arm A baseline.

The analyzer DOES NOT auto-append to the experiment doc and DOES NOT
update `behavioral_graduation_candidates.md` — both are human steps (the
graduation status flip is PR #6).

**Tertiary criterion (local-model replication):** intentionally NOT
computed by this analyzer. Pre-reg names tertiary as "informational only —
NOT gating." If a future run feeds local-qwen records to the same JSONL,
gate them by the `model` field and produce a side-by-side comparison
block in a follow-up PR.

**Dual-source contract for the verdict matrix:** the same graduation
matrix is encoded in three places — the pre-reg
([37_cross_session_graduation.md] §"Graduation paths" +
§"Falsification conditions"), the protocol
([protocols/37_cross_session_graduation_reproduction.md] §D), and this
file's :func:`overall_verdict` + :data:`EXIT_CODE_FOR_LABEL`. The smoke
test ``test_verdict_labels_appear_in_preregistration_and_protocol``
asserts the analyzer's labels appear verbatim in the protocol §D table
— that's the structural enforcement that keeps the three in lockstep.
Bumping a label requires updating all three in the same commit.

NOT RUN BY CI — read-only post-processing; no LLM, no cost. The smoke test
at `tests/behavioral/test_exp37_analyzer_smoke.py` covers the logic.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make `maxim` importable when run from the repo root (kept symmetric with the
# harness; not required by the analyzer itself).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ─── Constants ───────────────────────────────────────────────────────────
#
# Schema version, experiment id, scenarios, and ARMS are imported from the
# harness as the single source of truth (B1 fold). The harness module lives
# under `scripts/` and is loaded via importlib because `scripts/` is not a
# package. If the harness bumps SCHEMA_VERSION (e.g., 1.0 → 1.1 when a new
# corroborating metric ships) the analyzer's strict-schema check tracks
# automatically — no separate bump required.

ANALYZER_VERSION = "1.0"


def _load_harness_constants() -> dict[str, Any]:
    """Load the harness module by file path and return the constants the
    analyzer depends on. Falls back to hardcoded defaults if the harness
    isn't reachable (defensive — covers test isolation contexts)."""
    import importlib.util

    harness_path = _REPO_ROOT / "scripts" / "benchmark_cross_session.py"
    if not harness_path.exists():  # pragma: no cover — defensive
        return {
            "SCHEMA_VERSION": "1.0",
            "EXPERIMENT_ID": "exp37_cross_session_graduation",
            "SCENARIOS": ("fire_pit", "sharp_rock"),
            "ARMS": ("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        }
    spec = importlib.util.spec_from_file_location("exp37_harness_for_analyzer", harness_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("exp37_harness_for_analyzer", module)
    spec.loader.exec_module(module)
    return {
        "SCHEMA_VERSION": module.SCHEMA_VERSION,
        "EXPERIMENT_ID": module.EXPERIMENT_ID,
        "SCENARIOS": module.SCENARIOS,
        "ARMS": module.ARMS,
    }


_HARNESS = _load_harness_constants()
SUPPORTED_SCHEMA_VERSION: str = _HARNESS["SCHEMA_VERSION"]
EXPERIMENT_ID: str = _HARNESS["EXPERIMENT_ID"]
SCENARIOS: tuple[str, ...] = _HARNESS["SCENARIOS"]
HARNESS_ARMS: tuple[str, ...] = _HARNESS["ARMS"]

# Per pre-reg amendment 2026-06-XX (exp37_metric_pivot.md Path 2): primary
# metric pivoted to ``positive_approach_engagement_fraction`` — the share of
# the agent's on-target fire engagement that goes to the positive-approach
# affordance ``fire_pit_warm_self``. The substrate-transfer hypothesis under
# LLM-AUT predicts B is biased toward warm_self when it engages with fire,
# producing a HIGHER ratio than A's exploration-driven engagement. The
# 2026-05-31 swap to per_action_failure_rate is retained as the robustness
# cross-check (its variance-survival math is degenerate when Arm A never
# touches the fire — the failure mode this pivot resolves).
#
# DIRECTION FLIP: with the new primary, "better" is HIGHER (warm_self
# preference is the predicted Arm B signal). _compute_primary_isolation
# takes the direction as a kwarg so old callers (robustness check on the
# legacy metric) keep "decrease" semantics.
PRIMARY_METRIC = "positive_approach_engagement_fraction"
PRIMARY_METRIC_DIRECTION = "increase"
ROBUSTNESS_METRIC = "per_action_failure_rate"
ROBUSTNESS_METRIC_DIRECTION = "decrease"
TURN_COUNT_FIELD = "turn_count_binned"

# Corroborating metrics — (record_field, predicted_direction, label, none_handling).
# direction: "decrease" means Arm B should be < Arm A (less = better);
# "increase" means Arm B should be > Arm A (more = better).
# none_handling: how missing/None per-trial values map onto a numeric sample.
#   "drop"  — None values are filtered out before stats (default).
#   "censor_to_max" — None means "never reached the threshold"; censor to
#     the trial's ``turn_count_binned + 1`` so it's strictly worse than any
#     trial that did reach the threshold. This is the pre-reg's intent for
#     time-to-safe-steady-state: "never reached" must count as the strongest
#     failure on Arm A, not as missing data that erases the metric.
CORROBORATING_METRICS: tuple[tuple[str, str, str, str], ...] = (
    (
        "affordance_preference_safe_fraction",
        "increase",
        "Affordance-preference safe-fraction (safe-on-target / on-target total)",
        "drop",
    ),
    (
        "tool_class_diversity",
        "decrease",
        "Tool-class diversity (fewer dead-end tools tried)",
        "drop",
    ),
    (
        "time_to_safe_steady_state_turns",
        "decrease",
        "Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1)",
        "censor_to_max",
    ),
    # exp37_metric_pivot.md (Path 2): action-index at which the agent
    # first calls ``fire_pit_warm_self``. Substrate transfer predicts B's
    # first warm_self comes EARLIER than A's exploration-driven discovery
    # (or at least no later), giving a meaningful corroborating signal
    # for the new primary. ``censor_to_max`` keeps "never reached" sorting
    # as strictly worse than any reached session (parallel to
    # time_to_safe_steady_state_turns handling).
    (
        "time_to_first_warm_self_action",
        "decrease",
        "Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1)",
        "censor_to_max",
    ),
)

ABLATION_ARMS: tuple[str, ...] = ("B-wire-a-off", "B-wire-1-off", "B-nac-bias-off")
ABLATION_LABELS: dict[str, str] = {
    "B-wire-a-off": "Wire-A annotation off",
    "B-wire-1-off": "Wire 1 variance annotation off",
    "B-nac-bias-off": "NAc reward bias zeroed",
}


# ─── Errors ──────────────────────────────────────────────────────────────


class AnalyzerError(RuntimeError):
    """Raised for any analyzer-fatal condition (schema mismatch, missing data)."""


# ─── IO ──────────────────────────────────────────────────────────────────


_REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "trial_pair_id",
    "arm",
    "scenario",
    "experiment",
)


def load_records(in_path: Path, *, expected_schema: str) -> list[dict[str, Any]]:
    """Read newline-delimited records, validate schema version, return all.

    Schema-version handling (I5 fold): the harness emits BOTH
    ``_format_version`` (canonical per CLAUDE.md persistence invariant) and
    ``schema_version`` (experiment-scoped mirror) in lockstep. We accept
    either as the version indicator; if BOTH are present and disagree, that
    is itself a harness bug worth surfacing — refuse the file rather than
    silently choosing one.

    I4 fold: required fields are validated per-record so a malformed JSONL
    surfaces as ``AnalyzerError`` with file:line context rather than a
    bare ``KeyError`` later in ``group_records``.
    """
    if not in_path.exists():
        raise AnalyzerError(f"Input file does not exist: {in_path}")
    records: list[dict[str, Any]] = []
    seen_versions: set[str] = set()
    for ln, line in enumerate(in_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalyzerError(f"{in_path}:{ln} not valid JSON: {exc}") from exc

        # Per-record required-field check (I4 fold).
        missing_fields = [f for f in _REQUIRED_RECORD_FIELDS if f not in rec]
        if missing_fields:
            raise AnalyzerError(
                f"{in_path}:{ln} record missing required field(s) {missing_fields!r}. "
                f"This is either a harness bug or a hand-edited JSONL — refuse rather than guess."
            )

        # Schema-version dual-field consistency (I5 fold).
        fv = rec.get("_format_version")
        sv = rec.get("schema_version")
        if fv is not None and sv is not None and str(fv) != str(sv):
            raise AnalyzerError(
                f"{in_path}:{ln} _format_version ({fv!r}) and schema_version ({sv!r}) "
                f"disagree; harness must emit both in lockstep per the harness "
                f"contract. Refuse rather than guess which is canonical."
            )
        version = fv or sv or "0.x"
        seen_versions.add(str(version))

        if rec.get("experiment") != EXPERIMENT_ID:
            raise AnalyzerError(
                f"{in_path}:{ln} unexpected experiment {rec.get('experiment')!r}; expected {EXPERIMENT_ID!r}"
            )
        records.append(rec)
    if not records:
        raise AnalyzerError(f"{in_path} contains zero records.")
    mismatched = {v for v in seen_versions if v != expected_schema}
    if mismatched:
        raise AnalyzerError(
            f"{in_path} carries schema version(s) {sorted(mismatched)!r}; "
            f"analyzer supports {expected_schema!r}. Either bump the analyzer "
            f"in lockstep with the harness or re-run trials on the matching version."
        )
    return records


def group_records(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group by ``(scenario, arm)`` so per-criterion stats can iterate."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        out[(rec["scenario"], rec["arm"])].append(rec)
    # Stable per-group ordering by trial_pair_id so paired-trial math
    # reads samples in the same trial order across arms.
    for key in out:
        out[key].sort(key=lambda r: r["trial_pair_id"])
    return out


def assert_design_complete(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    arms: tuple[str, ...],
    scenarios: tuple[str, ...],
    expected_trials: int,
) -> None:
    """Refuse to compute on incomplete data — silent missing pairs would
    poison the variance-survival check."""
    missing: list[str] = []
    for scenario in scenarios:
        for arm in arms:
            records = grouped.get((scenario, arm), [])
            if len(records) < expected_trials:
                missing.append(f"{scenario}:{arm} has {len(records)} records, expected {expected_trials}")
            elif len(records) > expected_trials:
                missing.append(
                    f"{scenario}:{arm} has {len(records)} records, MORE than expected "
                    f"{expected_trials} — duplicates? check append-only JSONL"
                )
            # Verify trial_pair_ids are 1..expected_trials.
            ids = {r["trial_pair_id"] for r in records}
            if ids != set(range(1, expected_trials + 1)):
                missing.append(
                    f"{scenario}:{arm} trial_pair_id set {sorted(ids)} != expected "
                    f"{list(range(1, expected_trials + 1))}"
                )
    if missing:
        raise AnalyzerError(
            "Incomplete design — refusing to produce a verdict on partial data:\n  - " + "\n  - ".join(missing)
        )


# ─── Statistical helpers ─────────────────────────────────────────────────


def _extract_metric(
    records: list[dict[str, Any]],
    field: str,
    *,
    none_handling: str = "drop",
) -> list[float]:
    """Pull a per-trial metric. Missing/null values map onto floats per
    the ``none_handling`` strategy:

      - ``"drop"`` — None values become NaN (caller filters them out).
      - ``"censor_to_max"`` — None means "the threshold was never met";
        censor each None to ``turn_count_binned + 1`` for that trial so
        it ranks strictly worse than any trial that reached the threshold.
        Falls back to NaN if the trial has no ``turn_count_binned`` field.

    Used by C1 fold: ``time_to_safe_steady_state_turns`` is censored so
    Arm A "never reached" trials count as the maximum-failure outcome,
    not as missing data that erases the strongest signal.
    """
    out: list[float] = []
    for r in records:
        v = r.get(field)
        if v is None:
            if none_handling == "censor_to_max":
                cap = r.get(TURN_COUNT_FIELD)
                if cap is None:
                    out.append(float("nan"))
                else:
                    out.append(float(cap) + 1.0)
            else:
                out.append(float("nan"))
        else:
            out.append(float(v))
    return out


def _filter_finite(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def _safe_mean(values: list[float]) -> float | None:
    finite = _filter_finite(values)
    return statistics.fmean(finite) if finite else None


def _safe_sd(values: list[float]) -> float | None:
    finite = _filter_finite(values)
    if len(finite) < 2:
        return None
    return statistics.stdev(finite)


def _percentile_band(values: list[float]) -> tuple[float, float] | None:
    """Empirical [2.5%, 97.5%] band via linear interpolation.

    With N=5 trials the band is essentially the min-to-max range; with
    larger N it tightens. The pre-reg's "95th-percentile band" language
    is the conventional ``[p2.5, p97.5]`` two-sided band.
    """
    finite = sorted(_filter_finite(values))
    if len(finite) < 2:
        return None
    # statistics.quantiles is exclusive on 0/100 — use n=40 to get 2.5%/97.5%
    # boundaries (indexes 0 and 38). For N<3 this is degenerate, hence the
    # check above. For N=5, linear interpolation lands close to min/max which
    # matches the conservative reading of the pre-reg.
    qs = statistics.quantiles(finite, n=40, method="inclusive")
    return (qs[0], qs[-1])


def _is_structurally_absent(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    scenario: str,
    metric: str,
    *,
    arms: tuple[str, ...],
) -> bool:
    """True iff ``metric`` is IDENTICAL across every arm's every trial in
    ``scenario`` — i.e. zero variance everywhere, so nothing the agent did ever
    moved it.

    Such a metric was never exercised in this world (the canonical case is
    ``positive_approach_engagement_fraction`` in a scenario with no
    positive-approach affordance — structurally 0 for every arm). A primary
    FAIL on a structurally-absent metric is a FALSE NEGATIVE: it implies the
    hypothesis was tested and rejected when in fact it was never tested. The
    caller reports **N/A / inconclusive** instead of FAIL.

    Folds the Exp 37 ``sharp_rock`` artifact (the analyzer reported FAIL on
    ``positive_approach_engagement_fraction``, a fire-pit metric inapplicable to
    sharp_rock) — counter_prior_substrate_experiment.md §6.3.

    Returns False when there are <2 finite samples (let the normal
    insufficient-data path handle it) — absence is a claim about constancy
    across a populated design, not about missing data.
    """
    all_vals: list[float] = []
    for arm in arms:
        all_vals.extend(_filter_finite(_extract_metric(grouped.get((scenario, arm), []), metric)))
    if len(all_vals) < 2:
        return False
    first = all_vals[0]
    return all(v == first for v in all_vals)


# ─── Per-criterion computation ───────────────────────────────────────────


@dataclass(frozen=True)
class ScenarioVerdict:
    """Per-scenario verdict (SHAPE-FROZEN at 1.0 (CC3)).

    Adding required fields post-1.0 is a breaking change for any consumer
    that constructs verdicts directly (e.g., PR #6's status-flip code may
    branch on the named fields). Adding optional fields with defaults at
    the end is non-breaking.
    """

    scenario: str
    primary_pass: bool
    isolation_pass: bool
    primary_pass_robustness: bool  # per_action_failure_rate cross-check
    corroborating_hits: int
    corroborating_wrong_direction: int  # C2: count of metrics with ≥1 SD AGAINST prediction
    corroborating_details: list[dict[str, Any]]
    secondary_hits: int
    secondary_details: list[dict[str, Any]]
    a_mean: float | None
    a_sd: float | None
    a_band: tuple[float, float] | None
    b_mean: float | None
    c_mean: float | None
    notes: list[str]
    # Optional descriptive corroborating block — DESCRIPTIVE only (NOT pre-reg
    # gated). Populated when ``FAILURE_CLASS[scenario]`` declares a non-empty
    # ``direct_approach_tools`` (cradle_activation_fixes.md P2). Shape:
    # ``{"a_mean": float|None, "b_mean": float|None, "delta": float|None,
    # "predicted_direction": "increase"|"same_or_higher", "note": str|None}``.
    # None for scenarios with no approach affordance (e.g. sharp_rock).
    approach_descriptive: dict[str, Any] | None = None
    # True when PRIMARY_METRIC is structurally absent (identical across every
    # arm/trial — never exercised). Such a scenario reports the primary as
    # N/A / inconclusive, NOT FAIL, and is excluded from the overall-verdict
    # gating (counter_prior_substrate_experiment.md §6.3 — folds the Exp 37
    # sharp_rock artifact). Optional/defaulted per CC3 (non-breaking add).
    primary_structural_absence: bool = False


@dataclass(frozen=True)
class AnalysisResult:
    """Top-level analyzer return (SHAPE-FROZEN at 1.0 (CC3)).

    PR #6 (graduation status flip) reads this struct: ``overall_label``
    drives the [[behavioral_graduation_candidates.md]] row 1 Earned/
    Reframed/Partial flip; ``overall_rationale`` becomes the audit-trail
    citation. Markdown is the human-readable report appended to the
    experiment doc.
    """

    scenarios: list[ScenarioVerdict]
    overall_label: str
    overall_rationale: str
    markdown: str
    # Exp 38 counter-prior interaction verdict — populated only when BOTH the
    # consistent control and deceptive scenarios are present (else None, and the
    # overall_label comes from the legacy Exp 37 ``overall_verdict``). Optional/
    # defaulted per CC3 (non-breaking add).
    counter_prior: CounterPriorVerdict | None = None


PRIMARY_SD_SHIFT_THRESHOLD = 1.0


def _compute_primary_isolation(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    scenario: str,
    metric: str,
    *,
    direction: str = "decrease",
    sd_threshold: float = PRIMARY_SD_SHIFT_THRESHOLD,
) -> tuple[bool, bool, float | None, float | None, tuple[float, float] | None, float | None, list[str]]:
    """Returns ``(primary_pass, isolation_pass, a_mean, a_sd, a_band, b_mean, c_mean, notes)``.

    **Pre-reg amendment 2026-06-05 (`exp37_sd_shift.md`)**: the primary
    rule swapped from "Arm B mean outside Arm A's empirical percentile
    band" to "Arm B mean differs from Arm A's mean by ≥ ``sd_threshold``
    × A's SD in the predicted direction." Rationale: the validation
    smoke (5 Arm A trials, 2026-06-05) showed
    ``positive_approach_engagement_fraction`` piles up at the [0, 1]
    ceiling under LLM-AUT — 3 of 5 trials at 1.0. The empirical
    percentile band collapses to [floor, ceiling] and the "B outside
    band" predicate is structurally impossible for any bounded metric
    that frequently hits its bound. The mean-shift-in-SD-units test
    handles bounded distributions cleanly and is the same statistical
    shape already used for corroborating metrics (precedent: per pre-reg
    §Corroborating, "≥1 SD effect on the Arm A baseline").

    ``direction`` controls the predicted side:

    - ``"decrease"`` (legacy, robustness check): ``(B - A) / A.sd ≤ -threshold``.
    - ``"increase"`` (pivot, primary): ``(B - A) / A.sd ≥ threshold``.

    Zero-SD fallback: if Arm A's SD is None or 0 (all trials produced
    the same value — a true zero-variance case, e.g., the metric is
    structurally constrained), pass on directional sign + non-zero
    shift. Matches the existing corroborating zero-SD fallback (I2 fold).

    Isolation still uses the two-sided empirical percentile band
    ([p2.5, p97.5]); the "general caution" confound is direction-agnostic
    and tolerant of A piling up at one bound (isolation_pass requires C
    INSIDE A's range, which is naturally true when A's range is narrow).
    """
    notes: list[str] = []
    a_vals = _extract_metric(grouped.get((scenario, "A"), []), metric)
    b_vals = _extract_metric(grouped.get((scenario, "B"), []), metric)
    c_vals = _extract_metric(grouped.get((scenario, "C"), []), metric)

    a_mean = _safe_mean(a_vals)
    a_sd = _safe_sd(a_vals)
    a_band = _percentile_band(a_vals)
    b_mean = _safe_mean(b_vals)
    c_mean = _safe_mean(c_vals)

    if a_mean is None or b_mean is None or a_band is None:
        notes.append(f"Insufficient data for {scenario}: a_mean={a_mean}, b_mean={b_mean}, a_band={a_band}")
        return (False, False, a_mean, a_sd, a_band, b_mean, c_mean, notes)

    # Primary: mean shift in SD units (pre-reg amendment 2026-06-05).
    delta = b_mean - a_mean
    if a_sd is None or a_sd == 0:
        # Zero-variance Arm A: any non-zero directional shift passes
        # (this is the I2-fold zero-SD fallback applied to primary).
        if direction == "increase":
            primary_pass = delta > 0
        else:  # decrease
            primary_pass = delta < 0
        if primary_pass:
            notes.append(
                f"Zero-SD on Arm A for {scenario} primary: passing on directional "
                f"shift (delta={delta:+.4f}, direction={direction!r})."
            )
    else:
        delta_sd = delta / a_sd
        if direction == "increase":
            primary_pass = delta_sd >= sd_threshold
        else:  # decrease
            primary_pass = delta_sd <= -sd_threshold

    isolation_pass = True
    if c_mean is None:
        notes.append(f"Arm C has no data for {scenario}; isolation check undecidable.")
        isolation_pass = False
    elif not (a_band[0] <= c_mean <= a_band[1]):
        isolation_pass = False
        notes.append(
            f"Arm C mean {c_mean:.4f} for {scenario} falls outside Arm A's band "
            f"[{a_band[0]:.4f}, {a_band[1]:.4f}] — 'general caution' confound."
        )

    return (primary_pass, isolation_pass, a_mean, a_sd, a_band, b_mean, c_mean, notes)


def _compute_corroborating(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    scenario: str,
) -> list[dict[str, Any]]:
    """Returns one dict per corroborating metric.

    ``pass`` — True when (B − A) / A.sd ≥ 1.0 in the predicted direction.
    ``wrong_direction`` — True when (B − A) / A.sd ≥ 1.0 in the OPPOSITE
    direction (i.e. the bio claim is contradicted by ≥1 SD). C2 fold: the
    pre-reg §Corroborating says "if ALL THREE diverge from prediction while
    primary still hits, that's evidence the primary is a measurement
    artifact — investigate before claiming EARNED rather than ignore."
    The overall verdict reader uses this flag to upgrade an otherwise
    EARNED-footnoted outcome to PARTIAL — investigation gate.

    Zero-SD fallback (I2 fold): when ``a_sd == 0`` but ``a_mean`` and
    ``b_mean`` are both present, PASS when the delta moves in the predicted
    direction by ≥ 1 unit of the natural metric — this catches integer
    metrics (``tool_class_diversity`` is integer-valued; N=5 trials of the
    same value is plausible) where the SD collapses but the signal is real.
    """
    out: list[dict[str, Any]] = []
    for field, direction, label, none_handling in CORROBORATING_METRICS:
        a_vals = _extract_metric(grouped.get((scenario, "A"), []), field, none_handling=none_handling)
        b_vals = _extract_metric(grouped.get((scenario, "B"), []), field, none_handling=none_handling)
        a_mean = _safe_mean(a_vals)
        a_sd = _safe_sd(a_vals)
        b_mean = _safe_mean(b_vals)

        record: dict[str, Any] = {
            "field": field,
            "label": label,
            "direction": direction,
            "a_mean": a_mean,
            "a_sd": a_sd,
            "b_mean": b_mean,
            "delta": None,
            "delta_in_sd_units": None,
            "pass": False,
            "wrong_direction": False,
            "note": None,
        }
        if a_mean is None or b_mean is None:
            record["note"] = "Insufficient data (a_mean / b_mean missing)"
            out.append(record)
            continue

        delta = b_mean - a_mean
        record["delta"] = delta

        # Predicted-direction sign: +1 for "increase", -1 for "decrease".
        predicted_sign = 1.0 if direction == "increase" else -1.0
        # Actual delta-direction sign (when delta is non-zero).
        actual_sign = (1.0 if delta > 0 else -1.0) if delta != 0 else 0.0

        if a_sd is None or a_sd == 0:
            # I2 fallback: integer metric with degenerate SD. Pass on
            # directional sign + non-zero shift.
            if delta == 0:
                record["note"] = "Zero SD on Arm A AND zero shift on B (degenerate)."
            elif actual_sign == predicted_sign:
                record["pass"] = True
                record["note"] = (
                    "Zero-SD fallback: Arm A baseline collapsed but shift moves in the predicted direction; PASS."
                )
            else:
                record["wrong_direction"] = True
                record["note"] = "Zero-SD fallback: shift moves AGAINST the predicted direction."
            out.append(record)
            continue

        delta_sd_units = delta / a_sd
        record["delta_in_sd_units"] = delta_sd_units

        if direction == "increase":
            record["pass"] = delta_sd_units >= 1.0
            record["wrong_direction"] = delta_sd_units <= -1.0
        else:  # decrease
            record["pass"] = delta_sd_units <= -1.0
            record["wrong_direction"] = delta_sd_units >= 1.0

        out.append(record)
    return out


def _compute_secondary(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    scenario: str,
    metric: str,
) -> list[dict[str, Any]]:
    """For each ablation, check whether Arm B's delta shrinks toward Arm A by ≥1 SD."""
    a_vals = _extract_metric(grouped.get((scenario, "A"), []), metric)
    b_vals = _extract_metric(grouped.get((scenario, "B"), []), metric)
    a_mean = _safe_mean(a_vals)
    a_sd = _safe_sd(a_vals)
    b_mean = _safe_mean(b_vals)

    out: list[dict[str, Any]] = []
    for arm in ABLATION_ARMS:
        ab_vals = _extract_metric(grouped.get((scenario, arm), []), metric)
        ab_mean = _safe_mean(ab_vals)
        record: dict[str, Any] = {
            "arm": arm,
            "label": ABLATION_LABELS[arm],
            "a_mean": a_mean,
            "b_mean": b_mean,
            "ablated_mean": ab_mean,
            "shrinkage": None,
            "shrinkage_in_sd_units": None,
            "pass": False,
            "note": None,
        }
        if a_mean is None or b_mean is None or ab_mean is None or a_sd is None or a_sd == 0:
            record["note"] = "Insufficient data for ablation comparison."
            out.append(record)
            continue

        # Shrinkage toward A (per pre-reg §Secondary): the ablated arm's
        # mean is CLOSER to A's mean than B's mean is, AND on the SAME
        # SIDE of A as B (overshoot past A doesn't count — that's an
        # opposite-direction effect, not a shrinkage). I1 fold.
        signed_b = b_mean - a_mean
        signed_ablated = ab_mean - a_mean
        same_side = (signed_b * signed_ablated) > 0
        delta_b = abs(signed_b)
        delta_ablated = abs(signed_ablated)
        shrinkage = delta_b - delta_ablated
        shrinkage_sd = shrinkage / a_sd if a_sd else 0.0
        record["shrinkage"] = shrinkage
        record["shrinkage_in_sd_units"] = shrinkage_sd
        record["overshoot"] = (not same_side) and ab_mean != a_mean
        record["pass"] = same_side and shrinkage_sd >= 1.0
        if record["overshoot"]:
            record["note"] = (
                f"Ablation overshoots past Arm A baseline (B side: {signed_b:+.4f}, "
                f"ablated side: {signed_ablated:+.4f}). Opposite-direction effect, "
                f"not shrinkage — does NOT count as secondary-criterion PASS."
            )
        out.append(record)
    return out


def evaluate_scenario(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    scenario: str,
    *,
    arms: tuple[str, ...] = HARNESS_ARMS,
) -> ScenarioVerdict:
    primary_pass, isolation_pass, a_mean, a_sd, a_band, b_mean, c_mean, notes = _compute_primary_isolation(
        grouped, scenario, PRIMARY_METRIC, direction=PRIMARY_METRIC_DIRECTION
    )

    # Structural-absence detection (counter_prior_substrate_experiment.md §6.3):
    # if PRIMARY_METRIC never varied across any arm, the primary FAIL above is a
    # false negative — report N/A and exclude the scenario from overall gating.
    primary_structural_absence = _is_structurally_absent(grouped, scenario, PRIMARY_METRIC, arms=arms)
    if primary_structural_absence:
        notes.append(
            f"PRIMARY_METRIC {PRIMARY_METRIC!r} is structurally absent for {scenario} "
            f"(identical across every arm/trial — the approach affordance was never "
            f"exercised). Reporting primary as N/A / inconclusive, not FAIL; this "
            f"scenario is excluded from the overall-verdict gating."
        )

    # Robustness cross-check via the legacy per-action failure rate.
    # exp37_metric_pivot.md (Path 2): the OLD primary is retained as the
    # robustness signal — divergence between the new primary's verdict
    # and the legacy primary's verdict flags that the substrate-driven
    # warm_self preference shift may not correspond to a touch-avoidance
    # shift. Pre-reg amendment notes both signals should be inspected.
    primary_rob_pass, _, _, _, rob_band, b_rob_mean, _, rob_notes = _compute_primary_isolation(
        grouped, scenario, ROBUSTNESS_METRIC, direction=ROBUSTNESS_METRIC_DIRECTION
    )
    if primary_pass != primary_rob_pass:
        notes.append(
            f"Primary / robustness divergence on {scenario}: "
            f"positive-approach-engagement primary={primary_pass} "
            f"vs per-action-failure-rate robustness={primary_rob_pass}. "
            f"Substrate may be biasing toward warm_self without reducing touch "
            f"(or vice versa). Investigate before claiming the verdict "
            f"(per protocol §1)."
        )
    notes.extend(rob_notes)

    corroborating = _compute_corroborating(grouped, scenario)
    secondary = _compute_secondary(grouped, scenario, PRIMARY_METRIC)
    approach_descriptive = _compute_descriptive_approach(grouped, scenario)

    return ScenarioVerdict(
        scenario=scenario,
        primary_pass=primary_pass,
        isolation_pass=isolation_pass,
        primary_pass_robustness=primary_rob_pass,
        corroborating_hits=sum(1 for r in corroborating if r["pass"]),
        corroborating_wrong_direction=sum(1 for r in corroborating if r.get("wrong_direction")),
        corroborating_details=corroborating,
        secondary_hits=sum(1 for r in secondary if r["pass"]),
        secondary_details=secondary,
        a_mean=a_mean,
        a_sd=a_sd,
        a_band=a_band,
        b_mean=b_mean,
        c_mean=c_mean,
        notes=notes,
        approach_descriptive=approach_descriptive,
        primary_structural_absence=primary_structural_absence,
    )


def _compute_descriptive_approach(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    scenario: str,
) -> dict[str, Any] | None:
    """Compute the ``fire_approach_action_count`` descriptive corroborating
    metric for a scenario (cradle_activation_fixes.md P2).

    Returns ``None`` when the scenario has no approach affordance (e.g.
    ``sharp_rock`` has empty ``direct_approach_tools`` by design). Otherwise
    returns a dict with A mean, B mean, and the predicted direction
    ("same_or_higher" — substrate transfer predicts B's positive-approach
    count is unchanged or increases relative to A while
    ``failure_class_action_count`` drops).

    NOT a pre-reg gate — the analyzer renders this block descriptively
    in the markdown but does NOT promote it to pass/fail in the verdict.
    """
    # Probe Arm A records for the field — when the harness's
    # FAILURE_CLASS table declares no approach affordances for a scenario,
    # ``compute_metrics`` still emits ``fire_approach_action_count: 0`` for
    # every record, so presence is universal. The signal is whether any A
    # or B record shows a non-zero count.
    a_vals = _extract_metric(grouped.get((scenario, "A"), []), "fire_approach_action_count")
    b_vals = _extract_metric(grouped.get((scenario, "B"), []), "fire_approach_action_count")

    # Asymmetric design: scenarios with no approach affordance always emit
    # 0 — render nothing rather than display a degenerate row. Detect by:
    # if BOTH a_vals and b_vals are all zero (or empty), suppress.
    def _all_zero(vs: list[float]) -> bool:
        return len(vs) == 0 or all(v == 0 for v in vs)

    if _all_zero(a_vals) and _all_zero(b_vals):
        return None

    a_mean = _safe_mean(a_vals)
    b_mean = _safe_mean(b_vals)
    delta = (b_mean - a_mean) if (a_mean is not None and b_mean is not None) else None

    note: str | None = None
    if a_mean is None or b_mean is None:
        note = "Insufficient data (a_mean / b_mean missing)"
    elif delta is not None and delta < 0:
        note = (
            "Δ < 0: Arm B shows FEWER positive-approach actions than Arm A. "
            "Substrate transfer predicts the positive edge ('fire = warm') is preserved; "
            "investigate whether the agent avoided fire entirely on B (general caution) "
            "or specifically lost the approach association."
        )

    return {
        "a_mean": a_mean,
        "b_mean": b_mean,
        "delta": delta,
        "predicted_direction": "same_or_higher",
        "note": note,
    }


# ─── Exp 38 counter-prior interaction ────────────────────────────────────
#
# counter_prior_substrate_experiment.md §5. These metrics are FROZEN in the
# pre-registration doc (docs/experiments/38_counter_prior_substrate.md) before
# the first fire — do NOT re-tune them post-hoc (the Exp 37 sharp_rock
# metric-drift is the cautionary tale).

COUNTER_PRIOR_CONSISTENT_SCENARIO = "fire_pit"  # matched control — warm_self safe
COUNTER_PRIOR_DECEPTIVE_SCENARIO = "deceptive_fire"  # counter-prior — warm_self harmful
# Warm_self-preference channel (decoupled from the safe/approach label so it
# measures warm_self engagement in BOTH scenarios — see the harness's
# ``warm_self_engagement_fraction`` derivation).
COUNTER_PRIOR_METRIC = "warm_self_engagement_fraction"
FIRST_CONTACT_FIELD = "first_contact_warm_self"
# Interaction must be negative by ≥ this many SD of the pooled Arm-A baseline.
INTERACTION_SD_SHIFT_THRESHOLD = 1.0


@dataclass(frozen=True)
class CounterPriorVerdict:
    """Exp 38 counter-prior interaction verdict (SHAPE-FROZEN at 1.0 (CC3)).

    The top-level criterion for Exp 38 — supersedes the per-scenario Exp 37
    gating when both the consistent control and the deceptive counter-prior
    scenarios are present. See counter_prior_substrate_experiment.md §5.
    """

    # Interaction primary: Δ_deceptive(B−A) − Δ_consistent(B−A) on
    # warm_self-engagement-fraction, normalized by pooled Arm-A SD.
    delta_deceptive: float | None
    delta_consistent: float | None
    interaction: float | None
    pooled_a_sd: float | None
    interaction_sd_units: float | None
    interaction_pass: bool
    # First-contact isolation: P(warm_self on first contact) per arm/scenario.
    p_a_deceptive: float | None
    p_b_deceptive: float | None
    p_a_consistent: float | None
    p_b_consistent: float | None
    first_contact_deceptive_drop: float | None  # p_a_dec − p_b_dec (predicted > 0)
    first_contact_consistent_drop: float | None  # p_a_con − p_b_con (predicted ≈ 0)
    first_contact_interaction: float | None  # dec_drop − con_drop (predicted > 0)
    first_contact_pass: bool
    # Secondary: ablation reversion on the deceptive scenario (≥1 must revert).
    ablation_details: list[dict[str, Any]]
    ablation_hits: int
    # Derived flags feeding the verdict tree.
    avoids_both: bool  # general-caution confound: B reduces warm_self everywhere
    verdict_label: str
    verdict_rationale: str
    notes: list[str]


def _extract_bool_proportion(records: list[dict[str, Any]], field: str) -> float | None:
    """Proportion of records whose ``field`` is True, over the non-None values.

    Returns None when every value is None (the entity was never engaged in any
    trial — no first-contact signal). Used for ``first_contact_warm_self``.
    """
    vals = [bool(r[field]) for r in records if r.get(field) is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def compute_counter_prior(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    consistent: str = COUNTER_PRIOR_CONSISTENT_SCENARIO,
    deceptive: str = COUNTER_PRIOR_DECEPTIVE_SCENARIO,
    metric: str = COUNTER_PRIOR_METRIC,
    sd_threshold: float = INTERACTION_SD_SHIFT_THRESHOLD,
) -> CounterPriorVerdict | None:
    """Compute the Exp 38 counter-prior interaction (§5). Returns None unless
    BOTH the consistent control and the deceptive scenario have data — so an
    Exp 37 run (no deceptive_fire) falls through to the legacy verdict path.

    FROZEN decision logic (pre-registered):

      * **Interaction primary** — ``interaction = Δ_dec(B−A) − Δ_con(B−A)`` on
        warm_self-fraction; ``interaction_pass`` iff
        ``interaction / pooled_A_sd ≤ −sd_threshold`` (B reduces warm_self
        *specifically* in the deceptive world). Zero-SD fallback: pass on a
        negative interaction.
      * **First-contact primary** — ``dec_drop = P_A(dec) − P_B(dec)`` and
        ``con_drop`` likewise; ``first_contact_pass`` iff ``dec_drop > 0`` AND
        ``(dec_drop − con_drop) > 0`` (avoidance is specific to deceptive).
      * **Secondary** — ≥1 ablation reverts B's deceptive warm_self toward A
        (reuses ``_compute_secondary``).

    Verdict tree (§5 outcome matrix):
      - interaction_pass AND first_contact_pass AND ablation_hits≥1
        → SUBSTRATE_MATTERS
      - interaction_pass AND first_contact_pass AND ablation_hits==0
        → VOID_NOT_ATTRIBUTABLE
      - avoids_both (B reduces warm_self in BOTH by ≥1 SD, interaction not
        specific) → VOID_GENERAL_CAUTION
      - otherwise (B did not specifically avoid the deceptive hearth)
        → DOMINANCE
    """
    if not grouped.get((deceptive, "A")) or not grouped.get((consistent, "A")):
        return None

    notes: list[str] = []

    a_dec = _extract_metric(grouped.get((deceptive, "A"), []), metric)
    b_dec = _extract_metric(grouped.get((deceptive, "B"), []), metric)
    a_con = _extract_metric(grouped.get((consistent, "A"), []), metric)
    b_con = _extract_metric(grouped.get((consistent, "B"), []), metric)

    a_dec_mean = _safe_mean(a_dec)
    b_dec_mean = _safe_mean(b_dec)
    a_con_mean = _safe_mean(a_con)
    b_con_mean = _safe_mean(b_con)
    a_dec_sd = _safe_sd(a_dec)
    a_con_sd = _safe_sd(a_con)

    delta_deceptive = (b_dec_mean - a_dec_mean) if (a_dec_mean is not None and b_dec_mean is not None) else None
    delta_consistent = (b_con_mean - a_con_mean) if (a_con_mean is not None and b_con_mean is not None) else None
    interaction = (
        (delta_deceptive - delta_consistent) if (delta_deceptive is not None and delta_consistent is not None) else None
    )
    # Pooled Arm-A baseline SD across both scenarios — the warm_self-fraction's
    # baseline variability under a fresh session (the normalizer the pre-reg
    # names "≥1 SD of the Arm-A baseline").
    pooled_a_sd = _safe_sd(_filter_finite(a_dec) + _filter_finite(a_con))

    interaction_sd_units: float | None = None
    interaction_pass = False
    if interaction is not None:
        if pooled_a_sd is None or pooled_a_sd == 0:
            interaction_pass = interaction < 0
            if interaction_pass:
                notes.append(
                    f"Zero-SD pooled Arm-A baseline; interaction passes on negative sign ({interaction:+.4f})."
                )
        else:
            interaction_sd_units = interaction / pooled_a_sd
            interaction_pass = interaction_sd_units <= -sd_threshold

    # First-contact proportions.
    p_a_dec = _extract_bool_proportion(grouped.get((deceptive, "A"), []), FIRST_CONTACT_FIELD)
    p_b_dec = _extract_bool_proportion(grouped.get((deceptive, "B"), []), FIRST_CONTACT_FIELD)
    p_a_con = _extract_bool_proportion(grouped.get((consistent, "A"), []), FIRST_CONTACT_FIELD)
    p_b_con = _extract_bool_proportion(grouped.get((consistent, "B"), []), FIRST_CONTACT_FIELD)
    fc_dec_drop = (p_a_dec - p_b_dec) if (p_a_dec is not None and p_b_dec is not None) else None
    fc_con_drop = (p_a_con - p_b_con) if (p_a_con is not None and p_b_con is not None) else None
    fc_interaction = (fc_dec_drop - fc_con_drop) if (fc_dec_drop is not None and fc_con_drop is not None) else None
    first_contact_pass = (
        fc_dec_drop is not None and fc_dec_drop > 0 and fc_interaction is not None and fc_interaction > 0
    )
    if fc_dec_drop is None or fc_con_drop is None:
        notes.append("First-contact proportion missing for ≥1 arm (entity never engaged in some cell).")

    # Secondary — ablation reversion on the deceptive scenario warm_self metric.
    ablation_details = _compute_secondary(grouped, deceptive, metric)
    ablation_hits = sum(1 for r in ablation_details if r["pass"])

    # General-caution confound: B reduces warm_self in BOTH scenarios by ≥1 SD
    # of each scenario's own Arm-A baseline (so the reduction is not specific to
    # the deceptive world — that's caution, not counter-prior learning).
    def _dropped(delta: float | None, sd: float | None) -> bool:
        if delta is None:
            return False
        if sd is None or sd == 0:
            return delta < 0
        return (delta / sd) <= -sd_threshold

    avoids_both = _dropped(delta_deceptive, a_dec_sd) and _dropped(delta_consistent, a_con_sd)

    substrate_signal = interaction_pass and first_contact_pass
    if interaction_pass != first_contact_pass:
        notes.append(
            f"Interaction primary (pass={interaction_pass}) and first-contact primary "
            f"(pass={first_contact_pass}) DISAGREE — both are pre-registered primaries; "
            f"SUBSTRATE_MATTERS requires both. Treating as no clean substrate signal."
        )

    if substrate_signal and ablation_hits >= 1:
        label, rationale = (
            VERDICT_CP_SUBSTRATE_MATTERS,
            "B reduces warm_self specifically in the deceptive world (interaction + "
            "first-contact both pass) AND ≥1 ablation reverts it toward Arm A. The "
            "substrate overrides a strong, wrong prior — the positive thesis result.",
        )
    elif substrate_signal and ablation_hits == 0:
        label, rationale = (
            VERDICT_CP_VOID_NOT_ATTRIBUTABLE,
            "B avoids the deceptive hearth (interaction + first-contact pass) but NO "
            "ablation abolishes the avoidance — the avoidance is not attributable to "
            "the substrate wires (prompt or within-session memory). Void for this run.",
        )
    elif avoids_both:
        label, rationale = (
            VERDICT_CP_VOID_GENERAL_CAUTION,
            "B reduces warm_self in BOTH the deceptive AND consistent worlds by ≥1 SD "
            "— a general-caution confound, not counter-prior learning. The interaction "
            "is not specific to the harmful hearth. Void.",
        )
    else:
        label, rationale = (
            VERDICT_CP_DOMINANCE,
            "B keeps warming the deceptive hearth — even direct cross-session pain does "
            "NOT override the LLM's fire→warm prior. Dominance demonstrated: a stronger "
            "result than the Exp 37 null (there WAS a behavioral gap and the substrate "
            "did not fill it).",
        )

    return CounterPriorVerdict(
        delta_deceptive=delta_deceptive,
        delta_consistent=delta_consistent,
        interaction=interaction,
        pooled_a_sd=pooled_a_sd,
        interaction_sd_units=interaction_sd_units,
        interaction_pass=interaction_pass,
        p_a_deceptive=p_a_dec,
        p_b_deceptive=p_b_dec,
        p_a_consistent=p_a_con,
        p_b_consistent=p_b_con,
        first_contact_deceptive_drop=fc_dec_drop,
        first_contact_consistent_drop=fc_con_drop,
        first_contact_interaction=fc_interaction,
        first_contact_pass=first_contact_pass,
        ablation_details=ablation_details,
        ablation_hits=ablation_hits,
        avoids_both=avoids_both,
        verdict_label=label,
        verdict_rationale=rationale,
        notes=notes,
    )


# ─── Verdict matrix ──────────────────────────────────────────────────────
#
# Pre-reg §"Graduation paths from this experiment" + §"Falsification
# conditions" together name the following outcomes. Documented in three
# places (pre-reg, protocol §D table, this function) — see B2 note in the
# module docstring + the cross-source test in the smoke suite.

VERDICT_EARNED = "EARNED"
VERDICT_EARNED_FOOTNOTED = "EARNED (footnoted)"
VERDICT_PARTIAL_REFRAMED = "PARTIAL — reframed"
VERDICT_PARTIAL_FALSIFIED = "PARTIAL — falsified"  # I3 fold
VERDICT_PARTIAL_INVESTIGATION = "PARTIAL — investigation gate"

# Exp 38 counter-prior labels (counter_prior_substrate_experiment.md §5).
# Both SUBSTRATE_MATTERS and DOMINANCE are *complete, diagnostic* outcomes
# ("diagnostic either way", §1) — exit 0. The two VOID outcomes mean the run
# did not cleanly discriminate (confound / non-attributable) — exit 4.
VERDICT_CP_SUBSTRATE_MATTERS = "COUNTER-PRIOR — substrate matters"
VERDICT_CP_DOMINANCE = "COUNTER-PRIOR — dominance demonstrated"
VERDICT_CP_VOID_GENERAL_CAUTION = "COUNTER-PRIOR — void (general caution)"
VERDICT_CP_VOID_NOT_ATTRIBUTABLE = "COUNTER-PRIOR — void (not substrate-attributable)"


def overall_verdict(scenario_verdicts: list[ScenarioVerdict]) -> tuple[str, str]:
    """Apply the pre-reg's graduation matrix across scenarios.

    Returns ``(label, rationale)``. Both scenarios must pass independently
    for EARNED; failure on either scenario downgrades the overall verdict.

    Scenarios whose PRIMARY_METRIC is structurally absent (N/A) are EXCLUDED
    from every aggregate — they were never testable, so they neither pass nor
    fail (counter_prior_substrate_experiment.md §6.3). This is the fix for the
    Exp 37 sharp_rock artifact: previously sharp_rock's structurally-0 primary
    FAILed and dragged the overall to investigation; now it is simply N/A.
    """
    gated = [v for v in scenario_verdicts if not v.primary_structural_absence]
    if not gated:
        return (
            VERDICT_PARTIAL_INVESTIGATION,
            "Every scenario's PRIMARY_METRIC is structurally absent (N/A) — nothing "
            "testable in this design. Check the harness FAILURE_CLASS / metric wiring.",
        )
    primary_ok = all(v.primary_pass for v in gated)
    isolation_ok = all(v.isolation_pass for v in gated)
    corroborating_total = sum(v.corroborating_hits for v in gated)
    wrong_direction_total = sum(v.corroborating_wrong_direction for v in gated)
    secondary_total = sum(v.secondary_hits for v in gated)
    # 3 corroborating metrics × N gated scenarios.
    corroborating_slots = sum(len(v.corroborating_details) for v in gated)

    if not (primary_ok and isolation_ok):
        return (
            VERDICT_PARTIAL_INVESTIGATION,
            "Primary or isolation FAIL on ≥1 scenario. Root-cause required before bio-attribution can ship.",
        )

    # C2 fold: pre-reg §Corroborating literal wording — "If ALL THREE
    # diverge from the prediction while the primary still hits, that's
    # evidence the primary is a measurement artifact — investigate before
    # claiming EARNED rather than ignore." Applied across scenarios: if ALL
    # corroborating slots register ≥1 SD wrong-direction shifts, the primary
    # cannot be trusted regardless of secondary outcomes.
    if corroborating_slots > 0 and wrong_direction_total == corroborating_slots:
        return (
            VERDICT_PARTIAL_INVESTIGATION,
            f"All {corroborating_slots} corroborating metrics diverged from prediction "
            f"by ≥1 SD; per pre-reg §Corroborating, primary may be a measurement "
            f"artifact — investigate before claiming EARNED.",
        )

    if secondary_total >= 1 and corroborating_total >= 1:
        return (
            VERDICT_EARNED,
            "All criteria pass: primary + isolation + secondary + ≥1 corroborating.",
        )

    if secondary_total >= 1 and corroborating_total == 0:
        return (
            VERDICT_EARNED_FOOTNOTED,
            "Primary + isolation + secondary pass; 0 corroborating metrics hit. "
            "Earn with footnote per pre-reg §'Graduation paths'.",
        )

    # I3 fold: pre-reg §Falsification #3 "Secondary catastrophic FAIL" is
    # a distinct outcome from the reframe path — secondary FAILS AND
    # 0 corroborating means bio-attribution is unsupported AND the
    # behavioral signal isn't corroborated, so the primary itself is
    # suspect. The reframe path retains the behavioral claim minus
    # bio-attribution; the falsified path retracts even the behavioral
    # claim.
    if secondary_total == 0 and corroborating_total == 0:
        return (
            VERDICT_PARTIAL_FALSIFIED,
            "Secondary catastrophic FAIL (pre-reg §Falsification #3): primary + "
            "isolation pass but ALL ablations failed AND 0 corroborating hit. "
            "Bio-attribution unsupported AND behavioral signal uncorroborated — "
            "investigate before any release-notes claim.",
        )

    if secondary_total == 0:
        return (
            VERDICT_PARTIAL_REFRAMED,
            "Primary + isolation pass + ≥1 corroborating hit but ALL 3 ablations "
            "failed to shrink the delta. Bio-attribution unsupported; release notes "
            "must reframe as 'cross-session memory surfacing via LLM in-context "
            "reasoning, not bio-substrate reward learning.'",
        )

    return ("PARTIAL — undetermined", "Logic gap; review scenario verdicts manually.")


# Exit-code mapping (consumed by PR #6 + downstream automation).
EXIT_CODE_FOR_LABEL: dict[str, int] = {
    VERDICT_EARNED: 0,
    VERDICT_EARNED_FOOTNOTED: 0,
    VERDICT_PARTIAL_REFRAMED: 3,
    VERDICT_PARTIAL_FALSIFIED: 5,
    VERDICT_PARTIAL_INVESTIGATION: 4,
    # Exp 38 counter-prior — both clean outcomes exit 0 (diagnostic either way);
    # the two void/confound outcomes exit 4 (re-run / investigate).
    VERDICT_CP_SUBSTRATE_MATTERS: 0,
    VERDICT_CP_DOMINANCE: 0,
    VERDICT_CP_VOID_GENERAL_CAUTION: 4,
    VERDICT_CP_VOID_NOT_ATTRIBUTABLE: 4,
}


# ─── Markdown rendering ──────────────────────────────────────────────────


def _fmt(v: float | None, *, places: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{places}f}"


def _render_counter_prior(cp: CounterPriorVerdict) -> list[str]:
    """Render the Exp 38 counter-prior interaction section (§5)."""
    lines: list[str] = []
    lines.append("### Counter-prior interaction (Exp 38 primary)")
    lines.append("")
    lines.append(f"**Verdict: {cp.verdict_label}**")
    lines.append("")
    lines.append(cp.verdict_rationale)
    lines.append("")
    lines.append("**Interaction primary — warm_self-engagement-fraction**")
    lines.append("")
    lines.append("| Quantity | Value |")
    lines.append("|---|---|")
    lines.append(f"| Δ deceptive (B − A) | {_fmt(cp.delta_deceptive)} |")
    lines.append(f"| Δ consistent (B − A) | {_fmt(cp.delta_consistent)} |")
    lines.append(f"| Interaction (Δ_dec − Δ_con) | {_fmt(cp.interaction)} |")
    lines.append(f"| Pooled Arm-A SD | {_fmt(cp.pooled_a_sd)} |")
    lines.append(f"| Interaction in SD units | {_fmt(cp.interaction_sd_units, places=2)} |")
    lines.append(
        f"| Predicted | ≤ −{INTERACTION_SD_SHIFT_THRESHOLD} SD → **{'PASS' if cp.interaction_pass else 'FAIL'}** |"
    )
    lines.append("")
    lines.append("**First-contact isolation — P(warm_self on first contact)**")
    lines.append("")
    lines.append("| Arm | Deceptive | Consistent |")
    lines.append("|---|---|---|")
    lines.append(f"| A | {_fmt(cp.p_a_deceptive, places=2)} | {_fmt(cp.p_a_consistent, places=2)} |")
    lines.append(f"| B | {_fmt(cp.p_b_deceptive, places=2)} | {_fmt(cp.p_b_consistent, places=2)} |")
    lines.append("")
    lines.append(
        f"Deceptive drop (A − B) = {_fmt(cp.first_contact_deceptive_drop, places=2)}; "
        f"consistent drop = {_fmt(cp.first_contact_consistent_drop, places=2)}; "
        f"first-contact interaction = {_fmt(cp.first_contact_interaction, places=2)} → "
        f"**{'PASS' if cp.first_contact_pass else 'FAIL'}** "
        f"(need deceptive-drop > 0 AND interaction > 0)."
    )
    lines.append("")
    lines.append("**Secondary — ablation reversion on the deceptive hearth (≥1 must revert)**")
    lines.append("")
    lines.append("| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |")
    lines.append("|---|---|---|---|---|---|")
    for s in cp.ablation_details:
        shr_disp = _fmt(s["shrinkage_in_sd_units"], places=2)
        res = "PASS" if s["pass"] else "FAIL"
        note_suffix = f" — {s['note']}" if s.get("note") else ""
        lines.append(
            f"| {s['label']} | {_fmt(s['a_mean'])} | {_fmt(s['b_mean'])} | "
            f"{_fmt(s['ablated_mean'])} | {shr_disp} | **{res}**{note_suffix} |"
        )
    lines.append("")
    lines.append(f"Ablation reversion hits: **{cp.ablation_hits} / {len(cp.ablation_details)}**")
    lines.append("")
    if cp.notes:
        lines.append("**Counter-prior notes / warnings**")
        lines.append("")
        for n in cp.notes:
            lines.append(f"- {n}")
        lines.append("")
    return lines


def render_markdown(
    verdicts: list[ScenarioVerdict],
    verdict_overall: tuple[str, str],
    *,
    in_path: Path,
    heading_suffix: str = "",
    counter_prior: CounterPriorVerdict | None = None,
) -> str:
    """Render the report. ``heading_suffix`` (e.g., ``"2026-06-15 run-12"``)
    disambiguates repeated Results sections when the report is appended to
    the experiment doc more than once on the minor-version heartbeat."""
    label, rationale = verdict_overall
    lines: list[str] = []
    suffix = f" — {heading_suffix}" if heading_suffix else ""
    lines.append(f"## Results{suffix}")
    lines.append("")
    lines.append(f"Source: `{in_path}` · Analyzer version: `{ANALYZER_VERSION}` · Schema: `{SUPPORTED_SCHEMA_VERSION}`")
    lines.append("")
    lines.append(f"### Overall verdict: **{label}**")
    lines.append("")
    lines.append(rationale)
    lines.append("")

    # Exp 38 counter-prior interaction renders first (it IS the overall verdict
    # when present); the per-scenario tables below become supporting detail.
    if counter_prior is not None:
        lines.extend(_render_counter_prior(counter_prior))

    for v in verdicts:
        lines.append(f"### Scenario: `{v.scenario}`")
        lines.append("")
        lines.append("**Primary + isolation**")
        lines.append("")
        lines.append("| Arm | Mean | Predicted | Verdict |")
        lines.append("|---|---|---|---|")
        a_band_str = f"[{_fmt(v.a_band[0])}, {_fmt(v.a_band[1])}]" if v.a_band else "—"
        lines.append(f"| A | {_fmt(v.a_mean)} | baseline · 95% band {a_band_str} | — |")
        # Structural absence (counter_prior_substrate_experiment.md §6.3): the
        # metric never varied across any arm — report N/A, not FAIL.
        primary_str = "N/A" if v.primary_structural_absence else ("PASS" if v.primary_pass else "FAIL")
        # Pre-reg amendment 2026-06-05 (exp37_sd_shift.md): primary
        # criterion is mean-shift-in-SD-units (≥ ``PRIMARY_SD_SHIFT_THRESHOLD``
        # in the predicted direction), not the legacy empirical
        # percentile band. The label reflects what reviewers should
        # check against.
        if v.a_sd is not None and v.a_sd > 0 and v.a_mean is not None and v.b_mean is not None:
            delta_sd = (v.b_mean - v.a_mean) / v.a_sd
            sd_str = f"Δ = {delta_sd:+.2f} SD"
        else:
            sd_str = "Zero-SD fallback"
        sign = "≥" if PRIMARY_METRIC_DIRECTION == "increase" else "≤"
        threshold_sign = "+" if PRIMARY_METRIC_DIRECTION == "increase" else "−"
        predicted_str = f"{sd_str} (need {sign}{threshold_sign}{PRIMARY_SD_SHIFT_THRESHOLD} SD)"
        lines.append(f"| B | {_fmt(v.b_mean)} | {predicted_str} | **{primary_str}** |")
        isolation_str = "PASS" if v.isolation_pass else "FAIL"
        lines.append(f"| C | {_fmt(v.c_mean)} | ∈ A's band | **{isolation_str}** |")
        lines.append("")
        lines.append(
            f"Robustness (legacy per-action failure rate, decrease): "
            f"{'PASS' if v.primary_pass_robustness else 'FAIL'}"
            + (
                " — DIVERGES from positive-approach-engagement primary; "
                "see protocol §1 (substrate may be biasing warm_self without "
                "reducing touch, or vice versa)"
                if v.primary_pass != v.primary_pass_robustness
                else ""
            )
        )
        lines.append("")

        lines.append("**Corroborating metrics (≥1 must pass)**")
        lines.append("")
        lines.append("| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        for c in v.corroborating_details:
            a_disp = f"{_fmt(c['a_mean'])} ± {_fmt(c['a_sd'])}"
            delta_disp = _fmt(c["delta_in_sd_units"], places=2)
            res = "PASS" if c["pass"] else "FAIL"
            note_suffix = f" — {c['note']}" if c.get("note") else ""
            lines.append(
                f"| {c['label']} | {a_disp} | {_fmt(c['b_mean'])} | {delta_disp} | "
                f"{c['direction']} | **{res}**{note_suffix} |"
            )
        lines.append("")
        lines.append(f"Corroborating hits: **{v.corroborating_hits} / {len(v.corroborating_details)}**")
        lines.append("")

        # Descriptive corroborating: fire_approach_action_count
        # (cradle_activation_fixes.md P2). Rendered only when the scenario
        # has a non-zero approach signal. NOT pre-reg gated — pure
        # documentation of WHY the primary metric moved (positive substrate
        # edge inspection).
        if v.approach_descriptive is not None:
            ad = v.approach_descriptive
            lines.append("**Descriptive corroborating — `fire_approach_action_count` (NOT pre-reg gated)**")
            lines.append("")
            lines.append("| Arm | Mean count |")
            lines.append("|---|---|")
            lines.append(f"| A | {_fmt(ad['a_mean'], places=2)} |")
            lines.append(f"| B | {_fmt(ad['b_mean'], places=2)} |")
            delta_disp = _fmt(ad["delta"], places=2)
            lines.append("")
            lines.append(f"Δ (B − A) = {delta_disp}; predicted direction: {ad['predicted_direction']}.")
            if ad.get("note"):
                lines.append("")
                lines.append(f"_Note: {ad['note']}_")
            lines.append("")

        lines.append("**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**")
        lines.append("")
        lines.append("| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        for s in v.secondary_details:
            shr_disp = _fmt(s["shrinkage_in_sd_units"], places=2)
            res = "PASS" if s["pass"] else "FAIL"
            note_suffix = f" — {s['note']}" if s.get("note") else ""
            lines.append(
                f"| {s['label']} | {_fmt(s['a_mean'])} | {_fmt(s['b_mean'])} | "
                f"{_fmt(s['ablated_mean'])} | {shr_disp} | **{res}**{note_suffix} |"
            )
        lines.append("")
        lines.append(f"Secondary hits: **{v.secondary_hits} / {len(v.secondary_details)}**")
        lines.append("")

        if v.notes:
            lines.append("**Notes / warnings**")
            lines.append("")
            for n in v.notes:
                lines.append(f"- {n}")
            lines.append("")
        lines.append("")
    return "\n".join(lines)


# ─── CLI ─────────────────────────────────────────────────────────────────


def run_analysis(
    *,
    in_path: Path,
    expected_trials: int,
    arms: tuple[str, ...],
    scenarios: tuple[str, ...],
    strict_schema_version: str,
    heading_suffix: str = "",
) -> AnalysisResult:
    """Top-level driver. Returns a frozen :class:`AnalysisResult`."""
    records = load_records(in_path, expected_schema=strict_schema_version)
    grouped = group_records(records)
    assert_design_complete(grouped, arms=arms, scenarios=scenarios, expected_trials=expected_trials)
    verdicts = [evaluate_scenario(grouped, s, arms=arms) for s in scenarios]

    # Exp 38: when both the consistent control and deceptive counter-prior
    # scenarios are present, the cross-scenario interaction IS the overall
    # verdict (it supersedes the per-scenario Exp 37 gating). Otherwise fall
    # through to the legacy per-scenario verdict.
    counter_prior = compute_counter_prior(grouped)
    if counter_prior is not None:
        overall_label = counter_prior.verdict_label
        overall_rationale = counter_prior.verdict_rationale
    else:
        overall_label, overall_rationale = overall_verdict(verdicts)

    markdown = render_markdown(
        verdicts,
        (overall_label, overall_rationale),
        in_path=in_path,
        heading_suffix=heading_suffix,
        counter_prior=counter_prior,
    )
    return AnalysisResult(
        scenarios=verdicts,
        overall_label=overall_label,
        overall_rationale=overall_rationale,
        markdown=markdown,
        counter_prior=counter_prior,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exp 37 cross-session graduation analyzer")
    parser.add_argument(
        "--in", dest="in_path", type=Path, required=True, help="JSONL emitted by scripts/benchmark_cross_session.py"
    )
    parser.add_argument("--out", type=Path, default=None, help="Markdown output (default: stdout)")
    parser.add_argument(
        "--trials", type=int, default=5, help="Expected trials per (scenario, arm) — design-completeness check"
    )
    parser.add_argument(
        "--arms",
        default="A,B,C,B-wire-a-off,B-wire-1-off,B-nac-bias-off",
        help="Comma-separated arms expected in the JSONL",
    )
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument(
        "--strict-schema-version",
        default=SUPPORTED_SCHEMA_VERSION,
        help="Refuse records whose _format_version / schema_version differs.",
    )
    parser.add_argument(
        "--heading-suffix",
        default="",
        help="Disambiguate the Markdown '## Results' heading "
        "(e.g., '2026-06-15 run-12') when appending to an experiment doc "
        "that already has a Results section from a prior run.",
    )
    args = parser.parse_args(argv)

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    scenarios = tuple(s.strip() for s in args.scenarios.split(",") if s.strip())

    try:
        result = run_analysis(
            in_path=args.in_path,
            expected_trials=args.trials,
            arms=arms,
            scenarios=scenarios,
            strict_schema_version=args.strict_schema_version,
            heading_suffix=args.heading_suffix,
        )
    except AnalyzerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.out is None:
        print(result.markdown)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.markdown)
        print(f"Wrote {args.out} — verdict: {result.overall_label}", file=sys.stderr)

    # Exit codes consumed by PR #6 (graduation status flip) and any future
    # automation that branches on verdict severity. Contract in protocol §D:
    #   0 = EARNED or EARNED (footnoted)
    #   2 = AnalyzerError (schema mismatch, incomplete design, parse error)
    #   3 = PARTIAL — reframed (release-notes reframe required; needs user auth)
    #   4 = PARTIAL — investigation gate (primary/isolation FAIL OR wrong-
    #       direction corroborating triggers measurement-artifact concern)
    #   5 = PARTIAL — falsified (secondary catastrophic FAIL per pre-reg §3)
    return EXIT_CODE_FOR_LABEL.get(result.overall_label, 4)


if __name__ == "__main__":
    sys.exit(main())
