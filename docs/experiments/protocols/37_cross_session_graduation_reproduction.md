# Reproduction — Exp 37 (Cross-Session Behavioral Delta, Cradle graduation)

**Companion:** [37_cross_session_graduation.md](../37_cross_session_graduation.md)
**Pre-registration:** locked 2026-05-30 in the companion doc (PR #304).
**Owning plan:** [docs/plans/benchmarking_1_0.md](../../plans/benchmarking_1_0.md) §"Acceptance criteria for 1.0" + [docs/plans/behavioral_graduation_candidates.md](../../plans/behavioral_graduation_candidates.md) Tier 1 row 1.
**Harness:** [scripts/benchmark_cross_session.py](../../../scripts/benchmark_cross_session.py)
**Smoke test:** [tests/behavioral/test_exp37_harness_smoke.py](../../../tests/behavioral/test_exp37_harness_smoke.py)

## What this protocol pins

Every field needed to **re-run Exp 37 and get a comparable result** on a future minor-version heartbeat trigger (per [behavioral_graduation_candidates.md](../../plans/behavioral_graduation_candidates.md) lifecycle). If any of these drift, the re-run is no longer the same experiment and the Earned/Stale verdict must be revisited.

| Field | Value | Notes |
|---|---|---|
| Primary model | `claude-sonnet` (Anthropic Sonnet, the profile that resolves at the time of run) | Tertiary replication may use `qwen2.5-14b-instruct` (local). The exact API model ID is captured per-run in the JSONL record's `version_info` via `get_version_info()`. |
| Embodiment | `bodies/infant_humanoid` | Built-in seed; ships in `src/maxim/_data/components/bodies/infant_humanoid.yaml`. |
| Scenarios | `fire_pit`, `sharp_rock` | Goal strings + entity-name detection rules locked in `SCENARIO_GOAL` + `FAILURE_CLASS` in the harness. Cradle item YAMLs at `_data/components/items/cradle_{fire_pit,sharp_rock}.yaml`. |
| Peaceful (Arm C) prior | `cradle infant explores the puzzle door and button` | Targets cradle phase 3 entities (`lever_door` + `button`) — no thermal/laceration affordances. |
| Sim max turns | 12 (default in harness) | Override via `--sim-max-turns`; not part of pre-reg, so reproductions may tighten. |
| Trial count per arm per scenario | 5 (pre-registered) | `--trials 5`. |
| Seed base | 42 (default) | Per-trial seed = `seed_base + trial_pair_id`. The seed is logged per-record; the harness does NOT thread it into the sim — the LLM backend is sampled by Anthropic's API. The seed exists as a reproducibility hook for any future deterministic-replay layer; today it primarily identifies trial-pair ordering. |
| Prompt version | Captured from `get_version_info()` → `git_hash` per-run | If git_hash differs across runs, prompt construction may have drifted; surface that to the user before claiming reproduction. |
| Encoder version | Captured from `importlib.metadata.version("sentence-transformers")` per-run | Mirrors the same field-pin discipline as Roy-3c-bisect (encoder drift was an outside cause; pinning is mandatory). |
| Harness version | `1.0` (pinned in `HARNESS_VERSION`) | Bump on any schema-affecting change. |
| Schema version | `1.0` (pinned in `SCHEMA_VERSION`) | The analyzer (PR #4) reads this and rejects mismatched records. |
| Cradle arc | `cradle` builtin in `src/maxim/simulation/arcs.py` (phases 1+2+3) | The arc selection is driven by goal-string keyword scorer in `select_arc_for_goal`. If the arc shape changes (phase reorder, entity manifest edit), re-runs are not directly comparable. |

## Implementation decisions locked here (the pre-reg deliberately stops at "what passes the gate"; these are the "how the harness measures it" choices)

### 1. Primary metric — current state (post pre-reg amendments 2026-05-31 and 2026-06-XX)

**Current primary (Path 2 pivot per [exp37_metric_pivot.md](../../plans/exp37_metric_pivot.md)):**

```python
PRIMARY_METRIC = "positive_approach_engagement_fraction"
PRIMARY_METRIC_DIRECTION = "increase"  # B > A.p97.5 (higher warm_self share is better)
ROBUSTNESS_METRIC = "per_action_failure_rate"
ROBUSTNESS_METRIC_DIRECTION = "decrease"  # legacy primary, retained as robustness signal
```

**Positive-approach-engagement-fraction:** `fire_pit_warm_self_count / engagement_count`, where `engagement_count = fire_pit_warm_self_count + fire_pit_observe_count + fire_pit_touch_count + pick_up_fire_pit_count`. Sessions with zero on-target engagement emit `0.0` (denominator clamped to 1). For sharp_rock the metric is structurally 0 — no positive-approach affordance in that scenario by design. This is the scenario-asymmetric design the 2026-06-XX amendment explicitly endorses.

**Why the pivot.** The 2026-06-04 cradle smoke (PR E branch live) confirmed PR D + PR E work end-to-end: the LLM-AUT reasons about thermal homeostasis, finds `fire_pit_warm_self`, and restores body temperature to set-point. But the LLM also discovers the SAFE warming path (`warm_self` + `shelter`) and never NEEDS to touch the fire. `failure_class_action_count` is structurally 0 on Arm A → `a_sd = 0` → variance-survival math is impossible (B mean cannot be negative). The pivot reframes the substrate-transfer claim from "B touches less" to "B prefers warm_self more strongly when engaging with fire," which directly measures what the substrate-transferred reward bias predicts.

**Legacy `per_action_failure_rate` retained as robustness signal.** Divergence between primary and robustness flags substrate weirdness (warm_self preference shift without touch reduction, or vice versa). Both signals should be inspected when interpreting the verdict. The pre-2026-05-31 per-turn metric (`primary_metric_repeat_failure_action_rate`) is still emitted by the harness for record-schema stability but is no longer consumed by the analyzer's pass/fail logic.

Edge cases (positive-approach-engagement-fraction):

- Sessions with zero on-target engagement → fraction = 0.0 (denominator clamped). This is the "agent never engaged" path; consistent with "no signal to interpret."
- All-engagement-is-warm_self sessions → fraction = 1.0. Plausible on Arm B but ceiling effect possible; the corroborating `time_to_first_warm_self_action` provides additional discrimination.
- Sessions where the agent ONLY touches (no warm_self / observe) → fraction = 0.0 (the substrate-transferred preference signal indicates avoidance, not engagement).

**New corroborating: `time_to_first_warm_self_action`.** Action-index at which the agent first calls a positive-approach affordance. Substrate transfer predicts B's first warm_self comes earlier than A's exploration-driven discovery. None values (never reached) censored to `total_actions + 1` (mirrors `time_to_safe_steady_state_turns` handling).

### 2. Failure-class detection rules

Per-scenario rules live in `FAILURE_CLASS` at the top of the harness. They follow the entity affordance tool naming convention (`{entity_name}_{affordance_name}` per `embodiment/tool_bridge.py::_resolve_tool_name`) and the body-level pick_up convention (`infant_humanoid_pick_up` with `params.object` carrying the entity name).

| Scenario | Failure-class tools | Safe (on-target) tools | Approach (positive subset of safe) | Engagement (PRIMARY denominator) |
|---|---|---|---|---|
| `fire_pit` | `fire_pit_touch`, `infant_humanoid_pick_up(object=fire_pit)` | `fire_pit_observe`, `fire_pit_warm_self` | `fire_pit_warm_self` | `fire_pit_warm_self`, `fire_pit_observe`, `fire_pit_touch`, `infant_humanoid_pick_up(object=fire_pit)` |
| `sharp_rock` | `sharp_rock_touch`, `infant_humanoid_pick_up(object=sharp_rock)` | `sharp_rock_examine` | _none (scenario asymmetric by design)_ | `sharp_rock_examine`, `sharp_rock_touch`, `infant_humanoid_pick_up(object=sharp_rock)` |

`fire_pit_warm_self` is a proximity affordance (cradle_activation_fixes.md P2 amendment): the agent stands close enough to feel the radiated heat without making contact, producing a positive `core_temperature` write (+0.2) that drives toward homeostasis. Counted as a safe choice in `affordance_preference_safe_fraction` AND tracked separately as the **`fire_approach_action_count`** descriptive corroborating metric. The Approach column is scenario-asymmetric by design — sharp_rock has no proximity-positive analog.

If a future cradle-arc revision renames any affordance on `fire_pit` from `observe` / `warm_self` / `touch` to something else, the harness MUST be updated in lockstep — the rules are pinned to the YAML's affordance names. Drift between the YAML and the rules silently underestimates the safe-fraction (and silently zeros the approach metric).

To prevent silent drift, `run_benchmark` calls `_assert_failure_class_matches_yaml(scenario)` at startup (when bio-stack imports are available — skipped for smoke-test contexts). The check loads `_data/components/items/cradle_<scenario>.yaml` via `ComponentRegistry`, extracts the actual affordance names from the entity's modulators, and asserts that every affordance referenced by `FAILURE_CLASS[scenario]` (including the new `direct_approach_tools` set) is present in the YAML. A rename produces a loud `RuntimeError` rather than a $14 zero-signal trial run.

### 2.1. `fire_approach_action_count` — descriptive corroborating metric (NOT pre-reg gated)

Added in cradle_activation_fixes.md P2 alongside the drive-system calibration (infant_humanoid `core_temperature` `comfort_band: 0.4 → 0.25`, `pain_scale: 0.5 → 1.5`) and the `fire_pit_warm_self` proximity affordance. Counts the number of `fire_pit_warm_self` calls per session (0 for `sharp_rock` by construction).

**Hypothesis under substrate transfer (Arm B vs Arm A):**

- `fire_approach_action_count`: same or HIGHER on B (positive substrate edge "fire = warm" transferred)
- `failure_class_action_count`: LOWER on B (negative substrate edge "touch = pain" transferred)

The discrimination between the two edges is what the substrate-transfer claim predicts — bare avoidance (`A=0, B=0`) means the LLM's adult prior dominated and neither edge formed. Bare overcorrection (`A=high, B=high` for both) means the substrate isn't differentiating.

**Operationally:** the analyzer emits both counts in the per-scenario descriptive block. Per the 2026-06-XX pivot, `fire_approach_action_count`'s spiritual successor — `positive_approach_engagement_fraction` — is now the primary metric and IS pre-reg gated (the substrate-transfer hypothesis on the positive edge is what the pivot measures). The raw `fire_approach_action_count` field is retained for descriptive interpretation (e.g., "B's higher fraction reflects 4 warm_self calls vs A's 2"). `failure_class_action_count` is similarly retained descriptively; both feed the analyzer's narrative block but neither gates the verdict on its own.

### 3. Run-count: 65 runs (per [the pre-reg's trial-structure table](../37_cross_session_graduation.md#trial-structure))

The pre-reg's trial-structure table enumerates **60 target runs** + **5 Arm C peaceful priors** (shared across scenarios per trial-pair, since peaceful substrate is scenario-agnostic) = **65 runs total / ~$13.65**.

The Arm A target runs are reused as the substrate-prior for Arm B and the three B-family ablation arms — `shutil.copytree` (with `ignore=shutil.ignore_patterns("*.lock", "*.lock.tmp")` to skip stale advisory lock files) copies the post-Arm-A `MAXIM_DATA_HOME` before each B-family resume so each ablation arm starts from an identical snapshot. This sharing keeps the run count from doubling.

### 4. Fixture YAMLs vs Cradle-arc narration — DEFERRED to PR #5

The kickoff identified that the existing Cradle arc may be too LLM-variable to give stable scenario-scoping. The harness currently uses **goal-string narration** (`SCENARIO_GOAL`) rather than dedicated `_data/components/scenes/exp37_*.yaml` fixtures. The decision to introduce fixtures is deferred to PR #5 (trial execution), where 1-2 pilot real-LLM runs against the existing arc will reveal whether the narrator reliably surfaces `fire_pit` (for the fire_pit scenario) and `sharp_rock` (for the sharp_rock scenario) per Cradle phase 1+2 entity manifests.

Pilot acceptance bar: ≥80% of pilot runs surface the target entity in the first 3 turns. If below, ship dedicated fixture YAMLs in PR #5 that pin the entity manifest.

### 5. Cost-cap two-layer defense (per-record safety net + pair-boundary projection)

The pre-reg's `--cost-cap` flag protects against runaway LLM spend, but a naive "abort whenever cumulative exceeds cap" check creates a worse problem: aborting mid-trial-pair leaves the analyzer with half-written paired-trial data (e.g., Arm A written but Arm B not started, with no way to compute the paired delta). The harness uses a two-layer defense:

- **Pair-boundary projection (preferred):** at the start of each `(trial_id, scenario)` iteration, the harness projects the upcoming pair's worst-case cost using `observed_max_record_cost × arm_count + (arm_c_prior_cost if needed)`. If `cumulative + projected > cap`, the pair never starts and the JSONL stays clean. Error message: "Aborting cleanly between trial pairs."
- **Per-record safety net (backup):** for the FIRST pair (no history to project from) and for runaway single-arm cost balloons, `_check_cost` fires AFTER each record is written if cumulative exceeds cap. Partial-pair data is the price of catching the breach in flight. Analyzer (PR #4) discards incomplete `(trial_pair_id, scenario)` groups.

The `observed_max_record_cost` starts at `0.0` so the first pair's projection always passes (no data yet); after the first pair runs, it adapts to actual observed costs and subsequent pair projections become predictive.

### 6. Append-only JSONL idempotency

The output JSONL is append-only across re-runs (matches `scripts/run_v1_phases.sh` precedent), but the harness enforces idempotency by refusing to emit a record whose `(experiment, trial_pair_id, arm, scenario)` key already appears in the file. Recovery flow: pass a different `--out` or delete the existing file. The key tuple is the analyzer's primary dedup unit.

### 7. Stage-0b actions.jsonl header skip

Real `actions.jsonl` written by `src/maxim/simulation/report.py::save_action_log` carries a header line at offset 0: `{"_format_version": "1.1", "_record_kind": "header", "session_id": "..."}`. The harness's `_load_latest_session` skips any line where `_record_kind == "header"` BEFORE treating per-line records as actions. The mock backend (`_mock_sim`) writes the same header to keep the mock-real contract symmetric — silent drift here would cause the smoke test to mask a real-run off-by-one in primary-metric computation.

### 8. Persistence-envelope `_format_version` on JSONL records

Per CLAUDE.md's `_format_version` invariant, every JSONL record carries `_format_version` (alongside the experiment-scoped `schema_version`, which mirrors it for analyzer-side branching). Bump both in lockstep when the record shape changes.

## Reproducibility envelope (record in `version_info` on every JSONL record)

Every record the harness writes includes a `version_info` block populated by `_capture_versions()`. At minimum:

```json
{
  "version_info": {
    "harness_version": "1.0",
    "version": "0.9.1",
    "git_hash": "a6f721c",
    "git_message": "Merge pull request #310 ...",
    "sentence_transformers_version": "..."
  }
}
```

If `version_info.git_hash` differs across runs you intend to compare, the prompt-construction surface MAY have drifted — open the diff and confirm before relying on cross-run comparisons.

## A. Pre-flight (~2 min, no LLM cost)

```bash
# 1. Smoke test green from the worktree?
cd /Users/dennyschaedig/Scripts/Maxim-wt-cross-session
PYTHONPATH=src python -m pytest tests/behavioral/test_exp37_harness_smoke.py -v
# Expect 17/17 PASSED.

# 2. Pre-reg + benchmarking docs match the harness's understanding?
git log --oneline main | grep -E "Exp 37|REWARD_BIAS_DISABLED" | head -4
# Expect:
#   feat(1.0): add MAXIM_NAC_REWARD_BIAS_DISABLED env var for Exp 37 ablation arm 3
#   Merge pull request #304 from dennys246/feat/1-0-graduation-cross-session
#   docs(1.0): pre-register Exp 37 cross-session graduation + accept benchmarking scope
#   docs(1.0): benchmarking gate scoping doc (sibling to graduation gate)

# 3. Leader healthy at the URL in ~/.config/maxim/peer.yml (if running real-LLM trials)?
curl -si --max-time 10 \
  -H "Authorization: Bearer $(awk '/api_key:/ {print $2}' ~/.config/maxim/peer.yml)" \
  "$(awk '/url:/ {print $2}' ~/.config/maxim/peer.yml)/models" | head -3
```

## B. Dry-run against the mock backend (~1 s, no cost, no LLM)

```bash
cd /Users/dennyschaedig/Scripts/Maxim-wt-cross-session
PYTHONPATH=src python scripts/benchmark_cross_session.py \
    --scenario both --arms A,B,C \
    --trials 1 \
    --out /tmp/exp37_dryrun.jsonl \
    --cost-cap 1.0 \
    --sim-max-turns 6 \
    --mock-llm
cat /tmp/exp37_dryrun.jsonl | jq -c '{trial_pair_id, arm, scenario, primary_metric_repeat_failure_action_rate}'
# Expect 6 records: 2 scenarios × 3 arms × 1 trial.
```

## C. Trial execution (Phase 5 of the implementation queue — NOT this PR)

**NOT RUN BY THIS PROTOCOL** — trial execution lands in a separate PR after the analysis script (PR #4) and explicit user authorization. The kickoff that authorized this PR explicitly forbids real-LLM trial runs in the harness/protocol PR.

When the trial-execution PR opens, the canonical invocation will be:

```bash
# Real-LLM, full design. ~10 h wall time, ~$14 cost.
PYTHONPATH=src python scripts/benchmark_cross_session.py \
    --scenario both \
    --trials 5 \
    --model claude-sonnet \
    --out docs/experiments/data/37_results.jsonl \
    --cost-cap 20.0 \
    --sim-max-turns 12 \
    --seed-base 42
```

The harness writes append-only — re-runs add new records rather than overwriting, so partial progress is preserved across interruptions. To resume after an abort, re-invoke with the same args and a *new* `--workdir` (the per-arm sandboxes are fresh per run; the resume point comes from the existing JSONL records via the analyzer's per-pair completeness check, not from any state in the workdir).

## D. Analysis (`scripts/analyze_exp37.py`)

The analyzer reads `37_results.jsonl` and computes the pre-registered criteria in order:

- **Primary criterion variance-survival rule:** Arm B mean must lie ABOVE Arm A's 97.5th-percentile band, computed across the same N=5 paired trials (direction-flipped by 2026-06-XX pivot per §1). Percentile band uses `statistics.quantiles(values, n=40, method="inclusive")`; with N=5 this is essentially the empirical [min, max]. The one-sided check matches the predicted direction `B > A` (higher positive-approach-engagement-fraction is better).
- **Isolation rule:** Arm C mean must fall WITHIN `[A.p2.5, A.p97.5]`. If Arm C also rises above A's band, the analyzer flags the "general caution" confound — though now in the direction of "general preference for safe affordances regardless of substrate inheritance."
- **Secondary criterion (ablation attribution):** for each of the 3 B-family ablation arms, shrinkage = `|B - A| − |ablated − A|`; PASS if `shrinkage / A.sd ≥ 1.0`. Per pre-reg, ≥1 ablation must PASS. Direction-agnostic.
- **Corroborating metrics:** affordance-preference safe-fraction (direction `B > A`), tool-class diversity (direction `B < A`), time-to-safe-steady-state (direction `B < A`), AND `time_to_first_warm_self_action` (direction `B < A`, NEW per 2026-06-XX pivot — substrate transfer predicts B reaches warm_self earlier than A's exploration). Each PASSES when `(B - A) / A.sd ≥ 1` in the predicted direction. Per pre-reg, ≥1 must hit; if ALL diverge from prediction while the primary still hits, the analyzer notes the "measurement artifact" concern.
- **Robustness cross-check (legacy primary):** the analyzer ALSO computes the variance-survival test using the legacy `per_action_failure_rate` (direction `decrease`). Divergence between the new primary (positive-approach-engagement) and robustness (failure-rate) flags substrate weirdness — warm_self preference shift without touch reduction (or vice versa). Inspect both signals before claiming the verdict.
- **Schema enforcement:** records with `_format_version != "1.0"` (or missing the field entirely) are refused. Mixed schema versions in one file are a hard error.
- **Design completeness:** every `(scenario, arm)` combination must have exactly the expected number of trials. Missing or duplicate `trial_pair_id`s cause a hard error — analyzer never produces a verdict on partial data.

### Invocation

```bash
PYTHONPATH=src python scripts/analyze_exp37.py \
    --in docs/experiments/data/37_results.jsonl \
    --out docs/experiments/data/37_results.md \
    --trials 5 \
    --strict-schema-version 1.0
```

Exit codes:

- `0` — EARNED or EARNED (footnoted)
- `2` — analyzer error (schema mismatch, incomplete data, JSON parse error)
- `3` — PARTIAL — reframed (release notes must drop bio-attribution; needs user authorization)
- `4` — PARTIAL — investigation gate (primary or isolation FAIL; delay 1.0 ship)

### Output

The analyzer emits a Markdown block intended to be appended (by a human) to `docs/experiments/37_cross_session_graduation.md` as a Results section. Block shape:

```markdown
## Results

Source: ... · Analyzer version: 1.0 · Schema: 1.0

### Overall verdict: **EARNED**
<rationale>

### Scenario: `fire_pit`
**Primary + isolation** — table of A/B/C means + bands + pass/fail flags
**Corroborating metrics (≥1 must pass)** — table of 3 metrics + Δ in SD units
**Secondary criterion — ablation attribution** — table of 3 ablations + shrinkage in SD units
**Notes / warnings** — robustness divergence, missing-data flags, etc.

### Scenario: `sharp_rock`
<same shape>
```

### Verdict matrix (from pre-reg §"Graduation paths" + §"Falsification conditions")

| Primary | Isolation | Secondary | Corroborating | Label | Exit | Action |
|---|---|---|---|---|---|---|
| PASS | PASS | ≥1 PASS | ≥1 hit, 0 wrong-dir | EARNED | 0 | Flip `behavioral_graduation_candidates.md` row 1 to Earned; PR #6 |
| PASS | PASS | ≥1 PASS | 0 hits | EARNED (footnoted) | 0 | Earn with footnote; corroborating metrics need refinement |
| PASS | PASS | 0 PASS | ≥1 hit | PARTIAL — reframed | 3 | Release notes drop bio-attribution; needs user authorization |
| PASS | PASS | 0 PASS | 0 hits | PARTIAL — falsified | 5 | Pre-reg §Falsification #3 catastrophic FAIL: bio AND behavioral claim retracted; investigate before any release-notes wording |
| PASS | PASS | any | all wrong-dir ≥1 SD | PARTIAL — investigation gate | 4 | Pre-reg §Corroborating: primary may be a measurement artifact; investigate before claiming EARNED |
| FAIL | * | * | * | PARTIAL — investigation gate | 4 | Root-cause required; delay 1.0 |
| * | FAIL | * | * | PARTIAL — investigation gate | 4 | Same as above (general-caution confound) |

The analyzer DOES NOT auto-append to the experiment doc and DOES NOT update `behavioral_graduation_candidates.md` — both are human steps (PR #6).

### Forward obligations (when PR #6 flips row 1 to Earned)

Per CLAUDE.md Principle 5 (regression-guard / experiment citation per invariant), the EARNED entry in `behavioral_graduation_candidates.md` must declare:

- **Re-run on:** `encoder swap`, `substrate-pipeline change`, `minor-version heartbeat` (per [behavioral_graduation_candidates.md] Tier 1 trigger taxonomy).
- **Regression guard:** `docs/experiments/protocols/37_cross_session_graduation_reproduction.md` §D + `tests/behavioral/test_exp37_analyzer_smoke.py` + `tests/behavioral/test_exp37_harness_smoke.py`.

When PR #6 ships, the rest of the discipline cascades: any future minor-version heartbeat re-runs the harness, runs this analyzer with `--strict-schema-version 1.0`, and checks the verdict label is still `EARNED`. If the label flips to `Stale` or `Broken`, the entry blocks the next release per the graduation-candidates lifecycle.

## E. Updating `behavioral_graduation_candidates.md` (Phase 6 — separate PR)

Per the pre-reg's graduation matrix, the row 1 status flips based on the verdict:

| Outcome | Action |
|---|---|
| All criteria pass | Earned, with this protocol as the `Re-run on:` trigger reference |
| Primary pass + secondary fail + ≥1 corroborating | Reframed; release notes pull bio-attribution (requires user authorization) |
| Primary or isolation fail | Stays PARTIAL; investigation gate fires |

## Cross-references

- [37_cross_session_graduation.md](../37_cross_session_graduation.md) — the pre-registration.
- [benchmarking_1_0.md](../../plans/benchmarking_1_0.md) — accepted 1.0 benchmark scope.
- [behavioral_graduation_candidates.md](../../plans/behavioral_graduation_candidates.md) — Tier 1 row 1 lifecycle.
- [11_cradle_sensorimotor_poc.md](../11_cradle_sensorimotor_poc.md) — Cradle infrastructure substrate.
- [10_cross_session_enrichment.md](../10_cross_session_enrichment.md) — partial prior evidence + `--resume-sim` mechanics.
- [36_roy_5b_confound_isolation.md](../36_roy_5b_confound_isolation.md) — confound-isolation discipline (Arm C exists to honor this).
- [scripts/run_v1_phases.sh](../../../scripts/run_v1_phases.sh) — closest-sibling paired-trial harness (the design template).
- [feedback_invariant_two_tier_tracking.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_invariant_two_tier_tracking.md) — graduation-tag discipline.