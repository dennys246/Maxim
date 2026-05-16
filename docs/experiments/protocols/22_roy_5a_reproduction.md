# Reproduction — Roy-5a cosine-localization disambiguator

**Companion:** [22_roy_5a.md](../22_roy_5a.md)
**Iteration spec:** [scenarios/roy/roy_5a_iteration.yaml](../../../scenarios/roy/roy_5a_iteration.yaml)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../../scenarios/roy/roy_2pc_holdout.yaml) (reused from Roy-2pc / Roy-2c / Roy-4)
**A/B partners:** [20_roy_2c.md](../20_roy_2c.md), [21_roy_4.md](../21_roy_4.md)
**Owning plan:** [roy_5_encoder_alignment_disambiguator.md § Stage 1](../../plans/roy_5_encoder_alignment_disambiguator.md)
**Persistence prerequisite:** [PR #248](https://github.com/dennys246/Maxim/pull/248) (wires `EC.save()` + `ATL.save()` into `simulation/report.py::save_aut_state`)

Roy-5a is the **disambiguating diagnostic for the three H1 sub-hypotheses** (H1c threshold tuning / H1b encoder A-B / H1a encoder subspace incompatibility). It re-runs the same priming + fixture + arms as Roy-2c / Roy-4 with one structural change: the persisted EC state (`aut_ec.json`, written by `EntorhinalCortex.save()` since PR #248) is now read post-hoc by [`scripts/analyze_roy_5_cosine_localization.py`](../../../scripts/analyze_roy_5_cosine_localization.py), which computes three pairwise cosine matrices (`M_tt` / `M_dt` / `M_dd`) between priming and arm-A centroids and decodes max cosine over food-bearing priming clusters into one of three pre-registered sub-hypotheses.

## Pre-registered decoding

| `max(M_tt food-bearing)` | Verdict | Stage 2 next step |
|---|---|---|
| `≥ 0.40` | **H1c** — threshold/centroid tuning | 2a env-var sweep (`MAXIM_EC_PATTERN_THRESHOLD_TEXT` / `MAXIM_EC_FROZEN_TEXT`) |
| `0.20 ≤ max < 0.40` | **H1b** — encoder model fit | 2b encoder A/B (kill `LinguisticEncoder._get_encoder` singleton, sweep alternative sentence-transformer models) |
| `< 0.20` (incl. n/a) | **H1a** — encoder subspace incompatibility | 2c → Stage 3 cradle-arc redesign + Hebbian retest |

The "n/a" case (e.g., no text-modality food-bearing centroids exist in priming, so M_tt is empty for food rows) decodes to H1a per [`tests/unit/test_roy_5_cosine_localization.py::test_negative_infinity_decodes_h1a`](../../../tests/unit/test_roy_5_cosine_localization.py). This is the case Roy-5a's actual run produced — see [22_roy_5a.md § Result](../22_roy_5a.md).

## Prerequisites

Same as [21_roy_4_reproduction.md § Prerequisites](21_roy_4_reproduction.md), plus:

- **PR #248 merged.** Without the persistence wiring, `aut_ec.json` does NOT land in session_dirs and the analyzer has no centroid state to read. Verify with `git log --oneline main | grep persist-ec-and-atl` returning a hit.
- `tests/unit/test_save_aut_state.py` should pass on the local checkout — confirms the wiring.
- `tests/unit/test_roy_5_cosine_localization.py` should pass — confirms the analyzer.

## A. Pre-flight (3 min)

```bash
# 1. Spec parses?
maxim roy run scenarios/roy/roy_5a_iteration.yaml --dry-run

# 2. Persistence wiring lives in current checkout?
grep -n "ec=aut_memory_hub.ec\|atl=aut_memory_hub.atl" \
  src/maxim/simulation/orchestrator.py
# Expect 2 matches (one each).

# 3. Analyzer + tests pass?
python -m pytest tests/unit/test_save_aut_state.py tests/unit/test_roy_5_cosine_localization.py -q
# Expect 28 passed.

# 4. Leader healthy?
curl -si --max-time 10 \
  -H "Authorization: Bearer $(awk '/api_key:/ {print $2}' ~/.config/maxim/peer.yml)" \
  "$(awk '/url:/ {print $2}' ~/.config/maxim/peer.yml)/models" | head -3
```

## B. Roy-5a priming + 3 arms (≈25-28 min wall)

```bash
# Kill any stale sims first.
pkill -f "maxim.*sim" 2>/dev/null

# IMPORTANT — load-bearing env vars:
#
# MAXIM_SUBSTRATE_PATH=1   (LOAD-BEARING for the analyzer's verdict)
#   Gates MemoryHub._encoder wiring. Without this set, BioEnrichmentPipeline's
#   text-modality EC fire (bio_enrichment.py:576) short-circuits on
#   encoder=None. The initial Roy-5a run did NOT set this and produced
#   the n/a / empty-matrix path; Roy-5a-substrate-on added it explicitly
#   and produced the structurally clean H1a verdict. ALWAYS SET THIS for
#   any Roy iteration that expects text-modality cosine matrices to be
#   meaningful. (Wire-A in release_0_9_1.md will flip the default once
#   it ships; until then, this is opt-in per release_0_9_1.md Stage 2.)
#
# MAXIM_EC_TRACE_ACTIVATIONS=1   (not load-bearing for verdict, useful for cross-check)
#   The analyzer reads centroids from disk (aut_ec.json), not from JSONL
#   trace events. But the per-tick traces remain useful as a cross-check
#   on which priming nodes fired in which test ticks — and on whether
#   text-modality routing is firing at all (zero text fires in the
#   trace JSONL is the signal that MAXIM_SUBSTRATE_PATH wasn't set).
#
# The load-bearing artifact is the per-session aut_ec.json, written
# automatically since PR #248.

MAXIM_SUBSTRATE_PATH=1 \
MAXIM_EC_TRACE_ACTIVATIONS=1 \
MAXIM_LOG_FILE=/tmp/roy_5a_ec_trace.jsonl \
  maxim roy run scenarios/roy/roy_5a_iteration.yaml > /tmp/roy_5a_run.log 2>&1
```

The runner orchestrates: 5 priming stages (act1_neonatal_a → act1_neonatal_b → act2_cradle_a → act2_cradle_b → act3_consolidation) at 10 turns each, then 3 arms (a / b / c) at 10 turns each from the engineered `roy_2pc_holdout.yaml` fixture. Per-stage session_ids land in `~/.maxim/roy/roy-5a/result.json`; the priming `final_session_id` is the session arm A resumes from.

## C. Post-run analysis (1 min)

```bash
# Extract the 4 session_ids from the run result.
PRIMING_SID=$(jq -r '.priming.final_session_id' ~/.maxim/roy/roy-5a/result.json)
ARM_A_SID=$(jq -r '.arms.a.session_id'   ~/.maxim/roy/roy-5a/result.json)
ARM_B_SID=$(jq -r '.arms.b.session_id'   ~/.maxim/roy/roy-5a/result.json)
ARM_C_SID=$(jq -r '.arms.c.session_id'   ~/.maxim/roy/roy-5a/result.json)

# Confirm all four sessions have aut_ec.json + aut_nac.json.
for sid in $PRIMING_SID $ARM_A_SID $ARM_B_SID $ARM_C_SID; do
  ls ~/.maxim/sim_reports/$sid/aut_ec.json ~/.maxim/sim_reports/$sid/aut_nac.json
done

# Run the analyzer.
python scripts/analyze_roy_5_cosine_localization.py \
  --priming-dir ~/.maxim/sim_reports/$PRIMING_SID \
  --arm-a-dir   ~/.maxim/sim_reports/$ARM_A_SID \
  --arm-b-dir   ~/.maxim/sim_reports/$ARM_B_SID \
  --arm-c-dir   ~/.maxim/sim_reports/$ARM_C_SID \
  --output-json /tmp/roy_5a_analysis.json
```

Expected output structure:

```
Priming dir:         ~/.maxim/sim_reports/<priming_sid>
  EC centroids:      N from <priming_sid>/aut_ec.json
  Food-bearing clusters: M from <priming_sid>/aut_nac.json
Arm a dir:           ~/.maxim/sim_reports/<arm_a_sid>
  EC centroids:      K from <arm_a_sid>/aut_ec.json
…

======================================================================
ROY-5a VERDICT: <H1c | H1b | H1a>
======================================================================
  max(M_tt food-bearing) = <value or n/a>
  Decoding: max < 0.2 → H1a; 0.2 ≤ max < 0.4 → H1b; max ≥ 0.4 → H1c
  Next stage: <stage description>

  <verdict explanation>
======================================================================

=== Arm a ===
  Priming centroids:  text=N  interoception=N
  Arm centroids:      text=N  interoception=N
  Food-bearing priming clusters: M  (in text modality: K)
  Max cosine over food-bearing priming centroids:
    M_tt: <value or n/a>
    M_dt: <value or n/a>
    M_dd: <value or n/a>

…
```

Exit code:
- `0` — clean H1c / H1b / H1a verdict on arm A.
- `2` — indeterminate (missing required input, no food-bearing priming centroids found, priming has zero centroids).
- `1` — analysis ran but arm A produced no usable signal in any modality pair (empty M_tt AND empty M_dt AND empty M_dd). Should never happen if Roy-5a ran to completion against the standard fixture.

## D. Food-cluster extraction (cross-check)

The analyzer's `load_priming_food_clusters` reads `aut_nac.json::cluster_reward_bias` and pulls the middle field from any 3-field UTS-separator compound key whose third field ends in `sense_food_source`. Cross-check by hand:

```bash
python3 -c "
import json
with open('$HOME/.maxim/sim_reports/$PRIMING_SID/aut_nac.json') as f:
    d = json.load(f)
flat = d.get('cluster_reward_bias', {})
food = set()
for k in flat:
    parts = k.split('\\x1f')
    if len(parts) == 3 and parts[2].endswith('sense_food_source'):
        food.add(parts[1])
print(f'food clusters: {sorted(food)}')
print(f'count: {len(food)}')
"
```

This is the same 3-field UTS-separator pattern [`scripts/analyze_roy_4_coactivation.py::load_priming_food_clusters`](../../../scripts/analyze_roy_4_coactivation.py) uses. Both analyzers consume identical NAc compound-key semantics — drift between the two is structurally prevented by [`tests/unit/test_roy_5_cosine_localization.py::TestFoodClusterExtraction`](../../../tests/unit/test_roy_5_cosine_localization.py).

## E. Stage 2 routing (conditional on verdict)

| Verdict | Stage 2 implementation branch | Estimated scope |
|---|---|---|
| **H1c** | 2a — `MAXIM_EC_PATTERN_THRESHOLD_TEXT` / `MAXIM_EC_FROZEN_TEXT` env-var sweep against Roy-2c spec | ~80 LOC |
| **H1b** | 2b — kill `LinguisticEncoder._get_encoder` singleton, A/B 2-3 alternative sentence-transformer models | ~200 LOC |
| **H1a** (Roy-5a's actual outcome) | 2c → Stage 3 cradle-arc redesign + Hebbian retest (re-runs `scripts/analyze_roy_4_coactivation.py`) | ~250 LOC (Stage 3); +780 LOC iff Stage 3 PASSES and Stage 4a resurrects cross_modal_substrate_binding.md |

Each Stage 2 branch ships its own PR with two-lens pre-merge review per [feedback_review_before_ship.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md). The verdict is the **input** to Stage 2 routing, not the green-light to start implementation immediately — see [22_roy_5a.md § Recommended next step](../22_roy_5a.md) for the user-action items that gate Stage 3 specifically (which is the branch Roy-5a's run triggered).

## Troubleshooting

- **`aut_ec.json` missing** — PR #248 didn't merge before the run. Re-run after `git pull --ff-only main` shows `feat(sim): persist EC + ATL state in sim_reports session dirs` in `git log -3`.
- **No food-bearing clusters found** — priming arc didn't fire `sense_food_source` enough times for NAc to register a `cluster_reward_bias` entry. Re-run; if reproducible, inspect the priming arc YAML.
- **`max(M_tt food-bearing) = n/a` (H1a verdict) but you expected H1c/H1b** — Roy-5a's actual run produced this. Check the EC-trace capture table in [22_roy_5a.md § Result](../22_roy_5a.md) for the "text-modality silence" interpretation and the user-action items in § Recommended next step.
- **Different cluster IDs between priming and arm A** — expected on arm B/C (blank substrate produces fresh UUIDs); UNEXPECTED on arm A (substrate-primed arm should restore the priming cluster IDs). If arm A has fresh UUIDs, the resume_session load path didn't pick up `aut_ec.json`. Confirm `MAXIM_LOG_FILE=/tmp/roy_5a_ec_trace.jsonl grep -c "Restored AUT EC" /tmp/roy_5a_run.log` returns ≥1.

## Cross-iteration notes

Roy-5a uses the **same** priming arc structure as Roy-2c / Roy-4 / Roy-2pc. The single structural change vs Roy-4 is the EC persistence wiring landed in PR #248. Iteration-to-iteration variance in priming activity (e.g., whether text-modality nodes fire, how many, which cluster IDs land in NAc bias) is expected — the analyzer's verdict is robust to this variance because it keys off **what's in the persisted EC dump at session end**, not what fired transiently during the run.

The trace-event count divergence between Roy-4 (text=154, interoception=152) and Roy-5a (text=0, interoception=151) is documented in [22_roy_5a.md § Result](../22_roy_5a.md) and flagged for follow-up investigation. It does not change the H1a verdict — the verdict triggers on the absence of text-modality food clusters in the persisted state, observable directly from `aut_ec.json`, and that condition holds regardless of how many text events fired transiently.
