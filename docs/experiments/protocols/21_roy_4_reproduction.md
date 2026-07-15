# Reproduction — Roy-4 EC-activation co-activation instrumentation

**Companion:** [21_roy_4.md](../21_roy_4.md)
**Iteration spec:** [scenarios/roy/roy_4_iteration.yaml](../../../scenarios/roy/roy_4_iteration.yaml)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../../scenarios/roy/roy_2pc_holdout.yaml) (reused from Roy-2pc / Roy-2c)
**A/B partner:** [20_roy_2c.md](../20_roy_2c.md)
**Owning release plan:** [release_0_9_1.md § Stage 0d](../../plans/archive/release_0_9_1.md)
**1.1 design plan:** [cross_modal_substrate_binding.md § Stage 1](../../plans/archive/cross_modal_substrate_binding.md)

Roy-4 is the **validation gate for the 1.1 cross-modal binding implementation**. It re-runs the same priming + fixture + arms as Roy-2c with one structural change: `MAXIM_EC_TRACE_ACTIVATIONS=1` set in the runner environment. The instrumentation emits per-tick `sim_ec_activation` JSONL events from `similarity/ec.py::pattern_complete_or_separate`; the post-hoc analyzer (`scripts/analyze_roy_4_coactivation.py`) computes a pairwise co-activation matrix over the priming session, applies a proposed Hebbian binding rule, and reports whether any test-phase active nodes have would-have-bound edges to priming `sense_food_source` clusters.

## Pre-registered diagnostic

| Outcome | Diagnosis |
|---|---|
| **PASS** — at least one test-phase active node has a would-have-bound edge to a priming `sense_food_source` cluster | Cross-modal binding plan is JUSTIFIED. Greenlight Stages 2-6 of [cross_modal_substrate_binding.md](../../plans/archive/cross_modal_substrate_binding.md). |
| **FAIL** — no would-have-bound edges between priming and test clusters under the proposed rule | Encoder alignment is too severe for Hebbian binding alone. Cancel binding plan Stages 2-6; redirect to a 1.2+ encoder-replacement research direction. |

The proposed Hebbian rule under test:
- **Tick bucket:** events sharing the same integer `tick` field (1-second buckets in the default instrumentation).
- **Min co-firing count:** pair must co-fire in at least 5 ticks during priming.
- **Salience weighting:** each co-firing tick contributes `min(activation_a, activation_b)`; the bound-edge weight is the sum across all co-firings.
- **Min bound weight:** 0.5 (overrideable via `--min-weight`).

## Prerequisites

Same as [20_roy_2c_reproduction.md § Prerequisites](20_roy_2c_reproduction.md), plus:

- **0.9.1 codebase or later** with Stage 0d (`MAXIM_EC_TRACE_ACTIVATIONS` env-var gate at `similarity/ec.py::_ec_trace_enabled`). Older checkouts ignore the env var; no events get emitted regardless.
- `tests/unit/test_ec_trace_activations.py` should pass on the local checkout — confirms the env-var resolver + emission path.

## A. Pre-flight (3 min)

```bash
# 1. Spec parses?
maxim roy run scenarios/roy/roy_4_iteration.yaml --dry-run

# 2. Env-var resolver smoke test (0.9.1+ only)
MAXIM_EC_TRACE_ACTIVATIONS=1 python -c "
from maxim.similarity.ec import _ec_trace_enabled
assert _ec_trace_enabled() is True
print('env-var resolver OK')
"

# 3. End-to-end emission smoke test
MAXIM_EC_TRACE_ACTIVATIONS=1 PYTHONPATH=src python -c "
from maxim.simulation import sim_logger as sl
sl.enable_sim_logging(log_path='/tmp/ec_emit_smoke.jsonl', debug=False)
from maxim.similarity.ec import EntorhinalCortex
ec = EntorhinalCortex()
tok = sl._current_agent_id.set('smoke-1')
try:
    r1 = ec.pattern_complete_or_separate(embedding=[1.0, 0.0, 0.0], modality='text')
    ec.register_substrate_node(r1.node_id, [1.0, 0.0, 0.0], 'text')
    ec.pattern_complete_or_separate(embedding=[1.0, 0.0, 0.0], modality='text')
finally:
    sl._current_agent_id.reset(tok)
sl.disable_sim_logging()
"
grep -c EC_TRACE /tmp/ec_emit_smoke.jsonl  # Expect 2
rm /tmp/ec_emit_smoke.jsonl

# 4. Leader healthy?
curl -si --max-time 10 \
  -H "Authorization: Bearer $(awk '/api_key:/ {print $2}' ~/.config/maxim/peer.yml)" \
  "$(awk '/url:/ {print $2}' ~/.config/maxim/peer.yml)/models" | head -3
```

## B. Roy-4 priming + 3 arms (≈25-28 min wall)

```bash
# Kill any stale sims first.
pkill -f "maxim.*sim" 2>/dev/null

# IMPORTANT: MAXIM_EC_TRACE_ACTIVATIONS=1 is the load-bearing variable.
# It must be set in the RUNNER environment — the YAML does NOT carry it.
# The env-var reader at similarity/ec.py::_ec_trace_enabled reads
# os.environ at every pattern_complete_or_separate call.
#
# Pair with MAXIM_LOG_FILE to capture the events. Without MAXIM_LOG_FILE
# the bridge logger has no JSONL sink and events land only in the sim
# session log (~/.maxim/sim_reports/<sid>/sim_log_*.jsonl) — usable but
# split per arm.
MAXIM_EC_TRACE_ACTIVATIONS=1 \
MAXIM_LOG_FILE=/tmp/roy_4_ec_trace.jsonl \
MAXIM_BACKEND_TRACE=1 \
  maxim roy run scenarios/roy/roy_4_iteration.yaml 2>&1 | tee /tmp/roy_4_run.log
```

Expect ~25-28 min wall — same shape as Roy-2c since the substrate-primary 30s/turn timeout dominates.

## C. Post-hoc co-activation analysis (~30s)

```bash
# Identify per-arm session ids
jq '.arms | {a: .a.session_id, b: .b.session_id, c: .c.session_id}' \
   ~/.maxim/roy/roy-4/result.json

# Identify the priming session
jq '.priming.session_id // empty' ~/.maxim/roy/roy-4/result.json

# Per-arm sim_log JSONL files (these contain EC_TRACE events with native
# sim_log format — distinct from the bridge format in MAXIM_LOG_FILE).
PRIM_SID=$(jq -r '.priming.session_id // empty' ~/.maxim/roy/roy-4/result.json)
for arm in a b c; do
  SID=$(jq -r ".arms.$arm.session_id" ~/.maxim/roy/roy-4/result.json)
  ls ~/.maxim/sim_reports/$SID/sim_log_*.jsonl 2>/dev/null
done

# The unified MAXIM_LOG_FILE contains all sessions; partition by session_id
# if you want per-arm JSONLs. Alternatively, point the analyzer at the
# per-session sim_log JSONLs directly.
python scripts/analyze_roy_4_coactivation.py \
    --priming-jsonl /tmp/roy_4_ec_trace.jsonl \
    --arm-a-jsonl /tmp/roy_4_ec_trace.jsonl \
    --priming-nac ~/.maxim/sim_reports/$PRIM_SID/aut_nac.json \
    --output-json /tmp/roy_4_analysis.json
```

Default Hebbian rule: `--min-cofire-count 5 --min-weight 0.5`. Pass `--min-cofire-count N --min-weight W` to sweep.

## Expected output shapes

### Priming session — EC events

```bash
# Count of EC_TRACE events from the priming session
jq -c 'select(.data.subsystem == "EC_TRACE")' /tmp/roy_4_ec_trace.jsonl | wc -l
# Expected: hundreds to low thousands per priming session (50 turns × multiple
# pattern_complete_or_separate calls per turn — sensors, linguistic, drives).

# Modality breakdown
jq -c 'select(.data.subsystem == "EC_TRACE") | .data.modality_tag' \
   /tmp/roy_4_ec_trace.jsonl | sort | uniq -c
# Expected: mix of linguistic, drive, sensor — exact ratio depends on the
# substrate path triggered by sensor encoder + linguistic encoder.
```

### Pass criteria check

The analyzer prints `ROY-4 OUTCOME: PASS` or `ROY-4 OUTCOME: FAIL` to stdout, and the exit code mirrors that (0 PASS, 1 FAIL). The JSON bundle at `--output-json` includes:

- `outcome` — "PASS" or "FAIL"
- `priming_food_clusters` — list of cluster IDs the analyzer identified as `sense_food_source`-keyed
- `arms.a.matched_edges` — the would-have-bound edges connecting a priming food cluster to a test-phase active node

A PASS requires `arms.a.matched_edges` to be non-empty.

### Pre-registered diagnostic (record actual outcome here on re-run)

Run on YYYY-MM-DD: outcome **PASS** / **FAIL**. Top matched edges:

```
node_a ↔ node_b  weight  cofire_count  modality_pair
...
```

If PASS, the binding rule reproduces the result. If FAIL, the binding rule does NOT close the Roy-2c gap under the default `(min_cofire=5, min_weight=0.5)`; sweep tighter and looser parameters with `--min-cofire-count N --min-weight W` to characterize the failure surface.

## What to do if it fails

**`grep -c EC_TRACE /tmp/roy_4_ec_trace.jsonl` returns 0** — env var didn't propagate. Verify with `env | grep MAXIM_EC_TRACE_ACTIVATIONS` BEFORE the run. Note the var must be set in the parent shell, not exported inside the YAML; the YAML has no env-var mechanism.

**EC events present but only `modality_tag=linguistic` (no drives/sensors)** — the cradle priming arc didn't activate the sensor encoder. Check whether `MAXIM_SUBSTRATE_PATH=1` is needed for the embodiment path; on 0.9.1 substrate-primary AUT this should be active by default.

**Priming food clusters = 0 in the analyzer output** — NAc didn't write `sense_food_source` keys to `_cluster_reward_bias`. Re-verify the priming session's `aut_nac.json` contains a `_cluster_reward_bias` entry. If missing, the priming didn't reward the substrate path — investigate the cradle arc's `sense_food_source` calls in `~/.maxim/sim_reports/$PRIM_SID/actions.jsonl`.

**Analyzer reports PASS at `--min-cofire-count 1 --min-weight 0.01` but FAIL at default `(5, 0.5)`** — the binding rule's hyperparameters are over-conservative. Note this in the outcome doc; the 1.1 implementation can tune defaults based on this finding.

**Analyzer reports FAIL across all reasonable `(min_cofire, min_weight)` pairs** — the priming-cluster ↔ test-cluster pairs genuinely never co-fire. This is the H1-encoder-alignment-is-fatal outcome; cancel the binding plan and document the negative result.

## Cleanup

```bash
unset MAXIM_EC_TRACE_ACTIVATIONS MAXIM_LOG_FILE MAXIM_BACKEND_TRACE
# Roy artifacts persist in ~/.maxim/roy/ and ~/.maxim/sim_reports/ — keep them for analysis.
```

## What changed vs Roy-2c

The YAML is functionally identical to `roy_2c_iteration.yaml` except for the `name` field (`roy-4` vs `roy-2c`) and description text. The load-bearing variable is the env var, not the YAML.

```bash
diff scenarios/roy/roy_2c_iteration.yaml scenarios/roy/roy_4_iteration.yaml | head -50
```

## Related docs

- [`21_roy_4.md`](../21_roy_4.md) — outcome doc with the recorded analysis result
- [`20_roy_2c.md`](../20_roy_2c.md) — Roy-2c's H1 confirmation that motivates Roy-4
- [`release_0_9_1.md`](../../plans/archive/release_0_9_1.md) — owning release plan; Stage 0d is the instrumentation work
- [`cross_modal_substrate_binding.md`](../../plans/archive/cross_modal_substrate_binding.md) — 1.1 plan; Stage 1 is Roy-4 itself
- [`persona_convergence_crucible.md`](../../plans/deferred/persona_convergence_crucible.md) — Roy iteration log
