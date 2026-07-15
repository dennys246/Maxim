# Reproduction — Roy-5b cross-modal binding retest with naming-event scaffolding

**Companion:** [35_roy_5b.md](../35_roy_5b.md)
**Iteration spec:** [scenarios/roy/roy_5b_iteration.yaml](../../../scenarios/roy/roy_5b_iteration.yaml)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../../scenarios/roy/roy_2pc_holdout.yaml) (reused from Roy-2pc / Roy-2c / Roy-4)
**Baseline:** [21_roy_4.md](../21_roy_4.md) — the Roy-4 FAIL that motivates Roy-5b
**Stage 1 prereq:** [22_roy_5a.md](../22_roy_5a.md) — H1a confirmation
**Owning plan:** [roy_5_encoder_alignment_disambiguator.md § Stage 3](../../plans/archive/roy_5_encoder_alignment_disambiguator.md)
**Phase 1 PR (prerequisite):** #295 (merged 2026-05-28)

Roy-5b retests the Hebbian binding rule that Roy-4 refuted, with the **single structural change** of a deliberate drive→linguistic co-firing scaffold via the `infant_humanoid_naming_v1` body. The analyzer (`scripts/analyze_roy_4_coactivation.py`) is reused as-is — only the upstream sim differs.

## Pre-registered diagnostic

| Outcome | Diagnosis |
|---|---|
| **PASS** — at least one test-phase active node has a would-have-bound edge to a priming `sense_food_source` cluster under DEFAULT rule | Naming-event scaffold rescues the binding mechanism. Greenlight Stage 4a: resurrect [cross_modal_substrate_binding.md](../../plans/archive/cross_modal_substrate_binding.md) with the corrected-scaffold prerequisite. |
| **FAIL** — zero would-have-bound edges between priming and test clusters across the full Roy-4 parameter sweep | Mechanism is dead even under deliberately-scaffolded co-firing. Greenlight Stage 4b: promote [jepa_cross_modal_alignment.md](../../plans/deferred/jepa_cross_modal_alignment.md) from "Stage 4b candidate" to "1.2 implementation in flight". |

The Hebbian rule under test is **unchanged** from Roy-4:
- **Tick bucket:** events sharing the same integer `tick` field (1-second buckets).
- **Min co-firing count:** pair must co-fire in at least 5 ticks during priming.
- **Salience weighting:** each co-firing tick contributes `min(activation_a, activation_b)`.
- **Min bound weight:** 0.5.

Sweep grid (also unchanged): `min_cofire ∈ {1, 2, 3, 5}` × `min_weight ∈ {0.01, 0.1, 0.5}`. **DO NOT relax beyond this range** — the question is whether the SAME rule that failed on Roy-4 succeeds under the corrected scaffold; relaxing thresholds would conflate the scaffold question with a tuning question.

## Prerequisites

Same as [21_roy_4_reproduction.md § Prerequisites](21_roy_4_reproduction.md), plus:

- **Phase 1 infrastructure live** (PR #295 merged at commit 9d9dae6 or later). Verify with:

  ```bash
  .venv/bin/python -c "
  from maxim.embodiment.naming_events import NamingPattern, collect_sensor_values
  from maxim.embodiment.body import Embodiment
  from maxim.embodiment.component_registry import ComponentRegistry
  reg = ComponentRegistry()
  entity = reg.instantiate('bodies/infant_humanoid_naming_v1')
  body = Embodiment(entity)
  entity.modulators['arms'].vital_metrics['thermal'] = 0.85
  values = collect_sensor_values(body)
  assert values['arms.thermal'] == 0.85, 'modulator collector broken'
  assert entity.metadata.get('naming_events') is not None
  print('OK — Phase 1 infrastructure live')
  "
  ```

- **Naming utterance smoke** — direct percept-source validation (no LLM cost):

  ```bash
  .venv/bin/python <<'PY'
  from maxim.embodiment.body import Embodiment
  from maxim.embodiment.component_registry import ComponentRegistry
  from maxim.embodiment.percepts import EmbodimentPerceptSource
  reg = ComponentRegistry()
  entity = reg.instantiate("bodies/infant_humanoid_naming_v1")
  body = Embodiment(entity)
  entity.vital_metrics["hunger"] = 0.85
  entity.vital_metrics["thirst"] = 0.85
  entity.modulators["arms"].vital_metrics["thermal"] = 0.85
  source = EmbodimentPerceptSource(body)
  source._last_poll = 0.0
  percept = source.next_percept()
  text = percept.text if hasattr(percept, "text") else str(percept)
  for u in ("hungry", "thirsty", "warm"):
      assert u in text, f"missing utterance: {u}"
  print("OK — all three utterances reach percept text")
  PY
  ```

  **If the smoke fails (any utterance missing), STOP.** Phase 1's modulator-walking fix didn't reach the runtime path; the full Roy-5b run cannot validate the binding rule.

## A. Pre-flight (3 min)

```bash
# 1. Spec parses?
maxim roy run scenarios/roy/roy_5b_iteration.yaml --dry-run

# 2. Env-var resolver smoke test (inherited from Roy-4 — env var unchanged)
MAXIM_EC_TRACE_ACTIVATIONS=1 python -c "
from maxim.similarity.ec import _ec_trace_enabled
assert _ec_trace_enabled() is True
print('env-var resolver OK')
"

# 3. Phase 1 infrastructure + naming utterance smoke (see Prerequisites above)

# 4. Leader healthy?
curl -si --max-time 10 \
  -H "Authorization: Bearer $(awk '/api_key:/ {print $2}' ~/.config/maxim/peer.yml)" \
  "$(awk '/url:/ {print $2}' ~/.config/maxim/peer.yml)/models" | head -3
```

## B. Roy-5b priming + 3 arms (≈25-28 min wall)

```bash
# Kill any stale sims first.
pkill -f "maxim.*sim" 2>/dev/null

# Clean prior run artifacts so the JSONL contains only this run.
rm -f /tmp/roy_5b_ec_trace.jsonl /tmp/roy_5b_run.log

# IMPORTANT: TWO load-bearing env vars must both be set in the RUNNER
# environment — the YAML carries neither.
#
# MAXIM_SUBSTRATE_PATH=1 — required for LinguisticEncoder → EC text-
#   modality routing. Without it, MemoryHub._encoder is never wired and
#   text-modality EC events are silently zero in the JSONL — the Roy
#   analyzer then produces degenerate "all-interoception" verdicts that
#   LOOK like a substrate result but reflect the env-var gate. See
#   .claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_substrate_path_env_var_for_roy.md
#   for the canonical lesson (we have burned multiple Roy iterations on
#   exactly this gotcha).
#
# MAXIM_EC_TRACE_ACTIVATIONS=1 — instrumentation gate at
#   similarity/ec.py::_ec_trace_enabled. Without it, no sim_ec_activation
#   events emit and the analyzer has nothing to read.
#
# Pair with MAXIM_LOG_FILE to capture the events into a JSONL the
# analyzer can read after the run completes. Without MAXIM_LOG_FILE the
# bridge logger has no JSONL sink and events land only in the sim
# session logs (~/.maxim/sim_reports/<sid>/sim_log_*.jsonl) — usable but
# split per arm.
MAXIM_SUBSTRATE_PATH=1 \
MAXIM_EC_TRACE_ACTIVATIONS=1 \
MAXIM_LOG_FILE=/tmp/roy_5b_ec_trace.jsonl \
MAXIM_BACKEND_TRACE=1 \
  maxim roy run scenarios/roy/roy_5b_iteration.yaml 2>&1 | tee /tmp/roy_5b_run.log
```

Expect ~25-28 min wall — same shape as Roy-4 since the substrate-primary 30s/turn timeout dominates and the priming arc length is identical.

## C. Post-hoc co-activation analysis (~30s)

The analyzer is reused as-is — the disambiguator plan explicitly notes "the analyzer is reusable as-is" because Roy-5b's single-variable change is the embodiment, not the analysis surface.

```bash
# Identify per-arm session ids
jq '.arms | {a: .a.session_id, b: .b.session_id, c: .c.session_id}' \
   ~/.maxim/roy/roy-5b/result.json

# Identify the priming session
jq '.priming.session_id // empty' ~/.maxim/roy/roy-5b/result.json

# Per-arm sim_log JSONL files (these contain EC_TRACE events with native
# sim_log format — distinct from the bridge format in MAXIM_LOG_FILE).
PRIM_SID=$(jq -r '.priming.session_id // empty' ~/.maxim/roy/roy-5b/result.json)
for arm in a b c; do
  SID=$(jq -r ".arms.$arm.session_id" ~/.maxim/roy/roy-5b/result.json)
  ls ~/.maxim/sim_reports/$SID/sim_log_*.jsonl 2>/dev/null
done

# The unified MAXIM_LOG_FILE contains all sessions; partition by session_id
# if you want per-arm JSONLs. Alternatively, point the analyzer at the
# per-session sim_log JSONLs directly.
python scripts/analyze_roy_4_coactivation.py \
    --priming-jsonl /tmp/roy_5b_ec_trace.jsonl \
    --arm-a-jsonl /tmp/roy_5b_ec_trace.jsonl \
    --priming-nac ~/.maxim/sim_reports/$PRIM_SID/aut_nac.json \
    --output-json /tmp/roy_5b_analysis.json
```

Default Hebbian rule: `--min-cofire-count 5 --min-weight 0.5`. Sweep with `--min-cofire-count N --min-weight W`.

## Expected output shapes

### Priming session — EC events

```bash
# Count of EC_TRACE events from the priming session
jq -c 'select(.subsystem == "EC_TRACE")' /tmp/roy_5b_ec_trace.jsonl | wc -l
# Expected: similar order of magnitude to Roy-4 (148 priming events) +
# additional text-modality events from the naming utterances. The exact
# delta depends on how often drives cross thresholds across the 50
# priming turns; expect 50-150 additional text-modality EC events from
# the hungry/thirsty/warm utterances entering the LinguisticEncoder.

# Modality breakdown — KEY DIAGNOSTIC
jq -c 'select(.subsystem == "EC_TRACE") | .modality_tag' \
   /tmp/roy_5b_ec_trace.jsonl | sort | uniq -c
# Expected: BOTH linguistic AND drive present. The naming-event scaffold
# closes Roy-4's "linguistic empty during substrate-primary priming"
# gap. If linguistic is still empty, Phase 1's runtime wiring is broken;
# do NOT proceed to analyzer — investigate.
```

### Pass criteria check

The analyzer prints `ROY-4 OUTCOME: PASS` or `ROY-4 OUTCOME: FAIL` (the script name is Roy-4 because the analyzer is reused; the outcome semantics are Roy-5b's). Exit code mirrors that (0 PASS, 1 FAIL). JSON bundle at `--output-json` includes:

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

If PASS, the corrected-scaffold prerequisite is validated. Stage 4a authorization PR proposes resurrecting the cancelled binding plan.
If FAIL, the mechanism is dead under deliberately-scaffolded co-firing. Stage 4b authorization PR proposes promoting JEPA to 1.2 in-flight.

## What to do if it fails

**Smoke fails (any utterance missing from percept text)** — Phase 1's modulator-walking fix isn't reaching the runtime path. Check (a) the body's metadata.naming_events parsed at construction, (b) `EmbodimentPerceptSource.__init__` populated `self._naming_patterns`, (c) `next_percept` advanced past the rate gate. Do NOT proceed to the full Roy-5b run until this clears.

**`jq '.modality_tag' | sort | uniq -c` shows zero linguistic events in priming** — `MAXIM_SUBSTRATE_PATH=1` was NOT set in the runner environment. The LinguisticEncoder text path is opt-in via that env var; without it, `MemoryHub._encoder=None` and the text path silently short-circuits. See [feedback_substrate_path_env_var_for_roy.md](../../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_substrate_path_env_var_for_roy.md). Re-run with the env var set. Phase 2 of Roy-5b (2026-05-28) initially repeated this gotcha by templating from `21_roy_4_reproduction.md` which omits the env var — Roy-4 ran with it set in user shell.

**Priming food clusters = 0 in the analyzer output** — NAc didn't write `sense_food_source` keys to `_cluster_reward_bias` during priming. Roy-5b's priming arc is shape-identical to Roy-4's, so this would mean a regression in the cradle arc's `sense_food_source` calls. Inspect `~/.maxim/sim_reports/$PRIM_SID/actions.jsonl` for sense_food_source tool calls.

**Analyzer reports PASS only at `--min-cofire-count 1 --min-weight 0.01` but FAIL at default `(5, 0.5)`** — the corrected scaffold produces weak co-firing but not at the level the default Hebbian rule requires. This is ambiguous; surface to user per the disambiguator plan's "DO NOT relax the parameter sweep" rule. The verdict is NOT auto-PASS; record actual sweep numbers and pause for classification.

**Analyzer reports FAIL across the full sweep** — the mechanism is dead under deliberately-scaffolded co-firing. This is the Stage 4b verdict; record actual sweep numbers in exp 35 and propose the JEPA promotion in the authorization PR.

## Cleanup

```bash
unset MAXIM_EC_TRACE_ACTIVATIONS MAXIM_LOG_FILE MAXIM_BACKEND_TRACE
# Roy artifacts persist in ~/.maxim/roy/ and ~/.maxim/sim_reports/ — keep them for analysis.
```

## What changed vs Roy-4

The single-variable change is the embodiment:

```bash
diff scenarios/roy/roy_4_iteration.yaml scenarios/roy/roy_5b_iteration.yaml | head -50
```

The Roy-4 spec uses `bodies/infant_humanoid` (no naming events); Roy-5b uses `bodies/infant_humanoid_naming_v1` (declares `naming_events:` metadata). Same priming arc shape, same fixture, same arms, same Hebbian rule defaults, same parameter sweep.

The structural difference flows through `EmbodimentPerceptSource`: when the body opts in via `naming_events:` metadata, the percept source appends a `=== Body Utterances ===` section to the body-state text on every tick where a drive crossed threshold (with hysteresis preventing per-tick re-emission). The text reaches the substrate-primary AUT via the same percept path the sensor snapshot uses, so `LinguisticEncoder` fires on it in the same agent-loop tick `SensorEncoder` fires on the drive state — closing the co-firing gap Roy-4 identified.

## Related docs

- [`35_roy_5b.md`](../35_roy_5b.md) — outcome doc
- [`21_roy_4.md`](../21_roy_4.md) — Roy-4 baseline (FAIL on standard cradle arc)
- [`22_roy_5a.md`](../22_roy_5a.md) — Stage 1 result, H1a confirmation
- [`34_wire_a_post_fix_a_b.md`](../34_wire_a_post_fix_a_b.md) — the exp that authorized Stage 3
- [`roy_5_encoder_alignment_disambiguator.md`](../../plans/archive/roy_5_encoder_alignment_disambiguator.md) — owning plan; Stage 3 is Roy-5b itself
- [`cross_modal_substrate_binding.md`](../../plans/archive/cross_modal_substrate_binding.md) — Stage 4a destination (resurrected on PASS)
- [`jepa_cross_modal_alignment.md`](../../plans/deferred/jepa_cross_modal_alignment.md) — Stage 4b destination (promoted on FAIL)
- [`persona_convergence_crucible.md`](../../plans/deferred/persona_convergence_crucible.md) — Roy iteration log
