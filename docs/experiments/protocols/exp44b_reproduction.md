# Exp 44b — Reproduction runbook

How to reproduce the counterfactual campaign from a clean machine. Written from the
2026-08-10/11 pilot, so every gotcha below is one we actually hit.

**Never run this on the leader** (Exp 37 cascade lesson). Any second machine works; the
campaign is self-contained under its own `MAXIM_DATA_HOME`s and touches nothing in
`~/.maxim` except the shared model cache (symlinked read-only), so it is safe to run
while other work continues elsewhere.

## 0. Setup (once)

```bash
git clone <repo> ~/Maxim-exp44b && cd ~/Maxim-exp44b
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[llm-llama,llm-server,semantic]"
export PYTHONPATH="$HOME/Maxim-exp44b/src"        # absolute, own line (Exp 42b lesson)

# MANDATORY — config.json is the single source of truth and is NOT redirected by
# MAXIM_DATA_HOME, so sub-sims inherit it. Drift between the budgeter's n_ctx and
# the served n_ctx silently 500s the capture stage.
maxim config set llm.profile qwen2.5-32b-instruct
maxim config set llm.n_ctx 16384
maxim doctor 2>/dev/null | grep -i "n_ctx\|profile\|vram_context_fit"   # verify FIT
```

## 1. Pilot (1 seed/arm) — always before a confirmatory run

```bash
PYTHONUNBUFFERED=1 nohup caffeinate -is python scripts/exp44/campaign.py \
  --config scripts/exp44/campaign_44b.json \
  --workdir ~/exp44b/pilot \
  --arms A_green_safe,B_purple_safe,CTRL_transplant_A_into_B --seeds 1 \
  > ~/exp44b/run.out 2>&1 &

pgrep -fl campaign.py        # MUST be exactly one (the runner also enforces this)
```

Notes that cost us time:
- `PYTHONUNBUFFERED=1` — without it the runner's few stdout lines sit in the buffer and
  `run.out` looks dead, which is what triggered three double-launches.
- Keep `run.out` **outside** the workdir so cleaning the workdir cannot eat it.
- `--arms` is load-bearing once the 44c companion arms exist in the config: omitting it
  starts those too.
- Per-stage progress lives in `arms/<arm>/seed<N>/logs/{learn,capture}.log`, **not** in
  `run.out`. Learn runs at tick speed; capture is ~50-90 s/turn on a 32B local model.

## 2. Read the verdicts (the manifest is the source of truth)

```bash
tail -10 ~/exp44b/pilot/manifest.jsonl
```

| stage | pass condition |
|---|---|
| `campaign_start` | `executed_git_hash` matches `git rev-parse HEAD` |
| `learn` | `ok: true`, `max_abs_cluster_bias ≥ 0.9` (44c collision arm: ≥ 0.15) |
| `capture` | `ok: true`, `n_pairs ≥ 5`, `annotation_fraction ≥ 0.5` |
| `requery` | `ok: true`, named file exists, no `.partial` left behind |

```bash
# transplant-control validity
ls ~/exp44b/pilot/arms/CTRL_transplant_A_into_B/seed1/control_void.json 2>/dev/null \
  && echo "CONTROL VOID" || echo "control VALID"
```

## 3. Stats (free to re-run; reads cached re-query files)

```bash
cd ~/Maxim-exp44b && git pull      # safe: stats-only re-analysis
PYTHONPATH=src python scripts/exp44/stats_counterfactual.py \
  --campaign ~/exp44b/pilot --config scripts/exp44/campaign_44b.json
cat ~/exp44b/pilot/stats.json
```

⚠️ **Do NOT re-run `campaign.py` right after pulling** unless you intend to execute every
arm the newest config contains.

## 4. Mandatory qualitative checks (statistics alone will mislead you)

The pilot's most important findings came from reading artifacts, not from `stats.json`.

```bash
# 4a. What does the annotation ACTUALLY say? (found the name-mismatch + entangled-axes
#     findings — see 44b_pilot.md F1/F2/F3)
python3 -c "
import json
r = json.loads(open('$HOME/exp44b/pilot/arms/A_green_safe/seed1/capture.jsonl').readline())
print('\n'.join(l for l in r['prompt_full'].splitlines() if 'prior experience' in l.lower()))"

# 4b. Full vs ablated, end to end — confirm the ONLY delta is the annotation
python3 -c "
import json
r = json.loads(open('$HOME/exp44b/pilot/arms/A_green_safe/seed1/capture.jsonl').readline())
print(r['prompt_full']); print('='*100); print(r['prompt_ablated'])" | less

# 4c. Determinism of temp-0 re-query (the method's core assumption)
PYTHONPATH=src MAXIM_LLM_PROFILE=qwen2.5-32b-instruct python scripts/exp44/rerun_ablated_offline.py \
  --log ~/exp44b/pilot/arms/A_green_safe/seed1/capture.jsonl \
  --out /tmp/det_probe.jsonl --entropy-samples 0 --determinism-check 12

# 4d. Capture integrity (duplicate decision_ids ⇒ two writers raced the file)
python3 -c "
import json; from collections import Counter
rows=[json.loads(l) for l in open('$HOME/exp44b/pilot/arms/A_green_safe/seed1/capture.jsonl') if l.strip()]
d={k:v for k,v in Counter(r['decision_id'] for r in rows).items() if v>1}
print(f'{len(rows)} rows; duplicates: {d or \"NONE — single writer\"}')"
```

## 5. Confirmatory run (only after the freeze is signed)

Sign the checklist in [exp44b_preregistration.md](exp44b_preregistration.md), commit it
plus `campaign_44b.json` (that commit hash **is** the freeze), then:

```bash
PYTHONUNBUFFERED=1 nohup caffeinate -is python scripts/exp44/campaign.py \
  --config scripts/exp44/campaign_44b.json --workdir ~/exp44b/confirmatory_run1 \
  > ~/exp44b/confirmatory.out 2>&1 &
```

Before analysis: `git status` clean, and the manifest's `executed_git_hash` equal to the
freeze commit.

## Recovery

| symptom | cause | action |
|---|---|---|
| `ERROR: another campaign (pid N) already holds …` | second launch | correct — tail logs, do not relaunch |
| learn loops at `passed_gate=False`, bias 0.0, no `drive:cold` line | wrong body | config `embodiment` must be `bodies/infant_humanoid_chilled` |
| `annotation_fraction` 0.0 on a substrate arm | broken counter (pre-#489) or substrate not surfacing | pull; then grep the capture for `prior experience` |
| capture wedged, `down_500` / `_llm_unavailable` in the log | budgeter vs served n_ctx drift | fix via `maxim config`, kill, relaunch |
| duplicate `decision_id`s | two writers (pre-#494) | delete `capture.jsonl` + `capture_verified.json`, re-run (learn stays cached) |
| stale lock after a crash | holder pid dead | automatic takeover; no manual `rm` needed |

**Everything is resumable.** Verified stage markers (`learn_verified.json`,
`capture_verified.json`) are skipped on re-run and carry config fingerprints, so editing
the config invalidates exactly the affected stages. Killing mid-stage loses only that
stage.
