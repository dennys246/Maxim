# Exp 44 — LLM-primary body_state ablation: overnight run-card

> **⏸ DEFERRED 2026-08-27 — runbook for the body-state arms A/B/C that never validly ran. **Revive with** `acting_coach_body_state_ablation.md`; re-verify the drive-pain cadence caveat before any overnight burn.**

**Purpose:** a step-by-step operational runbook for running the Exp 44 arms on the
Mac Mini, with the **go/no-go gates baked in** so an overnight run doesn't burn ~48h
reproducing the 2026-07-15 blockers. Pre-registration + arm design live in
[acting_coach_body_state_ablation.md](acting_coach_body_state_ablation.md); the
vehicle + validity floor in [controlled_llm_primary_embodied_harness.md](../archive/controlled_llm_primary_embodied_harness.md).
Read those for the *why*; this is the *how*.

## ⚠️ Read first — is it even worth running tonight?

Exp 44 is an **LLM sim**, not a scripted probe: ~100–150 s/turn × ~24 turns × seeds × 3 arms
≈ **1–2 days** for a full 10-seed sweep. Two open questions the docs flag that should be
answered before committing that:

1. **Track 1 changed the drive-pain cadence** (per-iteration `evaluate_failures`, now
   state-based not onset-based). Prior Exp 44 numbers are stale, and
   [transition_based_drive_pain.md](transition_based_drive_pain.md) (revival trigger
   fired) arguably should land first so the drive-pain being measured is onset-based. **If it
   hasn't landed, note that this run measures the state-based cadence** and flag it in the writeup.
2. **Strategic:** the harness doc explicitly says decide "run the arms vs invest in
   substrate-native work" before committing 48h — Exp 44 is substrate-signal-into-LLM
   scaffolding, in tension with the grounded-language pivot. The calibrated prior is **A ≈ B ≈ C
   within noise** (a valid null that settles "wire body_state?" and corrects the B3.1 "shipped"
   overclaim). That's a legitimate result, but weigh it against the cost.

**Recommendation:** run **Step 2 (one attended validation seed, ~30 min) tonight**, read its
gates, and only launch the Step 3 overnight sweep if it passes. Do NOT skip Step 2. If you want
guaranteed-productive overnight compute with zero babysitting, the **S4 orient replication seeds
(no LLM) are the safer use** — Exp 44 wants a human to confirm the first seed acts.

---

## Step 1 — ops preflight (MANDATORY; every 2026-07-15 blocker was here)

Do NOT co-locate a `maxim-leader` / experiment session on this box while the sweep runs (Exp 37
cradle-cascade). Then pin model + n_ctx through config (single source of truth):

```bash
# 1. Serve the RIGHT model — the AUT is a tool-caller; a reasoning model (r1-distill) emits
#    0 actions. Spawn qwen at 16k so the server context matches the budgeter's belief.
MAXIM_LLM_N_CTX=16384 maxim --llm qwen2.5-32b-instruct   # spawns/replaces the :8100 server at 16k

# 2. Persist it so sub-sims inherit it (config.json is the single source; --llm alone does NOT
#    update llm.profile, so the singleton check would later see drift).
maxim config set llm.profile qwen2.5-32b-instruct
maxim config set llm.n_ctx 16384

# 3. VERIFY before spending a night on it:
maxim doctor 2>/dev/null | grep -iE "n_ctx|profile|llm"      # profile=qwen2.5-32b, n_ctx=16384
curl -s http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-32b-instruct","messages":[{"role":"user","content":"say hi"}],"max_tokens":8}' \
  | head -c 300      # a clean JSON reply = server healthy; a 500 = n_ctx/overflow, STOP and fix
```

**Gate 1:** doctor shows qwen + n_ctx 16384, and the curl returns a real completion (not a 500).
If the curl 500s, the budgeter↔server n_ctx are still misaligned — do not proceed.

---

## Step 2 — ONE attended validation seed (the go/no-go; ~30 min)

Run a single arm-A seed at a reduced turn cap and WATCH it. This confirms the three things that
gate the whole sweep. Direct arc run (live visibility; bypasses the harness subprocess capture):

```bash
MAXIM_DETERMINISTIC_SCENE_EMBODIMENT=1 MAXIM_DISABLE_IMAGINATION=1 \
MAXIM_LOG_FILE=/tmp/exp44_valseed.jsonl \
  maxim --sim cradle_pref_a --embodiment bodies/infant_humanoid_chilled \
    --aut-mode llm-primary --aut-model qwen2.5-32b-instruct \
    --interactive false --sim-max-turns 24 2>&1 | tee /tmp/exp44_valseed.log
```

- `MAXIM_DETERMINISTIC_SCENE_EMBODIMENT=1` — G1: `warmth_alpha_harm`'s `self_effect` writes to
  the AUT body so harm is deterministic + attributable (not narrator-improvised). **Load-bearing
  for validity** — without it the safe-vs-harm signal degenerates.
- `MAXIM_DISABLE_IMAGINATION=1` — controlled arc presents only its two declared warmth entities
  (no improvised distractors).

**Gate 2 — read all three from the log / `actions.jsonl` before launching the sweep:**
1. **Does the AUT actually act?** Nonzero `warmth_alpha_harm` / `warmth_beta_safe` affordance
   calls. If it fritters on `respond`/`sense` and never warms → tool-calling reliability is the
   dominant noise source; the arm effect is unmeasurable. STOP (this is G2 territory — do not
   patch it in, it confounds the arms; reconsider the run).
2. **Does harm fire?** Grep the log for an `arms.thermal` breach / embodiment_failure when
   `warmth_alpha_harm` is called. If harm is absent even with the deterministic flag → G1 isn't
   actually wired for this body/arc; STOP and fix before any arm means anything.
3. **Is there headroom?** Eyeball arm-A `safe_pref`. If ~0.95+, Qwen32B's "don't touch what
   burns you" prior is already at ceiling — the primary metric can't move, and the only possible
   signal is the secondary drive-regulation metric (likely via arm B). Decide whether that's
   worth the sweep or whether the null is already the answer.

**Also verify** `cradle_pref_a` loads as an ARC (warmth affordances registered), not as a
free-text goal string — a prior direct run mis-loaded it as a goal. If the log shows no
`warmth_*` tools registered, the arc didn't load; STOP.

---

## Step 3 — the overnight arm sweep (ONLY if Gate 2 passed)

The harness (`benchmark_exp42_preference.py`) runs sub-sims via `subprocess.run(capture=True)`,
so the terminal is **silent for the whole run by design** — `--out` grows per completed (arm, seed).
Silence ≠ hang. `--resume` skips completed pairs, so a partial overnight run continues next session.

Arms differ ONLY by env vars exported in the launching shell (recorded per-run for provenance):

| Arm | body_state prompt | coach body layers | env |
|-----|-------------------|-------------------|-----|
| **A** (status quo) | OFF | — | *(nothing)* |
| **B** (fresh body_state, no coach layers) | ON | OFF | `MAXIM_ENABLE_BODY_STATE_PROMPT=1 MAXIM_DISABLE_COACH_BODY_LAYERS=1` |
| **C** (full: body_state + coach layers) | ON | ON | `MAXIM_ENABLE_BODY_STATE_PROMPT=1` |

**Shared across all arms** (deterministic harm + clean scene, matching the validation seed):
```bash
export MAXIM_DETERMINISTIC_SCENE_EMBODIMENT=1 MAXIM_DISABLE_IMAGINATION=1
```

First pass: **fewer seeds, reduced turn cap** (the doc's advice — a full 10×3 is 1–2 days).
`--arms cradle_pref_a,cradle_pref_b` is the counterbalance (safe/harm source swap), both run per arm.

```bash
# ── Arm A ──
env -u MAXIM_ENABLE_BODY_STATE_PROMPT -u MAXIM_DISABLE_COACH_BODY_LAYERS \
  python scripts/benchmark_exp42_preference.py \
    --aut-mode llm-primary --aut-model qwen2.5-32b-instruct \
    --arms cradle_pref_a,cradle_pref_b --trials 5 --seed-base 42 \
    --sim-max-turns 24 --timeout-s 3000 --resume \
    --out ~/exp44_arms.jsonl

# ── Arm B ──  (body_state ON, coach layers OFF)
MAXIM_ENABLE_BODY_STATE_PROMPT=1 MAXIM_DISABLE_COACH_BODY_LAYERS=1 \
  python scripts/benchmark_exp42_preference.py \
    --aut-mode llm-primary --aut-model qwen2.5-32b-instruct \
    --arms cradle_pref_a,cradle_pref_b --trials 5 --seed-base 42 \
    --sim-max-turns 24 --timeout-s 3000 --resume \
    --out ~/exp44_arms.jsonl

# ── Arm C ──  (full: body_state ON, coach layers ON)
MAXIM_ENABLE_BODY_STATE_PROMPT=1 \
  python scripts/benchmark_exp42_preference.py \
    --aut-mode llm-primary --aut-model qwen2.5-32b-instruct \
    --arms cradle_pref_a,cradle_pref_b --trials 5 --seed-base 42 \
    --sim-max-turns 24 --timeout-s 3000 --resume \
    --out ~/exp44_arms.jsonl
```

Run them **sequentially in one shell** (or a small wrapper) — do not parallelize; the box serves
one qwen32b server and concurrent sub-sims would 500 under contention. All three append to the
same `--out`; the arm env is recorded per-run so the analyzer separates them.

**Provenance:** every run row records its `git_hash` + arm env. **Verify the git_hash contains
the Track-1 cadence + the deterministic-embodiment seam before trusting any verdict** (the
stale-checkout lesson).

---

## Step 4 — read the result (pre-registered)

- Primary metric: `safe_pref` (fraction of warmth choices at the safe source), per arm, pooled
  across the counterbalance (a/b), via the Exp 42 analyzer on `actions.jsonl`.
- Pre-registered verdict: **A ≈ B ≈ C within noise = the valid null** (body_state doesn't move
  LLM-primary safe-preference → settles "wire body_state?" and corrects the B3.1 overclaim). A
  clean separation (B or C > A) is the ~20–30% alternative; if it separates it's most likely B
  (fresh body_state as information) on the secondary drive-regulation metric, not the ceilinged
  primary.
- **Fill the writeup honestly incl. the cadence caveat** (state-based drive-pain if
  transition_based_drive_pain hasn't landed) and the seed count (a 5-seed first pass is not the
  10-seed bar — say so).

## Failure-mode quick ref (from 2026-07-15)

| Symptom | Cause | Fix |
|---|---|---|
| 0 actions in ~475 s | reasoning model served (r1-distill can't tool-call) | Step 1: serve qwen2.5-32b-instruct |
| `down_500` / `_llm_unavailable` / "no eligible providers" | budgeter n_ctx (32768) > server n_ctx (13312) → overflow 500 | Step 1: `MAXIM_LLM_N_CTX=16384` at spawn |
| AUT warms at harm source, feels nothing | `MAXIM_DETERMINISTIC_SCENE_EMBODIMENT` not set (G1 floor) | export it (Steps 2–3) |
| improvised merchant/book distractors in scene | imagination on | `MAXIM_DISABLE_IMAGINATION=1` |
| terminal silent for 20–40 min | harness `subprocess.run(capture=True)` by design | not a hang; watch `--out` grow |
