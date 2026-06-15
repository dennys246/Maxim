# Exp 40 — Counter-Prior in the Goldilocks Zone (Qwen-32B) — pre-registration

**Status:** PRE-REGISTERED, frozen 2026-06-15. Executes the Exp 38 §8 follow-up
("re-run at Qwen14B + Qwen32B… Qwen32B's anomalous +1.43 SD signal is the one to
watch"). **Metrics are inherited verbatim from [38_counter_prior_substrate.md](38_counter_prior_substrate.md) §5
— no new metric is defined here.** This doc only states the model, the scope, and
the Goldilocks-specific predictions; results append to the Exp 38 doc (shared
analyzer) and are synthesized back here.

---

## 1. The decisive cell

The counter-prior line has tested two of the four cells of a 2×2:

| | frontier models | Goldilocks-zone local |
|---|---|---|
| **prior-aligned** (safe fire, Exp 37) | dominance / ceiling | **signal** — Qwen-32B `+1.43 SD` |
| **counter-prior** (burning hearth, Exp 38) | dominance ×4 | **❓ this experiment** |

Every counter-prior *dominance* result we hold (Sonnet 4.6, GPT-4o, DeepSeek-V3,
R1-Distill-Qwen-32B) is from a model where Exp 37 found the substrate signal
**already swamped** by the prior. The one model where Exp 37 found a *positive*
substrate signal — base **Qwen2.5-32B-Instruct**, the centre of the Goldilocks
band (see [project_exp37_goldilocks_zone]; Δ `+1.43 SD`) — has **never** had its
prior falsified. So the entire dominance claim rests on cells where the substrate
was expected to lose. This is the gap.

**Why base Qwen-32B specifically, and not R1.** R1-Distill-Qwen-32B is a 32B local
model that *did* run the counter-prior → dominance. But R1 is reasoning-distilled,
and Exp 37/38 showed reasoning **amplifies the prior maladaptively**
([project_exp37_r1_reasoning_amplifies_substrate]). The clean Goldilocks test is
the *base* instruct model — the exact model that produced the `+1.43 SD`
prior-aligned signal — with no reasoning confound.

---

## 2. Design

Identical to Exp 38 except **model** and **arm scope**:

| Field | Value |
|---|---|
| Model | `qwen2.5-32b-instruct` (base; the Exp 37 Goldilocks model) — local GGUF on the leader, **$0** |
| Scenarios | `fire_pit` (consistent control) + `deceptive_fire` (counter-prior) — i.e. `--scenario counter_prior` |
| Arms | **all six** — `A, B, C, B-wire-a-off, B-wire-1-off, B-nac-bias-off` (= Exp 38) |
| Trials | 5 paired |
| `--sim-max-turns` | 12 (unchanged — keeps A's session from over-training, Exp 38 §7.3) |
| `--seed-base` | 42 |

Total: 2 scenarios × 6 arms × 5 trials = **60 sub-sims**, ~30 min/sim local ≈ **~30 h**
(no rush — runs under tmux across sessions, `--resume` recovers any interruption).

The deceptive world, the matched control, the separate-worlds routing
(`cradle_deceptive` replaces fire_pit with the hearth), and the inversion
(`cradle_false_hearth.yaml::warm_self.self_effect`) are all unchanged from Exp 38 §2.

---

## 3. Metrics & verdict — inherited from Exp 38 §5 (FROZEN)

The two pre-registered primaries and the verdict tree are reused **verbatim**:

- **Primary 1 (interaction):** `Δ_deceptive(B−A) − Δ_consistent(B−A)` on
  `warm_self_engagement_fraction`; **PASS iff `interaction / pooled_A_sd ≤ −1.0`**.
- **Primary 2 (first-contact):** `dec_drop > 0` AND `(dec_drop − con_drop) > 0` on
  `first_contact_warm_self`.
- **Verdict tree:** Exp 38 §5 table, emitted verbatim by `analyze_exp37.py`.

**Full verdict tree applies (single fire, fully attributable).** All six arms run,
so the Exp 38 §5 tree is exercised end to end — including the ablation **Secondary**
that gates the positive node. This fire resolves the cell in one pass, no follow-up:

- **Dominance** (B keeps warming the hearth) → **"dominance demonstrated"**, exit 0
  — the cell is filled, the honest thesis is complete.
- **Both primaries PASS + ≥1 ablation reverts toward Arm A** → **"substrate matters"**,
  exit 0 — the positive thesis result, **substrate-attributable in this same fire**.
- **Both primaries PASS, 0 ablations revert** → **"void (not substrate-attributable)"**,
  exit 4 — a real signal the substrate can't explain (prompt / within-session); investigate.
- **`avoids_both`** (B warms less in *both* scenarios by ≥1 SD) → **"void (general caution)"**, exit 4.

No new metric, no moved goalpost — the full Exp 38 frozen tree, run at the Goldilocks model.

---

## 4. Pre-registered predictions (priors, before firing)

- **H1 — dominance even in the zone** (highest prior). Consistent with 4× frontier
  dominance + R1-32B dominance. Completes the honest thesis: *the substrate signal
  is real but survives only when the task agrees with the prior; falsify the prior
  and it collapses at every size, Goldilocks band included.* Nothing in the 1.0
  framing changes — it becomes airtight.
- **H2 — caution in the zone.** Both primaries PASS *and* ≥1 ablation reverts: the
  experienced agent treats the *costly* hearth more warily than the *safe* fire,
  attributable to the substrate **in this fire**. Would be the **first
  adaptive-behavioral positive** in the line and would un-pull the 1.0
  behavioral-drive claim in a scoped band — no follow-up fire needed.
- **H3 — general caution / over-generalization** (`avoids_both`, or avoids even the
  safe fire). Verdict **"void (general caution)"**, exit 4. Maladaptive transfer;
  interesting, not a clean win.

All three are diagnostic and all three improve the 1.0 record.

---

## 5. Run plan (leader, local, $0)

Runs **on the leader** (where the GGUF is served); the cradle-smoke hardenings make
a leader-local fire safe (singleton spawn guard reuses the live server; harness
preflight rejects a stray local spawn). Local-model fires report **zero tokens /
`cost=$0` and a `Missing pricing` WARNING — this is normal**; validate health via
**actions, not tokens** (see `docs/bugs/mac_mini_local_fire_oddities.md`).

```bash
# 0. Serve the model on the leader (runtime swap; does NOT touch config.json by design)
maxim --list-models | grep -i qwen2.5-32b      # confirm the GGUF is downloaded
maxim --llm qwen2.5-32b-instruct                # spawns llama-cpp-server with the 32B

# 1. SMOKE (~30 min, 1 sub-sim) — confirm the hearth surfaces + the metric populates
python scripts/benchmark_cross_session.py \
  --scenario deceptive_fire --model qwen2.5-32b-instruct \
  --arms A --trials 1 --subsim-timeout-s 5400 \
  --out /tmp/exp40_smoke.jsonl
#   PASS the smoke iff the record shows: tool_usage has hearth_* calls,
#   turns populated, warm_self_engagement_fraction non-null (NOT all-zero tokens
#   alone — local fires are always zero-token; check the ACTIONS).

# 2. FULL FIRE (~30 h, 60 sub-sims, all six arms) — tmux across sessions; --resume recovers any kill
python scripts/benchmark_cross_session.py \
  --scenario counter_prior --model qwen2.5-32b-instruct \
  --arms A,B,C,B-wire-a-off,B-wire-1-off,B-nac-bias-off --trials 5 --subsim-timeout-s 5400 \
  --out docs/experiments/data/40_results_qwen32b.jsonl

# 3. ANALYZE — appends a Results section to the Exp 38 doc (shared analyzer);
#    all six arms (analyzer default) so the ablation Secondary is gated, not reported absent
python scripts/analyze_exp37.py \
  --in docs/experiments/data/40_results_qwen32b.jsonl \
  --scenarios fire_pit,deceptive_fire --trials 5 \
  --out docs/experiments/38_counter_prior_substrate.md \
  --heading-suffix "qwen32b (Exp 40, Goldilocks)"
```

Pre-fire validity checklist (Exp 38 §7 + the cross-model playbook):
1. Smoke first — hearth tools surface for **this** model, metric non-null.
2. `qwen2.5-32b-instruct` has no cloud prefix → routes local (no cloud spawn);
   confirm the preflight logs reuse of the served server, not a fresh local spawn.
3. tmux + `--resume`; the leader's `main` carries the harness + `cradle_deceptive`.

---

## 6. What it means for 1.0

This fills the last cell of the counter-prior matrix in the one regime the shipped
1.0 product actually runs in (LLM-AUT **+** substrate annotation) at the one size
where the substrate was ever measurably doing something. **H1 makes the honest
"substrate real but prior-dominated" thesis complete and unattackable; H2 would be
the first scoped positive that un-pulls the behavioral-drive claim — substrate-
attributable in this same fire (all six arms).** Either result strengthens the
release — which is exactly why it runs before the announcement ships.

---

## 7. Regression guard / experiment citation

- **Regression guard (engineering):** reuses Exp 38's guards —
  `tests/behavioral/test_exp37_harness_smoke.py` + `tests/behavioral/test_exp37_analyzer_smoke.py`
  (no new harness/analyzer surface; this is a new model + arm scope only).
- **Experiment (behavioral):** this doc (pre-registration) + the
  "## Results — qwen32b (Exp 40…)" section the analyzer appends to
  [38_counter_prior_substrate.md](38_counter_prior_substrate.md); synthesis folded back here after the fire.
