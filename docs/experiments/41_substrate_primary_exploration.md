# Exp 41 — Substrate-Primary Exploration: Can the unmasked substrate override its own prior? (pre-registration)

**Status:** PRE-REGISTERED — not yet fired. Metrics FROZEN at write time.
**Graduates:** [behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md) Tier 1 row **#6** (substrate-primary AUT mode) — the strong-thesis row reframed-settled at 1.0.
**Builds on:** [39_substrate_primary_counter_prior.md](39_substrate_primary_counter_prior.md) (harness + deceptive arc + the fixation finding) and the exploration policy shipped per [../plans/substrate_exploration_policy.md](../plans/substrate_exploration_policy.md).
**Plan / harness:** `scripts/benchmark_cross_session.py` (substrate-primary path), `src/maxim/simulation/substrate_telemetry.py`, new analyzer `scripts/analyze_exp41_exploration.py` (setup deliverable).

---

## 1. Why this is the test that closes the loop

The 1.0 experiment series settled an honest but deflationary verdict: the substrate is **real and causally-structured** but does **not drive adaptive behavior**. Two facts pin it:

- **Exp 38 / 40 (counter-prior under LLM-AUT):** carried substrate — including direct cross-session pain at the exact entity — does **not** override a wrong LLM prior. Dominance across Sonnet 4.6 / GPT-4o / DeepSeek-V3 / R1 and at base Qwen-32B (the one model with a positive Exp 37 signal). Prior-agreement, not scale, is the gating variable.
- **Exp 39 (substrate-primary, LLM removed from the action path):** the bio-systems form EC clusters, accumulate NAc causal links, and emit drive-conditioned proposals — but **fixate on the first high-confidence link with no exploration**, so the Phase-1 hypothesis test never runs. Mechanism alive, behavior inert.

These jointly imply that **under LLM-AUT you can never cleanly demonstrate substrate-driven behavior** — the LLM prior masks it everywhere except a narrow Goldilocks band, and even there a counter-prior collapses it. The only regime where the substrate can be shown to drive *adaptive* behavior is the one where the LLM prior is absent: **substrate-primary**. Exp 39 opened that regime and immediately hit the wall that this experiment is designed to break.

**The key reframing of "prior" in substrate-primary mode.** With the LLM removed, the `fire→warm` belief does not vanish — it relocates. `NAc.recommend_action` carries a hand-coded **drive-affinity cold-start heuristic** ([decisions/nac.py:1690-1709](../../src/maxim/decisions/nac.py)) that gives warmth-seeking actions a positive score *before any learning*. That heuristic is the substrate-internal analog of the LLM's prior. The deceptive hearth (whose `warm_self` is inverted to hurt) puts that built-in prior in direct conflict with embodied pain feedback. So:

> **Exp 41 asks the actual thesis question, finally unmasked:** given the ability to explore, does the substrate learn — from embodied pain alone, with no language and no LLM — to *override its own built-in drive-affinity prior* and stop performing a harmful-but-prior-favored action within a single session?

A PASS converts the thesis from "real but inert" to "real and adaptive." A FAIL tells us fixation is deeper than selection-stochasticity (the substrate's learning signal itself doesn't differentiate), which is an equally valuable, equally publishable result.

---

## 2. Setup deliverables (before any run)

All three must land before Exp 41 fires. Each is small and independently testable.

1. **Exploration policy** — shipped per [../plans/substrate_exploration_policy.md](../plans/substrate_exploration_policy.md). The on/off toggle and ε / bonus-weight must be controllable from the harness (via `config.json::sim.substrate_explore_*` + its `_FIELD_TO_ENV` env override, per the dev standard). Exploration-off must reproduce Exp 39's deterministic `max()` byte-for-byte (regression-pinned).
2. **`cradle_prelinguistic_deceptive` arc** — **LANDED 2026-06-17** ([simulation/arcs.py](../../src/maxim/simulation/arcs.py), `BUILTIN_ARCS["cradle_prelinguistic_deceptive"]`). Derived from `cradle_prelinguistic` by the same `_swap_fire_to_hearth_phase` transform Exp 38/40 used for `cradle_deceptive`: swap `items/cradle_fire_pit` → `items/cradle_false_hearth`. Invoked by exact arc name. The consistent control stays `cradle_prelinguistic` (safe `cradle_fire_pit`). **Drive tuning still required** — see Spike findings below.
3. **Harness + analyzer — LANDED 2026-06-19.** `scripts/benchmark_exp41_exploration.py` (dedicated substrate-primary 2×2 harness — runs the four arms with `--aut-mode substrate-primary` + the cold body, extracts per-third `harm_rate`/`warm_self_rate` from each run's `actions.jsonl`, append-only records JSONL, `--mock` for CI) + `scripts/analyze_exp41_exploration.py` (FROZEN §4/§5 executor: H1/H2/SD, verdict matrix, exit codes 0/4/5, robust SD≈0 sign-test). It is a *dedicated* harness, not bolted onto `benchmark_cross_session.py` (Exp 37) — the substrate-primary 2×2 shares none of Exp 37's cross-session/cost/cloud/ablation machinery and would have risked that shipped harness. Regression guards: [tests/behavioral/test_exp41_pipeline.py](../../tests/behavioral/test_exp41_pipeline.py) (13 tests: every verdict-matrix corner + metric extraction + mock pipeline).

**Spike findings (2026-06-17 — pre-build, exploration OFF).** A substrate-primary spike on this arc ($0, local) established three facts that reshape the prerequisites (full detail in [../plans/substrate_exploration_policy.md](../plans/substrate_exploration_policy.md) "Empirical validation"):
- **Tool availability is confirmed, not at risk.** After phase-0 activation the proposer sees 35 candidates including `hearth_warm_self`/`hearth_touch`. The hearth is reachable; the original "is the affordance even available" worry is closed.
- **The "prior" in substrate-primary is NOT a warmth-affinity prior on `warm_self`.** The drive-affinity heuristic gives `warm_self`/`touch` **no** cold-start boost; instead the agent fixates on `sense_food_source` (hunger-driven, safe, snowballing causal-link confidence to ~1.66 while the hearth stays flat at ~0.35). So the thing exploration must overcome is **food-fixation**, and the thing that makes warming *relevant* is a **thermal drive deficit** — not an affinity prior toward the hearth.
- **Drive-tuning requirement (load-bearing for a valid run) — ADDRESSED 2026-06-18.** A valid run needs a *sustained* cold so warmth-seeking stays drive-relevant (the spike's −0.15 recovered to 0). Two pieces landed: (1) **root-cause code fix** — `runtime/agent_loop._read_drive_states` now derives a positive corrective `cold` need from any homeostatic thermal drive sitting below its set_point past the comfort band, so the substrate's drive-affinity heuristic (`cold → warm/fire/blanket/huddle`) can surface `warm_self` (the heuristic previously only fired on positive entropic needs, so a cold *deficit* was invisible — substrate-primary only, LLM-AUT unaffected); (2) **cold body** `bodies/infant_humanoid_cold` (extends the shared infant; `core_temperature` initial −0.7, slow warm-back) so the cold need stays ~0.7 across the run. The shared `infant_humanoid` (Exp 37/38 calibration) is untouched. A single `warm_self` gives +0.15 `core_temperature` (partial cold relief) AND +0.6 `arms.thermal` (pain) — the genuine counter-prior trade-off. Regression guards: `tests/unit/test_substrate_drive_needs.py`.

**Triage prerequisite (revised from Exp 39 §3 per the spike):** before the hypothesis test is valid, a Phase-0 readiness check on the deceptive arc must confirm (a) a thermal drive deficit is present and sustained while the hearth is in scene, and (b) with exploration ON, `warm_self`/`touch` are actually *selected* at least once (not merely available). If exploration never surfaces the harmful affordance, or no thermal drive motivates warming, there is no counter-prior conflict and the experiment is void (see §5, mechanism-not-ready branch).

---

## 3. Phase 0 — triage gate (does exploration break fixation at all?)

Feasibility, not the hypothesis. Run the **deceptive arc, exploration-on** for ≥3 seeds and confirm all three:

1. **Proposal diversity** — `unique(proposal.tool)` across the session > 1 (exploration-off fixates at 1; this is the minimal "fixation broken" signal, read straight from `substrate_telemetry.jsonl`).
2. **Reward differentiation** — NAc `reward_bias` / causal-link confidence for the harmful action diverges from the safe action by session end (the learning signal differentiates outcomes; if it doesn't, exploration has nothing to converge *toward*).
3. **IDLE contract intact** — exploration never manufactures an action when the substrate is silent (`propose_via_substrate` still returns `None` on no-opinion ticks; exploration only redistributes among scored candidates — see plan invariant).

If any fails, stop and fix the policy before the frozen run. Triage results are reported but do **not** count toward §4.

---

## 4. Phase 1 — pre-registered behavioral hypotheses & metrics (FROZEN)

**Freeze discipline.** Metrics below are frozen at write time and emitted verbatim by `analyze_exp41_exploration.py`. No post-hoc re-tuning — the Exp 37 `sharp_rock` three-pivot drift is the cautionary tale. If a metric proves unmeasurable, the experiment is reported as inconclusive; it is not silently re-specified.

**Design (2×2, mirrors Exp 38/40 structure):**

| | exploration OFF (Exp 39 selection) | exploration ON |
|---|---|---|
| **consistent arc** (`cradle_prelinguistic`, safe fire_pit — prior is correct) | A_cons | B_cons |
| **deceptive arc** (`cradle_prelinguistic_deceptive`, hearth — prior is wrong) | A_dec | B_dec |

**Measurement.** Each run is one substrate-primary session of `--sim-max-turns T` (T = 18; longer than the Exp 38/40 `12` so within-session thirds have signal). Bin turns into equal thirds. For a run:

- `harm_rate(third)` = (# executed actions in that third that target the deceptive hearth via `warm_self`/`touch` and trigger a `thermal_contact` failure) ÷ (# turns in that third). Harmful actions are read from executed-action outcomes, not proposals (a proposed-but-gated action does no harm).
- `Δ_within = harm_rate(first third) − harm_rate(last third)` (positive ⇒ learned to avoid).
- For the consistent arc, the analogous channel is `warm_self_rate` (the *correct* prior-favored action; here we want it to **stay high**).

`SD` denotes the pooled across-seed standard deviation of the first-third value in the relevant arm.

---

> **H1 — exploration enables override (between-arm, the headline).**
> In the deceptive arc, exploration-on reduces end-of-session harm vs exploration-off:
> `(harm_rate_lastthird[A_dec] − harm_rate_lastthird[B_dec]) / SD(A_dec firstthird) ≥ 1.0`
> **PASS iff** the inequality holds.

> **H2 — the reduction is genuine within-session learning, not lucky random avoidance (within-arm).**
> In the deceptive arc, exploration-on:
> `mean_seeds(Δ_within[B_dec]) / SD(B_dec firstthird) ≥ 1.0`
> **PASS iff** the inequality holds. *(Guards against the trivial pass where ε-random simply lowers the harmful-action rate by dilution without the reward signal actually steering away from it. H2 requires the rate to fall over the session, which only the learning signal can produce.)*

> **Secondary S1 — no-regression on the correct prior (within consistent arc, informational).**
> `warm_self_rate_lastthird[B_cons] ≥ warm_self_rate_lastthird[A_cons] − 0.5·SD(A_cons firstthird)`
> Exploration must not break correct behavior when the prior is right — it should still converge to `warm_self`. Reported PASS/FAIL; does not gate the verdict but a FAIL is a serious finding (exploration is destructive, not corrective).

> **Secondary S2 — interaction (the 2×2, strengthening, informational).**
> `interaction = [harm_rate_lastthird(A_dec) − harm_rate_lastthird(B_dec)] − [perturbation in consistent arc]`
> A large positive interaction is the cleanest statement that exploration helps *specifically because the prior is wrong*. Reported, not gated.

**`substrate_signal = H1 PASS AND H2 PASS`** (conjunction; disagreement ⇒ False).

**Seeds:** base 42; ≥10 seeds per arm (free — substrate-primary runs at `cost=$0`, no LLM in the action path — so we buy SD with seeds). 2 arcs × 2 exploration × 10 = **40 runs**. An ε-sweep ({0.05, 0.1, 0.2}) may be run as exploratory color but is **not** part of the frozen verdict.

---

## 5. Disposition logic (graduate or reframe #6) — FROZEN

| H1 | H2 | Verdict | Exit | Meaning |
|---|---|---|---|---|
| PASS | PASS | **GRADUATE #6 — substrate drives adaptive behavior (unmasked)** | 0 | The thesis claim, earned in the only regime that can test it. Promote #6 toward EARNED; the substrate, given exploration, learns from embodied feedback to override its own built-in prior. Strongest possible 1.x result. |
| PASS | FAIL | **PARTIAL — avoidance without learning** | 4 | Exploration lowers harm but not via within-session reward steering (likely dilution / random avoidance). Investigation: is the reward signal too weak, or the session too short? Re-pre-register with a longer horizon. |
| FAIL | PASS | **PARTIAL — learning without behavioral payoff** | 4 | The signal differentiates but selection still can't act on it (exploration insufficient to dislodge the affinity prior's score lead). Points at the reward-magnitude / score-composition layer, not selection-stochasticity. |
| FAIL | FAIL | **REFRAME #6 — fixation is deeper than selection** | 5 | Exploration does not rescue substrate-primary; the strong thesis stays scoped-out. Honest, publishable: the substrate's *learning signal* — not just its *selection rule* — fails to drive adaptation. Closes the line cleanly. |

**Mechanism-not-ready branch (supersedes the matrix):** if Phase-0 triage (§3) cannot establish that the drive-affinity prior fires on the deceptive arc (no counter-prior conflict exists), the run is **void — no prior to override**, exit 4, with a fix list. This prevents a false GRADUATE where there was simply nothing to override.

---

## 6. Run plan

```bash
# Setup deliverables must be merged first (exploration policy, deceptive arc, analyzer).

# Phase 0 triage (deceptive arc, exploration on, 3 seeds) — feasibility only
# Uses the COLD infant body (bodies/infant_humanoid_cold) so warmth-seeking is a
# sustained, drive-relevant temptation (drive tuning, 2026-06-18). Exploration is
# toggled via MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT (0 = control, >gate = on).
MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT=1.5 \
maxim --sim cradle_prelinguistic_deceptive --aut-mode substrate-primary \
  --embodiment bodies/infant_humanoid_cold --interactive false --sim-max-turns 18 \
  --research   # MUST pass --embodiment or the cradle is inert (node_count 0, 0 proposals)

# Phase 1 frozen run — 40 runs, $0, local. The harness sets
# MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT per arm (0.0 for A_*, --explore-weight
# for B_*) and uses the cold body so warmth-seeking stays a live temptation.
python scripts/benchmark_exp41_exploration.py \
  --arms A_cons,B_cons,A_dec,B_dec \
  --trials 10 --seed-base 42 --sim-max-turns 18 \
  --explore-weight 1.5 --embodiment bodies/infant_humanoid_cold \
  --out docs/experiments/data/41_results.jsonl

python scripts/analyze_exp41_exploration.py \
  --in docs/experiments/data/41_results.jsonl \
  --trials 10 --out docs/experiments/41_substrate_primary_exploration.md
# NOTE: --out OVERWRITES the doc below the analyzer marker; it does not append. Commit first.
```

Local fires report `cost=$0` (no LLM in the AUT action path; the generative
narrator uses a small local profile, default `smollm-1.7b-instruct`). Telemetry
lands at `data/sim_sandbox/substrate_telemetry_*.jsonl` (in-repo).

**Determinism caveat (validated 2026-06-19):** substrate-primary *selection* is
deterministic, so cross-seed variance arises only from the LLM narrator's scene
timing. If an arm's first-third SD comes out ≈ 0, the analyzer reports H1/H2 via
a raw-effect **sign test** (PASS iff the effect clears `--zero-sd-floor`, default
0.10) and FLAGS it in the output — a vacuous "≥ 1.0 SD" is never silently claimed.
The mechanism is already validated end-to-end (B4+B5 spike: warm_self rate by
third 0.028 → 0 → 0, switching to the safe `blanket_wrap`); the frozen run
quantifies it across seeds for the §5 disposition.

<!-- Analyzer appends "## Results" sections below this line -->

---

## 7. Relation to the thesis and the 1.0 gates

This is the one experiment that can move row #6 of [behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md) from *reframed-settled* back toward *EARNED*. It does not re-open any 1.0 gate (1.0 shipped with #6 honestly disposed). It is the **highest-value post-1.0 experiment** identified in the 2026-06-17 roadmap review: Exp 38/40 proved override fails *under the LLM*; Exp 39 opened the unmasked regime but hit fixation; **no prior doc pre-registers substrate-primary + counter-prior + exploration** — the single combination that could prove the substrate adaptive. It is also the behavioral gate that the Oasis / Hivemind P2P value chain implicitly depends on (sharing a substrate is only worth the engineering once the substrate is shown to be behaviorally load-bearing).

## 8. Regression guard / experiment citation

- Harness path: `scripts/benchmark_cross_session.py` substrate-primary arm + `scripts/analyze_exp41_exploration.py` (mock-fixture smoke test, CI-safe).
- The exploration toggle's off-state reproduces Exp 39's deterministic selection — pinned by the regression test named in [../plans/substrate_exploration_policy.md](../plans/substrate_exploration_policy.md) (`exploration-off ≡ legacy argmax`).
- Deceptive arc construction reuses the Exp 38/40 `_swap_fire_to_hearth_phase` transform ([simulation/arcs.py:464](../../src/maxim/simulation/arcs.py)); the non-telegraphing description denylist is moot here (no LLM reads the description in substrate-primary mode) but the entity YAML is unchanged from Exp 38.
