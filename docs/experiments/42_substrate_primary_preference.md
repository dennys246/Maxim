# Exp 42 — Substrate-Primary Preference: does the unmasked substrate learn to prefer the safe source over the harmful one? (GRADUATE — frozen result folded; gating-OFF ablation graduates identically → B7 redundant, B8 load-bearing)

**Status:** FIRED 2026-06-23 — **GRADUATE #6**; MAINTAINED by the
[Exp 42b](42b_drive_pain_fold_revalidation.md) re-validation on 2026-07-29.
The pre-fire draft and frozen decision rules remain below as the historical record.
**Graduates:** [behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md) Tier 1 row **#6** (substrate-primary AUT mode).
**Builds on:** [41_substrate_primary_exploration.md](41_substrate_primary_exploration.md) (VOID — see §1). Reuses the shipped exploration policy ([../plans/substrate_exploration_policy.md](../plans/archive/substrate_exploration_policy.md)), drive-need derivation, scene-harm wiring (B4/B5), cold body, and the dedicated harness/analyzer pattern.

---

## 1. Why this redesign, and exactly what it claims

Exp 41 fired VOID: the substrate-primary embodied loop is **real** (try → pain → avoid) but the design couldn't adjudicate the strong thesis — the harm-rate metric floored (try-once dynamics) and exploration wasn't the differentiator (a drive-tempting harmful action got engaged in both arms). Exp 42 fixes both by construction.

**The claim — DISCRIMINATION, stated honestly.** With a SAFE and a HARMFUL warmth source both present (both relieve the cold drive; only one hurts), does the unmasked substrate (no LLM) learn from embodied pain to **prefer the safe source**? This is the *behavioral half of #6* — "substrate-learned outcomes drive adaptive action selection" — and it is what substrate-primary can cleanly test.

**Out of scope (named so it isn't conflated):** the harder claim that the substrate **overrides a prior that favors the harmful action** is the LLM-AUT line (Exp 38/40), where it was tested-and-dominated and stays pulled. Here the drive-affinity prior favors *both* warmth sources roughly equally, so Exp 42 tests discrimination among prior-attractive options, not override of a counter-prior. A GRADUATE here earns "substrate drives adaptive behavior"; it does not re-open the override question.

**The reframed, bias-proof design.** Manipulated factor = **which source is harmful**, counterbalanced across two arcs. Exploration is held ON in all arms (Exp 41 settled it as the enabling condition, not a differentiator). The single primary test — the agent prefers the SAFE source **in both arms** — is sufficient and robust: a fixed identity / name / position bias *cannot* satisfy it, because the safe identity flips between arms (a bias toward identity-α makes "safe" preferred in only one arm). So "safe preferred in both arms" simultaneously demonstrates adaptive preference AND rules out non-learning explanations.

---

## 2. Setup deliverables (before any run)

1. **Two generic warmth-source identities, each in a safe and a harmful variant** (4 component YAMLs; identities `alpha`/`beta`, suffixes `_safe`/`_harm`): `warmth_alpha_safe`, `warmth_alpha_harm`, `warmth_beta_safe`, `warmth_beta_harm`. Symmetric names (both contain "warm", neither carries an extra affinity keyword like "fire") so the identities have no baseline-affinity asymmetry. Each `warm_self` (and `touch`) carries:
   - safe: `{cold: -0.3, arms.thermal: +0.05}` — relieves the cold drive, no comfort-band breach (no pain → positive/neutral causal link).
   - harmful: `{cold: -0.3, arms.thermal: +0.60}` — same cold relief, but `arms.thermal` breaches the 0.5 band → pain → negative causal link.
   **Why two variants per identity (not one safe + one harmful entity):** the counterbalance needs each *identity* to appear as both safe and harmful across arms, and two components can't share an `entity.name` (the all-components-into-one-registry test collides — the Exp 41 `infant_humanoid_cold` lesson). The analyzer reads safety from the `_safe`/`_harm` suffix and identity from the `alpha`/`beta` prefix; the substrate never reads names semantically (no LLM; the affinity heuristic only substring-matches "warm"), so the suffix doesn't leak.
2. **Two counterbalanced arcs** (substrate-primary, prelinguistic), both placing **both** identities in scene from phase 0:
   - `cradle_pref_a`: `warmth_alpha_harm` + `warmth_beta_safe` (predict: prefer β).
   - `cradle_pref_b`: `warmth_alpha_safe` + `warmth_beta_harm` (predict: prefer α).
   Thermal-focused (warming is the dominant motivated behavior — see deliverable 3); other cradle entities (food) may stay but are non-competing because hunger is rate-limited below the selection gate.
3. **Entropic-cold body** `bodies/infant_humanoid_chilled` (extends `infant_humanoid`; Exp 41's `infant_humanoid_cold` left untouched). Two changes vs the base:
   - **Add an ENTROPIC `cold` drive** (rises over time like hunger, `drift_direction: up`, starts ~0.6) that the warmth sources' `cold: -0.3` relieves. Entropic (not homeostatic) is load-bearing: a homeostatic cold drifts *toward comfort* and **satiates** after a few warms → warming stops → the exploitation metric floors (the Exp 41 trap through a side door). An entropic cold **regenerates** → warming *recurs* → many exploitation warm-choices. The affinity table already maps `cold → (warm, fire, blanket, huddle)`, so the raw entropic `cold` value drives warm-tool selection directly (no `_read_drive_states` derivation needed; that derivation is for the homeostatic case).
   - **Weaken hunger/thirst** (keep them — per design discussion — but rate-limit): drop their `drift_rate` enough that their value stays **< 0.5** across the run, so they never enter the drive-affinity selection competition (the gate is `drive_value > 0.5`); also lower `deprivation_pain` for a cleaner pain cascade. The depletion *rate*, not the valence, is what removes the competition in substrate-primary selection. Phase 0 (§3) gates that warming actually dominates.
4. **Analyzer** `scripts/analyze_exp42_preference.py` — exploitation-phase `safe_pref` per arm + the corroborating identity-flip and per-source link-sign signals; frozen verdict matrix (§5); SD≈0 sign-test (substrate-primary is ~deterministic); mock-fixture CI smoke test. Mirrors the Exp 41 analyzer.
5. **Harness** `scripts/benchmark_exp42_preference.py` — arms = the two counterbalanced arcs × N seeds, exploration ON (weight 1.5), cold body, `--aut-mode substrate-primary`; per-run warm-action extraction per source from `actions.jsonl`; `--mock`; `--resume`. Mirrors the Exp 41 harness.

---

## 2a. Building-block validation (pre-triage spike, single arm A, smollm)

Before the formal triage, a sequence of single-arm spikes drove the setup parameters (recorded so the freeze isn't a guess):

- **Arc resolution.** `select_arc_for_goal` lowercases the goal before the exact dict lookup, so arc keys MUST be lowercase (`cradle_pref_a`/`cradle_pref_b`) — a capitalized key silently falls through to the `"cradle"` substring keyword and resolves the wrong arc.
- **No sense/read fixation.** First spikes saw the agent fixate on `read_*_heat_output` (62×) instead of warming: the entity name `warmth_*` made its sense/read tools match the `cold→"warm"` affinity, and as always-successful tools they snowballed. **Fix:** the warmth entities carry **no sensors** — only the abstract `glow` modulator's affordances. `warm_self`/`touch` still get the affinity boost (the affordance name carries "warm"); no sense/read tools are generated.
- **Per-source attribution is RELATIVE, not absolute.** `evaluate_failures()` applies drift *before* the breach check using wall-clock dt (body.py:145-154), so no `arms.thermal` drift rate gives a clean transient breach: too fast erases the harmful breach before attribution; too slow lets it linger and mis-attribute to later *safe* actions. The discrimination is therefore relative — the harmful source takes a **fresh** breach on every warm (more negative weight) while the safe source only inherits lingering noise — and the verdict is the agent's **behavioral** preference, not the absolute sign of either link.
- **Validated outcome (spike 4, cold drift 0.08, 24 turns):** 14 recurring warm/contact actions; exploitation phase **7/7 on the safe source (`safe_pref` = 1.00)**; per-source nets harm **−0.25** vs safe **+0.14** (clean separation); weak hunger never competed. K=7 exploit samples at 24 turns → real runs use **~40 turns** to clear K≥10.

## 3. Phase 0 — triage gate (is the metric measurable?) — Exp-41-mandated

Triage both arcs (≥3 seeds, exploration ON). Freeze §4 only if ALL hold; else re-tune (do NOT freeze a floored metric):

1. **Enough exploitation warm-choices.** After explore-first has tried each warmth source once, the run yields ≥ **K = 10** further warm/contact actions (`warm_self` + `touch`, the exploitation phase) — so `safe_pref` is a stable ratio, not a 1–2-action artifact. This is the specific Exp 41 failure mode; gate hard. The validation spike hit K=7 at 24 turns, so the run uses ~40 turns; if warming satiates (too few exploitation choices), speed cold re-drift (raise the entropic `drift_rate`) and re-triage.
2. **Both sources tried early.** Explore-first surfaces both `warm_self` affordances in the first third — otherwise there is nothing to prefer between.
3. **Per-source learning differentiates (relative).** The harmful source ends MORE NEGATIVE than the safe source in NAc (readable from `substrate_telemetry.jsonl` `causal_links` per `tool:warmth_*_warm_self` / `tool:warmth_*_touch`) — validated as harm −0.25 vs safe +0.14. If the two end indistinguishable in the substrate → **VOID — no learnable contrast**.

## 3a. Triage outcome + the mechanism chain it forced (n=1 per arm, smollm)

The triage **PASSED** — but only after three substrate-primary mechanism gaps surfaced and were fixed. The single-arm spike (§2a) looked clean; the *counterbalanced both-arms* triage exposed each gap in turn (recorded so the freeze and the writeup are honest about what is load-bearing):

1. **Triage VOID #1 — meta-tool fixation (introspection).** The agent spent ~95% of actions on cognitive introspection tools (`temporal_patterns`, `system_stats`, …) instead of warming — they always succeed, so their causal confidence snowballs and wins `recommend_action`'s argmax. **Fix B6:** `propose_via_substrate` filters `INTROSPECTION_TOOL_NAMES` out of the substrate-primary candidate set (substrate-primary only; LLM-AUT untouched).
2. **Triage VOID #2 — fixation relocated to `sense_presence`.** With introspection gone, the next always-succeeding zero-stakes tool (sensing) took over. The drive-affinity nudge (~0.5) can't beat a snowballed causal score (~2.0). **Fix B7 (drive-gating / motivated attention):** when a drive exceeds its gate threshold, `recommend_action` restricts the exploitation-phase candidate set to drive-relevant tools — a hard attentional gate. **Opt-in, default OFF** (`sim.drive_gate_enabled` / `MAXIM_SIM_DRIVE_GATE_ENABLED`), so global learned-link-primacy semantics are unchanged; the experiment enables it.
3. **Triage REFRAME — discrimination collapsed under dense warming.** Gating produced dense warming (100/85 contacts) but the safe source ended as negative as the harmful one (arm B: both `neg≈0.96`). Cause: bystander mis-attribution — once the harmful source's `arms.thermal` breach lingers, *every* tool executing during it (incl. the gentle safe warm) was blamed. **Fix B8 (delta-attribution):** a drive-spec failure is attributed to an affordance only if its own delta is intrinsically harmful to that sensor (would breach a healthy sensor). Global embodiment bug fix; the SEM pain cascade + Exp 37/38 are preserved (the *causing* tool still reports; only bystanders are spared).

**Result (B6+B7+B8, shipped config path, n=1/arm, 40 turns):** GRADUATE — `safe_pref` **0.990 / 0.986** (both arms ≥ 0.66), identity flip **+0.975**, per-source nets harm −0.25/−0.32 vs safe **+0.99** (distinct), K = 96/141 exploitation contacts. The counterbalance is textbook: `id_pref_a` swings 0.010 (α harmful → avoided) → 0.986 (α safe → preferred), so the preference tracks *safety*, not identity.

**Pre-registered prediction (and how the frozen run REFUTED it).** At triage time the working hypothesis was that drive-gating was *load-bearing* — an enabling "motivation" mechanism (analogous to exploration in Exp 41) without which the agent would re-fixate on always-succeeding zero-stakes tools, so the graduated claim would be the narrower **"discrimination within motivated attention."** The frozen run **mandated a gating-OFF ablation arm** to test exactly this. **The ablation overturned the prediction:** with gating fully OFF, discrimination is statistically identical (see §Results — Arm A `safe_pref` 0.984, Arm B 0.965, same C1 flip / C2 signs). Gating changes only the *volume* of warming (treatment Arm B spikes to 106 contacts on some seeds vs the ablation's tight ~56–64; the toggle demonstrably fired), **not** the discrimination. So the honest claim is the stronger, unqualified one: the substrate discriminates safe from harmful **from its own clean credit assignment, with no motivated-attention crutch**. What carries the result is **B8 (delta-attribution: clean per-source learning) + the pre-existing drive-affinity heuristic**, not B7.

**Consequence: B7 drive-gating did NOT earn its behavioral weight.** Per dormancy-over-deletion it is marked `Dormant` (kept wired, default-OFF; no new work builds on it) rather than removed; resurrection requires a future experiment that earns it. The load-bearing mechanism is **B8** — its two-channel scope limitation (the side_effects channel is delta-attributed; the parallel `_publish_drive_pain`→PainBus channel is not) is therefore the priority follow-up, tracked in [transition_based_drive_pain.md](../plans/deferred/transition_based_drive_pain.md).

**Substrate-isolation evidence.** The substrate's contribution (vs residual ordering/affinity) rests on **C1 (counterbalance identity-flip: +0.96) + C2 (per-source link signs: harm net < safe net, both arms)** — C1 proves the preference tracks the *swapped* safety contingency (not a fixed bias), C2 proves the learned substrate carries the safe/harm distinction. Both hold in both treatment and ablation.

---

## 4. Phase 1 — pre-registered metric (FROZEN on authorization after §3)

**Design:** counterbalanced A/B, substrate-primary, exploration ON in both, cold body, `--sim-max-turns ~40` (**CORRECTION 2026-07-28: the runs were 30 turns, not 40.** `--sim-max-turns` is a ceiling only; the generative loop ends on `narrator.is_done`, driven by the arc's per-phase `turns_max` = 6+12+12 = 30, so the flag never bound. The frozen result IS a 30-turn result — the K≥10 validity gate was cleared regardless. Replications must keep 30 for comparability; see [42b](42b_drive_pain_fold_revalidation.md).) (validation spike: K=7 at 24 turns → ~40 for K≥10; T finalized at triage). ≥ 10 seeds/arm; `cost=$0`.

**Exploitation phase (the operationalization).** Per run, from `actions.jsonl`, find the tick after which **both** warmth sources have each been selected via a contact affordance (`warm_self` or `touch`) at least once (the explore-first discovery point). All contact actions (`warm_self` + `touch` — identical `self_effect`, aggregated for sample density per the spike) on either source *after* that point are the **exploitation phase** — choices driven by learned scores, not the forced first trial.

- `safe_pref(arm)` = (exploitation warm actions on the SAFE source) / (all exploitation warm actions), per arm, pooled across seeds.
- `id_pref_a(arm)` = exploitation warm share on source-**a** specifically — the identity channel for the corroborating flip.

`SD` = pooled across-seed sample stdev (SD≈0 → flagged sign-test, per the Exp 41 analyzer).

---

> **H1 (PRIMARY, sufficient, bias-proof) — the agent prefers the SAFE source in BOTH arms.**
> `safe_pref(cradle_pref_a) ≥ 0.66` **AND** `safe_pref(cradle_pref_b) ≥ 0.66`.
> **PASS iff** both hold. A fixed identity/name/position bias cannot satisfy this (the safe identity flips between arms), so passing both arms *is* the demonstration that embodied feedback — not a static bias — drives the preference. (Threshold illustrative until triage; freezes then.)

> **Corroborating C1 — identity-preference flips with the harm assignment.** `id_pref_a(cradle_pref_b) − id_pref_a(cradle_pref_a) ≥ +0.33` (source-a is preferred when it's safe, avoided when harmful). Reported; quantifies the flip H1 implies. Not a separate gate.

> **Corroborating C2 — per-source learning signs (relative).** Across runs, the harmful-source contact links (`tool:*_warm_self` / `tool:*_touch`) end MORE NEGATIVE than the safe-source links in NAc (validated harm −0.25 vs safe +0.14). Relative, not absolute, per §2a (the lingering-breach mechanic adds symmetric noise). Reported; ties the behavior to the substrate signal.

**`substrate_signal = H1`** (both arms ≥ threshold). C1/C2 strengthen but do not gate.

**Freeze discipline:** the 0.66 threshold and K=10 freeze at authorization after triage and are NOT re-tuned post-hoc (the Exp 41 / `sharp_rock` cautionary tale). If triage can't clear §3, the design is revised *before* freezing, not after firing.

---

## 5. Disposition logic (graduate or reframe #6) — frozen with §4

| Outcome | Verdict | Exit | Meaning |
|---|---|---|---|
| `safe_pref ≥ 0.66` in **both** arms | **GRADUATE #6 — substrate drives adaptive, feedback-tracked behavior** | 0 | The behavioral half of #6, earned: the unmasked substrate learns from embodied pain to prefer the safe source, and the preference tracks the feedback across the counterbalance (not a fixed bias). Promote #6 toward EARNED (substrate-primary discrimination; override stays the LLM-AUT line). |
| `safe_pref ≥ 0.66` in **one** arm only | **PARTIAL — asymmetric preference** | 4 | Safe is preferred only when it aligns with some residual bias (name/position/exploration order). Investigate the tiebreak; the claim isn't clean. |
| `safe_pref < 0.66` in **both** arms | **REFRAME #6 — no adaptive preference even with a safe escape** | 5 | Given a safe alternative, the substrate still doesn't converge to it. The strong claim stays scoped-out — honest, publishable (learning is internal-real per Exp 41 but doesn't translate to sustained preference). |
| triage fails §3 | **VOID — metric not measurable / no learnable contrast** | 4 | Flooring (< K exploitation choices) or indistinguishable sources; fix setup, do not freeze. |

---

## 6. Run plan (provisional — finalize at freeze)

The harness sets the substrate-primary knobs per sub-sim (exploration ON +
drive-gating ON in the treatment arms); `MAXIM_SIM_DRIVE_GATE_ENABLED` is
default-OFF globally, so the **gating-OFF ablation arm just omits it**.

```bash
# Phase 0 triage — PASSED at n=1/arm (GRADUATE; see §3a). The harness enables
# exploration + drive-gating internally; a manual single-arm smoke is:
MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT=1.5 MAXIM_SIM_DRIVE_GATE_ENABLED=1 \
maxim --sim cradle_pref_a --aut-mode substrate-primary \
  --embodiment bodies/infant_humanoid_chilled --interactive false --sim-max-turns 40 --research

# Phase 1 frozen run — counterbalanced, $0, local (≥10 seeds/arm).
python scripts/benchmark_exp42_preference.py \
  --arms cradle_pref_a,cradle_pref_b --trials 10 --seed-base 42 --sim-max-turns 40 \
  --explore-weight 1.5 --embodiment bodies/infant_humanoid_chilled \
  --out docs/experiments/data/42_results.jsonl
python scripts/analyze_exp42_preference.py --in docs/experiments/data/42_results.jsonl --trials 10 \
  --out docs/experiments/42_substrate_primary_preference.md

# Gating-OFF ablation arm — same arcs, gating disabled via the env override.
# RESULT: also GRADUATE (identical discrimination) → gating is NOT load-bearing;
# B8 + drive-affinity carry the preference. See §Results.
MAXIM_SIM_DRIVE_GATE_ENABLED=0 python scripts/benchmark_exp42_preference.py \
  --arms cradle_pref_a,cradle_pref_b --trials 10 --seed-base 42 --sim-max-turns 40 \
  --explore-weight 1.5 --embodiment bodies/infant_humanoid_chilled \
  --out docs/experiments/data/42_results_gateoff.jsonl
```

The harness defaults drive-gating ON for the treatment arms but **respects a
parent-env override** (`env.get("MAXIM_SIM_DRIVE_GATE_ENABLED", "1")`), so the
ablation command above works as-is — no code change needed.

## 7. Relation to the thesis and #6

Cleanest available test of "the unmasked substrate drives adaptive behavior": (a) removes the LLM-prior confound (substrate-primary), (b) avoids Exp 41's flooring (exploitation-phase preference, not harm-rate), (c) controls for fixed bias by construction (counterbalance + the both-arms test). **Outcome: GRADUATE** (10 seeds/arm, both arms; see §Results) — and the gating-OFF ablation graduating *identically* makes the evidence stronger than pre-registered: the substrate discriminates safe from harmful from its **own clean credit assignment (B8) + drive-affinity**, with no motivated-attention crutch (B7 turned out redundant). This is the strongest 1.x evidence the substrate is behaviorally load-bearing — the prerequisite the Oasis / Hivemind value chain assumes. The claim is **discrimination** in the substrate-primary regime; the LLM-AUT override result (Exp 38/40) is untouched — substrate-primary *discriminates* where LLM-AUT *does not override* a wrong prior, and both are honest, complementary halves of #6.

## 8. Regression guard / deliverable citations (to populate as built)

- Harness + analyzer: `scripts/benchmark_exp42_preference.py` + `scripts/analyze_exp42_preference.py`, pinned by [tests/behavioral/test_exp42_pipeline.py](../../tests/behavioral/test_exp42_pipeline.py) (18 tests: verdict-matrix corners incl. K-validity VOID + threshold boundary, exploitation-phase discovery extraction, C1 identity-flip, C2 net reduction, mock pipeline → GRADUATE, resume idempotency — all CI-safe, no subprocess/LLM).
- Counterbalanced arcs + generic warmth entities: `simulation/arcs.py` (`cradle_pref_a`/`cradle_pref_b`) + `_data/components/items/warmth_{alpha,beta}_{safe,harm}.yaml` + entropic-cold body `_data/components/bodies/infant_humanoid_chilled.yaml`.
- Reuses shipped + guarded mechanism: exploration policy ([tests/unit/test_substrate_exploration.py](../../tests/unit/test_substrate_exploration.py)), drive-need derivation ([tests/unit/test_substrate_drive_needs.py](../../tests/unit/test_substrate_drive_needs.py)), B4/B5 scene-harm ([tests/unit/test_substrate_primary_scene_harm.py](../../tests/unit/test_substrate_primary_scene_harm.py)).

<!-- Analyzer appends "## Results" sections below this line -->

## Results

Frozen run: 10 seeds/arm × 2 counterbalanced arms, substrate-primary, smollm narrator, 40 turns, `cost=$0`, git `0d6ca70f`. Treatment = exploration + drive-gating ON; ablation = drive-gating OFF (`MAXIM_SIM_DRIVE_GATE_ENABLED=0`), all else equal.

**Headline: GRADUATE #6, and the gating-OFF ablation graduates *identically* → drive-gating (B7) is NOT load-bearing.** The pre-registered "discrimination within motivated attention" caveat is refuted: the substrate discriminates safe from harmful from its own clean credit assignment (B8 delta-attribution) + the pre-existing drive-affinity heuristic. Gating changed only warming *volume* (treatment Arm B spikes to 106 contacts on some seeds; the ablation sits tight at ~56–64 — the toggle demonstrably fired), not the discrimination. → B7 marked `Dormant` (did not earn behavioral weight); **B8 is the mechanism that carries #6**.

### Treatment (gating ON) — GRADUATE (exit 0)

- H1 (both arms ≥ 0.66): **True** — A `safe_pref` 0.984, B 0.975
- C1 identity-flip +0.959 PASS · C2 (harm net < safe net) PASS · 10/10 valid both arms, 0 floored

| arm | safe id | safe_pref | SD | id_pref_a | harm net | safe net |
|---|---|---|---|---|---|---|
| cradle_pref_a | β | 0.984 | 0.002 | 0.016 | −0.250 | 0.990 |
| cradle_pref_b | α | 0.975 | 0.015 | 0.975 | −0.307 | 0.990 |

### Gating-OFF ablation — GRADUATE (exit 0)

- H1 (both arms ≥ 0.66): **True** — A `safe_pref` 0.984, B 0.965
- C1 identity-flip +0.949 PASS · C2 PASS · 10/10 valid both arms, 0 floored

| arm | safe id | safe_pref | SD | id_pref_a | harm net | safe net |
|---|---|---|---|---|---|---|
| cradle_pref_a | β | 0.984 | 0.002 | 0.016 | −0.250 | 0.990 |
| cradle_pref_b | α | 0.965 | 0.001 | 0.965 | −0.321 | 0.990 |

The counterbalance flips cleanly in both runs: `id_pref_a` ≈ 0.016 when α is harmful (avoided) → ≈ 0.97 when α is safe (preferred), so the preference tracks the *swapped safety contingency*, not source identity. **Scope:** substrate-primary is near-deterministic (LLM removed), so this is a mechanism-level result — the substrate, given correct credit assignment, learns and acts on a counterbalanced safety contingency. Override / LLM-prior dominance remains the separate Exp 38/40 line.
