# B4 (Exp 37 / T1-2 with floors): confounding-lens design review

Reviewed: b4_exp37_floors_prereg.md DRAFT v1 (2026-10-18)

Lens: does each gate isolate the cause it is credited with, and could a positive or a null arise for some other
reason? I read the prereg, `docs/plans/baseline_floors.md`, `37_cross_session_graduation.md`, L6/L8, the T1-2 ledger
row, `scripts/benchmark_cross_session.py`, `scripts/analyze_exp37.py`, every committed `37_*.jsonl` / `38_*.jsonl`,
`tools/narrative.py::ExamineTool`, `simulation/sim_types.py::build_resume_prompt` and
`simulation/orchestrator.py::_restore_aut_from_session`. The scratch computations are not committed.

## What I verified (with numbers)

- **Per-arm SDs, `fire_pit` primary, Qwen32B anchor** (stdev, n=5 each): June 2026-06-08: A 0.266, B 0.298, C 0.333.
  1.1 heartbeat 2026-08-21: A 0.328, B 0.279, C 0.217. Old-hash August: A 0.119. Mean of the seven is 0.263 and the
  maximum 0.333. The SD of the paired B − C differences is 0.298 (June) and 0.384 (heartbeat). The prereg's σ = 0.28
  per arm, which implies σ_diff ≈ 0.40 unpaired, is consistent with that and is not tuned. Other fires: Qwen14B A
  0.283 / B 0.149 / C 0.261; R1 A 0.145 / B 0.354 / C 0.399; Mistral A 0.000 (ceiling) / C 0.548.
- **Pairing buys nothing in the record.** corr(B, C) by trial is −0.18 (heartbeat) and 0.56 (June), and across all ten
  `fire_pit` fires it ranges from −0.55 to 0.91. corr(A, B) is −0.75 (heartbeat) and 0.97 (June). At n = 5 none of
  these differs from 0.
- **Ties.** 13 of the 50 B − C pairs across the ten committed `fire_pit` fires are exactly 0 (26 %), because the metric
  is a ratio of 2–6 engagement actions and so takes lattice values in {0, .2, .25, .33, .5, .67, .75, .8, 1}.
  Wilcoxon's `zero_method="wilcox"` drops those pairs.
- **The seed is not honoured on the anchor topology.** `router.py` stamps `seed_honoured=isinstance(backend,
  _LlamaCppBackend)`, and `maxim_peer_backend.py` contains no `seed` at all. A harness on a different box from the
  server (the prereg's Model control) routes through the peer backend. `--seed` (`cli_parser.py`) seeds Python,
  numpy and torch only.
- **`_llm_unavailable` in the anchor-era fire.** The 1.1 heartbeat has 4 of 60 rows with `_llm_unavailable ≥ 1`
  (6.7 %): one each in B, C, B-wire-a-off and B-nac-bias-off, none in A. Every other valid fire has 0, except Exp 38
  DeepSeek with 1 (in B).
- **Engagement = 0 scores 0.0, not None.** In `compute_metrics`, `positive_approach_engagement_fraction = approach /
  engagement if engagement > 0 else 0.0`, and `warm_self_engagement_fraction` does the same.
- **`deceptive_fire`'s frozen direction is DECREASE.** `analyze_exp37.py::compute_counter_prior` passes only if
  `interaction / pooled_A_sd ≤ −1`, that is, if B reduces warm_self specifically on the deceptive hearth. In the committed
  Exp 38 data, B's warm_self rises over A in three of four models (DeepSeek 0.70 → 0.95, R1 0.47 → 0.60, Sonnet
  0.27 → 0.47; GPT-4o 0.72 → 0.69). No Qwen32B `deceptive_fire` fire exists, so the anchor's headroom and SD there are
  unknown.
- **Resume restores memory only.** `_restore_aut_from_session` loads hippocampus, NAc, EC and ATL and nothing else; no
  body or world sensor state carries over, so a carried thermal state is not a confound. NAc loads with
  `apply_decay=False`, so the save→resume gap cannot decay biases. `hippocampus_consolidation.py` still decays
  consolidation pressure by wall-clock time (`_PRESSURE_DECAY_RATE`).
- **The narrator still receives the resume prompt in code.** `orchestrator.py` builds `_build_resume_prompt(...)`
  and passes it to `orchestrator_source.inject_cli`, and that prompt names the previous goal (C's says "puzzle door
  and button"), the summary and the tool usage. The ledger says the cradle path never consumes it (#1226 parked), so
  the prereg's structural probe is the right instrument.
- **FearGatedExecutor is always on in the historical fires.** `orchestrator.py` wraps the AUT executor
  unconditionally ("AUT FearGatedExecutor active"), and no env or config toggle exists in `src/`.
- **The C prior is shared across scenarios** (`run_benchmark`: `arm_c_prior` is built once per trial), so the second
  scenario's C resumes from a prior that is several sims older than the prior B resumes from. B resumes from the same
  trial's A, which ran immediately before it.
- **Power, simulated** (beta-binomial per-trial propensity, 2–6 engagements, arm SD ≈ 0.27–0.32, n = 25, one-sided
  Wilcoxon). True C 0.6 / B 0.8 gives power 0.76 at α 0.05 and 0.44 at α 0.05/6. C 0.5 / B 0.7 gives 0.71 and
  0.37. C 0.6 / B 0.7 gives 0.29. The prereg's formula reproduces (n = 24.2). At α/6, the first Holm step, the same
  formula needs n = 41 (σ 0.28) or 57 (σ 0.33).

## DO-NOT-BUILD

### DNB-1. `deceptive_fire`'s gate direction and headroom rule are written for an increase metric, and its metric decreases

- **Claim attacked:** "G1 B − C > 0 on each scored scenario"; "a scenario whose A or C mean is ≥ 0.85 is NO_HEADROOM";
  `deceptive_fire` is a co-primary "with its own metric".
- **Alternative explanation:** on the deceptive hearth `warm_self` hurts, so learning from the consequence predicts that
  warm_self falls (Exp 38's frozen logic in `compute_counter_prior`). Read as written, G1 credits a RISE in warm_self
  as memory, and that rise is what perseveration produces. The committed Exp 38 data show exactly that rise: B > A in
  three of four models (DeepSeek +0.25, R1 +0.13, Sonnet +0.21). The headroom rule is inverted as well. For a
  decreasing metric the room is at the top, so "C ≥ 0.85 → NO_HEADROOM" excludes the case with the most room to fall
  and scores the case where C ≈ 0 and nothing can fall. G2 and G3 ("B > floor") carry the same sign error.
- **Fix (prereg text):** "Each scored scenario declares its predicted direction: `fire_pit` increase, `deceptive_fire`
  decrease (Exp 38, `analyze_exp37.py::compute_counter_prior`). Every gate is signed by that direction: G1 is
  `dir·(B − C) > 0`, G2 is `dir·(B − bare_llm) > 0`, G3 is `dir·(B − llm_notes) > 0`. Headroom is read on the
  predicted side: NO_HEADROOM when the comparator mean is within δ of the bound the effect moves toward (≥ 1 − δ for
  increase, ≤ δ for decrease)."

### DNB-2. On `fire_pit`, B − C > 0 is also predicted by perseveration, not only by a learned value, and #845 recall carries the perseveration

- **Claim attacked:** "G1 isolation: what the agent carried, not being resumed." T1-2's mechanism column reads "Wire-A
  annotation + Wire-1 annotation + NAc reward bias → action selection".
- **Alternative explanation:** in `fire_pit`, warm_self is at once the prior-favoured, the rewarded and the
  already-done action. Any memory that "I warmed myself at the fire" raises warm_self in B and not in C, because C's
  peaceful prior holds no such memory: hippocampus memories surfaced on resume, `ExamineTool`'s "You recall: …"
  lines (#845, which bypass all three wires; the ledger says "B − C would carry it"), and affordance familiarity
  (B starts knowing that `fire_pit_warm_self` exists, while C must discover it). So B − C > 0 confirms that "something
  about the fire was remembered", not that "a reward bias steered selection". With the three ablations dropped and
  #845 left at "report only", nothing in the arm set can separate the row's mechanism from a text-recall or
  familiarity channel. The Exp 38 data in DNB-1 show that, in this world family, perseveration is the dominant
  carried effect.
- **Fix (prereg text; pick one, owner decision):**
  - (a, preferred, no new arm) "G1 is the cross-world interaction, with C in place of A:
    `I = (B − C)_fire_pit − (B − C)_deceptive_fire > 0`. Learning from consequence predicts I > 0; perseveration and
    familiarity predict I ≈ 0. Both scenarios are then required, so Open decision 1's `fire_pit`-only option is
    removed."
  - (b) "Add one arm, `B_recall_off`: B with #845 recall ablated **and** the three wires off (the T1-2 channel set
    off together). G1b is B − B_recall_off in the predicted direction." Note #889: `MAXIM_NAC_REWARD_BIAS_DISABLED`
    was once void in practice, so the arm needs a structural probe that the bias is off.
  - (c) If neither is adopted, the verdict text must narrow: "FLOORS_BEATEN shows that a Maxim-carried memory, by any
    channel including examine recall, moved behaviour. It does not show Wire-A, Wire-1 or NAc reward bias, and the
    ledger's mechanism column may not be cited for it."

### DNB-3. G2 and G3 compare LEVELS, so they can pass or fail on single-session scaffolding and information with zero memory effect

- **Claim attacked:** the title's "beyond a bare LLM and an LLM with notes"; FLOORS_BEATEN → T1-2 MAINTAINED/EARNED.
- **Alternative explanation:** B > `llm_notes` holds whenever A − `bare_llm` (scaffolding plus prompt: `pfc_preamble`,
  `acting_coach`'s "Act to bring it back" drive modulation, `embodiment/body.py::format_body_state_for_prompt` DRIVE
  annotations, Wire-1 bands) exceeds the gap between what notes add and what Maxim's memory adds. A worked case: Maxim's
  memory adds 0.15 (G1 passes), single-session scaffolding adds 0.30, and notes add 0.30. Then B = A + 0.15 and
  `llm_notes` = A, so G3 passes by 0.15 while notes bought twice what the substrate bought. The reverse holds too: a
  floor arm whose prompt over-performs in one session fails G3 for reasons unrelated to memory. A single-session
  `bare_llm` is a fair BF2 floor for the full agent's level, but as built it is not a test of the cross-session part.
- **Fix (prereg text; pick one):**
  - (a) "Add `llm_notes_C`: `bare_llm` resumed with notes from a peaceful-goal bare session (prior shared across
    scenarios, as C's is). G3 becomes `dir·[(B − C) − (llm_notes − llm_notes_C)] > 0`: Maxim's isolated carry beats
    notes' isolated carry. The level comparison B vs `llm_notes` is reported." The cost is about +1.5 sims per
    trial-scenario.
  - (b) Keep the level gates as BF2 literally states them, but the verdict text says "FLOORS_BEATEN: the resumed Maxim
    agent out-scores the resumed notes agent. Whether its *carry* exceeds notes' carry is not tested." Also report the
    difference-in-differences `(B − A) − (llm_notes − bare_llm)`, labelled with L8's anti-correlation caveat. Under
    (b) the ledger prose may not say "beats just give it notes" for the cross-session part.

## SHOULD-FIX

### SF-1. Holm over G1–G3 × scenarios couples the binding L8 gate to the floors and under-powers the conjunction

- **Claim attacked:** "G1–G3 × the scored scenarios form one family, Holm-corrected at 0.05"; n = 25 for 80 % power.
- **Alternative explanation:** Holm's step-down compares G1's p against α/(k − rank + 1). When the floors' p-values
  are large they rank after G1, and G1 is held to α/6 or α/5. A weak `bare_llm` therefore makes the binding isolation
  gate harder, and NOT_ISOLATED becomes more likely because of a floor result. At α/6 the simulated power for a true
  0.2 effect at n = 25 is 0.37–0.44, not 0.8. Moreover FLOORS_BEATEN is an all-must-pass claim. An
  intersection-union test needs no multiplicity correction, but its power is roughly the product of the per-gate
  powers (0.8⁶ ≈ 0.26 if independent). So a real effect most likely yields ISOLATED_FLOOR_MATCHED or NOT_ISOLATED.
- **Fix:** "Fixed-sequence gatekeeping: G1 is tested first at α = 0.05 (Holm across scenarios only if more than one G1
  test is gating; under DNB-2(a) there is one). Only if G1 passes are G2 and G3 tested, each at α = 0.05, as an
  intersection-union test for FLOORS_BEATEN. n is set for per-gate power 0.9: `2(1.645 + 1.282)²σ²/δ²` = 34 at σ 0.28
  and δ 0.20. Report the conjunction power."

### SF-2. "Pairing by trial seed" pairs nothing on the anchor topology

- **Claim attacked:** "the same trial seed across arms; one-sided Wilcoxon signed-rank on the paired differences …
  Pairing by seed can only help."
- **Alternative explanation:** the peer backend sends no seed and `seed_honoured` is False for it, so LLM sampling is
  unseeded. Even an honoured seed diverges after the first differing prompt. B and C share only the trial index and
  the Python RNG. The committed corr(B, C) ranges from −0.55 to 0.91 (heartbeat −0.18). A negative correlation makes a
  paired test worse than an unpaired one, so "can only help" is false. With 26 % exact-zero pairs, signed-rank drops a
  quarter of the trials, and it tests the pseudo-median, not the mean difference the n formula assumes. Two tests
  ("paired, with the unpaired beside it") with neither declared as governing is a fork.
- **Fix:** "The governing test for every gate is a one-sided permutation test on the difference of arm means,
  unpaired (10⁴ permutations). The paired Wilcoxon (Pratt zero handling) is reported only. Each row stamps
  `seed_honoured` from B0. Pairing is credited only where the arms share a prior (B vs A, B2 vs B)."

### SF-3. Engagement = 0 scores as 0.0, so a floor that never touches the fire loses, and on the deceptive hearth "never engaged" reads as perfect avoidance

- **Claim attacked:** G2 and G3 on both metrics.
- **Alternative explanation:** `compute_metrics` returns 0.0 when `fire_pit_engagement_count == 0`. A bare or notes
  arm that wanders, or a different action budget per turn, scores 0 on `fire_pit` (G2/G3 "pass" for Maxim) and a
  perfect 0 on `deceptive_fire` (a floor "learns avoidance" by never going there).
- **Fix:** "A trial with zero engagement is undefined (None) for both ratio metrics and is excluded from the gate. The
  per-arm zero-engagement rate and the mean engagement count are reported. If the zero-engagement rates of the two
  arms in a gate differ by more than 20 pp, that gate is INCOMPLETE, not a floor result."

### SF-4. ReAct verbatim and "the same parser" contradict each other, and two prompt sections decide the floor

- **Claim attacked:** OB1 says there is no author freedom and so nothing to sandbag; the Information control.
- **Alternative explanation:** published ReAct emits `Thought: … / Action: tool[arg]` and is few-shot with
  task-specific exemplars. Maxim's parser extracts JSON tool calls. Either the parser rejects ReAct output (bare loses
  on invalid actions, sandbagging in Maxim's favour) or the format is adapted (no longer verbatim), and in both cases
  someone writes the exemplars. Separately, what the bare arm sees decides much of G2. Is the Body State block with
  its `(DRIVE: outside comfort band …)` annotations an observation (embodiment) or bio, and does the drive-modulation
  line from `acting_coach` carry over? A bare arm that never learns its arm is cold has no reason to warm itself.
- **Fix:** "The bare prompt is frozen as a golden file in this prereg before the run: ReAct's zero-shot instruction
  text verbatim, adapted only to the shared JSON action syntax (the adaptation diff shown), no exemplars. The Body
  State block, including DRIVE and WARN annotations, is environment observation and is given to every LLM arm.
  `acting_coach` and `pfc_preamble` are scaffolding (Maxim arms only, as in `maxim_all_off`). Precondition per gate:
  each arm's invalid-action or parse-failure rate is within 5 pp of B's; otherwise that gate is INCOMPLETE."

### SF-5. Turning FearGatedExecutor OFF changes the Maxim arm from every historical fire and needs a `src/` toggle that is not in the work list

- **Claim attacked:** "this design keeps it OFF on every arm."
- **Alternative explanation:** every committed Exp 37/38 fire ran with it ON (`orchestrator.py` wires it
  unconditionally). OFF removes one extra LLM review per tool call, and possible blocks, so B4's "full" is not T1-2's
  "full". Any comparison with the historical Goldilocks numbers is a different configuration, and a FLOORS_BEATEN
  would describe "full minus FearGate". There is no toggle today.
- **Fix:** "Add to Harness work: a FearGate toggle for the sim AUT (config key preferred over env var. An env var
  comes with an autouse conftest scrub and an entry in the brief's env table). The arm label is `full_nofeargate`,
  and the ledger prose says so. The dry run counts FearGate blocks with the gate ON for the Maxim arms. If any block
  touches a fire or hearth tool, the OFF decision returns to the owner."

### SF-6. The typed-abort / `_llm_unavailable` rule, read literally, voids the fire; read per row, it allows differential missingness

- **Claim attacked:** "A fire with typed aborts or `_llm_unavailable` > 0 in a gated row cannot move the row."
- **Alternative explanation:** at the heartbeat's 6.7 % per-sim rate, the chance that 425 sims include none is about
  e⁻²⁸. If "row" means the ledger row, the run cannot move T1-2 by construction. If it means a result row, plain
  exclusion is biased, because the heartbeat's four hits were all in resumed arms (longer prompts) and none in A. More
  context pressure in Maxim's resumed arms means arm-dependent loss.
- **Fix:** "A result row with a typed abort or `_llm_unavailable ≥ 1` is excluded, and its (trial, scenario) group is
  re-run once with the same seed (`_complete_clean_groups` semantics). Per-arm exclusion counts are reported. The fire
  is INCOMPLETE if any gated arm loses more than 10 % of its trials or if two gated arms' exclusion rates differ by
  more than 10 pp. A fire that is not INCOMPLETE may move the row."

### SF-7. The ledger-owed stamps are missing, and arm order is fixed

- **Claim attacked:** the matching controls, as complete.
- **Alternative explanation:** the T1-2 ledger entry (2026-10-18 S1 walk) says B4 "must stamp the `apply_decay` in use
  and the save→resume gap". The prereg omits both. The gap differs by design: B resumes from the A that ran just
  before it, while the second scenario's C resumes from a prior that is several sims (hours) older, and hippocampal
  consolidation pressure decays by wall-clock time. The order inside a trial is fixed (A → B → C…), so serving drift
  within a trial (cache warmth, contention, thermal state on the box) lines up with arm.
- **Fix:** "Each row stamps `apply_decay` and `resume_gap_s` (prior's `saved_at` → this sim's start). The non-dependent
  arms (`bare_llm`, `maxim_all_off`, the C prior, the `llm_notes` prior) run in a seeded random order within each
  trial, with B after A and C after its prior. The order is stamped."

### SF-8. The headroom rule is not tied to δ, and NOT_ISOLATED reads a null as a finding

- **Claim attacked:** the 0.85 threshold; "`NOT_ISOLATED` → T1-2 records that no isolated cross-session delta was
  found".
- **Alternative explanation:** the anchor's current position is A 0.71–0.75 and C 0.80 (heartbeat). At C = 0.80,
  B − C ≥ 0.2 requires B ≈ 1.0 on nearly every trial. A scenario at 0.80–0.85 is scored, yet the design effect cannot
  be expressed there, so the run is set up to return NOT_ISOLATED, recorded as a negative. The 0.85 threshold is not
  derived from δ. A failure to reject at 80 % power is not evidence of absence.
- **Fix:** "NO_HEADROOM when the gate's comparator mean is beyond 1 − δ = 0.80 on the predicted side (C for G1, the
  floor for G2/G3). NOT_ISOLATED is reported with the 95 % CI of B − C and the minimum detectable effect at the
  achieved n. The ledger records 'not detected; MDE = x', never 'no delta'."

### SF-9. `llm_notes`' prior is not matched to B's construction

- **Claim attacked:** "llm_notes: bare_llm resumed with its Reflexion notes from its own session"; cost counts "two
  llm_notes sessions".
- **Alternative explanation:** B's prior is that trial's A row, which is the dose-0 measurement. `llm_notes`' prior is
  an extra bare session that is not the `bare_llm` row, so `llm_notes − bare_llm` is not the analogue of B − A, and it
  costs one more sim per trial-scenario.
- **Fix:** "`llm_notes` resumes from that trial's `bare_llm` session (its reflection is written at that session's
  end), exactly as B resumes from A. The experience is matched in count (one prior session of 12 turns); its content
  is each arm's own behaviour, and the record says so."

## NIT

- **N1. σ provenance.** The prereg cites A's SDs (Qwen14B 0.283, heartbeat 0.328). The gate's variance is B − C's:
  report sd(B − C) 0.298 / 0.384 on the anchor, and say that σ 0.28 per arm reproduces it.
- **N2. Dose.** B2 − B mixes "a second fire session" with "a second resume"; there is no dose-2 C. Page's L on
  A → B → B2 inherits L8's anti-correlation with the starting level. Label both descriptive and confounded with
  resume count.
- **N3. `deceptive_fire` has no anchor data.** Every committed `deceptive_fire` fire used cloud models or R1. Its A/C
  position and SD on Qwen32B are unknown, and n was derived from `fire_pit`. Declare that any pilot used to check
  headroom is excluded from the confirmatory set, and freeze n before it.
- **N4. Recall stamp is post-treatment.** The per-row "You recall:" count depends on whether the agent chose
  `examine`. Report it; never condition a gate on it or use it as a covariate.
- **N5. Shared C prior.** C's two scenario rows within a trial share a prior and are correlated. Per-scenario tests are
  unaffected. Under DNB-2(a) this correlation slightly lowers Var(I); state it.
- **N6. Narrator probe scope.** The probe should assert both that `build_resume_prompt` text never reaches the
  narrator's model input and that the resume prompt's `prev_goal` ("puzzle door and button" in C) does not appear in
  any narrator call. The code path that builds and injects it exists (`orchestrator.py`, `inject_cli`).
- **N7. Mixed-scenario verdict.** The table does not say what happens when G1 passes on one scenario and fails on the
  other, or when one scenario is NO_HEADROOM. Define a per-scenario verdict and a combination rule (moot under
  DNB-2(a)).
