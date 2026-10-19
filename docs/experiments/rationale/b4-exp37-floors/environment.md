# B4 design review: environment lens (does the narrated cradle world and the serving stack afford B4, and can it run?)

Reviewed: b4_exp37_floors_prereg.md DRAFT v1 (2026-10-18)

Reviewer: environment lens (charter: [DESIGN_REVIEW.md](../../DESIGN_REVIEW.md)). Paper only: no sim, LLM or rig
command was run. Numbers come from the committed `docs/experiments/data/37_*.jsonl` / `38_*.jsonl` (scratch
scripts, not committed) and from reading `src/` and `scripts/` at this worktree's HEAD.

## What I verified

**Measured from committed data or by running pure code (no LLM):**

- **Per-sim durations (`duration_s`), Qwen2.5-32B.** June fire (`37_results_qwen32b`): median per arm 1,526–1,613 s,
  max 1,670 s, 0 of 60 over 1,800 s. 1.1 heartbeat (`37_heartbeat_1.1_qwen32b`, August, the most recent full-stack
  Qwen32B fire): median per arm 1,673–1,746 s, p90 up to 1,958 s, max 2,034 s, **9 of 60 over 1,800 s, 41 of 60 over
  1,700 s**. L8 old-hash A: 1,528–1,567 s. Qwen14B: about 750 s. R1-distill: about 1,850–2,220 s.
- **Wall clock per fire.** Fire span ÷ summed `duration_s`: 1.13–1.21 for local fires. The heartbeat's 34.5 h for 60
  rows plus 5 unrecorded C priors works out to about 1,900 s of wall per sim. It includes one unexplained 11,464 s gap
  before trial 1's `sharp_rock` A. The roughly 3,500 s gaps before each C row are the unrecorded peaceful prior plus C.
- **`_llm_unavailable`.** Heartbeat: 4 of 60 rows (6.7 %) carry one each: fire_pit B, B-wire-a-off, B-nac-bias-off and
  sharp_rock C. June Qwen32B: 0 of 60. Old-hash A: 0 of 5. Every row ends with `finish_reason: "cancel"` and
  `turns: 12`.
- **Headroom on `fire_pit`, Qwen32B.** June: A 0.420, B 0.800, C 0.667. Heartbeat: **A 0.750 (SD 0.328), B 0.567,
  C 0.800 (SD 0.217)**. Old-hash A 0.710 (SD 0.119). B − C across all six committed fires: Qwen14B −0.10, Qwen32B June
  +0.13, Mistral24B +0.20 (A at 1.000), R1 +0.04, Sonnet −0.22, heartbeat −0.23. The mean is about −0.03.
- **`deceptive_fire` has never run on Qwen32B.** No Exp 37 file contains it. Exp 38 (`38_results_*`, other models)
  puts `warm_self_engagement_fraction` A at 0.27–0.72 and C at 0.33–0.83. The engagement denominator is 0–8 per
  session, often 1–3.
- **Engagement counts on `fire_pit`.** 1–6 per session in every Qwen fire, so the fraction moves in coarse steps
  (1/3, 1/2, 2/3, …). `compute_metrics` codes zero engagement as **0.0**, not as missing:
  `positive_approach_engagement_fraction = approach_actions / engagement_actions if engagement_actions > 0 else 0.0`.
- **Arc routing.** I ran `select_arc_for_goal` in-process (pure, no LLM). `PEACEFUL_GOAL` ("cradle infant explores
  the puzzle door and button"), the `fire_pit` goal and the `sharp_rock` goal **all select the same `cradle` arc**.
  Its first phase activates `items/cradle_fire_pit`, `cradle_food` and `cradle_cool_air`. The `deceptive_fire` goal
  selects `cradle_deceptive` (the hearth).
- **Phase schedule is deterministic at 12 turns.** `Narrator.decide` advances `_phase_idx` only when
  `_turns_in_phase >= phase.turns_max`; the LLM's `phase` field is advisory. The cradle `turns_max` values are
  4, 2, 3, 3, 5, …, so a 12-turn session runs:
  - turns 1–4, exploration: fire pit, food and cool air;
  - turns 5–6, pain_consequence: the fire pit is presented again;
  - turns 7–9, object_introduction: the sharp rock and blanket appear;
  - turns 10–12, discrimination.

  **`tool_discovery` (lever door and button) would start at turn 13, so it never activates.**
- **Committed corroboration.** The `sharp_rock` A sessions, whose goal names the rock and blanket, called `fire_pit_*`
  tools 22 times in 5 heartbeat sessions and 23 times in 5 June sessions. The goal string does not change the world;
  the arc does.
- **Encoder.** `version_info.sentence_transformers_version` is `"not-installed"` in every local-model Exp 37 file:
  June Qwen14B, Qwen32B, Mistral and R1, the August heartbeat, and the L8 old-hash fire. Only the Sonnet fire has
  5.4.0. Without it, `similarity/encoder.py::_get_encoder` falls back to `_fallback_embed`, a SHA-256 bag-of-words
  hash. The sub-sims run the harness's interpreter (`assert_repo_interpreter`).
- **Seed.** `--seed` calls `seed_all` (Python `random`, numpy and torch in the sim process). The peer backend's
  request payload (`maxim_peer_backend.py`) carries `model`, `messages`, `max_tokens`, `temperature`, `stop` and
  `tools`, but **no seed**. `router.py::_served_call_for` records `seed_honoured` only for the in-process
  `_LlamaCppBackend`, and says an owned local server "is spawned without `--seed`". AUT and narrator calls use
  temperature 0.2–0.3 (`exec_agent.py`, `llm_worker.py`).
- **AUT turn is a wall-clock window.** `generative_runner.py` calls
  `bridge.send_and_wait(narrative, timeout=_aut_turn_timeout_s())`. The default is 30 s (`config_loader.py`,
  `sim.aut_turn_timeout_s`). On timeout the AUT is "IDLE" and the narrator moves on. `bridge._send_and_wait` collects
  whatever actions land before the deadline or settle. Neither the timeout value nor the count of timed-out turns is
  stamped into `report.json`. `benchmark_cross_session.py` does not export `MAXIM_SIM_AUT_TURN_TIMEOUT_S`.
- **Harness timeouts.** `run_one_sim(timeout_s=1800)` and `--subsim-timeout-s` default to 1800. A timeout is a failed
  row followed by a fire abort.
- **Systemic-failure guard.** The early abort runs on cloud models only. The local-model branch resets the counter:
  "the equivalent signal … deferred to a follow-up".
- **Turn binning.** `respond` occurs 32 times in 60 heartbeat sessions and `say` never, so `turn_count_binned` is 1–4
  where the true count is 12.
- **World-side harm channels that depend on the agent.** `_data/reflexes/infant.yaml::thermal_contact` fires
  `damage_component(arms)` on keywords in the percept text ("searing heat", "burning sensation", "reaches into", …).
  It runs in `integration/bio_enrichment.py::_evaluate_reflexes`, a bio-stack module. The cradle phase-1 instruction
  tells the narrator to "use set_entity_sensor to increase arms.thermal toward 0.8" when the infant approaches the
  fire.
- **Imagination.** The live `--sim` path builds the scene manifest with the AUT's NAc top-5 tool biases
  (`orchestrator.py`, W2 Hookup 1) unless `MAXIM_DISABLE_IMAGINATION` or `MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL`
  is set. The Exp 37 harness sets neither.

**Inferred (not measured):**

- Qwen2.5-32B Q4_K_M (about 19–20 GB) does not fit the RTX 5080's 16 GB, so the only box that serves it at full
  speed is big-mac-mini (48 GB, per docs). That box is also the Minecraft rig.
- The bare arms will not be much shorter. The narrator makes two calls per turn (decide and generate) on the same
  32B server whatever the AUT arm, and the 30 s window plus settle bound the AUT's share of each turn.
- The full-Maxim prompt is roughly 5–8k tokens per AUT call (heartbeat about 97k input tokens per 12-turn session).
  The B0 `llm_calls` stamp of a dry run measures it properly.
- The unexplained 11,464 s heartbeat gap is consistent with a host sleep or a re-run, but nothing records which.

---

## DO-NOT-BUILD

### DNB-1. Arm C is not a "wrong content" control: its peaceful prior is a fire-pit session.

**Claim attacked:** "C | Full Maxim resumed from a peaceful-prior session (`PEACEFUL_GOAL`) | Resumed, wrong content:
the L8 gate's comparator", and G1 "B − C: what the agent carried, not being resumed". The reproduction protocol says
the peaceful goal "targets cradle phase 3 entities (lever_door + button) — no thermal/laceration affordances".

**Evidence:**
- `select_arc_for_goal(PEACEFUL_GOAL)` returns the same `cradle` arc as `fire_pit` (run in-process).
- The arc's phase index advances only on `turns_max`, so a 12-turn prior spends turns 1–6 with the fire pit
  (with the narrator told to present it twice) and turns 7–12 with the rock and blanket.
- The lever door and button would activate at turn 13. **They never appear.**
- Committed corroboration: the `sharp_rock` goal, a different goal on the same arc, produced 22–23 `fire_pit_*`
  calls per five sessions.

So the C prior carries fire-pit experience of the same kind as A. In `fire_pit`, B and C differ mainly in the goal
string of the prior session. The historical "C ≈ B" ("any resumed prior shifts behaviour up",
37_cross_model_results.md) is what this predicts, and so is the near-zero mean B − C across the six fires. It also
falsifies the claim in the #845 entry that "arm C's peaceful prior mostly recalls nothing for those targets":
`examine(fire_pit)` in C can recall the prior's fire text exactly as in B.

For `deceptive_fire`, the shared C prior is a safe-fire-pit session (warm_self rewarded). C is then a
prior-reinforcing dose, not a neutral one.

**Fix (prereg text):**
> "Arm C's prior runs an arc that activates no thermal entity: a frozen arc selected by exact name
> (`cradle_peaceful`: the cradle's act-2/act-3 phases with `world_entities` limited to `cradle_lever_door`,
> `cradle_button` and `cradle_blanket`; first phase entered at turn 1), not a goal string routed by
> `select_arc_for_goal`. The C prior is a recorded row (`arm: "C_prior"`) with its own `tool_usage`, activated-entity
> list and `report.json`. A structural probe asserts, on every C-prior row, zero `fire_pit_*` / `hearth_*` tool calls
> and that neither entity was activated; a C-prior row that fails the probe refuses its trial. The dry run reports the
> C prior's `examine` recall targets. A second arm C′ (resumed from a fire-free prior with the *same session count*)
> is not needed if the probe holds."

This is a `src/` change (a new builtin arc), owed before the harness, so it goes to the owner with Open decision 2.

### DNB-2. Every committed local fire ran the substrate on hash embeddings, and nothing in B4 stops a repeat.

**Claim attacked:** that B carries "substrate memory" (Wire-A cluster bias, ATL, episodic recall) in a fire whose
apparatus the record establishes.

**Evidence:**
- `sentence_transformers_version: "not-installed"` is stamped in every local Qwen, Mistral and R1 fire, including the
  August heartbeat on big-mac-mini and the L8 old-hash fire.
- The fallback, `_fallback_embed`, is a SHA-256 of the sorted words. Two texts that differ by any word get unrelated
  vectors, so similarity-based retrieval degrades to exact match.
- `require_semantic_encoder` exists for exactly this, but the harness does not call it, and `report.json` stamps no
  `using_fallback` flag.

On such a box, a `NOT_ISOLATED` verdict says nothing about the claim. The brief's invariant ("code whose OUTPUT IS
EVIDENCE must assert its apparatus before measuring") applies directly.

**Fix (prereg text):**
> "Before the first sub-sim, the harness calls `similarity.encoder.require_semantic_encoder(context="B4")` in the
> sub-sims' interpreter (a one-shot `maxim` subprocess, exit 3 on failure). Every row stamps the encoder's model id
> and `using_fallback`, read from the sub-sim's own report. A row with `using_fallback: true` is `failed`. The encoder
> model identity joins the one-fire stamp."

### DNB-3. The AUT's turn is a 30 s wall-clock window, and the arms differ in prompt length by design.

**Claim attacked:** "Budget: `--sim-max-turns` 12 for every arm; the same wall-clock cap" as a parity control for G2
(B > `bare_llm`) and G3 (B > `llm_notes`), and for A − `maxim_all_off`.

**Evidence:**
- The narrator waits `sim.aut_turn_timeout_s` (default 30 s) for the AUT, then moves on. Actions that land late fall
  into the next scene, or are cut off.
- Qwen32B on this stack takes about 30 s per action (37_cross_model_results.md, R1 section), so the window binds
  near the median.
- Prefill time scales with prompt length. `bare_llm` (ReAct, no bio sections) and `maxim_all_off` will answer
  faster than full Maxim, so they will land more actions per turn and miss fewer turns. That moves the engagement
  denominator and the warm_self count for reasons of latency, not memory.
- Neither the timeout nor the number of timed-out turns is recorded (`report.py` has neither), and the harness does
  not set the timeout, so today it is operator-attested `config.json` state.

**Fix (prereg text):**
> "For B4, the AUT turn is bounded by action, not time: `maxim config set sim.aut_turn_timeout_s 600` (verified with
> `maxim doctor` and stamped in each row from the sub-sim's resolved setting), well above the dry run's measured p99
> AUT response time for the slowest arm. Each row stamps `aut_turns_timed_out` and per-turn action counts. A gated
> row with any timed-out turn is `failed` (re-run under the same stamp). The dry run reports per-arm AUT latency
> (median, p99) and actions per turn. If an arm's p99 exceeds the window, the window is raised before freeze, never
> after data."

Cost note: this lengthens full-stack sims only where they were being cut off. It also removes a hidden source of the
within-fire SD.

### DNB-4. `deceptive_fire`'s gate and headroom rule point the wrong way.

**Claim attacked:** "G1 (L8, binding): B − C > 0 on each scored scenario" and "a scenario whose A mean is ≥ 0.85 or
whose C mean is ≥ 0.85 … is reported `NO_HEADROOM`", applied to `deceptive_fire`'s `warm_self_engagement_fraction`.

**Evidence:**
- `warm_self` on the hearth hurts: `cradle_false_hearth.yaml`, `arms.thermal: 0.6` breaches the 0.5 comfort band.
  Learning from A's pain predicts B warms **less**: a decrease, which is Exp 38's pre-registered direction.
- A one-sided "B − C > 0" test on that scenario scores the maladaptive direction (Exp 38's R1 "amplification") as the
  pass. For a decrease metric, the room to move is at the floor (A or C ≤ 0.15), not the ceiling.
- Zero engagement codes as 0.0, which on this scenario reads as perfect avoidance.
- The scenario has never run on Qwen32B, so whether its metric is in range is unknown. Exp 38 on other models puts
  A at 0.27–0.72, with 0–8 engagements per session.

**Fix (prereg text):**
> "Each scored scenario states its predicted direction: `fire_pit` B − C > 0; `deceptive_fire` B − C < 0 (carried
> pain lowers hearth warming). Headroom is read on the side the prediction moves toward: `fire_pit` is `NO_HEADROOM`
> when A or C ≥ 1 − δ − 0.05; `deceptive_fire` when A or C ≤ δ + 0.05. A session with zero engagement with the
> scenario entity is *missing* for the fraction (not 0.0). Engagement counts per arm are reported, and an arm whose
> engagement rate differs from B's by more than <pre-set> is flagged, since the fraction is then measured on
> different exposure. `deceptive_fire` enters the confirmatory fire only if the pilot (SF-2) shows its A in range on
> the chosen model."

### DNB-5. The weak-evidence void rule, as written, very probably voids the whole campaign.

**Claim attacked:** "A fire with typed aborts or `_llm_unavailable` > 0 in a gated row cannot move the row."

**Evidence:** the heartbeat, on this model and box family, had 4 of 60 rows with one `_llm_unavailable` (6.7 %); June
had 0 of 60. B4 gates on roughly 300 rows (B, C, `bare_llm` and `llm_notes`, × 25 trials, × 2 scenarios, plus A for
headroom). At the heartbeat rate the expected number of affected rows is about 20, and P(none) ≈ e⁻²⁰. Even at 1 %,
P(none) ≈ 5 %. As written, the rule means a 7–10 day campaign almost certainly cannot move T1-2. That outcome is fixed
before any data exists, and the harness's systemic-failure guard is off for local models, so nothing stops a degraded
fire early either.

**Fix (prereg text):**
> "The void rule applies per row and per trial, not per fire. A gated row with any `_llm_unavailable`, typed abort,
> timed-out AUT turn (DNB-3) or fallback encoder (DNB-2) is `failed`; its whole trial (every arm of that trial and
> scenario) is excluded and re-run under the same serving stamp with a new trial index and the next seed. The number
> of excluded trials and their causes is reported per arm. If more than 20 % of trials are excluded, or the exclusions
> are unbalanced across arms (Fisher exact on excluded-by-arm, p < 0.05), the fire is `INCOMPLETE`. The harness gains
> a local-model systemic guard: three consecutive rows with `_llm_unavailable` ≥ 3, or zero non-`respond` actions,
> abort the fire. The `--resume` path re-runs excluded trials whole."

---

## SHOULD-FIX

### SF-1. Cost: the 190 h is an underestimate, and one box cannot hold a single stamp that long without rules.

**Claim attacked:** "≈ 425 sims ≈ 190 h on one box; `fire_pit` alone ≈ 95 h", and "one fire, one serving stamp … a
change mid-campaign refuses every row after it".

**Evidence:**
- 190 h is 425 × the June median, with no overhead. The most recent full-stack Qwen32B fire ran about 1,750 s median
  per sim and about 1,900 s of wall per sim. The bare arms keep the narrator's two 32B calls per turn and the turn
  windows, so they will not be "much shorter".
- Estimate: 425 × about 0.85 × 1,900 s ≈ **190–225 h, about 8–9.5 days of exclusive time**. That is before the dry
  run, the pilot (SF-2), DNB-3's longer windows where they bind, and the DNB-5 re-runs (more than 6 % at the
  heartbeat rate).
- `fire_pit` alone: about 4–5 days. n = 16 on both scenarios: about 5–6 days.
- Saving: `llm_notes`'s first session can be the `bare_llm` row itself, resumed with its end-of-session reflection.
  That is exactly how B reuses A, and the reflection runs after the session, so it cannot change it. This takes 1 sim
  off every trial and scenario (8.5 → 7.5) and pairs the floors the way A and B are paired.
- **The box.** Qwen32B at speed means big-mac-mini (inferred: it does not fit the 5080's 16 GB). The harness must run
  on another box: the Mac peer, over the LAN to :8100, not the tunnel. The tunnel's 530s/502s become
  `_llm_unavailable`, and its URL is a stamp field.
- **Rig conflicts.** big-mac-mini is also the Minecraft rig that Sessions S, P and H need in 1.3.3–1.3.6. A 9-day
  exclusive window there needs an explicit reservation (S8), with Paper servers, bridges and ollama stopped for the
  whole window (§5 "quiet box").
- **What the stamp sees, and what it misses.** Over 9 days, macOS auto-updates, host sleep (the unexplained 3.2 h
  heartbeat gap), server crashes and restarts, a `maxim peer update`, and a DHCP change of the endpoint all happen.
  - The B0 stamp catches a model-file, quantization, n_ctx, endpoint or build change, which refuses everything after
    it and kills the campaign.
  - It does **not** catch a macOS or Metal update or a restart with identical fields, which L8 names among the
    suspects.

**Fix (prereg text):**
> "Topology: server = big-mac-mini :8100 (LAN address pinned, no tunnel); harness = <named peer box>; one harness on
> the server for the whole campaign; no other consumer of :8100 (`concurrent_sims` false on every row); macOS
> automatic updates off and `caffeinate -dims` held on both boxes for the window; the stamp gains the server host's
> OS build, the llama-cpp-python version, the GPU backend and a server process start time.
>
> The unit of comparability is the trial: every gate statistic is a within-trial difference. So a stamp change does
> not end the fire. It closes the current block; trials straddling it are excluded whole and re-run. The fire is a
> set of blocks with one stamp each, and the analysis reports per-block A and B − C (descriptive block × arm check).
> A block with fewer than 5 complete trials is dropped. Any change to model file, quantization, n_ctx or KV type
> ends the fire (those are the model, not the environment).
>
> Cost is restated from the dry run's measured per-arm wall time before freeze, together with the box reservation."

Chunking with `--resume` keeps within-fire comparability under this rule. The current `_apply_resume_prune` re-runs
whole (trial, scenario) groups. It must also re-run a trial's shared C prior when either scenario's group is pruned,
and keep workdirs outside `/tmp` (see the NIT).

### SF-2. Headroom: on today's evidence a scorable `fire_pit` fire on Qwen32B is unlikely. Choose the model by pilot.

**Claim attacked:** "Model: one fixed model for the campaign (Qwen2.5-32B Q4_K_M, the anchor)", with the 0.85
`NO_HEADROOM` rule and δ = 0.20.

**Evidence:**
- On the most recent Qwen32B stack, A = 0.71–0.75 and **C = 0.80**. A one-sided B − C of δ = 0.20 needs B ≈ 1.00,
  on a bounded metric whose SD there must shrink toward 0. The power calculation (σ 0.28) does not hold near the
  ceiling.
- The 0.85 rule would still score such a fire, so a `NOT_ISOLATED` verdict would be a ceiling artefact recorded as a
  finding. That is the S7/L6 void the rule exists to prevent.
- The "anchor" argument has no value left: L8 says the anchor result cannot be re-derived.
- Qwen14B (June: A 0.53, C 0.62, about 750 s per sim) would leave room and run in about 40 % of the time, on the
  5080 box, which frees big-mac-mini.

**Fix (prereg text):**
> "A pre-registered pilot (its own data file, never pooled) runs A, C-prior and C, n = 6 trials, on `fire_pit` (and
> `deceptive_fire`) for each candidate model (Qwen2.5-32B on big-mac-mini; Qwen2.5-14B on the RTX 5080), on the
> confirmatory code and stamp. The confirmatory model is the first in that order whose pilot A and C means both lie in
> [0.20, 1 − δ − 0.10] for `fire_pit` (and in [δ + 0.10, 0.80] for `deceptive_fire`). If none qualifies, B4 records
> `NO_HEADROOM (pilot)` for that scenario, and T1-2 stays STALE with that reason; the owner chooses whether to widen the
> candidate set. The in-fire headroom rule (DNB-4's δ-aware thresholds) still applies to the confirmatory fire."

### SF-3. "Paired by seed" pairs no LLM sampling.

**Claim attacked:** "Seeds plumbed … the same seed per trial across all arms", "pairing by trial seed", and
baseline_floors' "Same temperature, top_p, and seed per paired trial".

**Evidence:**
- The served path (`_MaximPeerBackend` → llama_cpp.server :8100) sends no seed, and the owned server is spawned
  without `--seed`. B0 records `seed_honoured` false for it.
- AUT and narrator calls sample at temperature 0.2–0.3.
- `--seed` seeds only in-process Python randomness. PYTHONHASHSEED cannot be changed after the interpreter starts,
  so "Sets PYTHONHASHSEED" applies only to children.
- Even a per-request seed would not pair two arms past their first differing token: the prompts differ by design and
  the narrator reacts to the AUT.

What the pairing really pairs is the *trial block*, the arms run close in time, which is still worth having under L8
(slow drift in the serving environment).

**Fix (prereg text):**
> "The trial index, not the seed, is the pairing unit: arms of one trial run consecutively under one stamp, and the
> signed-rank test pairs them for that reason. `--seed` is plumbed and stamped (it fixes in-process randomness:
> imagination, tie-breaks), and every row stamps `seed_honoured` from B0. No claim of sampling-level pairing is made.
> Within each trial the order of the independent chains (A→B→B2, C-prior→C, `bare_llm`→`llm_notes`,
> `maxim_all_off`) is randomized per trial from the seed and frozen in the manifest, so arm is not confounded with
> position in the 4–5 h trial."

### SF-4. The world the arms live in is not the same world: imagination and reflexes are bio-stack features.

**Claim attacked:** "Information: the same observation text, tool list, parser and retry policy across the LLM arms".

**Evidence:**
- **Imagination.** On the live `--sim` path the scene manifest is generated from the AUT's NAc top-5 biases, so B, C,
  A, `maxim_all_off` (no NAc) and `bare_llm` each get a different set of pre-instantiated entities, and so different
  tools and distractors. The harness sets neither `MAXIM_DISABLE_IMAGINATION` nor
  `MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL`. The brief notes that LLM-primary AUTs engage improvised entities.
- **Reflexes.** The infant `thermal_contact` reflex turns narrator words ("searing heat", "reaches into") into arm
  damage only where the bio stack's reflex registry runs. The cradle instruction also asks the narrator to raise
  `arms.thermal` toward 0.8 on approach. The world's harm contingency on `fire_pit` therefore depends on the arm and on
  the narrator's wording.
- **Narrator.** The narrator receives `last_aut_response` and paces on it, and ReAct's
  "Thought:/Action:" text is not Maxim's reply text. The observation stream is endogenous, so "the same observation
  text" can only mean the same narrator code with the same inputs class.

**Fix (prereg text):**
> "All arms run with `MAXIM_DISABLE_IMAGINATION=1` (declared S6 apparatus setting; the world is the arc's declared
> entities only). Each arm declares which world mechanics act on it: self_effect application, drive ticks, reflex
> registry, narrator sensor writes. Parity is on what the world does, so the reflex and self_effect paths run in every
> arm (they are world physics, owned by the orchestrator), or in none; the dry run's structural probe shows which. Each
> row stamps the activated-entity list, per-turn narrator phase and any world-side sensor or damage writes. The arm
> parity test (baseline_floors §Guards) asserts identical entity activation and tool lists per turn given identical
> narrator text."

### SF-5. n_ctx, and the ReAct transcript, are not sized.

**Claim attacked:** that OB1's "published ReAct format, verbatim" and one fixed n_ctx serve every arm.

**Evidence:**
- L8: 4,096 overflows the full prompt (10 dead calls). The full prompt is budgeted (worker clamp). A verbatim ReAct
  loop appends every Thought, Action and Observation, and a 12-turn narrator transcript plus few-shot exemplars grows
  without bound. `llm_notes` adds the notes.
- Under one stamp, n_ctx is fixed, and an overflow is an HTTP 500, then `_llm_unavailable` (the n_ctx-drift lesson),
  then a voided row (DNB-5).
- Separately, verbatim ReAct's `Action: tool[arg]` grammar is not Maxim's tool-call parser, so "verbatim" and "same
  parser" cannot both hold literally.

**Fix (prereg text):**
> "The dry run records each arm's maximum prompt tokens at turn 12 from B0's `llm_calls`. n_ctx is set by
> `maxim config set llm.n_ctx` to at least 1.25 × the largest arm's maximum plus `max_tokens`, verified by
> `maxim doctor`, and frozen in the stamp. The bare arms' history-truncation policy (none expected at that n_ctx) is
> frozen. OB1 is restated as 'the ReAct Thought/Action/Observation structure verbatim, with Maxim's tool-call syntax
> and parser in the Action slot, and no few-shot exemplars' (or names the exemplars), so the parser parity holds."

### SF-6. Harness timeouts must be frozen above the measured tail.

**Evidence:** the `--subsim-timeout-s` default is 1,800 s. 9 of 60 heartbeat sims ran longer than that (subprocess
wall is longer still), and a timeout is a failed row followed by a fire abort. DNB-3's longer turn window lengthens
the tail.

**Fix (prereg text):**
> "`--subsim-timeout-s` = max(3 × dry-run p99 sim wall, 7,200), stamped in the manifest. A sub-sim timeout is a
> `failed` row and excludes its trial (DNB-5); it does not abort the fire unless the local systemic guard trips."

---

## NIT

- **`sharp_rock` is dropped for the right reason, stated wrong.** It is not just "zero engagement": the `sharp_rock`
  goal runs the same `cradle` arc as `fire_pit` (22–23 `fire_pit_*` calls per five sessions), and the rock enters at
  turn 7. The old fires' second scenario was the fire-pit world again, with a metric that cannot move. Say so in
  §Not claimed. It also means no committed fire tested a second world.
- **Per-turn binning is degenerate and format-dependent.** `TURN_BOUNDARY_TOOLS = {say, respond}`, but `respond`
  occurs about 0.5 times per session and `say` never, so `turn_count_binned` is 1–4, not 12. A ReAct arm's closing
  convention differs again. Report no binned metric across arms. Stamp the bridge's `_turn_count` on each action
  instead, so turn-level analyses use the real turn.
- **12-turn session shape.** At 12 turns the fire pit is foregrounded only in turns 1–6. Turns 7–12 add the rock and
  blanket, and the arc never reaches its recall phase. The fraction pools both halves. Report the turn-1–6 and
  turn-7–12 fractions descriptively; that needs the per-action turn stamp above.
- **Workdirs and disk.** Past fires used `/tmp/exp37_*`. macOS's periodic cleanup can remove old files under
  `/private/tmp` during a 9-day campaign, and `--resume` needs A's home. Put `--workdir` on a persistent volume.
  Sandboxes are about 200 MB each (harness docstring), so about 85 GB for 425 sims unless `--cleanup-after-trial` is
  set after the trial's rows and reports are copied.
- **Dose B2 doubles A's exposure only if A engaged.** The dose count is "sessions carried", not "fire exposures". The
  per-row engagement count (DNB-4) lets the dose curve be read against actual exposure.
- **The narrator "done" path.** `Narrator.decide` ends the sim if the LLM returns `done: true`. No committed row ended
  early (all 12 turns), but a gated row with fewer than 12 turns should be `failed`, not scored.

---

## Summary by severity

- **DO-NOT-BUILD (5):**
  1. C's peaceful prior routes to the same `cradle` arc and spends turns 1–6 at the fire pit. The door and button
     never activate at 12 turns. G1 does not isolate carried content (fix: a frozen fire-free arc plus a C-prior row
     and probe).
  2. Every committed local fire ran on the hash-embedding fallback (fix: `require_semantic_encoder` preflight and a
     per-row `using_fallback`).
  3. The AUT turn is a 30 s wall-clock window, so prompt length, which is the manipulated variable, changes actions
     per turn (fix: an action-bounded window, stamped, with timed-out rows failed).
  4. `deceptive_fire` gates in the wrong direction, with its headroom read on the wrong side and zero engagement coded
     0.0.
  5. The fire-level `_llm_unavailable` void rule almost surely voids a roughly 300-gated-row campaign (fix: exclude
     whole trials, rows failed one by one).
- **SHOULD-FIX (6):**
  1. Cost is about 190–225 h, about 8–9.5 days of exclusive big-mac-mini (rig conflict). It needs pinned LAN
     topology, OS/sleep/update controls, OS build added to the stamp, and blocks by trial rather than
     one-stamp-or-nothing. `llm_notes` can reuse the `bare_llm` session to save a sim per trial.
  2. On today's Qwen32B, A ≈ 0.71–0.75 and C ≈ 0.80, so δ = 0.2 is near-infeasible. Choose the model by a headroom
     pilot (Qwen14B is a candidate) with δ-aware thresholds.
  3. The server honours no seed, so pairing is by trial block. Randomize chain order within each trial.
  4. Imagination (NAc-biased manifest) and the reflexes make each arm's world different. Disable imagination and
     declare the world mechanics per arm.
  5. Size n_ctx and the ReAct transcript from the dry run. Verbatim ReAct conflicts with "same parser".
  6. The 1,800 s sub-sim timeout is shorter than 15 % of the heartbeat sims.
- **NIT (6):** the stated reason for dropping `sharp_rock`; degenerate turn binning; the 12-turn session shape;
  `/tmp` workdirs and about 85 GB of disk; dose versus actual exposure; narrator early `done`.
