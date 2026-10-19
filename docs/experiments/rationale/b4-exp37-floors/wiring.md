# B4 four-lens design review: wiring lens

Reviewed: b4_exp37_floors_prereg.md DRAFT v1 (2026-10-18)

Lens question: real consumers and real credit paths (D43). Will every arm run what the prereg says, and can every
gate be computed from the rows? Read at the e-paper worktree (base `68b70466`). One offline probe was run (below).
No sim or LLM call was made.

## What I verified

- **Harness path.** Every arm the harness can run today is a `maxim --sim` subprocess
  (`scripts/benchmark_cross_session.py::_real_sim`). It passes `--embodiment bodies/infant_humanoid`, no `--seed`, and
  no `--aut-model`. On the peer path it also omits `--language-model`. `run_benchmark` hard-codes three arm blocks
  (A, the B family, C). The C prior is not a row: only its cost is added.
- **AUT construction in the sim.** `simulation/orchestrator.py::start_simulation_mode` builds the AUT with
  `AgentConfig(with_bio_stack=True, with_pain_bridge=entity_ref is not None, ...)`. It then adds bio wiring outside
  `create_full_agent`:
  - `PainInterceptorExecutor` and `AnticipatoryPainExecutor` (with `PerceivedPainAssessor(nac=...)`);
  - an unconditional `FearGatedExecutor`;
  - `build_default_network(nac=...)`;
  - the `ImaginationTrigger` and the W2 substrate-aware scene manifest (`aut_nac.get_agent_tool_biases`);
  - `SubstrateTelemetry`, `bridge.percept_anxiety_hook`, and `examine` recall through the AUT hippocampus.

  `_aut_worker` always runs `run_agentic_loop` with those handles. No CLI flag or env var selects a different AUT
  stack.
- **Probe, `maxim_all_off` as the prereg defines it.** I built `create_full_agent` with `with_bio_stack=False,
  remembers=False, learns=False, with_pain_bridge=True, embodiment_ref="bodies/infant_humanoid"`, the sim's own config
  shape. It raises `ValueError: build_executor: entity_ref='bodies/infant_humanoid' was provided but pain_bus is None`
  (`runtime/bootstrap.py::build_executor`). Script:
  `/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/15239c88-ef6e-47b7-8925-0c2ae097dbd4/scratchpad/b4w/probe_all_off.py`.
- **#1226, narrator resume prompt: VERIFIED on the generative path, by reading the code.**
  - `cli.py` sets `generative=True` when `select_arc_for_goal(goal)` matches. Offline, all three B4 goals match:
    `fire_pit` → `cradle`, `deceptive_fire` → `cradle_deceptive`, and `PEACEFUL_GOAL` → `cradle`.
  - In `start_simulation_mode`, `run_generative_campaign` runs synchronously. `_finish_runner` then sets
    `stop_event`, and only after that is `build_resume_prompt` injected into `orchestrator_source`.
  - `run_agentic_loop` (orchestrator) checks `stop_event` first: `loop_gates.py::pre_tick_gate` returns BREAK before
    `sim_adapter.next_percept()`.
  - `run_generative_campaign` takes only `goal` and never reads the resume data (no resume reference in
    `campaign_runner.py`, `generative_runner.py` or `arcs.py`).
  - The AUT's `bridge.percept_source` is a separate `ConversationalSource`.
  - The orchestrator's own bio stack uses `load_persisted=False`.

  Two caveats:
  - The path choice is silent. `_use_generative` is set under `except Exception: pass` in `cli.py`, and when it is
    false the orchestrator loop DOES consume the resume prompt (which includes the prior `tool_usage`).
  - The choice is not stamped in `report.json`.
- **A resumed home resumes again (B2 = dose 2).**
  - `_own_session` finds the spawn by run id among the new session dirs, and `depends_on` lists every inherited
    session.
  - `report.py::resume_stamp` requires `context_dir == state_dir` and every store `loaded` or `absent`.
  - `serving_provenance.py::_session_index` gives A = 0, B = 1, B2 = 2, C = 1, which is a free structural dose check.
  - NAc restores with `apply_decay=False` (`_restore_aut_from_session`). SCN is saved but not in `RESUME_STORES`, so
    it is not carried.
- **Seed.** `--seed` parses with `--sim` (`cli_parser.py`). `cli.py` calls `seed_all` (stdlib random, numpy, torch,
  `PYTHONHASHSEED` env, the global seed). It reaches:
  - the LSH hyperplanes (`similarity/semantic.py`: `get_global_seed() or 42`);
  - the shuffled-stimulus seed (opt-in env only);
  - the B0 stamp `global_seed`, which is echoed into the row's `sims`.

  It does not reach LLM sampling on the peer path. `_MaximPeerBackend` sends no seed, and
  `router.py::_served_call_for` stamps `seed_honoured=True` only for the in-process `_LlamaCppBackend`.
- **B0 stamps, per sim.**
  - `llm_calls`, per router serial: entries keyed `(provider_key, model, backend_kind, endpoint, served_id,
    quantization, seed, seed_honoured, temp)`, plus the joined `identity` and `identity_source`, plus `kv_quant`.
  - Also `sampling_complete`, `model_changed_during_run`, `concurrent_sims` (one process only),
    `leader_disk_heads`, `scenario_sha256`, `global_seed`, `session_index`, and
    `aut_model_attribution: "not_established"`.
  - On the peer path, identity is the served-id alias (`server_reported`), and quantization and KV are `None`.
  - `leader_disk_heads` is NOT in `_provenance._EVIDENCE_PROVENANCE`, so rows do not carry it.
  - The AUT shares the narrator's router unless `--aut-model` is passed (`aut_router = llm_router`).
- **B0b status.** Not landed. It is homed as a 1.3.6 bullet (#1227, #1228). The gate's `judge_b0_fields` fails any
  `aut_model_attribution` other than the literal `"not_established"`.
- **The gate on the B4 data.**
  - `judge_row_file` judges each row's `sims`, never `depends_on`.
  - `lint_bf2_obligation.py::data_time` needs a numeric top-level `ts` on harness rows. Exp 37's `build_record`
    writes `wall_clock_iso` only and does not use `_provenance.stamp_harness_row`, which requires `ts`.
  - `lint_evidence_gate.py` allows exactly one scope for a kind with a complete-run rule.
  - `lint_prereg_precedes_data.py` maps prereg `b4_exp37_floors_prereg.md` to token `b4`. A post-2026-08-29 `.jsonl`
    with no `ts` FAILs.
  - The pass table has no Exp 37 kind at all.
- **Analyzer.** `analyze_exp37.py` loads `SCHEMA_VERSION`, `EXPERIMENT_ID` and `ARMS` from the harness module. Its
  `--strict-schema-version` default is the harness's `SCHEMA_VERSION`, and `assert_design_complete` defaults to the
  harness `ARMS`. The harness comment by `SCHEMA_VERSION` says committed 1.0 files "must still reproduce without a
  flag".
- **The `_llm_unavailable` and typed-abort gates are computable from rows.** `_llm_unavailable` is a `tool_usage` key
  (four occurrences in the heartbeat fire). Typed aborts show in `finish_reason`, and `_evidence_records.py::FINISH_OK`
  refuses non-citable ones.
- **Timing.** The heartbeat Qwen32B fire had 9 of 60 sims over 1,800 s (p90 1,838 s), which is the harness's default
  `--subsim-timeout-s`.

## DO-NOT-BUILD

### D1. Three of the seven arms have no selector on the path the harness drives, and `maxim_all_off` as written cannot be constructed

- **Claim attacked.** `bare_llm`, `llm_notes` and `maxim_all_off` run "what runs" in the Arms table, with
  `maxim_all_off` defined by `create_full_agent` flags plus `build_executor(pain_bus=None, nac=None)` and
  `memory_hub=None`.
- **Evidence.**
  - `simulation/orchestrator.py::start_simulation_mode` hard-codes `with_bio_stack=True` and hard-wires `_aut_worker`
    to `run_agentic_loop` with bio handles. `maxim --sim` exposes no way to choose another AUT stack.
  - The bare and notes arms need an AUT seam that does not exist until the 1.3.5 slices land (B2's entry condition,
    `baseline_floors.md` §B2; `roadmap_1_3_x.md` §1.3.5).
  - The probe above shows `maxim_all_off` raising on Exp 37's embodiment (`runtime/bootstrap.py::build_executor`:
    `entity_ref` with `pain_bus=None`).
  - Even if it built, the sim adds bio around the factory product: the pain executors with
    `PerceivedPainAssessor(nac)`, `build_default_network(nac)`, `ImaginationTrigger` with W2 biases,
    `SubstrateTelemetry`, `examine` recall and `percept_anxiety_hook`. Flags on `create_full_agent` do not reach
    these, so "scaffolding vs bio" would not be what runs.
  - No row says which stack actually ran. `arm_effective` is not built (M46 names it as needed).
- **Fix (prereg text).**

  > Arms are selected at the sim seam, not the factory. After the 1.3.5 `start_simulation_mode` slices, one
  > `maxim --sim` option (for example `--aut-stack full|all_off|bare|notes`) is the only selector; the harness passes
  > it for every arm. The prereg carries a table of every sim-side attachment and its disposition per arm:
  > `create_full_agent` bio stack, `PainInterceptorExecutor`, `AnticipatoryPainExecutor` / `PerceivedPainAssessor`,
  > `FearGatedExecutor`, `DefaultNetwork`, `ImaginationTrigger` plus the W2 manifest, `SubstrateTelemetry`,
  > `percept_anxiety_hook`, `examine` recall, and `_restore_aut_from_session`.
  >
  > `maxim_all_off` keeps the embodiment (so the tools and observation text match) and gets a PainBus with no learner
  > subscribed; that substitution is recorded.
  >
  > `report.json` stamps `arm_effective`, read from the objects actually built rather than echoed from the flag, and
  > it is echoed into the row's `sims`. The harness refuses a row whose `arm_effective` differs from its `arm`. A
  > mocked one-turn test per arm asserts `arm_effective`, and a deletion probe shows that a forced wrong stack is
  > refused.

### D2. On the prescribed apparatus the model identity is only attested, so by the project's own rule B4 cannot move T1-2

- **Claim attacked.** "B0's serving provenance must be identical across every row (model file hash, quantization, KV
  type, n_ctx, endpoint, server build) … B0b … so the agent under test's model is attested, not inferred." The
  verdict is then supposed to move T1-2.
- **Evidence.**
  - The harness runs on a different box from the server (CLAUDE.md; the prereg's own Model bullet), so every call goes
    through `_MaximPeerBackend`. `serving_provenance.py::_join_identity` then gives `identity_source:
    "server_reported"`: the `/v1/models` served-id alias, with `quantization` and `kv_quant` `None`. A model file hash
    exists only for an owned local spawner.
  - n_ctx is the client's belief (`configured_n_ctx`, `*_router_n_ctx`). The trusted `served_n_ctx` is out of B0's
    scope.
  - "Server build" is `leader_disk_heads`, the HEAD on disk, which `baseline_floors_b0.md` says is "never code
    identity". `_provenance._EVIDENCE_PROVENANCE` does not echo it into rows.
  - `baseline_floors_b0.md` (fields table) says that `server_reported` and `provider_name` are treated as attested
    under the weak-evidence rule. CLAUDE.md: a record whose model or context size is only operator-attested "cannot be
    the sole evidence for a status change". That collides with the owner decision that B4 resolves T1-2 before 1.4.0.
- **Fix (prereg text).**

  > Before the run, the leader reports established serving identity: the sha256 of the served model file, the
  > server's n_ctx and KV type (for example from llama-server `/props` through the leader proxy), stamped as a new
  > identity source that the gate accepts. That is a src change plus a gate change, and the gate change gets an
  > adversarial design pass.
  >
  > Otherwise the owner records a committed, machine-readable exception for T1-2 that names attested identity. The
  > strict option (wait for established identity) is listed first and is the default recommendation.
  >
  > Either way, the cross-row refusal compares the tuple `(provider_key, model, backend_kind, endpoint, served_id,
  > identity, identity_source)` for every entry on the AUT and narrator serials, plus `leader_disk_heads`, which
  > `_EVIDENCE_PROVENANCE` gains. Temperature and call mix are excluded, because they differ by arm legitimately.

### D3. Nothing proves the resumed arm received what its prior carried; the upstream plan's B4 stamps were dropped

- **Claim attacked.** G1: "B − C … what the agent carried". The B, B2 and C rows are taken as resumed because
  `_own_session` checks `resume_loaded`.
- **Evidence.**
  - `report.py::resume_stamp` counts a store as fine when it is `absent`, meaning the prior never wrote it.
  - `report.py` (the end-of-session AUT store saves) swallows save failures at `logger.debug`, and returns early on
    `StoreOverwriteRefused` (#939/#971). So a prior whose NAc or hippocampus failed to save yields a resume that reads
    `resume_loaded: true`.
  - `experience_clock_decay.md` §Impact, "T1-2 (Exp 37) re-fire", names stamps "landed now in the harness" with a
    **hard fail**:
    - for each of the three bias maps, the count, sum |v|, max |v| and target-key values, at save and right after load;
    - `saved_at`, the save→resume gap, and the `apply_decay` in use;
    - tick count, submit count, median and p90 per-turn latency;
    - end-of-session magnitudes.

    None is in `scripts/benchmark_cross_session.py` (grep finds no `saved_at`, `apply_decay`, tick, submit or latency
    stamp), and the prereg omits them. The ledger's S1 walk for T1-2 repeats the obligation, and
    `simulation-experiments.md` §4 asks for passes, submits and latency "per arm when a result could depend on decay".
  - `llm_notes` has no store in `RESUME_STORES`, so a resume whose notes failed to load passes `_own_session`.
- **Fix (prereg text).** Bring the experience-clock plan's stamp list into Matching controls, verbatim, including the
  hard fail. Then:

  > For every resumed row, each store the prior session held in-run must be `loaded`, never `absent`, with the saved
  > count equal to the loaded count. Otherwise the row is refused, not recorded as data.
  >
  > `llm_notes` stamps `notes_loaded` and its note count, and the harness refuses the row when the count is 0 or the
  > load failed.
  >
  > The prereg states which NAc decay regime the fire runs on (the experience-clock S3–S6 work lands in 1.3.6, before
  > B4) and stamps it.

## SHOULD-FIX

### S1. FearGatedExecutor is ON in every `maxim --sim`, with no switch, and the prereg's reason is wrong

- **Claim attacked.** "Keeps it OFF on every arm (it costs an extra LLM review per call)."
- **Evidence.** In `orchestrator.py::start_simulation_mode` the AUT executor is unconditionally wrapped in
  `FearGatedExecutor(aut_executor, FearAgent(llm=llm_router))`, on the narrator's router (#1228). No env var or flag
  turns it off. `agents/fear_agent.py::FearAgent.review_action` is pattern-only: the LLM is used only by
  `_analyze_with_llm` for code. On cradle affordance tools it makes no LLM call, and it denies only on shell, file,
  network or movement-pain predictions.
- **Fix.**

  > FearGatedExecutor is ON in every Maxim arm as shipped. It is pattern-only on the cradle tool set; the bare arms
  > have none. Each row stamps its review and deny counts (the `_reviews` counter exists; add denials), and a gated
  > row with any denial is refused.

  The alternative is a toggle, which needs a new env var with a `tests/conftest.py` scrub and a brief-table row.

### S2. Imagination, and the W2 substrate-aware manifest, make the world differ by arm

- **Claim attacked.** Information parity: the same observation text across the LLM arms.
- **Evidence.**
  - `orchestrator.py` builds `ImaginationTrigger` whenever an embodiment is set. On the generative path it calls
    `generate_scene_manifest` with `aut_nac.get_agent_tool_biases(...)`, so in B and C the LLM-generated scene entity
    list is steered by the carried NAc.
  - `bare_llm` and `llm_notes` have no NAc, so their world differs.
  - The harness never sets `MAXIM_DISABLE_IMAGINATION`, and `simulation-experiments.md` already warns that imagination
    distractors break controlled A/Bs.
- **Fix.** Decide and record. The recommendation is `MAXIM_DISABLE_IMAGINATION=1` on every arm, stamped per row. The
  prereg then says that this removes the W2 world-builder channel from B, which narrows what "carried" means relative
  to June. The other option keeps imagination on and runs the bare arms through the same world builder with null
  biases, which the bare-arm guard must then allow.

### S3. Turn #1226's reading-level verification into a per-row stamp plus a probe that can fail

- **Claim attacked.** "A structural probe asserts the cradle campaign's narrator never receives `build_resume_prompt`
  text in any arm."
- **Evidence.** The verification above holds only on the generative path. That path is chosen silently (`cli.py`:
  `select_arc_for_goal` under `except Exception: pass`) and never stamped. B0's call counts cannot tell the narrator
  orchestrator loop from the generative narrator, because both use `llm_router`.
- **Fix.**

  > `report.json` stamps `campaign_runner`, `arc`, and whether `orchestrator_source` delivered any percept
  > (`orchestrator_percepts_consumed`). The harness refuses a cross-session row unless the runner is
  > `generative_runner` and the count is 0.
  >
  > An offline test runs mocked `start_simulation_mode(generative=True, resume_session=<a prior holding a sentinel in
  > its findings and tool_usage>)`, captures every prompt sent to every router, and asserts the sentinel is absent. Its
  > deletion probe forces `generative=False` and must see the sentinel.

### S4. "Seeds plumbed" pairs the Python RNGs only

- **Claim attacked.** The same seed per trial means matched sampling.
- **Evidence.** `seed_all` seeds Python RNGs and the LSH planes (`semantic.py`). The peer backend sends no seed, and B0
  stamps `seed_honoured: false` (`router.py::_served_call_for`).
- **Fix.**

  > `--seed` pairs the Python RNGs and the LSH planes. LLM sampling is unseeded on the peer path, and
  > `seed_honoured: false` is stamped and expected. The harness refuses a row whose `sims[*].global_seed` differs from
  > the row's `seed`. A trial's prior and resumed sessions share the seed, which keeps the LSH planes consistent.
  > `--seed` moves the planes off the historical 42, which is harmless because nothing is compared with June.

### S5. The evidence entry as written cannot be built: one kind, one scope

- **Claim attacked.** "A new `b4_verdict` kind … its `complete.sets` and scopes … a `COMPLETE_RULES` entry; the BF2
  obligation (M46)."
- **Evidence.**
  - `lint_evidence_gate.py` allows exactly one scope for a ruled kind. `FLOORS_BEATEN` must support an unqualified
    MAINTAINED or EARNED, and `ISOLATED_FLOOR_MATCHED` must support `PARTIAL (narrow: …)`. No Exp 37 kind exists to
    carry the raise.
  - The existing complete rules key on one seed field (`_evidence_records.py::_keyed`). B4 needs
    (arm, scenario, trial_pair_id).
  - Inherited priors that are not rows (the C prior, the first `llm_notes` session) are never judged, because
    `judge_row_file` ignores `depends_on`.
  - Exp 37 rows lack `ts`, so `lint_bf2_obligation.py::data_time` and `lint_prereg_precedes_data.py` fail.
  - The analyzer writes markdown, not a stamped verdict.
  - The obligation lint accepts `PARTIAL (narrow)` for any obliging value, which is wrong for `NOT_ISOLATED`.
  - M46 is unbuilt and needs B0b and `arm_effective`.
- **Fix.**

  > Two kinds: `b4_verdict` (scope null; targets MAINTAINED, EARNED and RE-VALIDATED ← `FLOORS_BEATEN`) and
  > `b4_narrow_verdict` (scope `narrow`; PARTIAL ← `ISOLATED_FLOOR_MATCHED`; `obliges` T1-2, clears
  > `FLOORS_BEATEN`), on the `b1_verdict` precedent. The prereg states what `NOT_ISOLATED` sets T1-2 to and how that
  > is enforced.
  >
  > A new `COMPLETE_RULES` rule keyed by (arm, scenario, trial_pair_id) gets its gate-design pass.
  >
  > Every inherited prior is emitted as its own row (`C_prior`, `llm_notes_prior`), so the gate judges it.
  >
  > Rows go through `_provenance.stamp_harness_row` with an epoch `ts`. The data file is named `b4_…`. The analyzer
  > emits `_provenance.stamp_verdict`.
  >
  > The obligation is the existing `obliges` mechanism, a partial M46. Full M46 waits on B0b and `arm_effective`.
  >
  > B0b's landing must change `judge_b0_fields`'s literal `"not_established"` check, or every post-B0b row fails.
  > The harness passes `--aut-model <anchor profile>` so that `aut_router is not llm_router`, and refuses a row whose
  > AUT and language router serials are equal.

### S6. A `SCHEMA_VERSION` bump on the shared constant breaks the historical files' analysis

- **Claim attacked.** "A second frozen mapping … the analyzer's second arm set."
- **Evidence.** The analyzer's default `--strict-schema-version` and its default arms both come from the harness's
  single `SCHEMA_VERSION` and `ARMS`. A bump makes the committed 1.0 files fail by default, against the harness's own
  comment that they "must still reproduce without a flag". `tests/unit/test_harness_rows_1003.py` and the smoke tests
  read `harness.ARMS`.
- **Fix.**

  > Leave `SCHEMA_VERSION`, `EXPERIMENT_ID` and `ARMS` untouched. Add `B4_EXPERIMENT_ID = "b4_exp37_floors"` and
  > `B4_ARMS`. The analyzer dispatches on the record's `experiment`, and already refuses mixed files. The smoke test
  > gains a B4 `arms_seen == set(B4_ARMS)` case, and the old-set tests run unedited.

### S7. The #845 recall stamp "which arm they came from" cannot be computed

- **Evidence.** `tools/narrative.py::ExamineTool.execute` keeps recalled memory ids in a local `recalled` dict. Its
  output is text only, and `actions.jsonl` truncates `output` to 1,000 characters. Recall from earlier in the same
  session also renders as "You recall:", since only the last three percepts are excluded.
- **Fix.** Either ExamineTool returns structured `recalled_ids` with each one's origin session (a src change; it feeds
  Open decision 2), or the stamp is "count of recall lines, origin unknown", and the prereg says so.

### S8. The wall-clock cap is unspecified, and its default binds the Maxim arms

- **Evidence.** The default `--subsim-timeout-s` is 1,800 s. The heartbeat fire had 9 of 60 sims above it. The bare
  arms are shorter, so a binding cap truncates the Maxim arms only. A timeout raises and aborts the fire.
- **Fix.** Freeze `--subsim-timeout-s` at a value that does not bind (for example 3,600 s, at least 2× the measured
  p90) and stamp per-sim duration. A sim within 10 % of the cap is reported.

### S9. Server contention over a 95–190 h fire is not detected

- **Evidence.** `serving_provenance.py` says `concurrent_sims` covers one process only. A second consumer of the
  leader is the documented cause of HTTP 500s.
- **Fix.** Add leader-side evidence for each sim window (a request count from the leader proxy covering the window),
  or state exclusive use of the leader as an operator-attested condition of the fire.

## NIT

- **N1. Arm name.** Arm "B2" collides with stage B2 (#1352, `bare_llm`). Rename it `B_dose2`.
- **N2. Cost projection.** `run_benchmark::_project_pair_cost` counts only the A/B/C arms. Add the new arms, or the
  projection undercounts.
- **N3. C prior on resume.** On `--resume`, a re-run (trial, scenario) group rebuilds the C prior, so one trial's two
  scenarios can use different priors. Accept this and say so, or make the prior a row (S5) and reuse it.
- **N4. Dose check.** Require `session_index` per row: A = 0, B = 1, `B_dose2` = 2, C = 1, priors = 0. This is the
  free structural dose check.
- **N5. Ledger triggers (question 7).**
  - The harness change itself (`scripts/` only) fires no row's trigger by its letter.
  - The src changes this review asks for do:
    - a notes store in `report.py::RESUME_STORES` fires T1-16 (EARNED: "the resume path (simulation/report.py
      RESUME_STORES)"), so a walk is owed;
    - the `--aut-stack` seam edits `orchestrator.py`, which is fenced until 1.3.5;
    - every new env toggle (FearGate, #845 recall) needs its `tests/conftest.py` scrub and a brief-table row;
    - gate-code changes (the new complete rule, B0b's attribution literal, the new identity source) each get the
      adversarial gate-design pass.
- **N6. A `docs/wiring/` entry.** Proposed: "An LLM-path arm is selected at the sim seam, not the factory." The sim
  wraps bio around whatever `create_full_agent` returns: pain executors, FearGate, DefaultNetwork, imagination and the
  W2 manifest, `examine` recall. Factory flags alone therefore do not make an arm, and only an `arm_effective` stamp
  read from the built objects proves which stack ran.
