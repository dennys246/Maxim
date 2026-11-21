# Exp 63c: WIRING lens (design review, 2026-11-09)

Reviewed: `docs/experiments/exp63c_scripted_recall_prereg.md` DRAFT v1, against the code at origin/main `d4bbb360`,
with Exp 63b's prereg and its wiring lens as the base. I did not read the other lenses' files. I re-read every
apparatus claim in the code. Offline probes (no sim, no LLM, scratch `HOME`, `MAXIM_OFFLINE=1`) and reads of 63b's
committed data (`docs/experiments/data/63b_carried_recall/`) back the numbers below. The probe script is
`C:\Users\denny\AppData\Local\Temp\claude\5f0e565c4ebe3bec\296036da-cc5c-4ebb-8a45-b9fe0292b7c8\scratchpad\lens_wiring\probe.py`.

**Verdict: the stimulus seam is right. Fold before building.** A scripted line reaches the shipped addressed path
exactly as 63b's narrator lines did. Same call (`bridge.send_and_wait`), same salience and novelty, same source tag,
same ENTER line, same `publish_arrival`, same trace. There is no DO-NOT-BUILD on the stimulus.

There is one DO-NOT-BUILD, on attribution. **Option S2 cannot be built as written.** In this protocol the narrator and
the AUT share one router, so "the narrator's router made zero calls" never holds. What S2 wants to say is already in
B0's fields, and the judge can derive it without a new literal or a `src/` change (W1).

The engine change is the right kind of seam. Three of its stated mechanisms are wrong, though, and it needs refusals
it does not list (S1–S3). The `63c` key has six sites that fall through silently to the wrong experiment (S4).

---

## DO-NOT-BUILD

### W1. S2 ("sole consumer") rests on a premise the protocol makes false. Its honest content needs neither a script nor a new literal.

**The premise.** S2's stamp needs "the narrator's router made zero calls" over the window. In the 63b/63c argv there is
no `--aut-model`, so `orchestrator.py::_build_aut_router` returns the shared `llm_router` (`aut_router = llm_router`
when `aut_model is None`). The narrator and the AUT are one router object. 63b's committed reports show it:
`language_router_serial == aut_router_serial == 1`, and `llm_calls` holds serial 1 only (phase 1: 25 started, 24
success, 1 exhausted; phase 2: 49/48/1). Every entry is `lane-large`, `identity_source: local_file`, temp 0.3. In a
scripted run the narrator's loop makes no call, but its router is the AUT's router, and it carries every AUT call. So
S2's condition fails on every run. Gated, it ABORTs every attempt. Reworded around "the narrator's loop", it is not
observable: `models/language/call_accounting.py` attributes calls to routers, never to agents or loops (its own
docstring, and #1227).

**The two questions asked.**
- *Does a zero-call router trip `b0_problems`' "not a router in llm_calls"?* No.
  `serving_provenance.py::serving_provenance` keeps every role serial in `llm_calls` even at zero counts (`if serial not
  in role_serials and not any(r["counts"].values()): continue` drops only NON-role idle serials).
  `call_accounting.py::register_router` registers at construction with zero counts. `o19_verdict.py::_b0_router_problems`
  accepts `entries == []` with `success == 0`. The check is moot here, though: no router is idle.
- *Can any non-AUT consumer of the run's server exist in a scripted sim?* Taking each channel in turn:
  - **Narrator planning:** none. `_finish_runner` sets `stop_event` before `run_agentic_loop(orch_agent, …)` is entered
    and before the stall thread starts. Pinned by `tests/unit/test_orchestrator_characterization_golden.py::test_pre_campaign_turns_run_without_a_narrator`
    (`narrator_calls == []`).
  - **On the shared router, AUT cognition by design:** FearAgent (`fear_llm=llm_router`, `llm_for_fear = llm_router`), the
    imagination designers and the scene manifest (orchestrator.py, the `aut_imagination_trigger … llm_router` branch).
    B0 doc §"What the code forced": they "run on the narrator's router today" (#1228).
  - **Counted, other serials:** ExecAgent's private router (module-level accounting, so its counts survive, and its
    serial enters `llm_calls` when it made a call).
  - **After the window:** `sim_teardown.py::run_roundup` calls `analyze_simulation(report, llm_router=llm_router)` after
    `save_run_report` has taken the end snapshot. It is the same server and the same model, outside the accounting, as
    in 63b.
  - **Uncounted bypasses:** `LLMRouter.warmup` → `backend.warmup()` (not through `_complete_text_locked`), and the two the
    accounting docstring names (`simulation_generator.py`, `interactive.py`'s `LLMAgent`), neither reachable on this path.
    Another process on :8100 is excluded only by the quiet-box rule. The served reader sees the model id, never the
    consumers.
  - The research Writer/Reviewer (`research_orchestrator._build_research_llm`) and session-end consolidation are not on
    this path: no `--research`, and `on_session_end_lightweight` in sim mode.

**What S2 wants is already established.** "Every call served in this run came from model X" is B0's union-over-routers
statement. It holds in 63b's committed data today, with a narrator: one serial, one `local_file` identity. The script
adds nothing to it. The draft's justification, "honest only because the stimulus is scripted", inverts the logic: the
claim is about the window's serving, not about who spoke.

**Is S2 buildable without #1227/#1228?** Not as drafted. The new literal also has three readers that require
`"not_established"` today:
- `serving_provenance.py` writes it;
- `o19_verdict.py::b0_problems` and `_evidence_records.py::judge_b0_fields` refuse anything else.

Changing it means a `src/` change, an evidence-gate change for every sim record, and a claim that
`docs/plans/baseline_floors_b0.md` forbids: "no comparison may claim the AUT's model from a B0 record".

**Fold (S2′, buildable now, no `src/` change):**
- Keep `aut_model_attribution: "not_established"`.
- Add a 63c-only C4‴ in `complete_problems`: every entry on every serial in each phase's `llm_calls` has
  `identity_source == "local_file"` and `identity == PROTOCOL["63c"]["model_sha256"]`, a SHA-256 of the rig's GGUF
  pinned before the first marker. A failure is an ABORT.
- Have the prereg state, in one sentence: every LLM call counted in each phase's window was served by the file with that
  SHA-256. Calls are attributed to routers, not to agents, so the AUT's own slot is not established. The uncounted
  channels are the bypasses above, which are not reachable on this path and are same-model.
- Retro-apply it descriptively to 63b's data, which satisfies it.

This meets OD-c4's intent ("stamp the served model") honestly, and it is S3 plus a real check. Offer the owner S1 (full
B0b), S2′, or S3. Do not offer S2.

---

## SHOULD-FIX

### S1. The engine change: `finish_reason="max_turns"` would fabricate a guard event; C1 should ride `completed`

Today the pre-campaign runner ends with `completed`. `_runner_outcome("precampaign_runner", …)` returns `None` unless a
turn recorded an `error`, and `_resolve_finish_reason` then gives `completed`. Stamping `max_turns` would claim
`initiated_by: max_turns_guard` for an event that never happened (`stall_detector.py::_check_max_turns` is the only
writer, and its thread never polls because `stop_event` is already set).

The draft's reason for the relabel is that a short script would pass C1 by construction. That is answered by the
content pin (S2), which fixes the script's length at 8 inside `o19_verdict.py`, a bound file. The usable-outcome sets
already accept `completed`: `_provenance.py::SPAWN_SUCCESS_FINISH_REASONS` and `_evidence_records.py::finish_ok`.

The relabel would also be global unless it is specially scoped. `test_pre_campaign_turns_run_without_a_narrator`
asserts `finish_reason == "completed"`. `research_orchestrator`, `experiment.run_campaign`, `benchmark.py`,
`roy_runner.py` and `curriculum_runner.py` all read the precampaign finish reason through `is_simulation_run_failure`.

**Fold:**
- 63c's C1 reads `finish_reason == "completed"`, `turns == 8`, an empty `llm_finish_context`, and the S2 content pin.
- The engine change does not touch `finish_reason`.
- Typed aborts stay typed: an AUT death is `aut_died` (`_aut_worker`'s except branch), a failed turn is `error`, and a
  process stop is `yielded`.

### S2. "`read_recorded` so `scenario_sha256` covers it" is the wrong mechanism. Pin the content hash instead.

`capture_serving_start` takes `scenario_inputs.mark()` at sim entry (`orchestrator.py`, `capture_start_provenance`). The
CLI would read the file before `start_simulation_mode`, so before the mark. `scenario_inputs.since(mark)` then omits
it. The probe confirmed this: zero entries after loads made before the mark. What covers the script is the other branch:
a list passed in `scenario_objects` has its content hashed into `provenance.scenario_content_sha256`
(`serving_provenance.py::_content_sha256`, `json.dumps(sort_keys=True, default=str)`), and that hash feeds
`scenario_sha256`.

Two consequences:
- **`scenario_sha256` cannot be the pin.** It also hashes `scenario_inputs`, which holds 102 absolute host paths of
  component specs in 63b's reports (`/Users/dennys/RMSrv/…`).
- **The content hash depends on the loader's dict shape,** not on the file's bytes. `experiment.load_campaign_turns` adds
  `at`, `cli_input`, `metadata`. The research loader strips the text and adds `role` and `tag`.

**Fold:**
- `PROTOCOL["63c"]` pins `scenario_content_sha256`, and C4′ for 63c compares the report's value with it.
- A test pins the judge's stdlib copy of `_content_sha256` over the committed script, loaded by the chosen loader, equal
  to the shipped one.
- The prereg also gives the file's own SHA-256 (descriptive).
- Strengthen C5(i) for 63c: each turn's `cue_text == script[turn-1]` exactly, not just `len()`. Both are pipeline
  independent, and the script is known.
- Drop "so `scenario_sha256` covers it" from §Apparatus.

### S3. Front-gate the engine change: one existing loader, and the refusals it needs

Existing infrastructure covers most of it. Use it.
- **A loader exists:** `simulation/experiment.py::load_campaign_turns`. It reads through `read_recorded`, takes the
  `cli_input` key every shipped scenario uses (279 of 279 percept lines in `scenarios/`), and returns exactly the
  `pre_campaign_turns` shape. Use it, and do not write a fourth loader. The script file then uses the existing percept
  YAML format. The loader keeps the defaults `salience 0.8` and `novelty 0.7`, which equal `SendMessageTool`'s implicit
  `send_and_wait` defaults. The script must not set them; the S2 pin enforces that.
- **A flag nearly exists:** `--campaign` ("Campaign YAML(s) for `--sim` research mode") is parsed today and silently
  ignored on the plain goal path (`cli.py`: read only in the research and benchmark branches). Either extend it to the
  goal path or add `--sim-script`. That is the owner's call. If it is a new flag, a `--campaign` on the goal path should
  then refuse rather than be ignored (an existing silent no-op; file it, do not fold it here).
- **Neither alternative path fits:**
  - `--sim research --campaign` (`start_research_mode`) passes no `resume_session` and adds Writer/Reviewer LLM calls.
  - `api.imagine(scenario=…, resume=…)` has no argv for C4. It is also broken (below).

**Refusals the change must carry** (`run_precampaign_turns` enforces none of them):
- `len(turns) != --sim-max-turns`: refuse at parse time. The runner ignores `max_turns`, and the only cap enforcer, the
  stall detector, starts after the runner returns, so a 16-line script under a cap of 8 delivers 16.
- **A goal that selects an arc** (`select_arc_for_goal` not None, or `--dm`): refuse. The generative runner would run
  first, set `stop_event`, and every scripted `send_and_wait` would then break immediately into empty turns
  (`bridge._send_and_wait` breaks on `stop_event`).
- **Interactive ON:** refuse, or require `--interactive false`.
- **A `--sim <file>.yaml` path, or `--research`:** refuse.
- **An empty or whitespace line:** refuse. `take_user_input` treats `""` as unaddressed, so nothing is published.

It is a flag, not an env var, so no conftest scrub is owed. Keep it that way. The caller is the 63c PROTOCOL argv;
grep for it before calling it shipped.

**Out of scope, but found:** `api.imagine(scenario=…)` builds turns from `p.get("text", "")`, while every scenario uses
`cli_input`. The probe showed 0 of 6, 0 of 7 and 0 of 6 non-empty turns on `aversion_learning`, `causal_learning` and
`concept_formation`. So `api.imagine` with a scenario sends only empty turns, silently. The root cause is three
divergent turn loaders (`api.py`, `research_orchestrator._load_campaign_turns`, `experiment.load_campaign_turns`). It is
a separate issue with its own owner question: unifying them changes `api.imagine`'s behaviour. Do not band-aid it inside
63c's engine PR.

### S4. The `63c` key: every site, and the six that fall through silently

The additive pattern from 63b's wiring S6 holds. Today every dispatch is `in ("63", "63b")` or `== "63b"`, so a `63c` key
lands on a wrong branch without error at these sites:

| Site | Falls through to | Effect |
|---|---|---|
| `o19_verdict.py::complete_problems` C4′ (`in ("63","63b")`) | skipped | memory strategy never checked |
| `complete_problems` C4″ (`== "63b"`) | skipped | B0 checks not restated: the 63b strand-the-campaign bug returns |
| `judge` C5 (`in ("63","63b")`) | skipped | an attempt is complete without C5 |
| `o19_rerun.py::exp63_attempt_problems` (`== "63b"` else) | **`exp63_c5_problems`, Exp 63's old judge** | wrong instrument |
| `o19_rerun.py::_run_phases` (`in ("63","63b")`) | skipped | the harness's completeness disagrees with the verdict |
| `o19_rerun.py::mock_phase` (`== "63b"` / `== "63"` / else) | **`_mock_stores_and_lines`, Exp 10's mock**, with `finish_reason: "max_turns"` hard-coded | the mock tests nothing |

These sites fail closed: `judge`'s gate dispatch and `_gate_passes` raise "no gates". `prepare`'s
`check_memory_strategy` and `smoke` (`!= "63b"` refuses) are also gated.

**Every site, in full:**
- `o19_verdict.py`:
  - `PROTOCOL["63c"]` (experiment, scope `63c_scripted_recall`, kind `exp63c_verdict`, prereg, phases with the new
    argv, and the new pins `scenario_content_sha256` and `model_sha256`);
  - `EXPERIMENTS` (keep the literal `frozenset(` anchor that `test_lint_evidence_gate.py` edits by);
  - `experiment_of`'s docstring;
  - `protocol_problems`' message;
  - `required_files`;
  - `complete_problems` C1 (S1), C4 (`phase_argv` from PROTOCOL: automatic), C4′ (plus the S2 pin), C4″, C4‴ (W1);
  - `judge` (C5 and gates);
  - `_gate_passes`;
  - `LEAK_GATES["63c"]` (63b's tuple; `protocol_problems` refuses a missing entry);
  - `PRE_1079_KEYS` untouched and byte-identical;
  - `EXP63_TURNS` shared;
  - the module docstring and the `EXIT` comment.
- `o19_rerun.py`:
  - `mock_phase` plus a `mock63c_session` (or `mock63b_session` with the 63c script);
  - `_mock_b0_fields` (today 63b only; the mock must now write `completed` and `scenario_content_sha256`);
  - `prepare` (`check_memory_strategy`);
  - `_run_phases` and `exp63_attempt_problems`;
  - `smoke` and `smoke_checks` (relabel "new records per narrator input");
  - `check_argv_parses` (it passes only once the engine change is merged: build order step 3 before step 5, as drafted);
  - `served_reader` and `check_model_config` are generic (nothing owed).
- `_evidence_records.py::O19_KINDS`: add `exp63c_verdict`.
- `docs/experiments/evidence_pass_table.json`: 63b's entry shape, landing before the ledger PR (it is read from the
  merge-base).
- Tests:
  - `test_o19_rerun.py`'s exclusion `experiment_of(k) not in ("63", "63b")` (line 71) must gain `63c`, or 63c falls into
    the Exp 10/09 prereg-text loop;
  - the "is not 10, 09, 63 or 63b" message test;
  - a `test_exp63c_protocol_is_the_preregs`.
- **Tag ruleset:** `o19-markers` (24345734) includes `refs/tags/o19/**` (63b wiring, "Markers"), which covers
  `refs/tags/o19/63c/`. `ruleset_problems` re-reads it at preflight, so nothing is owed beyond preflight.

**Fold:** route every dispatch through one `is_carried_recall(exp)` plus one `carried_recall_variant(exp)` table, as 63b's
S6 suggested. Then add a test that, for each key in `PROTOCOL`, a mock attempt reaches its own C5 and its own gates: the
deletion probe is "drop 63c from one site → that test fails".

### S5. 63c inherits 63b's gates by reference, but neither the bound files nor the prereg lint reach 63b's prereg

The draft says, "Everything 63b states … stands here unless this file changes it." Two mechanisms do not reach 63b's
text:
- `check_bound` binds only `PROTOCOL["63c"]["prereg"]` plus the two scripts (`BOUND_FILES`).
- `lint_prereg_precedes_data.py::parent_chain("63c")` is `["63"]`, not `["63b", "63"]`. The probe confirmed it:
  `parent_token` strips the letter suffix.

So an edit to 63b's prereg after 63c's data is checked by neither. 63b's prereg is frozen, so the risk is small, but
63c's gates would be defined by a file its own verdict does not bind.

**Fold:** make 63c's prereg self-contained for every gate it reads (restate G1–G5, I1, R2′ and the verdict table, short
but complete), or add 63b's prereg to 63c's bound set in `bound_blobs`. The first is cleaner.

### S6. The power replay's stubbed AUT is not calibrated to the real AUT's records

The go rule decides whether the rig is spent, so the replay's fidelity matters. 63b's committed stores show what the real
AUT on this model and goal writes:
- about one action record per turn (phase 2: 16 heard-line records and 15 action records, mostly `sense_tools` and
  `sense_presence`, plus `base_humanoid_listen`/`_move` and one `respond`);
- outcomes that hold capability descriptions ("Pick up an object…", "open, drink, pull…", "nearby") and LLM
  `decision.reasoning` text, both of which `content_tokens` reads;
- `context.active_goal` on every record.

Policy (a), "outcomes holding no planted word", is not that. `mock63b_session`'s fixed `_MOCK63B_REPLY` is not either.

**Fold:**
- Add a policy (d), "63b-real": replay 63b's committed phase-1 and phase-2 action records, verbatim per turn, as the
  competitors.
- Apply every policy in phase 1 too. In 63c the AUT's reply to phase-1 line 8 ("Describe your plan in one sentence")
  lands in full, and it becomes a carried record that may restate the planted facts (see S7).
- Report per policy whether the decisive turn is won by the planted fact's own record or by a carried echo. That is the
  draft's descriptive "right memory" figure, so the forecast and the result read the same quantity.

### S7. Pacing and turn 8: state what 63b measured, and make the smoke check it

"As the narrator path does" is true of the per-turn wait (one bridge, 120 s, 3 s settle). The time between turns is
not the same. 63b's committed logs:

| | Phase 1 | Phase 2 |
|---|---|---|
| ENTER-to-ENTER gaps | 30.0–39.2 s | 35.8–91.6 s |
| AUT's one action after ENTER | 21.6–24.9 s | 21.6–25.4 s |
| AUT idle before the next ENTER (the narrator planning) | 8.1–14.3 s | 10.4–67.7 s |
| AUT `sim_deliberation`/`sim_thought` lines more than 3 s after a turn's last action | 0 | 0 |

In 63c the next line follows about 3 s after settle. The zero count suggests the shorter gap will not interleave turns,
but it is one model on one rig.

Turn 8 also differs more than the draft says. In 63b the max-turns guard cut turn 8: phase 1's turn 8 has one action 1.0 s
after ENTER, and phase 2's turn 8 has none. In 63c turn 8 runs to settle, so the phase-1 store gains a full reply to "Describe
your plan", which is carried into phase 2 (S6).

**Fold:**
- The smoke reports, per turn: idle before the next ENTER, `timed_out`, AUT traces in the window, and late ids (C5's
  late-arrival case).
- The prereg states 63b's numbers as above.
- The turn-8 sentence also says the phase-1 store gains a reply 63b never had.

---

## NIT

- §Apparatus: "the same one a person typing into a sim makes" is false. A person's free text in a sim goes through
  `_process_line`'s else branch: `bridge.percept_source.inject_cli(line, salience=0.9, novelty=0.8)` when interactive or
  paused, otherwise `orchestrator_source.inject_cli("USER GUIDANCE: …")` to the narrator. That path has no
  `send_and_wait`, no ENTER line, no `turn_count`, and no anxiety hook. Talk/CLI mode reaches `take_user_input` through
  `pending_cli_input`. All of them meet at `take_user_input`'s `addressed` and `publish_arrival(source="cli")`, so the
  claim's "addressed path" holds. Say "the same call 63b's narrator made; a person's input meets it at `take_user_input`".
  The claim's parenthesis is already right.
- Turn counting is cleaner in 63c. `bridge.inject_pain` increments `_turn_count` without an ENTER line, so in 63b a
  narrator `inject_pain` would have counted toward `turns` and the max-turns guard (63b's runs had 8 ENTER lines for 8
  turns, so it did not happen). With no narrator tools, `turns ==` the number of ENTER lines. Worth a line.
- `run_precampaign_turns` runs on the main thread before the `try/except KeyboardInterrupt` around the narrator loop, and
  it checks neither `stop_event` nor `yield_requested()` between turns. A dead AUT is still typed (`aut_died`), and later
  turns are delivered into a stopped bridge. A process stop mid-script shows up as `yielded` only at teardown. Fine for a
  non-interactive rig run; worth one line in the engine issue.
- `introspector.full_analysis(seed_keywords=[])` runs after turn 8 while the AUT loop is still live. It does no
  hippocampus recall with empty keywords, but `concept_query` reads `hub.atl.recall`. No gate reads ATL. It writes
  `sim_reports/campaign_analysis_*.json`, a file that `_provenance.list_sessions` (directories only) ignores.
- `--sim-run-full-turns` only sets the narrator's `min_finish_turns`, and it is inert with no narrator. Keep it for argv
  parity with 63b, or drop it. Either way the prereg should say which.
- The script file's path in argv should be repo-relative (the harness `Popen`s from its own cwd). `scenario_inputs`
  records a realpath only if the file is read after the mark, which this design does not do (S2).

---

## What I verified (file::symbol)

**Stimulus equivalence (Q1).** Every item matches 63b, except the pacing and turn 8 (S7).
- **The call.** `campaign_runner.py::run_precampaign_turns` calls `bridge.send_and_wait(text, salience=turn.get("salience",
  0.8), novelty=turn.get("novelty", 0.7))`. `tools.py::SendMessageTool.execute` calls `self._bridge.send_and_wait(text)`
  with the signature defaults 0.8/0.7 (probe: `SimulationBridge.send_and_wait` signature). Identical when the script sets
  neither key.
- **One method body for both callers** (`bridge.py::SimulationBridge._send_and_wait`):
  - the ENTER line `sim_log("EXEC", f"Bridge.send_and_wait ENTER turn={…} text_len={len(text)}")`, so C5(i)'s `text_len`
    is `len` of the unstripped text, as is `cue_text = percept_text(observation)` (`agent_loop.py`, `_percept_text_for_cycle`);
  - `percept_anxiety_hook` (wired on the bridge at `orchestrator.py`'s pain-layer block, before any runner);
  - `percept_source.inject_cli`;
  - `_turn_count += 1`.
- **The percept.** `conversational_source.py::ConversationalSource.inject_cli` gives `source="cli"`,
  `percept.cli_input = text`, and `scenario_tag user_input_<step>`.
- **The addressed path.** `loop_perception.py::take_user_input` takes `"cli"` ∈ `ADDRESSED_SOURCES` (probe:
  `['cli','speech','transcript','voice']`) → `publish_arrival(state, text, source="cli")` → `own_capture_id`.
- **The trace.** The `enrichment_trace` carries `own_ids`, `cue_text` and `recall_failed`
  (`bio_enrichment.py::BioEnrichmentPipeline.enrich` / `_query_hippocampus_traced`), and its agent id is
  `_aut_agent_id = "sim_aut"` (orchestrator.py).
- **The judge.** `turn_windows` and `_enter_text_lens` filter on `e == "sim_exec"` and the regex only, so the ENTER line's
  agent context (main thread here, `sim_orchestrator` in 63b) does not matter.

**No narrator, no nudges (Q2).**
- `orchestrator.py` runs `if pre_campaign_turns: … _finish_runner(…)` before the stdin reader, the `StallDetector`
  thread and `run_agentic_loop(orch_agent, …)`.
- `stall_detector.py::StallDetector.run` breaks on a set `stop_event` before its first `_poll`.
- The kickoff or resume prompt goes to `orchestrator_source` after the stop, and nothing plans on it.
- Pinned by the golden precampaign arm (`narrator_calls == []`).
- Nudges and the diversity checkpoint only ever go to `orchestrator_source`, never to the AUT, in either design.

**Resume (Q2).**
- `_restore_aut_from_session` runs before the AUT thread starts and before any runner.
- `_load_resume_context_at` runs after the runner, but before `end_run`, and updates `resume_record` in place.
- `report.py::resume_stamp` needs `context_loaded` and `state_dir == context_dir`: both are set. A failed context load
  fails C2: an ABORT, fail-closed.
- Mode, router, bio stack and `aut_bio_enrichment_pipeline` do not depend on `pre_campaign_turns`. The only other uses
  are the provenance capture and the tail.

**finish_reason and its consumers (Q3).**
- Producers: `_runner_outcome`, `_finish_runner`, `sim_teardown.py::_resolve_finish_reason`, `_check_max_turns`,
  `_aut_worker`.
- Consumers outside the change:
  - `sim_types.py::is_simulation_run_failure` and `simulation_exit_code`, which `cli.py`'s exit code uses;
  - `_provenance.py::spawn_evidence` / `SPAWN_SUCCESS_FINISH_REASONS`;
  - `_evidence_records.py::finish_ok`;
  - `research_orchestrator`, `experiment.run_campaign`, `benchmark.py`, `roy_runner.py`, `curriculum_runner.py`;
  - the golden test.
- `scenario_sha256` is produced only by `serving_provenance.py` and consumed by `b0_problems`, `judge_b0_fields` and
  `_provenance.py`'s projection.
- `pre_campaign_turns` producers: `api.imagine`, `research_orchestrator`, `experiment.run_campaign`, the `/new` carry
  settings.

**Attribution (Q5).** `serving_provenance.py::serving_provenance`, `call_accounting.py` (`register_router`, `delta`,
`window_problems`), `o19_verdict.py::b0_problems` / `_b0_router_problems`, `_evidence_records.py` (the
`aut_model_attribution` literal), `docs/plans/baseline_floors_b0.md` §B0b, and 63b's two committed `report.json`
provenance blocks (one shared serial).

**Not verified:** the live ruleset (no `gh api` call; 63b's wiring read it on 2026-11-02, and preflight re-reads it); any
live run.
