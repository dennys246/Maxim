# Exp 63c: WIRING lens, delta round 1 (2026-11-09)

Reviewed: `docs/experiments/exp63c_scripted_recall_prereg.md` DRAFT v3 against the code at origin/main `d4bbb360`
(this worktree), with the first round's `wiring.md` and `environment.md` as the delta's base. I re-read every claim the
delta asks about in the code, and did not take the first round's folds on trust. Offline probes ran in WSL with a
scratch `HOME` and `MAXIM_OFFLINE=1`, and they confirmed that `maxim` imported from this worktree's `src/`. I ran no
sim, suite or LLM. Probe scripts: `C:\Users\denny\AppData\Local\Temp\claude\5f0e565c4ebe3bec\296036da-cc5c-4ebb-8a45-b9fe0292b7c8\scratchpad\delta_wiring\`.

**STOP: no further round owed.** Every finding below is a NIT, or a fail-closed specification gap in a class the design
already covers: the judge and harness site list, the served reader's start-up race (#1616), and the content pin.
None of them changes the claim, the stimulus seam or the gates' logic. The SHOULD-FIX items still have to land, in the
freeze PR (prereg text) or in build-order steps 3–4 (code). Their code then gets the three-lens code round, as planned.

**Verdict:** v3 folds the first round's wiring W1, S1–S5 and environment F1, F4 and F6 correctly in substance. S2′ is
buildable with no `src/` change, and every field it needs is already in each phase's `llm_calls`. The engine change
composes with `--resume-sim`. Five places are under-specified, and each would cost an attempt or strand a verdict if
it were built literally as written (S1–S5).

---

## SHOULD-FIX

### S1. S2′'s harness-read clause has a teardown race of the same kind as #1616. Bound the window.

v3 (OD-c4/OD-c5) says: every harness `/v1/models` read "after the first OK one" names the model. Today's C4
(`o19_verdict.py::complete_problems`, the `served_model` block) drops reads whose `served` is `None`. v3 makes a
failed read after the first OK one an incomplete attempt.

**The race.** `o19_rerun.py`'s phase loop reads every `SERVED_EVERY_S = 30.0` s for as long as `proc.poll() is None`.
The sim's owned server is stopped at `atexit` (`runtime/local_server_spawner.py`, `atexit.register(self.stop)`), which
runs while the process is still alive. A read that lands between the server's stop and the process's exit therefore
gets a connection error, after an OK read. That error is the shutdown, not a different model. In 63b's rows it did
not happen: the last reads came 10–20 s before `end_ts`. The chance is roughly (server-stop time) / 30 s per phase.
That is small, but the fail is false, and it falls on a campaign the release is waiting on.

**Fold:** judge the reads from the first OK read to the **last** OK read, inclusive. Every read in that span must have
a non-null `served` and `match is True`. A failed read after the last OK one is reported and not judged. Say in the
prereg that this span excludes the start-up race (#1616) and the shutdown race alike. A mid-run failure still lies
inside the span, so it still fails.

### S2. Say which fields S2′ compares. The SHA-256 attaches per server, not per call.

The check can be built from the report (`provenance.llm_calls[serial].entries[]`, and the same block in each row's
`sims[0]`). An entry carries `provider_key`, `model` ("mistral-7b", the profile name), `backend_kind`, `endpoint`,
`served_id`, `identity` and `identity_source` (verified on 63b's `20261107_102003/report.json`). Two of these do not
mean what "every call entry names the PROTOCOL model" suggests:
- **`identity`** is not a per-call value. `serving_provenance.py::_owned_identity` hashes the owned spawner's model
  file once (`utils/file_digest.py::BackgroundDigest`, the full SHA-256 of the realpath). `_join_identity` then copies
  that hash onto every `peer` entry whose endpoint root equals the owned server's root. The per-entry statement is
  therefore "this call went to the owned server, and that server was spawned from a file with SHA-256 X, unchanged"
  (`model_changed_during_run`, which `b0_problems` already gates).
- **`served_id`** is the peer backend's id, cached at first use (`router.py`, the ServedCall builder:
  `str(backend.served_model or "")`). It is not read per call, and it is `""` when discovery failed. In 63b it is the
  fresh-home path `/tmp/o19-home-*/models/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf`.

**Fold:** C4‴ for 63c requires, on every entry of every serial in each phase:
- `backend_kind == "peer"`;
- `identity_source == "local_file"`;
- `identity == PROTOCOL["63c"]["model_sha256"]`;
- `basename(served_id) == MODEL_GGUF`, with an empty `served_id` failing (fail-closed; name it in the prereg).

It also requires `sum(n) == success` for each serial, which `_b0_router_problems` already checks. The comparator must
be a stdlib restatement in the judge, never `_served_model_matches`, because the judge imports no `maxim`. The pin's
value can be taken now from 63b's committed `identity`, `3e0039fd…ccf9`. Preflight re-hashes the realpath of
`check_model_config(...)["model_path"]`, which the harness reaches through the fresh home's `models` symlink. The
prereg sentence should then read: "every counted call went to the owned server, spawned from the file with this
SHA-256 and unchanged during the run; the cached served id names that file." Calling it a C4-class completeness check
is right: C4 is "the pre-registered apparatus", and a mismatch makes the attempt incomplete. Also say which channels
fall **inside** the accounting: the scene manifest (`orchestrator.py`, `generate_scene_manifest`, after
`capture_start_provenance`) and the designer calls on `llm_router`. The channels outside it are the ones v3 names:
`backend.warmup` (it bypasses `_complete_text_locked`), the roundup (`sim_teardown.py::run_roundup`, after
`save_run_report`'s end snapshot) and another process on :8100. The harness reads span the roundup, but by served
name only.

### S3. Binding 63b's prereg through `check_bound` also needs the evidence gate. Do not touch the lint's parent chain.

**The mechanism is right; the parent chain is not.** `lint_prereg_precedes_data.py` answers one question: did a
governing prereg reach `main` before the first record? It computes that through `parent_token` / `parent_chain` /
`_Index.governing`, which time when the prereg was ADDED (`--diff-filter=A`) and when its PRE-DATA amendments landed.
It does not enforce immutability. Making `63c`'s chain `["63b", "63"]` would change `parent_token`'s rule for every
lettered token, so do not do it. Immutability after the data is `check_bound`'s job, so binding 63b's prereg by hash
is the right tool.

**The missed consumer.** `_evidence_records.py::_rejudge_with` requires
`set(bound_files) == {O19_JUDGE, O19_RERUN, protocol[exp]["prereg"]}` **exactly**. If `bound_blobs` adds 63b's
prereg, every 63c verdict stamps four bound paths. The evidence gate then refuses it with "bound_files names …, not
exactly …". That is fail-closed, but it would surface only at the ledger PR, after the data, and it would need a
post-data edit to the gate.

**Fold:**
- Give `PROTOCOL["63c"]` a key, e.g. `"base_preregs": ("docs/experiments/exp63b_carried_recall_prereg.md",)`.
- `o19_verdict.py::bound_blobs` iterates `(prereg, *base_preregs, *BOUND_FILES)`.
- `_rejudge_with`'s `want` adds `protocol[exp].get("base_preregs", ())`, read from the judge's own table, as `prereg`
  already is. Then the 63 and 63b records keep their three-path set.
- `protocol_problems` validates the key: each path exists, and it is never the campaign's own prereg.
- Add both sites to §Apparatus "Judge sites", and add a test that a 63c mock verdict passes `_rejudge_with`'s
  bound-files check.

### S4. Pins and lines: the judge cannot read the script, and there are two scripts

- **Two files, two pins.** `--sim-script` must hold exactly `--sim-max-turns` lines, so phase 1 and phase 2 are
  separate files. `load_campaign_turns` has no phase selector. §Apparatus says "the script's
  `scenario_content_sha256`", singular. It should be one pin per phase, `PROTOCOL["63c"]["phases"][i]`, compared by C1
  per phase.
- **The judge is stdlib-only** (`o19_verdict.py` imports no `yaml` and no `maxim`), so it cannot load the YAML for
  "each turn's `cue_text` equals its script line". **Fold:** `PROTOCOL["63c"]` holds the 16 lines as literals
  (covered by `verdict_source_sha256`). The judge compares `cue_text` and the ENTER `text_len` with them. A unit test
  pins that `experiment.load_campaign_turns(file)`'s texts equal the literals, and that the shipped
  `serving_provenance._content_sha256([turns])` equals each phase's pin. v3 dropped the first round's test sentence
  (wiring S2); restore it.
- **What the pin hashes** (probe): `orchestrator.py` passes `scenario_objects=(dm_campaign, pre_campaign_turns)` to
  `capture_start_provenance`. So on this path the pin is `sha256(json.dumps([turns], sort_keys=True, default=str))`
  over the loader's dicts (`at`, `cli_input`, `metadata`, `novelty`, `phase`, `salience`, `text`). Two loads give the
  same hash, and so do two copies of the file at different paths: it is path-free and stable. `pre_campaign_turns` is
  never mutated between the CLI and that capture. YAML block scalars keep their trailing `"\n"`, so the literals must
  be written exactly as loaded.
- **Harness-side, before the marker:** `phase_argv` puts the script path in the phase's `extra`, and the sim
  `Popen` has **no `cwd=`**, so it inherits the harness's cwd. A repo-relative path then resolves only when the
  operator runs from the repo root. A missing or edited file fails phase 1 after the marker, which costs an attempt.
  `check_argv_parses` only parses argv. **Fold:** `Popen(..., cwd=REPO_ROOT)` for 63c, or resolve the path against
  `REPO_ROOT`. Preflight loads each script with `load_campaign_turns` and compares its content hash with the pin
  before the marker.

### S5. The refusal "an empty line" cannot be built on the named loader as written

`experiment.py::load_campaign_turns` silently skips a percept whose `cli_input` is empty (`if not text: continue`),
along with any percept that uses another key, such as `text:`. It keeps whitespace-only lines (probe: 6 percepts gave 3
turns, `"   "` kept). So the engine cannot see an empty line after loading. A dropped line surfaces only as a count
mismatch, or as a content-pin mismatch.

**Fold:** the engine refuses whitespace-only texts after the load, plus the length refusal (len ≠ `--sim-max-turns`,
with the flag required explicitly when `--sim-script` is given, rather than defaulting to 50 in `cli.py::_sim_turn_caps`).
The prereg should say that a dropped empty line is caught by the length refusal and the pin, not by a separate check.
Changing the loader to raise would change `experiment.run_campaign`'s behaviour. That is out of scope.

---

## NIT

- **The `--dm` reason is wrong on this path.** In `cli.py` the goal-string branch (`if _is_goal_string and not
  _is_legacy_agent:`) `sys.exit`s on both of its arms, before the later `if _wants_dm and _is_goal_string:` block. So
  `--dm` with a goal string is silently ignored today: it neither selects the generative runner nor stops the bridge.
  Refusing it is still right (fail-closed). Reword the reason, or file the dead branch as an existing silent no-op.
- **Make the refusals a whitelist.** `--sim-script` should be accepted only on the plain goal-string orchestrator
  call, and refused everywhere else: the `--sim agent` legacy branch, `benchmark`, research, a YAML `--sim`, `--dm`,
  and `--continuous`. The arc check also needs care: `select_arc_for_goal` sits inside an `except Exception: pass` in
  `cli.py`. Under `--sim-script`, an arc probe that raises must refuse rather than read as "no arc". The goal's arc is
  `None` (probe).
- **C1 is not an `experiment_of` site today.** `complete_problems`' `finish_reason != "max_turns"` check is
  unconditional, so 63c's `completed` needs a new dispatch there. Add it to §Apparatus "Judge sites" next to
  `mock_phase`'s hard-coded `"max_turns"` (`o19_rerun.py::mock_phase`). Dropping the first round's "empty
  `llm_finish_context`" clause is fine: under `completed`, a finish-context status cannot exist
  (`sim_teardown.py::_resolve_finish_reason`).
- **#1616 and S2′ (item e).** Either of #1616's options composes with S1's span. Option A (wait for the port) has to
  wait inside the existing poll loop, under its `deadline` and `proc.poll()`, so that a sim whose server never listens
  still times out. That case gives zero reads and fails C4 "never read", fail-closed. Option B (`not_ready`) must apply
  only before the first OK read. The fix is in `o19_rerun.py`, a `BOUND_FILES` member. That is fine for a new campaign,
  and the evidence gate re-judges 63/63b with their bound blobs (`bound_judge`) plus the HEAD judge (`o19_verdict.py`
  only), so their committed verdicts are unaffected. Landing it before the smoke means the smoke exercises it, which is
  good. It needs no `changelog.d` fragment (`scripts/`, not `src/`).
- **The status line still reads "DRAFT v2"** under a v3 title.
- **Smoke echo channels: add the cue path itself.** v3 lists NAc, ATL and goal-path fills, but the main way a fact
  "surfaces before its cue" (policy pX) is the cue path on turns 1, 3 and 6. A phase-1 fact's own-capture id can
  appear in `enrichment_trace.memory_ids` with `memory_paths == "cue"`. It is observable already, so name it.

---

## (d) Smoke checks: each one's source, with no `src/` instrumentation needed

All of these were read from 63b's committed `run_log.jsonl.gz` and `actions.jsonl.gz`:

| Check | Source today |
|---|---|
| 8 ENTER lines, `text_len` | `sim_exec` "Bridge.send_and_wait ENTER turn=… text_len=…" (`bridge.py::_send_and_wait`) |
| `cue_text` = line | `enrichment_trace.cue_text` (`bio_enrichment.py::BioEnrichmentPipeline.enrich`) |
| AUT action inside its window | `actions.jsonl` `timestamp` against the ENTER `t` |
| No call in flight at ENTER | `peer_backend_call` (`t`, `latency_ms`; DEBUG level, captured by the run log in 63b), interval `[t − latency, t]`. It carries no agent, so it reads "no call from any consumer", which is the stronger statement with no narrator. A call that never completes shows only as `dispatch_exhausted` / `http_request_failed`. |
| Per-turn timing | ENTER `t`, action timestamps, the bridge's "timed out" line. `campaign_results`' `sent_at`/`duration_ms` live only in `SimulationResult`, not in `report.json`. |
| Turn-1 `own_ids`, `hippocampus_size` | `enrichment_trace` |
| Zero narrator submits | no `agent_id == "sim_orchestrator"` deliberation/thought events |
| Peak prompt | `peer_backend_call.input_tokens` (the request's `max_tokens` is not logged) |
| NAc "Predictions from past actions" | `sim_enrichment` system `nac`: "N prediction(s): event→outcome (conf)", the same `[:3]` cap as the render (`_log_enrichment_contributions` vs `format_thought_response`). A `plan:<intent>` outcome appears there in full. 63b's NAc store holds such outcomes, e.g. `plan:I'll locate the guard using base_humanoid_listen.:success` (`bridges/planning_bridge.py`). |
| ATL "Related concepts" | `sim_enrichment` system `atl`, the same `[:5]` cap as the render |
| Goal-path fills | `enrichment_trace.memory_paths` / `memory_ids` |
| Imagination, designer | `sim_imagination` phases `cache_hit` / `index_hit` / `gate_rejected` / `design` (`imagination/trigger.py`). Designer LLM time appears only as `peer_backend_call` by time. The manifest's text is not logged (a count only); its phrases appear as the pre-ENTER imagination events. |

The `sim_enrichment` lines carry no turn or trace id. Join them to the `enrichment_trace` emitted just before on the
same agent, by order and `t`. The working-memory channel logs only a count, but it is within-session and not restored
on resume (the persisted AUT stores are atl, ec, hippocampus, nac and scn), so it cannot show a phase-1 fact.

## (b) The engine change against its producers and consumers

- **`--resume-sim` composes.** `start_simulation_mode` takes both. `_restore_aut_from_session` runs before the
  runner. `_load_resume_context_at` runs after it and injects only into `orchestrator_source`. `NEW_RUN_DROPS` already
  drops `pre_campaign_turns` for `/new`.
- **`completed` is accepted downstream:** `_evidence_records.FINISH_OK`, `_provenance.SPAWN_SUCCESS_FINISH_REASONS` and
  `simulation_exit_code` (exit 0, so `returncode == 0`).
- **What breaks** if the change ships without these: C1's unconditional `max_turns`, `mock_phase`'s hard-coded
  `max_turns`, the `smoke` refusal (`!= "63b"`), the six fall-through sites (v3 lists them), and `_rejudge_with` (S3).
  Nothing in `src/` outside the CLI branch reads the new flag. The caller-grep owed is the 63c `PROTOCOL` argv.

## What I verified (file::symbol)

`call_accounting.py::ServedCall`; `router.py` ServedCall builder; `serving_provenance.py::capture_serving_start`,
`_content_sha256`, `_owned_identity`, `_join_identity`; `file_digest.py::BackgroundDigest`; `orchestrator.py`
(`capture_start_provenance(scenario_objects=…)`, the manifest block, the precampaign block, the resume block,
`NEW_RUN_DROPS`); `campaign_runner.py::run_precampaign_turns`; `experiment.py::load_campaign_turns` (probe);
`sim_teardown.py::_resolve_finish_reason`, `end_run` order, `finish_run`; `local_server_spawner.py` atexit;
`cli.py` goal-string branch, `_sim_turn_caps`, the `--dm` branch; `o19_verdict.py::complete_problems`, `b0_problems`,
`_b0_router_problems`, `phase_argv`, `bound_blobs`, `check_bound`, `PROTOCOL`; `o19_rerun.py::served_reader`, the phase
loop, `check_model_config`, `prepare`, `smoke_checks`; `_evidence_records.py::bound_judge`, `rejudge_o19`,
`_rejudge_with`; `lint_prereg_precedes_data.py::parent_token`, `parent_chain`, `_Index.governing`;
`bio_enrichment.py::enrich`, `format_thought_response`, `_log_enrichment_contributions`; 63b's committed reports, rows
and run logs; issue #1616 (OPEN, options unchosen).

**Not verified:** any live run; the server-stop duration behind S1's probability; whether `sim_logger` truncates
long `sim_enrichment` summaries. 63b's longest line, 230 characters, was not cut.
