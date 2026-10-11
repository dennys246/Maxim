# Baseline floors B0 — the provenance fields a floor comparison needs (approach note)

> **Approach note v10, 2026-10-10**, for [baseline_floors.md](baseline_floors.md) Stage B0 (PR #1224). It
> writes into a gate's inputs (the evidence gate reads `report.json` provenance and harness rows), so it got
> an adversarial design pass before code (CLAUDE.md "Faster without lowering the bar" (1)): fifteen passes,
> each folded; the last came back clean (§Review record). **Owner decision 2026-10-10: split.** B0 ships what survived; attributing
> calls to the agent under test (the AUT) is **B0b**, after [#1227](https://github.com/dennys246/Maxim/issues/1227)
> and [#1228](https://github.com/dennys246/Maxim/issues/1228).

## What B0 lets a reader establish

- Which **models and serving** answered the run's LLM calls (the union over every router in the sim's
  window), per router object, with **known call temperatures** and a **completeness** check.
- Which **scenario inputs**, **global seed** and **session index** the run had.
- That the record's schema is the one its executed code writes (no opting out).

**Not established by B0:** which model served the AUT. Every record stamps
`aut_model_attribution: "not_established"`, and no comparison may claim the AUT's model from a B0 record.
In particular, `aut_router_serial`'s entries are the AUT **worker's** router calls, not the AUT's: its
FearAgent, imagination designers and ExecAgent goal proposals run on other routers today (#1228). Also not
established: `top_p`/`top_k`/repeat penalty (server defaults), and whether a sub-sim ran the arm its row
claims (`arm_effective`, which enters with M46 / B4).

## What the code forced (summary of the passes)

- **The unit is the call, per router object.** One router serves calls from several providers
  (`LLMRouter._candidate_providers` fallbacks, `_inject_placement_tail`, `_model_for_tier` downgrades), so
  identity is per (provider, served model) per router. Temperature is a per-call argument (0.3, 0.0 or the
  caller's), so there is no run-level temperature or top_p.
- **The seed is honoured only by the in-process `_LlamaCppBackend`.** An owned local server is spawned
  without `--seed` (`LocalServerSpawner._build_cmd`).
- **Owned servers are reached through `_MaximPeerBackend`**, so the router never resolves a GGUF; a local
  file hash comes from the process's running spawner, and peer routing has no local file at all.
- **Calls cannot be attributed to a role yet.** `RequestContext.agent` is unreliable (#1227). Router objects
  are reliable, but the AUT's FearAgent, imagination designers and scene manifest run on the narrator's
  router today, and its ExecAgent builds a private router with the narrator's profile (#1228); calls made on
  the narrator's router are invisible to any per-router check. Hence the split.

## Fields, sources, and how each could lie

Written by `simulation/report.py::run_provenance` into `report.json`'s `provenance`, and projected into
harness rows by `scripts/_provenance.py::sim_evidence` (`_EVIDENCE_PROVENANCE` and the projection's suffix
list extended for every new key).

| Field | Source | How it could lie | Defence |
|---|---|---|---|
| `provenance_schema` (= 2) | `report.py::PROVENANCE_SCHEMA`, written by `run_provenance` | A record or projection that drops it would skip B0's checks | Required by the executed code (§What the gate checks), never by the record's say-so |
| `llm_calls`: `{<router serial>: {entries: [{provider_key, model, identity, identity_source, quantization, kv_quant, seed, seed_honoured, temp, n}], counts: {started, success, exhausted, budget_reject, raised}}}`; `<role>_router_serial` per worker | A module-level, strongly held accumulator in `router.py`, keyed by a process-unique serial each `LLMRouter` takes at construction (not `id()`), **registered with zero counts at construction** so an absent serial and a zero-call serial never look alike. Written in a `finally` wrapping the whole body of `_complete_text_locked` (the no-providers return precedes today's `try`): each dispatch started gets exactly one terminal outcome. On success `_try_provider` returns the served `(provider_key, model_override or model)` (usage is `None` on Paths A and C); temp is the call's parameter. Reported as the delta between a snapshot at sim start and one at report time, over EVERY serial (role or not): the delta iterates the END snapshot's keys with `start.get(k)` as a zero baseline (a router built mid-sim, e.g. the ExecAgent's, is kept), refuses if a start key is missing from the end (strong refs guarantee end ⊇ start), and drops entries whose delta `n` is 0 (a model that served only before the sim is not in the served set). Entries within a serial are matched on the full recorded tuple `(provider_key, model, quantization, kv_quant, seed, seed_honoured, temp)`, never by list position, with the same zero-baseline and refuse-if-missing rules. A negative delta in any count or any `n` refuses (the accumulator only grows). The accumulator is keyed by that tuple (one entry per tuple); a snapshot with a duplicate tuple refuses, and per serial the entries' `n` must sum to its `success` count, checked on BOTH snapshots (so the delta inherits it); the `finally` leaves the outcome unset unless a path sets it explicitly (a success that returns without the served tuple is caught by `started == sum of outcomes`); a non-finite `temp` is rejected at record time; so entries and counts cannot disagree silently. `identity` and `identity_source` are joined at report time, outside the tuple. `identity`, from the background hash, is joined onto entries at report time, not recorded per call. A test builds a router mid-sim with no role key and asserts its calls are in the delta | A completion outside `_complete_text_locked`; calls from an earlier sim in the same process; a router built outside the report's view or collected before report | Per serial, started == sum of outcomes, else `sampling_complete: false`. The delta excludes earlier sims (`benchmark.py`'s loop) and pre-sim calls (the CLI's auto-curate router). Strong refs keep collected routers' counts. Router bypasses named in the docstring: `simulation_generator.py` and `interactive.py`'s `LLMAgent` continuation call `backend.complete` directly; `research_orchestrator.py` is a separate entry point. Calls `_complete_text` drops before the lock never start |
| `concurrent_sims` | Sims register on start and deregister in a `finally`; any overlap marks every sim involved, including one inside another's window | Two sims in one process share the accumulator | Stamped and refused. A crashed sim's stuck registration makes later sims read concurrent (fail-closed) |
| `identity` / `identity_source` | `local_file`: SHA-256 of `realpath(spawner._model_path)`, only when `llm_server._active_spawner.is_running` and its `base_url` equals the provider's. `server_reported`: `_MaximPeerBackend`'s `/v1/models` served id (covers a reused server accepted on a basename match). `provider_name`: cloud model string | A served id is an operator alias; the model changes mid-run; a hash races a file swap | `server_reported`/`provider_name` are named, so the weak-evidence rule treats them as attested. Captured at start and report; `model_changed_during_run` stamped. The hash runs in a thread from sim start (cost stated), joined with a bounded timeout; timeout, failure, a changed (dev, inode, size, mtime) between spawn and hash end, or a different spawner object or `_model_path` between start and report → `local_file_unhashed`, refused |
| `quantization` | Profile `quantization` (inferred from the filename) | Filename lies | Pinned by `identity` when `local_file` |
| `kv_quant` | The owned spawner's actual spawn args (`--type_k`/`--type_v`); no flag → `f16`; not owned → `None` | Config and spawn args drift (docs/lessons/sim-n-ctx-config-drift.md) | Spawn args only |
| `seed`, `seed_honoured` | The backend's own `backend.cfg.seed` (-1 → `None`); honoured = the backend instance is `_LlamaCppBackend` | A future seed-honouring class reads false | The test enumerates every class `_get_backend_for_provider` can build plus `BACKEND_CLASSES` |
| `<role>_leader_disk_head` (source `leader_reported`) | Peer `/v1/debug/version` (`get_version_info`) | It reports the HEAD on disk at request time (pulled but not restarted reads new), short, no dirty flag, the proxy not the upstream server | Named for what it is; never code identity |
| `scenario_inputs` (`[{path, sha256}]`) + `scenario_sha256` over the goal text and that list | Hooked at the scenario/embodiment/entity YAML loader | A loader the hook misses still yields a valid-looking hash | The list shows what is covered; narrator-generated worlds and `simulation_generator.py` output are outside it, said in the docstring |
| `global_seed` | `utils.seeding.get_global_seed()` (set by `--seed` through `seed_all`) | Not every path consumes it; `--seed` does not set `MAXIM_LLM_SEED`; a Minecraft world seed is different | Stamp only what was set; all three stated |
| `session_index` | 0 only when no `--resume-sim`, no persistent agent adopted (`_adopt_persistent_agent`) and every store started empty; prior report's index + 1 when `resume.resume_loaded`; else `None` | An adopted agent carries prior state into a "fresh" run | `None`, never 0, whenever prior state could exist |
| `aut_model_attribution` | The literal `"not_established"` | — | B0b replaces it |

## What the gate checks (`scripts/_evidence_records.py::judge_sim_fields`)

- **The schema comes from the code that ran.** At gate time `git show <executed_git_hash>:src/maxim/simulation/report.py`
  is AST-parsed for `PROVENANCE_SCHEMA`: a plain top-level single-target `Assign` of an int literal, the last
  top-level one wins, anything else (including `PROVENANCE_SCHEMA: int = 2`) is unparseable and fails. The
  record must equal it, and B0's checks apply when it is >= 2. The anchor is the first commit where
  `git log --first-parent --diff-merges=first-parent --reverse -G'^PROVENANCE_SCHEMA\s*=' <ctx.base> -- src/maxim/simulation/report.py`
  matches (from the gate's base ref, never HEAD), confirmed by AST there. A descendant of the anchor must
  carry a parseable value >= the anchor's (a rename or downgrade fails). The gate fails if it finds no anchor
  or if `git merge-base --is-ancestor` exits 128 (a shallow clone cannot silently disable it).
- Whether call fields are required comes from the `<role>_profile` keys (the existing `_router_n_ctx`
  pattern), never from which keys a record carries. When any is set: each `<role>_router_serial` names a
  serial present in `llm_calls` (registered at construction, so `started == 0` is a real zero, allowed for a
  substrate-primary sim); and **every** serial in `llm_calls`, role or not, has well-formed entries and
  balanced counts. Entries: `identity_source` in {`local_file`, `server_reported`, `provider_name`}
  (`local_file_unhashed` refused), `identity` 64-hex when `local_file`, `seed_honoured` a bool.
  `sampling_complete` true; `model_changed_during_run` false; `concurrent_sims` false.
- `scenario_sha256` 64-hex; `session_index` an int or `None`; `aut_model_attribution == "not_established"`.
- Not here: cross-arm equality (M46); whether attested identity suffices (the weak-evidence rule, at a
  status change).
- Mock writers (`scripts/o19_rerun.py`, `scripts/benchmark_cross_session.py` mock) are already refused as
  mock; they gain the keys so their shape matches. A harness row is in-process (no router, no call fields)
  only by its `harness_family`, never by missing keys.

## B0b — AUT attribution (after #1227 and #1228)

What the passes established, so B0b does not relearn it:

- Attribute at the **call site**, not after the fact: a role tag bound around each AUT dispatch in a field
  the request dict cannot override (#1227), counted per (role tag, serial) in the same accumulator.
- #1228 wires every AUT-side LLM slot to `aut_router` (FearAgent, the imagination designers, the ExecAgent's
  router; and decides whether the scene manifest and the DM/fixture runners' `llm_router` arguments are
  world generation or AUT cognition). Floor runs need a dedicated AUT router (`aut_router is not llm_router`).
- An AUT model claim needs every AUT-tagged call on the AUT's serial; an AUT-tagged call on any other serial,
  or an untagged call on the AUT's serial, refuses the claim.
- Walking objects for routers cannot work: AUT-side state lives in separate locals, and routers passed as
  arguments are held by nothing.

## As built (deviations from the design, 2026-10-10, after the code review)

- **Why not existing infrastructure.** `runtime/llm_call_registry.py` tracks only calls in flight and drops
  them at their end; `llm_server.register_router`'s weak references lose the counts of collected routers. So
  the accumulator is its own module, `models/language/call_accounting.py`, written at the same site.
- **The served call is built once, by the attempt.** `_try_provider` builds it from its own provider, model,
  backend and temperature (`_served_call_for`), guarded: accounting never changes a dispatch, and a failure
  leaves the window incomplete. The key is `(provider_key, model, backend_kind, endpoint, served_id,
  quantization, seed, seed_honoured, temp)`, recorded at call time so identity can be joined for routers the
  report holds no object for; the endpoint is the configured base URL (no DNS at call time). `quantization`
  is recorded only for the in-process llama backend. The transformers and other backends fall to
  `provider_name`: a model name, attested, even for a local file.
- **The start stamp stays plain JSON.** The capture (call snapshot, scenario mark, spawner) is held in
  `simulation/serving_provenance.py`; the stamp carries an integer `serving_token`.
- **Sims register per thread**, so a sim that dies before its report never taints a later one: a new sim on
  the same thread drops the earlier registration, a dead thread's registration is dropped, and only a live
  sim on another thread marks overlap. A sub-sim run synchronously on its parent's thread replaces the
  parent's registration.
- **The owned server is hashed from its spawn** (`runtime/llm_server.py::set_active_spawner`, used by
  `lane_backends`), so a server spawned inside the window is a fresh spawn, not a model change; a replaced
  server or a changed file is. Spawner state is read through public accessors (`active_spawner()`,
  `active_spawner_digest()`, `LocalServerSpawner.model_path` / `kv_cache_type`, which the spawn command also
  reads). Peer probes go through `LLMRouter.serving_probes()`.
- **Scenario files loaded before the sim count.** Loaders read through `utils/scenario_inputs.py::read_recorded`
  and parse the bytes it returns (the hash covers what was parsed); the window lists each distinct file once;
  scenario data handed over as `pre_campaign_turns` enters `scenario_sha256` by content; a DM campaign carries `source_inputs` and the sim re-records them after
  its mark; the component registry re-records a ref's files on every cache hit.
- **Report time waits for in-flight calls.** Each role router gets up to 30 s (`LLMRouter.wait_idle`) before
  the end snapshot.
- **Completeness is judged on routers active in the window**, balanced in the window and at both snapshots;
  idle non-role routers are left out of `llm_calls`.
- **No anchor on the gate's base counts as pre-B0 unless the clone is shallow** (then the gate fails): inside
  B0's own PR, main has no anchor yet. The `-G` pattern uses `[[:space:]]`; git's regex has no `\s`.
- **The mock report writers are unchanged**: the gate refuses mock records before judging fields.

## Persistence and env

No new env var (`MAXIM_LLM_SEED` exists, with no side effect, so no conftest scrub). `report.json` already
carries `_format_version`; `public_format_freeze.md` freezes the substrate bundles only, so new keys inside
`provenance` are additive.

## Companion edits owed when #1224 is on main

`baseline_floors.md` §B0 still lists a provenance `arm` equal to the row `arm`, `top_p` and a world seed, and
its matching-controls row says "server version (stamped by B0)"; M46 says "Needs B0's `arm` stamp". B0's PR
rebases on main after #1224 merges and corrects them to this note: no arm verification in B0, temperatures
only, `global_seed`, `leader_disk_head` (not code identity), AUT attribution in B0b, and M46's model-parity
check waits for B0b.

## Out of scope

B0b; cross-arm parity and `arm_effective` (M46 / B4); request-context identity (#1227); the dual-LLM wiring
(#1228); the vLLM draft's trusted `served_n_ctx`; hashing a remote GGUF; stamping `simulation_generator.py`'s
model.

## Review record

Eleven adversarial passes, 2026-10-10. v1 → v2: identity per role → per (provider, model) per router;
completeness from the router's own counts; `local_file` only for a running owned spawner; `server_version`
→ `leader_disk_head`; the honoured-seed check by backend instance; schema anchored on the executed commit;
`session_index` 0 needs no adopted agent; `scenario_inputs` listed; `global_seed`; "same sampling" narrowed
to temperatures. v2 → v3: one terminal outcome per dispatch; `local_file_unhashed`; `arm_effective` moved to
M46/B4. v3 → v5: the served provider/model from `_try_provider`; the schema anchor's form, ref and downgrade
rule; `f16` for an owned spawner. v5 → v7: `shared_router` drew a fail-open three rounds running and was
removed; per-agent keying failed on request-context identity (#1227). v7 → v9 (owner decision): attribution
by router object, a strongly held serial-keyed accumulator, a start-to-report delta, concurrent sims refused;
round 8 found the AUT's FearAgent and ExecAgent on the narrator's model (#1228). Round 10: AUT-side calls on
the narrator's router are invisible to any per-router check, so AUT attribution needs the call-site tag.
v10 (owner decision): split; B0 makes no AUT claim; B0b after #1227 and #1228. Round 11 (on v10's plan):
`aut_router_serial`'s entries named as the worker's calls, not the AUT's; every serial judged, not only role
serials; routers registered at construction so absent ≠ zero. Round 12: the delta iterates the end
snapshot with a zero baseline, refuses a vanished start key, drops zero-delta entries; identity joined at
report time. Round 13: entries matched on the full tuple with the same rules; a negative delta refuses. Round 14: one entry per tuple, duplicates refuse, entries' `n` sum to `success`. Round 15: CLEAN; nits folded (the sum check on both snapshots, no default outcome, finite temp).
