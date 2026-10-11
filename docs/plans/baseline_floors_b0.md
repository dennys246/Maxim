# Baseline floors B0 — the provenance fields a floor comparison needs (approach note)

> **Approach note v7, 2026-10-10**, for [baseline_floors.md](baseline_floors.md) Stage B0 (PR #1224). It
> writes into a gate's inputs (the evidence gate reads `report.json` provenance and harness rows), so it
> got an adversarial design pass before code (CLAUDE.md "Faster without lowering the bar" (1)). Six passes folded
> (§Review record).

## What B0 must let a reader establish

That two arms of a floor comparison ran on the **same set of models and serving** (every call in the
process, not attributed to a role), with **known call temperatures**,
and which scenario inputs, global seed and session they were. Whether a sub-sim ran the **arm
configuration its row claims** (`arm_effective`) is NOT B0: it enters with its first consumer (M46 / B4),
since it needs a census of every toggle a harness varies and a lint over the harnesses. Other sampling parameters (`top_p`, `top_k`, repeat penalty) are server defaults and are not
established by B0. B0 stamps and checks well-formedness; cross-arm **equality** is M46's job.

## Revisions to the plan's B0 list, from the code

- **The unit is the call, not the role.** One router serves a role from several providers
  (`LLMRouter._candidate_providers` falls back on backoff, drain and context routing; a budget block routes
  to local; `_inject_placement_tail` appends a cloud-fallback tail; `_model_for_tier` downgrades the model).
  `aut_router` defaults to the same object as `llm_router`, and FearAgent's `generate_json(temperature=0.0)`
  calls go through it. So identity and sampling are stamped per (provider, served model), per router.
- **No per-role attribution.** The only call-level identity is `RequestContext.agent`, and it is not
  reliable: an explicit dict `"agent"` overrides the contextvar the orchestrator binds, ExecAgent always
  sends its class constant, and threads without a `context_scope` carry none. Every per-role scheme the
  design passes tried (per role, `shared_router`, per agent) drew a fail-open from that layer. A floor
  comparison does not need roles: both arms run the same narrator configuration, so equality of the
  process-wide call set establishes the parity. Per-role attribution waits for a fix to request-context
  identity (its own issue).
- **No run-level `temperature`/`top_p`**: temperature is a per-call argument (0.3, 0.0 or the caller's).
- **The seed is honoured only by the in-process `_LlamaCppBackend`.** An owned local server is spawned
  without `--seed` (`LocalServerSpawner._build_cmd`), so it ignores `MAXIM_LLM_SEED`.
- **No provenance `arm`, and `arm_effective` moves to its consumer.** The value of the plan's "provenance arm
  equals row arm" check is proving the sub-sim ran the arm the row claims (the env-not-propagated class).
  That needs the toggles the sim RESOLVED, from each consumer's own reader (config_loader for `sim.*`,
  `nac.py`'s parser, `tool_dispatch`'s, `memory_hub`'s substrate gate, the embodiment id, the fear gate),
  plus a lint that every MAXIM_* var a harness sets is covered. That is M46/B4's instrument, built with it.
- **No local file hash under peer routing**, and an owned local server is reached through
  `_MaximPeerBackend`, so the router never resolves a GGUF; identity comes from the spawner when the process
  owns one.

## Fields, sources, and how each could lie

Written by `simulation/report.py::run_provenance` into `report.json`'s `provenance`, and projected into
harness rows by `scripts/_provenance.py::sim_evidence` (both `_EVIDENCE_PROVENANCE` and the projection's
suffix list are extended for every new key).

| Field | Source | How it could lie | Defence |
|---|---|---|---|
| `provenance_schema` (= 2) | `report.py::PROVENANCE_SCHEMA`, written by `run_provenance` | A record or projection that drops the key would skip B0's checks | Required by commit ancestry (§What the gate checks); the record's own claim never exempts it |
| `llm_calls`: one list per process of `{router_id, provider_key, model, identity, identity_source, quantization, kv_quant, seed, seed_honoured, temp, n}`, deduplicated by router object, plus `llm_call_counts`: `{started, success, exhausted, budget_reject, raised}` over the whole process. **No per-role attribution** (§Revisions) | A router-local counter in `_complete_text_locked`: a `finally` wrapping the WHOLE body (the no-providers return precedes today's `try`) gives each dispatch started exactly one terminal outcome (success, exhausted, budget_reject, raised). On success `_try_provider` returns the served `(provider_key, model_override or model)` with its status (usage is `None` on Paths A and C, and `resp.model` can be `None`); the temperature is the call's parameter. Calls `_complete_text` drops before the lock (disabled, cost ceiling, lock timeout) never start and are not counted, said in the docstring. Not `_note_provider_success`, which runs per provider attempt and returns early when cancelled | A path that completes outside `_complete_text_locked` | Complete = process-wide started == sum of terminal outcomes, else `sampling_complete: false` (stamped). Unattributed threads (no `context_scope`, e.g. `sim.dm`) are counted like any other. Router bypasses named in the docstring: `simulation_generator.py` and `interactive.py`'s `LLMAgent` continuation (`backend.complete` directly), ExecAgent's `ChatLLMAgent` fallback when no router is set |
| `identity` / `identity_source` | `local_file`: SHA-256 of `realpath(spawner._model_path)`, only when `llm_server._active_spawner.is_running` and its `base_url` equals the provider's. `server_reported`: `_MaximPeerBackend`'s `/v1/models` served id (covers the reuse path, where another process's server was accepted on a basename match). `provider_name`: cloud model string | A served id is an operator-chosen alias; the model changes mid-run | `server_reported`/`provider_name` are named, so the weak-evidence rule treats them as attested. Identity is captured at start and at report time; `model_changed_during_run` is stamped, as the code stamp does. Hashing a multi-GB GGUF runs in a thread from sim start (a fresh process per sim makes a cache useless; cost stated), joined with a bounded timeout. Timeout, failure, a changed (dev, inode, size, mtime) between spawn and hash end, or a different spawner object or `_model_path` between start and report (the hot-swap in `lane_backends`) → `identity_source: local_file_unhashed`, which the gate refuses |
| `quantization` | Profile `quantization` (inferred from the filename) | Filename lies | Pinned by `identity` when `local_file` |
| `kv_quant` | The owned spawner's actual spawn args, else `None` | `LaneConfig.kv_quant_mode` and spawn args drift (the n_ctx-drift class, docs/lessons/sim-n-ctx-config-drift.md) | Spawn args only, never the config. An owned spawner with no KV flag stamps `f16`; `None` only when the server is not owned |
| `seed`, `seed_honoured` | The backend's own `backend.cfg.seed` (-1 → `None`), not the router's; honoured = the provider's backend instance is `_LlamaCppBackend` | A future class that honours seeds reads false | The test enumerates every class `_get_backend_for_provider` can build plus `BACKEND_CLASSES` |
| `<role>_leader_disk_head` (source `leader_reported`) | Peer `/v1/debug/version` (`get_version_info`) | It runs `git rev-parse --short HEAD` on request: the HEAD on disk (a leader pulled but not restarted reads new), short, no dirty flag, and the proxy rather than the upstream server | Named for what it is. It never stands in for code identity |
| `scenario_inputs`: list of `{path, sha256}`, plus `scenario_sha256` over the goal text and that list | Hooked at the scenario/embodiment/entity YAML loader | A loader the hook misses still yields a valid-looking hash | The list shows what is covered. Narrator-generated worlds and `simulation_generator.py` (whose `LLMAgent` calls `backend.complete` directly, bypassing the router; its model is not stamped) are outside it, said in the docstring |
| `global_seed` | `utils.seeding.get_global_seed()` (set by `--seed` through `seed_all`) | Not every path consumes it; `--seed` does not set `MAXIM_LLM_SEED`; a Minecraft world seed is a different thing | Stamp only what was set; all three stated |
| `session_index` | 0 only when no `--resume-sim`, no persistent agent adopted (`_adopt_persistent_agent`) and every store started empty; prior report's index + 1 when `resume.resume_loaded`; else `None` | An adopted agent carries prior state into a "fresh" run | `None`, never 0, whenever prior state could exist |

## What the gate checks (`scripts/_evidence_records.py::judge_sim_fields`)

- The schema is required by the code that ran, not the record's say-so, with no pin PR: at gate time
  `git show <executed_git_hash>:src/maxim/simulation/report.py` is AST-parsed for the literal
  `PROVENANCE_SCHEMA` (a plain top-level single-target `Assign` of an int literal; the last top-level one
  wins; anything else, including `PROVENANCE_SCHEMA: int = 2`, is unparseable and fails); the record must equal it, a descendant of
  the anchor must carry a value >= the anchor's (a downgrade fails), and B0's checks apply when it is >= 2. A rename or move
  cannot read as pre-B0: the anchor is the first first-parent commit on main where
  `git log --first-parent --reverse -G'^PROVENANCE_SCHEMA\s*=' <ctx.base> -- src/maxim/simulation/report.py`
  matches (run from the gate's base ref, never HEAD, which is a merge ref on a pull request),
  confirmed by AST there, and any executed commit descending from it without a parseable constant fails.
  Gate code that knows B0 exists fails if it finds no anchor or if `git merge-base --is-ancestor` exits
  128 (ancestry undecidable) (a shallow clone
  cannot silently disable it); the `git log` pins `--diff-merges=first-parent`.
- Whether call fields are required comes from the `<role>_profile` keys (the existing `_router_n_ctx`
  pattern), never from which call keys a record happens to carry. When any `<role>_profile` is set: the process counts balance (`sampling_complete` true), and `llm_calls` is
  non-empty unless `started == 0` (a substrate-primary sim) of well-formed entries, `identity_source` in
  {`local_file`, `server_reported`, `provider_name`} (`local_file_unhashed` refused), `identity` 64-hex when
  `local_file`, `seed_honoured` a bool, `sampling_complete` true.
  `model_changed_during_run` false.
- `scenario_sha256` 64-hex; `session_index` an int or `None`.
- Not here: cross-arm equality and the arm-table comparison (M46); whether attested identity suffices
  (the weak-evidence rule, at a status change).
- Mock writers (`scripts/o19_rerun.py`, `scripts/benchmark_cross_session.py` mock) are already refused as
  mock; they gain the keys so their shape matches.
- A row is in-process (no router, so no call fields) only by the gate's harness spec (`harness_family`),
  never by missing keys.

## Persistence and env

No new env var (`MAXIM_LLM_SEED` exists, with no side effect, so no conftest scrub). `report.json` already
carries `_format_version`; `public_format_freeze.md` freezes the substrate bundles only, so new keys inside
`provenance` are additive.

## Companion edits owed when #1224 is on main

`baseline_floors.md` §B0 still lists a provenance `arm` equal to the row `arm`, `top_p` and a world seed,
and its matching-controls row says "server version (stamped by B0)"; M46 says "Needs B0's `arm` stamp". B0's
PR rebases on main after #1224 merges and corrects all three to this note (no arm verification in B0;
temperatures only; `global_seed`; `leader_disk_head`, which is not code identity).

## Out of scope

Cross-arm parity and `arm_effective` (M46 / B4); the vLLM draft's trusted `served_n_ctx`; hashing a remote GGUF; stamping
`simulation_generator.py`'s model (it bypasses the router: its own issue).

## Review record

v1 → v2 (adversarial pass, 2026-10-10): identity per role → per (provider, model) per router (A);
completeness from `MetricsRegistry` (lane-keyed, leader-side, empty in a sim) → the router's own counts
(B); `local_file` only for a running owned spawner, else `server_reported`, with a mid-run change stamp
(C); `server_version` → `leader_disk_head` (D); seed table → `_LlamaCppBackend` instance check (E);
schema-2 by record → by commit anchor (F); `session_index` 0 needs no adopted agent (G);
`scenario_inputs` list + generator gap (H); `world_seed` → `global_seed` (I); `arm` dropped →
`arm_effective` (J); "same sampling" narrowed to temperatures (K).

v2 → v3 (delta pass, 2026-10-10): completeness counted per dispatch with one terminal outcome each, on a
router-local counter (B, folded wrongly in v2); required routers from `<role>_profile`, in-process only by
harness spec (new); `local_file` identity must be 64-hex, a failed/raced hash is `local_file_unhashed`
(new); calls keyed by agent (new); schema required by commit ancestry, path-independent (design); f16
stamped for an owned spawner (design); two more router bypasses named and the seed read from the backend
(design). `arm_effective` (new fail-open: missed toggles) moved out of B0 to its consumer, M46 / B4.

v3 → v4 (round 3, 2026-10-10): served provider/model returned by `_try_provider`, temperature from the
parameter, `finally` around the whole body, pre-lock drops named (B, folded wrongly in v3); `shared_router`
derived by the producer and checked to name a present router (new); zero-call routers allowed only when
the counter proves it (fail-closed caveat); schema anchor by AST on the executed commit, `-G` anchor,
fails without one (design); companion edits to the plan and M46 (consistency).

v4 → v5 (round 4, 2026-10-10): a descendant's schema value may not fall below the anchor's (new); a
`shared_router` target must carry its own calls, and both roles' profiles must match (new); the constant's
form, the `git log` ref and multiple assignments specified (nits).

v5 → v6 (round 5, 2026-10-10): `shared_router` drew a fail-open in three straight rounds, so it is removed
rather than patched again: calls and counts are keyed by agent, each role names its agents and is judged
only on them. Multi-target/tuple assignment unparseable; undecidable ancestry fails; `--diff-merges` pinned.

v6 → v7 (round 6, 2026-10-10): per-agent keying drew the same fail-open one layer down (request-context
agent identity is unreliable: dict override, class constants, unscoped threads). Per the divergence rule,
B0 stops attributing calls to roles: one process-wide call set and one process-wide completeness count,
which is what a floor's parity needs. Per-role attribution deferred to a request-context identity issue.
