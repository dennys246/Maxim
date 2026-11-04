# Wire-integrity review: #1315 item 10a (d0729a0f..fb3ae6f3)

Probes ran in a copied tree (`scratchpad/review10a/copy`, no `.git`), `PYTHONPATH=<copy>/src`, with `HOME` and `XDG_CONFIG_HOME` pointed at scratch dirs. No sims, no network, no full suite.
Disclosure: I ran one `git fetch -q origin` in the worktree to check main for drift. It only moves remote-tracking refs. There were no new commits past d0729a0f.

Count: 1 BLOCKER, 4 SHOULD-FIX, 6 NIT.

---

## Contract 1: LLMRouter construction and dispatch

### Producer and consumer map (outside the diff)

LLMRouter construction sites:

| Site | Engaged first? | What happens to a refusal at build |
|---|---|---|
| `runtime/lane_backends.py::LaneBackendManager._build_local_backend` (LLMRouter at about line 1228) | via the caller | `except Exception` → WARNING "Failed to create LLM router", returns None (**swallowed**) |
| `runtime/lane_backends.py::LaneBackendManager._build_remote_backend` (about line 1439) | via the caller | `except Exception` → WARNING, returns None (**swallowed**, verified) |
| `simulation/orchestrator.py` (lines 969 and 1526): `build_primary_router` None → `LLMRouter(load_llm_config())` fallback | cli/api | the fallback raises loudly if the global profile is cloud; if it is local, it **substitutes** the global in-process profile |
| `embodied_runtime/agentic_runtime.py` (line 897): same fallback | cli/api, **but not** `python -m maxim.embodied_runtime.selfy` | same as the row above |
| `agents/exec_agent.py` (207), `peer/drain_routing.py` (89), `simulation/benchmark.py` (965), `simulation/research_orchestrator.py` (106), `cli.py` (1150, 1781) | cli/api | raises (loud) |
| `maxim.create.router` → `build_primary_router` | **no** | swallowed → None → `ConfigurationError` |
| `scripts/exp44/rerun_ablated_offline.py`, `scripts/behavioral_convergence_exp{3,4}_*.py` | **no** | n/a |

LLM and endpoint paths that do not go through `LLMRouter._complete_text`:
- **`LLMRouter.warmup` → `_get_backend` → cached backend `.warmup()`/`.complete()`.** This bypasses the dispatch check. On a router built before engage whose backend is not yet cached, `_get_backend_for_provider` raises `OfflineRefusal` inside the `LLMWarmup` thread. Nothing catches it there, so `_ready_event` is never set and `wait_ready` blocks until its timeout (120 s by default). This is NIT N1.
- **`LLMRouter.serving_probes`** (consumer: `simulation/serving_provenance.py`) calls `/v1/models` on peer backends built before engage. `utils.http` refuses the call, and `peer_backend.get_json` swallows the refusal at DEBUG. Only the socket latch and the http guard stand in the way.
- **Probes (latch-only, no typed refusal):**
  - `lane_backends._validate_remote_urls` → `_MaximPeerBackend.for_url().health_check()` (see S2);
  - `console/server.py::post_probe`;
  - `simulation/roy_runner.py`;
  - `bench/recovery_time.py`;
  - `runtime/llm_server.py`;
  - `doctor/checks.py`.
- **`agents/llm_agent.py::LLMAgent._get_backend`** builds `_LlamaCppBackend` directly (consumer: `simulation/simulation_generator.py`). It is in-process, so it is consistent with V8.
- **`runtime/function_router.py`** only resolves tiers and never dispatches, so it is not a bypass.
- **`advisor/_session.py`** uses its own closed-local backend and engages first, so it is fine.
- **`leader_proxy` / `llm_server`** spawn and forward. Offline, `_auto_spawn_ruled_out` stops the spawn and the latch refuses the rest.

Under the latch, every one of these Python paths is refused at the socket or at the utils.http guard, or is in-process. I found no live network reach through a router path. The findings below are about where a refusal goes, and about who never engages.

### Findings

**S1 SHOULD-FIX: `llm.offline` is not read at a router build, so library and script entry points run online.**
- Where: `maxim/create.py::router`, `maxim/models/language/router.py::LLMRouter.__init__`, `runtime/lane_backends.py::build_primary_router`.
- Plan §6 says "`llm.offline` is read at CLI startup and at the first router or `api` verb build." The diff engages only in `cli.py::main` and seven `api` verbs.
- Scenario: config.json has `llm.offline: true`. A Python program calls `maxim.create.router("claude-sonnet")`, `maxim.create.pool(llm_router=...)`, `LLMRouter(load_llm_config())`, or one of the scripts that builds a router directly. The latch never engages and the cloud lane runs.
- Verified: with `XDG_CONFIG_HOME` holding `{"llm":{"offline":true}}`, `LLMRouter(load_llm_config(profile_override="claude-sonnet"))` built with `backend=anthropic`, and `_offline.is_offline()` stayed False. `maxim.create.router` also left it False.
- Fix: call `engage_offline_if_configured()` at the top of `LLMRouter.__init__`, before `_refuse_offline_lanes`. The router already lazy-imports `maxim.runtime`, so this adds no new layering. Also call it at the top of `build_primary_router`, so the auto-spawn guard sees it. Add a test that `create.router` engages from config.

**S2 SHOULD-FIX: lane builders swallow the router refusal, so a peer or cloud lane is silently dropped or replaced.**
- Where: `runtime/lane_backends.py::LaneBackendManager._build_remote_backend` / `_build_local_backend` (`except Exception` → None), `::_validate_remote_urls`, `simulation/orchestrator.py` and `embodied_runtime/agentic_runtime.py` (the None → `LLMRouter(load_llm_config())` fallback).
- The changelog says "A cloud, peer or loopback-server lane refuses with `OfflineRefusal` at router build". Plan §6 says the CLI refuses at startup unless every lane resolves in-process, "refused by name". Through the shipped build path, three things happen instead:
  1. **Probe-and-drop.** `_validate_remote_urls` probes a peer `remote_url`. `utils.http` refuses the probe, `health_check` turns that into outcome `other`, and the lane's `remote_url` is dropped with the log line "Falling back to local model selection". The lane is rebuilt as an in-process local lane: a silent model substitution. The probe result is also written to `probe_cache` with outcome `other`. With the 60 s TTL, the lane is still dropped for up to a minute after going back online. (Verified: the WARNING and the ledger shows two `utils.http request` refusals.)
  2. **Build refusal swallowed.** With `MAXIM_SKIP_REMOTE_PROBE=1`, or for a cloud placement or tail, `LLMRouter` raises `router[build]`. `_build_remote_backend` catches it, logs "Failed to create remote LLM router", and returns None. (Verified: `LaneBackendManager({"large": LaneConfig(remote_url=...)}).get_backend("large")` → None, with only a WARNING.)
  3. **The caller's fallback.** `api.run` then raises `ConfigurationError("No LLM backend available. Set --language-model or MAXIM_LLM_PROFILE.")`, which hides the real cause. The orchestrator and agentic_runtime instead fall back to `LLMRouter(load_llm_config())`, which either raises or silently runs the global default local profile.
- Network stays fail-closed, so this is not a BLOCKER. It does break the stated contract, and an offline run can report a model the operator did not configure.
- Fix:
  - Re-raise `OfflineRefusal` in the two lane builders, for example with `except OfflineRefusal: raise` ahead of `except Exception`.
  - Under `is_offline()`, have `_validate_remote_urls` skip probing and keep `remote_url`, so the build refuses by name and nothing is cached.
  - Have the None-router fallbacks in the orchestrator and agentic_runtime not run under the latch.
  - Add a startup predicate over the built lanes, as the plan specifies.
  - Pin it with a test that goes through `build_primary_router` and a peer lane, not through `LLMRouter` directly.

---

## Contract 2: provider-type vocabulary

### Map

| Place | Vocabulary |
|---|---|
| `router.py::_provider_type` | `type`, else `backend`, else `cfg.backend`; lower-cased; `-` → `_` |
| `router.py::_provider_is_cloud` | anthropic, claude, openai, openai_compatible, openai_compat |
| `router.py::_get_backend_for_provider` | the LLAMA and TORCH sets, anthropic/claude, openai/gpt, openai_compat*, then `BACKEND_CLASSES` (openai, maxim_peer) |
| `lane_backends.py::BACKEND_CLASSES` / `_classify_backend` | `maxim_peer` (self-hosted), `openai` (cloud) |
| `cloud_dispatch.normalize_providers` | `cfg.backend`, default `llama_cpp` |
| Profiles | only `llama_cpp`, `openai`, `pytorch`, `anthropic` occur; peer lanes are `maxim_peer` |

The in-process set `_LLAMA_TYPES ∪ _TORCH_TYPES` is exactly the two local branches of `_get_backend_for_provider`, and every profile and lane type falls on one side of it.
- `gpt`, `maxim_peer`, `openai_compat` and unknown types are all "not in-process", so they are refused. Fail-closed, consistent.
- `_LlamaCppBackend` ignores `base_url`, so a llama lane cannot be pointed at an endpoint.
- A transformers lane can still fetch: `hf_local_files_only` defaults to False, and so can a llama lane through `ensure_available`. Both are backstopped by the `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` env vars `engage()` sets and by the socket latch. The typed refusals are item 10b's.

### Findings
- **N2 NIT:** `_provider_is_cloud` leaves out `gpt` while `_get_backend_for_provider` accepts it. This is pre-existing and does not affect the offline check, which uses the in-process allow-list. It is worth a single classification table at some point.

---

## Contract 3: process entry points

### Map

| Entry | Engages first? |
|---|---|
| `maxim` console script, `python -m maxim` → `cli.main` | yes. Probe: importing `maxim.cli`, `maxim.api` and `maxim.create` causes zero spawn or connect audit events, so nothing runs before `main`. |
| api `run`, `imagine`, `connect`, `diagnose`, `campaign`, `benchmark`, `research` | yes (AST test) |
| api `download_model` | no. It fetches from the network; CLI parity is missing |
| api `observe`, `recall`, `list_models`, `delete_model`, `on`, tools, `register_persona` | no, but these are local and need no engage |
| `maxim.create.router` / `pool`, `maxim.load.*`, direct `LLMRouter` | **no** (S1) |
| `maxim-diagnostics` console script (`utils/reachy_diagnostics.py::main`: spawns `ping`, `date`, `connect_ex`, utils.http) | **no** |
| `python -m maxim.embodied_runtime.selfy` (builds the full embodied runtime and router) | **no** |
| `python -m maxim.models.download` (the download hint printed by `cli.py` and `doctor`) | **no** |
| scripts that build routers (`exp44/rerun_ablated_offline.py`, `behavioral_convergence_*`) | **no** (S1); scripts that spawn `maxim` go through `cli.main` and are fine |

### Findings

**S3 SHOULD-FIX: `maxim doctor` and `maxim.diagnose()` crash with an uncaught `OfflineRefusal` under `llm.offline`.**
- Where: `doctor/checks.py::run_all_checks` → `check_llm_server_exposure` (`getaddrinfo('127.0.0.1', 8100)`).
- This is new: before the diff, `llm.offline` had no effect, so doctor ran. Now the operator's diagnostic tool dies with a traceback on the first check that does not catch the refusal.
- Verified: `maxim.diagnose()` and `cli.main(['doctor'])`, both with config `llm.offline: true` → `OfflineRefusal: ... socket.getaddrinfo refused: name lookup ('127.0.0.1', 8100)`. Nine refusals were on the ledger (5 utils.http, 2 getaddrinfo, 2 Popen).
- Fix: either have `run_all_checks` report each network or spawn check as "refused (offline)" (catch `OfflineRefusal` per check and record it), or document it and exempt doctor. This is an owner decision. The present state, a crash, is the one outcome neither the notes nor the docs describe.

- **N3 NIT:** `maxim-diagnostics`, `selfy`'s `__main__` and `python -m maxim.models.download` do not engage. Under offline, the CLI refuses the download that `python -m maxim.models.download` performs, and the CLI's own hint points to that module. Either engage in those `main()`s, or list them in the docs as outside the latch.
- **N4 NIT:** `api.download_model` does not engage, while the CLI equivalent does. The changelog wording ("verbs that build a run") excludes it, but plan §6 says `llm.offline` "wins over every auto-download switch".

---

## Contract 4: `CONFIG_FORMAT_VERSION` 1.4 → 1.5

### Map
- **Writers** (`config_writer` with_format_version/downgrade/restore) use the constant. Correct.
- **Fixtures:** `config_schema_by_version.json` and `config_resolution_by_version.json` gained a 1.5 entry, and `test_config_format_version_856` passes. The only fields missing from the config CLI registry are the structured `lanes.*.extra`/`placement` paths and `llm.offline`.
- **Docs:** `docs/user/configuration.md` is updated. The `docs/agents/runtime-tools.md` line "config 1.4" refers to stop_budget_s and is still correct.
- **No other open PR touches the config schema.** #1431 `talk_tools` is not on main.
- `test_advisor_gate.py` writes a `"1.4"` file on purpose (reading an older version), which is fine.

### Findings

**B1 BLOCKER: a hard-coded version assert outside the diff now fails.**
- Where: `tests/unit/test_bounded_stop_side_1438.py::test_g7_config_1_4_carries_the_console_stop_budget` asserts `CONFIG_FORMAT_VERSION == "1.4"`.
- Verified: in the copy, `AssertionError: assert '1.5' == '1.4'`. CI's fast suite will be red.
- Fix: assert that the field exists and that its first appearance in the schema fixture is "1.4", not that the live version equals "1.4". The same pattern as `test_config_version_owns_llm_offline` works.

**S4 SHOULD-FIX: `maxim config set llm.offline true` is refused with "Unknown field path".**
- Where: `runtime/config_cli.py::_cmd_set` / `_cmd_unset` / `get` / `list`, and `config_loader.py` (line 1119), all key on `_FIELD_TO_ENV`.
- Every settable path must have an env alias. `llm.offline` deliberately has none (no `MAXIM_LLM_OFFLINE`), so it was left out of `_FIELD_TO_ENV`. As a result:
  - `maxim config set`, `unset` and `get` refuse it, and `maxim config list` does not show it;
  - the operator can only turn offline on or off by hand-editing config.json, against the "configure via `maxim config`" rule;
  - the new `"llm.offline"` entry in `_coerce_for_field` cannot be reached from the CLI.
- Verified: `cli.main(['config','set','llm.offline','false'])` printed "Unknown field path: 'llm.offline'" and the file was unchanged. Of the 58 scalar fields in 1.5, `llm.offline` is the only one missing from `_FIELD_TO_ENV`.
- Fix: separate "settable paths" from "env aliases". For example, map `llm.offline` to None and have `_ABSORBED_ENV_VARS` and the env-resolution helpers skip None, or add a `_SETTABLE_WITHOUT_ENV` set that config_cli and the loader also consult. Add a round-trip test for `maxim config set llm.offline true`, and one that `MAXIM_LLM_OFFLINE` is not read.

- **N5 NIT:** `maxim config downgrade` on a 1.5 file says "nothing to set aside", which is correct. The new warning fires only from a newer build's file, and `test_downgrade_warns_when_it_drops_llm_offline` covers it. It is noted here only to confirm the producer (`entries` maps field path to raw value) matches what the warning reads.

---

## Contract 5: the guards moved into `_auto_spawn_ruled_out`

### Map
- One caller: `lane_backends.py::build_lane_configs…` (line 2252) → `_maybe_auto_spawn_server`.
- Tests: `test_lane_served_nctx.py`, `test_leader_fail_closed.py`, `integration/test_bootstrap.py` and the new test. 79 passed in the copy.
- Order online is unchanged: env opt-out, then GPU. Offline is checked first, before the :8100 probe, `lsof` and spawn.

### Findings
- **N6 NIT:** offline, a GPU host whose large lane would have been auto-spawned now gets an in-process llama_cpp lane: the model loads in the Maxim process. Plan §6 says such a lane "is refused by name". Either the implementation or the note should change. V8 allows in-process lanes, so the docs only need to say which happens.

---

## Other observations
- **Possible flake.** `tests/integration/test_offline_latch.py::test_router_refuses_a_pre_engage_lane_at_dispatch` failed once in a 9-file batch, but its output was not captured. It then passed in 10 straight runs, including 6 in parallel. I could not reproduce it, so it is unconfirmed. Worth watching: the pre-engage `generate_json` goes through the abandonable worker thread.
- **Environment-only failures in the copy, not defects:** the four `test_offline_latch` git-stamp tests need `.git`, and the two `test_advisor_gate` special-token tests need the vocab fixtures.
- **Drain prober thread:** `_build_local_backend` starts the auto-undrain prober thread *before* the router build refuses, so offline with a mesh config it probes in the background and the latch refuses each probe. Pre-existing ordering; item 10b territory.

## Verdict
SHOULD NOT MERGE AS IS. B1 turns the suite red, and S1–S4 mean the router claim holds only for routers built directly after `cli.main` or the api verbs, with refusals swallowed into substitutions on the lane path and an un-settable config key.
