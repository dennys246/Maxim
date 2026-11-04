# Executor lens: #1315 item 10a (d0729a0f..fb3ae6f3)

Probes ran in a copied tree (`scratchpad/review10a/copy`, PYTHONPATH pinned to the copy's `src/`, `maxim.__file__`
confirmed). Four `test_offline_latch.py` provenance tests fail in the copy only because it has no `.git`
(`ScenarioBroken: the gate needs a git checkout`). That is a copy artifact, not a finding.

## SHOULD-FIX

### S1. A config.json that does not parse leaves the latch down, and the run goes online
`runtime/config_loader.py::engage_offline_if_configured`
- **Scenario.** `{"llm": {"offline": true},}` (a trailing comma), a UTF-8 BOM (a Windows editor), or a file that
  cannot be read. `engage_offline_if_configured` logs one WARNING and returns False. The docstring says such a file
  is "refused loudly by the load that follows", but on the run path it is not:
  - `config_loader.resolve_setting` catches every `load_config` error and falls back to defaults;
  - `lane_backends.py` (around line 423) logs "Failed to load config.json" and carries on;
  - `role.py` and `cli_utils.py` (around line 414) warn and fall through.

  So an operator who set `llm.offline: true` and made a typo gets an online run with cloud or peer lanes live, and
  only a WARNING that is printed before logging is configured. The "fail-closed" note covers a file that fails
  validation, not one that fails to parse.
- **Verified.** probe1.py: an empty file, a BOM and a directory each return False with a WARNING. A BOM file makes
  `load_config` raise `ConfigurationError`. The fall-through handlers were confirmed by reading.
- **Fix.**
  - Preferred: when the file exists but cannot be read or parsed, refuse startup (raise `ConfigurationError` naming
    `llm.offline`), or engage, since the setting cannot be known.
  - Weaker: scan the raw text for `"offline"\s*:\s*true` and engage on a match.
  - Either way, correct the docstring. The owner should decide between fail-closed and refuse.

### S2. `LLMRouter.generate()` re-types `OfflineRefusal` as `ModelInferenceError`
`models/language/router.py::LLMRouter.generate`
- **Scenario.** A router built before `engage()` has a cloud lane; after the engage, `router.generate("hi")` is
  called. The dispatch refusal is caught by `except Exception` and re-raised as `ModelInferenceError`, with the
  refusal only as `__cause__`.
  - `LeaderRefusal` has a pass-through (`except LeaderRefusal: raise`); `OfflineRefusal` gets none.
  - Callers that treat `ModelInferenceError` as an ordinary model failure absorb it: `agent_pool.py` around line 394
    re-raises it as a model error, and `tool_pain_bridge.py` around line 806 returns None.
  - `OfflineRefusal`'s own docstring says it is a `RuntimeError` precisely so that catch-and-continue sites cannot
    absorb it.
  - `generate_json` and `route` propagate the refusal as itself, so the three entry points are inconsistent.
- **Verified.** probe2.py, case B: `generate raised ModelInferenceError | cause: OfflineRefusal`;
  `generate_json raised OfflineRefusal`; `route raised OfflineRefusal`.
- **Fix.** `except (LeaderRefusal, OfflineRefusal): raise` in `generate`, plus a unit test.

### S3. `warmup()` bypasses the router predicate: the thread dies or calls a cached backend
`models/language/router.py::LLMRouter.warmup`
- **(a) Uncached backend.** Router built before the engage (cloud lane), then `engage()`, then `warmup()`:
  - `warmup()` returns True;
  - the `LLMWarmup` thread raises `OfflineRefusal` and dies, so `_ready_event` is never set;
  - `wait_ready(120)` blocks for the full timeout and returns False ("timed out"). The orchestrator AUT path,
    research and foundry all call `warmup(); wait_ready(120.0)`.
- **(b) Cached backend.** The backend was cached before the engage. `_get_backend()`'s fast path returns it with no
  check, and warmup calls `backend.complete("Hello")` directly:
  - the router refuses nothing and the ledger stays empty;
  - only the socket hook stands in the way, and it cannot see a send on a pooled connection opened before the latch.
- The changelog says "at router build and at every call"; this call is missed.
- **Verified.** probe2.py:
  - case C: `thread died: OfflineRefusal`, then `wait_ready False after 3.0s`, `ready_event set: False`;
  - case D: `fake backend.complete calls after engage via warmup: 1`, `ledger: []`.
- **Fix.** Call `self._refuse_offline_lanes("router[warmup]")` at the top of `warmup()`, on the caller's thread, so
  the refusal is typed and synchronous. Optionally, `_warmup_thread` should set `_ready_event` in a `finally`.

### S4. A refused lane at router build is swallowed into a WARNING, and the run falls back to another router
`runtime/lane_backends.py::LaneBackendManager._build_local_backend` (the `except Exception` around `LLMRouter(...)`,
about line 1228) and the remote-lane builder (about line 1439)
- **Scenario.**
  1. With `llm.offline: true` and `lanes.large.remote_url` set to a peer, `LLMRouter(...)` raises `OfflineRefusal`.
  2. It is logged as "Failed to create LLM router" and None is returned.
  3. `orchestrator.py` (about line 969), `agentic_runtime.py` (about line 897) and `research_orchestrator.py` then
     fall back to `LLMRouter(load_llm_config())`.
  4. If the default profile is a local llama_cpp one, the run proceeds on a model the operator did not configure,
     with one WARNING.

  Further swallows sit at the AUT path (`orchestrator.py` about line 1531, `except Exception` → the shared router),
  `research_orchestrator` (heuristics), `benchmark.py` around line 965, and CLI curate (DEBUG).
- §6 says "the CLI refuses llm.offline at startup unless every lane resolves to an in-process backend". That refusal
  is not reached on the shipped path; the router refuses, but its caller swallows it.
- **Verified.** By reading the handlers; not run end to end, because the run path spawns and probes.
- **Fix.** Follow the precedent: `except (LeaderRefusal, OfflineRefusal): raise` (or an `OfflineRefusal` re-raise)
  ahead of each `except Exception` around router construction. If the plan defers this to 10b's swallow census, say
  so in the PR and changelog: today the changelog claims a refusal that the user sees only as a WARNING.

### S5. `llm.offline` is not reachable from `maxim config set` / `unset`, nor from doctor's resolved-config rows
`runtime/config_writer.py::set_field`, `unset_field`; `runtime/config_loader.py::_coerce_for_field`
- **Scenario.** `maxim config set llm.offline true` raises
  `ConfigurationError: unknown field path 'llm.offline'`. `set_field` and `unset_field` gate on `_FIELD_TO_ENV`, and
  the key deliberately has no env alias. Consequences:
  - the operator can only hand-edit JSON (CLAUDE.md: configure through `maxim config`);
  - `maxim doctor`'s resolved-config walk (`doctor/checks.py`, about line 387, iterates `_FIELD_TO_ENV`) never shows
    whether offline is on;
  - the `"llm.offline"` entry this diff adds to `_coerce_for_field`'s bool set is dead, because both of its callers
    (the env path and `set_field`) are unreachable for the key.
- **Verified.** Probe: `set_field("llm.offline", "true")` raises the error above. A config with the key survives a
  `set_field` of another field (checked: preserved, stamped 1.5; also from files stamped 1.4, 1.1 and unstamped).
- **Fix.** Let `set_field`/`unset_field` accept the `LLMConfigSection` fields that have no env alias (an explicit
  `_CONFIG_ONLY_FIELDS` set), and show the key in doctor. If that is out of scope, drop the dead coerce entry and
  document hand-editing in `configuration.md`.

### S6. Library entry points that build a router never read `llm.offline`
`create.py::router` (which calls `build_primary_router`), `api.py::download_model`, and `LLMRouter.__init__`
- **Scenario.** §6: "`llm.offline` is read at CLI startup and at the **first router** or `api` verb build". Only the
  CLI and the seven api verbs call `engage_offline_if_configured`.
  - `maxim.create.router()` (a public, documented verb), `maxim.create.agent`/`pool` with a router, and
    `maxim.download_model()` (an HF fetch) run online in a library process whose config.json sets
    `llm.offline: true`.
  - The `maxim-diagnostics` console script (`reachy_diagnostics:main`) does not engage either.
- **Verified.** By reading; `grep engage_offline_if_configured|_engage_offline()` shows only `cli.py` and `api.py`.
- **Fix.**
  - Call `engage_offline_if_configured()` in `build_primary_router` (or `LLMRouter.__init__`), which covers every
    router build.
  - Add it to `download_model` (and to `list_models`, if that probes).
  - Narrow the docs and changelog wording accordingly.

## NIT

- **N1. An empty or whitespace-only config.json logs a spurious WARNING on every `maxim` command.**
  `engage_offline_if_configured` → "could not read ... to check llm.offline: Expecting value". `load_config` treats
  such a file as defaults. Verified: probe1.py, case A. Fix: `if not raw.strip(): return _offline.is_offline()`,
  mirroring `load_config`.
- **N2. Deletion probe survives: emptying `api._engage_offline`'s body fails no test.** The AST test checks only that
  the call is named, and `test_api_run_engages_before_its_checks` monkeypatches the helper. Verified: delprobe
  `api_engage_body` gives 100 passed. Fix: a test that writes `llm.offline: true` and asserts `api._engage_offline()`
  raises the latch (or calls `api.connect` with the registry stubbed).
- **N3. New router lines run only in child processes, so the in-process diff-coverage gate does not see them.**
  - `router.py::_refuse_offline_lanes`: the `_offline.refuse(...)` line;
  - `_complete_text`'s `self._refuse_offline_lanes("router[dispatch]")` line.

  Both are exercised only by the subprocess red gates. Verified with `--cov`: lines 385 and 993 are missing. Fix: two
  in-process unit tests (build refused with a cloud lane; dispatch refused after a pre-engage build). They are cheap,
  and they also guard S3.
- **N4. `test_a_file_that_fails_validation_still_engages` uses `pytest.raises(Exception)`.** It should name
  `ConfigurationError`, so that an unrelated error does not satisfy the precondition.
- **N5. Out of 10a's scope, but on its path.**
  - `ensure_blackwell_guards`' `nvidia-smi` refusal is still caught by `except Exception` at DEBUG (§6 wants WARNING).
  - `_maybe_auto_spawn_tunnel_daemon` is not ruled out offline; its `cloudflared_already_running` will meet the
    spawn refusal.

  Both are presumably 10b; listed so they are not lost.

## Checked and sound

- Every public dispatch reaches the predicate. `generate`, `generate_json` (including the ANSWER_ONLY and TOOL_PROMPT
  branches) and `route` all go through `_complete_text`, which checks before `abandons_on_stop`, so the helper thread
  never runs. `generate_json`'s earlier `_get_backend()` refuses via `router[backend]` too. Probe: both raise
  `OfflineRefusal`.
- `_normalize_providers`:
  - empty `providers` synthesises `{"local": type = cfg.backend or "llama_cpp"}`, judged by the same
    `_provider_type` that `_get_backend_for_provider` dispatches on, so the predicate and the dispatch cannot
    disagree;
  - a provider with an empty type and an empty `cfg.backend` is refused (fail-closed);
  - `"gpt"`, `"maxim_peer"`, `"openai_compatible"` and unknown types are all refused;
  - `_providers` is assigned only in `__init__`, so no lane is added after the build check.
- `_get_backend_for_provider` refuses before any non-in-process backend is constructed. Its cache fast path is safe on
  the dispatch path, because `_complete_text` has already refused.
- `engage_offline_if_configured`:
  - only a JSON `true` engages (string, int, false, absent, non-dict JSON and a missing llm section do not);
  - `MAXIM_OFFLINE` still counts;
  - it reads the same `config_path()` that `load_config` uses (there is no `--config` override in the CLI);
  - `UnicodeDecodeError` is a `ValueError`, so it is caught.
- `cli.main`: the engage is the first statement. Importing `maxim.cli` and `maxim.api` raises no
  spawn/connect/lookup audit event (an audit-hook probe at import came back empty), so "before any subprocess or
  socket" holds.
- The seven api verbs engage first. `_run_preflight` moved verbatim, with the engage ahead of the goal and robot
  checks.
- `_auto_spawn_ruled_out`: same semantics and order as the guards it replaced, with offline first. The deletion
  probe was caught.
- Config 1.5:
  - the schema and resolution fixtures, the format-version tests and persistence compat all pass (288 passed);
  - the downgrade warning fires;
  - the key is not in `_NON_SECURITY_FIELDS`;
  - the key round-trips through `set_field` of other fields.
- Function-length lint is clean in the worktree (rule T: `run` 224 → 209, `_maybe_auto_spawn_server` 259 → 255;
  exemption text present). The no-silent-swallows lint is clean.
- Deletion probes caught: router build check, router dispatch check, `router[backend]` check, auto-spawn offline
  return, `is True` → truthy, the downgrade warning, the engage in `api.connect`. Survived: N2.

Verdict: SOUND WITH CONDITIONS. The router predicate and the startup engage work as claimed on their direct paths, but
S1 (a config that does not parse runs online) and S2–S4 (the refusal is re-typed, bypassed or swallowed between the
router and the user) should be folded, or explicitly deferred to 10b with the docs narrowed, before merge.
