# #1315 item 10a: Architecture lens review

Diff: `d0729a0f..fb3ae6f3` (branch `feat/1315-item10a-offline-router`). All probes ran in a copied tree
(`scratchpad/review10a/copy`, `PYTHONPATH` set to its `src`); nothing in the shared worktree was edited. The one
read-only command run in the worktree was `scripts/lint_function_length.py`, which needs git for the touch rule.

## BLOCKER

### B1. The router-build refusal is swallowed on the shipped path; the claim holds only for a hand-built `LLMRouter`
**Where:** `runtime/lane_backends.py::LaneBackendManager` (both router builders: `return LLMRouter(llm_config, ...)`
and `return LLMRouter(remote_cfg)`, each wrapped in `except Exception as e: logger.warning(...); return None`),
reached from `build_primary_router`. The new `models/language/router.py::LLMRouter._refuse_offline_lanes("router[build]")`
raises inside those handlers.

**Consequence:** under `llm.offline`, a peer, loopback-server or cloud lane does not refuse with `OfflineRefusal`
on any real entry point. It logs one WARNING and `build_primary_router` returns `(None, manager)`. Then:
- `simulation/orchestrator.py` (the `build_primary_router(logger=logger)` call) and
  `embodied_runtime/agentic_runtime.py` fall back to `LLMRouter(load_llm_config())`. If the default profile is a
  local GGUF, the run **silently uses a different model** from the operator's configured lane. If it is disabled,
  the run has no LLM. This is the silent-substitution class the weak-evidence rule exists for.
- `api.run` raises `ConfigurationError("No LLM backend available. Set --language-model or MAXIM_LLM_PROFILE.")`,
  which names the wrong cause and drops the typed refusal.

The changelog fragment ("A cloud, peer or loopback-server lane refuses with `OfflineRefusal` at router build") and
§6's "the CLI refuses `llm.offline` at startup unless every lane resolves to an in-process backend ... refused by
name" are not what an operator sees. The flipped red gates (`test_offline_latch.py::test_router_refuses_cloud_and_remote_lanes_at_build`,
`::…_at_dispatch`) and `test_offline_startup_10a.py` all construct `LLMRouter(...)` directly. That is the
hand-composed sequence that CLAUDE.md's "a fix ships with a CALLER" lesson warns about. Egress still fails closed,
because the socket latch refuses any connect, so this is about the typed refusal and the silent fallback, not a leak.

**Verified:** in the copy, with config `{"llm":{"offline":true},"lanes":{"large":{"remote_url":"http://127.0.0.1:8100"}}}`,
calling `build_primary_router()` after the engage gives
`WARNING … Failed to create remote LLM router for large: offline (llm.offline): router[build] refused: lanes that are not in-process: lane-large`
and `RESULT router= None`. Nothing raises. The ledger has the entry; the caller never sees it.

**Fix (in the seam, not per caller):** let `OfflineRefusal` propagate through the lane manager's two router-build
handlers (`except OfflineRefusal: raise` ahead of `except Exception`). Better still, make the offline predicate a
`build_primary_router` / `LaneBackendManager` check that names every non-in-process lane, which is §6's "startup
predicate checks the built lanes". Then add one red gate on `build_primary_router` (or `api.run`) rather than on a
bare `LLMRouter`. The swallow census (10b) would eventually list these handlers, but this PR's own refusal is the
one being absorbed, so it belongs here.

## SHOULD-FIX

### S1. `llm.offline` is not read at the router build, so `maxim.create.router()` and other non-verb paths run online
**Where:** `runtime/config_loader.py::engage_offline_if_configured` has exactly two production callers:
`cli.py::main` and `api.py::_engage_offline` (7 verbs). §6 says "`llm.offline` is read at CLI startup **and at the
first router** or `api` verb build". The router build reads only the latch.

**Consequence:** any public path that builds an LLM without `cli.main` or one of the 7 verbs ignores the key:
`maxim.create.router()` (which calls `build_primary_router`), so also `create.pool(llm_router=create.router(...))`;
`api.download_model` (an HF download, the most direct network verb in `api`); and `console/handle.py`'s router
rebuild on a config-generation change, so a `maxim serve` whose config gains `llm.offline: true` rebuilds a router
online. The docs promise a restart for that, but the rebuild builds online anyway. A list of entry points is the
same shape §6 rejects for enforcement ("a list of sites cannot be the enforcement").

**Verified:** in the copy, with config `{"llm":{"offline":true}}`,
`maxim.create.router(model='claude-sonnet')` built a router and `_offline.is_offline()` stayed `False`, while
`load_config().llm.offline` was `True`.

**Fix:** make `engage_offline_if_configured()` the first statement of `runtime/lane_backends.py::build_primary_router`,
next to the existing fail-fast seam `_require_valid_user_profiles()` (the step 0a precedent: one seam that every run
path reaches, plus `cli`). It has to come before `detect capabilities`, which spawns. Keep `cli.main` and the verbs
for ordering, since they run before any subprocess. Add `api.download_model` to the verbs, or show that its 10b
call-site refusal covers it; that refusal needs the latch up, so the engage is needed either way.

### S2. `llm.offline` cannot be set, unset, got or listed through `maxim config`
**Where:** `runtime/config_cli.py` (`set`/`get`/`unset`/`list` all gate on `_FIELD_TO_ENV`),
`runtime/config_writer.py::set_field`/`unset_field` (same gate), and `runtime/config_loader.py::_FIELD_TO_ENV`, which
correctly has no env alias for the key. The new `"llm.offline"` entry in `_coerce_for_field` is therefore
unreachable.

**Consequence:** the single-source config path (CLAUDE.md: "configure via `maxim config`") refuses the key with
`Unknown field path: 'llm.offline'`. The only way to turn offline on or off is to hand-edit JSON. A security toggle
that the operator cannot inspect with `maxim config get` or `list` is a usability and audit gap. The root cause is
that `_FIELD_TO_ENV` conflates "a settable field" with "a field that has an env alias".

**Verified:** in the copy, `python -m maxim config set llm.offline false` and `... true` both exit with
"Unknown field path", and the file stayed unchanged.

**Fix:** give the CLI and writer a field registry independent of env aliases. For example, `_SETTABLE_FIELDS =
_FIELD_TO_ENV.keys() | {"llm.offline"}`, or an alias map whose value may be `None`, consulted by
`config_cli`/`config_writer` while `resolve_setting` skips the env leg. Add a test that `maxim config set llm.offline true`
round-trips and that `unset` restores online. Otherwise, drop the dead `_coerce_for_field` entry and document
hand-editing.

### S3. The user docs and changelog present `llm.offline` as an operator feature while most commands crash under the latch
**Where:** `docs/user/configuration.md` (the new `llm.offline` paragraph), `changelog.d/1315-item10a-offline-router.added.md`.
`docs/agents/llm-routing.md`'s `MAXIM_OFFLINE` row still says "**Not yet safe for operators: until item 10's call-site
refusals land, much of Maxim meets an uncaught `OfflineRefusal`**".

**Consequence:** this PR makes the latch reachable from a persisted config key and documents it without that
caveat. If 1.3.4 is cut before 10b lands, operators get a key under which, for example, `maxim doctor` crashes with a
traceback. Robot `connect` is also refused by design (V8), and the user docs do not say so.

**Verified:** in the copy, with `llm.offline: true`, `python -m maxim doctor` exits 1 with
`OfflineRefusal: offline (llm.offline): socket.getaddrinfo refused` from `doctor/checks.py::_check_one_robot`. The
latch refused it; no connection was made.

**Fix:** carry the llm-routing.md caveat into the user doc and the changelog fragment ("experimental, not yet
usable end to end until #1315 item 10b"), and add that a robot or any endpoint, loopback included, is refused. As
the alternative owner decision, hold the user-doc paragraph until 10b. State the item 10a/10b split in plan §6 or §7.

## NIT

### N1. The raw reader drifts from `load_config` at the edges
**Where:** `config_loader.py::engage_offline_if_configured`.
- An **empty** `config.json` is valid for `load_config` (it gives defaults), but the raw reader logs
  `WARNING could not read … to check llm.offline: Expecting value` on every command. Verified in the copy.
- An unreadable path or directory produces a WARNING and runs online. `load_config` raises `ConfigDirUnreadable`
  through `utils/paths.py::config_file_present`. The docstring's "refused loudly by the load that follows" holds only
  on paths that do load config afterwards.
- A non-bool value (`"true"`, `1`) does not engage. `load_config` refuses it, verified, so this fails closed only
  where config is later loaded.

**Fix:** reuse `config_file_present` (and its raise) plus the same strip-empty rule, so the two readers cannot
disagree. Consider making any present `offline` value other than JSON `false` engage, which is fail-closed and lets
the validated load report the type error.

### N2. Plan §1A.6 and §7 are stale on version ownership
They say "#1431 takes 1.4". On main, 1.4 is `console.stop_budget_s` (#1438, unreleased), #1431 (`talk_tools`) has
not landed and would now need 1.6, and the fixture has no `talk_tools`. 1.5 is the right mechanical choice, because
the fixtures are append-only and pinned per version. Fix the plan text so #1431's session does not also claim 1.5.
Note too that 1.4 and 1.5 will both first ship in one PyPI release, which is harmless: the downgrade warning still
fires on 1.5 to 1.4.

### N3. `LLMRouter.warmup` dispatches outside `_complete_text`
Its no-`warmup()` fallback calls `backend.complete(...)` directly. A cloud backend built before `engage()` and cached
in `self._backend` would get one warmup call without the dispatch predicate; the socket latch still refuses it, and
an SDK keep-alive connection is a stated residual. Consider calling `_refuse_offline_lanes("router[warmup]")` there,
so "checked at every dispatch" is literally true.

## Sound

- **Layering.** `models/language/router.py` imports only the stdlib leaf `maxim._offline` at module level. It does
  not read config, so `models/language` gains no dependency on `runtime`. The config read lives in
  `runtime/config_loader.py`, the config owner, and the CLI and api call it lazily. No `LAYER_RULES` boundary is
  crossed.
- **One mechanism.** The router refuses through the existing `_offline.refuse()`, so refusals are booked in the
  ledger before the raise. That is one refusal path with `utils/http.py`, and no new exception type.
- **The predicate is an allow-list, keyed on the type that picks the constructor.** `_IN_PROCESS_TYPES` is built
  from the same `_LLAMA_TYPES`/`_TORCH_TYPES` tuples that `_get_backend_for_provider` branches on, and both use
  `_provider_type`. An unknown or new type (including new `BACKEND_CLASSES` entries) is refused, which fails closed.
  `_get_backend_for_provider` refuses before any SDK client is built. The three check points (build, backend
  construction, `_complete_text`) cover a router built before `engage()`, and every completion funnels through
  `_complete_text`, apart from N3.
- **Auto-spawn.** `_auto_spawn_ruled_out` returns before the :8100 probe, the lsof fallback and the spawn. The rule-T
  extraction carries the edit: 259 → 255.
- **Config.** The field is in the SHAPE-FROZEN `LLMConfigSection` with a default. It has no env alias, stays off
  `_NON_SECURITY_FIELDS` (so it counts as a security field), and has 1.5 fixture entries in both version fixtures.
  The downgrade warning fires on dropping `true`, which is the stated fail-open of §1A.6. Pushing the future-minor
  tests to "1.99" is the right repair.
- **Startup order.** `cli.main` engages before `cli_signals` and the guarded body. Importing `maxim.cli` raised no
  spawn, connect or DNS audit event (verified with an audit hook). `_run_preflight` (rule T: `run` 224 → 209) is a
  verbatim move plus the engage. The baseline note matches.
- **Red gates flipped legitimately.** The removed `xfail(strict=True)` markers are flips, not re-points. The census
  gate stays red. The targeted tests pass in the copy: 42 unit tests and 4 integration router/config tests.
- **Lints in the copy:** function length (also run in the worktree with git: clean), no-silent-swallows (clean),
  orphan modules (clean), claude_md invariants (PASS).

## Verdict
REQUEST CHANGES: the layering is sound, but on the shipped `build_primary_router` path the refusal is swallowed
into a WARNING and a fallback (B1), and the config key is read at two entry points rather than at the router-build
seam that §6 names (S1).
