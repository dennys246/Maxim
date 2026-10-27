# #1338 item 6, delta round: wire integrity

Scope: the staged fold on top of `d362b481` (`git diff --cached`). Read-only review; the four touched test files
plus `test_leader_proxy.py` pass at the staged tree (182 passed, 4 xfailed).

## Producer to consumer map (outside the diff)

| Contract | Producer | Consumers outside the diff | Effect of the fold |
|---|---|---|---|
| `LLMRouter.generate(prompt, *, system, max_tokens, temperature, request_context) -> str` | `models/language/router.py::LLMRouter.generate` (new) | `runtime/agent_pool.py::_generate_llm_response` (in diff); `bridges/tool_pain_bridge.py::ToolPainBridge._generate_reflection` (`self._llm.generate(prompt, max_tokens=100)`); `create.router` docstring and `docs/user/python-api.md` example | AgentPool: an LLM turn instead of a swallowed `AttributeError`. ToolPainBridge: no production caller passes `llm=` (`runtime/bootstrap.py` builds it without one), so the Reflexion path stays on its template in production; a caller that passes a router now gets text instead of `None`. Its `except Exception` (pre-existing) also catches `ModelInferenceError` and `LeaderRefusal` at DEBUG. |
| Same name, other types | `agents/llm_agent.py::LLMAgent.generate` / `ChatLLMAgent.generate`, `simulation/narrator.py::generate`, `hivemind/signing.py::BundleSigner.generate` | `simulation/interactive.py`, `simulation/simulation_generator.py`, `simulation/generative_runner.py` | None: these call their own classes, not a router. `LLMRouter` has no subclass and no `__getattr__`; `FunctionRouter` has no `generate`. Test fakes (`tests/conftest.py::FakeRouter`, `test_atonia_open_items_other.py::_FailingRouter`) define `generate(prompt, **kw)` and still satisfy the new call (keywords only past `prompt`). No fear-agent, bench, console or scripts caller exists. |
| `/v1/debug/ping` `server_llm` | `leader_proxy.py::_ProxyHandler` ping handler + `set_server_llm_status` (written only by `cli.py::_run_server_mode` / `_report_server_not_ready`) | `doctor/cli.py` (reads `service`, `uptime_s`, `auth_enabled`), `peer/cli.py::_check_proxy_ping` (returns the dict), `peer/admin_core.py` restart poll + 404 diagnosis (read `service`, `llm_ready`), `docs/troubleshooting/leader_proxy_debug.md` | Additive field, read by no code; every reader uses `.get`. No schema pin, contract_surface or OpenAPI file names the ping body (grep of `src/`, `docs/`, `scripts/`, console assets: none). |
| Server mode stays up when not ready (LeaderProxy runs) | `cli.py::_run_server_mode` | `peer/admin_core.py` restart poll (`/v1/debug/ping` `llm_ready`, bounded 15 + 30 polls, then exit 0 "still loading"); `/v1/admin/llm-swap` → `lane_backends.swap_llm_server`; `/v1/admin/restart`; `_early_leader_boot` / `lane_backends._maybe_start_leader_proxy` (who sets `_leader_proxy_server`) | Peers never hang: every poll is bounded, and the first ~180 s are unchanged (the proxy already answered while `wait_ready` ran). Recovery via `peer llm` / `peer update` + `restart` now has a live target. Inference sent to the not-ready leader still fails at the upstream (no silent success). |
| `create.router` raises `ConfigurationError` | `create.py::router` + `_router_missing_reason` | No caller in `src/` or `scripts/`; docs `python-api.md`, `getting-started.md`, `stable_api.md`, `create.py` docstrings (updated); tests `test_router_generate_1338.py` | Callers that checked `None` now get the typed error (documented as breaking under Changed). `_router_missing_reason` reads `_BUILTIN_PROFILES[normalize_llm_profile(model)]["cloud"/"api_key_env"]`, both present (e.g. `claude-sonnet-4-6`). |
| `TurnResult.error_type` | `agent_pool.py::run_turn`, `run_round` | None outside `agent_pool.py`; `TurnResult` is not serialized, not frozen, not re-exported; every construction is keyword (field inserted before `duration_ms` is safe) | Additive. |

## Findings

1. **SHOULD-FIX, fail-closed: `server_llm` goes stale after a recovery.** `set_server_llm_status` is written only at
   server-mode startup. After the recovery the docs prescribe (`maxim peer llm <model>` → `swap_llm_server`), ping
   keeps `server_llm: {"ready": false, ...}` beside a live `llm_ready: true`. It errs toward "not ready", so nothing
   passes that should not, but `leader_proxy_debug.md` tells the operator to read that field. Either have the swap
   success path set `ready=True` (reason `"swapped to <model>"`) or document the field as the startup verdict only
   and point at `llm_ready` for live state. Follow-up issue is enough.

2. **NIT: `maxim peer restart` ignores `server_llm`.** When the restarted leader comes back not ready, the peer polls
   `llm_ready` for its bounded window and then prints "LLM model is still loading ... requests will queue" (exit 0)
   although the leader has reported a terminal NOT READY. Bounded and unchanged in timing from before the fold; reading
   `server_llm.ready is False` there would let it say "NOT READY: <reason>". Fail-closed (inference still fails).

3. **NIT: a LeaderProxy on a non-default port is not seen.** `leader_proxy_running()` reads the default-port
   singleton only, so a server mode whose proxy runs on `MAXIM_LEADER_PROXY_PORT != 8099`, or one sharing another
   process's proxy, exits 1 instead of staying up. Fail-closed (the old behaviour).

4. **NIT: `ToolPainBridge` Reflexion now reachable but its handler is the old swallow.** The `except Exception`
   around `self._llm.generate` logs at DEBUG and also eats `LeaderRefusal`. No production caller wires `llm=`, so
   this is dormant capability; the CHANGELOG line "the Reflexion path of `ToolPainBridge` called a `.generate` no
   router had" is accurate but should not be read as a shipped Reflexion fix (zero non-test callers pass a router).
   Fail-closed (falls back to no reflection).

5. **NIT: `stable_api.md` lists `LLMRouter.generate` without `request_context`.** The signature in the doc omits the
   fifth keyword; harmless, but the stable table is the signature reference.

No BLOCKER. No SHOULD-FIX fails open.

**STOP: no further round owed.** Finding 1 goes to a follow-up issue (or a one-line fold, which as a `src/` change
would itself need a delta-scoped round).
