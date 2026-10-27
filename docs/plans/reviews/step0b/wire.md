# Step 0b (offline latch): wire-integrity review

Diff: `git diff origin/main...HEAD` (63f196bb + merge). Read-only; probes run: grep across `src/ scripts/ tests/ docs/
.github/`, one `MAXIM_OFFLINE=1 python -c "import maxim, maxim.cli, maxim.simulation.orchestrator"` (clean, ledger
empty), one `MAXIM_OFFLINE=1 python -m maxim doctor --json` (crashes, F4).

## Producer → consumer table

| Contract | Producers (before → after) | Consumers (before → after) | What each consumer gets, before → after | Probe |
|---|---|---|---|---|
| Vocabulary: canonical truthy set | `cluster_bias_annotation.TRUTHY_DISABLE_VALUES` = {1,true,t,yes,y,on} → alias of `_offline.TRUTHY_VALUES` = {1,true,yes,on}; `annotation_disabled_via_env` → delegates to `_offline.truthy` | src: `agent_loop` (MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION ×2, MAXIM_DISABLE_VARIANCE_ANNOTATION), `tool_dispatch` + `scripts/survival_world/r2_learned_bias.py` (MAXIM_OPERANT_ONLY_CREDIT), `substrate_proposal` (MAXIM_PLACE_CODE_EXTEROCEPTION), `fixture_orchestrator` + `orchestrator` (MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL, MAXIM_DISABLE_IMAGINATION, MAXIM_DETERMINISTIC_SCENE_EMBODIMENT), `audio_orient_wiring` (MAXIM_SIM_AUDIO_ORIENT), `acting_coach` (MAXIM_DISABLE_COACH_BODY_LAYERS), `internet_access` (MAXIM_INTERNET_LOWERED, now via `truthy` directly), new: MAXIM_OFFLINE. Tests: wire_a, wire_1, acting_coach, test_offline_latch (updated) | `t`/`y` was ON → now OFF for all 11 toggles. No script/doc/workflow/scenario sets any of them to `t`/`y` (regex sweep); only an operator shell could. | grep, regex sweep |
| Replica parsers (not changed) | `integration/memory_hub.py::body_state_prompt_enabled` (6-value), `decisions/nac.py::_REWARD_BIAS_DISABLE_TRUTHY` (6-value), `scripts/benchmark_exp42_preference.py::_TRUTHY` (6-value) | agent_factory `_maybe_wire_body_state`; NAc reward bias; Exp 44 `_ablation_arm` | unchanged: still accept `t`/`y` → now DISAGREE with the canonical set (F1, F2) | read |
| Call: `utils/http.py` registry clients | `_EndpointRegistry.get_client` → + `trust_env=not is_offline()`, + request hook; `_checked_client` → strict when offline, + hook | `request`/`get`/`post`, `fetch_url`, `stream_post` (peer backend streaming), `raw_proxy_forward[_streaming]` (leader proxy), `download_to_file` (`_download_client`), `address_checked_client` (OpenAI SDK with a base_url) | online: identical (hook is a no-op; env read per request). offline: `OfflineRefusal` before connect, booked as `utils.http request`. NOT covered: Anthropic SDK client, OpenAI SDK client with no base_url / policy `any` (returns None → SDK's own client) — deferred to item 10 router refusal per notes §6 | read httpx paths |
| Lifecycle: `close_all` on engage | `_offline.on_engage(close_all)` at http import (new) | every pooled registry client; in-flight callers on another thread | in-flight call on a closed pool dies with an httpx transport error (or `RuntimeError("client has been closed")` in the get_client→request window), NOT `OfflineRefusal`, unbooked (F6). RLock makes the get_client→is_offline→engage→close_all re-entry safe. | read |
| Call: `code_tree.git_out` | → returns None when `sys.modules["maxim._offline"].is_offline()` | `head_commit`/`short_commit` → "unknown"; `code_paths`/`code_tree_sha256`/`_top_level_problem` → their failure strings; `scripts/_provenance.py` (path-loaded; probe live only if the process imported maxim) | before: real git; after (offline): "unknown"/failure — fail-closed in every reader | read |
| Payload: report.json code stamp | `report.py::_code_stamp` → offline: `executed_git_hash`/`code_tree_sha256` = "unknown: offline", dirty = True | `_evidence_records.unknown` (prefix "unknown:" → unknown ✓), `judge_provenance` (dirty→fail), `lint_prereg_precedes_data` (dirty→"dirty without allowance"), `o19_verdict` C3 + marker/bound-blob checks (Refusal), `grounding_census` (prints) | all fail closed; dirty-flag readers mislabel the cause (F7). Mid-run engage: start real, end "unknown: offline" → C3 mismatch, fail-closed ✓ | read |
| Call: `get_version_info` | → offline early-return `{"version"}` only | `cli` main menu (version only ✓), leader_proxy `/version` + `/v1/debug/version`, `peer/cli _cmd_version` (mismatch only if both have hash ✓), `maxim_peer_backend.probe_leader_disk_head` (None ✓), `benchmark_cross_session._capture_versions` | absent `git_hash` = indistinguishable from a PyPI install (F8) | read |
| Gate: `session_internet_lowered` | → `is_offline() or truthy(MAXIM_INTERNET_LOWERED)` | `load_internet_policy` → source "session-lowered"; `lower_internet_for_session` (no write ✓); `live_internet_policy_getter` warning; `internet_access_toggle` remedy; `effective_internet_enabled` → cli/agentic_runtime `state.data["internet_access"]` (launch snapshot) | tools disabled ✓; but every message names MAXIM_INTERNET_LOWERED / "turned off from inside the session" (F3) | read |
| Gate: `maxim/__init__` self-engage | new | every process importing maxim with MAXIM_OFFLINE truthy: `python -m maxim`/console script sub-sims, mp spawn children (fork_exec raises no audit event, so a latched parent can still start them), scripts importing maxim. A latched parent cannot `Popen` a sub-sim (refused, loud). | `import maxim`/cli/orchestrator import clean; first spawn/DNS anywhere raises (F4) | ran |
| Test lifecycle: conftest pop + `_reset_offline_latch` | new | network_guard (independent), `_isolate_maxim_internet_lowered_env` (offline writes no such var ✓), monkeypatch (its teardown runs first; `_reset_for_tests` may re-set MAXIM_OFFLINE=1 from `_saved_env`, then the fixture's `finally` restores the setup snapshot ✓) | library flags not reset (F5) | read |

## Findings

**BLOCKER:** none.

**SHOULD-FIX**

- **F1 `scripts/benchmark_exp42_preference.py::_ablation_arm` / `_TRUTHY`.** Exp 44 arm labels still use the
  6-value set. Scenario: `MAXIM_ENABLE_BODY_STATE_PROMPT=t MAXIM_DISABLE_COACH_BODY_LAYERS=t` → harness records
  `ablation_arm: "B"`, but `acting_coach_config_from_env` now leaves layers 2+4 ON (arm C behaviour) and
  `memory_hub` still wires body state. A record mislabelled silently. `tests/behavioral/test_exp42_pipeline.py::TestExp44ArmDerivation`
  pins `t`/`y` as truthy ("t/y must count"), so the test enforces the divergence. Fix: harness imports
  `maxim._offline.truthy` (stdlib leaf, safe for a standalone script) and the test flips.
- **F2 `integration/memory_hub.py::body_state_prompt_enabled`, `decisions/nac.py::_REWARD_BIAS_DISABLE_TRUTHY`.**
  Both replicas still accept `t`/`y`, and their comments are now false ("mirrors `annotation_disabled_via_env`";
  "test_body_state_wiring.py pins t/y so divergence from the canonical set fails loudly", which pins the replica, not
  agreement). Their stated reason for replicating (no `prompts/` import) is gone: `maxim._offline` is a top-level
  stdlib leaf. Root fix: both call `_offline.truthy`; `tests/unit/test_body_state_wiring.py` flips; add them to
  `test_one_truthy_parser`'s agreement table. Otherwise the CHANGELOG's "one parser" claim is false for 2 toggles.
- **F3 `utils/internet_access.py::EffectiveInternetPolicy.disabled_reason` / `live_internet_policy_getter` /
  `tools/internet_search.py` toggle remedy.** Offline maps to source `"session-lowered"`, so the LLM-facing error
  says "turned off from inside the session" and the remedy says "start a new session without
  MAXIM_INTERNET_LOWERED"; the startup WARNING says that variable "is set". None is true offline; following the
  remedy does nothing. Needs a distinct source (e.g. `"offline"`) or offline-aware text.
- **F4 `docs/agents/llm-routing.md` MAXIM_OFFLINE row.** Documents a live operator variable, but item 10's call-site
  refusals are not in this step: `MAXIM_OFFLINE=1 maxim doctor --json` crashes with an uncaught `OfflineRefusal`
  from `doctor/checks.py::check_peer_url_reachable` (`socket.gethostbyname`). ~70 `subprocess` sites in `src/`
  catch `OSError`/`CalledProcessError` only, and stdlib helpers that spawn under `except OSError` (`ctypes.util.find_library`
  on Linux, `platform.processor`) will raise through imports such as soundfile/sounddevice on Linux (not probed
  here, macOS uses dyld). State in the brief that the variable is not yet operator-safe until item 10 lands.

**NIT**

- **F5 `_offline._reset_for_tests`** restores env, not `huggingface_hub.constants.HF_HUB_OFFLINE` /
  `ultralytics.utils.ONLINE`; an in-process engage with either imported leaks into later tests (order-dependent).
- **F6 `utils/http.py::close_all` via `on_engage`.** An in-flight request on another thread fails as an httpx transport
  error (router may book it as a peer failure) or a bare closed-client `RuntimeError`; neither is in the ledger.
  The docstring's "cannot keep sending" holds for registry clients only; SDK-owned pools are item 10's.
- **F7 `simulation/report.py::_code_stamp`** stamps `working_tree_dirty_src_scripts: True` offline, so
  `lint_prereg_precedes_data` and `judge_provenance` report "dirty without allowance". Fail-closed, but misattributed.
- **F8 `maxim/__init__.py::get_version_info`** offline is indistinguishable from a PyPI install to `/version`
  readers; consider `"git_hash": "unknown: offline"`-style marker if any consumer must tell them apart (none does today).
- Other independent six-value parsers (`orchestrator._env_trace`, `cli_utils`, `models/download.py`,
  `models/language/config.py`, `gpu_compat`, `plotting`) still accept `t`/`y`. Out of this diff's contract, but they
  contradict CLAUDE.md's "canonical parser" line; worth one follow-up issue.
