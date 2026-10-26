# Atonia red gates: delta round 2, wire-integrity lens

Scope: the staged fold on `test/atonia-red-gates` (`git diff --cached` over 11c90ab1): `tests/conftest.py`,
the O6 re-target, the pooled-client, library-flags and `_reset_for_tests` gates in `test_offline_latch.py`, the new
router-build placement reader, the v6.4 kind/layer pins, `_classes_named` in the §1A.11 gate, and the advisor
observer's `root`/`open_dir` changes. Read-only; probes in the session scratchpad.

## Producer → consumer, per changed gate

### O6 (`test_atonia_open_items_other.py::test_o6_inbound_comms_reach_the_prompt_framed`)

| Hop | Production | The gate | Match |
|---|---|---|---|
| Producer | `TwilioChannel.receive_sms/receive_voice` → `gateway.receive_inbound(channel="twilio:sms", metadata={conversation_context, ...})`; `comms/api.py::inject_command` → `receive_inbound(channel="api")`, no metadata | `receive_inbound("sms", ..., INJECTION)`, no metadata | Matches the `/api/command` producer. Not the Twilio one: see S1 |
| Gateway | `runtime/bootstrap.py::build_comms_stack`: `CommunicationGateway(bus=bus)`, called from `cli.py` and `embodied_runtime/agentic_runtime.py` | `CommunicationGateway(bus=AgentBus())` | Yes |
| Bus → memory | `MemoryAgent.__init__` subscribes `_on_percept`; `source.startswith("comms:") and content` → `_comms_messages` | the same, through a real bus | Yes |
| Context | `runtime/agent_loop.py` (run_agentic_loop) calls `memory.build_context()` and submits to `LLMWorker` | `memory.build_context()` | Yes |
| Prompt | `LLMWorker._prompt_builder.build_prompt(LLMRequest)` (`api.py`, `cli.py`, orchestrator, `console/handle.py`, `agentic_runtime.py` all build an `LLMWorker`) | the same, with a router stub and a hand-shaped `LLMRequest` | Yes |

Is "PromptBuilder does not read inbound comms at all" true of the shipped path? **Yes.** I checked these:
- `comms_messages` has one reader, `exec_agent.py::_build_llm_context`, dormant since 2026-10-06 (`ExecAgent._propose_goal`'s docstring). `ExecAgent._on_percept`'s `comms:` branch also feeds only that dormant worker.
- `build_observation_section` renders `transcript_chunk`, `detections` and `cli_input`. It does not render `content`, and `make_text_percept` sets only `content`.
- `build_prompt` reads `context.cli_inputs` and `triggering_input`, and comms never populate either.
- The memory sections (`relevant_memories`, `concept_context`) are dormant and unfed (#845). They render only `goal`/`action`/`success`/`transcript` keys in any case.
- The episodic capture stores `cli_input`/`raw_transcript_text`, both None for a comms percept, plus `observations = percept.metadata`.
- `memory_hub.on_percept_received` encodes `content` into substrate concept chunks (not verbatim, and not wired in the gate). See N2.

With `--runxfail`, the gate fails at the intended assertion, `INJECTION in prompt`, with a non-empty tool prompt. It does not fail at a `require`.

### Pooled client (`test_offline_latch.py::test_a_client_pooled_before_engage_cannot_send_after`)

| Hop | Production | The gate | Match |
|---|---|---|---|
| Pool | `utils/http.py::_EndpointRegistry._clients`: one `httpx.Client` per registered endpoint, including the four `_ensure_external_endpoint` endpoints used by `fetch_url` | `register_endpoint` + `http.get` twice | Yes. Probe: the two requests reuse ONE client port (keep-alive pooled) |
| Close | `engage()` step 3 → `on_engage(close_all)` registered at `utils/http.py` import (notes §1A.1) | read through `engage_callbacks()` + behavioural | Yes |
| After | A request after `close_all` builds a NEW client and connects anew (probe: new port). Only the latch's connect refusal stops it, and V8 refuses loopback | expects a refusal or ledger entry, and the listener sees nothing | Yes. It is discriminating: without `close_all` the pooled socket sends with no audit event and the listener sees a 3rd request |

The pools outside the registry are `openai_backend`'s `address_checked_client`, plus the Anthropic and OpenAI SDKs' own clients. `close_all` does not close them. The notes cover them by another layer: the §1A.1 router predicate refuses at dispatch, and cloud SDKs are never built after the latch. See N1.

### Library flags (`test_engage_sets_already_imported_library_flags`)

- `huggingface_hub` 1.26.0 checks `constants.is_offline_mode()`, which returns the module global `HF_HUB_OFFLINE`, inside `hf_request_event_hook` on every request. Setting `constants.HF_HUB_OFFLINE` after import therefore changes behaviour, and the gate proves it by `OfflineModeIsEnabled` in the exception chain, not by reading the flag.
- `spy="first"` installs the socket spy before any import, so nothing egresses even today.
- The gate fails today, as it should: `maxim._offline does not exist`.

### Router-build placement reader (`test_profile_readers_v9.py`)

- `_placement_entry_to_provider` is called only from `_inject_placement_tail`, on `cfg.placement[1:]`: the FALLBACK tail, from `_build_local_backend` and the remote builder.
- The PRIMARY LOCAL lane in `build_primary_router` resolves `load_llm_config(profile_override=primary.model)` in `_build_local_backend`. That is the same call the existing `build_model_path (load_llm_config)` reader drives.
- So both production paths are covered. The new branch's comment, "build_primary_router compiles each LOCAL placement entry through this mapper", is inaccurate. See N3.

### conftest `_reset_offline_latch`

- conftest forces `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE="1"` at module import. The fixture snapshots per test, after that forcing, and restores the snapshot, so it re-asserts "1" and does not fight the forcing.
- None of the five keys is touched by any other autouse fixture.
- An autouse function fixture is set up before a test's `monkeypatch` and torn down after it, so the two restore the same values.
- `pytest.fail` when the module is loaded without `_reset_for_tests` is fail-closed.
- The module-top `MAXIM_OFFLINE` pop is not in conftest. It is pinned separately by `test_conftest_pops_maxim_offline_before_importing_maxim` (strict xfail).

### Kind/layer pins, §1A.11 class lookup, advisor `root`/`open_dir`

- These are test-side tightenings. Each pins one value where the gate previously accepted a set (fail-closed).
- `_classes_named` finds `MaximCommandTool` and `execute_sandbox_script`'s class, and `_NO_CLASS_NAMES` is checked for equality.
- `_REGISTRY_READERS`/`_REGISTRY_WRITERS` treats an unknown method as a writer, plus a loud guard.
- None of these changes a producer or consumer outside the diff.

## Order dependence

The run covered the seven atonia files, `test_tool_dispatch.py` and `test_api_{core,expansion,surface}.py`: 189 passed and 186 xfailed in both default and reversed order (reversed with a trylast `items.reverse()` plugin). **There is no order dependence.**

## Findings

- **S1 — SHOULD-FIX, fail-open (latent): O6 drives the `/api/command` producer, not the Twilio one.**
  - `TwilioChannel.receive_sms/receive_voice` also put the inbound body into `percept.metadata["conversation_context"]`, through `ConversationManager.get_context_for_prompt`, whose docstring says "for injection into LLM context".
  - If item 4's fix renders that carrier, the gate cannot see it: its percept carries no such metadata. It would XPASS while the Twilio path puts the text unframed into the prompt.
  - Fold: parametrise the producer over (a) the current `receive_inbound` call and (b) the Twilio path. For (b), call `TwilioChannel.receive_sms` on a `__new__`-built channel with `gateway` and a real `ConversationManager(bus)`, or pass `metadata={"conversation_context": conv_manager.get_context_for_prompt(conv)}`.
  - It is test-only, and both arms fail today for the same reason.
- **N1 — NIT:** the pooled-client gate covers the registry's pools only. A sentence in its docstring should name the SDK-owned pools and the notes layer that covers them (router predicate, SDKs never built after the latch), so a reader does not take the gate as full pool coverage.
- **N2 — NIT:** `memory_hub.on_percept_received` encodes comms `content` into substrate concept chunks. Concept names can surface later through `concept_context`, which is dormant today. This is out of O6's verbatim check. Note it in the O6 docstring as an unframed indirect carrier for item 4's scope.
- **N3 — NIT:** fix the placement reader's comment. The mapper compiles the placement TAIL, and the primary LOCAL lane goes through `load_llm_config(profile_override=...)`, already the `build_model_path` reader. Suggested label: "router-build placement tail (lane_backends._placement_entry_to_provider, LOCAL)".
- **N4 — NIT:** O6 uses channel `"sms"`, which is in `_KNOWN_CHANNELS`, so `PerceptContext.channel` is set. Production's channels (`"twilio:sms"`, `"api"`) are not in that set, so production gets `channel=None`. This does not matter to today's assertion, but S1's Twilio arm would match production here too.

## Verdict

One SHOULD-FIX (S1) is fail-open, though only latent: it bites only if item 4's fix renders `conversation_context`. There is no BLOCKER. By the stop rule, a fail-open SHOULD-FIX means **a delta round is owed for the S1 fold**: test-only and one gate, so scope that round to the O6 diff alone. If the owner instead files S1 as an item-4 follow-up, written into the O6 docstring, then no further round is owed for this fold.
