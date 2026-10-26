# Red gates: delta round 3 (Architecture and Wire integrity, narrow)

Scope: the two folds in `447fbcf8` that answer delta2's fail-open findings: Architecture S1 (the reset guard and
aliased imports) and Wire S1 (O6 over the Twilio producer). I read the code and ran it. Nothing else in the commit
is in scope.

## 1. `test_no_caller_of_reset_for_tests_outside_tests` / `_reset_references`

**Runs today:** PASS on the real tree (`src/`, `scripts/`). Nothing outside `tests/` references the name.

**Probe** (a scratch tree fed to `_reset_references`):

| Form | Caught |
|---|---|
| `from maxim import _offline as o; o._reset_for_tests()` | yes (Attribute) |
| `importlib.import_module("maxim._offline")` + `getattr(m, "_reset_for_tests")()` | yes (str Constant) |
| `from ._offline import *` / `from maxim._offline import *` | yes (star branch) |
| `from maxim._offline import _reset_for_tests [as r]; r()` | yes (alias) |
| `import maxim._offline; maxim._offline._reset_for_tests()` | yes |
| `operator.attrgetter("_reset_for_tests")`, `vars(m)[f"_reset_for_tests"]` | yes (Constant inside a JoinedStr) |
| `getattr(m, "_reset" + "_for_tests")` | no: dynamic, out of model per §0 |
| `importlib.reload(maxim._offline)`, `m._ENGAGED = False` | no: not a reset reference (see N1) |

The in-test probe proves the three new branches (alias, string, star) plus a clean negative. It does not include the
module-alias attribute form, but that branch predates the fold.

**False positives.** None on today's tree. Two are latent, and both fail closed:
- The match is on the name, not the module. `maxim.runtime.llm_call_registry` defines its own `_reset_for_tests`, so
  a future `src/`/`scripts/` caller of the registry reset would trip this guard.
- The str-Constant branch matches docstrings. If step 0b's `src/maxim/_offline.py` names `_reset_for_tests` in a
  docstring, the guard goes red on the implementing PR. The `def` is not matched.

Both are loud, so they are NITs (N2).

## 2. O6 over two producers (`tests/unit/test_atonia_open_items_other.py`)

**Producers are the real ones.** `receive_inbound` has exactly three production callers:
- `comms/api.py::inject_command`, with `channel="api"` and no metadata. The test's `_inbound_via_api_command` call
  is the same.
- `TwilioChannel.receive_sms`, reached from `POST /api/twilio/sms`. The test calls the real method.
- `TwilioChannel.receive_voice` (see N3).

The test builds the channel with `__new__`, which skips only the twilio `Client`. `receive_sms` reads only
`conv_manager` and `gateway`. Production builds `ConversationManager(bus=bus, nac=nac)`. `nac` is read only in
`_archive`, so `ConversationManager(bus)` is equivalent on this path.

**Bus path is real.** `CommunicationGateway(bus=AgentBus())` publishes to a real bus, which reaches
`MemoryAgent._on_percept` (subscribed in `__init__`), then `_comms_messages`, then `build_context()`, then
`LLMWorker._prompt_builder.build_prompt`. `MemoryAgent` drops `percept.metadata`, so a fix that renders
`conversation_context` has to plumb it through `build_context`, and the gate exercises that plumbing.

**Fails today at the intended assertion.** With `--runxfail`, both arms raise `AssertionError: ... does not read
inbound comms at all`, after every `require`.

**Flip probe.** I monkeypatched `PromptBuilder.build_prompt` to append each rendering:

| Rendering | api_command | twilio_sms |
|---|---|---|
| `frame_tool_output` around each `comms_messages` content | flips | flips |
| the same content, raw | red (unframed) | red (unframed) |
| framed content + raw `conversation_context` | flips (no cc on this producer; correct) | **red** (unframed) |
| framed `conversation_context` only | red (text absent; correct, `/api/command` text is not rendered) | flips |

So a correct fix flips both arms, and an unframed carrier keeps the affected arm red. Wire S1 is closed.

**Gate, last hop.** The gate builds the `LLMRequest` by hand. In production, `agent_loop` (`submit_context`) also
passes caller-assembled text fields: `triggering_input`, `deferred_inputs`, `context_pool_text`,
`conversation_history_text` and `prefetch_context`. I checked whether any of them carries comms text today:
- `ContextPool.add_percept` renders only transcript, detections and `cli_input`, and only for agent_loop's
  observation percepts.
- `triggering_input` comes from `cli_input`.

None does, so "fails today" holds for the shipped path. The consequences are N4 and N5.

## 3. Notes v6.5, §1A.9 invariant 3

The text is close. The reset guard is now cited, and the residuals (pre-latch connections, native networking) are
stated honestly. It still says slightly more than the guard does:
- "No non-test caller can disengage it" is a stronger claim than "no static reference to `_reset_for_tests` in
  `src/` or `scripts/`".
- `importlib.reload(maxim._offline)` re-executes the module in the SAME module dict, which resets module-level latch
  state that the installed hook reads. Direct writes to private state behave the same way. The guard covers neither.

Both are in-process code, which is out of model per §0, so this fails closed as a claim and is a NIT (N1).

## Findings

No BLOCKER. No SHOULD-FIX.

- **N1 (wording, docs):** invariant 3's "no non-test caller can disengage it". Suggested replacement: "no reference
  to `_reset_for_tests` in `src/` or `scripts/` (attribute, alias, star import, string); dynamically built names,
  `importlib.reload` and private-state writes are in-process code, out of model (§0)". Optional, for step 0b: keep
  the latch state where a reload cannot reach it, such as a closure captured by the hook, or a flag the hook checks
  that is never reassigned at module top level after first import.
- **N2 (fail-closed):** the name-based match collides with `llm_call_registry._reset_for_tests` and with docstrings.
  For step 0b, say in the guard's docstring that `_offline.py`'s own docstring must not spell the name, or exclude
  `Expr`-docstring Constants.
- **N3 (fail-closed coverage):** `receive_voice` (`twilio:voice`) has the same shape as `receive_sms`. A fix that
  branches on channel could frame SMS and leave voice raw. A third arm is about four lines.
- **N4 (fail-closed):** a correct item-4 fix routed through agent_loop's request fields (for example comms text
  passed framed as `triggering_input`) cannot flip this gate, because the hand-shaped request leaves those fields
  empty. The docstring should name the seam the gate pins: the fix lands at `build_context` → `PromptBuilder`.
- **N5 (latent fail-open, needs two independent renderings):** if a fix frames the text at `PromptBuilder` and a
  separate change also routes it raw through a caller-assembled field, the gate XPASSes. That needs a second
  unframed route through code that today never touches comms. It is the same class as delta2 Wire N2 (substrate
  concept chunks). Add one docstring sentence listing those fields as outside the verbatim check.

## Verdict

**STOP: no further round owed.** Both delta2 fail-open findings are closed and proven by probes. Every remaining
finding is a NIT that fails closed or needs wording, plus one latent double-rendering case (N5) in an
already-recorded class.
