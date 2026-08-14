# `Tool.cancel()` is a non-abstract no-op on the Tool ABC

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

- **[engineering] `Tool.cancel()` is a non-abstract no-op on the Tool ABC** (CC11, v1.0 freeze, 2026-04-29). Default implementation returns `None`; existing third-party `Tool` subclasses keep working without modification. Heavy built-in tools (`HttpFetchTool`, `InternetSearchTool`) override it to set a `threading.Event` (`self._cancelled`) which `execute()` checks at safe points — BEFORE side-effect accounting (rate-limit slots, request-time stamps) and BEFORE the network call. `cancel()` is called from a different thread than `execute()`; implementations must be thread-safe (the `threading.Event` pattern is the canonical shape) and must never raise. `execute()` clears the event at entry so the registered singleton tool stays usable after a prior cancellation; the cancel signal applies only to in-flight or imminent work, not to subsequent calls. **No 1.0 dispatch path calls `Tool.cancel()`** — it ships as forward-compat infrastructure for 1.1+ MCP-subprocess and async-cancel work. The `Tool.timeout` field is also not currently enforced by `runtime/executor.py`; both are independent reservations of contract surface for the same future cancellation pathway. Adding `@abstractmethod` post-1.0 is a breaking change for every third-party Tool subclass — do not do it. Regression guard: [tests/unit/test_tool_cancel.py::test_cancel_has_no_caller_in_executor_dispatch](tests/unit/test_tool_cancel.py) — if a future refactor wires `cancel()` into the executor, update that test and document the new caller here.
