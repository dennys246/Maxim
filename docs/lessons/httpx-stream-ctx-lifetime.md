# httpx stream contexts must outlive their consumers

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] httpx stream contexts must outlive their consumers:** Calling `ctx = client.stream(...)` then `raw = ctx.__enter__()` opens a live HTTP stream, but Python GC will call `ctx.__exit__()` (which closes the stream) as soon as `ctx` goes out of scope. `raw_proxy_forward_streaming()` in `utils/http.py` originally returned `StreamingResponse(_raw=raw)` without storing `ctx`. The function returning caused `ctx` to fall out of scope; GC closed the stream before `_proxy_request` could call `iter_bytes()` — resulting in `httpx.StreamClosed`, 0 chunks forwarded through Cloudflare, and `JSONDecodeError` on the Mac peer. Every inference call silently returned an empty body. Fix: `StreamingResponse._stream_ctx: Any | None = None` holds the context alive; `close()` calls `_stream_ctx.__exit__(None, None, None)` in cleanup. **Rule: any code that enters an httpx stream context manager manually via `.__enter__()` MUST store a reference to the context manager that lives at least as long as the consumer reading the stream.** The `_stream_ctx` field in `StreamingResponse` is load-bearing — do not set it to `None` or remove it. Regression guard: [src/maxim/utils/http.py](src/maxim/utils/http.py) — `StreamingResponse._stream_ctx` field declaration + `close()` enters the context manager's `__exit__`; structural enforcement via the dataclass shape.
