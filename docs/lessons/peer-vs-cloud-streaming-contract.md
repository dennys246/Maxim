# Streaming contract difference between `_MaximPeerBackend` and `_OpenAIBackend` is intentional

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] Streaming contract difference between `_MaximPeerBackend` and `_OpenAIBackend` is intentional:** `_OpenAIBackend._stream_response` silently collects partial output when a chunk iteration raises mid-stream (cloud providers' first-token-latency UX expects "got some tokens" > "nothing"). `_MaximPeerBackend._stream_response` raises `BackendDown` on any mid-stream failure (malformed JSON chunk, `HTTPConnectionError` during `iter_lines`, or empty content) so the router can fail over to a different provider. These are different contracts for different backends, not a bug in either. Do NOT "fix" the peer backend to match the cloud one — that re-introduces the class of silent-partial-output bugs Plan 3 was designed to eliminate. Regression guards: `test_streaming_mid_stream_malformed_chunk_raises_backend_down` + `test_streaming_connection_error_mid_stream_raises_backend_down` + `test_streaming_empty_content_raises_backend_down`.
