# HTTP client debugging

**Scope:** diagnosing outbound HTTP issues in Maxim after Plan 1 R1. Every outbound call in `src/maxim/` now flows through `maxim/utils/http.py`, which exposes structured events + metrics. This runbook is the first stop when:

- Peer → leader calls fail and you need to know why
- A tool fetch / model download hangs or errors
- Request/agent/session IDs aren't showing up where you expected
- A migration commit broke something

See also: [llm_routing.md](../architecture/llm_routing.md) for the authoritative layer-by-layer design.

## Enabling the trace

Two env vars, both off by default:

```bash
MAXIM_HTTP_TRACE=1       # bumps http_request events from DEBUG to INFO
MAXIM_LOG_FILE=/tmp/x.jsonl  # attaches a JSONL file handler via StructuredFormatter
```

When `MAXIM_LOG_FILE` is set, the root logger runs at DEBUG so the file captures everything — `stdout` still applies its verbosity-based filter (WARNING by default) so your terminal stays readable.

**Works for every entry path**: `maxim --sim`, `maxim doctor`, `maxim peer <subcmd>`, `maxim tunnel <subcmd>`. An early `configure_logging` call in `cli.py::main` ensures subcommand dispatch doesn't bypass the handler attachment — this wasn't true before PR #91 (commit `c8a07e9`), so if you're on an older branch and see an empty JSONL file, rebase.

## Reading the JSONL — field name gotcha

**The `StructuredFormatter` uses short single-letter keys for top-level fields.** A naive `jq 'select(.event=="http_request")'` returns nothing even though events are flowing. Use the actual keys:

| Short | Meaning |
|---|---|
| `t` | timestamp (float) |
| `l` | level (`I`/`W`/`D`/`E` one-letter) |
| `s` | source (logger name tail, e.g. `http`) |
| `e` | **event name** (`http_request`, `http_request_failed`, `startup_phase`) |

Everything else in the `data` dict is flattened to the top level (`endpoint`, `url`, `method`, `status`, `latency_ms`, `request_id`, `agent_id`, etc.).

### Canonical example — live capture against the RTX leader

```bash
$ rm -f /tmp/r1-smoke.jsonl
$ MAXIM_HTTP_TRACE=1 MAXIM_LOG_FILE=/tmp/r1-smoke.jsonl maxim peer version
Local:  v0.2.1 (e281ebf)
Leader: v0.2.1 (d875fb9)

$ cat /tmp/r1-smoke.jsonl
{"t":1776044...,"l":"I","s":"http","e":"http_request","endpoint":"_external","url":"https://maxim.dennyschaedig.com/v1/debug/version","method":"GET","status":200,"latency_ms":385.3,"request_id":"80c70a10d26d49029356c90c80df0210","agent_id":null,"session_id":null,"lane":null}
```

One outbound GET, 385ms latency, HTTP 200, request_id populated.

## Common queries

### Filter by event type

```bash
# All HTTP events (requests + failures)
jq -c 'select(.e | startswith("http_"))' /tmp/x.jsonl

# Just successful requests
jq -c 'select(.e == "http_request")' /tmp/x.jsonl

# Just failures
jq -c 'select(.e == "http_request_failed")' /tmp/x.jsonl
```

### Per-endpoint filter

```bash
# Only the leader endpoint
jq -c 'select(.e == "http_request" and .endpoint == "leader")' /tmp/x.jsonl

# Only external (tool/download/peer-cli) calls
jq -c 'select(.e == "http_request" and .endpoint == "_external")' /tmp/x.jsonl
```

### Latency percentiles for one endpoint

```bash
jq -r 'select(.e == "http_request" and .endpoint == "leader") | .latency_ms' /tmp/x.jsonl \
  | sort -n \
  | awk '{a[NR]=$1} END {print "p50:", a[int(NR*0.5)], "p99:", a[int(NR*0.99)]}'
```

Compare against `http.metrics_snapshot()["latency"]["leader"]` which exposes the same reservoir via the metrics API.

### Per-agent debugging (once callers bind context)

```bash
# All events for a specific agent
jq -c 'select(.agent_id == "npc-mother")' /tmp/x.jsonl

# Per-agent latency p99
jq -r 'select(.e == "http_request" and .agent_id == "npc-mother") | .latency_ms' /tmp/x.jsonl \
  | sort -n \
  | awk '{a[NR]=$1} END {print a[int(NR*0.99)]}'
```

**"My `agent_id` is null"** — that's because the caller isn't binding a `RequestContext` before issuing the HTTP call. This is normal for `maxim doctor` (probes are cluster-scoped, not agent-scoped) and for startup probes. It becomes populated when:

- The agent loop binds context before a completion call (Plan 2 R2b will wire this via `_normalize_request_context`)
- A sim orchestrator binds context per NPC turn (shipped in the percept/reaction work)
- Plan 3's `_MaximPeerBackend.complete_with_usage` builds context from the legacy `request_context` dict

Context binding is the **caller's responsibility**, not `utils/http.py`'s. The HTTP client reads from the contextvar and sends what it finds; if the contextvar is None it emits `agent_id: null` and that's correct.

## Diagnosing specific failures

### Pool exhaustion

**Symptom:** calls to a registered endpoint start timing out under multi-agent load.

```bash
python -c "from maxim.utils import http; import json; print(json.dumps(http.metrics_snapshot()['pool_exhausted_total'], indent=2))"
```

If `{endpoint: count}` shows a non-zero count, the connection pool is saturated. Default is `DEFAULT_POOL_PER_ENDPOINT = 10`. Raise it by re-registering the endpoint with a larger `max_pool_connections`:

```python
from maxim.utils import http
http.register_endpoint(http.HTTPEndpoint(
    name="leader",
    base_url="https://...",
    max_pool_connections=50,  # was 10
    # ... other fields ...
))
```

Note: re-registration closes the existing client and creates a new one. In-flight requests on the old client complete normally; new requests use the new pool.

### Cloudflare 403 / error 1010

**Symptom:** peer probes return HTTP 403 with a Cloudflare error page; leader is definitely up.

This was the 2026-04-12 incident (commit `8b52cbd`). Cloudflare Bot Fight Mode rejects any request without a recognizable User-Agent. Pre-R1, one urllib call site forgot to set `User-Agent: maxim-peer/1.0` and got silently 403'd.

Post-R1, User-Agent is set once on the `_external` endpoint's `default_headers` at registration. The Cloudflare path is structurally impossible to break now — every call through `http.fetch_url` inherits User-Agent from the endpoint. Verify:

```python
from maxim.utils import http
http._ensure_external_endpoint()
print(http.get_endpoint("_external").default_headers)
# {'User-Agent': 'maxim-peer/1.0'}
```

If a new Cloudflare 403 appears, it's almost certainly NOT a User-Agent issue — check Cloudflare's WAF rules + whether your tunnel config routes to the right upstream port.

### TimeoutPolicy tuning

**Symptom:** a specific endpoint routinely hits `HTTPTimeout` but completes just fine under curl.

`TimeoutPolicy` has three layered timeouts:

```python
TimeoutPolicy(connect_s=3.0, read_s=30.0, total_s=60.0)
```

- `connect_s` bounds DNS + TCP handshake
- `read_s` bounds per-chunk read latency
- `total_s` is the pool acquisition + total request budget

Slow leader under load? Raise `read_s`. Slow tunnel? Raise `connect_s`. Long-running inference? Raise `total_s`. The three factories available: `TimeoutPolicy()` (default 3/30/60), `TimeoutPolicy.fast()` (1/2/5), `TimeoutPolicy.long()` (5/120/600). Endpoints register with their own policy; call sites can override per-request via the `timeout=` kwarg.

### Header sanitizer rejections

**Symptom:** `http_header_rejected_total` metric is non-zero; a request that "should have worked" is returning `HTTPClientError` without reaching the wire.

```bash
python -c "from maxim.utils import http; import json; print(json.dumps(http.metrics_snapshot()['header_rejected_total'], indent=2))"
```

The field name in the output tells you which header was rejected (`X-Maxim-Agent-Id`, `Authorization`, etc.). Sanitizer rejects:

- CR / LF (log injection risk)
- Control characters (0x00–0x1F + 0x7F)
- Non-ASCII bytes (0x80+)
- Values > `MAX_HEADER_VALUE_LEN = 256`

Common cause: user-supplied content (agent name, session ID from a UI) contains a newline or Unicode character. Fix by cleaning the input at the caller, not by relaxing the sanitizer — the sanitizer is load-bearing security.

### "My event has no latency_ms"

`http_request_failed` events have `latency_ms` set to the time elapsed before the exception. `http_request` events have `latency_ms` set to the successful request duration. Both always populate the field. If it's missing, you're looking at a non-HTTP event (e.g. `startup_phase` has `duration_ms`, not `latency_ms`).

## Startup phase timings

`http.record_startup_phase(phase, duration_ms)` is the helper. **Reserved but not yet emitted** — no call sites invoke it as of R1. Plan 2's `detect_role()` or a future `cli.py` startup timer is the natural first caller. If you're adding one:

```python
from maxim.utils import http
import time

t0 = time.monotonic()
# ... do startup phase work ...
http.record_startup_phase("peer_config", (time.monotonic() - t0) * 1000)
```

Later, `jq 'select(.e == "startup_phase")'` lets you see where cold start is spending time.

## Rolling back a migration

Pre-R1, every call site imported `urllib.request` locally. Post-R1, that import is forbidden by a CI grep invariant:

```bash
grep -rn "urllib.request.urlopen" src/maxim/ | grep -v "utils/http.py"
# Expected: empty
```

Rolling back a migration commit means reverting the specific file to its pre-R1 state, NOT flipping a config flag — `utils/http.py` is a hard dependency now. Use `git log --follow src/maxim/<file>` to find the pre-R1 commit and `git checkout <commit> -- src/maxim/<file>` to restore.

If the CI grep fires after your revert, you missed a `urllib.request.urlopen` import. Either finish the revert or finish the migration; don't leave the tree half-migrated.

## Related commits + incidents

- **`8b52cbd`** (2026-04-12) — the Cloudflare User-Agent fix. One of the two bugs that motivated R1.
- **`d875fb9`** (2026-04-12) — the persisted-profile clobbering bug. The other motivating incident — prompted role-scoped persistence in Plan 2 R2a.
- **`14d955d`** (PR #88) — Plan 1 R1 step 1: `utils/http.py` created.
- **`5a143e5`** (PR #90) — Plan 1 R1 step 9: leader_proxy inference hot path migrated, last urllib site.
- **`c8a07e9`** (PR #91, pending) — dual-format logging fix for subcommand entry paths.
