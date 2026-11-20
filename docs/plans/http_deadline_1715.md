# An opt-in whole-request deadline for `fetch_url` (#1715, approach note v3)

Status: v3, BUILD (the v3 pass had two SHOULD-FIX findings, both fail-closed and covered by the planned tests, folded
here; the stop rule applies). v1 got DO NOT BUILD in the adversarial design pass (2026-11-20: two fail-open findings). v2 had no
DO-NOT-BUILD findings and four SHOULD-FIX findings, all folded here (§v2 pass). Owner decision 2026-11-20: #1715 option (a). Step 1 (rename `total_s` to `pool_s`) is done.

## Why

httpx 0.28.1 / httpcore 1.0.9 time the pool wait, the connect, and each socket read and write; nothing times the
whole request, and DNS is not timed. `tools/http_fetch.py` fetches URLs the MODEL chooses, so a hostile server can
hold a call indefinitely:
- a body drip of one byte per `read_s - ε`;
- a header drip, up to h11's 16 KiB incomplete-event limit;
- an endless stream of complete `100 Continue` responses, which httpcore loops over and h11 never limits.

`docs/user/stable_api.md` says the `TimeoutPolicy` "is the load-bearing hard interruption" for `HttpFetchTool`.
Today that is false.

**v1 was wrong**, and the design pass showed why:
- A check in `_read_capped`'s chunk loop sees nothing until httpx's `ByteChunker` has 64 KiB.
- No check outside `client.stream` can run between header reads.

The deadline must therefore live in the **network stream**, where every `recv` happens.

**Front-gate check:** this rides on infrastructure that already exists. The address-checked `_external*` clients
already install a custom httpcore network backend (`_checked_transport` sets the pool's `_network_backend`, pinned
by `tests/unit/test_http_public_only.py`). The new wrapper sits at that seam. The unrestricted `_external` client is
the one exception: it is a plain `httpx.Client` and needs the backend set after it is built (below). This adds no
thread, client or endpoint.

## What

- **`_REQUEST_DEADLINE: ContextVar[float | None]`** (an absolute `time.monotonic()` instant; default `None`).
- **`_DeadlineStream(httpcore.NetworkStream)`** wraps a stream.
  - `read`, `write` and `start_tls` each get `timeout = min(timeout, remaining)`. When nothing remains they raise
    `httpcore.ReadTimeout` (`WriteTimeout` for a write) before calling the inner stream.
  - `start_tls` returns a wrapped stream.
  - `close` and `get_extra_info` delegate.
  - With the contextvar `None` it is a pure pass-through, so every existing caller is byte-for-byte unchanged.
- **Every backend the `_external*` clients use returns `_DeadlineStream`s:**
  - `_AddressCheckingBackend.connect_tcp` wraps what it returns and clamps its connect timeout the same way.
  - The unrestricted `_external` (policy `"any"`) client stays a plain `httpx.Client(trust_env=...)`, so the
    `SSL_CERT_FILE`/`SSL_CERT_DIR`/netrc handling it has today is kept. After it is built, its default
    transport's pool and every non-`None` transport in `client._mounts` get a pass-through `_DeadlineBackend`. A
    `None` mount is a `NO_PROXY` host, which uses the default transport, so it is already covered.
  - The `_proxy_mounts` transports of the checked clients get the same backend. `HTTPProxy` and `SOCKSProxy` read
    `_network_backend` per connection, so a proxied request is bounded too.
- **`fetch_url(..., deadline_s: float | None = None)`** is a keyword, not a `TimeoutPolicy` field. A field would
  silently do nothing on `request`, `stream` and `download_to_file` (the push-no-ops-into-types class).
  - `fetch_url` sets the contextvar for the duration of the call and resets it in a `finally`.
  - A non-finite or non-positive value raises `ValueError`.
  - The pool timeout `fetch_url` passes httpx is clamped to `deadline_s` too. The pool wait is not stream I/O, so
    without the clamp a call could take `deadline_s + pool_s`.
  - A deadline timeout raises inside the existing `try`, so metrics and the structured log still run. The wrapper
    raises `_DeadlineExceeded`, a subclass of `httpcore.ReadTimeout` (or of `ConnectTimeout` when the budget runs
    out before or during a connect). It raises it both when no budget is left and when the inner call times out
    with the budget spent, so `_classify_httpx_error` can give it its own `fix_hint` ("the whole request passed
    its deadline"). It still classifies as `HTTPTimeout`. A `timeout=None` from httpcore counts as no bound: the
    remaining budget is used. httpx's `map_httpcore_exceptions` re-raises it as a plain `httpx.ReadTimeout` (or
    `ConnectTimeout` or `WriteTimeout`), keeping the original only as `__cause__`. So the classifier tests
    `exc.__cause__` for the deadline marker, BEFORE its generic timeout branch, the same pattern it already uses for
    `NonPublicAddressRefused`. Each of read, write and connect has its own marked subclass.
- **Pooled keep-alive connections** read the contextvar on each call, so a connection opened by an undeadlined call
  and reused by a deadlined one is bounded (and the reverse is unaffected). The contextvar is per thread and per
  task, so concurrent callers sharing a pool never see each other's deadline.
- **The bound it gives:** after the untimed DNS lookup, no socket operation and no pool wait runs past the
  deadline, so the call returns or raises within about `deadline_s`, plus one `getaddrinfo`, plus Python-side work
  between reads.
- **Residual: HTTPS through an `https://` proxy.** On that path `start_tls` returns httpcore's `TLSinTLSStream`,
  whose handshake and reads loop over raw `recv` calls, each with the full timeout. One clamped call can then last
  several times the remaining budget. This applies only with an operator-set `https://` proxy on a non-strict
  endpoint. `public_only` (`HttpFetchTool` with `block_private_ips`) never uses a proxy. Documented, not bounded.

## Callers

- **`HttpFetchTool`:** one absolute deadline per tool call, `timeout_s + 2.0` (the slack its old `total_s` had).
  The robots.txt hops and the main fetch each pass the *remaining* budget, so six robots redirects cannot multiply
  it. Once the budget is spent, the call refuses the way an unreadable robots.txt does today (fail closed).
- **`internet_search`:** both sites, `deadline_s = timeout_s + 2.0`. Its hosts are fixed, but its
  `stable_api.md` row claims the same shape.
- **`stable_api.md`:** the `HttpFetchTool` and `InternetSearchTool` rows say "whole-request deadline (soft only
  for DNS)".

## Inputs, and how each could lie

| Input | How it could lie | Handling |
|---|---|---|
| Body rate | a drip under `read_s` per byte | every `recv` is clamped to the remaining budget |
| Headers | a header drip, or an endless `100 Continue` flood | the same: header reads are stream reads |
| TLS handshake | a slow handshake | `start_tls` is clamped |
| Connect | N blackholed A records, each with the full `connect_s` | each attempt is clamped to the remaining budget |
| DNS | a stalling resolver | NOT bounded: documented residual (system resolver timeouts apply) |
| Clock | wall-clock jumps | `time.monotonic()` only |
| Other callers | a pooled connection shared across calls | the contextvar is read per call: `None` means pass-through |
| Threads and tasks | a deadline leaking across threads | `ContextVar`, reset in a `finally` |
| Proxy | the proxy drips | the proxy transports get the deadline backend; HTTPS-in-HTTPS is the documented residual |
| Pool | every slot busy | the pool timeout is clamped to `deadline_s` |
| A future backend | someone adds an `_external*` client or mount without the wrapper | a test asserts that every `_external*` client's default transport, and every transport in its mounts, has a deadline backend |

## Tests (red first, real local sockets)

- A body drip slower than the deadline in total, faster than `read_s` per byte: refused at about `deadline_s`, before
  the whole body arrives.
- A header drip, and a `100 Continue` flood: both refused at about `deadline_s`.
- No `deadline_s`: unchanged (the drip completes).
- A reused keep-alive connection, first without a deadline and then with one: the second call is bounded.
- The contextvar is `None` after a call, including after a raise.
- `HttpFetchTool`: robots redirects plus the main fetch share one budget (the deadline is passed through
  `check_robots_txt` and `_fetch_robots_txt`). Running out of budget there refuses the fetch without caching.
- A deadline expiry carries the deadline `fix_hint`; an ordinary read timeout does not.
- Every `_external*` client's pool backend wraps its streams.

## Not in scope

Bounding DNS; `request()`/`get()`/`post()`/`download_to_file`; the Reachy readback (#1706's post-call check already
bounds its reference's age). The existing dead `PoolTimeout` branch in `_classify_httpx_error` predates this
change and goes to a follow-up note on #1715.
