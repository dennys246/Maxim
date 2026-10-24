# Item 10 (offline): adversarial design review

Reviewer: adversarial design pass on `docs/plans/atonia_gate_approach_notes.md` §6, read-only, against the worktree
at `d9dffe6e`. Paths are under `src/maxim/` unless they start with `tests/`, `docs/` or `site-packages/` (the repo
venv: huggingface_hub 1.26.0, transformers 5.5.4, sentence-transformers 5.4.0, faster-whisper 1.1.1, ultralytics
8.3.248). Classes: **DO-NOT-BUILD** (the design as written fails open), **SHOULD-FIX** (a real gap; fold before
code), **NIT**. Each finding says whether it fails open or closed.

Where the note's factual claims hold: the six request functions exist (`request`, `fetch_url`, `stream_post`,
`download_to_file`, `raw_proxy_forward`, `raw_proxy_forward_streaming`); five of them use
`_EndpointRegistry.get_client`, and `download_to_file` builds `_download_client`. `SentenceTransformer` appears 3
times (encoder, semantic, clip_encoder). The urllib3 bypass is live (`tools/internet_search.py::_search_duckduckgo_lite`,
`pool.request("GET", ...)`). `hf_local_files_only` is reachable only as the env var `MAXIM_LLM_HF_LOCAL_FILES_ONLY`
(`models/language/transformers_backend.py::_get_bool_config`; `LLMConfig` has no such field).
`read_last_record` builds the record with `LaneDecisionRecord(**payload)`. `CONFIG_FORMAT_VERSION` is `"1.3"`. One
claim is wrong (finding 4).

---

## DO-NOT-BUILD

### 1. A refusal raised at a site whose caller catches it and falls back to a fetch path the guard does not see (fail-open)

The design raises `OfflineRefusal` at each site and expects the raise to propagate. Several sites sit inside
`except Exception` handlers whose fallback is **another fetch**, and that fetch goes through no Maxim code at all:

- `models/vision/ultralytics_engine.py::_ensure_weight_file` (L52–87): `attempt_download_asset(asset_name)` sits in
  `try … except Exception as e: logger.warning(...)`, after which the function **returns the bare `asset_name`**. The
  caller then runs `YOLO(seg_path)` (`UltralyticsEngine` L147) with a bare name such as `yolov8n-seg.pt`. Ultralytics
  resolves that name by downloading it itself, through `site-packages/ultralytics/utils/downloads.py::safe_download`
  (L323–331). That path uses `urllib` and, on a retry, **a `curl` subprocess**. It never touches `utils/http.py`, so
  the "HTTP guard is the backstop" argument does not hold. The note's "refusal of `attempt_download_asset`" ends up
  as a WARNING followed by the download anyway. `_load_pose_model`'s `YOLO(self._pose_model_path)` (L187) has the
  same shape.
- `models/vision/rtm_engine.py::_resolve_model_path` (L423–432): `except Exception as exc: logger.warning("Auto-download
  failed …")`, then `FileNotFoundError`. This one fails closed, but it turns the refusal into a generic "not found".
- `models/language/llama_backend.py::_fetch_missing_model` (L42–46): `except Exception as e: warn("Model download …
  failed")`, and by its own docstring the refusal is "not remembered". So every `complete()` re-enters
  `ensure_available`, re-raises and re-warns. It fails closed, but loudly per call.
- `similarity/encoder.py::_get_encoder` (L61–63) and `similarity/semantic.py::_get_or_load_model` (L101–103): a broad
  `except Exception` leads to `return None`, and then to the hash or lexical fallback. Nothing caches the result, so the
  note's "one WARNING per process" needs a dedupe that does not exist yet.
- `comms/channels/twilio_channel.py::TwilioChannel.send` (L42–51): `except Exception: logger.exception(...); return
  False` (see finding 6).

**Fold.**
(a) Refuse at the **last call before the library**, not at a helper the caller can step around. Under offline,
`YOLO(x)` runs only if `os.path.isfile(x)`, else `refuse`, and `_ensure_weight_file` refuses instead of returning the
bare name.
(b) Make the refusal impossible to hide. `offline.refuse(site, target)` appends to a process-level ledger (site,
target, time) **before** raising. The observer's report and `decision_log` read the ledger rather than relying on
the exception surviving its callers. A swallowed refusal is then still recorded and still sets
`status: "degraded"`.
(c) The red gate asserts the end state of each case: no file written, a ledger entry, and the socket spy (finding 10)
empty. It does not assert only that the site raised.
(d) Decide the exception's base class in the note. If it is not an `HTTPError` subclass, then `except _http.HTTPError`
handlers such as `tools/http_fetch.py` L539 and `hivemind/substrate_client.py` L121 do not misreport it as a
transport failure. More importantly, the peer auto-drain path (`peer/drain_routing.py`, `_maybe_schedule_auto_drain`)
must not treat a refusal as a dead node and persist drain state.

### 2. Proxies: a URL that passes the loopback check still goes to `HTTP(S)_PROXY`, and a loopback proxy passes a connect-level check (fail-open)

- The "any"-policy registry client, `utils/http.py::_EndpointRegistry.get_client` (L495), is a plain
  `httpx.Client(...)` with the default `trust_env=True`. `_ensure_external_endpoint` registers `_external` as "any",
  so `fetch_url`, `download_to_file`'s endpoint, and both `raw_proxy_forward*` variants use it. httpx then routes by
  `get_environment_proxies()`. In `site-packages/httpx/_utils.py::get_environment_proxies` (L30–75), loopback is
  **not** excluded unless `NO_PROXY` names it. So `http://127.0.0.1:8100/...`, an exact match for the registered
  local endpoint, is sent to whatever `HTTP_PROXY` points at, which can be off the machine.
- The checked clients add `_proxy_mounts` (L657–690), each an `httpx.HTTPTransport(proxy=...)`. If the "transport
  wrapper" is implemented at the network-backend level, the way `_AddressCheckingBackend` is, it sees the **proxy's**
  host and not the target. A loopback proxy (`HTTP_PROXY=http://127.0.0.1:3128`, an SSH SOCKS forward, a corporate
  agent) then passes "loopback exact match" while relaying everything. `HTTP_PROXY` is adversary 3 (the process's
  surroundings).

**Fold.** Under offline, every client this module builds uses `trust_env=False` and `mounts={}`, so it is strict and
uses no proxy. The check sits at the `httpx.BaseTransport.handle_request` level, on `request.url`, the same parsed
`httpx.URL` the transport dials. It is done once per request, so every redirect hop is covered. Pin a test with
`HTTP_PROXY` set to a loopback listener and assert that the request is refused, not proxied.

### 3. The claim "every fetch except the local LLM" is false for a CLI `llm.offline` run with host tools on (fail-open)

- `tools/filesystem.py` L1011 runs `["/bin/bash", "-c", command]`. `tools/code_tools.py` L165,
  `simulation/sandbox.py` L122 and `utils/sandbox_executor.py` L688 also run model-chosen commands. A model-chosen
  `curl https://…` has no literal for the census grep (`subprocess` with `pip/git/docker/curl`) to match.
- Worse, `tools/base.py::host_tool_env` (L506) filters to `HOST_TOOL_ENV_ALLOW` (L484), which **strips
  `MAXIM_OFFLINE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `YOLO_OFFLINE`**. A model-run
  `python -c "from sentence_transformers import …"` or `pytest` in a tool subprocess therefore runs online. That
  breaks V5's promise that offline "still covers sub-sims and spawned helpers".
- The observer is protected by item 2/3's ceiling. The CLI `llm.offline` path is not, and the note states the
  guarantee without that limit.

**Fold.** (a) Couple the latch to item 2: when it is engaged, every executor's effective ceiling drops `NETWORK`,
`COMMS` and anything `Effect.ALL` (bash, execute_file), whatever the `ToolPolicy`. (b) Add the offline variables to
`HOST_TOOL_ENV_ALLOW`. They are raise-only, so propagating them cannot weaken anything. (c) Otherwise, state the claim
as "Maxim's own fetches; model-chosen subprocesses are the OS sandbox's (AD5 layer 3)".

---

## SHOULD-FIX

### 4. Factual error: the "transport wrapper on every client" does not reach the cloud SDKs, and a router built before `engage` keeps its backends (fail-open)

- `models/language/anthropic_backend.py` L232 calls `Anthropic(api_key=..., timeout=...)` with **no `http_client`**,
  so the SDK builds its own httpx client.
- `models/language/openai_backend.py` L279–284 passes a checked client only when `address_checked_client` returns one.
  For `https://` with `allow_local_endpoints=True` the policy is `"any"` (`utils/net.py::base_url_address_policy`,
  L104–106), and `address_checked_client` returns `None` (L743–744). The SDK then builds its own client too.
- So the note's "the SDK's client [is] covered too" is false. The router predicate is the only control.
- The predicate is "decided at router build". The observer builder engages the latch mid-process. A Python-API user
  who already called a cloud-backed verb holds an `LLMRouter` whose `_backends` cache (`router.py::_get_backend_for_provider`,
  L370–436) still contains a live Anthropic client. So does any background thread on that router (LLMWarmup, the
  auto-undrain prober started in `runtime/lane_backends.py` L1213–1216).

**Fold.** Check `offline.is_offline()` per dispatch as well, in `_try_provider` (or `_invoke_backend`) against the
same predicate, and build-time filtering stays as the fast path. Alternatively, `engage` refuses while a non-local
router is alive; `call_accounting.register_router` already counts routers. Correct the note's sentence.

### 5. DNS is egress: refuse before name resolution, and ultralytics resolves at import (fail-open)

- `utils/http.py::_AddressCheckingBackend.connect_tcp` (L585–604) calls `socket.getaddrinfo` **before** it decides.
  A refusal built on that layer has already sent a DNS query, and a DNS query can carry data out.
- `site-packages/ultralytics/utils/__init__.py` L910 sets `ONLINE = is_online()` **at import**. `is_online` (L772–784)
  resolves `one.one.one.one` and `dns.google`. It is skipped only when `YOLO_OFFLINE` is the literal string
  `"true"`; `"1"` does not count.
- `ONLINE` also gates three other egress paths:
  - GA analytics on every predict (`ultralytics/utils/events.py` L64–70, called from `callbacks/hub.py` L88);
  - `check_requirements` auto-installing with pip (`checks.py` L478);
  - the PyPI version check (`checks.py` L293).
- The note sets none of the `YOLO_*` variables.

**Fold.** Do the request-level check of finding 2 before any resolution. `engage` sets `YOLO_OFFLINE=true` (the
literal) and ultralytics `SETTINGS["sync"]=False`. Add `ultralytics` to `engage`'s `sys.modules` check (finding 7).

### 6. The census misses whole classes of network code (fail-open against adversary 4, and against today's code)

The note's pattern list (`urllib3`, `requests`, `httpx.`, `socket.create_connection`, `subprocess` + literal binaries)
does not match these:
- `hardware/reachy/controller.py`: `ReachyMini(...)` (L289, a websocket to the daemon), `_resolve_mdns`'s
  `socket.gethostbyname` (L155–175, mDNS on the LAN), and `_start_ssh_tunnel` (L222–245, `ssh -L 8000:127.0.0.1:8000
  user@host`).
- The reachy_mini SDK carries its own stack: `io/ws_client.py`, `utils/discovery.py`, and a WebRTC central relay that
  defaults to `https://pollen-robotics-reachy-mini-central.hf.space` (`media/central_signaling_relay.py` L46).
- `comms/channels/twilio_channel.py`: SMS and voice to Twilio through its own `requests`, swallowed into
  `return False` (finding 1).
- Direct sockets:
  - `utils/reachy_diagnostics.py` L82, `socket.socket(AF_INET, SOCK_STREAM)`;
  - `doctor/checks.py` L2588 and `doctor/cli.py` L479, `socket.gethostbyname(host)`;
  - `doctor/checks.py` L3557, a connect to the robot.
- The model-chosen shell (finding 3), and `tunnel/daemon_spawner.py` (cloudflared).

**Fold.** Census three sets:
(a) an **import roster** of network-capable third-party modules (`reachy_mini`, `twilio`, `ddgs`, `anthropic`,
`openai`, `huggingface_hub`, `transformers`, `sentence_transformers`, `faster_whisper`, `ultralytics`, `llama_cpp`,
`keras`/`tensorflow`, `urllib3`, `requests`), each pinned with its offline handling;
(b) every `socket.` attribute use (`socket.socket(`, `gethostbyname`, `getaddrinfo`, `create_connection`) and `asyncio`
connection opens;
(c) **every** `subprocess`/`os.system`/`Popen` site, not only those naming a binary, each classified.
State in the note whether a LAN robot or mDNS counts as "leaving the machine". It does on the wire, so I recommend it
counts.

### 7. Add a socket-level latch as the backstop, not just a census (fail-open against adversary 4)

A grep census pins today's call sites. A runtime latch enforces the property for all of them, and
`tests/network_guard.py` already shows the mechanism works in this codebase. It patches `socket.socket.connect`,
`connect_ex`, `getaddrinfo` and `gethostbyname` (`install`).

**Fold.** `engage` installs the same guard in production: refuse any non-loopback TCP connect and any non-literal
name lookup, with the one exact local-endpoint allowance. It records into finding 1's ledger. It covers HF's own
httpx session, the Anthropic SDK, urllib3, ddgs, twilio and anything added later. This is consistent with §0, which
is about adversarial in-process Python, not about library code. C-level sockets (llama.cpp, gstreamer, ctranslate2)
stay with the OS sandbox, and the note should say so. Keep the census as the explanation of each site, with the latch
as the enforcement.

### 8. "Loopback" is not "local" (fail-open on the CLI path; moot for the observer under V4)

- `utils/net.py::is_loopback_url` (L74–91) accepts `localhost`. That name is resolved at connect time through
  `/etc/hosts`, which is adversary 3, so it contradicts the note's "decided on the literal, not on DNS".
- A loopback literal can be a forwarder. Maxim itself creates one: the Reachy controller's `ssh -L 8000:…` (finding
  6). Others are any `socat`, an `ssh -L`, or a cloudflared access forward.
- The note never says where "the configured local server" comes from. If it is a config string, the allowance is
  adversary-3 controlled.

**Fold.** Accept only IP literals (`127.0.0.1`, `::1`), compared on `httpx.URL.host` and `port` with the default
port normalized. Take the allowed endpoint from Maxim's own spawner record (pidfile and port of a llama-server Maxim
started) rather than from config. Write the guarantee as "Maxim dials only that loopback port". The parse
differentials I tried (`\@`, `#@`, `%2f@`, userinfo, `127.0.0.1.`, `::ffff:127.0.0.1`) parse identically in
`urllib.parse` and `httpx.URL` today. Deciding on the `httpx.URL` the transport dials removes the question entirely.

### 9. Config: an older reader ignores `llm.offline` (fail-open); a running server never sees the setting (fail-open)

- `runtime/config_loader.py::_check_format_version` (L1240–1316) and the parse at L1715–1725 **tolerate an unknown
  key, with a WARNING**, in a file whose minor version is newer. A 1.3 build reading a 1.4 config with
  `llm.offline: true` therefore runs online. Typical cases are a pinned venv, a peer box, or a downgrade. This is the
  documented #856 behavior and cannot be changed in 1.3, but the note does not acknowledge it.
- `XDG_CONFIG_HOME` (`config_loader.py` L617) moves the config file, so an empty one means offline is off. That is
  adversary 3, but on the CLI path only.
- "Read once at CLI and api startup" has no counterpart in the Python API: `import maxim` loads lazily, and a user
  can construct `LLMRouter` directly. After #1411, a running `maxim serve` re-reads config on the next router build,
  but the latch read does not. `maxim config set llm.offline true` therefore takes effect only after a restart, and
  nothing tells the operator.

**Fold.**
- Each router build, and the first `is_offline()` call, also reads `llm.offline`, raise-only. That removes the need
  for a "startup" hook.
- `maxim doctor` prints the latch state and its source.
- The note states that offline does not survive a downgrade below 1.4.
- The rest of the plan is consistent with `docs/agents/persistence-config.md` §3:
  - a minor bump for any field on a config section;
  - `"1.4"` entries appended to both fixtures (append-only lint);
  - `downgrade_config` moving `llm.offline` to the sidecar;
  - `restore_preserved` treating it as security-relevant because it is not in `_NON_SECURITY_FIELDS`.
- Add a `config_resolution_by_version.json` case for `llm.offline: false` written explicitly. Under the 1.2+ presence
  rule, an explicit `false` is a value the operator set, and the observer must still force it on.

### 10. The red gate as written passes for no reason in-process, and "the network guard was never reached" is only partly observable (fail-open in the test)

- `tests/conftest.py` L44–50 sets `HF_HUB_OFFLINE=1` at import. By the time the red gate runs, earlier tests in the
  worker have usually imported `huggingface_hub`, and `constants.HF_HUB_OFFLINE` was frozen at that import
  (`site-packages/huggingface_hub/constants.py` L202). Deleting the env var then changes nothing. Separately,
  `engage`'s own `sys.modules` refusal would fire because of the earlier imports.
- "The guard was never reached" can be implemented as `len(network_guard.attempts)` unchanged. But
  `tests/network_guard.py::_blocked_connect` and `_blocked_name` **do not record** loopback connects or IP-literal
  lookups. A refusal that happens after an IP-literal `getaddrinfo`, or after a loopback proxy connect (finding 2), is
  invisible to it.
- The CI netns (`scripts/ci_netns.sh`) blocks everything anyway, so a case can only be green for Maxim's reason if the
  spy proves it.

**Fold.** Run the red gate in a **fresh subprocess** (`sys.executable -c`) that installs `tests.network_guard` itself
and imports nothing before the case. Add a spy that wraps `_real_connect` and `_real_getaddrinfo` and records
**every** call, loopback and literal included. Assert three things: the `OfflineRefusal` ledger entry, zero spy
records (except the allowed endpoint), and no file written.

### 11. `engage`'s `sys.modules` refusal is too strict for HF and misses others (fail-closed for HF; fail-open for ultralytics)

- In huggingface_hub 1.26, every Hub request goes through `site-packages/huggingface_hub/utils/_http.py::hf_request_event_hook`
  (L250–262). The hook calls `constants.is_offline_mode()`, which returns the module global **at call time**.
  transformers 5.5 reaches `is_offline_mode` through `transformers/utils/hub.py` (L40, L363).
- So, after import, setting `huggingface_hub.constants.HF_HUB_OFFLINE = True` and probing `is_offline_mode()` works.
  Refusing instead makes the observer impossible to build in any API process that imported transformers first, which
  is a common library-user state.
- Two things are missed. `ultralytics` (`ONLINE` is frozen at import; finding 5) is not checked at all.
  `constants.ENDPOINT` (L68) is also frozen at import, so the note's `HF_ENDPOINT` pin does nothing after import.
- The note's "unroutable" `HF_ENDPOINT` is unspecified. A `.invalid` name still leaks a DNS query. A loopback port may
  have a listener.

**Fold.** For each already-imported library, set its offline constant and probe it, refusing only if the probe
disagrees. Refuse for ultralytics, because `ONLINE` cannot be re-derived safely. Pin `HF_ENDPOINT` to an IP-literal
discard address, or drop the pin and rely on the offline flag together with finding 7's latch.

### 12. The latch's env read and test reset (fail-open in one edge; fail-closed in tests)

- "`MAXIM_OFFLINE` truthy at first read" means a raise after the first read is ignored, for example by a harness that
  sets it after `import maxim`.
- The latch is process-global and one-way. Once one test engages it, every later test on that xdist worker is
  offline, and the conftest env scrub does not reset the in-process latch.

**Fold.** `is_offline()` re-reads the env on every call. That is raise-only, so it can never disengage. Provide a
test-only `_reset_for_tests()` that only an autouse conftest fixture calls. The census fails any `src/` caller of it.

---

## NIT

### 13. The router predicate and the transformers fold contradict each other (n/a)

The predicate admits only in-process `llama_cpp`, plus loopback `maxim_peer`/openai-compatible. That refuses
`pytorch`/`transformers` lanes under offline, yet §6 also folds `local_files_only` and forced
`trust_remote_code=False` into `transformers_backend`. Pick one: either admit `pytorch` with `local_files_only`
(`llm.backend` is `Literal["llama_cpp", "pytorch"]`), or mark the transformers fold "defence in depth, unreachable
offline". Also, `MAXIM_LLM_HF_TRUST_REMOTE_CODE` (`transformers_backend.py` L35, through `_get_bool_config`) must lose
to offline.

### 14. Cross-version `decision_log` reads fail quietly (fail-closed)

`runtime/decision_log.py::read_last_record` (L282) catches `TypeError` and returns `None`. A 1.3 build reading a 1.4
record with `offline` shows "no record" instead of the record. `~/.maxim` is shared across worktrees and versions.
Records are written only at `MAXIM_PROVENANCE_VERBOSITY>=1`, so "decision_log stamps it" is not a guarantee at
verbosity 0. The observer report should read the ledger from finding 1 instead.

### 15. Close the registry's clients on engage (fail-closed)

Clients cached in `_registry._clients` before `engage` lack the wrapper. None of them follow redirects (only
`download_to_file` and the SDK client do), so the function-top guard covers them. `engage()` should still call
`close_all()`, so that every client after engage is a wrapped one and no pre-latch pooled keep-alive connection
survives.

### 16. Ingress is out of scope (n/a)

The leader binds `0.0.0.0` (`runtime/local_server_spawner.py` L9, L464). Offline is egress-only, and the note should
say that a leader under `llm.offline` still serves inbound peers. The remote-update refusal is already named.

---

**Verdict:** DO-NOT-BUILD as written. Three fail-open holes must be folded first: refusals swallowed into
non-Maxim downloads (ultralytics `YOLO(name)` → urllib/curl), env proxies carrying even "loopback" requests off the
machine, and host-tool subprocesses that strip the offline variables. With those and SHOULD-FIX 4–12 folded, chiefly
a socket-level latch and a refusal ledger, the design is sound.
