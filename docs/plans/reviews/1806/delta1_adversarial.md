# Device registry (#1806): delta round 1, adversarial lens (inputs and trust)

Reviewer: adversarial lens (inputs and trust), delta round 1, read-only, 2026-11-25, v2 @ f0369379 (code base origin/main b7380ee7; reachy-mini SDK 1.8.3 in `.venv`). No network, no robot, no probes run; every claim below was read in code.

**Verdict: BUILD after SHOULD-FIX.** v2 closes the round-1 DO-NOT-BUILD findings (A1–A4). The §4.3 SDK claim checks out. What remains are gaps in the new verbs and in the window inside `ReachyMini.__init__`, and five of them are fail-open, so the stop rule is **not** met (see the end).

## Disposition of round-1 findings

| A# | v2 status | Note |
|---|---|---|
| A1 | FOLDED | The SDK claim is verified (§1 below). Placement is looser than it should be (N8), and the pre-init window depends on A8/A14 |
| A2 | FOLDED | The loader refuses `launch.operational_mode`, and `_validate_operational_grant` runs on typed flags. Residual (round 1 too): `launch.llm` can still pick a cloud model with nothing typed. The plan block should print each value's source |
| A3 | FOLDED | Typed `pin:`, the registry refuses a non-supporting class, selfy/agentic_runtime propagate, and the stable API is labelled unpinned. Residuals: N3 (a deleted line downgrades silently) and N11 (the class flag is a declaration) |
| A4 | FOLDED | Discovery is out of resolution. `relocate` details are in N10 |
| A5 | PARTIAL | The cwd fallback is removed, but the data-home file is still agent-writable whenever cwd contains it (N2), and the relative-`MAXIM_DATA_HOME` behaviour is unspecified (N6) |
| A6 | FOLDED | An explicit name is exact. The note should say that `MAXIM_DEVICE` and `default:` count as explicit (N7) |
| A7 | FOLDED | IPv4-only, the refuse list, port 8000, trailing dot, the 4-candidate cap. Canonical form and `hostname:` are in N9 |
| A8 | PARTIAL | "Proxies disabled" has no mechanism in `fetch_url`. The WS proxy pre-check was dropped, and the SDK's `requests` media POST is a third proxy path inside `__init__` (N4) |
| A9 | FOLDED | Sim flags are checked in the pre-filter and on the session, null never matches, the format is checked, `repin` exists, and the cache is documented. The daemon value is `hexdigest()[:16]`, so lowercase matches. `repin`'s spec is in N1 |
| A10 | PARTIAL | An unpinned entry no longer caches or browses, but `embody` still runs it with only a label, against §3a's "never targets … an unverified device". Round 1's `--unpinned` gate is missing (N3) |
| A11 | FOLDED | The verified address and pin are carried as typed args, and a refusal exits non-zero. The note contradicts itself on unnamed launches (N7) |
| A12 | PARTIAL | The sanitiser drops round 1's bidi/format-char stripping and the TXT key shape checks, and "never reaches a prompt" still names no enforcing check |
| A13 | FOLDED | The ambiguous empty result is stated, and check layer 1 probes the permission |
| A14 | PARTIAL | WebRTC signalling to `wlan_ip` is opened **inside** `ReachyMini.__init__`, before step 3 and before step 5's comparison (N5) |
| A15 | PARTIAL | `MAXIM_DEVICE` is named and scrubbed, and `MAXIM_ROBOT_TIMEOUT` is fixed. The relative-`MAXIM_DATA_HOME` outcome is unspecified (N6). `MAXIM_REACHY_HOST` gets a row, but v2 does not say whether it feeds the resolver or is removed (it must not become a fourth address source) |
| A16 | FOLDED | `default:` lives in robots.yaml (D2′), consistently in §3.1, §7 and §9 |

## New findings (introduced or exposed by v2)

| # | Finding | Severity | Direction | Covered class? |
|---|---|---|---|---|
| N1 | `repin` adopts whatever answers at an address that just failed the pin, and the mismatch fix hint points the operator at it. No uniqueness check, no TTY rule, no context beyond the two ids | SHOULD-FIX | fail-open (operator-mediated) | **No** (new verb) |
| N2 | `<data_home>/robots.yaml` is inside the agent's cwd write scope whenever cwd ⊇ data home (`maxim` run from `~` or `~/.maxim`). A singularity or approved-active agent can delete `pin:`, or repoint `host:`/`pin:` | SHOULD-FIX | fail-open | Yes (A2/A5) |
| N3 | A missing `pin:` line means UNPINNED, and `embody`/`--device` proceed on an unpinned entry with only a WARNING. A hand or agent deletion therefore silently brings the mix-up back | SHOULD-FIX | fail-open | Yes (A3/A10) |
| N4 | `fetch_url(address_policy="private")` cannot disable proxies (the private endpoint is `strict=False`, and the proxy wins per #921). The WS proxy check was dropped. `requests.post(/api/media/release)` in `__init__` follows its own env-proxy rules | SHOULD-FIX | fail-closed for motion (step 3 refuses); side effects before it | Yes (A8) |
| N5 | `ReachyMini.__init__` opens WebRTC to the status-reported `wlan_ip:8443` (and `camera.open()`) before step 3. v2 only compares `wlan_ip` afterwards, as a WARNING | SHOULD-FIX | fail-open (egress outside the address policy and the latch) | Yes (A14) |
| N6 | The outcome for a relative `MAXIM_DATA_HOME` is unspecified. A fallback to `~/.maxim` is an isolation escape onto real hardware. `find_config_file`'s `except Exception` branch must raise, not return "no entries" | SHOULD-FIX | fail-open if read as a fallback | Yes (A5/A15) |
| N7 | §4.3 step 4 ("raise, never headless; selfy/registry propagate") contradicts D13 ("unnamed legacy launch keeps headless"). It is also undefined whether `default:`/`MAXIM_DEVICE` count as named | NIT | fail-closed | Yes (A6/A11) |
| N8 | Step 3 is placed "before the first motion/wake/torque command", which is looser than "immediately after `ReachyMini(` returns". `get_status` must be `wait=True` with a bound, and a `TimeoutError` must refuse | NIT | fail-closed | Yes (A1) |
| N9 | The IPv4 literal must be canonical (`str(ipaddress.IPv4Address(x))`, which refuses inet_aton octal/hex/short forms). The new `hostname:` field has no defined role in §4.2's candidate list | NIT | fail-closed | Yes (A7) |
| N10 | `relocate`: the Python form has no confirmation, the CLI's non-TTY behaviour is unstated, and the note does not say whether the candidate is pin-verified and hygiene-checked before the write, or which file it writes | NIT | fail-closed (connect re-verifies) | Yes (A4) |
| N11 | `supports_identity_pin = True` is a declaration, not enforcement. The registry already needs the live controller's verified identity for its reuse branch, so it should check that value after every connect | NIT | fail-open for buggy operator-installed plugins | Yes (A3) |
| N12 | `MAXIM_DEVICE` is inherited by every child (`utils/child_env.py::child_env` copies the full env), so a harness `--mode` sub-run becomes a named-device robot run | NIT | fail-closed (verified device only) | Yes (A15) |
| N13 | Name collision: the SDK's `utils/hardware_id.py::get_pin` is a pairing PIN derived from the raw serial, while Maxim's `pin:` is the hashed id | NIT | n/a | n/a |

---

## 1. The §4.3 SDK claim: verified, and what runs before step 3

**Verified.**
- `io/protocol.py::DaemonStatus` carries `hardware_id: Optional[str] = None`, `simulation_enabled: Optional[bool]` and `mockup_sim_enabled: Optional[bool]`.
- `daemon/daemon.py::Daemon._publish_status` sends `self.status().model_dump_json()` at 1 Hz through `ws_server.publish_status`. That is the same `_status` object REST serves; `hardware_id=get_hardware_id()` is set in `Daemon.__init__`, and `_setup_backend` sets the sim flags.
- `io/ws_client.py::WSClient._dispatch` stores it in `_last_status`.
- `WSClient.get_status(wait=True, timeout=5.0)` **clears** `status_received` and waits for the next push, so a call after `__init__` returns a fresh status from this socket.

So the session re-check is real and does bind to the motor socket.

**But `ReachyMini.__init__` sends commands before step 3 can run** (`reachy_mini.py::ReachyMini.__init__`, in order):
1. The WS connect: `_initialize_client` → `_connect_single` → `WSClient(host, port)` (websockets `connect(uri)`, proxy-honouring).
2. `self.set_automatic_body_yaw(automatic_body_yaw)`, which sends `SetAutomaticBodyYawCmd` over the WS. Maxim passes `False`: a config write, not motion.
3. `_configure_mediamanager`, which calls `client.get_status()`. Then:
   - under `no_media` with daemon media present: `release_media()` → `WSClient._media_request`, a **`requests.post`** to `http://{host}:{port}/api/media/release`. `requests` follows its own env-proxy and NO_PROXY rules.
   - under `default`/`auto` on a network connection: `MediaManager(signalling_host=daemon_status.wlan_ip or "localhost")` → `MediaManager._init_webrtc` → `find_producer_peer_id_by_name(wlan_ip, 8443, …)` and `camera.open()`.

**Is the pre-filter adequate for these?** For the motor axis, yes: none of them moves a joint, and step 3 still refuses before Maxim's first command. For the side effects (a media release on the wrong daemon, WebRTC egress), the pre-filter is the only protection. It is adequate only if all three of these hold:
- (a) the dial target is the exact literal the pre-filter probed. v2 does this: a forced `network` mode, a literal, port 8000.
- (b) no proxy applies to the `ws://` URI or the `requests` POST. Not ensured, see N4.
- (c) `wlan_ip` is vetted before `ReachyMini(`. Not ensured, see N5.

The note should state this residual explicitly, as round 1's fix step 4 asked.

## 2. Findings

### N1. `repin` can persist the exact mix-up the pin exists to stop (SHOULD-FIX, fail-open, new class)

§3.1 says `repin` is an "operator-confirmed identity change … old→new shown", and A9's fold names it in the mismatch fix hint. The common cause of a mismatch is not an audio-board swap. It is a DHCP reassignment that put robot B, or another daemon, at rosie's `host:`. The operator gets `DeviceIdentityMismatch`, the hint offers `repin`, the operator confirms two opaque 16-hex strings, and from then on the entry `rosie` (its `body:` ref, its `launch:` defaults, its name) drives robot B. The new id comes from the same unauthenticated endpoint that just failed.

Fix:
- The mismatch hint leads with `relocate` ("the robot may have moved"). `repin` comes second, as "only after replacing the audio board".
- `repin` shows the new daemon's sanitised `robot_name`, `wlan_ip` and version alongside the ids.
- It refuses a null id, a sim or mockup daemon, and **an id already pinned by another entry**. Apply the same uniqueness rule at `add`.
- The CLI refuses on a non-TTY without an explicit flag.
- The Python `repin(name, *, expected_new=…)` requires the caller to pass the new id, so a programmatic repin cannot "accept whatever answers".

### N2. The data-home robots.yaml is still agent-writable (SHOULD-FIX, fail-open, A2/A5 class)

`utils/filesystem_policy.py::check_write_permission` grants `CwdAccessLevel.FULL` (singularity) and `SUPERVISED` (active, approval) writes to any path under cwd. It has no exclusion for `data_home()`. The removed cwd fallback (D11) closes the planted-`./robots.yaml` case. It does not close the case where cwd contains the data home, and `maxim` launched from `~` is the ordinary case. An agent can then delete `pin:` (N3), or set `host:` and `pin:` to another daemon. §6's row claims this vector closed. The `launch.operational_mode` refusal does limit the damage: capability cannot be raised.

Fix: `check_write_permission` refuses agent writes to `data_home()/robots.yaml` and `data_home()/util/device_state.json` at every access level, ideally to the whole operator-config set (config.json is the same pattern; "Filing is not fixing" says fix the sibling sites in the same PR). Add a test that runs with cwd = data home under singularity.

### N3. A deleted `pin:` line downgrades silently (SHOULD-FIX, fail-open, A3/A10 class)

§4.1 says "An absent `pin:` means `DevicePin.UNPINNED`". §4.2 says an unpinned entry "connects as labelled unpinned". §3a's Safety line says `embody` "never targets … an unverified device". These conflict: a hand edit, a bad merge or N2 removes one line, and the next `embody rosie` connects to whatever holds `rosie`'s host, with one WARNING.

Fix:
- In a writer-managed file (one carrying `_format_version`), `add` writes an explicit value, `pin: <hex>` or `pin: unpinned`, and a missing key is a `RobotsConfigError`. Only a legacy file without `_format_version` maps a missing key to UNPINNED.
- `embody` and `--device` on an UNPINNED entry need `--unpinned` (round 1's A10 fix). `check` reports unpinned as a failing layer.

### N4. "Proxies disabled" has no mechanism, and the SDK adds a third proxy path (SHOULD-FIX, fail-closed for motion, A8 class)

`utils/http.py::fetch_url` takes `address_policy` but has no proxy switch. Its docstring for #921 says "an env proxy that applies to the URL wins, with one warning", and `_checked_client` mounts `_proxy_mounts` unless `strict`. v2's "probes … with proxies disabled" therefore needs a new strict private endpoint (a `specs` row with `strict_address_check=True`) plus the matching `lint_http_surface` update. Name it in §4.2 and in PR 1's scope.

Separately, v2 dropped round 1's pre-`ReachyMini(` check that no proxy applies to `ws://<literal>:8000/ws/sdk`. The `requests` media POST inside `__init__` (§1) follows yet another set of env-proxy rules.

Fix: before `ReachyMini(`, run one predicate, `urllib.request.getproxies()` plus `proxy_bypass(literal)` over `ws`, `socks`, `https`, `http` and `all`. Refuse with a fix hint if any proxy would apply. That covers both websockets and `requests`. Step 3 stays the backstop.

### N5. WebRTC to `wlan_ip` happens before any check (SHOULD-FIX, fail-open, A14 class)

`ReachyMini._configure_mediamanager` → `MediaManager._init_webrtc(signalling_host=daemon_status.wlan_ip, 8443)` and `camera.open()` run inside the constructor. A daemon at the pinned literal that reports `wlan_ip` as an arbitrary, possibly public, host gets Maxim to dial out to it before step 3 or step 5 runs. That dial is outside `address_policy`, and outside the V8 offline latch, which §6 checks only for discovery.

Fix: validate `wlan_ip` in the step-1 pre-filter from the REST status. It must be an IPv4 literal and non-public, and it must equal the verified literal or be in the same class. Otherwise refuse, or connect with `media_backend="no_media"` and report it. Keep step 5 as the session-side comparison. Also say whether the latch blocks a robot connect.

### N6. The relative-`MAXIM_DATA_HOME` outcome must be a loud error (SHOULD-FIX, fail-open if misread, A5/A15 class)

§6 says the data-home branch "requires an absolute `MAXIM_DATA_HOME`" but not what happens otherwise. Two readings would be fail-open:
- Falling back to `Path.home()/".maxim"` turns an isolated harness run into a run against the operator's real robots.yaml. `hardware/config.py::find_config_file`'s own comment names this hazard ("could connect a trial to real hardware").
- Enforcing the rule only in `find_config_file` while `utils/paths.py::data_home` still accepts a relative path leaves `device_state.json` and the writer target cwd-relative.

Also, once the cwd list is gone, `find_config_file`'s `except Exception` branch would return `None`, which means "no entries" and therefore an unnamed mDNS connect.

Fix:
- Enforce the rule in `data_home()` (one place) with a typed error, or in a `robots_yaml_path()` used by the reader, the writer and the state file alike.
- A relative value, including a literal `~/…`, raises, and never falls back.
- The `find_config_file` exception branch raises `RobotsConfigError`.

### N7. Headless fallback vs typed raise (NIT, fail-closed)

Say it once: an identity mismatch on a **pinned** entry raises and exits non-zero however the entry was selected, and only `DeviceUnreachable` on an unnamed, unpinned legacy launch degrades to headless. Also state that `MAXIM_DEVICE` and `default:` are explicit names, so a typo has no primary fallback.

### N8. Step 3 placement (NIT, fail-closed)

Put the session check on the line right after `ReachyMini(` returns in `controller.py::ReachyMiniController.connect`, before `_connect_released_media`, `_assert_auto_yaw_off`, `_probe_state_shape` and `_clear_latch_after_connect`. Use `client.get_status(wait=True, timeout=≤5)`, and make `TimeoutError`, a `None` id or an `AssertionError` a refusal. Add a test with a fake WS daemon whose session id differs from its REST id.

### N9. Canonical IPv4 and `hostname:` (NIT, fail-closed)

- Hygiene checks and the dial both use `str(ipaddress.IPv4Address(raw))`. An inet_aton-only form (`010.6.0.63`, `0x0a.6.0.63`, `167772223`) is refused rather than passed to getaddrinfo.
- Define `hostname:`. Either it is display or relocate metadata only, or it is a resolution source, in which case it is listed in §4.2's order and under the 4-candidate cap.

### N10. `relocate` details (NIT, fail-closed)

State that the candidate must pass hygiene and a REST pin match before anything is written, and that the write goes only to `device_state.json`, not to `host:`. The CLI refuses on a non-TTY without `--accept`. The Python `relocate(name, candidate)` is the operator act; say so in its docstring.

### N11. Enforce the pin through the result, not the flag (NIT)

After every connect, `RobotRegistry.connect_robot` checks `controller.verified_identity == pin`, which it needs anyway for the reuse branch, and refuses otherwise. `supports_identity_pin` then becomes a fast pre-check, not the guarantee.

### N12. `MAXIM_DEVICE` inheritance (NIT, fail-closed)

`child_env` copies the whole environment. The plan block and the run log should name the selection source ("device rosie (from MAXIM_DEVICE)"). Harnesses that spawn `maxim --mode` should strip the variable, or `lint_harness_provenance` should require it to be stripped.

### N13. "pin" naming (NIT)

The SDK's `get_pin()` is a different, secret-ish value taken from the last 5 chars of the raw serial. Maxim docs and fix hints should say `pin:` is the public 16-hex hardware id, and Maxim should never read or display the SDK's `get_pin()`.

## Stop-rule outcome

**Not met.** N2, N3, N5 and N6 are fail-open in covered classes. N1 is fail-open in a new class (the `repin` verb v2 introduced). N4 and N7–N13 are fail-closed or n/a in covered classes. No finding is DO-NOT-BUILD. Fold N1–N6 into the note (each is a paragraph of spec, plus the `check_write_permission` exclusion and the strict private endpoint in PR 0/PR 1 scope). A delta-scoped round 2 on exactly those folds is then owed before PR 0. The NITs can be folded without a round.
