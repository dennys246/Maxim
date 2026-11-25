# Device registry (#1806): adversarial design pass (inputs and trust)

Reviewer: adversarial lens (inputs and trust), read-only, 2026-11-25. Material: `docs/plans/device_registry_1806_approach.md` DRAFT v1, read against branch `plan/1806-device-registry` at `b7380ee74614306404297fca9a4ff6e04345bd6f` (origin/main) and the installed reachy-mini SDK 1.8.3 (`.venv/lib/python3.12/site-packages/reachy_mini/`, websockets 15.0.1, httpx as installed). No network access, no robot contact, no probes were run. Every claim below was read in code unless it is marked **unverified**.

**Verdict: DO NOT BUILD as written.** The idea holds up: per-kind stores, discovery treated as a hint, and refusing on a mismatch. But the core claim of §4, that "the pin is enforced inside the seam … no caller can skip it", does not hold against the code it would ride on:
- The REST check and the SDK's WebSocket dial are different connections. They can go to different hosts (connection_mode `auto` tries localhost first, a hostname is resolved again, the discovered port is not the dialled port, and the two libraries apply proxy rules differently).
- The pin is an optional free-form config key, which the registry's signature filter drops silently.
- Discovery step 4 is an auto-connect.
- A persisted `launch.operational_mode` is a capability grant that no flag was typed for, and a planted `./robots.yaml` or a singularity agent can write one.

Each of these is fixable on paper. The note should be revised and given a delta round before PR 1.

| # | Finding | Class | Direction |
|---|---|---|---|
| 1 | The identity check is not bound to the connection that carries motor commands. The REST probe and `ReachyMini(...)`'s WebSocket can reach different daemons (`connection_mode: auto` tries localhost first; a hostname is resolved again; the probe uses the SRV port while the dial uses the fixed 8000; a tunnel's local :8000 can be any local listener). Fix: dial an IP literal and re-check `hardware_id` from the WS session's own `DaemonStatus` before the first command | DO-NOT-BUILD | fail-open |
| 2 | Per-device `launch.operational_mode` plus `--save-defaults` is a persisted capability grant. It reaches `singularity` with no flag typed (DECISIONS #829 decision 2), raises over the run mode's capability, and outlives #922's time box. A planted `./robots.yaml` (#5), or a singularity agent with full CWD write, can write it | DO-NOT-BUILD | fail-open |
| 3 | "No caller can skip the pin" is false. `expected_hardware_id` in the free-form `config:` dict is dropped silently by `RobotRegistry.connect_robot`'s signature filter (a typo, a plugin, or an older controller); YAML `expected_hardware_id:` is `None`, which means unpinned; `selfy`'s `except Exception` → debug falls back to an unpinned mDNS connect; `maxim.run(robot=)` and `maxim.connect(type, config=)` never read the pin | DO-NOT-BUILD | fail-open |
| 4 | Resolver step 4 (a discover() browse "filtered to the pinned id", first match wins, written back to the cache) is an auto-connect from an unauthenticated advertisement, against §6's "never auto-connects". Since `hardware_id` is public, any LAN host can win by advertising it, and the win becomes sticky in `device_state.json` | DO-NOT-BUILD | fail-open |
| 5 | The `./robots.yaml` cwd fallback (and the cwd-only branch when `data_home()` fails) is not in §6. A repo checkout can plant a whole entry with its own host, pin, `type:` and `launch:`. The writer's target file is unspecified | SHOULD-FIX | fail-open |
| 6 | "Unknown name refused" contradicts `resolve_robot_entry`, which falls back to the primary on a miss. §5 wants every path consolidated onto that fallback, so `--device typo` / `embody typo` would embody the primary | SHOULD-FIX | fail-open |
| 7 | Candidate address hygiene. `address_policy="private"` means non-public: loopback, 0.0.0.0, 169.254/16 (metadata), multicast and CGNAT/Tailscale are all admitted. A discovered SRV port is attacker-chosen. An IPv6 literal breaks the SDK's `f"ws://{host}:{port}"`. Trailing dots and the candidate count are unbounded | SHOULD-FIX | fail-open |
| 8 | Proxy divergence. `fetch_url(address_policy="private")` lets an env proxy win (httpx rules), while websockets 15's sync `connect` proxies `ws://` through `ws`/`socks`/`https`/`http` (urllib rules). `HTTPS_PROXY` alone proxies the WS but not the probe | SHOULD-FIX | fail-open |
| 9 | `hardware_id` edge cases. It is `None` on any daemon not on Linux (a Lite on macOS or Windows, the desktop app, sim) and on a daemon started before USB audio enumerated (the value is cached at `Daemon.__init__`). A sim/mockup daemon on the robot reports the real id. The value is the first Pollen audio device in `iterdir` order. Pin format and case are unvalidated. A legitimate audio-board swap has no re-pin verb | SHOULD-FIX | fail-closed (sim with the same id: fail-open) |
| 10 | Unpinned entries go through the resolver with nothing to verify, so "first match wins" becomes "first answer wins": the DHCP mix-up is back, and it is written to the cache. `embody` "never targets … an unverified device", yet unpinned entries "keep working" | SHOULD-FIX | fail-open |
| 11 | `embody`'s preflight result is not what the launch dials. The launch re-reads robots.yaml (selfy) and re-resolves in the controller, and a refused connect degrades to "headless mode" instead of a non-zero exit | SHOULD-FIX | fail-open |
| 12 | TXT and status strings are named "escaped" but no sanitiser exists for them. The zeroconf path decodes raw bytes (ESC passes); Rich markup (`[link=…]`) is interpreted if printed through `Console`; `dns-sd` TXT parsing splits on whitespace (key injection) | SHOULD-FIX | fail-open |
| 13 | `discover` cannot produce `DiscoveryUnavailable` for the macOS permission case. The SDK's `_filter_alive` swallows `OSError` (EHOSTUNREACH) and `_dnssd_browse` returns `[]` on `OSError`, so a missing permission reads as "no robots" | NIT | fail-closed |
| 14 | WebRTC media goes to `daemon_status.wlan_ip` from the WS status. That is a second endpoint outside the address policy and the pin | NIT | fail-open |
| 15 | Inputs missing from §6: `MAXIM_DATA_HOME` (relocates robots.yaml, the state file and the pin; a relative value resolves from cwd), the unnamed selection env var, and `MAXIM_ROBOT_TIMEOUT` (a bare `float()` raises) | NIT | n/a |
| 16 | §7 says `connect`'s default is written by the robots.yaml writer, while §3 and D2 put it in `config.json` | NIT | n/a |

---

## 1. The pin is not bound to the dialled session (DO-NOT-BUILD, fail-open)

§4 verifies each candidate with `GET /api/daemon/status` through `fetch_url` and then lets `ReachyMiniController.connect` hand an address to `ReachyMini(...)`. In 1.8.3 the SDK makes its own connection, and four things can send it somewhere other than what was verified.

**(a) `connection_mode: auto` dials localhost first.** `reachy_mini/reachy_mini.py::ReachyMini._initialize_client`:
```python
if requested_mode == "auto":
    try:
        client = self._connect_single(host="localhost", port=self.port, timeout=timeout)
        selected: ConnectionMode = "localhost_only"
    except Exception as err:
        ...
        client = self._connect_single(host=self.host, port=self.port, timeout=timeout)
```
`controller.py::ReachyMiniController.__init__` accepts `connection_mode="auto"` from robots.yaml, and `connect` probes `self._host` but passes `connection_mode=effective_mode` unchanged. A developer Mac running `reachy-mini-daemon --sim`, the desktop-app daemon, or any listener on `localhost:8000` gets the session, while the probe verified the LAN robot.

**(b) Hostnames are resolved three or four times.** `connect` uses `probe_host = self._host`, which can be a hostname. `_port_open` resolves it, `_daemon_status` (fetch_url) resolves it again, and `ReachyMini(host=connect_host)` passes the same string to `io/ws_client.py::WSClient.__init__` (`uri = f"ws://{host}:{port}/ws/sdk"`; `ws_sync.connect(uri)`), which resolves it once more. §6 itself says "every robot defaults to `reachy-mini.local`". With two robots on the LAN, each resolution can land on a different one. `fetch_url` does not report which address it dialled (`utils/http.py::_AddressCheckingBackend.connect_tcp` tries each resolved address in turn and returns only the stream), so the resolver cannot know which IP to cache or pass on.

**(c) Probe port and dial port differ.** Discovery returns `DiscoveredRobot.port` from SRV (attacker-chosen, §6 row 1), but `connect` always dials `port=8000`. A probe at `addr:SRVport` verifies nothing about `addr:8000`.

**(d) Tunnel mode checks the wrong listener.** With `tunnel: true` or `localhost_only`, `probe_host = "127.0.0.1"`. `_wait_local_port(8000)` returns True for any listener already there (for example a local sim daemon), even when `ssh -L` then fails with `ExitOnForwardFailure`. This is the "status endpoint served by a different process" case. On a single daemon, REST and `/ws/sdk` share one FastAPI port, but nothing guarantees that the thing on local :8000 is the far end of the tunnel.

**Verified mitigation in the SDK.** The WS session carries identity itself. `io/protocol.py::DaemonStatus` has `hardware_id: Optional[str]`. `daemon/daemon.py` fills it from `get_hardware_id()`. `io/ws_server.py::WSServer.publish_status` pushes it at 1 Hz. `io/ws_client.py::WSClient` stores it (`_last_status`, `get_status`). `ReachyMini._configure_mediamanager` already calls `self.client.get_status()` inside `__init__`.

**Proposed fix (one rule, stated in §4):**
1. The resolver turns every candidate into **IPv4 literals** itself, probes each literal at **the port the SDK will dial** (8000 for the Reachy kind), and hands the controller the literal, never a hostname.
2. `connect` forces `connection_mode="network"` for any non-tunnel entry. It refuses `auto` when a pin is set, or deprecates `auto` outright, since a pin and "localhost first" cannot coexist.
3. **Right after `ReachyMini(...)` returns, before `_connect_released_media`, `_assert_auto_yaw_off`, `_probe_state_shape` or any command Maxim sends**, read `self._mini.client.get_status(...).hardware_id`. On a mismatch or `None`, disconnect and raise `DeviceIdentityMismatch`. This binds the identity to the socket that carries motor commands, whatever the resolution, proxy or tunnel did. The REST probe stays, as the cheap pre-filter and the `check` layer.
4. **Residual:** `ReachyMini.__init__` itself sends `SetAutomaticBodyYaw` and, under `no_media`, calls `release_media()` before Maxim can check. This is acceptable once the target is a probed literal. Say so in the note.
5. Add a regression guard: a fake WS daemon whose status `hardware_id` differs from the REST one must be refused.

## 2. A persisted launch default is a capability grant (DO-NOT-BUILD, fail-open)

§3a: "An entry may carry `launch: {mode, operational_mode, llm, interactive}` … `--save-defaults` writes the flags in use". §3a also says "Capability rules are unchanged (#829/#924)". They would not be unchanged:

- `DECISIONS.md` 2026-09-28 (#829), decision 2: "**Singularity is reachable only by the explicit launch flag**, announced on stderr and in the log." A `launch.operational_mode: singularity` read from a file is not the explicit flag. One `maxim embody reachy --operational-mode singularity --save-defaults` makes every later bare `maxim embody reachy` a singularity run.
- `docs/agents/runtime-tools.md` (#924 entry): "only `--operational-mode` at launch raises it". A saved `{mode: sleep, operational_mode: active}` raises over the run mode's own capability (`sleep` → passive) with no flag typed.
- #829 decision 2 says the grant persists only across in-process run-mode re-execs, and #922's time box will supersede that. A file grant persists forever.
- **Writers other than the operator.** `utils/filesystem_policy.py` gives `singularity` `CwdAccessLevel.FULL` over `cwd`, and `active` cwd edits under approval. With the cwd fallback (#5), an agent can write `./robots.yaml` with `launch.operational_mode: singularity`, and the next `maxim embody` from that directory starts in singularity. The model has then raised its own capability across a restart, which is the exact thing #924 forbids. A cloned repo can ship the same file.
- `launch.llm` has the same shape for egress: a saved cloud model sends perception to a provider on every run with nothing typed. That is less severe, but it is the same "a file chose for you" problem.

**Proposed fix:**
- `launch.operational_mode` may only **lower** the capability the run mode implies. Apply the one predicate, `modes/definitions.py::raises_capability`, at load time: a raising value is refused with a message naming the flag. `singularity` is never persistable.
- `--save-defaults` refuses, rather than drops, a raising `--operational-mode`, so the operator knows.
- The launch plan block (§3a step 3) prints the source of each value ("operational_mode: active (from run mode exploration)" vs "(from flag)").
- Launch defaults are read only from the data-home robots.yaml, never the cwd one (#5).
- Add an `engineering` invariant with a regression guard: a robots.yaml `launch.operational_mode: singularity` without the flag runs passive or active, never singularity.

## 3. The pin is omittable, and silently so (DO-NOT-BUILD, fail-open)

§4: "`ReachyMiniController.connect` takes an `expected_hardware_id` config key … no caller can skip it (CLAUDE.md 'push silent-no-op invariants into types')". A config key is the opposite of a type-level invariant:

- `hardware/registry.py::RobotRegistry.connect_robot` filters the config to the constructor signature and logs dropped keys at **DEBUG** (`"config keys not taken by %s constructor"`). A typo (`expected_hardwareid`), a plugin controller, or a `RobotRegistry` override of `reachy_mini` through `maxim.robots` entry points (`api.py::_discover_robot_plugins`) drops the pin, and the connection goes through unpinned. §3b's "Every path then gets the check, including … plugins" is therefore false: plugins get it only if they implement it.
- YAML `expected_hardware_id:` with no value, or `~`, parses to `None`, which means "unpinned legacy". A pin deleted by mistake is indistinguishable from never pinned.
- `embodied_runtime/selfy.py` (the connect block around `resolve_connection_config`) wraps the robots.yaml load in `except Exception as e: self.log.debug("robots.yaml connection config not loaded: %s", e)` and continues with `{"robot_name": self.name, "media_backend": …}`. That is an mDNS connect with no host and no pin. §5's fix makes `from_yaml` raise, and this handler turns that raise back into an unpinned connect. It shares the root cause, so it is in scope ("Filing is not fixing").
- `api.py::_connect_robot_for_run` (`maxim.run(robot="reachy_mini")`) and `api.py::connect(robot_type, config=…)` pass no robots.yaml config, so the controller falls back to `<robot_name>.local`. That is exactly the colliding name §6 lists.

**Is the Python bypass acceptable?** For `maxim.connect("reachy_mini", config={"host": "10.x"})`, yes: the caller's code is the operator, and an explicit literal host is an operator act. It must be labelled, and §3b's "a Python caller cannot skip it" must be corrected. The silent cases are not acceptable: the dropped key, the `None` pin, the selfy fallback, and the bare-mDNS `run(robot=)`.

**Proposed fix:**
- Make the pin a **required keyword-only constructor parameter** whose value is either a validated 16-hex `HardwareId` or an explicit `UNPINNED` sentinel, so that omitting it is a `TypeError` (the `build_executor(pain_bus=…)` pattern). Better still, carry it on the typed `RobotConfig` (`identity:`), not inside free-form `config:`.
- The registry passes it explicitly and refuses a controller class that does not accept it when the entry is pinned.
- An unknown key adjacent to the pin (`expected_*`) is an error, not a DEBUG line.
- selfy's handler must re-raise for a named entry.
- `run(robot=)` and `connect(type)` with no host resolve through the default device (D2) when one is set, and otherwise log one WARNING that the connection is unpinned.

## 4. Discovery step 4 is an auto-connect (DO-NOT-BUILD, fail-open)

§6 row 1: "A discovery result is a *hint*. It never auto-adds or auto-connects." §4 step 4: "a `discover()` browse filtered to the pinned id … The first match wins, and its address is written back to the state file." The filter is on TXT `unit_id`, which the advertiser writes (`utils/discovery.py::MdnsServiceRegistration._do_register`; on macOS `_dnssd_resolve` even prefers the TXT `address` over DNS: `if "address" in properties: addresses.append(properties["address"])`). Verification then checks a `hardware_id` that §6 calls public: it is in TXT `unit_id` and served unauthenticated at `GET /api/daemon/hardware-id` (`daemon/app/routers/daemon.py::get_robot_hardware_id`).

Without step 4, an impostor has to hold the cached or pinned IP, or the robot's hostname. With step 4, it only has to advertise while the real robot is off or has moved. The result is then written to `device_state.json` and tried at step 2 on every later connect, while the impostor stays up. The impact is the one §6 names: motor commands over cleartext WS, and fabricated sensor data.

**Proposed fix:**
- Step 4 may *suggest*, but it never connects in a run that can actuate. `embody` and `connect(device=)` refuse with "robot not at its known addresses; found a candidate at X advertising your robot's id — run `maxim connect reachy` to accept it". `maxim connect` asks the operator to confirm on a TTY and refuses when not on one, unless `--accept-discovered` is passed.
- Only an operator-confirmed address goes into the state file.
- §6 should state the residual: with no daemon authentication, the pin stops mix-ups, and only operator confirmation stands between a forged advertisement and the motors.
- Bound step 4: limit the number of candidates (an advertiser can publish hundreds), allow one probe per address, and fix the port (#7).

## 5. The cwd `robots.yaml` is an unlisted input (SHOULD-FIX, fail-open)

`hardware/config.py::find_config_file` searches `[data_home()/robots.yaml, "robots.yaml"]`, so `./robots.yaml` is read whenever the data-home file is absent. If `data_home()` raises, only cwd is searched (`"robots.yaml search limited to cwd"`). Every reader goes through `load_robots_config`: selfy, `agentic_runtime` body wiring, and doctor. Before the first `maxim device add`, `cd`-ing into a checkout that carries a `robots.yaml` and running `maxim embody reachy` uses an entry the operator never wrote, and that entry carries its own `host:`, its own matching `expected_hardware_id`, `type:`, `body:` and `launch:` (#2). A pin taken from the same file as the host proves nothing about operator intent. The note also never says which file `device add` writes, or whether it copies entries out of a cwd file into the data home.

**Proposed fix:**
- The device registry reads and writes only `data_home()/robots.yaml`.
- A cwd file is ignored by `maxim device|connect|embody`, with one WARNING naming it. Ideally the legacy cwd fallback is removed in PR 0, as part of the "one resolver" fold. If removing it is an owner decision, list it in §9.
- The writer's target is that one path, created `0600`.
- Add the cwd file and `MAXIM_DATA_HOME` (#15) to §6.

## 6. "Unknown name refused" vs `resolve_robot_entry` (SHOULD-FIX, fail-open)

`hardware/config.py::resolve_robot_entry` returns the primary when `robot_id` misses and there is one robot or an explicit primary. §6 ("An unknown name is refused, never defaulted") and §5 ("`agentic_runtime.py` … should call the one resolver") pull in opposite directions: consolidating every path onto this function makes `maxim embody reachyy` or `--device rosie-old` embody the primary.

**Fix:** split the rule. An *explicitly named* entry has no fallback, and a miss is `DeviceNotFound`. The primary fallback applies only when no name was given. Make both a parameter of the one resolver (`explicit: bool` or two functions), with a test for each.

## 7. Candidate address hygiene (SHOULD-FIX, fail-open)

- `utils/net.py::is_public_address` defines public. `"private"` is its complement: loopback, `0.0.0.0`, `169.254.0.0/16` (including `169.254.169.254`), multicast, and CGNAT `100.64/10` (Tailscale). Loopback is legitimate for a Lite or a tunnel, and CGNAT is legitimate for a tailnet robot (the owner's rig uses Tailscale). But a cached, discovered or hostname candidate that resolves to loopback should never stand in for a *remote* entry: that is the local-sim-daemon mix-up of #1(a). Fix: a per-entry address class (`local`, `lan`, `tailnet`) recorded at `add` time. A candidate outside the entry's class is refused. Unspecified, multicast and link-local are never valid. Tailnet means any node in a possibly shared tailnet; the pin, not the class, separates nodes, so say that in §6.
- Port: probe and dial the kind's fixed port (8000). Ignore the SRV port, or require it to equal 8000.
- IPv6: `WSClient` builds `f"ws://{host}:{port}/ws/sdk"` and `ReachyMini` builds `f"http://{self.client.host}:{self.client.port}"` without brackets, so an IPv6 literal (with or without a `%zone`) produces a broken URI. The SDK's own discovery is IPv4-only (`socket.inet_ntoa`; `_dnssd_resolve` uses `AF_INET`), as is `controller.py::_resolve_mdns` (`gethostbyname`). Fix: v1 candidates are IPv4 only. Write that in §6 instead of "IPv6-first or AAAA stalls … Names normalised".
- Trailing dot: normalise `reachy-mini.local.` to `reachy-mini.local` before comparing names and before writing the cache, and refuse labels with characters outside `[A-Za-z0-9-.]`.
- The refusal message ("names each candidate and why it failed") must not echo response bodies from whatever answered. An attacker-chosen address can be any LAN HTTP service.

## 8. Proxy divergence between the probe and the WS (SHOULD-FIX, fail-open)

- `utils/http.py::_checked_client` mounts `_proxy_mounts` unless `strict`, and the private policy endpoint is registered `strict=False` (`_ensure_external_endpoint`). An env proxy therefore carries the probe (#921: "the proxy wins for operator paths, loudly").
- `websockets/sync/client.py::connect` defaults to `proxy=True` → `websockets/uri.py::get_proxy`, which for `ws://` tries `["ws", "socks", "https", "http"]` from `urllib.request.getproxies()`, with `urllib.request.proxy_bypass` semantics.
- httpx's `_utils.get_environment_proxies` maps only `http`/`https`/`all` to their own scheme mounts.

So `HTTPS_PROXY=…` alone leaves the `http://` probe direct and checked, and sends the WS through the proxy. A SOCKS system proxy has the same effect. NO_PROXY parsing also differs (curl rules vs urllib), and the http.py docstring records that re-deriving these rules "was wrong twice in review".

**Fix:**
- The device probe uses a **strict** private endpoint, since a LAN robot never belongs behind a proxy.
- Before constructing `ReachyMini`, `connect` calls `websockets.uri.get_proxy` (or `urllib.request.proxy_bypass` plus `getproxies`) on the exact `ws://<literal>:8000/ws/sdk` and refuses with a fix hint if a proxy would apply. The SDK does not expose `proxy=None`.
- #1's in-session check also covers this residual.

## 9. `hardware_id` edge cases (SHOULD-FIX, mostly fail-closed)

From `utils/hardware_id.py::get_hardware_id` / `_read_raw_audio_serial`:
- `None` whenever `/sys/bus/usb/devices` is absent, which means every daemon not running on Linux: a Lite on a Mac or Windows host, the desktop-app daemon, and a sim on macOS. The docstring says "null when no robot is attached, e.g. on a developer machine". For Lite owners on macOS, the pin can never be set. The note has to say what `add` does then. Recommendation: `add` refuses to pin `None` and records `unpinned (daemon reports no hardware id)`, allowed only for the `local` address class (#7).
- **Cached at daemon start.** `Daemon.__init__` evaluates `hardware_id=get_hardware_id()` once. A wireless robot whose daemon starts before the USB audio device enumerates reports `None` until the daemon restarts (**unverified** that this boot race happens). A pinned entry then refuses, which is correct, and the fix hint must say "restart the daemon".
- **The first matching device in `iterdir()` order.** A host with two Pollen audio devices (two Lites on one Linux box, or a Pollen accessory with the same VID:PID) can report either one, so a pin can flip after a reboot (**unverified** in practice).
- **The same id from a sim.** `reachy-mini-daemon --sim` or the mockup on the robot's own CM4 reads the same sysfs and reports the real id. Fix: identity match AND `simulation_enabled is not True` AND `mockup_sim_enabled is not True` for an entry pinned to hardware (`io/protocol.py::DaemonStatus` has both fields). This case is fail-open: a "verified" real robot that is actually a sim produces silently wrong perception.
- **Validation.** The pin must match `^[0-9a-f]{16}$` after lowercasing, and the daemon value must be a `str` of the same shape. Anything else is a mismatch, never a pass.
- **A legitimate board swap** changes the id (the id is the audio chip's serial). D6 refuses, correctly, but no verb in §3 re-pins. Add `maxim device pin <name>` (operator verb, TTY confirm, showing the old and new id), and name it in the mismatch fix hint.

## 10. Unpinned entries defeat the resolver (SHOULD-FIX, fail-open)

§4 "A hand-written entry with no pin keeps working". For such an entry, "each candidate is verified" has nothing to verify, so the cache, hostname and discovery steps take the first answer and write it back. That is the DHCP-reassignment mix-up §6 row 3 exists to stop, and it is now persisted. §3a also says `embody` "never targets an unregistered or unverified device", which contradicts unpinned entries working.

**Fix:**
- An unpinned entry uses only its explicit `host:` (or the legacy mDNS name), with no cache, no discovery and no write-back.
- `embody` of an unpinned entry needs `--unpinned` (or a one-time `maxim device pin`), and the plan block says `UNPINNED`.
- `check` reports it as a failing layer, not a note.

## 11. `embody`'s preflight is not the launch's connection (SHOULD-FIX, fail-open)

§3a says that after resolving and preflighting, `embody` "enters the same `_main_impl` robot path … `Maxim(robot_id=…, robot_name=…)` → `RobotRegistry.connect_robot`". That path re-reads robots.yaml in selfy and re-resolves in `ReachyMiniController.connect` (host or mDNS). It never sees the preflight's verified address, so the plan block can show address A while the launch dials B. It also cannot see a cache or discovery result at all, so a launch that preflight passed can fail, or, for an unpinned entry, land elsewhere. A refused connect then logs "No robot connected — running in headless mode" (`selfy.py`, after `connect_robot`) and keeps running.

**Fix:**
- `embody` passes the `ResolvedDevice` (IPv4 literal + pin) into the launch config as overriding values, and the controller re-verifies in-session (#1).
- A refused or failed connect under `embody` exits non-zero with the typed error. It never degrades to headless.
- The plan block is printed from the same object the launch uses.

## 12. Untrusted strings in the terminal (SHOULD-FIX, fail-open)

§6 says TXT and status strings are "length-bounded and escaped", but no helper for this exists in `src/maxim` today. A grep for control-character handling finds only `runtime/config_writer.py` (the `isprintable()`/`unicode_escape` loop) and the TTY reader.

- On Linux, `utils/discovery.py::_RobotCollector.add_service` does `v.decode()` on raw TXT bytes, so ESC/CSI/OSC sequences pass straight through (OSC 8 hyperlinks, title setting, screen clearing). A non-UTF-8 value raises inside the zeroconf listener thread and silently drops that service.
- On macOS, the values are whatever `dns-sd -L` prints, and `_dnssd_resolve` splits TXT pairs on whitespace. A value `robot_name=x unit_id=<id>` injects a second key. (Whether `dns-sd` escapes non-printables in TXT output is **unverified**.)
- The instance name from `dns-sd -B` is passed as an argv element to `dns-sd -L`. A leading `-` could be read as an option (**unverified**).
- If `discover`/`check`/`show` print through Rich `Console.print` (cli.py imports Rich), bracket markup in a TXT value is interpreted unless escaped.
- Daemon status `robot_name`, `version`, `error` and `wlan_ip` reach `check` output and the launch plan in the same way.

**Fix:**
- One `hardware/` sanitiser (escape non-printables like `config_writer.py`, strip bidi controls, `rich.markup.escape`, a length cap), used by every renderer of hint or status text.
- Maxim-side parsing of `DiscoveredRobot.properties`, accepting only known keys with shape checks (`unit_id` = 16 hex, `address` = IPv4 literal).
- A test that feeds `\x1b]8;;…` and `[link=…]` through `discover`'s renderer.
- §6's "never reach a prompt" has no enforcing check. Name one (a test that no device or status string enters the context prompt), or add an M-row.

## 13. The macOS permission case is invisible to `find_robots` (NIT, fail-closed)

`utils/discovery.py::_filter_alive` catches `OSError` on every `create_connection`, which includes macOS Local Network `EHOSTUNREACH`. `_dnssd_browse` returns `[]` on `(FileNotFoundError, OSError)`. So `find_robots` returns `[]` both when no robot exists and when permission is missing, and `DiscoveryUnavailable (multicast permission)` cannot be derived from it. Under the V8 latch, `_offline.py::OfflineRefusal` is a `RuntimeError`, so it does propagate through those handlers, and that part works. **Fix:** `discover` runs its own permission probe (the `check` layer 1 probe) before or after `find_robots`, and reports "0 found, and local network access is blocked" separately.

## 14. WebRTC signalling host comes from the status payload (NIT, fail-open)

`ReachyMini._configure_mediamanager` builds `MediaManager(signalling_host=daemon_status.wlan_ip or "localhost", …)`. Camera and microphone media therefore go to an address the daemon reports, not to the verified literal. That address is outside `address_policy` and outside the pin. It is benign for an honest daemon (it reports its own wlan IP; over Tailscale or Ethernet that differs from the dialled address). **Fix:** document it in §6, and have `check` flag a `wlan_ip` that is public or differs in class from the entry's address class.

## 15. Missing inputs in §6 (NIT)

- `MAXIM_DATA_HOME`: relocates robots.yaml, `device_state.json` and therefore the pin. `utils/paths.py::data_home` takes `Path(raw)` as given, so a relative value resolves from cwd.
- The "env" in "`--device` > env > default" is not named. Name it, give it a brief-table row, and add a conftest scrub in the same commit (CLAUDE.md lesson).
- `MAXIM_ROBOT_TIMEOUT`: `selfy.py` does `float(os.environ.get(...))` with no guard, so a malformed value raises at startup.
- `MAXIM_REACHY_HOST` is read only by `utils/reachy_diagnostics.py`. Once the probe is consolidated, it either feeds the resolver or is removed. It must not become a fourth address source outside §4's order.

## 16. Where `connect`'s default lives (NIT)

§7 lists "`connect`'s default" among the robots.yaml writer's operations. §3 ("stored in `config.json` through `maxim config`") and D2 say config.json. Pick config.json, which matches the single source of truth, and fix §7. The CI writer allow-list grep needs the correct file.
