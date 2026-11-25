# Device registry (#1806): delta round 2, adversarial lens (inputs and trust)

Reviewer: adversarial lens (inputs and trust), delta round 2, read-only, 2026-11-25, v3 @ 8540cf19 (code at origin/main b7380ee7; reachy-mini SDK 1.8.3, websockets 15.0.1, requests 2.34.2 in `.venv`). Scope: delta-1 adversarial N1–N6, and any new fail-open defect those v3 folds introduce. No network, no robot. One offline probe of proxy resolution ran under `env -i` with a fresh `mktemp` HOME (§N4).

**Verdict: BUILD after SHOULD-FIX. The stop rule is NOT met.** N5 and N6 are closed. The N2 fold names a seam that no production path calls: `check_write_permission` is reachable only through `WriteFileTool._mode_config`, which `bootstrap.py` never sets, and `edit_file` never calls it at all. As written, the deny would ship with no caller, and the agent could still rewrite robots.yaml. That is fail-open in a covered class. The N3 and N4 folds each leave a narrower fail-open path in a covered class.

## Disposition of delta-1 N1–N6

| Finding | v3 status | Note |
|---|---|---|
| N1 `repin` adopts an impostor | **PARTIAL** (residual fail-closed / operator-mediated) | v3 closes the main path: the uniqueness refusal, TTY or `--yes`, network-first hint, and old/new id + model + version + `wlan_ip` + last-seen shown. Not folded from the delta-1 ask: uniqueness at `add`, and `expected_new=` for the Python verb. The Python `confirm=lambda old, new:` gets only the two ids, not the context the CLI shows. It is unstated whether `repin <name> <host>` also rewrites `host:`, which would make it relocate plus repin with one confirmation. Uniqueness cannot help when robot B is unregistered, so the shown context is the only bar there. That residual is in the N1 class, now covered |
| N2 agent-writable robots.yaml | **OPEN** (fail-open) | The fold puts the deny in `utils/filesystem_policy.py::check_write_permission`. Its only call site is `tools/filesystem.py::WriteFileTool.execute`, behind `if self._mode_config is not None`. `runtime/bootstrap.py` registers `WriteFileTool(allowed_dirs=allowed_dirs)` with no `mode_config`, and nothing in `src/` or `scripts/` calls `set_mode_config`. `EditFileTool` never consults the mode config. In every mode, `allowed_dirs` includes cwd. So `edit_file ~/.maxim/robots.yaml` (cwd = `~`) can repoint `host:` and `hardware_id:` with no gate this design adds. See E2 for the right seam |
| N3 deleted pin downgrades silently | **PARTIAL** (fail-open on non-flag paths) | `--unpinned` now gates `embody` and `--device`, so a deleted `hardware_id:` line refuses there (fail-closed). Still open: selection step 2 (`MAXIM_DEVICE`) is not named in the `--unpinned` rule, and neither is step 4 (single entry or `primary:` under `--mode`). On those paths a deleted line still connects unpinned with a WARNING. The strict-entries rule catches typos, not deletion. Delta-1's other half was not adopted: in a writer-managed file, a missing key is an error, and the writer writes `hardware_id: unpinned` explicitly. Adopting it closes every path at once |
| N4 proxies | **PARTIAL** (fail-closed for motion; side-effect egress fail-open) | `fetch_url(proxy="never")` is feasible and clean (E4a). The connect-refusal predicate is defined by env-var names ("`HTTP(S)_PROXY`/`ALL_PROXY` set and not covered by `NO_PROXY`"), not by what the two consumers actually resolve. The probe shows three divergences (E4b). It also misses macOS system proxies. Step 3 still guards motion |
| N5 WebRTC to `wlan_ip` | **CLOSED** for egress, conditional on N4 | Requiring REST `wlan_ip == dialled IP` is sound for security: the dialled literal is already canonical, private and hygiene-checked, so WebRTC can only reach that host. The SDK reads `wlan_ip` from the **WS session's** status (`_configure_mediamanager` → `self.client.get_status()`), not from REST. The equality therefore carries over only if REST and WS reach the same daemon, which is N4's predicate (NF2). It also over-refuses legitimate setups (NF1, fail-closed) |
| N6 relative `MAXIM_DATA_HOME` | **CLOSED** | A relative value (including a literal `~/…`) raises in robots.yaml resolution and in the writer, and the `find_config_file` exception branch raises. `device_state.json` stays under `data_home()`, which can be relative, but it is only read or written after a pinned robots.yaml load, and that load raises first. No fallback to `~/.maxim` remains |

## New findings (introduced or exposed by the v3 folds)

| # | Finding | Class | Fail-open / closed | Covered? |
|---|---|---|---|---|
| NF1 | The step-1 equality `wlan_ip == dialled IP` refuses legitimate setups. The daemon sets `wlan_ip = get_ip_address("wlan0")`, and only on `wireless_version` (`daemon/daemon.py`). It therefore refuses: a wired robot (eth IP ≠ wlan0 IP, or `None` when Wi-Fi is down); every pinned **`tunnel:`** entry (dial 127.0.0.1, `wlan_ip` = LAN IP); the CGNAT/Tailscale literals §4.2 explicitly allows; and a Lite or Linux-host daemon (`wlan_ip` None). This contradicts §4.2/§4.3 step 2 support for tunnel and Tailscale | A14/N5 | **fail-closed** (availability; invites an ad-hoc bypass later) | Yes |
| NF2 | The N4 refusal predicate does not match the consumers. websockets uses `urllib.request.getproxies()` + `proxy_bypass("ip:port")` and prefers `ws`/`socks`. requests uses its own `should_bypass_proxies` (CIDR-aware, hostname without port). Evidence is in E4b. A proxied WS can reach a different daemon, whose session-reported `wlan_ip` drives WebRTC + `camera.open()` and receives `SetAutomaticBodyYaw` before step 3. That reopens N5's egress whenever the predicate passes falsely | A8/N4/N5 | **fail-open** (side effects and egress, not motion; step 3 holds) | Yes |
| NF3 | The N2 deny targets a function with no live caller (see N2 above). It also leaves `edit_file` uncovered, does not case-fold (APFS is case-insensitive: `~/.maxim/Robots.yaml` is the same file), and D14's "an agent should never be able to rewrite its own trust anchors" overclaims, because singularity `bash` can still `sed -i` the file (the scope must be stated the way `git_dir_write_refusal`'s docstring states it) | A2/A5/N2 | **fail-open** | Yes |
| NF4 | The new strict endpoint must also go into `_EXTERNAL_POLICY_ENDPOINTS`, or be added to `_DEADLINE_ENDPOINTS` by hand. Otherwise the `proxy="never"` probe loses the #1715 deadline backend, and a dripping impostor stalls the pre-filter | A8 | fail-closed (a hang, not an accept) | Yes |
| NF5 | The tunnel case under the N4 rule: `ws://127.0.0.1:8000` is proxied whenever `http_proxy` is set and `NO_PROXY` lacks `127.0.0.1` (urllib does not bypass loopback by default). v3 then refuses, which is correct, but the fix hint names `NO_PROXY=<ip>` and needs to name `127.0.0.1` for tunnel entries | A8 | fail-closed | Yes |

## Evidence

### E1. N1 (`repin`)
In v3, §4.2 lists the four controls. §3b's Python form is `repin("rosie", host=..., confirm=lambda old, new: ...)`, and §3.1 has `repin <name> [<host>]`. `add` only says "a null id → refused unless unpinned=True", with no uniqueness rule. Two entries sharing one pin cause no motion hazard (both are the same claimed robot), so that residual is fail-closed. Suggested text:
- `add` applies the same uniqueness refusal.
- `confirm=` receives the same candidate record the CLI shows (sanitised `robot_name`, model, version, `wlan_ip`, last-seen).
- `repin` with a `<host>` writes the `host:` change and the id change as one shown diff.

### E2. N2: where the deny must live
- `src/maxim/utils/filesystem_policy.py::check_write_permission` has exactly one call site, `src/maxim/tools/filesystem.py::WriteFileTool.execute`, and that site is gated on `self._mode_config is not None`.
- `src/maxim/runtime/bootstrap.py` (the tool registry build) calls `WriteFileTool(allowed_dirs=allowed_dirs)` and `EditFileTool(allowed_dirs=allowed_dirs)`. Neither passes a mode config, and `rg set_mode_config src scripts` finds only the definitions.
- `get_mode_filesystem_config` gives passive, active and singularity alike `allowed_dirs=[workspace, cwd] + extra`.
- The live per-mode write gate is `agents/autonomy.py::AutonomyPolicy.verdict`. It matches `tool_name == "write_file"` only, so `edit_file` is not approval-gated in active mode either. That gap is pre-existing and outside this gate's scope, but it is the same seam.

The seam every file-writing tool already calls is `tools/base.py::git_dir_write_refusal`, the #957 precedent. `WriteFileTool._is_path_allowed`, `EditFileTool._is_path_allowed` and `tools/sandbox.py::WriteSandboxFileTool` all call it.

Fix:
- Add a sibling `control_plane_write_refusal(real_path)` and call it from the same three places.
- Compare case-folded, with stat identity for existing files, against `data_home()/robots.yaml`, `robots.yaml.bak`, `util/device_state.json`, and (D14) `~/.config/maxim/`.
- Write its docstring scope the way #957's is written ("narrows; bash can still write").
- Test: singularity, cwd = data home, through **both** `write_file` and `edit_file`, verified red by deleting the call.

This is about 30 lines and fits PR 0. The CLAUDE.md caller-grep rule ("a fix ships with a caller") catches exactly this.

### E3. N3
- §3a: "Unpinned entries need `--unpinned`. That applies to `embody` and to `--device`."
- §3.1: selection steps 2 (`MAXIM_DEVICE`) and 4 (`primary:`) are not covered.
- §4.1: "An absent key means `DevicePin.UNPINNED`."

Combined with E2, an agent removes one line, and the next `maxim --mode exploration` drives whatever holds the primary entry's `host:`. Fix, either of:
- apply `--unpinned` (or a refusal) to every registry-selected path except legacy `--robot`/`--robot-name`; or
- adopt "missing `hardware_id:` in a `_format_version` file is a `RobotsConfigError`; the writer writes `hardware_id: unpinned`".

The second also covers future selection paths.

### E4a. N4: `fetch_url(proxy="never")` feasibility
`src/maxim/utils/http.py::_ensure_external_endpoint` registers `(name, policy, strict)` rows. `_EXTERNAL_POLICY_ENDPOINTS["private"]` is `strict=False`. `_checked_client(strict=True)` builds `mounts=None, trust_env=False`, so there is no proxy at all. A fifth row, `("_external_policy_private_direct", "private", True)`, plus a `proxy=` keyword that selects it, is clean. `lint_http_surface.py` does not pin these names. Add the row through `_EXTERNAL_POLICY_ENDPOINTS` (or by hand to `_DEADLINE_ENDPOINTS`), or the deadline backend is lost (NF4).

### E4b. N4/NF2: the consumers disagree
Probe: `env -i HOME=<mktemp>`, `python -I`, no network.

| Environment | websockets `get_proxy(ws://10.6.0.63:8000/ws/sdk)` | requests `get_environ_proxies(http://10.6.0.63:8000/…)` |
|---|---|---|
| `http_proxy=…`, `no_proxy=10.6.0.0/24` | **proxied** (urllib has no CIDR) | `{}`, direct |
| `ws_proxy=…` only (not in v3's list) | **proxied** | `{'ws': …}` (the http POST is direct) |
| `http_proxy=…`, `no_proxy=10.6.0.63:8000` | direct | **proxied** (the IP branch compares without the port) |

On macOS with no proxy env vars set, `urllib.request.getproxies()` and `proxy_bypass()` fall through to the System Configuration proxies, which both libraries then use. v3's env-name rule does not see them.

Fix: the predicate calls each consumer's own resolver:
- `websockets.uri.get_proxy(parse_uri(ws_url)) is None`, and
- `requests.utils.get_environ_proxies(http_url)` yields no proxy for `http`/`all`.

Refuse unless both say direct, and name the failing consumer in the hint. `check` layer 10 uses the same predicate. Pin the probe table above as unit tests.

### E5. N5/NF1: `wlan_ip` semantics
- `reachy_mini/daemon/utils.py::get_ip_address(ifname="wlan0")` reads the wlan0 address via SIOCGIFADDR.
- `daemon/daemon.py` sets it only `if self.wireless_version`.
- `reachy_mini.py::_configure_mediamanager` reads `self.client.get_status()` (the WS session) and passes `signalling_host=daemon_status.wlan_ip or "localhost"`. Under `no_media` it does not use `wlan_ip`: the only extra request is the `/api/media/release` POST to the daemon URL.

Suggested rule:
- Apply the check only when the backend will open WebRTC.
- If `wlan_ip` ≠ the dialled IP, the candidate is canonical private IPv4, and a REST probe at `wlan_ip:8000` returns the same pinned `hardware_id` with sim flags off, accept it. Otherwise connect `no_media` and report (delta-1's alternative), or refuse.
- `wlan_ip` None means signalling to `localhost`: refuse WebRTC or use `no_media`, and state which.

This keeps NF1's configurations working without reopening the egress.

### E6. N6
Read against `utils/paths.py::data_home` (it accepts any `MAXIM_DATA_HOME`) and `hardware/config.py::find_config_file` (an `except Exception` returns cwd-only today). v3 §4.1 makes both the relative value and the unresolvable branch raise `RobotsConfigError` for resolution and the writer. Every consumer of `device_state.json` sits behind that load. Closed.

## Stop-rule outcome

**STOP RULE NOT MET.**
- **N2/NF3** is fail-open. The deny's named seam has no production caller, and `edit_file` bypasses it. The class is covered (A2/A5/N2), but the finding is not fail-closed.
- **N3** (`MAXIM_DEVICE` and the primary path) and **NF2** (proxy predicate mismatch, side-effect egress) are fail-open in covered classes.
- NF1, NF4, NF5 and the N1 residuals are fail-closed in covered classes.

No finding is DO-NOT-BUILD. Fold the E2, E3 and E4b fixes (all spec text, about 30–60 `src/` lines in PR 0/PR 1). A delta round 3 scoped to exactly those three folds is then owed. If its reviewer finds only fail-closed issues in covered classes, the stop rule ends the rounds there.
