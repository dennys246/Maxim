# Device registry (#1806): delta round 3, adversarial lens (inputs and trust)

Reviewer: adversarial lens (inputs and trust), delta round 3, read-only, 2026-11-25, v4 staged on 0630487a. Code read at the worktree's b7380ee7 (origin/main has since moved to 0dd412e4; not re-read). SDK reachy-mini in `.venv`, websockets 15.0.1, requests 2.34.2. Scope: the v4 folds of NF2, NF3 and N3 only (§12 round-3 scope), plus any new fail-open defect they introduce. One offline resolver probe ran under `env -i HOME=$(mktemp -d)`, `python -I` (E1). No network, no robot.

**Verdict: BUILD after SHOULD-FIX. STOP RULE NOT MET.** Each of the three folds is the right shape and closes the main path the delta-2 finding named. Each still leaves a narrower fail-open path, and every one falls in an already-covered class (NF2/N5, NF3/A2, N3). None is DO-NOT-BUILD, and all are spec-text fixes in the same PRs.

## Re-check of the round-3 scope

| Fold | v4 status | Note |
|---|---|---|
| NF2: the proxy check asks each library's resolver | **PARTIAL** (fail-open on media egress) | Both functions exist: `websockets.uri.get_proxy(uri: WebSocketURI)` (`uri.py`) and `requests.utils.get_environ_proxies(url, no_proxy=None)` (`utils.py`). Asking the consumers' own resolvers is sound for the URLs it asks about, and on darwin both fall through to `urllib`'s System Configuration proxies, so the macOS claim holds. The problem is which URLs get asked. The SDK opens a **third** proxied client the predicate never asks about: `media/webrtc_utils.py::get_producer_list` → `websockets.sync.client.connect(f"ws://{wlan_ip}:8443")`, a different port and (now that v4 accepts `wlan_ip` ≠ the dialled IP) possibly a different host. The probe (E1) shows "both direct" passing while that connection is proxied, including under the fix hint's own `NO_PROXY=127.0.0.1` for a tunnel entry. Two spec mechanics are also wrong as written (R3-2, fail-closed) |
| NF3: `control_plane_write_refusal` on the #957 seam | **PARTIAL** (fail-open on path resolution) | The seam is right. The 4 `git_dir_write_refusal` call sites are `filesystem.py` `WriteFileTool._is_path_allowed` and `EditFileTool._is_path_allowed` (both of `EditFileTool`'s `write_text` calls run after it), plus `sandbox.py` `WriteSandboxFileTool` and `WriteToSharedOutputsTool`. They cover every production agent file-write tool. `respond` picks its own path; `learned_index` and `git_commit` are system-chosen or fixed argv; the sim `SandboxEnvironment._resolve` is realpath-confined; there is no notebook or patch tool. Symlinks into `~/.config/maxim` are caught, because realpath resolves every link in the path. But the **denied set is computed from the wrong inputs**: a literal `~/.config/maxim` while the live config dir honours `XDG_CONFIG_HOME` (R3-3), and the agent process's own `data_home()` while harness children run with a temporary `MAXIM_DATA_HOME` (R3-4). A case-fold with no stat identity also misses normalisation-equivalent spellings (R3-5). There is a third sandbox write path, dormant (R3-6) |
| N3: unpinned refusal on every selection path | **PARTIAL** (fail-open on its precondition and on legacy no-match) | §3.1 now covers `MAXIM_DEVICE`, legacy flags and step 5. Three gaps remain. (a) The rule fires only "while the registry holds a pinned entry", so deleting **every** pin (one line in a single-robot registry) turns it off. (b) §3a and §4.2 still say legacy launches are "unchanged" and that `--unpinned` applies to "`embody`/`--device`", contradicting §3.1. (c) A legacy name (steps 2 and 4, including the env var `MAXIM_ROBOT_NAME`) that matches **no** entry still dials `<name>.local` unpinned while pins exist |

## New findings (introduced or exposed by the v4 folds)

| # | Finding | Class | Fail-open / closed | Covered? |
|---|---|---|---|---|
| R3-1 | The predicate asks only about `ws://<ip>:8000/ws/sdk` and `http://<ip>:8000/…`. When media is WebRTC, `ReachyMini._configure_mediamanager` also calls `find_producer_peer_id_by_name(wlan_ip or "localhost", 8443)`, a Python `websockets` connect with `proxy=True` (the default) to `ws://<signalling_host>:8443`. Probe (E1): with `no_proxy=10.6.0.63:8000,10.6.0.0/24`, :8000 is direct for both clients but :8443 is proxied. With a tunnel entry and `NO_PROXY=127.0.0.1` (v4's own hint), the tunnel dial is direct for both, but `ws://<LAN wlan_ip>:8443` goes through `http_proxy`. v4's NF1 fold makes this reachable, because it now accepts `wlan_ip` ≠ the dialled IP. The GStreamer signaller's own `ws://<wlan_ip>:8443` (`webrtc_client_gstreamer.py`, Rust) is a fourth client whose proxy behaviour nobody has checked | NF2/N5 (A8) | **fail-open** (side-effect egress; the proxy answers the producer lookup; no motion, since step 3 holds) | Yes |
| R3-2 | Spec mechanics. `get_proxy` takes a `WebSocketURI`: passing the string raises `AttributeError: 'str' object has no attribute 'host'` (E1), so the spec must say `get_proxy(parse_uri(url))`. `get_environ_proxies` returns the **whole** `getproxies()` dict whenever the URL is not bypassed (e.g. `{'https': …}`, `{'socks': …}`, `{'no': '127.0.0.1', 'http': …}`). "Says direct" therefore has to mean `select_proxy(url, get_environ_proxies(url)) is None`, not an empty dict. Read literally, it over-refuses `https_proxy`-only and `socks_proxy`-only setups. The spec does not say that a resolver exception refuses | NF2 | fail-closed (over-refusal, or a crash), provided "exception ⇒ refuse" is written down | Yes |
| R3-3 | The deny names a literal `~/.config/maxim/`. `runtime/config_loader.py` (the `config.json` path function) and `peer/config.py::peer_config_path` resolve `$XDG_CONFIG_HOME/maxim` when that variable is set. With it set, the live `mesh.yml`/`api_key`/`config.json` sit outside the deny, and the literal directory protects nothing | NF3 (A2) | **fail-open** | Yes |
| R3-4 | The deny uses the agent process's `data_home()`. Harness children run agents with `MAXIM_DATA_HOME=<tmp>` (`scripts/exp49/run_trials.py`, `exp44/campaign.py`, `benchmark_*`, `o19_rerun.py`, `grounding_census.py`, …). Their deny protects `<tmp>/robots.yaml`, not the operator's `~/.maxim/robots.yaml`, which the next real `maxim embody` reads. This is reachable whenever the child's `allowed_dirs` (workspace + cwd) contains `~/.maxim`, e.g. when cwd is `~` | NF3 (A2/N6) | **fail-open** (narrow: needs cwd to contain the home data dir) | Yes |
| R3-5 | The note says "case-folded realpaths" only. APFS is also **normalisation**-insensitive: a home or data-home path with a non-ASCII component (`José` NFC vs NFD) names the same file but does not compare equal after `casefold()`. #957 handles this class with `_same_entry` (stat identity) for existing entries, and the control-plane files normally exist | NF3 | **fail-open** (narrow) | Yes |
| R3-6 | "The two sandbox write paths" are not all of them. `CreateSandboxScriptTool` → `utils/sandbox_executor.py::SandboxExecutor.write_script` writes `<sandbox>/scripts/<name>.<py|sh>` with `name` **unsanitised and no realpath containment**: `name="../../…/.config/maxim/x"` lands inside the denied directory. That path has no #957 check either, so it is another site of the same pattern. It is dormant: no production code constructs `SandboxExecutor`, so `bootstrap.py` never registers the sandbox tools. The forced `.py`/`.sh` extension cannot produce `robots.yaml`, `device_state.json` or `mesh.yml` | NF3 (A2) | fail-closed in practice (unreachable; no consumer of `.py` files in the config dir) | Yes; fix in PR 0 anyway (≤10 lines, same pattern) |
| R3-7 | The stated residual names only "singularity's `bash`". `execute_file` (`MAXIM_ALLOW_EXECUTE_FILE`) and `run_tests` (`MAXIM_ALLOW_RUN_TESTS`, a model-chosen argv run on the host) are the same code-execution class. With `write_file` they can run any code. Also, the gate on `bash` is the `MAXIM_ALLOW_BASH` opt-in, not only the mode | NF3 overclaim | documentation (does not change enforcement) | Yes |
| R3-8 | `robots.yaml.bak` (named in delta-2 E2) is not in the v4 deny set. Poisoning it only matters if the operator restores from it | NF3 | fail-closed (operator-mediated) | Yes |
| R3-9 | The N3 precondition "while the registry holds a pinned entry" can be switched off by deleting every pin. §4.1 still maps an absent key to `UNPINNED`, and the step-5 headless fallback also keys on "no pinned entry". The file-tool route to that edit is closed only as far as D14 is complete (see R3-3, R3-4, R3-5). Code execution can forge a pin outright, so this adds nothing there. The delta-2 alternative closes it for every path: a writer-managed (`_format_version`) file must carry `hardware_id:` (`unpinned` explicitly), and a missing key is a `RobotsConfigError` | N3 | **fail-open**, conditional on a D14 gap | Yes |
| R3-10 | Spec contradiction. §3.1 says the rule "covers `MAXIM_DEVICE`, the legacy flags and step 5". §3a says "Legacy `--robot`/`--robot-name` launches are unchanged". §4.2 says an unpinned entry "needs `--unpinned` for `embody`/`--device`". Someone building from §3a ships the legacy-flag path unrefused | N3 | **fail-open** if built from §3a/§4.2 | Yes |
| R3-11 | Steps 2 and 4 (typed legacy `--robot-name`, and the env var `MAXIM_ROBOT_NAME`) use today's matching. A name that matches **no** entry falls through to the legacy `<name>.local` dial with no pin, even while the registry holds pins. The rule covers resolving to an unpinned **entry**, and "no name and no `primary:`" covers step 5 only. An inherited env var can therefore cause an unpinned mDNS dial on a pinned machine | N3 (A-mDNS) | **fail-open** (legacy behaviour kept, but not stated as a residual) | Yes |

## Evidence

### E1. Resolver probe (offline)

`env -i HOME=<mktemp> PATH=/usr/bin:/bin .venv/bin/python -I p.py`. Targets: `ws://10.6.0.63:8000/ws/sdk`, `ws://10.6.0.63:8443` (signalling), `ws://127.0.0.1:8000/ws/sdk` (tunnel) and `http://10.6.0.63:8000/api/x`.

| Env | ws :8000 | ws :8443 | tunnel ws | requests dict | requests `select_proxy` |
|---|---|---|---|---|---|
| `http_proxy`, `no_proxy=10.6.0.63` | direct | direct | **proxied** | `{}` | direct |
| `http_proxy`, `no_proxy=10.6.0.63:8000,10.6.0.0/24` | direct | **proxied** | proxied | `{}` | direct |
| `http_proxy`, `no_proxy=127.0.0.1` (tunnel hint) | proxied | **proxied** | direct | `{'http':…, 'no':…}` | proxied (the LAN URL; the tunnel's `http://127.0.0.1` is direct) |
| `https_proxy` only | proxied | proxied | proxied | `{'https': …}` | **direct** |
| `socks_proxy` only | proxied | proxied | proxied | `{'socks': …}` | **direct** |
| `get_proxy("ws://…")` passed as a string | `AttributeError` | | | | |

- Row 2 is R3-1: "both direct" for :8000 while :8443 is proxied.
- Row 3 is the tunnel case. The predicate (on `127.0.0.1`) passes, while the SDK's signalling lookup to the LAN `wlan_ip` is proxied.
- Rows 4–5 are R3-2: a non-empty dict does not mean the POST is proxied.

SDK sites:
- `reachy_mini.py::_configure_mediamanager` (`signalling_host=daemon_status.wlan_ip or "localhost"`, `daemon_url=self._daemon_http_url`, the dialled host) → `media_manager.py` → `media/webrtc_utils.py::get_producer_list` (`websockets.sync.client.connect`).
- `io/ws_client.py`: `ws_sync.connect(uri)` and `requests.post`.
- `websockets/sync/client.py::connect`: `proxy=True` by default → `get_proxy(ws_uri)`.

**Suggested fix:**
- Run the predicate over **every URL the SDK will open** for the chosen media backend: the sdk WS and the dial-host HTTP. When WebRTC is on, add `ws://<signalling_host>:8443`, where `signalling_host` is the `wlan_ip` that step 1 accepted, or `localhost`.
- If any of them resolves to a proxy, force `no_media` (the v4 fallback) rather than refuse the motion session.
- PR 1 records whether the GStreamer signaller honours proxies; if that is unknown, the rule is "any configured proxy ⇒ `no_media`".
- Pin rows 1–5 as unit tests.

### E2. The #957 write sites (NF3)

- `rg "git_dir_write_refusal\(" src/` finds exactly four: `tools/filesystem.py` (`WriteFileTool._is_path_allowed`, `EditFileTool._is_path_allowed`) and `tools/sandbox.py` (`WriteSandboxFileTool.execute`, `WriteToSharedOutputsTool.execute`).
- `rg "effect = .*WRITE" src/maxim/tools` adds only these:
  - `request_directory_change`: no file write;
  - `git_commit`: fixed argv;
  - `respond`: path chosen by `ResponseOutput`;
  - `create_sandbox_script`: R3-6.
- Effect.ALL tools: `bash`, `execute_file` and `run_tests` (R3-7).
- `rg "SandboxExecutor\(" src/` finds no constructor, so `bootstrap.py`'s sandbox branch is unreachable in production.
- Config dir: `runtime/config_loader.py` builds `Path(xdg) if xdg else Path.home()/".config"` / `"maxim"`. Its docstring says it mirrors `peer.config.peer_config_path`.

**Suggested text:**
- The denied set is computed from the **canonical resolvers**: the config-loader's config-dir function, `data_home()`, **and** the default `Path.home()/".maxim"`. Take the realpath of each.
- Compare case-folded, plus `_same_entry` stat identity when the target exists, exactly as #957 does.
- Include `robots.yaml.bak`.
- Route `write_script` through `contained_path` + both refusals.
- Widen the residual sentence to "any code-execution tool the operator opted into (`bash`, `execute_file`, `run_tests`)".

### E3. N3

- §3.1: "If any path resolves to an **unpinned** entry while the registry holds a **pinned** entry…".
- §3a: "Legacy `--robot`/`--robot-name` launches are unchanged".
- §4.2: "It needs `--unpinned` for `embody`/`--device` (§3a)".
- §4.1: "An absent key means `DevicePin.UNPINNED`".

**Suggested text:**
- (a) Adopt "writer-managed file ⇒ `hardware_id:` required (`unpinned` explicit)". That removes the precondition's dependence on the file's current contents.
- (b) Make §3a and §4.2 say what §3.1 says.
- (c) While the registry holds any pinned entry, a legacy name that matches no entry is refused unless `--unpinned` is given. Otherwise state the `<name>.local` fall-through as a residual and log it at WARNING with its source.

## Stop-rule outcome

**STOP RULE NOT MET.**
- These are fail-open in already-covered classes:
  - R3-1 (third SDK client; NF2/N5);
  - R3-3, R3-4, R3-5 (deny-set inputs; NF3);
  - R3-9, R3-10, R3-11 (N3).
- These are fail-closed or documentation, in covered classes: R3-2, R3-6, R3-7 and R3-8.

The findings are converging. Every one is a narrower instance of a class the note already handles. The fixes are spec text plus about 30 `src/` lines in PR 0/PR 1. A delta round 4 scoped to R3-1, R3-3/4/5 and R3-9/10/11 is owed. If that round finds only fail-closed issues in covered classes, the stop rule ends it there.
