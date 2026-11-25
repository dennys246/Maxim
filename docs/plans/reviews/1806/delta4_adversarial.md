# Device registry (#1806): delta round 4, adversarial lens (inputs and trust)

Reviewer: adversarial lens (inputs and trust), delta round 4, read-only, 2026-11-25, v5 working tree on 0630487a. Code read at the worktree's b7380ee7. SDK reachy-mini in `.venv`. Scope: R3-1, R3-3/4/5 and R3-9 only (§13 round-4 scope), plus any new fail-open defect those folds introduce. One offline filesystem probe ran under a throwaway directory (E2). No network, no robot, no sim.

**Verdict: BUILD after SHOULD-FIX. STOP RULE NOT MET.** R3-9 is closed. R3-1, R3-3, R3-4 and R3-5 are each the right shape but remain PARTIAL: each still has one narrow fail-open path. All of them fall in classes the note already covers (N5/E4 wrong-body media, NF3 deny-set inputs and matching, the R3-10 spec contradiction). None is DO-NOT-BUILD. The fixes are spec text plus about 25 `src/` lines in PR 0/PR 1.

## Re-check of the round-4 scope

| Fold | v5 status | Note |
|---|---|---|
| R3-1: every URL the media backend opens | **PARTIAL** (fail-open on backend choice; fail-closed over-refusal on WebRTC) | **What the SDK opens.** With `media_backend` `default`/`auto` and `connection_mode="network"` (forced at step 2), `_configure_mediamanager` always picks **WEBRTC**, unless the daemon reports `no_media`. WEBRTC opens three things:<br>1. `ws://<signalling_host>:8443` through Python `websockets` (`webrtc_utils.get_producer_list`). The resolvers can check this one.<br>2. The same URI through the GStreamer `webrtcsrc` signaller (`webrtc_client_gstreamer.py`, `signaller.set_property("uri", …)`), which is Rust code. The resolvers cannot check this one.<br>3. `requests` calls to `daemon_url = http://<dial host>:8000/api/media/*` at sound-operation time. This is the URL already checked.<br><br>**Is "unverifiable ⇒ no_media" implementable?** Yes. The backend can be decided before `ReachyMini(` from three inputs: the requested string, the forced network mode, and step 1's REST `no_media`.<br><br>**Two problems.** (a) An explicit LOCAL-class backend opens **no URL**, so the check passes without testing anything (R4-1). (b) The GStreamer signaller is used by **every** WEBRTC connect, so the rule as written forces `no_media` on every remote pinned connect (R4-2) |
| R3-3: XDG resolver | **PARTIAL** (narrow fail-open) | The resolved config directory is correct for the agent's own env. Two gaps remain. Unlike data_home, there is no default `~/.config/maxim` alongside the XDG-resolved directory. And every "default" path reads `$HOME` (R4-4) |
| R3-4: both data_home() and default `~/.maxim` | **PARTIAL** (narrow fail-open) | This closes the `MAXIM_DATA_HOME=<tmp>` child case. "Default `~/.maxim`" is `Path.home()`, which follows the child's `HOME`. A child run with `HOME=<tmp>` (the scratch-HOME practice, `sudo -H`, a launchd job) whose allowed roots contain the operator's real home leaves the real files outside the deny (R4-4) |
| R3-5: realpath + casefold + `_same_entry` | **PARTIAL** (narrow fail-open) | `_same_entry(target, denied)` is `os.path.samefile`, so it matches only an **existing denied file**. It never matches a file **inside** the denied config directory, and never a denied file that does not exist yet (a fresh machine's `robots.yaml`, a missing `util/device_state.json`). realpath does not canonicalise APFS firmlinks (probe E2) or NFC/NFD spelling. #957 itself handles this with an **ancestor walk** (`git_dir_write_refusal`'s loop), which the note does not ask for (R4-5). Hardlinks to existing denied files are caught by inode (R4-6). `~` is not an evasion (R4-7) |
| R3-9: writer-managed files require `hardware_id:` | **CLOSED** | Deleting one pin line in a writer-managed file is now a `RobotsConfigError`. An attacker who cannot write robots.yaml cannot strip the rule: the only stripping path is a file edit, which D14 denies to the agent file tools. **A silent "upgrade" is safe.** The writer's first rewrite of a legacy file emits `hardware_id: unpinned` for absent keys, which preserves meaning (absent already meant UNPINNED), and it takes `.bak` first. Nothing else writes `_format_version` into robots.yaml before PR 2. The residual is the **downgrade**: deleting `_format_version` as well. It adds no attacker capability (R4-8) |

## New findings

| # | Finding | Class | Fail-open / closed | Covered? |
|---|---|---|---|---|
| R4-1 | `media_backend` set to `local`, or to a deprecated alias (`gstreamer`, `gstreamer_no_video`, `sounddevice_*`), resolves to `MediaBackend.LOCAL`, and the SDK honours that **regardless of `connection_mode`** (`reachy_mini.py`: `MediaBackend(media_backend.lower())`, then `_resolve_backend`). LOCAL opens the **Maxim host's** camera IPC socket and local GStreamer audio. On a pinned network connect, perception then comes from whatever daemon runs locally (a sim) and from the host microphone, never from the robot. LOCAL opens no URL, so §4.2's "every URL that backend will open" check passes without testing anything. The value is reachable from robots.yaml `config: {media_backend: …}`, which the registry's constructor-signature filter passes through, and from the `selfy`/`connection.py` `media_backend=` parameter | N5 / E4 (media target is not the robot) | **fail-open** (wrong-body perception; no motion) | Yes |
| R4-2 | The GStreamer signaller is part of **every** WEBRTC connect, and WEBRTC is the only remote media backend. So "cannot be checked (the GStreamer signaller, until verified) ⇒ `no_media`" is unconditional: every pinned network connect runs blind, and PR 2b's strict default makes `embody exploration/live` always refuse | R3-1 / NF2 | fail-closed (over-refusal) | Yes |
| R4-3 | Stale text contradicts the folds. (a) §4.3 step 5 still says "the proxy check does not cover `wlan_ip:8443` or the GStreamer signaller", while §4.2 now covers both. (b) The §6 threats row for `<data_home>/robots.yaml` still says "a deny in `check_write_permission` (§5)", which §5 itself says has no production caller. Building from either section ships the old behaviour | R3-10 (spec contradiction) | **fail-open if built from §4.3 / §6** (documentation) | Yes |
| R4-4 | The denied set is computed only from the agent process's env. The XDG fold has no default-path partner: data_home gets both `data_home()` and default `~/.maxim`, while the config dir gets only the `$XDG_CONFIG_HOME`-resolved one. Both default entries use `Path.home()`, i.e. `$HOME`. A child whose `HOME` or `XDG_CONFIG_HOME` differs from the operator's, and whose allowed roots contain the operator's home, leaves the real `~/.config/maxim/*` and `~/.maxim/robots.yaml` unprotected | NF3 / R3-3, R3-4 | **fail-open** (narrow: needs a re-homed child with cwd over the real home) | Yes |
| R4-5 | `_same_entry` is applied only as a leaf comparison. Spellings that realpath leaves alone then evade the deny, and a case-folded string compare misses them too: the firmlink `/System/Volumes/Data/Users/<u>/…` (E2), and an NFD spelling of a non-ASCII component. They evade it for (i) any file under the denied **config directory**, and (ii) a **not-yet-existing** `robots.yaml` / `device_state.json`. Case (ii) is the plant-a-registry-on-a-fresh-machine case. Reachability is narrow: containment is an exact, case-sensitive prefix test on the realpath'd allowed roots, so the odd spelling must already be in an allowed root (an override root or `accessible_folders` spelled through the firmlink), or in a non-ASCII component below one | NF3 / R3-5 | **fail-open** (narrow) | Yes |
| R4-6 | Hardlinks. `samefile` catches a hardlink to an existing denied **file** (same inode). It misses a hardlink to a file **inside** the denied config directory, since there is no per-entry comparison. The agent file tools cannot create links. One needs code execution (the R3-7 residual) or a link that already exists | NF3 / R3-7 | fail-closed in practice | Yes |
| R4-7 | `~` expansion is not an evasion. `_rebase_relative` treats `~/…` as absolute and does not rebase it. `sanitize_path` (normpath) and `Path.write_text` never expand it, so the check (`realpath` of the same unexpanded string) and the write agree on `<cwd>/~/…`. Spec note: `control_plane_write_refusal` must take the same `real_path` string that `_is_path_allowed` passes to `git_dir_write_refusal`. If it re-expands `~` it only over-refuses | NF3 | fail-closed | Yes |
| R4-8 | R3-9 downgrade. Deleting `_format_version` **and** the pin lines returns the file to "hand-written legacy": absent means UNPINNED, and §3.1's pinned-registry rule switches off. That needs robots.yaml write access, which the D14 deny covers for file tools and which already suffices to forge a pin, so it adds no attacker capability. The non-adversarial case is an operator hand-edit that drops the header. Separately, §4.1 says the value "must be 16 lowercase hex" yet the writer emits `unpinned`. The literal (and its case) needs stating, or a strict parser refuses the writer's own output | N3 / R3-9 | fail-closed against an attacker; documentation | Yes |

## Evidence

### E1. SDK media paths (reachy-mini in `.venv`)

**Backend choice.** `reachy_mini.py::_configure_mediamanager`, in order:
1. `no_media` is honoured.
2. Daemon `no_media` is honoured.
3. `default`/`auto`: if `connection_mode == "localhost_only" and is_local_camera_available()`, LOCAL; otherwise WEBRTC.
4. Any other string goes to `MediaBackend(x)` with no `connection_mode` check (R4-1). An unknown string falls back to auto-detect.

**The enum** (`media_manager.py::MediaBackend`) has three primary values, plus deprecated aliases that `_DEPRECATED_BACKENDS` maps:
- `NO_MEDIA`, `LOCAL`, `WEBRTC`;
- `GSTREAMER`, `GSTREAMER_NO_VIDEO`, `SOUNDDEVICE_*` → `LOCAL`.

**What each backend opens:**
- **LOCAL** (`_init_camera` + `_init_audio`): `GStreamerCamera.open()` on the host IPC socket (`daemon/utils.py::is_local_camera_available`, `CAMERA_SOCKET_PATH`), plus host GStreamer audio. A camera failure only warns ("continuing without camera").
- **WEBRTC** (`_init_webrtc`):
  - `find_producer_peer_id_by_name(signalling_host, 8443)` → `websockets.sync.client.connect("ws://host:8443")`, with `proxy=True` by default;
  - `GstWebRTCClient`, whose signaller `uri = ws://{signaling_host}:{port}`;
  - `daemon_url` = the controller's `_daemon_http_url` (dial host :8000), used by `_requests.post/get/delete` for `/api/media/{stop_sound,clear_incoming_audio,play_sound,sounds/upload,sounds}`.
- **Not checked here:** the `webrtcsrc` STUN default. If it is a public STUN server, that is UDP egress, not a proxy question, and outside this scope.

**Maxim sources of the string:**
- `hardware/reachy/controller.py::__init__(media_backend="default")`;
- `hardware/registry.py` passes `config:` keys accepted by the constructor signature;
- `embodied_runtime/selfy.py` and `connection.py` (`media_backend: str = "default"`).

**Suggested fix (R4-1, R4-2):**
- For a pinned entry that is not `localhost_only`, **allowlist the backend**, not just its URLs: `default`/`auto`/`webrtc`/`no_media`. Any other value is a `RobotsConfigError` at load and a refusal in the controller (≈5 lines). A red gate covers `config: {media_backend: gstreamer}`.
- Replace the unconditional GStreamer clause with this rule: if **no** proxy is configured anywhere (every `*_proxy`/`all_proxy` env var in either case, and `urllib.request.getproxies()`, which includes macOS System Configuration, are all empty), the signaller cannot be proxied, so media is allowed. Otherwise force `no_media` until PR 1 records the signaller's proxy behaviour.

### E2. Firmlink probe (offline, scratch directory)

```
d=$(mktemp -d ~/.cache_probe_XXXX); touch $d/f
python3 -I: os.path.exists('/System/Volumes/Data'+d)            -> True
            os.path.realpath('/System/Volumes/Data'+d)          -> '/System/Volumes/Data/Users/…/.cache_probe_E8et'  (not canonicalised)
            os.path.samefile(fl+'/f', d+'/f')                    -> True
            os.path.samefile(fl, d)                              -> True
```

- realpath and case-folding leave the firmlink spelling distinct from the canonical one. Stat identity matches it, for directories as well as files.
- So the matcher must apply stat identity to **each existing ancestor** against **each denied directory** (`data_home()`, default data home, config dir). For a denied file that does not exist yet, it matches the deepest existing ancestor by stat identity and then compares the remaining tail NFC-normalised and case-folded.
- This is `git_dir_write_refusal`'s own loop (`tools/base.py`, the "Walk the existing ancestors" block), reused rather than reinvented.
- The containment check in `WriteFileTool._is_path_allowed` and `EditFileTool._is_path_allowed` (`real_path.startswith(allowed + os.sep)`, realpath'd roots, case-sensitive) is what keeps R4-5 narrow. Every mode's `allowed_dirs` is non-empty (`filesystem_policy.get_mode_filesystem_config`).

### E3. Env-derived roots (R4-4)

- `utils/paths.py::data_home`: `Path(MAXIM_DATA_HOME) or Path.home()/".maxim"`.
- `peer/config.py::peer_config_path` and `runtime/config_loader.py::config_path`: `Path(XDG_CONFIG_HOME) or Path.home()/".config"`.
- Every input is an env var.

**Suggested text:** the denied roots are the union of:
- the resolver outputs;
- the env-default paths (`Path.home()/".maxim"`, `Path.home()/".config/maxim"`);
- the account's real home from `pwd.getpwuid(os.getuid()).pw_dir`, joined with `.maxim` and `.config/maxim`.

That makes the R3-4 reasoning symmetric for XDG, and independent of `HOME`.

### E4. Write path (R4-7)

`filesystem.py`: `_rebase_relative` (skips `~` paths, because `expanduser` makes them absolute) → `sanitize_path` (normpath only) → `Path(validated_path)` → `_is_path_allowed(realpath(str(path)))` → `path.write_text`. No step expands `~`.

## Stop-rule outcome

**STOP RULE NOT MET.**
- Fail-open, all in already-covered classes and all narrow except R4-1:
  - R4-1 (LOCAL backend; N5/E4);
  - R4-3 (stale §4.3 / §6 text; R3-10);
  - R4-4 and R4-5 (deny-set inputs and matching; NF3).
- Fail-closed or documentation, in covered classes: R4-2, R4-6, R4-7, R4-8.

The findings are converging. Each is a narrower instance of a covered class, with spec-text fixes plus about 25 `src/` lines (a backend allowlist, a pwd-home root, and reuse of the #957 ancestor walk). A delta round 5 limited to R4-1, R4-3, R4-4 and R4-5 would be enough. If its findings are all fail-closed, the stop rule ends the pass there.
