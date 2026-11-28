# Device registry (#1806): delta round 5 check, adversarial lens (inputs and trust)

Reviewer: adversarial lens, delta round 5 check, read-only, 2026-11-25, v6 @ b0cbd8f7. Code read at the worktree's b7380ee7 (`tools/base.py`, `hardware/reachy/controller.py`, `embodied_runtime/selfy.py`). SDK reachy-mini in `.venv`. Scope: R4-1, R4-4 and R4-5 only, plus any new fail-open defect those folds introduce. One offline probe ran under `env -i HOME=$(mktemp -d)` (E3). No network, no robot, no sim.

**Verdict: BUILD after SHOULD-FIX. STOP RULE NOT MET.** R4-1 is closed as a set: no string the allowlist admits can resolve to LOCAL. Its one residual is a normalisation desync on `no_media` (R5-1). R4-4 is PARTIAL: the union covers re-homing, but not an operator's non-default `MAXIM_DATA_HOME`/`XDG_CONFIG_HOME` or a different uid (R5-2). R4-5 is PARTIAL: #957 does have a reusable ancestor walk, but the v6 prose describes a different algorithm (the deepest existing ancestor only). Built as written, that misses an overwrite of an **existing** robots.yaml (R5-3). Every finding is in a covered class (N5/E4, NF3, R3-10 spec contradiction). Each fix is one or two sentences of spec text and no new `src/` scope.

## Re-check

| Fold | v6 status | Note |
|---|---|---|
| R4-1: media-backend allowlist `{default, auto, webrtc, no_media}` | **CLOSED** (one spec nit, R5-1) | **What the SDK does with the string** (`reachy_mini.py::_configure_mediamanager`): it compares `media_backend.lower()` to `"no_media"`, then to `("default","auto")`, and otherwise calls `MediaBackend(media_backend.lower())`. An unknown value falls back to auto, which is WEBRTC unless `connection_mode == "localhost_only"`.<br><br>**What resolves to LOCAL** (`media_manager.py`): `local`, plus `gstreamer`, `gstreamer_no_video` and `sounddevice_{no_video,opencv}` through `_DEPRECATED_BACKENDS`. The enum members `DEFAULT` and `DEFAULT_NO_VIDEO` are aliases for `"local"` and `"gstreamer_no_video"`, but no string value `"default"` reaches the enum, because the string path catches it first. Case variants of any of these are LOCAL, because of `.lower()`.<br><br>**Why the allowlist excludes LOCAL.** Under any plausible Maxim normalisation (none, `lower`, `casefold`, `strip`, NFKC), a string whose `.lower()` is a LOCAL name normalises to that same LOCAL name, which is not in the allowlist. Unknown values are refused (fail-closed). Even if one slipped through, it would fall back to WEBRTC in network mode, not LOCAL. `connection_mode` stays `network` once requested (`_initialize_client`).<br><br>**Enforcement points.** In `src/`, the controller's `ReachyMini(` call is the one constructor that the registry reaches, so the controller check is the chokepoint. The load check and the `selfy` check are earlier copies of it. `connection.py`'s field flows into `selfy`. The other `ReachyMini(` sites (`audio_localization.py`, `scripts/reachy_streamer.py`) are not registry paths and are out of scope here |
| R4-4: denied roots = resolver ∪ `Path.home()` default ∪ pwd home | **PARTIAL** (narrow fail-open, R5-2) | The union now covers `HOME=<tmp>`, `XDG_CONFIG_HOME=<tmp>` and `MAXIM_DATA_HOME=<tmp>` children (E2: `Path.home()` follows `HOME`, `pwd` does not).<br><br>`MAXIM_DATA_HOME` pointing at the real `~/.maxim`'s parent (`MAXIM_DATA_HOME=~`) is harmless: the pwd/default `~/.maxim` entries still deny the real files.<br><br>**Two paths remain.**<br>(a) An operator whose own process uses a **non-default** `MAXIM_DATA_HOME` or `XDG_CONFIG_HOME`, with a harness child that re-points it. The child's union is {its tmp, `~/.maxim`, pwd `~/.maxim`}, which never includes the operator's real registry location.<br>(b) A **different uid**: root, or `sudo -u`. `pwd.getpwuid(os.getuid())` is then the other account's home, so the operator's home is outside the set.<br><br>The note's sentence "a child re-homed … therefore cannot unprotect the operator's files" overstates. Neither path is agent-reachable: the agent cannot set its own env or uid. Both are launcher configuration |
| R4-5: reuse #957's ancestor walk | **PARTIAL** (fail-open if built as written, R5-3) | **The walk exists and is reusable.** `git_dir_write_refusal`'s "Walk the existing ancestors" loop visits **every** existing ancestor of `real`, up to `/`, and tests each by stat identity (`_same_entry(current, parent/.git)`). Swapping that test for `_same_entry(current, denied_root)` plus an NFC/casefold compare of `relpath(real, current)` is sound when every ancestor is visited (E3, second block: every case refused).<br><br>**The v6 text describes something else:** "finds the **deepest** existing ancestor, compares it … with each denied root". The deepest existing ancestor is the file itself when it exists, or an existing subdirectory. So, built as written, the check passes:<br>1. an overwrite of an **existing** `robots.yaml` (E3 case F);<br>2. a new `util/device_state.json` while `util/` exists (B);<br>3. a new file under an existing subdirectory of the config dir (E).<br><br>v6 also drops v5's realpath+casefold string compare, and the walk never matches a denied root that does not exist yet. A fresh machine with no `~/.config/maxim` then admits `~/.config/maxim/<file>`, even spelled canonically (C, D) |

## New findings

| # | Finding | Class | Fail-open / closed | Covered? |
|---|---|---|---|---|
| R5-1 | **Allowlist normalisation desync on `no_media`.** The SDK takes the no-media branch only when `media_backend.lower() == "no_media"`. If Maxim's allowlist check normalises more than that (`strip`, NFKC or casefold, the latter two both used in §5's path matching), a value such as `" no_media"` or `"ｎｏ_media"` is classified `no_media` by Maxim, but the SDK treats it as unknown, and unknown means auto, which is WEBRTC. Maxim's media-side checks (the proxy rule, the `wlan_ip is None` rule, `media_required`) are then skipped as moot while the SDK opens `ws://<signalling_host>:8443`. That host may be `localhost`, i.e. the wrong body. Reach is low: the value comes from operator-written robots.yaml or code. **Fix (one sentence):** match exactly on `value.lower()`, as the SDK does, and pass the **matched canonical literal** to `ReachyMini(`, never the raw string | N5 / E4 (media target is not the robot) | **fail-open** (narrow; only if built with a different normalisation) | Yes |
| R5-2 | **The root union is still derived from env and uid.** It does not include (a) the operator's non-default `MAXIM_DATA_HOME`/`XDG_CONFIG_HOME` when a harness child re-points them, or (b) the operator's home when the child runs under another uid. **Fix:** either add an env-independent name rule (deny any path whose case-folded tail is `.maxim/{robots.yaml, robots.yaml.bak, util/device_state.json}` or under `.config/maxim/`, at any root), and state (a) as a residual: an operator-chosen custom data home is protected only in processes that inherit it. Or have the #1593 `child_env` seam pass the parent's resolved roots down. Then correct the "cannot unprotect" sentence | NF3 / R4-4 | **fail-open** (narrow; launcher configuration, not agent-reachable) | Yes |
| R5-3 | **The matching prose contradicts the #957 loop it cites.** "Deepest existing ancestor" misses an existing robots.yaml overwrite, existing intermediate directories, and any not-yet-existing denied root. **Fix (spec text):**<br>1. Walk **every** existing ancestor, as `git_dir_write_refusal` does.<br>2. Anchor each denied root that does not exist at its own deepest existing ancestor, plus a tail; or keep v5's realpath+casefold string compare alongside the walk.<br>3. Add red gates for an existing-file overwrite and an absent config dir | NF3 / R3-10 (spec contradiction) | **fail-open if built from §5 as written** | Yes |

## Evidence

### E1. SDK resolution (`.venv/.../reachy_mini/`)

- `media/media_manager.py::MediaBackend` has these members:
  - `NO_MEDIA="no_media"`, `LOCAL="local"`, `WEBRTC="webrtc"`;
  - `DEFAULT=LOCAL`;
  - `GSTREAMER`, `GSTREAMER_NO_VIDEO`, `SOUNDDEVICE_NO_VIDEO`, `SOUNDDEVICE_OPENCV`, and `DEFAULT_NO_VIDEO=GSTREAMER_NO_VIDEO`.
- `_DEPRECATED_BACKENDS` maps all four deprecated members to `LOCAL`.
- `reachy_mini.py::_configure_mediamanager` checks, in order:
  1. `lower()=="no_media"`;
  2. daemon `no_media`;
  3. `lower() in ("default","auto")`, which is LOCAL only if `localhost_only` and the IPC camera is present, else WEBRTC;
  4. `MediaBackend(lower())`, which honours LOCAL in any mode; a `ValueError` means auto-detect, as in step 3.
- `_initialize_client` keeps `network` when it is requested. A reconnect (`reachy_mini.py:233`) reuses the string that was passed, which is the controller's per-connect value.

### E2. Home resolution

`env -i HOME=/tmp/xx python3 -I`:

- `Path.home()` returns `/tmp/xx`;
- `pwd.getpwuid(os.getuid()).pw_dir` returns the account's real home.

So the pwd entry closes the re-home case, but it follows uid, not the operator.

### E3. Matching probe

The probe is at `/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/fc699e42-5cb0-4089-8c73-d65cffebfe42/scratchpad/probe_r5.py`. It ran in a scratch tmpdir with roots `<h>/.maxim` (with `util/` present) and `<h>/.config/maxim` (absent at first).

**First block: v6's text, built literally** (deepest existing ancestor, stat-compared to each root, then an NFC+casefold tail):

```
A new robots.yaml (ancestor=.maxim)                 -> refused
B new util/device_state.json (deepest=util)         -> ALLOWED (fail-open)
C config dir absent, new config.yaml                -> ALLOWED (fail-open)
D config dir absent, case variant                   -> ALLOWED (fail-open)
E config dir exists, new file in existing subdir    -> ALLOWED (fail-open)
F EXISTING robots.yaml overwrite                    -> ALLOWED (fail-open)
```

**Second block: the #957-style walk over every ancestor,** run after the config dir had been created:

```
A, B, C, D, E, F -> refused
```

The walk is sound when the root exists. When the root is absent, `samefile` can never match it, so absent roots need anchoring or the string compare (R5-3).

## Stop-rule outcome

**STOP RULE NOT MET.** Three findings are fail-open, all narrow and all in already-covered classes:

- R5-1: the allowlist normalisation, N5/E4;
- R5-2: env- and uid-derived roots, NF3;
- R5-3: matching prose that contradicts the #957 walk, NF3/R3-10.

The findings are converging. Each fix is spec text: use exact `lower()` and pass the canonical literal; add a name rule or a stated residual, and correct one sentence; walk every ancestor, and anchor absent roots or keep the string compare. None needs new `src/` scope beyond what PR 0 already carries. A round 6 limited to checking those three sentences would be enough.
