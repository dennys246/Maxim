# #1806 delta 10: re-confirmation of the "Networked is defined as a positive test" bullet

Reviewer: safety lens, re-confirmation, read-only, 2026-11-25, @ 06a69848

Scope: exactly the §4.2 sub-bullet "**"Networked" is defined as a positive test**" and its sub-bullets (delta `d7877f15..06a69848`), checked against M9-1..M9-3 ([delta9_media_scope.md](delta9_media_scope.md)). Code read at b7380ee7 (`src/maxim/hardware/reachy/controller.py`, `src/maxim/hardware/config.py`). SDK read at `.venv/lib/python3.12/site-packages/reachy_mini/` (`reachy_mini.py`, `media/media_manager.py`, `daemon/utils.py`).

**Verdict: PARTIAL.** M9-1, M9-2 and M9-3 are closed as written. The LOCAL alias list is complete. The unpinned manual-SSH-forward `localhost_only` case is the stated residual and is acceptable as stated. One new narrow fail-open remains: the positive test keys on "not `tunnel: true`", but `tunnel:` has no type rule, while the controller reads it by truthiness. That is the same loader/consumer desync class as R5-1 and M9-3, applied to a third key. The fold is one line.

## Findings

| ID | Severity | Finding | Fail mode | Proposed fold |
|---|---|---|---|---|
| M10-1 | SHOULD-FIX | **`tunnel:` is untyped, but it is now a safety-classifier input.** The bullet allows LOCAL only when the entry "is not `tunnel: true`". The note types `primary:` (`type(v) is bool`, §3) and `connection_mode`/`media_backend` (§4.2), but not `tunnel:`. The controller keeps the raw value (`self._tunnel = tunnel`) and branches on truthiness (`if self._tunnel:` in `connect`; `tunnelled = self._tunnel or ...`). Take an entry with a quoted `tunnel: "true"` (or `tunnel: 1`), `connection_mode: localhost_only` and `media_backend: local`. A classifier that implements the text literally (`tunnel is True`) reads it as not-a-tunnel and allows LOCAL. The controller then opens SSH to the remote `host`, and the SDK honours the explicit `MediaBackend("local")` against the Maxim host's IPC socket. Perception comes from the wrong body. The reverse case (`"false"` string) is fail-closed. | fail-open under a literal reading (wrong-body camera). Same class as R5-1/M9-3 (Maxim's classification and the consumer's reading can disagree) | Type `tunnel:` like `primary:`: `type(v) is bool` at load, anything else is a `RobotsConfigError`. The classifier and the controller then read one coerced bool, so they cannot disagree. Add a red gate: quoted `tunnel: "true"` + `localhost_only` + `local` is refused at load. |
| M10-2 | NIT | The bullet states "unknown values and non-strings are `RobotsConfigError` at load" and "the controller passes the canonical literal", but neither of the five red gates covers them. (`Network` + `local` covers the `.lower()` match only.) | none (gate coverage) | Add one red gate: `connection_mode: bogus` and `connection_mode: true` are both `RobotsConfigError`. Have a fake `ReachyMini` assert that it received `"network"` for a `Network` entry. |
| M10-3 | NIT | The residual sentence names only the Lite-on-a-Mac example. It does not name the case the owner is actually accepting: an unpinned, non-tunnel `localhost_only` entry that is really a hand-run `ssh -L 8000` forward. | none (stated residual, wording) | Add "or a hand-run SSH forward" to the residual sentence, so the UNPINNED label is known to cover it. |

## Confirmed closed

- **M9-1 (`auto`).** `auto` is now networked for the rule. LOCAL + `auto` is refused at load and in the controller, the changelog lists it, and a red gate covers it. This is correct: `_initialize_client` falls back to `self.host` with `selected = "network"`, and `_configure_mediamanager` runs inside the same `__init__`.
- **M9-2 (`tunnel: true`).** A tunnel is now networked regardless of the controller's forced transport mode. Unpinned tunnels dial `127.0.0.1` in `network` mode, so `self.connection_mode == "network"` and `default`/`auto`/unknown resolve to WEBRTC (`_configure_mediamanager`: LOCAL needs `connection_mode == "localhost_only"`). A red gate covers it, subject to M10-1.
- **M9-3 (case).** `.lower()` matches `_normalize_connection_mode` and `_initialize_client` exactly. Unknown values and non-strings are refused at load, and the canonical literal is passed on.
- **Absent key.** An absent key is networked, matching the controller default `connection_mode="network"`. A red gate (absent + `gstreamer`) covers it.
- **The LOCAL alias list is complete.** An explicit string is resolved by `MediaBackend(media_backend.lower())`, a value lookup. The values that land on LOCAL are `local`, plus the four `_DEPRECATED_BACKENDS` keys `gstreamer`, `gstreamer_no_video`, `sounddevice_no_video` and `sounddevice_opencv`. That is exactly the note's five. `DEFAULT` and `DEFAULT_NO_VIDEO` are enum aliases by member name only. The string `default` is special-cased earlier, and `default_no_video` raises `ValueError` and so goes to the auto-detect fallback. That fallback is LOCAL only when `connection_mode == "localhost_only"`, which the networked cases never reach. No sixth string reaches LOCAL in network mode.

## The residual (unpinned, non-tunnel `localhost_only` that is really a manual SSH forward)

Under `default`/`auto` (or an unknown string), this entry still picks LOCAL whenever `is_local_camera_available()` is true. On macOS/Linux that is a bare `os.path.exists(CAMERA_SOCKET_PATH)`, so a stale socket left by a crashed local daemon or sim counts. An explicit `local` is honoured too. This is exactly "`localhost_only` entries that cannot be proven keep today's behaviour, labelled UNPINNED".

**It is acceptable as stated.** An unpinned entry has no identity to prove against, and nothing on the host can tell a port-forward from an onboard daemon. So no positive test exists, and refusing would break the real onboard/Lite case. The remedy is the existing pin path: a pinned manual tunnel fails the onboard proof and dials network mode (§4.3 step 2), so it can never pick LOCAL. The plan block already prints UNPINNED and the media backend (§3 step 3). This is pre-existing behaviour, not a regression. Only wording is owed (M10-3).

## Evidence

- `controller.py::ReachyMiniController.__init__`: `tunnel: bool = False`, then `self._tunnel = tunnel`, with no coercion. `connect`: `if self._tunnel: ... effective_mode = "localhost_only"`. Also `tunnelled = self._tunnel or self._connection_mode == "localhost_only"`.
- `config.py`: the robot `config:` is a free-form dict (host, connection_mode, tunnel, ...). No type check on `tunnel` exists today, and the note adds none (grep: only `primary:` gets `type(v) is bool`).
- `media_manager.py`: `MediaBackend` values and `_DEPRECATED_BACKENDS` (4 keys → LOCAL); `DEFAULT = LOCAL` and `DEFAULT_NO_VIDEO = GSTREAMER_NO_VIDEO` are name aliases.
- `reachy_mini.py::_configure_mediamanager`: the `no_media`, then `default`/`auto`, then explicit/unknown branches. Both auto-detect paths gate LOCAL on `self.connection_mode == "localhost_only" and is_local_camera_available()`.
- `daemon/utils.py::is_local_camera_available`: `os.path.exists(CAMERA_SOCKET_PATH)` on non-Windows.

## Stop-rule outcome

**STOP RULE NOT MET**, narrowly. M10-2 and M10-3 are NITs, and the manual-forward case is a stated residual. M10-1 is a fail-open under a literal reading of the new text, in the already-covered R5-1/M9-3 desync class. The fold is one line ("`tunnel:` is `type(v) is bool` at load") plus one red gate. If the author folds M10-1 exactly as proposed, I would accept it without another round, since the fold is mechanical and fail-closed. That decision is the author's, under the "Faster without lowering the bar" rule.
