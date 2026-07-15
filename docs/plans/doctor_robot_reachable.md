# `maxim doctor` — robot reachability check (`check_robot_reachable`)

**Status:** PLANNED (2026-07-15, from the Reachy WS-era review round). One
PR-sized change. The spec below is the contract; a condensed TODO pointing
here lives in `src/maxim/doctor/checks.py`.

## Motivation

The 2026-07-15 Reachy bring-up burned hours on failures that a doctor check
would have named in seconds: SDK↔daemon **era mismatch** after a reflash
(zenoh 1.2.6 client vs WebSocket 1.8.3 daemon — unfixable by any network
debugging), macOS **Local Network permission** making the whole LAN look
dead, `.local` **resolution flakes** from venv python, and a daemon that
answers HTTP while its robot backend isn't ready. Today none of `maxim
doctor`'s checks know robots exist; the Reachy diagnostics live in two
near-duplicate standalone surfaces (`src/maxim/utils/reachy_diagnostics.py`
→ `maxim-diagnostics` console script, and `scripts/check_reachy_connection.py`)
that drift from each other (the ssh-hint format bug existed in exactly one
of them).

**Front-gate (ride existing infra?):** yes — this is one new check function
on the existing doctor surface. No new mechanism, no new module, no new
config. The doctor already has the exact pattern needed: pure functions
returning `CheckResult`, platform-aware fix strings with detected values
filled in, a data-driven `--retry` loop keyed on `retry_id`, and
lazy-imports-inside-the-function for fast startup.

## Design

### Function

`check_robot_reachable(info: PlatformInfo) -> list[CheckResult]` in
`src/maxim/doctor/checks.py`. One `CheckResult` per configured robot (plus
the gate result). Wired into `run_all_checks()` for the leader/solo branch
(robots are orthogonal to peer/leader roles, but a peer machine is never
the robot operator today — revisit if that changes).

All imports (`maxim.hardware.config`, `maxim.utils.http`,
`importlib.metadata`, `socket`) live **inside the function body** per the
doctor fast-start rule.

### Gate (never a failure)

`load_robots_config()` (`maxim.hardware.config`, search paths
`~/.maxim/robots.yaml` then `./robots.yaml`) — if no config file or zero
robots: return a single `info` result, `"no robots configured — skipping
robot checks (add ~/.maxim/robots.yaml to enable)"`. A machine without
robots must never fail doctor because of this check.

### Per-robot probe sequence (mirrors the hardware-validated order)

| # | Probe | Timeout | On failure |
|---|---|---|---|
| 1 | **Resolve host** — config `host:` used verbatim (skip resolution); else IPv4-only `socket.gethostbyname("<name>.local")` (the IPv4-only call sidesteps the getaddrinfo/IPv6-first flake; TCC can still block it) | ~2 s | `fail` — fix string branches on macOS (below) |
| 2 | **TCP :8000** (`socket.create_connection`) — the single WS-era control port | 1.5–2 s (doctor's peer-probe convention; the <1 s budget bends here exactly as it does for mesh probes) | `fail` |
| 3 | **`GET /api/daemon/status`** via `maxim.utils.http.fetch_url` (never raw urllib — CI grep) → extract `version`, `state` | 2 s | `warn` (port open but API unreadable → "pre-1.5 daemon or endpoint moved") |
| 4 | **Era/pin coherence** — daemon `version` vs local `importlib.metadata.version("reachy-mini")` (metadata, not `__version__` — pre-1.5 SDKs don't have the attribute) | — | cross-era (either side < 1.5) → `fail`; same era but minor drift → `warn`; match → fold into the `ok` message (`"reachy_mini reachable (daemon 1.8.3, state=running, SDK match)"`) |

Optional 5th (informational only, never fail): `state != "running"` →
`warn` with "daemon up but backend not ready — motion/WS connects will be
refused (1013); journalctl -u reachy-mini-daemon on the robot".

### Fix strings (copy-paste runnable, real values filled in)

- **Resolution failure, macOS:** `"grant this terminal Local Network
  permission (System Settings → Privacy & Security → Local Network) and/or
  set host: <ip> in ~/.maxim/robots.yaml (DHCP reservation recommended)"`.
  Non-macOS: the `host:` half only.
- **:8000 closed:** `"ssh pollen@<resolved-ip> → systemctl status
  reachy-mini-daemon; after a Wi-Fi change, reboot the robot (daemon binds
  at startup)"`.
- **Cross-era mismatch:** the exact command — `"pip install
  'reachy_mini==<daemon-version>' (daemon <dv> vs SDK <lv> are on opposite
  sides of the v1.5.0 zenoh→WebSocket pivot; see
  docs/embodiment/reachy_mini/troubleshooting.md)"`.
- **Minor drift:** same command as a `warn`-level suggestion.

### Retry loop

`retry_id="robot"` on every non-ok robot result; register one callable in
`cli._retry_loop`'s `retryable_fns` that re-runs `check_robot_reachable`
(the whole list — per-robot retry ids are overkill at ≤2 robots; revisit
with the multi-robot registry work).

## Consolidation (follow-through, same PR or the next)

1. `src/maxim/utils/reachy_diagnostics.py` (`maxim-diagnostics`) becomes a
   thin wrapper: parse args → run `check_robot_reachable` → print in its
   current style. Kills the duplicate probe logic; keeps the console-script
   entry point (it's shipped in pyproject).
2. `scripts/check_reachy_connection.py` gets a deprecation header pointing
   at `maxim doctor` + the wrapper, then is deleted one release later
   (it is a line-for-line twin of the utils module and has already drifted
   once).
3. `docs/embodiment/reachy_mini/troubleshooting.md` gains "`maxim doctor`"
   as the first line of the fast discriminating sequence.

## Testing (all offline — doctor rule: mock network/process calls)

In `tests/unit/test_doctor.py`, one class `TestCheckRobotReachable`:

- **Gate:** no config / empty config → single `info`, status never fail.
  (Patch `maxim.hardware.config.load_robots_config` — the ORIGINAL module
  path, per the doctor testing note.)
- **Happy path:** host + port + status + era all good → `ok`, message
  carries daemon version/state.
- **Resolution failure:** `gethostbyname` raising `socket.gaierror` →
  `fail`; macOS `PlatformInfo` → fix mentions Local Network; linux → only
  `host:` advice.
- **Port closed:** `fail`, fix carries the resolved IP in the ssh command.
- **Status unreadable:** port ok, `fetch_url` raises → `warn` "pre-1.5
  daemon or endpoint moved".
- **Era matrix:** (client 1.2 / daemon 1.8) → fail; (1.8 / 1.2) → fail;
  (1.8.3 / 1.8.4) → warn; (1.8.3 / 1.8.3) → ok. Patch
  `importlib.metadata.version`.
- **Backend not ready:** `state="starting"` → warn with the 1013 hint.
- **retry_id** present on every non-ok result.

## Regression guards

- The `TestCheckRobotReachable` class above (the check's own guard).
- The existing WS-era transport invariant in CLAUDE.md already covers the
  probe semantics (`tests/unit/test_reachy_connection_options.py` pins
  :8000-not-:7447 at the controller); this check reuses, not re-derives,
  those facts.
- After consolidation step 1, the wrapper keeps
  `tests/unit/test_reachy_connection_options.py`-style coverage via the
  doctor tests — no separate probe-logic tests to maintain twice.

## Non-goals

- No robot AUTO-fix (`--fix` stays explicit opt-in per doctor rules; we
  never restart daemons or rewrite robots.yaml).
- No motion/torque probing (doctor checks must stay fast and side-effect
  free; enabling torque is not a diagnostic).
- No zenoh-era (< 1.5) support — the era check NAMES the mismatch; it does
  not accommodate it.
- Peer-machine robot checks — out of scope until a peer actually operates
  a robot.
