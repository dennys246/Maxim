# Peer Update — Pip/Dev dual-mode

**Status:** IMPLEMENTED (2026-04-18). 3-lens review folded. All 3 stages shipped.
**Scope:** ~250-300 LOC across 3 stages.
**Target version:** 0.3.1
**Gates:** Nothing. Quality-of-life for non-dev deployments.
**Depends on:** Plan 4 C3.5 (SHIPPED — `admin_core.py` + mesh-aware update verbs).
**Blocks:** Nothing. Enables pip-only leader deployments without a git checkout.
**Parent:** [reactive_peer_mesh_roadmap.md](../reactive_peer_mesh_roadmap.md) (feeds C8 cross-version compatibility story).

## Goal

Today `maxim peer update` assumes the leader runs from a git checkout. The handler at [leader_proxy.py:944](../../src/maxim/runtime/leader_proxy.py#L944) derives `repo_root` from `Path(__file__).parents[3]` and runs `git fetch` → `git pull --rebase` → `pip install -e .`. A pip-only install has no git repo — the handler crashes on the first `git status`.

Users who deploy Maxim via `pip install pymaxim` on a headless server should be able to update with `maxim peer update` just like dev users. The server should auto-detect its install mode and do the right thing.

**Two concrete outcomes:**

1. **Pip-installed leaders update via `pip install --upgrade pymaxim[...]`** — extras are auto-detected and preserved.
2. **Dev (git) leaders keep the current behavior unchanged** — `git pull --rebase` + `pip install -e .`.

## CLI surface

```bash
# Auto-detect mode (pip if pip-installed, dev if git checkout)
maxim peer update                      # upgrade pymaxim to latest PyPI release
maxim peer update --dry-run            # show current vs latest version (pip) or pending commits (dev)
maxim peer update --version 0.3.1      # pin to specific PyPI version (pip mode only)

# Force dev mode (errors if no git repo on leader)
maxim peer update --dev                # git pull origin/main + pip install -e .
maxim peer update --dev feat/foo       # git pull origin/feat/foo + pip install -e .
maxim peer update --dev --force        # stash dirty tree first (existing behavior)

# Mesh-aware (inherits mode from target node)
maxim peer --node rtx-5080 update                 # drain → update (auto-detect) → resume
maxim peer --node rtx-5080 update --dev feat/foo   # drain → update (dev) → resume
```

**Argument rules:**
- `--dev` switches to git mode. Without it, server auto-detects.
- Positional arg after `--dev` is the branch (default: `main`). `--branch` flag is kept as an alias for backward compatibility with existing scripts/docs.
- `--version X.Y.Z` is pip-mode only. Passing it with `--dev` is a client-side error (print message + exit 1, no HTTP call).
- `--force` only applies to dev mode (git stash). Ignored in pip mode (pip has no dirty-tree concept).
- `--dry-run` works in both modes: pip reports current vs available version, dev reports pending commits.

## Wire protocol

**Request body** (POST `/v1/admin/update`):

```json
{
  "mode": "auto",
  "branch": "main",
  "dry_run": false,
  "force": false,
  "version": null
}
```

New fields:
- `mode`: `"auto"` | `"pip"` | `"dev"`. Default `"auto"` for backward compatibility — existing peers that don't send `mode` get `"auto"`.
- `version`: `str | null`. Only valid when mode resolves to pip. Pinned PyPI version.

Existing fields unchanged: `branch`, `dry_run`, `force`.

**Important: `dry_run` server default is `True`** (safe-by-default). The existing server at [leader_proxy.py:1048](../../src/maxim/runtime/leader_proxy.py#L1048) uses `body.get("dry_run", True)`. The client always sends `dry_run` explicitly. New code MUST preserve this server-side `True` default — omitting the field from a request body should preview, not mutate. (Review fold: Protocol lens, severity BLOCKING.)

**Response body — pip mode:**

```json
{
  "status": "updated",
  "install_mode": "pip",
  "from_version": "0.3.0",
  "to_version": "0.3.1",
  "extras_preserved": ["semantic", "llm-llama"]
}
```

```json
{
  "status": "up_to_date",
  "install_mode": "pip",
  "current_version": "0.3.0",
  "message": "Already at latest version."
}
```

```json
{
  "status": "preview",
  "install_mode": "pip",
  "current_version": "0.3.0",
  "latest_version": "0.3.1",
  "extras_detected": ["semantic", "llm-llama"],
  "pending_commits": ["0.3.0 → 0.3.1"],
  "message": "0.3.0 → 0.3.1 available. Send dry_run=false to apply."
}
```

**Old-client backward compat for preview:** The `pending_commits` field in pip preview responses contains a synthetic `["0.3.0 → 0.3.1"]` entry. Old clients (0.3.0) read `pending_commits` and display `"1 pending commit(s): 0.3.0 → 0.3.1"` — not perfect, but informative. New clients use `current_version`/`latest_version` directly. (Review fold: Protocol lens, severity MAJOR.)

**Response body — dev mode:** unchanged from today (`status`, `branch`, `commits_applied`, `pip_output`), plus `"install_mode": "dev"`.

## Server-side install mode detection

At [leader_proxy.py:944](../../src/maxim/runtime/leader_proxy.py#L944), before branching into git or pip logic:

```python
def _detect_install_mode() -> str:
    """'dev' if editable git install, 'pip' if standard pip install."""
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / ".git").is_dir():
        return "dev"
    return "pip"
```

Mode resolution: `"auto"` calls `_detect_install_mode()`. `"dev"` asserts `.git` exists (409 if not). `"pip"` skips the git check.

## Extras auto-detection

The logic already exists at [leader_proxy.py:1358-1377](../../src/maxim/runtime/leader_proxy.py#L1358) (`_handle_debug_deps`). Extract the `extra_checks` dict into a module-level constant so both `_handle_debug_deps` and the new pip-upgrade path can reuse it:

```python
_EXTRA_IMPORT_MAP: dict[str, str] = {
    "semantic": "sentence_transformers",
    "llm-llama": "llama_cpp",
    "llm-torch": "torch",
    "llm-anthropic": "anthropic",
    "llm-openai": "openai",
    "vision": "cv2",
    "audio": "sounddevice",
    "search": "duckduckgo_search",
    "tts": "piper",
    "yolo": "ultralytics",
}

def _detect_installed_extras() -> list[str]:
    """Return list of pymaxim extras currently importable AND in the allowlist."""
    detected = [name for name, mod in _EXTRA_IMPORT_MAP.items()
                if _try_import(mod)]
    return [e for e in detected if e in _ALLOWED_EXTRAS]
```

**Allowlist filter is mandatory** (Review fold: Security lens, severity MAJOR). Detected extras MUST be validated against the existing `_ALLOWED_EXTRAS` set before being passed to any pip command. If `_EXTRA_IMPORT_MAP` and `_ALLOWED_EXTRAS` ever diverge, the allowlist wins. Unify the two into a single source of truth where practical — `_EXTRA_IMPORT_MAP` keys should be a subset of `_ALLOWED_EXTRAS`.

## Stages

### Stage 1 — Server-side pip update path

**The core change.** Add `_run_pip_upgrade()` alongside existing `_run_pip_install()` in [leader_proxy.py](../../src/maxim/runtime/leader_proxy.py).

**What's built:**

1. **`_detect_install_mode()`** — checks for `.git` dir. ~5 LOC.

2. **`_detect_installed_extras()`** — extracted from existing `_handle_debug_deps` logic. Shared constant `_EXTRA_IMPORT_MAP`. ~15 LOC.

3. **`_get_current_version()`** — `importlib.metadata.version("pymaxim")`. ~5 LOC.

4. **`_get_latest_pypi_version()`** — `pip index versions pymaxim` (pip >=21.2) or fallback to `pip install pymaxim==__dummy__` error parse. Only called for dry-run preview. ~15 LOC.

5. **`_run_pip_upgrade(version, extras)`** — the parallel to `_run_pip_install()`:
   ```python
   def _run_pip_upgrade(self, version: str | None, extras: list[str]) -> str | None:
       old_version = _get_current_version()
       
       # ── Pre-flight: disk space check ──────────────────────────────
       # (Review fold: Ops lens, severity MAJOR)
       has_heavy = bool({"llm-torch", "vision", "yolo"} & set(extras))
       min_gb = 6.0 if has_heavy else 1.0
       usage = shutil.disk_usage(sys.prefix)
       free_gb = usage.free / (1 << 30)
       if free_gb < min_gb:
           self._send_json(507, {
               "error": f"Insufficient disk space: {free_gb:.1f} GB free, need ~{min_gb:.0f} GB.",
               "fix": "Free space or remove unused models with: maxim --delete-model <name>",
           })
           return None
       
       # ── Pre-flight: cache old version for network-independent rollback ─
       # (Review fold: Ops lens, cross-confirmed with Security lens)
       rollback_dir = Path(tempfile.mkdtemp(prefix="maxim-rollback-"))
       rollback_spec = f"pymaxim=={old_version}"
       if extras:
           rollback_spec += f"[{','.join(extras)}]"
       subprocess.run(
           [sys.executable, "-m", "pip", "download", rollback_spec,
            "-d", str(rollback_dir), "--index-url", "https://pypi.org/simple/"],
           capture_output=True, timeout=120,
       )
       
       # ── Upgrade ───────────────────────────────────────────────────
       spec = "pymaxim"
       if extras:
           spec += f"[{','.join(extras)}]"
       if version:
           spec += f"=={version}"
       
       # Pin index URL to prevent rogue pip.conf / PIP_INDEX_URL redirect
       # (Review fold: Security lens, severity MEDIUM)
       cmd = [sys.executable, "-m", "pip", "install", "--upgrade",
              "--index-url", "https://pypi.org/simple/", spec]
       
       try:
           result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
       except subprocess.TimeoutExpired:
           # (Review fold: Ops lens, severity BLOCKING — 120s too short for torch)
           self._send_json(500, {"error": "pip upgrade timed out (600s). Environment may be inconsistent."})
           shutil.rmtree(rollback_dir, ignore_errors=True)
           return None
       except Exception as e:
           self._send_json(500, {"error": f"pip upgrade failed: {e}"})
           shutil.rmtree(rollback_dir, ignore_errors=True)
           return None
       
       if result.returncode != 0:
           # Rollback from local cache — network-independent
           rollback = subprocess.run(
               [sys.executable, "-m", "pip", "install",
                "--no-index", "--find-links", str(rollback_dir), rollback_spec],
               capture_output=True, timeout=120,
           )
           rollback_status = "complete" if rollback.returncode == 0 else "INCOMPLETE"
           shutil.rmtree(rollback_dir, ignore_errors=True)
           self._send_json(500, {
               "error": "pip upgrade failed, rollback " + rollback_status,
               "stderr": result.stderr[-500:],
           })
           return None
       
       shutil.rmtree(rollback_dir, ignore_errors=True)
       return result.stdout[-500:]
   ```
   ~60 LOC.

6. **Branch `_handle_admin_update` on resolved mode.** After `_parse_admin_update_body`, resolve mode. If pip: call the pip path. If dev: call existing git path unchanged.

   The mode resolution adds ~15 LOC to the handler:
   ```python
   requested_mode = body.get("mode", "auto")
   resolved_mode = requested_mode if requested_mode != "auto" else _detect_install_mode()
   
   if resolved_mode == "dev" and not (Path(repo_root) / ".git").is_dir():
       self._send_json(409, {
           "error": "No git repository found on leader.",
           "fix": "Clone the repo to use --dev mode, or omit --dev to update via pip.",
       })
       return
   
   if resolved_mode == "pip":
       return self._handle_pip_update(body)
   # else: existing git path unchanged
   ```

7. **`_handle_pip_update(body)`** — orchestrates the pip path:
   - Get current version via `_get_current_version()`
   - Detect installed extras via `_detect_installed_extras()` (allowlist-filtered)
   - If dry-run: query latest version via `_get_latest_pypi_version()` (15s timeout; returns `None` on failure) + return preview with synthetic `pending_commits` for old-client compat
   - If not dry-run: call `_run_pip_upgrade(version, extras)`, then **compare version before vs after** to distinguish `updated` from `up_to_date` (Review fold: Ops lens, severity MEDIUM — `pip install --upgrade` returns 0 whether it upgraded or not)
   - ~40 LOC.

   ```python
   def _handle_pip_update(self, body: dict) -> None:
       old_version = _get_current_version()
       extras = _detect_installed_extras()
       version = body.get("version")
       dry_run = body.get("dry_run", True)  # server default is True (safe-by-default)
       
       if dry_run:
           latest = _get_latest_pypi_version()  # None on failure
           if latest and latest == old_version:
               self._send_json(200, {"status": "up_to_date", "install_mode": "pip",
                                     "current_version": old_version, "message": "Already at latest version."})
           else:
               self._send_json(200, {"status": "preview", "install_mode": "pip",
                                     "current_version": old_version,
                                     "latest_version": latest,  # may be None
                                     "extras_detected": extras,
                                     "pending_commits": [f"{old_version} → {latest or '?'}"],
                                     "message": f"{old_version} → {latest or 'unknown'} available."})
           return
       
       pip_output = self._run_pip_upgrade(version, extras)
       if pip_output is None:
           return  # error response already sent
       
       new_version = _get_current_version()
       if new_version == old_version:
           self._send_json(200, {"status": "up_to_date", "install_mode": "pip",
                                 "current_version": old_version, "message": "Already at latest version."})
       else:
           self._send_json(200, {"status": "updated", "install_mode": "pip",
                                 "from_version": old_version, "to_version": new_version,
                                 "extras_preserved": extras})
   ```

8. **`_get_latest_pypi_version()`** — `pip index versions pymaxim` with **15s timeout** (Review fold: Ops lens, severity LOW — cold pip cache can hang). Returns `None` on any failure (timeout, PyPI down, old pip). The dry-run path gracefully handles `None`. ~15 LOC.

**Validation:**
- `version` field validated against `^[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9.]+)?$` (PEP 440 subset, allows `rc1`, `post1`, `dev0`). Rejects injection (`; rm -rf /` etc.). All pip commands use `subprocess.run` with list args (no shell).
- `mode` validated against `{"auto", "pip", "dev"}`.
- Detected extras filtered through `_ALLOWED_EXTRAS` before any pip command.

**Pass gate:**
- `test_pip_update_detects_mode_pip`: no `.git` dir → resolved mode is `"pip"`
- `test_pip_update_detects_mode_dev`: `.git` dir present → resolved mode is `"dev"`
- `test_pip_update_dev_no_git_returns_409`: mode=`"dev"` + no `.git` → 409
- `test_pip_upgrade_preserves_extras`: mock subprocess → pip command includes `[semantic,llm-llama]`
- `test_pip_upgrade_extras_filtered_by_allowlist`: extra detected but not in `_ALLOWED_EXTRAS` → excluded from pip command
- `test_pip_upgrade_rollback_on_failure`: pip returncode=1 → rollback from local cache attempted (uses `--no-index --find-links`)
- `test_pip_upgrade_rollback_uses_local_cache`: rollback subprocess uses `--no-index --find-links /tmp/maxim-rollback-*`
- `test_pip_upgrade_version_pin`: version=`"0.3.1"` → pip command includes `==0.3.1`
- `test_pip_upgrade_version_injection_rejected`: version=`"0.3.1; rm -rf /"` → 400
- `test_pip_upgrade_pins_index_url`: pip command includes `--index-url https://pypi.org/simple/`
- `test_pip_upgrade_timeout_600s`: subprocess called with `timeout=600`
- `test_pip_upgrade_up_to_date_detection`: old_version == new_version after pip → status `"up_to_date"`, not `"updated"`
- `test_pip_upgrade_disk_check_rejects_low_space`: mock `shutil.disk_usage` with 0.5GB free → 507 response
- `test_pip_preview_includes_pending_commits`: dry-run pip response has `pending_commits` list (old-client compat)
- `test_pip_preview_pypi_down_graceful`: `_get_latest_pypi_version` returns None → preview response has `latest_version: null`
- `test_dev_mode_unchanged`: mode=`"dev"` with `.git` → existing git pull path runs (regression)

**Scope:** ~180 LOC in `leader_proxy.py`, ~150 LOC tests.

### Stage 2 — Client-side arg parsing + display

**What's built:**

1. **`_cmd_update()` in [cli.py:508](../../src/maxim/peer/cli.py#L508)** — add `--dev`, `--version`, and positional branch after `--dev`:
   ```python
   mode = "auto"
   version: str | None = None
   
   # In the parse loop:
   if a == "--dev":
       mode = "dev"
       # Next non-flag arg is the branch
       if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
           i += 1
           branch = argv[i]
   elif a == "--version":
       i += 1
       version = argv[i] if i < len(argv) else None
   ```
   ~15 LOC.

2. **Client-side validation** — `--version` + `--dev` is an error. Print and exit 1 before making the HTTP call. ~5 LOC.

3. **`update_on_target()` in [admin_core.py:35](../../src/maxim/peer/admin_core.py#L35)** — add `mode` and `version` params, pass through in JSON body. ~10 LOC.

4. **Response display for pip mode** — handle `install_mode: "pip"` responses:
   ```
   Updated! 0.3.0 → 0.3.1
     Extras preserved: semantic, llm-llama
   
   Restart maxim on the leader to load new code:
     maxim peer restart
   ```
   For dry-run:
   ```
   0.3.0 → 0.3.1 available.
     Detected extras: semantic, llm-llama
   
   Run without --dry-run to apply:
     maxim peer update
   ```
   ~25 LOC.

5. **Old-leader detection** (Review fold: Protocol lens, severity MEDIUM). When the response lacks `install_mode` and the client sent `mode: "pip"` or `--version`, the leader is too old to handle pip mode. Print a hint:
   ```
   Leader does not support pip update mode (requires 0.3.1+).
   Upgrade the leader first: pip install --upgrade pymaxim
   ```
   Check: `if data.get("install_mode") is None and mode != "auto"`. ~5 LOC.

6. **`--branch` warning in pip mode** (Review fold: Protocol lens, severity LOW). If the user passes `--branch` without `--dev`, and the server resolves to pip mode (`install_mode: "pip"` in response), print a one-line warning: `"Note: --branch is ignored in pip mode. Use --dev <branch> for git updates."` ~3 LOC.

7. **`_run_node_update()` in [mesh_cli.py:582](../../src/maxim/peer/mesh_cli.py#L582)** — parse `--dev`, `--version`, forward through `update_on_target()`. ~10 LOC.

8. **Backward compatibility.** Old peers that don't send `mode` get `"auto"`. Old leaders that don't understand `mode` ignore it and run the git path (existing behavior). The field is additive.

**Pass gate:**
- `test_cmd_update_dev_flag`: `--dev feat/foo` → mode=`"dev"`, branch=`"feat/foo"`
- `test_cmd_update_dev_default_branch`: `--dev` alone → mode=`"dev"`, branch=`"main"`
- `test_cmd_update_version_flag`: `--version 0.3.1` → mode=`"auto"`, version=`"0.3.1"`
- `test_cmd_update_version_dev_conflict`: `--dev --version 0.3.1` → exit 1, no HTTP call
- `test_cmd_update_no_flags`: bare `maxim peer update` → mode=`"auto"`, version=`None` (backward compat)
- `test_pip_response_display`: mock pip-mode response → correct terminal output
- `test_old_leader_detection`: response missing `install_mode` + client sent `mode: "pip"` → hint printed
- `test_branch_warning_in_pip_mode`: `--branch main` without `--dev` + pip response → warning printed

**Scope:** ~75 LOC across `cli.py` + `admin_core.py` + `mesh_cli.py`, ~70 LOC tests.

### Stage 3 — Docs + CLI reference

**What's built:**

1. Update [docs/user/cli-reference.md](../../user/cli-reference.md) — add pip mode examples to `maxim peer update` section.
2. Update [docs/troubleshooting/remote_update.md](../../troubleshooting/remote_update.md) — add pip deployment troubleshooting.
3. Update this plan's status.

**Scope:** ~40 LOC docs.

## What this plan does NOT include

- **Auto-detection of available PyPI version on the client side.** The client doesn't check `pip index` locally — it sends the request and lets the server handle version discovery. The server is the authority on what's installed.
- **Mixed-mode mesh.** Node A on pip, node B on git. Both work — each node detects its own mode. The mesh-aware `--node X update` sends `mode: "auto"` and the target resolves it locally. No coordination needed.
- **`pip install git+https://github.com/...@branch`.** This is a third mode that mixes pip and git semantics. The rollback story is unclear (what's the "old version" of a git URL install?). If a user wants git-based updates, they should clone the repo and use `--dev`.
- **Hot reload after pip upgrade.** Same as today — user must run `maxim peer restart` after update. The `os.execv` restart will pick up the new pip-installed code.
- **Pinning to pre-release versions.** `--version 0.3.1rc1` — the version regex accepts PEP 440 suffixes, so this works, but we don't advertise or test it.
- **PEP 668 (`externally-managed-environment`) detection.** On Debian 12+ / Ubuntu 23.04+, system Python rejects pip installs. The pip error message is clear; adding pre-flight detection and a 409 with venv guidance is a follow-up if user reports come in.
- **Async poll pattern for long upgrades.** The `/v1/admin/install` endpoint already uses background threads + polling via `/v1/debug/install-status`. The pip upgrade path uses a synchronous 600s timeout instead — simpler, and 600s covers even cold torch downloads. Migrate to async if the synchronous path proves insufficient.
- **Request rejection during upgrade window.** The leader serves inference while pip replaces packages on disk. A 503 gate during the upgrade window would prevent stale-module edge cases, but the risk is low (Python caches loaded modules) and the existing git path has the same exposure.

## Backward compatibility

**Old peer (0.3.0) → new leader (0.3.1):** peer doesn't send `mode`, leader gets `"auto"`, resolves to detected mode. Works. Old client reads `status` field — pip-mode `"updated"` response displays as `"Updated! 0 commit(s) applied:"` (empty list, functional but degraded). Old client dry-run reads `pending_commits` — pip preview includes synthetic entry `["0.3.0 → 0.3.1"]` so the user sees version info.

**New peer (0.3.1) → old leader (0.3.0):** peer sends `mode: "pip"`. Old leader's `_parse_admin_update_body()` ignores unknown fields and runs the git path. If the leader is a git checkout, the update proceeds as before (harmless mode mismatch). If the leader is pip-only, git commands crash with 500. New client detects missing `install_mode` in response and prints: `"Leader does not support pip update mode (requires 0.3.1+)."` This was already broken pre-plan — the new client just gives a better error.

**The clean upgrade path for pip-only leaders:** update the leader first (`pip install --upgrade pymaxim`), then peers can use the new `--version` / `--dev` flags.

## Risks

1. **`pip index versions` availability.** Added in pip 21.2 (2021-07). Older pips don't have it. `_get_latest_pypi_version()` has a 15s timeout and returns `None` on any failure. The dry-run path gracefully handles `None` (`latest_version: null` in response). The non-dry-run path doesn't need it.
2. **Extras detection false positives.** If `sentence_transformers` is installed globally but not via `pymaxim[semantic]`, we'll include `semantic` in the upgrade spec. This is harmless — pip will see it's already satisfied. The allowlist filter prevents any extra outside `_ALLOWED_EXTRAS` from reaching a pip command.
3. **Extras detection false negatives.** If an extra was installed but its import module was renamed upstream, we'll miss it. Low risk — the `_EXTRA_IMPORT_MAP` is maintained alongside the extras in `pyproject.toml`.
4. **Version rollback with extras.** Rollback installs from pre-cached local wheels (`--no-index --find-links`), so it's network-independent. If the pre-cache step failed silently (e.g., old version yanked from PyPI before the upgrade started), rollback falls through to network download as a last resort.
5. **Concurrent updates.** Same risk as today — two peers updating the same leader simultaneously could race. The existing `MAXIM_ALLOW_REMOTE_UPDATE` gate is the protection (one operator, one update at a time).
6. **Disk space.** The pre-flight check uses `shutil.disk_usage(sys.prefix)` with conservative thresholds (6GB for torch-bearing extras, 1GB otherwise). Returns 507 with actionable fix hint. False positives possible if `sys.prefix` is on a different partition from pip's temp dir.
7. **PEP 668 `externally-managed-environment`.** On Debian 12+ / Ubuntu 23.04+, pip refuses to install into system Python without `--break-system-packages`. The upgrade will fail, trigger rollback (which also fails for the same reason), and return a 500. **Deferred:** detecting this pre-flight and returning a 409 with venv guidance is a follow-up. Document as known limitation for now — the error message from pip is clear enough.
8. **Live process module cache during upgrade.** Python caches imported modules in `sys.modules`. Already-loaded code is safe. Lazy imports triggered during an in-flight request mid-upgrade could load a mix of old and new modules. This is the same risk as the existing git path and is mitigated by the existing "restart after update" requirement.

## Files touched

| File | Stage | LOC estimate |
|---|---|---|
| `src/maxim/runtime/leader_proxy.py` | 1 | ~180 |
| `src/maxim/peer/cli.py` | 2 | ~25 |
| `src/maxim/peer/admin_core.py` | 2 | ~15 |
| `src/maxim/peer/mesh_cli.py` | 2 | ~15 |
| `tests/unit/test_peer_update_pip.py` (new) | 1, 2 | ~220 |
| `docs/user/cli-reference.md` | 3 | ~20 |
| `docs/troubleshooting/remote_update.md` | 3 | ~20 |

## Stage ordering

Stage 1 (server) and Stage 2 (client) are independent — the server handles missing `mode` field gracefully, so either can ship first. Stage 3 (docs) depends on both.

**Recommended order:** 1 → 2 → 3. Server first so the new endpoint behavior is testable via curl before the CLI is updated.

## Review history

- **2026-04-18 (draft):** Initial 3-stage plan.
- **2026-04-18 (3-lens parallel review):** Security + Supply Chain, Operational Failure Modes, Backward Compat + Wire Protocol. 9 findings total, 7 folded into plan:
  - **BLOCKING (Ops):** 120s timeout too short for torch downloads. Raised to 600s.
  - **BLOCKING (Protocol):** `dry_run` server default is `True`, plan spec said `false`. Fixed.
  - **MAJOR (Security):** Detected extras bypass `_ALLOWED_EXTRAS`. Added allowlist filter.
  - **MAJOR (Protocol):** Old-client dry-run shows "0 pending commit(s)". Added synthetic `pending_commits` to pip preview.
  - **MAJOR (Ops):** No disk space pre-check. Added `shutil.disk_usage` pre-flight.
  - **MEDIUM (Security):** No `--index-url` pinning. Pinned to `https://pypi.org/simple/`.
  - **MEDIUM (Ops):** "Already current" indistinguishable from "upgraded". Added before/after version comparison.
  - **MEDIUM (Protocol):** New peer + old leader gets cryptic error. Added client-side `install_mode` detection hint.
  - **MEDIUM (Ops + cross-confirmed):** Rollback fails under same conditions as upgrade. Added local wheel pre-caching.
  - Deferred: PEP 668 detection (document as known limitation), async poll pattern for long upgrades (600s timeout sufficient for v1).
