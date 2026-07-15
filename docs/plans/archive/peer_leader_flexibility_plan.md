# Peer / Leader LLM Flexibility

**Status:** Active, parallel to spine (does not gate 1.0). Parallel-safe with substrate_plan (zero file overlap — see "Parallel execution" section below).
**Target version:** 0.2.2 (ships alongside `cleanup_wave.md`).
**Structure:** Foundation wave (F0) followed by phases P1–P9. Each phase is independently verifiable and rollback-safe.
**Runtime blast radius:** Every `build_primary_router` call path on every peer AND every leader. This is critical-path LLM routing code. Treat it accordingly.

## Goal

Make peer and leader LLM routing honest and responsive to user intent. Specifically:

1. **Respect `--llm` precedence.** Explicit local profile on the command line runs locally, period — regardless of peer config state.
2. **Size tier detection correctly on Apple Silicon.** A 24GB Mac should pick and run a 14B quantized model. Currently it caps at 7B.
3. **Fail gracefully when the leader is unreachable.** Cloudflare 502s should not wedge a peer into retry storms.
4. **Auto-download on demand, safely.** Preflight disk space, atomic downloads, integrity checks, concurrent-invocation protection.
5. **Surface routing decisions.** A lane-decision log and actionable `maxim doctor` checks so users can see what's happening without reading source.

None of this changes the LLM router's interface (substrate_plan's non-goal "no new LLM router features" is preserved — all changes live outside `models/language/router.py`).

## Leader migration — CRITICAL READ-FIRST

All `build_primary_router` fixes execute on the **leader** too, not just peers. The leader runs the same startup path and applies the same tier-detection logic. Three scenarios change behavior on existing leaders after this wave lands:

### Scenario 1: expanded tier table on existing leader (P4a)

**Symptom:** Leader currently runs mistral-7b by default. After P4a, `detect_tiers` picks qwen2.5-14b for any GPU ≥16GB VRAM. First `maxim peer restart` after upgrade causes the leader to attempt loading a model that probably isn't downloaded yet, auto-spawn fails, fallback path runs.

**Risk:** Silent regression. User runs `maxim peer update && maxim peer restart`, leader comes back with a *different* model than before. Any tests/sims calibrated against the previous model now see different behavior.

**Mitigation:** **The first time `build_primary_router` runs on a host where `~/.maxim/util/active_llm_model.txt` is absent AND no `MAXIM_LLM_PROFILE` is set, write the *current* tier-detected profile to the persisted-model file BEFORE applying the new tier table.** This pins the existing leader to whatever it was running pre-upgrade. Users who want the upgrade run `maxim peer llm qwen2.5-14b` explicitly.

This mitigation is implemented in **F0.3**, *before* P4a ships, so the migration is complete before behavior changes.

### Scenario 2: leader has qwen2.5-14b selected but GGUF not downloaded

**Symptom:** P4a's expanded tier table selects qwen2.5-14b. The leader has never downloaded it. Auto-spawn fails at the `profile_has_local_file` check. Tier detection's `profile_available=_profile_has_local_file` callback should fall through to the next tier, but if F0.3's persisted-pin is wrong or absent, the leader ends up with no working large tier.

**Mitigation:** The tier-detection walk already takes `profile_available` as a callback. It correctly falls through to the next-largest profile. Verify in P4a's tests that the fall-through works with a fresh `~/.maxim/models/` directory.

### Scenario 3: peer config on the leader itself

Some users have configured the leader as a peer of another machine (multi-leader mesh topology, experimental). In that case, `peer.yml` exists on the leader and points at another machine. P1's `--llm` precedence change plus the new probe (P6) change what the leader does on startup. 

**Mitigation:** Document explicitly in P1 and P6 that these changes affect both peers and leaders. Mesh topology is out of scope for this wave — we pin "leader runs its own hardware, peer.yml is absent on leader" as the supported configuration.

## Parallel execution with substrate_plan

**Zero file overlap with substrate_plan:**

| This plan touches | substrate_plan touches |
|---|---|
| `runtime/lane_backends.py` | `agents/bus.py` (Percept) |
| `runtime/capabilities.py` | `similarity/ec.py` |
| `runtime/lane_models.py` | `memory/atl.py`, `memory/hippocampus.py` |
| `runtime/llm_server.py` | `decisions/nac.py` |
| `runtime/local_server_spawner.py` | `agents/prompt_builder.py` (B1) |
| `models/language/openai_backend.py` | `runtime/context_pool.py` |
| `models/language/config.py` (metadata only) | new `prompts/assembler.py`, `prompts/acting_coach.py` |
| `models/download.py` | `_data/prompts/planning/replanning.txt` |
| `cli.py`, `cli_parser.py`, `peer/cli.py` | new `similarity/linguistic_encoder.py` |
| `doctor/checks.py` | new `tests/fixtures/substrate/` |
| new `utils/storage.py` | |

Merge-conflict risk is **nil**. The closest contact is `models/language/config.py` (we add metadata fields; substrate doesn't touch the file at all).

**Sequencing recommendation:** land peer/leader commits A+B first (~days) because they unblock Mac-based substrate iteration. Substrate F0 can land in parallel with any of this wave. Full substrate P1+ work is weeks per phase and isn't blocked by anything here.

## Current state — deep dive (verified against code)

Paths walked through `build_primary_router` for the failing case (Mac + peer config + `--llm mistral-7b-instruct-v0.2` + dead leader tunnel):

1. **Persisted-model restore** — [lane_backends.py:597-604](../../src/maxim/runtime/lane_backends.py#L597-L604). Reads `~/.maxim/util/active_llm_model.txt` into `MAXIM_LLM_PROFILE` if unset. Written by `maxim peer llm <model>` and by LLMRouter on successful init.

2. **Peer config auto-load** — [lane_backends.py:606-618](../../src/maxim/runtime/lane_backends.py#L606-L618). Loads `~/.config/maxim/peer.yml`, calls `apply_peer_config_to_env` which uses `os.environ.setdefault` to populate `MAXIM_LANE_LARGE_REMOTE_URL` / `_REMOTE_API_KEY` / `_REMOTE_MODEL`.

3. **The "reconcile" trap** — [lane_backends.py:627-632](../../src/maxim/runtime/lane_backends.py#L627-L632):
    ```python
    if _has_peer_config:
        _remote_url = os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL", "").strip()
        _llm_profile = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
        if _remote_url and _llm_profile:
            os.environ.setdefault("MAXIM_LANE_LARGE_REMOTE_MODEL", _llm_profile)
            os.environ.pop("MAXIM_LLM_PROFILE", None)
    ```
    This flips `--llm <local>` into "ask the leader for <local>", which fails when the leader has a different model loaded. P1 removes this block and replaces it with the correct precedence.

4. **Compute detection** — [capabilities.py:72-114](../../src/maxim/runtime/capabilities.py#L72-L114). On MPS:
    - `torch.cuda.is_available()` → False → `vram_gb=0.0`
    - `torch.backends.mps.is_available()` → True → `has_gpu=True, gpu_type="mps"`
    - `psutil.virtual_memory().total` → `ram_gb=24.0`

    Unified memory means actual Metal-accessible VRAM is ~18GB on a 24GB Mac. Reported as 0.0. P2 fixes this.

5. **Tier detection** — [lane_models.py:99-179](../../src/maxim/runtime/lane_models.py#L99-L179). Hard-excludes `gpu_type="mps"` at [line 139](../../src/maxim/runtime/lane_models.py#L139) from the large-tier branch. Routes MPS to the medium tier only. P3 drops this exclusion.

6. **Medium tier walk** — [lane_models.py:58-77](../../src/maxim/runtime/lane_models.py#L58-L77):
    ```python
    _MEDIUM_RAM_TIERS = (
        (16.0, "mistral-7b-instruct-v0.2"),
        (8.0, "phi-3-mini-4k-instruct"),
    )
    ```
    Tops out at mistral-7b for anything ≥16GB. No row for 13B/14B. P4a adds qwen2.5-14b to the **large** tier table and lets MPS enter via P3.

7. **Placeholder lane injection** — [lane_backends.py:655-666](../../src/maxim/runtime/lane_backends.py#L655-L666). If `detect_tiers` didn't create a large tier but peer config set a remote URL, a placeholder large lane is injected pointing at the leader.

8. **Remote URL validation** — [lane_backends.py:735-760](../../src/maxim/runtime/lane_backends.py#L735-L760). `_validate_remote_urls` probes loopback / private-IP URLs only. Public URLs (Cloudflare tunnels) are trusted. P6 changes this to probe all remote URLs with a tiered timeout and result caching.

9. **Auto-spawn** — [lane_backends.py:763-860](../../src/maxim/runtime/lane_backends.py#L763-L860). Hard no-op conditions: `has_gpu=False`, lane has `remote_url`, no model profile, `llama_cpp.server` not importable, GGUF file missing, or (per the fix in commit `1a5eb85`) the lane's profile is a cloud profile. P5 adds disk-preflight + auto-download before auto-spawn runs.

10. **Auto-spawn n_ctx** — [lane_backends.py:966](../../src/maxim/runtime/lane_backends.py#L966). Uses `_safe_int_env("MAXIM_AUTO_SPAWN_N_CTX", 8192)`. **This is not derived from profile metadata.** The profile's declared `n_ctx` (e.g. 32768 for qwen2.5-14b) is independent of what the spawner actually loads. P4c derives this from profile metadata + VRAM budget.

11. **`LaneConfig` dataclass** — [worker_pool.py:73-90](../../src/maxim/runtime/worker_pool.py#L73-L90). Fields: `name`, `max_workers`, `queue_size`, `requires_gpu`, `model_profile`, `device`, `n_gpu_layers`, `remote_url`, `remote_api_key`, `remote_model`. **No `n_ctx` field.** P4c adds it.

### Pre-existing bugs discovered during investigation

These are real bugs in the existing code that this wave must fix, not just design gaps in the plan. Each gets called out in the relevant phase:

1. **`download_file` leaks partial files on `URLError`.** [download.py:253-255](../../src/maxim/models/download.py#L253-L255): generic `Exception` path cleans up the partial file; `URLError` path only prints and returns False, leaving the file on disk. Next run's `profile_has_local_file` check passes on a corrupted file → load crash → retry → loop. Fixed in **F0.1**.

2. **`download_file` doesn't catch `KeyboardInterrupt`.** Ctrl+C mid-download leaves a partial file with the final name. Same failure mode as above. Fixed in **F0.1**.

3. **`LLM_MODELS` has no integrity field.** [download.py:38-95](../../src/maxim/models/download.py#L38-L95) has `size_gb`, `url`, `filename`, `quantization`, `description`, but **no `sha256`** and **no `expected_bytes`**. There's no way to detect a silently-truncated or corrupted file short of attempting to load it. Fixed in **F0.2**.

4. **Profile `n_ctx` is advertised but not applied.** Profile declares `n_ctx=32768` for qwen2.5-14b, auto-spawn uses env-default 8192, router's provider config uses yet another path. Three values can diverge and the router's "does this prompt fit" check uses the wrong one. Partially fixed in **P4c** (auto-spawn derives from profile + capability).

5. **No file locking pattern in the codebase.** `grep -rn 'fcntl\|flock' src/maxim` returns nothing substantive. Concurrent auto-downloads would race on the same target file. **F0.4** introduces `utils/filelock.py` with a minimal flock wrapper + Windows fallback.

6. **`LaneConfig` has no `n_ctx` field.** P4c adds it.

7. **`LocalServerSpawner._build_cmd` doesn't support KV cache quantization.** [local_server_spawner.py:337-357](../../src/maxim/runtime/local_server_spawner.py#L337-L357). `llama.cpp`'s `--ctk q4_0 --ctv q4_0` flags would quadruple effective context budget on tight-VRAM cards but are not exposed. Added in **P4c** as an optional degree of freedom.

---

## F0 — Foundation (prerequisite, blocks P5+ and P4)

Five items, all small, all prerequisites for the numbered phases. Ship as a single commit before any phase-bearing work.

### F0.1 — Download atomicity

**Bug:** [download.py:229-261](../../src/maxim/models/download.py#L229-L261). Partial files leak on `URLError` and `KeyboardInterrupt`. Rename-on-success semantics missing.

**Design:** Download to `{dest}.partial`. Only `os.replace()` to `{dest}` after verification. Catch all exception types including `BaseException` (covers `KeyboardInterrupt`). Register an atexit cleanup for any `.partial` files the current process created.

**Implementation sketch:**

```python
def download_file(url: str, dest_path: Path, *, expected_bytes: int | None = None,
                  expected_sha256: str | None = None, desc: str = "") -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return True
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
    # Clean up any stale partial from a prior crashed run
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        urlretrieve(url, tmp_path, reporthook=_progress_hook)
    except BaseException as e:  # includes KeyboardInterrupt
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        if isinstance(e, KeyboardInterrupt):
            raise
        print(f"  Download failed: {e}")
        return False
    # Verify before rename
    if expected_bytes is not None and tmp_path.stat().st_size != expected_bytes:
        tmp_path.unlink()
        print(f"  Download size mismatch (got {tmp_path.stat().st_size}, expected {expected_bytes})")
        return False
    if expected_sha256 is not None:
        actual = _sha256_of_file(tmp_path)
        if actual != expected_sha256:
            tmp_path.unlink()
            print(f"  Download SHA256 mismatch")
            return False
    os.replace(tmp_path, dest_path)
    return True
```

**Hazards:**
- `urlretrieve` isn't the most robust HTTP client — doesn't resume interrupted downloads, no retries. Acceptable for 0.2.2; a future item can migrate to `huggingface_hub.hf_hub_download` for resumable downloads.
- `profile_has_local_file` needs to ignore `.partial` files. [llm_server.py:141-156](../../src/maxim/runtime/llm_server.py#L141-L156) currently just checks `Path(model_path).is_file()`. Add a check that the suffix is not `.partial`.

**Tests:**
- `test_download_atomicity_urlerror`: mock `urlretrieve` to raise `URLError`, assert `.gguf` does not exist and `.partial` does not exist.
- `test_download_atomicity_keyboardinterrupt`: mock to raise `KeyboardInterrupt`, assert same, assert the exception propagates.
- `test_download_atomicity_size_mismatch`: mock to return wrong byte count, assert verification fails and cleanup happens.
- `test_profile_has_local_file_ignores_partial`: create `X.gguf.partial`, assert `profile_has_local_file("X")` returns False.

**Exit:** `urlretrieve` is never called in a path where failure leaves orphan files.

**Rollback:** Simple revert — no schema changes, no state migration.

### F0.2 — `LLM_MODELS` integrity metadata

**Bug:** No size or hash verification for downloaded models. A truncated or corrupted file silently passes `profile_has_local_file`.

**Design:** Add `expected_bytes` and `sha256` fields to each `LLM_MODELS` entry. `expected_bytes` is authoritative (it's checked on every download); `sha256` is optional at first (can be populated as we gather them from HF mirrors).

**Implementation sketch:**

```python
LLM_MODELS: dict[str, dict[str, Any]] = {
    "qwen2.5-14b-instruct": {
        "description": "Alibaba Qwen2.5 14B Instruct...",
        "size_gb": 8.5,
        "expected_bytes": 9123456789,  # Precise — checked on download
        "sha256": None,  # Optional — populate when known
        "quantization": "Q4_K_M",
        "url": "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "filename": "Qwen2.5-14B-Instruct.Q4_K_M.gguf",
    },
    ...
}
```

**How to get `expected_bytes`:** HF repos expose file size via the `/api/models/{repo}` metadata endpoint. One-time scrape: `huggingface_hub.hf_api.HfApi().list_repo_files(..., detail=True)`. Land the numbers we know; gate verification on `expected_bytes is not None` so an unmigrated profile still downloads (without size check).

**Hazards:**
- Wrong expected_bytes (typo, upstream file re-quantized) causes legitimate downloads to be rejected. Maintenance burden. Mitigation: add `MAXIM_SKIP_DOWNLOAD_VERIFY=1` escape hatch.

**Tests:**
- `test_download_verifies_size_when_known`: mock urlretrieve to create a file of the wrong size, assert verification fails.
- `test_download_skips_verify_when_unknown`: profile has `expected_bytes=None`, verification path skipped.

**Exit:** All `llama_cpp`-backend profiles in `LLM_MODELS` have `expected_bytes` set. `sha256` is populated where available.

**Rollback:** Remove the verification call from F0.1's `download_file`; the fields stay but are ignored.

### F0.3 — Leader profile pinning (migration safety)

**Bug:** P4a changes the default tier-detected profile on existing leaders. Without this fix, `maxim peer restart` after upgrade silently switches the leader's model.

**Design:** In `build_primary_router`, *after* persisted-model restore at [lane_backends.py:597-604](../../src/maxim/runtime/lane_backends.py#L597-L604), check if the persisted-model file exists. If absent AND `MAXIM_LLM_PROFILE` is unset AND this machine has ever run Maxim before (heuristic: `~/.maxim/util/` exists and contains at least one file), **write the currently-effective tier-detected profile to the persisted-model file before applying the new tier table**.

```python
# After the existing persisted-model restore block:
_pinning_file = Path(data_home()) / "util" / "active_llm_model.txt"
_pinning_marker = Path(data_home()) / "util" / "tier_pinned_at_upgrade"
if not _pinning_file.exists() and not _pinning_marker.exists():
    # Machine has never run this version. Detect what the PRE-upgrade
    # tier table would have picked, pin it, then let the new tier table
    # take over on the next startup.
    try:
        pre_upgrade_profile = _detect_pre_upgrade_profile(capabilities)
        if pre_upgrade_profile:
            _write_persisted_model(pre_upgrade_profile)
            _pinning_marker.touch()
            if logger:
                logger.info(
                    "Pinned leader to pre-upgrade profile %r for safe migration. "
                    "Run `maxim peer llm <model>` to change.", pre_upgrade_profile,
                )
    except Exception as e:
        logger.debug("Pre-upgrade pin failed (non-fatal): %s", e)
```

`_detect_pre_upgrade_profile` uses the **old** `_INFER_VRAM_TIERS` table (hardcoded inline for this migration function). Fires exactly once per machine — the marker file prevents repeat execution after the user has explicitly swapped models.

**Hazards:**
- If `data_home()` returns a directory that never existed before (new install), the "has ever run" heuristic is False and pinning is skipped. That's correct — new installs don't need migration.
- If the user deletes `~/.maxim/util/tier_pinned_at_upgrade` to force a re-pin, they get a second pin on next startup. Not a bug; they asked for it.
- If P4a ships before F0.3, the pinning is too late — the new tier table already ran. **F0.3 must ship BEFORE P4a lands on any machine.** Ordering-critical.

**Tests:**
- `test_leader_pinning_fresh_install`: no `util/` directory, pinning does not fire.
- `test_leader_pinning_existing_install_no_persisted_model`: `util/` exists with other files, no `active_llm_model.txt`, pinning writes the old-tier-table result.
- `test_leader_pinning_idempotent`: marker file present, pinning is skipped on second run.

**Exit:** An existing leader with mistral-7b previously selected by tier detection stays on mistral-7b after upgrade; only explicit `maxim peer llm qwen2.5-14b` moves it.

**Rollback:** Delete the marker file on all affected machines. No data corruption possible (pinning writes a single profile name to one file).

### F0.4 — `utils/filelock.py` minimal lock primitive

**Bug:** No file locking pattern in the codebase. P5's concurrent-download protection requires one.

**Design:** A minimal context-manager file lock with POSIX `fcntl.flock` and a Windows `msvcrt.locking` fallback. Non-blocking mode only (attempt lock; raise `LockContended` on failure). ~60 LOC total including Windows path.

```python
class LockContended(RuntimeError):
    pass

@contextmanager
def file_lock(path: Path, *, timeout_s: float = 0.0) -> Iterator[None]:
    """Acquire an exclusive lock on `path`, creating it if needed.

    timeout_s=0 → non-blocking, raises LockContended immediately if held.
    timeout_s>0 → poll for the lock up to timeout, then raise.
    Automatically released on context exit via fcntl.LOCK_UN / msvcrt.LK_UNLCK.
    """
    ...
```

**Hazards:**
- NFS file locks are notoriously flaky. Document that `MAXIM_DATA_HOME` on NFS is not officially supported. This is the same surface substrate_plan already uses for checkpoints.
- Stale locks from crashed processes: `fcntl.flock` releases automatically on process exit, so stale locks are only a concern if the OS crashes. Not worth hand-rolling expiry.

**Tests:**
- `test_filelock_contention`: hold lock in one thread, assert second acquire raises `LockContended`.
- `test_filelock_release_on_exit`: use `with` block, acquire in another thread after exit, succeeds.
- `test_filelock_timeout`: hold lock, second call with `timeout_s=0.2` waits ~0.2s then raises.

**Exit:** P5's download code can serialize concurrent invocations via `with file_lock(...)`.

**Rollback:** Trivial — unused outside P5.

### F0.5 — `utils/storage.py` footprint reporter

**Bug:** No tracking of Maxim's disk usage. P5 needs this for the disk preflight, and `maxim doctor` benefits from surfacing it.

**Design:** ~100 LOC helper module. See P5 for the full `StorageReport` and `can_download` interface. Module-level cache with 60s TTL + cache-bypass escape hatch.

```python
# src/maxim/utils/storage.py
@dataclass(frozen=True)
class StorageReport:
    data_home: Path
    fs_free_gb: float
    fs_total_gb: float
    subdir_sizes_gb: dict[str, float]  # approximate; one scandir level deep
    total_maxim_gb: float
    walked_at: float  # monotonic time, used by cache

_REPORT_CACHE: StorageReport | None = None
_CACHE_TTL_S = 60.0
_CACHE_LOCK = threading.Lock()

def report_storage(*, force: bool = False) -> StorageReport:
    """Walk ~/.maxim/ subdirs and return a size report. 60s cached."""

def can_download(size_gb: float, *, headroom_gb: float = 2.0,
                 soft_budget_gb: float | None = None) -> tuple[bool, str]:
    """Return (ok, reason) — checks filesystem free + optional soft budget."""

def format_report(report: StorageReport) -> str:
    """Human-readable summary for doctor / preflight messages."""
```

**Hazards:**
- **Scanning is bounded to one level deep.** The plan's earlier draft said "walks `~/.maxim/` subdirs." On HDD with years of accumulated `sessions/` and `provenance/`, recursive walks can be seconds. Fix: `os.scandir(subdir)` for each immediate subdirectory and sum only top-level file sizes + count of nested entries. Result is an approximation (subdirectories-of-subdirectories aren't summed), which is good enough for "where is space going" guidance.
- **Cache invalidation:** `MAXIM_DATA_HOME` can move at runtime (unusual but legal). The cache is invalidated when `data_home()` returns a different path than the cached report.

**Tests:**
- `test_storage_report_walks_subdirs`: create fake `~/.maxim/` structure, assert sizes sum correctly.
- `test_storage_cache_ttl`: call twice within 60s, assert single walk. Call after 61s, assert second walk.
- `test_storage_cache_force_refresh`: `force=True` bypasses the cache.
- `test_can_download_rejects_insufficient_space`: fake small `fs_free_gb`, assert `(False, ...)`.
- `test_can_download_respects_soft_budget`: cumulative footprint + requested exceeds `soft_budget_gb`, assert rejection.

**Exit:** `from maxim.utils.storage import report_storage, can_download, format_report` imports cleanly. `report_storage()` on a real `~/.maxim/` completes in <500ms (typical) or <100ms (cached).

**Rollback:** Trivial — unused outside P5 and P8.

---

## P1 — `--llm <local>` precedence over peer config

**Bug:** [lane_backends.py:627-632](../../src/maxim/runtime/lane_backends.py#L627-L632) rewrites a user's `--llm mistral-7b` into "ask the leader for mistral-7b" when peer config is active. The leader ignores the model name and serves whatever it has loaded.

**Files touched:** `runtime/lane_backends.py` (replace the reconcile block), tests.

**Design:**

Remove the reconcile block at lines 627-632 entirely. Add a new `_apply_local_llm_override(lane_configs, logger)` helper, called from `build_primary_router` right after `_apply_cloud_cli_overrides` at line 684. Logic:

1. Read `MAXIM_LLM_PROFILE` env var (set by `--llm` via `cli_utils.py:63`).
2. If unset, no-op.
3. Look up the profile in `_BUILTIN_PROFILES`. If it's a cloud profile (`profile.get("cloud")`), no-op — `_apply_cloud_cli_overrides` already handles this direction.
4. Determine which lane(s) the profile would be assigned to. For this wave: only the large tier. (Medium/small local profiles are out of scope — they're handled by env-level `MAXIM_LANE_{NAME}_REMOTE_URL` directly.)
5. If the target lane has a `remote_url`, clear it. Log the override:
    ```
    Lane 'large' local override: --llm=mistral-7b-instruct-v0.2 clears remote_url=https://maxim.dennyschaedig.com/v1
    ```

**Precedence table (final):**

| Config | Outcome |
|---|---|
| No `--llm`, peer config | Peer wins (unchanged) |
| `--llm <cloud>`, peer config | Cloud wins (commit `1a5eb85` via `_apply_cloud_cli_overrides`) |
| `--llm <local>`, peer config | Local wins (new P1 behavior) |
| `--llm <cloud>` AND `--cloud-lane large <other_cloud>` | `--cloud-lane` wins (more specific, targets a specific lane) |
| `MAXIM_LANE_LARGE_REMOTE_URL` env AND `--llm <local>` | `--llm` wins (CLI beats env for this wave; document in open questions) |

**Hazards:**
- **Existing users relying on the reconcile block** (they explicitly pass `--llm qwen2.5-14b` to make the leader swap to it) will break. Count of such users is unknown. Mitigation: add a deprecation log line in F0 that warns when the reconcile path would have fired:
    ```
    DEPRECATED: --llm <local> with peer config now runs locally (was: rewritten as
    remote model name). If you wanted the leader to use this model, run
    `maxim peer llm <model>` explicitly.
    ```
- **Mode-specific lane override** is defined as "large tier only" for this wave. P1 does NOT auto-clear medium/small remote URLs even if they happen to exist. Documented explicitly in the design above.

**Implementation sketch:**

```python
def _apply_local_llm_override(
    lane_configs: dict[str, LaneConfig],
    logger: Any | None,
) -> dict[str, LaneConfig]:
    """Clear the large lane's remote_url when --llm names a local profile.

    CLI intent ("I typed --llm mistral-7b") beats peer config intent
    ("I have a peer configured") for local profiles. Cloud profiles are
    handled separately by _apply_cloud_cli_overrides.
    """
    profile_name = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
    if not profile_name:
        return lane_configs
    try:
        from maxim.models.language.config import _BUILTIN_PROFILES
    except Exception:
        return lane_configs
    profile_data = _BUILTIN_PROFILES.get(profile_name, {})
    if profile_data.get("cloud"):
        return lane_configs  # cloud path handled elsewhere

    out = dict(lane_configs)
    for lane_name in ("large",):  # scoped to large tier only for this wave
        cfg = out.get(lane_name)
        if cfg is None or not cfg.remote_url:
            continue
        if logger is not None:
            logger.info(
                "Lane '%s' local override: --llm=%s clears remote_url=%s",
                lane_name, profile_name, cfg.remote_url,
            )
        out[lane_name] = dataclasses.replace(
            cfg, remote_url=None, remote_model=None, remote_api_key=None,
            model_profile=profile_name,
        )
    return out
```

**Tests:**
- `test_local_llm_override_clears_remote`: peer config → `remote_url` set, `MAXIM_LLM_PROFILE=mistral-7b-instruct-v0.2`, assert cleared.
- `test_local_llm_override_skips_cloud`: `MAXIM_LLM_PROFILE=claude-sonnet-4-6`, assert no-op (cloud path).
- `test_local_llm_override_respects_medium_lane`: peer config on large lane only, medium lane has no `remote_url`, assert unchanged.
- `test_precedence_table`: parameterized over the precedence table above.

**Exit:** `maxim --llm mistral-7b-instruct-v0.2 --sim "test"` on a Mac with peer config spawns a local llama-cpp-server for mistral. Verification:
```bash
MAXIM_LLM_PROFILE=mistral-7b-instruct-v0.2 python -c "
from maxim.runtime.lane_backends import build_primary_router
router, mgr = build_primary_router()
# Assert no lane points at the remote
for name, backend in mgr._backends.items():
    print(name, backend.base_url if hasattr(backend, 'base_url') else 'local')
"
```

**Rollback:** Revert the commit. The removed reconcile block is preserved in git history and can be restored if users push back on the precedence change.

---

## P2 — Apple Silicon VRAM reporting

**Bug:** [capabilities.py:72-114](../../src/maxim/runtime/capabilities.py#L72-L114) reports `vram_gb=0.0` on MPS because it only queries `torch.cuda.get_device_properties`. Apple Silicon has unified memory; the GPU shares the full RAM pool minus OS reservation.

**Files touched:** `runtime/capabilities.py` (add MPS branch), tests.

**Design:**

When `torch.backends.mps.is_available()` returns True AND `platform.machine()` is `arm64`, compute `vram_gb = min(ram_gb * headroom_factor, ceiling_gb)` where:

- `headroom_factor`: default `0.75` (Apple Metal recommends ≥25% free for OS and other GPU clients). Configurable via `MAXIM_MPS_VRAM_HEADROOM`, clamped to `[0.25, 0.85]`.
- `ceiling_gb`: `64.0` (no quantized GGUF currently needs more than ~48GB). Prevents absurd values on 192GB M2 Ultras from polluting reports.

Intel Macs (`platform.machine() != "arm64"`) keep the existing `vram_gb=0.0` behavior. PyTorch's MPS support on Intel is deprecated and we don't want to guess at discrete-AMD-GPU sizing.

**Implementation sketch:**

```python
def detect_compute_resources() -> tuple[bool, str | None, float, float]:
    has_gpu = False
    gpu_type: str | None = None
    vram_gb = 0.0
    ram_gb = 0.0

    # RAM (needed first for the MPS branch below)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        ram_gb = _fallback_ram_from_proc()

    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            has_gpu = True
            props = torch.cuda.get_device_properties(0)
            gpu_type = props.name
            vram_gb = props.total_memory / (1024**3)
        else:
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and getattr(mps, "is_available", lambda: False)():
                has_gpu = True
                gpu_type = "mps"
                if _is_apple_silicon():
                    vram_gb = _mps_effective_vram_gb(ram_gb)
                # else: leave vram_gb at 0.0 — Intel Mac MPS unsupported
    except Exception:
        pass

    return has_gpu, gpu_type, vram_gb, ram_gb


def _is_apple_silicon() -> bool:
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mps_effective_vram_gb(ram_gb: float) -> float:
    """Apple Silicon unified memory — Metal recommends ≥25% OS headroom."""
    import os
    try:
        headroom = float(os.environ.get("MAXIM_MPS_VRAM_HEADROOM", "0.75"))
    except ValueError:
        headroom = 0.75
    headroom = max(0.25, min(headroom, 0.85))
    ceiling = 64.0
    return min(ram_gb * headroom, ceiling)
```

**Hazards:**
- **macOS memory pressure at startup is not consulted.** The user's actual free memory depends on Chrome tabs / Xcode / etc. The static `0.75` factor is a guess about idle state. Mitigation: document that `MAXIM_MPS_VRAM_HEADROOM=0.5` is a reasonable setting for "I run with Chrome open".
- **`platform.machine()` returns `arm64` even when running under Rosetta 2.** A Rosetta Python wouldn't actually have Metal access (PyTorch would see no MPS), so this is a non-issue — we check MPS first.
- **The ceiling of 64GB is arbitrary.** It's there to prevent 192GB M2 Ultra reports from implying we have 144GB of usable VRAM, which is misleading. Revisit if a new profile grows beyond 48GB effective.

**Tests:**
- `test_detect_mps_apple_silicon_24gb`: mock MPS + arm64 + 24GB RAM, assert `(True, "mps", 18.0, 24.0)`.
- `test_detect_mps_apple_silicon_8gb`: `(True, "mps", 6.0, 8.0)`.
- `test_detect_mps_apple_silicon_192gb_capped`: assert `vram_gb == 64.0` (ceiling).
- `test_detect_mps_intel_mac`: mock MPS + x86_64, assert `vram_gb == 0.0`.
- `test_detect_mps_headroom_env_override`: `MAXIM_MPS_VRAM_HEADROOM=0.5`, assert 24GB → 12GB.
- `test_detect_mps_headroom_clamp`: set to `0.95`, assert clamped to `0.85`.

**Testability note:** Mocking MPS cleanly requires patching both `torch.backends.mps.is_available` and `torch.cuda.is_available`. To simplify tests, split `detect_compute_resources` into a pure function `_compute_resources_from_state(has_cuda, has_mps, cuda_props, platform_machine, ram_gb)` and a thin wrapper. Test the pure function with direct inputs; smoke-test the wrapper once.

**Exit:** On a 24GB Apple Silicon Mac, `python -c "from maxim.runtime.capabilities import detect_compute_resources; print(detect_compute_resources())"` prints `(True, "mps", 18.0, 24.0)`.

**Rollback:** Revert the `_mps_effective_vram_gb` branch; vram_gb returns to 0.0 and MPS cannot enter the large tier (P3 falls through).

---

## P3 — MPS into the large tier

**Bug:** [lane_models.py:139](../../src/maxim/runtime/lane_models.py#L139) hard-excludes `gpu_type="mps"` from the large-tier branch.

**Files touched:** `runtime/lane_models.py`, tests.

**Design:**

Drop the `caps.gpu_type not in ("mps", None)` exclusion. The large-tier condition becomes `caps.has_gpu and caps.vram_gb >= 4.0`. With P2 landed, this means any Apple Silicon Mac with effective VRAM ≥4GB (i.e. ≥6GB total RAM at default headroom) enters the large-tier branch.

The existing `elif caps.gpu_type == "mps"` fallback becomes unreachable under P2+P3 and is deleted. (Retaining dead code is worse than documenting the absence; `git log` preserves history.)

**Implementation sketch:**

```python
# Before:
if caps.has_gpu and caps.gpu_type not in ("mps", None) and caps.vram_gb >= 4.0:
    profile = env_profile or _pick_infer_profile(caps.vram_gb, ...)
    tiers["large"] = LaneConfig(..., requires_gpu=True, device="gpu", n_gpu_layers=-1)
elif caps.has_gpu and caps.gpu_type == "mps":
    m_profile = env_profile or _pick_medium_profile(caps.ram_gb, profile_available)
    if m_profile is not None:
        tiers["medium"] = LaneConfig(..., device="auto")

# After:
if caps.has_gpu and caps.vram_gb >= 4.0:
    profile, n_ctx = _pick_infer_profile_with_ctx(  # new signature from P4c
        caps.vram_gb, profile_available=profile_available,
    )
    if profile is not None:
        tiers["large"] = LaneConfig(
            name="large",
            max_workers=1,
            requires_gpu=(caps.gpu_type != "mps"),  # MPS uses device="auto"
            model_profile=profile,
            device="auto" if caps.gpu_type == "mps" else "gpu",
            n_gpu_layers=-1,
            n_ctx=n_ctx,  # P4c field
        )
```

**Hazards:**
- **`requires_gpu=True` on MPS is wrong** — that field is historically used to route cloud backends around local-GPU requirements. For MPS we want `requires_gpu=False` so the lane doesn't reserve a CUDA worker. Verify by tracing how `requires_gpu` is consumed.
- **Removing the elif branch** changes behavior for Macs below 6GB RAM (now there's no large tier and also no medium tier created from the MPS branch). These machines fall through to the "no GPU" branch at [lane_models.py:161](../../src/maxim/runtime/lane_models.py#L161) which still creates a medium tier. Double-check the fall-through still works.

**Tests:**
- `test_tier_detection_24gb_mac`: caps with `(True, "mps", 18.0, 24.0)`, assert large tier created with `device="auto"`.
- `test_tier_detection_8gb_mac_below_threshold`: caps with `(True, "mps", 3.0, 4.0)` (hypothetical sub-minimal), assert falls through to medium tier via the "no GPU" branch.
- `test_tier_detection_cuda_still_works`: caps with `(True, "RTX 5080", 16.0, 64.0)`, assert large tier unchanged (`device="gpu"`, `requires_gpu=True`).
- `test_tier_detection_apple_silicon_large_not_medium`: assert no "medium" entry is created for an MPS Mac that qualifies for "large".

**Exit:** `detect_tiers(RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=18.0, ram_gb=24.0))` returns a dict with a `"large"` key and no `"medium"` key.

**Rollback:** Revert the condition change. MPS loses large-tier access, P4a's qwen2.5-14b becomes unreachable on Mac.

---

## P4a — Tier table row for qwen2.5-14b

**Bug:** `_INFER_VRAM_TIERS` tops out at `llama-2-13b-chat` for 14GB+ GPUs. qwen2.5-14b is a strictly better model (32K context, stronger tool-calling) and has no row.

**Files touched:** `runtime/lane_models.py` (table), tests.

**Design:**

```python
_INFER_VRAM_TIERS: tuple[tuple[float, str], ...] = (
    (16.0, "qwen2.5-14b-instruct"),     # 14B Q4 ~9.5GB weights, 32K max ctx
    (14.0, "llama-2-13b-chat"),          # 13B Q4 ~8GB, 4K ctx
    (8.0, "llama-3-8b-instruct"),        # 8B Q4 ~5GB, 8K ctx
    (4.0, "mistral-7b-instruct-v0.2"),   # 7B Q4 ~4.5GB, 4K ctx
    (0.0, "smollm-1.7b-instruct"),       # 1.7B Q4 ~1GB, CPU fallback
)
```

Paired with F0.3 (leader pinning), an existing leader with 16GB+ VRAM that already runs mistral-7b stays on mistral-7b. A *fresh* install on a 16GB+ GPU gets qwen2.5-14b.

**Hazards:**
- **P4a in isolation produces a non-functional lane** for users who don't have qwen2.5-14b downloaded. P5's auto-download fills this gap. Until P5 lands, the tier table's `profile_available` callback catches this and falls through to llama-2-13b-chat. Verify in tests.
- **F0.3 ordering is non-negotiable.** P4a cannot ship without F0.3 deployed to every leader that might be affected. Enforce via CI: the P4a commit depends on the F0.3 commit.

**Tests:**
- `test_infer_tier_18gb_picks_qwen2_5_14b`: no availability filter, assert `qwen2.5-14b-instruct`.
- `test_infer_tier_18gb_qwen_unavailable_picks_llama13b`: profile_available returns False for qwen, assert fallback to llama-2-13b-chat.
- `test_infer_tier_16gb_qwen_boundary`: vram=16.0 exactly, assert qwen2.5-14b.
- `test_infer_tier_15gb_picks_llama13b`: vram=15.0, assert llama-2-13b.
- `test_infer_tier_existing_tests_unchanged`: any existing parameterized test over the tier table continues to pass for the rows that are unchanged.

**Exit:** Fast suite passes. Regression test covering `_pick_infer_profile(vram_gb=18.0)` returns `qwen2.5-14b-instruct`.

**Rollback:** Revert the table change.

---

## P4b — Profile metadata + drift guard

**Bug:** P4c's dynamic n_ctx formula needs per-profile architecture metadata (layer count, KV head count, head dimension). These fields don't exist on profiles today.

**Files touched:** `models/language/config.py` (add fields to 11 `llama_cpp` profiles), new `tests/unit/test_profile_metadata.py`.

**Design:**

Extend each `llama_cpp` profile in `_BUILTIN_PROFILES` with:

```python
"qwen2.5-14b-instruct": {
    # ... existing fields ...
    "n_ctx": 32768,              # Maximum context the model was trained with
    "arch": {
        "n_layers": 48,
        "n_kv_heads": 8,         # GQA: 8 KV heads for 40 attention heads
        "head_dim": 128,
        "kv_type_bytes": 2,      # f16 default; q4_0 = 0.5 effectively
        "weights_gb": 9.5,       # At Q4_K_M; see quantization handling below
    },
},
```

**The 11 profiles to annotate:**

`mistral-7b-instruct-v0.2`, `smollm-1.7b-instruct`, `llama-2-7b-chat`, `llama-2-13b-chat`, `llama-3-8b-instruct`, `phi-2`, `phi-3-mini-4k-instruct`, `qwen2.5-14b-instruct`, `qwen2-7b-instruct`, `gemma-2b-it`, `gemma-7b-it`.

Values come from each model's HuggingFace config.json, cross-referenced against the GGUF metadata. Document the source in a comment above each profile entry.

**Quantization handling:** `weights_gb` is profile-specific — always Q4_K_M in our existing download registry. If a user swaps in a different quantization (by dropping a different GGUF into `~/.maxim/models/`), the profile's `weights_gb` is wrong. Fix: when the GGUF file exists on disk, `_effective_weights_gb(profile_name)` returns `Path(model_path).stat().st_size / 1024**3` instead of the profile field. Fall back to the profile field only when the file is absent (pre-download estimation).

**Drift guard (new test file):**

```python
# tests/unit/test_profile_metadata.py
import pytest
from maxim.models.language.config import _BUILTIN_PROFILES

_LLAMA_CPP_PROFILES = [
    name for name, data in _BUILTIN_PROFILES.items()
    if data.get("backend") == "llama_cpp"
]

@pytest.mark.parametrize("profile_name", _LLAMA_CPP_PROFILES)
def test_profile_has_architecture_metadata(profile_name: str):
    data = _BUILTIN_PROFILES[profile_name]
    arch = data.get("arch")
    assert arch is not None, f"{profile_name} missing 'arch' metadata"
    for field in ("n_layers", "n_kv_heads", "head_dim", "kv_type_bytes", "weights_gb"):
        assert field in arch, f"{profile_name} missing arch.{field}"

@pytest.mark.parametrize("profile_name", _LLAMA_CPP_PROFILES)
def test_profile_architecture_metadata_sane(profile_name: str):
    arch = _BUILTIN_PROFILES[profile_name]["arch"]
    assert 8 <= arch["n_layers"] <= 128, f"{profile_name} n_layers out of range"
    assert 1 <= arch["n_kv_heads"] <= arch["n_layers"], f"{profile_name} n_kv_heads out of range"
    assert 32 <= arch["head_dim"] <= 256, f"{profile_name} head_dim out of range"
    assert arch["kv_type_bytes"] in (0.5, 1, 2, 4), f"{profile_name} kv_type_bytes invalid"
    assert 0.3 <= arch["weights_gb"] <= 100, f"{profile_name} weights_gb out of range"

def test_known_profile_values():
    """Spot-check a handful of profiles against known-good values.

    This is the last line of defense against typos in the metadata —
    the drift guard above catches structure, this catches specific values.
    """
    qwen = _BUILTIN_PROFILES["qwen2.5-14b-instruct"]["arch"]
    assert qwen["n_layers"] == 48
    assert qwen["n_kv_heads"] == 8
    assert qwen["head_dim"] == 128

    mistral = _BUILTIN_PROFILES["mistral-7b-instruct-v0.2"]["arch"]
    assert mistral["n_layers"] == 32
    assert mistral["n_kv_heads"] == 8  # GQA
    assert mistral["head_dim"] == 128
```

**Hazards:**
- **Metadata can drift from reality.** The spot-check test catches canonical profiles; the structural test catches shape errors. Combined, they stop bad metadata from landing. A bad value that passes both (e.g., `n_layers=32` for a model that's actually 48) produces wrong n_ctx estimates that are conservative in one direction and OOM in the other.
- **User-added custom profiles** via `llm.json` have no `arch` field. P4c's `estimate_max_ctx` falls back to `_NCTX_SAFE_FALLBACK` for these. If the fallback table doesn't have them either, the old `MAXIM_AUTO_SPAWN_N_CTX=8192` default applies.

**Tests:** See drift guard above.

**Exit:** All 11 `llama_cpp` profiles have `arch` fields. Drift guard passes.

**Rollback:** Trivial — `arch` fields are additive and unused until P4c reads them.

---

## P4c — Dynamic n_ctx sizing + KV quantization + hot-swap refresh ✅ DONE

**Status:** Landed. `LaneConfig.n_ctx` + `kv_quant_mode` fields, `estimate_max_ctx` formula, `_NCTX_SAFE_FALLBACK` measured table, `_pick_infer_profile_with_ctx` walker, `--llm-n-ctx` CLI flag (forwards to `MAXIM_LLM_N_CTX` env), `LocalServerSpawner._build_cmd` emits `--type_k`/`--type_v` for non-f16 modes, and `_estimate_swap_n_ctx` recomputes context budget on hot-swap. 26 new tests in `tests/unit/test_estimate_max_ctx.py`.

**Bug:** The existing `MAXIM_AUTO_SPAWN_N_CTX=8192` default is a static guess. It's wrong in both directions: a 24GB Mac with qwen2.5-14b can run 16K+, a 16GB discrete GPU can run only 4K safely. No mechanism adjusts for hardware.

**Files touched:** `runtime/lane_models.py` (new `estimate_max_ctx`, fallback table, `_pick_infer_profile_with_ctx`), `runtime/worker_pool.py` (add `LaneConfig.n_ctx`), `runtime/lane_backends.py` (thread `n_ctx` through auto-spawn, hot-swap refresh), `runtime/local_server_spawner.py` (optional `--ctk`/`--ctv` args), `cli_parser.py` (`--llm-n-ctx`), tests.

**Design:**

**Formula-based estimate with retry-on-failure:**

```python
# runtime/lane_models.py

def estimate_max_ctx(
    profile_meta: dict,
    vram_budget_gb: float,
    *,
    safety_margin_gb: float = 1.5,
    kv_quant_mode: str = "f16",  # "f16", "q8_0", "q4_0"
) -> int:
    """Largest n_ctx (multiple of 1024) that fits weights + KV + margin.

    Returns 0 if the weights alone exceed the budget — caller should
    fall through to a smaller profile.
    """
    arch = profile_meta.get("arch")
    if arch is None:
        return 0  # missing metadata — caller must use fallback table
    weights_gb = arch["weights_gb"]
    n_layers = arch["n_layers"]
    n_kv_heads = arch["n_kv_heads"]
    head_dim = arch["head_dim"]
    kv_type_bytes = {"f16": 2, "q8_0": 1, "q4_0": 0.5}.get(kv_quant_mode, 2)

    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * kv_type_bytes
    available_kv_gb = vram_budget_gb - weights_gb - safety_margin_gb
    if available_kv_gb <= 0:
        return 0

    max_tokens = int(available_kv_gb * (1024**3) / kv_bytes_per_token)
    # Cap at the profile's declared maximum (don't exceed training context)
    profile_max = profile_meta.get("n_ctx", max_tokens)
    max_tokens = min(max_tokens, profile_max)
    # Round down to a multiple of 1024 (llama.cpp prefers but doesn't require)
    return (max_tokens // 1024) * 1024


def _pick_infer_profile_with_ctx(
    vram_gb: float,
    profile_available: ProfileAvailabilityCheck | None = None,
    *,
    kv_quant_mode: str = "f16",
    min_viable_ctx: int = 2048,
) -> tuple[str | None, int]:
    """Walk the tier table, picking the largest profile that produces a
    viable n_ctx for the given VRAM budget.

    Returns (profile_name, n_ctx). profile_name is None if no profile fits
    (caller should emit a diagnostic and fall back to CPU smollm).
    """
    from maxim.models.language.config import _BUILTIN_PROFILES
    check = profile_available or (lambda _: True)
    for min_vram, profile_name in _INFER_VRAM_TIERS:
        if vram_gb < min_vram:
            continue
        if not check(profile_name):
            continue
        profile_data = _BUILTIN_PROFILES.get(profile_name, {})
        # Try formula first
        n_ctx = estimate_max_ctx(profile_data, vram_gb, kv_quant_mode=kv_quant_mode)
        # Cross-check against fallback table
        fallback = _lookup_fallback_nctx(profile_name, vram_gb)
        if fallback > 0:
            n_ctx = min(n_ctx, fallback) if n_ctx > 0 else fallback
        if n_ctx >= min_viable_ctx:
            return profile_name, n_ctx
        # If formula said 0 (weights too big) or below minimum, try KV quant
        if kv_quant_mode == "f16":
            n_ctx_q4 = estimate_max_ctx(profile_data, vram_gb, kv_quant_mode="q4_0")
            if n_ctx_q4 >= min_viable_ctx:
                # Caller needs to know we picked q4_0 — return a marker?
                # Simplest: return a tuple with kv mode too.
                return profile_name, n_ctx_q4  # TODO: also return kv_mode
        # Fall through to next-smaller profile
    return None, 0
```

**The fallback table (measured values):**

```python
_NCTX_SAFE_FALLBACK: dict[str, list[tuple[float, int]]] = {
    # profile: [(min_vram_gb, n_ctx), ...] — walk largest-first
    "qwen2.5-14b-instruct": [
        (24.0, 16384),
        (18.0, 8192),
        (16.0, 4096),
    ],
    "llama-2-13b-chat": [
        (16.0, 4096),
        (14.0, 2048),
    ],
    "llama-3-8b-instruct": [
        (12.0, 8192),
        (8.0, 4096),
    ],
    "mistral-7b-instruct-v0.2": [
        (8.0, 4096),
        (4.0, 2048),
    ],
    # Add measured values per new profile.
}
```

**`LaneConfig.n_ctx` field:**

```python
# runtime/worker_pool.py
@dataclass
class LaneConfig:
    # ... existing fields ...
    n_ctx: int | None = None  # None → spawner uses its default
    kv_quant_mode: str = "f16"  # "f16", "q8_0", "q4_0"
```

**Auto-spawn n_ctx source of truth:**

```python
# runtime/lane_backends.py, replacing MAXIM_AUTO_SPAWN_N_CTX default
resolved_n_ctx = (
    infer_cfg.n_ctx  # from tier detection + estimate_max_ctx
    or _safe_int_env("MAXIM_AUTO_SPAWN_N_CTX", 0)  # env override
    or _safe_int_env("MAXIM_LLM_N_CTX", 0)  # CLI override
    or 8192  # final fallback
)
spawner = LocalServerSpawner(
    model_path=model_path,
    ...
    n_ctx=resolved_n_ctx,
    kv_quant_mode=infer_cfg.kv_quant_mode,  # new spawner arg
)
```

**`--llm-n-ctx` CLI override** at `cli_parser.py`:

```python
core.add_argument(
    "--llm-n-ctx",
    type=int,
    default=None,
    dest="llm_n_ctx",
    metavar="N",
    help="Override auto-computed llama.cpp n_ctx. Use for tuning against "
         "specific VRAM budgets. Warning: a value that exceeds the formula "
         "estimate may OOM the GPU at load time.",
)
```

**`LocalServerSpawner` KV quant args** at `runtime/local_server_spawner.py`:

```python
def __init__(
    self,
    *,
    ...
    kv_quant_mode: str = "f16",
) -> None:
    ...
    self._kv_quant_mode = kv_quant_mode

def _build_cmd(self) -> list[str]:
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", self._model_path,
        ...
        "--n_ctx", str(self._n_ctx),
    ]
    if self._kv_quant_mode in ("q8_0", "q4_0"):
        cmd.extend(["--type_k", self._kv_quant_mode, "--type_v", self._kv_quant_mode])
    ...
```

**Hot-swap refresh:** `swap_llm_server` at [lane_backends.py:1158](../../src/maxim/runtime/lane_backends.py#L1158) already notifies routers of the new `n_ctx` at line 1227-1231. Update that path to call `_pick_infer_profile_with_ctx` for the swap target (not just use the profile default), so `maxim peer llm qwen2.5-14b` on a 16GB card picks 4096, not 32768.

**Hazards:**
- **The formula assumes f16 KV by default**; if llama-cpp-server was started with `--type_k q8_0 --type_v q8_0` via unrelated env tuning, the formula over-estimates. Not a common path but worth noting. Fix: read the actual running server's KV mode via llama-cpp-server's `/v1/models` response (does it expose this? unclear — investigate during implementation).
- **`min_viable_ctx=2048`** is my guess at "below this the model is useless for agentic work." Could be too aggressive — some tool-use workflows work at 1024. Tune after first real run.
- **Quantization-aware weights sizing** is only done for on-disk GGUF files. Pre-download estimation uses profile `weights_gb` which assumes Q4_K_M. If the LLM_MODELS registry ever adds a non-Q4 variant, profile weights_gb needs to match.
- **`llama_cpp.server` CLI flags for KV quant** may differ between versions. `--type_k` and `--type_v` are the current names but older versions used `--ctk`/`--ctv`. Pin a minimum `llama_cpp_python` version in `pyproject.toml` extras.

**Tests:**
- `test_estimate_max_ctx_qwen_24gb_mac`: 18GB effective, assert ≥8192, ≤32768.
- `test_estimate_max_ctx_qwen_16gb_discrete`: 16GB, assert 4096 (from fallback table).
- `test_estimate_max_ctx_qwen_12gb_returns_zero`: weights alone exceed budget.
- `test_estimate_max_ctx_respects_profile_cap`: 96GB budget, assert returns `profile.n_ctx` not an absurd number.
- `test_pick_infer_profile_walks_down_on_zero`: vram=12GB, qwen returns 0, assert fallback to llama-2-13b-chat or smaller.
- `test_pick_infer_profile_kv_quant_rescue`: qwen at 16GB returns 4096 f16 OR tries q4_0 to get more.
- `test_lane_config_n_ctx_plumbed_to_spawner`: assert `LocalServerSpawner(n_ctx=4096)` command includes `--n_ctx 4096`.
- `test_cli_llm_n_ctx_overrides_formula`: `--llm-n-ctx 2048`, assert spawner gets 2048 regardless of formula.
- `test_hot_swap_recomputes_n_ctx`: `swap_llm_server("qwen2.5-14b-instruct")` on a 16GB fixture, assert spawner started with n_ctx=4096.

**Exit:** Parameterized test matrix passes for (profile × vram_gb) coverage. Verification:
```bash
python -c "
from maxim.runtime.lane_models import _pick_infer_profile_with_ctx
print(_pick_infer_profile_with_ctx(18.0))  # (qwen2.5-14b-instruct, 8192 or 16384)
print(_pick_infer_profile_with_ctx(16.0))  # (qwen2.5-14b-instruct, 4096)
print(_pick_infer_profile_with_ctx(12.0))  # (llama-3-8b-instruct, 8192)
"
```

**Rollback:** Revert in reverse order — first remove the auto-spawn wiring change (reverts to old env-default), then remove `LaneConfig.n_ctx`, then remove the estimate function. Spawner KV-quant args are additive and can stay.

---

## P5 — Auto-download on first use with disk preflight ✅ DONE

**Status:** Landed. `models/download.py::ensure_available()` composes the F0.1–F0.5 building blocks (atomic download, integrity check, advisory `~/.maxim/util/download.lock`, storage preflight, soft budget). `_ensure_lane_profiles_available()` in `lane_backends.py` runs after `_apply_local_llm_override` and re-walks tier detection (with the missing profile filtered out) on download failure. `--auto-download` CLI flag forwards to `MAXIM_AUTO_DOWNLOAD_MODELS`. 17 new tests in `tests/unit/test_ensure_available.py`.

See F0.1 / F0.2 / F0.4 / F0.5 for prerequisites. This phase wires everything together.

**Bug:** Tier detection falls back silently when the selected profile's GGUF is missing. No user-visible prompt, no download path, no disk preflight.

**Files touched:** `models/download.py` (new `ensure_available`), `runtime/lane_backends.py` (call `ensure_available` after tier detection), `cli_parser.py` (`--auto-download` flag), `doctor/checks.py` (surface in P8), tests.

**Design:**

New `ensure_available(profile_name, *, auto: bool, interactive: bool, logger) -> bool` in `models/download.py`:

1. If `profile_has_local_file(profile_name)`, return True.
2. If profile not in `LLM_MODELS`, log warning and return False.
3. Load `report = report_storage()` (F0.5).
4. Compute `size_gb = LLM_MODELS[profile_name]["size_gb"]`.
5. `ok, reason = can_download(size_gb, headroom_gb=2.0, soft_budget_gb=_get_soft_budget())`.
6. If not `ok`: print actionable error showing `report` + `reason`, return False.
7. If `auto` flag or `MAXIM_AUTO_DOWNLOAD_MODELS=1`: proceed without prompt.
8. Else if `interactive` AND `sys.stdin.isatty()`: prompt with 30s timeout, proceed on 'y' / fail on 'n' or timeout.
9. Else: print actionable error with exact download command, return False.
10. Acquire `file_lock(~/.maxim/models/.download.lock)` (F0.4). If lock contended, print "Another maxim process is downloading — wait or kill it", return False.
11. Call `download_file` (F0.1) with `expected_bytes` and `sha256` from `LLM_MODELS`.
12. Return True.

**Call site** in `build_primary_router`:

```python
# After tier detection, before auto-spawn:
for lane_name, cfg in list(lane_configs.items()):
    if cfg.remote_url or cfg.model_profile is None:
        continue
    if not profile_has_local_file(cfg.model_profile):
        ok = ensure_available(
            cfg.model_profile,
            auto=_auto_download_enabled(),
            interactive=sys.stdin.isatty(),
            logger=logger,
        )
        if not ok:
            # Fall through: tier detection will re-walk with the missing
            # profile filtered out
            lane_configs = detect_tiers(
                capabilities,
                profile_available=lambda p: profile_has_local_file(p) and p != cfg.model_profile,
            )
            break
```

**Interactive prompt with timeout:**

```python
def _prompt_yes_no_with_timeout(question: str, timeout_s: float = 30.0) -> bool | None:
    """Returns True/False on user response, None on timeout."""
    import select, sys
    print(question, end="", flush=True)
    if sys.platform == "win32":
        # Windows: simpler threading-based approach
        return _prompt_windows(question, timeout_s)
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        print(" [timeout]")
        return None
    response = sys.stdin.readline().strip().lower()
    return response in ("y", "yes")
```

**Opt-in precedence:**

1. `--auto-download` CLI flag (highest)
2. `MAXIM_AUTO_DOWNLOAD_MODELS=1` env var
3. Interactive prompt if `sys.stdin.isatty()` AND neither above is set AND the command is `--sim`
4. Hard fail with actionable error otherwise

**Soft budget:** Reads `MAXIM_DATA_BUDGET_GB` env var. If set, passes to `can_download`. Optional; off by default.

**Hazards:**
- **All the F0.1-F0.5 hazards** compound here.
- **Interactive prompt deadlocks** if stdin is technically a tty but unattended. The 30s timeout + explicit actionable error on timeout handles this.
- **Soft budget UX** when the budget is near-full: suggestions should target the largest subdir first. `format_report` from F0.5 sorts by size for this.
- **Race condition on prompt:** user answers 'y', but between the prompt and the download start, another process grabs the file lock. Our code prints "Another maxim process is downloading" — not a user-visible bug, but confusing. Document as expected.

**Tests:**
- `test_ensure_available_file_exists_noop`: profile already downloaded, returns True without prompting.
- `test_ensure_available_disk_full`: mock `can_download` returns False, assert error message includes storage report.
- `test_ensure_available_auto_flag_skips_prompt`: `auto=True`, assert no prompt, proceeds to download.
- `test_ensure_available_interactive_yes`: mock tty stdin, inject 'y', assert proceeds.
- `test_ensure_available_interactive_no`: mock tty stdin, inject 'n', assert returns False.
- `test_ensure_available_interactive_timeout`: mock tty stdin with no input, assert returns False after timeout.
- `test_ensure_available_no_tty_no_flag_hard_fails`: assert returns False, error message mentions `--auto-download`.
- `test_ensure_available_concurrent_lock_contention`: take the file lock in a background thread, call `ensure_available`, assert returns False with "another process downloading" message.
- `test_ensure_available_fall_through_on_fail`: first profile not downloaded, `ensure_available` returns False, tier detection re-walks and picks a smaller profile that IS downloaded.
- `test_soft_budget_rejection`: `MAXIM_DATA_BUDGET_GB=10` with 9GB used, requesting 2GB, assert rejection.

**Exit:**
- `maxim --sim "test"` on a cold 24GB Mac with stdin=tty: prompts, downloads on yes.
- Same with stdin piped: exits with actionable error.
- Second concurrent invocation: polite "another process downloading" message.

**Rollback:** Revert the call site in `build_primary_router`. `ensure_available` becomes dead code; F0.1-F0.5 infrastructure stays (they're independently useful).

---

## P6 — Probe remote URLs with caching + structured outcomes ✅ DONE

**Status:** Landed. New `ProbeResult` + `probe_llm_server` in `runtime/llm_server.py` with two-attempt retry and structured outcome classification (ok / auth_rejected / dns_fail / tls_error / connection_refused / timeout / http_5xx / other). New `runtime/probe_cache.py` with TTL-bounded on-disk cache, single-URL eviction, full clear. `_validate_remote_urls` rewritten to probe ALL remote URLs (public + loopback) and use cache short-circuiting; `auth_rejected` keeps the lane wired with a key-rotation hint. `peer/cli.py` clears the cache on `connect`, `forget`, `restart`, `update`, `llm` commands. Env knobs: `MAXIM_SKIP_REMOTE_PROBE`, `MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S` (clamped 0.2-5.0), `MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S` (0.5-10.0), `MAXIM_REMOTE_PROBE_CACHE_TTL_S` (0-600). 32 new tests in `tests/unit/test_probe_remote.py`.

**Bug:** `_validate_remote_urls` at [lane_backends.py:735-760](../../src/maxim/runtime/lane_backends.py#L735-L760) skips public URLs. Dead tunnels wedge peers into retry storms.

**Files touched:** `runtime/llm_server.py` (enhanced probe with structured outcome), `runtime/lane_backends.py` (updated `_validate_remote_urls`), new `runtime/probe_cache.py` (~80 LOC), `peer/cli.py` (cache clearing call sites), tests.

**Design:**

**Structured probe outcome:**

```python
# runtime/llm_server.py
@dataclass(frozen=True)
class ProbeResult:
    url: str
    outcome: Literal["ok", "auth_rejected", "dns_fail",
                     "connection_refused", "tls_error",
                     "timeout", "http_5xx", "other"]
    detail: str  # short diagnostic
    latency_ms: float | None

def probe_llm_server(
    url: str,
    *,
    api_key: str | None = None,
    first_timeout_s: float = 0.8,
    retry_timeout_s: float = 2.5,
) -> ProbeResult:
    """Probe GET /v1/models with optional Bearer auth. Two-attempt retry."""
    # First attempt: aggressive timeout
    result = _probe_once(url, api_key, first_timeout_s)
    if result.outcome in ("ok", "auth_rejected"):
        return result
    # Retry with longer budget for slow leaders
    retry = _probe_once(url, api_key, retry_timeout_s)
    return retry  # may also be failed; caller decides

def _probe_once(url: str, api_key: str | None, timeout_s: float) -> ProbeResult:
    """Single probe attempt. Classifies errors into structured outcomes."""
    base = url.rstrip("/")
    probe = base + "/models" if base.endswith("/v1") else base + "/v1/models"
    req = urllib.request.Request(probe)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            latency_ms = (time.monotonic() - start) * 1000
            return ProbeResult(url, "ok", f"HTTP {resp.status}", latency_ms)
    except urllib.error.HTTPError as e:
        latency_ms = (time.monotonic() - start) * 1000
        if e.code == 401:
            return ProbeResult(url, "auth_rejected", "401 Unauthorized", latency_ms)
        if 500 <= e.code < 600:
            return ProbeResult(url, "http_5xx", f"HTTP {e.code}", latency_ms)
        return ProbeResult(url, "other", f"HTTP {e.code}", latency_ms)
    except socket.gaierror as e:
        return ProbeResult(url, "dns_fail", str(e), None)
    except ssl.SSLError as e:
        return ProbeResult(url, "tls_error", str(e), None)
    except socket.timeout:
        return ProbeResult(url, "timeout", f"{timeout_s}s", None)
    except (ConnectionRefusedError, ConnectionResetError) as e:
        return ProbeResult(url, "connection_refused", str(e), None)
    except Exception as e:
        return ProbeResult(url, "other", f"{type(e).__name__}: {e}", None)
```

**Cache module:**

```python
# runtime/probe_cache.py
_CACHE_PATH = data_home() / "util" / "last_probe_status.json"

def load_cache() -> dict[str, dict]:
    """Returns {url: {"outcome": ..., "probed_at": ..., "detail": ...}}."""

def save_cache(cache: dict[str, dict]) -> None:
    """Atomic write via atomic_io."""

def is_fresh(entry: dict, ttl_s: float) -> bool:
    return time.time() - entry.get("probed_at", 0) < ttl_s

def clear_cache() -> None:
    """Best-effort delete. Called from peer/cli.py handlers."""
    try:
        _CACHE_PATH.unlink(missing_ok=True)
    except OSError:
        pass

def clear_cache_for_url(url: str) -> None:
    """Remove a single entry. Used when a specific remote is known stale."""
```

**Updated `_validate_remote_urls`:**

```python
def _validate_remote_urls(lane_configs, logger):
    if os.environ.get("MAXIM_SKIP_REMOTE_PROBE", "").strip().lower() in ("1", "true", "yes"):
        return dict(lane_configs)

    ttl_s = _safe_float_env("MAXIM_REMOTE_PROBE_CACHE_TTL_S", 60.0)
    first_timeout = _safe_float_env("MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S", 0.8)
    retry_timeout = _safe_float_env("MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S", 2.5)

    cache = probe_cache.load_cache()
    out = dict(lane_configs)
    for name, cfg in lane_configs.items():
        url = cfg.remote_url
        if not url:
            continue

        cached = cache.get(url)
        if cached and probe_cache.is_fresh(cached, ttl_s):
            result = ProbeResult(
                url=url,
                outcome=cached["outcome"],
                detail=cached.get("detail", ""),
                latency_ms=cached.get("latency_ms"),
            )
        else:
            result = probe_llm_server(
                url,
                api_key=cfg.remote_api_key,
                first_timeout_s=first_timeout,
                retry_timeout_s=retry_timeout,
            )
            cache[url] = {
                "outcome": result.outcome,
                "detail": result.detail,
                "probed_at": time.time(),
                "latency_ms": result.latency_ms,
            }

        if result.outcome in ("ok", "auth_rejected"):
            continue  # reachable (auth-rejected still counts as "server up")

        _log_probe_failure(logger, name, url, result)
        out[name] = dataclasses.replace(
            cfg, remote_url=None, remote_model=None, remote_api_key=None,
        )

    probe_cache.save_cache(cache)
    return out


def _log_probe_failure(logger, name, url, result):
    """Outcome-specific warning messages with fix hints."""
    hints = {
        "dns_fail": f"Check hostname spelling in peer.yml or $MAXIM_LANE_{name.upper()}_REMOTE_URL",
        "tls_error": "Check TLS certificate validity on the leader",
        "connection_refused": "Leader is not accepting connections — is it running?",
        "timeout": f"Leader did not respond within {result.detail} — is it cold-loading a model?",
        "http_5xx": "Leader returned a server error — check `maxim peer logs`",
        "auth_rejected": f"Auth token rejected — run `maxim peer key` to rotate",
        "other": f"Unexpected error: {result.detail}",
    }
    logger.warning(
        "Lane '%s' probe failed (outcome=%s): %s. Fix: %s. "
        "Falling back to local model selection.",
        name, result.outcome, result.detail, hints.get(result.outcome, "unknown"),
    )
```

**Cache clearing call sites** in `peer/cli.py`:

```python
# _cmd_restart, _cmd_update, _cmd_connect, _cmd_forget
def _cmd_restart(argv):
    # ... existing restart logic ...
    try:
        from maxim.runtime.probe_cache import clear_cache
        clear_cache()
    except Exception as e:
        logger.debug("Probe cache clear failed (non-fatal): %s", e)
```

Also add cache clear to `_cmd_llm` (model swap) with `clear_cache_for_url(cfg.url)` — the leader becomes transiently unavailable during swap, and we want the next probe to re-check rather than trust a stale "ok".

**Env knobs:**

- `MAXIM_SKIP_REMOTE_PROBE=1` — bypass entirely (CI escape hatch)
- `MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S` — default 0.8, clamp [0.2, 5.0]
- `MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S` — default 2.5, clamp [0.5, 10.0]
- `MAXIM_REMOTE_PROBE_CACHE_TTL_S` — default 60, clamp [0, 600]

**Hazards:**
- **Probe cache file contention** across parallel invocations. Mitigation: the cache uses `atomic_write_json` — concurrent writes lose some updates but never corrupt the file. Read-modify-write races can drop a probe result but next invocation re-probes and repairs. Not worth a lock.
- **401 treated as "alive"** means wrong API key passes the probe but every real request fails with 401 in retry loops. Fix: the structured outcome distinguishes `ok` from `auth_rejected`. `auth_rejected` emits a different warning and still counts as "reachable" so fallback doesn't fire — but the user gets a clear "rotate your key" hint instead of silent failure.
- **Cache invalidation on leader reachability change that ISN'T a peer command.** If the leader crashes unrelated to any user action, the cache can be stale for up to 60s. Accepted tradeoff per the user's approval of the TTL design.
- **`maxim peer llm <model>` swap window**: the leader is unreachable for 30-90s during swap. Cache cleared per the call site, next probe fires fresh, sees the swap-in-progress server, probably returns 5xx or timeout, drops the remote, falls back to local. Not ideal but correct. Better UX: `_cmd_llm` could *wait* for the swap to complete and then re-probe, writing an updated cache entry. Treat this as an enhancement for a follow-up.

**Tests:**
- `test_probe_ok_healthy`: mock server returns 200, assert `outcome="ok"`.
- `test_probe_auth_rejected`: mock 401, assert `outcome="auth_rejected"`.
- `test_probe_dns_fail`: mock `socket.gaierror`, assert `outcome="dns_fail"`.
- `test_probe_tls_error`: mock `ssl.SSLError`, assert `outcome="tls_error"`.
- `test_probe_connection_refused`: mock `ConnectionRefusedError`.
- `test_probe_timeout`: mock slow server, assert `outcome="timeout"`, retry fires.
- `test_probe_http_502`: mock 502, assert `outcome="http_5xx"`.
- `test_cache_load_missing_file`: assert empty dict, no exception.
- `test_cache_save_atomic`: interrupt mid-write, assert no corruption.
- `test_cache_fresh_entry_skips_probe`: populate cache, call `_validate_remote_urls`, assert probe function not called.
- `test_cache_stale_entry_reprobes`: populate with old `probed_at`, assert probe fires.
- `test_cache_clear_on_peer_restart`: run `_cmd_restart`, assert cache file missing.
- `test_cache_clear_for_url_on_peer_llm`: run `_cmd_llm`, assert specific URL entry removed but other entries preserved.
- `test_validate_drops_remote_on_outcome_dns_fail`: assert lane's `remote_url` cleared.
- `test_validate_preserves_remote_on_outcome_auth_rejected`: assert lane's `remote_url` **preserved** (auth-gated leader is still a leader; user needs to fix auth, not fall back locally).
- `test_skip_remote_probe_env`: assert early return, no probe, no cache update.

**Exit:**
- Dead Cloudflare tunnel: one warning line, 3.3s worst-case delay, remote_url cleared, local fallback takes over.
- Healthy tunnel: first invocation ~800ms, subsequent invocations within 60s: instant (cached).
- Auth-rejected: leader stays wired, user sees clear "rotate your key" hint.

**Rollback:** Revert in reverse: first the call sites in `peer/cli.py` (stop clearing cache), then `_validate_remote_urls` (revert to public-URL skip), then delete `probe_cache.py` and `ProbeResult`. Can be rolled back incrementally.

---

## P7 — Strip HTML error bodies from backend logs

**Bug:** [openai_backend.py](../../src/maxim/models/language/openai_backend.py) logs `str(last_err)` verbatim, which for Cloudflare 502s is ~4KB of HTML.

**Files touched:** `models/language/openai_backend.py`, small test.

**Design:**

```python
def _sanitize_error_message(err: BaseException, *, max_len: int = 500) -> str:
    """Collapse HTML error bodies to a short summary. Truncate long strings."""
    import re
    msg = str(err)
    # Check anywhere in the first 200 chars — exception chains can prefix
    # the message with "RuntimeError: " before the HTML begins.
    head = msg[:200]
    if "<!DOCTYPE" in head or "<html" in head.lower():
        title_match = re.search(r"<title>([^<]+)</title>", msg, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "HTML error body"
        return f"<{title}, {len(msg)} bytes>"
    if len(msg) > max_len:
        return msg[:max_len] + "… (truncated)"
    return msg
```

All call sites in `openai_backend.py` that log `str(last_err)` switch to `_sanitize_error_message(last_err)`.

**Hazards:**
- **JSON error bodies from Anthropic** are also verbose but not HTML. `_sanitize_error_message` truncates them at `max_len` — good enough without structured parsing.
- **Error chain prefixes** (`"RuntimeError: <actual error>"`) are handled by checking the first 200 chars instead of just lstrip.

**Tests:**
- `test_sanitize_html_body`: input starts with `<!DOCTYPE`, assert output is `<title, N bytes>`.
- `test_sanitize_html_with_title`: input has `<title>502 Bad Gateway</title>`, assert output is `<502 Bad Gateway, N bytes>`.
- `test_sanitize_html_after_prefix`: input is `"RuntimeError: <!DOCTYPE html>..."`, assert HTML detector fires.
- `test_sanitize_short_plain_message`: input is `"Connection refused"`, assert returned unchanged.
- `test_sanitize_long_plain_message`: input is 2KB of plain text, assert truncated to `max_len` with ellipsis.

**Exit:** A Cloudflare 502 in backend logs shows as `[WARN] OpenAI call failed [req=abc123]: <502 Bad Gateway, 4123 bytes>` instead of 4KB of HTML.

**Rollback:** Trivial.

---

## P8 — `maxim doctor` actionable checks ✅ DONE

**Status:** Landed. Four new check functions in `doctor/checks.py`: `check_tier_effectiveness` (compares actual vs ideal tier choice and emits the exact `python -m maxim.models.download` command for the gap), `check_peer_vs_local_conflict` (info notice when --llm + peer config will run locally per P1), `check_remote_reachability` (uses P6's structured probe with outcome-specific fix hints), `check_storage_footprint` (fail/warn/ok bands on free disk + per-subdir breakdown). All wired into `run_all_checks` — environment + peer connectivity sections both extended. 16 new tests in `tests/unit/test_doctor_p8_checks.py`.

**Bug:** Doctor reports tier detection passed when the routing decision is actually broken.

**Files touched:** `doctor/checks.py`, tests.

**Design:**

Four new `CheckResult`-returning functions:

1. **`check_tier_effectiveness`**: Compare what tier detection picked vs. what the capability allows. If a 24GB Mac picked mistral-7b because qwen2.5-14b isn't downloaded, surface the gap with the exact download command.

2. **`check_peer_vs_local_conflict`**: If peer config is active AND `MAXIM_LLM_PROFILE` names a local profile AND that profile's lane assignment would be the large tier, report P1 will clear the remote_url on next run (informational, not a failure).

3. **`check_remote_reachability`**: Use P6's `probe_llm_server` against each lane's `remote_url`. Report structured outcomes. If `auth_rejected`, specifically call out `maxim peer key`.

4. **`check_storage_footprint`**: Use F0.5's `report_storage()` and print the top-N subdirs by size. Fails if `fs_free_gb < 10`, warns if `< 20`.

**Hazards:**
- Adding network probes to doctor makes doctor slower. 1.5s × N lanes max. Document.
- Check 1 requires running tier detection twice — once ignoring availability, once respecting it — to see the gap. Cheap (no LLM calls) but non-trivial.

**Tests:**
- `test_check_tier_effectiveness_reports_gap`: fake caps with 18GB vram, qwen not downloaded, assert warning with the download command.
- `test_check_peer_vs_local_conflict_triggers`: fake env with peer config + `MAXIM_LLM_PROFILE=mistral-7b-instruct-v0.2`, assert info-level result.
- `test_check_remote_reachability_auth_rejected`: mock probe → 401, assert warn with `maxim peer key` hint.
- `test_check_storage_footprint_low_disk`: mock `fs_free_gb=5`, assert fail.

**Exit:** `maxim doctor` on a Mac with a dead tunnel and an undownloaded-but-tier-preferred profile prints actionable fix hints for both.

**Rollback:** Revert the check additions. Existing doctor checks unchanged.

---

## P9 — Lane decision log (NEW) ✅ DONE

**Status:** Landed. New `runtime/decision_log.py` writes append-only JSONL at `~/.maxim/util/lane_decisions.jsonl`, one record per `build_primary_router` invocation. Records carry serialized `RuntimeCapabilities`, redacted env snapshot (API keys masked, URLs reduced to hostname), tier decisions with source/profile/n_ctx/remote_host fields, peer_config_loaded flag, optional probe_results + auto_download_triggered. Rotation drops the oldest entries past `MAX_RECORDS=1000`. New `maxim doctor --last-decision` flag pretty-prints the most recent record. 14 new tests in `tests/unit/test_decision_log.py`.

**Bug:** None of the fixes emit a traceable record of "why did we pick this configuration." Post-mortem debugging of "why is my sim running qwen2.5-14b instead of mistral-7b" requires reading source code and guessing at env var state at the time of the run.

**Files touched:** `runtime/lane_backends.py` (add decision recorder), new `runtime/decision_log.py`, tests.

**Design:**

Append-only JSONL at `~/.maxim/util/lane_decisions.jsonl`. One record per `build_primary_router` invocation:

```python
@dataclass
class LaneDecisionRecord:
    timestamp: float
    pid: int
    maxim_version: str
    caps: dict  # serialized RuntimeCapabilities
    env: dict   # MAXIM_LLM_PROFILE, MAXIM_LANE_*_REMOTE_URL (URL host only, no key)
    peer_config_loaded: bool
    tier_decisions: dict[str, dict]  # lane_name -> {profile, n_ctx, source, remote_url_cleared}
    probe_results: dict[str, str]    # url -> outcome
    auto_download_triggered: list[str]  # profiles we fetched
```

Source field traces the decision origin: `tier_table`, `env_override`, `cli_override`, `peer_config`, `persisted_model`, `pinned_at_upgrade`.

Log rotation: cap at 1000 entries, prune oldest. `maxim doctor --last-decision` prints the most recent entry.

**Hazards:**
- **Leaks hostnames to disk.** Mitigation: redact URL to host-only (`urlparse().hostname`), never log full URL including paths or query strings.
- **File contention across parallel invocations**: append-only semantics with `O_APPEND` are atomic for small writes on POSIX. Each record is ~500 bytes, well under POSIX's `PIPE_BUF` atomic-write threshold.

**Tests:**
- `test_decision_log_appends_on_build`: call `build_primary_router`, assert new line in log.
- `test_decision_log_redacts_api_keys`: set `MAXIM_LANE_LARGE_REMOTE_API_KEY=secret`, assert 'secret' not in any log line.
- `test_decision_log_rotates_at_1000`: append 1001 records, assert file has 1000 most recent.
- `test_doctor_prints_last_decision`: populate log, run `maxim doctor --last-decision`, assert latest record is formatted.

**Exit:** Post-sim, user can run `tail -1 ~/.maxim/util/lane_decisions.jsonl | jq .` and see exactly why the run picked its configuration.

**Rollback:** Revert the recorder call in `build_primary_router` and delete the module. Log file stays on disk as harmless historical data.

---

## Shipping order

Split into **six commits** for bisect-ability:

**Commit A — Foundation (F0.1–F0.5).** Atomic downloads, integrity metadata, leader pinning, file locks, storage reporter. ~300 LOC. Required before P4a or P5 can ship. Zero behavior change on its own — all infrastructure, no call sites wired yet.

**Commit B — Capability detection (P2, P3).** MPS VRAM reporting, MPS into large tier. ~80 LOC. Requires F0.3 already deployed. On its own, widens the large tier to Apple Silicon but still picks mistral-7b (P4a not in yet).

**Commit C — Tier table + dynamic n_ctx (P4a, P4b, P4c).** Tier row addition, profile metadata annotations, dynamic n_ctx formula + KV quant + hot-swap refresh + `--llm-n-ctx` CLI flag. ~700 LOC (400 metadata + 300 logic). Requires Commit A (F0.3) and Commit B (P2/P3). After this commit, a 24GB Mac runs qwen2.5-14b at 8K+ context.

**Commit D — Routing precedence + probe (P1, P6, P7).** `--llm` local override, structured probe + cache, HTML error sanitization. ~400 LOC. Independent of A/B/C — can ship before or after depending on priority.

**Commit E — Auto-download (P5).** Wire F0.1-F0.5 into `build_primary_router` via `ensure_available`. `--auto-download` CLI flag. Preflight + interactive prompt. ~200 LOC. Depends on Commit A.

**Commit F — Observability (P8, P9).** Doctor checks + lane decision log. ~350 LOC. Depends on all prior commits to report accurate state.

**Fast suite must stay green after each commit.** Baseline is 3610 tests as of commit `0a0aaad`.

### Critical ordering constraints

- **F0.3 (Commit A) must land before P4a (Commit C).** The leader-pinning migration relies on the old tier table being present to compute the pre-upgrade profile. Without F0.3, P4a silently changes every existing leader's model.
- **Commits B and C together** are what actually unblocks the user's Mac. Shipping B without C is a half-measure (MPS into large tier, but still picking mistral-7b because qwen2.5-14b isn't in the tier table).
- **Commit D is independent** and can ship in parallel with B/C as long as F0 is already in.

## Test strategy

### Unit tests

All phases have unit tests. Heavy mocking for hardware-dependent code. Factor `detect_compute_resources` into a pure function + wrapper so tests don't depend on PyTorch's MPS API surface.

### Integration smoke test: `test_build_primary_router_scenarios.py`

New parameterized test covering the end-to-end routing decision:

| Scenario | Caps | Peer config | `--llm` | Expected |
|---|---|---|---|---|
| Solo Mac, no peer | `mps/24GB` | absent | none | large=qwen2.5-14b local, n_ctx≥8K |
| Mac + healthy peer | `mps/24GB` | configured | none | large=remote (probe ok, cache ok) |
| Mac + dead peer | `mps/24GB` | configured, unreachable | none | large=qwen2.5-14b local (probe failed, fallback) |
| Mac + local override | `mps/24GB` | configured | `mistral-7b-instruct-v0.2` | large=mistral local (P1), remote cleared |
| Mac + cloud override | `mps/24GB` | configured | none + `--cloud-lane large claude-sonnet` | large=claude (commit 1a5eb85 behavior) |
| Leader CUDA 16GB | `cuda/16GB/64GB` | absent | none | large=qwen2.5-14b local, n_ctx=4096 (P4c formula) |
| Cold machine + auto-download + tty | `mps/24GB`, empty `~/.maxim/models/` | absent | none, auto-download | prompts, downloads, proceeds |
| Cold machine + auto-download + no tty | same | absent | none, no auto-download | hard fail with actionable error |
| Pinned leader first upgrade | `cuda/16GB/64GB`, `util/` exists without persisted | absent | none | picks old-tier mistral-7b (F0.3 pin) |

Each scenario mocks `detect_compute_resources`, the peer config file, the probe response, `profile_has_local_file`, and stdin isatty. ~300 LOC but catches integration regressions the unit tests miss.

### Regression coverage

Every fix in this wave corresponds to a test that would have caught the bug. Specifically:

- F0.1: URLError mid-download → file leak regression.
- F0.2: Truncated download → silent bad load regression.
- F0.3: Leader upgrade → silent model switch regression.
- P1: `--llm <local>` + peer config → remote routing regression.
- P2: MPS → vram_gb=0.0 regression.
- P4c: qwen2.5-14b + 16GB GPU → OOM regression.
- P5: Cold machine → silent fallback regression.
- P6: Dead tunnel → retry storm regression.
- P7: HTML error → log flood regression.

### Manual test plan

Some scenarios can't be unit-tested without real hardware or network. Document in a manual test section for release verification:

1. 24GB Mac with peer config pointing at a dead leader → starts local sim, no hang.
2. 16GB CUDA leader with existing mistral-7b → `peer update && peer restart` preserves mistral-7b (F0.3).
3. Fresh install on 24GB Mac with `--auto-download` → downloads qwen2.5-14b, runs sim.
4. Parallel `maxim --sim` invocations → second one reports "another process downloading" and exits.
5. `MAXIM_SKIP_REMOTE_PROBE=1 maxim --sim ...` on a dead tunnel → still retries as before (no probe short-circuit).

## Observability & backwards compat

### Lane decision log (P9)

Documented above. Gives post-mortem traceability.

### Deprecation notices

In Commit D (P1), log a deprecation warning for anyone whose behavior changes:

```
DEPRECATED: --llm <local_profile> with peer config now runs locally.
Previously this was rewritten as "send <local_profile> as model name to
leader" which ignored the model name. If you wanted the leader to use
this model, run `maxim peer llm <profile>` explicitly.
```

Shown once per session.

### Migration path for custom `llm.json`

Users with custom profiles in `~/.maxim/config/llm.json` that mirror `_BUILTIN_PROFILES`:

- **P4b arch metadata:** custom profiles without `arch` fields fall back to `_NCTX_SAFE_FALLBACK` for n_ctx estimation. If the fallback table doesn't have them, auto-spawn uses `MAXIM_AUTO_SPAWN_N_CTX=8192` (unchanged from today).
- **F0.2 integrity metadata:** custom profiles with `LLM_MODELS` entries needing `expected_bytes` are flagged in doctor but not auto-fixed.

Document in `docs/user/custom_llm_profiles.md` (create if missing).

## Non-goals

- **Model registry overhaul.** Adding qwen2.5-14b metadata, not rewriting the profile system.
- **Auto-failover from dead peer to fresh local download.** P5 and P6 make this possible in principle, but the compound UX ("leader down → auto-download → prompt interactively → start local") deserves its own design pass.
- **Resumable downloads.** `urlretrieve` doesn't support them. Migrate to `huggingface_hub.hf_hub_download` in a follow-up.
- **Peer discovery / mesh routing.** `maxim.mesh.peer_registry` has aspirational code; not touching.
- **Changing `maxim peer llm` semantics.** Stays as-is (hot-swap on the leader).
- **HTTP retry policy changes.** The cancellation-primitive wave in commit `cb360a9` already made retries interruptible. Not re-touching retry counts or backoffs.
- **Custom llm.json validation at startup.** User's responsibility for now.
- **Runtime-mid-sim hot-swap to undownloaded model.** Downloads only happen at `build_primary_router` time. A future `peer llm --auto-download <model>` can lift this.

## Risks & open questions

Resolved by this revision:

- ~~Disk-space handling.~~ F0.5 + P5.
- ~~qwen2.5-14b on 16GB cards.~~ P4c formula.
- ~~Startup probe timeout.~~ P6 design.
- ~~Probe cache invalidation on peer state changes.~~ P6 4a.
- ~~Auto-download UX.~~ P5 tty-gated prompt with 30s timeout.

Still open:

1. **Resumable downloads via `huggingface_hub`.** Larger change than this wave can absorb. Track as a follow-up for 0.3.
2. **`maxim peer llm <model>` when the model isn't on the leader.** Admin endpoint could fetch it. Deferred.
3. **Leader-side soft budget.** P5's `MAXIM_DATA_BUDGET_GB` applies per-process; a leader serving many peers might want a shared budget. Deferred; single-user assumption holds for 0.2.2.
4. **`min_viable_ctx=2048` is my guess.** Tune after first real run with tight VRAM.
5. **Probe cache cross-invocation contention.** Atomic writes lose some updates; this is accepted as a cache, not a source of truth. If real-world usage shows stale cache hurting, upgrade to a per-file advisory lock.
6. **Leader tier-detection behavior on Commit C.** F0.3 pins existing leaders, but leader hardware upgrades between Commit A and Commit C land windows could get unpinned. Mitigation: leaders should run `maxim peer llm <desired>` explicitly after any hardware change regardless of this wave. Document in CLAUDE.md.
7. **Storage report walks on slow filesystems.** Cache helps but first-walk latency on NFS or network drives could block startup. Mitigation: hard deadline on the walk (500ms) + partial-result flag. Implement in F0.5 if it actually bites.
8. **Metadata drift between HF model config and our hardcoded values.** Spot-check tests catch typos but not upstream config changes. If Qwen ever re-releases qwen2.5-14b with different `n_kv_heads`, our formula will be wrong until someone notices. Consider a CI job that pulls HF configs and diffs against our arch metadata.

## Issues discovered during investigation

Real bugs in existing code, not design gaps in the plan. All are addressed by one of the phases above; restating here as a checklist for implementation:

| # | Bug | Location | Fixed by |
|---|---|---|---|
| 1 | Partial download leak on `URLError` | [download.py:253-255](../../src/maxim/models/download.py#L253-L255) | F0.1 |
| 2 | No `KeyboardInterrupt` handler on download | [download.py:249-261](../../src/maxim/models/download.py#L249-L261) | F0.1 |
| 3 | `LLM_MODELS` has no integrity fields | [download.py:38-95](../../src/maxim/models/download.py#L38-L95) | F0.2 |
| 4 | Profile `n_ctx` diverges from auto-spawn `n_ctx` | [lane_backends.py:966](../../src/maxim/runtime/lane_backends.py#L966) | P4c |
| 5 | No file locking primitive in codebase | — | F0.4 |
| 6 | `LaneConfig` has no `n_ctx` field | [worker_pool.py:73-90](../../src/maxim/runtime/worker_pool.py#L73-L90) | P4c |
| 7 | `LocalServerSpawner` no KV quant support | [local_server_spawner.py:337-357](../../src/maxim/runtime/local_server_spawner.py#L337-L357) | P4c |
| 8 | Public URL probe opt-out | [lane_backends.py:747-748](../../src/maxim/runtime/lane_backends.py#L747-L748) | P6 |
| 9 | HTML error body logged verbatim | [openai_backend.py](../../src/maxim/models/language/openai_backend.py) | P7 |
| 10 | Silent tier fall-through on missing model | [lane_backends.py:763-860](../../src/maxim/runtime/lane_backends.py#L763-L860) | P5 |
| 11 | No traceable routing decision log | — | P9 |
| 12 | `profile_has_local_file` accepts `.partial` files | [llm_server.py:141-156](../../src/maxim/runtime/llm_server.py#L141-L156) | F0.1 |

## References

- [lane_backends.py::build_primary_router](../../src/maxim/runtime/lane_backends.py) — the orchestration function every phase modifies
- [capabilities.py::detect_compute_resources](../../src/maxim/runtime/capabilities.py) — broken VRAM detection (P2)
- [lane_models.py::detect_tiers](../../src/maxim/runtime/lane_models.py) — tier selection (P3, P4a)
- [llm_server.py::llm_server_responding_at](../../src/maxim/runtime/llm_server.py) — existing probe function (P6 base)
- [download.py::download_file](../../src/maxim/models/download.py) — existing download path with atomicity bugs (F0.1)
- [atomic_io.py](../../src/maxim/utils/atomic_io.py) — existing atomic-write pattern (reuse for probe cache)
- [peer/config.py](../../src/maxim/peer/config.py) — peer.yml load path
- [CLAUDE.md architectural invariants](../../../CLAUDE.md) — "LLM access goes through `models/language/router.py`; backends should not be imported directly from outside `models/language/`." Preserved by this wave — nothing imports backends directly.
- [substrate_plan.md](substrate_plan.md) — spine work (parallel execution, zero file overlap, see "Parallel execution" section above)
- [cleanup_wave.md](cleanup_wave.md) — parallel UX wave shipping in 0.2.2
- Commits from prior stability wave (all in `main`): `1a5eb85` (`--cloud-lane` precedence + Claude IDs), `3a30d23` (declarative mode filter), `984ac5f` (bio tier + concept quality), `cb360a9` (interruptible retries + cancellation primitive), `0a0aaad` (phantom cost fix), `c4e7cd7` (orchestrator ping-pong + max-turns termination).
