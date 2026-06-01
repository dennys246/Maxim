# Config Unification — Single-Source Operator Configuration with Layered Precedence

**Status:** Drafted 2026-06-01. Ships as ONE PR with six folded stages (C1 config.json loader + schema, C2 `maxim config` CLI verbs, C3 role detection unification, C4 per-tier remote routing migration, C5 doctor surfacing, C6 deprecation warnings + docs). Worktree: `Maxim-wt-config-unification`. Branch: `feat/v1-config-unification`. Sibling track to [leader_ux_profile_management.md](leader_ux_profile_management.md) under the 1.0 plan — independent of §B5 Hivemind and Tier 1 graduations.

**Triggered by:** the same Mac Mini leader setup that surfaced [leader_ux_profile_management.md](leader_ux_profile_management.md) hit a deeper class of problem. The operator spent ~2 hours fighting:

1. `MAXIM_ROLE=leader` set in shell A, not inherited into tmux session created from shell B → role detection silently fell back to `solo` → tunnel didn't route
2. `~/.cloudflared/config.yaml` (yaml extension from cloudflared's auto-write) vs `_cloudflared_config_exists()` only checking `.yml` → leader-mode detector returned `solo`
3. Stale `~/.config/maxim/mesh.yml` + `peer.yml` from earlier exploration → role detector in `runtime/role.py` returned `peer` → SECOND role detector in `runtime/leader_mode.py` returned `solo` → silent divergence
4. `MAXIM_LANE_LARGE_REMOTE_URL` env var from when this Mac was a peer leaking into leader-mode startup → "self-hosted" lane classification

None of these failure modes was discoverable from the symptoms (`role=solo` in one log line, no other signal). The operator had no single place to look to answer "what does this leader think it's configured as?" The actual answer is scattered across ~96 environment variables, 4 declarative config files (`peer.yml`, `mesh.yml`, `profiles.yml`, `api_key`), and 1 external config (`~/.cloudflared/config.yml`).

**The deeper problem:** Maxim has 96 distinct `MAXIM_*` env vars + 4 internal config files + 1 external (cloudflared). There is no canonical "what is this instance configured to do?" surface. Two role-detection functions exist in two modules ([runtime/role.py::detect_role](src/maxim/runtime/role.py) and [runtime/leader_mode.py::detect_role](src/maxim/runtime/leader_mode.py)) with different decision orders and different file-extension assumptions. The 2026-04 `role_divergence` event was added specifically to surface this drift, but the underlying duplication remains.

**Companion docs:** rides into 1.0 alongside [leader_ux_profile_management.md](leader_ux_profile_management.md). The two are deliberately independent — profile management is "what models can I run"; config unification is "how is the instance wired up." Together they target the "Maxim as a real leader is annoying to set up" first-touch problem.

---

## Pre-implementation two-lens review fold (2026-06-01)

Before any implementation code lands, the two-lens review prompts at the bottom of this doc were executed in parallel against this plan. Findings folded in commit `<this commit>` BEFORE C1 implementation begins:

**CRITICAL (6 — must change before implementation):**
- **CR-cross (I-3 + IM3):** inline-string API-key mode REJECTED, not deprecated — see "Security: API keys do NOT live in config.json"
- **CR-cross (C-3 + CR2):** C3 role-detector order rewritten with explicit seven-rank table and eight-cell regression matrix — see C3
- **C-1:** empty-string env vars treated as UNSET via `_env_is_set` rule — see "Precedence chain"
- **C-2:** coercion table pinned (truthy/falsy sets + range validation) — see "Coercion table"
- **CR1:** config.json vs profiles.yml `_format_version` divergence justified (CLI-canonical vs hand-edit-canonical) — see "Schema versioning"
- **CR3:** precedence chain logs on convergence AND mismatch, not just mismatch — see "Precedence chain"

**IMPORTANT (9 folded):**
- **I-1:** keyring URI resolved lazily at lane backend construction — see "API key reference resolution timing"
- **I-2:** API-key file deletion handled lazily via `BackendAuthFailed` — see same section
- **I-4:** once-per-startup deprecation INFO mechanism via module-level `_warned_envs` set — see "Precedence chain"
- **I-5:** concurrent `maxim config set` locks BEFORE the read — see C2 "Fold I-5"
- **I-6:** unknown-nested-key handling tied to `_format_version` — see "Validation" + lane-tier-name override
- **IM1:** per-section CC3 path declarations (one path-a case: `LaneTierConfig`) — see C1 IM1 fold table
- **IM2:** canonical writer module `runtime/config_writer.py` + CI grep allow-list — see C2 IM2 fold
- **IM4:** unknown-key forward-compat split tied to `_format_version` — folded into Validation
- **IM5:** peer.yml → config.json migration shaped as Option (iii); 8-reader / 2-writer / 10-test audit folded into C4 IM5 section

**NICE folded:**
- **N1:** `_format_version` declared first in dataclass field order — see C1 dataclass design
- **N2:** doctor surfaces all config-related WARN cases in one place — see C5
- **N-3 + coverage gap:** `MAXIM_PEER_PROBE_KEY`, `MAXIM_SKIP_REMOTE_PROBE`, three `MAXIM_REMOTE_PROBE_*` knobs added to Out of scope at 1.0
- **N-2 (telemetry):** `role_divergence` event kept-for-one-minor with `deprecated: true` data field, removed 1.2 — see C3

**Deferred to opportunistic folds during implementation:**
- N-1 (negative-int rejection) — handled by range validation in the coercion table
- N3 (C4 cut-out option) — recorded as planning-discipline note in the implementation-order section below

---

## Front-gate scope pressure (CLAUDE.md Principle 3)

**Question:** does this need to be its own mechanism, or can it ride on existing infrastructure?

**Existing infrastructure surveyed:**

| Candidate | Why insufficient (or sufficient) |
|---|---|
| Existing 96 `MAXIM_*` env vars | **Insufficient.** Env vars are session-scoped, don't survive tmux/login transitions cleanly, don't persist preferences across `maxim` invocations, and have no override-source tracking. The Mac Mini regression is the canonical failure mode. |
| `~/.config/maxim/peer.yml` | **Cannot ride on it.** peer.yml is purpose-built for peer→leader pairing — not a generic config surface. Extending it to hold role/llm/lane settings would re-create the mesh.yml-parser-extension trap CLAUDE.md explicitly forbids. |
| `~/.config/maxim/mesh.yml` | **Cannot ride on it.** Mesh topology is a different concern; the hand-rolled YAML parser is deliberately limited and frozen. |
| `~/.config/maxim/profiles.yml` (new from leader-UX PR) | **Cannot ride on it.** Profile catalog is a different concern (model definitions, not runtime preferences). Cross-cutting their schemas would couple unrelated lifecycles. |
| Existing CLI flags (`--llm`, `--role`, etc.) | **Already the right surface for transient overrides.** Stays as the top-precedence layer. |
| `maxim.utils.atomic_io.atomic_write_json` | **Rides on it.** The file format is JSON with `_format_version` per CC1 — atomic writes use the existing helper. |
| `models/language/profile_loader.py` precedence pattern (user > builtin with WARNING) | **Rides on it conceptually.** The user-wins-with-loud-logging pattern from leader-UX generalizes to the layered-precedence chain here. |

**Verdict:** new mechanism is required for the schema + loader + override chain. Everything else (atomic write, JSON format, doctor integration shape, CLI verb pattern) rides on existing infrastructure laid down by the leader-UX PR and the prior atomic-io / format-version work.

**Specific reason this earns new mechanism:** there is no existing "instance-level operator preference" config surface. The two-role-detector divergence + the 96-env-var sprawl + the silent-fallback-to-solo failure mode are all symptoms of the missing canonical layer. Adding one new file (`config.json`) + one loader + one CLI verb pair collapses ~15 daily-use env vars + 1 of the 2 role detectors + the lane-routing triplet into a single source-of-truth surface that doctor can introspect.

---

## The current sprawl (motivation evidence)

A complete inventory of what an operator MIGHT need to set today to run a working Mac Mini leader serving qwen2.5-32b through a tunnel:

**Env vars (12+ for the basic case):**
- `MAXIM_ROLE=leader`
- `MAXIM_LLM_ENABLED=1`
- `MAXIM_LLM_PROFILE=qwen2.5-32b-instruct`
- `MAXIM_LLM_N_CTX=16384`
- `MAXIM_LLM_BACKEND=llama_cpp`
- `MAXIM_AUTO_DOWNLOAD_MODELS=1`
- `MAXIM_AUTO_SPAWN_LLM_SERVER=1`
- `MAXIM_AUTO_SPAWN_TUNNEL=1`
- `MAXIM_PROXY_MAX_CONCURRENT=4`
- `MAXIM_PROXY_RATE_LIMIT_RPM=0`
- `MAXIM_DATA_BUDGET_GB=50`
- `MAXIM_LANE_LARGE_REMOTE_URL=` (must be UNSET if this box is the leader — silently breaks if leaked from a previous peer setup)

**Config files (4):**
- `~/.config/maxim/peer.yml` (must NOT exist for leader mode)
- `~/.config/maxim/mesh.yml` (must NOT exist for leader mode unless multi-node)
- `~/.config/maxim/profiles.yml` (optional, for custom GGUFs)
- `~/.cloudflared/config.yml` (MUST be `.yml` not `.yaml` per the role detector's narrow check)

**Two role detectors with different decision orders (see investigation report 2026-06-01):**
- `runtime/role.py::detect_role` — env → mesh.yml → peer.yml → `--llm` flag → default leader
- `runtime/leader_mode.py::detect_role` — env → `/etc/cloudflared/config.yml` → `~/.cloudflared/config.yml` → default solo

The two detectors can disagree (`role_divergence` event was added to surface this, not fix it). The Mac Mini operator hit divergence on day one.

**Total surface area an operator has to mentally model:** ~17 settings + 4 files + 2 detectors + their interaction rules. That's the cost of "set up a leader."

**Goal:** collapse to 1 file + 1 detector + ~15 settings explicitly named in the schema, with override tracking so any deviation is logged.

---

## Design

### Schema (FROZEN at 1.0)

**File location:** `~/.config/maxim/config.json` (declarative-config layer, same dir as `peer.yml` / `mesh.yml` / `profiles.yml`).

**Format:** JSON via `maxim.utils.atomic_io.atomic_write_json` for writes; `json.load` for reads. JSON over YAML because the file is **CLI-canonical** (the `maxim config` verb family is the operator's canonical path; hand-edit is the escape hatch) — JSON parses faster, has stricter validation, and avoids the YAML-anchor-aliases-attack surface for a file Maxim writes on every operator interaction. This sibling-file lifecycle distinction matters for CC1 — see "Schema versioning" below.

```jsonc
{
  "_format_version": "1.0",

  // Operator role — collapses both role-detection functions onto one source.
  // "leader" binds 0.0.0.0 + spawns tunnel + spawns llama-cpp-server.
  // "peer" routes via lane.large.remote_url. "solo" is local-only.
  "role": "leader",

  // LLM core settings — what model, how big a context, do we auto-download.
  "llm": {
    "enabled": true,
    "profile": "qwen2.5-32b-instruct",
    "n_ctx": 16384,
    "backend": "llama_cpp",
    "auto_download": true
  },

  // Per-tier remote routing. For a leader, these are typically all null
  // (self-hosted). For a peer, lane.large.remote_url points at the leader.
  // Multi-leader setups can split tiers across different machines.
  "lanes": {
    "large":  {"remote_url": null, "remote_model": null, "remote_api_key_ref": null},
    "medium": {"remote_url": null, "remote_model": null, "remote_api_key_ref": null},
    "small":  {"remote_url": null, "remote_model": null, "remote_api_key_ref": null}
  },

  // Cloud-LLM fallback config. enabled=false by default; flip to true and
  // set max_lanes + session_budget to allow cloud providers as fallback.
  "cloud": {
    "enabled": false,
    "max_lanes": 0,
    "fallback_model": null,
    "session_budget_usd": 5.0,
    "redaction_policy": "standard"
  },

  // Leader-proxy admission control.
  "proxy": {
    "max_concurrent": 4,
    "rate_limit_rpm": 0
  },

  // Auto-spawn behavior for llama-cpp-server and cloudflared.
  // Setting any to false drops to "manage these myself" mode.
  "auto_spawn": {
    "llm_server": true,
    "tunnel": true,
    "port": 8100,
    "timeout_s": 120
  },

  // Data paths + disk budget.
  "data": {
    "home": null,                  // null → ~/.maxim default
    "budget_gb": null              // null → unlimited
  }
}
```

**Field semantics:**

| Field path | Type | Default | Replaces env var |
|---|---|---|---|
| `role` | `"leader"\|"peer"\|"solo"` | (computed if absent) | `MAXIM_ROLE` |
| `llm.enabled` | bool | `true` | `MAXIM_LLM_ENABLED` |
| `llm.profile` | string (resolves through aliases) | none → CLI must supply | `MAXIM_LLM_PROFILE` |
| `llm.n_ctx` | int (≥256) | `8192` | `MAXIM_LLM_N_CTX` |
| `llm.backend` | `"llama_cpp"\|"pytorch"\|...` | `"llama_cpp"` | `MAXIM_LLM_BACKEND` |
| `llm.auto_download` | bool | `false` | `MAXIM_AUTO_DOWNLOAD_MODELS` |
| `lanes.<tier>.remote_url` | string\|null | null | `MAXIM_LANE_<TIER>_REMOTE_URL` |
| `lanes.<tier>.remote_model` | string\|null | null | `MAXIM_LANE_<TIER>_REMOTE_MODEL` |
| `lanes.<tier>.remote_api_key_ref` | string\|null | null | `MAXIM_LANE_<TIER>_REMOTE_API_KEY` (renamed — see security note) |
| `cloud.enabled` | bool | `false` | `MAXIM_LLM_CLOUD_ENABLED` |
| `cloud.max_lanes` | int (≥0) | `0` | `MAXIM_MAX_CLOUD_LANES` |
| `cloud.fallback_model` | string\|null | null | `MAXIM_CLOUD_FALLBACK_MODEL` |
| `cloud.session_budget_usd` | float (≥0) | `5.0` | `MAXIM_CLOUD_SESSION_BUDGET` |
| `cloud.redaction_policy` | `"standard"\|"relaxed"\|"strict"` | `"standard"` | `MAXIM_LLM_REDACTION_POLICY` |
| `proxy.max_concurrent` | int (≥0) | `4` | `MAXIM_PROXY_MAX_CONCURRENT` |
| `proxy.rate_limit_rpm` | int (≥0) | `0` | `MAXIM_PROXY_RATE_LIMIT_RPM` |
| `auto_spawn.llm_server` | bool | `true` | `MAXIM_AUTO_SPAWN_LLM_SERVER` |
| `auto_spawn.tunnel` | bool | `true` | `MAXIM_AUTO_SPAWN_TUNNEL` |
| `auto_spawn.port` | int | `8100` | `MAXIM_AUTO_SPAWN_PORT` |
| `auto_spawn.timeout_s` | int | `120` | `MAXIM_AUTO_SPAWN_TIMEOUT_S` |
| `data.home` | string\|null | null | `MAXIM_DATA_HOME` |
| `data.budget_gb` | float\|null | null | `MAXIM_DATA_BUDGET_GB` |

**Schema versioning:** `_format_version` at root per CC1. Future schema changes either add OPTIONAL fields with defaults (non-breaking, bump to `1.1`) or introduce REQUIRED fields with a migration step (`2.0`, requires loader migration in same commit).

**Fold CR1: config.json carries `_format_version`; profiles.yml does NOT — why these sibling files diverge.**

`profile_loader.py:48-52` says profiles.yml explicitly does NOT carry `_format_version` because it's "primarily operator-authored." The pre-fold draft described config.json as "primarily machine-written" but then required `_format_version` — an inconsistency the architecture lens flagged.

The defensible answer: **config.json is CLI-canonical** (`maxim config set` / `maxim config edit` is the operator's canonical path; hand-editing the JSON file is the escape hatch), whereas **profiles.yml is hand-edit-canonical** (operators add custom GGUFs by writing YAML directly; the `maxim model add` CLI verb is convenience). CC1 applies to "files Maxim writes" — config.json is in that class because the CLI verbs write it on every operator interaction; profiles.yml is not because the operator writes it primarily. The earlier "primarily machine-written" phrasing at the top of this section has been updated to match this resolution.

Unknown-key handling (next bullet list) is tied to `_format_version` because of this CC1 placement: a `1.0` reader on a `1.1`-written file tolerates unknown additive keys at every level; a `1.0` reader on a `1.0`-declared file rejects unknown keys at every level as typo detection.

**Out of scope at 1.0** (deliberately NOT absorbed):
- Debug / trace flags (`MAXIM_HEARTBEAT`, `MAXIM_LANE_TRACE`, `MAXIM_HTTP_TRACE`, etc.) — these are debug-mode opt-ins, not preferences worth persisting
- Research toggles (`MAXIM_NAC_*`, `MAXIM_EC_TRACE_*`, `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION`, etc.) — short-lived A/B testing flags
- Robot / embodiment config (`MAXIM_ROBOT_NAME`, `MAXIM_REACHY_HOST`) — different lifecycle (per-robot, not per-instance)
- TTS / audio (`MAXIM_TTS_*`, `MAXIM_WHISPER_*`) — feature-specific, low daily-use friction
- Comms (`MAXIM_COMMS_*`) — feature-specific
- **Probe timing knobs** (`MAXIM_SKIP_REMOTE_PROBE`, `MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S`, `MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S`, `MAXIM_REMOTE_PROBE_CACHE_TTL_S`) — env-only tuning surface for one-off network-tuning sessions; not daily-use (coverage-gap fold from Executor review)
- **`MAXIM_PEER_PROBE_KEY`** — process-internal per Plan 3 R2.6's instance-attribute pattern, not operator-facing (N-3 fold)

The absorbed set is ~22 daily-use settings. The remaining ~74 env vars stay as-is (debug-mode + research + feature-specific + probe-tuning + process-internal).

### Precedence chain

**CLI args > env vars > config.json > builtin defaults**

Same shape as `kubeconfig`, `gh`, `npm`, `pyproject.toml`. Loud override logging at every level mismatch — AND every level convergence (see CR3 fold below).

```python
# Module-level state for once-per-startup deprecation logging (I-4 fold).
_warned_envs: set[str] = set()

def _env_is_set(name: str) -> bool:
    """Fold C-1: POSIX shells can `export FOO=` (empty string) and the result
    is still 'present' to os.environ.get. The Mac Mini trigger was exactly
    a leaked-then-emptied env var. Empty-after-strip is treated as unset."""
    raw = os.environ.get(name)
    return raw is not None and raw.strip() != ""

def resolve_setting(field_path: str, cli_value: Any | None) -> tuple[Any, str]:
    """Returns (effective_value, source) where source is one of:
       'cli' | 'env' | 'config' | 'default'"""
    if cli_value is not None:
        return cli_value, "cli"
    env_name = _env_var_for(field_path)
    config_value = _read_from_config(field_path)
    if _env_is_set(env_name):
        env_value = os.environ[env_name]
        # Fold CR3: log on EVERY shadow, not just mismatch. The Mac Mini
        # failure mode was "two sources of truth set, operator doesn't know
        # which wins" — logging only on mismatch hides convergence-by-accident
        # until the operator edits one side later.
        if config_value is not None:
            if config_value == env_value:
                logger.info(
                    "Config: %s has source=env AND config.json sets the same "
                    "value ('%s'). config.json is ignored until env var is unset.",
                    field_path, env_value
                )
            else:
                logger.warning(
                    "Config override: %s='%s' (env) shadows '%s' (config.json)",
                    field_path, env_value, config_value
                )
        # Once-per-startup deprecation INFO for absorbed env vars (I-4 fold)
        if env_name in _ABSORBED_ENV_VARS and env_name not in _warned_envs:
            _warned_envs.add(env_name)
            logger.info(
                "config: %s is set. Consider migrating to config.json via "
                "`maxim config set %s <value>` (env vars deprecated in 1.1).",
                env_name, field_path
            )
        return _coerce(env_value, field_path), "env"
    if config_value is not None:
        return config_value, "config"
    return _builtin_default(field_path), "default"
```

**Override logging is load-bearing.** Every effective field's source is logged at startup at INFO level (one summary line per field that resolved). Shadowing AND convergence at multiple layers are both logged so the operator can spot drift in a single grep.

**Once-per-startup deprecation log mechanism (I-4 fold).** `_warned_envs: set[str]` is a module-level set in `config_loader.py`. `_ABSORBED_ENV_VARS` is a module-level frozenset of every env var the schema absorbs. The batch fires from any code path that calls `resolve_setting` — but each env var name lands in the set after the first call, so subsequent calls (any per-request lane resolution, etc.) are silent. Regression guard: `test_deprecation_info_fires_once_across_100_resolve_calls`.

### Coercion table (C-2 fold)

The original draft hand-waved `_coerce(env_value, field_path)`. Both reviewers flagged this as a silent-semantic-drift risk across the C4 migration window (`lane_backends.py:1410` uses a different truthy set than `lane_backends.py:2024`). The canonical coercion is pinned here:

| Source type | Truthy set | Falsy set | Out-of-set behavior |
|---|---|---|---|
| `bool` | `{"1", "true", "yes", "on"}` (case-insensitive, post-strip) | `{"0", "false", "no", "off"}` (case-insensitive, post-strip) — empty-string is rejected by `_env_is_set` above | `ConfigurationError("expected bool-like, got '<value>'")` |
| `int` | parsed via `int(value.strip())` | — | `ConfigurationError`; range-validated post-parse (e.g., `n_ctx ≥ 256`, `max_concurrent ≥ 0`, `port` in 1..65535); out-of-range → `ConfigurationError` with the valid range |
| `float` | parsed via `float(value.strip())` | — | `ConfigurationError`; range-validated where applicable (e.g., `cloud.session_budget_usd ≥ 0`) |
| `Literal[...]` enum | exact match against the enum set, post-strip + lowercase | — | `ConfigurationError("expected one of {...}, got '<value>'")` |
| `string` | post-strip; empty strings rejected per `_env_is_set` | — | — |

The truthy / falsy sets match the existing `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION` parser and the leader-UX PR's profile-loader parser (the canonical references). C4's migration audit verifies each absorbed env var's existing parser is equivalent — any mismatch triggers a regression test that pins the new behavior before the env var is absorbed.

### Security: API keys do NOT live in config.json

Per the CLAUDE.md mesh.yml two-layer-split invariant: declarative config (`peer.yml`, `mesh.yml`, `profiles.yml`, `config.json`) uses plain `atomic_write_text` / `atomic_write_json`. Credentials use `atomic_write_secret` with mode 0600.

`lanes.<tier>.remote_api_key_ref` holds a **reference**, not the key itself. Exactly **two** resolution modes are supported:
1. **File path** (value starts with `/` or `~`) → mode-0600 file read at the path
2. **Keyring URI** (value matches `keyring:<service>:<account>`) → resolved via system keychain (macOS Keychain, Linux Secret Service)

**Fold (review cross-confirmation I-3 + IM3): inline-plain-string mode is REJECTED, not deprecated.**

The original draft had a third mode — "if value is a plain string, treat as inline key + log WARNING." Both review lenses independently flagged this as a footgun: `maxim config set lanes.large.remote_api_key_ref sk-abc123` would cheerfully write the literal key into mode-0644 `config.json`, and the WARNING is invisible to operators following the obvious copy-paste path. The mesh.yml two-layer-split invariant explicitly says credentials live behind `atomic_write_secret` mode-0600, not in mode-0644 declarative config.

Therefore: any string value passed to `lanes.<tier>.remote_api_key_ref` that does NOT start with `/`, `~`, or `keyring:` raises `ConfigurationError` at config-load time, with a fix hint pointing at:
- `maxim config set lanes.<tier>.remote_api_key_ref ~/.config/maxim/api_key` (file path), OR
- `maxim config set lanes.<tier>.remote_api_key_ref keyring:maxim:<account>` (keyring URI)

CI grep regression guard (added to `.github/workflows/test.yml`): `grep -E '"remote_api_key_ref": "(sk-|gsk_|AKIA|xoxb-)' tests/fixtures/` must return zero matches. The grep mirrors the existing `urllib.request.urlopen` allow-list discipline.

**Migration from legacy `MAXIM_LANE_<TIER>_REMOTE_API_KEY` env vars:** the C6 migration shim writes the env-var value to `~/.config/maxim/api_key.<tier>` (mode 0600 via `atomic_write_secret`) and writes the *path* to `config.json::lanes.<tier>.remote_api_key_ref`. Inline migration is never the path.

### API key reference resolution timing (I-2 fold)

API key references are resolved **lazily at lane backend construction**, not at config-load time. The motivating cases:
- Path-mode file exists at config load but is deleted before first inference — resolved-at-load would hold a stale value in memory; resolved-at-request raises `BackendAuthFailed` from `_MaximPeerBackend.for_url` consistent with the Plan 2 R2b typed hierarchy
- Keyring extra not installed (`pip install keyring` not done) — resolved-at-load blocks `maxim doctor` and `maxim config get` (the exact recovery verbs the operator will run), so resolution must be deferred. `maxim config get lanes.large.remote_api_key_ref` prints the literal `keyring:<service>:<account>` URI with a `[unresolved: keyring not installed]` annotation rather than raising

`maxim doctor` runs eager probes for each `lanes.*.remote_api_key_ref`:
- File path → reports `[file missing]` / `[mode != 0600]` / `[ok]`
- Keyring URI → reports `[keyring not installed]` / `[entry missing]` / `[ok]`
- Inline string (legacy from pre-migration env var) → reports `[INLINE — MIGRATE]` with a fix hint pointing at the migration shim

Default `~/.config/maxim/api_key` (the existing leader API key file) is read by reference at `lanes.large.remote_api_key_ref: "~/.config/maxim/api_key"` for the canonical case.

### What `config.json` does NOT replace

- **`peer.yml`** — kept for the `peer connect` one-shot pairing flow. The CLI verb still writes it; role detection still reads its existence. `config.json` may carry `lanes.large.remote_url` pointing at the same leader, but the two files coexist (peer.yml is the operator-facing "you connected to X"; config.json is the runtime resolution).
- **`mesh.yml`** — kept for multi-node topology. `config.json` is per-instance; mesh.yml is per-cluster.
- **`profiles.yml`** — kept for custom profile catalog. `config.json` only references profiles by name.
- **`~/.cloudflared/config.yml`** — that's cloudflared's own config, not Maxim's.

Why coexist instead of collapse: each of the kept files has a specific operator interaction surface (`peer connect`, `peer add-node`, `model add`) and a specific lifecycle. Collapsing them would re-create the kind of overloading-one-file-with-multiple-concerns trap CLAUDE.md flags repeatedly.

---

## The six stages, folded into one PR

| Stage | What lands | LOC est. | Frozen at 1.0? |
|---|---|---|---|
| **C1** | `maxim.config_loader` module + JSON schema validation + precedence-chain resolver | ~250 LOC + tests | **Yes — schema** |
| **C2** | `maxim config {get,set,list,edit,path}` CLI verbs | ~200 LOC + tests | Partially — verb names + flag names |
| **C3** | Role-detector unification — `runtime/role.py` reads from `config_loader` first, falls through to the existing decision tree as compat fallback. `runtime/leader_mode.py::detect_role` is replaced with a thin wrapper that calls `runtime/role.py`. Closes the two-detector divergence. | ~80 LOC delta | No — internal refactor |
| **C4** | Per-tier remote routing migrates to `lanes.<tier>.*`. Lane backend resolution reads from `config_loader`. Env vars still work (deprecation warning fires). | ~120 LOC delta | No — non-breaking migration |
| **C5** | `maxim doctor` adds a "Resolved config" section showing every effective field's value + source. Replaces piecemeal env-var checks. | ~80 LOC | No — purely additive surface |
| **C6** | Deprecation warnings: when an absorbed env var is set, log INFO at startup pointing at the equivalent `config.json` setting. Docs: new `configuration.md` consolidates env-var table → config schema. | ~50 LOC + ~200 doc lines | No — additive |

Total: ~780 LOC src + ~400 LOC tests + ~250 doc lines. ~2-3 days plus review. Larger than the leader-UX PR by ~50%; comparable to a Wave 1 builder-unification PR.

---

## C1 — config.json loader + schema

**Goal:** at module import (or lazy-singleton on first access), load `~/.config/maxim/config.json`, validate, expose typed access.

**Lives at:** new module `src/maxim/runtime/config_loader.py`.

**Key API:**

```python
@dataclass(frozen=True)
class MaximConfig:
    """SHAPE-FROZEN at 1.0 (CC3). See config_unification.md schema table.
    Adding optional fields with defaults is non-breaking; adding required
    fields requires _format_version 2.0 + migration."""
    _format_version: str = "1.0"  # N1 fold: declared first per underscore-sort-first convention
    role: Literal["leader", "peer", "solo"] | None = None
    llm: LLMConfigSection = field(default_factory=LLMConfigSection)
    lanes: LanesConfigSection = field(default_factory=LanesConfigSection)
    cloud: CloudConfigSection = field(default_factory=CloudConfigSection)
    proxy: ProxyConfigSection = field(default_factory=ProxyConfigSection)
    auto_spawn: AutoSpawnConfigSection = field(default_factory=AutoSpawnConfigSection)
    data: DataConfigSection = field(default_factory=DataConfigSection)

def load_config(path: Path | None = None) -> MaximConfig:
    """Read config.json, validate against schema, return typed dataclass.
    Returns defaults-only MaximConfig if file is missing."""

def resolve_setting(field_path: str, cli_value: Any | None = None) -> tuple[Any, str]:
    """The precedence chain. Returns (effective_value, source)."""
```

**Fold IM1: per-section CC3 path declarations** — each section dataclass MUST declare path (a) escape-hatch or path (b) shape-frozen in its class docstring. The split below is reasoned per-section:

| Section type | CC3 path | Why |
|---|---|---|
| `MaximConfig` (root) | **(b) shape-frozen** | Top-level shape is the schema contract. Section additions go inside section types, not at root. |
| `LLMConfigSection` | **(b) shape-frozen** | `backend` is a frozen enum; adding fields here is rare and review-gated. |
| `LanesConfigSection` | **(b) shape-frozen** | Keyed by the frozen tier-name set (`large`, `medium`, `small` per the lane-tier-names invariant). Adding a new tier post-1.0 is a major-version bump. |
| `LaneTierConfig` (per-tier inner type) | **(a) escape-hatch** with `extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)` | The one section where (a) is genuinely right — `remote_url`/`remote_model`/`remote_api_key_ref` will likely grow forward (probable additions: `remote_health_path`, `remote_timeout_s`, `remote_routing_weight`). The `extra` dict's values must be JSON-serializable per CC3 (str/int/float/bool/None + nested list/dict only). |
| `CloudConfigSection` | **(b) shape-frozen** | `redaction_policy` is a frozen enum; cloud-LLM contract is tightly coupled to the redaction layer. |
| `ProxyConfigSection` | **(b) shape-frozen** | Two-field admission-control surface; additions are review-gated. |
| `AutoSpawnConfigSection` | **(b) shape-frozen** | Four-field operator-explicit surface; additions are review-gated. |
| `DataConfigSection` | **(b) shape-frozen** | Two-field path/budget surface. |

Each dataclass class docstring carries the `SHAPE-FROZEN at 1.0 (CC3)` marker (path b) or names the `extra` field's purpose (path a). Regression-guard test: `test_each_config_section_declares_cc3_path` greps for the marker in each docstring.

**Validation** (folded I-6 + IM4 — unknown-key handling tied to `_format_version`):

- **`_format_version` matches the loader's known major.minor** (same-version case): unknown keys at EITHER top level OR nested inside typed sections → `ConfigurationError` with the key listed. This is typo-detection: `lanes.lerge.remote_url` must fail loudly, not silently fall through (the exact silent-mis-configure case this PR was meant to close).
- **`_format_version` is a future minor within the same major** (e.g., loader knows `1.0`, file says `1.1`): unknown keys at either level log WARNING and continue. This is the forward-compat case — a newer Maxim in a heterogeneous mesh can write 1.1 fields a 1.0 reader tolerates.
- **`_format_version` is a future major** (e.g., loader knows `1.0`, file says `2.0`): `ConfigurationError` — major bump is breaking per CC1.
- **Unknown nested tier name** inside `lanes.<tier>.*` (e.g., `lanes.lerge`) is treated as an unknown nested key under a typed section. Tier names are FROZEN at 1.0 per the existing `[engineering] Lane tier names` invariant — `large`, `medium`, `small` are the only valid keys, and `ConfigurationError` fires regardless of `_format_version`.
- Type mismatch → `ConfigurationError` naming field + expected type
- Invalid enum value → `ConfigurationError` listing valid values
- `role: "client"` (the old leader_mode.py term) → auto-coerce to `"peer"` with WARNING (compat)
- Invalid JSON syntax → `ConfigurationError` at startup, blocks `maxim` from running. `maxim doctor` surfaces the parse error before next inference attempt.

**Regression guards:**
- `tests/unit/test_config_loader.py::TestMinimumValidConfig` — empty file → all defaults
- `TestPrecedenceChain` — CLI > env > config > default for each absorbed field, parametrized
- `TestOverrideLogging` — every shadow logs WARNING with both values
- `TestSchemaErrors` — every ConfigurationError path

---

## C2 — `maxim config` CLI verbs

**CLI surface:**

```bash
# Inspect
maxim config get                    # full config + sources
maxim config get role               # one field with source marker
maxim config get llm.profile        # nested via dot path
maxim config path                   # print resolved config.json path
maxim config list                   # human-readable summary, all effective fields

# Mutate
maxim config set role leader
maxim config set llm.profile qwen2.5-32b-instruct
maxim config set lanes.large.remote_url https://big-mac.example.com/v1
maxim config set lanes.large.remote_api_key_ref ~/.config/maxim/api_key

# Edit (opens $EDITOR)
maxim config edit
```

**Round-trip:** writes via `atomic_write_json` preserve key ordering (Python 3.7+ dict ordering is guaranteed). Comments NOT preserved (JSON doesn't have them — operators wanting comments use a separate `notes.md` next to it).

**Exit codes:** 0 success, 1 environmental failure (write permission, etc.), 2 operator error (unknown field, bad value type).

### Fold IM2: canonical writer module + CI grep allow-list

`config.json` is a declarative-config file that the `maxim config set` verb writes from runtime — exactly the "operator-explicit one-shot setup verb" exception the mesh.yml two-layer-split invariant allows. To match the discipline `mesh_setup.py` ships with, the writer surface is enforced by a CI grep allow-list:

- **Canonical writer module:** `src/maxim/runtime/config_writer.py` exposing `write_config(config: MaximConfig) -> Path`. Uses `atomic_write_json` (declarative config, not secret), holds a `filelock.FileLock` around the read-modify-write cycle.
- **CI grep enforcement** in `.github/workflows/test.yml`: `grep -rn "atomic_write_json.*config\.json\|config_loader.*\.dump\|json\.dump.*config" src/maxim/` allow-lists only `config_writer.py` + its test file. Any new caller is a CI failure with a migration hint pointing at `config_writer.write_config`.
- New verbs that need to mutate config.json (e.g., `peer connect` after the IM5 migration) call `write_config`, never inline.

This mirrors the `write_mesh_config` allow-list pattern enforced by `mesh_setup.py`.

### Fold I-5: concurrent-set lock discipline

`maxim config set` from two tmux panes is a real operator workflow. The `filelock.FileLock` around the RMW cycle is necessary but NOT sufficient — a stale in-memory `MaximConfig` from before the lock was acquired would clobber the other pane's just-written field.

Required pattern inside `config_writer.write_config(new_value)`:
1. Acquire `FileLock(config_path + ".lock")`
2. Re-read config.json from disk INSIDE the lock (no caching across the lock boundary)
3. Apply the field delta to the freshly-read dataclass
4. Atomic-write via `atomic_write_json`
5. Release lock

The `tests/integration/test_drain_state_concurrent.py` pattern is the regression-guard template — adapt as `tests/integration/test_config_writer_concurrent.py::test_concurrent_set_different_fields_both_persist`.

**Regression guards:**
- `tests/unit/test_config_cli.py::TestRoundTrip::test_set_then_get`
- `TestUnknownFieldRefused` for both get and set
- `TestSetCoercion` — `set proxy.max_concurrent 4` writes int not string

---

## C3 — Role-detector unification (closes the two-detector divergence)

**The current state:** two `detect_role` functions in two modules with different decision orders + different file-extension assumptions. The 2026-06-01 Mac Mini regression hit divergence on day one.

### Fold CR2 + C-3: full decision order with explicit precedence cells

Both reviewers flagged the original draft's ordering as under-specified for the actual Mac Mini failure modes (stale peer.yml + cloudflared present; `--llm` flag + cloudflared present; etc.). The original draft put peer.yml ABOVE cloudflared, which means a stale peer.yml on a now-leader machine STILL silently steers role detection to peer. That preserves the very bug this PR is meant to fix.

The unified decision order (first match wins) — and the Mac Mini cells that each rank pins:

| Rank | Signal | Returns | Mac Mini cell pinned |
|---|---|---|---|
| 1 | `MAXIM_ROLE` env var ∈ `{leader, peer, solo}` (after strip + lowercase) | env value | explicit operator override always wins |
| 2 | `config.json::role` ∈ `{leader, peer, solo}` | config value | **NEW** — operator's persisted intent; bypasses peer.yml legacy signal |
| 3 | `mesh.yml` present + parseable | `peer` | multi-node setup; operator-explicit |
| 4 | `~/.cloudflared/config.yml` OR `~/.cloudflared/config.yaml` present | `leader` | **MOVED UP from leader_mode.py + extension widened** — system-level tunnel provisioning is a strong "this is a leader" signal; promoted above peer.yml so a stale peer.yml on a now-leader doesn't override it. **The `.yaml` extension widening fixes Mac Mini Trigger #2 as a side effect.** |
| 5 | `peer.yml` present (legacy) | `peer` | **DEMOTED below cloudflared** — preserves zero-config peer flow but no longer overrides a real leader signal. Auto-migration shim (see IM5 fold in C4) auto-writes `config.json::role=peer` from peer.yml on first startup, so rank 2 takes over on the next run. |
| 6 | `--llm <local-profile>` CLI flag + no `peer.yml` + no `mesh.yml` + no cloudflared | `solo` | local-only inference path |
| 7 | (default) | `leader` | nothing else matched |

**Cells that the regression-guard tests pin** (each row is one test in `tests/unit/test_role_unification.py`):

- `test_config_json_role_wins_over_env_var` — explicit env always trumps config (rank 1)
- `test_config_json_role_wins_over_peer_yml_legacy` — fixes Trigger #3 in the canonical form
- `test_stale_peer_yml_plus_cloudflared_yml_resolves_to_leader_not_peer` — the actual Mac Mini bug (rank 4 above rank 5)
- `test_stale_peer_yml_plus_cloudflared_yaml_resolves_to_leader_not_peer` — extension-widening (Trigger #2)
- `test_cli_flag_solo_vs_cloudflared_leader_precedence` — `--llm` + cloudflared present → leader (rank 4 above rank 6), because cloudflared is a system-level "this box is provisioned as a tunneled leader" signal that overrides the local-only `--llm` hint
- `test_cli_flag_solo_with_no_peer_signals` — `--llm` alone → solo (rank 6, no higher-rank signals)
- `test_no_signals_at_all_defaults_to_leader` — rank 7 fallback
- `test_yaml_and_yml_extensions_both_accepted` — extension widening unit test

**The fix:**
1. `runtime/role.py::detect_role` becomes the **single source of truth** with the seven-rank order above.
2. `runtime/leader_mode.py::detect_role` is replaced with a thin wrapper that calls `runtime/role.py::detect_role` and translates `leader|peer|solo` → `RoleDecision(role, bind_host)`. The cloudflared-config-exists branch is REMOVED from leader_mode.py (moved into role.py rank 4 with extension widening).
3. The legacy `RoleName = Literal["leader", "peer", "client", "solo"]` in leader_mode.py drops `"client"` (auto-coerced to `"peer"` per the env-var read path's WARNING in the loader's validation rules above). One minor version of compat for any external consumers.
4. `role_divergence` event is REMOVED. Telemetry-keep-for-one-minor compat: the event continues firing with `data={"reason": "single detector", "deprecated": true}` through 1.1 to avoid silently dropping a column from external dashboards (N-2 fold). Removed entirely in 1.2.
5. `role_detected` event gains a `config_json_present` field for telemetry on adoption.

**Migration:** non-breaking for the rank 1, 3, 6, 7 cases. Rank 4 widens (cloudflared `.yaml` now accepted). Rank 5 demotes (peer.yml still works as a peer signal but no longer overrides cloudflared). Setups with stale peer.yml on a now-leader machine get the auto-migration shim in C4.

**Regression guards:** the eight-cell test list above. The wrapper consistency test `test_leader_mode_detect_role_returns_same_as_role_py` runs the full eight-cell matrix through both surfaces and asserts identical results.

---

## C4 — Per-tier remote routing migrates to lanes.* in config.json

**Today:** `MAXIM_LANE_LARGE_REMOTE_URL` + `_API_KEY` + `_MODEL` for each of large/medium/small = 9 env vars. The Mac Mini regression had a stale `MAXIM_LANE_LARGE_REMOTE_URL` from when this box was a peer, silently re-routing the leader's own large lane.

**The fix:**
- `lanes.<tier>.remote_url/remote_model/remote_api_key_ref` in `config.json`
- `lane_backends.py` resolves via `resolve_setting("lanes.large.remote_url", cli_value=None)` → respects precedence chain
- Env vars still work (precedence chain handles them) but emit DEPRECATION INFO at startup when set (the once-per-startup mechanism specified in the precedence-chain section)

**Self-hosted classification preservation note** (from peer.yml audit, 2026-06-01): when `lanes.<tier>.remote_url` resolves via config.json OR the peer.yml compat-read path (next subsection), the lane backend MUST auto-classify it as self-hosted infrastructure — i.e., the cloud-lane gate does NOT fire and redaction policy is not enforced. The legacy `apply_peer_config_to_env` set `MAXIM_MAX_CLOUD_LANES=1` as a side effect to signal exactly this; the post-C4 path replaces that side effect with a direct classification at lane resolution time (`lane_backends.classify_self_hosted_lane`). Cloud-provider opt-in stays explicit via `config.json::cloud.enabled = true` + `cloud.max_lanes > 0`. Regression guard: `test_config_json_lanes_large_does_not_trigger_cloud_gate`.

### Fold IM5: peer.yml → config.json migration (Option iii, full consumer audit)

The 2026-06-01 audit identified **8 readers** of peer.yml in `src/`, **2 writers** (both operator-explicit CLI verbs), and **10 test files** exercising the surface. The migration shape was selected after the audit confirmed no hidden consumers.

**The semantic change that actually fixes Mac Mini Trigger #3:** `runtime/role.py::_peer_yml_exists` stops contributing to role detection (handled in C3 — peer.yml demoted to rank 5 below cloudflared). peer.yml-implies-peer becomes purely a *lane-routing* compat signal, no longer a *role* signal.

**Per-consumer migration table:**

| # | Consumer | File:line | Migration |
|---|---|---|---|
| 1 | Startup peer.yml → env bridge | `runtime/lane_backends.py:1063-1080` | `apply_peer_config_to_env(peer_cfg)` becomes a read-fallback path INSIDE `resolve_setting` for the three `lanes.large.*` fields. The env-mutation side effect is removed — `resolve_setting` returns values directly. Self-hosted classification preserved per the note above. `_has_peer_config` flag is replaced by `resolve_setting` source attribution. |
| 2 | Role detection | `runtime/role.py:67-73` (`_peer_yml_exists`) | Demoted to C3 rank 5 (below cloudflared). The "peer.yml present implies role=peer" semantics survive in isolation but no longer override cloudflared/config.json signals. |
| 3 | mesh synthesis | `peer/mesh_config.py::synthesize_from_peer_config` (lines 565-580) | No semantic change. Still reads peer.yml directly when mesh.yml absent. Operator-explicit `maxim peer init-mesh` flow unchanged. |
| 4 | Doctor checks | `doctor/checks.py:1425, 1685, 1788, 1832, 2313` (5 sites) | Each `read_peer_config()` call routes through a new shared helper `_resolve_leader_url_with_source() -> tuple[url, key, source]` that prefers config.json::lanes.large.*, falls through to peer.yml with source marker `"peer.yml [deprecated]"`. C5's "Resolved Config" section surfaces the source. |
| 5 | Doctor retry-loop display | `doctor/cli.py:235, 427` (2 sites) | Same shared helper as #4. |
| 6 | Roy preflight | `simulation/roy_runner.py:340-352` | Replaces direct `read_peer_config()` with `resolve_setting("lanes.large.remote_url", cli_value=None)`. The `source` annotation gains a `"config.json"` value alongside `"env"` and `"peer.yml"`. |
| 7 | `maxim peer connect` writer | `peer/cli.py:265-266` | **Writes BOTH** `config.json::lanes.large.*` (canonical) AND peer.yml (compat) during the 1.x deprecation window. peer.yml gets a header comment `# DEPRECATED — config.json::lanes.large.* is canonical as of 1.0. peer.yml will be retired in 2.0.` |
| 8 | `maxim peer show` / `key` / `key set` / `forget` | `peer/cli.py:284-307, 313-374` | `show` reads config.json first, falls through to peer.yml. `key set` updates both files. `forget` deletes both. `key` (read) tries config.json::lanes.large.remote_api_key_ref first, falls through. |
| 9 | mesh-setup compat | `peer/mesh_setup.py:213` (`maxim peer init-mesh`) | No change. Existing `read_peer_config()` call continues to work as a compat reader through 1.x. |

**Auto-migration shim** runs once at first `load_config()` invocation when:
- `config.json` is absent AND
- `peer.yml` is present

Action: write a minimal `config.json` with `role: "peer"` + `lanes.large.remote_url/remote_api_key_ref/remote_model` populated from peer.yml fields, atomic-write the new file, log INFO `"config: auto-migrated peer.yml → config.json (peer.yml preserved for 1.x compat, retired in 2.0)"`. peer.yml is NOT deleted. Subsequent startups read config.json directly and skip the shim. The shim is idempotent — if config.json exists, it never fires.

**One-shot deprecation INFO log per peer.yml read** (via the once-per-startup mechanism): every consumer that falls through to peer.yml reads emits `"config: lanes.large.* resolved from peer.yml (deprecated — run `maxim peer connect <url>` to migrate to config.json)"` exactly once per startup. The log fires from the shared helper, not from each call site.

**1.x → 2.0 retirement path:**
- 1.0: dual-write + compat-read with deprecation INFO
- 1.x (minor versions): peer.yml read-only-compat; `maxim peer connect` still writes both
- 2.0: peer.yml read removed entirely. File left in place (we don't delete operator data); `maxim doctor` flags it as `[orphaned — safe to delete]`.

**Test impact:** the 10 test files exercising peer.yml gain parallel coverage for the config.json path. New autouse fixture `_isolate_config_json_env` (template: `tests/conftest.py::_isolate_maxim_llm_profile_env` per the existing CLAUDE.md pattern). Deprecation INFO log captured via `caplog` fixture. Migration shim's idempotency pinned by `test_load_config_runs_migration_shim_only_once`.

**Regression guards:**
- `tests/unit/test_lane_routing_via_config.py::test_config_lanes_drive_backend_resolution`
- `test_env_var_still_wins_with_deprecation_warning`
- `test_peer_connect_writes_to_both_files` (during the compat window)
- `test_peer_yml_demoted_below_cloudflared_in_role_detection` (the Mac Mini Trigger #3 fix)
- `test_auto_migration_shim_writes_config_from_peer_yml`
- `test_auto_migration_shim_idempotent_when_config_exists`
- `test_peer_yml_deprecation_info_fires_once_per_startup`
- `test_config_json_lanes_large_does_not_trigger_cloud_gate` (self-hosted classification preservation)

---

## C5 — `maxim doctor` "Resolved config" section

**New section in doctor output:**

```
━━━ Resolved Config ━━━
  ✓ role: leader                          [source=config.json]
  ✓ llm.profile: qwen2.5-32b-instruct     [source=env, shadows config=qwen2.5-14b]
  ✓ llm.n_ctx: 16384                      [source=cli]
  ✓ llm.auto_download: true               [source=config.json]
  ✓ lanes.large.remote_url: <self-hosted> [source=default]
  ✓ lanes.large.remote_api_key_ref:       [path=~/.config/maxim/api_key, mode=0600, ok]
  ✓ proxy.max_concurrent: 4               [source=default]
  ⚠ MAXIM_LANE_LARGE_REMOTE_URL is set in env (http://127.0.0.1:8100/v1)
    but config.json::lanes.large.remote_url is null. The env var wins.
    → If this box is a leader, unset the env var. Run `maxim config get` to verify.
  ⚠ peer.yml present (deprecated). Migrate via `maxim peer connect <url>` which
    now writes config.json. peer.yml will be retired in 2.0.
```

**Fold N2: doctor surfaces all config-related WARN cases in one place.** The "Resolved Config" section is the single answer to "what does this instance think it's configured as?" — collapsing what previously required cross-referencing 96 env vars, 4 files, and 2 role detectors. Cases that fire as WARN rows in the section:

- env var shadowing config.json value (mismatch)
- env var and config.json agreeing on a value (convergence — CR3 fold)
- legacy `role: "client"` coerced to `peer`
- `lanes.*.remote_api_key_ref` path file missing
- `lanes.*.remote_api_key_ref` path file mode != 0600
- `lanes.*.remote_api_key_ref` is an inline-string (pre-migration legacy from `MAXIM_LANE_*_REMOTE_API_KEY` env var) — marked `[INLINE — MIGRATE]`
- `lanes.*.remote_api_key_ref` is a `keyring:` URI but keyring not installed (`[unresolved]`)
- peer.yml present (compat-read deprecation INFO)
- `_format_version > loader_known` — forward-compat warning summary

**Regression guards:**
- `tests/unit/test_doctor.py::TestResolvedConfigSection::test_shows_every_absorbed_field`
- `test_override_chain_visible_for_shadow_and_convergence`
- `test_inline_api_key_flagged_as_migrate`
- `test_peer_yml_present_flagged_as_deprecated`

---

## C6 — Deprecation warnings + docs

**At startup, if an absorbed env var is set,** emit an INFO log:

```
INFO | config: MAXIM_LANE_LARGE_REMOTE_URL is set. Consider moving to config.json
      via `maxim config set lanes.large.remote_url <value>` (env vars will be
      deprecated in 1.1).
```

INFO not WARN because env vars still work; this is gentle guidance, not a fault.

**Docs:**
- New `docs/user/configuration.md` — primary operator config doc, table of every absorbed field, precedence chain explanation, migration guide
- `docs/user/llm-setup.md` — cross-link, remove the env-var sprawl table (move to configuration.md)
- `docs/user/peer-setup.md` — show `maxim config set lanes.large.remote_url` as the modern path, env-var path moves to "alternative"
- `docs/user/getting-started.md` — `maxim config` mentioned as the first-touch operator surface

---

## Edge cases & non-obvious design choices

| Case | Decision | Rationale |
|---|---|---|
| `config.json` is missing | Loader returns defaults-only `MaximConfig`. Silent no-op. | The empty case is the common case; loud warning would be noise. |
| `config.json` exists but is empty `{}` | Same — defaults-only. | Equivalent to missing. |
| `config.json` has invalid JSON syntax | `ConfigurationError` at startup, blocks `maxim` from running. C5's "Resolved Config" doctor section surfaces the parse error and the offending line. | Same shape as profiles.yml — broken config should fail fast, not silently mis-route. |
| `config.json` has a future schema field (`_format_version: 1.1`) | Loader warns once, tolerates unknown additive keys at every level, continues. | Forward-compat — newer Maxim installs in a heterogeneous mesh can write 1.1; older readers tolerate. |
| `config.json` has `_format_version: 2.0` | `ConfigurationError` — major version bump is breaking. | Per CC1; loader's job to refuse incompatible majors. |
| Env var sets `MAXIM_ROLE=client` (old name) | Auto-coerce to `peer` with WARNING. | leader_mode.py used the old name; the WARNING makes the deprecation visible. |
| Env var set to empty string (`export MAXIM_LANE_LARGE_REMOTE_URL=`) | Treated as UNSET per the `_env_is_set` rule (C-1 fold). Precedence falls through to config.json. | POSIX shells can `export FOO=` and the result is still "present" to `os.environ.get`. The Mac Mini trigger was a leaked-then-emptied env var. |
| Both env var and config.json set the same field to the SAME value | INFO log "field X has source=env AND config.json sets the same value" (CR3 fold). | Logs convergence, not just divergence — operator's two-sources-of-truth confusion is the bug class to surface. |
| Two API key references point at the same file | Allowed. | Multiple lanes can share a key (common in the same-leader case). |
| `lanes.<tier>.remote_api_key_ref` is a string that LOOKS like a key (not a path or keyring URI) | `ConfigurationError` at load time (cross-confirmed I-3/IM3 fold). Fix hint points at file-path or keyring-URI form. | Inline-string mode was rejected per the cross-confirmed review fold. Migration shim writes legacy `MAXIM_LANE_<TIER>_REMOTE_API_KEY` env vars to `~/.config/maxim/api_key.<tier>` mode-0600 and references by path. |
| `lanes.large.remote_api_key_ref` path file exists at config load but is deleted before first inference | `BackendAuthFailed` raised lazily at lane backend construction (I-2 fold). | Lazy resolution at request time avoids stale-in-memory holding of the secret. Doctor reports `[file missing]` for path-mode refs as an eager probe. |
| `lanes.large.remote_api_key_ref` is a `keyring:` URI but keyring package not installed | Lazy resolution: `maxim config get` prints `[unresolved: keyring not installed]` annotation, does NOT raise (I-1 fold). Raises `BackendAuthFailed` at lane backend construction if actually used. | Eager raise at config-load would block `maxim doctor` / `maxim config get` — the exact recovery verbs the operator runs when debugging. |
| User passes `--config /custom/path/config.json` | CLI flag overrides the default path. Round-trip through `maxim config set --config /custom/path role leader` writes to the custom path. | Useful for testing + multi-instance setups. |
| `maxim config set` run concurrently from two tmux panes | `filelock.FileLock` acquired BEFORE the read, held through the write (I-5 fold). Re-read happens inside the lock — no caching across the boundary. | Mirrors the `peer.yml` / drain state pattern. Lock-acquire-after-read would let the stale in-memory dataclass clobber the other pane's write. |
| Stale peer.yml on a now-leader machine | Cloudflared rank 4 in C3 overrides peer.yml rank 5 → role resolves to leader. Auto-migration shim writes `config.json::role=peer` ONLY if peer.yml is present AND cloudflared is absent — preserving the legitimate zero-config peer flow. | Fixes Mac Mini Trigger #3. |

---

## Regression guards summary

Every test lives at `tests/unit/test_config_*.py` (or the cross-referenced files named in the per-fold sections above):

**Schema + loader (C1):**
- Minimum valid config → all defaults
- Each section dataclass declares its CC3 path (IM1 fold)
- Empty string env vars treated as unset (`_env_is_set` rule — C-1 fold)
- Coercion table: bool / int / float / Literal / string parametrized rejection cases (C-2 fold)
- Schema errors: invalid JSON syntax, bad enum, wrong type
- Unknown-key handling tied to `_format_version`: same-version → reject typo, future-minor → tolerate (IM4 fold)
- Unknown nested tier name in `lanes.<typo>` always rejected (lane-tier names FROZEN)
- Future `_format_version: 1.1` minor → tolerate; `2.0` major → reject
- `_format_version` declared first in dataclass field order (N1 fold)

**Precedence chain (C1):**
- CLI > env > config > default parametrized over every absorbed field
- Shadow logs WARNING with both values
- **Convergence** (both layers set, values agree) logs INFO (CR3 fold)
- Deprecation INFO fires exactly once across 100 `resolve_setting` calls per env var (I-4 fold)

**API key references (C1):**
- File-path mode resolves lazily at lane backend construction (I-2 fold)
- File-deleted-between-load-and-first-inference raises `BackendAuthFailed` lazily
- Keyring URI resolves lazily; missing keyring package does NOT block `maxim config get` (I-1 fold)
- Inline-string value at config-load → `ConfigurationError` (cross-confirmed I-3/IM3 fold)
- CI grep on `tests/fixtures/` for inline-key patterns (sk-/gsk_/AKIA/xoxb-) returns zero matches

**CLI verbs (C2):**
- Round-trip: set then get returns the written value
- Unknown field refused (both get and set)
- Set coercion (e.g., `set proxy.max_concurrent 4` writes int not string)
- Concurrent set from two processes both persist (I-5 fold; mirrors `test_drain_state_concurrent.py`)
- CI grep allow-list on `config_writer.py` enforces single writer (IM2 fold)

**Role unification (C3) — eight-cell matrix** (CR2 fold):
- `test_config_json_role_wins_over_env_var`
- `test_config_json_role_wins_over_peer_yml_legacy`
- `test_stale_peer_yml_plus_cloudflared_yml_resolves_to_leader_not_peer`
- `test_stale_peer_yml_plus_cloudflared_yaml_resolves_to_leader_not_peer`
- `test_cli_flag_solo_vs_cloudflared_leader_precedence`
- `test_cli_flag_solo_with_no_peer_signals`
- `test_no_signals_at_all_defaults_to_leader`
- `test_leader_mode_detect_role_returns_same_as_role_py` over the full eight-cell matrix

**Lane routing + peer.yml migration (C4 + IM5 fold):**
- `test_config_lanes_drive_backend_resolution`
- `test_env_var_still_wins_with_deprecation_warning`
- `test_peer_connect_writes_to_both_files` (dual-write during compat window)
- `test_peer_yml_demoted_below_cloudflared_in_role_detection` (Trigger #3 fix)
- `test_auto_migration_shim_writes_config_from_peer_yml`
- `test_auto_migration_shim_idempotent_when_config_exists`
- `test_auto_migration_shim_does_not_fire_when_cloudflared_present` (preserves leader case)
- `test_peer_yml_deprecation_info_fires_once_per_startup`
- `test_config_json_lanes_large_does_not_trigger_cloud_gate` (self-hosted classification)

**Doctor section (C5):**
- Shows every absorbed field with source
- Override chain visible for both shadow AND convergence cases (N2 fold)
- Inline API key flagged `[INLINE — MIGRATE]`
- peer.yml present flagged as deprecated
- File-mode != 0600 flagged
- Keyring not installed flagged `[unresolved]`

**Tests + isolation:**
- New autouse fixture `_isolate_config_json_env` in `tests/conftest.py` (template: `_isolate_maxim_llm_profile_env`)
- Deprecation INFO captured via `caplog`

**Total new test count:** ~120-140 (somewhat above the leader-UX PR's 136 due to the eight-cell role matrix + peer.yml migration coverage).

---

## Two-lens review prompts (run after implementation, before PR)

### Executor lens prompt

> Review the implementation in `src/maxim/runtime/config_loader.py` + `src/maxim/runtime/config_cli.py` + the C3 unification edits in `runtime/role.py` and `runtime/leader_mode.py`. Focus on: silent-no-op failure modes (does any branch swallow an error?), precedence-chain correctness (does CLI > env > config > default hold for every absorbed field, including the edge case where env is empty-string vs None?), atomic-write correctness on `config.json` under concurrent writes, API key reference resolution (file path vs keyring URI vs inline), and migration consistency (is `runtime/leader_mode.py::detect_role` actually equivalent to `runtime/role.py::detect_role` after C3?). Verify every absorbed env var has a matching config field, and that the deprecation INFO log fires exactly once per env var per startup (not per call).

### Architecture lens prompt

> Review the design against CLAUDE.md invariants. Specifically: (1) does the schema honor CC3 — the dataclass section types should each declare option (a) escape-hatch or (b) shape-frozen? (2) does `config.json` correctly sit in the declarative config layer per the mesh.yml two-layer-split invariant, and do API key references correctly stay OUT of the JSON file? (3) does the precedence chain match the codebase's avoid-silent-no-op discipline (loud logging on every override)? (4) does the role-detector unification actually CLOSE the divergence, or does it just paper over it with a wrapper? Test: with `config.json::role` absent AND `MAXIM_ROLE` absent AND `~/.cloudflared/config.yml` present, do BOTH detectors return the same answer? (5) is the leaky-abstraction risk on `lanes.large.remote_api_key_ref` real — what happens when the referenced file doesn't exist at startup vs first request? (6) is the schema-versioning story (1.0 → 1.1 minor non-breaking, 2.0 major breaking, ignored fields forward-compat) compatible with CC1's `_format_version` discipline?

---

## What this does NOT cover (deferred)

- **Live reload of config.json without restart** — mirrors peer.yml/mesh.yml/profiles.yml; out of scope at 1.0
- **Schema-driven CLI help generation** — `maxim config set --help` could enumerate every field from the dataclass, but adds complexity; defer to 1.1
- **Secrets vault integration beyond keyring** — Vault/AWS Secrets Manager/etc. are post-1.0
- **Multi-profile config (`config.json` per environment)** — single-profile for now; profile-switching is a 1.1+ ask if users want it
- **Migration tooling (`maxim config migrate-from-env`)** — initially a doc, not a verb; could grow into a verb if adoption demands it
- **GUI / web UI for editing** — CLI + hand-edit only
- **Cluster-wide config broadcast (Hivemind layer 2)** — `config.json` stays per-instance; multi-node coordination is mesh.yml's job

---

## Implementation order within the single PR

1. **C1** (schema + loader) lands first — foundation everything else builds on. Two-lens architecture review focuses here.
2. **C2** (CLI verbs + canonical writer module `runtime/config_writer.py`) — operator-facing surface, wraps C1.
3. **C3** (role unification) — internal refactor, riskiest because it touches role detection. Two-lens executor review focuses here.
4. **C4** (lane routing migration + peer.yml → config.json migration shim per IM5) — backward-compat migration, deprecation warnings.
5. **C5** (doctor section) — pure additive UI.
6. **C6** (docs + deprecation INFO log) — wraps the user-facing story.

Single PR opened against `main` after all six commits + post-implementation two-lens review folds. Worktree: `Maxim-wt-config-unification`. Branch: `feat/v1-config-unification`.

**N3 planning-discipline note:** if post-implementation review-fold pressure on the PR exceeds two rounds, C4 (lane-routing migration + peer.yml migration shim) is the explicit cut-out candidate — it's a compat-preserving env-var → config-field migration that can ship as a follow-up PR without blocking the Mac-Mini-bug-fixing C1+C2+C3 surface. The other stages are tightly coupled (C5 depends on C1/C2 having resolved-source data; C6 wraps the whole story); do not split them.

---

## Estimated impact on the original Mac Mini operator setup

**Today** (12 env vars + 4 files + 2 detectors + interaction rules):

```bash
export MAXIM_ROLE=leader
export MAXIM_LLM_ENABLED=1
export MAXIM_LLM_PROFILE=qwen2.5-32b-instruct
export MAXIM_LLM_N_CTX=16384
export MAXIM_AUTO_DOWNLOAD_MODELS=1
export MAXIM_AUTO_SPAWN_LLM_SERVER=1
export MAXIM_AUTO_SPAWN_TUNNEL=1
# Make sure these are NOT set if you're a leader:
unset MAXIM_LANE_LARGE_REMOTE_URL
unset MAXIM_LANE_LARGE_REMOTE_API_KEY
unset MAXIM_LANE_LARGE_REMOTE_MODEL
# Make sure peer.yml + mesh.yml don't exist:
rm -f ~/.config/maxim/peer.yml ~/.config/maxim/mesh.yml
# Make sure cloudflared config is .yml not .yaml:
mv ~/.cloudflared/config.yaml ~/.cloudflared/config.yml 2>/dev/null
# Set everything in shell rc and reload:
echo 'export MAXIM_ROLE=leader' >> ~/.zshrc
# ... 11 more lines ...
source ~/.zshrc
```

**Post-C1-C6** (1 file, 1 command):

```bash
maxim config set role leader
maxim config set llm.profile qwen2.5-32b-instruct
maxim config set llm.n_ctx 16384
maxim config set llm.auto_download true
# Done. No env vars to leak between shells, no .yml/.yaml extension trap,
# no stale peer.yml to nuke. `maxim doctor` shows the full resolved config
# in one place.
```

Or even shorter — `maxim config edit` opens `$EDITOR` on a single file. Same operator outcome.
