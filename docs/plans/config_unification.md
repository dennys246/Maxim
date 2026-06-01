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

**Format:** JSON via `maxim.utils.atomic_io.atomic_write_json` for writes; `json.load` for reads. JSON over YAML because the file is primarily machine-written (CLI verbs, operator hand-edits secondary); JSON parses faster, has stricter validation, and avoids the YAML-anchor-aliases-attack surface for a file Maxim itself writes.

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

**Out of scope at 1.0** (deliberately NOT absorbed):
- Debug / trace flags (`MAXIM_HEARTBEAT`, `MAXIM_LANE_TRACE`, `MAXIM_HTTP_TRACE`, etc.) — these are debug-mode opt-ins, not preferences worth persisting
- Research toggles (`MAXIM_NAC_*`, `MAXIM_EC_TRACE_*`, `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION`, etc.) — short-lived A/B testing flags
- Robot / embodiment config (`MAXIM_ROBOT_NAME`, `MAXIM_REACHY_HOST`) — different lifecycle (per-robot, not per-instance)
- TTS / audio (`MAXIM_TTS_*`, `MAXIM_WHISPER_*`) — feature-specific, low daily-use friction
- Comms (`MAXIM_COMMS_*`) — feature-specific

The absorbed set is ~22 daily-use settings. The remaining ~74 env vars stay as-is (debug-mode + research + feature-specific).

### Precedence chain

**CLI args > env vars > config.json > builtin defaults**

Same shape as `kubeconfig`, `gh`, `npm`, `pyproject.toml`. Loud override logging at every level mismatch.

```python
def resolve_setting(field_path: str, cli_value: Any | None) -> tuple[Any, str]:
    """Returns (effective_value, source) where source is one of:
       'cli' | 'env' | 'config' | 'default'"""
    if cli_value is not None:
        return cli_value, "cli"
    env_value = os.environ.get(_env_var_for(field_path))
    if env_value is not None:
        config_value = _read_from_config(field_path)
        if config_value is not None and config_value != env_value:
            logger.warning(
                "Config override: %s='%s' (env) shadows '%s' (config.json)",
                field_path, env_value, config_value
            )
        return _coerce(env_value, field_path), "env"
    config_value = _read_from_config(field_path)
    if config_value is not None:
        return config_value, "config"
    return _builtin_default(field_path), "default"
```

**Override logging is load-bearing.** Every effective field's source is logged at startup at INFO level (one summary line per field that resolved). Mismatches between layers are logged at WARNING level individually so the operator can spot drift in a single grep.

### Security: API keys do NOT live in config.json

Per the CLAUDE.md mesh.yml two-layer-split invariant: declarative config (`peer.yml`, `mesh.yml`, `profiles.yml`, `config.json`) uses plain `atomic_write_text` / `atomic_write_json`. Credentials use `atomic_write_secret` with mode 0600.

`lanes.<tier>.remote_api_key_ref` holds a **reference**, not the key itself. Resolution order:
1. If value looks like a file path (starts with `/` or `~`) → read mode-0600 file at that path
2. If value is a string like `"keyring:<service>:<account>"` → resolve via system keychain (macOS Keychain, Linux Secret Service)
3. If value is a plain string (legacy escape hatch) → treat as inline key BUT log WARNING that keys-in-plaintext-config is deprecated

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
    role: Literal["leader", "peer", "solo"] | None
    llm: LLMConfigSection
    lanes: LanesConfigSection
    cloud: CloudConfigSection
    proxy: ProxyConfigSection
    auto_spawn: AutoSpawnConfigSection
    data: DataConfigSection
    _format_version: str = "1.0"

def load_config(path: Path | None = None) -> MaximConfig:
    """Read config.json, validate against schema, return typed dataclass.
    Returns defaults-only MaximConfig if file is missing."""

def resolve_setting(field_path: str, cli_value: Any | None = None) -> tuple[Any, str]:
    """The precedence chain. Returns (effective_value, source)."""
```

**Validation:**
- Unknown top-level keys → `ConfigurationError` with key listed
- Unknown nested keys → WARNING (forward-compat: future schemas may add fields, log but don't fail)
- Type mismatch → `ConfigurationError` naming field + expected type
- Invalid enum value → `ConfigurationError` listing valid values
- `role: "client"` (the old leader_mode.py term) → auto-coerce to `"peer"` with WARNING (compat)

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

**Regression guards:**
- `tests/unit/test_config_cli.py::TestRoundTrip::test_set_then_get`
- `TestUnknownFieldRefused` for both get and set
- `TestSetCoercion` — `set proxy.max_concurrent 4` writes int not string

---

## C3 — Role-detector unification (closes the two-detector divergence)

**The current state:** two `detect_role` functions in two modules with different decision orders + different file-extension assumptions. The 2026-06-01 Mac Mini regression hit divergence on day one.

**The fix:**
1. `runtime/role.py::detect_role` becomes the **single source of truth**. Its decision order extends to:
   - `config.json::role` (NEW, highest priority after env var)
   - env var
   - mesh.yml exists → peer
   - peer.yml exists → peer
   - `--llm` flag → solo
   - default leader
2. `runtime/leader_mode.py::detect_role` is replaced with a thin wrapper that calls `runtime/role.py::detect_role` and translates `leader|peer|solo` → `RoleDecision(role, bind_host)`. The cloudflared-config-exists branch (current leader_mode.py:55-58) becomes a fallback INSIDE `role.py::detect_role` if config.json + env + mesh.yml + peer.yml all fail to specify role. **The fallback widens to accept both `.yml` AND `.yaml` extensions** — fixing the 2026-06-01 Mac Mini bug as a side effect.
3. `role_divergence` event is REMOVED (no longer possible; one detector).
4. `role_detected` event gains a `config_json_present` field for telemetry on adoption.

**Migration:** non-breaking. Existing setups without `config.json` follow the existing decision tree exactly. Setups with `config.json::role` get a deterministic result regardless of env var hygiene.

**Regression guards:**
- `tests/unit/test_role_unification.py::test_config_json_role_wins`
- `test_no_config_json_falls_through_to_existing_logic`
- `test_yaml_extension_now_accepted` (the Mac Mini bug)
- `test_leader_mode_detect_role_returns_same_as_role_py` — pin the wrapper-vs-source consistency

---

## C4 — Per-tier remote routing migrates to lanes.* in config.json

**Today:** `MAXIM_LANE_LARGE_REMOTE_URL` + `_API_KEY` + `_MODEL` for each of large/medium/small = 9 env vars. The Mac Mini regression had a stale `MAXIM_LANE_LARGE_REMOTE_URL` from when this box was a peer, silently re-routing the leader's own large lane.

**The fix:**
- `lanes.<tier>.remote_url/remote_model/remote_api_key_ref` in `config.json`
- `lane_backends.py` resolves via `resolve_setting("lanes.large.remote_url", cli_value=None)` → respects precedence chain
- Env vars still work (precedence chain handles them) but emit DEPRECATION INFO at startup when set
- `maxim peer connect` writes `lanes.large.remote_url` + `lanes.large.remote_api_key_ref` into `config.json` instead of (or in addition to) `peer.yml`. peer.yml stays for backward-compat reads.

**Regression guards:**
- `tests/unit/test_lane_routing_via_config.py::test_config_lanes_drive_backend_resolution`
- `test_env_var_still_wins_with_deprecation_warning`
- `test_peer_connect_writes_to_both_files` (during the compat window)

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
  ✓ proxy.max_concurrent: 4               [source=default]
  ⚠ MAXIM_LANE_LARGE_REMOTE_URL is set in env (http://127.0.0.1:8100/v1)
    but config.json::lanes.large.remote_url is null. The env var wins.
    → If this box is a leader, unset the env var. Run `maxim config get` to verify.
```

The override-summary table is the single answer to "what does this instance think it's configured as?" — collapsing what previously required cross-referencing 96 env vars, 4 files, and 2 role detectors.

**Regression guards:**
- `tests/unit/test_doctor.py::TestResolvedConfigSection::test_shows_every_absorbed_field`
- `test_override_chain_visible`

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
| `config.json` has invalid JSON syntax | `ConfigurationError` at startup, blocks `maxim` from running. | Same shape as profiles.yml — broken config should fail fast, not silently mis-route. `maxim doctor` surfaces the parse error before next inference attempt. |
| `config.json` has a future schema field (`_format_version: 1.1`) | Loader warns once, ignores unknown fields, continues. | Forward-compat — newer Maxim installs in a heterogeneous mesh can write 1.1; older readers tolerate. |
| `config.json` has `_format_version: 2.0` | `ConfigurationError` — major version bump is breaking. | Per CC1; loader's job to refuse incompatible majors. |
| Env var sets `MAXIM_ROLE=client` (old name) | Auto-coerce to `peer` with WARNING. | leader_mode.py used the old name; we coerce silently was tempting but the WARNING makes the deprecation visible. |
| Two API key references point at the same file | Allowed. | Multiple lanes can share a key (common in the same-leader case). |
| `lanes.large.remote_api_key_ref` is a string that LOOKS like a key (not a path) | Treat as inline key + log WARNING that this is deprecated. | Operator escape hatch; gentle migration. |
| `lanes.large.remote_api_key_ref` is a `keyring:` URI but keyring package not installed | `ConfigurationError` with hint to `pip install keyring`. | Graceful failure with actionable fix. |
| User passes `--config /custom/path/config.json` | CLI flag overrides the default path. | Useful for testing + multi-instance setups. |
| `maxim config set` is run concurrently from two tmux panes | `filelock.FileLock` around the read-modify-write. | Mirrors `peer.yml` / drain state pattern. |

---

## Regression guards summary

Every test lives at `tests/unit/test_config_*.py`:
- Schema-shape: minimum valid config, all defaults
- Precedence chain: CLI > env > config > default (parametrized over every absorbed field)
- Override logging: every shadow logs WARNING with both values
- Schema errors: invalid JSON, bad enum, wrong type, unknown required field
- Migration: missing file → defaults, future minor version → ignored fields
- Role unification: config.json wins, fallback through existing tree, both yml AND yaml extensions accepted
- Lane routing via config: drives backend resolution, env var deprecation warning
- Doctor section: shows every absorbed field with source
- CLI verbs: round-trip, unknown field refused, set coercion

**Total new test count:** ~80-100 (similar to the leader-UX PR's 136).

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
2. **C2** (CLI verbs) — operator-facing surface, wraps C1.
3. **C3** (role unification) — internal refactor, riskiest because it touches role detection. Two-lens executor review focuses here.
4. **C4** (lane routing migration) — backward-compat migration, deprecation warnings.
5. **C5** (doctor section) — pure additive UI.
6. **C6** (docs + deprecation INFO log) — wraps the user-facing story.

Single PR opened against `main` after all six commits + two-lens review folds. Worktree: `Maxim-wt-config-unification`. Branch: `feat/v1-config-unification`.

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
