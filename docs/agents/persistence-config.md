# Persistence & Config — working brief

> Part of the CLAUDE.md satellite layer. Read this whole file before editing `utils/atomic_io.py`, `utils/format_version.py`, `utils/seeding.py`, `utils/paths.py`, `runtime/config_loader.py`, `runtime/config_writer.py`, `runtime/role.py`, `runtime/leader_mode.py`, any persisted-JSON shape, or any frozen dataclass. The slim CLAUDE.md core + this brief are intended to be sufficient context for work in this area. Full incident narratives: docs/lessons/.

## 1. Mental model — the three-layer state map

Every file Maxim reads or writes belongs to exactly one of three layers, and the layer decides who may write it, with what primitive, and when:

1. **Declarative operator intent** — `~/.config/maxim/config.json`, `mesh.yml`, `peer.yml`, `profiles.yml`. These express what the operator *wants*. They are written ONLY by operator-explicit one-shot CLI verbs (`maxim config set ...`, `init-mesh`/`add-node`/`remove-node`), never by runtime code, never by admin APIs, never as a side effect of another operation. Automatic runtime writes to this layer are forbidden (the C2 invariant); the temptation to "commit operator intent for them" always creates a two-source-of-truth reconciliation problem.
2. **Mutable runtime state** — `~/.maxim/util/{name}.{role}.txt|.json` (drained nodes, `active_llm_model.{role}.txt`, rate-limit overrides, `lane_decisions.jsonl`, request traces). This is what's *actually happening*. Runtime code writes here freely, but every read-modify-write cycle is serialized under a `filelock.FileLock`, filenames are role-scoped via `MAXIM_ROLE`, and writes go through the atomic helpers. The spec-vs-status split (Kubernetes-style) means layers 1 and 2 never need reconciliation — drift between them (e.g. `llm.profile` vs `active_llm_model.{role}.txt`) is *by design* and surfaced loudly, not auto-healed.
3. **Persisted bio/session state** — `~/.maxim/` session dirs, `hippocampus.json`, `nac.json`, `ec.json`, memory stores, caches. Written via `atomic_write_json` + `with_format_version`, versioned under the `_format_version` contract, and any value that is *hashed* before crossing this boundary uses the stable-hash helpers (builtin `hash()` is per-process-randomized and permanently unmatchable after restart).

Role detection sits upstream of all three layers: it runs ONCE at process start (`runtime/role.py::detect_and_apply_role`, called from `cli.py::_main_impl` right after `configure_logging`), exports `MAXIM_ROLE`, and everything downstream reads the env var. Role decides which layer-2 filenames apply and which layer-1 files are authoritative.

### "Which file may I write from code?" — decision table

| Target | Runtime code may write? | Sanctioned route |
|---|---|---|
| `config.json` | NO — operator-explicit CLI verbs only | `runtime/config_writer.py` (`write_config` / `mutate_config` / `set_field`); CI grep allow-lists callers |
| `mesh.yml` | NO — operator-explicit setup verbs only | `peer/mesh_setup.py` → `peer/mesh_config.py::write_mesh_config`; CI grep allow-lists callers (see docs/agents/llm-routing.md) |
| `peer.yml` | NEVER (read-only; the one-time migration reads it, never deletes it) | — |
| `~/.maxim/util/*` | YES | `filelock.FileLock` around the full RMW + `atomic_write_text`; validate against `mesh.yml`'s node set at write time where applicable |
| Persisted JSON (bio/session) | YES | `atomic_write_json(path, with_format_version(payload))` |
| Anything containing a secret (API keys, cluster keys, tokens) | — | `atomic_write_secret` — the function name is the "this contains secrets" signal; never pass `preserve_mode=True` to `atomic_write_text` directly |

If you want a mutable field *in* a declarative file ("operator committed their intent"), stop: it belongs in `~/.maxim/util/` with a `maxim peer list-<thing>` surface.

## 2. Key files

| Area | Key files |
|---|---|
| Atomic persistence | `src/maxim/utils/atomic_io.py` (`atomic_write_json`, `atomic_write_text`, `atomic_write_secret`), `src/maxim/utils/paths.py` (data path resolution) |
| Format versioning | `src/maxim/utils/format_version.py` (`with_format_version`, `check_format_version`); envelope `schema_version` in `memory/snapshot.py` |
| Stable hashing | `src/maxim/utils/seeding.py` (`stable_hash_32`, `stable_hash_64_signed`) |
| Config load/precedence | `src/maxim/runtime/config_loader.py` (`resolve_setting`, `_env_is_set`, env coercers, `_maybe_migrate_from_peer_yml`, `_apply_lane_config_to_env`) |
| Config write | `src/maxim/runtime/config_writer.py` (the ONLY sanctioned config.json writer) |
| Role detection | `src/maxim/runtime/role.py` (`detect_role`, `detect_and_apply_role`), `src/maxim/runtime/leader_mode.py` (thin back-compat wrapper) |

## 3. Invariants & lessons

**[engineering] `~/.config/maxim/config.json` is the operator-config layer and `runtime/config_writer.py` is its ONLY sanctioned writer** (`write_config`/`mutate_config`/`set_field`: atomic_write_json + with_format_version under a FileLock acquired BEFORE the read). Precedence: CLI > env > config.json > builtin defaults via `resolve_setting`; empty-string env vars are UNSET; shadow (env mismatches config) logs WARNING, convergence INFO. Declarative config files take operator-explicit one-shot writes only — no automatic runtime / admin-API writes (mesh.yml rule). API key refs accept ONLY file paths or `keyring:` URIs — inline plaintext keys are rejected at load. Full history: [docs/lessons/config-json-writer-canonical.md](../lessons/config-json-writer-canonical.md). Regression guard: CI grep in [.github/workflows/test.yml](../../.github/workflows/test.yml) ("config_unification.md C2 + C6 invariants (IM2 fold)") allow-lists callers of `write_config` / `mutate_config` / `set_field` — new callers fail CI; tests in [tests/unit/test_config_writer.py](../../tests/unit/test_config_writer.py) + [tests/unit/test_config_cli.py](../../tests/unit/test_config_cli.py) + [tests/unit/test_config_loader.py](../../tests/unit/test_config_loader.py) pin every fold from the pre-implementation review.

**[engineering] `runtime/role.py::detect_role` is the single source of truth for role detection, and role detection is the first runtime action** — never add a second detector (the pre-C3 pair silently disagreed and regressed the Mac Mini on day one). Seven-rank order: `MAXIM_ROLE` env → `config.json::role` → `mesh.yml` (peer) → cloudflared config (leader) → `peer.yml` (peer) → `--llm` local + no peer config (solo) → default leader. `cli.py::_main_impl` calls `runtime/role.py::detect_and_apply_role(raw_argv)` immediately after `configure_logging`, BEFORE subcommand dispatch; downstream code reads `os.environ["MAXIM_ROLE"]` — never re-detects, never calls `detect_role()` a second time, never infers from `peer.yml` existence. Persisted state is split per role (`active_llm_model.{role}.txt`). `leader_mode.py::detect_role` is a thin back-compat wrapper (legacy `client` term drops in 1.2); `ConfigurationError` from config.json surfaces as WARNING, never silently corrupts rank order. Full history: [docs/lessons/detect-role-single-source.md](../lessons/detect-role-single-source.md) + [docs/lessons/role-detection-first.md](../lessons/role-detection-first.md). Regression guard: [tests/unit/test_role_unification.py](../../tests/unit/test_role_unification.py) (the eight-cell matrix pinning every Mac Mini failure mode + the ConfigurationError surface) + [tests/unit/test_role_detection.py](../../tests/unit/test_role_detection.py) (the legacy seven-rank tests updated for C3 ordering) + [tests/unit/test_leader_mode.py](../../tests/unit/test_leader_mode.py) (wrapper translation contract) + [src/maxim/runtime/role.py::detect_and_apply_role](../../src/maxim/runtime/role.py) + [src/maxim/cli.py::main](../../src/maxim/cli.py) — call site structurally precedes subcommand dispatch.

**[engineering] `_maybe_migrate_from_peer_yml` writes peer.yml → config.json on first startup iff config.json absent + peer.yml present + cloudflared config ABSENT** (the absent-cloudflared condition protects the leader-with-stale-peer.yml case — never auto-flip a leader to peer). peer.yml is NEVER deleted; the API key goes to `~/.config/maxim/api_key` via `atomic_write_secret` under `umask(0o077)`; `_migration_attempted` is set only AFTER a successful write (transient OSError retries next load). `_apply_lane_config_to_env` is idempotent via `_lane_env_applied` — without the guard a second call re-attributes source config→env and breaks doctor attribution. Full history: [docs/lessons/peer-yml-auto-migration.md](../lessons/peer-yml-auto-migration.md). Regression guard: [tests/unit/test_lane_routing_via_config.py](../../tests/unit/test_lane_routing_via_config.py) (migration shim + idempotency + retry-on-transient-failure + self-hosted classification + peer.yml fallback).

**[engineering] Auto-save must not run under the hippocampus RWLock write block — read-lock-under-write self-deadlocks.** Before calling anything inside a held write block on a non-reentrant RWLock, audit the callee chain for ANY lock acquisition (save → dump → `read()`); a lock-taking persistence call belongs in the public wrapper after release (NOTE tombstones forbid re-adding it at the old sites). Corollary: a conftest fixture that globally disables a default-on config flag is a CI blind-spot signal. Full history: [docs/lessons/autosave-outside-rwlock-write-block.md](../lessons/autosave-outside-rwlock-write-block.md). Regression guard: [tests/integration/test_persistent_agent_campaign.py](../../tests/integration/test_persistent_agent_campaign.py) (`test_sleep_with_autosave_does_not_deadlock` — sleep runs in a thread with a bounded join, so a regression fails fast instead of hanging the suite).

### Reference: frozen-dataclass path-(a)/(b) registry (detail for core stub A05)

The core rule (see CLAUDE.md): every `@dataclass(frozen=True)` that persists or crosses a wire MUST declare path (a) or (b) in its class docstring before merge. Current rosters:

- **Path (a) — escape-hatch** (all fields defaulted + `extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)` — the `hash=False, compare=False` is load-bearing; `extra` values must be JSON-serializable; `__post_init__` rejects extra keys colliding with declared field names): `Episode`, `SensorReading`, `LaneTierConfig`, `ProviderPlacement`, `LaneTierPlacement`.
- **Path (b) — SHAPE-FROZEN** (docstring carries the `SHAPE-FROZEN at 1.0 (CC3)` marker + rejection rationale; adding a *required* field post-1.0 is a major-version bump; an optional trailing defaulted field is non-breaking but still review-gated): `Reaction`, `ReactionContext`, `TraceSnapshot`, `PerceptContext`, `CouplingSpec`, `ModulationSpec`, `HomeostaticDriveSpec`, `EntropicDriveSpec`, `TemporalSignature`, `ValenceSignal`, `MeshNode`, `MeshConfig`, `ModelPricing`, `VRAMProjection`, `MaximConfig`, `LLMConfigSection`, `LanesConfigSection`, `CloudConfigSection`, `ProxyConfigSection`, `AutoSpawnConfigSection`, `DataConfigSection`, `SimConfigSection`, plus `_InFlightCall` (llm_call_registry). `TraceSnapshot` is shape-frozen because it is reachable from `ReactionContext.bindings` — an `extra` dict on it would re-open the isolation back-channel `ReactionContext` itself closes.
- **Typed exception hierarchies** (`models/language/types.py::BackendError`, `utils/http.py::HTTPError`) follow the same spirit with different mechanics: explicit keyword-only `__init__` per subclass, no `**kwargs`/`extra` (re-opens the silent-typo bug class).
- **Out of scope:** runtime-ephemeral config dataclasses (`HippocampusConfig`, `RetrievalConfig`, `NACConfig`, ...) that never cross a wire or session boundary — append defaulted fields freely. Persisted *and* wire-crossing types are the audit surface.

### Reference: `_format_version` dual-convention detail (detail for core stub A06)

Two version conventions coexist by design; do not "unify" them:

- **`_format_version` (string, root-level)** — the broad contract EVERY persisted JSON file carries. Writers: `atomic_write_json(path, with_format_version(payload))`. Loaders: `check_format_version(data, "<file_type>", log=logger)` after `json.load` — a missing field returns the `"0.x"` sentinel and warns once per file_type per process; old files load, never silently fail. Bump it when the file *shape* changes. Adding the field post-1.0 would itself have been breaking (new code writes it, strict old validators reject unknown keys) — which is why it was freeze-critical at 1.0.
- **Envelope `schema_version` (int, `memory/snapshot.py`)** — the *migration trigger* for the six envelope-conformant bio-systems. Bio-system files written via direct `save(path)` carry BOTH at the JSON root. Shape change on a bio-system file → bump `schema_version` and register a migration, OR bump `_format_version` — not the legacy strings.
- **Tombstoned:** the legacy payload-layer `"version": "1.0"` strings (ATL, NAc, SCN, etc.) are dead per snapshot.py — never bump them.
- **List-rooted files** (File*Store, last_run, web_cache, probe_cache) are wrapped in a thin `{kind, items}` (or `{entries}`, `{tools}`, `{sessions}`) dict so the version field has a slot; loaders accept both the v1.0 dict and the pre-1.0 bare list.
- Wire payloads are versioned independently of session payloads under the same contract (e.g. `Percept.to_wire_dict` vs `to_dict` — see docs/agents/llm-routing.md).

Regression guard: [tests/integration/test_persistence_compat.py](../../tests/integration/test_persistence_compat.py). Full history: [docs/lessons/format-version-contract.md](../lessons/format-version-contract.md).

### Reference: stable-hash converted sites (detail for core stub A25)

The core rule: persistence-crossing values hash via `utils/seeding.py::stable_hash_32` / `stable_hash_64_signed`, never builtin `hash()` (PYTHONHASHSEED randomization). The five converted sites, for pattern-matching when you touch nearby code:

1. `similarity/signature.py` — structural/context hash (cross-process similarity collapse straddled NAc's EC gate).
2. `memory/context_index.py` — MinHash (reloaded `SimilarityIndex` returned `[]` for its own stored text; the `test_context_index.py::test_similar_text_found` CI flake).
3. `similarity/lsh.py::SemanticLSH.hash` — took a `seed` PARAMETER that still routed through randomized `hash()`; looking deterministic is not being deterministic.
4. `similarity/semantic.py::_fallback_hash` — same trap as 3.
5. `decisions/nac.py::_register_causal_in_ec`.

Sum-then-branch-on-sign sites use the SIGNED 64-bit variant (an unsigned digest collapses every hyperplane bit to 1). Persisted files carry `hash_scheme: "stable-sha256-v1"`; loaders WARN when absent (pre-fix files' hashes are permanently dead). A same-process test passes over this whole bug class — guards MUST be two-process with differing PYTHONHASHSEED. Regression guard: [tests/unit/test_stable_hash_two_process.py](../../tests/unit/test_stable_hash_two_process.py). Full history: [docs/lessons/stable-hash-persistence.md](../lessons/stable-hash-persistence.md).

## 4. Live gotchas / known gaps

- **`atomic_write_json` is detection-guarded only:** the ad-hoc `grep -rn "os.replace" src/maxim/ | grep -v atomic_io.py` currently surfaces several hand-rolled writer sites (report/dashboard/plotting-adjacent files per the Phase-0 audit) and nothing fails CI on them. The rule stands; the sweep is unclaimed work — do not add new hand-rolled sites.
- **`cli.py::main` naming drift:** the startup ordering (configure_logging → detect_and_apply_role → ... → dispatch) lives in `cli.py::_main_impl`; `main()` is now a thin BackendError-surfacing wrapper. Guards citing `cli.py::main` mean the `_main_impl` body.
- **`config.json::llm.profile` vs `active_llm_model.{role}.txt` drift is by design** — `maxim --llm <model>` updates runtime state only, never config.json; the singleton check fails loud with the `maxim config set llm.profile` resolution. Details home in docs/agents/llm-routing.md (declarative-vs-runtime-model-state).
- **Pre-stable-hash persisted files are permanently dead** for hash-keyed lookups (no `hash_scheme` marker) — loaders warn; do not try to "repair" them.
- **Two advisory-file-lock abstractions coexist** (`maxim.utils.process_lock` for model downloads, `filelock.FileLock` for drain/config state); unification is a deferred shell plan (`docs/plans/deferred/cross_platform_file_lock.md`).
- **Versioning:** `pyproject.toml` and `src/maxim/__init__.py` must stay in sync (rule lives in CLAUDE.md core — not restated here).

## 5. Env vars owned

- `MAXIM_DATA_BUDGET_GB` = optional soft cap on `~/.maxim` disk usage; refuses model downloads over budget = `utils/storage.py` + `models/download.py` (resolved via `runtime/config_loader.py`).

(`MAXIM_ROLE` is documented in docs/agents/llm-routing.md's env table; its detection/export semantics are the merged role invariant above. Long env-var rationales: docs/lessons/claude-md-2026-08-13-pre-diet.md.)

## 6. Lesson archive

- [docs/lessons/config-json-writer-canonical.md](../lessons/config-json-writer-canonical.md)
- [docs/lessons/detect-role-single-source.md](../lessons/detect-role-single-source.md)
- [docs/lessons/role-detection-first.md](../lessons/role-detection-first.md)
- [docs/lessons/peer-yml-auto-migration.md](../lessons/peer-yml-auto-migration.md)
- [docs/lessons/autosave-outside-rwlock-write-block.md](../lessons/autosave-outside-rwlock-write-block.md)
- [docs/lessons/frozen-dataclass-forward-compat.md](../lessons/frozen-dataclass-forward-compat.md)
- [docs/lessons/format-version-contract.md](../lessons/format-version-contract.md)
- [docs/lessons/stable-hash-persistence.md](../lessons/stable-hash-persistence.md)

Cross-refs: also see docs/agents/llm-routing.md for mesh.yml parser/state specifics (frozen dialect, declarative-vs-`~/.maxim/util` mechanics) and the profile-vs-active-model drift lesson; docs/agents/bio-memory.md for the L06 subject system (Hippocampus), the NAc+EC pair-persistence invariant, and hivemind bundle/scrub format; docs/agents/runtime-tools.md for the rest of the cli.py startup-ordering cluster (LeaderProxy, logging-before-dispatch).