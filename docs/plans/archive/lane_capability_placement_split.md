# Separate Lane *Capability* from *Placement* (origin)

> **ARCHIVED (2026-07-15 plans audit):** ✅ ALL PHASES SHIPPED (PRs #357–#360, #362) + CLAUDE.md invariant landed. `ProviderPlacement`/`Origin`/`LaneTierPlacement` + 3 regression-test files verified in tree. Only remainder — per-placement hardware tuning — is documented in the CLAUDE.md invariant as an explicit 1.1+ extension.


**Status:** DRAFT 2026-06-09
**Author:** Denny + Claude
**Target:** pre-1.0 architecture cleanup (gate item)
**Motivates:** clean local+cloud coexistence on one node; "use cloud when local
is unavailable" as a first-class policy; removing the size-vs-origin conflation
in the lane system before the lane config schema is frozen for 1.0.

## The conflation (why this exists)

The lane system collapses **two orthogonal axes** into one "lane" concept:

1. **Capability / size** — how heavy a model the *work* needs:
   `large` / `medium` / `small`. This is a property of the **task**, assigned by
   [`FunctionRouter`](../../src/maxim/runtime/function_router.py) (`DEFAULT_TIER_ORDER
   = ["large","medium","small"]`; each function declares a `tier` + `fallback`).
   The names are FROZEN (`_VALID_TIER_NAMES` in
   [config_loader.py](../../src/maxim/runtime/config_loader.py)).

2. **Origin / placement** — *where/how* a model runs: local llama-cpp, cloud
   (Anthropic/OpenAI/…), or a self-hosted peer. This is a property of
   **infrastructure**, independent of how heavy the task is.

Today there is **no first-class placement axis.** A lane's origin is smeared
across `LaneConfig` fields ([worker_pool.py:73](../../src/maxim/runtime/worker_pool.py))
and inferred heuristically:

- `model_profile` set, no `remote_url` → **local** llama-cpp.
- `remote_url` set → **cloud** or **self-hosted**, disambiguated by
  *URL-sniffing* in [`LaneBackendManager._classify`](../../src/maxim/runtime/lane_backends.py)
  (`_is_cloud_url(remote_url)`).
- `--cloud-lane <tier> <model>` rewrites a tier's `model_profile` to a *cloud
  profile* (and clears `remote_url`) — i.e. you express "I want cloud" by
  reassigning the **size** tier to a cloud model of matching size
  (`claude-haiku` ↦ `small`, `claude-sonnet` ↦ `large`).
- `--cloud-fallback <model>` is a *second* placement mechanism, injected
  separately as a secondary provider in
  [`_maybe_inject_cloud_fallback`](../../src/maxim/runtime/lane_backends.py).

So "placement" is expressed three different ways (implicit field combos,
URL-sniffing, two overlapping CLI flags), all routed through the size-tier knob.
The user-visible symptom: `--cloud-lane small claude-haiku` reads as "the
*small-size* lane is now cloud," conflating *what the work needs* with *where it
runs*. There is no clean way to say "for the **large** capability, prefer
local-Mistral, then peer-leader, then cloud-Sonnet."

**This is a design wart to resolve before the lane config schema freezes at
1.0** — once `lanes.<tier>` is frozen (it is already CC3-shape-tracked), adding
a placement axis is a breaking change.

## Target model

Keep the **capability axis unchanged** (3 frozen tiers, `FunctionRouter`
unchanged). Introduce an explicit **placement axis**: per capability tier, an
ordered list of candidate providers, resolved at dispatch by
availability/health/policy.

Sketch (names to be finalized in the design phase):

```
Origin = Enum("local", "cloud", "peer")

@dataclass(frozen=True)
class ProviderPlacement:
    origin: Origin
    model: str                 # profile name (local) or cloud model id or peer model
    url: str | None = None     # for cloud/peer
    api_key_ref: str | None = None
    timeout_s: float | None = None
    # ... (CC3 escape-hatch `extra: dict` if needed)

# LaneConfig gains (additive, backward-compatible):
placement: tuple[ProviderPlacement, ...] = ()   # ordered preference; () = derive from legacy fields
```

A small **resolver** picks the active `ProviderPlacement` at dispatch time
(first healthy/eligible one), replacing `_classify`'s URL-sniffing with an
explicit `origin` read. The existing single-backend lane is the degenerate
1-element placement.

This makes the two axes independent:
- **Capability** answers "how smart must this be?" → tier (function-driven).
- **Placement** answers "where does that capability run, in what order of
  preference?" → per-tier ordered providers (infra/policy-driven).

It also **subsumes** `--cloud-lane` and `--cloud-fallback` into one concept:
both become edits to a tier's placement list (prepend/append a cloud provider).
`local primary → cloud fallback` is just `placement=[local…, cloud…]`.

## Phase 0 outcome — finalized design (2026-06-09)

Read-only audit complete; every cited symbol verified against the live tree.
**One name-drift correction:** the CLI-override fn is `_apply_cloud_cli_overrides`
([lane_backends.py](../../src/maxim/runtime/lane_backends.py)), not
`_apply_cloud_lane_overrides` (corrected in References below).

### Front-gate scope decision (the load-bearing call)

The word "placement" was conflating two separable things:

1. **The placement *resolution*** (ordered preference + first-healthy failover)
   **already exists** — in `LLMRouter` at the provider layer (`providers` dict +
   `routing.provider_priority` + `_try_provider`'s specific-before-general
   failover). Both `_maybe_inject_cloud_fallback` and `_build_remote_backend`
   already compile down to an ordered `provider_priority`. A *second* resolver in
   `LaneBackendManager` would duplicate `_try_provider` and re-introduce the
   multi-call hazard the `_MaximPeerBackend` one-call invariant exists to kill.
2. **The placement *type*** (explicit per-provider `origin`) **does not exist** —
   origin is *derived* by `_classify`'s URL-sniff, and the two CLI flags express
   it incoherently. This is the actual wart.

**Decision: placement is a first-class TYPE + a compile step, NOT a new
resolver.** It rides on `LLMRouter`'s existing provider-priority/failover as the
resolution engine. A per-tier ordered `placement` tuple *compiles* into the
`providers` / `provider_priority` of the single `LLMRouter` that already serves
the lane. `_classify`'s URL-sniff becomes "read `origin`"; the two CLI flags
become two edits (prepend cloud / append cloud) to one ordered list. This is the
plan's "minimal win" framing and is strictly safer than a new resolver.

### Subtle load-bearing behavior the compile step MUST preserve exactly

Cloud gating has **two distinct, non-overlapping paths** today:

- `get_backend`'s `MAXIM_MAX_CLOUD_LANES` cap fires **only** for `kind=="cloud"`,
  which `_classify` returns **only** for a `remote_url` on a *public host*.
- `--cloud-lane` / `--cloud-fallback` set a cloud *profile* (anthropic/openai
  backend) with **no `remote_url`** → `_classify` returns `"local"` → they do
  **not** consume a cap slot; they're gated instead by `MAXIM_LLM_CLOUD_ENABLED`
  + the router's `_validate_cloud_config` (key/redaction/cost).

**Invariant:** legacy-field derivation must reproduce `_classify`'s exact output
for every existing config, and the cap/gate logic must key off the *same* origin
values — introducing an explicit `origin` must NOT start counting cloud-*profile*
lanes against the cap. Pinned by a before/after classification-matrix test.

### Finalized type shapes

```python
# worker_pool.py — LaneConfig is a runtime (non-frozen) dataclass; field is additive
class Origin(str, Enum):
    LOCAL = "local"   # llama-cpp profile on this box
    CLOUD = "cloud"   # metered provider (anthropic/openai/…)
    PEER  = "peer"    # self-hosted behind tunnel/LAN (one-HTTP-call backend)

@dataclass(frozen=True)
class ProviderPlacement:           # NEW frozen dataclass → CC3 path (a) escape-hatch
    origin: Origin
    model: str | None = None       # profile name (local) | cloud model id | peer model
    url: str | None = None
    api_key_ref: str | None = None
    timeout_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

placement: tuple[ProviderPlacement, ...] = ()   # () = derive from legacy fields
```

`Origin`'s `PEER` value renames the legacy `"self-hosted"` concept at the type
layer; `_classify`'s legacy *string* outputs (`"self-hosted"`/`"cloud"`/`"local"`)
stay during the transition so `get_lane_kind` callers/tests don't break.

### Config.json schema (Phase 3, additive — confirmed declared field)

`LaneTierConfig` is already CC3 **path (a)** (has an `extra` dict), so a declared
optional `placement: tuple[LaneTierPlacement, ...] = ()` is non-breaking and
type-validated (preferred over burying in `extra`). `_VALID_TIER_NAMES` and the
`large/medium/small` keying stay frozen; empty tuple derives a 1-element
placement from the existing `remote_url`/`remote_model`/`remote_api_key_ref`/
`timeout_s` so existing config.json files behave identically.

### Derivation direction (one-way, for back-compat)

Env / tier-detection / auto-spawn all keep landing in the **legacy** LaneConfig
fields; `placement` is *computed* from them when `placement == ()`. Only Phase 3's
new config.json `placement` and the re-expressed CLI flags *write* placement
directly.

### Migration table (read-sites of the smeared origin fields, fresh grep)

| Site | Reads | Phase-2 disposition |
|---|---|---|
| `_classify` | `remote_url`, `_peer_owned`, `_is_cloud_url` | **replace** with "read active placement `origin`"; keep `_classify` as legacy-derivation fallback |
| `get_backend` | `model_profile`/`remote_url` (None-gate), `kind` (cap) | gate on "has ≥1 placement"; cap keys off active(primary) origin |
| `_build_backend`/`_build_local_backend`/`_build_remote_backend` | all origin fields | dispatch on placement origin; compile placement list → `providers`/`provider_priority` |
| `_maybe_inject_cloud_fallback` | `cfg.name`, `MAXIM_CLOUD_FALLBACK_MODEL` | Phase 3: "append cloud placement" |
| `_apply_cloud_cli_overrides` | `MAXIM_CLOUD_LANE_<T>_MODEL` | Phase 3: "prepend cloud placement" (keep env alias) |
| `_apply_local_llm_override` | `MAXIM_LLM_PROFILE` | unchanged (clears remote → derives local placement) |
| `_ensure_lane_profiles_available` | `remote_url`, `model_profile` | unchanged (local-origin entries) |
| `_maybe_auto_spawn_server` | `remote_url`, `model_profile`; rewrites both | unchanged — rewrites legacy fields → derives placement |
| `_validate_remote_urls` | `remote_url` | unchanged |
| `describe` / `get_lane_kind` | `model_profile`, `remote_url`, `_classify` | surface placement list |
| `lane_models.detect_tiers`/`apply_lane_env_overrides`/`apply_tier_config_overrides` | builds/sets legacy fields | unchanged (legacy fields stay the env/detection landing zone) |
| `doctor/checks.py` (`model_profile` @ ~105, ~2533) | `model_profile` | Phase 3: show capability × placement |
| `config_loader.LaneTierConfig` / `_FIELD_TO_ENV` | schema | Phase 3: add `placement` |
| `config_writer` / `peer/cli.py` / `roy_runner` / `cli_utils` | set/read lanes.large.remote_* + cloud env | unchanged P1-2; aliases preserved P3 |

### Forks resolved at checkpoint (2026-06-09)

1. **Scope:** type + compile step, ride on `LLMRouter` (no new resolver). ✅
2. **Config schema:** declared optional `placement` field on `LaneTierConfig`. ✅
3. **Alias lifetime:** `--cloud-lane`/`--cloud-fallback` (+ env vars) re-expressed
   as placement edits, kept working through 1.x with one-shot deprecation INFO,
   drop in 1.2. ✅

### Phase 3 status — COMPLETE (3a + 3b + 3c, 2026-06-10)

- **3a** (PR #359, merged): builders compile from the primary placement; fence removed.
- **3b** (PR #360): config.json `placement` declarative schema (`LaneTierPlacement`) + `validate_placement_coherence` enforced at load + writer round-trip.
- **3c** (this PR): runtime producer (`config.json placement → LaneConfig.placement`) + multi-element **tail-injection** compile (`_inject_placement_tail`/`_placement_entry_to_provider`) + `--cloud-lane`/`--cloud-fallback`/`--llm` re-expressed as placement edits + doctor capability×placement view.

**Implementation finding (3c) — cloud-profile primary dispatch.** "Cloud profile" (`claude-sonnet`) vs "cloud URL" are distinct: a cloud *profile* with no url builds correctly only via the **profile-driven** `_build_local_backend` (which `load_llm_config` resolves to the right backend, e.g. anthropic), NOT the openai-assuming `_build_remote_backend`. So `_build_backend` routes a CLOUD-origin primary **without a url** to the local/profile path (forcing `cloud_enabled=True`); CLOUD-with-url and PEER go remote. Derived CLOUD always carries a url, so this only affects explicit cloud-profile placements — no regression. The CLAUDE.md `[engineering]` invariant ("Lane = capability tier; placement is a separate ordered axis") landed with 3c.

### Phase-3 follow-up notes (from the 3a two-lens review, 2026-06-09)

Phase 3a (builders compile from the **primary** placement; fence removed) shipped
clean and behavior-preserving. The review surfaced four obligations the 3b/3c
session must honor (3a is correct for 1-element placements; these are seams, not
3a defects):

- **Multi-element compile is a builder *inversion*, not an add-on (3c).** 3a
  dispatches the whole lane to ONE builder keyed on `placement[0].origin` and
  emits one provider. A `[LOCAL primary, CLOUD fallback]` (or `[PEER, CLOUD]`)
  placement is inherently cross-origin and cannot be assembled by a
  branch-on-primary structure. 3c should introduce a single
  `_compile_placement_to_providers(cfg, placement) -> (providers, provider_priority)`
  that maps **each** entry to a provider dict (factoring the per-entry shape now
  inlined in `_build_remote_backend` + the cloud-provider shape in
  `_maybe_inject_cloud_fallback`), then build ONE `LLMRouter`. Budget for the
  restructure rather than slotting into the current shape.
- **Double cloud-fallback is the #1 reconciliation item (3c).**
  `_build_local_backend` still calls the env-driven `_maybe_inject_cloud_fallback`
  unconditionally. When `--cloud-fallback` also becomes an appended CLOUD
  placement, the provider would be injected **twice**. Gate the env path on
  `cfg.placement == ()` (derived-only) or fold it into the compile helper so it
  fires exactly once. (TODO marker left at the call site.)
- **Placement coherence validation is owed by 3b.** Removing the fence traded a
  loud construction failure for a runtime failure on a malformed explicit
  placement. **DISCHARGED in 3b:** `worker_pool.validate_placement_coherence`
  (PEER needs `url`; LOCAL needs `model`; CLOUD needs `url` **or** `model` — the
  CLOUD rule is looser than the original note because a cloud model id resolves
  its base URL from the provider profile) is the single rule set, invoked at the
  config-loader parse boundary (`_parse_lane_tier_placement`) and re-raised as
  `ConfigurationError` with JSON context. It lives at the producer boundary, NOT
  in `ProviderPlacement.__post_init__`, so the runtime value type stays
  permissive (derived placements are coherent by construction). **3c** reuses
  the same validator after each CLI placement edit (catching `ValueError`).
- **Legacy `remote_*` + `placement` coexistence (3c doctor).** A tier may carry
  both; `derive_placement` short-circuits on `cfg.placement`, so placement wins
  and the legacy `remote_*` fields are ignored — the right default (no
  reject-both validation; matches the env-shadows-config-with-a-WARNING
  precedent). 3c's doctor "Resolved Config" view should WARN when both are set,
  and surface the resolved placement so operators see what the structured field
  produced.
- **No env-var parity for `placement` (intentional).** `placement` is a
  structured list → no `MAXIM_LANE_*_PLACEMENT` scalar / `_FIELD_TO_ENV` entry;
  the scalar `MAXIM_LANE_*_REMOTE_*` family remains the env path (deriving a
  1-element placement), and CLI flags are the runtime placement-override surface.
- **Cap semantics for multi-element placements (3c).** `MAXIM_MAX_CLOUD_LANES`
  is keyed on `placement_kind` = `placement[0].origin` (primary only). A
  `[LOCAL, CLOUD-fallback]` lane therefore won't consume a cap slot — which
  *matches* legacy `--cloud-fallback` (gated by `cloud_enabled` +
  `_validate_cloud_config`, not the cap). Confirm this is intended and document
  it; don't let it change by accident.
- **Hardware boundary (1.1+).** `n_gpu_layers`/`device`/`n_ctx`/`kv_quant_mode`
  stay lane-level; heterogeneous-hardware LOCAL placements aren't expressible at
  1.0 (documented on `ProviderPlacement`).

## Phases (proposed — the executing session should front-gate scope first)

Per CLAUDE.md "Front-gate scope pressure at design time": before building,
answer *"does placement need to be its own mechanism, or can it ride on the
existing per-tier backend + cloud-fallback?"* Name the answer in the design
section. (Working hypothesis: yes — the two overlapping CLI flags + URL-sniffing
+ implicit field combos are evidence the implicit model is already too weak.)

- **Phase 0 — Audit + design (read-only).** Enumerate every read site of
  `model_profile` / `remote_url` / `remote_model` / `_classify` /
  `_is_cloud_url` across `lane_backends.py`, `worker_pool.py`, `function_router.py`,
  `config_loader.py`, `config_writer.py`, `doctor/`. Confirm the capability axis
  is already clean (it is). Finalize `Origin` / `ProviderPlacement` /
  `LaneConfig.placement` shapes and the resolver contract. Decide backward-compat
  derivation (legacy fields → 1-element placement). Output: design section in
  this doc + a per-call-site migration table.

- **Phase 1 — Introduce the placement type (additive, no behavior change).**
  Add `Origin` + `ProviderPlacement` + `LaneConfig.placement` with a derivation
  shim: when `placement == ()`, synthesize it from the existing
  `model_profile`/`remote_url` fields so nothing changes. Add the resolver but
  keep `_classify` as the fallback. Pure-additive; full suite stays green.

- **Phase 2 — Route dispatch through placement.** `LaneBackendManager.get_backend`
  resolves the active `ProviderPlacement` and dispatches on its explicit
  `origin` instead of URL-sniffing. `_classify` becomes "read `origin`."
  Preserve every load-bearing behavior: singleton reuse / auto-spawn gating,
  the cloud-lane cap (`MAXIM_MAX_CLOUD_LANES`), `_validate_cloud_config`
  (key/redaction/cost gates), per-tier `timeout_s`, the one-HTTP-call peer
  contract. Regression-test each.

- **Phase 3 — Re-express the CLI/env surface as placement edits.**
  `--cloud-lane`/`--cloud-fallback` (+ `MAXIM_CLOUD_LANE_<TIER>_MODEL`,
  `MAXIM_CLOUD_FALLBACK_MODEL`) become placement-list edits (keep as
  deprecated-but-working aliases through 1.x). Add the `placement` field to the
  `config.json` `lanes.<tier>` schema (additive; respect the CC3 frozen-dataclass
  rule + `_VALID_TIER_NAMES`); update `config_loader`, `config_writer`, and the
  `maxim doctor` "Resolved Config" view to show capability × placement.

- **Phase 4 — Docs + invariant.** CLAUDE.md `[engineering]` invariant: "lane =
  capability tier; placement (local/cloud/peer) is a separate ordered axis
  resolved at dispatch." Regression-guard citations. Update the lane docs and
  the env-var table.

## Tradeoffs / honest concerns

- **The lane config schema is (becoming) frozen at 1.0.** The placement field
  MUST be additive with a legacy-derivation default; `_VALID_TIER_NAMES` stays
  frozen; `LaneTierConfig` follows the CC3 escape-hatch rules. This is the main
  reason to do it *before* 1.0.
- **Many read sites.** `model_profile`/`remote_url`/`_classify` are read across
  the lane bootstrap, doctor, config loader/writer. The migration table (Phase 0)
  is the risk-management tool; do it before touching code.
- **Don't regress the load-bearing lane behaviors.** Singleton-reuse,
  auto-spawn role gating, `MAXIM_MAX_CLOUD_LANES`, `_validate_cloud_config`,
  the `_MaximPeerBackend` one-call contract, per-tier timeouts — each has a
  CLAUDE.md invariant and tests. The resolver must preserve them exactly.
- **Scope discipline.** The *minimal* win is "make origin explicit so the axes
  aren't smeared" (Phases 0-2 + alias-preserving 3). A rich health/cost-based
  auto-router is a tempting over-reach — keep it as an explicit follow-on, not
  part of this cleanup, unless the front-gate decides otherwise.

## Out of scope

- Changing the 3 capability tiers or `FunctionRouter`'s function→tier map.
- Cost/latency-based *automatic* placement selection (a policy engine) — this
  plan delivers the *ordered-preference + first-healthy* resolver; smarter
  selection is a follow-on.
- Prompt caching / backend internals (separate, shipped work).
- Mesh/peer transport changes.

## References (current-state, verified 2026-06-09)

- [src/maxim/runtime/worker_pool.py](../../src/maxim/runtime/worker_pool.py) — `LaneConfig` dataclass (the fields that smear origin).
- [src/maxim/runtime/lane_backends.py](../../src/maxim/runtime/lane_backends.py) — `LaneBackendManager._classify` (URL-sniffing), `_apply_cloud_cli_overrides`, `_maybe_inject_cloud_fallback`, `BACKEND_CLASSES`, auto-spawn/singleton.
- [src/maxim/runtime/function_router.py](../../src/maxim/runtime/function_router.py) — capability axis (`DEFAULT_TIER_ORDER`, function→tier map). Unchanged by this plan.
- [src/maxim/runtime/config_loader.py](../../src/maxim/runtime/config_loader.py) — `_VALID_TIER_NAMES` (frozen), `lanes.<tier>` schema.
- CLAUDE.md "Architectural invariants" — lane tier names; `BACKEND_CLASSES` dispatch; `_MaximPeerBackend` one-call; CC3 frozen-dataclass audit; config_unification `lanes.<tier>.timeout_s`.
