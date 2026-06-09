# Separate Lane *Capability* from *Placement* (origin)

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
- [src/maxim/runtime/lane_backends.py](../../src/maxim/runtime/lane_backends.py) — `LaneBackendManager._classify` (URL-sniffing), `_apply_cloud_lane_overrides`, `_maybe_inject_cloud_fallback`, `BACKEND_CLASSES`, auto-spawn/singleton.
- [src/maxim/runtime/function_router.py](../../src/maxim/runtime/function_router.py) — capability axis (`DEFAULT_TIER_ORDER`, function→tier map). Unchanged by this plan.
- [src/maxim/runtime/config_loader.py](../../src/maxim/runtime/config_loader.py) — `_VALID_TIER_NAMES` (frozen), `lanes.<tier>` schema.
- CLAUDE.md "Architectural invariants" — lane tier names; `BACKEND_CLASSES` dispatch; `_MaximPeerBackend` one-call; CC3 frozen-dataclass audit; config_unification `lanes.<tier>.timeout_s`.
