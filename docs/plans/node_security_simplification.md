# Node Security — Immediate Fixes + Deferred Config Unification

**Status:** Phase 1 in progress (immediate fixes), Phase 2 deferred  
**Scope:** ~25 LOC immediate + ~300 LOC deferred config unification  
**Target version:** 0.4 (immediate fixes land with Plan 4)  
**Part of:** [llm_path_refinement.md](llm_path_refinement.md)  
**Depends on:** nothing — these fixes are self-contained

## What this plan is

Two things:

1. **Phase 1 — Immediate security fixes** (~25 LOC). Four concrete bugs in the auth + help-text surface. Ship before 0.4 release.
2. **Phase 2 — Config surface unification** (~300 LOC, deferred). Consolidate the dual role-detection modules + document the formal auth model. No user-visible behavior change; purely internal cleanup. Revive after Plan 4 ships.

---

## Phase 1 — Immediate fixes

### Fix 1: Timing-safe auth comparison

**File:** `src/maxim/runtime/leader_proxy.py:324`  
**Gap:** `auth == f"Bearer {self.api_key}"` — Python string comparison short-circuits on first differing byte. An attacker with access to the leader's HTTP port can measure response time differences to oracle individual key characters.

**Fix:** Replace with `secrets.compare_digest(auth, f"Bearer {self.api_key}")`.  
Add `import secrets` to module imports. ✅ **Shipped.**

### Fix 2: Rate-limiter bucket key

**File:** `src/maxim/runtime/leader_proxy.py:259`  
**Gap:** The full `Authorization: Bearer <token>` header is used as the `PeerRateLimiter` dict key. Two problems:
- The raw auth token is stored in a Python dict keyed by the token value — if the dict is ever introspected or logged, it exposes the key
- Since all peers share ONE cluster key, every authenticated peer maps to the same bucket → per-peer rate limiting is silently a cluster-wide rate limit

**Fix:** Use `self.client_address[0]` (source IP) as the bucket key.  
Source IP is what actually identifies "which peer" in a single-shared-key setup. ✅ **Shipped.**

### Fix 3: Wrong help text — key show

**File:** `src/maxim/tunnel/cli.py:582`  
**Gap:** `_cmd_key_show()` prints "They'll set it with: maxim tunnel key export" — but `maxim tunnel key export` is a **leader** command that prints snippets. Peers use `maxim peer key set <key>` or `maxim peer connect <url> --key <key>`.

**Fix:** Replace with accurate workflow instruction. ✅ **Shipped.**

### Fix 4: Wrong help text — key rotate post-action hint

**File:** `src/maxim/tunnel/cli.py:598`  
**Gap:** After rotating, the only hint is "Run `maxim tunnel key export` for peer setup snippets." No mention of what peers need to do.

**Fix:** Add peer-side instruction: "Then on each peer: maxim peer key set <new-key>". ✅ **Shipped.**

### Fix 5: Help string in module docstring

**File:** `src/maxim/tunnel/cli.py:55`  
**Gap:** "key rotate   Generate a new API key (invalidates peers)" — technically true but misleading; "invalidates" sounds like it removes peers from the config, not just forces a key update.

**Fix:** "key rotate   Generate a new API key (peers must run `maxim peer key set`)". ✅ **Shipped.**

---

## Phase 2 — Config surface unification (deferred)

**Revive when:** Plan 4 operator visibility is fully shipped.

### Background

Two separate role-detection modules exist:

| Module | Accepts | Owns |
|---|---|---|
| `runtime/leader_mode.py::detect_role()` | `"leader"/"client"/"solo"` | bind_host logic, server start |
| `runtime/role.py::detect_and_apply_role()` | `"leader"/"peer"/"solo"` | MAXIM_ROLE export, observability |

`leader_mode.py` predates the R2a role detection work. The discrepancy (`"client"` vs `"peer"`) is technical debt. `maxim doctor` surfaces divergence between them as a `role_divergence` warning.

### Goal

Single role detection path. `role.py` wins (it's the authoritative source post-R2a). `leader_mode.py` should accept `"peer"` as a synonym for `"client"` in the transition period, then be simplified to read `MAXIM_ROLE` from env rather than re-detecting.

### Scope

- `leader_mode.py`: accept `"peer"` as synonym for `"client"` → eventually read `MAXIM_ROLE` directly
- `role.py::detect_and_apply_role()`: only entry point for new code
- `doctor/checks.py::check_role()`: remove the divergence check once the two are aligned
- Documentation: formal auth model doc in `docs/architecture/`
- Tests: confirm `"client"` → `"peer"` synonym works; no regression on existing leader_mode tests

### Non-goals

- No JWT or per-peer keys in Phase 2 — that's a potential Phase 3 or post-1.0 item
- No changes to the actual auth protocol
- No changes to peer.yml format or key storage

---

## Security model (current, as of Plan 3)

- **Auth:** single shared cluster key, 256-bit entropy, 0600 perms at rest
- **Transport:** end-to-end TLS via Cloudflare tunnel (or direct TLS termination)
- **Scope:** one leader + N peers in a trusted cluster — not a multi-tenant public API
- **Key rotation:** `maxim tunnel key rotate` + `maxim peer key set` on each peer + `maxim` restart on leader

This model is appropriate for the current use case (personal/lab deployment). Multi-tenant API key management (per-peer JWT, revocation, audit log) would be a separate post-1.0 initiative.
