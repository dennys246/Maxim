# Key-Drift Detection — Proactive API Key Mismatch Surface for Peer ↔ Leader

**Status:** Drafted 2026-06-03. Follow-up to [config_unification.md](archive/config_unification.md). Triggered by a real Mac Mini setup 2026-06-03 where a peer's stored API key was 2 months stale and the leader had rotated 2 days prior. The 401 the peer would have received contained no hint that the key was the problem — operators have to guess between auth, hostname, leader state, or network as the cause.

**Scope:** small follow-on PR (~150 LOC src + ~40 tests). Independent of any other 1.0 work. Ships after config_unification.md merges.

---

## Why proactive detection vs the reactive message fix

The reactive fix shipped in config_unification.md (the UX-fold commit, 2026-06-03) improves the **error message** at 401 time:

- `peer/probe_classify.py::classify_probe_outcome` for `outcome="auth_rejected"` now says "your stored API key may be stale" and points at `maxim tunnel key show` + `maxim peer connect`.
- `doctor/cli.py::_peer_test` and `doctor/checks.py::check_remote_reachability` both inherit the message via the classifier.
- `lane_backends._PROBE_FIX_HINTS["auth_rejected"]` carries the same hint for the runtime backoff path.

That fix unblocks the operator **when** they hit the 401. It doesn't tell them their key is stale **before** they hit the 401 — e.g., during `maxim doctor` on a quiet peer, or during background lane-health probes. This plan adds the proactive detector.

## Design — fingerprint endpoint + peer-side comparison

**Leader side:** new endpoint on the LeaderProxy.

```
GET /v1/admin/key-fingerprint
Authorization: (not required)
Response: 200 OK
Content-Type: application/json
{
  "fingerprint": "<sha256(api_key)[:16] in hex>",
  "rotated_at_unix": 1717372800,
  "format": "sha256-16"
}
```

- No authentication. The fingerprint is a 16-character hex prefix of `sha256(api_key)`. With a 256-bit key, the prefix gives a 2^64 search space — infeasible to brute-force back to the key. The endpoint discloses only that the key has a particular fingerprint, not the key itself.
- `rotated_at_unix` is the Unix timestamp of the last `maxim tunnel key rotate` (read from a marker file the rotate verb writes alongside `api_key`).
- `format` reserves the version field for future fingerprint algorithm changes (e.g., HMAC-based comparison with per-peer salt for unlinkability).
- Endpoint lives in `runtime/leader_proxy.py` adjacent to the existing `/v1/debug/ping`.

**Peer side:** new check + a CLI verb.

- `maxim.peer.key_fingerprint::compute_fingerprint(api_key: str) -> str` — pure helper. Implements the same `sha256[:16]` shape as the leader.
- `maxim.peer.key_fingerprint::detect_drift(leader_url: str, stored_key: str) -> KeyDriftResult` — fetches `/v1/admin/key-fingerprint` (one HTTP call, no auth header), compares against `compute_fingerprint(stored_key)`. Returns a frozen dataclass with `status: Literal["match", "mismatch", "endpoint_missing", "probe_failed"]` + `leader_rotated_at: int | None` + `peer_key_age_s: int | None` + `detail: str`.
- `doctor/checks.py::check_key_drift(url, api_key)` wraps `detect_drift` and emits a `CheckResult`:
  - `match` → `ok`
  - `mismatch` → `warn` with the same rotate-and-re-paste fix the reactive 401 message uses
  - `endpoint_missing` → `info` (older leader; gracefully degrades)
  - `probe_failed` → `info` (network issue — the broader peer-reachability check already catches this)

The new doctor check slots into the existing "Peer Connectivity" section in `run_all_checks`, right after `check_peer_auth`.

## peer.yml + config.json schema (additive, non-breaking)

The peer's stored key gains a paired timestamp so the doctor can also flag "your key is ~90+ days old" independently of leader rotation:

- `peer.yml::api_key_set_at: ISO-8601` — optional. Written by `maxim peer connect`. Absence means pre-1.x peer.yml — fall back to file mtime as a proxy.
- `config.json::lanes.<tier>.extra.api_key_set_at: ISO-8601` — same value, written into the `LaneTierConfig.extra` escape-hatch dict (the IM1 fold made it path-(a) exactly so additive fields like this don't need a schema bump).

The `maxim tunnel key rotate` verb writes `~/.config/maxim/api_key.rotated_at` (Unix timestamp). The fingerprint endpoint reads this for `rotated_at_unix`.

## CLI surface additions

- `maxim peer test <url>` — already runs the probe chain; add the fingerprint comparison as step 5 between `/v1/models` and the chat completion round-trip. Doesn't slow happy-path (it's one extra unauth GET, ~30ms typical).
- `maxim peer key-drift` — new verb that runs ONLY the fingerprint comparison. For scripts that want a fast (<1s) drift check without the full chat completion.

## Why not do this in config_unification.md

config_unification.md was scoped to "operator-config layer cleanup" — schema, precedence chain, role detector, lane routing, doctor surface. Key-drift detection touches a different surface area:

- New HTTP endpoint on LeaderProxy
- New peer-side helper module
- New CLI verb
- Schema additions to peer.yml + config.json

That's larger and orthogonal. The config_unification UX fold (the reactive message improvement) was the small piece that fit cleanly in the same PR; the proactive detector earns its own plan + PR.

## Out of scope at this design

- **Server-pushed rotation events** — the leader writes a rotation event to a known endpoint that peers subscribe to. Larger change (requires either polling or a real pub/sub channel). Defer to 1.2 if operator demand justifies.
- **Multi-leader fingerprint reconciliation** — when a peer has multiple leaders configured (large lane → leader-A, medium lane → leader-B), each leader has its own fingerprint. The detector handles per-lane drift but doesn't reconcile cross-leader. Out of scope.
- **Fingerprint algorithm versioning** — the `"format": "sha256-16"` field reserves the slot. Adding a second format is a 1.1+ ask, not 1.0.

## Sizing + sequencing

| Module | LOC est. | Tests |
|---|---:|---:|
| `runtime/leader_proxy.py` — `/v1/admin/key-fingerprint` endpoint | ~40 | 5 |
| `peer/key_fingerprint.py` — `compute_fingerprint`, `detect_drift`, `KeyDriftResult` dataclass | ~60 | 10 |
| `doctor/checks.py::check_key_drift` + wiring into `run_all_checks` | ~30 | 6 |
| `peer/cli.py` — fingerprint step in `peer test`, new `peer key-drift` verb | ~30 | 5 |
| `peer/config.py` + `config_loader.LaneTierConfig.extra` schema additions | ~15 | 5 |
| Plan doc + docs/user updates | ~80 doc lines | — |
| **Total** | **~175 src + ~80 doc** | **31 tests** |

Single PR. Ships after config_unification.md merges so it can build on the canonical writer module + the doctor "Resolved Config" section.

## Decision log

- **Why `sha256[:16]` not full sha256?** 16 hex chars = 64 bits of comparison entropy. Brute-forcing back to the key requires 2^64 work — infeasible. Full sha256 (64 chars) is mildly more secure but harder to read in doctor output. 16-char prefix is a usability/security tradeoff that matches the API-key truncation shown elsewhere in doctor (`OYpVQC…EqV2q8`).
- **Why no auth on the endpoint?** The fingerprint is intentionally a public-ish disclosure. An attacker who can hit the endpoint learns "the leader's key has fingerprint X." They cannot derive the key from X. The only marginal exposure is timing — but timing-channels on a 256-bit key are not exploitable. Authenticated alternatives would require a shared secret peers have *before* they know the key, which is the very problem this endpoint solves.
- **Why ISO-8601 timestamps in peer.yml but Unix in the endpoint?** Peer.yml is operator-hand-edit-readable (ISO is friendlier). The endpoint is machine-machine (Unix is more compact and easier to compare). Conversion is one line.

---

## Tracking — what this depends on, what depends on it

**Depends on (merged before this ships):**
- `config_unification.md` C1-C7a — for the canonical writer module, `config.json::lanes.<tier>.extra` escape-hatch, doctor surface
- `config_unification.md` post-impl UX fold (commit shipping 2026-06-03) — for the reactive 401 message that this plan's proactive detector complements

**Future work that builds on this:**
- Server-pushed rotation events (1.2 if demand justifies — see "Out of scope")
- Per-peer fingerprint salting for unlinkability (1.2+)
- Cross-leader reconciliation when peers have multiple-leader config (1.2+)