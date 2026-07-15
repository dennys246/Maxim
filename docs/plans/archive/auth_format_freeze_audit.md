# Auth Format-Freeze Audit (CC13)

**Status:** SHIPPED (2026-06-15) on branch `feat/1-0-cc13-auth-format-freeze`. All four surfaces landed as a single PR (~396 insertions across 10 files + 1 new doc). Two-lens pre-merge review (Executor + Architecture) folded — no CRITICAL findings; one Executor IMPORTANT (non-ASCII Bearer credential → `TypeError`/500 hardened to a clean 401) + one Architecture IMPORTANT (CC3 docstring cross-reference) + four MINOR/NIT folds applied. Full fast suite green (7997 passed). Originally DRAFT (2026-06-04).

**What shipped:**
- **A1** — documented the reserved `api_key_ref` URI scheme namespace (`pkcs11:`/`fido2:`/`tpm:`/`vault:`/`op://`/`env:`) in [`docs/user/stable_api.md`](../user/stable_api.md) as a closed enum (deny-by-default validator unchanged) + a parametrized guard test pinning the reserved schemes are rejected at 1.0.
- **A2** — added the reserved-null `signer_identity: str | None` field to the Hivemind bundle manifest ([`hivemind/bundle.py`](../../src/maxim/hivemind/bundle.py)) + published the `signature_algorithm` registry at [`docs/user/hivemind_bundle_format.md`](../user/hivemind_bundle_format.md). Doc-only (no validator), consistent with the no-verification-yet decision.
- **A3** — added three reserved-null sibling fields (`cluster_keys`, `cluster_trust_anchors`, `cluster_auth_mode`) to `MeshConfig` ([`peer/mesh_config.py`](../../src/maxim/peer/mesh_config.py)) as `tuple[str, ...] | None` (frozen-dataclass-hashable). Not serialized/parsed at 1.0 — 1.1 lands parser+writer+round-trip together on activation; `mesh_setup.py` add/remove-node carry `TODO(1.1)` forwarding markers.
- **A4** — refactored leader-proxy `_check_auth` ([`runtime/leader_proxy.py`](../../src/maxim/runtime/leader_proxy.py)) to parse the auth scheme before the credential and dispatch on it; unknown schemes return a distinct 401 (never a 400, never a silent accept). Bearer matched case-insensitively per RFC 7235; constant-time credential comparison preserved.

**Regression guard:** [tests/unit/test_config_loader.py::TestAPIKeyRefValidation](../../tests/unit/test_config_loader.py) + [tests/unit/test_hivemind_bundle.py](../../tests/unit/test_hivemind_bundle.py) (`signer_identity` round-trip + backward-compat) + [tests/unit/test_mesh_config.py::TestClusterAuthFormatFreeze](../../tests/unit/test_mesh_config.py) + [tests/unit/test_leader_proxy.py::TestProxyAuthSchemeDispatch](../../tests/unit/test_leader_proxy.py).

Scoped into 1.0 as Section 7 / CC13. Parallel-track work — ran alongside benchmarking (`benchmarking_1_0.md`) without touching any core system under measurement.

**Scope:** narrow format-freeze audit of the security-shaped surfaces shipping in 1.0, so the future hardware-token / signed-bundle / mTLS work (planned for 1.1+ alongside Hivemind P2P) is not boxed out by 1.0 schema choices. **This plan does NOT implement authentication**. It freezes the shapes such that 1.1+ can add WebAuthn / FIDO2 / PIV / hardware-signed bundles without a breaking format change.

## Motivation

Maxim is about to ship a substrate-sharing layer (Hivemind PRs A–D) and is already running multi-machine peer/leader meshes. Every existing auth surface is symmetric bearer-style:

- `lanes.<tier>.remote_api_key_ref` — peer auth to leader proxy, file path or `keyring:` URI
- `mesh.yml::cluster_key` — single shared bearer for mesh membership
- Hivemind bundle `signature` / `signature_algorithm` — reserved-null at 1.0, 1.1+ verification
- Leader proxy `Authorization: Bearer <key>` — peer→leader API auth

The pieces in flight (Hivemind P2P protocol 1.2, poison-resistance `trusted_sources` hook, bundle signing) all imply asymmetric / hardware-backed credentials in the medium term. A user surfaced the question after working with a YubiKey on another project: "we put all secret management on the user, should we standardize an auth pipeline?"

**The implementation answer is no for 1.0** — full pluggable auth (provider abstraction, WebAuthn enrollment, credential rotation UX) is multi-week scope that competes with the Tier 1 graduations + Hivemind tracks already on the 7–11 week critical path. The 1.1+ Hivemind P2P plan already reserves the natural slots (`trusted_sources`, bundle signatures).

**The format-freeze answer is yes** — every slot above is something 1.0 schema-freezes. If the slot can't grow `pkcs11:` / `fido2:` / `ed25519-hardware` later without breaking, post-1.0 hardening forces either a 2.0 bump or a parallel-field migration. ~half-day of design review now buys multi-year format flexibility.

## What this plan does NOT include

- **No new auth providers.** No FIDO2 enrollment, no WebAuthn flows, no PIV/PKCS#11 reader code. Those land 1.1+ alongside Hivemind P2P.
- **No `cluster_key` rotation mechanism.** Reserved scope but separate plan.
- **No bundle-signing implementation.** Slot already reserved per Hivemind PR D; this plan only audits whether the slot shape is right.
- **No leader proxy auth changes.** Bearer stays the 1.0 default; this plan only checks whether the wire format admits alternates later.

## Audit surfaces

### A1. `lanes.<tier>.remote_api_key_ref` URI scheme namespace

**Current state** ([src/maxim/runtime/config_loader.py::_validate_api_key_ref](../../src/maxim/runtime/config_loader.py)): accepts (a) file paths starting with `/` or `~`, (b) `keyring:<service>:<account>` URIs. Any other string is rejected with a fix-hint. Inline plaintext keys are rejected (per the cross-confirmed I-3 + IM3 fold).

**Question:** does the namespace cleanly admit `pkcs11:<slot>:<id>`, `fido2:<credential-id>`, `tpm:<handle>`, `vault:<path>`, `op://<vault>/<item>` (1Password CLI), or `env:<VAR_NAME>` later without a breaking validator change?

**Risk:** the validator's *deny-by-default* posture is correct for plaintext safety but means every new URI scheme is a breaking validator change. A user who pip-upgrades 1.0 → 1.1 and finds their `pkcs11:` ref rejected (because 1.0's validator didn't know the scheme) has hit a regression.

**Deliverable:** decide between:

1. **Frozen short-list (status quo):** document the two accepted schemes in `docs/user/stable_api.md` as a *closed enum*. 1.1+ adds new schemes by extending the validator and the doc. Each addition is a non-breaking *widening*. Pro: simple, what we already have. Con: every new scheme requires a code change in `config_loader.py`.

2. **Open URI namespace with registry:** widen the validator to accept any `<scheme>:<opaque>` URI matching a regex (`^[a-z][a-z0-9+.-]*:`), but maintain an internal registry of *known-good* schemes that resolve to actual key material. Unknown schemes pass validation but fail at `resolve_api_key_ref()` with a clear error. Pro: 1.1+ can ship new schemes purely as resolver plugins without touching the loader. Con: validation is now two-stage; failure mode is "syntactically valid but unresolvable."

**Recommendation:** Option 1 with a documented namespace reservation. Document the schemes `pkcs11:`, `fido2:`, `tpm:`, `vault:`, `op:`, `env:` as **reserved for future use** in the doc — implementations will land 1.1+. This freezes user expectations without code changes and matches the existing "extend the validator at scheme-addition time" pattern.

**LOC estimate:** ~0 (doc-only) for Option 1; ~30 LOC + 8 tests for Option 2.

---

### A2. Hivemind bundle `signature_algorithm` registry

**Current state** ([src/maxim/hivemind/bundle.py:221](../../src/maxim/hivemind/bundle.py)): `signature: str | None` and `signature_algorithm: str | None` in `compose_bundle` and manifest serialization. Both reserved-null at 1.0. Module docstring states: "Callers that want signing build their own ZIP with a populated signature field and a custom verifier; this module does NOT validate."

**Question:** when 1.1+ verification lands, what algorithm names are recognized? Right now any string is accepted (because there's no validator). That's fine for *writing* — the slot accepts whatever the producer chose — but for *cross-instance verification* (the whole Hivemind point), producers and consumers need a shared vocabulary.

**Risk:** Hivemind P2P (1.2) will absorb bundles from heterogeneous producers. If Producer A writes `signature_algorithm: "ed25519"` and Producer B writes `signature_algorithm: "Ed25519"` and Consumer C only knows `"ed25519"`, signature verification silently fails to even attempt validation. Worse: a malicious producer can write `signature_algorithm: "always_pass"` and a naive verifier might dispatch by string match.

**Deliverable:** publish a **reserved registry** of recognized `signature_algorithm` values in `docs/user/hivemind_bundle_format.md`:

- `ed25519` — bare Ed25519 signature, signing key is a 32-byte raw seed
- `ed25519-pgp` — Ed25519 wrapped in OpenPGP detached signature format
- `ed25519-ssh` — Ed25519 wrapped in SSH signature format (`ssh-keygen -Y sign`)
- `webauthn` — WebAuthn assertion (authenticator data + clientDataJSON + signature)
- `fido2-cose` — Raw FIDO2 COSE signature
- `pkcs7` — CMS/PKCS#7 detached signature (X.509 chain)
- Reserved: `hsm:*`, `kms:*`, `vendor:*` prefixes for hardware-bound / cloud-KMS / vendor-specific extensions

**Validator stays absent at 1.0** (consistent with current docstring) — the registry is documentation-only. 1.1+ verification implementations dispatch on the registry and reject unknown values with a clear error rather than silently accepting.

**Also reserve:** a `signer_identity: str | None` field on the manifest, parallel to `contributor_id`, for the "who claims to have signed this" string. Without the slot reserved at 1.0, 1.1+ has to wedge identity into the existing string fields. Cost is ~5 LOC + 1 test.

**LOC estimate:** ~10 LOC (add `signer_identity` field + 1 test) + doc.

---

### A3. `mesh.yml::cluster_key` shape

**Current state** ([src/maxim/peer/mesh_config.py:182](../../src/maxim/peer/mesh_config.py)): `cluster_key: str` — single shared symmetric secret. Sent as `Authorization: Bearer <cluster_key>` for mesh membership.

**Question:** does the field shape support a future where mesh authentication is asymmetric (each node has a keypair, the mesh trusts a list of public keys)? Or where the cluster has multiple acceptable keys (rotation window)?

**Risk:** the field is named `cluster_key` (singular). Post-1.0 widening to `cluster_keys: list[str]` for rotation is a breaking parser change — `mesh.yml::parse_mesh_config` is dialect-frozen per the existing invariant in CLAUDE.md.

**Deliverable:** the audit answer here is mostly *don't widen the field, add a sibling*. Reserve the following sibling field names in `peer/mesh_config.py::MeshConfig` (declared `None` at 1.0, parser tolerates absent):

- `cluster_keys: list[str] | None` — rotation list, when present `cluster_key` is the active write key and `cluster_keys` is the accepted-on-read list
- `cluster_trust_anchors: list[str] | None` — list of trusted public keys (for asymmetric mesh auth)
- `cluster_auth_mode: str | None` — `"bearer"` (default), `"asymmetric"`, `"mtls"`

This costs ~15 LOC of declared-but-null fields + parser tolerance. No behavior change at 1.0. 1.1+ wires the actual mode dispatch. Without the reservation, every addition is either a `mesh.yml` v2 dialect (heavy) or a separate sidecar file (split source of truth, exactly the failure class the C3 config-unification work fixed).

**Note:** `mesh.yml` parser dialect is FROZEN per the existing CLAUDE.md invariant. Adding reserved fields with `None` defaults that the parser ignores when absent is a non-breaking extension *within* the frozen dialect (it's the same shape the dialect already permits — flat top-level scalars).

**LOC estimate:** ~15 LOC + 3 tests.

---

### A4. Leader proxy `Authorization:` header set

**Current state** ([src/maxim/runtime/leader_proxy.py](../../src/maxim/runtime/leader_proxy.py)): peer→leader auth is `Authorization: Bearer <api_key>`. Forwarded as-is to llama-cpp-server upstream.

**Question:** does the wire format admit `Authorization: Signature ...` (RFC 9421 HTTP Message Signatures) or future `Authorization: HSM-Sig ...` schemes without a breaking change?

**Risk:** the *header itself* is RFC-defined and obviously extensible — Bearer / Basic / Digest / Signature / Negotiate are all `Authorization:` variants. No risk at the HTTP layer.

**The real risk is at the proxy validator**: if the proxy hard-codes "split on whitespace, check token equals api_key", a future `Authorization: Signature keyId=...` request looks like a malformed Bearer token.

**Deliverable:** audit the proxy's auth check to confirm:

1. It parses the scheme (first token) before the credential, doesn't assume Bearer
2. Unknown schemes return 401 with a clear error, not 400 / not silent accept
3. Multiple acceptable schemes can coexist (1.0 ships only Bearer; the dispatch table is reserved)

Also reserve the proxy-side ability to require **mutual TLS** for peer→leader auth — currently the proxy doesn't terminate TLS (cloudflared does). For 1.1+ scenarios where a peer runs mTLS directly to a leader, the proxy code shouldn't assume cleartext-or-cloudflared.

**LOC estimate:** ~20 LOC (auth scheme dispatch refactor) + 4 tests.

---

### A5. `contributor_id` / signer-identity coupling on bundles

**Current state:** Hivemind bundles carry `contributor_id` (per-link / per-node provenance, per PR A) but no signer identity. The bundle manifest as a whole has the reserved `signature` / `signature_algorithm` slots but no field declaring *who* signed it.

**Question:** in 1.1+ when bundle verification lands, how does a verifier answer "did this bundle's signer have the authority to claim this `contributor_id`?"

**Risk:** if 1.0 ships with `contributor_id` as a free-form producer-controlled string and 1.1+ wants to bind it to a verified identity, the binding has to be retrofitted onto data already in the wild. With a `signer_identity` field reserved at 1.0 (per A2), 1.1+ can require `signer_identity` matches a key bound to the claimed `contributor_id` in a trust registry.

**Deliverable:** already covered by A2 (`signer_identity: str | None` field reservation). Cross-link from PR D's bundle docs to the A2 registry doc.

---

## Implementation order

This is one PR, not a series. Total scope: ~50 LOC + ~16 tests + 1 docs page.

1. **A1:** doc-only — extend `docs/user/stable_api.md` (or its successor) with the api_key_ref scheme reservation.
2. **A2:** add `signer_identity: str | None` field to bundle manifest + write `docs/user/hivemind_bundle_format.md` with algorithm registry.
3. **A3:** add three reserved fields to `MeshConfig` dataclass with `None` defaults; extend parser to tolerate (likely already does — confirm).
4. **A4:** refactor proxy auth scheme dispatch (~20 LOC); the rest of the change is structural reservation, not behavior.
5. **A5:** doc cross-link.

## Pre-merge review

Two-lens review (Executor + Architecture) — this is a freeze-decision, exactly the kind of change where pre-merge review pays off. Specifically watch for:

- Have we boxed out any *known-planned* future scheme? (PGP, mTLS, HSM-backed signatures, hardware token credentials)
- Have we reserved any namespace that we won't actually want?
- Is `signer_identity` the right name, or does Hivemind already have a better one we should reuse?

## What this unlocks for 1.1+

- **Bundle signing:** `signature_algorithm` registry exists; verifier dispatches by string match
- **Hardware-backed peer auth:** `lanes.<tier>.remote_api_key_ref` accepts `pkcs11:` / `fido2:` schemes when 1.1+ implements the resolvers
- **Mesh rotation:** `cluster_keys` rotation list, `cluster_trust_anchors` for asymmetric auth — schema already in place
- **Mesh asymmetric auth:** `cluster_auth_mode: "asymmetric"` flips dispatch
- **mTLS at the proxy:** scheme dispatch table accepts new schemes without breaking Bearer clients

## What this does NOT unlock

- **Pluggable auth provider abstraction** — that's the multi-week 1.1+ work this plan explicitly defers
- **YubiKey enrollment UX** — separate 1.1+ track, depends on this freeze landing
- **Per-tool authorization scopes** — separate concern, not format-frozen by anything here

## Related plans

- [maxim_hivemind.md](maxim_hivemind.md) — the substrate-sharing layer this audit hardens for
- [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) — peer/leader mesh, where `cluster_key` lives
- [config_unification.md](config_unification.md) — where `api_key_ref` validation landed
- [v1_refinement.md](v1_refinement.md) Section 7 / CC13 — this plan's index entry

## Estimate

- **Wall time:** 0.5–1 day implementation + 0.5 day review. Genuinely parallel to benchmarking — touches only doc + freeze-shape fields, not behavior under measurement.
- **LOC:** ~50 src + ~16 tests + 1 doc page.
- **Risk:** low (format-freeze additions with `None` defaults, no behavior change at 1.0).
