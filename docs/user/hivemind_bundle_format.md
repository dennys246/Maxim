# Hivemind Bundle Format — Signature & Identity Registry

This page is the canonical, append-only registry of the **signature** vocabulary for Hivemind substrate bundles. It documents the recognized `signature_algorithm` values and the `signer_identity` field so that the 1.2 peer-to-peer Hivemind protocol's heterogeneous producers and consumers share one string vocabulary instead of guessing.

**1.0 does not implement bundle signing or verification.** The `signature`, `signature_algorithm`, and `signer_identity` manifest fields are reserved-null at 1.0. This page is **documentation only** — there is no validator in the 1.0 codebase that rejects or accepts an algorithm string. It exists to freeze the vocabulary so a 1.1+ verifier can dispatch on a known registry rather than inventing names that collide with other producers'. See the [Auth Format-Freeze Audit](../plans/auth_format_freeze_audit.md) (CC13) for the why.

## Manifest signature fields

Every bundle's `manifest.json` carries three reserved auth slots, all `None` at 1.0:

| Field | Type | Meaning |
|---|---|---|
| `signature` | `str \| None` | The detached signature material (encoding is algorithm-specific — see below). |
| `signature_algorithm` | `str \| None` | A value from the [recognized registry](#recognized-signature_algorithm-values). Tells a verifier how to interpret `signature`. |
| `signer_identity` | `str \| None` | The "who claims to have signed this" string, parallel to `contributor_id`. Lets a 1.1+ verifier bind a *verified* identity to the *claimed* `contributor_id` in a trust registry. |

The fields are produced by [`compose_bundle`](../../src/maxim/hivemind/bundle.py) (all default `None`) and round-trip through [`extract_bundle`](../../src/maxim/hivemind/bundle.py) / `read_bundle_manifest` unchanged. Older bundles written before `signer_identity` existed load cleanly — the field is optional and read via `.get(...)`.

## Recognized `signature_algorithm` values

A 1.1+ verifier dispatches on these strings and **rejects unknown values** rather than silently accepting an unverified bundle. Matching is exact and case-sensitive (`ed25519`, never `Ed25519`).

| Value | Meaning |
|---|---|
| `ed25519` | Bare Ed25519 signature; signing key is a 32-byte raw seed. |
| `ed25519-pgp` | Ed25519 wrapped in an OpenPGP detached signature. |
| `ed25519-ssh` | Ed25519 wrapped in an SSH signature (`ssh-keygen -Y sign`). |
| `webauthn` | WebAuthn assertion (authenticator data + clientDataJSON + signature). |
| `fido2-cose` | Raw FIDO2 COSE signature. |
| `pkcs7` | CMS / PKCS#7 detached signature with an X.509 chain. |

### Reserved prefixes

These namespace prefixes are reserved for hardware-bound, cloud-KMS, and vendor-specific extensions. A verifier that does not implement a given prefix treats it as unknown and refuses verification:

| Prefix | For |
|---|---|
| `hsm:*` | Hardware Security Module–bound keys (e.g. `hsm:pkcs11:...`). |
| `kms:*` | Cloud KMS–managed keys (e.g. `kms:aws:...`, `kms:gcp:...`). |
| `vendor:*` | Vendor-specific signature schemes not covered above. |

## Contract

- **Append-only.** Once an algorithm name or reserved prefix is listed here, its meaning does not change. New values may be added in a minor release (a non-breaking widening of the verifier's dispatch table).
- **Unknown → refuse.** A future verifier MUST reject an unrecognized `signature_algorithm` rather than skip verification or treat the bundle as unsigned. A malicious producer setting `signature_algorithm: "always_pass"` must not slip past a string-match dispatch.
- **Documentation-only at 1.0.** No code path in 1.0 reads, writes, or validates these fields beyond round-tripping the manifest. Producers that want signing today build their own ZIP with a populated `signature` field and a custom verifier (per the [bundle module docstring](../../src/maxim/hivemind/bundle.py)).
- **`signer_identity` vs `contributor_id`.** `contributor_id` is the free-form, producer-controlled provenance string (per-link / per-node, set at compose time). `signer_identity` is reserved for the *cryptographically attested* identity a 1.1+ verifier checks against a trust anchor. They are intentionally separate so a verified signer can be required to match the claimed contributor. Like `contributor_id`, `signer_identity` shares the reserved `_*` namespace discipline used across the Hivemind layer (see [`hivemind/merge.py`](../../src/maxim/hivemind/merge.py) `_validate_source`): a 1.1+ implementer must NOT reuse an existing sentinel such as `_consensus` (which already means "aggregated across contributors" in merge provenance) or `_identity` to express a multi-signer or attested-identity concept — pick a fresh, distinct value.

## See also

- [Substrate Sharing](substrate-sharing.md) — the user-facing export / import / merge workflow
- [Maxim Hivemind + Oasis](../hivemind.md) — the 1.1 Oasis and 1.2 P2P roadmap this format feeds
- [Auth Format-Freeze Audit](../plans/auth_format_freeze_audit.md) — CC13, the freeze decision behind this registry
- [Stable API](stable_api.md) — the broader 1.0 contract surface (including the `api_key_ref` URI namespace reservation)