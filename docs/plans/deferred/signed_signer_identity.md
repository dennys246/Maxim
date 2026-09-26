# Bind `signer_identity` into the bundle signature — deferred on triggers

> **REVIVED 2026-09-25 — trigger (a) fired.** Step 1 landed with release format v2
> ([oasis_entry_index_v2.md](../oasis_entry_index_v2.md)): `signer_identity` is inside the signed bytes
> (a detached signature over the raw manifest), v1 is legacy-only. Step 2 (drop v1) stays on trigger (b).

> **DEFERRED 2026-09-25 (owner).** public_oasis Phase 0 item 6 asked for `signer_identity` to be
> covered by the signature. Investigation showed the unsigned label is **not exploitable by
> construction**, so the wire-format change is deferred to a moment when the format changes anyway.

## Why it is safe today

`hivemind/signing.py::bundle_signing_payload` signs the manifest minus `signature`,
`signature_algorithm` and `signer_identity` (they are written after signing), plus the slices. The
label is still safe because verification (`bundle.py::verify_bundle_signature_parts`) checks the
signature with the key trusted **for the claimed identity**: relabelling a bundle makes the wrong
key check it and it fails. `hive pull`'s label pre-screen only routes the bundle to that verification
(`--require-signed` plus every Queen key). The one residual case — one key trusted under two
identities — is refused at verification and at registration (`tests/unit/test_signer_identity_binding.py`).
The comparison is on DECODED key bytes (a 32-byte key has four valid base64 spellings). Displays
(`substrate export`'s summary, the Oasis store log) say "claimed signer". Follow-up, not security:
registration still accepts a malformed key (it fails loudly at verification instead) because test
fixtures and open PRs use placeholder keys — tighten it in a fixture sweep.

## What binding costs, and why not now

Adding the identity to the signed bytes changes the payload, so every existing signed bundle stops
verifying unless the scheme is versioned and the old one kept — and keeping it leaves the unsigned
path open anyway, which gains nothing while that path is not exploitable.

## The plan (owner, 2026-09-25)

1. **At the next release-format change** — public_oasis Phase 0 item 7, the Queen-signed entry
   index, which changes the release format anyway — introduce signing scheme **v2** with
   `signer_identity` inside the signed payload. Verifiers accept **v1 and v2**, warning on v1.
2. **At the next major release (2.0)** — drop v1: verifiers refuse unversioned bundles.

## Revive triggers

- (a) public_oasis Phase 0 item 7 (the entry index) starts — do step 1 in the same format change.
- (b) the next major release is planned — do step 2 in it.
- (c) **any code path starts trusting `signer_identity` without verifying it** (a display that
  implies verification, a trust decision on the label alone, a new verifier that tries several keys)
  — that turns today's harmless gap into a real hole; bind the identity before merging it.
