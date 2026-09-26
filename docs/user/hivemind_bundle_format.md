# Hivemind Bundle Format — Signature & Identity Registry

This page is the canonical, append-only registry of the **signature** vocabulary for Hivemind substrate bundles. It documents the recognized `signature_algorithm` values and the `signer_identity` field so that the 1.2 peer-to-peer Hivemind protocol's heterogeneous producers and consumers share one string vocabulary instead of guessing.

**`ed25519` is implemented and verified as of 1.2 (P2P Slice A); the other listed algorithms remain reserved documentation.** The `signature`, `signature_algorithm`, and `signer_identity` manifest fields were reserved-null through 1.1. Since 1.2, `maxim substrate export --sign` computes an `ed25519` signature (via the `[sign]` extra, `cryptography`) over the sig-excluded manifest plus the raw slice bytes (see [`hivemind/signing.py`](../../src/maxim/hivemind/signing.py)), and `maxim substrate ingest --require-signed --trust-key <id>=<pubkey>` verifies it and **refuses** a bundle whose signature is absent, of an unknown algorithm, from an untrusted signer, or invalid — refusal, never admit-with-clamps. Unsigned bundles still ingest when the receiver does not pass `--require-signed` (the experimental tier). This page still freezes the vocabulary so a producer/consumer share the algorithm strings; only `ed25519` has a validator today. See the [Auth Format-Freeze Audit](../plans/archive/auth_format_freeze_audit.md) (CC13) for the why.

## Manifest signature fields

Through bundle schema 2, a bundle's `manifest.json` carried three auth slots (all `None` at 1.0; a **legacy v1** signature lives in them). Schema-3 releases carry `signer_identity` only, and their signature in the `signature.json` member ([release format v2](#release-format-v2-bundle-schema-3)); bundles composed since then carry no null slots at all:

| Field | Type | Meaning |
|---|---|---|
| `signature` | `str \| None` | The detached signature material (encoding is algorithm-specific — see below). |
| `signature_algorithm` | `str \| None` | A value from the [recognized registry](#recognized-signature_algorithm-values). Tells a verifier how to interpret `signature`. |
| `signer_identity` | `str \| None` | The "who claims to have signed this" string, parallel to `contributor_id`. Lets a 1.1+ verifier bind a *verified* identity to the *claimed* `contributor_id` in a trust registry. |

Legacy fields round-trip through [`extract_bundle`](../../src/maxim/hivemind/bundle.py) / `read_bundle_manifest` unchanged. Older bundles written before `signer_identity` existed load cleanly — the field is optional and read via `.get(...)`.

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

## Manifest `encoder_provenance` field (1.1, artifact stamping)

Bundles composed from 1.1+ carry an additive `encoder_provenance` manifest key
(same additive-optional precedent as `signer_identity` — read via `.get(...)`,
absent on pre-1.1 bundles):

| Sub-field | Meaning |
|---|---|
| `observed_embedding_dims` | Per-modality **sorted set of dims measured on the actual arrays** in the shipped `ec.json` slice at write time. Never declared — a mixed-dim slice (the cross-space corruption class fixed in `ec_merge`) is visible here instead of discovered at merge time. `{}` for NAc-only bundles. |
| `recorded` | The source `ec.json`'s **encode-time encoder stamps** (`EC.record_encoder_provenance` — realized `using_fallback`, measured `embedding_dim`, sensor-name set, normalization modes), passed through verbatim. `None` for pre-stamping payloads — an honest unknown, **never** fabricated from the exporting process's own encoder (the exporter's encoder need not match the writer's). |

**Merge caveat (pinned for 1.2):** `recorded` describes the *composing
substrate's own encoders only*. A substrate that previously imported foreign
nodes via `ec_merge` ships arrays its local stamps do not describe; the 1.2
P2P merge must union provenance per-contributor rather than trusting a merged
substrate's local stamps. `observed_embedding_dims` is the measured backstop,
but dims alone cannot distinguish a 384-dim fallback from a real 384-dim model.

## Contract

- **Append-only.** Once an algorithm name or reserved prefix is listed here, its meaning does not change. New values may be added in a minor release (a non-breaking widening of the verifier's dispatch table).
- **Unknown → refuse.** A future verifier MUST reject an unrecognized `signature_algorithm` rather than skip verification or treat the bundle as unsigned. A malicious producer setting `signature_algorithm: "always_pass"` must not slip past a string-match dispatch.
- **`ed25519` validated since 1.2; the rest documentation-only.** `hivemind/signing.py` + `hivemind/bundle.py::verify_bundle_signature` implement and verify `ed25519`; every other listed algorithm and reserved prefix still has no validator and is refused as unverifiable when `--require-signed` is set (unknown → refuse, above). Through 1.1 no code path read, wrote, or validated these fields beyond round-tripping the manifest.
- **`signer_identity` vs `contributor_id`.** `contributor_id` is the free-form, producer-controlled provenance string (per-link / per-node, set at compose time). `signer_identity` is reserved for the *cryptographically attested* identity a 1.1+ verifier checks against a trust anchor. They are intentionally separate so a verified signer can be required to match the claimed contributor. Like `contributor_id`, `signer_identity` shares the reserved `_*` namespace discipline used across the Hivemind layer (see [`hivemind/merge.py`](../../src/maxim/hivemind/merge.py) `_validate_source`): a 1.1+ implementer must NOT reuse an existing sentinel such as `_consensus` (which already means "aggregated across contributors" in merge provenance) or `_identity` to express a multi-signer or attested-identity concept — pick a fresh, distinct value.

## Release format v2 (bundle schema 3)

A **signed** bundle is a release (`maxim substrate export --sign --license SPDX [--key-file PATH]
[--release-sequence N] [--agent-id ID]`). Only a release is schema 3: an **unsigned** bundle is written at schema 2, which it
needs nothing beyond, so 1.3.x peers and Oasis servers keep reading contributions — and refuse a release
they cannot verify.

- **`signature.json`** (a ZIP member): `{"signature_scheme": 2, "signature_algorithm": "ed25519",
  "signature": "<base64>"}`. It signs the raw, uncompressed bytes of **every other member**, ordered by
  UTF-8 name, each framed with an 8-byte big-endian length, after the domain tag `maxim-bundle-v2`. So
  the manifest is signed as bytes: nothing is written after signing, and a member added later fails.
- **Signed manifest fields:** `signer_identity`, `release_sequence` (an int, 1 ≤ n ≤ 2^53−1, strictly
  increasing per signer), `license` (an SPDX id; published bundles use `CDLA-Permissive-2.0`), and
  `entry_index` (`{"version": 1, "entries": [{"id", "modality", "digest"}, ...]}`).
- **An entry** is one situation cluster: its EC node (if the bundle has one) plus the NAc rows keyed by
  its id (`cluster_fear`, `cluster_reward_bias`, `cluster_reward_source`) and the inherent markers on
  them. Its digest is `sha256` of the RFC 8785 (JCS) serialization of that projection; `source` /
  `contributors` are left out, so the same entry from two exporters has one digest. `maxim substrate
  inspect --entries <bundle>` prints every entry's digest for any bundle.
- **Members:** a release is EXACTLY `manifest.json`, its declared slices and `signature.json` — no
  more, no fewer, names `[A-Za-z0-9._-]`, no duplicates, `signature.json` never declared as a slice.
  Every member is read at most once, under the V6 size caps enforced on the actual decompressed bytes;
  the member set is checked before any slice or undeclared member is decompressed.
- **One agent's learning.** A release ships only the exporting agent's NAc rows (`--agent-id` names it
  when the state holds several; rows filed under any other agent are dropped, with a count), under the
  agent segment `_agent`. Ingest re-keys the situation rows to `--receiver-agent-id`, which it therefore
  requires for a release. Percept valences, per-tool outcome stats and the node-keyed `reward_bias`
  cannot be re-keyed onto a receiver situation, and an agent reads only rows under its own id, so ingest
  drops such rows filed under another agent (`IngestReport.foreign_rows_dropped`).
- **Receiver state.** A receiver's ingestion journal records each verified release (signer public key,
  identity, scheme, sequence, signed-payload digest, license). From it, ingest refuses a second
  payload for an admitted `(key, sequence)` (equivocation) and a v1 bundle from a key whose v2 release
  it admitted (downgrade), and treats a re-zipped copy of an admitted release as the same release.
  Releases are additive: a lower sequence is not refused, and `hive pull` ingests ascending. The
  downgrade rule is not: a v1 bundle has no sequence, so once a key's v2 release is admitted, a v1
  bundle from that key arriving later is refused in that session even if it is an older lineage.
- **Producing and publishing.** A signing key's `release_sequence` comes from its counter
  (`~/.maxim/util/hive_release_sequence.json`, keyed by the public key), committed with the release, so
  a sequence is never re-used; keep the Queen key in its own `--key-file`. `maxim oasis publish
  --queen-key` verifies a release before it enters the release tier and ids it by its signed-payload
  digest.
- **A newly added Oasis refuses v1** (`hive add` writes `accept_v1: false`); `maxim hive trust <name>
  --accept-v1` takes its legacy v1 lineages. Entries registered before this change keep accepting v1.
- **Legacy v1** (schema ≤ 2, the signature in the manifest) still verifies until 2.0, against the
  manifest as stored. An unsigned bundle carries no signature fields at all.

## Provenance at export

A bundle's links and EC nodes carry `source` / `contributors`. A receiver accepts only the manifest's
own `contributor_id` or `"local"` there, so export decides what material it can honestly ship:

- **`maxim substrate export`** (a contribution) ships only **your own learning** — rows whose
  provenance is yours alone. Material you received from others (a donor source, `"_consensus"`, a
  foreign contributor list) is dropped, along with cluster-keyed NAc rows naming a dropped node, and
  the command prints how many rows it dropped.
- **`maxim substrate export --release --sign`** composes a **release** from merged contributions: every
  row is re-authored as yours (`source` and `contributors` = your `contributor_id`), and the signature carries the
  provenance. `--release` without `--sign` is refused.

Rows with no provenance fields (cluster fear and reward bias, outcome statistics, priors) cannot be
told apart, so a received fear folded into one of your clusters is exported as yours.

## See also

- [Substrate Sharing](substrate-sharing.md) — the user-facing export / import / merge workflow
- [Maxim Hivemind + Oasis](../hivemind.md) — the gated 1.2 Oasis/P2P roadmap this format feeds
- [Auth Format-Freeze Audit](../plans/archive/auth_format_freeze_audit.md) — CC13, the freeze decision behind this registry
- [Stable API](stable_api.md) — the broader 1.0 contract surface (including the `api_key_ref` URI namespace reservation)
