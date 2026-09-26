# Public format 1 — the freeze record

**Status:** FROZEN 2026-09-26 (public_oasis Phase 0 item 2, part 2). Part 1, the pre-freeze hardening,
shipped as #915; the two ingest defects it surfaced are #914 (merged, #917) and #913 (scheduled, 1.3.1
— a merge-layer fix that changes no byte of the format).

Once a stranger downloads a release, its bytes are a public promise: every later Maxim must still
verify and ingest them, and must not change what it writes by accident. This file says what that
promise covers, how it may change, and what it does NOT promise.

Guard: [tests/unit/test_public_format_freeze.py](../../tests/unit/test_public_format_freeze.py) over
the deterministic fixtures in [tests/fixtures/public_format_1/](../../tests/fixtures/public_format_1/)
(built by its `build.py` from a public, test-only key — NOT a Queen key).

## What is frozen

**The two bundle shapes.** A zip `substrate_bundle` with members `manifest.json`, `nac.json`,
`ec.json`, and — signed only — `signature.json`:

| shape | `schema_version` | payload framing | who writes it |
|---|---|---|---|
| signed release (scheme v2) | `3` | v2 | `maxim substrate export --release --sign`, the Oasis release tier |
| unsigned bundle | `2` | v1 | plain `maxim substrate export`, the experimental tier, the Exp 56/61 harnesses |

**The manifest**: the field set the fixtures carry (`kind`, `schema_version`, `_format_version`,
`contributor_id`, `body_ref`, `created_at`, `contents`, `capability_map`, `affordance_namespace`,
`domain`, `encoder_provenance`, `identity_filter_applied`, `identity_threshold`, `license`; signed
releases add `signer_identity`, `release_sequence` and `entry_index`), read by one strict parser
(finite floats, integer schema fields). The fixtures pin these in the shapes they carry; a populated
`encoder_provenance.recorded` block and an identity-filtered export are not exercised by them.

**The signature** (`signature.json`: `signature`, `signature_algorithm: "ed25519"`,
`signature_scheme: 2`) is detached, over the **v2 payload** (`signing.py::bundle_signing_payload_v2`):
a sequence of parts, each framed by an 8-byte big-endian length prefix — first the tag
`maxim-bundle-v2`, then, for every member except `signature.json` in UTF-8 name-byte order, the member
name followed by its raw uncompressed bytes. No canonical JSON is involved in what is signed.
`release_sequence` is an integer in `[1, 2^53 − 1]`.

**The payload identity** (release id, dedup key; `bundle.py::content_payload_digest`) is the SHA-256
of the manifest plus the declared slices in that framing — v2 framing for a schema-3 manifest, the v1
framing (manifest minus its signature fields) for schema 2. It needs no key and grants no authority.

**The entry index** (`version: 1`): one entry per cluster — every EC node, plus every cluster that
only NAc rows name — as `{id, modality, digest}`. The digest is `"sha256:" + SHA-256(JCS(projection))`
(RFC 8785) over the projection `{id, ec_node: {modality, embedding, geometry, count (else
member_count), domain}, nac: {field: {cluster ␟ signature: value}}, inherent: [sorted markers]}` for
the fields `cluster_fear`, `cluster_reward_bias`, `cluster_reward_source`; the agent segment is not in
the projection.

**The agent token.** In a signed release every agent-keyed field (the cluster-keyed rows,
`inherent_bias_keys`, `percept_valences`, the Welford rows, `reward_bias`, link `event_context`) names
the agent as the token `_agent`. At ingest a receiver re-keys the situation rows (with their
`inherent_bias_keys`) and the links to itself; the other agent-keyed rows (`percept_valences`, the Welford rows, `reward_bias`) cannot land on
a receiver situation and are dropped (`keep_agent_rows`), not re-keyed. *(Corrected 2026-09-26, #913:
the frozen text said every field is re-keyed. A documentation error, not a format change.)*

**The slice shapes**: `nac.json`, its links and `ec.json` nodes carry only the allowlisted fields
(#915; `bundle.py::_BUNDLE_NAC_FIELDS`, `_BUNDLE_LINK_FIELDS`, `_BUNDLE_EC_NODE_FIELDS`, and
`event_context` only `agent_id`); NAc cluster keys are `agent ␟ cluster ␟ signature` (`\x1f`); node
ids match `^[A-Za-z0-9_.\-]{1,128}$`; link ids are derived (`sha256(event ␟ outcome)[:16]`), never
copied; store release ids are the 64-hex payload identity.

**The §5 adapter constants** a reader enforces on what it admits: the archive limits
(`MAX_BUNDLE_ENTRIES` 16 members, 64 MiB per member, 128 MiB total) and the value bounds in
`ingest.py` (`MAX_NODES_PER_SLICE`, `MAX_FOREIGN_COUNT`, `MAX_FOREIGN_TOTAL_OBSERVATIONS`,
`CAP_FOREIGN_CONFIDENCE`, `FOREIGN_FEAR_DISCOUNT`, `MAX_FOREIGN_DELTAS`, `MAX_FOREIGN_EMBEDDING_NORM`).
Tightening one can refuse a published bundle, so each is a recorded decision.

**Pinned by** the golden file (payload identity, entry digests, the SHA-256 of every member) and
`test_the_frozen_constants` (every name and value above, the allowlists included). Both directions are
checked: the published bytes still VERIFY and INGEST — each frozen row kind read back as published —
and today's producer still COMPOSES them byte for byte.

## The change rule

A format change is allowed. It is a **recorded decision**, never a side effect:

1. a line in §Freeze log below (date, PR, what changed, whether published bytes still read);
2. a dated amendment to [sharing_threat_model.md](sharing_threat_model.md) §5 if any receiver
   validation duty changes;
3. a regenerated fixture (`python tests/fixtures/public_format_1/build.py`) in the SAME commit.

This sits on top of the versioning contract in sharing_threat_model.md §2 and does not replace it:
an **additive** manifest key still needs no `schema_version` bump (readers `.get` it with an honest
default) — but because the producer is pinned byte for byte, adding one is still a freeze-log line and
a regenerated fixture. A new slice still needs a schema bump.

**Readers keep their promise:** a change that stops a published fixture from verifying or ingesting is
a breaking change. It needs a new `schema_version` / `signature_scheme`, with the old one still read
until the horizon below.

Not a format change: merge semantics on the receiver — how admitted rows fold into a store that
already holds state (#914, #913). The ingest test reads the fixtures into an EMPTY receiver, so it
pins what a reader accepts, not how it merges.

## The compatibility horizon

Every `1.x` release reads public format 1 — both shapes — through `verify_bundle_zip` /
`ingest_bundle` (and so `maxim substrate ingest` and `hive pull`) as published. Dropping either shape
is a **major-version** decision, announced in the CHANGELOG at least one minor release ahead.
Producers add a new shape only with a new `schema_version`.

Scheme v1 is NOT part of public format 1: it is read only where a caller or registry entry sets
`accept_v1` (with a deprecation warning; a newly added Oasis defaults to refusing it), and it is
refused from 2.0. Unsigned (schema 2) bundles are identified in the v1 framing, which serializes the
manifest with `json.dumps(default=str)` — the non-canonical encoding v2 was designed to avoid; their
identity is for dedup only and carries no trust.

## Declared, not promised — for the item 3 privacy read

The freeze promises bytes, not privacy. These properties of format 1 must be judged on the actual
exemplar (public_oasis item 3) before the first upload; none is fixed here.

- **Linkability across releases.** `contributor_id` is a stable public identity in every manifest;
  `signer_identity` in every release. Below that, EC node ids, cluster ids and the derived link ids
  are stable across one agent's releases, so releases link as one agent even where every key reads
  `_agent`.
- **Timing.** `created_at`, link `last_observed` and `temporal_delta`, and the `release_sequence`,
  reveal when an agent learned and how often it publishes.
- **Activity volume.** `total_observations`, link `observation_count` / `prediction_history`, node
  `count`.
- **Merge lineage.** Per-link and per-node `source` / `contributors` name the contributors whose
  material an agent merged.
- **Text in keys.** Cluster, percept and goal keys, event signatures (`tool:use:<action>` tails after
  redaction), `body_ref`, `capability_map`, and `encoder_provenance.recorded` tokens (model and sensor
  names survive the redaction).
- **Text-embedding inversion.** EC embeddings ship in full; an embedding of sim-derived text can be
  partially inverted back toward that text.
- **Agent ids in unsigned bundles.** Only signed releases re-token the agent: an unsigned (schema 2)
  bundle carries the producing agent's id in every agent-keyed field (`␟`-keyed cluster rows,
  `:`-keyed `reward_bias`, link `event_context.agent_id`), and — since the own-rows filter applies to
  signed releases — possibly other agents' ids too.

## Freeze log

| date | PR | change | published bytes still read? |
|---|---|---|---|
| 2026-09-26 | this PR | public format 1 frozen (signed schema 3 / scheme v2, unsigned schema 2) | — (baseline) |
