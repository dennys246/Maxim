# Oasis release format v2 — the Queen-signed entry index (design)

> **DECIDED 2026-09-25 (owner) — design v2, ready to build.** [public_oasis.md](public_oasis.md) Phase 0
> **item 7** ("Releases carry a Queen-signed entry index — owed by social_referencing"), carrying the
> deferred **signing scheme v2** ([deferred/signed_signer_identity.md](deferred/signed_signer_identity.md),
> trigger (a)) and the **license field** (item 4). It lands **before item 2** (the format freeze). v1 of
> this design had a two-lens design review (security + architecture;
> [reviews/oasis_entry_index_v2/](reviews/oasis_entry_index_v2/)): six DO-NOT-BUILD, fourteen SHOULD-FIX.
> The owner revised two decisions on it (rollback model; signature mechanism) and approved the rest.
> A re-read of v2 (both lenses, one reader) found two contradictions — the signed member set vs "an
> appended copy dedups", and the V8 key switch re-admitting pre-change releases — plus gaps (mandatory
> receiver re-keying, pipeline order, JCS, store migration, node-less entries inert on merge); all folded.
>
> **BUILD — PR A (format + verification) shipped 2026-09-25:** the envelope and members, the signed
> payload, the entry projection and index, the verifier (bounded, before V1, v1 before migration),
> `inspect --entries`, `--release-sequence` / `--license` / `--agent-id`, `hive pull
> --receiver-agent-id`. Two owner decisions made on its code review changed this design, recorded
> in the sections they touch: **own rows only** (§Agent segment) and **only signed releases are
> schema 3** (§Envelope). **PR B — receiver state — shipped 2026-09-25:** the journal records each
> verified release; ingest refuses equivocation and downgrade and dedups on the signed payload; the
> registry's `accept_v1` reaches ingest as `--refuse-v1`; `hive pull` ingests in ascending sequence
> (§Receiver state, §Registry: `accept_v1`). **PR C — producer and store — shipped 2026-09-25, and
> item 7 is BUILT:** a per-key release counter, named key files, and a store that verifies on publish
> with payload-digest ids (§Producer, §Oasis store, each with its amendments).

## Why

[social_referencing.md](social_referencing.md) design 3: an agent fetches **whole** signed releases in the
background, verifies them once, and a pure `hivemind` selector picks the entries that match its current
situation **locally**, admitting them **per entry**. Nothing in today's format names an entry, and the
signature can only vouch for the release as a whole. A server-cut slice (one entry served alone) stays
rejected: it would need the server's signature, not the Queen's.

Front-gate: this rides the existing bundle — no new store, no new route. An entry is a deterministic
**projection** of the slices the bundle already carries; the signed manifest lists each projection's
digest. The one new ZIP member is the detached signature.

## Decisions

**(a) One entry = one situation cluster** (owner): the cluster's EC node when the bundle has one, plus
every NAc row keyed by that cluster id — `cluster_fear`, `cluster_reward_bias`, `cluster_reward_source`
— and the `inherent_bias_keys` naming those rows. **Node-less entries are allowed** (a NAc-only lineage
such as the Exp 45 orient release carries cluster rows with no EC node; its entries have `ec_node: null`).
Rows not keyed by a situation — `links`, `priors`, `event_outcome_welford`, `goal_reward_bias`,
`percept_valences` — stay whole-bundle only and are never consult-selectable.

**(b) Ordering and anti-equivocation, not blanket rollback refusal** (owner, revised on review). Releases
are ADDITIVE (each carries more wants and fears; none replaces the last), so "refuse below the highest
sequence seen" would refuse legitimate releases (a first pull ingests newest-first). Instead:
- `release_sequence` (signed, per signing key, strictly increasing at the producer) **orders** releases;
  `hive pull` ingests in ascending sequence.
- **Equivocation refusal:** one `(signing key, sequence)` binds to one signed-payload digest; a second,
  different release claiming the same pair is refused (by the store at publish, and by a receiver that
  has already admitted the first).
- **Dedup on the signed-payload digest** (V8), not the ZIP bytes — a re-zipped copy (same members,
  recompressed or reordered) is the same release and is not summed twice. A copy with an APPENDED member
  is not the same release: it fails verification (every member is signed). The journal's existing
  entries key on the ZIP sha256, so `has_digest` checks **both** keys (a pre-change release re-pulled is
  not admitted twice); a v1 signed bundle's payload digest is its v1 signed payload's sha256, an unsigned
  bundle keeps the ZIP sha256.
- **Per-entry supersession is reserved in the format now** for social_referencing S1: an entry `id` is
  stable across one signer's releases, its digest is that entry's version, and a higher sequence's
  version supersedes a lower one. Supersession is keyed by **`signer_identity`** (it survives a key
  rotation); equivocation is keyed by **public key** (a rotated key starts clean). S1 implements it; the
  freeze does not reopen for it.

**(c) Signing scheme v2; v1 legacy-only** (owner, tightened on review). v2 signs the whole manifest,
including `signer_identity`. A **new** registry entry refuses v1 (`accept_v1: false`, the default for
`hive add` from this change on — the public Oasis has no v1 history, so a first-contact client cannot be
downgraded); existing entries keep `accept_v1: true` so the Exp 56/61 lineages keep verifying, until
2.0 removes v1. An unknown `signature_scheme` is refused, never read as v1. v1 is accepted only when the
manifest **as read** has `schema_version ≤ 2`.

**(d) A signed `license` (SPDX id)**, required on every signed release; published bundles use
**`CDLA-Permissive-2.0`** (owner, item 4). Plain unsigned exports default to `null`. A signed bundle is
a release artifact and must carry one — the security lens's view, taken over the architecture lens's
"signing is not publishing": a signed bundle without terms is exactly the artifact that loses them when
passed on. The Oasis store refuses to publish without one. Ingest journals the license per
release; a `--release` composition whose inputs carry a non-permissive license warns. No automatic
license-compatibility logic.

**(e) A detached signature over raw bytes** (owner, on review). No canonical JSON in the signature.

## The format

### Envelope and members

`schema_version` **3**. New ZIP member **`signature.json`**:
`{"signature_scheme": 2, "signature_algorithm": "ed25519", "signature": "<base64>"}`.
A v3 manifest carries no `signature` / `signature_algorithm` fields (they live in `signature.json`).
*Amended on PR A's review (owner):* only a signed release is schema 3; an unsigned bundle is written at
schema 2 (it needs nothing schema 3 added), so 1.3.x peers and Oasis servers keep reading contributions
and refuse only a release they cannot verify.

**Verification reads the manifest as stored, before any migration** (both review lenses: the v2→v3
migration rewrites `schema_version`, which the v1 payload signs, so verifying the migrated manifest
would refuse every existing signed release). The migrated manifest is only consumed afterwards.

New signed manifest fields: `signer_identity` (now inside the signature), `release_sequence` (int, not
bool, 1 ≤ n ≤ 2^53−1), `license` (SPDX charset, ≤ 64 chars), `entry_index`
(`{"version": 1, "entries": [{"id", "modality", "digest"}, ...]}`, sorted by `id`, ids unique and
node-id charset, count within the V6 caps). Unknown keys inside an index entry are **signed and
ignored** — S1's future fields do not reopen the freeze.

### The signed payload

```
b"maxim-bundle-v2"
for each ZIP member except signature.json, sorted by UTF-8 name bytes: name, UNCOMPRESSED member bytes
```
Member rules (refused otherwise): names are ASCII from `[A-Za-z0-9._-]` (no paths, no case games, no
cp437/UTF-8 name-decoding divergence); **no duplicate central-directory names** (`zipfile.read` returns
the last copy, so a signer and a consumer could see different bytes); the manifest may not declare
`signature.json` as a slice. Every member is signed, so a member appended after signing fails
verification. Hashing uncompressed bytes makes the payload digest stable across re-zips.
Each part is framed with an 8-byte big-endian length prefix, as v1 is. The domain tag differs from v1's
`b"manifest"`, so neither signature can be replayed as the other. Signing raw bytes removes every
canonicalization hazard (`default=str` stringifying numpy floats, NaN, a lone surrogate crashing
`ensure_ascii=False`, duplicate-key parser divergence) and lets a non-Python verifier check a release.
The **signed-payload digest** (`sha256` of those bytes) is the release's identity for dedup and
equivocation.

### Agent segment normalized at export

The exporter rewrites the agent segment of every NAc composite key (`agent␟cluster␟x`,
`agent␟entity␟x`, …) to a fixed token before signing — the owed `agent_id` normalization, which also
stops local agent ids shipping (`taught.zip` carries `donor_taught_42` today). An exporter holding rows
for more than one local agent id **refuses** to export (it cannot normalize without collapsing them).
*Amended on PR A's review (owner, "own rows only"):* the exporter ships ONE agent's rows — the single
real id, or `--agent-id` when there are several — and drops rows under any other id (including the
token itself, which an ingested release leaves) with a count; relabelling them shipped a −0.9 valence
as +0.5 in a review probe. Ingest drops the non-situation agent-keyed rows (percept valences, outcome
stats, node-keyed `reward_bias`) filed under another agent, since re-keying covers situation rows
only; transferring those is deferred ([deferred/transfer_non_situation_nac_rows.md](deferred/transfer_non_situation_nac_rows.md)). A
v2 verifier refuses any other agent segment, so one digest can never cover two values. The token makes
re-keying MANDATORY at the receiver: `rekey_nac_state` keeps the token when `to_agent_id` is absent, and
NAc reads filter by agent id, so every value would silently read 0.0 (D43-class). So ingesting a
token-keyed bundle without `receiver_agent_id` is **refused**, and `hive pull` forwards the operator's
`--receiver-agent-id` (shipped in PR A).

### The entry projection (what a digest covers)

```
{"id": <cluster id>,
 "ec_node": null | {"modality", "embedding", "geometry", "count", "domain"},   # NOT source/contributors
 "nac": {field: {"<cluster>\x1f<third>": value, ...} for the three cluster-keyed fields},
 "inherent": sorted(inherent_bias_keys naming those rows)}
```
Built from the slices **as parsed from the signed bytes**, then serialized with **RFC 8785 (JCS)** →
`sha256` → `"sha256:<hex>"`. JCS via the `rfc8785` package, added to the `[sign]` extra that
verification already needs (a hand-rolled JCS is only acceptable pinned to RFC 8785's number test vectors
— ECMAScript number formatting is not Python's `repr`). `count` uses ingest's `count` / `member_count`
fallback. `source`/`contributors` are excluded (the exporter re-stamps them), so the
same entry from two exporters has the same digest.

### What the verifier refuses (signed v2 only)

In order, and **moved ahead of V1 / gate 7 / V8** (which run before the signature today): V6 caps →
signature (scheme dispatch) and payload digest → index → V1 → gate 7 → V8 (on the payload digest). Index checks: unknown `version`; duplicate,
unsorted or badly shaped ids; an index `modality` that differs from its node's; a non-3-part cluster key
(including `cluster_reward_source`); an agent segment other than the token; a dangling inherent marker;
**any cluster-keyed row or EC node not covered by exactly one entry**; a recomputed digest that differs.
Every failure is an `IngestRefused`, never a crash: `_loads_strict` refuses duplicate keys
(`object_pairs_hook` — the detached signature does not stop two parsers reading a duplicate-key manifest
differently), and non-finite floats and lone surrogates are refused before JCS.
Unsigned bundles are not held to the index rule (unsigned compose stays byte-identical in content), and
their index — if any — has no authority: anything that dedups per entry uses **recomputed** digests,
never an index's claim.

## Receiver state (anti-equivocation, v2-seen)

**No new state file and no runtime writes to `hive.json`** (it is declarative by design). Each
`IngestionJournal` entry of a signed bundle records `signer_key` (the decoded public key, hex),
`signer_identity`, `signature_scheme`, `release_sequence`, `payload_digest` and `license`. Two rules are
derived from it inside `ingest_bundle`, from its existing required `journal` parameter (`hive pull`
reaches ingest only through argv, so nothing else can carry them): **equivocation** (a second payload for
a `(key, sequence)` already admitted → refuse) and **downgrade** (once a v2 release from a key has been
admitted, a v1 bundle from that key → refuse). Keyed by **public key**, so two mirrors and `hive
remove`/`add` behave. Limits, stated: the journal is per receiver session directory
(`substrate_ingest_journal.json`), not registry-wide; and a release that verified but was refused by a
later gate (e.g. gate 7) is not journalled, so it does not seed either rule. *Amended on PR B's review:* dedup
also keys on a payload identity computed over exactly what ingest reads (`content_payload_digest`:
manifest + declared slices, equal to the verified digest when the bundle verifies), so a release first
admitted unverified, or re-packaged, is not merged twice; only verified
entries carry signer fields, so unverified admissions seed neither ordering rule. The downgrade rule
refuses every v1 bundle from a key once its v2 release is admitted — including an older lineage that
arrives later, since v1 has no sequence (owner kept it; a `created_at` narrowing is backdatable by the
key holder it guards against). An older maxim reading the new journal/registry fields ignores them. `_run_pull`'s pre-screen
reads `signature.json`, not the manifest's `signature` field (empty in v3).

## Producer

- `compose_bundle(..., release: SignedRelease | None = None)`, where `SignedRelease` is a frozen value
  `(signer, release_sequence, license)` — forgetting a field is a TypeError at construction, not a
  runtime default. (Runtime-ephemeral: passed in, never persisted.) `reauthor` stays as #903 left it.
- `signing.py` gains a named key path (`--key-file`), so the Queen key and the host's development key
  are separate files, not one default.
- The sequence counter lives in `~/.maxim/util/hive_release_sequence.json`, keyed by signer identity
  (amended below: by public key),
  written with `atomic_write_json` + `_format_version`. Signing **refuses** when a key exists but its
  counter entry does not (a restored key must not restart at 1) unless `--release-sequence N` is given;
  `--release-sequence` may only move the counter forward. The Queen key is kept separate from the
  host's development key (experiment bundles must never advance the Queen's sequence).
- `maxim substrate export --sign` builds v2; `--license` is required with `--sign`.

*Amended on PR C's build and its review:* the counter is keyed by the signing key's PUBLIC KEY (hex),
not the identity — the same key the receiver journal keys on, so the Queen key and a development key can
never share a counter even under one identity. A key minted on this host is registered in the counter at
`0` the moment it is minted (so "has this key released before?" is counter state, not a per-process
flag); a key the counter never saw (minted elsewhere, restored, copied) must name its first
`--release-sequence`. The producer gets its release from `signing.counted_release`, whose commit
(`commit_release_sequence`, under a `FileLock`, re-checking it still moves forward) runs INSIDE
`compose_bundle`, between writing the signed `.tmp` and moving it onto the output path: a signed release
at its output path always has its counter record, whatever crashes when; a failed compose burns no
number. `export --release` warns when the session merged inputs under a license outside a deliberately
attribution-free set (CDLA-Permissive-1.0/2.0, CC0-1.0 — a release strips per-row provenance) or under
no license at all (decision (d); a warning, no compatibility engine).

## Oasis store

`publish_release` **verifies** (the store is configured with the Queen public keys: `maxim oasis serve
--queen-key IDENTITY=PUBKEY` — amended below: `oasis publish --queen-key`): signature, index, license present, and no equivocation against releases
it already holds. A Queen release's id becomes its signed-payload digest (same `^[0-9a-f]{64}$` shape);
releases already on disk are renamed once (a store migration), and `test_hive_pull_e2e.py`'s direct
`{sha256(raw)}.zip` writes change with it. The experimental tier keeps ZIP-sha ids — one id shape, two
meanings by tier, stated. Clients only shape-check ids, so none breaks. `list_releases` summaries carry `signature_scheme`,
`release_sequence`, `license` (shipped in PR A); clients use them for ordering only, never trust. *Amended on
PR C's build:* the Queen keys are given to `maxim oasis publish --queen-key IDENTITY=PUBKEY` (publishing is
where verification happens; `serve` only serves what was published). The store takes v2 releases only (a
v1 bundle has no index or license); a re-zipped copy of a held release is idempotent; the equivocation
check is `bundle.find_equivocation`, the same predicate the receiver journal applies, over held releases
verified by KEY BYTES under every given key (relabelling a key cannot hide its history), serialized with
the write under a lock; a file at the id that does not verify is replaced on publish; the migration
(`OasisStore.migrate_release_ids`, run by `serve`/`publish`; `status` only reports) renames by
`content_payload_digest`, which needs no key, and never deletes a DIFFERENT file sharing an identity.

## Reader (the Phase-1 caller)

`maxim substrate inspect --entries <bundle>` projects any bundle — including the unsigned schema-1/2
gated evidence — and prints each entry's id and recomputed digest. It is what shows, mechanically, that a
published release carries the same entries as the gated `taught.zip` it was built from (gated ZIPs are
never rewritten, so their recorded sha256 values hold). The verifier is the other real caller: `hive
pull` refuses any index that does not match. Per-entry admission and local selection remain S1's.

## Node-less entries are indexed, not merged

A node-less entry is covered by the index (so a NAc-only release is fully signed per entry), but on
ingest it is **inert, exactly as today**: with no EC node there is no `id_map` entry, and
`rekey_nac_state` drops the row. The Exp 45 orient release reaches robots by loading its merged NAc
directly (`live_3_learn.py`), not through ingest. Making node-less entries mergeable is out of scope, and
S1's selector must know they cannot be admitted.

## Registry: `accept_v1`

Absent means `true` (legacy entries); `hive add` writes an explicit `false`; it joins `POLICY_FIELDS`,
fail-loud as a bool; `hive trust --accept-v1` toggles it. `hive remove` then `add` flips an entry to
`false` — stated.

## Out of scope (named, with owners)

- **Per-entry journal and selection** — social_referencing S1, with the DECISIONS.md amendment to the
  ingest contract §1 (exactly-once per entry digest for indexed bundles).
- **Fetch-by-hash / centroids in the index** — only if a route ever serves entries separately.

## Owner decisions recorded here (2026-09-25)

- **License:** `CDLA-Permissive-2.0` for published bundles (item 4). Minecraft EULA vs trained state:
  not examined, not a known blocker.
- **Public ids:** role-based, never personal — signer `maxim-queen`; contributor ids per lineage
  (`exp56-taught-want`, `exp61-survival-fear`).
- **Privacy-read exemplar (item 3):** the Exp 56 **seed-42 1.20.4 re-baseline** (the plan's "seed 43"
  lineage exists nowhere — stale text), plus one Exp 61 `fear.zip` pulled from the rig and checked
  against `exp61_pairs.jsonl`'s sha256. The read itself is the owner's.

## Guards (to build with the code)

- A v2 release verifies; changing any manifest byte, slice byte, or adding/removing a member fails.
- `signer_identity` relabelled → fails (inside the signature now).
- v1↔v2 replay never verifies (domain tag); an unknown scheme is refused, never read as v1.
- **A schema-2 bundle signed under v1 still verifies after the bump** (verify-before-migrate).
- A new registry entry refuses v1; an existing one accepts it with a warning.
- A re-zipped / member-appended release dedups (payload digest); an equivocating release is refused.
- Agent-segment normalization: any other segment is refused; two exporters' same entry → same digest.
- Every index refusal above, each as an `IngestRefused`; a lone-surrogate / NaN slice refuses, not crashes.
- A node-less (NAc-only) release verifies with `ec_node: null` entries.
- `SignedRelease` with a missing field raises; signing with a key but no counter refuses.
- The store refuses an unsigned, unlicensed, mis-indexed or equivocating publish.
- `substrate inspect --entries` gives the same digests for `taught.zip` and a release built from it.
Each proven by deleting its mechanism.

## Docs that change with the code

`docs/user/hivemind_bundle_format.md`; `docs/agents/bio-memory.md` (the signature invariant's payload
line, plus an entry-index invariant with its `Regression guard:`); `docs/agents/persistence-config.md`
(the counter file); `deferred/signed_signer_identity.md` (revived); `public_oasis.md` items 6/7;
`CHANGELOG [Unreleased]`; a new **DECISIONS.md** entry for scheme v2, ordering/anti-equivocation and the
license field (owned by this change; S1 owns only the §1 per-entry amendment).
