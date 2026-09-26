# Security lens — oasis_entry_index_v2.md (2026-09-25)

Read against `signing.py`, `bundle.py` (PR #903 still OPEN; read from its diff), `ingest.py`, `hive_cli.py`, `registry.py` and `store.py`. The domain tag is sound: `b"maxim-bundle-v2"` vs `b"manifest"` in a length-prefixed frame stops v1↔v2 replay.

## DO-NOT-BUILD

**D1 — The migration breaks every existing signed release.** The v2→v3 migration must set `schema_version=3`. Ingest verifies the MIGRATED manifest, and v1 signs `schema_version`. So every Exp 56/61 release fails, and "v1 accepted until 2.0" is false. *Fix:* verify the manifest as read, before migration.

**D2 — The rollback rule refuses legitimate releases and relies on a dedup that can be dodged.** Releases add to each other (want, fear); they do not replace each other. `list_releases` sorts newest-first, so on a first pull `_run_pull` ingests N and then refuses N−1. `--domain` pulls break the same way. The design allows an equal sequence because "the journal dedups", but the journal hashes the unsigned ZIP bytes. A re-zip, or an appended undeclared member, replays the same release and sums its counts again (row J). *Fix:* bind (signer key, sequence) → signed-payload digest. Refuse an equal sequence with a different payload as equivocation. Key V8 dedup on the payload digest. The stale-entry threat (prior D3) needs signed per-entry supersession, or ingest sorted by sequence, not a blanket "below highest".

**D3 — Dropping the agent segment lets two rows share one digest.** `a1␟c␟x` and `a2␟c␟x` collapse to one key; #903 says donor agent ids survive in these keys. The digest then attests to only one value, and an `a2` inherent marker appears to cover `a1`'s row. *Fix:* the exporter canonicalizes the agent segment before signing, and the verifier refuses a collision.

## SHOULD-FIX

**S1 — First-contact downgrade.** A fresh client, or a `--trust-key` ingest, has no `v2_seen`. *Fix:* accept v1 only when the as-read `schema_version ≤ 2`. Refuse an unknown `signature_scheme`; never fall through to v1. Refuse v1 outright on the public Oasis, which has no v1 history (a registry `accept_v1` flag, false for new entries).

**S2 — State keying.** Key the state by the decoded public key, across the whole registry, not by (url, identity). Otherwise two mirrors each start at 0, and a rotated key that reuses `maxim-queen` gets its releases refused. Advance the state after signature + index verification in an `--apply` run, not after ingest: a verified release refused by gate 7 still proves N exists. Pass the state to ingest as a required keyword (D43; `hive_cli` reaches ingest only through argv). Write `hive.json` locked and atomically.

**S3 — Counter integrity.** A key restored without its counter restarts at 1. The same host key signs experiment bundles (prior D4), so one leaked test bundle with a high sequence locks clients out. *Fix:* refuse to sign when the key exists but the counter does not, unless `--release-sequence` is given. Keep the Queen key and the dev key separate.

**S4 — The store must verify.** `publish_release` only checks that a signature is present. Give the store the Queen pubkey. It verifies the signature and index, refuses a duplicate or lower (signer, sequence), and addresses releases by payload digest. Clients may use `list_releases` summaries only for ordering.

**S5 — The index is authoritative only when verified.** On the `allow_unsigned` path the index is never checked. If the S1 journal trusts digests an index claims, an unsigned bundle can pre-claim a Queen entry's digest and get it deduped out. *Fix:* key the journal on recomputed digests.

**S6 — Projection identity.** Nodes carry no id, so identical nodes share a digest. `source`/`contributors`, which #903 restamps per exporter, break the "same digest across exporters" guard. *Fix:* include `id` and list the node fields the projection covers.

**S7 — Index checks.** The verifier itself refuses:
- an unknown `version`;
- duplicate, unsorted or badly shaped ids (`_NODE_ID_CHARSET`, V6 cap);
- an index `modality` that differs from the node's;
- non-3-part cluster keys (`cluster_reward_source` is unchecked today);
- dangling inherent markers.

Order: V6 caps → signature → index, parsed with `_loads_strict`. A lone `\ud800` under `ensure_ascii=False` raises `UnicodeEncodeError`, a crash instead of an `IngestRefused`.

**S8 — Canonicalization.** v2 drops `default=str` and NaN. Manifest and projection share one `canonical_json` (`allow_nan=False`, `TypeError` at compose). Digests are computed from re-parsed bytes, because `np.float32` turns into a string. Stronger: sign the raw `manifest.json` bytes with a detached signature, which also ends duplicate-key parser divergence. Otherwise say the verifier is Python-only, or adopt RFC 8785.

## NIT

**N1** — `release_sequence` must be an int that is not a bool, ≤ 2^53−1. `signature_scheme` must be exactly `2`.

**N2** — Require a license on signed releases, capped for SPDX charset and length (it is displayed). Journal the license per ingest, so that `--release` re-authoring does not relabel differently-licensed material.

**N3** — Guards: a re-zipped release is deduped; a migrated v1 release verifies; an agent-segment collision is refused.
