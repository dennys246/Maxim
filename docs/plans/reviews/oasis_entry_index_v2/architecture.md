# Architecture lens — oasis_entry_index_v2.md (2026-09-25)

The scope is right: the entry is a projection over the existing slices, with no new store or route. Three things must change before build.

## DO-NOT-BUILD

**A1. Rollback state must not live in `hive.json`** (§Verification). `persistence-config.md` makes `hive.json` declarative and CI-grep-exempt *because* it has no runtime writer, and `registry.py` says the same. The design also fails on its own terms:
- `hive remove` then `hive add` wipes the state.
- Two mirror names give two counters, so a replay through one of them passes.
- `--trust-key` ingests get no guard at all.

Fix: add no new state. Record `signer_identity`, `signature_scheme`, `release_sequence` and `license` in each `IngestionJournal` entry. That journal is already per-receiver, atomic, and written after admission. Derive `highest_sequence`/`v2_seen` from it, keyed by identity rather than URL. That closes both gaps and answers question 2.

**A2. The v2→v3 migration breaks every existing signed release.** `ingest_bundle` verifies the *migrated* manifest. The migration must set `schema_version: 3`, and that field is inside the v1 payload. So "v1 accepted with a warning" actually becomes "v1 refused as tampered". Fix: verify the raw pre-migration manifest and consume the migrated one. Guard: a schema-2 bundle signed under v1 still verifies after the bump.

**A3. Dropping cluster rows from NAc-only bundles guts real lineages.**
- `docs/experiments/data/45_queen_mind_orient_v0_1.zip` is NAc-only, and its entire payload is 8 `cluster_reward_bias` rows.
- `orient_merge_arm.py` composes with `ec_substrate_nodes=None`.
- The Exp 56/61 `dangling` arm deliberately ships EC-less cluster rows.

Fix: apply "no unindexed state" only when verifying signed v2 bundles, and leave unsigned compose byte-identical. Signed compose drops rows whose cluster was filtered out, and reports the drop. Before the freeze, decide how an EC-less (orient) release is expressed, or state that it cannot be one.

## SHOULD-FIX

**S1. The type, not a keyword.** "Required with a signer" is a runtime `ValueError`: every `compose_bundle` parameter has a default. Replace `signer=` with `release: SignedRelease | None`, a frozen value holding the signer, sequence and license, with a CC3 docstring. Signing callers today:
- `cli.py` export;
- 24 `signer=` sites in 8 test files.

No script signs. The exp56/exp61/orient harnesses export unsigned and are unaffected.

**S2. The counter.** It changes on every export, so it belongs in `~/.maxim/util/`, not `~/.config/maxim/`. Key it by identity, not by key file: rotation means deleting the key (`signing.py`), which would reset the counter to 1 and make every receiver refuse. The equality rule needs the digest: the same sequence with the same digest is allowed; the same sequence with a different digest is refused. Add a producer-side caller: `oasis publish` refuses a sequence at or below the store's max for that signer.

**S3. Normalize the agent segment in the slice.** Dropping it only in the projection collides `a1␟c␟x` with `a2␟c␟x`, so one digest covers two values. The shipped keys also leak local ids: taught.zip carries `donor_taught_42`. Rewrite the segment to a fixed token in the export scrub; this is the owed `agent_id` normalization. Receivers already rekey through `rekey_nac_state(to_agent_id=…)`. The projection then becomes a plain row filter.

**S4. The D43 framing.** The verifier is a real non-test caller (`hive pull` → `--require-signed`), but nothing reads the entries. Call the index a truthful format reservation. A cheap Phase-1 consumer exists: `substrate inspect --entries`, which projects any bundle (including unsigned schema-1/2 evidence). That shows mechanically that the published release carries the same entries as the gated taught.zip. Gated zips are never rewritten, so their sha256 values hold.

**S5. S1 needs supersession, reserved now.** Keyed on digest alone, a changed cluster produces a new digest, and overlapping releases then double-count. Freeze these rules now:
- an entry `id` is stable across one signer's releases;
- the digest is that entry's version;
- a higher `release_sequence` supersedes a lower one.

Also put `id` inside the projection, and freeze an ignore-but-signed rule for unknown entry keys, so S1's fields do not reopen the freeze.

**S6. License.** Default to `null` on every export, including `--sign`, because signing is not publishing. `oasis publish` refuses a `null` license. Record the license in the journal (A1), so a later export can warn when one of its inputs is not permissive. No automatic compatibility logic.

## NIT

- Name the canonical JSON exactly, float repr included: either RFC 8785 or "Python `json.dumps`". Non-Python producers are anticipated.
- Add `signature_scheme`, `release_sequence` and `license` to `store._SUMMARY_KEYS`.

## Docs and order

These change in the same PR:
- `docs/user/hivemind_bundle_format.md`
- `bio-memory.md` — its signature invariant's payload line is wrong under v2; add an entry-index invariant with a `Regression guard:`
- `persistence-config.md` (where the counter lives)
- `deferred/signed_signer_identity.md` (mark revived)
- `public_oasis.md` items 6/7
- `CHANGELOG [Unreleased]`
- a new DECISIONS entry for scheme v2, rollback and license, owned by this change. S1 keeps only the §1 per-entry amendment.

Order: this change and its fold land first; then item 2's freeze pass reads the v3 shape, S5 included.
