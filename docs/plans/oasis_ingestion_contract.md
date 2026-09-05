# Oasis ingestion contract (1.2 build, first piece)

**Status: ADOPTED 2026-09-05** (design pass + implementation in the same PR series, per
the roadmap's sequencing decision 1 — tighten-only merge lands WITH the V1–V10 adapter).
Authority above this document: [sharing_threat_model.md](sharing_threat_model.md) §5 is
the frozen receiver validation contract this adapter implements; §4 is the attack matrix
each duty answers. This document pins what §5 leaves as *adapter decisions*: the receiver
states, the step order, the operator surface, the constants, and where the
poison-resistance slice's tighten-only clamp and inherent-class admission rule sit in
that order ([coding_habits_oasis.md](coding_habits_oasis.md) §4;
[roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md) §"The 1.2 poison-resistance slice").

## 1. What a bundle merges INTO

**The persisted NAc+EC pair of an AT-REST agent or session directory — never a live
bio-stack.** Definable now because D28 made `create.agent()` actually fresh: a fresh
receiver is a real, empty, format-versioned pair on disk, not an absent file.

Receiver states and legality:

| Receiver state | Definition | Ingestion legal? |
|---|---|---|
| **Fresh** | `create.agent(name)` → `shutdown()`: an at-rest dir whose NAc/EC pair exists and is empty of learning | **Yes** — the canonical Oasis consumer ("born flinching" rides this state later) |
| **Loaded / at-rest** | An existing session or agent dir with learned state, no running process owning it | **Yes** — the merge-into-experience case; tighten-only protects the receiver's aversions |
| **Mid-session** | A running runtime holds the pair in memory | **NO** — the runtime persists at session end and would clobber the ingest (or the ingest clobbers unsaved learning). Same one-shot/boot-pickup discipline `merge-nac` already documents. The adapter cannot detect a foreign process; the contract makes the rule explicit and the CLI prints it. |

**File-level application is the sanctioned mechanism**, matching the `invalidate` verb
precedent: `substrate_merge`'s `ec_nodes` output already contains every receiver node
(deep-copied) plus the folded/inserted donor nodes, and `EC.ingest_substrate_nodes`
documents that it "touches nothing else: signatures, LSH tables and encoder provenance
are left alone" — so splicing `ec_payload["substrate_nodes"] = merged` produces the same
persisted artifact as a live load→ingest→save round-trip. NAc is written whole via
`with_format_version(merged, _NAC_FORMAT_VERSION)`, the `merge-nac` precedent. Both
layouts are supported (`aut_nac.json`/`aut_ec.json` session layout; `nac.json`/`ec.json`
agent layout), detected as a pair.

## 2. Operator surface — decided against the front-gate

**A new `ingest` subcommand on the existing `maxim substrate` verb.** Not a new
top-level verb, not a new module family, not an overload of `import`.

Why anything new is needed (the front-gate sentence): the existing surface cannot do
this because (a) `import` is contractually **side-effect-free extraction** — its
docstring and tests pin that it never merges; silently teaching it to validate+merge+
journal would change a shipped contract underneath its callers; and (b) leaving the
V1–V10 sequence to operators hand-running `substrate_merge` is precisely the
shipped-the-pieces-not-the-composition defect (D43's lesson, restated in
`substrate_merge`'s own docstring): a security pipeline that exists as a recipe is a
pipeline nobody runs. The duties compose in ONE callable
(`hivemind/ingest.py::ingest_bundle`), and the CLI verb is its one thin operator caller.
Everything inside routes through existing seams — `read_bundle_manifest`,
`assert_bundle_body_compatible`, `scrub_nac_state_for_bundle`, `is_identity_bearing`,
`filter_identity_bearing_links`, `substrate_merge` with the reserved
`trusted_sources`/`validate_node`/`validate_link`/`strict_geometry` parameters — never
re-implemented (the gate-7 rule generalized, stated in the threat model itself).

Dry-run by default; `--apply` writes (the `invalidate` precedent). A pre-ingest backup
of both receiver files is written before any mutation (the `merge-nac` precedent).

## 3. Step order — which duty fires where

Refusals raise `IngestRefused` (a `ValueError` subclass) carrying the duty tag
(`"V1"`…`"V10"`, `"gate7"`, `"V8"`) and a reason; the CLI turns it into the rc=2
contract. Order is load-bearing: cheap/structural checks run before anything is parsed,
parsing before anything is trusted, trust before anything is merged, and nothing is
written until every duty has passed.

1. **V6 — resource caps, from the ZIP central directory, BEFORE decompression.**
   Entry count, per-entry uncompressed size, total uncompressed size (compressed-size
   caps wave through high-ratio bombs). Constants in §5.
2. **Manifest** — `read_bundle_manifest` (existing seam: migration chain,
   `schema_version` refusal, `_format_version` check, `kind` check).
3. **V1 front door** — `manifest.contributor_id` must be operator-attested
   (`--trust <id>`, repeatable). Refusal, never admit-with-clamps. The id itself is
   validated via `_validate_source` (reserved `_*` refused).
4. **Gate 7** — `assert_bundle_body_compatible(manifest, receiver_body=…)`; the
   existing function, the existing exceptions, `--allow-unverified-body` passes
   through.
5. **V8 — journal gate.** Bundle digest (SHA-256 of the file bytes) already admitted →
   refused (`--force-digest` is the explicit operator override); contributor
   tombstoned → refused, no override.
6. **V7 — declared slices only.** `nac.json`/`ec.json` are read from the ZIP **only if
   declared in `manifest.contents`**, in memory (nothing extracted to disk — the
   undeclared-member hazard never reaches a loader). Undeclared members are reported
   in the result, never read. A declared slice missing from the ZIP namelist is a V3
   measured-class mismatch → refused.
7. **V2/V9/V3/V1 — the payload admission pass** (parse + validate + normalize, one
   sweep per slice):
   - JSON parsed with `parse_constant` refusing `NaN`/`Infinity` (V2 — `json.loads`
     accepts them by default).
   - Structural shape: slices must be JSON objects of the documented shapes; node/key
     counts capped (§5).
   - **V2 bounds**: every numeric field finite; `cluster_reward_bias`,
     `percept_valences`, `goal_reward_bias` in [-1, 1] and `reward_bias` in [0, 1] —
     out-of-range is *refused* (an honest composer cannot produce it);
     `predicted_value` in [-1, 1]; `confidence` in [0, 1] and additionally **capped**
     at `CAP_FOREIGN_CONFIDENCE` (a max-fold field — one asserted 1.0 would otherwise
     be permanent); `observation_count`/node `count`/Welford `n` non-negative and
     **capped** at `MAX_FOREIGN_COUNT` (a foreign count must not buy centroid
     dominance by assertion); `last_observed` clamped to the receiver's now
     (far-future timestamps freeze decay); the donor's `saved_at` is **dropped** so
     `nac_merge`'s later-of-two rule keeps the receiver's decay clock;
     `observed_deltas` truncated to `MAX_FOREIGN_DELTAS` (< the merge's `[-100:]`
     window) and `memory_ids` emptied (row K — foreign material cannot evict local
     history).
   - **V9 hygiene**: no `NAC_KEY_SEP` (`\x1f`) byte inside node ids or key *segments*;
     no pre-crafted `#`-collision masquerade in node ids; per-node `domain` in the
     reserved `_*` namespace (including `_identity`) → **refused** (row N is an
     attack shape, and the honest composer strips identity nodes at compose — so
     refusal costs honest senders nothing). To close row N's stamping half, foreign
     per-node `domain` is **stripped before the merge** (so `ec_merge_aligned`'s
     `domain or` fold can never stamp a LOCAL survivor) and re-stamped from
     `manifest.domain` (a gated assertion the operator saw) onto *inserted* foreign
     nodes only, after the merge.
   - **V3 geometry**: foreign nodes must carry a `geometry` stamp; unstamped →
     refused at admission (the fold-level `strict_geometry=True` alone only blocks
     FOLDS — unstamped nodes still insert). `--allow-unstamped-geometry` is the
     explicit legacy override (same shape as gate 7's `allow_unverified`): the
     SHA-manifested `53_agents` evidence archive predates the stamp, and the
     permissive-legacy allowance in the threat model exists for exactly this
     operator-attested case — default is refusal. `observed_embedding_dims` is
     re-measured from the actual arrays and compared to the manifest (measured
     class); mismatch → refused.
   - **V1 payload provenance sweep**: every in-payload `source`/`contributors` value
     must be `manifest.contributor_id` or the honest self-reference `"local"`
     (what every real exporter stamps); any OTHER id (trusted-id stuffing,
     attribution laundering, a multi-party set — a single-author bundle asserting
     consensus) or any reserved `_*` sentinel (`"_consensus"` in payload fields,
     which `_validate_source` never sees) → refused. Admitted material is then
     **receiver-stamped**: all provenance normalized to `manifest.contributor_id` —
     the receiver's own trust decision, never copied bytes.
8. **Inherent-class admission** (the poison-resistance slice's entry rule): a payload
   declaring `inherent_bias_keys` is refused unless `manifest.contributor_id` is in
   the operator's `--inherent-trust` set (Queen provenance). Refused loudly, not
   stripped — an inherent-class claim from a non-Queen source is a
   privilege-escalation attempt on the safety floor, not a cleanable field. A
   locally-learned bias has no self-promotion path: `NAc.mark_inherent_bias` is a
   curation-surface API with no learning-path caller (grep-guarded by its test).
9. **V4 — receiver-side quarantine + scrub, re-run on receipt.**
   `filter_identity_bearing_links` (bundle threshold 2) + the Welford-key identity
   filter + `scrub_nac_state_for_bundle` run on the incoming payload exactly as
   compose runs them — never trusting the sender's filter. This is also the payload
   SHAPE validation the loaders do not provide and the prompt-injection gate (row L:
   merged keys reach prompt annotation). Dropped/suspect entries and per-entity-class
   valence entries are reported in the result.
10. **Merge** — `substrate_merge(receiver…, donor…, strict_geometry=True,
    trusted_sources={contributor}, validate_node=…, validate_link=…,
    receiver_agent_id=…)`. The reserved parameters are the *belt* behind the V1 door,
    exactly as the threat model assigns them. **The tighten-only clamp fires INSIDE
    `substrate_merge`, post-fold** (the roadmap-decided seam — never a new merge
    function): for every signed bias dict (`cluster_reward_bias`,
    `percept_valences`, `goal_reward_bias`), a key the RECEIVER held at a negative
    value may deepen but is never raised toward zero by the fold
    (`merged[k] = min(merged[k], receiver[k])` when `receiver[k] < 0`).
    **Sign-scoped by construction: a fold whose receiver value is ≥ 0 or absent is
    byte-untouched** — the taught-want arms merge positive valence, and the gauntlet
    guard test verifies the byte-untouched property BY EXECUTION on the real taught
    seed-43 state (a clamp leaking into positive folds would silently change
    benchmark arm 2). This closes the `_merge_mean_clamped` annihilation hole: a
    +0.9 donor no longer erases a −0.9 aversion.
11. **V5/V8 — journal write** (with `--apply`; the dry run reports what would be
    written). The journal entry is the durable receiver-stamped attribution record:
    digest, contributor, timestamp, per-slice counts, `biases_rekeyed`/`dropped`,
    nodes folded/inserted — because the NAc bias dicts are provenance-free and
    `cluster_reward_source` promotes to `"mixed"`, the journal (not the state) is
    what makes a later distrust decision actionable. Written via
    `atomic_write_json(with_format_version(…))`.
12. **Write-back** (with `--apply`): backup both receiver files, then **EC first,
    then NAc**. A crash between the two writes must not mint the D2 dangling shape:
    NAc-first would leave merged biases naming donor clusters the EC file does not
    yet hold (dangling — exactly what `invalidate` exists to eliminate); EC-first
    leaves nodes without their biases, which is benign. (`invalidate` writes NAc
    first for the same reason with the risk running the other way — it *deletes*
    nodes.)

## 4. Idempotence

V8's journal semantics ARE the idempotence contract: ingesting the same bundle twice is
a **refusal**, not a no-op merge (counts/Welford SUM per ingestion, and repeated folds
walk `_merge_mean_clamped` geometrically — row J; "idempotent by re-merge" is
mathematically unavailable on these semantics, so the honest contract is
exactly-once-per-digest). `--force-digest` is the operator's eyes-open replay. The
journal lives in the receiver dir (`substrate_ingest_journal.json`) — per-receiver by
construction, and the slow-poison audit surface (repeated near-identical bundles from
one contributor are visible in it even when each merge is individually in-clamp).

**Tombstones** (row J's resurrection half): `IngestionJournal.add_tombstone(contributor)`
refuses every later bundle from that contributor. The operator surface for the full
distrust flow (tombstone + `prune_nac_cluster_biases` driven by the journal's
attribution records — threat model V5) is follow-up 1.2 adapter work, named here so its
absence is a decision: this PR ships the journal mechanics and the refusal path; the
pruning verb rides with the Queen-curation tooling.

## 5. Adapter constants

Frozen requirement (threat model V6/V2): the caps EXIST and refuse loudly. The values
are adapter decisions, tunable in review:

| Constant | Value | Duty | Rationale |
|---|---|---|---|
| `MAX_BUNDLE_ENTRIES` | 16 | V6 | 3 canonical members today; headroom for 1.2+ slices |
| `MAX_ENTRY_UNCOMPRESSED_BYTES` | 64 MiB | V6 | real slices are ~350 KB; 64 MiB is generous without letting a bomb through |
| `MAX_TOTAL_UNCOMPRESSED_BYTES` | 128 MiB | V6 | ditto, whole archive |
| `MAX_NODES_PER_SLICE` | 50 000 | V6 | post-parse structural cap (EC nodes / links / dict keys per slice) |
| `MAX_FOREIGN_COUNT` | 1 000 | V2 (row B) | foreign evidence weight ≤ ~1000 local observations; taught archive maxes at 518 |
| `CAP_FOREIGN_CONFIDENCE` | 0.9 | V2 (row M) | `confidence` max-folds; a foreign 1.0 would be permanent |
| `MAX_FOREIGN_DELTAS` | 50 | V2 (row K) | half the merge's `[-100:]` window — local history survives |
| foreign `memory_ids` | emptied | V2/V4 | episodes never ship; their IDs don't either (compose already enforces; receiver re-enforces) |

## 6. Interaction with the inherent bias class (semantics shipped this PR)

Defined in [coding_habits_oasis.md](coding_habits_oasis.md) §4; this PR ships the
**semantics only** — the marker, decay exemption, tighten-only, Queen-gated entry.
Queen distribution-at-creation and Gauntlet #3 ride Slice 1, after the four-arm data.

- **Marker**: `inherent_bias_keys` on the NAc state dict (list of
  `cluster_reward_bias` composite keys), persisted by `NAc.dump()`/`load_state()`,
  absent-tolerant (old files load as empty — the additive-key precedent).
- **Decay-exempt**: `decay_cluster_reward_biases` skips inherent keys entirely — no
  decay, no pruning ("innate fears do not extinguish"; pruning-exempt-but-decaying
  would extinguish to an un-pruned ~0, which is extinction with extra steps).
- **Tighten-only**: the §3.10 clamp applies to inherent keys like every negative
  bias — unconditionally, in the plan's words, because the clamp itself is
  unconditional in `substrate_merge`; the class does not depend on a future scoping
  of the clamp.
- **Merge/compose transport**: `nac_merge` unions the field (a rebuilt-dict merge that
  dropped it would be the D43 delete-state shape); `rekey_nac_state` re-keys it
  through the id map (a marker naming a re-keyed bias must follow it);
  `scrub_nac_state_for_bundle` re-keys its tsig third like `cluster_reward_bias` and
  drops markers whose entry did not survive the scrub.
- **Entry**: §3.8 — Queen provenance only, refusal on anything else.

## 7. Explicitly out of scope (so absence is a decision)

- `maxim substrate merge-nac` remains a trusted-local verb with none of V1–V9 (threat
  model's out-of-scope declaration; the warning-banner hardening is separate follow-up).
- Signature verification (reserved until P2P key distribution — the 1.2 trust boundary
  is the CHANNEL; V1 gates on ids the operator chose to trust).
- The distrust/prune operator flow (§4 above).
- Queen distribution-at-creation, Gauntlet #3, and every Slice-1 code-world item
  (roadmap sequencing decision 2).
- V10 has no reader to write yet: the adapter validates `capability_map`'s shape and
  otherwise carries it; the frozen rule ("a missed key is unverifiable, not
  'no capability'") binds the first future reader and is pinned in
  `derive_capability_map`'s docstring.

## 8. Guards

Threat model §6's closing rule binds: **an adapter PR implementing a duty without its
guard test has not shipped the duty.** The battery in
`tests/unit/test_hivemind_ingest.py` maps one-to-one onto §4's attack rows
(I/J/K/L/M/N each get a refusing or clamping test) plus the two rows added by the
poison-resistance slice: positive-donor-erases-aversion → clamped;
non-Queen-source-ships-inherent-class → refused. The byte-untouched guard runs the real
taught seed-43 merge through `substrate_merge` and compares against the pre-clamp fold
path by execution. The end-to-end proof
(`tests/integration/test_oasis_ingest_e2e.py`) composes the taught seed-43 bundle from
`docs/experiments/data/53_agents/` via the real CLI export (`--body-ref
--body-yaml`) and ingests it into a FRESH `create.agent()` receiver through the real
CLI ingest verb.
