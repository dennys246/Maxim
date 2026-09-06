# Sharing threat model + bundle compatibility freeze (1.2 gate 4)

**Status: FROZEN 2026-09-04** (two-lens review round folded pre-freeze: security lens
F1–F13, code-accuracy lens A1–A8 — both reports in the round record). This document
closes roadmap gate 4 ("bundle/version compatibility and the sharing threat model must
be frozen") for 1.2 Oasis. It is the authority the 1.2 ingestion adapter implements
against; changing the receiver validation contract (§5) after the first shared bundle
requires a dated amendment here. Everything in it **extends existing `hivemind/`
surfaces — nothing forks**: the reserved parameter shapes (`trusted_sources` /
`validate_node` / `validate_link` / `strict_geometry`), the composition scrub, the
identity quarantine, and the gate-7 refusal semantics are the mechanisms; this
document assigns them duties — and names precisely where a mechanism is WEAKER than
its name suggests, because an adapter will implement against these words.

**Ordering dependency (stated so the freeze is honest):** this document merges AFTER
(a) PR #624 — the gate-7 caller wiring (`assert_bundle_body_compatible` on the CLI
import path, `prune_nac_cluster_biases`) — and (b) the gate-3 D8 pre-registration
branch. Statements below that lean on either say so.

## 1. Trust model

The adversary is the **bundle author** (or anyone who modified a bundle in transit —
at 1.2 there is no signature verification, only reserved slots). A bundle is a ZIP of
attacker-controlled bytes; every manifest field AND every payload field is an
**assertion**, not a fact. Verification classes:

| Class | Fields | Receiver stance |
|---|---|---|
| **Measured at compose, re-measurable at receipt** | `encoder_provenance.observed_embedding_dims`, `contents` vs the ZIP namelist | Re-measure at admission (duty **V3** — no code does this today) and refuse a mismatch. |
| **Assertions the receiver can gate on but not verify** | Manifest: `body_ref`, `affordance_namespace`, `capability_map`, `contributor_id`, `domain`. Payload (in `ec.json`/`nac.json`): per-node `geometry`, `count`, `domain`, `source`/`contributors`, every numeric field, every key string | Gate loudly (refusal semantics), never treat as evidence. Absence is *unverifiable*, not compatible (the gate-7 / format-version `"0.x"` reasoning). |
| **Reserved, unverified at 1.2** | `signature`, `signature_algorithm`, `signer_identity` | Ignore (never treat a present-but-unverified signature as trust). |

A merged substrate's own recorded provenance describes only its local encoders — the
per-contributor union rule pinned in `compose_bundle`'s MERGE SEMANTICS docstring
stands; a receiver must never trust a merged substrate's local stamps for foreign
nodes.

## 2. Compatibility freeze (the versioning contract)

- **Envelope**: `manifest.schema_version` (int, currently **2**) is the bundle's
  structural version. A receiver **refuses** `schema_version` greater than it supports
  (enforced in both `extract_bundle` and `read_bundle_manifest`); older versions
  migrate forward through `migrate_bundle_envelope`'s registered per-version
  migrations (v1→v2 shipped; missing gate-7 fields default to honest-unknown and are
  then refused as unverifiable by body-checking receivers). Migrations transform the
  **manifest only** — `extract_bundle` copies payload members through untouched, so a
  change to a *payload's keyed content* can never ride the migration seam
  (d43_merge_correctness.md §5).
- **Additive manifest keys** need NO schema bump — readers use `.get` with an honest
  default (the `signer_identity` precedent). **New slices** (a new file in the ZIP)
  require a schema bump + migration.
- **`_format_version`** (string, "1.0") is the house persisted-JSON envelope on the
  MANIFEST and is orthogonal to `schema_version`. **The payload slices are NOT
  format-validated today** (accuracy-lens A3, verified): `ec.json` is written as a
  bare `{"substrate_nodes": ...}` with no envelope; `nac.json` carries
  `_format_version` only when the exported session file happened to (`NAc.dump()`
  does not stamp), and the bundle-path consumers (`NAc.load_state`,
  `EC.ingest_substrate_nodes`) run no `check_format_version`. Payload shape
  validation is therefore an ADAPTER duty (V4), not something the loaders provide.
- **Unknown slices**: `extract_bundle` writes only ZIP members routed through
  `_safe_join`, but it extracts EVERY member, declared or not; the ingestion adapter
  MUST ignore any file not declared in `manifest.contents` and MUST NOT feed
  undeclared slices to any loader (duty V7).
- **Encoder coupling**: a bundle is meaningful only under the encoder geometry it was
  encoded in. Geometry compatibility is per-NODE (`geometry` tags + the merge's
  equality gate), not per-bundle; `encoder_provenance` is diagnostic context, not the
  gate.

## 3. What already holds (verified against code, 2026-09-04) — with its exact limits

- **Episodes never ship** — `compose_bundle` has no code path that reads hippocampus
  state; structural, not policy.
- **Identity quarantine at compose** — `filter_identity_bearing_links` +
  event-signature filtering at threshold 2 (bundle-stricter), plus the UNCONDITIONAL
  content scrub (`scrub_nac_state_for_bundle`, AST-guarded). **Compose-side only** —
  a malicious author simply does not run it (row L).
- **ZIP-slip** — every extract path routes through `_safe_join` (pre-validated before
  any write).
- **Bias clamps** — `nac_merge` folds via mean-with-clamp: `reward_bias` to
  `[0, max_reward_bias]`, `cluster_reward_bias` to `[-1.0, 1.0]`, `percept_valences`
  to `[-1.0, 1.0]`. Zero-prior: absence is no evidence. **Limits:** clamps bound
  MAGNITUDE only; `predicted_value` has no clamp at all; link `confidence` folds by
  `max()`; `priors` resolve by higher asserted confidence; and Python min/max NaN
  semantics make `_merge_mean_clamped` map an asserted `NaN` to **+hi** (row M).
- **Geometry equality gate** — two nodes that BOTH declare geometries fold only when
  equal; `strict_geometry=True` additionally refuses unstamped pairs **at the FOLD
  level only**: an unstamped foreign node that matches nothing still INSERTS into the
  merged substrate (accuracy-lens A2, verified). Admission-level refusal is duty V3.
- **Trust gating parameter** — `trusted_sources` filters right-side LINKS in
  `nac_merge` and right-side NODES in `ec_merge_aligned` by contributor subset. **It
  silently drops (never raises), reads attacker-authored contributor fields, and
  touches nothing else** — `reward_bias`, `cluster_reward_bias`, `percept_valences`,
  `event_outcome_welford`, `outcome_index`, `priors`, `total_observations` all fold
  unfiltered, and the bias/valence dicts carry no provenance at all (security-lens
  F1/F2, accuracy-lens A5). It is defense-in-depth, not the trust boundary (V1).
- **Body namespace refusal** — `assert_bundle_body_compatible` (gate 7):
  cross-body and undeclared-body bundles are refused before anything is written **on
  the CLI import path as of PR #624** (the ordering dependency above; before #624 the
  function had zero non-test callers).

## 4. Attack surface (asset × mechanism × residual gap)

| # | Attack | Mechanism | Holds today | Residual gap → §5 duty |
|---|---|---|---|---|
| A | **Bias-key steering** — ship `cluster_reward_bias` entries that bias the receiver toward attacker-chosen actions | Merged biases feed action selection and prompt annotation | Per-key clamp ±1.0; zero-prior; re-key drops donor keys whose clusters didn't survive alignment; targeting EXISTING clusters requires knowing their UUIDs | Clamps bound magnitude, not INTENT: a bundle whose own `ec.json` aligns to local clusters legitimately re-keys onto them. **V1, V5, V8**. |
| B | **Count/evidence inflation** — assert huge `count` / Welford `n` so count-weighted folds are dominated by assertion | `ec_merge_aligned` weights centroids by `count` with **no cap**; frozen-centroid modalities hide the same inflation. Same axis as D8's M3 (counts already conflate local query traffic with evidence — gate-3 prereg) | Nothing bounds a claimed count | **V2**. |
| C | **Geometry lies** — wrong or absent `geometry` tags fold or insert incomparable vectors | Both-stamped mismatches never fold | Unstamped foreign nodes pass the fold gate by default and INSERT even under `strict_geometry` | **V3**. |
| D | **Valence poisoning** — `percept_valences` steering Pavlovian responses | Clamped to [-1,1]; identity-bearing keys dropped at compose | In-range non-identity poison merges unfiltered (no provenance on the dict) | **V1** (front door) + **V4** (report). |
| E | **Identity smuggling** — rely on the receiver trusting the SENDER's identity filter | Compose-side scrub + filter | Receiver never re-checks | **V4**. |
| F | **Resource attacks** — zip bomb, million-node slices | `_safe_join` stops slip only | No size/entry caps; compressed-size caps are routable (high-ratio bombs) | **V6**. |
| G | **Body/capability lies** — false `body_ref` / `capability_map` | Refusal gates declared mismatches | A LIE (wrong-but-matching `body_ref`) passes | Accepted at 1.2 (assertion by design); capability-map reader rules are **V10**. |
| H | **Unknown-slice injection** | `_safe_join` bounds the path | Undeclared members still extract | **V7**. |
| I | **Provenance forgery** — trusted-id stuffing (`contributors: ["queen-key"]`), attribution laundering (framing a trusted third party, or the receiver's own id), reserved-sentinel injection (`"_consensus"` in PAYLOAD fields, which `_validate_source` never sees) | All contributor/source fields in the payload are attacker bytes; the trust filter does a subset test on them | Nothing validates in-payload provenance | **V1** (the front door exists precisely because of this row). |
| J | **Replay / repeated ingestion** — re-ingest one honest bundle k times: summed counts manufacture k× evidence from trusted material; repeated small bundles walk `_merge_mean_clamped`'s 50/50 fold geometrically toward the attacker's value (in-clamp, low-and-slow); a pruned contributor's state resurrects on the next replay | Counts/Welford SUM per ingestion; the pairwise mean re-weights 50/50 every merge regardless of local evidence mass; no bundle identity, no dedup, no tombstones | Nothing | **V8**. |
| K | **Tail-truncation eviction** — a hostile link shipping 100 fabricated `observed_deltas` (or 50 `memory_ids`) EVICTS the receiver's entire local history for that field in one merge (`[-100:]` / `[-50:]` keep the concatenated TAIL) | Destructive replacement, not inflation — no clamp applies | Nothing | **V2** (list-length caps below the truncation windows). |
| L | **Payload free-text / prompt injection** — hand-written `nac.json` with arbitrary strings in event signatures, `event_context`, `outcome_signature`, entity classes; merged biases feed PROMPT ANNOTATION, so attacker strings reach the receiver's LLM prompt | The scrub battery is compose-side only | Receiver runs no content scrub | **V4**. |
| M | **Numeric-field poisoning** — unclamped `predicted_value` (ship `1e18`); `NaN`/`Infinity` (accepted by `json.loads`!) where NaN folds to **+hi** in `_merge_mean_clamped`; `confidence` max-fold (assert 1.0 once, permanent); `priors` higher-confidence-wins; far-future `saved_at`/`last_observed` max-fold poisons the decay clock so foreign biases never decay | Clamps cover three dicts only | Every other numeric field unbounded | **V2**. |
| N | **Domain stamping** — foreign nodes stamp `domain` onto undomained local survivors on fold (`target["domain"] = ... or norm_r.get("domain")`); shipping `domain: "_identity"` makes merged RECEIVER nodes identity-quarantined — silently excluded from every future export (self-hiding poison + knowledge suppression) | Nothing validates per-node domain | Nothing | **V9**. |

## 5. The receiver validation contract (what the 1.2 Oasis ingestion adapter MUST do)

Each duty names the seam it rides. **The adapter routes through these — it does not
re-implement them** (the gate-7 rule generalized).

- **V1 — trust is a FRONT-DOOR, bundle-level decision on the manifest, before any
  merge call.** The adapter refuses a bundle whose `manifest.contributor_id` is not
  operator-attested — refusal, never admit-with-clamps. In-payload provenance is
  attacker data (row I): at admission every in-payload contributor set must equal
  `{manifest.contributor_id}` (a single-author bundle asserting multi-party consensus
  is refused), reserved `_*` sentinels in payload provenance are refused, and the
  admitted contributor tag is stamped BY THE RECEIVER from its own trust decision,
  never copied from the payload. The `trusted_sources` parameter (which silently
  drops, covers links+nodes only, and reads the very fields being validated) is
  defense-in-depth BEHIND this door, never the door. (Seams: `read_bundle_manifest`
  + a payload-provenance sweep; `trusted_sources` stays as belt.)
- **V2 — every asserted number is bounded at admission.** Counts and Welford `n`
  capped (a foreign count must not buy centroid dominance by assertion alone — the
  cap's value/shape is an adapter decision); ALL numeric payload fields validated
  finite (`NaN`/`Infinity` refused — `json.loads` accepts them) and in-range
  (`predicted_value`, `confidence`, priors); monotone-max folds (`confidence`,
  `saved_at`/`last_observed`) are capped or ignored for foreign input so one
  assertion cannot become permanent or freeze decay; list fields (`observed_deltas`,
  `memory_ids`) length-capped BELOW the merge's tail-truncation windows so foreign
  material cannot evict local history (row K). (Seams: `validate_node` /
  `validate_link` / a pre-merge normalization pass.)
- **V3 — foreign geometry is verified, strictly.** Foreign merges run
  `strict_geometry=True` AND the adapter refuses unstamped foreign nodes at
  admission (the parameter alone only blocks FOLDS — unstamped nodes still insert);
  `observed_embedding_dims` and `contents` are re-measured against the actual
  arrays/namelist. The permissive-legacy allowance exists for a receiver's OWN
  pre-stamp files, not for strangers. (Seams: `strict_geometry` + `validate_node` +
  the measured-class manifest fields.)
- **V4 — re-run BOTH quarantine and content scrub on receipt.** The receiver applies
  `is_identity_bearing` (bundle threshold) AND `scrub_nac_state_for_bundle` (pure —
  it runs identically on a received state) to the incoming payload itself, reporting
  dropped/suspect entries — never trusting the sender's filter. This is also the
  payload SHAPE validation the loaders do not provide (§2) and the prompt-injection
  gate (row L: free text in merged keys reaches prompt annotation). Valence entries
  are reported per entity class before merging. (Seams: `hivemind/identity.py`,
  `scrub_nac_state_for_bundle`, `validate_link`.)
- **V5 — attribution is receiver-stamped and durable.** Contributor tags survive the
  fold where the shapes allow (EC contributors union), and the adapter records
  per-bundle attribution in the V8 journal — because the NAc bias/valence dicts are
  provenance-FREE and `cluster_reward_source` promotes to `"mixed"`, so per-entry
  attribution inside NAc state is structurally impossible; the journal, not the
  state, is what makes a later distrust decision actionable. Pruning rides
  `prune_nac_cluster_biases` (PR #624) driven by the journal.
- **V6 — resource caps before parsing, on UNCOMPRESSED sizes.** Max entry count and
  max per-entry/total uncompressed size read from the ZIP central directory BEFORE
  decompression (a compressed-size cap waves through high-ratio bombs); max
  nodes/keys per slice before JSON parse. Numbers are adapter constants; the frozen
  requirement is that they exist and refuse loudly.
- **V7 — only declared slices are read.** Ingestion ignores ZIP members absent from
  `manifest.contents`; an undeclared member is reported.
- **V8 — an ingestion journal with dedup and tombstones.** The adapter records each
  admitted bundle's digest + contributor + timestamp; re-ingestion of a seen digest
  is refused (or explicitly operator-forced); a pruned/distrusted contributor gets a
  tombstone so replays cannot resurrect their state (row J). The journal is also the
  slow-poison audit surface: repeated near-identical bundles from one contributor
  are visible in it even when each merge is individually in-clamp.
- **V9 — payload identifier + domain hygiene.** Node/cluster ids and key strings are
  validated against an identifier charset at admission (no `NAC_KEY_SEP` bytes — an
  embedded `\x1f` corrupts re-keyed bias parsing; no pre-crafted `#`-suffix
  masquerade); foreign per-node `domain` values in reserved namespaces (`_*`,
  including `_identity`) are refused or stripped, and a foreign node must not stamp
  a domain onto a surviving LOCAL node (row N).
- **V10 — capability-map readers treat entries as unverifiable claims.** A missed
  key is unverifiable, not "no capability" (the collision-context limit pinned in
  `derive_capability_map`'s docstring, PR #624).

**Dated amendments to §5** (the freeze's own change mechanism; both surfaced by the
adapter's pre-merge review round, 2026-09-05):

- **V1 (2026-09-05):** the in-payload contributor-set equality admits ONE additional
  value — the composer's honest self-reference `"local"`, which every real exporter
  stamps (`EC`/`NAc` stamp their own material `source: "local"`; refusing it refuses
  every honest bundle ever composed). It is normalized to `manifest.contributor_id` at
  receiver stamping, so it cannot launder attribution to any third party; every OTHER
  id and every reserved `_*` sentinel refuses exactly as frozen.
- **V6 (2026-09-05):** "read from the ZIP central directory BEFORE decompression"
  quietly assumed the directory numbers were truthful — they are attacker bytes, and
  a binary-patched header declaring 10 bytes over an 800 MB stream routes the
  declared-size check (measured: ~1.3 GB in-memory expansion through a naive
  ``zf.read``). The duty's caps therefore bind the ACTUAL decompressed byte count:
  the adapter reads every member it consumes through a bounded streaming read. The
  declared-size pass stays as the cheap first gate.
- **V3 (2026-09-05):** the adapter adds an operator-attested override
  (`--allow-unstamped-geometry`, default refusal) admitting unstamped foreign nodes
  from a legacy archive the operator vouches for — the SHA-manifested `53_agents`
  evidence bundles predate the geometry stamp and are the 1.2 case study's donor
  material. This extends the permissive-legacy allowance from "a receiver's OWN
  pre-stamp files" to "a channel-trusted pre-stamp archive", consistent with §5's own
  statement that the effective 1.2 trust boundary is the CHANNEL.

**Out of scope BY DECLARATION** (so absence is a decision, not an oversight):

- **`maxim substrate merge-nac` is NOT an ingestion path for foreign material.** It
  folds a local `nac.json` into the runtime persistence path with none of V1–V9 —
  by design, for trusted-local same-substrate files (its own docstring's one-shot
  discipline). Foreign substrate MUST arrive as a bundle through the adapter;
  handing `merge-nac` a stranger's file defeats this entire document. (Follow-up
  hardening — a warning banner on the verb — is 1.2 adapter work, not frozen here.)
- **Signature computation/verification** stays reserved until the P2P layer defines
  key distribution. Consequence, stated plainly: until then NOTHING in the bundle
  authenticates its author, so the effective 1.2 trust boundary is the CHANNEL — how
  the operator obtained the bundle — matching the Queen-tier curation framing in
  maxim_hivemind.md. V1 gates on an id the operator chose to trust, not on proof.
- **Verifying `body_ref` against anything** (assertion by design, gate-7 record);
  **behavioral vetting of a want's content** (Gauntlet #2 is the quality gate, not a
  security gate — a malicious want that passes the gauntlet is V1's problem).

## 6. Citations

Costing and merge mechanics: [d43_merge_correctness.md](d43_merge_correctness.md);
gate-7 record: [roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md) §Gates +
[oasis_case_study_taught_orient.md](oasis_case_study_taught_orient.md) §1; D8 count
provenance: the gate-3 D8 pre-registration
(`docs/experiments/protocols/d8_read_mutation_preregistration.md`, lands with the
gate-3 branch — the ordering dependency in the header). Bundle mechanics:
`src/maxim/hivemind/bundle.py` (`compose_bundle`, `extract_bundle`,
`migrate_bundle_envelope`, `_safe_join`, `assert_bundle_body_compatible`,
`scrub_nac_state_for_bundle`), `src/maxim/hivemind/merge.py` (`nac_merge`,
`ec_merge_aligned`, `substrate_merge`, `_merge_mean_clamped`, `_merge_link_pair` —
the tail-truncation site), `src/maxim/hivemind/identity.py`.
Regression guards: the existing hivemind suites pin §3's mechanisms
(`tests/unit/test_hivemind_bundle.py`, `test_hivemind_merge.py`,
`test_hivemind_identity.py`); §5's duties gain guards when the adapter ships — an
adapter PR that implements a duty without its guard test has not shipped the duty.

> **2026-09-05 (adapter shipped — citation only, no contract change):** the 1.2
> ingestion adapter is `src/maxim/hivemind/ingest.py::ingest_bundle` (operator surface
> `maxim substrate ingest`; step order + adapter decisions:
> [oasis_ingestion_contract.md](oasis_ingestion_contract.md)). §5's duty guards live in
> `tests/unit/test_hivemind_ingest.py` (one refusing/clamping test per §4 attack row) and
> `tests/integration/test_oasis_ingest_e2e.py` (the real-archive end-to-end proof).
