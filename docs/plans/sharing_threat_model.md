# Sharing threat model + bundle compatibility freeze (1.2 gate 4)

**Status: FROZEN 2026-09-04** — this document closes roadmap gate 4 ("bundle/version
compatibility and the sharing threat model must be frozen") for 1.2 Oasis. It is the
authority the 1.2 ingestion adapter implements against; changing the receiver
validation contract (§5) after the first shared bundle requires a dated amendment here.
Everything in it **extends existing `hivemind/` surfaces — nothing forks**: the
reserved parameter shapes (`trusted_sources` / `validate_node` / `validate_link` /
`strict_geometry`), the unconditional composition scrub, the identity quarantine, and
the gate-7 refusal semantics are the mechanisms; this document assigns them duties.

## 1. Trust model

The adversary is the **bundle author** (or anyone who modified a bundle in transit —
at 1.2 there is no signature verification, only reserved slots). A bundle is a ZIP of
attacker-controlled bytes; every manifest field is an **assertion**, not a fact.
Three kinds of manifest content, by verification class:

| Class | Fields | Receiver stance |
|---|---|---|
| **Measured at compose, re-measurable at receipt** | `encoder_provenance.observed_embedding_dims`, `contents` | Re-measure; a mismatch with the actual arrays is tampering or corruption → refuse. |
| **Assertions the receiver can gate on but not verify** | `body_ref`, `affordance_namespace`, `capability_map`, `contributor_id`, `domain`, per-node `geometry`, per-node `count` | Gate loudly (refusal semantics), never treat as evidence. Absence is *unverifiable*, not compatible (the gate-7 / format-version `"0.x"` reasoning). |
| **Reserved, unverified at 1.2** | `signature`, `signature_algorithm`, `signer_identity` | Ignore (never treat a present-but-unverified signature as trust). |

A merged substrate's own recorded provenance describes only its local encoders — the
per-contributor union rule pinned in `compose_bundle`'s MERGE SEMANTICS docstring
stands; a receiver must never trust a merged substrate's local stamps for foreign
nodes.

## 2. Compatibility freeze (the versioning contract)

- **Envelope**: `manifest.schema_version` (int, currently **2**) is the bundle's
  structural version. A receiver **refuses** `schema_version` greater than it supports
  (enforced in `extract_bundle` / `read_bundle_manifest`); older versions migrate
  forward through `migrate_bundle_envelope`'s registered per-version migrations
  (v1→v2 shipped; missing gate-7 fields default to honest-unknown and are then
  refused as unverifiable by body-checking receivers). Migrations transform the
  **manifest only** — `extract_bundle` copies payload members through untouched, so a
  change to a *payload's keyed content* can never ride the migration seam
  (d43_merge_correctness.md §5).
- **Additive manifest keys** need NO schema bump — readers use `.get` with an honest
  default (the `signer_identity` precedent). **New slices** (a new file in the ZIP)
  require a schema bump + migration.
- **`_format_version`** (string, "1.0") is the house persisted-JSON envelope and is
  orthogonal to `schema_version`; both coexist by design. Payload slices carry their
  own formats (`nac.json` via the NAc format version, `ec.json` via EC's) and are
  validated by their own loaders.
- **Unknown slices**: `extract_bundle` writes only ZIP members routed through
  `_safe_join`; the ingestion adapter MUST ignore any file not declared in
  `manifest.contents` and MUST NOT feed undeclared slices to any loader.
- **Encoder coupling**: a bundle is meaningful only under the encoder geometry it was
  encoded in. Geometry compatibility is per-NODE (`geometry` tags + the merge's
  equality gate), not per-bundle; `encoder_provenance` is diagnostic context, not the
  gate.

## 3. What already holds (verified against code, 2026-09-04)

- **Episodes never ship** — `compose_bundle` has no code path that reads hippocampus
  state; structural, not policy.
- **Identity quarantine at compose** — `filter_identity_bearing_links` +
  event-signature filtering at threshold 2 (bundle-stricter), plus the UNCONDITIONAL
  content scrub (`scrub_nac_state_for_bundle`, AST-guarded).
- **ZIP-slip** — every extract path routes through `_safe_join`.
- **Bias clamps** — `nac_merge` folds via mean-with-clamp: `reward_bias` to
  `[0, max_reward_bias]`, `cluster_reward_bias` to `[-1.0, 1.0]`
  (`max_cluster_reward_bias`, matching `NACConfig`), `percept_valences` to
  `[-1.0, 1.0]`. Zero-prior: absence is no evidence, never a zero vote.
- **Geometry equality gate** — two nodes that BOTH declare geometries fold only when
  equal; `strict_geometry=True` additionally refuses unstamped pairs.
- **Trust gating parameter** — `trusted_sources` filters foreign links by contributor
  set; `validate_node` / `validate_link` hooks reserved on the merge signatures.
- **Body namespace refusal** — `assert_bundle_body_compatible` (gate 7): cross-body
  and undeclared-body bundles are refused before anything is written on the CLI path.

## 4. Attack surface (asset × mechanism × residual gap)

| # | Attack | Mechanism | Holds today | Residual gap → §5 duty |
|---|---|---|---|---|
| A | **Bias-key steering** — ship `cluster_reward_bias` entries that bias the receiver toward attacker-chosen actions | Merged biases feed action selection and prompt annotation | Per-key clamp ±1.0; zero-prior; re-key drops donor keys whose clusters didn't survive alignment; targeting a receiver's EXISTING cluster requires knowing its UUIDs | Clamps bound magnitude, not INTENT: a bundle whose own `ec.json` aligns to local clusters legitimately re-keys onto them. Duty **V1, V5**. |
| B | **Count/evidence inflation** — assert huge `count` on nodes (and inflated Welford counts on links) so the count-weighted centroid mean and confidence folds are dominated by assertion | `ec_merge_aligned` weights centroids by `count` with **no cap**; frozen-centroid modalities hide the same inflation (centroid still, counts poisoned for the NEXT merge). Same axis as D8's M3 (counts already conflate local query traffic with evidence) | Nothing bounds a claimed count | Duty **V2**. |
| C | **Geometry lies** — wrong or absent `geometry` tags fold incomparable vectors (the invisible-corruption class: same dim, different basis) | Equality gate only fires when BOTH nodes are stamped; unstamped foreign nodes pass by default (`strict_geometry=False`) | Both-stamped mismatches never fold | Duty **V3**. |
| D | **Valence poisoning** — `percept_valences` shipping fear/attraction for common entity classes steers Pavlovian responses | Clamped to [-1,1]; identity-bearing keys dropped | A -1.0 fear valence for a common entity class is in-range and non-identity | Duty **V1** (trust gate) + **V4** (report). |
| E | **Identity smuggling** — rely on the receiver trusting the SENDER's identity filter | Compose-side scrub + filter | Receiver never re-checks | Duty **V4**. |
| F | **Resource attacks** — zip bomb, million-node `ec.json`, million-key bias dict | `_safe_join` stops slip only | No size/entry caps anywhere | Duty **V6**. |
| G | **Body/capability lies** — false `body_ref` / `capability_map` | Refusal semantics gate declared mismatches | A LIE (wrong-but-matching `body_ref`) passes; `capability_map` has no reader yet | Accepted at 1.2: `body_ref` is an assertion by design (gate 7 record); a capability-map READER must treat entries as unverifiable claims and a missed key as unverifiable, not "no capability". |
| H | **Unknown-slice injection** — extra ZIP members land on disk for some later loader to find | `_safe_join` bounds the path | Members not in `manifest.contents` still extract | Duty **V7**. |

## 5. The receiver validation contract (what the 1.2 Oasis ingestion adapter MUST do)

Each duty names the existing seam it rides. **The adapter routes through these — it
does not re-implement them** (the gate-7 rule generalized).

- **V1 — trust is fail-closed.** Ingestion runs with `trusted_sources` set to the
  operator-approved contributor set; an unattested contributor is refused, not
  admitted-with-clamps. (Seam: the existing `trusted_sources` parameter; the default
  `None` = admit-all remains only for trusted-internal merges.)
- **V2 — claimed evidence weight is bounded.** A foreign node's `count` (and link
  observation counts) is capped at admission so no assertion can dominate a local
  centroid or confidence fold. The cap's value and shape (hard cap vs log-scale) is an
  adapter design decision; the frozen requirement is: **a foreign count must not be
  able to buy centroid dominance by assertion alone.** (Seam: `validate_node` /
  a pre-merge normalization pass.)
- **V3 — foreign bundles merge `strict_geometry=True`.** Unstamped foreign nodes are
  refused (the permissive-legacy allowance exists for a receiver's OWN pre-stamp
  files, not for strangers), and `observed_embedding_dims` is re-measured against the
  actual arrays at admission. (Seam: existing `strict_geometry` + the manifest's
  measured-class fields.)
- **V4 — re-run the quarantine on receipt.** The receiver applies
  `is_identity_bearing` (bundle threshold) to incoming keys itself and reports
  dropped/suspect entries — never trusting the sender's filter. Valence entries are
  reported per entity class at admission so a poisoning attempt is visible before it
  merges. (Seam: `hivemind/identity.py`; `validate_link`.)
- **V5 — provenance is preserved through the fold.** Contributor tags survive merging
  (already: contributors union) so a poisoned want remains attributable and a later
  distrust decision can prune by contributor. (Seam: existing contributor tracking;
  pruning rides `prune_nac_cluster_biases`-style state surgery.)
- **V6 — resource caps before parsing.** Max bundle size, max ZIP entries, max
  nodes/keys per slice, enforced before any JSON is loaded into memory. (Seam: the
  admission front door; numbers are adapter constants, frozen requirement is that
  they exist and refuse loudly.)
- **V7 — only declared slices are read.** Extraction/ingestion ignores ZIP members
  absent from `manifest.contents`; an undeclared member is reported. (Seam:
  `extract_bundle`.)

**Non-goals at 1.2** (recorded so their absence is a decision, not an oversight):
signature computation/verification (slots stay reserved until the P2P layer defines
key distribution); verifying `body_ref` against anything (assertion by design);
behavioral vetting of a want's *content* (Gauntlet #2 is the quality gate, not a
security gate — a malicious want that passes the gauntlet is out of scope here and in
scope for V1's trust decision).

## 6. Citations

Costing and merge mechanics: [d43_merge_correctness.md](d43_merge_correctness.md);
gate-7 record: [roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md) §Gates +
[oasis_case_study_taught_orient.md](oasis_case_study_taught_orient.md) §1; D8 count
provenance: [../experiments/protocols/d8_read_mutation_preregistration.md](../experiments/protocols/d8_read_mutation_preregistration.md);
bundle mechanics: `src/maxim/hivemind/bundle.py` (`compose_bundle`,
`extract_bundle`, `migrate_bundle_envelope`, `_safe_join`,
`assert_bundle_body_compatible`), `src/maxim/hivemind/merge.py` (`nac_merge`,
`ec_merge_aligned`, `substrate_merge`, `_merge_mean_clamped`), `src/maxim/hivemind/identity.py`.
Regression guards: the existing hivemind suites pin §3's mechanisms
(`tests/unit/test_hivemind_bundle.py`, `test_hivemind_merge.py`,
`test_hivemind_identity.py`); §5's duties gain guards when the adapter ships — an
adapter PR that implements a duty without its guard test has not shipped the duty.
