# HF Hive Repository — public discovery + submission channel for signed substrate bundles

**Status:** Deferred shell plan — tracking only, no active work. Filed 2026-09-07 from a
1.2-window design conversation. The owner's original framing: "a Hugging Face bucket,
alongside the already semi-thought-through Space, to be a repository for people to submit
memories/substrate, validated through a gauntlet on steroids, then added to a main stream
for discovery and request."

**Verdict from the design pass:** the idea is the natural end-state of Oasis, but it
decomposes into two components with very different maturity, and only one of them is
launch-window work:

- **Phase 1 — read-only public distribution** (Queen-signed releases on HF + the
  replay/inspection Space): rides entirely on shipped 1.2 infrastructure, adds **no new
  trust decisions**, and is the right 1.2/1.3 launch companion.
- **Phase 2 — public submissions + gauntlet + promotion to a "main stream"**: this IS the
  Queen-tier promotion pipeline that Slice D explicitly deferred with its blockers named
  ([hivemind_p2p_scope.md](../hivemind_p2p_scope.md) §Slice D), plus a public front door
  on top. It cannot ship honestly before those prerequisites land, and it re-opens
  decision point 4 (no project-hosted Oasis) deliberately. 1.3+ work.

**Revive triggers (per phase):**

- **Phase 1:** revive at the 1.2 release transaction — it is a launch companion, not
  release scope; can be pulled forward any time visibility is wanted. Sole prerequisite:
  the public format-freeze check (§3).
- **Phase 2:** revive only when ALL of: (a) the four promotion prerequisites from the
  P2P scope doc have landed (bundle→gauntlet adapter → `sign_existing_bundle` →
  Gauntlet #3 → promotion provenance link); (b) there is a defensible answer to "what
  does *validated* mean for an arbitrary foreign bundle" (§5 — today R1 bounds the
  shared want to a layout-local cache entry, so validation is domain- and
  layout-specific by construction); and (c) the owner deliberately re-opens decision
  point 4. Two of three is not a revive.

---

## 1. Why Hugging Face specifically

Unchanged from the 2026-09-01 Space decision: HF owns Pollen Robotics; the Reachy Mini is
an HF product; the "what do I run on my Reachy Mini" audience congregates there. A bundle
repository next to the replay Space makes the discovery story one destination: *watch* a
Maxim learn (Space), *take* what it learned home (repository), eventually *give back*
(Phase 2).

## 2. The load-bearing property that makes Phase 1 cheap

The shipped 1.2 trust design already treats the distribution channel as untrusted:

- Trust anchors are **ed25519 Queen signatures** (Slice A) verified by the consumer, plus
  the operator's `hive.json` Queen keys — never the host that served the bytes.
- The receiver re-runs the full V1–V10 admission pipeline on every ingest
  ([oasis_ingestion_contract.md](../oasis_ingestion_contract.md)) regardless of origin.

So hosting signed release bundles on HF (or any static host) adds **zero new trust
surface** for consumers. HF is just a mirror; a tampered bundle fails
`assert_bundle_signature` exactly as it would from a hostile Oasis. This is the
front-gate reason Phase 1 needs almost no new mechanism.

## 3. Phase 1 — read-only Hive Repository (1.2/1.3 launch companion)

**Shape:** an HF dataset repo holding the Queen's signed release bundles + manifests,
plus a static `releases.json` index mirroring the response shape of
`GET /v1/substrate/releases` (the "**static Oasis contract**": any dumb HTTPS host that
serves that index + the bundle files is pull-compatible). Companion: the replay/inspection
Space (Gradio, curated session artifacts stepped through turn by turn — an Exp 52 nursery
run, a cradle sim, an Exp 56 arm; **no live agent loop**, per the 2026-09-01 decision:
per-turn LLM cost, longitudinal value invisible in ephemeral containers, concurrency
failure modes).

**Exemplar content:** the Exp 56 taught-want release (the seed-43 taught archive lineage)
with its `body_ref` + `capability_map` (gate 7, PR #658) — the bundle the headline claim
was earned on, so the public artifact and the published claim are the same object.

**Consumer path — decide smallest-first at revive time:**
- (a) nothing new: document `hive add <name> <hf-resolve-url> --queen-key <pub>` if the
  static index satisfies the existing client's URL scheme;
- (b) a small static-host mode in `hivemind/substrate_client.py` if the served-Oasis
  client assumes endpoint behavior a static host can't provide.
Front-gate rule: (b) only with the specific incapacity named.

**The one real prerequisite — public format-freeze check.** Publishing bundles to a
public repository hardens the bundle wire boundary far beyond two coordinated repos:
strangers' downloaded bundles must ingest correctly across future versions.
`read_bundle_manifest` already carries `schema_version` refusal + a migration chain, and
`hivemind/` is mypy-gated as a wire boundary — but a deliberate "are we ready to freeze
this shape in public" pass (manifest fields, slice shapes, the §5 adapter constants as
compatibility surface) is a named gate, not a formality.

**Publication-privacy pass (open question Q3):** bundles are NAc+EC only, identity-scrubbed
at compose and again at ingest — but cluster/percept/goal keys carry sim-derived text.
Before the first public upload, one human read of the actual exemplar's key material.

## 4. Phase 2 — submissions + gauntlet + main stream (1.3+)

**Honest framing: this is Slice D's deferred promotion pipeline with a public front
door.** "Submit → gauntlet on steroids → main stream" maps exactly onto
`contribute → experimental/ → gauntlet battery → promote → releases/`. Everything the
scope doc names as the promotion blocker is a blocker here, plus:

- **Decision point 4 re-opened:** a submission-accepting HF repo IS a project-hosted
  reference Oasis in effect (curation labor, abuse handling, availability). Deferring it
  was an operational-burden decision; Phase 2 must revisit it eyes-open, not erode it.
- **The gauntlet is the hard part and is partly a research question, not a feature.**
  Gauntlets #1/#2 are domain-specific and don't score a bundle; Gauntlet #3 is
  design-only. R1 (`CACHE-CONFIRMED`) bounds what any gauntlet can currently certify:
  the shared want is an exact-key cache entry at the 0.85 threshold — valid on the
  taught layout, architecturally unreachable elsewhere. Until the cross-cluster
  generalization channel (designed 1.3 work) exists, "validated" can only mean
  "passed a domain-specific battery on a matched apparatus" — which must be stated on
  the stream, never rounded up to "safe and useful for you."
- **A vacuous gauntlet is worse than none.** A rubber-stamp validator on a public stream
  is the vacuous-guard shape (a mechanism that does not really run looks exactly like one
  that ran and found nothing) with the project's signature on the output. Per-submission
  gauntlet runs cost live sims; the compute/funding model is an open question, and
  "we'll score it cheaply" is not an answer.
- **Threat model extension:** public submissions add repo-side threats (spam, slow-poison
  campaigns across many identities, gauntlet-overfit bundles) that the receiver-side
  V1–V10 belt does not address — [sharing_threat_model.md](../sharing_threat_model.md)
  §4 needs a Phase-2 addendum before any submission channel opens.

**Explicit Phase-2 non-goals (carried from the 1.2 decisions unless re-decided):** no
auto-promotion, no reputation system, no live Oasis↔Oasis sync, no unsigned content on
the main stream, ever.

## 5. Open questions (answer at revive time, in the pre-design review round)

- **Q1 — static Oasis contract vs HF-specific client** (§3 consumer path a/b).
- **Q2 — submission mechanism** (Phase 2): HF PRs into a `contributions/` area (public,
  auditable, spam-exposed) vs `hive contribute` to a project-run Oasis that mirrors
  promoted releases to HF (shipped path, but stands up decision-point-4 infrastructure).
- **Q3 — publication-privacy pass** for bundle key material (§3).
- **Q4 — gauntlet compute + funding** per submission (Phase 2).
- **Q5 — licensing** of published bundles (a bundle is trained state, not code — pick a
  license posture before the first upload, not after).

## 6. Related

- [hivemind_p2p_scope.md](../hivemind_p2p_scope.md) — Slice D block: the deferred
  promotion pipeline + its four prerequisites; decision point 4.
- [oasis_ingestion_contract.md](../oasis_ingestion_contract.md) — the receiver-side
  V1–V10 pipeline every pulled bundle passes; §6/§7 schedule Gauntlet #3.
- [sharing_threat_model.md](../sharing_threat_model.md) — the frozen receiver contract;
  Phase 2 owes it a repo-side addendum.
- [coding_habits_oasis.md](../coding_habits_oasis.md) §4 — Gauntlet #3 design; the
  inherent bias class Queen promotion would exercise.
- [minecraft_benchmark.md](../minecraft_benchmark.md) Part II — R1's cache-not-concept
  bound; the survival benchmark as a candidate future gauntlet bench.
- [maxim_hivemind.md](../maxim_hivemind.md) — the vision authority this serves.
