# Hivemind P2P — the 1.2 build scope (front-gated)

**Status:** SCOPING — design pass, no code yet. This is the front-gated build scope for
the 1.2 Hivemind peer-to-peer slice. The vision authority is
[maxim_hivemind.md](maxim_hivemind.md) (unchanged); this doc decides *what actually gets
built*, against an audit of what already exists, and resolves the open decision points that
gate implementation. Per the project's **design-time scope pressure** and **audit-before-
building** rules, every sub-component below is front-gated against reuse before it is
allowed to be its own mechanism.

## The reframing (from the 2026-09-06 infrastructure audit)

The roadmap line "Full Hivemind protocol (~600 LOC): peer discovery, substrate-snapshot
exchange protocol, conflict-resolution semantics, poison-resistance defenses, well-known
reference servers" reads as five net-new mechanisms. The audit says otherwise — **most of it
already exists or is reserved-and-inert**, and the genuinely-new surface is smaller and
sharper than the line implies:

| Roadmap sub-component | Audit finding | 1.2 verdict |
|---|---|---|
| **Conflict-resolution semantics** | ALREADY SHIPPED in the merge math — `nac_merge`/`ec_merge_aligned` do Bayesian confidence aggregation, multi-source weighting, valence-preserving distributions, provenance retention. | **Reuse. Zero new code.** |
| **Poison-resistance defenses** | Receiver-side defenses SHIPPED — the V1–V10 `ingest_bundle` pipeline (front-door `trusted_sources` refusal, resource/zip-bomb caps, NaN/Inf refusal, identity quarantine, tombstone/digest journal) + the wired-but-inert `trusted_sources`/`validate_link`/`validate_node` merge hooks. | **Activate + policy-wire. Little new mechanism.** |
| **Substrate-snapshot exchange protocol** | No wire exchange exists, BUT the authenticated HTTP server pattern does — `leader_proxy` has bearer auth (constant-time, scheme-dispatched for a future `Signature`), admission/rate-limit gating, `/v1/*` routing, `X-Maxim-*` headers; `utils/http` is the sanctioned client. | **New ENDPOINTS on the existing server pattern, not a new transport.** |
| **Peer discovery** | Nothing — no mDNS/gossip/DHT/bootstrap. Only static config (`mesh.yml`, `peer.yml`). | **Static registry (peers/Oases list) + manual add for 1.2. Gossip/DHT deferred to 1.3+.** |
| **Well-known reference servers** | Nothing; decision point 4 defers the *project-run* one. | **Optional static entries in the registry; no project-hosted server in 1.2.** |
| **Bundle signing/verification** | Manifest slots (`signature`/`signature_algorithm`/`signer_identity`) exist but compute and validate NOTHING; no crypto dep. | **Genuinely new — the one real cryptographic build. Decision point 2, resolved below.** |

**Net:** the 1.2 P2P slice is **(a) substrate HTTP endpoints on the existing auth'd server,
(b) bundle signing + verification, (c) a static peer/Oasis registry + manual add, (d)
consumer trust-policy wiring that activates the reserved merge hooks, (e) the `maxim oasis`
/ `hive` CLI.** Conflict resolution and the receiver-side poison pipeline are reused as-is.
This is why the LOC estimate holds even though four of the six roadmap items shrink to
reuse — the endpoints + signing + registry + CLI carry it.

## Front-gate verdicts (design-time scope pressure)

Each component names why it rides existing infra, or the specific reason it cannot.

- **Exchange transport → RIDES ON `leader_proxy` + `utils/http`.** The unit of exchange is a
  bundle (a ZIP file) and a manifest (JSON). That is authenticated request/response, which
  the leader/Oasis HTTP server already does; the client side is `utils/http.fetch_url`/
  `download_to_file`. A new symmetric/gossip transport is NOT needed for pull-based release
  distribution + push-based contribution. *Reason it's not its own mechanism:* bundle
  transfer is HTTP GET/POST of a file, and the server already authenticates, admits, and
  rate-limits `/v1/*`.
- **Auth → RIDES ON the scheme-dispatched bearer in `leader_proxy._check_auth`.** It already
  401s unknown schemes cleanly (the CC13 auth format-freeze), so a `Signature` scheme is an
  additive branch, not a rewrite. Console-side pairing (`/api/pair/request` → `/api/pair/
  claim`, spoken-code, rate-limited) is the reusable admission-bootstrap pattern for adding
  a peer without pre-shared secrets.
- **Rate-limit / admission → RIDES ON `runtime/rate_limit.py::KeyedRateLimiter`** + the
  proxy's concurrency gate. Sock-puppet mitigation is "practical, not theoretical" (the
  design's own words) — per-source rate limits + the tiered trust policy, not a consensus
  protocol.
- **Conflict resolution → RIDES ON `hivemind/merge.py` (shipped).** No new code.
- **Receiver trust boundary → RIDES ON `hivemind/ingest.py` (shipped V1–V10).** 1.2 supplies
  the POLICY (`trusted_sources`) and the signing check in front of it, not a new pipeline.
- **Bundle signing → NEEDS OWN (the one real new mechanism).** *Reason:* asymmetric release
  signatures (a Queen publishes with a private key; any consumer verifies with the public
  key) cannot ride the symmetric bearer tokens — bearer auth proves "you may talk to this
  server," signing proves "this substrate is the Queen's, unmodified." Different property,
  no existing surface computes it.
- **Discovery → static registry NOW; gossip DEFERRED.** *Reason a static list suffices for
  1.2:* the design's own phasing is "published reference Oases + manual peer addition" for
  1.2; gossip/DHT is 1.3+. A `peers.yml`-shaped registry (the shipped `mesh.yml` pattern)
  is the minimum that works and front-gates the hardest, least-certain component out.

## The scoped build (slices)

Each slice ships with its callers (the "a fix ships with a caller" rule — no reserved
capacity without a consumer), an `[engineering]` invariant where it establishes a contract,
and a guard test. Sizes are rough.

### Slice A — Bundle signing + verification (~150 LOC + optional dep)
Activate the reserved manifest slots. `compose_bundle(..., signer=...)` computes an
**ed25519** signature over the canonical (nac.json + ec.json + the signed manifest fields);
`assert_bundle_signature(bundle, *, trusted_keys)` verifies it and is called INSIDE
`ingest_bundle` behind a policy flag. New optional extra `[sign]` → `cryptography`, added via
`utils/optional_deps.py::EXTRA_FOR_IMPORT` (the canonical surface — no ad-hoc `try/import`).
Keys managed like the tunnel bearer keys (`tunnel/keys.py` pattern): a keypair file under
`~/.config/maxim/`, public key exportable. **Engineering invariant:** a bundle whose manifest
declares `signature_algorithm` but fails verification is REFUSED by ingest (loud, `IngestRefused`),
never admitted-with-clamps — the same "refuse, don't dilute" rule the V1 front door already
uses. **Ship-with-caller:** `maxim substrate export --sign` produces one; `maxim substrate
ingest --require-signed` consumes it; guard test round-trips sign→verify→tamper-detect.

### Slice B — Substrate HTTP endpoints on the Oasis server (~200 LOC) — SHIPPED (dormant)
**Shipped:** `hivemind/store.py` (`OasisStore`, two-tier releases/experimental, thread-serialized accept path), `hivemind/oasis_endpoints.py` (transport-agnostic handlers), `hivemind/substrate_client.py` (client over `utils/http`), the three `/v1/substrate/*` routes on `leader_proxy` (gated by an injected `OasisStore`, default an authenticated 404), and `utils/atomic_io.atomic_write_bytes`. Two-lens review folded a cross-confirmed HIGH (lost-update race on the provenance log → store lock + validate-before-write + fail-loud corrupt-log) plus the client error-contract, `Content-Length` 500, and `start_leader_proxy` module-global fixes. Guards: `test_oasis_store.py` (incl. concurrency + corrupt-log) + `test_oasis_exchange_e2e.py` (real round-trip). **Honest status:** the endpoints + client ship with a proven end-to-end test but NO production caller — `maxim oasis serve` (Slice C) is the only thing that injects a store, so the surface is dormant until C. This is a deliberate slicing choice (the endpoints cannot have a production caller without the CLI), not a satisfied ship-with-a-caller.

Add a substrate surface to the existing authenticated server (the `leader_proxy` handler
pattern, or a sibling handler sharing `_check_auth`/`_check_admission`):
- `GET /v1/substrate/releases` — list the Oasis's published Queen-tier releases (manifests only).
- `GET /v1/substrate/bundle/<id>` — download a signed bundle (streamed).
- `POST /v1/substrate/contribute` — accept an experimental-tier contribution (admission-gated,
  rate-limited per source, quarantined until promoted — lands in the experimental tier, never
  the Queen tier directly).
Client side is `utils/http` (`fetch_url` for manifests, `download_to_file` for bundles).
**Engineering invariant:** contributions land in the experimental tier tagged with provenance;
promotion to Queen tier is a SEPARATE gated operation (Slice D), never a side effect of receipt.
**Ship-with-caller:** the `maxim oasis serve` command starts it; the pull/contribute CLI (Slice C)
exercises it; guard test drives the three endpoints against a fixture Oasis.

### Slice C — `maxim oasis` / `hive` CLI + static registry (~150 LOC)
- `maxim oasis serve` (start the substrate endpoints), `oasis publish <bundle>` (sign + list a
  release), `oasis status`.
- `maxim hive pull [--domain <d>] [--from <oasis>]` (fetch + `ingest` signed releases, default
  `trusted_sources = {configured Queen keys}`), `hive contribute <bundle> --to <oasis>`,
  `hive add <name> <url> [--queen-key <pub>]` / `hive list` (the static registry).
- **Registry:** `hive.yml` under `~/.config/maxim/` — a list of known Oases (name, URL, optional
  Queen public key, subscribed domains), shape-frozen like `mesh.yml`. Manual add; optional
  seeded well-known entries (community-run, NOT project-hosted in 1.2).
**Ship-with-caller:** these ARE the callers for Slices A/B/D; guard tests on argument parsing +
a dry-run pull against a fixture.

### Slice D — Consumer trust policy + Queen-tier wiring (~100 LOC)
Wire the default consumer policy through the shipped hooks: a fresh Maxim pulls Queen-tier
releases with `trusted_sources = {queen-keys}`; the experimental tier is opt-in
(`hive subscribe --experimental <oasis>`). Promotion (experimental → Queen) runs the existing
gauntlets (Gauntlet #1 `probe_policy`, Gauntlet #2 Exp 53 readout, Gauntlet #3 coding-safety
from the poison-resistance slice) — the promotion command re-runs the battery before signing a
release, the "sleep-replay at fleet scale" the design frames. **No new merge code** — this is
policy assembly over `ingest_bundle`'s `trusted_sources`/`inherent_trusted_sources` params and
the signing check. **Engineering invariant:** default trust is Queen-only; the experimental tier
requires an explicit opt-in per Oasis.

## Decision points — resolved or teed up (they gate implementation)

1. **CLI naming** → **RESOLVE: `maxim oasis` (formal, serving/publishing) + `maxim hive`
   (user-facing pull/contribute/registry).** Matches the doc's tentative; both read cleanly.
2. **Bundle signing (a pre-1.2 GATE, decision point 2)** → **RESOLVE: sign Queen-tier releases
   (REQUIRED for promoted-domain sharing), optional for experimental-tier contributions.** This
   is the tiered-trust answer to friction-vs-onboarding: the casual consumer's exposure is the
   curation pipeline (signed Queen releases), while experimental contribution stays low-friction.
   ed25519 via a new `[sign]` optional extra. A documented trust policy ships with Slice A (the
   design requires "a documented trust policy before promoted-domain sharing").
3. **Substrate domains starter set** → **RESOLVE: ship the seeded set** (combat, cooking, medical,
   fantasy, robotics, conversation, generic) already named in decision point 3; community-extensible
   post-1.2. This is just the domain-tag vocabulary the bundle format already carries.
4. **Public reference Oasis** → **DEFER (unchanged): no project-hosted Oasis in 1.2.** The registry
   allows community-run reference entries; the project runs none (operational burden vs. value).
5. **Oasis hardware floor** → **RESOLVE: Mac-Mini-class** (~16 GB unified, CPU inference, optional
   GPU offload) — the transitional Oasis target, matching the doc's tentative and the operator's
   big-mac-mini.

## Explicitly deferred (front-gated OUT of 1.2)

- **Gossip / DHT / mDNS peer discovery** → 1.3+. Static registry + manual add is the 1.2 minimum.
- **Continuous Oasis↔Oasis sync** → 1.3+. 1.2 is pull-releases + push-contributions, not live sync.
- **Sock-puppet-proof consensus** → never promised; practical mitigations only (rate limits +
  tiered trust + provenance blacklists), per the design.
- **Project-hosted public reference Oasis** → post-1.2 (decision 4).
- **ATL / reflexes / cerebellum in the bundle** → the shipped bundle is NAc + EC only; the wider
  bundle the vision sketches (atl.json, reflexes.yaml, cerebellum/) is not required for the 1.2
  taught-want sharing claim and rides the migration registry when it lands.

## Sequencing + dependencies

```
Slice A (signing) ──┐
                    ├─→ Slice B (endpoints) ─→ Slice C (CLI+registry) ─→ Slice D (trust policy)
Gauntlet #3 (poison ┘        (B needs A's verify;    (C is the caller       (D wires the
  slice, already            C needs B; D needs        for A/B)               default policy +
  scheduled)                the Queen tier B/C define)                       promotion gauntlets)
```

Signing (A) is the root — it's the pre-1.2 gate and every other slice references it. Each slice
is its own PR with a two-lens round (the P2P surface is a wire boundary; typing + review pay).
`hivemind/` is already in the CI mypy step (gate 8), so new files there are mypy-clean by contract.

## Discipline hooks

- **Front-gate honored:** only signing and the endpoints/registry/CLI are new; conflict
  resolution, the receiver poison pipeline, auth, admission, and rate-limiting are reused with
  the specific reason named above.
- **Ship-with-a-caller:** no slice reserves capacity without a consumer — the CLI (C) is the
  caller for A/B/D, and D's promotion re-runs the real gauntlets.
- **Two-tier invariants:** every contract above enters `[engineering]` (signing-refusal, tier
  landing, default-Queen-trust) with a guard test; nothing here claims behavioral weight — the
  sharing *claims* are Exp 56 (earned) and Exp 57 (the ladder, in prereg).
- **HTTP via `utils/http`; persistence via `atomic_io`; optional deps via `optional_deps.py`** —
  the cross-cutting core invariants apply to every new file.

## Owner decisions (2026-09-06) — DECIDED

- **Signing algorithm / dep:** **ed25519 via `cryptography`** as a new `[sign]` optional extra
  (real asymmetric verify — a Queen signs with a private key, any consumer verifies with the
  public key). The dependency-free HMAC path is rejected.
- **Slice granularity:** **four PRs, A→B→C→D** as sequenced (no C+D fold).
- **1.2 exchange surface:** **pull-releases + push-contributions.** Live Oasis↔Oasis
  bidirectional sync is explicitly held to 1.3.
