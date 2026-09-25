# Public Oasis on own hardware (`oasis.pymaxim.bio`)

**Status:** PROPOSED 2026-09-19 (**Phase 0 SCHEDULED 2026-09-24** — [social_referencing.md](social_referencing.md) depends on it) — scoping pass only, no code written, nothing deployed. Produced
from a read of the frozen [sharing_threat_model.md](sharing_threat_model.md), the shipped
`src/maxim/hivemind/` receiver, and the Slice D deferral in
[hivemind_p2p_scope.md](archive/hivemind_p2p_scope.md). **Recommendation: publish, do not accept.**
**Consumer (2026-09-24):** [social_referencing.md](social_referencing.md) depends on this plan's **Phase 0**, which is **SCHEDULED** for that reason (owner, 2026-09-24); its release mirror may read signed releases from a Phase 1 Oasis anonymously and read-only (never with the leader key), but does not require Phase 1 (a private `oasis serve` suffices).
Phase 1 is buildable now and adds no new trust decision; Phase 2 stays deferred behind conditions
this document names. Merging this plan is not the decision — see §The decision record this needs.

**Sibling plan, deliberately not duplicated:** [deferred/hf_hive_repository.md](deferred/hf_hive_repository.md)
already decomposes *public distribution vs public submission* and its trust argument (§2) applies
verbatim here. That plan's host is Hugging Face; this one's host is the owner's own hardware at a
domain he controls. This document owns only what differs — the self-hosted deployment, the
multi-site question, the shared-credential gap, and decision point 4 — and cites the HF plan for
the rest.

---

## The one fact that decides it

**There is no "accept from a stranger" mode in this codebase, and its absence is structural, not an
oversight.** `hivemind/ingest.py::ingest_bundle` refuses any bundle whose `manifest.contributor_id`
is not in the operator-supplied `trusted_sources` set — *"Refusal, never admit-with-clamps"* — and
the operator surface `maxim substrate ingest --trust` declares that flag `required=True`
(`hivemind/cli.py`, the `p_ingest` parser). Every mechanism behind that door is designated
defense-in-depth by the frozen threat model, never the boundary: the trust-gating row of §3 says
`trusted_sources` *"is defense-in-depth, not the trust boundary (V1)"*, and §4 row A says the clamps
*"bound magnitude, not INTENT."*

So a public Oasis that merges strangers' bundles does not *weaken* the trust model — it deletes the
only component of it that is a trust decision, and substitutes nothing. The thing that would
substitute is the Queen-tier promotion gauntlet, which was deferred 2026-09-06 with four named
prerequisites (bundle→gauntlet adapter, `sign_existing_bundle`, Gauntlet #3, a promotion provenance
link). **Zero of the four have landed**, and [roadmap_1_4.md](roadmap_1_4.md) §What is NOT in 1.4
keeps promotion *"unchanged from 1.2's WRITE-ONLY posture."*

## Front-gate scope pressure

*Does this need to be its own mechanism, or can it ride on existing infrastructure?*

**It rides entirely, for Phase 1 — no new mechanism at all.** The pieces already exist and already
compose:

- `maxim oasis serve` injects an `OasisStore` into the leader proxy (`hivemind/oasis_cli.py::_run_serve`).
- `GET /v1/substrate/releases` and `/v1/substrate/bundle/<id>` are routed **before** the proxy's
  bearer check in `runtime/leader_proxy.py::LeaderProxyHandler.do_GET`, so the release tier is
  already a public read surface by design.
- `hivemind/store.py::OasisStore.publish_release` refuses an unsigned bundle, so the release tier
  cannot hold unsigned content.
- Consumers verify against the Queen key, never the host that served the bytes
  (`hivemind/ingest.py::ingest_bundle` under `require_signed`), which is why the distribution
  channel is untrusted by construction.
- Cloudflare Tunnel to the rig is the recorded pattern for exposing a live service.

Phase 1 therefore adds **documentation, a format-freeze pass and a privacy read** — no `src/`
change is required to stand it up. That is the front-gate answer: the only thing being built is a
deployment and a promise about it.

Phase 2 (accepting submissions into trusted state) *is* its own mechanism, several of them, and
that is precisely the argument for not doing it yet.

## What the receiver enforces today, and where it stops

Traced end to end through `ingest_bundle` for one hostile bundle. **Bounded and refused** (each with
a guard in `tests/unit/test_hivemind_ingest.py`): zip bombs, on the *actual* decompressed byte count
(V6, amended 2026-09-05 after a binary-patched header routed the declared-size gate); `NaN`/`Infinity`
(which `json.loads` accepts); unbounded counts (capped at `MAX_FOREIGN_COUNT = 1_000`); list fields
capped below the merge's tail-truncation windows so foreign material cannot evict local history;
unstamped foreign geometry; reserved `_*` domains; `NAC_KEY_SEP` bytes in keys; undeclared ZIP
members; replays (digest dedup); identity-bearing content, re-checked receiver-side rather than
trusted from the sender; inherent-class markers from anyone but a Queen. Two asymmetric protections
are genuinely strong: `merge.py::tighten_negative_biases` means *"ship enthusiasm to erase fear"*
stops working, and foreign fear is discounted rather than trusted at face value.

**What gets through, if you put the stranger on the allowlist** — which is what "public" means:

A well-formed, correctly signed bundle whose `ec.json` nodes legitimately align to the receiver's
clusters and whose `cluster_reward_bias` entries sit inside `[-1, 1]` and steer action selection
toward an attacker-chosen tool. Every clamp passes because the values are *in range*; counts sit
under the cap; geometry is stamped; the scrub finds no identity-bearing strings; the signature
verifies because it was signed with the stranger's own key, which the operator chose to trust. The
tighten-only clamp does not help: it skips any key the receiver did not already hold at a negative
value (`merge.py::tighten_negative_biases`, the `rv >= 0.0 or key not in merged_field` guard), so
introducing a *new* want is unimpeded. Free text in merged keys reaches prompt annotation (§4 row L).

**The first thing an adversary would try is therefore not a malformed bundle** — the adapter is
genuinely good at those — but a well-formed one that is simply a lie about what an agent learned.
Nothing in the pipeline evaluates whether a want is *true*, and the threat model says so by
declaration: behavioral vetting of a want's content *"is not a security gate — a malicious want that
passes the gauntlet is V1's problem."* V1 is the allowlist. Public removes the allowlist.

## What is missing for public acceptance

| Capability | State |
|---|---|
| Contributor identity beyond a signature | **Neither.** A signature proves key-possession, not who. No reputation/scoring exists (`grep -rniE "reputation\|trust_level\|karma" src/maxim/hivemind/` → empty) and it is an explicit non-goal in the HF plan. |
| Quarantine a human promotes from | **Storage yes, review no.** `store.py::accept_contribution` lands the bundle in `experimental/` with a provenance record and *"a contribution's arrival changes no trusted state."* But `maxim oasis` has exactly three verbs — `serve`, `publish`, `status` — so there is no verb to inspect, score, or promote. `OasisStore.open_contribution` has zero production callers, named as reserved capacity. |
| Rate limiting | **Partial and off.** `runtime/rate_limiter.py::PeerRateLimiter` is generic, keyed by **source IP** not contributor, and defaults to **unlimited** (`MAXIM_PROXY_RATE_LIMIT_RPM=0`). One real cap exists: a 16 MiB body limit. Nothing contributor-aware lives in `hivemind/`. |
| Revoking something already merged | **Absent.** `IngestionJournal.add_tombstone` is forward-only (it refuses *future* bundles) and has no CLI caller. `merge.py::prune_nac_cluster_biases` removes cluster-keyed `cluster_reward_bias`/`cluster_fear`/`reward_bias` entries **only** — it cannot undo EC node inserts, `percept_valences`, `predicted_value`, or merged counts, and it selects by cluster id, not by contributor or digest. Nothing connects the journal's attribution to the prune; the threat model names that link as owed (V5). |
| Abuse/moderation for well-formed garbage | **Neither**, and named as an owed threat-model addendum: public submissions add spam, multi-identity slow-poison campaigns and gauntlet-overfit bundles *"that the receiver-side V1–V10 belt does not address."* |

One further gap, specific to self-hosting and not covered by any existing plan: **`oasis serve`
authenticates with the leader's single bearer key** (`hivemind/oasis_cli.py::_run_serve` calls
`tunnel.keys.read_key()`), and `leader_proxy.py::LeaderProxyHandler._check_auth` compares one
`api_key` for every route. The same credential that would let a stranger POST a contribution also
authorizes `/v1/chat/completions` inference against the leader's GPU. There is no per-capability
scoping and no per-contributor credential. The Console's authorization-tier work
([deferred/console_tunnel_hardening.md](deferred/console_tunnel_hardening.md) PR 4) covers
`maxim serve`, a *different* server — so this gap has no plan today.

## Phases

### Phase 0 — prerequisites (SCHEDULED 2026-09-24: [social_referencing.md](social_referencing.md) S0 depends on it)

Cheap, and all of it is owed regardless of whether Phase 1 ships.

1. **A CI lane that installs `cryptography`.** Today no lane does (`.github/workflows/test.yml`
   installs `.[semantic,test]`), so all 14 tests in `tests/unit/test_hivemind_signing.py` and the
   signed-bundle arms of `test_oasis_store.py` are **skipped in CI**. They pass locally in 0.77 s —
   the mechanism works; it is simply unguarded. Signature verification is the only thing that raises
   the trust boundary above "the channel", so publishing bundles the world verifies while its guard
   never runs in CI is the vacuous-guard shape. **This is already 1.3.1's top item** — Phase 1 waits
   for it rather than duplicating it. *(2026-09-25: built — the `unit-tests` job installs the `console`
   + `sign` extras from `pyproject.toml` and runs with `--require-extras=console,sign`, which turns a skip
   for a missing required extra into a failure: the positive control.)*
2. **Public format-freeze pass.** Manifest fields, slice shapes and the §5 adapter constants become
   a public compatibility surface the moment a stranger downloads a bundle. A named gate, not a
   formality (HF plan §3).
3. **Publication-privacy read.** Bundles are NAc+EC only and identity-scrubbed twice, but cluster,
   percept and goal keys carry sim-derived text. One human read of the actual exemplar's key
   material before the first upload.
4. **Licensing posture** for published bundles — trained state, not code. Pick before the first
   upload, not after (HF plan Q5).
5. **Pulls never send the leader key** (added 2026-09-24, shared with social_referencing S0):
   `hive_cli.py::_run_pull` defaults to `read_key()`, the leader key that also grants inference, so any
   registered Oasis receives it. Reads become anonymous and rate-limited, or use a read-only scoped
   credential.
6. **`signer_identity` covered by the signature** (added 2026-09-24, shared with social_referencing
   S0): today it can be relabelled without breaking verification.
7. **Releases carry a Queen-signed entry index** (added 2026-09-24) — **owed by social_referencing
   only**, not regardless of Phase 1 (a new release-format feature for its local selection and
   per-entry journal, so a consumer verifies the Queen per entry without server-cut slices). It must
   **land before item 2**, or the format freeze reopens at once.

### Phase 1 — publish only (entry: Phase 0 complete)

Serve the Queen release tier read-only at `oasis.pymaxim.bio`, from the rig, over a Cloudflare
Tunnel. `maxim oasis serve` with a bearer key set, `maxim oasis publish` for each signed release.
The GET routes are public by design; the POST route stays closed because the key is never shared.

Exemplar content: the Exp 56 taught-want release and the 1.3.0 survival lineage — the bundles the
published claims were earned on, so the public artifact and the published claim are the same object.

**Multi-site, concretely.** The release tier is content-addressed immutable blobs, so a second site
is `rsync` plus DNS — no merge story is needed and **no replication is required to start**. The
experimental tier is *not* safely multi-writer: `contributions.json` is a read-modify-write file
guarded by a process-local `threading.Lock`, so two sites accepting writes lose audit records with
no CRDT to reconcile them. And merged substrate has no multi-writer story at all — `maxim substrate
ingest` requires its receiver session to be **at rest**. The deployment that works is **one writer,
N read-only mirrors**; anything else needs a merge story that does not exist.

**Exit criterion:** a stranger pulls a release with `hive add` + `hive pull` and the signature
verifies against the published Queen key. That is the whole proof.

### Phase 2 — accept submissions (entry: DEFERRED; all conditions, not a majority)

Carried from the HF plan's revive trigger, with two additions this deployment forces. **All six:**

1. The four Slice D promotion prerequisites have landed.
2. *"Validated"* has a defensible meaning for an arbitrary foreign bundle. Today R1 bounds the
   shared want to an exact-key cache entry, so validation is domain- and layout-specific by
   construction — it must be stated that way on any public stream, never rounded up to "safe and
   useful for you."
3. The owner deliberately re-opens decision point 4 (§The decision record this needs).
4. [sharing_threat_model.md](sharing_threat_model.md) has a **dated §4 addendum** for repo-side
   threats (spam, multi-identity campaigns, gauntlet-overfit bundles). If Phase 2 changes any
   receiver validation duty in §5, that requires a dated amendment there — the freeze's own change
   mechanism, and this plan asserts no exemption from it.
5. **Per-capability credentials** on the substrate endpoints, so a contribute token is not an
   inference token.
6. **A revocation story**, because tombstones are forward-only and the prune cannot reach EC
   inserts or valences. Accepting material you cannot un-accept is the part that does not decay
   gracefully.

**A vacuous gauntlet is worse than none** — a rubber-stamp validator on a public stream is the
vacuous-guard shape with the project's signature on the output.

## What this plan explicitly does NOT do

- **Does not accept a stranger's bundle into trusted state.** Not in Phase 1, not conditionally,
  not behind a flag.
- **Does not open `POST /v1/substrate/contribute` to the public**, even into the quarantine tier,
  until per-capability credentials exist — the token is currently also an inference token.
- **Does not make `pymaxim.bio` a hub.** The Oasis is served from the rig on its own subdomain via
  tunnel; the Astro site keeps linking to Oases rather than routing to them. The discovery-only
  stance is preserved, not reversed (see below).
- **Does not build reputation, auto-promotion, live Oasis↔Oasis sync, or unsigned release
  content** — carried non-goals.
- **Does not claim research value.** See below.
- **Does not touch `src/`.**

## What the research gets from it: nothing, and the project already says so

Plainly: this is product work, and calling it research would be dishonest. Every EARNED row in this
repo is a controlled experiment with pre-registered arms, a frozen metric and a refusal condition. A
public pool produces uncontrolled, unattributable, un-preregistered material with no arms — it
cannot produce an EARNED row, and [maxim_hivemind.md](maxim_hivemind.md) §Confound discipline
already classifies it: *"Bootstrap is the end-user convenience path; raw is the research path."*
Phase -1/Phase 0 work is required to run on un-primed substrate to count at all.

There is one honest future research asset: a corpus of real foreign bundles is good adversarial test
material *for a gauntlet*. But that is an instrument, it requires the gauntlet to exist first, and it
is a Phase 2 byproduct — not a reason to open submissions.

## Cost to the roadmap

- **Phase 0 + 1:** days, not weeks, and item 1 is already 1.3.1 scope (items 5–7, added 2026-09-24,
  add perhaps a week: two hardening fixes and one release-format feature). Slips nothing if Phase 1
  waits for the crypto lane.
- **Phase 2:** reopens Slice D plus a threat-model addendum, per-capability auth, a reviewer verb,
  revocation and a gauntlet that is *"partly a research question, not a feature."* That is a
  release's worth of work and would displace 1.4's ladder.

## The decision record this needs

**The premise worth correcting:** a public Oasis at `oasis.pymaxim.bio` does **not** reverse the
discovery-only website stance, provided the conditions in §What this plan explicitly does NOT do
hold. That stance is about the *website* — [deferred/maxim_console.md](deferred/maxim_console.md):
*"The website is at most a read-only directory of public Oases (a phonebook), never a hub instances
connect to."* An Oasis served from the rig on its own subdomain is a peer in the mesh, which
[maxim_hivemind.md](maxim_hivemind.md) already anticipates (*"Public Oasis — eventual reference
instances … that anyone can connect to"*), under a topology that stays flat (*"No hierarchy … The
Hivemind mesh has no root"*).

What it **does** reverse is **decision point 4** — [hivemind_p2p_scope.md](archive/hivemind_p2p_scope.md)
§Decision points: *"no project-hosted Oasis in 1.2. The registry allows community-run reference
entries; the project runs none (operational burden vs. value)."* That was an **operational-burden**
decision, and Phase 1's burden is genuinely small: static signed blobs, no inbound trust, no
curation labor. The decision record accompanying this plan re-opens point 4 **for publication only**
and re-affirms the deferral for submissions.

A note on honesty about who decides: the DECISIONS.md entry in this plan's PR is a proposal until
the owner merges it. It is a separate commit so the plan can land without it.

## Open questions

- **Q1 — static Oasis contract?** Can a dumb HTTPS host serving `releases.json` + the blobs satisfy
  the existing client, or does `hivemind/substrate_client.py` assume endpoint behavior a static
  mirror cannot provide? Decide smallest-first at build time; the fallback is only justified with
  the specific incapacity named.
- **Q2 — availability posture.** A published Oasis people pull from is an uptime promise. What is
  the stated SLA, and what does the site say when the rig is down for an experiment?
- **Q3 — does this compete with the HF plan's Phase 1, or complement it?** Both are read-only
  distribution. Running both means two mirrors of one release tier — cheap, but the index must not
  fork.

## Related

- [sharing_threat_model.md](sharing_threat_model.md) — FROZEN 2026-09-04; the receiver contract
  Phase 2 owes a repo-side addendum.
- [deferred/hf_hive_repository.md](deferred/hf_hive_repository.md) — the sibling plan; §2 carries
  the trust argument this document cites rather than restates.
- [archive/hivemind_p2p_scope.md](archive/hivemind_p2p_scope.md) — Slice D deferral, its four
  prerequisites, and decision point 4.
- [archive/oasis_ingestion_contract.md](archive/oasis_ingestion_contract.md) — the V1–V10 step order.
- [maxim_hivemind.md](maxim_hivemind.md) — the vision authority; trust topology and confound
  discipline.
- [roadmap_1_3_x.md](roadmap_1_3_x.md) — the crypto CI lane Phase 0 depends on.
