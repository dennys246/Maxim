# Maxim Hivemind + Oasis (federated bio-substrate)

**Status:** ACTIVE design. Implementation phased across 1.1 + 1.2.
**Supersedes:** [archive/mother_maxim_plan.md](archive/mother_maxim_plan.md) (2,224 lines, designed pre-substrate-primary pivot — wrong architecture for the post-pivot world).
**Companion plans:** [grounded_language_acquisition.md](grounded_language_acquisition.md) (substrate-primary AUT mode — the cognition layer the Hivemind shares), [v1_refinement.md](v1_refinement.md) §B5 (the 1.0 shareability infrastructure that enables this).
**Operating context:** the substrate-primary pivot decided 2026-05-09 reframed Maxim's headline thesis from "LLM with bio-augmentation" toward "bio-substrate carries the cognition; LLM is one possible action selector among others." That reframing makes the substrate itself naturally shareable — which the old Mother Maxim plan didn't anticipate, because it was designed when memories (LLM dialogue + episode traces) were the unit of value rather than learned bio-substrate (NAc weights, EC concepts, reflexes).

---

## Vision

Two complementary layers form the federated cognition fabric:

- **Maxim Oasis** — a persistent, substrate-primary Maxim instance that runs on its own hardware (your Mac Mini, my server, anyone's box). The Oasis is a *real bio-agent* — she has her own NAc / EC / ATL / Hippocampus / Default Network / pain system, runs her own agent loop, and processes contributions from connected Maxims as percepts. She is **not a database**. Multiple Oases coexist; nobody is THE Oasis.

- **Maxim Hivemind** — the peer-to-peer substrate-sharing layer that connects Oases to each other and to substrate-primary Maxims. Substrate snapshots (NAc graphs, EC centroids, reflex thresholds, ATL concepts — never raw episodes) flow as portable bundles. Confidence aggregates across instances. Patterns confirmed by many Maxims become axiomatic; patterns from one Maxim stay local.

Both layers run alongside (not replacing) the existing **LLM-AUT mode**, which becomes the Hivemind's primary data source during the 1.1 → 1.2 window.

```
       ┌────────────────────────────────────────────────────────────┐
       │                     Maxim Hivemind                          │
       │  (peer-to-peer substrate-snapshot exchange + aggregation)   │
       └─┬──────────┬─────────────┬────────────┬─────────────┬──────┘
         │          │             │            │             │
       Oasis A    Oasis B      Oasis C      Substrate-     Substrate-
     (Mac Mini) (server)    (RPi cluster)  primary AUT-1  primary AUT-2
        ▲          ▲            ▲             ▲             ▲
        │          │            │             │             │
        │  contributions       │       ┌──────┴──────┐      │
        │  from connected      │       │             │      │
        │  Maxims              │       │   Local     │   sync via
        │                      │       │  learning   │   Hivemind
        │                      │       │             │      │
   ┌────┴──┐   ┌──────┐    ┌───┴───┐   └─────────────┘      │
   │ AUT-A │   │AUT-B │    │ AUT-C │                         │
   │ LLM-  │   │LLM-  │    │ LLM-  │                         │
   │ mode  │   │ mode │    │ mode  │                         │
   └───────┘   └──────┘    └───────┘                         │
       (existing user base — generates substrate as          │
        a side effect of LLM-driven action selection)        │
                                                              │
                          (substrate-primary instances also  ┘
                           contribute as they learn locally)
```

---

## Why this synthesis works (the load-bearing insight)

The old Mother Maxim plan tried to store and redistribute *user memories* (episode logs, dialogue, action traces). That's privacy-heavy, schema-fragile, and creates legitimate "is this even useful?" questions about what aggregation actually adds.

The pivot reframes what's worth sharing. **Every LLM-AUT run already trains a bio-substrate as a side effect** — NAc forms causal links from outcomes, EC clusters concepts, ATL builds semantic structure, hippocampus captures episodes. The substrate is being trained right now by your existing user base, just by an LLM-driven action selector instead of a substrate-driven one.

That changes everything:

1. **The Hivemind has rich seed data immediately.** It doesn't have to wait for substrate-primary mode to mature. Day-1 contributions come from existing LLM-AUT users.
2. **Substrate-primary Maxims bootstrap from accumulated experience.** When substrate-primary mode lands in 1.1, new instances can pull a baseline substrate from the Hivemind rather than starting from zero. This radically accelerates the substrate-primary work.
3. **The privacy surface collapses.** Distilled bio-substrate (NAc weights, EC centroids) doesn't carry the same PII risk as raw episodes. The 700-LOC dual-pass deidentification pipeline from the old plan becomes optional polish rather than a load-bearing safety system.
4. **LLM-AUT mode gets a permanent role.** It's not a transitional thing waiting to be replaced; it's the data-collection mechanism that perpetually feeds the Hivemind. Users running D&D campaigns today are inadvertently training tomorrow's substrate-primary cognition.

The Oasis is the **bridge agent** that absorbs LLM-AUT contributions, processes them via her own bio-stack, distills consensus patterns, and broadcasts to the Hivemind. The Hivemind is the **distribution layer** that lets substrate-primary Maxims pull bootstrap substrate and contribute back as they learn autonomously.

---

## Confound discipline (must preserve)

If substrate-primary Maxims bootstrap from patterns distilled from LLM-AUT experience, the "substrate carries cognition" claim weakens unless we're careful. The discipline that keeps the headline experiment honest:

- **Track "raw substrate" vs "primed substrate" as separate experimental conditions.** A Maxim that converged from zero is the strongest thesis demonstration; a Maxim that bootstrapped from collective experience is a different (also valid) claim.
- **The grounded-language plan's Phase -1 and Phase 0 require raw substrate.** That stays — those phases run with `--hivemind disabled` and must produce findings on un-primed substrate to count.
- **Bootstrap is the end-user convenience path; raw is the research path.** Both ship. They're additive.
- **Provenance tagging on every NAc link / EC node** — every pattern carries a tag for which Maxim it originated from (or "consensus" if aggregated from many). When a substrate-primary Maxim does something interesting, we can trace which patterns are local-learning vs imported.

The discipline is similar to how biology handles this: human children learn from observing adults (priming), but also learn from their own experience (raw). Both are real cognition; the distinction matters for research, not for everyday function.

---

## What's shareable, ranked

| Substrate component | Shareability | Privacy risk | Hivemind value |
|---|---|---|---|
| **Reflex specs + thresholds** | Trivial — pure config | Zero | High (innate-response evolution) |
| **NAc causal weights** (`tool:X → outcome:Y, confidence Z`) | High — natural Bayesian aggregation | Low (no episode content) | Very high — direct cognition transfer |
| **EC concept centroids** | Medium — encoder-dependent | Low (after identity-bearing detection) | High (concept formation transfer) |
| **Cerebellum forward models** | Medium — parameter snapshots | Zero | Medium (motor learning transfer) |
| **ATL semantic concepts** | Medium — depends on cluster purity | Low | Medium (cross-domain bridging) |
| **Hippocampus episodes** | LOW — literal experience records | **HIGH** — full PII surface | Low (single-experience traces, redundant after distillation) |
| **Working memory / current-session state** | None — ephemeral by design | High | Zero |

**Default shareability policy:** ship everything in the top section (reflexes, NAc weights, EC centroids, cerebellum, ATL). NEVER ship hippocampus episodes through the Hivemind. Episodes are local-only by construction.

The single exception: an Oasis may *receive* hippocampus episodes from connected Maxims as private contributions to her own substrate (she processes them as percepts via her bio-stack), but she never re-broadcasts them. Episodes feed her substrate; only her distilled substrate ever leaves.

---

## The Oasis

### What an Oasis is

A persistent, substrate-primary Maxim instance running on stable hardware, configured to:

- Accept substrate contributions from connected LLM-AUT Maxims (over the network or via file drop)
- Process those contributions as percepts through her own bio-stack (NAc/EC/ATL/Hippocampus all run continuously)
- Distill consensus patterns: aggregate NAc links across contributors via Bayesian confidence math; merge EC clusters by centroid proximity; promote reflexes that fired well across many contributors
- Broadcast her distilled substrate to the Hivemind on a configurable cadence (hourly, daily, on-demand)
- Sync substrate snapshots from peer Oases and from substrate-primary Maxim instances

### Why "Oasis"

Bio fit: an oasis is a sustaining gathering place that travelers approach to refresh themselves and that produces life from accumulated water/nutrients. Maxims approach an Oasis to contribute experience and to receive distilled patterns. Multiple oases coexist in a desert ecosystem; none is canonical; together they form a network.

### Hardware footprint

- A Mac Mini, an old laptop, or a Raspberry Pi 5 cluster is sufficient to run an Oasis IF substrate-primary mode is the operating mode (no LLM, just bio-systems running)
- During the transitional phase (1.1) when substrate-primary is still maturing, Oases may need a small LLM (e.g., qwen-7B) to process LLM-AUT contributions and produce reasonable behavior. Hardware budget grows accordingly.
- Once substrate-primary works (post-Phase 0/1), Oases can run pure-substrate and Mac-Mini-class hardware suffices.

### Multiple Oases, no central authority

There is no THE Oasis. Each operator runs their own. Oases sync substrate with each other via the Hivemind. A natural evolution emerges:
- **Personal Oasis** — your Mac Mini, hosting substrate from your own Maxims and friends
- **Community Oasis** — a Discord server or research group running an Oasis their members contribute to
- **Public Oasis** — eventual reference instances at e.g. `oasis.maxim-project.org` that anyone can connect to

The Hivemind doesn't care which kind of Oasis you connect to; it's all peer-to-peer substrate exchange.

---

## The Hivemind

### What the Hivemind is

The Hivemind is the protocol + data format + exchange mechanism for substrate snapshots between Oases and substrate-primary Maxims. It is not a service; nobody hosts "the Hivemind." It's a peer-to-peer convention with optional well-known reference servers.

### Substrate snapshot bundle (the unit of exchange)

A versioned, signed archive containing:

```
maxim-substrate-v1.0/
├── manifest.json           # version, contributor metadata, signature, domain tags
├── nac.json                # causal links + confidence + provenance tags
├── ec.json                 # concept centroids + cluster metadata
├── atl.json                # semantic concepts
├── reflexes.yaml           # innate response specs
├── cerebellum/             # forward model parameters (binary)
└── README.md               # optional human-readable description
```

The bundle format extends the existing `_format_version` contract per [CLAUDE.md](../../CLAUDE.md) v1.0 freeze. Schema-stable, version-aware, scrubbable.

### Substrate domains

Bundles are tagged by **substrate domain** (combat, cooking, medical, fantasy, robotics, conversation, ...). Subscribers opt into specific domains. This serves two goals:

1. **Relevance** — a medical-AI hobbyist doesn't want combat patterns polluting their substrate
2. **Curation** — domain maintainers can curate their domain's substrate (flag bad patterns, validate new contributions)

Default domains ship with the project; users can define new ones.

### Conflict resolution

Two Maxims learn opposite things ("X is good" vs "X is bad"). On merge, the Hivemind uses NAc's existing confidence math:

- **Confidence aggregates Bayesian-style** across contributors
- **Multi-source confirmation** weighted higher than single-source
- **Outcome valence** (positive vs negative) preserved as separate distributions, not collapsed
- **Provenance** preserved so a substrate-primary Maxim can selectively trust patterns from specific contributors

When real conflict persists (e.g., 50 Maxims learned "X is good" and 50 learned "X is bad"), the Hivemind preserves both with full distributions. The local Maxim's behavior remains coherent because it weights its own learning + recent context above any single Hivemind pattern.

### Poison resistance

Adversaries try to corrupt the Hivemind by spreading bad learning. Defenses:

1. **Multi-source consensus** — patterns require N independent contributors before promotion
2. **Domain curation** — domain maintainers can flag specific contributors as untrusted
3. **Local provenance tracking** — every pattern carries source tags; receivers can blacklist sources
4. **Outcome correlation check** — if Hivemind-derived patterns lead to negative-valence outcomes locally, weight them lower over time (the substrate naturally distrusts patterns that hurt it)
5. **Identity-bearing concept detection** — patterns that map to specific named entities (likely PII or impersonation) are quarantined automatically

This won't be perfect. Pure peer-to-peer systems with open contribution always have an arms race with abuse. Mitigations are practical, not theoretical.

---

## Architecture details

### Three-layer flow

```
LLM-AUT Maxim (running on user's machine)
       │
       │ 1. Generates experience: action selection by LLM,
       │    bio-substrate learns from outcomes as side effect
       │
       │ 2. User opts in: contribute substrate snapshot
       │    (optionally on session-end, optionally periodic)
       ▼
Maxim Oasis (running on operator's machine)
       │
       │ 3. Processes contribution as percept-stream:
       │    her own NAc/EC/ATL absorb the patterns
       │
       │ 4. Continuously distills: aggregates across all
       │    contributors, prunes low-confidence, promotes consensus
       │
       │ 5. Periodically publishes: distilled substrate
       │    snapshot pushed to Hivemind
       ▼
Maxim Hivemind (peer-to-peer protocol)
       │
       │ 6. Substrate snapshots flow between Oases,
       │    between substrate-primary Maxims, and bidirectionally
       │
       ▼
Substrate-primary Maxim (1.1+, bootstrapped from Hivemind)
       │
       │ 7. Pulls bootstrap substrate from Hivemind
       │
       │ 8. Learns autonomously from local experience
       │
       │ 9. Contributes its own learning back to Hivemind
       │    (closing the loop)
```

### Key design decisions

- **Oases are full agents, not databases.** Each Oasis runs her own bio-stack and has her own emergent behavior. Querying her is asking *her*, not searching a record store.
- **No hierarchy.** Multiple Oases coexist; no single "canonical" Oasis. The Hivemind mesh has no root.
- **Distillation, not replication.** The Hivemind ships *learned patterns*, not raw episodes. Privacy and scale both benefit.
- **Opt-in throughout.** Users opt in to contributing. Operators opt in to running an Oasis. Substrate domains are opt-in subscriptions. Nothing is automatic; nothing is centralized; nothing leaves a Maxim without explicit consent.
- **Provenance is preserved, not erased.** Every pattern carries source tags. This enables trust management, debugging, and the raw-vs-primed experimental distinction.

---

## Phasing

| Version | Substrate-primary | Oasis | Hivemind |
|---|---|---|---|
| **1.0** | B5: Phase -1 + Phase 0 harness + shareability infrastructure (snapshot bundle format, NAc merge ops, EC merge ops, provenance tagging, identity-bearing concept detection, substrate domains). ~700 LOC. | Format exists. No Oasis software runs yet. | None. |
| **1.1** | Substrate-primary AUT mode lands. Phase 0 validation runs (raw substrate, no Hivemind). Phase 1 starts. | **Oasis software ships** (~800 LOC). Single-Oasis instance hostable on a Mac Mini. CLI: `maxim oasis serve`. LLM-AUT users can opt in to contribute via `maxim contribute --to oasis://...`. Each Oasis can sync substrate with another Oasis directly (`maxim oasis sync --peer ...`). | Direct Oasis-to-Oasis sync only. No mesh discovery yet. |
| **1.2** | Phase 1 ships. Phase 2 starts. Substrate-primary Maxims pull bootstrap from Hivemind. | Multi-Oasis federation. Curation tools (mark-trusted, mark-untrusted). Domain maintainer roles. | **Full Hivemind protocol** (~600 LOC): peer discovery, substrate-snapshot exchange protocol, conflict-resolution semantics, poison-resistance defenses, well-known reference servers (optional). |
| **1.3+** | Phase 3 starts (from-scratch sequence model). | Oasis becomes a substrate-primary instance (no LLM needed). Mac-Mini-class hardware suffices. | Cross-version migration tooling. Curation registry. Domain ecosystem. |

**Total scope across 1.0 + 1.1 + 1.2:** ~2,100 LOC of net-new work. The old Mother Maxim plan was 3,800 LOC; this synthesis is ~55% the scope and does more.

---

## What this changes about B5 (the 1.0 work)

The grounded-language plan's B5 (Phase -1 + Phase 0 harness, ~700 LOC) needs the shareability infrastructure baked in from day one. Without it, the Hivemind becomes a 1.3+ retrofit (expensive). With it, the Hivemind is a 1.2 turn-on.

**Concretely added to B5:**

| Feature | Approx LOC | Purpose |
|---|---|---|
| Substrate snapshot bundle format (zip + manifest + signature) | ~150 | Shareable unit |
| `nac.merge(other_nac)` / `ec.merge(other_ec)` Bayesian-aggregation library functions | ~200 | Conflict resolution math |
| Provenance tags on NAc links + EC nodes (source instance ID) | ~100 | Trust management + experimental hygiene |
| Identity-bearing concept detection (port from old Mother plan, simplified) | ~80 | Privacy from EC clusters that map to PII |
| Substrate domain tagging (per-pattern domain field) | ~50 | Selective sharing |
| `maxim substrate export` / `maxim substrate import` CLI verbs | ~80 | Manual exchange (no service required yet) |

**Total B5 add: ~660 LOC.** Brings B5 to ~1,360 LOC total. Still much smaller than the old Mother plan and now enables both substrate-primary mode AND the eventual Hivemind.

---

## Privacy + safety

The old Mother Maxim plan dedicated 700 LOC to dual-pass deidentification because it shipped raw episodes. This plan ships only distilled substrate, which is dramatically less PII-bearing. The privacy surface collapses to two real concerns:

1. **EC concept clusters that map to identity-bearing tokens** (e.g., a cluster centered on the user's name). The "bio-systems already know what identities are" detection from the old plan handles this — port the idea, simplify the implementation. ~80 LOC in B5.

2. **Substrate-pattern triangulation** — could an adversary infer specific user behavior from aggregated NAc weights? In principle, yes, after enough cross-contribution observation. Mitigations:
   - High contributor-count requirement before publication (k-anonymity for substrate)
   - Differential-privacy-style noise injection on confidence values (post-1.2)
   - Operator-level opt-in to triangulation-resistant sharing modes

The hippocampus-episodes-stay-local rule is the load-bearing privacy invariant. Episode-level data never leaves a Maxim through the Hivemind. Period.

---

## Open hard problems

These don't block 1.0 or even 1.1, but need design before any cross-Oasis sharing goes live (1.2):

| Problem | Why it's hard | When |
|---|---|---|
| **Schema evolution across Maxim versions** | NAc structure changes between v1.0 and v1.1. How do v1.0-distilled patterns merge with v1.1 substrates? | 1.2 — when first cross-version share happens. Mitigated by `_format_version` infrastructure + migration registry. |
| **Catastrophic forgetting via merge** | Pulling in a large bootstrap could overwrite carefully-learned local patterns. | 1.2 — solved by provenance + merge weighting (local Maxim's own patterns weighted higher). |
| **Pattern-quality filtering** | LLM-AUT might do something stupid that the substrate over-learns. Mother's aggregation needs to filter on multi-source consensus + outcome-valence. | 1.1 — Oasis distillation engine. Handled by requiring N-source consensus + valence weighting. |
| **Poison resistance under sock-puppet attack** | Adversary spins up many fake Maxims to inject poisoned patterns. | 1.2+ — practical mitigations only (rate limits, domain curation, provenance blacklists). No theoretical solution. |
| **Discovery and bootstrap** | First user has no patterns. How does a fresh Maxim find an Oasis? | 1.2 — published reference Oases + manual peer addition. |

---

## Glossary

| Term | Definition |
|---|---|
| **Maxim Hivemind** | The peer-to-peer substrate-sharing layer. Not a service; a protocol + data format. Acceptable shorthand: "Hivemind", "the Hive", "the network". |
| **Maxim Oasis** | A persistent substrate-primary Maxim instance that absorbs contributions and broadcasts distilled substrate. Acceptable shorthand: "Oasis", "your Oasis". |
| **Substrate snapshot bundle** | The portable, versioned, signed archive containing one Maxim's substrate. The unit of exchange in the Hivemind. |
| **Substrate domain** | A tag (combat, cooking, medical, ...) that scopes substrate sharing. Subscribers opt into specific domains. |
| **Contribution** | A substrate snapshot sent from a Maxim to an Oasis (or directly to the Hivemind). Always opt-in. |
| **Distillation** | The Oasis's process of aggregating contributions into consensus patterns. Uses NAc Bayesian confidence math + EC centroid merging. |
| **Provenance tag** | A source-instance-ID attached to every NAc link / EC node, enabling trust management and the raw-vs-primed experimental distinction. |
| **Raw substrate** | A Maxim's substrate built only from its own experience. The headline-experiment configuration. |
| **Primed substrate** | A Maxim's substrate bootstrapped from the Hivemind. The end-user-convenience configuration. |
| **Mother Maxim** | (Deprecated.) The old name for what is now the Maxim Oasis. Retained in archive for historical context. |

---

## Decision points still open

1. **CLI naming convention** — `maxim oasis ...` and `maxim hivemind ...` (long but clear) vs `maxim oasis ...` and `maxim hive ...` (shorter, branded "Hive" as the network). Tentative: `oasis` + `hivemind` formal, `hive` as user-facing shorthand.
2. **Bundle signing** — required signatures on contributions (more friction, more trust) vs optional (easier onboarding, weaker trust). Tentative: optional in 1.1, recommended in 1.2, required for promoted-to-canonical-domains in 1.3.
3. **Substrate domains shipped at 1.0** — define a starter set (combat, cooking, medical, fantasy, robotics, conversation, generic) vs let the community define them post-1.2. Tentative: starter set in B5 to seed the format; community-extensible later.
4. **Public reference Oasis** — does the project run a public Oasis at e.g. `oasis.maxim-project.org`? Operational burden vs. ecosystem-bootstrapping value. Tentative: defer to post-1.2 once the protocol is stable; community can run its own reference Oases meanwhile.
5. **Hardware floor for an Oasis in 1.1** — does the 1.1 Oasis (which still needs an LLM during the transitional phase) target Mac-Mini-class hardware (~16GB unified memory + CPU inference) or assume a leader-class GPU? Tentative: Mac-Mini-class with optional GPU offload.

---

## Iteration log

### 2026-05-09 — Plan created from synthesis

Triggered by:
1. The 2026-05-09 substrate-primary pivot (see [grounded_language_acquisition.md](grounded_language_acquisition.md) iteration log).
2. The user's observation that LLM-AUT mode can serve as a perpetual data-source for substrate distillation, even before substrate-primary mode matures — which rescues the core of the old Mother Maxim plan and gives the Hivemind seed data from day one.
3. The user's framing of the architecture as "collective Hivemind + environment Oasis providing perception" — the metaphor that crystallized the naming.

Supersedes the 2,224-line `mother_maxim_plan.md` (now archived). The old plan's good ideas preserved:
- "Mother is a full agent, not a database" → Oasis is a full substrate-primary Maxim
- Bio-system identity-bearing concept detection → ported into B5's identity scrubbing
- Multi-tenant separation → preserved via substrate domains + opt-in subscription
- Memory coalescence engine → reframed as the Oasis's distillation process

Old plan ideas DROPPED:
- Pecking Order Graph hierarchy → flat peer-to-peer mesh instead
- Database backend (SQL) → portable file-based substrate snapshots instead
- Dual-pass deidentification (700 LOC) → reduced to ~80 LOC identity-bearing concept detection because we never ship raw episodes
- Central Mother server → multiple peer Oases with no canonical authority
- REST `/v1/contribute` and `/v1/recall` API → Hivemind peer-to-peer protocol (1.2)

Names changed:
- "Mother Maxim" → "Maxim Oasis" (multiple instances, sustaining gathering place metaphor)
- The implicit-collective-network → "Maxim Hivemind" (explicit collective cognition layer)
