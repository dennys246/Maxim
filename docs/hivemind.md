# Maxim Hivemind + Oasis

> **Status:** Design complete (2026-05-09). Implementation phased: shareability infrastructure in 1.0 (B5), Oasis software in 1.1, full Hivemind P2P protocol in 1.2. **Supersedes the older Mother Maxim plan.**

## What this is

The **Maxim Hivemind** is a peer-to-peer substrate-sharing layer that lets multiple Maxim instances exchange learned bio-substrate (NAc causal weights, EC concept centroids, reflex thresholds, Cerebellum forward models, ATL semantic concepts). It is **not** a service — nobody hosts "the Hivemind." It is a protocol + portable data format + exchange convention.

A **Maxim Oasis** is a persistent, [substrate-primary](substrate_primary.md) Maxim instance that runs on operator hardware (a Mac Mini, a server, an old laptop, a Raspberry Pi 5 cluster), accepts substrate contributions from connected Maxims, processes those contributions as percepts through her own bio-stack, distills consensus patterns, and broadcasts to the Hivemind. **Multiple Oases coexist; nobody is THE Oasis.** Each Oasis is a real bio-agent in her own right, not a passive database.

The two layers together form the federated cognition fabric:

```
       ┌────────────────────────────────────────────────────────────┐
       │                     Maxim Hivemind                          │
       │  (peer-to-peer substrate-snapshot exchange + aggregation)   │
       └─┬──────────┬─────────────┬────────────┬─────────────┬──────┘
         │          │             │            │             │
       Oasis A    Oasis B      Oasis C      Substrate-     Substrate-
     (Mac Mini) (server)    (RPi cluster)  primary AUT-1  primary AUT-2
        ▲          ▲            ▲             ▲             ▲
        │  contributions       │       ┌──────┴──────┐      │
        │  from connected      │       │   Local     │   sync via
        │  Maxims              │       │  learning   │   Hivemind
        │                      │       └─────────────┘      │
   ┌────┴──┐   ┌──────┐    ┌───┴───┐                         │
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

## Why this exists

The headline insight that motivated promoting the old Mother Maxim plan from deferred to active:

**Every LLM-AUT run already trains a bio-substrate as a side effect.** When a user runs `maxim --sim "...some D&D campaign..."`, the bio-systems are running underneath: NAc forms causal links from tool outcomes, EC clusters concepts, ATL builds semantic structure, Hippocampus captures episodes. The substrate is being trained right now by your existing user base, just by an LLM-driven action selector instead of a substrate-driven one.

That changes the federated-learning calculus radically:

1. **The Hivemind has rich seed data immediately.** It doesn't have to wait for substrate-primary mode to mature. Day-1 contributions come from existing LLM-AUT users.
2. **Substrate-primary Maxims bootstrap from accumulated experience.** When substrate-primary mode lands in 1.1, new instances can pull a baseline substrate from the Hivemind rather than starting from zero. Massively accelerates substrate-primary development.
3. **The privacy surface collapses.** Distilled bio-substrate (NAc weights, EC centroids) doesn't carry the same PII risk as raw episodes. The 700-LOC dual-pass deidentification pipeline from the old Mother Maxim plan becomes optional polish rather than load-bearing safety.
4. **LLM-AUT mode gets a permanent role.** It's not transitional waiting to be replaced; it's the perpetual data-collection mechanism that feeds the Hivemind. Users running D&D campaigns today are inadvertently training tomorrow's substrate-primary cognition.

## What's shareable — and what stays local

| Substrate component | Shareability | Privacy risk | Hivemind value |
|---|---|---|---|
| **Reflex specs + thresholds** | Trivial — pure config | Zero | High (innate-response evolution) |
| **NAc causal weights** (`tool:X → outcome:Y, confidence Z`) | High — natural Bayesian aggregation | Low (no episode content) | Very high — direct cognition transfer |
| **EC concept centroids** | Medium — encoder-dependent | Low (after identity-bearing detection) | High (concept formation transfer) |
| **Cerebellum forward models** | Medium — parameter snapshots | Zero | Medium (motor learning transfer) |
| **ATL semantic concepts** | Medium — depends on cluster purity | Low | Medium (cross-domain bridging) |
| **Hippocampus episodes** | LOW — literal experience records | **HIGH** — full PII surface | Low (single-experience traces, redundant after distillation) |
| **Working memory / current-session state** | None — ephemeral by design | High | Zero |

**Default policy:** ship everything in the top section. **NEVER** ship hippocampus episodes through the Hivemind. Episodes are local-only by construction. The single exception: an Oasis may *receive* hippocampus episodes from connected Maxims as private contributions to her own substrate (she processes them as percepts), but she never re-broadcasts them. Episodes feed the Oasis's substrate; only the Oasis's distilled substrate ever leaves.

## The Oasis

### What an Oasis does

A persistent substrate-primary Maxim instance running on stable hardware, configured to:

- Accept substrate contributions from connected LLM-AUT or substrate-primary Maxims (via network or file drop)
- Process those contributions as percepts through her own bio-stack (NAc/EC/ATL/Hippocampus all run continuously)
- Distill consensus patterns: aggregate NAc links across contributors via Bayesian confidence math; merge EC clusters by centroid proximity; promote reflexes that fired well across many contributors
- Broadcast her distilled substrate to the Hivemind on a configurable cadence (hourly, daily, on-demand)
- Sync substrate snapshots from peer Oases and from substrate-primary Maxim instances

### Why "Oasis"?

A bio-coherent metaphor: an oasis is a sustaining gathering place that travelers approach to refresh themselves and that produces life from accumulated water/nutrients. Maxims approach an Oasis to contribute experience and to receive distilled patterns. Multiple oases coexist in a desert ecosystem; no single one is canonical; together they form a network.

### Hardware footprint

- A Mac Mini, an old laptop, or a Raspberry Pi 5 cluster is sufficient to run an Oasis IF substrate-primary mode is the operating mode (no LLM, just bio-systems running)
- During the transitional phase (1.1) when substrate-primary is still maturing, Oases may need a small LLM (e.g., qwen-7B) to process LLM-AUT contributions and produce reasonable behavior. Hardware budget grows accordingly.
- Once substrate-primary works (post-Phase 0/1), Oases can run pure-substrate and Mac-Mini-class hardware suffices.

### Multiple Oases, no central authority

There is no THE Oasis. Each operator runs their own. Oases sync substrate with each other via the Hivemind. A natural evolution emerges:

- **Personal Oasis** — your Mac Mini, hosting substrate from your own Maxims and friends
- **Community Oasis** — a Discord server or research group running an Oasis their members contribute to
- **Public Oasis** — eventual reference instances (e.g., `oasis.maxim-project.org`) that anyone can connect to

The Hivemind doesn't care which kind of Oasis you connect to. It's all peer-to-peer substrate exchange.

## The Hivemind

### Substrate snapshot bundle

The unit of exchange. A versioned, signed archive containing:

```
maxim-substrate.zip
├── manifest.json           # _format_version, schema_version, contributor_id, domain, signature slots
├── nac.json                # causal links + confidence + provenance tags  (1.0)
└── ec.json                 # concept centroids + cluster metadata          (1.0)
```

ATL, reflexes, and Cerebellum payloads are reserved for 1.1 (the migration-registry seam is in place to add them without a format break). Schema-stable, version-aware, scrubbable. Extends the existing `_format_version` contract per [CLAUDE.md](../CLAUDE.md) v1.0 freeze.

### Substrate domains

Bundles are tagged by **substrate domain** (combat, cooking, medical, fantasy, robotics, conversation, ...). Subscribers opt into specific domains. This serves two purposes:

1. **Relevance** — a medical-AI hobbyist doesn't want combat patterns polluting their substrate
2. **Curation** — domain maintainers can curate their domain (flag bad patterns, validate new contributions)

Default domains ship with the project; users can define new ones.

### Conflict resolution

Two Maxims learn opposite things ("X is good" vs "X is bad"). On merge, the Hivemind uses NAc's existing confidence math:

- **Confidence aggregates Bayesian-style** across contributors
- **Multi-source confirmation** weighted higher than single-source
- **Outcome valence** (positive vs negative) preserved as separate distributions, not collapsed
- **Provenance** preserved so a substrate-primary Maxim can selectively trust patterns from specific contributors

When real conflict persists (e.g., 50 Maxims learned "X is good" and 50 learned "X is bad"), the Hivemind preserves both with full distributions. The local Maxim's behavior remains coherent because it weights its own learning + recent context above any single Hivemind pattern.

### Poison resistance

Adversaries trying to corrupt the Hivemind by spreading bad learning are mitigated by:

1. **Multi-source consensus** — patterns require N independent contributors before promotion
2. **Domain curation** — domain maintainers can flag specific contributors as untrusted
3. **Local provenance tracking** — every pattern carries source tags; receivers can blacklist sources
4. **Outcome correlation check** — if Hivemind-derived patterns lead to negative-valence outcomes locally, weight them lower over time (the substrate naturally distrusts patterns that hurt it)
5. **Identity-bearing concept detection** — patterns that map to specific named entities (likely PII or impersonation) are quarantined automatically

This won't be perfect — pure peer-to-peer systems with open contribution always have an arms race with abuse. Mitigations are practical, not theoretical.

## Phasing

| Version | Substrate-primary | Oasis | Hivemind |
|---|---|---|---|
| **1.0** | B5: Phase -1 + Phase 0 harness + shareability infrastructure (snapshot bundle format, NAc/EC merge ops, provenance tagging, identity-bearing concept detection, substrate domains, export/import CLI). ~1,360 LOC. | Format exists. No Oasis software runs yet. | None. |
| **1.1** | Substrate-primary AUT mode lands. Phase 0 validation runs (raw substrate, no Hivemind). Phase 1 starts. | **Oasis software ships** (~800 LOC). Single-Oasis instance hostable on a Mac Mini. CLI: `maxim oasis serve`. LLM-AUT users can opt in to contribute via `maxim contribute --to oasis://...`. Direct Oasis-to-Oasis sync. | Direct Oasis-to-Oasis sync only. No mesh discovery yet. |
| **1.2** | Phase 1 ships. Phase 2 starts. Substrate-primary Maxims pull bootstrap from Hivemind. | Multi-Oasis federation. Curation tools (mark-trusted, mark-untrusted). Domain maintainer roles. | **Full Hivemind protocol** (~600 LOC): peer discovery, substrate-snapshot exchange, conflict-resolution semantics, poison-resistance defenses, optional well-known reference servers. |
| **1.3+** | Phase 3 starts (from-scratch sequence model). | Oasis becomes a substrate-primary instance (no LLM needed). Mac-Mini-class hardware suffices. | Cross-version migration tooling. Curation registry. Domain ecosystem. |

## Confound discipline: raw vs primed substrate

If substrate-primary Maxims bootstrap from patterns distilled from LLM-AUT experience, the "substrate carries cognition" claim weakens unless we're careful. The discipline that keeps the headline experiment honest:

- **Track "raw substrate" vs "primed substrate" as separate experimental conditions.** A Maxim that converged from zero is the strongest thesis demonstration; a Maxim that bootstrapped from collective experience is a different (also valid) claim.
- **The grounded-language plan's Phase -1 and Phase 0 require raw substrate.** That stays — those phases run with `--hivemind disabled` and must produce findings on un-primed substrate to count.
- **Bootstrap is the end-user-convenience path; raw is the research path.** Both ship.
- **Provenance tagging on every NAc link / EC node** — every pattern carries a tag for which Maxim it originated from (or "consensus" if aggregated from many). When a substrate-primary Maxim does something interesting, we can trace which patterns are local-learning vs imported.

The discipline is similar to how biology handles this: human children learn from observing adults (priming), but also learn from their own experience (raw). Both are real cognition; the distinction matters for research, not for everyday function.

## What got dropped from the old Mother Maxim plan

The 2,224-line `mother_maxim_plan.md` (now archived at [docs/plans/archive/mother_maxim_plan.md](plans/archive/mother_maxim_plan.md)) was designed in the LLM-AUT-only world where memories (dialogue + episode traces) were the unit of value. The 2026-05-09 substrate-primary pivot reframed the architecture significantly. What got dropped:

- **Pecking Order Graph hierarchy** → flat peer-to-peer mesh instead. Multiple Oases coexist; no central authority.
- **Database backend (SQL)** → portable file-based substrate snapshots instead. Easier to host, easier to share.
- **Dual-pass deidentification (700 LOC)** → reduced to ~80 LOC identity-bearing concept detection because we never ship raw episodes.
- **Central Mother server** → multiple peer Oases with no canonical authority.
- **REST `/v1/contribute` and `/v1/recall` API** → Hivemind peer-to-peer protocol (1.2).
- **Tenant isolation** → substrate domains + opt-in subscription handle the same separation.

What was preserved:

- **Mother-as-full-agent** → Oasis is a substrate-primary Maxim instance
- **Bio-system identity-bearing concept detection** → ported into B5
- **Substrate-stays-private invariant** → hippocampus-episodes-never-leave
- **Multi-tenant separation** → substrate domains

## Glossary

| Term | Definition |
|---|---|
| **Maxim Hivemind** | The peer-to-peer substrate-sharing layer. Not a service; a protocol + data format. Acceptable shorthand: "Hivemind", "the Hive", "the network". |
| **Maxim Oasis** | A persistent substrate-primary Maxim instance that absorbs contributions and broadcasts distilled substrate. Acceptable shorthand: "Oasis", "your Oasis". |
| **Substrate snapshot bundle** | The portable, versioned, signed archive containing one Maxim's substrate. The unit of exchange. |
| **Substrate domain** | A tag (combat, cooking, medical, ...) that scopes substrate sharing. Subscribers opt into specific domains. |
| **Contribution** | A substrate snapshot sent from a Maxim to an Oasis (or directly to the Hivemind). Always opt-in. |
| **Distillation** | The Oasis's process of aggregating contributions into consensus patterns. Uses NAc Bayesian confidence math + EC centroid merging. |
| **Provenance tag** | A source-instance-ID attached to every NAc link / EC node. Enables trust management and the raw-vs-primed experimental distinction. |
| **Raw substrate** | A Maxim's substrate built only from its own experience. The headline-experiment configuration. |
| **Primed substrate** | A Maxim's substrate bootstrapped from the Hivemind. The end-user-convenience configuration. |
| **Mother Maxim** | (Deprecated.) The old name for what is now the Maxim Oasis. |

## See also

- [Substrate-Primary Mode](substrate_primary.md) — The cognition layer the Hivemind shares
- [docs/plans/maxim_hivemind.md](plans/maxim_hivemind.md) — Full architectural plan
- [docs/plans/grounded_language_acquisition.md](plans/grounded_language_acquisition.md) — The substrate-primary research program the Hivemind enables
- [docs/plans/archive/mother_maxim_plan.md](plans/archive/mother_maxim_plan.md) — Predecessor plan (archived)
