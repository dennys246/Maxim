# Grounded word binding — the substrate-primary language demo, with an Oasis boost

> **PROPOSED 2026-09-24 — plan only, no code; not a 1.4 rung.** The concrete path for the
> [grounded-language line](grounded_language_acquisition.md) (a PARALLEL line per
> [roadmap_1_4.md](roadmap_1_4.md) §Parallel lines), written so the **language side** and the
> **Oasis side** ([maxim_hivemind.md](maxim_hivemind.md), [public_oasis.md](public_oasis.md)) work to
> one contract. Each experiment below enters through its own prereg and the four-lens design review
> ([DESIGN_REVIEW.md](../experiments/DESIGN_REVIEW.md)) before its harness is built. Nothing here
> powers E1–E3; a stage that touches a survival rung enters as a declared arm or not at all.

## The demo, in one paragraph

A substrate-primary Maxim — no LLM anywhere in its action path — plays the survival world. When
something happens to it, the game says so in text (`[minecraft:system] … drowned`), and a teacher
player can say a word in chat at the moment it applies. The agent binds each heard word to the world
situation it named, from its own experience. Later, **hearing the word alone** — before its sensors
see anything — reactivates that situation and the fear or want already attached to it, and the agent
acts on it. Then a second, fresh agent **pulls those bindings from the Oasis** and responds to the
same word without ever having had the experience. The bindings travel the same shipped bundle path
that carried the taught want (Exp 56) and the drowning fear (Exp 61).

**What the demo may claim, at most:** *a word, bound from experience, reactivates the situation it
named and changes what the agent does; and a word one agent learned works for another.* That is
word-to-world binding (receptive, like a dog learning "walk") — **not** language understanding, not
production, and not learning from the internet. Every public line about the demo stays inside that
sentence, and names the rung reached (Stage 3 or Stage 4 below), never the program's endpoint.

## What exists today (verified 2026-09-24)

| Piece | Status |
|---|---|
| Substrate-primary action selection: `agent_loop.py::propose_via_substrate` → `NAc.recommend_action` over `{modality: EC cluster}` | **Shipped**, production caller (Exp 42/56/60/61/62) |
| World/interoception/audio situation encoding (`SensorEncoder`, 384-d) — cluster ids ARE ATL concept ids | **Shipped** |
| Learned situation fear (Wire-4 `cluster_fear`) and want (`cluster_reward_bias`) keyed on world clusters | **Shipped**, EARNED (Exp 60/56) |
| Game text as a percept: bridge `--system_messages` (#807, default OFF) → `MinecraftPerceptSource` → `[minecraft:system] <text>` | **Shipped** — observed, pooled, **never reaches action selection** |
| Text encoding (`LinguisticEncoder`, 768-d, EC modality `"text"` + ATL activation) | **Capability** — only via `MemoryHub` with `MAXIM_SUBSTRATE_PATH=1`; the substrate-primary loop bypasses it |
| Text clusters in action selection | **Absent** — `_SUBSTRATE_CHANNELS` is interoception/audio/world only |
| Retrospective binding (`memory/percept_trace_buffer.py::PerceptTraceBuffer`) | **Capability** — zero production constructors (the same gap as R4 delayed credit) |
| Cross-modal comparison 384 ↔ 768 | **Undefined** — `similarity/ec.py::_cosine` returns 0.0 across dims; the projection is [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) |
| Substrate → text output | **Absent** — `recommend_action` returns `params: {}`; the bridge has no chat action |
| Bundle export / ingest / merge (`substrate export|ingest`, `hive add|pull`, `oasis serve|publish`) | **Shipped**, CLI callers; bundle = manifest + **NAc + EC only** |
| ATL in bundles | **Reserved** slot (`hivemind/bundle.py` docstring), never built |
| Text-modality EC in bundles | Not excluded, never tested; merges at the generic 0.44 threshold (sensors use 0.85) |
| Pull into a live agent | **Absent by design** — ingest targets an at-rest session/agent directory only |
| Public Oasis | **PROPOSED** ([public_oasis.md](public_oasis.md)): publish-only, promotion still WRITE-ONLY |
| Paired data | Audit 2026-09-20: labels (3 templates, 1 trace) → REDESIGN THE SOURCE. Re-audit 2026-09-21 (EXPLORATORY: staged deaths, n=5, post-hoc window): death messages name the cause; sensors ~1 s earlier separate the causes |

## Front-gate: does this need new mechanisms?

Almost none — it rides on shipped infrastructure, and the one genuine addition is a slot the bundle
format already reserved.

- **The binding** rides on the **ATL** as a typed relationship (a new `RelationshipRegistry` type,
  working name `NAMES`: text concept → world concept, with `weight` and `confidence`). This is also
  the bio-faithful home: the ATL is the semantic hub where modality-specific spokes meet
  (hub-and-spoke); binding a heard word to a sensed situation is exactly the hub's job. Because a
  binding is a relation between two *ids*, it needs **no 384 ↔ 768 comparison** — the dimensional
  mismatch blocks generalisation to unheard wordings (Stage 5), not binding of heard ones.
- **The look-back** that pairs a message with the situation ~1 s before it rides on
  `PerceptTraceBuffer` — which R4's delayed credit needs too. **One mechanism, two consumers**; its
  design goes through R4's design review (the re-audit's next step 2), and whichever line wires it
  first owns it.
- **The cue** rides on the existing proposal: a heard, bound word contributes the bound world
  cluster to what `recommend_action` / `anticipatory_threat_need` see — declared as its own
  `heard` source on the proposal, never disguised as a sensed world cluster.
- **The transfer** rides on bundle → ingest → merge. The addition is the reserved **ATL payload**,
  scoped to a binding slice (below). That is a wire-format change: CC3/`_format_version`, and the
  hivemind mypy gate, apply.

## Stages

Each stage is independently useful and ends in a recorded result; a later stage never starts on an
unrecorded earlier one.

**Stage 0 — the data source (language side; entry, already the line's next step).**
A capture with **natural, unstaged** deaths (an agent or scripted wanderer, not console commands) to
measure the lag between a situation and its message — the distribution fixes the look-back window.
Plus a **teacher channel**: a scripted player that says a word in chat when a situation applies
(e.g. `water`, `dark`, `food`) — game-native (players chat; D1 holds), and it escapes the
three-template ceiling. Re-run `scripts/paired_data_audit.py` unchanged on both. *Exit:* a lag
distribution and a vocabulary count, recorded, window fixed from data.

**Stage 1 — text on the substrate path (language side; engineering, recording-only).**
The substrate-primary loop encodes text percepts through the ONE text encoder (`LinguisticEncoder`
→ EC `"text"` + ATL), not a second path. No influence on selection. *Guard:* a real-loop test in
which a `[minecraft:system]` percept produces a text EC node and ATL concept on the substrate path.

**Stage 2 — binding (language side; recording-only).**
On each text percept, the look-back finds the world situation(s) in the window and strengthens a
`NAMES` relation (co-occurrence-weighted; a word heard across many situations binds weakly to each —
the specificity is the measurement). *Measured, not acted on:* binding accuracy on held-out
pairings (T1: held-out instances of heard words) against **the nearest-template baseline** and a
**shuffled-binding** control. *Exit:* the accuracy table recorded, pass or fail.

**Stage 3 — comprehension: Experiment A (language side; the first demo-able rung).**
A bound word, heard, adds its bound world cluster(s) to the proposal as `heard`. Prediction: the
fear/want keyed on that cluster fires **before** the sensors reach the situation — e.g. warned
`water` at the shore, the agent turns away earlier than an unwarned one. **Run raw: no bundle
ingested** (the line's standing rule — Phase -1/0 run with the Hivemind off; bootstrap is the
convenience path, raw is the research path).
Arms (minimum, fixed in the prereg): **cued** (word heard, binding learned) · **no-cue** · **cue
with a shuffled binding** (the word points at the wrong situation) · **unbound word** (heard, never
paired). The DV is the executed choice at a timepoint where the sensors do not yet discriminate —
otherwise the sensors, not the word, drive it. *Claim if EARNED:* a word bound from experience
changes behaviour. Ledger row in [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md).

**Stage 4 — the Oasis boost: Experiment B (both sides).**
A donor that passed Stage 3 exports its bindings; a fresh receiver pulls them **before boot** and
hears the word without the pairing experience. Arms modelled on Exp 56: **taught** (bindings + both
endpoint EC nodes) · **isolated** (learns by itself for the same budget) · **dangling** (bindings
without the world EC nodes — must fail, as Exp 56's dangling arm did) · **naive** (nothing).
Transport: the real CLI path (`oasis publish` → `hive add` → `hive pull --apply`, **signed** — Exp
61's prereg said "signed" and its harness passed no `--sign`; this one must, and must assert it).
*Claim if EARNED:* a word one agent learned works for another. Still receptive binding only.

**Stage 5 — generalisation: the projection (both plans revive together).**
Words the agent never heard: T2 paraphrases, T3 **web text** (the only place web text enters — it
has no sensor side, so it is a held-out TEST set, never training data), T4 invented words (must
fail). Needs the 384 ↔ 768 projection ([deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md))
and its own prereg with T1–T4 and the nearest-template baseline fixed in advance.

**Not in this plan:** production (the substrate saying anything — the grounded plan's Phases 1–3),
live mid-session pulls, a public contribution path, and any rung of E1–E3.

## The contract between the two sides

**Language side delivers** (Stages 0–3): the capture + teacher channel; text on the substrate path;
`NAMES` relations in the ATL with `weight`, `confidence`, provenance; the `heard` proposal source;
Experiment A's prereg and record.

**Oasis side delivers** (Stage 4):
1. **The ATL binding slice in the bundle** — the reserved ATL payload, scoped to `NAMES` relations
   plus the concepts at both ends. Export pulls the relation's **world** endpoint's EC node (already
   exported) and its **text** endpoint's EC node (768-d, modality `"text"`, geometry-stamped).
   `_format_version` + CC3 forward-compat on every new frozen type; hippocampus episodes still never
   ship.
2. **Ingest checks for it** (extending V1–V10): a relation whose endpoint node is absent is dropped
   with a count (the dangling rule, as for cluster biases); weights clamped; foreign relations
   discounted as foreign fear is (`FOREIGN_FEAR_DISCOUNT` precedent) — the discount value is the
   prereg's to fix, not this plan's.
3. **Merge re-keys both endpoints** through the aligned-EC id map (the D43 lesson: the merge must
   hand back the map, or a merged relation reads out as nothing). **Text-modality merge threshold**
   calibrated rather than inheriting the generic 0.44 — a text node merging with the wrong text node
   silently rewires a word.
4. **Transport** for the demo: a private `oasis serve` + signed `oasis publish` + `hive pull`. The
   public Oasis (public_oasis.md Phase 1) is NOT a prerequisite; if it lands first, the demo may use
   it read-only. Promotion stays WRITE-ONLY; nothing here needs `accept`.

**Shared, owned by neither alone:** the `NAMES` relation shape (defined once, in the ATL layer, and
imported by the bundle code), the look-back mechanism (R4's design review), and the claim sentence
above.

## Guards and disciplines that apply

- **A template is a label.** If Stage 0's vocabulary is still templates, Stage 3 may run but its
  claim is "a label bound from experience", and the record says so.
- **A fix ships with a caller:** the `heard` source is wired into the real loop, and Stage 3's
  guard is a strict red gate on the real composition, not a hand-composed sequence.
- **Prove each guard by deleting the mechanism** (drop the `NAMES` relation → the cue arm must
  collapse to no-cue).
- **Provenance:** harnesses assert the `maxim` their sub-sims import is their own repo; gated data
  commits prereg-first on `main`.
- **Rig:** runs on big-mac-mini only once the box is quiet; the Paper server and the model server
  are the operator's to start.

## Open questions for the owner

1. **The teacher channel's vocabulary** — a handful of situation words the teacher says (and when),
   or only the game's own messages? The first gives a richer vocabulary; the second is purer D1.
2. **Where the demo stops for the 1.4 lead-up:** Stage 3 alone (raw), or Stage 3 + Stage 4 (the
   Oasis boost)? Stage 4 depends on the ATL payload, the largest single piece of new work here.
3. **Who wires `PerceptTraceBuffer` first** — this line (Stage 2) or R4. Whichever it is, its
   design goes through R4's review so the second consumer does not fork it.

## Where this is referenced

[grounded_language_acquisition.md](grounded_language_acquisition.md) (its concrete near path) ·
[maxim_hivemind.md](maxim_hivemind.md) (the ATL payload's first consumer) ·
[public_oasis.md](public_oasis.md) (Stage 4 may use it; does not require it) ·
[roadmap_1_4.md](roadmap_1_4.md) §Parallel lines ·
records: [paired_data_audit_2026-09-20.md](../experiments/paired_data_audit_2026-09-20.md),
[paired_data_audit_reaudit_2026-09-21.md](../experiments/paired_data_audit_reaudit_2026-09-21.md).
