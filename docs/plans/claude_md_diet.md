# CLAUDE.md Diet — Live Invariants In, Archaeology Out

**Status:** DRAFT (2026-08-10)
**Motivation:** External critique (2026-08-10), point 6, verified: CLAUDE.md is ~29.4K words
/ ~62K tokens — roughly a third of a 200K context consumed before any work starts, in long
prose paragraphs (worst format for attention). It is a monotonically growing incident ledger.
The document built to make agent sessions effective is now the largest single tax on them.

**Key insight making this cheap:** Principle 5 already did the hard work. Most lessons carry a
`Regression guard:` that ENFORCES the rule mechanically (CI grep, test path, typed
constructor). Where a guard exists, the narrative is redundant with the enforcement — a
session doesn't need the 5,600-character Exp 42b story to obey a rule CI enforces; it needs
the rule, the trigger condition, and the pointer.

## Target

- CLAUDE.md ≤ **10K tokens** (~4.5K words), same section skeleton.
- Zero information deleted from the repo: full prose moves to `docs/lessons/<slug>.md`
  (one file per lesson/invariant, named by its current bold title slug).

## Compression contract (per invariant)

Each Lessons-learned / Architectural-invariants entry compresses to at most ~4 lines:

```
**[engineering] <rule statement — the imperative, kept verbatim where possible>.**
<trigger: when this applies / the one-line failure signature>. Full history:
[docs/lessons/<slug>.md](docs/lessons/<slug>.md). Regression guard: <unchanged>.
```

What stays in the compressed form, always: the tag, the rule as an imperative, the
regression-guard line (lint compatibility verified — `scripts/lint_claude_md_invariants.py`
matches the opener pattern + guard field only, not prose length), and the lesson link.
What moves out: incident narrative, dates, PR numbers, review-round attribution, dead-end
hypotheses, "what does NOT work" digressions (these go in the lesson file's body).

**Exception — keep full text in CLAUDE.md for:** rules with NO mechanical guard (the prose
IS the enforcement; ~the process invariants: review-round discipline, merge-target
verification) and the "Working principles" section (it governs how new entries are written).

## Section-by-section treatment

| Section | Action |
|---|---|
| Project overview, required checks, running-sims rules | Keep (already tight) |
| Lessons learned (~13 entries, some 5K+ chars) | Compress per contract |
| Working principles | Keep, light trim |
| Architectural invariants (~45 bullets) | Compress per contract; guards unchanged |
| doctor section | Move maintenance guide to docs/user/ or doctor module docstring; keep 5-line summary |
| Key commands | Keep |
| Env var table (95 MAXIM_* entries with paragraph-length comments) | Keep table, one line per var; long rationales move to the owning module docstring or lesson file |
| Testing / API / Active initiatives | Keep, trim shipped-history to links |

## Stages

**Stage 1 — mechanical split.** Script-assisted: extract each tagged invariant's full text to
`docs/lessons/<slug>.md`, leave the compressed stub. Run the Principle 5 lint + eyeball diff.
**Stage 2 — hand pass.** Compress non-tagged prose (env table comments, doctor section,
shipped-history). Verify every `Regression guard:` path still resolves.
**Stage 3 — operator review.** This is Denny's operating document — final cut is a user
review, not a merge-on-green. One PR, reviewed as text.

## Risks

- **Losing enforcement weight.** Mitigation: rules are kept verbatim as imperatives; only
  narrative moves. Where past sessions demonstrably needed the narrative (e.g. the
  head-frame actuation lesson's debugging checklist), the compressed stub keeps the
  checklist line and links the story.
- **Link rot.** Stage 2 includes a link-resolution check (extend the Principle 5 lint to
  existence-check cited paths — already tracked follow-up; this plan is its natural vehicle).
- **Memory duplication.** `~/.claude/.../memory/` files that restate lessons now point at
  `docs/lessons/` instead of duplicating (repo is the source of truth per memory rules).

## Non-goals

- No rule changes. This plan changes WHERE prose lives, never WHAT is required.
- No deletion of history (dormancy-over-deletion applies to docs too).

**Regression guard:** `scripts/lint_claude_md_invariants.py` (must stay green through the
split) + a token-count assertion added to the same lint (fail if CLAUDE.md exceeds ~12K
tokens, so the ledger cannot silently regrow).

---

## Appendix — session kickoff brief (added 2026-08-13)

Saved here per `feedback_save_kickoff_prompts_durably`. A dedicated session executes this
plan with one **scope extension** the operator requested: alongside the per-incident
`docs/lessons/<slug>.md` archive, build a **`docs/agents/` satellite layer** — per-subsystem
briefs an agent loads on demand instead of carrying everything in-context. Record this
extension as a deliberate deviation in this plan (do not silently diverge from the Stages
above; amend them).

**Two output layers, different jobs:**
- `docs/lessons/<slug>.md` — per-incident ARCHIVE (full narrative, dates, PR numbers,
  dead-end hypotheses). Write-once, linked from compressed stubs. As specified above.
- `docs/agents/<subsystem>.md` — per-subsystem WORKING BRIEF, synthesized not archived:
  the mental model, key files table, that subsystem's invariants as one-liners with
  `Regression guard:` pointers, live gotchas, links into `docs/lessons/`. Self-contained:
  an agent working in that area reads the slim CLAUDE.md core + exactly one brief.
  Suggested cut (validate against the actual invariant clusters in lens 5): `bio-memory`
  (Hippocampus/EC/ATL/NAc/SCN + substrate encoding), `llm-routing` (lanes, backends,
  proxy, timeouts, mesh/peer), `embodiment` (SEM, drives, pain channels, Reachy safety),
  `simulation-experiments` (sim discipline, apparatus standards, provenance, graduation),
  `persistence-config` (atomic_io, _format_version, config.json layers, role detection),
  `runtime-tools` (agent loop, executor, builders, buses).
- CLAUDE.md core keeps a **routing table**: "touching X → read docs/agents/X.md" — the
  briefs only pay off if discovery is one hop.

**Phase 0 (before any edit) — parallel multi-lens review.** Launch 5 parallel subagents
over the current CLAUDE.md, each returning a structured report:
1. **Enforcement lens** — for every invariant: is the guard real (does the cited
   test/grep/type actually enforce it)? Classify mechanically-guarded (safe to compress
   hard) vs prose-is-the-enforcement (keep full text, per the Exception above).
2. **Condensation lens** — per entry: the ≤4-line compressed form per the contract above,
   flagging duplicated content (several lessons retell the same incident) and candidates
   for merging.
3. **Retrieval lens** — what does a session actually need ALWAYS in context (commands,
   checks, hard safety rules, the routing table) vs on-demand (subsystem briefs) vs
   almost-never (incident archaeology)? This lens drives the three-layer split.
4. **Truth lens** — entries whose claims have drifted from the code (stale line numbers,
   "pending" items that shipped, superseded wording). Fix during the split, cite evidence.
5. **Information-architecture lens** — cluster the invariants/lessons by subsystem and
   propose the `docs/agents/` file cut with a table of contents per brief.
Synthesize the five reports into a design fold recorded in this plan, THEN execute
Stages 1–3 as amended.

**Hard constraints (unchanged from the plan):** zero information deleted from the repo;
rules kept verbatim as imperatives; every `Regression guard:` / `Roy experiment:` line
survives in CLAUDE.md; `scripts/lint_claude_md_invariants.py` green throughout (+ add the
token-count assertion); measure tokens before/after and report both; ≤10K-token target for
CLAUDE.md core; Stage 3 is an operator text-review — open the PR, do NOT merge on green.
Work in a fresh worktree from the start (`git worktree add ../Maxim-wt-claudemd -b
chore/claude-md-diet`), `PYTHONPATH=<worktree>/src` absolute on its own line.

---

## Design fold — Phase 0 synthesis (2026-08-13)

Five parallel lens reviews ran over the full document before any edit; reports live at
[docs/plans/reviews/claude_md_diet_phase0/](reviews/claude_md_diet_phase0/) (1_enforcement,
2_condensation, 3_retrieval, 4_truth, 5_information_architecture). Headline findings:

- **Inventory is 100 entries** (36 Lessons-learned + 64 Architectural-invariant bullets),
  not the ~58 the plan estimated. Sections weigh: invariants 112.4K chars, lessons 71.0K,
  env table 23.2K — 81% of the file.
- **Enforcement: 0 BROKEN guards out of ~150 checks.** 45 MECHANICAL + 26 STRUCTURAL
  (~88%) → hard compression is safe for the bulk, confirming the plan's key insight.
  10 PROSE entries keep (near-)full text per the Exception: L07 module-extraction, L14
  dead-code, L15 env-scrubs, L21 review-round, L22 merge-target, A03 bus convention,
  A04 atomic_write_json (cited grep is detection-only and currently matches ~7 hand-rolled
  `os.replace` sites — flagged, NOT fixed here; no-band-aid rule says that's its own task),
  A12 typed-transports, A14 WorkerPool ownership, A34 SCN intake (missing-is-the-signal).
- **Truth: 21 drifted claims**, all fixed during the split (see 4_truth.md for the full
  table). Biggest: the Active-initiatives section still said "v0.7.0 / Gating 1.0" (repo is
  1.0.6, theme 1.1); the no-silent-swallows CI lint and harness-provenance CI lint SHIPPED
  but were still cited as "tracked follow-up"; `build_executor` call-site list was stale
  (CLI/orchestrator migrated into AgentFactory); the build_pain_bus DefaultNetwork
  paragraph was superseded by the build_default_network entry in the same file. Citation
  style rules adopted from the audit: **cite `file::symbol`, never `file:line`** (every
  drifted reference was a line number; symbols all held), and **drop volatile counts** or
  round them (~).
- **Condensation: ready-to-paste stubs drafted for all 100 entries** (71 compressed, 27
  already at contract length, 2 kept full per the Exception). 14 duplicate retellings
  (D1–D14) and 9 merge groups (M1–M9) identified; adopted below.

### The measured constraint conflict, and its resolution (FLAGGED FOR STAGE-3 REVIEW)

Measurement: the existing `Regression guard:` / `Roy experiment:` lines total **~39K chars
(~9.8K tokens) verbatim** (~8K tokens with decorative parentheticals stripped, all
references kept). The appendix constraint "every guard line survives **in CLAUDE.md**" and
the ≤10K-token core are therefore mathematically incompatible — the guard lines alone
consume the entire budget. Both the retrieval and IA lenses independently recommended the
same resolution, adopted here as a **recorded amendment** (the appendix authorizes exactly
this: "do not silently diverge; amend"):

> Every entry survives as a compressed stub **with its guard line intact** — but
> subsystem-local stubs live in the OWNING `docs/agents/<subsystem>.md` brief, not in the
> CLAUDE.md core. The core keeps: all safety-critical stubs (hardware-damage,
> experiment-retraction, cost-blowout class), all cross-cutting stubs (rules that fire in
> any subsystem), and all PROSE/process entries. `scripts/lint_claude_md_invariants.py` is
> extended to audit the briefs with the same opener→guard-field rule, so Principle-5
> enforcement covers every stub wherever it lives, plus a CLAUDE.md token ceiling and a
> link-existence check.

If the operator prefers the letter of the original constraint (all guards in CLAUDE.md),
the fallback is a compact rule→guard index appendix (~5K tokens) — rejected here because it
duplicates every guard line and still lands the core at ~12K tokens. Decide at Stage 3.

### Adopted design

1. **Three layers.** `CLAUDE.md` core (always in context) / `docs/agents/<subsystem>.md`
   six working briefs (on-demand) / `docs/lessons/<slug>.md` per-incident archive
   (forensics). Plus a frozen full snapshot of the pre-diet CLAUDE.md at
   `docs/lessons/claude-md-2026-08-13-pre-diet.md` so "zero information deleted" is
   trivially auditable in review.
2. **Brief cut** (lens 5, validated against actual clusters): `bio-memory`, `llm-routing`
   (one file, two hard-titled halves: router/backends + topology), `embodiment` (hardware
   -safety section FIRST), `simulation-experiments`, `persistence-config`, `runtime-tools`.
   Hivemind homes in bio-memory (promote to a 7th brief when 1.2 P2P starts). Imagination/
   foundry/cradle home in simulation-experiments. Straddler rule: home = the brief whose
   trigger paths an editor is inside when the rule fires; cross-refs are one line, never a
   restatement. Each brief: mental model (synthesized), key files, invariants (stubs with
   guards), gotchas, env vars owned, lesson links.
3. **Routing table** replaces + absorbs the Quick-reference table (area → key files →
   brief), placed right after the required-checks section.
4. **Merges adopted:** M1 (one-HTTP-call: lesson+invariant → one stub), M2 (probe entry
   point: three entries → one), M3 (role detection: lesson+invariant → one; the lesson's
   stale 5-rank order dropped as superseded), M6 (context-similarity pair → one entry, two
   rules), M7 (three grep-guarded name bans → one "removed identifiers" stub), M4 (six
   canonical builders → one entry with a 6-row builder table + per-builder clauses, all
   guard lines retained), M5/M8 (shared lesson files for drive-pain and timeout-plan
   siblings). Net: ~100 entries → ~85 unique stubs.
5. **Env table:** core keeps the ~14 session-critical vars one line each; the rest move to
   the owning brief one line each; paragraph rationales go to lesson files. The canonical
   truthy-parser note is stated once, not per-var.
6. **Kept-full in core:** Working principles (narrative retellings trimmed to their home
   lessons), L21/L22 process invariants (PR archaeology trimmed to lesson files, rules +
   SCOPE TRIGGER + "different reader" rationale kept), the no-band-aid guardrail, required
   checks, sim-cost discipline (3 safety bullets stay in core; the rest moves to the
   simulation brief).
7. **L22 stays untagged** (adding a tag is a rule change needing operator sign-off; noted
   here). The ~7 `os.replace` violations of A04 are surfaced as a finding, not silently
   fixed or silently ignored.
8. **Stub-integrity cautions honored:** load-bearing conditional instructions survive in
   stubs (A36 tripwire do-not-relax, A28 actuation checklist + vendor-docs-first, A34
   producer field-name checklist, A05 pick-(a)/(b) rule, A20 do-not-remove-B8); `Regression
   guards:`/plural and dual guard+Roy entries copied exactly; hardware-damage entries get
   the least compression.

### Amended stages

- **Stage 0 (done):** Phase 0 five-lens review + this fold.
- **Stage 1:** script-assisted extraction of every compressed entry's full prose to
  `docs/lessons/<slug>.md` (+ the frozen pre-diet snapshot); lesson files for the n_ctx
  three-leg history, the doctor maintenance guide, and shipped-history blocks.
- **Stage 1b (new):** write the six `docs/agents/` briefs (parallel agents, one per brief,
  from the lens-5 per-brief specs + lens-1 classifications + lens-4 corrections).
- **Stage 2:** rewrite CLAUDE.md core per the retrieval-lens skeleton with the condensation
  stubs + truth fixes; extend the lint (brief audit + ≤12K-token ceiling on CLAUDE.md +
  link existence); verify green; measure tokens.
- **Stage 3 (operator):** text review of the PR. Do NOT merge on green. Explicit review
  asks: the guard-placement amendment above; L22 tag; the A04 `os.replace` finding.
