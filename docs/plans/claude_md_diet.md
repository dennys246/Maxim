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
