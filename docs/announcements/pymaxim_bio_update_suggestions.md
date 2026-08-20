# pymaxim.bio — 1.0.9 website handoff (live audit 2026-08-19)

> **LIVE CONTENT AUDIT COMPLETE; WEBSITE FIXES STILL REQUIRED BEFORE THE 1.0.9
> PYPI CUT.** All 38 routes in the 2026-08-19 sitemap were fetched and compared
> with the release-candidate ledgers. The browser surface was unavailable, so
> responsive layout, visual accessibility, focus order, and copy-button behavior
> still require a human/browser pass. D24 records the verified content and
> canonical-domain defects below.

Spec for the **maxim-web** repo (Astro/Starlight → pymaxim.bio). This is *what to say
and how to frame it*, not the Astro code. Everything here is discovery-only + honest —
the site's credibility is the product's credibility, so under-claim before over-claim.

## Headline status after the August evidence pass

Do **not** promote Exp 44 to the home headline for 1.0.9. The original result is
exploratory, and the Exp 44b pilot is explicitly not a result. The pilot found
that the transplant control is name-mismatched and that the two reported axes
encode the same entity/affordance pair rather than independent effects. The
confirmatory campaign is not frozen.

The defensible 1.0.9 headline is the repository positioning:

> **A bio-inspired LLM harness that carries experience-grounded memory, causal
> links, drives, and valence across sessions without fine-tuning model weights.**

The substrate augments prompt context in the default LLM-primary path. It does
not generally override the LLM's prior. Substrate-primary discrimination is a
separate, narrowly graduated mechanism result.

Exp 44 may appear only as **exploratory evidence** with the original modest-N,
residual-color caveats plus the later Exp 44b name-mismatch and non-independent-axis
findings. See [Exp 44](../experiments/44_substrate_counterfactual.md) and the
[Exp 44b pilot](../experiments/44b_pilot.md).

## Verified P0 corrections for maxim-web

1. **Retire the Exp 48 v1 apparatus claim.** Remove every `PASS`/`GRADUATE`
   disposition and the `0.875 vs 0.448` numbers from `/research/cradle/`,
   `/research/experiments/`, and evidence cards. Current truth is **PARTIAL,
   apparatus-v2**: mother effect re-earned at `0.649 vs 0.167` (`+0.482`), but
   LEARNED-v2 missed by `0.001`; the completed sweep indicates credit-tipped
   phase-locked attractor selection, not graded orienting skill.
2. **Regenerate the experiment index.** The live site stops at Exp 48 and
   hard-codes `76`. Add Exp 49 as COMPLETE (H1 supported; H2/H3 pass), add Exp
   50 as PRE-REGISTERED (not a result), and derive the displayed count from the
   source collection.
3. **State architecture debt.** `/concepts/architecture/` says Maxim enforces
   strict one-way dependencies. The intended boundary currently has 33 audit
   findings and no CI regression gate. Baseline enforcement is a 1.1 gate.
4. **State conditional safety wiring.** `/reference/tools/` says every
   non-introspection tool call passes the fear circuit. `with_fear_gate` defaults
   false, stable `maxim.run()` does not enable it, and wrapper construction logs
   and continues on failure. Describe behavior only when the wrapper is active.
5. **Correct EC complexity.** `/memory/overview/` claims approximate search is
   `~10ms regardless of memory size`. Indexed signature queries use LSH; the
   substrate hot path performs an exact same-modality centroid scan, `O(Nd)`.
6. **Correct stable API examples.** Custom-tool examples must pass a non-empty
   `goal` to `maxim.run()`. The call remains a blocking service loop until
   interruption/runtime shutdown; goal completion does not stop it and
   `goal=None` installs no terminal reader. D18 still tracks registration lifetime.
7. **Fix the documentation alias.** `docs.pymaxim.bio` currently serves the
   marketing homepage and declares `https://pymaxim.bio/` canonical. Make it a
   path-preserving permanent redirect; `/` should land on
   `https://pymaxim.bio/getting-started/`.

## Suggested page / section changes

1. **Home hero** — use the scoped LLM-harness positioning above. Supporting
   evidence can point to maintained Exp 42 discrimination and real-hardware
   sensorimotor learning; do not elevate Exp 44's exploratory result.

2. **A "Proof" / "Evidence" page** (new or expand existing) — a short, honest gallery of
   the graduated/validated results, each one sentence + a link. Pull from the
   [graduation-candidates](../plans/behavioral_graduation_candidates.md) EARNED/POSITIVE rows:
   - **Real-hardware sensorimotor learning** (Reachy Mini sound-orient policy,
     no LLM in the action path) — Exp 45 series, with the current healthy-hardware
     and sensor-fold caveats from the graduation ledger.
   - **Operant orienting investigation** — Exp 48 is PARTIAL, not a proof card;
     present it as a case study in apparatus correction and falsification.
   - **Substrate-primary safe-vs-harm discrimination** — Exp 42 GRADUATE.
   - **Exploratory substrate-to-LLM influence** — Exp 44 belongs in an
     exploratory/in-flight section with the Exp 44b control findings, not among
     graduated proof cards.
   Each card: the claim, the honest caveat, the link. This page is the credibility spine.

3. **A "Vision → next" note** — the substrate now *learns from lived experience* in
   llm-primary (Phase 1, #437): the agent self-builds its substrate from use, so it stays
   fresh instead of only being pre-loaded. Frame as the direction, not a shipped headline
   claim yet (the behavioral self-build validation is future work).

4. **Local-first framing stays central** (already the position): Console is
   127.0.0.1-only, the tunnel carries the *resource* not the UI, Oasis contribution is a
   *local decision*. The website is discovery, not a service. Don't let the new results
   tempt a "sign up" / hosted framing — that contradicts the whole architecture.

## What NOT to put on the site yet

- Oasis as a live/available feature — it's the *next build*, not shipped. "Coming: peer
  substrate sharing (Oasis)" at most, clearly future.
- Any "the agent has memory of you across sessions that changes its behavior" claim
  without the Goldilocks limits — Exp 37/38 showed the naive version over-claims.
  Exp 44 is exploratory and Exp 44b has unresolved control/interpretation findings.
- Benchmark/leaderboard-style numbers — the results are mechanism demonstrations at
  modest N, not competitive benchmarks. Presenting them as benchmarks invites the wrong
  scrutiny.

## Acceptance checks

- Every sitemap route returns 200 or an intentional permanent redirect.
- No live page contains Exp 48's retired disposition or v1 numbers.
- The experiment count is generated and Exp 49/50 have correct statuses.
- Canonical tags resolve only to `pymaxim.bio`; `docs.pymaxim.bio/<path>` redirects
  to the corresponding canonical path.
- Python snippets compile and CLI commands are checked against 1.0.9 `--help`.
- The home and docs surfaces link to PyPI, GitHub, the defect/limits ledgers, and
  exact experiment sources.
- Complete a visual/mobile/keyboard/accessibility pass in a real browser; this
  could not be attested by the content crawl.

## Cross-repo note

The engine facts these claims rest on live in this repo's `docs/experiments/` +
`docs/plans/behavioral_graduation_candidates.md`. When a claim on the site changes,
update it here first (the experiment docs are the source of truth), then the site — same
direction as the `maxim serve` OpenAPI contract flow.
