# pymaxim.bio — website handoff (1.0.9 live audit 2026-08-19 · 1.1 audit 2026-08-25)

## 1.1 audit — 2026-08-25 (gates the `1.1.0` final cut; roadmap step 5b / item 16 / D24)

**Scope.** Every route in the live sitemap (39) plus the source of each page in
maxim-web `main` (`ae3dfef`), compared against release truth at `cfe489de` (the local
`1.1.0` cut as first written; rebased to `c579a5c1` on `release/1.1.0-final` with identical
file content, so every check below holds for that commit): `docs/experiments/52_nurture.md`, the graduation ledger, CHANGELOG
`[1.1.0]` + `[1.1.0rc1]`, `docs/bugs/README.md`, `docs/limits/README.md`, and the
**exact 1.1.0 wheel** (`python -I -m build` from `cfe489de` ≡ `c579a5c1`, installed in a clean venv;
every `maxim` flag and subcommand flag used on the site checked against `--help`; all
44 Python snippets compiled; `imagine(persona=)` confirmed to warn). Fixes:
**maxim-web PR #7** (`docs/1-1-website-audit`), `pnpm build` clean (40 pages), 2,137
internal hrefs + fragments resolve in `dist/`. The browser surface was again
unavailable — the human-only list at the end still stands.

**P0 re-verification (2026-08-20 fixes, live today):** Exp 48 v1 numbers gone and
disposition PARTIAL ✓ · index derives its count (82 live) with Exp 49 COMPLETE and 50
PRE-REGISTERED ✓ · architecture page carries the D19 accepted-debt wording with no
hard-coded count ✓ · tools page states fear gating is conditional and `run()` does
not enable it ✓ · EC hot path documented as O(N·d) with the ~10 ms figure withdrawn ✓
· custom-tool examples pass a non-empty `goal` ✓ · `docs.pymaxim.bio/<path>` → **308**
path-preserving to `pymaxim.bio/<path>`, `/` → `/getting-started/`, canonical only
`pymaxim.bio` ✓ (308, not 301 — permanent and method-preserving; acceptable).
**Version truth:** no page says 1.1.0 is out; install snippets are plain
`pip install pymaxim` (resolves to 1.0.9 until the cut, 1.1.0 after — correct both
before and after step 5b; nothing to change). PyPI at audit time: 1.0.9 stable,
1.1.0rc1 pre-release.

### Findings (route → claim on the site → truth source → fix in PR #7)

| Route | Claim on the site (before) | Truth source | Fix |
|---|---|---|---|
| `/` | Evidence cards: Exp 45 / 42 / "where it didn't hold up" (Exp 48 corrected); no Exp 52 | `52_nurture.md`; graduation ledger Earned row; Exp 45 row (`_big` block 2026-08-24) | New first card "Learning to want: orienting taught through hunger relief" (Exp 52, with one-session / n = 12 / sign-only-relief caveats); Exp 45 card: pre-repair magnitude "re-validated on the repaired robot in August 2026 — one session"; Exp 48 sentence now says the fix became Exp 52, link retitled "the apparatus case study" |
| `/getting-started/` | `maxim.imagine(goal=…, persona="adversarial")` | `api.py::imagine` — `persona=` is a 1.1 deprecated alias for `mode=`, dropped in 1.2 (CHANGELOG rc1 upgrade note) | Snippet uses `mode="adversarial"`; Evidence page added to "Where next" |
| `/installation/` | Python 3.10+, extras, `--list-models` / `--llm` / `--auto-download` / `--language-model` / `--mode exploration` | `pyproject.toml`, wheel `--help` | Verified, no change |
| `/reference/tools/` | "`register_tool()` is currently one-shot … open 1.1 decision (D18)"; "under the `adventure_architect` persona"; "`refinement` persona"; `cancel()` "reserved for 1.1+" | D18 FIXED 2026-08-23 (persistent; `unregister_tool`, `clear_registered_tools`, `list_registered_tools`); persona system removed (#482, `tools_dm.py`); `cli_parser.py` `--dm` | Persistent-registration bullet with the three new symbols and the behaviour change; DM/architect flow via `maxim --sim "<goal>" --dm`; "systematic measurement runs"; cancel wording "as of 1.1". Fear-gate section verified correct; `goal` non-empty in both examples ✓ |
| `/concepts/communication/` | outbound `send_message`, "like every side-effecting tool, is reviewed by the fear circuit" | `AgentConfig.with_fear_gate` default False; stable `run()` does not enable it (D24 P0 #4 residual) | "reviewed … *when the gate is active* (CLI on, stable Python API off)" + link to tool safety |
| `/concepts/architecture/` | D19 accepted-debt wording, `maxim --audit-architecture`, burn-down 1.1.x | D19 FIXED 2026-08-24 | Verified, no count hard-coded, no change |
| `/memory/overview/` | EC: LSH index vs O(N·d) centroid scan; ~10 ms withdrawn | D21 | Verified, no change |
| `/research/evidence/` | Graduated: Exp 10 / 42 / 45 / 49; Exp 48 "under investigation … next step randomised order"; Exp 45 "large-step arms remain n=1 per side, multi-rep block queued"; "Architecture layering … 33 open audit findings and no CI gate yet" | Exp 52 EARNED; Exp 48 SUPERSEDED (ledger); Exp 45 row `_big` block n = 8/side 2026-08-24, D30/D31, L9; D19; L8; roadmap item 18 | New graduated card "Caregiver-taught orienting through hunger relief" (both phases, all numbers, all caveats); Exp 48 moved under "Superseded" with the verdict standing for the constant-credit apparatus; Exp 45 caveat rewritten (both halves passed, one session, D30/D31, L9); L8 paragraph under Exp 37; architecture bullet → enforced against a reviewed baseline (no count); new bullet "Loudness / onset salience is not in 1.1" |
| `/research/cradle/` | Exp 48 "Built, embodied, and PARTIAL"; table stops at 48; "sanctioned next step is randomised stimulus order"; `--aut-mode substrate-primary` "slated for v1.1" (×2) | `52_nurture.md`; Exp 48 record ("re-run with `--credit constant`"); `cli-reference.md` (`--aut-mode` shipped, `[experimental]`) | Exp 52 section (mechanism change vs 48, Phase A + Phase B tables, gate v3, weak-seed L1 explanation, scope limits, links); table row 52 EARNED, 48 "superseded by 52"; Design-vs-built split into EARNED (52) / superseded case study (48); status bullets; `--aut-mode substrate-primary` "ships in 1.1 as an experimental opt-in" |
| `/research/experiments/` (+ `src/data/experiments.json`) | 82 entries; Exp 48 "current example" of Partial; 44b pilot only; 37 graduation entry "in-flight" | `docs/experiments/README.md` (+ this PR's rows), 44b §S4 (2026-08-24), L8 | +Exp 52 (recorded), +H2 loudness bench (reference-only, "NOT part of 1.1"); 48 finding notes supersession; 44b finding carries the S4 non-stationarity result (not a result promoting Exp 44); 37 graduation → `partial` with L8; 37 cross-model finding carries L8; count derived → 84 |
| `/research/experiments/cross-session-learning/`, `/research/experiments/substrate-primary-evidence/` | Exp 37 PARTIAL rows without the time-reproducibility limit | L8 (2026-08-22: same commit, same seeds — 0.42 June vs 0.71 August) | L8 added to the Exp 37 row and the PARTIAL bullet on both pages |
| `/guides/simulation/` | "Personas … `Persona` dataclass … `--sim-mode` (preferred) or the deprecated `--persona`"; `/persona` command; "Use `--persona campaign`" | `cli_parser.py` (`--persona`/`--sim-persona` REMOVED in 1.1; `--sim-mode` is a free-form label); `tools.py` `approach` frames | Section rewritten as "Modes (the persona system is gone)": `--sim-mode` is a label, strategy comes from goal text + `approach` frames (adversarial/sweep/cooperative/confused/escalating); `/persona` row and thread diagram entry removed |
| `/guides/reachy-mini/` | extra pins `reachy-mini[gstreamer]>=1.8.3,<2.0`; orient section credited by relief; no loudness claim | `pyproject.toml` `reachy = ["reachy-mini>=1.8.3,<2.0"]` | Pin corrected; rest verified (no loudness / startle / onset-salience claim) |
| `/research/behaviors/audio/` | "not yet buildable", no loudness claim | roadmap item 18 (bench done, design 1.1.1) | Added one status line: nothing in 1.1 reads sound level; loudness / onset salience is not part of the audio path |
| `/research/behaviors/overview/`, `/research/behaviors/vision/` | `StartleResponse` described as startle | roadmap §Bio-fidelity corrections ("Drop 'startle' — this is ORIENTING") | Naming note: the class implements orienting (superior colliculus), not a startle brace; name unchanged in 1.1 |
| `/systems/nucleus-accumbens/` | `--aut-mode substrate-primary` "opt-in and slated for v1.1" | `cli-reference.md` | "ships in 1.1 as an experimental opt-in"; llm-primary stays the default |
| `/reference/cli/`, `/guides/networking/` | `maxim doctor --as/--json/--retry/--last-decision`, `maxim roy diff --json`, `maxim model add --local --chat-format`, `peer`, `tunnel`, `config` | wheel subcommand `--help` | All present in 1.1.0 — verified, no change |
| `docs.pymaxim.bio/*`, `/` | alias + canonical | live crawl | 308 path-preserving, `/` → `/getting-started/`, canonical `pymaxim.bio` — verified |
| all routes | Oasis/Hivemind availability, hosted/sign-up framing, benchmark numbers, "remembers you" without Goldilocks limits | this doc §What NOT to put on the site | None found (Oasis stays "next build, not available"; Hivemind only as merge mechanics; Console local-first) |

**Engine-side changes in the same pass (this repo, `docs/website-1-1-audit`):**
`docs/experiments/README.md` gains the Exp 52 row (complete — EARNED) and the H2
loudness bench row (bench note; not a result), and the Exp 48 row notes its
supersession — the site index is derived from that table, so the engine changed first.

### Acceptance checks — 1.1 (state at PR time; re-run the live half after deploy)

- [x] Every sitemap route returns 200 or an intentional permanent redirect (39 routes, live).
- [x] No live page carries Exp 48's retired disposition or v1 numbers (0.875 / 0.448 only inside the retirement caution).
- [x] Exp 52 present on home, evidence, cradle and the index with its caveats; Exp 48 marked superseded; Exp 44 exploratory only; 44b not promoted; Exp 50 pre-registered.
- [x] No loudness / startle / onset-salience claim for 1.1; "startle" flagged as a naming mislabel.
- [x] Stable-API contract repairs reflected: `register_tool` persistent (+3 symbols), persona removal, `goal` non-empty, fear gate opt-in. (`load.agent()` / `MemoryCorruptionError` are not described on the site — nothing to correct.)
- [x] Experiment count derived (84 after this PR), no hard-coded architecture count.
- [x] Python snippets compile (44/44); CLI commands checked against the **1.1.0 wheel** `--help` (top-level + doctor / roy diff / peer / tunnel / config / model add).
- [x] Canonical tags resolve only to `pymaxim.bio`; `docs.pymaxim.bio/<path>` redirects path-preservingly.
- [x] `pnpm build` clean; 2,137 internal links + fragments resolve in `dist/`.
- [ ] **After merge/deploy:** re-crawl the sitemap (`curl -s https://pymaxim.bio/sitemap-0.xml | grep -o '<loc>[^<]*'`, then fetch each; on a sinkholed network use `curl --resolve pymaxim.bio:443:172.67.172.44`) and confirm "84 experiments", the Exp 52 card, and no `currently one-shot` / `33 open audit` strings.
- [ ] **Human-only:** visual / mobile / keyboard / accessibility pass in a real browser (Starlight sidebar + the new home card at narrow widths; focus order on the copy-install button; contrast of the `reference-only` and `partial` status chips; alt text on the favicon/og image).
- [ ] After `1.1.0` is published: `pip install pymaxim` in a fresh venv resolves 1.1.0 and `maxim --help` matches the wheel checked here (the site needs no text change for that).

---

# pymaxim.bio — 1.0.9 website handoff (live audit 2026-08-19)

> **LIVE CONTENT AUDIT COMPLETE (2026-08-19); the P0 website fixes shipped 2026-08-20
> and 1.0.9 was published 2026-08-23 — the remaining items below gate the 1.1
> cut, not 1.0.9.** All 38 routes in the 2026-08-19 sitemap were fetched and compared
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
   **Update 2026-08-24 (D19 landed):** the 33 findings are now a reviewed
   accepted-debt baseline and CI fails on additions — the page's "CI enforcement
   gated on 1.1" wording should become "enforced against a reviewed accepted-debt
   baseline shipped in the wheel; run `maxim --audit-architecture` for the current
   count; burn-down is 1.1.x". Do NOT hard-code the count on the site (the 76 lesson):
   it changes with the first burn-down commit.
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
