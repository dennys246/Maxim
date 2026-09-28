# pymaxim.bio — website handoff (1.0.9 live audit 2026-08-19 · 1.1 audit 2026-08-25 · 1.1.4 refresh 2026-09-05 · 1.3.0 refresh 2026-09-19 · 1.3.1 refresh 2026-09-27)

## 1.3.1 refresh — 2026-09-27 ("Hardening"; PyPI `upload_time` 2026-09-27T22:06Z, signed tag `v1.3.1` @ `7e695a58`)

**Framing rule.** 1.3.1 makes **no new behavioural claim**. It is fixes plus the guard that holds
each one. The site says so in its first sentence about the release, then lists what was hardened
(security, sharing, the Python API) and sends API users to the upgrade breaks before anything else.
1.3.0's claims (Exp 60, Exp 61, R3) **stand**; 1.3.1 lists three corrections to 1.3.0 (Exp 61
unsigned, the `invalidate` upgrade step, the void Exp 37/38 NAc-bias-off arm), and the site
carries all three. **Exp 62** stays on
the site as ledger evidence only: earned on the ledger 2026-09-20, **not** a 1.3.1 release claim
until its different-reader pass is recorded — the site says exactly that beside it. "Influenced is
not controlled" and "bio-inspired is not a neuroscience simulation" hold throughout: every security
item names its limit (passive → active is ungated, #924; fencing does not stop a model following
injected text, #823; the sandbox's import hook is defense in depth, not the boundary), and the
memory-strength phases are presented as **recording only**, never as a feature. Release links:
[GitHub Release](https://github.com/dennys246/Maxim/releases/tag/v1.3.1) ·
[PyPI 1.3.1](https://pypi.org/project/pymaxim/1.3.1/). Site fix: **maxim-web
`docs/1-3-1-hardening`**; this section is its engine companion and lands first.

**Engine truth read on `origin/main` (`8d6a2c85`), not from memory:** `release_1_3_1.md`; CHANGELOG
`[1.3.1]` including "Correction to 1.3.0" (`[Unreleased]` is empty on main — the "Correction to
1.2.0" is open as #942, and the Exp 56 fact it records holds regardless: the ledger's Exp 56 row on
main already says the harness exports unsigned, and `scripts/exp56/common.py::export_bundle` has no
`--sign`, checked at `9905d4d8` and today); the graduation ledger's "Trigger walk — 2026-09-27" and
the Exp 10 / 37 / 56 / 60 / 61 / 62 rows; `docs/experiments/data/rerun_exp10_2026-09-27/README.md`;
the Exp 38 write-up's 2026-09-27 correction. The v1.3.1 score card
(`docs/limits/score_cards/2026-09-27-claude.md`) was read but is **not linked**: the site links no
score card anywhere, so it gains none now.

**Verified against the artifact, not a proxy.**

- **Clean venv** (`python3 -m venv` on 3.14, `pip install 'pymaxim[console]==1.3.1'`, later the
  `sign` extra for the signing flow). `maxim.__version__` = 1.3.1, imported from the venv.
  `ComponentRegistry().list_refs()` = **97** = the regenerated `components.json` count
  (`--label "pymaxim 1.3.1 (PyPI wheel)"`); the only catalog diff is the label and one description's
  moved plan path.
- **CLI flags:** every `--flag` in a site bash block or inline `maxim …` span — **93** across **19**
  verb groups — checked against the 1.3.1 wheel's `--help` (argparse and the hand-written helps of
  `doctor`, `roy`, `peer`, `tunnel`, `config`). All exist. A second pass checked the other direction —
  **required** flags each command omits — and found a real, pre-1.3.1 defect: **every
  `maxim substrate ingest` example on the site lacked the required `--trust <contributor-id>`**, so
  none of them ran as written (`the following arguments are required: --trust`).
- **The sharing flow, run end to end on the wheel** (two throwaway sessions built with the engine's
  own test helpers, in a scratch `$HOME`): `keygen` → `export --sign` refused without `--license` →
  `export --sign --license CDLA-Permissive-2.0` = schema 3, `release_sequence` 1, `signature.json`;
  unsigned export = schema 2 → `ingest` of the release refused without `--receiver-agent-id`, then
  verified and dry-ran with it (fear discounted ×0.75) → `oasis publish` refused without
  `--queen-key` and refused the unsigned bundle, published the release with it → `oasis serve` on
  loopback + `hive add` (registry wrote `accept_v1: false`) → `hive pull` refused without
  `--receiver-agent-id`, verified and dry-ran with it using the local leader key (loopback only).
- **Python:** **48/48** site snippets compile. The capture and memory snippets **run** on the wheel:
  the published Hippocampus capture raised `TypeError: … missing 1 required keyword-only argument:
  'encoding'`; the rewritten one runs, as do the load, retrieval, strategy, `/reference/cli/`
  inspect, `/guides/simulation/outputs/` load and `maxim.campaign` argument-error snippets.
- **Build:** `pnpm build` clean, **155** pages; **12,217** internal hrefs + fragments resolve in
  `dist/` (URL-decoding checker), 0 broken; all **154** sitemap routes present; the
  `live-site-check` release-line regex matches `1.3.1 "Hardening"` in source and built HTML.

**Engine defects found while running the snippets (not filed; owner's call):**
1. `HippocampusConfig(persistence_path="~/…")` is used verbatim — `save()` writes a literal `./~/`
   directory under the working directory. Same for `maxim.load.nac("~/…")`
   (`FileNotFoundError`). The site now passes absolute paths and says the path is not expanded.
2. `maxim.load.hippocampus("~/…")` returns an **empty store without raising** (a missing
   absolute path does raise `FileNotFoundError`, as its docstring promises), so a later
   `save()` would overwrite. The site tells readers to check `len(...)` before saving.
3. `Hippocampus(config)` with an existing `persistence_path` starts empty and its next `save()`
   overwrites the file (3 memories → 1, measured). #939 names `create.hippocampus` / `create.atl`;
   this is the constructor underneath it and may want to be in #939's "done when". Note also that
   `Hippocampus.from_config(config)` with the path only on the config reopened **0**, while
   `from_config(config, persistence_path=P)` reopened 3.
4. The docs site's old snippet pointed at `~/.maxim/memory/hippocampus.json` — the running agent's
   own store — so, with (2), copying it could clobber real memory. The site example now uses a
   scratch file and warns.

### Findings (route → claim before → truth source → fix)

| Route | Before | Truth source | Fix |
|---|---|---|---|
| `/getting-started/` | 1.3.0 "Oasis-2" version line; Exp 61 "exported through the shipped signed-bundle path"; Exp 56 "exported as a signed bundle"; `diagnose()` unexplained | PyPI `upload_time`; `release_1_3_1.md`; CHANGELOG "Correction to 1.3.0"; Exp 56 ledger row; `common.py::export_bundle`; #942 (open) | 1.3.1 "Hardening", 2026-09-27 UTC, four links; "no new behavioural claim" first, then security / sharing / API in three bullets; 1.3.0 kept as "which 1.3.1 leaves standing"; "through the shipped export and ingest path"; Exp 56 "(unsigned, through the shipped CLI)"; a dated "Corrected in 1.3.1" paragraph (Exp 61 unsigned, `invalidate`, the void ablation arm), with Exp 56 attributed to the ledger and its 1.2.0-notes correction named as pending, not as a 1.3.1 correction; Exp 62 "not a claim of 1.3.1 until a second reader's review is recorded"; `diagnose()` == `maxim doctor`, real network probes on a peer-configured machine |
| `/installation/` | no upgrade guidance; no `sign` extra | `release_1_3_1.md` §Upgrading | New **Upgrading to 1.3.1**: `encoding=` required (with `EncodingSignals.unmeasured("api")` and its import), passive enforced ("maxim active"), `hive pull --receiver-agent-id`, `--api-key` for remote/LAN, `export --sign --license` + one-time `--release-sequence N`, `oasis publish --queen-key` + digest ids, `hive trust --accept-v1`, `--session` lookup, `diagnose()` probes, internet policy / proxy, config downgrade; `sign` row in the extras table |
| `/systems/hippocampus/` | `hippo.capture(perception)` (raises on 1.3.1); `persistence_path="~/.maxim/memory/hippocampus.json"`; `Hippocampus(config, strategy=CompositeStrategy(...))` (no such parameter); staged formation described as live; config listing without the strength keys | wheel signatures; CHANGELOG Phase 2b-i/2c-*/2d-*/2S-*, #817; #939 + the measurements above | Runnable capture with `encoding=` and `save()` on a scratch absolute path; a caution: the constructor does not load and overwrites, #939, reopen with `maxim.load.hippocampus`, `~` not expanded, never point an example at the live store; strategy chosen by `memory_strategy` name (`composite` is a fixed 0.6/0.4 blend; unknown names raise); new **What 1.3.1 records, and does not yet use** — recording only under the default strategy, `strength` an uncalibrated placeholder (~12 s of run experience); staged formation marked "designed, not running" (#817, pool capped at 32); config listing gains `strength_*`, `retro_*`, `dedup_window_s` |
| `/concepts/operating-modes/` | "old names survive only as voice aliases"; passive "edits require approval"; "Maxim singularity" switches to autonomous; `autonomy_level` "requires human approval" via a queue | `modes/definitions.py` (wheel); #821, #826, #827, #828, #922, #924; #800–802, #823, #824, #825 | Both name sets are live and map 1:1; passive's real table (workspace, host-acting tools refused at dispatch, the tool list); "Maxim singularity" refused; more autonomy fails with no approver (#922); new **Enforced since 1.3.1** with every limit stated — passive → active ungated (#924), sandbox import hook is defense in depth and no shipped runtime wires the sandbox tools, fencing cannot stop a model following injected text |
| `/reference/cli/` | `ingest` without `--trust`; `export … [--sign]` with no license; `invalidate` recipe without `--modality`; `oasis publish signed-bundle.zip`; `hive pull` without `--receiver-agent-id` / `--api-key`; voice table "passive / active / singularity" | wheel `--help` + the runs above; CHANGELOG `[1.3.1]` Added/Fixed | Verb list with `--trust`, `--receiver-agent-id`, `--sign --license`, `inspect --entries`, `keygen --key-file`; the runnable `invalidate … --modality world --drop-geometry <stale-tag> --apply` with a dated correction; new bullets: signed export is a v2 release, format 1 frozen, receiver journal refuses equivocation/downgrade; merge keeps the receiver's links (#913), aligned situations fold (#914), allowlist scrub, unsigned export ships only your own learning; keys only where they belong; `--accept-v1`; singularity phrase refused; inspect snippet gains its missing `Path` import and a None guard |
| `/guides/oasis/` | "exports that as a signed bundle"; `export --sign` without `--license`; `ingest` without `--trust`; `oasis publish signed-bundle.zip`; `hive pull` without `--receiver-agent-id` | same | "a bundle, signed or not"; **Signing a release** (v2 contents, `--license`, per-key counter, `--release-sequence N` once, `--key-file`, format freeze with a link to `public_format_freeze.md`); runnable ingest; receiver journal, #913 (607 → 443), #914, allowlist scrub; publish/pull with `--queen-key`, `--receiver-agent-id`, `--api-key`, digest ids; `--accept-v1` default; the flow's wheel run stated; Exp 56 "with an unsigned bundle" |
| `/reference/tools/` | `ModeSwitchTool` "(passive/active/singularity)"; sandbox tool "Run Python in an isolated sandbox"; `maxim.run()` silent on modes | #821, #800–802, #826 | Singularity refused; sandbox row: runs only approved content, fails closed, no shipped runtime wires it; `run()` starts passive and host-acting tools are refused, registered tools keep working |
| `/concepts/prompt-system/` | foundational preamble "a hardcoded paraphrase … a pip-installed agent gets an empty preamble" (D32) | CHANGELOG Fixed "pip installs get the foundational preamble"; wheel `_data/CONSTITUTION.md` + `_load_foundational_context()` output | Read from the Constitution, which ships in the wheel; §1 hard constraints word for word incl. the actuator-speed one; header `=== AGENT BEHAVIOR RULES ===` |
| `/guides/dm-campaigns/` | "`npc_model=` and `prompt_handler=` … raise `NotImplementedError`" | CHANGELOG Fixed; wheel run | `prompt_handler=` works since 1.3.1; `interactive=True` with it raises `ValueError`; `npc_model=` still raises |
| `/guides/simulation/outputs/` | `maxim.load.nac("~/.maxim/…")` | wheel run (`FileNotFoundError`) | `Path.home()` path; "`load.*` take the path as written" |
| `/research/evidence/` | Exp 10 "earned"; Exp 56 "signed bundle"; Exp 61 "shipped signed-bundle path"; Exp 62 unqualified; no #889 entry | ledger Exp 10 row + re-run README; Exp 56/61 rows; Exp 62 row's trigger-walk note; #889 | Exp 10 **MAINTAINED (narrow), 2026-09-27** with the five-part scope (exact reload of 100; new fields round-trip on all 100; 3 per observed resume turn; one turn per phase because every run stopped on D13, #935; link accumulation not re-shown) plus "3 is the enrichment cap" and "too thin for negative transfer"; Exp 56/61 unsigned; Exp 62 "not a claim of the 1.3.1 release"; corrections list gains #889 (Exp 37 then 38), unsigned bundles, the Exp 10 re-run |
| `/research/experiments/cross-session-learning/`, `/roy-harness/` | Exp 37 ablations listed with NAc-bias as valid; Exp 38 "0.40, 0.34 and 0.21" attributed to three wires; Exp 10 "re-validated" only | Exp 37 row; Exp 38 write-up's 2026-09-27 correction; Exp 10 re-run README | NAc-bias-off void (#889) — only Wire-A and Wire-1 attribute; Exp 38's 0.21 arm says nothing about NAc reward bias; Exp 10 September re-run paragraph and bound |
| `/research/experiments/world-seam/`, `/memory/engrams/`, landing | "exported as a signed bundle" / "shipped signed-bundle path" | Exp 56/61 rows; `common.py` | "a bundle", with dated unsigned corrections on both walkthroughs; engrams "signed or not" |
| `src/data/experiments.json` | Exp 10 "clean re-validation" only; Exp 56 / 61 "signed"; Exp 37 cross-model silent on #889 | as above | Exp 10 summary + finding carry the narrow re-run; Exp 56 / 61 unsigned with the date; Exp 37 cross-model names the void arm. 99 rows, none added |
| `src/data/components.json` | "pymaxim 1.3.0 (PyPI wheel)" | the 1.3.1 wheel registry | "pymaxim 1.3.1 (PyPI wheel)", 97, derived |

**Left as is, deliberately.** Exp 60, Exp 61 and R3's numbers and scopes, byte for byte; the
landing's 1.3 card (only its Exp 56 "signed" word moved); every "signed bundle" sentence that is
about the signing feature itself. No score card is linked (the site links none). Exp 62's page and
numbers are unchanged; only its release status is stated.

**Not carried to the site.** The release-integrity items (green-nightly gate, the network-free test
process, the prereg lint over 1.3's own experiments) — true and guarded, but engineering process
rather than something a user can act on; the site's evidence pages already point at the ledger.
The removed internals (`Executor.get_last_rpe`, `SupervisionPolicy.allowed_mode_transitions`) and the
validation refusals, which the site links to via the release notes' Upgrading section.

**Different-reader pass (before either PR).** An independent agent re-checked every changed claim
in both diffs against the sources above and the wheel, re-running the cheap measurements. Fifteen
findings, all folded: the Exp 56 correction was mis-attributed to 1.3.1 (it is #942's); the
`diagnose()` "~12 s" is for an **unreachable** leader and a stalling one takes longer; "three memories
… leave one" misread the clobber (all three are lost; the one is a new capture); an unknown
`memory_strategy` raises at the first `sleep()` in Python, not at construction; `load.hippocampus`'s
silent empty result on a `~` path (now defect 2 above); the preamble carries the **four** §1
constraints from the Constitution's Runtime Preamble section (plus one line that is not in §1);
"no shipped runtime attaches" an approver (#827) rather than "none can be attached"; the receiver
journal's limits (per receiver session, verified admissions only, `--require-signed` only); the Exp 10
commit is operator-attested; the byte cap is #825; "only Wire-A and Wire-1 attribute" read as if both
did (Wire-1 did not) — now "the only valid ablations"; an unreadable internet policy fails closed;
the strength model removes a memory "that is not recalled"; and this section's own framing line.

### Acceptance checks — 1.3.1

- [x] Version line 1.3.1 "Hardening", 2026-09-27 (UTC upload time), links to the release notes, CHANGELOG, GitHub Release and PyPI 1.3.1.
- [x] "No new behavioural claim" stated first; 1.3.0's claims stand with the three corrections; Exp 62 labelled ledger evidence, not a 1.3.1 claim.
- [x] No sentence says Exp 56 or Exp 61 used a signed bundle; every remaining "signed bundle" is about the signing feature.
- [x] Every security item carries its limit (#924, #823, defense-in-depth sandbox); memory strength appears only as recording.
- [x] Exp 10 reads MAINTAINED (narrow), 2026-09-27, with the one-turn / D13 / #935 / no-link-accumulation scope wherever its status appears.
- [x] Exp 37 / 38's NAc-bias-off arm is void on every page that lists the ablations; attribution only to Wire-A and Wire-1.
- [x] Every upgrade break is reflected in the snippets and guides that touch it (`encoding=`, passive, `--receiver-agent-id`, `--api-key`, `--license`, `--release-sequence`, `--queen-key`, `--accept-v1`, `--session`, `diagnose()`); #939 warned where the Hippocampus API is documented.
- [x] A different reader checked every changed claim against the engine sources before the PRs; its 15 findings are folded.
- [x] 93 CLI flags / 19 verb groups checked against the 1.3.1 wheel; the required-flag pass's `--trust` defect fixed; the sharing flow run end to end on the wheel.
- [x] 48/48 Python snippets compile; the capture, load, retrieval, strategy and inspect snippets run on the wheel.
- [x] Counts derived: components 97 from the wheel registry; experiments 99 from `experiments.json`.
- [x] `pnpm build` clean (155 pages); 12,217 internal links + fragments resolve; 154/154 sitemap routes present.
- [ ] **After merge/deploy:** re-crawl the live sitemap, confirm the version line via `live-site-check`, and spot-check `/installation/#upgrading-to-131`, `/concepts/operating-modes/#enforced-since-131` and `/research/evidence/#cross-session-memory-persistence`.
- [ ] **Human-only (carried since 1.0.9):** visual / mobile / keyboard / accessibility pass in a real browser — the longer installation page and the two new cautions on `/systems/hippocampus/`.


## Engram pages — 2026-09-25 (not a release refresh; engine truth = `docs/wiring/engram-formation.md`)

**Framing rule.** An engram is scored on four counts — forms, stays specific, is recalled from a
partial cue, changes behaviour — and the site states each family's score, not its design. The one
family that is an engram on all four counts **without the LLM** is the situation engram (an EC
sensor cluster carrying NAc fear or want; Exp 45/52/53b/56/60/61/62 rung A EARNED). Episodic and
semantic traces change behaviour only as LLM prompt text. Motor engrams are designed and have **no
production caller**. Bio-inspired, never "Maxim has engrams like a brain". Fix plan and issues:
`docs/plans/engram_formation.md` (#908–#911, #899, #848).

### Findings (route → claim before → truth source → fix)

- `/systems/cerebellum/` §Motor programs and engrams → program executor with pain gates + 10 % gate
  tightening; engrams "decay after roughly two days"; "context modulates motor execution" → #909
  (no `ProgramExecutor`/`execute_program` in `src/`; no engram decay code; `form_engram` /
  `query_engrams` / `cerebellum_modulator_factory` have no production caller) → rewrite as "designed,
  not wired": the forward model trains live on real readings; nothing reads its predictions; state
  not persisted until #908 lands. Say "Dormant", link the plan's E7.
- `/systems/cerebellum/` persistence / "live Cerebellum backing every SEM tool" (if present) → #908,
  #909 → remove.
- `/embodiment/sem-protocol/` "engram similarity", "the engram system learn[s]" → #909 → name what
  actually learns (NAc, forward model) and drop the engram clause.
- `/memory/overview/`, `/systems/hippocampus/` "engram co-allocation" → an analogy, keep it as one,
  but link the new page so the analogy is not read as a claim.
- **New page** `/memory/engrams/` → the five families with the four-count table, the situation
  engram's EARNED evidence and its two limits (cosine separates only neutral→extreme swings; the
  `time_of_day` wrap, #899 — the drowning fear misses in the last ~5 % of the day), and the honest
  "prompt text only" for episodic/semantic.

### Acceptance checks — engram pages

No "engram" sentence on the site describes a mechanism without its caller status; every EARNED
number quoted matches its ledger row; `/systems/cerebellum/` contains no program-executor or
two-day-decay claim; `pnpm build` clean; links resolve.

## 1.3.0 refresh — 2026-09-19 ("Oasis-2"; PyPI `upload_time` 2026-09-19T20:24Z, tag `v1.3.0` @ `ad541dd7`)

**Framing rule, applied.** 1.3 moves the LEARNING SIGNAL from a teacher to the world — the game's
own pain — not "the reward" (the reward half, eat-when-hungry, came out prior-driven in R2).
Minecraft is the instrument, never a demo: no page says "Maxim plays Minecraft" or implies general
competence. The honest shape, used verbatim across the site: a substrate-primary agent, no LLM in
the action path, learns a fear keyed to a situation from game-native pain, and that fear transfers
to an agent that never felt it. Every claim carries its scope (one contingency, one pool, one world
layout, substrate-primary) and the four named non-claims (extinction, scaling, another pool,
hive-side promotion). R3 is presented as an instrument and a frozen baseline with **nothing
graduated**, and its "survival is a ceiling — the drive buys cost, not life" reading is stated
wherever its numbers appear. Site fix: **maxim-web PR #25** (`docs/1-3-0-oasis-2`); this section is
its engine companion.

**Correcting the record on 1.2.1.** The site DID cover 1.2.1 (maxim-web #24, merged 2026-09-10) and
inherited its "completes the spoken-code device-pairing loop end to end" headline. That sentence is
gone: `/reference/cli/` now states the correction — the pieces ship (contract 0.5.0 and its pairing
screen, `make_pairing_announcer`, the device speak sink, the audio fix), the composition does not;
**no shipped command constructs the announcer**, so a plain `maxim serve` runs with pairing disabled
and pairing needs an embedder owning both the console and a live robot handle. Verified live on the
wheel, not inferred: `/api/hello` → `{"contract_version":"0.5.0","auth":"bearer","pairing":"none"}`
and `POST /api/pair/request` → **409**. D87's two hardware-owed audio properties are named beside it.

**Verified against the artifact, not a proxy.**

- **Clean venv** (`python3 -m venv` on 3.14, `pip install pymaxim==1.3.0`, then the `console`
  extra). `ComponentRegistry().list_refs()` = **97** = the regenerated `components.json` count
  (`--label "pymaxim 1.3.0 (PyPI wheel)"`); no components added or removed since 1.2.0 — the 1.3
  body work MODIFIED `minecraft_player` (gains the `is_in_water` sensor and the `flee` /
  `escape_water` affordances), which the catalog diff shows and the count does not. **No**
  Mineflayer/bridge JS anywhere in the wheel, so the site keeps saying the live seam needs a
  checkout — now naming Paper **1.20.4**.
- **CLI flags:** every `--flag` in a site bash block (63 across 15 verb groups) checked against the
  1.3.0 wheel's `--help`. **One real defect found and fixed:** the site documented
  `maxim substrate keygen --identity <id>` in two places (`/reference/cli/`, `/guides/oasis/`); the
  flag is **`--signer-id`**. It came from the 1.2.0 refresh misreading the help text's metavar
  (`signer_identity to bind the key to`) as the flag name — a command nobody had run. **48** Python
  snippets compile.
- **The score card's own finding, honoured:** `maxim substrate invalidate --drop-geometry` does not
  run as written in these notes — `--help` on the wheel shows `--drop-geometry` takes the stale tag
  as a VALUE and needs `--session`, plus `--apply` to write anything. The site publishes the
  two-step form instead (census first, then the tag it prints).
- **Engine truth read at the tag**, not from a summary: `release_1_3_0.md`, CHANGELOG `[1.3.0]`,
  `exp60_*` / `exp61_*` / `r3_survival_benchmark_prereg.md` §Outcome, the Exp 56 prereg's
  §Re-baseline runs, the graduation ledger's Tier-1 rows, and
  `docs/limits/score_cards/2026-09-19-claude.md`.
- **Build:** `pnpm build` clean, 54 pages; **3,416** internal hrefs + fragments resolve in `dist/`
  (URL-decoding checker); all 53 sitemap routes present.

### Findings (route → claim before → truth source → fix)

| Route | Before | Truth source | Fix |
|---|---|---|---|
| `/getting-started/` | 1.2.1 "Spoken-code loop"; "It completes the spoken-code device-pairing loop end to end" | PyPI `upload_time`; `release_1_3_0.md`; the 1.3.0 §Correction | 1.3.0 "Oasis-2" (2026-09-19 UTC); the learning signal moved from teacher to world; Exp 60 / Exp 61 / R3 in three bullets with R3's "not life"; the four non-claims and the BLOCKED dark=danger stated; the 1.2.1 overclaim removed, not re-worded |
| `/research/evidence/` | 1.2 claims only; seam bullet said claims "landed in 1.2" | `exp60_*`, `exp61_*`, `r3_*` §Outcome | Three new entries: Exp 60 (with the positive-link caveat and the fear-only first placement), Exp 61 (with the three control arms and the loop-OFF gate wording), R3 as an instrument with its two post-data amendments disclosed; seam bullet now names 1.2 AND 1.3 and the 1.20.4 move |
| `/research/evidence/` § where it didn't hold up | no Exp 58 | Exp 58 prereg; Exp 60 §pivot | "Dark means danger" recorded as BLOCKED at the instrument — the sensors cannot separate the situations, so the cue was swapped to water |
| `/research/limits/` | L11 table with A4 0.0566 / 0.9984 as "the re-measure" | `l11_sensor_dilution.md` 2026-09-15 row | The 09-15 re-read of the SAME trace added beside it (A4 0.0881 / 1.0 / 0.6852, same verdict); the 09-04 numbers stand as measured under the old `saturation` range, and the delta is that one sensor |
| `/reference/cli/` | `keygen --identity`; hello without `pairing`; pairing "available" | wheel `--help`; live `maxim serve` probe; 1.3.0 §Correction | `--signer-id`; `"pairing":"none"` in the hello body; the pairing bullet rewritten to the correction (409, no shipped command wires it, embedder needed) with D87 kept; `cluster_fear` transport + the 0.75 discount; the two-step `invalidate` recipe |
| `/guides/oasis/` | `keygen --identity`; bundles carry wants only | CHANGELOG § Wire-4 fear travels in bundles | `--signer-id`; a bundle can carry a learned situation FEAR, clamped + allowlisted on export, bounded and discounted ×0.75 on ingest, dropped if its world node did not survive; pair 1.3 exporters with 1.3 receivers |
| `/guides/simulation/` | seam "is apparatus for 1.2" | 1.2/1.3 outcomes | "carried the 1.2 and 1.3 behavioural claims" |
| landing | 1.2 headline first | `exp60_*`, `exp61_*`, `r3_*` | New first card: the fear learned from the world and transferred, with the control twin, the 12/12-vs-zero shape, and R3's cost-not-life reading |
| `src/data/experiments.json` | 89 rows | the three §Outcome sections | Exp 60 + Exp 61 `recorded`, R3 `reference-only` (an instrument, nothing graduated); Exp 56 row annotated with RB-1 and its disclosed duplicate row |

**Left as is, deliberately.** The 1.2 claims (Exp 56 earned, Exp 57 partial with its total-experience
cost) are untouched — 1.3 did not revisit them, and the Exp 56 re-baseline is a same-seed
reproduction, so the site says exactly that and does not upgrade it to a replication. The nine
simulation pages keep their "at 1.1.3" verification stamps. The components count stays 97 because
nothing was added — the body diff is visible in the catalog instead.

**Not carried to the site.** 1.4's name (decided at release from the highest earned result — the
site presents "generalization and multi-step credit" as a direction, never a promise) and the
perception fabric / microduck, which wait for a second robot body.

### Acceptance checks — 1.3.0

- [x] Version line 1.3.0 "Oasis-2", date 2026-09-19 (UTC upload time), all four release links resolve.
- [x] No page says or implies Maxim plays Minecraft; every mention calls it the instrument.
- [x] Exp 60 and Exp 61 each carry their scope and their named non-claims; Exp 61's "first loop-live submersion" wording preserved (the lifecycle's loop-OFF gate submerges once before the measured run).
- [x] R3 presented as an instrument with nothing graduated; "survival is a ceiling / the drive buys cost, not life" wherever its numbers appear; both amendments disclosed.
- [x] The 1.2.1 overclaim is gone and the correction is stated where pairing is documented, verified live (`pairing":"none"`, 409).
- [x] The 1.2 claims are byte-unchanged; the Exp 56 re-baseline is described as a same-seed reproduction, not a replication.
- [x] Every CLI flag re-checked against the 1.3.0 wheel `--help` (one defect found and fixed: `keygen --signer-id`); 48/48 Python snippets compile; the `invalidate` recipe is the runnable two-step form.
- [x] Counts derived: components 97 from the wheel registry, experiments 92 from `experiments.json`.
- [x] `pnpm build` clean (54 pages); 3,416 internal links + fragments resolve; 53/53 sitemap routes present.
- [ ] **After merge/deploy:** re-crawl the live sitemap, confirm the version line via `live-site-check`, and confirm `/research/evidence/` serves the three new entries and `/reference/cli/` the pairing correction.
- [ ] **Human-only (carried since 1.0.9):** visual / mobile / keyboard / accessibility pass in a real browser — the new R3 table at narrow widths and the longer evidence page's heading rhythm.

---

## 1.1.4 refresh — 2026-09-05 ("The world seam"; PyPI `upload_time` 2026-09-05T02:47Z, tag `v1.1.4` @ `db4410de`)

**Framing rule, applied.** 1.1.4 is infrastructure only — no behavioural claim; Minecraft is
the instrument, not a demo. Every page that names the seam calls it apparatus for 1.2. The
L11 verdict is stated with both halves everywhere it appears: `mitigation-confirmed`, NOT
`retired-eligible` — A0 fully blind at N=16 (separation 0.0), A4 0.0566 vs the 0.70 bar; L11
stays ACTIVE with A4 as partial mitigation, world-only (interoception and audio byte-identical
to 1.1.3, so Exp 53b was not re-staled — the site previously said shipping *would* re-stale it).
Site fix: **maxim-web PR #20** (`docs/1-1-4-refresh`); this section is pymaxim PR #621.

**Verified against the artifact, not a proxy.**

- **Clean venv** (`python3 -m venv` on 3.14, `pip install pymaxim==1.1.4`; then the `console`
  extra). Wheel contains `_data/components/bodies/minecraft_player.yaml` — **16** `modality: world`
  sensors, counted by parsing the shipped YAML — `items/minecraft_bread.yaml`,
  `simulation/minecraft.py`, `simulation/minecraft_harness.py`, `embodiment/backends/minecraft.py`;
  **no** Mineflayer/bridge JS anywhere in the wheel (`find`), so the site says the live seam needs a
  checkout. `ComponentRegistry().list_refs()` = **94** = the regenerated `components.json` count
  (`pnpm build:components -- --source <wheel>/_data/components --label "pymaxim 1.1.4 (PyPI wheel)"`;
  +2 vs the 1.1.2 catalog: `bodies/minecraft_player`, `items/minecraft_bread`).
- **`maxim serve`**, run under an isolated `HOME`: banner prints `http://127.0.0.1:<port>/#token=…`;
  `GET /api/hello` → 200 `{"contract_version":"0.4.0","auth":"bearer"}`; `/api/identity`, `/docs`,
  `/openapi.json`, `/api/diagnose` → **401** without a token and 200 with `Authorization: Bearer`;
  `?token=` in the query → 401; `/api/pair/status` → **404** (A9.1 is post-`db4410de`, correctly
  undocumented); startup `WARNING Console UI contract mismatch … '0.3.0' … '0.4.0'` (the vendored
  `maxim-ui.json` says 0.3.0). `--show-token` prints the bare token. Without the extra the command
  exits with `OptionalDependencyError … pip install 'pymaxim[console]'`. Flags from `maxim serve --help`.
- **CLI flags:** every `--flag` in a site bash block (129 uses) checked against the wheel's
  `--help` for top-level, `doctor`, `peer`, `tunnel`, `config`, `model add`, `substrate`, `roy diff`,
  `serve` — all present (the two `--host` hits belong to `maxim-diagnostics`, a separate entry point
  in the wheel). **48** Python snippets compile.
- **Paths:** `maxim.utils.paths.resolve_user_state("sim_sandbox")` → `<data_home>/sim_sandbox`
  (D69); research reports are still `Path("data")/"sim_reports"/research_<id>` at `v1.1.4`
  (`research_orchestrator.py:119`, unchanged from `v1.1.3`) — the site says both.
- **Engine truth read from the tag**, not main: `release_1_1_4.md`, CHANGELOG `[1.1.4]`, limits
  ledger §L11 (disposition MITIGATED) + the tracking doc's 2026-09-04 row, graduation ledger
  (Exp 42 / 48 / 53b triggers fired and discharged without re-stale), bugs ledger D51 scope /
  D67 / D68 / D76–D79. `docs/experiments/README.md` is unchanged `v1.1.3`→`v1.1.4`, so the
  experiment index keeps its derived 85 rows. The 0.89 ms / ~240-node scan figures come from
  `docs/experiments/data/ec_scan_cost_2026-09-03.json` (the 0.31 ms spot check in the plan is
  marked informal there and is not on the site).
- **Build:** `pnpm build` clean, 53 pages; **4,927** internal hrefs + fragments resolve in `dist/`
  (checker URL-decodes; the crèche fragment resolves); canonical tags only `pymaxim.bio`; all 52
  sitemap routes present in `dist/`. Live, pre-merge: `docs.pymaxim.bio/<path>` → **308**
  path-preserving, `/` → `/getting-started/`; GitHub `releases/latest` → `v1.1.4`.

### Findings (route → claim before → truth source → fix)

| Route | Before | Truth source | Fix |
|---|---|---|---|
| `/getting-started/` | 1.1.3 "Reachability"; release-notes link to the announcements folder | PyPI `upload_time`; `release_1_1_4.md` | 1.1.4 "The world seam" (published 2026-09-05 UTC), links to `release_1_1_4.md`, CHANGELOG, GitHub release; four sentences on what 1.1.4 is — a seam, a world-only encoding change, console auth, **no behavioural claim** |
| `/research/limits/` | "Mitigation — selected, not shipped … not yet re-measured on a real body"; "Shipping the mitigation re-stales Exp 53b" | limits ledger §L11, tracking doc 2026-09-04 row, graduation ledger | Mitigation shipped **world-only** (N=6 stability 0.97→0.62 is why), scan-cost measured first; pre-registered re-measure table (A0 0.0 / A4 0.0566, clusters 1 / 3, stability 1.0 vacuous / 0.9984); verdict both halves; range re-centring finding (0.926 → 0.747); 53b NOT re-staled |
| `/research/evidence/` | no mention of the seam; loudness "1.1.1, 1.1.2 and 1.1.3" | release note; CHANGELOG (no loudness in 1.1.4) | "The Minecraft world seam (1.1.4) is apparatus, not a result" bullet — nothing on the page changed because of it, bridge JS is repo-only; loudness spans 1.1.1–1.1.4 |
| `/memory/overview/`, `/systems/entorhinal-cortex/` | scan "in pure Python"; "no published latency figure at all" | CHANGELOG PR 0/PR 1; `ec_scan_cost_2026-09-03.json` | vectorized since 1.1.4, still exact, decision-equivalent by test, not an index — LSH correction stands; committed cost figures with conditions |
| `/embodiment/sem-protocol/` | sensor table without `modality` | wheel `embodiment/spec.py` (declarable tags `world`/`audio`; interoception refused; sub-sensor refused) | `modality` row + paragraph; range re-centring note |
| `/reference/cli/`, `/installation/` | no `maxim serve` anywhere on the site (so no curl to fix — checked) | `maxim serve --help`; live probe above | New "Console server" section: bearer auth always on, fail-closed, `#token=` URL, `--show-token` / `--rotate-token` / `--dump-openapi`, `/api/hello`, sandbox-mode exception, **UI bundle 0.3.0 vs server 0.4.0 warning** (the bundled UI is not promised as current); `console` extra row |
| `/reference/components/`, `/embodiment/component-library/` | pinned to v1.1.2, 92 components | wheel registry | regenerated from the 1.1.4 wheel, 94 (derived), links pinned to `v1.1.4` |
| `/guides/simulation/{outputs,sandboxing,index}/` | trace "under `data/sim_sandbox/` relative to the working directory" | D69 / `resolve_user_state` | trace under the data home since 1.1.4; research reports still CWD-relative; section stamp notes what 1.1.4 changed and that the seam is not covered |
| `/research/behaviors/audio/` | "none of 1.1.1, 1.1.2 or 1.1.3 shipped it" | CHANGELOG | "none of 1.1.1 through 1.1.4" |
| `docs/plans/sandbox.md` (repo doc, not rendered) | "the engine's own posture (`maxim serve` binds 127.0.0.1, no auth) is kept" | `maxim serve --help` sandbox paragraph | dated note: bearer auth in every mode except sandbox; decisions 2 and 3 stand |

**Left as is, deliberately.** The nine simulation pages keep their "at 1.1.3" verification stamps
(each claim carries the version it was checked at; re-stamping without re-reading would be a proxy).
Landing hero, experiment index and evidence cards: unchanged — no result moved. The Exp 53b card's
"re-validated 2026-09-02" wording stands (1.1.4 discharged its triggers without re-stale).

**Not documented (post-`db4410de`, `[Unreleased]`):** A9.1 spoken-code pairing, `/api/pair/*`,
console contract 0.5.0.

### Acceptance checks — 1.1.4

- [x] Version line 1.1.4 "The world seam", date 2026-09-05 (UTC upload time), release links resolve.
- [x] No page presents the world seam as a capability or a demo; every mention names it apparatus for 1.2.
- [x] L11 stated with both halves wherever dilution / the encoding appears (limits, evidence, SEM, getting-started); nowhere says the limit is fixed; A4 described as world-only.
- [x] EC scan described as exact and vectorized; no approximation language re-introduced.
- [x] `maxim serve` documented with the token (no pre-existing quickstart or curl lacked it — there were none); bundled-UI skew stated, no UI screenshot or promise.
- [x] Nothing post-`db4410de` documented (`/api/pair/*` → 404 on the wheel).
- [x] Counts derived: components 94 from the wheel registry, experiments 85 from `experiments.json`, world sensors parsed from the shipped body.
- [x] Install commands + every CLI flag re-checked against the 1.1.4 wheel `--help`; 48/48 Python snippets compile.
- [x] `pnpm build` clean (53 pages); 4,927 internal links + fragments resolve; sitemap routes present; canonical only `pymaxim.bio`; `docs.pymaxim.bio/<path>` → 308 path-preserving (live).
- [ ] **After merge/deploy:** re-crawl the live sitemap (recipe in the 1.1 section) and confirm the version line via the `live-site-check` workflow, `/reference/cli/` serves the Console section, `/research/limits/` carries "mitigation-confirmed", and no `selected, not shipped` / `re-stales Exp 53b` / `blob/v1.1.2` strings remain live.
- [ ] **Human-only (carried since 1.0.9):** visual / mobile / keyboard / accessibility pass in a real browser — the new limits table and the CLI Console section at narrow widths, focus order on the copy-install button, contrast of the status chips, alt text on the favicon/og image.

---

## 1.1 audit — 2026-08-25 (gated the `1.1.0` final cut — published 2026-08-26; roadmap step 5b / item 16 / D24)

> **Post-audit additions (2026-08-26):** the Exp 53/53b hardware-readout result landed after this table was written and went live via maxim-web #8 (evidence card + cradle/home pointers, caveats incl. the three-bin representation and "demo, not evidence" for the video) and #9 (release-notes link, video link). Engine sources: `docs/experiments/53_cross_context_readout.md`, `docs/experiments/README.md`.

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
- [x] **After merge/deploy (done 2026-08-26, #7 merged as `97fc13d4`, Workers Build auto-deployed):** all 39 routes 200, canonical only `pymaxim.bio`, `docs.pymaxim.bio/<path>` → 308, "84 experiments", Exp 52 card live, no stale strings (`currently one-shot`, `33 open audit`, `slated for v1.1`, `--persona`, v1 numbers, `[gstreamer]`). Recipe kept — re-crawl the sitemap (`curl -s https://pymaxim.bio/sitemap-0.xml | grep -o '<loc>[^<]*'`, then fetch each; on a sinkholed network use `curl --resolve pymaxim.bio:443:172.67.172.44`) and confirm "84 experiments", the Exp 52 card, and no `currently one-shot` / `33 open audit` strings.
- [ ] **Human-only:** visual / mobile / keyboard / accessibility pass in a real browser (Starlight sidebar + the new home card at narrow widths; focus order on the copy-install button; contrast of the `reference-only` and `partial` status chips; alt text on the favicon/og image).
- [x] After `1.1.0` is published: `pip install pymaxim` in a fresh venv resolves 1.1.0 and `maxim --help` matches the wheel checked here (the site needs no text change for that). **DONE 2026-08-26** — verified by the web session post-publish (fresh venv → 1.1.0; flag set identical); Maxim-web #9 adds the release-notes pointers.

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
