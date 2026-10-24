# pymaxim.bio — website handoff (1.0.9 live audit 2026-08-19 · 1.1 audit 2026-08-25 · 1.1.4 refresh 2026-09-05 · 1.3.0 refresh 2026-09-19 · 1.3.1 refresh 2026-09-27 · 1.3.2 corrections 2026-10-15 · 1.3.2 refresh 2026-10-16 · 1.3.3 refresh 2026-10-24)

## 1.3.3 refresh — 2026-10-24 ("Release pipeline"; PyPI `upload_time` 2026-10-24T00:00Z, signed tag `v1.3.3` @ `c9619217`)

**Framing rule.** 1.3.3 makes **no new behavioural claim**. It ships three things: the fixes maxim-pulse found
in the Console, the Console's live stream from a clean `pymaxim[console]` install, and the first release CI
publishes from a signed tag. The security changes lead. **Every spawned llama-cpp-server now requires a key,
solo included (#1314)**, so the reader must upgrade every maxim on a host together and stop any server an older
maxim started. The site claims no more than `release_1_3_3.md` does. Never write "JEPA". Never imply a benchmark
result nobody ran: the bare-LLM floor B2 has not run. Release links: [GitHub Release](https://github.com/dennys246/Maxim/releases/tag/v1.3.3)
· [PyPI 1.3.3](https://pypi.org/project/pymaxim/1.3.3/). Site fix: **maxim-web `docs/1-3-3`
([#39](https://github.com/dennys246/Maxim-web/pull/39), merged 2026-10-24 after PyPI)**. This section is its engine
companion.

**Engine truth, read from the files, not from memory.** `docs/announcements/release_1_3_3.md`, CHANGELOG
`[1.3.3]`, ledger rows T1-2 and T1-13/14/15, and `evidence_exceptions.json` `release-1.3.3-T1-2`. These were
read on `origin/release/1.3.3`, then re-checked on `origin/main` at `c9619217` (the #1433 merge). All four are
byte-identical between the two refs. Where those files are silent, the release branch's code settled the
point: `models/download.py::_auto_download_enabled` (a set `MAXIM_AUTO_DOWNLOAD_MODELS` decides before config,
so `0` beats a config `true`), `cli_parser.py` (`--clear-memory` help and `--yes`), `doctor/checks.py`
("Loopback lane auth"), `decisions/nac.py` (`decay_interval_hours` dead; `offline_wall_decay` default off) and
`integration/memory_hub.py` (links decay through `decay_all(0.95)` at session end, not per tick).

**Verified against the artifact, not a proxy.**

- **Clean venv** (`python3 -m venv` on 3.14, `pip install 'pymaxim[console]==1.3.3'`). `maxim.__version__` =
  1.3.3. `CONSOLE_CONTRACT_VERSION` = 0.6.0. The vendored `maxim-ui.json` says maxim-pulse v0.4.2 @ `bcd0d1e`,
  contract 0.6.0. `websockets` 17.3 came with the extra, and a live `maxim serve` in a scratch home upgraded
  `/ws` with the bearer token and sent its `identity` frame.
- **Measured in a scratch home:** `maxim --clear-memory nac` with stdin not a terminal listed the agent's
  `nac.json`, refused, and exited 2; with `--yes` it cleared it and exited 0. `NACConfig().offline_wall_decay`
  is False; `InvalidAgentIdError`, `maxim.download_model`, doctor's `check_loopback_lane_auth` and
  `maxim.utils.set_internet_access(..., source="operator")` exist.
- **Flags:** every `--flag` the site PR adds exists in the 1.3.3 `--help` (`--insecure` on `oasis serve`).
- **Components:** regenerated from the wheel (`--label "pymaxim 1.3.3 (PyPI wheel)"`). Only the label changed;
  the 97 refs equal `ComponentRegistry().list_refs()`; links pinned to `v1.3.3`.
- **Live after deploy:** the version line reads 1.3.3 "Release pipeline", 2026-10-24 UTC; 156/156 sitemap routes
  return 200; the new anchors resolve.

**Before the wheel (release branch):**

- `pnpm build` clean, **157** pages. **13,112** internal hrefs and fragments resolve, 0 broken. The checker
  catches a planted missing page and a planted missing fragment.
- No "JEPA" anywhere in the site's `src/`. No PEP 740 claim: the CHANGELOG says attestations were *requested*,
  not verified. No "every extra is tested": only `tts` and `console` have extras-matrix rosters.
- The component YAMLs are unchanged between `v1.3.2` and the release branch. The catalog stays labelled 1.3.2
  until it was regenerated from the 1.3.3 wheel (above).

### Findings (route → claim before → truth source → fix)

| Route | Before | Truth source | Fix |
|---|---|---|---|
| `/getting-started/` | 1.3.2 "Decomposition" version line | announcement; CHANGELOG `[1.3.3]`; PyPI `upload_time` | 1.3.3 "Release pipeline", 2026-10-24 UTC (a `TODO-1.3.3-DATE` placeholder until PyPI), links to the Release, CHANGELOG `v1.3.3` and PyPI, and a "read the upgrade notes first" pointer. A **caution box first** (every server keyed, upgrade every maxim on a host together). Then "no new behavioural claim", explained as no new research claim, with the software's behaviour changing on purpose. Then the three things it ships, the other security and behaviour changes, and the evidence at the cut. 1.3.2 kept as its own section |
| `/installation/` | "Upgrading to 1.3.2" first; "auto-downloads on first run"; the Console extra did not mention `/ws` | Security migration; #1343; #1347 | New **Upgrading to 1.3.3**. **Security first**: the key file (`~/.config/maxim/llm_server_key`, `%APPDATA%\maxim\llm_server_key`), `pip install --upgrade` in every environment, stopping an old server (`pkill -f llama_cpp.server`, the command `llm_server.py`'s own message gives), `MAXIM_LANE_<TIER>_REMOTE_API_KEY` only for a hand-set lane, a writable `~/.config/maxim` in containers. Then #1345 / #957 / #997 / #1311. Then the behaviour changes, each with what to do, **`--clear-memory` first** (#1337), NAc offline decay, #1343, #1310, #1413/#1414, `:` agent ids. Then the Console (#1347, #1332 with the overclaim correction, #1334, #1412, pulse v0.4.2) and **Release integrity**. The model step is now consent-based |
| `/reference/cli/` | "1.3.2 vendors maxim-pulse v0.4.0"; "Since 1.3.2, the voice can be provisioned", read as working; `--clear-memory` "Clear persistent memory" | `[1.3.3]` Changed + Fixed; `cli_parser.py` | v0.4.2 (`bcd0d1e`), contract 0.6.0, 0.7.0 planned for 1.3.4, the pinned-bundle rule. `/ws` from a clean install. **The 1.3.2 "verified Piper voice provisioning" overclaim corrected in plain words.** `pairing_ready` = non-silent audio synthesized once. `--clear-memory` scope (data home, agent stores in every agent home, `./data` only with `--home-dir`, `sim_reports/` never), `--yes`, exit codes 0/1/2 |
| `/systems/nucleus-accumbens/` | Decay section built on `decay_interval_hours` | `nac.py` (dead config); #1361; #1102; `InvalidAgentIdError` | Links decay at session end; per-tick bias decay; fear has none. Since 1.3.3 no offline decay, fear never fades (imported too), `offline_wall_decay=True` opt-in, `apply_decay=` deprecated (removed 1.5.0; ignore the `--resume-sim` warning), load clamp, `:` ids refused, NaN/inf `ValueError` |
| `/guides/simulation/outputs/`, `/memory/engrams/`, `/research/experiments/world-seam/` | "`NAc.load()` ages every bias by wall clock"; fear "decays by wall clock … when saved state is loaded" | #1361 | No offline decay since 1.3.3, with values still clamped. world-seam keeps the at-the-time fact (the experiments ran under load decay) and adds the 1.3.3 truth |
| `/concepts/operating-modes/` | the 1.3.2 section ended the story; `.git/config` writable by file tools | #1345; #957 | New **Since 1.3.3: the operator owns the internet switch**: the agent can only turn internet off for its own session; the operator turns it on with `maxim.utils.set_internet_access(True)`; #1399's limit. The file tools refuse git config, which narrows the git-filter route but does not close it (`bash`) |
| `/guides/benchmarks/`, `/guides/simulation/cli/` | "A suite with no `scoring` block passes automatically" | #1310 | The strict verdict (≥1 threshold applied, every applied threshold held, every expectation met, no errored run), exit 1 otherwise. Before 1.3.3, expectations were validated against an empty record. `cognitive_suite` cannot pass yet (#1393, #1394; #1391 open), with a "expect FAIL" line by the quick start. No results stated |
| `/configuration/`, `/guides/networking/`, `/embodiment/quickstart/`, `/guides/simulation/cli/` | "download GGUFs on first run", "skip the download prompt" | #1343; `download.py` | Consent on every path; `llm.auto_download` is a grant everywhere; a set env var overrides config; the background server never asks, so grant it ahead on a leader |
| `/research/evidence/` | Exp 37 re-run "planned for 1.4" | T1-2 row; `release-1.3.3-T1-2` | 1.3.2 and 1.3.3 each shipped T1-2 STALE under a per-release owner exception; the re-run is the dose-response B4, before 1.4.0, and 1.4.0 waits for it. Goldilocks map only with L8's caveat |

**Left as is, deliberately.** Every historical "since 1.3.2" fact. The 1.3.2 section and "Upgrading to 1.3.2"
(their anchors are linked). Exp 60/61/62 PARTIAL as 1.3.2 left them. The hippocampus and outputs cautions:
#939 and #950 are closed, and 1.3.2 already rewrote those cautions. The still-owed grounding corrections are
held for their own PR.

**Not carried to the site.** The 1.4.0 release-gate function ceiling (#1323), the prompt-roster internals (D82,
`Executor.set_forbid_source`), the #1311 key-reader internals, and the SCN/NAc credit-path fixes (#1363, #1365)
beyond the NaN `ValueError`. The site reaches them through the CHANGELOG link.

**Two independent reviewers (before the PR).**
1. A **fresh reader** (a 1.3.2 user who never saw the plans) found **13** issues, all folded. The material
   ones:
   - "stop any server an older maxim started" gave no way to find one;
   - `pip install --upgrade` sat above "upgrade every maxim together";
   - "per-machine key" against a home-directory path;
   - `--clear-memory` ranked third despite being the change that can delete data, and with `--yes` it now
     reaches every agent;
   - the NAc opt-in is Python-only (now stated: no CLI or config switch);
   - no operator command to turn internet back on;
   - "lane" and `<TIER>` unexplained;
   - "no new behavioural claim" beside "behaviour changes";
   - the evidence paragraph's jargon;
   - the benchmark rule stated three ways.
2. A **claim check** against the release branch covered about 115 claims: **0 wrong, 0 overclaims**, 6
   imprecise, all folded:
   - links decay at session end;
   - "synthesized non-silent audio", not "spoken audible audio";
   - the full four-part benchmark rule;
   - `scheme://host[:port]`;
   - "no offline decay" in place of "exactly as saved", given the clamp;
   - the `--clear-memory` scope.

### Acceptance checks — 1.3.3

- [x] Version line 1.3.3 "Release pipeline", 2026-10-24 (UTC upload time); links to the Release, CHANGELOG,
  PyPI 1.3.3; `live-site-check` line format kept.
- [x] The security change leads: every llama-cpp-server keyed, upgrade every maxim on a host together, stop old servers, lane key, container home.
- [x] "No new behavioural claim" stated; no result changed status; T1-2 STALE under `release-1.3.3-T1-2` with the Goldilocks caveat; Exp 60/61/62 PARTIAL; B2 not implied.
- [x] Every correction in the brief is on the site (Piper overclaim, v0.4.2, `/ws`, download consent, `--clear-memory`, internet toggle, NAc, benchmarks).
- [x] Release integrity: signed tag, green nightly on the exact commit, exact-wheel smoke test, trusted publishing, pinned newest pulse bundle; extras testing limited to `tts` and `console`; no PEP 740 claim.
- [x] Fresh-reader and claim-check passes folded (13 + 6).
- [x] `pnpm build` clean (157 pages); 13,112 internal links + fragments resolve.
- [x] **On the wheel:** clean venv with `pymaxim[console]==1.3.3`; `--help` shows `--clear-memory` / `--yes` as written; `maxim doctor` has "Loopback lane auth"; `maxim.download_model` exported; `/ws` upgrades on a clean console install; regenerate components (`--label "pymaxim 1.3.3 (PyPI wheel)"`, links `v1.3.3`); sweep the site's flags against `--help`.
- [x] **After merge/deploy:** live sitemap re-crawled (156/156 return 200); version line confirmed live; `/getting-started/`, `/installation/#upgrading-to-133`, `#release-integrity` and the operating-modes 1.3.3 section spot-checked.
- [ ] **Human-only (carried):** a visual, mobile and keyboard pass in a real browser, especially the new caution box.

## 1.3.2 refresh — 2026-10-16 ("Decomposition"; PyPI `upload_time` 2026-10-16T20:47Z, signed tag `v1.3.2` @ `59d35ad3`)

**Framing rule.** The security fix leads. For anyone who runs a leader, 1.3.2 is a security release
([GHSA-7qh4-mp34-88v9](https://github.com/dennys246/Maxim/security/advisories/GHSA-7qh4-mp34-88v9),
published 2026-10-16T20:51Z). The site's first words about the release say who is affected, what changed
(default role `solo`, a leader without a usable key refuses to start, remote admin opt-in, peers on `:8099`),
and the one upgrade rule that matters: **upgrade each leader on its own machine and cold-restart it, never via
`maxim peer update && maxim peer restart`**. After that, the release is what it says it is: **no new
behavioural claim**, the decomposition, breaks that each name their migration, and the evidence at the cut
(T1-13/14/15 PARTIAL, already on the site since the corrections section below; T1-16 EARNED narrow and T1-1
SUPERSEDED; T1-2 STALE under `release-1.3.2-T1-2`). Release links: [GitHub Release](https://github.com/dennys246/Maxim/releases/tag/v1.3.2)
(the release notes; there is no `release_1_3_2.md`) · [PyPI 1.3.2](https://pypi.org/project/pymaxim/1.3.2/).
Site fix: **maxim-web `docs/1-3-2`**; this section is its engine companion.

**Engine truth read on `origin/main` (`59d35ad3`), not from memory:** CHANGELOG `[1.3.2]` (its body is
byte-identical to the `[Unreleased]` the site was prepared from at `56c25d97`, plus the release header and
the Security entry); the GitHub Release body; the published advisory; ledger rows T1-1, T1-16 (and the
T1-2/13/14/15 rows from the corrections pass).

**Verified against the artifact, not a proxy.**

- **Clean venv** (`python3 -m venv` on 3.14, `pip install 'pymaxim[console]==1.3.2'`). `maxim.__version__` =
  1.3.2. `CONSOLE_CONTRACT_VERSION` = 0.6.0; the vendored UI's `maxim-ui.json` says maxim-pulse v0.4.0,
  contract 0.6.0. `ComponentRegistry().list_refs()` = **97** = the regenerated `components.json` count
  (`--label "pymaxim 1.3.2 (PyPI wheel)"`); the only catalog diffs are the label and one description that
  now names `runtime/substrate_proposal` (the `agent_loop` move).
- **Measured in a scratch home** (throwaway `HOME`, `XDG_CONFIG_HOME`, `MAXIM_DATA_HOME`):
  - a fresh `Hippocampus(config)` over an existing file raises `StoreOverwriteRefused` on `save()`;
    `create.hippocampus` over it raises at construction; `overwrite=True` replaces;
  - `~` paths expand, a missing `~` file raises `FileNotFoundError`;
  - `load.session` with a prefix matching two runs raises `RunDirAmbiguous`; an exact id or a unique prefix loads;
  - default role with `MAXIM_ROLE` unset resolves `('solo', 'default')`;
  - `maxim config set proxy.allow_remote_update true` reads back `True [source=config]`; `config unset`,
    `downgrade`, `restore-preserved` exist.
- **CLI flags:** every `--flag` on the site — **149** unique strings — swept against the 1.3.2 wheel's
  `--help` (top-level verbs, the argparse sharing verbs, `model add`, `roy run`, `python -m
  maxim.models.download`). Every `maxim` flag exists; the rest are other tools' flags (pip, docker, nmcli, the
  Reachy daemon, CSS tokens) or flags the site says do not exist (`--resume-session`, `--fixture`,
  `--prompt-profile`, `--explore-weight` on `maxim`). One hazard for the next sweep: **`maxim tunnel setup
  --help` ignores `--help` and runs setup**; sweep only argparse verbs.
- **Python:** **48/48** site snippets compile; the store, reopen-and-save, session and inspect snippets run
  on the wheel.
- **Build:** `pnpm build` clean, **157** pages; **13,078** internal hrefs + fragments resolve, 0 broken; all
  **156** sitemap routes present.

**Engine findings while running the snippets (not filed; owner's call):**
1. `maxim.load.hippocampus(path)` returns a store whose `config.persistence_path` is `None`, so a bare `save()`
   raises `ValueError: No persistence path specified`; `save(path)` works. The site documents `save(path)`.
2. The `[1.3.2]` #829 bullet says "every runtime request … may only lower" capability. Measured: the
   model's `mode_switch` (#924) and the robot runtime's re-exec requests are lower-only, but the robot
   runtime's phrase table ("maxim active", `StateManager.request_mode_active`) still sets the state manager's
   mode passive → active with no capability check, and any audio in the room can say it. Per the wheel's own
   comment in `modes/state_manager.py`, dispatch reads `Executor.effective_operational_mode()` (the launch
   grant, else the run mode), which the phrase does not change, so the gap is in the label, not in what tools
   run. The site states it that way.
3. `maxim config get role` labels the default `solo` as `[source=env]` with `MAXIM_ROLE` unset.

### Findings (route → claim before → truth source → fix)

| Route | Before | Truth source | Fix |
|---|---|---|---|
| `/getting-started/` | 1.3.1 "Hardening" version line; nothing on 1.3.2 | PyPI `upload_time`; Release body; advisory; CHANGELOG `[1.3.2]` | 1.3.2 "Decomposition", 2026-10-16 UTC, links to the Release, CHANGELOG, advisory, PyPI; a **danger box first** (who was exposed, what fails closed now, upgrade on the leader + cold restart, never `peer update && peer restart`); then "no new behavioural claim", the decomposition, five what-changes bullets, the evidence at the cut (incl. Exp 63); 1.3.1 kept as its own section |
| `/installation/` | "Upgrading to 1.3.1" only | CHANGELOG Security migration + breaks | New **Upgrading to 1.3.2**: **Leaders first** (doctor; `config set role leader`; `tunnel key rotate` + `peer key set`; `chmod 600`; `proxy.allow_remote_update`; legacy env ignored; `:8099`), then memory stores (#939/#950/#971/#1071, `save(path)` after `load.*`), runs and data (#932), configuration (format 1.3, `unset`, `downgrade`), internet (#832/#966–#968), modes and host tools (#829/#924/#949), moved internals |
| `/guides/networking/` | leader "binds `0.0.0.0`"; default fallback **leader** (rank 7, contradicting "Solo … The default"); debug routes "Bearer-gated or localhost-only"; remote peer management unqualified | Security entry; advisory | Leader row: :8099 the one ingress, refuses to start without a key, llama-cpp-server on `127.0.0.1`; rank 7 → **solo** with the advisory link and `role_default_with_mesh_key`; a fail-closed paragraph (exit 2, loopback never auth, remote admin opt-in, legacy env ignored, peers on :8099); debug routes need the key from localhost too; two troubleshooting rows; "since 1.3.2 opt-in on the leader" |
| `/configuration/` | `role` comment without default; no 1.3.2 commands | Security entry; #974; config format | `role` default solo; `proxy.allow_remote_update`; `config unset` / `downgrade` / `restore-preserved` with what each is for; format 1.3 |
| `/reference/cli/` | console contract 0.5.0; UI bundle story ends at 1.2.1; pairing "no shipped command"; `oasis serve` refuses only non-loopback | `ui_bundle.py` + wheel manifest; #1268 F1; Security entry | 0.6.0 and the five run-history/health routes, stall detection and its limits, opt-in `/ws` kinds; 1.3.2 vendors pulse v0.4.0 (0.6.0); Piper provisioning as capabilities with **no pymaxim caller**; `oasis serve` refuses without a key on any bind, loopback included |
| `/concepts/operating-modes/` | `mode_switch` refuses only code-executing modes; "passive → active is not gated (#924, open)" | #829, #924, #949, #832, #966–#968, #921 | `mode_switch` refuses any raise; new **Since 1.3.2: capability goes down, not up** (lower-only for the model and robot runtime, the human phrase still raises, `--operational-mode`, host-tool env and git opt-ins with #957's limit, internet fail-closed with the proxy limit) |
| `/systems/hippocampus/` | caution box: the constructor silently overwrites; `~` not expanded; #939 open | #939/#950/#971/#1071; wheel runs | Rewritten for `StoreOverwriteRefused`, `create.*` refusing at construction, load-then-`save(path)` (runnable), `overwrite=True`, `~` expanded |
| `/guides/simulation/outputs/`, `/orchestrator/` | research results under `./data/sim_reports/`; `actions.jsonl` 1.1; `load.session` prefix "newest first"; `~` not expanded | #932; `report.py` (`_ACTIONS_JSONL_FORMAT_VERSION` 1.2) | Research under `<data home>/sim_reports/research_<id>/`; 1.2 header and `rpe` / `pain` / `nac_failure_booked`; `RunDirAmbiguous`; `sessions/` note corrected (it held `maxim.run(learning=True)`'s agent through 1.3.1) |
| `/systems/nucleus-accumbens/` | `_record_outcome()` in `runtime/agent_loop.py` | `runtime/tool_dispatch.py::record_outcome` | One dispatch path; a failure booked once by the bridge |
| `/research/evidence/`, `/research/experiments/cross-session-learning/`, `/memory/engrams/`, `experiments.json` | Exp 10 "maintained (narrow), 2026-09-27"; no Exp 63 | T1-1 SUPERSEDED, T1-16 EARNED (narrow) | Persistence row earned (narrow) 2026-10-04 by Exp 63 with its scope and caveats (#1064, #1260, floor not measured); Exp 10's September re-run kept as the record; new `exp-63` entry superseding `exp-10`; the cross-session line's best is Exp 63 |
| `/reference/components/` | label 1.3.1; card links pinned to `v1.1.4` | wheel registry | Label 1.3.2, links pinned to `v1.3.2` |

**Left as is, deliberately.** The 1.3.1 section and "Upgrading to 1.3.1" (history; their anchors are linked).
Every Exp 60/61/62 number and status from the corrections pass. Snippets the wheel runs unchanged.

**Not carried to the site.** The decomposition's internals (slice names, ratchets, coverage floors, mypy set)
beyond one sentence; entity file format 1.1, the provenance and evidence-gate machinery, and the removed
internals, which the upgrade section reaches through the release notes link.

**Different-reader pass (before either PR).** An independent agent re-checked every changed claim against CHANGELOG
`[1.3.2]`, the Release, the advisory, the ledger and the installed wheel, running the store, session, config
and internet-policy cases live in a scratch home. Nine findings, all folded. The material ones: the
"Maxim active" phrase was described as a CLI-loop human path (it is the robot runtime's phrase table, any
audio can say it, and it does not change what dispatch enforces); `load.atl(p).save()` is a silent no-op, not
a `ValueError` like Hippocampus and NAc; the leader-first text under-stated who was exposed (the Python API
reached the leader default on any GPU host, and :8100 listened on every interface with its key in argv);
`nac_failure_booked` is the booked signature, not a boolean; `create.*` refusal is three named functions,
not all of `create`; the bridge books a failure only where it is present; the proxy limit is downloads and
backend URLs, not the agent's fetch; and Exp 10's manifest finding still said "measurably different
behavior". Plus an explicit `pip install --upgrade` step on the leader and the cloud auto-detect note.

### Acceptance checks — 1.3.2

- [x] Version line 1.3.2 "Decomposition", 2026-10-16 (UTC upload time), links to the Release, CHANGELOG, advisory and PyPI 1.3.2; the `live-site-check` line format kept.
- [x] The security fix leads, with the advisory link, the changed defaults (solo, remote admin opt-in, :8099) and the on-the-machine cold-restart rule; `peer update && peer restart` named as what not to do.
- [x] "No new behavioural claim" stated; Exp 60/61/62 never EARNED in the present tense; Exp 63 narrow; Exp 37 stale.
- [x] A different reader checked every changed claim against the CHANGELOG, Release, advisory, ledger and wheel; its 9 findings are folded.
- [x] Every upgrade break that touches a site snippet or guide is reflected (stores, run dirs, `actions.jsonl`, config, internet, modes, host tools, console).
- [x] 149 flags swept; 48/48 snippets compile; store/session/config snippets run on the wheel.
- [x] Counts derived: components 97 from the wheel registry; experiments from `experiments.json`.
- [x] `pnpm build` clean (157 pages); 13,078 internal links + fragments resolve; 156/156 sitemap routes present.
- [ ] **After merge/deploy:** re-crawl the live sitemap; confirm the version line via `live-site-check`; spot-check `/getting-started/`, `/installation/#leaders-first-the-security-fix` and `/guides/networking/`.
- [ ] **Human-only (carried):** visual / mobile / keyboard pass in a real browser, especially the new danger box.


## 1.3.2 corrections — 2026-10-15 (phase 1 of the 1.3.2 refresh; not a release refresh — 1.3.2 is not on PyPI)

**Framing rule.** The project applied its own stricter bar (BF2, `docs/plans/baseline_floors.md`) to
itself and lost that comparison on its three survival-fear rows. The site says so plainly, as a
credibility point, not a retraction: each row still beats its internal ablation; none shows behaviour
beyond a no-bio learner. Exp 60, 61 and 62 are **never** EARNED in the present tense anywhere on the
site. No bare-LLM floor result is implied: B0 shipped provenance only, B1 was the non-LLM floor, and
B2 (`bare_llm`) has not run; the site says B2 is planned and that no new LLM-path claim can be EARNED
until it exists. Nothing on the site describes the 1.3.2 release itself (version line, features,
upgrade breaks) — that is phase 2, after `v1.3.2` is on PyPI and tagged. Site fix: **maxim-web `docs/1-3-2-corrections`**; this
section is its engine companion.

**Engine truth read on `origin/main` (`cfc3a491`), not from memory:** the graduation ledger's T1-1,
T1-2, T1-13, T1-14, T1-15 and T1-16 rows; `docs/plans/baseline_floors.md` (BF2, §Why this plan
exists, Stages B0–B5); `docs/experiments/b1_exp60_floor_prereg.md` (frozen v5);
`docs/experiments/data/b1_exp60_floor_48ce40118c20_verdict.json` (`FLOOR_MATCHED_Q`; `tests.H1`
p 0.0035, `tests.H2` p 1.0; `first_post_rate` fear 1.0 / q_explore 1.0 / random 0.25; `n_clean` 8 per
arm); CHANGELOG `[Unreleased]` "Evidence status at the cut" (T1-2 STALE, exception
`release-1.3.2-T1-2` — **not** carried yet, since it is a release statement); #1283, #1294, #1296;
#1260, #1261, #1289, #1299, #1301; #1300.

**Checks.** `pnpm build` clean, **156** pages (one new: the B1 manifest entry); **12,980** internal
hrefs + fragments resolve in `dist/` (URL-decoding checker), 0 broken; all **155** sitemap routes
present, including `/research/experiments/b1-exp60-floor/`. No snippet, CLI flag or components count
changed, so nothing was re-checked against a wheel; that is phase 2's job, on the 1.3.2 wheel.

### Findings (route → claim before → truth source → fix)

| Route | Before | Truth source | Fix |
|---|---|---|---|
| `/research/evidence/` | Exp 60 / 61 / 62 "earned"; no floor axis; "how a claim is earned" never explained; Exp 37 "partial"; Goldilocks map without "unverifiable, not refuted"; Exp 12 cited with no caveat | T1-13/14/15 rows; `baseline_floors.md` BF2; B1 prereg + verdict; T1-2 row; L8; #1301 | New **How a claim is earned** (one BF2 paragraph: ablation vs floor, the substrate and LLM floors, `partial (narrow: floor matched)`, pre-rule rows "not measured"); summary table gains a **Floor** column beside status (matched-measured / matched-argued / not measured / —) and the three rows move below the earned ones; new **The floors** section (B1 numbers, the tie is "not shown", never "equivalent", the 31-vs-0 efficiency gap, the three analytical reasons for 61/62, what stands / what is gone / what they show instead, B2 not run, B4 in 1.4; absolute links to the prereg, the verdict and `baseline_floors.md`); a Status line on each of the three claims; Exp 37 **stale since 2026-10-02 (was partial)** with L8's caveat; Exp 12 caveat (#1261 receipts, #1260 `respond` unavailable, both fixed, raw scoring files never committed, #1301 open); four dated entries in the corrections list; "Where this is going" gains the E1–E3 obligations (mechanism twin as a run arm; a contingency a reasonable floor can lose) as planned work, plus the two engineering notes (the narrator hears the agent's words; the 1.4.0 decomposition gate, #1300) |
| `/getting-started/` | "earned" / "earned, the headline" on the Exp 60 / 61 bullets; Exp 62 "on the engine's evidence ledger" | T1-13/14/15 rows | Both bullets carry the partial status and date; a new bullet states the stricter bar, the 8/8 tie, what still stands and the 31-vs-0 gap; Exp 62 "partial (floor matched, by argument)". The 1.3.1 version line is untouched (phase 2) |
| landing (`index.astro`) | fear card ends on R3, no floor | same | One sentence on the stricter bar: the floor escaped as well, both results now partial, what they still show; a "The floors →" link |
| `/research/experiments/world-seam/` | "earned" ×3 in the claim summary and the experiments table; no floor | same; B1 verdict | Partial statuses and dates; a **Then the floor** paragraph; "Behaviour beyond a no-bio learner" first under *What it does not show*; a Status line atop each of the three experiment sections; B1 row in the table |
| `/memory/engrams/` | situation engram "EARNED ×7"; "three most recent EARNED rows", each "EARNED <date>"; "the earned drowning fear" | same | "EARNED ×4, PARTIAL ×3"; the three rows "earned <date>; PARTIAL, floor matched, <date>" with a lead paragraph saying the ablation results stand and what is gone |
| `/systems/fear-circuit/`, `/guides/oasis/` | Exp 60 / 61 shown "to drive behavior"; Exp 62 "earned the day after 1.3.0" | same | One sentence each: partial since October 2026, what still stands (transport for Exp 61) |
| `/research/experiments/cross-session-learning/`, `/roy-harness/` | Exp 37 "partial"; Goldilocks map uncaveated; Exp 12 "clean pass" uncaveated | T1-2 row; L8; #1301 | Exp 37 stale since 2026-10-02, re-run with floors in 1.4 (B4); "unverifiable, not refuted" wherever the map appears; Exp 12 caveat paragraph, including that the Acting Coach's "3 respond hits vs 1" is confounded by #1260 |
| `/research/experiments/` (vocabulary) | "Earned — met its gates"; "Partial — some gates failed" | BF2 | Earned also needs the floor since 2026-10-10; partial includes *floor matched* (Exp 60–62) |
| `src/data/experiments.json` | exp-60/61/62 `earned`; exp-12 and exp-37 uncaveated; no B1 | as above | exp-60/61/62 → `partial`, the qualifier in the summary and finding (not `phase`, which renders as "Covers:"), findings rewritten (what stands, the floor, what they show); exp-12 caveat + summary; exp-37 graduation and cross-model stale + L8 wording; new `b1-exp60-floor` entry (`reference`, follows exp-60, walkthrough to `/research/evidence/#the-floors…`) |

**Left as is, deliberately.** Every Exp 60 / 61 / 62 number and scope, byte for byte — only status,
floor and "what it shows" lines were added. Exp 56, 52, 42, 45, 53b, 49 keep EARNED with "floor: not
measured" (BF2: pre-rule rows keep their status until their next re-run). The 1.3.1 version line, the
upgrade guide and the components catalog (phase 2).

**Not carried yet.** Anything about the 1.3.2 release: its version line, the `release-1.3.2-T1-2`
exception wording, the release's features and upgrade breaks (phase 2,
after PyPI + tag). **Exp 63 (T1-16)** and Exp 10's move
to SUPERSEDED (2026-10-04): the site still shows Exp 10 "maintained (narrow), 2026-09-27", which the
ledger no longer says; held for the owner's call on whether it lands in phase 1 or with the 1.3.2
refresh.

**Different-reader pass (before either PR).** An independent agent re-checked every changed claim against the
sources above. Eleven findings, all folded: the Exp 61 status sentence attached "by construction" to a
"transferred" prior (it is the same *defensive* prior; the by-construction point is the state-value
twin); "Maxim lost that comparison" overclaimed a tie and a single measured row (now "did not beat it:
on Exp 60 the floor matched; 61 and 62 followed by argument"); three pages applied Exp 60's measured
8/8 to Exp 61 as if run; "Pavlovian" was unsourced and invites the prediction-error reading the B1
prereg rules out (now "context-conditioned"); Exp 37 was said to carry both narrator defects (#1301
says only #1260); the defects' window is 2026-04-03 to the fixes (#1299 merged 2026-10-16 UTC); the
E1–E3 contingency is conditional on wanting to EARN over the floor, and the twin is the *mechanism*
twin; "measures that contradiction as much as the coach" implied an unmeasured magnitude; #1260 is a
simulator defect, not a narrator one; the status qualifier had been put in the manifest's `phase`
field, which renders as "Covers:" (moved out; Exp 62 keeps "rung A"); and lay glosses for "tabular
Q-learner" and "ablation" on the landing and getting-started pages.

### Acceptance checks — 1.3.2 corrections

- [x] No page says Exp 60, 61 or 62 is EARNED in the present tense; each shows its partial status, date and one line on the floor.
- [x] `/research/evidence/` explains BF2 in one plain paragraph, shows the floor as an axis beside the ablation, and links the B1 prereg, the verdict and `baseline_floors.md` with absolute GitHub links.
- [x] "Not shown" for FEAR > Q, never "equivalent"; the 31-vs-0 efficiency gap stated; what stands / what is gone / what they show instead stated.
- [x] No sentence implies a bare-LLM result; B2 described as planned, and no new LLM-path claim earnable until it exists.
- [x] Exp 37 stale with L8's "unverifiable, not refuted" wherever the Goldilocks map appears; Exp 12 carries the #1301 caveat, including the Acting Coach count.
- [x] E1–E3 obligations described as planned, nothing claimed.
- [x] Nothing describes the 1.3.2 release.
- [x] `pnpm build` clean (156 pages); 12,980 internal links + fragments resolve; 155/155 sitemap routes present.
- [ ] **After merge/deploy:** spot-check `/research/evidence/#the-floors-the-first-comparison-with-a-no-bio-learner`, `/research/experiments/b1-exp60-floor/` and the landing card live.
- [ ] **Phase 2 (after `v1.3.2` is on PyPI):** the 1.3.2 refresh section, in this file's usual shape.


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
