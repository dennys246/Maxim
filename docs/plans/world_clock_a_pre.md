# World clock, A-pre: a split clock and a perceived-time port, outside the fence

> **PROPOSED 2026-11-24, plan only.** 1.3.5 line 2 ([release_1_3_5.md](release_1_3_5.md), owner decision D9). No
> `src/` change in this PR. Every mechanism enters as `[engineering]`. The owner decisions it rests on are in
> §Decisions; nothing in it is decided by wording.
> **Input:** [overdrive_refactor_candidates.md](overdrive_refactor_candidates.md) §A3, and Overdrive's position, read
> with `git show` at Overdrive `origin/main` `9623be7`: `docs/roadmap.md` Phase 3 tier A (OD-A1 to OD-A25, OD-H6,
> OD-I8, OD-I13) and `docs/design/reviews/maxim_response_2/response.md`. Where those differ from Overdrive's
> `docs/maxim_upstream/world_clock_release_DRAFT.md` (last changed 2026-11-16, `0748fee`), the roadmap rows win; the
> draft is background only.
> **Rides on:** [experience_clock_decay.md](experience_clock_decay.md) (#1361). This plan adds no second experience
> clock and owns no decay, eligibility or temporal window.
> **Guard approach note:** [world_clock_a_pre_guard_note.md](world_clock_a_pre_guard_note.md) (a gate, so it gets an
> adversarial design pass before code).
> Every `file::symbol` here was re-checked on Maxim `origin/main` `f145e27b` (2026-11-24). Line numbers are hints.

## What A-pre is

**The claim.** Time reaches an agent from two sources, and both are injected:
- a **machinery clock**, for stamps, windows and cadences;
- a **perceived-time port**, the only time a prompt or tool result may contain.

Their host defaults return exactly what today's code reads, so robots, the console and pulse are unchanged.

**A-pre** is the half that touches no fenced file. The fenced files are `runtime/agent_loop.py`,
`simulation/orchestrator.py`, `decisions/nac.py`'s selection path and `runtime/tool_dispatch.py`.

**Done means:**
- the planted-render test is green: no time from the machinery clock, and no host time zone, reaches a built prompt;
- the M72 lint passes on A-pre's site families;
- WC5, robots unchanged, holds.

1.3.5 needs only this plan reviewed and PR 0 merged (§Exit). The other PRs merge whenever they are green (merge order
step 5).

**Overdrive ids this plan takes:** OD-A2, OD-A3, OD-A5, and the A-pre parts of OD-A4. OD-A6 is planned here but lands
after #1200, at the A-pre/A-post boundary (§PR 5). Everything else is placed in §Placements.

## Motivation and front gate

**Today one wall clock serves three different jobs.**
1. **What the agent is told.** `PromptBuilder.build_prompt` and `PromptBuilder._build_engage_prompt` render
   `datetime.now()`. `ContextPool._entries_to_text` and `ContextPool.get_context_text` stamp every observation
   `[HH:MM:SS]` through `time.localtime`. `retrieval/temporal_signal.py` resolves "yesterday" against
   `datetime.now()`. And `TemporalSignature.from_timestamp` bins SCN phases in host local time, which reaches the
   model through the `[TEMPORAL]` tag (`agents/statistician_agent.py`) and the `temporal_patterns` tool.
2. **What the agent's machinery counts.** Examples: the Hippocampus's 30 s de-duplication window
   (`Hippocampus.store_observation`, `dedup_window_s`); capture and access stamps; salience and gaze stamps;
   `SpecSensor.read`'s `SensorReading` timestamp.
3. **What the process watches:** heartbeats, timeouts and motor and IO limits.

An embedded agent, one whose host steps it in world time (Overdrive tier A, O6), needs (1) and (2) to come from the
host and (3) to stay on the wall. Maxim gains from this too:
- **Replay:** a run whose prompts carry no host time and whose windows count injected time can be replayed with one
  variable changed (candidates doc §Why).
- **Portable prompts:** a prompt built under `TZ=UTC` and one built under `TZ=Pacific/Auckland` differ today only by
  the host's time zone.

**Front gate: does this need its own mechanism?** Only the port does.
- **The machinery clock rides on what exists.** Its host default is the same `time.time()` / `time.monotonic()` read
  each site makes today, read through the module attribute at call time (so `tests/unit/_loop_harness.py::_StepClock`
  still reaches it). On a robot, decay and drift already move to `runtime/experience_time.py::ExperienceClockDriver`
  under #1361. This plan does not touch that half.
- **The perceived-time port is new**, and the reason is specific. `ExperienceClockDriver` separates experience time
  from wall time, but it has no notion of "time the agent is told", and nothing else in `src/` does. Every prompt
  site calls the wall clock directly. The port is a two-method protocol in the existing `time/` package: no bus, no
  thread, no persisted state.
- **No new lint.** The guard extends M72 (#1361's planned lint, still unbuilt) rather than adding a second one (owner,
  2026-11-24).

## Decisions

**Owner, 2026-11-24, asked together at the start of this plan:**
- **D-WC1. Scope: narrow; G17 stands.** A-pre is OD-A2 to OD-A6. It adds no `Rng` and no `IdSource`. Id and
  randomness sites are listed in the census and placed as follows:
  - OD-A7 (`similarity/semantic.py::_initialize`'s `or 42`, `seed_all` off the embedded path, `NAc.record_event`'s
    `time.time_ns()` ids) is a new release-gate item **after the 1.4.0 cut**, not a widened `1305.owned-rng` (owner
    decision 2026-11-19, re-cited here).
  - Session ids wait for the post-fence resume stage (DECISIONS.md G17, unchanged).
  - The draft's wider `AgentTime` bundle is not adopted.
- **D-WC2. OD-A5 phase source: the port for prompt-reaching producers; NAc keeps the host calendar.**
  `TemporalSignature.from_timestamp` gains an optional calendar. Called without one, it reads exactly what it reads
  today. Producers whose output reaches the model pass the port's calendar; under "no calendar declared" they emit no
  weekly, monthly or annual phase. NAc's anchor-similarity callers (`decisions/temporal_credit.py` and the
  `TemporalSignature.now()` read in `decisions/nac.py`) pass nothing, so OD-A20's decline (2026-11-19) holds and its
  two red gates stay red.
- **D-WC3. OD-A6: the PainBus and ReactionBus de-duplication refractories count the machinery clock, after #1200.**
  `PainBus.publish`'s per-`(entity, failure_mode)` window and `ReactionBus`'s `DEFAULT_REFRACTORY_S` window read the
  machinery clock, whose host default is today's `time.monotonic()`. Motor and IO rate limits stay on the wall. A
  static check refuses any motor-dispatch module that reads the injected clock (I3). This amends
  experience_clock_decay.md §(g) (amended in this PR). It lands after #1200's PainBus rewiring, so it is placed in
  1.3.6.
- **D-WC4. The guard is an extended M72 plus a long-horizon run.** M72 becomes one AST lint with two scopes:
  machinery (the step-path set) and prompt modules. A long-horizon scripted run (Overdrive BD-14.2) is a strict red
  gate, because static scans miss the consolidation, sleep and decay paths. The draft's `sys.setprofile` tripwire is
  dropped. The M72 row and §(g) are amended in this PR.

**Already settled, cited rather than re-asked:**
- **Is embedding a supported use?** (draft §8 Q1) Yes. The candidates doc takes tier A on engineering grounds, and A4's
  front gate passed (owner, 2026-11-19).
- **Does A-pre take its own patch number?** (draft §8 Q2) No. D9 puts it in 1.3.5 line 2.
- **Rng, `IdSource` and #1305:** decided 2026-11-19 (see D-WC1).
- **OD-H6:** Overdrive withdrew the loop-interval argument. Under a host-called `think`, `llm_submit_interval` needs no
  change for embedders, and it lands inside cand. B1 (OD-A15).
- **OD-A20:** declined 2026-11-19 (see D-WC2).

**Decided by this plan (engineering; reviewers may contest):**
- **#1769: cost accounting owns the wall clock.** Spend is real money, and a monthly ceiling is a calendar month of
  the bill. So `CostTracker` keys its month on the wall clock, never the caller's `timestamp`, and the M72 allowlist
  says why (category `cost`). This fixes the mixed-clock reset at its root, and it lands in PR 1.

## The design

### Two collaborators in `time/agent_time.py`

| Collaborator | Surface | Host default (robots, console, pulse) | Under embedding |
|---|---|---|---|
| `MachineryClock` | `now_s() -> float` (epoch seconds, the unit of every persisted stamp today); `monotonic_s() -> float` | `time.time()` and `time.monotonic()`, read **through the `time` module attribute at call time** | the host's world epoch seconds and world monotonic seconds |
| `PerceivedTime` (the port) | `now() -> PerceivedMoment \| None`; `perceive(machinery_s: float) -> PerceivedMoment \| None` | `datetime.now()` and `datetime.fromtimestamp(machinery_s)`, local time, as today | the host's world calendar, or `None` (no calendar declared) |

- **`PerceivedMoment`** carries the components the render sites and `TemporalSignature` use: year, month, day,
  weekday, hour, minute, second and day of year. It is a frozen value that never persists or crosses a wire, so CC3
  does not apply; its docstring says so.
- **`None` means "omit".** A render site that gets `None` renders nothing: no section, no `[HH:MM:SS]`, no relative
  date. That is how Overdrive declares no calendar (npcs_and_sem.md §3.1, "No time field reaches cognition").
- **Rendering a machinery stamp goes through `perceive`, never `time.localtime`.** This is the root fix for
  `context_pool`. Its stamps are machinery values, so rendering one through the host's `localtime` leaks the host
  time zone even when the stamp is world time.
- **No ambient state.** Collaborators are passed explicitly, with defaults kept, through the builders that construct
  their consumers. That means `runtime/bio_stack.py::build_bio_stack`, `runtime/agent_factory.py::AgentFactory`, and
  the `PromptBuilder` / `ContextPool` constructors. There is no module global and no `ContextVar`. Maxim rejected the
  `ContextVar` back channel in `proprioception/pain_bus.py`'s docstring, and an ambient that falls back to the wall
  when unset is a silent fallback.
- **Callers.** Every production builder passes the host default explicitly, so the seam runs on the shipped path from
  its first PR. The non-default caller is the embedded entry point (OD-A10, cand. A4), which is not in this plan.
  Until then the embedded behaviour is exercised by tests only, and each PR's description says so ("A fix ships with
  a CALLER": the robot path is the caller, and the embedded path is capability).

### A stamp moves with every reader that subtracts from it

If a stamp moves to the machinery clock while its reader still subtracts `time.time()`, the embedded result is
garbage: world epoch minus wall epoch. Under the host default the two are the same read, so nothing shows. So the
census lists **stamp–reader pairs**, not just sites, and a PR moves a whole pair or none of it.

Example: the Hippocampus's capture and access stamps (`Hippocampus._build_capture_record`, `Hippocampus.get`,
`Hippocampus._touch_internal`) pair with `sleep()`, eviction (`Hippocampus._evict_one`) and
`memory/hippocampus_consolidation.py`'s `time.time()` reads. Under the host default the machinery read **is**
`time.time()`, so #1361 Q5 ("don't extend no-offline-forgetting to Hippocampus/ATL now") is untouched for robots.

### What stays on the wall, on purpose

This plan amends experience_clock_decay.md §(g)'s table (D-WC3, D-WC4). The M72 allowlist is that table, keyed by
`file::function`, with one category per entry:

| Category | Examples |
|---|---|
| `process` | heartbeat, stall detector, LLM timeouts, TTFT keepalive, the driver's stall cap |
| `scheduling` | `idle_sleep_s`, `target_period`, `llm_submit_interval` (OD-H6), the 120 s `_submitted_recently` window |
| `motor-io` | motor and IO rate limits, reflex cooldowns that gate a motor (embodiment brief §1) |
| `real-world-io` | `HttpFetchTool.execute`'s `accessed_at` (a real fetch's UTC time); `exec_agent.py::_build_llm_context`'s workspace-file mtimes (a real file's date); `FocusOnSoundTool.execute`'s "faded …s ago" (robot-only). None is on the embedded roster (OD-A11, OD-A17). |
| `cost` | `CostTracker` (#1769) |
| `pending:<plan or PR>` | a site this plan places elsewhere (A-post, #1361 S3–S5). It must name where it moves, and it leaves the allowlist in that PR. |

Off the embedded path entirely, and out of scope: `mesh/*`, `utils/web_cache.py`, `utils/output_watcher.py` and
`runtime/worker_pool.py`. A world-time stamp on a mesh message to another machine would be a wire bug.

## Site families and PRs

Sizes: **S** is under about 150 changed lines, **M** is about 150–600. Each PR lands its red gates first as
`xfail(strict=True)` and flips them in place; a gate is never removed or re-pointed. Each PR wires its own consumers
from the builders, so no PR ships a seam without its caller. Each `src/` PR carries a `changelog.d/` fragment, runs
`scripts/run_mypy_public.sh` and `scripts/prepr.sh`, and gets its own three-lens code review.

| PR | Family | What | Size |
|---|---|---|---|
| **0** | census + gates | `scripts/world_clock_census.py` (AST only; imports nothing from `maxim`) prints every time, calendar, stamp, id and randomness site in `src/maxim/`, its family, its stamp–reader pairs, and its placement (A-pre PR n, A-post, `pending`, or allowlist category), so the plan carries no volatile counts. Plus every red gate in §Gates, landed red, and the WC5 characterization guards, landed green. No `src/` change. | S |
| **1** | the clock | `time/agent_time.py` (both collaborators, host defaults, `PerceivedMoment`); the call-time-read test under `_StepClock` (I2); the motor-dispatch import ban (I3); **M72 built**, both scopes, as a per-file ratchet against `origin/main` (the guard note); #1769's fix | M |
| **2** | prompt and observation time (OD-A3) | `PromptBuilder.build_prompt` and `PromptBuilder._build_engage_prompt` render from the port (and `build_identity_section`'s unused `date_str`/`time_str` parameters go); `ContextPool._entries_to_text` and `ContextPool.get_context_text` render stamps through `perceive`; `retrieval/temporal_signal.py` resolves relative dates in perceived space, mapping candidate stamps through `perceive`, and resolves none under `None`. M72's prompt scope reaches zero outside the allowlist. Flips G-render and G-tz. | M |
| **3** | calendar phase (OD-A5, D-WC2) | `TemporalSignature.from_timestamp(ts, *, calendar=None)` and `TemporalSignature.now(*, calendar=None)`. **The census classifies every caller** as prompt-reaching or credit-anchoring. There are more than a dozen outside `time/` on `f145e27b`. Prompt-reaching: `StatisticianAgent._assess_with_temporal_context` (the `[TEMPORAL]` source), `exec_agent.py`'s phase dict, the `temporal_patterns` tool (`tools/introspection.py`, `session.py`), `simulation/introspection.py`, and `similarity/signature.py::SituationSignature.from_memory` if the census shows it reaches the prompt. Credit-anchoring: `similarity/encoder.py`, `bridges/tool_pain_bridge.py`, `bridges/escalation_bridge.py`, `memory/concept_extractor.py`, `integration/memory_hub.py`. Only the prompt-reaching callers pass the port's calendar. A credit-anchoring caller passes nothing, because its signature is compared with NAc's anchors, and OD-A20's decline keeps both sides on the host calendar. A caller doing both is split at the census, never guessed. **Mixed-calendar reads.** The statistician's anomaly score queries SCN state that was learned from credit-anchoring (host-calendar) signatures. When the port is not the host default, it emits nothing: it never compares a world-calendar query against host-calendar bins. Flips G-phase; OD-A20's gates stay red. | S–M |
| **4** | Hippocampus, attention and live-path stamps (OD-A5's window half, OD-A4's stamp half) | the 30 s de-duplication window; capture and access stamps with every paired reader (§A stamp moves…); `SalienceMap.update`, `SalienceMap.record_tracking_target`, `salience/salience_network.py`'s `now=` defaults, `attention/gaze_history.py`; `SpecSensor.read`'s `SensorReading` stamp; the `field(default_factory=time.time)` stamps an embedded turn constructs outside the fence (by census), passed explicitly with the defaults kept. Flips G-window and the A-pre half of G-long and G-pace. Re-sized after PR 0; split 4a/4b if the census says M+. | M |
| **5** | refractories (OD-A6, D-WC3) | `PainBus.publish` and `ReactionBus` dedup windows on `MachineryClock.monotonic_s`; I3's check extended to the reflex and motor paths. **After #1200**, so in 1.3.6, at the boundary. Flips G-refractory. | S |

## Placements (nothing dropped)

| Item | Where | Why there |
|---|---|---|
| **OD-A4's root fix**: a tool's duration from the host in world time, replacing `execute_and_learn`'s `exec_start`/`exec_elapsed` wall reads, which `_observe_energy_cost` turns into NEGATIVE NAc valence above 2.0 s | **A-post**, the first PR after the fence lifts on `runtime/tool_dispatch.py`, with B2 (#1150): the duration becomes a field of the settle receipt. B2's parity guard needs it first (candidates §A3). | the file is fenced. Its red gate, **G-tool-duration**, lands in PR 0 (equal host-supplied world duration, two wall durations either side of 2.0 s, same valence) and flips there. |
| **OD-I8**: `Embodiment.evaluate_failures` takes drive drift from `time.time()` | **#1361 S4**, already decided (Q4: drift only through `Embodiment.advance(dt)` on the experience clock; `_last_poll` deleted) | S4 deletes the read, so A-pre adds nothing. G4 is its gate. |
| `FailureMode.last_fired` (`embodiment/sem.py`), and the `PainSignal` stamps in `Embodiment._publish_drive_pain` / `_publish_pain` | A-post (OD-A8), with #1361 S4 | `body.py` is S4's file and `evaluate_failures` is pinned (touch-it-shrink-it) |
| **#1769** | PR 1 | decided above |
| Fenced ids and stamps: `agent_loop.py`, `orchestrator.py`, `tool_dispatch.py` (OD-A8) | A-post, 1.3.6, Session A | fence |
| NAc stamps and anchors (after #1361 S3b) | A-post | #1361 owns the windows |
| OD-A7: ids and RNG | after the 1.4.0 cut (D-WC1) | owner 2026-11-19 |
| G17: session ids | the post-fence resume stage | unchanged |
| `DefaultNetwork.step(now)` (OD-A9) | A-post | its own draft |
| Hippocampus memory ids (`uuid4` in `Hippocampus._build_capture_record`) and `llm_fallback`'s `fallback-{time.time_ns()}` request id | with OD-A7 | ids are not A-pre (D-WC1) |
| The oscillator's elapsed-time step and entrainment | tier B (`time_sense_DRAFT.md`) | it changes SCN dynamics under the host clock, so it can't ship with "robots unchanged" |
| `llm_fallback`'s canned time/date answers | **gone**: removed with the short-question shortcut (#1689) | nothing to move |

**#1305, checked as line 2 asks.** #1305 (`1305.owned-rng`, a 1.4.0 gate item, milestone 1.3.5) keeps the seeded-run
half: the process-global `random` / `np.random` draws. **OD-I13** belongs on #1305's side too: seeded runs need a seed
and sampler on the backends (`backend_protocol.py::complete_with_usage` takes none) as much as they need owned RNG.
This plan neither widens #1305 nor takes OD-A7. The OD-I13 note goes to #1305 as a comment when this plan's PR opens.

## Gates

Every gate is `xfail(strict=True)` from PR 0, in `tests/unit/test_world_clock_a_pre.py`, flipped in place by the PR
named. Each is written against today's surface wherever it can be, so it fails by assertion rather than by a missing
name. Where a gate needs a collaborator that does not exist yet, PR 0's description lists it. Each guard is proven by
deletion in the PR that flips it.

| Gate | Test | Flips in |
|---|---|---|
| **G-render** (OD-A2's planted-render test) | A sentinel with unusual seconds is planted **in the machinery clock**, never the port, with `TZ` pinned. It must not appear in any built prompt or tool result, in raw, clock (`%H:%M:%S`, `%I:%M %p`, `[HH:MM:SS]`) or date form. **End to end: origin offset only** (shifting the machinery origin changes no difference, so nothing rendered may change). **At render level, on a frozen store: offset and scale**, so relative renders can't leak it either. | 2 |
| **G-tz** (OD-WC6) | The same embedded prompt, built in two processes with `TZ` 13 h apart (`time.tzset()`, so POSIX only; the test skips loudly elsewhere and runs in CI on Linux), with the port declaring no calendar | 2 |
| **G-phase** | Under a world calendar on the port, the `[TEMPORAL]` source and `temporal_patterns` bin by the world calendar; with `None`, they emit no week/month/year phase; NAc's anchor similarity is byte-identical either way (D-WC2) | 3 |
| **G-window** | Two identical observations 31 wall seconds but 0 machinery seconds apart de-duplicate, and so do 0 wall / 31 machinery the other way round, under an **advancing** fake clock | 4 |
| **G-pace** (OD-WC2, A-pre scope) | A scripted host session over A-pre's families (Hippocampus observe/consolidate/sleep, context pool, salience) at 1× and 20× fake wall pace, with jitter: identical state digest, ids excluded (D-WC1) and the exclusion stated in the test | 4 |
| **G-long** (BD-14.2) | A long-horizon scripted run through consolidation, sleep and decay, with the wall clock patched to jump a huge, varying amount per read, two jump schedules, and background workers off (`build_memory_hub(start_background_workers=False)`). The digest must not change. The A-pre-scoped half flips in 4; the full-loop half flips in A-post (strict red until then). | 4 / A-post |
| **G-refractory** | Two distinct pain events 0.2 machinery seconds apart are both dispatched at 20× wall pace; a motor-dispatch read of the injected clock fails I3 | 5 |
| **G-tool-duration** | see §Placements | A-post |
| **G-cost-month** (#1769) | A tracker built on a wall clock in month M, recording with a caller `timestamp` of 0.0, keeps its month total | 1 |

**Exit tests chosen from OD-WC1 to OD-WC7:**
- **WC5, robots unchanged, is mandatory**, and lands green in PR 0 as a characterization guard:
  - the existing suite on host defaults;
  - the loop characterization golden still under `_StepClock`;
  - a pre-A-pre Hippocampus home and NAc pair loaded and saved byte-identically, with stamp units unchanged;
  - a prompt built under a pinned `TZ` and a fake wall clock, byte-identical to today's.
- **WC6** is G-tz; **WC2** is G-pace, scoped to A-pre's families; **WC4**, in its static form, is M72 at zero
  outside the allowlist on A-pre's families, plus G-long.
- **Not in A-pre:**
  - WC1, bit-identical across processes: ids are excluded by D-WC1;
  - WC3, pause, and WC7, step size: both are decay and drift, so #1361's.

  All three are listed so A-post picks them up.

## New invariants (each `[engineering]`, entering its brief in the PR that builds its guard)

- **I1. Prompt-reaching modules render time only through the perceived-time port.** A machinery stamp is rendered
  through `perceive`, never `time.localtime`. Regression guard: M72's prompt scope + `test_world_clock_a_pre.py`
  (G-render, G-tz). Brief: runtime-tools.md.
- **I2. Host defaults read the clock at call time, through the module attribute.** Regression guard:
  `test_world_clock_a_pre.py::test_step_clock_reaches_host_defaults`. Brief: bio-memory.md (`time/`).
- **I3. Motor dispatch never reads an injected clock.** Regression guard: an AST import ban over `motion/`,
  `hardware/controller.py` and `ReachyMiniController`, proven by planting an import. Brief: embodiment.md §1.
- **I4. A stamp moves with every reader that subtracts from it.** Regression guard: G-long + the census's pair list
  (PR 0). Brief: bio-memory.md.
- **I5. Cost accounting runs on the wall clock.** Regression guard: G-cost-month. Brief: llm-routing.md.

**Rules with no check yet:** M72's prompt scope is a hand-kept module list, so a new module that renders time
escapes it. That is mechanization backlog row **M99**, added in this PR: derive the scope from the prompt's producers
(the callers of `agents/prompt_budgeter.py::PromptBudgeter`'s section adds, plus the tool-result builders). M72's own row is amended in this PR (D-WC4).

## Impact

**Ledger rows, checked by id.** Under the host defaults every moved read returns today's value, and WC5 proves it,
so no row walks. The rows whose mechanisms these families touch:
- **T1-2** (prompt time, LLM-AUT);
- **T1-4** (PainBus, PR 5);
- **T1-6** and **T1-9** (NAc paths; no NAc read moves in A-pre);
- **T1-13**, **T1-14** and **T1-15** (PainBus, survival rigs);
- **T1-16** (Hippocampus window and recall, PR 4; `temporal_signal`, PR 2).

Each PR's description re-states this for its own family. A PR that cannot show WC5 green walks the row instead.

**Overlaps:**
- **#1361:** no S-slice file is touched. `body.py` stays S4's.
- **#1200:** PR 5 waits for it.
- **Line 4 and Session A:** none (no fenced file).
- **CI cost:** PRs are scoped to the census, never swept repo-wide.

**Process:**
- **To Overdrive.** When this plan merges, the OD-A ids it takes (§What A-pre is) and the ids it defers (§Placements)
  go to the Overdrive session that owns `docs/maxim_upstream` (Overdrive step 3, `phase3/upstream-sync`, O77). If
  none is running, they go to the owner, to route through the roadmap's Phase 3 rows.
- **Owner questions found while building** go to the owner at the start of the PR that finds them.

## Exit (1.3.5 line 2)

This plan has passed its three-lens review, and PR 0 has merged. PRs 1–4 ride 1.3.5 if green before the cut;
otherwise they follow. PR 5 is 1.3.6.
