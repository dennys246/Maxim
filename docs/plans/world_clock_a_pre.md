# World clock, A-pre: a split clock and a perceived-time port, outside the fence

> **ACCEPTED 2026-11-24 (#1787); PR 0 corrected two of its premises (§PR 0's corrections).** 1.3.5 line 2 ([release_1_3_5.md](release_1_3_5.md), owner decision D9). No
> `src/` change in this PR. Every mechanism enters as `[engineering]`. The owner decisions it rests on are in
> §Decisions, and nothing in it is decided by wording.
> **Review:** a three-lens plan review (Executor, Architecture, Wire integrity) ran on `1e1eed8d`. Its findings are
> folded here. Three of them changed owner decisions, which the owner re-took on 2026-11-24 (D-WC2 revised, D-WC5 to
> D-WC7).
> **Input:** [overdrive_refactor_candidates.md](overdrive_refactor_candidates.md) §A3, and Overdrive's position. Read
> with `git show` at Overdrive `origin/main` `9623be7`:
> - `docs/roadmap.md`: Phase 3 tier A (OD-A1 to OD-A25), OD-H6, OD-I8 and OD-I13;
> - `docs/design/reviews/maxim_response_2/response.md`;
> - its consumer contracts: `docs/design/npcs_and_sem.md` §3 and `docs/design/realtime_lanes_coop.md` row 10.
>
> Overdrive's `docs/maxim_upstream/world_clock_release_DRAFT.md` (`0748fee`, 2026-11-16) is background only. Where it
> differs from the roadmap rows, the rows win.
> **Rides on:** [experience_clock_decay.md](experience_clock_decay.md) (#1361). This plan adds no second experience
> clock and owns no decay, eligibility or temporal-credit window.
> **Guard approach note:** [world_clock_a_pre_guard_note.md](world_clock_a_pre_guard_note.md).
> Every `file::symbol` here was checked on Maxim `origin/main` `f145e27b` (2026-11-24). Lists of callers are
> illustrative. PR 0's census is authoritative.

## What A-pre is

**The claim.** Time reaches an embedded agent from two injected sources:
- a **machinery clock**, for windows;
- a **perceived-time port**, the only time a prompt or tool result may contain.

Their host defaults read exactly what today's code reads, so robots, the console and pulse are unchanged.

**A-pre's scope is three families:**
- prompt and observation time;
- **closed** windows, meaning those whose stamp and reader live in one place and are never persisted;
- the guard.

It touches no **hot or fenced file**. The release plan names six (release_1_3_5.md §Why the plan changed):
`runtime/agent_loop.py`, `agents/exec_agent.py`, `cli.py`, `simulation/orchestrator.py`, `runtime/tool_dispatch.py`
and `models/language/router.py`. To those it adds `decisions/nac.py`'s selection path. `runtime/loop_setup.py` is
loop-slice territory, so A-pre doesn't edit it either.

**Done means:**
- under the "no calendar" port, no host time and no host time zone reaches a built prompt or tool result (G-render,
  G-tz);
- the closed windows count the machinery clock (G-window);
- the M72 lint passes on A-pre's families;
- WC5, robots unchanged, holds.

1.3.5 needs only this plan reviewed and PR 0 merged (§Exit). The other PRs merge when green (merge order step 5).

**Overdrive ids:**
- **Taken:** OD-A2 and OD-A3; the SCN-render and window parts of OD-A5.
- **Planned for 1.3.6, at the A-pre/A-post boundary:** OD-A6.
- **Moved out of A-pre against Overdrive's A-pre label, and reported back:**
  - OD-A4's stamp half and OD-A5's capture/access and salience halves (D-WC6);
  - OD-A7 (D-WC1).

§Placements lists the rest.

## Motivation and front gate

**Today one wall clock serves three jobs.**
1. **What the agent is told:**
   - `PromptBuilder.build_prompt` and `PromptBuilder._build_engage_prompt` render `datetime.now()`;
   - `ContextPool._entries_to_text` and `ContextPool.get_context_text` stamp each observation `[HH:MM:SS]` through
     `time.localtime`;
   - SCN-derived text reaches the model: the `[TEMPORAL]` tags (`StatisticianAgent._assess_with_temporal_context`, and the oscillator-coherence line in `StatisticianAgent.summarize_for_context`),
     the `temporal_patterns` tool (`tools/introspection.py::TemporalPatternsTool`), introspection's `current_phase`
     (`simulation/introspection.py`) and the relative-date resolution in `retrieval/temporal_signal.py`.
2. **What the machinery counts.** In A-pre's scope: the Hippocampus's 30 s de-duplication window
   (`Hippocampus.store_observation`, `dedup_window_s`) and the dedup refractories.
3. **What the process watches:** heartbeats, timeouts, and motor and IO limits.

An embedded agent, one whose host steps it in world time (Overdrive tier A, O6), needs (1) and (2) to come from the
host and (3) to stay on the wall. Overdrive's contract is stricter than "from the host": **"No time field reaches
cognition"** (npcs_and_sem.md §3.1), with Overdrive declaring no calendar. Maxim gains too:
- a prompt that carries no host time can be replayed with one variable changed;
- today the same prompt built under two time zones comes out differently.

**Front gate: does this need its own mechanism?** Only the port does.
- **The machinery clock rides on existing reads.** Its host default is the same `time.time()` / `time.monotonic()`
  read each site makes today, read through the `time` module attribute at call time.
- **It diverges from OD-A2's wording on purpose.** OD-A2 and candidates §A3 say "for robots the machinery clock is
  Maxim's `ExperienceClockDriver`". That holds for decay and cadences, which are #1361's and A-post. It does not hold
  for A-pre's windows. The driver counts experience µs with pauses excluded and turns quantised (5 s per text-sim
  turn), so moving a wall window onto it would change robot behaviour. In A-pre the machinery clock is therefore the
  window source, defaulting to the wall.
- **One source under embedding.** Under embedding the host feeds both the machinery clock and the driver's world time.
  Nothing in A-pre makes them agree. The embedded entry point (OD-A10) must derive both from one host world-time
  source (§Placements), so the plan does not create a second clock in disguise.
- **The port is new**, and the reason is specific. `ExperienceClockDriver` separates experience time from wall time,
  but it has no notion of "time the agent is told". Every prompt site calls the wall clock directly. The port is a
  small protocol in the existing `time/` package: no bus, no thread, no persisted state.
- **No new lint.** The guard extends M72 (#1361's planned lint, still unbuilt) rather than adding a second one.

## Decisions

**Owner, 2026-11-24, asked together at the start:**
- **D-WC1. The scope is narrow, and G17 stands.** A-pre is OD-A2 to OD-A6 at most. It adds no `Rng` and no `IdSource`.
  - OD-A7 is a new release-gate item **after the 1.4.0 cut**, not a widened `1305.owned-rng` (owner decision
    2026-11-19). It covers `similarity/semantic.py::_initialize`'s `or 42`, `seed_all` off the embedded path, and
    `NAc.record_event`'s `time.time_ns()` ids.
  - Session ids wait for the post-fence resume stage (DECISIONS.md G17).
- **D-WC3. OD-A6: the dedup refractories count the machinery clock, after #1200.** Motor and IO rate limits stay on
  the wall, and a static check refuses a motor-dispatch path that reads the injected clock (I3).
  experience_clock_decay.md §(g) is amended in this PR.
- **D-WC4. The guard is an extended M72 plus a long-horizon run.** M72 becomes one AST lint with two scopes,
  machinery and prompt, and Overdrive's BD-14.2 long-horizon run becomes a strict red gate. The draft's
  `sys.setprofile` tripwire is dropped. The M72 row and §(g) are amended in this PR.

**Owner, 2026-11-24, re-taken after the plan review:**
- **D-WC2 (revised). In A-pre the port has two modes, host or none.**
  - The original D-WC2 ("the port's calendar for prompt-reaching producers; NAc keeps the host calendar") could not
    be built as worded. The prompt-facing consumers build no signatures of their own: they query **the one SCN
    index**, whose bins the credit producers register on the host calendar.
  - So the port is either the host default (today, unchanged) or **none**, which is what Overdrive declares. Under
    none, every calendar or SCN-derived render emits nothing.
  - The SCN stays one host-calendar index for credit, and OD-A20's decline holds.
  - A world-calendar port is deferred to tier B / A-post and recorded as a gap (§What Overdrive still does not get).
- **D-WC5. #1769: the whole `CostTracker` runs on the wall.** That means the month key, the hourly and daily rolling
  windows and the EMAs. `record(timestamp=)` is deprecated and ignored for accounting, with a `DeprecationWarning`.
  Spend is real money, and its windows are the bill's. This lands in PR 1.
- **D-WC6. Closed pairs only.**
  - **What A-pre moves:** only windows whose stamp and reader live in one place: the Hippocampus de-duplication
    window, which is in memory and never persisted, and the refractories (1.3.6).
  - **Capture/access stamps and `SensorReading` stamps move later**, as whole reader sets. Their readers span the
    memory agent's echo filter, the salience/planning/spatial bridges, SCN registration, `SituationSignature`'s
    persisted EC hash and retrieval's `time_diff`. They also persist with no clock domain. They move with a
    clock-domain marker in the home, alongside AgentHome (OD-A12) / A-post.
  - **Salience and gaze** are fed by DefaultNetwork, so they move with OD-A9.
- **D-WC7. Hot files: a default now, required at A-post.**
  - The port reaches `PromptBuilder` through `agents/llm_worker.py::LLMWorker` (not a hot file) as a keyword-only
    argument with the host default. No hot file is edited.
  - `exec_agent.py`'s SCN phase dict moves to A-post.
  - The embedded entry point (OD-A10) takes both collaborators as **required** arguments, and a test proves an
    embedded agent carries the non-default.
  - The flip to required on every constructor is placed in A-post, when the fence lifts.

**Settled earlier, cited rather than re-asked:**
- **Draft §8 Q1, is embedding supported?** Yes: the candidates doc takes tier A on engineering grounds, and A4's front
  gate passed (owner, 2026-11-19).
- **Draft §8 Q2, its own patch number?** No: D9 puts A-pre in 1.3.5.
- **Draft §8 Q3** is D-WC1, and **Q4** is D-WC4.
- **OD-H6:** Overdrive withdrew the loop-interval argument. Under a host-called `think`, `llm_submit_interval` needs no
  change for embedders, and it lands inside cand. B1 (OD-A15).
- **OD-A20:** declined 2026-11-19.

## The design

### Two collaborators in `time/agent_time.py`

| Collaborator | Surface | Host default | Embedded (A-pre) |
|---|---|---|---|
| `MachineryClock` | `now_s() -> float`, `monotonic_s() -> float` | `time.time()` / `time.monotonic()`, read **through the `time` module attribute at call time** | the host's world seconds |
| `PerceivedTime` (the port) | `now() -> datetime \| None`; `perceive(stamp_s: float) -> datetime \| None` | `datetime.now()` / `datetime.fromtimestamp(stamp_s)`, local, as today | `NO_CALENDAR`: both return `None` |

- **Returning `datetime`, not a new moment type.** Every render site keeps calling `strftime` on what it gets, so
  locale-dependent `%A`/`%B` output stays byte-identical. There is no hand-rolled formatter.
- **`None` means "omit".** A render site that gets `None` renders nothing: no datetime section, no `[HH:MM:SS]`, no
  relative date, no `[TEMPORAL]` tag, no SCN phase, no rhythmic pattern.
- **SCN-derived renders ask the port, not the SCN.** A consumer that queries the SCN to render text first asks the port
  whether a calendar is declared. Under `NO_CALENDAR` it emits nothing. Under the host default it is unchanged.
  - The rule is keyed on what the port *declares* (`port.now() is None`), never on an identity check against the host
    default.
  - SCN registration is untouched: every registrar keeps its host-calendar input.
- **No ambient state.** Collaborators are passed explicitly, with defaults kept. There is no module global and no
  `ContextVar`: `proprioception/pain_bus.py`'s docstring rejects that back channel, and an ambient read that falls
  back to the wall when unset is a silent fallback.
  - **The construction chain:**
    - the port: `LLMWorker` → `PromptBuilder`;
    - `ContextPool`'s render, from its constructor, built in `runtime/loop_setup.py::build_loop_run`. That builder
      keeps the default in A-pre (D-WC7);
    - the SCN-render consumers, from their constructors;
    - the machinery clock, through `Hippocampus.__init__`, whose 8 production construction sites keep the default.
  - `runtime/bio_stack.py::build_bio_stack` (pinned at 475 lines) is not edited in A-pre.
- **Callers, honestly.** Production callers run the host default through the new seam, so robots are the caller from
  the first PR. The non-default caller is the embedded entry point (OD-A10), which is outside this plan. Until then
  the embedded behaviour is capability: each PR's fragment says so (a `.added`, never a `.fixed`, except for #1769,
  which has a real caller in `router.py`), and names OD-A10 as the caller that makes it a fix. G-long's full-loop half
  stays strict red as the visible marker.

### A stamp moves with every reader that subtracts from it (I4)

If a stamp moves while one of its readers still subtracts `time.time()`, the embedded result is garbage: world epoch
minus wall epoch. Under the host default both reads are the same, so nothing shows. This is why D-WC6 keeps A-pre to
closed pairs.

The census lists every **stamp–reader pair**, classified by the reader's *input* clock as well as where its output
goes. It covers readers of `EpisodicMemory.timestamp`, `Percept.timestamp`, `SensorReading.timestamp` and
`PainSignal.timestamp` across `src/`. A PR moves a whole pair or none of it.

A-pre's closed pairs:
- the Hippocampus window: `_recent_observations` is written and read only inside `store_observation`, under its own
  lock, and never persisted;
- in 1.3.6, each bus's refractory map.

### What stays on the wall, on purpose

The M72 allowlist is experience_clock_decay.md §(g)'s table (amended here, D-WC3 and D-WC4). It is keyed by
`file::function`, and each entry has one category:

| Category | Examples |
|---|---|
| `process` | heartbeat, stall detector, LLM timeouts, TTFT keepalive, the driver's stall cap |
| `scheduling` | `idle_sleep_s`, `target_period`, `llm_submit_interval` (OD-H6), the 120 s `_submitted_recently` window |
| `motor-io` | motor and IO rate limits; `PainDetector`'s cooldown for motion pain types (`MOVEMENT_FAILURE`, `EXCESSIVE_VELOCITY`, `DIRECTION_THRASHING`, `EXCESSIVE_ACCELERATION`) |
| `real-world-io` | `HttpFetchTool.execute`'s `accessed_at`, `exec_agent.py::_build_llm_context`'s workspace mtimes, `FocusOnSoundTool.execute`'s "faded …s ago" (robot-only). None is on the embedded roster (OD-A11, OD-A17). |
| `cost` | `CostTracker` (D-WC5) |
| `pending:<where>` | a site this plan places elsewhere. It names where it moves, and it leaves the allowlist in that PR. |

Out of scope, because they are off the embedded path: `mesh/*`, `utils/web_cache.py`, `utils/output_watcher.py` and
`runtime/worker_pool.py`.

## Site families and PRs

Sizes: **S** is under about 150 changed lines, **M** about 150–600.
- **Gates:** each red gate lands `xfail(strict=True, raises=AssertionError)` in the PR named in §Gates, before the PR
  that fixes it, and is flipped in place, never removed or re-pointed. **Pre-surface form:** a gate whose collaborator
  keyword doesn't exist yet opens with an assertion that it does (`inspect.signature(...).parameters`), so before
  that keyword lands the gate fails by `AssertionError`, never by `TypeError`.
- **Each `src/` PR** carries a `changelog.d/` fragment, runs `scripts/run_mypy_public.sh` and `scripts/prepr.sh`, and
  gets its own three-lens code review.

| PR | Family | What | Size |
|---|---|---|---|
| **0** | census + today's gates | `scripts/world_clock_census.py` (AST only; imports nothing from `maxim`). It prints every time, calendar, stamp, id and randomness site in `src/maxim/`, the stamp–reader pairs, each `TemporalSignature` and SCN caller with its input clock and output use, and each site's placement. Plus the gate writable against today's surface (G-cost-month) and WC5's characterization guards, landed green. G-tool-duration moved out: see §PR 0's corrections. No `src/` change. | S |
| **1** | the clock | `time/agent_time.py`: both collaborators, `NO_CALENDAR` and the host defaults. Also: I2's call-time test; I3's motor-dispatch ban; **M72 built** (two scopes and the alias resolver from `scripts/_clock_refs.py`, shared with the census; a ratchet migrating to the absolute form; see the guard note); #1769 (D-WC5). Before M72 goes live, every in-scope site the census prints `UNPLACED` (88 sites at PR 0; stamp readers and SCN callers can print `UNPLACED` too) gets an allowlist entry, a `pending:` pin or a move. G-render, G-tz, G-scn and G-window land red here, because their surface is born here. | M |
| **2** | prompt and observation time (OD-A3) | `PromptBuilder.build_prompt` and `PromptBuilder._build_engage_prompt` render from the port, threaded through `LLMWorker`; `build_identity_section`'s unused `date_str`/`time_str` parameters go; `ContextPool._entries_to_text` and `ContextPool.get_context_text` render stamps through `perceive`, and omit them under `NO_CALENDAR`. Two tool results that hand the model a raw epoch are omitted under `NO_CALENDAR` too: `tools/introspection.py::_format_episodic_memory`'s `timestamp` (used by `memory_recall`) and `SceneSummaryTool.execute`'s `timestamp`. The render changes; the stamp does not (D-WC6). Flips G-render and G-tz. | M |
| **3** | SCN-derived renders (OD-A5, D-WC2 revised) | both `[TEMPORAL]` sources (`StatisticianAgent._assess_with_temporal_context` and `StatisticianAgent.summarize_for_context`), `TemporalPatternsTool`, `simulation/introspection.py`'s `current_phase`, and `retrieval/temporal_signal.py`'s relative-date resolution each emit or resolve nothing under `NO_CALENDAR`. Registration and NAc are unchanged. Flips G-scn. | S |
| **4** | the Hippocampus window (OD-A5's window half) | `Hippocampus.__init__(…, machinery_clock=HOST)`; `store_observation`'s window reads `now_s()`, the same `time.time()` it reads today. Flips G-window and G-long's A-pre half. | S |
| **5** | refractories (OD-A6, D-WC3), **1.3.6, after #1200** | `PainBus.publish` and `ReactionBus.publish` windows on `MachineryClock.monotonic_s`. Their `get(key, 0.0)` sentinel becomes `None`: under a clock whose origin is near 0, a 0.0 sentinel drops the first event. **`PainDetector`'s pair is `_emit_pain` (the writer: `_last_pain_time[signal.pain_type] = signal.timestamp`) and `PainDetector.record_tool_error` (the reader, an inline `now - _last_pain_time.get(pain_type, 0)`).** The writer picks its clock by pain type: tool types move to the machinery clock, and motion types (read through `_can_emit_pain`) stay `motor-io` on the wall. The reader's `0` sentinel becomes `None` too. Nothing outside `pain.py` reads `_last_pain_time`. I3 is extended to the reflex paths. Flips G-refractory. | S–M |

## Placements (nothing dropped)

| Item | Where | Why there |
|---|---|---|
| **OD-A4's root fix**: a tool's duration from the host in world time. `execute_and_learn`'s `exec_start`/`exec_elapsed` wall reads reach only its logs today: no caller passes `elapsed_s` to `record_outcome`, so `_observe_energy_cost` (NEGATIVE NAc valence above 2.0 s) has never run (#1788) | **A-post**: the first PR after the fence lifts on `runtime/tool_dispatch.py`, with B2 (#1150). The duration becomes a field of the settle receipt and is wired into `record_outcome` there. B2's parity guard needs it (candidates §A3). | fenced file. PR 0 pins today's no-booking green (WC5). **G-tool-duration** lands red in the wiring PR: two runs doing the same work, the wall jumping either side of 2.0 s at an equal host world duration, book the same valence. |
| OD-A4's stamp half: Hippocampus capture/access stamps (`Hippocampus._build_capture_record`, `get`, `_touch_internal`, `consolidated_at`), `SpecSensor.read`'s `SensorReading` stamp, the `field(default_factory=time.time)` stamps on the embedded path | **with AgentHome (OD-A12) / A-post**, each as a whole reader set, with a **clock-domain marker** written into the home only when non-default, so host-default files stay byte-identical. A non-default home also **bumps the envelope `schema_version`** (persistence-config.md §3), so a build from before the marker refuses it (`memory/snapshot.py` refuses a newer `schema_version`; `check_format_version` only warns), and a loader refuses a mismatched domain | D-WC6. Without the marker, a home written under injection and loaded under the host default makes sleep and eviction (`ConsolidationMixin.sleep`, `_evict_one`) read every trace as ~54 years old. Moving the forgetting readers is also memory-strength's call (#1361 Q5), asked there. |
| `SituationSignature.from_memory`'s own calendar binning (a second copy of `TemporalSignature`'s), persisted as the EC `temporal_hash` | with the stamp half | it bins capture stamps, so it moves with them, and the duplication is collapsed then |
| SCN registration of machinery stamps (`integration/memory_hub.py`'s `MemoryHub._on_memory_captured_scn`, `memory/concept_extractor.py`, `time/scn.py`) | with the stamp half | today they bin wall stamps, consistently. They would only mix clocks once the stamps move. |
| Salience and gaze (`SalienceMap`, `salience/salience_network.py`, `attention/gaze_history.py`) | **A-post with OD-A9** (`DefaultNetwork.step(now)`) | their only feeder, `default_network/network.py::_process_tick`, passes a wall `now`, and DefaultNetwork has its own wall readers |
| `exec_agent.py`'s SCN phase dict and anomaly score | A-post (hot file, D-WC7) | it emits under the host default until then. Under `NO_CALENDAR` it is the last SCN-derived render, listed as `pending:A-post`. |
| The flip to a required collaborator on every constructor | A-post (D-WC7) | fence |
| `CostTracker.get_totals(now=)` and `get_per_provider_totals(now=)` | PR 1, with D-WC5: deprecated and ignored like `record(timestamp=)`, so no caller clock reaches window pruning | the same root cause as #1769 |
| `router.py` stops passing `timestamp=` to `CostTracker.record` | A-post (hot file) | until then every call warns, hidden by pytest's `ignore::DeprecationWarning`; the deprecated keyword has this one production caller |
| **One world-time source** for the embedded machinery clock and the driver's injection point (candidates §A3's "the host's injection point into `ExperienceClockDriver`") | **OD-A10's entry point** / A-post | it constructs both from one host source, so they cannot disagree |
| **OD-I8**: `Embodiment.evaluate_failures` takes drive drift from `time.time()` | **#1361 S4**, already decided (Q4: drift only through `Embodiment.advance(dt)`) | S4 deletes the read, and G4 is its gate |
| `FailureMode.last_fired`; the `PainSignal` stamps in `Embodiment._publish_drive_pain` / `_publish_pain`; `body.py`'s `TemporalSignature.now()` | A-post (OD-A8), with #1361 S4 | `body.py` is S4's file |
| Fenced ids and stamps: `agent_loop.py`, `orchestrator.py`, `tool_dispatch.py` (OD-A8) | A-post, 1.3.6, Session A | fence |
| NAc stamps and anchors | A-post, after #1361 S3b | #1361 owns the windows |
| OD-A7: ids and RNG; Hippocampus `uuid4` memory ids; `llm_fallback`'s `fallback-{time.time_ns()}` | after the 1.4.0 cut (D-WC1) | owner, 2026-11-19 |
| G17: session ids | the post-fence resume stage | unchanged |
| The oscillator's elapsed-time step and entrainment; the world's day length | tier B (`time_sense_DRAFT.md`) | it changes SCN dynamics under the host clock |
| `llm_fallback`'s canned time/date answers | **gone**: removed with the short-question shortcut (#1689) | — |

**Open for the stamp half's design** (asked by Overdrive's upstream lane after the merge, #1787; not decided here):
1. **Can the clock-domain marker and the `schema_version` bump land without OD-A12's store routing,** on the one home
   an agent already has? Nothing technical ties them together, but each subset still moves its whole reader set
   (D-WC6). Which lands first is the owner's placement call.
2. **Which calendar does `SituationSignature.from_memory`'s `temporal_hash` bin on once the stamps move?** OD-A20's
   decline covers SCN registration and NAc anchor similarity, not this. Binning world-time stamps with host-local
   `datetime.fromtimestamp` mixes clocks, and `NO_CALENDAR` leaves nothing to bin on. The hash is persisted (EC), so a
   re-bin needs a migration or a `hash_scheme`-style stamp.

**#1305, checked as line 2 asks.** #1305 (`1305.owned-rng`, a 1.4.0 gate item, milestone 1.3.5) keeps the seeded-run
half: the process-global `random` / `np.random` draws. **OD-I13** belongs on #1305's side too: seeded runs need a seed
and a sampler on the backends (`backend_protocol.py::complete_with_usage` takes neither) as much as they need owned RNG.
This plan neither widens #1305 nor takes OD-A7. The OD-I13 note is on #1305 (with Overdrive's correction narrowing it).

## What Overdrive still does not get from A-pre

These are gaps, stated so the upstream lane can track them. None is hidden by wording.
- **No world calendar** (D-WC2 revised). That leaves out `realtime_lanes_coop.md` row 10 (one world clock for every
  `TemporalSignature` producer, phase from world time and the world's day length) and npcs_and_sem.md §3.2's interim
  coarse readout. Under `NO_CALENDAR` an embedded agent renders no calendar at all.
- **The SCN stays on host time internally,** for credit (OD-A20 declined). npcs_and_sem.md §3.2 says NPCs must not run
  against the world clock with their SCN on host time. That stays a known divergence. Its gates are OD-A20's two red
  gates.
- **Stamps stay wall** until the stamp half lands (D-WC6). Overdrive's "`SensorReading.timestamp` is body machinery,
  never rendered verbatim" (§3.1) holds in A-pre, because under `NO_CALENDAR` no stamp renders.
- **Ids, randomness and tool duration** follow D-WC1 and §Placements.

## Gates

All gates are in `tests/unit/test_world_clock_a_pre.py`, each `xfail(strict=True, raises=AssertionError)`, so an
`ImportError` or `NameError` is never counted as the expected failure. Each is proven by deletion in the PR that flips
it. Tests that write state use a scratch `HOME` and a tmp `state_path`.

| Gate | Lands → flips | Test |
|---|---|---|
| **G-render** (OD-A2's planted-render test) | 1 → 2 | **Under `NO_CALENDAR`:** a sentinel with unusual seconds is planted **in the machinery clock and in the patched wall clock**, with `TZ` pinned. It must not appear in any built prompt or tool result, in raw, clock (`%H:%M:%S`, `%I:%M %p`, `[HH:MM:SS]`) or date form. End to end: origin offset only, on a `ContextPool(port=NO_CALENDAR)` and an `LLMWorker(port=NO_CALENDAR)` built **directly**, not through `runtime/loop_setup.py::build_loop_run`, which keeps the default in A-pre (D-WC7). At render level, on a frozen store: offset and scale. The patch target is named per PR (today `maxim.agents.prompt_builder.datetime`; from PR 2, the port). |
| **G-tz** (OD-WC6) | 1 → 2 | the same `NO_CALENDAR` prompt built in two subprocesses (each with a scratch `HOME`) with `TZ` 13 h apart, using `time.tzset()`, so POSIX only. It skips **loudly** elsewhere and runs in CI on Linux. The prompts must be byte-identical. |
| **G-scn** | 1 → 3 | under `NO_CALENDAR`: no `[TEMPORAL]` tag, no `temporal_patterns` bins, no `current_phase`, no relative-date resolution. NAc anchor similarity and the SCN bins must be byte-identical to the host-default run (D-WC2 revised). |
| **G-window** | 1 → 4 | under an **advancing** fake clock, two identical observations **0 machinery seconds and 31 wall seconds apart de-duplicate**. **31 machinery seconds and 0 wall seconds apart do not.** |
| **G-pace** (OD-WC2, A-pre scope) | 1 → 4 | a scripted session over A-pre's families, at 1× and 20× fake wall pace with jitter. The state digest must be identical. Ids are excluded (D-WC1), and the test says so. |
| **G-long** (BD-14.2) | 1 → 4 (A-pre half) / A-post (full loop) | a long-horizon scripted run through observe, consolidate, sleep and decay. **`time.time` only** is patched, to jump a huge and varying amount per read, and two jump schedules are compared. `monotonic` jumps would trip `Deadline` / `wait_session_ended`. Built without ATL, so `_restore_on_open` cannot revive the ConceptExtractor, and with `integration/memory_hub.py::build_memory_hub(start_background_workers=False)`. The digest scope is named: A-pre's families only (the window). NAc, ATL, SCN and stamps are excluded until their PRs. |
| **G-refractory** | 5 → 5 | two distinct pain events with **the same `(entity, failure_mode)`**, 0.6 machinery seconds apart, are both dispatched at 20× wall pace, on each bus and for a `PainDetector` tool-pain type; the first event under a clock whose origin is at 0 is not dropped, on each bus and in `record_tool_error`; and a motor-dispatch read of the injected clock fails I3 |
| **G-tool-duration** | A-post → A-post (#1788) | see §Placements. Moved out of PR 0: today it could only pass vacuously |
| **G-cost-month** (#1769) | 0 → 1 | a tracker on the wall clock, recording with a caller `timestamp` of 0.0, keeps its month, hourly and daily totals and its EMAs; a `get_totals(now=0.0)` read reports the same totals as a read with no `now` |

### PR 0's corrections (owner decisions 2026-11-24)

PR 0's precondition tests found two premises of this plan false on main:
- **The tool-duration wall read reaches no learning.** OD-A4 assumed `execute_and_learn`'s duration feeds
  `_observe_energy_cost`. No caller passes `record_outcome`'s `elapsed_s`, so robots book no energy cost, and every
  `tool_result` link learns a 0.0 s delay (persisted `temporal_delta`, read by `NAc.predict_delay`): `elapsed_s`
  has both sinks. G-tool-duration would pass vacuously. PR 0 pins both green. The wiring, which also changes NAc's
  delay predictions, and G-tool-duration move to A-post with B2 (#1788). #1788 is `silent-wrong`. Its owner exception, needed past
  2026-12-01 because the fix lands later, is committed in [silent_wrong_exceptions.json](silent_wrong_exceptions.json).
- **A saved store is not byte-stable across processes.** The Hippocampus `context_index`, NAc's `outcome_index` and
  the EC's inverted indices (`similarity/indices.py`) are written from a `set`, so their order follows
  `PYTHONHASHSEED`. A separate small `src/` PR (#1789) makes every such writer sort its output, with a two-process
  guard; WC5's re-saved file is then the old store with that one index sorted, compared byte for byte.

**Exit tests chosen from OD-WC1 to OD-WC7:**
- **WC5, robots unchanged, is mandatory.** It lands green in PR 0 as characterization:
  - the existing suite on host defaults;
  - the loop characterization golden still under `tests/unit/_loop_harness.py::_StepClock`;
  - a Hippocampus home and NAc pair saved before A-pre (`tests/fixtures/world_clock_wc5/`), loaded and saved back
    unchanged. Each writer serialises one id index from a `set` (the Hippocampus `context_index`, NAc's
    `outcome_index`), whose order follows `PYTHONHASHSEED` in a store saved before #1789. Re-saving sorts that one index
    (#1789), so the re-saved file must equal the original with that index sorted, exact for every byte;
  - the shipped dispatch seam handing NAc no tool duration: no energy cost, a 0.0 s `tool_result` delay (#1788);
  - a prompt built under a pinned `TZ` with `maxim.agents.prompt_builder.datetime` patched, byte-identical to
    today's. From PR 2 it is patched at the port, with the patch target named per PR.
- **WC6** is G-tz. **WC2** is G-pace. **WC4**, in its static form, is M72 at zero outside the allowlist on A-pre's
  families, with an absolute per-file pin once a family reaches zero, plus G-long.
- **Not in A-pre:** WC1 (ids, by D-WC1), WC3 (pause) and WC7 (step size), the latter two being #1361's decay and drift.
  A-post picks them up.

## New invariants (each `[engineering]`, entering its brief in the PR that builds its guard)

- **I1. Prompt-reaching modules render time only through the perceived-time port.** That includes SCN-derived text,
  and a stamp is rendered through `perceive`, never `time.localtime`.
  - Regression guard: M72's prompt scope + `test_world_clock_a_pre.py` (G-render, G-tz, G-scn).
  - Brief: runtime-tools.md.
- **I2. The machinery clock's host default reads at call time, through the `time` module attribute.**
  `_StepClock` patches only `time.*`, so this covers `MachineryClock` alone. The port's `datetime` default is
  patched by name, as WC5 does.
  - Regression guard: `test_world_clock_a_pre.py::test_step_clock_reaches_machinery_default`.
  - Brief: bio-memory.md (`time/`).
- **I3. Motor dispatch never reads an injected clock.**
  - Regression guard: an AST test over `motion/`, `hardware/controller.py` and `hardware/reachy/controller.py`. It
    refuses an import of `maxim.time.agent_time`, **and** a `MachineryClock` / `PerceivedTime` annotation or
    parameter, and is proven by planting each. Stated blind spot: an untyped object passed in.
  - Brief: embodiment.md §1.
- **I4. A stamp moves with every reader that subtracts from it.**
  - Regression guard: process invariant — mechanization backlog M100 (the census's `--check`). Until then, G-long and
    the census's pair list carry it.
  - Brief: bio-memory.md.
- **I5. Cost accounting runs on the wall clock, in every window.**
  - Regression guard: G-cost-month.
  - Brief: llm-routing.md.

**Rules with no check yet, and their mechanization backlog rows (added in this PR):**
- **M99:** M72's prompt scope is a hand-kept list. M99 derives it from the prompt's producers.
- **M100:** I4's pair check, `scripts/world_clock_census.py --check`. It fails when a stamp and a reader that
  subtracts from it sit on different placements.

M72's own row is amended (D-WC4).

## Impact

**Ledger rows, checked by id.** Under the host defaults every moved read returns today's value, and WC5 proves it, so
no row walks. These are the rows whose mechanisms A-pre's families touch:
- **T1-2:** prompt time, LLM-AUT;
- **T1-4:** PainBus, PR 5;
- **T1-13, T1-14 and T1-15:** PainBus, survival rigs;
- **T1-16:** the Hippocampus window, PR 4; `temporal_signal`, PR 3.

**T1-6** and **T1-9** are untouched: no NAc read moves. Each PR's description re-states this for its own family. A PR
that cannot show WC5 green walks the row instead.

**Overlaps:**
- **#1361:** no S-slice file is touched, and `body.py` stays S4's.
- **#1200:** PR 5 waits for it.
- **Line 4 and Session A:** none.
- **CI cost:** PRs are scoped to the census, never swept.

**Process:**
- When this plan merges, the OD-A ids it takes, moves and defers (§What A-pre is, §Placements, §What Overdrive still
  does not get) go to the Overdrive session that owns `docs/maxim_upstream`: Overdrive step 3, `phase3/upstream-sync`
  (O77). If none is running, they go to the owner.
- Owner questions found while building go to the owner at the start of the PR that finds them.

## Exit (1.3.5 line 2)

This plan has passed its three-lens review, and PR 0 has merged. PRs 1–4 ride 1.3.5 if green before the cut;
otherwise they follow. PR 5 is 1.3.6.
