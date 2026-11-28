# 1.3.5: what ships, what rides along, and the work toward Overdrive

**Status:** proposed 2026-11-22, the day 1.3.4 shipped. The owner took decisions D1–D13, #1601 and #1737 the same
day, as did M97. D14 and the third batch were decided later the same day. A three-lens review round (executor, architecture, wire integrity) and a delta round
ran the same day, and their findings are folded in. Numbers were read on `origin/main` b98954f7 (v1.3.4) and from `gh` on 2026-11-22. Re-check any number before
citing it later.

**What this plan replaces.** It replaces "1.3.5 = the orchestrator" in [roadmap_1_3_x.md](roadmap_1_3_x.md)
§1.3.5. Under the 2026-10-24 rule, labels name lines and numbers are given at the cut. The orchestrator keeps its
line, the head of the chain to 1.4.0, and takes whichever release number it is ready for. The next number, 1.3.5,
goes to the lines below.

## Why the plan changed

- **The planned mainline has nobody on it.** No `start_simulation_mode` slice has landed since 2026-10-30, and no
  branch exists for one. The function is 1,895 lines, 895 over gate item `1323.functions-1000`. Phone Phase 0 has no
  code either.
- **Finished work is waiting, on one machine only.** These are synced on v1.3.4, and each either has no PR or has a
  draft PR whose synced head is not pushed:
  - #1185 (`fix/1185-approval-intent-whole-word`);
  - #1519 (`fix/1519-frame-untrusted-wm-text`);
  - #1228/#1225 (`fix/1228-1225-aut-router-and-bio-flags`);
  - #1680 (`fix/1680-write-paper-kwarg`);
  - #1414 (`fix/goal-call-closure`, draft PR #1747);
  - atonia 10a and 10b (draft PRs #1603 and #1735);
  - loop slice 6 (`refactor/loop-slice6-intake`; merged as #1758 on 2026-11-22).
- **They collide in the hot files.** In this plan, **the hot files** are `runtime/agent_loop.py`,
  `agents/exec_agent.py`, `cli.py`, `simulation/orchestrator.py`, `runtime/tool_dispatch.py` and
  `models/language/router.py`. `git diff --name-only origin/main...<branch>` on 2026-11-22 shows:
  - #1185, 10a, 10b, #1228/#1225 and #1680 all edit `orchestrator.py`;
  - #1185 and #1519 edit `tool_dispatch.py`;
  - #1519, 10a and 10b edit `router.py`;
  - most of them edit `agent_loop.py`, `exec_agent.py` or `cli.py`.

  Loop slice 7b moves about 1,190 lines out of `_run_agentic_loop_body`. Every week the fixes wait, the rebase gets
  worse, and two of the fixes are security fixes.
- **The issue list is growing, but not from user reports.** 241 issues are open: 173 were opened since 1.3.3 and 95
  closed. About 90 of the open ones came from our own review rounds and about 8 from live runs. None has a milestone,
  and 29 carry a label.
- **The nightly went red without anyone noticing.** The scheduled run failed the Model-cache and Slow lanes from
  2026-10-31 to 2026-11-21, and nothing reported it. That is the same gap that decided the Test/CI grade on the
  1.3.1 score card.
- **Overdrive needs things from Maxim.** Maxim Overdrive is the game in which every NPC is a Maxim agent. It has a
  ranked list of what it needs: [overdrive_refactor_candidates.md](overdrive_refactor_candidates.md) here, and
  Overdrive's roadmap Phase 3 there. The owner's priority (2026-11-22) is anything that moves Overdrive forward, and
  using Overdrive to move Maxim itself forward.

## Owner decisions (2026-11-22)

- **D1. 1.3.5 is defined by its lines, not by the orchestrator.** The orchestrator line takes its number at its own
  cut. **Risk:** B2's entry condition is the `start_simulation_mode` decomposition (baseline_floors.md §B2), and B4
  is the longest chain to 1.4.0 (roadmap_1_3_x.md §Risks 4). If line 4 stalls, 1.4.0 slips. D9's owner for line 4
  is the mitigation.
- **D2. The `MaximAgent` breaking change ships in a patch.** `MaximAgent(shared_router=)` is required. #1228 landed it
  with its `changelog.d/1228-maxim-agent-router.breaking.md` fragment, which carries the migration; 10b's own V29
  version (`llm_router=`) was dropped in favour of it when 10b merged main (2026-11-25).
- **D3. 1.3.5 takes its own T1-2 clause.** It is called `release-1.3.5-T1-2`, in
  `docs/experiments/evidence_exceptions.json`, under the 2026-10-18 per-patch amendment (which covers 1.3.3 to
  1.3.6). It lands on `main` before the bump PR. At the clause's commit, someone re-checks by hand that T1-2 is the
  only STALE or BROKEN row (M25 is unbuilt).
- **D4. Loop slices 7a and 7b ride 1.3.5 only if their three-lens round finishes before the cut.** Otherwise they
  wait for the loop line. Slice 6 rides 1.3.5.
- **D5. Session A's file claim is split so the B2 code prerequisites can run beside the orchestrator line.** There
  are three:
  1. the `_llm_unavailable` fallback leaves the loop for a module with no bio imports;
  2. `execute_and_learn` becomes importable without bio modules (#1150);
  3. the `runtime` and `agents` `__init__`s become lazy.

  They stay exit items of the loop line. Who holds each one is a proposal in the sessions table below, not part of
  the decision.
- **D6. Phone Phase 0 leaves 1.3.5.** Its item P0-2 is console admission control: console_tunnel_hardening PR 3, re-homed
  for the third time. It moves to line 6, item 3, which builds the shared admission primitive that PR 3 rides on.
  P0-6 waits for contract 0.7.0 or its own minor bump. The other P0 items go to the next release after 1.3.5, with
  owners still owed. The owner still owes the #784 decision. The re-homing is recorded in
  [deferred/phone_device.md](deferred/phone_device.md) and register O22.
- **D7. Merge order, for this release only:** behaviour fixes first, verbatim moves last (§Merge order). It is a
  sequence for one release, not a standing rule, so it needs no check.
- **D8 and D13: Overdrive is engineering in 1.3.5, not evidence.**
  - The candidates doc's rule stands: **game runs never go on Maxim's ledger.** The transfer-check venue for tier C
    stays an open owner question (candidates §Cautions and §Not decided here).
  - Overdrive's **O10** already allows real Maxim against the world "inside strict red-gate tests". So a
    Maxim-owned harness may drive Overdrive's world server under strict red gates, which needs no change to
    Overdrive's decisions.
  - Overdrive's **O6** ("Mode (a) is dropped", replaced by the host-driven agent) still holds for the game. It is
    recorded in both `decisions/boundaries.md` and `decisions/npcs_and_sem.md`, and neither changes.
  - Line 3 covers the rest.
- **D9. Two fresh sessions.** One takes the world-clock A-pre plan (line 2). It starts first, because its plan
  review can begin now. The other takes the orchestrator line (line 4). Session A keeps the loop slices. Lines 1, 5
  and 6 are owned per item in the sessions table.
- **D10. `_run_agentic_loop_body` gets an `accepted_new_debt` record.**
  - This **supersedes** the owner decision of 2026-10-29 that the body was "deliberately left without" one (O21,
    roadmap_1_4.md T0).
  - The record has no repays-in field, so its `reason` names 1.5 as the release that repays it.
  - It lands in its own PR before the 1.4.0 bump PR, never in the bump (M56 item 11).
  - It does not lift item `1323.functions-1000`: through `split_from`, the body is still held to ≤1,000 lines, and
    it is 2,140.
  - First, read what the flagged block is.
- **D11. #1226 (narrator resume) becomes a required item in B4's (#1354) design review.** B4 either switches the
  resumed narrator off or adds an arm that does. It is recorded in the B4 prereg's open items. The parked branch
  `feat/1226-narrator-resume` is deleted once B4's prereg covers it.
- **D12. Sharing the Mac mini: game and playtest traffic never runs while an experiment holds the rig's leader.**
  The guard extends `scripts/box_quiet.py` with a rig kind, not a second waiter. One server, one leader.
  - `box_quiet.py` counts only processes on its own machine, so the rig kind runs **on the mini** and marks when an
    experiment holds it.
  - The game's clients run on the laptop, so they cannot see that mark. The refusal they get comes from the mini's
    proxy: line 6 item 3's admission refuses non-experiment clients while the mark is set. A client-side check is an
    ask to Overdrive.
  - Until both exist, backlog row M96 carries the rule.
- **#1601. llm.json tiers and functions are made to apply, and a bad entry fails loudly.** It ships with a
  `changelog.d/<slug>.changed.md` fragment, since existing files start to matter.
- **#1737. The GPU-less router gets an approach note in 1.3.5,** designed together with line 6 item 5 in one note.
  It is built in the loop line's release unless the note shows it is small. Who writes the note is a proposal in the
  sessions table. Session V filed #1737 and is told about the move.
- **M97. Before a Maxim release cut, someone reads Overdrive's weekly unpinned drift job (`drift.yml`).** That job is
  Overdrive's canary for Maxim's embedder surface. A red or missing run will be listed in the release check once backlog
  row M97 is built; until then the read is on the checklist. Owner, 2026-11-22.

**Third batch (owner, 2026-11-22):**
- **D14.** The 2026-10-18 per-patch T1-2 amendment covers 1.3.3 to 1.3.6. Under D1 a 1.3.7 before 1.4.0 is
  possible. The amendment is **not** extended now. That is decided when a 1.3.7 is actually planned, because an
  exception should not be granted before anyone needs it.
- **D15.** The sessions table below is confirmed as written.
- **D16. The proxy rule** (line 6 item 6): `NO_PROXY` is honoured on the backend call path, where an embedder
  makes the call. The 2026-09-28 "the proxy wins" rule keeps holding for Maxim's own operator commands. Both are
  pinned by tests.
- **D17.** The owner grants one short rig window, about 30 minutes outside any experiment run, for line 6 item 1.
  It runs from the laptop, before item 2 reaches the mini.
- **D18.** An Overdrive session writes `maxim_response_2` from §Answers to Overdrive, through Overdrive's process.

**Fourth batch (owner, 2026-11-25):**
- **D19. #1802 joins line 5.** A failure mode's pain has no onset latch, so it re-fires on every tick while the
  trigger holds and floods the PainBus learners. Overdrive's slice 2 session reported it, and a sustained heat stage
  would hit it directly. Its companion #1803 (`_human_is_driving()` is process-wide) is filed but stays out of
  1.3.5 (owner-decision on per-agent ownership).

## The lines

### Line 1 (headline): gates that hold

These are the agent's safety gates: plan approval, the framing of untrusted text, and the offline latch.

| Item | Issues | State on 2026-11-22 | Hot files |
|---|---|---|---|
| Plan approval as a gate | #1185, with #1736 (retired on the branch), #1734 | local branch, 18 commits, no PR | agent_loop, orchestrator, tool_dispatch, cli |
| Untrusted working-memory text framed | #1519, with #1733 | local branch, 23 commits, no PR | agent_loop, exec_agent, tool_dispatch, router |
| The SimulationAdapter fails open to "yes" | #1723 | not started; a two-line fail-closed fix | none |
| Autonomy refusals | #1550, #1549 | not started; ride on #1185 | agent_loop |
| The advisory follow-ups | #1751 (logs tool output at WARNING) first, then #1749, #1750, #1752 | filed 2026-11-22 | none |
| One prompt-line grammar | #1753 | after #1519 | none |
| The offline latch, `llm.offline` (**AvServ hardline**, with #1783) | #1315 item 10a (#1603), then 10b (#1735) | merged 2026-11-25 (0f50fdc9, 77880606); #1783 final | router, cli, exec_agent, orchestrator |
| Goal calls closed on the fallback path | #1414 (PR #1747) | draft; synced head not pushed | agent_loop, exec_agent |
| llm.json tiers apply | #1601 | decided (above) | router; after 10a and 10b |

Not in this line:
- #1729, the voice ceiling (#1519 part b), needs its own approach note;
- item 10c starts after 10b.

### Line 2: Overdrive's asks outside the fence

These come from step 1 of [overdrive_refactor_candidates.md](overdrive_refactor_candidates.md) §Order, plus
Overdrive's "issues now" table. Maxim can do them before the 1.4.0 cut. They also make Maxim's own runs replayable,
which needs #1305 too, so #1305 rides here.

| Item | Overdrive id | Maxim issue or plan | Size | Hot files |
|---|---|---|---|---|
| A caller's time of 0.0 becomes the wall clock (5 sites) | OD-I1 | #1684 | S | none |
| A pickle-capable load of the global embeddings file | OD-I3, cand. A5 | #1685, with #1531 (atomic save) | S | none |
| ARCHITECTURE.md says DM NPCs are built through AgentFactory | OD-I4 | #1686 | S | none |
| pymaxim caps `rich<14` for every embedder | OD-I6 | **to file** (`pyproject.toml` `[project] dependencies`) | S | none |
| Seeded runs are not reproducible (the global `random`/`np.random`) | cand. A2's first half | #1305 (a 1.4.0 gate item) | M | to be checked by the world-clock plan |
| The perceived-time port, A-pre half: prompt and observation time and the SCN-derived renders through the port (host or no-calendar), the hippocampus's 30 s window on the injected clock; the refractories in 1.3.6; live-path stamps in A-post as whole reader sets | OD-A2 to OD-A6, cand. A3 A-pre | [world_clock_a_pre.md](world_clock_a_pre.md) (PROPOSED 2026-11-24; owner decisions D-WC1 to D-WC7), from Overdrive's roadmap rows, the draft as background; then one PR per site family | M–L | none (A-pre by definition) |
| One `AgentHome` per agent | OD-A12, cand. A1 | with #984 | M | to be checked |

**Already done:** **OD-I7** (`apply_damage` returned 1.0 on a part with no sub-sensors) shipped in 1.3.4 as #1719
(`DamageRefused`). It goes in the answers to Overdrive (below).

**After the 1.4.0 cut:**
- the A-post half, with #1361 S3 to S6 in the loop line;
- cand. B2 (#1150 with A6);
- the stepped `think` (B1);
- A2's per-agent key, a new gate item (owner decision 2026-11-19).

### Line 3: the Overdrive bridge, as engineering (D8, D13)

**Why.** Maxim's evidence comes from Minecraft on the rig and from the Reachy, and both are slow, shared and hard to
replay: robot run #1708 has been blocked for weeks. Overdrive's world server is seeded, its physics are verified on
three grids, it runs on any box, and the repo is public. Even if it never earns a ledger row, a world we control lets
us exercise Maxim's loop, its body seams and its fear path under controlled, replayable conditions. That is test
value on its own.

**What line 3 does in 1.3.5:**
1. **The world side (Overdrive's work, through Overdrive's process).** The first scenario is a flooded room with
   `is_in_water` taken from the head's liquid fraction and `oxygen` owned by the world (Overdrive's `gameplay.md` §5
   item 1). None of that exists yet: Overdrive's `src/simcore` has heat but no liquids or oxygen.
2. **A Maxim-owned harness skeleton under strict red gates.**
   - It lives in `scripts/` and installs Overdrive's `simcore` as a test-environment install, never as a pymaxim
     dependency.
   - It asserts its own repo's interpreter ([lint_harness_provenance.py](../../scripts/lint_harness_provenance.py)).
   - It stamps the Overdrive commit as code under test, since the harness-provenance rule covers only the `maxim`
     import.
   - It drives the world server directly and bypasses Overdrive's SEM adapter. Overdrive's `boundaries.md` §13 keeps
     real Maxim behind the adapter a strict red gate until the embeddable contract ships.
   - Its first tests are strict red gates (`xfail(strict=True)`), as O10 allows. They run in a named CI cell that
     installs `simcore`, and without `simcore` they fail loudly. A strict xfail that is skipped because the import
     failed never runs.
3. **A design brief, not a prereg.** Its open questions:
   - What would make an Overdrive result count as evidence? That is the tier C venue, which is the owner's call.
   - Is there a contingency in Overdrive that BF2's floor can lose? T1-13 (Exp 60) is PARTIAL because B1's tabular
     Q-learner matched FEAR (8/8); FEAR already beats RANDOM. The same one-step contingency in a second world gives Q
     the same easy task. That is M46's and E1's redesign question.

   If the brief ever leads to a claim, that claim gets all four design lenses ([DESIGN_REVIEW.md](../experiments/DESIGN_REVIEW.md)),
   because a new world, a new bridge and a new harness make a new design.

Line 3 does not gate the 1.3.5 cut.

### Line 4: the orchestrator, and the B2 prerequisites (they ship when ready)

- **The orchestrator slices.** `start_simulation_mode` comes down toward ≤1,000 lines, descendants included, with
  the riders in [roadmap_1_3_x.md](roadmap_1_3_x.md) §1.3.5. Prune that list first: #1259, #1301, #1451, #1453,
  #1454 and #1457 are closed, and #1028 was closed in the 2026-10-24 triage.
  - #1093, #1123 and #1455 (from the 1.3.4 carry list) ride here. #1093 and #1123 are the same class: the
    orchestrator prompt names parts or sensors the body lacks.
  - The first new slice waits until the line 1 branches that edit `orchestrator.py` have merged: #1185, 10a, 10b,
    #1228 and #1680. **All five have merged.** The line runs as session maxim-e6.
  - **The approach is settled** in [orchestrator_simrun_approach.md](orchestrator_simrun_approach.md): staged frozen
    run-state bundles (`SimRun`, `SimAut`, `SimOrch`, `SimAutLate`), each unpacked into the original local names, under
    strict rule (d). The owner decided it on 2026-11-27, after three review rounds. The #863 orchestrator fixes
    land before the builder moves (#1838). The slices also build in the setup-abort teardown (owner, 2026-11-27).
- **B2's prerequisites, beside it (D5).**
- **The known trap is settled by D10.**

### Line 5: fix groups that keep chipping away

| Group | Issues | Axis | Size | Hot files |
|---|---|---|---|---|
| The agent gets the wrong model or the wrong stores (B0b) | #1225, #1227, #1228 (branch exists), #1728; the `arm_effective` stamp that M46's floor check needs is built with #1227. Whatever release it lands in, it must land before B4 (owner 2026-10-18). | Runtime correctness | S–M | exec_agent, orchestrator, cli |
| CI that tells the truth | M95 (open an issue when a nightly goes red); the flakes #1710, #1534, #1557, #954, #1553, #1576, #1493, #1456, each closed on a structural probe; the tooling items #1313 (which unblocks M60b), #1271, #1325, #1384, #1031, #1246 | Test/CI truthfulness | ~3–4 days | none |
| Unsafe loads and stray writes | #1666; #1390 with #1676; #1342; #1404's residuals (`$HOME` to `~`); the 12 hand-rolled atomic renames down to 0 (#1685 and #1531 are in line 2) | Runtime correctness, persistence | ~2 days | #1390 touches exec_agent and cli |
| Small standalone defects | #1680 (branch exists), #1692, #1742, #1596 (`--sandbox docker` silently runs with no sandbox), #1670, #1802 (failure-mode pain has no onset latch, D19) | Runtime correctness | S each | #1680: cli, orchestrator, exec_agent; #1596: orchestrator; #1802: embodiment/sem.py, body.py |
| Console truth and health | #1430, #1285, #1286, #1287, #1291 | Runtime correctness | S–M each | none |
| Routing setup truth | #1396 (routers built outside `build_primary_router` never ask to download a model), #1295 (doctor checks the installed version against published advisories) | Runtime correctness | S–M | #1396: router, after 10a and 10b |
| Bio-memory state, the small items | #982, #988, #985, #1474, #1471, #1532, #1575 | Runtime correctness | S each | none |

**Housekeeping in this PR:** the stale text in the other plan docs that still say 1.3.5 is the orchestrator, or
that Phone Phase 0 is in it (wire-integrity review). They are folded on this branch:
- phone_device.md;
- console_tunnel_hardening.md;
- outstanding.md O21 and O22;
- roadmap_1_3_x.md §1.3.5, §1.3.6, §Path, §Sequence, §Risks, §After 1.4.0;
- roadmap_1_4.md;
- baseline_floors.md §B2;
- advisor_mode.md;
- the README "Now" line;
- the B4 prereg (D11).

**Housekeeping with the release:**
- a `1.3.5` milestone on every issue above;
- a security-label pass on #1685, #1609, #1632, #1342 and #1666;
- CLAUDE.md's test count;
- removing the clean, merged worktrees (look at the dirty ones first);
- a blind re-score at the 1.3.5 tag, the first since 1.3.1.

### Line 6: the peer mesh, for Overdrive

**The workflow (owner, 2026-11-22).** An LLM is served on the Mac mini. The owner develops and tests Overdrive from
the laptop: the world server and the Maxim agents run on the laptop, and their model calls go to the mini over
Tailscale. That also keeps the harness off the leader box, as the no-harness-on-leader lesson requires.

**What works today.**
- On the mini, a `maxim` CLI leader spawns `llama_cpp.server` on `127.0.0.1:8100` behind `LeaderProxy` on
  `0.0.0.0:8099` (bearer mesh key, 413 context gate). No tunnel is needed: Tailscale's WireGuard encrypts plain
  `http://<mini>:8099/v1`.
- On the laptop, `maxim peer connect <url>` writes `lanes.large`, and `maxim peer test` and `maxim doctor --as peer`
  probe it.
- An embedder on the laptop gets the peer through `create.router` and `create.agent`.

**What hurts it, worst first** (read on b98954f7; the behaviour of `llama_cpp.server` under load is inferred, and
item 2 measures it):
- **The mini generates one sequence at a time, and nothing says so.**
  - `local_server_spawner.py::_build_cmd` spawns `llama_cpp.server` with no parallel slots.
  - The proxy admits `proxy.max_concurrent` (default 4).
  - Anything over the cap gets an instant 429, with no queue (`leader_proxy.py::_ProxyHandler._check_admission`).
  - No endpoint advertises capacity.
- **#1609:** `interrupt_requests` is on by default (llama-cpp-python 0.3.36). A concurrent keyed request, even a
  `/v1/models` probe, truncates a reply that is still streaming.
- **Cancel does not free the mini.** The peer backend checks cancellation only before and after a call. A killed
  caller may leave an orphan generation holding the only slot.
- **The mini is shared with the Minecraft and experiment runs.** D12 settles that.
- **There is no public surface for embedders.** Overdrive's gateway has to build the private `_MaximPeerBackend`
  and an `LLMConfig` by hand.
- **Smaller gaps:**
  - `peer connect` sets only `lanes.large`;
  - each `LLMRouter` serialises its calls on `_inference_lock`;
  - timeouts are per read, not per call;
  - the keys: #1632, #1633 and #1478;
  - there is no per-client accounting.

**Items, in order:**

| # | Item | Closes / files | Size | Design pass |
|---|---|---|---|---|
| 1 | **Measure first:** N concurrent buffered and streamed calls against the mini's proxy (the 429 rate, latency and truncations), and what a client disconnect does to an upstream generation. It runs off the leader, in the rig window D17 grants under D12, and **before** item 2 reaches the mini, so #1609's truncations get a baseline. | answers the transport half of Overdrive's cancellation question | S | no |
| 2 | Spawn with `--interrupt_requests False`, plus a `_build_cmd` guard test. Reaching the mini takes `maxim peer update && maxim peer restart`, which needs `proxy.allow_remote_update`: an owner step. | #1609 | S | no |
| 3 | **The shared admission primitive in `_check_admission`:** a bounded FIFO with a deadline instead of an instant 429; a cap that defaults to the upstream's real parallelism; capacity advertised on `/v1/models` or `/v1/debug/status`; per-client accounting; and D12's hold mark: while an experiment holds the leader, non-experiment clients are refused, naming the holder. The proxy cannot tell clients apart today, because `_check_admission` buckets by IP and every peer shares one cluster key. So the approach note must name how a client proves it is the experiment (a distinct key or principal, with #1632's key work), or D12 has no refusal path. Whether the leader should serve llama.cpp's own `llama-server` with parallel slots is decided from item 1's numbers. Then console_tunnel_hardening PR 3's console half (`limit_concurrency`, `ws_max_size`, the input cap, the `/api/pair/claim` rate limit) follows as a PR on this primitive (D6). | to file ("LeaderProxy: queueing admission and advertised capacity"); PR 3 | M | **yes, a gate**: an approach note first |
| 4 | Keys: the server holds the mesh key, then show and rotate, then the creation lock | #1632, #1633, #1478 | M | yes for #1632 |
| 5 | **The question, not the answer: how does an embedder make one backend call?** Overdrive's gateway resolves placement itself (its pushback, OD-A24) and builds the private `_MaximPeerBackend`. The design pass starts from the front-gate question: can a one-provider `LLMRouter` from `create.router` do it, once its router defects (the `_inference_lock` serialisation, per-read timeouts, no whole-call deadline) are fixed in the router? Only if not does a public factory get built, inside `models/language/` with an allow-listed importer, like the precedent `closed_local.py`. Two invariants are in play: CLAUDE.md's router-only LLM access (and [llm-router-only-access.md](../lessons/llm-router-only-access.md) with test.yml's "1.0 guard promotion" grep), and llm-routing.md's "placement rides on `provider_priority`, NOT a new resolver". An amendment to either needs a DECISIONS entry. #1737 is designed in the same note. | to file (OD-A24, OD-A14); #1737 | M | **yes** |
| 6 | Pin with tests that `NO_PROXY` is honoured on the backend path and that the proxy wins for operator commands (D16), and expose `strict_address_check` | to file (OD-I10) | S | no (decided, D16) |
| 7 | `peer connect` sets all three lanes, or says which it left local; plus a "Tailscale peer" troubleshooting page (the 8099 URL, the lanes, `doctor --as peer`, the one-generation fact) | to file | S | no |
| 8 | Typed errors instead of empty content from `openai_compatible` backends | to file (OD-I9) | M | no; it may slip while the mini stays on `maxim_peer` |
| 9 | Restart hygiene | #1627, #1613, #1498 item 3 | S each | no |

**Overdrive's side, not Maxim's:**
- set `max_in_flight` for `maxim_peer` from the advertised capacity (about 1 today);
- classify a 429 as overloaded;
- its `owner_lan` HTTPS check;
- the Phase 4 fake-server tests.

**Not in 1.3.5:** streaming tokens to the caller (OD-A25). It belongs to the stepped-`think` redesign after 1.4.0.

**Where the OD ids live.** OD-A14, OD-A23 to OD-A25, OD-I9 and OD-I10 are in Overdrive's
`docs/design/inference_placement.md` on its main (landed 7cfea02, 2026-11-21).

### Answers to Overdrive

The last batch of answers went to Overdrive as `docs/design/reviews/maxim_response_1/`. This batch becomes
`maxim_response_2`, relayed by the owner or an Overdrive session through Overdrive's own process:
- OD-I7 is fixed (#1719, 1.3.4).
- OD-I6 (the `rich` cap): the issue number, once it is filed.
- OD-A25 (streaming): after 1.4.0, with the stepped `think`.
- OD-A23 (cancellation), transport half: item 1's measurement.
- OD-A24 and OD-A14: item 5's design pass, which starts from the question above.
- OD-I9 and OD-I10: items 8 and 6.
- Game runs stay off Maxim's ledger, and the harness runs under O10's strict red gates (D8, D13).
- `max_in_flight` on `maxim_peer`: about 1 until item 3 advertises capacity.

### Not in 1.3.5

| Item | Why |
|---|---|
| #1707, #1708 | waiting on Pollen and the robot |
| #1516 and the experience-stream plan | a 1.4 line |
| #1226 | D11: an item for B4's design review |
| #1418, #1336, #1428, #1431 (Console features) | need contract 0.7.0 (#1428 rides R3's bump), a pulse release, or a tools-seam design |
| #1439 (Console Talk reaches the body-less agent on a Reachy) | no bounded patch: its fix is the embodied-device contract ([embodied_device.md](embodied_device.md)) |
| #1417 (retire the env projection) | L; it touches the router configuration path after 10a and 10b. It goes in the next release. |
| #845 | already homed with the loop line |
| Phone Phase 0, except P0-2 | D6 |
| The Exp 38 harm probe | still owed before any Exp 38 citation |

## Sessions for 1.3.5 (confirmed by the owner, D15)

| Session | Owns | Shares (grant recorded here) |
|---|---|---|
| A (loop) | `agent_loop.py`, the loop slices 6/7a/7b, #1150, B2 prerequisite 1 | grants the line 1 fixes their `agent_loop.py` hunks; may grant B2 prerequisite 3 |
| New: world clock (D9) | line 2's plan and its PRs | none in the hot files by definition |
| New: orchestrator (D9) | `orchestrator.py`, line 4 | `cli.py` hunks for its slices, granted by A; starts after the line 1 merges that touch `orchestrator.py` |
| V (atonia) | 10a, 10b, #1601, and the one note covering line 6 item 5 and #1737 (one note, one owner) | `router.py` and `cli.py` hunks in 10a/10b |
| H (security and mesh) | #1185, #1519 and their followers; line 6 items 2–4 and 9 (`leader_proxy.py`, `local_server_spawner.py`, `tunnel/`) | hot-file hunks from A and the orchestrator session, recorded on each PR |
| Unassigned (owner) | line 5's groups, line 6 items 1 and 6–8, line 3's harness | each takes a session when it starts |

Before anyone starts, the owner asks each holding session to push its local branches as drafts, as a backup. That is
an action for this release, not a rule.

## Merge order

1. **Behaviour fixes:** #1747, #1185, #1519, #1228 and #1680. The order among them comes from `scripts/pr_overlap.py`
   across the branches, and each passes `pr_merge_readiness`.
2. **10a, then 10b** (V22). Then #1601, #1396 and line 6's `models/language/` items (5 and 8).
3. **Loop slice 6** (merged, #1758). Then 7a, 7b and the B2 edge are rebased onto it (D4).
4. **The first new orchestrator slice,** after step 1's `orchestrator.py` branches.
5. **Everything else** merges whenever it is green: the line 2, 5 and 6 PRs that touch no hot file.

When a fix is rebased onto code a slice has already moved, it goes to that code's new home. The rebased fix must
still pass `lint_function_length.py`: inside `_run_agentic_loop_body`, `start_simulation_mode` or `_main_impl`, rule
T (touch it, shrink it) applies. All five fix branches pass it standalone today.

## Exit

1.3.5 is ready to cut when all of these hold:
- line 1 has merged (#1753 and 10c may ride later);
- **AvServ hardline (owner, 2026-11-23):** #1603 and #1735 have merged, and the `maxim.advisor` conformance note
  against AvServ plan 50 §18 (#1783, `docs/user/advisor.md` §Conformance) is final: re-checked on the merged code, its
  DRAFT status removed, and AvServ told whether 10a or 10b changed anything in §18;
- line 2's S items are fixed, the world-clock A-pre plan has passed its review, and its first PR has merged;
- line 6 items 1, 2 and 7 have merged or run, and the approach notes for items 3 and 5 have passed review (item 1
  needs the owner's rig window);
- the nightly on the merge commit is green (`check_nightlies.py --at <sha>`), and `release-1.3.5-T1-2` is on `main`
  with T1-2 re-checked as the only STALE or BROKEN row;
- Overdrive's drift job has been read (M97).

**Only some of this is checked by a tool.** `scripts/changelog_fragments.py check` covers the fragments and the bump
section. The evidence gate covers the T1-2 clause. `check_nightlies.py` covers the nightly. Everything else on this
list is the release session's checklist, and no check enforces it.

Lines 3, 4 and 5 do not gate the cut. Whatever has merged ships.
