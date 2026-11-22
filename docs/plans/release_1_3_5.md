# 1.3.5: what ships, what rides along, and Overdrive as Maxim's test world

**Status:** proposed 2026-11-22, the day 1.3.4 shipped. Owner decisions D1–D7 below were taken the same day; D8–D11
are open. The numbers in this plan were read on `origin/main` b98954f7 (v1.3.4) and from `gh` on 2026-11-22.
Re-check any of them before citing it later.

This plan replaces "1.3.5 = the orchestrator" in [roadmap_1_3_x.md](roadmap_1_3_x.md) §1.3.5. The orchestrator
remains a line, the head of the chain to 1.4.0, but it no longer decides when 1.3.5 ships.

## Why the plan changed

- **The planned mainline has nobody on it.** No `start_simulation_mode` slice has landed since 2026-10-30, and no
  branch exists for one. The function is 1,895 lines, 895 over gate item `1323.functions-1000`. Phone Phase 0 has no
  code either.
- **Finished work is waiting, only on one machine.** These branches are finished or nearly so, synced on v1.3.4, and
  either have no PR or a draft PR whose synced head is not pushed:
  - #1185 (`fix/1185-approval-intent-whole-word`);
  - #1519 (`fix/1519-frame-untrusted-wm-text`);
  - #1228/#1225 (`fix/1228-1225-aut-router-and-bio-flags`);
  - #1680 (`fix/1680-write-paper-kwarg`);
  - #1414 (`fix/goal-call-closure`, draft PR #1747);
  - atonia 10a and 10b (draft PRs #1603 and #1735);
  - loop slice 6 (`refactor/loop-slice6-intake`).
- **Those branches collide.** Most of them edit `runtime/agent_loop.py`, `agents/exec_agent.py` and `cli.py`. Loop
  slice 7b moves about 1,190 lines out of `_run_agentic_loop_body`. Holding the release for the orchestrator would
  leave two security fixes unreleased, and every week they wait makes the rebase worse.
- **The issue list is growing, but not from user reports.** 241 issues are open: 173 opened since 1.3.3 and 95
  closed. About 90 of the open ones came from our own review rounds and about 8 from live runs. No issue has a
  milestone, and about 30 carry a label.
- **The nightly was red for 22 days** (2026-10-31 to 2026-11-21: the Model-cache and Slow lanes) and nothing
  reported it. The 1.3.1 score card's deciding Test/CI gap was the same thing.
- **Maxim Overdrive** (the game repo, where every NPC is a Maxim agent) has a ranked list of what it needs from
  Maxim: [overdrive_refactor_candidates.md](overdrive_refactor_candidates.md) here, and Overdrive's roadmap Phase 3.
  The owner's priority (2026-11-22) is anything that moves Overdrive. Overdrive should also become a test world for
  Maxim itself, since we control that world completely.

## Owner decisions (2026-11-22)

- **D1.** 1.3.5 is defined by its lines below, not by the orchestrator. The orchestrator keeps its line and ships in
  whichever release it is ready for (the 2026-10-24 numbering rule).
- **D2.** 10b's breaking change (`MaximAgent(llm_router=)` required, V29) is accepted in a patch release, with a
  migration note in the CHANGELOG.
- **D3.** 1.3.5 takes its own T1-2 clause, `release-1.3.5-T1-2`, in `docs/experiments/evidence_exceptions.json`,
  under the 2026-10-18 per-patch amendment. The 1.3.4 clause covered 1.3.4 only. The clause lands on `main` before
  the bump PR.
- **D4.** Loop slices 7a and 7b ride 1.3.5 only if their three-lens round finishes before the cut. Otherwise they go
  in 1.3.6. Slice 6 rides 1.3.5.
- **D5.** The orchestrator gets an owner now. Session A's file claim is split so the B2 code prerequisites (#1150 and
  the lazy `runtime`/`agents` `__init__`s) can run beside it, since they do not touch `orchestrator.py`.
- **D6.** Phone Phase 0 leaves 1.3.5. It needs contract 0.7.0, which has not shipped.
- **D7.** Merge order: behaviour fixes first, verbatim moves last (§Merge order).

## The lines

### Line 1 (headline): gates that hold

The agent's safety gates: plan approval, the framing of untrusted text, and the offline latch. Almost all of the
code exists.

| Item | Issues | State on 2026-11-22 |
|---|---|---|
| Plan approval as a gate | #1185, with #1736 (retired on the branch), #1734 | local branch, 18 commits, no PR |
| Untrusted working-memory text framed | #1519, with #1733 | local branch, 23 commits, no PR |
| The SimulationAdapter fails open to "yes" | #1723 | not started; a two-line fail-closed fix |
| Autonomy refusals | #1550, #1549 | not started; ride on #1185 |
| The advisory follow-ups | #1751 (logs tool output at WARNING) first, then #1749, #1750, #1752 | filed 2026-11-22 |
| One prompt-line grammar | #1753 | after #1519 lands |
| The offline latch, `llm.offline` | #1315 item 10a (#1603), then 10b (#1735) | draft PRs; re-synced heads not pushed |
| Goal calls closed on the fallback path | #1414 (PR #1747) | draft; re-synced head not pushed |

Not in this line: #1729 (the voice ceiling, #1519 part b) needs its own approach note. Item 10c starts after 10b.

### Line 2: Overdrive's asks that touch no fenced file

These are the rows in [overdrive_refactor_candidates.md](overdrive_refactor_candidates.md) §Order step 1 and
Overdrive's "issues now" table, which Maxim can do before the 1.4.0 cut. None of them edits `orchestrator.py`,
`agent_loop.py` or `tool_dispatch.py`, so this line does not collide with line 1 or the loop slices. This is also the
work that lets Maxim's own experiments replay bit for bit.

| Item | Overdrive id | Maxim issue / plan | Size |
|---|---|---|---|
| A caller's time of 0.0 becomes the wall clock (5 sites) | OD-I1 | #1684 | S |
| Pickle-capable load of the global embeddings file | OD-I3, cand. A5 | #1685 (with #1531, which makes its save atomic) | S |
| ARCHITECTURE.md says DM NPCs are built through AgentFactory | OD-I4 | #1686 | S |
| pymaxim caps `rich<14` for every embedder | OD-I6 | **to file** (pyproject.toml:50; the cap holds Overdrive's import-linter at 2.6) | S |
| The perceived-time port, A-pre half: prompt and observation time through the port, live-path stamps outside the fence, the hippocampus's 30 s wall window, reaction gates on the injected clock | OD-A2 to OD-A6, cand. A3 A-pre | **plan owed**: Overdrive's `docs/maxim_upstream/world_clock_release_DRAFT.md` becomes a Maxim plan with a three-lens plan review, then one PR per site family | M–L |
| One `AgentHome` per agent | OD-A12, cand. A1 | with #984 (the orchestrator still shares learned state) | M |

Already done: **OD-I7** (`apply_damage` returned 1.0 on a part with no sub-sensors) shipped in 1.3.4 as #1719
(`DamageRefused`). Tell Overdrive, so its roadmap row closes.

Not before the 1.4.0 cut: the A-post half (with #1361 S3–S6 in 1.3.6), cand. B2 (#1150 with A6), and the stepped
`think` (B1, after 1.4.0). A2's per-agent key is a new gate item after the cut (owner decision 2026-11-19).

### Line 3: Overdrive as Maxim's test world (paper in 1.3.5)

**Why.** Maxim's evidence comes from Minecraft on the rig and from the Reachy. Both are slow, shared and hard to
replay. The rig holds one run at a time (see the no-harness-on-leader lesson), and the robot run #1708 has been
blocked for weeks. Overdrive's world server is seeded, its physics are verified on three grids, it runs on any box,
and we write every rule. A result that holds in a second world we built is stronger evidence than a third run in the
same world.

**The first experiment: Exp 60 transfer, drowning avoidance in Overdrive physics.** It is item 1 of Overdrive's
`docs/design/research/states_of_matter/gameplay.md` §5:
- a flooded room;
- `is_in_water` from the head's liquid fraction;
- `oxygen` owned by the world;
- the earned cue-before-harm pattern;
- a negative control.

It needs no Maxim change. Ledger row **T1-13** (Exp 60) is PARTIAL, with the floor matched under BF2. Beating BF2's
`tabular_q` and `random` floors in a second world is the evidence that row lacks. The design review decides whether
it can.

**How, in Maxim's process:**
1. A design brief. It is a derivative experiment, so it gets the confounding and bio-faithful lenses
   ([DESIGN_REVIEW.md](../experiments/DESIGN_REVIEW.md)). If the review finds the claim is new, it gets all four.
2. A pre-registration frozen before data, with a ledger row.
3. A Maxim-owned harness in `scripts/`. It imports Overdrive's `simcore` as a test-environment install, never as a
   pymaxim dependency. It asserts its own repo's interpreter
   ([lint_harness_provenance.py](../../scripts/lint_harness_provenance.py)).
4. Code review, a dry run, then the run, on any box except the leader.

**A conflict to settle first (D8).** Overdrive's owner decision **O6** says Overdrive builds "no interim mode in which
Maxim's agent runs its own loop beside the world". Its real NPCs wait for tier A's host-driven agent, after Maxim's
1.4.0 cut. A Maxim harness that runs today's loop against Overdrive's world server is that interim mode, used for
testing. The proposal keeps O6 for the game and lets Maxim harnesses drive the world server, as candidates tier C
already assumes ("a transfer check in a Maxim-owned harness").

**Already in place on the Overdrive side:** a scheduled job runs Overdrive's gate unpinned against the newest pymaxim.
That makes Overdrive a downstream canary for Maxim's embedder surface. Line 5 should add reading it to the release
checklist.

### Line 4: the orchestrator and the B2 prerequisites (they ship when ready)

- `start_simulation_mode` slices toward ≤1,000 lines, descendants included, with the riders in
  [roadmap_1_3_x.md](roadmap_1_3_x.md) §1.3.5. Prune that list first: #1259, #1301, #1451, #1453, #1454 and #1457
  are closed, and #1028 was closed in the 2026-10-24 triage.
- B2's code prerequisites, from §1.3.6, beside it (D5): #1150 and the lazy `__init__`s. #1150 is also Overdrive's
  OD-A18 (cand. B2), so one split serves both.
- A known trap: the lint reports one "new debt outside debt_snapshot" entry for `_run_agentic_loop_body`. By the
  2026-10-29 decision, 1.4.0 ships only with it ≤200 lines or an `accepted_new_debt` record (D10).

### Line 5: fix groups that keep chipping away

Each group is small, touches few hot files, and moves a score-card axis.

| Group | Issues | Axis | Size |
|---|---|---|---|
| The agent gets the wrong model or the wrong stores | #1225, #1227, #1228 (branch exists), #1728 | Runtime correctness | S–M |
| Seeing a red CI | M95 (open an issue when a nightly goes red); flakes #1710, #1534, #1557, #954, #1553, #1576, #1493, #1456, each closed on a structural probe | Test/CI truthfulness | 2–3 days |
| Unsafe loads and stray writes | #1666, #1390 with #1676, #1342, and the 12 hand-rolled atomic renames to 0 (#1685 and #1531 are in line 2) | Runtime correctness, persistence | ~2 days |
| Small standalone defects | #1680 (branch exists), #1692, #1742, #1596 (`--sandbox docker` silently runs with no sandbox), #1670 | Runtime correctness | S each |
| Leader, tunnel and keys | #1609, #1632, #1613, #1627, #1633, #1478 (`peer/` and `tunnel/`; no hot files) | Security-adjacent | M |
| Bio-memory state, the small items | #982, #988, #985, #1474, #1471, #1532, #1575 | Runtime correctness | S each |

**Housekeeping, with the release:**
- a `1.3.5` milestone on every issue above;
- a security-label pass on #1685, #1609, #1632, #1342 and #1666;
- fix the stale plan text: §1.3.5 and §1.3.6 in roadmap_1_3_x.md, and [README.md](README.md)'s "Now" line;
- CLAUDE.md's test count (it says 15,427; the shards ran about 19,680);
- remove the 76 worktrees that are clean and merged; look at the 5 dirty ones first;
- a blind re-score at the 1.3.5 tag (none since 1.3.1).

### Not in 1.3.5

| Item | Why |
|---|---|
| #1707, #1708 | waiting on Pollen and the robot |
| #1516 and the experience-stream plan | a 1.4 line |
| #1226 | owner decision owed (D11) |
| #1418, #1336 (Console) | need contract 0.7.0 and a pulse release |
| Phone Phase 0 | D6 |
| The Exp 38 harm probe | still owed before any Exp 38 citation |

## Merge order

1. **Behaviour fixes:** #1747, #1185, #1519, #1228 and #1680. The order among them comes from
   `scripts/pr_overlap.py` across the branches. Each passes `pr_merge_readiness`. Push every branch as a draft now,
   as a backup.
2. **10a, then 10b** (V22).
3. **Loop slice 6.** Then rebase 7a/7b and the B2 edge onto it (D4).
4. Line 2 and line 5 PRs touch none of the three hot files, so they merge whenever they are green.

Before merging, every fix that touches a section a slice has already moved goes to the section's new home (rule T).

## Exit

1.3.5 is ready to cut when all of these hold:
- line 1 has merged, except #1753 and 10c, which may ride 1.3.6;
- line 2's S items are fixed, the world-clock A-pre plan has passed its review, and its first PR has merged;
- line 3's design review is complete and its prereg drafted (paper only);
- the nightly on the merge commit is green, and `release-1.3.5-T1-2` is on `main`.

## Open owner decisions

- **D8.** Overdrive's O6: may a Maxim-owned harness drive Overdrive's world server with today's loop, for experiments
  only? (Proposed: yes; the game keeps O6.)
- **D9.** Who owns the orchestrator line, and who owns the world-clock A-pre plan? Both are fresh sessions; neither
  collides with Session A's slices.
- **D10.** `_run_agentic_loop_body`: an `accepted_new_debt` record, or ≤200 lines by 1.4.0?
- **D11.** #1226 (narrator resume): revive or drop.
- **Atonia, carried:** #1601 (llm.json tiers never apply) and #1737 (the GPU-less router; approach note first).
