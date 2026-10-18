# Development flow: faster without lowering the bar

**Adopted 2026-09-29** (owner decision), after a session in which two gate-shaped PRs took most of the
time: M1a (sim `report.json` provenance, PR #999: "two-lens round 1, a security design review, then rounds
2–6", per its body) and the #998 fix branch (four rounds as observed in the session). The bar did its
job: every round found something real. The time went into finding the same class of problem one layer at
a time, and into re-running a ~12-minute suite (as of 2026-09-29; by 2026-10-16 it was 15,427 tests and ~38 min
serial under coverage in CI, [ci-efficiency-2026-10-16.md](ci-efficiency-2026-10-16.md)) that each fold invalidated (two full runs were lost to
edits made mid-run -- a session observation). These practices keep every check and change the order
and the plumbing around them.

What does **not** change: a different reader before merge, deletion probes on every guard, the full
lint set finishing before a push, and a full suite on the final state before a push. Each caught
something real in that session (vacuous tests, a CI-only lint failure, a green test hiding a regression).

## 1. A gate gets an adversarial design pass before it is built

**Applies to:** any mechanism whose job is to refuse, attest or bind evidence -- provenance stamps, the
clean-tree flag, ledger and prereg lints, allowances, security boundaries.

**Do:** write a one-page approach note (what the mechanism trusts, what it reads, what it refuses), and
have a reader red-team the note before any code: enumerate every input the mechanism consults (config,
environment, index, refs, the filesystem, the caller) and how each could make it lie. Fold the answer
into the design, then build. The four-lens experiment design review (`docs/experiments/DESIGN_REVIEW.md`)
is the same idea for experiments.

**Why:** M1a's digest first hashed `git diff` text. Rounds 2-5 then found, one per round, that git config
(an external diff tool made every dirty tree hash as clean), ignore rules, index flags, inherited `GIT_*`
variables and replace refs could each hide code. The round-4 design -- hash on-disk content, ask git only
for the path set, pin every git input -- was reachable on day one by asking "what does git consult?"
before writing the first version. The allow-dirty question (a harness allowance inherited by sub-sims)
reached a security review mid-PR; it belonged in the approach note.

**Stop rule for a gate's review rounds:** when the REVIEWER classifies every finding in a round as
fail-closed (it can only make a clean tree read dirty, never the reverse) and in a class already covered,
those findings go to a follow-up issue, not onto the branch, and no further round is owed. The author
does not make that classification: an author-attested "it's only a NIT" deciding whether a gate is
reviewed is exactly the weak evidence CLAUDE.md rules out. Any `src/` fold after the last round still
gets a round (a delta-scoped one, §6, is enough). A gate inside an experiment harness gets both passes:
this one and the four-lens design review.

## 2. A fold never invalidates a running suite

**Do:**
- Run the full suite on a **snapshot**: `scripts/suite_at_commit.sh` ([how-to](../agents/ci-testing.md)) checks a commit out into a
  throwaway worktree and runs the suite there. Snapshot uncommitted tracked work with
  `scripts/suite_at_commit.sh "$(git stash create)"` (an unsigned commit object, no ref moved), or commit
  it. Keep folding in the working tree meanwhile. The result names the commit it tested -- that is all it
  proves: it is not evidence of which diff was REVIEWED.
- **One full suite per fold batch, not per round.** Narrow tests plus deletion probes vet a fold. The last
  full suite runs on the exact commit being pushed.
  *Note 2026-10-16:* the owner adopted an alternative: CI's green, non-cancelled `unit-tests` run on the PR's
  head SHA counts as that final full-suite run. It takes effect only when mechanization backlog M64
  (`scripts/pr_merge_readiness.py` asserts that run) lands; until then the rule above stands as written.
- Narrow test runs alongside a snapshot suite share `~/.maxim`, ports and other global state; keep them
  small. (The §3 xdist trial found no such collisions, but it was reverted for coverage order noise.)
- Never edit `src/` in a tree whose suite is running. (This cost two full runs in one session.)

## 3. Parallel test runs

The xdist trial (M22) ran and was reverted: `pytest-xdist` is installed (the `[test]` extra and CI's pip line)
but the suite runs serially again, and serial stays the reference. #1339 ran it under `-n auto --dist loadfile`
(about 17 min against ~35 min serial); #1359 reverted it because coverage is test-order dependent, so the
exact-pin coverage ratchet flakes under a reshuffled order. Re-landing waits on M60, after the 1.3.3 release.
The measurements, the unmet pre-registered coverage condition and the three determinism fixes the trial
surfaced live in one place: [docs/agents/ci-testing.md](../agents/ci-testing.md) §1.

## 4. Decisions up front

At the start of an issue, list the owner decisions it will need (scope choices, fence exceptions, design
forks such as "where does an allowance live") and ask them together, before building. Mid-build questions
stall the work and tend to arrive one at a time.

## 5. Keep the pipeline full

While a PR waits for the owner's merge, start the next issue in its own `.worktrees/<slug>` (CLAUDE.md's
worktree rule), so review, CI and merge latency overlap with work. Rebase the next branch once the first
merges if they touch the same files.

## 6. Review prompts carry the delta and a stop condition

A follow-up review round gets exactly the delta since the last round ("check only this"), the findings it
is meant to verify, and the stop rule from §1. It reads the rest only to judge the delta. First rounds
keep the full three-lens brief ([../CODE_REVIEW.md](../CODE_REVIEW.md); the third lens, wire integrity, since 2026-10-05).

## 7. From the Session B follow-up cleanup (2026-10-08 to 10-10)

Each of these cost a round or a red main once; most now have a check, and the rest are habits.

- **Stacked PRs: check every base after each merge.** GitHub merged #1183 into its parent branch, not `main`, so its
  parent #1182 then carried both changes. After any stacked merge, confirm the next PR's base is `main` before
  merging it.
- **"Closes #N" once per issue.** "Closes: #1, #2, #3" closes only the first; repeat the keyword for each.
- **A lint that scans tracked files is tested with its own new files tracked.** #1175's backlog lint was green only
  because its test file was untracked; once tracked, the lint failed on its own examples. Run it after
  `git add -N <new files>` before pushing.
- **Delegated agents never `git stash` and never `git checkout -- <file>` to undo a probe.** The stash stack is shared
  across worktrees and sessions, and checkout reverts uncommitted work; both happened once. Probes copy the file
  to the scratchpad and copy it back.
- **A dependency that arrives only transitively will vanish.** CI installed the package with `--no-deps`, so
  `httpx`/`httpcore` came only through `huggingface-hub`; its 2.x release dropped them and `main` went red at an
  unchanged commit (#1188). When the same commit goes green to red, audit the install, not the code. Regression
  guard: `tests/unit/test_ci_workflow_shape.py::test_no_job_installs_the_package_without_its_core_dependencies`.
- **Reproduce where it fails.** Two identical local runs never showed #1103's coverage flip; two CI runs of one
  commit, diffed by `executed_lines`, named it in one step (the fast suite was reaching the runner's real Docker
  daemon).
- **Pin settings to what the platform offers.** #1171 pinned a GitHub setting the repo cannot enable, and the
  nightly stayed red until it was re-pinned from a `--snapshot` (#1205). Snapshot first; a decision that needs a
  setting the platform lacks goes back to the owner.
- **Parallel PRs collide on the next backlog number.** Three did (M39–M41). Take the number from `main` right before
  pushing; `scripts/lint_backlog_ids.py` now fails a duplicate or a dangling citation.
- **A leaked credential is closed by rotating it, then proving it.** Rotate on the machine that checks the key (the
  leader), update every client (peers), then confirm: old key 401, new key 200, no key 401. Redacting the text
  only cleans the tip of history.

## `scripts/suite_at_commit.sh`

The how-to (invocations, the throwaway worktree, and the M47 coverage pre-check) moved on 2026-10-16 to
[docs/agents/ci-testing.md](../agents/ci-testing.md) §4.

## The CLAUDE.md entry, verbatim, before the 2026-10-16 diet (links relative to the repo root)

**[engineering] Faster without lowering the bar (adopted 2026-09-29).** (1) A GATE (provenance stamp, clean flag, ledger/prereg lint, allowance, security boundary) gets an adversarial DESIGN pass on a one-page approach note before code: enumerate every input it consults (config, env, index, refs, filesystem, caller) and how each could lie. When the REVIEWER classifies every finding in a gate's round as fail-closed and in an already-covered class, those go to a follow-up issue, not onto the branch, and no further round is owed; any `src/` fold after the last round still gets a (delta-scoped) round. (2) A fold never invalidates a running suite: snapshot (`scripts/suite_at_commit.sh`, on a commit or `git stash create`) and keep working; one full suite per fold batch, and the last on the exact commit pushed. (3) Owner decisions an issue needs are asked together at its start. (4) While a PR awaits merge, start the next issue in its own worktree. (5) Follow-up review rounds get the exact delta and the stop rule. Every check stays: a different reader, deletion probes, lint-then-push, the full suite. Full practices: [docs/lessons/development-flow-speed.md](docs/lessons/development-flow-speed.md). Regression guard: process invariant — mechanization backlog M20–M24.
