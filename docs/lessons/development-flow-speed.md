# Development flow: faster without lowering the bar

**Adopted 2026-09-29** (owner decision), after a session in which two gate-shaped PRs took most of the
time: M1a (sim `report.json` provenance, PR #999: "two-lens round 1, a security design review, then rounds
2–6", per its body) and the #998 fix branch (four rounds as observed in the session). The bar did its
job: every round found something real. The time went into finding the same class of problem one layer at
a time, and into re-running a ~12-minute suite that each fold invalidated (two full runs were lost to
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
- Run the full suite on a **snapshot**: `scripts/suite_at_commit.sh` (below) checks a commit out into a
  throwaway worktree and runs the suite there. Snapshot uncommitted tracked work with
  `scripts/suite_at_commit.sh "$(git stash create)"` (an unsigned commit object, no ref moved), or commit
  it. Keep folding in the working tree meanwhile. The result names the commit it tested -- that is all it
  proves: it is not evidence of which diff was REVIEWED.
- **One full suite per fold batch, not per round.** Narrow tests plus deletion probes vet a fold. The last
  full suite runs on the exact commit being pushed.
- Narrow test runs alongside a snapshot suite share `~/.maxim`, ports and other global state, the same
  collision risk §3's xdist trial (M22) will measure; keep them small until that trial is recorded.
- Never edit `src/` in a tree whose suite is running. (This cost two full runs in one session.)

## 3. Parallel test runs

`pytest-xdist` is not installed. If the suite is xdist-safe, `-n auto` cuts the ~12-minute run several
times over. Tests that share `~/.maxim`, ports or other global state may collide under xdist; a trial run
names them, and each is either isolated (a `tmp_path` home, a free port) or marked to run serially.
Until that trial is recorded here, the serial run stays the reference. Backlog row M22.

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

## `scripts/suite_at_commit.sh`

```bash
scripts/suite_at_commit.sh            # the suite at HEAD, in a throwaway worktree
scripts/suite_at_commit.sh <commit>   # ... at any commit
```

It creates a uniquely named `.worktrees/suite-<sha>.XXXXXX` under the MAIN checkout (so concurrent runs on
one commit never share or delete each other's tree), prunes stale worktree records first, runs the fast
suite and `test_memory_hub.py` there with an absolute `PYTHONPATH` (so the installed package cannot shadow
it), prints the commit next to the result, and removes the worktree. No `-x`, deliberately: the full
failure list is worth more than the first. It tests a commit by design: untracked files are exactly what
it keeps out of the run.
