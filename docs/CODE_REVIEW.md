# The three-lens code review (run BEFORE the PR opens)

Every sub-plan and every change to `src/` gets a pre-merge review round of **three different readers**, run in
parallel, whose findings are folded into the same branch before the PR opens (CLAUDE.md, "Every sub-plan and every
`src/` change gets a pre-merge review round"; history in [lessons/review-round-discipline.md](lessons/review-round-discipline.md)). A
`scripts/` change gets the round when it imports `maxim.*` or reads or writes records `src/` produces
(`report.json`, JSONL, persisted stores). Each lens reports **BLOCKER / SHOULD-FIX / NIT** with a `file::symbol` and a
concrete failure scenario. The table below summarises each lens; the review prompt carries its full brief.

| Lens | Reads | Asks (summary) |
|---|---|---|
| **Executor** | the code inside the diff's hunks and its path to the production entry point | Does it work on the real path, fail when the mechanism is deleted, and refuse when it should? |
| **Architecture** | the diff against the codebase's design | Is it the root cause, the right layer and owner, true to the invariants and honest in its docs? |
| **Wire integrity** | every line **outside** the hunks that produces or consumes a contract the diff touches | What does each neighbouring producer, consumer, store, hook and learner get, before and after? |

**Boundaries.** Executor owns the code inside the hunks and its path to the entry point. Wire integrity owns every
line outside the hunks that produces or consumes a touched contract. Architecture owns whether a contract should
exist and where. Wire integrity *reports* duplicated logic (probe 9); Architecture decides whether to collapse it. A
lens never drops a finding because another lens might cover it; two lenses finding the same thing is signal.

The experiment design review (four lenses, before a harness is built) is a separate charter,
[experiments/DESIGN_REVIEW.md](experiments/DESIGN_REVIEW.md). Its wiring lens and this one **read** the same
knowledge base, [wiring/](wiring/), before reviewing, and add to it.

## Why a third lens (adopted 2026-10-05)

Two independent passes over the history (the documents, and the issue/PR/git record) produced 30 + 39 table rows (about a dozen
incidents appear in both, and several rows group related incidents). The tables, with provenance, are in [lessons/wire-integrity-review-incidents.md](lessons/wire-integrity-review-incidents.md);
the synthesis is in [lessons/wire-integrity-review.md](lessons/wire-integrity-review.md). What they show, stated as
incidents (every figure is conditioned on the defects that were found; none is a catch rate):

- **Detection latency in the record:** median about 5 months, range about 1 day to 7.5 months (D58, #908, #1085 and
  #840 each ran dead for 5 to 7.5 months).
- **All eight sampled breaking changes that had a recorded executor + architecture round missed the de-wiring in
  their scope** (#411, 5a1dd499, 8a09ae85, 4619e941, 60702417, #1029, the #985 fixes #973/#983/#986, and the 1.2.1
  pairing PRs; 8a09ae85's round covered its results rather than its code, and #1029 fixed one site of N). In #908 the
  review folds *added* two of the three vacuous save calls; in #851 both lenses saw the fragile guard and folded a doc
  note. This is a sample of breaking changes: it shows the gap exists, not how often rounds catch de-wiring.
- **The catches that did happen came from readers looking outside the diff:** an executor reader at code next to an
  unrelated change (D37, D77, D79, #972, #851, #873), and fixes whose reader found an older defect next door
  (#839 → #840/#841, #1083 → #1085, #971 → #982; #1047's run exposed #1052). One fix verifiably *created* the next
  de-wiring, caught by the following PR's round: #973/#983/#986 → #985.
- **A blind backtest** (2026-10-05): a fresh reader applied this charter, cold, to 4619e941 and 60702417 as of their
  own dates, and found #908's load/save split and #851's leaked-entry guard, plus defects the issues did not name.
  Its gaps became probe 10, probe 6's runtime state, probe 1's no-op caller rule, probe 8's constructor list and the
  off-path claim rule.

## The wire-integrity lens

### Its method is keyed to contract types, never to named systems

So it covers bio systems that do not exist yet and wirings nobody has built. Whatever the system, its wiring is
made of these contracts:

| Contract type | Producer side | Consumer side |
|---|---|---|
| **Call** | a function, method or class: signature, return shape, defaults, exceptions | every call site, including dynamic ones (registries, dispatch tables, `getattr(obj, name)`) |
| **Payload / field** | every constructor of a dataclass, dict or message that sets a field or key | every reader of it, including `getattr(x, "f", default)` and string-keyed dicts |
| **Vocabulary** | a closed set of strings shared across modules: event types, relation types, tier and mode names | every place that compares or switches on them |
| **Bus / event topic** | every publisher of a topic or event type | every subscriber, and its preconditions (identity, polarity, timing) |
| **Builder / factory** | every construction site of a type and the collaborators it passes | everything that assumes a collaborator is wired, and anything that later replaces the built object |
| **Persisted state** | the save path, key set and format | the load path, the clear path, every restore site and every gate on it |
| **Cross-process / wire format** | a bundle, export, HTTP payload, or a `report.json` / JSONL field | the importer, the peer, the `scripts/` verdict writer or harness that parses it |
| **Gate** | an env var, config key, default, constant, set membership, **or a predicate over mutable runtime state** (a collection's emptiness, a flag another path sets) | every branch that reads it: can it become unreachable, permanently on, or stuck? |
| **Operator surface** | a CLI flag, `maxim config` key, or YAML field (robot, embodiment, scenario) | the code that honours it, and every place that writes it |
| **Identity** | what stamps the key: `agent_id`, node id, entity class, cluster key | every lookup by that key end to end, including what happens on `None` or a default |
| **Lifecycle** | who calls start, tick, end, connect, shutdown, register, push, and in what order | everything that assumes the matching end ran |
| **Execution context** | the thread, event loop, `ContextVar` or lock a producer writes from | the consumer that reads in another context or at another time |
| **Learning signal** | every component that observes an outcome (success, failure, neutral, pain, reward) | every learner that claims to learn from it: does each value reach it, with identity and polarity intact, through the production entry point? |
| **LLM-facing text** | a prompt, annotation, tool description or schema | the loop and tools that must honour what the text promises |
| **Claims about the wiring** | a brief, an invariant, a docstring, a release note or user doc saying a thing is live | the code: does the claimed path exist, and is the invariant's guard more than the call itself? |

**For a new system or a new wiring**, the reviewer starts by writing this table for it: its producers, consumers,
stores, hooks, signals and claims, and the production entry point each is reached through. A store is a mechanism
only when it is **written → keyed → read → acts** ([wiring/engram-formation.md](wiring/engram-formation.md) §1).
Written and never read is a record; read and never written is a dead branch. Anything the table cannot place gets a
finding, or a `Dormant since` marker with a caller scan.

### The output: a producer → consumer table, not prose

For **every contract the diff touches** (adds, removes, renames, re-types, re-defaults, moves, re-gates, or newly
reads), one row:

| Contract | Producers (before → after) | Consumers (before → after) | What each consumer gets, before → after | Probe |
|---|---|---|---|---|

**Search scope:** `src/`, `scripts/`, `tests/`, `scenarios/`, the YAML asset directories, `.github/workflows/` (CI greps
name symbols), `pyproject.toml` entry points, and the claim surfaces: `docs/agents/` briefs, `CLAUDE.md`,
`CHANGELOG.md`, `docs/user/`, `docs/wiring/` and `DECISIONS.md`. Search string literals and registries as well
as symbol names. Mark test-only producers and consumers: a test-only caller has masked zero production callers
repeatedly (D58, D43, #908).

**An empty table** is valid only when the diff changes function bodies alone: no `def`, `class`, field, constant,
default, string key, env var, config key, topic, return shape, return value's meaning (success, failure, `None`),
raised exception, shared-state write, or call to an in-tree callee is added, removed or changed. State it as the list of changed functions from the hunks, each with the search that found no other
reader.

**A claim is a contract too.** A result in a commit message, PR body or release note that was measured on a PoC,
script or test path the production entry point does not take is a finding (4619e941's "NAc bias 0.015" came from the
PoC script).

### The ten probes (each from a recurring shape in the history)

1. **Connected.** Every new or changed symbol, field, key or topic has a production producer AND a production
   consumer, both on the shipped entry point. A caller of a mechanism that does nothing under the production config
   is **not** a connection: a fix for a zero-callers finding on a save, flush or publish must come with probe 5's
   round trip, or the fold adds a vacuous call (what #908's review folds did). (Rank 1: 1.2.1, #909, #817, D43, D45,
   D9, #822, #1084.)
2. **Every site.** Every construction site of every type the diff touches passes the new collaborator, and the object
   you wired is the one that survives (no later swap, no second instance built elsewhere). (Rank 2: D77/D79/D86,
   D41/D42, PainBus, MemoryHub, #972.)
3. **Call contract.** Every caller of a changed signature, across `src/` AND `scripts/`, still matches it, and no call
   is wrapped in an `except` or `getattr(..., default)` that would hide a mismatch. (D58, #840, #841, D60, #845, D68.)
4. **Outcomes stay distinct.** Every outcome value (fail, neutral, unreached, `None`) reaches the consumer as itself;
   no fallback returns success; no learner loses a polarity. (D53, D56, #870, #873.)
5. **Persistence symmetry.** Load, save and clear derive the path and key set from one source; every gate (fresh,
   load, restore) is applied at every restore site; nothing re-initialises after a restore. Prove it with a round trip
   through the production builder that goes red when the path line is deleted. Run this before probe 1 for any store.
   (#908, D42, D28, #972, #982, #939.)
6. **Gates.** For every changed default, constant, set membership, env gate or runtime-state predicate, and every
   **existing** default a new branch now reads (`persistence_path=None` made #908's save a no-op without being
   changed): which branches read it, can any become unreachable or permanently on, and what clears the state it
   tests? (#1085, #889, #596, #851, Exp 60's default autonomy, D37.)
7. **Round trips.** For every returned object, merged dict or payload: does the key set survive, and does any caller
   discard part of it? (D43, the `nac_merge` key omission, discarded side effects.)
8. **Identity end to end.** List every production constructor of the payload and check each sets the identity field;
   then follow the key from producer to consumer, including what happens on `None` or a default. (D54/D57: none of
   seven `ReactionContext` sites set `agent_id`; Exp 32 Bug A; per-agent stashes; #1042's tier namespace.)
9. **One seam.** Is the logic duplicated elsewhere (a second dispatcher, a prompt vs the executor, an N-way vs a
   pairwise version, two resolution paths)? Report it; Architecture decides the collapse. (D82, D85, D32, sim-n-ctx,
   #1042's narrator paths, #1093.)
10. **Paired lifecycle.** For every start, register, push or open, list every exit path (success, failure, exception,
    early return, a signal dropped by a cooldown or refractory gate, a collaborator never wired) and show the
    matching end, pop or close runs on each. The side that owns the invocation owns its retirement. (#851, D41,
    #982, the decay ticks.)

Every probe that finds a silent path proves it **by deletion**: delete the mechanism and show the guard goes red
(CLAUDE.md, "prove a guard by deleting the mechanism").

### What it is not

Not a second architecture review: it does not judge the design, it maps the neighbours. Not a second executor review:
it does not re-run the diff's own path, it asks what *other* code now receives. See **Boundaries** above.

### Its findings feed the knowledge base

A wiring finding that generalises (a new contract type, a recurring shape, a seam that keeps breaking) adds or updates
an entry in [wiring/](wiring/). A shape the lens catches twice becomes a candidate for a structural fix, preferably
the one that repeatedly worked: make the collaborator a **required keyword-only** parameter, so forgetting it is a
`TypeError` (`build_executor(pain_bus=)`, `build_memory_hub(load_persisted=)`, `geometry=`, `agent_id`), per
CLAUDE.md's "push silent-no-op invariants into types".

## When each lens runs

- **All three** on every sub-plan, every `src/` change, and every `scripts/` change in scope (above).
- **Delta rounds** after a fold: the lens whose finding was folded, plus wire integrity on every `src/` or in-scope
  `scripts/` fold unless the fold is test-only or docs-only. The wire lens may return the empty table; the author does
  not decide that the fold touches no contract.
- **None** for a docs-only touch-up that claims nothing new.

## Mechanization

A rule followed by attention is where slips land (CLAUDE.md, "Enforced, or on the backlog"). Mechanization backlog
**M37** ([plans/outstanding.md](plans/outstanding.md)) holds the scans that would take most of this lens's search off
the reader, and the check that the lens ran at all. The lens stays a reader's job either way: outcome polarity,
object swaps, lifecycle exits and duplicated-logic drift need judgment.
