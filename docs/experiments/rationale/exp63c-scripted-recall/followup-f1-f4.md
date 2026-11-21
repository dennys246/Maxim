# Exp 63c: confounding lens, scoped follow-up on delta round 1's F1–F4 (2026-11-10)

Reader: a fresh confounding reviewer (not the delta-round-1 reader). Scope: only whether F1–F4 of
[delta1-confounding](delta1-confounding.md) are resolved, on main d40eb4c0 (#1620 merged) with the freeze re-run's
outputs (`replay/`). No new replays were run; the saved outputs were tallied.

## Verdict

All four RESOLVED in the evidence. Two text folds were asked for (one SHOULD-FIX, one NIT set), after which the prereg
can freeze. Freezing does not wait on the judge port, provided the prereg claims only what F3 below allows. The folds
are made in prereg v5.

## F1: RESOLVED

Cue A at turn 3 is decisive under all 16 policies in base, neutral, bare and neutral + no goal word. With neutral or
bare phase-1 text, the only carried record shown at turn 3 is P1.2 (the planted fact), 3 v 0 against P2.1.aut, with no
P1.x.aut present. Across all six output files no decisive turn has a phase-1 AUT record as winner, and under neutral,
bare and no-goal no decisive cue turn has a P1.x.aut record in its shown set: no win depends on text the replay wrote.
The lowest k across the go-rule variants is 1 (b4); no armed control is decisive. The forecast names echo and
starvation.

## F2: RESOLVED

Each cue's goal keys are a subset of its fact's on the new keying: A {guard} ⊆ {guard}, B {guard} ⊆ {guard}, C {guard}
⊆ {guard} ("sleeps" dropped), D {dungeon} ⊆ {dungeon}. Control 6 reads `drap`, which no planted fact shares. A
harness test is owed to repeat the subset check.

NIT (folded): the replay stored answer keys raw ("knife", "small", "grate") while the prereg says they are stored as
`key_tokens`, so completion read silently false on facts B and D. The replay now keys them; completion holds on every
decisive turn.

## F3: RESOLVED for the simulator; the judge half is owed

`fidelity.py` re-runs every `simulate()` call (with and without the carried records) through the shipped
`BioEnrichmentPipeline` over a real Hippocampus: 0 mismatches over 256 calls in each of three variants. Dropping the
top cue hit is caught on 45 of 48 calls, so the check can detect a fault; the goal-fill probe moves only 1 of 48
because the probed policies rarely take that path.

The prereg may claim now: the retrieval and counterfactual traces the forecast rests on match shipped recall; the
goal-fill branch was checked on 12–15 shipped goal-path calls per variant (cite that, not the 1/48 probe). Only after
the 63c judge port re-derives R3d: that the decisive verdicts (every valid tie order, and the graph path) match the
shipped judge; until then the forecast's k values are "simulator-judged". Also folded: the check ran with
`bypass_gate=True` (retrieval once enrichment fires, not the gate); the fidelity outputs are committed; the old "276
calls" line is reworded as a port self-check.

## F4: RESOLVED in the report spec; the replay gives a proxy

Contested is computed from the best pair's margin (replay.py has no explicit flag). Under base, neutral, bare and
neutral + no goal word, 4 of 53 decisive cue turns (4 of 52 under no-goal) are contested: e2 and e3 at cue B (3 v 2),
cpart and cpart2 at cue C (4 v 3). Under the realistic policies (r, p, pX, stuckX) none is, so a PASS will probably
take the "uncontested" ledger wording. Both two-records files add pA at cue A (3 v 2): 5 of 46. Every "3 v 3" line is a
loss, not a win. NIT (folded): define contested over the best pair, and have the harness emit the flag.

## Two AUT records per turn, b4 = 0/4: SHOULD-FIX (forecast wording only; folded)

Two AUT records per turn is not a named go-rule variant and is above 63b's rate (about one), so the go rule stands. The
forecast sentence "at least one decisive cue turn is expected under every policy" was too broad: it is now scoped to
about one AUT record per turn, with the stated echo level at two (b4 k = 0, e1 2/4, e2 and e3 1/4). That failure can
only produce a NOT SHOWN, never a false PASS. Optional: the smoke reports AUT records per turn, and at two or more the
replay is re-run at the observed rate before the run (still a design step, no data).

## Text drift (folded)

The status block and the replay section still described the result as provisional or superseded; they now carry the
freeze numbers.
