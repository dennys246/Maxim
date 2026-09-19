# pymaxim 1.3.0 — "Oasis-2"

**Released 2026-09-19 (UTC — PyPI `upload_time`).** `pip install --upgrade pymaxim`

1.2 showed that a want a teacher put into one agent can travel to another. 1.3 takes the teacher
out. The reward now comes from the world itself: on a live Minecraft server (Paper 1.20.4), with no
language model anywhere in the action path, an agent that feels air-hunger underwater learns a fear
of that situation — and a second agent that never felt the pain inherits the fear and acts on it
the first time it is submerged. Minecraft here is an instrument, not a demo: a world Maxim does not
control, with real pain, real relief, and states the agent can be measured in.

## The two claims

**Exp 60 EARNED — anticipatory drowning-avoidance.** An agent held underwater until air-hunger
pain, then rescued, carries a fear keyed to the underwater situation (the Wire-4 cluster fear,
written by the pain bus onto the world cluster co-active at the pain). On later submersions it
leaves the water *before* the pain would fire. Its yoked twin — same pain exposure, fear
subscriber detached — never does. Five seeds per arm, six pain-free probe placements each: FEAR
1.0 against ABLATED 0.0 on every seed, exact permutation p = 1/252 (the floor for five against
five); the fear is specific (water −1.0, shore 0.0). Latency to air after training: median
1.72 s, all inside the 4.34 s window before pain.
[Prereg + outcome](../experiments/exp60_drowning_avoidance_prereg.md).

**Exp 61 EARNED — the headline: a survival fear transfers between agents.** A donor learns the fear
the hard way and exports its substrate through the shipped signed-bundle path. A fresh receiver
ingests it — discounted ×0.75 at the ingest bound, because vicarious fear is real and weaker than
direct — reboots, and on its first loop-live submersion leaves the water, never having felt the
pain. 12/12 transferred receivers against 0/24 isolated, 0/12 where the donor's cluster shipped but
its fear did not, and 0/24 where the fear shipped without its cluster: Fisher one-sided
p = 8.0 × 10⁻¹⁰, all six frozen gates PASS, one code hash, zero refusals.
[Prereg + outcome](../experiments/exp61_shared_fear_prereg.md).

## The instrument — R3, the survival benchmark

R3 measures, on a depth-calibrated frozen gauntlet, what a carried drive buys at the one moment it
matters: one unrescued submersion per fresh agent. Five arms × 12 agents, time to air:

| arm | what it carries | median time to air |
|---|---|---|
| A | the innate health reflex only | 28.0 s |
| B | learns the fear there, in the water | 8.6 s |
| C | carries a fear it learned earlier | 3.2 s |
| D | carries a fear it received (Exp 61's receiver) | 3.1 s |
| E | trained like C, fear detached | 28.1 s |

Every agent in every arm survived — by design, with regeneration on, survival is a ceiling. What
the drive buys is the cost it removes: about 25 s of latency, about 11 health points and about
22 s of oxygen pain. **Not life.** E ≡ A (the exposure alone buys nothing), D ≡ C (the received fear
acts like the learned one). R3 is an instrument and a baseline; nothing graduated.

It is also reported with its one post-data change in the open. The frozen report read INCOMPLETE
on two rules that turned out to be facts about the instrument, not the agents: a code-hash rule no
bench could satisfy, and a loop-cadence band that, on 3-second events, measured the tick phase
rather than the cadence. Both were amended after the data, instrument-only, reviewed by two
independent lenses, and disclosed beside the frozen report — the recount moved the carried-fear
median *down*, against the claim, and no contrast changed.
[Prereg, amendments + outcome](../experiments/r3_survival_benchmark_prereg.md).

## Exp 56, re-baselined

The 1.2 headline (a taught want transfers) was re-run on the new platform because its own
re-run triggers fired. Every rate is identical to the 1.16.5 campaign (taught 0.84 raw / 0.80
decisive against isolated 0.22; all four gates and the anti-vacuity kit PASS), with the
Minecraft version measured from the server into every row. One duplicated row from an operator
restart is disclosed, left in the file, and moves no gate.

## What is not claimed

- **"Dark = danger" is BLOCKED at the instrument** (Exp 58): the shipped sensors cannot separate the
  dark situation from the safe one, so no fear can key on it. The contingency earned on water instead.
- **Eat-when-hungry is prior-driven** (R2), not learned.
- **The fear's reach to another pool** is Exp 62 (designed, not run). Nothing here says Maxim fears
  water anywhere.
- **Extinction, scaling, and hive-side promotion** of a shared fear are untested.
- **The survival reflex tier** (planned as Phase 1b) is deferred: the innate health reflex plus the
  learned fear covered every case measured, and the reflex is built only if a hostile's
  onset-to-death window turns out shorter than the loop's ≈ 1 s reaction.

## A correction to 1.2.1

1.2.1 said it "completes the spoken-code device-pairing loop end to end". It shipped the pieces —
the 0.5.0 console contract, the announcer factory, the device speak sink, an audio format fix — but
not the composition: no shipped command wires the announcer, so `maxim serve` still signs in by
token, and pairing needs an embedder that owns both the console and a live robot handle. The 1.2.1
changelog entry is left as published; this is the correction.

## Scope, stated plainly

Every result above is substrate-primary — the drives, the fear, the credit and the selection run
in Maxim's bio-inspired substrate with no language model choosing actions. "Bio-inspired" is an
engineering stance, not a neuroscience simulation. The Minecraft bridge is a development tool and
is not in the wheel; reproducing the experiments needs a checkout, a Paper 1.20.4 server and the
bridge.

## What's next

1.4's working direction is generalization and multi-step credit on the same world: an agent that
has to trade a want against a fear, reach food several moves away, and learn when to come up for
air ([roadmap](../plans/roadmap_1_4.md)). Its name is decided at release from the highest result it
earns. The perception fabric and the microduck wait for a second robot body.

## Upgrading

`pip install --upgrade pymaxim`. Two things change for persisted state:

- **Minecraft world nodes read as a stale geometry on first load.** A gained modality's geometry
  tag now includes the sensors' declared ranges (the `saturation` range changed and the old tag
  could not see it, so stored world nodes silently matched across a changed space). Existing
  world nodes load with a one-line warning; `maxim substrate invalidate --drop-geometry` removes
  them. Interoception and audio are untouched.
- **NAc state and bundles can now carry `cluster_fear`** (the learned situation fear). Bundles
  export it clamped and allowlisted; ingest bounds it, refuses an out-of-allowlist failure mode,
  and applies the 0.75 foreign-fear discount. Pair 1.3 exporters with 1.3 receivers.

Full changelog: [CHANGELOG.md](https://github.com/dennys246/Maxim/blob/main/CHANGELOG.md#130---2026-09-19--oasis-2).
