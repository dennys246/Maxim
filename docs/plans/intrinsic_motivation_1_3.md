# Intrinsic motivation — curiosity/competence as learned reward (1.3 kickoff, DRAFT)

**Status: DRAFT / co-design (opened 2026-09-08, alongside `survival_world_1_3.md`, while
Exp 57's confirmatory ladder runs).** A sibling research line to the survival benchmark —
it shares the mining world but NOT the survival claim. This is the brainstorm a
pre-registration is drafted from *after* Exp 57 lands and the mechanism below is built; it is
not itself a prereg yet.

## The thesis

Agents are exploratory because novelty and competence-gain are intrinsically rewarding — you
try new things, and *succeeding at something you couldn't do before* feels good; once
mastered, it stops (learning to drive at 16 vs. the commute today; the reward returns in a
new city). 1.3 asks whether giving Maxim that drive — **novelty → exploration → competence →
salience → (maybe) hierarchy** — produces broader, more autonomous learning, and whether the
resulting *wants* are shareable through the Oasis fabric.

This is distinct from every prior experiment, which learns from a **teacher's** contingent
reward. Here the reward is a property the **agent** computes about its own experience.

## Honest state of the machinery

- **Novelty → exploration: WIRED (as selection, not value).** The explore policy nudges
  selection toward untried tools with `bonus = weight / (1 + visits)` — novelty that decays
  with use. But it is a *selection nudge*; it does NOT write `reward_bias`, so novelty does
  not currently ACCRUE as learned value. The curiosity *reward* is the thing to build.
- **Acquire → unlock a tool: WIRED and game-native.** Acquiring an acquirable reparents +
  registers affordances (`tool_bridge`); crafting/tools are real game mechanics.
- **Success signal: WIRED.** A positive tool outcome yields `causal_pos` confidence
  (`tool_dispatch` → NAc causal links).
- **Competence → salience → "want more"; hierarchy: NOT wired.** The reward that turns
  competence-gain into durable salience, and any abstraction that would let it organize into
  a hierarchy, are the builds (the latter is the same missing primitive R1 identified).

## The reward equation (the core design)

Both inputs already exist — **success** (`causal_pos` outcome confidence) and **novelty**
(the explore bonus). The question is how to combine them into the small bounded reward-bias
the curiosity term writes. Three formulations of rising fidelity:

| # | form | reads |
|---|---|---|
| (i) | novelty only | rewards trying novel things (even if you fail — flailing) |
| (ii) | **success × novelty** | rewards *succeeding at something still novel*; decays to ~0 once mastered. Computable today. |
| (iii) | **learning progress** — reward ∝ Δsuccess | rewards *getting better*; mastered → no Δ → no reward; new context → new Δ → reward. Most bio-faithful (Oudeyer-style competence-gain IM). |

**A product, not a sum:** mastered driving is *successful but unrewarding*, so `success +
novelty` wrongly keeps paying out; only the product (ii) / derivative (iii) collapse once
routine. This is the design constraint the car analogy encodes.

**Context-conditioned novelty (the "new city" clause):** key the visit/novelty count on the
`(tool, cluster)` pair, not the tool alone, so a mastered tool RE-IGNITES in a novel
situation cluster. Rides the existing cluster machinery.

### The pre-registered question (draft)

**Does a curiosity/competence reward change behaviour, and which form — `success × novelty`
(ii) vs learning-progress `Δsuccess` (iii) — better predicts exploration and learning?**
Arms: **off** (ablation — no curiosity reward), **(ii)**, **(iii)**. Dependent measures
(draft): breadth of tools acquired/used, coverage of the material field, and whether
interest decays with mastery and re-ignites in novel contexts (the signature of a real
learning-progress drive). "off" is the load-bearing control: if (ii)/(iii) don't beat it,
the mechanism buys nothing and ships as a null.

## The build (ride existing infra — Principle 3)

The minimal build extends the explore policy so it writes a SMALL bounded `reward_bias` from
`success × novelty` (v1) — and a variant that writes `Δsuccess` (v2, tracking a per-`(tool,
cluster)` success EMA). No new subsystem; a value-write on signals already computed. Novelty
becomes per-`(tool, cluster)`.

## Discipline (non-negotiable)

- **New mechanism → `[engineering]` first**, graduating to `[behavioral]` only when the
  experiment above earns it (the two-tier invariant-tracking principle, CLAUDE.md
  §"Working principles for new mechanisms").
- **D1 line, drawn precisely.** A *general* curiosity/competence reward is a legitimate
  mechanism hypothesis (why agents explore) — fine to build and test. Inventing a *specific*
  "gold = +reward" to make one behaviour look successful is outcome-gaming — not fine. Test:
  the reward is a property the AGENT computes (its own novelty/competence), never a bespoke
  world signal handed to a target behaviour, and never TUNED to pass a benchmark.
- **Confound guardrail vs survival.** Curiosity reward must NOT silently power the R3
  survival benchmark — else "it survived" confounds "learned the contingency" with "curiosity
  wandered it into food." It is its OWN line here, or a DECLARED ablation arm in R3, never an
  undeclared default.
- **Front-gate scope pressure:** it rides the explore policy + causal outcomes + clusters —
  no new subsystem. Kept as one value-write, not a new bus/bridge.

## Connection to a finding Maxim already has

Learning-progress reward is the intrinsic-reward form of the Exp 37/38/40 **Goldilocks zone**
(substrate signal appears only where priors leave headroom): reward peaks in the
learnable-but-not-yet-mastered band — the same shape those experiments found. This rhymes
with the architecture rather than bolting on.

## Test bed + the hierarchy stretch

- **World:** the **mining classroom** in [survival_world_1_3.md](survival_world_1_3.md) — a
  game-native local block census, veins of varyingly novel/rare materials, `mine_block` as
  the operant acquire, materials tied to game-native uses so salience is earned. Novelty,
  competence, and salience are all exercised there.
- **Hierarchy (stretch, 1.3+):** "does acquired value/competence self-organize into a
  hierarchy of materials/skills?" needs a hierarchical / situation-kind representation the
  flat cluster substrate lacks — the SAME primitive R1 flagged for spatial generalization.
  Build that one channel and both spatial farming AND material hierarchy unlock.

## Open questions

1. v1 = (ii) `success × novelty` now, with (iii) `Δsuccess` as the comparison arm? (Lean: yes.)
2. Context-conditioned novelty (`(tool, cluster)`) from day one? (Lean: yes — else no re-ignition.)
3. Competence = successful tool *use* (empowerment) as the success signal, vs raw unlock
   events? (Lean: use, grounded in outcomes; unlock is a novelty spike, already captured.)
4. Bound + decay constants for the reward-bias write (kept small so curiosity augments, never
   dominates, the learned/operant signal).

## Sequencing

Finish Exp 57 → ship 1.2 → 1.3: build the curiosity reward-write (v1 ii + v2 iii,
`(tool,cluster)` novelty) as `[engineering]` → run the off/(ii)/(iii) comparison in the
mining world → if earned, graduate + fold into the shared-want fabric; the hierarchy stretch
waits on the R1 abstraction channel.
