# The 1.3 survival world — learning reward FROM the game (kickoff, DRAFT)

**Status: DRAFT / co-design in progress (opened 2026-09-08, while Exp 57's confirmatory
ladder runs).** This is the design surface for the 1.3 survival line: R3 (survival
benchmark), R4 (structure formation), and the shared-perception fabric they exist to
measure. It is NOT yet a pre-registration — it's the brainstorm that a prereg is drafted
from once the build list below is real and the world elements are chosen.

## The shift 1.3 makes: teacher-reward → game-reward

Everything through Exp 57 learns from a **teacher** — a harness caregiver delivers
relief-signed operant credit for the right action at the right situation (the Exp 52
mechanism). 1.3's north star is **intrinsic, game-derived reward**: the agent must
*discover* that an action produces survival value from the game's own state (health, food,
damage, light, hostiles), with no teacher naming the target. That is a strictly harder
problem, and it is what "give it an objective to survive" means precisely.

## The load-bearing reality: R2 says the intrinsic loop is NOT wired

R2 (`../experiments/r2_drive_premise_check.md`) returned `PREMISE-NULL`: the world-owned
`health`/`food` drives generate pain but nothing wires drive state to *corrective action*.
So "learn reward from the game" is not a config or a new world — it is a BUILD, and R2's
three breaks are its to-do list:

1. **Corrective-need derivation.** `_read_drive_states` emits raw sensor values; the NAc
   drive prior has no `food`/`health` corrective affinity and name-matches the passive
   `read_*` tools, not `eat`/`attack_nearest`. A deficit must map to a positive corrective
   need on the right affordance (the pattern `cold` already uses).
2. **Measured-relief credit for interoceptive world-owned drives.** `eat`'s `self_effect`
   is credit-withheld on the live body (`drive_credit_withheld`); the measured-relief path
   exists only for exteroceptive/azimuth transitions. Corrective affordances can't
   self-learn until this Phase-2 path is built.
3. **Executable corrective acts.** The bridge `eat`/`attack_nearest` throw on empty
   inventory / no mob, and there's no acquire affordance — so on a void world the
   corrective acts don't exist. The world must actually afford them.

## The second dependency: R1 says there's no cross-cluster generalization

R1 (`../experiments/r1_cross_layout.md`) returned `CACHE-CONFIRMED`: a learned want is an
exact-key `(agent, cluster, tool)` cache with NO similarity/hierarchical readout, so it
fires only at the exact situation cluster it was taught on. **Any spatial / cross-context
learning goal (below) therefore depends on building a generalization channel** —
similarity-weighted bias read across neighbouring clusters, or a coarse "situation-kind"
cluster above the fine one. This is the single most reused 1.3 primitive; several world
elements need it.

## The contract (owner decision D1): game-native pressure ONLY

Never invent a sensor whose purpose is to make a target behaviour rewarding — that
measures the apparatus designer, not the agent. Motivation must come from what the game
already exposes (health, food, light_level, hostiles, oxygen, damage events). Corollary:
a world *mechanic* that creates a situation (mobs spawn in darkness; a day/night tick) is
fair; a bespoke "shelter-goodness" sensor is not.

## The world-elements ladder (brainstorm — by which missing mechanism each needs)

Ordered by how much of the intrinsic loop each requires, so we can sequence from
tractable-on-today's-substrate toward the R4 frontier. "Shareable?" flags whether the
learned want is a target for the 1.3 sharing-fabric claim (one agent learns it, another
ingests it).

### Tier 0 — avoidance / immediate (may ride EXISTING negative-credit path)
Pain forms *negative* causal links (weighted −0.5 in `recommend_action`) — this path is
more wired than the positive corrective-affinity prior R2 found dead. So AVOIDANCE may be
the most tractable starting contingency, distinct from APPROACH.
- **Damage avoidance:** an action taken while a hostile is dealing damage accrues a
  negative link → the agent stops doing it / flees. Immediate, game-native. Shareable.
- **Darkness = danger (avoidance form):** being in a low-`light_level` cluster reliably
  precedes mob damage → negative valence on the dark-situation cluster. Rides negative
  credit, not the corrective prior. *Caveat:* `light_level` read DEAD (0 everywhere) in
  Exp 56 — verify the sensor works before relying on it. Shareable (the headline "share a
  fear").

### Tier 1 — approach / immediate corrective (needs R2 breaks 1+2, and 3 for the act)
- **Eat when hungry:** `food` low → `eat` → relief. The canonical R2 build. Needs the
  corrective-need derivation + measured-relief credit + a food supply in the world.
- **Seek light:** move toward / place a light in a dark cluster to remove the mob threat —
  the APPROACH counterpart of the Tier-0 avoidance. Needs the corrective prior + a
  light-affording action.

**Dining hall — the R2-break-1 flagship (owner idea, 2026-09-08).** The cleanest first
classroom AND the concrete proof that R2's break #1 is closed: a controlled room the agent
enters at LOW health/food (controlled onset — the nursery's controlled-stimulus trick, no
waiting for natural depletion), food on hand. `food`/`health` low → `eat` → measured relief
→ operant credit on (deficit-situation, `eat`). If the agent learns to eat when in deficit
and *not* otherwise, the drive→corrective-action link R2 found dead is alive. Standalone,
buildable the moment the corrective-need derivation + measured-relief credit land — no
generalization, no delay. **Caveat on "transfer":** teaching food-heals-**ME** here does NOT
auto-transfer to feed-the-**wolf** (the husbandry classroom below) — that is other-healing in
a different situation cluster, i.e. the R1 cross-context wall, not a free consequence of the
dining hall. The dining hall proves rung 1; it does not hand you rung 3.

### Tier 2 — spatial / cross-context (needs the R1 generalization channel)
- **"Resource is over there":** a location→resource association that generalizes across
  *similar* locations rather than firing at one exact cluster. Directly the R1 wall.
- **Return-to-safety:** navigate back to a known lit/safe region. Spatial memory + recall.

### Mining / novelty-salience classroom (owner idea, 2026-09-08)
> The **reward mechanism** (curiosity/competence as learned reward — the success×novelty vs
> learning-progress design) is spun out to its own line:
> [intrinsic_motivation_1_3.md](intrinsic_motivation_1_3.md). This section keeps the WORLD
> (the mining classroom) that serves as its test bed.
The agent senses the material blocks around it (a game-native local block census —
nearest-block-of-type, D1-clean) among veins of varyingly **novel** and **salient**
materials (gold, silver, …). Three separable claims of increasing build cost:
- **(a) Novelty-driven exploration — CLEAN, ~wired now.** Does the agent preferentially
  explore *novel* veins, with interest *decaying as it visits/acquires* them? This rides
  the existing explore policy (`substrate_explore_bonus_weight`, bonus = weight/(1+visits)
  — novelty literally decays with visits). Novelty is the agent's OWN drive, not an invented
  world reward, so it's D1-clean. A great early probe of the exploration machinery.
  Design (owner 2026-09-08): each material is **wholly novel on first encounter, decays
  SLOWLY**; **rarity is set by vein frequency** — and rarity ≈ persistent novelty for free
  (a rare material is encountered/visited less, so its explore bonus decays slower). Both are
  world mechanics (vein density), D1-clean, no invented sensor. The **acquire** act is a real
  `mine_block` (an operant cost), so (a) is not pure perception — it already has an operant
  loop, which sets up (b).
- **(b) Salience learned through interaction + TOOL acquisition (owner 2026-09-08) — a NEW
  MECHANISM: competence/curiosity as intrinsic reward.** The owner's hypothesis: acquiring a
  material unlocks a tool the agent can call itself; having more callable tools raises the
  salience of acquiring more — a competence snowball, "why we're exploratory." Two honest
  facts:
  - The unlock half is WIRED and game-native: acquiring an acquirable reparents + registers
    tools (`tool_bridge`), and crafting/tools are real game mechanics — D1-clean.
  - The reward half is NOT wired: the explore bonus is a decaying SELECTION nudge
    (`weight/(1+visits)`), it does NOT write reward-bias, so novelty/competence does not yet
    ACCRUE as learned value. "Novelty provides reward, that's why we explore" is the
    hypothesis, not the current state — building it is a genuine new mechanism (intrinsic /
    competence-based motivation; cf. empowerment/curiosity in the literature).
  - **Minimal build (ride existing infra, per Principle 3):** let the explore bonus (and/or
    a tool-count/competence term) write a SMALL bounded reward-bias — turning the selection
    nudge into value that accrues. Enters as `[engineering]`; graduates to `[behavioral]`
    only when an experiment earns it.
  - **The reward EQUATION (owner idea 2026-09-08 — the car analogy).** Learning to drive at
    16 = rewarding (novel + succeeding); routine driving = not (mastered); a new city = again
    (renewed novelty). This unifies (2) and (3) in ONE term and argues for a specific shape:
    - Both signals already exist: **success** = a positive tool outcome (`causal_pos`
      confidence from `tool_dispatch`), **novelty** = the explore bonus (`weight/(1+visits)`,
      decaying). No new sensors.
    - The analogy argues for a **product, not a sum**: mastered driving is *successful but
      unrewarding*, so `success + novelty` would wrongly still reward it; `success × novelty`
      rewards "succeeding at something still novel" and decays to ~0 once routine. Unlocking a
      tool spikes novelty; using it well provides success — so the product subsumes both (2)
      and (3) without separate mechanisms.
    - Three formulations of rising fidelity — worth comparing, not pre-deciding:
      (i) novelty-only; (ii) **success × novelty** (the v1 — computable now); (iii)
      **learning-progress** = reward ∝ Δsuccess (the *derivative* — reward for GETTING better;
      mastered → no improvement → no reward; new context → new improvement → reward). The car
      analogy actually fits (iii) best, and (iii) is the most bio-faithful (Oudeyer-style
      learning-progress / competence-gain IM). Comparing (ii) vs (iii) is itself an experiment.
    - **Novelty must be CONTEXT-conditioned** (the "new city" clause): key the novelty/visit
      count on the `(tool, cluster)` pair, not the tool alone, so a mastered tool RE-IGNITES
      in a novel situation cluster. This rides the existing cluster machinery.
    - **Connection to a finding Maxim already has:** learning-progress IM is the intrinsic-
      reward form of the Exp 37/38/40 **Goldilocks zone** (substrate signal appears only where
      priors leave headroom). So this rhymes with the architecture rather than bolting on —
      reward peaks in the learnable-but-not-mastered band, the same shape those experiments found.
  - **The D1 line, drawn precisely (this is a mechanism, not outcome-gaming):** a *general*
    intrinsic-curiosity reward is a legitimate architectural HYPOTHESIS about why agents
    explore — fine to build and test. Inventing a *specific* "gold = +reward" to make one
    behaviour look successful is outcome-gaming — not fine. The test: the reward must be a
    property of the AGENT (novelty/competence it computes), never a bespoke world signal
    handed to a target behaviour, and it must never be TUNED to pass a benchmark.
  - **Confound guardrail:** intrinsic curiosity reward must NOT silently power the survival
    benchmark (R3) — else "it survived" confounds "it learned the contingency" with "curiosity
    made it wander into food." So it is EITHER its own experiment line (does curiosity produce
    broader exploration / tool-breadth vs an ablation?) OR a DECLARED ablation arm in R3
    (survival with/without curiosity reward), never an undeclared default.
- **(c) Hierarchy through acquired salience — SPECULATIVE, needs the abstraction build.**
  "A hierarchy of materials forms" needs a hierarchical / situation-kind representation the
  flat cluster substrate does NOT have — the SAME missing primitive R1 identified. So this
  is a fascinating north-star, downstream of the generalization/abstraction channel, not an
  early rung. Captures a real 1.3+ question: does acquired value (and competence) organize
  into structure?

**Cross-cutting note — "intrinsic motivation" may be the most novel 1.3 contribution, and
it is its OWN line.** Novelty→exploration, interaction→competence, competence→salience, and
whether that self-organizes into hierarchy is a coherent research program distinct from the
survival benchmark. It reuses two already-flagged missing primitives (the R1
generalization/abstraction channel for hierarchy; a value-accruing curiosity term for
salience). Treat it as a sibling of R3/R4, sharing the world but not the survival claim.

### Wolf husbandry / taming classroom (owner idea, 2026-09-08) — an entity that regulates the agent's homeostasis
The husbandry north-star realized NATURALLY (not by a hand-wired bond), and the cleanest test
of "an entity adjusts the agent's own homeostasis/entropy." Two rungs at different depths — do
not bundle them, or the hard half rides a mechanism that isn't there:
- **Defense (single-step; buildable after the survival loop).** A wild wolf attacks a hostile
  that is ALSO attacking the agent (the commensal shared-enemy first contact) → the hostile
  deals less damage → the agent's health drive is RELIEVED by the wolf's action → operant
  credit lands (Exp 52 path). The relief is game-native; nothing hand-codes "wolf = good." The
  agent learns to value/stay-near the defender because it mechanically keeps it alive.
- **Upkeep (delayed, multi-step — R4 frontier).** Maintain the ally at a resource cost so it
  keeps defending: pay now, defense later. The SAME delayed-credit primitive crafting needs;
  downstream of the R4 build, NOT free with defense.
- **D1 mechanic correction (verify actuation before designing).** Minecraft wolves have NO
  hunger bar — inventing one is a synthetic mechanic (D1 violation, the R2/instrument lesson).
  Native mechanics giving the SAME learnable loop: **bones TAME** (wild→yours, one-time),
  **meat HEALS** a tamed wolf, tamed wolves auto-attack what attacks the owner. So upkeep =
  "heal the wolf (a resource cost) → it survives to keep defending you," every link a real
  game mechanic — no invented sensor.
- **Action-dependence is load-bearing (the R2 trap restated).** Auto-friendly as a STARTING
  state is fine (it gives the first positive contact), but the LEARNED behaviour must causally
  gate the relief — if the wolf defends regardless of anything the agent does, the outcome is
  action-independent and there is nothing to learn (exactly R2's dead loop). Feeding/healing
  must actually change whether the defender persists.
- **Homeostasis framing:** the wolf becomes an entity the agent learns to maintain *because*
  maintaining it regulates the agent's own state (threat down, at a resource cost) — the
  husbandry vision, formed through the world's physics rather than a hand-wired bond. Breeding
  (population dynamics over time) is a further axis — park it at 1.4+, don't inflate 1.3.

### Crafting classroom (owner idea, 2026-09-08) — the delayed-credit / structure showcase (R4)
Crafting is combinatorial (resource → intermediate → tool → better resource) and self-
generating, so it reads as a natural novelty/salience testbed. But the honest framing: its
VALUE is that it FORCES multi-step delayed credit — gather → craft intermediate (no payoff) →
craft tool (still none) → use the tool to reach a resource you could not before. The reward is
several steps removed from the first action = precisely the R4 primitive that does not exist.
- **The trap (R1 restated):** if crafting is just "discover the fixed recipe set," it is an
  exact-key recipe CACHE with no generalization — R1 all over again. The version worth building
  is the one whose reward is delayed and whose structure (wood→planks→sticks→tools) could
  transfer. So crafting is not a free classroom; it is R4's showcase, built AFTER the delayed-
  credit substrate, not alongside it. A "rich garbage-yard of resources" (owner framing) is a
  good dense discovery space — but Goldilocks applies: too rich → everything novel → no
  discrimination; density must leave prior-headroom.
- **Novelty must ignite-then-fade, not stay flat (correction to "each combo equally novel").**
  Permanently-equal novelty kills the learning signal. Equal-on-first-encounter, yes;
  equal-forever, no — use the learning-progress (v2 / (iii)) form from
  [intrinsic_motivation_1_3.md](intrinsic_motivation_1_3.md), context-keyed on `(craft_action,
  ingredient-cluster)` so each new combo re-ignites and mastered ones fade.
- **Embody the station through SEM** (each recipe an affordance acted on) — the SEM binding is
  the right vehicle, but it rides on the delayed-credit reward, which is the actual build.

### Tier 3 — R4 frontier (delayed, multi-step credit)
- **Farming (the owner's idea):** sense seeds on the ground → (auto-plant next day) →
  crop grows → harvest → eat → relief. A payoff delayed a full day across multiple steps.
  Depends on BOTH the R4 delayed-credit path AND the R1 generalization channel (spatial).
  The auto-plant is a smart scoping move: it removes the motor "plant" act so the test
  isolates the seed→growth→food *association* + spatial + delay. This is the deep end —
  the north-star rung we build toward, not a starting rung.
- **Shelter-building:** `place_block` to enclose before night → prevents damage. Multi-step
  construction with a delayed payoff — R4's canonical case (the "does tick-anchored credit
  reach a delayed construction" question stated in the benchmark doc).

## Why each element is also a SHARING target

The 1.3 thesis is shared perception/wants. Every learned survival want above is a candidate
for the Oasis fabric: agent A learns "dark = danger" the hard way; agent B ingests A's
substrate and avoids the dark on first night. That composition — *survival* wants shared —
is the 1.3 headline, sitting on Exp 56/57's transfer+scale foundation.

## World topology (decided 2026-09-08): ONE world, marked "classrooms"

Curriculum-for-claims, but as **regions of a single world**, not separate world files —
one server/bridge (reuse `~/exp56_server`), one encoder, one spawn; each classroom is a
force-loaded region built by `setup_world` (a dark box, a food area, a farm plot), the way
Exp 56's slot enclosures already are. Two conditions make this sound rather than a confound:

1. **Isolation is PROTOCOL-level, not geographic.** Marked regions do not isolate the
   substrate by themselves — a want learned in classroom A carries in the same agent. For a
   clean per-rung CLAIM, use a **fresh agent per classroom/cohort**, trained only there.
   (Running ONE agent A→B→C is a different, later claim — *curriculum learning*: does A help
   B? — legitimate, but name it as such.)
2. **The situation must be defined by the contingency FEATURE, not the classroom's
   coordinates.** This is the load-bearing insight and it changes the body:
   - Exp 57's `bench57` world channel is **offsets-only** (situations = *locations*) — right
     for Exp 57, wrong here. Survival contingencies are about **state** (dark, hungry,
     threatened), not place. If the world channel encodes the classroom's coordinates, every
     want keys on "danger *at these coords*" and hits R1's cache wall — it won't fire at a
     dark spot elsewhere.
   - So the **survival body senses the contingency features** — `light_level`,
     `hostile_count`/`nearest_hostile_dist`, `food`, `health`, `oxygen` — as its world
     channel, and each contingency should **recur at VARIED positions within its classroom**
     so the learned want keys on the feature (dark), not the location. That variation is a
     free within-world generalization check: does "dark = danger" fire at a new dark spot?
   - Farming (Tier 3) is the exception that re-adds position deliberately — it IS a spatial
     claim, so its body carries offsets too, and it's where the R1 generalization channel is
     exercised on purpose.

Net: one world, marked classrooms, fresh agent per claim, feature-based world channel
(offsets added back only for the spatial/farming rung). One world even lets a want learned
in classroom A be probed in classroom C — the cross-context test, in situ.

**Refinement (portability comes from the BODY, not from moving around).** With a
feature-only world channel, an aversion want is portable BY CONSTRUCTION: "dark" clusters on
`light_level` regardless of coordinates, so the want fires at any dark place — one cave is
enough, no need to vary positions for the dark/hunger rungs. "Varied positions" only matters
when position IS the situation (the farming/spatial rung), which is exactly where the R1
generalization channel is the thing under test. So:
- **Dark-danger classroom = a CAVE** (owner idea): more game-native than a lit room darkened
  by hand — a cave is dark + mob-spawning + enclosed by construction. Keep the MEASURED
  contingency "dark → mob → damage," i.e. let `light_level` (+ hostiles) be the discriminating
  feature, not cave navigation/depth — the feature-only body ensures that.
- **Opening all classrooms and touring ONE agent between them** (owner idea) is a lovely
  setup, but it's a DIFFERENT thing from per-rung isolation: a single agent visiting
  cave→farm→mine tests **curriculum learning** (does surviving the dark help it learn hunger?)
  and makes a live demo. Per-rung CLAIMS still use a fresh agent per classroom. Both are on
  the menu — name which one each run is.

## Minecraft version — build-platform decision (2026-09-08)

The bridge uses **mineflayer `^4.20.0`** and does NOT hardcode a Minecraft version — it
negotiates the protocol from the server it connects to. Exp 56/57 ran on **Paper 1.16.5**
(Java 11–16). Versions are **not freely interchangeable**: the transport layer is flexible
within mineflayer's supported range, but game MECHANICS (wolf taming, crafting recipes, food
values, mob AI) and block/item/entity NAMES differ across versions, so a version move needs a
sensor **port + verify pass**, and it counts as a "Minecraft bridge protocol change" — a
re-run trigger for the Exp 56/57 graduation guards (any reused apparatus, e.g. the shared-want
fabric, must be re-baselined on the new version).

**Requirements for any chosen version:** (1) mature mineflayer support — NOT the newest
release (mineflayer lags fresh versions and is buggy on them); (2) Paper/Spigot in offline
mode with RCON + daylight/mob-spawn control (deterministic classrooms); (3) every sensor
re-verified on that version (the instrument lesson); (4) matching Java.

**Decision (proposed):** the 1.3 world's headline mechanics — husbandry (wolves), crafting,
farming — are richer/cleaner on a MODERN version, so move the 1.3 line to **one modern *stable*
version and commit to it** (target **1.20.1 / 1.20.4**: mature mineflayer support, Java 17;
treat 1.21.x as "verify mineflayer support first"). Eat the one-time sensor re-verify + the
apparatus re-baseline; do NOT run two versions long-term (operational overhead + `~/.maxim`
collision risk). Staying on 1.16.5 is the zero-port-cost alternative but buys simpler mechanics
and an old Java pin — not worth it given husbandry/crafting/farming are the 1.3 headline.

## World vs. substrate: what gets built, and who builds it

A recurring confusion worth pinning: **RCON builds the WORLD; the AGENT builds the substrate;
nothing is hand-built in Minecraft.**
- **The world (blocks + entities)** is assembled by **RCON command scripts** — `/fill`,
  `/setblock`, `/summon`, `/clone`, `/effect` — run by the harness. Reproducible, seeded, no
  manual block placement. A "cave" is just stone + air + darkness placed at coordinates; RCON
  holds no concept of "cave."
- **The substrate (clusters + learned bias)** is the AGENT's, formed from its **game-native
  sensors** as it experiences the world: a cave becomes a recurring region of sensor-space
  (`light_level` low, block-census stone-enclosed, `y_altitude` low, no sky) that the EC
  clusters, and "cave = danger" is a LEARNED bias on that cluster after mobs attack it there.
  There is no "cave" symbol and no `is_in_cave` flag — handing one would be engineering the
  concept (a D1 violation). **Design consequence:** a classroom is only learnable if
  game-native sensors render it SEPARABLE from its surroundings; verify that separability
  through the real encoder BEFORE running (the instrument lesson — and note `light_level` read
  DEAD in Exp 56, so cave-distinctness may have to rest on block-census / altitude / sky).
- **Operator setup (hosting, not building):** stand up a Java Paper server for the chosen
  version (offline mode, RCON enabled + password, daylight/mob-spawn gamerules — the
  `~/exp56_server` pattern), run the mineflayer bridge (`node index.js`), run the harness. The
  only optional hand-work is sketching a layout on a Mac Java client — and even then the real
  classroom is rebuilt by RCON.

## Open design questions (to resolve before a prereg)

1. Is AVOIDANCE (negative-credit, Tier 0) genuinely more tractable than APPROACH on the
   current substrate? A cheap first probe (like R2) could settle it and pick the first rung.
2. Does `light_level` actually read on the live bridge (it was dead in Exp 56)? Verify first.
3. The generalization-channel design (R1's build): similarity-weighted read vs a
   hierarchical "situation-kind" cluster — which, and does it disturb the exact-key
   guarantees Exp 56/57 rely on?
4. R4 delayed-credit: does the existing eligibility-trace / temporal-phase machinery reach a
   one-day-delayed payoff, or is a new credit path required? (The R4 pre-registered question.)
5. World scope: one biome/world with all elements, or a curriculum of worlds by tier?

## Sequencing (proposed)

1.2 SHIPPED (Exp 56 EARNED + Exp 57 PARTIAL, 2026-09-09). Open 1.3 with: (a) the survival-loop
build (R2's 3 breaks) + the **dining hall** (R2-break-1 flagship) + a cheap avoidance-vs-
approach probe; (b) R3 survival benchmark (instrument + frozen baseline, Goldilocks-calibrated)
— **single-step wolf defense** rides here; (c) the R1 generalization channel (unlocks the
dining-hall→wolf transfer + spatial); (d) R4 delayed credit (farming / shelter / **crafting** /
**wolf upkeep**); (e) the shared-survival-want fabric. Breeding is 1.4+. The owner's world
design feeds (b)–(e).
