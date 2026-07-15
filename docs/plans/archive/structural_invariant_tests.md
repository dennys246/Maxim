# Structural Invariant Tests — Test-Discipline Companion to the Regression-Guard Convention

> **ARCHIVED (2026-07-15 plans audit):** ✅ ALL 3 STAGES SHIPPED — Stage 1 statistic-shape tests (PR #279, `tests/unit/test_statistic_shapes.py`), Stage 2 trajectory invariants (PR #280, `tests/substrate/test_trajectory_invariants.py`), Stage 3 multi-agent marker + CI lint (PR #281, `scripts/lint_multi_agent_marker.py`, wired in test.yml). The `Drafted` header below is stale.


**Status:** Drafted 2026-05-27. Ships as three separate PRs in stage order. Companion plan to the [behavioral graduation candidates](../behavioral_graduation_candidates.md) discipline.
**Triggered by:** the regression-guard convention work (PRs #274/#275/#276/#277) made invariants auditable by grep. Three classes of bug shape are still only caught after-the-fact (e.g., Wire 1 statistic degeneracy, P4 multi-agent silent-merge, sequential drift across operations). Lifting them into the test suite turns "remember to test this" into structural enforcement.

## Front-gate scope pressure (retroactive)

Added 2026-05-27 per CLAUDE.md Principle 3.

**Question:** does this need to be its own mechanism, or can it ride on existing infrastructure?

**Existing infrastructure surveyed:**

| Candidate | Why insufficient (or sufficient) |
|---|---|
| `tests/unit/` + `tests/substrate/` + `tests/integration/` | **Already the right home** — Stages 1/2/3 each land in one of these existing directories. The new pieces are *test classes + a marker + a CI lint*, not new test infrastructure |
| Existing regression-guard CI grep lint ([.github/workflows/test.yml](../../.github/workflows/test.yml)) | **Already the right lint mechanism** — Stage 3's multi-agent marker lint rides on the existing CI lint job alongside the regression-guard lint |
| `pytest.mark` registration in `pyproject.toml` | **Already the right marker registration site** — Stage 3 adds `multi_agent_modes` + `single_agent_only` to the existing `[tool.pytest.ini_options].markers` table |
| Roy harness | Catches the *result* of these bug classes (behavioral regression) but is expensive and slow. Structural unit tests catch the *cause* cheaply. Complementary, not duplicative |
| Property-based testing (Hypothesis) | Plausible expansion, but adds a dependency and authoring overhead. Stage 1's parametrized cases are simpler and directly maps to the Wire 1 lesson |

**Verdict:** could-ride-on-existing — the discipline rides entirely on existing infrastructure. Tests are *content* within an existing test framework, not new mechanism in the Principle 3 sense. The three stages add net-new test cases, a marker, and a CI lint — all plumbed through existing pytest + CI + marker-registration surfaces.

**Specific reason:** the gap is content (these three classes of structural check have no current test surface), not infrastructure. Pytest + CI + marker system already handle every plumbing concern. Tying tests to existing directories + workflows minimizes the surface area of the discipline.

## Why these three, in this order

| Stage | Catches | Cost | Specificity | Why this order |
|---|---|---|---|---|
| 1 — Statistic-shape tests | Wire 1 class: key-embedded values produce degenerate statistics on binary-alternating input | ~100 LOC | Highest — maps one shipped lesson directly to one structural check | Cheapest + most directly tied to a known incident |
| 2 — Scripted-action trajectory tests | Sequential drift across ticks: tier-progression direction, reward_bias bounds, decay non-negativity, orphan eligibility traces | ~250 LOC + harness | High — covers a class Roy is expensive to catch | Builds on Stage 1 fixtures; bypasses LLM mock-quality trap |
| 3 — Multi-agent marker + fixture | P4 class: `agent_id` silently merges per-agent state | ~150 LOC + CI lint | High — directly enforces the P4 wiring rule | Touches existing tests, lands last to minimize merge friction with stages 1+2 |

Each stage is its own PR. Stages 2 and 3 can swap order if Stage 2's harness design takes longer than expected.

## Stage 1 — Statistic-shape tests

**Goal:** for every accumulator field in the codebase, assert that binary-alternating inputs produce a non-degenerate statistic. Catches the Wire 1 class (CLAUDE.md L37) structurally: "if your statistic accumulator's KEY embeds the very dimension you want the statistic to vary over, the per-key statistic is structurally 0 on every binary-input case."

**Accumulators in scope** (verify exhaustiveness before shipping):

| Accumulator | Location | Input shape | Expected non-degenerate signal |
|---|---|---|---|
| `NAc._event_outcome_welford` | [src/maxim/decisions/nac.py](../../src/maxim/decisions/nac.py) | Stream of reward values per `(agent_id, event_signature)` | Variance > 0 after ≥2 alternating values |
| `NAc._reward_bias` | [src/maxim/decisions/nac.py](../../src/maxim/decisions/nac.py) | `(signature, reward)` pairs | Bias differs across distinct signatures with distinct reward histories |
| `MemoryRecord.promotion_pressure` | [src/maxim/memory/store.py](../../src/maxim/memory/store.py) | Context-diverse recall events | Pressure grows when contexts differ; stagnates on identical-context recalls |
| `CausalLink.variance_estimate` | [src/maxim/decisions/causal_link.py](../../src/maxim/decisions/causal_link.py) | Reward signal stream — but note Wire 1's KEY trap | If still on `CausalLink`, surfaces as structurally 0 (the Wire 1 fix moved it to NAc); test should DOCUMENT the moved-statistic invariant, not re-introduce the bug |

**Test shape (one parametrized class):**

```python
class TestStatisticShapes:
    @pytest.mark.parametrize("accumulator_case", [
        welford_case,
        reward_bias_case,
        promotion_pressure_case,
    ])
    def test_non_degenerate_on_binary_alternating_input(self, accumulator_case):
        ...
```

Each `*_case` is a small dataclass with: `setup_fn` (returns fresh accumulator), `feed_fn` (signature-tolerant input drive), `assert_non_degenerate_fn` (the statistic-specific threshold).

**Lives at:** `tests/unit/test_statistic_shapes.py` (new file).

**Pass criterion:** all parametrized cases pass; non-degenerate threshold met.

**Honest caveat:** binary-alternating is one specific degenerate-shape probe. A statistic could be non-degenerate on alternating input but still degenerate on production input shapes. This test catches the Wire 1 class specifically; broader degenerate-shape coverage requires more cases.

## Stage 2 — Scripted-action trajectory tests

**Goal:** assert mechanical invariants across N-turn agent execution without paying Roy's LLM cost. Catches sequential drift: the interaction between operations over time, where individual operations are already unit-tested but their combination isn't.

**Design choice: scripted actions, not mocked LLM.** A deterministic mocked LLM sounds simple but the agent loop has many LLM call sites (action selection, planning, narrator, AUT, deliberation, Acting Coach). A mock returning garbage exercises error paths rather than happy paths. Instead, drive the agent loop with **pre-baked action sequences** that bypass LLM calls entirely. Lower complexity, no mock-quality false-negative risk.

**Trajectory invariants to assert:**

| Invariant | Check shape | Reference |
|---|---|---|
| Memory tier progression is one-way | After N turns, no memory transitioned in the wrong direction; `TierTransitionError` never raised in production code paths | CLAUDE.md L115 |
| `NAc._reward_bias ∈ [0, max_reward_bias]` at every tick | Sample `_reward_bias` map after each tick; assert bounds | CLAUDE.md L157 |
| Decay non-negativity | After `decay_eligibility` + `decay_reward_biases`, no trace value > pre-tick value | CLAUDE.md L159 |
| No orphan eligibility traces | Every trace at tick N has either a corresponding event in the last `temporal_window_seconds` OR has been pruned | CLAUDE.md L160 |
| Promotion pressure monotonicity | `promotion_pressure` only grows or decays; never jumps | CLAUDE.md L153 |

**Harness shape** (single fixture, reused across invariant tests):

```python
@pytest.fixture
def scripted_agent():
    """Agent loop driven by a deterministic action queue, no LLM calls."""
    # build agent with bio-stack
    # replace `propose_action` / similar with queue.pop()
    # yield trajectory_recorder
    ...
```

Trajectory invariants are assertions over the recorded trajectory, not over single operations.

**Lives at:** `tests/substrate/test_trajectory_invariants.py` (new file, matches existing `tests/substrate/` pattern).

**Pass criterion:** all invariants hold across at least three different scripted scenarios (smoke / drift / failure-recovery).

**Honest caveat:** scripted actions skip the real selection logic. This is fine for mechanical-invariant tests but means trajectory tests do NOT validate that bio-systems produce the right actions — that stays Roy's job.

## Stage 3 — Multi-agent marker + fixture (scoped)

**Goal:** turn the P4 multi-agent rule (CLAUDE.md L43 — "any per-agent runtime stash MUST be a `dict[agent_id, value]` from day one") into a forcing function on every new test that touches per-agent state.

**Scoping pushback against blanket-fixture:** parametrizing every NAc/Hippocampus/MemoryHub test under three modes triples the test count for those modules. Some tests are genuinely single-agent and shouldn't run 3×. Marker-based opt-in is lighter than blanket-autouse, still catches the silent-merge class structurally.

**Marker definition:**

```python
@pytest.mark.multi_agent_modes
def test_nac_reward_bias_keyed_per_agent(...):
    """Auto-parametrized over: single_agent, two_agents_shared_instance, two_agents_isolated_instances."""
```

**Three modes (documented in fixture docstring loudly):**

1. **`single_agent`** — baseline; one agent, default path
2. **`two_agents_shared_instance`** — TWO agents sharing ONE bio-system instance. **Not a production case** — this is a silent-merge tripwire. Production uses separate instances per `AgentFactory`. Test must demonstrate `agent_id`-keyed state isolation despite shared instance.
3. **`two_agents_isolated_instances`** — TWO agents with separate bio-system instances. Production case. Isolation is structurally guaranteed by separate objects; the test confirms attribution flows through correctly.

**CI lint addition (in same PR):** any new test in `tests/unit/test_nac*.py`, `tests/unit/test_hippocampus*.py`, `tests/integration/test_memory_hub.py`, or any new test whose body touches `_temporal_anchors`/`_reward_bias`/`_event_outcome_welford`/per-agent stash fields MUST declare `@pytest.mark.multi_agent_modes` OR `@pytest.mark.single_agent_only` (explicit opt-out). Lint script lives at `scripts/lint_multi_agent_marker.py`; wired into `.github/workflows/test.yml` `lint` job alongside the existing CLAUDE.md regression-guard lint.

**Lives at:**
- Fixture: `tests/conftest.py` (or `tests/multi_agent_fixtures.py` if conftest gets crowded)
- Marker registration: `pyproject.toml` `[tool.pytest.ini_options].markers`
- CI lint: `scripts/lint_multi_agent_marker.py`

**Pass criterion:** the existing tests covering per-agent state (≥5 known sites per the P4 lesson) annotated with the marker and passing under all three modes. CI lint rejects new tests touching per-agent state without the marker.

**Honest caveat:** the lint can't perfectly detect "touches per-agent state" — it's a heuristic on imports + field references. False positives are possible; treat the marker requirement as a forcing function, not an unbreakable rule.

## Out of scope (for now)

- **Trajectory tests with real LLM mocked.** Reframed to scripted-action above per the design choice. Real-LLM-mock trajectory tests stay deferred unless scripted-action coverage proves insufficient.
- **Property-based testing (Hypothesis).** Plausible expansion of Stage 1 / Stage 2 once basic shape tests are in place. Not in initial scope.
- **Fuzz testing.** Bio-pipeline fuzz is interesting but expensive to design well; not in scope until the structural-invariant baseline is solid.
- **Performance / latency regression tests.** Different discipline (benchmarking, sibling to graduation candidates).

## Cross-references

- [CLAUDE.md "Working principles for new mechanisms"](../../../CLAUDE.md#working-principles-for-new-mechanisms) — Principle 5 (regression-guard / experiment citation) is the convention these tests structurally enforce in code.
- [behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md) — sibling 1.0-gate discipline. Trajectory tests (Stage 2) are mechanical; graduation experiments are behavioral; both pair with benchmarking.
- CLAUDE.md L37 (Wire 1 statistic-shape lesson) — Stage 1 generalizes this into a check.
- CLAUDE.md L43 (P4 multi-agent rule) — Stage 3 enforces this on new tests.
- CLAUDE.md L115, L153, L157, L159, L160 — trajectory invariants assertable in Stage 2.
- [tests/substrate/test_sem_pain_cascade.py](../../tests/substrate/test_sem_pain_cascade.py) — existing pattern for substrate-level integration tests; Stage 2 lives alongside.

## What success looks like

After all three stages ship:

- A new accumulator field added without thinking about the Wire 1 trap → Stage 1's parametrized test fails OR the author proactively adds a `*_case` entry. Either way the trap is surfaced before merge.
- A new bio-system change that breaks `_reward_bias` bounds across multi-turn execution → Stage 2's trajectory test catches it without needing a Roy run.
- A new test on NAc per-agent state that forgets `agent_id` keying → Stage 3's marker requirement forces the test author to think about the multi-agent shape.

Combined with the regression-guard discipline already shipped, the codebase has structural enforcement at four layers:

1. **Tag**: `[engineering]` / `[behavioral]` (Principle 1)
2. **Cite**: `Regression guard:` / `Roy experiment:` (Principle 5)
3. **Test**: structural-invariant tests (this plan)
4. **Validate**: Roy / graduation candidates ([behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md))

Each layer catches different failure modes. None is sufficient alone.
