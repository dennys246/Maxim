# Substrate learns from lived experience (close the write-path in llm-primary + imagination)

> **⏸ DEFERRED 2026-08-27 — Phase 1 ✅ SHIPPED (PR #437, cluster-reward write path closed in llm-primary; Exp 52's relief credit builds on it). Phases 2 (imagined provenance) + 3 (rehearsal) unbuilt; no roadmap owner. **Revive when** imagination is allowed to write to the substrate (the imagined-provenance flag is the hard prerequisite).**

**Status:** DESIGN (2026-07-28). Motivated by the Exp 44 counterfactual: a pre-loaded
substrate faded from "rewarding" to "neutral" within ~27 llm-primary decisions
because the loaded bias could only **decay** — nothing reinforced it. Root cause is
architectural, not experimental. Investigation: three parallel code traces (write-path
mechanism / contamination-and-decay / imagination-and-choke-point), findings inline.

**Front-gate — ride existing infra?** **Yes, entirely.** `runtime/tool_dispatch.py::record_outcome`
([tool_dispatch.py:72](../../../src/maxim/runtime/tool_dispatch.py#L72)) is a single per-action
choke point through which **all four** experience sources already dispatch. No new
mechanism, no new bus — the fix is to *populate a field that's currently `None`* and
*respect one guardrail*. No new plan-doc mechanism per the front-gate rule.

---

## The gap

The NAc cluster-reward substrate has a **write** path and a **read** path:

- **Write:** `record_outcome` → `NAc.update_cluster_reward((agent_id, cluster_id, tool_sig), reward)`
  ([tool_dispatch.py:322](../../../src/maxim/runtime/tool_dispatch.py#L322), [nac.py:2216](../../../src/maxim/decisions/nac.py#L2216)).
- **Read:** Wire-A prompt annotation (`get_agent_tool_biases`) + substrate-primary `recommend_action`.

The write path only fires when `cluster_id` is non-empty. **`cluster_id` is captured
ONLY by `propose_via_substrate`** ([agent_loop.py:1052-1087](../../../src/maxim/runtime/agent_loop.py#L1052)),
which runs only in substrate-primary. In **llm-primary, imagination, and real-hardware**
paths the LLM/agent chooses the action, `propose_via_substrate` never runs, the
`SensorEncoder` isn't even built ([agent_loop.py:1403](../../../src/maxim/runtime/agent_loop.py#L1403),
gated `if aut_mode == "substrate-primary"`), so `cluster_id=None` and the entire
`if active_clusters:` credit block is skipped ([tool_dispatch.py:258](../../../src/maxim/runtime/tool_dispatch.py#L258)).

**Net:** the substrate is *write-only in substrate-primary, read-only-decaying everywhere
else.* An agent's own lived experience never reinforces it. A pre-loaded/persisted bias
just bleeds off (pure-decay half-life ≈ 208 ticks). The gating is **incidental** — the
code's own comment says the cluster credit "only fires from propose_via_substrate
**today**; LLM-primary proposals leave `cluster_id` as None"
([tool_dispatch.py:233-240](../../../src/maxim/runtime/tool_dispatch.py#L233)).

**The signal is already there.** On the main llm-primary path the real embodied outcome
`drive_potential_diff` is already computed ([agent_loop.py:2726](../../../src/maxim/runtime/agent_loop.py#L2726))
and passed to `record_outcome` ([:2895](../../../src/maxim/runtime/agent_loop.py#L2895)).
Only the cluster *key* is missing. Closing the write path is "capture the cluster id,"
not "invent a signal."

---

## The unifying choke point

`record_outcome` is where causal links (`nac.observe`, [:217](../../../src/maxim/runtime/tool_dispatch.py#L217)),
goal credit (`nac.credit_goal`, [:231](../../../src/maxim/runtime/tool_dispatch.py#L231)), and
cluster reward (`nac.update_cluster_reward`, [:322](../../../src/maxim/runtime/tool_dispatch.py#L322))
are all written. **All four experience sources flow through it:**

| Source | Dispatch site |
|---|---|
| substrate-primary ticks | `agent_loop.py` `_record_outcome` (:2878/3027/3145/3189/3231) |
| llm-primary actions | `loop_controller.py:205` + `agent_loop.py:2465` |
| imagined-entity interactions | via the agent's **own body tools** → same path |
| real-hardware actions | same executor/dispatch path |

So closing/provenance-tagging the write path **here** covers all four uniformly. **Two
things sit outside it** and must be handled alongside any provenance work:
1. **Pain bridges write directly** — `bridges/tool_pain_bridge.py` (:172/288/364/476) and
   `bridges/pain_bridge.py` (:248/297) call `nac.record_outcome`/`record_outcome_full`,
   bypassing `tool_dispatch.record_outcome`.
2. **Affordance *encoding*** (EC node creation) happens at entity-registration time
   (`_encode_entity_affordances` → `encode_decomposed`), a different write path from
   outcome-crediting.

---

## Phase 1 — close the write path for REAL experience (small, clean, high-value)

Make an llm-primary / real-hardware agent's own drive-relief/pain outcomes reinforce
the cluster-reward substrate. **This is the core fix and the thesis advancement:** the
substrate learns from lived experience regardless of who chose the action. The pre-load
becomes *optional* — the agent self-builds a substrate from use.

### Changes (~2-3, no edits to the 5-7 `record_outcome` call sites)

1. **Build the `SensorEncoder` in llm-primary too.** Widen the guard at
   [agent_loop.py:1403](../../../src/maxim/runtime/agent_loop.py#L1403) to also construct it
   when embodied in llm-primary (graceful `None` when `memory_hub.ec` is absent).
2. **Encode the interoception cluster at action-execution time and stash it on the
   proposal.** At section-4 entry (~[agent_loop.py:2509](../../../src/maxim/runtime/agent_loop.py#L2509),
   before `executor.execute`), when `aut_mode != "substrate-primary"`, encoder present,
   embodiment present, and the proposal has no clusters: `encode_sensors(current
   drive-state, modality=INTEROCEPTION_TAG)` and attach via `dataclasses.replace`
   (`LLMProposal` is `frozen=True`, [llm_types.py:178](../../../src/maxim/agents/llm_types.py#L178)).
   The existing call sites already read `getattr(proposal, "cluster_id"/"clusters", None)`,
   so they pick it up automatically. Pre-action drive state is the correct key (credit
   attaches to the state the action was taken *from*). Update the `LLMProposal.cluster_id`
   field docstring — it currently says "None for every LLM-primary proposal."
3. **THE GUARDRAIL — drive-relief-only credit.** Route the cluster write through the
   `drive_potential_diff`-present path (the shape of the existing `operant_only` branch,
   [tool_dispatch.py:291-299](../../../src/maxim/runtime/tool_dispatch.py#L291)), and
   **suppress the tool-success floor** ([:300-301](../../../src/maxim/runtime/tool_dispatch.py#L300))
   for llm-primary. Non-negotiable: without it, the LLM's broad always-succeed action
   stream (`say`/`sense`/`examine`) floods the interoception cluster with generic "+1
   this tool ran," snowballs to the cap, and drowns the real drive-relief differential —
   the `credit_on_progress_not_execution` pathology, amplified by llm-primary's wide
   action distribution.

### Why it's safe

- **Contamination-clean (off-policy learning).** The LLM *samples which action to try*;
  the cluster id comes from EC's own `encode_sensors` (not hand-curated) and the reward
  is the body's real `drive_potential_diff`. The substrate learns `(cluster, tool) → real
  outcome`. Neither interim-contamination heuristic is tripped (the rule targets
  *hand-curated semantic decisions upstream of encoding*; this substitutes nothing).
- **Cheap.** `encode_sensors` does NOT call the sentence-transformer — it's a SHA-derived
  numeric basis ([encoder.py:445-488](../../../src/maxim/similarity/encoder.py#L445)) with a
  delta-gate ([:607-610](../../../src/maxim/similarity/encoder.py#L607)) that makes
  unchanged-drive repeats near-free. One encode per executed action (same cadence as
  substrate-primary's one-per-proposal).
- **Cross-mode cluster-id continuity NOT required.** `get_agent_tool_biases` aggregates
  per-tool across clusters ([nac.py:2370](../../../src/maxim/decisions/nac.py#L2370)), so a
  fresh llm-primary cluster still surfaces to the prompt.

### The shelf-life dissolves (the math)

`reward_bias_alpha=0.15` ([nac.py:249](../../../src/maxim/decisions/nac.py#L249)), decay
`b·(1−1/300)=b·0.99667` per tick ([nac.py:2532-2537](../../../src/maxim/decisions/nac.py#L2532),
runs ungated at [agent_loop.py:4255](../../../src/maxim/runtime/agent_loop.py#L4255)). One
signed reinforcement (±0.15) offsets **~45 ticks** of decay. Reinforced ≈once/45 ticks or
more, the bias pins near the cap. So once the write path is closed, *living keeps the
substrate fresh* — the pre-load freeze/tau workaround becomes unnecessary.

### Regression guards (Phase 1)

- Unit: an llm-primary embodied outcome with a positive `drive_potential_diff` produces a
  non-empty `_cluster_reward_bias` entry for the acted tool; a tool-success-only outcome
  (no `drive_potential_diff`) produces **none** (the guardrail).
- Unit: `encode_sensors` is invoked at most once per executed action (not per idle tick).
- Behavioral: an llm-primary cradle_pref_neutral run with NO pre-load builds a
  green-favoring bias from its own warming, and Gate-B-style substrate annotation appears
  and *stays* (no decay-out) across the run — the self-taught counterpart to the pre-load
  counterfactual.

---

## Phase 2 — imagined-experience provenance (prerequisite for any imagination→substrate feed)

Encoding is already fixed (imagined affordances EC-encode at
[trigger.py:559/778/1007](../../../src/maxim/imagination/trigger.py#L559)). The real gaps:

- Provenance is **retroactive and weak**: `CausalLink.imagined` is set only at session end
  by fragile basename-matching (`tag_imagined_links`, [nac.py:1724](../../../src/maxim/decisions/nac.py#L1724))
  that mostly *misses* under the observe-only design; the **cluster-reward surface has no
  imagined flag at all**; EC nodes don't use the `source="imagined"` slot that already
  exists ([ec.py:436](../../../src/maxim/similarity/ec.py#L436), but `encode_decomposed`
  passes `source="local"`).
- **Fix:** thread an `imagined` flag at **write time** through `record_outcome` → both
  `nac.observe` and `update_cluster_reward` (replacing the retroactive matcher), **plus
  the pain bridges** that bypass `record_outcome`; add a cluster-reward analog of
  `decay_imagined_links`; register imagined-derived EC nodes `source="imagined"`. Without
  all three, letting imagined actions feed the substrate writes fiction into the real
  substrate with no way to decay it out.

## Phase 3 — imagination-as-rehearsal (a design decision, not plumbing)

Imagined entities register **observe-only** (`register_scene`,
[trigger.py:555/776](../../../src/maxim/imagination/trigger.py#L555)) with no callable tools,
so the agent never *executes* an imagined affordance → never gets direct credit. Letting
imagination feed the substrate in the strong sense (the agent "practices" an imagined
action and learns from the imagined outcome) means deciding whether imagined affordances
become **callable**. Bigger cognitive-design scope; gated on Phase 2's provenance so
practice-derived learning is quarantinable. (Note: W2's `MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL`
is the *opposite* direction — it *reads* substrate biases to shape *what gets imagined*
([orchestrator.py:1514](../../../src/maxim/simulation/orchestrator.py#L1514)) — and writes
nothing; not part of this.)

---

## Honest caveats

- **Production hot-path change.** Phase 1 touches `record_outcome` + the agent loop for
  *every* llm-primary agent. Needs a real two-lens review; the guardrail is load-bearing.
- **Front-runs `credit_on_progress_not_execution`** (deferred): the drive-relief-only
  guardrail *is* progress-based credit for the cluster surface. Landing Phase 1 either
  subsumes part of that plan or should be reconciled with it.
- **Changes what LLM-AUT *is*.** Today the LLM-AUT's substrate is static within a session;
  after Phase 1 it grows from use. That's the intended thesis advancement, but it's a
  behavioral change worth stating in release notes and re-checking against Exp 37/38
  (substrate now reinforced, not just annotated).
