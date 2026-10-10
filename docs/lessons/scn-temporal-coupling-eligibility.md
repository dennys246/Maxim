# (borderline: mechanism-description frame would say engineering; affordance concept transfer ships with measured cross-en

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

- **[behavioral] (borderline: mechanism-description frame would say engineering; affordance concept transfer ships with measured cross-entity transfer via this PoC) SCN temporal coupling for eligibility traces (first SCN-substrate PoC).** `NAc._temporal_anchors` stores `(original_activation, TemporalSignature)` per `(agent_id, node_id)`. When fast-decay eligibility traces expire, `distribute_reward` falls back to temporal similarity — nodes activated in the same temporal phase as the reward still receive credit at `NACConfig.temporal_credit_weight` (default 0.3x, env-var `MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT`). Session-scoped — NOT persisted. Cross-session transfer uses `reward_bias` (persisted). `_temporal_anchors` are pruned in `decay_eligibility` when both the fast trace expired AND the anchor is older than `temporal_window_seconds`. Roy experiment: [docs/experiments/temporal_credit_validation.md](docs/experiments/temporal_credit_validation.md) (named-experiment citation pending stricter Roy validation per the borderline note).

---

**Corrected 2026-10-08 (1.4 grounding line, GL0 truth pass; #1120).** Appended, not rewritten. No cross-entity transfer was ever measured: the "measured cross-entity transfer via this PoC" in the tag above is false. The cited validation, [temporal_credit_validation.md](../experiments/temporal_credit_validation.md), never produced a result: its runner was executed about 19 times on the owner's Mac on 2026-04-24/25 while the script was being debugged, no result was recorded, and the most complete runs' logs carry no temporal-credit entries (the history is in that doc's status block). Component-level affordance transfer does not exist either: components close to their compound never form their own EC nodes (#1120 §4). The live stub in [docs/agents/bio-memory.md](../agents/bio-memory.md) keeps its `[behavioral]` tag by owner decision (2026-10-08: run the experiment rather than demote) and is marked **evidence PENDING**; a redesigned, pre-registered run of the SCN phase-similarity fallback credit is owed, with a four-lens design review first ([#1180](https://github.com/dennys246/Maxim/issues/1180)).

---

**2026-10-10 (#1180 parked; owner).** Appended, not rewritten. The `[behavioral]` stub is demoted to
`[engineering]`: a code map showed the fallback cannot change behaviour today (only negative rewards reach
`TemporalCreditDistributor.distribute`; `reward_bias` is clamped at ≥ 0; selection's cluster paths bypass it).
The behavioural test waits for GL2c's positive producer (`docs/experiments/temporal_credit_validation.md`,
PARKED block).
