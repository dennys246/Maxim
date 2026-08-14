# (borderline: mechanism-description frame would say engineering; affordance concept transfer ships with measured cross-en

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

- **[behavioral] (borderline: mechanism-description frame would say engineering; affordance concept transfer ships with measured cross-entity transfer via this PoC) SCN temporal coupling for eligibility traces (first SCN-substrate PoC).** `NAc._temporal_anchors` stores `(original_activation, TemporalSignature)` per `(agent_id, node_id)`. When fast-decay eligibility traces expire, `distribute_reward` falls back to temporal similarity — nodes activated in the same temporal phase as the reward still receive credit at `NACConfig.temporal_credit_weight` (default 0.3x, env-var `MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT`). Session-scoped — NOT persisted. Cross-session transfer uses `reward_bias` (persisted). `_temporal_anchors` are pruned in `decay_eligibility` when both the fast trace expired AND the anchor is older than `temporal_window_seconds`. Roy experiment: [docs/experiments/temporal_credit_validation.md](docs/experiments/temporal_credit_validation.md) (named-experiment citation pending stricter Roy validation per the borderline note).
