"""Decision provenance (decision_provenance.md Stages 1+2).

The motivating failure: Exp 48's re-run took four configurations across two
machines to establish that exploration was outscoring the learned signal
~20:1 — a fact one structured field would have shown in a single jq query.
These tests pin the new ``sim_recommend_action`` fields:

- ``score_components`` — the named decomposition for the selected tool
  (causal / reward_bias / learned_bias / drive / explore)
- ``runner_up_score`` / ``n_candidates`` / ``visit_count``
- ``explore_decisive`` — would argmax WITHOUT the explore term have produced
  a different outcome?
- ``learned_margin`` — the winner's learned-bias lead over the runner-up,
  the quantity that must exceed the novelty gap (~0.11 at weight 1.5) for
  learning to be expressible.

Plus the plan's non-goal guard: instrumentation is pure observation — the
same seeded NAc makes byte-identical selections with telemetry on and off.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig


def _read_recommend_records(log_path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    return [r for r in records if r.get("subsystem") == "NAc_RECOMMEND"]


def _capture(tmp_path: Path, fn) -> list[dict[str, Any]]:
    """Run ``fn`` with sim logging enabled; return NAc_RECOMMEND records."""
    from maxim.simulation.sim_logger import disable_sim_logging, enable_sim_logging

    log_path = tmp_path / "sim_log.jsonl"
    enable_sim_logging(log_path=str(log_path))
    try:
        fn()
    finally:
        disable_sim_logging()
    return _read_recommend_records(log_path)


class TestScoreDecomposition:
    """Stage 1: the components are recorded for the selected tool."""

    def test_success_path_carries_all_fields(self, tmp_path: Path) -> None:
        nac = NAc(config=NACConfig())
        nac.update_cluster_reward("sim_aut", "c1", "tool:feed_self", reward=10.0)
        nac.update_cluster_reward("sim_aut", "c1", "tool:observe", reward=2.0)

        result: list[Any] = []
        recs = _capture(
            tmp_path,
            lambda: result.append(
                nac.recommend_action(
                    agent_id="sim_aut",
                    available_tools=["feed_self", "observe"],
                    current_cluster_id="c1",
                )
            ),
        )
        assert result[0] is not None and result[0]["tool_name"] == "feed_self"
        assert len(recs) == 1
        data = recs[0]["data"]
        comp = data["score_components"]
        assert set(comp) == {"causal", "reward_bias", "learned_bias", "drive", "explore"}
        # The winner's whole score is learned cluster bias here.
        assert comp["learned_bias"] == pytest.approx(data["best_score"], abs=1e-3)
        assert comp["explore"] == 0.0
        assert data["n_candidates"] == 2
        assert data["runner_up_score"] is not None
        assert data["runner_up_score"] < data["best_score"]
        assert data["visit_count"] == 0.0
        # Exploration off → structurally not decisive.
        assert data["explore_decisive"] is False
        # learned_margin = winner's learned bias − runner-up's learned bias.
        expected_margin = nac.cluster_reward_bias("sim_aut", "c1", "tool:feed_self") - nac.cluster_reward_bias(
            "sim_aut", "c1", "tool:observe"
        )
        assert data["learned_margin"] == pytest.approx(expected_margin, abs=1e-3)

    def test_components_sum_to_best_score(self, tmp_path: Path) -> None:
        """The decomposition is complete: no score term lives outside it.

        Review fold (Executor #2): every one of the five components must
        be NONZERO in this scenario — a fixture where causal/reward_bias
        stay at 0.0 would let a future score term added in that region
        without a ``comp`` write pass this guard silently.
        """
        nac = NAc(config=NACConfig(substrate_explore_bonus_weight=1.5))
        # learned_bias: cluster-keyed credit.
        nac.update_cluster_reward("sim_aut", "c1", "tool:warm_self", reward=5.0)
        # causal: positive outcome links for the same tool signature.
        for _ in range(3):
            nac.observe(
                event_type="tool",
                event_signature="tool:warm_self",
                outcome_type="result",
                outcome_signature="success",
                outcome_valence=Valence.POSITIVE,
                delta_seconds=1.0,
            )
        # reward_bias: node-keyed recognition credit on the tool signature.
        nac.credit_node("sim_aut", "tool:warm_self", reward=5.0)

        recs = _capture(
            tmp_path,
            lambda: nac.recommend_action(
                agent_id="sim_aut",
                available_tools=["warm_self", "look_around"],
                current_drives={"cold": 0.9},  # drive: "cold" substring hits warm affinity
                current_cluster_id="c1",
            ),
        )
        data = recs[0]["data"]
        comp = data["score_components"]
        # All five components carry signal — the completeness sum cannot
        # be satisfied by a zeroed region.
        assert comp["causal"] > 0.0
        assert comp["reward_bias"] > 0.0
        assert comp["learned_bias"] > 0.0
        assert comp["drive"] > 0.0
        assert comp["explore"] > 0.0
        assert sum(comp.values()) == pytest.approx(data["best_score"], abs=1e-3)

    def test_no_scores_path_reports_zero_candidates(self, tmp_path: Path) -> None:
        nac = NAc(config=NACConfig())
        recs = _capture(
            tmp_path,
            lambda: nac.recommend_action(agent_id="sim_aut", available_tools=["inert_tool"]),
        )
        data = recs[0]["data"]
        assert data["n_candidates"] == 0
        assert data["score_components"] is None
        assert data["explore_decisive"] is None

    def test_no_tools_path_keeps_none_sentinel(self, tmp_path: Path) -> None:
        """'No tools available' (None) stays distinguishable from 'tools
        scored nothing' (0) — mirrors the _consulted_on_empty design."""
        nac = NAc(config=NACConfig())
        recs = _capture(
            tmp_path,
            lambda: nac.recommend_action(agent_id="sim_aut", available_tools=[]),
        )
        data = recs[0]["data"]
        assert data["n_candidates"] is None

    def test_sub_threshold_path_carries_provenance(self, tmp_path: Path) -> None:
        """passed_gate=False with a best_tool still explains itself."""
        nac = NAc(config=NACConfig())
        nac.update_cluster_reward("sim_aut", "c1", "tool:feed_self", reward=0.5)  # ≈0.075 < 0.3 gate

        recs = _capture(
            tmp_path,
            lambda: nac.recommend_action(
                agent_id="sim_aut",
                available_tools=["feed_self"],
                current_cluster_id="c1",
            ),
        )
        data = recs[0]["data"]
        assert data["passed_gate"] is False
        assert data["best_tool"] == "feed_self"
        assert data["score_components"] is not None
        assert data["n_candidates"] == 1
        assert data["explore_decisive"] is False


class TestExploreDecisive:
    """Stage 2: the counterfactual field — the Exp 48 finding as one boolean."""

    def _explore_nac(self) -> NAc:
        return NAc(config=NACConfig(substrate_explore_bonus_weight=1.5))

    def test_explore_first_gate_flips_selection(self, tmp_path: Path) -> None:
        """Learned bias favors A; the explore-first hard gate forces the
        untried B. Without the explore term, A wins → decisive True."""
        nac = self._explore_nac()
        nac.update_cluster_reward("sim_aut", "c1", "tool:alpha_learned", reward=10.0)

        results: list[Any] = []

        def run() -> None:
            # Call 1: both untried; explore-first picks the higher-scored
            # alpha_learned (learned + full bonus).
            results.append(
                nac.recommend_action(
                    agent_id="sim_aut",
                    available_tools=["alpha_learned", "beta_novel"],
                    current_cluster_id="c1",
                )
            )
            # Call 2: beta_novel is now the only untried tool — the hard
            # gate forces it over alpha_learned's learned bias.
            results.append(
                nac.recommend_action(
                    agent_id="sim_aut",
                    available_tools=["alpha_learned", "beta_novel"],
                    current_cluster_id="c1",
                )
            )

        recs = _capture(tmp_path, run)
        assert results[0] is not None and results[0]["tool_name"] == "alpha_learned"
        assert results[1] is not None and results[1]["tool_name"] == "beta_novel"
        assert len(recs) == 2
        # Call 1: explore contributed everywhere but the argmax matches the
        # no-explore argmax (alpha_learned wins both ways) → not decisive.
        assert recs[0]["data"]["explore_decisive"] is False
        # Call 2: the counterfactual picks alpha_learned; actual is
        # beta_novel → exploration decided this action.
        d2 = recs[1]["data"]
        assert d2["explore_decisive"] is True
        assert d2["visit_count"] == 0.0
        assert d2["score_components"]["explore"] == pytest.approx(1.5, abs=1e-6)
        # The gate overrode the score ordering: margin over the runner-up
        # (alpha_learned) is negative on the learned axis.
        assert d2["learned_margin"] < 0.0

    def test_learned_dominance_is_not_decisive(self, tmp_path: Path) -> None:
        """Once both tools are tried and learned bias dominates the residual
        novelty difference, the counterfactual agrees with the actual pick."""
        nac = self._explore_nac()
        nac.update_cluster_reward("sim_aut", "c1", "tool:alpha_learned", reward=10.0)

        results: list[Any] = []

        def run() -> None:
            for _ in range(3):
                results.append(
                    nac.recommend_action(
                        agent_id="sim_aut",
                        available_tools=["alpha_learned", "beta_novel"],
                        current_cluster_id="c1",
                    )
                )

        recs = _capture(tmp_path, run)
        # Call 3: both tried; alpha_learned's +1.0 learned bias beats
        # beta_novel's residual novelty → actual == counterfactual.
        assert results[2] is not None and results[2]["tool_name"] == "alpha_learned"
        assert recs[2]["data"]["explore_decisive"] is False

    def test_explore_suppression_below_gate_is_decisive(self, tmp_path: Path) -> None:
        """The explore-first gate can select a tool that then fails
        min_confidence while the no-explore argmax would have PASSED —
        actual outcome None vs counterfactual proposal → decisive."""
        # Weight below min_confidence: an untried zero-base tool scores
        # 0.25 < 0.3 and the proposal is suppressed.
        nac = NAc(config=NACConfig(substrate_explore_bonus_weight=0.25))
        nac.update_cluster_reward("sim_aut", "c1", "tool:alpha_learned", reward=10.0)

        results: list[Any] = []

        def run() -> None:
            # Call 1 picks alpha_learned (untried, learned + bonus).
            results.append(
                nac.recommend_action(
                    agent_id="sim_aut",
                    available_tools=["alpha_learned", "beta_novel"],
                    current_cluster_id="c1",
                )
            )
            # Call 2: gate forces untried beta_novel at 0.25 → below the
            # 0.3 gate → returns None, while alpha_learned alone would
            # have passed.
            results.append(
                nac.recommend_action(
                    agent_id="sim_aut",
                    available_tools=["alpha_learned", "beta_novel"],
                    current_cluster_id="c1",
                )
            )

        recs = _capture(tmp_path, run)
        assert results[1] is None
        d2 = recs[1]["data"]
        assert d2["passed_gate"] is False
        assert d2["explore_decisive"] is True


class TestPureObservation:
    """The plan's binding non-goal: if instrumentation changes a selection,
    it has a bug."""

    def _seeded(self) -> NAc:
        nac = NAc(config=NACConfig(substrate_explore_bonus_weight=1.5))
        nac.update_cluster_reward("sim_aut", "c1", "tool:alpha_learned", reward=10.0)
        nac.update_cluster_reward("sim_aut", "c1", "tool:gamma_mild", reward=2.0)
        return nac

    def test_selection_sequence_identical_with_and_without_telemetry(self, tmp_path: Path) -> None:
        from maxim.simulation.sim_logger import disable_sim_logging, enable_sim_logging

        tools = ["alpha_learned", "beta_novel", "gamma_mild"]

        def sequence(nac: NAc) -> list[str | None]:
            out = []
            for _ in range(6):
                r = nac.recommend_action(
                    agent_id="sim_aut",
                    available_tools=tools,
                    current_drives={"hunger": 0.8},
                    current_cluster_id="c1",
                )
                out.append(r["tool_name"] if r else None)
            return out

        silent = sequence(self._seeded())

        log_path = tmp_path / "sim_log.jsonl"
        enable_sim_logging(log_path=str(log_path))
        try:
            logged = sequence(self._seeded())
        finally:
            disable_sim_logging()

        assert silent == logged

    def test_golden_alternation_sequence_pins_selection(self) -> None:
        """Golden-sequence pin in the ALTERNATION regime (review fold,
        Architecture #1 — BLOCKING).

        The on-vs-off test above covers only the emission path: the
        provenance block runs in BOTH arms, so a state-mutating bug
        inside it (e.g. a stray ``_visit_count`` write) cancels out and
        that test stays green — verified by bug injection during the
        pre-merge review. This pin is the guard that actually carries
        the plan's non-goal ("if any stage changes a selection, it has
        a bug"): the sequence below was generated from origin/main
        BEFORE the provenance block existed, in a regime where
        visit-count arithmetic is behaviorally decisive (close biases +
        exploration → the novelty term flips the argmax tick-to-tick).
        The injected visit-count bug diverges it at step 7; the clean
        provenance code matches it exactly.

        If this fails after an intentional selection-policy change,
        regenerate the golden sequence from the pre-change commit and
        justify the diff — do NOT just paste the new output.
        """
        nac = NAc(config=NACConfig(substrate_explore_bonus_weight=1.5))
        nac.update_cluster_reward("sim_aut", "c1", "tool:alpha", reward=3.0)
        nac.update_cluster_reward("sim_aut", "c1", "tool:beta", reward=2.0)

        seq = []
        for _ in range(12):
            r = nac.recommend_action(
                agent_id="sim_aut",
                available_tools=["alpha", "beta"],
                current_cluster_id="c1",
            )
            seq.append(r["tool_name"] if r else None)

        assert seq == [
            "alpha",
            "beta",
            "alpha",
            "beta",
            "alpha",
            "alpha",
            "beta",
            "alpha",
            "alpha",
            "beta",
            "alpha",
            "alpha",
        ]
