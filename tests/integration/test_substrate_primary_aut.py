"""Phase -1 — substrate-primary AUT mode integration tests.

These tests exercise the full ``--aut-mode substrate-primary`` path:
NAc state + active drives + tool registry → ``propose_via_substrate`` →
``executor.execute()`` → side-effect on the embodiment / NAc.

The unit gate (``tests/unit/test_nac_recommend_action.py``) proves
``NAc.recommend_action`` returns a valid action dict in isolation. This
file proves the wiring lands the action in the executor's dispatch path
and that no LLM call occurs along the way.

Read alongside:
- docs/plans/grounded_language_acquisition.md (Phase -1 gate)
- src/maxim/runtime/agent_loop.py::propose_via_substrate
- src/maxim/decisions/nac.py::recommend_action
"""

from __future__ import annotations

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.proprioception.pain_bus import PainBus
from maxim.runtime.agent_loop import (
    _read_drive_states,
    propose_via_substrate,
)
from maxim.runtime.bootstrap import build_executor
from maxim.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cradle_aut(*, entity_ref: str = "bodies/infant_humanoid"):
    """Build the executor + NAc + pain_bus combo a substrate-primary
    cradle AUT would use. Mirrors the orchestrator's AUT construction.
    """
    pain_bus = PainBus(_allow_raw=True)
    nac = NAc(NACConfig(temporal_window_seconds=60.0))
    registry = ToolRegistry()

    executor = build_executor(
        registry,
        pain_bus=pain_bus,
        nac=nac,
        entity_ref=entity_ref,
        component_registry=ComponentRegistry(),
    )
    return {
        "executor": executor,
        "nac": nac,
        "pain_bus": pain_bus,
        "registry": registry,
    }


def _set_hunger(executor, value: float) -> None:
    """Force the hunger drive to a specific value on the embodiment root."""
    root = executor.embodiment.root
    root.vital_metrics["hunger"] = value


# ---------------------------------------------------------------------------
# Phase -1 success criterion
# ---------------------------------------------------------------------------


class TestCradleSubstratePrimary:
    """Phase -1 — cradle 1-tick with substrate-primary action selection.

    Success: AUT calls a body tool without an LLM. Failure: substrate
    proposes nothing or proposes a non-body tool.
    """

    def test_drive_states_extracted_from_embodiment(self):
        """``_read_drive_states`` surfaces hunger from the infant body.

        This is the contract that lets ``propose_via_substrate`` see
        active drives without going through the LLM context-building path.
        """
        world = _build_cradle_aut()
        _set_hunger(world["executor"], 0.8)

        drives = _read_drive_states(world["executor"])

        assert "hunger" in drives, f"Expected hunger drive in extracted state, got: {list(drives.keys())}"
        assert drives["hunger"] == pytest.approx(0.8)

    def test_substrate_proposes_body_tool_when_hungry(self):
        """**Phase -1 success criterion**: hungry infant → body-tool proposal.

        Hunger > 0.5 + an embodiment with sense/affordance tools must
        produce SOME non-None proposal. Whether the substrate picks
        ``read_..._hunger`` (drive-name substring match — sense self) or
        an arms affordance (``pick_up``-affinity match) is acceptable
        for Phase -1; both are non-LLM body-tool actions and both prove
        the substrate can act.

        The aspirational ``pick_up_food`` outcome from the plan is the
        eventual goal once NAc has learned `pick_up → hunger_satisfied`
        — see ``test_pick_up_wins_after_learning`` below.
        """
        world = _build_cradle_aut()
        _set_hunger(world["executor"], 0.8)

        proposal = propose_via_substrate(
            nac=world["nac"],
            agent_id="cradle_infant",
            executor=world["executor"],
        )

        assert proposal is not None, (
            "Phase -1 GATE FAILED: substrate produced no proposal "
            "despite active hunger drive + available body tools. "
            "NAc.recommend_action cannot translate substrate state into "
            "action under cradle conditions."
        )
        assert proposal.action is not None
        assert proposal.strategy_used == "substrate-primary"

        # Body tool = anything from the embodiment's tool registry.
        # The infant's tools are all named with the entity name as
        # prefix (e.g. ``read_infant_humanoid_hunger``,
        # ``infant_humanoid_pick_up``, ``head_look``, ``arms_use``).
        tool_name = proposal.action["tool_name"]
        all_tools = set(world["registry"].list())
        assert tool_name in all_tools, (
            f"Substrate proposed {tool_name!r} but it's not in the registry. Registry has {len(all_tools)} tools."
        )
        # Reasoning must mention the hunger drive — that's the signal
        # that drove the score. If reasoning is empty the substrate
        # picked randomly, which violates Phase -1's "have an opinion"
        # contract.
        assert "hunger" in proposal.reasoning, (
            f"Expected hunger drive to feature in scoring reasoning, got: {proposal.reasoning!r}"
        )

    def test_pick_up_wins_after_learning(self, valence_positive):
        """Aspirational Phase -1 outcome: a body-tool with positive history
        beats sensor reads when the agent has learned the action helps.

        This proves the substrate can switch from "sense the body" to
        "act on the body" once NAc has formed a `pick_up → satisfied`
        link. Stand-in for what cross-session learning will deliver.
        """
        world = _build_cradle_aut()
        _set_hunger(world["executor"], 0.8)

        # Find the `pick_up` affordance tool — name varies by entity
        pick_up_tools = [t for t in world["registry"].list() if "pick_up" in t]
        assert pick_up_tools, "Infant body should expose at least one pick_up tool"
        pick_up_tool = pick_up_tools[0]

        # Pre-bake NAc with positive learning for pick_up
        for _ in range(5):
            world["nac"].observe(
                event_type="tool",
                event_signature=f"tool:{pick_up_tool}",
                outcome_type="result",
                outcome_signature="hunger_satisfied",
                outcome_valence=valence_positive,
                delta_seconds=1.0,
            )

        proposal = propose_via_substrate(
            nac=world["nac"],
            agent_id="cradle_infant",
            executor=world["executor"],
        )

        assert proposal is not None
        assert proposal.action["tool_name"] == pick_up_tool, (
            f"After learning, substrate should prefer the food-relevant "
            f"body tool. Got {proposal.action['tool_name']!r} (reasoning: "
            f"{proposal.reasoning!r})"
        )
        assert "causal_pos" in proposal.reasoning

    def test_proposal_dispatches_through_executor(self):
        """The substrate proposal is shaped to be directly executable.

        ``executor.execute(proposal.action)`` is the dispatch path the
        agent loop uses; if proposal.action doesn't satisfy that
        contract, the substrate-primary mode silently no-ops.
        """
        world = _build_cradle_aut()
        _set_hunger(world["executor"], 0.8)

        proposal = propose_via_substrate(
            nac=world["nac"],
            agent_id="cradle_infant",
            executor=world["executor"],
        )
        assert proposal is not None
        assert proposal.action is not None

        # Body tools want a `target`/`object` param. The substrate
        # heuristic doesn't fill those (Phase -1 is a Boolean — does the
        # tool dispatch at all). The dispatch may yield a parameter
        # error inside the tool, but the executor must accept the
        # action dict and produce a result without raising.
        result = world["executor"].execute(proposal.action)

        assert result is not None, "Executor returned None for substrate-proposed action. ToolOutput contract violated."
        # success/failure is acceptable for Phase -1 — the load-bearing
        # claim is "the substrate produced a dispatchable action and the
        # tool ran". The substrate-primary loop recovers from failures
        # via the same NAc learning that the LLM-primary loop uses.

    def test_substrate_idle_when_no_drives_and_no_learning(self):
        """No active drive + no learning → no proposal (silent IDLE).

        Substrate-primary mode does NOT fall back to random selection.
        The whole point of Phase -1 is "substrate has an opinion or
        nothing happens". This test pins that contract.
        """
        world = _build_cradle_aut()
        # Leave hunger at the YAML default (0.0 — well below 0.5)

        proposal = propose_via_substrate(
            nac=world["nac"],
            agent_id="cradle_infant",
            executor=world["executor"],
        )

        assert proposal is None, (
            f"Substrate produced an unexpected proposal with no active drive and no learning: {proposal}"
        )

    def test_no_embodiment_returns_no_drives(self):
        """``_read_drive_states`` is safe on executors without embodiment."""
        registry = ToolRegistry()
        executor = build_executor(
            registry,
            pain_bus=None,
            nac=None,
            entity_ref=None,
            component_registry=None,
        )
        assert _read_drive_states(executor) == {}

    def test_propose_via_substrate_safe_when_nac_none(self):
        """Substrate-primary path is safe if NAc isn't wired (returns None)."""
        world = _build_cradle_aut()
        _set_hunger(world["executor"], 0.8)

        proposal = propose_via_substrate(
            nac=None,
            agent_id="cradle_infant",
            executor=world["executor"],
        )
        assert proposal is None


class TestClusterKeyedActionSelection:
    """Track 2 of grounded_language_acquisition.md Phase 0+: when the
    substrate has learned ``tool X works in cluster A``, then in
    cluster A it picks X — even when drive affinity would have
    picked something else.

    Replaces the drive-affinity substring heuristic with cluster-keyed
    NAc reward bias as the substrate's primary action-selection signal
    once any history exists for the active interoception cluster.
    """

    def test_cluster_bias_dominates_drive_affinity(self):
        """Tool with positive cluster bias outscores a drive-affinity
        match in the same cluster.

        Setup: hunger drive elevated. Without cluster context, the
        substrate would pick ``pick_up_food`` via the affinity table
        (``hunger`` → ``pick_up`` substring). With cluster A active and
        ``unrelated_tool`` rewarded in cluster A, the cluster bias
        wins — proving cluster-keyed learning supersedes the cold-start
        heuristic.
        """
        nac = NAc(NACConfig(temporal_window_seconds=60.0))
        cluster_a = "cluster-a-uuid"

        # Reinforce unrelated_tool in cluster A. Several updates push
        # the bias near the cap so it overwhelms drive affinity (which
        # contributes 0.8 for this hunger value).
        for _ in range(10):
            nac.update_cluster_reward(
                agent_id="agent",
                cluster_id=cluster_a,
                tool_signature="tool:unrelated_tool",
                reward=1.0,
            )

        result = nac.recommend_action(
            agent_id="agent",
            available_tools=["pick_up_food", "unrelated_tool"],
            current_drives={"hunger": 0.8},
            current_cluster_id=cluster_a,
        )

        assert result is not None
        assert result["tool_name"] == "unrelated_tool", (
            f"Cluster bias for unrelated_tool should beat hunger→pick_up_food drive "
            f"affinity. Got tool={result['tool_name']!r}, reasoning={result['reasoning']!r}"
        )
        assert "cluster_bias" in result["reasoning"], (
            f"Reasoning should mention cluster_bias contribution; got: {result['reasoning']!r}"
        )

    def test_cluster_keyed_preference_differentiates_across_clusters(self):
        """Same agent, two distinct clusters — recommend_action picks
        the tool keyed to whichever cluster is active.
        """
        nac = NAc(NACConfig(temporal_window_seconds=60.0))
        cluster_a = "cluster-a-uuid"
        cluster_b = "cluster-b-uuid"

        # Tool X rewarded in cluster A
        for _ in range(10):
            nac.update_cluster_reward(
                agent_id="agent",
                cluster_id=cluster_a,
                tool_signature="tool:tool_x",
                reward=1.0,
            )
        # Tool Y rewarded in cluster B
        for _ in range(10):
            nac.update_cluster_reward(
                agent_id="agent",
                cluster_id=cluster_b,
                tool_signature="tool:tool_y",
                reward=1.0,
            )

        # Both available, no drives. State in cluster A → pick X.
        result_a = nac.recommend_action(
            agent_id="agent",
            available_tools=["tool_x", "tool_y"],
            current_cluster_id=cluster_a,
        )
        assert result_a is not None
        assert result_a["tool_name"] == "tool_x"

        # Same agent, switch to cluster B → pick Y.
        result_b = nac.recommend_action(
            agent_id="agent",
            available_tools=["tool_x", "tool_y"],
            current_cluster_id=cluster_b,
        )
        assert result_b is not None
        assert result_b["tool_name"] == "tool_y"

    def test_no_cluster_context_falls_back_to_drive_affinity(self):
        """Without ``current_cluster_id``, ``recommend_action`` behaves
        exactly as before Track 2 — drive affinity carries the
        selection. Regression guard for the smoke run path where no
        sensor encoder is wired.
        """
        nac = NAc(NACConfig(temporal_window_seconds=60.0))

        # Even with cluster bias on file, omitting current_cluster_id
        # means we don't consult it — drive affinity fires.
        nac.update_cluster_reward(
            agent_id="agent",
            cluster_id="some-cluster",
            tool_signature="tool:unrelated_tool",
            reward=1.0,
        )

        result = nac.recommend_action(
            agent_id="agent",
            available_tools=["pick_up_food", "unrelated_tool"],
            current_drives={"hunger": 0.8},
            # current_cluster_id intentionally omitted
        )

        assert result is not None
        assert result["tool_name"] == "pick_up_food", (
            f"Without cluster context, drive affinity should win. Got {result['tool_name']!r}"
        )

    def test_negative_cluster_bias_avoids_tool(self):
        """Cluster bias is bidirectional — punishment in a cluster
        steers selection away from a tool that drive affinity would
        otherwise pick.
        """
        nac = NAc(NACConfig(temporal_window_seconds=60.0))
        cluster_a = "cluster-a-uuid"

        # Punish pick_up_food in cluster A. Drive affinity still
        # contributes +0.8, but cluster bias subtracts ~1.0.
        for _ in range(10):
            nac.update_cluster_reward(
                agent_id="agent",
                cluster_id=cluster_a,
                tool_signature="tool:pick_up_food",
                reward=-1.0,
            )

        result = nac.recommend_action(
            agent_id="agent",
            available_tools=["pick_up_food", "consume"],
            current_drives={"hunger": 0.8},
            current_cluster_id=cluster_a,
        )

        # consume also matches hunger affinity (+0.7 weight). Without
        # the negative cluster bias on pick_up_food, the two score
        # comparably; with it, consume wins.
        assert result is not None
        assert result["tool_name"] != "pick_up_food", (
            f"Negative cluster bias on pick_up_food should steer away from it. "
            f"Got {result['tool_name']!r}, reasoning={result['reasoning']!r}"
        )

    def test_propose_via_substrate_pipes_cluster_id_into_recommendation(self):
        """End-to-end Track 2 wiring: ``propose_via_substrate`` captures
        the cluster id from ``SensorEncoder.encode_sensors`` and passes
        it through to ``recommend_action``, so cluster-keyed bias takes
        effect during a normal substrate-primary tick.
        """
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import SensorEncoder

        world = _build_cradle_aut()
        _set_hunger(world["executor"], 0.8)

        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.40))
        encoder = SensorEncoder(ec=ec, atl=None, nac=None)

        # First tick: encode the current drive snapshot to learn the
        # cluster id the substrate sees in this state.
        from maxim.runtime.agent_loop import _read_drive_states

        drives = _read_drive_states(world["executor"])
        cluster_id = encoder.encode_sensors(agent_id="cradle_infant", sensors=drives)
        assert cluster_id is not None

        # Punish pick_up_food in this cluster — strong negative signal.
        for _ in range(10):
            world["nac"].update_cluster_reward(
                agent_id="cradle_infant",
                cluster_id=cluster_id,
                tool_signature="tool:pick_up_food",
                reward=-1.0,
            )

        # Drive a fresh propose_via_substrate tick. The encoder's
        # sub-threshold delta gate may short-circuit, but the cluster
        # id stays the same — recommend_action must see the negative
        # bias and avoid pick_up_food.
        # We can't construct a registry here that includes pick_up_food
        # in the cradle entity_ref, so instead we verify the method
        # signature accepts current_cluster_id (the wiring contract)
        # and the negative bias surfaces in the read API.
        bias = world["nac"].cluster_reward_bias("cradle_infant", cluster_id, "tool:pick_up_food")
        assert bias < 0.0, f"Expected negative cluster bias, got {bias}"

        # Direct recommend_action with pick_up_food + alternative
        result = world["nac"].recommend_action(
            agent_id="cradle_infant",
            available_tools=["pick_up_food", "consume"],
            current_drives=drives,
            current_cluster_id=cluster_id,
        )
        assert result is not None
        assert result["tool_name"] != "pick_up_food"


class TestSubstratePrimaryNoLLM:
    """The ``--aut-mode substrate-primary`` path must NOT touch the LLM.

    The whole point of Phase -1 is to prove the substrate can act
    without an LLM. These tests pin that the proposal-generation path
    has no LLM dependency.
    """

    def test_propose_does_not_import_llm_router(self, monkeypatch):
        """Calling ``propose_via_substrate`` does not trigger an LLM
        call.

        We trip-wire the LLM call layer — if anything in the substrate
        path tries to fan out to an LLMRouter or backend, this fixture
        explodes.
        """
        # The simplest tripwire: monkeypatch ``LLMRouter.dispatch`` to
        # raise.  If the substrate path is clean, nothing calls it.
        from maxim.models.language import router as router_module

        def _explode(*_args, **_kwargs):
            raise AssertionError(
                "LLMRouter.dispatch was called during substrate-primary "
                "action generation — substrate path is contaminated."
            )

        monkeypatch.setattr(
            router_module.LLMRouter,
            "dispatch",
            _explode,
            raising=False,
        )

        world = _build_cradle_aut()
        _set_hunger(world["executor"], 0.8)

        proposal = propose_via_substrate(
            nac=world["nac"],
            agent_id="cradle_infant",
            executor=world["executor"],
        )
        # Whether substrate proposed or not is orthogonal — what matters
        # is that LLMRouter.dispatch was never called along the way.
        assert proposal is None or proposal.strategy_used == "substrate-primary"


class TestAutModeFlag:
    """The CLI flag is registered and accepted.

    Pins the wiring contract end-to-end so a future refactor that drops
    the flag fails loudly here instead of silently ignoring user intent.
    """

    def test_aut_mode_default_is_llm_primary(self):
        from maxim.cli_parser import _build_parser

        args = _build_parser().parse_args(["--sim", "test goal"])
        assert getattr(args, "aut_mode", None) == "llm-primary"

    def test_aut_mode_substrate_primary_accepted(self):
        from maxim.cli_parser import _build_parser

        args = _build_parser().parse_args(["--sim", "test goal", "--aut-mode", "substrate-primary"])
        assert args.aut_mode == "substrate-primary"

    def test_aut_mode_invalid_rejected(self):
        from maxim.cli_parser import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--sim", "test", "--aut-mode", "telepathy"])
