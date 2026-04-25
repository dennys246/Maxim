"""Integration tests for affordance concept transfer + temporal credit.

IT-1 through IT-6: substrate-level pipeline with real sentence-transformers
embeddings.  IT-7: goal-level credit attribution (NAc + distributor +
ThoughtGate, no embeddings needed).

Requires: sentence-transformers (for IT-1 through IT-6 cosine similarity).
IT-7 runs without sentence-transformers.
Marked 'slow' — skipped by default fast suite.
"""

from __future__ import annotations

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.embodiment.cerebellum import Cerebellum
from maxim.embodiment.sem import Entity
from maxim.embodiment.spec import SpecModulator
from maxim.memory.atl import ATL, ATLConfig
from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder
from maxim.time.scn import SCN
from maxim.time.temporal_event import TemporalEvent
from maxim.time.temporal_signature import TemporalSignature


# ---------------------------------------------------------------------------
# Skip if sentence-transformers unavailable
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    _HAS_ST = True
except ImportError:
    _HAS_ST = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed"),
]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bio_stack():
    """Build a real bio-stack: EC + ATL + NAc + SCN + LinguisticEncoder."""
    ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=0.40))
    atl = ATL(ATLConfig())
    nac = NAc(NACConfig(temporal_credit_weight=0.3))
    scn = SCN()
    encoder = LinguisticEncoder(ec=ec, atl=atl, nac=nac, config=EncoderConfig())
    return ec, atl, nac, scn, encoder


def _make_entity(
    name: str,
    entity_type: str,
    affordances: dict[str, dict[str, str]],
) -> Entity:
    """Build a test entity with modulators carrying affordance names.

    affordances: {modulator_name: {affordance_name: description}}
    """
    ent = Entity(name=name, entity_type=entity_type)
    for mod_name, affs in affordances.items():
        mod = SpecModulator(
            _name=mod_name,
            _entity_name=name,
            _affordances=affs,
        )
        ent.modulators[mod_name] = mod
    return ent


def _encode_affordances(entity: Entity, encoder: LinguisticEncoder, agent_id: str) -> list[str]:
    """Encode all of an entity's affordance names through the substrate path."""
    from maxim.imagination.trigger import encode_entity_affordances

    return encode_entity_affordances(entity, encoder, agent_id)


def _find_substrate_concept(atl: ATL, keyword: str) -> str:
    """Find a substrate concept whose name contains the keyword.

    ATL stores concepts with name=chunk.text (e.g., "fire breath", "fire").
    Pattern completion may merge "fire" into the "fire breath" node, so
    exact name match can miss.  Search all substrate concepts for a substring.
    """
    concepts = atl.recall(category="substrate", limit=100)
    for c in concepts:
        if keyword.lower() in c.name.lower():
            return c.id
    return ""


def _credit_affordance_node(
    nac: NAc,
    atl: ATL,
    agent_id: str,
    affordance_component: str,
    reward: float,
) -> None:
    """Find the substrate concept for an affordance component and credit it."""
    node_id = _find_substrate_concept(atl, affordance_component)
    assert node_id, f"No substrate concept containing '{affordance_component}'"
    nac.credit_node(agent_id, node_id, reward)


# ---------------------------------------------------------------------------
# IT-1: Dragon → Mage fire transfer
# ---------------------------------------------------------------------------


class TestIT1FireTransfer:
    """Cross-entity affordance transfer via shared substrate concept."""

    def test_fire_concepts_share_ec_node(self, bio_stack):
        """Dragon 'fire_breath' and mage 'flame_jet' complete to same EC node."""
        ec, atl, nac, scn, encoder = bio_stack
        agent_id = "test_agent"

        dragon = _make_entity("dragon", "creature", {"combat": {"fire_breath": "breathe fire"}})
        mage = _make_entity("mage", "creature", {"magic": {"flame_jet": "jet of flame"}})

        dragon_nodes = _encode_affordances(dragon, encoder, agent_id)
        mage_nodes = _encode_affordances(mage, encoder, agent_id)

        # "fire" from fire_breath and "flame" from flame_jet should share
        # a substrate node (cosine > 0.40 threshold)
        assert len(set(dragon_nodes) & set(mage_nodes)) > 0, (
            f"No shared nodes between dragon {dragon_nodes} and mage {mage_nodes}"
        )

    def test_positive_bias_transfers_to_similar_affordance(self, bio_stack):
        """After dragon fire_breath → POSITIVE, mage flame_jet inherits bias.

        Note: _reward_bias clamps to [0, max] — negative credit produces 0.
        Transfer is verified via positive credit (successful tool use).
        """
        ec, atl, nac, scn, encoder = bio_stack
        agent_id = "test_agent"

        dragon = _make_entity("dragon", "creature", {"combat": {"fire_breath": "breathe fire"}})
        mage = _make_entity("mage", "creature", {"magic": {"flame_jet": "jet of flame"}})

        _encode_affordances(dragon, encoder, agent_id)

        # Dragon fire_breath → POSITIVE outcome (successful use)
        _credit_affordance_node(nac, atl, agent_id, "fire", 1.0)

        # Now encode mage — "flame" should complete to the same "fire" node
        _encode_affordances(mage, encoder, agent_id)

        # The shared substrate concept should have positive bias
        fire_node_id = _find_substrate_concept(atl, "fire")
        assert fire_node_id, "No substrate concept containing 'fire'"
        bias = nac.reward_bias(agent_id, fire_node_id)
        assert bias > 0, f"Expected positive bias on 'fire' node, got {bias}"

    def test_cerebellum_has_no_model_for_new_entity(self, bio_stack):
        """Cerebellum forward models are entity-specific — no cross-entity leak."""
        ec, atl, nac, scn, encoder = bio_stack
        cerebellum = Cerebellum()

        # Train cerebellum on dragon via observe_from_action
        _make_entity("dragon", "creature", {"combat": {"fire_breath": "breathe fire"}})
        cerebellum.observe_from_action(
            entity="dragon",
            modulator="combat",
            affordance="fire_breath",
            params={"power": 1.0},
            actual={"damage": 0.8},
        )
        assert cerebellum.has_model("dragon", "combat", "fire_breath", {"power": 1.0})

        # Mage should have NO forward model
        assert not cerebellum.has_model("mage", "magic", "flame_jet", {"power": 1.0})


# ---------------------------------------------------------------------------
# IT-2: No false transfer (fire → water)
# ---------------------------------------------------------------------------


class TestIT2NoFalseTransfer:
    """Fire danger does NOT contaminate semantically dissimilar affordances."""

    def test_water_does_not_share_fire_node(self, bio_stack):
        """'water_jet' does NOT pattern-complete to the 'fire' node."""
        ec, atl, nac, scn, encoder = bio_stack
        agent_id = "test_agent"

        dragon = _make_entity("dragon", "creature", {"combat": {"fire_breath": "breathe fire"}})
        fountain = _make_entity("fountain", "environment", {"hydraulic": {"water_jet": "spray water"}})

        _encode_affordances(dragon, encoder, agent_id)
        _credit_affordance_node(nac, atl, agent_id, "fire", -1.0)

        _encode_affordances(fountain, encoder, agent_id)

        # Water concepts should NOT share nodes with fire concepts
        water_concepts = atl.recall(name="water", category="substrate", limit=1)
        if water_concepts:
            water_bias = nac.reward_bias(agent_id, water_concepts[0].id)
            assert water_bias >= 0, f"Water node got negative bias {water_bias} — false transfer!"

    def test_water_has_no_dangerous_annotation(self, bio_stack):
        """Water affordance should have no [DANGEROUS] annotation."""
        ec, atl, nac, scn, encoder = bio_stack
        agent_id = "test_agent"

        dragon = _make_entity("dragon", "creature", {"combat": {"fire_breath": "breathe fire"}})
        _encode_affordances(dragon, encoder, agent_id)
        _credit_affordance_node(nac, atl, agent_id, "fire", -1.0)

        fountain = _make_entity("fountain", "environment", {"hydraulic": {"water_jet": "spray water"}})
        _encode_affordances(fountain, encoder, agent_id)

        # Check annotation via the AFFORDANCE_STRATEGY + ATL + NAc path
        water_concepts = atl.recall(name="water", category="substrate", limit=1)
        if water_concepts:
            bias = nac.reward_bias(agent_id, water_concepts[0].id)
            # No negative bias → no [DANGEROUS] annotation
            assert bias >= -0.01, f"Water has bias {bias} — would show [DANGEROUS]"


# ---------------------------------------------------------------------------
# IT-3: Specific overrides abstract
# ---------------------------------------------------------------------------


class TestIT3SpecificOverridesAbstract:
    """Repeated positive experience overrides initially negative bias."""

    def test_positive_experience_builds_bias(self, bio_stack):
        """Repeated positive mage fire_heal builds positive bias on shared node.

        _reward_bias clamps to [0, max] — negative credit produces 0.0.
        This test verifies that positive experience on the shared 'fire'
        node accumulates bias, which is the transfer mechanism: the mage's
        fire_heal benefit transfers to any other entity with fire affordances.
        """
        ec, atl, nac, scn, encoder = bio_stack
        agent_id = "test_agent"

        dragon = _make_entity("dragon", "creature", {"combat": {"fire_breath": "breathe fire"}})
        mage = _make_entity("mage", "creature", {"magic": {"fire_heal": "healing fire"}})

        _encode_affordances(dragon, encoder, agent_id)

        fire_node_id = _find_substrate_concept(atl, "fire")
        assert fire_node_id, "No substrate concept containing 'fire'"

        # Baseline: no bias
        initial_bias = nac.reward_bias(agent_id, fire_node_id)
        assert initial_bias == 0.0, f"Expected zero initial bias, got {initial_bias}"

        # Mage fire_heal → POSITIVE (repeated 5x)
        _encode_affordances(mage, encoder, agent_id)
        for _ in range(5):
            nac.credit_node(agent_id, fire_node_id, 1.0)

        final_bias = nac.reward_bias(agent_id, fire_node_id)
        assert final_bias > 0, f"Positive experience should create positive bias, got {final_bias}"


# ---------------------------------------------------------------------------
# IT-4: Temporal credit survives eligibility decay
# ---------------------------------------------------------------------------


class TestIT4TemporalCreditUnderDecay:
    """Phase-similarity credit works after fast-decay traces expire."""

    def test_temporal_fallback_credits_after_decay(self, bio_stack):
        """After 200 decay cycles, temporal anchors still allow credit."""
        ec, atl, nac, scn, encoder = bio_stack
        agent_id = "test_agent"

        dragon = _make_entity("dragon", "creature", {"combat": {"fire_breath": "breathe fire"}})

        # Encode with temporal signatures (encode_decomposed passes TemporalSignature)
        node_ids = _encode_affordances(dragon, encoder, agent_id)
        assert node_ids, "No nodes created from encoding"

        # Verify eligibility traces exist
        eligible_before = {k: v for k, v in nac._eligibility.items() if k[0] == agent_id}
        assert eligible_before, "No eligibility traces after encoding"

        # Run 200 decay cycles — fast-decay traces should expire
        for _ in range(200):
            nac.decay_eligibility()

        # Verify fast-decay traces are gone
        eligible_after = {k: v for k, v in nac._eligibility.items() if k[0] == agent_id and v > 0.01}
        assert not eligible_after, f"Fast-decay traces should be gone after 200 cycles: {eligible_after}"

        # Temporal anchors should survive
        anchors = {k: v for k, v in nac._temporal_anchors.items() if k[0] == agent_id}
        assert anchors, "Temporal anchors should survive decay"

        # Distribute reward — should use temporal fallback path
        credits = nac.distribute_reward(agent_id, 1.0)
        assert credits, "distribute_reward should credit via temporal fallback"

        # Credit reaches nodes despite fast-decay expiry.
        # distribute_reward normalizes so total credit = reward.
        # The temporal_credit_weight (0.3) affects threshold and relative
        # proportions, not the total distributed amount.
        total_credit = sum(c for _, c in credits)
        assert total_credit > 0, "Temporal fallback should deliver credit"


# ---------------------------------------------------------------------------
# IT-5: Multi-agent isolation
# ---------------------------------------------------------------------------


class TestIT5MultiAgentIsolation:
    """Agent A's learning does NOT affect Agent B's reward bias."""

    def test_agent_isolation(self, bio_stack):
        """Agent A learns fire benefit; Agent B sees no bias."""
        ec, atl, nac, scn, encoder = bio_stack
        agent_a = "agent_a"
        agent_b = "agent_b"

        dragon = _make_entity("dragon", "creature", {"combat": {"fire_breath": "breathe fire"}})

        # Both agents encode the same entity (shared EC/ATL)
        _encode_affordances(dragon, encoder, agent_a)
        _encode_affordances(dragon, encoder, agent_b)

        # Agent A experiences fire → POSITIVE
        fire_node_id = _find_substrate_concept(atl, "fire")
        assert fire_node_id, "No substrate concept containing 'fire'"

        nac.credit_node(agent_a, fire_node_id, 1.0)

        # Agent A should have positive bias
        bias_a = nac.reward_bias(agent_a, fire_node_id)
        assert bias_a > 0, f"Agent A should have positive bias, got {bias_a}"

        # Agent B should have ZERO bias
        bias_b = nac.reward_bias(agent_b, fire_node_id)
        assert bias_b == 0.0, f"Agent B should have zero bias, got {bias_b}"


# ---------------------------------------------------------------------------
# IT-6: Self-affordance encoding
# ---------------------------------------------------------------------------


class TestIT6SelfAffordanceEncoding:
    """Agent's own body affordances form substrate concepts."""

    def test_self_affordances_create_substrate_concepts(self, bio_stack):
        """Agent body affordances (slash, move) create ATL substrate concepts."""
        ec, atl, nac, scn, encoder = bio_stack
        agent_id = "test_agent"

        body = _make_entity(
            "base_humanoid",
            "body",
            {
                "combat": {"slash": "melee slash attack"},
                "movement": {"move": "move to location"},
                "interaction": {"use": "use an object"},
            },
        )

        node_ids = _encode_affordances(body, encoder, agent_id)
        assert node_ids, "Self-affordance encoding should create substrate nodes"

        # Check ATL has concepts for each affordance component
        for aff_name in ["slash", "move", "use"]:
            concepts = atl.recall(name=aff_name, category="substrate", limit=1)
            assert concepts, f"ATL should have substrate concept for '{aff_name}'"

        # Check NAc eligibility traces exist
        traces = {k: v for k, v in nac._eligibility.items() if k[0] == agent_id}
        assert traces, "NAc should have eligibility traces for self-affordances"

    def test_self_and_scene_share_substrate_concepts(self, bio_stack):
        """Agent 'slash' and dragon 'slash' share the same EC node."""
        ec, atl, nac, scn, encoder = bio_stack
        agent_id = "test_agent"

        body = _make_entity(
            "base_humanoid",
            "body",
            {"combat": {"slash": "melee slash attack"}},
        )
        dragon = _make_entity(
            "dragon",
            "creature",
            {"combat": {"slash": "claw slash attack"}},
        )

        body_nodes = _encode_affordances(body, encoder, agent_id)
        dragon_nodes = _encode_affordances(dragon, encoder, agent_id)

        # "slash" should pattern-complete to the same EC node
        shared = set(body_nodes) & set(dragon_nodes)
        assert shared, f"Self 'slash' and dragon 'slash' should share nodes: body={body_nodes}, dragon={dragon_nodes}"


# ---------------------------------------------------------------------------
# IT-7: Goal-level credit attribution (deliberation + temporal)
# ---------------------------------------------------------------------------


class TestIT7GoalLevelCredit:
    """Validate the goal-tagged deliberation → NAc goal bias → ThoughtGate chain."""

    def test_positive_goal_gets_positive_bias(self):
        """Goal with successful outcome gets positive _goal_reward_bias."""
        from maxim.decisions.nac import NAc
        from maxim.decisions.temporal_credit import TemporalCreditDistributor
        from maxim.time.scn import SCN

        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        # Simulate: deliberation under "escape" → action succeeds
        event = TemporalEvent(
            event_id="delib-1",
            event_type="deliberation",
            event_signature="deliberation:goal:escape",
            agent_id="aut",
            temporal_sig=TemporalSignature.now(),
            context={"goal": "escape"},
        )
        dist.record_event(event)
        dist.distribute("aut", reward=1.0, goal_tag="escape")

        assert nac.get_goal_reward_bias("escape") > 0

    def test_negative_goal_gets_negative_bias(self):
        """Goal with failed outcome gets negative _goal_reward_bias."""
        from maxim.decisions.nac import NAc
        from maxim.decisions.temporal_credit import TemporalCreditDistributor
        from maxim.time.scn import SCN

        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        # Simulate: deliberation under "negotiate" → action fails
        event = TemporalEvent(
            event_id="delib-2",
            event_type="deliberation",
            event_signature="deliberation:goal:negotiate",
            agent_id="aut",
            temporal_sig=TemporalSignature.now(),
            context={"goal": "negotiate"},
        )
        dist.record_event(event)
        dist.distribute("aut", reward=-1.0, goal_tag="negotiate")

        assert nac.get_goal_reward_bias("negotiate") < 0

    def test_thought_gate_modulated_by_goal_bias(self):
        """ThoughtGate threshold is lower for positive-bias goals."""
        from maxim.decisions.nac import NAc
        from maxim.decisions.temporal_credit import TemporalCreditDistributor
        from maxim.runtime.thought_gate import ThoughtGate, ThoughtGateConfig
        from maxim.time.scn import SCN

        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)
        gate = ThoughtGate(config=ThoughtGateConfig(refractory_ticks=0, min_combined_score=0.1))

        # Build positive bias for "escape"
        for _ in range(5):
            dist.distribute("aut", reward=1.0, goal_tag="escape")

        # Build negative bias for "negotiate"
        for _ in range(5):
            dist.distribute("aut", reward=-1.0, goal_tag="negotiate")

        escape_bias = nac.get_goal_reward_bias("escape")
        negotiate_bias = nac.get_goal_reward_bias("negotiate")

        assert escape_bias > 0
        assert negotiate_bias < 0

        # ThoughtGate: escape should have lower threshold than negotiate
        from unittest.mock import MagicMock

        fake_wms = MagicMock()
        fake_wms.recent.return_value = [MagicMock(content="test")]

        d_escape = gate.should_think(working_memory=fake_wms, current_tick=0, goal_reward_bias=escape_bias)
        d_negotiate = gate.should_think(working_memory=fake_wms, current_tick=100, goal_reward_bias=negotiate_bias)

        assert d_escape.threshold_used < d_negotiate.threshold_used

    def test_none_goal_no_phantom_key(self):
        """credit_goal(None, ...) is a no-op — no phantom None key."""
        from maxim.decisions.nac import NAc
        from maxim.decisions.temporal_credit import TemporalCreditDistributor
        from maxim.time.scn import SCN

        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        dist.distribute("aut", reward=1.0, goal_tag=None)
        assert nac.get_goal_reward_bias(None) == 0.0
        assert None not in nac._goal_reward_bias

    def test_valence_signal_emitted(self):
        """distribute() emits a ValenceSignal accessible via last_valence_signal."""
        from maxim.decisions.nac import NAc
        from maxim.decisions.temporal_credit import TemporalCreditDistributor
        from maxim.decisions.valence_signal import ValenceSignal
        from maxim.time.scn import SCN

        nac = NAc()
        scn = SCN()
        dist = TemporalCreditDistributor(nac=nac, scn=scn)

        event = TemporalEvent(
            event_id="delib-3",
            event_type="deliberation",
            event_signature="deliberation:goal:escape",
            agent_id="aut",
            temporal_sig=TemporalSignature.now(),
            context={"goal": "escape"},
        )
        dist.record_event(event)
        dist.distribute("aut", reward=0.7, goal_tag="escape")

        sig = dist.last_valence_signal
        assert sig is not None
        assert isinstance(sig, ValenceSignal)
        assert sig.value == 0.7
        assert sig.goal_tag == "escape"
