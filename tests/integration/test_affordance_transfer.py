"""Integration tests for affordance concept transfer + temporal credit.

**What "transfer" means here.** Every sharing these tests pin is NAME similarity: two affordance names whose
sentence embeddings are close complete into one EC substrate node, so NAc bias on that node is read by both.
Similarity by CONSEQUENCE (two things that hurt the same way) is the grounding line's job, not this file's
(#1120).

**Which tests need a semantic encoder** (``_require_semantic_encoder`` skips them under the hash fallback):

- IT-1 ``test_fire_concepts_share_ec_node`` / ``test_positive_bias_transfers_to_similar_affordance``,
- IT-2 (all), and the three #1120 red gates in ``TestIssue1120RedGates``.

Encoder-independent (they run, and mean the same thing, under the hash fallback): IT-1's Cerebellum test, IT-3
(``credit_node`` bookkeeping), IT-4 (eligibility decay / temporal anchors), IT-5 (per-agent NAc isolation), IT-6
(self-affordance concepts; its sharing is the identical string "slash") and IT-7 (goal-level credit, no
embeddings at all).

Nodes are looked up by the ids ``encode_decomposed`` returns for each affordance (one id per chunk, in chunk
order ``[compound, component, component, ...]``), never by name: a by-name lookup is what made #1120's
controls miss the fountain's ``'water jet'`` node.

The fixture runs at the PRODUCTION threshold (``ECConfig()``, 0.44). The retired 0.40 appears only in the
``bio_stack_at_retired_040`` fixture, used by one red gate that documents why it was retired.

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

# The pattern-completion threshold retired by Exp 24–26 (running-mean centroid drift; see the
# ``ECConfig.pattern_complete_threshold`` comment). Used ONLY by the red gate that pins why it was retired.
_RETIRED_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _build_bio_stack(ec_config: ECConfig):
    ec = EntorhinalCortex(ec_config)
    atl = ATL(ATLConfig())
    nac = NAc(NACConfig(temporal_credit_weight=0.3))
    scn = SCN()
    encoder = LinguisticEncoder(ec=ec, atl=atl, nac=nac, config=EncoderConfig())
    return ec, atl, nac, scn, encoder


@pytest.fixture()
def bio_stack():
    """Build a real bio-stack (EC + ATL + NAc + SCN + LinguisticEncoder) at the PRODUCTION threshold.

    ``ECConfig()`` and not a literal, so a future default change moves this fixture with production
    (NAc's ``get_threshold_overrides`` holds a coupled copy of the default; see ``ec.py``).
    """
    return _build_bio_stack(ECConfig())


@pytest.fixture()
def bio_stack_at_retired_040():
    """The same stack at the RETIRED 0.40 threshold — for the centroid-drift red gate only."""
    stack = _build_bio_stack(ECConfig(pattern_complete_threshold=_RETIRED_THRESHOLD))
    assert stack[0].config.pattern_complete_threshold == _RETIRED_THRESHOLD
    return stack


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
        # Capability-only by helper design — no sub-sensors are accepted.
        mod = SpecModulator(
            _name=mod_name,
            _entity_name=name,
            _affordances=affs,
            _abstract=True,
        )
        ent.modulators[mod_name] = mod
    return ent


def _dragon() -> Entity:
    return _make_entity("dragon", "creature", {"combat": {"fire_breath": "breathe fire"}})


def _mage() -> Entity:
    return _make_entity("mage", "creature", {"magic": {"flame_jet": "jet of flame"}})


def _fountain() -> Entity:
    return _make_entity("fountain", "environment", {"hydraulic": {"water_jet": "spray water"}})


def _encode_affordances(entity: Entity, encoder: LinguisticEncoder, agent_id: str) -> list[str]:
    """Encode all of an entity's affordance names through the orchestrator's entry point (flat id list)."""
    from maxim.imagination.trigger import encode_entity_affordances

    return encode_entity_affordances(entity, encoder, agent_id)


def _encode_nodes_by_affordance(entity: Entity, encoder: LinguisticEncoder, agent_id: str) -> dict[str, list[str]]:
    """Encode each affordance through the production affordance encoder; return its node ids per affordance.

    Uses the same factory ``encode_entity_affordances`` uses (``_make_aff_encoder``), so the ids are the ones
    production would produce. Each list is one node id per chunk, in chunk order ``[compound, comp1, comp2, ...]``
    (``LinguisticEncoder.encode_decomposed`` appends one id per chunk, duplicates included).
    """
    from maxim.imagination.trigger import _make_aff_encoder

    aff_encoder = _make_aff_encoder(encoder)
    assert aff_encoder is not None, "production affordance-encoder factory failed to build"
    return {
        aff_name: aff_encoder.encode_decomposed(aff_name, "text", agent_id)
        for mod in entity.modulators.values()
        for aff_name in mod.affordances
    }


def _all_nodes(by_affordance: dict[str, list[str]]) -> set[str]:
    return {nid for ids in by_affordance.values() for nid in ids}


def _require_semantic_encoder(encoder) -> None:
    """Skip when the substrate has no semantics to test (D61).

    `tests/conftest.py` deliberately sets `HF_HUB_OFFLINE=1`,
    `TRANSFORMERS_OFFLINE=1` and an isolated `HF_HOME` — correct isolation, no
    network in tests and no polluting the developer's model cache. Unless the
    model is pre-seeded into that cache, `LinguisticEncoder` uses the
    bag-of-words hash fallback, whose own docstring says outright: *"Not
    semantically meaningful — paraphrase collapse will NOT work with this."*

    Against that encoder, assertions about "fire" and "flame" sharing a node,
    or "water" not sharing one, cannot be made: **the positive tests fail
    honestly; the negative controls pass VACUOUSLY**, because a hash encoder
    separates everything — exactly the shape of the outcome they assert.
    Skipping is not a fix, it is honest reporting: a skipped test is not a
    passing test.
    """
    # `using_fallback` LOADS the model before answering. The old check read `encoder._model is None` before anything had
    # loaded it (the model is lazy), so it was true on every run and these tests skipped even with the model cached:
    # the slow lane's first real run (#940) showed it. D61's remedy, pre-seeding the cache, could never have worked.
    if encoder.using_fallback:
        pytest.skip(
            "semantic encoder unavailable (conftest forces HF offline) — the hash fallback "
            "cannot express paraphrase similarity, so this assertion is untestable here, and "
            "the negative controls would pass vacuously. See D61."
        )


# ---------------------------------------------------------------------------
# IT-1: Dragon → Mage fire transfer
# ---------------------------------------------------------------------------


class TestIT1FireTransfer:
    """Cross-entity affordance transfer via a shared substrate node (name similarity, compound level)."""

    def test_fire_concepts_share_ec_node(self, bio_stack):
        """Mage 'flame_jet' completes into the dragon's 'fire_breath' COMPOUND node.

        Measured with paraphrase-mpnet-base-v2: 'flame jet' ~ 'fire breath' = 0.601, above the production 0.44
        (#1120). The sharing is compound-to-compound by name; it does NOT go through a shared "fire"/"flame"
        component node. A component completes into whatever existing node it is nearest. On a fresh EC that is
        its own compound (~0.73), so ``fire`` and ``breath`` never form nodes; a component forms its own node only
        when no existing node is near it (e.g. the mage's ``jet``, after ``flame jet`` had merged into
        ``fire breath``). See the #1120 red gate ``test_components_get_their_own_nodes``.
        """
        ec, atl, nac, scn, encoder = bio_stack
        _require_semantic_encoder(encoder)
        agent_id = "test_agent"

        # The dragon goes through the orchestrator's entry point, so it stays exercised.
        dragon_flat = _encode_affordances(_dragon(), encoder, agent_id)
        assert dragon_flat, "encode_entity_affordances returned no nodes for the dragon"
        dragon_compound = dragon_flat[0]  # one affordance → its compound chunk's node comes first

        mage = _encode_nodes_by_affordance(_mage(), encoder, agent_id)
        assert mage["flame_jet"][0] == dragon_compound, (
            f"'flame jet' did not complete into the 'fire breath' node: dragon={dragon_flat}, mage={mage}"
        )

    def test_positive_bias_transfers_to_similar_affordance(self, bio_stack):
        """Positive credit on the dragon's 'fire_breath' node is read through the mage's own 'flame_jet' node.

        The read is keyed by the node id the MAGE's encoding returned, so the test fails if the mage forms its
        own node. Only POSITIVE bias can transfer: ``reward_bias`` is clamped to [0, max] (#910).

        Confound: crediting widens that node's completion threshold (0.44 − bias) before the mage encodes. The
        0.601 match clears the unwidened 0.44 too, which ``test_fire_concepts_share_ec_node`` shows without
        credit. The unrelated-affordance control (water must read 0.0) is NOT here: under this same widening
        the fountain collapses into the credited node, which the red gate
        ``test_credit_widening_does_not_absorb_water`` pins.
        """
        ec, atl, nac, scn, encoder = bio_stack
        _require_semantic_encoder(encoder)
        agent_id = "test_agent"

        dragon = _encode_nodes_by_affordance(_dragon(), encoder, agent_id)
        nac.credit_node(agent_id, dragon["fire_breath"][0], 1.0)

        mage = _encode_nodes_by_affordance(_mage(), encoder, agent_id)
        bias = nac.reward_bias(agent_id, mage["flame_jet"][0])
        assert bias > 0, f"mage 'flame_jet' node {mage['flame_jet'][0]} carries no bias ({bias}); dragon={dragon}"

    def test_cerebellum_has_no_model_for_new_entity(self, bio_stack):
        """Cerebellum forward models are entity-specific — no cross-entity leak (encoder-independent)."""
        ec, atl, nac, scn, encoder = bio_stack
        cerebellum = Cerebellum()

        # Train cerebellum on dragon via observe_from_action
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
    """Water affordances do not share nodes with fire affordances (dragon-only scenario, no credit).

    Under credit they DO share: the widening gate ``test_credit_widening_does_not_absorb_water`` pins it (#1181).
    """

    def test_water_does_not_share_with_fire_dragon_only(self, bio_stack):
        """After dragon then fountain at 0.44, the fountain's nodes are disjoint from the dragon's.

        Low power, by design: in this scenario water and fire separate trivially ('water jet' ~ 'fire breath' =
        0.298, #1120), so this guards only against a gross collapse. The scenarios where a false transfer does
        appear are the red gates in ``TestIssue1120RedGates``. No bias assertion: ``reward_bias`` is never
        negative (#910), so "water got no harm" cannot fail.
        """
        ec, atl, nac, scn, encoder = bio_stack
        _require_semantic_encoder(encoder)
        agent_id = "test_agent"

        dragon = _encode_nodes_by_affordance(_dragon(), encoder, agent_id)
        fountain = _encode_nodes_by_affordance(_fountain(), encoder, agent_id)

        shared = _all_nodes(fountain) & _all_nodes(dragon)
        assert not shared, f"fountain shares {shared} with the dragon: dragon={dragon}, fountain={fountain}"

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "#1120: crediting 'fire breath' up to the cap (bias = max_reward_bias 0.20) widens its completion "
            "threshold to 0.44 - 0.20 = 0.24, well below 'water jet'~'fire breath' 0.298, so the fountain "
            "collapses into the credited fire node and reads its bias — reward-driven widening absorbs an "
            "unrelated name (#1181); a flip caused by changing reward_bias_alpha / max_reward_bias is not a fix "
            "of #1181"
        ),
    )
    def test_credit_widening_does_not_absorb_water(self, bio_stack):
        """Credit at the bias cap on the fire node must not pull the fountain into it (dragon-only, 0.44).

        Strict red gate: asserts the correct behaviour and flips when reward-driven threshold widening stops
        absorbing affordances this dissimilar. The node is credited until ``reward_bias`` reaches
        ``max_reward_bias``, so the margin is the full cap and not a ``reward_bias_alpha`` knife-edge (one
        +1.0 credit gives 0.15, leaving 0.44 - 0.15 = 0.29 only 0.008 under 0.298). The precondition (without
        credit the two separate) is ``test_water_does_not_share_with_fire_dragon_only``.
        """
        ec, atl, nac, scn, encoder = bio_stack
        _require_semantic_encoder(encoder)
        agent_id = "test_agent"

        dragon = _encode_nodes_by_affordance(_dragon(), encoder, agent_id)
        fire_node = dragon["fire_breath"][0]
        cap = nac.config.max_reward_bias
        for _ in range(100):
            if nac.reward_bias(agent_id, fire_node) >= cap:
                break
            nac.credit_node(agent_id, fire_node, 1.0)
        if nac.reward_bias(agent_id, fire_node) != cap:
            pytest.fail(
                f"precondition: credit did not bring the fire node to the cap: "
                f"bias={nac.reward_bias(agent_id, fire_node)}, cap={cap}"
            )

        fountain = _encode_nodes_by_affordance(_fountain(), encoder, agent_id)
        shared = _all_nodes(fountain) & _all_nodes(dragon)
        assert not shared, f"fountain collapsed into the credited fire node: dragon={dragon}, fountain={fountain}"
        assert nac.reward_bias(agent_id, fountain["water_jet"][0]) == 0.0


# ---------------------------------------------------------------------------
# #1120 red gates: the false transfers the controls were written for
# ---------------------------------------------------------------------------


class TestIssue1120RedGates:
    """Strict red gates for the defects #1120's investigation measured. Each asserts the CORRECT behaviour.

    Each uses ``raises=AssertionError`` and checks its precondition with ``pytest.fail`` (a different
    exception type), so a broken precondition FAILS the test instead of hiding inside the expected xfail.
    """

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "#1120 §4: each component is ~0.73 to its compound and completes into it ('fire'~'breath' 0.297 to "
            "each other), so component concepts never exist and 'fire' and 'breath' cannot be told apart"
        ),
    )
    def test_components_get_their_own_nodes(self, bio_stack):
        """``fire_breath`` → ["fire breath", "fire", "breath"] must give three distinct nodes (fresh EC, 0.44).

        A flip here is an affordance-decomposition change, which fires the GL5 successor's re-run trigger. Owner:
        the grounding line's GL4 (``docs/plans/latent_forward_model.md``).
        """
        ec, atl, nac, scn, encoder = bio_stack
        _require_semantic_encoder(encoder)

        nodes = _encode_nodes_by_affordance(_dragon(), encoder, "test_agent")
        if len(nodes["fire_breath"]) != 3:
            pytest.fail(f"precondition: expected 3 chunks for 'fire_breath', got {nodes['fire_breath']}")
        assert len(set(nodes["fire_breath"])) == 3, f"components collapsed into the compound: {nodes}"

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "#1120: 'water jet' and 'flame jet' share the mage's 'jet' component node at 0.44 "
            "('water jet'~'jet' 0.785) — name similarity, not consequence; flips only if word-level sharing "
            "stops (grounding line GL4)"
        ),
    )
    def test_water_does_not_share_with_fire_side_after_mage(self, bio_stack):
        """Dragon → mage → fountain at 0.44: the fountain shares no node with the dragon or the mage."""
        ec, atl, nac, scn, encoder = bio_stack
        _require_semantic_encoder(encoder)
        agent_id = "test_agent"

        dragon = _encode_nodes_by_affordance(_dragon(), encoder, agent_id)
        mage = _encode_nodes_by_affordance(_mage(), encoder, agent_id)
        # Positive control: the scenario provably CAN share a node.
        if mage["flame_jet"][0] != dragon["fire_breath"][0]:
            pytest.fail(f"precondition: 'flame jet' did not complete into 'fire breath': {dragon}, {mage}")

        fountain = _encode_nodes_by_affordance(_fountain(), encoder, agent_id)
        shared = _all_nodes(fountain) & (_all_nodes(dragon) | _all_nodes(mage))
        assert not shared, f"fountain shares {shared} with the fire side: dragon={dragon}, mage={mage}, {fountain}"

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "#1120 §3: running-mean text-centroid drift at the retired 0.40 (Exp 24–26) collapses the fountain "
            "into 'fire breath' once the mage has pulled the centroid; documentary: pins why 0.40 was retired; "
            "no fix is planned; an XPASS means text completion stopped drifting"
        ),
    )
    def test_retired_040_mage_present_collapse(self, bio_stack_at_retired_040):
        """At the RETIRED 0.40, dragon → mage → fountain: the fountain's compound is not the dragon's compound.

        Documents why 0.40 is retired (and that this file's old fixture threshold was the amplifier). Production
        runs at ``ECConfig()``; this gate deliberately does not. The precondition runs the same scenario WITHOUT
        the mage on a fresh stack at 0.40 and requires the fountain to separate there, so the collapse the
        assertion pins is the mage-driven drift and not the lower threshold alone.
        """
        _ec0, _atl0, _nac0, _scn0, encoder0 = _build_bio_stack(ECConfig(pattern_complete_threshold=_RETIRED_THRESHOLD))
        _require_semantic_encoder(encoder0)
        dragon0 = _encode_nodes_by_affordance(_dragon(), encoder0, "test_agent")
        fountain0 = _encode_nodes_by_affordance(_fountain(), encoder0, "test_agent")
        if fountain0["water_jet"][0] == dragon0["fire_breath"][0]:
            pytest.fail(
                f"precondition: at 0.40 WITHOUT the mage the fountain already collapses into the fire node, so "
                f"the gate cannot isolate mage-driven drift: dragon={dragon0}, fountain={fountain0}"
            )

        ec, atl, nac, scn, encoder = bio_stack_at_retired_040
        _require_semantic_encoder(encoder)
        agent_id = "test_agent"

        dragon = _encode_nodes_by_affordance(_dragon(), encoder, agent_id)
        _encode_nodes_by_affordance(_mage(), encoder, agent_id)
        fountain = _encode_nodes_by_affordance(_fountain(), encoder, agent_id)
        assert fountain["water_jet"][0] != dragon["fire_breath"][0], (
            f"fountain collapsed into the fire node: dragon={dragon}, fountain={fountain}"
        )


# ---------------------------------------------------------------------------
# IT-3: credit_node bookkeeping
# ---------------------------------------------------------------------------


class TestIT3SpecificOverridesAbstract:
    """NAc credit bookkeeping on one encoded node (encoder-independent)."""

    def test_credit_node_accumulates_on_one_node(self, bio_stack):
        """Repeated positive credit on one node accumulates its ``reward_bias`` (bounded).

        This pins ``credit_node`` bookkeeping only: it credits a node and reads the SAME node, so it says
        nothing about transfer between entities (that is IT-1). It holds under any encoder.
        """
        ec, atl, nac, scn, encoder = bio_stack
        agent_id = "test_agent"

        dragon = _encode_nodes_by_affordance(_dragon(), encoder, agent_id)
        node_id = dragon["fire_breath"][0]

        assert nac.reward_bias(agent_id, node_id) == 0.0, "expected zero bias before any credit"

        nac.credit_node(agent_id, node_id, 1.0)
        after_one = nac.reward_bias(agent_id, node_id)
        assert after_one > 0, f"one positive credit produced no bias ({after_one})"

        for _ in range(4):
            nac.credit_node(agent_id, node_id, 1.0)
        after_five = nac.reward_bias(agent_id, node_id)
        assert after_five > after_one, f"repeated credit did not accumulate: {after_one} -> {after_five}"


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
    """Agent A's learning does NOT affect Agent B's reward bias (encoder-independent)."""

    def test_agent_isolation(self, bio_stack):
        """Agent A learns fire benefit; Agent B sees no bias on the same node.

        Pins NAc's per-agent keying. Both agents encode the same entity into a shared EC/ATL, so they reach the
        same node under any encoder; no semantics needed.
        """
        ec, atl, nac, scn, encoder = bio_stack
        agent_a = "agent_a"
        agent_b = "agent_b"

        nodes_a = _encode_nodes_by_affordance(_dragon(), encoder, agent_a)
        nodes_b = _encode_nodes_by_affordance(_dragon(), encoder, agent_b)
        node_id = nodes_a["fire_breath"][0]
        assert nodes_b["fire_breath"][0] == node_id, f"agents did not share the node: {nodes_a}, {nodes_b}"

        nac.credit_node(agent_a, node_id, 1.0)

        bias_a = nac.reward_bias(agent_a, node_id)
        assert bias_a > 0, f"Agent A should have positive bias, got {bias_a}"

        bias_b = nac.reward_bias(agent_b, node_id)
        assert bias_b == 0.0, f"Agent B should have zero bias, got {bias_b}"


# ---------------------------------------------------------------------------
# IT-6: Self-affordance encoding
# ---------------------------------------------------------------------------


class TestIT6SelfAffordanceEncoding:
    """Agent's own body affordances form substrate concepts (encoder-independent)."""

    def test_self_affordances_create_substrate_concepts(self, bio_stack):
        """Agent body affordances (slash, move) create ATL substrate concepts.

        Encoder-independent: pins that the encoding path writes ATL concepts and NAc eligibility traces, not
        any similarity between them.
        """
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
        """Agent 'slash' and dragon 'slash' share the same EC node.

        Encoder-independent: both affordances are the IDENTICAL string "slash", which completes into one node
        under any encoder, the hash fallback included. It pins self/scene sharing of one name, not similarity.
        """
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
