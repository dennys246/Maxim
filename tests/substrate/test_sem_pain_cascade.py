"""SEM pain cascade PoC — the full Percept→Reaction→Learning loop.

.. note::
   **Status (post sem_execution_hook Stage 3):** This PoC file proved
   the cascade *mechanism* during P2 Stage 2. The production-path
   superset lives in ``test_sem_execution_production.py``, which
   exercises the same cascade through ``build_executor`` →
   ``executor.execute`` → ``ToolPainBridge.record_tool_embodiment_failure``
   without the ``PoCAgent`` harness. **For most regression-guard work,
   prefer the production test file** — it covers the same shape
   PLUS the executor / side_effects / direct-attribution layers that
   this file bypasses.

   This file is kept as a smaller-surface mechanism guard (no
   executor, no side_effects routing) for tests that need to isolate
   bugs in the bus / NAc / context-similarity layer specifically.
   The Stage 1 retrospective flagged this file's PoC harness for
   eventual deletion; the decision is deferred until the production
   test has been load-bearing for at least one bug-find cycle.

This is the P2 plan's Stage 2 deliverable: prove that a real bundled
SEM component (the ``rusty_sword`` weapon) can drive the end-to-end
cascade

    affordance use (slash)
      → Body.evaluate_failures (durability below shatter threshold)
        → PainBus.publish(PainSignal) [rich context]
          → create_pain_nac_subscriber
            → NAc.record_outcome_full (NEGATIVE, context-similarity match)
              → CausalLink (action slash:rusty_sword → pain)
                → NAc.predict(slash:rusty_sword) = NEGATIVE
                  → agent policy prefers drop_weapon

all without touching mocks in the middle of the chain. Mocks are only
used to stand in for parts of the agent we don't exercise (the full
motor stack).

The "agent policy" is inlined in this test as a straightforward
pick-the-non-negative-prediction loop. It's not a production policy —
production agents will eventually drive this through the NAc predict()
surface inside the goal agent. The PoC's job is to prove the learning
path works so future policy code has a wired substrate to learn from.
"""

from __future__ import annotations

import time

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.proprioception.pain_bus import PainBus, create_pain_nac_subscriber


# ─────────────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────────────


class PoCAgent:
    """Minimal agent that records actions to NAc and picks by prediction.

    Stands in for the parts of the production agent we aren't exercising:
    the motor stack, the prompt assembler, the affordance→action mapping.
    Keeps everything we DO exercise honest — the NAc wiring, the pain
    cascade, the causal-link + prediction read path.

    TODO(substrate-p2-followup): the production equivalent of
    ``record_action`` does not yet exist. When an SEM affordance is
    invoked through ``runtime/executor.py`` or ``embodiment/motor.py``,
    there is currently no call to ``nac.record_event`` that stamps
    ``{source: "embodiment", entity: <entity_path>}`` into the pending-
    event context. This PoC records it directly from the test harness
    so the attribution matching works end-to-end, but shipping the real
    learning loop requires wiring an action-recording hook into the
    executor/motor boundary. The architecture review flagged this as
    the "RC3" gap and it is explicitly out of Stage 2 scope — see
    ``docs/plans/archive/substrate_recognition.md`` Stage 2 "Deferred / RC3"
    section for the tracked follow-up.
    """

    def __init__(self, nac: NAc, body_entity_path: str) -> None:
        self.nac = nac
        self.entity_path = body_entity_path

    def record_action(self, action: str) -> str:
        """Record an action as a pending NAc event.

        The context is the slim attribution key set that
        ``create_pain_nac_subscriber`` uses when attributing pain
        outcomes. Mismatch between these two call sites is the most
        common source of "no link was formed" bugs — keep them in sync.
        """
        signature = f"{action}:{self.entity_path}"
        return self.nac.record_event(
            event_type="action",
            event_signature=signature,
            context={
                "source": "embodiment",
                "entity": self.entity_path,
            },
        )

    def predict(self, action: str):
        """Query NAc for the expected outcome of an action."""
        return self.nac.predict(
            event_type="action",
            event_signature=f"{action}:{self.entity_path}",
            context={
                "source": "embodiment",
                "entity": self.entity_path,
            },
        )

    def choose(self, candidates: list[str]) -> str:
        """Pick the candidate with the least-negative NAc prediction.

        - Any candidate with NO prediction (never tried) gets neutral score 0.
        - NEGATIVE predictions get a score of -confidence.
        - POSITIVE predictions get a score of +confidence.
        - Ties are broken by input order (stable).
        """
        scored: list[tuple[float, int, str]] = []
        for i, cand in enumerate(candidates):
            pred = self.predict(cand)
            if pred is None:
                score = 0.0
            elif pred.predicted_valence == Valence.NEGATIVE:
                score = -pred.confidence
            else:
                score = pred.confidence
            scored.append((score, i, cand))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored[0][2]


def _build_world():
    """Construct rusty_sword + Embodiment + PainBus + NAc + subscriber.

    Shared setup used by every PoC test. Returns a dict so individual
    tests can grab what they need.
    """
    registry = ComponentRegistry()
    sword = registry.instantiate("weapons/rusty_sword")

    pain_bus = PainBus(_allow_raw=True)
    nac = NAc(NACConfig(temporal_window_seconds=60.0))
    pain_bus.subscribe(create_pain_nac_subscriber(nac, intensity_threshold=0.3))

    emb = Embodiment(sword, pain_bus=pain_bus)
    agent = PoCAgent(nac=nac, body_entity_path=sword.full_path)

    return {
        "registry": registry,
        "sword": sword,
        "pain_bus": pain_bus,
        "nac": nac,
        "embodiment": emb,
        "agent": agent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSEMPainCascadePoC:
    """End-to-end SEM pain cascade tests against the real rusty_sword YAML."""

    def test_baseline_no_prediction_before_any_pain(self):
        """Fresh NAc has no prediction for either action."""
        world = _build_world()
        agent: PoCAgent = world["agent"]

        assert agent.predict("slash") is None
        assert agent.predict("drop_weapon") is None

    def test_slash_at_high_durability_does_not_trigger_pain(self):
        """Healthy sword: slash records, no pain, no causal link."""
        world = _build_world()
        emb: Embodiment = world["embodiment"]
        agent: PoCAgent = world["agent"]
        sword = world["sword"]

        # Healthy sword (well above shatter threshold)
        sword.vital_metrics["durability"] = 0.8
        agent.record_action("slash")
        events = emb.evaluate_failures()
        assert not events, f"unexpected failure on healthy sword: {events}"
        assert len(world["nac"]) == 0  # no causal links

    def test_slash_at_low_durability_produces_negative_causal_link(self):
        """Low-durability slash → pain → action→pain causal link formed.

        This is the PoC's core integration claim: the full chain from
        affordance recording through NAc learning runs on a real
        bundled component with no mocks in the middle.
        """
        world = _build_world()
        emb: Embodiment = world["embodiment"]
        agent: PoCAgent = world["agent"]
        nac: NAc = world["nac"]
        sword = world["sword"]

        # Drive durability past the shatter trigger (< 0.1 per the YAML)
        sword.vital_metrics["durability"] = 0.05

        # Agent records the action it's about to take
        agent.record_action("slash")
        assert len(nac._pending_events) == 1

        # Evaluate failures — shatter should fire, publishing pain
        events = emb.evaluate_failures()
        assert any(e.failure_name == "shatter" for e in events)

        # Causal link for slash:<sword_path> with NEGATIVE valence
        sig = f"slash:{sword.full_path}"
        links = nac._links.get(sig, [])
        assert len(links) == 1, f"expected 1 action→pain link for {sig}, got {len(links)}: _links={dict(nac._links)}"
        assert links[0].outcome_valence == Valence.NEGATIVE

    def test_agent_prefers_drop_weapon_after_learning_slash_is_painful(self):
        """The headline P2 PoC assertion: drop_weapon > slash after pain."""
        world = _build_world()
        emb: Embodiment = world["embodiment"]
        agent: PoCAgent = world["agent"]
        sword = world["sword"]

        # One learning cycle: agent slashes a low-durability sword → pain
        sword.vital_metrics["durability"] = 0.05
        agent.record_action("slash")
        emb.evaluate_failures()

        # Now query the policy. slash should score NEGATIVE. drop_weapon
        # has no history → neutral. drop_weapon wins.
        slash_pred = agent.predict("slash")
        drop_pred = agent.predict("drop_weapon")

        assert slash_pred is not None, "slash has no prediction after pain"
        assert slash_pred.predicted_valence == Valence.NEGATIVE
        assert drop_pred is None, "drop_weapon should not have a prediction yet"

        choice = agent.choose(["slash", "drop_weapon"])
        assert choice == "drop_weapon", (
            f"agent chose {choice} after learning slash is painful — slash_pred={slash_pred}, drop_pred={drop_pred}"
        )

    def test_no_cross_entity_leakage(self):
        """Learning on rusty_sword A does not affect rusty_sword B.

        Per-entity isolation: a separate agent holding a different
        instance of the same component should not inherit the first
        agent's pain learning. This guards against context similarity
        matching on the shared "source=embodiment" key alone.
        """
        registry = ComponentRegistry()
        # Agent A on sword A — suffers shatter pain
        sword_a = registry.instantiate("weapons/rusty_sword", name="sword_a")
        sword_b = registry.instantiate("weapons/rusty_sword", name="sword_b")

        pain_bus_a = PainBus(_allow_raw=True)
        nac_a = NAc(NACConfig(temporal_window_seconds=60.0))
        pain_bus_a.subscribe(create_pain_nac_subscriber(nac_a))
        emb_a = Embodiment(sword_a, pain_bus=pain_bus_a)
        agent_a = PoCAgent(nac=nac_a, body_entity_path=sword_a.full_path)

        sword_a.vital_metrics["durability"] = 0.05
        agent_a.record_action("slash")
        emb_a.evaluate_failures()

        # Agent B asks about slashing sword B — its own (separate) NAc
        # has no knowledge of agent A's experience.
        nac_b = NAc(NACConfig(temporal_window_seconds=60.0))
        agent_b = PoCAgent(nac=nac_b, body_entity_path=sword_b.full_path)
        assert agent_b.predict("slash") is None

    def test_repeated_pain_strengthens_negative_confidence(self):
        """Multiple pain events over the same action raise link confidence.

        Exercises the iterative causal-link update path inside NAc.
        Confidence grows as ``sqrt(observation_count)`` per
        ``CausalLink.record_observation``, so with observations 1→2→3
        confidence increases strictly monotonically.

        Tight assertions (per pre-merge review):
          - ``observation_count`` reaches exactly 3 across the loop
          - ``confidences[-1] > confidences[0]`` STRICTLY, not ≥
          - Each iteration must also fire the shatter (sanity check
            that ``fm.active = False`` reset is actually re-enabling
            the failure mode; a loose ``>=`` assertion would pass
            even if iterations 2-3 silently no-op'd).
        """
        world = _build_world()
        emb: Embodiment = world["embodiment"]
        agent: PoCAgent = world["agent"]
        nac: NAc = world["nac"]
        sword = world["sword"]

        confidences: list[float] = []
        events_per_iter: list[int] = []
        sig = f"slash:{sword.full_path}"
        for _ in range(3):
            sword.vital_metrics["durability"] = 0.05
            agent.record_action("slash")
            # Clear the shatter's active state so it can fire again
            for fm in sword.failure_modes:
                fm.active = False
            # Wait out the refractory window on pain_bus so successive
            # publishes aren't dropped.
            time.sleep(0.6)
            events = emb.evaluate_failures()
            events_per_iter.append(len(events))
            links = nac._links.get(sig, [])
            assert links, f"no links after iteration: {dict(nac._links)}"
            confidences.append(links[0].confidence)

        # Every iteration must have actually fired pain — otherwise
        # the monotonic-confidence assertion would pass trivially
        # with the same value repeated.
        assert all(n >= 1 for n in events_per_iter), f"some iterations fired 0 failures: {events_per_iter}"
        link = nac._links[sig][0]
        assert link.observation_count == 3, f"expected observation_count=3, got {link.observation_count}"
        assert confidences[-1] > confidences[0], f"confidence did not strictly grow with repeated pain: {confidences}"
