"""Exteroception/interoception multi-modality seam — regression guards.

Root cause pinned here (docs/plans/exteroception_interoception_seam.md +
memory ``reference_extero_intero_dilution_root_cause``): the substrate-primary
proposer merged exteroceptive sensors (azimuth) into ONE ``encode_sensors``
call with the interoceptive drives, so direction was one term in a text-embed
sum dominated by ~5 drives — left(-0.7) and right(+0.7) pattern-completed onto
the SAME EC cluster and the agent was structurally blind to direction (the
embodied cradle orient sim at chance).

The seam fix: one ``encode_sensors(modality=tag)`` call per declarative
:class:`~maxim.embodiment.sensory_streams.ModalityChannel` (interoception,
audio), a ``{modality: cluster_id}`` set through ``recommend_action`` (additive
bias sum) and ``record_outcome`` (credit routed by source). EC is already
multi-modality; these tests pin the plumbing that was collapsing it.
"""

from __future__ import annotations

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.spec import _parse_entity
from maxim.runtime.agent_loop import propose_via_substrate
from maxim.similarity.ec import EntorhinalCortex
from maxim.similarity.encoder import SensorEncoder

# ── fixtures ──────────────────────────────────────────────────────────────


def _entropic_drive(initial: float) -> dict:
    """A [0,1] entropic drive with NO drift — values stay put between calls."""
    return {
        "unit": "ratio",
        "range": [0, 1],
        "initial": initial,
        "drive": {
            "drift_mode": "entropic",
            "drift_direction": "up",
            "drift_rate": 0.0,
            "deprivation_threshold": 0.7,
            "deprivation_pain": 0.3,
            "satisfaction_threshold": 0.3,
        },
    }


def _multi_drive_body() -> Embodiment:
    """The dilution fixture: 5 interoceptive drives + a driveless azimuth
    sensor — the ``bodies/infant_operant`` shape that measured at chance."""
    d = {
        "name": "infant",
        "entity_type": "body",
        "sensors": {
            "azimuth": {"unit": "normalized", "range": [-1, 1], "initial": 0.0},  # NO drive
            "hunger": _entropic_drive(0.5),
            "thirst": _entropic_drive(0.4),
            "fatigue": _entropic_drive(0.3),
            "boredom": _entropic_drive(0.2),
            "loneliness": _entropic_drive(0.6),
        },
    }
    return Embodiment(root=_parse_entity(d))


class _StubRegistry:
    def __init__(self, names):
        self._names = list(names)

    def list(self):  # noqa: D401 - stub
        return list(self._names)


class _StubExecutor:
    def __init__(self, names, embodiment=None):
        self.registry = _StubRegistry(names)
        self.embodiment = embodiment


class _ClusterRecordingNac:
    """Stub NAc capturing the cluster context recommend_action receives."""

    def __init__(self):
        self.seen_cluster_id: str | None = None
        self.seen_clusters: dict | None = None

    def recommend_action(self, *, agent_id, available_tools, current_drives=None, **kwargs):  # noqa: D401 - stub
        self.seen_cluster_id = kwargs.get("current_cluster_id")
        clusters = kwargs.get("current_clusters")
        self.seen_clusters = dict(clusters) if clusters else None
        return None


def _cluster_context(nac: _ClusterRecordingNac) -> dict[str, str]:
    """Union view of the cluster context: fold the legacy scalar in as
    interoception so the assertion is meaningful both pre- and post-seam."""
    ctx = dict(nac.seen_clusters or {})
    if nac.seen_cluster_id and "interoception" not in ctx:
        ctx["interoception"] = nac.seen_cluster_id
    return ctx


# ── THE dilution regression (probe-first: written failing) ────────────────


class TestDilutionRegression:
    def test_left_right_distinct_cluster_context_on_multi_drive_body(self):
        """A multi-drive body hearing a sound hard-left(-0.7) vs hard-right
        (+0.7) MUST present recommend_action with distinguishable cluster
        context. Pre-seam this fails: the merged {drives+azimuth} encode
        collapses both onto one interoception cluster (the measured
        ``8ee443bb`` collapse) → the orient policy has no state to condition
        on and the embodied infant measures at chance."""
        emb = _multi_drive_body()
        body = emb.root
        executor = _StubExecutor(["infant_turn_left", "infant_turn_right"], embodiment=emb)
        enc = SensorEncoder(ec=EntorhinalCortex(), atl=None)
        nac = _ClusterRecordingNac()

        body.vital_metrics["azimuth"] = -0.7
        propose_via_substrate(nac=nac, agent_id="infant", executor=executor, sensor_encoder=enc)
        left = _cluster_context(nac)

        body.vital_metrics["azimuth"] = 0.7
        propose_via_substrate(nac=nac, agent_id="infant", executor=executor, sensor_encoder=enc)
        right = _cluster_context(nac)

        assert left, "left proposal produced no cluster context at all"
        assert right, "right proposal produced no cluster context at all"
        assert left != right, (
            "left(-0.7) and right(+0.7) produced IDENTICAL cluster context "
            f"({left}) — exteroceptive direction diluted among the drives; "
            "the agent is blind to left-vs-right"
        )
        # The direction lives in its OWN exteroceptive cluster space, and the
        # interoceptive cluster is direction-blind by design (labeled lines).
        assert left.get("audio") and right.get("audio")
        assert left["audio"] != right["audio"]
        assert left.get("interoception") == right.get("interoception")

    def test_single_channel_body_keeps_legacy_interoception_shape(self):
        """A drives-only body (no exteroceptive sensor — the reachy / Exp 42
        shape) produces exactly ONE interoception cluster, with the legacy
        ``current_cluster_id`` scalar still populated — byte-identical
        selection inputs for pre-seam bodies."""
        d = {
            "name": "b",
            "entity_type": "body",
            "sensors": {"hunger": _entropic_drive(0.5), "thirst": _entropic_drive(0.4)},
        }
        emb = Embodiment(root=_parse_entity(d))
        executor = _StubExecutor(["b_warm_self"], embodiment=emb)
        enc = SensorEncoder(ec=EntorhinalCortex(), atl=None)
        nac = _ClusterRecordingNac()
        propose_via_substrate(nac=nac, agent_id="b", executor=executor, sensor_encoder=enc)
        assert nac.seen_clusters is not None
        assert set(nac.seen_clusters) == {"interoception"}
        assert nac.seen_cluster_id == nac.seen_clusters["interoception"]


# ── ModalityClusters loud guard + legacy fold ─────────────────────────────


class TestModalityClustersGuard:
    def test_none_normalizes_to_empty(self):
        from maxim.decisions.nac import require_valid_modality_clusters

        assert require_valid_modality_clusters(None) == {}

    def test_empty_tag_raises(self):
        from maxim.decisions.nac import require_valid_modality_clusters

        with pytest.raises(ValueError, match="modality tag"):
            require_valid_modality_clusters({"": "node-1"})

    def test_empty_cluster_id_raises(self):
        from maxim.decisions.nac import require_valid_modality_clusters

        with pytest.raises(ValueError, match="cluster id"):
            require_valid_modality_clusters({"audio": ""})

    def test_returns_a_copy(self):
        from maxim.decisions.nac import require_valid_modality_clusters

        src = {"audio": "a"}
        out = require_valid_modality_clusters(src)
        out["audio"] = "mutated"
        assert src["audio"] == "a"

    def test_fold_legacy_scalar_becomes_interoception(self):
        from maxim.decisions.nac import fold_legacy_cluster_id

        assert fold_legacy_cluster_id(None, "n1") == {"interoception": "n1"}

    def test_fold_explicit_set_wins_over_scalar(self):
        from maxim.decisions.nac import fold_legacy_cluster_id

        folded = fold_legacy_cluster_id({"interoception": "explicit"}, "scalar")
        assert folded["interoception"] == "explicit"

    def test_interoception_tag_pinned_across_modules(self):
        # decisions/ must not import embodiment/, so the tag is duplicated;
        # this pin is the drift guard.
        from maxim.decisions.nac import INTEROCEPTION_MODALITY
        from maxim.embodiment.sensory_streams import INTEROCEPTION_TAG

        assert INTEROCEPTION_MODALITY == INTEROCEPTION_TAG == "interoception"

    def test_modality_channel_rejects_empty_tag(self):
        from maxim.embodiment.sensory_streams import ModalityChannel

        with pytest.raises(ValueError):
            ModalityChannel("", lambda e: {}, lambda e: {})

    def test_recommend_action_raises_on_malformed_clusters(self):
        from maxim.decisions.nac import NAc, NACConfig

        nac = NAc(NACConfig())
        with pytest.raises(ValueError):
            nac.recommend_action(
                agent_id="a",
                available_tools=["t"],
                current_clusters={"audio": ""},
            )


# ── recommend_action: additive multi-cluster bias ─────────────────────────


class TestMultiClusterSelection:
    def _nac(self):
        from maxim.decisions.nac import NAc, NACConfig

        return NAc(NACConfig())

    def test_bias_sums_additively_across_clusters(self):
        """A tool rewarded in the audio cluster AND the interoception cluster
        scores the SUM; a tool rewarded in only one scores that one term. The
        additive sum is the deliberately binding-free late convergence."""
        nac = self._nac()
        # Saturate biases well past the 0.3 gate (alpha 0.3, cap 1.0).
        for _ in range(30):
            nac.update_cluster_reward(agent_id="a", cluster_id="intero-1", tool_signature="tool:both", reward=0.5)
            nac.update_cluster_reward(agent_id="a", cluster_id="audio-L", tool_signature="tool:both", reward=0.5)
            nac.update_cluster_reward(agent_id="a", cluster_id="audio-L", tool_signature="tool:audio_only", reward=1.0)
        both = nac.cluster_reward_bias("a", "intero-1", "tool:both") + nac.cluster_reward_bias(
            "a", "audio-L", "tool:both"
        )
        audio_only = nac.cluster_reward_bias("a", "audio-L", "tool:audio_only")
        assert both > audio_only  # the sum beats the single term

        rec = nac.recommend_action(
            agent_id="a",
            available_tools=["both", "audio_only"],
            current_clusters={"interoception": "intero-1", "audio": "audio-L"},
            min_confidence=0.0,
        )
        assert rec is not None
        assert rec["tool_name"] == "both"
        assert "cluster_bias[interoception]" in rec["reasoning"]
        assert "cluster_bias[audio]" in rec["reasoning"]

    def test_audio_cluster_alone_differentiates_direction(self):
        """The de-dilution payoff at the selection layer: with identical
        interoceptive state, opposite audio clusters flip the recommended
        turn — the substrate can now condition action on direction."""
        nac = self._nac()
        for _ in range(20):
            nac.update_cluster_reward(agent_id="a", cluster_id="audio-L", tool_signature="tool:turn_left", reward=1.0)
            nac.update_cluster_reward(agent_id="a", cluster_id="audio-R", tool_signature="tool:turn_right", reward=1.0)

        rec_l = nac.recommend_action(
            agent_id="a",
            available_tools=["turn_left", "turn_right"],
            current_clusters={"interoception": "intero-1", "audio": "audio-L"},
            min_confidence=0.0,
        )
        rec_r = nac.recommend_action(
            agent_id="a",
            available_tools=["turn_left", "turn_right"],
            current_clusters={"interoception": "intero-1", "audio": "audio-R"},
            min_confidence=0.0,
        )
        assert rec_l is not None and rec_l["tool_name"] == "turn_left"
        assert rec_r is not None and rec_r["tool_name"] == "turn_right"

    def test_legacy_scalar_equivalent_to_folded_set(self):
        """Byte-identical back-compat: ``current_cluster_id=X`` and
        ``current_clusters={"interoception": X}`` produce the same
        recommendation (same tool, same confidence)."""
        nac = self._nac()
        for _ in range(20):
            nac.update_cluster_reward(agent_id="a", cluster_id="c1", tool_signature="tool:warm", reward=1.0)
        via_scalar = nac.recommend_action(
            agent_id="a",
            available_tools=["warm", "idle"],
            current_cluster_id="c1",
            min_confidence=0.0,
        )
        via_set = nac.recommend_action(
            agent_id="a",
            available_tools=["warm", "idle"],
            current_clusters={"interoception": "c1"},
            min_confidence=0.0,
        )
        assert via_scalar is not None and via_set is not None
        assert via_scalar["tool_name"] == via_set["tool_name"] == "warm"
        assert via_scalar["confidence"] == via_set["confidence"]

    def test_no_clusters_is_byte_identical_legacy_path(self):
        """LLM-primary / single-cluster agents pass no clusters — empty sum,
        selection driven by causal links + drive affinity exactly as before."""
        nac = self._nac()
        rec = nac.recommend_action(
            agent_id="a",
            available_tools=["warm_self"],
            current_drives={"cold": 0.9},
            min_confidence=0.0,
        )
        assert rec is not None
        assert rec["tool_name"] == "warm_self"
        assert "cluster_bias" not in rec["reasoning"]


# ── record_outcome: credit routing by source ──────────────────────────────


class _CtxStub:
    def add_outcome(self, **kwargs):  # noqa: D401 - stub
        pass


def _record(nac, *, clusters=None, cluster_id=None, drive_potential_diff=None, success=True):
    from maxim.runtime.tool_dispatch import record_outcome

    record_outcome(
        agent_id="a",
        tool_name="turn_left",
        success=success,
        result_summary="ok",
        error=None,
        reasoning="",
        recent_outcomes=[],
        max_recent=10,
        llm_worker=None,
        context_pool=_CtxStub(),
        nac=nac,
        clusters=clusters,
        cluster_id=cluster_id,
        drive_potential_diff=drive_potential_diff,
    )


class TestCreditRouting:
    def _nac(self):
        from maxim.decisions.nac import NAc, NACConfig

        return NAc(NACConfig())

    def test_generic_success_writes_interoception_only(self):
        """The write-side complement of de-dilution: uniform tool-success
        credit lands on the interoception cluster and NEVER on the
        direction-bearing audio cluster (probe 3: the uniform floor drowns
        the operant signal it leaks onto)."""
        nac = self._nac()
        _record(nac, clusters={"interoception": "I1", "audio": "A1"})
        assert nac.cluster_reward_bias("a", "I1", "tool:turn_left") > 0.0
        assert nac.cluster_reward_bias("a", "A1", "tool:turn_left") == 0.0

    def test_drive_relief_writes_interoception_only(self):
        nac = self._nac()
        _record(nac, clusters={"interoception": "I1", "audio": "A1"}, drive_potential_diff=0.2)
        assert nac.cluster_reward_bias("a", "I1", "tool:turn_left") > 0.0
        assert nac.cluster_reward_bias("a", "A1", "tool:turn_left") == 0.0

    def test_extero_only_body_books_no_generic_cluster_credit(self):
        """A body with an audio cluster but no interoception cluster gets NO
        generic tool-success cluster write at all — there is no interoceptive
        slot, and the audio cluster is off-limits to the uniform floor."""
        nac = self._nac()
        _record(nac, clusters={"audio": "A1"})
        assert nac.cluster_reward_bias("a", "A1", "tool:turn_left") == 0.0

    def test_operant_pending_action_keyed_on_audio_cluster(self, monkeypatch):
        """Operant routing: in operant-only mode the pending action is keyed
        on the DIRECTION-BEARING audio cluster, so the caregiver's
        credit_operant_reward reinforces (audio_cluster, tool) — the pair the
        orient policy conditions on."""
        monkeypatch.setenv("MAXIM_OPERANT_ONLY_CREDIT", "1")
        nac = self._nac()
        _record(nac, clusters={"interoception": "I1", "audio": "A1"})
        credited = nac.credit_operant_reward("a", 1.0)
        assert credited == ("A1", "tool:turn_left")
        assert nac.cluster_reward_bias("a", "A1", "tool:turn_left") > 0.0
        assert nac.cluster_reward_bias("a", "I1", "tool:turn_left") == 0.0

    def test_operant_pending_falls_back_to_interoception_without_extero(self, monkeypatch):
        """Single-cluster bodies (pre-seam probes): no exteroceptive channel →
        the pending operant action keys on interoception, as before."""
        monkeypatch.setenv("MAXIM_OPERANT_ONLY_CREDIT", "1")
        nac = self._nac()
        _record(nac, cluster_id="I1")
        assert nac.credit_operant_reward("a", 1.0) == ("I1", "tool:turn_left")

    def test_legacy_scalar_routes_exactly_as_before(self):
        """Pre-seam callers passing only cluster_id are byte-identical: the
        scalar folds to interoception and takes the generic write."""
        nac = self._nac()
        _record(nac, cluster_id="C1")
        assert nac.cluster_reward_bias("a", "C1", "tool:turn_left") > 0.0

    def test_malformed_clusters_raise_loudly(self):
        with pytest.raises(ValueError):
            _record(self._nac(), clusters={"audio": ""})


# ── LLMProposal carries the cluster set ───────────────────────────────────


class TestProposalClusters:
    def test_proposal_defaults_clusters_none(self):
        from maxim.agents.llm_worker import LLMProposal

        p = LLMProposal(
            request_id="r",
            action=None,
            reasoning="",
            strategy_used=None,
            confidence=0.0,
            mode_goal_achieved=False,
        )
        assert p.clusters is None
        assert p.cluster_id is None

    def test_propose_via_substrate_populates_both_alias_and_set(self):
        """The proposal carries the full set AND the legacy interoception
        alias, so pre-seam consumers reading ``cluster_id`` keep working."""
        from maxim.decisions.nac import NAc, NACConfig

        emb = _multi_drive_body()
        emb.root.vital_metrics["azimuth"] = -0.7
        executor = _StubExecutor(["infant_turn_left", "infant_turn_right"], embodiment=emb)
        enc = SensorEncoder(ec=EntorhinalCortex(), atl=None)
        # Explore bonus guarantees a never-tried tool clears the (zeroed)
        # gate, so a cold substrate still returns a proposal to inspect.
        nac = NAc(NACConfig(substrate_explore_bonus_weight=0.5))
        proposal = propose_via_substrate(
            nac=nac, agent_id="infant", executor=executor, sensor_encoder=enc, min_confidence=0.0
        )
        assert proposal is not None
        assert proposal.clusters is not None
        assert set(proposal.clusters) == {"interoception", "audio"}
        assert proposal.cluster_id == proposal.clusters["interoception"]


# ── Closing test: the multi-drive orient probe LEARNS end-to-end ──────────


class TestMultiDriveOrientLearnsEndToEnd:
    """The seam's raison d'être (probe-4 shape, multi-drive body, production
    paths): a hungry infant with 5 interoceptive drives and a driveless
    azimuth sensor, taught ONLY by a mother's contingent operant reward,
    learns to turn toward the sound. Pre-seam this sat at chance (~0.5) —
    the drives diluted direction out of the encode. Runs the REAL
    propose_via_substrate → record_outcome → credit_operant_reward loop
    (no LLM, deterministic embedding hash, ~1s)."""

    def test_orient_rises_from_chance_to_learned(self, monkeypatch):
        import numpy as np

        from maxim.decisions.nac import NAc, NACConfig
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool
        from maxim.runtime.tool_dispatch import record_outcome

        monkeypatch.setenv("MAXIM_OPERANT_ONLY_CREDIT", "1")

        rng = np.random.default_rng(7)
        nac = NAc(NACConfig())
        enc = SensorEncoder(ec=EntorhinalCortex(), atl=None)

        d = {
            "name": "infant",
            "entity_type": "body",
            "sensors": {
                "azimuth": {"unit": "normalized", "range": [-1, 1], "initial": 0.0},  # NO drive
                "hunger": _entropic_drive(0.5),
                "thirst": _entropic_drive(0.4),
                "fatigue": _entropic_drive(0.3),
                "boredom": _entropic_drive(0.2),
                "loneliness": _entropic_drive(0.6),
            },
            "modulators": {
                "orient": {
                    "abstract": True,
                    "affordances": {
                        "turn_left": {"params": {}, "description": "l", "self_effect": {"azimuth": 0.3}},
                        "turn_right": {"params": {}, "description": "r", "self_effect": {"azimuth": -0.3}},
                    },
                }
            },
        }
        emb = Embodiment(root=_parse_entity(d))
        body = emb.root
        mod = body.modulators["orient"]
        actions = ["turn_left", "turn_right"]
        tools = {a: ModulatorAffordanceTool(body, mod, a, mod.affordances[a], a, embodiment=emb) for a in actions}
        executor = _StubExecutor(actions, embodiment=emb)

        epsilon = 0.2
        ticks = 400
        directed: list[int] = []
        for _ in range(ticks):
            side = -1.0 if rng.random() < 0.5 else 1.0
            az = side * float(rng.uniform(0.3, 0.9))
            body.vital_metrics["azimuth"] = az

            proposal = propose_via_substrate(
                nac=nac,
                agent_id="infant",
                executor=executor,
                sensor_encoder=enc,
                min_confidence=0.0,
            )
            if rng.random() < epsilon or proposal is None or not proposal.action:
                action = actions[int(rng.integers(len(actions)))]
                clusters = proposal.clusters if proposal is not None else None
            else:
                action = proposal.action["tool_name"]
                clusters = proposal.clusters

            tools[action].execute()
            az_after = float(body.vital_metrics.get("azimuth", az))
            progress = abs(az) - abs(az_after)
            was_directed = progress > 1e-9
            directed.append(1 if was_directed else 0)

            # Production outcome path: registers the pending operant action
            # (keyed on the audio cluster) and routes any generic credit.
            record_outcome(
                agent_id="infant",
                tool_name=action,
                success=True,
                result_summary="turned",
                error=None,
                reasoning="",
                recent_outcomes=[],
                max_recent=5,
                llm_worker=None,
                context_pool=_CtxStub(),
                nac=nac,
                clusters=clusters,
            )
            # The mother: contingent operant reward for turning toward her.
            if was_directed:
                nac.credit_operant_reward("infant", 1.0)

        first = sum(directed[:50]) / 50.0
        settled = sum(directed[-100:]) / 100.0
        # Pre-seam: settled ≈ 0.5 (chance) — direction diluted, nothing to
        # condition on. Post-seam: the audio cluster carries direction and
        # the operant credit lands on it, so the policy forms. ε=0.2 caps
        # the ceiling at ~1 − ε/2 = 0.9.
        assert settled >= 0.75, (
            f"multi-drive orient did not learn: settled directedness {settled:.2f} "
            f"(first-50 {first:.2f}, chance ≈ 0.5) — the seam is not carrying "
            "direction to the selection surface"
        )
