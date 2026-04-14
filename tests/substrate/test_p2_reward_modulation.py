"""P2 reward-modulated recognition tests.

Two test surfaces:

1. **Mechanism tests (fast suite)** — synthetic embeddings with a
   stub encoder. Controls the vector geometry so the pass criteria
   can be asserted deterministically without the sentence-transformers
   dependency. These are the CI gate for the mechanism.

2. **Validation sweep (slow suite)** — paraphrase-mpnet-base-v2 on
   ``scenarios/substrate/p2_reward_modulation.yaml``. Produces the
   result JSON for ``docs/experiments/p2_reward_modulation_sweep.md``.
   Gated on the same ≥30% / ≤5% criteria.

Run fast subset::

    python -m pytest tests/substrate/test_p2_reward_modulation.py::TestP2Mechanism -xvs

Run full sweep (requires sentence-transformers)::

    python -m pytest tests/substrate/test_p2_reward_modulation.py::TestP2ValidationSweep -xvs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from maxim.decisions.nac import NACConfig, NAc
from maxim.memory.atl import ATL
from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import LinguisticEncoder

from tests.substrate.p2_metrics import (
    compute_p2_metrics,
    load_clusters_with_targets,
)

logger = logging.getLogger(__name__)

FIXTURE_PATH = Path(__file__).parent.parent.parent / "scenarios" / "substrate" / "p2_reward_modulation.yaml"
RESULTS_DIR = Path(__file__).parent.parent.parent / "docs" / "experiments" / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Mechanism tests — synthetic embeddings, fast suite
# ─────────────────────────────────────────────────────────────────────────────


def _make_synthetic_clusters(num_target: int = 5, num_distractor: int = 5):
    """Build clusters of (cluster_name, sentence_list) with synthetic IDs.

    Each cluster has 3 sentences named ``{cluster}_s{0,1,2}``. The
    sentence strings are placeholders — the actual embeddings come
    from a lookup table keyed by sentence string.
    """
    clusters = []
    for k in range(num_target):
        name = f"target_{k}"
        clusters.append({"name": name, "sentences": [f"{name}_s0", f"{name}_s1", f"{name}_s2"]})
    for k in range(num_distractor):
        name = f"distractor_{k}"
        clusters.append({"name": name, "sentences": [f"{name}_s0", f"{name}_s1", f"{name}_s2"]})
    targets = [f"target_{k}" for k in range(num_target)]
    return clusters, targets


def _build_embedding_table(
    num_target: int,
    num_distractor: int,
    dim: int = 20,
) -> dict[str, list[float]]:
    """Construct controlled embeddings for mechanism testing.

    For each cluster ``k`` assign two orthogonal basis indices:
    ``primary = k`` and ``secondary = k + num_target + num_distractor``.
    The three sentences of cluster ``k`` are then placed at
    controlled cosine distances from the primary axis::

        s0: e_primary                 (cos=1.0)
        s1: 0.92*e_primary + 0.392*e_secondary  (cos≈0.92)
        s2: 0.60*e_primary + 0.800*e_secondary  (cos=0.60)

    With base threshold 0.80 and centroid update, s2 lands below
    threshold in baseline (separates) but above ``0.80 - 0.20 = 0.60``
    in the rewarded pass (completes). Target clusters go 2→1 nodes;
    distractor clusters stay at 2→2 nodes.
    """
    num_clusters = num_target + num_distractor
    total = num_clusters * 2
    assert dim >= total, f"dim={dim} must be ≥ {total}"

    table: dict[str, list[float]] = {}

    def _basis(i: int) -> list[float]:
        v = [0.0] * dim
        v[i] = 1.0
        return v

    def _mix(a: float, i: int, b: float, j: int) -> list[float]:
        v = [0.0] * dim
        v[i] = a
        v[j] = b
        return v

    for k in range(num_clusters):
        if k < num_target:
            name = f"target_{k}"
        else:
            name = f"distractor_{k - num_target}"
        primary = k
        secondary = num_clusters + k
        table[f"{name}_s0"] = _basis(primary)
        table[f"{name}_s1"] = _mix(0.92, primary, 0.392, secondary)
        table[f"{name}_s2"] = _mix(0.60, primary, 0.80, secondary)

    return table


class StubEncoder(LinguisticEncoder):
    """LinguisticEncoder subclass that embeds from a static lookup.

    Used for the mechanism tests so we can control vector geometry
    without loading sentence-transformers.
    """

    def __init__(self, ec, atl, nac, embeddings: dict[str, list[float]]):
        super().__init__(ec=ec, atl=atl, nac=nac)
        self._emb_map = embeddings
        self._model_loaded = True
        self._using_fallback = False
        self._model = None

    def embed(self, text: str) -> list[float]:
        if text not in self._emb_map:
            raise KeyError(f"StubEncoder has no embedding for {text!r}")
        return list(self._emb_map[text])


class TestP2Mechanism:
    """Mechanism tests with controlled synthetic embeddings.

    Pass criteria match the plan: target_reduction ≥30%,
    distractor_interference ≤5%.
    """

    BASE_THRESHOLD = 0.80
    REWARD = 2.0  # drives bias to max_reward_bias cap (0.20)

    def _factory(self, embeddings, nac_config=None):
        """Build a (encoder, ec, atl, nac) factory closure."""

        def make(threshold: float):
            ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=threshold))
            atl = ATL()
            nac = NAc(nac_config or NACConfig())
            encoder = StubEncoder(ec, atl, nac, embeddings)
            return encoder, ec, atl, nac

        return make

    def test_rewarded_collapse_and_non_interference(self):
        """Target clusters collapse ≥30%; distractors change ≤5%."""
        clusters, targets = _make_synthetic_clusters()
        embeddings = _build_embedding_table(num_target=5, num_distractor=5)
        factory = self._factory(embeddings)

        metrics = compute_p2_metrics(
            clusters=clusters,
            target_clusters=targets,
            make_encoder_nac=factory,
            threshold=self.BASE_THRESHOLD,
            reward=self.REWARD,
            shuffle=False,
        )

        print("\n" + metrics.summary())
        print(metrics.per_cluster_table())

        # Baseline sanity: fragmentation must be present for the
        # rewarded collapse to be meaningful. With the synthetic
        # embedding geometry (s0=primary, s1=0.92 cos, s2=0.60 cos)
        # and baseline threshold 0.80, s2 separates at baseline
        # giving within-cluster pair-collapse rate of 1/3 per cluster.
        assert 0.2 < metrics.baseline_target_mean < 0.5, (
            f"baseline target collapse rate {metrics.baseline_target_mean:.2%} out of expected range "
            "(~33%) — synthetic geometry may have drifted"
        )
        # Target gain: need ≥+30 pp
        assert metrics.target_gain >= 0.30, f"target_gain={metrics.target_gain:+.1%} pp, need >=+30 pp"
        # Non-interference: distractor drift stays within 5 pp
        assert metrics.distractor_drift <= 0.05, f"distractor_drift={metrics.distractor_drift:.1%} pp, need <=5 pp"
        # Monotone direction: every target cluster must gain or stay flat
        # (never lose collapse rate under reward bias)
        assert metrics.target_monotone_fraction == 1.0, (
            f"target_monotone_fraction={metrics.target_monotone_fraction:.0%}, "
            "reward bias should never DECREASE cluster collapse rate"
        )

    def test_sanity_floor_target_monotone(self):
        """Reward bias is monotone non-decreasing on target clusters.

        The reward-bias pathway only WIDENS EC's recognition radius —
        it never tightens it. Rewarded collapse rate must be >=
        baseline collapse rate for every target cluster.
        """
        clusters, targets = _make_synthetic_clusters()
        embeddings = _build_embedding_table(num_target=5, num_distractor=5)
        factory = self._factory(embeddings)
        metrics = compute_p2_metrics(
            clusters=clusters,
            target_clusters=targets,
            make_encoder_nac=factory,
            threshold=self.BASE_THRESHOLD,
            reward=self.REWARD,
            shuffle=False,
        )
        for name in targets:
            base = metrics.baseline_rates[name]
            rew = metrics.rewarded_rates[name]
            assert rew >= base, f"{name}: rewarded {rew:.2%} < baseline {base:.2%}"

    def test_degenerate_control_alpha_zero(self):
        """With alpha=0, reward-bias pathway has no effect (control)."""
        clusters, targets = _make_synthetic_clusters()
        embeddings = _build_embedding_table(num_target=5, num_distractor=5)
        # alpha=0 → credit_node adds 0 to bias → no threshold change
        config = NACConfig(reward_bias_alpha=0.0)
        factory = self._factory(embeddings, nac_config=config)

        metrics = compute_p2_metrics(
            clusters=clusters,
            target_clusters=targets,
            make_encoder_nac=factory,
            threshold=self.BASE_THRESHOLD,
            reward=self.REWARD,
            shuffle=False,
        )
        # With alpha=0, rewarded and baseline must produce identical
        # per-cluster collapse rates. Target gain must be exactly 0.
        assert metrics.target_gain == 0.0, (
            f"alpha=0 but target_gain={metrics.target_gain:+.1%} pp (non-zero means the control is broken)"
        )
        assert metrics.rewarded_target_mean == metrics.baseline_target_mean

    def test_per_agent_isolation(self):
        """Rewarding agent A must not widen recognition for agent B.

        Uses direct NAc manipulation to verify
        ``get_threshold_overrides`` key on agent_id correctly.
        """
        nac = NAc()
        nac.credit_node(agent_id="alice", node_id="n1", reward=2.0)
        alice_overrides = nac.get_threshold_overrides("alice")
        bob_overrides = nac.get_threshold_overrides("bob")
        assert "n1" in alice_overrides
        assert "n1" not in bob_overrides
        assert alice_overrides["n1"] < 0.40  # base - positive bias
        # Bob sees base threshold unchanged
        assert bob_overrides == {}

    def test_decay_returns_bias_toward_baseline(self):
        """After repeated decay, reward_bias drops toward zero."""
        nac = NAc(NACConfig(reward_bias_decay_tau=10.0))
        nac.credit_node(agent_id="a", node_id="n", reward=2.0)
        initial = nac.reward_bias("a", "n")
        assert initial > 0.0

        # Apply decay many times
        for _ in range(200):
            nac.decay_reward_biases()

        final = nac.reward_bias("a", "n")
        # Either pruned below 0.001 threshold or strictly less than initial
        assert final < initial * 0.5, f"decay didn't shrink bias: initial={initial} final={final}"


# ─────────────────────────────────────────────────────────────────────────────
# Fixture loader test
# ─────────────────────────────────────────────────────────────────────────────


class TestFixtureLoader:
    """Sanity-check the YAML fixture shape."""

    @pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="p2_reward_modulation.yaml not found")
    def test_fixture_loads(self):
        clusters, targets = load_clusters_with_targets(str(FIXTURE_PATH))
        assert len(clusters) >= 10
        assert len(targets) == 5
        # Every cluster has ≥2 sentences
        for c in clusters:
            assert len(c["sentences"]) >= 2, f"cluster {c['name']} has too few sentences"
        # Targets are a subset of cluster names
        cluster_names = {c["name"] for c in clusters}
        for t in targets:
            assert t in cluster_names


# ─────────────────────────────────────────────────────────────────────────────
# Validation sweep — slow suite, requires sentence-transformers
# ─────────────────────────────────────────────────────────────────────────────


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _has_sentence_transformers(), reason="sentence-transformers not installed")
@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="p2_reward_modulation.yaml not found")
class TestP2ValidationSweep:
    """Full P2 validation sweep on the YAML fixture with paraphrase-mpnet.

    Produces results at ``docs/experiments/results/p2_reward_modulation_sweep.json``
    consumed by the lab notebook entry.
    """

    MODEL = "paraphrase-mpnet-base-v2"
    THRESHOLD = 0.70
    REWARD = 2.0

    def _factory(self):
        from maxim.similarity.encoder import EncoderConfig

        def make(threshold: float):
            ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=threshold))
            atl = ATL()
            nac = NAc()
            encoder = LinguisticEncoder(
                ec=ec,
                atl=atl,
                nac=nac,
                config=EncoderConfig(model_name=self.MODEL),
            )
            return encoder, ec, atl, nac

        return make

    def test_single_seed(self):
        """Smoke test — one seed end-to-end."""
        clusters, targets = load_clusters_with_targets(str(FIXTURE_PATH))
        metrics = compute_p2_metrics(
            clusters=clusters,
            target_clusters=targets,
            make_encoder_nac=self._factory(),
            threshold=self.THRESHOLD,
            reward=self.REWARD,
            shuffle=True,
            seed=0,
        )
        print("\n" + metrics.summary())
        print(metrics.per_cluster_table())
        assert len(metrics.baseline_rates) > 0

    def test_sweep_10_seeds(self):
        """Full 10-seed sweep — the P2 gate (collapse-rate semantics).

        Records results to docs/experiments/results/p2_reward_modulation_sweep.json.

        Pass criteria (Stage 3 restatement):
          - mean target_gain >= 30 pp
          - mean distractor_drift <= 5 pp
          - target monotone fraction (reward non-decreasing) >= 0.5
            on the average seed
        """
        import statistics

        clusters, targets = load_clusters_with_targets(str(FIXTURE_PATH))
        factory = self._factory()

        results = []
        for seed in range(10):
            metrics = compute_p2_metrics(
                clusters=clusters,
                target_clusters=targets,
                make_encoder_nac=factory,
                threshold=self.THRESHOLD,
                reward=self.REWARD,
                shuffle=True,
                seed=seed,
            )
            results.append(
                {
                    "seed": seed,
                    "target_gain": metrics.target_gain,
                    "distractor_drift": metrics.distractor_drift,
                    "baseline_target_mean": metrics.baseline_target_mean,
                    "rewarded_target_mean": metrics.rewarded_target_mean,
                    "baseline_distractor_mean": metrics.baseline_distractor_mean,
                    "rewarded_distractor_mean": metrics.rewarded_distractor_mean,
                    "target_monotone_fraction": metrics.target_monotone_fraction,
                    "final_bias_mean": metrics.final_bias_mean,
                    "passes": metrics.passes_p2(),
                }
            )
            print(f"\n[seed={seed}] {metrics.summary()}")

        gains = [r["target_gain"] for r in results]
        drifts = [r["distractor_drift"] for r in results]
        monotones = [r["target_monotone_fraction"] for r in results]
        mean_gain = statistics.mean(gains)
        std_gain = statistics.stdev(gains) if len(gains) > 1 else 0.0
        mean_drift = statistics.mean(drifts)
        std_drift = statistics.stdev(drifts) if len(drifts) > 1 else 0.0
        mean_monotone = statistics.mean(monotones)

        means_pass = mean_gain >= 0.30 and mean_drift <= 0.05 and mean_monotone >= 0.5
        seeds_passing = sum(1 for r in results if r["passes"])

        summary = {
            "model": self.MODEL,
            "threshold": self.THRESHOLD,
            "reward": self.REWARD,
            "metric": "within-cluster pair collapse rate, baseline vs rewarded",
            "pass_criteria": {
                "mean_target_gain_pp": 0.30,
                "mean_distractor_drift_pp": 0.05,
                "mean_target_monotone_fraction": 0.5,
            },
            "seeds": len(results),
            "seeds_passing": seeds_passing,
            "mean_target_gain": round(mean_gain, 4),
            "std_target_gain": round(std_gain, 4),
            "mean_distractor_drift": round(mean_drift, 4),
            "std_distractor_drift": round(std_drift, 4),
            "mean_target_monotone_fraction": round(mean_monotone, 4),
            "means_pass": means_pass,
            "per_seed": results,
        }

        print(f"\n{'=' * 60}")
        print(f"P2 Reward Modulation Sweep — {'PASS' if means_pass else 'FAIL'}")
        print(f"  Model: {self.MODEL} @ {self.THRESHOLD}")
        print(f"  Target gain:       {mean_gain:+.1%} pp ± {std_gain:.1%} (need >=+30 pp)")
        print(f"  Distractor drift:  {mean_drift:.1%} pp ± {std_drift:.1%} (need <=5 pp)")
        print(f"  Target monotone:   {mean_monotone:.0%} of clusters (need >=50%)")
        print(f"  Seeds passing individually: {seeds_passing}/{len(results)}")
        print(f"{'=' * 60}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "p2_reward_modulation_sweep.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {out_path}")

        assert means_pass, (
            f"P2 means failed: gain={mean_gain:+.1%} pp drift={mean_drift:.1%} pp monotone={mean_monotone:.0%}"
        )
