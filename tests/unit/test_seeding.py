"""Tests for deterministic seeding (S4 — simulator_upgrades_plan)."""

from __future__ import annotations

import os
import random

import pytest

from maxim.utils.seeding import get_global_seed, make_agent_rng, seed_all


@pytest.fixture(autouse=True)
def _isolate_seed_state(monkeypatch):
    """Reset global seed state after each test."""
    import maxim.utils.seeding as mod

    original = mod._global_seed
    yield
    mod._global_seed = original
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)


# ── seed_all ─────────────────────────────────────────────────────────────────


class TestSeedAll:
    def test_sets_stdlib_random(self):
        seed_all(42)
        a = random.random()
        seed_all(42)
        b = random.random()
        assert a == b

    def test_sets_numpy_random(self):
        np = pytest.importorskip("numpy")
        seed_all(42)
        a = np.random.random()
        seed_all(42)
        b = np.random.random()
        assert a == b

    def test_sets_pythonhashseed_env(self):
        seed_all(123)
        assert os.environ.get("PYTHONHASHSEED") == "123"

    def test_different_seeds_produce_different_output(self):
        seed_all(42)
        a = random.random()
        seed_all(99)
        b = random.random()
        assert a != b

    def test_get_global_seed(self):
        assert get_global_seed() is None or isinstance(get_global_seed(), int)
        seed_all(42)
        assert get_global_seed() == 42

    def test_seed_zero(self):
        """Seed 0 is valid and produces deterministic output."""
        seed_all(0)
        a = random.random()
        seed_all(0)
        b = random.random()
        assert a == b
        assert get_global_seed() == 0


# ── make_agent_rng ───────────────────────────────────────────────────────────


class TestMakeAgentRng:
    def test_deterministic(self):
        rng1 = make_agent_rng(42, "agent_0")
        rng2 = make_agent_rng(42, "agent_0")
        assert rng1.random() == rng2.random()

    def test_different_agents_uncorrelated(self):
        rng_a = make_agent_rng(42, "agent_0")
        rng_b = make_agent_rng(42, "agent_1")
        # Generate sequences and check they differ
        seq_a = [rng_a.random() for _ in range(10)]
        seq_b = [rng_b.random() for _ in range(10)]
        assert seq_a != seq_b

    def test_different_seeds_uncorrelated(self):
        rng1 = make_agent_rng(42, "agent_0")
        rng2 = make_agent_rng(99, "agent_0")
        assert rng1.random() != rng2.random()

    def test_returns_random_instance(self):
        rng = make_agent_rng(42, "test")
        assert isinstance(rng, random.Random)

    def test_many_agents_all_different(self):
        """10 agents from the same base seed all produce distinct sequences."""
        first_values = set()
        for i in range(10):
            rng = make_agent_rng(42, f"agent_{i}")
            first_values.add(rng.random())
        assert len(first_values) == 10


# ── CLI parser ───────────────────────────────────────────────────────────────


class TestCLIParser:
    def test_seed_flag_parsed(self):
        from maxim.cli_parser import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--seed", "42"])
        assert args.seed == 42

    def test_seed_flag_default_none(self):
        from maxim.cli_parser import _build_parser

        parser = _build_parser()
        args = parser.parse_args([])
        assert args.seed is None


# ── semantic.py LSH seed ─────────────────────────────────────────────────────


class TestSemanticLSHSeed:
    def test_lsh_uses_global_seed(self):
        """When global seed is set, LSH planes are derived from it."""
        np = pytest.importorskip("numpy")

        seed_all(42)
        from maxim.similarity.semantic import SemanticEmbedderConfig, NeuralSemanticLSH

        cfg = SemanticEmbedderConfig(require_gpu=False)
        e1 = NeuralSemanticLSH(config=cfg)
        if not e1._healthy:
            pytest.skip("SemanticEngine not healthy (missing deps)")

        seed_all(42)
        e2 = NeuralSemanticLSH(config=cfg)

        np.testing.assert_array_equal(e1._random_planes, e2._random_planes)

    def test_lsh_different_seed_different_planes(self):
        """Different global seeds produce different LSH planes."""
        np = pytest.importorskip("numpy")

        seed_all(42)
        from maxim.similarity.semantic import SemanticEmbedderConfig, NeuralSemanticLSH

        cfg = SemanticEmbedderConfig(require_gpu=False)
        e1 = NeuralSemanticLSH(config=cfg)
        if not e1._healthy:
            pytest.skip("SemanticEngine not healthy (missing deps)")

        seed_all(99)
        e2 = NeuralSemanticLSH(config=cfg)

        assert not np.array_equal(e1._random_planes, e2._random_planes)
