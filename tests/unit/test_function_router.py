"""Tests for runtime/function_router.py — Lane Tier Architecture Phase 1."""

from __future__ import annotations

import pytest

from maxim.runtime.function_router import (
    DEFAULT_FUNCTIONS,
    DEFAULT_TIER_ORDER,
    FunctionRouter,
    FunctionSpec,
    TierRestrictionError,
    TierUnavailableError,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

ALL_TIERS = {"large", "medium", "small"}


@pytest.fixture()
def full_router() -> FunctionRouter:
    """Router with all three tiers available."""
    return FunctionRouter(available_tiers=ALL_TIERS)


@pytest.fixture()
def two_tier_router() -> FunctionRouter:
    """Router with large + small (no medium) — typical leader setup."""
    return FunctionRouter(available_tiers={"large", "small"})


@pytest.fixture()
def no_gpu_router() -> FunctionRouter:
    """Router with medium + small only — Mac peer / CPU-only setup."""
    return FunctionRouter(available_tiers={"medium", "small"})


# ─── Basic resolution ────────────────────────────────────────────────────────


class TestBasicResolution:
    def test_primary_tier_available(self, full_router: FunctionRouter) -> None:
        tier, boost = full_router.resolve("agent_inference")
        assert tier == "large"
        assert boost == 2  # restricted function gets priority boost

    def test_small_tier_function(self, full_router: FunctionRouter) -> None:
        tier, boost = full_router.resolve("concept_extraction")
        assert tier == "small"
        assert boost == 0

    def test_medium_tier_function(self, full_router: FunctionRouter) -> None:
        tier, boost = full_router.resolve("campaign_summary")
        assert tier == "medium"
        assert boost == 0

    def test_all_default_functions_resolve_with_all_tiers(self, full_router: FunctionRouter) -> None:
        """Every default function should resolve when all tiers exist."""
        for name in DEFAULT_FUNCTIONS:
            tier, _boost = full_router.resolve(name)
            assert tier in ALL_TIERS, f"{name} resolved to unknown tier {tier!r}"


# ─── Fallback chains ────────────────────────────────────────────────────────


class TestFallbackChains:
    def test_fallback_to_medium(self, no_gpu_router: FunctionRouter) -> None:
        """fear_review: large → [medium]. No GPU → should fall back to medium."""
        tier, boost = no_gpu_router.resolve("fear_review")
        assert tier == "medium"

    def test_fallback_to_small(self) -> None:
        """campaign_summary: medium → [small]. Only small available."""
        router = FunctionRouter(available_tiers={"small"})
        tier, _boost = router.resolve("campaign_summary")
        assert tier == "small"

    def test_fallback_skips_unavailable(self) -> None:
        """concept_extraction: small → [medium]. Only large available — should fail."""
        router = FunctionRouter(available_tiers={"large"})
        with pytest.raises(TierUnavailableError):
            router.resolve("concept_extraction")

    def test_fallback_tracks_count(self, no_gpu_router: FunctionRouter) -> None:
        """Fallback events should be counted for diagnostics."""
        no_gpu_router.resolve("fear_review")
        no_gpu_router.resolve("fear_review")
        stats = no_gpu_router.fallback_stats()
        assert stats["fear_review"] == 2

    def test_no_fallback_count_when_primary_available(self, full_router: FunctionRouter) -> None:
        full_router.resolve("fear_review")
        assert full_router.fallback_stats() == {}

    def test_upgrade_fallback(self, two_tier_router: FunctionRouter) -> None:
        """concept_extraction: small → [medium]. Medium missing, should fail."""
        # small IS available, so it resolves to primary
        tier, _boost = two_tier_router.resolve("concept_extraction")
        assert tier == "small"


# ─── Restrictions ────────────────────────────────────────────────────────────


class TestRestrictions:
    def test_restricted_function_fails_when_tier_missing(self) -> None:
        """agent_inference has fallback=() — must fail if large is gone."""
        router = FunctionRouter(available_tiers={"medium", "small"})
        with pytest.raises(TierRestrictionError, match="restricted"):
            router.resolve("agent_inference")

    def test_restricted_small_function_fails_when_small_missing(self) -> None:
        """json_repair: small, fallback=() — must fail if small is gone."""
        router = FunctionRouter(available_tiers={"large", "medium"})
        with pytest.raises(TierRestrictionError, match="restricted"):
            router.resolve("json_repair")

    def test_restricted_function_works_when_tier_present(self, full_router: FunctionRouter) -> None:
        tier, _boost = full_router.resolve("agent_inference")
        assert tier == "large"

    def test_narrative_transcription_restricted_to_small(self) -> None:
        """narrative_transcription: small, fallback=() — never bounces up."""
        router = FunctionRouter(available_tiers={"large", "medium"})
        with pytest.raises(TierRestrictionError):
            router.resolve("narrative_transcription")


# ─── Unknown functions ───────────────────────────────────────────────────────


class TestUnknownFunctions:
    def test_unknown_function_routes_to_largest(self, full_router: FunctionRouter) -> None:
        tier, boost = full_router.resolve("some_new_function")
        assert tier == "large"
        assert boost == 0

    def test_unknown_function_routes_to_medium_when_no_large(self, no_gpu_router: FunctionRouter) -> None:
        tier, _boost = no_gpu_router.resolve("some_new_function")
        assert tier == "medium"

    def test_unknown_function_routes_to_small_when_only_small(self) -> None:
        router = FunctionRouter(available_tiers={"small"})
        tier, _boost = router.resolve("some_new_function")
        assert tier == "small"


# ─── No tiers available ─────────────────────────────────────────────────────


class TestNoTiers:
    def test_no_tiers_raises_restriction_for_restricted(self) -> None:
        """Restricted function with no tiers → TierRestrictionError."""
        router = FunctionRouter(available_tiers=set())
        with pytest.raises(TierRestrictionError, match="restricted"):
            router.resolve("agent_inference")

    def test_no_tiers_raises_unavailable_for_non_restricted(self) -> None:
        """Non-restricted function with no tiers → TierUnavailableError."""
        router = FunctionRouter(available_tiers=set())
        with pytest.raises(TierUnavailableError):
            router.resolve("fear_review")

    def test_no_tiers_unknown_function_raises(self) -> None:
        router = FunctionRouter(available_tiers=set())
        with pytest.raises(TierUnavailableError, match="No tiers available"):
            router.resolve("unknown")


# ─── Priority boost ─────────────────────────────────────────────────────────


class TestPriorityBoost:
    def test_agent_inference_has_boost(self, full_router: FunctionRouter) -> None:
        _, boost = full_router.resolve("agent_inference")
        assert boost == 2

    def test_non_restricted_has_zero_boost(self, full_router: FunctionRouter) -> None:
        _, boost = full_router.resolve("concept_extraction")
        assert boost == 0

    def test_boost_preserved_on_fallback(self) -> None:
        """Custom function with boost should keep it after fallback."""
        funcs = {
            "important_task": FunctionSpec(
                name="important_task",
                tier="large",
                fallback=("medium",),
                priority_boost=3,
            ),
        }
        router = FunctionRouter(functions=funcs, available_tiers={"medium", "small"})
        tier, boost = router.resolve("important_task")
        assert tier == "medium"
        assert boost == 3


# ─── Callable available_tiers (mesh-ready) ───────────────────────────────────


class TestCallableAvailableTiers:
    def test_callable_tiers_are_queried_each_resolve(self) -> None:
        """Callable should be re-evaluated on each resolve() call."""
        tiers = {"small"}

        def get_tiers() -> set[str]:
            return tiers

        router = FunctionRouter(available_tiers=get_tiers)

        # Only small available — agent_inference (restricted to large) fails
        with pytest.raises(TierRestrictionError):
            router.resolve("agent_inference")

        # Peer comes online, advertises large
        tiers.add("large")
        tier, _boost = router.resolve("agent_inference")
        assert tier == "large"

    def test_callable_tier_removal(self) -> None:
        """Tier disappearing mid-session should cause fallback."""
        tiers: set[str] = {"large", "medium", "small"}

        router = FunctionRouter(available_tiers=lambda: tiers)

        tier, _boost = router.resolve("fear_review")
        assert tier == "large"

        # GPU goes down
        tiers.discard("large")
        tier, _boost = router.resolve("fear_review")
        assert tier == "medium"

    def test_static_set_works_as_before(self) -> None:
        """Plain set should still work (wrapped in lambda internally)."""
        router = FunctionRouter(available_tiers={"large", "small"})
        tier, _boost = router.resolve("agent_inference")
        assert tier == "large"


# ─── Health check callback ───────────────────────────────────────────────────


class TestHealthCheck:
    def test_health_check_can_block_tier(self) -> None:
        """Health check returning False should skip the tier."""

        def health(tier: str) -> bool:
            return tier != "large"  # large is "unhealthy"

        router = FunctionRouter(
            available_tiers=ALL_TIERS,
            health_check=health,
        )
        # fear_review: large → [medium]; large unhealthy → medium
        tier, _boost = router.resolve("fear_review")
        assert tier == "medium"

    def test_health_check_blocks_restricted_function(self) -> None:
        """Restricted function with unhealthy tier should raise."""

        def health(tier: str) -> bool:
            return tier != "large"

        router = FunctionRouter(
            available_tiers=ALL_TIERS,
            health_check=health,
        )
        with pytest.raises(TierRestrictionError):
            router.resolve("agent_inference")

    def test_health_check_not_called_for_missing_tier(self) -> None:
        """Health check should NOT be called if tier isn't in available set."""
        calls: list[str] = []

        def health(tier: str) -> bool:
            calls.append(tier)
            return True

        router = FunctionRouter(
            available_tiers={"small"},
            health_check=health,
        )
        router.resolve("concept_extraction")
        # Only "small" should be checked, not "large" or "medium"
        assert calls == ["small"]


# ─── Custom tier ordering ───────────────────────────────────────────────────


class TestTierOrdering:
    def test_custom_tier_order(self) -> None:
        """Custom tier_order should affect _largest_available."""
        router = FunctionRouter(
            available_tiers={"medium", "small"},
            tier_order=["small", "medium", "large"],  # prefer small
        )
        tier, _boost = router.resolve("unknown_function")
        assert tier == "small"  # smallest-first ordering

    def test_four_tier_support(self) -> None:
        """Extra tier names should work with custom ordering."""
        funcs = {
            "heavy_task": FunctionSpec(
                name="heavy_task",
                tier="xlarge",
                fallback=("large",),
            ),
        }
        router = FunctionRouter(
            functions=funcs,
            available_tiers={"xlarge", "large", "medium", "small"},
            tier_order=["xlarge", "large", "medium", "small"],
        )
        tier, _boost = router.resolve("heavy_task")
        assert tier == "xlarge"


# ─── Custom function tables ─────────────────────────────────────────────────


class TestCustomFunctions:
    def test_custom_function_table(self) -> None:
        funcs = {
            "my_func": FunctionSpec(name="my_func", tier="small"),
        }
        router = FunctionRouter(functions=funcs, available_tiers={"small"})
        tier, _boost = router.resolve("my_func")
        assert tier == "small"

    def test_none_functions_uses_defaults(self) -> None:
        router = FunctionRouter(functions=None, available_tiers=ALL_TIERS)
        tier, _boost = router.resolve("agent_inference")
        assert tier == "large"


# ─── Introspection methods ──────────────────────────────────────────────────


class TestIntrospection:
    def test_get_function(self, full_router: FunctionRouter) -> None:
        spec = full_router.get_function("agent_inference")
        assert spec is not None
        assert spec.tier == "large"
        assert spec.fallback == ()

    def test_get_function_missing(self, full_router: FunctionRouter) -> None:
        assert full_router.get_function("nonexistent") is None

    def test_list_functions_returns_copy(self, full_router: FunctionRouter) -> None:
        funcs = full_router.list_functions()
        assert "agent_inference" in funcs
        # Mutating the copy shouldn't affect the router
        funcs.pop("agent_inference")
        assert full_router.get_function("agent_inference") is not None

    def test_available_tiers_snapshot(self, full_router: FunctionRouter) -> None:
        tiers = full_router.available_tiers()
        assert tiers == ALL_TIERS

    def test_fallback_stats_empty_initially(self, full_router: FunctionRouter) -> None:
        assert full_router.fallback_stats() == {}


# ─── Default function table consistency ──────────────────────────────────────


class TestDefaultFunctionTable:
    def test_all_tiers_referenced_exist_in_default_order(self) -> None:
        """Every tier and fallback tier in DEFAULT_FUNCTIONS should be a known tier."""
        known = set(DEFAULT_TIER_ORDER)
        for name, spec in DEFAULT_FUNCTIONS.items():
            assert spec.tier in known, f"{name}.tier={spec.tier!r} not in {known}"
            for fb in spec.fallback:
                assert fb in known, f"{name}.fallback contains {fb!r} not in {known}"

    def test_no_self_referential_fallback(self) -> None:
        """A function's fallback chain should not include its own primary tier."""
        for name, spec in DEFAULT_FUNCTIONS.items():
            assert spec.tier not in spec.fallback, f"{name}: primary tier {spec.tier!r} is also in fallback chain"

    def test_restricted_functions_have_boost(self) -> None:
        """Functions with empty fallback (restricted) should have priority_boost > 0."""
        for name, spec in DEFAULT_FUNCTIONS.items():
            if not spec.fallback and spec.tier == "large":
                assert spec.priority_boost > 0, (
                    f"{name}: restricted to 'large' but priority_boost={spec.priority_boost}"
                )

    def test_name_matches_key(self) -> None:
        """Each spec's name field should match its dict key."""
        for key, spec in DEFAULT_FUNCTIONS.items():
            assert key == spec.name, f"Key {key!r} != spec.name {spec.name!r}"

    def test_fallback_is_tuple(self) -> None:
        """All fallback fields should be tuples, not lists (frozen dataclass safety)."""
        for name, spec in DEFAULT_FUNCTIONS.items():
            assert isinstance(spec.fallback, tuple), (
                f"{name}.fallback is {type(spec.fallback).__name__}, expected tuple"
            )


# ─── Thread safety ───────────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_fallback_counting(self) -> None:
        """Fallback counts must be accurate under concurrent resolve() calls."""
        import threading

        router = FunctionRouter(available_tiers={"medium", "small"})
        num_threads = 8
        calls_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(calls_per_thread):
                router.resolve("fear_review")  # large → [medium] fallback

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = router.fallback_stats()
        assert stats["fear_review"] == num_threads * calls_per_thread


# ─── FunctionSpec immutability ───────────────────────────────────────────────


class TestFunctionSpecImmutability:
    def test_frozen_spec_rejects_attribute_assignment(self) -> None:
        spec = DEFAULT_FUNCTIONS["agent_inference"]
        with pytest.raises(AttributeError):
            spec.tier = "small"  # type: ignore[misc]

    def test_fallback_tuple_rejects_mutation(self) -> None:
        """Tuple fallback cannot be mutated (unlike list)."""
        spec = DEFAULT_FUNCTIONS["concept_extraction"]
        with pytest.raises(AttributeError):
            spec.fallback.append("large")  # type: ignore[attr-defined]


# ─── Exception handling in callables ─────────────────────────────────────────


class TestCallableExceptionHandling:
    def test_failing_available_tiers_callable(self) -> None:
        """If available_tiers callable raises, tier is treated as unavailable."""

        def broken_tiers() -> set[str]:
            raise RuntimeError("mesh down")

        router = FunctionRouter(
            functions={"test": FunctionSpec(name="test", tier="large", fallback=("small",))},
            available_tiers=broken_tiers,
        )
        # Should not crash — should treat all tiers as unavailable
        with pytest.raises(TierUnavailableError):
            router.resolve("test")

    def test_failing_health_check(self) -> None:
        """If health_check raises, tier is treated as unavailable."""

        def broken_health(tier: str) -> bool:
            raise ConnectionError("peer unreachable")

        router = FunctionRouter(
            available_tiers=ALL_TIERS,
            health_check=broken_health,
        )
        # fear_review: large → [medium] — both health checks fail
        with pytest.raises(TierUnavailableError):
            router.resolve("fear_review")
