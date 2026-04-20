"""Integration tests for the Asset Foundry (E1 + E2).

Tests the full pipeline: generate → validate → SEM protocol → gauntlet → score.
Uses fallback generation (no LLM) so tests run offline.
E2 tests add mocked LLM router and entity context section coverage.

Read alongside:
- simulation/foundry.py — the foundry implementation
- agents/prompt_builder.py — build_entity_context_section
- docs/plans/deferred/asset_foundry_plan.md — design doc
"""

from __future__ import annotations

from unittest.mock import MagicMock

from maxim.simulation.foundry import (
    CandidateSpec,
    FoundryRunner,
    GauntletResult,
    ScoringConfig,
    generate_candidates,
    run_gauntlet,
    run_sem_protocol_tests,
    score_result,
    validate_candidate,
)


# ---------------------------------------------------------------------------
# F-0: Generation
# ---------------------------------------------------------------------------


class TestGeneration:
    def test_generate_fallback_produces_candidates(self):
        """Fallback generation (no LLM) produces valid candidates."""
        candidates = generate_candidates(
            theme="fantasy weapons",
            count=3,
            genre="fantasy",
            category="weapons",
            llm_router=None,
        )
        assert len(candidates) == 3
        for c in candidates:
            assert c.name
            assert c.category == "weapons"
            assert c.genre == "fantasy"
            assert c.source == "fallback"
            assert "sensors" in c.spec
            assert "modulators" in c.spec

    def test_generate_mixed_categories(self):
        """Without category, distributes across categories."""
        candidates = generate_candidates(
            theme="medieval",
            count=7,
            genre="fantasy",
            category=None,
        )
        categories = {c.category for c in candidates}
        assert len(categories) > 1, "Expected multiple categories"

    def test_unique_names(self):
        """Generated candidates have unique names."""
        candidates = generate_candidates(
            theme="test",
            count=5,
            genre="fantasy",
            category="weapons",
        )
        names = [c.name for c in candidates]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"


# ---------------------------------------------------------------------------
# F-1: Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_spec_passes(self):
        """A well-formed spec passes validation."""
        spec = {
            "name": "test_sword",
            "sensors": {
                "durability": {"unit": "ratio", "range": [0, 1], "initial": 0.8},
            },
            "modulators": {
                "combat": {
                    "affordances": {
                        "slash": {"params": {"target": "str"}, "description": "Slash"},
                    }
                }
            },
            "failure_modes": [
                {"name": "break", "trigger": {"field": "durability", "op": "<", "value": 0.1, "pain": 0.5}}
            ],
        }
        result = validate_candidate(spec)
        assert result.valid, f"Expected valid, got errors: {result.errors}"

    def test_missing_name_rejects(self):
        result = validate_candidate({"sensors": {}, "modulators": {}})
        assert not result.valid
        assert any("name" in e.lower() for e in result.errors)

    def test_no_sensors_rejects(self):
        result = validate_candidate({"name": "x", "modulators": {"m": {"affordances": {"a": {"description": "d"}}}}})
        assert not result.valid
        assert any("sensor" in e.lower() for e in result.errors)

    def test_no_affordances_rejects(self):
        result = validate_candidate(
            {
                "name": "x",
                "sensors": {"s": {"unit": "r", "range": [0, 1], "initial": 0.5}},
                "modulators": {},
            }
        )
        assert not result.valid

    def test_invalid_trigger_field_rejects(self):
        """Trigger referencing nonexistent sensor is rejected."""
        result = validate_candidate(
            {
                "name": "x",
                "sensors": {"durability": {"unit": "r", "range": [0, 1], "initial": 0.5}},
                "modulators": {"m": {"affordances": {"a": {"description": "d"}}}},
                "failure_modes": [
                    {"name": "f", "trigger": {"field": "nonexistent", "op": "<", "value": 0.1, "pain": 0.5}}
                ],
            }
        )
        assert not result.valid
        assert any("nonexistent" in e for e in result.errors)

    def test_pain_out_of_range_rejects(self):
        result = validate_candidate(
            {
                "name": "x",
                "sensors": {"s": {"unit": "r", "range": [0, 1], "initial": 0.5}},
                "modulators": {"m": {"affordances": {"a": {"description": "d"}}}},
                "failure_modes": [{"name": "f", "trigger": {"field": "s", "op": "<", "value": 0.1, "pain": 1.5}}],
            }
        )
        assert not result.valid

    def test_no_failure_modes_warns(self):
        """No failure modes is a warning, not a rejection."""
        result = validate_candidate(
            {
                "name": "x",
                "sensors": {"s": {"unit": "r", "range": [0, 1], "initial": 0.5}},
                "modulators": {"m": {"affordances": {"a": {"description": "d"}}}},
            }
        )
        assert result.valid
        assert any("failure" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# F-2: SEM Protocol Tests
# ---------------------------------------------------------------------------


class TestSEMProtocol:
    def test_valid_sword_passes_all(self):
        """The seed rusty_sword spec passes all SEM protocol tests."""
        from maxim.embodiment.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        spec = registry.get("weapons/rusty_sword")
        entity_spec = spec.get("entity", spec)

        result = run_sem_protocol_tests(entity_spec)
        assert result.valid, f"Expected valid, got errors: {result.errors}"

    def test_broken_spec_fails_instantiation(self):
        """A spec with invalid structure fails T1."""
        result = run_sem_protocol_tests({"name": "broken", "sensors": "not_a_dict"})
        assert not result.valid
        assert any("T1" in e for e in result.errors)

    def test_fallback_generated_passes(self):
        """Fallback-generated specs pass SEM protocol tests."""
        candidates = generate_candidates("test", count=1, genre="fantasy", category="weapons")
        assert len(candidates) == 1
        result = run_sem_protocol_tests(candidates[0].spec)
        assert result.valid, f"Fallback spec failed SEM tests: {result.errors}"


# ---------------------------------------------------------------------------
# F-2: Gauntlet
# ---------------------------------------------------------------------------


class TestGauntlet:
    def test_gauntlet_runs_on_seed_component(self):
        """Gauntlet exercises a known-good seed component."""
        from maxim.embodiment.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        spec = registry.get("weapons/rusty_sword")
        entity_spec = spec.get("entity", spec)

        candidate = CandidateSpec(
            name="rusty_sword",
            spec=entity_spec,
            category="weapons",
            genre="fantasy",
        )
        result = run_gauntlet(candidate)

        assert result.status in ("pass", "partial")
        assert result.encounters_completed >= 2
        assert len(result.affordances_used) > 0, "Expected at least one affordance used"

    def test_gauntlet_on_fallback_candidate(self):
        """Gauntlet runs on a fallback-generated candidate."""
        candidates = generate_candidates("test", count=1, genre="fantasy", category="weapons")
        result = run_gauntlet(candidates[0])
        assert result.status in ("pass", "partial", "infra_error")


# ---------------------------------------------------------------------------
# F-3: Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_perfect_score(self):
        """A perfect gauntlet result gets a high score."""
        gauntlet = GauntletResult(
            candidate_name="perfect",
            encounters_completed=3,
            encounters_total=3,
            hippocampal_captures=3,
            nac_observations=5,
            pain_signals=2,
            affordances_used={"slash", "parry", "throw"},
        )
        candidate = CandidateSpec(
            name="perfect",
            spec={
                "name": "perfect",
                "modulators": {
                    "combat": {
                        "affordances": {"slash": {}, "parry": {}, "throw": {}},
                    }
                },
                "failure_modes": [{"name": "f1"}, {"name": "f2"}],
            },
            category="weapons",
            genre="fantasy",
        )
        score = score_result(gauntlet, candidate)
        assert score.total_score >= 0.7
        assert score.bucket == "promote"

    def test_zero_engagement_rejected(self):
        """Zero bio-system engagement gets rejected."""
        gauntlet = GauntletResult(candidate_name="inert")
        candidate = CandidateSpec(
            name="inert",
            spec={"name": "inert", "modulators": {}, "failure_modes": []},
            category="items",
            genre="fantasy",
        )
        score = score_result(gauntlet, candidate)
        assert score.total_score < 0.4
        assert score.bucket == "reject"

    def test_infra_error_scores_zero(self):
        """Infrastructure errors score 0."""
        gauntlet = GauntletResult(candidate_name="broken", status="infra_error", error="OOM")
        candidate = CandidateSpec(
            name="broken",
            spec={"name": "broken", "modulators": {}, "failure_modes": []},
            category="items",
            genre="fantasy",
        )
        score = score_result(gauntlet, candidate)
        assert score.total_score == 0.0

    def test_custom_thresholds(self):
        """Custom scoring config changes bucket boundaries."""
        gauntlet = GauntletResult(
            candidate_name="borderline",
            encounters_completed=3,
            encounters_total=3,
            hippocampal_captures=1,
            nac_observations=1,
            pain_signals=0,
            affordances_used={"use"},
        )
        candidate = CandidateSpec(
            name="borderline",
            spec={
                "name": "borderline",
                "modulators": {"m": {"affordances": {"use": {}}}},
                "failure_modes": [],
            },
            category="items",
            genre="fantasy",
        )
        # With low thresholds, borderline should promote
        config = ScoringConfig(promote_threshold=0.2, reject_threshold=0.1)
        score = score_result(gauntlet, candidate, config)
        assert score.bucket == "promote"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestFoundryRunner:
    def test_dry_run(self, tmp_path, monkeypatch):
        """Dry run generates + validates but skips gauntlet."""
        from maxim.utils.paths import _reset_caches

        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        _reset_caches()

        try:
            runner = FoundryRunner(theme="test", genre="fantasy", category="weapons", dry_run=True)
            result = runner.run(count=3)

            assert result.generated == 3
            assert result.validated > 0
            assert result.tested == 0  # Dry run skips gauntlet
        finally:
            _reset_caches()

    def test_full_pipeline(self, tmp_path, monkeypatch):
        """Full pipeline generates, validates, tests, and scores."""
        from maxim.utils.paths import _reset_caches

        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        _reset_caches()

        try:
            runner = FoundryRunner(theme="test", genre="fantasy", category="weapons")
            result = runner.run(count=2)

            assert result.generated == 2
            assert result.validated > 0
            assert result.tested > 0

            # Check output directory structure
            run_dir = tmp_path / "foundry" / result.run_id
            assert (run_dir / "config.yaml").exists()
            assert (run_dir / "report.md").exists()
            assert (run_dir / "scores.json").exists()
            assert (run_dir / "candidates").is_dir()
        finally:
            _reset_caches()


# ---------------------------------------------------------------------------
# E2: LLM-driven generation (mocked router)
# ---------------------------------------------------------------------------

# Realistic entity spec JSON that a real LLM would return
_MOCK_LLM_ENTITY = {
    "name": "plasma_lance",
    "entity_type": "weapon",
    "metadata": {"genre": "cyberpunk", "foundry_theme": "cyberpunk weapons"},
    "synonyms": ["energy spear", "ion lance", "plasma pike", "shock javelin", "charged spear"],
    "sensors": {
        "charge": {"unit": "ratio", "range": [0, 1], "initial": 0.8},
        "durability": {"unit": "ratio", "range": [0, 1], "initial": 0.9},
        "heat": {"unit": "degrees", "range": [0, 500], "initial": 20},
    },
    "modulators": {
        "combat": {
            "affordances": {
                "plasma_thrust": {
                    "params": {"target": "str", "force": "float"},
                    "description": "High damage thrust, drains charge quickly",
                },
                "sweep_strike": {
                    "params": {"target": "str"},
                    "description": "Wide area attack, moderate charge drain",
                },
                "heat_vent": {
                    "params": {},
                    "description": "Release built-up heat, brief vulnerability window",
                },
            }
        }
    },
    "failure_modes": [
        {
            "name": "overheated",
            "trigger": {"field": "heat", "op": ">=", "value": 450, "pain": 0.6},
        },
        {
            "name": "charge_depleted",
            "trigger": {"field": "charge", "op": "<", "value": 0.05, "pain": 0.3},
        },
    ],
}


def _make_mock_router(spec: dict = _MOCK_LLM_ENTITY) -> MagicMock:
    """Build a mock LLM router that returns a realistic entity spec."""
    router = MagicMock()
    router.generate_json.return_value = dict(spec)
    return router


class TestLLMGeneration:
    """E2: foundry generation with a mocked LLM router."""

    def test_generate_with_llm_router(self):
        """LLM router produces candidates with source='llm'."""
        router = _make_mock_router()
        candidates = generate_candidates(
            theme="cyberpunk weapons",
            count=2,
            genre="cyberpunk",
            category="weapons",
            llm_router=router,
        )
        assert len(candidates) == 2
        for c in candidates:
            assert c.source == "llm"
            assert "sensors" in c.spec
            assert "modulators" in c.spec

    def test_llm_generated_includes_synonyms(self):
        """LLM-generated specs include the synonyms field."""
        router = _make_mock_router()
        candidates = generate_candidates(
            theme="test",
            count=1,
            genre="cyberpunk",
            category="weapons",
            llm_router=router,
        )
        assert len(candidates) == 1
        assert "synonyms" in candidates[0].spec
        assert len(candidates[0].spec["synonyms"]) >= 3

    def test_llm_generated_validates(self):
        """LLM-generated specs pass validation."""
        router = _make_mock_router()
        candidates = generate_candidates(
            theme="test",
            count=1,
            genre="cyberpunk",
            category="weapons",
            llm_router=router,
        )
        result = validate_candidate(candidates[0].spec)
        assert result.valid, f"Validation errors: {result.errors}"

    def test_llm_generated_passes_sem_protocol(self):
        """LLM-generated specs pass SEM protocol tests."""
        router = _make_mock_router()
        candidates = generate_candidates(
            theme="test",
            count=1,
            genre="cyberpunk",
            category="weapons",
            llm_router=router,
        )
        result = run_sem_protocol_tests(candidates[0].spec)
        assert result.valid, f"SEM protocol errors: {result.errors}"

    def test_llm_full_pipeline(self, tmp_path, monkeypatch):
        """Full foundry pipeline with mocked LLM completes generate → score."""
        from maxim.utils.paths import _reset_caches

        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        _reset_caches()

        try:
            router = _make_mock_router()
            runner = FoundryRunner(
                theme="cyberpunk weapons",
                genre="cyberpunk",
                category="weapons",
                llm_router=router,
            )
            result = runner.run(count=2)

            assert result.generated == 2
            assert result.validated > 0
            assert result.tested > 0
            # LLM-generated weapons with multiple affordances should score well
            all_scores = result.promoted + result.review + result.rejected
            assert len(all_scores) > 0, "Expected at least one scored candidate"
        finally:
            _reset_caches()

    def test_llm_failure_falls_back(self):
        """When LLM raises, generation falls back to template."""
        router = MagicMock()
        router.generate_json.side_effect = RuntimeError("LLM unavailable")
        candidates = generate_candidates(
            theme="test",
            count=1,
            genre="fantasy",
            category="weapons",
            llm_router=router,
        )
        assert len(candidates) == 1
        # Spec is still valid (came from template fallback inside design())
        result = validate_candidate(candidates[0].spec)
        assert result.valid, f"Fallback spec should be valid: {result.errors}"


# ---------------------------------------------------------------------------
# E2: Entity context section (prompt_builder)
# ---------------------------------------------------------------------------


class TestEntityContextSection:
    """Tests for build_entity_context_section in prompt_builder."""

    def test_builds_from_full_spec(self):
        """Produces strategy text from a complete entity spec."""
        from maxim.agents.prompt_builder import build_entity_context_section

        text = build_entity_context_section(_MOCK_LLM_ENTITY)
        assert "=== Entity: plasma_lance (weapon) ===" in text
        assert "plasma_thrust" in text
        assert "heat_vent" in text
        assert "Failure risk:" in text
        assert "heat >= 450" in text
        assert "charge < 0.05" in text

    def test_empty_for_sparse_spec(self):
        """Returns empty string for spec with no sensors or modulators."""
        from maxim.agents.prompt_builder import build_entity_context_section

        text = build_entity_context_section({"name": "empty"})
        assert text == ""

    def test_params_included(self):
        """Affordance params appear in the output."""
        from maxim.agents.prompt_builder import build_entity_context_section

        text = build_entity_context_section(_MOCK_LLM_ENTITY)
        assert "target: str" in text

    def test_no_failure_modes_ok(self):
        """Spec with modulators but no failure modes still produces output."""
        from maxim.agents.prompt_builder import build_entity_context_section

        spec = {
            "name": "simple",
            "entity_type": "item",
            "sensors": {"charge": {"unit": "ratio", "range": [0, 1], "initial": 1.0}},
            "modulators": {"use": {"affordances": {"activate": {"description": "Turn on"}}}},
        }
        text = build_entity_context_section(spec)
        assert "activate: Turn on" in text
        assert "Failure risk" not in text

    def test_entity_spec_field_on_llm_request(self):
        """LLMRequest dataclass accepts entity_spec field."""
        import dataclasses

        from maxim.agents.llm_types import LLMRequest

        fields = {f.name for f in dataclasses.fields(LLMRequest)}
        assert "entity_spec" in fields, "LLMRequest must have an entity_spec field"
