"""Integration tests for the Asset Foundry (E1).

Tests the full pipeline: generate → validate → SEM protocol → gauntlet → score.
Uses fallback generation (no LLM) so tests run offline.

Read alongside:
- simulation/foundry.py — the foundry implementation
- docs/plans/deferred/asset_foundry_plan.md — design doc
"""

from __future__ import annotations


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
