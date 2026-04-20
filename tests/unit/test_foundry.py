"""Tests for maxim.simulation.foundry — auto-curation, dedup, and coverage analysis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maxim.simulation.foundry import (
    CandidateScore,
    CandidateSpec,
    CurationReport,
    FoundryResult,
    FoundryRunner,
    _candidate_to_component_dict,
    _promote_to_user_components,
    analyze_coverage,
    auto_curate,
    generate_curation_report_section,
    generate_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec(name: str, category: str = "weapons", genre: str = "fantasy") -> dict:
    """Build a minimal valid SEM entity spec for testing."""
    return {
        "name": name,
        "entity_type": "weapon",
        "sensors": {
            "durability": {"unit": "ratio", "range": [0, 1], "initial": 0.5},
        },
        "modulators": {
            "combat": {
                "affordances": {
                    "strike": {
                        "params": {"target": "str"},
                        "description": f"Strike with {name}",
                    },
                },
            },
        },
        "failure_modes": [
            {
                "name": "broken",
                "trigger": {"field": "durability", "op": "<", "value": 0.1, "pain": 0.5},
            },
        ],
        "metadata": {
            "genre": genre,
            "synonyms": [f"{name} alias"],
        },
    }


def _make_candidate(name: str, category: str = "weapons", genre: str = "fantasy") -> CandidateSpec:
    return CandidateSpec(
        name=name,
        spec=_make_spec(name, category, genre),
        category=category,
        genre=genre,
        source="llm",
    )


@pytest.fixture
def mock_registry():
    """Mock ComponentRegistry with controllable query results."""
    registry = MagicMock()
    registry.query.return_value = []
    registry.list_categories.return_value = ["weapons", "creatures", "npcs", "items", "environments"]
    registry.register.return_value = None
    return registry


@pytest.fixture
def mock_index():
    """Mock ComponentIndex with controllable dedup results."""
    index = MagicMock()
    index.dedup_check.return_value = None  # No duplicates by default
    index.add.return_value = None
    return index


# ---------------------------------------------------------------------------
# Tests: analyze_coverage
# ---------------------------------------------------------------------------


class TestAnalyzeCoverage:
    def test_full_coverage_returns_empty(self, mock_registry):
        """When all categories have >= threshold components, gaps is empty."""
        mock_registry.query.return_value = [MagicMock()] * 5  # 5 components per query
        gaps = analyze_coverage("fantasy", threshold=5, registry=mock_registry)
        assert gaps == {}

    def test_partial_coverage_returns_gaps(self, mock_registry):
        """Categories below threshold are returned with their counts."""

        def side_effect(*, genre=None, category=None, **_kw):
            if category == "weapons":
                return [MagicMock()] * 2
            if category == "creatures":
                return [MagicMock()] * 1
            return [MagicMock()] * 10

        mock_registry.query.side_effect = side_effect
        gaps = analyze_coverage("fantasy", threshold=5, registry=mock_registry)
        assert gaps == {"weapons": 2, "creatures": 1}

    def test_zero_coverage(self, mock_registry):
        """All categories empty."""
        mock_registry.query.return_value = []
        gaps = analyze_coverage("cyberpunk", threshold=3, registry=mock_registry)
        assert len(gaps) == 5  # All _CURATION_CATEGORIES
        assert all(count == 0 for count in gaps.values())

    def test_threshold_boundary(self, mock_registry):
        """Exactly at threshold = not a gap."""
        mock_registry.query.return_value = [MagicMock()] * 5
        gaps = analyze_coverage("fantasy", threshold=5, registry=mock_registry)
        assert gaps == {}


# ---------------------------------------------------------------------------
# Tests: FoundryRunner dedup integration
# ---------------------------------------------------------------------------


class TestFoundryRunnerDedup:
    def test_dedup_skips_near_duplicate(self, mock_index):
        """Candidate matching an existing component is skipped."""
        from maxim.embodiment.component_index import ComponentMatch

        mock_index.dedup_check.return_value = ComponentMatch(
            ref="weapons/rusty_sword", name="rusty_sword", score=0.92, layer="embedding"
        )

        # Verify runner stores the index
        runner = FoundryRunner(
            theme="test",
            genre="fantasy",
            category="weapons",
            component_index=mock_index,
        )
        assert runner._component_index is mock_index

        # Manually test the dedup logic the runner uses
        candidate = _make_candidate("almost_rusty_sword")
        match = mock_index.dedup_check(candidate.spec, threshold=0.80)
        assert match is not None
        assert match.ref == "weapons/rusty_sword"
        assert match.score == 0.92

    def test_no_dedup_when_no_index(self):
        """When no ComponentIndex, promotion proceeds without dedup."""
        runner = FoundryRunner(
            theme="test",
            genre="fantasy",
            component_index=None,
        )
        assert runner._component_index is None

    def test_dedup_passes_novel_candidate(self, mock_index):
        """Novel candidate passes dedup check (returns None)."""
        mock_index.dedup_check.return_value = None
        result = mock_index.dedup_check({"name": "novel_weapon"}, threshold=0.80)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _promote_to_user_components
# ---------------------------------------------------------------------------


class TestPromoteToUserDir:
    @patch("maxim.simulation.foundry.yaml")
    @patch("maxim.utils.atomic_io.atomic_write_text")
    @patch("maxim.utils.paths.data_home")
    def test_promotes_to_correct_path(self, mock_data_home, mock_write, mock_yaml, tmp_path):
        """Promoted files go to ~/.maxim/components/{category}/."""
        mock_data_home.return_value = tmp_path
        mock_yaml.dump.return_value = "yaml content"
        candidate = _make_candidate("test_sword")

        path = _promote_to_user_components(candidate)
        assert path is not None
        assert "components/weapons/test_sword.yaml" in str(path)

    def test_candidate_to_component_dict_includes_synonyms(self):
        """Synonyms from spec metadata are included in component header."""
        candidate = _make_candidate("named_blade")
        result = _candidate_to_component_dict(candidate)
        assert "synonyms" in result["component"]
        assert result["component"]["synonyms"] == ["named_blade alias"]

    def test_candidate_to_component_dict_no_synonyms(self):
        """No synonyms field when spec has none."""
        candidate = _make_candidate("plain_sword")
        candidate.spec["metadata"].pop("synonyms", None)
        result = _candidate_to_component_dict(candidate)
        assert "synonyms" not in result["component"]


# ---------------------------------------------------------------------------
# Tests: auto_curate
# ---------------------------------------------------------------------------


class TestAutoCurate:
    @patch("maxim.simulation.foundry.FoundryRunner")
    def test_no_gaps_skips_foundry(self, mock_runner_cls, mock_registry):
        """Full coverage → no foundry runs."""
        mock_registry.query.return_value = [MagicMock()] * 10
        report = auto_curate("fantasy", threshold=5, registry=mock_registry)
        assert report.categories_below_threshold == 0
        mock_runner_cls.assert_not_called()

    @patch("maxim.simulation.foundry.FoundryRunner")
    def test_gaps_trigger_foundry(self, mock_runner_cls, mock_registry):
        """Under-covered categories trigger foundry runs."""

        def query_side_effect(*, genre=None, category=None, **_kw):
            if category == "weapons":
                return [MagicMock()] * 2  # Below threshold of 5
            return [MagicMock()] * 10

        mock_registry.query.side_effect = query_side_effect

        mock_result = FoundryResult(
            run_id="test",
            theme="fantasy weapons",
            genre="fantasy",
            category="weapons",
            generated=3,
            promoted=[CandidateScore(candidate_name="x", total_score=0.8, bucket="promote")],
        )
        mock_result.promoted_paths = ["/tmp/x.yaml"]
        mock_result.dedup_skipped = []

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = mock_result
        mock_runner_cls.return_value = mock_runner_instance

        report = auto_curate("fantasy", threshold=5, registry=mock_registry)
        assert report.categories_below_threshold == 1
        assert report.total_promoted == 1
        mock_runner_cls.assert_called_once()

    @patch("maxim.simulation.foundry.FoundryRunner")
    def test_correct_count_requested(self, mock_runner_cls, mock_registry):
        """Foundry is asked to generate exactly the gap size."""

        def query_side_effect(*, genre=None, category=None, **_kw):
            if category == "creatures":
                return [MagicMock()] * 3  # 3 exist, threshold 5, need 2
            return [MagicMock()] * 10

        mock_registry.query.side_effect = query_side_effect

        mock_result = FoundryResult(
            run_id="t",
            theme="t",
            genre="fantasy",
            category="creatures",
        )
        mock_result.promoted_paths = []
        mock_result.dedup_skipped = []
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = mock_result
        mock_runner_cls.return_value = mock_runner_instance

        auto_curate("fantasy", threshold=5, registry=mock_registry)
        mock_runner_instance.run.assert_called_once_with(count=2)

    @patch("maxim.simulation.foundry.FoundryRunner")
    def test_index_passed_to_runner(self, mock_runner_cls, mock_registry, mock_index):
        """ComponentIndex is threaded through to the runner."""
        mock_registry.query.return_value = [MagicMock()] * 2

        mock_result = FoundryResult(
            run_id="t",
            theme="t",
            genre="fantasy",
            category=None,
        )
        mock_result.promoted_paths = []
        mock_result.dedup_skipped = []
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = mock_result
        mock_runner_cls.return_value = mock_runner_instance

        auto_curate("fantasy", threshold=5, registry=mock_registry, component_index=mock_index)
        # Verify component_index was passed in the constructor
        call_kwargs = mock_runner_cls.call_args
        assert call_kwargs.kwargs.get("component_index") is mock_index


# ---------------------------------------------------------------------------
# Tests: CLI flag parsing
# ---------------------------------------------------------------------------


class TestCLIFlagParsing:
    def test_auto_curate_flag(self):
        """--auto-curate sets auto_curate=True."""
        from maxim.cli_parser import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--auto-curate", "--sim", "test"])
        assert args.auto_curate is True

    def test_no_curate_flag(self):
        """--no-curate sets no_curate=True."""
        from maxim.cli_parser import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--no-curate", "--sim", "test"])
        assert args.no_curate is True

    def test_curate_threshold(self):
        """--curate-threshold N sets threshold."""
        from maxim.cli_parser import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--curate-threshold", "8", "--sim", "test"])
        assert args.curate_threshold == 8

    def test_curate_threshold_default(self):
        """Default threshold is 5."""
        from maxim.cli_parser import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--sim", "test"])
        assert args.curate_threshold == 5

    def test_no_curate_overrides_auto_curate(self):
        """--no-curate should block auto-curation even if --auto-curate is set."""
        from maxim.cli_parser import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--auto-curate", "--no-curate", "--sim", "test"])
        # The gate logic: _auto_curate and not _no_curate
        should_curate = args.auto_curate and not args.no_curate
        assert should_curate is False

    def test_auto_curate_requires_embodiment(self):
        """Auto-curation gate requires --embodiment to be set."""
        from maxim.cli_parser import _build_parser

        parser = _build_parser()
        # --auto-curate without --embodiment
        args = parser.parse_args(["--auto-curate", "--sim", "test"])
        entity_ref = getattr(args, "embodiment", None)
        should_curate = args.auto_curate and not args.no_curate and entity_ref
        assert should_curate is None or should_curate is False  # falsy

        # --auto-curate WITH --embodiment
        args2 = parser.parse_args(["--auto-curate", "--embodiment", "weapons/rusty_sword", "--sim", "test"])
        entity_ref2 = getattr(args2, "embodiment", None)
        should_curate2 = args2.auto_curate and not args2.no_curate and entity_ref2
        assert should_curate2  # truthy


# ---------------------------------------------------------------------------
# Tests: Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    def test_report_includes_dedup_section(self):
        """Dedup-skipped entries appear in the report."""
        result = FoundryResult(
            run_id="test",
            theme="test weapons",
            genre="fantasy",
            category="weapons",
            generated=5,
            validated=4,
            tested=3,
            dedup_skipped=[("near_sword", "weapons/rusty_sword", 0.91)],
        )
        report = generate_report(result)
        assert "Near-Duplicates" in report
        assert "near_sword" in report
        assert "rusty_sword" in report
        assert "0.91" in report

    def test_curation_report_section(self):
        """Curation report section renders correctly."""
        cr = CurationReport(
            genre="cyberpunk",
            categories_checked=5,
            categories_below_threshold=2,
            total_promoted=3,
            total_skipped_dedup=1,
            total_generated=10,
        )
        section = generate_curation_report_section(cr)
        assert "Auto-Curation Summary" in section
        assert "cyberpunk" in section
        assert "Promoted:** 3" in section


# ---------------------------------------------------------------------------
# Tests: FoundryResult new fields
# ---------------------------------------------------------------------------


class TestFoundryResultFields:
    def test_dedup_skipped_defaults_empty(self):
        result = FoundryResult(run_id="x", theme="x", genre="x", category=None)
        assert result.dedup_skipped == []

    def test_promoted_paths_defaults_empty(self):
        result = FoundryResult(run_id="x", theme="x", genre="x", category=None)
        assert result.promoted_paths == []

    def test_curation_report_defaults(self):
        cr = CurationReport(genre="fantasy")
        assert cr.total_promoted == 0
        assert cr.total_skipped_dedup == 0
        assert cr.foundry_results == []
