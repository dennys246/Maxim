"""Tests for scripts/analyze_roy_5_cosine_localization.py.

The analyzer is the Stage 1 deliverable of
``docs/plans/roy_5_encoder_alignment_disambiguator.md``. Its verdict
gates a 1.1+ implementation branch, so the H1c/H1b/H1a decoding +
ingestion paths get pinned here against the same shape of inputs that
the real Roy-5a run produces.

Coverage required by the Roy-5a plan:

- Verdict decoding correctness across the boundary cosines
  (0.19, 0.20, 0.39, 0.40 exactly, 0.41).
- Cross-modal matrix non-empty when priming has interoception nodes
  and arm has text nodes — regression guard against silent
  same-modality-only filtering (which would zero out M_dt, the
  exact matrix Roy-4 confirmed is non-empty in practice).
- Food-cluster extraction from synthetic NAc-shaped dicts matches the
  3-field UTS-separator pattern that Roy-4's analyzer also uses.
- Graceful handling when an arm session has zero centroids of a given
  modality (the analyzer must return CosineMatrix(is_empty=True) and
  short-circuit max-over-rows to -inf without raising).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The analyzer lives outside src/ so we have to add scripts/ to sys.path
# before importing. This mirrors how analyze_roy_4_coactivation.py is
# tested elsewhere.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import analyze_roy_5_cosine_localization as a5  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Verdict decoding — boundary semantics
# ─────────────────────────────────────────────────────────────────────────


class TestVerdictDecoding:
    """The H1c/H1b/H1a boundary semantics are the PUBLIC contract of this analyzer.

    Stage 2 implementation branches consume the verdict directly:
    H1c → Stage 2a env-var sweep (~80 LOC).
    H1b → Stage 2b encoder A/B (~200 LOC).
    H1a → Stage 2c → Stage 3 cradle-arc redesign (~250-1150 LOC).

    A wrong verdict here mis-scopes ~weeks of work, so the boundaries
    are tested for exact-equality semantics (>= vs strict).
    """

    def test_below_h1b_lower_bound_decodes_h1a(self) -> None:
        v = a5.decode_verdict(0.19)
        assert v.sub_hypothesis == "H1a"
        assert "Stage 3" in v.next_stage

    def test_exactly_h1b_lower_bound_decodes_h1b(self) -> None:
        """0.20 is INCLUSIVE on the H1b side — boundary is `>= 0.20`."""
        v = a5.decode_verdict(0.20)
        assert v.sub_hypothesis == "H1b"
        assert "encoder A/B" in v.next_stage

    def test_within_h1b_band_decodes_h1b(self) -> None:
        v = a5.decode_verdict(0.39)
        assert v.sub_hypothesis == "H1b"

    def test_exactly_h1c_lower_bound_decodes_h1c(self) -> None:
        """0.40 is INCLUSIVE on the H1c side — boundary is `>= 0.40`.
        0.40 was EC's pattern_complete_threshold at the time Roy-5 shipped;
        the EC default moved to 0.44 in Phase 3 of
        docs/plans/ec_centroid_drift_fix.md but this Roy-5 boundary stays
        at 0.40 for verdict-stability of historical Roy-2c decodings.
        See the H1C_LOWER_BOUND module docstring for rationale."""
        v = a5.decode_verdict(0.40)
        assert v.sub_hypothesis == "H1c"
        assert "threshold" in v.next_stage.lower()

    def test_above_h1c_lower_bound_decodes_h1c(self) -> None:
        v = a5.decode_verdict(0.41)
        assert v.sub_hypothesis == "H1c"

    def test_negative_infinity_decodes_h1a(self) -> None:
        """No food-bearing rows in M_tt → -inf → still decodes to H1a
        (the parsimonious "encoder doesn't see them as related at all"
        explanation)."""
        v = a5.decode_verdict(float("-inf"))
        assert v.sub_hypothesis == "H1a"

    def test_constants_are_public_contract(self) -> None:
        """The two threshold constants are the public contract. If they
        drift, Stage 2 implementation branches mis-fire — pin them."""
        assert a5.H1C_LOWER_BOUND == 0.40
        assert a5.H1B_LOWER_BOUND == 0.20


# ─────────────────────────────────────────────────────────────────────────
# Cross-modal matrix non-empty (M_dt)
# ─────────────────────────────────────────────────────────────────────────


def _make_centroid(node_id: str, modality: str, *vec: float) -> a5.Centroid:
    return a5.Centroid(node_id=node_id, embedding=tuple(vec), modality=modality)


class TestCrossModalMatrix:
    """M_dt (priming interoception × arm text) must be non-empty when both
    sides have non-empty centroid sets of the requested modalities.

    Regression guard: an earlier scaffold draft used a same-modality-only
    filter that would have silently zeroed out M_dt — but Roy-4 confirmed
    both modalities fire in both phases, so M_dt should be non-empty on
    real data. The cross-modality pair (interoception, text) is the
    diagnostic of interest for H1a "structurally incompatible
    subspaces".
    """

    def test_m_dt_non_empty_when_both_sides_have_required_modalities(self) -> None:
        priming = [
            _make_centroid("p_intero_1", "interoception", 1.0, 0.0, 0.0),
            _make_centroid("p_intero_2", "interoception", 0.0, 1.0, 0.0),
            _make_centroid("p_text_1", "text", 0.5, 0.5, 0.0),
        ]
        arm = [
            _make_centroid("a_text_1", "text", 1.0, 0.0, 0.0),
            _make_centroid("a_text_2", "text", 0.0, 0.5, 0.5),
        ]
        m_dt = a5._compute_matrix(priming, arm, rows_modality="interoception", cols_modality="text")
        assert not m_dt.is_empty
        # 2 priming interoception rows × 2 arm text cols
        assert len(m_dt.rows) == 2
        assert len(m_dt.cols) == 2
        assert len(m_dt.values) == 2
        assert len(m_dt.values[0]) == 2

    def test_m_dt_records_correct_modality_labels(self) -> None:
        """The matrix's row/col modality labels must reflect the
        ACTUAL modality filter applied, not the side it came from. A
        future caller might inspect ``rows_modality`` to print labels
        in tooling — drift here would be silent."""
        priming = [_make_centroid("p_intero_1", "interoception", 1.0, 0.0)]
        arm = [_make_centroid("a_text_1", "text", 0.0, 1.0)]
        m = a5._compute_matrix(priming, arm, rows_modality="interoception", cols_modality="text")
        assert m.rows_modality == "interoception"
        assert m.cols_modality == "text"

    def test_m_dt_with_only_same_modality_in_arm_returns_empty(self) -> None:
        """Honest-empty case: if the arm has ZERO text-modality
        centroids, M_dt is genuinely empty and is_empty must be True
        (so the analyzer's verdict-decoding short-circuit handles it)."""
        priming = [_make_centroid("p_intero_1", "interoception", 1.0, 0.0)]
        arm = [_make_centroid("a_intero_1", "interoception", 1.0, 0.0)]
        m = a5._compute_matrix(priming, arm, rows_modality="interoception", cols_modality="text")
        assert m.is_empty


# ─────────────────────────────────────────────────────────────────────────
# Food-cluster extraction (UTS-separator pattern)
# ─────────────────────────────────────────────────────────────────────────


class TestFoodClusterExtraction:
    """The 3-field UTS-separator compound key is the canonical NAc shape
    Roy-4's analyzer also reads from. The Roy-5a analyzer must accept
    the same shape so the two analyzers cannot drift on cluster-ID
    semantics — they're consumed by Stage 2 implementations that key
    NAc lookups by these IDs.
    """

    def test_extracts_three_field_uts_compound(self, tmp_path: Path) -> None:
        """Canonical shape: `agent_id\\x1fcluster_id\\x1ftool_name`."""
        nac = {
            "cluster_reward_bias": {
                "sim_aut\x1fcluster-uuid-1\x1ftool:sense_food_source": 0.5,
                "sim_aut\x1fcluster-uuid-2\x1ftool:sense_food_source": 0.3,
                "sim_aut\x1fcluster-uuid-3\x1ftool:sense_water_source": 0.4,  # not food
            }
        }
        nac_path = tmp_path / "aut_nac.json"
        nac_path.write_text(json.dumps(nac))
        clusters = a5.load_priming_food_clusters(nac_path)
        assert clusters == {"cluster-uuid-1", "cluster-uuid-2"}

    def test_legacy_nested_shape_as_fallback(self, tmp_path: Path) -> None:
        """The pre-flat-compound shape used by older fixtures and tests."""
        nac = {
            "_cluster_reward_bias": {
                "agent-x": {
                    "cluster-A": {"sense_food_source": 0.5},
                    "cluster-B": {"sense_water_source": 0.3},
                }
            }
        }
        nac_path = tmp_path / "aut_nac.json"
        nac_path.write_text(json.dumps(nac))
        clusters = a5.load_priming_food_clusters(nac_path)
        assert clusters == {"cluster-A"}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        clusters = a5.load_priming_food_clusters(tmp_path / "no_such_file.json")
        assert clusters == set()

    def test_non_food_tool_keys_skipped(self, tmp_path: Path) -> None:
        nac = {
            "cluster_reward_bias": {
                "sim_aut\x1fcluster-1\x1ftool:sense_water_source": 0.5,
                "sim_aut\x1fcluster-2\x1ftool:examine": 0.3,
            }
        }
        nac_path = tmp_path / "aut_nac.json"
        nac_path.write_text(json.dumps(nac))
        clusters = a5.load_priming_food_clusters(nac_path)
        assert clusters == set()


# ─────────────────────────────────────────────────────────────────────────
# Graceful zero-centroid handling
# ─────────────────────────────────────────────────────────────────────────


class TestEmptyArmGracefulHandling:
    """When an arm session has zero centroids of a given modality, the
    analyzer must return CosineMatrix(is_empty=True) with empty rows/cols
    AND short-circuit max-over-rows to -inf — without raising. Arms B
    and C in Roy-5a are blank-substrate by design, so this case is the
    expected production input, not an error.
    """

    def test_zero_text_in_arm_yields_empty_m_tt(self) -> None:
        priming = [_make_centroid("p_text_1", "text", 1.0, 0.0)]
        arm = [_make_centroid("a_intero_1", "interoception", 1.0, 0.0)]
        m_tt = a5._compute_matrix(priming, arm, rows_modality="text", cols_modality="text")
        assert m_tt.is_empty
        assert m_tt.rows == ("p_text_1",)
        assert m_tt.cols == ()

    def test_zero_centroids_arm_yields_all_empty_matrices(self) -> None:
        priming = [
            _make_centroid("p_text_1", "text", 1.0, 0.0),
            _make_centroid("p_intero_1", "interoception", 0.0, 1.0),
        ]
        result = a5.analyze_arm(
            arm_label="b",
            priming_centroids=priming,
            arm_centroids=[],
            food_clusters={"p_text_1"},
        )
        assert result.M_tt.is_empty
        assert result.M_dt.is_empty
        assert result.M_dd.is_empty
        assert result.max_food_cosine_tt == float("-inf")
        assert result.max_food_cosine_dt == float("-inf")
        assert result.max_food_cosine_dd == float("-inf")

    def test_no_food_rows_in_matrix_yields_negative_infinity_max(self) -> None:
        """If the matrix is non-empty but NO row's node_id is in
        food_clusters, _max_over_food_clusters returns -inf — and
        decode_verdict translates that to H1a."""
        priming = [_make_centroid("p_text_1", "text", 1.0, 0.0)]
        arm = [_make_centroid("a_text_1", "text", 0.5, 0.5)]
        m_tt = a5._compute_matrix(priming, arm, rows_modality="text", cols_modality="text")
        assert not m_tt.is_empty
        # food_clusters has no overlap with priming rows
        max_food = a5._max_over_food_clusters(m_tt, {"some-other-cluster"})
        assert max_food == float("-inf")


# ─────────────────────────────────────────────────────────────────────────
# EC ingestion + end-to-end smoke (synthetic fixture)
# ─────────────────────────────────────────────────────────────────────────


class TestEcIngestion:
    """Round-trip the on-disk EC.save() schema through load_ec_centroids."""

    def test_load_ec_centroids_round_trips_save_shape(self, tmp_path: Path) -> None:
        """The fixture mirrors EntorhinalCortex.save()'s exact substrate_nodes
        shape — the load path must accept it without coercion."""
        ec_dump = {
            "_format_version": "1.0",
            "version": "1.0",
            "config": {},
            "lsh": {},
            "inverted": {},
            "signatures": {},
            "substrate_nodes": {
                "uuid-1": {
                    "embedding": [0.1, 0.2, 0.3],
                    "modality": "text",
                    "count": 5,
                },
                "uuid-2": {
                    "embedding": [0.4, 0.5, 0.6],
                    "modality": "interoception",
                    "count": 1,
                },
            },
        }
        ec_path = tmp_path / "aut_ec.json"
        ec_path.write_text(json.dumps(ec_dump))
        centroids = a5.load_ec_centroids(ec_path)
        assert len(centroids) == 2
        by_id = {c.node_id: c for c in centroids}
        assert by_id["uuid-1"].embedding == (0.1, 0.2, 0.3)
        assert by_id["uuid-1"].modality == "text"
        assert by_id["uuid-2"].modality == "interoception"

    def test_load_ec_centroids_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert a5.load_ec_centroids(tmp_path / "no_such_ec.json") == []

    def test_load_ec_centroids_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        ec_path = tmp_path / "aut_ec.json"
        ec_path.write_text("not json {")
        assert a5.load_ec_centroids(ec_path) == []

    def test_load_ec_centroids_skips_malformed_node_entries(self, tmp_path: Path) -> None:
        """One bad node shouldn't poison the whole load — the analyzer is
        a diagnostic, not a strict parser."""
        ec_dump = {
            "substrate_nodes": {
                "good": {"embedding": [1.0, 0.0], "modality": "text", "count": 1},
                "bad-1": "not a dict",
                "bad-2": {"embedding": "not a list", "modality": "text"},
                "bad-3": {"embedding": [1.0, 0.0], "modality": 42},  # modality not str
            }
        }
        ec_path = tmp_path / "aut_ec.json"
        ec_path.write_text(json.dumps(ec_dump))
        centroids = a5.load_ec_centroids(ec_path)
        assert len(centroids) == 1
        assert centroids[0].node_id == "good"


class TestEndToEndSmoke:
    """End-to-end: synthetic priming + arm-A dirs → main() exit code."""

    @pytest.fixture
    def synthetic_session_dirs(self, tmp_path: Path) -> dict[str, Path]:
        """Build three session dirs (priming + arm A + arm B) with
        substrate-shaped centroids. Arm A's text-modality centroids
        are deliberately positioned to land in the H1b band relative
        to the priming food clusters (~0.30 max cosine), so the smoke
        test pins the H1b verdict end-to-end."""
        priming_dir = tmp_path / "priming"
        arm_a_dir = tmp_path / "arm_a"
        arm_b_dir = tmp_path / "arm_b"
        for d in (priming_dir, arm_a_dir, arm_b_dir):
            d.mkdir()

        # Priming: 2 food-bearing text centroids (parallel unit vectors,
        # both pointing in the (1, 0) direction) + 1 interoception centroid.
        # Both food clusters point in the same direction so each one's
        # max-row cosine vs arm-A is identical and predictable.
        priming_ec = {
            "substrate_nodes": {
                "food-1": {"embedding": [1.0, 0.0], "modality": "text", "count": 5},
                "food-2": {"embedding": [1.0, 0.0], "modality": "text", "count": 3},
                "intero-1": {"embedding": [0.0, 1.0], "modality": "interoception", "count": 1},
            }
        }
        (priming_dir / "aut_ec.json").write_text(json.dumps(priming_ec))
        nac = {
            "cluster_reward_bias": {
                "sim_aut\x1ffood-1\x1ftool:sense_food_source": 0.5,
                "sim_aut\x1ffood-2\x1ftool:sense_food_source": 0.3,
            }
        }
        (priming_dir / "aut_nac.json").write_text(json.dumps(nac))

        # Arm A: arm-text-1 = (0.30, 0.954) — unit vector at acos(0.30) ≈ 72.5°
        # from the food direction (1, 0). cos((1,0), (0.30, 0.954)) =
        # 0.30 / sqrt(0.09 + 0.91) = 0.30, which lands cleanly in H1b
        # [0.20, 0.40). arm-text-2 points orthogonally for a lower cosine.
        arm_a_ec = {
            "substrate_nodes": {
                "arm-text-1": {"embedding": [0.30, 0.954], "modality": "text", "count": 1},
                "arm-text-2": {"embedding": [0.0, 1.0], "modality": "text", "count": 1},
                "arm-intero-1": {"embedding": [0.0, 1.0], "modality": "interoception", "count": 1},
            }
        }
        (arm_a_dir / "aut_ec.json").write_text(json.dumps(arm_a_ec))

        # Arm B: blank substrate (empty EC)
        arm_b_ec = {"substrate_nodes": {}}
        (arm_b_dir / "aut_ec.json").write_text(json.dumps(arm_b_ec))

        return {
            "priming": priming_dir,
            "arm_a": arm_a_dir,
            "arm_b": arm_b_dir,
            "tmp": tmp_path,
        }

    def test_main_returns_zero_on_h1b_smoke(
        self, synthetic_session_dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = synthetic_session_dirs["tmp"] / "analysis.json"
        rc = a5.main(
            [
                "--priming-dir",
                str(synthetic_session_dirs["priming"]),
                "--arm-a-dir",
                str(synthetic_session_dirs["arm_a"]),
                "--arm-b-dir",
                str(synthetic_session_dirs["arm_b"]),
                "--output-json",
                str(output),
            ]
        )
        assert rc == 0
        bundle = json.loads(output.read_text())
        assert bundle["verdict"]["sub_hypothesis"] == "H1b"
        # arm B should produce empty matrices (blank substrate)
        assert bundle["arms"]["b"]["M_tt"]["is_empty"] is True
        # Arm A produces all three non-empty matrices
        assert bundle["arms"]["a"]["M_tt"]["is_empty"] is False
        captured = capsys.readouterr()
        assert "ROY-5a VERDICT: H1b" in captured.out

    def test_main_returns_two_on_missing_priming_ec(self, tmp_path: Path) -> None:
        priming_dir = tmp_path / "priming"
        arm_a_dir = tmp_path / "arm_a"
        priming_dir.mkdir()
        arm_a_dir.mkdir()
        # No aut_ec.json — priming centroids will be empty
        (priming_dir / "aut_nac.json").write_text(json.dumps({"cluster_reward_bias": {}}))
        rc = a5.main(
            [
                "--priming-dir",
                str(priming_dir),
                "--arm-a-dir",
                str(arm_a_dir),
            ]
        )
        assert rc == 2

    def test_main_returns_two_on_missing_food_clusters(self, tmp_path: Path) -> None:
        priming_dir = tmp_path / "priming"
        arm_a_dir = tmp_path / "arm_a"
        priming_dir.mkdir()
        arm_a_dir.mkdir()
        # Has centroids but no food bias
        (priming_dir / "aut_ec.json").write_text(
            json.dumps({"substrate_nodes": {"node-1": {"embedding": [1.0, 0.0], "modality": "text", "count": 1}}})
        )
        (priming_dir / "aut_nac.json").write_text(json.dumps({"cluster_reward_bias": {}}))
        rc = a5.main(
            [
                "--priming-dir",
                str(priming_dir),
                "--arm-a-dir",
                str(arm_a_dir),
            ]
        )
        assert rc == 2


# ─────────────────────────────────────────────────────────────────────────
# Review-fold tests — added after architecture + bio-fidelity reviews
# ─────────────────────────────────────────────────────────────────────────


class TestNaNAndBoolRejection:
    """Strict numeric guards on the embedding loader. NaN / inf / bool
    silently flowing through ``_cosine`` mis-decode the H1c/H1b/H1a
    boundary — masquerading as H1a or H1c depending on direction.
    Architecture review #3.
    """

    def test_nan_embedding_skipped(self, tmp_path: Path) -> None:
        ec_dump = {
            "substrate_nodes": {
                "nan-node": {"embedding": [1.0, float("nan"), 0.0], "modality": "text", "count": 1},
                "ok-node": {"embedding": [1.0, 0.0, 0.0], "modality": "text", "count": 1},
            }
        }
        ec_path = tmp_path / "aut_ec.json"
        ec_path.write_text(json.dumps(ec_dump))
        centroids = a5.load_ec_centroids(ec_path)
        # Only the ok-node survives; the NaN-bearing one is skipped.
        assert len(centroids) == 1
        assert centroids[0].node_id == "ok-node"

    def test_inf_embedding_skipped(self, tmp_path: Path) -> None:
        ec_dump = {
            "substrate_nodes": {
                "inf-node": {"embedding": [float("inf"), 0.0], "modality": "text", "count": 1},
            }
        }
        ec_path = tmp_path / "aut_ec.json"
        ec_path.write_text("{}")
        # NaN/inf serialize to JSON as the JS literal `NaN` / `Infinity`,
        # which standard json.loads rejects. Write the embedding via
        # Python and re-serialize through allow_nan=True for the test.
        with ec_path.open("w") as f:
            json.dump(ec_dump, f, allow_nan=True)
        centroids = a5.load_ec_centroids(ec_path)
        assert centroids == []

    def test_bool_embedding_skipped(self, tmp_path: Path) -> None:
        """Python bool is an int subclass; ``isinstance(True, int)`` is
        True. The strict numeric guard MUST reject bool to prevent a
        bool-valued embedding from being silently coerced to [1.0, 0.0]."""
        ec_dump = {
            "substrate_nodes": {
                "bool-node": {"embedding": [True, False], "modality": "text", "count": 1},
            }
        }
        ec_path = tmp_path / "aut_ec.json"
        ec_path.write_text(json.dumps(ec_dump))
        centroids = a5.load_ec_centroids(ec_path)
        assert centroids == []


class TestDimensionMismatch:
    """Architecture review #6 — embedding dimension mismatch between
    priming and arm silently produces 0.0 cosines, masquerading as H1a.
    The Stage 2b encoder A/B branch is exactly the situation that would
    introduce this (different sentence-transformer model with different
    embedding dimension), so it needs a loud warning.
    """

    def test_dimension_mismatch_warns_on_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        priming = [_make_centroid("p_text_1", "text", 1.0, 0.0, 0.0)]  # 3D
        arm = [_make_centroid("a_text_1", "text", 1.0, 0.0)]  # 2D — mismatch
        a5._compute_matrix(priming, arm, rows_modality="text", cols_modality="text")
        captured = capsys.readouterr()
        assert "dimension mismatch" in captured.err
        assert "0.0" in captured.err  # references the silent-failure mechanism

    def test_same_dimensions_no_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        priming = [_make_centroid("p_text_1", "text", 1.0, 0.0)]
        arm = [_make_centroid("a_text_1", "text", 0.5, 0.5)]
        a5._compute_matrix(priming, arm, rows_modality="text", cols_modality="text")
        captured = capsys.readouterr()
        assert "dimension mismatch" not in captured.err


class TestJsonBundleFlags:
    """Architecture review #7 — the JSON bundle's ``_is_na`` flags let
    downstream Stage 2 consumers branch cleanly on the n/a case without
    derserializing -inf or guessing from None vs missing key."""

    def test_bundle_carries_is_na_flag_at_verdict_level(self, tmp_path: Path) -> None:
        """When max_food_cosine_tt is -inf (the H1a-via-empty path),
        the bundle must carry max_food_cosine_tt_is_na: True so
        downstream consumers can branch without TypeError on null < 0.20."""
        priming_dir = tmp_path / "priming"
        arm_a_dir = tmp_path / "arm_a"
        priming_dir.mkdir()
        arm_a_dir.mkdir()
        # Priming has interoception food cluster only (the Roy-5a actual case)
        (priming_dir / "aut_ec.json").write_text(
            json.dumps(
                {
                    "substrate_nodes": {
                        "food-intero-1": {"embedding": [1.0, 0.0], "modality": "interoception", "count": 1}
                    }
                }
            )
        )
        (priming_dir / "aut_nac.json").write_text(
            json.dumps({"cluster_reward_bias": {"sim_aut\x1ffood-intero-1\x1ftool:sense_food_source": 0.5}})
        )
        (arm_a_dir / "aut_ec.json").write_text(
            json.dumps(
                {
                    "substrate_nodes": {
                        "arm-intero-1": {"embedding": [1.0, 0.0], "modality": "interoception", "count": 1}
                    }
                }
            )
        )
        output = tmp_path / "analysis.json"
        rc = a5.main(
            [
                "--priming-dir",
                str(priming_dir),
                "--arm-a-dir",
                str(arm_a_dir),
                "--output-json",
                str(output),
            ]
        )
        assert rc == 0
        bundle = json.loads(output.read_text())
        # Verdict-level flag
        assert bundle["verdict"]["sub_hypothesis"] == "H1a"
        assert bundle["verdict"]["max_food_cosine_tt"] is None
        assert bundle["verdict"]["max_food_cosine_tt_is_na"] is True
        # Arm-level flags (mirror the verdict shape per consumer expectation)
        assert bundle["arms"]["a"]["max_food_cosine_tt_is_na"] is True
        assert bundle["arms"]["a"]["max_food_cosine_dd_is_na"] is False  # M_dd is non-empty


class TestExitCodeOneOnNoArmASignal:
    """Architecture review #4 — empty-matrix-in-every-modality is rc=1
    (no usable signal), not rc=0 with an H1a verdict. The pre-fold
    order was wrong: decode_verdict ran first and returned 0 before
    the empty-matrix check could fire."""

    def test_arm_a_zero_centroids_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        priming_dir = tmp_path / "priming"
        arm_a_dir = tmp_path / "arm_a"
        priming_dir.mkdir()
        arm_a_dir.mkdir()
        (priming_dir / "aut_ec.json").write_text(
            json.dumps(
                {"substrate_nodes": {"food-1": {"embedding": [1.0, 0.0], "modality": "interoception", "count": 1}}}
            )
        )
        (priming_dir / "aut_nac.json").write_text(
            json.dumps({"cluster_reward_bias": {"sim_aut\x1ffood-1\x1ftool:sense_food_source": 0.5}})
        )
        # arm A has the file but with zero centroids
        (arm_a_dir / "aut_ec.json").write_text(json.dumps({"substrate_nodes": {}}))
        rc = a5.main(["--priming-dir", str(priming_dir), "--arm-a-dir", str(arm_a_dir)])
        assert rc == 1
        captured = capsys.readouterr()
        assert "no centroids in any modality" in captured.err


class TestVisionModalityWarning:
    """Architecture review #5 — modalities outside {text, interoception}
    are silently dropped. Surface a warning so future iterations that
    fire vision-modality nodes don't have their signal invisible."""

    def test_vision_modality_in_priming_warns(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        priming_dir = tmp_path / "priming"
        arm_a_dir = tmp_path / "arm_a"
        priming_dir.mkdir()
        arm_a_dir.mkdir()
        (priming_dir / "aut_ec.json").write_text(
            json.dumps(
                {
                    "substrate_nodes": {
                        "food-1": {"embedding": [1.0, 0.0], "modality": "interoception", "count": 1},
                        "vision-1": {"embedding": [0.5, 0.5], "modality": "vision", "count": 1},
                    }
                }
            )
        )
        (priming_dir / "aut_nac.json").write_text(
            json.dumps({"cluster_reward_bias": {"sim_aut\x1ffood-1\x1ftool:sense_food_source": 0.5}})
        )
        (arm_a_dir / "aut_ec.json").write_text(
            json.dumps(
                {
                    "substrate_nodes": {
                        "arm-intero-1": {"embedding": [1.0, 0.0], "modality": "interoception", "count": 1}
                    }
                }
            )
        )
        a5.main(["--priming-dir", str(priming_dir), "--arm-a-dir", str(arm_a_dir)])
        captured = capsys.readouterr()
        assert "modalities outside" in captured.err
        assert "vision" in captured.err


class TestRoy4Roy5FoodClusterParity:
    """Architecture review #1 — drift between Roy-4 and Roy-5 food-cluster
    extraction would mis-route NAc lookups in Stage 2 implementations.
    PR #248 fold replaced Roy-5's local implementation with a delegating
    import to Roy-4's loader, so the two analyzers now share one source
    of truth. This test pins the delegation."""

    def test_roy_5_delegates_to_roy_4_loader(self, tmp_path: Path) -> None:
        nac = {
            "cluster_reward_bias": {
                "sim_aut\x1fcluster-X\x1ftool:sense_food_source": 0.5,
                "sim_aut\x1fcluster-Y\x1ftool:sense_water_source": 0.3,
            }
        }
        nac_path = tmp_path / "aut_nac.json"
        nac_path.write_text(json.dumps(nac))
        # Load via Roy-5's analyzer
        roy_5_result = a5.load_priming_food_clusters(nac_path)
        # Load via Roy-4's analyzer directly
        import importlib.util

        roy_4_path = Path(__file__).resolve().parents[2] / "scripts" / "analyze_roy_4_coactivation.py"
        spec = importlib.util.spec_from_file_location("analyze_roy_4_coactivation", roy_4_path)
        assert spec is not None and spec.loader is not None
        roy_4_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(roy_4_mod)
        roy_4_result = roy_4_mod.load_priming_food_clusters(nac_path)
        assert roy_5_result == roy_4_result == {"cluster-X"}
