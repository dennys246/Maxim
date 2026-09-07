"""Guard tests for the Exp 57 dose-response ladder harness (scripts/exp57 +
scripts/analyze_exp57).

The prereg's sign-off requires the harness PR to carry these
(docs/experiments/protocols/exp57_dose_response_ladder_preregistration.md):
the frozen GATE-constant pins; the LEFT-ASSOCIATIVE fold weighting (1/4, 1/4,
1/2 for three contributors, NOT equal-weight 1/N); the DETERMINISTIC bias-
decisive coverage DV; the sliding-window tau with right-censoring; the
Jonckheere-Terpstra permutation trend test + its two censoring-artifact guards;
the anti-vacuity kit; the ScriptedBridge one-client faithfulness; and the
seed-parameterization that drives R0's multi-seed requirement. Anything needing
a world uses the scripted mock — NO live server, NO real LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from exp56 import common as C  # noqa: E402
from exp57 import common57 as X  # noqa: E402


@pytest.fixture(autouse=True)
def _operant_only(monkeypatch):
    monkeypatch.setenv("MAXIM_OPERANT_ONLY_CREDIT", "1")


# ── frozen GATE constants + ladder constants ─────────────────────────────


class TestFrozenGates:
    def test_analyzer_gate_constants_pin(self):
        import analyze_exp57 as A

        # These are FROZEN at the harness-merge commit; a drift is a retune and
        # must be a pre-registered amendment (prereg §Amendment rule).
        assert A.GATES_V1["delta_eff"] == 0
        assert A.GATES_V1["p_threshold"] == 0.05
        assert tuple(A.GATES_V1["rungs"]) == (1, 2, 4, 8)
        assert A.GATES_V1["cohorts_min"] == 20
        assert A.GATES_V1["endpoint_spearman_min"] == 0.0

    def test_ladder_constants(self):
        assert X.G == 4
        assert len(X.CONTINGENCY_SLOTS) == 4
        assert X.CRITERION_TARGET == 3.0 / 4.0
        assert X.WINDOW_TARGET == 3

    def test_cohort_slot_to_target_is_a_permutation(self):
        m = X.cohort_slot_to_target(42)
        assert len(m) == X.G
        assert len(set(m.values())) == X.G  # distinct affordances per slot
        assert all(a in C.AFFORDANCES for a in m.values())
        assert X.cohort_slot_to_target(42) == X.cohort_slot_to_target(42)


# ── the LEFT-ASSOCIATIVE fold weighting (1/4, 1/4, 1/2) ──────────────────


def _synthetic_snapshot(*, cid: str, value: float, tool="minecraft_bench57_aff_c", agent="donor", geom="geomA", dim=8):
    """A minimal valid (NAc, EC) snapshot: one world node + one cluster bias.

    The embedding is identical across snapshots so three contributors' nodes
    ALIGN (cosine 1.0, same geometry) through the real ingest and their bias
    keys collapse to one shared key — the arrangement that makes the pairwise
    convex combination observable.
    """
    emb = [1.0] + [0.0] * (dim - 1)
    tsig = f"tool:{tool}"
    key = f"{agent}\x1f{cid}\x1f{tsig}"
    nac = {
        "version": "1.0",
        "links": {},
        "outcome_index": {},
        "priors": {},
        "total_observations": 0,
        "reward_bias": {},
        "goal_reward_bias": {},
        "cluster_reward_bias": {key: value},
        "cluster_reward_source": {key: "operant"},
        "inherent_bias_keys": [],
        "percept_valences": {},
        "event_outcome_welford": {},
    }
    ec = {
        cid: {
            "embedding": emb,
            "modality": "world",
            "count": 1,
            "source": "local",
            "domain": None,
            "geometry": geom,
        }
    }
    return nac, ec


class TestFoldWeighting:
    def test_three_contributors_are_convex_1_4_1_4_1_2(self, tmp_path):
        # Same cluster id + embedding + geometry -> the three donor nodes align,
        # so their re-keyed bias keys collapse to one shared key and the pairwise
        # _merge_mean_clamped fold gives weights (1/4, 1/4, 1/2):
        #   r0 = a; r1 = mean(a, b); r2 = mean(mean(a, b), c) = a/4 + b/4 + c/2.
        a, b, c = 0.8, 0.4, 0.6
        snaps = [
            _synthetic_snapshot(cid="node0", value=a),
            _synthetic_snapshot(cid="node0", value=b),
            _synthetic_snapshot(cid="node0", value=c),
        ]
        merged = X.fold_snapshots(snaps, "recvX", workdir=tmp_path / "fold", contributor_ids=["cA", "cB", "cD"])
        biases = merged["cluster_reward_bias"]
        # One shared key, carrying the receiver agent id after re-key.
        keys = [k for k in biases if k.split("\x1f")[1] == "node0"]
        assert len(keys) == 1, biases
        assert keys[0].split("\x1f")[0] == "recvX"
        expected = a / 4 + b / 4 + c / 2  # 0.6
        assert biases[keys[0]] == pytest.approx(expected, abs=1e-9)

    def test_not_equal_weight_1_over_n(self, tmp_path):
        # The distinguishing property vs nac_merge_many's equal-weight 1/N fold:
        # equal weight would give (a+b+c)/3 = 0.6 here too, so pick values whose
        # convex and equal means DIFFER.
        a, b, c = 0.9, 0.0, 0.0
        snaps = [_synthetic_snapshot(cid="node0", value=v) for v in (a, b, c)]
        merged = X.fold_snapshots(snaps, "recvX", workdir=tmp_path / "fold", contributor_ids=["cA", "cB", "cD"])
        key = next(k for k in merged["cluster_reward_bias"] if k.split("\x1f")[1] == "node0")
        convex = a / 4 + b / 4 + c / 2  # 0.225
        equal = (a + b + c) / 3  # 0.30
        assert merged["cluster_reward_bias"][key] == pytest.approx(convex, abs=1e-9)
        assert convex != pytest.approx(equal, abs=1e-6)


# ── the coverage DV (bias-decisive argmax from provenance) ───────────────


class TestCoverageDV:
    def test_contingency_covered_requires_decisive_argmax(self):
        taught = "minecraft_bench57_aff_c"
        # covered: argmax is the taught tool, learned_bias > 0, margin > 0.
        assert X.contingency_covered(
            {"best_tool": taught, "score_components": {"learned_bias": 0.5}, "learned_margin": 0.3}, taught
        )
        # argmax is a DIFFERENT tool -> not covered.
        assert not X.contingency_covered(
            {"best_tool": "minecraft_bench57_aff_a", "score_components": {"learned_bias": 0.5}, "learned_margin": 0.3},
            taught,
        )
        # learned_bias == 0 -> not covered (a causal/reward-bias win is not the claim).
        assert not X.contingency_covered(
            {"best_tool": taught, "score_components": {"learned_bias": 0.0}, "learned_margin": 0.3}, taught
        )
        # margin not positive -> not covered.
        assert not X.contingency_covered(
            {"best_tool": taught, "score_components": {"learned_bias": 0.5}, "learned_margin": 0.0}, taught
        )
        assert not X.contingency_covered(
            {"best_tool": taught, "score_components": {"learned_bias": 0.5}, "learned_margin": None}, taught
        )

    def test_coverage_counts_only_the_taught_contingency(self):
        # merged_nac with a decisive world-cluster bias for aff_c at cluster
        # "cidC" — read through recommend_action provenance on a bare NAc. A
        # uniform baseline cluster bias on every tool (the merged causal/link
        # signal the balanced schedule produces for every executed affordance)
        # puts all tools in the score table so learned_margin is defined; the
        # taught tool's HIGHER bias makes it the learned-bias-decisive argmax.
        agent = "recvX"
        crb = {
            f"{agent}\x1fcidC\x1ftool:minecraft_bench57_{aff}": (0.8 if aff == "aff_c" else 0.1)
            for aff in C.AFFORDANCES
        }
        merged = {
            "version": "1.0",
            "links": {},
            "outcome_index": {},
            "priors": {},
            "total_observations": 0,
            "reward_bias": {},
            "goal_reward_bias": {},
            "cluster_reward_bias": crb,
            "cluster_reward_source": {},
            "inherent_bias_keys": [],
            "percept_valences": {},
            "event_outcome_welford": {},
        }
        slot_to_target = {0: "aff_c", 1: "aff_a", 2: "aff_b", 3: "aff_d"}
        clusters = {0: {"world": "cidC"}, 1: {}, 2: {}, 3: {}}
        cov, detail = X.coverage(merged, clusters, slot_to_target, receiver_agent_id=agent, return_detail=True)
        assert detail[0] is True  # taught aff_c is the learned-bias-decisive argmax
        assert not any(detail[g] for g in (1, 2, 3))  # other contingencies have no active cluster
        assert cov == pytest.approx(0.25)

    def test_coverage_raises_on_drive_prior_leak(self, monkeypatch):
        # L12 zero-prior net (code-lens finding 2): a nonzero drive component at
        # d1=0 would corrupt the best_tool argmax the DV reads, so coverage MUST
        # raise rather than silently miscount — a can't-fail check is not a check.
        import maxim.decisions.nac as nacmod

        class _LeakCap:
            def __init__(self):
                self.events = [
                    {
                        "data": {
                            "best_tool": "minecraft_bench57_aff_c",
                            "score_components": {"learned_bias": 0.5, "drive": 0.9},
                            "learned_margin": 0.3,
                        }
                    }
                ]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(X.C, "RecommendCapture", _LeakCap)
        monkeypatch.setattr(nacmod.NAc, "recommend_action", lambda self, **k: None)
        empty = {
            "version": "1.0",
            "links": {},
            "outcome_index": {},
            "priors": {},
            "total_observations": 0,
            "reward_bias": {},
            "goal_reward_bias": {},
            "cluster_reward_bias": {},
            "cluster_reward_source": {},
            "inherent_bias_keys": [],
            "percept_valences": {},
            "event_outcome_welford": {},
        }
        with pytest.raises(AssertionError, match="drive-prior leak"):
            X.coverage(
                empty, {0: {"world": "c"}}, {0: "aff_c", 1: "aff_a", 2: "aff_b", 3: "aff_d"}, receiver_agent_id="a"
            )


# ── tau (sliding window + right-censoring) ───────────────────────────────


class TestTau:
    def test_sustained_window(self):
        # C = 0.75, W = 3: the first t whose trailing 3 checkpoints are all >= C.
        series = [0.0, 0.0, 0.75, 0.75, 0.75, 1.0]
        assert X.tau(series, 0.75, 3, 6) == 5

    def test_single_noisy_checkpoint_does_not_trip(self):
        series = [0.75, 0.0, 0.75, 0.75, 0.75, 0.75]
        # windows: t3=[.75,0,.75] no, t4=[0,.75,.75] no, t5=[.75,.75,.75] yes.
        assert X.tau(series, 0.75, 3, 6) == 5

    def test_right_censored_sentinel(self):
        series = [0.25, 0.5, 0.5, 0.25, 0.5, 0.5]
        assert X.tau(series, 0.75, 3, 6) == 7  # K_max + 1
        assert X.is_censored(7, 6)
        assert not X.is_censored(5, 6)

    def test_immediate_when_all_high(self):
        assert X.tau([1.0] * 6, 0.75, 3, 6) == 3


# ── Jonckheere-Terpstra permutation trend + censoring guards ─────────────


class TestJonckheere:
    def test_significant_on_strictly_decreasing(self):
        import analyze_exp57 as A

        groups = [[20.0, 19.0, 18.0], [14.0, 13.0, 12.0], [9.0, 8.0, 7.0], [4.0, 3.0, 2.0]]
        res = A.jt_permutation_test(groups, permutations=2000, seed=1)
        assert res["p_value"] < 0.05, res

    def test_not_significant_on_flat(self):
        import analyze_exp57 as A

        groups = [[7.0, 7.0, 7.0]] * 4
        res = A.jt_permutation_test(groups, permutations=2000, seed=1)
        assert res["p_value"] > 0.05, res


def _base_row(cohort, rung, condition, t, coverage, tau, k_max, *, criterion=0.75, window=3):
    return {
        "cohort": cohort,
        "rung": rung,
        "condition": condition,
        "t": t,
        "coverage": coverage,
        "tau": tau,
        "tau_censored": tau >= k_max + 1,
        "criterion": criterion,
        "window": window,
        "k_max": k_max,
        "per_contingency": {},
        "fold_report": {},
        "mock": False,
    }


def _n1_spread(base: float, cohort: int, cohorts: int) -> float:
    """Per-cohort N=1 endpoint coverage centred on ``base`` but VARYING across
    cohorts, so the L2 seed-variance gate (which forbids a seed-invariant N=1
    arm) is satisfied. Symmetric quarter-steps keep the median at ``base``;
    snapped to the {0,¼,½,¾,1} coverage lattice and clamped."""
    step = 0.25 * (cohort - (cohorts - 1) / 2)
    v = round((base + step) * 4) / 4
    return min(1.0, max(0.0, v))


def _make_rows(
    tau_creche,
    tau_single,
    endpoint_creche,
    none_cov,
    *,
    cohorts,
    k_max=6,
    criterion=0.75,
    window=3,
    vary_n1=True,
):
    rows = []
    for rung in (1, 2, 4, 8):
        for cohort in range(cohorts):
            for t in range(1, k_max + 1):
                cov = endpoint_creche[rung] if t == k_max else 0.0
                # Vary the N=1 endpoint across cohorts so L2 passes (a genuine
                # isolated arm is seed-VARIANT); other rungs stay constant.
                if rung == 1 and vary_n1 and t == k_max:
                    cov = _n1_spread(endpoint_creche[1], cohort, cohorts)
                rows.append(
                    _base_row(
                        cohort, rung, "creche", t, cov, tau_creche[rung], k_max, criterion=criterion, window=window
                    )
                )
            for t in range(1, k_max + 1):
                rows.append(
                    _base_row(
                        cohort,
                        rung,
                        "single_matched",
                        t,
                        0.0,
                        tau_single[rung],
                        k_max,
                        criterion=criterion,
                        window=window,
                    )
                )
            for t in range(1, k_max + 1):
                rows.append(
                    _base_row(
                        cohort, rung, "creche_none", t, none_cov, k_max + 1, k_max, criterion=criterion, window=window
                    )
                )
    return rows


class TestAnalyzerGates:
    def test_pass_verdict(self):
        import analyze_exp57 as A

        rows = _make_rows(
            tau_creche={1: 5, 2: 4, 4: 3, 8: 2},
            tau_single={1: 5, 2: 9, 4: 13, 8: 17},
            endpoint_creche={1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0},
            none_cov=0.0,
            cohorts=3,
        )
        report = A.analyze(rows, cohorts_min=3, permutations=1500)
        assert report["verdict"] == "PASS", report
        assert report["gates"]["MONOTONICITY"] is True
        assert report["gates"]["NOT_JUST_MORE_DATA"] is True
        assert report["gates"]["NOISE_FLOOR"] is True

    def test_guard_ii_endpoint_must_rise(self):
        import analyze_exp57 as A

        # JT on tau is decreasing (significant), but endpoint coverage is FLAT
        # -> guard (ii) fails -> MONOTONICITY False -> not a PASS.
        rows = _make_rows(
            tau_creche={1: 5, 2: 4, 4: 3, 8: 2},
            tau_single={1: 5, 2: 9, 4: 13, 8: 17},
            endpoint_creche={1: 0.5, 2: 0.5, 4: 0.5, 8: 0.5},
            none_cov=0.0,
            cohorts=3,
        )
        report = A.analyze(rows, cohorts_min=3, permutations=1500)
        assert report["gates"]["MONOTONICITY"] is False
        assert report["gate_details"]["MONOTONICITY"]["guard_ii_endpoint_rises"] is False

    def test_guard_i_drops_censored_rungs(self):
        import analyze_exp57 as A

        # Bottom rung fully CENSORED (tau = K_max+1); rungs 2/4/8 FLAT. Overall
        # JT is significant only because the censored rung reinforces the
        # decreasing alternative (the anti-conservative artifact) — dropping it
        # leaves a flat trend, so guard (i) fails and MONOTONICITY is not a PASS.
        k_max = 6
        rows = _make_rows(
            tau_creche={1: k_max + 1, 2: 3, 4: 3, 8: 3},
            tau_single={1: 5, 2: 9, 4: 13, 8: 17},
            endpoint_creche={1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0},
            none_cov=0.0,
            cohorts=3,
            k_max=k_max,
        )
        report = A.analyze(rows, cohorts_min=3, permutations=2000)
        d = report["gate_details"]["MONOTONICITY"]
        assert d["pass_jt"] is True, d  # overall JT fires (artifact)
        assert d["guard_i_drop_censored"] is False, d  # but the guard catches it
        assert report["gates"]["MONOTONICITY"] is False

    def test_not_just_more_data_fails_is_partial(self):
        import analyze_exp57 as A

        # MONOTONICITY passes but pooling costs MORE total experience than one
        # agent seeing everything -> PARTIAL (owner call, not a silent pass).
        rows = _make_rows(
            tau_creche={1: 5, 2: 4, 4: 3, 8: 2},
            tau_single={1: 5, 2: 3, 4: 3, 8: 3},  # single_matched cheaper than N*creche
            endpoint_creche={1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0},
            none_cov=0.0,
            cohorts=3,
        )
        report = A.analyze(rows, cohorts_min=3, permutations=1500)
        assert report["gates"]["NOT_JUST_MORE_DATA"] is False
        assert report["verdict"] == "PARTIAL", report

    def test_noise_floor_fails_when_untaught_gains_coverage(self):
        import analyze_exp57 as A

        rows = _make_rows(
            tau_creche={1: 5, 2: 4, 4: 3, 8: 2},
            tau_single={1: 5, 2: 9, 4: 13, 8: 17},
            endpoint_creche={1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0},
            none_cov=0.75,  # untaught creche reaches criterion — a leak
            cohorts=3,
        )
        report = A.analyze(rows, cohorts_min=3, permutations=1500)
        assert report["gates"]["NOISE_FLOOR"] is False
        assert report["verdict"] == "FAIL"

    def test_mock_rows_refuse_verdict(self):
        import analyze_exp57 as A

        rows = _make_rows(
            tau_creche={1: 5, 2: 4, 4: 3, 8: 2},
            tau_single={1: 5, 2: 9, 4: 13, 8: 17},
            endpoint_creche={1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0},
            none_cov=0.0,
            cohorts=3,
        )
        rows[0]["mock"] = True
        report = A.analyze(rows, cohorts_min=3, permutations=500)
        assert report["verdict"] == "NO-VERDICT"

    def test_dirty_rows_refuse_verdict(self):
        import analyze_exp57 as A

        rows = _make_rows(
            tau_creche={1: 5, 2: 4, 4: 3, 8: 2},
            tau_single={1: 5, 2: 9, 4: 13, 8: 17},
            endpoint_creche={1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0},
            none_cov=0.0,
            cohorts=3,
        )
        rows[3]["working_tree_dirty_src_scripts"] = True
        report = A.analyze(rows, cohorts_min=3, permutations=500)
        assert report["verdict"] == "NO-VERDICT"

    def test_underpowered_refuses_verdict(self):
        import analyze_exp57 as A

        rows = _make_rows(
            tau_creche={1: 5, 2: 4, 4: 3, 8: 2},
            tau_single={1: 5, 2: 9, 4: 13, 8: 17},
            endpoint_creche={1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0},
            none_cov=0.0,
            cohorts=3,
        )
        report = A.analyze(rows, cohorts_min=20, permutations=500)  # frozen power
        assert report["verdict"] == "NO-VERDICT"
        assert any("frozen power" in p for p in report["problems"])

    def test_l2_seed_invariant_n1_refuses_verdict(self):
        import analyze_exp57 as A

        # N=1 endpoint coverage IDENTICAL across cohorts -> seed-invariant ->
        # effective-n collapsed -> L2 fails -> NO-VERDICT (methodology finding 2).
        rows = _make_rows(
            tau_creche={1: 5, 2: 4, 4: 3, 8: 2},
            tau_single={1: 5, 2: 9, 4: 13, 8: 17},
            endpoint_creche={1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0},
            none_cov=0.0,
            cohorts=3,
            vary_n1=False,  # force seed-invariance
        )
        report = A.analyze(rows, cohorts_min=3, permutations=500)
        assert report["gates"]["L2_SEED_VARIANCE"] is False
        assert report["verdict"] == "NO-VERDICT"
        assert any("seed-invariant" in p for p in report["problems"])

    def test_noise_floor_arm_absent_refuses_verdict(self):
        import analyze_exp57 as A

        # A PASS-shaped dataset with the creche_none arm DROPPED must NOT pass
        # NOISE-FLOOR vacuously at 0.0 < C — it must refuse (finding 3).
        rows = [
            r
            for r in _make_rows(
                tau_creche={1: 5, 2: 4, 4: 3, 8: 2},
                tau_single={1: 5, 2: 9, 4: 13, 8: 17},
                endpoint_creche={1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0},
                none_cov=0.0,
                cohorts=3,
            )
            if r["condition"] != "creche_none"
        ]
        report = A.analyze(rows, cohorts_min=3, permutations=500)
        assert report["gates"]["NOISE_FLOOR"] is None
        assert report["verdict"] == "NO-VERDICT"
        assert any("creche_none arm absent" in p for p in report["problems"])

    def test_anti_vacuity_required_for_pass(self, tmp_path):
        # The verdict authority must REFUSE a scored PASS without the no-op kit
        # (prereg: "no verdict without it"; finding 4). Run the CLI on a complete
        # PASS dataset WITHOUT --assert-noop-fails -> NO-VERDICT / exit 4.
        import subprocess

        rows = _make_rows(
            tau_creche={1: 5, 2: 4, 4: 3, 8: 2},
            tau_single={1: 5, 2: 9, 4: 13, 8: 17},
            endpoint_creche={1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0},
            none_cov=0.0,
            cohorts=3,
        )
        import json as _json

        f = tmp_path / "ladder.jsonl"
        f.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
        analyzer = REPO / "scripts" / "analyze_exp57.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--in",
                str(f),
                "--gate",
                "v1",
                "--cohorts-min",
                "3",
                "--permutations",
                "500",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 4, proc.stdout + proc.stderr
        assert "no verdict without it" in proc.stdout


# ── the anti-vacuity kit collapses ───────────────────────────────────────


class TestAntiVacuity:
    def test_noop_kit_collapses_must_collapse_variants(self, tmp_path):
        snaps = [
            _synthetic_snapshot(cid="node0", value=0.8),
            _synthetic_snapshot(cid="node1", value=0.4, tool="minecraft_bench57_aff_a"),
        ]
        kit = X.noop_coverage_kit(
            snapshots=snaps,
            contributor_ids=["cA", "cB"],
            receiver_agent_id="recvX",
            slot_to_target={0: "aff_c", 1: "aff_a", 2: "aff_b", 3: "aff_d"},
            workdir=tmp_path / "kit",
        )
        assert kit["kit_pass"] is True, kit
        assert kit["receiver_unchanged"]["coverage"] == 0.0
        assert kit["empty_fold"]["coverage"] == 0.0


# ── seed parameterization (R0's multi-seed requirement) ──────────────────


class TestSeedParameterization:
    def test_contributor_seeds_depend_on_seed_base(self):
        a = X.contributor_seeds(42, 4, 4, salt=1)
        b = X.contributor_seeds(43, 4, 4, salt=1)
        assert a != b, "different seed-base must give different contributor seeds"
        assert len(set(a)) == 4, "contributors within a cohort are independent"

    def test_schedules_diverge_across_contributors(self):
        # Independent seeds -> different presentation ORDER (the coverage-
        # widening driver); byte-identical contributors would force a flat curve.
        s1, order1 = X.contributor_schedule(101, reps_per_cell=1)
        s2, order2 = X.contributor_schedule(202, reps_per_cell=1)
        assert s1 != s2
        assert order1 != order2 or s1 != s2
        # Same seed is reproducible.
        assert X.contributor_schedule(101, reps_per_cell=1)[0] == s1

    def test_schedule_is_balanced_within_contingency(self):
        trials, order = X.contributor_schedule(7, reps_per_cell=2)
        assert sorted(order) == list(range(X.G))
        # Each contingency block has 2 states x 8 affs x 2 reps = 32 trials.
        from collections import Counter

        per_contingency = Counter(g for g, _s, _a in trials)
        assert all(v == 2 * 8 * 2 for v in per_contingency.values())

    def test_schedule_extends_for_single_matched_budget(self):
        # single_matched trains ONE contributor to N*K_max trials, more than one
        # balanced pass (G*2*|AFF|*reps = 64 at reps=1). The schedule must REPEAT
        # passes to reach that budget (code-lens finding 1) — not truncate.
        one_pass = X.G * 2 * len(C.AFFORDANCES) * 1
        assert one_pass == 64
        big = 8 * 30  # rung 8 * a realistic K_max
        trials, order = X.contributor_schedule(11, reps_per_cell=1, min_trials=big)
        assert len(trials) >= big, "schedule must cover N*K_max for single_matched"
        assert sorted(order) == list(range(X.G))
        # Each repeated pass stays balanced: counts are equal across contingencies.
        from collections import Counter

        per_contingency = Counter(g for g, _s, _a in trials)
        assert len(set(per_contingency.values())) == 1, "repeated passes stay balanced"

    def test_schedule_default_is_one_pass(self):
        # No min_trials -> exactly one balanced pass (backward compatible).
        trials, _ = X.contributor_schedule(11, reps_per_cell=1)
        assert len(trials) == X.G * 2 * len(C.AFFORDANCES) * 1


# ── ScriptedBridge one-client faithfulness (reused apparatus) ────────────


# ── the richer bench57 world channel separates the G=4 slots ─────────────


class TestWorldSeparation:
    """The Exp 57 apparatus fix: minecraft_bench's DIRECTION-BLIND world channel
    (distance-magnitude + altitude) could resolve only ~5-6 clusters, so the
    four FROZEN contingency slots — same distance, same y, differing only in
    BEARING — barely separated (jitter-fragile). minecraft_bench57 adds the
    SIGNED offset_x/offset_z so the four slots encode to four DISTINCT world
    clusters through the real SensorEncoder + world_ranges() +
    world_sensors_for_slot().

    MEASURED THROUGH THE REAL ENCODER (all 7 declared bench57 world sensors):
    max pairwise cosine 0.8387 — below the 0.85 pattern-completion threshold
    (so the four slots DO land in four distinct clusters), but by a THIN
    ~0.011 margin. The offsets are the only sensors that differ across the
    four FROZEN slots; the other five kept world sensors are identical across
    all four and dilute the discriminating signal. The task's stated
    offline figure (0.4947) was measured on an offsets-only / smaller probe,
    not the full 7-sensor body through the SHA-basis encoder — see the PR
    report's separation finding. This guard pins that the slots still resolve
    (< 0.85), NOT that the margin is robust."""

    def _cosine(self, a, b):
        import math

        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def test_four_slots_separate_below_the_pattern_threshold(self):
        from maxim.similarity.ec import EntorhinalCortex
        from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

        ranges = X.world_ranges()
        assert {"offset_x", "offset_z"} <= set(ranges), "bench57 must declare the signed-position sensors"

        def embed(sensors):
            ec = EntorhinalCortex()
            enc = SensorEncoder(ec=ec, config=SensorEncoderConfig())
            filtered = {k: v for k, v in sensors.items() if k in ranges}
            nid = enc.encode_sensors(agent_id="sep", sensors=filtered, modality="world", ranges=ranges)
            assert nid is not None, f"slot {sensors} encoded to designed-rest (no cluster)"
            # world is a frozen-centroid modality: the stored node embedding is
            # the first-observation embedding, so this is the encode vector.
            return list(ec._substrate_nodes[nid][0])

        slot_embeds = [embed(X.world_sensors_for_slot(s)) for s in X.CONTINGENCY_SLOTS]
        max_cos = max(self._cosine(slot_embeds[i], slot_embeds[j]) for i in range(X.G) for j in range(i + 1, X.G))
        # 0.8387 through the real encoder — below 0.85, so the four slots do
        # resolve, but the margin is thin (see the class docstring's finding).
        assert max_cos < 0.85, f"the four slots do not separate (max pairwise cosine {max_cos:.4f})"

    def test_four_distinct_clusters_and_distinct_from_rest(self):
        from maxim.similarity.ec import EntorhinalCortex
        from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

        ranges = X.world_ranges()
        ec = EntorhinalCortex()
        enc = SensorEncoder(ec=ec, config=SensorEncoderConfig())

        def encode(sensors):
            filtered = {k: v for k, v in sensors.items() if k in ranges}
            return enc.encode_sensors(agent_id="shared", sensors=filtered, modality="world", ranges=ranges)

        rest_id = encode(X.world_sensors_for_slot(C.FROZEN["rest_anchor"]))
        slot_ids = [encode(X.world_sensors_for_slot(s)) for s in X.CONTINGENCY_SLOTS]
        assert all(sid is not None for sid in slot_ids), slot_ids
        assert len(set(slot_ids)) == X.G, f"slots collapsed to {len(set(slot_ids))} clusters: {slot_ids}"
        if rest_id is not None:  # rest is non-neutral (time_of_day) so it clusters
            assert rest_id not in slot_ids, "a contingency slot collapsed onto the rest cluster"


class TestScriptedBridgeOneClient:
    def test_second_overlapping_client_is_rejected_busy(self):
        import socket
        import time

        srv = C.ScriptedBridgeServer(seed=1)
        try:
            a = socket.create_connection(("127.0.0.1", srv.port), timeout=2)
            time.sleep(0.2)
            b = socket.create_connection(("127.0.0.1", srv.port), timeout=2)
            b.settimeout(2.0)
            data = b.recv(4096).decode()
            assert "busy" in data.lower(), f"second client should be rejected busy, got {data!r}"
            a.close()
            b.close()
        finally:
            srv.close()
