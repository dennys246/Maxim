"""scripts/check_repo_settings.py — the merge gate's settings stay as committed (M6, D63; owner decisions 2026-10-06).

Every test drives `observe()` through a fake `gh api` built from the committed expected file, then perturbs ONE
setting, so the positive controls pin each kind of drift against the real expected shape.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import check_repo_settings as S

EXPECTED = json.loads(S.EXPECTED.read_text(encoding="utf-8"))
REPO = "o/r"
BYPASS_KEY = f"repos/{REPO}/rulesets/rule-suites?ref=refs/heads/main&rule_suite_result=bypass&time_period=month&per_page=100&page=1"
ENV_KEY = f"repos/{REPO}/environments/settings-check"


def _responses(exp: dict) -> dict:
    """The API responses that observe() would normalise back into `exp`."""
    return {
        f"repos/{REPO}": {
            "default_branch": exp["default_branch"],
            "security_and_analysis": copy.deepcopy(exp["security_and_analysis"]),
        },
        f"repos/{REPO}/branches/main/protection": copy.deepcopy(exp["protection"]),
        f"repos/{REPO}/rules/branches/main?per_page=100": copy.deepcopy(exp["effective_rules"]),
        **{f"repos/{REPO}/rulesets/{rid}": {"enforcement": e} for rid, e in exp["ruleset_enforcement"].items()},
        f"repos/{REPO}/code-scanning/default-setup": copy.deepcopy(exp["codeql_default_setup"]),
        BYPASS_KEY: [],
        ENV_KEY: {"deployment_branch_policy": exp["token_environment"]["deployment_branch_policy"]},
        f"{ENV_KEY}/deployment-branch-policies?per_page=100": {
            "branch_policies": [{"id": 1, **b} for b in exp["token_environment"]["branch_policies"]]
        },
    }


def _api(responses: dict):
    def api(path: str):
        if path not in responses:
            raise AssertionError(f"unexpected path {path}")
        value = responses[path]
        if isinstance(value, Exception):
            raise value
        return value

    return api


def _check(mutate=None) -> list[str]:
    responses = _responses(EXPECTED)
    if mutate:
        mutate(responses)
    return S.compare(EXPECTED, S.observe(REPO, _api(responses)))


def test_the_committed_settings_compare_clean():
    assert _check() == []


@pytest.mark.parametrize(
    ("label", "mutate", "needle"),
    [
        (
            "a required check removed",
            lambda r: r[f"repos/{REPO}/branches/main/protection"]["required_status_checks"]["contexts"].pop(),
            "protection.",
        ),
        (
            "admins un-enforced",
            lambda r: r[f"repos/{REPO}/branches/main/protection"]["enforce_admins"].update(enabled=False),
            "protection.",
        ),
        (
            "strict mode off",
            lambda r: r[f"repos/{REPO}/branches/main/protection"]["required_status_checks"].update(strict=False),
            "protection.",
        ),
        (
            "the default branch renamed",
            lambda r: r[f"repos/{REPO}"].update(default_branch="trunk"),
            "default_branch drifted",
        ),
        (
            "a new ruleset rule on main",
            lambda r: r[f"repos/{REPO}/rules/branches/main?per_page=100"].append(
                {"type": "update", "parameters": None, "ruleset_id": 99}
            )
            or r.update({f"repos/{REPO}/rulesets/99": {"enforcement": "active"}}),
            "effective_rules drifted",
        ),
        (
            "push protection switched off",
            lambda r: r[f"repos/{REPO}"]["security_and_analysis"]["secret_scanning_push_protection"].update(
                status="disabled"
            ),
            "security_and_analysis.secret_scanning_push_protection",
        ),
        (
            "non-provider patterns changed (pinned disabled 2026-10-10; any change is drift)",
            lambda r: r[f"repos/{REPO}"]["security_and_analysis"]["secret_scanning_non_provider_patterns"].update(
                status="enabled"
            ),
            "security_and_analysis.secret_scanning_non_provider_patterns",
        ),
        (
            "CodeQL loses python",
            lambda r: r[f"repos/{REPO}/code-scanning/default-setup"]["languages"].remove("python"),
            "codeql_default_setup.",
        ),
    ],
)
def test_each_kind_of_drift_FAILS(label, mutate, needle):
    out = _check(mutate)
    assert any(needle in p for p in out), (label, out)


def test_a_ruleset_switched_to_evaluate_FAILS():
    rid = next(iter(EXPECTED["ruleset_enforcement"]))
    out = _check(lambda r: r.update({f"repos/{REPO}/rulesets/{rid}": {"enforcement": "evaluate"}}))
    assert any("ruleset_enforcement." in p for p in out), out


def test_an_unacknowledged_bypass_FAILS_and_an_acknowledged_one_passes():
    suite = {"id": 7, "actor_name": "someone", "pushed_at": "2026-10-06T00:00:00Z", "after_sha": "abc"}
    key = BYPASS_KEY
    out = _check(lambda r: r.update({key: [suite]}))
    assert any("was BYPASSED on main: suite 7 by someone" in p for p in out), out
    acked = {**EXPECTED, "acknowledged_bypasses": [{"id": 7, "reason": "release hotfix", "date": "2026-10-06"}]}
    responses = _responses(EXPECTED)
    responses[key] = [suite]
    assert S.compare(acked, S.observe(REPO, _api(responses))) == []


def test_bypasses_are_read_past_the_first_page():
    page2 = BYPASS_KEY.replace("&page=1", "&page=2")
    first = [{"id": i, "actor_name": "a"} for i in range(100)]
    acked = {**EXPECTED, "acknowledged_bypasses": [{"id": i} for i in range(100)]}
    responses = _responses(EXPECTED)
    responses[BYPASS_KEY] = first
    responses[page2] = [{"id": 999, "actor_name": "late"}]
    out = S.compare(acked, S.observe(REPO, _api(responses)))
    assert any("suite 999 by late" in p for p in out), out


def test_a_missing_or_unfenced_token_environment_is_drift():
    missing = _check(lambda r: r.update({ENV_KEY: S.NotFound("404")}))
    assert any("token_environment drifted" in p and "MISSING" in p for p in missing), missing
    unfenced = _check(lambda r: r.update({ENV_KEY: {"deployment_branch_policy": None}}))
    assert any("token_environment" in p for p in unfenced), unfenced
    widened = _check(
        lambda r: r[f"{ENV_KEY}/deployment-branch-policies?per_page=100"]["branch_policies"].append(
            {"id": 2, "name": "*", "type": "branch"}
        )
    )
    assert any("token_environment.branch_policies drifted" in p for p in widened), widened


def test_a_new_github_field_is_named_not_dumped():
    out = _check(lambda r: r[f"repos/{REPO}/branches/main/protection"].update(lock_branch_v2={"enabled": False}))
    assert out == [
        'protection.lock_branch_v2 drifted: a NEW GitHub field (review it, then pin it): {"enabled": false}'
    ], out


def test_an_unprotected_branch_is_drift_not_unverifiable():
    out = _check(lambda r: r.update({f"repos/{REPO}/branches/main/protection": S.Unprotected("x")}))
    assert any("protection drifted" in p and "UNPROTECTED" in p for p in out), out


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update({f"repos/{REPO}/branches/main/protection": S.CannotVerify("HTTP 401")}),
        lambda r: r.update({f"repos/{REPO}/code-scanning/default-setup": {"state": "configured"}}),  # field missing
        lambda r: r.update({BYPASS_KEY: {"message": "x"}}),
        lambda r: r[f"repos/{REPO}"].pop("security_and_analysis"),  # a token that cannot see it (#1081 D3)
        lambda r: r[f"repos/{REPO}"].update(security_and_analysis=None),
    ],
)
def test_an_unreadable_part_cannot_verify_never_passes(mutate):
    responses = _responses(EXPECTED)
    mutate(responses)
    with pytest.raises(S.CannotVerify):
        S.observe(REPO, _api(responses))


def test_no_token_is_exit_2(monkeypatch, capsys):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert S.main([]) == 2
    assert "SETTINGS_READ_TOKEN" in capsys.readouterr().err


# ── the static half: a required check maps to exactly one unconditional job ──────────────────────────


def _wf(tmp_path: Path, jobs: str) -> Path:
    d = tmp_path / "workflows"
    d.mkdir()
    (d / "test.yml").write_text(f"on: push\njobs:\n{jobs}")
    return d


def test_the_real_workflows_satisfy_the_static_rules():
    assert S.static_problems(EXPECTED["protection"]["required_status_checks"]["contexts"]) == []


def test_a_skippable_required_job_FAILS(tmp_path):
    wf = _wf(tmp_path, "  lint:\n    if: github.event_name == 'push'\n    runs-on: x\n    steps: []\n")
    assert any("job-level `if:`" in p for p in S.static_problems(["lint"], wf))


def test_a_required_job_needing_a_skippable_job_FAILS(tmp_path):
    wf = _wf(
        tmp_path,
        "  gate:\n    if: github.event_name == 'push'\n    runs-on: x\n    steps: []\n"
        "  mid:\n    needs: gate\n    runs-on: x\n    steps: []\n"
        "  lint:\n    needs: [mid]\n    runs-on: x\n    steps: []\n",
    )
    assert any("needs gate" in p for p in S.static_problems(["lint"], wf))


def test_a_missing_or_duplicated_required_job_FAILS(tmp_path):
    wf = _wf(
        tmp_path,
        "  a:\n    name: lint\n    runs-on: x\n    steps: []\n  b:\n    name: lint\n    runs-on: x\n    steps: []\n",
    )
    assert any("is the name of 2 jobs" in p for p in S.static_problems(["lint"], wf))
    assert any("is the name of 0 jobs" in p for p in S.static_problems(["unit-tests"], wf))
