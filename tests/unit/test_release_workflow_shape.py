"""The publish workflow's shape (#1372): .github/workflows/release.yml, read PARSED (a comment satisfies nothing).

Each pin is a way the workflow could publish the wrong bytes, publish without the owner, or hold a credential it
should not. Design and its adversarial pass: docs/plans/archive/release_pipeline.md."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEXT = (ROOT / ".github/workflows/release.yml").read_text()
WF = yaml.safe_load(TEXT)
JOBS = WF["jobs"]
ON = WF.get("on", WF.get(True))  # PyYAML reads the bare key `on` as True


def _runs(job: str) -> list[str]:
    return [re.sub(r"\s+", " ", str(s.get("run", ""))) for s in JOBS[job]["steps"]]


def _index(job: str, needle: str) -> int:
    hits = [i for i, r in enumerate(_runs(job)) if needle in r]
    assert len(hits) == 1, (job, needle, hits)
    return hits[0]


def test_it_runs_on_v_tags_and_a_dispatch_rehearsal_only():
    assert ON == {"push": {"tags": ["v*"]}, "workflow_dispatch": None}
    assert WF["concurrency"] == {"group": "release-${{ github.ref }}", "cancel-in-progress": False}


def test_every_step_runs_with_pipefail():
    """Executor review B1: GitHub's default `bash -e {0}` has no pipefail, so `audit | tee` passed a failed audit."""
    assert WF["defaults"] == {"run": {"shell": "bash"}}
    assert not any("shell" in s for job in JOBS.values() for s in job["steps"])


def test_permissions_are_per_job_and_only_publish_can_mint_an_oidc_token():
    assert WF["permissions"] == {}
    holders = [name for name, job in JOBS.items() if (job.get("permissions") or {}).get("id-token")]
    assert holders == ["publish"] and JOBS["publish"]["permissions"]["id-token"] == "write"
    writers = [name for name, job in JOBS.items() if (job.get("permissions") or {}).get("contents") == "write"]
    assert writers == ["github-release"]


def test_publish_waits_for_the_owner_and_never_runs_on_a_rehearsal():
    gate = "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
    assert JOBS["publish"]["environment"]["name"] == "pypi"
    assert JOBS["publish"]["if"] == gate and JOBS["github-release"]["if"] == gate
    assert set(JOBS["publish"]["needs"]) == {"build", "smoke"}


def test_no_stored_credential_is_used():
    assert not re.search(r"PYPI_TOKEN|TWINE_PASSWORD|password:|secrets\.(?!GITHUB_TOKEN)", TEXT)


def test_every_action_is_pinned_by_commit_sha():
    uses = [s["uses"] for job in JOBS.values() for s in job["steps"] if "uses" in s]
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u) for u in uses), uses


def test_build_installs_what_its_scripts_import():
    """Wire review B1: lint_release_gates.py fails without pyyaml, so a build job without it can never publish."""
    install = _runs("build")[_index("build", "pip install")]
    assert "pyyaml" in install and "build" in install and "twine" in install
    assert _index("build", "pip install") < _index("build", "lint_release_gates.py")


def test_the_tagger_identity_is_pinned():
    assert WF["env"]["RELEASE_TAGGER_EMAIL"] == "dennys@rmdig.ai"


def test_build_verifies_then_vendors_then_builds_then_audits():
    order = [
        _index("build", "check_release_tag.py tag"),
        _index("build", "check_release_tag.py environment"),
        _index("build", "check_release_tag.py ruleset"),
        _index("build", "check_nightlies.py --at"),
        _index("build", "vendor_console_ui.py --verify-pin"),
        _index("build", "vendor_console_ui.py --from-pin"),
        _index("build", "python -I -m build"),
        _index("build", "audit_release_build.py"),
    ]
    assert order == sorted(order), order
    assert JOBS["build"]["steps"][0]["with"]["fetch-depth"] == 0  # N2


def test_the_release_path_never_uses_the_ci_only_relaxations():
    """Architecture review of R4: in a tag-push run the tag exists, so `releasing()` reads False; either flag would
    turn the console contract check into a note and the freshness check into a no-op during the real publish."""
    for flag in ("--only-when-releasing", "--contract-drift-ok-unless-releasing", "--force"):
        assert flag not in TEXT, flag


def test_artifacts_are_checked_against_build_outputs_before_use():
    """SF6: digests travel as job outputs, not inside the artifact they vouch for."""
    outputs = JOBS["build"]["outputs"]
    assert {"wheel_sha256", "sdist_sha256", "tag_object", "console"} <= set(outputs)
    for job in ("smoke", "publish", "github-release"):
        assert any("needs.build.outputs.wheel_sha256" in r and "sha256sum -c" in r for r in _runs(job)), job


def test_publish_re_checks_after_the_approval_wait_then_proves_what_pypi_serves():
    upload = next(i for i, s in enumerate(JOBS["publish"]["steps"]) if "gh-action-pypi-publish" in s.get("uses", ""))
    for needle in ("check_release_tag.py still", "check_nightlies.py --at", "vendor_console_ui.py --verify-pin"):
        assert _index("publish", needle) < upload, needle
    step = JOBS["publish"]["steps"][upload]["with"]
    assert step["attestations"] is True and step["skip-existing"] is True
    assert _index("publish", "pypi.org/pypi/pymaxim") > upload


def test_the_smoke_test_imports_the_installed_wheel_not_a_checkout():
    run = " ".join(_runs("smoke"))
    assert not any("actions/checkout" in s.get("uses", "") for s in JOBS["smoke"]["steps"])
    for needle in ('"smoke-venv" in here.parts', "maxim.diagnose()", "ui_dist", "-I -m maxim --help", "__version__"):
        assert needle in run, needle


def test_job_names_do_not_collide_with_required_checks():
    """N5: check_repo_settings --static needs each required context to be exactly one job across workflows."""
    names = {job.get("name") for job in JOBS.values()}
    assert not names & {"lint", "unit-tests", "Release build (wheel contents + version)"}
