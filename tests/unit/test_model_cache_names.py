"""The nightly warm list is derived from source (scripts/model_cache_names.py) and must cover
every model a marked test or the production loaders name. Verified to fail on the pre-fix
workflow's hand-kept tuple, which lacked all-MiniLM-L6-v2 (every nightly run since 2026-08-21)."""

from __future__ import annotations


from scripts import model_cache_names as M

PRODUCTION_DEFAULTS = {"all-mpnet-base-v2", "paraphrase-mpnet-base-v2", "clip-ViT-B-32", "all-MiniLM-L6-v2"}


def test_derived_list_covers_production_defaults_and_marked_tests() -> None:
    names = set(M.model_names())
    assert PRODUCTION_DEFAULTS <= names, names
    assert "paraphrase-MiniLM-L6-v2" in names  # the test_model_comparison sweep arm


def test_marked_test_files_are_found() -> None:
    files = {p.name for p in M.marked_test_files()}
    assert "test_baselines.py" in files and "test_clip_encoder.py" in files


def test_the_lane_installs_and_requires_what_it_collects() -> None:
    """Both nightly lanes install the console + sign extras through the shared setup action (#1117) and fail a skip
    for them with --require-extras."""
    action = (M.REPO_ROOT / ".github" / "actions" / "model-cache-setup" / "action.yml").read_text()
    assert ".[semantic,test,console,sign]" in action
    workflow = (M.REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    for job in ("model-cache-tests:", "slow-tests:"):
        start = workflow.index(job)
        body = workflow[start : workflow.index("\n  # ──", start)]
        assert "uses: ./.github/actions/model-cache-setup" in body, job
        assert "--require-extras=console,sign" in body, job
