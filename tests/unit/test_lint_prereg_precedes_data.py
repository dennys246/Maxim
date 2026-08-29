"""Fixture-repo tests for scripts/lint_prereg_precedes_data.py (roadmap 1.1.x item 16.8).

The positive control for the CI step: a real git repo built in tmp_path with the
order WRONG (data before its pre-registration reached the ref) must fail, and the
same repo with the order right must pass. Verified to fail 5/5 order-sensitive cases
when the assertion is inverted (i.e. the lint's `<` is what carries the rule).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import lint_prereg_precedes_data as L

T0 = 1_800_000_000  # epoch seconds; commits are placed relative to this


def _git(root: Path, *args: str, when: int | None = None) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    if when is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = f"{when} +0000"
    r = subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True)
    return r.stdout


class Repo:
    """A tiny experiments tree: result doc → prereg link, data entries, commits at chosen times."""

    def __init__(self, root: Path) -> None:
        self.root = root
        _git(root, "init", "-q", "-b", "main")
        _git(root, "config", "commit.gpgsign", "false")
        (root / "docs/experiments/protocols").mkdir(parents=True)
        (root / "docs/experiments/data").mkdir(parents=True)

    def write(self, rel: str, text: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def commit(self, msg: str, when: int) -> None:
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "-m", msg, when=when)

    def result_doc(self, token: str, prereg_name: str, extra: str = "") -> None:
        self.write(
            f"docs/experiments/{token}_thing.md",
            f"# Exp {token}\n\nPre-registered in [protocols/{prereg_name}](protocols/{prereg_name}).\n{extra}",
        )

    def prereg(self, name: str, amendments: str = "") -> None:
        self.write(f"docs/experiments/protocols/{name}", f"# prereg\n\nfrozen gates\n\n## Amendments\n\n{amendments}")

    def data(self, name: str, ts: list[float], extra: dict | None = None) -> None:
        rows = [json.dumps({"ts": t, "event": "start", **(extra or {})}) for t in ts]
        self.write(f"docs/experiments/data/{name}", "\n".join(rows) + "\n")


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return Repo(tmp_path)


def run(repo: Repo, **kw) -> int:
    kw.setdefault("grandfathered", {})
    return L.lint(repo.root, "main", **kw)


def test_prereg_before_data_passes(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 3600])
    repo.commit("data", T0 + 7200)
    assert run(repo) == 0
    assert "1 governed data entry checked" in capsys.readouterr().out


def test_data_before_prereg_fails(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.data("61_results.jsonl", [T0 - 3600])  # first record an hour BEFORE the prereg lands
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg + data (the 53b shape)", T0)
    assert run(repo) == 1
    err = capsys.readouterr().err
    assert "61_results.jsonl" in err and "not before the data" in err


def test_same_commit_fails_even_with_later_ts(repo: Repo) -> None:
    """Data whose ts is later than the squash time but whose prereg is IN the squash: the
    prereg's first-commit time equals the data's fallback — strict `<` fails."""
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.write("docs/experiments/data/61_rows.jsonl", json.dumps({"event": "row"}) + "\n")  # no ts → fallback
    repo.commit("squash", T0)
    assert run(repo) == 1


def test_pre_data_amendment_after_data_fails(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    repo.prereg("exp61_preregistration.md", "**Amendment 1 — 2026-01-01, PRE-DATA, structural.** text\n")
    repo.commit("amendment after the fact", T0 + 300)
    assert run(repo) == 1
    assert "PRE-DATA amendment 1" in capsys.readouterr().err


def test_post_data_amendment_is_noted_not_judged(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    repo.prereg("exp61_preregistration.md", "**Amendment 1 — 2026-01-01, POST-DATA relabel.** text\n")
    repo.commit("post-data amendment", T0 + 300)
    assert run(repo) == 0
    assert "not marked PRE-DATA" in capsys.readouterr().out


def test_lettered_token_is_governed_by_parent_prereg(repo: Repo, capsys) -> None:
    """61b data is governed by the 61b prereg AND the 61 prereg (the 53/53b shape)."""
    repo.result_doc(
        "61",
        "exp61_preregistration.md",
        "Delta: [protocols/exp61b_preregistration.md](protocols/exp61b_preregistration.md)",
    )
    repo.prereg("exp61b_preregistration.md")
    repo.commit("61b prereg only", T0)
    repo.data("61b_results.jsonl", [T0 + 100])
    repo.commit("61b data", T0 + 200)
    repo.prereg("exp61_preregistration.md")
    repo.commit("parent prereg lands late", T0 + 300)
    assert run(repo) == 1
    assert "exp61_preregistration.md" in capsys.readouterr().err


def test_dry_run_entries_are_skipped(repo: Repo) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.data("61_dry_run_nonfrozen.jsonl", [T0 - 9999])
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg + shakedown", T0)
    assert run(repo) == 0


def test_allow_dirty_must_be_echoed_in_result_doc(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100], {"allow_dirty": True})
    repo.commit("data", T0 + 200)
    assert run(repo) == 1
    assert "allow_dirty" in capsys.readouterr().err
    repo.result_doc("61", "exp61_preregistration.md", "Run with `--allow-dirty`: records carry `allow_dirty: true`.")
    repo.commit("echo", T0 + 300)
    assert run(repo) == 0


def test_grandfathered_entry_is_reported_and_must_still_fail(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.data("61_results.jsonl", [T0 - 3600])
    repo.prereg("exp61_preregistration.md")
    repo.commit("squash", T0)
    gf = {"docs/experiments/data/61_results.jsonl": "the incident"}
    assert run(repo, grandfathered=gf) == 0
    out = capsys.readouterr().out
    assert "GRANDFATHERED (still failing)" in out and "the incident" in out
    # A grandfathered entry that now passes is stale — the lint says so.
    repo.data("61_results.jsonl", [T0 + 3600])
    repo.commit("rewritten", T0 + 7200)
    assert run(repo, grandfathered=gf) == 1
    assert "now PASSES" in capsys.readouterr().err


def test_missing_ref_is_exit_2_not_pass(repo: Repo) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    assert L.lint(repo.root, "no-such-ref", grandfathered={}) == 2


def test_token_rules() -> None:
    assert L.token_of("exp53b_cross_context_readout_delta_preregistration.md") == "53b"
    assert L.token_of("h1_healthy_hardware_doa_preregistration.md") == "h1"
    assert L.token_of("44b_pilot") == "44b"
    assert L.token_of("53b_cross_context_readout.jsonl") == "53b"
    assert L.parent_token("53b") == "53" and L.parent_token("53") is None and L.parent_token("h1") is None


def test_real_repo_grandfather_list_names_existing_files() -> None:
    for key in L.GRANDFATHERED:
        assert (L.REPO_ROOT / key).exists(), key
