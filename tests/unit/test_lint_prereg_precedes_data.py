"""Fixture-repo tests for scripts/lint_prereg_precedes_data.py (roadmap 1.1.x item 16.8).

The positive control for the CI step: a real git repo built in tmp_path with the
order WRONG (data before its pre-registration reached the ref) must fail, and the
same repo with the order right must pass. The merge-commit case fails
without `--first-parent` (the review-round BLOCKER); the others fail with the `<` inverted.
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
    kw.setdefault("not_governed", {})
    kw.setdefault("ungoverned_reruns", {})
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
    repo.write("docs/experiments/data/61_inputs.json", json.dumps({"kind": "input"}) + "\n")  # no ts → fallback
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
    assert "POST-DATA — reported, not judged" in capsys.readouterr().out


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


def test_dry_run_entries_are_skipped(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.data("61_dry_run_nonfrozen.jsonl", [T0 - 9999])  # a shakedown that predates the prereg — exempt by name
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg + shakedown", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    assert run(repo) == 0
    assert "1 governed data entry checked" in capsys.readouterr().out


def test_allow_dirty_must_be_echoed_in_result_doc(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100], {"allow_dirty": True})
    repo.commit("data", T0 + 200)
    assert run(repo) == 1
    assert "allow_dirty" in capsys.readouterr().err
    repo.result_doc(
        "61",
        "exp61_preregistration.md",
        "Run with `--allow-dirty`: `data/61_results.jsonl` carries `allow_dirty: true`.",
    )
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


def test_merge_committed_prereg_is_judged_at_merge_time_not_branch_time(repo: Repo, capsys) -> None:
    """The incident under the brief's mandated merge style: prereg committed on a branch at T0,
    data at T0+100 (still on the branch), --no-ff merge to main at T0+1000. Without --first-parent
    the lint read the BRANCH time and passed (both review lenses caught it)."""
    repo.result_doc("61", "exp61_preregistration.md")
    repo.commit("doc", T0 - 10)
    _git(repo.root, "checkout", "-q", "-b", "feat")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg on branch", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data on branch", T0 + 200)
    _git(repo.root, "checkout", "-q", "main")
    _git(repo.root, "merge", "-q", "--no-ff", "-m", "merge", "feat", when=T0 + 1000)
    assert run(repo) == 1
    assert "not before the data" in capsys.readouterr().err


def test_dirty_stamp_without_allowance_fails(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100], {"provenance": {"working_tree_dirty_src_scripts": True}})
    repo.commit("data", T0 + 200)
    assert run(repo) == 1
    assert "without allow_dirty: true" in capsys.readouterr().err


def test_naive_iso_ts_fails_unless_grandfathered(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.write("docs/experiments/data/61_results.jsonl", json.dumps({"ts": "2026-08-10T11:53:49", "event": "x"}) + "\n")
    repo.commit("data", T0 + 200)
    assert run(repo) == 1
    assert "naive ISO-8601" in capsys.readouterr().err
    assert run(repo, grandfathered={"docs/experiments/data/61_results.jsonl": "naive pilot"}) == 0


def test_malformed_amendment_header_is_exit_2_not_skip(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md", "**Amendment 1 - 2026-01-01, structural.** no class, hyphen not em dash\n")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    assert run(repo) == 2
    assert "unclassified amendment" in capsys.readouterr().err


def test_wrapped_amendment_header_is_parsed(repo: Repo) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg(
        "exp61_preregistration.md",
        "**Amendment 1 — 2026-01-01, PRE-DATA, structural (harness dry run at non-frozen\nconstants).** text\n",
    )
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    assert run(repo) == 0


def test_zero_governed_entries_is_exit_2(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")  # a prereg link, but no prereg file and no data
    repo.commit("doc only", T0)
    assert run(repo) == 2
    assert "zero governed" in capsys.readouterr().err


def test_unlinked_prereg_still_governs_its_data(repo: Repo, capsys) -> None:
    repo.write("docs/experiments/61_thing.md", "# no prereg link here\n")
    repo.data("61_results.jsonl", [T0 - 100])
    repo.prereg("exp61_preregistration.md")
    repo.commit("squash", T0)
    assert run(repo) == 1


def test_post_2026_08_29_jsonl_without_ts_fails(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", L.TS_REQUIRED_FROM + 10)
    repo.write("docs/experiments/data/61_rows.jsonl", json.dumps({"event": "row"}) + "\n")
    repo.commit("data without ts", L.TS_REQUIRED_FROM + 3600)
    assert run(repo) == 1
    assert "must carry epoch `ts`" in capsys.readouterr().err


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


# ── 1.3.1: the flat `*_prereg.md` layout, instruments, token collisions ─────────────────────────


def _flat_prereg(repo: Repo, name: str) -> None:
    repo.write(f"docs/experiments/{name}", "# prereg\n\nfrozen gates\n")


def test_a_flat_prereg_governs_its_data(repo: Repo, capsys) -> None:
    """1.3.0's own experiments (Exp 60, 61, R3) used this layout and were read by nobody."""
    repo.data("exp60_trials.jsonl", [T0 - 3600])  # data BEFORE the prereg reached main
    _flat_prereg(repo, "exp60_drowning_avoidance_prereg.md")
    repo.commit("squash", T0)
    assert run(repo) == 1
    assert "exp60_drowning_avoidance_prereg.md" in capsys.readouterr().err


def test_a_script_beside_the_records_is_not_a_record(repo: Repo, capsys) -> None:
    _flat_prereg(repo, "exp60_drowning_avoidance_prereg.md")
    repo.write("docs/experiments/data/exp60_oxygen_window_check.py", "print('check')\n")  # same commit
    repo.commit("prereg + its check script", T0)
    repo.data("exp60_trials.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    assert run(repo) == 0
    assert "1 governed data entry checked" in capsys.readouterr().out


def test_a_token_collision_is_declared_not_judged_and_goes_stale(repo: Repo, capsys) -> None:
    repo.data("r2_drive_premise.json", [T0 - 3600])  # an earlier, un-preregistered experiment on rung R2
    repo.commit("premise check", T0 - 3000)
    _flat_prereg(repo, "r2_learned_bias_prereg.md")
    repo.data("r2_learned_bias_rows.jsonl", [T0 + 100])
    repo.commit("prereg", T0)
    ng = {"docs/experiments/data/r2_drive_premise.json": "the premise check"}
    assert run(repo) == 1  # undeclared: judged against the learned-bias prereg, and fails
    capsys.readouterr()
    assert run(repo, not_governed=ng) == 0  # declared: noted, not judged; the learned-bias rows still are
    out, err = capsys.readouterr()
    assert "NOT GOVERNED (token collision)" in out and "r2_drive_premise" not in err
    # Stale: once nothing matches its token, the declaration must go.
    (repo.root / "docs/experiments/r2_learned_bias_prereg.md").unlink()
    (repo.root / "docs/experiments/data/r2_learned_bias_rows.jsonl").unlink()
    _flat_prereg(repo, "exp60_x_prereg.md")
    repo.commit("drop", T0 + 500)
    repo.data("exp60_trials.jsonl", [T0 + 600])
    repo.commit("other data", T0 + 700)
    assert run(repo, not_governed=ng) == 1
    assert "matches no prereg" in capsys.readouterr().err


def test_real_repo_governs_the_1_3_experiments() -> None:
    preregs, _docs, _notes = L.prereg_map(L.REPO_ROOT)
    for token in ("60", "61", "62", "r3"):
        assert any(p.name.endswith("_prereg.md") for p in preregs.get(token, ())), token


def test_real_repo_not_governed_list_names_existing_files() -> None:
    for key in L.NOT_GOVERNED:
        assert (L.REPO_ROOT / key).exists(), key


def test_a_blockquoted_pre_data_amendment_is_judged(repo: Repo, capsys) -> None:
    """Exp 60 wrote its freeze amendment as `> **Amendment 2 — …**`; a regex anchored at `**` never saw it."""
    repo.write("docs/experiments/exp60_x_prereg.md", "# prereg\n\nfrozen gates\n")
    repo.commit("prereg", T0)
    repo.data("exp60_trials.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    repo.write(
        "docs/experiments/exp60_x_prereg.md",
        "# prereg\n\n> **Amendment 1 — 2026-09-15, PRE-DATA, late.**\n\nfrozen gates\n",
    )
    repo.commit("late amendment", T0 + 300)
    assert run(repo) == 1
    assert "PRE-DATA amendment 1" in capsys.readouterr().err


def test_a_result_doc_linking_a_flat_prereg_can_echo_allow_dirty(repo: Repo, capsys) -> None:
    repo.write("docs/experiments/exp60_x_prereg.md", "# prereg\n\nfrozen gates\n")
    repo.write(
        "docs/experiments/60_results.md",
        "Prereg: [exp60_x_prereg.md](exp60_x_prereg.md). `exp60_trials.jsonl` carries `allow_dirty: true`.\n",
    )
    repo.commit("prereg", T0)
    repo.data("exp60_trials.jsonl", [T0 + 100], {"allow_dirty": True})
    repo.commit("data", T0 + 200)
    assert run(repo) == 0, capsys.readouterr().err


def test_a_not_governed_entry_naming_a_missing_file_fails(repo: Repo, capsys) -> None:
    _flat_prereg(repo, "exp60_x_prereg.md")
    repo.commit("prereg", T0)
    repo.data("exp60_trials.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    assert run(repo, not_governed={"docs/experiments/data/gone.json": "x"}) == 1
    assert "no longer exists" in capsys.readouterr().err


# ── M1b PR 4: re-runs are governed ────────────────────────────────────────


def test_rerun_tokens_and_the_parent_chain() -> None:
    assert L.token_of("rerun_exp09_2026-09-24") == "9"
    assert L.token_of("rerun_exp61_2026-10-01") == "61"
    assert L.token_of("0_x") == "0"
    assert L.parent_token("42d53") == "42" and L.parent_token("53b") == "53"
    assert L.parent_chain("53bd53") == ["53b", "53"]
    assert L.rerun_by_name("53b_x_replication_2026-08-28.jsonl") and not L.rerun_by_name("61_results.jsonl")


def _governed(repo: Repo, amendments: str = "") -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md", amendments)


def test_a_rerun_without_its_own_declaration_fails(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.data("rerun_exp61_2026-10-01.jsonl", [T0 + 300])
    repo.commit("data", T0 + 400)
    assert run(repo) == 1
    assert "a re-run needs its own PRE-DATA declaration" in capsys.readouterr().err


def test_a_scoped_amendment_before_the_rerun_passes_and_spares_the_original(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("original data", T0 + 200)
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, for `rerun_exp61_2026-10-01.jsonl`, the re-run.**\n")
    repo.commit("amendment", T0 + 300)
    repo.data("rerun_exp61_2026-10-01.jsonl", [T0 + 400])
    repo.commit("rerun data", T0 + 500)
    assert run(repo) == 0, capsys.readouterr().err


def test_a_scoped_amendment_after_the_rerun_fails(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.commit("prereg", T0)
    repo.data("rerun_exp61_2026-10-01.jsonl", [T0 + 100])
    repo.commit("rerun data", T0 + 200)
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, for `rerun_exp61_2026-10-01.jsonl`, late.**\n")
    repo.commit("amendment", T0 + 300)
    assert run(repo) == 1
    assert "not before the data" in capsys.readouterr().err


def test_an_unscoped_late_amendment_fails_the_original_and_says_to_scope_it(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, meant for a re-run.**\n")
    repo.commit("amendment", T0 + 300)
    assert run(repo) == 1
    assert "scope it" in capsys.readouterr().err


def test_rescoping_after_the_data_does_not_inherit_the_old_time(repo: Repo, capsys) -> None:
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, for `rerun_exp61_a.jsonl`, first.**\n")
    repo.commit("prereg", T0)
    repo.data("rerun_exp61_a.jsonl", [T0 + 100])
    repo.data("rerun_exp61_b.jsonl", [T0 + 150])
    repo.commit("data", T0 + 200)
    _governed(
        repo, "**Amendment 1 — 2026-10-01, PRE-DATA, for `rerun_exp61_a.jsonl`, `rerun_exp61_b.jsonl`, first.**\n"
    )
    repo.commit("rescope", T0 + 300)
    assert run(repo) == 1
    err = capsys.readouterr().err
    assert "rerun_exp61_b.jsonl:" in err and "rerun_exp61_a.jsonl:" not in err


def test_a_post_to_pre_label_flip_dates_from_the_flip(repo: Repo, capsys) -> None:
    _governed(repo, "**Amendment 1 — 2026-10-01, POST-DATA, a note.**\n")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, a note.**\n")
    repo.commit("flip", T0 + 300)
    assert run(repo) == 1
    assert "PRE-DATA amendment 1" in capsys.readouterr().err


def test_amendment_1_is_not_timed_by_amendment_10(repo: Repo, capsys) -> None:
    _governed(repo, "**Amendment 10 — 2026-10-01, POST-DATA, an old one.**\n")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    _governed(
        repo,
        "**Amendment 10 — 2026-10-01, POST-DATA, an old one.**\n\n**Amendment 1 — 2026-10-01, PRE-DATA, late.**\n",
    )
    repo.commit("amendment 1", T0 + 300)
    assert run(repo) == 1


def test_an_informal_pre_data_header_dates_from_when_it_said_so(repo: Repo, capsys) -> None:
    _governed(repo, "**Amendment 1 (pre-data; apparatus).** text\n")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, apparatus.** text\n")
    repo.commit("data + header normalised", T0 + 200)
    assert run(repo) == 0, capsys.readouterr().err


def test_a_scope_for_that_does_not_parse_is_exit_2(repo: Repo, capsys) -> None:
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, for the re-run.**\n")
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("all", T0)
    assert run(repo) == 2
    assert "does not parse" in capsys.readouterr().err


def test_a_scope_naming_a_future_entry_is_a_note_and_a_foreign_one_fails(repo: Repo, capsys) -> None:
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, for `rerun_exp61_later.jsonl`, planned.**\n")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    assert run(repo) == 0
    assert "(yet)" in capsys.readouterr().out
    repo.data("62_other.jsonl", [T0 + 300])
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, for `62_other.jsonl`, wrong experiment.**\n")
    repo.commit("foreign", T0 + 400)
    assert run(repo) == 1
    assert "not its experiment's data" in capsys.readouterr().err


def test_a_rerun_prereg_governs_only_its_scope(repo: Repo, capsys) -> None:
    """A re-run pre-registration for an experiment must not retroactively govern the original data."""
    _governed(repo)
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("original", T0 + 200)
    repo.write(
        "docs/experiments/protocols/exp61_rerun_preregistration.md",
        "# re-run\n\n**Scope:** `rerun_exp61_2026-10-01.jsonl`\n\ngate copied\n",
    )
    repo.commit("rerun prereg", T0 + 300)
    repo.data("rerun_exp61_2026-10-01.jsonl", [T0 + 400])
    repo.commit("rerun data", T0 + 500)
    assert run(repo) == 0, capsys.readouterr().err


def test_an_explicit_rerun_of_an_unregistered_experiment_fails_unless_listed(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.data("rerun_exp9_2026-10-01.jsonl", [T0 + 100])
    repo.data("9_replication.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    assert run(repo) == 1
    err = capsys.readouterr().err
    assert "rerun_exp9_2026-10-01.jsonl" in err and "9_replication" not in err  # a word alone: out of scope
    assert run(repo, ungoverned_reruns={"docs/experiments/data/rerun_exp9_2026-10-01.jsonl": "why"}) == 0


def test_a_rerun_needs_ts_on_every_session_report(repo: Repo, capsys) -> None:
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, for `rerun_exp61_s`, the re-run.**\n")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.write("docs/experiments/data/rerun_exp61_s/a/report.json", json.dumps({"ts": T0 + 300}))
    repo.write("docs/experiments/data/rerun_exp61_s/b/report.json", json.dumps({"finish_reason": "completed"}))
    repo.write("docs/experiments/data/rerun_exp61_s/a/aut_nac.json", json.dumps({"ts": T0 - 9999}))
    repo.commit("data", T0 + 400)
    assert run(repo) == 1
    err = capsys.readouterr().err
    assert "carries `ts` on every record" in err and "not before the data" not in err  # aut_*.json is not a record


def test_the_echo_names_this_entry_in_the_paragraph(repo: Repo, capsys) -> None:
    repo.result_doc(
        "61", "exp61_preregistration.md", "\n`data/1961_x.jsonl` ran with `allow_dirty: true`.\n\n`61_x.jsonl` too.\n"
    )
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.data("61_x.jsonl", [T0 + 100], {"allow_dirty": True})
    repo.commit("data", T0 + 200)
    assert run(repo) == 1  # 1961_x contains 61_x but is a different entry; 61_x's own paragraph never says allow_dirty
    assert "names the entry" in capsys.readouterr().err


def test_an_rb_section_declares_a_rerun(repo: Repo, capsys) -> None:
    _governed(repo, "")
    repo.write(
        "docs/experiments/protocols/exp61_preregistration.md",
        "# prereg\n\n- **RB-1 — new platform.** Data: `docs/experiments/data/61_platform/`.\n- **Next bullet**\n",
    )
    repo.commit("prereg", T0)
    repo.write("docs/experiments/data/61_platform/rows.jsonl", json.dumps({"ts": T0 + 100}) + "\n")
    repo.commit("data", T0 + 200)
    assert run(repo) == 1
    assert "a re-run needs its own PRE-DATA declaration" in capsys.readouterr().err


def test_the_exception_lists_are_frozen(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.data("61_results.jsonl", [T0 - 100])
    repo.write(  # the ref's copy of the lint already lists the entry (the shrink-only check passes)
        "scripts/lint_prereg_precedes_data.py",
        'EXCEPTIONS_FROZEN = 1\nGRANDFATHERED = {"docs/experiments/data/61_results.jsonl": "x"}\n',
    )
    repo.commit("squash", T0)
    gf = {"docs/experiments/data/61_results.jsonl": "the incident"}
    assert run(repo, grandfathered=gf, frozen_at=T0 + 1) == 0  # committed before the freeze
    assert run(repo, grandfathered=gf, frozen_at=T0) == 1  # committed at/after the freeze
    assert "before the freeze" in capsys.readouterr().err


def test_the_exception_lists_only_shrink_against_the_ref(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.data("61_results.jsonl", [T0 - 100])
    repo.write("scripts/lint_prereg_precedes_data.py", "EXCEPTIONS_FROZEN = 1\nGRANDFATHERED = {}\n")
    repo.commit("squash", T0)
    gf = {"docs/experiments/data/61_results.jsonl": "the incident"}
    assert run(repo, grandfathered=gf, frozen_at=T0 + 1) == 1
    assert "added to GRANDFATHERED after the freeze" in capsys.readouterr().err


def test_classify_is_the_json_surface(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.data("rerun_exp9_x.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    assert run(repo, ungoverned_reruns={"docs/experiments/data/rerun_exp9_x.jsonl": "why"}, as_json=True) == 0
    doc = json.loads(capsys.readouterr().out)
    assert set(doc) == {"_format_version", "entries", "failures"} and doc["_format_version"] == "1.0"
    assert {r["status"] for r in doc["entries"]} <= set(L.STATUSES)
    rows = {r["entry"].split("/")[-1]: r for r in doc["entries"]}
    assert rows["61_results.jsonl"]["status"] == "PASS" and rows["61_results.jsonl"]["rerun"] is False
    assert rows["rerun_exp9_x.jsonl"]["status"] == "UNGOVERNED_RERUN" and rows["rerun_exp9_x.jsonl"]["rerun"] is True


# ── M1b PR 4 review-round folds ──────────────────────────────────────────


def test_a_scope_line_in_an_original_prereg_fails_and_governs_nothing_away(repo: Repo, capsys) -> None:
    _governed(repo, "**Scope:** `rerun_exp61_2026-10-01`\n")
    repo.data("61_results.jsonl", [T0 - 100])  # data before its prereg: must still FAIL, not go out of scope
    repo.commit("squash", T0)
    assert run(repo) == 1
    err = capsys.readouterr().err
    assert "belongs only in a re-run pre-registration" in err and "61_results.jsonl:" in err


def test_a_rerun_prereg_names_its_entries(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.write("docs/experiments/protocols/exp61_rerun_x_preregistration.md", "# re-run\n\nno scope\n")
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("all", T0)
    assert run(repo) == 1
    assert "names its entries in a `**Scope:**` line" in capsys.readouterr().err


def test_a_foreign_scope_line_fails(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.write(
        "docs/experiments/protocols/exp10_rerun_x_preregistration.md", "# re-run\n\n**Scope:** `rerun_exp61_x.jsonl`\n"
    )
    repo.commit("preregs", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.data("rerun_exp61_x.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    assert run(repo) == 1
    assert "not its experiment's data" in capsys.readouterr().err


@pytest.mark.parametrize("middle", ["", "**Amendment 1 — 2026-10-01, POST-DATA, withdrawn.**\n"])
def test_a_declaration_that_lapses_dates_from_its_return(repo: Repo, capsys, middle: str) -> None:
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, early.**\n")
    repo.commit("prereg", T0)
    _governed(repo, middle)
    repo.commit("lapse", T0 + 50)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, early.**\n")
    repo.commit("return", T0 + 300)
    assert run(repo) == 1
    assert "PRE-DATA amendment 1" in capsys.readouterr().err


def test_an_oxford_comma_scope_names_every_entry(repo: Repo, capsys) -> None:
    """Ordinary entries named in a late scoped amendment are each judged: one dropped by the parser would be
    silently never judged (a re-run would still fail for lacking a declaration, so these are not re-runs)."""
    _governed(repo)
    repo.commit("prereg", T0)
    for e in ("61_a.jsonl", "61_b.jsonl", "61_c.jsonl"):
        repo.data(e, [T0 + 100])
    repo.commit("data", T0 + 200)
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, for `61_a.jsonl`, `61_b.jsonl`, and `61_c.jsonl`, late.**\n")
    repo.commit("late", T0 + 300)
    assert run(repo) == 1
    err = capsys.readouterr().err
    assert all(f"61_{x}.jsonl:" in err for x in "abc"), err


@pytest.mark.parametrize("written", ["`rerun_exp61_d/`", "`docs/experiments/data/rerun_exp61_d`", "`rerun_exp61\n_d`"])
def test_a_scope_name_is_normalised(repo: Repo, capsys, written: str) -> None:
    _governed(repo)
    repo.commit("prereg", T0)
    repo.write("docs/experiments/data/rerun_exp61_d/rows.jsonl", json.dumps({"ts": T0 + 100}) + "\n")
    repo.commit("data", T0 + 200)
    _governed(repo, f"**Amendment 1 — 2026-10-01, PRE-DATA, for {written}, late.**\n")
    repo.commit("late", T0 + 300)
    assert run(repo) == 1  # judged (and late), never a silent "(yet)" note
    assert "rerun_exp61_d:" in capsys.readouterr().err


def test_a_scope_name_that_is_a_nested_path_is_exit_2(repo: Repo, capsys) -> None:
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, for `rerun_exp61_d/sub.jsonl`, x.**\n")
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("all", T0)
    assert run(repo) == 2
    assert "not a top-level name" in capsys.readouterr().err


def test_a_prose_bold_line_in_history_declares_nothing(repo: Repo, capsys) -> None:
    _governed(repo, "**Amendment 1 is drafted as pre-data but not in force yet.** notes\n")
    repo.commit("prose", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    _governed(repo, "**Amendment 1 — 2026-10-01, PRE-DATA, in force.**\n")
    repo.commit("real", T0 + 300)
    assert run(repo) == 1


def test_new_data_inside_a_listed_path_is_not_excused(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.commit("prereg", T0)
    repo.data("rerun_exp9_x.jsonl", [T0 + 100])
    repo.write(
        "scripts/lint_prereg_precedes_data.py",
        'EXCEPTIONS_FROZEN = 1\nUNGOVERNED_RERUNS = {"docs/experiments/data/rerun_exp9_x.jsonl": "x"}\n',
    )
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    listed = {"docs/experiments/data/rerun_exp9_x.jsonl": "x"}
    assert run(repo, ungoverned_reruns=listed, frozen_at=T0 + 250) == 0
    repo.data("rerun_exp9_x.jsonl", [T0 + 100, T0 + 400])  # rows appended after the freeze
    repo.commit("append", T0 + 500)
    assert run(repo, ungoverned_reruns=listed, frozen_at=T0 + 250) == 1
    assert "has changed since" in capsys.readouterr().err


def test_rerun_words_need_a_boundary() -> None:
    assert not L.rerun_by_name("61_prerun_calibration.jsonl") and not L.rerun_by_name("61_unreplicated.jsonl")
    assert L.rerun_by_name("61_replication.jsonl")


def test_the_echo_does_not_accept_a_longer_sibling_name() -> None:
    assert L._names_entry("see `data/61_x.jsonl` with allow_dirty", "61_x") == []
    assert L._names_entry("see `data/61_x/`, allow_dirty.", "61_x")


def _shallow() -> bool:
    out = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"], cwd=L.REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    return out != "false"


@pytest.mark.skipif(
    _shallow(),
    reason="needs full history; the unit-test job checks out depth 1. The lint job (fetch-depth 0) runs the lint, "
    "whose stale-list checks fail on any drift of the nine listed re-runs; this pin adds the word-only entries",
)
def test_the_real_repo_classifies_its_reruns_as_decided() -> None:
    doc = L.classify_all(L.REPO_ROOT)
    by = {r["entry"].split("/")[-1]: r for r in doc["entries"]}
    for name in (
        "52d53_phaseB_embodied.jsonl",
        "53d53_cross_context_readout.jsonl",
        "53d53_phase2_aborted_run.jsonl",
        "53b_cross_context_readout_replication_2026-08-28.jsonl",
        "exp56_rebaseline_1204",
    ):
        assert by[name]["status"] == "GRANDFATHERED" and by[name]["rerun"], name
    for name in (
        "42d53_results.jsonl",
        "42d53_results_gateoff.jsonl",
        "rerun_exp09_2026-09-24",
        "rerun_exp10_2026-09-27",
    ):
        assert by[name]["status"] == "UNGOVERNED_RERUN", name
    for name in (
        "45d_magnitude_replication.jsonl",
        "48_rebaseline_v4.jsonl",
        "selection_dynamics_rebaseline_2026-09-03.json",
    ):
        assert by[name]["status"] == "OUT_OF_SCOPE", name


def test_a_ref_without_the_lint_fails_the_lists_closed(repo: Repo, capsys) -> None:
    _governed(repo)
    repo.data("61_results.jsonl", [T0 - 100])
    repo.commit("squash", T0)  # no scripts/lint_prereg_precedes_data.py on the ref (moved or renamed)
    assert run(repo, grandfathered={"docs/experiments/data/61_results.jsonl": "x"}, frozen_at=T0 + 1) == 1
    assert "added to GRANDFATHERED after the freeze" in capsys.readouterr().err


# ── M1b PR 5b-2: `kind: "prereg"` clauses of the exceptions file (EXCEPTED) ─────────────────────────────────


def _clause(repo: Repo, name: str, **kw) -> dict:
    import hashlib

    data = (repo.root / "docs/experiments/data" / name).read_bytes()
    return {"id": "p1", "kind": "prereg", "path": f"docs/experiments/data/{name}",
            "sha256": hashlib.sha256(data).hexdigest(), "owner": "o", "reason": "owner-named", "date": "2026-10-01",
            **kw}  # fmt: skip


def _late_data(repo: Repo) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.data("61_results.jsonl", [T0 - 3600])  # before the prereg: a substantive FAIL
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg + data", T0)


def test_a_pinned_exceptions_clause_excuses_a_substantive_failure(repo: Repo, capsys) -> None:
    _late_data(repo)
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([_clause(repo, "61_results.jsonl")]))
    repo.commit("clause", T0 + 10)
    assert run(repo) == 0
    assert "EXCEPTED (exceptions file; still failing)" in capsys.readouterr().out


def test_a_clause_pinned_to_other_bytes_is_inert(repo: Repo, capsys) -> None:
    _late_data(repo)
    repo.write(
        "docs/experiments/evidence_exceptions.json", json.dumps([_clause(repo, "61_results.jsonl", sha256="0" * 64)])
    )
    repo.commit("clause", T0 + 10)
    assert run(repo) == 1
    assert "does not pin the entry at HEAD (inert)" in capsys.readouterr().out


def test_a_clause_never_excuses_list_hygiene(repo: Repo, capsys) -> None:
    """A GRANDFATHERED listing of an entry that now passes is a list-hygiene FAIL; a clause cannot excuse it."""
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 3600])
    repo.commit("data", T0 + 7200)
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([_clause(repo, "61_results.jsonl")]))
    repo.commit("clause", T0 + 7300)
    assert run(repo, grandfathered={"docs/experiments/data/61_results.jsonl": "x"}) == 1
    assert "listed as GRANDFATHERED but now PASSES" in capsys.readouterr().err


def test_a_clause_on_a_passing_entry_is_noted_stale(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 3600])
    repo.commit("data", T0 + 7200)
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([_clause(repo, "61_results.jsonl")]))
    repo.commit("clause", T0 + 7300)
    assert run(repo) == 0
    assert "clause is STALE" in capsys.readouterr().out


def test_a_directory_entry_is_pinned_by_its_tree_id(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.data("61_runs/a.jsonl", [T0 - 3600])
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg + data", T0)
    tree = _git(repo.root, "rev-parse", "HEAD:docs/experiments/data/61_runs").strip()
    clause = {"id": "p1", "kind": "prereg", "path": "docs/experiments/data/61_runs", "tree": tree, "owner": "o",
              "reason": "r", "date": "2026-10-01"}  # fmt: skip
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([clause]))
    repo.commit("clause", T0 + 10)
    assert run(repo) == 0
    repo.data("61_runs/b.jsonl", [T0 - 3500])  # a file added later changes the tree: the clause goes inert
    repo.commit("more data", T0 + 20)
    assert run(repo) == 1


def test_a_clause_excuses_an_ungoverned_rerun(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.data("rerun_exp99_x.jsonl", [T0 + 3600])  # a re-run whose experiment has no pre-registration
    repo.commit("data", T0 + 7200)
    assert run(repo) == 1
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([_clause(repo, "rerun_exp99_x.jsonl")]))
    repo.commit("clause", T0 + 7300)
    assert run(repo) == 0
    assert "EXCEPTED (exceptions file) — an ungoverned re-run" in capsys.readouterr().out


def test_a_clause_is_inert_while_the_working_copy_differs(repo: Repo, capsys) -> None:
    _late_data(repo)
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([_clause(repo, "61_results.jsonl")]))
    repo.commit("clause", T0 + 10)
    repo.data("61_results.jsonl", [T0 - 3600, T0 - 3500])  # uncommitted change to the excused entry
    assert run(repo) == 1
    assert "working copy differs from HEAD" in capsys.readouterr().out


def test_a_clause_never_excuses_a_record_form_failure(repo: Repo, capsys) -> None:
    """Late data whose records also carry a naive ISO ts: the clause would excuse the lateness, never the ts."""
    repo.result_doc("61", "exp61_preregistration.md")
    repo.write("docs/experiments/data/61_results.jsonl", json.dumps({"ts": "2026-08-10T11:53:49", "event": "x"}) + "\n")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg + data", T0)
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([_clause(repo, "61_results.jsonl")]))
    repo.commit("clause", T0 + 10)
    assert run(repo) == 1
    assert "naive ISO-8601" in capsys.readouterr().err


def test_a_malformed_clause_never_acts(repo: Repo, capsys) -> None:
    _late_data(repo)
    bad = {k: v for k, v in _clause(repo, "61_results.jsonl").items() if k != "owner"}
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([bad]))
    repo.commit("clause", T0 + 10)
    assert run(repo) == 1
    assert "is malformed (lacks a required field): it never acts" in capsys.readouterr().out


def test_a_clause_not_yet_on_main_never_acts(repo: Repo) -> None:
    """Clauses are read from the ref (main): one on a branch ahead of main is reviewed before it acts."""
    _late_data(repo)
    _git(repo.root, "checkout", "-q", "-b", "feat")
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([_clause(repo, "61_results.jsonl")]))
    repo.commit("clause on a branch", T0 + 10)
    assert run(repo) == 1


def test_a_clause_on_an_ungoverned_rerun_still_judges_the_records_form(repo: Repo, capsys) -> None:
    repo.result_doc("61", "exp61_preregistration.md")
    repo.prereg("exp61_preregistration.md")
    repo.commit("prereg", T0)
    repo.write(
        "docs/experiments/data/rerun_exp99_x.jsonl", json.dumps({"ts": "2026-08-10T11:53:49", "event": "x"}) + "\n"
    )
    repo.commit("data", T0 + 7200)
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([_clause(repo, "rerun_exp99_x.jsonl")]))
    repo.commit("clause", T0 + 7300)
    assert run(repo) == 1
    out = capsys.readouterr()
    assert "naive ISO-8601" in out.err and "clause does not apply" in out.out


@pytest.mark.parametrize(
    "header",
    ["**Amendment 1 — 2026-10-01, apparatus, PRE-DATA, x.**\n", "**Amendment 1 — Oct 1, 2026, PRE-DATA, x.**\n"],
)
def test_a_header_todays_grammar_accepts_is_read_the_same_in_history(repo: Repo, capsys, header: str) -> None:
    """#1014: the lenient historical reading wanted the class as the FIRST clause and a comma-free date, so these
    current-grammar headers were never matched in history: a false "not on main in its current form"."""
    _governed(repo, header)
    repo.commit("prereg", T0)
    repo.data("61_results.jsonl", [T0 + 100])
    repo.commit("data", T0 + 200)
    assert run(repo) == 0, capsys.readouterr().err


def test_an_allowance_on_an_excepted_ungoverned_rerun_names_the_fix(repo: Repo, capsys) -> None:
    """#1037: an ungoverned re-run has no result doc of its own, so "no result doc names the entry" named no fix."""
    repo.data("rerun_exp9_x.jsonl", [T0 + 100], {"allow_dirty": True})
    repo.commit("data", T0 + 200)
    repo.write("docs/experiments/evidence_exceptions.json", json.dumps([_clause(repo, "rerun_exp9_x.jsonl")]))
    repo.commit("clause", T0 + 300)
    assert run(repo) == 1
    assert "land a re-run pre-registration" in capsys.readouterr().err
