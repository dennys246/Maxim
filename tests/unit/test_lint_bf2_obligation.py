"""The BF2 obligation lint (scripts/lint_bf2_obligation.py; spec docs/plans/archive/baseline_floors_b1_evidence.md §3):
a floor-matched B1 verdict obliges its row below PARTIAL or at PARTIAL (narrow) re-dated; verdicts are append-only by
(data_sha256, value); the data time is read only from sha-bound bytes; only an established FLOORS_BEATEN clears.
Driven on the evidence gate's temp-repo rig, the obliged row swapped to the rig's T1-1."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import _evidence_records as R  # noqa: E402
import lint_bf2_obligation as B  # noqa: E402
from tests.unit.test_lint_evidence_gate import (  # noqa: E402
    DATA,
    Rig,
    _prov,
    _row,
    ledger,
    t1,
    t3,
)

DAY = "2026-10-02"
TS = dt.datetime(2026, 10, 2, 12, tzinfo=dt.timezone.utc).timestamp()
OBLIGING = frozenset({"FLOOR_MATCHED_Q", "FLOOR_MATCHED_CHANCE"})


TABLE = {"b1_verdict": {"obliges": {"row": "T1-1", "values": sorted(OBLIGING), "clears": ["FLOORS_BEATEN"]}}}


@pytest.fixture
def rig(tmp_path) -> Rig:
    r = Rig(tmp_path / "repo")
    r.base_sha = "0" * 40  # stamped into rows written before the first commit (provenance is not judged here)
    return r


def _ledger(status: str) -> str:
    return ledger([t1("T1-1", status)], [t3("T3-9", "**Status: STALE 2026-09-30**.")])


EARNED = "**Status: EARNED 2026-09-16**."
NARROW = f"**Status: PARTIAL {DAY}** (narrow: floor matched)."


def _verdict(rig: Rig, name: str, value: str, *, ts: float = TS, data_name: str | None = None) -> str:
    data_name = data_name or f"{name}.jsonl"
    data_rel = f"{DATA}/{data_name}"
    if not (rig.root / data_rel).exists():
        rig.write(data_rel, json.dumps(_row(rig, ts=ts)) + "\n")
    data = (rig.root / data_rel).read_bytes()
    rec = {"record_kind": "verdict", "kind": "b1_verdict", "verdict": value, "mock": False, "data": data_rel,
           "data_sha256": hashlib.sha256(data).hexdigest(), "scope": {"all_rows": True}, "provenance": _prov(rig)}  # fmt: skip
    rig.write(f"{DATA}/{name}.json", json.dumps(rec))
    return f"{DATA}/{name}.json"


def _ctx(rig: Rig) -> R.Ctx:
    return R.Ctx(repo=R.Repo(rig.root), base=rig.base_sha, ref="HEAD", legacy={}, prereg={DATA: "PASS"}, table=TABLE,
                 retired={})  # fmt: skip


def _problems(rig: Rig) -> list[str]:
    return B.obligation_problems(R.Repo(rig.root), rig.base_sha, ctx=_ctx(rig))


def _scenario(rig: Rig, head_status: str, *, value: str = "FLOOR_MATCHED_Q") -> list[str]:
    rig.base(_ledger(EARNED))
    _verdict(rig, "v1", value)
    rig.head(_ledger(head_status))
    return _problems(rig)


def test_no_b1_verdict_is_clean(rig) -> None:
    rig.base(_ledger(EARNED))
    rig.head(_ledger("**Status: EARNED 2026-09-17**."))
    assert _problems(rig) == []


def test_a_floor_matched_verdict_with_the_row_left_earned_fails(rig) -> None:
    (p,) = _scenario(rig, EARNED)
    assert "T1-1 is EARNED" in p and "owner rule BF2" in p


@pytest.mark.parametrize(
    "status",
    [NARROW, "**Status: PARTIAL 2026-10-05** (narrow: floor matched).", "**Status: STALE 2026-10-02**."],
)
def test_the_obligation_is_met_at_partial_narrow_redated_or_below(rig, status) -> None:
    assert _scenario(rig, status) == []


@pytest.mark.parametrize(
    "status",
    [
        "**Status: PARTIAL 2026-10-01** (narrow: floor matched).",  # dated before the verdict's data
        f"**Status: PARTIAL {DAY}**.",  # unqualified PARTIAL is the wider claim
        f"**Status: PARTIAL {DAY}** (rung A).",  # another scope
        f"**Status: RE-VALIDATED-BY-TESTS {DAY}**.",  # rank 2 but not PARTIAL
    ],
)
def test_the_obligation_is_not_met_otherwise(rig, status) -> None:
    assert _scenario(rig, status)


@pytest.mark.parametrize("value", ["INCOMPLETE", "FLOORS_BEATEN"])  # the beaten one is not established on the rig
def test_an_incomplete_or_unestablished_beaten_verdict_obliges_nothing(rig, value) -> None:
    assert _scenario(rig, EARNED, value=value) == []


# ── append-only by (data_sha256, value) ───────────────────────────────────────────────────────────────────


def _on_main(rig: Rig, value: str = "FLOOR_MATCHED_Q") -> str:
    path = _verdict(rig, "v1", value)
    rig.base(_ledger(NARROW))
    return path


def test_deleting_a_verdict_on_main_fails(rig) -> None:
    path = _on_main(rig)
    (rig.root / path).unlink()
    rig.head(_ledger(EARNED))
    assert any("append-only" in p for p in _problems(rig)), _problems(rig)


def test_rewriting_a_verdict_value_fails(rig) -> None:
    path = _on_main(rig)
    rec = json.loads((rig.root / path).read_text())
    rig.write(path, json.dumps({**rec, "verdict": "INCOMPLETE"}))
    rig.head(_ledger(EARNED))
    assert any("append-only" in p for p in _problems(rig)), _problems(rig)


def test_moving_a_verdict_keeps_its_identity(rig) -> None:
    path = _on_main(rig)
    moved = f"{DATA}/sub/v1.json"
    rig.write(moved, (rig.root / path).read_bytes())
    (rig.root / path).unlink()
    rig.head(_ledger(NARROW))
    assert _problems(rig) == []


def test_two_values_over_one_data_file_fail(rig) -> None:
    rig.base(_ledger(EARNED))
    _verdict(rig, "v1", "FLOOR_MATCHED_Q")
    _verdict(rig, "v2", "FLOOR_MATCHED_CHANCE", data_name="v1.jsonl")
    rig.head(_ledger(NARROW))
    assert any("differing verdicts" in p for p in _problems(rig)), _problems(rig)


# ── the data time: sha-bound bytes only; ties fail ────────────────────────────────────────────────────────


def test_data_whose_bytes_no_longer_match_the_verdict_is_unreadable(rig) -> None:
    rig.base(_ledger(EARNED))
    _verdict(rig, "v1", "FLOOR_MATCHED_Q")
    rig.write(f"{DATA}/v1.jsonl", json.dumps(_row(rig, ts=TS - 86400 * 30)) + "\n")  # backdated after stamping
    rig.head(_ledger(NARROW))
    assert any("cannot be read" in p and "sha256" in p for p in _problems(rig)), _problems(rig)


def test_two_data_files_tied_for_newest_fail(rig) -> None:
    rig.base(_ledger(EARNED))
    _verdict(rig, "v1", "FLOOR_MATCHED_Q")
    rig.write(f"{DATA}/v2.jsonl", json.dumps(_row(rig, ts=TS, seed=2)) + "\n")
    _verdict(rig, "v2", "FLOOR_MATCHED_CHANCE")
    rig.head(_ledger(NARROW))
    assert any("tie for the newest" in p for p in _problems(rig)), _problems(rig)


# ── clearing: only an ESTABLISHED, prereg-PASS FLOORS_BEATEN, and only when newer ──────────────────────────


class _Established:
    status, prereg, data_prereg = R.ESTABLISHED, "PASS", "PASS"


def test_a_newer_established_beaten_verdict_clears_and_an_older_one_does_not(rig, monkeypatch) -> None:
    monkeypatch.setattr(B, "judge_entry", lambda path, ctx: _Established())
    rig.base(_ledger(EARNED))
    _verdict(rig, "v1", "FLOOR_MATCHED_Q", ts=TS)
    _verdict(rig, "v2", "FLOORS_BEATEN", ts=TS + 86400)
    rig.head(_ledger(EARNED))
    assert _problems(rig) == []
    rig2 = Rig(rig.root.parent / "repo2")
    rig2.base_sha = "0" * 40
    rig2.base(_ledger(EARNED))
    _verdict(rig2, "v1", "FLOOR_MATCHED_Q", ts=TS + 86400)
    _verdict(rig2, "v2", "FLOORS_BEATEN", ts=TS)
    rig2.head(_ledger(EARNED))
    assert B.obligation_problems(R.Repo(rig2.root), rig2.base_sha, ctx=_ctx(rig2))


def test_the_obligation_and_its_demotion_land_together(rig) -> None:
    """The data PR carries the demotion: with the verdict and the move in one diff the lint passes; the verdict alone
    (the move left for later) fails."""
    rig.base(_ledger(EARNED))
    _verdict(rig, "v1", "FLOOR_MATCHED_Q")
    rig.head(_ledger(NARROW))
    assert _problems(rig) == []
    rig.head(_ledger(EARNED))
    assert _problems(rig)


def test_the_obligation_lives_in_the_pass_table_and_is_pinned_to_the_b1_harness() -> None:
    import lint_evidence_gate as G
    from survival_world import b1_floor_run

    table = json.loads((REPO / "docs/experiments/evidence_pass_table.json").read_text())
    row, obliging, clearing = B.obligations(table)["b1_verdict"]
    entry = table["b1_verdict"]
    assert row == "T1-13" and entry["rows"] == [row]
    assert obliging == set(entry["targets"]["PARTIAL"]) and obliging <= set(b1_floor_run.VERDICTS)
    assert clearing == {"FLOORS_BEATEN"} <= set(b1_floor_run.VERDICTS)
    assert G.obliges_problem(entry) is None
    bad = {**entry, "obliges": {**entry["obliges"], "values": ["EARNED"]}}  # a value its targets never list
    assert G.obliges_problem(bad) is not None
    assert G.obliges_problem({**entry, "obliges": {**entry["obliges"], "clears": ["FLOOR_MATCHED_Q"]}}) is not None


def test_the_obligation_is_read_from_the_table_it_is_given(rig) -> None:
    """The lint takes its obligations from ``ctx.table`` (the gate's MERGE-BASE pass table): with no `obliges` there, a
    floor-matched verdict obliges nothing — so a PR's own table edit cannot loosen (or add) an obligation in that PR."""
    rig.base(_ledger(EARNED))
    _verdict(rig, "v1", "FLOOR_MATCHED_Q")
    rig.head(_ledger(EARNED))
    ctx = _ctx(rig)
    ctx.table = {}
    assert B.obligation_problems(R.Repo(rig.root), rig.base_sha, ctx=ctx) == []
    assert _problems(rig)  # the same tree with the obligation in the base table fails


def test_the_real_repository_is_clean(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)  # locally a missing base skips (0); clean is also 0
    assert B.main([]) == 0


def test_a_newer_unestablished_beaten_verdict_does_not_clear(rig) -> None:
    rig.base(_ledger(EARNED))
    _verdict(rig, "v1", "FLOOR_MATCHED_Q", ts=TS)
    _verdict(rig, "v2", "FLOORS_BEATEN", ts=TS + 86400)  # the rig's judge does not establish it
    rig.head(_ledger(EARNED))
    assert any("owner rule BF2" in p for p in _problems(rig)), _problems(rig)


@pytest.mark.parametrize(
    ("head_obliges", "why"),
    [
        (None, "may not be removed"),
        (
            {"row": "T1-13", "values": ["FLOOR_MATCHED_Q"], "clears": ["FLOORS_BEATEN"]},
            "dropped ['FLOOR_MATCHED_CHANCE']",
        ),
        (
            {
                "row": "T1-13",
                "values": ["FLOOR_MATCHED_Q", "FLOOR_MATCHED_CHANCE"],
                "clears": ["FLOORS_BEATEN", "INCOMPLETE"],
            },
            "added ['INCOMPLETE']",
        ),
        (
            {"row": "T1-1", "values": ["FLOOR_MATCHED_Q", "FLOOR_MATCHED_CHANCE"], "clears": ["FLOORS_BEATEN"]},
            "row 'T1-13'",
        ),
    ],
)
def test_an_obliges_block_on_main_may_only_tighten(head_obliges, why) -> None:
    import lint_evidence_gate as G

    real = json.loads((REPO / "docs/experiments/evidence_pass_table.json").read_text())["b1_verdict"]
    base = {"b1_verdict": real}
    head_entry = {k: v for k, v in real.items() if k != "obliges"}
    if head_obliges is not None:
        head_entry["obliges"] = head_obliges
    (p,) = G.obliges_change_problems(base, {"b1_verdict": head_entry})
    assert why in p, p
    tighter = {**real, "obliges": {**real["obliges"], "clears": []}}  # fewer clearing values: a tightening
    assert G.obliges_change_problems(base, {"b1_verdict": tighter}) == []
    assert G.obliges_change_problems({}, {"b1_verdict": real}) == []  # adding a block is free (inert in its own PR)


def test_an_unhashable_obliging_value_is_a_shape_problem_not_a_crash() -> None:
    import lint_evidence_gate as G

    real = json.loads((REPO / "docs/experiments/evidence_pass_table.json").read_text())["b1_verdict"]
    assert G.obliges_problem({**real, "obliges": {**real["obliges"], "values": [["X"]]}}) is not None
