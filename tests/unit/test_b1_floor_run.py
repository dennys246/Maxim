"""The B1 floor driver (scripts/survival_world/b1_floor_run.py) against its prereg
(docs/experiments/b1_exp60_floor_prereg.md): the frozen order, the verdict, the episode-end and terminal rules, the
rig check, resume, the copied Exp 60 preflight sequence, the `params={}` guard, and the trial itself on the scripted
water bridge (a floor policy armed through the hook, episodes ended in the frozen order, Q learning shown offline)."""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world import b1_floor_run as B  # noqa: E402
from survival_world import floor_policies as FP  # noqa: E402
from survival_world.common import InstrumentError  # noqa: E402

# ── the order ───────────────────────────────────────────────────────────────────────────────────────

PREREG_ORDER = {
    11: "FRWQP",
    12: "RQFPW",
    13: "QPRWF",
    14: "PWQFR",
    15: "WFPRQ",
    16: "PQWRF",
    17: "WPFQR",
    18: "FWRPQ",
}


def test_the_order_is_the_prereg_williams_table():
    assert {int(s): o for s, o in B.FROZEN["order"].items()} == PREREG_ORDER
    letter = {v: k for k, v in B.ARM_LETTER.items()}
    for seed, seq in PREREG_ORDER.items():
        assert B.ORDER[seed] == tuple(letter[c] for c in seq)


def test_square_a_puts_every_arm_once_in_every_position():
    rows = [PREREG_ORDER[s] for s in (11, 12, 13, 14, 15)]
    for pos in range(5):
        assert sorted(r[pos] for r in rows) == sorted("FRQPW")
    for r in rows:
        assert sorted(r) == sorted("FRQPW")


def test_policy_seeds_differ_per_arm_and_seed_and_are_stable():
    seeds = {(a, s): B.policy_seed(a, s) for a in B.ARMS for s in B.SEEDS}
    assert len(set(seeds.values())) == len(seeds)
    assert B.policy_seed("random", 11) == B.policy_seed("random", 11)


# ── episode ends and terminals ──────────────────────────────────────────────────────────────────────


def _s(t, w, h=20.0, o=20.0):
    return {"t": t, "in_water": w, "health": h, "oxygen": o}


@pytest.mark.parametrize(
    "samples,pains,want",
    [
        ([_s(0, True), _s(1, False)], [], "surfaced"),
        ([_s(0, True), _s(1, True)], [], "cap"),
        ([_s(0, False), _s(1, False)], [], "never_submerged"),
        ([_s(0, True), _s(1, True, h=19)], [], "dirty"),
        ([_s(0, True), _s(1, False)], [{"failure_mode": "drive:oxygen"}], "dirty"),
    ],
)
def test_episode_end_reason(samples, pains, want):
    assert B.episode_end_reason(samples, pains=pains) == want


def test_training_terminals_by_bridge_truth():
    st = {}
    assert B.training_terminal(_s(0, True, o=19), st, oxygen_max=12, saturating_pain=False) is None
    assert B.training_terminal(_s(1, True, o=12), st, oxygen_max=12, saturating_pain=False) == "pain"
    st = {}
    B.training_terminal(_s(0, True), st, oxygen_max=12, saturating_pain=False)
    assert B.training_terminal(_s(1, True), st, oxygen_max=12, saturating_pain=True) == "pain"
    assert B.training_terminal(_s(2, False), st, oxygen_max=12, saturating_pain=False) == "surfaced"
    st = {}
    assert B.training_terminal(_s(0, False), st, oxygen_max=12, saturating_pain=False) is None  # not yet submerged
    B.training_terminal(_s(1, True), st, oxygen_max=12, saturating_pain=False)
    assert B.training_terminal(_s(2, True, h=18), st, oxygen_max=12, saturating_pain=False) == "damage"


def test_the_pain_cursor_counts_each_allowlisted_pain_once():
    signals = [{"t": 0, "failure_mode": "drive:oxygen", "intensity": 1.0}]
    cur = B.PainCursor(signals)  # starts at the end: earlier pain is not this policy's
    assert cur() == 0.0
    signals.append({"t": 1, "failure_mode": "drive:oxygen", "intensity": 0.5})
    signals.append({"t": 1, "failure_mode": "drive:food", "intensity": 1.0})
    signals.append({"t": 1, "failure_mode": "drive:health", "intensity": 0.25})
    assert cur() == pytest.approx(0.75)
    assert cur() == 0.0


# ── the rig check ───────────────────────────────────────────────────────────────────────────────────


def test_tick_query_parses_the_rate_capped_by_mspt():
    assert B.parse_tick_query("Target tick rate: 20.0 per second.\nAverage time per tick: 3.1ms") == 20.0
    assert B.parse_tick_query("Target tick rate: 20.0 per second.\nAverage time per tick: 62.5ms") == 16.0
    assert B.parse_tick_query("Unknown command") is None


def test_rig_problems():
    now = 10_000.0
    idle = {"infer_lane": {"in_flight": 0}}
    assert B.rig_problems(None, None, 20.0, now=now) == []  # no proxy on the box: passes
    assert B.rig_problems(idle, {"recent": [{"timestamp": now - 500}]}, 20.0, now=now) == []
    assert B.rig_problems({"infer_lane": {"in_flight": 1}}, {"recent": []}, 20.0, now=now)
    assert B.rig_problems({"infer_lane": {}}, {"recent": []}, 20.0, now=now)  # idle cannot be shown
    assert B.rig_problems(idle, {"recent": [{"timestamp": now - 30}]}, 20.0, now=now)  # the proxy's own field
    assert B.rig_problems(idle, {"recent": [{"ts": now - 500}]}, 20.0, now=now)  # no `timestamp`: not shown idle
    assert B.rig_problems(idle, {"recent": []}, 19.0, now=now)
    assert B.rig_problems(None, None, None, now=now)


# ── resume and attempts ─────────────────────────────────────────────────────────────────────────────


def test_attempts_done_reruns_only_an_instrument_refusal_once():
    rows = [
        {"kind": "session"},
        {"kind": "row", "arm": "fear", "seed": 11, "refusal": None},
        {"kind": "row", "arm": "random", "seed": 11, "refusal": "x", "refusal_cause": "instrument"},
        {"kind": "row", "arm": "q_explore", "seed": 11, "refusal": "x", "refusal_cause": "behavioural"},
        {"kind": "row", "arm": "q_explore_raw", "seed": 11, "refusal": "x", "refusal_cause": "instrument"},
        {"kind": "row", "arm": "q_explore_raw", "seed": 11, "refusal": "y", "refusal_cause": "instrument"},
    ]
    done = B.attempts_done(rows)
    assert done[("fear", 11)]["done"]
    assert not done[("random", 11)]["done"] and done[("random", 11)]["attempts"] == 1
    assert done[("q_explore", 11)]["done"]
    assert done[("q_explore_raw", 11)]["done"] and done[("q_explore_raw", 11)]["attempts"] == 2


# ── the verdict ─────────────────────────────────────────────────────────────────────────────────────


def _row(arm, seed, surfaced, *, attempt=1, refusal=None, cause=None, campaign="c1"):
    pl = {"surfaced": surfaced, "latency_s": 2.0 if surfaced else None, "dirty": False, "censored": not surfaced}
    return {
        "kind": "row",
        "campaign_id": campaign,
        "run_id": f"run-{arm}",
        "arm": arm,
        "seed": seed,
        "attempt": attempt,
        "refusal": refusal,
        "refusal_cause": cause,
        "apparatus_record": "a.json",
        "apparatus_record_sha256": "abc",
        "pre": {"placements": [pl]},
        "post": {"placements": [pl], "shore_roam": {"calls": []}},
    }


def _start(arm, seed, attempt=1, campaign="c1"):
    return {
        "kind": "attempt_start",
        "campaign_id": campaign,
        "run_id": f"run-{arm}",
        "arm": arm,
        "seed": seed,
        "attempt": attempt,
    }


def _attempt(arm, seed, surfaced, **kw):
    """An attempt as the harness writes it: its attempt_start header, then its row."""
    row = _row(arm, seed, surfaced, **kw)
    return [_start(arm, seed, row["attempt"], row["campaign_id"]), row]


def _campaign(rates):
    rows = [{"kind": "session", "campaign_id": "c1"}]
    for arm, hits in rates.items():
        for i, s in enumerate(B.SEEDS):
            rows += _attempt(arm, s, i < hits)
    return rows


def _drop_key(rows, arm, seed):
    return [r for r in rows if not (r.get("arm") == arm and r.get("seed") == seed)]


def test_fear_beating_both_floors_is_floors_beaten():
    v = B.compute_verdict(_campaign({"fear": 8, "random": 0, "q_explore": 0, "q_explore_prior": 8}))
    assert v["verdict"] == "FLOORS_BEATEN" and v["holm_shown"] == {"H1": True, "H2": True}
    assert v["first_post_rate"]["q_explore_prior"] == 1.0 and v["n_clean"]["q_explore_raw"] == 0


def test_a_beaten_floor_that_never_tried_escape_is_annotated_under_explored():
    rows = _campaign({"fear": 8, "random": 0, "q_explore": 0})
    assert B.compute_verdict(rows)["annotation"] == "floor under-explored"  # no coverage recorded = not shown tried
    for r in rows:
        if r.get("arm") == "q_explore":
            r["key_coverage"] = {"first_select_key_tried": True}
    v = B.compute_verdict(rows)
    assert v["verdict"] == "FLOORS_BEATEN" and v["annotation"] == "floor: beaten"
    assert v["q_explore_under_explored_seeds"] == []


def test_a_q_tie_is_floor_matched_q():
    v = B.compute_verdict(_campaign({"fear": 8, "random": 1, "q_explore": 8}))
    assert v["verdict"] == "FLOOR_MATCHED_Q" and v["tests"]["H2"]["p_one_sided"] == 1.0


def test_a_chance_tie_is_floor_matched_chance():
    v = B.compute_verdict(_campaign({"fear": 6, "random": 5, "q_explore": 0}))
    assert v["verdict"] == "FLOOR_MATCHED_CHANCE"


def test_holm_is_step_down():
    assert B.holm({"H1": 0.01, "H2": 0.04}, 0.05) == {"H1": True, "H2": True}
    assert B.holm({"H1": 0.03, "H2": 0.04}, 0.05) == {"H1": False, "H2": False}  # 0.03 > 0.025 stops it
    assert B.holm({"H1": 0.001, "H2": None}, 0.05) == {"H1": True, "H2": False}


def test_a_collapsed_fear_is_incomplete_not_a_floor_match():
    v = B.compute_verdict(_campaign({"fear": 5, "random": 0, "q_explore": 0}))
    assert v["verdict"] == "INCOMPLETE" and v["incomplete_cause"] == "fear_sanity_gate"


def test_a_missing_gated_seed_is_incomplete():
    rows = _campaign({"fear": 8, "random": 0, "q_explore": 0})
    v = B.compute_verdict(_drop_key(rows, "random", 18))
    assert v["verdict"] == "INCOMPLETE" and v["incomplete_cause"] == "too_few_clean_seeds"


def test_an_instrument_rerun_counts_and_a_behavioural_or_unrecorded_rerun_refuses():
    rows = _campaign({"fear": 8, "random": 0, "q_explore": 0})
    first = _row("random", 11, False, refusal="rig", cause="instrument")
    rows = _drop_key(rows, "random", 11) + [_start("random", 11, 1), first] + _attempt("random", 11, False, attempt=2)
    v = B.compute_verdict(rows)
    assert v["verdict"] == "FLOORS_BEATEN" and v["attempts"]["random/11"] == 2
    i = rows.index(first)
    rows[i] = {**first, "refusal_cause": "behavioural"}
    assert B.compute_verdict(rows)["incomplete_cause"] == "rerun_after_behavioural_refusal"
    rows[i] = {**first, "refusal": B.UNRECORDED_REFUSAL, "refusal_cause": "unrecorded"}
    assert B.compute_verdict(rows)["incomplete_cause"] == "rerun_after_unrecorded_attempt"
    rows[i] = {**first, "refusal": None}
    assert B.compute_verdict(rows)["incomplete_cause"] == "rerun_after_clean_row"


# ── attempt_start pairing and the unrecorded cause (approach note §2 item 6, D2) ─────────────────────


def test_a_row_without_its_attempt_start_is_incomplete():
    rows = _campaign({"fear": 8, "random": 0, "q_explore": 0})
    rows.remove(_start("random", 13))
    v = B.compute_verdict(rows)
    assert v["verdict"] == "INCOMPLETE" and v["incomplete_cause"] == "attempt_start_mismatch"
    assert "'random', 13, 1" in v["reason"]


def test_an_attempt_start_without_its_row_is_incomplete():
    """A deleted row leaves its start behind: never read as an absent (or present) key."""
    rows = _campaign({"fear": 8, "random": 0, "q_explore": 0, "q_explore_prior": 1})
    rows = [r for r in rows if not (r.get("kind") == "row" and r["arm"] == "q_explore_prior" and r["seed"] == 11)]
    v = B.compute_verdict(rows)
    assert v["verdict"] == "INCOMPLETE" and v["incomplete_cause"] == "attempt_start_mismatch"


def test_a_duplicated_attempt_start_is_incomplete():
    rows = _campaign({"fear": 8, "random": 0, "q_explore": 0})
    rows.append(_start("fear", 12))
    assert B.compute_verdict(rows)["incomplete_cause"] == "attempt_start_mismatch"


def test_attempt_starts_count_for_the_one_campaign_check():
    rows = _campaign({"fear": 8, "random": 0, "q_explore": 0})
    rows.append(_start("fear", 12, 2, campaign="c2"))
    assert B.compute_verdict(rows)["incomplete_cause"] == "mixed_campaigns"


def test_an_unrecorded_final_attempt_is_a_refused_key():
    unrec = {"refusal": B.UNRECORDED_REFUSAL, "cause": "unrecorded"}
    rows = _campaign({"fear": 8, "random": 0, "q_explore": 0, "q_explore_prior": 0})
    rows = _drop_key(rows, "q_explore_prior", 14) + _attempt("q_explore_prior", 14, False, **unrec)
    v = B.compute_verdict(rows)  # a reported arm: the verdict stands, the refusal is listed
    assert v["verdict"] == "FLOORS_BEATEN" and v["n_clean"]["q_explore_prior"] == 7
    assert [(r["arm"], r["cause"]) for r in v["refused"]] == [("q_explore_prior", "unrecorded")]
    rows = _drop_key(rows, "fear", 14) + _attempt("fear", 14, False, **unrec)
    assert B.compute_verdict(rows)["incomplete_cause"] == "too_few_clean_seeds"  # a gated arm lost a key


def test_the_refusal_cause_vocabulary():
    assert B.REFUSAL_CAUSES == (None, "instrument", "behavioural", "unrecorded")
    assert B.FROZEN["max_attempts"] == 2


@pytest.mark.parametrize("cause", B.REFUSAL_CAUSES)
@pytest.mark.parametrize("attempt", [1, 2])
def test_key_done_is_the_one_rule_for_resume_and_the_live_loop(cause, attempt):
    row = {
        "kind": "row",
        "arm": "fear",
        "seed": 11,
        "attempt": attempt,
        "refusal": cause and "x",
        "refusal_cause": cause,
    }
    want = cause != "instrument" or attempt >= B.FROZEN["max_attempts"]
    assert B.key_done(row, attempt) is want
    assert B._after_attempt(row, attempt) == {"attempts": attempt, "done": want}
    prior = [{**row, "attempt": 1, "refusal": "x", "refusal_cause": "instrument"}] if attempt == 2 else []
    assert B.attempts_done([*prior, row])[("fear", 11)] == {"attempts": attempt, "done": want}


def test_attempts_done_treats_unrecorded_as_final():
    rows = [{"kind": "row", "arm": "fear", "seed": 11, "attempt": 1, "refusal": "x", "refusal_cause": "unrecorded"}]
    assert B.attempts_done(rows)[("fear", 11)] == {"attempts": 1, "done": True}


def test_orphan_starts_and_their_closing_rows():
    rows = [
        {"kind": "session", "campaign_id": "c1"},
        _start("fear", 11),
        _row("fear", 11, True),
        _start("random", 11),  # the process died inside this attempt
        _start("random", 11),  # a duplicate start is closed once
    ]
    orphans = B.orphan_starts(rows)
    assert [B._attempt_key(r) for r in orphans] == [("random", 11, 1)]

    def base(arm, seed, attempt):
        return {"kind": "row", "arm": arm, "seed": seed, "attempt": attempt, "provenance": {"p": 1}, "refusal": None}

    closing = B.closing_rows(rows, base)
    assert len(closing) == 1
    c = closing[0]
    assert (c["arm"], c["seed"], c["attempt"], c["campaign_id"], c["run_id"]) == ("random", 11, 1, "c1", "run-random")
    assert c["refusal_cause"] == "unrecorded" and c["refusal_step"] == "unrecorded"
    assert c["refusal"] == B.UNRECORDED_REFUSAL and c["provenance"] == {"p": 1}
    assert isinstance(c["ts"], float)
    assert B.attempts_done(rows + closing)[("random", 11)] == {"attempts": 1, "done": True}
    assert B.orphan_starts(rows + closing) == []


# ── _run itself: the write order and the resume close (the live path, rig and agent stubbed) ─────────


def _stub_run(monkeypatch, tmp_path, *, tree_refusal=None, real_append_refusal=False):
    import json as _json

    gate = tmp_path / "gate.json"
    gate.write_text(_json.dumps({"run_gate": {"pass": True}}))
    anchor = tmp_path / "anchor.json"
    anchor.write_text(_json.dumps({"measured": {"t_damage_onset_min_s": 20.0}}))
    monkeypatch.delenv("MAXIM_SUBSTRATE_TOOL_WHITELIST", raising=False)
    monkeypatch.setattr(B, "evidence_out_paths_or_exit", lambda root, paths, **kw: [Path(paths[0])])
    monkeypatch.setattr(B, "in_process_code_provenance", lambda *a, **k: {"code_tree_sha256": "t"})
    if not real_append_refusal:
        monkeypatch.setattr(B, "append_refusal", lambda out, prov: tree_refusal)
    monkeypatch.setattr(
        B,
        "load_apparatus_record",
        lambda p: {"record": {"ts": 1.0}, "path": "a.json", "sha256": "abc", "abs_path": p, "bytes": b""},
    )
    monkeypatch.setattr(B, "is_gated_path", lambda root, p: False)
    monkeypatch.setattr(B, "instrument_check_authorizes", lambda a: None)
    monkeypatch.setattr(B, "min_pain_edge_s", lambda a: 10.0)
    monkeypatch.setattr(B, "ANCHOR_FILE", anchor)
    monkeypatch.setattr(B, "anchor_record_mismatch", lambda m, a: None)
    monkeypatch.setattr(B, "apparatus_geometry", lambda g: {})
    monkeypatch.setattr(B, "wait_for_quiet_rig", lambda rcon, url, **kw: {"proxy": "not_checked"})

    class _Rcon:
        def __init__(self, *a):
            pass

        def close(self):
            pass

    monkeypatch.setattr(B.C, "RconControl", _Rcon)
    calls = []

    def fake_run_arm_seed(ses, arm, seed, attempt):
        last = _json.loads(Path(ses.out_path_for_test).read_text().splitlines()[-1])
        calls.append(((arm, seed, attempt), last))
        return B.row_base(ses, arm, seed, attempt)

    monkeypatch.setattr(B, "run_arm_seed", fake_run_arm_seed)
    return gate, calls


def _run_args(gate, out, **kw):
    import argparse

    base = dict(
        arms="fear,random", diagnostic=True, resume=None, out=str(out), write_experiment_results=False,
        allow_dirty=False, apparatus_record="a.json", gate_record=str(gate), proxy_url="", rcon_host="h",
        rcon_port=1, rcon_password="p", username="maxim",
    )  # fmt: skip
    base.update(kw)
    return argparse.Namespace(**base)


def _patch_session_out(monkeypatch, out):
    orig = B.Session.__init__

    def init(self, args, **kw):
        orig(self, args, **kw)
        self.out_path_for_test = out

    monkeypatch.setattr(B.Session, "__init__", init)


def _lines(path):
    import json as _json

    return [_json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_run_writes_an_attempt_start_before_every_attempt(monkeypatch, tmp_path):
    gate, calls = _stub_run(monkeypatch, tmp_path)
    out = tmp_path / "b1.jsonl"
    _patch_session_out(monkeypatch, out)
    assert B._run(_run_args(gate, out)) == 0
    # at each attempt's start, the newest line on disk is that attempt's own attempt_start header
    for key, last in calls:
        assert last["kind"] == "attempt_start" and B._attempt_key(last) == key
        assert last["record_kind"] == "harness_header" and last["mock"] is False
    kinds = [(r["kind"], r.get("arm")) for r in _lines(out)]
    assert kinds == [
        ("session", None),
        ("attempt_start", "fear"),
        ("row", "fear"),
        ("attempt_start", "random"),
        ("row", "random"),
    ]
    starts = [r for r in _lines(out) if r["kind"] == "attempt_start"]
    session = _lines(out)[0]
    assert all(
        s["run_id"] == session["run_ids"][s["arm"]] and s["campaign_id"] == session["campaign_id"] for s in starts
    )


def test_resume_closes_an_orphan_start_and_never_reruns_it(monkeypatch, tmp_path):
    import json as _json

    gate, calls = _stub_run(monkeypatch, tmp_path)
    out = tmp_path / "b1.jsonl"
    _patch_session_out(monkeypatch, out)
    session = {"kind": "session", "campaign_id": "diag-c", "run_ids": {a: f"r-{a}" for a in B.ARMS}, "diagnostic": True}
    orphan = {**_start("fear", 99, campaign="diag-c"), "run_id": "r-fear", "ts": 1.0, "record_kind": "harness_header"}
    out.write_text(_json.dumps(session) + "\n" + _json.dumps(orphan) + "\n")
    assert B._run(_run_args(gate, out, resume=str(out))) == 4  # a refusal row was written
    assert [k for k, _ in calls] == [("random", 99, 1)]  # fear is consumed, not rerun
    lines = _lines(out)
    closing = lines[2]
    assert closing["kind"] == "row" and B._attempt_key(closing) == ("fear", 99, 1)
    assert closing["refusal_cause"] == "unrecorded" and closing["record_kind"] == "harness_row"
    assert closing["run_id"] == "r-fear" and closing["campaign_id"] == "diag-c"
    assert closing["provenance"] == {"code_tree_sha256": "t"} and closing["apparatus_record_sha256"] == "abc"
    assert B.orphan_starts(lines) == []


def test_a_harness_written_campaign_crashed_mid_attempt_resumes_through_the_real_append_refusal(monkeypatch, tmp_path):
    """The PR-C review's blocker 2: the starts the HARNESS writes must carry its provenance, or the real
    `append_refusal` refuses every resume ("another code tree (None)") and D2's closing path is dead. Here the harness
    writes its own campaign, dies inside the second attempt, and is resumed with the real `append_refusal`."""
    gate, calls = _stub_run(monkeypatch, tmp_path, real_append_refusal=True)
    out = tmp_path / "b1.jsonl"
    _patch_session_out(monkeypatch, out)
    real = B.run_arm_seed

    def dies_in_random(ses, arm, seed, attempt):
        if arm == "random":
            raise KeyboardInterrupt  # the process ends mid-attempt: its start is on disk, its row never is
        return real(ses, arm, seed, attempt)

    monkeypatch.setattr(B, "run_arm_seed", dies_in_random)
    with pytest.raises(KeyboardInterrupt):
        B._run(_run_args(gate, out))
    assert [r["kind"] for r in _lines(out)] == ["session", "attempt_start", "row", "attempt_start"]
    assert all(r["provenance"] == {"code_tree_sha256": "t"} for r in _lines(out))
    monkeypatch.setattr(B, "run_arm_seed", real)
    assert B._run(_run_args(gate, out, resume=str(out))) == 4  # resumed (not refused) and the orphan closed
    lines = _lines(out)
    assert lines[-1]["kind"] == "row" and lines[-1]["refusal_cause"] == "unrecorded"
    assert B._attempt_key(lines[-1]) == ("random", 99, 1) and B.orphan_starts(lines) == []


def test_a_resume_on_a_changed_tree_is_refused_before_closing_anything(monkeypatch, tmp_path):
    import json as _json

    gate, calls = _stub_run(monkeypatch, tmp_path, tree_refusal="another code tree")
    out = tmp_path / "b1.jsonl"
    session = {"kind": "session", "campaign_id": "diag-c", "run_ids": {a: f"r-{a}" for a in B.ARMS}, "diagnostic": True}
    body = _json.dumps(session) + "\n" + _json.dumps(_start("fear", 99, campaign="diag-c")) + "\n"
    out.write_text(body)
    assert B._run(_run_args(gate, out, resume=str(out))) == 2
    assert out.read_text() == body and calls == []


@pytest.mark.parametrize(
    "mutate,cause",
    [
        (lambda rows: rows + [_row("fear", 99, True)], "foreign_seed_rows"),
        (lambda rows: rows + [{**_row("fear", 11, True), "campaign_id": "c2", "attempt": 2}], "mixed_campaigns"),
        (lambda rows: rows[:-1] + [{**rows[-1], "apparatus_record_sha256": "other"}], "mixed_apparatus_records"),
        (lambda rows: rows[:-1] + [{**rows[-1], "run_id": "run-fear"}], "run_ids_not_one_per_arm"),
        (lambda rows: rows + [{**_row("fear", 11, True), "attempt": 3}], "attempt_numbering"),
    ],
)
def test_structural_refusals(mutate, cause):
    v = B.compute_verdict(mutate(_campaign({"fear": 8, "random": 0, "q_explore": 0})))
    assert v["verdict"] == "INCOMPLETE" and v["incomplete_cause"] == cause


def test_an_unclean_first_post_in_a_clean_row_refuses():
    rows = _campaign({"fear": 8, "random": 0, "q_explore": 0})
    assert rows[2]["kind"] == "row"
    rows[2]["post"]["placements"][0] = {**rows[2]["post"]["placements"][0], "dirty": True}
    assert B.compute_verdict(rows)["incomplete_cause"] == "unclean_first_post_in_clean_row"


def test_training_summary_and_key_coverage():
    eps = [
        {"index": 1, "reason": "pain", "calls": [1, 2, 3], "greedy_in_water": {"best": ["a"]}},
        {"index": 2, "reason": "surfaced", "calls": [1], "greedy_in_water": {"best": ["e"]}},
        {"index": 3, "reason": "surfaced", "calls": [1], "greedy_in_water": {"best": ["e"]}},
    ]
    row = {"training": {"episodes": eps}, "pre": {"placements": [{"calls": [1, 1]}], "shore_roam": {"calls": [1]}}}
    s = B.training_summary(row)
    assert s["first_surfaced_episode"] == 2 and s["executed_actions_to_criterion"] == 3 + 4
    assert s["greedy_in_water_after"] == {"1": {"best": ["a"]}, "2": {"best": ["e"]}}
    k = ["clusters", [["world", "w"]]]
    picks = [
        {"key": k, "action": "mc_escape_water", "phase": "qtrain-1", "in_water": True},
        {"key": k, "action": "mc_flee", "phase": "post-placement-0", "in_water": True},
        {"key": ["clusters", [["world", "x"]]], "action": "mc_flee", "phase": "post-placement-0", "in_water": True},
    ]
    cov = B.key_coverage({"picks": picks})
    assert cov == {
        "first_post_selects": 2,
        "share_visited_with_escape_tried": 0.5,
        "first_select_key_tried": True,
        "distinct_water_keys": 2,
    }
    picks.append({"key": ["clusters", [["world", "x"]]], "action": "mc_escape_water", "phase": "pre-placement-3"})
    assert B.key_coverage({"picks": picks})["share_visited_with_escape_tried"] == 1.0  # the pre-probe counts
    assert B.training_summary({"training": None}) is None and B.key_coverage(None) is None


# ── the copied Exp 60 preflight sequence ────────────────────────────────────────────────────────────


def _trial_calls(fn) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "trial"
        ):
            calls.append((node.lineno, node.col_offset, node.func.attr))
    return [name for *_pos, name in sorted(calls)]


def test_the_preflight_sequence_is_exp60s_in_its_order():
    from survival_world import exp60_run

    exp60 = _trial_calls(exp60_run._run)
    start, end = exp60.index("check_fingerprint"), exp60.index("deaths")
    want = exp60[start : end + 1]
    b1 = _trial_calls(B.run_arm_seed)
    i = b1.index("check_fingerprint")
    assert b1[i : i + len(want)] == want, (b1, want)


# ── params={} never reaches the bridge for an affordance that needs params ───────────────────────────


@pytest.mark.timeout(120)
def test_affordances_with_required_params_refuse_empty_params_before_execute(tmp_path):
    from maxim.simulation.minecraft_harness import FakeBridgeServer, build_minecraft_aut

    srv = FakeBridgeServer(state_interval_s=0.1)
    aut = build_minecraft_aut(
        agent_id="b1_params", bridge_port=srv.port, persistence_dir=str(tmp_path), entity_ref="bodies/minecraft_player"
    )
    try:
        needs = []
        for name in aut.executor.registry.list():
            tool = aut.executor.registry.get(name)
            try:
                tool._validate_input({})
            except ValueError:
                needs.append(name)
                reached = []
                orig = tool.execute
                tool.execute = lambda *a, **k: reached.append(1)  # type: ignore[method-assign]
                try:
                    result = aut.executor.execute({"tool_name": name, "params": {}})
                finally:
                    tool.execute = orig  # type: ignore[method-assign]
                assert not reached, f"{name} reached execute with params={{}}"
                assert getattr(result, "success", None) is False, (name, result)
        suffixes = {n.rsplit("minecraft_player_", 1)[-1] for n in needs}
        assert {"move_to", "turn", "mine_block", "place_block"} <= suffixes, sorted(suffixes)
    finally:
        aut.client.close()
        srv.close()


# ── the apparatus re-verify ─────────────────────────────────────────────────────────────────────────


def test_apparatus_geometry_and_reverify():
    from survival_world.scripted_water import ScriptedWaterBridge, ScriptedWaterControl
    from survival_world.setup_world import water_anchor_record, water_classroom_geometry

    geom = water_classroom_geometry(101, 50)
    anchor = water_anchor_record(geom)
    build = B.apparatus_geometry(anchor)
    assert list(build["pool"]) == list(anchor["pool"])
    with pytest.raises(InstrumentError):
        B.apparatus_geometry({**anchor, "submerged": [0, 0, 0]})

    srv = ScriptedWaterBridge(shore={"x": 0.0, "y": 64.0, "z": 0.0}, submerged={"x": 0.0, "y": 60.0, "z": 5.0})
    rcon = ScriptedWaterControl(srv)

    class _Trial:
        def check_surface_cell_air(self):
            return None

    try:
        assert B.reverify_apparatus(rcon, build, _Trial()) == 10
        from survival_world.setup_world import water_classroom_verifications

        rcon.failing_blocks.add(water_classroom_verifications(dict(build))[0][0])
        with pytest.raises(InstrumentError):
            B.reverify_apparatus(rcon, build, _Trial())
    finally:
        srv.close()


# ── the trial on the scripted water bridge ──────────────────────────────────────────────────────────

SHORE = {"x": 10.0, "y": 64.0, "z": 10.0}
SUBMERGED = {"x": 10.0, "y": 60.0, "z": 20.0}
GEOM = {"shore": [10.0, 64.0, 10.0], "submerged": [10.0, 60.0, 20.0], "deaths_objective": "exp60_deaths"}
FAST = {
    **B.FROZEN,
    "loop_warm_s": 0.5,
    "placements_per_probe": 1,
    "shore_roam_s": 1.0,
    "K_executed_episodes": 3,
    "fingerprint": {},
}


def _trial(tmp_path, arm, seed=99, drain=3.0):
    from maxim.simulation.minecraft_harness import MinecraftSyncPump, build_minecraft_aut
    from survival_world.common import make_fresh_encoder
    from survival_world.scripted_water import ScriptedWaterBridge, ScriptedWaterControl

    srv = ScriptedWaterBridge(shore=SHORE, submerged=SUBMERGED, oxygen_drain_per_s=drain, escape_blocks=True)
    rcon = ScriptedWaterControl(srv)
    agent_id = f"b1_smoke_{arm}"
    home = tmp_path / agent_id
    aut = build_minecraft_aut(
        agent_id=agent_id, bridge_port=srv.port, persistence_dir=str(home), entity_ref="bodies/minecraft_player"
    )
    aut.bio.memory_hub.on_session_start(owner=aut)  # the harness owns it, as the scripts do (#1438 PR 0)
    pump = MinecraftSyncPump(aut, interval_s=0.25)
    pump.start()
    trial = B.FloorWaterTrial(
        arm=arm,
        aut=aut,
        rcon=rcon,
        username="maxim",
        geom=GEOM,
        frozen=FAST,
        probe_cap_s=3.0,
        train_cap_s=5.0,
        persistence_dir=home,
        agent_id=agent_id,
        encoder=make_fresh_encoder(aut),
    )
    trial.attach_instruments()
    inner = B.make_policy(arm, seed, trial)
    if inner is not None:
        trial.set_policy(inner)

    def close():
        trial.detach_instruments()
        pump.stop()
        aut.client.close()
        srv.close()

    return trial, rcon, close


@pytest.mark.timeout(300)
def test_the_prior_learner_learns_to_escape_through_the_real_loop(tmp_path):
    trial, rcon, close = _trial(tmp_path, "q_explore_prior")
    events: list[tuple[str, float]] = []
    orig_tp, inner = rcon.teleport, trial.armed.inner
    orig_end = inner.end_episode

    def tp(name, pos):
        events.append(("tp", time.monotonic()))
        return orig_tp(name, pos)

    def end(reason, **kw):
        events.append(("end", time.monotonic()))
        return orig_end(reason, **kw)

    rcon.teleport, inner.end_episode = tp, end
    try:
        trial.resolve_tools()
        trial.deaths0 = trial.deaths()
        trial.probe("pre")
        training = trial.train_executed()
        post = trial.probe("post")
    finally:
        close()
    reasons = [e["reason"] for e in training["episodes"]]
    assert "surfaced" in reasons, reasons
    assert inner.actions == ("minecraft_player_escape_water", "minecraft_player_flee")
    assert training["episodes"][-1]["greedy_in_water"]["best"] == ["minecraft_player_escape_water"], training
    assert post["placements"][0]["surfaced"], post["placements"][0]
    # the surfacing is credited to the escape that caused it (rule 16: the harness's end_episode closes it)
    terminal = [u for u in inner.updates if u["terminal"]]
    assert terminal and all(u["action"].endswith("_escape_water") for u in terminal), terminal
    # every credited update is for an action the loop started executing (rule 5)
    started = {s["tool"] for s in trial.started}
    assert inner.updates and all(u["action"] in started for u in inner.updates)
    # the frozen order: every end_episode comes before the rescue teleport that follows it
    for i, (kind, _t) in enumerate(events):
        if kind == "end":
            assert events[i + 1][0] == "tp", events[i : i + 2]
    # a policy armed only for windows: nothing it picked has a phase outside one
    assert all(p["phase"] for p in inner.picks)
    rec = B.policy_record(trial.armed, "q_explore_prior", 99, 0.0)
    assert rec["class"] == "TabularQPolicy" and rec["violations"] == [] and rec["tool_sets"]


@pytest.mark.timeout(180)
def test_random_acts_across_the_repertoire_and_fear_runs_without_a_policy(tmp_path):
    trial, _rcon, close = _trial(tmp_path, "random")
    try:
        trial.resolve_tools()
        trial.deaths0 = trial.deaths()
        out = trial.probe("pre")
    finally:
        close()
    picked = {p["action"] for p in trial.armed.inner.picks}
    assert len(picked) > 1 and any(not a.endswith("escape_water") for a in picked), picked
    assert out["placements"][0]["end_reason"] in FP.TERMINAL_REASONS
    assert out["shore_roam"]["calls"] is not None

    trial, _rcon, close = _trial(tmp_path, "fear")
    try:
        assert trial.armed is None and trial.substrate_policy is None
        trial.resolve_tools()
        trial.deaths0 = trial.deaths()
        out = trial.probe("pre")
    finally:
        close()
    assert out["placements"][0]["calls"] == []  # no fear: the structural floor, as in Exp 60


@pytest.mark.timeout(300)
@pytest.mark.parametrize("arm", ["random", "q_explore_prior"])
def test_run_arm_seed_end_to_end_writes_a_serialisable_row_and_tears_down(tmp_path, monkeypatch, arm):
    """The live per-arm-seed path itself (not an AST of it) on the scripted bridge, with the fingerprint, rig and
    pool re-verify stubbed (no live server). A crash in the row's derived fields used to skip the teardown."""
    import argparse
    import json
    import os

    from survival_world.scripted_water import ScriptedWaterBridge, ScriptedWaterControl

    for k, v in {"loop_warm_s": 0.5, "placements_per_probe": 1, "shore_roam_s": 1.0, "K_executed_episodes": 2}.items():
        monkeypatch.setitem(B.FROZEN, k, v)
    monkeypatch.setitem(B.FROZEN, "fingerprint", {})
    monkeypatch.setattr(B, "reverify_apparatus", lambda rcon, geom, trial: 10)
    monkeypatch.setattr(B.FloorWaterTrial, "check_fingerprint", lambda self, m: {})
    monkeypatch.setattr(B, "rig_check", lambda rcon, url: {"proxy": "not_checked"})
    srv = ScriptedWaterBridge(shore=SHORE, submerged=SUBMERGED, escape_blocks=True)
    rcon = ScriptedWaterControl(srv)
    args = argparse.Namespace(bridge_host="127.0.0.1", bridge_port=srv.port, username="maxim", proxy_url="")
    ses = B.Session(
        args, campaign_id="c", run_ids={a: a for a in B.ARMS}, provenance={}, apparatus={}, apparatus_path="p",
        apparatus_sha256="s", apparatus_check={}, geom={**GEOM, "surface_y": 61}, build_geom={}, probe_cap_s=3.0,
        train_caps={a: 5.0 for a in B.ARMS}, rcon=rcon,
    )  # fmt: skip
    cwd = os.getcwd()
    try:
        rec = B.run_arm_seed(ses, arm, 99, 1)
    finally:
        srv.close()
    assert os.getcwd() == cwd
    json.dumps(rec)  # strict: the writer no longer stringifies a non-JSON value
    assert rec["refusal"] is None, (rec["refusal_step"], rec["refusal"])
    assert rec["policy"]["class"] == ("RandomPolicy" if arm == "random" else "TabularQPolicy")
    assert (rec["key_coverage"] is None) == (arm == "random")
    assert rec["post"]["placements"][0]["t_first_water"] is not None


@pytest.mark.timeout(120)
def test_a_raising_before_rescue_never_skips_the_rescue(tmp_path):
    """The shared WaterTrial.loop_window: a callback that raises still lets the rescue teleport and the loop stop run
    (the bot is never left submerged with its loop alive)."""
    trial, rcon, close = _trial(tmp_path, "random")

    def boom(samples, t0):
        raise RuntimeError("callback failed")

    try:
        trial.resolve_tools()
        n = len(rcon.commands)
        with pytest.raises(RuntimeError, match="callback failed"):
            trial.loop_window(1.0, enter=lambda: trial.submerge("x"), on_sample=lambda s: False, label="x",
                              before_rescue=boom)  # fmt: skip
        tps = [c for c in rcon.commands[n:] if c.startswith("tp ")]
        assert tps and tps[-1] == f"tp maxim {SHORE['x']} {SHORE['y']} {SHORE['z']}", tps
    finally:
        close()


def test_was_executed_reads_the_start_spy_and_stops_at_since():
    trial = B.FloorWaterTrial.__new__(B.FloorWaterTrial)
    trial.started = [{"t": 1.0, "tool": "a"}, {"t": 2.0, "tool": "b"}, {"t": 3.0, "tool": "c"}]
    assert trial.was_executed("b", 1.5) and trial.was_executed("c", 3.0)
    assert not trial.was_executed("a", 1.5)  # started before the pick: not this pick's execution
    assert not trial.was_executed("z", 0.0)


def test_a_missing_observation_key_is_a_recorded_violation():
    armed = FP.ArmedPolicy(FP.RandomPolicy(1))
    armed.arm()
    armed.select(agent_id="a", available_tools=("t",), clusters=None, observation={"is_in_water": 1.0})
    assert armed.all_violations() == ["observation lacks ['oxygen']"]


@pytest.mark.timeout(120)
def test_liveness_runs_the_policy_disarmed(tmp_path):
    trial, _rcon, close = _trial(tmp_path, "random")
    try:
        assert trial.check_liveness() >= 1
    finally:
        close()
    assert trial.armed.tool_sets, "liveness never asked the (disarmed) policy: the forwarding is gone"
    assert trial.armed.inner.picks == []


@pytest.mark.timeout(120)
def test_a_policy_that_kills_the_loop_refuses_the_window(tmp_path):
    trial, _rcon, close = _trial(tmp_path, "fear")

    class Bad:  # unwrapped: the hook's validation raises SubstratePolicyError and the loop thread dies
        def select(self, **kw):
            return {"tool_name": "no_such_tool", "params": {}}

    trial.substrate_policy = Bad()
    try:
        trial.resolve_tools()
        trial.deaths0 = trial.deaths()
        with pytest.raises(B.Refusal, match="exited before"):
            trial.placement("pre-placement-0")
    finally:
        close()


@pytest.mark.parametrize(("sub", "flag"), [("run", "--out"), ("verdict", "--data")])
def test_the_real_command_line_parser_builds(sub, flag, capsys):
    """The CLI itself, as the rig invokes it: building the parser crashed every invocation (a help string called
    `campaign_out_path` with a placeholder id it rejects) and no test ever constructed the real parser."""
    with pytest.raises(SystemExit) as exc:
        B.main([sub, "--help"])
    assert exc.value.code == 0
    assert flag in capsys.readouterr().out


def test_rig_problems_say_whether_waiting_can_clear_them():
    now = 10_000.0
    idle = {"infer_lane": {"in_flight": 0}}
    assert [p.transient for p in B.rig_problems({"infer_lane": {"in_flight": 1}}, {"recent": []}, 20.0, now=now)] == [
        True
    ]
    assert [p.transient for p in B.rig_problems(idle, {"recent": [{"timestamp": now - 30}]}, 20.0, now=now)] == [True]
    assert [p.transient for p in B.rig_problems(idle, {"recent": []}, 19.0, now=now)] == [True]
    assert [p.transient for p in B.rig_problems({"infer_lane": None}, {"recent": []}, 20.0, now=now)] == [False]
    assert [p.transient for p in B.rig_problems(idle, {"recent": None}, 20.0, now=now)] == [False]
    assert [p.transient for p in B.rig_problems(None, None, None, now=now)] == [False]


class _Clock:
    """A fake monotonic clock that ``sleep`` advances, so a polling regression fails in milliseconds."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_a_refusal_that_cannot_clear_stands_at_once_and_a_transient_one_is_waited_out(monkeypatch):
    calls = []

    def stuck(rcon, url):
        calls.append(1)
        raise B.RigNotQuiet("no lane metrics", transient=False)

    clock = _Clock()
    monkeypatch.setattr(B.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(B.time, "sleep", clock.sleep)
    monkeypatch.setattr(B, "rig_check", stuck)
    with pytest.raises(B.RigNotQuiet):
        B.wait_for_quiet_rig(None, "http://x", max_wait_s=600, poll_s=15)
    assert calls == [1]  # no polling

    seq = [B.RigNotQuiet("busy", transient=True), {"proxy": "present"}]

    def busy_then_idle(rcon, url):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(B, "rig_check", busy_then_idle)
    rec = B.wait_for_quiet_rig(None, "http://x", max_wait_s=600, poll_s=15)
    assert rec["proxy"] == "present" and rec["waited_s"] == 15.0


@pytest.mark.parametrize(
    ("exc_name", "transient"),
    [("HTTPTimeout", True), ("HTTPServerError", True), ("HTTPRateLimited", True), ("HTTPAuthError", False)],
)
def test_a_busy_proxy_is_transient_and_a_refusing_one_is_not(monkeypatch, exc_name, transient):
    from maxim.utils import http

    exc_cls = getattr(http, exc_name)

    def boom(*a, **k):
        raise exc_cls.__new__(exc_cls)

    monkeypatch.setattr(http, "fetch_url", boom)

    class _R:
        def command(self, cmd):
            return "Target tick rate: 20.0 per second.\nAverage time per tick: 3.0ms"

    with pytest.raises(B.RigNotQuiet) as info:
        B.rig_check(_R(), "http://127.0.0.1:8099")
    assert info.value.transient is transient


def test_a_rig_that_cannot_clear_refuses_the_campaign_before_any_attempt(monkeypatch, tmp_path):
    gate, calls = _stub_run(monkeypatch, tmp_path)
    out = tmp_path / "b1.jsonl"
    _patch_session_out(monkeypatch, out)

    def stuck(rcon, url, **kw):
        raise B.RigNotQuiet("no lane metrics", transient=False)

    monkeypatch.setattr(B, "wait_for_quiet_rig", stuck)
    assert B._run(_run_args(gate, out)) == 3
    assert calls == [] and (not out.exists() or out.read_text() == "")


def test_a_non_transient_rig_refusal_mid_campaign_stops_without_spending_the_rerun(monkeypatch, tmp_path):
    gate, calls = _stub_run(monkeypatch, tmp_path)
    out = tmp_path / "b1.jsonl"
    _patch_session_out(monkeypatch, out)

    def rig_goes_bad(ses, arm, seed, attempt):
        calls.append(((arm, seed, attempt), None))
        return {**B.row_base(ses, arm, seed, attempt), "refusal": "rig", "refusal_cause": "instrument",
                "refusal_step": "rig", "rig_refusal_transient": False}  # fmt: skip

    monkeypatch.setattr(B, "run_arm_seed", rig_goes_bad)
    assert B._run(_run_args(gate, out)) == 3
    assert [k for k, _ in calls] == [("fear", 99, 1)]  # stopped: no rerun, no next arm

    # the rig is fixed: --resume gets that key's rerun back (its refusal is an instrument cause on attempt 1)
    def ok(ses, arm, seed, attempt):
        calls.append(((arm, seed, attempt), None))
        return B.row_base(ses, arm, seed, attempt)

    monkeypatch.setattr(B, "run_arm_seed", ok)
    assert B._run(_run_args(gate, out, resume=str(out))) == 0
    assert [k for k, _ in calls][1:] == [("fear", 99, 2), ("random", 99, 1)]


def test_the_apparatus_check_must_be_from_the_campaigns_utc_day():
    import datetime as dt

    noon = dt.datetime(2026, 10, 14, 12, tzinfo=dt.timezone.utc).timestamp()
    assert B.apparatus_day_problem(noon - 3600, noon) is None
    assert "not today" in B.apparatus_day_problem(noon - 86400, noon)
    late = dt.datetime(2026, 10, 14, 23, 50, tzinfo=dt.timezone.utc).timestamp()
    assert "not today" in B.apparatus_day_problem(late, late + 1200)  # across UTC midnight: a fresh check
    assert "no epoch ts" in B.apparatus_day_problem(None, noon)
    assert "no epoch ts" in B.apparatus_day_problem(True, noon)


@pytest.mark.parametrize(
    ("kw", "want"),
    [({"diagnostic": False}, 3), ({"diagnostic": True}, 0)],
)
def test_a_gated_campaign_refuses_a_water_check_from_another_utc_day(monkeypatch, tmp_path, kw, want):
    """The same-UTC-day rule at its real call site in _run (not only the pure helper)."""
    gate, calls = _stub_run(monkeypatch, tmp_path)
    out = tmp_path / "b1.jsonl"
    _patch_session_out(monkeypatch, out)
    monkeypatch.setattr(B, "is_gated_path", lambda root, p: True)
    monkeypatch.setattr(B, "committed_record_problem", lambda *a: None)
    stale = {"record": {"ts": 1.0}, "path": "a.json", "sha256": "abc", "abs_path": "a.json", "bytes": b""}
    monkeypatch.setattr(B, "load_apparatus_record", lambda p: stale)  # ts 1.0 = 1970: never today
    assert B._run(_run_args(gate, out, **kw)) == want
    if want == 3:
        assert calls == [] and (not out.exists() or out.read_text() == "")


def test_the_rig_check_sends_this_boxs_key_to_a_loopback_proxy_only(monkeypatch):
    """The proxy's debug routes need the key, loopback included (2026-10-14); a remote URL never gets it."""
    from maxim.utils import http

    sent: list[dict] = []

    def _fetch(url, **kw):
        sent.append(kw.get("headers") or {})
        raise http.HTTPConnectionError("x")

    monkeypatch.setattr(http, "fetch_url", _fetch)
    monkeypatch.setattr("maxim.tunnel.keys.read_key", lambda *a, **k: "k" * 43)

    class _R:
        def command(self, cmd):
            return "Target tick rate: 20.0 per second.\nAverage time per tick: 3.0ms"

    B.rig_check(_R(), "http://127.0.0.1:8099")
    assert sent[0] == {"Authorization": "Bearer " + "k" * 43}
    sent.clear()
    B.rig_check(_R(), "http://leader.example:8099")
    assert sent[0] == {}
