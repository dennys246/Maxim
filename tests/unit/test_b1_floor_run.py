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
    assert B.rig_problems(idle, {"recent": [{"ts": now - 500}]}, 20.0, now=now) == []
    assert B.rig_problems({"infer_lane": {"in_flight": 1}}, {"recent": []}, 20.0, now=now)
    assert B.rig_problems({"infer_lane": {}}, {"recent": []}, 20.0, now=now)  # idle cannot be shown
    assert B.rig_problems(idle, {"recent": [{"ts": now - 30}]}, 20.0, now=now)
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


def _campaign(rates):
    rows = [{"kind": "session", "campaign_id": "c1"}]
    for arm, hits in rates.items():
        rows += [_row(arm, s, i < hits) for i, s in enumerate(B.SEEDS)]
    return rows


def test_fear_beating_both_floors_is_floors_beaten():
    v = B.compute_verdict(_campaign({"fear": 8, "random": 0, "q_explore": 0, "q_explore_prior": 8}))
    assert v["verdict"] == "FLOORS_BEATEN" and v["holm_shown"] == {"H1": True, "H2": True}
    assert v["first_post_rate"]["q_explore_prior"] == 1.0 and v["n_clean"]["q_explore_raw"] == 0


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
    rows = [r for r in rows if not (r.get("arm") == "random" and r.get("seed") == 18)]
    v = B.compute_verdict(rows)
    assert v["verdict"] == "INCOMPLETE" and v["incomplete_cause"] == "too_few_clean_seeds"


def test_an_instrument_rerun_counts_and_a_behavioural_rerun_refuses():
    rows = _campaign({"fear": 8, "random": 0, "q_explore": 0})
    first = _row("random", 11, False, refusal="rig", cause="instrument")
    rerun = {**_row("random", 11, False), "attempt": 2}
    rows = [r for r in rows if not (r.get("arm") == "random" and r.get("seed") == 11)] + [first, rerun]
    v = B.compute_verdict(rows)
    assert v["verdict"] == "FLOORS_BEATEN" and v["attempts"]["random/11"] == 2
    rows[-2] = {**first, "refusal_cause": "behavioural"}
    assert B.compute_verdict(rows)["incomplete_cause"] == "rerun_after_behavioural_refusal"
    rows[-2] = {**first, "refusal": None}
    assert B.compute_verdict(rows)["incomplete_cause"] == "rerun_after_clean_row"


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
    rows[1]["post"]["placements"][0] = {**rows[1]["post"]["placements"][0], "dirty": True}
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
    assert cov == {"first_post_selects": 2, "share_visited_with_escape_tried": 0.5, "distinct_water_keys": 2}
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
    aut.bio.memory_hub.on_session_start()
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
