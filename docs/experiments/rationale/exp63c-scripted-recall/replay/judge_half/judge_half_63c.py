"""Exp 63c fidelity check, judge half (BINDING run on the harness head; prereg §Power replay, "Fidelity check").

Every simcore.r3d() call the frozen replay makes is ALSO judged by the shipped 63c judge port, taken from the harness
commit: the 63c layer's per-turn judge (exp63b_turn under c_key_tokens, explain=True, every valid tie ordering), and on
it the 63c rules: R3d searched on the cue turns (3, 4, 5, 7) only; control status by the carried-store rule
(exp63c_control_status); C5(k) (an armed control decisive) and C5(q) (a decisive order whose winner shares no cue
non-goal key); and the per-cue-turn report (_exp63c_report: contested, uncontested_literal, completion, winner is the
planted fact). Records are the replay's own (to_dict, capture_seq = pool index, horizon = last, no holes); the trace
logs the simulator's (ids, paths), which the judge-free half showed equal the shipped recall's 768/768.

Compared per turn: port decisive vs simulator decisive. Per policy: k, contested wins, completion, planted winners,
control status vs the replay's armed flag. Also checked: the replay's script lines equal PROTOCOL["63c"]'s literals.

python judge_half_63c.py <o19_verdict.py at the harness head> <script_module> <intents_module> [policies...]
"""

import importlib
import importlib.util
import json
import runpy
import sys

spec = importlib.util.spec_from_file_location("o19v_port", sys.argv[1])
v = importlib.util.module_from_spec(spec)
sys.modules["o19v_port"] = v
spec.loader.exec_module(v)
key = v.keying_of("63c")
assert key is v.c_key_tokens, "keying_of('63c') must be the post-#1620 port"

S = importlib.import_module(sys.argv[2])
lines_p1 = tuple(line for line, _role, _a in S.P1)
lines_p2 = tuple(line for line, _role in S.P2)
LINES_OK = (lines_p1, lines_p2) == v.EXP63C_LINES
assert set(S.CUE_OF) == set(v.EXP63C_CUE_TURNS), (S.CUE_OF, v.EXP63C_CUE_TURNS)

import simcore  # noqa: E402

POLICIES = sys.argv[4:]
STAT = {"calls": 0, "nonconform": 0, "mismatch": 0, "over_bound": 0, "max_orderings": 0, "c5k": 0, "c5q": 0,
        "examples": []}
PER = {}  # policy -> per-turn results
_orig = simcore.r3d


def judge(pool, cue, carried_ids, own_ids, goal):
    records = {}
    for i, m in enumerate(pool):
        d = json.loads(json.dumps(m.to_dict(), default=str))
        d["capture_seq"] = i
        records[m.id] = d
    shown = simcore.simulate(pool, cue, goal, own_ids)
    trace = {"memory_ids": [i for i, _ in shown], "memory_paths": [p for _, p in shown],
             "own_ids": list(own_ids), "cue_text": cue, "goal_path_horizon": len(pool) - 1, "goal_path_holes": []}
    carried = set(carried_ids) & set(records)
    x = v.exp63b_turn(trace, records, carried, goal, key=key, explain=True)
    return x, trace, records, carried


def checked(pool, cue, carried_ids, own_ids=(), goal=simcore.GOAL):
    d = _orig(pool, cue, carried_ids, own_ids, goal)
    turn = STAT["calls"] % 8 + 1
    policy = POLICIES[STAT["calls"] // 8] if STAT["calls"] // 8 < len(POLICIES) else "?"
    STAT["calls"] += 1
    x, tr, records, carried = judge(pool, cue, carried_ids, own_ids, goal)
    STAT["max_orderings"] = max(STAT["max_orderings"], x.get("orderings", 0))
    STAT["over_bound"] += bool(x.get("over_bound"))
    STAT["nonconform"] += not x.get("conforms")
    port_dec = bool(x.get("decisive"))
    if port_dec != bool(d["decisive"]) or not x.get("conforms"):
        STAT["mismatch"] += port_dec != bool(d["decisive"])
        if len(STAT["examples"]) < 6:
            STAT["examples"].append({"policy": policy, "turn": turn, "sim": d["decisive"], "port": port_dec,
                                     "conforms": x.get("conforms")})
    base_records = {m: records[m] for m in carried}
    row = {"turn": turn, "sim": bool(d["decisive"]), "port": port_dec}
    if turn in v.EXP63C_CONTROL_TURNS:
        row["control"] = v.exp63c_control_status(turn, base_records, goal, key=key)
        if port_dec and row["control"] == "armed":
            STAT["c5k"] += 1
    if port_dec:
        nongoal = set(key(cue)) - set(key(goal))
        for o in x.get("orders", []):
            winners = {c for c, _d, sc, sd in o["pairs"] if sc > sd}
            if not any(nongoal & v.p_record_tokens(records[c], key=key) for c in winners):
                STAT["c5q"] += 1
                break
    if turn in v.EXP63C_CUE_TURNS:
        rep = v._exp63c_report(turn, x, tr, records, carried, base_records, goal, key=key)
        row.update({k: rep[k] for k in ("contested", "uncontested_literal", "completion", "winner_is_planted_fact",
                                        "score")})
    PER.setdefault(policy, []).append(row)
    return d


simcore.r3d = checked
sys.argv = ["replay.py", *sys.argv[2:]]
runpy.run_path("replay.py", run_name="__main__")

print(f"\nJUDGE HALF (63c layer): script lines == PROTOCOL['63c'] literals: {LINES_OK}")
print(f"r3d calls={STAT['calls']} non-conforming={STAT['nonconform']} decisive mismatches={STAT['mismatch']} "
      f"over_bound={STAT['over_bound']} max tie orderings={STAT['max_orderings']} C5(k)={STAT['c5k']} C5(q)={STAT['c5q']}")
tot_k = tot_contested = tot_dec = 0
for pol, rows in PER.items():
    cues = [r for r in rows if r["turn"] in v.EXP63C_CUE_TURNS]
    dec = [r for r in cues if r["port"]]
    tot_k += len(dec)
    tot_contested += sum(1 for r in dec if not r["uncontested_literal"])
    ctrls = {r["turn"]: r["control"] for r in rows if "control" in r}
    print(f"  {pol:7s} k={len(dec)}/4 turns={[r['turn'] for r in dec]} "
          f"contested(any order best)={[r['turn'] for r in dec if r['contested']]} "
          f"not-uncontested-literal={[r['turn'] for r in dec if not r['uncontested_literal']]} "
          f"completion={all(r['completion'] for r in dec) if dec else None} "
          f"planted-winner={all(r['winner_is_planted_fact'] for r in dec) if dec else None} controls={ctrls}")
print(f"TOTAL decisive cue turns={tot_k}; not uncontested-literal={tot_contested}")
for e in STAT["examples"]:
    print("  DIFF", json.dumps(e, default=str))
